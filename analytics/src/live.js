import { readStatistics } from './statistics.js';

const ONLINE_TIMEOUT = 60000;
const CHECK_INTERVAL = 15000;

// Socket attachments contain temporary page IDs and city keys, never IPs.
export class VisitorHub {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    this.pending = Promise.resolve();
    this.revision = 0;
    ctx.blockConcurrencyWhile(async () => {
      this.revision = await ctx.storage.get('revision') || 0;
    });
  }

  exclusive(action) {
    const result = this.pending.then(action);
    this.pending = result.catch(() => {});
    return result;
  }

  sockets() {
    return this.ctx.getWebSockets().filter(socket => socket.readyState === 1);
  }

  async snapshot() {
    const now = Date.now();
    const online = new Set();
    for (const socket of this.sockets()) {
      const state = socket.deserializeAttachment();
      if (state?.cityKey && now - state.lastSeen < ONLINE_TIMEOUT) online.add(state.cityKey);
    }
    const data = await readStatistics(this.env.DB, Math.floor(now / 1000), online);
    data.revision = ++this.revision;
    await this.ctx.storage.put('revision', this.revision);
    return data;
  }

  async publish() {
    const data = await this.snapshot();
    const message = JSON.stringify({ type: 'stats', data });
    for (const socket of this.sockets()) {
      try { socket.send(message); } catch { socket.close(1011, 'Connection lost'); }
    }
    return data;
  }

  async scheduleCheck() {
    if (this.sockets().length && await this.ctx.storage.getAlarm() === null) {
      await this.ctx.storage.setAlarm(Date.now() + CHECK_INTERVAL);
    }
  }

  fetch(request) {
    return this.exclusive(async () => {
      const url = new URL(request.url);
      if (url.pathname === '/live') {
        if (this.sockets().length >= 1000) return new Response('Live map is busy', { status: 503 });
        const eventId = url.searchParams.get('event');
        const [client, server] = Object.values(new WebSocketPair());
        this.ctx.acceptWebSocket(server);
        server.serializeAttachment({ eventId, cityKey: null, lastSeen: Date.now() });
        if (eventId) {
          const event = await this.env.DB.prepare(
            'SELECT city_key FROM visitor_events WHERE event_id = ? AND expires_at > ?'
          ).bind(eventId, Math.floor(Date.now() / 1000)).first();
          if (event?.city_key) {
            server.serializeAttachment({ eventId, cityKey: event.city_key, lastSeen: Date.now() });
          }
        }
        await this.publish();
        await this.scheduleCheck();
        return new Response(null, { status: 101, webSocket: client });
      }
      if (url.pathname === '/update' && request.method === 'POST') {
        const { eventId, cityKey } = await request.json();
        for (const socket of this.sockets()) {
          const state = socket.deserializeAttachment();
          if (state?.eventId === eventId) socket.serializeAttachment({ ...state, cityKey, lastSeen: Date.now() });
        }
        const data = await this.publish();
        await this.scheduleCheck();
        return Response.json(data, { headers: { 'Cache-Control': 'no-store' } });
      }
      if (url.pathname === '/stats') {
        return Response.json(await this.snapshot(), { headers: { 'Cache-Control': 'no-store' } });
      }
      return new Response('Not found', { status: 404 });
    });
  }

  webSocketMessage(socket, message) {
    return this.exclusive(async () => {
      if (message !== 'ping') {
        socket.close(1008, 'Unsupported message');
        return;
      }
      const state = socket.deserializeAttachment();
      if (!state) return;
      const wasExpired = Date.now() - state.lastSeen >= ONLINE_TIMEOUT;
      socket.serializeAttachment({ ...state, lastSeen: Date.now() });
      socket.send('pong');
      if (wasExpired && state.cityKey) await this.publish();
      await this.scheduleCheck();
    });
  }

  webSocketClose(socket, code, reason) {
    return this.exclusive(async () => {
      socket.serializeAttachment(null);
      try { socket.close(code === 1005 || code === 1006 ? 1000 : code, reason); } catch { /* already closed */ }
      await this.publish();
    });
  }

  webSocketError(socket) {
    return this.webSocketClose(socket, 1011, 'Connection lost');
  }

  alarm() {
    return this.exclusive(async () => {
      let changed = false;
      for (const socket of this.sockets()) {
        const state = socket.deserializeAttachment();
        if (!state || Date.now() - state.lastSeen >= ONLINE_TIMEOUT) {
          socket.serializeAttachment(null);
          socket.close(1000, 'Heartbeat expired');
          changed = true;
        }
      }
      if (changed) await this.publish();
      if (this.sockets().length) await this.ctx.storage.setAlarm(Date.now() + CHECK_INTERVAL);
    });
  }
}
