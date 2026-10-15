import { apiURL } from './client.js';
import { visit } from './visit.js';

// A connection belongs to the open page, including when the footer is offscreen.
export function connectLive(onStats, onStatus) {
  const endpoint = apiURL('/live');
  if (!endpoint || !globalThis.WebSocket) return () => {};
  let socket;
  let retry;
  let heartbeat;
  let watchdog;
  let attempts = 0;
  let paused = false;

  function clearTimers() {
    clearTimeout(retry);
    clearInterval(heartbeat);
    clearTimeout(watchdog);
  }

  function close() {
    clearTimers();
    const old = socket;
    socket = null;
    if (old) old.close(1000, 'Page left');
  }

  function open() {
    if (paused || socket) return;
    const url = new URL(endpoint);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    if (visit.eventId) url.searchParams.set('event', visit.eventId);
    const connection = new WebSocket(url);
    socket = connection;
    connection.addEventListener('open', () => {
      if (socket !== connection) return;
      attempts = 0;
      onStatus(true);
      const ping = () => {
        if (connection.readyState !== WebSocket.OPEN) return;
        connection.send('ping');
        clearTimeout(watchdog);
        watchdog = setTimeout(() => connection.close(4000, 'Heartbeat timed out'), 15000);
      };
      heartbeat = setInterval(ping, 20000);
      ping();
    });
    connection.addEventListener('message', event => {
      if (socket !== connection) return;
      if (event.data === 'pong') { clearTimeout(watchdog); return; }
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'stats') onStats(message.data);
      } catch { /* ignore unrelated or invalid messages */ }
    });
    connection.addEventListener('close', () => {
      if (socket !== connection) return;
      socket = null;
      clearTimers();
      onStatus(false);
      if (!paused) retry = setTimeout(open, Math.min(15000, 1000 * 2 ** attempts++));
    });
    connection.addEventListener('error', () => connection.close());
  }

  window.addEventListener('visitor-map:visit', () => { close(); open(); });
  window.addEventListener('pagehide', () => { paused = true; close(); });
  window.addEventListener('pageshow', () => { paused = false; open(); });
  window.addEventListener('online', () => { if (!socket) { clearTimeout(retry); open(); } });
  open();
  return close;
}
