import { lookupLegacyLocation } from './geolocation.js';
import { readStatistics } from './statistics.js';
export { VisitorHub } from './live.js';

const EVENT_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const EVENT_TTL = 86400;
const encoder = new TextEncoder();

function integer(value, fallback, min, max) {
  const result = Number(value);
  return Number.isInteger(result) && result >= min && result <= max ? result : fallback;
}

function json(body, status = 200, headers = {}) {
  return Response.json(body, {
    status,
    headers: { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...headers },
  });
}

function originAllowed(origin, env) {
  return Boolean(origin) && String(env.ALLOWED_ORIGINS || '')
    .split(',').map(value => value.trim()).filter(Boolean).includes(origin);
}

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
}

// Only Cloudflare's client header is accepted; a browser-supplied IP in the
// payload, X-Forwarded-For, or query string is never used for attribution.
export function publicClientIP(request) {
  const value = request.headers.get('CF-Connecting-IP')?.trim().toLowerCase();
  if (!value || value.length > 45) return null;
  if (/^\d+\.\d+\.\d+\.\d+$/.test(value)) {
    const parts = value.split('.').map(Number);
    if (parts.some(part => part > 255) || parts.join('.') !== value) return null;
    const [a, b] = parts;
    if (a === 0 || a === 10 || a === 127 || a >= 224 ||
        (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) ||
        (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127) ||
        (a === 198 && (b === 18 || b === 19))) return null;
    return value;
  }
  if (!/^[0-9a-f:]+$/.test(value)) return null;
  try {
    const ip = new URL(`https://[${value}]/`).hostname.slice(1, -1);
    if (ip === '::' || ip === '::1' || /^(fc|fd|ff|fe[89ab])/.test(ip)) return null;
    return ip;
  } catch {
    return null;
  }
}

async function dailyHash(ip, secret, now) {
  const key = await crypto.subtle.importKey('raw', encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const day = new Date(now * 1000).toISOString().slice(0, 10);
  const digest = await crypto.subtle.sign('HMAC', key, encoder.encode(`${day}\n${ip}`));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

async function readEvent(request) {
  if (!request.body || Number(request.headers.get('Content-Length')) > 512) return null;
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 512) { await reader.cancel(); return null; }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    const body = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    return body && EVENT_ID.test(body.eventId) ? body.eventId.toLowerCase() : null;
  } catch {
    return null;
  } finally {
    reader.releaseLock();
  }
}

async function withinRateLimit(db, hash, now, limit) {
  const window = Math.floor(now / 60) * 60;
  const row = await db.prepare(`
    INSERT INTO visitor_rate_limits (client_hash, window_start, request_count, expires_at)
    VALUES (?, ?, 1, ?)
    ON CONFLICT (client_hash) DO UPDATE SET
      request_count = CASE WHEN window_start = excluded.window_start THEN request_count + 1 ELSE 1 END,
      window_start = excluded.window_start, expires_at = excluded.expires_at
    RETURNING request_count
  `).bind(hash, window, now + 3600).first();
  return row.request_count <= limit;
}

async function reserveLookup(db, now, limit) {
  const month = new Date(now * 1000).toISOString().slice(0, 7);
  const row = await db.prepare(`
    INSERT INTO visitor_ipinfo_budget (month, attempts, updated_at) VALUES (?, 1, ?)
    ON CONFLICT (month) DO UPDATE SET attempts = attempts + 1, updated_at = excluded.updated_at
    WHERE attempts < ?
    RETURNING attempts
  `).bind(month, now, limit).first();
  return Boolean(row);
}

function cityKey(location) {
  return JSON.stringify([location.countryCode,
    location.region.normalize('NFKC').toLowerCase(), location.city.normalize('NFKC').toLowerCase()]);
}

async function locationFor(ip, hash, env, now, fetcher) {
  const cached = await env.DB.prepare(
    'SELECT location_json FROM visitor_geo_cache WHERE client_hash = ? AND expires_at > ?'
  ).bind(hash, now).first();
  if (cached) return cached.location_json ? JSON.parse(cached.location_json) : null;
  const limit = integer(env.IPINFO_MONTHLY_LIMIT, 45000, 1, 50000);
  if (!await reserveLookup(env.DB, now, limit)) return null;
  const location = await lookupLegacyLocation(ip, env.IPINFO_TOKEN, fetcher,
    code => console.warn(`Visitor geolocation unavailable: ${code}`));
  const ttl = location ? integer(env.IP_CACHE_TTL_SECONDS, 86400, 60, 86400) : 300;
  await env.DB.prepare(`
    INSERT INTO visitor_geo_cache (client_hash, location_json, expires_at) VALUES (?, ?, ?)
    ON CONFLICT (client_hash) DO UPDATE SET location_json = excluded.location_json, expires_at = excluded.expires_at
  `).bind(hash, location ? JSON.stringify(location) : null, now + ttl).run();
  return location;
}

async function recordEvent(db, eventId, location, now) {
  const result = await db.prepare(`
    INSERT OR IGNORE INTO visitor_events
      (event_id, city_key, country_code, city, region, latitude, longitude, recorded_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).bind(eventId, location ? cityKey(location) : null, location?.countryCode ?? null,
    location?.city ?? null, location?.region ?? null, location?.latitude ?? null,
    location?.longitude ?? null, now, now + EVENT_TTL).run();
  return result.meta.changes > 0;
}

export function createWorker(options = {}) {
  const clock = options.now || (() => Math.floor(Date.now() / 1000));
  const fetcher = options.fetch || ((...args) => fetch(...args));
  const inFlight = new Map();

  async function track(request, env) {
    const origin = request.headers.get('Origin');
    if (!originAllowed(origin, env)) return json({ error: 'Origin not allowed' }, 403);
    const headers = cors(origin);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers });
    if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405, { ...headers, Allow: 'POST, OPTIONS' });
    if (!env.IPINFO_TOKEN || !env.HMAC_SECRET || env.HMAC_SECRET.length < 32) {
      return json({ error: 'Tracking is not configured' }, 503, headers);
    }
    if (request.cf?.botManagement?.verifiedBot) return json({ recorded: false, reason: 'bot' }, 202, headers);
    const ip = publicClientIP(request);
    if (!ip) return json({ error: 'Client IP unavailable' }, 400, headers);
    const eventId = await readEvent(request);
    if (!eventId) return json({ error: 'Invalid event' }, 400, headers);
    const existing = await env.DB.prepare('SELECT event_id FROM visitor_events WHERE event_id = ?').bind(eventId).first();
    if (existing) return json({ recorded: false, reason: 'duplicate' }, 200, headers);
    const now = clock();
    const hash = await dailyHash(ip, env.HMAC_SECRET, now);
    const rate = integer(env.TRACK_RATE_LIMIT, 10, 1, 120);
    if (!await withinRateLimit(env.DB, hash, now, rate)) {
      return json({ recorded: false, reason: 'rate_limited' }, 429, { ...headers, 'Retry-After': '60' });
    }
    // Merge same-IP lookups inside an instance; D1 enforces the monthly ceiling
    // across instances. A location outage still produces an unlocated pageview.
    if (!inFlight.has(hash)) {
      inFlight.set(hash, locationFor(ip, hash, env, now, fetcher).finally(() => inFlight.delete(hash)));
    }
    let location = null;
    try { location = await inFlight.get(hash); } catch { /* preserve the pageview */ }
    const recorded = await recordEvent(env.DB, eventId, location, now);
    let latest;
    if (recorded && env.VISITOR_HUB) {
      try {
        const response = await env.VISITOR_HUB.getByName('world').fetch('https://visitor-map.internal/update', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ eventId, cityKey: location ? cityKey(location) : null }),
        });
        if (response.ok) latest = await response.json();
      } catch { /* live updates must not undo a recorded pageview */ }
    }
    return json({ recorded, located: recorded && Boolean(location),
      ...(!recorded ? { reason: 'duplicate' } : {}), ...(latest ? { stats: latest } : {}) }, 200, headers);
  }

  async function stats(request, env, ctx) {
    const headers = { 'Access-Control-Allow-Origin': '*' };
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers });
    if (request.method !== 'GET') return json({ error: 'Method not allowed' }, 405, { ...headers, Allow: 'GET, OPTIONS' });
    if (env.VISITOR_HUB) {
      try {
        const response = await env.VISITOR_HUB.getByName('world').fetch('https://visitor-map.internal/stats');
        if (response.ok) return json(await response.json(), 200, headers);
      } catch { /* cumulative statistics remain available without the live room */ }
    }
    const ttl = integer(env.STATS_CACHE_SECONDS, 0, 0, 300);
    const cache = options.cache === undefined ? globalThis.caches?.default : options.cache;
    const key = new Request(new URL('/stats', request.url));
    if (ttl && cache) {
      try {
        const hit = await cache.match(key);
        if (hit) return hit;
      } catch { /* the database remains available if the edge cache fails */ }
    }
    const response = json(await readStatistics(env.DB, clock()), 200,
      { ...headers, 'Cache-Control': ttl ? `public, max-age=${ttl}` : 'no-store' });
    if (ttl && cache && ctx?.waitUntil) ctx.waitUntil(cache.put(key, response.clone()).catch(() => {}));
    return response;
  }

  return {
    async fetch(request, env, ctx) {
      try {
        const path = new URL(request.url).pathname;
        if (!['/track', '/stats', '/health', '/live'].includes(path)) return json({ error: 'Not found' }, 404);
        if (!env.DB) return json({ error: 'Database not configured' }, 503);
        if (path === '/track') return await track(request, env);
        if (path === '/stats') return await stats(request, env, ctx);
        if (path === '/live') {
          if (!originAllowed(request.headers.get('Origin'), env)) return json({ error: 'Origin not allowed' }, 403);
          if (request.method !== 'GET' || request.headers.get('Upgrade')?.toLowerCase() !== 'websocket') {
            return json({ error: 'WebSocket connection required' }, 426);
          }
          const eventId = new URL(request.url).searchParams.get('event');
          if (eventId && !EVENT_ID.test(eventId)) return json({ error: 'Invalid event' }, 400);
          if (!env.VISITOR_HUB) return json({ error: 'Live map unavailable' }, 503);
          const target = new URL('https://visitor-map.internal/live');
          if (eventId) target.searchParams.set('event', eventId.toLowerCase());
          return env.VISITOR_HUB.getByName('world').fetch(new Request(target, { headers: { Upgrade: 'websocket' } }));
        }
        if (request.method !== 'GET') return json({ error: 'Method not allowed' }, 405, { Allow: 'GET' });
        await env.DB.prepare('SELECT 1 AS ready').first();
        const configured = Boolean(env.IPINFO_TOKEN && env.HMAC_SECRET?.length >= 32);
        return json({ status: configured ? 'ok' : 'not_configured', provider: 'ipinfo-legacy' }, configured ? 200 : 503);
      } catch {
        // Do not log requests, IPs, tokens, or provider responses.
        const origin = request.headers.get('Origin');
        const headers = new URL(request.url).pathname === '/stats'
          ? { 'Access-Control-Allow-Origin': '*' }
          : originAllowed(origin, env) ? cors(origin) : {};
        return json({ error: 'Statistics temporarily unavailable' }, 503, headers);
      }
    },
    async scheduled(_event, env) {
      const now = clock();
      const oldestMonth = new Date((now - 62 * 86400) * 1000).toISOString().slice(0, 7);
      await env.DB.batch([
        env.DB.prepare('DELETE FROM visitor_events WHERE expires_at <= ?').bind(now),
        env.DB.prepare('DELETE FROM visitor_geo_cache WHERE expires_at <= ?').bind(now),
        env.DB.prepare('DELETE FROM visitor_rate_limits WHERE expires_at <= ?').bind(now),
        env.DB.prepare('DELETE FROM visitor_ipinfo_budget WHERE month < ?').bind(oldestMonth),
      ]);
    },
  };
}

export default createWorker();
