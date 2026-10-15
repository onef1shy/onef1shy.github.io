import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { createWorker, publicClientIP } from '../src/index.js';
import { parseLegacyLocation } from '../src/geolocation.js';
import { projectLocation, validStats } from '../../assets/visitor-map/map.js';
import { projection } from '../../assets/visitor-map/projection.js';

const migration = readFileSync(new URL('../migrations/0001_visitors.sql', import.meta.url), 'utf8');
const origin = 'https://onef1shy.github.io';
const location = { city: 'Tokyo', region: 'Tokyo', country: 'JP', loc: '35.6762,139.6503' };

// Real SQLite executes the production migration, trigger, conflict handling,
// indexes, and rollback. The adapter only supplies D1's promise-based interface.
function d1(sqlite) {
  function prepared(sql, values = []) {
    return {
      bind: (...bound) => prepared(sql, bound),
      first: async () => sqlite.prepare(sql).get(...values) || null,
      all: async () => ({ results: sqlite.prepare(sql).all(...values), success: true }),
      run: async () => ({ meta: { changes: sqlite.prepare(sql).run(...values).changes }, success: true }),
    };
  }
  return {
    prepare: prepared,
    async batch(statements) {
      sqlite.exec('BEGIN');
      try {
        const results = [];
        for (const statement of statements) results.push(await statement.all());
        sqlite.exec('COMMIT');
        return results;
      } catch (error) {
        sqlite.exec('ROLLBACK');
        throw error;
      }
    },
  };
}

function harness(overrides = {}) {
  const sqlite = new DatabaseSync(':memory:');
  sqlite.exec(migration);
  const env = {
    DB: d1(sqlite), IPINFO_TOKEN: 'test-only-not-a-real-token',
    HMAC_SECRET: 'test-only-32-character-secret-do-not-deploy',
    ALLOWED_ORIGINS: origin, TRACK_RATE_LIMIT: '120', STATS_CACHE_SECONDS: '0',
    ...overrides.env,
  };
  let now = 1792110000;
  const calls = [];
  const worker = createWorker({
    now: () => now,
    cache: overrides.cache || null,
    fetch: async (url, options) => {
      calls.push({ url, options });
      return overrides.fetch ? overrides.fetch(url, options) : Response.json(location);
    },
  });
  return {
    sqlite, env, worker, calls,
    advance: seconds => { now += seconds; },
    async track({ id = randomUUID(), ip = '8.8.8.8', requestOrigin = origin, payload, headers = {} } = {}) {
      return worker.fetch(new Request('https://visitors.example/track', {
        method: 'POST', headers: { Origin: requestOrigin, 'CF-Connecting-IP': ip, 'Content-Type': 'text/plain', ...headers },
        body: payload === undefined ? JSON.stringify({ eventId: id }) : payload,
      }), env);
    },
    async stats() {
      return (await worker.fetch(new Request('https://visitors.example/stats'), env)).json();
    },
  };
}

test('one pageview produces one accurate city aggregate and no public identifiers', async () => {
  const h = harness();
  const response = await h.track();
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
  assert.deepEqual(await response.json(), { recorded: true, located: true });
  const stats = await h.stats();
  assert.ok(validStats(stats));
  assert.equal(stats.totalPageviews, 1);
  assert.equal(stats.locatedPageviews, 1);
  assert.equal(stats.countries, 1);
  assert.equal(stats.cityCount, 1);
  assert.equal(stats.cities[0].latitude, 35.6762);
  assert.equal(stats.cities[0].longitude, 139.6503);
  assert.equal(h.calls[0].url, 'https://ipinfo.io/8.8.8.8/json');
  assert.equal(h.calls[0].options.headers.Authorization, `Bearer ${h.env.IPINFO_TOKEN}`);
  const publicData = JSON.stringify(stats);
  assert.ok(!publicData.includes('8.8.8.8'));
  assert.ok(!publicData.includes(h.env.IPINFO_TOKEN));
  assert.ok(!publicData.includes('client_hash'));
  assert.ok(!publicData.includes('event_id'));
});

test('concurrent retries of one event are counted exactly once', async () => {
  const h = harness({ fetch: async () => {
    await new Promise(resolve => setTimeout(resolve, 15));
    return Response.json(location);
  } });
  const id = randomUUID();
  const responses = await Promise.all(Array.from({ length: 20 }, () => h.track({ id })));
  const results = await Promise.all(responses.map(response => response.json()));
  assert.equal(results.filter(result => result.recorded).length, 1);
  const stats = await h.stats();
  assert.equal(stats.totalPageviews, 1);
  assert.equal(stats.cities[0].pageviews, 1);
  assert.equal(h.calls.length, 1);
});

test('separate pageviews count separately while reusing IP geolocation', async () => {
  const h = harness();
  await h.track();
  await h.track();
  const stats = await h.stats();
  assert.equal(stats.totalPageviews, 2);
  assert.equal(stats.cities[0].pageviews, 2);
  assert.equal(h.calls.length, 1);
});

test('unapproved origins, missing client IP, malformed and oversized bodies do not spend quota', async () => {
  const h = harness();
  assert.equal((await h.track({ requestOrigin: 'https://untrusted.example' })).status, 403);
  assert.equal((await h.track({ requestOrigin: '' })).status, 403);
  assert.equal((await h.track({ ip: '' })).status, 400);
  assert.equal((await h.track({ ip: '127.0.0.1' })).status, 400);
  assert.equal((await h.track({ payload: '{}' })).status, 400);
  assert.equal((await h.track({ payload: 'x'.repeat(513) })).status, 400);
  assert.equal(h.calls.length, 0);
  assert.equal((await h.stats()).totalPageviews, 0);
});

test('payload and X-Forwarded-For cannot override the Cloudflare client IP', async () => {
  const h = harness();
  await h.track({ payload: JSON.stringify({ eventId: randomUUID(), ip: '1.1.1.1' }), headers: { 'X-Forwarded-For': '1.1.1.1' } });
  assert.equal(h.calls[0].url, 'https://ipinfo.io/8.8.8.8/json');
  assert.equal(publicClientIP(new Request('https://x', { headers: { 'CF-Connecting-IP': '2606:4700:4700::1111' } })), '2606:4700:4700::1111');
});

test('per-IP rate limit rejects refresh spam and recovers in the next minute', async () => {
  const h = harness({ env: { TRACK_RATE_LIMIT: '2' } });
  await h.track();
  await h.track();
  assert.equal((await h.track()).status, 429);
  assert.equal((await h.stats()).totalPageviews, 2);
  h.advance(61);
  assert.equal((await h.track()).status, 200);
  assert.equal((await h.stats()).totalPageviews, 3);
  assert.equal(h.calls.length, 1);
});

test('the monthly ceiling is global across IPs and never drops the pageview', async () => {
  const h = harness({ env: { IPINFO_MONTHLY_LIMIT: '1' } });
  await h.track({ ip: '8.8.8.8' });
  const response = await h.track({ ip: '1.1.1.1' });
  assert.deepEqual(await response.json(), { recorded: true, located: false });
  const stats = await h.stats();
  assert.equal(stats.totalPageviews, 2);
  assert.equal(stats.locatedPageviews, 1);
  assert.equal(h.calls.length, 1);
  assert.ok(validStats(stats));
});

test('separate Worker instances share an atomic monthly lookup ceiling', async () => {
  const h = harness({ env: { IPINFO_MONTHLY_LIMIT: '1' } });
  let otherCalls = 0;
  const other = createWorker({ cache: null, now: () => 1792110000, fetch: async () => {
    otherCalls++;
    return Response.json(location);
  } });
  await Promise.all([
    h.track(),
    other.fetch(new Request('https://visitors.example/track', {
      method: 'POST', headers: { Origin: origin, 'CF-Connecting-IP': '1.1.1.1' },
      body: JSON.stringify({ eventId: randomUUID() }),
    }), h.env),
  ]);
  assert.equal(h.calls.length + otherCalls, 1);
  const stats = await h.stats();
  assert.equal(stats.totalPageviews, 2);
  assert.equal(stats.locatedPageviews, 1);
});

test('a failed edge cache does not take the statistics endpoint offline', async () => {
  const h = harness({
    env: { STATS_CACHE_SECONDS: '60' },
    cache: { async match() { throw new Error('cache unavailable'); } },
  });
  await h.track();
  const stats = await h.stats();
  assert.equal(stats.totalPageviews, 1);
  assert.ok(validStats(stats));
});

test('stats caching never caches tracking requests or private lookup data', async () => {
  const entries = new Map();
  const h = harness({
    env: { STATS_CACHE_SECONDS: '60' },
    cache: {
      async match(request) { return entries.get(request.url)?.clone(); },
      async put(request, response) { entries.set(request.url, response); },
    },
  });
  await h.track();
  const pending = [];
  const response = await h.worker.fetch(new Request('https://visitors.example/stats?unused=1'), h.env,
    { waitUntil: promise => pending.push(promise) });
  await Promise.all(pending);
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=60');
  assert.deepEqual([...entries.keys()], ['https://visitors.example/stats']);
  const cached = await h.worker.fetch(new Request('https://visitors.example/stats'), h.env);
  assert.equal((await cached.json()).totalPageviews, 1);
  assert.ok(!JSON.stringify(await entries.values().next().value.clone().json()).includes('client_hash'));
});

test('provider outages and Lite responses do not fabricate map locations', async () => {
  for (const fetch of [
    async () => new Response('rate limited', { status: 429 }),
    async () => new Response(null, { status: 302, headers: { Location: 'https://untrusted.example' } }),
    async () => { throw new Error('network unavailable'); },
    async () => Response.json({ country_code: 'US', country: 'United States' }),
  ]) {
    const h = harness({ fetch });
    assert.equal((await h.track()).status, 200);
    await h.track();
    const stats = await h.stats();
    assert.equal(stats.totalPageviews, 2);
    assert.equal(stats.locatedPageviews, 0);
    assert.equal(stats.cityCount, 0);
    assert.equal(h.calls.length, 1);
    assert.ok(validStats(stats));
  }
});

test('a failed aggregate update rolls back both the event and the total', async () => {
  const h = harness();
  assert.throws(() => h.sqlite.prepare(`INSERT INTO visitor_events
    (event_id, city_key, recorded_at, expires_at) VALUES (?, 'invalid', 1, 2)`).run(randomUUID()));
  assert.equal((await h.stats()).totalPageviews, 0);
  assert.equal(h.sqlite.prepare('SELECT COUNT(*) AS count FROM visitor_events').get().count, 0);
});

test('hourly cleanup removes short-lived data while preserving every historical city and total', async () => {
  const h = harness();
  await h.track();
  h.advance(86401);
  await h.worker.scheduled({}, h.env);
  for (const table of ['visitor_events', 'visitor_geo_cache', 'visitor_rate_limits']) {
    assert.equal(h.sqlite.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count, 0);
  }
  const stats = await h.stats();
  assert.equal(stats.totalPageviews, 1);
  assert.equal(stats.cityCount, 1);
  assert.equal(stats.cities[0].city, 'Tokyo');
});

test('raw IPs and tokens are absent from persisted tables', async () => {
  const h = harness();
  await h.track();
  const tables = h.sqlite.prepare("SELECT name FROM sqlite_master WHERE type = 'table'").all();
  for (const { name } of tables) {
    const data = JSON.stringify(h.sqlite.prepare(`SELECT * FROM ${name}`).all());
    assert.ok(!data.includes('8.8.8.8'), name);
    assert.ok(!data.includes(h.env.IPINFO_TOKEN), name);
  }
});

test('coordinates within the displayed latitude range fit the map, including the date line', () => {
  for (const latitude of [projection.latitudeBounds[0], -55, 0, 60, projection.latitudeBounds[1]]) {
    for (const longitude of [-180, -120, 0, 120, 180]) {
      const [x, y] = projectLocation(longitude, latitude);
      assert.ok(x >= 0 && x <= projection.width, `${longitude}, ${latitude}: x=${x}`);
      assert.ok(y >= 0 && y <= projection.height, `${longitude}, ${latitude}: y=${y}`);
    }
  }
  const origin = parseLegacyLocation({ city: 'Null Island', country: 'GH', loc: '0,0' });
  assert.equal(origin.latitude, 0);
  assert.equal(origin.longitude, 0);
  assert.equal(parseLegacyLocation({ city: 'Bad', country: 'US', loc: ',' }), null);
});

test('stats reject inconsistent aggregates rather than displaying misleading counts', () => {
  assert.equal(validStats({ version: 1, totalPageviews: 0, locatedPageviews: 1, countries: 0, cityCount: 0, cities: [] }), false);
});

test('live origin checks prevent unwanted connections and a room outage never loses a pageview', async () => {
  let calls = 0;
  const h = harness({ env: { VISITOR_HUB: { getByName() { return { fetch: async () => {
    calls++;
    throw new Error('Room unavailable');
  } }; } } } });
  const live = headers => h.worker.fetch(new Request('https://visitors.example/live', { headers }), h.env);
  assert.equal((await live({ Origin: 'https://untrusted.example', Upgrade: 'websocket' })).status, 403);
  assert.equal((await live({ Origin: origin })).status, 426);
  assert.equal(calls, 0);
  assert.equal((await h.track()).status, 200);
  assert.equal((await h.stats()).totalPageviews, 1);
});
