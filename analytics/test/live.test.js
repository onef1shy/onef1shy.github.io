import test from 'node:test';
import assert from 'node:assert/strict';
import { VisitorHub } from '../src/live.js';

function socket(cityKey = null, lastSeen = Date.now()) {
  let state = { eventId: 'private-page-id', cityKey, lastSeen };
  return {
    readyState: 1, messages: [],
    serializeAttachment(value) { state = value; },
    deserializeAttachment() { return state; },
    send(message) { this.messages.push(message); },
    close() { this.readyState = 3; },
  };
}

async function room(sockets) {
  const values = new Map();
  let alarm = null;
  const ctx = {
    getWebSockets: () => sockets,
    blockConcurrencyWhile(action) { this.ready = action(); },
    storage: {
      get: async key => values.get(key),
      put: async (key, value) => values.set(key, value),
      getAlarm: async () => alarm,
      setAlarm: async value => { alarm = value; },
    },
  };
  const city = key => ({ city_key: key, country_code: key, city: key,
    region: '', latitude: 1, longitude: 1, pageviews: 1,
    first_seen: 1, last_seen: Math.floor(Date.now() / 1000) });
  const env = { DB: {
    prepare: () => ({}),
    batch: async () => [
      { results: [{ pageviews: 2, located_pageviews: 2, first_seen: 1, last_seen: 2 }] },
      { results: [city('SG'), city('JP')] },
    ],
  } };
  const hub = new VisitorHub(ctx, env);
  await ctx.ready;
  return { hub, ctx, env };
}

test('a city stays orange until its last open page leaves, regardless of recent history', async () => {
  const observer = socket();
  const a = socket('SG');
  const b = socket('SG');
  const { hub } = await room([observer, a, b]);
  assert.deepEqual((await hub.snapshot()).cities.map(city => city.online), [true, false]);
  await hub.webSocketClose(a, 1000, 'Page left');
  assert.equal(JSON.parse(observer.messages.at(-1)).data.cities[0].online, true);
  await hub.webSocketClose(b, 1000, 'Page left');
  const final = JSON.parse(observer.messages.at(-1)).data;
  assert.deepEqual(final.cities.map(city => city.online), [false, false]);
  assert.equal(final.totalPageviews, 2);
  assert.ok(!JSON.stringify(final).includes('private-page-id'));
});

test('heartbeats preserve presence, abandoned connections expire, and revisions survive hibernation', async () => {
  const observer = socket();
  const visitor = socket('SG', Date.now() - 61000);
  const { hub, ctx, env } = await room([observer, visitor]);
  assert.equal((await hub.snapshot()).cities[0].online, false);
  await hub.webSocketMessage(visitor, 'ping');
  assert.ok(visitor.messages.includes('pong'));
  assert.equal(JSON.parse(observer.messages.at(-1)).data.cities[0].online, true);
  visitor.serializeAttachment({ ...visitor.deserializeAttachment(), lastSeen: Date.now() - 61000 });
  await hub.alarm();
  assert.equal(visitor.readyState, 3);
  assert.equal(JSON.parse(observer.messages.at(-1)).data.cities[0].online, false);
  const before = (await hub.snapshot()).revision;
  const restored = new VisitorHub(ctx, env);
  await ctx.ready;
  assert.ok((await restored.snapshot()).revision > before);
});
