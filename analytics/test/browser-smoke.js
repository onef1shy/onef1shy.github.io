import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = process.env.PREVIEW_URL || 'http://127.0.0.1:5501';
const api = 'https://visitor-map.test';
const now = Math.floor(Date.now() / 1000);
const fixtures = [
  ['Shanghai', 'Shanghai', 'CN', 31.2304, 121.4737, 48],
  ['New York', 'New York', 'US', 40.7128, -74.0060, 40],
  ['Tokyo', 'Tokyo', 'JP', 35.6762, 139.6503, 25],
  ['London', 'England', 'GB', 51.5072, -0.1276, 24],
  ['Hangzhou', 'Zhejiang', 'CN', 30.2741, 120.1551, 15],
  ['Sydney', 'New South Wales', 'AU', -33.8688, 151.2093, 5],
].map(([city, region, countryCode, latitude, longitude, pageviews], index) => ({
  city, region, countryCode, latitude, longitude, pageviews,
  firstSeen: now - 500000, lastSeen: index % 2 ? now - 172800 : now,
  online: index === 1,
}));
const stats = {
  version: 1, totalPageviews: 163, locatedPageviews: 157,
  countries: 5, cityCount: 6, firstSeen: now - 500000, lastSeen: now,
  generatedAt: now, revision: 1, cities: fixtures,
};

const browser = await chromium.launch({
  executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
});
try {
  for (const [name, viewport] of [['desktop', { width: 1280, height: 900 }], ['mobile', { width: 390, height: 844 }]]) {
    const context = await browser.newContext({ viewport, reducedMotion: 'reduce' });
    const page = await context.newPage();
    const errors = [];
    const requests = [];
    let trackRequests = 0;
    let statsRequests = 0;
    let currentStats = stats;
    let nextStatsGate = null;
    let liveSocket;
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => requests.push(request.url()));
    await page.route('**/assets/visitor-map/config.js', route => route.fulfill({
      contentType: 'text/javascript',
      body: `export default { apiBase: '${api}', trackOrigins: [], trackLocalhost: true, respectDoNotTrack: false };`,
    }));
    await page.routeWebSocket('wss://visitor-map.test/live**', socket => {
      liveSocket = socket;
      socket.onMessage(message => { if (message === 'ping') socket.send('pong'); });
      socket.send(JSON.stringify({ type: 'stats', data: currentStats }));
    });
    await page.route(`${api}/**`, async route => {
      const isTrack = new URL(route.request().url()).pathname === '/track';
      if (isTrack) trackRequests++; else statsRequests++;
      const body = isTrack ? { recorded: true, located: true } : currentStats;
      if (!isTrack && nextStatsGate) {
        const gate = nextStatsGate;
        nextStatsGate = null;
        gate.started();
        await gate.wait;
      }
      return route.fulfill({
        contentType: 'application/json', headers: { 'Access-Control-Allow-Origin': '*' },
        body: JSON.stringify(body),
      });
    });
    await page.goto(base, { waitUntil: 'networkidle' });
    assert.equal(trackRequests, 1, `${name}: tracking must start without scrolling to the map`);
    assert.ok(statsRequests >= 1, `${name}: statistics must load before scrolling`);
    assert.ok(liveSocket, `${name}: live updates must connect before scrolling`);
    await page.locator('[data-visitor-map]').scrollIntoViewIfNeeded();
    await page.waitForFunction(() => document.querySelector('[data-stat="pageviews"]').textContent === '163');
    assert.equal(await page.locator('.visitor-map-marker').count(), 6);
    assert.equal(await page.locator('[data-stat="countries"]').textContent(), '5');
    assert.equal(await page.locator('[data-stat="cities"]').textContent(), '6');
    assert.equal(await page.locator('.visitor-map-marker.is-online').count(), 1);
    assert.equal(await page.locator('.visitor-map-marker[aria-label^="Shanghai"]').evaluate(el => getComputedStyle(el).fill), 'rgb(255, 23, 23)');
    assert.equal(await page.locator('.visitor-map-marker[aria-label^="New York"]').evaluate(el => getComputedStyle(el).fill), 'rgb(247, 154, 0)');
    const marker = page.locator('.visitor-map-marker[aria-label^="New York"]');
    if (name === 'desktop') await marker.hover(); else await marker.click();
    await page.waitForFunction(() => !document.querySelector('.visitor-map-tooltip').hidden);
    assert.match(await page.locator('.visitor-map-tooltip').textContent(), /New York.*40 pageviews/);
    await marker.focus();
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.visitor-map-tooltip').isVisible(), false);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `${name}: no horizontal overflow`);
    assert.equal(await page.locator('.visitor-map-world').evaluate(image => image.complete && image.naturalWidth > 0), true);
    const metrics = await page.locator('.visitor-map-metrics').boundingBox();
    const backToTop = await page.locator('#backToTop').boundingBox();
    if (backToTop) {
      assert.ok(metrics.x + metrics.width <= backToTop.x || backToTop.x + backToTop.width <= metrics.x ||
        metrics.y + metrics.height <= backToTop.y || backToTop.y + backToTop.height <= metrics.y,
      `${name}: floating button must not cover visitor statistics`);
    }
    await page.locator('.site-footer').screenshot({ path: `/tmp/onef1shy-owned-map-${name}.png` });

    // A pushed visitor must appear immediately, even with an older GET in flight.
    let started;
    let release;
    const waiting = new Promise(resolve => { started = resolve; });
    nextStatsGate = { started, wait: new Promise(resolve => { release = resolve; }) };
    await page.evaluate(() => window.dispatchEvent(new Event('visitor-map:tracked')));
    await waiting;
    currentStats = {
      ...stats, revision: 2, totalPageviews: 164, locatedPageviews: 158, countries: 6, cityCount: 7,
      cities: [...fixtures, { city: 'Singapore', region: 'Singapore', countryCode: 'SG',
        latitude: 1.2897, longitude: 103.8501, pageviews: 1, firstSeen: now, lastSeen: now, online: true }],
    };
    liveSocket.send(JSON.stringify({ type: 'stats', data: currentStats }));
    await page.waitForFunction(() => document.querySelector('[data-stat="pageviews"]').textContent === '164', null, { timeout: 1500 });
    const oldResponse = page.waitForResponse(`${api}/stats`);
    release();
    await oldResponse;
    assert.equal(await page.locator('[data-stat="pageviews"]').textContent(), '164', `${name}: late old response must not undo a push`);
    assert.equal(await page.locator('.visitor-map-marker[aria-label^="Singapore"]').count(), 1);
    assert.equal(await page.locator('.visitor-map-marker.is-online[aria-label^="Singapore"]').count(), 1);
    currentStats = { ...currentStats, revision: 3, cities: currentStats.cities.map(city => ({ ...city, online: false })) };
    liveSocket.send(JSON.stringify({ type: 'stats', data: currentStats }));
    await page.waitForFunction(() => document.querySelectorAll('.visitor-map-marker.is-online').length === 0, null, { timeout: 1500 });
    assert.equal(await page.locator('.visitor-map-marker[aria-label^="Singapore"]').evaluate(el => getComputedStyle(el).fill), 'rgb(255, 23, 23)');
    assert.deepEqual(errors, [], `${name}: JavaScript errors`);
    assert.ok(!requests.some(url => /mapmyvisitors|ipinfo\.io|waust\.at/.test(url)), `${name}: no provider scripts or browser-side geolocation calls`);
    console.log(`${name}: immediate initial load, pushed visitors, online orange/offline red, and stale-response protection passed.`);
    await context.close();
  }

  const page = await browser.newPage();
  await page.route('**/assets/visitor-map/config.js', route => route.fulfill({
    contentType: 'text/javascript',
    body: 'export default { apiBase: "", trackOrigins: [], trackLocalhost: false, respectDoNotTrack: true };',
  }));
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('[data-visitor-map]').scrollIntoViewIfNeeded();
  assert.equal(await page.locator('[data-stat="pageviews"]').textContent(), '—');
  assert.equal(await page.locator('.visitor-map-marker').count(), 0);
  console.log('Unconfigured preview: no fabricated counts or sample markers.');
} finally {
  await browser.close();
}
