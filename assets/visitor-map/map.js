import { apiURL, requestJSON } from './client.js';
import { projection } from './projection.js?v=20261016-no-antarctica';
import { connectLive } from './live.js';
import { visit } from './visit.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const number = new Intl.NumberFormat('en-US');
const countries = typeof Intl.DisplayNames === 'function'
  ? new Intl.DisplayNames(['en'], { type: 'region' }) : null;
const nonnegative = value => Number.isSafeInteger(value) && value >= 0;

export function projectLocation(longitude, latitude) {
  const radians = Math.PI / 180;
  const rotated = ((longitude - projection.centralMeridian + 180) % 360 + 360) % 360 - 180;
  const x = projection.translate[0] + projection.scale * rotated * radians;
  const y = projection.translate[1] - projection.scale * projection.verticalScale *
    1.25 * Math.log(Math.tan(Math.PI / 4 + 0.4 * latitude * radians));
  return [x, y];
}

export function validStats(data) {
  if (!data || data.version !== 1 || !Array.isArray(data.cities)) return false;
  if (data.revision !== undefined && !nonnegative(data.revision)) return false;
  if (!['totalPageviews', 'locatedPageviews', 'countries', 'cityCount'].every(key => nonnegative(data[key]))) return false;
  if (data.cities.length !== data.cityCount || data.locatedPageviews > data.totalPageviews) return false;
  let located = 0;
  const countryCodes = new Set();
  for (const city of data.cities) {
    if (!city || typeof city.city !== 'string' || !city.city.trim() || city.city.length > 160 ||
        !/^[A-Z]{2}$/.test(city.countryCode) || typeof city.region !== 'string' || city.region.length > 160 ||
        !Number.isFinite(city.latitude) || Math.abs(city.latitude) > 90 ||
        !Number.isFinite(city.longitude) || Math.abs(city.longitude) > 180 ||
        !nonnegative(city.pageviews) || city.pageviews === 0 ||
        !nonnegative(city.firstSeen) || !nonnegative(city.lastSeen) ||
        (city.online !== undefined && typeof city.online !== 'boolean')) return false;
    located += city.pageviews;
    countryCodes.add(city.countryCode);
  }
  return located === data.locatedPageviews && countryCodes.size === data.countries;
}

function setupMap(root) {
  const url = apiURL('/stats');
  if (!url) return;
  const canvas = root.querySelector('.visitor-map-canvas');
  const layer = root.querySelector('.visitor-map-points');
  const tooltip = root.querySelector('.visitor-map-tooltip');
  const status = root.querySelector('.visitor-map-status');
  let busy = false;
  let refreshQueued = false;
  let hasStats = false;
  let timer;
  let pinned = null;
  let latestRevision = -1;

  function hideTooltip() {
    tooltip.hidden = true;
    if (pinned) pinned.setAttribute('aria-pressed', 'false');
    pinned = null;
  }

  function showTooltip(marker, city, pin = false) {
    if (pinned && !pin) return;
    const country = countries?.of(city.countryCode) || city.countryCode;
    tooltip.replaceChildren();
    const title = document.createElement('strong');
    title.textContent = [city.city, city.region, country].filter(Boolean).join(', ');
    const count = document.createElement('span');
    count.textContent = `${number.format(city.pageviews)} ${city.pageviews === 1 ? 'pageview' : 'pageviews'}`;
    if (city.online) count.textContent += ' · Online now';
    tooltip.append(title, count);
    tooltip.hidden = false;
    const bounds = canvas.getBoundingClientRect();
    const point = marker.getBoundingClientRect();
    const left = point.left + point.width / 2 - bounds.left - tooltip.offsetWidth / 2;
    let top = point.top - bounds.top - tooltip.offsetHeight - 8;
    if (top < 0) top = point.bottom - bounds.top + 8;
    tooltip.style.left = `${Math.max(4, Math.min(left, bounds.width - tooltip.offsetWidth - 4))}px`;
    tooltip.style.top = `${top}px`;
    if (pin) {
      if (pinned) pinned.setAttribute('aria-pressed', 'false');
      pinned = marker;
      marker.setAttribute('aria-pressed', 'true');
    }
  }

  function render(data) {
    if (!validStats(data)) return;
    if (data.revision !== undefined) {
      if (data.revision < latestRevision) return;
      latestRevision = data.revision;
    } else if (latestRevision >= 0) return;
    hideTooltip();
    root.querySelector('[data-stat="pageviews"]').textContent = number.format(data.totalPageviews);
    root.querySelector('[data-stat="countries"]').textContent = number.format(data.countries);
    root.querySelector('[data-stat="cities"]').textContent = number.format(data.cityCount);
    const fragment = document.createDocumentFragment();
    // Large dots first so small neighboring cities remain accessible on top.
    for (const city of [...data.cities].sort((a, b) => b.pageviews - a.pageviews)) {
      if (city.latitude < projection.latitudeBounds[0] || city.latitude > projection.latitudeBounds[1]) continue;
      const [x, y] = projectLocation(city.longitude, city.latitude);
      const marker = document.createElementNS(SVG_NS, 'circle');
      marker.setAttribute('cx', x.toFixed(3));
      marker.setAttribute('cy', y.toFixed(3));
      marker.setAttribute('r', Math.min(11, 4 + Math.log2(city.pageviews + 1)).toFixed(2));
      marker.setAttribute('class', `visitor-map-marker${city.online === true ? ' is-online' : ''}`);
      marker.setAttribute('tabindex', '0');
      marker.setAttribute('role', 'button');
      marker.setAttribute('aria-pressed', 'false');
      marker.setAttribute('aria-label', `${city.city}, ${countries?.of(city.countryCode) || city.countryCode}: ${number.format(city.pageviews)} pageviews`);
      marker.addEventListener('pointerenter', () => showTooltip(marker, city));
      marker.addEventListener('pointerleave', () => { if (!pinned) tooltip.hidden = true; });
      marker.addEventListener('focus', () => showTooltip(marker, city));
      marker.addEventListener('blur', () => { if (!pinned) tooltip.hidden = true; });
      const select = event => {
        event.stopPropagation();
        if (pinned === marker) hideTooltip();
        else showTooltip(marker, city, true);
      };
      marker.addEventListener('click', select);
      marker.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(event); }
        if (event.key === 'Escape') hideTooltip();
      });
      fragment.append(marker);
    }
    layer.replaceChildren(fragment);
    status.hidden = true;
    hasStats = true;
  }

  async function refresh() {
    if (busy) {
      refreshQueued = true;
      return;
    }
    busy = true;
    try {
      const data = await requestJSON(url, { cache: 'no-store' });
      if (!validStats(data)) throw new Error('Invalid statistics');
      render(data);
    } catch {
      if (!hasStats) {
        status.textContent = 'Visitor statistics are temporarily unavailable.';
        status.hidden = false;
      }
    } finally {
      busy = false;
      if (refreshQueued) {
        refreshQueued = false;
        void refresh();
      }
    }
  }

  // Load on every page opening; subsequent visits arrive through the live socket.
  if (visit.result?.stats) render(visit.result.stats);
  void refresh();
  connectLive(render, connected => {
    clearInterval(timer);
    if (!connected) {
      void refresh();
      timer = setInterval(() => { if (!document.hidden) void refresh(); }, 2000);
    }
  });
  window.addEventListener('visitor-map:tracked', event => {
    if (validStats(event.detail?.stats)) render(event.detail.stats);
    else void refresh();
  });
  window.addEventListener('pagehide', () => clearInterval(timer));
  document.addEventListener('visibilitychange', () => { if (!document.hidden) void refresh(); });
  document.addEventListener('click', event => { if (!canvas.contains(event.target)) hideTooltip(); });
  canvas.addEventListener('click', event => { if (!event.target.closest('.visitor-map-marker')) hideTooltip(); });
  window.addEventListener('resize', hideTooltip);
}

if (typeof document !== 'undefined') {
  const root = document.querySelector('[data-visitor-map]');
  if (root) setupMap(root);
}
