import config from './config.js';
import { isLocalHostname } from './client.js';

export function mayTrack() {
  if (typeof location === 'undefined') return false;
  if (config.respectDoNotTrack &&
      (navigator.doNotTrack === '1' || navigator.globalPrivacyControl === true)) return false;
  if (isLocalHostname(location.hostname)) return config.trackLocalhost === true;
  return config.trackOrigins.includes(location.origin);
}

export const visit = {
  eventId: mayTrack() && globalThis.crypto?.randomUUID ? crypto.randomUUID() : null,
  result: null,
};

export function beginVisit() {
  visit.eventId = mayTrack() && globalThis.crypto?.randomUUID ? crypto.randomUUID() : null;
  visit.result = null;
  window.dispatchEvent(new Event('visitor-map:visit'));
}
