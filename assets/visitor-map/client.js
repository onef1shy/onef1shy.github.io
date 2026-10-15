import config from './config.js';

export function isLocalHostname(hostname) {
  return ['localhost', '127.0.0.1', '[::1]', '::1'].includes(hostname);
}

export function apiURL(path) {
  if (!config.apiBase) return null;
  try {
    const base = new URL(config.apiBase);
    if (base.username || base.password || base.pathname !== '/' || base.search || base.hash) return null;
    if (base.protocol !== 'https:' && !(base.protocol === 'http:' && isLocalHostname(base.hostname))) return null;
    return new URL(path, base).href;
  } catch {
    return null;
  }
}

export async function requestJSON(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(url, {
      credentials: 'omit', referrerPolicy: 'no-referrer',
      ...options, signal: controller.signal,
    });
    if (!response.ok) {
      const error = new Error('Visitor statistics request failed');
      error.status = response.status;
      throw error;
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}
