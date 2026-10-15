// Provider-specific code stays here so Legacy can be replaced without changing
// tracking, storage, or the public stats format. No Lite fallback is used.
export function parseLegacyLocation(data) {
  if (!data || data.bogon || typeof data.city !== 'string' || !data.city.trim()) return null;
  if (typeof data.country !== 'string' || !/^[A-Z]{2}$/.test(data.country)) return null;
  if (typeof data.loc !== 'string') return null;
  const parts = data.loc.split(',');
  if (parts.length !== 2 || parts.some(part => !part.trim())) return null;
  const [latitude, longitude] = parts.map(Number);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  if (Math.abs(latitude) > 90 || Math.abs(longitude) > 180) return null;
  return {
    countryCode: data.country,
    city: data.city.trim().slice(0, 160),
    region: typeof data.region === 'string' ? data.region.trim().slice(0, 160) : '',
    latitude,
    longitude,
  };
}

export async function lookupLegacyLocation(ip, token, fetcher = fetch, onFailure = () => {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3500);
  try {
    const response = await fetcher(`https://ipinfo.io/${encodeURIComponent(ip)}/json`, {
      headers: { Authorization: `Bearer ${token.trim()}`, Accept: 'application/json' },
      signal: controller.signal,
      redirect: 'manual',
    });
    if (!response.ok) { onFailure(`http_${response.status}`); return null; }
    const location = parseLegacyLocation(await response.json());
    if (!location) onFailure('missing_legacy_fields');
    return location;
  } catch (error) {
    onFailure(error?.name === 'AbortError' ? 'timeout' : 'network_error');
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
