import { readFile } from 'node:fs/promises';
import { parseEnv } from 'node:util';
import { parseLegacyLocation } from '../src/geolocation.js';

try {
  const secrets = parseEnv(await readFile(new URL('../.dev.vars', import.meta.url), 'utf8'));
  if (!secrets.IPINFO_TOKEN?.trim()) throw new Error('missing_token');
  const response = await fetch('https://ipinfo.io/8.8.8.8/json', {
    headers: { Authorization: `Bearer ${secrets.IPINFO_TOKEN.trim()}`, Accept: 'application/json' },
    signal: AbortSignal.timeout(12000),
    redirect: 'manual',
  });
  if (!response.ok) {
    console.error(`Legacy verification failed: HTTP ${response.status}. No secret values were printed.`);
    process.exitCode = 1;
  } else if (!parseLegacyLocation(await response.json())) {
    console.error('Legacy verification failed: valid city, country, and loc fields are not available. Lite is not supported.');
    process.exitCode = 1;
  } else {
    console.log('IPinfo Legacy verified: city, country, and loc are present and valid. Token was not printed.');
  }
} catch (error) {
  console.error(error.message === 'missing_token'
    ? 'Fill IPINFO_TOKEN in analytics/.dev.vars before running this command.'
    : 'Legacy verification could not complete. Check the network and the local secret file; no secret values were printed.');
  process.exitCode = 1;
}
