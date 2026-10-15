// Run explicitly after approving a Cloudflare deployment. Values go through
// stdin to Wrangler, never through process arguments or console output.
import { readFile } from 'node:fs/promises';
import { parseEnv } from 'node:util';
import { spawnSync } from 'node:child_process';

try {
  const local = parseEnv(await readFile(new URL('../.dev.vars', import.meta.url), 'utf8'));
  const IPINFO_TOKEN = local.IPINFO_TOKEN?.trim();
  const HMAC_SECRET = local.HMAC_SECRET;
  if (!IPINFO_TOKEN || !HMAC_SECRET || HMAC_SECRET.length < 32) throw new Error('missing_secrets');
  const result = spawnSync(process.execPath, [
    new URL('../../node_modules/wrangler/bin/wrangler.js', import.meta.url).pathname,
    'secret', 'bulk', '--config', 'analytics/wrangler.jsonc',
  ], {
    cwd: new URL('../../', import.meta.url),
    input: JSON.stringify({ IPINFO_TOKEN, HMAC_SECRET }),
    encoding: 'utf8', stdio: ['pipe', 'inherit', 'inherit'],
  });
  process.exitCode = result.status ?? 1;
} catch {
  console.error('Configure both secrets in analytics/.dev.vars first. No secret values were printed.');
  process.exitCode = 1;
}
