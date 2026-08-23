/**
 * Server-side proxy: browser → Vercel → Railway FastAPI.
 *
 * Why a function rather than a plain vercel.json rewrite: the backend is
 * guarded by X-API-Key, and a rewrite cannot attach a header. Injecting it
 * here keeps the key in Vercel's server environment and out of the client
 * bundle, so the Railway service can stay closed to unauthenticated traffic.
 *
 * Path contract mirrors the Vite dev proxy, which strips the prefix:
 *   browser /api/draft/board  →  Railway /draft/board
 *
 * Env (Vercel project settings):
 *   BACKEND_URL      https://<service>.up.railway.app   (no trailing slash)
 *   BACKEND_API_KEY  same value as API_KEY on Railway
 *   BACKEND_ADMIN_KEY  same value as ADMIN_API_KEY on Railway (optional)
 */

const HOP_BY_HOP = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length',
]);

export default async function handler(req, res) {
  const backend = process.env.BACKEND_URL;
  if (!backend) {
    res.status(500).json({ detail: 'BACKEND_URL is not configured' });
    return;
  }

  // req.query.path holds the catch-all segments; everything else is real query.
  const { path, ...query } = req.query;
  const segments = Array.isArray(path) ? path : [path].filter(Boolean);
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    for (const v of Array.isArray(value) ? value : [value]) search.append(key, v);
  }
  const qs = search.toString();
  const target = `${backend.replace(/\/$/, '')}/${segments.join('/')}${qs ? `?${qs}` : ''}`;

  const headers = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers[key] = value;
  }
  if (process.env.BACKEND_API_KEY) headers['x-api-key'] = process.env.BACKEND_API_KEY;
  if (process.env.BACKEND_ADMIN_KEY) headers['x-admin-key'] = process.env.BACKEND_ADMIN_KEY;

  let body;
  if (!['GET', 'HEAD'].includes(req.method)) {
    body = typeof req.body === 'string' || Buffer.isBuffer(req.body)
      ? req.body
      : JSON.stringify(req.body ?? {});
    headers['content-type'] = headers['content-type'] || 'application/json';
  }

  let upstream;
  try {
    upstream = await fetch(target, { method: req.method, headers, body });
  } catch (err) {
    res.status(502).json({ detail: `Upstream unreachable: ${err.message}` });
    return;
  }

  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase()) && key.toLowerCase() !== 'content-encoding') {
      res.setHeader(key, value);
    }
  });
  res.status(upstream.status);
  res.send(Buffer.from(await upstream.arrayBuffer()));
}
