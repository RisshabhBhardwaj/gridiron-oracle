/**
 * Server-side proxy: browser → Vercel → Railway FastAPI.
 *
 * Why a function rather than a plain vercel.json rewrite to Railway: the
 * backend is guarded by X-API-Key, and a rewrite cannot attach a header.
 * Injecting it here keeps the key in Vercel's server environment and out of
 * the client bundle, so Railway stays closed to unauthenticated traffic.
 *
 * Why the path arrives as ?__path= instead of a [...path] filename: Vercel's
 * zero-config /api directory is not Next.js and does not implement catch-all
 * filename routing. `api/[...path].js` matched only single-segment requests and
 * never populated req.query.path, so every call silently proxied to the backend
 * root — /api/backtest returned the API index message with a 200. The rewrite in
 * vercel.json is the contract instead; this file makes no assumption about its
 * own name.
 *
 * Path contract mirrors the Vite dev proxy, which strips the prefix:
 *   browser /api/draft/board  →  Railway /draft/board
 *
 * Env (Vercel project settings):
 *   BACKEND_URL        https://<service>.up.railway.app   (no trailing slash)
 *   BACKEND_API_KEY    same value as API_KEY on Railway
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

  // __path carries the segments after /api/, injected by the vercel.json
  // rewrite. Everything else in req.query is real caller query state.
  const { __path, ...query } = req.query || {};
  const rawPath = Array.isArray(__path) ? __path.join('/') : (__path || '');
  const segments = String(rawPath).split('/').filter(Boolean);

  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    for (const v of Array.isArray(value) ? value : [value]) search.append(key, v);
  }
  const qs = search.toString();
  const target = `${backend.replace(/\/$/, '')}/${segments.join('/')}${qs ? `?${qs}` : ''}`;

  const headers = {};
  for (const [key, value] of Object.entries(req.headers || {})) {
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
