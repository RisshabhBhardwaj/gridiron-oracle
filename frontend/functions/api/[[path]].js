/**
 * Cloudflare Pages Function: browser → Cloudflare Pages → Railway FastAPI.
 * Port of the Vercel proxy (frontend/api/proxy.js); behaviour is the same.
 *
 * Why a function rather than a plain redirect or rewrite: the backend is
 * guarded by X-API-Key, and a rewrite cannot attach a header. Injecting it
 * here keeps the key in Cloudflare's server environment (a Pages secret) and
 * out of the client bundle, so Railway stays closed to unauthenticated traffic.
 *
 * Path contract (mirrors the Vite dev proxy, which strips the prefix):
 *   browser /api/draft/board?week=3  →  ${BACKEND_URL}/draft/board?week=3
 *   browser /api                     →  ${BACKEND_URL}/
 * The `[[path]]` catch-all matches zero or more segments, so `/api` is handled
 * too. Segments are taken from the raw, still percent-encoded URL pathname, not
 * from context.params (which may be decoded), so the policy sees exactly what
 * the backend would route on.
 *
 * Which methods and paths are forwarded is decided by the policy shared with
 * Vercel (frontend/api/proxy-policy.js, imported unchanged) before any upstream
 * fetch. The proxy attaches no admin key, so settings writes cannot be made
 * through the public site.
 *
 * SPA fallback: Pages serves index.html for unknown paths only when the build
 * output has no 404.html. Checked 2026-11-12: frontend/public/ does not exist
 * and the Vite build emits no 404.html. Do not add one, or deep links such as
 * /player/123 would return 404 instead of the app.
 *
 * Env (wrangler.toml [vars] and Pages secrets):
 *   BACKEND_URL        https://<service>.up.railway.app   (trailing slash is stripped)
 *   BACKEND_API_KEY    same value as API_KEY on Railway   (Pages secret)
 */

import { decide } from '../../api/proxy-policy.js';

const UPSTREAM_TIMEOUT_MS = 60_000;

const HOP_BY_HOP = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'content-length',
]);

/** Headers Cloudflare adds to the incoming request; they mean nothing upstream. */
const isCloudflareHeader = (name) => name.startsWith('cf-') || name === 'cdn-loop';

/** @param {string} detail @param {number} status */
const json = (detail, status) =>
  new Response(JSON.stringify({ detail }), {
    status,
    headers: { 'content-type': 'application/json' },
  });

/**
 * @param {{ request: Request, env: { BACKEND_URL?: string, BACKEND_API_KEY?: string } }} context
 * @returns {Promise<Response>}
 */
export async function onRequest({ request, env }) {
  const backend = env.BACKEND_URL;
  if (!backend) return json('BACKEND_URL is not configured', 500);

  const url = new URL(request.url);
  const segments = url.pathname.replace(/^\/api(?=\/|$)/, '').split('/').filter(Boolean);

  const verdict = decide(request.method, segments);
  if (!verdict.allow) return json(verdict.detail, verdict.status);

  const target = `${backend.replace(/\/$/, '')}/${segments.join('/')}${url.search}`;

  const headers = new Headers();
  for (const [key, value] of request.headers) {
    const name = key.toLowerCase();
    if (HOP_BY_HOP.has(name) || isCloudflareHeader(name) || name === 'x-api-key') continue;
    headers.set(key, value);
  }
  if (env.BACKEND_API_KEY) headers.set('x-api-key', env.BACKEND_API_KEY);

  let body;
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    const raw = await request.arrayBuffer();
    if (raw.byteLength > 0) body = raw;
    else if (request.method === 'POST') body = '{}';
    if (body !== undefined && !headers.has('content-type')) {
      headers.set('content-type', 'application/json');
    }
  }

  let upstream;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
      return json(`Upstream timed out after ${UPSTREAM_TIMEOUT_MS / 1000}s`, 504);
    }
    return json(`Upstream unreachable: ${err?.message ?? 'unknown error'}`, 502);
  }

  const out = new Headers();
  upstream.headers.forEach((value, key) => {
    const name = key.toLowerCase();
    if (!HOP_BY_HOP.has(name) && name !== 'content-encoding') out.append(key, value);
  });
  return new Response(upstream.body, { status: upstream.status, headers: out });
}
