# Frontend hosting: Cloudflare Pages

Status: built and deployed 2026-11-12. See [decision 0009](../decisions/0009-frontend-on-cloudflare-pages.md).

The frontend is served by the Cloudflare Pages project `gridiron-oracle` in the owner's **personal** Cloudflare account (`72b23948192d06b3cc23f218e381fea9`). Never use the Stackd team account. The backend stays on Railway.

- Production URL: `https://gridiron.risshabh.com` (custom domain), plus `https://gridiron-oracle.pages.dev`.
- Vercel (`gridironoracle.vercel.app`) is **still live** until the owner retires it. `frontend/api/` and `frontend/vercel.json` stay in the repo until then.

## How it works

| Piece | File |
|---|---|
| Project config | `frontend/wrangler.toml` (`pages_build_output_dir = "dist"`, `BACKEND_URL` in `[vars]`) |
| `/api/*` proxy | `frontend/functions/api/[[path]].js`. Maps `/api/x/y?q` to `BACKEND_URL/x/y?q`, adds `x-api-key` from the `BACKEND_API_KEY` secret, drops a client-sent key, and strips hop-by-hop and `cf-*` headers. When the backend is unreachable, times out (60 s), or its host answers without the app (Railway `x-railway-fallback`, or an edge 502/504 or non-JSON 503), it returns **503** `{detail, code: "backend_unavailable", reason}` with `Retry-After: 300`. It is 503, not 502, because on the custom domain Cloudflare replaces a Function's 502 with its own plain page. |
| Access policy | `frontend/api/proxy-policy.js`, shared with the Vercel proxy. Settings writes are 403, hidden routes (`docs`, `openapi.json`, `integrity`, ...) are 404, and POST is allowed only to `scenario` and `mock-draft/pick`. |
| SPA fallback | Built in: Pages serves `index.html` for unknown paths because the build has no `404.html`. Do not add one. |
| Tests | `frontend/src/test/pages-proxy.test.tsx`, `proxy-policy.test.tsx` |

Secrets (Pages, set by the owner, never in git): `BACKEND_API_KEY` (= the backend's `API_KEY`). A Pages secret takes effect on the next deployment.

## Deploy

Deploys are manual (no Git integration). Deploy only a committed, clean tree whose frontend gate passed, and whose frontend matches the backend Railway is running.

```bash
cd frontend
npm ci && npm run typecheck && npm run test && npm run build
export CLOUDFLARE_ACCOUNT_ID=72b23948192d06b3cc23f218e381fea9
wrangler whoami   # must show the personal account
wrangler pages deploy --project-name gridiron-oracle --branch main \
  --commit-hash "$(git rev-parse --short HEAD)" --commit-dirty=false
```

`--branch main` publishes to production. Any other branch name gives a preview URL.

## Changing the backend (Railway → OVH VPS)

The backend address is one line: `BACKEND_URL` in `frontend/wrangler.toml` `[vars]` (https, no trailing slash). Change it, commit, and run the deploy above. If the new backend has a different `API_KEY`, also run `wrangler pages secret put BACKEND_API_KEY --project-name gridiron-oracle` before deploying. Nothing else in the frontend names the backend.

Rollback: Cloudflare dashboard → Workers & Pages → `gridiron-oracle` → Deployments → pick an earlier one → Rollback.

## Current production deploy (2026-11-14)

Production runs `main` at `ffcd5d3` (Pages deploy `549ff5d3`, by the coordinator with owner approval). The Pages Function proxies `/api/*` to `https://gridiron-api.risshabh.com` (Cloudflare Tunnel to the api on the OVH box). This build reads the served release `causal_20261114` for Backtest and Season Review, and includes UI phase 2 items 1-3.

Verified live 2026-11-14 by the coordinator: `/backtest` and `/season` render with 0 console errors at 1440 and 390 px; `/api/backtest` returns release `causal_20261114` with 158/160.

Previous production deploys: `d77a1738` (`main` `49c2706` + `2d9d372`, 2026-11-12), before that the live-parity branch `agent/cloudflare-live` (`cfb46f9`), now superseded and due for retirement.

Keep the frontend and api in step: when an api change alters a response shape (for example a re-freeze that adds a pinned group), deploy the frontend in the same window. Rollback: redeploy the previous deployment from the Pages dashboard, or rebuild the previous commit with the command above.
