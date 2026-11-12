# Frontend hosting: Cloudflare Pages

Status: built and deployed 2026-11-12. See [decision 0009](../decisions/0009-frontend-on-cloudflare-pages.md).

The frontend is served by the Cloudflare Pages project `gridiron-oracle` in the owner's **personal** Cloudflare account (`72b23948192d06b3cc23f218e381fea9`). Never use the Stackd team account. The backend stays on Railway.

- Production URL: `https://gridiron.risshabh.com` (custom domain), plus `https://gridiron-oracle.pages.dev`.
- Vercel (`gridironoracle.vercel.app`) is **still live** until the owner retires it. `frontend/api/` and `frontend/vercel.json` stay in the repo until then.

## How it works

| Piece | File |
|---|---|
| Project config | `frontend/wrangler.toml` (`pages_build_output_dir = "dist"`, `BACKEND_URL` in `[vars]`) |
| `/api/*` proxy | `frontend/functions/api/[[path]].js`. Maps `/api/x/y?q` to `BACKEND_URL/x/y?q`, adds `x-api-key` from the `BACKEND_API_KEY` secret, drops a client-sent key, and strips hop-by-hop and `cf-*` headers. Has a 60 s timeout (504), and returns 502 when the backend is unreachable. |
| Access policy | `frontend/api/proxy-policy.js`, shared with the Vercel proxy. Settings writes are 403, hidden routes (`docs`, `openapi.json`, `integrity`, ...) are 404, and POST is allowed only to `scenario` and `mock-draft/pick`. |
| SPA fallback | Built in: Pages serves `index.html` for unknown paths because the build has no `404.html`. Do not add one. |
| Tests | `frontend/src/test/pages-proxy.test.tsx`, `proxy-policy.test.tsx` |

Secrets (Pages, set by the owner, never in git): `BACKEND_API_KEY` (= Railway `API_KEY`). A Pages secret takes effect on the next deployment.

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

## Current production deploy (2026-11-12)

Production runs the **live-parity build**, branch `agent/cloudflare-live`: commit `989101b` (what Vercel and Railway run), plus the Pages Function, this branch's `proxy-policy.js`, and `functions/api/odds.js`, a port of the old Vercel odds handler. That backend has no `/odds` route. `ODDS_API_KEY` is an optional Pages secret; without it, odds show mock lines.

When `main` is pushed and Railway runs it, redeploy from `main`, which has no odds function: `/odds` is a backend route there. Drop `agent/cloudflare-live` after that.
