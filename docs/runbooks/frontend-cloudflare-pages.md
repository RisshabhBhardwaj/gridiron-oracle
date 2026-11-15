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

## Current production deploy (2026-11-15)

Production runs `main` at `9bcbd6e` (Pages deploy `b2469f78`, 2026-11-15 ~01:10 UTC, by the manager with the owner's yes relayed by the coordinator). It adds UI phase 2 items 4-7 (player page, draft board, one error pattern and loading states, projections table and accessibility) to the build below. The box api was redeployed from the same commit just before Pages (`/explain` now answers 404 `{explanation_available: false}` instead of 503 when no breakdown is published; rollback image tag `pre-explain404-20261115T0101Z` on the api image). `9bcbd6e` is deliberately older than `main`: the forward producer (`de79477`) needs migrations 0028-0030 on the box first.

Verified live 2026-11-15 at 1440 and 390 px: `/`, `/season`, `/draft`, `/backtest` and a not-found page have 0 console errors and no sideways scroll. `/projections` and the player page log one browser network 404 each, the expected "no forecast for this week" answer from `/api/team-games` and `/api/predict` (no forward-week rows until the forward producer runs on the box); both pages show the calm "No forecast yet" note with next steps. The explain panel can't be seen in the UI until a week has a forecast; the api 404 was checked directly.

Previous production deploys: `549ff5d3` (`main` `ffcd5d3`, 2026-11-14), `d77a1738` (`main` `49c2706` + `2d9d372`, 2026-11-12), before that the live-parity branch `agent/cloudflare-live` (`cfb46f9`), now superseded and due for retirement.

Keep the frontend and api in step: when an api change alters a response shape (for example a re-freeze that adds a pinned group), deploy the frontend in the same window. Rollback: redeploy the previous deployment from the Pages dashboard, or rebuild the previous commit with the command above.
