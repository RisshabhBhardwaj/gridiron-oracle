# Gridiron Oracle: current status

As of 2026-11-12 (Cloudflare section); the rest as of 2026-10-17. Checked by running the gates on the commit named below. Items marked unverified were not run.
Update this file at the end of every working session. See `_internal-docs/_platform/DOCS_STANDARD.md`.
Full handoff with branch map and resume steps: `.agent-work/gridiron/HANDOFF-2026-10-11.md` (local, not committed).

## Frontend on Cloudflare Pages (2026-11-12)

- Live at `https://gridiron.risshabh.com` and `https://gridiron-oracle.pages.dev` (Pages project `gridiron-oracle`, owner's personal Cloudflare account). Runbook: `docs/runbooks/frontend-cloudflare-pages.md`. Decision: [0009](../decisions/0009-frontend-on-cloudflare-pages.md).
- Production is the live-parity build: branch `agent/cloudflare-live` at `cfb46f9`, which is `989101b` + Pages Function + odds port + calm backend-down state. The `main`-based work is on `agent/cloudflare-migration`. Neither branch is pushed or merged.
- **The backend is offline.** The Railway service is gone (Railway 404 "Application not found") while the owner moves to the OVH VPS. Visitors see the pages and one calm line, "Live data is moving to a new server and will be back soon". Vercel (`gridironoracle.vercel.app`) is still live, but its `/api` is broken the same way, until the owner retires it.
- Owner steps once the VPS is up: set `BACKEND_URL` in `frontend/wrangler.toml` and redeploy; `wrangler pages secret put BACKEND_API_KEY --project-name gridiron-oracle` with the new VPS `API_KEY`; `ODDS_API_KEY` is optional.

## State of `main`

`main` is at the integrated head `6bc2c3d` (local merges of six worker branches, 2026-10-17; see below). **Nothing is pushed.** `main` is ahead of `origin/main`. Pushing deploys production: Railway services `api` and `weekly-refresh` auto-deploy from `main`. No training was run this session; the rerun and re-freeze are deferred to the future VPS (see `docs/MIGRATION.md`).

## What works (verified on 6bc2c3d, 2026-10-17, by the manager)

| Check | Result |
|---|---|
| Backend, CI-equivalent (no DB) | 2306 passed, 30 skipped, 123 deselected (exit 0) |
| Ruff | clean |
| Radon rank F in CI scope (`backend/app ml pipeline`) | none |
| Frontend typecheck, tests, build | typecheck clean, 95 tests pass, build OK (Tailwind 4) |
| Evidence manifest and serving manifest | OK (32 pinned cells match SHA-256) |
| Engine (C++) | ctest 72/72 pass on this head |
| Backend `-m integration` against local Postgres 18 | not rerun this session |
| CI on GitHub | **not run.** No branch has been pushed. |
| Docker image build / `make up` | not executed (no cached base layers; disk tight) — compose config validates |

One Hypothesis deadline flake was seen under parallel load (`test_stacking_ensemble` temporal-ordering, 200 ms deadline); the same file passes 57/57 in isolation on both `main` and the branch. Not a product failure.

Local DB note: with the local Postgres up (`gor-pg18`), 7 failed and 2 errors in data-dependent tests also fail on `989e221`. The local DB has no games, pbp or 2026 rows. Not caused by these changes.

## Built, on `main`

- Audit blocker fixes: the API-key middleware checks the routed path (a crafted Host header could bypass it; reproduced, now 401); the materializer no longer needs `.git` (uses the platform commit variable, else `unknown`); the Vercel odds handler refuses non-GET.
- Learner disclosure: `ml/served_learner.py` discloses the pinned `causal_20260821_fixed` (was a stale set).
- Forward rows take lines, rest, venue and lines-derived script columns from the target game (decision 0006, median fallback now superseded).
- Refresh correctness and known ML bug fixes from the candidate (see git log).
- Public Settings read-only (decision 0007); Base UI primitives (decision 0004).
- Production data: a read-only pg_dump was restored locally as database `gor_prod_20261010` in `gor-pg18`. It is a cut: `games` 2019 to 2026 (2232 rows, 193 with null total line), `pbp_plays` 2019 to 2026, `game_logs`, `feature_matrix`, `depth_charts` 2024 to 2026 only, `projections` 2024 to 2025, `team_game_predictions` 2026. Training history for 2019 to 2025 is in `ml/cache/*.parquet`.

## Landed 2026-10-17 (six worker branches, each gated by the manager, merged locally)

- `gor-model-fixes` (3524bfc, incl. ml-05 C1/C2): unique OOF run ids; feature contract fails closed on unknown names; Optuna tunes on an earlier season; conformal intervals on prior seasons only; NaN fumble target aligned with backtest scoring; `summarize_causal_evals.py` rebuilt to fail loudly with no inputs (the 74/75 count stays unreproduced until a rerun); serving leakage regression test.
- `gor-line-snapshots` (efeb5ee): `game_line_snapshots` table + migration 0026, `get_line_as_of`, Elo-implied fallback replacing 45.0/2.5. Migration rendered offline only — run `alembic upgrade head` on a scratch DB before the rerun, and on production DBs before normalize runs.
- `gor-migration-ready` (213fae0): non-root Dockerfile, pinned/split requirements, localhost-bound Postgres 18 compose, `.env.example`, `docs/MIGRATION.md`.
- `gor-forward-ops` (bfa9ffb; absorbs fwd-06 + ops-02): forward rows from real prior games cut at as-of, played-week guard, season-scoped enrichment, ingest failure handling, forward-week structural acceptance gate. Residual: `summary.errors > 0` still does not fail the normalize step (follow-up decision).
- `gor-weights-wire` (6050c44): bucket weights wired settings -> service -> labelled API; identity at defaults proven. Non-default bucket weights do not move served numbers until a rerun stores per-row bucket attributions (hook ready in `weighting.py`); `fantasy_scoring` and the `engine_exposure_cap` -> `kelly_fraction` mapping work today.
- `gor-frontend-odds` (42fc33c..0c16b21): backend odds quota guard (`ODDS_MIN_REMAINING`), Vercel handler retired, `/alerts/ws` removed (A-03), Tailwind 4, all hand-rolled controls replaced with primitives, plain-language copy. No pixel-level visual check was run.

The old WIP branches (`gor-ml-05-causal-fixes`, `gor-fwd-06-prior-games`, `gor-ops-02`, `gor-weights-08-wire`) are superseded by the merges above.

## Next steps (priority order)

1. On the future VPS: copy `ml/cache/*.parquet` (2019-2025, gitignored) + the games data, run `alembic upgrade head` on a scratch DB, then the walk-forward rerun, `make freeze-baseline`, per-cell causal evals, and `scripts/summarize_causal_evals.py` to regenerate the headline count honestly. Full recipe: `docs/MIGRATION.md`.
2. During that rerun, store per-row bucket attributions so user bucket weights move served numbers; also verify the forward-week structural gate against the real DB.
3. Owner-only migration steps (VPS order, DNS, secrets, data move, old-host teardown): checklist in `docs/MIGRATION.md`.
4. Push only when you approve it. Then turn on branch protection with the required checks, after GitHub CI is green on the pushed head.
5. Follow-ups: normalize `summary.errors` failure decision; odds quota guard needs a shared cache before multi-replica serving.

## To-dos and reminders

- Keep `docs/DATA_SOURCES.md` documenting `load_pfr_advstats` (a test enforces it).
- Evidence entries in `releases/artifacts/MANIFEST.json` are "provisional"; flip to frozen after a clean rebuild reproduces them.
- The baseline JSON and nine baseline files still contain absolute `/Users/...` paths. Fix them only by re-freezing with the repo-relative writer (`scripts/freeze_baseline.py`), never by hand.
- `scripts/generate_sleeper_draft_profiles.py` `build_profiles` is rank F (CC > 40). Outside the CI gate, which scans `backend/app ml pipeline` only.
- The Settings page is view-only unless the build sets `VITE_SETTINGS_EDITABLE=true`. The public Vercel build must not set it.
- The Sleeper league this app models is "La Hospitale" (8 teams). Its profile data is in the restored `sleeper_league_*` tables.

## Open questions

- K and DEF are in the league's starting lineup but not modelled (out of scope by design). Confirm the owner wants that to stay.
- Is `verify_db.py` wanted? It was archived on 2026-10-07 as unreferenced.

## Migration readiness (2026-10-17)

On `main` at `6bc2c3d`, not pushed. The repo side of the VPS move is done: portable paths (remaining `/Users/...` hits are pinned artifacts — re-freeze only), hardened runtime, pinned deps, localhost-bound Postgres 18 compose, completed `.env.example`, and the runbook in `docs/MIGRATION.md` (bootstrap, secret names, pg_dump/pg_restore, OWNER-ONLY DNS cutover, rollback, VPS training recipe). Image build and `make up` were not executed (no cached base layers on a tight disk); compose config validates.

## Decisions

- [0001 Causal walk-forward evaluation](../decisions/0001-causal-walk-forward-evaluation.md)
- [0002 Artifact-backed, fail-closed serving](../decisions/0002-artifact-backed-fail-closed-serving.md)
- [0003 Betting and Kelly sizing stay in scope](../decisions/0003-betting-and-kelly-sizing-in-scope.md)
- [0004 shadcn on Base UI](../decisions/0004-shadcn-on-base-ui.md)
- [0009 Frontend on Cloudflare Pages](../decisions/0009-frontend-on-cloudflare-pages.md)
- [0005 SP0-SP4 rebuild plan retired; built pieces kept](../decisions/0005-sp0-sp2-plan-retired.md)
- [0006 Forward rows take game context from the target game](../decisions/0006-forward-rows-take-target-game-context.md) (median fallback superseded)
- [0007 Public Settings are read-only](../decisions/0007-public-settings-read-only.md)
- [0008 Owner decisions of 2026-10-10 and their status](../decisions/0008-owner-decisions-2026-10-10.md)
