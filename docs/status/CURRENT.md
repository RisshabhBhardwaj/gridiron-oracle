# Gridiron Oracle: current status

As of 2026-11-14 (Cloudflare, rerun, `main` state and host sections); the verified-gate table and feature lists as of 2026-10-17. Checked by running the gates on the commit named below. Items marked unverified were not run.
Update this file at the end of every working session. See `_internal-docs/_platform/DOCS_STANDARD.md`.
Full handoff with branch map and resume steps: `.agent-work/gridiron/HANDOFF-2026-10-11.md` (local, not committed).

## Frontend on Cloudflare Pages (2026-11-12)

- Live at `https://gridiron.risshabh.com` and `https://gridiron-oracle.pages.dev` (Pages project `gridiron-oracle`, owner's personal Cloudflare account). Runbook: `docs/runbooks/frontend-cloudflare-pages.md`. Decision: [0009](../decisions/0009-frontend-on-cloudflare-pages.md).
- **LIVE on the VPS-4 (2026-11-12):** production is the `main` build (`49c2706` + `BACKEND_URL` commit `2d9d372`; Pages deploy `d77a1738`): Pages Function proxy → `https://gridiron-api.risshabh.com` (Cloudflare Tunnel to the shared OVH VPS-4 in BHS) → api on the box. Seed = restored `gor_prod_20261010` (2232 games, alembic 0026). Verified live by the manager: site 200, `/api/season/current` 200, season projections 200 (706 rows), `/api/odds` 200 configured with 25 events (events-endpoint + market-key translation fix), settings PUT 403, `/api/docs` 404, only 22/tcp open on the box. Host layout: `infra/hosts/vps1/` + `docs/runbooks/vps4-shared-host.md`; `/srv/HOST.md` on the box; ATS notified (its `/srv/ats` envelope reserved, not deployed by Gridiron).
- **Rerun done 2026-11-14 (decision [0011](../decisions/0011-causal-rerun-2026-11-14.md)):** release `causal_20261114` is served (frozen from C2 `226e0ec`, evidence C3 `c3dfc11`). `/health` reports `artifact_mode_ready: true`. Status reads `degraded` only because MLflow is not deployed on the box; that check is reported but not required. 158/160 cell-seasons beat both naive and trailing-3; 157/160 beat Marcel. Bucket weights now move served numbers, and `served_learner` is disclosed per cell. `feature_matrix` freshness lapses 24 h after the last ETL (about 2026-11-15 12:25 UTC), and then readiness drops to warn. Still open: healthchecks URLs (owner), `AGE_RECIPIENT` (owner), the forward-week producer (see Open questions), `agent/cloudflare-live` retirement. Vercel project `gridiron-oracle` was retired 2026-11-12.
- The old live-parity branch `agent/cloudflare-live` (`cfb46f9`, local only) is superseded by the `main` deploy above; retire it once `main` has soaked.

## Box B host (verified 2026-11-14 by the Claude manager)

- Manager changed 2026-11-13: Muse cancelled; the Claude Gridiron manager now owns Gridiron and the Box B host layer (`/srv/HOST.md`) for Gridiron, ATS and statuta. Host key matches the internal brief.
- Timers (2026-11-14 after the rerun): backup daily 23:00 UTC passes (uploads **unencrypted** because `AGE_RECIPIENT` is unset). `train-small` (daily 01:00) and `train-full` (Fri 22:00) are re-enabled with pinned recipes (`bash scripts/train_window.sh small|full`); neither promotes. **`gridiron-etl.timer` is stopped** pending a coordinator/owner call (decision 0011).
- ATS backup timer (21:30 UTC, owner-approved) **installed 2026-11-14** (ATS main 68a2137, runs as `deploy`, containerised pg_dump → age → rclone); first run uploaded and size-verified. `/srv/HOST.md` has the timer/owner table.
- Rerun 2026-11-14: production DB backfilled to 2019-2026 (feature_matrix 91,299 rows, Elo with 0 nulls), alembic 20261114_0027; 32-cell OOF rebuild took 51 min; serving gap 6 min 47 s. `weekly_refresh.py` passes steps 1-6 and stops at step 7 (no forward-week `projections` rows; decision 0011). Box dumps and rollback: `~/rerun_backup/`, `docs/runbooks/vps4-causal-rerun.md`.

## State of `main`

`main` is pushed (first push 2026-11-12, owner-approved, gitleaks-clean before every push). Release commit C3 `c3dfc11` is CI-green (run 43593398397, all 6 jobs) as of 2026-11-14; docs commits follow it. Pushing `main` deploys nothing: Railway is retired, Vercel is removed, and Pages is direct-upload (deploys are manual, see the runbook). The box checkout is synced by hand; sync only descendants of C3.

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
| CI on GitHub (2026-11-14) | **green on `d6c51c6`** (run 43559700365, all 6 jobs). Fixed: Alembic deps pinned, push gate allows `.env.example`, CPU torch at the pinned version, `CurrentSeason` week effect, GNU `mktemp` template in `e2e/lib/http_assert.sh`. Branch protection not set (owner decision). Nightly slow tests pass. |
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

1. Forward-week weekly projections: wire `build_forward_features.py` and `materialize_forward_projections.py` into `weekly_refresh.py` step 4, with the guards moved ahead of the `projections` write (owner decision; decision 0011). Until then, every in-season refresh stops at step 7.
2. Decide on the Tuesday ETL timer (stopped). Measure `weekly_refresh.py` peak memory against the 1.5 GB `etl` limit first. Today's run used the 10 GB training container.
3. UI review and polish pass (decision 0010 item 3) now that data is fresh.
4. Owner-only migration steps (VPS order, DNS, secrets, data move, old-host teardown): checklist in `docs/MIGRATION.md`.
5. Owner: turn on branch protection with the required checks (CI is green on `main`).
6. Follow-ups: normalize `summary.errors` failure decision; odds quota guard needs a shared cache before multi-replica serving.

## To-dos and reminders

- Old full local DB (PG16, as of 2026-08-27; 6 GB of it is re-downloadable `staging_nflreadpy`) is now only a backup: `~/Projects/Archive/backups/gridiron_oracle/infra_oracle_pg16_20260827.dump` (moved from `~/Projects/_backups/`; also copied to Box B 2026-11-14 and restored there as scratch DB `oracle_hist_20260827`) (530 MB, `pg_restore`-verified 2026-10-23). Its Docker volume was deleted. 
- Keep `docs/DATA_SOURCES.md` documenting `load_pfr_advstats` (a test enforces it).
- Evidence entries in `releases/artifacts/MANIFEST.json` are "provisional"; flip to frozen after a clean rebuild reproduces them.
- The baseline JSON and nine baseline files still contain absolute `/Users/...` paths. Fix them only by re-freezing with the repo-relative writer (`scripts/freeze_baseline.py`), never by hand.
- `scripts/generate_sleeper_draft_profiles.py` `build_profiles` is rank F (CC > 40). Outside the CI gate, which scans `backend/app ml pipeline` only.
- The Settings page is view-only unless the build sets `VITE_SETTINGS_EDITABLE=true`. The public Vercel build must not set it.
- The Sleeper league this app models is "La Hospitale" (8 teams). Its profile data is in the restored `sleeper_league_*` tables.

## Open questions

- Baseline re-freeze: **done 2026-11-14** (decisions 0010, 0011).
- Owner: wire the forward-week producer into the weekly refresh? Re-enable the Tuesday ETL timer now (it will stop at step 7 each week but keeps data fresh)?
- Owner: rotate the production DB password? It was printed into a session's tool output (not a file) by the Alembic `%` crash on 2026-11-14.
- Owner: create a healthchecks.io account and the four checks, then paste the ping URLs into the box `.env` (`HEALTHCHECKS_*_URL`).
- Owner: set `AGE_RECIPIENT` (an age public key) so Gridiron backups are encrypted.
- K and DEF are in the league's starting lineup but not modelled (out of scope by design). Confirm the owner wants that to stay.
- Is `verify_db.py` wanted? It was archived on 2026-10-07 as unreferenced.

## Migration readiness (2026-10-17)

Historical (2026-10-17, `6bc2c3d`); superseded by the live VPS-4 deploy above. The repo side of the VPS move is done: portable paths (remaining `/Users/...` hits are pinned artifacts — re-freeze only), hardened runtime, pinned deps, localhost-bound Postgres 18 compose, completed `.env.example`, and the runbook in `docs/MIGRATION.md` (bootstrap, secret names, pg_dump/pg_restore, OWNER-ONLY DNS cutover, rollback, VPS training recipe). Image build and `make up` were not executed (no cached base layers on a tight disk); compose config validates.

## Decisions

- [0001 Causal walk-forward evaluation](../decisions/0001-causal-walk-forward-evaluation.md)
- [0002 Artifact-backed, fail-closed serving](../decisions/0002-artifact-backed-fail-closed-serving.md)
- [0003 Betting and Kelly sizing stay in scope](../decisions/0003-betting-and-kelly-sizing-in-scope.md)
- [0004 shadcn on Base UI](../decisions/0004-shadcn-on-base-ui.md)
- [0009 Frontend on Cloudflare Pages](../decisions/0009-frontend-on-cloudflare-pages.md)
- [0010 Owner decisions of 2026-11-13: Claude takeover, data pull and rerun, UI review](../decisions/0010-owner-decisions-2026-11-13.md)
- [0011 Causal rerun of 2026-11-14: release causal_20261114 is served](../decisions/0011-causal-rerun-2026-11-14.md)
- [0005 SP0-SP4 rebuild plan retired; built pieces kept](../decisions/0005-sp0-sp2-plan-retired.md)
- [0006 Forward rows take game context from the target game](../decisions/0006-forward-rows-take-target-game-context.md) (median fallback superseded)
- [0007 Public Settings are read-only](../decisions/0007-public-settings-read-only.md)
- [0008 Owner decisions of 2026-10-10 and their status](../decisions/0008-owner-decisions-2026-10-10.md)
