# Gridiron Oracle: current status

As of 2026-10-11. Checked by running the gates on the commit named below. Items marked unverified were not run.
Update this file at the end of every working session. See `_internal-docs/_platform/DOCS_STANDARD.md`.
Full handoff with branch map and resume steps: `.agent-work/gridiron/HANDOFF-2026-10-11.md` (local, not committed).

## State of `main`

`main` is at the integrated head `82517f0` (local, fast-forwarded from `integration/gor-2026-10-10-r3`, now deleted). It holds the ML known-bug fixes, refresh correctness, the frontend design system, the audit blocker fixes, the forward-row target-game context, and the owner decisions record (0008). **Nothing is pushed.** `main` is ahead of `origin/main`. Pushing deploys production: Railway services `api` and `weekly-refresh` auto-deploy from `main`.

## What works (verified on 82517f0, 2026-10-11)

| Check | Result |
|---|---|
| Backend, CI-equivalent (no DB): pytest with coverage | 2144 passed, 31 skipped, coverage 81.09% (exit 0) |
| Ruff | clean |
| Radon rank F in CI scope (`backend/app ml pipeline`) | none |
| Frontend typecheck, tests, build | typecheck clean, 13 files / 77 tests pass, build OK |
| Gitleaks, `main..HEAD` | no leaks |
| Evidence manifest and serving manifest | OK (32 pinned cells match SHA-256) |
| Engine (C++) | unchanged since 2026-10-10; ctest not rerun on this head |
| Backend `-m integration` against local Postgres 18 | 96 passed, 15 skipped (no 2026 serving data locally), exit 0 (earlier run on the candidate) |
| CI on GitHub | **not run.** No branch has been pushed. |

Local DB note: with the local Postgres up (`gor-pg18`), 7 failed and 2 errors in data-dependent tests also fail on `989e221`. The local DB has no games, pbp or 2026 rows. Not caused by these changes.

## Built, on `main`

- Audit blocker fixes: the API-key middleware checks the routed path (a crafted Host header could bypass it; reproduced, now 401); the materializer no longer needs `.git` (uses the platform commit variable, else `unknown`); the Vercel odds handler refuses non-GET.
- Learner disclosure: `ml/served_learner.py` discloses the pinned `causal_20260821_fixed` (was a stale set).
- Forward rows take lines, rest, venue and lines-derived script columns from the target game (decision 0006, median fallback now superseded).
- Refresh correctness and known ML bug fixes from the candidate (see git log).
- Public Settings read-only (decision 0007); Base UI primitives (decision 0004).
- Production data: a read-only pg_dump was restored locally as database `gor_prod_20261010` in `gor-pg18`. It is a cut: `games` 2019 to 2026 (2232 rows, 193 with null total line), `pbp_plays` 2019 to 2026, `game_logs`, `feature_matrix`, `depth_charts` 2024 to 2026 only, `projections` 2024 to 2025, `team_game_predictions` 2026. Training history for 2019 to 2025 is in `ml/cache/*.parquet`.

## Unfinished work (committed WIP, NOT gated, NOT merged)

These branches hold partial work. Each has a worktree-free resume path: `git worktree add /private/tmp/gor-wt/<name> <branch>`.

| Branch | Head | What is there | What is missing |
|---|---|---|---|
| `agent/gor-ml-05-causal-fixes` | 286e7b8 | C1: Elo implied probability from pregame snapshot (52e8cde). C2: enrich_elo capped at LAST_COMPLETE_SEASON (3321b21). WIP: OOF/tft/utils changes. | Gate. Run-id uniqueness is only partly done (tft/utils WIP; lgbm, xgb and stacking not confirmed). Optuna, conformal and NaN-target fixes approved but not done. |
| `agent/gor-fwd-06-prior-games` | 234bae5 | Forward rows built from real prior games (WIP in `build_forward_week`). | Test file rewrite and gate. Decision 0006 rule still applies to lines. |
| `agent/gor-ops-02` | 8e04b0b | Q3 guard refuses forward builds for played weeks (c656de2). Weekly enrichment scoped to its season (26c68e3). WIP: H-4 source-failure handling in `normalize.py`, `nflreadpy_adapter.py`, new test. | H-3, H-5 (partly), H-6 check, C-1 forward-week acceptance gate. Gate. |
| `agent/gor-weights-08-wire` | f0346a9 | `ml/bucket_weights.py` only. | Wiring to the API and UI, native-model export (needs the rerun), tests. |

Not started (no commits): the line snapshot table (decision 0008 item 3); the security stream items A-03 (remove `/alerts/ws`), A-04 (non-root container), A-05 (pinned requirements), A-06 (localhost-only ports, Postgres 18 in `infra/`); the odds cutover to `odds.py` with a budget guard; Tailwind 4 and the remaining hand-rolled controls.

## Next steps (priority order)

1. Gate and merge the four WIP branches above, one at a time, each with the full gate. Resolve conflicts in `build_forward_week` (fwd-06 and ops-02 both touch it).
2. Implement the approved line-snapshot design (decision 0008, item 3), replacing the median fallback in `pipeline/feature_engineer.py`.
3. Implement the three approved training fixes: Optuna validation on an earlier season (`ml/catboost_model.py` ~L307-312); conformal intervals on prior seasons only (`ml/train.py` ~L1177); NaN fumble target (`pipeline/feature_engineer.py` ~L322).
4. Run the Elo rerun and refreeze (owner-approved). Inputs: `ml/cache/*.parquet` for 2019 to 2025 and the `games` table. Then `make freeze-baseline`. Long job: give the owner the command or run in background with a log.
5. Wire the Settings weights (decision 0008, item 6): native-model export during the rerun, a group-weight layer with identity at default weights, and the `fantasy_scoring` and `kelly_fraction` mappings.
6. Remaining security items (A-03 to A-06) and the odds cutover with a budget guard (decision 0008, item 9).
7. Frontend: Tailwind 4, remaining hand-rolled controls, copy and error-message fixes (decision 0008, item 13).
8. Push only when you approve it. Then turn on branch protection with the required checks, after GitHub CI is green on the pushed head.

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

## Migration-ready pass (2026-10-17, branch `agent/gor-migration-ready`, base `main` d8cd04a)

Not on `main`, not pushed. Non-root `USER` + dropped build tools in `backend/Dockerfile`; pinned `backend/requirements.txt` with test/dev split (`requirements-test.txt`, `requirements-dev.txt`); compose ports bound to `127.0.0.1`, Postgres 18; new `docs/MIGRATION.md` runbook; `.env.example` completed. Gate green on the branch: 2144 passed, 31 skipped; ruff clean; evidence/serving manifests OK; `docker compose config` validates. Image build and `make up` not executed (no cached base layers; not pulling GBs on a tight disk). Remaining `/Users/...` paths are pinned artifacts (`releases/*.json`, `ml/onnx/*.onnx`) — re-freeze only.

## Decisions

- [0001 Causal walk-forward evaluation](../decisions/0001-causal-walk-forward-evaluation.md)
- [0002 Artifact-backed, fail-closed serving](../decisions/0002-artifact-backed-fail-closed-serving.md)
- [0003 Betting and Kelly sizing stay in scope](../decisions/0003-betting-and-kelly-sizing-in-scope.md)
- [0004 shadcn on Base UI](../decisions/0004-shadcn-on-base-ui.md)
- [0005 SP0-SP4 rebuild plan retired; built pieces kept](../decisions/0005-sp0-sp2-plan-retired.md)
- [0006 Forward rows take game context from the target game](../decisions/0006-forward-rows-take-target-game-context.md) (median fallback superseded)
- [0007 Public Settings are read-only](../decisions/0007-public-settings-read-only.md)
- [0008 Owner decisions of 2026-10-10 and their status](../decisions/0008-owner-decisions-2026-10-10.md)
