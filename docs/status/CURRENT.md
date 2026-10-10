# Gridiron Oracle: current status

As of 2026-10-07 (written during the documentation reset; verified against the code on that date unless marked unverified). Frontend lines updated 2026-10-10 on branch `agent/gor-frontend-02-design-system` (typecheck, tests and build run that day).
Update this file at the end of every working session. See `_internal-docs/_platform/DOCS_STANDARD.md`.

## What works, what is partial, what is unverified

Checked on 2026-10-07 by running the tests, not by trusting older documents.

| Check | Result |
|---|---|
| Backend tests (offline subset) | 1305 passed, 0 failed, 27 skipped |
| Frontend typecheck, 77 tests, build | all pass (2026-10-10, branch `agent/gor-frontend-02-design-system`) |
| C++ engine, 72 tests | all pass |
| Lint (`ruff` 0.15.12) | clean |
| Evidence manifest check | passes; every entry is marked "provisional" |

**Model quality.**
- Owner-verified claim: on causal out-of-fold evaluation (15 stat/position cells, seasons 2021 to 2025), the model beat both the naive (last season) and trailing-3-games baselines in 74 of 75 cell-seasons. **Reproduction artifact not committed.** A per-cell eval CSV set for the current candidate is not in the repo.
- How to reproduce: for each cell, run `python -m ml.eval_causal --oof <pinned stack CSV from releases/current_baseline.json> --stat <stat> --position <pos> --out <csv>` (needs the database for baselines), then count rows where `beats_naive` and `beats_trailing3` are both true per cell-season. `scripts/summarize_causal_evals.py` does that aggregation but is hard-coded to older file names.
- History: in August 2026 an audit found a data leak; one report records the earlier headline falling from 74/75 to 15/75 after the fix, and the manifest labels all evidence "provisional". The 74/75 figure above is the owner-verified result on the rebuilt candidate.
- Committed, directly readable: `reports/marcel_vs_stack.json` (fantasy points only) has 16 scored position-season cells; the model beat naive and trailing-3 in all 16 and the Marcel baseline in 15.
- Against draft rankings (ADP), an earlier analysis (archived, not re-checked) found the model ranks players better across positions but worse within a position.

**Partly built:** season projections (an earlier note says ranges were too narrow); the draft board; the user settings page (saved weights have no consumer in the projection code that we found).

**Not verified:** live deployment health, the Railway cron refresh, the Bayesian/Monte Carlo layer on the served path, engine connection to any live odds feed, accuracy of the drive simulator.

**Known UI gaps (updated 2026-10-10):** fixed on the branch above, not yet merged: the "no forecast" banner uses the requested season and week; `Dashboard.tsx` and `SeasonProjections.tsx` are removed; the player page says "No line yet" instead of "Book Line 0" and no longer shows generated demo lines; Settings is view-only on the public site; opening player search no longer crashes the app; Current Season cards open the player page. Still open: the hand-rolled controls listed in `docs/DESIGN_SYSTEM.md`, jargon copy (`docs/USERS.md`), `ErrorBanner` still prints a developer command ("make up") and raw error text, `GameDetail` prints an internal game id, units on the player page are always "yds", and nothing consumes the saved Settings weights.


## In progress

None recorded.

## Next steps (priority order)

1. Commit a reproducer for the owner-verified 74/75 baseline-beating count: generate per-cell eval CSVs for the current candidate with `ml.eval_causal` and aggregate them. Update `scripts/summarize_causal_evals.py`, which is hard-coded to old `*_20260809.csv` file names and to `reports/eval_causal_stack_*.csv` files that no longer exist.
2. Decide the fate of the prediction-surfaces rebuild plan (SP0/SP2 labels still appear in code; the plan itself is archived). Season/rest-of-season endpoints were described as fail-closed until playing-time (SP2); current behaviour not re-checked.
3. Betting is in scope (decision 0003): verify the engine's live odds connection, show a real book line in the UI (it shows "No line yet" until a live line exists), and check that projection ranges are calibrated (an earlier note says season ranges were too narrow).
4. Done on the frontend branch (2026-10-10): the no-forecast banner takes the season and week from the request.
5. Wire or remove the Settings weights (`user_config.json`): no consumer was found in projection code.
6. Rewrite jargon UI copy (list in `docs/USERS.md`).
7. Continue the UI migration in `docs/DESIGN_SYSTEM.md` (approved 2026-10-10, decision 0004): the remaining hand-rolled controls are listed there with file and line.

## To-dos and reminders

- Keep `docs/DATA_SOURCES.md` documenting `load_pfr_advstats` (a test enforces it).
- Evidence entries in `releases/artifacts/MANIFEST.json` are all "provisional"; flip to frozen after a clean rebuild reproduces them.
- `ml/experiments/team_game_model/win_probability_2025_*.json` are unlisted in the manifest (non-blocking note).
- The Settings page is view-only unless the build sets `VITE_SETTINGS_EDITABLE=true` (see `docs/runbooks/DEVELOPMENT.md`). The public Vercel build must not set it.

## Open questions

- Is the SP0/SP2 prediction-surfaces rebuild plan still active? (Archived copy: `_internal-docs/gridiron_oracle/archive-2026-10-07/docs/superpowers/specs/`.)
- Is `verify_db.py` wanted? It was tracked and archived on 2026-10-07 as unreferenced; restore it from the archive if so.
- Which fantasy league format is the target? Draft ranks assume 8-team PPR.

## Decisions made

- [0001 Causal walk-forward evaluation](../decisions/0001-causal-walk-forward-evaluation.md)
- [0002 Artifact-backed, fail-closed serving](../decisions/0002-artifact-backed-fail-closed-serving.md)
- [0003 Betting and Kelly sizing stay in scope](../decisions/0003-betting-and-kelly-sizing-in-scope.md)
