# Gridiron Oracle: current status

As of 2026-10-10. Checked against the code by running the gates on the branches named below. Items marked unverified were not run here.
Update this file at the end of every working session. See `_internal-docs/_platform/DOCS_STANDARD.md`.

## What works, what is partial, what is unverified

| Check | Result (2026-10-10) |
|---|---|
| Candidate `integration/gor-2026-10-10-r2` (133d4b5): CI-equivalent pytest, no DB | 2107 passed, 30 skipped, coverage 80.81% (exit 0) |
| Candidate: ruff | clean |
| Candidate: frontend typecheck, 77 tests, build | all pass |
| Candidate: evidence manifest, gitleaks on `main..HEAD`, CI radon gate (`backend/app ml pipeline`) | pass |
| Candidate: C++ engine | no engine files changed since `bee3338`; last full ctest run (72 pass) was on the subagent's build |
| Candidate: pytest with the local Postgres up | 7 failed, 2 errors, all in DB-backed tests that also fail on `989e221`. The local DB has no games, pbp or 2026 rows. |
| `agent/gor-ml-04-forward-game-context` (77bcf7f): targeted forward-row and served-learner tests | 30 passed |

Not merged to `main`. The merge was blocked by the auto-mode classifier. Nothing is pushed.

**Model quality.**
- Owner-verified claim: on causal out-of-fold evaluation (15 stat/position cells, seasons 2021 to 2025), the model beat both the naive (last season) and trailing-3-games baselines in 74 of 75 cell-seasons. **Reproduction artifact not committed.** A per-cell eval CSV set for the current candidate is not in the repo.
- How to reproduce: for each cell, run `python -m ml.eval_causal --oof <pinned stack CSV from releases/current_baseline.json> --stat <stat> --position <pos> --out <csv>` (needs the database for baselines), then count rows where `beats_naive` and `beats_trailing3` are both true per cell-season. `scripts/summarize_causal_evals.py` does that aggregation but is hard-coded to older file names.
- History: in August 2026 an audit found a data leak; one report records the earlier headline falling from 74/75 to 15/75 after the fix, and the manifest labels all evidence "provisional". The 74/75 figure above is the owner-verified result on the rebuilt candidate.
- Committed, directly readable: `reports/marcel_vs_stack.json` (fantasy points only) has 16 scored position-season cells; the model beat naive and trailing-3 in all 16 and the Marcel baseline in 15.

**Partly built:** season projections; the draft board (served on an 8-team league, see open questions); the Settings page (saved weights have no consumer in the projection code).

**Not verified:** live deployment health; the Railway cron refresh; in-season weekly refresh (step 7 acceptance rejects unplayed windows, and 2026 stack artifacts are not pinned); the Bayesian/Monte Carlo layer on the served path; engine connection to any live odds feed; accuracy of the drive simulator; GitHub Actions itself (none of the branches has run on GitHub).

**Known UI gaps:** on the candidate, not on `main`. Still open: the hand-rolled controls listed in `docs/DESIGN_SYSTEM.md`, jargon copy (`docs/USERS.md`), `ErrorBanner` prints a developer command and raw error text, `GameDetail` prints an internal game id, units on the player page are always "yds".

## In progress

- Candidate `integration/gor-2026-10-10-r2` (133d4b5) is the merge of the ready branches: ML known bugs, refresh correctness, frontend design system. Excludes forward rows, availability commit `04033cf`, and the two leftover branches.
- `agent/gor-ml-04-forward-game-context` (77bcf7f, off the candidate): serves the approved learner (`causal_20260821_fixed`) and takes lines, rest and venue for forward rows from the target game. Missing lines take training medians. See decision 0006. Full gate not yet run on it.
- Phase 2 audit passes A/B/C (required by the brief): not started.

## Next steps (priority order)

1. Owner confirms the open items below. Nothing merges to `main` until then.
2. Confirm the missing-line fallback (45.0 total, 2.5 spread) before the VPS materialize run (decision 0006).
3. Run the Phase 2 audit passes on the candidate plus `ml-04` and record evidence under `.agent-work/gridiron/audit/`.
4. Commit a reproducer for the owner-verified 74/75 count: per-cell eval CSVs from `ml.eval_causal`, aggregated by an updated `scripts/summarize_causal_evals.py`.
5. Wire or remove the Settings weights (`user_config.json`). No projection code reads them. The design file lists the options.
6. Verify the engine's live odds connection and show a real book line in the UI (it shows "No line yet" until a live line exists).
7. Refresh cannot run in season until the step 7 acceptance design and 2026 stack artifacts are decided.
8. Rewrite jargon UI copy (list in `docs/USERS.md`).

## To-dos and reminders

- Keep `docs/DATA_SOURCES.md` documenting `load_pfr_advstats` (a test enforces it).
- Evidence entries in `releases/artifacts/MANIFEST.json` are all "provisional"; flip to frozen after a clean rebuild reproduces them.
- `ml/experiments/team_game_model/win_probability_2025_*.json` are unlisted in the manifest (non-blocking note).
- `scripts/generate_sleeper_draft_profiles.py` `build_profiles` is rank F (CC > 40). Outside the CI gate, which scans `backend/app ml pipeline` only.
- The Settings page is view-only unless the build sets `VITE_SETTINGS_EDITABLE=true` (see `docs/runbooks/DEVELOPMENT.md`). The public Vercel build must not set it.
- Branch protection is not set on `main`. The owner must add required checks in GitHub.

## Open questions

- League format. The code and the Sleeper profile generator model an 8-team league: `ml/vor.py` (`LEAGUE_TEAMS = 8`), `backend/app/api/mock_draft.py` (rejects anything but 8 managers), the DraftBoard UI (8 leaguemates), and `scripts/generate_sleeper_draft_profiles.py` (8 owners). The owner's stated format is 18 teams. The owner must say whether the 18-team league is a different league or the product should move to 18. Moving to 18 needs 18 manager profiles and changes the served draft board. Also unconfirmed: TE count (working assumption 1 TE), and whether K and DEF enter the board (the model excludes them by design).
- Is `verify_db.py` wanted? It was archived on 2026-10-07 as unreferenced.

## Decisions made

- [0001 Causal walk-forward evaluation](../decisions/0001-causal-walk-forward-evaluation.md)
- [0002 Artifact-backed, fail-closed serving](../decisions/0002-artifact-backed-fail-closed-serving.md)
- [0003 Betting and Kelly sizing stay in scope](../decisions/0003-betting-and-kelly-sizing-in-scope.md)
- [0004 shadcn on Base UI](../decisions/0004-shadcn-on-base-ui.md)
- [0005 SP0-SP4 rebuild plan retired; built pieces kept](../decisions/0005-sp0-sp2-plan-retired.md)
- [0006 Forward rows take game context from the target game](../decisions/0006-forward-rows-take-target-game-context.md)
- [0007 Public Settings are read-only](../decisions/0007-public-settings-read-only.md)
