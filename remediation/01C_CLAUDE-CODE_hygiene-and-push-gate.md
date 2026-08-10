# 01C · CLAUDE CODE · Hygiene and push gate

| | |
|---|---|
| **Order** | Wave 1, stream C — run first |
| **Concurrency** | ⇄ **CONCURRENT** with `01A` and `01B` |
| **Depends on** | nothing — fully independent of the ML work |
| **Blocks** | nothing |
| **Findings** | C-07, C-23, C-24, C-25, C-28, C-29, C-30, C-31, C-32 (9 findings) |
| **Runtime** | one session |

### Setup — run before pasting the prompt

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_01C_hygiene -b fix/01c-hygiene audit-baseline-2026-08-09
cd ../go_01C_hygiene
```

Interpreter: `/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`.
This session writes nothing to Postgres, MLflow, or `ml/oof/`.

---

You are working in `gridiron_oracle` in a worktree branched from `audit-baseline-2026-08-09`. Read
`CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` first. This session fixes nine small, independent findings
that touch no ML artifact.

## GUARDRAILS

- **Do not run any trainer or restack**, and do not touch `ml/oof/`, `ml/stacking_ensemble.py`,
  `ml/train.py`, `ml/adp_eval.py`, `scripts/train_*.sh`, `ml/train_all_models.sh`,
  `scripts/materialize_stack_projections.py` or `backend/app/api/draft.py` — a concurrent session
  (`01A`) owns those.
- Do not touch feature definitions in `pipeline/feature_engineer.py` or `pipeline/features/` — session
  `02` owns those. **Exception:** C-25 below concerns `pipeline/pbp_pipeline.py:198` only; limit yourself
  to that call and its documentation.
- No database writes.

## C-07 + C-30 — Push gate

`.githooks/pre-push:38-60` sets `range="$local_sha"` for a new remote ref and runs a one-commit
`git diff-tree`; it never enumerates the new commits. **Reproduced:** commit a `.env`, then commit only
`README.md`, feed the tip with a zero remote SHA to the real hook → **exit 0**. On existing branches it
diffs endpoints, so add-then-delete within the range is invisible. It scans path names only
(`:11-22`) — no content, secret or size scan — and `.env` patterns are root-anchored.

`core.hooksPath` points at `.git/hooks`; `.githooks/pre-push` is a copy installed by `make hooks`, so any
other clone has **no gate at all** (content currently byte-identical — verified).

- For new refs, scan `git rev-list --objects --root <local> --not --remotes=<remote>` over every
  introduced commit and tree.
- Add content-secret scanning (gitleaks) and an object-size limit.
- **Enforce server-side or as a required CI check.** The local hook is convenience, not a boundary — say
  so in `CONTRIBUTING.md`.
- Set `core.hooksPath=.githooks` in bootstrap so clones are covered without `make hooks`.
- Add a test reproducing the ancestor bypass; it must exit nonzero after your fix.

## C-23 — FantasyPros importer alias collision

`scraper/adapters/fantasypros_adp_importer.py:40-50` maps both `rank`→adp and `avg`→adp, so a standard
FantasyPros export silently imports **Rank as ADP** — wrong by an order of magnitude, on the documented
2026 import path. Fix the alias precedence and add a fixture test using a real FP export column shape.

## C-24 — Coaching seed data

`data/coaching/coaching_2026.csv` has Terrelle Pryor as NE DC, and Kingsbury and Schwartz each
coordinating two teams; every row literally says "verify"; the adapter performs no semantic validation.
`team_coaching` has **zero consumers**, so it is currently inert. Either correct it or delete it — do not
leave provably wrong seed data in the tree. Add position/team sanity validation (one DC per team, coach
names resolvable) and record provenance plus verification status.

## C-25 — "PFR stub gone" is only half true

The retired `scraper/adapters/pro_football_ref.py` raises on construction with zero callers — that part is
real and two reviews verified it. But `pipeline/pbp_pipeline.py:198` still calls
`nfl.load_pfr_advstats(seasons=season, stat_type="rec")` and `pipeline/features/feature_row.py:190`
sources drop rate from it. Either retire that path too, or correct every claim in README/docs/CHANGELOG
that says PFR is gone. Do not leave the documentation false.

## C-28 — Season caps are advisory

- `ml/season_constants.cap_seasons` clamps the **lower** bound to `TRAIN_SEASON_START=2019` as well as the
  upper bound.
- `ml/utils._parse_seasons:639-649` drops out-of-range seasons with a *warning*, not an error:
  `_parse_seasons("2018-2024")` silently returns 2019–2024, leaving
  `backend/tests/test_xgb_model.py::TestParseSeasons::test_range_notation` **red**.
- `assert_not_fitting_incomplete_season` is never called in production code — the pre-Week-1 guarantee
  rests on a log line.

Raise on out-of-range instead of warning; call the assert at trainer entrypoints; fix the red test.
(Touch only `_parse_seasons` and the season constants in `ml/utils.py` — leave feature definitions alone.)

## C-29 — Evidence lives under ignored paths

`.gitignore:274` ignores `reports/`, `ml/oof/`, `ml/experiments/` while ~20 artifacts are force-added, so
re-running an eval produces no `git status` signal and evidence drifts silently — which already happened
(`gate_*.json` at 15:46 vs `current_baseline.json` at 17:14). Move shipped evidence to an un-ignored
`releases/artifacts/` tree with a manifest. **Coordinate:** do not move anything under `ml/oof/` — set up
the structure and the policy; `02` will populate it with rebuilt artifacts.

## C-31 — Metric mislabeling

`reports/eval_causal_volume_summary.json` reports mean Poisson deviance under the key `pooled_mae`
(source `ml/eval_metrics.py`). Rename it, and check every other report key against what is actually
computed — assume there are more.

## C-32 — e2e smoke test

`e2e/smoke_test.sh:125-130` calls `/predict` with obsolete `player_id`/`position` params instead of the
required `player`, so it cannot validate the current contract; assertions are `grep -qE '\[|\]'`. Fix the
request contract and make the assertions meaningful (status code, schema shape, field presence).
`bash -n` passing is not a test.

## Done criteria

- The ancestor-bypass reproduction exits **nonzero**; gate enforced in CI; `core.hooksPath=.githooks`.
- FP fixture test passes and proves Rank is no longer imported as ADP.
- Coaching data corrected or removed; validation in place.
- No documentation claims PFR is gone while `load_pfr_advstats` is still called.
- `_parse_seasons` raises on out-of-range; `TestParseSeasons` green.
- `releases/artifacts/` structure and policy documented.
- Report-key names match computed metrics.
- Smoke test exercises the real `/predict` contract.
