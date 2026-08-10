# 01C — hygiene and push gate: handoff

**Branch:** `fix/01c-hygiene` (worktree `../go_01C_hygiene`), one commit on top of
`audit-baseline-2026-08-09`.
**Findings closed:** C-07, C-23, C-24, C-25, C-28, C-29, C-30, C-31, C-32.
**Nothing under `ml/oof/` was moved or rewritten.** No database writes. No
trainer or restack was run.

## Items other sessions must pick up

### → schema work (03B / C-12): `team_coaching` needs three columns

C-24 added `source`, `verified_by`, `verified_on` to
`coaching_adapter.CREATE_TEAM_COACHING`.
`alembic/versions/20260806_0003_draft_and_coaching.py` creates `team_coaching`
without them, and `CREATE TABLE IF NOT EXISTS` is a no-op against an existing
table — so on an Alembic-migrated database `upsert_coaching` would fail with
`column "source" does not exist`.

Cannot fire today: no seed CSV ships, the table has zero consumers, and nothing
invokes the adapter. **Add the three columns in a new revision before anything
writes to `team_coaching`.** Also noted in `data/coaching/README.md`.

### → feature/retrain work (02 / C-01): populate `releases/artifacts/`

The structure, policy, and verifier are in place and passing; every entry is
`status: "provisional"` because the audit invalidated the evidence itself.

- Register rebuilt artifacts in `releases/artifacts/MANIFEST.json` and flip them
  to `frozen` as they are accepted.
- Add `ml/oof` to `WATCHED_DIRS` in `scripts/verify_evidence_manifest.py` once
  the rebuilt OOFs are registered — it deliberately covers only `reports/` and
  `ml/experiments/` today so this session did not touch `ml/oof/`.
- `scripts/summarize_causal_evals.py --check` will go red after any eval rerun;
  regenerate with the script rather than hand-editing the summaries.

### → whoever configures the repository on GitHub

The `push-gate` job in `.github/workflows/ci.yml` is the authoritative gate and
must be marked a **required status check** on `main`. The local hook is
convenience only and is skippable with `--no-verify`; `CONTRIBUTING.md` says so
explicitly. Server-side secret scanning and push protection should be enabled
alongside it.

## Not verified in this session

- **`e2e/smoke_test.sh` was not executed end to end.** The assertion helper
  (`e2e/lib/http_assert.sh`) is tested against a live stub server in
  `backend/tests/test_http_assert.py`, including cases proving it fails when it
  should, and the script passes `bash -n`. The `/predict` request itself has
  never been run against a live app + database.
- **The `push-gate` CI job has never run on a GitHub runner.** The hook it
  invokes was run locally over the full branch history (exit 0) and against 17
  reproduction cases. `gitleaks` is not installed locally, so its success path
  is unexercised; the hook now probes for `gitleaks git` vs `gitleaks detect`
  rather than assuming, distinguishes findings (exit 1) from tool failure, and
  CI validates the invocation against the pinned binary before the gate runs.
- **The gate has never run against a real remote** (`git push`), only through
  its stdin protocol.

## Pre-existing failures left in place

`pytest backend/tests/ -m "not integration and not slow and not network"`:
**913 passed, 8 skipped, 2 failed.** Both failures reproduce on the untouched
audit baseline:

| Test | Owner |
|---|---|
| `test_kalman_tracker.py::TestSchemaSync::test_feature_row_fields_in_feature_matrix` | 03B (C-12) |
| `test_train_pipeline.py::TestStackingInferenceHelpers::test_stacking_mlflow_with_mock_models_uses_their_predictions` | 01A (C-04 / C-05) |

The audit counted ~3 genuine failures at HEAD; the third,
`test_xgb_model.py::TestParseSeasons::test_range_notation`, is fixed here.

`ruff check .` reports 5 pre-existing errors, all in files this session was
told not to touch (`ml/adp_eval.py` ×2, `ml/feature_groups.py`,
`scripts/reprojection_gate.py`, `scripts/yardage_diagnostic_report.py`). CI's
lint step is therefore red independent of this branch.

## Incidental fix outside the nine findings

`.gitignore`'s packaging block had an unanchored `lib/`, which matched every
directory named `lib` at any depth and silently excluded
`e2e/lib/http_assert.sh` — a file the smoke test sources, so it would have been
absent from every fresh clone. Anchored to `/lib/` and `/lib64/`. Same class as
the mid-line-comment ignore bug the audit credited as fully fixed.
