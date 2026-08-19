# Remediation regression locks

The tests below guard the adversarial-audit findings.  The audit-baseline
worktree is `/Users/risshabh/Projects/Active/go_06_baseline` at `191e151`.
Where a test imports a newly introduced module it cannot run verbatim there;
those use a minimal behavioural lock that only imports/modules-inspects code
which already existed on the baseline.

| Finding / requirement | Current lock | Baseline-red evidence |
| --- | --- | --- |
| C-01 as-of feature contract | `test_feature_contract.py`, `test_audit_baseline_locks.py::test_default_model_features_exclude_raw_target_game_snap_participation` | The behavioural lock raises because baseline `FEATURE_COLS` contains `snap_pct_off`. |
| C-03 draft no-actuals | `test_draft_api.py::test_preseason_rank_is_not_driven_by_target_season_availability` | Earlier draft implementation reads target-season weekly OOF; this is retained as a source/behavioural release lock. |
| C-04 fail closed | `test_train_pipeline.py::TestStackingInferenceHelpers::test_stacking_artifact_backed_exception_never_falls_back` | Baseline falls back to Kalman in artifact-backed mode. |
| C-06 artifact authority | `test_audit_baseline_locks.py::test_serving_artifact_selection_does_not_use_mtime`, `test_artifact_containment.py` | Baseline selects artifacts by mtime. |
| C-07 push-gate ancestry | `test_push_gate.py::test_new_ref_ancestor_bypass_is_blocked` | Baseline hook permits the ancestor bypass. |
| C-08 promotion gate | `test_reprojection_gate.py::{test_gate_rejects_regressed_candidate_against_distinct_prior_release,test_gate_fails_closed_without_frozen_prior_release_evidence}` | Baseline accepts missing/self-comparison gate evidence. |
| C-10/C-11 materialization | `test_serving_correctness.py` | Baseline omits five cells and emits constant pseudo-percentiles. |
| C-18 two-team aggregation | `test_phase4_phase6.py::TestPhase4Buckets::test_team_game_aggregates_do_not_combine_opponents` | Baseline combines opponent team totals under `game_id`. |
| C-12 / NEW-02 schema parity | `test_schema_database.py`, `TestSchemaSync` | Baseline ORM/Alembic lacks live `feature_matrix` surface. |
| C-34 baseline/eval behaviour | `test_eval_integrity.py`, `test_audit_baseline_locks.py::test_reprojection_gate_does_not_select_candidate_evidence_by_glob` | Baseline invents provenance and uses a glob-selected candidate input. |

Run the current focused remediation locks:

```bash
.venv_311/bin/python -m pytest \
  backend/tests/test_audit_baseline_locks.py \
  backend/tests/test_feature_contract.py \
  backend/tests/test_reprojection_gate.py \
  backend/tests/test_serving_correctness.py \
  backend/tests/test_schema_database.py -q
```

Prove the three portable behavioural locks red against the baseline without
modifying that worktree:

```bash
cd /Users/risshabh/Projects/Active/go_06_baseline
GRIDIRON_REPO_ROOT="$PWD" PYTHONPATH="$PWD" \
  /Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python -m pytest \
  /Users/risshabh/Projects/Active/gridiron_oracle/backend/tests/test_audit_baseline_locks.py -q
```
