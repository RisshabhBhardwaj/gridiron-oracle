# 07 · CODEX · Final remediation re-audit — 2026-08-19

## Scope and evidence

This re-audit follows the 2026-08-11 findings at `1e543bf`.  The remediation
implementation is commit `68d0998`; the candidate materialization record is
commit `9a25cfe`.  The candidate's source manifest is
`releases/candidates/causal_20260810/release_manifest.json` and pins every
executable artifact by SHA-256.

## Requirement-level result

| Requirement | Status | Current evidence |
| --- | --- | --- |
| NEW-01: 15 loadable Ridge artifacts | **Done** | All 15 are regenerated in the hardened `{learner_order, weights, intercept}` schema, manifest-pinned, and loaded through `InferenceClient`. |
| Wave 04 evaluation/gates/parity/baselines | **Implemented; promotion correctly blocked** | The gate requires a distinct `frozen_baseline` measurement, fails closed for missing evidence/self-comparison, rejects regression, and uses explicit candidate OOF paths. Evaluation rejects absent provenance, uses `CohortSpec`, and scores a common finite cohort. All 15 serving-vs-OOF reports are pinned. |
| Stale causal summaries | **Done** | The two contaminated machine-readable summaries were removed rather than regenerated from invalid evidence. |
| Full declared materialization | **Done** | Run `stack_materialize_20260819T145053Z` upserted 80,404 rows across all 15 declared cells. |
| Uncertainty integrity | **Done, with explicit coverage limitation** | 64,525 rows have causal 90% conformal bounds calibrated only from earlier OOF seasons; 15,879 first-season rows are explicitly unavailable. `p25`/`p75` stay null because they are not posterior percentiles. |
| Wave 05 test hardening | **Done for portable audit locks** | Current portable locks: 3 passed. The same file run against `audit-baseline-2026-08-09` fails 2/3 (raw target-game feature and glob-based gate selection). The focused serving/gate/eval/schema suite passed 106 tests, 1 environment-dependent DB skip. |
| Alembic / production reconciliation | **Done** | Revisions `20260810_0008` and `20260810_0009` adopt the two legacy Sleeper tables and 26 `feature_matrix` columns. ORM and database parity are covered by a DB-backed test; scratch upgrade → downgrade → upgrade completed. |

## Release decision

**The candidate is not promoted.** Its status is
`candidate_only_pending_wave_04_promotion_no_valid_prior_frozen_baseline`.
This is intentional and correct: every pre-remediation release is invalidated
by C-01, so using one as a frozen baseline would recreate the exact
self-comparison/fabricated-evidence defect Wave 04 prevents.

Consequently, `releases/current_baseline.json` is deliberately not re-frozen
to the candidate. It remains a stale release that readiness rejects, rather
than silently promoting an un-gated candidate. A final promoted baseline
requires a separately captured, valid prior causal release (or an approved
governance decision defining a new first-release promotion policy). No code or
artifact can truthfully manufacture that historical evidence.

## Verification commands run

```bash
.venv_311/bin/python -m pytest \
  backend/tests/test_serving_correctness.py \
  backend/tests/test_artifact_containment.py \
  backend/tests/test_reprojection_gate.py \
  backend/tests/test_eval_integrity.py \
  backend/tests/test_audit_baseline_locks.py \
  backend/tests/test_schema_database.py -q
# 106 passed, 1 skipped

.venv_311/bin/python -m pytest \
  backend/tests/test_audit_baseline_locks.py \
  backend/tests/test_feature_contract.py \
  backend/tests/test_reprojection_gate.py \
  backend/tests/test_serving_correctness.py \
  backend/tests/test_schema_database.py -q
# 18 passed, 1 skipped
```
