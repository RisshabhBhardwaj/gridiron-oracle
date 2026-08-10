# 03A · CLAUDE CODE · Serving correctness

| | |
|---|---|
| **Order** | Wave 3, stream A |
| **Concurrency** | ⇄ **CONCURRENT** with `03B` and `03C` |
| **Depends on** | `02` merged and its wave gate passed |
| **Blocks** | `04`, `05` |
| **Findings** | C-04, C-09, C-10, C-11 |
| **Shared resources** | **This is the only wave-3 session that writes to Postgres or uses MLflow.** |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_03A_serving -b fix/03a-serving main
cd ../go_03A_serving
```

Interpreter: `/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`.
`03B` uses a scratch DB and `03C` reads only, so you own writes to `:15439`.

---

Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` rows C-04, C-09, C-10, C-11. Do not touch features,
schema migrations, or the draft rank basis — concurrent sessions own those.

## C-04 — Make inference actually fail closed

`ml/inference_client.py:89-143` correctly raises on missing or mismatched artifacts.
`ml/train.py:737-790` catches every failure and returns Kalman estimates or zeros, unmarked, without
consulting `PRODUCT_MODE`. `backend/tests/test_train_pipeline.py:515-578` **locks this in** — that test
asserts the bug.

- In artifact-backed mode, propagate all artifact / coef / model failures. `ml/train.py:1083` already shows
  the correct pattern; apply it at `:781`.
- Validate coefficient files strictly. `inference_client.py:476-497` currently defaults a missing intercept
  to 0, ignores unknown keys, and accepts JSON `NaN` — a file of
  `{"lgbm": NaN, "catboost": 0.4, "unknown_learner": 99}` loads as `([nan, 0.4], 0.0, ['lgbm','catboost'])`.
  Require exact schema, explicit `learner_order`, finite weights and intercept, no unknown keys, finite outputs.
- Any fallback that does occur must stamp a `degraded` flag and the real `pipeline_run_id`. **A Kalman
  estimate must never be indistinguishable from a model projection.**
- Rewrite the test that locks the fallback; add one asserting no Kalman fallback under
  `PRODUCT_MODE=artifact_backed`.

## C-11 — Stop serving constants as percentiles

`scripts/materialize_stack_projections.py:85-95` derives one global residual-MAD band per (stat, position)
from the artifact's own `y_true`. Every WR row has `ceiling-projection` = 4.3677; 492 WR rows have negative
floors. `backend/app/api/predict.py:222-224` maps `floor→p10`, `ceiling→p90`. `posterior_samples` is NULL on
all rows, and the on-conflict clause (`:127-140`) does **not** clear posterior/boom/bust — a stale posterior
can survive an overwrite and pair with a new mean.

- Per-player residual bands or real posterior quantiles. If neither is available, null the percentile fields
  and expose an explicit `interval_method`. **Do not ship a constant in a field named `p90`.**
- Fix the on-conflict clause to overwrite every mutable field including nulls.
- Fix negative floors (clamp at 0 for non-negative stats, or use an asymmetric interval).

## C-10 — Materialize the full cell matrix

`scripts/materialize_stack_projections.py:47-58` `CELLS` omits `receiving_yards`×3 and `rushing_yards`×2 —
the five stacks backing the 24/25 claim. Only 10 of 15 cells reach the DB.

- Add the five yardage cells, or explicitly declare them unserved in the manifest. Do not leave the gap
  undocumented.
- `:104-107` asserts `max_train_season` rather than reading real artifact provenance. Read it from the artifact.
- Verify SHA-256 against the manifest before materializing anything (`01A` gave you pinned selection).
- Materialize the **post-`02` rebuilt artifacts**, not the `_20260809` set.

## C-09 — Make the manifest true

`releases/current_baseline.json` pins `git_commit=a4fb0e9` (HEAD is well past it), records
`git_dirty: true`, omits yardage, and declares `require_posterior_samples: true` with
`approved_pipeline_run_ids: []` — a policy every shipped row violates. `ml/backtest.py:159-214` would reject
100% of rows, so trusted replay is impossible. `backend/app/services/runtime_status.py:152-176` validates
only that `model_version` exists, and called this manifest "ok".

- Regenerate at a clean HEAD after the `02` retrain, listing all 15 cells.
- Register the real materialize run ID.
- Reconcile `require_posterior_samples` with what C-11 actually produces.
- Make readiness **reject** stale, dirty, or commit-mismatched manifests instead of passing them.
- Also fix `product_mode: artifact_backed` contradicting `.env.example`'s `graceful_fallback` default.

## Done criteria

- No constant-offset value served in a percentile-named field.
- Every declared cell present in the DB, or explicitly excluded in the manifest.
- Manifest matches HEAD and the actual DB run; readiness fails on a stale or dirty manifest.
- `PRODUCT_MODE=artifact_backed` provably fails closed under test, including for a malformed coef file
  containing `NaN` and an unknown key.
