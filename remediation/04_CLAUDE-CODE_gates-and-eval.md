# 04 · CLAUDE CODE · Gates and eval methodology

| | |
|---|---|
| **Order** | Wave 4 — after all of wave 3 merges |
| **Concurrency** | ⛔ **SOLO** |
| **Depends on** | `03A`, `03B`, `03C` all merged |
| **Blocks** | `05` |
| **Findings** | C-08, C-13, C-21, C-27, C-34 |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree remove ../go_03A_serving ../go_03B_schema ../go_03C_draft   # after merging
git checkout main && git pull
git checkout -b fix/04-gates
```

Run in the main tree. Needs Postgres, MLflow and `.venv_311`.

---

Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` rows C-08, C-13, C-21, C-27, C-34.

Every gate in this repo currently passes by construction. Fixing them **after** the `02` retrain is
deliberate: gate baselines must be frozen from real prior evidence, and there was no valid prior evidence
until the leak fix landed.

## C-08 — Gates that can actually fail

- `ml/experiments/reprojection_gate/gate_*.json` records `baseline_mae == candidate_mae == oof_mae`
  bit-for-bit (QB 2024 = 5.661643174131408; WR 2024 = 5.241194793602187) with `promote: true`. The
  candidate is compared to itself; `x <= x*1.05` cannot fail. **Freeze baselines from the previously
  promoted release**, never from the candidate's own OOF.
- `scripts/reprojection_gate.py:100-127` **passes when frozen evidence is missing.** Fail closed.
- It tolerates a default 5% regression, and `promotion_gate` in `ml/stacking_ensemble.py` uses
  `min_improvement_pct=0.0` (any tie promotes). Zero tolerated regression unless explicitly approved with a
  recorded reason and approver.
- `_load_oof:37-49` blindly concatenates globs across learners and positions, duplicating rows. Pin to the
  manifest.
- `ml/stacking_ensemble.py:577-599` hard-fails only when an env var is explicitly artifact-backed — which is
  how `ml/oof/promotion_fantasy_ppr_QB.json`, `_WR.json` and `promotion_rushing_yards_RB.json` came to be
  **`false` and shipped anyway**. Committing a false promotion must become impossible.

## C-13 — Eval that means what it says

- `ml/eval_causal.py:94-113` scores the model on all rows but each baseline only on its own finite subset,
  then compares them. WR 2024: model 1,793 rows, naive 1,491, trailing-3 1,668 — and the reported `n` is the
  model's. Score all three on **one predeclared intersection**; emit per-predictor missing counts and both
  `n_model` and `n_baseline`.
- `_normalize_oof:32-48` invents `max_train_season = season-1` when absent, then asserts it `< season` at
  `:113` — making `max_train_season_ok: True` tautological in every committed eval CSV. **Fail when the
  column is missing.** Require real signed fold/run provenance.
- `ml/eval_cohort.py`'s `CohortSpec` exists and is unused by the OOF scorer. Wire it in.

## C-21 — Serve what was evaluated

Shipped `ridge_*_coefs.json` are the final fit over all folds *including the eval seasons*; the OOF values
came from per-fold meta models. Traced: Godwin 2024 wk5 stored 16.4452 vs. recompute from shipped coefs
16.3073; St. Brown 2024 wk2 stored 12.670 vs. 12.605.

- Ship walk-forward-consistent coefficients, or record the divergence per cell in the manifest.
- `reports/serving_divergence_passing_yards_QB.json` compares the DB against the CSV it was copied from —
  corr = 1.0 by construction, proving upsert integrity rather than inference parity. Replace with a real
  comparison: **live inference vs. the evaluated OOF**, for all cells.

## C-27 — A/B on the real model class

`ml/feature_groups.py:140-143` measures group deltas with `Ridge(alpha=10)` on a single holdout season while
mutating module-global `ml_utils.FEATURE_COLS`. A Ridge delta does not transfer to an LGBM/CatBoost stack, so
the −0.05 promote threshold is measured on the wrong model class.

- Gate on the actual stack, or state the proxy limitation prominently in `reports/feature_group_ab_*.json`.
- Rerun Phase-4 A/B on the post-`02` features — the old report evaluated features that were both leaked and
  wrongly aggregated, so its conclusions carry no information.

## C-34 — Make eval fast enough to actually run

`ml/baselines.py:94-126` does a full-DataFrame boolean scan per eval row: ~97s per position-season set, and
one reviewer's 20-cell recheck never completed and had to be interrupted. Vectorize with
`groupby`/`shift`/`rolling` joins (<2s). **Assert bit-identical output against the current implementation on
at least one cell before replacing it** — a fast wrong baseline is worse than a slow right one.

## Done criteria

- A deliberately regressed candidate is **rejected** by the gate — demonstrate it.
- Missing evidence fails closed; no promotion record can be committed as `false`.
- All predictors scored on one cohort, with per-predictor counts reported.
- No fabricated provenance anywhere; missing `max_train_season` raises.
- Live-inference-vs-OOF divergence measured and recorded for all cells.
- Vectorized baselines proven identical to the old implementation on a reference cell.
