# Pipeline Walkthrough — What to Do Now

**Date:** 2026-03-06  
**Purpose:** Step-by-step guide to get interceptions fixed, CatBoost trained, stacking run, and projections updated.

---

## 0. Prediction System: Stop or Leave?

| What's running | Action |
|----------------|--------|
| **Backend API** (`make up` / docker) | **Leave it running.** It serves from the DB. When you run the projection pipeline, new projections are written to the DB and the next API call will serve them. No restart needed. |
| **Training job** (e.g. `train_all_models.sh` in progress) | **Stop it.** Use Ctrl+C. Avoid running multiple training jobs at once (resource contention, potential corruption). |
| **MLflow server** | **Leave it running.** Stacking and the projection pipeline need it to load models. |

---

## 1. Full ETL (Interceptions + Feature Matrix)

**Why:** Rebuild `game_logs` and `feature_matrix` with correct interceptions. The Kalman fix (`STAT_SOURCE_COL["interceptions"] = "passing_interceptions"`) is in code; ETL must run to populate the tables.

**Command:**
```bash
cd "$(git rev-parse --show-toplevel)"
bash pipeline/run_full_etl.sh
```

**Runtime:** ~60–120 min (depends on nflreadpy cache).

**Prerequisites:** Docker running (PostgreSQL). The script starts it if needed.

---

## 2. CatBoost Training

**Why:** CatBoost OOF files are needed for stacking. You have XGB, LGBM, TFT; CatBoost was not run.

**Option A — Run only CatBoost (recommended):**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}
export DATABASE_URL=${DATABASE_URL:-postgresql://oracle:oracle@localhost:5432/oracle}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

# Run CatBoost for all stat/position combos (38 total)
for pos in WR RB QB TE; do
  case $pos in
    QB) STAT_SET="pass_attempts completions passing_yards passing_tds interceptions rushing_yards rushing_tds carries fantasy_ppr fumbles" ;;
    RB) STAT_SET="carries rushing_yards rushing_tds receptions receiving_yards receiving_tds targets fantasy_ppr fumbles" ;;
    WR) STAT_SET="receptions receiving_yards receiving_tds targets carries rushing_yards rushing_tds fantasy_ppr fumbles" ;;
    TE) STAT_SET="receptions receiving_yards receiving_tds targets carries rushing_yards rushing_tds fantasy_ppr fumbles" ;;
  esac
  for stat in $STAT_SET; do
    echo "→ CatBoost: $stat / $pos"
    .venv_311/bin/python3.11 -m ml.catboost_model --seasons "2019-2025" --target "$stat" --position "$pos" --n-trials 20 --out-dir ml/oof
  done
done
```

**Option B — Use train_all_models.sh with --resume:**
```bash
bash ml/train_all_models.sh --resume
```
This skips XGB, LGBM, TFT (already done) and runs CatBoost, then stacking, then pipeline. Ensure `ml/checkpoints/done/catboost_*` do **not** exist so CatBoost runs.

**Runtime:** ~3–5 hrs for CatBoost.

---

## 3. Stacking (Ridge Meta-Learner)

**Why:** Combines XGB + LGBM + CatBoost + TFT OOF into one prediction per game. The NaN/dedup fix is in place.

**Clear stacking checkpoints** (so it re-runs with the fix):
```bash
rm -f ml/checkpoints/done/stack_*.done
```

**Run stacking:**
```bash
.venv_311/bin/python3.11 -m ml.stacking_ensemble --oof-dir ml/oof --target pass_attempts --out-dir ml/oof
# Repeat for each stat, or use the loop below:
```

**Or run for all stats:**
```bash
for stat in pass_attempts completions passing_yards passing_tds interceptions \
            rushing_yards rushing_tds carries receptions receiving_yards receiving_tds \
            targets fantasy_ppr fumbles; do
  echo "→ Stacking: $stat"
  .venv_311/bin/python3.11 -m ml.stacking_ensemble --oof-dir ml/oof --target "$stat" --out-dir ml/oof
done
```

**Runtime:** ~10 min total.

**Output:** `ml/oof/ridge_{stat}_coefs.json` for each stat.

---

## 4. Projection Pipeline (Bayesian + Monte Carlo + DB Write)

**Why:** Produces final projections (point, floor, ceiling, boom/bust) and writes to the DB. Uses the Ridge coefs from Step 3.

**Command:**
```bash
.venv_311/bin/python3.11 -m ml.train \
  --seasons 2019 2020 2021 2022 2023 2024 2025 \
  --all-weeks \
  --resume
```

`--resume` skips (season, week) already in the projections table.

**Runtime:** ~4–8 hrs (full NUTS MCMC). Use `--n-bayesian-samples 500` for faster runs (less calibration).

---

## 5. Optional: Backtest

**Why:** Validate that the pipeline beats baseline on held-out seasons.

**Command:**
```bash
.venv_311/bin/python3.11 -m ml.run_backtest
```

---

## Summary Checklist

| Step | Command | Runtime |
|------|---------|---------|
| 0 | Stop any running training; leave backend/MLflow up | — |
| 1 | `bash pipeline/run_full_etl.sh` | 60–120 min |
| 2 | CatBoost (38 stat×pos combos) | 3–5 hrs |
| 3 | Stacking (all stats) | ~10 min |
| 4 | `python -m ml.train --seasons 2019 2020 2021 2022 2023 2024 2025 --all-weeks --resume` | 4–8 hrs |
| 5 | `python -m ml.run_backtest` (optional) | ~30 min |

---

## One-Liner (After ETL)

If you prefer a single script after ETL is done:

```bash
bash ml/train_all_models.sh --resume
```

With `--resume`, it skips XGB, LGBM, TFT (checkpoints exist), runs CatBoost, stacking, and the projection pipeline. Ensure ETL has completed first.
