#!/bin/bash
# ml/train_all_models.sh
#
# Train all base learners (XGB + LGB + CatBoost + TFT), stack them, then
# re-run the projection pipeline with real stacking inference.
#
# PARALLELISM:
#   Steps 1-3 (XGB, LGB, CatBoost) run in PARALLEL as background processes.
#   Each model type trains sequentially across its stat/pos combos internally,
#   but all three models train simultaneously — ~3× wall-clock speedup.
#   Step 4 (TFT) starts only after all three tree models finish.
#
# PREREQUISITES:
#   1. PostgreSQL running with feature_matrix populated (run pipeline/run_full_etl.sh)
#   2. MLflow server running (handled by 'make up', starts on port 5001).
#
# RUNTIME ESTIMATES (Apple Silicon M1/M2/M3, parallel tree training):
#
#   --fast-mode (recommended for first run / time-constrained):
#   XGB+LGB+CB parallel (38 stat×pos combos each):  ~3-5 hrs (bottleneck = CB)
#   TFT fast    (3 epochs, 0 optuna trials):         ~2-4 hrs
#   Stacking:                                        ~10 min
#   Pipeline    (ml.train --fast, Gaussian approx):  ~20-40 min
#   TOTAL fast:                                      ~6-10 hrs
#
#   Full mode:
#   XGB+LGB+CB parallel:                            ~3-5 hrs
#   TFT full    (30 epochs, 20 optuna trials):       ~18-22 hrs
#   Stacking:                                        ~10 min
#   Pipeline    (ml.train, full NUTS MCMC):          ~4-8 hrs
#   TOTAL full:                                      ~26-36 hrs
#
# PROGRESS & RESUME:
#   With --resume: skips tasks that have a checkpoint file. Run again after
#   a crash to continue where you left off. Checkpoints: ml/checkpoints/done/
#
# Run from project root:  bash ml/train_all_models.sh
# Fast mode (~8-12 hrs):  bash ml/train_all_models.sh --fast-mode
# Resume after crash:     bash ml/train_all_models.sh --fast-mode --resume

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
# Respect caller-provided MLflow URI; default only when unset.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5001}"
export DATABASE_URL=${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

# Use .venv_311 so all project packages (xgboost, lightgbm, mlflow,
# pytorch_forecasting, nflreadpy, psycopg2, optuna, etc.) are importable.
# Bare /opt/homebrew/bin/python3.11 does NOT have these installed.
VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venv_311"
if [ ! -f "$VENV_DIR/bin/python3.11" ]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Create it: python3.11 -m venv .venv_311 && pip install -r requirements.txt"
  exit 1
fi
PYTHON="$VENV_DIR/bin/python3.11"

# Artifact-backed runs may not silently substitute null PBP/routes/depth/weather
# groups or operate without historical prop lines. Development fallback runs are
# intentionally exempt so local unit tests remain self-contained.
if [ "${PRODUCT_MODE:-graceful_fallback}" = "artifact_backed" ]; then
  "$PYTHON" scripts/verify_artifact_prerequisites.py
fi
OOF_DIR=ml/oof

# Per-position stat sets — matches POSITION_STAT_MAP in ml/train.py.
# Training a separate model per (stat, position) captures position-specific
# feature importance. Irrelevant stat/position combos (e.g. QB receiving_yards)
# are skipped to save compute.
#
# QB:    11 stats (passing-heavy set + rushing + protection)
# RB:    9 stats (rushing-primary + receiving out of backfield)
# WR/TE: 9 stats (receiving-primary + occasional rushing)
# sacks_taken excluded: actual_sacks_taken is NULL until PBP pipeline populates it (Phase 4)
QB_STATS="pass_attempts completions passing_yards passing_tds interceptions rushing_yards rushing_tds carries fantasy_ppr fumbles"
RB_STATS="carries rushing_yards rushing_tds receptions receiving_yards receiving_tds targets fantasy_ppr fumbles"
WR_STATS="receptions receiving_yards receiving_tds targets carries rushing_yards rushing_tds fantasy_ppr fumbles"
TE_STATS="receptions receiving_yards receiving_tds targets carries rushing_yards rushing_tds fantasy_ppr fumbles"

POSITIONS="WR RB QB TE"
# All unique stats across all positions (union of position stat sets).
# Used in TFT (all-position model) and stacking loops.
STATS="pass_attempts completions passing_yards passing_tds interceptions \
       rushing_yards rushing_tds carries receptions receiving_yards receiving_tds \
       targets fantasy_ppr fumbles"

mkdir -p "$OOF_DIR"

# ── Resume / checkpoint (--resume skips completed tasks) ───────────────────────
RESUME=false
FAST_MODE=false
for arg in "$@"; do
  [[ "$arg" == "--resume" ]]    && RESUME=true
  [[ "$arg" == "--fast-mode" ]] && FAST_MODE=true
done
CHECKPOINT_DIR="ml/checkpoints/done"
mkdir -p "$CHECKPOINT_DIR"

if [[ "$RESUME" == "false" ]]; then
  echo "Clean start: purging stale OOF predictions and checkpoints..."
  rm -f "$OOF_DIR"/*.csv
  rm -f "$CHECKPOINT_DIR"/*.done
fi

should_skip() {
  local step=$1 stat=$2 pos=${3:-}
  [[ "$RESUME" != "true" ]] && return 1
  if [[ -n "$pos" ]]; then
    [[ -f "$CHECKPOINT_DIR/${step}_${stat}_${pos}.done" ]]
  else
    [[ -f "$CHECKPOINT_DIR/${step}_${stat}.done" ]]
  fi
}
# TFT: also consider done if OOF file exists (stat ran to completion, checkpoint may have missed)
should_skip_tft() {
  local stat=$1
  [[ "$RESUME" != "true" ]] && return 1
  [[ -f "$CHECKPOINT_DIR/tft_${stat}.done" ]] && return 0
  # Fallback: OOF written = stat complete (TFT writes OOF at end of all folds)
  ls "$OOF_DIR"/tft_${stat}_*.csv 1>/dev/null 2>&1 && return 0
  return 1
}
mark_done() {
  local step=$1 stat=$2 pos=${3:-}
  if [[ -n "$pos" ]]; then
    touch "$CHECKPOINT_DIR/${step}_${stat}_${pos}.done"
  else
    touch "$CHECKPOINT_DIR/${step}_${stat}.done"
  fi
}
[[ "$RESUME"    == "true" ]] && echo "Resume mode: skipping completed tasks (checkpoints in $CHECKPOINT_DIR)"
[[ "$FAST_MODE" == "true" ]] && echo "Fast mode: TFT max_epochs=3 + 0 optuna trials; ml.train Gaussian approx (no NUTS MCMC)"

# ── Log directory (Issue 20: log rotation) ────────────────────────────────────
LOG_DIR="ml/logs"
mkdir -p "$LOG_DIR"
RUN_STAMP=$(date '+%Y%m%d_%H%M')
echo "Training logs: $LOG_DIR/${RUN_STAMP}_*.log"

# ── Steps 1-3: XGBoost + LightGBM + CatBoost (parallel) ──────────────────────
# Each model type is wrapped in a shell function and launched as a background
# process. All three train simultaneously — ~3× wall-clock speedup vs sequential.
# TFT (Step 4) starts only after all three finish (wait below).

run_xgb() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  STEP 1 — XGBoost base learner  [PID $$]"
  echo "════════════════════════════════════════════════════════════════"
  local fail=0
  for pos in $POSITIONS; do
    case $pos in
      QB) STAT_SET="$QB_STATS" ;;
      RB) STAT_SET="$RB_STATS" ;;
      WR) STAT_SET="$WR_STATS" ;;
      TE) STAT_SET="$TE_STATS" ;;
    esac
    for stat in $STAT_SET; do
      if should_skip xgb "$stat" "$pos"; then
        echo "  [XGB] SKIP (done): $stat/$pos"
        continue
      fi
      echo "→ [XGB] stat=$stat  position=$pos"
      if $PYTHON -m ml.xgb_model \
        --seasons "2019-2025" \
        --target "$stat" \
        --position "$pos" \
        --n-trials 20 \
        --out-dir "$OOF_DIR" \
        > "$LOG_DIR/${RUN_STAMP}_xgb_${stat}_${pos}.log" 2>&1; then
        mark_done xgb "$stat" "$pos"
        echo "  [XGB] ✓ $stat/$pos"
      else
        echo "  [XGB] SKIP: $stat/$pos (insufficient data or error — see log)"
        fail=1
      fi
    done
  done
  return $fail
}

run_lgbm() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  STEP 2 — LightGBM base learner  [PID $$]"
  echo "════════════════════════════════════════════════════════════════"
  local fail=0
  for pos in $POSITIONS; do
    case $pos in
      QB) STAT_SET="$QB_STATS" ;;
      RB) STAT_SET="$RB_STATS" ;;
      WR) STAT_SET="$WR_STATS" ;;
      TE) STAT_SET="$TE_STATS" ;;
    esac
    for stat in $STAT_SET; do
      if should_skip lgbm "$stat" "$pos"; then
        echo "  [LGB] SKIP (done): $stat/$pos"
        continue
      fi
      echo "→ [LGB] stat=$stat  position=$pos"
      if $PYTHON -m ml.lgbm_model \
        --seasons "2019-2025" \
        --target "$stat" \
        --position "$pos" \
        --n-trials 20 \
        --out-dir "$OOF_DIR" \
        > "$LOG_DIR/${RUN_STAMP}_lgbm_${stat}_${pos}.log" 2>&1; then
        mark_done lgbm "$stat" "$pos"
        echo "  [LGB] ✓ $stat/$pos"
      else
        echo "  [LGB] SKIP: $stat/$pos (insufficient data or error — see log)"
        fail=1
      fi
    done
  done
  return $fail
}

run_catboost() {
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  STEP 3 — CatBoost base learner  [PID $$]"
  echo "════════════════════════════════════════════════════════════════"
  local fail=0
  for pos in $POSITIONS; do
    case $pos in
      QB) STAT_SET="$QB_STATS" ;;
      RB) STAT_SET="$RB_STATS" ;;
      WR) STAT_SET="$WR_STATS" ;;
      TE) STAT_SET="$TE_STATS" ;;
    esac
    for stat in $STAT_SET; do
      if should_skip catboost "$stat" "$pos"; then
        echo "  [CB]  SKIP (done): $stat/$pos"
        continue
      fi
      echo "→ [CB]  stat=$stat  position=$pos"
      if $PYTHON -m ml.catboost_model \
        --seasons "2019-2025" \
        --target "$stat" \
        --position "$pos" \
        --n-trials 20 \
        --out-dir "$OOF_DIR" \
        > "$LOG_DIR/${RUN_STAMP}_catboost_${stat}_${pos}.log" 2>&1; then
        mark_done catboost "$stat" "$pos"
        echo "  [CB]  ✓ $stat/$pos"
      else
        echo "  [CB]  SKIP: $stat/$pos (insufficient data or error — see log)"
        fail=1
      fi
    done
  done
  return $fail
}

# Launch all three in parallel.
# set -e is disabled for this block so background job failures don't abort parent.
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEPS 1-3 — XGBoost + LightGBM + CatBoost (running in parallel)"
echo "  Logs: $LOG_DIR/${RUN_STAMP}_[xgb|lgbm|catboost]_*.log"
echo "════════════════════════════════════════════════════════════════"

set +e
run_xgb &
XGB_PID=$!
run_lgbm &
LGB_PID=$!
run_catboost &
CB_PID=$!

wait $XGB_PID; XGB_STATUS=$?
wait $LGB_PID; LGB_STATUS=$?
wait $CB_PID;  CB_STATUS=$?
set -e

echo ""
echo "─── Tree model training complete ───────────────────────────────"
[ $XGB_STATUS -eq 0 ] && echo "  XGBoost:   ✓ OK"   || echo "  XGBoost:   ✗ had errors (check logs)"
[ $LGB_STATUS -eq 0 ] && echo "  LightGBM:  ✓ OK"   || echo "  LightGBM:  ✗ had errors (check logs)"
[ $CB_STATUS  -eq 0 ] && echo "  CatBoost:  ✓ OK"   || echo "  CatBoost:  ✗ had errors (check logs)"
echo "────────────────────────────────────────────────────────────────"

# ── Step 4: TFT ────────────────────────────────────────────────────────────────
# TFT trains across all positions at once (--position all).
# --fast-mode: max_epochs=3, n_optuna_trials=0  (~2-4 hrs total)
# Full mode:   max_epochs=30, n_optuna_trials=20 (~18-22 hrs total)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 4 — Temporal Fusion Transformer (TFT)"
[[ "$FAST_MODE" == "true" ]] && echo "  (fast mode: max_epochs=3, 0 optuna trials)" || echo "  (full mode: max_epochs=30, 20 optuna trials)"
echo "════════════════════════════════════════════════════════════════"

# Build TFT flags based on mode
TFT_FAST_FLAG=""
TFT_TRIALS=20
[[ "$FAST_MODE" == "true" ]] && TFT_FAST_FLAG="--fast" && TFT_TRIALS=0

for stat in $STATS; do
  if [[ "$stat" == "passing_yards" ]]; then
    echo "  SKIP: tft $stat (disabled pending validation for QB passing-yards collapse)"
    continue
  fi
  if should_skip_tft "$stat"; then
    echo "  SKIP (done): tft $stat"
    continue
  fi
  echo "→ TFT: stat=$stat  (all positions)"
  if $PYTHON -m ml.tft_model \
    --seasons "2019-2025" \
    --target "$stat" \
    --position all \
    --n-trials "$TFT_TRIALS" \
    --out-dir "$OOF_DIR" \
    $TFT_FAST_FLAG \
    2>&1 | tee "$LOG_DIR/${RUN_STAMP}_tft_${stat}.log"; then
    mark_done tft "$stat" ""
  else
    echo "  SKIP: $stat (error — see log)"
  fi
done

# ── Step 5: Stacking (Ridge meta-learner) — PER POSITION ─────────────────────
# Train a separate Ridge per (stat, position) to eliminate cross-position
# intercept contamination. Cross-position Ridge is the root cause of negative
# XGB coefficients and biased intercepts (e.g. passing_yards intercept = -9
# when trained on QB+WR+RB+TE combined).
#
# Saves: ml/oof/ridge_{stat}_{position}_coefs.json for each valid combo.
# train.py _load_ridge_coefs() prefers the position-specific file.
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 5 — Ridge stacking meta-learner (per position)"
echo "════════════════════════════════════════════════════════════════"

for pos in $POSITIONS; do
  case $pos in
    QB) STAT_SET="$QB_STATS" ;;
    RB) STAT_SET="$RB_STATS" ;;
    WR) STAT_SET="$WR_STATS" ;;
    TE) STAT_SET="$TE_STATS" ;;
  esac

  for stat in $STAT_SET; do
    if should_skip stack "$stat" "$pos"; then
      echo "  SKIP (done): stacking $stat/$pos"
      continue
    fi
    echo "→ Stacking: stat=$stat  position=$pos"
    if $PYTHON -m ml.stacking_ensemble \
      --oof-dir "$OOF_DIR" \
      --target "$stat" \
      --position "$pos" \
      --out-dir "$OOF_DIR" \
      2>&1 | tee "$LOG_DIR/${RUN_STAMP}_stack_${stat}_${pos}.log"; then
      mark_done stack "$stat" "$pos"
    else
      echo "  SKIP: $stat/$pos (need ≥2 base-learner OOF files)"
    fi
  done
done

# ── Step 6: Re-run projection pipeline with real stacking ─────────────────────
# Now that ridge_*_coefs.json files exist, _run_stacking_step() in train.py
# will load real XGB+LGB+CB+TFT MLflow models instead of the Kalman proxy.
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 6 — Projection pipeline (with real stacking inference)"
# Pre-flight: warn loudly if MLflow is unreachable. Step 6 will use Kalman
# proxy fallback, but projections won't use the stacked ensemble models.
if ! curl -sf "${MLFLOW_TRACKING_URI}/health" > /dev/null 2>&1; then
  echo "  ⚠ WARNING: MLflow not reachable at ${MLFLOW_TRACKING_URI}"
  echo "  ⚠ Projections will use Kalman proxy only — NOT the full stacked ensemble."
  echo "  ⚠ Start MLflow (docker compose -f infra/docker-compose.yml up -d mlflow)"
  echo "  ⚠ then re-run Step 6:  bash ml/train_all_models.sh --fast-mode --resume"
fi
[[ "$FAST_MODE" == "true" ]] && echo "  (fast mode: Gaussian approximation — skips NUTS MCMC, ~30 min)" || echo "  (full mode: NUTS MCMC, 2 chains — ~4-8 hrs)"
echo "════════════════════════════════════════════════════════════════"

# --fast-mode: Gaussian approximation (no NUTS MCMC) — ~30 min
# Full mode:   NUTS MCMC, 500 samples, 2 chains     — ~4-8 hrs
TRAIN_FAST_FLAG=""
[[ "$FAST_MODE" == "true" ]] && TRAIN_FAST_FLAG="--fast"

$PYTHON -m ml.train \
  --seasons 2019 2020 2021 2022 2023 2024 2025 \
  --all-weeks \
  $TRAIN_FAST_FLAG \
  $([ "$RESUME" = "true" ] && echo "--resume")

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALL DONE"
echo "  Ridge coefs written to ml/oof/ridge_*_coefs.json"
echo "  Projections updated in the database"
echo "  MLflow runs visible at: $MLFLOW_TRACKING_URI"
echo "════════════════════════════════════════════════════════════════"
