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
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:15091}"
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
PURGE_OOF=false
for arg in "$@"; do
  [[ "$arg" == "--resume" ]]    && RESUME=true
  [[ "$arg" == "--fast-mode" ]] && FAST_MODE=true
  [[ "$arg" == "--purge-oof" ]] && PURGE_OOF=true
done
CHECKPOINT_DIR="ml/checkpoints/done"
mkdir -p "$CHECKPOINT_DIR"

# ── Clean slate is opt-in, and never destroys release artifacts ───────────────
#
# A normal (non---resume) start used to run `rm -f "$OOF_DIR"/*.csv`, which
# deletes the entire *serving* artifact set — every tracked stack the release
# manifest pins and the shipped projections were materialized from. "Clean
# start" and "destroy the release" were spelled the same way, and the recovery
# path for a deleted stack is restore-from-archive, not regeneration.
#
# Checkpoints are cheap to rebuild and pin nothing, so a non-resume run still
# clears those. Purging OOF CSVs now requires --purge-oof *and* passes the
# manifest guard, which refuses if any pinned artifact is in the blast radius.
if [[ "$RESUME" == "false" ]]; then
  echo "Clean start: clearing checkpoints (OOF artifacts are preserved)."
  rm -f "$CHECKPOINT_DIR"/*.done
fi

if [[ "$PURGE_OOF" == "true" ]]; then
  echo "--purge-oof requested: checking $OOF_DIR against the release manifest…"
  if ! "$PYTHON" scripts/guard_release_artifacts.py --check-purge "$OOF_DIR"; then
    echo "ABORTING: refusing to purge release artifacts. See the list above." >&2
    exit 1
  fi
  echo "Guard passed; purging OOF CSVs in $OOF_DIR."
  rm -f "$OOF_DIR"/*.csv
fi

# Expected output artifact for a (step, stat, pos) cell. Base learners write
# {learner}_{stat}_{pos}_*.csv; the stacking step writes stack_{stat}_{pos}_*.csv.
cell_artifact_glob() {
  local step=$1 stat=$2 pos=${3:-}
  if [[ "$step" == "stack" ]]; then
    echo "$OOF_DIR/stack_${stat}_${pos}_*.csv"
  else
    echo "$OOF_DIR/${step}_${stat}_${pos}_*.csv"
  fi
}

cell_artifact_exists() {
  local step=$1 stat=$2 pos=${3:-}
  # shellcheck disable=SC2086 # deliberate glob expansion
  compgen -G "$(cell_artifact_glob "$step" "$stat" "$pos")" >/dev/null 2>&1
}

# A checkpoint alone is not evidence the work produced anything. A `.done` marker
# left behind by an interrupted or purged run made every resume skip a cell whose
# CSV did not exist, and the script still exited 0. Require both.
should_skip() {
  local step=$1 stat=$2 pos=${3:-}
  [[ "$RESUME" != "true" ]] && return 1
  local marker
  if [[ -n "$pos" ]]; then
    marker="$CHECKPOINT_DIR/${step}_${stat}_${pos}.done"
  else
    marker="$CHECKPOINT_DIR/${step}_${stat}.done"
  fi
  [[ -f "$marker" ]] || return 1
  if cell_artifact_exists "$step" "$stat" "$pos"; then
    return 0
  fi
  echo "  [!] stale checkpoint $(basename "$marker") — no artifact on disk; re-running."
  rm -f "$marker"
  return 1
}

# Checkpoint only after the artifact is on disk. Written the other way round, the
# marker is a claim that a later resume trusts without rechecking.
mark_done() {
  local step=$1 stat=$2 pos=${3:-}
  if ! cell_artifact_exists "$step" "$stat" "$pos"; then
    echo "  [!] refusing to checkpoint ${step}/${stat}/${pos}: no artifact matching $(cell_artifact_glob "$step" "$stat" "$pos")" >&2
    return 1
  fi
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

# ── Steps 1-2: LightGBM + CatBoost (parallel) ────────────────────────────────
# Each model type is wrapped in a shell function and launched as a background
# process, so both train simultaneously.
#
# XGBoost and TFT are deliberately absent. The Phase-5 contract is
# lgbm + catboost only: TFT for MAE drag (~7.8) and XGB for toxic causal-meta
# weights on short history plus the 2022 QB/RB/WR OOF collapse. Every shipped
# ridge_*_coefs.json is a clean two-learner fit.
#
# This script used to train all four and then stack them, so one run of the
# documented recovery path rewrote the clean coef files with xgb/tft keys and
# inference began executing killed learners. Removing the training blocks is only
# half the fix — the allowlist is enforced inside ml.stacking_ensemble (see
# ml/artifact_manifest.py), which raises if a killed learner's OOF is even
# present in the discovery directory.

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
echo "  STEPS 1-2 — LightGBM + CatBoost (running in parallel)"
echo "  Logs: $LOG_DIR/${RUN_STAMP}_[lgbm|catboost]_*.log"
echo "════════════════════════════════════════════════════════════════"

set +e
run_lgbm &
LGB_PID=$!
run_catboost &
CB_PID=$!

wait $LGB_PID; LGB_STATUS=$?
wait $CB_PID;  CB_STATUS=$?
set -e

echo ""
echo "─── Tree model training complete ───────────────────────────────"
[ $LGB_STATUS -eq 0 ] && echo "  LightGBM:  ✓ OK"   || echo "  LightGBM:  ✗ had errors (check logs)"
[ $CB_STATUS  -eq 0 ] && echo "  CatBoost:  ✓ OK"   || echo "  CatBoost:  ✗ had errors (check logs)"
echo "────────────────────────────────────────────────────────────────"

# Step 3 (TFT) removed with Step 1 (XGB): see the Phase-5 note above the
# run_lgbm definition. Both learners are killed, so training them produces OOFs
# that the stacker now refuses to consume.

# ── Step 3: Stacking (Ridge meta-learner) — PER POSITION ─────────────────────
# Train a separate Ridge per (stat, position) to eliminate cross-position
# intercept contamination. Cross-position Ridge is the root cause of negative
# XGB coefficients and biased intercepts (e.g. passing_yards intercept = -9
# when trained on QB+WR+RB+TE combined).
#
# Saves: ml/oof/ridge_{stat}_{position}_coefs.json for each valid combo.
# train.py _load_ridge_coefs() prefers the position-specific file.
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 3 — Ridge stacking meta-learner (per position)"
echo "════════════════════════════════════════════════════════════════"

# Cells that failed or produced nothing. The loop used to print "SKIP" on
# failure and carry on, so the script exited 0 with missing stacks; the final
# expected-cell check below turns that into a non-zero exit.
STACK_FAILED=()

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
      # mark_done itself refuses to checkpoint without an artifact.
      if ! mark_done stack "$stat" "$pos"; then
        STACK_FAILED+=("${stat}/${pos} (exit 0 but no artifact)")
      fi
    else
      echo "  FAILED: stacking $stat/$pos — see $LOG_DIR/${RUN_STAMP}_stack_${stat}_${pos}.log"
      STACK_FAILED+=("${stat}/${pos}")
    fi
  done
done

# ── Expected-cell matrix ──────────────────────────────────────────────────────
# A long job must not report success with cells missing.
#
# The *required* set is the release cell matrix from the manifest, not the
# stat×position cross product the loops above iterate. Those loops walk 37
# combinations, but many are irrelevant by design (QB receiving_yards, and
# anything with too little data for ≥2 base learners), so the cross product is
# not a statement about what must exist — the release manifest is. Requiring all
# 37 would make this script exit 1 on a correct tree and never reach Step 4.
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Expected-cell verification (release matrix)"
echo "════════════════════════════════════════════════════════════════"

REQUIRED_CELLS=()
while IFS= read -r cell; do
  [[ -n "$cell" ]] && REQUIRED_CELLS+=("$cell")
done < <("$PYTHON" scripts/guard_release_artifacts.py --list-cells)

if (( ${#REQUIRED_CELLS[@]} == 0 )); then
  echo "ABORTING: could not read the release cell matrix from the manifest." >&2
  exit 1
fi
echo "Release matrix declares ${#REQUIRED_CELLS[@]} required cells."

MISSING_CELLS=()
for cell in "${REQUIRED_CELLS[@]}"; do
  stat="${cell%%:*}"; pos="${cell##*:}"
  cell_artifact_exists stack "$stat" "$pos" || MISSING_CELLS+=("${stat}/${pos}")
done

# Failures outside the release matrix are reported but not fatal: a cell that
# never had enough data to stack is not a regression.
if (( ${#STACK_FAILED[@]} > 0 )); then
  echo "Stacking cells that did not produce an artifact (${#STACK_FAILED[@]}):"
  printf '  %s\n' "${STACK_FAILED[@]}"
  echo "  (fatal only for cells in the release matrix — see below)"
fi
if (( ${#MISSING_CELLS[@]} > 0 )); then
  echo "" >&2
  echo "Missing REQUIRED stack artifacts (${#MISSING_CELLS[@]} of ${#REQUIRED_CELLS[@]}):" >&2
  printf '  %s\n' "${MISSING_CELLS[@]}" >&2
  echo "Refusing to report success with an incomplete release cell matrix." >&2
  exit 1
fi
echo "All ${#REQUIRED_CELLS[@]} required release cells have stack artifacts."

# ── Step 4: Re-run projection pipeline with real stacking ─────────────────────
# Now that ridge_*_coefs.json files exist, _run_stacking_step() in train.py
# loads the real LGBM + CatBoost MLflow models instead of the Kalman proxy.
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  STEP 4 — Projection pipeline (with real stacking inference)"
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
