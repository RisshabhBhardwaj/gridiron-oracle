#!/usr/bin/env bash
# Volume-target OOF (causal): targets (WR/TE/RB), carries (RB), pass_attempts (QB).
set -uo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
unset MLFLOW_TRACKING_URI || true
PY="${PY:-.venv_311/bin/python}"
SEASONS=2019-2025
TRIALS="${TRIALS:-10}"
LOG="${LOG:-ml/logs/volume_oof_$(date +%Y%m%dT%H%M%S).log}"
mkdir -p ml/logs ml/oof ml/checkpoints/done
exec >>"$LOG" 2>&1
echo "=== volume OOF start $(date) ==="

run_tree() {
  local model="$1" target="$2" pos="$3"
  local ck="${model}_${target}_${pos}"
  # Require an actual OOF CSV — stale .done markers without artifacts must not skip.
  if [[ -f "ml/checkpoints/done/${ck}.done" ]] && ls ml/oof/${model}_${target}_${pos}_*.csv >/dev/null 2>&1; then
    echo "SKIP $ck"
    return 0
  fi
  rm -f "ml/checkpoints/done/${ck}.done"
  echo ">>> $model $target/$pos $(date)"
  if $PY -m "ml.${model}_model" --seasons "$SEASONS" --target "$target" --position "$pos" \
      --n-trials "$TRIALS" --out-dir ml/oof --no-mlflow; then
    touch "ml/checkpoints/done/${ck}.done"
    echo "OK $ck $(date)"
  else
    echo "FAILED $ck exit=$?"
  fi
}

for model in lgbm catboost; do
  run_tree "$model" targets WR
  run_tree "$model" targets TE
  run_tree "$model" targets RB
  run_tree "$model" carries RB
  run_tree "$model" pass_attempts QB
done

for pos_target in "WR:targets" "TE:targets" "RB:targets" "RB:carries" "QB:pass_attempts"; do
  pos="${pos_target%%:*}"
  target="${pos_target##*:}"
  ck="stack_${target}_${pos}"
  if [[ -f "ml/checkpoints/done/${ck}.done" ]] && ls ml/oof/stack_${target}_${pos}_*.csv >/dev/null 2>&1; then
    echo "SKIP $ck"; continue
  fi
  rm -f "ml/checkpoints/done/${ck}.done"
  echo ">>> stack $target/$pos $(date)"
  # Explicit newest tree OOFs — avoid stale combined/xgb/tft pollution
  lgbm=$(ls -t ml/oof/lgbm_${target}_${pos}_*.csv 2>/dev/null | head -1)
  catb=$(ls -t ml/oof/catboost_${target}_${pos}_*.csv 2>/dev/null | head -1)
  if [[ -z "$lgbm" || -z "$catb" ]]; then
    echo "FAILED $ck missing tree OOF (lgbm=$lgbm cat=$catb)"
    continue
  fi
  if $PY -m ml.stacking_ensemble --oof "$lgbm" "$catb" --target "$target" --position "$pos" \
      --out-dir ml/oof --no-mlflow; then
    touch "ml/checkpoints/done/${ck}.done"
    echo "OK $ck $(date)"
  else
    echo "FAILED $ck exit=$?"
  fi
done

echo "=== volume OOF done $(date) ==="
ls -lh ml/oof/*{targets,carries,pass_attempts}* 2>/dev/null || true
