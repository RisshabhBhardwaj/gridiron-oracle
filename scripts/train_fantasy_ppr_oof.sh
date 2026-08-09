#!/usr/bin/env bash
# Generate causal OOF for fantasy_ppr across positions (LAST_COMPLETE seasons only).
# Skips checkpoints; --no-mlflow; append-only log (nohup-safe).
set -uo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
unset MLFLOW_TRACKING_URI || true

PY="${PY:-.venv_311/bin/python}"
SEASONS=2019-2025
mkdir -p ml/logs ml/oof ml/checkpoints/done
LOG="${LOG:-ml/logs/fantasy_ppr_oof_$(date +%Y%m%dT%H%M%S).log}"

exec >>"$LOG" 2>&1
echo "=== fantasy_ppr OOF generation start $(date) ==="
echo "log=$LOG seasons=$SEASONS db=$DATABASE_URL no_mlflow=1 pid=$$"

mark() { touch "ml/checkpoints/done/$1"; }
done_ck() { [[ -f "ml/checkpoints/done/$1.done" ]]; }

run_tree() {
  local model="$1" pos="$2" mod="$3"
  local ck="${model}_fantasy_ppr_${pos}"
  if done_ck "$ck"; then
    echo "SKIP $ck (checkpoint present)"
    return 0
  fi
  echo ">>> $model fantasy_ppr/$pos $(date)"
  if $PY -m "$mod" --seasons "$SEASONS" --target fantasy_ppr --position "$pos" \
      --n-trials 20 --out-dir ml/oof --no-mlflow; then
    mark "$ck"
    echo "OK $ck $(date)"
  else
    echo "FAILED $model $pos exit=$?"
  fi
}

for pos in QB RB WR TE; do
  run_tree xgb "$pos" ml.xgb_model
  run_tree lgbm "$pos" ml.lgbm_model
  run_tree catboost "$pos" ml.catboost_model
done

if done_ck "tft_fantasy_ppr"; then
  echo "SKIP tft_fantasy_ppr (checkpoint present)"
else
  echo ">>> tft fantasy_ppr $(date)"
  if $PY -m ml.tft_model --seasons "$SEASONS" --target fantasy_ppr --position all \
      --n-trials 0 --out-dir ml/oof --fast --no-mlflow; then
    mark "tft_fantasy_ppr"
    echo "OK tft_fantasy_ppr $(date)"
  else
    echo "FAILED tft exit=$?"
  fi
fi

for pos in QB RB WR TE; do
  ck="stack_fantasy_ppr_${pos}"
  if done_ck "$ck"; then
    echo "SKIP $ck (checkpoint present)"
    continue
  fi
  echo ">>> stacking fantasy_ppr/$pos $(date)"
  if $PY -m ml.stacking_ensemble --oof-dir ml/oof --target fantasy_ppr --position "$pos" \
      --out-dir ml/oof --no-mlflow; then
    mark "$ck"
    echo "OK $ck $(date)"
  else
    echo "FAILED stack $pos exit=$?"
  fi
done

echo "=== fantasy_ppr OOF generation done $(date) ==="
ls -lh ml/oof/*fantasy_ppr* 2>/dev/null || true
ls ml/checkpoints/done/*fantasy* 2>/dev/null || true
