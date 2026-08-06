#!/usr/bin/env bash
# Generate causal OOF for fantasy_ppr across positions (LAST_COMPLETE seasons only).
set -euo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:15091}"

PY="${PY:-.venv_311/bin/python}"
SEASONS=2019-2025
mkdir -p ml/logs ml/oof ml/checkpoints/done
LOG="ml/logs/fantasy_ppr_oof_$(date +%Y%m%dT%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "=== fantasy_ppr OOF generation start $(date) ==="
echo "log=$LOG seasons=$SEASONS db=$DATABASE_URL"

mark() { touch "ml/checkpoints/done/$1"; }

for pos in QB RB WR TE; do
  echo ">>> xgb fantasy_ppr/$pos $(date)"
  $PY -m ml.xgb_model --seasons "$SEASONS" --target fantasy_ppr --position "$pos" --n-trials 20 --out-dir ml/oof \
    && mark "xgb_fantasy_ppr_${pos}.done" || echo "FAILED xgb $pos"

  echo ">>> lgbm fantasy_ppr/$pos $(date)"
  $PY -m ml.lgbm_model --seasons "$SEASONS" --target fantasy_ppr --position "$pos" --n-trials 20 --out-dir ml/oof \
    && mark "lgbm_fantasy_ppr_${pos}.done" || echo "FAILED lgbm $pos"

  echo ">>> catboost fantasy_ppr/$pos $(date)"
  $PY -m ml.catboost_model --seasons "$SEASONS" --target fantasy_ppr --position "$pos" --n-trials 20 --out-dir ml/oof \
    && mark "catboost_fantasy_ppr_${pos}.done" || echo "FAILED catboost $pos"
done

echo ">>> tft fantasy_ppr $(date)"
$PY -m ml.tft_model --seasons "$SEASONS" --target fantasy_ppr --position all --n-trials 0 --out-dir ml/oof --fast \
  && mark "tft_fantasy_ppr.done" || echo "FAILED tft"

for pos in QB RB WR TE; do
  echo ">>> stacking fantasy_ppr/$pos $(date)"
  $PY -m ml.stacking_ensemble --oof-dir ml/oof --target fantasy_ppr --position "$pos" --out-dir ml/oof \
    && mark "stack_fantasy_ppr_${pos}.done" || echo "FAILED stack $pos"
done

echo "=== fantasy_ppr OOF generation done $(date) ==="
ls -lh ml/oof/*fantasy_ppr* 2>/dev/null || true
