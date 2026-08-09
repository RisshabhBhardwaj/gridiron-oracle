#!/usr/bin/env bash
# Phase 5 restack: drop TFT (always) and XGB (toxic causal meta weights on short history).
# Keeps LGBM + CatBoost only. Archives prior 4-learner stacks first.
set -euo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
unset MLFLOW_TRACKING_URI || true
PY="${PY:-.venv_311/bin/python}"

STAMP=$(date +%Y%m%dT%H%M%S)
ARCH="ml/oof/_archive_stack4_${STAMP}"
mkdir -p "$ARCH"
for pos in QB RB WR TE; do
  for f in ml/oof/stack_fantasy_ppr_${pos}_*.csv ml/oof/ridge_fantasy_ppr_${pos}_coefs.json; do
    [[ -e "$f" ]] && cp -n "$f" "$ARCH/" || true
  done
done
echo "Archived prior stacks → $ARCH"

for pos in QB RB WR TE; do
  echo ">>> restack fantasy_ppr/$pos exclude=tft,xgb $(date)"
  rm -f "ml/checkpoints/done/stack_fantasy_ppr_${pos}.done"
  $PY -m ml.stacking_ensemble \
    --oof-dir ml/oof \
    --target fantasy_ppr \
    --position "$pos" \
    --out-dir ml/oof \
    --exclude tft,xgb \
    --no-mlflow
  touch "ml/checkpoints/done/stack_fantasy_ppr_${pos}.done"
done

echo "=== Phase 5 restack done $(date) ==="
ls -lh ml/oof/stack_fantasy_ppr_*.csv ml/oof/ridge_fantasy_ppr_*_coefs.json
