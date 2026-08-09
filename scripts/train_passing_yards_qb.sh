#!/usr/bin/env bash
# Phase 2 closeout: regenerate QB passing_yards tree OOF + ridge stack, then diagnose vs DB.
set -euo pipefail
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
LOG="${LOG:-ml/logs/passing_yards_QB_$(date +%Y%m%dT%H%M%S).log}"
mkdir -p ml/logs ml/oof ml/checkpoints/done reports

# Stale .done markers without CSVs — force regenerate
rm -f ml/checkpoints/done/{xgb,lgbm,catboost}_passing_yards_QB.done
rm -f ml/checkpoints/done/tft_passing_yards.done
rm -f ml/checkpoints/done/stack_passing_yards_QB.done

exec >>"$LOG" 2>&1
echo "=== QB passing_yards OOF start $(date) ==="

for model in lgbm catboost; do
  echo ">>> $model passing_yards/QB $(date)"
  $PY -m "ml.${model}_model" --seasons "$SEASONS" --target passing_yards --position QB \
    --n-trials "$TRIALS" --out-dir ml/oof --no-mlflow
  touch "ml/checkpoints/done/${model}_passing_yards_QB.done"
done

echo ">>> stack passing_yards/QB exclude=tft,xgb $(date)"
$PY -m ml.stacking_ensemble --oof-dir ml/oof --target passing_yards --position QB \
  --exclude tft,xgb --out-dir ml/oof --no-mlflow
touch ml/checkpoints/done/stack_passing_yards_QB.done

STACK=$(ls -t ml/oof/stack_passing_yards_QB_*.csv | head -1)
echo "Diagnosing $STACK"
$PY scripts/diagnose_serving_divergence.py --stat passing_yards --position QB --oof "$STACK" \
  --json-out reports/serving_divergence_passing_yards_QB.json

# Causal eval if history available / build from OOF y_true
$PY - <<'PY'
from pathlib import Path
import pandas as pd
from ml.eval_causal import score_oof_against_baselines
stack = sorted(Path("ml/oof").glob("stack_passing_yards_QB_*.csv"))[-1]
oof = pd.read_csv(stack)
hist = oof.rename(columns={"y_true": "passing_yards"})[["player_id","season","week","passing_yards"]].copy()
# Need fuller history — try game_logs export if present
hpath = Path("ml/oof/_history_passing_yards_2019_2025.csv")
if not hpath.exists():
    hist.to_csv(hpath, index=False)
else:
    hist = pd.read_csv(hpath)
table = score_oof_against_baselines(oof, hist, stat="passing_yards", position="QB")
Path("reports").mkdir(exist_ok=True)
table.to_csv("reports/eval_causal_stack_passing_yards_QB.csv", index=False)
print(table.to_string(index=False))
PY

echo "=== QB passing_yards done $(date) ==="
