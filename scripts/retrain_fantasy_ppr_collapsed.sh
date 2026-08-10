#!/usr/bin/env bash
# Retrain fantasy_ppr trees for positions whose 2022 OOF collapsed to a near-constant.
# Then Phase-5 restack (lgbm+catboost only).
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
LOG="${LOG:-ml/logs/fantasy_ppr_retrain_collapsed_$(date +%Y%m%dT%H%M%S).log}"
mkdir -p ml/logs ml/oof ml/checkpoints/done

# Bust feature caches so 2022 fold sees current matrix
rm -f ml/cache/feature_matrix_*.parquet

exec >>"$LOG" 2>&1
echo "=== collapsed-fold retrain start $(date) log=$LOG trials=$TRIALS ==="

for pos in QB RB WR; do
  for model in lgbm catboost; do
    mod="ml.${model}_model"
    ck="${model}_fantasy_ppr_${pos}"
    echo ">>> $model fantasy_ppr/$pos $(date)"
    $PY -m "$mod" --seasons "$SEASONS" --target fantasy_ppr --position "$pos" \
      --n-trials "$TRIALS" --out-dir ml/oof --no-mlflow
    touch "ml/checkpoints/done/${ck}.done"
    echo "OK $ck $(date)"
  done
done

# Collapse guard: refuse to restack if any 2022 fold still near-constant
$PY - <<'PY'
import sys
from pathlib import Path
import pandas as pd
failed = []
for pos in ["QB", "RB", "WR"]:
    for learner in ["lgbm", "catboost"]:
        paths = sorted(Path("ml/oof").glob(f"{learner}_fantasy_ppr_{pos}_*.csv"))
        paths = [p for p in paths if "_archive" not in str(p) and "combined" not in p.name]
        if not paths:
            failed.append(f"{learner}/{pos}: missing OOF")
            continue
        # Latest by dated filename (paths is already sorted), not mtime: the
        # guard must inspect the same file a restack would consume, and mtime
        # makes that depend on touches and checkout order rather than on which
        # run is genuinely newest.
        path = paths[-1]
        df = pd.read_csv(path)
        g = df[df["season"] == 2022]
        nunique = g["y_pred"].nunique()
        std = float(g["y_pred"].std()) if len(g) else 0.0
        print(f"{path.name} 2022 unique={nunique} std={std:.3f}")
        if nunique < max(20, len(g) // 50) or std < 0.5:
            failed.append(f"{path.name}: collapse unique={nunique} std={std:.3f}")
if failed:
    print("COLLAPSE GUARD FAILED:", *failed, sep="\n  ")
    sys.exit(2)
print("Collapse guard OK")
PY

bash scripts/rebuild_fantasy_ppr_stack_phase5.sh
echo "=== collapsed-fold retrain done $(date) ==="
