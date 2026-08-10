#!/usr/bin/env bash
# Phase 5 restack: drop TFT (always) and XGB (toxic causal meta weights on short history).
# Keeps LGBM + CatBoost only. Archives prior 4-learner stacks first.
set -euo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
unset MLFLOW_TRACKING_URI || true
PY="${PY:-.venv_311/bin/python}"

# Archive OUT of the serving directory, rather than copying it.
#
# This used `cp -n`, so every "archived" four-learner stack stayed in ml/oof/
# beside its replacement. That is the direct cause of the poisoning: the stale
# stack remained visible to every discovery glob and every mtime-based selector,
# so a fresh clone or a stray `touch` would serve it. `mv` is the whole fix — an
# archived artifact has to actually leave.
#
# The .sha256 sidecar moves with its CSV; leaving it behind orphans a digest
# that describes nothing in the directory.
STAMP=$(date +%Y%m%dT%H%M%S)
ARCH="ml/oof/_archive_stack4_${STAMP}"
mkdir -p "$ARCH"
for pos in QB RB WR TE; do
  for f in ml/oof/stack_fantasy_ppr_${pos}_*.csv \
           ml/oof/stack_fantasy_ppr_${pos}_*.csv.sha256 \
           ml/oof/ridge_fantasy_ppr_${pos}_coefs.json; do
    [[ -e "$f" ]] || continue
    mv "$f" "$ARCH/"
  done
done
echo "Archived prior stacks → $ARCH ($(find "$ARCH" -type f | wc -l | tr -d ' ') files moved out of ml/oof/)"

for pos in QB RB WR TE; do
  echo ">>> restack fantasy_ppr/$pos $(date)"
  rm -f "ml/checkpoints/done/stack_fantasy_ppr_${pos}.done"
  # No --exclude: the two-learner allowlist is enforced inside
  # ml.stacking_ensemble (ml/artifact_manifest.py), so it cannot be lost by
  # forgetting a flag here. The stacker raises if a killed learner's OOF is
  # present in the discovery directory.
  $PY -m ml.stacking_ensemble \
    --oof-dir ml/oof \
    --target fantasy_ppr \
    --position "$pos" \
    --out-dir ml/oof \
    --no-mlflow
  # Checkpoint only after the artifact exists — a marker written first is a
  # claim a later resume trusts without rechecking.
  ls ml/oof/stack_fantasy_ppr_${pos}_*.csv >/dev/null
  touch "ml/checkpoints/done/stack_fantasy_ppr_${pos}.done"
done

echo "=== Phase 5 restack done $(date) ==="
ls -lh ml/oof/stack_fantasy_ppr_*.csv ml/oof/ridge_fantasy_ppr_*_coefs.json
