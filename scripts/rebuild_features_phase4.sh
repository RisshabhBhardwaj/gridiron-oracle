#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
PY="${PY:-.venv_311/bin/python}"
LOG="ml/logs/feature_rebuild_phase4_seq.log"
mkdir -p ml/logs
exec > >(tee -a "$LOG") 2>&1
echo "=== sequential feature rebuild start $(date) ==="
for y in 2020 2021 2022 2023 2024 2025; do
  echo ">>> season $y $(date)"
  $PY -m pipeline.feature_engineer --seasons "$y"
  echo "<<< season $y done $(date)"
done
echo "=== sequential feature rebuild complete $(date) ==="
