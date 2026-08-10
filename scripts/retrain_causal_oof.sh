#!/usr/bin/env bash
# Rebuild the release-cell base OOFs and two-learner stacks without touching
# serving artifacts, manifests, gates, or materialized projections.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv_311/bin/python}"
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

STAMP="${STAMP:-$(date +%Y%m%dT%H%M%S)}"
OUT_DIR="ml/oof/rebuild_${STAMP}"
LOG_DIR="ml/logs/rebuild_${STAMP}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Prove the source matrix is causal before spending hours training.  This also
# prevents a stale target-game snap column from being hidden by a new allowlist.
"$PY" scripts/verify_feature_contract.py --seasons 2021 2022 2023 2024 2025

CELLS=()
while IFS= read -r CELL; do
  [[ -n "$CELL" ]] && CELLS+=("$CELL")
done < <("$PY" scripts/guard_release_artifacts.py --list-cells)
(( ${#CELLS[@]} == 15 )) || { echo "Expected 15 release cells; got ${#CELLS[@]}" >&2; exit 1; }

for CELL in "${CELLS[@]}"; do
  STAT="${CELL%%:*}"
  POS="${CELL##*:}"
  echo "=== $STAT/$POS ==="
  "$PY" -m ml.lgbm_model --seasons "2019-2025" --target "$STAT" --position "$POS" \
    --n-trials 20 --out-dir "$OUT_DIR" --no-mlflow --no-cache >"$LOG_DIR/lgbm_${STAT}_${POS}.log" 2>&1
  "$PY" -m ml.catboost_model --seasons "2019-2025" --target "$STAT" --position "$POS" \
    --n-trials 20 --out-dir "$OUT_DIR" --no-mlflow --no-cache >"$LOG_DIR/catboost_${STAT}_${POS}.log" 2>&1
  LGBM=$(find "$OUT_DIR" -maxdepth 1 -name "lgbm_${STAT}_${POS}_*.csv" -print -quit)
  CATBOOST=$(find "$OUT_DIR" -maxdepth 1 -name "catboost_${STAT}_${POS}_*.csv" -print -quit)
  [[ -n "$LGBM" && -n "$CATBOOST" ]] || { echo "Missing base OOF for $STAT/$POS" >&2; exit 1; }
  "$PY" -m ml.stacking_ensemble --oof "$LGBM" "$CATBOOST" --target "$STAT" \
    --position "$POS" --out-dir "$OUT_DIR" --no-mlflow >"$LOG_DIR/stack_${STAT}_${POS}.log" 2>&1
done

find "$OUT_DIR" -maxdepth 1 -type f -name '*.csv' -print0 | sort -z | xargs -0 shasum -a 256 >"$OUT_DIR/SHA256SUMS"
echo "Causal OOF rebuild complete: $OUT_DIR"
echo "Not served: release manifest and projection materialization were intentionally untouched."
