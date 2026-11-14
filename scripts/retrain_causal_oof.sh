#!/usr/bin/env bash
# Rebuild the causal walk-forward OOFs for every serving cell, with per-row
# bucket attributions, without touching serving artifacts, manifests, gates or
# materialized projections.
#
# For each cell in ml.artifact_manifest.REQUIRED_SERVING_CELLS (32):
#   LightGBM OOF + attributions -> CatBoost OOF + attributions -> Ridge stack
#   (+ blended attributions). Output goes to ml/oof/rebuild_<STAMP>/.
#
# Safe to stop and re-run with the same STAMP: a cell that already has its
# stack and attribution file is skipped, and a base learner whose OOF and
# attribution file exist is not retrained.
#
# Environment (all optional):
#   STAMP          run id, default now. Re-use it to resume.
#   ONLY_CELLS     space-separated "stat:POS" list, e.g. "receiving_yards:WR targets:TE".
#                  Default: all 32.
#   SEASONS        training range, default ml.season_constants.train_seasons_arg().
#   CONTRACT_SEASONS  seasons the feature contract is checked on (space-separated),
#                  default: every training season (2019-LAST_COMPLETE_SEASON).
#   N_TRIALS       Optuna trials per base learner, default 20.
#   TRAIN_OUT_ROOT persistent output root (the box's `train_out` volume). When set,
#                  OUT_DIR defaults to $TRAIN_OUT_ROOT/oof/rebuild_<STAMP> and
#                  LOG_DIR to $TRAIN_OUT_ROOT/logs/rebuild_<STAMP>.
#   OUT_DIR        output directory, default ml/oof/rebuild_<STAMP>.
#   LOG_DIR        log directory, default ml/logs/rebuild_<STAMP>.
#   PY             python, default .venv_311/bin/python.
#   DATABASE_URL   required unless the default local URL is right.
#   SKIP_CONTRACT  set to 1 to skip the feature-contract check (dry runs only).
#   STOP_FILE      path checked before each cell (default $TRAIN_OUT_ROOT/STOP when
#                  TRAIN_OUT_ROOT is set, else unused). If it exists the script
#                  exits 75 after the cell in progress, so the box can hand CPU
#                  back to its neighbours without losing a cell. Remove the file
#                  and re-run with the same STAMP to resume.
#
# macOS bash 3.2 compatible: no associative arrays.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv_311/bin/python}"
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

STAMP="${STAMP:-$(date +%Y%m%dT%H%M%S)}"
if [[ -n "${TRAIN_OUT_ROOT:-}" ]]; then
  OUT_DIR="${OUT_DIR:-$TRAIN_OUT_ROOT/oof/rebuild_${STAMP}}"
  LOG_DIR="${LOG_DIR:-$TRAIN_OUT_ROOT/logs/rebuild_${STAMP}}"
else
  OUT_DIR="${OUT_DIR:-ml/oof/rebuild_${STAMP}}"
  LOG_DIR="${LOG_DIR:-ml/logs/rebuild_${STAMP}}"
fi
N_TRIALS="${N_TRIALS:-20}"
SEASONS="${SEASONS:-$("$PY" -c 'from ml.season_constants import train_seasons_arg; print(train_seasons_arg())')}"
# The contract is checked on every season the trainers read (ml.season_constants.train_seasons),
# so a leak in the backfilled 2019-2023 rows cannot hide behind a shorter check.
CONTRACT_SEASONS="${CONTRACT_SEASONS:-$("$PY" -c 'from ml.season_constants import train_seasons; print(" ".join(map(str, train_seasons())))')}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Prove the source matrix is causal before spending hours training. This also
# prevents a stale target-game snap column from being hidden by a new allowlist.
if [[ "${SKIP_CONTRACT:-0}" != "1" ]]; then
  # shellcheck disable=SC2086
  "$PY" scripts/verify_feature_contract.py --seasons $CONTRACT_SEASONS
fi

ALL_CELLS=()
while IFS= read -r CELL; do
  [[ -n "$CELL" ]] && ALL_CELLS+=("$CELL")
done < <("$PY" -c 'from ml.artifact_manifest import REQUIRED_SERVING_CELLS as C; print("\n".join(f"{s}:{p}" for s, p in sorted(C)))')
(( ${#ALL_CELLS[@]} == 32 )) || { echo "Expected 32 serving cells; got ${#ALL_CELLS[@]}" >&2; exit 1; }

CELLS=()
if [[ -n "${ONLY_CELLS:-}" ]]; then
  for WANT in $ONLY_CELLS; do
    FOUND=0
    for CELL in "${ALL_CELLS[@]}"; do
      [[ "$CELL" == "$WANT" ]] && { CELLS+=("$CELL"); FOUND=1; }
    done
    (( FOUND == 1 )) || { echo "ONLY_CELLS entry '$WANT' is not a serving cell" >&2; exit 1; }
  done
else
  CELLS=("${ALL_CELLS[@]}")
fi

first_file() { find "$OUT_DIR" -maxdepth 1 -name "$1" -print -quit; }

train_base() {  # model stat pos
  local MODEL="$1" STAT="$2" POS="$3" OOF ATTR
  OOF="$(first_file "${MODEL}_${STAT}_${POS}_*.csv")"
  if [[ -n "$OOF" ]]; then
    ATTR="$OUT_DIR/attr_$(basename "${OOF%.csv}").parquet"
    if [[ -f "$ATTR" ]]; then echo "  $MODEL: already done"; return 0; fi
    echo "  $MODEL: OOF without attributions; retraining" >&2
    rm -f "$OOF" "$OOF.sha256" "${OOF%.csv}.csv.sha256"
  fi
  "$PY" -m "ml.${MODEL}_model" --seasons "$SEASONS" --target "$STAT" --position "$POS" \
    --n-trials "$N_TRIALS" --out-dir "$OUT_DIR" --no-mlflow --no-cache --attributions \
    >"$LOG_DIR/${MODEL}_${STAT}_${POS}.log" 2>&1
}

STOP_FILE="${STOP_FILE:-${TRAIN_OUT_ROOT:+$TRAIN_OUT_ROOT/STOP}}"

for CELL in "${CELLS[@]}"; do
  if [[ -n "$STOP_FILE" && -e "$STOP_FILE" ]]; then
    echo "stop file $STOP_FILE present: pausing before $CELL (re-run with STAMP=$STAMP to resume)" >&2
    exit 75
  fi
  STAT="${CELL%%:*}"
  POS="${CELL##*:}"
  echo "=== $STAT/$POS ==="
  T0=$(date +%s)
  if [[ -n "$(first_file "stack_${STAT}_${POS}_*.csv")" ]] && ls "$OUT_DIR"/attr_stack_"${STAT}"_"${POS}"_*.parquet >/dev/null 2>&1; then
    echo "  stack: already done"
    continue
  fi
  train_base lgbm "$STAT" "$POS"
  train_base catboost "$STAT" "$POS"
  LGBM="$(first_file "lgbm_${STAT}_${POS}_*.csv")"
  CATBOOST="$(first_file "catboost_${STAT}_${POS}_*.csv")"
  [[ -n "$LGBM" && -n "$CATBOOST" ]] || { echo "Missing base OOF for $STAT/$POS" >&2; exit 1; }
  "$PY" -m ml.stacking_ensemble --oof "$LGBM" "$CATBOOST" --target "$STAT" \
    --position "$POS" --out-dir "$OUT_DIR" --no-mlflow --attributions \
    >"$LOG_DIR/stack_${STAT}_${POS}.log" 2>&1
  T1=$(date +%s)
  printf '%s\t%s\t%s\n' "$STAT:$POS" "$((T1 - T0))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$OUT_DIR/TIMINGS.tsv"
done

find "$OUT_DIR" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.parquet' \) -print0 | sort -z | bash scripts/sha256_files.sh >"$OUT_DIR/SHA256SUMS"
echo "Causal OOF rebuild complete: $OUT_DIR"
echo "Not served: release manifest and projection materialization were intentionally untouched."
