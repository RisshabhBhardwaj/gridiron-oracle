#!/usr/bin/env bash
# Training-window recipes, run INSIDE the box's `training` container (/app).
# The host wrappers infra/hosts/vps1/scripts/train_{small,full}.sh call this via
# TRAIN_SMALL_CMD / TRAIN_FULL_CMD; env assignments cannot ride those variables
# (they are word-split, not shell-parsed), so the recipe lives here.
#
#   scripts/train_window.sh full    resumable 32-cell causal OOF rebuild with
#                                   per-row bucket attributions (Fri-Mon window)
#   scripts/train_window.sh small   canary: 3 cells, few trials, nightly. Proves
#                                   the trainers, DB and attribution path run on
#                                   the box. Output is never promoted.
#   RELEASE_ID=causal_20261114 scripts/train_window.sh finalfit
#                                   final fit of the pinned base models (decision
#                                   0012): replay gate, then one model per learner
#                                   with non-zero ridge weight, then MODELS.json.
#                                   Reads params_*.json and the OOF of the last
#                                   complete rebuild (or OOF_STAMP=...); never runs
#                                   Optuna. ONLY_CELLS="passing_yards:QB ..." fits
#                                   only those cells and does not write MODELS.json
#                                   (the 3-cell slice). CODE_GIT_SHA=<sha> is needed
#                                   because the box is not a git checkout.
#
# Output lands in $TRAIN_OUT_ROOT (the `train_out` volume, default /train_out):
#   oof/rebuild_<STAMP>/   full rebuild (stacks, ridge coefs, attr_*.parquet,
#                          TIMINGS.tsv, SHA256SUMS)    logs/rebuild_<STAMP>/
#   oof/small_<date>/      canary output (last 3 kept)
#   final/<RELEASE_ID>/    finalfit output: models, MODELS.json, .entries/
#   full_stamp             STAMP of the rebuild in progress; a killed window
#                          resumes it next time (per-cell resume).
#   last_full_stamp        STAMP of the last COMPLETE rebuild.
# Promotion (select -> promote -> freeze) is a manager step, never automatic.
#
# One trainer at a time: an flock on the volume makes a second window exit 75
# (EX_TEMPFAIL) without pinging its dead-man URL.
# bash 3.2 compatible (no associative arrays).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
case "$MODE" in full | small | finalfit) ;; *) echo "usage: $0 full|small|finalfit" >&2; exit 2 ;; esac

export TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:-/train_out}"
export PY="${PY:-python3}"
mkdir -p "$TRAIN_OUT_ROOT/oof" "$TRAIN_OUT_ROOT/logs"

exec 9>"$TRAIN_OUT_ROOT/.train.lock"
if ! flock -n 9; then
  echo "another training run holds $TRAIN_OUT_ROOT/.train.lock; skipping this window" >&2
  exit 75
fi

if [[ "$MODE" == "finalfit" ]]; then
  if [[ -z "${RELEASE_ID:-}" ]]; then
    echo "finalfit needs RELEASE_ID, e.g. RELEASE_ID=causal_20261114" >&2
    exit 2
  fi
  RELEASE_DIR="releases/candidates/$RELEASE_ID"
  [[ -d "$RELEASE_DIR" ]] || { echo "no release directory $RELEASE_DIR (is the repo synced?)" >&2; exit 1; }
  OOF_STAMP="${OOF_STAMP:-$(cat "$TRAIN_OUT_ROOT/last_full_stamp" 2>/dev/null || true)}"
  if [[ -z "$OOF_STAMP" ]]; then
    echo "no complete rebuild: $TRAIN_OUT_ROOT/last_full_stamp is missing; pass OOF_STAMP=<stamp>" >&2
    exit 1
  fi
  SRC="$TRAIN_OUT_ROOT/oof/rebuild_$OOF_STAMP"
  OUT="${FINAL_OUT_DIR:-$TRAIN_OUT_ROOT/final/$RELEASE_ID}"
  mkdir -p "$OUT"
  CELL_ARGS=()
  if [[ -n "${ONLY_CELLS:-}" ]]; then
    # shellcheck disable=SC2206
    CELL_ARGS=(--cells $ONLY_CELLS)
  fi
  "$PY" -m ml.final_fit fit --release-dir "$RELEASE_DIR" --params-dir "$SRC" --oof-dir "$SRC" \
    --out-dir "$OUT" --release-id "$RELEASE_ID" ${CELL_ARGS[@]+"${CELL_ARGS[@]}"}
  if [[ -z "${ONLY_CELLS:-}" ]]; then
    "$PY" -m ml.final_fit assemble --release-dir "$RELEASE_DIR" --out-dir "$OUT" --release-id "$RELEASE_ID"
    echo "final fit complete: $OUT (MODELS.json written)"
  else
    echo "partial final fit (ONLY_CELLS): $OUT; MODELS.json not written"
  fi
elif [[ "$MODE" == "full" ]]; then
  POINTER="$TRAIN_OUT_ROOT/full_stamp"
  if [[ -s "$POINTER" ]]; then
    STAMP="$(cat "$POINTER")"
    echo "resuming rebuild $STAMP"
  else
    STAMP="$(date -u +%Y%m%dT%H%M%S)"
    echo "$STAMP" >"$POINTER"
    echo "starting rebuild $STAMP"
  fi
  # A subset run must never be recorded as a complete rebuild.
  unset ONLY_CELLS
  # `touch $TRAIN_OUT_ROOT/STOP` pauses after the current cell (exit 75, pointer
  # kept); delete it and start the same recipe again to resume.
  STAMP="$STAMP" N_TRIALS="${N_TRIALS:-20}" bash scripts/retrain_causal_oof.sh
  N_STACKS="$(find "$TRAIN_OUT_ROOT/oof/rebuild_$STAMP" -maxdepth 1 -name 'stack_*.csv' | wc -l | tr -d ' ')"
  if [[ "$N_STACKS" != "32" ]]; then
    echo "rebuild $STAMP has $N_STACKS stack files, expected 32; not marking it complete" >&2
    exit 1
  fi
  echo "$STAMP" >"$TRAIN_OUT_ROOT/last_full_stamp"
  rm -f "$POINTER"
  echo "full rebuild complete: $TRAIN_OUT_ROOT/oof/rebuild_$STAMP"
else
  STAMP="small_$(date -u +%Y%m%d)"
  STAMP="$STAMP" N_TRIALS="${N_TRIALS:-5}" \
    OUT_DIR="$TRAIN_OUT_ROOT/oof/$STAMP" LOG_DIR="$TRAIN_OUT_ROOT/logs/$STAMP" \
    ONLY_CELLS="passing_yards:QB receiving_yards:WR targets:TE" \
    bash scripts/retrain_causal_oof.sh
  # Keep the three newest canaries.
  # shellcheck disable=SC2012
  ls -1d "$TRAIN_OUT_ROOT"/oof/small_* 2>/dev/null | sort -r | tail -n +4 | while IFS= read -r OLD; do
    rm -rf "$OLD" "$TRAIN_OUT_ROOT/logs/$(basename "$OLD")"
  done
fi
