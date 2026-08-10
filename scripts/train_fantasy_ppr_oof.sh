#!/usr/bin/env bash
# Generate causal OOF for fantasy_ppr across positions (LAST_COMPLETE seasons only).
#
# Phase-5 contract: lgbm + catboost only. XGB and TFT are killed learners and are
# not trained here; ml.stacking_ensemble enforces the allowlist independently, so
# this script cannot resurrect them by accident.
#
# Resumable via checkpoints; --no-mlflow; append-only log (nohup-safe).
set -euo pipefail
cd "$(dirname "$0")/.."
export DATABASE_URL="${DATABASE_URL:-postgresql://oracle:oracle@localhost:15439/oracle}"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
unset MLFLOW_TRACKING_URI || true

PY="${PY:-.venv_311/bin/python}"
SEASONS=2019-2025
POSITIONS="QB RB WR TE"
LEARNERS="lgbm catboost"
mkdir -p ml/logs ml/oof ml/checkpoints/done
LOG="${LOG:-ml/logs/fantasy_ppr_oof_$(date +%Y%m%dT%H%M%S).log}"

exec >>"$LOG" 2>&1
echo "=== fantasy_ppr OOF generation start $(date) ==="
echo "log=$LOG seasons=$SEASONS db=$DATABASE_URL no_mlflow=1 pid=$$"

CK_DIR="ml/checkpoints/done"

# mark() used to write "$1" while done_ck() read "$1.done", so a checkpoint never
# matched its own marker: every resume retrained everything, and no marker was
# ever trusted. Both now agree on the .done suffix.
mark()    { touch "$CK_DIR/$1.done"; }
unmark()  { rm -f "$CK_DIR/$1.done"; }

# A marker alone is not evidence of an artifact. Require the CSV too, matching
# the pattern already used in scripts/train_volume_oof.sh — a stale marker from
# an interrupted or purged run must not skip real work.
done_ck() {
  local ck="$1" glob="$2"
  [[ -f "$CK_DIR/${ck}.done" ]] || return 1
  if compgen -G "$glob" >/dev/null 2>&1; then
    return 0
  fi
  echo "  [!] stale checkpoint ${ck}.done — no artifact matching $glob; re-running."
  unmark "$ck"
  return 1
}

FAILED=()

run_tree() {
  local model="$1" pos="$2" mod="$3"
  local ck="${model}_fantasy_ppr_${pos}"
  local glob="ml/oof/${model}_fantasy_ppr_${pos}_*.csv"
  if done_ck "$ck" "$glob"; then
    echo "SKIP $ck (checkpoint + artifact present)"
    return 0
  fi
  unmark "$ck"
  echo ">>> $model fantasy_ppr/$pos $(date)"
  if $PY -m "$mod" --seasons "$SEASONS" --target fantasy_ppr --position "$pos" \
      --n-trials 20 --out-dir ml/oof --no-mlflow; then
    # Validate the output exists *before* checkpointing. A marker written first
    # is a claim a later resume trusts without rechecking.
    if compgen -G "$glob" >/dev/null 2>&1; then
      mark "$ck"
      echo "OK $ck $(date)"
    else
      echo "FAILED $ck: trainer exited 0 but wrote no artifact matching $glob"
      FAILED+=("$ck (no artifact)")
    fi
  else
    echo "FAILED $model $pos exit=$?"
    FAILED+=("$ck")
  fi
}

for pos in $POSITIONS; do
  for model in $LEARNERS; do
    case "$model" in
      lgbm)     run_tree lgbm     "$pos" ml.lgbm_model ;;
      catboost) run_tree catboost "$pos" ml.catboost_model ;;
    esac
  done
done

# ── Stacking ──────────────────────────────────────────────────────────────────
# No --exclude: the allowed learner set is a checked invariant inside
# ml.stacking_ensemble (ml/artifact_manifest.py). This restack previously ran
# --oof-dir with no exclusion at all, so it rebuilt the killed four-learner
# stack from whatever OOFs happened to be in ml/oof/.
for pos in $POSITIONS; do
  ck="stack_fantasy_ppr_${pos}"
  glob="ml/oof/stack_fantasy_ppr_${pos}_*.csv"
  if done_ck "$ck" "$glob"; then
    echo "SKIP $ck (checkpoint + artifact present)"
    continue
  fi
  unmark "$ck"
  echo ">>> stacking fantasy_ppr/$pos $(date)"
  if $PY -m ml.stacking_ensemble --oof-dir ml/oof --target fantasy_ppr --position "$pos" \
      --out-dir ml/oof --no-mlflow; then
    if compgen -G "$glob" >/dev/null 2>&1; then
      mark "$ck"
      echo "OK $ck $(date)"
    else
      echo "FAILED $ck: stacker exited 0 but wrote no artifact matching $glob"
      FAILED+=("$ck (no artifact)")
    fi
  else
    echo "FAILED stack $pos exit=$?"
    FAILED+=("$ck")
  fi
done

# ── Expected-cell matrix ──────────────────────────────────────────────────────
# Exit non-zero if any declared cell is missing. Without this the script reported
# success while cells were absent, which is how partial artifact sets shipped.
MISSING=()
for pos in $POSITIONS; do
  for model in $LEARNERS; do
    compgen -G "ml/oof/${model}_fantasy_ppr_${pos}_*.csv" >/dev/null 2>&1 \
      || MISSING+=("${model}_fantasy_ppr_${pos}")
  done
  compgen -G "ml/oof/stack_fantasy_ppr_${pos}_*.csv" >/dev/null 2>&1 \
    || MISSING+=("stack_fantasy_ppr_${pos}")
done

echo "=== fantasy_ppr OOF generation done $(date) ==="
ls -lh ml/oof/*fantasy_ppr* 2>/dev/null || true
ls "$CK_DIR"/*fantasy* 2>/dev/null || true

if (( ${#FAILED[@]} > 0 )); then
  echo "FAILURES (${#FAILED[@]}):"
  printf '  %s\n' "${FAILED[@]}"
fi
if (( ${#MISSING[@]} > 0 )); then
  echo "MISSING CELLS (${#MISSING[@]}):"
  printf '  %s\n' "${MISSING[@]}"
  echo "Refusing to report success with an incomplete cell matrix."
  exit 1
fi
if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
echo "All declared fantasy_ppr cells present."
