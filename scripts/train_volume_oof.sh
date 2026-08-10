#!/usr/bin/env bash
# Volume-target OOF (causal): targets (WR/TE/RB), carries (RB), pass_attempts (QB).
#
# Phase-5 contract: lgbm + catboost only. ml.stacking_ensemble enforces the
# allowlist independently of this script.
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
LOG="${LOG:-ml/logs/volume_oof_$(date +%Y%m%dT%H%M%S).log}"
mkdir -p ml/logs ml/oof ml/checkpoints/done
exec >>"$LOG" 2>&1
echo "=== volume OOF start $(date) ==="

CELLS=("WR:targets" "TE:targets" "RB:targets" "RB:carries" "QB:pass_attempts")
FAILED=()

run_tree() {
  local model="$1" target="$2" pos="$3"
  local ck="${model}_${target}_${pos}"
  local glob="ml/oof/${model}_${target}_${pos}_*.csv"
  # Require an actual OOF CSV — stale .done markers without artifacts must not skip.
  if [[ -f "ml/checkpoints/done/${ck}.done" ]] && compgen -G "$glob" >/dev/null 2>&1; then
    echo "SKIP $ck"
    return 0
  fi
  rm -f "ml/checkpoints/done/${ck}.done"
  echo ">>> $model $target/$pos $(date)"
  if $PY -m "ml.${model}_model" --seasons "$SEASONS" --target "$target" --position "$pos" \
      --n-trials "$TRIALS" --out-dir ml/oof --no-mlflow; then
    # Validate output before checkpointing, never the other way round.
    if compgen -G "$glob" >/dev/null 2>&1; then
      touch "ml/checkpoints/done/${ck}.done"
      echo "OK $ck $(date)"
    else
      echo "FAILED $ck: trainer exited 0 but wrote no artifact matching $glob"
      FAILED+=("$ck (no artifact)")
    fi
  else
    echo "FAILED $ck exit=$?"
    FAILED+=("$ck")
  fi
}

for model in lgbm catboost; do
  for pos_target in "${CELLS[@]}"; do
    run_tree "$model" "${pos_target##*:}" "${pos_target%%:*}"
  done
done

# Latest OOF by dated filename, not by mtime. `ls -t` ordered by modification
# time, so a `touch` on a stale OOF — or a fresh checkout, where every file
# carries the checkout timestamp — silently changed which predictions were
# stacked.
latest_by_name() {
  compgen -G "$1" 2>/dev/null | sort | tail -1
}

for pos_target in "${CELLS[@]}"; do
  pos="${pos_target%%:*}"
  target="${pos_target##*:}"
  ck="stack_${target}_${pos}"
  if [[ -f "ml/checkpoints/done/${ck}.done" ]] && compgen -G "ml/oof/stack_${target}_${pos}_*.csv" >/dev/null 2>&1; then
    echo "SKIP $ck"; continue
  fi
  rm -f "ml/checkpoints/done/${ck}.done"
  echo ">>> stack $target/$pos $(date)"
  lgbm=$(latest_by_name "ml/oof/lgbm_${target}_${pos}_*.csv")
  catb=$(latest_by_name "ml/oof/catboost_${target}_${pos}_*.csv")
  if [[ -z "$lgbm" || -z "$catb" ]]; then
    echo "FAILED $ck missing tree OOF (lgbm=$lgbm cat=$catb)"
    FAILED+=("$ck (missing tree OOF)")
    continue
  fi
  if $PY -m ml.stacking_ensemble --oof "$lgbm" "$catb" --target "$target" --position "$pos" \
      --out-dir ml/oof --no-mlflow; then
    # Validate output before checkpointing, never the other way round.
    if compgen -G "ml/oof/stack_${target}_${pos}_*.csv" >/dev/null 2>&1; then
      touch "ml/checkpoints/done/${ck}.done"
      echo "OK $ck $(date)"
    else
      echo "FAILED $ck: stacker exited 0 but wrote no artifact"
      FAILED+=("$ck (no artifact)")
    fi
  else
    echo "FAILED $ck exit=$?"
    FAILED+=("$ck")
  fi
done

# ── Expected-cell matrix ──────────────────────────────────────────────────────
MISSING=()
for pos_target in "${CELLS[@]}"; do
  pos="${pos_target%%:*}"
  target="${pos_target##*:}"
  for model in lgbm catboost; do
    compgen -G "ml/oof/${model}_${target}_${pos}_*.csv" >/dev/null 2>&1 \
      || MISSING+=("${model}_${target}_${pos}")
  done
  compgen -G "ml/oof/stack_${target}_${pos}_*.csv" >/dev/null 2>&1 \
    || MISSING+=("stack_${target}_${pos}")
done

echo "=== volume OOF done $(date) ==="
ls -lh ml/oof/*{targets,carries,pass_attempts}* 2>/dev/null || true

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
echo "All declared volume cells present."
