#!/usr/bin/env bash
# Phase-5 yardage repair: receiving_yards (WR/TE/RB) + rushing_yards (RB/QB).
# Trees = lgbm+catboost only. The allowlist is a checked invariant inside
# ml.stacking_ensemble (ml/artifact_manifest.py), not a CLI flag here.
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
LOG="${LOG:-ml/logs/yardage_oof_$(date +%Y%m%dT%H%M%S).log}"
mkdir -p ml/logs ml/oof ml/checkpoints/done reports
exec >>"$LOG" 2>&1
echo "=== yardage OOF start $(date) ==="

CELLS=("WR:receiving_yards" "TE:receiving_yards" "RB:receiving_yards" "RB:rushing_yards" "QB:rushing_yards")
FAILED=()

# Latest OOF by dated filename, not mtime. `ls -t` ordered by modification time,
# so a `touch` on a stale OOF — or a fresh checkout, where every file carries the
# checkout timestamp — silently changed which predictions got stacked.
latest_by_name() {
  compgen -G "$1" 2>/dev/null | sort | tail -1
}

run_tree() {
  local model="$1" target="$2" pos="$3"
  local ck="${model}_${target}_${pos}"
  local glob="ml/oof/${model}_${target}_${pos}_*.csv"
  if [[ -f "ml/checkpoints/done/${ck}.done" ]] && compgen -G "$glob" >/dev/null 2>&1; then
    echo "SKIP $ck"; return 0
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
  for pair in "${CELLS[@]}"; do
    run_tree "$model" "${pair##*:}" "${pair%%:*}"
  done
done

for pair in "${CELLS[@]}"; do
  pos="${pair%%:*}"; target="${pair##*:}"
  ck="stack_${target}_${pos}"
  if [[ -f "ml/checkpoints/done/${ck}.done" ]] && compgen -G "ml/oof/stack_${target}_${pos}_*.csv" >/dev/null 2>&1; then
    echo "SKIP $ck"; continue
  fi
  rm -f "ml/checkpoints/done/${ck}.done"
  lgbm=$(latest_by_name "ml/oof/lgbm_${target}_${pos}_*.csv")
  catb=$(latest_by_name "ml/oof/catboost_${target}_${pos}_*.csv")
  if [[ -z "$lgbm" || -z "$catb" ]]; then
    echo "FAILED $ck missing trees"
    FAILED+=("$ck (missing tree OOF)")
    continue
  fi
  echo ">>> stack $target/$pos $(date)"
  if $PY -m ml.stacking_ensemble --oof "$lgbm" "$catb" --target "$target" --position "$pos" \
      --out-dir ml/oof --no-mlflow; then
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

$PY - <<'PY'
from pathlib import Path
import json, os
import pandas as pd
from sqlalchemy import create_engine, text
from ml.eval_causal import score_oof_against_baselines

engine = create_engine(os.environ["DATABASE_URL"])
summary = []
for target, pos, col in [
    ("receiving_yards","WR","actual_receiving_yards"),
    ("receiving_yards","TE","actual_receiving_yards"),
    ("receiving_yards","RB","actual_receiving_yards"),
    ("rushing_yards","RB","actual_rushing_yards"),
    ("rushing_yards","QB","actual_rushing_yards"),
]:
    paths = sorted(Path("ml/oof").glob(f"stack_{target}_{pos}_*.csv"))
    paths = [p for p in paths if "_archive" not in str(p)]
    if not paths:
        summary.append({"target": target, "position": pos, "status": "missing"}); continue
    stack = paths[-1]
    oof = pd.read_csv(stack); oof["position"] = pos
    with engine.connect() as c:
        hist = pd.read_sql(text(f"""
            SELECT player_id, season, week, {col} AS {target}
            FROM feature_matrix
            WHERE position = :pos AND season BETWEEN 2019 AND 2025 AND {col} IS NOT NULL
        """), c, params={"pos": pos})
    table = score_oof_against_baselines(oof, hist, stat=target, position=pos)
    out = Path(f"reports/eval_causal_stack_{target}_{pos}.csv")
    table.to_csv(out, index=False)
    beat = int((table.beats_naive & table.beats_trailing3).sum())
    summary.append({
        "target": target, "position": pos, "beat_both": f"{beat}/{len(table)}",
        "artifact": stack.name,
    })
    print(f"{target}/{pos}: {beat}/{len(table)}")
    print(table.to_string(index=False))
Path("reports/eval_causal_yardage_summary.json").write_text(json.dumps(summary, indent=2))
print("saved reports/eval_causal_yardage_summary.json")
PY

# ── Expected-cell matrix ──────────────────────────────────────────────────────
MISSING=()
for pair in "${CELLS[@]}"; do
  pos="${pair%%:*}"; target="${pair##*:}"
  for model in lgbm catboost; do
    compgen -G "ml/oof/${model}_${target}_${pos}_*.csv" >/dev/null 2>&1 \
      || MISSING+=("${model}_${target}_${pos}")
  done
  compgen -G "ml/oof/stack_${target}_${pos}_*.csv" >/dev/null 2>&1 \
    || MISSING+=("stack_${target}_${pos}")
done

echo "=== yardage OOF done $(date) ==="

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
echo "All declared yardage cells present."
