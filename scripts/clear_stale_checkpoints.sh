#!/usr/bin/env bash
# Clear stale .done markers so missing OOF CSVs can be regenerated.
# Checkpoints currently claim fantasy_ppr (and others) are done even when
# ml/oof/ has no corresponding CSVs — --resume alone will skip them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DONE="$ROOT/ml/checkpoints/done"

if [[ ! -d "$DONE" ]]; then
  echo "No checkpoint dir at $DONE — nothing to clear."
  exit 0
fi

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 <stat|all>"
  echo "  e.g. $0 fantasy_ppr"
  echo "       $0 all"
  exit 2
fi

if [[ "$TARGET" == "all" ]]; then
  count=$(find "$DONE" -name '*.done' | wc -l | tr -d ' ')
  rm -f "$DONE"/*.done
  echo "Cleared $count checkpoint markers under $DONE"
else
  # Portable (macOS/BSD): no mapfile
  count=0
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    rm -f "$f"
    count=$((count + 1))
  done < <(find "$DONE" -name "*${TARGET}*" -name '*.done')
  if [[ "$count" -eq 0 ]]; then
    echo "No markers matching *$TARGET*"
    exit 0
  fi
  echo "Cleared $count markers matching *$TARGET*"
fi

echo "Next: bash ml/train_all_models.sh --fast-mode --resume"
echo "Seasons are capped at LAST_COMPLETE_SEASON via ml/season_constants.py / _parse_seasons."
