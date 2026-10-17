#!/usr/bin/env python3
"""
Aggregate per-cell causal-eval CSVs into one reproducible headline count.

Each input row is one evaluated cell-season with the model's score, the two
baseline scores, and whether the model beat each baseline. This script counts
rows where the model beat BOTH baselines, per cell and in total
("beat_both/cell_seasons", e.g. the owner-verified 74/75 claim), and records
exactly which files went into the count.

Inputs are discovered, never hard-coded: every
``reports/eval_causal_<stat>_<position>.csv`` present is aggregated, and the
output records its inputs. When no input exists the script fails loudly and
prints the commands a rerun must execute instead of inventing a number.

Producing the inputs (a VPS training run does this after `make freeze-baseline`):

    for each (stat, position) cell:
        python -m ml.eval_causal \\
            --oof <pinned stack CSV from releases/current_baseline.json> \\
            --stat <stat> --position <position> \\
            --history <player_id,season,week,<stat> baseline history CSV> \\
            --out reports/eval_causal_<stat>_<position>.csv
    python scripts/summarize_causal_evals.py
    python scripts/summarize_causal_evals.py --check   # fail if the file is stale

The pinned stack CSV for a cell is ``get_manifest().resolve_stack(stat,
position)``; the history CSV needs the database for baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"

GENERATED_BY = "python scripts/summarize_causal_evals.py"
OUTPUT_NAME = "causal_baseline_beats.json"
INPUT_GLOB = "eval_causal_*.csv"

REQUIRED_COLUMNS = frozenset({
    "stat",
    "season",
    "metric",
    "n",
    "model_score",
    "naive_score",
    "trailing3_score",
    "beats_naive",
    "beats_trailing3",
})

PRODUCER_HELP = """\
No per-cell causal-eval CSVs found ({pattern} under {reports}).
Nothing is written: a baseline-beating count with no inputs would be invented.

To regenerate (VPS training run, after `make freeze-baseline`):
  1. For each (stat, position) cell, score the pinned stack artifact:
       stack_csv=$(python -c "from ml.artifact_manifest import get_manifest; print(get_manifest().resolve_stack('<stat>', '<position>'))")
       python -m ml.eval_causal --oof "$stack_csv" --stat <stat> --position <position> \\
           --history <baseline-history CSV> --out reports/eval_causal_<stat>_<position>.csv
     (the history CSV needs the database for baselines)
  2. Rerun this script, then `python scripts/summarize_causal_evals.py --check`.
"""


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    raise ValueError(f"not a boolean: {value!r}")


def discover_inputs(eval_dir: Path) -> list[Path]:
    """Return sorted per-cell eval CSVs. Empty when the rerun never produced them."""
    return sorted(eval_dir.glob(INPUT_GLOB))


def summarize_cell(path: Path) -> dict:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    metrics = sorted(df["metric"].unique())
    if len(metrics) != 1:
        raise ValueError(f"{path.name}: mixed metrics {metrics}")
    metric = metrics[0]

    beats = df["beats_naive"].map(_to_bool) & df["beats_trailing3"].map(_to_bool)
    positions = sorted(df["position"].unique()) if "position" in df.columns else []
    return {
        "source_report": path.relative_to(REPO_ROOT).as_posix()
        if path.is_relative_to(REPO_ROOT)
        else path.name,
        "stat": str(df["stat"].iloc[0]) if df["stat"].nunique() == 1 else "mixed",
        "position": positions[0] if len(positions) == 1 else positions,
        "metric": metric,
        "beat_both": f"{int(beats.sum())}/{len(df)}",
        "beat_both_count": int(beats.sum()),
        "cell_seasons": int(len(df)),
        "seasons": sorted(int(s) for s in df["season"].unique()),
        "rows": int(pd.to_numeric(df["n"], errors="coerce").fillna(0).sum()),
    }


def build(eval_dir: Path = REPORTS) -> dict:
    """Aggregate every discovered per-cell CSV. Raises FileNotFoundError with
    producer instructions when there is nothing to aggregate."""
    inputs = discover_inputs(eval_dir)
    if not inputs:
        raise FileNotFoundError(
            PRODUCER_HELP.format(pattern=INPUT_GLOB, reports=eval_dir)
        )
    cells = [summarize_cell(path) for path in inputs]
    return {
        "schema_version": 1,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_note": (
            "`metric` is carried through from the per-cell eval CSVs, which get "
            "it from ml.eval_metrics.primary_score. A count target is scored "
            "with Poisson deviance and must never be summarised under an "
            "MAE-shaped key (audit C-31)."
        ),
        "inputs": [
            path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT)
            else path.name
            for path in inputs
        ],
        "headline": {
            "beat_both": sum(c["beat_both_count"] for c in cells),
            "cell_seasons": sum(c["cell_seasons"] for c in cells),
        },
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=REPORTS,
        help="directory holding per-cell eval_causal_*.csv files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="summary path (default: <eval-dir>/causal_baseline_beats.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if a summary differs from what this script would write",
    )
    args = parser.parse_args(argv)
    out = args.out or (args.eval_dir / OUTPUT_NAME)

    try:
        built = build(args.eval_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.check:
        if not out.exists():
            print(f"MISSING {out}", file=sys.stderr)
            return 1
        current = json.loads(out.read_text())
        for key in ("inputs", "headline", "cells"):
            if current.get(key) != built[key]:
                print(f"STALE   {out} — regenerate with {GENERATED_BY}", file=sys.stderr)
                return 1
        if "pooled_mae" in out.read_text():
            print(
                f"MISLABEL {out} — 'pooled_mae' names a metric that is not "
                "MAE and not pooled (audit C-31)",
                file=sys.stderr,
            )
            return 1
        print(f"ok {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(built, indent=2) + "\n")
    headline = built["headline"]
    print(f"wrote {out}  ({headline['beat_both']}/{headline['cell_seasons']} cell-seasons beat both baselines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
