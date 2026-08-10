#!/usr/bin/env python3
"""
Regenerate the causal-eval summary JSONs from the per-cell eval CSVs — C-31.

`reports/eval_causal_volume_summary.json` reported **mean Poisson deviance**
under the key `pooled_mae`. Two errors in one key:

  * the metric is not MAE — `ml/eval_metrics.primary_score` routes count targets
    (`targets`, `carries`, `pass_attempts`) to `poisson_deviance`;
  * it is not pooled — the value is the *unweighted mean across seasons* of the
    per-season scores, not a score recomputed over the pooled rows. (Verified:
    the committed values match `df.model_score.mean()` to 6 decimals, and differ
    from the n-weighted mean in the 4th.)

Both summaries were hand-assembled, with no generator, so nothing stopped the
next one from being mislabelled the same way. This script is the generator. It
carries the metric name through from the CSVs rather than assuming one, so a
count target can never again be summarised under an MAE-shaped key.

    python scripts/summarize_causal_evals.py
    python scripts/summarize_causal_evals.py --check   # fail if the files are stale
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

# (output file, cells) — cells are (target, position) in report order.
GROUPS: dict[str, list[tuple[str, str]]] = {
    "eval_causal_volume_summary.json": [
        ("targets", "WR"),
        ("targets", "TE"),
        ("targets", "RB"),
        ("carries", "RB"),
        ("pass_attempts", "QB"),
    ],
    "eval_causal_yardage_summary.json": [
        ("receiving_yards", "WR"),
        ("receiving_yards", "TE"),
        ("receiving_yards", "RB"),
        ("rushing_yards", "RB"),
        ("rushing_yards", "QB"),
    ],
}

# Recorded when the summary was first assembled; the eval CSVs do not name the
# stack artifact they scored, so this mapping is carried forward explicitly
# rather than silently dropped.
ARTIFACTS: dict[tuple[str, str], str] = {
    ("receiving_yards", "WR"): "stack_receiving_yards_WR_20260809.csv",
    ("receiving_yards", "TE"): "stack_receiving_yards_TE_20260809.csv",
    ("receiving_yards", "RB"): "stack_receiving_yards_RB_20260809.csv",
    ("rushing_yards", "RB"): "stack_rushing_yards_RB_20260809.csv",
    ("rushing_yards", "QB"): "stack_rushing_yards_QB_20260809.csv",
}


def summarize_cell(target: str, position: str) -> dict:
    path = REPORTS / f"eval_causal_stack_{target}_{position}.csv"
    df = pd.read_csv(path)

    metrics = sorted(df["metric"].unique())
    if len(metrics) != 1:
        raise ValueError(f"{path.name}: mixed metrics {metrics}")
    metric = metrics[0]

    beats = df["beats_naive"].astype(bool) & df["beats_trailing3"].astype(bool)
    record = {
        "target": target,
        "position": position,
        "metric": metric,
        "beat_both": f"{int(beats.sum())}/{len(df)}",
        "mean_season_score": float(df["model_score"].mean()),
        "n_weighted_score": float(
            (df["model_score"] * df["n"]).sum() / df["n"].sum()
        ),
        "seasons": int(len(df)),
        "rows": int(df["n"].sum()),
        "source_report": path.relative_to(REPO_ROOT).as_posix(),
    }
    artifact = ARTIFACTS.get((target, position))
    if artifact:
        record["artifact"] = artifact
    return record


def build(name: str) -> dict:
    return {
        "schema_version": 1,
        "generated_by": GENERATED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_note": (
            "`metric` is carried through from the per-cell eval CSVs, which get "
            "it from ml.eval_metrics.primary_score. `mean_season_score` is the "
            "unweighted mean of per-season scores; `n_weighted_score` weights by "
            "row count. Neither is a score recomputed over pooled rows."
        ),
        "cells": [summarize_cell(t, p) for t, p in GROUPS[name]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if a summary differs from what this script would write",
    )
    args = parser.parse_args()

    failures = 0
    for name in GROUPS:
        built = build(name)
        out = REPORTS / name
        if args.check:
            if not out.exists():
                print(f"MISSING {name}", file=sys.stderr)
                failures += 1
                continue
            current = json.loads(out.read_text())
            # Timestamps are expected to differ; content is not.
            if current.get("cells") != built["cells"]:
                print(f"STALE   {name} — regenerate with {GENERATED_BY}", file=sys.stderr)
                failures += 1
            if "pooled_mae" in out.read_text():
                print(
                    f"MISLABEL {name} — 'pooled_mae' names a metric that is not "
                    "MAE and not pooled (audit C-31)",
                    file=sys.stderr,
                )
                failures += 1
        else:
            out.write_text(json.dumps(built, indent=2) + "\n")
            print(f"wrote {out.relative_to(REPO_ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
