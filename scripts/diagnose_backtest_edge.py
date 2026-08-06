#!/usr/bin/env python3
"""
Summarize where the current backtest has edge and where it fails.

This is intentionally lightweight: it reads the frozen backtest CSV and prints
stat/position diagnostics without touching the database or model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


YARDAGE_STATS = {"passing_yards", "rushing_yards", "receiving_yards"}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "eval_season",
        "position",
        "stat",
        "n_games",
        "stack_mae",
        "naive_mae",
        "rolling_mae",
        "baseline_improvement_pct",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Backtest CSV missing required columns: {missing}")
    return df


def _summarize(df: pd.DataFrame) -> dict:
    by_stat = (
        df.groupby("stat")
        .agg(
            rows=("stat", "size"),
            games=("n_games", "sum"),
            stack_mae=("stack_mae", "mean"),
            naive_mae=("naive_mae", "mean"),
            rolling_mae=("rolling_mae", "mean"),
            mean_improvement_pct=("baseline_improvement_pct", "mean"),
            median_improvement_pct=("baseline_improvement_pct", "median"),
            positive_rows=("baseline_improvement_pct", lambda s: int((s > 0).sum())),
            negative_rows=("baseline_improvement_pct", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values("mean_improvement_pct")
    )
    by_position_stat = (
        df.groupby(["position", "stat"])
        .agg(
            rows=("stat", "size"),
            games=("n_games", "sum"),
            stack_mae=("stack_mae", "mean"),
            naive_mae=("naive_mae", "mean"),
            mean_improvement_pct=("baseline_improvement_pct", "mean"),
        )
        .reset_index()
        .sort_values("mean_improvement_pct")
    )
    yardage = by_position_stat[by_position_stat["stat"].isin(YARDAGE_STATS)]
    return {
        "overall": {
            "rows": int(len(df)),
            "eval_seasons": [int(x) for x in sorted(df["eval_season"].unique())],
            "mean_improvement_pct": float(df["baseline_improvement_pct"].mean()),
            "median_improvement_pct": float(df["baseline_improvement_pct"].median()),
            "positive_rows": int((df["baseline_improvement_pct"] > 0).sum()),
            "negative_rows": int((df["baseline_improvement_pct"] < 0).sum()),
        },
        "by_stat": by_stat.to_dict(orient="records"),
        "worst_yardage": yardage.head(10).to_dict(orient="records"),
        "best_edges": by_position_stat.sort_values("mean_improvement_pct", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }


def _print(summary: dict) -> None:
    overall = summary["overall"]
    print("Backtest Edge Diagnostics")
    print("=" * 25)
    print(
        "rows={rows} seasons={eval_seasons} mean_improvement={mean_improvement_pct:.2f}% "
        "median_improvement={median_improvement_pct:.2f}% positive_rows={positive_rows} "
        "negative_rows={negative_rows}".format(**overall)
    )
    print("\nBy stat (sorted worst first):")
    for row in summary["by_stat"]:
        print(
            "  {stat:16} mean={mean_improvement_pct:8.2f}% "
            "median={median_improvement_pct:8.2f}% stack_mae={stack_mae:8.3f} "
            "naive_mae={naive_mae:8.3f}".format(**row)
        )
    print("\nWorst yardage position/stat slices:")
    for row in summary["worst_yardage"]:
        print(
            "  {position:2} {stat:16} mean={mean_improvement_pct:8.2f}% "
            "stack_mae={stack_mae:8.3f} naive_mae={naive_mae:8.3f}".format(**row)
        )
    print("\nImmediate product focus:")
    print("  1. Diagnose passing_yards target scale and baseline mismatch.")
    print("  2. Diagnose rushing_yards outliers and low-snap filtering.")
    print("  3. Keep TD/counting-stat pipelines stable while yardage is repaired.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose backtest edge by stat and position.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ml/backtest_results/backtest_2019_2025.csv"),
        help="Backtest CSV to inspect.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summary = _summarize(_load(args.csv))
    _print(summary)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
