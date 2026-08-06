#!/usr/bin/env python3
"""Create an auditable target-by-target diagnostic report from a feature CSV.

The report deliberately separates structural failures (leakage, duplicates,
units, missingness) from forecast-quality failures (residuals and coverage).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.reliability import YARDAGE_STATS, sha256_json
from ml.utils import FEATURE_COLS, TARGET_COL_MAP


def build_report(df: pd.DataFrame, target: str, position: str) -> dict:
    if target not in TARGET_COL_MAP:
        raise ValueError(f"Unknown target {target!r}")
    target_col = TARGET_COL_MAP[target]
    cohort = df[df["position"].eq(position)].copy() if "position" in df else df.copy()
    observed_col = target_col if target_col in cohort else "y_true" if "y_true" in cohort else None
    if observed_col is None:
        raise ValueError(f"Missing {target_col} (feature matrix) or y_true (OOF) column")
    cohort = cohort[cohort[observed_col].notna()].copy()
    feature_cols = [c for c in FEATURE_COLS if c in cohort]
    leakage_columns = [c for c in feature_cols if c.startswith("actual_")]
    duplicate_rows = int(cohort.duplicated(["player_id", "game_id"]).sum())
    output: dict = {
        "target": target,
        "position": position,
        "is_yardage": target in YARDAGE_STATS,
        "schema_hash": sha256_json(feature_cols),
        "rows": int(len(cohort)),
        "season_rows": {str(k): int(v) for k, v in cohort.groupby("season").size().items()},
        "target_distribution": cohort[observed_col].describe(percentiles=[.01, .05, .5, .95, .99]).to_dict(),
        "missingness": {c: float(cohort[c].isna().mean()) for c in feature_cols},
        "provenance": {
            "duplicate_player_games": duplicate_rows,
            "feature_actual_columns": leakage_columns,
            "observed_column": observed_col,
            "temporal_ordered": bool(cohort.sort_values(["season", "week", "player_id"]).index.equals(cohort.index)),
        },
    }
    if {"y_true", "y_pred"}.issubset(cohort.columns):
        residual = cohort["y_true"] - cohort["y_pred"]
        output["residuals"] = residual.describe(percentiles=[.05, .5, .95]).to_dict()
        output["mae"] = float(residual.abs().mean())
    if {"floor", "ceiling", observed_col}.issubset(cohort.columns):
        output["interval_coverage_80"] = float(
            ((cohort[observed_col] >= cohort["floor"]) & (cohort[observed_col] <= cohort["ceiling"])).mean()
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Feature/OOF CSV")
    parser.add_argument("--target", required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(pd.read_csv(args.input), args.target, args.position)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
