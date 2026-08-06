"""
Causal evaluation from OOF predictions (preferred over DB projection replay).

OOF folds from the walk-forward trainers already assert
max(train_seasons) < val_season. This module:

  1. Loads OOF CSVs for a (stat, model) pair
  2. Attaches true prev-season / trailing-3 / Kalman baselines
  3. Scores with target-type-aware metrics
  4. Emits the headline evaluation table

Usage:
  python -m ml.eval_causal --oof-dir ml/oof --stat fantasy_ppr --out reports/eval_fantasy_ppr.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.baselines import attach_baselines
from ml.eval_metrics import primary_score
from ml.stat_resolution import VALID_POSITION_STATS, assert_position_stats_resolvable

logger = logging.getLogger(__name__)


def _normalize_oof(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    out = df.copy()
    if "predicted" not in out.columns and "y_pred" in out.columns:
        out = out.rename(columns={"y_pred": "predicted"})
    if "actual" not in out.columns and "y_true" in out.columns:
        out = out.rename(columns={"y_true": "actual"})
    if "stat" not in out.columns:
        out["stat"] = stat
    required = {"player_id", "season", "week", "actual", "predicted"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"OOF missing columns {sorted(missing)}")
    # Provenance for BacktestRunner causality assert
    if "max_train_season" not in out.columns:
        # Expanding-window OOF: max train season is the fold's prior season.
        out["max_train_season"] = out["season"].astype(int) - 1
    return out


def score_oof_against_baselines(
    oof: pd.DataFrame,
    history: pd.DataFrame,
    *,
    stat: str,
    position: str | None = None,
) -> pd.DataFrame:
    """
    Return one summary row per (season, position) with model vs baselines.
    `history` must contain player_id, season, week, and the resolved actual column
    named `stat` (or fantasy_points_ppr renamed upstream).
    """
    df = _normalize_oof(oof, stat)
    if position and "position" in df.columns:
        df = df[df["position"] == position].copy()
    if df.empty:
        return pd.DataFrame()

    # Build baseline frame keyed like eval rows.
    eval_rows = df[["player_id", "season", "week"]].copy()
    if "position" in df.columns:
        eval_rows["position"] = df["position"].values
    else:
        eval_rows["position"] = position or "ALL"

    hist = history.copy()
    if stat not in hist.columns:
        raise ValueError(f"history must include column {stat!r}")

    with_base = attach_baselines(eval_rows, hist, stat_col=stat, trailing_n=3)
    merged = df.merge(
        with_base,
        on=["player_id", "season", "week"],
        how="inner",
        suffixes=("", "_base"),
    )
    rows = []
    group_cols = ["season"]
    if "position" in merged.columns:
        group_cols.append("position")
    for keys, cohort in merged.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = cohort["actual"].to_numpy(dtype=float)
        pred = cohort["predicted"].to_numpy(dtype=float)
        metric_name, model_score = primary_score(stat, actual, pred)
        naive = cohort["naive_baseline"].to_numpy(dtype=float)
        rolling = cohort["rolling_baseline"].to_numpy(dtype=float)
        # Drop NaN baselines for fair per-baseline scores
        naive_mask = np.isfinite(naive)
        roll_mask = np.isfinite(rolling)
        _, naive_score = primary_score(stat, actual[naive_mask], naive[naive_mask]) if naive_mask.any() else (metric_name, np.nan)
        _, roll_score = primary_score(stat, actual[roll_mask], rolling[roll_mask]) if roll_mask.any() else (metric_name, np.nan)
        record = {
            "stat": stat,
            "metric": metric_name,
            "n": int(len(cohort)),
            "model_score": model_score,
            "naive_score": naive_score,
            "trailing3_score": roll_score,
            "beats_naive": bool(model_score < naive_score) if np.isfinite(naive_score) else None,
            "beats_trailing3": bool(model_score < roll_score) if np.isfinite(roll_score) else None,
            "max_train_season_ok": bool((cohort["max_train_season"] < cohort["season"]).all()),
        }
        for col, val in zip(group_cols, keys):
            record[col] = val
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True, help="OOF CSV path")
    parser.add_argument("--stat", required=True)
    parser.add_argument("--position", default=None)
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="CSV with player_id,season,week,<stat> for baseline construction",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    assert_position_stats_resolvable(VALID_POSITION_STATS)
    oof = pd.read_csv(args.oof)
    history = pd.read_csv(args.history)
    table = score_oof_against_baselines(
        oof, history, stat=args.stat, position=args.position
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
