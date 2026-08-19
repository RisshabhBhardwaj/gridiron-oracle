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
from ml.eval_cohort import CohortSpec, DEFAULT_COHORT, filter_cohort_frame
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
    if "max_train_season" not in out.columns:
        raise ValueError(
            "OOF missing max_train_season provenance; do not infer it from the "
            "evaluation season. A causal assertion over an invented value is tautological."
        )
    out["max_train_season"] = pd.to_numeric(out["max_train_season"], errors="coerce")
    if out["max_train_season"].isna().any():
        raise ValueError("OOF has non-numeric max_train_season provenance")
    if not (out["max_train_season"] < pd.to_numeric(out["season"], errors="raise")).all():
        raise ValueError("OOF contains non-causal max_train_season provenance")
    return out


def score_oof_against_baselines(
    oof: pd.DataFrame,
    history: pd.DataFrame,
    *,
    stat: str,
    position: str | None = None,
    cohort_spec: CohortSpec = DEFAULT_COHORT,
) -> pd.DataFrame:
    """
    Return one summary row per (season, position) with model vs baselines.
    `history` must contain player_id, season, week, and the resolved actual column
    named `stat` (or fantasy_points_ppr renamed upstream).
    """
    df = _normalize_oof(oof, stat)
    if position and "position" in df.columns:
        df = df[df["position"] == position].copy()
    df = filter_cohort_frame(df, spec=cohort_spec)
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
        actual = pd.to_numeric(cohort["actual"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(cohort["predicted"], errors="coerce").to_numpy(dtype=float)
        naive = pd.to_numeric(cohort["naive_baseline"], errors="coerce").to_numpy(dtype=float)
        rolling = pd.to_numeric(cohort["rolling_baseline"], errors="coerce").to_numpy(dtype=float)
        # A comparison only means something when every contestant is scored on
        # the same declared population.  Do not report a model score on rows
        # where a baseline is undefined.
        common = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(naive) & np.isfinite(rolling)
        metric_name, model_score = primary_score(stat, actual[common], pred[common]) if common.any() else ("unavailable", np.nan)
        _, naive_score = primary_score(stat, actual[common], naive[common]) if common.any() else (metric_name, np.nan)
        _, roll_score = primary_score(stat, actual[common], rolling[common]) if common.any() else (metric_name, np.nan)
        record = {
            "stat": stat,
            "metric": metric_name,
            "n": int(common.sum()),
            "n_model": int((np.isfinite(actual) & np.isfinite(pred)).sum()),
            "n_naive": int((np.isfinite(actual) & np.isfinite(naive)).sum()),
            "n_trailing3": int((np.isfinite(actual) & np.isfinite(rolling)).sum()),
            "cohort_rule": cohort_spec.describe()["rule"],
            "min_prior_games": cohort_spec.min_prior_games,
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
