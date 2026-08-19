"""
Causal baseline predictors for evaluation.

These are the honest comparison targets for every model cell:
  - prev_season_mean: mean of the same stat in the prior season (games-level)
  - trailing_n_mean: mean of the prior N games in the same season (shift-1)
  - kalman_baseline: FeatureMatrix kalman_est_* (incumbent, not a naive proxy)

No baseline may use information from the evaluation game itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def prev_season_mean(
    history: pd.DataFrame,
    *,
    player_id: str,
    season: int,
    stat_col: str,
    min_games: int = 1,
) -> Optional[float]:
    """
    Mean of `stat_col` for `player_id` in season-1.

    Returns None when fewer than `min_games` prior-season observations exist.
    """
    prior = history[
        (history["player_id"] == player_id)
        & (history["season"] == season - 1)
        & history[stat_col].notna()
    ]
    if len(prior) < min_games:
        return None
    return float(prior[stat_col].mean())


def trailing_n_mean(
    history: pd.DataFrame,
    *,
    player_id: str,
    season: int,
    week: int,
    stat_col: str,
    n: int = 3,
    min_games: int = 1,
) -> Optional[float]:
    """
    Mean of the prior `n` same-season games strictly before `week` (shift-1).

    Cross-season leakage is forbidden: only rows with the same season and
    week < evaluation week are eligible.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    prior = history[
        (history["player_id"] == player_id)
        & (history["season"] == season)
        & (history["week"] < week)
        & history[stat_col].notna()
    ].sort_values("week")
    if len(prior) < min_games:
        return None
    window = prior.tail(n)
    return float(window[stat_col].mean())


def attach_baselines(
    eval_rows: pd.DataFrame,
    history: pd.DataFrame,
    *,
    stat_col: str,
    trailing_n: int = 3,
    kalman_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach naive_baseline, rolling_baseline, and optional kalman_baseline.

    `eval_rows` must contain player_id, season, week.
    `history` must contain player_id, season, week, and `stat_col`.
    """
    required = {"player_id", "season", "week"}
    missing = required - set(eval_rows.columns)
    if missing:
        raise ValueError(f"eval_rows missing columns: {sorted(missing)}")
    if stat_col not in history.columns:
        raise ValueError(f"history missing stat column {stat_col!r}")

    out = eval_rows.copy()
    out["_eval_order"] = np.arange(len(out))
    clean = history.loc[:, ["player_id", "season", "week", stat_col]].copy()
    clean[stat_col] = pd.to_numeric(clean[stat_col], errors="coerce")
    clean = clean.sort_values(["player_id", "season", "week"])

    # Previous-season mean is a group aggregate joined onto season + 1. This
    # preserves the reference implementation's exact-season (not "last known
    # season") contract while avoiding an O(rows × history) Boolean scan.
    prior = (
        clean.dropna(subset=[stat_col])
        .groupby(["player_id", "season"], as_index=False)[stat_col]
        .mean()
        .rename(columns={stat_col: "naive_baseline"})
    )
    prior["season"] = prior["season"].astype(int) + 1
    out = out.merge(prior, on=["player_id", "season"], how="left", sort=False)

    # For rows from game-log history (the normal evaluation case), groupwise
    # shift then rolling is the same causal computation as trailing_n_mean:
    # only rows strictly before this week's row are visible.
    clean["rolling_baseline"] = (
        clean.groupby(["player_id", "season"], sort=False)[stat_col]
        .transform(lambda values: values.shift(1).rolling(trailing_n, min_periods=1).mean())
    )
    rolling = clean[["player_id", "season", "week", "rolling_baseline"]].drop_duplicates(
        ["player_id", "season", "week"], keep="last"
    )
    out = out.merge(rolling, on=["player_id", "season", "week"], how="left", sort=False)

    # Evaluation rows occasionally describe a scheduled game not yet present in
    # history. Retain the reference behavior for only those exceptional rows;
    # normal historical evaluation remains fully vectorized.
    known = pd.MultiIndex.from_frame(rolling[["player_id", "season", "week"]])
    requested = pd.MultiIndex.from_frame(out[["player_id", "season", "week"]])
    missing_history = ~requested.isin(known)
    if missing_history.any():
        out.loc[missing_history, "rolling_baseline"] = [
            np.nan if (value := trailing_n_mean(
                clean,
                player_id=str(row.player_id),
                season=int(row.season),
                week=int(row.week),
                stat_col=stat_col,
                n=trailing_n,
            )) is None else value
            for row in out.loc[missing_history].itertuples(index=False)
        ]

    if kalman_col and kalman_col in out.columns:
        out["kalman_baseline"] = pd.to_numeric(out[kalman_col], errors="coerce")
    else:
        out["kalman_baseline"] = None
    out = out.sort_values("_eval_order").drop(columns="_eval_order")
    return out
