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
    naive: list[Optional[float]] = []
    rolling: list[Optional[float]] = []
    kalman: list[Optional[float]] = []

    for row in out.itertuples(index=False):
        naive.append(
            prev_season_mean(
                history,
                player_id=str(getattr(row, "player_id")),
                season=int(getattr(row, "season")),
                stat_col=stat_col,
            )
        )
        rolling.append(
            trailing_n_mean(
                history,
                player_id=str(getattr(row, "player_id")),
                season=int(getattr(row, "season")),
                week=int(getattr(row, "week")),
                stat_col=stat_col,
                n=trailing_n,
            )
        )
        if kalman_col and kalman_col in out.columns:
            val = getattr(row, kalman_col, None)
            kalman.append(float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None)
        else:
            kalman.append(None)

    out["naive_baseline"] = naive
    out["rolling_baseline"] = rolling
    out["kalman_baseline"] = kalman
    return out
