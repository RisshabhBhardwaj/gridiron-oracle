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
    out = attach_marcel(out, history, stat_col=stat_col)
    out = out.sort_values("_eval_order").drop(columns="_eval_order")
    return out


MARCEL_LAG_WEIGHTS: tuple[tuple[int, float], ...] = ((1, 5.0), (2, 4.0), (3, 3.0))
MARCEL_REGRESSION_GAMES = 30.0


def attach_marcel(
    eval_rows: pd.DataFrame,
    history: pd.DataFrame,
    *,
    stat_col: str,
    regression_games: float = MARCEL_REGRESSION_GAMES,
) -> pd.DataFrame:
    """Attach a causal Marcel rate (prior three seasons only, shrunk to position mean).

    Same-season games of the evaluation year are never used. Contemporaneous
    expected-fantasy (xFP) is not a Marcel input.
    """
    out = eval_rows.copy()
    if stat_col not in history.columns:
        out["marcel_baseline"] = np.nan
        return out
    needed = {"player_id", "season", stat_col}
    if needed - set(history.columns):
        out["marcel_baseline"] = np.nan
        return out

    hist = history.loc[:, [c for c in ["player_id", "season", "position", stat_col] if c in history.columns]].copy()
    hist[stat_col] = pd.to_numeric(hist[stat_col], errors="coerce")
    hist = hist.dropna(subset=[stat_col])
    if hist.empty:
        out["marcel_baseline"] = np.nan
        return out

    group_cols = ["player_id", "season"]
    if "position" in hist.columns:
        group_cols.append("position")
    rates = hist.groupby(group_cols, as_index=False).agg(
        rate=(stat_col, "mean"),
        games=(stat_col, "size"),
    ).drop_duplicates(["player_id", "season"], keep="first")
    pos_means = pd.DataFrame(columns=["season", "position", "pos_mean"])
    if "position" in rates.columns:
        pos_means = (
            rates.groupby(["season", "position"], as_index=False)["rate"]
            .mean()
            .rename(columns={"rate": "pos_mean"})
        )

    for lag, _weight in MARCEL_LAG_WEIGHTS:
        lagged = rates.rename(columns={"rate": f"rate_{lag}", "games": f"games_{lag}"})
        lagged["season"] = lagged["season"].astype(int) + lag
        merge_cols = ["player_id", "season", f"rate_{lag}", f"games_{lag}"]
        out = out.merge(lagged[merge_cols], on=["player_id", "season"], how="left")

    if "position" in out.columns and not pos_means.empty:
        prior_pos = pos_means.rename(columns={"season": "_pos_season"})
        prior_pos["season"] = prior_pos["_pos_season"].astype(int) + 1
        out = out.merge(
            prior_pos[["season", "position", "pos_mean"]],
            on=["season", "position"],
            how="left",
        )
    else:
        out["pos_mean"] = np.nan

    numer = pd.Series(0.0, index=out.index)
    denom = pd.Series(0.0, index=out.index)
    for lag, weight in MARCEL_LAG_WEIGHTS:
        games = pd.to_numeric(out.get(f"games_{lag}"), errors="coerce").fillna(0.0)
        rate = pd.to_numeric(out.get(f"rate_{lag}"), errors="coerce")
        contrib = weight * games
        numer = numer + contrib * rate.fillna(0.0)
        denom = denom + contrib
    pos_mean = pd.to_numeric(out["pos_mean"], errors="coerce")
    shrink = pos_mean.notna() & (denom > 0)
    numer = numer.where(~shrink, numer + regression_games * pos_mean.fillna(0.0))
    denom = denom.where(~shrink, denom + regression_games)
    marcel = numer / denom.replace(0.0, np.nan)
    marcel = marcel.where(denom > 0)
    out["marcel_baseline"] = marcel
    drop_cols = [c for c in out.columns if c.startswith("rate_") or c.startswith("games_") or c == "pos_mean"]
    return out.drop(columns=drop_cols, errors="ignore")
