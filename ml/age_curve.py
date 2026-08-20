"""Walk-forward age curves for season-long fantasy rates.

Marcel's canonical form is age-adjusted; :func:`ml.baselines.attach_marcel` is
not. This module supplies the missing multiplier and *fits it from data*
rather than hardcoding constants -- the repo already carries one hand-tuned
fudge (``std *= 3.0``) and does not need a second.

Causality: the curve for target season S is fit only on seasons < S. Calling
:func:`fit_age_curve` with rows from S or later raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ELIGIBLE = ("QB", "RB", "WR", "TE")
MIN_BUCKET_ROWS = 25
SHRINK_ROWS = 40.0
AGE_MIN, AGE_MAX = 21, 36


def season_age(birth_date: pd.Series, season: pd.Series) -> pd.Series:
    """Age on 1 September of the season year. NaT births yield NaN."""
    births = pd.to_datetime(birth_date, errors="coerce")
    years = pd.to_numeric(season, errors="coerce")
    age = years - births.dt.year
    # Birthdays after 1 September have not happened by kickoff.
    age = age - ((births.dt.month > 9) | ((births.dt.month == 9) & (births.dt.day > 1))).astype(float)
    return age.where(births.notna() & years.notna())


def fit_age_curve(observations: pd.DataFrame, *, max_season: int) -> dict[tuple[str, int], float]:
    """Fit ``actual_rate / expected_rate`` by (position, age) on seasons <= max_season.

    ``observations`` needs position, age, actual_rate, expected_rate. Buckets
    thinner than ``MIN_BUCKET_ROWS`` are shrunk toward 1.0 in proportion to
    their weight, so a sparse age never swings a projection on its own.
    """
    required = {"position", "age", "actual_rate", "expected_rate", "season"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"age-curve observations missing {sorted(missing)}")
    seasons = pd.to_numeric(observations["season"], errors="coerce")
    if (seasons > max_season).any():
        raise ValueError(
            f"age curve for max_season={max_season} was handed rows from "
            f"season {int(seasons.max())}; that is target-season leakage"
        )

    frame = observations.copy()
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce").round()
    frame["actual_rate"] = pd.to_numeric(frame["actual_rate"], errors="coerce")
    frame["expected_rate"] = pd.to_numeric(frame["expected_rate"], errors="coerce")
    frame = frame.dropna(subset=["age", "actual_rate", "expected_rate"])
    frame = frame[frame["expected_rate"] > 0.5]
    frame = frame[frame["age"].between(AGE_MIN, AGE_MAX)]
    frame["ratio"] = frame["actual_rate"] / frame["expected_rate"]
    # Trim the tail: a single 6x breakout must not define an age bucket.
    frame = frame[frame["ratio"].between(0.1, 4.0)]
    if frame.empty:
        return {}

    curve: dict[tuple[str, int], float] = {}
    for position in ELIGIBLE:
        subset = frame[frame["position"].astype(str).str.upper() == position]
        if len(subset) < MIN_BUCKET_ROWS:
            continue
        overall = float(subset["ratio"].mean())
        for age, bucket in subset.groupby("age"):
            n = float(len(bucket))
            raw = float(bucket["ratio"].mean())
            weight = n / (n + SHRINK_ROWS)
            curve[(position, int(age))] = weight * raw + (1.0 - weight) * overall
    return curve


def age_multiplier(curve: dict[tuple[str, int], float], position: str, age: float | None) -> float:
    """Look up the fitted multiplier, falling back to 1.0 outside the curve."""
    if age is None or not np.isfinite(age):
        return 1.0
    key = (str(position).upper(), int(round(float(age))))
    return float(curve.get(key, 1.0))


def build_observations(
    game_logs: pd.DataFrame,
    players: pd.DataFrame,
    *,
    max_season: int,
    min_games: int = 6,
) -> pd.DataFrame:
    """Assemble (position, age, actual_rate, expected_rate) rows for seasons <= max_season.

    ``expected_rate`` is the player's prior-season per-game rate, so the fitted
    ratio measures year-over-year change at a given age rather than raw level.
    """
    logs = game_logs.copy()
    logs["season"] = pd.to_numeric(logs["season"], errors="coerce")
    logs = logs[logs["season"] <= max_season]
    logs["fantasy_points_ppr"] = pd.to_numeric(logs["fantasy_points_ppr"], errors="coerce")
    logs = logs.dropna(subset=["fantasy_points_ppr", "season"])
    rates = (
        logs.groupby(["player_id", "season"], as_index=False)
        .agg(actual_rate=("fantasy_points_ppr", "mean"), games=("fantasy_points_ppr", "size"))
    )
    rates = rates[rates["games"] >= min_games]
    prior = rates.rename(columns={"actual_rate": "expected_rate"})[["player_id", "season", "expected_rate"]]
    prior["season"] = prior["season"].astype(int) + 1
    frame = rates.merge(prior, on=["player_id", "season"], how="inner")

    roster = players.rename(columns={"id": "player_id"})[["player_id", "position", "birth_date"]]
    frame = frame.merge(roster, on="player_id", how="left")
    frame["position"] = frame["position"].astype(str).str.upper()
    frame["age"] = season_age(frame["birth_date"], frame["season"])
    return frame[["player_id", "season", "position", "age", "actual_rate", "expected_rate"]]
