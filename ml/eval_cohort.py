"""
Explicit pregame evaluation cohort definition.

Membership is based solely on prior completed games, never the target game's
snap count or any other realized participation quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.utils import MIN_PRIOR_GAMES


@dataclass(frozen=True)
class CohortSpec:
    """Immutable description of who enters evaluation."""

    min_prior_games: int = MIN_PRIOR_GAMES

    def describe(self) -> dict[str, Any]:
        return {
            "min_prior_games": self.min_prior_games,
            "rule": "player has at least this many completed games earlier in the same season",
        }


DEFAULT_COHORT = CohortSpec()


def normalize_offense_pct(raw: float | None) -> float | None:
    """Legacy display helper; it is not used for cohort membership."""
    if raw is None:
        return None
    value = float(raw)
    return value / 100.0 if value > 1.0 else value


def filter_cohort_frame(
    df: pd.DataFrame,
    *,
    spec: CohortSpec = DEFAULT_COHORT,
) -> pd.DataFrame:
    """Apply the shared pregame eligibility rule to a GameLog-like frame."""
    if df.empty:
        return df
    required = {"player_id", "season", "week"}
    if missing := required.difference(df.columns):
        raise KeyError(
            f"Cohort filter requires {sorted(required)!r}; missing {sorted(missing)!r}"
        )
    order = ["player_id", "season", "week"] + (["game_id"] if "game_id" in df.columns else [])
    ordered = df.sort_values(order).copy()
    ordered["_prior_games"] = ordered.groupby(["player_id", "season"], sort=False).cumcount()
    return ordered.loc[ordered["_prior_games"] >= spec.min_prior_games].drop(columns="_prior_games").copy()
