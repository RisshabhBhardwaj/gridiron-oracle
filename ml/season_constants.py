"""
Season constants derived from data availability, not scattered literals.

As of August 2026 (pre-Week 1):
  - CURRENT_SEASON = 2026 (rosters/schedules/depth exist; game logs do not)
  - LAST_COMPLETE_SEASON = 2025 (last season with full game logs / PBP / NGS)

Training, OOF, and causal evaluation must cap at LAST_COMPLETE_SEASON.
ETL may still refresh CURRENT_SEASON context tables (rosters, schedule, depth).
"""

from __future__ import annotations

from datetime import date

# Bump LAST_COMPLETE_SEASON after the Super Bowl when the prior season is final.
LAST_COMPLETE_SEASON: int = 2025
CURRENT_SEASON: int = 2026

# Inclusive training range used by walk-forward learners.
TRAIN_SEASON_START: int = 2019


def train_seasons() -> list[int]:
    """Seasons eligible for model training and OOF evaluation."""
    return list(range(TRAIN_SEASON_START, LAST_COMPLETE_SEASON + 1))


def train_seasons_arg() -> str:
    """CLI form accepted by `_parse_seasons` (e.g. '2019-2025')."""
    return f"{TRAIN_SEASON_START}-{LAST_COMPLETE_SEASON}"


def etl_seasons() -> list[int]:
    """
    Seasons for ETL refresh.

    Includes CURRENT_SEASON for context-only tables (rosters/schedules/depth).
    Game-stat sources for CURRENT_SEASON may be empty until Week 1.
    """
    return list(range(TRAIN_SEASON_START, CURRENT_SEASON + 1))


def cap_seasons(seasons: list[int], *, complete_only: bool = True) -> list[int]:
    """Drop seasons beyond the configured cap."""
    cap = LAST_COMPLETE_SEASON if complete_only else CURRENT_SEASON
    return [s for s in seasons if TRAIN_SEASON_START <= s <= cap]


def assert_not_fitting_incomplete_season(seasons: list[int]) -> None:
    """Refuse to treat an incomplete season as a training/eval fold."""
    bad = [s for s in seasons if s > LAST_COMPLETE_SEASON]
    if bad:
        raise ValueError(
            f"Seasons {bad} exceed LAST_COMPLETE_SEASON={LAST_COMPLETE_SEASON}. "
            f"Pre-Week-1 {CURRENT_SEASON} data is context-only; do not fit on it. "
            f"Pass --seasons {train_seasons_arg()} instead."
        )


def today_iso() -> str:
    return date.today().isoformat()
