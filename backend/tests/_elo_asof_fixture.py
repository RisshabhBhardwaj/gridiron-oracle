"""Shared schedule for the as-of Elo tests (audit C1).

Six teams play a fixed round-robin in 2023 and 2024 (weeks 1-5), then 2024
week 6 adds two more games. Scores come from a seeded RNG drawn in the same
order for every variant, so flipping one game's result changes only that game.
"""

from __future__ import annotations

import random

import pandas as pd

ASOF_TEAMS = ("MIN", "GB", "KC", "BUF", "DET", "CHI")

_ROUND_ROBIN: dict[int, list[tuple[str, str]]] = {
    1: [("MIN", "KC"), ("GB", "BUF"), ("DET", "CHI")],
    2: [("MIN", "BUF"), ("GB", "DET"), ("KC", "CHI")],
    3: [("MIN", "DET"), ("GB", "CHI"), ("KC", "BUF")],
    4: [("MIN", "CHI"), ("GB", "KC"), ("BUF", "DET")],
    5: [("MIN", "GB"), ("KC", "DET"), ("BUF", "CHI")],
}

# (season, week, home, away)
ASOF_GAMES: tuple[tuple[int, int, str, str], ...] = tuple(
    (season, week, home, away)
    for season in (2023, 2024)
    for week in sorted(_ROUND_ROBIN)
    for home, away in _ROUND_ROBIN[week]
) + ((2024, 6, "KC", "MIN"), (2024, 6, "GB", "CHI"))

ASOF_TARGET = (2024, 5, "MIN", "GB")   # the row under test; MIN hosts GB
ASOF_LATER = (2024, 6, "KC", "MIN")    # a later game in which MIN plays
ASOF_EARLIER = (2024, 1, "MIN", "KC")  # an earlier game in which MIN plays (positive control)


def asof_game_id(season: int, week: int, home: str, away: str) -> str:
    return f"{season}_{week:02d}_{away}_{home}"


def asof_games_frame(flip: tuple[int, int, str, str] | None = None) -> pd.DataFrame:
    """One row per game with the Elo fit columns. ``flip`` swaps that game's scores."""
    rng = random.Random(20241005)
    rows = []
    for season, week, home, away in ASOF_GAMES:
        home_score = rng.randint(10, 35)
        away_score = rng.randint(10, 35)
        if home_score == away_score:
            home_score += 1
        if flip == (season, week, home, away):
            home_score, away_score = away_score, home_score
        rows.append({
            "game_id": asof_game_id(season, week, home, away),
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
        })
    return pd.DataFrame(rows)
