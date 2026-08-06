"""
ml/win_eval.py

Brier / MAE scoring for season win totals and playoff probabilities.

Actuals come from completed games (home_score / away_score). Predictions come
from SeasonSimulation.team_win_totals / playoff_probs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class WinEvalResult:
    season: int
    n_teams: int
    wins_mae: float
    wins_rmse: float
    playoff_brier: Optional[float]
    per_team: list[dict]


def actual_wins_from_schedule(schedule_df: pd.DataFrame) -> dict[str, float]:
    """
    Count regular-season wins from a schedule with scores.

    Expects columns: home_team, away_team, home_score, away_score.
    Ties → 0.5 each. Rows missing scores are skipped.
    """
    wins: dict[str, float] = {}
    for _, row in schedule_df.iterrows():
        hs, aws = row.get("home_score"), row.get("away_score")
        if hs is None or aws is None or (isinstance(hs, float) and np.isnan(hs)):
            continue
        home, away = str(row["home_team"]), str(row["away_team"])
        wins.setdefault(home, 0.0)
        wins.setdefault(away, 0.0)
        hs_f, aws_f = float(hs), float(aws)
        if hs_f > aws_f:
            wins[home] += 1.0
        elif aws_f > hs_f:
            wins[away] += 1.0
        else:
            wins[home] += 0.5
            wins[away] += 0.5
    return wins


def actual_playoff_made(
    schedule_df: pd.DataFrame,
    berths: int = 7,
) -> dict[str, int]:
    """
    Binary playoff indicator from final win totals (top `berths` per conference).

    Uses the same AFC/NFC sets as SeasonSimulator. Tie-break: alphabetical
    (stable, not NFL tiebreakers — good enough for Brier scaffolding).
    """
    from ml.season_simulator import _AFC, _NFC

    wins = actual_wins_from_schedule(schedule_df)
    made = {t: 0 for t in wins}
    for conf in (_AFC, _NFC):
        conf_teams = [(wins.get(t, 0.0), t) for t in conf if t in wins]
        conf_teams.sort(key=lambda x: (-x[0], x[1]))
        for _, t in conf_teams[:berths]:
            made[t] = 1
    return made


def brier_score(probs: dict[str, float], outcomes: dict[str, int]) -> float:
    """Mean Brier score over teams present in both dicts."""
    keys = sorted(set(probs) & set(outcomes))
    if not keys:
        return float("nan")
    return float(np.mean([(probs[k] - outcomes[k]) ** 2 for k in keys]))


def evaluate_win_projections(
    season: int,
    team_win_totals: dict[str, dict[str, float]],
    schedule_df: pd.DataFrame,
    playoff_probs: Optional[dict[str, float]] = None,
) -> WinEvalResult:
    """Compare projected win means / playoff probs to realized schedule."""
    actual_wins = actual_wins_from_schedule(schedule_df)
    teams = sorted(set(team_win_totals) & set(actual_wins))
    if not teams:
        raise ValueError("No overlapping teams between projections and schedule")

    pred = np.array([team_win_totals[t]["wins_mean"] for t in teams], dtype=float)
    act = np.array([actual_wins[t] for t in teams], dtype=float)
    wins_mae = float(np.mean(np.abs(pred - act)))
    wins_rmse = float(np.sqrt(np.mean((pred - act) ** 2)))

    playoff_brier: Optional[float] = None
    actual_po: dict[str, int] = {}
    if playoff_probs:
        actual_po = actual_playoff_made(schedule_df)
        playoff_brier = brier_score(playoff_probs, actual_po)

    per_team = []
    for t in teams:
        row = {
            "team": t,
            "wins_pred": float(team_win_totals[t]["wins_mean"]),
            "wins_actual": float(actual_wins[t]),
        }
        if playoff_probs and t in playoff_probs:
            row["playoff_prob"] = float(playoff_probs[t])
            row["playoff_actual"] = int(actual_po.get(t, 0))
        per_team.append(row)

    return WinEvalResult(
        season=season,
        n_teams=len(teams),
        wins_mae=wins_mae,
        wins_rmse=wins_rmse,
        playoff_brier=playoff_brier,
        per_team=per_team,
    )
