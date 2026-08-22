"""
backend/app/api/team_game.py

Phase 4 (Coherent Prediction Hierarchy) — the first real game-outcome
endpoint. Reads from team_game_predictions, materialized by
scripts/materialize_team_game_predictions.py; this route never fits a
model per-request.

Coach identity (home_coach/away_coach) comes directly from nflreadpy's
schedules feed, unverified against any second source. An earlier version of
this note flagged Baltimore's 2026 home_coach ("Jesse Minter") as wrong on
the assumption John Harbaugh was still head coach — that assumption was
stale (Minter is in fact the real 2026 HC; Harbaugh was let go), not a data
bug. Lesson: don't assert a specific coach entry is wrong without a current
source, since coaching changes happen faster than any static knowledge can
track. What IS still true and worth documenting: the coach-tendency feature
degrades gracefully for a name with no prior head-coaching record in
`games` (falls back to the population median rather than misattributing
another coach's history), so a genuinely wrong or brand-new name doesn't
silently corrupt a prediction — it just loses that one feature's signal.
"""

from __future__ import annotations

from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.app.core.config import settings

router = APIRouter(prefix="", tags=["team_game"])

_COACH_DATA_CAVEAT = (
    "Predictions are a materialized Ridge model, not a per-request live fit. "
    "Coach identity (used for the tendency feature) comes directly from "
    "nflreadpy's schedules feed and is not independently verified here. A "
    "coach with no prior head-coaching record in this database's games "
    "table (e.g. newly promoted) falls back to the population-median "
    "tendency rather than misattributing another coach's history."
)


class TeamGamePrediction(BaseModel):
    game_id: str
    team: str
    opponent: str
    season: int
    week: int
    is_home: bool
    points: Optional[float] = None
    yards: Optional[float] = None
    pass_rate: Optional[float] = None
    win_probability: Optional[float] = None
    model_run_id: str
    note: str = _COACH_DATA_CAVEAT


class TeamGameWeekResponse(BaseModel):
    season: int
    week: int
    count: int
    games: list[TeamGamePrediction]


def _row_to_prediction(row: dict) -> TeamGamePrediction:
    return TeamGamePrediction(
        game_id=row["game_id"], team=row["team"], opponent=row["opponent"],
        season=row["season"], week=row["week"], is_home=bool(row["is_home"]),
        points=row["points"], yards=row["yards"],
        pass_rate=row["pass_rate"], win_probability=row["win_probability"],
        model_run_id=row["model_run_id"],
    )


@router.get("/team-games/{season}/{week}", response_model=TeamGameWeekResponse)
def team_game_week(season: int, week: int) -> TeamGameWeekResponse:
    try:
        conn = psycopg2.connect(settings.database_url)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT game_id, team, opponent, season, week, is_home,
                       points, yards, pass_rate, win_probability, model_run_id
                FROM team_game_predictions
                WHERE season = %s AND week = %s
                ORDER BY team
                """,
                (season, week),
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No team-game predictions for season={season} week={week}. "
                   f"Run scripts/materialize_team_game_predictions.py first.",
        )
    return TeamGameWeekResponse(
        season=season, week=week, count=len(rows),
        games=[_row_to_prediction(r) for r in rows],
    )


@router.get("/team-games/{season}/{week}/{team}", response_model=TeamGamePrediction)
def team_game_for_team(season: int, week: int, team: str) -> TeamGamePrediction:
    try:
        conn = psycopg2.connect(settings.database_url)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}") from exc
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT game_id, team, opponent, season, week, is_home,
                       points, yards, pass_rate, win_probability, model_run_id
                FROM team_game_predictions
                WHERE season = %s AND week = %s AND team = %s
                """,
                (season, week, team.upper()),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No team-game prediction for team={team.upper()} season={season} week={week}.",
        )
    return _row_to_prediction(dict(row))
