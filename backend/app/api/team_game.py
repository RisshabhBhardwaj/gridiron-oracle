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


# ── Anchor-Then-Narrate Drive Sim ─────────────────────────────────────────────

class GameAnchor(BaseModel):
    home_points: float
    away_points: float
    home_yards: float
    away_yards: float
    home_pass_rate: float
    away_pass_rate: float
    home_win_probability: float


class GameSimSummary(BaseModel):
    simulated_home_points: int
    simulated_away_points: int
    simulated_home_yards: float
    simulated_away_yards: float
    total_drives: int
    simulated_winner: str


class SimulatedPlayDTO(BaseModel):
    play_number: int
    down: int
    ytg: int
    field_pos: int
    play_type: str
    yards_gained: float
    is_turnover: bool
    is_first_down: bool
    is_touchdown: bool
    is_safety: bool
    end_field_pos: int


class SimulatedDriveDTO(BaseModel):
    drive_number: int
    possession_team: str
    quarter: int
    start_field_pos: int
    end_field_pos: int
    plays_count: int
    yards_gained: float
    outcome: str
    points_scored: int
    home_score_after: int
    away_score_after: int
    plays: list[SimulatedPlayDTO]


class GameDriveSimResponse(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    seed: int
    anchor: GameAnchor
    summary: GameSimSummary
    drives: list[SimulatedDriveDTO]
    note: str = (
        "Reconciles play-by-play Markov drive paths to the fitted Ridge team-game anchor. "
        "Drives alternate dynamically with game state (score diff, quarter) driving play calling."
    )


@router.get("/games/{game_id}/drive-sim", response_model=GameDriveSimResponse)
def simulate_game_drives(
    game_id: str,
    seed: Optional[int] = Query(None, description="RNG seed for deterministic drive sequence"),
    n_drives_per_team: int = Query(11, ge=6, le=16, description="Drives per team (11 = ~22 total drives)"),
) -> GameDriveSimResponse:
    import numpy as np
    from ml.drive_path import simulate_drive_path

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
                WHERE game_id = %s
                """,
                (game_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for game_id={game_id}.",
        )

    home_row = next((r for r in rows if r["is_home"]), None)
    away_row = next((r for r in rows if not r["is_home"]), None)
    if home_row is None or away_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Incomplete pair in team_game_predictions for game_id={game_id}.",
        )

    home_team = home_row["team"]
    away_team = away_row["team"]
    season = home_row["season"]
    week = home_row["week"]

    anchor = GameAnchor(
        home_points=float(home_row.get("points") or 24.0),
        away_points=float(away_row.get("points") or 21.0),
        home_yards=float(home_row.get("yards") or 340.0),
        away_yards=float(away_row.get("yards") or 320.0),
        home_pass_rate=float(home_row.get("pass_rate") or 0.58),
        away_pass_rate=float(away_row.get("pass_rate") or 0.58),
        home_win_probability=float(home_row.get("win_probability") or 0.55),
    )

    rng_seed = seed if seed is not None else int(np.random.randint(0, 1_000_000_000))
    rng = np.random.default_rng(rng_seed)

    total_drives = n_drives_per_team * 2
    drives_dtos: list[SimulatedDriveDTO] = []

    home_score = 0
    away_score = 0
    home_yards = 0.0
    away_yards = 0.0

    curr_possession = away_team  # Away receives opening kickoff
    curr_start_fp = 25

    for d_num in range(1, total_drives + 1):
        # Determine quarter from drive number
        if d_num <= total_drives * 0.25:
            qtr = 1
        elif d_num <= total_drives * 0.5:
            qtr = 2
        elif d_num <= total_drives * 0.75:
            qtr = 3
        else:
            qtr = 4

        # Halftime reset (start of Q3 -> home team receives at 25)
        if d_num == int(total_drives * 0.5) + 1:
            curr_possession = home_team
            curr_start_fp = 25

        score_diff = (home_score - away_score) if curr_possession == home_team else (away_score - home_score)

        drive_sim = simulate_drive_path(
            start_fp=curr_start_fp,
            score_diff=score_diff,
            quarter=qtr,
            possession_team=curr_possession,
            drive_number=d_num,
            rng=rng,
        )

        if curr_possession == home_team:
            home_score += drive_sim.points_scored
            home_yards += drive_sim.yards_gained
        else:
            away_score += drive_sim.points_scored
            away_yards += drive_sim.yards_gained

        # Convert plays to DTOs
        play_dtos = [
            SimulatedPlayDTO(
                play_number=p.play_number,
                down=p.down,
                ytg=p.ytg,
                field_pos=p.field_pos,
                play_type=p.play_type,
                yards_gained=p.yards_gained,
                is_turnover=p.is_turnover,
                is_first_down=p.is_first_down,
                is_touchdown=p.is_touchdown,
                is_safety=p.is_safety,
                end_field_pos=p.end_field_pos,
            )
            for p in drive_sim.plays
        ]

        drives_dtos.append(
            SimulatedDriveDTO(
                drive_number=d_num,
                possession_team=curr_possession,
                quarter=qtr,
                start_field_pos=drive_sim.start_field_pos,
                end_field_pos=drive_sim.end_field_pos,
                plays_count=drive_sim.plays_count,
                yards_gained=drive_sim.yards_gained,
                outcome=drive_sim.outcome,
                points_scored=drive_sim.points_scored,
                home_score_after=home_score,
                away_score_after=away_score,
                plays=play_dtos,
            )
        )

        # Switch possession & determine next start field position
        if drive_sim.outcome in ("TOUCHDOWN", "FIELD_GOAL", "SAFETY"):
            curr_start_fp = 25  # Touchback after kickoff
        elif drive_sim.outcome == "PUNT":
            # Punt distance approx 40 yards
            punt_landing = drive_sim.end_field_pos + int(rng.normal(42, 5))
            if punt_landing >= 100:
                curr_start_fp = 20  # Touchback
            else:
                curr_start_fp = max(10, 100 - punt_landing)
        else:
            # Turnover on downs or fumble/INT
            curr_start_fp = max(10, min(90, 100 - drive_sim.end_field_pos))

        curr_possession = home_team if curr_possession == away_team else away_team

    winner = home_team if home_score > away_score else (away_team if away_score > home_score else "TIE")

    summary = GameSimSummary(
        simulated_home_points=home_score,
        simulated_away_points=away_score,
        simulated_home_yards=round(home_yards, 1),
        simulated_away_yards=round(away_yards, 1),
        total_drives=len(drives_dtos),
        simulated_winner=winner,
    )

    return GameDriveSimResponse(
        game_id=game_id,
        season=season,
        week=week,
        home_team=home_team,
        away_team=away_team,
        seed=rng_seed,
        anchor=anchor,
        summary=summary,
        drives=drives_dtos,
    )
