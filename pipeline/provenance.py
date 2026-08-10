"""As-of provenance storage and assertions for causal feature inputs.

This module deliberately raises instead of substituting a current snapshot.
Feature-contract code can call these assertions before it materializes a model
frame.  No function in this module creates or alters schema; revision
``20260809_0005`` owns the tables and columns.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values

from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

logger = logging.getLogger(__name__)


class ProvenanceError(RuntimeError):
    """A requested feature lacks a source value that is legal as of kickoff."""


def height_to_inches(value: Any) -> float | None:
    """Normalize nflverse roster height values (72, ``6-0``, or ``6'0``)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if value == value else None
    text = str(value).strip().replace("'", "-").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        pass
    for separator in ("-", "\""):
        if separator in text:
            feet, inches = text.split(separator, 1)
            try:
                return float(int(feet) * 12 + int(inches.rstrip('"')))
            except ValueError:
                return None
    return None


def backfill_player_season_profiles(
    db_url: str, seasons: Iterable[int] | None = None
) -> int:
    """Recover season-scoped physical profiles from retained roster staging rows.

    ``source_captured_at`` records when this project captured the nflverse row;
    ``effective_season`` is the value used by the as-of join.  The latter, not a
    retrospective download date, proves that a profile belongs to the feature
    season.
    """
    season_list = sorted(set(seasons)) if seasons is not None else None
    with psycopg2.connect(_psycopg2_dsn(db_url)) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if season_list:
                cur.execute(
                    """
                    SELECT raw_data, ingested_at FROM staging_nflreadpy
                    WHERE source_type = 'rosters' AND season = ANY(%s)
                    ORDER BY id
                    """,
                    (season_list,),
                )
            else:
                cur.execute(
                    """
                    SELECT raw_data, ingested_at FROM staging_nflreadpy
                    WHERE source_type = 'rosters' ORDER BY id
                    """
                )
            staged = cur.fetchall()

            latest: dict[tuple[str, int], tuple] = {}
            for record in staged:
                raw = record["raw_data"]
                player_id = raw.get("gsis_id") or raw.get("player_id")
                season = raw.get("season")
                if not player_id or season is None:
                    continue
                key = (str(player_id), int(season))
                latest[key] = (
                    key[0], key[1], height_to_inches(raw.get("height")),
                    _as_float(raw.get("weight")), "nflreadpy.load_rosters",
                    record["ingested_at"],
                )

            if not latest:
                return 0
            execute_values(
                cur,
                """
                INSERT INTO player_season_profiles
                    (player_id, effective_season, height_inches, weight_lbs, source, source_captured_at)
                VALUES %s
                ON CONFLICT (player_id, effective_season) DO UPDATE SET
                    height_inches = EXCLUDED.height_inches,
                    weight_lbs = EXCLUDED.weight_lbs,
                    source = EXCLUDED.source,
                    source_captured_at = EXCLUDED.source_captured_at
                """,
                list(latest.values()),
            )
    logger.info("Backfilled %d player-season profile rows", len(latest))
    return len(latest)


def assert_player_profiles_asof(conn: Any, season: int) -> None:
    """Require every player-game to use a profile effective no later than its season."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM game_logs gl
            LEFT JOIN player_season_profiles psp
              ON psp.player_id = gl.player_id
             AND psp.effective_season <= gl.season
            WHERE gl.season = %s AND psp.player_id IS NULL
            """,
            (season,),
        )
        missing = cur.fetchone()[0]
    if missing:
        raise ProvenanceError(
            f"{missing} player-game rows in season {season} lack an effective player profile"
        )


def assert_depth_charts_pregame(conn: Any, season: int) -> None:
    """Reject depth-chart values whose source publication is missing or late."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM feature_matrix fm
            JOIN games g ON g.id = fm.game_id
            JOIN depth_charts dc
              ON dc.player_id = fm.player_id AND dc.season = fm.season AND dc.week = fm.week
            WHERE fm.season = %s
              AND (g.kickoff_at IS NULL OR dc.published_at IS NULL OR dc.published_at >= g.kickoff_at)
            """,
            (season,),
        )
        invalid = cur.fetchone()[0]
    if invalid:
        raise ProvenanceError(
            f"{invalid} depth-chart rows in season {season} lack a pre-kickoff publication time"
        )


def assert_weather_forecasts_pregame(conn: Any, season: int) -> None:
    """Require a stored forecast snapshot captured strictly before each kickoff."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM feature_matrix fm
            JOIN games g ON g.id = fm.game_id
            LEFT JOIN LATERAL (
                SELECT wf.game_id
                FROM weather_forecasts wf
                WHERE wf.game_id = g.id
                  AND wf.captured_at < wf.kickoff_at
                  AND wf.kickoff_at = g.kickoff_at
                ORDER BY wf.captured_at DESC
                LIMIT 1
            ) wf ON TRUE
            WHERE fm.season = %s AND wf.game_id IS NULL
            """,
            (season,),
        )
        missing = cur.fetchone()[0]
    if missing:
        raise ProvenanceError(
            f"{missing} player-game rows in season {season} lack a pre-kickoff weather forecast"
        )


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None
