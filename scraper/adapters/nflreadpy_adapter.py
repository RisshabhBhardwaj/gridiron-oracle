"""
scraper/adapters/nflreadpy_adapter.py

Primary data adapter for Gridiron Oracle.
Ingests player stats, rosters, schedules, and snap counts from nflreadpy
(https://github.com/nflverse/nflreadpy) into the staging_nflreadpy PostgreSQL table.

Data flow:
  nflreadpy → Pydantic validation → staging_nflreadpy  (success)
                                  → dead_letter          (failure)

Standalone usage:
  python scraper/adapters/nflreadpy_adapter.py --seasons 2025
  python scraper/adapters/nflreadpy_adapter.py --dry-run --seasons 2025
  python scraper/adapters/nflreadpy_adapter.py --dry-run --source player_stats
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import json
import time
from datetime import date, datetime
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values, Json as PGJson


class _DateEncoder(json.JSONEncoder):
    """JSON encoder that converts date/datetime objects to ISO strings."""

    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def _safe_pg_json(d: dict) -> PGJson:
    """Wrap a dict in PGJson using the date-safe encoder."""
    return PGJson(d, dumps=lambda v: json.dumps(v, cls=_DateEncoder))
from pydantic import BaseModel, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SOURCE_PLAYER_STATS   = "player_stats"
SOURCE_ROSTERS       = "rosters"
SOURCE_SCHEDULES     = "schedules"
SOURCE_SNAP_COUNTS   = "snap_counts"
SOURCE_DEPTH_CHARTS   = "depth_charts"
SOURCE_NEXTGEN_STATS  = "nextgen_stats"
SOURCE_TEAM_STATS     = "team_stats"
SOURCE_FTN_CHARTING   = "ftn_charting"
SOURCE_PARTICIPATION  = "participation"
SOURCE_COMBINE        = "combine"

ALL_SOURCES = [
    SOURCE_PLAYER_STATS, SOURCE_ROSTERS, SOURCE_SCHEDULES, SOURCE_SNAP_COUNTS,
    SOURCE_DEPTH_CHARTS, SOURCE_NEXTGEN_STATS, SOURCE_TEAM_STATS,
    SOURCE_FTN_CHARTING, SOURCE_PARTICIPATION, SOURCE_COMBINE,
]

BATCH_SIZE = 500
MIN_DELAY_BETWEEN_SOURCES = 1.0  # seconds — CLAUDE.md §3 data rules


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nan_to_none(v: Any) -> Any:
    """
    Polars converts null→None in .to_dicts(), but float NaN stays as NaN.
    Pydantic treats NaN as invalid for most numeric fields, so coerce to None.
    """
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply NaN→None conversion to every value in a Polars-sourced row dict."""
    return {k: _nan_to_none(v) for k, v in row.items()}


def _psycopg2_dsn(db_url: str) -> str:
    """
    Strip the SQLAlchemy driver specifier so psycopg2 can parse the URL.
    e.g. 'postgresql+asyncpg://...' → 'postgresql://...'
    """
    return re.sub(r"postgresql\+\w+://", "postgresql://", db_url)


# ── Pydantic validation models ────────────────────────────────────────────────
#
# Each model corresponds to one nflreadpy load_*() source.
# Fields are Optional[T] = None for anything that may be absent in real data.
# model_config extra="ignore" silently drops columns not yet modelled.
# The adapter calls _coerce_row() before validation to handle NaN → None.
#
# Required fields (no default) will raise ValidationError if absent or null
# → those rows go straight to dead_letter (never silently dropped).

class PlayerStatsRow(BaseModel):
    """
    One row from nfl.load_player_stats(summary_level='week').
    Docs: https://nflreadr.nflverse.com/articles/dictionary_player_stats.html
    """
    model_config = {"extra": "ignore"}

    player_id: str                              # gsis_id — required
    season: int
    week: int
    player_display_name: Optional[str] = None
    position: Optional[str] = None
    position_group: Optional[str] = None
    season_type: Optional[str] = None
    game_id: Optional[str] = None
    team: Optional[str] = None
    opponent_team: Optional[str] = None
    # Passing
    completions: Optional[int] = None
    attempts: Optional[int] = None
    passing_yards: Optional[float] = None
    passing_tds: Optional[int] = None
    passing_interceptions: Optional[int] = None
    passing_air_yards: Optional[float] = None
    passing_yards_after_catch: Optional[float] = None
    passing_first_downs: Optional[int] = None
    passing_epa: Optional[float] = None
    passing_cpoe: Optional[float] = None
    # Rushing
    carries: Optional[int] = None
    rushing_yards: Optional[float] = None
    rushing_tds: Optional[int] = None
    rushing_fumbles: Optional[int] = None
    rushing_epa: Optional[float] = None
    # Additional fumble types (nflverse player_stats has all three)
    receiving_fumbles: Optional[int] = None
    sack_fumbles: Optional[int] = None
    # Receiving
    receptions: Optional[int] = None
    targets: Optional[int] = None
    receiving_yards: Optional[float] = None
    receiving_tds: Optional[int] = None
    receiving_air_yards: Optional[float] = None
    receiving_yards_after_catch: Optional[float] = None
    receiving_first_downs: Optional[int] = None
    receiving_epa: Optional[float] = None
    racr: Optional[float] = None
    target_share: Optional[float] = None
    air_yards_share: Optional[float] = None
    wopr: Optional[float] = None
    # Fantasy
    fantasy_points: Optional[float] = None
    fantasy_points_ppr: Optional[float] = None


class RosterRow(BaseModel):
    """
    One row from nfl.load_rosters().
    Note: player_id here is 'gsis_id' (not 'player_id' as in player_stats).
    Docs: https://nflreadr.nflverse.com/articles/dictionary_rosters.html
    """
    model_config = {"extra": "ignore"}

    season: int
    gsis_id: Optional[str] = None              # may be null for historical players
    team: Optional[str] = None
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    depth_chart_position: Optional[str] = None
    jersey_number: Optional[int] = None
    # nflverse sends height as float (inches, e.g. 72.0 = 6'0"); accept both
    height: Optional[str | float] = None
    weight: Optional[float] = None
    birth_date: Optional[date] = None
    college: Optional[str] = None
    years_exp: Optional[int] = None
    entry_year: Optional[int] = None
    rookie_year: Optional[int] = None
    status: Optional[str] = None
    headshot_url: Optional[str] = None
    espn_id: Optional[str] = None
    pfr_id: Optional[str] = None
    rotowire_id: Optional[str] = None
    draft_club: Optional[str] = None
    draft_number: Optional[int] = None


class ScheduleRow(BaseModel):
    """
    One row from nfl.load_schedules().
    Docs: https://nflreadr.nflverse.com/articles/dictionary_schedules.html
    """
    model_config = {"extra": "ignore"}

    game_id: str                                # required — primary key
    season: int
    week: int
    home_team: str
    away_team: str
    game_type: Optional[str] = None
    gameday: Optional[date] = None
    gametime: Optional[str] = None
    weekday: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    result: Optional[float] = None
    total: Optional[float] = None
    overtime: Optional[int] = None
    div_game: Optional[int] = None
    roof: Optional[str] = None
    surface: Optional[str] = None
    temp: Optional[float] = None
    wind: Optional[float] = None
    stadium: Optional[str] = None
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    away_moneyline: Optional[float] = None
    home_moneyline: Optional[float] = None
    away_spread_odds: Optional[float] = None
    home_spread_odds: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    home_qb_name: Optional[str] = None
    away_qb_name: Optional[str] = None
    home_rest: Optional[int] = None
    away_rest: Optional[int] = None


class SnapCountRow(BaseModel):
    """
    One row from nfl.load_snap_counts().
    Note: player_id here is 'pfr_player_id' (Pro Football Reference), not gsis_id.
    Docs: https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html
    """
    model_config = {"extra": "ignore"}

    game_id: str                                # required
    season: int
    week: int
    game_type: Optional[str] = None
    player: Optional[str] = None               # player name string
    pfr_player_id: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    opponent: Optional[str] = None
    offense_snaps: Optional[int] = None
    offense_pct: Optional[float] = None
    defense_snaps: Optional[int] = None
    defense_pct: Optional[float] = None
    st_snaps: Optional[int] = None
    st_pct: Optional[float] = None


class DepthChartRow(BaseModel):
    """
    One row from nfl.load_depth_charts().
    depth_team: 1=starter, 2=second string, 3=third string. Use for WR1/WR2/WR3 role.
    2025+ uses ESPN schema: team, pos_rank, pos_abb (no season/week) — we inject those.
    Docs: https://nflreadr.nflverse.com/articles/dictionary_depth_charts.html
    """
    model_config = {"extra": "ignore"}

    season: Optional[int] = None   # injected when absent (2025+ ESPN schema)
    week: Optional[int] = None    # injected when absent
    club_code: Optional[str] = None   # team (2025+) or club_code (legacy)
    team: Optional[str] = None       # 2025+ column
    game_type: Optional[str] = None
    depth_team: Optional[str] = None  # pos_rank (2025+) or depth_team (legacy)
    pos_rank: Optional[int] = None   # 2025+ column
    gsis_id: Optional[str] = None
    position: Optional[str] = None    # pos_abb (2025+) or position (legacy)
    pos_abb: Optional[str] = None    # 2025+ column
    depth_position: Optional[str] = None
    formation: Optional[str] = None


class NextGenStatsRow(BaseModel):
    """
    One row from nfl.load_nextgen_stats(). Weekly advanced metrics (2016+).
    Docs: https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html
    """
    model_config = {"extra": "ignore"}

    player_id: Optional[str] = None             # gsis_id
    player_display_name: Optional[str] = None
    season: int
    week: int
    avg_time_to_throw: Optional[float] = None
    avg_completion_above_expectation: Optional[float] = None
    avg_separation: Optional[float] = None     # WR/TE: yards from nearest defender at target
    avg_cushion: Optional[float] = None        # WR/TE: pre-snap distance from defender
    max_speed: Optional[float] = None          # fastest recorded speed (mph)
    avg_speed: Optional[float] = None


class TeamStatsRow(BaseModel):
    """
    One row from nfl.load_team_stats(). Team-level game stats for pace/game script.
    Docs: https://nflreadr.nflverse.com/articles/dictionary_team_stats.html
    """
    model_config = {"extra": "ignore"}

    team: Optional[str] = None
    season: int
    week: Optional[int] = None
    completions: Optional[int] = None
    attempts: Optional[int] = None
    passing_yards: Optional[float] = None
    carries: Optional[int] = None
    rushing_yards: Optional[float] = None
    total_plays: Optional[int] = None
    total_yards: Optional[float] = None


class FtnChartingRow(BaseModel):
    """
    One row from nfl.load_ftn_charting(). Play-level; 2022+.
    nflverse uses: nflverse_game_id, nflverse_play_id, season, week,
    is_drop, is_contested_ball. No receiver_id — store play-level only.
    """
    model_config = {"extra": "ignore"}
    game_id: Optional[str] = None           # map from nflverse_game_id
    nflverse_game_id: Optional[str] = None
    play_id: Optional[str | int] = None    # nflverse_play_id is int
    nflverse_play_id: Optional[int] = None
    season: Optional[int] = None
    week: Optional[int] = None
    is_drop: Optional[bool] = None
    is_contested_ball: Optional[bool] = None
    is_catchable_ball: Optional[bool] = None
    is_created_reception: Optional[bool] = None


class ParticipationRow(BaseModel):
    """
    One row from nfl.load_participation(). Play-level; 2016+.
    nflverse uses: nflverse_game_id, play_id (int), offense_players (;‑sep gsis_ids),
    defense_players, route. All Optional for lenient validation.
    """
    model_config = {"extra": "ignore"}
    game_id: Optional[str] = None           # map from nflverse_game_id
    nflverse_game_id: Optional[str] = None
    play_id: Optional[str | int | float] = None
    season: Optional[int] = None
    week: Optional[int] = None
    offense_players: Optional[str] = None  # ;‑separated gsis_ids (2023+)
    defense_players: Optional[str] = None
    route: Optional[str] = None


class CombineRow(BaseModel):
    """
    One row from nfl.load_combine(). 40yd, bench, vertical, broad; for cold-start.
    nflverse uses pfr_id (no gsis_id); player_id resolved in normalize via players.pfr_id.
    """
    model_config = {"extra": "ignore"}
    player_id: Optional[str] = None
    pfr_id: Optional[str] = None       # nflverse primary id; resolve→gsis_id in normalize
    season: Optional[int] = None
    forty: Optional[float] = None
    bench_press: Optional[int] = None
    vertical_jump: Optional[float] = None
    broad_jump: Optional[float] = None


VALIDATOR_MAP: dict[str, type[BaseModel]] = {
    SOURCE_PLAYER_STATS:    PlayerStatsRow,
    SOURCE_ROSTERS:        RosterRow,
    SOURCE_SCHEDULES:      ScheduleRow,
    SOURCE_SNAP_COUNTS:    SnapCountRow,
    SOURCE_DEPTH_CHARTS:   DepthChartRow,
    SOURCE_NEXTGEN_STATS:  NextGenStatsRow,
    SOURCE_TEAM_STATS:     TeamStatsRow,
    SOURCE_FTN_CHARTING:   FtnChartingRow,
    SOURCE_PARTICIPATION:  ParticipationRow,
    SOURCE_COMBINE:        CombineRow,
}

# Column name holding the week for each source (None = season-level, no week)
WEEK_COL: dict[str, Optional[str]] = {
    SOURCE_PLAYER_STATS:    "week",
    SOURCE_ROSTERS:        None,
    SOURCE_SCHEDULES:      "week",
    SOURCE_SNAP_COUNTS:    "week",
    SOURCE_DEPTH_CHARTS:   "week",
    SOURCE_NEXTGEN_STATS:  "week",
    SOURCE_TEAM_STATS:     "week",
    SOURCE_FTN_CHARTING:   "week",
    SOURCE_PARTICIPATION:  "week",
    SOURCE_COMBINE:        None,
}

# Most informative columns to show in dry-run preview per source
PREVIEW_KEYS: dict[str, list[str]] = {
    SOURCE_PLAYER_STATS: [
        "player_id", "player_display_name", "position", "team",
        "week", "receiving_yards", "fantasy_points_ppr",
    ],
    SOURCE_ROSTERS: [
        "gsis_id", "full_name", "position", "team", "years_exp", "status",
    ],
    SOURCE_SCHEDULES: [
        "game_id", "week", "home_team", "away_team",
        "home_score", "away_score", "roof",
    ],
    SOURCE_SNAP_COUNTS: [
        "game_id", "player", "position", "team",
        "offense_snaps", "offense_pct",
    ],
    SOURCE_DEPTH_CHARTS: [
        "gsis_id", "club_code", "position", "week", "depth_team",
    ],
    SOURCE_NEXTGEN_STATS: [
        "player_id", "season", "week", "avg_separation", "avg_cushion", "max_speed",
    ],
    SOURCE_TEAM_STATS: [
        "team", "season", "week", "attempts", "carries", "total_plays",
    ],
    SOURCE_FTN_CHARTING: [
        "nflverse_game_id", "nflverse_play_id", "season", "week",
        "is_drop", "is_contested_ball",
    ],
    SOURCE_PARTICIPATION: [
        "nflverse_game_id", "play_id", "offense_players", "defense_players", "route",
    ],
    SOURCE_COMBINE: [
        "player_id", "season", "forty", "bench_press", "vertical_jump",
    ],
}


# ── DDL — run once on first connect ──────────────────────────────────────────

_CREATE_STAGING = """
CREATE TABLE IF NOT EXISTS staging_nflreadpy (
    id           SERIAL       PRIMARY KEY,
    source_type  VARCHAR(50)  NOT NULL,
    season       INTEGER      NOT NULL,
    week         INTEGER,
    raw_data     JSONB        NOT NULL,
    ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    processed    BOOLEAN      NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_staging_nflreadpy_lookup
    ON staging_nflreadpy (source_type, season, week);
CREATE INDEX IF NOT EXISTS idx_staging_nflreadpy_unprocessed
    ON staging_nflreadpy (processed) WHERE NOT processed;
"""

_CREATE_DEAD_LETTER = """
CREATE TABLE IF NOT EXISTS dead_letter (
    id             SERIAL       PRIMARY KEY,
    source         VARCHAR(100) NOT NULL,
    error_message  TEXT         NOT NULL,
    raw_payload    JSONB        NOT NULL,
    ingested_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
"""


# ── Adapter ───────────────────────────────────────────────────────────────────

class NFLReadPyAdapter:
    """
    Fetches data from nflreadpy, validates each row with a Pydantic model,
    and bulk-inserts to PostgreSQL.

    Context-manager usage (recommended for live ingest):
        with NFLReadPyAdapter(os.environ["DATABASE_URL"]) as adapter:
            adapter.run_full_ingest(seasons=[2025])

    Dry-run (no DB required — use for local testing):
        adapter = NFLReadPyAdapter(db_url="")
        adapter.run_full_ingest(seasons=[2025], dry_run=True)
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None

    # ── Connection management ─────────────────────────────────────────────

    def connect(self) -> None:
        if not self._db_url:
            raise ValueError("db_url is empty — cannot connect to PostgreSQL.")
        dsn = _psycopg2_dsn(self._db_url)
        logger.info("Connecting to PostgreSQL…")
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_tables()
        logger.info("Connected and tables verified.")

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("DB connection closed.")

    def __enter__(self) -> "NFLReadPyAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn and not self._conn.closed:
            if exc_type:
                self._conn.rollback()
            self.close()

    # ── DDL ───────────────────────────────────────────────────────────────

    def _ensure_tables(self) -> None:
        """Create staging and dead_letter tables if they don't exist."""
        assert self._conn
        from pipeline.schema import ensure_schema

        ensure_schema(self._conn)

    # ── DB writes ─────────────────────────────────────────────────────────

    def _flush_staging(self, batch: list[tuple]) -> None:
        """Bulk-insert a batch of validated rows into staging_nflreadpy."""
        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO staging_nflreadpy
                    (source_type, season, week, raw_data, ingested_at)
                VALUES %s
                """,
                batch,
            )
        self._conn.commit()

    def _write_dead_letter(self, source: str, error: str, raw: dict) -> None:
        """
        Insert one failed-validation row into dead_letter.
        CLAUDE.md §3: failed rows go to dead_letter — never silently dropped.

        Uses _safe_pg_json() to handle date/datetime objects in raw dicts
        (coerced Polars rows may contain Python date objects).
        Non-fatal: logs and rolls back on DB error rather than propagating.
        """
        assert self._conn
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dead_letter
                        (source, error_message, raw_payload, ingested_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (source, error, _safe_pg_json(raw)),
                )
            self._conn.commit()
        except Exception as dl_exc:
            # Never let dead-letter writes crash the validation loop.
            logger.warning(
                "dead_letter write failed (source=%s): %s", source, dl_exc
            )
            try:
                self._conn.rollback()
            except Exception:
                pass

    # ── nflreadpy calls with exponential backoff ──────────────────────────

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _load_with_retry(self, loader_fn, **kwargs):
        """
        Wrap a nflreadpy load_* call with exponential backoff.
        Retries 3 times: waits 2s → 4s → 8s (capped at 60s).
        CLAUDE.md §3: exponential back-off on 429s / network failures.
        """
        return loader_fn(**kwargs)

    # ── Core validation loop ──────────────────────────────────────────────

    def _process_source(
        self,
        source_type: str,
        df,                          # polars.DataFrame from nflreadpy
        season: int,
        week_col: Optional[str],
        dry_run: bool,
    ) -> tuple[int, int, list[dict]]:
        """
        Iterate a Polars DataFrame row-by-row:
          1. _coerce_row: NaN → None
          2. Pydantic validation
          3a. Success  → bulk insert into staging_nflreadpy
          3b. Failure  → dead_letter (never silent)

        Returns: (success_count, failure_count, first_5_valid_rows)
        The preview list is always built; only printed in dry_run mode.
        """
        validator_cls = VALIDATOR_MAP[source_type]
        staging_batch: list[tuple] = []
        success_count = 0
        failure_count = 0
        preview: list[dict] = []
        source_label = f"nflreadpy.{source_type}"
        ingested_at = datetime.utcnow()

        records: list[dict] = df.to_dicts()
        logger.info("  Validating %d rows [%s season=%d]…", len(records), source_type, season)

        for raw_row in records:
            coerced = _coerce_row(raw_row)
            # Normalize player_id → gsis_id for depth_charts (nflverse uses either)
            if source_type == SOURCE_DEPTH_CHARTS:
                if coerced.get("gsis_id") is None:
                    coerced["gsis_id"] = coerced.get("player_id")
                # 2025+ ESPN schema lacks season/week — inject from fetch context
                if coerced.get("season") is None:
                    coerced["season"] = season
                if coerced.get("week") is None:
                    coerced["week"] = 1  # ESPN daily charts; use week 1 as placeholder
                # Map 2025+ columns to legacy names for normalize compatibility
                if coerced.get("club_code") is None and coerced.get("team"):
                    coerced["club_code"] = coerced["team"]
                if coerced.get("depth_team") is None and coerced.get("pos_rank") is not None:
                    coerced["depth_team"] = str(coerced["pos_rank"])
                if coerced.get("position") is None and coerced.get("pos_abb"):
                    coerced["position"] = coerced["pos_abb"]
            # Map nflverse columns for FTN + participation (play-level schemas)
            if source_type == SOURCE_FTN_CHARTING:
                coerced["game_id"] = coerced.get("game_id") or coerced.get("nflverse_game_id")
                coerced["play_id"] = coerced.get("play_id") or coerced.get("nflverse_play_id")
            elif source_type == SOURCE_PARTICIPATION:
                coerced["game_id"] = coerced.get("game_id") or coerced.get("nflverse_game_id")
                coerced["season"] = coerced.get("season") or season
                # Parse week from nflverse_game_id (e.g. 2019_01_GB_CHI → 1)
                if coerced.get("week") is None and coerced.get("game_id"):
                    parts = str(coerced["game_id"]).split("_")
                    if len(parts) >= 2:
                        try:
                            coerced["week"] = int(parts[1])
                        except (ValueError, TypeError):
                            pass
            # Map nflverse player ID columns for nextgen + combine
            elif source_type == SOURCE_NEXTGEN_STATS:
                coerced["player_id"] = coerced.get("player_id") or coerced.get("player_gsis_id")
                coerced["avg_completion_above_expectation"] = (
                    coerced.get("avg_completion_above_expectation")
                    or coerced.get("completion_percentage_above_expectation")
                )
            elif source_type == SOURCE_COMBINE:
                coerced["bench_press"] = coerced.get("bench_press") or coerced.get("bench")
                coerced["vertical_jump"] = coerced.get("vertical_jump") or coerced.get("vertical")

            try:
                validated = validator_cls.model_validate(coerced)
            except ValidationError as exc:
                failure_count += 1
                if not dry_run and self._conn:
                    self._write_dead_letter(source_label, str(exc), coerced)
                else:
                    logger.debug("Dead-letter (dry-run): %s", exc)
                continue

            success_count += 1
            row_dict = validated.model_dump(mode="json")

            if len(preview) < 5:
                preview.append(row_dict)

            if not dry_run and self._conn:
                week = getattr(validated, week_col, None) if week_col else None
                staging_batch.append(
                    (source_type, season, week, _safe_pg_json(row_dict), ingested_at)
                )
                if len(staging_batch) >= BATCH_SIZE:
                    self._flush_staging(staging_batch)
                    staging_batch.clear()

        # Flush remainder
        if not dry_run and self._conn and staging_batch:
            self._flush_staging(staging_batch)

        return success_count, failure_count, preview

    # ── Public fetch methods ──────────────────────────────────────────────

    def fetch_player_stats(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """
        Load weekly player stats and write to staging.
        Returns: (total_success, total_failure) across all requested seasons.
        """
        import nflreadpy as nfl

        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_player_stats: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_player_stats, seasons=season)
            except Exception as exc:
                logger.error("load_player_stats FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_PLAYER_STATS, df, season, WEEK_COL[SOURCE_PLAYER_STATS], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_PLAYER_STATS, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_rosters(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load season rosters and write to staging."""
        import nflreadpy as nfl

        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_rosters: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_rosters, seasons=season)
            except Exception as exc:
                logger.error("load_rosters FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_ROSTERS, df, season, WEEK_COL[SOURCE_ROSTERS], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_ROSTERS, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_schedules(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load game schedules and write to staging."""
        import nflreadpy as nfl

        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_schedules: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_schedules, seasons=season)
            except Exception as exc:
                logger.error("load_schedules FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_SCHEDULES, df, season, WEEK_COL[SOURCE_SCHEDULES], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_SCHEDULES, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_snap_counts(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load snap counts and write to staging (available from 2012 onwards)."""
        import nflreadpy as nfl

        valid_seasons = [s for s in seasons if s >= 2012]
        if skipped := [s for s in seasons if s < 2012]:
            logger.warning("Snap counts unavailable before 2012, skipping: %s", skipped)

        total_ok = total_fail = 0
        for season in valid_seasons:
            logger.info("fetch_snap_counts: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_snap_counts, seasons=season)
            except Exception as exc:
                logger.error("load_snap_counts FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_SNAP_COUNTS, df, season, WEEK_COL[SOURCE_SNAP_COUNTS], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_SNAP_COUNTS, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_depth_charts(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load depth charts and write to staging (available from 2001 onwards)."""
        import nflreadpy as nfl

        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_depth_charts: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_depth_charts, seasons=season)
            except Exception as exc:
                logger.error("load_depth_charts FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_DEPTH_CHARTS, df, season, WEEK_COL[SOURCE_DEPTH_CHARTS], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_DEPTH_CHARTS, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_nextgen_stats(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load Next Gen Stats and write to staging (available from 2016 onwards)."""
        import nflreadpy as nfl

        valid_seasons = [s for s in seasons if s >= 2016]
        if skipped := [s for s in seasons if s < 2016]:
            logger.warning("Next Gen Stats unavailable before 2016, skipping: %s", skipped)

        total_ok = total_fail = 0
        for season in valid_seasons:
            logger.info("fetch_nextgen_stats: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_nextgen_stats, seasons=season)
            except Exception as exc:
                logger.error("load_nextgen_stats FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_NEXTGEN_STATS, df, season, WEEK_COL[SOURCE_NEXTGEN_STATS], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_NEXTGEN_STATS, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_team_stats(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load team stats and write to staging."""
        import nflreadpy as nfl

        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_team_stats: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_team_stats, seasons=season)
            except Exception as exc:
                logger.error("load_team_stats FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_TEAM_STATS, df, season, WEEK_COL[SOURCE_TEAM_STATS], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_TEAM_STATS, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_ftn_charting(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load FTN charting (drops, contested catches); 2022+."""
        import nflreadpy as nfl
        total_ok = total_fail = 0
        for season in seasons:
            if season < 2022:
                continue
            logger.info("fetch_ftn_charting: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_ftn_charting, seasons=season)
            except Exception as exc:
                logger.error("load_ftn_charting FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_FTN_CHARTING, df, season, WEEK_COL[SOURCE_FTN_CHARTING], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_FTN_CHARTING, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_participation(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load participation (routes run); 2016+."""
        import nflreadpy as nfl
        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_participation: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_participation, seasons=season)
            except Exception as exc:
                logger.error("load_participation FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_PARTICIPATION, df, season, WEEK_COL[SOURCE_PARTICIPATION], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_PARTICIPATION, season, ok, fail, preview)
        return total_ok, total_fail

    def fetch_combine(
        self, seasons: list[int], dry_run: bool = False
    ) -> tuple[int, int]:
        """Load combine (40yd, bench, vertical); draft-year level."""
        import nflreadpy as nfl
        total_ok = total_fail = 0
        for season in seasons:
            logger.info("fetch_combine: season=%d", season)
            try:
                df = self._load_with_retry(nfl.load_combine, seasons=season)
            except Exception as exc:
                logger.error("load_combine FAILED season=%d: %s", season, exc)
                continue
            ok, fail, preview = self._process_source(
                SOURCE_COMBINE, df, season, WEEK_COL[SOURCE_COMBINE], dry_run
            )
            total_ok += ok
            total_fail += fail
            _print_preview(SOURCE_COMBINE, season, ok, fail, preview)
        return total_ok, total_fail

    def run_full_ingest(
        self,
        seasons: list[int],
        sources: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> dict[str, tuple[int, int]]:
        """
        Orchestrate all four sources sequentially with a 1-second gap between each.
        CLAUDE.md §3: minimum 1-second delay between external requests.

        Returns: {source_type: (success_count, failure_count)}
        """
        sources = sources or ALL_SOURCES
        results: dict[str, tuple[int, int]] = {}

        dispatch = {
            SOURCE_PLAYER_STATS:    self.fetch_player_stats,
            SOURCE_ROSTERS:        self.fetch_rosters,
            SOURCE_SCHEDULES:      self.fetch_schedules,
            SOURCE_SNAP_COUNTS:    self.fetch_snap_counts,
            SOURCE_DEPTH_CHARTS:   self.fetch_depth_charts,
            SOURCE_NEXTGEN_STATS:  self.fetch_nextgen_stats,
            SOURCE_TEAM_STATS:     self.fetch_team_stats,
            SOURCE_FTN_CHARTING:   self.fetch_ftn_charting,
            SOURCE_PARTICIPATION:  self.fetch_participation,
            SOURCE_COMBINE:        self.fetch_combine,
        }

        for i, source in enumerate(sources):
            if i > 0:
                logger.debug("Sleeping %.1fs between sources…", MIN_DELAY_BETWEEN_SOURCES)
                time.sleep(MIN_DELAY_BETWEEN_SOURCES)
            ok, fail = dispatch[source](seasons, dry_run=dry_run)
            results[source] = (ok, fail)

        _print_summary(results)
        return results


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_preview(
    source: str, season: int, ok: int, fail: int, rows: list[dict]
) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {source.upper()}  season={season}  ✓ {ok} rows  ✗ {fail} dead-letter")
    print(f"{'─' * 72}")
    if not rows:
        print("  (no valid rows returned)")
        return
    keys = PREVIEW_KEYS.get(source, list(rows[0].keys())[:6])
    col_w = 22
    header = "  " + "  ".join(f"{k:<{col_w}}" for k in keys)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        vals = []
        for k in keys:
            v = row.get(k)
            s = "-" if v is None or v == "" else str(v)
            vals.append(s[:col_w])
        print("  " + "  ".join(f"{v:<{col_w}}" for v in vals))


def _print_summary(results: dict[str, tuple[int, int]]) -> None:
    print(f"\n{'═' * 72}")
    print("  INGEST SUMMARY")
    print(f"{'═' * 72}")
    total_ok = total_fail = 0
    for source, (ok, fail) in results.items():
        icon = "✓" if fail == 0 else "⚠"
        print(f"  {icon}  {source:<20}  {ok:>6} staged   {fail:>4} dead-letter")
        total_ok += ok
        total_fail += fail
    print(f"{'─' * 72}")
    print(f"     {'TOTAL':<20}  {total_ok:>6} staged   {total_fail:>4} dead-letter")
    print(f"{'═' * 72}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest NFL data from nflreadpy into staging_nflreadpy."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=None, metavar="YEAR",
        help="Season year(s) to ingest (e.g. --seasons 2024 2025). Default: current season.",
    )
    parser.add_argument(
        "--source", choices=ALL_SOURCES + ["all"], default="all",
        help="Which data source to fetch. Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and preview data without writing to the database.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve seasons — default to current NFL season
    if args.seasons:
        seasons = args.seasons
    else:
        try:
            import nflreadpy as nfl
            seasons = [nfl.get_current_season()]
            logger.info("Resolved current season: %d", seasons[0])
        except Exception:
            seasons = [2025]
            logger.info("Could not resolve current season; defaulting to 2025.")

    sources = ALL_SOURCES if args.source == "all" else [args.source]
    logger.info("Seasons=%s  Sources=%s  dry_run=%s", seasons, sources, args.dry_run)

    db_url = os.environ.get("DATABASE_URL", "")

    if args.dry_run:
        adapter = NFLReadPyAdapter(db_url="")
        adapter.run_full_ingest(seasons=seasons, sources=sources, dry_run=True)
    else:
        if not db_url:
            logger.error(
                "DATABASE_URL not set. Use --dry-run to test without a database."
            )
            raise SystemExit(1)
        with NFLReadPyAdapter(db_url) as adapter:
            adapter.run_full_ingest(seasons=seasons, sources=sources, dry_run=False)


if __name__ == "__main__":
    main()
