"""
pipeline/normalize.py

Reads validated rows from staging_nflreadpy and upserts them into the four
core production tables: teams, players, games, game_logs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FK-SAFE PROCESSING ORDER — THIS SEQUENCE IS MANDATORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — schedules  →  teams  +  games
  • game_logs.game_id has a FK constraint referencing games.id
  • games.home_team / away_team reference team abbreviations
  • teams must be inserted before games (FK on team abbreviations isn't
    enforced in current DDL, but inserting teams first is logically correct)
  • games must be inserted before game_logs

Step 2 — rosters  →  players
  • game_logs.player_id has a FK constraint referencing players.id
  • players must exist before game_logs can reference them

Step 3 — player_stats  →  game_logs
  • game_logs FK references BOTH games.id AND players.id
  • If this step runs before Step 1 or Step 2, every row will raise
    psycopg2.IntegrityError because the referenced IDs don't exist yet
  • Breaking this order causes FK violations on every INSERT — the rows
    are silently counted as errors in NormalizeSummary.errors

Step 4 — snap_counts  →  updates game_logs.offense_snaps / offense_pct
  (join via player full_name exact match, with rapidfuzz fuzzy fallback)

DO NOT REORDER these steps. If you're adding a new source type, identify
its FK dependencies and insert it at the correct position in the sequence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GAME_ID DERIVATION (seasons 2019, 2020, 2021, 2024)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

nflreadpy omits the game_id column from player_stats rows for these seasons.
_build_game_id_lookup() reconstructs it by joining on (season, week,
home_team, away_team) from the schedules feed. Any row that can't be matched
is counted in NormalizeSummary.errors and skipped — never silently dropped.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS FILE IS ~1500 LINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This file intentionally uses raw psycopg2 + execute_values rather than an
ORM. The bulk-upsert patterns (ON CONFLICT DO UPDATE, execute_values batches)
require direct SQL that an ORM would obscure or make slower. The length is
accounted for by four distinct normalization passes (one per step above) each
with their own query logic, plus the game_id derivation and snap-count fuzzy
matching. It is not incidental complexity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pure transform functions (schedule_row_to_game, etc.) are separated from
DB logic so they can be unit-tested without a database connection.

Standalone usage:
  python pipeline/normalize.py --seasons 2025
  python pipeline/normalize.py --dry-run --seasons 2025
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path so scraper/ is importable when run
# directly (e.g. python pipeline/normalize.py) or from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2
from rapidfuzz import fuzz, process
import psycopg2.extras
from psycopg2.extras import execute_values
from pydantic import ValidationError

from scraper.adapters.nflreadpy_adapter import (
    SOURCE_PLAYER_STATS,
    SOURCE_ROSTERS,
    SOURCE_SCHEDULES,
    SOURCE_SNAP_COUNTS,
    SOURCE_DEPTH_CHARTS,
    SOURCE_NEXTGEN_STATS,
    SOURCE_TEAM_STATS,
    SOURCE_FTN_CHARTING,
    SOURCE_PARTICIPATION,
    SOURCE_COMBINE,
    PlayerStatsRow,
    RosterRow,
    ScheduleRow,
    SnapCountRow,
    _coerce_row,
    _psycopg2_dsn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


# ── DDL — production tables (idempotent) ─────────────────────────────────────

_CREATE_PRODUCTION = """
CREATE TABLE IF NOT EXISTS teams (
    id          VARCHAR(10)  PRIMARY KEY,
    name        TEXT,
    city        TEXT,
    stadium     TEXT,
    roof_type   TEXT,
    surface     TEXT,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS players (
    id           TEXT         PRIMARY KEY,
    full_name    TEXT,
    position     TEXT,
    team         TEXT,
    height       TEXT,
    weight       FLOAT,
    birth_date   DATE,
    college      TEXT,
    years_exp    INTEGER,
    entry_year   INTEGER,
    status       TEXT,
    headshot_url TEXT,
    espn_id      TEXT,
    pfr_id       TEXT,
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS games (
    id               TEXT         PRIMARY KEY,
    season           INTEGER      NOT NULL,
    week             INTEGER      NOT NULL,
    game_type        TEXT,
    home_team        TEXT         NOT NULL,
    away_team        TEXT         NOT NULL,
    home_score       INTEGER,
    away_score       INTEGER,
    gameday          DATE,
    gametime         TEXT,
    weekday          TEXT,
    stadium          TEXT,
    roof             TEXT,
    surface          TEXT,
    temp             FLOAT,
    wind             FLOAT,
    spread_line      FLOAT,
    total_line       FLOAT,
    away_moneyline   FLOAT,
    home_moneyline   FLOAT,
    home_rest        INTEGER,
    away_rest        INTEGER,
    home_qb_name     TEXT,
    away_qb_name     TEXT
);

CREATE TABLE IF NOT EXISTS game_logs (
    id                           SERIAL   PRIMARY KEY,
    player_id                    TEXT     NOT NULL REFERENCES players(id),
    game_id                      TEXT     NOT NULL REFERENCES games(id),
    season                       INTEGER  NOT NULL,
    week                         INTEGER  NOT NULL,
    season_type                  TEXT,
    team                         TEXT,
    opponent_team                TEXT,
    position                     TEXT,
    completions                  INTEGER,
    attempts                     INTEGER,
    passing_yards                FLOAT,
    passing_tds                  INTEGER,
    passing_interceptions        INTEGER,
    passing_air_yards            FLOAT,
    passing_yards_after_catch    FLOAT,
    passing_first_downs          INTEGER,
    passing_epa                  FLOAT,
    passing_cpoe                 FLOAT,
    carries                      INTEGER,
    rushing_yards                FLOAT,
    rushing_tds                  INTEGER,
    rushing_fumbles              INTEGER,
    receiving_fumbles            INTEGER,
    sack_fumbles                 INTEGER,
    rushing_epa                  FLOAT,
    receptions                   INTEGER,
    targets                      INTEGER,
    receiving_yards              FLOAT,
    receiving_tds                INTEGER,
    receiving_air_yards          FLOAT,
    receiving_yards_after_catch  FLOAT,
    receiving_first_downs        INTEGER,
    receiving_epa                FLOAT,
    racr                         FLOAT,
    target_share                 FLOAT,
    air_yards_share              FLOAT,
    wopr                         FLOAT,
    offense_snaps                INTEGER,
    offense_pct                  FLOAT,
    fantasy_points               FLOAT,
    fantasy_points_ppr           FLOAT,
    CONSTRAINT uq_game_logs_player_game UNIQUE (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_game_logs_player_season
    ON game_logs (player_id, season);
CREATE INDEX IF NOT EXISTS idx_game_logs_season_week
    ON game_logs (season, week);

CREATE TABLE IF NOT EXISTS depth_charts (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    team        TEXT NOT NULL,
    position    TEXT,
    depth_rank  FLOAT NOT NULL,
    PRIMARY KEY (player_id, season, week)
);
CREATE INDEX IF NOT EXISTS idx_depth_charts_lookup
    ON depth_charts (player_id, season, week);

CREATE TABLE IF NOT EXISTS nextgen_stats (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    avg_separation  FLOAT,
    avg_cushion    FLOAT,
    max_speed      FLOAT,
    avg_completion_above_expectation FLOAT,
    PRIMARY KEY (player_id, season, week)
);
CREATE INDEX IF NOT EXISTS idx_nextgen_stats_lookup
    ON nextgen_stats (player_id, season, week);

CREATE TABLE IF NOT EXISTS team_game_stats (
    team        TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    pass_attempts   INTEGER,
    rush_attempts   INTEGER,
    total_plays     INTEGER,
    total_yards     FLOAT,
    PRIMARY KEY (team, season, week)
);

CREATE TABLE IF NOT EXISTS ftn_play (
    game_id     TEXT NOT NULL,
    play_id     INTEGER NOT NULL,
    season      INTEGER,
    week        INTEGER,
    is_drop     BOOLEAN DEFAULT FALSE,
    is_contested_ball BOOLEAN DEFAULT FALSE,
    is_catchable_ball BOOLEAN,
    is_created_reception BOOLEAN,
    PRIMARY KEY (game_id, play_id)
);
CREATE INDEX IF NOT EXISTS idx_ftn_play_lookup
    ON ftn_play (game_id, season, week);

CREATE TABLE IF NOT EXISTS ftn_player_game (
    player_id   TEXT NOT NULL,
    game_id     TEXT NOT NULL,
    season      INTEGER,
    week        INTEGER,
    drops       INTEGER DEFAULT 0,
    contested_catches INTEGER DEFAULT 0,
    targets     INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_ftn_player_game_lookup
    ON ftn_player_game (player_id, season, week);

CREATE TABLE IF NOT EXISTS participation_player_game (
    player_id   TEXT NOT NULL,
    game_id     TEXT NOT NULL,
    season      INTEGER,
    week        INTEGER,
    routes_run  INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_participation_player_game_lookup
    ON participation_player_game (player_id, season, week);

CREATE TABLE IF NOT EXISTS combine (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    forty       FLOAT,
    bench_press INTEGER,
    vertical_jump FLOAT,
    broad_jump  FLOAT,
    PRIMARY KEY (player_id, season)
);
"""


# ── Pure transform functions ──────────────────────────────────────────────────
# These functions are pure: no DB access, no side effects.
# Fully covered by unit tests in backend/tests/test_normalize.py.

def schedule_row_to_game(row: ScheduleRow) -> dict[str, Any]:
    """Map a validated ScheduleRow → dict ready for INSERT INTO games."""
    return {
        "id":             row.game_id,
        "season":         row.season,
        "week":           row.week,
        "game_type":      row.game_type,
        "home_team":      row.home_team,
        "away_team":      row.away_team,
        "home_score":     row.home_score,
        "away_score":     row.away_score,
        "gameday":        row.gameday,
        "gametime":       row.gametime,
        "weekday":        row.weekday,
        "stadium":        row.stadium,
        "roof":           row.roof,
        "surface":        row.surface,
        "temp":           row.temp,
        "wind":           row.wind,
        "spread_line":    row.spread_line,
        "total_line":     row.total_line,
        "away_moneyline": row.away_moneyline,
        "home_moneyline": row.home_moneyline,
        "home_rest":      row.home_rest,
        "away_rest":      row.away_rest,
        "home_qb_name":   row.home_qb_name,
        "away_qb_name":   row.away_qb_name,
    }


def schedule_row_to_teams(row: ScheduleRow) -> list[dict[str, Any]]:
    """
    Extract team stubs from a ScheduleRow.
    Returns [{"id": home_abbr}, {"id": away_abbr}].
    Full team metadata (stadium, surface) comes from the teams API — here
    we just establish the primary key so game_logs FK constraints hold.
    """
    return [{"id": row.home_team}, {"id": row.away_team}]


def _roster_height_to_str(h: Optional[str | float]) -> Optional[str]:
    """Convert nflverse height (float inches) to string, e.g. 72.0 → '6-0'."""
    if h is None:
        return None
    if isinstance(h, str):
        return h
    try:
        inches = int(round(float(h)))
        return f"{inches // 12}-{inches % 12}"
    except (TypeError, ValueError):
        return None


def roster_row_to_player(row: RosterRow) -> Optional[dict[str, Any]]:
    """
    Map a validated RosterRow → players dict.
    Returns None if gsis_id is absent (no usable primary key).
    """
    if not row.gsis_id:
        return None
    return {
        "id":           row.gsis_id,
        "full_name":    row.full_name,
        "position":     row.position,
        "team":         row.team,
        "height":       _roster_height_to_str(row.height),
        "weight":       row.weight,
        "birth_date":   row.birth_date,
        "college":      row.college,
        "years_exp":    row.years_exp,
        "entry_year":   row.entry_year,
        "status":       row.status,
        "headshot_url": row.headshot_url,
        "espn_id":      str(row.espn_id) if row.espn_id is not None else None,
        "pfr_id":       row.pfr_id,
    }


def player_stats_row_to_game_log(row: PlayerStatsRow) -> Optional[dict[str, Any]]:
    """
    Map a validated PlayerStatsRow → game_logs dict.
    Returns None if game_id is absent (no usable FK).
    """
    if not row.game_id:
        return None
    return {
        "player_id":                   row.player_id,
        "game_id":                     row.game_id,
        "season":                      row.season,
        "week":                        row.week,
        "season_type":                 row.season_type,
        "team":                        row.team,
        "opponent_team":               row.opponent_team,
        "position":                    row.position,
        "completions":                 row.completions,
        "attempts":                    row.attempts,
        "passing_yards":               row.passing_yards,
        "passing_tds":                 row.passing_tds,
        "passing_interceptions":       row.passing_interceptions,
        "passing_air_yards":           row.passing_air_yards,
        "passing_yards_after_catch":   row.passing_yards_after_catch,
        "passing_first_downs":         row.passing_first_downs,
        "passing_epa":                 row.passing_epa,
        "passing_cpoe":                row.passing_cpoe,
        "carries":                     row.carries,
        "rushing_yards":               row.rushing_yards,
        "rushing_tds":                 row.rushing_tds,
        "rushing_fumbles":             row.rushing_fumbles,
        "receiving_fumbles":           row.receiving_fumbles,
        "sack_fumbles":                row.sack_fumbles,
        "rushing_epa":                 row.rushing_epa,
        "receptions":                  row.receptions,
        "targets":                     row.targets,
        "receiving_yards":             row.receiving_yards,
        "receiving_tds":               row.receiving_tds,
        "receiving_air_yards":         row.receiving_air_yards,
        "receiving_yards_after_catch": row.receiving_yards_after_catch,
        "receiving_first_downs":       row.receiving_first_downs,
        "receiving_epa":               row.receiving_epa,
        "racr":                        row.racr,
        "target_share":                row.target_share,
        "air_yards_share":             row.air_yards_share,
        "wopr":                        row.wopr,
        "offense_snaps":               None,   # filled by snap_counts pass
        "offense_pct":                 None,
        "fantasy_points":              row.fantasy_points,
        "fantasy_points_ppr":          row.fantasy_points_ppr,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

@dataclass
class NormalizeSummary:
    teams_upserted: int = 0
    players_upserted: int = 0
    games_upserted: int = 0
    game_logs_upserted: int = 0
    staging_rows_processed: int = 0
    skipped: int = 0
    errors: int = 0

    def log(self) -> None:
        logger.info(
            "Normalize complete: %d teams | %d players | %d games | %d game_logs "
            "| %d staging rows | %d skipped | %d errors",
            self.teams_upserted, self.players_upserted, self.games_upserted,
            self.game_logs_upserted, self.staging_rows_processed,
            self.skipped, self.errors,
        )
        print(
            f"\n{'═' * 72}\n"
            f"  NORMALIZE SUMMARY\n"
            f"{'═' * 72}\n"
            f"  Teams upserted    : {self.teams_upserted}\n"
            f"  Players upserted  : {self.players_upserted}\n"
            f"  Games upserted    : {self.games_upserted}\n"
            f"  Game logs upserted: {self.game_logs_upserted}\n"
            f"  Staging rows      : {self.staging_rows_processed} processed\n"
            f"  Skipped           : {self.skipped}\n"
            f"  Errors            : {self.errors}\n"
            f"{'═' * 72}\n"
        )


# ── Normalizer class ──────────────────────────────────────────────────────────

class Normalizer:
    """
    Reads unprocessed rows from staging_nflreadpy, applies transform functions,
    and upserts into production tables.

    Usage:
        with Normalizer(os.environ["DATABASE_URL"]) as norm:
            summary = norm.run(seasons=[2025])
            summary.log()
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        dsn = _psycopg2_dsn(self._db_url)
        logger.info("Normalizer: connecting to PostgreSQL…")
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_production_tables()
        logger.info("Normalizer: connected and production tables verified.")

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "Normalizer":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn and not self._conn.closed:
            if exc_type:
                self._conn.rollback()
            self.close()

    # ── DDL ───────────────────────────────────────────────────────────────

    def _ensure_production_tables(self) -> None:
        assert self._conn
        # Alembic-managed schema; CREATE IF NOT EXISTS remains as empty-DB bootstrap.
        from pipeline.schema import ensure_schema

        ensure_schema(self._conn)
        with self._conn.cursor() as cur:
            # Narrow backfills kept here until promoted into a dedicated revision.
            cur.execute(
                "ALTER TABLE game_logs ADD COLUMN IF NOT EXISTS receiving_fumbles INTEGER"
            )
            cur.execute(
                "ALTER TABLE game_logs ADD COLUMN IF NOT EXISTS sack_fumbles INTEGER"
            )
            cur.execute(
                "ALTER TABLE games ADD COLUMN IF NOT EXISTS precipitation_bucket SMALLINT"
            )
        self._conn.commit()

    # ── Staging reader ────────────────────────────────────────────────────

    def _fetch_unprocessed(
        self, source_type: str, seasons: Optional[list[int]] = None
    ) -> list[dict[str, Any]]:
        """Return raw_data dicts for unprocessed staging rows of a given source."""
        assert self._conn
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if seasons:
                cur.execute(
                    """
                    SELECT id, raw_data FROM staging_nflreadpy
                    WHERE source_type = %s AND NOT processed AND season = ANY(%s)
                    ORDER BY id
                    """,
                    (source_type, seasons),
                )
            else:
                cur.execute(
                    """
                    SELECT id, raw_data FROM staging_nflreadpy
                    WHERE source_type = %s AND NOT processed
                    ORDER BY id
                    """,
                    (source_type,),
                )
            return [dict(r) for r in cur.fetchall()]

    def _mark_processed(self, staging_ids: list[int]) -> None:
        assert self._conn
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE staging_nflreadpy SET processed = TRUE WHERE id = ANY(%s)",
                (staging_ids,),
            )
        self._conn.commit()

    # ── Upsert helpers ────────────────────────────────────────────────────

    def _upsert_teams(self, team_dicts: list[dict]) -> int:
        """Insert team stubs if they don't exist yet. Returns # rows affected."""
        if not team_dicts:
            return 0
        # Deduplicate by id
        seen: set[str] = set()
        unique = []
        for t in team_dicts:
            if t["id"] not in seen:
                seen.add(t["id"])
                unique.append(t)

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO teams (id, updated_at)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                [(t["id"], datetime.utcnow()) for t in unique],
            )
        self._conn.commit()
        return len(unique)

    def _upsert_games(self, game_dicts: list[dict]) -> int:
        if not game_dicts:
            return 0
        cols = [
            "id", "season", "week", "game_type", "home_team", "away_team",
            "home_score", "away_score", "gameday", "gametime", "weekday",
            "stadium", "roof", "surface", "temp", "wind",
            "spread_line", "total_line", "away_moneyline", "home_moneyline",
            "home_rest", "away_rest", "home_qb_name", "away_qb_name",
        ]
        rows = [tuple(g.get(c) for c in cols) for g in game_dicts]

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO games (
                    id, season, week, game_type, home_team, away_team,
                    home_score, away_score, gameday, gametime, weekday,
                    stadium, roof, surface, temp, wind,
                    spread_line, total_line, away_moneyline, home_moneyline,
                    home_rest, away_rest, home_qb_name, away_qb_name
                ) VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    home_score     = EXCLUDED.home_score,
                    away_score     = EXCLUDED.away_score,
                    roof           = EXCLUDED.roof,
                    surface        = EXCLUDED.surface,
                    temp           = EXCLUDED.temp,
                    wind           = EXCLUDED.wind,
                    spread_line    = EXCLUDED.spread_line,
                    total_line     = EXCLUDED.total_line,
                    home_qb_name   = EXCLUDED.home_qb_name,
                    away_qb_name   = EXCLUDED.away_qb_name,
                    away_moneyline = EXCLUDED.away_moneyline,
                    home_moneyline = EXCLUDED.home_moneyline
                """,
                rows,
            )
        self._conn.commit()
        return len(game_dicts)

    def _upsert_players(self, player_dicts: list[dict]) -> int:
        if not player_dicts:
            return 0
        cols = [
            "id", "full_name", "position", "team", "height", "weight",
            "birth_date", "college", "years_exp", "entry_year",
            "status", "headshot_url", "espn_id", "pfr_id",
        ]
        rows = [tuple(p.get(c) for c in cols) for p in player_dicts]

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO players (
                    id, full_name, position, team, height, weight,
                    birth_date, college, years_exp, entry_year,
                    status, headshot_url, espn_id, pfr_id, updated_at
                ) VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    full_name    = EXCLUDED.full_name,
                    position     = EXCLUDED.position,
                    team         = EXCLUDED.team,
                    status       = EXCLUDED.status,
                    pfr_id       = COALESCE(EXCLUDED.pfr_id, players.pfr_id),
                    updated_at   = NOW()
                """,
                [r + (datetime.utcnow(),) for r in rows],
            )
        self._conn.commit()
        return len(player_dicts)

    def _upsert_player_stubs_from_stats(self, staging_rows: list[dict]) -> int:
        """
        Upsert minimal player stubs from player_stats staging rows.

        load_rosters() only returns the *current* active roster snapshot, so
        many historical player_ids (traded, cut, injured players) are absent
        from the players table. This method fills that gap by extracting
        player_id + display_name + position + team from player_stats rows and
        doing an ON CONFLICT DO NOTHING insert — existing roster records are
        left untouched; only genuinely missing players get a stub row.

        Must be called BEFORE _upsert_game_logs to satisfy the FK constraint.
        """
        stubs: dict[str, dict] = {}
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)  # accept both staging wrapper and raw dict
            pid = raw.get("player_id")
            if not pid:
                continue
            if pid not in stubs:
                stubs[pid] = {
                    "id":        pid,
                    "full_name": raw.get("player_display_name") or raw.get("player_name"),
                    "position":  raw.get("position"),
                    "team":      raw.get("team"),
                }

        if not stubs:
            return 0

        stub_list = list(stubs.values())
        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO players (id, full_name, position, team, updated_at)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    (s["id"], s["full_name"], s["position"], s["team"], datetime.utcnow())
                    for s in stub_list
                ],
            )
        self._conn.commit()
        logger.info("Upserted %d player stubs from player_stats", len(stub_list))
        return len(stub_list)

    def _build_game_id_lookup(self) -> dict[tuple, str]:
        """
        Build a lookup dict {(season, week, team): game_id} from the games table.

        Each game creates two entries — one for the home team, one for the away team —
        so either team abbreviation can be used as the lookup key.

        Used to derive game_id for player_stats rows that lack it (nflreadpy omits
        game_id for seasons 2019-2021 and 2024).
        """
        assert self._conn
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, season, week, home_team, away_team FROM games")
            rows = cur.fetchall()

        lookup: dict[tuple, str] = {}
        for row in rows:
            s, w = row["season"], row["week"]
            lookup[(s, w, row["home_team"])] = row["id"]
            lookup[(s, w, row["away_team"])] = row["id"]

        logger.info("Built game_id lookup: %d entries from %d games", len(lookup), len(rows))
        # Each game contributes 2 lookup entries (home + away). If the lookup is empty
        # after schedules should have been processed, game_id derivation will silently
        # fail for every pre-2022/2024 row — assert early to catch scheduling order bugs.
        assert len(lookup) > 0, (
            "game_id lookup is empty — schedules step must run before player_stats normalization"
        )
        return lookup

    def _upsert_game_logs(self, log_dicts: list[dict], summary: NormalizeSummary) -> int:
        if not log_dicts:
            return 0
        cols = [
            "player_id", "game_id", "season", "week", "season_type",
            "team", "opponent_team", "position",
            "completions", "attempts", "passing_yards", "passing_tds",
            "passing_interceptions", "passing_air_yards", "passing_yards_after_catch",
            "passing_first_downs", "passing_epa", "passing_cpoe",
            "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
            "receiving_fumbles", "sack_fumbles", "rushing_epa",
            "receptions", "targets", "receiving_yards", "receiving_tds",
            "receiving_air_yards", "receiving_yards_after_catch", "receiving_first_downs",
            "receiving_epa", "racr", "target_share", "air_yards_share", "wopr",
            "offense_snaps", "offense_pct", "fantasy_points", "fantasy_points_ppr",
        ]
        # Process in batches, catch FK violations row-by-row on failure
        upserted = 0
        batch: list[tuple] = []

        def _flush(b: list[tuple]) -> int:
            assert self._conn
            try:
                with self._conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO game_logs (
                            player_id, game_id, season, week, season_type,
                            team, opponent_team, position,
                            completions, attempts, passing_yards, passing_tds,
                            passing_interceptions, passing_air_yards,
                            passing_yards_after_catch, passing_first_downs,
                            passing_epa, passing_cpoe,
                            carries, rushing_yards, rushing_tds, rushing_fumbles,
                            receiving_fumbles, sack_fumbles, rushing_epa,
                            receptions, targets, receiving_yards,
                            receiving_tds, receiving_air_yards,
                            receiving_yards_after_catch, receiving_first_downs,
                            receiving_epa, racr, target_share, air_yards_share,
                            wopr, offense_snaps, offense_pct,
                            fantasy_points, fantasy_points_ppr
                        ) VALUES %s
                        ON CONFLICT (player_id, game_id) DO UPDATE SET
                            season             = EXCLUDED.season,
                            week               = EXCLUDED.week,
                            season_type        = EXCLUDED.season_type,
                            team               = EXCLUDED.team,
                            opponent_team      = EXCLUDED.opponent_team,
                            position           = EXCLUDED.position,
                            completions        = EXCLUDED.completions,
                            attempts           = EXCLUDED.attempts,
                            passing_yards      = EXCLUDED.passing_yards,
                            passing_tds        = EXCLUDED.passing_tds,
                            passing_interceptions = EXCLUDED.passing_interceptions,
                            passing_air_yards  = EXCLUDED.passing_air_yards,
                            passing_yards_after_catch = EXCLUDED.passing_yards_after_catch,
                            passing_first_downs = EXCLUDED.passing_first_downs,
                            passing_epa        = EXCLUDED.passing_epa,
                            passing_cpoe       = EXCLUDED.passing_cpoe,
                            carries            = EXCLUDED.carries,
                            rushing_yards      = EXCLUDED.rushing_yards,
                            rushing_tds        = EXCLUDED.rushing_tds,
                            rushing_fumbles    = EXCLUDED.rushing_fumbles,
                            receiving_fumbles  = EXCLUDED.receiving_fumbles,
                            sack_fumbles      = EXCLUDED.sack_fumbles,
                            rushing_epa        = EXCLUDED.rushing_epa,
                            receptions         = EXCLUDED.receptions,
                            targets            = EXCLUDED.targets,
                            receiving_yards    = EXCLUDED.receiving_yards,
                            receiving_tds      = EXCLUDED.receiving_tds,
                            receiving_air_yards = EXCLUDED.receiving_air_yards,
                            receiving_yards_after_catch = EXCLUDED.receiving_yards_after_catch,
                            receiving_first_downs = EXCLUDED.receiving_first_downs,
                            receiving_epa      = EXCLUDED.receiving_epa,
                            racr               = EXCLUDED.racr,
                            target_share       = EXCLUDED.target_share,
                            air_yards_share    = EXCLUDED.air_yards_share,
                            wopr               = EXCLUDED.wopr,
                            offense_snaps      = EXCLUDED.offense_snaps,
                            offense_pct        = EXCLUDED.offense_pct,
                            fantasy_points     = EXCLUDED.fantasy_points,
                            fantasy_points_ppr = EXCLUDED.fantasy_points_ppr
                        """,
                        b,
                    )
                self._conn.commit()
                return len(b)
            except psycopg2.IntegrityError as exc:
                self._conn.rollback()
                logger.warning("game_logs batch FK violation, skipping batch (%s)", exc)
                summary.errors += len(b)
                return 0

        for log in log_dicts:
            batch.append(tuple(log.get(c) for c in cols))
            if len(batch) >= BATCH_SIZE:
                upserted += _flush(batch)
                batch.clear()

        if batch:
            upserted += _flush(batch)

        return upserted

    # ── Source-level processors ───────────────────────────────────────────

    def _process_schedules(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        games: list[dict] = []
        all_teams: list[dict] = []
        staging_ids: list[int] = []

        for sr in staging_rows:
            raw = sr["raw_data"]
            try:
                row = ScheduleRow.model_validate(_coerce_row(raw))
            except ValidationError as exc:
                logger.warning("schedules re-validation failed: %s", exc)
                summary.errors += 1
                staging_ids.append(sr["id"])
                continue
            games.append(schedule_row_to_game(row))
            all_teams.extend(schedule_row_to_teams(row))
            staging_ids.append(sr["id"])

        summary.teams_upserted += self._upsert_teams(all_teams)
        summary.games_upserted += self._upsert_games(games)
        summary.staging_rows_processed += len(staging_ids)
        if staging_ids:
            self._mark_processed(staging_ids)

    def _process_rosters(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        players: list[dict] = []
        staging_ids: list[int] = []

        for sr in staging_rows:
            raw = sr["raw_data"]
            try:
                row = RosterRow.model_validate(_coerce_row(raw))
            except ValidationError as exc:
                logger.warning("rosters re-validation failed: %s", exc)
                summary.errors += 1
                staging_ids.append(sr["id"])
                continue
            player = roster_row_to_player(row)
            if player is None:
                summary.skipped += 1
            else:
                players.append(player)
            staging_ids.append(sr["id"])

        # Deduplicate by id — nflverse rosters can have duplicate gsis_id
        # (e.g. traded player on two teams). ON CONFLICT cannot affect a row
        # twice in one statement; last occurrence wins.
        by_id: dict[str, dict] = {}
        for p in players:
            by_id[p["id"]] = p
        players_deduped = list(by_id.values())

        summary.players_upserted += self._upsert_players(players_deduped)
        summary.staging_rows_processed += len(staging_ids)
        if staging_ids:
            self._mark_processed(staging_ids)

    def _process_player_stats(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        # Build game_id lookup once for the full batch.
        # Games table must already be populated (schedules step runs first).
        game_id_lookup = self._build_game_id_lookup()
        derived = 0
        no_match = 0

        logs: list[dict] = []
        staging_ids: list[int] = []

        for sr in staging_rows:
            raw = sr["raw_data"]

            # Derive game_id when absent (seasons 2019-2021, 2024 lack it in nflreadpy).
            if not raw.get("game_id"):
                key = (raw.get("season"), raw.get("week"), raw.get("team"))
                gid = game_id_lookup.get(key)
                if gid:
                    raw = {**raw, "game_id": gid}   # shallow copy; original staging unchanged
                    derived += 1
                else:
                    logger.debug(
                        "No schedule match for season=%s week=%s team=%s — skipping",
                        raw.get("season"), raw.get("week"), raw.get("team"),
                    )
                    no_match += 1
                    summary.skipped += 1
                    staging_ids.append(sr["id"])    # mark processed to avoid retrying
                    continue

            try:
                row = PlayerStatsRow.model_validate(_coerce_row(raw))
            except ValidationError as exc:
                logger.warning("player_stats re-validation failed: %s", exc)
                summary.errors += 1
                staging_ids.append(sr["id"])
                continue

            log = player_stats_row_to_game_log(row)
            if log is None:
                summary.skipped += 1
            else:
                logs.append(log)
            staging_ids.append(sr["id"])

        if derived:
            logger.info("Derived game_id from schedules for %d rows", derived)
        if no_match:
            logger.warning("Could not derive game_id for %d rows (no schedule match)", no_match)

        # Upsert player stubs BEFORE game_logs to satisfy the FK constraint.
        # load_rosters() only returns the current active roster; historical
        # players (traded/cut/IR) would otherwise cause FK violations.
        self._upsert_player_stubs_from_stats(staging_rows)

        summary.game_logs_upserted += self._upsert_game_logs(logs, summary)
        summary.staging_rows_processed += len(staging_ids)
        if staging_ids:
            self._mark_processed(staging_ids)

    def _build_name_to_gsis_lookup(self) -> dict[str, str]:
        """Return {full_name: gsis_id} for all players with a full_name."""
        assert self._conn
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, full_name FROM players WHERE full_name IS NOT NULL"
            )
            rows = cur.fetchall()
        lookup = {row["full_name"]: row["id"] for row in rows if row["full_name"]}
        logger.info("Built name → gsis_id lookup: %d entries", len(lookup))
        return lookup

    def _process_snap_counts(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """
        Update game_logs.offense_snaps and game_logs.offense_pct from snap_counts.

        Join chain:
          snap_counts.player (name string)
            → players.full_name (case-insensitive exact or rapidfuzz WRatio ≥90)
            → players.id (gsis_id)
            → UPDATE game_logs SET offense_snaps, offense_pct
               WHERE player_id = gsis_id AND game_id = snap_count.game_id

        All updates are batched and committed in a single transaction per
        flush (at most one transaction per BATCH_SIZE rows) rather than one
        commit per row.
        """
        name_lookup = self._build_name_to_gsis_lookup()
        # Build a case-insensitive version of the lookup for the exact-match step.
        # Snap count sources (PFR, nflreadpy, ESPN) may use different casing.
        name_lookup_lower = {k.lower(): v for k, v in name_lookup.items()}
        name_keys = list(name_lookup.keys())

        matched = 0
        unmatched = 0
        fuzzy_matched = 0
        staging_ids: list[int] = []

        # Accumulate (snaps, pct, gsis_id, game_id) tuples for batch UPDATE.
        pending_updates: list[tuple] = []

        def _flush_updates(updates: list[tuple]) -> int:
            """Execute accumulated snap updates in one transaction."""
            if not updates:
                return 0
            flushed = 0
            try:
                with self._conn.cursor() as cur:
                    for offense_snaps, offense_pct, gsis_id, game_id in updates:
                        cur.execute(
                            """
                            UPDATE game_logs
                            SET offense_snaps = %s, offense_pct = %s
                            WHERE player_id = %s AND game_id = %s
                            """,
                            (offense_snaps, offense_pct, gsis_id, game_id),
                        )
                        if cur.rowcount > 0:
                            flushed += 1
                        else:
                            logger.debug(
                                "snap_counts: no game_log row for player_id=%s game_id=%s — skipped",
                                gsis_id, game_id,
                            )
                self._conn.commit()
            except Exception as exc:
                self._conn.rollback()
                logger.warning("snap_counts batch UPDATE failed: %s", exc)
                summary.errors += len(updates)
                flushed = 0
            return flushed

        for sr in staging_rows:
            raw = sr["raw_data"]
            staging_ids.append(sr["id"])

            try:
                row = SnapCountRow.model_validate(_coerce_row(raw))
            except ValidationError as exc:
                logger.debug("snap_counts re-validation failed: %s", exc)
                summary.errors += 1
                continue

            if not row.player:
                logger.debug("snap_count row has no player name — skipping")
                summary.skipped += 1
                continue

            # 1. Try case-insensitive exact name match first.
            gsis_id = name_lookup.get(row.player) or name_lookup_lower.get(row.player.lower())

            # 2. Try fuzzy match fallback (handles nickname variants, punctuation).
            if gsis_id is None:
                best_match = process.extractOne(
                    row.player, name_keys, scorer=fuzz.WRatio, score_cutoff=90
                )
                if best_match:
                    best_match_str = best_match[0]
                    gsis_id = name_lookup[best_match_str]
                    logger.debug(
                        "Fuzzy match snap_counts: '%s' -> '%s' (score: %.1f)",
                        row.player, best_match_str, best_match[1]
                    )
                    fuzzy_matched += 1

            if gsis_id is None:
                logger.debug(
                    "No gsis_id found for player='%s' — skipping",
                    row.player,
                )
                unmatched += 1
                summary.skipped += 1
                continue

            pending_updates.append((row.offense_snaps, row.offense_pct, gsis_id, row.game_id))

            # Flush in batches to keep transaction size bounded.
            if len(pending_updates) >= BATCH_SIZE:
                matched += _flush_updates(pending_updates)
                pending_updates.clear()

        # Flush any remaining updates.
        matched += _flush_updates(pending_updates)

        if matched:
            logger.info(
                "snap_counts: updated %d game_log rows with snap data "
                "(%d via exact match, %d via fuzzy fallback)",
                matched, matched - fuzzy_matched, fuzzy_matched
            )
        if unmatched:
            logger.debug(
                "snap_counts: %d rows skipped (no matching player name in players table)",
                unmatched,
            )

        summary.staging_rows_processed += len(staging_ids)
        if staging_ids:
            self._mark_processed(staging_ids)

    def _process_depth_charts(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """Upsert depth_charts from staging. depth_rank = 1 (WR1), 2 (WR2), etc."""
        if not staging_rows:
            return
        rows: list[tuple] = []
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)
            gsis_id = raw.get("gsis_id") or raw.get("player_id")
            if not gsis_id:
                continue
            season = raw.get("season")
            week = raw.get("week")
            team = raw.get("club_code") or raw.get("team", "")
            if season is None or week is None or not team:
                continue
            depth_team = raw.get("depth_team") or raw.get("pos_rank")
            try:
                rank = float(str(depth_team or "1"))
            except (TypeError, ValueError):
                rank = 1.0
            position = raw.get("position") or raw.get("pos_abb")
            rows.append((gsis_id, season, week, team, position, rank))

        # Deduplicate: keep best (lowest) depth_rank per (player_id, season, week)
        seen: dict[tuple, tuple] = {}
        for r in rows:
            key = (r[0], r[1], r[2])
            if key not in seen or r[5] < seen[key][5]:
                seen[key] = r
        rows = list(seen.values())

        staging_ids = [sr["id"] for sr in staging_rows]
        if staging_ids:
            self._mark_processed(staging_ids)
        summary.staging_rows_processed += len(staging_rows)

        if not rows:
            return

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO depth_charts (player_id, season, week, team, position, depth_rank)
                VALUES %s
                ON CONFLICT (player_id, season, week) DO UPDATE SET
                    team = EXCLUDED.team,
                    position = EXCLUDED.position,
                    depth_rank = EXCLUDED.depth_rank
                """,
                rows,
            )
        self._conn.commit()
        logger.info("Upserted %d depth_chart rows", len(rows))

    def _process_nextgen_stats(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """Upsert nextgen_stats from staging."""
        if not staging_rows:
            return
        rows: list[tuple] = []
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)
            pid = raw.get("player_id") or raw.get("gsis_id")
            if not pid:
                continue
            season = raw.get("season")
            week = raw.get("week")
            if season is None or week is None:
                continue
            sep = raw.get("avg_separation")
            cushion = raw.get("avg_cushion")
            speed = raw.get("max_speed")
            cpoe = raw.get("avg_completion_above_expectation")
            rows.append((pid, season, week, sep, cushion, speed, cpoe))

        staging_ids = [sr["id"] for sr in staging_rows]
        if staging_ids:
            self._mark_processed(staging_ids)
        summary.staging_rows_processed += len(staging_rows)

        if not rows:
            return

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO nextgen_stats (
                    player_id, season, week,
                    avg_separation, avg_cushion, max_speed,
                    avg_completion_above_expectation
                ) VALUES %s
                ON CONFLICT (player_id, season, week) DO UPDATE SET
                    avg_separation = EXCLUDED.avg_separation,
                    avg_cushion = EXCLUDED.avg_cushion,
                    max_speed = EXCLUDED.max_speed,
                    avg_completion_above_expectation = EXCLUDED.avg_completion_above_expectation
                """,
                rows,
            )
        self._conn.commit()
        logger.info("Upserted %d nextgen_stats rows", len(rows))

    def _process_team_stats(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """Upsert team_game_stats from staging."""
        if not staging_rows:
            return
        rows: list[tuple] = []
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)
            team = raw.get("team")
            season = raw.get("season")
            week = raw.get("week")
            if not team or season is None or week is None:
                continue
            attempts = raw.get("attempts")
            carries = raw.get("carries")
            total_plays = raw.get("total_plays")
            total_yards = raw.get("total_yards")
            rows.append((team, season, week, attempts, carries, total_plays, total_yards))

        staging_ids = [sr["id"] for sr in staging_rows]
        if staging_ids:
            self._mark_processed(staging_ids)
        summary.staging_rows_processed += len(staging_rows)

        if not rows:
            return

        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO team_game_stats (
                    team, season, week, pass_attempts, rush_attempts,
                    total_plays, total_yards
                ) VALUES %s
                ON CONFLICT (team, season, week) DO UPDATE SET
                    pass_attempts = EXCLUDED.pass_attempts,
                    rush_attempts = EXCLUDED.rush_attempts,
                    total_plays = EXCLUDED.total_plays,
                    total_yards = EXCLUDED.total_yards
                """,
                rows,
            )
        self._conn.commit()
        logger.info("Upserted %d team_game_stats rows", len(rows))

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, seasons: Optional[list[int]] = None) -> NormalizeSummary:
        """
        Process all unprocessed staging rows in FK-safe order:
          schedules → rosters → player_stats → snap_counts
        """
        summary = NormalizeSummary()

        # Order matters: game_logs must exist before snap_counts can update them.
        for source_type in [
            SOURCE_SCHEDULES,
            SOURCE_ROSTERS,
            SOURCE_PLAYER_STATS,
            SOURCE_SNAP_COUNTS,
            SOURCE_DEPTH_CHARTS,
            SOURCE_NEXTGEN_STATS,
            SOURCE_TEAM_STATS,
            SOURCE_FTN_CHARTING,
            SOURCE_PARTICIPATION,
            SOURCE_COMBINE,
        ]:
            staging_rows = self._fetch_unprocessed(source_type, seasons)
            if not staging_rows:
                logger.info("No unprocessed rows for source_type=%s", source_type)
                continue

            logger.info(
                "Processing %d staging rows for source_type=%s",
                len(staging_rows), source_type,
            )
            if source_type == SOURCE_SCHEDULES:
                self._process_schedules(staging_rows, summary)
            elif source_type == SOURCE_ROSTERS:
                self._process_rosters(staging_rows, summary)
            elif source_type == SOURCE_PLAYER_STATS:
                self._process_player_stats(staging_rows, summary)
            elif source_type == SOURCE_SNAP_COUNTS:
                self._process_snap_counts(staging_rows, summary)
            elif source_type == SOURCE_DEPTH_CHARTS:
                self._process_depth_charts(staging_rows, summary)
            elif source_type == SOURCE_NEXTGEN_STATS:
                self._process_nextgen_stats(staging_rows, summary)
            elif source_type == SOURCE_TEAM_STATS:
                self._process_team_stats(staging_rows, summary)
            elif source_type == SOURCE_FTN_CHARTING:
                self._process_ftn_charting(staging_rows, summary)
            elif source_type == SOURCE_PARTICIPATION:
                self._process_participation(staging_rows, summary)
            elif source_type == SOURCE_COMBINE:
                self._process_combine(staging_rows, summary)

        return summary

    def _process_ftn_charting(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """Upsert FTN charting play-level data (is_drop, is_contested_ball). 2022+."""
        rows: list[tuple] = []
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)
            gid = raw.get("game_id") or raw.get("nflverse_game_id")
            pid_raw = raw.get("play_id") or raw.get("nflverse_play_id")
            if not gid or pid_raw is None:
                continue
            try:
                play_id = int(pid_raw)
            except (TypeError, ValueError):
                continue
            season = raw.get("season")
            week = raw.get("week")
            is_drop = bool(raw.get("is_drop"))
            is_contested = bool(raw.get("is_contested_ball"))
            is_catchable = raw.get("is_catchable_ball") if raw.get("is_catchable_ball") is not None else None
            is_created = raw.get("is_created_reception") if raw.get("is_created_reception") is not None else None
            rows.append((gid, play_id, season, week, is_drop, is_contested, is_catchable, is_created))
        if not rows:
            self._mark_processed([sr["id"] for sr in staging_rows])
            return
        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO ftn_play (game_id, play_id, season, week, is_drop, is_contested_ball, is_catchable_ball, is_created_reception)
                VALUES %s
                ON CONFLICT (game_id, play_id) DO UPDATE SET
                    season = EXCLUDED.season, week = EXCLUDED.week,
                    is_drop = EXCLUDED.is_drop, is_contested_ball = EXCLUDED.is_contested_ball,
                    is_catchable_ball = EXCLUDED.is_catchable_ball, is_created_reception = EXCLUDED.is_created_reception
                """,
                rows,
            )
        self._conn.commit()
        self._mark_processed([sr["id"] for sr in staging_rows])
        summary.staging_rows_processed += len(staging_rows)
        logger.info("Upserted %d ftn_play rows", len(rows))

    def _process_participation(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """
        Aggregate participation to player-game routes_run.
        offense_players is semicolon-separated gsis_ids (2023+). Each player on the
        play gets +1 routes_run. Pre-2023 often has empty offense_players → no rows.
        """
        from collections import defaultdict
        agg: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"season": None, "week": None, "routes": 0}
        )
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)
            gid = raw.get("game_id") or raw.get("nflverse_game_id")
            offense = raw.get("offense_players") or ""
            if not gid or not offense:
                continue
            # Semicolon-separated gsis_ids (nflverse 2023+)
            pids = [x.strip() for x in offense.replace(",", ";").split(";") if x.strip()]
            season = raw.get("season")
            week = raw.get("week")
            for pid in pids:
                key = (pid, gid)
                agg[key]["season"] = season
                agg[key]["week"] = week
                agg[key]["routes"] += 1
        rows = [(pid, gid, a["season"], a["week"], a["routes"]) for (pid, gid), a in agg.items()]
        if not rows:
            self._mark_processed([sr["id"] for sr in staging_rows])
            return
        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO participation_player_game (player_id, game_id, season, week, routes_run)
                VALUES %s
                ON CONFLICT (player_id, game_id) DO UPDATE SET
                    season = EXCLUDED.season, week = EXCLUDED.week,
                    routes_run = EXCLUDED.routes_run
                """,
                rows,
            )
        self._conn.commit()
        self._mark_processed([sr["id"] for sr in staging_rows])
        summary.staging_rows_processed += len(staging_rows)
        logger.info("Upserted %d participation_player_game rows", len(rows))

    def _process_combine(
        self, staging_rows: list[dict], summary: NormalizeSummary
    ) -> None:
        """Upsert combine (40yd, bench, etc.) by player and season (draft year)."""
        if not staging_rows:
            return
        # nflverse uses pfr_id; resolve via players.pfr_id → id (gsis_id)
        pfr_to_gsis: dict[str, str] = {}
        assert self._conn
        with self._conn.cursor() as cur:
            cur.execute("SELECT pfr_id, id FROM players WHERE pfr_id IS NOT NULL")
            for row in cur.fetchall():
                pfr_to_gsis[row[0]] = row[1]

        rows: list[tuple] = []
        for sr in staging_rows:
            raw = sr.get("raw_data", sr)
            pid = raw.get("player_id") or raw.get("gsis_id") or pfr_to_gsis.get(raw.get("pfr_id") or "")
            season = raw.get("season")
            if not pid or season is None:
                continue
            forty = raw.get("forty") or raw.get("x40_yard")
            bench = raw.get("bench_press") or raw.get("bench")
            vert = raw.get("vertical_jump") or raw.get("vertical")
            broad = raw.get("broad_jump") or raw.get("broad")
            rows.append((pid, season, forty, bench, vert, broad))
        if not rows:
            self._mark_processed([sr["id"] for sr in staging_rows])
            summary.staging_rows_processed += len(staging_rows)
            return
        assert self._conn
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO combine (player_id, season, forty, bench_press, vertical_jump, broad_jump)
                VALUES %s
                ON CONFLICT (player_id, season) DO UPDATE SET
                    forty = EXCLUDED.forty, bench_press = EXCLUDED.bench_press,
                    vertical_jump = EXCLUDED.vertical_jump, broad_jump = EXCLUDED.broad_jump
                """,
                rows,
            )
        self._mark_processed([sr["id"] for sr in staging_rows])
        summary.staging_rows_processed += len(staging_rows)
        logger.info("Upserted %d combine rows", len(rows))


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize staging_nflreadpy rows into production tables."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=None, metavar="YEAR",
        help="Only process rows for these seasons. Default: all unprocessed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Connect and count rows but do not upsert.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set.")
        raise SystemExit(1)

    with Normalizer(db_url) as norm:
        if args.dry_run:
            for source in [SOURCE_SCHEDULES, SOURCE_ROSTERS, SOURCE_PLAYER_STATS]:
                rows = norm._fetch_unprocessed(source, args.seasons)
                print(f"  {source}: {len(rows)} unprocessed rows in staging")
            return
        summary = norm.run(seasons=args.seasons)
        summary.log()


if __name__ == "__main__":
    main()
