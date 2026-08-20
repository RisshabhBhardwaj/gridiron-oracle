"""Forward-only dated injury / practice capture.

nflverse injuries died after 2024. Persist ESPN (and later nfl.com) reports
with captured_at so SP2 can select published_at < kickoff_at. 2025 history
is gone; do not manufacture it.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import psycopg2

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn
from scraper.adapters.espn_adapter import EspnAdapter

logger = logging.getLogger(__name__)

CREATE_INJURY_REPORTS = """
CREATE TABLE IF NOT EXISTS injury_reports (
    player_id TEXT,
    player_name TEXT NOT NULL,
    team TEXT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    practice_status TEXT,
    injury_type TEXT,
    source TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    raw_payload JSONB,
    PRIMARY KEY (season, week, source, player_name, captured_at)
)
"""


def capture_week(database_url: str, season: int, week: int, source: str = "espn") -> int:
    if source != "espn":
        raise ValueError(f"Unsupported injury source {source!r}")
    adapter = EspnAdapter(db_url=database_url)
    frame = adapter.fetch_injury_report(week=week, season=season)
    captured_at = datetime.now(timezone.utc)
    dsn = normalize_dsn(database_url)
    n = 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_INJURY_REPORTS)
            for raw in frame.to_dict(orient="records"):
                cur.execute(
                    """
                    INSERT INTO injury_reports (
                        player_id, player_name, team, season, week,
                        practice_status, injury_type, source, captured_at, raw_payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        raw.get("player_id"),
                        str(raw.get("player_name") or ""),
                        raw.get("espn_team") or raw.get("team"),
                        season,
                        week,
                        raw.get("practice_status"),
                        raw.get("injury_type"),
                        source,
                        captured_at,
                        json.dumps({k: (None if v != v else v) for k, v in raw.items()}, default=str),
                    ),
                )
                n += cur.rowcount
        conn.commit()
    logger.info("Captured %d injury_reports rows for %s week %s", n, season, week)
    return n


def infer_capture_week(database_url: str) -> tuple[int, int]:
    """Return (season, week) for the next/current game window."""
    dsn = normalize_dsn(database_url)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT season, week
                    FROM games
                    WHERE kickoff_at IS NOT NULL
                      AND kickoff_at > NOW() - INTERVAL '3 days'
                    ORDER BY kickoff_at ASC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if row:
                    return int(row[0]), int(row[1])
            except Exception:
                conn.rollback()
            cur.execute(
                """
                SELECT season, week
                FROM games
                WHERE gameday IS NOT NULL
                ORDER BY gameday DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("Cannot infer NFL week: games table is empty")
            return int(row[0]), int(row[1])


def capture_current_week(database_url: str, source: str = "espn") -> int:
    season, week = infer_capture_week(database_url)
    return capture_week(database_url, season, week, source=source)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    capture_week(args.database_url, args.season, args.week)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
