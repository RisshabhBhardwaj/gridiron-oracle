"""Pregame Sleeper weekly consensus capture.

Snapshots must be taken before kickoff. A post-kickoff row is as contaminated
as any other postgame feature. Keep load_ff_rankings out of promoted artifacts
until DynastyProcess licensing (rebuild Q4) is answered.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import psycopg2

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

CREATE_CONSENSUS = """
CREATE TABLE IF NOT EXISTS consensus_projections_weekly (
    source TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    player_id TEXT NOT NULL,
    projection FLOAT,
    scoring TEXT NOT NULL DEFAULT 'ppr',
    captured_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (source, season, week, player_id, captured_at)
)
"""


def ensure_table(database_url: str) -> None:
    dsn = normalize_dsn(database_url)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_CONSENSUS)
        conn.commit()


def insert_rows(
    database_url: str,
    *,
    source: str,
    season: int,
    week: int,
    scoring: str,
    rows: list[dict],
    captured_at: datetime | None = None,
) -> int:
    if source == "fantasypros_ecr":
        raise ValueError(
            "FantasyPros ECR via DynastyProcess stays out of promoted capture "
            "until the redistribution license is confirmed"
        )
    stamp = captured_at or datetime.now(timezone.utc)
    dsn = normalize_dsn(database_url)
    n = 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_CONSENSUS)
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO consensus_projections_weekly
                        (source, season, week, player_id, projection, scoring, captured_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        source,
                        season,
                        week,
                        str(row["player_id"]),
                        None if row.get("projection") is None else float(row["projection"]),
                        scoring,
                        stamp,
                    ),
                )
                n += 1
        conn.commit()
    logger.info("Inserted %d consensus rows source=%s %s w%s", n, source, season, week)
    return n


SLEEPER_PROJECTIONS = "https://api.sleeper.app/v1/projections/nfl/{season}/{week}?season_type=regular"


def fetch_sleeper_projections(season: int, week: int, timeout_s: float = 20.0) -> dict[str, float]:
    """Map sleeper_id → pts_ppr. Empty dict if the API is unreachable."""
    from urllib.request import Request, urlopen

    url = SLEEPER_PROJECTIONS.format(season=season, week=week)
    request = Request(url, headers={"User-Agent": "gridiron-oracle/sleeper-consensus"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    out: dict[str, float] = {}
    if not isinstance(payload, dict):
        return out
    for sleeper_id, body in payload.items():
        if not isinstance(body, dict):
            continue
        points = body.get("pts_ppr")
        if points is None:
            continue
        try:
            out[str(sleeper_id)] = float(points)
        except (TypeError, ValueError):
            continue
    return out


def capture_pregame(database_url: str = DEFAULT_HOST_DATABASE_URL) -> int:
    """Snapshot Sleeper weekly PPR before the week's first kickoff."""
    dsn = normalize_dsn(database_url)
    now = datetime.now(timezone.utc)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT season, week, MIN(kickoff_at) AS first_kick
                FROM games
                WHERE kickoff_at IS NOT NULL AND kickoff_at > %s
                GROUP BY season, week
                ORDER BY season, week
                LIMIT 1
                """,
                (now,),
            )
            upcoming = cur.fetchone()
            if not upcoming:
                logger.info("No upcoming kickoff; skipping Sleeper consensus capture")
                return 0
            season, week, first_kick = upcoming
            if first_kick is not None and first_kick <= now:
                logger.info("Kickoff already passed for %s w%s; refusing postgame snapshot", season, week)
                return 0
            cur.execute(
                "SELECT sleeper_id, gsis_id FROM fantasy_player_ids WHERE sleeper_id IS NOT NULL"
            )
            id_map = {str(sleeper_id): str(gsis_id) for sleeper_id, gsis_id in cur.fetchall()}
    if not id_map:
        logger.warning("fantasy_player_ids is empty; run ff_playerids ingest first")
        return 0
    projections = fetch_sleeper_projections(int(season), int(week))
    rows = [
        {"player_id": gsis, "projection": projections[sleeper_id]}
        for sleeper_id, gsis in id_map.items()
        if sleeper_id in projections
    ]
    if not rows:
        return 0
    return insert_rows(
        database_url,
        source="sleeper",
        season=int(season),
        week=int(week),
        scoring="ppr",
        rows=rows,
        captured_at=now,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--ensure-table", action="store_true")
    parser.add_argument("--capture", action="store_true", help="Pregame Sleeper snapshot")
    args = parser.parse_args()
    if args.ensure_table:
        ensure_table(args.database_url)
    if args.capture:
        capture_pregame(args.database_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
