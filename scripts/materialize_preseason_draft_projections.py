#!/usr/bin/env python3
"""Materialize causal week-0 draft projections; never reads target-season OOF."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.draft_projection import build_preseason_projections
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)


def materialize(season: int, as_of: date, database_url: str, *, dry_run: bool = False) -> int:
    with psycopg2.connect(normalize_dsn(database_url)) as conn:
        players = pd.read_sql("SELECT id, full_name, position, team, entry_year FROM players", conn)
        logs = pd.read_sql(
            "SELECT player_id, season, week, season_type, fantasy_points_ppr FROM game_logs WHERE season < %s",
            conn, params=(season,),
        )
        projections = build_preseason_projections(players, logs, season=season)
        if dry_run:
            return len(projections)
        with conn.cursor() as cur:
            rows = [
                (season, as_of, "preseason_historical_per_game", r.player_id, r.player_name,
                 r.position, r.team, float(r.per_game_mean), float(r.games_played_prior), float(r.projection))
                for r in projections.itertuples(index=False)
            ]
            psycopg2.extras.execute_values(cur, """
                INSERT INTO draft_preseason_projections
                    (season, as_of, source, player_id, player_name, position, team,
                     per_game_mean, games_played_prior, projection)
                VALUES %s
                ON CONFLICT (season, as_of, source, player_id) DO UPDATE SET
                    player_name = EXCLUDED.player_name, position = EXCLUDED.position,
                    team = EXCLUDED.team, per_game_mean = EXCLUDED.per_game_mean,
                    games_played_prior = EXCLUDED.games_played_prior, projection = EXCLUDED.projection,
                    created_at = NOW()
            """, rows)
        conn.commit()
    return len(projections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = materialize(args.season, args.as_of, args.database_url, dry_run=args.dry_run)
    logger.info("%s %d preseason draft projections", "Would materialize" if args.dry_run else "Materialized", count)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
