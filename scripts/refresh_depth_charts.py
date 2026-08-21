#!/usr/bin/env python3
"""Refresh, normalize, and validate the current depth-chart snapshot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.normalize import Normalizer
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn
from scraper.adapters.nflreadpy_adapter import NFLReadPyAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL))
    parser.add_argument("--required-player", action="append", default=[])
    args = parser.parse_args()
    with NFLReadPyAdapter(args.database_url) as adapter:
        staged, failed = adapter.fetch_depth_charts([args.season])
    with Normalizer(args.database_url) as normalizer:
        summary = normalizer.run(seasons=[args.season])
    with psycopg2.connect(normalize_dsn(args.database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(DISTINCT team), COUNT(DISTINCT player_id), MAX(ingest_at) "
                "FROM depth_charts WHERE season = %s",
                (args.season,),
            )
            teams, players, latest = cur.fetchone()
            if args.required_player:
                cur.execute(
                    "SELECT player_id FROM depth_charts WHERE season = %s AND player_id = ANY(%s)",
                    (args.season, args.required_player),
                )
                found = {row[0] for row in cur.fetchall()}
            else:
                found = set()
    report = {
        "season": args.season, "staged": staged, "failed": failed,
        "normalized": summary.staging_rows_processed, "teams": teams,
        "players": players, "latest_ingest_at": latest.isoformat() if latest else None,
        "missing_required_players": sorted(set(args.required_player) - found),
    }
    print(json.dumps(report, indent=2))
    if failed or teams != 32 or not latest or report["missing_required_players"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
