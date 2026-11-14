#!/usr/bin/env python3
"""Friday availability-only refresh: update ``availability_status`` on the next forward week.

Fetches a fresh availability snapshot and changes only the status column of that week's forward
rows (see ``pipeline/availability_refresh.py``). Exit codes: 0 done, 1 refused or the source was
unavailable (nothing written), 2 usage. Never touches a prediction.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import season_constants
from pipeline.availability_refresh import (
    AvailabilityRefreshRefused,
    refresh_availability,
    resolve_target_week,
)
from pipeline.availability_store import AvailabilityFetchError
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=season_constants.CURRENT_SEASON)
    parser.add_argument("--week", type=int, default=None, help="Default: the week of the next kickoff")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    conn = psycopg2.connect(normalize_dsn(args.database_url))
    try:
        week = args.week or resolve_target_week(conn, args.season, datetime.now(timezone.utc))
        report = refresh_availability(conn, args.season, week)
    except (AvailabilityRefreshRefused, AvailabilityFetchError) as exc:
        print(f"Refused, nothing written: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
