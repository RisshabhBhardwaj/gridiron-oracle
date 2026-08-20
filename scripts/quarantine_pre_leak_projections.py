#!/usr/bin/env python3
"""Remove the pre-leak materialization so it cannot be served."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

STALE_RUN_ID = "stack_materialize_20260809T171811Z"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import psycopg2

    dsn = normalize_dsn(args.database_url)
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM projections WHERE pipeline_run_id = %s",
                (STALE_RUN_ID,),
            )
            n = int(cur.fetchone()[0])
            if args.dry_run:
                print(f"would delete {n} rows for {STALE_RUN_ID}")
                return 0
            cur.execute("DELETE FROM projections WHERE pipeline_run_id = %s", (STALE_RUN_ID,))
        conn.commit()
    print(f"deleted {n} rows for {STALE_RUN_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
