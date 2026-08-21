#!/usr/bin/env python3
"""Build one auditable, causally legal forward feature snapshot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.feature_engineer import FeatureEngineer
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--as-of", type=datetime.fromisoformat)
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL))
    args = parser.parse_args()
    as_of = args.as_of or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        parser.error("--as-of must include timezone information")
    with FeatureEngineer(args.database_url) as engineer:
        count = engineer.build_forward_week(season=args.season, week=args.week, as_of=as_of)
    with psycopg2.connect(normalize_dsn(args.database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO forecast_runs
                     (pipeline_run_id, season, week, as_of, feature_rows, source_summary)
                     VALUES (%s, %s, %s, %s, %s, %s)
                     ON CONFLICT (pipeline_run_id) DO UPDATE SET
                       season=EXCLUDED.season, week=EXCLUDED.week, as_of=EXCLUDED.as_of,
                       feature_rows=EXCLUDED.feature_rows, source_summary=EXCLUDED.source_summary""",
                (args.pipeline_run_id, args.season, args.week, as_of, count,
                 json.dumps({"kind": "forward_feature_snapshot", "as_of": as_of.isoformat()})),
            )
        conn.commit()
    print(json.dumps({"pipeline_run_id": args.pipeline_run_id, "feature_rows": count, "as_of": as_of.isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
