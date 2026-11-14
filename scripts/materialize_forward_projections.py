#!/usr/bin/env python3
"""Forward projections for one pre-built feature snapshot: a thin CLI over ml.forward_serving.

All logic (guards G1-G5, inference with the pinned base models, the availability cohort and
the single write) lives in ``ml/forward_serving.py``; see decision 0012. This file opens a
connection, calls ``run_forward`` and prints its report. Exit codes: 0 done, 1 a guard
refused (nothing written), 2 usage.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.forward_serving import ForwardGuardError, LibraryVersionMismatch, run_forward
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--feature-run-id", default=None, help="Snapshot id; default is the newest for the week")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL))
    parser.add_argument("--dry-run", action="store_true", help="Run every guard and the models; write nothing")
    parser.add_argument(
        "--allow-no-availability", action="store_true",
        help="Store every row with a NULL availability status when no snapshot exists (explicit override)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    conn = psycopg2.connect(normalize_dsn(args.database_url))
    try:
        report = run_forward(
            conn, args.season, args.week,
            feature_run_id=args.feature_run_id,
            dry_run=args.dry_run,
            allow_no_availability=args.allow_no_availability,
        )
    except (ForwardGuardError, LibraryVersionMismatch) as exc:
        print(f"Refused, nothing written: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
