#!/usr/bin/env python3
"""Put back the week that ``--replace-week`` archived, when the refresh died before it could.

The weekly refresh restores the archive itself on any failure it can catch. After a kill (OOM, host
reboot, timeout) between the producer and the promotion, the week is unserved and ``projections_replaced``
still holds the old rows. Run this to restore them. It does nothing when there is no archive.

    python scripts/restore_replaced_week.py --season 2026 --week 11
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import closing
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.schema import normalize_dsn  # noqa: E402
from pipeline.week_archive import restore_week  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error("set --database-url or DATABASE_URL")
    with closing(psycopg2.connect(normalize_dsn(args.database_url))) as conn:
        restored = restore_week(conn, args.season, args.week)
    print(f"restored {restored} rows for season={args.season} week={args.week}" if restored
          else f"nothing archived for season={args.season} week={args.week}; nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
