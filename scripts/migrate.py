#!/usr/bin/env python3
"""Apply or stamp Alembic migrations."""

from __future__ import annotations

import argparse
import sys

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import database_url_from_env, stamp_head, upgrade_head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("upgrade", "stamp"),
        help="upgrade = apply migrations; stamp = mark existing DB as current",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=f"Database URL (default: DATABASE_URL or {DEFAULT_HOST_DATABASE_URL})",
    )
    args = parser.parse_args()
    url = args.db_url or database_url_from_env()
    if args.action == "upgrade":
        upgrade_head(url)
    else:
        stamp_head(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
