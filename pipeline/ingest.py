"""
pipeline/ingest.py

CLI entry point for `make ingest` — triggers the full ETL pipeline.

Usage:
    python -m pipeline.ingest [--seasons YEAR...] [--dry-run]

This is the main entry point invoked by:
    make ingest  →  python -m pipeline.ingest

Internally delegates to pipeline/orchestrator.py which handles
all ETL stages in the correct FK processing order:
    1. Schedules
    2. Rosters
    3. Player stats (nflreadpy)
    4. Snap counts
    5. Feature engineering
    6. Projection generation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """
    Parse CLI arguments and run the full ETL pipeline.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline.ingest",
        description="Gridiron Oracle — full ETL ingestion pipeline",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Seasons to ingest (default: all available from 2019)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate pipeline steps without writing to database",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Override DATABASE_URL (default: read from env)",
    )
    args = parser.parse_args(argv)

    db_url = args.db_url or os.environ.get(
        "DATABASE_URL",
        "postgresql://oracle:oracle@localhost:5432/oracle",
    )

    logger.info(
        "Starting ingest: seasons=%s dry_run=%s db=%s",
        args.seasons or "all",
        args.dry_run,
        db_url.split("@")[-1] if "@" in db_url else "configured",
    )

    try:
        from pipeline.orchestrator import Orchestrator

        orch = Orchestrator(db_url=db_url)
        summary = orch.run(seasons=args.seasons, dry_run=args.dry_run)
        failed_steps = [
            step
            for season_result in summary.season_results
            for step in season_result.steps
            if not step.ok
        ]
        if failed_steps:
            logger.error("Ingest completed with %d failed step(s).", len(failed_steps))
            return 1
        logger.info("Ingest complete.")
        return 0

    except Exception as exc:
        logger.error("Ingest failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
