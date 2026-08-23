#!/usr/bin/env python3
"""
Weekly cloud refresh cron sequencer for Gridiron Oracle.

Runs weekly (e.g. Tuesday morning post-MNF) in Railway / Cloud to incrementally
ingest the latest week of data, re-run feature engineering & ID bridge,
materialize models into pending pipeline_run_ids, verify acceptance,
and atomically promote to the approved pipeline_run_id ('weekly_auto_{season}').

Fail-loud contract: Every step aborts on error, logging details and writing a
dead_letter row to Postgres so failures are never silent.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.orchestrator import Orchestrator, _write_dead_letter
from pipeline.schema import normalize_dsn
from pipeline.staging_retention import purge_stale_staging
from scraper.adapters.ff_playerids import upsert_playerids
from scripts.prune_stale_pipeline_runs import (
    SEASON_START_WEEK_TABLES,
    check_guard,
    delete_stale_rows,
    load_keep_run_ids,
    vacuum_tables,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("weekly_refresh")

WEEKLY_SOURCES = [
    "player_stats",
    "rosters",
    "schedules",
    "snap_counts",
    "depth_charts",
    "nextgen_stats",
    "team_stats",
]


def resolve_season_and_week(cur, default_season: int = 2026) -> tuple[int, int, int]:
    """
    Resolve (season, completed_week, start_week) based on kickoffs in games table.
    """
    cur.execute(
        """
        SELECT COALESCE(MAX(week), 0)
        FROM games
        WHERE season = %s AND kickoff_at < NOW()
        """,
        (default_season,),
    )
    row = cur.fetchone()
    completed_week = int(row[0]) if row and row[0] is not None else 0
    start_week = completed_week + 1
    return default_season, completed_week, start_week


def promote_run_id(
    cur,
    season: int,
    start_week: int,
    pending_run_id: str,
    approved_run_id: str,
) -> None:
    """
    Atomically swap pending_run_id -> approved_run_id for (season, start_week).
    """
    logger.info(
        "Atomically promoting %s -> %s for season=%d, start_week=%d",
        pending_run_id,
        approved_run_id,
        season,
        start_week,
    )
    for table in SEASON_START_WEEK_TABLES:
        # 1. Delete any old rows for this (season, start_week) that are not the pending run
        cur.execute(
            f"""
            DELETE FROM {table}
            WHERE season = %s AND start_week = %s
              AND (pipeline_run_id IS NULL OR pipeline_run_id <> %s)
            """,
            (season, start_week, pending_run_id),
        )
        # 2. Update pending rows to approved run id
        cur.execute(
            f"""
            UPDATE {table}
            SET pipeline_run_id = %s
            WHERE season = %s AND start_week = %s
              AND pipeline_run_id = %s
            """,
            (approved_run_id, season, start_week, pending_run_id),
        )

    # For projections (keyed on season, week)
    cur.execute(
        """
        DELETE FROM projections
        WHERE season = %s
          AND (pipeline_run_id IS NULL OR pipeline_run_id <> %s)
        """,
        (season, pending_run_id),
    )
    cur.execute(
        """
        UPDATE projections
        SET pipeline_run_id = %s
        WHERE season = %s
          AND pipeline_run_id = %s
        """,
        (approved_run_id, season, pending_run_id),
    )


def log_database_size(cur) -> None:
    """Log current physical database size to monitor 512 MB cap."""
    try:
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        size_pretty = cur.fetchone()[0]
        cur.execute("SELECT pg_database_size(current_database())")
        size_bytes = cur.fetchone()[0]
        mb = size_bytes / (1024 * 1024)
        logger.info("Database physical size: %s (%.2f MB / 512 MB limit)", size_pretty, mb)
    except Exception as exc:
        logger.warning("Could not query pg_database_size: %s", exc)


def run_weekly_refresh(
    database_url: str,
    season: int = 2026,
    dry_run: bool = False,
) -> int:
    dsn = normalize_dsn(database_url)
    t_start = time.monotonic()
    logger.info("Starting weekly refresh sequence for season=%d (dry_run=%s)", season, dry_run)

    # 1. Resolve current completed week & start_week
    step_name = "resolve_week"
    try:
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                season, completed_week, start_week = resolve_season_and_week(cur, season)
        logger.info(
            "Resolved season=%d: completed_week=%d -> target start_week=%d",
            season,
            completed_week,
            start_week,
        )
        if start_week > 18:
            logger.info("Season %d has completed all 18 weeks. Nothing new to refresh.", season)
            return 0
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    if dry_run:
        logger.info("Dry-run mode enabled: skipping live mutations.")
        return 0

    pending_run_id = f"weekly_auto_{season}_pending"
    approved_run_id = f"weekly_auto_{season}"

    # 2. Ingest weekly sources via Orchestrator
    step_name = "orchestrator_ingest"
    try:
        logger.info("Step 2: Running Orchestrator with weekly sources %s...", WEEKLY_SOURCES)
        orch = Orchestrator(dsn)
        orch_res = orch.run(seasons=[season], sources=WEEKLY_SOURCES)
        if not orch_res or not orch_res[0].ok:
            raise RuntimeError(f"Orchestrator run failed: {orch_res}")
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 3. Upsert Player ID Bridge
    step_name = "ff_playerids_bridge"
    try:
        logger.info("Step 3: Refreshing fantasy_player_ids bridge...")
        n_ids = upsert_playerids(dsn)
        logger.info("Bridge upserted %d player ID records.", n_ids)
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 4. Materialize stack projections with pending run id
    step_name = "materialize_stack_projections"
    try:
        logger.info("Step 4: Materializing stack projections with pipeline_run_id=%s...", pending_run_id)
        from scripts.materialize_stack_projections import main as mat_stack_main
        old_argv = sys.argv
        sys.argv = [
            "materialize_stack_projections.py",
            "--database-url",
            dsn,
            "--pipeline-run-id",
            pending_run_id,
        ]
        try:
            mat_stack_main()
        finally:
            sys.argv = old_argv
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 5. Materialize season simulation with pending run id
    step_name = "materialize_season_simulation"
    try:
        logger.info(
            "Step 5: Materializing season simulation (season=%d, start_week=%d, run_id=%s)...",
            season,
            start_week,
            pending_run_id,
        )
        from scripts.materialize_season_simulation import materialize as mat_sim_materialize
        mat_sim_materialize(
            season=season,
            start_week=start_week,
            end_week=18,
            n_simulations=300,
            database_url=dsn,
            pipeline_run_id=pending_run_id,
        )
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 6. Materialize team game predictions
    step_name = "materialize_team_game_predictions"
    try:
        logger.info("Step 6: Materializing team game predictions (season=%d, week=%d)...", season, start_week)
        from scripts.materialize_team_game_predictions import materialize as mat_team_game
        mat_team_game(
            season=season,
            week=start_week,
            database_url=dsn,
            pipeline_run_id=pending_run_id,
        )
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 7. Verification / acceptance check
    step_name = "verify_acceptance"
    try:
        logger.info("Step 7: Verifying season acceptance (season=%d, start_week=%d)...", season, start_week)
        from scripts.verify_season_acceptance import main as verify_main
        old_argv = sys.argv
        sys.argv = [
            "verify_season_acceptance.py",
            "--season",
            str(season),
            "--start-week",
            str(start_week),
        ]
        try:
            rc = verify_main()
            if rc != 0:
                raise RuntimeError(f"verify_season_acceptance returned non-zero exit code: {rc}")
        finally:
            sys.argv = old_argv
    except Exception as exc:
        logger.error("[%s] FAILED (acceptance check aborted promotion): %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 8. Atomic Promote in a single transaction
    step_name = "atomic_promote"
    try:
        logger.info("Step 8: Performing atomic promotion...")
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                promote_run_id(
                    cur,
                    season=season,
                    start_week=start_week,
                    pending_run_id=pending_run_id,
                    approved_run_id=approved_run_id,
                )
            conn.commit()
        logger.info("Atomic promotion complete.")
    except Exception as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 9. Retention & pruning
    step_name = "retention_and_pruning"
    try:
        logger.info("Step 9: Running staging retention and pipeline run pruning...")
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                n_purged = purge_stale_staging(cur)
                logger.info("Staging retention purged %d processed rows.", n_purged)
                keep_ids = load_keep_run_ids()
                violations = check_guard(cur, keep_ids)
                if violations:
                    logger.warning("Prune guard found violations (%d); skipping stale run delete: %s", len(violations), violations)
                else:
                    deleted = delete_stale_rows(cur, keep_ids)
                    logger.info("Pruned stale pipeline rows: %s", deleted)
            conn.commit()
    except Exception as exc:
        logger.warning("[%s] Non-fatal retention warning: %s", step_name, exc, exc_info=True)

    # 10. Vacuum and size check
    step_name = "vacuum_and_size"
    try:
        logger.info("Step 10: Running VACUUM (ANALYZE) on core tables...")
        vacuum_tables(dsn)
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                log_database_size(cur)
    except Exception as exc:
        logger.warning("[%s] Non-fatal vacuum warning: %s", step_name, exc)

    elapsed = time.monotonic() - t_start
    logger.info("Weekly refresh completed successfully in %.2fs.", elapsed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season to refresh (default: 2026)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: calculate target week and exit without writes",
    )
    args = parser.parse_args()
    return run_weekly_refresh(
        database_url=args.database_url,
        season=args.season,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
