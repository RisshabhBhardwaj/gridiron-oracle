#!/usr/bin/env python3
"""
Weekly cloud refresh cron sequencer for Gridiron Oracle.

Runs weekly (e.g. Tuesday morning post-MNF) in Railway / Cloud to incrementally
ingest the latest week of data, re-run feature engineering & ID bridge,
materialize models into pending pipeline_run_ids, verify acceptance,
and atomically promote to the approved pipeline_run_id ('weekly_auto_{season}').
Each run writes under its own pending id ('weekly_auto_{season}_pending_<uuid>') and promotes only that id;
the start of every run clears what earlier, failed or killed runs left (pipeline/pending_runs.py).

Fail-loud contract: Every step aborts on error, logging details and writing a
dead_letter row to Postgres so failures are never silent.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import ExitStack, closing
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml import season_constants
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.availability_refresh import default_run_ids, reapply_latest_availability
from pipeline.orchestrator import Orchestrator, OrchestratorSummary, _write_dead_letter
from pipeline.pending_runs import (
    clear_leftovers, new_pending_run_id, require_promotion_complete, require_run_intact,
)
from pipeline.refresh_lock import season_refresh_lock
from pipeline.schema import normalize_dsn
from pipeline.week_archive import discard_week, restore_week
from pipeline.week_lock import lock_week_with_retry
from pipeline.staging_retention import purge_processed_staging, purge_stale_staging
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


def _orchestrator_failure(summary: OrchestratorSummary) -> str | None:
    """
    Return the first failed step across all seasons as
    "season=<y> step=<name>: <error>", or None when every step succeeded.

    Orchestrator.run() records step failures in StepResult.ok / .error and
    returns normally, so the caller has to look for them here.
    """
    for season_result in summary.season_results:
        for step in season_result.steps:
            if not step.ok:
                detail = step.error or "failed without an error message"
                return f"season={season_result.season} step={step.name}: {detail}"
    for step in summary.global_steps:
        if not step.ok:
            detail = step.error or "failed without an error message"
            return f"run-level step={step.name}: {detail}"
    return None


def current_season() -> int:
    """
    The season the weekly refresh targets, from ml/season_constants.py.

    The weekly job refreshes the season after the last complete one. The two
    constants must agree on that, so a bump of only one of them fails here
    instead of refreshing the wrong season.
    """
    current = season_constants.CURRENT_SEASON
    last_complete = season_constants.LAST_COMPLETE_SEASON
    if current != last_complete + 1:
        raise RuntimeError(
            f"CURRENT_SEASON={current} must equal LAST_COMPLETE_SEASON+1 "
            f"({last_complete + 1}); update ml/season_constants.py before refreshing."
        )
    return current


def _require_approved_run_id(approved_run_id: str, season: int) -> None:
    """Raise unless the manifest approves approved_run_id, the set step 9 keeps."""
    keep_ids = load_keep_run_ids()
    if approved_run_id not in keep_ids:
        raise RuntimeError(
            f"{approved_run_id} is not in projection_policy.approved_pipeline_run_ids of "
            f"releases/current_baseline.json, so season {season} cannot be promoted. "
            "Approve the run id through the release process first."
        )


def resolve_season_and_week(cur, season: int | None = None) -> tuple[int, int, int]:
    """
    Resolve (season, completed_week, start_week) based on kickoffs in games table.
    """
    season = current_season() if season is None else season
    cur.execute(
        """
        SELECT COALESCE(MAX(week), 0)
        FROM games
        WHERE season = %s AND kickoff_at < NOW()
        """,
        (season,),
    )
    row = cur.fetchone()
    completed_week = int(row[0]) if row and row[0] is not None else 0
    start_week = completed_week + 1
    return season, completed_week, start_week


def promote_run_id(
    cur,
    season: int,
    start_week: int,
    pending_run_id: str,
    approved_run_id: str,
) -> dict[str, int]:
    """
    Relabel pending_run_id rows as approved_run_id. Relabel only: this deletes nothing.

    Returns the rows each UPDATE relabelled, so the caller can check them before it commits (review N12). The
    projections relabel is split: ``projections`` counts the target week's rows (what the producer recorded),
    ``projections_other_weeks`` the rest of the season (the stack rows).

    pending_run_id is the run's own id (``new_pending_run_id``), so a failed or killed earlier run's rows,
    which carry another id, are never relabelled. The caller promotes only after that run's step 7 passed.
    Materializers never overwrite a row served by another approved run, so
    approved rows keep their ids and stay served. Leftovers of other runs are not served; the next
    refresh's ``clear_leftovers`` and the step 9 prune (which keeps only the manifest's approved ids)
    remove them. Scope of the relabel:
      - season tables: (season, start_week)
      - projections: season (this refresh's season only; the pinned stack rows
        for earlier seasons are not promoted)
      - team_game_predictions: (season, week=start_week)
    """
    logger.info(
        "Atomically promoting %s -> %s for season=%d, start_week=%d",
        pending_run_id,
        approved_run_id,
        season,
        start_week,
    )
    relabelled: dict[str, int] = {}
    for table in SEASON_START_WEEK_TABLES:
        cur.execute(
            f"""
            UPDATE {table}
            SET pipeline_run_id = %s
            WHERE season = %s AND start_week = %s
              AND pipeline_run_id = %s
            """,
            (approved_run_id, season, start_week, pending_run_id),
        )
        relabelled[table] = cur.rowcount

    # The target week first, on its own, so its count can be compared to what the producer recorded.
    cur.execute(
        """
        UPDATE projections
        SET pipeline_run_id = %s
        WHERE season = %s AND week = %s
          AND pipeline_run_id = %s
        """,
        (approved_run_id, season, start_week, pending_run_id),
    )
    relabelled["projections"] = cur.rowcount
    cur.execute(
        """
        UPDATE projections
        SET pipeline_run_id = %s
        WHERE season = %s
          AND pipeline_run_id = %s
        """,
        (approved_run_id, season, pending_run_id),
    )
    relabelled["projections_other_weeks"] = cur.rowcount
    cur.execute(
        """
        UPDATE team_game_predictions
        SET model_run_id = %s
        WHERE season = %s AND week = %s
          AND model_run_id = %s
        """,
        (approved_run_id, season, start_week, pending_run_id),
    )
    relabelled["team_game_predictions"] = cur.rowcount
    return relabelled


def _restore_after_failure(dsn: str, season: int, week: int, replace_week: bool) -> None:
    """After ``--replace-week`` the old week is archived. Put it back when the refresh fails.

    A no-op without ``--replace-week`` and when nothing was archived. A failed restore leaves the week
    unserved, so it is logged at CRITICAL and dead-lettered under its own source.
    """
    if not replace_week:
        return
    try:
        with closing(psycopg2.connect(dsn)) as conn:
            restored = restore_week(conn, season, week)
        logger.error("--replace-week: the refresh failed; restored %d archived rows for week %d", restored, week)
    except Exception as exc:
        logger.critical("--replace-week: RESTORE FAILED, week %d may be unserved: %s", week, exc, exc_info=True)
        _write_dead_letter(
            dsn, "weekly_refresh/restore_replaced_week",
            f"restore of the archived week {week} failed: {exc}. Run scripts/restore_replaced_week.py", season,
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


def produce_forward_week(
    dsn: str, season: int, week: int, *, replace_week: bool = False, pending_id: str | None = None,
) -> dict:
    """Capture availability, build the forward features at as_of=now, run the producer.

    The order is the causality rule: an availability snapshot fetched after ``as_of`` is
    unusable, so it is captured first. Any failure raises; the caller dead-letters and stops.
    """
    from datetime import datetime, timezone

    from ml.forward_serving import run_forward
    from pipeline.availability_store import capture_snapshot
    from scripts.build_forward_features import build_forward_snapshot

    now = datetime.now(timezone.utc)
    snapshot_id = f"forward_features_{season}_w{week:02d}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    with closing(psycopg2.connect(dsn)) as conn:
        availability = capture_snapshot(conn, season, week)
    logger.info("Availability snapshot captured: %s", availability)
    # as_of is read after the capture, so the snapshot's fetch time is not after it.
    built = build_forward_snapshot(
        dsn, season, week, as_of=datetime.now(timezone.utc), pipeline_run_id=snapshot_id,
    )
    with closing(psycopg2.connect(dsn)) as conn:
        report = run_forward(
            conn, season, week, feature_run_id=built["pipeline_run_id"], replace_week=replace_week,
            pending_id=pending_id,
        )
        # The producer stored the statuses known at its own as_of. A Friday refresh may have stored a
        # newer snapshot since (a replace after Friday): apply it to the games not started (review N2).
        reapply_latest_availability(
            conn, season, week, run_ids=default_run_ids(season, pending_id), reason="after_producer",
        )
    logger.info("Forward producer wrote: %s", {k: report.get(k) for k in ("rows_served", "rows_not_served")})
    return report



class _Tee:
    """Write to the real stdout and keep a copy."""

    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, text):
        self.parts.append(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


def _run_gate(gate) -> tuple[int, str]:
    """Run the acceptance gate; return (exit code, the reason it printed).

    The gate prints a JSON report whose ``diagnosis`` says why it failed. Step 7 used to keep only the
    exit code, so a dead letter read "returned non-zero exit code: 1" and the operator had to rerun the
    gate to learn why. The output still reaches stdout; the reason is empty when it is not readable JSON.
    """
    import contextlib
    import json

    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        rc = gate()
    try:
        reason = str(json.loads("".join(tee.parts)).get("diagnosis") or "")
    except (ValueError, AttributeError):
        reason = ""
    return rc, reason

def run_weekly_refresh(
    database_url: str,
    season: int | None = None,
    dry_run: bool = False,
    replace_week: bool = False,
) -> int:
    dsn = normalize_dsn(database_url)
    t_start = time.monotonic()
    if season is None:
        season = current_season()
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
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    if dry_run:
        logger.info("Dry-run mode enabled: skipping live mutations.")
        return 0

    pending_run_id = new_pending_run_id(season)  # this run's own id; step 8 promotes exactly this
    approved_run_id = f"weekly_auto_{season}"

    # Fail closed before any write. Step 8 promotes to approved_run_id and step 9
    # deletes every row whose id is not in the manifest, so a season the manifest
    # does not approve would be promoted and then pruned.
    step_name = "approval_check"
    try:
        _require_approved_run_id(approved_run_id, season)
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # One refresh per season at a time (review N8): the cleanup below deletes every pending row of the season,
    # so a second refresh must not start while this one is between its checks and its promotion. The lock
    # lives on its own connection until this function returns; a dead process frees it.
    step_name = "refresh_lock"
    with ExitStack() as held:
        try:
            season_lock = held.enter_context(season_refresh_lock(dsn, season))
        except (Exception, SystemExit) as exc:
            logger.error("[%s] FAILED: %s", step_name, exc)
            _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
            return 1
        return _refresh_steps(
            dsn, season, start_week, replace_week, pending_run_id, approved_run_id, t_start, season_lock,
        )


def _refresh_steps(
    dsn: str, season: int, start_week: int, replace_week: bool, pending_run_id: str, approved_run_id: str,
    t_start: float, season_lock,
) -> int:
    """Steps 2-10 of the weekly refresh, run while the season's refresh lock is held (``season_lock``)."""
    # Clear what earlier runs left (review N1): restore an archived week or stop, then delete leftover
    # pending rows so a failed or killed run's unchecked rows can never be served.
    step_name = "clear_leftovers"
    try:
        with closing(psycopg2.connect(dsn)) as conn:
            clear_leftovers(conn, season)
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 2. Ingest weekly sources via Orchestrator
    step_name = "orchestrator_ingest"
    try:
        logger.info("Step 2: Running Orchestrator with weekly sources %s...", WEEKLY_SOURCES)
        orch = Orchestrator(dsn)
        orch_res = orch.run(seasons=[season], sources=WEEKLY_SOURCES)
        failure = _orchestrator_failure(orch_res)
        if failure is not None:
            raise RuntimeError(f"Orchestrator run failed: {failure}")
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 3. Upsert Player ID Bridge
    step_name = "ff_playerids_bridge"
    try:
        logger.info("Step 3: Refreshing fantasy_player_ids bridge...")
        n_ids = upsert_playerids(dsn)
        logger.info("Bridge upserted %d player ID records.", n_ids)
    except (Exception, SystemExit) as exc:
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
            # main() reports a refusal as a return code, not an exception. Treat it as a failure
            # here, or the run would continue to step 8 and promote without this step's rows.
            rc = mat_stack_main()
            if rc != 0:
                raise RuntimeError(f"materialize_stack_projections refused (return code {rc})")
        finally:
            sys.argv = old_argv
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1

    # 4b. Forward producer (decision 0012): the pinned base models write the forward week's
    # projections under the pending id. Order is causal: availability is captured first, then
    # features are built with as_of=now, so the snapshot's fetch time is never after as_of.
    step_name = "forward_producer"
    try:
        logger.info("Step 4b: Forward producer for season=%d week=%d (run id %s)...",
                    season, start_week, pending_run_id)
        produce_forward_week(dsn, season, start_week, replace_week=replace_week, pending_id=pending_run_id)
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        _restore_after_failure(dsn, season, start_week, replace_week)
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
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        _restore_after_failure(dsn, season, start_week, replace_week)
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
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        _restore_after_failure(dsn, season, start_week, replace_week)
        return 1

    # 7. Verification / acceptance check
    step_name = "verify_acceptance"
    try:
        logger.info("Step 7: Verifying season acceptance (season=%d, start_week=%d)...", season, start_week)
        from scripts.verify_season_acceptance import main as verify_main
        # C-1: the statistical gate scores the pending candidate against realized
        # points, which do not exist for a forward (unplayed) week — its join is
        # always empty there, so it can never pass and would block every promotion.
        # For a forward week the structural gate applies instead: the pending run
        # must have produced rows for the target week on every surface. The
        # statistical gate still runs whenever the window has realized data.
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM game_logs
                        WHERE season = %s AND week BETWEEN %s AND %s
                    )
                    """,
                    (season, start_week, 18),
                )
                row = cur.fetchone()
                has_realized = bool(row[0]) if row else False
        old_argv = sys.argv
        if has_realized:
            sys.argv = [
                "verify_season_acceptance.py",
                "--season",
                str(season),
                "--start-week",
                str(start_week),
                "--database-url",
                dsn,
            ]
        else:
            logger.info(
                "Step 7: start_week=%d has no realized game_logs; running the forward-week "
                "structural gate on %s instead of the statistical gate.",
                start_week, pending_run_id,
            )
            sys.argv = [
                "verify_season_acceptance.py",
                "--season",
                str(season),
                "--start-week",
                str(start_week),
                "--structural-only",
                "--pipeline-run-id",
                pending_run_id,
                "--approved-run-id",
                approved_run_id,
                "--database-url",
                dsn,
            ]
        try:
            rc, gate_reason = _run_gate(verify_main)
            if rc != 0:
                raise RuntimeError(
                    f"verify_season_acceptance returned non-zero exit code: {rc}"
                    + (f" ({gate_reason})" if gate_reason else "")
                )
        finally:
            sys.argv = old_argv
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED (acceptance check aborted promotion): %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        _restore_after_failure(dsn, season, start_week, replace_week)
        return 1

    # 8. Atomic Promote in a single transaction
    step_name = "atomic_promote"
    try:
        logger.info("Step 8: Performing atomic promotion...")
        # The season lock is only as good as its connection: prove it still answers before relabelling.
        season_lock.ensure_alive()
        with psycopg2.connect(dsn) as conn:
            # Every cleanup and every Friday write takes the week lock, so holding it here serializes them
            # against the relabel (review N12). It is released by the commit or rollback below.
            if not lock_week_with_retry(conn, season, start_week, sleep=time.sleep):
                raise RuntimeError(
                    f"another writer holds the week lock for season={season} week={start_week}; the promotion "
                    "did not start, nothing was promoted and the archived week, if any, is kept"
                )
            with conn.cursor() as cur:
                # Only a run whose producer wrote its forecast_runs row, and whose rows are all still there.
                recorded = require_run_intact(cur, season, start_week, pending_run_id)
                relabelled = promote_run_id(
                    cur,
                    season=season,
                    start_week=start_week,
                    pending_run_id=pending_run_id,
                    approved_run_id=approved_run_id,
                )
                # A cleanup that committed since the check would leave a zero or partial relabel: raise before
                # the commit, so the transaction rolls back and discard_week never runs.
                require_promotion_complete(relabelled, recorded)
                # Same transaction: the archived old week is dropped only if the promotion commits.
                discard_week(cur, season, start_week)
            conn.commit()
        logger.info("Atomic promotion complete.")
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        _restore_after_failure(dsn, season, start_week, replace_week)
        return 1

    # A Friday refresh that stored a snapshot between the producer and the promotion could not update the
    # pending rows (it names the approved ids), so apply the newest snapshot to the rows now served (review
    # N9). The promotion has committed and stays: a failure here is logged and dead-lettered, and the next
    # Friday refresh catches up.
    step_name = "reapply_after_promotion"
    try:
        with closing(psycopg2.connect(dsn)) as conn:
            reapply_latest_availability(
                conn, season, start_week, run_ids=default_run_ids(season), reason="after_promotion",
            )
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED (the promotion stands): %s", step_name, exc, exc_info=True)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)

    # 9. Retention & pruning
    step_name = "retention_and_pruning"
    # Fail closed: the prune below deletes every row whose id is not kept. If the
    # run just promoted is not in the keep set, that delete would remove it.
    try:
        keep_ids = load_keep_run_ids()
        if approved_run_id not in keep_ids:
            raise RuntimeError(f"{approved_run_id} is not in the keep set; refusing to prune")
    except (Exception, SystemExit) as exc:
        logger.error("[%s] FAILED: %s", step_name, exc)
        _write_dead_letter(dsn, f"weekly_refresh/{step_name}", str(exc), season)
        return 1
    try:
        logger.info("Step 9: Running staging retention and pipeline run pruning...")
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                # Two sweeps, deliberately. purge_processed_staging clears
                # the high-volume capture payloads as soon as normalize has
                # consumed them: one depth-chart capture stages ~469k rows /
                # 242 MB against a 512 MB database, all ingested the same
                # day, so the age-gated sweep below can never reach them and
                # the next capture would run the database out of space.
                n_immediate = purge_processed_staging(cur)
                logger.info("Staging purge (processed, high-volume) removed %d rows.", n_immediate)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season to refresh (default: CURRENT_SEASON from ml/season_constants.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: calculate target week and exit without writes",
    )
    parser.add_argument(
        "--replace-week",
        action="store_true",
        help="MANUAL ONLY, never from a timer: the producer archives the target week's projection rows to "
             "projections_replaced, deletes them and writes the new ones in one transaction (a failure inside "
             "the producer rolls back). A later failure restores the archive; a promotion discards it. A kill "
             "between the two needs scripts/restore_replaced_week.py. Requires --dump-path to a non-empty dump.",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=None,
        help="Database dump taken before --replace-week (must exist and be non-empty)",
    )
    args = parser.parse_args(argv)
    if args.dump_path is not None and not args.replace_week:
        parser.error("--dump-path is only used with --replace-week")
    if args.replace_week:
        if args.dump_path is None:
            parser.error("--replace-week requires --dump-path: take a database dump first")
        if not args.dump_path.is_file() or args.dump_path.stat().st_size == 0:
            parser.error(f"--dump-path {args.dump_path} must be an existing, non-empty file")
    return run_weekly_refresh(
        database_url=args.database_url,
        season=args.season,
        dry_run=args.dry_run,
        replace_week=args.replace_week,
    )


if __name__ == "__main__":
    sys.exit(main())
