#!/usr/bin/env python3
"""
Prune orphaned/unapproved pipeline_run_id rows from the Neon-hosted
materialization tables.

Two months of ad-hoc materialization runs (scripts/materialize_season_
simulation.py, scripts/materialize_stack_projections.py, etc.) have left
rows in the DB tagged with pipeline_run_ids that were never approved into
releases/current_baseline.json's projection_policy.approved_pipeline_run_ids
— i.e. never wired up to actually be served (see get_season_projections /
get_projections, both of which fail closed unless a row's pipeline_run_id
is in that approved set). Neon bills by physical size (capped at 512 MB on
this project), so that dead weight is a real cost, not just clutter.

Pruned tables:
  * season_simulation_weeks  (all rows, any season)
  * season_simulations       (all rows, any season)
  * season_team_wins         (all rows, any season)
  * projections              (season >= 2024 ONLY — 2023-and-earlier rows
                               in this table are a different retention
                               concern, out of scope for this script)

Safety:
  * --dry-run defaults to TRUE. An unqualified invocation of this script
    NEVER deletes anything — you must pass --no-dry-run explicitly to
    execute deletes.
  * Guard clause (runs even in --dry-run, BEFORE any delete): for every
    (season, start_week) pair currently served by an approved run in
    season_simulations, confirm season_simulation_weeks and
    season_team_wins also have at least one row under an approved run id
    for that same pair. Since this script only ever deletes NON-kept rows,
    a kept run's own rows are never touched — so this should always pass
    UNLESS the manifest and the data have already diverged (e.g. the
    per-week breakdown for a served (season, start_week) was actually
    written under a stale, unapproved run id while the season aggregate
    was written under the approved one). That divergence is exactly what
    this guard exists to catch, and it refuses to run (non-zero exit, no
    deletes) rather than silently deleting a currently-served surface.
  * VACUUM cannot run inside a transaction block in Postgres, so deletes
    are committed first (their own transaction), then VACUUM (ANALYZE) is
    run per table on a fresh autocommit connection.

Usage:
  # Safe by default — prints what WOULD be deleted, deletes nothing.
  python scripts/prune_stale_pipeline_runs.py --database-url "$DATABASE_URL"

  # Actually prune (after reviewing the dry-run output above).
  python scripts/prune_stale_pipeline_runs.py --database-url "$DATABASE_URL" --no-dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = ROOT / "releases" / "current_baseline.json"

# Tables keyed by (season, start_week, ...) — the guard applies across these.
SEASON_START_WEEK_TABLES = (
    "season_simulation_weeks",
    "season_simulations",
    "season_team_wins",
)
# projections is keyed by (season, week) with no start_week concept, and is
# only pruned for season >= 2024 — handled separately below.
PROJECTIONS_MIN_SEASON = 2024


class GuardViolation(RuntimeError):
    """Raised when pruning would zero out a currently-served (season, start_week)."""


def load_keep_run_ids(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[str]:
    """Read projection_policy.approved_pipeline_run_ids from the baseline manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Baseline manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    raw = (payload.get("projection_policy") or {}).get("approved_pipeline_run_ids") or []
    ids = [str(item) for item in raw if item]
    if not ids:
        raise ValueError(
            f"No approved_pipeline_run_ids in {manifest_path}'s projection_policy; "
            "refusing to prune with an empty keep set (that would delete everything)."
        )
    return ids


def check_guard(cur, keep_ids: list[str]) -> list[dict]:
    """
    Return a list of violation dicts (empty if the guard passes).

    For every (season, start_week) currently served by an approved run in
    season_simulations, every OTHER season/start_week-keyed table must also
    have at least one row under an approved run id for that same pair —
    otherwise pruning (which only deletes non-kept rows) would leave that
    table with zero rows for a combination the manifest says is served.
    """
    violations: list[dict] = []
    cur.execute(
        "SELECT DISTINCT season, start_week FROM season_simulations "
        "WHERE pipeline_run_id = ANY(%s)",
        (keep_ids,),
    )
    kept_pairs = cur.fetchall()
    if not kept_pairs:
        # Nothing currently served — nothing for the guard to protect.
        return violations

    kept_seasons = [s for s, _ in kept_pairs]
    kept_start_weeks = [w for _, w in kept_pairs]
    for table in SEASON_START_WEEK_TABLES:
        cur.execute(
            f"""
            SELECT kp.season, kp.start_week
            FROM UNNEST(%s::int[], %s::int[]) AS kp(season, start_week)
            WHERE NOT EXISTS (
                SELECT 1 FROM {table} t
                WHERE t.season = kp.season AND t.start_week = kp.start_week
                  AND t.pipeline_run_id = ANY(%s)
            )
            """,
            (kept_seasons, kept_start_weeks, keep_ids),
        )
        for season, start_week in cur.fetchall():
            violations.append({
                "table": table,
                "season": season,
                "start_week": start_week,
                "reason": (
                    f"{table} has no row under an approved pipeline_run_id for "
                    f"(season={season}, start_week={start_week}), but season_simulations "
                    f"does — the manifest and the data have diverged for this combination."
                ),
            })
    return violations


def count_would_delete(cur, keep_ids: list[str]) -> dict[str, int]:
    """Row counts each table WOULD lose if pruned (never executes a DELETE)."""
    counts: dict[str, int] = {}
    for table in SEASON_START_WEEK_TABLES:
        cur.execute(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE pipeline_run_id IS NULL OR NOT (pipeline_run_id = ANY(%s))",
            (keep_ids,),
        )
        counts[table] = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM projections "
        "WHERE season >= %s AND (pipeline_run_id IS NULL OR NOT (pipeline_run_id = ANY(%s)))",
        (PROJECTIONS_MIN_SEASON, keep_ids),
    )
    counts["projections (season >= %d)" % PROJECTIONS_MIN_SEASON] = cur.fetchone()[0]
    return counts


def delete_stale_rows(cur, keep_ids: list[str]) -> dict[str, int]:
    """Execute the deletes. Caller is responsible for commit/rollback."""
    deleted: dict[str, int] = {}
    for table in SEASON_START_WEEK_TABLES:
        cur.execute(
            f"DELETE FROM {table} "
            f"WHERE pipeline_run_id IS NULL OR NOT (pipeline_run_id = ANY(%s))",
            (keep_ids,),
        )
        deleted[table] = cur.rowcount

    cur.execute(
        "DELETE FROM projections "
        "WHERE season >= %s AND (pipeline_run_id IS NULL OR NOT (pipeline_run_id = ANY(%s)))",
        (PROJECTIONS_MIN_SEASON, keep_ids),
    )
    deleted["projections (season >= %d)" % PROJECTIONS_MIN_SEASON] = cur.rowcount
    return deleted


def vacuum_tables(database_url: str) -> None:
    """VACUUM (ANALYZE) cannot run inside a transaction block, so this uses
    its own autocommit connection, run AFTER the delete transaction commits."""
    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in (*SEASON_START_WEEK_TABLES, "projections"):
                logger.info("VACUUM (ANALYZE) %s", table)
                cur.execute(f"VACUUM (ANALYZE) {table}")
    finally:
        conn.close()


def run(database_url: str, keep_ids: list[str], dry_run: bool) -> int:
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            violations = check_guard(cur, keep_ids)
            if violations:
                logger.error(
                    "Guard check FAILED — refusing to run. %d violation(s):", len(violations)
                )
                for v in violations:
                    logger.error("  %s", v["reason"])
                return 1
            logger.info("Guard check passed: no kept (season, start_week) would be zeroed out.")

            counts = count_would_delete(cur, keep_ids)

        if dry_run:
            logger.info("DRY RUN (default) — no rows deleted, no VACUUM run. Would delete:")
            for table, n in counts.items():
                logger.info("  %s: %d row(s)", table, n)
            print(json.dumps(counts, indent=2))
            return 0

        # Non-dry-run: delete inside a transaction, commit, THEN vacuum
        # autocommit-style (VACUUM cannot run inside a tx block).
        with conn.cursor() as cur:
            deleted = delete_stale_rows(cur, keep_ids)
        conn.commit()
        logger.info("Deleted:")
        for table, n in deleted.items():
            logger.info("  %s: %d row(s)", table, n)
    finally:
        conn.close()

    if not dry_run:
        vacuum_tables(database_url)
        logger.info("Prune complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument(
        "--keep-run-ids", nargs="*", default=None,
        help="pipeline_run_ids to keep. Defaults to releases/current_baseline.json's "
        "projection_policy.approved_pipeline_run_ids.",
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH,
        help="Baseline manifest to read the default keep set from.",
    )
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=True,
        help="Default TRUE. Pass --no-dry-run to actually delete + VACUUM.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    keep_ids = args.keep_run_ids if args.keep_run_ids else load_keep_run_ids(args.manifest_path)
    logger.info("Keep set (%d id(s)): %s", len(keep_ids), keep_ids)

    return run(args.database_url, keep_ids, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
