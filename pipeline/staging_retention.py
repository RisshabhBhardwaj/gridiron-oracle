"""
Retention for the `staging_nflreadpy` ingestion capture log.

`staging_nflreadpy` is a deliberately append-only capture of every raw
ingestion payload this project has pulled from nflreadpy. It is not
pruned by the normal ETL path, so it grows without bound unless something
explicitly deletes old, already-processed rows. This module provides that
"something" as a small, reusable, importable function — it does not run
anything on import, and it is not itself a script. `scripts/weekly_refresh.py`
(a separate task) is expected to import `purge_stale_staging` and call it as
one step of its cron sequence.

Safety notes:
  * Only rows already marked `processed` are eligible — anything still
    awaiting processing is never touched, regardless of age.
  * `source_type = 'rosters'` rows are carved out entirely (see the comment
    above the DELETE below for why).
  * The retention window defaults to 14 days but is caller-configurable.

This module deliberately never opens its own connection against a
production database URL — callers pass in a live cursor/connection so this
code stays test-friendly and side-effect-free on import.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 14


def purge_stale_staging(cur, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Delete processed, non-roster `staging_nflreadpy` rows older than the
    retention window. Returns the number of rows deleted.

    Takes an open psycopg2 cursor (not a connection/database_url) and does
    not manage transactions itself — the caller is responsible for
    commit/rollback, same convention as
    scripts/prune_stale_pipeline_runs.py's delete_stale_rows(), which this
    mirrors so a future scripts/weekly_refresh.py can compose this with its
    own connection/transaction lifecycle.
    """
    # `rosters` rows are excluded from this retention sweep no matter how
    # old or how long-processed they are. pipeline/provenance.py's
    # backfill_player_season_profiles selects
    # `WHERE source_type = 'rosters'` with NO `processed` filter, relying on
    # `ORDER BY id` recency over staging_nflreadpy to find the latest roster
    # snapshot for as-of provenance. If this retention sweep ever purged
    # processed roster rows, that recency lookup would silently degrade
    # (older/fewer snapshots surviving than actually exist) with no error
    # anywhere — hence the explicit `source_type <> 'rosters'` carve-out
    # below. Do not remove it in a future cleanup without re-checking that
    # call site.
    # retention_days is coerced to int and interpolated directly (not
    # passed as a %s bind param) because psycopg2 parameters are adapted as
    # literal values, not as raw SQL tokens — a %s inside `interval '%s
    # days'` would still get quoted/escaped as a string, not spliced in as
    # the brief's literal `interval '14 days'`. The int() coercion is what
    # keeps this injection-safe despite the interpolation.
    cur.execute(
        f"""
        DELETE FROM staging_nflreadpy
         WHERE processed
           AND source_type <> 'rosters'
           AND ingested_at < now() - interval '{int(retention_days)} days'
        """
    )
    deleted = cur.rowcount
    logger.info(
        "purge_stale_staging: deleted %d row(s) from staging_nflreadpy "
        "(processed, source_type != 'rosters', older than %d day(s))",
        deleted,
        retention_days,
    )
    return deleted
