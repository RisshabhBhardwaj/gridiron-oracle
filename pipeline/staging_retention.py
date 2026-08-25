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
    #
    # A non-positive retention window is refused outright, same convention
    # as scripts/prune_stale_pipeline_runs.py's load_keep_run_ids() guard
    # against an empty keep set: `interval '-1 days'` flips the WHERE
    # clause's `ingested_at < now() - interval '-1 days'` into
    # `ingested_at < now() + interval '1 days'`, which matches essentially
    # every row (including rows ingested moments ago), not just stale ones.
    # `retention_days=0` is refused too — it purges everything already
    # processed, which is never the intent of a "retention window".
    try:
        retention_days_int = int(retention_days)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"retention_days must be a positive integer; got {retention_days!r}"
        ) from exc
    if retention_days_int <= 0:
        raise ValueError(
            f"retention_days must be a positive integer; got {retention_days!r} "
            "(a non-positive window would delete rows that aren't stale)"
        )

    cur.execute(
        f"""
        DELETE FROM staging_nflreadpy
         WHERE processed
           AND source_type <> 'rosters'
           AND ingested_at < now() - interval '{retention_days_int} days'
        """
    )
    deleted = cur.rowcount
    logger.info(
        "purge_stale_staging: deleted %d row(s) from staging_nflreadpy "
        "(processed, source_type != 'rosters', older than %d day(s))",
        deleted,
        retention_days_int,
    )
    return deleted


# Source types whose staged payloads are pure capture volume: once
# normalize has consumed them the production table holds everything of
# value, and nothing in this codebase reads them back. `rosters` is
# deliberately absent — see the carve-out comment in purge_stale_staging.
HIGH_VOLUME_SOURCE_TYPES: tuple[str, ...] = (
    "depth_charts",
    "player_stats",
    "snap_counts",
    "nextgen_stats",
    "team_stats",
    "schedules",
    "participation",
    "ftn_charting",
    "combine",
)


def purge_processed_staging(
    cur, source_types: "tuple[str, ...]" = HIGH_VOLUME_SOURCE_TYPES
) -> int:
    """Delete ALREADY-PROCESSED staging rows for high-volume source types
    without waiting out a retention window. Returns rows deleted.

    purge_stale_staging's 14-day age gate assumes the staging table is
    small enough that two weeks of captures can sit around for debugging.
    On the 512 MB serving database that assumption is false: nflreadpy's
    depth-chart feed returns every published snapshot of the season, so one
    weekly capture staged 469,064 rows / 242 MB — 55% of the entire
    database — which normalize collapses to about 3,000 production rows.
    Every one of those was ingested the same day, so the age gate made the
    sweep a no-op and the next week's capture would have run the database
    out of space.

    `processed` is still the hard precondition: a row normalize has not yet
    consumed is never touched, whatever its source type. `rosters` is
    excluded by construction (it is absent from HIGH_VOLUME_SOURCE_TYPES)
    because pipeline/provenance.py reads roster payloads back with no
    `processed` filter — the same call site purge_stale_staging carves out.

    Callers own the transaction, same convention as purge_stale_staging.
    Follow this with a VACUUM (scripts/weekly_refresh.py step 10 already
    runs one) — DELETE leaves the pages as reusable free space rather than
    returning them, which is what stops the table growing week over week.
    """
    if not source_types:
        return 0
    if "rosters" in source_types:
        raise ValueError(
            "'rosters' cannot be purged by source type: pipeline/provenance.py "
            "reads those payloads back with no `processed` filter. See the "
            "carve-out comment in purge_stale_staging."
        )
    cur.execute(
        """
        DELETE FROM staging_nflreadpy
         WHERE processed
           AND source_type = ANY(%s)
        """,
        (list(source_types),),
    )
    deleted = cur.rowcount
    logger.info(
        "purge_processed_staging: deleted %d processed row(s) from "
        "staging_nflreadpy for source_type in %s",
        deleted,
        source_types,
    )
    return deleted
