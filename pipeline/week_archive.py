"""Archive and restore of a forward week around ``--replace-week`` (review M1).

``projections`` is unique on (player_id, game_id, stat), without the run id, so a replaced week cannot
be staged beside the served rows. The producer copies the old rows into ``projections_replaced`` in the
same transaction that deletes them and inserts the new ones. Then either:

* the promotion (weekly refresh step 8) discards the copy in its own transaction, or
* a failure in steps 5-8 calls ``restore_week``, which puts the old rows back.

Restore never acts on a week with no archive, so it cannot delete rows that were never replaced. The
copy uses the live column list of ``projections``, so a column added later fails loudly (the archive
lacks it) instead of silently dropping data.
"""
from __future__ import annotations

import logging

from pipeline.week_lock import try_lock_week

logger = logging.getLogger(__name__)

ARCHIVE_TABLE = "projections_replaced"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(cur) -> str:
    cur.execute(
        "-- wa:columns\nSELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = 'projections' ORDER BY ordinal_position"
    )
    names = [r[0] for r in cur.fetchall()]
    if not names:
        raise RuntimeError("the projections table has no columns in this schema; refusing to archive")
    return ", ".join(_quote(n) for n in names)


def archived_rows(cur, season: int, week: int) -> int:
    """Rows of the week still in the archive (an earlier replace that was neither promoted nor restored)."""
    cur.execute(
        "-- wa:existing\nSELECT count(*) FROM projections_replaced WHERE season = %s AND week = %s",
        (season, week),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def archive_week(cur, season: int, week: int) -> int:
    """Copy the week's rows to the archive, then delete them. Caller owns the transaction."""
    cols = _columns(cur)
    cur.execute(
        f"INSERT INTO projections_replaced ({cols}) SELECT {cols} FROM projections "
        "WHERE season = %s AND week = %s",
        (season, week),
    )
    archived = int(cur.rowcount)
    cur.execute("DELETE FROM projections WHERE season = %s AND week = %s", (season, week))
    return archived


def discard_week(cur, season: int, week: int) -> None:
    """Drop the week's archive. Called in the promotion transaction, so it vanishes only on success."""
    cur.execute("DELETE FROM projections_replaced WHERE season = %s AND week = %s", (season, week))


def restore_week(conn, season: int, week: int) -> int:
    """Put the archived rows back, replacing whatever the failed refresh left for the week.

    Returns the rows restored; 0 (and no change) when the week has no archive. Raises when another
    writer holds the week, or on any error, after rolling back.
    """
    try:
        if not try_lock_week(conn, season, week):
            raise RuntimeError(
                f"another writer holds the lock for season={season} week={week}; the archive was not restored"
            )
        with conn.cursor() as cur:
            n = archived_rows(cur, season, week)
            if n == 0:
                conn.rollback()
                return 0
            cols = _columns(cur)
            cur.execute("DELETE FROM projections WHERE season = %s AND week = %s", (season, week))
            cur.execute(
                f"INSERT INTO projections ({cols}) SELECT {cols} FROM projections_replaced "
                "WHERE season = %s AND week = %s",
                (season, week),
            )
            discard_week(cur, season, week)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    logger.warning("restored %d archived rows for season=%d week=%d", n, season, week)
    return n


__all__ = ["ARCHIVE_TABLE", "archive_week", "archived_rows", "discard_week", "restore_week"]
