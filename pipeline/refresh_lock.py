"""One weekly refresh per season at a time (review N8).

A refresh's ``clear_leftovers`` deletes every pending row of the season, including the rows another refresh
has written and is about to promote. Two refreshes of one season must therefore never overlap, and the weekly
timer alone does not guarantee it: a manual run or a retry can start while a scheduled one is still going.

The lock is a session-level Postgres advisory lock on a connection of its own (TCP keepalives on). It lives as long as that
connection, so a process that dies frees it when the server drops the connection, and nothing has to be
cleaned up by hand. The namespace differs from the week lock's (``pipeline/week_lock.py``), which guards single
writes inside a transaction. The attempt does not wait: a second refresh refuses at once.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2

logger = logging.getLogger(__name__)

# "GRRF" as a 32-bit int; not WEEK_LOCK_NAMESPACE. The second key is the season.
REFRESH_LOCK_NAMESPACE = 0x47525246


# The lock is only as good as its connection. Keepalives make a silently dropped link (a NAT, a firewall)
# fail within about a minute instead of leaving a lock that looks held on one side and is gone on the other.
KEEPALIVES = {"keepalives": 1, "keepalives_idle": 30, "keepalives_interval": 10, "keepalives_count": 3}


class RefreshAlreadyRunning(RuntimeError):
    """Another weekly refresh of this season holds the lock; this one changed nothing."""


@contextmanager
def season_refresh_lock(dsn: str, season: int) -> Iterator[None]:
    """Hold the season's refresh lock for the whole ``with`` block, or raise ``RefreshAlreadyRunning``."""
    conn = psycopg2.connect(dsn, **KEEPALIVES)
    try:
        conn.autocommit = True  # no transaction stays open for the length of the run
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s, %s)", (REFRESH_LOCK_NAMESPACE, int(season)))
            row = cur.fetchone()
        if not (row and row[0] is True):
            raise RefreshAlreadyRunning(
                f"another weekly refresh of season {season} is running (or its connection is still open). "
                "Nothing was changed. Wait for it to finish, or stop it, and run again."
            )
        try:
            yield
        finally:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s, %s)", (REFRESH_LOCK_NAMESPACE, int(season)))
            except Exception:  # the connection is closed next, which frees the lock anyway
                logger.warning("could not release the refresh lock explicitly; closing the connection frees it", exc_info=True)
    finally:
        conn.close()


__all__ = ["REFRESH_LOCK_NAMESPACE", "RefreshAlreadyRunning", "season_refresh_lock"]
