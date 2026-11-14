"""One writer per (season, week) for the forward ``projections`` rows (review L2).

The forward producer (weekly refresh step 4b) and the Friday availability refresh both write the same
week's rows. A transaction-scoped advisory lock keeps them from interleaving: it is released by the
commit or rollback that ends the write transaction, so a crash cannot leave it held. The attempt is
non-blocking: a second writer refuses at once and the caller retries, instead of queueing behind a
transaction of unknown length.
"""
from __future__ import annotations

import time
from typing import Callable

# Two-int advisory key: (namespace, season * 100 + week). The namespace keeps this lock from colliding
# with any other advisory lock the application takes. "GOFW" as a 32-bit int.
WEEK_LOCK_NAMESPACE = 0x474F4657


def week_lock_key(season: int, week: int) -> tuple[int, int]:
    return WEEK_LOCK_NAMESPACE, int(season) * 100 + int(week)


def try_lock_week(conn, season: int, week: int) -> bool:
    """Take the (season, week) write lock for the current transaction. False when another holds it."""
    with conn.cursor() as cur:
        cur.execute(
            "-- wk:lock\nSELECT pg_try_advisory_xact_lock(%s, %s)", week_lock_key(season, week)
        )
        row = cur.fetchone()
    return bool(row and row[0] is True)


def lock_week_with_retry(
    conn, season: int, week: int, *, attempts: int = 3, delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """``try_lock_week`` up to ``attempts`` times, doubling ``delay`` between tries.

    For callers that must not give up on the first collision (a restore after a failed refresh). Each
    failed try ends its transaction before sleeping, so nothing is held while waiting. True means the
    lock is held by the current transaction.
    """
    for i in range(max(1, attempts)):
        if try_lock_week(conn, season, week):
            return True
        conn.rollback()
        if i < attempts - 1:
            sleep(delay * (2 ** i))
    return False
