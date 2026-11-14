"""Each weekly refresh writes under its own pending run id (review N1).

Promotion relabels pending rows as approved. Under one shared pending name it relabelled every pending
row of the season, including rows a failed or killed run left behind that no step 7 ever checked. So:

* each run gets ``weekly_auto_<season>_pending_<uuid>``; step 8 promotes exactly that id;
* ``clear_leftovers`` runs at the start of every refresh and removes what earlier runs left (and puts
  back a week a killed ``--replace-week`` run had archived);
* ``require_run_intact`` refuses a promotion whose forward producer never wrote its ``forecast_runs`` row,
  or whose rows are no longer the number the producer recorded (a second refresh cleared them).

``projections`` is unique on (player_id, game_id, stat) with no run id, so a run's upsert takes over a
row with the same key. Rows a later run does not rewrite keep the older id until the cleanup removes them.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from pipeline.week_archive import archived_weeks, restore_week
from pipeline.week_lock import try_lock_week

logger = logging.getLogger(__name__)

# Tables whose run id column starts with the pending prefix while a refresh is in flight.
_SEASON_TABLES = ("season_simulation_weeks", "season_simulations", "season_team_wins")


class LeftoverRunsError(RuntimeError):
    """The start-of-refresh cleanup could not finish; the refresh must not go on."""


class RunNotProducedError(RuntimeError):
    """The run to promote has no forward-producer row: its pending rows were not produced by this run."""


def pending_prefix(season: int) -> str:
    return f"weekly_auto_{season}_pending"


def new_pending_run_id(season: int) -> str:
    return f"{pending_prefix(season)}_{uuid.uuid4().hex}"


def _weeks_with_pending(cur, season: int, prefix: str) -> list[int]:
    parts = [
        "SELECT week FROM projections WHERE season = %s AND starts_with(pipeline_run_id, %s)",
        *(f"SELECT start_week FROM {t} WHERE season = %s AND starts_with(pipeline_run_id, %s)"
          for t in _SEASON_TABLES),
        "SELECT week FROM team_game_predictions WHERE season = %s AND starts_with(model_run_id, %s)",
    ]
    cur.execute("-- pr:weeks\n" + " UNION ".join(parts) + " ORDER BY 1", (season, prefix) * len(parts))
    return [int(r[0]) for r in cur.fetchall() if r[0] is not None]


def _delete_pending_once(conn, season: int, prefix: str) -> dict[str, int] | None:
    """One attempt. None when another writer holds a week lock (nothing changed)."""
    with conn.cursor() as cur:
        weeks = _weeks_with_pending(cur, season, prefix)
    for week in weeks:
        if not try_lock_week(conn, season, week):
            conn.rollback()
            return None
    deleted: dict[str, int] = {}
    with conn.cursor() as cur:
        for table, column in (
            ("projections", "pipeline_run_id"),
            *((t, "pipeline_run_id") for t in _SEASON_TABLES),
            ("team_game_predictions", "model_run_id"),
        ):
            cur.execute(f"SELECT count(*) FROM {table} WHERE starts_with({column}, %s)", (prefix,))
            if int(cur.fetchone()[0]):
                cur.execute(f"DELETE FROM {table} WHERE starts_with({column}, %s)", (prefix,))
                deleted[table] = int(cur.rowcount)
    conn.commit()
    return deleted


def clear_leftovers(
    conn, season: int, *, restore: Callable = restore_week, attempts: int = 3, delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Start of every refresh: put back archived weeks, then delete the pending rows earlier runs left.

    Returns ``{"restored": {week: rows}, "deleted": {table: rows}}``. Raises ``LeftoverRunsError`` when an
    archive cannot be restored or a week stays locked by another writer, before anything is deleted.
    The pending rows of other seasons are untouched; only the ids that start with this season's prefix go.
    """
    prefix = pending_prefix(season)
    report: dict = {"restored": {}, "deleted": {}}
    with conn.cursor() as cur:
        weeks = archived_weeks(cur, season)
    conn.rollback()
    for week in weeks:
        try:
            report["restored"][week] = restore(conn, season, week)
        except Exception as exc:
            raise LeftoverRunsError(
                f"an earlier --replace-week run left week {week} archived and it could not be restored: {exc}. "
                f"Restore it first: scripts/restore_replaced_week.py --season {season} --week {week}"
            ) from exc
        logger.warning("restored %s rows of the archived week %d left by an earlier run", report["restored"][week], week)

    for i in range(max(1, attempts)):
        deleted = _delete_pending_once(conn, season, prefix)
        if deleted is not None:
            break
        if i < attempts - 1:
            sleep(delay * (2 ** i))
    else:
        raise LeftoverRunsError(
            f"another writer holds a week lock for season {season} (the Friday availability refresh or a "
            f"second refresh) after {attempts} attempts; the leftovers of earlier runs were not cleared. "
            "Retry when it has finished."
        )
    report["deleted"] = deleted
    if deleted:
        logger.warning(
            "cleared leftover pending rows of earlier runs (%s*): %s", prefix,
            ", ".join(f"{t}={n}" for t, n in deleted.items()),
        )
    return report


def _recorded_rows(cur, season: int, week: int, pending_id: str) -> list:
    cur.execute(
        "-- pr:producer\nSELECT source_summary->>'projection_rows' FROM forecast_runs "
        "WHERE season = %s AND week = %s "
        "AND source_summary->>'kind' = 'forward_projections' AND source_summary->>'pipeline_run_id' = %s",
        (season, week, pending_id),
    )
    return [r[0] for r in cur.fetchall()]


def require_run_producer(cur, season: int, week: int, pending_id: str) -> None:
    """Refuse to promote a pending id whose forward producer never recorded itself for the week."""
    if not _recorded_rows(cur, season, week, pending_id):
        raise RunNotProducedError(
            f"no forward_projections forecast_runs row for {pending_id} (season={season} week={week}); "
            "this run's producer did not write, so nothing is promoted"
        )


def require_run_intact(cur, season: int, week: int, pending_id: str) -> int:
    """Promotion guard: the producer wrote for this run, and every row it wrote is still there.

    Runs inside the promotion transaction, before the relabel. Refuses when the week holds none of the run's
    rows, or a different number than the producer recorded (``projection_rows``): something deleted or
    rewrote them after step 7 (review N8). Nothing is written, so the archive of a replaced week stays.
    Returns the count.
    """
    recorded = _recorded_rows(cur, season, week, pending_id)
    if not recorded:
        require_run_producer(cur, season, week, pending_id)  # raises with the "no row" message
    expected = {None if r is None else int(r) for r in recorded}
    if None in expected or len(expected) != 1:
        raise RunNotProducedError(
            f"the forward_projections row for {pending_id} (season={season} week={week}) records no single "
            "row count to check the promotion against; nothing is promoted"
        )
    cur.execute(
        "-- pr:count\nSELECT count(*) FROM projections WHERE season = %s AND week = %s AND pipeline_run_id = %s",
        (season, week, pending_id),
    )
    found = int(cur.fetchone()[0])
    (want,) = expected
    if found == 0 or found != want:
        raise RunNotProducedError(
            f"{pending_id} has {found} projection rows for season={season} week={week}, but its producer wrote "
            f"{want}; the rows changed after step 7 (another refresh's cleanup?). Nothing is promoted and the "
            "archived week, if any, is kept."
        )
    return found


__all__ = [
    "LeftoverRunsError", "RunNotProducedError", "clear_leftovers", "new_pending_run_id", "pending_prefix",
    "require_run_intact", "require_run_producer",
]
