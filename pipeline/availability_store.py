"""Availability snapshots: fetch, store with the fetch time, read back as of a moment.

The source is nflverse via ``nflreadpy``: ``load_injuries`` (weekly ``report_status``)
and ``load_rosters_weekly`` (``status``: RES, INA, SUS ...). Neither file carries a
per-row publication time (checked 2026-11-14), so every fetch is stored whole with
the time it finished. Nothing in a snapshot can have been published after that time,
so a producer run with ``as_of`` at or after it may use it, and an earlier run may not.
Tables are created by ``alembic/versions/20261114_0028_availability_status.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import psycopg2.extras

from pipeline.availability import (
    AvailabilityResult,
    normalize_roster_status,
    resolve_statuses,
    select_snapshot,
)

logger = logging.getLogger(__name__)

SOURCE = "nflverse:load_injuries+load_rosters_weekly"
# Roster codes that say nothing about a game's availability are not stored.
_UNINFORMATIVE_ROSTER = {"ACT", "DEV"}


class AvailabilityFetchError(RuntimeError):
    """The source could not be read, or returned nothing for the week."""


class NoAvailabilitySnapshot(RuntimeError):
    """No snapshot was fetched at or before ``as_of``; availability is unknown, not clear."""


def fetch_source_frames(season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injury reports and weekly-roster rows for one season, as pandas frames (network)."""
    import nflreadpy

    injuries = nflreadpy.load_injuries([season])
    rosters = nflreadpy.load_rosters_weekly([season])
    to_pandas = lambda df: df.to_pandas() if hasattr(df, "to_pandas") else df  # noqa: E731
    return to_pandas(injuries), to_pandas(rosters)


def _week_rows(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    mask = pd.Series(True, index=frame.index)
    if "season" in frame.columns:
        mask &= frame["season"].astype("Int64") == season
    if "week" in frame.columns:
        mask &= frame["week"].astype("Int64") == week
    return frame.loc[mask]


def _clean(value):
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


def build_report_rows(injuries: pd.DataFrame, rosters: pd.DataFrame, season: int, week: int) -> list[dict]:
    """One row per player carrying anything informative for the target week."""
    rows: dict[str, dict] = {}
    inj = _week_rows(injuries, season, week)
    for r in inj.itertuples(index=False):
        pid = _clean(getattr(r, "gsis_id", None))
        if not pid:
            continue
        rows[str(pid)] = {
            "player_id": str(pid),
            "team": _clean(getattr(r, "team", None)),
            "report_status": _clean(getattr(r, "report_status", None)),
            "practice_status": _clean(getattr(r, "practice_status", None)),
            "roster_status": None,
            "date_modified": _clean(getattr(r, "date_modified", None)),
        }
    ros = _week_rows(rosters, season, week)
    for r in ros.itertuples(index=False):
        pid = _clean(getattr(r, "gsis_id", None))
        status = _clean(getattr(r, "status", None))
        if not pid or status is None or str(status).upper() in _UNINFORMATIVE_ROSTER:
            continue
        row = rows.setdefault(str(pid), {
            "player_id": str(pid), "team": _clean(getattr(r, "team", None)),
            "report_status": None, "practice_status": None, "roster_status": None,
            "date_modified": None,
        })
        row["roster_status"] = str(status)
    return list(rows.values())


def capture_snapshot(
    conn,
    season: int,
    week: int,
    *,
    fetch: Callable[[int, int], tuple[pd.DataFrame, pd.DataFrame]] = fetch_source_frames,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict:
    """Fetch the source and store it as one snapshot, in one transaction.

    ``fetched_at`` is read after the fetch returns, so it is an upper bound on when
    anything in the snapshot was published. A failed or empty fetch writes nothing.
    """
    try:
        injuries, rosters = fetch(season, week)
    except Exception as exc:
        raise AvailabilityFetchError(f"availability source unavailable: {exc}") from exc
    fetched_at = now()
    if fetched_at.tzinfo is None:
        raise ValueError("now() must be timezone-aware")
    rows = build_report_rows(injuries, rosters, season, week)
    if not rows:
        raise AvailabilityFetchError(
            f"no injury or roster rows for season={season} week={week}; "
            "an empty source is not an all-clear"
        )
    snapshot_id = f"avail_{season}_w{week:02d}_{fetched_at.strftime('%Y%m%dT%H%M%SZ')}"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO availability_snapshots "
                "(snapshot_id, season, week, fetched_at, source, n_reports) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (snapshot_id, season, week, fetched_at, SOURCE, len(rows)),
            )
            psycopg2.extras.execute_batch(
                cur,
                "INSERT INTO availability_reports "
                "(snapshot_id, player_id, team, report_status, practice_status, roster_status, date_modified) "
                "VALUES (%(snapshot_id)s, %(player_id)s, %(team)s, %(report_status)s, "
                "%(practice_status)s, %(roster_status)s, %(date_modified)s)",
                [{**r, "snapshot_id": snapshot_id} for r in rows],
                page_size=500,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    logger.info("availability snapshot %s stored: %d reports", snapshot_id, len(rows))
    return {"snapshot_id": snapshot_id, "fetched_at": fetched_at, "n_reports": len(rows), "source": SOURCE}


def load_availability(
    conn,
    season: int,
    week: int,
    player_ids: list[str],
    as_of: datetime,
    *,
    snapshot_id: Optional[str] = None,
) -> AvailabilityResult:
    """Statuses for ``player_ids`` from the newest snapshot known at ``as_of``.

    Raises ``NoAvailabilitySnapshot`` when none was fetched by then: unknown availability
    is not the same as everyone being healthy.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT snapshot_id, fetched_at FROM availability_snapshots "
            "WHERE season = %s AND week = %s",
            (season, week),
        )
        snapshots = [{"snapshot_id": r[0], "fetched_at": r[1]} for r in cur.fetchall()]
    chosen = select_snapshot(snapshots, as_of)
    if snapshot_id is not None:
        chosen = next((s for s in snapshots if s["snapshot_id"] == snapshot_id and s["fetched_at"] <= as_of), None)
    if chosen is None:
        raise NoAvailabilitySnapshot(
            f"no availability snapshot for season={season} week={week} was fetched at or before "
            f"{as_of.isoformat()}; availability is unknown"
        )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT player_id, report_status, practice_status, roster_status, date_modified "
            "FROM availability_reports WHERE snapshot_id = %s",
            (chosen["snapshot_id"],),
        )
        columns = ["player_id", "report_status", "practice_status", "roster_status", "date_modified"]
        reports = pd.DataFrame(cur.fetchall(), columns=columns)
    result = resolve_statuses(player_ids, reports, as_of, snapshot_fetched_at=chosen["fetched_at"])
    result.snapshot_id = chosen["snapshot_id"]
    logger.info(
        "availability %s: newest report %s old, %d reports used, %d ignored after as_of, statuses %s",
        chosen["snapshot_id"], result.newest_report_age, result.n_reports_used,
        result.ignored_after_as_of, result.counts,
    )
    return result


__all__ = [
    "AvailabilityFetchError",
    "NoAvailabilitySnapshot",
    "build_report_rows",
    "capture_snapshot",
    "fetch_source_frames",
    "load_availability",
    "normalize_roster_status",
]
