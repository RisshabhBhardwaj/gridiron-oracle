"""Availability snapshots: fetch, store with the fetch time, read back as of a moment.

The source is nflverse via ``nflreadpy``: ``load_injuries`` (weekly ``report_status``)
and ``load_rosters_weekly`` (``status``: RES, INA, SUS ...). The source has no trustworthy
per-row publication time, so every fetch is stored whole with the time it finished.
``validate_source`` refuses a file whose columns, labels, ids or dates do not look like nflverse's
(required columns, label allow-lists, gsis ids, ``date_modified <= fetched_at``) before anything is stored. Nothing in a snapshot can have been published after that time,
so a producer run with ``as_of`` at or after it may use it, and an earlier run may not.
Tables are created by ``alembic/versions/20261114_0028_availability_status.py``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import psycopg2.extras

from pipeline.availability import (
    _SEVERITY,
    AvailabilityResult,
    normalize_report_status,
    normalize_roster_status,
    resolve_statuses,
    select_snapshot,
)

logger = logging.getLogger(__name__)

SOURCE = "nflverse:load_injuries+load_rosters_weekly"
# Roster codes that say nothing about a game's availability are not stored.
_UNINFORMATIVE_ROSTER = {"ACT", "DEV"}

# What a source file must look like before it is stored (security review M2). A file that does not
# match is refused whole: the old parser mapped every unknown value to "no designation", which is
# served, so schema drift or a bad file silently cleared Out and Reserve players.
# The label sets below are the known nflverse values; an unknown label is refused and logged, so the
# first dry-run capture shows what to add (extend these constants, never loosen the check).
REQUIRED_INJURY_COLUMNS = ("gsis_id", "week", "season", "report_status", "date_modified")
REQUIRED_ROSTER_COLUMNS = ("gsis_id", "status", "week", "season")
KNOWN_REPORT_STATUSES = frozenset({"out", "doubtful", "questionable", "probable"})
KNOWN_PRACTICE_STATUSES = frozenset({
    "full participation in practice", "limited participation in practice", "did not participate in practice",
    "full", "limited", "did not participate", "dnp",
})
KNOWN_ROSTER_CODES = frozenset({"ACT", "DEV", "TRC", "TRD", "RES", "INA", "SUS", "CUT", "RET", "EXE"})
GSIS_ID = re.compile(r"^00-\d{7}$")
MAX_REPORT_ROWS = 5000  # a week has a few hundred designations; this bounds what a bad file can store


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


def _report_rank(row: dict) -> tuple:
    """Lower is more restrictive. Order-independent: ties break on the stored text."""
    status = normalize_report_status(row.get("report_status"))
    rank = _SEVERITY.index(status) if status in _SEVERITY else len(_SEVERITY)
    return (rank, str(row.get("practice_status")), str(row.get("date_modified")))


def _roster_rank(code) -> tuple:
    status = normalize_roster_status(code)
    return (_SEVERITY.index(status) if status in _SEVERITY else len(_SEVERITY), str(code))


def build_report_rows(
    injuries: pd.DataFrame, rosters: pd.DataFrame, season: int, week: int, *, stats: Optional[dict] = None
) -> list[dict]:
    """One row per player carrying anything informative for the target week.

    A player with several rows keeps the most restrictive one, whatever the order of the file.
    ``stats["duplicates"]`` counts the extra rows that were merged away.
    """
    rows: dict[str, dict] = {}
    duplicates = 0
    inj = _week_rows(injuries, season, week)
    for r in inj.itertuples(index=False):
        pid = _clean(getattr(r, "gsis_id", None))
        if not pid:
            continue
        row = {
            "player_id": str(pid),
            "team": _clean(getattr(r, "team", None)),
            "report_status": _clean(getattr(r, "report_status", None)),
            "practice_status": _clean(getattr(r, "practice_status", None)),
            "roster_status": None,
            "date_modified": _clean(getattr(r, "date_modified", None)),
        }
        if str(pid) in rows:
            duplicates += 1
            if _report_rank(row) >= _report_rank(rows[str(pid)]):
                continue
        rows[str(pid)] = row
    ros = _week_rows(rosters, season, week)
    seen_roster: set[str] = set()
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
        if str(pid) in seen_roster:
            duplicates += 1
            if _roster_rank(str(status)) >= _roster_rank(row["roster_status"]):
                continue
        seen_roster.add(str(pid))
        row["roster_status"] = str(status)
    if stats is not None:
        stats["duplicates"] = duplicates
    return list(rows.values())


def _refuse(message: str) -> AvailabilityFetchError:
    logger.warning("availability source refused: %s", message)
    return AvailabilityFetchError(f"availability source refused: {message}")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    if frame is None or (frame.empty and len(frame.columns) == 0):
        return  # nothing came back: capture_snapshot reports it as an empty source
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise _refuse(f"{name} frame is missing required column(s) {missing}")


def _check_labels(frame: pd.DataFrame, column: str, known: frozenset, name: str, *, upper: bool = False) -> None:
    if frame.empty or column not in frame.columns:
        return
    unknown: set[str] = set()
    for value in frame[column]:
        if _clean(value) is None:
            continue
        text = str(value).strip()
        if (text.upper() if upper else text.lower()) not in known:
            unknown.add(text)
    if unknown:
        raise _refuse(f"unknown {name} label(s) {sorted(unknown)[:10]}; not treating them as served")


def _check_ids(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        return
    bad = sorted({str(v) for v in frame["gsis_id"] if _clean(v) is not None and not GSIS_ID.match(str(v))})
    if bad:
        raise _refuse(f"{name} has id(s) not in gsis format 00-NNNNNNN: {bad[:10]}")


def _check_dates(frame: pd.DataFrame, fetched_at: datetime) -> None:
    if frame.empty or "date_modified" not in frame.columns:
        return
    raw = frame["date_modified"]
    stamps = pd.to_datetime(raw, utc=True, errors="coerce")
    unreadable = raw.notna() & stamps.isna()
    if unreadable.any():
        raise _refuse(f"{int(unreadable.sum())} injury row(s) have an unreadable date_modified")
    late = stamps.notna() & (stamps > pd.Timestamp(fetched_at))
    if late.any():
        raise _refuse(
            f"{int(late.sum())} injury row(s) have date_modified after the fetch time {fetched_at.isoformat()}; "
            "a report cannot be published after it was fetched"
        )


def validate_source(
    injuries: pd.DataFrame, rosters: pd.DataFrame, season: int, week: int, fetched_at: datetime
) -> None:
    """Refuse a source file that does not look like nflverse's, before anything is stored."""
    _require_columns(injuries, REQUIRED_INJURY_COLUMNS, "injuries")
    _require_columns(rosters, REQUIRED_ROSTER_COLUMNS, "rosters")
    inj, ros = _week_rows(injuries, season, week), _week_rows(rosters, season, week)
    _check_labels(inj, "report_status", KNOWN_REPORT_STATUSES, "report_status")
    _check_labels(inj, "practice_status", KNOWN_PRACTICE_STATUSES, "practice_status")
    _check_labels(ros, "status", KNOWN_ROSTER_CODES, "roster status", upper=True)
    _check_ids(inj, "injuries")
    _check_ids(ros, "rosters")
    _check_dates(inj, fetched_at)


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
    validate_source(injuries, rosters, season, week, fetched_at)
    stats: dict = {}
    rows = build_report_rows(injuries, rosters, season, week, stats=stats)
    if not rows:
        raise AvailabilityFetchError(
            f"no injury or roster rows for season={season} week={week}; "
            "an empty source is not an all-clear"
        )
    if len(rows) > MAX_REPORT_ROWS:
        raise _refuse(f"{len(rows)} rows for one week exceeds the cap of {MAX_REPORT_ROWS}")
    logger.info("availability source for season=%s week=%s: %d rows, %d duplicate row(s) merged to the most "
                "restrictive", season, week, len(rows), stats["duplicates"])
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
