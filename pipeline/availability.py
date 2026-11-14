"""Who is expected to play in a forward week (decision 0012 D3 and the 2026-11-14 addendum).

The forward producer computes and stores a row for every rostered depth-chart
player, with an ``availability_status``. The API and the weekly acceptance check
serve only rows whose status is not excluded. This module holds the rules, with
no database and no heavy imports, so the API, the producer and the Friday refresh
share one definition.

Stored values (``projections.availability_status``; NULL means "no report against
this player", which counts as expected to play):

    Questionable, Doubtful   served, status shown next to the projection
    Out, Reserve, Suspended, Inactive   stored, never served

Causality: a status comes only from a report known at or before ``as_of``. The
nflverse injury file (``nflreadpy.load_injuries``) has no per-row timestamp (the
columns were checked on 2026-11-14), so each fetch is stored as a snapshot with
its fetch time, which bounds when every row in it was published. A row's own
``date_modified``, when a source supplies one, is honoured as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

import pandas as pd

OUT = "Out"
DOUBTFUL = "Doubtful"
QUESTIONABLE = "Questionable"
RESERVE = "Reserve"
SUSPENDED = "Suspended"
INACTIVE = "Inactive"

STORED_STATUSES: tuple[str, ...] = (QUESTIONABLE, DOUBTFUL, OUT, RESERVE, SUSPENDED, INACTIVE)
EXCLUDED_STATUSES: frozenset[str] = frozenset({OUT, RESERVE, SUSPENDED, INACTIVE})

# Most restrictive first: when two sources disagree, the player is treated as the
# less available of the two.
_SEVERITY: tuple[str, ...] = (SUSPENDED, RESERVE, OUT, INACTIVE, DOUBTFUL, QUESTIONABLE)

_REPORT_STATUS = {"out": OUT, "doubtful": DOUBTFUL, "questionable": QUESTIONABLE}
# nflverse weekly-roster ``status`` codes. ACT, DEV (practice squad), TRD and TRC say
# nothing about a game's availability, so they map to no status. CUT, RET and EXE mean
# the player is not on the active roster.
_ROSTER_STATUS = {
    "RES": RESERVE,
    "INA": INACTIVE,
    "SUS": SUSPENDED,
    "CUT": INACTIVE,
    "RET": INACTIVE,
    "EXE": INACTIVE,
}


def normalize_report_status(value: Any) -> Optional[str]:
    """Out / Doubtful / Questionable from an injury report; anything else is no designation."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return _REPORT_STATUS.get(str(value).strip().lower())


def normalize_roster_status(value: Any) -> Optional[str]:
    """Reserve / Inactive / Suspended from a weekly-roster status code."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return _ROSTER_STATUS.get(str(value).strip().upper())


def is_served(status: Optional[str]) -> bool:
    """True when a row with this status is shown to users."""
    return status is None or status not in EXCLUDED_STATUSES


def served_predicate(column: str = "availability_status") -> str:
    """SQL condition that keeps only expected-to-play rows."""
    excluded = ", ".join(f"'{s}'" for s in sorted(EXCLUDED_STATUSES))
    return f"({column} IS NULL OR {column} NOT IN ({excluded}))"


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")


def select_snapshot(snapshots: Iterable[dict], as_of: datetime) -> Optional[dict]:
    """The newest snapshot fetched at or before ``as_of``; later fetches are not yet known."""
    _require_aware(as_of, "as_of")
    known = [s for s in snapshots if s["fetched_at"] <= as_of]
    return max(known, key=lambda s: s["fetched_at"]) if known else None


@dataclass
class AvailabilityResult:
    """Statuses for a set of players, and what they were derived from."""

    statuses: dict[str, Optional[str]]
    as_of: datetime
    n_reports_used: int = 0
    ignored_after_as_of: int = 0
    newest_report_at: Optional[datetime] = None
    newest_report_age: Optional[timedelta] = None
    snapshot_fetched_at: Optional[datetime] = None
    snapshot_id: Optional[str] = None
    counts: dict[str, int] = field(default_factory=dict)


def _most_restrictive(candidates: list[str]) -> Optional[str]:
    for status in _SEVERITY:
        if status in candidates:
            return status
    return None


def resolve_statuses(
    player_ids: Iterable[str],
    reports: pd.DataFrame,
    as_of: datetime,
    *,
    snapshot_fetched_at: Optional[datetime] = None,
) -> AvailabilityResult:
    """Status per player from reports known at or before ``as_of``.

    ``reports`` has ``player_id``, ``report_status``, ``roster_status`` and an optional
    ``date_modified``. A row dated after ``as_of`` is ignored. A row with no date is
    bounded by ``snapshot_fetched_at`` (the caller selected a snapshot fetched at or
    before ``as_of``). A player with no usable report maps to None, never to a guess.
    """
    _require_aware(as_of, "as_of")
    ids = [str(p) for p in player_ids]
    frame = reports.copy() if reports is not None else pd.DataFrame()
    for col in ("player_id", "report_status", "roster_status", "date_modified"):
        if col not in frame.columns:
            frame[col] = None
    frame["player_id"] = frame["player_id"].astype(str)

    stamps = pd.to_datetime(frame["date_modified"], utc=True, errors="coerce")
    after = stamps.notna() & (stamps > pd.Timestamp(as_of))
    ignored = int(after.sum())
    usable = frame.loc[~after].copy()
    usable_stamps = stamps.loc[~after]

    by_player: dict[str, list[str]] = {}
    for row in usable.itertuples(index=False):
        found = [
            s for s in (
                normalize_report_status(row.report_status),
                normalize_roster_status(row.roster_status),
            ) if s is not None
        ]
        if found:
            by_player.setdefault(row.player_id, []).extend(found)

    statuses = {pid: _most_restrictive(by_player.get(pid, [])) for pid in ids}

    newest: Optional[datetime] = None
    if usable_stamps.notna().any():
        newest = usable_stamps.max().to_pydatetime()
    elif snapshot_fetched_at is not None and len(usable):
        newest = snapshot_fetched_at
    counts: dict[str, int] = {}
    for status in statuses.values():
        if status is not None:
            counts[status] = counts.get(status, 0) + 1
    return AvailabilityResult(
        statuses=statuses,
        as_of=as_of,
        n_reports_used=len(usable),
        ignored_after_as_of=ignored,
        newest_report_at=newest,
        newest_report_age=(as_of - newest) if newest is not None else None,
        snapshot_fetched_at=snapshot_fetched_at,
        counts=counts,
    )
