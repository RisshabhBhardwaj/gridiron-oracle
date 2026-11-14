"""Friday availability-only refresh of a forward week (decision 0012, addendum item 8).

Updates only ``projections.availability_status`` for the target week's forward rows, so what
is served follows the newest injury and roster reports without recomputing a prediction. It
writes in one transaction, records itself in ``forecast_runs`` under its own key
(``availability_refresh_<season>_w<WW>_<UTC stamp>``) and refuses unless:

* the week has games and at least one has not started (no score, kickoff after the new ``as_of``);
* forward projection rows exist for those games;
* the fresh snapshot has injury-report rows (a report or practice status), since roster rows alone
  would clear every designation.

Work is per game (coordinator answer 6): rows of a game that has started are frozen, and only rows of
games still to be played are updated, so a Thursday game never stops Friday's update of Sunday.
The open-games check runs twice: before the fetch (no pointless call) and after it, because the
new ``as_of`` is read after the fetch and the clock may have crossed a kickoff meanwhile.
Prediction columns, history rows and other weeks are never touched.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import psycopg2.extras

from pipeline.availability_store import capture_snapshot, load_availability

logger = logging.getLogger(__name__)


class AvailabilityRefreshRefused(RuntimeError):
    """A guard refused the refresh. Nothing was written."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _open_games(conn, season: int, week: int, as_of: datetime) -> tuple[list[str], int]:
    """(ids of games that have not started, number frozen) for the week at ``as_of``.

    A game is frozen when it has a score, a kickoff at or before ``as_of``, or no kickoff time (it
    cannot be shown not to have started). Refuses when the week has no games or none is open.
    """
    with conn.cursor() as cur:
        cur.execute(
            "-- av:games\nSELECT id, kickoff_at, home_score, away_score FROM games "
            "WHERE season = %s AND week = %s ORDER BY kickoff_at, id",
            (season, week),
        )
        games = cur.fetchall()
    if not games:
        raise AvailabilityRefreshRefused(f"no games for season={season} week={week}")
    open_ids = [
        str(g[0]) for g in games
        if g[2] is None and g[3] is None and g[1] is not None and pd.Timestamp(g[1]) > pd.Timestamp(as_of)
    ]
    if not open_ids:
        raise AvailabilityRefreshRefused(
            f"every game in week {week} has started at {as_of.isoformat()}; nothing left to update"
        )
    return open_ids, len(games) - len(open_ids)


def _forward_rows(conn, season: int, week: int, run_ids: list[str], game_ids: list[str]) -> dict[str, Optional[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "-- av:rows\nSELECT DISTINCT player_id, availability_status FROM projections "
            "WHERE season = %s AND week = %s AND pipeline_run_id = ANY(%s) AND game_id = ANY(%s)",
            (season, week, run_ids, game_ids),
        )
        rows = cur.fetchall()
    if not rows:
        raise AvailabilityRefreshRefused(
            f"no forward projection rows for season={season} week={week} under {run_ids} "
            "for games that have not started; run the weekly refresh first"
        )
    return {str(pid): status for pid, status in rows}


def resolve_target_week(conn, season: int, now: datetime) -> int:
    """The week of the next kickoff after ``now``: the forward week a Friday refresh is for.

    In a week with a Thursday game that week is still returned on Friday: its Thursday game is frozen
    and ``refresh_availability`` updates only the games that have not started.
    """
    with conn.cursor() as cur:
        cur.execute(
            "-- av:next_week\nSELECT week FROM games WHERE season = %s AND kickoff_at > %s "
            "ORDER BY kickoff_at LIMIT 1",
            (season, now),
        )
        row = cur.fetchone()
    if not row:
        raise AvailabilityRefreshRefused(f"no upcoming kickoff in season {season}; nothing to refresh")
    return int(row[0])


def _require_injury_reports(conn, snapshot_id: str) -> int:
    """Refuse a snapshot with no injury-report rows. By Friday every team has filed reports.

    ``capture_snapshot`` only refuses when injuries and rosters are both empty. Roster rows alone would
    map every Out/Doubtful player to no status, and the refresh would serve all of them.
    """
    with conn.cursor() as cur:
        cur.execute(
            "-- av:injury_rows\nSELECT count(*) FROM availability_reports WHERE snapshot_id = %s "
            "AND (report_status IS NOT NULL OR practice_status IS NOT NULL)",
            (snapshot_id,),
        )
        row = cur.fetchone()
    n = int(row[0]) if row else 0
    if n == 0:
        raise AvailabilityRefreshRefused(
            f"snapshot {snapshot_id} has no injury-report rows (no report or practice status); "
            "refusing to clear designations from roster rows alone. Nothing was updated"
        )
    return n


def default_run_ids(season: int) -> list[str]:
    """The ids a forward row can carry: approved by the manifest, or the pending id."""
    from ml.forward_serving import pending_run_id
    from pipeline.run_guard import approved_pipeline_run_ids

    return sorted(approved_pipeline_run_ids()) + [pending_run_id(season)]


def refresh_availability(
    conn,
    season: int,
    week: int,
    *,
    now: Callable[[], datetime] = _utcnow,
    capture: Callable = capture_snapshot,
    loader: Callable = load_availability,
    run_ids: Optional[list[str]] = None,
) -> dict:
    """Capture a fresh availability snapshot and update the week's status column. Returns a report."""
    run_ids = run_ids if run_ids is not None else default_run_ids(season)
    _open_games(conn, season, week, now())  # refuse before fetching when nothing can change
    snapshot = capture(conn, season, week)
    _require_injury_reports(conn, snapshot["snapshot_id"])
    as_of = now()
    if as_of.tzinfo is None:
        raise ValueError("now() must be timezone-aware")
    try:
        # The clock may have crossed a kickoff during the fetch: decide the open games at the new as_of.
        game_ids, frozen = _open_games(conn, season, week, as_of)
        current = _forward_rows(conn, season, week, run_ids, game_ids)
        result = loader(conn, season, week, sorted(current), as_of)
        changes = [
            {"player_id": pid, "status": result.statuses.get(pid), "season": season, "week": week,
             "run_ids": run_ids, "game_ids": game_ids}
            for pid in sorted(current)
            if result.statuses.get(pid) != current[pid]
        ]
        key = f"availability_refresh_{season}_w{week:02d}_{as_of.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        summary = {
            "kind": "availability_refresh", "as_of": as_of.isoformat(),
            "availability_snapshot_id": result.snapshot_id or snapshot.get("snapshot_id"),
            "players": len(current), "changed": len(changes), "counts": dict(result.counts),
            "games_updated": len(game_ids), "games_frozen": frozen,
        }
        with conn.cursor() as cur:
            if changes:
                psycopg2.extras.execute_batch(
                    cur,
                    "UPDATE projections SET availability_status = %(status)s "
                    "WHERE season = %(season)s AND week = %(week)s AND player_id = %(player_id)s "
                    "AND pipeline_run_id = ANY(%(run_ids)s) AND game_id = ANY(%(game_ids)s)",
                    changes,
                    page_size=500,
                )
            cur.execute(
                "INSERT INTO forecast_runs (pipeline_run_id, season, week, as_of, feature_rows, source_summary) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (key, season, week, as_of, len(changes), json.dumps(summary, default=str)),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    logger.info("availability refresh %s: %d of %d players changed", key, len(changes), len(current))
    return {
        "forecast_run_key": key, "season": season, "week": week, "as_of": as_of.isoformat(),
        "availability_snapshot_id": summary["availability_snapshot_id"],
        "players": len(current), "changed": len(changes), "unchanged": len(current) - len(changes),
        "games_updated": len(game_ids), "games_frozen": frozen,
    }


__all__ = ["AvailabilityRefreshRefused", "default_run_ids", "refresh_availability", "resolve_target_week"]
