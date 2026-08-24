"""
backend/app/api/season.py

/season/current endpoint — returns the most recent (season, week) for which
projections exist.  Used by the frontend to initialise the week/season selectors
without hardcoding CURRENT_WEEK in the TypeScript source.

Falls back to {season: current_calendar_year, week: 1} if the DB is unreachable
or has no projections yet.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from backend.app.core.config import settings

router = APIRouter(prefix="", tags=["meta"])


@router.get("/season/current")
def current_season() -> dict:
    """
    Return the most recent season + week from the projections table.

    Used by the frontend useCurrentSeason hook to avoid hardcoding CURRENT_WEEK.
    Response: { "season": int, "week": int }
    """
    try:
        import psycopg2

        conn = psycopg2.connect(settings.database_url)
        cur = conn.cursor()
        # Two different questions, deliberately answered from two sources.
        #
        # *Which season* is the latest one we hold data for: the union of both
        # projection surfaces, because 2026 lives almost entirely in
        # season_simulation_weeks and only marginally in projections.
        cur.execute(
            "SELECT season FROM ("
            "  SELECT season FROM projections"
            "  UNION ALL"
            "  SELECT season FROM season_simulation_weeks"
            ") s GROUP BY season ORDER BY season DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return {"season": datetime.now().year, "week": 1}
        season = int(row[0])

        # *Which week* it currently is: the schedule, never MAX(week) over the
        # projection tables. Those tables hold all 18 weeks the moment a season
        # is materialised, so MAX(week) reported week 18 in August — which the
        # frontend then used as every selector's default. Same derivation as
        # scripts/weekly_refresh.py:resolve_season_and_week.
        cur.execute(
            "SELECT COALESCE(MAX(week), 0) FROM games "
            "WHERE season = %s AND kickoff_at < NOW()",
            (season,),
        )
        played = cur.fetchone()
        conn.close()
        completed_week = int(played[0]) if played and played[0] is not None else 0
        week = min(max(completed_week + 1, 1), 18)
        return {"season": season, "week": week}
    except Exception:
        pass

    return {"season": datetime.now().year, "week": 1}
