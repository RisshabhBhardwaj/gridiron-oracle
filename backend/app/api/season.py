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
        cur.execute(
            "SELECT season, MAX(week) FROM projections "
            "GROUP BY season ORDER BY season DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return {"season": int(row[0]), "week": int(row[1])}
    except Exception:
        pass

    return {"season": datetime.now().year, "week": 1}
