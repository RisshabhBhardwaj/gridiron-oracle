"""
scraper/adapters/weather_adapter.py

OpenWeatherMap adapter for precipitation_bucket on outdoor games.
Populates games.precipitation_bucket: 0=none, 1=rain, 2=snow.

Uses free tier (1000 req/day). For dome games, precipitation_bucket=0.
For outdoor games: fetches current/forecast weather by stadium coords.

Standalone usage:
  python scraper/adapters/weather_adapter.py --db-url $DATABASE_URL

CLAUDE.md §6: OPENWEATHER_API_KEY in .env. Rate limit: 1s between requests.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# NFL stadium → (lat, lon) for OpenWeatherMap geocoding
# Coordinates for primary stadiums (nflverse stadium names may vary)
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Arrowhead Stadium": (39.0489, -94.4839),
    "AT&T Stadium": (32.7473, -97.0945),
    "Bank of America Stadium": (35.2258, -80.8528),
    "Caesars Superdome": (29.9511, -90.0811),
    "Empower Field at Mile High": (39.7439, -105.0201),
    "FedExField": (38.9076, -76.8644),
    "FirstEnergy Stadium": (41.5061, -81.6996),
    "Ford Field": (42.3400, -83.0456),
    "Gillette Stadium": (42.0909, -71.2644),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "Lambeau Field": (44.5013, -88.0622),
    "Levi's Stadium": (37.4032, -121.9704),
    "Lincoln Financial Field": (39.9007, -75.1675),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "Lumen Field": (47.5952, -122.3316),
    "M&T Bank Stadium": (39.2779, -76.6226),
    "Mercedes-Benz Stadium": (33.7554, -84.4008),
    "MetLife Stadium": (40.8128, -74.0742),
    "Nissan Stadium": (36.1665, -86.7713),
    "Paycor Stadium": (39.0954, -84.5160),
    "Raymond James Stadium": (27.9759, -82.5034),
    "SoFi Stadium": (33.9535, -118.3392),
    "Soldier Field": (41.8623, -87.6167),
    "State Farm Stadium": (33.5276, -112.2626),
    "TIAA Bank Field": (30.3239, -81.6372),
    "U.S. Bank Stadium": (44.9723, -93.2581),
    "Acrisure Stadium": (40.4468, -80.0158),
    "Highmark Stadium": (42.7738, -78.7870),
    "Allegiant Stadium": (36.0908, -115.1836),
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839),
}


def _precipitation_bucket_from_weather(weather_main: Optional[str], pop: Optional[float]) -> int:
    """
    Map OpenWeatherMap response to precipitation_bucket.
    0 = none, 1 = rain, 2 = snow/sleet.
    """
    if not weather_main:
        return 0
    w = weather_main.lower()
    if "snow" in w or "sleet" in w:
        return 2
    if "rain" in w or "drizzle" in w or "thunderstorm" in w:
        return 1
    if pop is not None and pop > 0.5:
        return 1  # high precip prob = treat as rain
    return 0


def fetch_precipitation_bucket(
    api_key: str,
    stadium: str,
    gameday: date,
    roof: Optional[str] = None,
) -> Optional[int]:
    """
    Fetch precipitation bucket for a game.
    Returns 0 for dome/closed roofs; otherwise calls OpenWeatherMap.
    """
    roof_lower = (roof or "").lower()
    if any(r in roof_lower for r in ("dome", "closed", "retractable")):
        return 0

    def _find_coords(s: str) -> Optional[tuple[float, float]]:
        if not s:
            return None
        s = s.strip()
        if s in STADIUM_COORDS:
            return STADIUM_COORDS[s]
        for name, c in STADIUM_COORDS.items():
            if s in name or name in s:
                return c
        return None

    coords = _find_coords(stadium) if stadium else None
    if not coords:
        return None

    lat, lon = coords
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping OpenWeatherMap fetch")
        return None

    # Use current weather for today; historical requires paid API
    today = date.today()
    if gameday == today:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
    elif gameday > today:
        # Forecast API (5-day, 3hr intervals)
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
    else:
        # Past: skip (historical API is paid; One Call historical = 5 days only)
        logger.debug("Historical weather for %s requires paid API; skipping", gameday)
        return None

    time.sleep(1.05)  # rate limit: 1 req/sec for free tier
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("OpenWeatherMap fetch failed for %s: %s", stadium, exc)
        return None

    if gameday == today:
        weather_main = (data.get("weather") or [{}])[0].get("main")
        pop = None
    else:
        # Forecast: find closest forecast to gameday noon
        forecasts = data.get("list") or []
        weather_main = None
        pop = None
        for f in forecasts[:16]:
            dt_str = f.get("dt_txt", "")
            if str(gameday) in dt_str:
                weather_main = (f.get("weather") or [{}])[0].get("main")
                pop = f.get("pop")
                break

    return _precipitation_bucket_from_weather(weather_main, pop)


def _dsn_for_psycopg2(url: str) -> str:
    """Convert SQLAlchemy-style URL to psycopg2-compatible DSN."""
    s = url.replace("postgresql+asyncpg://", "postgresql://")
    s = s.replace("postgresql+psycopg2://", "postgresql://")
    if "://" not in s and s:
        s = f"postgresql://{s}"
    return s


def enrich_games_precipitation(db_url: str, api_key: Optional[str] = None) -> int:
    """
    Update games.precipitation_bucket for outdoor games missing it.
    Returns count of games updated.
    """
    api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        logger.info("OPENWEATHER_API_KEY not set; skipping precipitation enrichment")
        return 0

    import psycopg2

    conn = psycopg2.connect(_dsn_for_psycopg2(db_url))

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, stadium, gameday, roof
            FROM games
            WHERE precipitation_bucket IS NULL
              AND gameday IS NOT NULL
              AND stadium IS NOT NULL
            ORDER BY gameday DESC
            LIMIT 500
            """
        )
        rows = cur.fetchall()

    updated = 0
    for game_id, stadium, gameday, roof in rows:
        if not stadium or not gameday:
            continue
        bucket = fetch_precipitation_bucket(api_key, stadium, gameday, roof)
        if bucket is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE games SET precipitation_bucket = %s WHERE id = %s",
                    (bucket, game_id),
                )
            updated += 1

    conn.commit()
    conn.close()
    if updated:
        logger.info("Updated precipitation_bucket for %d games", updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich games with precipitation from OpenWeatherMap")
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()
    if not args.db_url:
        parser.error("--db-url or DATABASE_URL required")
    enrich_games_precipitation(args.db_url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
