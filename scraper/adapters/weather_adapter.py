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
import json
import logging
import os
import time
from datetime import date, datetime, time as dt_time, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

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

# ``gametime`` in nflverse schedules is local stadium time.  A stored UTC
# kickoff is required to prove that a forecast snapshot predates the game.
TEAM_TIMEZONES: dict[str, str] = {
    "ARI": "America/Phoenix", "ATL": "America/New_York", "BAL": "America/New_York",
    "BUF": "America/New_York", "CAR": "America/New_York", "CHI": "America/Chicago",
    "CIN": "America/New_York", "CLE": "America/New_York", "DAL": "America/Chicago",
    "DEN": "America/Denver", "DET": "America/New_York", "GB": "America/Chicago",
    "HOU": "America/Chicago", "IND": "America/Indiana/Indianapolis", "JAX": "America/New_York",
    "KC": "America/Chicago", "LV": "America/Los_Angeles", "LAC": "America/Los_Angeles",
    "LAR": "America/Los_Angeles", "MIA": "America/New_York", "MIN": "America/Chicago",
    "NE": "America/New_York", "NO": "America/Chicago", "NYG": "America/New_York",
    "NYJ": "America/New_York", "PHI": "America/New_York", "PIT": "America/New_York",
    "SEA": "America/Los_Angeles", "SF": "America/Los_Angeles", "TB": "America/New_York",
    "TEN": "America/Chicago", "WAS": "America/New_York",
    # Historical/alternate abbreviations nflverse's schedules feed still uses
    # for some seasons (Rams as "LA" rather than "LAR"; Raiders' Oakland era).
    "LA": "America/Los_Angeles", "OAK": "America/Los_Angeles",
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


def _find_coords(stadium: str) -> Optional[tuple[float, float]]:
    if not stadium:
        return None
    stadium = stadium.strip()
    if stadium in STADIUM_COORDS:
        return STADIUM_COORDS[stadium]
    return next((coords for name, coords in STADIUM_COORDS.items() if stadium in name or name in stadium), None)


def _kickoff_at(gameday: date, gametime: Optional[str], home_team: Optional[str]) -> Optional[datetime]:
    """Convert nflverse's local stadium ``gameday``/``gametime`` to UTC."""
    if not gameday or not gametime or not home_team or home_team not in TEAM_TIMEZONES:
        return None
    try:
        local_time = dt_time.fromisoformat(str(gametime))
    except ValueError:
        return None
    return datetime.combine(gameday, local_time, ZoneInfo(TEAM_TIMEZONES[home_team])).astimezone(timezone.utc)


def fetch_forecast_snapshot(
    api_key: str, stadium: str, kickoff_at: datetime, roof: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Fetch one OpenWeather 5-day forecast record nearest to a future kickoff."""
    roof_lower = (roof or "").lower()
    if any(value in roof_lower for value in ("dome", "closed")):
        return {"forecast_for": kickoff_at, "temp_f": None, "wind_mph": 0.0, "precipitation_bucket": 0, "raw_payload": {"roof": roof}}
    coords = _find_coords(stadium)
    if not coords:
        return None
    if kickoff_at <= datetime.now(timezone.utc):
        return None
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; skipping OpenWeather forecast capture")
        return None
    lat, lon = coords
    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    time.sleep(1.05)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("list") or []
        selected = min(
            choices,
            key=lambda item: abs(datetime.fromtimestamp(item["dt"], timezone.utc) - kickoff_at),
        )
    except (Exception, ValueError, KeyError) as exc:
        logger.warning("OpenWeather forecast fetch failed for %s: %s", stadium, exc)
        return None
    main = selected.get("main") or {}
    wind = selected.get("wind") or {}
    weather_main = ((selected.get("weather") or [{}])[0]).get("main")
    return {
        "forecast_for": datetime.fromtimestamp(selected["dt"], timezone.utc),
        "temp_f": _celsius_to_fahrenheit(main.get("temp")),
        "wind_mph": _meters_per_second_to_mph(wind.get("speed")),
        "precipitation_bucket": _precipitation_bucket_from_weather(weather_main, selected.get("pop")),
        "raw_payload": selected,
    }


def capture_pregame_weather_forecasts(db_url: str, api_key: Optional[str] = None) -> int:
    """Persist real forecast snapshots for upcoming games; never backfill history."""
    api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        logger.info("OPENWEATHER_API_KEY not set; skipping pregame forecast capture")
        return 0
    import psycopg2

    captured_at = datetime.now(timezone.utc)
    with psycopg2.connect(_dsn_for_psycopg2(db_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, stadium, gameday, gametime, home_team, roof
                FROM games
                WHERE gameday BETWEEN CURRENT_DATE AND CURRENT_DATE + 5
                  AND home_score IS NULL AND stadium IS NOT NULL
                ORDER BY gameday, gametime
                """
            )
            games = cur.fetchall()
        persisted = 0
        for game_id, stadium, gameday, gametime, home_team, roof in games:
            kickoff = _kickoff_at(gameday, gametime, home_team)
            if kickoff is None or kickoff <= captured_at:
                logger.warning("Skipping %s: no future timezone-aware kickoff", game_id)
                continue
            forecast = fetch_forecast_snapshot(api_key, stadium, kickoff, roof)
            if forecast is None:
                continue
            with conn.cursor() as cur:
                cur.execute("UPDATE games SET kickoff_at = %s WHERE id = %s", (kickoff, game_id))
                cur.execute(
                    """
                    INSERT INTO weather_forecasts
                      (game_id, provider, kickoff_at, forecast_for, temp_f, wind_mph,
                       precipitation_bucket, captured_at, raw_payload)
                    VALUES (%s, 'openweather_5day', %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id, provider, captured_at) DO NOTHING
                    """,
                    (game_id, kickoff, forecast["forecast_for"], forecast["temp_f"],
                     forecast["wind_mph"], forecast["precipitation_bucket"], captured_at,
                     json.dumps(forecast["raw_payload"])),
                )
            persisted += 1
    logger.info("Captured %d pregame weather forecasts", persisted)
    return persisted


def _celsius_to_fahrenheit(value: Any) -> Optional[float]:
    try:
        return float(value) * 9 / 5 + 32
    except (TypeError, ValueError):
        return None


def _meters_per_second_to_mph(value: Any) -> Optional[float]:
    try:
        return float(value) * 2.236936
    except (TypeError, ValueError):
        return None


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

    coords = _find_coords(stadium)
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
    parser.add_argument("--capture-forecasts", action="store_true", help="store pre-kickoff forecast snapshots for the next five days")
    args = parser.parse_args()
    if not args.db_url:
        parser.error("--db-url or DATABASE_URL required")
    if args.capture_forecasts:
        capture_pregame_weather_forecasts(args.db_url)
    else:
        enrich_games_precipitation(args.db_url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
