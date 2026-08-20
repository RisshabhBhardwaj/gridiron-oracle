"""Open-Meteo previous-runs / forecast capture into weather_forecasts.

Free, no API key. Provider tag is ``open_meteo``. Does not replace OpenWeather
when a key is present; it fills the historically empty pregame table.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import psycopg2

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn
from scraper.adapters.weather_adapter import STADIUM_COORDS, _find_coords, _kickoff_at

logger = logging.getLogger(__name__)
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
PROVIDER = "open_meteo"


def forecast_url(lat: float, lon: float) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }
    return f"{OPEN_METEO}?{urlencode(params)}"


def fetch_forecast(lat: float, lon: float, timeout_s: float = 20.0) -> dict[str, Any]:
    request = Request(forecast_url(lat, lon), headers={"User-Agent": "gridiron-oracle/open-meteo"})
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def snapshot_from_payload(payload: dict[str, Any], kickoff_at: datetime) -> dict[str, Any]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    precips = hourly.get("precipitation") or []
    target = kickoff_at.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    stamp = target.strftime("%Y-%m-%dT%H:00")
    index = times.index(stamp) if stamp in times else 0
    temp = temps[index] if index < len(temps) else None
    wind = winds[index] if index < len(winds) else None
    precip = precips[index] if index < len(precips) else 0.0
    bucket = 0
    if precip and float(precip) > 0.05:
        bucket = 1 if (temp is None or float(temp) > 32) else 2
    return {
        "temp_f": None if temp is None else float(temp),
        "wind_mph": None if wind is None else float(wind),
        "precipitation_bucket": bucket,
        "raw_payload": payload,
    }


def capture_upcoming(database_url: str = DEFAULT_HOST_DATABASE_URL) -> int:
    dsn = normalize_dsn(database_url)
    captured_at = datetime.now(timezone.utc)
    n = 0
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, stadium, gameday, gametime, home_team, roof
                FROM games
                WHERE home_score IS NULL AND stadium IS NOT NULL
                """
            )
            games = cur.fetchall()
            for game_id, stadium, gameday, gametime, home_team, roof in games:
                coords = _find_coords(stadium)
                if not coords:
                    continue
                kickoff = _kickoff_at(gameday, gametime, home_team)
                if kickoff is None:
                    continue
                try:
                    payload = fetch_forecast(*coords)
                    snap = snapshot_from_payload(payload, kickoff)
                except Exception as exc:
                    logger.warning("Open-Meteo fetch failed for %s: %s", stadium, exc)
                    continue
                cur.execute(
                    """
                    INSERT INTO weather_forecasts
                        (game_id, provider, kickoff_at, forecast_for, temp_f, wind_mph,
                         precipitation_bucket, captured_at, raw_payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        game_id,
                        PROVIDER,
                        kickoff,
                        kickoff,
                        snap["temp_f"],
                        snap["wind_mph"],
                        snap["precipitation_bucket"],
                        captured_at,
                        json.dumps(snap["raw_payload"], default=str),
                    ),
                )
                n += cur.rowcount
        conn.commit()
    logger.info("Open-Meteo captured %d weather_forecasts rows", n)
    return n
