"""
scraper/adapters/weather.py

Weather adapter re-export.

The full implementation lives in scraper/adapters/weather_adapter.py.
This module re-exports WeatherAdapter under the WeatherAdapter alias so
that code importing from either module name works correctly.

Usage:
    from scraper.adapters.weather import WeatherAdapter
    adapter = WeatherAdapter(api_key=os.environ["OPENWEATHER_API_KEY"])
    conditions = adapter.fetch_game_weather(lat=..., lon=..., game_dt=...)
"""

from __future__ import annotations

from scraper.adapters.weather_adapter import WeatherAdapter  # noqa: F401

__all__ = ["WeatherAdapter"]
