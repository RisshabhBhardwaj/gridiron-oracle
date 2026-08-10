from datetime import date

from pipeline.provenance import height_to_inches
from scraper.adapters.weather_adapter import _kickoff_at, fetch_forecast_snapshot


def test_height_to_inches_accepts_roster_formats():
    assert height_to_inches(74) == 74
    assert height_to_inches("6-2") == 74
    assert height_to_inches("6'2\"") == 74
    assert height_to_inches("unknown") is None


def test_kickoff_is_timezone_aware_for_supported_home_team():
    kickoff = _kickoff_at(date(2026, 9, 13), "13:00", "IND")
    assert kickoff is not None
    assert kickoff.tzinfo is not None
    assert kickoff.hour == 17  # Indianapolis is UTC-4 in September.


def test_closed_dome_forecast_needs_no_network():
    kickoff = _kickoff_at(date(2099, 9, 13), "13:00", "IND")
    forecast = fetch_forecast_snapshot("unused", "Lucas Oil Stadium", kickoff, "dome")
    assert forecast["wind_mph"] == 0.0
    assert forecast["precipitation_bucket"] == 0
