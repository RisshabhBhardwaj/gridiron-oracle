"""Regression guards for scraper/adapters/weather_adapter.py's kickoff_at helper."""

from __future__ import annotations

from datetime import date

from scraper.adapters.weather_adapter import TEAM_TIMEZONES, _kickoff_at


def test_kickoff_at_combines_gameday_and_gametime_in_stadium_local_time():
    kickoff = _kickoff_at(date(2025, 9, 7), "13:00", "KC")
    assert kickoff is not None
    assert kickoff.hour == 18  # 13:00 Central -> 18:00 UTC (no DST offset change in Sept)


def test_kickoff_at_covers_legacy_team_abbreviations():
    """
    nflverse's schedules feed still uses "LA" (not "LAR") and "OAK" (pre-Vegas
    Raiders) for some rows. A missing alias meant games.kickoff_at silently
    stayed NULL for those games, which in turn meant the depth-chart
    publication guard's pre-kickoff proof could never fire for them.
    """
    for team in ("LA", "OAK", "LAR", "LV"):
        assert team in TEAM_TIMEZONES, f"{team} missing from TEAM_TIMEZONES"
    assert _kickoff_at(date(2025, 9, 7), "16:05", "LA") is not None
    assert _kickoff_at(date(2015, 9, 13), "13:00", "OAK") is not None


def test_kickoff_at_returns_none_for_unknown_team():
    assert _kickoff_at(date(2025, 9, 7), "13:00", "ZZZ") is None


def test_kickoff_at_returns_none_for_missing_gametime():
    assert _kickoff_at(date(2025, 9, 7), None, "KC") is None
