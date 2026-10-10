"""
Offline coverage tests for pipeline/provenance.py.

The as-of assertions take a count from a scripted cursor and must raise
ProvenanceError when any row is unsafe. backfill writes through FakeConn.
No database, no network.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import psycopg2
import pytest

import pipeline.provenance as prov
from backend.tests._pipeline_fakes import FakeConn, FakeConnFactory

T0 = datetime(2025, 9, 1, tzinfo=timezone.utc)
T1 = datetime(2025, 9, 8, tzinfo=timezone.utc)
URL = "postgresql+asyncpg://user:pw@db.test:5432/oracle"


def test_provenance_error_is_a_runtime_error():
    assert issubclass(prov.ProvenanceError, RuntimeError)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (72, 72.0),
        (72.0, 72.0),
        (float("nan"), None),
        ("6-0", 72.0),
        ("6'0", 72.0),
        ("6'2\"", 74.0),
        ("6\"2", 74.0),
        ("abc", None),
        ("6-x", None),
    ],
)
def test_height_to_inches_parses_each_nflverse_format(value, expected):
    assert prov.height_to_inches(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(None, None), ("230", 230.0), (
    "nan", None), ("abc", None), (5, 5.0)])
def test_as_float_returns_none_for_unusable_values(value, expected):
    assert prov._as_float(value) == expected


# ── backfill_player_season_profiles ───────────────────────────────────────────

@pytest.fixture
def db(monkeypatch) -> FakeConn:
    conn = FakeConn()
    monkeypatch.setattr(psycopg2, "connect", FakeConnFactory(conn))
    monkeypatch.setattr(prov, "execute_values", conn.execute_values)
    return conn


def test_backfill_keeps_latest_capture_per_player_season_and_scopes_to_seasons(db):
    db.respond("FROM staging_nflreadpy", rows=[
        {"raw_data": {"gsis_id": "00-1", "season": 2024, "height": "6-2", "weight": "230"},
         "ingested_at": T0},
        {"raw_data": {"gsis_id": "00-1", "season": 2024, "height": "6-3", "weight": "235"},
         "ingested_at": T1},  # later capture of the same player-season wins
        {"raw_data": {"player_id": "00-2", "season": 2025, "height": 73.0}, "ingested_at": T0},
        {"raw_data": {"season": 2025}, "ingested_at": T0},  # no player id: skipped
        {"raw_data": {"gsis_id": "00-3"}, "ingested_at": T0},  # no season: skipped
    ])
    written = prov.backfill_player_season_profiles(URL, seasons=[2025, 2024, 2024])

    assert written == 2
    rows = db.values_for("INSERT INTO player_season_profiles")[0]
    assert rows == [
        ("00-1", 2024, 75.0, 235.0, "nflreadpy.load_rosters", T1),
        ("00-2", 2025, 73.0, None, "nflreadpy.load_rosters", T0),
    ]
    _, params = db.statements_containing("FROM staging_nflreadpy")[0]
    assert params == ([2024, 2025],)  # sorted and de-duplicated
    assert db.connect_dsns == ["postgresql://user:pw@db.test:5432/oracle"]


def test_backfill_without_seasons_reads_all_roster_staging(db):
    db.respond("FROM staging_nflreadpy", rows=[
        {"raw_data": {"gsis_id": "00-1", "season": 2023, "height": "5-11"}, "ingested_at": T0},
    ])
    assert prov.backfill_player_season_profiles(URL) == 1

    sql, params = db.statements_containing("FROM staging_nflreadpy")[0]
    assert "ANY" not in sql
    assert params is None


def test_backfill_with_no_usable_rows_writes_nothing(db):
    db.respond("FROM staging_nflreadpy", rows=[{"raw_data": {}, "ingested_at": T0}])
    assert prov.backfill_player_season_profiles(URL, seasons=[2024]) == 0
    assert db.values_calls == []


# ── as-of assertions ──────────────────────────────────────────────────────────

def test_assert_player_profiles_asof_passes_when_every_row_has_a_profile(db):
    db.respond("FROM game_logs gl", rows=[(0,)])
    prov.assert_player_profiles_asof(db, 2024)  # no exception

    _, params = db.statements_containing("FROM game_logs gl")[0]
    assert params == (2024,)


def test_assert_player_profiles_asof_raises_with_the_missing_count(db):
    db.respond("FROM game_logs gl", rows=[(3,)])
    with pytest.raises(prov.ProvenanceError, match="3 player-game rows in season 2024"):
        prov.assert_player_profiles_asof(db, 2024)


def test_record_missing_profiles_inserts_placeholders_and_warns(db, caplog):
    db.respond("INSERT INTO player_season_profiles", rowcount=2)
    with caplog.at_level(logging.WARNING, logger="pipeline.provenance"):
        inserted = prov.record_missing_player_profiles_as_null(db, 2023)

    assert inserted == 2
    assert db.commits == 1
    sql, params = db.statements_containing("INSERT INTO player_season_profiles")[0]
    assert "no_retained_season_roster_snapshot" in sql
    assert params == (2023,)
    assert any("Recorded 2 NULL-only" in r.getMessage() for r in caplog.records)


def test_record_missing_profiles_with_nothing_missing_is_silent(db, caplog):
    db.respond("INSERT INTO player_season_profiles", rowcount=0)
    with caplog.at_level(logging.WARNING, logger="pipeline.provenance"):
        assert prov.record_missing_player_profiles_as_null(db, 2023) == 0
    assert db.commits == 1
    assert not any("NULL-only" in r.getMessage() for r in caplog.records)


def test_assert_depth_charts_pregame_raises_for_late_or_missing_publication(db):
    db.respond("FROM feature_matrix fm", rows=[(5,)])
    with pytest.raises(prov.ProvenanceError, match="5 depth-chart rows in season 2025"):
        prov.assert_depth_charts_pregame(db, 2025)


def test_assert_depth_charts_pregame_passes_when_all_publications_precede_kickoff(db):
    db.respond("FROM feature_matrix fm", rows=[(0,)])
    prov.assert_depth_charts_pregame(db, 2025)
    assert db.statements_containing("FROM feature_matrix fm")[0][1] == (2025,)


def test_assert_weather_forecasts_pregame_raises_for_uncovered_games(db):
    db.respond("LEFT JOIN LATERAL", rows=[(4,)])
    with pytest.raises(prov.ProvenanceError, match="4 player-game rows in season 2024"):
        prov.assert_weather_forecasts_pregame(db, 2024)


def test_assert_weather_forecasts_pregame_passes_when_forecasts_cover_kickoffs(db):
    db.respond("LEFT JOIN LATERAL", rows=[(0,)])
    prov.assert_weather_forecasts_pregame(db, 2024)
    assert db.statements_containing("LEFT JOIN LATERAL")[0][1] == (2024,)
