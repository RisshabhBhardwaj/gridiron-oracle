"""
Offline coverage tests for the SQL loaders and roof helpers in
pipeline/team_game_features.py.

The loaders were only exercised by DB-backed tests, which skip when no
database is reachable. They run here against FakeConn with scripted rows
and assert the frame shape, the filters applied, and the bound parameters.
No database, no network.
"""

from __future__ import annotations

import pandas as pd

import pipeline.team_game_features as tgf
from backend.tests._pipeline_fakes import FakeConn


def test_load_games_returns_one_row_per_scored_game_for_the_requested_seasons():
    conn = FakeConn()
    conn.respond("FROM games WHERE season = ANY", rows=[
        {"game_id": "2024_01_KC_BAL", "season": 2024, "week": 1, "home_team": "KC",
         "away_team": "BAL", "home_score": 27, "away_score": 20, "home_rest": 7,
         "away_rest": 7, "spread_line": -3.0, "total_line": 46.5, "roof": "outdoors",
         "surface": "grass", "home_coach": "A. Reid", "away_coach": "J. Harbaugh",
         "kickoff_at": None},
    ])
    df = tgf._load_games(conn, [2024])

    assert list(df["game_id"]) == ["2024_01_KC_BAL"]
    assert df.iloc[0]["home_score"] == 27
    assert df.iloc[0]["home_coach"] == "A. Reid"
    _, params = conn.statements_containing("FROM games WHERE season = ANY")[0]
    assert params == ([2024],)


def test_load_games_with_no_rows_returns_an_empty_frame():
    conn = FakeConn()
    conn.respond("FROM games WHERE season = ANY", rows=[])
    assert tgf._load_games(conn, [2024]).empty


def test_load_team_game_stats_maps_each_team_week_row():
    conn = FakeConn()
    conn.respond("FROM team_game_stats", rows=[
        {"team": "KC", "season": 2024, "week": 1, "pass_attempts": 34, "rush_attempts": 22,
         "total_plays": 56, "total_yards": 400.0},
    ])
    df = tgf._load_team_game_stats(conn, [2024])

    assert df.iloc[0]["team"] == "KC"
    assert df.iloc[0]["total_plays"] == 56
    sql, params = conn.statements_containing("FROM team_game_stats")[0]
    assert "COALESCE(total_plays, pass_attempts + rush_attempts)" in sql
    assert params == ([2024],)


def test_load_team_elo_keeps_distinct_team_weeks_with_elo_present():
    conn = FakeConn()
    conn.respond("FROM feature_matrix", rows=[
        {"season": 2024, "week": 1, "team": "KC", "team_off_elo": 1520.0, "team_def_elo": 1490.0},
    ])
    df = tgf._load_team_elo(conn, [2024])

    assert list(df.columns) == ["season", "week", "team", "team_off_elo", "team_def_elo"]
    assert df.iloc[0]["team_off_elo"] == 1520.0
    sql, _ = conn.statements_containing("FROM feature_matrix")[0]
    assert "team_off_elo IS NOT NULL" in sql


def test_load_coach_plays_drops_plays_with_no_resolvable_coach():
    conn = FakeConn()
    conn.respond("FROM pbp_plays pp", rows=[
        {"season": 2024, "week": 1, "play_type": "pass", "coach": "A. Reid"},
        {"season": 2024, "week": 1, "play_type": "run", "coach": None},  # posteam matched no game side
    ])
    df = tgf._load_coach_plays(conn, [2024])

    assert df["coach"].tolist() == ["A. Reid"]
    _, params = conn.statements_containing("FROM pbp_plays pp")[0]
    assert params == ([2024], tgf._NEUTRAL_SCORE_MARGIN, tgf._NEUTRAL_MIN_SECONDS_REMAINING)


def test_load_coach_plays_with_no_plays_returns_the_expected_empty_shape():
    conn = FakeConn()
    conn.respond("FROM pbp_plays pp", rows=[])
    df = tgf._load_coach_plays(conn, [2024])

    assert df.empty
    assert list(df.columns) == ["season", "week", "play_type", "coach"]


def test_load_home_team_roof_takes_the_modal_roof_per_home_team():
    conn = FakeConn()
    conn.respond("FROM games", rows=[
        ("KC", "outdoors", 8),
        ("KC", "dome", 2),  # a minority value does not redefine the stadium
        ("LV", "dome", 9),
        ("LV", "outdoors", 1),
    ])
    roofs = tgf._load_home_team_roof(conn)

    assert roofs == {"KC": "outdoors", "LV": "dome"}


def test_load_home_team_roof_skips_rows_without_a_home_team():
    conn = FakeConn()
    conn.respond("FROM games", rows=[(None, "dome", 5), ("KC", "dome", 3)])
    roofs = tgf._load_home_team_roof(conn)

    assert roofs == {"KC": "dome"}


def test_fill_missing_roof_only_fills_null_values_from_the_home_team_mode():
    games = pd.DataFrame([
        {"game_id": "G1", "home_team": "KC", "roof": None},
        {"game_id": "G2", "home_team": "KC", "roof": "outdoors"},  # present: kept
        {"game_id": "G3", "home_team": "LV", "roof": None},
    ])
    out = tgf._fill_missing_roof(games, {"KC": "outdoors", "LV": "dome"})

    assert out.set_index("game_id")["roof"].to_dict() == {
        "G1": "outdoors", "G2": "outdoors", "G3": "dome",
    }
    assert games["roof"].isna().sum() == 2  # the caller's frame is not mutated


def test_fill_missing_roof_leaves_unknown_home_teams_null():
    games = pd.DataFrame([{"game_id": "G1", "home_team": "ZZZ", "roof": None}])
    out = tgf._fill_missing_roof(games, {"KC": "dome"})
    assert pd.isna(out.iloc[0]["roof"])


def test_fill_missing_roof_passes_through_empty_or_roofless_frames():
    empty = pd.DataFrame()
    assert tgf._fill_missing_roof(empty, {"KC": "dome"}) is empty

    no_roof = pd.DataFrame([{"game_id": "G1", "home_team": "KC"}])
    assert tgf._fill_missing_roof(no_roof, {"KC": "dome"}) is no_roof


def test_is_turf_flags_every_artificial_spelling_and_treats_unknown_as_grass():
    surface = pd.Series(["fieldturf", "grass", None, "Matrixturf", "  GRASS ", "", "a_turf"])
    flags = tgf._is_turf(surface).tolist()

    # Artificial surfaces are 1; grass, blank and unknown (null) are 0.
    assert flags == [1, 0, 0, 1, 0, 0, 1]


def test_load_forward_games_selects_the_requested_week_whether_played_or_not():
    conn = FakeConn()
    conn.respond("FROM games WHERE season = %s AND week = %s", rows=[
        {"game_id": "2025_03_KC_BUF", "season": 2025, "week": 3, "home_team": "KC",
         "away_team": "BUF", "home_score": None, "away_score": None, "home_rest": 7,
         "away_rest": 6, "spread_line": None, "total_line": None, "roof": None,
         "surface": "grass", "home_coach": None, "away_coach": None, "kickoff_at": None},
    ])
    df = tgf._load_forward_games(conn, 2025, 3)

    assert list(df["game_id"]) == ["2025_03_KC_BUF"]
    assert pd.isna(df.iloc[0]["home_score"])  # a future game has no score yet
    _, params = conn.statements_containing("FROM games WHERE season = %s AND week = %s")[0]
    assert params == (2025, 3)


def test_load_latest_team_elo_picks_each_teams_most_recent_value_before_the_week():
    conn = FakeConn()
    conn.respond("FROM feature_matrix", rows=[
        {"team": "KC", "team_off_elo": 1530.0, "team_def_elo": 1480.0},
    ])
    df = tgf._load_latest_team_elo(conn, 2025, 3)

    assert df.iloc[0]["team"] == "KC"
    sql, params = conn.statements_containing("FROM feature_matrix")[0]
    assert "DISTINCT ON (team)" in sql
    assert "ORDER BY team, season DESC, week DESC" in sql
    assert params == (2025, 2025, 3)  # strictly before (season, week)


def test_build_team_game_forward_frame_returns_empty_when_no_games_are_scheduled(monkeypatch):
    conn = FakeConn()
    conn.respond("FROM games WHERE season = %s AND week = %s", rows=[])
    monkeypatch.setattr(tgf.psycopg2, "connect", lambda dsn: conn)

    df = tgf.build_team_game_forward_frame("postgresql://u:p@db.test/oracle", 2025, 30)

    assert df.empty
    assert conn.closed is True
