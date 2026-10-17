"""
Offline coverage tests for pipeline/normalize.py.

Pure transforms are tested on validated row models. The Normalizer's DB
paths run against backend/tests/_pipeline_fakes.FakeConn, which records
every statement and execute_values batch. Nothing here touches Postgres
or the network.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timezone
from typing import Any

import psycopg2
import pytest

import pipeline.normalize as normalize_mod
from backend.tests._pipeline_fakes import FakeConn
from pipeline.normalize import (
    NormalizeSummary,
    Normalizer,
    _roster_height_to_str,
    main,
    player_stats_row_to_game_log,
    roster_row_to_player,
    schedule_row_to_game,
    schedule_row_to_teams,
)
from scraper.adapters.nflreadpy_adapter import (
    PlayerStatsRow,
    RosterRow,
    ScheduleRow,
)

URL = "postgresql+asyncpg://user:pw@db.test:5432/oracle"

FK_ORDER = [
    "schedules", "rosters", "player_stats", "snap_counts", "depth_charts",
    "nextgen_stats", "team_stats", "ftn_charting", "participation", "combine",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def staged(row_id: int, raw: dict[str, Any]) -> dict[str, Any]:
    return {"id": row_id, "raw_data": raw, "ingested_at": datetime(2025, 9, 1, tzinfo=timezone.utc)}


def schedule_raw(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        game_id="2025_01_KC_BAL", season=2025, week=1, game_type="REG",
        home_team="KC", away_team="BAL", home_score=27, away_score=20,
        gameday="2025-09-04", gametime="20:20", weekday="Thursday",
        stadium="Arrowhead Stadium", roof="outdoors", surface="grass",
        temp=75.0, wind=5.0, spread_line=-3.0, total_line=46.5,
        away_moneyline=130.0, home_moneyline=-150.0, home_rest=7, away_rest=7,
        home_qb_name="P. Mahomes", away_qb_name="L. Jackson",
        home_coach="A. Reid", away_coach="J. Harbaugh",
    )
    base.update(overrides)
    return base


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> FakeConn:
    conn = FakeConn()

    def _connect(dsn: str, *args: Any, **kwargs: Any) -> FakeConn:
        conn.connect_dsns.append(dsn)
        return conn

    monkeypatch.setattr(normalize_mod.psycopg2, "connect", _connect)
    monkeypatch.setattr(normalize_mod, "execute_values", conn.execute_values)
    return conn


@pytest.fixture
def norm(fake_db: FakeConn) -> Normalizer:
    n = Normalizer(URL)
    n.connect()
    yield n
    n.close()


# ── pure transforms ───────────────────────────────────────────────────────────

def test_schedule_row_to_game_maps_each_field_for_the_games_insert():
    row = ScheduleRow.model_validate(schedule_raw())
    out = schedule_row_to_game(row)

    assert out["id"] == "2025_01_KC_BAL"
    assert out["gameday"] == date(2025, 9, 4)
    assert out["home_moneyline"] == -150.0
    assert out["away_coach"] == "J. Harbaugh"
    assert out["stadium"] == "Arrowhead Stadium"
    assert len(out) == 26


def test_schedule_row_to_teams_returns_home_then_away_stubs():
    row = ScheduleRow.model_validate(schedule_raw())
    assert schedule_row_to_teams(row) == [{"id": "KC"}, {"id": "BAL"}]


@pytest.mark.parametrize(
    ("height", "expected"),
    [
        (None, None),
        ("6-2", "6-2"),
        (74.0, "6-2"),
        (72.4, "6-0"),
        ([], None),  # not str/float: float() raises TypeError -> None
    ],
)
def test_roster_height_to_str(height, expected):
    assert _roster_height_to_str(height) == expected


@pytest.mark.parametrize(
    ("draft_number", "expected_round"),
    [(1, 1.0), (32, 1.0), (33, 2.0), (0, 1.0), (500, 7.0)],
)
def test_roster_row_to_player_derives_draft_round_and_clamps(draft_number, expected_round):
    row = RosterRow(
        season=2024, gsis_id="00-0033873", full_name="Patrick Mahomes",
        position="QB", team="KC", height=74.0, weight=230.0,
        espn_id="3139477", draft_number=draft_number, draft_club="KC",
    )
    out = roster_row_to_player(row)
    assert out is not None
    assert out["id"] == "00-0033873"
    assert out["height"] == "6-2"
    assert out["espn_id"] == "3139477"
    assert out["draft_round"] == expected_round


def test_roster_row_to_player_without_draft_or_espn_leaves_them_null():
    row = RosterRow(season=2024, gsis_id="00-1", full_name="No Draft")
    out = roster_row_to_player(row)
    assert out is not None
    assert out["draft_round"] is None
    assert out["espn_id"] is None


def test_roster_row_to_player_without_gsis_id_returns_none():
    assert roster_row_to_player(RosterRow(season=2024, gsis_id=None, full_name="X")) is None


def test_player_stats_row_to_game_log_maps_stats_and_nulls_snap_columns():
    row = PlayerStatsRow(
        player_id="00-0033873", season=2024, week=1, game_id="2024_01_KC_BAL",
        team="KC", opponent_team="BAL", position="QB", completions=20,
        attempts=30, passing_yards=280.0, passing_tds=2, fantasy_points_ppr=24.5,
    )
    out = player_stats_row_to_game_log(row)
    assert out is not None
    assert out["player_id"] == "00-0033873"
    assert out["game_id"] == "2024_01_KC_BAL"
    assert out["passing_yards"] == 280.0
    assert out["fantasy_points_ppr"] == 24.5
    # Snap columns are filled later by the snap_counts pass, never here.
    assert out["offense_snaps"] is None
    assert out["offense_pct"] is None


def test_player_stats_row_without_game_id_returns_none():
    row = PlayerStatsRow(player_id="00-1", season=2024, week=1, game_id=None)
    assert player_stats_row_to_game_log(row) is None


def test_normalize_summary_log_prints_every_counter(capsys):
    summary = NormalizeSummary(
        teams_upserted=2, players_upserted=5, games_upserted=1,
        game_logs_upserted=7, staging_rows_processed=9, skipped=3, errors=4,
    )
    summary.log()
    out = capsys.readouterr().out
    assert "Players upserted  : 5" in out
    assert "Game logs upserted: 7" in out
    assert "Staging rows      : 9 processed" in out
    assert "Skipped           : 3" in out
    assert "Errors            : 4" in out


# ── connection lifecycle ──────────────────────────────────────────────────────

def test_enter_connects_with_plain_dsn_and_exit_closes(fake_db: FakeConn):
    with Normalizer(URL) as norm:
        assert norm._conn is fake_db
    assert fake_db.connect_dsns == ["postgresql://user:pw@db.test:5432/oracle"]
    assert fake_db.autocommit is False
    assert fake_db.closed is True
    assert fake_db.rollbacks == 0


def test_exit_rolls_back_and_propagates_when_block_raises(fake_db: FakeConn):
    with pytest.raises(RuntimeError, match="boom"):
        with Normalizer(URL):
            raise RuntimeError("boom")
    assert fake_db.rollbacks == 1
    assert fake_db.closed is True


def test_exit_skips_rollback_when_connection_already_closed(fake_db: FakeConn):
    norm = Normalizer(URL)
    norm.connect()
    fake_db.closed = True
    norm.__exit__(None, None, None)
    assert fake_db.rollbacks == 0


def test_close_before_connect_is_a_noop(fake_db: FakeConn):
    Normalizer(URL).close()
    assert fake_db.connect_dsns == []
    assert fake_db.statements == []


# ── staging reader and purge ──────────────────────────────────────────────────

def test_fetch_unprocessed_filters_by_source_seasons_and_limit(norm, fake_db):
    fake_db.respond(
        "SELECT id, raw_data, ingested_at FROM staging_nflreadpy",
        rows=[{"id": 7, "raw_data": {"a": 1}, "ingested_at": "t"}],
    )
    out = norm._fetch_unprocessed("schedules", seasons=[2024, 2025], limit=500)

    assert out == [{"id": 7, "raw_data": {"a": 1}, "ingested_at": "t"}]
    sql, params = fake_db.statements[-1]
    assert "source_type = %s" in sql and "NOT processed" in sql
    assert "season = ANY(%s)" in sql and "LIMIT %s" in sql
    assert params == ("schedules", [2024, 2025], 500)


def test_fetch_unprocessed_without_filters_omits_season_and_limit(norm, fake_db):
    norm._fetch_unprocessed("rosters")
    sql, params = fake_db.statements[-1]
    assert "ANY" not in sql and "LIMIT" not in sql
    assert params == ("rosters",)


def test_mark_processed_deletes_rows_for_purge_on_consume_source(norm, fake_db):
    assert "depth_charts" in Normalizer.PURGE_ON_CONSUME
    norm._current_source_type = "depth_charts"
    norm._mark_processed([11, 12])

    deletes = fake_db.statements_containing("DELETE FROM staging_nflreadpy")
    assert deletes == [("DELETE FROM staging_nflreadpy WHERE id = ANY(%s)", ([11, 12],))]
    assert fake_db.statements_containing("UPDATE staging_nflreadpy") == []
    assert fake_db.commits == 1


def test_mark_processed_updates_rows_for_source_kept_for_downstream_readers(norm, fake_db):
    assert "rosters" not in Normalizer.PURGE_ON_CONSUME
    norm._current_source_type = "rosters"
    norm._mark_processed([3])

    updates = fake_db.statements_containing("UPDATE staging_nflreadpy SET processed = TRUE")
    assert updates[0][1] == ([3],)
    assert fake_db.statements_containing("DELETE FROM staging_nflreadpy") == []


def test_mark_processed_outside_run_falls_back_to_update(norm, fake_db):
    norm._current_source_type = None
    norm._mark_processed([5])
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")
    assert fake_db.statements_containing("DELETE") == []


def test_mark_processed_with_no_ids_touches_nothing(norm, fake_db):
    norm._mark_processed([])
    assert fake_db.statements == []
    assert fake_db.commits == 0


# ── upserts ───────────────────────────────────────────────────────────────────

def test_upsert_teams_dedupes_ids_and_returns_unique_count(norm, fake_db):
    n = norm._upsert_teams([{"id": "KC"}, {"id": "BAL"}, {"id": "KC"}])
    assert n == 2
    rows = fake_db.values_for("INSERT INTO teams")[0]
    assert [r[0] for r in rows] == ["KC", "BAL"]
    assert all(isinstance(r[1], datetime) for r in rows)
    assert fake_db.commits == 1


def test_upsert_teams_empty_is_noop(norm, fake_db):
    assert norm._upsert_teams([]) == 0
    assert fake_db.values_calls == []


def test_upsert_games_computes_kickoff_in_utc_and_nulls_unknown_home_team(norm, fake_db):
    games = [
        {"id": "G1", "season": 2025, "week": 1, "home_team": "KC", "away_team": "BAL",
         "gameday": date(2025, 9, 4), "gametime": "20:20"},
        {"id": "G2", "season": 2025, "week": 1, "home_team": "XYZ", "away_team": "ABC",
         "gameday": date(2025, 9, 4), "gametime": "13:00"},
    ]
    n = norm._upsert_games(games)
    assert n == 2

    rows = fake_db.values_for("INSERT INTO games")[0]
    assert rows[0][0] == "G1"
    # KC is America/Chicago (CDT, UTC-5 in September): 20:20 local is 01:20 UTC next day.
    assert rows[0][-1] == datetime(2025, 9, 5, 1, 20, tzinfo=timezone.utc)
    assert rows[1][0] == "G2"
    assert rows[1][-1] is None


def test_upsert_players_writes_eighteen_columns_per_player(norm, fake_db):
    players = [{"id": "00-1", "full_name": "A One", "position": "QB", "team": "KC"}]
    assert norm._upsert_players(players) == 1
    row = fake_db.values_for("INSERT INTO players (")[0][0]
    assert row[0] == "00-1" and row[1] == "A One"
    assert len(row) == 18  # 17 profile columns + updated_at
    assert isinstance(row[-1], datetime)


def test_upsert_players_empty_is_noop(norm, fake_db):
    assert norm._upsert_players([]) == 0
    assert fake_db.values_calls == []


def test_upsert_player_season_profiles_keeps_last_value_per_player_season(norm, fake_db):
    ingested = datetime(2025, 9, 1, tzinfo=timezone.utc)
    staging_rows = [
        {"raw_data": {"gsis_id": "00-1", "season": 2024, "height": "6-2", "weight": "230"},
         "ingested_at": ingested},
        {"raw_data": {"player_id": "00-2", "season": 2024, "height": 73.0, "weight": "not-a-number"},
         "ingested_at": ingested},
        {"raw_data": {"season": 2024}, "ingested_at": ingested},  # no player id: skipped
        {"raw_data": {"gsis_id": "00-3"}, "ingested_at": ingested},  # no season: skipped
        {"raw_data": {"gsis_id": "00-1", "season": 2024, "height": "6-3"}, "ingested_at": ingested},
    ]
    n = norm._upsert_player_season_profiles(staging_rows)
    assert n == 2

    rows = fake_db.values_for("INSERT INTO player_season_profiles")[0]
    # 00-1 appears twice; the later "6-3" row replaces the earlier "6-2" row, so weight is None.
    assert rows[0] == ("00-1", 2024, 75.0, None, "nflreadpy.load_rosters", ingested)
    assert rows[1] == ("00-2", 2024, 73.0, None, "nflreadpy.load_rosters", ingested)


def test_upsert_player_season_profiles_with_nothing_valid_writes_nothing(norm, fake_db):
    assert norm._upsert_player_season_profiles([{"raw_data": {}, "ingested_at": None}]) == 0
    assert fake_db.values_calls == []


def test_upsert_player_stubs_takes_first_occurrence_and_falls_back_to_player_name(norm, fake_db):
    staging_rows = [
        {"player_id": "00-1", "player_display_name": "Patrick Mahomes", "position": "QB", "team": "KC"},
        {"player_id": "00-1", "player_display_name": "Duplicate", "position": "WR", "team": "BUF"},
        {"raw_data": {"player_id": "00-2", "player_name": "Travis Kelce", "position": "TE", "team": "KC"}},
        {"position": "RB"},  # no player id: skipped
    ]
    n = norm._upsert_player_stubs_from_stats(staging_rows)
    assert n == 2

    rows = fake_db.values_for("INSERT INTO players (id, full_name, position, team, updated_at)")[0]
    assert [r[:4] for r in rows] == [
        ("00-1", "Patrick Mahomes", "QB", "KC"),
        ("00-2", "Travis Kelce", "TE", "KC"),
    ]
    sql = " ".join(fake_db.values_calls[-1][0].split())
    assert "ON CONFLICT (id) DO NOTHING" in sql


def test_upsert_player_stubs_empty_is_noop(norm, fake_db):
    assert norm._upsert_player_stubs_from_stats([{"raw_data": {"position": "QB"}}]) == 0
    assert fake_db.values_calls == []


def test_build_game_id_lookup_indexes_both_teams_of_each_game(norm, fake_db):
    fake_db.respond(
        "SELECT id, season, week, home_team, away_team FROM games",
        rows=[{"id": "G1", "season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL"}],
    )
    assert norm._build_game_id_lookup() == {(2024, 1, "KC"): "G1", (2024, 1, "BAL"): "G1"}


def test_build_game_id_lookup_asserts_when_schedules_not_loaded(norm, fake_db):
    with pytest.raises(AssertionError, match="schedules step must run"):
        norm._build_game_id_lookup()


def test_upsert_game_logs_flushes_in_batches_and_counts_every_row(norm, fake_db, monkeypatch):
    monkeypatch.setattr(normalize_mod, "BATCH_SIZE", 2)
    logs = [{"player_id": f"p{i}", "game_id": "G", "season": 2024, "week": 1} for i in range(5)]
    summary = NormalizeSummary()

    assert norm._upsert_game_logs(logs, summary) == 5
    batches = fake_db.values_for("INSERT INTO game_logs")
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [b[0][0] for b in batches] == ["p0", "p2", "p4"]
    assert fake_db.commits == 3
    assert summary.errors == 0


def test_upsert_game_logs_bad_row_costs_only_that_row_and_is_dead_lettered(
    norm, fake_db, monkeypatch
):
    """H-3: one bad row in a batch no longer drops the whole batch (up to 500 rows).

    The batch insert fails, rolls back, and is retried row by row in savepoints.
    Only the offending row is lost, and it is dead-lettered with its payload so
    the raw data survives for replay.
    """
    monkeypatch.setattr(normalize_mod, "BATCH_SIZE", 2)
    real = fake_db.execute_values

    def one_row_always_fails(cur, sql, argslist, **kwargs):
        if "INSERT INTO game_logs" in " ".join(sql.split()) and any(r[0] == "p0" for r in argslist):
            raise psycopg2.IntegrityError("foreign key violation")
        return real(cur, sql, argslist, **kwargs)

    monkeypatch.setattr(normalize_mod, "execute_values", one_row_always_fails)
    logs = [{"player_id": f"p{i}", "game_id": "G"} for i in range(5)]
    summary = NormalizeSummary()

    assert norm._upsert_game_logs(logs, summary) == 4
    assert summary.errors == 1  # only p0
    assert fake_db.rollbacks == 1  # the failed batch insert rolled back
    dead = fake_db.statements_containing("INSERT INTO dead_letter")
    assert len(dead) == 1
    assert dead[0][1][0] == "normalize/game_logs"
    assert "player_id=p0" in dead[0][1][1]


def test_upsert_game_logs_empty_is_noop(norm, fake_db):
    summary = NormalizeSummary()
    assert norm._upsert_game_logs([], summary) == 0
    assert fake_db.values_calls == []


# ── source processors ─────────────────────────────────────────────────────────

def test_process_schedules_upserts_valid_games_and_counts_invalid_rows(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_schedules(
        [
            staged(1, schedule_raw()),
            staged(2, {"game_id": "2025_01_X"}),  # missing season/week/teams
        ],
        summary,
    )
    assert summary.games_upserted == 1
    assert summary.teams_upserted == 2
    assert summary.errors == 1
    assert summary.staging_rows_processed == 2

    games = fake_db.values_for("INSERT INTO games")[0]
    assert [g[0] for g in games] == ["2025_01_KC_BAL"]
    teams = fake_db.values_for("INSERT INTO teams")[0]
    assert [t[0] for t in teams] == ["KC", "BAL"]
    # Both the valid and the invalid staging rows are retired, so the bad row is not retried forever.
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2],)


def test_process_rosters_dedupes_by_gsis_id_and_skips_rows_without_one(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_rosters(
        [
            staged(1, {"season": 2024, "gsis_id": "00-1", "full_name": "A One", "team": "KC", "height": 74.0}),
            staged(2, {"season": 2024, "gsis_id": "00-1", "full_name": "A One", "team": "BUF"}),
            staged(3, {"season": 2024, "gsis_id": None, "full_name": "No Id"}),
            staged(4, {"gsis_id": "00-9"}),  # missing season: invalid
        ],
        summary,
    )
    assert summary.players_upserted == 1
    assert summary.skipped == 1
    assert summary.errors == 1
    assert summary.staging_rows_processed == 4

    players = fake_db.values_for("INSERT INTO players (")[0]
    assert len(players) == 1
    assert players[0][0] == "00-1"
    assert players[0][3] == "BUF"  # last occurrence wins

    # Profiles keep the last row per (player, season), same as the roster dedupe:
    # the second 00-1 row carries no height, so the stored height is NULL.
    profiles = fake_db.values_for("INSERT INTO player_season_profiles")[0]
    assert [(p[0], p[1], p[2]) for p in profiles] == [("00-1", 2024, None)]
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4],)


def test_process_player_stats_derives_game_id_and_skips_unmatched_rows(norm, fake_db):
    fake_db.respond(
        "SELECT id, season, week, home_team, away_team FROM games",
        rows=[{"id": "2024_01_KC_BAL", "season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL"}],
    )
    staging = [
        staged(1, {"player_id": "00-1", "season": 2024, "week": 1, "team": "KC",
                   "player_display_name": "Patrick Mahomes", "position": "QB", "passing_yards": 280.0}),
        staged(2, {"player_id": "00-2", "season": 2024, "week": 1, "team": "ZZZ"}),  # no schedule match
        staged(3, {"season": 2024, "week": 1, "game_id": "2024_01_KC_BAL", "team": "KC"}),  # no player id
        staged(4, {"player_id": "00-3", "season": 2024, "week": 1, "game_id": "2024_01_KC_BAL",
                   "team": "KC", "position": "RB", "carries": 10}),
    ]
    summary = NormalizeSummary()
    norm._process_player_stats(staging, summary)

    assert summary.game_logs_upserted == 2
    assert summary.skipped == 1
    assert summary.errors == 1
    assert summary.staging_rows_processed == 4

    logs = fake_db.values_for("INSERT INTO game_logs")[0]
    assert [(r[0], r[1]) for r in logs] == [("00-1", "2024_01_KC_BAL"), ("00-3", "2024_01_KC_BAL")]

    # The derived game_id is applied to a copy: the staging row itself is unchanged.
    assert "game_id" not in staging[0]["raw_data"]

    # Stubs come from every row that has a player id, including the unmatched one.
    stub_ids = [r[0] for r in fake_db.values_for("INSERT INTO players (id, full_name, position, team, updated_at)")[0]]
    assert stub_ids == ["00-1", "00-2", "00-3"]
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4],)


def test_process_snap_counts_matches_exact_case_insensitive_and_fuzzy_names(norm, fake_db):
    fake_db.respond(
        "SELECT id, full_name FROM players WHERE full_name IS NOT NULL",
        rows=[
            {"id": "00-1", "full_name": "Patrick Mahomes"},
            {"id": "00-2", "full_name": "Travis Kelce"},
            {"id": "00-3", "full_name": None},
        ],
    )
    base = {"game_id": "2024_01_KC_BAL", "season": 2024, "week": 1}
    staging = [
        staged(1, {**base, "player": "Patrick Mahomes", "offense_snaps": 60, "offense_pct": 0.95}),
        staged(2, {**base, "player": "travis kelce", "offense_snaps": 55, "offense_pct": 0.90}),
        staged(3, {**base, "player": "Travis Kelcee", "offense_snaps": 10, "offense_pct": 0.15}),
        staged(4, {**base, "player": None, "offense_snaps": 1, "offense_pct": 0.01}),
        staged(5, {**base, "player": "Zzz Qqq", "offense_snaps": 2, "offense_pct": 0.02}),
        staged(6, {"season": 2024, "week": 1, "player": "Nobody"}),  # no game id: invalid
    ]
    summary = NormalizeSummary()
    norm._process_snap_counts(staging, summary)

    updates = fake_db.statements_containing("UPDATE game_logs")
    assert [params for _, params in updates] == [
        (60, 0.95, "00-1", "2024_01_KC_BAL"),
        (55, 0.90, "00-2", "2024_01_KC_BAL"),
        (10, 0.15, "00-2", "2024_01_KC_BAL"),
    ]
    assert summary.skipped == 2  # no player name, and an unmatched name
    assert summary.errors == 1  # the row that failed validation
    assert summary.staging_rows_processed == 6
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4, 5, 6],)


def test_process_snap_counts_flushes_full_batches_before_the_end(norm, fake_db, monkeypatch):
    monkeypatch.setattr(normalize_mod, "BATCH_SIZE", 2)
    fake_db.respond(
        "SELECT id, full_name FROM players WHERE full_name IS NOT NULL",
        rows=[{"id": f"00-{i}", "full_name": f"Player {i}"} for i in range(3)],
    )
    base = {"game_id": "G", "season": 2024, "week": 1}
    staging = [staged(i, {**base, "player": f"Player {i}", "offense_snaps": i, "offense_pct": 0.1})
               for i in range(3)]
    summary = NormalizeSummary()
    norm._process_snap_counts(staging, summary)

    assert len(fake_db.statements_containing("UPDATE game_logs")) == 3
    # Two flushes of game_logs updates (2 + 1) plus the staging retirement commit.
    assert fake_db.commits == 3
    assert summary.errors == 0


def test_process_snap_counts_update_failure_rolls_back_and_counts_each_pending_row(
    norm, fake_db
):
    fake_db.respond(
        "SELECT id, full_name FROM players WHERE full_name IS NOT NULL",
        rows=[{"id": "00-1", "full_name": "Patrick Mahomes"}],
    )
    fake_db.raise_on["UPDATE game_logs"] = psycopg2.DataError("bad value")
    base = {"game_id": "G", "season": 2024, "week": 1}
    staging = [staged(i, {**base, "player": "Patrick Mahomes", "offense_snaps": 5, "offense_pct": 0.5})
               for i in range(3)]
    summary = NormalizeSummary()
    norm._process_snap_counts(staging, summary)

    assert fake_db.rollbacks == 1
    assert summary.errors == 3
    assert summary.staging_rows_processed == 3


def test_resolve_depth_chart_week_picks_next_kickoff_or_final_week(norm):
    k1 = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
    k2 = datetime(2025, 9, 14, 17, 0, tzinfo=timezone.utc)
    k3 = datetime(2025, 9, 21, 17, 0, tzinfo=timezone.utc)
    schedule = {(2025, "KC"): [(1, k1), (2, k2), (3, k3)]}

    before_first = datetime(2025, 9, 1, tzinfo=timezone.utc)
    between = datetime(2025, 9, 10, tzinfo=timezone.utc)
    after_last = datetime(2025, 10, 1, tzinfo=timezone.utc)

    assert norm._resolve_depth_chart_week(schedule, 2025, "KC", before_first) == 1
    assert norm._resolve_depth_chart_week(schedule, 2025, "KC", between) == 2
    assert norm._resolve_depth_chart_week(schedule, 2025, "KC", after_last) == 3
    # ISO strings are parsed; a trailing Z is accepted.
    assert norm._resolve_depth_chart_week(schedule, 2025, "KC", "2025-09-10T00:00:00Z") == 2
    assert norm._resolve_depth_chart_week(schedule, 2025, "KC", "not-a-date") is None
    assert norm._resolve_depth_chart_week(schedule, 2025, "KC", None) is None
    assert norm._resolve_depth_chart_week(schedule, 2025, "LV", between) is None


def test_load_team_schedule_indexes_both_teams_sorted_by_kickoff(norm, fake_db):
    k1 = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
    k2 = datetime(2025, 9, 14, 17, 0, tzinfo=timezone.utc)
    fake_db.respond(
        "SELECT season, home_team, away_team, week, kickoff_at FROM games",
        rows=[(2025, "KC", "BAL", 2, k2), (2025, "KC", "DEN", 1, k1)],
    )
    schedule = norm._load_team_schedule({2025})

    assert schedule == {
        (2025, "KC"): [(1, k1), (2, k2)],
        (2025, "BAL"): [(2, k2)],
        (2025, "DEN"): [(1, k1)],
    }
    _, params = fake_db.statements[-1]
    assert params == ([2025],)


def test_load_team_schedule_without_seasons_does_not_query(norm, fake_db):
    assert norm._load_team_schedule(set()) == {}
    assert fake_db.statements == []


def test_process_depth_charts_resolves_weeks_and_keeps_newest_snapshot(norm, fake_db):
    k1 = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
    k2 = datetime(2025, 9, 14, 17, 0, tzinfo=timezone.utc)
    fake_db.respond(
        "SELECT season, home_team, away_team, week, kickoff_at FROM games",
        rows=[(2025, "KC", "BAL", 1, k1), (2025, "KC", "DEN", 2, k2)],
    )
    fake_db.respond("UPDATE players p SET team", rowcount=2)
    staging = [
        staged(1, {"gsis_id": "00-1", "season": 2025, "week": 1, "team": "KC",
                   "position": "QB", "depth_team": "1"}),
        # No week: resolved from the snapshot timestamp to the next KC game (week 2).
        staged(2, {"player_id": "00-2", "season": 2025, "team": "KC", "position": "WR",
                   "pos_rank": "2", "published_at": "2025-09-10T00:00:00Z"}),
        # Unparseable rank falls back to 1.0.
        staged(3, {"gsis_id": "00-3", "season": 2025, "week": 1, "team": "KC", "depth_team": "abc"}),
        # Same key as row 1 with a worse rank and no timestamp: row 1 stays.
        staged(4, {"gsis_id": "00-1", "season": 2025, "week": 1, "team": "KC", "depth_team": "3"}),
        # No week and no timestamp: cannot be placed, skipped.
        staged(5, {"gsis_id": "00-4", "season": 2025, "team": "KC"}),
        staged(6, {"season": 2025, "week": 1, "team": "KC"}),  # no player id
        # Same key as row 2: a later published_at replaces it, an earlier one does not.
        staged(7, {"gsis_id": "00-2", "season": 2025, "week": 2, "team": "KC", "depth_team": "5",
                   "published_at": "2025-09-13T00:00:00Z"}),
        staged(8, {"gsis_id": "00-2", "season": 2025, "week": 2, "team": "KC", "depth_team": "1",
                   "published_at": "2025-09-12T00:00:00Z"}),
    ]
    summary = NormalizeSummary()
    norm._process_depth_charts(staging, summary)

    rows = fake_db.values_for("INSERT INTO depth_charts")[0]
    by_key = {(r[0], r[2]): r for r in rows}
    assert len(rows) == 3
    assert by_key[("00-1", 1)][5] == 1.0
    assert by_key[("00-3", 1)][5] == 1.0
    assert by_key[("00-2", 2)][5] == 5.0  # the 2025-09-13 snapshot wins
    assert by_key[("00-2", 2)][6] == "2025-09-13T00:00:00Z"
    assert by_key[("00-2", 2)][7] == "nflreadpy.load_depth_charts"
    assert by_key[("00-1", 1)][7] is None  # no published_at: no source label

    assert summary.reconciled_players == 2
    assert summary.staging_rows_processed == 8
    reconcile = fake_db.statements_containing("UPDATE players p SET team")
    assert reconcile[0][1] == (2025,)
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4, 5, 6, 7, 8],)


def test_process_depth_charts_with_only_unplaceable_rows_marks_them_and_writes_nothing(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_depth_charts(
        [staged(9, {"season": 2025, "team": "KC"}), staged(10, {"gsis_id": "00-5", "season": 2025})],
        summary,
    )
    assert fake_db.values_for("INSERT INTO depth_charts") == []
    assert summary.staging_rows_processed == 2
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([9, 10],)


def test_process_depth_charts_empty_input_is_noop(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_depth_charts([], summary)
    assert fake_db.statements == []
    assert summary.staging_rows_processed == 0


def test_process_nextgen_stats_upserts_valid_rows_and_retires_all(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_nextgen_stats(
        [
            staged(1, {"player_id": "00-1", "season": 2024, "week": 1, "avg_separation": 2.8,
                       "avg_cushion": 6.1, "max_speed": 21.5,
                       "avg_completion_above_expectation": 1.2}),
            staged(2, {"gsis_id": "00-2", "season": 2024, "week": 1}),
            staged(3, {"player_id": "00-3", "season": 2024}),  # no week
            staged(4, {"season": 2024, "week": 1}),  # no player id
        ],
        summary,
    )
    rows = fake_db.values_for("INSERT INTO nextgen_stats")[0]
    assert rows == [
        ("00-1", 2024, 1, 2.8, 6.1, 21.5, 1.2),
        ("00-2", 2024, 1, None, None, None, None),
    ]
    assert summary.staging_rows_processed == 4
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4],)


def test_process_nextgen_stats_with_no_valid_rows_still_retires_inputs(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_nextgen_stats([staged(7, {"season": 2024})], summary)
    assert fake_db.values_for("INSERT INTO nextgen_stats") == []
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([7],)


def test_process_team_stats_upserts_totals_per_team_week(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_team_stats(
        [
            staged(1, {"team": "KC", "season": 2024, "week": 1, "attempts": 34, "carries": 22,
                       "total_plays": 62, "total_yards": 400.0}),
            staged(2, {"team": "KC", "season": 2024}),  # no week: skipped
        ],
        summary,
    )
    assert fake_db.values_for("INSERT INTO team_game_stats") == [[("KC", 2024, 1, 34, 22, 62, 400.0)]]
    assert summary.staging_rows_processed == 2


def test_process_team_stats_empty_input_is_noop(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_team_stats([], summary)
    assert fake_db.statements == []


def test_process_ftn_charting_maps_flags_and_skips_bad_play_ids(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_ftn_charting(
        [
            staged(1, {"game_id": "2024_01_KC_BAL", "play_id": "12", "season": 2024, "week": 1,
                       "is_drop": 1, "is_contested_ball": True, "is_catchable_ball": True,
                       "is_created_reception": False}),
            staged(2, {"nflverse_game_id": "2024_01_KC_BAL", "nflverse_play_id": 34,
                       "season": 2024, "week": 1}),
            staged(3, {"game_id": "2024_01_KC_BAL"}),  # no play id
            staged(4, {"game_id": "2024_01_KC_BAL", "play_id": "abc"}),  # not an int
        ],
        summary,
    )
    rows = fake_db.values_for("INSERT INTO ftn_play")[0]
    assert rows == [
        ("2024_01_KC_BAL", 12, 2024, 1, True, True, True, False),
        ("2024_01_KC_BAL", 34, 2024, 1, False, False, None, None),
    ]
    assert summary.staging_rows_processed == 4
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4],)


def test_process_ftn_charting_with_no_usable_rows_only_retires_inputs(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_ftn_charting([staged(3, {"game_id": "G"})], summary)
    assert fake_db.values_for("INSERT INTO ftn_play") == []
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([3],)


def test_process_participation_counts_routes_per_player_game(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_participation(
        [
            staged(1, {"game_id": "G1", "offense_players": "00-1;00-2", "season": 2023, "week": 1}),
            # Commas are accepted as separators and whitespace is stripped.
            staged(2, {"nflverse_game_id": "G1", "offense_players": "00-1, 00-3",
                       "season": 2023, "week": 1}),
            staged(3, {"game_id": "G2", "offense_players": ""}),  # empty: no routes
            staged(4, {"offense_players": "00-9"}),  # no game id
        ],
        summary,
    )
    rows = fake_db.values_for("INSERT INTO participation_player_game")[0]
    assert sorted(rows) == [
        ("00-1", "G1", 2023, 1, 2),
        ("00-2", "G1", 2023, 1, 1),
        ("00-3", "G1", 2023, 1, 1),
    ]
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4],)


def test_process_participation_with_no_routes_only_retires_inputs(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_participation([staged(3, {"game_id": "G2", "offense_players": ""})], summary)
    assert fake_db.values_for("INSERT INTO participation_player_game") == []
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([3],)


def test_process_combine_resolves_pfr_ids_and_reads_nflverse_aliases(norm, fake_db):
    fake_db.respond("SELECT pfr_id, id FROM players WHERE pfr_id IS NOT NULL", rows=[("PFR1", "00-1")])
    summary = NormalizeSummary()
    norm._process_combine(
        [
            staged(1, {"player_id": "00-2", "season": 2024, "forty": 4.41, "bench_press": 22,
                       "vertical_jump": 38.5, "broad_jump": 125}),
            staged(2, {"pfr_id": "PFR1", "season": 2024, "x40_yard": 4.5, "bench": 18,
                       "vertical": 36.0, "broad": 120}),
            staged(3, {"pfr_id": "UNKNOWN", "season": 2024}),  # pfr id not in players
            staged(4, {"player_id": "00-3"}),  # no season
        ],
        summary,
    )
    rows = fake_db.values_for("INSERT INTO combine")[0]
    assert rows == [
        ("00-2", 2024, 4.41, 22, 38.5, 125),
        ("00-1", 2024, 4.5, 18, 36.0, 120),
    ]
    assert summary.staging_rows_processed == 4
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([1, 2, 3, 4],)


def test_process_combine_with_no_usable_rows_still_counts_and_retires(norm, fake_db):
    fake_db.respond("SELECT pfr_id, id FROM players WHERE pfr_id IS NOT NULL", rows=[])
    summary = NormalizeSummary()
    norm._process_combine([staged(5, {"season": 2024})], summary)
    assert fake_db.values_for("INSERT INTO combine") == []
    assert summary.staging_rows_processed == 1
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([5],)


def test_process_combine_empty_input_is_noop(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_combine([], summary)
    assert fake_db.statements == []


def test_upsert_games_empty_is_noop(norm, fake_db):
    assert norm._upsert_games([]) == 0
    assert fake_db.values_calls == []


def test_process_snap_counts_with_no_name_matches_writes_no_updates(norm, fake_db):
    fake_db.respond(
        "SELECT id, full_name FROM players WHERE full_name IS NOT NULL",
        rows=[{"id": "00-1", "full_name": "Patrick Mahomes"}],
    )
    staging = [staged(1, {"game_id": "G", "season": 2024, "week": 1, "player": "Zzz Qqq",
                          "offense_snaps": 1, "offense_pct": 0.1})]
    summary = NormalizeSummary()
    norm._process_snap_counts(staging, summary)

    assert fake_db.statements_containing("UPDATE game_logs") == []
    assert summary.skipped == 1
    assert fake_db.rollbacks == 0


def test_process_snap_counts_logs_exact_and_fuzzy_match_totals(norm, fake_db, caplog):
    fake_db.respond(
        "SELECT id, full_name FROM players WHERE full_name IS NOT NULL",
        rows=[{"id": "00-1", "full_name": "Patrick Mahomes"},
              {"id": "00-2", "full_name": "Travis Kelce"}],
    )
    # Every UPDATE reports one row touched, so both matches count as applied.
    fake_db.respond("UPDATE game_logs", rowcount=1)
    base = {"game_id": "G", "season": 2024, "week": 1, "offense_snaps": 5, "offense_pct": 0.5}
    staging = [
        staged(1, {**base, "player": "Patrick Mahomes"}),  # exact
        staged(2, {**base, "player": "Travis Kelcee"}),  # fuzzy (WRatio 96)
    ]
    summary = NormalizeSummary()
    with caplog.at_level(logging.INFO, logger="pipeline.normalize"):
        norm._process_snap_counts(staging, summary)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "snap_counts: updated 2 game_log rows" in m and "1 via fuzzy fallback" in m
        for m in messages
    )


def test_process_team_stats_with_only_incomplete_rows_retires_them(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_team_stats([staged(4, {"team": "KC", "season": 2024})], summary)
    assert fake_db.values_for("INSERT INTO team_game_stats") == []
    assert summary.staging_rows_processed == 1
    assert fake_db.statements_containing("UPDATE staging_nflreadpy")[0][1] == ([4],)


def test_process_nextgen_stats_empty_input_is_noop(norm, fake_db):
    summary = NormalizeSummary()
    norm._process_nextgen_stats([], summary)
    assert fake_db.statements == []
    assert summary.staging_rows_processed == 0


def test_process_depth_charts_later_lower_rank_replaces_higher_rank(norm, fake_db):
    staging = [
        staged(1, {"gsis_id": "00-7", "season": 2025, "week": 3, "team": "KC", "depth_team": "4"}),
        staged(2, {"gsis_id": "00-7", "season": 2025, "week": 3, "team": "KC", "depth_team": "2"}),
    ]
    norm._process_depth_charts(staging, NormalizeSummary())
    rows = fake_db.values_for("INSERT INTO depth_charts")[0]
    assert [(r[0], r[5]) for r in rows] == [("00-7", 2.0)]


# ── run and dispatch ──────────────────────────────────────────────────────────

def test_run_visits_every_source_in_fk_safe_order(norm, fake_db):
    summary = norm.run(seasons=[2024])

    fetched = [params[0] for sql, params in fake_db.statements if "FROM staging_nflreadpy" in sql]
    assert fetched == FK_ORDER
    assert all(params[1] == [2024] for sql, params in fake_db.statements if "FROM staging_nflreadpy" in sql)
    assert summary == NormalizeSummary()
    assert norm._current_source_type is None


def test_run_loops_over_batches_until_a_source_is_empty(norm, monkeypatch):
    batches = {"schedules": [[staged(1, {}), staged(2, {})], [staged(3, {})], []]}
    fetch_calls: list[tuple[str, Any, Any]] = []
    dispatched: list[tuple[str, list[int], Any]] = []

    def fake_fetch(source_type, seasons=None, limit=None):
        fetch_calls.append((source_type, seasons, limit))
        queue = batches.get(source_type, [])
        return queue.pop(0) if queue else []

    def fake_dispatch(source_type, staging_rows, summary):
        dispatched.append((source_type, [r["id"] for r in staging_rows], norm._current_source_type))

    monkeypatch.setattr(norm, "_fetch_unprocessed", fake_fetch)
    monkeypatch.setattr(norm, "_dispatch", fake_dispatch)
    summary = norm.run()

    assert dispatched == [("schedules", [1, 2], "schedules"), ("schedules", [3], "schedules")]
    assert all(limit == Normalizer.STAGING_BATCH_SIZE for _, _, limit in fetch_calls)
    # schedules is fetched until empty (3 calls); the first-seen order is still FK-safe.
    assert list(dict.fromkeys(src for src, _, _ in fetch_calls)) == FK_ORDER
    assert summary == NormalizeSummary()


def test_dispatch_routes_each_source_type_to_its_processor(norm, monkeypatch):
    expected = {
        "schedules": "_process_schedules",
        "rosters": "_process_rosters",
        "player_stats": "_process_player_stats",
        "snap_counts": "_process_snap_counts",
        "depth_charts": "_process_depth_charts",
        "nextgen_stats": "_process_nextgen_stats",
        "team_stats": "_process_team_stats",
        "ftn_charting": "_process_ftn_charting",
        "participation": "_process_participation",
        "combine": "_process_combine",
    }
    hits: list[str] = []
    for source, method in expected.items():
        monkeypatch.setattr(norm, method, lambda rows, summary, _m=method: hits.append(_m))

    summary = NormalizeSummary()
    for source in expected:
        norm._dispatch(source, [], summary)
    norm._dispatch("unknown_source", [], summary)

    assert hits == list(expected.values())


# ── CLI entry point ───────────────────────────────────────────────────────────

def test_main_without_database_url_exits_nonzero(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["normalize.py"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_dry_run_counts_staging_rows_without_writing(monkeypatch, fake_db, capsys):
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setattr(sys, "argv", ["normalize.py", "--dry-run", "--seasons", "2024"])
    fake_db.respond("SELECT id, raw_data, ingested_at FROM staging_nflreadpy",
                    rows=[{"id": 1, "raw_data": {}, "ingested_at": None}] * 2)

    main()

    out = capsys.readouterr().out
    assert "schedules: 2 unprocessed rows in staging" in out
    assert "rosters: 2 unprocessed rows in staging" in out
    assert "player_stats: 2 unprocessed rows in staging" in out
    assert [params[0] for _, params in fake_db.statements] == ["schedules", "rosters", "player_stats"]
    assert all(params[1] == [2024] for _, params in fake_db.statements)
    assert fake_db.values_calls == []
    assert fake_db.statements_containing("DELETE") == []


def test_main_full_run_prints_the_normalize_summary(monkeypatch, fake_db, capsys):
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setattr(sys, "argv", ["normalize.py"])

    main()

    out = capsys.readouterr().out
    assert "NORMALIZE SUMMARY" in out
    assert "Teams upserted    : 0" in out


def test_main_verbose_raises_root_logger_to_debug(monkeypatch, fake_db, capsys):
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setattr(sys, "argv", ["normalize.py", "--verbose"])
    root = logging.getLogger()
    original = root.level
    try:
        main()
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original)
