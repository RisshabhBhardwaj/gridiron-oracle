"""scripts/backfill_history.py: column/season rules, comparison logic, and a two-database round trip."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backfill_history as bf  # noqa: E402


# ── pure rules ────────────────────────────────────────────────────────────────

def test_serial_id_is_never_copied_and_source_only_columns_are_reported():
    copy, source_only, target_only = bf.choose_columns(
        source_cols=["id", "player_id", "season", "old_col"],
        target_cols=["id", "player_id", "season", "new_col"],
        serial_cols=["id"],
        required_cols=["id", "player_id"],
    )
    assert copy == ["player_id", "season"]
    assert source_only == ["old_col"]
    assert target_only == ["new_col"]


def test_required_target_column_without_source_value_refuses():
    with pytest.raises(bf.BackfillError, match="no source value"):
        bf.choose_columns(["player_id"], ["player_id", "must_have"], [], ["player_id", "must_have"])


def test_season_bounds_stop_at_cap_and_before_target_start():
    assert bf.season_bounds("season", 2025, 2024) == [("<=", 2025), ("<", 2024)]
    assert bf.season_bounds("season", 2025, None) == [("<=", 2025)]
    assert bf.season_bounds(None, 2025, 2024) == []


def test_same_database_is_refused():
    with pytest.raises(bf.BackfillError, match="same database"):
        bf.assert_distinct_databases("postgresql://u:p@h:5432/db", "postgresql://u2:p2@h:5432/db")
    bf.assert_distinct_databases("postgresql://u@h/db_a", "postgresql://u@h/db_b")


def _avg(n=10, **values):
    return {"n": n, "kalman": 1.0, "team_off_elo": 1500.0, "elo_implied_win_prob": 0.5, **values}


def test_compare_flags_differences_and_ignores_rounding():
    source = {2023: _avg(), 2024: _avg()}
    target = {2023: _avg(elo_implied_win_prob=0.51), 2024: _avg(team_off_elo=1500.5, elo_implied_win_prob=0.52)}
    problems = bf.compare_averages(source, target)
    assert problems == []  # 1500.5 vs 1500 is inside the 1e-3 relative tolerance
    target[2024]["team_off_elo"] = 1700.0
    assert any("team_off_elo" in p for p in bf.compare_averages(source, target))


def test_compare_requires_the_elo_recompute_to_have_run():
    source = {2023: _avg(), 2024: _avg()}
    problems = bf.compare_averages(source, {2023: _avg(), 2024: _avg()})
    assert any("elo_implied_win_prob" in p and "recompute" in p for p in problems)


def test_compare_reports_row_count_and_missing_season():
    problems = bf.compare_averages({2023: _avg(n=5), 2024: _avg()}, {2023: _avg(n=6, elo_implied_win_prob=0.4)})
    assert any("row count" in p for p in problems)
    assert any("only one database" in p for p in problems)


# ── two real databases (skipped unless an admin URL is provided) ──────────────

ADMIN_URL = os.environ.get("BACKFILL_TEST_ADMIN_URL")
needs_db = pytest.mark.skipif(not ADMIN_URL, reason="BACKFILL_TEST_ADMIN_URL not set")

SCHEMA = """
CREATE TABLE players (id text PRIMARY KEY, full_name text NOT NULL);
CREATE TABLE game_logs (
    id serial PRIMARY KEY, player_id text NOT NULL, game_id text NOT NULL, season int NOT NULL, week int NOT NULL,
    UNIQUE (player_id, game_id));
CREATE TABLE feature_matrix (
    id serial PRIMARY KEY, player_id text NOT NULL, game_id text NOT NULL, season int NOT NULL, week int NOT NULL,
    snap_pct_off double precision, kalman double precision, UNIQUE (player_id, game_id));
"""


def _dsn(admin: str, name: str) -> str:
    parts = urlsplit(admin)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture
def two_dbs():
    import psycopg2

    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    tag = uuid.uuid4().hex[:8]
    names = {"src": f"bf_src_{tag}", "tgt": f"bf_tgt_{tag}"}
    cur = admin.cursor()
    for name in names.values():
        cur.execute(f'CREATE DATABASE "{name}"')
    urls = {k: _dsn(ADMIN_URL, v) for k, v in names.items()}
    for url in urls.values():
        conn = psycopg2.connect(url)
        conn.cursor().execute(SCHEMA)
        conn.commit()
        conn.close()
    try:
        yield urls
    finally:
        for name in names.values():
            cur.execute(f'DROP DATABASE IF EXISTS "{name}"')
        admin.close()


def _exec(url, statements):
    import psycopg2

    conn = psycopg2.connect(url)
    cur = conn.cursor()
    for statement in statements:
        cur.execute(statement)
    conn.commit()
    conn.close()


def _scalar(url, query):
    import psycopg2

    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute(query)
    value = cur.fetchone()[0]
    conn.close()
    return value


def _seed(two_dbs, *, leaked=False):
    _exec(two_dbs["src"], ["INSERT INTO players VALUES ('p1','A'),('p2','B')"] + [
        f"INSERT INTO game_logs (player_id, game_id, season, week) VALUES ('p1','g{y}',{y},1)" for y in (2022, 2023, 2024)
    ] + [
        f"INSERT INTO feature_matrix (player_id, game_id, season, week, snap_pct_off, kalman) "
        f"VALUES ('p1','g{y}',{y},1,{'0.7' if leaked and y == 2022 else 'NULL'},{y})" for y in (2022, 2023, 2024)
    ])
    # The target was cut: it has 2024 only, with its own (different) values and ids.
    _exec(two_dbs["tgt"], [
        "INSERT INTO players VALUES ('p1','A')",
        "INSERT INTO game_logs (player_id, game_id, season, week) VALUES ('p1','g2024',2024,1)",
        "INSERT INTO feature_matrix (player_id, game_id, season, week, kalman) VALUES ('p1','g2024',2024,1,999)",
    ])


@needs_db
def test_round_trip_fills_pre_cut_rows_only_and_is_idempotent(two_dbs):
    _seed(two_dbs)
    tables = ["players", "game_logs", "feature_matrix"]

    dry = bf.backfill(two_dbs["src"], two_dbs["tgt"], max_season=2025, tables=tables)
    assert {r.table: r.inserted for r in dry} == {"players": 1, "game_logs": 2, "feature_matrix": 2}
    assert _scalar(two_dbs["tgt"], "SELECT count(*) FROM feature_matrix") == 1  # rolled back

    bf.backfill(two_dbs["src"], two_dbs["tgt"], max_season=2025, tables=tables, apply=True)
    assert _scalar(two_dbs["tgt"], "SELECT count(*) FROM feature_matrix") == 3
    assert _scalar(two_dbs["tgt"], "SELECT count(*) FROM players") == 2
    # existing 2024 row untouched
    assert _scalar(two_dbs["tgt"], "SELECT kalman FROM feature_matrix WHERE season = 2024") == 999

    again = bf.backfill(two_dbs["src"], two_dbs["tgt"], max_season=2025, tables=tables, apply=True)
    assert all(r.inserted == 0 for r in again)
    assert _scalar(two_dbs["tgt"], "SELECT count(*) FROM game_logs") == 3


@needs_db
def test_max_season_caps_the_copy(two_dbs):
    _seed(two_dbs)
    _exec(two_dbs["src"], ["INSERT INTO game_logs (player_id, game_id, season, week) VALUES ('p1','g2020',2020,1)"])
    bf.backfill(two_dbs["src"], two_dbs["tgt"], max_season=2022, tables=["game_logs"], apply=True)
    assert _scalar(two_dbs["tgt"], "SELECT max(season) FROM game_logs WHERE season < 2024") == 2022


@needs_db
def test_snap_leak_in_the_source_blocks_the_commit(two_dbs):
    _seed(two_dbs, leaked=True)
    with pytest.raises(bf.BackfillError, match="snap_pct_off"):
        bf.backfill(two_dbs["src"], two_dbs["tgt"], max_season=2025, tables=["players", "feature_matrix"], apply=True)
    assert _scalar(two_dbs["tgt"], "SELECT count(*) FROM feature_matrix") == 1
