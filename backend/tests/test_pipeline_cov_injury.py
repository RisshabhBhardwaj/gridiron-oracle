"""
Offline coverage tests for pipeline/injury_pipeline.py.

Injury transforms run on small pandas frames. The DB writers use FakeConn
through the psycopg2.extras helpers. nflreadpy.load_injuries is replaced
per season. No network, no DB.
"""

from __future__ import annotations

import logging
import sys

import nflreadpy
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import pytest

import pipeline.injury_pipeline as inj
from backend.tests._pipeline_fakes import FakeConn, FakeConnFactory


def _raw(rows: list[dict]) -> pd.DataFrame:
    """Build a frame using nflreadpy.load_injuries column names."""
    base = {"gsis_id": None, "full_name": None, "season": 2024, "week": 1, "team": "KC",
            "position": "WR", "report_status": "Full", "practice_status": None,
            "primary_injury": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def _player_timeline(pid: str, statuses: list[tuple[int, str]], season: int = 2024) -> list[dict]:
    return [{"gsis_id": pid, "season": season, "week": w, "report_status": s} for w, s in statuses]


def _by_week(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    return df[df["player_id"] == pid].set_index("week")


# ── _process_injuries: streaks, time to return, event types ───────────────────

def test_process_injuries_computes_streak_return_and_event_types():
    raw = _raw(_player_timeline("00-1", [(1, "Full"), (2, "Out"), (3, "IR"), (4, "Full")]))
    out = _by_week(inj._process_injuries(raw), "00-1")

    assert list(out["games_missed_streak"]) == [0, 0, 1, 2]
    assert list(out["event_type"]) == ["active", "injury_start", "injury_ongoing", "return"]
    assert list(out["is_out"]) == [False, True, True, False]
    # Weeks until the next non-out game: week 2 -> week 4 is 2; week 3 -> week 4 is 1.
    assert out.loc[2, "time_to_return"] == 2
    assert out.loc[3, "time_to_return"] == 1
    assert pd.isna(out.loc[1, "time_to_return"])
    assert pd.isna(out.loc[4, "time_to_return"])


def test_process_injuries_sorts_unordered_input_before_streaks():
    raw = _raw(_player_timeline("00-1", [(4, "Full"), (2, "Out"), (1, "Full"), (3, "IR")]))
    out = _by_week(inj._process_injuries(raw), "00-1")
    assert list(out.index) == [1, 2, 3, 4]
    assert list(out["event_type"]) == ["active", "injury_start", "injury_ongoing", "return"]


def test_process_injuries_leaves_time_to_return_null_when_player_never_returns():
    raw = _raw(_player_timeline("00-2", [(10, "Out"), (11, "Out")]))
    out = _by_week(inj._process_injuries(raw), "00-2")
    assert out["event_type"].tolist() == ["injury_start", "injury_ongoing"]
    assert out["time_to_return"].isna().all()  # right-censored at season end


def test_process_injuries_falls_back_to_row_distance_when_week_gap_is_not_positive():
    # Week 5 of 2023 is out; the next active row is week 1 of 2024, so the week
    # difference is negative and the code counts row distance (1) instead.
    raw = _raw([
        {"gsis_id": "00-3", "season": 2023, "week": 5, "report_status": "Out"},
        {"gsis_id": "00-3", "season": 2024, "week": 1, "report_status": "Full"},
    ])
    out = inj._process_injuries(raw)
    out = out[out["player_id"] == "00-3"].set_index("season")
    assert out.loc[2023, "time_to_return"] == 1
    assert pd.isna(out.loc[2024, "time_to_return"])


def test_process_injuries_keeps_last_row_per_player_season_week():
    raw = _raw([
        {"gsis_id": "00-4", "season": 2024, "week": 1, "team": "KC"},
        {"gsis_id": "00-4", "season": 2024, "week": 1, "team": "BAL"},
    ])
    out = inj._process_injuries(raw)
    assert len(out) == 1
    assert out.iloc[0]["team"] == "BAL"


def test_process_injuries_fills_missing_source_columns_with_none():
    raw = pd.DataFrame([{"gsis_id": "00-5", "season": 2024, "week": 2, "report_status": "Full"}])
    out = inj._process_injuries(raw)
    assert out.iloc[0]["practice_status"] is None
    assert out.iloc[0]["injury_type"] is None
    assert out.iloc[0]["event_type"] == "active"


def test_process_injuries_casts_season_and_week_to_int():
    raw = _raw(_player_timeline("00-6", [(3, "Full")], season=2023))
    raw["season"] = raw["season"].astype(float)
    raw["week"] = raw["week"].astype(float)
    out = inj._process_injuries(raw)
    assert out.iloc[0]["season"] == 2023 and isinstance(out.iloc[0]["season"], (int, np.integer))
    assert out.iloc[0]["week"] == 3


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="_process_injuries assigns a per-row streak list built from groupby(player_id), "
           "which drops NaN keys, so a null gsis_id row makes the lengths disagree.",
)
def test_process_injuries_drops_rows_without_a_player_id():
    raw = _raw([
        {"gsis_id": None, "season": 2024, "week": 1, "report_status": "Out"},
        *_player_timeline("00-7", [(1, "Full")]),
    ])
    out = inj._process_injuries(raw)
    assert out["player_id"].tolist() == ["00-7"]


# ── _load_injuries ────────────────────────────────────────────────────────────

def test_load_injuries_concatenates_seasons_and_skips_empty_and_failed_ones(monkeypatch, caplog):
    seen_seasons: list[int] = []

    class _PolarsLike:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    def fake_load(seasons):
        season = seasons[0]
        seen_seasons.append(season)
        if season == 2010:
            raise RuntimeError("upstream 503")
        if season == 2011:
            return pd.DataFrame()  # empty season: skipped
        if season == 2012:
            return _PolarsLike(pd.DataFrame({"gsis_id": ["00-12"], "season": [2012]}))
        return pd.DataFrame({"gsis_id": [f"00-{season}"], "season": [season]})

    monkeypatch.setattr(nflreadpy, "load_injuries", fake_load)
    with caplog.at_level(logging.WARNING, logger="injury_pipeline"):
        df = inj._load_injuries()

    assert seen_seasons == inj.INJURY_SEASONS
    assert len(df) == len(inj.INJURY_SEASONS) - 2  # 2010 failed, 2011 empty
    assert "00-12" in set(df["gsis_id"])
    assert "2010" in " ".join(r.getMessage() for r in caplog.records)


def test_load_injuries_returns_empty_frame_when_every_season_fails(monkeypatch, caplog):
    def always_fail(seasons):
        raise RuntimeError("offline")

    monkeypatch.setattr(nflreadpy, "load_injuries", always_fail)
    with caplog.at_level(logging.ERROR, logger="injury_pipeline"):
        df = inj._load_injuries()

    assert df.empty
    assert any("No injury data could be loaded" in r.getMessage() for r in caplog.records)


# ── _write_to_db ──────────────────────────────────────────────────────────────

def test_write_to_db_maps_rows_and_nulls_missing_return_weeks(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(psycopg2.extras, "execute_values", conn.execute_values)
    df = pd.DataFrame([
        {"player_id": "00-1", "full_name": "A One", "season": 2024, "week": 2, "team": "KC",
         "position": "WR", "report_status": "Out", "practice_status": None, "injury_type": "Knee",
         "is_out": True, "games_missed_streak": 0, "time_to_return": 2.0, "event_type": "injury_start"},
        {"player_id": "00-2", "full_name": None, "season": 2024, "week": 3, "team": None,
         "position": None, "report_status": "Full", "practice_status": None, "injury_type": None,
         "is_out": False, "games_missed_streak": 1, "time_to_return": np.nan, "event_type": "return"},
    ])
    n = inj._write_to_db(conn, df)

    assert n == 2
    assert conn.commits == 1
    rows = conn.values_for("INSERT INTO injury_history")[0]
    assert rows[0] == ("00-1", "A One", 2024, 2, "KC", "WR", "Out", None, "Knee",
                       True, 0, 2, "injury_start")
    assert rows[1] == ("00-2", None, 2024, 3, None, None, "Full", None, None,
                       False, 1, None, "return")


def test_write_to_db_uses_upsert_on_the_composite_key(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(psycopg2.extras, "execute_values", conn.execute_values)
    df = inj._process_injuries(_raw(_player_timeline("00-1", [(1, "Full")])))
    inj._write_to_db(conn, df)
    sql, _ = conn.values_calls[0]
    assert "ON CONFLICT (player_id, season, week) DO UPDATE SET" in sql


# ── _update_feature_matrix_streaks ────────────────────────────────────────────

def test_update_streaks_batches_only_rows_with_a_positive_streak(monkeypatch):
    conn = FakeConn()
    batches: list[tuple[str, list[dict]]] = []

    def fake_execute_batch(cur, sql, argslist, page_size=100):
        batches.append((" ".join(sql.split()), list(argslist)))

    monkeypatch.setattr(psycopg2.extras, "execute_batch", fake_execute_batch)
    df = pd.DataFrame([
        {"player_id": "00-1", "season": 2024, "week": 1, "games_missed_streak": 0},
        {"player_id": "00-1", "season": 2024, "week": 2, "games_missed_streak": 1},
        {"player_id": "00-1", "season": 2024, "week": 3, "games_missed_streak": 2},
    ])
    inj._update_feature_matrix_streaks(conn, df)

    assert conn.commits == 2  # before the update and after it
    assert len(batches) == 1
    sql, batch = batches[0]
    assert sql.startswith("UPDATE feature_matrix SET games_missed_streak")
    assert batch == [
        {"player_id": "00-1", "season": 2024, "week": 2, "streak": 1},
        {"player_id": "00-1", "season": 2024, "week": 3, "streak": 2},
    ]
    assert all(isinstance(r["season"], int) and isinstance(r["streak"], int) for r in batch)


def test_update_streaks_with_no_streaks_writes_nothing(monkeypatch):
    conn = FakeConn()
    called: list[int] = []
    monkeypatch.setattr(psycopg2.extras, "execute_batch", lambda *a, **k: called.append(1))
    df = pd.DataFrame([{"player_id": "00-1", "season": 2024, "week": 1, "games_missed_streak": 0}])
    inj._update_feature_matrix_streaks(conn, df)
    assert called == []
    assert conn.commits == 1


# ── main and connection ───────────────────────────────────────────────────────

def test_main_writes_history_then_streaks_and_closes_connection(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(inj, "_get_conn", lambda: conn)
    monkeypatch.setattr(inj, "_load_injuries", lambda: _raw(
        _player_timeline("00-1", [(1, "Out"), (2, "Full")])
    ))
    monkeypatch.setattr(psycopg2.extras, "execute_values", conn.execute_values)
    batches: list[list[dict]] = []
    monkeypatch.setattr(
        psycopg2.extras, "execute_batch",
        lambda cur, sql, argslist, page_size=100: batches.append(list(argslist)),
    )

    inj.main()

    assert len(conn.values_for("INSERT INTO injury_history")[0]) == 2
    assert batches == [[{"player_id": "00-1", "season": 2024, "week": 2, "streak": 1}]]
    assert conn.closed is True


def test_main_still_closes_connection_when_loading_fails(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(inj, "_get_conn", lambda: conn)

    def explode():
        raise RuntimeError("load failed")

    monkeypatch.setattr(inj, "_load_injuries", explode)
    with pytest.raises(RuntimeError, match="load failed"):
        inj.main()
    assert conn.closed is True
    assert conn.values_calls == []


def test_get_conn_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL not set"):
        inj._get_conn()


def test_get_conn_strips_asyncpg_driver_before_connecting(monkeypatch):
    conn = FakeConn()
    factory = FakeConnFactory(conn)
    monkeypatch.setattr(psycopg2, "connect", factory)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db.test/oracle")

    assert inj._get_conn() is conn
    assert conn.connect_dsns == ["postgresql://u:p@db.test/oracle"]
