"""
backend/tests/test_ml_cov_backend_kalman.py

Coverage tests for backend/app/services/kalman.py (Kalman form estimates for
the API layer).

psycopg2.connect is replaced by an in-test fake connection, so no database is
contacted. The recompute fallback runs the real ml.kalman_tracker engine on a
tiny prior-game list and checks structural invariants (the estimate is finite,
std is the square root of the variance), not a fitted value.
"""

from __future__ import annotations

import math

import pytest
from fastapi import HTTPException

import ml.kalman_tracker as kt
from backend.app.services import kalman as kalman_mod
from backend.app.services.kalman import (
    VALID_STATS,
    KalmanService,
)


class _FakeCursor:
    def __init__(self, state):
        self._state = state

    def execute(self, sql, params=None):
        self._state["executed"].append((sql, params))
        if self._state.get("error_on_execute"):
            raise self._state["error_on_execute"]

    def fetchone(self):
        return self._state["one"]

    def fetchall(self):
        return self._state["all"]


class _FakeConn:
    def __init__(self, state):
        self._state = state

    def cursor(self):
        return _FakeCursor(self._state)

    def close(self):
        self._state["closed"] = True


@pytest.fixture
def db(monkeypatch):
    """Fake psycopg2.connect. Set state['one'], state['all'] or state['connect_error'] per test."""
    import psycopg2

    state = {"executed": [], "closed": False, "one": None, "all": [], "connect_error": None}

    def connect(dsn):
        state["dsn"] = dsn
        if state["connect_error"]:
            raise state["connect_error"]
        return _FakeConn(state)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return state


def test_valid_stats_whitelist_contains_core_stats() -> None:
    assert {"receiving_yards", "rushing_yards", "passing_yards", "targets", "fantasy_ppr"} <= VALID_STATS
    assert "drop_me_table" not in VALID_STATS


# ── get_estimate ──────────────────────────────────────────────────────────────

class TestGetEstimate:
    def test_invalid_stat_is_rejected_before_any_query(self, db) -> None:
        service = KalmanService("postgresql://db")
        with pytest.raises(HTTPException) as exc:
            service.get_estimate("00-1", "drop_tables; --", week=3, season=2025)
        assert exc.value.status_code == 422
        assert db["executed"] == []

    def test_reads_feature_matrix_row_and_computes_std(self, db) -> None:
        db["one"] = (72.5, 25.0)
        est = KalmanService("postgresql://db").get_estimate("00-1", "receiving_yards", week=4, season=2025)

        assert est is not None
        assert est.estimate == pytest.approx(72.5)
        assert est.variance == pytest.approx(25.0)
        assert est.std == pytest.approx(5.0)
        assert (est.player_id, est.stat, est.week, est.season) == ("00-1", "receiving_yards", 4, 2025)
        sql, params = db["executed"][0]
        assert "kalman_est_receiving_yards" in sql and "kalman_variance_receiving_yards" in sql
        assert params == ["00-1", 4, 2025]
        assert db["closed"] is True

    def test_null_variance_defaults_to_100(self, db) -> None:
        db["one"] = (10.0, None)
        est = KalmanService("postgresql://db").get_estimate("00-1", "targets", week=1, season=2025)
        assert est.variance == pytest.approx(100.0)
        assert est.std == pytest.approx(10.0)

    def test_missing_row_recomputes_from_game_logs(self, db, monkeypatch) -> None:
        db["one"] = None
        sentinel = object()
        service = KalmanService("postgresql://db")
        monkeypatch.setattr(service, "_recompute", lambda *args: sentinel)
        assert service.get_estimate("00-1", "targets", week=2, season=2025) is sentinel

    def test_null_estimate_recomputes(self, db, monkeypatch) -> None:
        db["one"] = (None, 4.0)
        service = KalmanService("postgresql://db")
        calls: list = []
        monkeypatch.setattr(service, "_recompute", lambda *args: calls.append(args) or None)
        assert service.get_estimate("00-1", "targets", week=2, season=2025) is None
        assert calls == [("00-1", "targets", 2, 2025)]

    def test_database_error_returns_none(self, db) -> None:
        db["connect_error"] = OSError("db down")
        assert KalmanService("postgresql://db").get_estimate("00-1", "targets", 1, 2025) is None


# ── get_trajectory ────────────────────────────────────────────────────────────

class TestGetTrajectory:
    def test_invalid_stat_is_rejected(self, db) -> None:
        with pytest.raises(HTTPException) as exc:
            KalmanService("postgresql://db").get_trajectory("00-1", "not_a_stat", 2025)
        assert exc.value.status_code == 422

    def test_skips_weeks_without_estimates_and_defaults_variance(self, db) -> None:
        db["all"] = [
            (1, 50.0, 20.0),
            (2, None, None),      # no estimate for week 2: dropped entirely
            (3, 60.0, None),      # null variance: defaults to 100
            (4, 65.5, 9.0),
        ]
        traj = KalmanService("postgresql://db").get_trajectory("00-1", "rushing_yards", 2025)
        assert traj.weeks == [1, 3, 4]
        assert traj.estimates == pytest.approx([50.0, 60.0, 65.5])
        assert traj.variances == pytest.approx([20.0, 100.0, 9.0])
        assert (traj.player_id, traj.stat, traj.season) == ("00-1", "rushing_yards", 2025)
        assert "kalman_est_rushing_yards" in db["executed"][0][0]

    def test_database_error_gives_empty_trajectory(self, db) -> None:
        db["connect_error"] = OSError("db down")
        traj = KalmanService("postgresql://db").get_trajectory("00-1", "targets", 2025)
        assert traj.weeks == [] and traj.estimates == [] and traj.variances == []


# ── _recompute ────────────────────────────────────────────────────────────────

class TestRecompute:
    def test_invalid_stat_returns_none_without_query(self, db) -> None:
        assert KalmanService("postgresql://db")._recompute("00-1", "bogus", 5, 2025) is None
        assert db["executed"] == []

    def test_no_prior_games_returns_none(self, db) -> None:
        db["all"] = []
        assert KalmanService("postgresql://db")._recompute("00-1", "receiving_yards", 1, 2025) is None
        sql, params = db["executed"][0]
        assert "FROM   game_logs" in sql
        assert params == ["00-1", 2025, 1]

    def test_runs_kalman_engine_on_prior_rows(self, db) -> None:
        db["all"] = [(10.0,), (20.0,), (None,), (30.0,)]
        est = KalmanService("postgresql://db")._recompute("00-1", "receiving_yards", 5, 2025)

        assert est is not None
        assert est.week == 5 and est.season == 2025 and est.stat == "receiving_yards"
        assert math.isfinite(est.estimate)
        assert est.variance > 0
        assert est.std == pytest.approx(math.sqrt(est.variance))

    def test_missing_estimate_key_returns_none(self, db, monkeypatch) -> None:
        db["all"] = [(7.0,)]
        monkeypatch.setattr(kt.KalmanFeatureEngineer, "compute_kalman_form", lambda self, rows, position=None: {})
        assert KalmanService("postgresql://db")._recompute("00-1", "targets", 3, 2025) is None

    def test_variance_defaults_to_100_when_engine_omits_it(self, db, monkeypatch) -> None:
        db["all"] = [(7.0,)]
        monkeypatch.setattr(
            kt.KalmanFeatureEngineer,
            "compute_kalman_form",
            lambda self, rows, position=None: {"kalman_est_targets": 6.0},
        )
        est = KalmanService("postgresql://db")._recompute("00-1", "targets", 3, 2025)
        assert est.estimate == pytest.approx(6.0)
        assert est.variance == pytest.approx(100.0)
        assert est.std == pytest.approx(10.0)

    def test_fumbles_use_sentinel_source_column(self, db, monkeypatch) -> None:
        seen: dict = {}

        def capture(self, rows, position=None):
            seen["rows"] = rows
            return {"kalman_est_fumbles": 0.5, "kalman_variance_fumbles": 1.0}

        db["all"] = [(1.0,)]
        monkeypatch.setattr(kt.KalmanFeatureEngineer, "compute_kalman_form", capture)
        est = KalmanService("postgresql://db")._recompute("00-1", "fumbles", 2, 2025)
        assert seen["rows"] == [{"_fumbles_combined": 1.0}]
        assert est.estimate == pytest.approx(0.5)
        assert est.std == pytest.approx(1.0)

    def test_null_source_rows_become_zero(self, db, monkeypatch) -> None:
        seen: dict = {}

        def capture(self, rows, position=None):
            seen["rows"] = rows
            return {"kalman_est_targets": 1.0, "kalman_variance_targets": 4.0}

        db["all"] = [(None,), (3.0,)]
        monkeypatch.setattr(kt.KalmanFeatureEngineer, "compute_kalman_form", capture)
        KalmanService("postgresql://db")._recompute("00-1", "targets", 3, 2025)
        assert seen["rows"] == [{"targets": 0.0}, {"targets": 3.0}]

    def test_database_error_returns_none(self, db) -> None:
        db["connect_error"] = OSError("db down")
        assert KalmanService("postgresql://db")._recompute("00-1", "targets", 3, 2025) is None


def test_module_exports_service_class() -> None:
    assert kalman_mod.KalmanService is KalmanService
