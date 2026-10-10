"""
backend/tests/test_ml_cov_verify_pipeline.py

Coverage tests for ml/verify_pipeline_setup.py (pipeline prerequisite checks).

Each check is exercised with stubs for MLflow, psycopg2 and the TFT prior
loader, plus a temporary ridge-coefficient directory. The module reports via
its logger, so assertions read caplog. main() is tested for its exit code
across the failure and warning branches.
"""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import ml.utils as ml_utils
from ml import verify_pipeline_setup as vps


@pytest.fixture(autouse=True)
def _info_logs(caplog):
    caplog.set_level(logging.INFO, logger=vps.logger.name)


# ── check_env ─────────────────────────────────────────────────────────────────

class TestCheckEnv:
    def test_missing_mlflow_uri_fails_first(self, monkeypatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://x")
        assert vps.check_env() == (False, None, None)

    def test_missing_database_url_fails_second(self, monkeypatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert vps.check_env() == (False, "http://mlflow:5000", None)

    def test_both_set_passes(self, monkeypatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
        assert vps.check_env() == (True, "http://mlflow:5000", "postgresql://u:p@h/db")


# ── MLflow checks ─────────────────────────────────────────────────────────────

class _FakeExperiment:
    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id


@pytest.fixture
def mlflow_fake(monkeypatch):
    """Records tracking URIs and serves experiments/runs from a dict."""
    import mlflow

    state = {
        "uri": None,
        "experiments": {},        # name -> experiment id, None (absent), or Exception to raise
        "finished_runs": set(),   # experiment ids that have a FINISHED run
        "search_experiments_error": None,
    }

    def set_uri(uri):
        state["uri"] = uri

    def search_experiments(max_results=None):
        if state["search_experiments_error"]:
            raise state["search_experiments_error"]
        return []

    def get_experiment_by_name(name):
        outcome = state["experiments"].get(name)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return _FakeExperiment(outcome)

    def search_runs(experiment_ids, max_results=None, filter_string=None):
        assert filter_string == "status = 'FINISHED'"
        if experiment_ids[0] in state["finished_runs"]:
            return pd.DataFrame([{"run_id": "r"}])
        return pd.DataFrame()

    monkeypatch.setattr(mlflow, "set_tracking_uri", set_uri)
    monkeypatch.setattr(mlflow, "search_experiments", search_experiments)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", get_experiment_by_name)
    monkeypatch.setattr(mlflow, "search_runs", search_runs)
    return state


class TestMlflowChecks:
    def test_reachable_when_search_succeeds(self, mlflow_fake, caplog) -> None:
        assert vps.check_mlflow_reachable("http://mlflow:5000") is True
        assert mlflow_fake["uri"] == "http://mlflow:5000"
        assert "MLflow server reachable" in caplog.text

    def test_unreachable_reports_error(self, mlflow_fake, caplog) -> None:
        mlflow_fake["search_experiments_error"] = ConnectionError("refused")
        assert vps.check_mlflow_reachable("http://mlflow:5000") is False
        assert "MLflow unreachable: refused" in caplog.text

    def test_experiment_status_per_stat_and_learner(self, mlflow_fake) -> None:
        mlflow_fake["experiments"] = {
            "xgb_receiving_yards": "10",    # exists and has a FINISHED run
            "lgbm_receiving_yards": "11",   # exists, no FINISHED run
            "tft_targets": RuntimeError("store unavailable"),  # lookup error -> False
        }
        mlflow_fake["finished_runs"] = {"10"}
        results = vps.check_experiments("http://mlflow:5000")

        assert mlflow_fake["uri"] == "http://mlflow:5000"
        assert set(results) == set(vps._STATS)
        assert set(results["receiving_yards"]) == set(vps._LEARNERS)
        assert results["receiving_yards"] == {"xgb": True, "lgbm": False, "catboost": False, "tft": False}
        assert results["targets"]["tft"] is False
        assert all(v is False for v in results["passing_tds"].values())


# ── Ridge coefficient file check ──────────────────────────────────────────────

class TestRidgeCoefs:
    def _write(self, directory, stat, payload) -> None:
        path = directory / f"ridge_{stat}_coefs.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))

    def test_complete_coefficients_pass(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(vps, "_OOF_DIR", tmp_path)
        full = {"xgb": 0.3, "lgbm": 0.2, "catboost": 0.3, "tft": 0.2, "intercept": 1.0}
        for stat in vps._STATS:
            self._write(tmp_path, stat, full)
        results = vps.check_ridge_coefs()
        assert all(results.values())
        assert set(results) == set(vps._STATS)

    def test_missing_learner_or_intercept_fails(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(vps, "_OOF_DIR", tmp_path)
        self._write(tmp_path, "receiving_yards", {"xgb": 1.0, "lgbm": 0.0, "catboost": 0.0, "tft": 0.0})
        self._write(tmp_path, "targets", {"xgb": 1.0, "lgbm": 0.0, "catboost": 0.0, "tft": 0.0, "intercept": 0.0})
        results = vps.check_ridge_coefs()
        assert results["receiving_yards"] is False  # no intercept
        assert results["targets"] is True

    def test_absent_and_corrupt_files_fail(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(vps, "_OOF_DIR", tmp_path)
        self._write(tmp_path, "fumbles", "{not json")
        results = vps.check_ridge_coefs()
        assert results["fumbles"] is False
        assert results["fantasy_ppr"] is False  # file never written


# ── Feature matrix and TFT prior checks ───────────────────────────────────────

class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql):
        self.sql = sql

    def fetchall(self):
        return self._rows


class _Connection:
    def __init__(self, rows, log):
        self._cursor = _Cursor(rows)
        self._log = log

    def cursor(self):
        return self._cursor

    def close(self):
        self._log["closed"] = True


@pytest.fixture
def psycopg_fake(monkeypatch):
    import psycopg2

    log: dict = {"dsn": None, "closed": False, "rows": [(2025, 3, 512)], "error": None}

    def connect(dsn):
        log["dsn"] = dsn
        if log["error"]:
            raise log["error"]
        return _Connection(log["rows"], log)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return log


class TestDatabaseChecks:
    def test_feature_matrix_with_rows_passes(self, psycopg_fake, caplog) -> None:
        assert vps.check_feature_matrix("postgresql+asyncpg://u:p@h/db") is True
        assert psycopg_fake["dsn"] == "postgresql://u:p@h/db"
        assert psycopg_fake["closed"] is True
        assert "512 rows for S2025W3" in caplog.text

    def test_empty_feature_matrix_fails(self, psycopg_fake, caplog) -> None:
        psycopg_fake["rows"] = []
        assert vps.check_feature_matrix("postgresql://u@h/db") is False
        assert "feature_matrix is empty" in caplog.text

    def test_feature_matrix_connection_error_fails(self, psycopg_fake, caplog) -> None:
        psycopg_fake["error"] = OSError("no route to host")
        assert vps.check_feature_matrix("postgresql://u@h/db") is False
        assert "feature_matrix check failed: no route to host" in caplog.text

    def test_tft_prior_rows_pass_when_loader_runs(self, monkeypatch, caplog) -> None:
        seen: dict = {}

        def loader(db_url, target_season, target_week, player_ids, position_filter=None):
            seen.update(db_url=db_url, season=target_season, week=target_week,
                        players=player_ids, position=position_filter)
            return pd.DataFrame()  # empty is acceptable: the check is that the loader works

        monkeypatch.setattr(ml_utils, "load_feature_matrix_prior", loader)
        assert vps.check_tft_prior_rows("postgresql://db") is True
        assert seen == {"db_url": "postgresql://db", "season": 2025, "week": 1,
                        "players": ["00-0031234"], "position": "WR"}
        assert "load_feature_matrix_prior works" in caplog.text

    def test_tft_prior_rows_fail_when_loader_raises(self, monkeypatch, caplog) -> None:
        def loader(*args, **kwargs):
            raise KeyError("feature_matrix")

        monkeypatch.setattr(ml_utils, "load_feature_matrix_prior", loader)
        assert vps.check_tft_prior_rows("postgresql://db") is False
        assert "load_feature_matrix_prior failed" in caplog.text


# ── Smoke test ────────────────────────────────────────────────────────────────

class TestSmokeTest:
    def _install_runner(self, monkeypatch, dry_run):
        import ml.train as train

        created: dict = {}

        class _Runner:
            def __init__(self, fast, mlflow_tracking_uri):
                created.update(fast=fast, uri=mlflow_tracking_uri)

            def dry_run(self, season, week, positions, stats):
                created.update(season=season, week=week, positions=positions, stats=stats)
                return dry_run()

        monkeypatch.setattr(train, "PipelineRunner", _Runner)
        return created

    def test_reports_projection_count(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        created = self._install_runner(monkeypatch, lambda: pd.DataFrame({"p": [1, 2, 3]}))
        assert vps.run_smoke_test() is True
        assert created == {"fast": True, "uri": "http://mlflow:5000", "season": 2025, "week": 1,
                           "positions": ["WR"], "stats": ["receiving_yards"]}
        assert "Dry-run produced 3 projections" in caplog.text

    def test_empty_dry_run_still_passes_with_warning(self, monkeypatch, caplog) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        created = self._install_runner(monkeypatch, lambda: pd.DataFrame())
        assert vps.run_smoke_test() is True
        assert created["uri"] == ""
        assert "Dry-run returned empty DataFrame" in caplog.text

    def test_exception_fails(self, monkeypatch, caplog) -> None:
        def boom():
            raise RuntimeError("cuda out of memory")

        self._install_runner(monkeypatch, boom)
        assert vps.run_smoke_test() is False
        assert "Dry-run failed: cuda out of memory" in caplog.text


# ── main() ────────────────────────────────────────────────────────────────────

def _all_green_experiments() -> dict:
    return {stat: {learner: True for learner in vps._LEARNERS} for stat in vps._STATS}


@pytest.fixture
def stub_checks(monkeypatch):
    """Replace every check in main() with a controllable stub."""
    state = {
        "env": (True, "http://mlflow:5000", "postgresql://db"),
        "reachable": True,
        "experiments": _all_green_experiments(),
        "ridge": {stat: True for stat in vps._STATS},
        "feature_matrix": True,
        "tft_prior": True,
        "smoke": True,
        "smoke_calls": 0,
    }
    monkeypatch.setattr(vps, "check_env", lambda: state["env"])
    monkeypatch.setattr(vps, "check_mlflow_reachable", lambda uri: state["reachable"])
    monkeypatch.setattr(vps, "check_experiments", lambda uri: state["experiments"])
    monkeypatch.setattr(vps, "check_ridge_coefs", lambda: state["ridge"])
    monkeypatch.setattr(vps, "check_feature_matrix", lambda db_url: state["feature_matrix"])
    monkeypatch.setattr(vps, "check_tft_prior_rows", lambda db_url: state["tft_prior"])

    def smoke():
        state["smoke_calls"] += 1
        return state["smoke"]

    monkeypatch.setattr(vps, "run_smoke_test", smoke)
    monkeypatch.setattr(sys, "argv", ["verify_pipeline_setup"])
    return state


class TestMain:
    def test_all_green_returns_zero(self, stub_checks, caplog) -> None:
        assert vps.main() == 0
        assert "All checks passed." in caplog.text
        assert "python -m ml.train --season 2025 --week 1" in caplog.text

    def test_missing_env_returns_one(self, stub_checks) -> None:
        stub_checks["env"] = (False, None, None)
        assert vps.main() == 1

    def test_unreachable_mlflow_returns_one(self, stub_checks) -> None:
        stub_checks["reachable"] = False
        assert vps.main() == 1

    def test_feature_matrix_failure_returns_one(self, stub_checks) -> None:
        stub_checks["feature_matrix"] = False
        assert vps.main() == 1

    def test_tft_prior_failure_returns_one(self, stub_checks) -> None:
        stub_checks["tft_prior"] = False
        assert vps.main() == 1

    def test_missing_models_and_ridge_are_warnings_only(self, stub_checks, caplog) -> None:
        stub_checks["experiments"]["targets"]["tft"] = False
        stub_checks["experiments"]["fumbles"] = {learner: False for learner in vps._LEARNERS}
        stub_checks["ridge"]["targets"] = False
        assert vps.main() == 0
        assert "targets: tft" in caplog.text
        assert "Missing or incomplete for: targets" in caplog.text
        assert "Some checks had warnings." in caplog.text

    def test_many_missing_stats_are_truncated_in_output(self, stub_checks, caplog) -> None:
        for stat in vps._STATS[:7]:
            stub_checks["experiments"][stat] = {learner: False for learner in vps._LEARNERS}
        assert vps.main() == 0
        assert "... and 2 more" in caplog.text

    def test_smoke_test_runs_only_with_flag(self, stub_checks, monkeypatch) -> None:
        vps.main()
        assert stub_checks["smoke_calls"] == 0
        monkeypatch.setattr(sys, "argv", ["verify_pipeline_setup", "--smoke-test"])
        assert vps.main() == 0
        assert stub_checks["smoke_calls"] == 1

    def test_smoke_test_failure_returns_one(self, stub_checks, monkeypatch) -> None:
        stub_checks["smoke"] = False
        monkeypatch.setattr(sys, "argv", ["verify_pipeline_setup", "--smoke-test"])
        assert vps.main() == 1
