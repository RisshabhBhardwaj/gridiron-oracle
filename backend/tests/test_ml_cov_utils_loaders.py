"""
backend/tests/test_ml_cov_utils_loaders.py

Coverage tests for the database loaders and ONNX helpers in ml/utils.py.

The feature-matrix queries run against an in-test psycopg2 fake (no socket is
opened). The leakage guard is exercised through the real contract check:
requesting a target-game field such as snap_pct_off must raise. ONNX export
and inference run for real into tmp_path, and the ONNX Runtime predictions are
compared with the native XGBoost booster.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import ml.utils as mu
from ml.utils import (
    FEATURE_COLS,
    MIN_PRIOR_GAMES,
    TARGET_COL_MAP,
    _compute_metrics,
    export_onnx,
    load_feature_matrix,
    load_feature_matrix_prior,
    run_onnx_inference,
)


# ── fake psycopg2 connection ──────────────────────────────────────────────────

class _Cursor:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._state["sql"].append(sql)
        self._state["params"].append(params)

    def fetchall(self):
        return list(self._state["rows"])


class _Conn:
    def __init__(self, state):
        self._state = state

    def cursor(self, cursor_factory=None):
        return _Cursor(self._state)

    def close(self):
        self._state["closed"] = True


@pytest.fixture
def db(monkeypatch):
    import psycopg2

    state = {"sql": [], "params": [], "rows": [], "closed": False, "dsn": None}

    def connect(dsn):
        state["dsn"] = dsn
        return _Conn(state)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return state


def _feature_row(player_id: str, season: int, week: int, games_played: float, position: str = "WR") -> dict:
    row = {
        "player_id": player_id, "game_id": f"{season}_{week:02d}_{player_id}", "season": season,
        "week": week, "position": position, "team": "KC",
    }
    for i, col in enumerate(FEATURE_COLS):
        row[col] = float((i % 7) + week)
    row["seas_games_played"] = games_played
    for target_col in TARGET_COL_MAP.values():
        row[target_col] = 10.0
    return row


# ── load_feature_matrix ───────────────────────────────────────────────────────

class TestLoadFeatureMatrix:
    def test_applies_pregame_eligibility_and_returns_all_columns(self, db) -> None:
        db["rows"] = [
            _feature_row("A", 2024, 2, games_played=1.0),
            _feature_row("B", 2024, 1, games_played=0.0),   # no prior game -> dropped
            _feature_row("C", 2024, 3, games_played=4.0),
        ]
        df = load_feature_matrix("postgresql+asyncpg://u@h/db", [2024], position_filter="WR")

        assert list(df["player_id"]) == ["A", "C"]
        assert MIN_PRIOR_GAMES == 1
        assert set(FEATURE_COLS) <= set(df.columns)
        assert db["dsn"] == "postgresql://u@h/db"
        assert db["closed"] is True
        sql, params = db["sql"][0], db["params"][0]
        assert "AND position = %s" in sql
        assert params == [[2024], "WR"]

    def test_no_position_filter_omits_clause(self, db) -> None:
        db["rows"] = [_feature_row("A", 2023, 4, games_played=2.0)]
        df = load_feature_matrix("postgresql://u@h/db", [2023])
        assert "position = %s" not in db["sql"][0]
        assert db["params"][0] == [[2023]]
        assert len(df) == 1

    def test_empty_result_gives_empty_frame_with_expected_columns(self, db) -> None:
        db["rows"] = []
        df = load_feature_matrix("postgresql://u@h/db", [2024])
        assert df.empty
        assert "player_id" in df.columns and "actual_fantasy_ppr" in df.columns

    def test_target_game_field_is_refused_by_contract(self, db) -> None:
        # snap_pct_off is a target-game participation field; it must never reach a model frame.
        db["rows"] = [_feature_row("A", 2024, 2, games_played=1.0)]
        with pytest.raises(AssertionError, match="prohibited feature input"):
            load_feature_matrix("postgresql://u@h/db", [2024], feature_cols=["snap_pct_off"])

    def test_missing_registered_column_is_refused(self, db) -> None:
        row = _feature_row("A", 2024, 2, games_played=1.0)
        del row[FEATURE_COLS[0]]
        db["rows"] = [row]
        with pytest.raises(AssertionError, match="missing registered features"):
            load_feature_matrix("postgresql://u@h/db", [2024])


# ── load_feature_matrix_prior ─────────────────────────────────────────────────

class TestLoadFeatureMatrixPrior:
    def test_no_players_short_circuits_without_query(self, db) -> None:
        assert load_feature_matrix_prior("postgresql://u@h/db", 2025, 1, []).empty
        assert db["sql"] == []

    def test_prior_rows_query_uses_strict_before_target_bound(self, db) -> None:
        db["rows"] = [
            _feature_row("A", 2024, 17, games_played=16.0),
            _feature_row("A", 2025, 0, games_played=0.0),   # no prior game -> dropped
        ]
        df = load_feature_matrix_prior("postgresql://u@h/db", 2025, 1, ["A"], position_filter="WR")

        assert list(df["season"]) == [2024]
        sql, params = db["sql"][0], db["params"][0]
        assert "(season < %s) OR (season = %s AND week < %s)" in sql
        assert "AND position = %s" in sql
        assert params == [["A"], 2025, 2025, 1, "WR"]

    def test_prior_loader_also_refuses_forbidden_fields(self, db) -> None:
        # The prior loader checks the full FEATURE_COLS list, so a row must carry them all.
        db["rows"] = [_feature_row("A", 2024, 2, games_played=1.0)]
        monkey_cols = list(FEATURE_COLS) + ["snap_pct_off"]
        original = list(mu.FEATURE_COLS)
        try:
            mu.FEATURE_COLS[:] = monkey_cols
            with pytest.raises(AssertionError, match="prohibited feature input"):
                load_feature_matrix_prior("postgresql://u@h/db", 2025, 1, ["A"])
        finally:
            mu.FEATURE_COLS[:] = original


# ── _compute_metrics NaN handling ─────────────────────────────────────────────

def test_compute_metrics_drops_paired_nans() -> None:
    y_true = np.array([1.0, np.nan, 3.0, 5.0])
    y_pred = np.array([2.0, 4.0, np.nan, 5.0])
    mae, rmse = _compute_metrics(y_true, y_pred)
    # Only rows 0 and 3 survive: errors are 1 and 0.
    assert mae == pytest.approx(0.5)
    assert rmse == pytest.approx(math.sqrt(0.5))


def test_compute_metrics_all_nan_returns_nan_pair() -> None:
    mae, rmse = _compute_metrics(np.array([np.nan]), np.array([1.0]))
    assert math.isnan(mae) and math.isnan(rmse)


# ── export_onnx and run_onnx_inference ────────────────────────────────────────

class _FakeCatBoost:
    def __init__(self, raise_on_save=None):
        self.saved: list = []
        self._raise = raise_on_save

    def save_model(self, path, format=None, export_parameters=None):
        if self._raise:
            raise self._raise
        self.saved.append((path, format, export_parameters))
        with open(path, "wb") as fh:
            fh.write(b"fake-onnx")


def test_export_onnx_none_model_returns_none(tmp_path) -> None:
    assert export_onnx(None, ["a"], "xgb", "receiving_yards", "WR", onnx_dir=tmp_path) is None


def test_export_onnx_catboost_uses_native_onnx_writer(tmp_path) -> None:
    model = _FakeCatBoost()
    path = export_onnx(model, ["a", "b"], "catboost", "receiving_yards", "WR", onnx_dir=tmp_path / "out")
    assert path == tmp_path / "out" / "catboost_receiving_yards_WR.onnx"
    assert path.read_bytes() == b"fake-onnx"
    assert model.saved == [(str(path), "onnx", {"prediction_type": "RawFormulaVal"})]


def test_export_onnx_catboost_failure_is_logged_and_returns_none(tmp_path, caplog) -> None:
    model = _FakeCatBoost(raise_on_save=RuntimeError("serializer crashed"))
    with caplog.at_level(logging.WARNING, logger=mu.logger.name):
        assert export_onnx(model, ["a"], "catboost", "fantasy_ppr", "all", onnx_dir=tmp_path) is None
    assert "ONNX export failed for catboost/fantasy_ppr/all: serializer crashed" in caplog.text


def test_export_onnx_unsupported_learner_returns_none(tmp_path) -> None:
    # onnx and onnxmltools are installed here, so the learner check is what rejects 'tft'.
    assert export_onnx(object(), ["a"], "tft", "receiving_yards", "WR", onnx_dir=tmp_path) is None


def test_xgboost_onnx_round_trip_matches_native_predictions(tmp_path) -> None:
    rng = np.random.default_rng(11)
    features = [f"f{i}" for i in range(4)]
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 2] + rng.normal(0, 0.05, 200)
    model = xgb.XGBRegressor(n_estimators=25, max_depth=3, learning_rate=0.3)
    model.fit(pd.DataFrame(X, columns=features), y)

    path = export_onnx(model, features, "xgb", "receiving_yards", "WR", onnx_dir=tmp_path)
    assert path is not None and path.exists()
    assert path.name == "xgb_receiving_yards_WR.onnx"

    onnx_pred = run_onnx_inference(path, X[:20])
    native_pred = model.predict(X[:20])
    assert onnx_pred is not None
    assert onnx_pred.shape == (20,)
    np.testing.assert_allclose(onnx_pred, native_pred, rtol=1e-3, atol=1e-3)


def test_run_onnx_inference_failure_returns_none(tmp_path, caplog) -> None:
    bogus = tmp_path / "not_a_model.onnx"
    bogus.write_bytes(b"\x00\x01 not protobuf")
    with caplog.at_level(logging.WARNING, logger=mu.logger.name):
        assert run_onnx_inference(bogus, np.zeros((2, 3), dtype=np.float32)) is None
    assert "ONNX inference failed (not_a_model.onnx)" in caplog.text
