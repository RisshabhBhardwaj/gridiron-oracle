"""
backend/tests/test_ml_cov_shap_service.py

Coverage tests for ml/shap_service.py (SHAP attributions for the /explain API).

Real XGBoost artifacts are used for the model path. MLflow and the network are
replaced: the reachability probe is stubbed at socket.create_connection and the
MLflow client is a fake. The synthetic fallback is checked for its invariants
(ordering, labels, skipped inputs, bounds), not exact values, because its RNG
is seeded from the built-in hash() of the stat and position.
"""

from __future__ import annotations

import contextlib
import socket
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import ml.shap_service as svc
from backend.app.core.runtime_mode import ArtifactRequiredError
from ml.shap_service import (
    FEATURE_LABELS,
    FactorAttribution,
    SHAPService,
    label,
)

# Sorted, because the model path builds its frame from sorted numeric keys.
MODEL_FEATURES = ["kalman_est_receiving_yards", "seas_avg_targets", "temp_f"]


@pytest.fixture(scope="module")
def trained_model():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({
        "kalman_est_receiving_yards": rng.normal(60, 15, 300),
        "seas_avg_targets": rng.normal(6, 2, 300),
        "temp_f": rng.normal(60, 10, 300),
    })
    y = 1.2 * X["kalman_est_receiving_yards"] + 4.0 * X["seas_avg_targets"] + 0.02 * X["temp_f"]
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.2)
    model.fit(X[MODEL_FEATURES], y)
    return model.get_booster()


@pytest.fixture(autouse=True)
def _keep_mlflow_global_uri_untouched(monkeypatch):
    # _load_model() calls mlflow.set_tracking_uri(); stub it so the process-wide URI never changes.
    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)


@pytest.fixture
def allow_fallbacks(monkeypatch):
    monkeypatch.setattr(svc, "fallbacks_allowed", lambda: True)


# ── label() ───────────────────────────────────────────────────────────────────

class TestLabel:
    def test_known_feature_uses_curated_text(self) -> None:
        assert label("temp_f") == FEATURE_LABELS["temp_f"]
        assert label("kalman_est_receiving_yards") == "Kalman estimated receiving yards (filtered ability)"

    def test_unknown_feature_is_title_cased(self) -> None:
        assert label("new_rookie_signal") == "New Rookie Signal"

    def test_every_label_is_a_plain_sentence_without_trailing_period(self) -> None:
        for key, text in FEATURE_LABELS.items():
            assert text[0].isupper(), key
            assert not text.endswith("."), key


# ── explain(): model artifact path ────────────────────────────────────────────

class TestExplainWithModelArtifact:
    def test_uses_cached_model_and_reports_source(self, trained_model, allow_fallbacks) -> None:
        service = SHAPService()
        service._model_cache["receiving_yards_WR"] = trained_model
        features = {
            "kalman_est_receiving_yards": 72.0,
            "seas_avg_targets": 7.0,
            "temp_f": 58.0,
            "player_id": "00-0033106",   # non-numeric, must be ignored
            "position": "TE",            # non-numeric, must be ignored
        }
        result = service.explain(features, stat="receiving_yards", position="WR", top_n=3)

        assert result.source == "model_artifact"
        assert {a.feature for a in result.attributions} == set(MODEL_FEATURES)
        values = {a.feature: a.value for a in result.attributions}
        assert values == {"kalman_est_receiving_yards": 72.0, "seas_avg_targets": 7.0, "temp_f": 58.0}

        impacts = [abs(a.impact) for a in result.attributions]
        assert impacts == sorted(impacts, reverse=True)
        top = result.attributions[0]
        assert top.feature == "kalman_est_receiving_yards"
        assert top.label == FEATURE_LABELS["kalman_est_receiving_yards"]
        assert isinstance(top, FactorAttribution)

    def test_impacts_match_tree_shap_values(self, trained_model, allow_fallbacks) -> None:
        import shap

        service = SHAPService()
        service._model_cache["receiving_yards_WR"] = trained_model
        features = {"kalman_est_receiving_yards": 72.0, "seas_avg_targets": 7.0, "temp_f": 58.0}
        result = service.explain(features, stat="receiving_yards", position="WR", top_n=3)

        frame = pd.DataFrame([[features[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
        expected = shap.TreeExplainer(trained_model).shap_values(frame)[0]
        by_feature = {a.feature: a.impact for a in result.attributions}
        for i, name in enumerate(MODEL_FEATURES):
            assert by_feature[name] == pytest.approx(round(float(expected[i]), 3), abs=1e-3)

    def test_top_n_limits_output(self, trained_model, allow_fallbacks) -> None:
        service = SHAPService()
        service._model_cache["receiving_yards_WR"] = trained_model
        features = {"kalman_est_receiving_yards": 50.0, "seas_avg_targets": 5.0, "temp_f": 40.0}
        result = service.explain(features, stat="receiving_yards", position="WR", top_n=1)
        assert len(result.attributions) == 1

    def test_model_failure_falls_back_to_synthetic(self, allow_fallbacks) -> None:
        service = SHAPService()
        service._model_cache["receiving_yards_WR"] = object()  # not a tree model
        result = service.explain(
            {"kalman_est_receiving_yards": 60.0, "temp_f": 55.0},
            stat="receiving_yards", position="WR",
        )
        assert result.source == "synthetic_fallback"
        assert result.attributions


# ── explain(): fallbacks and policy ───────────────────────────────────────────

class TestExplainPolicy:
    def test_synthetic_source_when_no_model_and_fallbacks_allowed(self, allow_fallbacks) -> None:
        result = SHAPService().explain({"kalman_est_receiving_yards": 80.0}, "receiving_yards", "WR")
        assert result.source == "synthetic_fallback"
        assert len(result.attributions) == 1

    def test_artifact_backed_mode_refuses_synthetic(self, monkeypatch) -> None:
        monkeypatch.setattr(svc, "fallbacks_allowed", lambda: False)
        with pytest.raises(ArtifactRequiredError, match="PRODUCT_MODE=artifact_backed"):
            SHAPService().explain({"kalman_est_receiving_yards": 80.0}, "receiving_yards", "WR")

    def test_artifact_backed_mode_still_uses_real_model(self, trained_model, monkeypatch) -> None:
        monkeypatch.setattr(svc, "fallbacks_allowed", lambda: False)
        service = SHAPService()
        service._model_cache["receiving_yards_WR"] = trained_model
        result = service.explain(
            {"kalman_est_receiving_yards": 60.0, "seas_avg_targets": 6.0, "temp_f": 60.0},
            "receiving_yards", "WR",
        )
        assert result.source == "model_artifact"


# ── _load_model() and the reachability probe ──────────────────────────────────

class _FakeRegistry:
    def __init__(self, registered=None, versions=None, error=None):
        self._registered = registered or []
        self._versions = versions or []
        self._error = error
        self.search_filters: list[str] = []

    def search_registered_models(self, filter_string):
        self.search_filters.append(filter_string)
        if self._error:
            raise self._error
        return self._registered

    def get_latest_versions(self, name, stages):
        assert stages == ["Production", "None"]
        return self._versions


@contextlib.contextmanager
def _ok_connection(*args, **kwargs):
    yield object()


class TestLoadModel:
    def test_no_tracking_uri_returns_none_and_caches(self) -> None:
        service = SHAPService(mlflow_tracking_uri="")
        assert service._load_model("receiving_yards", "WR") is None
        assert service._model_cache["receiving_yards_WR"] is None

    def test_cached_entry_is_returned_without_lookup(self) -> None:
        sentinel = object()
        service = SHAPService(mlflow_tracking_uri="http://127.0.0.1:15091")
        service._model_cache["targets_TE"] = sentinel
        assert service._load_model("targets", "TE") is sentinel

    def test_unreachable_server_is_skipped(self, monkeypatch) -> None:
        calls: list = []

        def refuse(address, timeout=None):
            calls.append(address)
            raise OSError("connection refused")

        monkeypatch.setattr(socket, "create_connection", refuse)
        service = SHAPService(mlflow_tracking_uri="http://mlflow.internal:15091")
        assert service._load_model("receiving_yards", "WR") is None
        assert service._mlflow_available is False
        assert calls == [("mlflow.internal", 15091)]

    def test_reachability_is_probed_once(self, monkeypatch) -> None:
        calls: list = []

        def refuse(address, timeout=None):
            calls.append(address)
            raise OSError("down")

        monkeypatch.setattr(socket, "create_connection", refuse)
        service = SHAPService(mlflow_tracking_uri="https://mlflow.example:443")
        service._load_model("a", "WR")
        service._load_model("b", "RB")
        assert len(calls) == 1

    def test_default_http_port_used_when_absent(self, monkeypatch) -> None:
        seen: list = []

        def record(address, timeout=None):
            seen.append(address)
            return _ok_connection()

        monkeypatch.setattr(socket, "create_connection", record)
        assert SHAPService(mlflow_tracking_uri="http://tracker.local")._mlflow_reachable() is True
        assert seen == [("tracker.local", 80)]

    def test_file_store_is_treated_as_reachable(self, monkeypatch) -> None:
        def must_not_connect(*args, **kwargs):
            raise AssertionError("local file store should not open a socket")

        monkeypatch.setattr(socket, "create_connection", must_not_connect)
        service = SHAPService(mlflow_tracking_uri="file:///tmp/mlruns-not-used")
        assert service._mlflow_reachable() is True

    def test_no_registered_model_returns_none(self, monkeypatch) -> None:
        import mlflow.tracking

        registry = _FakeRegistry(registered=[])
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _ok_connection())
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: registry)
        service = SHAPService(mlflow_tracking_uri="http://tracker.local:5000")
        assert service._load_model("receiving_yards", "WR") is None
        assert registry.search_filters == ["name LIKE 'xgb_receiving_yards%'"]

    def test_registered_model_without_version_returns_none(self, monkeypatch) -> None:
        import mlflow.tracking

        registry = _FakeRegistry(registered=[SimpleNamespace(name="xgb_receiving_yards_v1")], versions=[])
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _ok_connection())
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: registry)
        service = SHAPService(mlflow_tracking_uri="http://tracker.local:5000")
        assert service._load_model("receiving_yards", "WR") is None

    def test_loads_latest_version_and_caches_it(self, monkeypatch) -> None:
        import mlflow.tracking
        import mlflow.xgboost

        sentinel_model = object()
        loaded_uris: list[str] = []
        registry = _FakeRegistry(
            registered=[SimpleNamespace(name="xgb_receiving_yards_v2")],
            versions=[SimpleNamespace(run_id="run-42")],
        )
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _ok_connection())
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: registry)

        def fake_load(uri):
            loaded_uris.append(uri)
            return sentinel_model

        monkeypatch.setattr(mlflow.xgboost, "load_model", fake_load)
        service = SHAPService(mlflow_tracking_uri="http://tracker.local:5000")
        assert service._load_model("receiving_yards", "WR") is sentinel_model
        assert loaded_uris == ["runs:/run-42/model"]
        assert service._model_cache["receiving_yards_WR"] is sentinel_model

    def test_mlflow_error_is_swallowed(self, monkeypatch) -> None:
        import mlflow.tracking

        registry = _FakeRegistry(error=ConnectionError("registry timeout"))
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _ok_connection())
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: registry)
        service = SHAPService(mlflow_tracking_uri="http://tracker.local:5000")
        assert service._load_model("targets", "WR") is None
        assert service._model_cache["targets_WR"] is None


# ── _synthetic_shap() invariants ──────────────────────────────────────────────

class TestSyntheticShap:
    def test_primary_kalman_feature_pushes_projection_up(self) -> None:
        attrs = SHAPService()._synthetic_shap(
            {"kalman_est_receiving_yards": 80.0, "seas_avg_targets": 6.0, "temp_f": 50.0},
            stat="receiving_yards", position="WR", top_n=10,
        )
        primary = next(a for a in attrs if a.feature == "kalman_est_receiving_yards")
        assert primary.impact > 0
        assert primary.label == "Estimated: " + label("kalman_est_receiving_yards")

    def test_sorted_by_absolute_impact_and_truncated(self) -> None:
        features = {f"seas_metric_{i}": float(i + 1) for i in range(8)}
        attrs = SHAPService()._synthetic_shap(features, stat="targets", position="TE", top_n=3)
        assert len(attrs) == 3
        magnitudes = [abs(a.impact) for a in attrs]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_skips_none_nan_and_non_numeric_values(self) -> None:
        features = {
            "temp_f": 65.0,
            "wind_mph": None,
            "precipitation_bucket": float("nan"),
            "player_id": "00-0031234",
            "game_id": "2024_01_KC_BUF",
        }
        attrs = SHAPService()._synthetic_shap(features, stat="receiving_yards", position="WR", top_n=10)
        assert [a.feature for a in attrs] == ["temp_f"]
        assert attrs[0].value == pytest.approx(65.0)

    def test_values_are_rounded_to_three_decimals(self) -> None:
        attrs = SHAPService()._synthetic_shap({"days_rest": 6.123456}, stat="fantasy_ppr", position="RB", top_n=5)
        assert attrs[0].value == pytest.approx(6.123)
        assert attrs[0].impact == round(attrs[0].impact, 3)

    def test_weight_prefixes_set_impact_bounds(self) -> None:
        # Bounds that hold for any RNG draw: the primary Kalman feature is
        # 0.35 * max(v/100, 0.01) * U(0.5, 1.5), and a rest feature is
        # 0.02 * |v| * U(0.01, 0.15). With v=100 that is [0.175, 0.525] and [0.02, 0.3].
        features = {"kalman_est_targets": 100.0, "days_rest": 100.0}
        attrs = SHAPService()._synthetic_shap(features, stat="targets", position="WR", top_n=5)
        by_name = {a.feature: a.impact for a in attrs}
        assert 0.175 - 1e-3 <= by_name["kalman_est_targets"] <= 0.525 + 1e-3
        assert 0.02 - 1e-3 <= abs(by_name["days_rest"]) <= 0.3 + 1e-3
