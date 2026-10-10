"""
backend/tests/test_ml_cov_shap_analysis.py

Coverage tests for ml/shap_analysis.py (TreeExplainer wrapper for XGB and LGB).

Small real models are trained in tmp_path. The target depends almost entirely
on one feature, so the top SHAP feature is known in advance. For a tree model
the SHAP values are additive: expected_value + sum(shap) equals the prediction,
and that identity is checked directly.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import ml.shap_analysis as sa
from ml.shap_analysis import (
    MODEL_TYPE_AUTO,
    MODEL_TYPE_LGB,
    MODEL_TYPE_XGB,
    PredictionExplanation,
    SHAPAnalyzer,
    SHAPReport,
    generate_shap_report,
)

FEATURES = ["kalman_est_receiving_yards", "seas_avg_targets", "temp_f"]


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "kalman_est_receiving_yards": rng.normal(size=n),
        "seas_avg_targets": rng.normal(size=n),
        "temp_f": rng.normal(size=n),
    })
    df["player_id"] = [f"P{i}" for i in range(n)]
    df["target"] = 5.0 * df["kalman_est_receiving_yards"] + 0.05 * rng.normal(size=n)
    return df


@pytest.fixture(scope="module")
def xgb_model_path(tmp_path_factory):
    df = _frame()
    model = xgb.XGBRegressor(n_estimators=40, max_depth=3, learning_rate=0.2)
    model.fit(df[FEATURES], df["target"])
    path = tmp_path_factory.mktemp("shap_xgb") / "model.ubj"
    model.get_booster().save_model(str(path))
    return path


@pytest.fixture(scope="module")
def lgb_model_path(tmp_path_factory):
    df = _frame()
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 7, "learning_rate": 0.2, "verbose": -1,
         "min_data_in_leaf": 5, "seed": 1},
        lgb.Dataset(df[FEATURES], label=df["target"]),
        num_boost_round=40,
    )
    path = tmp_path_factory.mktemp("shap_lgb") / "model.txt"
    booster.save_model(str(path))
    return path


class TestDetectType:
    @pytest.mark.parametrize("name, expected", [
        ("m.ubj", MODEL_TYPE_XGB), ("m.model", MODEL_TYPE_XGB), ("m.json", MODEL_TYPE_XGB),
        ("m.txt", MODEL_TYPE_LGB), ("m.bin", MODEL_TYPE_LGB), ("m.lgb", MODEL_TYPE_LGB),
    ])
    def test_auto_detects_from_extension(self, name, expected) -> None:
        assert SHAPAnalyzer(name).model_type == expected

    def test_explicit_hint_overrides_extension(self) -> None:
        assert SHAPAnalyzer("m.txt", model_type=MODEL_TYPE_XGB).model_type == MODEL_TYPE_XGB

    def test_unknown_extension_warns_and_assumes_xgb(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=sa.logger.name):
            analyzer = SHAPAnalyzer("model.weird", model_type=MODEL_TYPE_AUTO)
        assert analyzer.model_type == MODEL_TYPE_XGB
        assert "Cannot auto-detect" in caplog.text

    def test_default_feature_cols_are_the_module_feature_list(self) -> None:
        assert SHAPAnalyzer("m.ubj").feature_cols is sa.FEATURE_COLS
        assert SHAPAnalyzer("m.ubj", feature_cols=FEATURES).feature_cols == FEATURES


class TestLoad:
    def test_xgb_load_builds_explainer(self, xgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES).load()
        assert analyzer._loaded is True
        assert analyzer._explainer is not None

    def test_lgb_load_builds_explainer(self, lgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(lgb_model_path), feature_cols=FEATURES).load()
        assert analyzer._loaded is True

    def test_missing_xgb_file_raises_runtime_error(self, tmp_path) -> None:
        analyzer = SHAPAnalyzer(str(tmp_path / "absent.ubj"), feature_cols=FEATURES)
        with pytest.raises(RuntimeError, match="Failed to load XGBoost model"):
            analyzer.load()

    def test_missing_lgb_file_raises_runtime_error(self, tmp_path) -> None:
        analyzer = SHAPAnalyzer(str(tmp_path / "absent.txt"), feature_cols=FEATURES)
        with pytest.raises(RuntimeError, match="Failed to load LightGBM model"):
            analyzer.load()


class TestPrepareFeatures:
    def test_orders_columns_and_fills_missing_with_zero(self) -> None:
        analyzer = SHAPAnalyzer("m.ubj", feature_cols=["temp_f", "not_there", "seas_avg_targets"])
        df = pd.DataFrame({"seas_avg_targets": [2.0], "temp_f": [9.0], "extra": [1.0]})
        X = analyzer._prepare_features(df)
        assert list(X.columns) == ["temp_f", "not_there", "seas_avg_targets"]
        assert X.iloc[0].tolist() == [9.0, 0.0, 2.0]


class TestRunXgb:
    def test_run_ranks_driving_feature_first(self, xgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(xgb_model_path), model_type=MODEL_TYPE_XGB, feature_cols=FEATURES)
        report = analyzer.run(_frame(200, seed=2), stat="receiving_yards")

        assert isinstance(report, SHAPReport)
        assert report.model_type == MODEL_TYPE_XGB
        assert report.stat == "receiving_yards"
        assert report.n_rows == 200
        assert report.feature_names == FEATURES
        assert report.shap_values.shape == (200, 3)
        assert list(report.importance_df["rank"]) == [1, 2, 3]
        assert report.importance_df.iloc[0]["feature"] == "kalman_est_receiving_yards"
        assert report.importance_df["mean_abs_shap"].is_monotonic_decreasing
        # mean_abs_shap is exactly the column mean of |shap_values|.
        expected = np.abs(report.shap_values).mean(axis=0)
        top = report.importance_df.set_index("feature")["mean_abs_shap"]
        for i, name in enumerate(FEATURES):
            assert top[name] == pytest.approx(expected[i])

    def test_run_samples_rows_when_over_max(self, xgb_model_path) -> None:
        report = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES).run(_frame(200, seed=3), max_rows=50)
        assert report.n_rows == 50
        assert report.shap_values.shape == (50, 3)

    def test_run_loads_lazily(self, xgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES)
        assert analyzer._loaded is False
        analyzer.run(_frame(20, seed=4))
        assert analyzer._loaded is True

    def test_explain_prediction_is_additive_and_sorted(self, xgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES).load()
        df = _frame(5, seed=5)
        row = df.iloc[2]
        explanation = analyzer.explain_prediction(row, stat="receiving_yards")

        assert isinstance(explanation, PredictionExplanation)
        assert explanation.player_id == "P2"
        assert explanation.stat == "receiving_yards"
        assert explanation.base_value == pytest.approx(float(analyzer._explainer.expected_value))

        # Prediction equals the booster margin (explainer additivity).
        dmat = xgb.DMatrix(df[FEATURES].iloc[[2]].values, feature_names=FEATURES)
        assert explanation.prediction == pytest.approx(float(analyzer._model.predict(dmat)[0]), abs=1e-4)

        contrib = explanation.contributions
        assert list(contrib.columns) == ["feature", "shap_value", "feature_value"]
        assert set(contrib["feature"]) == set(FEATURES)
        assert contrib.iloc[0]["feature"] == "kalman_est_receiving_yards"
        abs_vals = contrib["shap_value"].abs().tolist()
        assert abs_vals == sorted(abs_vals, reverse=True)
        for _, r in contrib.iterrows():
            assert r["feature_value"] == pytest.approx(row[r["feature"]])

    def test_explanation_to_string_lists_top_features(self, xgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES).load()
        explanation = analyzer.explain_prediction(_frame(3).iloc[0], stat="receiving_yards")
        lines = explanation.to_string(top_n=2).splitlines()
        assert lines[0].startswith(f"Prediction for P0 | receiving_yards: {explanation.prediction:.1f}")
        assert lines[1].startswith("Base value (mean):")
        assert lines[2] == "── Top contributing features ──"
        assert len(lines) == 3 + 2
        assert "kalman_est_receiving_yards" in lines[3]
        assert lines[3][2] in ("▲", "▼")


class TestRunLgb:
    def test_lgb_run_ranks_driving_feature_first(self, lgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(lgb_model_path), feature_cols=FEATURES)
        assert analyzer.model_type == MODEL_TYPE_LGB
        report = analyzer.run(_frame(150, seed=6), stat="targets")
        assert report.model_type == MODEL_TYPE_LGB
        assert report.importance_df.iloc[0]["feature"] == "kalman_est_receiving_yards"

    def test_lgb_explain_prediction_is_additive(self, lgb_model_path) -> None:
        analyzer = SHAPAnalyzer(str(lgb_model_path), feature_cols=FEATURES).load()
        df = _frame(4, seed=7)
        explanation = analyzer.explain_prediction(df.iloc[1])
        expected = float(analyzer._model.predict(df[FEATURES].iloc[[1]].values)[0])
        assert explanation.prediction == pytest.approx(expected, abs=1e-3)


class TestReportSaveAndRepr:
    def test_top_features_and_repr(self, xgb_model_path) -> None:
        report = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES).run(_frame(40, seed=8))
        assert len(report.top_features(2)) == 2
        assert report.top_features(2).iloc[0]["feature"] == "kalman_est_receiving_yards"
        assert repr(report) == "SHAPReport(stat=receiving_yards, model=xgb, rows=40, features=3)"

    def test_save_writes_csv_and_plots(self, xgb_model_path, tmp_path) -> None:
        report = SHAPAnalyzer(str(xgb_model_path), feature_cols=FEATURES).run(_frame(60, seed=9))
        out = tmp_path / "reports" / "xgb"
        report.save(str(out))

        csv = pd.read_csv(out / "shap_importance.csv")
        assert list(csv["feature"])[0] == "kalman_est_receiving_yards"
        assert list(csv.columns) == ["feature", "mean_abs_shap", "rank"]
        for name in ("shap_summary_beeswarm.png", "shap_summary_bar.png"):
            assert (out / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_shap_report_saves_when_out_dir_given(self, xgb_model_path, tmp_path, monkeypatch) -> None:
        # generate_shap_report uses the module default feature list; the model was trained on FEATURES.
        monkeypatch.setattr(sa, "FEATURE_COLS", FEATURES)
        out = tmp_path / "generated"
        report = generate_shap_report(
            str(xgb_model_path), _frame(30, seed=10), stat="fantasy_ppr", out_dir=str(out),
        )
        assert report.stat == "fantasy_ppr"
        assert (out / "shap_importance.csv").exists()

    def test_generate_shap_report_without_out_dir_writes_nothing(self, xgb_model_path, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(sa, "FEATURE_COLS", FEATURES)
        report = generate_shap_report(str(xgb_model_path), _frame(30, seed=11))
        assert report.n_rows == 30
        assert list(tmp_path.iterdir()) == []


class _FakeMlflowClient:
    """Stands in for mlflow.tracking.MlflowClient in run_from_mlflow."""

    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.calls: list[tuple] = []

    def download_artifacts(self, run_id, artifact_path, dst_path):
        self.calls.append((run_id, artifact_path, dst_path))
        return str(self.artifact_dir)


class TestRunFromMlflow:
    @pytest.fixture(autouse=True)
    def _no_global_tracking_uri(self, monkeypatch):
        import mlflow

        monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
        monkeypatch.setattr(sa, "FEATURE_COLS", FEATURES)

    def test_downloads_model_runs_shap_and_saves(self, monkeypatch, xgb_model_path, tmp_path) -> None:
        import mlflow.tracking

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "model.ubj").write_bytes(xgb_model_path.read_bytes())
        fake = _FakeMlflowClient(artifact_dir)
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: fake)

        out = tmp_path / "out"
        report = SHAPAnalyzer.run_from_mlflow(
            "run-123", "receiving_yards", _frame(40, seed=12),
            tracking_uri="http://mlflow.test", out_dir=str(out),
        )
        assert fake.calls == [("run-123", "receiving_yards_xgb", "/tmp")]
        assert report.n_rows == 40
        assert report.importance_df.iloc[0]["feature"] == "kalman_est_receiving_yards"
        assert (out / "shap_importance.csv").exists()

    def test_falls_back_to_first_ubj_file(self, monkeypatch, xgb_model_path, tmp_path) -> None:
        import mlflow.tracking

        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "weights.ubj").write_bytes(xgb_model_path.read_bytes())
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient(artifact_dir))

        report = SHAPAnalyzer.run_from_mlflow("run-9", "targets", _frame(20, seed=13))
        assert report.stat == "targets"

    def test_no_ubj_file_is_a_runtime_error(self, monkeypatch, tmp_path) -> None:
        import mlflow.tracking

        artifact_dir = tmp_path / "empty"
        artifact_dir.mkdir()
        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient(artifact_dir))
        with pytest.raises(RuntimeError, match="No .ubj file"):
            SHAPAnalyzer.run_from_mlflow("run-x", "targets", _frame(5))

    def test_download_failure_is_wrapped(self, monkeypatch) -> None:
        import mlflow.tracking

        class _Broken:
            def download_artifacts(self, *args, **kwargs):
                raise ConnectionError("tracking server down")

        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: _Broken())
        with pytest.raises(RuntimeError, match="Failed to download MLflow artifact"):
            SHAPAnalyzer.run_from_mlflow("run-y", "targets", _frame(5))

    def test_tracking_uri_defaults_to_env_then_local_store(self, monkeypatch) -> None:
        import mlflow
        import mlflow.tracking

        seen: list[str] = []
        monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: seen.append(uri))

        class _Client:
            def download_artifacts(self, *args, **kwargs):
                raise RuntimeError("stop after the URI is chosen")

        monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: _Client())
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(RuntimeError):
            SHAPAnalyzer.run_from_mlflow("r", "targets", _frame(3))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env.test:5000")
        with pytest.raises(RuntimeError):
            SHAPAnalyzer.run_from_mlflow("r", "targets", _frame(3))
        assert seen == ["mlruns", "http://env.test:5000"]
