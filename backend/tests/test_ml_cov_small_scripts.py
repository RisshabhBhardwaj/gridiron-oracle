"""
backend/tests/test_ml_cov_small_scripts.py

Coverage tests for small entry-point modules that had no tests:
  - ml/check_model_features.py  (MLflow feature-count audit)
  - ml/compare_actuals.py       (ad-hoc SQL report, runs at import time)
  - ml/tft_cli.py               (TFT argparse entry point)
  - ml/bayesian_model.py        (public alias; currently cannot be imported)

Everything runs offline. MLflow calls are replaced with stubs; SQL is replaced
with a stubbed engine and DataFrame reader.
"""

from __future__ import annotations

import runpy
from types import SimpleNamespace

import pandas as pd
import pytest

import ml.utils as ml_utils
from ml import check_model_features as cmf
from ml import tft_cli
from ml.utils import FoldResult


# ── bayesian_model alias ──────────────────────────────────────────────────────

@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason=(
        "Production bug: ml/bayesian_model.py imports BayesianLayer from ml/bayesian_layer.py, "
        "but that module defines BayesianProjection and no BayesianLayer, so importing "
        "ml.bayesian_model fails. Fix the alias in ml/bayesian_model.py, then this xfail flips."
    ),
)
def test_bayesian_model_alias_resolves() -> None:
    import importlib

    from ml.bayesian_layer import BayesianProjection

    alias_module = importlib.import_module("ml.bayesian_model")
    assert alias_module.BayesianModel is BayesianProjection


# ── check_model_features ──────────────────────────────────────────────────────

class _NFeatures:
    def __init__(self, n: int) -> None:
        self.n_features_in_ = n


class TestFeatureCount:
    def test_sklearn_style_attribute(self) -> None:
        assert cmf._get_feature_count(_NFeatures(113)) == 113

    def test_feature_names_in(self) -> None:
        assert cmf._get_feature_count(SimpleNamespace(feature_names_in_=["a", "b", "c"])) == 3

    def test_lightgbm_feature_name(self) -> None:
        assert cmf._get_feature_count(SimpleNamespace(feature_name_=["a", "b"])) == 2

    def test_feature_names_attribute(self) -> None:
        assert cmf._get_feature_count(SimpleNamespace(feature_names_=["a"])) == 1

    def test_xgboost_booster_feature_names(self) -> None:
        booster = SimpleNamespace(feature_names=["x", "y", "z", "w"])
        assert cmf._get_feature_count(SimpleNamespace(get_booster=lambda: booster)) == 4

    def test_booster_without_names_is_none(self) -> None:
        model = SimpleNamespace(get_booster=lambda: SimpleNamespace(feature_names=None))
        assert cmf._get_feature_count(model) is None

    def test_unknown_object_is_none(self) -> None:
        assert cmf._get_feature_count(object()) is None


def test_check_mlflow_running_requires_http_uri(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "/local/path")
    assert cmf.check_mlflow_running() is False
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15091")
    assert cmf.check_mlflow_running() is True


class _FakeMlflowRegistry:
    """Stub for the mlflow calls made by check_model_features.main()."""

    def __init__(self, models: dict, experiments: set[str]) -> None:
        self.models = models          # {(learner, stat): model object or Exception}
        self.experiments = experiments
        self.tracking_uri = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def get_experiment_by_name(self, name):
        if name not in self.experiments:
            return None
        return SimpleNamespace(experiment_id=name)

    def search_runs(self, experiment_ids, **kwargs):
        assert kwargs.get("filter_string") == "status = 'FINISHED'"
        return pd.DataFrame([{"run_id": f"run-{experiment_ids[0]}"}])

    def loader_for(self, learner):
        def load(model_uri):
            # model_uri is runs:/run-<learner>_<stat>/model
            run_key = model_uri.split("/")[1].removeprefix("run-")
            learner_name, _, stat = run_key.partition("_")
            outcome = self.models[(learner, stat)]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return load


def _patch_mlflow(monkeypatch, registry) -> None:
    import mlflow
    import mlflow.catboost
    import mlflow.lightgbm
    import mlflow.xgboost

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15091")
    monkeypatch.setattr(mlflow, "set_tracking_uri", registry.set_tracking_uri)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", registry.get_experiment_by_name)
    monkeypatch.setattr(mlflow, "search_runs", registry.search_runs)
    monkeypatch.setattr(mlflow.xgboost, "load_model", registry.loader_for("xgb"))
    monkeypatch.setattr(mlflow.lightgbm, "load_model", registry.loader_for("lgbm"))
    monkeypatch.setattr(mlflow.catboost, "load_model", registry.loader_for("catboost"))


@pytest.fixture
def mlflow_stub(monkeypatch):
    expected = len(ml_utils.FEATURE_COLS)
    registry = _FakeMlflowRegistry(
        models={
            ("xgb", "receiving_yards"): _NFeatures(expected),          # matches
            ("lgbm", "receiving_yards"): _NFeatures(expected - 5),     # stale feature count
            ("catboost", "receiving_yards"): SimpleNamespace(),        # no feature info
            ("xgb", "targets"): RuntimeError("artifact missing"),      # load failure
        },
        experiments={"xgb_receiving_yards", "lgbm_receiving_yards",
                     "catboost_receiving_yards", "xgb_targets"},
    )
    _patch_mlflow(monkeypatch, registry)
    return registry


class TestCheckModelFeaturesMain:
    def test_reports_mismatch_and_retrain_commands(self, mlflow_stub, capsys) -> None:
        assert cmf.main() == 0
        out = capsys.readouterr().out
        expected = len(ml_utils.FEATURE_COLS)
        assert mlflow_stub.tracking_uri == "http://127.0.0.1:15091"
        assert f"Expected features: {expected}" in out
        assert "OK: 1  |  Mismatched: 1" in out
        assert f"lgbm       receiving_yards      has {expected - 5} features" in out
        assert "catboost/receiving_yards: could not get feature count (got None)" in out
        assert "xgb/targets: load failed — artifact missing" in out
        assert "rm -f ml/checkpoints/done/lgbm_receiving_yards_QB.done" in out
        assert "rm -f ml/checkpoints/done/stack_receiving_yards.done" in out
        assert "bash ml/train_all_models.sh --resume" in out

    def test_all_matching_models_report_clean(self, monkeypatch, capsys) -> None:
        expected = len(ml_utils.FEATURE_COLS)
        registry = _FakeMlflowRegistry(
            models={("xgb", "receiving_yards"): _NFeatures(expected)},
            experiments={"xgb_receiving_yards"},
        )
        _patch_mlflow(monkeypatch, registry)
        assert cmf.main() == 0
        out = capsys.readouterr().out
        assert "OK: 1  |  Mismatched: 0" in out
        assert "All models have correct feature count." in out

    def test_experiment_without_finished_runs_is_skipped(self, monkeypatch, capsys) -> None:
        class _NoFinishedRuns(_FakeMlflowRegistry):
            def search_runs(self, experiment_ids, **kwargs):
                return pd.DataFrame()

        registry = _NoFinishedRuns(models={}, experiments={"xgb_receiving_yards"})
        _patch_mlflow(monkeypatch, registry)
        assert cmf.main() == 0
        assert "OK: 0  |  Mismatched: 0" in capsys.readouterr().out

    def test_empty_tracking_uri_returns_one(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
        assert cmf.main() == 1
        assert "Set MLFLOW_TRACKING_URI" in capsys.readouterr().out


# ── compare_actuals (script runs SQL at import time) ─────────────────────────

class TestCompareActuals:
    def _patch(self, monkeypatch, read_sql):
        import pandas
        import sqlalchemy

        created: list[str] = []

        def fake_create_engine(url):
            created.append(url)
            return "ENGINE"

        monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)
        monkeypatch.setattr(pandas, "read_sql", read_sql)
        return created

    def test_prints_query_result(self, monkeypatch, capsys) -> None:
        seen: dict = {}

        def fake_read_sql(query, engine):
            seen["query"] = query
            seen["engine"] = engine
            return pd.DataFrame({"full_name": ["QB One"], "stat": ["passing_yards"]})

        created = self._patch(monkeypatch, fake_read_sql)
        runpy.run_module("ml.compare_actuals", run_name="__main__")

        assert created == ["postgresql://oracle:oracle@localhost:15439/oracle"]
        assert seen["engine"] == "ENGINE"
        assert "WHERE pr.season = 2024 AND p.position = 'QB'" in seen["query"]
        assert "LIMIT 15" in seen["query"]
        out = capsys.readouterr().out
        assert "QB One" in out and "passing_yards" in out

    def test_query_error_is_printed(self, monkeypatch, capsys) -> None:
        def failing_read_sql(query, engine):
            raise RuntimeError("relation feature_matrix does not exist")

        self._patch(monkeypatch, failing_read_sql)
        runpy.run_module("ml.compare_actuals", run_name="__main__")
        assert "Error: relation feature_matrix does not exist" in capsys.readouterr().out


# ── tft_cli ───────────────────────────────────────────────────────────────────

class TestTftCliParser:
    def test_defaults(self) -> None:
        args = tft_cli.build_parser().parse_args(["--seasons", "2019-2025"])
        assert args.seasons == "2019-2025"
        assert args.target == "receiving_yards"
        assert args.position == "WR"
        assert args.out_dir == "ml/oof"
        assert args.hidden_size == 64
        assert args.max_epochs == 30
        assert args.n_trials == 20
        assert args.fast is False
        assert args.no_mlflow is False
        assert args.checkpoint is None
        assert args.db_url is None

    def test_rejects_unknown_target(self) -> None:
        with pytest.raises(SystemExit):
            tft_cli.build_parser().parse_args(["--seasons", "2024", "--target", "not_a_stat"])

    def test_seasons_is_required(self) -> None:
        with pytest.raises(SystemExit):
            tft_cli.build_parser().parse_args([])


class TestTftCliMain:
    @pytest.fixture
    def tft_stubs(self, monkeypatch, tmp_path):
        import ml.tft_training as tft_training

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        calls: dict = {"load": [], "train": [], "frame": pd.DataFrame({"x": [1]})}

        def fake_load(db_url, seasons, position_filter=None, feature_cols=None):
            calls["load"].append({"db_url": db_url, "seasons": list(seasons), "position": position_filter})
            return calls["frame"]

        def fake_train(**kwargs):
            calls["train"].append(kwargs)
            return calls["result"]

        calls["result"] = SimpleNamespace(
            fold_results=[
                FoldResult(fold_idx=0, train_seasons=[2019], val_season=2020, mae=2.0, rmse=3.0, n_train=10, n_val=5),
                FoldResult(fold_idx=1, train_seasons=[2019, 2020], val_season=2021, mae=4.0, rmse=5.0, n_train=15, n_val=5),
            ],
            run_id=None,
            oof_path=None,
        )
        monkeypatch.setattr(tft_cli, "load_feature_matrix", fake_load)
        monkeypatch.setattr(tft_training, "train", fake_train)
        return calls

    def test_fast_mode_trains_and_writes_completion_marker(self, tft_stubs, tmp_path, capsys) -> None:
        tft_cli.main([
            "--seasons", "2019-2024", "--target", "receiving_yards", "--fast",
            "--db-url", "postgresql://db/x", "--no-mlflow",
        ])

        assert tft_stubs["load"] == [
            {"db_url": "postgresql://db/x", "seasons": [2019, 2020, 2021, 2022, 2023, 2024], "position": "WR"},
        ]
        train_kwargs = tft_stubs["train"][0]
        assert train_kwargs["config"].max_epochs == 3
        assert train_kwargs["n_optuna_trials"] == 0
        assert train_kwargs["mlflow_tracking_uri"] == ""
        assert train_kwargs["checkpoint_path"] is None
        assert train_kwargs["position_filter"] == "WR"

        assert (tmp_path / "ml" / "checkpoints" / "done" / "tft_receiving_yards.done").exists()

        out = capsys.readouterr().out
        assert "Folds:      2" in out
        assert "Agg MAE:    3.000" in out
        assert "Agg RMSE:   4.000" in out
        assert "MLflow ID:  N/A" in out
        assert "OOF path:   N/A" in out

    def test_env_database_url_and_all_positions(self, tft_stubs, monkeypatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
        tft_cli.main(["--seasons", "2024", "--position", "ALL", "--no-mlflow", "--n-trials", "0"])
        assert tft_stubs["load"][0]["db_url"] == "postgresql://env/db"
        assert tft_stubs["load"][0]["position"] is None
        assert tft_stubs["train"][0]["position_filter"] is None
        assert tft_stubs["train"][0]["n_optuna_trials"] == 0
        assert tft_stubs["train"][0]["config"].max_epochs == 30

    def test_checkpoint_and_mlflow_flags_are_forwarded(self, tft_stubs) -> None:
        tft_cli.main([
            "--seasons", "2024", "--db-url", "postgresql://db/x",
            "--mlflow-uri", "http://mlflow:5000", "--checkpoint", "ml/checkpoints/a.ckpt",
            "--hidden-size", "32", "--max-epochs", "7", "--out-dir", "out/oof",
        ])
        kwargs = tft_stubs["train"][0]
        assert kwargs["mlflow_tracking_uri"] == "http://mlflow:5000"
        assert kwargs["checkpoint_path"] == "ml/checkpoints/a.ckpt"
        assert str(kwargs["out_dir"]) == "out/oof"
        assert kwargs["config"].hidden_size == 32
        assert kwargs["config"].max_epochs == 7

    def test_missing_database_url_is_a_parser_error(self, tft_stubs) -> None:
        with pytest.raises(SystemExit) as exc:
            tft_cli.main(["--seasons", "2024"])
        assert exc.value.code == 2
        assert tft_stubs["train"] == []

    def test_incomplete_season_is_refused_before_loading(self, tft_stubs) -> None:
        with pytest.raises(ValueError, match="exceed the season cap"):
            tft_cli.main(["--seasons", "2026", "--db-url", "postgresql://db/x"])
        assert tft_stubs["load"] == []

    def test_empty_frame_aborts_with_exit_code_one(self, tft_stubs) -> None:
        tft_stubs["frame"] = pd.DataFrame()
        with pytest.raises(SystemExit) as exc:
            tft_cli.main(["--seasons", "2024", "--db-url", "postgresql://db/x"])
        assert exc.value.code == 1
        assert tft_stubs["train"] == []
