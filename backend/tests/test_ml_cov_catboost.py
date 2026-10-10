"""
backend/tests/test_ml_cov_catboost.py

Coverage tests for ml/catboost_model.py (CatBoost base learner, walk-forward CV).

Training is real but small: the iteration count is reduced through the
module's DEFAULT_CATBOOST_PARAMS dict, and every call passes an explicit
out_dir under tmp_path so no artifacts land in the repository. MLflow is
either disabled, pointed at a local file store, or forced to fail; no
network is contacted. The global MLflow tracking URI is restored after each
test that changes it.
"""

from __future__ import annotations

import logging
import re
import sys
import time

import numpy as np
import pandas as pd
import pytest

import catboost as cb

import ml.catboost_model as cbm
from ml.catboost_model import (
    DEFAULT_CATBOOST_PARAMS,
    CatBoostTrainResult,
    _make_walk_forward_folds,
    _run_optuna,
    _train_fold,
    _walk_forward_cv,
    main,
    train,
)
from ml.utils import FEATURE_COLS, FoldResult, _compute_metrics

SMALL_ITERATIONS = 30


def _synthetic_df(seasons, n_per_season: int = 60, position: str = "WR", seed: int = 7) -> pd.DataFrame:
    frames = []
    for season in seasons:
        rng = np.random.default_rng(seed + season)
        n = n_per_season
        row = {
            "player_id": [f"P{season}_{i:03d}" for i in range(n)],
            "game_id": [f"{season}_01_AA_BB_{i}" for i in range(n)],
            "season": [season] * n,
            "week": rng.integers(1, 18, n).tolist(),
            "position": [position] * n,
            "team": ["MIN"] * n,
        }
        for feat in FEATURE_COLS:
            vals = rng.standard_normal(n) * 10 + 5
            vals[rng.random(n) < 0.1] = np.nan
            row[feat] = vals.tolist()
        y = 40.0 + 3.0 * np.nan_to_num(np.asarray(row["seas_avg_receiving_yards"], dtype=float)) + rng.normal(0, 2, n)
        row["actual_receiving_yards"] = y.tolist()
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(autouse=True)
def _fast_catboost(monkeypatch, tmp_path):
    monkeypatch.setitem(DEFAULT_CATBOOST_PARAMS, "iterations", SMALL_ITERATIONS)
    # CatBoost writes its training logs to the working directory unless train_dir is set.
    monkeypatch.chdir(tmp_path)
    # export_onnx defaults to the repository's ml/onnx directory; keep test exports under tmp_path.
    import ml.utils as ml_utils_mod

    monkeypatch.setattr(ml_utils_mod, "_ONNX_DIR", tmp_path / "onnx")


@pytest.fixture
def mlflow_restore():
    """Restore the process-wide MLflow tracking URI after a test that changes it."""
    import mlflow

    previous = mlflow.get_tracking_uri()
    yield mlflow
    mlflow.set_tracking_uri(previous)


# ── constants and dataclass ───────────────────────────────────────────────────

def test_default_params_are_deterministic_and_regression() -> None:
    assert DEFAULT_CATBOOST_PARAMS["loss_function"] == "RMSE"
    assert DEFAULT_CATBOOST_PARAMS["bootstrap_type"] == "Bernoulli"
    assert DEFAULT_CATBOOST_PARAMS["random_seed"] == 42
    assert DEFAULT_CATBOOST_PARAMS["verbose"] == 0


def test_train_result_defaults_oof_path_to_none() -> None:
    result = CatBoostTrainResult(
        fold_results=[], oof_df=pd.DataFrame(), best_params={}, run_id=None, feature_importances={},
    )
    assert result.oof_path is None


# ── make folds (imported helper, used by train) ──────────────────────────────

def test_expanding_window_folds_for_three_seasons() -> None:
    assert _make_walk_forward_folds([2021, 2022, 2023]) == [([2021], 2022), ([2021, 2022], 2023)]


# ── _train_fold ───────────────────────────────────────────────────────────────

class TestTrainFold:
    FEATURES = ["seas_avg_receiving_yards", "seas_avg_targets", "is_home"]

    def test_returns_model_oof_rows_and_metrics(self) -> None:
        df = _synthetic_df([2021, 2022])
        train_df = df[df["season"] == 2021]
        val_df = df[df["season"] == 2022]
        model, oof, fold = _train_fold(
            3, train_df, val_df, self.FEATURES, "actual_receiving_yards", {**DEFAULT_CATBOOST_PARAMS},
        )

        assert isinstance(model, cb.CatBoostRegressor)
        assert len(oof) == len(val_df)
        assert list(oof.columns) == ["player_id", "game_id", "season", "week", "position",
                                     "y_true", "y_pred", "fold_idx"]
        assert (oof["fold_idx"] == 3).all()
        np.testing.assert_allclose(oof["y_true"].to_numpy(), val_df["actual_receiving_yards"].to_numpy())

        mae, rmse = _compute_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
        assert fold.mae == pytest.approx(mae)
        assert fold.rmse == pytest.approx(rmse)
        assert fold.train_seasons == [2021]
        assert fold.val_season == 2022
        assert fold.n_train == len(train_df)
        assert fold.n_val == len(val_df)

    def test_missing_id_columns_are_omitted(self) -> None:
        df = _synthetic_df([2021, 2022])
        bare = df[["season", *self.FEATURES, "actual_receiving_yards"]]
        _, oof, _ = _train_fold(
            0, bare[bare["season"] == 2021], bare[bare["season"] == 2022],
            self.FEATURES, "actual_receiving_yards", {**DEFAULT_CATBOOST_PARAMS},
        )
        assert list(oof.columns) == ["season", "y_true", "y_pred", "fold_idx"]


# ── _walk_forward_cv ──────────────────────────────────────────────────────────

class TestWalkForwardCv:
    FEATURES = ["seas_avg_receiving_yards", "seas_avg_targets"]

    def test_one_result_per_fold_and_final_model(self) -> None:
        df = _synthetic_df([2021, 2022, 2023])
        folds = _make_walk_forward_folds([2021, 2022, 2023])
        results, oof, final = _walk_forward_cv(
            df, folds, self.FEATURES, "actual_receiving_yards", {**DEFAULT_CATBOOST_PARAMS},
        )
        assert [r.val_season for r in results] == [2022, 2023]
        assert [r.n_train for r in results] == [60, 120]
        assert len(oof) == 120  # 60 validation rows per fold
        assert set(oof["fold_idx"]) == {0, 1}
        assert isinstance(final, cb.CatBoostRegressor)

    def test_empty_fold_is_skipped(self) -> None:
        df = _synthetic_df([2021, 2022])
        folds = [([2021], 2022), ([2021], 2030)]  # 2030 has no rows
        results, oof, final = _walk_forward_cv(
            df, folds, self.FEATURES, "actual_receiving_yards", {**DEFAULT_CATBOOST_PARAMS},
        )
        assert len(results) == 1
        assert len(oof) == 60

    def test_no_completed_folds_gives_empty_outputs(self) -> None:
        df = _synthetic_df([2021, 2022])
        results, oof, final = _walk_forward_cv(
            df, [([2021], 2040)], self.FEATURES, "actual_receiving_yards", {**DEFAULT_CATBOOST_PARAMS},
        )
        assert results == [] and oof.empty and final is None

    def test_unsorted_training_seasons_trip_the_guard(self) -> None:
        df = _synthetic_df([2021, 2022, 2023])
        with pytest.raises(AssertionError):
            _walk_forward_cv(df, [([2022, 2021], 2023)], self.FEATURES, "actual_receiving_yards",
                             {**DEFAULT_CATBOOST_PARAMS})

    def test_validation_season_must_follow_training(self) -> None:
        df = _synthetic_df([2021, 2022])
        with pytest.raises(AssertionError):
            _walk_forward_cv(df, [([2021], 2021)], self.FEATURES, "actual_receiving_yards",
                             {**DEFAULT_CATBOOST_PARAMS})


# ── _run_optuna ───────────────────────────────────────────────────────────────

def test_optuna_returns_params_inside_search_ranges() -> None:
    df = _synthetic_df([2021, 2022])
    feats = ["seas_avg_receiving_yards", "seas_avg_targets"]
    best = _run_optuna(df[df["season"] == 2021], df[df["season"] == 2022], feats,
                       "actual_receiving_yards", n_trials=2)

    assert 100 <= best["iterations"] <= 1000
    assert 4 <= best["depth"] <= 10
    assert 1e-3 <= best["learning_rate"] <= 0.3
    assert 1e-3 <= best["l2_leaf_reg"] <= 10.0
    assert 0.5 <= best["subsample"] <= 1.0
    assert 0.5 <= best["colsample_bylevel"] <= 1.0
    # Fixed settings are merged into the returned dict.
    assert best["bootstrap_type"] == "Bernoulli"
    assert best["loss_function"] == "RMSE"
    assert best["random_seed"] == 42
    assert best["nan_mode"] == "Min"


# ── train() ───────────────────────────────────────────────────────────────────

class _FakeClock:
    """Stands in for the time module inside ml.catboost_model so run ids are predictable."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


class TestPseudoRunId:
    def test_is_eight_hex_digits_of_unix_seconds(self) -> None:
        assert cbm._pseudo_run_id(1_780_000_000) == f"{1_780_000_000:08x}"

    def test_unique_per_second_and_time_ordered(self) -> None:
        ids = [cbm._pseudo_run_id(t) for t in (1_780_000_000, 1_780_000_001, 1_780_086_400)]
        assert len(set(ids)) == 3
        assert ids == sorted(ids)

    def test_sorts_after_legacy_date_stamps(self) -> None:
        # Stacking discovery takes the newest file by name; old YYYYMMDD stamps must sort first.
        assert cbm._pseudo_run_id(1_780_000_000) > "20261231"


class TestTrain:
    def test_unknown_target_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="Unknown target 'bogus'"):
            train(_synthetic_df([2021, 2022]), [2021, 2022], target="bogus", out_dir=tmp_path)

    def test_happy_path_writes_oof_manifest_and_normalised_importances(self, tmp_path) -> None:
        df = _synthetic_df([2021, 2022, 2023])
        result = train(
            df, [2021, 2022, 2023], target="receiving_yards", n_optuna_trials=0,
            position_filter="WR", mlflow_tracking_uri="", out_dir=tmp_path,
        )

        assert len(result.fold_results) == 2
        assert len(result.oof_df) == 120
        assert result.run_id is None
        assert result.best_params["nan_mode"] == "Min"
        assert result.best_params["iterations"] == SMALL_ITERATIONS

        # OOF targets are exactly the held-out seasons' actuals.
        held_out = df[df["season"].isin([2022, 2023])]["actual_receiving_yards"].to_numpy()
        np.testing.assert_allclose(np.sort(result.oof_df["y_true"].to_numpy()), np.sort(held_out))

        assert result.oof_path is not None and result.oof_path.exists()
        assert result.oof_path.parent == tmp_path
        assert result.oof_path.name.startswith("catboost_receiving_yards_WR_")
        oof_on_disk = pd.read_csv(result.oof_path)
        assert len(oof_on_disk) == 120

        importances = result.feature_importances
        assert set(importances) == {f for f in FEATURE_COLS if f in df.columns}
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)

        manifests = list((tmp_path / "manifests").glob("catboost_receiving_yards_WR_*.json"))
        assert len(manifests) == 1

    def test_position_filter_restricts_training_rows(self, tmp_path) -> None:
        wr = _synthetic_df([2021, 2022], position="WR")
        rb = _synthetic_df([2021, 2022], position="RB", seed=99)
        df = pd.concat([wr, rb], ignore_index=True)
        result = train(df, [2021, 2022], target="receiving_yards", n_optuna_trials=0,
                       position_filter="WR", mlflow_tracking_uri="", out_dir=tmp_path)
        assert result.fold_results[0].n_val == 60  # only the WR rows
        assert set(result.oof_df["position"]) == {"WR"}

    def test_no_position_filter_uses_every_row(self, tmp_path) -> None:
        df = pd.concat([_synthetic_df([2021, 2022], position="WR"),
                        _synthetic_df([2021, 2022], position="RB", seed=5)], ignore_index=True)
        result = train(df, [2021, 2022], target="receiving_yards", n_optuna_trials=0,
                       position_filter=None, mlflow_tracking_uri="", out_dir=tmp_path)
        assert result.fold_results[0].n_val == 120
        assert result.oof_path.name.startswith("catboost_receiving_yards_all_")

    def test_empty_cohort_is_refused(self, tmp_path) -> None:
        df = _synthetic_df([2021, 2022], position="RB")
        with pytest.raises(ValueError, match="training dataframe is empty"):
            train(df, [2021, 2022], target="receiving_yards", n_optuna_trials=0,
                  position_filter="WR", mlflow_tracking_uri="", out_dir=tmp_path)

    def test_single_season_has_no_walk_forward_fold(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="at least 2 seasons"):
            train(_synthetic_df([2022]), [2022], target="receiving_yards", n_optuna_trials=0,
                  position_filter="WR", mlflow_tracking_uri="", out_dir=tmp_path)

    def test_mlflow_success_logs_run_and_names_oof_by_run_id(self, tmp_path, mlflow_restore) -> None:
        mlflow = mlflow_restore
        uri = (tmp_path / "mlruns").as_uri()
        df = _synthetic_df([2021, 2022, 2023])
        result = train(df, [2021, 2022, 2023], target="receiving_yards", n_optuna_trials=0,
                       position_filter="WR", mlflow_tracking_uri=uri, out_dir=tmp_path / "oof")

        assert result.run_id is not None and len(result.run_id) > 8
        assert result.oof_path.name == f"catboost_receiving_yards_WR_{result.run_id[:8]}.csv"
        assert result.oof_path.exists()
        run = mlflow.get_run(result.run_id)
        assert run.data.params["target"] == "receiving_yards"
        assert run.data.params["position_filter"] == "WR"
        expected_mae = np.mean([fr.mae for fr in result.fold_results])
        assert run.data.metrics["mae"] == pytest.approx(expected_mae, rel=1e-6)
        assert "training_data_hash" in run.data.tags

    def test_mlflow_failure_falls_back_to_timestamp_run_id(self, tmp_path, monkeypatch, caplog) -> None:
        import mlflow

        def broken(name):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(mlflow, "set_experiment", broken)
        previous = mlflow.get_tracking_uri()
        try:
            with caplog.at_level(logging.ERROR, logger=cbm.logger.name):
                result = train(_synthetic_df([2021, 2022]), [2021, 2022], target="receiving_yards",
                               n_optuna_trials=0, position_filter="WR",
                               mlflow_tracking_uri=(tmp_path / "unused").as_uri(), out_dir=tmp_path / "oof")
        finally:
            mlflow.set_tracking_uri(previous)

        assert result.run_id is None
        assert "MLflow logging failed: registry unavailable" in caplog.text
        assert result.oof_path is not None and result.oof_path.exists()
        # The pseudo run id is 8 hex digits of Unix seconds, and save_oof keeps run_id[:8].
        suffix = result.oof_path.stem.rsplit("_", 1)[-1]
        assert re.fullmatch(r"[0-9a-f]{8}", suffix)
        assert abs(int(suffix, 16) - time.time()) < 3600

    def test_same_day_pseudo_runs_keep_separate_oof_files(self, tmp_path, monkeypatch) -> None:
        # Without MLflow each run gets a pseudo run id. A date-only id gave both same-day runs
        # one file name, so the second run overwrote the first.
        clock = _FakeClock(1_780_000_000.0)
        monkeypatch.setattr(cbm, "time", clock, raising=False)
        df = _synthetic_df([2021, 2022, 2023])
        kwargs = dict(target="receiving_yards", n_optuna_trials=0, position_filter="WR",
                      mlflow_tracking_uri="", out_dir=tmp_path / "oof")
        first = train(df, [2021, 2022, 2023], **kwargs)
        clock.now += 60
        second = train(df, [2021, 2022, 2023], **kwargs)
        assert first.oof_path != second.oof_path
        assert first.oof_path.exists() and second.oof_path.exists()

    def test_uri_from_environment_when_not_passed(self, tmp_path, monkeypatch, mlflow_restore) -> None:
        mlflow = mlflow_restore
        monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "env_mlruns").as_uri())
        result = train(_synthetic_df([2021, 2022]), [2021, 2022], target="receiving_yards",
                       n_optuna_trials=0, position_filter="WR", out_dir=tmp_path / "oof")
        assert result.run_id is not None
        assert mlflow.get_tracking_uri().startswith("file:")
        assert (tmp_path / "env_mlruns").exists()

    def test_optuna_branch_runs_when_trials_requested(self, tmp_path) -> None:
        result = train(_synthetic_df([2021, 2022, 2023]), [2021, 2022, 2023],
                       target="receiving_yards", n_optuna_trials=2, position_filter="WR",
                       mlflow_tracking_uri="", out_dir=tmp_path)
        assert result.best_params["random_seed"] == 42
        assert 100 <= result.best_params["iterations"] <= 1000


# ── main() ────────────────────────────────────────────────────────────────────

class _FakeCache:
    calls: list = []
    frame: pd.DataFrame = pd.DataFrame()

    def __init__(self, *args, **kwargs):
        pass

    def load_or_fetch(self, seasons, db_url, position_filter=None, no_cache=False):
        _FakeCache.calls.append({"seasons": list(seasons), "db_url": db_url,
                                 "position": position_filter, "no_cache": no_cache})
        return _FakeCache.frame.copy()


class TestMain:
    def test_missing_database_url_exits_with_status_one(self, monkeypatch) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(sys, "argv", ["catboost_model", "--seasons", "2021-2023"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_incomplete_season_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://db/x")
        monkeypatch.setattr(sys, "argv", ["catboost_model", "--seasons", "2026"])
        with pytest.raises(ValueError, match="exceed the season cap"):
            main()

    def test_cli_trains_through_cache_and_prints_cv_summary(self, monkeypatch, tmp_path, capsys) -> None:
        import ml.parquet_cache as parquet_cache

        _FakeCache.calls = []
        _FakeCache.frame = _synthetic_df([2021, 2022, 2023])
        monkeypatch.setattr(parquet_cache, "FeatureCache", _FakeCache)
        monkeypatch.setenv("DATABASE_URL", "postgresql://db/x")
        monkeypatch.setattr(sys, "argv", [
            "catboost_model", "--seasons", "2021-2023", "--n-trials", "0",
            "--no-mlflow", "--out-dir", str(tmp_path / "oof"), "--no-cache",
        ])
        main()

        assert _FakeCache.calls == [{"seasons": [2021, 2022, 2023], "db_url": "postgresql://db/x",
                                     "position": "WR", "no_cache": True}]
        out = capsys.readouterr().out
        assert out.startswith("CatBoost CV MAE:")
        assert " | RMSE: " in out
        assert list((tmp_path / "oof").glob("catboost_receiving_yards_WR_*.csv"))

    def test_cli_falls_back_to_direct_loader_without_cache_module(self, monkeypatch, tmp_path) -> None:
        loads: list = []

        def direct_loader(db_url, seasons, position_filter=None, feature_cols=None):
            loads.append((db_url, list(seasons), position_filter))
            return _synthetic_df(list(seasons))

        monkeypatch.setitem(sys.modules, "ml.parquet_cache", None)  # forces ImportError
        monkeypatch.setattr(cbm, "load_feature_matrix", direct_loader)
        monkeypatch.setenv("DATABASE_URL", "postgresql://db/y")
        monkeypatch.setattr(sys, "argv", [
            "catboost_model", "--seasons", "2021 2022", "--position", "all", "--n-trials", "0",
            "--no-mlflow", "--out-dir", str(tmp_path / "oof"),
        ])
        main()
        assert loads == [("postgresql://db/y", [2021, 2022], None)]

    def test_empty_frame_exits_with_status_one(self, monkeypatch, tmp_path) -> None:
        import ml.parquet_cache as parquet_cache

        _FakeCache.calls = []
        _FakeCache.frame = pd.DataFrame()
        monkeypatch.setattr(parquet_cache, "FeatureCache", _FakeCache)
        monkeypatch.setenv("DATABASE_URL", "postgresql://db/x")
        monkeypatch.setattr(sys, "argv", [
            "catboost_model", "--seasons", "2021-2022", "--no-mlflow", "--out-dir", str(tmp_path),
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
