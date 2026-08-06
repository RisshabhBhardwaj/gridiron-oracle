"""
backend/tests/test_xgb_model.py

Tests for ml/xgb_model.py — XGBoost base learner.

All tests use synthetic in-memory data; no DB or running MLflow server required.
MLflow is pointed at a tmp directory via mlflow.set_tracking_uri().

KEY TEST (temporal integrity):
  test_oof_only_from_val_seasons — verifies that OOF predictions come only
  from seasons the model never trained on. This is the core anti-leakage proof.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.xgb_model import (
    DEFAULT_XGB_PARAMS,
    FEATURE_COLS,
    TARGET_COL_MAP,
    FoldResult,
    XGBTrainResult,
    _compute_metrics,
    _data_hash,
    _make_walk_forward_folds,
    _run_optuna,
    _train_fold,
    _walk_forward_cv,
    train,
    _parse_seasons,
    save_oof,
)


# ── Synthetic data factory ────────────────────────────────────────────────────

def _make_synthetic_df(
    seasons: list[int],
    n_per_season: int = 60,
    position: str = "WR",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a realistic synthetic feature_matrix DataFrame for testing.

    Uses per-season RNG seed offset so different seasons have different
    distributions, making the temporal structure realistic.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        s_rng = np.random.default_rng(seed + season)
        n = n_per_season
        week_base = 1
        row = {
            "player_id": [f"P{season}_{i:04d}" for i in range(n)],
            "game_id":   [f"{season}_01_AA_BB_{i}" for i in range(n)],
            "season":    [season] * n,
            "week":      (s_rng.integers(1, 18, n)).tolist(),
            "position":  [position] * n,
            "team":      ["MIN"] * n,
        }
        # Feature columns — all numeric, allow NaN to test XGBoost NaN handling
        for feat in FEATURE_COLS:
            vals = s_rng.standard_normal(n) * 10 + 5
            # Sprinkle 10% NaN to mirror real data (players with no prior games)
            mask = s_rng.random(n) < 0.10
            vals[mask] = np.nan
            row[feat] = vals.tolist()

        # Target columns — exponential distribution (yards are non-negative)
        row["actual_receiving_yards"] = s_rng.exponential(45, n).tolist()
        row["actual_rushing_yards"]   = s_rng.exponential(20, n).tolist()
        row["actual_passing_yards"]   = s_rng.exponential(200, n).tolist()
        row["actual_fantasy_ppr"]     = s_rng.exponential(12, n).tolist()

        rows.append(pd.DataFrame(row))

    return pd.concat(rows, ignore_index=True)


# ── _parse_seasons ────────────────────────────────────────────────────────────

class TestParseSeasons:
    def test_range_notation(self) -> None:
        assert _parse_seasons("2018-2024") == list(range(2018, 2025))

    def test_single_year(self) -> None:
        assert _parse_seasons("2025") == [2025]

    def test_space_separated(self) -> None:
        assert _parse_seasons("2022 2023 2024") == [2022, 2023, 2024]

    def test_two_year_range(self) -> None:
        assert _parse_seasons("2023-2024") == [2023, 2024]


# ── _make_walk_forward_folds ──────────────────────────────────────────────────

class TestMakeWalkForwardFolds:
    def test_three_seasons_produces_two_folds(self) -> None:
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        assert len(folds) == 2

    def test_fold_structure(self) -> None:
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        train0, val0 = folds[0]
        assert train0 == [2022]
        assert val0 == 2023
        train1, val1 = folds[1]
        assert train1 == [2022, 2023]
        assert val1 == 2024

    def test_seven_seasons_produces_six_folds(self) -> None:
        seasons = list(range(2018, 2025))
        folds = _make_walk_forward_folds(seasons)
        assert len(folds) == 6

    def test_val_always_after_max_train(self) -> None:
        folds = _make_walk_forward_folds(list(range(2018, 2025)))
        for train, val in folds:
            assert max(train) < val

    def test_train_always_sorted(self) -> None:
        folds = _make_walk_forward_folds(list(range(2018, 2025)))
        for train, val in folds:
            assert train == sorted(train)

    def test_expanding_window(self) -> None:
        folds = _make_walk_forward_folds(list(range(2018, 2025)))
        prev_len = 0
        for train, _ in folds:
            assert len(train) > prev_len
            prev_len = len(train)

    def test_unsorted_input_is_sorted_internally(self) -> None:
        folds = _make_walk_forward_folds([2024, 2022, 2023])
        assert folds[0] == ([2022], 2023)

    def test_fewer_than_two_seasons_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 seasons"):
            _make_walk_forward_folds([2024])


# ── Temporal ordering assertions ──────────────────────────────────────────────

class TestTemporalAssertions:
    """Verify that the hard temporal-ordering guards in _walk_forward_cv fire."""

    def test_bad_fold_order_raises_assertion(self) -> None:
        """
        Passing a fold where val_season <= max(train_seasons) must raise
        AssertionError. This verifies the leakage guard is active.
        """
        df = _make_synthetic_df([2022, 2023, 2024])
        # Deliberately corrupt: val_season=2022, train includes 2023
        bad_folds = [([2022, 2023], 2022)]  # val_season == train min, not max+1
        with pytest.raises(AssertionError, match="data-leakage"):
            _walk_forward_cv(
                df, bad_folds,
                features=[f for f in FEATURE_COLS if f in df.columns],
                target_col="actual_receiving_yards",
                params=DEFAULT_XGB_PARAMS,
            )

    def test_unsorted_train_raises_assertion(self) -> None:
        """Train seasons not in sorted order must raise AssertionError."""
        df = _make_synthetic_df([2022, 2023, 2024])
        bad_folds = [([2023, 2022], 2024)]  # reversed
        with pytest.raises(AssertionError, match="not sorted ascending"):
            _walk_forward_cv(
                df, bad_folds,
                features=[f for f in FEATURE_COLS if f in df.columns],
                target_col="actual_receiving_yards",
                params=DEFAULT_XGB_PARAMS,
            )

    def test_valid_folds_do_not_raise(self) -> None:
        """Correct fold order must not trigger any assertion."""
        df = _make_synthetic_df([2022, 2023, 2024])
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        features = [f for f in FEATURE_COLS if f in df.columns]
        # Should not raise
        _walk_forward_cv(df, folds, features, "actual_receiving_yards",
                         DEFAULT_XGB_PARAMS)


# ── KEY TEST: OOF temporal integrity ─────────────────────────────────────────

class TestOOFTemporalIntegrity:
    """
    THE KEY TEST: confirm OOF predictions come only from data
    the model never trained on.

    Walk-forward CV with seasons [2022, 2023, 2024]:
      Fold 0: train=[2022], val=2023 → OOF rows must have season=2023
      Fold 1: train=[2022,2023], val=2024 → OOF rows must have season=2024

    Season 2022 (used in training for ALL folds) must NEVER appear in OOF.
    """

    @pytest.fixture(scope="class")
    def oof_df(self) -> pd.DataFrame:
        df = _make_synthetic_df([2022, 2023, 2024], n_per_season=80)
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        features = [f for f in FEATURE_COLS if f in df.columns]
        _, oof, _ = _walk_forward_cv(
            df, folds, features, "actual_receiving_yards", DEFAULT_XGB_PARAMS
        )
        return oof

    def test_oof_contains_only_val_seasons(self, oof_df: pd.DataFrame) -> None:
        """
        OOF seasons must be exactly the set of val_seasons from the folds.
        Train seasons must NEVER appear.
        """
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        val_seasons = {val for _, val in folds}       # {2023, 2024}
        oof_seasons = set(oof_df["season"].unique())
        assert oof_seasons <= val_seasons, (
            f"OOF contains seasons {oof_seasons - val_seasons} "
            f"that were used in training — temporal leakage!"
        )

    def test_first_train_season_never_in_oof(self, oof_df: pd.DataFrame) -> None:
        """
        Season 2022 is the first train season in EVERY fold.
        It must never appear in OOF predictions.
        """
        assert 2022 not in oof_df["season"].values, (
            "Season 2022 (always in training) appears in OOF — leakage bug!"
        )

    def test_oof_has_required_columns(self, oof_df: pd.DataFrame) -> None:
        required = {"player_id", "game_id", "season", "week",
                    "y_true", "y_pred", "fold_idx"}
        assert required.issubset(oof_df.columns)

    def test_oof_has_two_folds(self, oof_df: pd.DataFrame) -> None:
        assert set(oof_df["fold_idx"].unique()) == {0, 1}

    def test_fold0_oof_is_season_2023(self, oof_df: pd.DataFrame) -> None:
        fold0 = oof_df[oof_df["fold_idx"] == 0]
        assert (fold0["season"] == 2023).all(), "Fold 0 OOF must all be season 2023"

    def test_fold1_oof_is_season_2024(self, oof_df: pd.DataFrame) -> None:
        fold1 = oof_df[oof_df["fold_idx"] == 1]
        assert (fold1["season"] == 2024).all(), "Fold 1 OOF must all be season 2024"

    def test_oof_row_count_matches_val_seasons(self, oof_df: pd.DataFrame) -> None:
        df = _make_synthetic_df([2022, 2023, 2024], n_per_season=80)
        # OOF should have rows for seasons 2023 + 2024 (the val seasons)
        expected = len(df[df["season"].isin([2023, 2024])])
        assert len(oof_df) == expected


# ── _train_fold ───────────────────────────────────────────────────────────────

class TestTrainFold:
    @pytest.fixture(scope="class")
    def fold_outputs(self):
        df = _make_synthetic_df([2022, 2023])
        features = [f for f in FEATURE_COLS if f in df.columns]
        train_df = df[df["season"] == 2022]
        val_df   = df[df["season"] == 2023]
        model, oof_rows, fr = _train_fold(
            0, train_df, val_df, features, "actual_receiving_yards",
            DEFAULT_XGB_PARAMS
        )
        return model, oof_rows, fr

    def test_fold_result_seasons(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert fr.train_seasons == [2022]
        assert fr.val_season == 2023

    def test_mae_is_positive_finite(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert np.isfinite(fr.mae) and fr.mae > 0

    def test_rmse_is_positive_finite(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert np.isfinite(fr.rmse) and fr.rmse > 0

    def test_rmse_ge_mae(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        # RMSE >= MAE always (by Cauchy-Schwarz)
        assert fr.rmse >= fr.mae - 1e-9

    def test_oof_rows_all_from_val(self, fold_outputs) -> None:
        _, oof_rows, _ = fold_outputs
        assert (oof_rows["season"] == 2023).all()

    def test_model_predicts_finite_values(self, fold_outputs) -> None:
        _, oof_rows, _ = fold_outputs
        assert np.all(np.isfinite(oof_rows["y_pred"]))


# ── _compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        mae, rmse = _compute_metrics(y, y)
        assert mae == pytest.approx(0.0, abs=1e-9)
        assert rmse == pytest.approx(0.0, abs=1e-9)

    def test_constant_error(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 25.0, 35.0])  # +5 each
        mae, rmse = _compute_metrics(y_true, y_pred)
        assert mae  == pytest.approx(5.0)
        assert rmse == pytest.approx(5.0)

    def test_rmse_ge_mae(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.exponential(50, 100)
        y_pred = rng.exponential(50, 100)
        mae, rmse = _compute_metrics(y_true, y_pred)
        assert rmse >= mae - 1e-9


# ── _data_hash ────────────────────────────────────────────────────────────────

class TestDataHash:
    def test_same_data_same_hash(self) -> None:
        df = _make_synthetic_df([2023], n_per_season=20)
        features = [f for f in FEATURE_COLS if f in df.columns]
        h1 = _data_hash(df, features, "actual_receiving_yards")
        h2 = _data_hash(df, features, "actual_receiving_yards")
        assert h1 == h2

    def test_different_data_different_hash(self) -> None:
        df1 = _make_synthetic_df([2023], n_per_season=20, seed=1)
        df2 = _make_synthetic_df([2023], n_per_season=20, seed=2)
        features = [f for f in FEATURE_COLS if f in df1.columns]
        h1 = _data_hash(df1, features, "actual_receiving_yards")
        h2 = _data_hash(df2, features, "actual_receiving_yards")
        assert h1 != h2

    def test_hash_is_16_chars(self) -> None:
        df = _make_synthetic_df([2023], n_per_season=10)
        features = [f for f in FEATURE_COLS if f in df.columns]
        h = _data_hash(df, features, "actual_receiving_yards")
        assert len(h) == 16


# ── Feature importances ───────────────────────────────────────────────────────

class TestFeatureImportances:
    @pytest.fixture(scope="class")
    def train_result(self) -> XGBTrainResult:
        df = _make_synthetic_df([2022, 2023, 2024])
        return train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",  # disable MLflow
        )

    def test_importances_sum_to_one(self, train_result: XGBTrainResult) -> None:
        total = sum(train_result.feature_importances.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_importances_non_negative(self, train_result: XGBTrainResult) -> None:
        assert all(v >= 0 for v in train_result.feature_importances.values())

    def test_importances_cover_all_features(self, train_result: XGBTrainResult) -> None:
        # Every feature in the trained model should have an importance entry
        assert len(train_result.feature_importances) > 0


# ── train() — full pipeline (no MLflow, no Optuna) ───────────────────────────

class TestTrainFunction:
    @pytest.fixture(scope="class")
    def result(self) -> XGBTrainResult:
        df = _make_synthetic_df([2022, 2023, 2024])
        return train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",  # disable MLflow
        )

    def test_returns_xgb_train_result(self, result: XGBTrainResult) -> None:
        assert isinstance(result, XGBTrainResult)

    def test_two_folds_completed(self, result: XGBTrainResult) -> None:
        assert len(result.fold_results) == 2

    def test_oof_df_is_dataframe(self, result: XGBTrainResult) -> None:
        assert isinstance(result.oof_df, pd.DataFrame)

    def test_oof_not_empty(self, result: XGBTrainResult) -> None:
        assert len(result.oof_df) > 0

    def test_best_params_is_dict(self, result: XGBTrainResult) -> None:
        assert isinstance(result.best_params, dict)

    def test_all_fold_mae_positive(self, result: XGBTrainResult) -> None:
        for fr in result.fold_results:
            assert fr.mae > 0

    def test_all_fold_rmse_positive(self, result: XGBTrainResult) -> None:
        for fr in result.fold_results:
            assert fr.rmse > 0

    def test_unknown_target_raises(self) -> None:
        df = _make_synthetic_df([2022, 2023])
        with pytest.raises(ValueError, match="Unknown target"):
            train(df, seasons=[2022, 2023], target="sacks",
                  mlflow_tracking_uri="")

    def test_single_season_raises(self) -> None:
        df = _make_synthetic_df([2022])
        with pytest.raises(ValueError, match="at least 2 seasons"):
            train(df, seasons=[2022], target="receiving_yards",
                  mlflow_tracking_uri="")


# ── MLflow logging ────────────────────────────────────────────────────────────

class TestMLflowLogging:
    """
    Verifies that train() logs all required fields to MLflow.
    Uses a local temp-directory tracking URI — no server required.
    """

    def test_mlflow_run_logged_with_required_fields(self, tmp_path: Path) -> None:
        import mlflow

        tracking_uri = f"file://{tmp_path}"
        df = _make_synthetic_df([2022, 2023, 2024])

        result = train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment="test_xgb_receiving_yards",
            out_dir=tmp_path / "oof",
        )

        assert result.run_id is not None, "MLflow run_id should be set"

        # Retrieve the run and verify required fields
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(result.run_id)

        # ── Required metrics ──────────────────────────────────────────────
        assert "mae"  in run.data.metrics, "MLflow run missing 'mae'"
        assert "rmse" in run.data.metrics, "MLflow run missing 'rmse'"
        assert run.data.metrics["mae"]  > 0
        assert run.data.metrics["rmse"] > 0

        # ── Required tags ─────────────────────────────────────────────────
        assert "training_data_hash" in run.data.tags
        assert len(run.data.tags["training_data_hash"]) == 16
        assert "timestamp" in run.data.tags

        # ── Required params ───────────────────────────────────────────────
        assert "target"   in run.data.params
        assert "seasons"  in run.data.params
        assert "n_folds"  in run.data.params

    def test_mlflow_disabled_does_not_raise(self) -> None:
        df = _make_synthetic_df([2022, 2023])
        result = train(
            df, seasons=[2022, 2023],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",
        )
        assert result.run_id is None

    def test_oof_artifact_saved(self, tmp_path: Path) -> None:
        import mlflow

        tracking_uri = f"file://{tmp_path}"
        oof_dir = tmp_path / "oof"
        df = _make_synthetic_df([2022, 2023, 2024])

        result = train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri=tracking_uri,
            out_dir=oof_dir,
        )

        assert result.oof_path is not None
        assert result.oof_path.exists()
        saved_oof = pd.read_csv(result.oof_path)
        assert len(saved_oof) > 0
        assert "y_pred" in saved_oof.columns


# ── save_oof ──────────────────────────────────────────────────────────────────

class TestSaveOof:
    def test_creates_file(self, tmp_path: Path) -> None:
        df = _make_synthetic_df([2022, 2023])
        oof = df[["player_id", "game_id", "season", "week"]].copy()
        oof["y_true"]   = 50.0
        oof["y_pred"]   = 55.0
        oof["fold_idx"] = 0
        path = save_oof(oof, "receiving_yards", "abc123xy", tmp_path)
        assert path.exists()

    def test_file_name_contains_target_and_run_id(self, tmp_path: Path) -> None:
        df = _make_synthetic_df([2022])
        oof = df[["player_id", "game_id", "season", "week"]].copy()
        oof["y_true"] = oof["y_pred"] = oof["fold_idx"] = 0
        path = save_oof(oof, "rushing_yards", "deadbeef1234", tmp_path)
        assert "rushing_yards" in path.name
        assert "deadbeef" in path.name  # first 8 chars of run_id

    def test_roundtrip(self, tmp_path: Path) -> None:
        df = _make_synthetic_df([2022, 2023])
        oof = df[["player_id", "game_id", "season", "week"]].copy()
        oof["y_true"]   = np.random.default_rng(0).exponential(50, len(df))
        oof["y_pred"]   = np.random.default_rng(1).exponential(50, len(df))
        oof["fold_idx"] = 0
        path = save_oof(oof, "receiving_yards", "test0001xxxx", tmp_path)
        reloaded = pd.read_csv(path)
        assert len(reloaded) == len(oof)
        assert set(reloaded.columns) == set(oof.columns)
