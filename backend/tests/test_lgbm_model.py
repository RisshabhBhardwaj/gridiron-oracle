"""
backend/tests/test_lgbm_model.py

Tests for ml/lgbm_model.py — LightGBM base learner.

Mirrors test_xgb_model.py in structure (same test classes, same assertions)
but for LightGBM-specific internals:
  - num_leaves in best_params (not max_depth)
  - DEFAULT_LGBM_PARAMS keys
  - lgbm_ prefix in OOF filename
  - Feature importances are gain-based (booster_.feature_importance)

KEY TEST: TestOOFTemporalIntegrity — identical proof to test_xgb_model.py:
  OOF rows come only from seasons the model never trained on.
  Season 2022 (training-only) must never appear in OOF.
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
    FEATURE_COLS,
    TARGET_COL_MAP,
    FoldResult,
    _compute_metrics,
    _data_hash,
    _make_walk_forward_folds,
    save_oof,
)
from ml.lgbm_model import (
    DEFAULT_LGBM_PARAMS,
    LGBMTrainResult,
    _run_optuna,
    _train_fold,
    _walk_forward_cv,
    train,
)


# ── Synthetic data factory ────────────────────────────────────────────────────

def _make_synthetic_df(
    seasons: list[int],
    n_per_season: int = 60,
    position: str = "WR",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Same factory as in test_xgb_model.py — synthetic feature_matrix rows
    with realistic NaN sprinkle (10% missing per feature column).
    """
    rows = []
    for season in seasons:
        s_rng = np.random.default_rng(seed + season)
        n = n_per_season
        row = {
            "player_id": [f"P{season}_{i:04d}" for i in range(n)],
            "game_id":   [f"{season}_01_AA_BB_{i}" for i in range(n)],
            "season":    [season] * n,
            "week":      s_rng.integers(1, 18, n).tolist(),
            "position":  [position] * n,
            "team":      ["MIN"] * n,
        }
        for feat in FEATURE_COLS:
            vals = s_rng.standard_normal(n) * 10 + 5
            mask = s_rng.random(n) < 0.10
            vals[mask] = np.nan
            row[feat] = vals.tolist()

        row["actual_receiving_yards"] = s_rng.exponential(45, n).tolist()
        row["actual_rushing_yards"]   = s_rng.exponential(20, n).tolist()
        row["actual_passing_yards"]   = s_rng.exponential(200, n).tolist()
        row["actual_fantasy_ppr"]     = s_rng.exponential(12, n).tolist()

        rows.append(pd.DataFrame(row))

    return pd.concat(rows, ignore_index=True)


# ── LightGBM-specific defaults ────────────────────────────────────────────────

class TestDefaultParams:
    def test_num_leaves_present(self) -> None:
        """num_leaves is the primary complexity param in LightGBM (not max_depth)."""
        assert "num_leaves" in DEFAULT_LGBM_PARAMS

    def test_no_max_depth_key(self) -> None:
        """max_depth is XGBoost's level-wise parameter. LightGBM uses num_leaves."""
        assert "max_depth" not in DEFAULT_LGBM_PARAMS

    def test_lgbm_specific_sampling_params(self) -> None:
        """feature_fraction + bagging_fraction + bagging_freq are LightGBM-specific."""
        assert "feature_fraction"  in DEFAULT_LGBM_PARAMS
        assert "bagging_fraction"  in DEFAULT_LGBM_PARAMS
        assert "bagging_freq"      in DEFAULT_LGBM_PARAMS

    def test_lambda_l1_l2_not_reg_alpha_lambda(self) -> None:
        """LightGBM uses lambda_l1/lambda_l2, not reg_alpha/reg_lambda (XGBoost)."""
        assert "lambda_l1"   in DEFAULT_LGBM_PARAMS
        assert "lambda_l2"   in DEFAULT_LGBM_PARAMS
        assert "reg_alpha"   not in DEFAULT_LGBM_PARAMS
        assert "reg_lambda"  not in DEFAULT_LGBM_PARAMS

    def test_min_child_samples_present(self) -> None:
        """min_child_samples (not min_child_weight) is LightGBM's leaf-size guard."""
        assert "min_child_samples" in DEFAULT_LGBM_PARAMS
        assert "min_child_weight"  not in DEFAULT_LGBM_PARAMS

    def test_num_leaves_is_63(self) -> None:
        """Default num_leaves=63 gives ~2^6-1 leaves, comparable to max_depth=6."""
        assert DEFAULT_LGBM_PARAMS["num_leaves"] == 63

    def test_verbosity_minus_one(self) -> None:
        """verbosity=-1 silences LightGBM INFO prints during training."""
        assert DEFAULT_LGBM_PARAMS["verbosity"] == -1


# ── Temporal ordering assertions ──────────────────────────────────────────────

class TestTemporalAssertions:
    """Same hard-guard tests as TestTemporalAssertions in test_xgb_model.py."""

    def test_bad_fold_order_raises_assertion(self) -> None:
        df = _make_synthetic_df([2022, 2023, 2024])
        bad_folds = [([2022, 2023], 2022)]   # val_season in training window
        features = [f for f in FEATURE_COLS if f in df.columns]
        with pytest.raises(AssertionError, match="data-leakage"):
            _walk_forward_cv(
                df, bad_folds, features, "actual_receiving_yards",
                DEFAULT_LGBM_PARAMS,
            )

    def test_unsorted_train_raises_assertion(self) -> None:
        df = _make_synthetic_df([2022, 2023, 2024])
        bad_folds = [([2023, 2022], 2024)]   # reversed train seasons
        features = [f for f in FEATURE_COLS if f in df.columns]
        with pytest.raises(AssertionError, match="not sorted ascending"):
            _walk_forward_cv(
                df, bad_folds, features, "actual_receiving_yards",
                DEFAULT_LGBM_PARAMS,
            )

    def test_valid_folds_do_not_raise(self) -> None:
        df = _make_synthetic_df([2022, 2023, 2024])
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        features = [f for f in FEATURE_COLS if f in df.columns]
        _walk_forward_cv(df, folds, features, "actual_receiving_yards",
                         DEFAULT_LGBM_PARAMS)


# ── KEY TEST: OOF temporal integrity ─────────────────────────────────────────

class TestOOFTemporalIntegrity:
    """
    THE KEY TEST: confirm LGBM OOF predictions come only from data the model
    never trained on.

    Walk-forward CV over [2022, 2023, 2024]:
      Fold 0: train=[2022],       val=2023 → OOF rows must have season=2023
      Fold 1: train=[2022,2023],  val=2024 → OOF rows must have season=2024

    Season 2022 (training data in ALL folds) must NEVER appear in OOF.
    """

    @pytest.fixture(scope="class")
    def oof_df(self) -> pd.DataFrame:
        df = _make_synthetic_df([2022, 2023, 2024], n_per_season=80)
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        features = [f for f in FEATURE_COLS if f in df.columns]
        _, oof, _ = _walk_forward_cv(
            df, folds, features, "actual_receiving_yards", DEFAULT_LGBM_PARAMS
        )
        return oof

    def test_oof_contains_only_val_seasons(self, oof_df: pd.DataFrame) -> None:
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        val_seasons = {val for _, val in folds}       # {2023, 2024}
        oof_seasons = set(oof_df["season"].unique())
        assert oof_seasons <= val_seasons, (
            f"OOF contains training seasons {oof_seasons - val_seasons} — leakage!"
        )

    def test_first_train_season_never_in_oof(self, oof_df: pd.DataFrame) -> None:
        assert 2022 not in oof_df["season"].values, (
            "Season 2022 (always in training) appears in OOF — temporal leakage!"
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
        expected = len(df[df["season"].isin([2023, 2024])])
        assert len(oof_df) == expected

    def test_oof_format_identical_to_xgb(self, oof_df: pd.DataFrame) -> None:
        """
        OOF columns must be identical to what xgb_model produces.
        The stacking meta-learner concatenates both OOF files — format parity
        is non-negotiable.
        """
        expected_cols = {"player_id", "game_id", "season", "week", "position",
                         "y_true", "y_pred", "fold_idx"}
        assert set(oof_df.columns) == expected_cols


# ── _train_fold ───────────────────────────────────────────────────────────────

class TestTrainFold:
    @pytest.fixture(scope="class")
    def fold_outputs(self):
        df = _make_synthetic_df([2022, 2023])
        features = [f for f in FEATURE_COLS if f in df.columns]
        train_df = df[df["season"] == 2022]
        val_df   = df[df["season"] == 2023]
        return _train_fold(0, train_df, val_df, features,
                           "actual_receiving_yards", DEFAULT_LGBM_PARAMS)

    def test_fold_result_seasons(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert fr.train_seasons == [2022]
        assert fr.val_season == 2023

    def test_mae_positive_finite(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert np.isfinite(fr.mae) and fr.mae > 0

    def test_rmse_positive_finite(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert np.isfinite(fr.rmse) and fr.rmse > 0

    def test_rmse_ge_mae(self, fold_outputs) -> None:
        _, _, fr = fold_outputs
        assert fr.rmse >= fr.mae - 1e-9

    def test_oof_rows_all_from_val(self, fold_outputs) -> None:
        _, oof_rows, _ = fold_outputs
        assert (oof_rows["season"] == 2023).all()

    def test_model_predicts_finite(self, fold_outputs) -> None:
        _, oof_rows, _ = fold_outputs
        assert np.all(np.isfinite(oof_rows["y_pred"]))

    def test_model_is_lgbm_regressor(self, fold_outputs) -> None:
        import lightgbm as lgb
        model, _, _ = fold_outputs
        assert isinstance(model, lgb.LGBMRegressor)


# ── Feature importances ───────────────────────────────────────────────────────

class TestFeatureImportances:
    @pytest.fixture(scope="class")
    def train_result(self) -> LGBMTrainResult:
        df = _make_synthetic_df([2022, 2023, 2024])
        return train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",
        )

    def test_importances_sum_to_one(self, train_result: LGBMTrainResult) -> None:
        total = sum(train_result.feature_importances.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_importances_non_negative(self, train_result: LGBMTrainResult) -> None:
        assert all(v >= 0 for v in train_result.feature_importances.values())

    def test_gain_based_importances(self, train_result: LGBMTrainResult) -> None:
        """
        LightGBM uses gain-based importances (total split gain per feature),
        which is more informative than split-count.
        Verify some features have non-zero gain (model actually used them).
        """
        non_zero = sum(1 for v in train_result.feature_importances.values() if v > 0)
        assert non_zero > 0


# ── train() — full pipeline ───────────────────────────────────────────────────

class TestTrainFunction:
    @pytest.fixture(scope="class")
    def result(self) -> LGBMTrainResult:
        df = _make_synthetic_df([2022, 2023, 2024])
        return train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",
        )

    def test_returns_lgbm_train_result(self, result: LGBMTrainResult) -> None:
        assert isinstance(result, LGBMTrainResult)

    def test_two_folds_completed(self, result: LGBMTrainResult) -> None:
        assert len(result.fold_results) == 2

    def test_oof_df_is_dataframe(self, result: LGBMTrainResult) -> None:
        assert isinstance(result.oof_df, pd.DataFrame)

    def test_oof_not_empty(self, result: LGBMTrainResult) -> None:
        assert len(result.oof_df) > 0

    def test_best_params_has_num_leaves(self, result: LGBMTrainResult) -> None:
        """num_leaves must be in best_params — verifies LightGBM-specific search."""
        assert "num_leaves" in result.best_params

    def test_best_params_no_max_depth(self, result: LGBMTrainResult) -> None:
        """max_depth must not be in best_params — this is XGBoost's parameter."""
        assert "max_depth" not in result.best_params

    def test_all_fold_mae_positive(self, result: LGBMTrainResult) -> None:
        for fr in result.fold_results:
            assert fr.mae > 0

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
            mlflow_experiment="test_lgbm_receiving_yards",
            out_dir=tmp_path / "oof",
        )

        assert result.run_id is not None

        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(result.run_id)

        assert "mae"   in run.data.metrics
        assert "rmse"  in run.data.metrics
        assert run.data.metrics["mae"]  > 0
        assert run.data.metrics["rmse"] > 0
        assert "training_data_hash" in run.data.tags
        assert len(run.data.tags["training_data_hash"]) == 16
        assert "timestamp" in run.data.tags
        assert "target"  in run.data.params
        assert "seasons" in run.data.params
        assert "n_folds" in run.data.params

    def test_mlflow_disabled_run_id_is_none(self) -> None:
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
        df = _make_synthetic_df([2022, 2023, 2024])
        result = train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri=f"file://{tmp_path}",
            out_dir=tmp_path / "oof",
        )
        assert result.oof_path is not None
        assert result.oof_path.exists()
        assert "lgbm_" in result.oof_path.name


# ── OOF filename prefix ───────────────────────────────────────────────────────

class TestOofPrefix:
    def test_lgbm_prefix_in_filename(self, tmp_path: Path) -> None:
        df = _make_synthetic_df([2022, 2023])
        result = train(
            df, seasons=[2022, 2023],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",
            out_dir=tmp_path,
        )
        assert result.oof_path is not None
        assert result.oof_path.name.startswith("lgbm_")

    def test_xgb_prefix_unchanged(self, tmp_path: Path) -> None:
        """Verify save_oof default prefix is still 'xgb' (backward-compat)."""
        df = _make_synthetic_df([2022, 2023])
        features = [f for f in FEATURE_COLS if f in df.columns]
        oof = df[["player_id", "game_id", "season", "week"]].copy()
        oof["y_true"] = oof["y_pred"] = oof["fold_idx"] = 0.0
        path = save_oof(oof, "receiving_yards", "abc12345", tmp_path)
        assert path.name.startswith("xgb_")

    def test_lgbm_oof_has_same_columns_as_xgb_oof(self, tmp_path: Path) -> None:
        """
        Stacking meta-learner requires identical column schemas.
        Both XGB and LGBM OOF files must have the same columns.
        """
        df = _make_synthetic_df([2022, 2023, 2024])
        from ml.xgb_model import train as xgb_train

        xgb_result = xgb_train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",
        )
        lgbm_result = train(
            df, seasons=[2022, 2023, 2024],
            target="receiving_yards",
            n_optuna_trials=0,
            position_filter=None,
            mlflow_tracking_uri="",
        )
        assert set(xgb_result.oof_df.columns) == set(lgbm_result.oof_df.columns)
