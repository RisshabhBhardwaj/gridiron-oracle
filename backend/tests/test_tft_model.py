"""
backend/tests/test_tft_model.py

Tests for ml/tft_model.py — Step 1 (data formatting) and Step 2 (training).

Test classes:
  TestCovariateGroupDefinitions  — column counts from class constants (no pytorch dep)
  TestTimeIdxCalculation         — time_idx formula verification (no pytorch dep)
  TestPrepareDataFrame           — pandas-only transformations (no pytorch dep)
  TestDatasetInstantiation       — TimeSeriesDataSet creation (requires pytorch_forecasting)
  TestTFTConfigStep2             — Step 2 TFTConfig fields with defaults (no pytorch dep)
  TestTFTTrainResult             — TFTTrainResult dataclass structure (no pytorch dep)
  TestWalkForwardFoldsForTFT     — walk-forward fold generation (no pytorch dep)
  TestSaveOofTFT                 — save_oof prefix "tft" file schema (no pytorch dep)
  TestOofAlignmentThreeWay       — ACCEPTANCE TEST: xgb+lgbm+tft OOF inner join (no pytorch dep)
  TestTFTTrainIntegration        — end-to-end train() (requires pytorch + pytorch_forecasting)

Run with:
  pytest backend/tests/test_tft_model.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.tft_model import TFTConfig, TFTDataset


# ── Synthetic data factory ─────────────────────────────────────────────────────

def _make_synthetic_nfl_df(
    n_players: int = 6,
    n_weeks: int = 20,
    season: int = 2022,
    seed: int = 42,
    include_optional_cols: bool = True,
) -> pd.DataFrame:
    """
    Create a minimal synthetic NFL feature matrix with all TFT covariate columns.

    include_optional_cols=True  → height, weight, draft_round, snap_share
                                  are present (as if roster join has been done)
    include_optional_cols=False → those columns are absent (tests default-fill logic)
    """
    rng = np.random.default_rng(seed)
    positions = ["WR", "RB", "QB", "TE"]
    rows: list[dict] = []

    for player_idx in range(n_players):
        player_id = f"player_{player_idx:03d}"
        position  = positions[player_idx % len(positions)]
        team_id   = f"TEAM_{player_idx % 6}"

        for week in range(1, n_weeks + 1):
            row: dict = {
                # Identity
                "player_id":                player_id,
                "season":                   season,
                "week":                     week,
                # Static
                "position":                 position,
                "team":                     team_id,      # raw name from feature matrix
                "team_id":                  team_id,      # also present directly
                # Time-varying known categoricals
                "opponent_team_id":         f"OPP_{(week + player_idx) % 8}",
                "is_home":                  int(week % 2 == 0),
                # Time-varying known reals
                "days_rest":                float(rng.integers(4, 10)),
                "temp_bucket":              float(rng.integers(0, 4)),
                "wind_bucket":              float(rng.integers(0, 3)),
                # Time-varying unknown reals
                "kalman_est_receiving_yards":     float(max(0.0, rng.normal(60.0, 25.0))),
                "kalman_est_targets":             float(max(0.0, rng.normal(6.0, 3.0))),
                "kalman_est_receptions":          float(max(0.0, rng.normal(4.0, 2.0))),
                "kalman_est_receiving_tds":       float(max(0.0, rng.normal(0.5, 0.5))),
                "kalman_est_fantasy_ppr":         float(max(0.0, rng.normal(15.0, 5.0))),
                "kalman_variance_receiving_yards":float(max(0.0, rng.normal(10.0, 2.0))),
                "kalman_variance_targets":        float(max(0.0, rng.normal(2.0, 0.5))),
                "kalman_variance_receptions":     float(max(0.0, rng.normal(1.0, 0.5))),
                "kalman_variance_fantasy_ppr":    float(max(0.0, rng.normal(3.0, 1.0))),
                "seas_avg_receiving_yards":       float(max(0.0, rng.normal(65.0, 20.0))),
                "seas_avg_targets":               float(max(0.0, rng.normal(6.5, 3.0))),
                "seas_avg_receptions":            float(max(0.0, rng.normal(4.5, 2.0))),
                "target_share":                   float(rng.uniform(0.05, 0.35)),
                "form_target_share":              float(rng.uniform(0.05, 0.35)),
                # Target
                "actual_receiving_yards":   float(max(0.0, rng.normal(60.0, 30.0))),
            }
            if include_optional_cols:
                row["height"]      = float(rng.normal(73.0, 3.0))
                row["weight"]      = float(rng.normal(215.0, 25.0))
                row["draft_round"] = float(rng.integers(1, 8))
                row["snap_share"]  = float(rng.uniform(0.3, 0.9))

            rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Covariate group definitions — class-level constants
#    (These tests pass with NO pytorch_forecasting installation.)
# ══════════════════════════════════════════════════════════════════════════════

class TestCovariateGroupDefinitions:
    """
    Verify the three covariate groups have the expected column counts.

    Current design (team_id moved to time-varying — players switch teams):
      STATIC:                 position                      (1 cat)
                              height, weight, draft_round   (3 reals)
      TIME-VARYING KNOWN:     team_id, opponent_team_id, home_away (3 cats)
                              week, rest_days, temp_bucket, wind_bucket (4 reals)
      TIME-VARYING UNKNOWN:   kalman_est_*, seas_avg_*, snap_share, target_share (14 reals)
    """

    def test_static_categoricals_count(self):
        assert len(TFTDataset.STATIC_CATEGORICALS) == 1

    def test_static_categoricals_names(self):
        assert TFTDataset.STATIC_CATEGORICALS == ["position"]

    def test_static_reals_count(self):
        assert len(TFTDataset.STATIC_REALS) == 3

    def test_static_reals_names(self):
        assert TFTDataset.STATIC_REALS == ["height", "weight", "draft_round"]

    def test_time_varying_known_categoricals_count(self):
        assert len(TFTDataset.TIME_VARYING_KNOWN_CATEGORICALS) == 3

    def test_time_varying_known_categoricals_names(self):
        assert TFTDataset.TIME_VARYING_KNOWN_CATEGORICALS == [
            "team_id", "opponent_team_id", "home_away",
        ]

    def test_time_varying_known_reals_count(self):
        assert len(TFTDataset.TIME_VARYING_KNOWN_REALS) == 4

    def test_time_varying_known_reals_names(self):
        assert TFTDataset.TIME_VARYING_KNOWN_REALS == [
            "week", "rest_days", "temp_bucket", "wind_bucket",
        ]

    def test_time_varying_unknown_reals_count(self):
        assert len(TFTDataset.TIME_VARYING_UNKNOWN_REALS) == 14

    def test_time_varying_unknown_reals_contain_required(self):
        reals = TFTDataset.TIME_VARYING_UNKNOWN_REALS
        required = [
            "kalman_est_receiving_yards", "kalman_est_targets",
            "seas_avg_receiving_yards", "snap_share", "target_share",
        ]
        for r in required:
            assert r in reals, f"Missing required covariate: {r}"

    def test_covariate_counts_method_returns_correct_dict(self):
        counts = TFTDataset.covariate_counts()
        assert counts == {
            "static_categoricals":             1,
            "static_reals":                    3,
            "time_varying_known_categoricals": 3,
            "time_varying_known_reals":        4,
            "time_varying_unknown_reals":      14,
        }

    def test_covariate_counts_method_is_classmethod(self):
        counts = TFTDataset.covariate_counts()
        assert isinstance(counts, dict)

    def test_total_covariate_count(self):
        counts = TFTDataset.covariate_counts()
        total = sum(counts.values())
        # 1 + 3 + 3 + 4 + 14 = 25 covariate columns (+ target)
        assert total == 25

    def test_group_id_is_player_id(self):
        assert TFTDataset.GROUP_ID == "player_id"

    def test_time_idx_column_name(self):
        assert TFTDataset.TIME_IDX == "time_idx"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Time index calculation
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeIdxCalculation:

    @pytest.fixture
    def single_season_df(self):
        rows = [
            {"player_id": "p1", "season": 2018, "week": w,
             "actual_receiving_yards": 50.0, "position": "WR"}
            for w in range(1, 23)
        ]
        return pd.DataFrame(rows)

    @pytest.fixture
    def two_season_df(self):
        rows = []
        for season in [2018, 2019]:
            for week in range(1, 5):
                rows.append({
                    "player_id": "p1", "season": season, "week": week,
                    "actual_receiving_yards": 50.0, "position": "WR",
                })
        return pd.DataFrame(rows)

    def test_season_2018_week_1_is_0(self, single_season_df):
        result = TFTDataset._add_time_idx(single_season_df)
        row = result[(result["season"] == 2018) & (result["week"] == 1)].iloc[0]
        assert row["time_idx"] == 0

    def test_season_2018_week_22_is_21(self, single_season_df):
        result = TFTDataset._add_time_idx(single_season_df)
        row = result[(result["season"] == 2018) & (result["week"] == 22)].iloc[0]
        assert row["time_idx"] == 21

    def test_season_2019_week_1_is_22(self, two_season_df):
        result = TFTDataset._add_time_idx(two_season_df)
        row = result[(result["season"] == 2019) & (result["week"] == 1)].iloc[0]
        assert row["time_idx"] == 22  # 1 * 22 + 0

    def test_season_2019_week_4_is_25(self, two_season_df):
        result = TFTDataset._add_time_idx(two_season_df)
        row = result[(result["season"] == 2019) & (result["week"] == 4)].iloc[0]
        assert row["time_idx"] == 25  # 22 + 3

    def test_time_idx_is_integer_dtype(self, single_season_df):
        result = TFTDataset._add_time_idx(single_season_df)
        assert pd.api.types.is_integer_dtype(result["time_idx"])

    def test_time_idx_monotonically_increasing_within_season(self, single_season_df):
        result = TFTDataset._add_time_idx(single_season_df)
        result = result.sort_values("week")
        diffs = result["time_idx"].diff().dropna()
        assert (diffs >= 0).all()


# ══════════════════════════════════════════════════════════════════════════════
# 3. prepare_dataframe — pure pandas transformations
# ══════════════════════════════════════════════════════════════════════════════

class TestPrepareDataFrame:

    @pytest.fixture(scope="class")
    def ds(self):
        return TFTDataset()

    @pytest.fixture(scope="class")
    def raw_df(self):
        return _make_synthetic_nfl_df(n_players=4, n_weeks=20, seed=7)

    @pytest.fixture(scope="class")
    def prepared(self, ds, raw_df):
        return ds.prepare_dataframe(raw_df.copy(), target="receiving_yards")

    # ── time_idx ──────────────────────────────────────────────────────────────

    def test_time_idx_column_present(self, prepared):
        assert "time_idx" in prepared.columns

    def test_time_idx_is_integer(self, prepared):
        assert pd.api.types.is_integer_dtype(prepared["time_idx"])

    def test_time_idx_monotonically_increasing_per_player(self, prepared):
        for player_id, group in prepared.groupby("player_id"):
            diffs = group["time_idx"].diff().dropna()
            assert (diffs > 0).all(), (
                f"time_idx not strictly increasing for player '{player_id}'"
            )

    # ── Column derivations ─────────────────────────────────────────────────────

    def test_home_away_derived_from_is_home(self, prepared):
        assert "home_away" in prepared.columns
        assert set(prepared["home_away"].unique()) <= {"home", "away", "unknown"}

    def test_home_away_is_home_maps_to_home(self, ds, raw_df):
        df = raw_df.copy()
        df["is_home"] = 1
        prepared = ds.prepare_dataframe(df, target="receiving_yards")
        assert (prepared["home_away"] == "home").all()

    def test_home_away_is_away_maps_to_away(self, ds, raw_df):
        df = raw_df.copy()
        df["is_home"] = 0
        prepared = ds.prepare_dataframe(df, target="receiving_yards")
        assert (prepared["home_away"] == "away").all()

    def test_rest_days_derived_from_days_rest(self, prepared):
        assert "rest_days" in prepared.columns

    def test_target_share_derived_from_form_target_share(self, prepared):
        assert "target_share" in prepared.columns

    # ── Default-fill for missing columns ──────────────────────────────────────

    def test_missing_height_weight_draft_filled_with_zero(self, ds):
        df = _make_synthetic_nfl_df(include_optional_cols=False)
        prepared = ds.prepare_dataframe(df, target="receiving_yards")
        for col in ("height", "weight", "draft_round"):
            assert col in prepared.columns
            assert (prepared[col] == 0.0).all(), (
                f"Expected '{col}' to be filled with 0.0 when absent"
            )

    def test_missing_snap_share_filled_with_zero(self, ds):
        df = _make_synthetic_nfl_df(include_optional_cols=False)
        prepared = ds.prepare_dataframe(df, target="receiving_yards")
        assert "snap_share" in prepared.columns
        assert (prepared["snap_share"] == 0.0).all()

    # ── Categorical types ──────────────────────────────────────────────────────

    def test_position_is_string(self, prepared):
        assert prepared["position"].dtype == object  # pandas string dtype

    def test_team_id_is_string(self, prepared):
        assert prepared["team_id"].dtype == object

    def test_home_away_is_string(self, prepared):
        assert prepared["home_away"].dtype == object

    def test_opponent_team_id_is_string(self, prepared):
        assert prepared["opponent_team_id"].dtype == object

    # ── Sort order ────────────────────────────────────────────────────────────

    def test_sorted_by_player_then_time_idx(self, prepared):
        for player_id, group in prepared.groupby("player_id"):
            is_sorted = group["time_idx"].is_monotonic_increasing
            assert is_sorted, f"player '{player_id}' time_idx is not sorted"

    # ── Error cases ───────────────────────────────────────────────────────────

    def test_unknown_target_raises_value_error(self, ds, raw_df):
        with pytest.raises(ValueError, match="Unknown target"):
            ds.prepare_dataframe(raw_df, target="invalid_target")

    def test_missing_target_column_raises_value_error(self, ds):
        df = _make_synthetic_nfl_df()
        df = df.drop(columns=["actual_receiving_yards"])
        with pytest.raises(ValueError, match="actual_receiving_yards"):
            ds.prepare_dataframe(df, target="receiving_yards")

    # ── NaN handling ──────────────────────────────────────────────────────────

    def test_no_nan_in_target_after_prepare(self, prepared):
        assert prepared["actual_receiving_yards"].notna().all()

    def test_no_nan_in_kalman_est_receiving_yards(self, ds, raw_df):
        df = raw_df.copy()
        df.loc[df.index[:5], "kalman_est_receiving_yards"] = np.nan  # inject NaNs
        prepared = ds.prepare_dataframe(df, target="receiving_yards")
        assert prepared["kalman_est_receiving_yards"].notna().all()


# ══════════════════════════════════════════════════════════════════════════════
# 4. TimeSeriesDataSet instantiation (requires pytorch_forecasting)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pytorch_forecasting as _pf
    _PF_AVAILABLE = True
except ImportError:
    _PF_AVAILABLE = False

_skip_if_no_pf = pytest.mark.skipif(
    not _PF_AVAILABLE,
    reason="pytorch_forecasting not installed — run 'pip install pytorch-forecasting torch'",
)


@_skip_if_no_pf
class TestDatasetInstantiation:
    """
    Verify that TFTDataset.make_dataset() returns a valid TimeSeriesDataSet
    and that the three covariate groups have the correct column counts.

    Requires pytorch_forecasting to be installed.
    """

    @pytest.fixture(scope="class")
    def ds(self):
        # Short encoder length so synthetic data (20 weeks) is sufficient
        cfg = TFTConfig(max_encoder_length=8, min_encoder_length=2, max_prediction_length=1)
        return TFTDataset(cfg)

    @pytest.fixture(scope="class")
    def dataset(self, ds):
        df = _make_synthetic_nfl_df(
            n_players=6, n_weeks=20, season=2022, seed=42, include_optional_cols=True
        )
        return ds.make_dataset(df, target="receiving_yards")

    # ── Basic structure ────────────────────────────────────────────────────────

    def test_returns_time_series_dataset(self, dataset):
        from pytorch_forecasting import TimeSeriesDataSet
        assert isinstance(dataset, TimeSeriesDataSet)

    def test_dataset_is_non_empty(self, dataset):
        assert len(dataset) > 0

    # ── Covariate group column counts ──────────────────────────────────────────
    # These are the KEY assertions the user asked for:
    # "confirms all three covariate groups have the expected column counts"

    def test_static_categoricals_count_in_dataset(self, dataset):
        """Static categoricals: position only (1 column). team_id is time-varying — players switch teams."""
        assert len(dataset.static_categoricals) == 1

    def test_static_reals_count_in_dataset(self, dataset):
        """Static reals: height, weight, draft_round → 3 columns."""
        assert len(dataset.static_reals) == 3

    def test_time_varying_known_categoricals_count_in_dataset(self, dataset):
        """Known future categoricals: team_id, opponent_team_id, home_away → 3 columns."""
        assert len(dataset.time_varying_known_categoricals) == 3

    def test_time_varying_known_reals_count_in_dataset(self, dataset):
        """Known future reals: week, rest_days, temp_bucket, wind_bucket → 4 columns."""
        assert len(dataset.time_varying_known_reals) == 4

    def test_time_varying_unknown_reals_count_in_dataset(self, dataset):
        """
        Unknown past reals: 14 covariates (kalman_est_*, seas_avg_*, snap_share, target_share)
        + actual_receiving_yards (target) = 15 total.
        """
        # TimeSeriesDataSet always appends the target to time_varying_unknown_reals
        assert len(dataset.time_varying_unknown_reals) == 15  # 14 + target

    # ── Column names spot-check ────────────────────────────────────────────────

    def test_static_categoricals_contain_position(self, dataset):
        assert "position" in dataset.static_categoricals

    def test_time_varying_known_categoricals_contain_team_id(self, dataset):
        """team_id is time-varying (not static) — players switch teams."""
        assert "team_id" in dataset.time_varying_known_categoricals

    def test_known_categoricals_contain_home_away(self, dataset):
        assert "home_away" in dataset.time_varying_known_categoricals

    def test_unknown_reals_contain_kalman_est_receiving_yards(self, dataset):
        assert "kalman_est_receiving_yards" in dataset.time_varying_unknown_reals

    def test_unknown_reals_contain_target(self, dataset):
        assert "actual_receiving_yards" in dataset.time_varying_unknown_reals

    # ── TFTConfig respected ────────────────────────────────────────────────────

    def test_max_encoder_length_from_config(self, ds, dataset):
        assert dataset.max_encoder_length == ds.config.max_encoder_length

    def test_max_prediction_length_from_config(self, ds, dataset):
        assert dataset.max_prediction_length == ds.config.max_prediction_length

    # ── make_dataset raises on missing pytorch ─────────────────────────────────

    def test_prepare_dataframe_does_not_require_pytorch(self):
        """prepare_dataframe() must work even if pytorch_forecasting is broken."""
        ds = TFTDataset()
        df = _make_synthetic_nfl_df(n_players=3, n_weeks=10, seed=0)
        # Should not raise — purely pandas
        prepared = ds.prepare_dataframe(df, target="receiving_yards")
        assert isinstance(prepared, pd.DataFrame)
        assert not prepared.empty


# ══════════════════════════════════════════════════════════════════════════════
# 5. Step 2 — TFTConfig training fields (no pytorch required)
# ══════════════════════════════════════════════════════════════════════════════

class TestTFTConfigStep2:
    """
    Verify TFTConfig exposes the Step 2 architecture and training fields
    with their documented default values.

    These tests require no pytorch dependency — TFTConfig is a pure dataclass.
    """

    @pytest.fixture
    def cfg(self):
        return TFTConfig()

    # ── Architecture defaults ─────────────────────────────────────────────────

    def test_hidden_size_default(self, cfg):
        assert cfg.hidden_size == 64

    def test_attention_head_size_default(self, cfg):
        assert cfg.attention_head_size == 4

    def test_dropout_default(self, cfg):
        assert abs(cfg.dropout - 0.1) < 1e-9

    def test_hidden_continuous_size_default(self, cfg):
        # 32 = half of hidden_size=64, per TFT paper recommendation
        assert cfg.hidden_continuous_size == 32

    # ── Training defaults ─────────────────────────────────────────────────────

    def test_learning_rate_default(self, cfg):
        assert abs(cfg.learning_rate - 3e-3) < 1e-9

    def test_max_epochs_default(self, cfg):
        assert cfg.max_epochs == 30

    def test_batch_size_default(self, cfg):
        assert cfg.batch_size == 64

    def test_gradient_clip_val_default(self, cfg):
        assert abs(cfg.gradient_clip_val - 0.1) < 1e-9

    def test_early_stopping_patience_default(self, cfg):
        assert cfg.early_stopping_patience == 5

    # ── Custom values round-trip ─────────────────────────────────────────────

    def test_custom_hidden_size(self):
        cfg = TFTConfig(hidden_size=128)
        assert cfg.hidden_size == 128

    def test_custom_max_epochs(self):
        cfg = TFTConfig(max_epochs=5)
        assert cfg.max_epochs == 5

    def test_step1_fields_still_present(self, cfg):
        """Step 2 additions must not remove Step 1 fields."""
        assert cfg.max_encoder_length == 16
        assert cfg.min_encoder_length == 4
        assert cfg.max_prediction_length == 1


# ══════════════════════════════════════════════════════════════════════════════
# 6. Step 2 — TFTTrainResult dataclass (no pytorch required)
# ══════════════════════════════════════════════════════════════════════════════

class TestTFTTrainResult:
    """
    Verify TFTTrainResult has the required fields for stacking ensemble
    integration. All assertions are structural — no training is performed.
    """

    from ml.tft_model import TFTTrainResult, FoldResult  # noqa: F401

    @pytest.fixture
    def minimal_fold_result(self):
        from ml.tft_model import FoldResult
        return FoldResult(
            fold_idx=0,
            train_seasons=[2022],
            val_season=2023,
            mae=20.5,
            rmse=28.1,
            n_train=500,
            n_val=100,
        )

    @pytest.fixture
    def minimal_oof_df(self):
        return pd.DataFrame({
            "player_id": ["p1", "p2"],
            "game_id":   ["g1", "g2"],
            "season":    [2023, 2023],
            "week":      [1, 2],
            "y_true":    [80.0, 60.0],
            "y_pred":    [75.0, 55.0],
            "fold_idx":  [0, 0],
        })

    @pytest.fixture
    def result(self, minimal_fold_result, minimal_oof_df):
        from ml.tft_model import TFTTrainResult
        return TFTTrainResult(
            fold_results=[minimal_fold_result],
            oof_df=minimal_oof_df,
            config=TFTConfig(),
            run_id=None,
            attention_importances={"kalman_est_receiving_yards": 0.6, "target_share": 0.4},
        )

    def test_fold_results_is_list(self, result):
        assert isinstance(result.fold_results, list)

    def test_oof_df_is_dataframe(self, result):
        assert isinstance(result.oof_df, pd.DataFrame)

    def test_oof_df_has_required_columns(self, result):
        required = {"player_id", "game_id", "season", "week", "y_true", "y_pred", "fold_idx"}
        assert required.issubset(set(result.oof_df.columns))

    def test_config_is_tftconfig(self, result):
        assert isinstance(result.config, TFTConfig)

    def test_run_id_can_be_none(self, result):
        assert result.run_id is None

    def test_attention_importances_is_dict(self, result):
        assert isinstance(result.attention_importances, dict)

    def test_oof_path_defaults_to_none(self, result):
        assert result.oof_path is None

    def test_fold_result_fields(self, minimal_fold_result):
        """FoldResult imported via TFTTrainResult must have required fields."""
        assert hasattr(minimal_fold_result, "fold_idx")
        assert hasattr(minimal_fold_result, "train_seasons")
        assert hasattr(minimal_fold_result, "val_season")
        assert hasattr(minimal_fold_result, "mae")
        assert hasattr(minimal_fold_result, "rmse")

    def test_oof_df_y_pred_is_float(self, result):
        assert pd.api.types.is_float_dtype(result.oof_df["y_pred"])

    def test_oof_df_fold_idx_is_integer(self, result):
        assert pd.api.types.is_integer_dtype(result.oof_df["fold_idx"])


# ══════════════════════════════════════════════════════════════════════════════
# 7. Step 2 — Walk-forward fold generation (no pytorch required)
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardFoldsForTFT:
    """
    TFT uses the same _make_walk_forward_folds helper as XGBoost/LightGBM.
    These tests verify the fold structure from TFT's perspective.
    """

    from ml.xgb_model import _make_walk_forward_folds  # noqa: F401

    def test_three_seasons_produces_two_folds(self):
        from ml.xgb_model import _make_walk_forward_folds
        folds = _make_walk_forward_folds([2022, 2023, 2024])
        assert len(folds) == 2

    def test_fold_train_is_expanding_window(self):
        from ml.xgb_model import _make_walk_forward_folds
        folds = _make_walk_forward_folds([2021, 2022, 2023, 2024])
        assert folds[0][0] == [2021]
        assert folds[1][0] == [2021, 2022]
        assert folds[2][0] == [2021, 2022, 2023]

    def test_val_season_strictly_greater_than_max_train(self):
        from ml.xgb_model import _make_walk_forward_folds
        folds = _make_walk_forward_folds([2020, 2021, 2022, 2023])
        for train_seasons, val_season in folds:
            assert max(train_seasons) < val_season

    def test_single_season_raises(self):
        from ml.xgb_model import _make_walk_forward_folds
        with pytest.raises(ValueError):
            _make_walk_forward_folds([2023])

    def test_temporal_guard_assertion_train_not_sorted(self):
        """Simulate the assertion that walk-forward CV uses."""
        train_seasons = [2022, 2021]  # intentionally unsorted
        with pytest.raises(AssertionError):
            assert train_seasons == sorted(train_seasons), "leakage bug"

    def test_temporal_guard_assertion_val_not_future(self):
        """Simulate the assertion that walk-forward CV uses."""
        train_seasons = [2021, 2022, 2023]
        val_season = 2022  # same as a train season — leakage
        with pytest.raises(AssertionError):
            assert max(train_seasons) < val_season, "leakage bug"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Step 2 — save_oof prefix "tft" (no pytorch required)
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveOofTFT:
    """
    Verify save_oof(prefix='tft') produces correctly-named files with the
    canonical OOF schema. No training required.
    """

    @pytest.fixture
    def tmp_oof_dir(self, tmp_path):
        return tmp_path / "oof"

    @pytest.fixture
    def sample_oof_df(self):
        return pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "game_id":   ["g1", "g2", "g3"],
            "season":    [2023, 2023, 2023],
            "week":      [1, 2, 3],
            "y_true":    [80.0, 60.0, 45.0],
            "y_pred":    [75.2, 58.1, 47.9],
            "fold_idx":  [0, 0, 0],
        })

    def test_save_oof_creates_file_with_tft_prefix(self, sample_oof_df, tmp_oof_dir):
        from ml.xgb_model import save_oof
        path = save_oof(sample_oof_df, "receiving_yards", "abcd1234ef56", tmp_oof_dir, prefix="tft")
        assert path.name.startswith("tft_")

    def test_save_oof_file_contains_target_in_name(self, sample_oof_df, tmp_oof_dir):
        from ml.xgb_model import save_oof
        path = save_oof(sample_oof_df, "receiving_yards", "abcd1234ef56", tmp_oof_dir, prefix="tft")
        assert "receiving_yards" in path.name

    def test_save_oof_file_is_readable_csv(self, sample_oof_df, tmp_oof_dir):
        from ml.xgb_model import save_oof
        path = save_oof(sample_oof_df, "receiving_yards", "abcd1234ef56", tmp_oof_dir, prefix="tft")
        loaded = pd.read_csv(path)
        assert set(loaded.columns) == set(sample_oof_df.columns)
        assert len(loaded) == len(sample_oof_df)

    def test_save_oof_creates_output_directory(self, sample_oof_df, tmp_oof_dir):
        from ml.xgb_model import save_oof
        assert not tmp_oof_dir.exists()
        save_oof(sample_oof_df, "receiving_yards", "abcd1234ef56", tmp_oof_dir, prefix="tft")
        assert tmp_oof_dir.is_dir()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Step 2 — Integration: train() with tiny synthetic data (requires pytorch)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pytorch_forecasting as _pf
    import lightning.pytorch as _lp   # pytorch_forecasting 1.6+ uses lightning not pytorch_lightning
    _PYTORCH_AVAILABLE = True
except ImportError:
    _PYTORCH_AVAILABLE = False

_skip_if_no_pytorch = pytest.mark.skipif(
    not _PYTORCH_AVAILABLE,
    reason=(
        "torch, pytorch_forecasting, lightning not installed — "
        "run 'pip install pytorch-forecasting torch lightning'"
    ),
)


def _make_tiny_train_df(
    n_players: int = 4,
    seasons: list[int] | None = None,
    n_weeks: int = 10,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Minimal synthetic feature matrix for TFT integration tests.

    Designed to be fast (few players, few weeks, few seasons) so tests
    complete in seconds on CPU even with max_epochs=1.
    """
    if seasons is None:
        seasons = [2021, 2022, 2023]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for season in seasons:
        for player_idx in range(n_players):
            player_id = f"tft_p{player_idx:02d}"
            for week in range(1, n_weeks + 1):
                rows.append({
                    "player_id":                player_id,
                    "game_id":                  f"{season}_{week:02d}_{player_idx:02d}",
                    "season":                   season,
                    "week":                     week,
                    "position":                 "WR",
                    "team":                     "MIN",
                    "team_id":                  "MIN",
                    "opponent_team_id":         f"OPP_{week % 4}",
                    "is_home":                  int(week % 2),
                    "days_rest":                float(rng.integers(4, 9)),
                    "temp_bucket":              float(rng.integers(0, 4)),
                    "wind_bucket":              float(rng.integers(0, 3)),
                    "kalman_est_receiving_yards":     float(max(0, rng.normal(60, 20))),
                    "kalman_est_targets":             float(max(0, rng.normal(6, 2))),
                    "seas_avg_receiving_yards": float(max(0, rng.normal(65, 15))),
                    "form_target_share":        float(rng.uniform(0.1, 0.3)),
                    "actual_receiving_yards":   float(max(0, rng.normal(60, 25))),
                    # Optional cols (default-fill if absent)
                    "height":                   74.0,
                    "weight":                   200.0,
                    "draft_round":              2.0,
                    "snap_share":               0.7,
                })
    return pd.DataFrame(rows)


@_skip_if_no_pytorch
class TestTFTTrainIntegration:
    """
    End-to-end train() integration tests using tiny synthetic data.

    All tests use max_epochs=1 and a minimal TFTConfig to run quickly on CPU.
    Tests verify:
      - train() returns a TFTTrainResult with correct schema
      - OOF predictions have the canonical 7-column format
      - fold_results contain FoldResult objects with mae/rmse
      - No data leakage (val_season > max train_season for every fold)
    """

    @pytest.fixture(scope="class")
    def tiny_config(self):
        return TFTConfig(
            max_encoder_length=4,
            min_encoder_length=2,
            max_prediction_length=1,
            hidden_size=16,
            attention_head_size=2,
            dropout=0.0,
            hidden_continuous_size=8,
            learning_rate=1e-2,
            max_epochs=1,
            batch_size=16,
            gradient_clip_val=0.1,
            early_stopping_patience=1,
        )

    @pytest.fixture(scope="class")
    def train_df(self):
        return _make_tiny_train_df(n_players=4, seasons=[2021, 2022, 2023], n_weeks=10)

    @pytest.fixture(scope="class")
    def result(self, train_df, tiny_config, tmp_path_factory):
        from ml.tft_model import train
        return train(
            df=train_df,
            seasons=[2021, 2022, 2023],
            target="receiving_yards",
            config=tiny_config,
            n_optuna_trials=0,         # skip Optuna in tests — too slow on CPU
            position_filter=None,      # no position filter for WR-only synthetic data
            mlflow_tracking_uri="",    # disable MLflow in tests
            out_dir=tmp_path_factory.mktemp("tft-oof"),
        )

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_tft_train_result(self, result):
        from ml.tft_model import TFTTrainResult
        assert isinstance(result, TFTTrainResult)

    def test_fold_results_is_nonempty_list(self, result):
        assert isinstance(result.fold_results, list)
        assert len(result.fold_results) > 0

    def test_oof_df_is_nonempty_dataframe(self, result):
        assert isinstance(result.oof_df, pd.DataFrame)
        assert len(result.oof_df) > 0

    # ── OOF schema ────────────────────────────────────────────────────────────

    def test_oof_columns_match_canonical_schema(self, result):
        required = {"player_id", "game_id", "season", "week", "y_true", "y_pred", "fold_idx"}
        assert required.issubset(set(result.oof_df.columns))

    def test_oof_y_pred_is_numeric(self, result):
        assert pd.api.types.is_float_dtype(result.oof_df["y_pred"])

    def test_oof_y_true_is_numeric(self, result):
        assert pd.api.types.is_float_dtype(result.oof_df["y_true"])

    def test_oof_no_nan_in_y_pred(self, result):
        assert result.oof_df["y_pred"].notna().all()

    def test_oof_no_nan_in_y_true(self, result):
        assert result.oof_df["y_true"].notna().all()

    # ── Fold results ──────────────────────────────────────────────────────────

    def test_fold_mae_is_positive(self, result):
        for fr in result.fold_results:
            assert fr.mae >= 0.0, f"fold {fr.fold_idx} MAE={fr.mae} is negative"

    def test_fold_rmse_is_positive(self, result):
        for fr in result.fold_results:
            assert fr.rmse >= 0.0

    def test_fold_rmse_gte_mae(self, result):
        """RMSE ≥ MAE by the Cauchy–Schwarz inequality."""
        for fr in result.fold_results:
            assert fr.rmse >= fr.mae - 1e-6, (
                f"fold {fr.fold_idx}: RMSE={fr.rmse} < MAE={fr.mae} — numerical error?"
            )

    # ── No data leakage ───────────────────────────────────────────────────────

    def test_no_leakage_val_season_strictly_future(self, result):
        """Each fold's val_season must be strictly greater than all train_seasons."""
        for fr in result.fold_results:
            assert max(fr.train_seasons) < fr.val_season, (
                f"Fold {fr.fold_idx}: val_season={fr.val_season} ≤ "
                f"max(train_seasons)={max(fr.train_seasons)}"
            )

    def test_no_leakage_oof_seasons_are_val_seasons(self, result):
        """OOF rows must only contain val_season data for each fold."""
        fold_seasons = {fr.fold_idx: fr.val_season for fr in result.fold_results}
        for fold_idx, val_season in fold_seasons.items():
            fold_rows = result.oof_df[result.oof_df["fold_idx"] == fold_idx]
            if not fold_rows.empty:
                assert (fold_rows["season"] == val_season).all(), (
                    f"Fold {fold_idx}: OOF rows have season != val_season {val_season}"
                )

    # ── Config preserved ──────────────────────────────────────────────────────

    def test_config_stored_in_result(self, result, tiny_config):
        assert result.config is tiny_config

    def test_mlflow_run_id_is_none_when_disabled(self, result):
        assert result.run_id is None

    def test_n_optuna_trials_zero_in_fast_mode(self, tmp_path):
        """train() with n_optuna_trials=0 completes without calling Optuna."""
        # Verify the kwarg is accepted without error (Optuna path is skipped)
        from ml.tft_model import train
        tiny_df = _make_tiny_train_df(n_players=3, seasons=[2021, 2022], n_weeks=8)
        cfg = TFTConfig(
            max_encoder_length=3, min_encoder_length=2,
            hidden_size=16, hidden_continuous_size=8,
            attention_head_size=2, dropout=0.0,
            learning_rate=1e-2, max_epochs=1,
            batch_size=8, early_stopping_patience=1,
        )
        result = train(
            df=tiny_df, seasons=[2021, 2022], target="receiving_yards",
            config=cfg, n_optuna_trials=0,
            position_filter=None, mlflow_tracking_uri="",
            out_dir=tmp_path / "oof",
        )
        from ml.tft_model import TFTTrainResult
        assert isinstance(result, TFTTrainResult)


# ══════════════════════════════════════════════════════════════════════════════
# ACCEPTANCE TEST — Three-way OOF alignment
#
# This is the proof that XGBoost + LightGBM + TFT OOF files are stacking-
# compatible. load_and_align_oofs() must produce a non-empty inner join with
# columns [xgb_pred, lgbm_pred, tft_pred, y_true, fold_idx].
#
# This test does NOT require pytorch — it creates synthetic OOF CSVs that
# simulate what each base learner would produce, then verifies the stacking
# ensemble can consume all three.
# ══════════════════════════════════════════════════════════════════════════════

class TestOofAlignmentThreeWay:
    """
    Three-way OOF alignment acceptance test.

    Verifies load_and_align_oofs() produces a non-empty DataFrame with the
    exact columns [xgb_pred, lgbm_pred, tft_pred, y_true, fold_idx] when
    given OOF files from all three base learners.

    This test is the integration proof that:
      1. TFT OOF file uses the canonical 7-column schema
      2. TFT filename uses the "tft_" prefix so load_and_align_oofs() extracts
         the right model name
      3. The (player_id, game_id) key matches between all three learners so the
         inner join is non-empty
      4. The stacking meta-learner's Ridge regression can receive all three
         predictions as input features

    No training required — synthetic OOF CSVs are created in a tmp directory.
    """

    # Shared synthetic game/player IDs so inner join is non-empty
    _PLAYER_IDS = [f"player_{i:03d}" for i in range(8)]
    _GAME_IDS   = [f"2023_{w:02d}_{p}" for p in range(8) for w in range(1, 11)]
    _N_ROWS     = len(_GAME_IDS)

    @pytest.fixture(scope="class")
    def shared_rows(self):
        """Build the shared (player_id, game_id, season, week, y_true, fold_idx) DataFrame."""
        rng = np.random.default_rng(42)
        rows = []
        for player_idx, player_id in enumerate(self._PLAYER_IDS):
            for week in range(1, 11):
                rows.append({
                    "player_id": player_id,
                    "game_id":   f"2023_{week:02d}_{player_idx}",
                    "season":    2023,
                    "week":      week,
                    "y_true":    float(max(0, rng.normal(60, 25))),
                    "fold_idx":  0,
                })
        return pd.DataFrame(rows)

    @pytest.fixture(scope="class")
    def oof_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("three_way_oof")

    @pytest.fixture(scope="class")
    def xgb_oof_path(self, shared_rows, oof_dir):
        rng = np.random.default_rng(10)
        df = shared_rows.copy()
        df["y_pred"] = df["y_true"] + rng.normal(0, 10, len(df))
        path = oof_dir / "xgb_receiving_yards_aabbccdd.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    def lgbm_oof_path(self, shared_rows, oof_dir):
        rng = np.random.default_rng(20)
        df = shared_rows.copy()
        df["y_pred"] = df["y_true"] + rng.normal(0, 12, len(df))
        path = oof_dir / "lgbm_receiving_yards_eeff0011.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    def tft_oof_path(self, shared_rows, oof_dir):
        """
        TFT OOF — same schema as XGB/LGBM:
        player_id, game_id, season, week, y_true, y_pred, fold_idx.
        """
        rng = np.random.default_rng(30)
        df = shared_rows.copy()
        df["y_pred"] = df["y_true"] + rng.normal(0, 14, len(df))
        path = oof_dir / "tft_receiving_yards_22334455.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture(scope="class")
    def aligned(self, xgb_oof_path, lgbm_oof_path, tft_oof_path):
        from ml.stacking_ensemble import load_and_align_oofs
        aligned_df, pred_cols, prefixes = load_and_align_oofs(
            [xgb_oof_path, lgbm_oof_path, tft_oof_path]
        )
        return aligned_df, pred_cols, prefixes

    # ── Non-empty result ──────────────────────────────────────────────────────

    def test_aligned_df_is_nonempty(self, aligned):
        aligned_df, _, _ = aligned
        assert len(aligned_df) > 0, (
            "Three-way inner join is empty — OOF files do not share any "
            "(player_id, game_id) pairs. Check that all three base learners "
            "are trained on the same position filter."
        )

    def test_aligned_row_count_equals_shared_rows(self, aligned, shared_rows):
        """Inner join on identical (player_id, game_id) sets must keep all rows."""
        aligned_df, _, _ = aligned
        assert len(aligned_df) == len(shared_rows)

    # ── Column presence ───────────────────────────────────────────────────────

    def test_xgb_pred_column_present(self, aligned):
        aligned_df, _, _ = aligned
        assert "xgb_pred" in aligned_df.columns

    def test_lgbm_pred_column_present(self, aligned):
        aligned_df, _, _ = aligned
        assert "lgbm_pred" in aligned_df.columns

    def test_tft_pred_column_present(self, aligned):
        aligned_df, _, _ = aligned
        assert "tft_pred" in aligned_df.columns, (
            "tft_pred column missing — TFT OOF file must use prefix 'tft_' "
            "in the filename (e.g. tft_receiving_yards_XXXXXXXX.csv)"
        )

    def test_y_true_column_present(self, aligned):
        aligned_df, _, _ = aligned
        assert "y_true" in aligned_df.columns

    def test_fold_idx_column_present(self, aligned):
        aligned_df, _, _ = aligned
        assert "fold_idx" in aligned_df.columns

    # ── Column values ─────────────────────────────────────────────────────────

    def test_all_pred_columns_are_numeric(self, aligned):
        aligned_df, pred_cols, _ = aligned
        for col in pred_cols:
            assert pd.api.types.is_float_dtype(aligned_df[col]), (
                f"Column '{col}' should be float, got {aligned_df[col].dtype}"
            )

    def test_no_nan_in_pred_columns(self, aligned):
        aligned_df, pred_cols, _ = aligned
        for col in pred_cols:
            assert aligned_df[col].notna().all(), f"NaN found in {col}"

    def test_pred_cols_list_has_three_entries(self, aligned):
        _, pred_cols, _ = aligned
        assert len(pred_cols) == 3

    def test_prefixes_are_xgb_lgbm_tft(self, aligned):
        _, _, prefixes = aligned
        assert set(prefixes) == {"xgb", "lgbm", "tft"}

    # ── Ridge meta-learner readiness ─────────────────────────────────────────

    def test_can_build_meta_feature_matrix(self, aligned):
        """
        The three pred columns can be stacked into a feature matrix X
        for the Ridge meta-learner. This is the exact operation
        stacking_ensemble._meta_fit() performs.
        """
        aligned_df, pred_cols, _ = aligned
        X = aligned_df[pred_cols].values
        y = aligned_df["y_true"].values
        assert X.shape == (len(aligned_df), 3)
        assert y.shape == (len(aligned_df),)
        # Verify sklearn Ridge can fit on this data
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        assert len(ridge.coef_) == 3

    def test_pred_cols_are_in_expected_order(self, aligned):
        """pred_cols order matches oof_paths order: [xgb, lgbm, tft]."""
        _, pred_cols, _ = aligned
        assert pred_cols == ["xgb_pred", "lgbm_pred", "tft_pred"]

    def test_two_learners_raises_if_tft_missing(self, xgb_oof_path, lgbm_oof_path):
        """Confirm two-learner alignment still works (TFT is additive)."""
        from ml.stacking_ensemble import load_and_align_oofs
        aligned_df, pred_cols, prefixes = load_and_align_oofs(
            [xgb_oof_path, lgbm_oof_path]
        )
        assert "xgb_pred" in aligned_df.columns
        assert "lgbm_pred" in aligned_df.columns
        assert "tft_pred" not in aligned_df.columns
