"""
ml/tft_model.py

Temporal Fusion Transformer (TFT) — third base learner in the stacking ensemble.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY TFT AS THE THIRD BASE LEARNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost and LightGBM treat each (player, game) row independently — they have
no explicit representation of the time axis. TFT is the statistically correct
choice for the third learner because it explicitly models:
  - Sequential game-by-game form trajectories (encoder RNN + attention)
  - Static player identity (position, height, weight) via separate embedding
  - Known future context (opponent, venue, weather) via skip connections
  - Variable selection networks that learn which covariates matter per player

This means TFT's errors are structurally different from XGB/LGBM's — it will
do better on players with consistent trajectory patterns and worse on
one-season wonders. This diversity is exactly what makes stacking valuable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COVARIATE GROUPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TFT partitions covariates into three groups with different architectures:

  STATIC — don't change week-to-week, processed by static embedding:
    Categoricals: position                               (1 column)
    Reals:        height, weight, draft_round            (3 columns)

  TIME-VARYING KNOWN — known at prediction time, fed into both encoder & decoder:
    Categoricals: team_id, opponent_team_id, home_away   (3 columns)
    NOTE: team_id is time-varying (not static) because players switch teams.
    This lets TFT model roster context and scheme changes when a player moves.
    Reals:        week, rest_days, temp_bucket,
                  wind_bucket                            (4 columns)

  TIME-VARYING UNKNOWN — only available up to prediction date, encoder only:
    Reals:        kalman_est for receiving/rushing/passing/usage,
                  seas_avg for yardage stats, snap_share,
                  target_share                           (14 columns)

Target: actual_receiving_yards  (y_true; also passed as time-varying unknown)
Group ID: player_id (each player = one independent time series)
Time index: (season - min_season) × 22 + (week − 1)  [globally monotonic int]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN DERIVATION (prepare_dataframe)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The feature matrix (produced by pipeline/feature_engineer.py) uses slightly
different column names from what TFT expects. prepare_dataframe() handles all
derivations purely in pandas before handing off to TimeSeriesDataSet:

  is_home (0/1 int)           → home_away ("home"/"away" str)  [categorical]
  days_rest (int)             → rest_days (float)
  kalman_est_target_share     → target_share (float)
  team (str)                  → team_id (str)                  [if team_id absent]

  height, weight, draft_round → 0.0  (if not joined from roster table yet)
  snap_share              → 0.0       (requires PBP data; Phase 4 enhancement)
  opponent_team_id        → "UNKNOWN" (if matchup join not performed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Data formatting (TFTDataset)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - TFTConfig dataclass
  - TFTDataset.prepare_dataframe() — pure pandas, no pytorch
  - TFTDataset.make_dataset()      — returns TimeSeriesDataSet (pytorch needed)
  - TFTDataset._make_dataset_raw() — builds TimeSeriesDataSet from pre-prepared df
                                     (used internally by walk-forward CV to avoid
                                      recomputing global time_idx per fold)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Training Split
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-forward CV, fold execution, Optuna tuning, and attention extraction now
live in `ml.tft_cv`. Higher-level orchestration stays in `ml.tft_training`,
MLflow/OOF persistence lives in `ml.tft_artifacts`, and CLI parsing/execution
lives in `ml.tft_cli`.

`ml.tft_model` remains the runtime/inference-facing half:
  - `TFTConfig`
  - `TFTDataset`
  - compatibility exports for `TFTTrainResult`, `train()`, and `main()`

This keeps existing imports stable while separating the lightweight pandas /
inference surface from the heavier training orchestration code.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

# ── Thread-safety: must happen BEFORE torch / pytorch_forecasting import ───────
# TFT requires PyTorch which ships its own libomp. When XGBoost + LightGBM are
# also loaded in the same process, three OMP runtimes conflict → SIGSEGV.
# configure_thread_env() sets KMP_DUPLICATE_LIB_OK + OMP_NUM_THREADS=1.
from ml.utils import configure_thread_env  # noqa: E402 — must precede torch import
configure_thread_env()

# Cap PyTorch thread pool at module import time so both inference and the
# separate training module inherit the same conservative runtime defaults.
try:
    import torch as _torch
    _torch.set_num_threads(1)
except ImportError:
    pass  # torch not installed — TFT will raise its own ImportError at dataset build

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


# Shared utilities — identical across all base learners.
# FoldResult, _make_walk_forward_folds, _data_hash, save_oof, _compute_metrics,
# and _parse_seasons are defined once in xgb_model and imported here so TFT
# produces OOF files in the same schema as XGBoost and LightGBM.
from ml.utils import TARGET_COL_MAP, FoldResult
from ml.feature_contract import assert_model_input_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TFTConfig:
    """
    Hyperparameters for TFT data formatting and model architecture.

    ── Step 1: data formatting ────────────────────────────────────────────────
    max_encoder_length: how many past weeks the TFT encoder sees.
      16 ≈ one full NFL regular season (17 weeks + a few for ramp-up).
    min_encoder_length: shortest history still included in a batch.
      4 ensures early-season predictions are not dropped entirely.
    max_prediction_length: always 1 — we predict one week at a time.

    ── Step 2: model architecture ────────────────────────────────────────────
    hidden_size: dimensionality of all internal TFT representations.
      64 is conservative; increase to 128+ for more complex patterns at cost
      of training time. Constrained by NFL dataset size (~20k rows).
    attention_head_size: number of multi-head attention heads.
      4 heads with hidden_size=64 → 16-dimensional per-head attention.
    dropout: applied at variable selection networks and within each LSTM.
      0.1 is conservative — increase if overfitting is observed.
    hidden_continuous_size: embedding size for continuous inputs before
      variable selection. Smaller than hidden_size to reduce parameters.
      32 (half of hidden_size=64) is the standard TFT paper recommendation.
    learning_rate: Adam initial LR. 3e-3 is typical for TFT on tabular data.
    max_epochs: hard cap on training epochs. Early stopping fires first.
    batch_size: number of windows per gradient step. 64 fits comfortably
      in CPU/MPS memory for sequences of length ≤ max_encoder_length.
    gradient_clip_val: gradient norm clipping. 0.1 stabilises TFT training
      which can otherwise exhibit large gradient spikes in early epochs.
    early_stopping_patience: stop if val_loss doesn't improve for this many
      epochs. 5 is sufficient given NFL data has low epoch-to-epoch variance.
    """
    # Step 1 — data formatting
    max_encoder_length:  int  = 16
    min_encoder_length:  int  = 4
    max_prediction_length: int = 1
    target: str = "receiving_yards"  # key in TARGET_COL_MAP

    # Step 2 — model architecture
    hidden_size:            int   = 64
    attention_head_size:    int   = 4
    dropout:                float = 0.1
    hidden_continuous_size: int   = 32   # half of hidden_size per TFT paper

    # Step 2 — training
    learning_rate:             float = 3e-3
    max_epochs:                int   = 30
    batch_size:                int   = 64
    gradient_clip_val:         float = 0.1
    early_stopping_patience:   int   = 5


# ── TFTDataset ────────────────────────────────────────────────────────────────

class TFTDataset:
    """
    Converts an NFL FeatureMatrix DataFrame into a pytorch_forecasting
    TimeSeriesDataSet — the input format required by TFT.

    The three covariate groups are declared as CLASS-LEVEL CONSTANTS so they
    can be inspected and tested without importing pytorch_forecasting.

    Usage (Step 1 — no pytorch needed):
        ds = TFTDataset()
        prepared_df = ds.prepare_dataframe(feature_matrix_df)

    Usage (Step 2 — requires pytorch_forecasting):
        dataset = ds.make_dataset(feature_matrix_df)
    """

    # ── Covariate group definitions ──────────────────────────────────────────
    # These are the canonical column names TFT expects after prepare_dataframe().
    # STATIC: player identity — doesn't change week-to-week
    STATIC_CATEGORICALS: list[str] = ["position"]
    STATIC_REALS:        list[str] = ["height", "weight", "draft_round"]

    # TIME-VARYING KNOWN: available at prediction time (future context)
    TIME_VARYING_KNOWN_CATEGORICALS: list[str] = ["team_id", "opponent_team_id", "home_away"]
    TIME_VARYING_KNOWN_REALS:        list[str] = ["week", "rest_days"]

    # TIME-VARYING UNKNOWN: only available up to current week (historical form).
    # Covers all stat families so TFT can learn from the relevant Kalman
    # estimates regardless of which target is being predicted.
    # The target column is appended to this group at dataset-build time
    # (TimeSeriesDataSet requires the target in time_varying_unknown_reals).
    TIME_VARYING_UNKNOWN_REALS: list[str] = [
        "kalman_est_receiving_yards", "kalman_est_targets", "kalman_est_receptions",
        "kalman_est_receiving_tds", "kalman_est_fantasy_ppr",
        "kalman_variance_receiving_yards", "kalman_variance_targets",
        "kalman_variance_receptions", "kalman_variance_fantasy_ppr",
        "seas_avg_receiving_yards", "seas_avg_targets", "seas_avg_receptions",
        "snap_share", "target_share",
    ]

    GROUP_ID:         str = "player_id"
    TIME_IDX:         str = "time_idx"
    WEEKS_PER_SEASON: int = 22  # 18 regular season + 4 playoff rounds (max)

    def __init__(self, config: Optional[TFTConfig] = None) -> None:
        self.config = config or TFTConfig()

    # ── Covariate count summary ───────────────────────────────────────────────

    @classmethod
    def covariate_counts(cls) -> dict[str, int]:
        """
        Return the column count of each covariate group.

        Useful for verifying the dataset definition without instantiating
        pytorch_forecasting objects.

        Returns:
            dict with keys:
              static_categoricals, static_reals,
              time_varying_known_categoricals, time_varying_known_reals,
              time_varying_unknown_reals
        """
        return {
            "static_categoricals":             len(cls.STATIC_CATEGORICALS),
            "static_reals":                    len(cls.STATIC_REALS),
            "time_varying_known_categoricals": len(cls.TIME_VARYING_KNOWN_CATEGORICALS),
            "time_varying_known_reals":        len(cls.TIME_VARYING_KNOWN_REALS),
            "time_varying_unknown_reals":      len(cls.TIME_VARYING_UNKNOWN_REALS),
        }

    # ── Step 1 helpers (pure pandas, no pytorch dependency) ───────────────────

    @classmethod
    def _add_time_idx(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a global monotonically increasing integer time index per player.

        Formula: (season − min_season) × WEEKS_PER_SEASON + (week − 1)

        This produces a gapless-ish global time axis across multiple seasons.
        Gaps within a season (bye weeks, injury absences) are handled by
        allow_missing_timesteps=True in TimeSeriesDataSet.

        Examples for min_season = 2018:
          Season 2018, Week 1  → time_idx = 0
          Season 2018, Week 22 → time_idx = 21
          Season 2019, Week 1  → time_idx = 22
          Season 2024, Week 17 → time_idx = 6 × 22 + 16 = 148
        """
        min_season = int(df["season"].min())
        df = df.copy()
        df[cls.TIME_IDX] = (
            (df["season"] - min_season) * cls.WEEKS_PER_SEASON + (df["week"] - 1)
        ).astype(int)
        return df

    @classmethod
    def _derive_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive / rename columns to match the TFT covariate group definitions.

        All transformations are purely pandas — no pytorch required.

        Derivations:
          is_home (0/1 int)       → home_away ("home"/"away")  [TFT categorical]
          days_rest (int)         → rest_days (float)
          kalman_est_target_share (float) → target_share (float)
          team (str)              → team_id (str)  [if team_id absent]

        Default-fill for columns not yet in the feature matrix:
          height, weight, draft_round  → 0.0   (joined in from roster in Phase 3)
          snap_share                   → 0.0   (PBP data; Phase 4 enhancement)
          team_id, opponent_team_id    → "UNKNOWN"
        """
        df = df.copy()

        # home_away — categorical string expected by TFT
        if "home_away" not in df.columns:
            if "is_home" in df.columns:
                df["home_away"] = (
                    df["is_home"]
                    .map({1: "home", 0: "away"})
                    .fillna("unknown")
                    .astype(str)
                )
            else:
                df["home_away"] = "unknown"

        # rest_days — rename from days_rest (feature_engineer output name)
        if "rest_days" not in df.columns:
            if "days_rest" in df.columns:
                df["rest_days"] = df["days_rest"].astype(float)
            else:
                df["rest_days"] = 7.0  # default: one-week rest

        # target_share — copy from kalman_est_target_share if not present
        if "target_share" not in df.columns:
            if "kalman_est_target_share" in df.columns:
                df["target_share"] = df["kalman_est_target_share"].astype(float)
            else:
                df["target_share"] = 0.0

        # team_id — derive from team column if not already present
        if "team_id" not in df.columns:
            if "team" in df.columns:
                df["team_id"] = df["team"].astype(str)
            else:
                df["team_id"] = "UNKNOWN"

        # ── Physical profile (Item 0 fix) ────────────────────────────────────
        # height, weight, draft_round are now stored in feature_matrix (pipeline
        # Item 0). Passthrough if present; default to 0.0 only as final fallback.
        for col in ("height", "weight", "draft_round"):
            if col not in df.columns:
                logger.info(
                    "Column '%s' absent from feature_matrix — defaulting to 0.0. "
                    "Re-run pipeline/feature_engineer.py to populate.",
                    col,
                )
                df[col] = 0.0
            else:
                # Median-fill any remaining NaN (undrafted players have NaN draft_round)
                median = df[col].median()
                df[col] = df[col].fillna(median if pd.notna(median) else 0.0)

        # ── Snap share ────────────────────────────────────────────────────────
        # TFT may use only the registered prior-game snap feature.  A missing
        # input is a contract failure, not a reason to turn a removed leak into
        # a constant zero column.
        if "prior_snap_share" not in df.columns:
            raise AssertionError("TFT requires approved prior_snap_share; refusing raw/zero snap fallback")
        df["snap_share"] = df["prior_snap_share"]
        logger.debug("snap_share ← prior_snap_share (lagged participation).")

        assert_model_input_columns(
            [*cls.STATIC_REALS, *cls.TIME_VARYING_KNOWN_REALS, *cls.TIME_VARYING_UNKNOWN_REALS],
            consumer="TFTDataset",
        )

        # ── Opponent team (Item 0 fix) ────────────────────────────────────────
        # feature_matrix has opponent_team column (TEXT).
        # Map to opponent_team_id (the categorical TFT expects).
        if "opponent_team_id" not in df.columns:
            if "opponent_team" in df.columns:
                df["opponent_team_id"] = df["opponent_team"].fillna("UNKNOWN").astype(str)
                logger.debug("opponent_team_id ← opponent_team (mapped from feature_matrix).")
            else:
                df["opponent_team_id"] = "UNKNOWN"

        # Cast all categorical columns to str (TimeSeriesDataSet requirement).
        # fillna BEFORE astype(str) — astype(str) turns NaN into "nan" string,
        # after which fillna is a no-op.
        for col in (*cls.STATIC_CATEGORICALS, *cls.TIME_VARYING_KNOWN_CATEGORICALS):
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN").astype(str)

        return df

    @staticmethod
    def _fill_missing_reals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Fill NaN in real-valued columns with per-column median (fallback: 0)."""
        df = df.copy()
        for col in cols:
            if col in df.columns and df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median if pd.notna(median) else 0.0)
        return df

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        target: str = "receiving_yards",
    ) -> pd.DataFrame:
        """
        Transform a raw FeatureMatrix DataFrame into the shape expected by
        make_dataset().

        This is a pure-pandas operation — no pytorch_forecasting import needed.
        Call this directly to inspect the transformed data before building the
        TimeSeriesDataSet.

        Steps:
          1. Validate target column exists.
          2. Add time_idx (global monotonic integer per player across seasons).
          3. Derive / rename covariate columns.
          4. Fill NaN in real columns with column median.
          5. Drop rows where target is still NaN.
          6. Sort by (player_id, time_idx).

        Args:
            df:     Feature matrix DataFrame. Required columns: player_id,
                    season, week, and the target column (e.g. actual_receiving_yards).
            target: Target key from TARGET_COL_MAP.

        Returns:
            Prepared DataFrame ready for make_dataset().

        Raises:
            ValueError: if target is unknown or target column is missing from df.
        """
        if target not in TARGET_COL_MAP:
            raise ValueError(
                f"Unknown target '{target}'. Valid targets: {list(TARGET_COL_MAP)}"
            )
        target_col = TARGET_COL_MAP[target]

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame. "
                f"Columns present: {sorted(df.columns.tolist())}"
            )

        df = self._add_time_idx(df)
        df = self._derive_columns(df)

        all_reals = (
            self.STATIC_REALS
            + self.TIME_VARYING_KNOWN_REALS
            + self.TIME_VARYING_UNKNOWN_REALS
            + [target_col]
        )
        df = self._fill_missing_reals(df, all_reals)

        # Drop rows where target is still NaN (can't train or evaluate on them)
        before = len(df)
        df = df[df[target_col].notna()].copy()
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN target '%s'.", dropped, target_col)

        df = df.sort_values([self.GROUP_ID, self.TIME_IDX]).reset_index(drop=True)

        logger.info(
            "prepare_dataframe: %d rows | %d players | seasons %s–%s | target=%s",
            len(df),
            df[self.GROUP_ID].nunique(),
            int(df["season"].min()),
            int(df["season"].max()),
            target_col,
        )
        return df

    def make_dataset(
        self,
        df: pd.DataFrame,
        target: str = "receiving_yards",
    ) -> Any:
        """
        Build a pytorch_forecasting TimeSeriesDataSet from the feature matrix.

        Calls prepare_dataframe() internally — pass a raw feature matrix directly.

        Args:
            df:     Raw or pre-prepared feature matrix DataFrame.
            target: Target key from TARGET_COL_MAP.

        Returns:
            pytorch_forecasting.TimeSeriesDataSet

        Raises:
            ImportError: if pytorch_forecasting is not installed.
            ValueError:  if target is unknown or target column is missing.
        """
        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.data.encoders import NaNLabelEncoder
        except ImportError as exc:
            raise ImportError(
                "pytorch_forecasting is required for make_dataset(). "
                "Install it with: pip install pytorch-forecasting torch"
            ) from exc

        target_col = TARGET_COL_MAP[target]
        df = self.prepare_dataframe(df, target=target)

        # Restrict covariate lists to columns actually present in df
        # (handles cases where some data enrichments haven't been joined yet)
        static_cats   = [c for c in self.STATIC_CATEGORICALS             if c in df.columns]
        static_reals  = [c for c in self.STATIC_REALS                    if c in df.columns]
        known_cats    = [c for c in self.TIME_VARYING_KNOWN_CATEGORICALS  if c in df.columns]
        known_reals   = [c for c in self.TIME_VARYING_KNOWN_REALS         if c in df.columns]
        # Target must be included in time_varying_unknown_reals per pytorch_forecasting API
        unknown_reals = [c for c in self.TIME_VARYING_UNKNOWN_REALS       if c in df.columns]

        logger.info(
            "Building TimeSeriesDataSet: "
            "static_cats=%d  static_reals=%d  known_cats=%d  known_reals=%d  "
            "unknown_reals=%d (+ target)  encoder=%d  pred=%d",
            len(static_cats), len(static_reals), len(known_cats), len(known_reals),
            len(unknown_reals),
            self.config.max_encoder_length,
            self.config.max_prediction_length,
        )

        dataset = TimeSeriesDataSet(
            df,
            time_idx=self.TIME_IDX,
            target=target_col,
            group_ids=[self.GROUP_ID],
            max_encoder_length=self.config.max_encoder_length,
            min_encoder_length=self.config.min_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            static_categoricals=static_cats,
            static_reals=static_reals,
            time_varying_known_categoricals=known_cats,
            time_varying_known_reals=known_reals,
            # Target appended here — required by TimeSeriesDataSet API
            time_varying_unknown_reals=unknown_reals + [target_col],
            target_normalizer=GroupNormalizer(groups=[self.GROUP_ID]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            # allow_missing_timesteps: players have bye weeks and injury absences
            allow_missing_timesteps=True,
            categorical_encoders={c: NaNLabelEncoder(add_nan=True) 
                                  for c in static_cats + known_cats + [self.GROUP_ID]},
        )
        return dataset

    def _make_dataset_raw(
        self,
        prepared_df: pd.DataFrame,
        target: str,
    ) -> Any:
        """
        Build a TimeSeriesDataSet directly from a **pre-prepared** DataFrame.

        Unlike make_dataset(), this method does NOT call prepare_dataframe()
        first. It is used internally by the walk-forward CV loop so that
        time_idx values computed on the FULL dataset are preserved across folds.

        ── Why this matters ──────────────────────────────────────────────────
        If prepare_dataframe() were called on a per-fold subset, _add_time_idx
        would recompute min_season from that subset. For fold
        (train=2018-2022, val=2023) the train subset has min_season=2018, so
        time_idx is consistent. But if we ever split differently, or if the
        val-only subset had a different min_season, the val time_idx values
        would be shifted relative to the train values — causing the
        TimeSeriesDataSet.from_dataset() join to fail silently.

        By computing time_idx once on the full dataset and passing pre-prepared
        subsets here, all folds share a globally consistent time axis.

        Args:
            prepared_df: Output of prepare_dataframe() — must already contain
                         time_idx, home_away, rest_days, and all covariate cols.
            target:      Target key from TARGET_COL_MAP.

        Returns:
            pytorch_forecasting.TimeSeriesDataSet

        Raises:
            ImportError: if pytorch_forecasting is not installed.
            ValueError:  if target is unknown.
        """
        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.data.encoders import NaNLabelEncoder
        except ImportError as exc:
            raise ImportError(
                "pytorch_forecasting is required for _make_dataset_raw(). "
                "Install it with: pip install pytorch-forecasting torch"
            ) from exc

        if target not in TARGET_COL_MAP:
            raise ValueError(
                f"Unknown target '{target}'. Valid: {list(TARGET_COL_MAP)}"
            )
        target_col = TARGET_COL_MAP[target]

        static_cats   = [c for c in self.STATIC_CATEGORICALS             if c in prepared_df.columns]
        static_reals  = [c for c in self.STATIC_REALS                    if c in prepared_df.columns]
        known_cats    = [c for c in self.TIME_VARYING_KNOWN_CATEGORICALS  if c in prepared_df.columns]
        known_reals   = [c for c in self.TIME_VARYING_KNOWN_REALS         if c in prepared_df.columns]
        unknown_reals = [c for c in self.TIME_VARYING_UNKNOWN_REALS       if c in prepared_df.columns]

        return TimeSeriesDataSet(
            prepared_df,
            time_idx=self.TIME_IDX,
            target=target_col,
            group_ids=[self.GROUP_ID],
            max_encoder_length=self.config.max_encoder_length,
            min_encoder_length=self.config.min_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            static_categoricals=static_cats,
            static_reals=static_reals,
            time_varying_known_categoricals=known_cats,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals + [target_col],
            target_normalizer=GroupNormalizer(groups=[self.GROUP_ID]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            categorical_encoders={c: NaNLabelEncoder(add_nan=True) 
                                  for c in static_cats + known_cats + [self.GROUP_ID]},
        )

__all__ = [
    "FoldResult",
    "TFTConfig",
    "TFTDataset",
    "main",
    "train",
]


def train(*args: Any, **kwargs: Any) -> Any:
    from ml.tft_training import train as _train

    return _train(*args, **kwargs)


def main() -> None:
    from ml.tft_training import main as _main

    _main()


def __getattr__(name: str) -> Any:
    if name in {
        "TFTTrainResult",
        "_extract_attention_importances",
        "_run_optuna_tft",
        "_train_fold_tft",
        "_walk_forward_cv_tft",
    }:
        import ml.tft_training as _tft_training

        return getattr(_tft_training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
