"""
Training orchestration + CLI half of the TFT stack.

`ml.tft_model` remains the runtime/inference-facing module that exposes
`TFTConfig` and `TFTDataset`. Fold execution, walk-forward CV, Optuna tuning,
and attention extraction live in `ml.tft_cv`. MLflow/OOF persistence lives in
`ml.tft_artifacts`, and CLI argument parsing lives in `ml.tft_cli`. This module
owns the higher-level programmatic orchestration that ties those pieces together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.tft_artifacts import persist_training_artifacts
from ml.cohort_guards import validate_training_cohort
from ml.tft_cv import (
    _compute_final_attention_importances,
    _run_optuna_tft,
    _walk_forward_cv_tft,
)
from ml.tft_model import TFTConfig, TFTDataset, logger
from ml.utils import (
    TARGET_COL_MAP,
    FoldResult,
    _data_hash,
    _make_walk_forward_folds,
)


@dataclass
class TFTTrainResult:
    """
    Full training result returned by train().

    Mirrors XGBTrainResult / LGBMTrainResult so the stacking ensemble
    can call all three base learners with the same interface.

    attention_importances: normalised variable-selection attention weights
      averaged across the final fold's validation set. Replaces the
      feature_importances from tree models — the TFT equivalent of asking
      "which input variables did the variable-selection network weight most?"
    """

    fold_results: list[FoldResult]
    oof_df: pd.DataFrame
    config: TFTConfig
    run_id: Optional[str]
    attention_importances: dict[str, float]
    oof_path: Optional[Path] = None

def train(
    df: pd.DataFrame,
    seasons: list[int],
    target: str = "receiving_yards",
    config: Optional[TFTConfig] = None,
    n_optuna_trials: int = 20,
    position_filter: Optional[str] = "WR",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    out_dir: Optional[Path] = None,
    checkpoint_path: Optional[str] = None,
) -> TFTTrainResult:
    """
    Train TFT with walk-forward CV, optional Optuna tuning, and MLflow logging.

    API mirrors ml.xgb_model.train() and ml.lgbm_model.train() so the
    stacking ensemble can call all three with the same interface.

    Optuna search (20 trials by default):
      Runs on the MOST RECENT fold only (same as XGB/LGBM). Searches:
        hidden_size:    categorical [32, 64, 128]
        dropout:        categorical [0.05, 0.1, 0.2]
        learning_rate:  log-uniform [1e-4, 1e-2]
      Each trial trains for _OPTUNA_EPOCHS=3 epochs. Best params are applied
      to all folds for the full walk-forward CV run.
      Pass n_optuna_trials=0 to skip Optuna and use base config values.

    Walk-forward CV note:
      prepare_dataframe() is called ONCE on the full df so that time_idx is
      globally consistent. Then per-fold datasets are built via
      _make_dataset_raw() (skips the re-prepare step). See Step 2 docstring
      for the full explanation.

    Args:
        df:                  Feature matrix DataFrame. Must contain player_id,
                             game_id, season, week, and all TFT covariate cols.
        seasons:             All seasons present in df (used to build folds).
        target:              One of: "receiving_yards", "rushing_yards",
                             "passing_yards", "fantasy_ppr".
        config:              TFTConfig. Defaults to TFTConfig() if None.
        n_optuna_trials:     Optuna trial count. Pass 0 to skip (use config values).
        position_filter:     If given, filter df to this position before training.
        mlflow_tracking_uri: Override MLFLOW_TRACKING_URI env var. Pass "" to
                             disable MLflow entirely.
        mlflow_experiment:   MLflow experiment name. Defaults to "tft_{target}".
        out_dir:             Directory for OOF CSV output. Defaults to ml/oof/.

    Returns:
        TFTTrainResult with fold_results, oof_df, config, run_id, importances.

    Raises:
        ValueError:    if target is unknown or fewer than 2 seasons provided.
        AssertionError if temporal ordering is violated in any fold.
        ImportError:   if pytorch_forecasting / pytorch_lightning not installed.
    """
    if config is None:
        config = TFTConfig(target=target)

    if target not in TARGET_COL_MAP:
        raise ValueError(
            f"Unknown target '{target}'. Valid: {list(TARGET_COL_MAP)}"
        )
    if target == "passing_yards":
        raise ValueError(
            "TFT for passing_yards is disabled pending validation. "
            "Historical artifacts for this target collapsed near zero and must not be retrained or served."
        )
    target_col = TARGET_COL_MAP[target]

    if out_dir is None:
        out_dir = Path(__file__).parent / "oof"

    if position_filter and position_filter.lower() != "all":
        df = df[df["position"] == position_filter].copy()
        logger.info("Position filter '%s': %d rows remain", position_filter, len(df))

    df = df[df[target_col].notna()].copy()
    validate_training_cohort(
        df,
        target=target,
        target_col=target_col,
        position_filter=(
            None if not position_filter or position_filter.lower() == "all"
            else position_filter
        ),
        source="tft_model",
    )

    logger.info(
        "Training data: %d rows | target=%s",
        len(df), target_col,
    )

    seasons_in_data = sorted(df["season"].unique().tolist())
    if len(seasons_in_data) < 2:
        logger.error(
            "Target '%s' lacks sufficient historical data for walk-forward CV. "
            "Found %d seasons (%s). Minimum 2 required. Skipping.",
            target, len(seasons_in_data), seasons_in_data
        )
        return TFTTrainResult(
            fold_results=[],
            oof_df=pd.DataFrame(),
            config=config,
            run_id=None,
            attention_importances={},
            oof_path=None,
        )

    folds = _make_walk_forward_folds(seasons_in_data)
    logger.info("Walk-forward folds: %d  seasons=%s", len(folds), seasons_in_data)

    tft_hash_cols = [c for c in df.columns if c not in ("player_id", "game_id")]
    data_hash = _data_hash(df, tft_hash_cols, target_col)

    ds = TFTDataset(config)
    logger.info("Preparing dataframe for TFT (computing global time_idx)…")
    all_prepared = ds.prepare_dataframe(df, target=target)

    if n_optuna_trials > 0:
        logger.info(
            "Optuna TFT: %d trials on fold (train=%s → val=%d)…",
            n_optuna_trials, folds[-1][0], folds[-1][1],
        )
        config = _run_optuna_tft(
            all_prepared, folds[-1], target, config, n_trials=n_optuna_trials
        )
        ds = TFTDataset(config)
    else:
        logger.info("Optuna disabled (n_trials=0). Using base TFTConfig params.")

    if checkpoint_path:
        logger.info("TFT fine-tune mode: checkpoint=%s (applied to last fold only)", checkpoint_path)
    logger.info("Running TFT walk-forward CV…")
    fold_results, oof_df, final_model = _walk_forward_cv_tft(
        all_prepared, folds, target, ds, config, checkpoint_path=checkpoint_path
    )

    if not fold_results:
        raise RuntimeError(
            "No TFT folds completed — the feature matrix may be empty or "
            "seasons may not match data in the DB."
        )

    agg_mae = float(np.mean([fr.mae for fr in fold_results]))
    agg_rmse = float(np.mean([fr.rmse for fr in fold_results]))
    logger.info("Aggregate CV: MAE=%.3f  RMSE=%.3f", agg_mae, agg_rmse)

    attention_importances = _compute_final_attention_importances(
        final_model=final_model,
        all_prepared=all_prepared,
        folds=folds,
        target=target,
        ds=ds,
        config=config,
    )

    run_id, oof_path = persist_training_artifacts(
        target=target,
        position_filter=position_filter,
        seasons_in_data=seasons_in_data,
        config=config,
        n_optuna_trials=n_optuna_trials,
        fold_results=fold_results,
        agg_mae=agg_mae,
        agg_rmse=agg_rmse,
        attention_importances=attention_importances,
        data_hash=data_hash,
        final_model=final_model,
        oof_df=oof_df,
        out_dir=out_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    )

    return TFTTrainResult(
        fold_results=fold_results,
        oof_df=oof_df,
        config=config,
        run_id=run_id,
        attention_importances=attention_importances,
        oof_path=oof_path,
    )


def main() -> None:
    from ml.tft_cli import main as _main

    _main()


if __name__ == "__main__":
    main()
