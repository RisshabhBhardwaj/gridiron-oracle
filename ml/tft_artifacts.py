"""
Artifact persistence helpers for TFT training.

This module isolates MLflow logging and OOF artifact persistence from the core
training orchestration in `ml.tft_training`.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ml.tft_model import TFTConfig, logger
from ml.utils import FoldResult, save_oof


def _save_local_oof(
    oof_df: pd.DataFrame,
    target: str,
    out_dir: Path,
    position: Optional[str] = None,
) -> Optional[Path]:
    if oof_df.empty:
        return None

    pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return save_oof(
        oof_df, target, pseudo_run_id, out_dir, prefix="tft", position=position or "all"
    )


def persist_training_artifacts(
    *,
    target: str,
    position_filter: Optional[str],
    seasons_in_data: list[int],
    config: TFTConfig,
    n_optuna_trials: int,
    fold_results: list[FoldResult],
    agg_mae: float,
    agg_rmse: float,
    attention_importances: dict[str, float],
    data_hash: str,
    final_model: Any,
    oof_df: pd.DataFrame,
    out_dir: Path,
    mlflow_tracking_uri: Optional[str],
    mlflow_experiment: Optional[str],
) -> tuple[Optional[str], Optional[Path]]:
    """
    Persist TFT training outputs to MLflow and/or local OOF artifacts.

    Returns:
        `(run_id, oof_path)` where either value may be `None`.
    """
    run_id: Optional[str] = None
    oof_path: Optional[Path] = None

    use_mlflow = mlflow_tracking_uri != ""
    if use_mlflow and mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5001"
        )

    if use_mlflow:
        try:
            import mlflow
            import mlflow.pytorch

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            exp_name = mlflow_experiment or f"tft_{target}"
            mlflow.set_experiment(exp_name)

            with mlflow.start_run() as run:
                run_id = run.info.run_id

                mlflow.log_params({
                    "target": target,
                    "position_filter": position_filter or "all",
                    "seasons": str(seasons_in_data),
                    "n_folds": len(fold_results),
                    "p_hidden_size": config.hidden_size,
                    "p_attention_head_size": config.attention_head_size,
                    "p_dropout": config.dropout,
                    "p_hidden_continuous_size": config.hidden_continuous_size,
                    "p_learning_rate": config.learning_rate,
                    "p_max_epochs": config.max_epochs,
                    "p_batch_size": config.batch_size,
                    "p_max_encoder_length": config.max_encoder_length,
                    "p_min_encoder_length": config.min_encoder_length,
                    "n_optuna_trials": n_optuna_trials,
                })

                mlflow.log_metrics({"mae": agg_mae, "rmse": agg_rmse})

                for fr in fold_results:
                    mlflow.log_metrics({
                        f"fold_{fr.fold_idx}_mae": fr.mae,
                        f"fold_{fr.fold_idx}_rmse": fr.rmse,
                        f"fold_{fr.fold_idx}_n_train": fr.n_train,
                        f"fold_{fr.fold_idx}_n_val": fr.n_val,
                    }, step=fr.fold_idx)

                if attention_importances:
                    mlflow.log_metrics({
                        f"fi_{k}": v for k, v in attention_importances.items()
                    })

                mlflow.set_tags({
                    "training_data_hash": data_hash,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                if final_model is not None:
                    mlflow.pytorch.log_model(final_model, "model")

                if not oof_df.empty:
                    oof_path = save_oof(oof_df, target, run_id, out_dir, prefix="tft", position=position_filter or "all")
                    mlflow.log_artifact(str(oof_path))

                logger.info(
                    "MLflow run logged: experiment=%s  run_id=%s",
                    exp_name, run_id,
                )

        except Exception as exc:
            logger.error("MLflow logging failed (continuing without it): %s", exc)
            if not oof_df.empty and oof_path is None:
                oof_path = _save_local_oof(oof_df, target, out_dir, position=position_filter)

    elif not oof_df.empty:
        oof_path = _save_local_oof(oof_df, target, out_dir, position=position_filter)

    return run_id, oof_path


__all__ = ["persist_training_artifacts"]
