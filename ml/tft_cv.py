"""
Walk-forward CV, fold execution, and model-introspection helpers for TFT.

This module isolates the execution engine from the higher-level orchestration
in `ml.tft_training`, which handles data validation, MLflow, OOF persistence,
and CLI concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.tft_model import TFTConfig, TFTDataset, logger
from ml.utils import TARGET_COL_MAP, FoldResult, _compute_metrics

_OPTUNA_EPOCHS: int = 3


def _extract_attention_importances(
    model: Any,
    val_loader: Any,
    train_dataset: Any,
) -> dict[str, float]:
    """
    Extract variable-selection attention weights from TFT as feature importances.

    The TFT variable selection networks produce a weight per input variable at
    each timestep. This function averages those weights over the validation
    batches to produce a single importance score per variable.

    Returns a normalised dict (values sum to 1.0). Returns empty dict on any
    error — missing importances must never abort a training run.

    ── How TFT variable importances differ from tree importances ────────────
    XGBoost/LightGBM: feature_importance() = average gain over all trees.
    TFT: attention weight = the fraction of "input budget" the variable
      selection network allocated to this variable at this timestep.
    Both are interpretability tools; neither is causal.
    """
    try:
        import torch

        model.eval()
        all_encoder_weights: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                raw_out = model(x)
                interp = model.interpret_output(raw_out, reduction="sum")
                enc_w = interp.get("encoder_variables")
                if enc_w is not None:
                    all_encoder_weights.append(enc_w.cpu().numpy())
                if len(all_encoder_weights) >= 10:
                    break

        if not all_encoder_weights:
            return {}

        avg_weights = np.stack(all_encoder_weights, axis=0).mean(axis=0)
        total = float(avg_weights.sum()) or 1.0
        norm_weights = avg_weights / total

        var_names = list(train_dataset.encoder_variables)
        return {
            name: float(w)
            for name, w in zip(var_names, norm_weights)
        }

    except Exception as exc:
        logger.warning(
            "Could not extract attention importances (skipping): %s", exc
        )
        return {}


def _compute_final_attention_importances(
    final_model: Any,
    all_prepared: pd.DataFrame,
    folds: list[tuple[list[int], int]],
    target: str,
    ds: TFTDataset,
    config: TFTConfig,
) -> dict[str, float]:
    """
    Compute attention importances for the final, most data-rich fold model.

    This stays in the execution layer because it rebuilds the final fold's
    datasets and performs model-side interpretation work.
    """
    if final_model is None:
        return {}

    try:
        import torch

        final_model.cpu()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        last_train_seasons, last_val_season = folds[-1]
        last_context_df = all_prepared[
            all_prepared["season"].isin(last_train_seasons + [last_val_season])
        ].copy()
        min_val_time_idx = int(
            (last_val_season - int(all_prepared["season"].min()))
            * TFTDataset.WEEKS_PER_SEASON
        )
        last_train_ds = ds._make_dataset_raw(
            all_prepared[all_prepared["season"].isin(last_train_seasons)].copy(),
            target,
        )
        from pytorch_forecasting import TimeSeriesDataSet

        last_val_ds = TimeSeriesDataSet.from_dataset(
            last_train_ds,
            last_context_df,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=min_val_time_idx,
        )
        last_val_loader = last_val_ds.to_dataloader(
            train=False, batch_size=config.batch_size, num_workers=0
        )
        return _extract_attention_importances(
            final_model, last_val_loader, last_train_ds
        )
    except Exception as exc:
        logger.warning("Attention importances skipped: %s", exc)
        return {}


def _run_optuna_tft(
    all_prepared_df: pd.DataFrame,
    last_fold: tuple[list[int], int],
    target: str,
    base_config: TFTConfig,
    n_trials: int = 20,
) -> TFTConfig:
    """
    Run Optuna on the most-recent walk-forward fold to tune TFT hyperparameters.

    Mirrors xgb_model._run_optuna() and lgbm_model._run_optuna(): runs only
    on the MOST RECENT fold because that fold is most representative of the
    current player-performance environment.
    """
    if n_trials == 0:
        logger.info("Optuna disabled (n_trials=0). Using base config params.")
        return base_config

    try:
        import dataclasses
        import lightning.pytorch as pl
        import optuna
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss

        torch.set_num_threads(1)
    except ImportError as exc:
        raise ImportError(
            "optuna, torch, pytorch_forecasting, and lightning are "
            "required for TFT hyperparameter search."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_seasons, val_season = last_fold
    min_season = int(all_prepared_df["season"].min())
    min_val_time_idx = int((val_season - min_season) * TFTDataset.WEEKS_PER_SEASON)

    ds = TFTDataset(base_config)
    train_df = all_prepared_df[
        all_prepared_df["season"].isin(train_seasons)
    ].copy()
    context_df = all_prepared_df[
        all_prepared_df["season"].isin(train_seasons + [val_season])
    ].copy()

    if train_df.empty:
        logger.warning("Optuna: empty training data — skipping search.")
        return base_config

    try:
        train_dataset = ds._make_dataset_raw(train_df, target)
        val_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            context_df,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=min_val_time_idx,
        )
    except Exception as exc:
        logger.warning("Optuna: could not build datasets (%s) — skipping.", exc)
        return base_config

    if len(val_dataset) == 0:
        logger.warning("Optuna: empty val dataset — skipping search.")
        return base_config

    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=base_config.batch_size, num_workers=0
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=base_config.batch_size, num_workers=0
    )

    def objective(trial: Any) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.2])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=base_config.attention_head_size,
            dropout=dropout,
            hidden_continuous_size=base_config.hidden_continuous_size,
            output_size=7,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=2,
            log_interval=-1,
        )
        trainer = pl.Trainer(
            max_epochs=_OPTUNA_EPOCHS,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=base_config.gradient_clip_val,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        if hasattr(val_loss, "item"):
            val_loss = val_loss.item()
        return float(val_loss)

    logger.info("Optuna TFT: starting %d trials on most-recent fold…", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(
        "Optuna TFT best val_loss=%.4f  params=%s",
        study.best_value,
        best,
    )

    return dataclasses.replace(
        base_config,
        hidden_size=best["hidden_size"],
        dropout=best["dropout"],
        learning_rate=best["learning_rate"],
    )


def _train_fold_tft(
    fold_idx: int,
    all_prepared_df: pd.DataFrame,
    train_seasons: list[int],
    val_season: int,
    target: str,
    ds: TFTDataset,
    config: TFTConfig,
    checkpoint_path: Optional[str] = None,
) -> tuple[Any, pd.DataFrame, Optional[FoldResult]]:
    """
    Train TFT on train_seasons, generate OOF predictions for val_season.

    Returns:
        (model, oof_df, fold_result)
        Returns (None, empty_df, None) if either dataset is empty.
    """
    try:
        import lightning.pytorch as pl
        import torch
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss

        torch.set_num_threads(1)
    except ImportError as exc:
        raise ImportError(
            "torch, pytorch_forecasting, and lightning are required. "
            "Install: pip install pytorch-forecasting torch lightning"
        ) from exc

    target_col = TARGET_COL_MAP[target]
    min_season = int(all_prepared_df["season"].min())

    train_df = all_prepared_df[
        all_prepared_df["season"].isin(train_seasons)
    ].copy()
    val_df = all_prepared_df[
        all_prepared_df["season"] == val_season
    ].copy()

    if train_df.empty or val_df.empty:
        logger.warning(
            "Fold %d: skipping — train empty=%s  val empty=%s",
            fold_idx, train_df.empty, val_df.empty,
        )
        return None, pd.DataFrame(), None

    train_dataset = ds._make_dataset_raw(train_df, target)
    context_df = all_prepared_df[
        all_prepared_df["season"].isin(train_seasons + [val_season])
    ].copy()
    min_val_time_idx = int(
        (val_season - min_season) * TFTDataset.WEEKS_PER_SEASON
    )
    try:
        val_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset,
            context_df,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=min_val_time_idx,
        )
    except Exception as exc:
        logger.warning("Fold %d: could not build val dataset: %s", fold_idx, exc)
        return None, pd.DataFrame(), None

    if len(val_dataset) == 0:
        logger.warning("Fold %d: empty val dataset after from_dataset(), skipping", fold_idx)
        return None, pd.DataFrame(), None

    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=config.batch_size, num_workers=0
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=config.batch_size, num_workers=0
    )

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        log_interval=-1,
    )

    ckpt_dir = Path("ml/checkpoints/tft") / f"{target}_fold{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    if checkpoint_path:
        logger.info(
            "  Fold %d: fine-tuning from checkpoint: %s", fold_idx, checkpoint_path
        )
    logger.info(
        "  Fold %d: training TFT on seasons=%s → val=%d  "
        "(encoder_len=%d  epochs_max=%d)",
        fold_idx, train_seasons, val_season,
        config.max_encoder_length, config.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

    raw_pred = model.predict(
        val_loader,
        mode="prediction",
        return_index=True,
    )
    preds_tensor = raw_pred.output
    index_df = raw_pred.index

    if isinstance(preds_tensor, torch.Tensor):
        y_pred_arr = preds_tensor.squeeze(-1).cpu().numpy().astype(float)
    else:
        y_pred_arr = np.array(preds_tensor, dtype=float).squeeze(-1)

    if "time_idx_first_prediction" in index_df.columns:
        idx_time_col = "time_idx_first_prediction"
    elif "time_idx" in index_df.columns:
        idx_time_col = "time_idx"
    else:
        idx_time_col = index_df.columns[-1]
        logger.warning(
            "Fold %d: could not find time_idx column in predict index; "
            "using column '%s' as fallback.", fold_idx, idx_time_col
        )

    lookup_cols = ["player_id", "time_idx", "season", "week", target_col]
    if "game_id" in all_prepared_df.columns:
        lookup_cols.append("game_id")
    lookup = (
        all_prepared_df[lookup_cols]
        .drop_duplicates(subset=["player_id", "time_idx"])
        .set_index(["player_id", "time_idx"])
    )

    oof_rows: list[dict[str, Any]] = []
    for i in range(len(index_df)):
        player_id = str(index_df["player_id"].iloc[i])
        time_idx = int(index_df[idx_time_col].iloc[i])
        y_pred = float(y_pred_arr[i])

        try:
            src = lookup.loc[(player_id, time_idx)]
            y_true = float(src[target_col])
            season = int(src["season"])
            week = int(src["week"])
            game_id = str(src["game_id"]) if "game_id" in lookup.columns else ""
        except KeyError:
            continue

        row: dict[str, Any] = {
            "player_id": player_id,
            "game_id": game_id,
            "season": season,
            "week": week,
            "y_true": y_true,
            "y_pred": y_pred,
            "fold_idx": fold_idx,
        }
        if "position" in lookup.columns:
            try:
                row["position"] = str(src["position"])
            except Exception:
                pass
        oof_rows.append(row)

    if not oof_rows:
        logger.warning(
            "Fold %d: no OOF rows could be reconstructed — "
            "check that val_season=%d is in all_prepared_df and that "
            "the time_idx lookup is correct.",
            fold_idx, val_season,
        )
        return model, pd.DataFrame(), None

    oof_df = pd.DataFrame(oof_rows)
    mae, rmse = _compute_metrics(
        oof_df["y_true"].values,
        oof_df["y_pred"].values,
    )

    logger.info(
        "  Fold %d: n_train=%d  n_val=%d  MAE=%.2f  RMSE=%.2f",
        fold_idx, len(train_df), len(val_df), mae, rmse,
    )

    fold_result = FoldResult(
        fold_idx=fold_idx,
        train_seasons=sorted(train_seasons),
        val_season=val_season,
        mae=mae,
        rmse=rmse,
        n_train=len(train_df),
        n_val=len(val_df),
    )
    return model, oof_df, fold_result


def _walk_forward_cv_tft(
    all_prepared_df: pd.DataFrame,
    folds: list[tuple[list[int], int]],
    target: str,
    ds: TFTDataset,
    config: TFTConfig,
    checkpoint_path: Optional[str] = None,
) -> tuple[list[FoldResult], pd.DataFrame, Any]:
    """
    Run walk-forward CV over all folds for TFT.

    Identical temporal ordering constraints to xgb_model._walk_forward_cv:
      assert train_seasons == sorted(train_seasons)
      assert max(train_seasons) < val_season
    """
    all_oof: list[pd.DataFrame] = []
    fold_results: list[FoldResult] = []
    final_model: Any = None

    for fold_idx, (train_seasons, val_season) in enumerate(folds):
        assert train_seasons == sorted(train_seasons), (
            f"Fold {fold_idx}: train_seasons={train_seasons} is not sorted ascending. "
            "Walk-forward CV requires a chronologically ordered expanding window. "
            "This is a data-leakage bug — aborting."
        )
        assert max(train_seasons) < val_season, (
            f"Fold {fold_idx}: val_season={val_season} is NOT strictly greater than "
            f"max(train_seasons)={max(train_seasons)}. "
            "Val season must always be in the future relative to training data. "
            "This is a data-leakage bug — aborting."
        )

        is_last_fold = fold_idx == len(folds) - 1
        ckpt = checkpoint_path if (checkpoint_path and is_last_fold) else None
        model, oof_df, fold_result = _train_fold_tft(
            fold_idx, all_prepared_df, train_seasons, val_season,
            target, ds, config, checkpoint_path=ckpt,
        )
        if fold_result is None:
            continue

        all_oof.append(oof_df)
        fold_results.append(fold_result)
        final_model = model

        import gc

        gc.collect()

    oof_df_full = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    return fold_results, oof_df_full, final_model


__all__ = [
    "_compute_final_attention_importances",
    "_extract_attention_importances",
    "_run_optuna_tft",
    "_train_fold_tft",
    "_walk_forward_cv_tft",
]
