"""
ml/inference_client.py

Model-artifact I/O for the projection pipeline.

Owns the 7 inference methods extracted from PipelineRunner:
  - load_and_run_stacking
  - run_tft_inference
  - align_features_to_model
  - build_inference_features
  - load_latest_mlflow_model
  - try_onnx_pred
  - load_ridge_coefs

PipelineRunner holds thin delegates to each of these so that
patch.object(runner, "_load_and_run_stacking", ...) continues to work
in existing tests.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ml.utils import FEATURE_COLS, run_onnx_inference, _ONNX_DIR

logger = logging.getLogger(__name__)


class InferenceClient:
    """
    Owns model-artifact loading and inference for the projection pipeline.

    Args:
        oof_dir:              Directory containing base learner OOF CSVs and
                              ridge_{stat}_{position}_coefs.json files.
        mlflow_tracking_uri:  MLflow tracking URI. Pass "" to disable.
    """

    def __init__(self, oof_dir: Path, mlflow_tracking_uri: str = "") -> None:
        self.oof_dir = oof_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri

    # ------------------------------------------------------------------
    # Primary stacking entry point
    # ------------------------------------------------------------------

    def load_and_run_stacking(
        self,
        kalman_df: pd.DataFrame,
        stat: str,
        position: str,
        *,
        season: Optional[int] = None,
        week: Optional[int] = None,
        dry_run_mode: bool = False,
        model_loader: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Load base models from MLflow and produce blended predictions.

        TFT: runs full TimeSeriesDataSet inference when not dry_run, TFT model
        exists, season/week/DB available, and prior rows exist. Else uses
        kalman_est_{stat} proxy.

        Args:
            model_loader: Optional callable with same signature as
                load_latest_mlflow_model. When supplied (by PipelineRunner's
                delegate), patch.object(runner, "_load_latest_mlflow_model")
                continues to work correctly. Defaults to self.load_latest_mlflow_model.

        Raises RuntimeError if neither XGB nor LGB model is found, which
        causes the caller (_run_stacking_step) to fall back to kalman_est.
        """
        import mlflow
        import mlflow.xgboost
        import mlflow.lightgbm

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        _load = model_loader if model_loader is not None else self.load_latest_mlflow_model

        # Build feature matrix: kalman_est_* columns pass through directly.
        X_df = self.build_inference_features(kalman_df)
        n_rows = len(X_df)

        # Load the most recent XGB, LGB, CB, and TFT models from MLflow.
        xgb_model = _load("xgb", stat, position)
        lgbm_model = _load("lgbm", stat, position)
        catboost_model = _load("catboost", stat, position)
        tft_model = _load("tft", stat, position)

        if xgb_model is None and lgbm_model is None and catboost_model is None:
            raise RuntimeError(
                f"No base model artifacts found for stat={stat} in MLflow "
                f"experiments 'xgb_{stat}' / 'lgbm_{stat}' / 'catboost_{stat}'."
            )

        # Collect base-learner predictions. Order must match ridge coefs: xgb, lgbm, catboost, tft.
        # ONNX fast path: try ml/onnx/{learner}_{stat}_{pos}.onnx first (2-5× faster than MLflow).
        # Falls back to the loaded MLflow model if ONNX file is absent or onnxruntime not installed.
        base_preds: list[np.ndarray] = []
        X_arr = X_df.values.astype("float32")  # shared float32 array for ONNX sessions

        xgb_onnx = self.try_onnx_pred("xgb", stat, position, X_arr, n_rows)
        if xgb_onnx is not None:
            base_preds.append(xgb_onnx)
            logger.info("XGB ONNX inference: %d predictions for stat=%s", n_rows, stat)
        elif xgb_model is not None:
            X_xgb, use_array = self.align_features_to_model(xgb_model, X_df)
            inp = X_xgb.values if use_array else X_xgb
            base_preds.append(np.array(xgb_model.predict(inp), dtype=float))
            logger.info("XGB MLflow inference: %d predictions for stat=%s", n_rows, stat)

        lgbm_onnx = self.try_onnx_pred("lgbm", stat, position, X_arr, n_rows)
        if lgbm_onnx is not None:
            base_preds.append(lgbm_onnx)
            logger.info("LGB ONNX inference: %d predictions for stat=%s", n_rows, stat)
        elif lgbm_model is not None:
            X_lgb, use_array = self.align_features_to_model(lgbm_model, X_df)
            inp = X_lgb.values if use_array else X_lgb
            base_preds.append(np.array(lgbm_model.predict(inp), dtype=float))
            logger.info("LGB MLflow inference: %d predictions for stat=%s", n_rows, stat)

        cb_onnx = self.try_onnx_pred("catboost", stat, position, X_arr, n_rows)
        if cb_onnx is not None:
            base_preds.append(cb_onnx)
            logger.info("CB ONNX inference: %d predictions for stat=%s", n_rows, stat)
        elif catboost_model is not None:
            X_cb, use_array = self.align_features_to_model(catboost_model, X_df)
            inp = X_cb.values if use_array else X_cb
            base_preds.append(np.array(catboost_model.predict(inp), dtype=float))
            logger.info("CB MLflow inference: %d predictions for stat=%s", n_rows, stat)

        # TFT: real inference when possible, else kalman proxy.
        tft_pred = self.run_tft_inference(
            kalman_df=kalman_df,
            stat=stat,
            position=position,
            tft_model=tft_model,
            season=season,
            week=week,
            dry_run_mode=dry_run_mode,
        )
        base_preds.append(tft_pred)

        # Apply Ridge blending weights + intercept.
        # Fail closed when position-specific coefs are missing — equal-weight
        # averaging silently destroyed QB passing_yards quality historically.
        ridge_result = self.load_ridge_coefs(stat, position=position)
        if ridge_result is None:
            raise FileNotFoundError(
                f"Missing Ridge coefs for stat={stat!r} position={position!r} "
                f"under {self.oof_dir}. Refusing equal-weight fallback. "
                f"Expected ridge_{stat}_{position}_coefs.json."
            )
        coefs, intercept = ridge_result
        if len(coefs) != len(base_preds):
            raise ValueError(
                f"Ridge coef length {len(coefs)} != base learner count "
                f"{len(base_preds)} for stat={stat!r} position={position!r}. "
                "Refusing silent equal-weight fallback."
            )
        stacked = sum(w * p for w, p in zip(coefs, base_preds)) + intercept

        logger.info(
            "Stacked estimates for stat=%s: n=%d mean=%.2f std=%.2f",
            stat, len(stacked), float(np.mean(stacked)), float(np.std(stacked)),
        )
        return np.asarray(stacked, dtype=float)

    # ------------------------------------------------------------------
    # TFT inference
    # ------------------------------------------------------------------

    def run_tft_inference(
        self,
        kalman_df: pd.DataFrame,
        stat: str,
        position: str,
        tft_model: Optional[object],
        *,
        season: Optional[int] = None,
        week: Optional[int] = None,
        dry_run_mode: bool = False,
    ) -> np.ndarray:
        """
        Run TFT model inference or fall back to kalman_est_{stat} proxy.

        Real TFT inference requires: tft_model loaded, not dry_run, season/week
        set, DATABASE_URL, and prior feature_matrix rows. Players with fewer
        than min_encoder_length (4) prior games get kalman proxy.
        """
        kalman_col = f"kalman_est_{stat}"
        fallback = (
            kalman_df[kalman_col].fillna(0.0).values.astype(float)
            if kalman_col in kalman_df.columns
            else np.zeros(len(kalman_df), dtype=float)
        )

        if stat == "passing_yards" and position == "QB":
            logger.warning(
                "TFT inference disabled for passing_yards/QB pending validation; using Kalman proxy instead."
            )
            return fallback
        if tft_model is None or dry_run_mode or season is None or week is None:
            return fallback
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return fallback
        try:
            from ml.utils import TARGET_COL_MAP, load_feature_matrix_prior
            from ml.tft_model import TFTConfig, TFTDataset

            target_col = TARGET_COL_MAP.get(stat)
            if not target_col:
                return fallback
            player_ids = kalman_df["player_id"].astype(str).tolist()
            prior_df = load_feature_matrix_prior(
                db_url, season, week, player_ids, position_filter=position
            )
            if prior_df.empty:
                return fallback
            # Build current rows from kalman_df with target placeholder
            current = kalman_df.copy()
            if target_col not in current.columns:
                current[target_col] = 0.0
            else:
                current[target_col] = current[target_col].fillna(0.0)
            # Align columns: prior_df has full feature_matrix schema.
            # Reindex current to match prior_df columns, fill missing with 0.
            current = current.reindex(columns=prior_df.columns, fill_value=0.0)
            combined = pd.concat([prior_df, current], ignore_index=True)
            cfg = TFTConfig(target=stat)
            ds = TFTDataset(cfg)
            prepared = ds.prepare_dataframe(combined, target=stat)
            min_season = int(prepared["season"].min())
            min_pred_time = int(
                (season - min_season) * TFTDataset.WEEKS_PER_SEASON + (week - 1)
            )
            train_prepared = prepared[
                (prepared["season"] < season)
                | ((prepared["season"] == season) & (prepared["week"] < week))
            ]
            if train_prepared.empty:
                return fallback
            from pytorch_forecasting import TimeSeriesDataSet

            train_ds = ds._make_dataset_raw(train_prepared, stat)
            pred_ds = TimeSeriesDataSet.from_dataset(
                train_ds,
                prepared,
                predict=False,
                stop_randomization=True,
                min_prediction_idx=min_pred_time,
            )
            if len(pred_ds) == 0:
                return fallback
            loader = pred_ds.to_dataloader(
                train=False, batch_size=cfg.batch_size, num_workers=0
            )
            raw = tft_model.predict(loader, mode="prediction", return_index=True)
            pred_arr = (
                raw.output.squeeze(-1).cpu().numpy().astype(float)
                if hasattr(raw.output, "cpu")
                else np.array(raw.output, dtype=float).squeeze(-1)
            )
            pred_by_player: dict[str, float] = {}
            for i in range(len(raw.index)):
                pid = str(raw.index["player_id"].iloc[i])
                pred_by_player[pid] = float(pred_arr[i])
            out = np.array(
                [pred_by_player.get(str(pid), np.nan) for pid in kalman_df["player_id"]],
                dtype=float,
            )
            mask = np.isnan(out)
            if mask.any():
                out[mask] = fallback[mask]
            logger.info(
                "TFT inference: %d predictions for stat=%s (position=%s)",
                len(out), stat, position,
            )
            return out
        except Exception as exc:
            logger.debug("TFT inference failed (%s), using kalman proxy", exc)
            return fallback

    # ------------------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------------------

    def align_features_to_model(
        self, model: object, X_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, bool]:
        """
        Align X_df to the model's expected feature set.

        Models in MLflow may have been trained with fewer features (e.g. 45) when
        feature_matrix had fewer columns. Inference passes all FEATURE_COLS (113).
        Subset and reorder to match model's feature_names_in_ / feature_name_ /
        feature_names_ to avoid "expected: 45, got 113" shape mismatch.

        Fallback: XGBoost loaded from MLflow often loses feature names; use
        n_features_in_ and take first N columns from X_df (FEATURE_COLS order).
        """
        expected: list[str] = []
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            expected = list(model.feature_names_in_)
        elif hasattr(model, "feature_name_") and model.feature_name_:
            expected = list(model.feature_name_)
        elif hasattr(model, "feature_names_") and model.feature_names_:
            expected = list(model.feature_names_)
        elif hasattr(model, "get_booster"):
            names = model.get_booster().feature_names
            expected = list(names) if names else []

        if expected:
            n = len(X_df)
            aligned = pd.DataFrame(
                {
                    col: (
                        X_df[col].values
                        if col in X_df.columns
                        else np.zeros(n, dtype=np.float64)
                    )
                    for col in expected
                },
                index=X_df.index,
            )
            return (aligned[expected], False)

        # Fallback: XGB/MLflow often loses feature names; use n_features_in_
        n_want = getattr(model, "n_features_in_", None)
        if n_want is not None and n_want > 0 and n_want < X_df.shape[1]:
            cols = [c for c in FEATURE_COLS if c in X_df.columns][:n_want]
            if len(cols) < n_want:
                cols = list(X_df.columns[:n_want])
            # use_array=True: pass .values to bypass column-name validation
            return (X_df[cols].copy(), True)
        return (X_df, False)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def build_inference_features(self, kalman_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a feature DataFrame aligned with XGB/LGB FEATURE_COLS.

        FEATURE_COLS uses kalman_est_* names (Bucket 1) which match the live
        feature_matrix columns directly — no column renaming required.
        Buckets 2-7: columns pass through from kalman_df if present; else 0.0.
        All missing columns (sparse data in dry_run) → 0.0.

        Returns:
            DataFrame with exactly FEATURE_COLS column order, all float.
        """
        work = kalman_df.copy()
        n = len(work)
        result = pd.DataFrame(
            {
                col: (
                    pd.to_numeric(work[col], errors="coerce").fillna(0.0).values
                    if col in work.columns
                    else np.zeros(n, dtype=np.float64)
                )
                for col in FEATURE_COLS
            },
            index=work.index,
        )
        return result

    # ------------------------------------------------------------------
    # MLflow model loading
    # ------------------------------------------------------------------

    def load_latest_mlflow_model(
        self,
        learner: str,
        stat: str,
        position: Optional[str] = None,
    ) -> Optional[object]:
        """
        Find the most recent finished MLflow run in the '{learner}_{stat}'
        experiment and return the loaded model. Returns None if no run exists
        or if any MLflow call fails.
        """
        try:
            import mlflow
            # Try position-specific experiment name first (XGB style)
            exp_name = f"{learner}_{stat}_{position}" if position else f"{learner}_{stat}"
            exp = mlflow.get_experiment_by_name(exp_name)

            # Fall back to general experiment name (LGBM/CatBoost/TFT style)
            if exp is None:
                exp_name = f"{learner}_{stat}"
                exp = mlflow.get_experiment_by_name(exp_name)

            if exp is None:
                logger.debug("MLflow experiment for '%s' '%s' not found.", learner, stat)
                return None

            filter_str = "status = 'FINISHED'"
            if position and learner != "tft":
                # Must match position exactly as logged (e.g. "QB") or "all"
                filter_str += f" AND params.position_filter = '{position}'"

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
                filter_string=filter_str,
            )

            if runs.empty:
                logger.debug(
                    "No finished runs for %s %s %s.", learner, stat, position
                )
                return None

            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            if learner == "xgb":
                return mlflow.xgboost.load_model(model_uri)
            elif learner == "lgbm":
                return mlflow.lightgbm.load_model(model_uri)
            elif learner == "catboost":
                import mlflow.catboost
                return mlflow.catboost.load_model(model_uri)
            elif learner == "tft":
                import mlflow.pytorch
                return mlflow.pytorch.load_model(model_uri)
            else:
                return mlflow.pyfunc.load_model(model_uri)

        except Exception as exc:
            logger.debug(
                "Could not load %s model for stat=%s: %s", learner, stat, exc
            )
            return None

    # ------------------------------------------------------------------
    # ONNX fast path
    # ------------------------------------------------------------------

    def try_onnx_pred(
        self,
        learner: str,
        stat: str,
        position: Optional[str],
        X_arr: np.ndarray,
        expected_rows: int,
    ) -> Optional[np.ndarray]:
        """
        Try ONNX fast-path for a single base learner.

        Looks for ml/onnx/{learner}_{stat}_{position}.onnx; returns predictions
        (shape [expected_rows]) or None if file is absent or onnxruntime is missing.
        """
        pos_tag = (position or "all").replace("/", "_")
        onnx_path = _ONNX_DIR / f"{learner}_{stat}_{pos_tag}.onnx"
        if not onnx_path.exists():
            return None
        preds = run_onnx_inference(onnx_path, X_arr)
        if preds is None or len(preds) != expected_rows:
            return None
        return preds

    # ------------------------------------------------------------------
    # Ridge coefficient loading
    # ------------------------------------------------------------------

    def load_ridge_coefs(
        self, stat: str, position: Optional[str] = None
    ) -> Optional[tuple[list[float], float]]:
        """
        Load Ridge blending coefficients + intercept.

        Lookup order:
          1. ml/oof/ridge_{stat}_{position}_coefs.json when position is given
          2. ml/oof/ridge_{stat}_coefs.json only for non-positioned callers

        The file is written by stacking_ensemble.py.
        Format: {"xgb": 0.4, "lgbm": 0.35, ..., "intercept": 1.2}.

        Returns (coefs_list, intercept) or None when no file is found.
        """
        candidates: list[Path] = []
        if position:
            candidates.append(self.oof_dir / f"ridge_{stat}_{position}_coefs.json")
        else:
            candidates.append(self.oof_dir / f"ridge_{stat}_coefs.json")

        for coef_path in candidates:
            if not coef_path.exists():
                continue
            try:
                with open(coef_path) as f:
                    data = json.load(f)
                intercept = float(data.pop("intercept", 0.0))
                learner_keys = ["xgb", "lgbm", "catboost", "tft"]
                coefs = [float(data[k]) for k in learner_keys if k in data]
                if not coefs:
                    continue
                logger.debug("Loaded Ridge coefs from %s: %s", coef_path.name, data)
                return coefs, intercept
            except Exception as exc:
                logger.debug(
                    "Could not load Ridge coefs from %s: %s", coef_path, exc
                )
        return None
