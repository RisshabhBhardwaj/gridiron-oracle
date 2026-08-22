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
import math
import numbers
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ml.utils import FEATURE_COLS, MISSING_VALUE_SENTINEL, run_onnx_inference, _ONNX_DIR
from ml.feature_contract import assert_model_input_columns

logger = logging.getLogger(__name__)

assert_model_input_columns(FEATURE_COLS, consumer="inference client")


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
        # Load Ridge first — Phase-5 stacks may keep only a subset (e.g. lgbm+catboost).
        # Fail closed when position-specific coefs are missing.
        ridge_result = self.load_ridge_coefs(stat, position=position)
        if ridge_result is None:
            raise FileNotFoundError(
                f"Missing Ridge coefs for stat={stat!r} position={position!r} "
                f"under {self.oof_dir}. Refusing equal-weight fallback. "
                f"Expected ridge_{stat}_{position}_coefs.json."
            )
        coefs, intercept, learner_order = ridge_result

        # build_inference_features leaves missing values as real NaN, matching
        # XGBoost's training input exactly (ml/xgb_model.py never fillna's).
        # LightGBM and CatBoost were trained on MISSING_VALUE_SENTINEL-filled
        # data instead (their sklearn APIs need a concrete float), so serving
        # must feed them that exact sentinel too, or "missing" silently reads
        # as a real, in-range observed value to the trained splits.
        X_df = self.build_inference_features(kalman_df)
        X_df_sentinel = X_df.fillna(MISSING_VALUE_SENTINEL)
        n_rows = len(X_df)
        X_arr = X_df.values.astype("float32")
        X_arr_sentinel = X_df_sentinel.values.astype("float32")

        def _frames_for(learner: str) -> tuple[pd.DataFrame, np.ndarray]:
            if learner in ("lgbm", "catboost"):
                return X_df_sentinel, X_arr_sentinel
            return X_df, X_arr

        # Only load/run learners present in the Ridge artifact (kill TFT/XGB when absent).
        def _pred_for(learner: str) -> np.ndarray:
            if learner == "tft":
                tft_model = _load("tft", stat, position)
                return self.run_tft_inference(
                    kalman_df=kalman_df,
                    stat=stat,
                    position=position,
                    tft_model=tft_model,
                    season=season,
                    week=week,
                    dry_run_mode=dry_run_mode,
                )
            learner_X_df, learner_X_arr = _frames_for(learner)
            model = _load(learner, stat, position)
            onnx = self.try_onnx_pred(learner, stat, position, learner_X_arr, n_rows)
            if onnx is not None:
                logger.info("%s ONNX inference: %d predictions for stat=%s", learner.upper(), n_rows, stat)
                return onnx
            if model is None:
                raise RuntimeError(
                    f"Ridge requires learner={learner!r} for stat={stat!r} position={position!r}, "
                    "but no ONNX/MLflow artifact was found."
                )
            X_aligned, use_array = self.align_features_to_model(model, learner_X_df)
            inp = X_aligned.values if use_array else X_aligned
            logger.info("%s MLflow inference: %d predictions for stat=%s", learner.upper(), n_rows, stat)
            return np.array(model.predict(inp), dtype=float)

        if not learner_order:
            raise RuntimeError(f"Ridge coef file for {stat}/{position} has no learner weights.")

        base_preds = [_pred_for(learner) for learner in learner_order]
        if len(coefs) != len(base_preds):
            raise ValueError(
                f"Ridge coef length {len(coefs)} != base learner count "
                f"{len(base_preds)} for stat={stat!r} position={position!r}. "
                "Refusing silent equal-weight fallback."
            )
        checked_preds: list[np.ndarray] = []
        for learner, prediction in zip(learner_order, base_preds):
            values = np.asarray(prediction, dtype=float)
            if values.shape != (n_rows,):
                raise ValueError(
                    f"Learner={learner!r} returned shape {values.shape}; expected "
                    f"({n_rows},) for stat={stat!r} position={position!r}."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Learner={learner!r} returned non-finite predictions for "
                    f"stat={stat!r} position={position!r}."
                )
            checked_preds.append(values)

        stacked = sum(w * p for w, p in zip(coefs, checked_preds)) + intercept
        if not np.isfinite(stacked).all():
            raise ValueError(
                f"Stacked output contains non-finite values for stat={stat!r} "
                f"position={position!r}."
            )

        logger.info(
            "Stacked estimates for stat=%s learners=%s: n=%d mean=%.2f std=%.2f",
            stat, ",".join(learner_order), len(stacked),
            float(np.mean(stacked)), float(np.std(stacked)),
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
        Build a feature DataFrame aligned with FEATURE_COLS.

        FEATURE_COLS uses kalman_est_* names (Bucket 1) which match the live
        feature_matrix columns directly — no column renaming required.
        Buckets 2-7: columns pass through from kalman_df if present; else NaN.

        Missing/absent values are left as real NaN — matching XGBoost's
        training input exactly (ml/xgb_model.py never fillna's). Callers that
        feed LightGBM or CatBoost must fillna(MISSING_VALUE_SENTINEL) on the
        result first, since those two were trained on sentinel-filled data
        (ml/lgbm_model.py, ml/catboost_model.py) — a bare 0.0 or NaN here
        would silently read as an in-range observed value to their learned
        splits instead of "missing". See load_and_run_stacking's _frames_for.

        Returns:
            DataFrame with exactly FEATURE_COLS column order, all float,
            NaN where missing.
        """
        work = kalman_df.copy()
        n = len(work)
        result = pd.DataFrame(
            {
                col: (
                    pd.to_numeric(work[col], errors="coerce").values
                    if col in work.columns
                    else np.full(n, np.nan, dtype=np.float64)
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
    ) -> Optional[tuple[list[float], float, list[str]]]:
        """
        Load Ridge blending coefficients + intercept.

        Lookup order:
          1. ml/oof/ridge_{stat}_{position}_coefs.json when position is given
          2. ml/oof/ridge_{stat}_coefs.json only for non-positioned callers

        The file is written by stacking_ensemble.py. Its schema is deliberately
        explicit and closed:
        ``{"learner_order": ["lgbm", "catboost"], "weights": {...},
        "intercept": 1.2}``.  A coefficient artifact is executable serving
        configuration, so malformed, partial, or non-finite files raise rather
        than being converted into an implicit equal-weight fallback.

        Returns (coefs_list, intercept, learner_order) or None when no file is found.
        learner_order is the subset of [xgb, lgbm, catboost, tft] present in the
        artifact — inference must only run those learners (Phase-5 kill policy).
        """
        candidates: list[Path] = []
        # A release manifest may pin executable coefficients outside ml/oof.
        # Resolve and SHA-verify that authority first; legacy manifests that do
        # not yet list coefficients retain the old lookup solely so their
        # malformed files fail with a useful schema error rather than masking
        # the release defect.
        if position:
            try:
                from ml.artifact_manifest import ManifestEntryMissing, get_manifest

                candidates.append(get_manifest().resolve_coefficient(stat, position))
            except ManifestEntryMissing:
                candidates.append(self.oof_dir / f"ridge_{stat}_{position}_coefs.json")
        else:
            candidates.append(self.oof_dir / f"ridge_{stat}_coefs.json")

        for coef_path in candidates:
            if not coef_path.exists():
                continue
            def _reject_json_constant(value: str):
                raise ValueError(f"JSON non-finite constant {value!r} is forbidden")

            try:
                with open(coef_path) as f:
                    data = json.load(f, parse_constant=_reject_json_constant)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid Ridge coefficient JSON at {coef_path}: {exc}") from exc

            expected_keys = {"learner_order", "weights", "intercept"}
            identity_selection = data.get("selection_mode") == "validated_best_base" if isinstance(data, dict) else False
            if identity_selection:
                expected_keys = {*expected_keys, "selection_mode"}
            if not isinstance(data, dict) or set(data) != expected_keys:
                raise ValueError(
                    f"Invalid Ridge coefficient schema at {coef_path}: expected exactly "
                    f"{sorted(expected_keys)}, got {sorted(data) if isinstance(data, dict) else type(data).__name__}."
                )

            learner_order = data["learner_order"]
            weights = data["weights"]
            if (
                not isinstance(learner_order, list)
                or not learner_order
                or not all(isinstance(learner, str) for learner in learner_order)
                or len(set(learner_order)) != len(learner_order)
            ):
                raise ValueError(f"Invalid learner_order in Ridge coefficients at {coef_path}")
            if not isinstance(weights, dict) or set(weights) != set(learner_order):
                raise ValueError(
                    f"Ridge weights at {coef_path} must name exactly learner_order; "
                    f"got weights={sorted(weights) if isinstance(weights, dict) else type(weights).__name__}."
                )

            if identity_selection and (
                len(learner_order) != 1
                or weights[learner_order[0]] != 1.0
                or data["intercept"] != 0.0
            ):
                raise ValueError(
                    f"Identity base selection at {coef_path} must have exactly one "
                    "learner with weight=1.0 and intercept=0.0"
                )

            raw_values = [weights[learner] for learner in learner_order] + [data["intercept"]]
            if any(isinstance(value, bool) or not isinstance(value, numbers.Real) for value in raw_values):
                raise ValueError(f"Ridge coefficients at {coef_path} must be JSON numbers, not strings/bools")

            try:
                from ml.artifact_manifest import assert_learner_policy
                assert_learner_policy(
                    stat,
                    learner_order,
                    context=f"Ridge coefficients {coef_path}",
                    require_minimum=not identity_selection,
                )
                coefs = [float(weights[learner]) for learner in learner_order]
                intercept = float(data["intercept"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Ridge coefficients at {coef_path} must be numeric: {exc}") from exc

            if not all(math.isfinite(value) for value in [*coefs, intercept]):
                raise ValueError(f"Ridge coefficients at {coef_path} must all be finite")

            logger.debug(
                "Loaded Ridge coefs from %s learners=%s", coef_path.name, learner_order,
            )
            return coefs, intercept, learner_order
        return None
