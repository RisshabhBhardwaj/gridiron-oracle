"""
ml/lgbm_model.py

LightGBM base learner — the second of three base models in the stacking ensemble.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY LIGHTGBM DIFFERS FROM XGBOOST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost grows trees level-wise (all leaves at depth d before expanding).
LightGBM grows leaf-wise: always splits the leaf with highest gain.

Consequences for this model:
  - num_leaves is the primary complexity parameter (not max_depth).
    A tree with depth=6 can have at most 2^6=64 leaves, but a leaf-wise
    tree with num_leaves=63 may have much shallower average depth.
    Overfitting risk is controlled by num_leaves + min_child_samples together.
  - Histogram-based splits are computed with GOSS (Gradient-based One-Side
    Sampling) + EFB (Exclusive Feature Bundling) → faster on wide datasets.
  - feature_fraction shuffles feature subsets per iteration (not per tree),
    making it distinct from XGBoost's colsample_bytree.
  - bagging_fraction + bagging_freq control row subsampling by iteration.

These differences mean the Optuna search space is genuinely different, not
just renamed parameters — the hyperparameter surfaces have different geometries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WALK-FORWARD CV (same constraint as xgb_model.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expanding-window only. Same hard AssertionError guard at every fold:
  assert train_seasons == sorted(train_seasons)
  assert max(train_seasons) < val_season

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOF FORMAT (identical to xgb_model.py — required for stacking)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Columns: player_id, game_id, season, week, y_true, y_pred, fold_idx
Saved to: ml/oof/lgbm_{target}_{run_id[:8]}.csv

The stacking meta-learner (stacking_ensemble.py) concatenates the XGBoost
and LightGBM OOF files as its input features. Identical column format is
non-negotiable.

Standalone usage:
  python -m ml.lgbm_model --seasons 2018-2024 --target receiving_yards
  python -m ml.lgbm_model --seasons 2018-2024 --target receiving_yards \\
      --position WR --n-trials 50 --out-dir ml/oof/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Thread-safety: must happen BEFORE any lightgbm / torch import ──────────────
# LightGBM ships its own libomp. If XGBoost or PyTorch also load their libomp,
# three OMP runtimes coexist → SIGSEGV on macOS ARM (exit code 139).
# configure_thread_env() sets KMP_DUPLICATE_LIB_OK + OMP_NUM_THREADS=1.
from ml.utils import configure_thread_env  # noqa: E402 — must precede lgbm import
configure_thread_env()

import lightgbm as lgb
import numpy as np
import pandas as pd

# ── Import shared utilities from xgb_model ────────────────────────────────────
# These are identical across all base learners — feature definitions, fold
# generation, metrics, OOF saving, DB loading, and the season-range parser.
from ml.utils import (
    FEATURE_COLS,
    TARGET_COL_MAP,
    FoldResult,
    _compute_metrics,
    _data_hash,
    _make_walk_forward_folds,
    _parse_seasons,
    export_onnx,
    load_feature_matrix,
    save_oof,
)
from ml.cohort_guards import validate_training_cohort
from ml.season_constants import assert_not_fitting_incomplete_season
from ml.reliability import build_training_manifest, write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── LightGBM defaults ─────────────────────────────────────────────────────────
#
# Key difference from XGBoost defaults:
#   - num_leaves=63 (leaf-wise, not max_depth)
#   - min_child_samples=20 (min rows per leaf, not min_child_weight)
#   - feature_fraction per iteration (not colsample_bytree per tree)
#   - bagging_fraction + bagging_freq for row subsampling by iteration
#   - lambda_l1 / lambda_l2 (LightGBM naming for L1/L2 regularisation)
#   - verbosity=-1 silences LightGBM's INFO/WARNING prints
#   - importance_type='gain' → total gain rather than split count, giving
#     a more informative importance signal comparable to XGBoost 'gain'
#
DEFAULT_LGBM_PARAMS: dict = {
    "n_estimators":      300,
    "num_leaves":        63,        # primary complexity param (leaf-wise)
    "learning_rate":     0.05,
    "feature_fraction":  0.8,       # column subsampling per iteration
    "bagging_fraction":  0.8,       # row subsampling fraction
    "bagging_freq":      5,         # apply bagging every N iterations
    "min_child_samples": 20,        # min rows in a leaf (anti-overfit)
    "lambda_l1":         0.1,       # L1 regularisation
    "lambda_l2":         1.0,       # L2 regularisation
    "objective":         "regression_l2",
    "metric":            "rmse",
    "random_state":      42,
    "verbosity":         -1,
    "n_jobs":            -1,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class LGBMTrainResult:
    """
    Full training result returned by train().
    Same field layout as XGBTrainResult for interoperability with the stacking
    meta-learner, which only cares about oof_df format.
    """
    fold_results: list[FoldResult]
    oof_df: pd.DataFrame              # player_id, game_id, season, week, y_true, y_pred, fold_idx
    best_params: dict
    run_id: Optional[str]            # MLflow run ID (None if MLflow disabled)
    feature_importances: dict[str, float]  # normalised gain-based importances, sum=1.0
    oof_path: Optional[Path] = None


# ── Optuna hyperparameter search (LightGBM-specific search space) ─────────────

def _run_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_trials: int = 50,
) -> dict:
    """
    Run Optuna on the most-recent fold to find best LightGBM hyperparams.
    Minimises RMSE on val_df.

    Search space is LightGBM-specific — not the same as XGBoost's:
      num_leaves      — leaf-wise tree complexity (15..511)
      min_child_samples — minimum data per leaf, key overfitting guard
      feature_fraction  — column subsampling per iteration (not per tree)
      bagging_fraction  — row subsampling fraction
      bagging_freq      — frequency of bagging application
      lambda_l1 / lambda_l2 — L1/L2 regularisation (different naming from XGB)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr = train_df[features].fillna(-9999).infer_objects(copy=False)
    y_tr = train_df[target_col].values
    X_va = val_df[features].fillna(-9999).infer_objects(copy=False)
    y_va = val_df[target_col].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 511),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            # Fixed non-tuned params
            "objective":    "regression_l2",
            "n_jobs":       -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        y_pred = model.predict(X_va)
        _, rmse = _compute_metrics(y_va, y_pred)
        return rmse

    logger.info("Optuna (LGBM): starting %d trials on most-recent fold…", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best RMSE=%.3f  params=%s", study.best_value, study.best_params)

    best = study.best_params.copy()
    best.update({
        "objective": "regression_l2", "metric": "rmse",
        "random_state": 42, "verbosity": -1, "n_jobs": -1,
    })
    return best


# ── Single-fold training ──────────────────────────────────────────────────────

def _train_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    params: dict,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, FoldResult]:
    """
    Train one LightGBM model on train_df, evaluate on val_df.

    NaN handling: LightGBM can handle NaN natively by routing them to the
    child with higher gain. We rely on this — no imputation is performed.
    Note: LightGBM sklearn API does not natively accept NaN in some versions,
    so we use fillna(-9999) as a sentinel that LightGBM treats as missing
    when learning splits. Sentinel value is documented here so reviewers know
    this is intentional, not a bug.

    Returns:
        (model, oof_rows_df, fold_result)
    """
    X_tr = train_df[features].fillna(-9999).infer_objects(copy=False)
    y_tr = train_df[target_col].values
    X_va = val_df[features].fillna(-9999).infer_objects(copy=False)
    y_va = val_df[target_col].values

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.log_evaluation(period=-1), lgb.early_stopping(50, verbose=False)],
    )
    y_pred = model.predict(X_va)
    mae, rmse = _compute_metrics(y_va, y_pred)

    logger.info(
        "  Fold %d: n_train=%d  n_val=%d  MAE=%.2f  RMSE=%.2f",
        fold_idx, len(train_df), len(val_df), mae, rmse,
    )

    id_cols = [c for c in ["player_id", "game_id", "season", "week", "position"] if c in val_df.columns]
    oof_rows = val_df[id_cols].copy()
    oof_rows["y_true"]   = y_va
    oof_rows["y_pred"]   = y_pred
    oof_rows["fold_idx"] = fold_idx
    oof_rows = oof_rows.reset_index(drop=True)

    fold_result = FoldResult(
        fold_idx=fold_idx,
        train_seasons=sorted(train_df["season"].unique().tolist()),
        val_season=int(val_df["season"].iloc[0]),
        mae=mae,
        rmse=rmse,
        n_train=len(train_df),
        n_val=len(val_df),
    )
    return model, oof_rows, fold_result


# ── Core walk-forward CV ──────────────────────────────────────────────────────

def _walk_forward_cv(
    df: pd.DataFrame,
    folds: list[tuple[list[int], int]],
    features: list[str],
    target_col: str,
    params: dict,
) -> tuple[list[FoldResult], pd.DataFrame, lgb.LGBMRegressor]:
    """
    Run walk-forward CV over all folds with the given LightGBM params.

    Enforces temporal ordering with hard AssertionErrors at EACH fold —
    identical constraint to xgb_model._walk_forward_cv:
      assert train_seasons == sorted(train_seasons)
      assert max(train_seasons) < val_season

    Returns:
        (fold_results, oof_df, final_model)
    """
    all_oof: list[pd.DataFrame] = []
    fold_results: list[FoldResult] = []
    final_model: Optional[lgb.LGBMRegressor] = None

    for fold_idx, (train_seasons, val_season) in enumerate(folds):

        # ── TEMPORAL ORDERING GUARD ────────────────────────────────────────
        # Same hard runtime assertions as xgb_model.py.
        # These are not comments — they halt training on data-leakage bugs.
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
        # ── END TEMPORAL GUARD ─────────────────────────────────────────────

        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df   = df[df["season"] == val_season].copy()

        if train_df.empty or val_df.empty:
            logger.warning(
                "Fold %d: skipping — train empty=%s  val empty=%s",
                fold_idx, train_df.empty, val_df.empty,
            )
            continue

        model, oof_rows, fold_result = _train_fold(
            fold_idx, train_df, val_df, features, target_col, params
        )
        all_oof.append(oof_rows)
        fold_results.append(fold_result)
        final_model = model

    oof_df = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    return fold_results, oof_df, final_model


# ── Main train entry point ────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    seasons: list[int],
    target: str = "receiving_yards",
    n_optuna_trials: int = 50,
    position_filter: Optional[str] = "WR",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> LGBMTrainResult:
    """
    Train LightGBM with walk-forward CV, Optuna tuning, and MLflow logging.

    API is identical to ml.xgb_model.train() so the stacking ensemble can
    call both with the same interface.

    Args:
        df:                 Feature matrix DataFrame. Must have all FEATURE_COLS,
                            identity cols, and the target column from TARGET_COL_MAP.
        seasons:            All seasons present in df.
        target:             One of: "receiving_yards", "rushing_yards",
                            "passing_yards", "fantasy_ppr".
        n_optuna_trials:    Optuna trial count. Pass 0 to skip (use defaults).
        position_filter:    If given, filter to this position before training.
        mlflow_tracking_uri: Pass "" to disable MLflow entirely.
        mlflow_experiment:  Defaults to "lgbm_{target}".
        out_dir:            Defaults to ml/oof/.

    Returns:
        LGBMTrainResult (oof_df format identical to XGBTrainResult).
    """
    if target not in TARGET_COL_MAP:
        raise ValueError(
            f"Unknown target '{target}'. Valid: {list(TARGET_COL_MAP)}"
        )
    target_col = TARGET_COL_MAP[target]

    if out_dir is None:
        out_dir = Path(__file__).parent / "oof"

    # ── Data prep ─────────────────────────────────────────────────────────────
    if position_filter:
        df = df[df["position"] == position_filter].copy()
        logger.info("Position filter '%s': %d rows remain", position_filter, len(df))

    df = df[df[target_col].notna()].copy()
    validate_training_cohort(
        df,
        target=target,
        target_col=target_col,
        position_filter=position_filter,
        source="lgbm_model",
    )

    features = [f for f in FEATURE_COLS if f in df.columns]

    logger.info(
        "Training data: %d rows | %d features | target=%s",
        len(df), len(features), target_col,
    )

    seasons_in_data = sorted(df["season"].unique().tolist())
    folds = _make_walk_forward_folds(seasons_in_data)
    logger.info("Walk-forward folds: %d  seasons=%s", len(folds), seasons_in_data)

    data_hash = _data_hash(df, features, target_col)
    manifest_path = write_manifest(
        build_training_manifest(
            df, target=target, position=position_filter or "all", target_col=target_col,
            feature_columns=features,
            source_contract={"learner": "lgbm", "walk_forward": True, "min_snap_pct": 0.34},
        ),
        out_dir / "manifests" / f"lgbm_{target}_{position_filter or 'all'}_{data_hash}.json",
    )

    # ── Optuna on most-recent fold ────────────────────────────────────────────
    if n_optuna_trials > 0:
        last_train_seasons, last_val_season = folds[-1]
        optuna_train = df[df["season"].isin(last_train_seasons)]
        optuna_val   = df[df["season"] == last_val_season]
        best_params = _run_optuna(
            optuna_train, optuna_val, features, target_col, n_optuna_trials
        )
    else:
        logger.info("Optuna disabled (n_trials=0). Using DEFAULT_LGBM_PARAMS.")
        best_params = DEFAULT_LGBM_PARAMS.copy()

    # ── Walk-forward CV with best params ──────────────────────────────────────
    logger.info("Running walk-forward CV with best params…")
    fold_results, oof_df, final_model = _walk_forward_cv(
        df, folds, features, target_col, best_params
    )

    if not fold_results:
        raise RuntimeError(
            "No folds completed — feature matrix may be empty or seasons missing."
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg_mae  = float(np.mean([fr.mae  for fr in fold_results]))
    agg_rmse = float(np.mean([fr.rmse for fr in fold_results]))
    logger.info("Aggregate CV: MAE=%.3f  RMSE=%.3f", agg_mae, agg_rmse)

    # ── Feature importances (gain-based, normalised) ──────────────────────────
    # LightGBM gain-based importance is the total split gain for each feature
    # across all trees. More informative than split-count ('split') because it
    # weights frequently-used but low-gain features less.
    if final_model is not None:
        raw_imp = final_model.booster_.feature_importance(importance_type="gain")
        total   = raw_imp.sum() or 1.0
        feature_importances = {f: float(v / total) for f, v in zip(features, raw_imp)}
    else:
        feature_importances = {f: 0.0 for f in features}

    # ── MLflow logging ────────────────────────────────────────────────────────
    run_id: Optional[str] = None
    oof_path: Optional[Path] = None

    use_mlflow = mlflow_tracking_uri != ""
    if use_mlflow and mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if use_mlflow:
        try:
            import mlflow
            import mlflow.lightgbm as mlflow_lgb

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            exp_name = mlflow_experiment or f"lgbm_{target}"
            mlflow.set_experiment(exp_name)

            with mlflow.start_run() as run:
                run_id = run.info.run_id

                # ── Params ────────────────────────────────────────────────
                mlflow.log_params({
                    "target":           target,
                    "position_filter":  position_filter or "all",
                    "seasons":          str(seasons_in_data),
                    "n_folds":          len(fold_results),
                    "n_optuna_trials":  n_optuna_trials,
                    **{f"p_{k}": v for k, v in best_params.items()
                       if k not in ("objective", "metric", "verbosity", "n_jobs")},
                })

                # ── Aggregate metrics ──────────────────────────────────────
                mlflow.log_metrics({"mae": agg_mae, "rmse": agg_rmse})

                # ── Per-fold metrics ───────────────────────────────────────
                for fr in fold_results:
                    mlflow.log_metrics({
                        f"fold_{fr.fold_idx}_mae":     fr.mae,
                        f"fold_{fr.fold_idx}_rmse":    fr.rmse,
                        f"fold_{fr.fold_idx}_n_train": fr.n_train,
                        f"fold_{fr.fold_idx}_n_val":   fr.n_val,
                    }, step=fr.fold_idx)

                # ── Feature importances ────────────────────────────────────
                mlflow.log_metrics({
                    f"fi_{k}": v for k, v in feature_importances.items()
                })

                # ── Tags ───────────────────────────────────────────────────
                mlflow.set_tags({
                    "training_data_hash": data_hash,
                    "timestamp":          datetime.utcnow().isoformat(),
                })
                mlflow.log_artifact(str(manifest_path), "manifests")

                # ── Model artifact ─────────────────────────────────────────
                if final_model is not None:
                    mlflow_lgb.log_model(final_model, "model")
                    onnx_path = export_onnx(
                        final_model, features, "lgbm", target,
                        position_filter or "all",
                        onnx_dir=out_dir / "onnx",
                    )
                    if onnx_path:
                        mlflow.log_artifact(str(onnx_path), "onnx")

                # ── OOF artifact ───────────────────────────────────────────
                if not oof_df.empty:
                    oof_path = save_oof(oof_df, target, run_id, out_dir,
                                       prefix="lgbm", position=position_filter or "all")
                    mlflow.log_artifact(str(oof_path))

                logger.info("MLflow run logged: experiment=%s  run_id=%s",
                            exp_name, run_id)

        except Exception as exc:
            logger.error("MLflow logging failed (continuing without it): %s", exc)
            # MLflow was configured but unreachable — still save OOF to disk.
            if not oof_df.empty and oof_path is None:
                pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                oof_path = save_oof(oof_df, target, pseudo_run_id, out_dir, prefix="lgbm", position=position_filter or "all")

    elif not oof_df.empty:
        pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        oof_path = save_oof(oof_df, target, pseudo_run_id, out_dir, prefix="lgbm", position=position_filter or "all")

    return LGBMTrainResult(
        fold_results=fold_results,
        oof_df=oof_df,
        best_params=best_params,
        run_id=run_id,
        feature_importances=feature_importances,
        oof_path=oof_path,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM base learner with walk-forward CV.",
    )
    parser.add_argument(
        "--seasons", required=True,
        help="Season range or list: '2018-2024' or '2022 2023 2024'.",
    )
    parser.add_argument(
        "--target", default="receiving_yards",
        choices=list(TARGET_COL_MAP),
    )
    parser.add_argument("--position", default="WR")
    parser.add_argument("--n-trials", type=int, default=50, dest="n_trials")
    parser.add_argument("--out-dir", default="ml/oof", dest="out_dir")
    parser.add_argument("--mlflow-uri", default=None, dest="mlflow_uri")
    parser.add_argument("--no-mlflow", action="store_true", dest="no_mlflow")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-cache", action="store_true", dest="no_cache",
        help="Skip Parquet cache and always query DB directly.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set.")
        raise SystemExit(1)

    seasons  = _parse_seasons(args.seasons)
    # Trainer entrypoint guarantee: an incomplete season must never enter a
    # walk-forward fold. `_parse_seasons` already raises on out-of-range input;
    # this is the explicit assert the pre-Week-1 claim rests on (audit C-28).
    assert_not_fitting_incomplete_season(seasons)
    position = None if args.position.lower() == "all" else args.position
    mlflow_tracking_uri = "" if args.no_mlflow else args.mlflow_uri

    logger.info(
        "Loading feature_matrix: seasons=%s  position=%s  target=%s",
        seasons, position, args.target,
    )
    try:
        from ml.parquet_cache import FeatureCache
        df = FeatureCache().load_or_fetch(
            seasons, db_url, position_filter=position, no_cache=args.no_cache,
        )
    except ImportError:
        df = load_feature_matrix(db_url, seasons, position_filter=position)

    if df.empty:
        logger.error("No data loaded. Run the ETL pipeline first: make ingest")
        raise SystemExit(1)

    result = train(
        df=df,
        seasons=seasons,
        target=args.target,
        n_optuna_trials=args.n_trials,
        position_filter=position,
        mlflow_tracking_uri=mlflow_tracking_uri,
        out_dir=Path(args.out_dir),
    )

    print(f"\n{'═' * 60}")
    print("  LightGBM training complete")
    print(f"{'═' * 60}")
    for fr in result.fold_results:
        print(f"  Fold {fr.fold_idx}  train={fr.train_seasons}  val={fr.val_season}"
              f"  MAE={fr.mae:.2f}  RMSE={fr.rmse:.2f}")
    agg_mae  = np.mean([fr.mae  for fr in result.fold_results])
    agg_rmse = np.mean([fr.rmse for fr in result.fold_results])
    print(f"{'─' * 60}")
    print(f"  Aggregate CV:  MAE={agg_mae:.2f}  RMSE={agg_rmse:.2f}")
    if result.run_id:
        print(f"  MLflow run_id: {result.run_id}")
    if result.oof_path:
        print(f"  OOF saved:     {result.oof_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
