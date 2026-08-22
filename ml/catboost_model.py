"""
ml/catboost_model.py

CatBoost base learner — the fourth base model in the stacking ensemble.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WALK-FORWARD CV (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expanding-window only. Same hard AssertionError guard at every fold:
  assert train_seasons == sorted(train_seasons)
  assert max(train_seasons) < val_season

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOF FORMAT (identical to others — required for stacking)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Columns: player_id, game_id, season, week, y_true, y_pred, fold_idx
Saved to: ml/oof/catboost_{target}_{run_id[:8]}.csv

Standalone usage:
  python -m ml.catboost_model --seasons 2018-2024 --target receiving_yards
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

from ml.utils import configure_thread_env
configure_thread_env()

import catboost as cb
import numpy as np
import pandas as pd

from ml.utils import (
    FEATURE_COLS,
    MISSING_VALUE_SENTINEL,
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

# ── CatBoost defaults ─────────────────────────────────────────────────────────

DEFAULT_CATBOOST_PARAMS: dict = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "bootstrap_type": "Bernoulli",
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CatBoostTrainResult:
    fold_results: list[FoldResult]
    oof_df: pd.DataFrame
    best_params: dict
    run_id: Optional[str]
    feature_importances: dict[str, float]
    oof_path: Optional[Path] = None


# ── Optuna hyperparameter search ──────────────────────────────────────────────

def _run_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_trials: int = 50,
) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # CatBoost can handle NaNs if configured, but let's be consistent and fill
    X_tr = train_df[features].fillna(MISSING_VALUE_SENTINEL).values
    y_tr = train_df[target_col].values
    X_va = val_df[features].fillna(MISSING_VALUE_SENTINEL).values
    y_va = val_df[target_col].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations":        trial.suggest_int("iterations", 100, 1000),
            "depth":             trial.suggest_int("depth", 4, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "bootstrap_type":    "Bernoulli",
            "loss_function":     "RMSE",
            "random_seed":       42,
            "verbose":           0,
            "thread_count":      -1,
            # nan_mode only applies to real NaN, and X_tr/X_va have none left
            # after fillna(MISSING_VALUE_SENTINEL) above — the sentinel itself
            # is what isolates "missing" here, as an out-of-range real value
            # the trees learn to split around. This param is effectively inert.
            "nan_mode":          "Min",
        }
        model = cb.CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=50,
            verbose=False,
        )
        y_pred = model.predict(X_va)
        _, rmse = _compute_metrics(y_va, y_pred)
        return rmse

    logger.info("Optuna (CatBoost): starting %d trials on most-recent fold…", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best RMSE=%.3f  params=%s", study.best_value, study.best_params)

    best = study.best_params.copy()
    best.update({
        "bootstrap_type": "Bernoulli",
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0,
        "thread_count": -1,
        "nan_mode": "Min",
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
) -> tuple[cb.CatBoostRegressor, pd.DataFrame, FoldResult]:
    
    X_tr = train_df[features].fillna(MISSING_VALUE_SENTINEL).values
    y_tr = train_df[target_col].values
    X_va = val_df[features].fillna(MISSING_VALUE_SENTINEL).values
    y_va = val_df[target_col].values

    model = cb.CatBoostRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=50,
        verbose=False,
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
) -> tuple[list[FoldResult], pd.DataFrame, cb.CatBoostRegressor]:
    all_oof: list[pd.DataFrame] = []
    fold_results: list[FoldResult] = []
    final_model: Optional[cb.CatBoostRegressor] = None

    for fold_idx, (train_seasons, val_season) in enumerate(folds):
        assert train_seasons == sorted(train_seasons)
        assert max(train_seasons) < val_season

        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df   = df[df["season"] == val_season].copy()

        if train_df.empty or val_df.empty:
            logger.warning("Fold %d: skipping — train empty=%s  val empty=%s",
                           fold_idx, train_df.empty, val_df.empty)
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
) -> CatBoostTrainResult:
    if target not in TARGET_COL_MAP:
        raise ValueError(f"Unknown target '{target}'")
    target_col = TARGET_COL_MAP[target]

    if out_dir is None:
        out_dir = Path(__file__).parent / "oof"

    if position_filter:
        df = df[df["position"] == position_filter].copy()

    df = df[df[target_col].notna()].copy()
    validate_training_cohort(
        df,
        target=target,
        target_col=target_col,
        position_filter=position_filter,
        source="catboost_model",
    )
    features = [f for f in FEATURE_COLS if f in df.columns]

    logger.info("Training data: %d rows | %d features | target=%s",
                len(df), len(features), target_col)

    seasons_in_data = sorted(df["season"].unique().tolist())
    folds = _make_walk_forward_folds(seasons_in_data)
    data_hash = _data_hash(df, features, target_col)
    manifest_path = write_manifest(
        build_training_manifest(
            df, target=target, position=position_filter or "all", target_col=target_col,
            feature_columns=features,
            source_contract={"learner": "catboost", "walk_forward": True, "min_snap_pct": 0.34},
        ),
        out_dir / "manifests" / f"catboost_{target}_{position_filter or 'all'}_{data_hash}.json",
    )

    if n_optuna_trials > 0:
        last_train_seasons, last_val_season = folds[-1]
        optuna_train = df[df["season"].isin(last_train_seasons)]
        optuna_val   = df[df["season"] == last_val_season]
        best_params = _run_optuna(
            optuna_train, optuna_val, features, target_col, n_optuna_trials
        )
    else:
        best_params = DEFAULT_CATBOOST_PARAMS.copy()
        best_params["nan_mode"] = "Min"

    logger.info("Running CPU walk-forward CV...")
    fold_results, oof_df, final_model = _walk_forward_cv(
        df, folds, features, target_col, best_params
    )

    if not fold_results:
        raise RuntimeError("No folds completed.")

    agg_mae  = float(np.mean([fr.mae  for fr in fold_results]))
    agg_rmse = float(np.mean([fr.rmse for fr in fold_results]))

    if final_model is not None:
        raw_imp = final_model.get_feature_importance()
        total = raw_imp.sum() or 1.0
        feature_importances = {f: float(v / total) for f, v in zip(features, raw_imp)}
    else:
        feature_importances = {f: 0.0 for f in features}

    run_id: Optional[str] = None
    oof_path: Optional[Path] = None

    use_mlflow = mlflow_tracking_uri != ""
    if use_mlflow and mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:15091")

    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            exp_name = mlflow_experiment or f"catboost_{target}"
            mlflow.set_experiment(exp_name)

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                mlflow.log_params({
                    "target": target,
                    "position_filter": position_filter or "all",
                    **{f"p_{k}": v for k, v in best_params.items()}
                })
                mlflow.log_metrics({"mae": agg_mae, "rmse": agg_rmse})

                for fr in fold_results:
                    mlflow.log_metrics({
                        f"fold_{fr.fold_idx}_mae": fr.mae,
                        f"fold_{fr.fold_idx}_rmse": fr.rmse,
                    }, step=fr.fold_idx)

                mlflow.log_metrics({f"fi_{k}": v for k, v in feature_importances.items()})
                mlflow.set_tags({"training_data_hash": data_hash, "timestamp": datetime.utcnow().isoformat()})
                mlflow.log_artifact(str(manifest_path), "manifests")

                if final_model is not None:
                    import mlflow.catboost
                    mlflow.catboost.log_model(final_model, "model")
                    onnx_path = export_onnx(
                        final_model, features, "catboost", target,
                        position_filter or "all",
                    )
                    if onnx_path:
                        mlflow.log_artifact(str(onnx_path), "onnx")

                if not oof_df.empty:
                    oof_path = save_oof(oof_df, target, run_id, out_dir, prefix="catboost", position=position_filter or "all")
                    mlflow.log_artifact(str(oof_path))
        except Exception as exc:
            logger.error("MLflow logging failed: %s", exc)
            # MLflow was configured but unreachable — still save OOF to disk.
            if not oof_df.empty and oof_path is None:
                pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                oof_path = save_oof(oof_df, target, pseudo_run_id, out_dir, prefix="catboost", position=position_filter or "all")
    elif not oof_df.empty:
        pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        oof_path = save_oof(oof_df, target, pseudo_run_id, out_dir, prefix="catboost", position=position_filter or "all")

    return CatBoostTrainResult(fold_results, oof_df, best_params, run_id, feature_importances, oof_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost base learner.")
    parser.add_argument("--seasons", required=True)
    parser.add_argument("--target", default="receiving_yards", choices=list(TARGET_COL_MAP))
    parser.add_argument("--position", default="WR")
    parser.add_argument("--n-trials", type=int, default=50, dest="n_trials")
    parser.add_argument("--out-dir", default="ml/oof", dest="out_dir")
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
        logger.error("DATABASE_URL not set. Export it before running CatBoost training.")
        raise SystemExit(1)
    seasons  = _parse_seasons(args.seasons)
    # Trainer entrypoint guarantee: an incomplete season must never enter a
    # walk-forward fold. `_parse_seasons` already raises on out-of-range input;
    # this is the explicit assert the pre-Week-1 claim rests on (audit C-28).
    assert_not_fitting_incomplete_season(seasons)
    position = None if args.position.lower() == "all" else args.position
    mlflow_tracking_uri = "" if args.no_mlflow else None

    try:
        from ml.parquet_cache import FeatureCache
        df = FeatureCache().load_or_fetch(
            seasons, db_url, position_filter=position, no_cache=args.no_cache,
        )
    except ImportError:
        df = load_feature_matrix(db_url, seasons, position_filter=position)
    if df.empty:
        sys.exit(1)

    result = train(
        df=df, seasons=seasons, target=args.target,
        n_optuna_trials=args.n_trials, position_filter=position,
        mlflow_tracking_uri=mlflow_tracking_uri, out_dir=Path(args.out_dir)
    )

    agg_mae  = np.mean([fr.mae  for fr in result.fold_results])
    agg_rmse = np.mean([fr.rmse for fr in result.fold_results])
    print(f"CatBoost CV MAE: {agg_mae:.2f} | RMSE: {agg_rmse:.2f}")

if __name__ == "__main__":
    main()
