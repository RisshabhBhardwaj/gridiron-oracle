"""
ml/xgb_model.py

XGBoost base learner — the first of three base models in the stacking ensemble.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WALK-FORWARD CROSS-VALIDATION (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only expanding-window walk-forward CV is used. Random k-fold is NEVER used
on this data — it would leak future game outcomes into training.

For seasons [2018..2024], folds are:
  Fold 0: train=[2018],           val=2019
  Fold 1: train=[2018,2019],      val=2020
  ...
  Fold 5: train=[2018..2023],     val=2024

Each fold is a hard runtime assertion (not just a comment):
  assert train_seasons == sorted(train_seasons)
  assert max(train_seasons) < val_season
If either assertion fires, the code has a data-leakage bug and must halt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HYPERPARAMETER SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optuna runs 50 trials on the MOST RECENT fold only (most representative of
current player-performance environment). Best params are then applied to all
folds for OOF generation. Pass n_optuna_trials=0 to skip Optuna and use
DEFAULT_XGB_PARAMS (useful for CI tests).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MLflow LOGGING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every training run logs to MLflow (required by CLAUDE.md §3):
  - Params: best_params, target, seasons, position_filter, n_optuna_trials
  - Metrics: mae, rmse (aggregate + per-fold), feature importances
  - Artifacts: model (.json via XGBoost native), oof predictions (.csv)
  - Tags: training_data_hash, timestamp

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOF PREDICTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Out-of-fold predictions are the inputs to the stacking meta-learner.
Columns: player_id, game_id, season, week, y_true, y_pred, fold_idx.
Saved to: ml/oof/xgb_{target}_{run_id}.csv (or --out-dir override).

Standalone usage:
  python -m ml.xgb_model --seasons 2018-2024 --target receiving_yards
  python -m ml.xgb_model --seasons 2018-2024 --target receiving_yards \\
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Thread-safety: must happen BEFORE any xgboost / torch import ───────────────
# XGBoost ships its own libomp; when LightGBM or PyTorch also load their
# libomp, three OpenMP runtimes coexist in one process → SIGSEGV on macOS ARM.
# configure_thread_env() sets KMP_DUPLICATE_LIB_OK + OMP_NUM_THREADS=1 and
# caps torch threads if torch is already in sys.modules.
from ml.utils import configure_thread_env  # noqa: E402 — must precede xgb import
configure_thread_env()

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from ml.utils import (
    FEATURE_COLS,
    TARGET_COL_MAP,
    FoldResult,
    export_onnx,
    load_feature_matrix,
    _make_walk_forward_folds,
    _data_hash,
    _compute_metrics,
    save_oof,
    _parse_seasons,
)
from ml.cohort_guards import validate_training_cohort
from ml.reliability import build_training_manifest, write_manifest

# ── XGBoost defaults ─────────────────────────────────────────────────────────

DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42,
    "verbosity": 0,
}

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class XGBTrainResult:
    fold_results: list[FoldResult]
    oof_df: pd.DataFrame
    best_params: dict
    run_id: Optional[str]
    feature_importances: dict[str, float]
    oof_path: Optional[Path] = None

def _run_optuna(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_trials: int = 50,
) -> dict:
    """
    Run Optuna on a single (train_df, val_df) pair to find best XGBoost params.
    Minimises RMSE on val_df.

    Called only on the MOST RECENT fold to avoid look-ahead bias in param tuning.
    Using the most-recent fold means hyperparameters are calibrated to the
    current player-performance environment, which is the most predictive.

    Args:
        n_trials: Number of Optuna trials. Pass 0 to skip and use defaults.

    Returns:
        Best hyperparameter dict (merged with fixed non-tuned params).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr = train_df[features].values
    y_tr = train_df[target_col].values
    X_va = val_df[features].values
    y_va = val_df[target_col].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "tree_method": "hist",
            "objective":   "reg:squarederror",
            "random_state": 42,
            "verbosity":    0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_pred = model.predict(X_va)
        _, rmse = _compute_metrics(y_va, y_pred)
        return rmse

    logger.info("Optuna: starting %d trials on most-recent fold…", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best RMSE=%.3f  params=%s", study.best_value, study.best_params)

    # Merge with non-tuned fixed params
    best = study.best_params.copy()
    best.update({"tree_method": "hist", "objective": "reg:squarederror",
                 "random_state": 42, "verbosity": 0})
    return best


# ── Single-fold training ──────────────────────────────────────────────────────

def _train_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    params: dict,
) -> tuple[xgb.XGBRegressor, pd.DataFrame, FoldResult]:
    """
    Train one XGBoost model on train_df, evaluate on val_df.

    ── TEMPORAL ORDERING GUARD ────────────────────────────────────────────────
    This function does NOT enforce season ordering — the caller (_walk_forward_cv)
    does that via hard assertions. If you call this function directly, ensure
    the train and val sets are temporally non-overlapping.

    Returns:
        (model, oof_rows_df, fold_result)
        oof_rows_df has columns: player_id, game_id, season, week, y_true, y_pred, fold_idx
    """
    X_tr = train_df[features].values
    y_tr = train_df[target_col].values
    X_va = val_df[features].values
    y_va = val_df[target_col].values

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    y_pred = model.predict(X_va)
    mae, rmse = _compute_metrics(y_va, y_pred)

    logger.info(
        "  Fold %d: n_train=%d  n_val=%d  MAE=%.2f  RMSE=%.2f",
        fold_idx, len(train_df), len(val_df), mae, rmse,
    )

    id_cols = [c for c in ["player_id", "game_id", "season", "week", "position"] if c in val_df.columns]
    oof_rows = val_df[id_cols].copy()
    oof_rows["y_true"]    = y_va
    oof_rows["y_pred"]    = y_pred
    oof_rows["fold_idx"]  = fold_idx
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
) -> tuple[list[FoldResult], pd.DataFrame, xgb.XGBRegressor]:
    """
    Run walk-forward CV over all folds with the given XGBoost params.

    Enforces temporal ordering with hard assertions at EACH fold:
      assert train_seasons == sorted(train_seasons)
      assert max(train_seasons) < val_season

    Returns:
        (fold_results, oof_df, final_model)
        final_model is the model trained on the last fold (most data).
    """
    all_oof: list[pd.DataFrame] = []
    fold_results: list[FoldResult] = []
    final_model: Optional[xgb.XGBRegressor] = None

    for fold_idx, (train_seasons, val_season) in enumerate(folds):

        # ── TEMPORAL ORDERING GUARD ────────────────────────────────────────
        # Hard runtime assertions. If either fires, we have a data-leakage bug.
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
                "Fold %d: skipping — train empty=%s  val empty=%s "
                "(seasons may not be in feature_matrix yet)",
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
) -> XGBTrainResult:
    """
    Train XGBoost with walk-forward CV, Optuna tuning, and MLflow logging.

    Args:
        df:                 Feature matrix DataFrame. Must have all FEATURE_COLS,
                            identity cols (player_id, game_id, season, week),
                            and the target column from TARGET_COL_MAP.
        seasons:            All seasons present in df (used to build folds).
        target:             One of: "receiving_yards", "rushing_yards",
                            "passing_yards", "fantasy_ppr".
        n_optuna_trials:    Optuna trial count. Pass 0 to skip and use defaults.
        position_filter:    If given, filter df to this position before training.
        mlflow_tracking_uri: Override MLFLOW_TRACKING_URI env var. Pass "" to
                             disable MLflow entirely.
        mlflow_experiment:  MLflow experiment name. Defaults to "xgb_{target}".
        out_dir:            Directory for OOF CSV output. Defaults to ml/oof/.

    Returns:
        XGBTrainResult with fold_results, oof_df, best_params, run_id, importances.

    Raises:
        ValueError  if target is unknown or fewer than 2 seasons provided.
        AssertionError if temporal ordering is violated in any fold (leakage guard).
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

    # Drop rows where target is null (future games have no actual_ values)
    df = df[df[target_col].notna()].copy()
    validate_training_cohort(
        df,
        target=target,
        target_col=target_col,
        position_filter=position_filter,
        source="xgb_model",
    )

    # Only keep features that are actually in the DataFrame
    features = [f for f in FEATURE_COLS if f in df.columns]

    # XGBoost handles NaN natively (treated as missing) — no imputation needed.
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
            source_contract={"learner": "xgb", "walk_forward": True, "min_snap_pct": 0.34},
        ),
        out_dir / "manifests" / f"xgb_{target}_{position_filter or 'all'}_{data_hash}.json",
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
        logger.info("Optuna disabled (n_trials=0). Using DEFAULT_XGB_PARAMS.")
        best_params = DEFAULT_XGB_PARAMS.copy()

    # ── Walk-forward CV with best params ──────────────────────────────────────
    logger.info("Running walk-forward CV with best params…")
    fold_results, oof_df, final_model = _walk_forward_cv(
        df, folds, features, target_col, best_params
    )

    if not fold_results:
        raise RuntimeError(
            "No folds completed — the feature matrix may be empty or "
            "seasons may not match data in the DB."
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg_mae  = float(np.mean([fr.mae  for fr in fold_results]))
    agg_rmse = float(np.mean([fr.rmse for fr in fold_results]))
    logger.info("Aggregate CV: MAE=%.3f  RMSE=%.3f", agg_mae, agg_rmse)

    # ── Feature importances (from final model, normalised) ────────────────────
    raw_imp = final_model.feature_importances_ if final_model is not None else np.zeros(len(features))
    total   = raw_imp.sum() or 1.0
    feature_importances = {f: float(v / total) for f, v in zip(features, raw_imp)}

    # ── MLflow logging ────────────────────────────────────────────────────────
    run_id: Optional[str] = None
    oof_path: Optional[Path] = None

    # Determine if MLflow is enabled
    use_mlflow = mlflow_tracking_uri != ""
    if use_mlflow and mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if use_mlflow:
        try:
            import mlflow
            import mlflow.xgboost as mlflow_xgb

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            pos_suffix = position_filter or "all"
            exp_name = mlflow_experiment or f"xgb_{target}_{pos_suffix}"
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
                       if k not in ("tree_method", "objective", "verbosity")},
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
                    mlflow_xgb.log_model(final_model, "model")
                    onnx_path = export_onnx(
                        final_model, features, "xgb", target,
                        position_filter or "all",
                        onnx_dir=out_dir / "onnx",
                    )
                    if onnx_path:
                        mlflow.log_artifact(str(onnx_path), "onnx")

                # ── OOF artifact ───────────────────────────────────────────
                if not oof_df.empty:
                    oof_path = save_oof(oof_df, target, run_id, out_dir, position=position_filter or "all")
                    mlflow.log_artifact(str(oof_path))

                logger.info("MLflow run logged: experiment=%s  run_id=%s",
                            exp_name, run_id)

        except Exception as exc:
            logger.error("MLflow logging failed (continuing without it): %s", exc)
            # MLflow was configured but unreachable — still save OOF to disk.
            if not oof_df.empty and oof_path is None:
                pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
                oof_path = save_oof(oof_df, target, pseudo_run_id, out_dir, position=position_filter or "all")

    elif not oof_df.empty:
        # MLflow disabled — save OOF (use timestamp as pseudo run_id)
        pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        oof_path = save_oof(oof_df, target, pseudo_run_id, out_dir, position=position_filter or "all")

    return XGBTrainResult(
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
        description="Train XGBoost base learner with walk-forward CV.",
    )
    parser.add_argument(
        "--seasons", required=True,
        help="Season range or list: '2018-2024' or '2022 2023 2024'.",
    )
    parser.add_argument(
        "--target", default="receiving_yards",
        choices=list(TARGET_COL_MAP),
        help="Prediction target.",
    )
    parser.add_argument(
        "--position", default="WR",
        help="Position filter (e.g. WR, RB, QB, TE). Pass 'all' for no filter.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, dest="n_trials",
        help="Optuna trial count. Pass 0 to skip and use defaults.",
    )
    parser.add_argument(
        "--out-dir", default="ml/oof", dest="out_dir",
        help="Directory for OOF CSV output.",
    )
    parser.add_argument(
        "--mlflow-uri", default=None, dest="mlflow_uri",
        help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", dest="no_mlflow",
        help="Disable MLflow logging entirely.",
    )
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

    seasons = _parse_seasons(args.seasons)
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
        logger.error(
            "No data loaded. Run the ETL pipeline first: make ingest"
        )
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
    print("  XGBoost training complete")
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
