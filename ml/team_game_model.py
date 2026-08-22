"""
ml/team_game_model.py

Phase 4 (Coherent Prediction Hierarchy) — team-game points model.

First slice of the team-game layer: predicts a team's own points scored in
a game from Elo, coach-fingerprinted neutral-script tendency, rest, and
venue — all knowable strictly before kickoff. Must run without Vegas by
design (only ~40% of 2026 games have spread_line/total_line); Vegas lines
are evaluated as a separate comparison, not baked into the default feature
set. Ridge over LGBM/CatBoost — 3,920 rows is not enough to justify a
tree ensemble (ml/model_floor.py's philosophy, same call here).

Once this points pipeline is verified end-to-end (walk-forward CV, 2025
holdout calibration, Vegas comparison on the lined subset), plays/pass_rate/
yards/win_probability are the same frame with a different target — see
pipeline/team_game_features.build_team_game_frame.

Usage:
    python -m ml.team_game_model --holdout-season 2025
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ml.utils import FoldResult, _compute_metrics, _make_walk_forward_folds
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.team_game_features import build_team_game_frame

logger = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parent / "experiments" / "team_game_model"

# Deliberately excludes spread_line/total_line — the model must produce a
# real number for the ~60% of 2026 games with no line yet. Vegas is compared
# against, not trained on; see compare_to_vegas().
FEATURE_COLS: list[str] = [
    "is_home", "rest", "opp_rest",
    "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
    "prior_coach_pass_rate",
    "is_dome", "is_turf",
]
TARGET_COL = "points"


@dataclass
class TeamGameTrainResult:
    holdout_season: int
    target: str
    alpha: float
    n_train: int
    n_holdout: int
    fold_results: list[dict]
    holdout_mae: float
    holdout_rmse: float
    constant_baseline_mae: float
    constant_baseline_rmse: float
    vegas_comparison: Optional[dict]
    calibration_buckets: list[dict]
    created_at: str


def _calibration_buckets(y_true: np.ndarray, y_pred: np.ndarray, n_buckets: int = 5) -> list[dict]:
    """
    Calibration, not just point error: within each predicted-score bucket,
    does the REALIZED average match the PREDICTED average? A model can have
    a low MAE while being systematically over/under-confident at the tails —
    this is what the plan's "calibration on held-out 2025" verify criterion
    actually asks for, distinct from holdout_mae.
    """
    if len(y_pred) < n_buckets:
        return []
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    buckets = []
    for i, (true_chunk, pred_chunk) in enumerate(
        zip(np.array_split(y_true_sorted, n_buckets), np.array_split(y_pred_sorted, n_buckets))
    ):
        if len(pred_chunk) == 0:
            continue
        buckets.append({
            "bucket": i,
            "n": int(len(pred_chunk)),
            "mean_predicted": float(pred_chunk.mean()),
            "mean_realized": float(true_chunk.mean()),
            "gap": float(true_chunk.mean() - pred_chunk.mean()),
        })
    return buckets


def _prepare_x(df: pd.DataFrame, fill_values: pd.Series) -> np.ndarray:
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    return X.values


def _prepare_xy(train_df: pd.DataFrame, other_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill NaNs (e.g. prior_coach_pass_rate for a coach's first tracked week)
    with TRAINING-frame medians only. Filling with the validation/holdout
    frame's own medians would let an imputation statistic peek at data the
    model is being scored against — a real, if mild, leak.
    """
    fill_values = train_df[FEATURE_COLS].median(numeric_only=True)
    X_train = _prepare_x(train_df, fill_values)
    X_other = _prepare_x(other_df, fill_values)
    y_train = train_df[TARGET_COL].astype(float).values
    return X_train, X_other, y_train


def compare_to_vegas(holdout_df: pd.DataFrame, y_pred: np.ndarray) -> Optional[dict]:
    """
    Implied team score from spread/total vs actual, on the subset of
    holdout games that actually carry a Vegas line. total_line is the
    combined game total; spread_line is the home team's line (negative =
    home favored). Implied home score = (total - spread) / 2.
    """
    lined = holdout_df.copy()
    lined["y_pred"] = y_pred
    lined = lined[lined["spread_line"].notna() & lined["total_line"].notna()]
    if lined.empty:
        return None
    home_spread = np.where(lined["is_home"] == 1, lined["spread_line"], -lined["spread_line"])
    implied = (lined["total_line"].values - home_spread) / 2.0
    model_mae, model_rmse = _compute_metrics(lined[TARGET_COL].values, lined["y_pred"].values)
    vegas_mae, vegas_rmse = _compute_metrics(lined[TARGET_COL].values, implied)
    return {
        "n_lined_rows": int(len(lined)),
        "model_mae": model_mae, "model_rmse": model_rmse,
        "vegas_implied_mae": vegas_mae, "vegas_implied_rmse": vegas_rmse,
    }


def train(
    holdout_season: int,
    alpha: float = 10.0,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
    min_season: int = 2019,
) -> TeamGameTrainResult:
    seasons = list(range(min_season, holdout_season + 1))
    df = build_team_game_frame(database_url, seasons)
    if df.empty:
        raise ValueError(f"No team-game rows built for seasons={seasons}")

    folds = _make_walk_forward_folds(sorted(df["season"].unique().tolist()))
    fold_results: list[FoldResult] = []
    for fold_idx, (train_seasons, val_season) in enumerate(folds):
        assert train_seasons == sorted(train_seasons)
        assert max(train_seasons) < val_season
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        X_tr, X_va, y_tr = _prepare_xy(train_df, val_df)
        y_va = val_df[TARGET_COL].astype(float).values
        model = Ridge(alpha=alpha).fit(X_tr, y_tr)
        mae, rmse = _compute_metrics(y_va, model.predict(X_va))
        fold_results.append(FoldResult(
            fold_idx=fold_idx, train_seasons=train_seasons, val_season=val_season,
            mae=mae, rmse=rmse, n_train=len(train_df), n_val=len(val_df),
        ))
        logger.info("Fold %d: train=%s val=%d MAE=%.2f RMSE=%.2f",
                    fold_idx, train_seasons, val_season, mae, rmse)

    train_seasons_final = [s for s in seasons if s < holdout_season]
    train_df = df[df["season"].isin(train_seasons_final)]
    holdout_df = df[df["season"] == holdout_season]
    if train_df.empty or holdout_df.empty:
        raise ValueError(f"Insufficient data to hold out season={holdout_season}")

    X_tr, X_ho, y_tr = _prepare_xy(train_df, holdout_df)
    y_ho = holdout_df[TARGET_COL].astype(float).values
    final_model = Ridge(alpha=alpha).fit(X_tr, y_tr)
    y_pred = final_model.predict(X_ho)
    holdout_mae, holdout_rmse = _compute_metrics(y_ho, y_pred)
    vegas = compare_to_vegas(holdout_df, y_pred)

    # A model that can't beat "always predict the training-set mean" carries
    # no signal — this is the floor any "beats Vegas" claim must clear first.
    constant_pred = np.full_like(y_ho, fill_value=float(y_tr.mean()))
    constant_mae, constant_rmse = _compute_metrics(y_ho, constant_pred)

    calibration = _calibration_buckets(y_ho, y_pred)

    result = TeamGameTrainResult(
        holdout_season=holdout_season, target=TARGET_COL, alpha=alpha,
        n_train=len(train_df), n_holdout=len(holdout_df),
        fold_results=[asdict(f) for f in fold_results],
        holdout_mae=holdout_mae, holdout_rmse=holdout_rmse,
        constant_baseline_mae=constant_mae, constant_baseline_rmse=constant_rmse,
        vegas_comparison=vegas,
        calibration_buckets=calibration,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / f"team_game_points_{holdout_season}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    logger.info("Wrote %s", out_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = train(args.holdout_season, alpha=args.alpha, database_url=args.database_url)
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
