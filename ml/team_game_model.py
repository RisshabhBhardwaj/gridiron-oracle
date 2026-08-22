"""
ml/team_game_model.py

Phase 4 (Coherent Prediction Hierarchy) — team-game outcome models.

Predicts, per team per game, from Elo, coach-fingerprinted neutral-script
tendency, rest, and venue — all knowable strictly before kickoff:
  - points   (team's own score)
  - plays    (team's own total offensive plays)
  - yards    (team's own total offensive yards)
  - pass_rate (team's own pass_attempts / total_plays)
  - win_probability — NOT independently fit. Derived from the points model's
    own predictions on both sides of a game (see derive_win_probability),
    so it's arithmetically consistent with the points number rather than a
    second opinion that can disagree with it — the same "coherent by
    construction" principle the whole program is built around.

Must run without Vegas by design (only ~40% of 2026 games have
spread_line/total_line). Vegas lines are evaluated as a separate
comparison for the points target only — compare_to_vegas's implied-score
formula doesn't have an obvious analog for plays/yards/pass_rate, and
fabricating one would be worse than not comparing.

Ridge over LGBM/CatBoost — 3,920 rows is not enough to justify a tree
ensemble (ml/model_floor.py's philosophy, same call here).

Usage:
    python -m ml.team_game_model --holdout-season 2025 --target points
    python -m ml.team_game_model --holdout-season 2025 --target plays
    python -m ml.team_game_model --holdout-season 2025 --win-probability
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
from scipy.stats import norm
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

# "plays" (total_plays) deliberately excluded: on 2025 holdout its Ridge MAE
# (6.897) is WORSE than the training-mean constant baseline (6.877) — this
# feature set carries no signal for play count, likely because plays run is
# driven by in-game dynamics (turnovers, OT, blowout tempo) that nothing
# pre-kickoff can see. Serving a number that loses to a constant would be
# the same defect the Vegas-comparison bug was: an authoritative-looking
# figure with no real information behind it. Add it back once a feature set
# actually beats the constant on held-out data.
TARGET_COLS: dict[str, str] = {
    "points": "points",
    "yards": "total_yards",
    "pass_rate": "pass_rate",
}


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


@dataclass
class WinProbabilityResult:
    holdout_season: int
    n_holdout_games: int
    residual_std: float
    brier_score: float
    log_loss: float
    accuracy: float
    calibration_buckets: list[dict]
    created_at: str


def _calibration_buckets(y_true: np.ndarray, y_pred: np.ndarray, n_buckets: int = 5) -> list[dict]:
    """
    Calibration, not just point error: within each predicted bucket, does
    the REALIZED average match the PREDICTED average? A model can have a
    low MAE while being systematically over/under-confident at the tails —
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


def _prepare_xy(
    train_df: pd.DataFrame, other_df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill NaNs (e.g. prior_coach_pass_rate for a coach's first tracked week)
    with TRAINING-frame medians only. Filling with the validation/holdout
    frame's own medians would let an imputation statistic peek at data the
    model is being scored against — a real, if mild, leak.
    """
    train_df = train_df[train_df[target_col].notna()]
    fill_values = train_df[FEATURE_COLS].median(numeric_only=True)
    X_train = _prepare_x(train_df, fill_values)
    X_other = _prepare_x(other_df, fill_values)
    y_train = train_df[target_col].astype(float).values
    return X_train, X_other, y_train


def compare_to_vegas(holdout_df: pd.DataFrame, y_pred: np.ndarray, target_col: str) -> Optional[dict]:
    """
    Implied team score from spread/total vs actual, on the subset of
    holdout games that actually carry a Vegas line. total_line is the
    combined game total; spread_line is the home team's line (negative =
    home favored). Implied home score = (total - spread) / 2.

    Points only — plays/yards/pass_rate have no analogous well-defined
    implied value from a spread/total pair.
    """
    if target_col != "points":
        return None
    lined = holdout_df.copy()
    lined["y_pred"] = y_pred
    lined = lined[lined["spread_line"].notna() & lined["total_line"].notna()]
    if lined.empty:
        return None
    home_spread = np.where(lined["is_home"] == 1, lined["spread_line"], -lined["spread_line"])
    implied = (lined["total_line"].values - home_spread) / 2.0
    model_mae, model_rmse = _compute_metrics(lined[target_col].values, lined["y_pred"].values)
    vegas_mae, vegas_rmse = _compute_metrics(lined[target_col].values, implied)
    return {
        "n_lined_rows": int(len(lined)),
        "model_mae": model_mae, "model_rmse": model_rmse,
        "vegas_implied_mae": vegas_mae, "vegas_implied_rmse": vegas_rmse,
    }


def train(
    holdout_season: int,
    target: str = "points",
    alpha: float = 10.0,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
    min_season: int = 2019,
    df: Optional[pd.DataFrame] = None,
) -> TeamGameTrainResult:
    if target not in TARGET_COLS:
        raise ValueError(f"Unknown target {target!r}. Known: {sorted(TARGET_COLS)}")
    target_col = TARGET_COLS[target]

    seasons = list(range(min_season, holdout_season + 1))
    if df is None:
        df = build_team_game_frame(database_url, seasons)
    if df.empty:
        raise ValueError(f"No team-game rows built for seasons={seasons}")

    folds = _make_walk_forward_folds(sorted(df["season"].unique().tolist()))
    fold_results: list[FoldResult] = []
    for fold_idx, (train_seasons, val_season) in enumerate(folds):
        assert train_seasons == sorted(train_seasons)
        assert max(train_seasons) < val_season
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[(df["season"] == val_season) & df[target_col].notna()]
        if train_df.empty or val_df.empty:
            continue
        X_tr, X_va, y_tr = _prepare_xy(train_df, val_df, target_col)
        y_va = val_df[target_col].astype(float).values
        model = Ridge(alpha=alpha).fit(X_tr, y_tr)
        mae, rmse = _compute_metrics(y_va, model.predict(X_va))
        fold_results.append(FoldResult(
            fold_idx=fold_idx, train_seasons=train_seasons, val_season=val_season,
            mae=mae, rmse=rmse, n_train=len(train_df), n_val=len(val_df),
        ))
        logger.info("Fold %d [%s]: train=%s val=%d MAE=%.2f RMSE=%.2f",
                    fold_idx, target, train_seasons, val_season, mae, rmse)

    train_seasons_final = [s for s in seasons if s < holdout_season]
    train_df = df[df["season"].isin(train_seasons_final)]
    holdout_df = df[(df["season"] == holdout_season) & df[target_col].notna()]
    if train_df.empty or holdout_df.empty:
        raise ValueError(f"Insufficient data to hold out season={holdout_season} for target={target}")

    X_tr, X_ho, y_tr = _prepare_xy(train_df, holdout_df, target_col)
    y_ho = holdout_df[target_col].astype(float).values
    final_model = Ridge(alpha=alpha).fit(X_tr, y_tr)
    y_pred = final_model.predict(X_ho)
    holdout_mae, holdout_rmse = _compute_metrics(y_ho, y_pred)
    vegas = compare_to_vegas(holdout_df, y_pred, target_col)

    # A model that can't beat "always predict the training-set mean" carries
    # no signal — this is the floor any "beats Vegas" claim must clear first.
    constant_pred = np.full_like(y_ho, fill_value=float(y_tr.mean()))
    constant_mae, constant_rmse = _compute_metrics(y_ho, constant_pred)

    calibration = _calibration_buckets(y_ho, y_pred)

    result = TeamGameTrainResult(
        holdout_season=holdout_season, target=target, alpha=alpha,
        n_train=len(train_df), n_holdout=len(holdout_df),
        fold_results=[asdict(f) for f in fold_results],
        holdout_mae=holdout_mae, holdout_rmse=holdout_rmse,
        constant_baseline_mae=constant_mae, constant_baseline_rmse=constant_rmse,
        vegas_comparison=vegas,
        calibration_buckets=calibration,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / f"team_game_{target}_{holdout_season}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    logger.info("Wrote %s", out_path)
    return result


def compute_oof_residual_std(train_df: pd.DataFrame, alpha: float, target_col: str = "points") -> float:
    """
    Pooled out-of-fold residual std across walk-forward folds within
    train_df — the honest input for a margin_std/uncertainty conversion.
    An in-sample residual std (model scored on the same rows it was fit on)
    is always optimistically small and overstates confidence.
    """
    fold_residuals: list[np.ndarray] = []
    seasons = sorted(train_df["season"].unique().tolist())
    for fold_train_seasons, fold_val_season in _make_walk_forward_folds(seasons):
        fold_train_df = train_df[train_df["season"].isin(fold_train_seasons)]
        fold_val_df = train_df[train_df["season"] == fold_val_season]
        if fold_train_df.empty or fold_val_df.empty:
            continue
        fold_fill = fold_train_df[FEATURE_COLS].median(numeric_only=True)
        fold_model = Ridge(alpha=alpha).fit(
            _prepare_x(fold_train_df, fold_fill), fold_train_df[target_col].astype(float).values
        )
        fold_pred = fold_model.predict(_prepare_x(fold_val_df, fold_fill))
        fold_residuals.append(fold_val_df[target_col].astype(float).values - fold_pred)

    if fold_residuals:
        return float(np.std(np.concatenate(fold_residuals)))

    # Not enough training seasons for even one CV fold — fall back to
    # in-sample as a last resort (normal use — 2019..holdout_season-1 —
    # always has multiple folds, so this path is a guard, not the default).
    logger.warning("No walk-forward folds available; falling back to in-sample residual_std")
    fill_values = train_df[FEATURE_COLS].median(numeric_only=True)
    fallback_model = Ridge(alpha=alpha).fit(
        _prepare_x(train_df, fill_values), train_df[target_col].astype(float).values
    )
    return float(np.std(
        train_df[target_col].astype(float).values - fallback_model.predict(_prepare_x(train_df, fill_values))
    ))


def derive_win_probability(
    holdout_season: int,
    alpha: float = 10.0,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
    min_season: int = 2019,
) -> WinProbabilityResult:
    """
    Win probability is NOT its own fit model — it's read off the points
    model's own predictions for both teams in a game, converted through the
    classic point-spread-to-win-probability transform: P(win) =
    Phi(predicted_margin / residual_std).

    residual_std comes from walk-forward OUT-OF-FOLD residuals (the same
    folds train()'s CV loop uses), not in-sample training residuals. A
    model always fits its own training data more closely than new data, so
    an in-sample residual_std understates real uncertainty and makes
    win_probability overconfident — pooling residuals across held-out folds
    is the honest version of "the model's own uncertainty."

    This keeps win_probability arithmetically tied to the points number:
    if the points model says a team is projected to lose by a field goal,
    win_probability is *implied* by that margin, not an independent guess
    that could disagree with it.
    """
    seasons = list(range(min_season, holdout_season + 1))
    df = build_team_game_frame(database_url, seasons)
    if df.empty:
        raise ValueError(f"No team-game rows built for seasons={seasons}")

    train_df = df[df["season"] < holdout_season]
    holdout_df = df[df["season"] == holdout_season]
    if train_df.empty or holdout_df.empty:
        raise ValueError(f"Insufficient data to hold out season={holdout_season}")

    residual_std = compute_oof_residual_std(train_df, alpha, target_col="points")

    # Margin residual std is sqrt(2) larger — two independent prediction
    # errors (team's own, opponent's) combine when both come from this model.
    margin_std = residual_std * np.sqrt(2)

    fill_values = train_df[FEATURE_COLS].median(numeric_only=True)
    X_tr = _prepare_x(train_df, fill_values)
    y_tr = train_df["points"].astype(float).values
    model = Ridge(alpha=alpha).fit(X_tr, y_tr)

    X_ho = _prepare_x(holdout_df, fill_values)
    y_pred = model.predict(X_ho)
    holdout_df = holdout_df.copy()
    holdout_df["pred_points"] = y_pred

    opp_pred = holdout_df[["game_id", "team", "pred_points"]].rename(
        columns={"team": "opponent", "pred_points": "opp_pred_points"}
    )
    paired = holdout_df.merge(opp_pred, on=["game_id", "opponent"], how="inner")

    margin = paired["pred_points"].values - paired["opp_pred_points"].values
    win_prob = norm.cdf(margin / margin_std)
    actual_win = paired["win"].values

    brier = float(np.mean((win_prob - actual_win) ** 2))
    eps = 1e-9
    clipped = np.clip(win_prob, eps, 1 - eps)
    logloss = float(-np.mean(actual_win * np.log(clipped) + (1 - actual_win) * np.log(1 - clipped)))
    accuracy = float(np.mean((win_prob >= 0.5).astype(int) == actual_win))
    calibration = _calibration_buckets(actual_win.astype(float), win_prob)

    result = WinProbabilityResult(
        holdout_season=holdout_season, n_holdout_games=int(len(paired)),
        residual_std=residual_std, brier_score=brier, log_loss=logloss, accuracy=accuracy,
        calibration_buckets=calibration, created_at=datetime.now(timezone.utc).isoformat(),
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / f"win_probability_{holdout_season}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    logger.info("Wrote %s", out_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--target", choices=sorted(TARGET_COLS), default="points")
    parser.add_argument("--win-probability", action="store_true")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.win_probability:
        result = derive_win_probability(args.holdout_season, alpha=args.alpha, database_url=args.database_url)
    else:
        result = train(args.holdout_season, target=args.target, alpha=args.alpha, database_url=args.database_url)
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
