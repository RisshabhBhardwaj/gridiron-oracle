"""
ml/stacking_ensemble.py

Ridge meta-learner for the 2-layer stacking ensemble.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1 — Base learners (run separately, produce OOF predictions):
  XGBoost  → ml/oof/xgb_{target}_{run_id[:8]}.csv
  LightGBM → ml/oof/lgbm_{target}_{run_id[:8]}.csv
  TFT      → ml/oof/tft_{target}_{run_id[:8]}.csv   (complete, 3-way alignment tested)

Layer 2 — Meta-learner (this file):
  Input:  Aligned matrix of base-learner OOF predictions
  Model:  Ridge regression, alpha tuned via RidgeCV (GCV)
  Output: ml/oof/stack_{target}_{run_id[:8]}.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY RIDGE FOR THE META-LEARNER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ridge regression is the canonical choice for a stacking meta-learner:
  1. Linear combination of base predictions → coefficient = "how much the
     ensemble trusts XGB vs LGBM". Directly interpretable.
  2. L2 regularisation prevents extreme coefficients when base learners are
     correlated (XGB and LGBM are often highly correlated on structured data).
  3. RidgeCV uses Generalized Cross-Validation (GCV) — an O(n) leave-one-out
     estimator — so alpha selection is essentially free computationally.
  4. The Ridge coefficients serve as feature importances for the ensemble,
     logged to MLflow alongside individual base-learner MAE/RMSE deltas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
META-LEARNER WALK-FORWARD CV
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base learners produce OOF predictions indexed by fold_idx. Each fold_idx
corresponds to one held-out season in the base learner's expanding window.

The meta-learner's own walk-forward mirrors this structure:
  If base folds are [0, 1, 2, 3]:
    Meta fold 0: train on base fold 0      → val on base fold 1
    Meta fold 1: train on base folds 0,1   → val on base fold 2
    Meta fold 2: train on base folds 0,1,2 → val on base fold 3

The meta-learner NEVER trains on any (player_id, game_id) pair that appears
in its own validation fold. Hard assertion at every meta fold:
  assert max(train_base_folds) < val_base_fold

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OOF ALIGNMENT (INNER JOIN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base learner OOFs are joined on (player_id, game_id). Only rows present in
ALL base learner OOFs are included (inner join). If a base learner filtered
to a different position or failed to predict some rows, those rows are
excluded from the stacking meta-learner's training data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STACKING IMPROVEMENT CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After meta CV, stacked_mae is compared to min(base_maes). Both are computed
on the same held-out rows (base folds 1..N-1 used as meta val folds).

If stacked_mae > min_base_mae:
  - UserWarning is raised (training is NOT aborted)
  - stacking_improved=False is logged as an MLflow tag (visible in UI)
  - StackResult.stacking_improved = False

Standalone usage:
  python -m ml.stacking_ensemble \\
      --oof ml/oof/xgb_receiving_yards_abc12345.csv \\
            ml/oof/lgbm_receiving_yards_def67890.csv \\
      --target receiving_yards
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.utils import TARGET_COL_MAP, _compute_metrics
from ml.reliability import promotion_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

RIDGE_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
# ElasticNet L1 ratios: 0=Ridge, 0.5=equal L1/L2, 1.0=Lasso
ENET_L1_RATIOS: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ENET_ALPHAS: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]

# Standard OOF column order — must match xgb_model.py / lgbm_model.py.
# "position" is optional (present when base learners include it).
_OOF_STANDARD_COLS = [
    "player_id", "game_id", "season", "week", "position", "y_true", "y_pred", "fold_idx"
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class MetaFoldResult:
    """Metrics for one meta walk-forward fold."""
    meta_fold_idx: int
    train_base_folds: list[int]     # base fold indices used for meta training
    val_base_fold: int              # base fold index used for meta validation
    alpha: float                    # Ridge alpha selected by RidgeCV
    mae: float
    rmse: float
    n_train: int
    n_val: int


@dataclass
class StackResult:
    """Full result returned by stack()."""
    meta_fold_results: list[MetaFoldResult]
    oof_df: pd.DataFrame              # Standard OOF cols + {prefix}_pred columns
    final_alpha: float                # Ridge alpha fitted on the full OOF
    ridge_coefs: dict[str, float]     # {pred_col: coef} — feature importances
    stacked_mae: float                # Mean MAE over meta folds
    stacked_rmse: float
    base_maes: dict[str, float]       # {prefix: MAE} computed on meta val rows
    base_rmses: dict[str, float]
    stacking_improved: bool           # stacked_mae <= min(base_maes)
    run_id: Optional[str]             # MLflow run ID; None if MLflow disabled
    oof_path: Optional[Path]          # Path to saved stacked OOF CSV


# ── OOF loading and alignment ─────────────────────────────────────────────────

def load_and_align_oofs(
    oof_paths: list[Path],
    position_filter: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load OOF CSVs, rename y_pred to {prefix}_pred, inner-join on (player_id, game_id).

    Prefix is extracted from the filename stem before the first underscore:
      "xgb_receiving_yards_abc12345"  → prefix "xgb"
      "lgbm_receiving_yards_def67890" → prefix "lgbm"

    Args:
        oof_paths:       List of OOF CSV file paths.
        position_filter: If given (e.g. "WR"), filter each OOF to that position
                         before aligning. OOF files must have a "position" column.
                         Rows without a position column are kept as-is (backward compat).

    Returns:
        aligned_df — joined DataFrame; identity/meta columns from the first file
        pred_cols  — ["{prefix}_pred", ...] in oof_paths order
        prefixes   — ["xgb", "lgbm", ...] in oof_paths order
    """
    if len(oof_paths) < 2:
        raise ValueError(
            f"Stacking requires ≥2 OOF files, got {len(oof_paths)}. "
            "Run xgb_model.py and lgbm_model.py first to produce OOF predictions."
        )

    _REQUIRED = {"player_id", "game_id", "season", "week", "y_true", "y_pred", "fold_idx"}

    loaded: list[tuple[str, pd.DataFrame]] = []
    seen_prefixes: set[str] = set()

    for path in oof_paths:
        path = Path(path)
        prefix = path.stem.split("_")[0]

        if prefix in seen_prefixes:
            raise ValueError(
                f"Duplicate OOF prefix '{prefix}'. Each base learner must produce a "
                "uniquely-prefixed OOF file (xgb_, lgbm_, tft_, ...)."
            )
        seen_prefixes.add(prefix)

        # Verify SHA-256 sidecar if present — detects silent file corruption.
        sidecar = path.with_suffix(".csv.sha256")
        if sidecar.exists():
            expected = sidecar.read_text().strip()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"OOF file '{path.name}' failed SHA-256 integrity check.\n"
                    f"  Expected: {expected}\n"
                    f"  Actual:   {actual}\n"
                    "Re-run the base learner to regenerate this file."
                )

        df = pd.read_csv(path)
        missing = _REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"OOF file '{path.name}' is missing required columns: {sorted(missing)}"
            )

        # Filter to the target position when requested (position-specific Ridge).
        # OOF files produced by current base-learner code include a "position" column.
        # Older files without it are kept as-is (the Ridge will be position-unaware for them).
        if position_filter and "position" in df.columns:
            before = len(df)
            df = df[df["position"] == position_filter].copy()
            logger.info(
                "  Position filter '%s': %d → %d rows in %s",
                position_filter, before, len(df), path.name,
            )
            if df.empty:
                logger.warning(
                    "OOF file '%s' has no rows for position='%s' — skipping this learner.",
                    path.name, position_filter,
                )
                seen_prefixes.discard(prefix)
                continue

        df = df.rename(columns={"y_pred": f"{prefix}_pred"})
        loaded.append((prefix, df))
        logger.info(
            "Loaded OOF '%s': %d rows, base folds %s",
            path.name, len(df), sorted(df["fold_idx"].unique().tolist()),
        )

    # Inner join all OOFs on (player_id, game_id).
    # Identity / meta columns (season, week, y_true, fold_idx) kept from first file only.
    # If a learner's OOF drops >50% of rows on join (e.g. TFT trained on a single
    # position while trees trained on all positions), that learner is excluded and
    # stacking proceeds with the remaining base learners.
    prefixes = [p for p, _ in loaded]
    _, merged = loaded[0]
    excluded_prefixes: set[str] = set()

    for prefix, df in loaded[1:]:
        n_before = len(merged)
        pred_col = f"{prefix}_pred"
        candidate = merged.merge(
            df[["player_id", "game_id", pred_col]],
            on=["player_id", "game_id"],
            how="inner",
        )
        n_after = len(candidate)
        if n_after < n_before * 0.5:
            logger.warning(
                "Inner join with '%s' OOF dropped >50%% of rows: %d → %d. "
                "EXCLUDING '%s' from stacking to preserve remaining base learners. "
                "Ensure all base learners are trained on the same position filter.",
                prefix, n_before, n_after, prefix,
            )
            excluded_prefixes.add(prefix)
            continue
        merged = candidate

    prefixes = [p for p in prefixes if p not in excluded_prefixes]
    pred_cols = [f"{p}_pred" for p in prefixes]

    if len(prefixes) < 2:
        raise ValueError(
            f"After excluding misaligned learners ({excluded_prefixes}), only "
            f"{len(prefixes)} base learner(s) remain ({prefixes}). "
            "Stacking requires ≥2 aligned base learners."
        )

    # Deduplicate: TFT (and others) may have multiple rows per (player_id, game_id).
    # Keep first occurrence so meta-learner trains on one row per game.
    n_before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["player_id", "game_id"], keep="first").reset_index(drop=True)
    if len(merged) < n_before_dedup:
        logger.info(
            "Deduplicated (player_id, game_id): %d → %d rows",
            n_before_dedup, len(merged),
        )

    # Drop rows with NaN or inf in any base-learner prediction (Ridge requires finite values).
    n_before_clean = len(merged)
    finite_mask = np.isfinite(merged[pred_cols]).all(axis=1)
    merged = merged.loc[finite_mask].reset_index(drop=True)
    if len(merged) < n_before_clean:
        logger.warning(
            "Dropped %d rows with NaN/inf in base-learner predictions (Ridge requires finite values)",
            n_before_clean - len(merged),
        )

    logger.info(
        "Aligned OOF: %d rows | learners=%s | base folds=%s",
        len(merged), prefixes, sorted(merged["fold_idx"].unique().tolist()),
    )
    return merged, pred_cols, prefixes


# ── Meta walk-forward CV ──────────────────────────────────────────────────────

def _meta_walk_forward_cv(
    aligned_df: pd.DataFrame,
    pred_cols: list[str],
) -> tuple[list[MetaFoldResult], pd.DataFrame, Optional[RidgeCV]]:
    """
    Walk-forward CV for the Ridge meta-learner.

    Folds are derived from the base learners' fold_idx values (integers where
    fold 0 = earliest held-out season, fold N-1 = most recent).

    Meta fold i:
      train → rows where fold_idx ∈ {0, ..., i}
      val   → rows where fold_idx == i+1

    This exactly mirrors the base learner expanding-window constraint and
    guarantees the meta-learner NEVER trains on its own validation rows.

    Hard assertion at every meta fold — same pattern as xgb_model.py:
      assert max(train_base_folds) < val_base_fold

    Returns:
        (meta_fold_results, meta_oof_df, last_ridge)
        meta_oof_df has the standard OOF columns + {prefix}_pred base columns.
    """
    base_folds = sorted(aligned_df["fold_idx"].unique().tolist())

    if len(base_folds) < 2:
        raise ValueError(
            f"Meta walk-forward CV requires ≥2 base folds, got: {base_folds}. "
            "Train base learners on ≥3 seasons to produce ≥2 OOF folds."
        )

    meta_fold_results: list[MetaFoldResult] = []
    all_meta_oof: list[pd.DataFrame] = []
    last_ridge: Optional[RidgeCV] = None

    for meta_fold_i in range(len(base_folds) - 1):
        train_base_folds = list(base_folds[: meta_fold_i + 1])
        val_base_fold    = base_folds[meta_fold_i + 1]

        # ── TEMPORAL ORDERING GUARD ────────────────────────────────────────────
        # Identical pattern to xgb_model.py / lgbm_model.py guards.
        # Meta-learner must never see future base-fold data during training.
        assert max(train_base_folds) < val_base_fold, (
            f"Meta fold {meta_fold_i}: val_base_fold={val_base_fold} is NOT strictly "
            f"greater than max(train_base_folds)={max(train_base_folds)}. "
            "This is a meta-learner data-leakage bug — aborting."
        )
        # ── END TEMPORAL GUARD ─────────────────────────────────────────────────

        train_df = aligned_df[aligned_df["fold_idx"].isin(train_base_folds)].copy()
        val_df   = aligned_df[aligned_df["fold_idx"] == val_base_fold].copy()

        # Drop rows with NaN in pred_cols (defensive; load_and_align_oofs already drops)
        train_df = train_df.dropna(subset=pred_cols)
        val_df   = val_df.dropna(subset=pred_cols)

        if train_df.empty or val_df.empty:
            logger.warning(
                "Meta fold %d: skipping (train=%d rows, val=%d rows).",
                meta_fold_i, len(train_df), len(val_df),
            )
            continue

        X_tr = train_df[pred_cols].values
        y_tr = train_df["y_true"].values
        X_va = val_df[pred_cols].values
        y_va = val_df["y_true"].values

        # Ridge (default) or ElasticNet meta-learner
        # ElasticNet can zero out weak base learners (useful when one model is clearly worse)
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(X_tr, y_tr)
        y_pred = ridge.predict(X_va)
        mae, rmse = _compute_metrics(y_va, y_pred)

        logger.info(
            "  Meta fold %d: train_folds=%s  val_fold=%d  alpha=%.4g"
            "  MAE=%.3f  RMSE=%.3f  n_train=%d  n_val=%d",
            meta_fold_i, train_base_folds, val_base_fold,
            ridge.alpha_, mae, rmse, len(train_df), len(val_df),
        )

        meta_oof = val_df[["player_id", "game_id", "season", "week"]].copy()
        meta_oof["y_true"]   = y_va
        meta_oof["y_pred"]   = y_pred
        meta_oof["fold_idx"] = meta_fold_i
        for col in pred_cols:
            meta_oof[col] = val_df[col].values
        meta_oof = meta_oof.reset_index(drop=True)

        meta_fold_results.append(MetaFoldResult(
            meta_fold_idx=meta_fold_i,
            train_base_folds=train_base_folds,
            val_base_fold=val_base_fold,
            alpha=float(ridge.alpha_),
            mae=mae,
            rmse=rmse,
            n_train=len(train_df),
            n_val=len(val_df),
        ))
        all_meta_oof.append(meta_oof)
        last_ridge = ridge

    if not meta_fold_results:
        raise RuntimeError(
            "No meta folds completed. Check that the aligned OOF has sufficient data."
        )

    meta_oof_df = pd.concat(all_meta_oof, ignore_index=True)
    return meta_fold_results, meta_oof_df, last_ridge


# ── Base learner metrics (computed on meta val rows for fair comparison) ───────

def _compute_base_metrics(
    meta_oof_df: pd.DataFrame,
    pred_cols: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute MAE and RMSE for each base learner on the same rows used for
    meta-learner validation (base folds 1..N-1).

    Computing on meta val rows (not the full aligned OOF) gives a fair
    apples-to-apples comparison: stacked_mae and base_mae are both measured
    on the same held-out samples.
    """
    y_true = meta_oof_df["y_true"].values
    maes:  dict[str, float] = {}
    rmses: dict[str, float] = {}

    for col in pred_cols:
        prefix = col.replace("_pred", "")
        mae, rmse = _compute_metrics(y_true, meta_oof_df[col].values)
        maes[prefix]  = mae
        rmses[prefix] = rmse
        logger.info(
            "  Base %-8s  MAE=%.3f  RMSE=%.3f  (on %d meta-val rows)",
            prefix, mae, rmse, len(meta_oof_df),
        )

    return maes, rmses


# ── OOF saving ────────────────────────────────────────────────────────────────

def _save_stack_oof(
    oof_df: pd.DataFrame,
    target: str,
    run_id: str,
    out_dir: Path,
    position: Optional[str] = None,
) -> Path:
    """
    Save stacked OOF CSV.

    Column order: standard OOF columns first, then {prefix}_pred base columns.
    Standard: player_id, game_id, season, week, y_true, y_pred, fold_idx
    Extra:    xgb_pred, lgbm_pred, tft_pred, ... (whatever base learners provided)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    position_tag = f"_{position}" if position else ""
    filename = f"stack_{target}{position_tag}_{run_id[:8]}.csv"
    path = out_dir / filename

    extra_pred_cols = [
        c for c in oof_df.columns
        if c.endswith("_pred") and c != "y_pred"
    ]
    ordered_cols = [c for c in _OOF_STANDARD_COLS + extra_pred_cols if c in oof_df.columns]
    oof_df[ordered_cols].to_csv(path, index=False)

    # Write SHA-256 sidecar so load_and_align_oofs can detect silent corruption.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".csv.sha256").write_text(digest + "\n")

    logger.info("Stacked OOF saved: %s  (%d rows)  sha256=%s…", path, len(oof_df), digest[:12])
    return path


# ── Main entry point ──────────────────────────────────────────────────────────

def stack(
    oof_paths: list[Path],
    target: str = "receiving_yards",
    position_filter: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    out_dir: Optional[Path] = None,
    use_elasticnet: bool = False,
) -> StackResult:
    """
    Train Ridge meta-learner on aligned base-learner OOF predictions.

    Designed to accept 2 base learners now (XGB + LGBM) and 3 later (+ TFT)
    without any changes to this function — just pass the additional OOF path.

    Args:
        oof_paths:           List of OOF CSV file paths — one per base learner.
                             The prefix is inferred from the filename.
        target:              Prediction target key (e.g. "receiving_yards").
                             Used for MLflow experiment name and OOF filename.
        position_filter:     If given (e.g. "WR"), filter OOF data to this
                             position before fitting Ridge. Produces a
                             position-specific coef file:
                             ridge_{target}_{position}_coefs.json.
                             Eliminates cross-position intercept contamination.
        mlflow_tracking_uri: MLflow tracking URI. Pass "" to disable entirely.
                             None → uses $MLFLOW_TRACKING_URI or localhost:5000.
        mlflow_experiment:   MLflow experiment name. Defaults to "stack_{target}".
        out_dir:             Output directory for stacked OOF. Defaults to ml/oof/.

    Returns:
        StackResult — meta fold results, stacked OOF, Ridge coefs, base learner
        comparison metrics, and MLflow run ID.
    """
    if target not in TARGET_COL_MAP:
        raise ValueError(
            f"Unknown target '{target}'. Valid: {list(TARGET_COL_MAP)}"
        )

    if out_dir is None:
        out_dir = Path(__file__).parent / "oof"

    # ── Load and align base learner OOFs ──────────────────────────────────────
    pos_label = f" [position={position_filter}]" if position_filter else ""
    logger.info("Loading and aligning %d OOF files%s…", len(oof_paths), pos_label)
    aligned_df, pred_cols, prefixes = load_and_align_oofs(
        [Path(p) for p in oof_paths],
        position_filter=position_filter,
    )

    # ── Meta walk-forward CV ──────────────────────────────────────────────────
    logger.info("Running meta walk-forward CV…")
    meta_fold_results, meta_oof_df, _ = _meta_walk_forward_cv(aligned_df, pred_cols)

    # ── Aggregate stacked metrics ─────────────────────────────────────────────
    if not meta_fold_results:
        logger.error("No meta fold results generated for target=%s. Aborting.", target)
        return StackResult(
            meta_fold_results=[],
            oof_df=pd.DataFrame(),
            final_alpha=1.0,
            ridge_coefs={},
            stacked_mae=0.0,
            stacked_rmse=0.0,
            base_maes={},
            base_rmses={},
            stacking_improved=False,
            run_id=None,
            oof_path=None,
        )

    stacked_mae  = float(np.mean([mf.mae  for mf in meta_fold_results]))
    stacked_rmse = float(np.mean([mf.rmse for mf in meta_fold_results]))
    logger.info("Stacked CV: MAE=%.3f  RMSE=%.3f", stacked_mae, stacked_rmse)

    # ── Base learner comparison (computed on the SAME meta val rows) ───────────
    logger.info("Computing base learner metrics on meta validation rows…")
    base_maes, base_rmses = _compute_base_metrics(meta_oof_df, pred_cols)

    for prefix, mae in base_maes.items():
        delta = mae - stacked_mae
        logger.info(
            "  %-8s  MAE=%.3f  vs stacked=%.3f  Δ=%.3f (%s)",
            prefix, mae, stacked_mae, delta,
            "stacking improves" if delta > 0 else "stacking does NOT improve",
        )

    # ── Stacking improvement check ────────────────────────────────────────────
    min_base_mae = min(base_maes.values())
    best_base    = min(base_maes, key=base_maes.get)
    stacking_improved = stacked_mae <= min_base_mae
    promotable, promotion_reason = promotion_gate(
        candidate_mae=stacked_mae,
        incumbent_mae=min_base_mae,
        min_improvement_pct=0.0,
    )
    promotion_path = out_dir / f"promotion_{target}_{position_filter or 'all'}.json"
    promotion_path.write_text(json.dumps({
        "target": target,
        "position": position_filter or "all",
        "candidate": "stack",
        "candidate_mae": stacked_mae,
        "incumbent_mae": min_base_mae,
        "promotable": promotable,
        "reason": promotion_reason,
    }, indent=2, sort_keys=True) + "\n")
    if os.environ.get("PRODUCT_MODE", "graceful_fallback") == "artifact_backed" and not promotable:
        raise RuntimeError(
            f"Refusing artifact-backed promotion for {target}/{position_filter or 'all'}: {promotion_reason}"
        )

    if not stacking_improved:
        warnings.warn(
            f"Stacking did not improve on the best base learner: "
            f"stacked_mae={stacked_mae:.4f} > min_base_mae={min_base_mae:.4f} "
            f"(best base: '{best_base}'). "
            "Possible causes: base learners are too correlated, insufficient "
            "training data, or too few meta folds. "
            "stacking_improved=False will be logged to MLflow.",
            UserWarning,
            stacklevel=2,
        )

    # ── Final Ridge on full OOF — production model ──────────────────────
    # Step 1: RidgeCV to select optimal regularization alpha (no sample weights—
    # RidgeCV does not support sample_weight; use unweighted for alpha selection).
    # Step 2: Refit Ridge(alpha=best) WITH time-based sample_weight so the final
    # production model weights recent seasons 2x vs. oldest seasons.
    # This captures distributional shift (rule changes, team changes) more
    # effectively than treating 2019 and 2024 as statistically identical.
    # Drop any remaining NaN rows (defensive; load_and_align_oofs already drops)
    fit_df = aligned_df.dropna(subset=pred_cols)
    X_all = fit_df[pred_cols].values
    y_all = fit_df["y_true"].values
    if use_elasticnet:
        # Walk-forward CV splits for alpha selection (avoids random k-fold on time-series)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(aligned_df["fold_idx"].unique()) - 1)))
        logger.info("Using ElasticNetCV as meta-learner (L1 ratios=%s, walk-forward CV)", ENET_L1_RATIOS)
        final_ridge = ElasticNetCV(
            l1_ratio=ENET_L1_RATIOS,
            alphas=ENET_ALPHAS,
            cv=tscv,
            max_iter=2000,
        )
        final_ridge.fit(X_all, y_all)
    else:
        # Step 1: alpha selection (unweighted)
        alpha_selector = RidgeCV(alphas=RIDGE_ALPHAS)
        alpha_selector.fit(X_all, y_all)
        best_alpha = float(alpha_selector.alpha_)
        logger.info("Time-weighted stacking: selected alpha=%.4g via RidgeCV", best_alpha)

        # Step 2: refit with time-based sample weights
        from sklearn.linear_model import Ridge
        seasons = fit_df["season"].values if "season" in fit_df.columns else np.ones(len(X_all))
        min_s, max_s = seasons.min(), seasons.max()
        if max_s > min_s:
            # Linear scale: oldest season → weight=1.0, most recent → weight=2.0
            sample_weight = 1.0 + (seasons - min_s) / (max_s - min_s)
        else:
            sample_weight = np.ones(len(seasons))
        logger.info(
            "Time-weighted stacking: season range %d–%d, weight range %.2f–%.2f",
            int(min_s), int(max_s), sample_weight.min(), sample_weight.max(),
        )
        final_ridge = Ridge(alpha=best_alpha)
        final_ridge.fit(X_all, y_all, sample_weight=sample_weight)
    final_alpha = float(getattr(final_ridge, 'alpha_', None) or final_ridge.alpha)
    ridge_coefs = {
        col: float(coef)
        for col, coef in zip(pred_cols, final_ridge.coef_)
    }
    logger.info("Final Ridge: alpha=%.4g  coefs=%s", final_alpha, ridge_coefs)

    # ── Persist Ridge coefficients + intercept for inference ─────────────────
    # ml/train.py _load_ridge_coefs() first tries the position-specific file:
    #   ridge_{target}_{position}_coefs.json  (e.g. ridge_passing_yards_QB_coefs.json)
    # then falls back to the position-agnostic:
    #   ridge_{target}_coefs.json
    # Keys: "xgb_pred" → "xgb", "lgbm_pred" → "lgbm", etc. + "intercept".
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_json: dict[str, float] = {
        col.removesuffix("_pred"): coef for col, coef in ridge_coefs.items()
    }
    coef_json["intercept"] = float(final_ridge.intercept_)
    # Position-specific filename when position_filter is set.
    if position_filter:
        coef_filename = f"ridge_{target}_{position_filter}_coefs.json"
    else:
        coef_filename = f"ridge_{target}_coefs.json"
    coef_path = out_dir / coef_filename
    with open(coef_path, "w") as _fh:
        json.dump(coef_json, _fh, indent=2)
    logger.info("Wrote Ridge coefs to %s: %s", coef_path, coef_json)

    # ── MLflow logging ─────────────────────────────────────────────────────────
    run_id: Optional[str] = None
    oof_path: Optional[Path] = None

    use_mlflow = mlflow_tracking_uri != ""
    if use_mlflow and mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

    if use_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            exp_name = mlflow_experiment or f"stack_{target}"
            mlflow.set_experiment(exp_name)

            with mlflow.start_run() as run:
                run_id = run.info.run_id

                # ── Params ────────────────────────────────────────────────────
                mlflow.log_params({
                    "target":          target,
                    "base_learners":   str(prefixes),
                    "n_oof_files":     len(oof_paths),
                    "n_meta_folds":    len(meta_fold_results),
                    "final_alpha":     final_alpha,
                    "ridge_alphas":    str(RIDGE_ALPHAS),
                })

                # ── Stacked metrics ───────────────────────────────────────────
                mlflow.log_metrics({
                    "stacked_mae":  stacked_mae,
                    "stacked_rmse": stacked_rmse,
                })

                # ── Base learner comparison ───────────────────────────────────
                # Logging deltas shows at a glance whether stacking adds value.
                for prefix, mae in base_maes.items():
                    mlflow.log_metrics({
                        f"{prefix}_mae":       mae,
                        f"{prefix}_rmse":      base_rmses[prefix],
                        f"{prefix}_mae_delta": mae - stacked_mae,  # positive = stack wins
                    })

                # ── Per meta fold metrics ─────────────────────────────────────
                for mf in meta_fold_results:
                    mlflow.log_metrics(
                        {
                            f"meta_fold_{mf.meta_fold_idx}_mae":   mf.mae,
                            f"meta_fold_{mf.meta_fold_idx}_rmse":  mf.rmse,
                            f"meta_fold_{mf.meta_fold_idx}_alpha": mf.alpha,
                        },
                        step=mf.meta_fold_idx,
                    )

                # ── Ridge coefficients = feature importances ──────────────────
                mlflow.log_metrics({
                    f"coef_{col}": coef for col, coef in ridge_coefs.items()
                })

                # ── Tags ──────────────────────────────────────────────────────
                # SHA-256 over sorted OOF paths + their last-modified timestamps.
                # Captures both which files were used and whether their contents
                # changed since the last run, without reading full file contents.
                _sorted_paths = sorted(str(Path(p).resolve()) for p in oof_paths)
                _hash_input = "\n".join(
                    f"{p}:{os.path.getmtime(p)}" for p in _sorted_paths
                ).encode()
                training_data_hash = hashlib.sha256(_hash_input).hexdigest()

                mlflow.set_tags({
                    "stacking_improved":  str(stacking_improved),
                    "timestamp":          datetime.utcnow().isoformat(),
                    "min_base_mae":       f"{min_base_mae:.6f}",
                    "best_base_learner":  best_base,
                    "training_data_hash": training_data_hash,
                })

                # ── OOF artifact ──────────────────────────────────────────────
                if not meta_oof_df.empty:
                    oof_path = _save_stack_oof(meta_oof_df, target, run_id, out_dir, position_filter)
                    mlflow.log_artifact(str(oof_path))
                    mlflow.log_artifact(str(promotion_path), "promotion")

                logger.info(
                    "MLflow run logged: experiment=%s  run_id=%s",
                    exp_name, run_id,
                )

        except Exception as exc:
            logger.error("MLflow logging failed (continuing without it): %s", exc)

    elif not meta_oof_df.empty:
        pseudo_run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        oof_path = _save_stack_oof(meta_oof_df, target, pseudo_run_id, out_dir, position_filter)

    return StackResult(
        meta_fold_results=meta_fold_results,
        oof_df=meta_oof_df,
        final_alpha=final_alpha,
        ridge_coefs=ridge_coefs,
        stacked_mae=stacked_mae,
        stacked_rmse=stacked_rmse,
        base_maes=base_maes,
        base_rmses=base_rmses,
        stacking_improved=stacking_improved,
        run_id=run_id,
        oof_path=oof_path,
    )


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _discover_oof_files(oof_dir: Path, target: str) -> list[Path]:
    """
    Auto-discover base-learner OOF files in *oof_dir* for *target*.

    Handles per-position training: if XGB (or LGB/TFT) was run once per
    position (WR/RB/TE/QB), each run writes a separate CSV with the same
    prefix pattern.  This function concatenates all matching CSVs per prefix
    into a single combined file so the meta-learner sees all positions.

    Returns:
        List of combined-CSV Paths — one per discovered prefix (xgb/lgbm/tft).
        Raises SystemExit if fewer than 2 prefixes are found.
    """
    import glob as _glob

    prefixes = ["xgb", "lgbm", "catboost", "tft"]
    combined_paths: list[Path] = []

    for prefix in prefixes:
        pattern = str(oof_dir / f"{prefix}_{target}_*.csv")
        files = sorted(_glob.glob(pattern))
        if not files:
            logger.debug("No %s OOF files found for target=%s — skipping.", prefix, target)
            continue

        if len(files) == 1:
            combined_paths.append(Path(files[0]))
            logger.info("Found 1 OOF file for prefix=%s: %s", prefix, files[0])
        else:
            # Multiple files = per-position runs and/or retrain stamps.
            # Sort by mtime ascending, then keep='last' so newer OOFs win on
            # (player_id, game_id, fold_idx) — avoids collapsed stale folds
            # poisoning a restack after retrain.
            paths = sorted((Path(f) for f in files), key=lambda p: p.stat().st_mtime)
            # Prefer position-tagged dated runs over stale *combined.csv blobs.
            paths = [p for p in paths if "combined" not in p.name] or paths
            dfs = [pd.read_csv(p) for p in paths]
            combined = pd.concat(dfs, ignore_index=True).drop_duplicates(
                subset=["player_id", "game_id", "fold_idx"],
                keep="last",
            )
            combined_path = oof_dir / f"{prefix}_{target}_combined.csv"
            combined.to_csv(combined_path, index=False)
            logger.info(
                "Combined %d OOF files for prefix=%s → %s  (%d rows) "
                "(newest wins: %s)",
                len(paths), prefix, combined_path, len(combined),
                paths[-1].name if paths else "?",
            )
            combined_paths.append(combined_path)

    if len(combined_paths) < 2:
        logger.error(
            "Need ≥2 base-learner OOF files in %s for target=%s. "
            "Run xgb_model.py and lgbm_model.py first.",
            oof_dir, target,
        )
        raise SystemExit(1)

    return combined_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stack base-learner OOF predictions with a Ridge meta-learner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Explicit file paths:\n"
            "  python -m ml.stacking_ensemble \\\n"
            "      --oof ml/oof/xgb_receiving_yards_abc12345.csv \\\n"
            "            ml/oof/lgbm_receiving_yards_def67890.csv \\\n"
            "      --target receiving_yards\n\n"
            "  # Auto-discover (supports per-position training runs):\n"
            "  python -m ml.stacking_ensemble \\\n"
            "      --oof-dir ml/oof --target receiving_yards"
        ),
    )
    parser.add_argument(
        "--oof", nargs="+", default=None,
        help="Explicit paths to base learner OOF CSVs (one per learner). "
             "Mutually exclusive with --oof-dir.",
    )
    parser.add_argument(
        "--oof-dir", default=None, dest="oof_dir",
        help="Directory to auto-discover OOF files for --target. "
             "Concatenates per-position runs automatically. "
             "Mutually exclusive with --oof.",
    )
    parser.add_argument(
        "--target", default="receiving_yards", choices=list(TARGET_COL_MAP),
    )
    parser.add_argument(
        "--position", default=None,
        help="Train a position-specific Ridge (e.g. WR, QB, RB, TE). "
             "Filters OOF rows to this position before fitting. "
             "Writes ridge_{target}_{position}_coefs.json. "
             "Pass 'all' or omit for cross-position (legacy behaviour).",
    )
    parser.add_argument("--out-dir", default="ml/oof", dest="out_dir")
    parser.add_argument("--mlflow-uri", default=None, dest="mlflow_uri")
    parser.add_argument("--no-mlflow", action="store_true", dest="no_mlflow")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated base-learner prefixes to drop (e.g. tft,xgb). "
             "Phase 5: fantasy_ppr kill TFT; QB/RB also drop XGB on short history.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.oof and args.oof_dir:
        parser.error("--oof and --oof-dir are mutually exclusive.")
    if not args.oof and not args.oof_dir:
        parser.error("Provide --oof (explicit paths) or --oof-dir (auto-discover).")

    exclude = {p.strip().lower() for p in args.exclude.split(",") if p.strip()}

    if args.oof_dir:
        oof_paths = _discover_oof_files(Path(args.oof_dir), args.target)
    else:
        oof_paths = [Path(p) for p in args.oof]
        for p in oof_paths:
            if not p.exists():
                logger.error("OOF file not found: %s", p)
                raise SystemExit(1)

    if exclude:
        kept = []
        for p in oof_paths:
            prefix = p.name.split("_", 1)[0].lower()
            if prefix in exclude:
                logger.info("Excluding %s (--exclude %s)", p.name, ",".join(sorted(exclude)))
                continue
            kept.append(p)
        oof_paths = kept
        if len(oof_paths) < 2:
            logger.error("After --exclude, need ≥2 OOF files; got %d", len(oof_paths))
            raise SystemExit(1)

    mlflow_tracking_uri = "" if args.no_mlflow else args.mlflow_uri
    position_filter = None if (not args.position or args.position.lower() == "all") else args.position

    result = stack(
        oof_paths=oof_paths,
        target=args.target,
        position_filter=position_filter,
        mlflow_tracking_uri=mlflow_tracking_uri,
        out_dir=Path(args.out_dir),
    )

    print(f"\n{'═' * 70}")
    print(f"  Stacking Ensemble — {args.target}")
    print(f"{'═' * 70}")
    if not result.meta_fold_results:
        print(f"  FAILED to stack {args.target}: Not enough valid base folds mapping.")
        print(f"{'═' * 70}\n")
        return

    print("  Base learner performance (on meta validation rows):")
    for prefix, mae in result.base_maes.items():
        rmse = result.base_rmses[prefix]
        print(f"    {prefix:<10} MAE={mae:.3f}  RMSE={rmse:.3f}")
    print(f"{'─' * 70}")
    print(f"  Stacked     MAE={result.stacked_mae:.3f}  RMSE={result.stacked_rmse:.3f}")
    print(f"  Improved over best base: {result.stacking_improved}")
    print(f"  Final Ridge alpha: {result.final_alpha}")
    print(f"  Ridge coefs: { {k: round(v, 4) for k, v in result.ridge_coefs.items()} }")
    print(f"{'─' * 70}")
    for mf in result.meta_fold_results:
        print(
            f"  Meta fold {mf.meta_fold_idx}: train_folds={mf.train_base_folds} "
            f"val_fold={mf.val_base_fold}  MAE={mf.mae:.3f}  alpha={mf.alpha:.4g}"
        )
    if result.run_id:
        print(f"  MLflow run_id: {result.run_id}")
    if result.oof_path:
        print(f"  Stacked OOF:   {result.oof_path}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
