"""
ml/bucket_attribution.py

Per-row bucket attributions of a served projection (decision 0008 item 6).

A served OOF projection is ``y = a + sum_l c_l * p_l`` where ``p_l`` is a base
learner's prediction and ``(a, c_l)`` are the Ridge meta-learner's intercept
and weights for that row's meta fold. LightGBM and CatBoost both expose exact
TreeSHAP contributions in the output space (identity link, RMSE / L2
objectives), so for each learner

    p_l = base_l + sum_f phi_{l,f}

and, grouping features into the Settings buckets (``ml/bucket_weights.py``),

    y = (a + sum_l c_l * base_l) + sum_g (sum_l c_l * phi_{l,g})

The first term is stored as ``base`` and each bracketed sum as the bucket's
attribution. The identity is exact up to float error, and
``max_residual`` reports how large that error was. A row set whose residual
exceeds ``ATTRIBUTION_TOLERANCE`` is rejected, never stored.

Nothing here retrains or changes a prediction. The attributions are written
next to each OOF so that the rerun stores them with the numbers they explain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ml.bucket_weights import BUCKET_KEYS, UNWEIGHTED_KEY, bucket_of

#: Stored columns, in order. ``base`` is the blended expected value; the rest
#: are bucket sums. ``unweighted`` is never rescaled but is kept so each row
#: can be checked to add up to the served value.
ATTR_COLUMNS: tuple[str, ...] = ("base", *BUCKET_KEYS, UNWEIGHTED_KEY)

#: Largest allowed |base + sum(buckets) - served value|, in stat units. Stats
#: are counts or yards (0..~600), so 1e-4 is far below anything a user sees
#: while still above float64 SHAP rounding for 500-tree models.
ATTRIBUTION_TOLERANCE: float = 1e-4

#: Identity columns that key an attribution row to its OOF row.
KEY_COLUMNS: tuple[str, ...] = ("player_id", "game_id", "season", "week")


class AttributionError(ValueError):
    """Raised when attributions cannot be computed or do not add up."""


def group_contributions(contrib: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    """Collapse per-feature SHAP values into the Settings buckets.

    ``contrib`` has shape ``(n_rows, len(features) + 1)``; the last column is
    the model's expected value (TreeSHAP convention for both LightGBM
    ``pred_contrib`` and CatBoost ``ShapValues``). Raises ``KeyError`` for a
    feature no bucket claims, so a new feature cannot slip in unattributed.
    """
    contrib = np.asarray(contrib, dtype=float)
    if contrib.ndim != 2 or contrib.shape[1] != len(features) + 1:
        raise AttributionError(
            f"contributions have shape {contrib.shape}; expected "
            f"(n, {len(features) + 1}) for {len(features)} features plus the expected value"
        )
    out = {"base": contrib[:, -1]}
    for key in (*BUCKET_KEYS, UNWEIGHTED_KEY):
        out[key] = np.zeros(contrib.shape[0], dtype=float)
    for index, feature in enumerate(features):
        out[bucket_of(feature)] += contrib[:, index]
    return pd.DataFrame(out, columns=list(ATTR_COLUMNS))


def lgbm_bucket_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Bucket attributions for a fitted ``lgb.LGBMRegressor`` on ``X``.

    ``X`` must be exactly the frame used for ``model.predict`` (missing values
    already replaced by the training sentinel).
    """
    contrib = model.predict(X, pred_contrib=True)
    return group_contributions(np.asarray(contrib), list(X.columns))


def catboost_bucket_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Bucket attributions for a fitted ``CatBoostRegressor`` on ``X``."""
    from catboost import Pool

    contrib = model.get_feature_importance(Pool(X.values), type="ShapValues")
    return group_contributions(np.asarray(contrib), list(X.columns))


def with_keys(attr: pd.DataFrame, oof_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the OOF row's identity columns (and its prediction) to ``attr``."""
    keys = [c for c in KEY_COLUMNS if c in oof_rows.columns]
    if len(attr) != len(oof_rows):
        raise AttributionError(
            f"attribution rows ({len(attr)}) != OOF rows ({len(oof_rows)})"
        )
    out = oof_rows[keys].reset_index(drop=True).copy()
    out["y_pred"] = oof_rows["y_pred"].to_numpy(dtype=float)
    for column in ATTR_COLUMNS:
        out[column] = attr[column].to_numpy(dtype=float)
    return out


def residual(frame: pd.DataFrame, served: Optional[np.ndarray] = None) -> np.ndarray:
    """``served - (base + sum of every bucket)`` per row."""
    total = frame[list(ATTR_COLUMNS)].to_numpy(dtype=float).sum(axis=1)
    target = frame["y_pred"].to_numpy(dtype=float) if served is None else served
    return np.asarray(target, dtype=float) - total


def max_residual(frame: pd.DataFrame, served: Optional[np.ndarray] = None) -> float:
    res = residual(frame, served)
    return float(np.max(np.abs(res))) if len(res) else 0.0


def assert_additive(frame: pd.DataFrame, *, context: str, served: Optional[np.ndarray] = None) -> float:
    """Raise unless every row adds up to its served value; return the worst residual."""
    worst = max_residual(frame, served)
    if not np.isfinite(worst) or worst > ATTRIBUTION_TOLERANCE:
        raise AttributionError(
            f"{context}: attributions do not add up to the served value "
            f"(max residual {worst:.6g} > {ATTRIBUTION_TOLERANCE}). Refusing to store them."
        )
    return worst


def blend(
    learner_attr: Mapping[str, pd.DataFrame],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Blend per-learner attributions with each row's meta weights.

    ``learner_attr[l]`` has KEY_COLUMNS plus ATTR_COLUMNS for learner ``l``.
    ``meta`` has KEY_COLUMNS, ``meta_intercept`` and ``meta_w_<l>`` per
    learner (the stack OOF's per-row meta-fold Ridge). Returns KEY_COLUMNS
    plus ATTR_COLUMNS for the served value. Rows are matched on
    ``(player_id, game_id)``; every meta row must have every learner.
    """
    merged = meta.copy()
    learners = list(learner_attr)
    for learner in learners:
        weight_col = f"meta_w_{learner}"
        if weight_col not in merged.columns:
            raise AttributionError(f"stack is missing {weight_col}; cannot blend {learner} attributions")
        part = learner_attr[learner][["player_id", "game_id", *ATTR_COLUMNS]].rename(
            columns={c: f"{learner}__{c}" for c in ATTR_COLUMNS}
        )
        if part.duplicated(["player_id", "game_id"]).any():
            raise AttributionError(f"{learner} attributions repeat a (player_id, game_id)")
        before = len(merged)
        merged = merged.merge(part, on=["player_id", "game_id"], how="left", validate="m:1")
        if len(merged) != before:
            raise AttributionError("attribution merge changed the row count")
        if merged[f"{learner}__base"].isna().any():
            missing = int(merged[f"{learner}__base"].isna().sum())
            raise AttributionError(f"{missing} stack rows have no {learner} attribution")

    out = merged[[c for c in KEY_COLUMNS if c in merged.columns]].copy()
    out["y_pred"] = merged["y_pred"].to_numpy(dtype=float)
    intercept = merged["meta_intercept"].to_numpy(dtype=float)
    for column in ATTR_COLUMNS:
        total = np.zeros(len(merged), dtype=float)
        for learner in learners:
            total += merged[f"meta_w_{learner}"].to_numpy(dtype=float) * merged[
                f"{learner}__{column}"
            ].to_numpy(dtype=float)
        out[column] = total + (intercept if column == "base" else 0.0)
    return out.reset_index(drop=True)


def attribution_path_for(oof_path) -> Path:
    """``lgbm_x_WR_ab12.csv`` -> ``attr_lgbm_x_WR_ab12.parquet`` (same directory)."""
    oof_path = Path(oof_path)
    return oof_path.with_name(f"attr_{oof_path.stem}.parquet")


def write_collected_attributions(
    enabled: bool, parts: Sequence[pd.DataFrame], oof_path
) -> Optional[Path]:
    """Write the per-fold attributions a trainer collected, beside its OOF.

    Returns None when attributions were not requested, no OOF file was
    written, or no fold produced any (nothing to explain).
    """
    if not enabled or oof_path is None or not parts:
        return None
    return write_attribution_file(pd.concat(parts, ignore_index=True), oof_path)


def write_attribution_file(frame: pd.DataFrame, oof_path) -> Path:
    """Write per-row attributions beside their OOF CSV and return the path.

    The file is parquet (float64, compressed) so that 32 cells stay far below
    the push gate's object size limit. A temp file plus rename keeps a half
    written file from ever looking complete.
    """
    path = attribution_path_for(oof_path)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path
