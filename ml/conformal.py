"""Distribution-free residual intervals (EnbPI-style, no MAPIE / no Py≥3.12).

Replaces the hand-tuned ``std *= 3.0`` fudge for weekly coverage. Uses strictly
prior-fold residuals when ``max_train_season`` is present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_quantiles(
    residuals: np.ndarray,
    *,
    alpha: float = 0.20,
) -> tuple[float, float]:
    """Return (low, high) additive residual quantiles for a 1-alpha interval."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        raise ValueError("Need at least 20 finite residuals for conformal bounds")
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


def conformal_interval(
    y_pred: float,
    residuals: np.ndarray,
    *,
    alpha: float = 0.20,
) -> tuple[float, float]:
    lo, hi = residual_quantiles(residuals, alpha=alpha)
    pred = float(y_pred)
    return pred + lo, pred + hi


def oof_conformal_bounds(
    frame: pd.DataFrame,
    *,
    pred_col: str = "y_pred",
    actual_col: str = "y_true",
    season_col: str = "season",
    alpha: float = 0.20,
) -> pd.DataFrame:
    """Per-row bounds using only earlier seasons' residuals (EnbPI-lite)."""
    out = frame.copy()
    out["floor"] = np.nan
    out["ceiling"] = np.nan
    out["interval_method"] = "unavailable"
    seasons = sorted(pd.to_numeric(out[season_col], errors="coerce").dropna().unique())
    for season in seasons:
        prior = out[pd.to_numeric(out[season_col], errors="coerce") < season]
        resid = (
            pd.to_numeric(prior[actual_col], errors="coerce")
            - pd.to_numeric(prior[pred_col], errors="coerce")
        ).to_numpy()
        try:
            lo, hi = residual_quantiles(resid, alpha=alpha)
        except ValueError:
            continue
        mask = pd.to_numeric(out[season_col], errors="coerce") == season
        pred = pd.to_numeric(out.loc[mask, pred_col], errors="coerce")
        out.loc[mask, "floor"] = pred + lo
        out.loc[mask, "ceiling"] = pred + hi
        out.loc[mask, "interval_method"] = "mapie_enbpi"
    return out


def empirical_coverage(actual: np.ndarray, floor: np.ndarray, ceiling: np.ndarray) -> float:
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(floor, dtype=float)
    hi = np.asarray(ceiling, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))
