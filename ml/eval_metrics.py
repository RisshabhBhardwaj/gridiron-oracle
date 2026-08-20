"""
Target-type-aware evaluation metrics.

MAE alone misleads on low-count events (TDs): predicting near-zero beats a
naive mean when the base rate is ~0.1. Count targets use Poisson deviance;
continuous targets use MAE + pinball; all targets report CRPS when samples exist.
"""

from __future__ import annotations

import numpy as np

from ml.stat_resolution import target_metric_family


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, tau: float = 0.5) -> float:
    """Pinball / quantile loss at level tau (0.5 ≈ MAE for median forecasts)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    delta = actual - predicted
    return float(np.mean(np.maximum(tau * delta, (tau - 1.0) * delta)))


def poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Poisson deviance. Predictions are clipped away from zero.
    Suitable for TD / count targets.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-6, None)
    # 2 * (y log(y/mu) - (y - mu)); define 0 log 0 = 0
    term = np.zeros_like(actual)
    nonzero = actual > 0
    term[nonzero] = actual[nonzero] * np.log(actual[nonzero] / predicted[nonzero])
    return float(np.mean(2.0 * (term - (actual - predicted))))


def crps_from_samples(samples: np.ndarray, actual: float) -> float:
    """Vendored CRPS; do not add the scoringrules package (needs Python ≥ 3.12)."""
    from ml.backtest import compute_crps_single
    return compute_crps_single(np.asarray(samples, dtype=float), float(actual))


def primary_score(stat: str, actual: np.ndarray, predicted: np.ndarray) -> tuple[str, float]:
    """Return (metric_name, value) for the stat's metric family."""
    family = target_metric_family(stat)
    if family == "count":
        return "poisson_deviance", poisson_deviance(actual, predicted)
    return "mae", mae(actual, predicted)
