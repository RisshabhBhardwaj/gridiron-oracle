"""Vendored multiple-testing corrections for promotion decisions.

Do not depend on the 24-star ``purgedcv`` package. Formulas follow Bailey &
López de Prado (Deflated Sharpe Ratio) at the level needed to refuse an
uncorrected selection across many (stat × position × learner × group) trials.
"""

from __future__ import annotations

import math


def effective_n_trials(
    n_stats: int,
    n_positions: int,
    n_learners: int,
    n_feature_groups: int = 1,
    n_alphas: int = 1,
) -> int:
    """Count independent selection trials the gate is optimizing over."""
    product = n_stats * n_positions * n_learners * n_feature_groups * n_alphas
    if product < 1:
        raise ValueError("effective_n_trials factors must all be >= 1")
    return int(product)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the observed Sharpe is a false discovery.

    Returns a p-value in (0, 1). Higher observed Sharpe and more observations
    push it down; more trials push it up. Non-finite inputs raise.
    """
    if n_trials < 1 or n_observations < 2:
        raise ValueError("n_trials >= 1 and n_observations >= 2 required")
    if not math.isfinite(observed_sharpe):
        raise ValueError("observed_sharpe must be finite")

    # Expected max Sharpe under the null of zero true SR, across n_trials.
    # E[max Z] ≈ (1 - γ) Φ^{-1}(1 - 1/n) + γ Φ^{-1}(1 - 1/(n e))
    euler = 0.5772156649015329
    if n_trials == 1:
        expected_max = 0.0
    else:
        from statistics import NormalDist

        dist = NormalDist()
        expected_max = (1.0 - euler) * dist.inv_cdf(1.0 - 1.0 / n_trials) + euler * dist.inv_cdf(
            1.0 - 1.0 / (n_trials * math.e)
        )

    se = math.sqrt(
        (1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
        / (n_observations - 1)
    )
    if se <= 0:
        return 0.0 if observed_sharpe > expected_max else 1.0
    z = (observed_sharpe - expected_max) / se
    from statistics import NormalDist

    return float(1.0 - NormalDist().cdf(z))


def require_dsr_pass(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    *,
    max_pvalue: float = 0.05,
) -> None:
    pvalue = deflated_sharpe_ratio(observed_sharpe, n_trials, n_observations)
    if pvalue > max_pvalue:
        raise ValueError(
            f"Deflated Sharpe p={pvalue:.4f} exceeds {max_pvalue} "
            f"(SR={observed_sharpe:.3f}, trials={n_trials}, n={n_observations})"
        )
