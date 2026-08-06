"""
ml/bayesian_model.py

Public alias for the Bayesian uncertainty layer (Layer 3 in the projection stack).
Full implementation: ml/bayesian_layer.py — read that file for derivations.

ARCHITECTURE POSITION
---------------------
  Layer 1: Kalman filter        → latent player ability estimate + variance
  Layer 2: XGB + LGB + CB + TFT → stacked point forecast (OOF Ridge)
  Layer 3: This layer           → posterior distribution over the forecast
  Layer 4: Monte Carlo          → projection / floor / ceiling / fantasy pts

PROBABILISTIC MODEL
-------------------
  Prior:      mu ~ Normal(stacked_estimate, sqrt(kalman_variance))
  Likelihood: sigma ~ HalfNormal(position_sigma_prior)  [learned from residuals]
  Posterior:  2000 NUTS draws via PyMC (2 chains, 500 tune, 1000 draw)

KEY INVARIANTS
--------------
  - fit() must be called before sample(); raises RuntimeError otherwise.
  - position_sigma_prior is calibrated per (stat, position) from walk-forward
    residuals; passing the wrong position silently biases uncertainty estimates.
  - Returned samples array shape: (n_draws,) — single stat, single player.
  - kalman_variance = 0 collapses the prior to a point mass at stacked_estimate.

KNOWN LIMITATIONS
-----------------
  - Single-stat: joint distribution over (yards, TDs, receptions) requires a
    Gaussian copula (Phase 3 — ml/copula_layer.py).
  - PyMC NUTS is ~30–60 s per (stat, position) pair; not suitable for real-time
    inference. Use the pre-sampled posteriors from train.py at serve time.

Usage:
    from ml.bayesian_model import BayesianModel
    # Equivalent to:
    from ml.bayesian_layer import BayesianLayer
"""

from __future__ import annotations

from ml.bayesian_layer import BayesianLayer as BayesianModel  # noqa: F401

__all__ = ["BayesianModel"]
