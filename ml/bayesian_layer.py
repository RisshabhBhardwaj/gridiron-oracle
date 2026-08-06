"""
ml/bayesian_layer.py

Bayesian uncertainty layer — third layer in the projection stack.

ARCHITECTURE POSITION
---------------------
  Layer 1: Kalman filter  → latent player ability estimates per stat
  Layer 2: XGB + LGB + TFT stacking ensemble → point forecast
  Layer 3: This module → posterior distribution over the forecast

PROBABILISTIC MODEL (per position)
-----------------------------------
  Prior on mean:
    mu ~ Normal(stacked_estimate, sqrt(kalman_variance))

  Likelihood spread (learned from historical residuals):
    sigma ~ HalfNormal(position_sigma_prior)

  Predictive distribution:
    yards_pred ~ Normal(mu, sigma)

WHY THIS MODEL
--------------
  * stacked_estimate is the best point forecast from Layer 2; we center
    the prior on it.
  * kalman_variance encodes Kalman filter uncertainty about the player's
    current ability level — wider Kalman spread → wider prior on mu.
  * sigma is a latent scale that the sampler estimates from historical
    residuals via fit(); it captures irreducible game-to-game variance.
  * The result is a full posterior over outcomes, enabling:
      - P(yards > threshold) for over/under pricing
      - Brier score calibration
      - Risk-aware Kelly sizing via the C++ engine

CACHING & PERFORMANCE
---------------------
  PyMC models compile pytensor graphs on first call — expensive (~1s).
  Subsequent calls using pm.set_data() swap parameter values without
  recompilation.

  fit() saves position_sigma_prior and compiles the model.
  posterior_samples() calls pm.set_data() then pm.sample().

  Each position (WR/RB/QB/TE) should use a separate BayesianProjection
  instance so position_sigma_prior is learned per position.

SAMPLER SETTINGS
----------------
  NUTS, 2 chains, 500 tune steps (standard settings for this model size).
  cores=1 avoids multiprocessing overhead inside Docker containers.
  target_accept=0.9 reduces divergences for the nested Normal(mu, sigma).
"""

from __future__ import annotations

import sys
import warnings
from typing import Optional

import numpy as np

# ── Thread-safety: must happen BEFORE pytensor / pymc / torch import ─────────
# pytensor links against OpenMP for multi-threaded ops. When XGBoost and
# LightGBM have already loaded their own libomp, a third OMP runtime from
# pytensor causes a deadlock / SIGSEGV on macOS ARM.
# configure_thread_env() sets KMP_DUPLICATE_LIB_OK + OMP_NUM_THREADS=1.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from ml.utils import configure_thread_env  # noqa: E402 — must precede pytensor
configure_thread_env()

import pytensor
import pymc as pm


# ---------------------------------------------------------------------------
# Module-level pytensor config: disable C compilation (macOS ARM safe)
# ---------------------------------------------------------------------------
# pytensor's C backend fails on macOS ARM (Apple Silicon) when the Xcode
# SDK C++ headers are not on the default clang search path. The numpy/
# python backend is ~2× slower for large models but fully correct. For
# this model (3 scalar nodes) the difference is negligible.
pytensor.config.cxx = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NUTS tuning / draw counts
_TUNE: int = 500
_DRAWS: int = 2_000

# Number of chains; keep at 2 for Docker (avoids spawning too many processes)
_CHAINS: int = 2

# Single-threaded inside container
_CORES: int = 1

# NUTS target acceptance probability — 0.9 reduces divergences for this
# hierarchical-ish model where sigma can get small.
_TARGET_ACCEPT: float = 0.9

# Default position_sigma_prior fallback when fit() hasn't been called
# (cold-start safety, not used if fit() is always called).
_DEFAULT_SIGMA_PRIOR: float = 20.0

# Minimum allowed position_sigma_prior to prevent degenerate distributions
_MIN_SIGMA_PRIOR: float = 1e-3

# KALMAN_VARIANCE_CAP — maximum kv passed to the NCP scale (sqrt(kv)) in NUTS.
#
# Root cause: cold-start players have kalman_variance = 1000.0 (INITIAL_VARIANCE
# from kalman_tracker.py). In the NCP parameterization:
#   mu = stack_est + mu_raw * sqrt(kv)
# This makes sqrt(1000) = 31.6. When position_sigma_prior ≈ 20, the prior on
# mu is WIDER than the prior on sigma, creating funnel geometry that causes
# NUTS divergences even with NCP.
#
# Cap kv at 4 × position_sigma_prior² so that sqrt(kv) ≤ 2 × sigma_prior —
# keeping the mu prior within a 2σ envelope of the observation scale.
# At inference time this is set per-player in posterior_samples().
_KALMAN_VARIANCE_CAP_FACTOR: float = 4.0  # cap = factor × sigma_prior²


# ---------------------------------------------------------------------------
# BayesianProjection
# ---------------------------------------------------------------------------

class BayesianProjection:
    """
    PyMC-based Bayesian uncertainty layer for a single player position.

    Typical usage:
        proj = BayesianProjection()
        proj.fit(position="WR", historical_residuals=residuals_array)

        samples = proj.posterior_samples(
            stacked_estimate=85.0,
            kalman_variance=120.0,
            n_samples=2000,
        )
        stats = proj.summary(samples)
        # stats["mean"], stats["p10"], stats["p50"], stats["p90"], stats["std"]

    One instance per position. Call fit() once; then posterior_samples()
    many times (cheap after first call due to model caching).
    """

    def __init__(self, stat: str = "") -> None:
        self.stat: str = stat                    # target stat (e.g. "rushing_tds"); set by fit()
        self.position_sigma_prior: float = _DEFAULT_SIGMA_PRIOR
        self._model: Optional[pm.Model] = None
        self._idata = None   # last InferenceData (for diagnostics)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        position: str,
        historical_residuals: np.ndarray,
    ) -> "BayesianProjection":
        """
        Estimate position_sigma_prior from historical residuals.

        position_sigma_prior = std(historical_residuals), lower-bounded at
        _MIN_SIGMA_PRIOR to prevent degenerate HalfNormal.

        Args:
            position: Position string e.g. "WR" (stored for reference only).
            historical_residuals: 1-D array of (actual - predicted) errors
                from the stacking ensemble, historically. Must have >= 1
                element (single-element yields std=0, uses _MIN_SIGMA_PRIOR).

        Returns:
            self (for chaining).

        Raises:
            ValueError: if historical_residuals is empty.
        """
        residuals = np.asarray(historical_residuals, dtype=float)
        if residuals.size == 0:
            raise ValueError("historical_residuals must not be empty.")

        self.position = position

        # std(ddof=1) for unbiased estimate; fallback to _DEFAULT_SIGMA_PRIOR
        # when only one observation is available.
        if residuals.size >= 2:
            sigma_est = float(np.std(residuals, ddof=1))
        else:
            sigma_est = _DEFAULT_SIGMA_PRIOR

        self.position_sigma_prior = max(sigma_est, _MIN_SIGMA_PRIOR)

        # Build (but do not yet sample) the compiled model.
        self._build_model(stack_est_init=0.0, kv_init=100.0)

        return self

    # ------------------------------------------------------------------
    # posterior_samples
    # ------------------------------------------------------------------

    def posterior_samples(
        self,
        stacked_estimate: float,
        kalman_variance: float,
        n_samples: int = _DRAWS,
        use_mcmc: bool = False,
    ) -> np.ndarray:
        """
        Draw posterior samples for yards_pred given a stacked estimate and
        Kalman variance.

        By default uses an analytical fast path (100-1000x faster than MCMC).
        The current model has no observed data/likelihood, so MCMC samples
        from the prior — analytically equivalent to:
            mu ~ Normal(stack_est, sqrt(kv))
            sigma ~ |HalfNormal(sigma_prior)|
            yards_pred ~ Normal(mu, sigma)
        Set use_mcmc=True to use PyMC NUTS sampling (for validation or when
        observed data is added in the future).

        Args:
            stacked_estimate: Point forecast from XGB+LGB+TFT stack.
            kalman_variance: Kalman posterior variance for this player/stat.
                Larger variance → wider prior on mu → more uncertainty.
            n_samples: Number of NUTS draws (per chain). Total posterior
                draws = n_samples * _CHAINS. Default 2000 → 4000 draws total.

        Returns:
            np.ndarray of shape (n_samples * _CHAINS,) — flattened posterior
            samples for yards_pred.

        Raises:
            RuntimeError: if fit() has not been called.
        """
        if self.position_sigma_prior == _DEFAULT_SIGMA_PRIOR and self._model is None:
            raise RuntimeError(
                "BayesianProjection.posterior_samples() called before fit(). "
                "Call fit(position, historical_residuals) first."
            )

        # Cap kalman_variance to prevent cold-start funnel geometry.
        kv_cap = _KALMAN_VARIANCE_CAP_FACTOR * (self.position_sigma_prior ** 2)
        kv_safe = float(min(max(kalman_variance, _MIN_SIGMA_PRIOR ** 2), kv_cap))

        # ── Analytical fast path ──────────────────────────────────────────
        # The model has no observed data, so MCMC samples from the prior:
        #   mu ~ Normal(stack_est, sqrt(kv))
        #   sigma ~ |Normal(0, sigma_prior)|  (half-normal)
        #   yards_pred ~ Normal(mu, sigma)
        # This is equivalent to sampling analytically with numpy.
        if not use_mcmc:
            total = n_samples * _CHAINS
            rng = np.random.default_rng()
            mu_samples = rng.normal(stacked_estimate, np.sqrt(kv_safe), size=total)
            sigma_samples = np.abs(rng.normal(0, self.position_sigma_prior, size=total))
            samples = rng.normal(mu_samples, sigma_samples)

            _COUNT_STATS = {
                "rushing_tds", "receiving_tds", "passing_tds",
                "interceptions", "fumbles", "carries", "targets",
                "sacks_taken", "pass_attempts", "completions",
                "receptions",
            }
            if self.stat in _COUNT_STATS:
                samples = np.clip(samples, 0.0, None)
            return samples.astype(float)

        # ── MCMC path (for validation or future models with likelihood) ───
        if self._model is None:
            self._build_model(stack_est_init=0.0, kv_init=100.0)

        # Update parameter values without recompiling.
        with self._model:
            pm.set_data({
                "stack_est": float(stacked_estimate),
                "kv":        kv_safe,
            })

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*divergences.*",
                    category=UserWarning,
                )
                idata = pm.sample(
                    draws=n_samples,
                    tune=_TUNE,
                    chains=_CHAINS,
                    cores=_CORES,
                    target_accept=_TARGET_ACCEPT,
                    progressbar=False,
                    return_inferencedata=True,
                )

        self._idata = idata

        # Flatten chains × draws → 1-D array.
        # idata.posterior["yards_pred"] shape: (chains, draws)
        samples: np.ndarray = idata.posterior["yards_pred"].values.flatten()
        samples = samples.astype(float)

        # Clip count stats at 0: TDs, INTs, fumbles, carries, targets are
        # non-negative by definition. A Normal prior can produce negative draws.
        # Clipping at 0 is the canonical fix without changing the likelihood family.
        _COUNT_STATS = {
            "rushing_tds", "receiving_tds", "passing_tds",
            "interceptions", "fumbles", "carries", "targets",
            "sacks_taken", "pass_attempts", "completions",
            "receptions",
        }
        if self.stat in _COUNT_STATS:
            samples = np.clip(samples, 0.0, None)

        return samples

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary(samples: np.ndarray) -> dict:
        """
        Compute summary statistics from posterior samples.

        Args:
            samples: 1-D array of posterior draws (from posterior_samples()).

        Returns:
            dict with keys:
                "mean"  — posterior mean
                "std"   — posterior standard deviation
                "p10"   — 10th percentile
                "p25"   — 25th percentile
                "p50"   — median
                "p75"   — 75th percentile
                "p90"   — 90th percentile
                "p5"    — 5th percentile
                "p95"   — 95th percentile
                "n"     — number of samples
        """
        arr = np.asarray(samples, dtype=float)
        pcts = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr, ddof=1)),
            "p5":   float(pcts[0]),
            "p10":  float(pcts[1]),
            "p25":  float(pcts[2]),
            "p50":  float(pcts[3]),
            "p75":  float(pcts[4]),
            "p90":  float(pcts[5]),
            "p95":  float(pcts[6]),
            "n":    int(arr.size),
        }

    # ------------------------------------------------------------------
    # rhat_ok
    # ------------------------------------------------------------------

    def rhat_ok(self, threshold: float = 1.1) -> bool:
        """
        Check R-hat convergence diagnostic on last posterior_samples() call.

        R-hat < threshold for all parameters indicates good chain mixing.
        Per Vehtari et al. (2021), threshold of 1.01 is recommended for
        publication; 1.1 is the traditional looser threshold.

        Args:
            threshold: Maximum allowed R-hat. Default 1.1.

        Returns:
            True if all R-hat values < threshold; False otherwise.
            Returns False if posterior_samples() has not been called.
        """
        if self._idata is None:
            return False

        import arviz as az  # optional — only needed for diagnostics
        summary = az.summary(self._idata, var_names=["mu", "sigma", "yards_pred"])
        return bool((summary["r_hat"] < threshold).all())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(
        self,
        stack_est_init: float,
        kv_init: float,
    ) -> None:
        """
        Construct the PyMC model with pm.Data containers.

        pm.Data containers allow pm.set_data() to swap values without
        recompiling the pytensor computation graph — critical for inference
        latency on repeated calls.

        Model:
            mu         ~ Normal(stack_est, sqrt(kv))
            sigma      ~ HalfNormal(position_sigma_prior)
            yards_pred ~ Normal(mu, sigma)
        """
        with pm.Model() as model:
            # Mutable data: updated per player per call via pm.set_data().
            stack_est = pm.Data("stack_est", stack_est_init)
            kv        = pm.Data("kv", kv_init)

            # Non-Centered Parameterization (NCP) to resolve NUTS divergences.
            # We sample from standard normals and manually apply the location and scale.
            
            # Prior on mean: centered at stacked estimate, spread by Kalman variance
            mu_raw = pm.Normal("mu_raw", mu=0.0, sigma=1.0)
            mu = pm.Deterministic("mu", stack_est + mu_raw * pm.math.sqrt(kv))

            # Scale: learned from historical residuals in fit().
            sigma = pm.HalfNormal("sigma", sigma=self.position_sigma_prior)

            # Likelihood / predictive target using NCP
            yards_pred_raw = pm.Normal("yards_pred_raw", mu=0.0, sigma=1.0)
            pm.Deterministic("yards_pred", mu + yards_pred_raw * sigma)

        self._model = model
