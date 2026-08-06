"""
backend/tests/test_bayesian_layer.py

Tests for ml/bayesian_layer.py — BayesianProjection class.

Test Strategy
-------------
We need tests that are:
  1. Fast — NUTS with 500 tune + 2000 draws takes ~5-10s per call.
  2. Robust — MCMC has sampling variance; tests use tolerances.
  3. Deterministic-ish — we set a fixed seed where possible.

Test categories:
  A. Unit tests (no sampling) — fit(), summary(), model structure
  B. Integration tests (sampling) — convergence, calibration, monotonicity
     Marked @pytest.mark.slow so CI can run `pytest -m "not slow"` fast.

Apple Silicon NOTE:
  Run with: export KMP_DUPLICATE_LIB_OK=TRUE && export OMP_NUM_THREADS=1
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

# Ensure ml/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_residuals(n: int = 200, std: float = 25.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, std, size=n)


def _make_projection(**kwargs):
    from ml.bayesian_layer import BayesianProjection
    proj = BayesianProjection()
    return proj


# ---------------------------------------------------------------------------
# A. Unit Tests — no MCMC sampling
# ---------------------------------------------------------------------------

class TestFit:
    """Tests for BayesianProjection.fit()."""

    def test_fit_returns_self(self):
        from ml.bayesian_layer import BayesianProjection
        proj = BayesianProjection()
        result = proj.fit("WR", _make_residuals(50))
        assert result is proj

    def test_sigma_prior_set_from_residuals(self):
        from ml.bayesian_layer import BayesianProjection
        residuals = _make_residuals(200, std=25.0)
        proj = BayesianProjection()
        proj.fit("WR", residuals)
        expected = float(np.std(residuals, ddof=1))
        assert abs(proj.position_sigma_prior - expected) < 0.5

    def test_fit_stores_position(self):
        from ml.bayesian_layer import BayesianProjection
        proj = BayesianProjection()
        proj.fit("QB", _make_residuals(50))
        assert proj.position == "QB"

    def test_fit_builds_model(self):
        from ml.bayesian_layer import BayesianProjection
        proj = BayesianProjection()
        assert proj._model is None
        proj.fit("RB", _make_residuals(50))
        assert proj._model is not None

    def test_fit_empty_residuals_raises(self):
        from ml.bayesian_layer import BayesianProjection
        proj = BayesianProjection()
        with pytest.raises(ValueError, match="empty"):
            proj.fit("WR", np.array([]))

    def test_fit_single_element_uses_default(self):
        from ml.bayesian_layer import BayesianProjection, _DEFAULT_SIGMA_PRIOR
        proj = BayesianProjection()
        proj.fit("WR", np.array([10.0]))
        assert proj.position_sigma_prior == _DEFAULT_SIGMA_PRIOR

    def test_fit_min_sigma_floor(self):
        from ml.bayesian_layer import BayesianProjection, _MIN_SIGMA_PRIOR
        # All identical → std=0 → floor
        proj = BayesianProjection()
        proj.fit("WR", np.array([5.0, 5.0, 5.0]))
        assert proj.position_sigma_prior >= _MIN_SIGMA_PRIOR

    def test_larger_residuals_larger_sigma_prior(self):
        from ml.bayesian_layer import BayesianProjection
        r_small = _make_residuals(200, std=10.0, seed=0)
        r_large = _make_residuals(200, std=40.0, seed=1)
        p_small = BayesianProjection(); p_small.fit("WR", r_small)
        p_large = BayesianProjection(); p_large.fit("WR", r_large)
        assert p_large.position_sigma_prior > p_small.position_sigma_prior


class TestSummary:
    """Tests for BayesianProjection.summary() — pure numpy, no MCMC."""

    def test_summary_keys(self):
        from ml.bayesian_layer import BayesianProjection
        samples = np.random.default_rng(0).normal(80.0, 15.0, 1000)
        s = BayesianProjection.summary(samples)
        for k in ("mean", "std", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "n"):
            assert k in s, f"missing key: {k}"

    def test_summary_mean_close(self):
        from ml.bayesian_layer import BayesianProjection
        samples = np.random.default_rng(1).normal(80.0, 15.0, 10000)
        s = BayesianProjection.summary(samples)
        assert abs(s["mean"] - 80.0) < 2.0

    def test_summary_n_equals_sample_count(self):
        from ml.bayesian_layer import BayesianProjection
        samples = np.ones(500)
        s = BayesianProjection.summary(samples)
        assert s["n"] == 500

    def test_summary_percentile_ordering(self):
        from ml.bayesian_layer import BayesianProjection
        samples = np.random.default_rng(2).normal(0, 1, 5000)
        s = BayesianProjection.summary(samples)
        assert s["p5"] < s["p10"] < s["p25"] < s["p50"] < s["p75"] < s["p90"] < s["p95"]

    def test_summary_std_close(self):
        from ml.bayesian_layer import BayesianProjection
        samples = np.random.default_rng(3).normal(0, 20.0, 50000)
        s = BayesianProjection.summary(samples)
        assert abs(s["std"] - 20.0) < 1.0

    def test_summary_works_on_scalar_ndarray(self):
        from ml.bayesian_layer import BayesianProjection
        s = BayesianProjection.summary(np.array([42.0]))
        assert s["n"] == 1
        assert s["mean"] == 42.0

    def test_summary_returns_floats(self):
        from ml.bayesian_layer import BayesianProjection
        samples = np.random.default_rng(4).normal(50, 10, 100)
        s = BayesianProjection.summary(samples)
        for k, v in s.items():
            if k != "n":
                assert isinstance(v, float), f"{k} should be float, got {type(v)}"
            else:
                assert isinstance(v, int), f"n should be int, got {type(v)}"


class TestPosteriorSamplesWithoutFit:
    """posterior_samples() should raise before fit() is called."""

    def test_raises_without_fit(self):
        from ml.bayesian_layer import BayesianProjection
        proj = BayesianProjection()
        with pytest.raises(RuntimeError, match="fit()"):
            proj.posterior_samples(80.0, 100.0, n_samples=10)


# ---------------------------------------------------------------------------
# B. Integration Tests — require MCMC sampling
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPosteriorSamplesSampling:
    """Integration tests that actually run the NUTS sampler."""

    @pytest.fixture(scope="class")
    def fitted_projection(self):
        """One fitted projection shared across integration tests to save time."""
        from ml.bayesian_layer import BayesianProjection
        residuals = _make_residuals(200, std=20.0, seed=7)
        proj = BayesianProjection()
        proj.fit("WR", residuals)
        return proj

    def test_returns_ndarray(self, fitted_projection):
        samples = fitted_projection.posterior_samples(80.0, 100.0, n_samples=500)
        assert isinstance(samples, np.ndarray)

    def test_sample_count(self, fitted_projection):
        from ml.bayesian_layer import _CHAINS
        n = 300
        samples = fitted_projection.posterior_samples(80.0, 100.0, n_samples=n)
        assert samples.shape == (n * _CHAINS,)

    def test_posterior_mean_close_to_stacked_estimate(self, fitted_projection):
        """
        With a tight Kalman variance (kv=1), the posterior mean should be
        close to stacked_estimate. Tolerance: 15 yards (MCMC is noisy).
        """
        samples = fitted_projection.posterior_samples(80.0, 1.0, n_samples=500)
        mean = np.mean(samples)
        assert abs(mean - 80.0) < 15.0, f"Expected mean ~80, got {mean:.1f}"

    def test_wide_kalman_variance_widens_posterior(self, fitted_projection):
        """
        Larger kalman_variance → wider mu prior → wider posterior spread.
        """
        s_tight = fitted_projection.posterior_samples(80.0, 1.0,    n_samples=500)
        s_wide  = fitted_projection.posterior_samples(80.0, 10000.0, n_samples=500)
        std_tight = np.std(s_tight)
        std_wide  = np.std(s_wide)
        assert std_wide > std_tight, (
            f"Wide KV should give wider posterior: tight_std={std_tight:.1f}, "
            f"wide_std={std_wide:.1f}"
        )

    def test_monotonic_percentiles_across_estimates(self, fitted_projection):
        """
        Higher stacked_estimate should shift posterior upward: p50 increases.
        """
        s_lo = fitted_projection.posterior_samples(40.0, 50.0, n_samples=300)
        s_hi = fitted_projection.posterior_samples(120.0, 50.0, n_samples=300)
        p50_lo = np.median(s_lo)
        p50_hi = np.median(s_hi)
        assert p50_hi > p50_lo, (
            f"Higher estimate should give higher p50: lo={p50_lo:.1f}, hi={p50_hi:.1f}"
        )

    def test_rhat_ok_after_sampling(self, fitted_projection):
        """R-hat < 1.1 indicates good NUTS chain convergence."""
        fitted_projection.posterior_samples(80.0, 100.0, n_samples=500, use_mcmc=True)
        assert fitted_projection.rhat_ok(threshold=1.1)

    def test_summary_from_posterior_samples(self, fitted_projection):
        """End-to-end: posterior_samples → summary works correctly."""
        from ml.bayesian_layer import BayesianProjection
        samples = fitted_projection.posterior_samples(80.0, 100.0, n_samples=500)
        s = BayesianProjection.summary(samples)
        assert s["n"] > 0
        assert s["p10"] < s["p50"] < s["p90"]

    def test_second_call_different_estimate(self, fitted_projection):
        """
        Second posterior_samples() call with different parameters should
        produce a shifted mean (model caching works correctly).
        """
        s1 = fitted_projection.posterior_samples(50.0, 50.0, n_samples=300)
        s2 = fitted_projection.posterior_samples(150.0, 50.0, n_samples=300)
        # Means should differ by at least 50 yards
        assert abs(np.mean(s2) - np.mean(s1)) > 50.0, (
            f"Second call should reflect new estimate: "
            f"mean1={np.mean(s1):.1f}, mean2={np.mean(s2):.1f}"
        )


@pytest.mark.slow
class TestMultiplePositions:
    """Each position gets a separate BayesianProjection instance."""

    def test_different_positions_different_sigma_prior(self):
        from ml.bayesian_layer import BayesianProjection
        # QBs have higher variance residuals (passing yards swing more)
        qb_residuals = _make_residuals(200, std=50.0, seed=10)
        wr_residuals = _make_residuals(200, std=20.0, seed=11)

        qb = BayesianProjection(); qb.fit("QB", qb_residuals)
        wr = BayesianProjection(); wr.fit("WR", wr_residuals)

        assert qb.position_sigma_prior > wr.position_sigma_prior

    def test_position_instances_independent(self):
        """Fitting one instance does not affect another."""
        from ml.bayesian_layer import BayesianProjection
        p1 = BayesianProjection(); p1.fit("WR", _make_residuals(50, std=20.0))
        p2 = BayesianProjection(); p2.fit("RB", _make_residuals(50, std=10.0))
        # sigma_priors differ
        assert p1.position_sigma_prior != p2.position_sigma_prior
        # models are separate objects
        assert p1._model is not p2._model


# ---------------------------------------------------------------------------
# C. Calibration Tests — posterior correctness against synthetic truth
# (M8 — audit finding: Bayesian layer correctness not tested)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPosteriorCalibration:
    """
    Verify posterior is calibrated against a known synthetic true value.

    Tests that:
      1. The 95% posterior credible interval contains the true generating value.
      2. The posterior mean is within 2σ of the true value (σ = residual std).

    These checks prove the posterior is not systematically biased when the
    stacked estimate equals the true value (ideal-prediction case).
    """

    @pytest.fixture(scope="class")
    def fitted_wr(self):
        from ml.bayesian_layer import BayesianProjection
        rng = np.random.default_rng(42)
        residuals = rng.normal(0.0, 20.0, 300)
        proj = BayesianProjection()
        proj.fit("WR", residuals)
        return proj, 20.0  # (model, residual_std)

    def test_95_pct_credible_interval_contains_true_value(self, fitted_wr):
        """
        95% posterior CI must contain the synthetic true value (80.0) when the
        stacked estimate equals the true value and Kalman variance is tight (1.0).
        """
        proj, _ = fitted_wr
        true_value = 80.0
        samples = proj.posterior_samples(true_value, kalman_variance=1.0, n_samples=1000)

        p2_5  = float(np.percentile(samples, 2.5))
        p97_5 = float(np.percentile(samples, 97.5))

        assert p2_5 <= true_value <= p97_5, (
            f"95% credible interval [{p2_5:.1f}, {p97_5:.1f}] does not contain "
            f"true_value={true_value}. Posterior is miscalibrated."
        )

    def test_posterior_mean_within_2sigma_of_true_value(self, fitted_wr):
        """
        Posterior mean must lie within 2 × residual_std of the true value
        when stacked_estimate = true_value and Kalman variance is tight.
        """
        proj, residual_std = fitted_wr
        true_value = 80.0
        samples = proj.posterior_samples(true_value, kalman_variance=1.0, n_samples=1000)
        posterior_mean = float(np.mean(samples))

        tolerance = 2.0 * residual_std
        assert abs(posterior_mean - true_value) < tolerance, (
            f"Posterior mean {posterior_mean:.1f} deviates more than 2σ "
            f"({tolerance:.1f} yds) from true value {true_value}. Model is biased."
        )

    def test_posterior_interval_width_scales_with_uncertainty(self, fitted_wr):
        """
        Higher Kalman variance → wider posterior interval.
        This verifies the Kalman-informed prior is wired into uncertainty correctly.
        """
        proj, _ = fitted_wr
        s_tight = proj.posterior_samples(80.0, kalman_variance=1.0,     n_samples=500)
        s_wide  = proj.posterior_samples(80.0, kalman_variance=10000.0, n_samples=500)

        width_tight = float(np.percentile(s_tight, 97.5) - np.percentile(s_tight, 2.5))
        width_wide  = float(np.percentile(s_wide,  97.5) - np.percentile(s_wide,  2.5))

        assert width_wide > width_tight, (
            f"High-uncertainty posterior (width={width_wide:.1f}) should be wider "
            f"than low-uncertainty posterior (width={width_tight:.1f})."
        )
