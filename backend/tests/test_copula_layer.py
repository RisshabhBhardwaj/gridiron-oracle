"""
backend/tests/test_copula_layer.py

Unit tests for ml/copula_layer.py.
All tests run without a database connection.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from ml.copula_layer import (
    CopulaLayer,
    SGPLeg,
    SGPResult,
    _to_uniform,
    _to_gaussian,
    _estimate_correlation_matrix,
    _draw_correlated_uniform,
    _quantile_transform,
    get_copula,
    reset_copula,
)


RNG = np.random.default_rng(42)


# ── COPULA MATH ───────────────────────────────────────────────────────────────

class TestCopulaMath:
    def test_to_uniform_output_range(self):
        """Uniform scores must be in (0, 1) exclusive."""
        samples = RNG.normal(0, 1, 500)
        u = _to_uniform(samples)
        assert u.min() > 0
        assert u.max() < 1
        assert len(u) == len(samples)

    def test_to_uniform_order_preserving(self):
        """Rank transform must be monotone: larger sample → larger score."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = _to_uniform(samples)
        assert np.all(np.diff(u) > 0)

    def test_to_gaussian_maps_from_uniform(self):
        """Gaussian scores from uniform (0.5) should be 0."""
        u = np.array([0.5])
        z = _to_gaussian(u)
        assert abs(z[0]) < 0.01

    def test_to_gaussian_finite(self):
        """Gaussian scores must be finite (no Inf from extreme quantiles)."""
        u = np.clip(RNG.uniform(0, 1, 1000), 1e-6, 1 - 1e-6)
        z = _to_gaussian(u)
        assert np.all(np.isfinite(z))

    def test_estimate_correlation_matrix_identity_when_uncorrelated(self):
        """Uncorrelated data should produce ~identity correlation matrix."""
        rng = np.random.default_rng(0)
        scores = rng.standard_normal((2000, 3))
        corr = _estimate_correlation_matrix(scores)
        off_diag = corr[np.triu_indices(3, k=1)]
        assert np.all(np.abs(off_diag) < 0.15), f"Expected near-zero off-diag, got {off_diag}"

    def test_estimate_correlation_matrix_detects_correlation(self):
        """Correlated data should produce near-target correlation matrix."""
        rng = np.random.default_rng(1)
        n = 3000
        z1 = rng.standard_normal(n)
        z2 = 0.7 * z1 + 0.714 * rng.standard_normal(n)  # ρ ≈ 0.7
        scores = np.column_stack([z1, z2])
        corr = _estimate_correlation_matrix(scores)
        assert abs(corr[0, 1] - 0.7) < 0.1, f"Expected ρ≈0.7, got {corr[0, 1]:.2f}"

    def test_estimate_correlation_matrix_psd(self):
        """Output matrix must be positive semi-definite."""
        rng = np.random.default_rng(2)
        scores = rng.standard_normal((500, 5))
        corr = _estimate_correlation_matrix(scores)
        eigenvalues = np.linalg.eigvalsh(corr)
        assert np.all(eigenvalues >= -1e-8), f"Non-PSD: min eigenvalue = {eigenvalues.min()}"

    def test_draw_correlated_uniform_shape(self):
        """Output shape should be (n_samples, n_players)."""
        corr = np.eye(4)
        u = _draw_correlated_uniform(corr, n_samples=500)
        assert u.shape == (500, 4)

    def test_draw_correlated_uniform_range(self):
        """All outputs must be in (0, 1)."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        u = _draw_correlated_uniform(corr, n_samples=1000)
        assert np.all(u > 0)
        assert np.all(u < 1)

    def test_draw_correlated_uniform_recovers_correlation(self):
        """Correlation in output should approximately match input matrix."""
        target_rho = 0.6
        corr = np.array([[1.0, target_rho], [target_rho, 1.0]])
        u = _draw_correlated_uniform(corr, n_samples=5000, rng=np.random.default_rng(99))
        # Convert back to normal space for Pearson correlation
        z = scipy_stats.norm.ppf(u)
        rho_actual = np.corrcoef(z[:, 0], z[:, 1])[0, 1]
        assert abs(rho_actual - target_rho) < 0.08, f"Expected ρ≈{target_rho}, got {rho_actual:.3f}"

    def test_quantile_transform_maps_uniformly(self):
        """Uniform inputs → samples that match the reference distribution."""
        ref = RNG.normal(100, 20, 2000)
        u = np.linspace(0.01, 0.99, 100)
        out = _quantile_transform(u, ref)
        # Median of output should be near reference median
        assert abs(np.median(out) - np.median(ref)) < 10


# ── COPULA LAYER CORE ─────────────────────────────────────────────────────────

class TestCopulaLayer:
    def test_structural_correlation_same_team(self):
        """Same-team players should have positive structural correlation."""
        c = CopulaLayer()
        c._player_metadata = {
            "p1": {"team": "MIN", "position": "WR"},
            "p2": {"team": "MIN", "position": "WR"},
        }
        rho = c._structural_correlation("p1", "p2", "receiving_yards", "receiving_yards")
        assert rho > 0.2

    def test_structural_correlation_qb_wr(self):
        """QB + WR same team should have highest correlation."""
        c = CopulaLayer()
        c._player_metadata = {
            "qb": {"team": "KC", "position": "QB"},
            "wr": {"team": "KC", "position": "WR"},
        }
        rho = c._structural_correlation("qb", "wr", "passing_yards", "receiving_yards")
        assert rho > 0.45

    def test_structural_correlation_opp_team_lower(self):
        """Opposing team players should have lower correlation than same-team."""
        c = CopulaLayer()
        c._player_metadata = {
            "p1": {"team": "MIN", "position": "WR"},
            "p2": {"team": "MIN", "position": "WR"},
            "p3": {"team": "GB",  "position": "WR"},
        }
        same_team = c._structural_correlation("p1", "p2", "receiving_yards", "receiving_yards")
        opp_team  = c._structural_correlation("p1", "p3", "receiving_yards", "receiving_yards")
        assert same_team > opp_team

    def test_build_correlation_matrix_shape(self):
        """build_correlation_matrix() should return (n, n) matrix."""
        c = CopulaLayer()
        nodes = [("p1", "receiving_yards"), ("p2", "receiving_yards"), ("p3", "receiving_yards")]
        corr = c.build_correlation_matrix(nodes)
        assert corr.shape == (3, 3)

    def test_build_correlation_matrix_diagonal_ones(self):
        """Diagonal must be 1.0."""
        c = CopulaLayer()
        nodes = [("p1", "receiving_yards"), ("p2", "receiving_yards"), ("p3", "receiving_yards")]
        corr = c.build_correlation_matrix(nodes)
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_build_correlation_matrix_symmetric(self):
        """Matrix must be symmetric."""
        c = CopulaLayer()
        nodes = [("p1", "receiving_yards"), ("p2", "receiving_yards"), ("p3", "receiving_yards")]
        corr = c.build_correlation_matrix(nodes)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_draw_joint_samples_shape(self):
        """draw_joint_samples should return (n_samples, n_players)."""
        rng = np.random.default_rng(0)
        c = CopulaLayer()
        marginals = {
            ("p1", "receiving_yards"): np.maximum(rng.normal(80, 20, 1000), 0),
            ("p2", "receiving_yards"): np.maximum(rng.normal(60, 15, 1000), 0),
        }
        nodes = [("p1", "receiving_yards"), ("p2", "receiving_yards")]
        joint = c.draw_joint_samples(nodes, marginals, n_samples=500, rng=rng)
        assert joint.shape == (500, 2)

    def test_draw_joint_samples_non_negative(self):
        """All joint stat draws should be non-negative."""
        rng = np.random.default_rng(1)
        c = CopulaLayer()
        marginals = {("p1", "receiving_yards"): np.maximum(rng.normal(50, 10, 500), 0)}
        nodes = [("p1", "receiving_yards")]
        joint = c.draw_joint_samples(nodes, marginals, n_samples=200, rng=rng)
        assert np.all(joint >= 0)

    def test_draw_joint_samples_single_player(self):
        """Single-player joint draw should still return (n, 1)."""
        rng = np.random.default_rng(2)
        c = CopulaLayer()
        marginals = {("p1", "receiving_yards"): rng.normal(100, 20, 1000)}
        nodes = [("p1", "receiving_yards")]
        joint = c.draw_joint_samples(nodes, marginals, n_samples=300)
        assert joint.shape == (300, 1)

    def test_fit_with_valid_oof_data(self):
        """fit() should populate _oof_corr without errors."""
        rng = np.random.default_rng(3)
        c = CopulaLayer()
        n_games = 50
        oof_df = pd.DataFrame({
            "player_id": ["p1"] * n_games + ["p2"] * n_games,
            "game_id":   [f"g{i}" for i in range(n_games)] * 2,
            "stat":      ["receiving_yards"] * (n_games * 2),
            "actual":    rng.normal(80, 20, n_games * 2),
            "predicted": rng.normal(75, 20, n_games * 2),
        })
        c.fit(oof_df)
        assert c._fitted is True

    def test_fit_handles_missing_columns_gracefully(self):
        """fit() with missing columns should not crash (fall back to structural)."""
        c = CopulaLayer()
        bad_df = pd.DataFrame({"player_id": ["p1"], "actual": [80.0]})
        c.fit(bad_df)  # Should not raise
        assert c._fitted is True


# ── SGP PRICING ───────────────────────────────────────────────────────────────

class TestSGPPricing:
    def _make_posterior(self, player_id: str, stat: str, mean: float, std: float, n: int = 2000):
        rng = np.random.default_rng(sum(ord(c) for c in player_id))
        return {player_id: {stat: np.maximum(rng.normal(mean, std, n), 0)}}

    def test_sgp_price_single_leg(self):
        """Single-leg SGP probability should match marginal probability."""
        c = CopulaLayer()
        posterior = self._make_posterior("p1", "receiving_yards", 80, 20)
        leg = SGPLeg(player_id="p1", stat="receiving_yards", threshold=70.0, over=True)
        result = c.price_sgp([leg], posterior, n_samples=5000)
        # With mean=80, std=20 → P(X > 70) ≈ 0.69
        assert 0.55 < result.parlay_probability < 0.85
        assert result.leg_probabilities[0] == pytest.approx(result.parlay_probability, abs=0.05)

    def test_sgp_result_fields(self):
        """SGPResult should have all expected fields."""
        c = CopulaLayer()
        posterior = self._make_posterior("p1", "receiving_yards", 80, 20)
        leg = SGPLeg(player_id="p1", stat="receiving_yards", threshold=70.0, over=True)
        result = c.price_sgp([leg], posterior, n_samples=1000)
        assert result.parlay_probability >= 0
        assert result.independent_probability >= 0
        assert result.n_samples_used == 1000
        assert len(result.leg_probabilities) == 1

    def test_sgp_empty_legs(self):
        """Empty legs should return probability 1.0."""
        c = CopulaLayer()
        result = c.price_sgp([], {})
        assert result.parlay_probability == 1.0

    def test_sgp_impossible_parlay_near_zero(self):
        """Two legs that can't both hit should have near-zero joint probability."""
        c = CopulaLayer()
        # Player with mean=80 yd — P(>150) is very low
        posterior = {
            "p1": {"receiving_yards": np.random.default_rng(42).normal(80, 15, 3000).clip(0)}
        }
        leg = SGPLeg(player_id="p1", stat="receiving_yards", threshold=150.0, over=True)
        result = c.price_sgp([leg], posterior, n_samples=3000)
        assert result.parlay_probability < 0.05

    def test_sgp_label(self):
        leg = SGPLeg(player_id="p1", stat="receiving_yards", threshold=94.5, over=True, name="Justin Jefferson")
        assert "Justin Jefferson" in leg.label
        assert "OVER" in leg.label
        assert "94.5" in leg.label

    def test_sgp_result_str_representation(self):
        """SGPResult.__str__ should not crash."""
        c = CopulaLayer()
        posterior = self._make_posterior("p1", "receiving_yards", 80, 20)
        leg = SGPLeg(player_id="p1", stat="receiving_yards", threshold=70.0, over=True)
        result = c.price_sgp([leg], posterior, n_samples=500)
        s = str(result)
        assert "SGP" in s


# ── SINGLETON ─────────────────────────────────────────────────────────────────

class TestCopulaSingleton:
    def setup_method(self):
        reset_copula()

    def teardown_method(self):
        reset_copula()

    def test_get_copula_returns_same_instance(self):
        c1 = get_copula()
        c2 = get_copula()
        assert c1 is c2

    def test_reset_copula_clears_singleton(self):
        c1 = get_copula()
        reset_copula()
        c2 = get_copula()
        assert c1 is not c2
