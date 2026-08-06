"""
backend/tests/test_monte_carlo.py

Tests for ml/monte_carlo.py — MonteCarloProjector + ProjectionResult.

All tests are fast (pure NumPy, no MCMC sampling).
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samples(mean: float, std: float = 15.0, n: int = 2000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, std, size=n)


def _projector():
    from ml.monte_carlo import MonteCarloProjector
    return MonteCarloProjector()


# ---------------------------------------------------------------------------
# TestProjectionResult
# ---------------------------------------------------------------------------

class TestProjectionResult:
    """Tests on the ProjectionResult dataclass."""

    def test_is_frozen(self):
        from ml.monte_carlo import ProjectionResult
        r = ProjectionResult(
            projection=80.0, floor=50.0, ceiling=110.0,
            p5=40.0, p25=65.0, p75=95.0, p95=120.0,
            boom_probability=0.3, bust_probability=0.1,
            fantasy_points_distribution=np.array([8.0, 9.0, 10.0]),
            fantasy_projection=9.0, fantasy_floor=8.0, fantasy_ceiling=10.0,
            n_samples=3,
        )
        with pytest.raises((AttributeError, TypeError)):
            r.projection = 99.0  # type: ignore[misc]

    def test_is_hashable(self):
        from ml.monte_carlo import ProjectionResult
        r = ProjectionResult(
            projection=80.0, floor=50.0, ceiling=110.0,
            p5=40.0, p25=65.0, p75=95.0, p95=120.0,
            boom_probability=0.3, bust_probability=0.1,
            fantasy_points_distribution=np.array([8.0, 9.0]),
            fantasy_projection=8.5, fantasy_floor=8.0, fantasy_ceiling=9.0,
            n_samples=2,
        )
        # Should not raise
        h = hash(r)
        assert isinstance(h, int)

    def test_can_be_put_in_set(self):
        from ml.monte_carlo import ProjectionResult
        r = ProjectionResult(
            projection=80.0, floor=50.0, ceiling=110.0,
            p5=40.0, p25=65.0, p75=95.0, p95=120.0,
            boom_probability=0.3, bust_probability=0.1,
            fantasy_points_distribution=np.array([8.0]),
            fantasy_projection=8.0, fantasy_floor=8.0, fantasy_ceiling=8.0,
            n_samples=1,
        )
        s = {r}
        assert len(s) == 1

    def test_equality_same_values(self):
        from ml.monte_carlo import ProjectionResult
        fp = np.array([8.0, 9.0])
        r1 = ProjectionResult(80.0, 50.0, 110.0, 40.0, 65.0, 95.0, 120.0, 0.3, 0.1, fp, 8.5, 8.0, 9.0, 2)
        r2 = ProjectionResult(80.0, 50.0, 110.0, 40.0, 65.0, 95.0, 120.0, 0.3, 0.1, fp.copy(), 8.5, 8.0, 9.0, 2)
        assert r1 == r2

    def test_n_samples_is_int(self):
        from ml.monte_carlo import MonteCarloProjector
        r = _projector().project(_samples(80.0), "receiving_yards", "WR")
        assert isinstance(r.n_samples, int)

    def test_all_floats_are_python_float(self):
        """Ensures JSON serializability — no numpy scalars."""
        from ml.monte_carlo import MonteCarloProjector
        r = _projector().project(_samples(80.0), "receiving_yards", "WR")
        for field in ("projection", "floor", "ceiling",
                      "boom_probability", "bust_probability",
                      "fantasy_projection", "fantasy_floor", "fantasy_ceiling"):
            val = getattr(r, field)
            assert isinstance(val, float), f"{field} should be float, got {type(val)}"


# ---------------------------------------------------------------------------
# TestProjectOrdering
# ---------------------------------------------------------------------------

class TestProjectOrdering:
    """floor < projection < ceiling must always hold."""

    def test_floor_lt_projection_lt_ceiling(self):
        r = _projector().project(_samples(80.0, std=20.0), "receiving_yards", "WR")
        assert r.floor < r.projection < r.ceiling

    def test_fantasy_floor_lt_projection_lt_ceiling(self):
        r = _projector().project(_samples(80.0, std=20.0), "receiving_yards", "WR")
        assert r.fantasy_floor < r.fantasy_projection < r.fantasy_ceiling

    def test_floor_lt_projection_lt_ceiling_rushing(self):
        r = _projector().project(_samples(60.0, std=15.0), "rushing_yards", "RB")
        assert r.floor < r.projection < r.ceiling

    def test_floor_lt_projection_lt_ceiling_passing(self):
        r = _projector().project(_samples(250.0, std=40.0), "passing_yards", "QB")
        assert r.floor < r.projection < r.ceiling

    def test_ordering_tight_distribution(self):
        """Even with very tight std (std=0.1), ordering holds."""
        samples = np.random.default_rng(5).normal(80.0, 0.1, 2000)
        r = _projector().project(samples, "receiving_yards", "WR")
        assert r.floor <= r.projection <= r.ceiling


# ---------------------------------------------------------------------------
# TestProjectMonotonicity
# ---------------------------------------------------------------------------

class TestProjectMonotonicity:
    """Higher posterior mean → higher projection and fantasy_projection."""

    def test_higher_mean_higher_projection(self):
        r_lo = _projector().project(_samples(40.0, seed=10), "receiving_yards", "WR")
        r_hi = _projector().project(_samples(120.0, seed=11), "receiving_yards", "WR")
        assert r_hi.projection > r_lo.projection

    def test_higher_mean_higher_fantasy_projection(self):
        r_lo = _projector().project(_samples(40.0, seed=12), "receiving_yards", "WR")
        r_hi = _projector().project(_samples(120.0, seed=13), "receiving_yards", "WR")
        assert r_hi.fantasy_projection > r_lo.fantasy_projection

    def test_higher_mean_higher_boom_probability(self):
        r_lo = _projector().project(_samples(30.0, seed=14, std=10.0), "receiving_yards", "WR")
        r_hi = _projector().project(_samples(130.0, seed=15, std=10.0), "receiving_yards", "WR")
        assert r_hi.boom_probability > r_lo.boom_probability


# ---------------------------------------------------------------------------
# TestBoomBustProbabilities
# ---------------------------------------------------------------------------

class TestBoomBustProbabilities:
    """Probability math correctness."""

    def test_boom_plus_bust_le_one(self):
        r = _projector().project(_samples(80.0, std=20.0), "receiving_yards", "WR")
        assert r.boom_probability + r.bust_probability <= 1.0

    def test_boom_plus_bust_le_one_all_positions(self):
        projector = _projector()
        for pos, stat, mean in [
            ("WR", "receiving_yards", 60.0),
            ("RB", "rushing_yards", 60.0),
            ("QB", "passing_yards", 220.0),
            ("TE", "receiving_yards", 45.0),
        ]:
            r = projector.project(_samples(mean, seed=99), stat, pos)
            assert r.boom_probability + r.bust_probability <= 1.0, (
                f"{pos}/{stat}: boom={r.boom_probability}, bust={r.bust_probability}"
            )

    def test_zero_yards_bust_high(self):
        """Near-zero projection → bust_probability close to 1."""
        samples = np.zeros(2000)
        r = _projector().project(samples, "receiving_yards", "WR")
        # bust_threshold for WR receiving_yards = 30; P(0 < 30) = 1.0
        assert r.bust_probability == pytest.approx(1.0)

    def test_elite_wr_boom_high(self):
        """150 WR yards (boom threshold = 100) → boom_probability should be high."""
        samples = np.random.default_rng(20).normal(150.0, 5.0, 2000)
        r = _projector().project(samples, "receiving_yards", "WR")
        assert r.boom_probability > 0.8, f"Expected boom_prob > 0.8, got {r.boom_probability:.3f}"

    def test_elite_qb_boom_high(self):
        """350 QB passing yards (boom threshold = 250) → boom_probability high."""
        samples = np.random.default_rng(21).normal(350.0, 10.0, 2000)
        r = _projector().project(samples, "passing_yards", "QB")
        assert r.boom_probability > 0.8

    def test_probabilities_between_zero_and_one(self):
        r = _projector().project(_samples(80.0), "receiving_yards", "WR")
        assert 0.0 <= r.boom_probability <= 1.0
        assert 0.0 <= r.bust_probability <= 1.0


# ---------------------------------------------------------------------------
# TestFantasyScoring
# ---------------------------------------------------------------------------

class TestFantasyScoring:
    """Fantasy points distribution correctness."""

    def test_receiving_yards_fantasy_rate(self):
        """receiving_yards: 0.1 pts/yd → 100 yards = 10 pts."""
        samples = np.full(2000, 100.0)
        r = _projector().project(samples, "receiving_yards", "WR")
        assert r.fantasy_projection == pytest.approx(10.0, abs=0.1)

    def test_rushing_yards_fantasy_rate(self):
        """rushing_yards: 0.1 pts/yd → 80 yards = 8 pts."""
        samples = np.full(2000, 80.0)
        r = _projector().project(samples, "rushing_yards", "RB")
        assert r.fantasy_projection == pytest.approx(8.0, abs=0.1)

    def test_passing_yards_fantasy_rate(self):
        """passing_yards: 0.04 pts/yd → 250 yards = 10 pts."""
        samples = np.full(2000, 250.0)
        r = _projector().project(samples, "passing_yards", "QB")
        assert r.fantasy_projection == pytest.approx(10.0, abs=0.1)

    def test_fantasy_distribution_length_matches_n_samples(self):
        n = 500
        samples = _samples(80.0, n=n)
        r = _projector().project(samples, "receiving_yards", "WR")
        assert r.fantasy_points_distribution.shape == (n,)
        assert r.n_samples == n

    def test_fantasy_distribution_is_ndarray(self):
        r = _projector().project(_samples(80.0), "receiving_yards", "WR")
        assert isinstance(r.fantasy_points_distribution, np.ndarray)

    def test_fantasy_projection_eq_median_of_distribution(self):
        samples = _samples(80.0)
        r = _projector().project(samples, "receiving_yards", "WR")
        expected = float(np.median(r.fantasy_points_distribution))
        assert r.fantasy_projection == pytest.approx(expected, abs=0.01)

    def test_receptions_stat_one_pt_each(self):
        """receptions: 1 pt/reception → 7 receptions = 7 pts."""
        samples = np.full(2000, 7.0)
        r = _projector().project(samples, "receptions", "WR")
        assert r.fantasy_projection == pytest.approx(7.0, abs=0.1)


# ---------------------------------------------------------------------------
# TestProjectNSamples
# ---------------------------------------------------------------------------

class TestProjectNSamples:
    """n_samples field reflects input array size."""

    def test_n_samples_2000(self):
        r = _projector().project(_samples(80.0, n=2000), "receiving_yards", "WR")
        assert r.n_samples == 2000

    def test_n_samples_100(self):
        r = _projector().project(_samples(80.0, n=100), "receiving_yards", "WR")
        assert r.n_samples == 100

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _projector().project(np.array([]), "receiving_yards", "WR")


# ---------------------------------------------------------------------------
# TestProjectionVsSingleValue
# ---------------------------------------------------------------------------

class TestProjectionValues:
    """Projection (p50) close to sample median."""

    def test_projection_close_to_sample_median(self):
        samples = _samples(80.0, std=20.0, n=10000)
        r = _projector().project(samples, "receiving_yards", "WR")
        assert abs(r.projection - float(np.median(samples))) < 0.1

    def test_floor_is_p10(self):
        samples = _samples(80.0, n=10000)
        r = _projector().project(samples, "receiving_yards", "WR")
        expected_floor = float(np.percentile(samples, 10))
        assert r.floor == pytest.approx(expected_floor, abs=0.01)

    def test_ceiling_is_p90(self):
        samples = _samples(80.0, n=10000)
        r = _projector().project(samples, "receiving_yards", "WR")
        expected_ceiling = float(np.percentile(samples, 90))
        assert r.ceiling == pytest.approx(expected_ceiling, abs=0.01)


# ---------------------------------------------------------------------------
# TestPositionHandling
# ---------------------------------------------------------------------------

class TestPositionHandling:
    """Unknown positions fall back gracefully."""

    def test_unknown_position_no_crash(self):
        r = _projector().project(_samples(80.0), "receiving_yards", "FLEX")
        assert isinstance(r.projection, float)

    def test_unknown_stat_no_crash(self):
        """Unregistered stat gives 0 fantasy pts/unit but doesn't crash."""
        r = _projector().project(_samples(5.0), "some_new_stat", "WR")
        assert r.fantasy_projection == 0.0

    def test_unknown_stat_thresholds_default_zero(self):
        samples = _samples(50.0)
        r = _projector().project(samples, "completely_new_stat", "WR")
        # boom_threshold = 0 → P(samples > 0) should be high for mean=50
        assert r.boom_probability > 0.9


# ---------------------------------------------------------------------------
# TestBatchProject
# ---------------------------------------------------------------------------

class TestBatchProject:
    """batch_project produces same results as individual project() calls."""

    @pytest.fixture
    def three_players(self):
        return {
            "player_a": _samples(80.0, seed=30),
            "player_b": _samples(60.0, seed=31),
            "player_c": _samples(120.0, seed=32),
        }

    def test_returns_dataframe(self, three_players):
        df = _projector().batch_project(three_players, "receiving_yards", "WR")
        assert isinstance(df, pd.DataFrame)

    def test_index_is_player_id(self, three_players):
        df = _projector().batch_project(three_players, "receiving_yards", "WR")
        assert set(df.index) == set(three_players.keys())

    def test_row_count(self, three_players):
        df = _projector().batch_project(three_players, "receiving_yards", "WR")
        assert len(df) == len(three_players)

    def test_expected_columns(self, three_players):
        df = _projector().batch_project(three_players, "receiving_yards", "WR")
        for col in ("projection", "floor", "ceiling",
                    "p5", "p25", "p75", "p95",
                    "boom_probability", "bust_probability",
                    "fantasy_projection", "fantasy_floor", "fantasy_ceiling",
                    "n_samples"):
            assert col in df.columns, f"missing column: {col}"

    def test_matches_single_project_projection(self, three_players):
        projector = _projector()
        df = projector.batch_project(three_players, "receiving_yards", "WR")
        for pid, samples in three_players.items():
            single = projector.project(samples, "receiving_yards", "WR")
            assert df.loc[pid, "projection"] == pytest.approx(single.projection, abs=1e-6)

    def test_matches_single_project_floor_ceiling(self, three_players):
        projector = _projector()
        df = projector.batch_project(three_players, "receiving_yards", "WR")
        for pid, samples in three_players.items():
            single = projector.project(samples, "receiving_yards", "WR")
            assert df.loc[pid, "floor"]   == pytest.approx(single.floor, abs=1e-6)
            assert df.loc[pid, "ceiling"] == pytest.approx(single.ceiling, abs=1e-6)

    def test_matches_single_project_boom_bust(self, three_players):
        projector = _projector()
        df = projector.batch_project(three_players, "receiving_yards", "WR")
        for pid, samples in three_players.items():
            single = projector.project(samples, "receiving_yards", "WR")
            assert df.loc[pid, "boom_probability"] == pytest.approx(single.boom_probability, abs=1e-6)
            assert df.loc[pid, "bust_probability"] == pytest.approx(single.bust_probability, abs=1e-6)

    def test_matches_single_project_fantasy(self, three_players):
        projector = _projector()
        df = projector.batch_project(three_players, "receiving_yards", "WR")
        for pid, samples in three_players.items():
            single = projector.project(samples, "receiving_yards", "WR")
            assert df.loc[pid, "fantasy_projection"] == pytest.approx(single.fantasy_projection, abs=1e-6)

    def test_empty_dict_returns_empty_dataframe(self):
        df = _projector().batch_project({}, "receiving_yards", "WR")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_ordering_in_batch_matches_single(self, three_players):
        """floor < projection < ceiling for all rows in the batch."""
        df = _projector().batch_project(three_players, "receiving_yards", "WR")
        assert (df["floor"] < df["projection"]).all()
        assert (df["projection"] < df["ceiling"]).all()

    def test_boom_plus_bust_le_one_batch(self, three_players):
        df = _projector().batch_project(three_players, "receiving_yards", "WR")
        assert ((df["boom_probability"] + df["bust_probability"]) <= 1.0).all()
