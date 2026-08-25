"""
backend/tests/test_backtest.py

Tests for ml/backtest.py — BacktestRunner + pure metric functions.

All tests are fast (pure NumPy, no MCMC, no DB).

Test categories:
  A. BacktestResult — frozen, hashable, field types
  B. compute_mae / compute_rmse — known-input reference tests
  C. compute_crps_single — hand-verified reference value
  D. compute_crps_batch — aggregation correctness
  E. compute_coverage — exact expected coverage
  F. BacktestRunner.evaluate() — end-to-end metric computation
  G. BacktestRunner.run() — structural tests with synthetic providers
  H. save_results — CSV output format
"""

from __future__ import annotations

import sys
import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _uniform_samples(lo: float, hi: float, n: int = 200, seed: int = 0) -> np.ndarray:
    return _rng(seed).uniform(lo, hi, n)


def _normal_samples(mu: float, sigma: float, n: int = 200, seed: int = 0) -> np.ndarray:
    return _rng(seed).normal(mu, sigma, n)


def _make_result(**overrides) -> "BacktestResult":
    from ml.backtest import BacktestResult
    defaults = dict(
        eval_season=2023, position="WR", stat="receiving_yards",
        n_games=100,
        stack_mae=15.0, stack_rmse=20.0, stack_crps=10.0,
        naive_mae=20.0, rolling_mae=18.0,
        coverage_80=0.78, coverage_50=0.49,
        baseline_improvement_pct=25.0,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _make_synthetic_providers(n_players: int = 10, n_weeks: int = 8, seed: int = 0):
    """
    Returns (data_provider, projection_provider) using purely synthetic data.
    Projections are unbiased ± noise around actuals so metrics are predictable.
    """
    rng = np.random.default_rng(seed)

    def _provider(eval_season, positions, stats):
        rows = []
        for pos in positions:
            for stat in stats:
                for pid_i in range(n_players):
                    pid = f"p_{pos}_{pid_i}"
                    for wk in range(1, n_weeks + 1):
                        actual = float(rng.normal(60.0, 20.0))
                        rows.append({
                            "player_id": pid,
                            "season": eval_season,
                            "week": wk,
                            "position": pos,
                            "stat": stat,
                            "actual_value": actual,
                            "naive_baseline": actual + float(rng.normal(0, 10)),
                            "rolling_baseline": actual + float(rng.normal(0, 12)),
                        })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _proj_provider(eval_season, positions, stats):
        rng2 = np.random.default_rng(seed + 1)
        rows = []
        for pos in positions:
            for stat in stats:
                for pid_i in range(n_players):
                    pid = f"p_{pos}_{pid_i}"
                    for wk in range(1, n_weeks + 1):
                        proj = float(rng2.normal(60.0, 15.0))
                        samples = rng2.normal(proj, 15.0, 50)
                        p10, p25, p75, p90 = np.percentile(samples, [10, 25, 75, 90])
                        rows.append({
                            "player_id": pid,
                            "season": eval_season,
                            "week": wk,
                            "stat": stat,
                            "projection": proj,
                            "floor": float(p10),
                            "ceiling": float(p90),
                            "p25": float(p25),
                            "p75": float(p75),
                            "posterior_samples": samples,
                        })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    return _provider, _proj_provider


# ---------------------------------------------------------------------------
# A. BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:

    def test_is_frozen(self):
        r = _make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.stack_mae = 99.0  # type: ignore[misc]

    def test_is_hashable(self):
        r = _make_result()
        h = hash(r)
        assert isinstance(h, int)

    def test_can_be_put_in_set(self):
        r = _make_result()
        s = {r, r}
        assert len(s) == 1

    def test_two_distinct_results_different_hash(self):
        r1 = _make_result(eval_season=2022)
        r2 = _make_result(eval_season=2023)
        assert r1 != r2

    def test_all_float_fields_are_python_float(self):
        r = _make_result()
        for field in ("stack_mae", "stack_rmse", "stack_crps",
                      "naive_mae", "rolling_mae", "coverage_80",
                      "coverage_50", "baseline_improvement_pct"):
            assert isinstance(getattr(r, field), float), \
                f"{field} should be float, got {type(getattr(r, field))}"

    def test_n_games_is_int(self):
        r = _make_result()
        assert isinstance(r.n_games, int)

    def test_eval_season_is_int(self):
        r = _make_result()
        assert isinstance(r.eval_season, int)


# ---------------------------------------------------------------------------
# B. compute_mae / compute_rmse
# ---------------------------------------------------------------------------

class TestComputeMAE:

    def test_perfect_predictions(self):
        from ml.backtest import compute_mae
        a = np.array([10.0, 20.0, 30.0])
        assert compute_mae(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        from ml.backtest import compute_mae
        preds   = np.array([10.0, 10.0, 10.0])
        actuals = np.array([0.0,  10.0, 20.0])
        # |10-0| + |10-10| + |10-20| / 3 = (10 + 0 + 10) / 3 ≈ 6.667
        assert compute_mae(preds, actuals) == pytest.approx(20.0 / 3.0, abs=1e-6)

    def test_returns_float(self):
        from ml.backtest import compute_mae
        result = compute_mae(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)

    def test_symmetric(self):
        from ml.backtest import compute_mae
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert compute_mae(a, b) == pytest.approx(compute_mae(b, a))


class TestComputeRMSE:

    def test_perfect_predictions(self):
        from ml.backtest import compute_rmse
        a = np.array([5.0, 10.0, 15.0])
        assert compute_rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        from ml.backtest import compute_rmse
        preds   = np.array([0.0])
        actuals = np.array([3.0])
        assert compute_rmse(preds, actuals) == pytest.approx(3.0)

    def test_rmse_ge_mae(self):
        """RMSE ≥ MAE always (Jensen's inequality)."""
        from ml.backtest import compute_mae, compute_rmse
        rng = np.random.default_rng(1)
        preds   = rng.normal(50, 15, 100)
        actuals = rng.normal(50, 15, 100)
        assert compute_rmse(preds, actuals) >= compute_mae(preds, actuals) - 1e-9

    def test_returns_float(self):
        from ml.backtest import compute_rmse
        result = compute_rmse(np.array([1.0]), np.array([2.0]))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# C. compute_crps_single — hand-verified reference
# ---------------------------------------------------------------------------

class TestComputeCRPSSingle:
    """
    Reference derivation for samples=[0,1,2,3,4], actual=2.0:
      mae_term = (2+1+0+1+2)/5 = 1.2
      spread   = (1/25) * Σ s_i*(2i-4) where i∈{0,1,2,3,4}
               = (1/25) * (0*-4 + 1*-2 + 2*0 + 3*2 + 4*4)
               = (1/25) * (0 - 2 + 0 + 6 + 16) = 20/25 = 0.8
      CRPS = 1.2 - 0.8 = 0.4
    """

    def test_hand_computed_reference(self):
        from ml.backtest import compute_crps_single
        samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert compute_crps_single(samples, 2.0) == pytest.approx(0.4, abs=1e-9)

    def test_perfect_deterministic_forecast(self):
        """All samples equal to actual → CRPS = 0."""
        from ml.backtest import compute_crps_single
        samples = np.full(100, 42.0)
        assert compute_crps_single(samples, 42.0) == pytest.approx(0.0, abs=1e-6)

    def test_crps_nonnegative(self):
        from ml.backtest import compute_crps_single
        rng = np.random.default_rng(2)
        for _ in range(20):
            samples = rng.normal(50, 15, 100)
            actual  = float(rng.normal(50, 20))
            assert compute_crps_single(samples, actual) >= -1e-9

    def test_crps_decreases_as_forecast_improves(self):
        """CRPS of forecast centered on actual < CRPS of badly off forecast."""
        from ml.backtest import compute_crps_single
        rng = np.random.default_rng(3)
        actual = 80.0
        good = rng.normal(80.0, 5.0,  500)  # centered on actual, tight
        bad  = rng.normal(50.0, 30.0, 500)  # off-center, spread out
        assert compute_crps_single(good, actual) < compute_crps_single(bad, actual)

    def test_empty_samples_raises(self):
        from ml.backtest import compute_crps_single
        with pytest.raises(ValueError, match="empty"):
            compute_crps_single(np.array([]), 5.0)

    def test_single_sample_equals_mae(self):
        """For 1 sample, CRPS = |sample - actual| (no spread to subtract)."""
        from ml.backtest import compute_crps_single
        # n=1: spread = (1/1) * s[0]*(2*0 - 1 + 1) = s[0]*0 = 0
        assert compute_crps_single(np.array([10.0]), 15.0) == pytest.approx(5.0)

    def test_returns_float(self):
        from ml.backtest import compute_crps_single
        result = compute_crps_single(np.array([1.0, 2.0, 3.0]), 2.0)
        assert isinstance(result, float)


class TestBacktestTrustPolicy:

    def test_loads_manifest_and_invalidations(self, tmp_path: Path):
        from ml.backtest import load_backtest_trust_policy

        baseline = tmp_path / "baseline.json"
        invalidations = tmp_path / "invalidations.json"
        baseline.write_text(json.dumps({
            "projection_policy": {
                "approved_pipeline_run_ids": ["run_a", "run_b"],
                "require_posterior_samples": True,
                "require_interval_columns": True,
            }
        }))
        invalidations.write_text(json.dumps({
            "invalid_pipeline_run_ids": ["run_bad"],
            "invalid_projection_targets": [
                {"position": "QB", "stat": "passing_yards", "reason": "bad"}
            ],
        }))

        policy = load_backtest_trust_policy(str(baseline), str(invalidations))
        assert policy.approved_pipeline_run_ids == frozenset({"run_a", "run_b"})
        assert policy.invalidated_pipeline_run_ids == frozenset({"run_bad"})
        assert ("QB", "passing_yards") in policy.invalidated_position_stats

    def test_validate_projection_trust_rejects_unapproved_rows(self):
        from ml.backtest import BacktestTrustPolicy, validate_projection_trust

        df = pd.DataFrame([{
            "player_id": "p1",
            "season": 2024,
            "week": 1,
            "position": "QB",
            "stat": "passing_yards",
            "projection": 245.0,
            "floor": 210.0,
            "ceiling": 280.0,
            "p25": 230.0,
            "p75": 260.0,
            "pipeline_run_id": "run_unknown",
            "posterior_samples": [240.0, 245.0, 250.0],
        }])
        policy = BacktestTrustPolicy(
            baseline_manifest_path="baseline.json",
            invalidation_manifest_path="invalidations.json",
            approved_pipeline_run_ids=frozenset({"run_ok"}),
            invalidated_pipeline_run_ids=frozenset(),
            invalidated_position_stats=frozenset(),
        )

        with pytest.raises(ValueError, match="not approved"):
            validate_projection_trust(df, policy, eval_season=2024)

    def test_validate_projection_trust_rejects_missing_posterior_samples(self):
        from ml.backtest import BacktestTrustPolicy, validate_projection_trust

        df = pd.DataFrame([{
            "player_id": "p1",
            "season": 2024,
            "week": 1,
            "position": "RB",
            "stat": "rushing_yards",
            "projection": 78.0,
            "floor": 50.0,
            "ceiling": 105.0,
            "p25": 65.0,
            "p75": 90.0,
            "pipeline_run_id": "run_ok",
            "posterior_samples": None,
        }])
        policy = BacktestTrustPolicy(
            baseline_manifest_path="baseline.json",
            invalidation_manifest_path="invalidations.json",
            approved_pipeline_run_ids=frozenset({"run_ok"}),
            invalidated_pipeline_run_ids=frozenset(),
            invalidated_position_stats=frozenset(),
        )

        with pytest.raises(ValueError, match="missing posterior_samples"):
            validate_projection_trust(df, policy, eval_season=2024)

    def test_validate_projection_trust_rejects_invalidated_target(self):
        from ml.backtest import BacktestTrustPolicy, validate_projection_trust

        df = pd.DataFrame([{
            "player_id": "p1",
            "season": 2024,
            "week": 1,
            "position": "QB",
            "stat": "passing_yards",
            "projection": 245.0,
            "floor": 210.0,
            "ceiling": 280.0,
            "p25": 230.0,
            "p75": 260.0,
            "pipeline_run_id": "run_ok",
            "posterior_samples": [240.0, 245.0, 250.0],
        }])
        policy = BacktestTrustPolicy(
            baseline_manifest_path="baseline.json",
            invalidation_manifest_path="invalidations.json",
            approved_pipeline_run_ids=frozenset({"run_ok"}),
            invalidated_pipeline_run_ids=frozenset(),
            invalidated_position_stats=frozenset({("QB", "passing_yards")}),
        )

        with pytest.raises(ValueError, match="explicitly invalidated"):
            validate_projection_trust(df, policy, eval_season=2024)

    def test_validate_projection_trust_accepts_approved_rows(self):
        from ml.backtest import BacktestTrustPolicy, validate_projection_trust

        df = pd.DataFrame([{
            "player_id": "p1",
            "season": 2024,
            "week": 1,
            "position": "WR",
            "stat": "receiving_yards",
            "projection": 78.0,
            "floor": 50.0,
            "ceiling": 105.0,
            "p25": 65.0,
            "p75": 90.0,
            "pipeline_run_id": "run_ok",
            "posterior_samples": [60.0, 80.0, 95.0],
        }])
        policy = BacktestTrustPolicy(
            baseline_manifest_path="baseline.json",
            invalidation_manifest_path="invalidations.json",
            approved_pipeline_run_ids=frozenset({"run_ok"}),
            invalidated_pipeline_run_ids=frozenset(),
            invalidated_position_stats=frozenset(),
        )

        validated = validate_projection_trust(df, policy, eval_season=2024)
        assert len(validated) == 1


# ---------------------------------------------------------------------------
# D. compute_crps_batch
# ---------------------------------------------------------------------------

class TestComputeCRPSBatch:

    def test_matches_mean_of_singles(self):
        from ml.backtest import compute_crps_single, compute_crps_batch
        rng = np.random.default_rng(4)
        samples_list = [rng.normal(50, 15, 100) for _ in range(20)]
        actuals = rng.normal(50, 20, 20)
        expected = float(np.mean([
            compute_crps_single(s, float(y))
            for s, y in zip(samples_list, actuals)
        ]))
        assert compute_crps_batch(samples_list, actuals) == pytest.approx(expected, rel=1e-9)

    def test_length_mismatch_raises(self):
        from ml.backtest import compute_crps_batch
        with pytest.raises(ValueError, match="length"):
            compute_crps_batch([np.array([1.0])], np.array([1.0, 2.0]))

    def test_all_perfect_gives_zero(self):
        from ml.backtest import compute_crps_batch
        actuals = np.array([10.0, 20.0, 30.0])
        samples_list = [np.full(100, a) for a in actuals]
        assert compute_crps_batch(samples_list, actuals) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# E. compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverage:

    def test_all_inside_gives_one(self):
        from ml.backtest import compute_coverage
        assert compute_coverage(
            floors=np.zeros(10), ceilings=np.full(10, 100.0), actuals=np.full(10, 50.0)
        ) == pytest.approx(1.0)

    def test_all_outside_gives_zero(self):
        from ml.backtest import compute_coverage
        assert compute_coverage(
            floors=np.full(10, 0.0), ceilings=np.full(10, 10.0), actuals=np.full(10, 50.0)
        ) == pytest.approx(0.0)

    def test_half_inside_gives_half(self):
        from ml.backtest import compute_coverage
        floors   = np.zeros(10)
        ceilings = np.full(10, 50.0)
        actuals  = np.array([25.0] * 5 + [75.0] * 5)  # 5 inside, 5 outside
        assert compute_coverage(floors, ceilings, actuals) == pytest.approx(0.5)

    def test_80pct_coverage_normal_distribution(self):
        """
        For X ~ N(mu, sigma), the interval [mu-1.28σ, mu+1.28σ] covers ~80%.
        Using 10k samples to reduce variance; tolerance ±3%.
        """
        from ml.backtest import compute_coverage
        rng = np.random.default_rng(5)
        mu, sigma = 80.0, 20.0
        n = 10_000
        actuals  = rng.normal(mu, sigma, n)
        floors   = np.full(n, mu - 1.2816 * sigma)
        ceilings = np.full(n, mu + 1.2816 * sigma)
        cov = compute_coverage(floors, ceilings, actuals)
        assert abs(cov - 0.80) < 0.03, f"Expected ~0.80 coverage, got {cov:.4f}"

    def test_result_between_zero_and_one(self):
        from ml.backtest import compute_coverage
        rng = np.random.default_rng(6)
        n = 50
        floors   = rng.uniform(0, 40, n)
        ceilings = floors + rng.uniform(10, 60, n)
        actuals  = rng.uniform(0, 100, n)
        cov = compute_coverage(floors, ceilings, actuals)
        assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# F. BacktestRunner.evaluate()
# ---------------------------------------------------------------------------

class TestBacktestRunnerEvaluate:

    @pytest.fixture
    def cohort(self):
        """100 synthetic player-games with known distribution."""
        rng = np.random.default_rng(10)
        n   = 100
        actuals     = rng.normal(70.0, 20.0, n)
        noise       = rng.normal(0.0, 10.0, n)
        projections = actuals + noise
        floors      = projections - 25.0
        ceilings    = projections + 25.0
        p25s        = projections - 13.0
        p75s        = projections + 13.0
        samples_list = [
            rng.normal(proj, 15.0, 50)
            for proj in projections
        ]
        naive   = actuals + rng.normal(0, 18.0, n)
        rolling = actuals + rng.normal(0, 14.0, n)
        return dict(
            actuals=actuals, projections=projections,
            floors=floors, ceilings=ceilings,
            p25s=p25s, p75s=p75s,
            posterior_samples_list=samples_list,
            naive_baselines=naive, rolling_baselines=rolling,
        )

    def test_returns_backtest_result(self, cohort):
        from ml.backtest import BacktestRunner, BacktestResult
        runner = BacktestRunner()
        result = runner.evaluate(**cohort, eval_season=2023, position="WR",
                                 stat="receiving_yards")
        assert isinstance(result, BacktestResult)

    def test_n_games_correct(self, cohort):
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert result.n_games == 100

    def test_coverage_80_between_0_and_1(self, cohort):
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert 0.0 <= result.coverage_80 <= 1.0

    def test_coverage_50_between_0_and_1(self, cohort):
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert 0.0 <= result.coverage_50 <= 1.0

    def test_baseline_improvement_pct_finite(self, cohort):
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert np.isfinite(result.baseline_improvement_pct)

    def test_stack_mae_positive(self, cohort):
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert result.stack_mae >= 0.0

    def test_stack_rmse_ge_stack_mae(self, cohort):
        """RMSE ≥ MAE always."""
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert result.stack_rmse >= result.stack_mae - 1e-9

    def test_stack_crps_nonneg(self, cohort):
        from ml.backtest import BacktestRunner
        result = BacktestRunner().evaluate(**cohort, eval_season=2023,
                                           position="WR", stat="receiving_yards")
        assert result.stack_crps >= -1e-9

    def test_better_predictions_lower_mae(self):
        """Stack with tighter noise should produce lower MAE."""
        from ml.backtest import BacktestRunner
        rng     = np.random.default_rng(11)
        n       = 200
        actuals = rng.normal(70.0, 20.0, n)
        samples = [rng.normal(70, 15, 50) for _ in actuals]

        def _evaluate(noise_std):
            proj  = actuals + rng.normal(0, noise_std, n)
            floor = proj - 25; ceiling = proj + 25
            p25   = proj - 13; p75     = proj + 13
            naive = actuals + rng.normal(0, 20, n)
            roll  = actuals + rng.normal(0, 18, n)
            return BacktestRunner().evaluate(
                actuals=actuals, projections=proj,
                floors=floor, ceilings=ceiling, p25s=p25, p75s=p75,
                posterior_samples_list=samples,
                naive_baselines=naive, rolling_baselines=roll,
                eval_season=2023, position="WR", stat="receiving_yards",
            )

        tight = _evaluate(5.0)
        noisy = _evaluate(30.0)
        assert tight.stack_mae < noisy.stack_mae

    def test_zero_naive_mae_gives_zero_improvement(self):
        """Guard: naive_mae=0 should not produce NaN/inf in improvement."""
        from ml.backtest import BacktestRunner
        n = 10
        actuals = np.zeros(n)
        proj    = np.full(n, 1.0)
        samples = [np.full(5, 1.0) for _ in range(n)]
        result = BacktestRunner().evaluate(
            actuals=actuals, projections=proj,
            floors=np.zeros(n), ceilings=np.full(n, 2.0),
            p25s=np.zeros(n), p75s=np.full(n, 1.5),
            posterior_samples_list=samples,
            naive_baselines=np.zeros(n),  # all zero → naive_mae = 1.0? No, naive=0, actual=0 → 0
            rolling_baselines=np.zeros(n),
            eval_season=2020, position="QB", stat="passing_yards",
        )
        assert np.isfinite(result.baseline_improvement_pct)


# ---------------------------------------------------------------------------
# G. BacktestRunner.run() — structural tests
# ---------------------------------------------------------------------------

class TestBacktestRunnerRun:

    def test_returns_dataframe(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        from ml.backtest import BacktestRunner, BacktestResult
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        for col in BacktestResult.__dataclass_fields__:
            assert col in df.columns, f"missing column: {col}"

    def test_one_row_per_season_position_stat(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2022, 2023], positions=["WR", "RB"],
            stats=["receiving_yards", "rushing_yards"],
        )
        # 2 seasons × 2 positions × 2 stats = 8 rows
        assert len(df) == 8

    def test_position_filter_respected(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["QB"], stats=["passing_yards"],
        )
        assert set(df["position"].unique()) == {"QB"}

    def test_coverage_80_between_0_and_1_all_rows(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        assert (df["coverage_80"] >= 0.0).all()
        assert (df["coverage_80"] <= 1.0).all()

    def test_baseline_improvement_pct_finite_all_rows(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        assert np.isfinite(df["baseline_improvement_pct"]).all()

    def test_empty_seasons_returns_empty_df(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[], positions=["WR"], stats=["receiving_yards"],
        )
        assert len(df) == 0

    def test_default_providers_raise(self):
        """Without injected providers, run() should raise NotImplementedError."""
        from ml.backtest import BacktestRunner
        runner = BacktestRunner()
        with pytest.raises(NotImplementedError):
            runner.run(seasons=[2023], positions=["WR"], stats=["receiving_yards"])


# ---------------------------------------------------------------------------
# H. save_results
# ---------------------------------------------------------------------------

class TestSaveResults:

    def test_creates_csv(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            BacktestRunner().save_results(df, path)
            assert Path(path).exists()

    def test_csv_has_correct_columns(self):
        from ml.backtest import BacktestRunner, BacktestResult
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            BacktestRunner().save_results(df, path)
            loaded = pd.read_csv(path)
        for col in BacktestResult.__dataclass_fields__:
            assert col in loaded.columns, f"missing column in CSV: {col}"

    def test_csv_row_count_matches_df(self):
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.csv")
            BacktestRunner().save_results(df, path)
            loaded = pd.read_csv(path)
        assert len(loaded) == len(df)

    def test_saves_to_nested_path(self):
        """Parent directories should be created automatically."""
        from ml.backtest import BacktestRunner
        dp, pp = _make_synthetic_providers()
        df = BacktestRunner(data_provider=dp, projection_provider=pp).run(
            seasons=[2023], positions=["WR"], stats=["receiving_yards"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "results.csv")
            BacktestRunner().save_results(df, path)
            assert Path(path).exists()


# ── Season Review position filter ───────────────────────────────────────────

class TestPositionFilterIsApplied:
    """
    `positions` used to be echoed into the response and applied to nothing:
    _load_csv filters on `stat` only, and with no backtest_results table in
    production the CSV path always ran. The Season Review chips were inert and
    the "Evaluated Sample Count" card summed every position regardless of the
    selection, so WR alone and WR+TE could not be reconciled.
    """

    @staticmethod
    def _svc():
        from backend.app.services.backtest import BacktestService

        svc = BacktestService.__new__(BacktestService)
        svc._model_version = "test"
        svc._db_url = ""
        return svc

    @staticmethod
    def _sample(summary, season: int = 2025) -> int:
        return sum(m.n_games for m in summary.by_season if m.season == season)

    def test_only_selected_positions_are_returned(self):
        summary = self._svc().get_summary(stat="receiving_yards", positions=["WR"])
        assert {m.position for m in summary.by_season} == {"WR"}

    def test_sample_counts_are_additive_across_positions(self):
        svc = self._svc()
        wr = self._sample(svc.get_summary(stat="receiving_yards", positions=["WR"]))
        te = self._sample(svc.get_summary(stat="receiving_yards", positions=["TE"]))
        both = self._sample(
            svc.get_summary(stat="receiving_yards", positions=["WR", "TE"])
        )
        assert wr > 0 and te > 0
        assert both == wr + te

    def test_overall_mae_responds_to_the_selection(self):
        svc = self._svc()
        wr = svc.get_summary(stat="receiving_yards", positions=["WR"]).overall_mae
        te = svc.get_summary(stat="receiving_yards", positions=["TE"]).overall_mae
        assert wr != te

    def test_overall_mae_is_weighted_by_sample_size(self):
        """
        A flat mean over (season, position) rows gave a 1,301-game TE season
        the same weight as a 2,500-game WR season. The combined figure must sit
        between the two single-position figures and nearer the larger sample.
        """
        svc = self._svc()
        wr = svc.get_summary(stat="receiving_yards", positions=["WR"]).overall_mae
        te = svc.get_summary(stat="receiving_yards", positions=["TE"]).overall_mae
        both = svc.get_summary(
            stat="receiving_yards", positions=["WR", "TE"]
        ).overall_mae
        assert min(wr, te) < both < max(wr, te)
        assert abs(both - wr) < abs(both - te)
