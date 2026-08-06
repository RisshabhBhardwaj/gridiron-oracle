"""
backend/tests/test_kalman_tracker.py

Tests for ml/kalman_tracker.py.

Structure:
  TestKalmanTracker         -- unit tests for KalmanTracker class
  TestKalmanFeatureEngineer -- integration tests for KalmanFeatureEngineer
  TestSchemaSync            -- every FeatureRow field must exist in FeatureMatrix

Run with:
  pytest backend/tests/test_kalman_tracker.py -v
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.kalman_tracker import (
    DEFAULT_Q,
    INITIAL_VARIANCE,
    KALMAN_STATS,
    POSITION_PRIORS,
    KalmanFeatureEngineer,
    KalmanTracker,
    _MIN_R,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(Q: float = DEFAULT_Q, x0: float = 0.0, R: float = 100.0) -> KalmanTracker:
    t = KalmanTracker(Q=Q, x0=x0)
    t.R = R
    return t


def _stat_rows(values: list[float], stat: str = "receiving_yards") -> list[dict]:
    """Build minimal prior_row dicts with one stat column."""
    return [{stat: v} for v in values]


# ---------------------------------------------------------------------------
# TestKalmanTracker
# ---------------------------------------------------------------------------

class TestKalmanTracker:

    # ── predict_sequence structure ──────────────────────────────────────────

    def test_returns_list_of_tuples(self):
        t = _make_tracker()
        result = t.predict_sequence([50.0, 60.0, 70.0])
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_length_matches_observations(self):
        t = _make_tracker()
        for n in [0, 1, 3, 8]:
            result = t.predict_sequence([50.0] * n)
            assert len(result) == n, f"Expected {n} results, got {len(result)}"

    def test_empty_observations_returns_empty(self):
        t = _make_tracker()
        assert t.predict_sequence([]) == []

    def test_raises_without_fit(self):
        t = KalmanTracker()
        with pytest.raises(RuntimeError, match="fit"):
            t.predict_sequence([50.0])

    # ── single observation ──────────────────────────────────────────────────

    def test_single_obs_posterior_between_prior_and_obs(self):
        """With x0=0, y=100, posterior mean must be strictly between 0 and 100."""
        t = _make_tracker(Q=1.0, x0=0.0, R=50.0)
        (mean, var), = t.predict_sequence([100.0])
        assert 0.0 < mean < 100.0

    def test_single_obs_variance_less_than_prior(self):
        t = _make_tracker(x0=0.0, R=50.0)
        (_, var), = t.predict_sequence([100.0])
        assert var < INITIAL_VARIANCE

    # ── variance test (monotonically decreasing) ────────────────────────────

    def test_variance_decreases_monotonically(self):
        """
        Posterior variance should decrease as more observations arrive
        (the filter gains confidence). Tested over 12 observations.
        """
        t = _make_tracker(Q=0.1, x0=60.0, R=200.0)
        obs = [60.0 + float(np.random.default_rng(42).normal(0, 5)) for _ in range(12)]
        results = t.predict_sequence(obs)
        variances = [v for _, v in results]
        for i in range(1, len(variances)):
            assert variances[i] < variances[i - 1], (
                f"Variance increased at step {i}: {variances[i-1]:.4f} → {variances[i]:.4f}"
            )

    def test_variance_strictly_positive(self):
        t = _make_tracker(Q=1.0, x0=0.0, R=10.0)
        results = t.predict_sequence([50.0] * 20)
        assert all(v > 0 for _, v in results)

    # ── convergence test ────────────────────────────────────────────────────

    def test_convergence_within_10_pct_after_8_games(self):
        """
        After 8+ weeks of stable data, the Kalman estimate should be within
        10% of the true mean (convergence test from spec).
        """
        true_mean = 80.0
        rng = np.random.default_rng(7)
        obs = [true_mean + float(rng.normal(0, 10)) for _ in range(12)]

        t = KalmanTracker(Q=1.0, x0=50.0)
        t.R = float(np.var(obs, ddof=1))
        results = t.predict_sequence(obs)
        est_after_8, _ = results[7]   # posterior after 8th observation

        assert abs(est_after_8 - true_mean) / true_mean < 0.10, (
            f"Kalman estimate {est_after_8:.2f} is not within 10% of true mean {true_mean}"
        )

    # ── breakout test ───────────────────────────────────────────────────────

    def test_breakout_delta_exceeds_rolling_delta(self):
        """
        A player with 5 consistent average weeks followed by a breakout week
        should update the Kalman posterior MORE than a 4-week rolling average.

        Why this works: 5 identical observations yield sample variance = 0,
        so R falls back to _MIN_R (very low). This makes the Kalman gain K ≈ 1,
        meaning the filter nearly fully adopts the breakout observation.
        The rolling average only gives the latest week 40% weight.
        """
        avg_yards = 60.0
        breakout  = 150.0
        n_avg     = 5

        prior_obs  = [avg_yards] * n_avg   # 5 consistent weeks
        obs_all    = prior_obs + [breakout]

        # ── Kalman delta ──────────────────────────────────────────────────
        # R from 5 identical values = 0 → falls back to _MIN_R → K ≈ 1
        raw_vals = [avg_yards] * n_avg
        r_est = float(np.var(raw_vals, ddof=1))
        R = max(r_est, _MIN_R)

        t = KalmanTracker(Q=1.0, x0=avg_yards)
        t.R = R
        results = t.predict_sequence(obs_all)
        est_before = results[n_avg - 1][0]   # after 5 prior weeks
        est_after  = results[n_avg][0]        # after breakout
        kalman_delta = abs(est_after - est_before)

        # ── Rolling 4-week delta ──────────────────────────────────────────
        FORM_WEIGHTS = [0.1, 0.2, 0.3, 0.4]

        def rolling_avg(vals):
            non_none = [v for v in vals if v is not None]
            n = min(len(non_none), 4)
            recent = non_none[-n:]
            w = FORM_WEIGHTS[-n:]
            return sum(wi * vi for wi, vi in zip(w, recent)) / sum(w)

        roll_before = rolling_avg(prior_obs)
        roll_after  = rolling_avg(obs_all)
        rolling_delta = abs(roll_after - roll_before)

        assert kalman_delta > rolling_delta, (
            f"Expected Kalman delta ({kalman_delta:.2f}) > rolling delta ({rolling_delta:.2f})"
        )

    # ── causality test ──────────────────────────────────────────────────────

    def test_causality_prefix_independence(self):
        """
        Adding future observations must not retroactively change past estimates.
        predict_sequence([y1..yk]) and predict_sequence([y1..yk, y_{k+1}])
        must agree on the first k results.
        """
        t = _make_tracker(Q=1.0, x0=50.0, R=100.0)
        obs = [55.0, 65.0, 60.0, 70.0, 80.0]

        results_short = t.predict_sequence(obs[:3])
        results_long  = t.predict_sequence(obs)

        for i, ((m_s, v_s), (m_l, v_l)) in enumerate(zip(results_short, results_long)):
            assert m_s == pytest.approx(m_l, rel=1e-10), (
                f"Mean changed at position {i} when future obs added: {m_s} vs {m_l}"
            )
            assert v_s == pytest.approx(v_l, rel=1e-10), (
                f"Var changed at position {i} when future obs added: {v_s} vs {v_l}"
            )

    # ── fit() ───────────────────────────────────────────────────────────────

    def test_fit_estimates_r_from_variance(self):
        data = pd.DataFrame({"receiving_yards": [50.0, 60.0, 70.0, 80.0]})
        t = KalmanTracker()
        t.fit(data, "receiving_yards")
        expected_r = float(np.var([50, 60, 70, 80], ddof=1))
        assert t.R == pytest.approx(expected_r, rel=1e-9)

    def test_fit_fallback_for_single_row(self):
        data = pd.DataFrame({"receiving_yards": [60.0]})
        t = KalmanTracker()
        t.fit(data, "receiving_yards")
        assert t.R == 100.0

    def test_fit_fallback_for_missing_column(self):
        data = pd.DataFrame({"other_col": [1.0, 2.0]})
        t = KalmanTracker()
        t.fit(data, "receiving_yards")
        assert t.R == 100.0

    def test_fit_minimum_r_enforced(self):
        """Perfectly constant player should not get R=0."""
        data = pd.DataFrame({"receiving_yards": [60.0, 60.0, 60.0, 60.0]})
        t = KalmanTracker()
        t.fit(data, "receiving_yards")
        assert t.R >= _MIN_R

    def test_fit_returns_self(self):
        data = pd.DataFrame({"receiving_yards": [50.0, 60.0]})
        t = KalmanTracker()
        result = t.fit(data, "receiving_yards")
        assert result is t


# ---------------------------------------------------------------------------
# TestKalmanFeatureEngineer
# ---------------------------------------------------------------------------

class TestKalmanFeatureEngineer:

    # ── cold-start (0 prior rows) ───────────────────────────────────────────

    def test_cold_start_returns_position_prior_for_wr(self):
        """With 0 prior rows and position=WR, kalman_est_* = WR position prior."""
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form([], position="WR")

        wr_priors = POSITION_PRIORS["WR"]
        for stat in KALMAN_STATS:
            assert result[f"kalman_est_{stat}"] == pytest.approx(wr_priors[stat]), (
                f"Cold-start WR: kalman_est_{stat} should be {wr_priors[stat]}"
            )

    def test_cold_start_variance_is_initial_variance(self):
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form([], position="WR")
        for stat in KALMAN_STATS:
            assert result[f"kalman_variance_{stat}"] == pytest.approx(INITIAL_VARIANCE)

    def test_cold_start_unknown_position_uses_zero_prior(self):
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form([], position="K")  # kicker — not in priors
        # Falls back to _DEFAULT_PRIOR (all zeros)
        assert result["kalman_est_receiving_yards"] == pytest.approx(0.0)

    # ── output schema ──────────────────────────────────────────────────────

    def test_output_has_est_and_variance_for_every_stat(self):
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form([], position="WR")
        for stat in KALMAN_STATS:
            assert f"kalman_est_{stat}" in result, f"Missing kalman_est_{stat}"
            assert f"kalman_variance_{stat}" in result, f"Missing kalman_variance_{stat}"

    def test_total_output_keys(self):
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form([], position="WR")
        assert len(result) == 2 * len(KALMAN_STATS)

    # ── with prior rows ─────────────────────────────────────────────────────

    def test_prior_rows_update_estimate(self):
        """After observations, kalman_est should differ from the cold-start prior."""
        prior_rows = [{"receiving_yards": 120.0} for _ in range(5)]
        eng = KalmanFeatureEngineer()
        cold = eng.compute_kalman_form([], position="WR")
        warm = eng.compute_kalman_form(prior_rows, position="WR")

        # After 5 observations of 120, estimate should be pulled toward 120
        assert warm["kalman_est_receiving_yards"] > cold["kalman_est_receiving_yards"]

    def test_variance_lower_after_observations(self):
        """Posterior variance after observations must be below INITIAL_VARIANCE."""
        prior_rows = [{"receiving_yards": 80.0} for _ in range(6)]
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form(prior_rows, position="WR")
        assert result["kalman_variance_receiving_yards"] < INITIAL_VARIANCE

    def test_none_stat_values_handled_gracefully(self):
        """None values in prior_rows must not crash — replaced by position prior."""
        prior_rows = [{"receiving_yards": None}, {"receiving_yards": 90.0}]
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form(prior_rows, position="WR")
        assert result["kalman_est_receiving_yards"] is not None

    def test_all_none_falls_back_to_cold_start_estimate(self):
        """If all stat values are None, estimate should match the position prior."""
        prior_rows = [{"receiving_yards": None}] * 4
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form(prior_rows, position="WR")
        # All None → effectively a cold-start (R=100, observations = all x0)
        # The posterior should still be close to the WR prior
        wr_prior = POSITION_PRIORS["WR"]["receiving_yards"]
        assert abs(result["kalman_est_receiving_yards"] - wr_prior) < wr_prior * 0.5

    def test_stats_are_independent(self):
        """receiving_yards observations must not contaminate rushing_yards estimate."""
        prior_rows = [{"receiving_yards": 200.0}] * 8   # extreme receiving game
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form(prior_rows, position="WR")

        # rushing_yards estimate for a WR should stay near the WR prior (2.0),
        # not be inflated by receiving_yards observations
        wr_rush_prior = POSITION_PRIORS["WR"]["rushing_yards"]
        assert result["kalman_est_rushing_yards"] == pytest.approx(wr_rush_prior, rel=0.5)

    def test_rb_prior_differs_from_wr_prior(self):
        """Position priors must differ — RB rushes more than WR."""
        eng = KalmanFeatureEngineer()
        wr = eng.compute_kalman_form([], position="WR")
        rb = eng.compute_kalman_form([], position="RB")
        assert rb["kalman_est_rushing_yards"] > wr["kalman_est_rushing_yards"]
        assert wr["kalman_est_receiving_yards"] > rb["kalman_est_receiving_yards"]


# ---------------------------------------------------------------------------
# TestSchemaSync — every FeatureRow field must exist in FeatureMatrix
# ---------------------------------------------------------------------------

_sqlmodel = pytest.importorskip  # alias for clarity


class TestSchemaSync:

    def test_feature_row_fields_in_feature_matrix(self):
        """
        Every field in FeatureRow (pipeline/feature_engineer.py) must exist
        in FeatureMatrix (backend/app/models/production.py) as a column.

        Skipped when sqlmodel is not installed (Docker-only dependency).
        """
        sqlmodel = pytest.importorskip("sqlmodel", reason="sqlmodel not installed")
        from pipeline.feature_engineer import FeatureRow
        from backend.app.models.production import FeatureMatrix

        fr_fields = {f.name for f in dataclasses.fields(FeatureRow)}
        fm_columns = {col.name for col in FeatureMatrix.__table__.columns}

        missing = fr_fields - fm_columns
        assert not missing, (
            f"FeatureRow fields missing from FeatureMatrix: {sorted(missing)}\n"
            "Update backend/app/models/production.py to add these columns."
        )

    def test_kalman_fields_present_in_feature_row(self):
        """All kalman_est_* and kalman_variance_* fields must be in FeatureRow."""
        from pipeline.feature_engineer import FeatureRow
        fr_fields = {f.name for f in dataclasses.fields(FeatureRow)}

        for stat in KALMAN_STATS:
            assert f"kalman_est_{stat}" in fr_fields, (
                f"kalman_est_{stat} missing from FeatureRow"
            )
            assert f"kalman_variance_{stat}" in fr_fields, (
                f"kalman_variance_{stat} missing from FeatureRow"
            )

    def test_no_form_fields_in_feature_row(self):
        """form_* fields must be completely removed from FeatureRow."""
        from pipeline.feature_engineer import FeatureRow
        fr_fields = {f.name for f in dataclasses.fields(FeatureRow)}
        form_fields = [f for f in fr_fields if f.startswith("form_")]
        assert not form_fields, (
            f"Stale form_* fields still in FeatureRow: {sorted(form_fields)}"
        )

    def test_no_form_fields_in_feature_matrix(self):
        """form_* columns must be completely removed from FeatureMatrix. Skipped without sqlmodel."""
        pytest.importorskip("sqlmodel", reason="sqlmodel not installed")
        from backend.app.models.production import FeatureMatrix
        fm_columns = {col.name for col in FeatureMatrix.__table__.columns}
        form_cols = [c for c in fm_columns if c.startswith("form_")]
        assert not form_cols, (
            f"Stale form_* columns still in FeatureMatrix: {sorted(form_cols)}"
        )


# ---------------------------------------------------------------------------
# TestKalmanProperty — Hypothesis property-based tests
# ---------------------------------------------------------------------------

try:
    from hypothesis import given, settings as h_settings, assume
    from hypothesis import strategies as st
    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

_requires_hypothesis = pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed",
)


@_requires_hypothesis
class TestKalmanProperty:
    """
    Property-based tests for KalmanTracker using Hypothesis.

    These tests exercise edge cases that hand-written unit tests miss:
      - Arbitrary Q/R ratios (including near-zero and very large)
      - Sequences of any length (empty, single, long)
      - Observations containing NaN-replacement values (position prior fill)
      - Monotonicity and variance bounds hold for ALL valid inputs
    """

    @_requires_hypothesis
    @given(
        observations=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=0, max_size=30),
        Q=st.floats(min_value=1e-4, max_value=1e4, allow_nan=False),
        R=st.floats(min_value=1e-4, max_value=1e4, allow_nan=False),
        x0=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False),
    )
    @h_settings(max_examples=200, deadline=500)
    def test_output_length_always_matches_input(self, observations, Q, R, x0):
        """len(predict_sequence(obs)) == len(obs) for any valid input."""
        t = KalmanTracker(Q=Q, x0=x0)
        t.R = R
        results = t.predict_sequence(observations)
        assert len(results) == len(observations)

    @_requires_hypothesis
    @given(
        observations=st.lists(st.floats(min_value=0.0, max_value=300.0, allow_nan=False), min_size=1, max_size=20),
        Q=st.floats(min_value=1e-4, max_value=100.0, allow_nan=False),
        R=st.floats(min_value=1e-4, max_value=1000.0, allow_nan=False),
    )
    @h_settings(max_examples=200, deadline=500)
    def test_variance_always_positive(self, observations, Q, R):
        """Posterior variance must be strictly positive for any Q, R, observations."""
        t = KalmanTracker(Q=Q, x0=0.0)
        t.R = R
        results = t.predict_sequence(observations)
        for i, (_, var) in enumerate(results):
            assert var > 0, f"Non-positive variance {var} at step {i}"

    @_requires_hypothesis
    @given(
        n=st.integers(min_value=2, max_value=20),
        obs_val=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
        Q=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False),
        R=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False),
    )
    @h_settings(max_examples=150, deadline=500)
    def test_variance_non_increasing_with_small_Q(self, n, obs_val, Q, R):
        """
        When Q is small (ability drifts slowly), posterior variance should be
        non-increasing over a run of identical observations. With large Q the
        process noise re-inflates variance each step, so we only test small Q.
        """
        assume(Q < R / 10)  # Q small relative to R → filter gains confidence
        t = KalmanTracker(Q=Q, x0=obs_val)
        t.R = R
        results = t.predict_sequence([obs_val] * n)
        variances = [v for _, v in results]
        for i in range(1, len(variances)):
            assert variances[i] <= variances[i - 1] + 1e-9, (
                f"Variance increased at step {i}: {variances[i-1]:.6f} → {variances[i]:.6f}"
            )

    @_requires_hypothesis
    @given(
        prior_vals=st.lists(st.floats(min_value=0.0, max_value=200.0, allow_nan=False), min_size=0, max_size=15),
        position=st.sampled_from(["QB", "WR", "RB", "TE"]),
    )
    @h_settings(max_examples=150, deadline=1000)
    def test_feature_engineer_never_crashes(self, prior_vals, position):
        """compute_kalman_form must not raise for any valid prior_rows / position."""
        prior_rows = [{"receiving_yards": v, "rushing_yards": v * 0.1} for v in prior_vals]
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form(prior_rows, position=position)
        assert len(result) == 2 * len(KALMAN_STATS)

    @_requires_hypothesis
    @given(
        prior_rows=st.lists(
            st.fixed_dictionaries({"receiving_yards": st.none()}),
            min_size=1, max_size=10,
        ),
        position=st.sampled_from(["QB", "WR", "RB", "TE"]),
    )
    @h_settings(max_examples=100, deadline=500)
    def test_all_none_rows_never_crashes(self, prior_rows, position):
        """All-None prior rows must fall back gracefully to cold-start, not raise."""
        eng = KalmanFeatureEngineer()
        result = eng.compute_kalman_form(prior_rows, position=position)
        assert result["kalman_est_receiving_yards"] is not None
