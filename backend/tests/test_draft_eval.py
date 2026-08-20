"""Component-level draft evaluation and the fitted rookie prior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.draft_eval import (
    within_position_precision,
    bootstrap_ci,
    calibration_ratio,
    evaluate_component,
    paired_difference,
    pooled_spearman,
    season_level_ci,
    top_n_precision,
)
from ml.rookie_priors import capital_bin, fit_rookie_curve, rookie_rate_and_games


def _frame(n_seasons: int = 6, n_players: int = 60, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2020, 2020 + n_seasons):
        for i in range(n_players):
            true = 200.0 - 3.0 * i
            rows.append({
                "season": season,
                "player_id": f"p{i}",
                "position": ["QB", "RB", "WR", "TE"][i % 4],
                "realized": true,
                "projection": true + rng.normal(scale=noise) if noise else true,
                "adp": float(i + 1),
                "is_rookie": i % 10 == 0,
            })
    return pd.DataFrame(rows)


class TestMetrics:
    def test_perfect_projection_scores_one(self):
        f = _frame()
        assert pooled_spearman(f, "projection") == pytest.approx(1.0)
        assert top_n_precision(f, "projection") == pytest.approx(1.0)
        assert calibration_ratio(f, "projection") == pytest.approx(1.0)

    def test_adp_orientation_is_handled(self):
        f = _frame()
        assert pooled_spearman(f, "adp", higher_is_better=False) == pytest.approx(1.0)
        assert top_n_precision(f, "adp", higher_is_better=False) == pytest.approx(1.0)

    def test_calibration_ratio_detects_systematic_underprojection(self):
        f = _frame()
        f["projection"] = f["realized"] * 0.24  # the rookie bug's magnitude
        assert calibration_ratio(f, "projection") == pytest.approx(0.24)

    def test_noise_degrades_rank_correlation(self):
        clean = pooled_spearman(_frame(noise=0.0), "projection")
        noisy = pooled_spearman(_frame(noise=90.0, seed=3), "projection")
        assert noisy < clean


class TestIntervalsAreValid:
    """A confidence interval that excludes its own estimate is a bug, not a wide interval.

    ``top_n_precision`` is a per-season statistic. Under a cluster bootstrap a
    single player can be resampled into several of the 24 slots, so the metric
    stops measuring what it names and the interval drifts off the estimate.
    """

    def test_cluster_bootstrap_ci_brackets_the_estimate(self):
        f = _frame(noise=40.0, seed=1)
        est, lo, hi = bootstrap_ci(f, lambda d: pooled_spearman(d, "projection"), n_boot=200)
        assert lo <= est <= hi

    def test_top_n_precision_ci_brackets_the_estimate(self):
        f = _frame(noise=40.0, seed=2)
        est, lo, hi = season_level_ci(f, lambda d: top_n_precision(d, "projection"))
        assert lo <= est <= hi

    def test_evaluate_component_reports_bracketing_intervals(self):
        out = evaluate_component(_frame(noise=40.0, seed=4), "projection", n_boot=200)
        for key in ("spearman", "within_position_precision", "calibration_ratio"):
            block = out[key]
            assert block["lo"] <= block["est"] <= block["hi"], key


class TestPairedDifference:
    def test_identical_columns_are_not_significant(self):
        f = _frame(noise=30.0, seed=5)
        f["copy"] = f["projection"]
        result = paired_difference(f, "projection", "copy", pooled_spearman, n_boot=200)
        assert result["diff"] == pytest.approx(0.0)
        assert not result["significant"]

    def test_a_real_improvement_is_detected(self):
        f = _frame(noise=0.0, seed=6)
        rng = np.random.default_rng(7)
        f["worse"] = f["realized"] + rng.normal(scale=200.0, size=len(f))
        result = paired_difference(f, "projection", "worse", pooled_spearman, n_boot=300)
        assert result["diff"] > 0
        assert result["significant"]


class TestRookieCurve:
    def _logs(self):
        rows = []
        # Better draft capital -> more points, the relationship the curve fits.
        for season in (2019, 2020, 2021):
            for i in range(60):
                per_game = 14.0 - 0.18 * i
                for week in range(1, 15):
                    rows.append({
                        "player_id": f"r{season}_{i}",
                        "season": season,
                        "fantasy_points_ppr": per_game,
                    })
        return pd.DataFrame(rows)

    def _players(self):
        rows = []
        for season in (2019, 2020, 2021):
            for i in range(60):
                rows.append({
                    "id": f"r{season}_{i}",
                    "position": ["QB", "RB", "WR", "TE"][i % 4],
                    "entry_year": season,
                    "draft_number": i * 4 + 1,
                })
        return pd.DataFrame(rows)

    def test_bins_are_closed_at_their_upper_edge(self):
        assert capital_bin(1) == "1-15"
        assert capital_bin(15) == "1-15"
        assert capital_bin(16) == "16-32"
        assert capital_bin(263) == "undrafted"

    def test_missing_capital_is_treated_as_undrafted(self):
        assert capital_bin(None) == "undrafted"
        assert capital_bin(float("nan")) == "undrafted"
        assert capital_bin("not a pick") == "undrafted"

    def test_refuses_target_season_rows(self):
        with pytest.raises(ValueError, match="leakage"):
            fit_rookie_curve(self._logs(), self._players(), max_season=2020)

    def test_earlier_picks_project_higher(self):
        curve = fit_rookie_curve(self._logs(), self._players(), max_season=2021)
        early, _ = rookie_rate_and_games(curve, "RB", 3)
        late, _ = rookie_rate_and_games(curve, "RB", 200)
        assert early > late

    def test_empty_curve_falls_back_without_raising(self):
        rate, games = rookie_rate_and_games({}, "WR", 12)
        assert rate == 0.0 and games == pytest.approx(14.0)

    def test_vacated_share_moves_the_rate_within_bounds(self):
        curve = fit_rookie_curve(self._logs(), self._players(), max_season=2021)
        low, _ = rookie_rate_and_games(curve, "WR", 20, vacated_share=0.0)
        high, _ = rookie_rate_and_games(curve, "WR", 20, vacated_share=1.0)
        assert high > low
        assert high / low == pytest.approx(1.15 / 0.85, rel=1e-6)


class TestPrecisionIsPositionAware:
    """Third occurrence of the illegal-oracle bug; this one reversed a conclusion.

    A cross-position "top 24 by realized points" oracle is 9-11 quarterbacks in
    a 1-QB league. Measured 2026-08-20 on the real board, the blind metric said
    the model beat ADP 0.451 to 0.396 while the position-aware metric said ADP
    led 0.556 to 0.486 -- same projections, opposite sign.
    """

    def _qb_heavy_universe(self) -> pd.DataFrame:
        rows = []
        for season in (2020, 2021, 2022):
            for i in range(12):
                rows.append({"season": season, "player_id": f"qb{i}", "position": "QB",
                             "realized": 320.0 - i, "qb_first": 1000.0 - i, "balanced": 100.0 - i})
            for pos, base in (("RB", 240.0), ("WR", 230.0), ("TE", 170.0)):
                for i in range(20):
                    rows.append({"season": season, "player_id": f"{pos}{i}", "position": pos,
                                 "realized": base - 2 * i, "qb_first": 10.0 - 0.1 * i,
                                 "balanced": base - 2 * i})
        return pd.DataFrame(rows)

    def test_blind_metric_rewards_a_quarterback_pile(self):
        f = self._qb_heavy_universe()
        assert top_n_precision(f, "qb_first") > top_n_precision(f, "balanced")

    def test_position_aware_metric_is_blind_to_cross_position_weighting(self):
        """Both boards order players identically *inside* each position.

        They differ only in how they price quarterbacks against everyone else,
        which is exactly the axis a positional metric must ignore. The blind
        metric reads a difference here; the position-aware one must not.
        """
        f = self._qb_heavy_universe()
        assert within_position_precision(f, "qb_first") == pytest.approx(
            within_position_precision(f, "balanced")
        )

    def test_position_aware_metric_is_perfect_on_a_perfect_board(self):
        f = self._qb_heavy_universe()
        f["perfect"] = f["realized"]
        assert within_position_precision(f, "perfect") == pytest.approx(1.0)

    def test_evaluate_component_reports_the_position_aware_metric(self):
        out = evaluate_component(_frame(noise=40.0, seed=8), "projection", n_boot=150)
        assert "within_position_precision" in out
        assert "warning" in out["top24_precision_position_blind"]
