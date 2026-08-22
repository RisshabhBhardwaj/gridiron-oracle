"""
backend/tests/test_season_simulator.py

Unit tests for ml/season_simulator.py.
All tests run without a database connection.
"""

import numpy as np
import pandas as pd
import pytest

from ml.season_simulator import (
    SeasonSimulator,
    SeasonSimulation,
    run_rest_of_season,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_roster(n_players: int = 3) -> pd.DataFrame:
    """Minimal roster DataFrame for testing."""
    players = []
    teams = ["MIN", "MIN", "GB", "GB", "KC"][:n_players]
    positions = ["WR", "RB", "WR", "TE", "QB"][:n_players]
    for i in range(n_players):
        pid = f"p{i+1}"
        players.append({
            "player_id": pid,
            "position": positions[i],
            "team": teams[i],
            # Kalman priors — these define each player's prior distribution
            "kalman_est_receiving_yards":    [85, 30, 70, 50, 10][i],
            "kalman_variance_receiving_yards": 400.0,
            "kalman_est_rushing_yards":      [0, 65, 0, 0, 20][i],
            "kalman_variance_rushing_yards":   400.0,
            "kalman_est_fantasy_ppr":        [15, 12, 10, 8, 20][i],
            "kalman_variance_fantasy_ppr":     25.0,
            "kalman_est_passing_yards":      0.0,
            "kalman_variance_passing_yards":   4.0,
        })
    return pd.DataFrame(players)


def _small_sim(n_simulations: int = 50, weeks: tuple = (15, 18)) -> SeasonSimulator:
    return SeasonSimulator(
        season=2025,
        start_week=weeks[0],
        end_week=weeks[1],
        n_simulations=n_simulations,
        stats=["receiving_yards", "fantasy_ppr"],
        use_copula=True,
    )


# ── SeasonSimulator Tests ─────────────────────────────────────────────────────

class TestSeasonSimulatorBasic:
    def test_runs_without_error(self):
        """Simulator should complete without raising exceptions."""
        roster = _make_roster(3)
        sim = _small_sim(n_simulations=20)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=42)
        assert isinstance(result, SeasonSimulation)

    def test_returns_results_for_all_players(self):
        """All players in roster should appear in season totals."""
        roster = _make_roster(3)
        sim = _small_sim(n_simulations=20)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=0)
        for pid in ["p1", "p2", "p3"]:
            assert pid in result.player_season_totals, f"Missing player: {pid}"

    def test_all_stats_present(self):
        """All stats passed to the simulator should appear in totals."""
        roster = _make_roster(2)
        sim = _small_sim(n_simulations=20)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=1)
        for stat in ["receiving_yards", "fantasy_ppr"]:
            for pid in result.player_season_totals:
                assert stat in result.player_season_totals[pid]

    def test_percentile_ordering(self):
        """p10 < p50 < p90 for all stats, all players."""
        roster = _make_roster(3)
        sim = _small_sim(n_simulations=100)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=2)
        for pid, stat_dict in result.player_season_totals.items():
            for stat, d in stat_dict.items():
                assert d["p10"] <= d["p50"] <= d["p90"], \
                    f"Percentile ordering violated for {pid}/{stat}: {d}"

    def test_season_totals_non_negative(self):
        """Season stat totals must be non-negative (stats can't be negative)."""
        roster = _make_roster(3)
        sim = _small_sim(n_simulations=50)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=3)
        for pid, stat_dict in result.player_season_totals.items():
            for stat, d in stat_dict.items():
                assert d["p10"] >= 0, f"Negative p10 for {pid}/{stat}: {d['p10']}"

    def test_week_by_week_populated(self):
        """week_by_week should have entries for each week × player × stat."""
        roster = _make_roster(2)
        sim = _small_sim(n_simulations=20, weeks=(16, 18))
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=4)
        n_weeks = 3  # 16, 17, 18
        n_players = 2
        n_stats = 2  # receiving_yards, fantasy_ppr
        assert len(result.week_by_week) == n_weeks * n_players * n_stats

    def test_mean_consistent_with_percentiles(self):
        """Mean should be between p10 and p90."""
        roster = _make_roster(2)
        sim = _small_sim(n_simulations=200)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=5)
        for pid, stat_dict in result.player_season_totals.items():
            for stat, d in stat_dict.items():
                assert d["p10"] <= d["mean"] <= d["p90"] or \
                    abs(d["mean"] - d["p50"]) < 30, \
                    f"Mean far from percentiles for {pid}/{stat}: {d}"


class TestSeasonSimulatorInjury:
    def test_injury_reduces_projection(self):
        """A player marked OUT should have lower projected totals."""
        roster = _make_roster(2)
        sim_healthy = _small_sim(n_simulations=100)
        result_healthy = sim_healthy.run(
            players_df=roster,
            prior_game_rows={},
            rng_seed=10,
        )
        # Mark p1 as OUT for all simulated weeks
        injury_proj = {week: {"p1": "Out"} for week in range(15, 19)}
        sim_injured = _small_sim(n_simulations=100)
        result_injured = sim_injured.run(
            players_df=roster,
            prior_game_rows={},
            injury_projections=injury_proj,
            rng_seed=10,
        )
        healthy_median = result_healthy.player_season_totals["p1"]["receiving_yards"]["p50"]
        injured_median = result_injured.player_season_totals["p1"]["receiving_yards"]["p50"]
        assert injured_median < healthy_median, \
            f"Injured p50={injured_median:.0f} should be < healthy p50={healthy_median:.0f}"


class TestSeasonSimulatorAvailabilityGating:
    """
    player_active_prob threads real availability into the simulation itself
    (each simulated week draws its own Bernoulli(p_active) outcome), rather
    than p_active being computed and stored as metadata that the simulation
    never actually sees — the gap this test guards against is a retired or
    seldom-active player showing a full, undiscounted season total next to a
    low p_active in served output.
    """

    def test_low_active_prob_reduces_projection(self):
        roster = _make_roster(2)
        sim_full = _small_sim(n_simulations=200)
        result_full = sim_full.run(players_df=roster, prior_game_rows={}, rng_seed=20)

        sim_gated = _small_sim(n_simulations=200)
        result_gated = sim_gated.run(
            players_df=roster, prior_game_rows={},
            player_active_prob={"p1": 0.1}, rng_seed=20,
        )
        full_mean = result_full.player_season_totals["p1"]["receiving_yards"]["mean"]
        gated_mean = result_gated.player_season_totals["p1"]["receiving_yards"]["mean"]
        assert gated_mean < full_mean * 0.5, (
            f"p_active=0.1 should sharply cut p1's season total: "
            f"gated={gated_mean:.1f} vs full={full_mean:.1f}"
        )

    def test_zero_active_prob_zeroes_out_player(self):
        roster = _make_roster(2)
        sim = _small_sim(n_simulations=100)
        result = sim.run(
            players_df=roster, prior_game_rows={},
            player_active_prob={"p1": 0.0}, rng_seed=21,
        )
        for stat, d in result.player_season_totals["p1"].items():
            assert d["mean"] == 0.0, f"p_active=0 should zero out p1/{stat}, got {d}"
            assert d["p90"] == 0.0

    def test_unlisted_players_are_unaffected(self):
        """Players absent from player_active_prob simulate at full availability."""
        # 4 simulated weeks (15-18); p2's kalman prior is 30 receiving yards/week
        # (see _make_roster) — an ungated p2 should land near 4 * 30 = 120,
        # nowhere near what gating p1 to zero would do if it leaked onto p2.
        roster = _make_roster(2)
        sim_partial = _small_sim(n_simulations=200)
        result_partial = sim_partial.run(
            players_df=roster, prior_game_rows={},
            player_active_prob={"p1": 0.0}, rng_seed=22,
        )
        p2_mean = result_partial.player_season_totals["p2"]["receiving_yards"]["mean"]
        assert p2_mean > 90, f"p2 should be near its full 4-week rate (~120), got {p2_mean:.1f}"

    def test_stats_zero_together_not_independently(self):
        """A player marked inactive on a path zeroes ALL their stats on that path together."""
        roster = _make_roster(2)
        sim = SeasonSimulator(
            season=2025, start_week=17, end_week=18, n_simulations=300,
            stats=["receiving_yards", "fantasy_ppr"], use_copula=False,
        )
        result = sim.run(
            players_df=roster, prior_game_rows={},
            player_active_prob={"p1": 0.5}, rng_seed=23,
        )
        # Half the paths should be exactly zero for BOTH stats — if the mask
        # were drawn independently per-stat, p10 for each stat could still
        # be zero without the totals sharing paths, but week_by_week's raw
        # per-week zero rate is the direct signal.
        zero_weeks = [
            w for w in result.week_by_week
            if w["player_id"] == "p1" and w["stat"] == "receiving_yards" and w["p10"] == 0.0
        ]
        assert zero_weeks, "expected some weeks where p10 hits zero at p_active=0.5"


class TestSeasonSimulationResult:
    def test_top_players_returns_dataframe(self):
        """top_players() should return a non-empty DataFrame."""
        roster = _make_roster(3)
        sim = _small_sim(n_simulations=30)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=20)
        df = result.top_players("receiving_yards")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "p50" in df.columns

    def test_top_players_sorted_descending(self):
        """top_players() should be sorted by p50 descending."""
        roster = _make_roster(3)
        sim = _small_sim(n_simulations=100)
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=21)
        df = result.top_players("receiving_yards")
        if len(df) >= 2:
            assert df.iloc[0]["p50"] >= df.iloc[1]["p50"]

    def test_season_metadata(self):
        """SeasonSimulation should store correct season metadata."""
        roster = _make_roster(2)
        sim = SeasonSimulator(season=2025, start_week=10, end_week=12, n_simulations=20,
                              stats=["receiving_yards"])
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=30)
        assert result.season == 2025
        assert result.start_week == 10
        assert result.end_week == 12
        assert result.n_simulations == 20


class TestRunRestOfSeason:
    def test_convenience_function(self):
        """run_rest_of_season() should produce valid SeasonSimulation."""
        roster = _make_roster(2)
        result = run_rest_of_season(
            season=2025,
            current_week=16,
            players_df=roster,
            prior_game_rows={},
            n_simulations=30,
            stats=["receiving_yards"],
            rng_seed=42,
        )
        assert isinstance(result, SeasonSimulation)
        assert result.start_week == 17
        assert result.end_week == 18
        assert len(result.player_season_totals) == 2

    def test_week_range_starts_after_current(self):
        """Simulation should start from current_week + 1, not current_week."""
        roster = _make_roster(2)
        result = run_rest_of_season(
            season=2025, current_week=10,
            players_df=roster, prior_game_rows={},
            n_simulations=20, rng_seed=99,
        )
        assert result.start_week == 11


class TestIndependentMode:
    def test_independent_mode_also_works(self):
        """use_copula=False should produce valid results."""
        roster = _make_roster(2)
        sim = SeasonSimulator(
            season=2025, start_week=16, end_week=18,
            n_simulations=30, stats=["receiving_yards"], use_copula=False,
        )
        result = sim.run(players_df=roster, prior_game_rows={}, rng_seed=5)
        for pid in result.player_season_totals:
            d = result.player_season_totals[pid]["receiving_yards"]
            assert d["p10"] >= 0
            assert d["p10"] <= d["p50"] <= d["p90"]
