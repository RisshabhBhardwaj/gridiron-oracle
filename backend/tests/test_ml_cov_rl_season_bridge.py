"""
backend/tests/test_ml_cov_rl_season_bridge.py

Coverage tests for:
  - ml/rl_hedging_agent.py      (Gymnasium hedging environment; RL libs not installed)
  - ml/season_simulator_bridge.py (C++ fast path with Python fallback)

The RL environment is exercised by bypassing __init__ (the constructor needs
gymnasium.spaces, which is absent here) and checking the step arithmetic
against hand-computed values. The C++ library is not built here, so the
bridge's fallback path is the one that runs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import ml.rl_hedging_agent as rl
from ml import season_simulator_bridge as bridge
from ml.rl_hedging_agent import MAX_BET_FRACTION, HedgingAgent, HedgingEnv


# ── rl_hedging_agent ──────────────────────────────────────────────────────────

def _bare_env(bankroll: float = 10000.0, max_steps: int = 100) -> HedgingEnv:
    """Build the env without __init__ (which needs gymnasium.spaces)."""
    env = HedgingEnv.__new__(HedgingEnv)
    env.initial_bankroll = bankroll
    env.max_steps = max_steps
    env.current_step = 0
    env.bankroll = bankroll
    env.exposure = 0.0
    return env


def _fixed_obs(prob: float) -> np.ndarray:
    return np.array([10000.0, 0.0, 0.05, prob, 60.0, 0.0], dtype=np.float32)


class TestHedgingAgentWithoutRlLibs:
    def test_constructor_requires_gymnasium(self) -> None:
        if rl.gym is not None:
            pytest.skip("gymnasium installed; constructor succeeds and the guard is not reached")
        with pytest.raises(ImportError, match="gymnasium and stable_baselines3"):
            HedgingAgent("PPO")

    def test_sizing_multiplier_is_neutral_without_a_trained_model(self) -> None:
        agent = HedgingAgent.__new__(HedgingAgent)
        agent.env = None
        agent.algo = "PPO"
        agent.model = None
        assert agent.get_sizing_multiplier({"bankroll": 5000.0}) == 1.0


class TestHedgingEnvObservation:
    def test_time_remaining_decreases_linearly(self) -> None:
        env = _bare_env(max_steps=100)
        for step, expected in [(0, 60.0), (50, 30.0), (100, 0.0)]:
            env.current_step = step
            assert env._get_obs()[4] == pytest.approx(expected)

    def test_observation_shape_dtype_and_ranges(self) -> None:
        np.random.seed(0)
        env = _bare_env()
        for _ in range(25):
            obs = env._get_obs()
            assert obs.shape == (6,)
            assert obs.dtype == np.float32
            assert env.bankroll == 10000.0
            assert 0.01 <= obs[2] <= 0.15   # edge
            assert 0.40 <= obs[3] <= 0.80   # win probability

    def test_reset_is_not_reachable_without_gymnasium(self) -> None:
        # reset() calls super().reset(); with no gymnasium base class that raises.
        if rl.gym is not None:
            pytest.skip("gymnasium installed")
        with pytest.raises(AttributeError):
            _bare_env().reset(seed=1)


class TestHedgingEnvStep:
    def test_winning_bet_grows_bankroll_by_capped_kelly_fraction(self, monkeypatch) -> None:
        env = _bare_env(bankroll=10000.0)
        monkeypatch.setattr(env, "_get_obs", lambda: _fixed_obs(prob=0.6))
        monkeypatch.setattr(rl.np.random, "rand", lambda: 0.1)  # below 0.6 -> win

        # p=0.6: odds=(1-p)/p, b=1/odds=1.5, Kelly=(b*p-(1-p))/b = 1/3, capped at 0.05.
        # Bet = 0.05 * 10000 = 500; win pays b * bet = 750.
        _, reward, terminated, truncated, _ = env.step(np.array([1.0], dtype=np.float32))
        assert env.bankroll == pytest.approx(10750.0)
        assert reward == pytest.approx(math.log(10750.0 / 10000.0))
        assert terminated is False
        assert truncated is False
        assert env.current_step == 1

    def test_losing_bet_costs_the_stake(self, monkeypatch) -> None:
        env = _bare_env(bankroll=10000.0)
        monkeypatch.setattr(env, "_get_obs", lambda: _fixed_obs(prob=0.6))
        monkeypatch.setattr(rl.np.random, "rand", lambda: 0.9)  # above 0.6 -> loss

        _, reward, _, _, _ = env.step(np.array([1.0], dtype=np.float32))
        assert env.bankroll == pytest.approx(9500.0)
        assert reward == pytest.approx(math.log(9500.0 / 10000.0))

    def test_zero_action_places_no_bet(self, monkeypatch) -> None:
        env = _bare_env(bankroll=10000.0)
        monkeypatch.setattr(env, "_get_obs", lambda: _fixed_obs(prob=0.6))
        monkeypatch.setattr(rl.np.random, "rand", lambda: 0.9)

        _, reward, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert env.bankroll == pytest.approx(10000.0)
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_negative_kelly_clips_to_no_bet(self, monkeypatch) -> None:
        # p=0.4 gives negative Kelly, so the stake is clipped to zero even on a win.
        env = _bare_env(bankroll=10000.0)
        monkeypatch.setattr(env, "_get_obs", lambda: _fixed_obs(prob=0.4))
        monkeypatch.setattr(rl.np.random, "rand", lambda: 0.1)

        _, reward, _, _, _ = env.step(np.array([1.5], dtype=np.float32))
        assert env.bankroll == pytest.approx(10000.0)
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_fraction_never_exceeds_max_bet_fraction(self, monkeypatch) -> None:
        env = _bare_env(bankroll=10000.0)
        monkeypatch.setattr(env, "_get_obs", lambda: _fixed_obs(prob=0.8))
        monkeypatch.setattr(rl.np.random, "rand", lambda: 0.99)  # loss

        env.step(np.array([1.5], dtype=np.float32))
        assert env.bankroll == pytest.approx(10000.0 - 10000.0 * MAX_BET_FRACTION)

    def test_episode_terminates_at_max_steps(self, monkeypatch) -> None:
        env = _bare_env(bankroll=10000.0, max_steps=5)
        env.current_step = 4
        monkeypatch.setattr(env, "_get_obs", lambda: _fixed_obs(prob=0.6))
        monkeypatch.setattr(rl.np.random, "rand", lambda: 0.1)
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert env.current_step == 5
        assert terminated is True


# ── season_simulator_bridge ───────────────────────────────────────────────────

def _players(n: int = 2) -> pd.DataFrame:
    rows = [
        {"player_id": "P1", "position": "WR", "team": "KC",
         "kalman_est_receiving_yards": 60.0, "kalman_variance_receiving_yards": 100.0,
         "injury_multiplier": 1.0, "snap_share": 0.9},
        {"player_id": "P2", "position": "TE", "team": "SF",
         "kalman_est_receiving_yards": 35.0, "kalman_variance_receiving_yards": 50.0,
         "injury_multiplier": 1.0, "snap_share": 0.7},
        {"player_id": "P3", "position": "RB", "team": "BUF",
         "kalman_est_receiving_yards": 20.0, "kalman_variance_receiving_yards": 30.0,
         "injury_multiplier": 0.0, "snap_share": 0.4},
    ]
    return pd.DataFrame(rows[:n])


class TestPlayerStructs:
    def test_from_row_uses_stat_columns_and_defaults(self) -> None:
        row = {"player_id": "00-0033106", "position": "TE", "team": "KC",
               "kalman_est_receiving_yards": 40.0}
        c = bridge._PlayerProjectionC.from_row(row, "receiving_yards")
        assert c.player_id == b"00-0033106"
        assert c.position == b"TE"
        assert c.team == b"KC"
        assert c.kalman_mean == pytest.approx(40.0)
        # Missing variance defaults to max(est*0.5, 5)**2 = 20**2.
        assert c.kalman_variance == pytest.approx(400.0)
        assert c.injury_multiplier == pytest.approx(1.0)
        assert c.snap_share == pytest.approx(1.0)

    def test_from_row_truncates_identifiers(self) -> None:
        row = {"player_id": "X" * 40, "position": "WIDE", "team": "ABCD"}
        c = bridge._PlayerProjectionC.from_row(row, "receiving_yards")
        assert c.player_id == b"X" * 31
        assert c.position == b"WID"
        assert c.team == b"ABC"
        # With no estimate, the mean is zero and the variance floor is 5**2.
        assert c.kalman_mean == 0.0
        assert c.kalman_variance == pytest.approx(25.0)

    def test_build_player_array_fields(self) -> None:
        arr = bridge._build_player_array(_players(2), "receiving_yards")
        assert arr.shape == (2,)
        assert arr[0]["player_id"] == b"P1"
        assert arr[0]["kalman_mean"] == pytest.approx(60.0)
        assert arr[0]["kalman_variance"] == pytest.approx(100.0)
        assert arr[0]["snap_share"] == pytest.approx(0.9, rel=1e-6)
        assert arr[1]["position"] == b"TE"

    def test_build_player_array_fills_missing_kalman_columns(self) -> None:
        df = pd.DataFrame([{"player_id": "Z9", "position": "QB", "team": "NO"}])
        arr = bridge._build_player_array(df, "passing_yards")
        assert arr[0]["kalman_mean"] == 0.0
        assert arr[0]["kalman_variance"] == pytest.approx(25.0)
        assert arr[0]["injury_multiplier"] == pytest.approx(1.0)


class TestSimSeason:
    def test_empty_roster_returns_empty_arrays(self) -> None:
        out = bridge.sim_season(_players(2).iloc[:0], "receiving_yards")
        assert set(out) == {"mean", "p10", "p50", "p90"}
        assert all(v.shape == (0,) for v in out.values())

    def test_python_path_shapes_ordering_and_row_alignment(self) -> None:
        out = bridge.sim_season(
            _players(3), "receiving_yards", n_simulations=60, start_week=1, end_week=4,
            use_cpp=False, rng_seed=11,
        )
        for key in ("mean", "p10", "p50", "p90"):
            assert out[key].shape == (3,)
            assert out[key].dtype == np.float32
        assert np.all(out["p10"] <= out["p50"])
        assert np.all(out["p50"] <= out["p90"])
        assert np.all(out["mean"] > 0)

    def test_fixed_seed_is_reproducible(self) -> None:
        kwargs = dict(n_simulations=40, start_week=1, end_week=3, use_cpp=False, rng_seed=5)
        first = bridge.sim_season(_players(2), "receiving_yards", **kwargs)
        second = bridge.sim_season(_players(2), "receiving_yards", **kwargs)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_cpp_request_falls_back_to_python_when_library_missing(self) -> None:
        from engine import python_bindings

        if python_bindings.LIB is not None:
            pytest.skip("C++ library is built; the fallback path is not the one under test")
        kwargs = dict(n_simulations=40, start_week=1, end_week=3, rng_seed=9)
        via_cpp_request = bridge.sim_season(_players(2), "receiving_yards", use_cpp=True, **kwargs)
        via_python = bridge.sim_season(_players(2), "receiving_yards", use_cpp=False, **kwargs)
        for key in via_python:
            np.testing.assert_array_equal(via_cpp_request[key], via_python[key])

    def test_python_helper_reports_zero_for_unknown_player_totals(self, monkeypatch) -> None:
        # When the simulator returns no totals for a player, the helper writes zeros.
        class _Result:
            player_season_totals = {}

        class _Sim:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, players_df, prior_game_rows, rng_seed):
                return _Result()

        import ml.season_simulator as season_simulator

        monkeypatch.setattr(season_simulator, "SeasonSimulator", _Sim)
        out = bridge._python_sim_season(_players(2), "receiving_yards", 10, 1, 2)
        assert out["mean"].tolist() == [0.0, 0.0]
        assert out["p90"].tolist() == [0.0, 0.0]

    def test_python_helper_extracts_stat_totals(self, monkeypatch) -> None:
        class _Result:
            player_season_totals = {
                "P1": {"receiving_yards": {"mean": 100.0, "p10": 80.0, "p50": 98.0, "p90": 125.0}},
                "P2": {"receiving_yards": {"mean": 40.0, "p10": 30.0, "p50": 39.0, "p90": 52.0}},
            }

        class _Sim:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, players_df, prior_game_rows, rng_seed):
                assert rng_seed == 3
                assert set(prior_game_rows) == {"P1", "P2"}
                assert prior_game_rows["P1"] == []
                return _Result()

        import ml.season_simulator as season_simulator

        monkeypatch.setattr(season_simulator, "SeasonSimulator", _Sim)
        out = bridge._python_sim_season(_players(2), "receiving_yards", 10, 1, 2, rng_seed=3)
        assert out["mean"].tolist() == pytest.approx([100.0, 40.0])
        assert out["p10"].tolist() == pytest.approx([80.0, 30.0])
        assert out["p50"].tolist() == pytest.approx([98.0, 39.0])
        assert out["p90"].tolist() == pytest.approx([125.0, 52.0])
