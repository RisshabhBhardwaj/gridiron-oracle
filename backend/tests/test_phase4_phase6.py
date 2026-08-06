"""Tests for Phase 4 feature groups and Phase 6 win accumulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.feature_groups import resolve_feature_cols
from ml.season_simulator import SeasonSimulator
from ml.utils import FEATURE_COLS, FEATURE_GROUP_OPP_ADJ_USAGE, FEATURE_GROUPS
from ml.win_eval import actual_wins_from_schedule, brier_score, evaluate_win_projections
from pipeline.feature_engineer import build_feature_row
from pipeline.features.buckets import (
    compute_opp_adj_usage,
    compute_pace_script,
    compute_progression_priors,
    compute_usage_shares,
)


def _game(**kwargs) -> dict:
    defaults = dict(
        home_team="MIN", away_team="GB",
        roof="outdoors", surface="grass",
        temp=52.0, wind=8.0,
        total_line=48.0, spread_line=-3.0,
        home_rest=7, away_rest=7,
    )
    defaults.update(kwargs)
    return defaults


def _stat_row(**kwargs) -> dict:
    defaults = dict(
        player_id="TEST-001", game_id="2024_01_MIN_GB",
        season=2024, week=1, position="WR",
        team="MIN", opponent_team="GB",
        receiving_yards=90.0, receiving_tds=1, targets=8, receptions=6,
        target_share=0.25, air_yards_share=0.28, fantasy_points_ppr=22.0,
        carries=0, rushing_yards=0.0, rushing_tds=0,
        attempts=0, passing_yards=0.0, passing_tds=0, completions=0,
        offense_pct=0.85,
    )
    defaults.update(kwargs)
    return defaults


class TestPhase4Buckets:
    def test_carry_share_causal(self) -> None:
        prior = [
            _stat_row(week=1, game_id="g1", carries=10),
            _stat_row(week=2, game_id="g2", carries=15),
        ]
        teammates = [
            _stat_row(player_id="RB1", week=1, game_id="g1", carries=10, position="RB"),
            _stat_row(player_id="RB2", week=1, game_id="g1", carries=10, position="RB"),
            _stat_row(player_id="RB1", week=2, game_id="g2", carries=5, position="RB"),
            _stat_row(player_id="RB2", week=2, game_id="g2", carries=10, position="RB"),
        ]
        all_rows = prior + teammates
        out = compute_usage_shares(prior, all_rows, "TEST-001", "MIN", 3, "WR")
        assert out["carry_share"] is not None
        assert abs(out["carry_share"] - (10 / 30 + 15 / 30) / 2) < 1e-6

    def test_pace_script_neutral_and_expected_pass(self) -> None:
        rows = [
            _stat_row(week=1, game_id="g1", attempts=35, carries=5, position="QB", player_id="QB1"),
            _stat_row(week=1, game_id="g1", attempts=0, carries=20, position="RB", player_id="RB1"),
        ]
        game = _game(spread_line=-2.0, total_line=50.0)
        out = compute_pace_script("MIN", 2, game, rows, is_home=1)
        assert out["team_pace"] is not None
        assert out["neutral_script_flag"] == 1.0
        assert out["expected_pass_attempts"] is not None
        assert out["expected_pass_attempts"] > 0

    def test_progression_exp_bucket(self) -> None:
        row = {"years_exp": 0, "birth_date": "2003-05-01", "draft_round": 1}
        out = compute_progression_priors(row, prior_rows=[], season=2024)
        assert out["exp_bucket"] == 0.0
        assert out["age"] == 21.0
        assert out["career_games"] == 0.0

        row2 = {"years_exp": 5}
        assert compute_progression_priors(row2, [{}] * 4, 2024)["exp_bucket"] == 2.0

    def test_opp_adj_target_share_scales_with_defense(self) -> None:
        seas = {"seas_avg_target_share": 0.20}
        matchup = {"opp_avg_targets_allowed": 40.0}
        usage = {"carry_share": 0.1, "snap_share_trailing": 0.8, "snap_share_trend": 0.0}
        rows = [
            _stat_row(week=1, game_id="g1", targets=10, target_share=0.2, player_id="A", position="WR"),
            _stat_row(week=1, game_id="g1", targets=20, target_share=0.1, player_id="B", position="WR"),
        ]
        out = compute_opp_adj_usage(seas, matchup, usage, rows, week=2, position="WR")
        assert out["opp_adj_target_share"] is not None
        assert abs(out["opp_adj_target_share"] - 0.20 * (40 / 30)) < 1e-6


class TestFeatureGroupRegistry:
    def test_groups_not_in_default_feature_cols(self) -> None:
        for cols in FEATURE_GROUPS.values():
            for c in cols:
                assert c not in FEATURE_COLS, f"{c} leaked into default FEATURE_COLS"

    def test_resolve_adds_group(self) -> None:
        cols = resolve_feature_cols(["opp_adj_usage"])
        for c in FEATURE_GROUP_OPP_ADJ_USAGE:
            assert c in cols


class TestBuildFeatureRowPhase4:
    def test_phase4_fields_populated(self) -> None:
        prior = [
            _stat_row(week=1, game_id="2024_01_MIN_GB", offense_pct=0.8, target_share=0.22),
            _stat_row(week=2, game_id="2024_02_MIN_CHI", offense_pct=0.9, target_share=0.28,
                      opponent_team="CHI"),
        ]
        target = _stat_row(
            week=3, game_id="2024_03_MIN_DET", opponent_team="DET",
            years_exp=4, birth_date="1998-01-01", draft_round=2,
        )
        all_rows = list(prior) + [
            _stat_row(player_id="TEAMMATE", week=1, game_id="2024_01_MIN_GB",
                      carries=5, targets=5, position="RB"),
            _stat_row(player_id="TEAMMATE", week=2, game_id="2024_02_MIN_CHI",
                      carries=5, targets=5, position="RB", opponent_team="CHI"),
        ]
        fr = build_feature_row(target, prior, _game(), all_rows)
        assert fr.years_exp == 4.0
        assert fr.exp_bucket == 2.0
        assert fr.career_games == 2.0
        assert fr.expected_pass_attempts is not None
        assert fr.neutral_script_flag == 1.0


class TestWinAccumulation:
    def test_accumulate_week_wins_increments(self) -> None:
        sim = SeasonSimulator(season=2024, start_week=1, end_week=1, n_simulations=4)
        schedule = pd.DataFrame([{"week": 1, "home_team": "MIN", "away_team": "GB"}])
        team_map = {"p1": "MIN", "p2": "GB"}
        week_paths = {
            ("p1", "fantasy_ppr"): np.array([100.0, 100.0, 10.0, 10.0]),
            ("p2", "fantasy_ppr"): np.array([10.0, 10.0, 100.0, 100.0]),
        }
        accum = {"MIN": np.zeros(4), "GB": np.zeros(4)}
        sim._accumulate_week_wins(schedule, week_paths, team_map, accum, 4)
        assert list(accum["MIN"]) == [1.0, 1.0, 0.0, 0.0]
        assert list(accum["GB"]) == [0.0, 0.0, 1.0, 1.0]

    def test_playoff_probs_all_make_small_conf(self) -> None:
        rng = np.random.default_rng(0)
        accum = {
            "BUF": np.array([12.0, 12.0]),
            "MIA": np.array([10.0, 10.0]),
            "NE": np.array([4.0, 4.0]),
            "NYJ": np.array([2.0, 2.0]),
        }
        probs = SeasonSimulator._playoff_probs_from_wins(accum, rng)
        assert probs["BUF"] == 1.0
        assert probs["NYJ"] == 1.0  # only 4 AFC teams → all top-7

    def test_brier_and_wins_eval(self) -> None:
        schedule = pd.DataFrame([
            {"home_team": "MIN", "away_team": "GB", "home_score": 24, "away_score": 17},
            {"home_team": "GB", "away_team": "MIN", "home_score": 20, "away_score": 10},
        ])
        wins = actual_wins_from_schedule(schedule)
        assert wins["GB"] == 1.0
        assert wins["MIN"] == 1.0
        assert abs(brier_score({"MIN": 0.8, "GB": 0.2}, {"MIN": 1, "GB": 0}) - 0.04) < 1e-9

        totals = {
            "MIN": {"wins_mean": 1.0, "wins_p10": 1.0, "wins_p90": 1.0},
            "GB": {"wins_mean": 1.0, "wins_p10": 1.0, "wins_p90": 1.0},
        }
        result = evaluate_win_projections(
            2024, totals, schedule, playoff_probs={"MIN": 0.5, "GB": 0.5}
        )
        assert result.wins_mae == 0.0
        assert result.playoff_brier is not None
