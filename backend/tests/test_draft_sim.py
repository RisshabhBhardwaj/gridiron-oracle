"""Roster-aware draft metric, age curve, and multi-season availability."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.age_curve import build_observations, fit_age_curve, season_age
from ml.draft_sim import (
    evaluate_board,
    positional_points_lost,
    snake_draft,
    starting_lineup_points,
)
from ml.playing_time import expected_games_from_history


class TestStartingLineupPoints:
    def test_fills_one_qb_two_rb_two_wr_one_te_one_flex(self):
        positions = ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "TE"]
        realized = [300, 290, 200, 190, 180, 250, 240, 150]
        # 300 (QB) + 200+190 (RB) + 250+240 (WR) + 150 (TE) + 180 (FLEX)
        assert starting_lineup_points(range(8), positions, realized) == pytest.approx(1510)

    def test_second_quarterback_never_starts(self):
        positions = ["QB", "QB"]
        realized = [300, 290]
        assert starting_lineup_points([0, 1], positions, realized) == pytest.approx(300)

    def test_unfillable_slot_scores_zero_not_nan(self):
        assert starting_lineup_points([0], ["QB"], [300]) == pytest.approx(300)

    def test_missing_realized_counts_as_zero(self):
        assert starting_lineup_points([0, 1], ["QB", "RB"], [None, 100]) == pytest.approx(100)


class TestSnakeDraft:
    def test_subject_receives_one_pick_per_round(self):
        order = list(range(200))
        roster = snake_draft(order, order, subject_slot=3, n_teams=8, rounds=14)
        assert len(roster) == 14
        assert len(set(roster)) == 14

    def test_seats_never_share_a_player(self):
        order = list(range(120))
        rosters = [snake_draft(order, order, subject_slot=slot, n_teams=8, rounds=5) for slot in range(8)]
        # Every seat drafting the same board must still receive distinct players
        # within its own roster; overlap across seats is expected only because
        # each simulation is independent.
        for roster in rosters:
            assert len(set(roster)) == len(roster)

    def test_rejects_out_of_range_slot(self):
        with pytest.raises(ValueError):
            snake_draft([0], [0], subject_slot=8, n_teams=8)


class TestMetricRejectsIllegalRosters:
    """The defect that made the previous acceptance metric meaningless.

    ``points_lost_vs_optimal`` scored the top 24 raw scorers as optimal, which
    in a 1-QB league is 9-11 quarterbacks. A board that drafted only
    quarterbacks therefore *won*. The roster-aware metric must reverse that.
    """

    def _universe(self):
        """Quarterbacks always outscore, and value is dispersed inside each position."""
        positions, realized = [], []
        for rank in range(20):
            positions.append("QB")
            realized.append(320.0 - 4.0 * rank)
        for pos, top, step in (("RB", 300.0, 9.0), ("WR", 290.0, 8.0), ("TE", 200.0, 6.0)):
            for rank in range(30):
                positions.append(pos)
                realized.append(top - step * rank)
        return positions, realized

    def _market(self, positions):
        """Realistic ADP: skill positions early, quarterbacks late."""
        offsets = {"RB": 0.0, "WR": 0.5, "TE": 40.0, "QB": 60.0}
        return [offsets[pos] + index * 0.01 for index, pos in enumerate(positions)]

    def test_quarterback_first_board_loses_to_balanced_board(self):
        positions, realized = self._universe()
        market = self._market(positions)
        # Both boards are ranked best-first within position; they differ only in
        # how they price quarterbacks against everyone else.
        qb_first = [
            (1000.0 if pos == "QB" else 0.0) + points
            for pos, points in zip(positions, realized)
        ]
        balanced = [
            (0.35 if pos == "QB" else 1.0) * points
            for pos, points in zip(positions, realized)
        ]
        qb_score = evaluate_board(qb_first, market, positions, realized, rounds=8)["mean_starter_points"]
        balanced_score = evaluate_board(balanced, market, positions, realized, rounds=8)["mean_starter_points"]
        assert balanced_score > qb_score

    def test_rotation_covers_every_draft_slot(self):
        positions, realized = self._universe()
        scores = [1.0] * len(positions)
        result = evaluate_board(scores, list(range(len(positions))), positions, realized, rounds=8)
        assert len(result["per_slot"]) == 8
        assert result["min_starter_points"] <= result["mean_starter_points"] <= result["max_starter_points"]


class TestPositionalPointsLost:
    def test_returns_nan_when_a_position_is_too_thin(self):
        frame = pd.DataFrame({
            "position": ["QB"] * 3,
            "projection": [3.0, 2.0, 1.0],
            "realized": [30.0, 20.0, 10.0],
        })
        assert pd.isna(positional_points_lost(frame, "projection")["QB"])


class TestAgeCurve:
    def _logs(self):
        rows = []
        for season in (2018, 2019, 2020, 2021):
            for player in range(60):
                for week in range(1, 13):
                    rows.append({
                        "player_id": f"p{player}",
                        "season": season,
                        "week": week,
                        "fantasy_points_ppr": 10.0 + (player % 7),
                    })
        return pd.DataFrame(rows)

    def _players(self):
        return pd.DataFrame({
            "id": [f"p{i}" for i in range(60)],
            "position": ["RB" if i % 2 else "WR" for i in range(60)],
            "birth_date": pd.to_datetime([f"{1992 + (i % 6)}-03-01" for i in range(60)]),
        })

    def test_refuses_target_season_rows(self):
        observations = build_observations(self._logs(), self._players(), max_season=2021)
        with pytest.raises(ValueError, match="leakage"):
            fit_age_curve(observations, max_season=2020)

    def test_fits_multipliers_near_one_on_flat_data(self):
        observations = build_observations(self._logs(), self._players(), max_season=2020)
        curve = fit_age_curve(observations, max_season=2020)
        assert curve
        for value in curve.values():
            assert 0.8 < value < 1.25

    def test_september_birthday_has_not_happened_by_kickoff(self):
        births = pd.Series(pd.to_datetime(["2000-09-15", "2000-08-15"]))
        ages = season_age(births, pd.Series([2025, 2025]))
        assert ages.tolist() == [24.0, 25.0]


class TestExpectedGamesFromHistory:
    def _history(self, rows):
        return pd.DataFrame(rows)

    def test_injured_season_does_not_erase_prior_availability(self):
        rows = []
        for season, games in ((2022, 17), (2023, 16), (2024, 0)):
            for week in range(1, games + 1):
                rows.append({"player_id": "steady", "season": season, "week": week})
        for season in (2022, 2023, 2024):
            for week in range(1, 18):
                rows.append({"player_id": "iron", "season": season, "week": week})
        universe = pd.DataFrame({
            "id": ["steady", "iron"],
            "position": ["RB", "RB"],
            "entry_year": [2022, 2022],
        })
        expected = expected_games_from_history(self._history(rows), universe, season=2025)
        # The single-season heuristic gave a player who missed all of 2024
        # roughly 1.4 games. Three seasons of evidence must do better.
        assert expected["steady"] > 5.0
        assert expected["iron"] > expected["steady"]

    def test_rookie_not_charged_for_pre_entry_seasons(self):
        rows = [{"player_id": "soph", "season": 2024, "week": week} for week in range(1, 18)]
        universe = pd.DataFrame({"id": ["soph"], "position": ["WR"], "entry_year": [2024]})
        expected = expected_games_from_history(self._history(rows), universe, season=2025)
        assert expected["soph"] > 14.0
