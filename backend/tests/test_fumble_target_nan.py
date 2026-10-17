"""The fumble training target matches backtest scoring on NaN components.

Backtest scoring (``ml.stat_resolution.read_gamelog_stat``) treats an
unreported fumble component — None or NaN — as contributing nothing, and
returns None only when nothing was reported. The training target must use
the same rule: previously it summed NaN into the total, so those rows
trained as NaN (dropped from fumble training) while the backtest scored
them as partial sums. Without the fix the partial-NaN cases below fail.
"""

from __future__ import annotations

import math

from ml.stat_resolution import read_gamelog_stat
from pipeline.feature_engineer import build_feature_row


def _game(**kwargs) -> dict:
    defaults = dict(
        home_team="MIN", away_team="GB",
        roof="outdoors", surface="grass",
        temp=52.0, wind=8.0,
        total_line=45.5, spread_line=-3.5,
        home_rest=7, away_rest=7,
    )
    defaults.update(kwargs)
    return defaults


def _target(**kwargs) -> dict:
    row = dict(
        player_id="TEST-001", game_id="2024_01_MIN_GB",
        season=2024, week=1, position="WR",
        team="MIN", opponent_team="GB",
        receiving_yards=90.0, receiving_tds=1, targets=8, receptions=6,
        target_share=0.25, air_yards_share=0.28, fantasy_points_ppr=22.0,
        carries=0, rushing_yards=0.0, rushing_tds=0,
        attempts=0, passing_yards=0.0, passing_tds=0, completions=0,
    )
    row.update(kwargs)
    return row


class TestFumbleTargetNanParity:
    def test_partial_nan_scores_partial_sum(self) -> None:
        target = _target(rushing_fumbles=1.0, receiving_fumbles=float("nan"), sack_fumbles=None)
        fr = build_feature_row(target, [], _game(), [target])
        assert fr.actual_fumbles == 1.0

    def test_all_nan_is_missing(self) -> None:
        nan = float("nan")
        target = _target(rushing_fumbles=nan, receiving_fumbles=nan, sack_fumbles=nan)
        fr = build_feature_row(target, [], _game(), [target])
        assert fr.actual_fumbles is None

    def test_all_none_is_missing(self) -> None:
        target = _target(rushing_fumbles=None, receiving_fumbles=None, sack_fumbles=None)
        fr = build_feature_row(target, [], _game(), [target])
        assert fr.actual_fumbles is None

    def test_recorded_zero_stays_zero(self) -> None:
        target = _target(rushing_fumbles=0, receiving_fumbles=0, sack_fumbles=0)
        fr = build_feature_row(target, [], _game(), [target])
        assert fr.actual_fumbles == 0.0

    def test_training_target_equals_backtest_scoring(self) -> None:
        nan = float("nan")
        cases = [
            {"rushing_fumbles": 1.0, "receiving_fumbles": nan, "sack_fumbles": None},
            {"rushing_fumbles": nan, "receiving_fumbles": 0.0, "sack_fumbles": nan},
            {"rushing_fumbles": nan, "receiving_fumbles": nan, "sack_fumbles": nan},
            {"rushing_fumbles": None, "receiving_fumbles": None, "sack_fumbles": None},
            {"rushing_fumbles": 2.0, "receiving_fumbles": 1.0, "sack_fumbles": 0.0},
        ]
        for components in cases:
            target = _target(**components)
            fr = build_feature_row(target, [], _game(), [target])
            expected = read_gamelog_stat(dict(target), "fumbles")
            if expected is None:
                assert fr.actual_fumbles is None, components
            else:
                assert fr.actual_fumbles is not None, components
                assert math.isclose(fr.actual_fumbles, expected), components
