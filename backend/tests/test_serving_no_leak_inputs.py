"""Serving model inputs contain no contemporaneous xFP or target-game snaps.

Same-week expected fantasy (``*_exp``) is realized opportunity, and
target-game snap/participation fields are known only after kickoff. Neither
may reach a model frame. The as-of contract forbids them; this test pins
the serving side: the served column list, every usable experiment group,
and the one place xFP is touched (playing-time, lagged only).
"""

from __future__ import annotations

import pytest

from ml.feature_contract import FORBIDDEN_MODEL_FIELDS, assert_model_input_columns
from ml.playing_time import build_playing_time_frame
from ml.utils import FEATURE_COLS, FEATURE_GROUPS


CONTEMPORANEOUS_XFP = sorted(
    n
    for n in FORBIDDEN_MODEL_FIELDS
    if n.endswith("_exp") or n in {"receptions_exp", "targets_exp", "carries_exp"}
)
TARGET_GAME_SNAPS = [
    "snap_pct_off",
    "offense_pct",
    "routes_run_pct",
    "blitz_exposure",
    "target_share_pbp",
    "air_yards_share_pbp",
]


class TestServingInputsHaveNoLeakFields:
    def test_served_columns_are_clean(self) -> None:
        overlap = sorted(set(FEATURE_COLS) & set(FORBIDDEN_MODEL_FIELDS))
        assert overlap == []
        for name in (*CONTEMPORANEOUS_XFP, *TARGET_GAME_SNAPS):
            assert name not in FEATURE_COLS, name

    def test_usable_experiment_groups_are_clean(self) -> None:
        from ml.feature_groups import resolve_feature_cols

        for group in FEATURE_GROUPS:
            if group == "progression":
                continue  # blocked below; years_exp/exp_bucket are forbidden
            cols = resolve_feature_cols([group])
            assert not (set(cols) & set(FORBIDDEN_MODEL_FIELDS)), group

    def test_progression_group_is_refused(self) -> None:
        from ml.feature_groups import resolve_feature_cols

        with pytest.raises(AssertionError, match="prohibited feature"):
            resolve_feature_cols(["progression"])

    def test_contract_rejects_every_leak_field(self) -> None:
        for name in (*CONTEMPORANEOUS_XFP, *TARGET_GAME_SNAPS):
            with pytest.raises(AssertionError, match="prohibited feature"):
                assert_model_input_columns([*FEATURE_COLS, name], consumer="test")


class TestPlayingTimeXfpIsLagOnly:
    def test_lagged_xfp_uses_prior_game_only(self) -> None:
        import pandas as pd

        history = pd.DataFrame(
            {
                "player_id": ["P1", "P1"],
                "season": [2024, 2024],
                "week": [1, 2],
                "position": ["WR", "WR"],
                "offense_pct": [80.0, 85.0],
                "fantasy_points_ppr": [20.0, 22.0],
                "rec_fantasy_points_exp": [18.0, 21.0],
            }
        )
        frame = build_playing_time_frame(history).sort_values("week").reset_index(drop=True)
        # Week 1 has no prior game: nothing contemporaneous may fill the lag.
        assert pd.isna(frame.loc[0, "lagged_rec_xfp"])
        # Week 2 sees week 1's xFP only.
        assert frame.loc[1, "lagged_rec_xfp"] == 18.0
        assert frame.loc[1, "lagged_actual_minus_expected"] == 20.0 - 18.0
