"""As-of Elo features for training rows (audit C1).

A row's Elo features may use only games strictly before that row's game. Each
test flips one game's result and checks the target row. The implied win
probability is checked against the same pregame snapshot as the other columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.tests._elo_asof_fixture import (
    ASOF_EARLIER,
    ASOF_LATER,
    ASOF_TARGET,
    asof_games_frame,
)
from ml.team_elo import TeamEloSystem, _expected_score

ELO_COLS = [
    "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
    "elo_matchup_diff", "elo_implied_win_prob",
]


def _target_features(flip=None) -> dict[str, float]:
    """Fit on the schedule (optionally with one result flipped) and enrich MIN vs GB, 2024 week 5."""
    elo = TeamEloSystem().fit(asof_games_frame(flip=flip))
    row = pd.DataFrame([{"team": "MIN", "opp_team": "GB", "is_home": 1}])
    out = elo.enrich_features(row, season=2024, week=5)
    return {col: float(out.iloc[0][col]) for col in ELO_COLS}


@pytest.fixture(scope="module")
def base() -> dict[str, float]:
    return _target_features()


def test_target_row_has_pregame_history(base):
    # Guards against passing for the wrong reason: a 1500 snapshot would be
    # unchanged by any result flip.
    assert base["team_off_elo"] != pytest.approx(1500.0)
    assert base["opp_def_elo"] != pytest.approx(1500.0)


def test_flipping_the_target_game_leaves_its_own_features_unchanged(base):
    assert _target_features(flip=ASOF_TARGET) == pytest.approx(base, rel=1e-12)


def test_flipping_a_later_game_leaves_the_target_row_unchanged(base):
    assert _target_features(flip=ASOF_LATER) == pytest.approx(base, rel=1e-12)


def test_flipping_an_earlier_game_changes_the_target_row(base):
    # Positive control: the snapshot is not constant, so a prior result must move it.
    flipped = _target_features(flip=ASOF_EARLIER)
    assert flipped["team_off_elo"] != pytest.approx(base["team_off_elo"])


def test_implied_win_probability_comes_from_the_pregame_snapshot(base):
    expected = _expected_score(
        (base["team_off_elo"] + base["team_def_elo"]) / 2,
        (base["opp_off_elo"] + base["opp_def_elo"]) / 2,
        a_is_home=True,
    )
    assert base["elo_implied_win_prob"] == pytest.approx(expected, rel=1e-12)
