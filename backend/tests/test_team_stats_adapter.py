"""
Regression guard for scraper/adapters/nflreadpy_adapter.py's TeamStatsRow.

nfl.load_team_stats() has no total_plays or total_yards column at all —
verified directly against the source (138 columns, neither name present).
Declaring them as plain Optional fields meant every row silently validated
to None instead of erroring, and team_game_stats.total_plays/total_yards
sat at 0% fill from day one. Fixed via @computed_field derivations from
attempts+carries and passing_yards+rushing_yards, which ARE present.
"""

from __future__ import annotations

from scraper.adapters.nflreadpy_adapter import TeamStatsRow


def test_total_plays_and_total_yards_are_derived_and_present_in_model_dump():
    """
    A plain @property is invisible to model_dump(mode="json") — the exact
    call _process_source makes before writing to staging — so a naive fix
    would look correct in a unit test that reads the attribute directly but
    still ship 0% fill. Assert on model_dump() output, not attribute access.
    """
    row = TeamStatsRow(
        team="KC", season=2023, week=1,
        attempts=30, carries=25, passing_yards=132.0, rushing_yards=96.0,
    )
    dumped = row.model_dump(mode="json")
    assert dumped["total_plays"] == 55
    assert dumped["total_yards"] == 228.0


def test_total_plays_and_total_yards_none_when_inputs_missing():
    row = TeamStatsRow(team="KC", season=2023, week=1)
    dumped = row.model_dump(mode="json")
    assert dumped["total_plays"] is None
    assert dumped["total_yards"] is None


def test_total_plays_and_total_yards_partial_inputs_stay_none():
    """Half the inputs present must not produce a silently-wrong partial sum."""
    row = TeamStatsRow(team="KC", season=2023, week=1, attempts=30, passing_yards=132.0)
    dumped = row.model_dump(mode="json")
    assert dumped["total_plays"] is None  # carries missing
    assert dumped["total_yards"] is None  # rushing_yards missing
