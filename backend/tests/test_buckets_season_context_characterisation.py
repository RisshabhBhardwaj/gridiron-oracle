"""Characterisation tests for build_season_feature_context.

The function is being refactored to lower its cyclomatic complexity. These
tests pin its output exactly: values (compared by repr, so float bits and
int-vs-float both matter), dict insertion order, value types, and the fact
that each week's inner dict is its own object. The golden output was captured
from the pre-refactor implementation; the spot checks were derived by hand.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from pipeline.features.buckets import SeasonFeatureContext, build_season_feature_context

_FIELDS = (
    "week",
    "game_id",
    "team",
    "player_id",
    "position",
    "carries",
    "attempts",
    "pass_attempts",
    "targets",
    "offense_pct",
    "target_share",
)


def _row(**kwargs: Any) -> dict[str, Any]:
    """A season row with every field present; unspecified fields are None."""
    row: dict[str, Any] = dict.fromkeys(_FIELDS)
    row.update(kwargs)
    return row


def _main_rows() -> list[dict[str, Any]]:
    """Synthetic season covering every branch of the scan and the per-week loop.

    Row order matters: it sets dict insertion order and float summation order.
    """
    return [
        # Week 1, game g1 (KC, BUF). Row order sets the scan order.
        # carries and attempts missing -> 0.0. offense_pct 80 is scaled to 0.8.
        _row(week=1, game_id="g1", team="KC", player_id="p1", position="wr",
             targets=8, offense_pct=80, target_share=0.25),
        # attempts missing -> pass_attempts is used. offense_pct 0.6 is kept as is.
        # target_share missing -> None, so this row is skipped for target share.
        _row(week=1, game_id="g1", team="BUF", player_id="p2", position="RB",
             carries=12, pass_attempts=2, targets=3, offense_pct=0.6),
        # attempts=0 is falsy -> pass_attempts is used. offense_pct 1.0 is not scaled.
        # target_share 0.0 is a real value and must be kept.
        _row(week=1, game_id="g1", team="KC", player_id="p3", position="rb",
             carries=10, attempts=0, pass_attempts=4, targets=2,
             offense_pct=1.0, target_share=0.0),
        _row(week=1, game_id="g1", team="BUF", player_id="p4", position="wr",
             attempts=5, targets=6, offense_pct=75, target_share=0.2),
        # Week 2, game g2. No player id: team totals count, the player does not.
        _row(week=2, game_id="g2", team="KC", position="qb", attempts=30),
        _row(week=2, game_id="g2", team="KC", player_id="p1", position="WR",
             carries=0, attempts=40, targets=10, offense_pct=90, target_share=0.3),
        # offense_pct missing -> snap share None.
        _row(week=2, game_id="g2", team="BUF", player_id="p2", position="RB",
             carries=15, targets=1, target_share=0.1),
        _row(week=2, game_id="g2", team="KC", player_id="p3", position="RB",
             carries=8, targets=4, offense_pct=50),
        # Empty player id: team totals count, the player does not.
        _row(week=2, game_id="g2", team="BUF", player_id="", attempts=6),
        # No game id: the player counts but no team totals are recorded, and the
        # carry lookup key ("", "BUF") misses.
        _row(week=2, team="BUF", player_id="p6", position="WR",
             carries=4, targets=5, offense_pct=60, target_share=0.15),
        # Week 3, game g3. DEN has zero carries, so its player is skipped for carry share.
        _row(week=3, game_id="g3", team="DEN", player_id="p5", position="TE",
             carries=0, attempts=35, targets=7, offense_pct=70, target_share=0.2),
        _row(week=3, game_id="g3", team="BUF", player_id="p2", position="RB",
             carries=9, offense_pct=0.9, target_share=0.05),
        # p4 changes position from WR to TE; the last value wins.
        _row(week=3, game_id="g3", team="BUF", player_id="p4", position="TE",
             targets=2, offense_pct=40),
        # String week. Empty team -> no team totals, and the carry lookup key misses.
        # position None -> "".
        _row(week="3", game_id="g3", team="", player_id="p7",
             carries=3, targets=4, target_share=0.5),
        # DEN plays two week-1 games. Both sort to week 1, in insertion order.
        _row(week=1, game_id="g6", team="DEN", player_id="p9", position="RB",
             carries=5, targets=1, offense_pct=0.5),
        _row(week=1, game_id="g7", team="DEN", player_id="p9", position="RB",
             carries=6, targets=3, offense_pct=0.5, target_share=0.4),
        # Week None -> 0. Zero-valued fields are kept, not treated as missing.
        _row(week=None, game_id="g5", team="KC", player_id="p8", position="WR",
             carries=0, attempts=0, pass_attempts=0, targets=0, offense_pct=0),
        # Player-only row in week 5. The week range comes from game_week only, so
        # this row never reaches the 1..4 grid as a prior week.
        _row(week=5, player_id="p10", position="RB", offense_pct=50, target_share=0.4),
    ]


def _no_game_rows() -> list[dict[str, Any]]:
    """Player rows with no game id or team: game_week stays empty, so max_week falls back to 18."""
    return [_row(week=2, player_id="p9x", position="rb", offense_pct=50, target_share=0.4)]


_GOLDEN_MAIN: dict[str, Any] = {
    "team_carries_by_game": {
        ("g1", "KC"): 10.0,
        ("g1", "BUF"): 12.0,
        ("g2", "KC"): 8.0,
        ("g2", "BUF"): 15.0,
        ("g3", "DEN"): 0.0,
        ("g3", "BUF"): 9.0,
        ("g6", "DEN"): 5.0,
        ("g7", "DEN"): 6.0,
        ("g5", "KC"): 0.0,
    },
    "team_pass_by_game": {
        ("g1", "KC"): 4.0,
        ("g1", "BUF"): 7.0,
        ("g2", "KC"): 70.0,
        ("g2", "BUF"): 6.0,
        ("g3", "DEN"): 35.0,
        ("g3", "BUF"): 0.0,
        ("g6", "DEN"): 0.0,
        ("g7", "DEN"): 0.0,
        ("g5", "KC"): 0.0,
    },
    "team_rush_by_game": {
        ("g1", "KC"): 10.0,
        ("g1", "BUF"): 12.0,
        ("g2", "KC"): 8.0,
        ("g2", "BUF"): 15.0,
        ("g3", "DEN"): 0.0,
        ("g3", "BUF"): 9.0,
        ("g6", "DEN"): 5.0,
        ("g7", "DEN"): 6.0,
        ("g5", "KC"): 0.0,
    },
    "league_ts": {
        1: {},
        2: {
            "WR": 0.25,
            "RB": 0.2,
            "TE": 0.2,
        },
        3: {
            "WR": 0.21250000000000002,
            "RB": 0.16666666666666666,
            "TE": 0.2,
        },
        4: {
            "WR": 0.21250000000000002,
            "RB": 0.15833333333333335,
            "TE": 0.2,
            "": 0.5,
        },
    },
    "league_snap": {
        1: {
            "WR": 0.0,
        },
        2: {
            "WR": 0.4,
            "RB": 0.7000000000000001,
            "TE": 0.75,
        },
        3: {
            "WR": 0.4833333333333334,
            "RB": 0.6166666666666667,
            "TE": 0.75,
        },
        4: {
            "WR": 0.4833333333333334,
            "RB": 0.6666666666666666,
            "TE": 0.6375,
        },
    },
    "league_carry": {
        1: {},
        2: {
            "WR": 0.0,
            "RB": 1.0,
            "TE": 0.0,
        },
        3: {
            "WR": 0.0,
            "RB": 1.0,
            "TE": 0.0,
        },
        4: {
            "WR": 0.0,
            "RB": 1.0,
            "TE": 0.0,
        },
    },
    "league_tgt_allowed": {
        1: 0.0,
        2: 4.6,
        3: 5.428571428571429,
        4: 5.222222222222222,
    },
    "team_game_volume": {
        "KC": [(0, 0.0, 0.0), (1, 4.0, 10.0), (2, 70.0, 8.0)],
        "BUF": [(1, 7.0, 12.0), (2, 6.0, 15.0), (3, 0.0, 9.0)],
        "DEN": [(1, 0.0, 5.0), (1, 0.0, 6.0), (3, 35.0, 0.0)],
    },
}

_EMPTY_GOLDEN: dict[str, Any] = {
    "team_carries_by_game": {},
    "team_pass_by_game": {},
    "team_rush_by_game": {},
    "league_ts": {week: {} for week in range(1, 20)},
    "league_snap": {week: {} for week in range(1, 20)},
    "league_carry": {week: {} for week in range(1, 20)},
    "league_tgt_allowed": {},
    "team_game_volume": {},
}


def _assert_identical(actual: Any, expected: Any, path: str) -> None:
    """Exact comparison: same container type, same dict key order, same int/float type, same float bits."""
    assert type(actual) is type(expected), f"{path}: {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(expected, dict):
        assert list(actual) == list(expected), f"{path}: keys or key order differ"
        for key in expected:
            _assert_identical(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: length differs"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_identical(a, e, f"{path}[{i}]")
    else:
        assert repr(actual) == repr(expected), f"{path}: {actual!r} != {expected!r}"


def _assert_matches_golden(ctx: SeasonFeatureContext, golden: dict[str, Any]) -> None:
    assert type(ctx) is SeasonFeatureContext
    assert [f.name for f in dataclasses.fields(ctx)] == list(golden)
    for name, expected in golden.items():
        _assert_identical(getattr(ctx, name), expected, name)


def test_main_rows_match_golden_output_exactly() -> None:
    ctx = build_season_feature_context(_main_rows())
    _assert_matches_golden(ctx, _GOLDEN_MAIN)


def test_each_week_gets_its_own_inner_dict() -> None:
    # A shared inner dict would pass repr and == checks but break once one week is written.
    ctx = build_season_feature_context(_main_rows())
    for mapping in (ctx.league_ts, ctx.league_snap, ctx.league_carry):
        inner = list(mapping.values())
        assert len({id(d) for d in inner}) == len(inner)


def test_empty_input_gives_the_default_eighteen_week_grid() -> None:
    ctx = build_season_feature_context([])
    _assert_matches_golden(ctx, _EMPTY_GOLDEN)


def test_player_rows_without_games_use_the_default_eighteen_week_grid() -> None:
    # No game rows, so max_week is 18 and the grid runs 1..19. The week-2 player
    # is a prior observation for weeks 3..19 only. Carry share is skipped because
    # the player has no team-game carries to divide by.
    ctx = build_season_feature_context(_no_game_rows())
    assert list(ctx.league_ts) == list(range(1, 20))
    assert ctx.league_ts[1] == {}
    assert ctx.league_ts[2] == {}
    for week in range(3, 20):
        assert ctx.league_ts[week] == {"RB": 0.4}
        assert ctx.league_snap[week] == {"RB": 0.5}
        assert ctx.league_carry[week] == {}
    assert ctx.league_tgt_allowed == {}
    assert ctx.team_game_volume == {}


def test_spot_values_match_hand_derivation() -> None:
    # Each expected value is written out from the rows in _main_rows(), so a
    # reviewer can check it without running the code.
    ctx = build_season_feature_context(_main_rows())
    # Week 2 RB target share: p3 contributes 0.0 (week 1) and p9 contributes 0.4 (week 1).
    assert ctx.league_ts[2]["RB"] == (0.0 + 0.4) / 2
    # Week 2 RB carry share: p3, p9 and p2 are all 1.0 from week-1 games.
    assert ctx.league_carry[2]["RB"] == 1.0
    # Targets in games before week 2: g1 KC 8+2, g1 BUF 3+6, g6 DEN 1, g7 DEN 3, g5 KC 0.
    assert ctx.league_tgt_allowed[2] == (10.0 + 9.0 + 1.0 + 3.0 + 0.0) / 5
    # Two week-1 DEN games keep insertion order; the week-3 game sorts last.
    assert ctx.team_game_volume["DEN"] == [(1, 0.0, 5.0), (1, 0.0, 6.0), (3, 35.0, 0.0)]
    # Week 4 TE carry share: p4 contributes 0.0 (BUF carries 12); p5 is excluded
    # because DEN had zero team carries in week 3.
    assert ctx.league_carry[4]["TE"] == 0.0


def _order_rows() -> list[dict[str, Any]]:
    """Four WRs with one prior target share each. Their mean depends on summation order in the last bit."""
    return [_row(week=1, player_id=f"w{i}", position="WR", target_share=v) for i, v in enumerate((0.1, 0.2, 0.3, 0.15))]


def test_league_mean_keeps_player_summation_order() -> None:
    # Forward order gives 0.18750000000000003; reversed order gives 0.18749999999999997.
    # Pinning the repr catches any change to the order players are summed in.
    ctx = build_season_feature_context(_order_rows())
    assert ctx.league_ts[2] == {"WR": (0.1 + 0.2 + 0.3 + 0.15) / 4}
    assert repr(ctx.league_ts[2]["WR"]) == "0.18750000000000003"


def test_input_rows_are_not_mutated() -> None:
    rows = _main_rows()
    before = copy.deepcopy(rows)
    build_season_feature_context(rows)
    assert rows == before

