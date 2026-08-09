"""Tests for season constants and incomplete-season capping."""

from __future__ import annotations

import warnings

import pytest

from ml.season_constants import (
    CURRENT_SEASON,
    LAST_COMPLETE_SEASON,
    TRAIN_SEASON_START,
    assert_not_fitting_incomplete_season,
    assert_seasons_within_cap,
    cap_seasons,
    train_seasons,
    train_seasons_arg,
)
from ml.utils import _parse_seasons
from scraper.adapters.rules_parser import RulesParser


def test_train_seasons_end_at_last_complete() -> None:
    seasons = train_seasons()
    assert seasons[0] == 2019
    assert seasons[-1] == LAST_COMPLETE_SEASON
    assert CURRENT_SEASON not in seasons or CURRENT_SEASON <= LAST_COMPLETE_SEASON


def test_parse_seasons_raises_on_incomplete() -> None:
    """Audit C-28: this used to drop 2026 with a warning and return 2019–2025,
    so a trainer asked for an incomplete season trained on something else and
    reported success."""
    with pytest.raises(ValueError, match="exceed the season cap"):
        _parse_seasons("2019-2026")

    assert train_seasons_arg() == f"2019-{LAST_COMPLETE_SEASON}"


def test_parse_seasons_raises_on_single_out_of_range_season() -> None:
    with pytest.raises(ValueError, match="exceed the season cap"):
        _parse_seasons(str(CURRENT_SEASON + 5))


def test_parse_seasons_incomplete_allowed_still_caps_at_current() -> None:
    """`complete_only=False` widens the cap to CURRENT_SEASON — it does not
    remove it. A season that does not exist yet is never a valid request."""
    assert _parse_seasons(str(CURRENT_SEASON), complete_only=False) == [CURRENT_SEASON]
    with pytest.raises(ValueError, match="exceed the season cap"):
        _parse_seasons(str(CURRENT_SEASON + 1), complete_only=False)


def test_parse_seasons_does_not_clamp_the_lower_bound() -> None:
    """Audit C-28: `TRAIN_SEASON_START` is a default, not a floor. Clamping it
    rewrote `2018-2024` as `2019-2024` without telling the caller."""
    assert _parse_seasons("2018-2024") == list(range(2018, 2025))
    assert cap_seasons([2017, 2018, 2019]) == [2017, 2018, 2019]
    assert TRAIN_SEASON_START == 2019  # still the documented default start


def test_cap_seasons_drops_only_the_upper_bound() -> None:
    assert cap_seasons([2019, LAST_COMPLETE_SEASON, CURRENT_SEASON]) == [
        2019,
        LAST_COMPLETE_SEASON,
    ]
    assert cap_seasons([CURRENT_SEASON], complete_only=False) == [CURRENT_SEASON]


def test_assert_seasons_within_cap() -> None:
    assert_seasons_within_cap(train_seasons())
    with pytest.raises(ValueError, match="exceed the season cap"):
        assert_seasons_within_cap([2019, CURRENT_SEASON])


def test_assert_not_fitting_incomplete_season() -> None:
    with pytest.raises(ValueError):
        assert_not_fitting_incomplete_season([2019, CURRENT_SEASON + 1])


@pytest.mark.parametrize(
    "module",
    ["ml/xgb_model.py", "ml/lgbm_model.py", "ml/catboost_model.py", "ml/tft_cli.py"],
)
def test_trainer_entrypoints_call_the_assert(module: str) -> None:
    """Audit C-28: the hard assert existed but was never called in production
    code, so the pre-Week-1 guarantee rested on a log line."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / module).read_text()
    assert "assert_not_fitting_incomplete_season(seasons)" in source, (
        f"{module} must assert the season cap at its CLI entrypoint"
    )


def test_rules_parser_has_2025_and_2026() -> None:
    rp = RulesParser()
    assert rp.get_season_features(2025)["kickoff_unified_rule"] == 1.0
    assert rp.get_season_features(2026)["dynamic_kickoff_refined"] == 1.0


def test_rules_parser_unknown_season_warns() -> None:
    rp = RulesParser()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feats = rp.get_season_features(2099)
    assert feats["kickoff_unified_rule"] == 1.0
    assert any("not catalogued" in str(w.message) for w in caught)
