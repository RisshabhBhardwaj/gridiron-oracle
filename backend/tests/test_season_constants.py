"""Tests for season constants and incomplete-season capping."""

from __future__ import annotations

import warnings

import pytest

from ml.season_constants import (
    CURRENT_SEASON,
    LAST_COMPLETE_SEASON,
    assert_not_fitting_incomplete_season,
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


def test_parse_seasons_drops_incomplete() -> None:
    parsed = _parse_seasons("2019-2026")
    assert CURRENT_SEASON not in parsed or CURRENT_SEASON <= LAST_COMPLETE_SEASON
    assert parsed[-1] == LAST_COMPLETE_SEASON
    assert train_seasons_arg() == f"2019-{LAST_COMPLETE_SEASON}"


def test_assert_not_fitting_incomplete_season() -> None:
    with pytest.raises(ValueError):
        assert_not_fitting_incomplete_season([2019, CURRENT_SEASON + 1])


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
