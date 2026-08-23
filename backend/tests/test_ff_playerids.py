"""
backend/tests/test_ff_playerids.py

Unit tests for scraper/adapters/ff_playerids.py's _opt_str() helper.

Regression test for a bug found during verification of the Sleeper->GSIS
player-id bridge (fantasy_player_ids table): pandas reads numeric-ID columns
(sleeper_id, espn_id, mfl_id, ...) as float64 whenever the source column
contains NaN, so integer ids arrive as Python floats (e.g. 13269.0).
str(13269.0) == "13269.0", which corrupted every id in the table and broke
joins against sleeper_league_draft_picks.player_id (stored as a clean
integer string, e.g. "13269").

No HTTP or DB calls required.
"""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scraper.adapters.ff_playerids import _opt_str


class TestOptStr:
    def test_whole_number_float_serializes_without_trailing_zero(self):
        assert _opt_str(13269.0) == "13269"

    def test_negative_whole_number_float(self):
        assert _opt_str(-42.0) == "-42"

    def test_zero_float(self):
        assert _opt_str(0.0) == "0"

    def test_nan_maps_to_none(self):
        assert _opt_str(float("nan")) is None
        assert _opt_str(math.nan) is None

    def test_none_maps_to_none(self):
        assert _opt_str(None) is None

    def test_non_integer_float_preserved(self):
        # Not expected in practice for id columns, but should not raise
        # and should not silently truncate a genuinely fractional value.
        assert _opt_str(13269.5) == "13269.5"

    def test_plain_string_unaffected(self):
        assert _opt_str("13269") == "13269"

    def test_string_with_whitespace_is_stripped(self):
        assert _opt_str("  13269  ") == "13269"

    def test_empty_string_maps_to_none(self):
        assert _opt_str("") is None
        assert _opt_str("   ") is None

    def test_int_input_unaffected(self):
        assert _opt_str(13269) == "13269"
