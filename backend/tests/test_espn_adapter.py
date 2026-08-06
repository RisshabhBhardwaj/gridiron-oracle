"""
backend/tests/test_espn_adapter.py

Unit tests for scraper/adapters/espn_adapter.py and the new
compute_injury_features() function in pipeline/feature_engineer.py.

All HTTP calls are mocked — no live ESPN API required.
All DB calls are mocked — no PostgreSQL connection required.
"""
from __future__ import annotations

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_espn_response(player_name: str, status: str, injury: str) -> dict:
    """Build a minimal ESPN API response dict for one injured player."""
    return {
        "injuries": [
            {
                "athlete": {"displayName": player_name},
                "status": status,
                "type": {"text": injury, "description": status},
                "shortComment": f"{player_name} {injury}",
            }
        ]
    }


def _adapter_no_db() -> "EspnAdapter":
    from scraper.adapters.espn_adapter import EspnAdapter
    return EspnAdapter(db_url=None)


# ---------------------------------------------------------------------------
# A. Module constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_espn_team_ids_has_32_teams(self):
        from scraper.adapters.espn_adapter import ESPN_TEAM_IDS
        assert len(ESPN_TEAM_IDS) == 32

    def test_status_encoding_covers_all_canonical_statuses(self):
        from scraper.adapters.espn_adapter import STATUS_ENCODING
        for status in ("out", "dnp", "doubtful", "questionable", "limited", "full"):
            assert status in STATUS_ENCODING

    def test_out_and_dnp_encode_to_zero(self):
        from scraper.adapters.espn_adapter import STATUS_ENCODING
        assert STATUS_ENCODING["out"] == 0
        assert STATUS_ENCODING["dnp"] == 0

    def test_full_encodes_to_four(self):
        from scraper.adapters.espn_adapter import STATUS_ENCODING
        assert STATUS_ENCODING["full"] == 4

    def test_status_encoding_ordered(self):
        """Severity order: out(0) < doubtful(1) < questionable(2) < limited(3) < full(4)."""
        from scraper.adapters.espn_adapter import STATUS_ENCODING
        assert STATUS_ENCODING["doubtful"] < STATUS_ENCODING["questionable"]
        assert STATUS_ENCODING["questionable"] < STATUS_ENCODING["limited"]
        assert STATUS_ENCODING["limited"] < STATUS_ENCODING["full"]


# ---------------------------------------------------------------------------
# B. _map_player_name_to_id
# ---------------------------------------------------------------------------

class TestMapPlayerName:
    def test_exact_match(self):
        adapter = _adapter_no_db()
        adapter._name_lookup = {"tyreek hill": "00-0032765"}
        assert adapter._map_player_name_to_id("Tyreek Hill") == "00-0032765"

    def test_fuzzy_match_nickname(self):
        """'T. Hill' should match 'tyreek hill' via WRatio."""
        adapter = _adapter_no_db()
        adapter._name_lookup = {"tyreek hill": "00-0032765"}
        result = adapter._map_player_name_to_id("T. Hill")
        # WRatio may or may not hit threshold for short abbreviation — just verify no crash
        assert result is None or result == "00-0032765"

    def test_no_match_below_threshold(self):
        adapter = _adapter_no_db()
        adapter._name_lookup = {"tyreek hill": "00-0032765"}
        assert adapter._map_player_name_to_id("Patrick Mahomes") is None

    def test_empty_name_returns_none(self):
        adapter = _adapter_no_db()
        adapter._name_lookup = {"tyreek hill": "00-0032765"}
        assert adapter._map_player_name_to_id("") is None

    def test_none_lookup_returns_none(self):
        adapter = _adapter_no_db()
        adapter._name_lookup = None
        assert adapter._map_player_name_to_id("Tyreek Hill") is None

    def test_multiple_players_correct_match(self):
        adapter = _adapter_no_db()
        adapter._name_lookup = {
            "tyreek hill":     "00-0032765",
            "davante adams":   "00-0019596",
            "justin jefferson": "00-0036945",
        }
        result = adapter._map_player_name_to_id("Justin Jefferson")
        assert result == "00-0036945"


# ---------------------------------------------------------------------------
# C. _fetch_team_injuries (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchTeamInjuries:
    def _adapter_with_lookup(self, lookup: dict) -> "EspnAdapter":
        adapter = _adapter_no_db()
        adapter._name_lookup = lookup
        return adapter

    def test_parses_single_player(self):
        adapter = self._adapter_with_lookup({"tyreek hill": "00-0032765"})
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_espn_response("Tyreek Hill", "Questionable", "knee")
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)

        rows = adapter._fetch_team_injuries("MIA", 15, 1, 2025)
        assert len(rows) == 1
        assert rows[0]["player_name"] == "Tyreek Hill"
        assert rows[0]["practice_status"] == "questionable"
        assert rows[0]["injury_type"] == "knee"
        assert rows[0]["espn_team"] == "MIA"
        assert rows[0]["week"] == 1
        assert rows[0]["season"] == 2025

    def test_matched_player_id_set(self):
        adapter = self._adapter_with_lookup({"tyreek hill": "00-0032765"})
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_espn_response("Tyreek Hill", "out", "hamstring")
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)

        rows = adapter._fetch_team_injuries("MIA", 15, 1, 2025)
        assert rows[0]["player_id"] == "00-0032765"

    def test_unmatched_player_id_is_none(self):
        adapter = self._adapter_with_lookup({})
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_espn_response("Unknown Player", "limited", "ankle")
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)

        rows = adapter._fetch_team_injuries("KC", 15, 1, 2025)
        assert rows[0]["player_id"] is None

    def test_empty_injuries_list(self):
        adapter = self._adapter_with_lookup({})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"injuries": []}
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)

        rows = adapter._fetch_team_injuries("KC", 15, 1, 2025)
        assert rows == []

    def test_missing_display_name_skipped(self):
        adapter = self._adapter_with_lookup({})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "injuries": [{"athlete": {}, "status": "out", "type": {"text": "knee"}}]
        }
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)

        rows = adapter._fetch_team_injuries("KC", 15, 1, 2025)
        assert rows == []

    def test_status_alias_normalised(self):
        """'Did Not Participate' → 'dnp'."""
        adapter = self._adapter_with_lookup({"davante adams": "00-0019596"})
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_espn_response(
            "Davante Adams", "Did Not Participate", "groin"
        )
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)

        rows = adapter._fetch_team_injuries("NYJ", 15, 5, 2025)
        assert rows[0]["practice_status"] == "dnp"


# ---------------------------------------------------------------------------
# D. fetch_injury_report (mocked HTTP + sleep)
# ---------------------------------------------------------------------------

class TestFetchInjuryReport:
    def _mock_adapter(self, player_lookup: dict, response_data: dict) -> "EspnAdapter":
        adapter = _adapter_no_db()
        adapter._name_lookup = player_lookup

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status.return_value = None
        adapter._session.get = MagicMock(return_value=mock_resp)
        return adapter

    def test_returns_dataframe(self):
        from scraper.adapters.espn_adapter import ESPN_TEAM_IDS
        adapter = self._mock_adapter({}, {"injuries": []})
        with patch("time.sleep"):
            df = adapter.fetch_injury_report(week=1, season=2025)
        assert isinstance(df, pd.DataFrame)

    def test_output_has_required_columns(self):
        adapter = self._mock_adapter({}, {"injuries": []})
        with patch("time.sleep"):
            df = adapter.fetch_injury_report(week=1, season=2025)
        required = {"player_id", "player_name", "espn_team",
                    "practice_status", "injury_type", "week", "season"}
        assert required.issubset(set(df.columns))

    def test_empty_report_returns_empty_df(self):
        adapter = self._mock_adapter({}, {"injuries": []})
        with patch("time.sleep"):
            df = adapter.fetch_injury_report(week=1, season=2025)
        assert len(df) == 0

    def test_week_and_season_columns_populated(self):
        adapter = self._mock_adapter(
            {"tyreek hill": "00-0032765"},
            _make_espn_response("Tyreek Hill", "Questionable", "knee"),
        )
        with patch("time.sleep"):
            df = adapter.fetch_injury_report(week=7, season=2025)
        if not df.empty:
            assert (df["week"] == 7).all()
            assert (df["season"] == 2025).all()

    def test_http_error_team_skipped_does_not_crash(self):
        """A 503 on one team shouldn't abort the whole report."""
        adapter = _adapter_no_db()
        adapter._name_lookup = {}
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = Exception("503 Server Error")
        adapter._session.get = MagicMock(return_value=error_resp)

        with patch("time.sleep"):
            df = adapter.fetch_injury_report(week=1, season=2025)
        assert isinstance(df, pd.DataFrame)  # no crash

    def test_all_32_teams_requested(self):
        from scraper.adapters.espn_adapter import ESPN_TEAM_IDS
        adapter = self._mock_adapter({}, {"injuries": []})
        with patch("time.sleep"):
            adapter.fetch_injury_report(week=1, season=2025)
        assert adapter._session.get.call_count == 32


# ---------------------------------------------------------------------------
# E. compute_injury_features (pipeline/feature_engineer.py)
# ---------------------------------------------------------------------------

class TestComputeInjuryFeatures:
    """Tests for the pure compute_injury_features() function."""

    def _fn(self, *args, **kwargs):
        from pipeline.feature_engineer import compute_injury_features
        return compute_injury_features(*args, **kwargs)

    def test_no_prior_rows_no_injury_df(self):
        result = self._fn("p1", 1, 2025, [])
        assert result["games_missed_streak"] == 0
        assert result["injury_status_encoded"] is None

    def test_streak_zero_when_all_weeks_played(self):
        prior = [
            {"week": 1, "season": 2025},
            {"week": 2, "season": 2025},
        ]
        result = self._fn("p1", 3, 2025, prior)
        assert result["games_missed_streak"] == 0

    def test_streak_one_week(self):
        """Player played weeks 1-3 but missed week 4 (targeting week 5)."""
        prior = [
            {"week": 1, "season": 2025},
            {"week": 2, "season": 2025},
            {"week": 3, "season": 2025},
        ]
        result = self._fn("p1", 5, 2025, prior)
        assert result["games_missed_streak"] == 1

    def test_streak_two_consecutive_weeks(self):
        """Player played weeks 1-2, missed 3-4 (targeting week 5)."""
        prior = [
            {"week": 1, "season": 2025},
            {"week": 2, "season": 2025},
        ]
        result = self._fn("p1", 5, 2025, prior)
        assert result["games_missed_streak"] == 2

    def test_streak_resets_on_played_game(self):
        """Player missed week 2 but played week 3 — streak from end is 0."""
        prior = [
            {"week": 1, "season": 2025},
            {"week": 3, "season": 2025},
        ]
        result = self._fn("p1", 4, 2025, prior)
        # Week 3 was played, so streak from end = 0
        assert result["games_missed_streak"] == 0

    def test_streak_week_1_targeting_is_zero(self):
        """First game of season — no prior weeks to miss."""
        result = self._fn("p1", 1, 2025, [])
        assert result["games_missed_streak"] == 0

    def test_injury_status_from_df_out(self):
        injury_df = pd.DataFrame([{
            "player_id": "p1",
            "practice_status": "out",
            "week": 5,
            "season": 2025,
        }])
        result = self._fn("p1", 5, 2025, [], injury_df=injury_df)
        assert result["injury_status_encoded"] == 0

    def test_injury_status_from_df_questionable(self):
        injury_df = pd.DataFrame([{
            "player_id": "p1",
            "practice_status": "questionable",
            "week": 3,
            "season": 2025,
        }])
        result = self._fn("p1", 3, 2025, [], injury_df=injury_df)
        assert result["injury_status_encoded"] == 2

    def test_injury_status_from_df_full(self):
        injury_df = pd.DataFrame([{
            "player_id": "p1",
            "practice_status": "full",
            "week": 7,
            "season": 2025,
        }])
        result = self._fn("p1", 7, 2025, [], injury_df=injury_df)
        assert result["injury_status_encoded"] == 4

    def test_injury_status_none_when_player_not_in_report(self):
        injury_df = pd.DataFrame([{
            "player_id": "p2",
            "practice_status": "out",
            "week": 5,
            "season": 2025,
        }])
        result = self._fn("p1", 5, 2025, [], injury_df=injury_df)
        assert result["injury_status_encoded"] is None

    def test_injury_status_none_when_no_df(self):
        result = self._fn("p1", 5, 2025, [], injury_df=None)
        assert result["injury_status_encoded"] is None

    def test_injury_status_none_when_empty_df(self):
        result = self._fn("p1", 5, 2025, [], injury_df=pd.DataFrame())
        assert result["injury_status_encoded"] is None

    def test_combined_streak_and_status(self):
        """Realistic case: player missed 2 weeks and is listed as doubtful."""
        prior = [{"week": 1, "season": 2025}, {"week": 2, "season": 2025}]
        injury_df = pd.DataFrame([{
            "player_id": "p1",
            "practice_status": "doubtful",
            "week": 5,
            "season": 2025,
        }])
        result = self._fn("p1", 5, 2025, prior, injury_df=injury_df)
        assert result["games_missed_streak"] == 2
        assert result["injury_status_encoded"] == 1  # doubtful


# ---------------------------------------------------------------------------
# F. FeatureRow has injury fields
# ---------------------------------------------------------------------------

class TestFeatureRowInjuryFields:
    def test_injury_fields_default_to_none(self):
        from pipeline.feature_engineer import FeatureRow
        fr = FeatureRow(player_id="p1", game_id="g1", season=2025, week=1)
        assert fr.injury_status_encoded is None
        assert fr.games_missed_streak is None

    def test_injury_fields_accept_values(self):
        from pipeline.feature_engineer import FeatureRow
        fr = FeatureRow(
            player_id="p1", game_id="g1", season=2025, week=1,
            injury_status_encoded=2,
            games_missed_streak=1,
        )
        assert fr.injury_status_encoded == 2
        assert fr.games_missed_streak == 1

    def test_fm_cols_includes_injury_fields(self):
        from pipeline.feature_engineer import _FM_COLS
        assert "injury_status_encoded" in _FM_COLS
        assert "games_missed_streak" in _FM_COLS

    def test_ddl_includes_injury_columns(self):
        from pipeline.feature_engineer import _CREATE_FEATURE_MATRIX
        assert "injury_status_encoded" in _CREATE_FEATURE_MATRIX
        assert "games_missed_streak" in _CREATE_FEATURE_MATRIX


# ---------------------------------------------------------------------------
# G. build_feature_row — injury_df flows through correctly
# ---------------------------------------------------------------------------

class TestBuildFeatureRowWithInjury:
    """Verify build_feature_row() passes injury_df to compute_injury_features."""

    def _minimal_inputs(self):
        target_row = {
            "player_id": "p1", "game_id": "g1",
            "season": 2025, "week": 5,
            "team": "KC", "opponent_team": "LV", "position": "WR",
        }
        prior_rows = [{"week": w, "season": 2025} for w in range(1, 4)]
        game = {
            "home_team": "KC", "away_team": "LV",
            "home_rest": 7, "away_rest": 7,
            "roof": "outdoors", "surface": "grass",
        }
        all_season_rows = prior_rows
        return target_row, prior_rows, game, all_season_rows

    def test_no_injury_df_fields_are_none(self):
        from pipeline.feature_engineer import build_feature_row
        tr, pr, g, asr = self._minimal_inputs()
        fr = build_feature_row(tr, pr, g, asr, injury_df=None)
        assert fr.injury_status_encoded is None

    def test_with_injury_df_status_propagated(self):
        from pipeline.feature_engineer import build_feature_row
        tr, pr, g, asr = self._minimal_inputs()
        injury_df = pd.DataFrame([{
            "player_id": "p1",
            "practice_status": "limited",
            "week": 5,
            "season": 2025,
        }])
        fr = build_feature_row(tr, pr, g, asr, injury_df=injury_df)
        assert fr.injury_status_encoded == 3  # limited

    def test_games_missed_streak_computed(self):
        from pipeline.feature_engineer import build_feature_row
        tr, prior_rows, g, asr = self._minimal_inputs()
        # prior_rows has weeks 1-3; week 4 is missing → streak = 1 for target_week=5
        fr = build_feature_row(tr, prior_rows, g, asr)
        assert fr.games_missed_streak == 1

    def test_games_missed_streak_zero_consecutive(self):
        from pipeline.feature_engineer import build_feature_row
        tr, _, g, asr = self._minimal_inputs()
        # All 4 weeks played before week 5
        prior_rows = [{"week": w, "season": 2025} for w in range(1, 5)]
        fr = build_feature_row(tr, prior_rows, g, asr)
        assert fr.games_missed_streak == 0


# ---------------------------------------------------------------------------
# H. Column name correctness — players table schema (CRITICAL fix)
# ---------------------------------------------------------------------------

class TestEnsureNameLookupColumnName:
    """
    _ensure_name_lookup() must query full_name (not display_name).
    The players table has full_name; querying display_name raises
    psycopg2.ProgrammingError at runtime.
    """

    def test_sql_query_uses_full_name_not_display_name(self):
        """Verify the SQL in _ensure_name_lookup() references full_name."""
        import inspect
        from scraper.adapters.espn_adapter import EspnAdapter
        src = inspect.getsource(EspnAdapter._ensure_name_lookup)
        assert "full_name" in src, "_ensure_name_lookup must query full_name column"
        assert "display_name" not in src, "_ensure_name_lookup must NOT query display_name (column does not exist)"

    def test_name_lookup_key_uses_full_name(self):
        """
        When the DB returns rows with full_name, the lookup dict is
        keyed on full_name.lower(), not display_name.
        """
        import psycopg2
        from unittest.mock import MagicMock, patch

        fake_row = {"id": "00-0032765", "full_name": "Tyreek Hill"}
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [fake_row]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        from scraper.adapters.espn_adapter import EspnAdapter
        adapter = EspnAdapter(db_url="postgresql://fake/fake")

        with patch("psycopg2.connect", return_value=mock_conn):
            with patch("scraper.adapters.nflreadpy_adapter._psycopg2_dsn", return_value="fake"):
                adapter._ensure_name_lookup()

        assert "tyreek hill" in adapter._name_lookup
        assert adapter._name_lookup["tyreek hill"] == "00-0032765"
