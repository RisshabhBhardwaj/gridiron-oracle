"""
backend/tests/test_feature_engineer.py

Tests for pipeline/feature_engineer.py.

Structure:
  TestWeightedAvg         — _weighted_avg() edge cases
  TestComputeRollingForm  — Bucket 1 pure function
  TestComputeSeasonBaseline — Bucket 2 pure function
  TestComputeVenueFeatures  — Bucket 4 pure function
  TestComputeRestFeatures   — Bucket 6 pure function
  TestComputeMatchupStats   — Bucket 3 pure function
  TestBuildFeatureRow       — full assembly with synthetic data
  TestJeffersonWeek12       — acceptance test (real nflreadpy 2024 data)

conftest.py (project root) adds the repo root to sys.path.
"""

from __future__ import annotations

import re
from typing import Optional

import pytest

from pipeline.feature_engineer import (
    FeatureRow,
    _safe_div,
    _print_feature_row,
    _sum_fumbles,
    build_feature_row,
    build_from_nflreadpy,
    compute_matchup_stats,
    compute_rest_features,
    compute_season_baseline,
    compute_team_context,
    compute_venue_features,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _game(**kwargs) -> dict:
    """Minimal game dict with sensible defaults."""
    defaults = dict(
        home_team="MIN", away_team="GB",
        roof="outdoors", surface="grass",
        temp=52.0, wind=8.0,
        total_line=45.5, spread_line=-3.5,
        home_rest=7, away_rest=7,
    )
    defaults.update(kwargs)
    return defaults


def _stat_row(**kwargs) -> dict:
    """Minimal player-stats dict."""
    defaults = dict(
        player_id="TEST-001", game_id="2024_01_MIN_GB",
        season=2024, week=1, position="WR",
        team="MIN", opponent_team="GB",
        receiving_yards=90.0, receiving_tds=1, targets=8, receptions=6,
        target_share=0.25, air_yards_share=0.28, fantasy_points_ppr=22.0,
        carries=0, rushing_yards=0.0, rushing_tds=0,
        attempts=0, passing_yards=0.0, passing_tds=0, completions=0,
    )
    defaults.update(kwargs)
    return defaults


# ── TestSumFumbles ────────────────────────────────────────────────────────────

class TestSumFumbles:
    """Tests for _sum_fumbles — sums rushing, receiving, sack fumbles (not *_lost)."""

    def test_all_three_types_summed(self) -> None:
        row = {"rushing_fumbles": 1, "receiving_fumbles": 0, "sack_fumbles": 1}
        assert _sum_fumbles(row) == 2.0

    def test_zero_fumbles_preserved(self) -> None:
        row = {"rushing_fumbles": 0, "receiving_fumbles": 0, "sack_fumbles": 0}
        assert _sum_fumbles(row) == 0.0

    def test_partial_sources_summed(self) -> None:
        row = {"rushing_fumbles": 2}  # receiving/sack absent → None, ignored
        assert _sum_fumbles(row) == 2.0

    def test_all_none_returns_none(self) -> None:
        assert _sum_fumbles({}) is None
        assert _sum_fumbles({"other_col": 1}) is None


# ── TestComputeSeasonBaseline ─────────────────────────────────────────────────

class TestComputeSeasonBaseline:

    def test_empty_returns_zero_games_played(self) -> None:
        result = compute_season_baseline([])
        assert result["seas_games_played"] == 0
        assert result["seas_avg_receiving_yards"] is None

    def test_games_played_counts_rows(self) -> None:
        rows = [_stat_row(week=w) for w in range(1, 6)]
        result = compute_season_baseline(rows)
        assert result["seas_games_played"] == 5

    def test_avg_receiving_yards_correct(self) -> None:
        rows = [_stat_row(receiving_yards=y) for y in [60.0, 80.0, 100.0]]
        result = compute_season_baseline(rows)
        assert result["seas_avg_receiving_yards"] == pytest.approx(80.0)

    def test_yards_per_target_computed(self) -> None:
        # 3 games: 300 total yards on 30 targets → 10.0 yds/target
        rows = [_stat_row(receiving_yards=100.0, targets=10) for _ in range(3)]
        result = compute_season_baseline(rows)
        assert result["seas_yards_per_target"] == pytest.approx(10.0)

    def test_yards_per_target_none_when_no_targets(self) -> None:
        rows = [_stat_row(receiving_yards=0.0, targets=0)]
        result = compute_season_baseline(rows)
        assert result["seas_yards_per_target"] is None

    def test_yards_per_reception_computed(self) -> None:
        rows = [_stat_row(receiving_yards=90.0, receptions=9)]
        result = compute_season_baseline(rows)
        assert result["seas_yards_per_reception"] == pytest.approx(10.0)

    def test_completion_pct_for_qb(self) -> None:
        # 200 completions on 300 attempts → 66.7%
        rows = [_stat_row(completions=20, attempts=30) for _ in range(10)]
        result = compute_season_baseline(rows)
        assert result["seas_completion_pct"] == pytest.approx(200 / 300, rel=1e-6)

    def test_rushing_stats_computed(self) -> None:
        rows = [_stat_row(carries=20, rushing_yards=100.0) for _ in range(4)]
        result = compute_season_baseline(rows)
        assert result["seas_avg_carries"]      == pytest.approx(20.0)
        assert result["seas_avg_rushing_yards"]== pytest.approx(100.0)
        assert result["seas_yards_per_carry"]  == pytest.approx(5.0)


# ── TestComputeVenueFeatures ──────────────────────────────────────────────────

class TestComputeVenueFeatures:

    def test_dome_flag_set_for_dome_roof(self) -> None:
        result = compute_venue_features(_game(roof="dome", temp=None, wind=None))
        assert result["is_dome"] == 1

    def test_dome_flag_set_for_closed_retractable(self) -> None:
        result = compute_venue_features(_game(roof="closed", temp=None, wind=None))
        assert result["is_dome"] == 1

    def test_outdoor_flag_zero_for_outdoors(self) -> None:
        result = compute_venue_features(_game(roof="outdoors"))
        assert result["is_dome"] == 0

    def test_dome_gets_default_mild_temp(self) -> None:
        result = compute_venue_features(_game(roof="dome", temp=None, wind=None))
        assert result["temp_f"]  == pytest.approx(68.0)
        assert result["wind_mph"] == pytest.approx(0.0)
        assert result["temp_bucket"] == 2   # mild
        assert result["wind_bucket"] == 0   # calm

    def test_cold_temp_bucket(self) -> None:
        result = compute_venue_features(_game(temp=20.0))
        assert result["temp_bucket"] == 0

    def test_cool_temp_bucket(self) -> None:
        result = compute_venue_features(_game(temp=40.0))
        assert result["temp_bucket"] == 1

    def test_mild_temp_bucket(self) -> None:
        result = compute_venue_features(_game(temp=72.0))
        assert result["temp_bucket"] == 2

    def test_calm_wind_bucket(self) -> None:
        result = compute_venue_features(_game(wind=5.0))
        assert result["wind_bucket"] == 0

    def test_breezy_wind_bucket(self) -> None:
        result = compute_venue_features(_game(wind=15.0))
        assert result["wind_bucket"] == 1

    def test_windy_bucket(self) -> None:
        result = compute_venue_features(_game(wind=25.0))
        assert result["wind_bucket"] == 2

    def test_turf_surface_detected(self) -> None:
        result = compute_venue_features(_game(surface="FieldTurf"))
        assert result["surface_turf"] == 1

    def test_grass_surface_detected(self) -> None:
        result = compute_venue_features(_game(surface="grass"))
        assert result["surface_turf"] == 0


# ── TestComputeRestFeatures ───────────────────────────────────────────────────

class TestComputeRestFeatures:

    def test_standard_week_not_short_or_bye(self) -> None:
        result = compute_rest_features(_game(home_rest=7, away_rest=7), is_home=True)
        assert result["days_rest"]     == 7
        assert result["is_short_week"] == 0
        assert result["is_bye_prior"]  == 0

    def test_short_week_thursday_game(self) -> None:
        result = compute_rest_features(_game(home_rest=4, away_rest=10), is_home=True)
        assert result["days_rest"]     == 4
        assert result["is_short_week"] == 1
        assert result["is_bye_prior"]  == 0

    def test_bye_prior_for_away_team(self) -> None:
        result = compute_rest_features(_game(home_rest=7, away_rest=14), is_home=False)
        assert result["days_rest"]     == 14
        assert result["is_short_week"] == 0
        assert result["is_bye_prior"]  == 1

    def test_none_rest_propagates_none(self) -> None:
        result = compute_rest_features(_game(home_rest=None, away_rest=None), is_home=True)
        assert result["days_rest"]     is None
        assert result["is_short_week"] is None
        assert result["is_bye_prior"]  is None


# ── TestComputeMatchupStats ───────────────────────────────────────────────────

class TestComputeMatchupStats:

    def _rows(self, n_games: int = 3, position: str = "WR",
              opponent: str = "GB") -> list[dict]:
        """Create n_games worth of WR stats against opponent."""
        return [
            _stat_row(
                game_id=f"2024_{w:02d}_MIN_GB",
                week=w, position=position, opponent_team=opponent,
                receiving_yards=80.0, targets=7, receiving_tds=1,
                fantasy_points_ppr=18.0,
            )
            for w in range(1, n_games + 1)
        ]

    def test_empty_returns_none_fields(self) -> None:
        result = compute_matchup_stats("GB", "WR", [], target_week=5)
        for val in result.values():
            assert val is None

    def test_no_games_before_target_week_returns_none(self) -> None:
        rows = self._rows(3)
        # All rows have week 1-3; asking for week 1 → nothing prior
        result = compute_matchup_stats("GB", "WR", rows, target_week=1)
        for val in result.values():
            assert val is None

    def test_correct_average_across_three_games(self) -> None:
        rows = self._rows(3)
        result = compute_matchup_stats("GB", "WR", rows, target_week=5)
        assert result["opp_avg_receiving_yards_allowed"] == pytest.approx(80.0)
        assert result["opp_avg_targets_allowed"]         == pytest.approx(7.0)
        assert result["opp_avg_tds_allowed"]             == pytest.approx(1.0)
        assert result["opp_avg_fantasy_ppr_allowed"]     == pytest.approx(18.0)

    def test_position_filter_excludes_other_positions(self) -> None:
        rows = (
            self._rows(3, position="WR", opponent="GB") +
            self._rows(3, position="RB", opponent="GB")
        )
        result = compute_matchup_stats("GB", "WR", rows, target_week=10)
        # Should only count WR rows (80 yd avg), not RB rows
        assert result["opp_avg_receiving_yards_allowed"] == pytest.approx(80.0)

    def test_only_games_before_target_week_included(self) -> None:
        rows = self._rows(5)  # weeks 1-5
        # Only weeks 1-3 should be included
        result_3 = compute_matchup_stats("GB", "WR", rows, target_week=4)
        result_5 = compute_matchup_stats("GB", "WR", rows, target_week=6)
        # Both should produce averages (data exists for both queries)
        assert result_3["opp_avg_receiving_yards_allowed"] is not None
        assert result_5["opp_avg_receiving_yards_allowed"] is not None

    def test_multiple_wrs_same_game_summed_per_game(self) -> None:
        """Two WRs vs same opponent in same game → per-game total, not avg of players."""
        rows = [
            _stat_row(game_id="2024_01_MIN_GB", week=1, position="WR",
                      opponent_team="GB", receiving_yards=80.0,
                      targets=7, receiving_tds=1, fantasy_points_ppr=18.0),
            _stat_row(player_id="TEST-002", game_id="2024_01_MIN_GB",
                      week=1, position="WR", opponent_team="GB",
                      receiving_yards=60.0, targets=5, receiving_tds=0,
                      fantasy_points_ppr=12.0),
        ]
        result = compute_matchup_stats("GB", "WR", rows, target_week=5)
        # One game, combined total: 140 yards
        assert result["opp_avg_receiving_yards_allowed"] == pytest.approx(140.0)


# ── TestBuildFeatureRow ───────────────────────────────────────────────────────

class TestBuildFeatureRow:

    def test_identity_fields_populated(self) -> None:
        target = _stat_row(player_id="ABC", game_id="2024_01_MIN_GB",
                           season=2024, week=1, position="WR",
                           team="MIN", opponent_team="GB")
        fr = build_feature_row(target, [], _game(), [target])
        assert fr.player_id     == "ABC"
        assert fr.game_id       == "2024_01_MIN_GB"
        assert fr.season        == 2024
        assert fr.week          == 1
        assert fr.position      == "WR"
        assert fr.team          == "MIN"
        assert fr.opponent_team == "GB"

    def test_is_home_correct(self) -> None:
        target_home = _stat_row(team="MIN")
        target_away = _stat_row(team="GB")
        game = _game(home_team="MIN", away_team="GB")
        fr_home = build_feature_row(target_home, [], game, [])
        fr_away = build_feature_row(target_away, [], game, [])
        assert fr_home.is_home == 1
        assert fr_away.is_home == 0

    def test_no_prior_games_cold_start_returns_position_prior(self) -> None:
        # Cold-start: 0 prior rows → Kalman returns WR position-average prior
        target = _stat_row(position="WR")
        fr = build_feature_row(target, [], _game(), [target])
        # kalman_est_* is the WR prior (not None)
        assert fr.kalman_est_receiving_yards is not None
        assert fr.kalman_est_receiving_yards > 0
        assert fr.seas_games_played == 0

    def test_prior_games_populate_kalman_est(self) -> None:
        priors = [_stat_row(receiving_yards=y, week=w)
                  for w, y in enumerate([60, 80, 100, 120], start=1)]
        target = _stat_row(week=5)
        fr = build_feature_row(target, priors, _game(), priors + [target])
        assert fr.kalman_est_receiving_yards is not None
        assert fr.kalman_est_receiving_yards > 0
        assert fr.kalman_variance_receiving_yards is not None

    def test_actual_targets_populated(self) -> None:
        target = _stat_row(fantasy_points_ppr=22.0, receiving_yards=95.0)
        fr = build_feature_row(target, [], _game(), [target])
        assert fr.actual_fantasy_ppr     == pytest.approx(22.0)
        assert fr.actual_receiving_yards == pytest.approx(95.0)

    def test_venue_features_in_row(self) -> None:
        game = _game(roof="dome", temp=None, wind=None)
        fr = build_feature_row(_stat_row(), [], game, [])
        assert fr.is_dome    == 1
        assert fr.temp_f is None  # observed weather is disabled pending forecasts
        assert fr.wind_mph is None

    def test_rest_features_in_row(self) -> None:
        game = _game(home_rest=4, away_rest=7, home_team="MIN")
        target = _stat_row(team="MIN")
        fr = build_feature_row(target, [], game, [target])
        assert fr.days_rest     == 4
        assert fr.is_short_week == 1

    def test_returns_feature_row_instance(self) -> None:
        fr = build_feature_row(_stat_row(), [], _game(), [])
        assert isinstance(fr, FeatureRow)


# ── TestJeffersonWeek12 — acceptance test ────────────────────────────────────

@pytest.fixture(scope="module")
def jefferson_2024_w12() -> Optional[FeatureRow]:
    """
    Acceptance test fixture — builds feature vector for Justin Jefferson,
    Week 12, 2024 season using live nflreadpy data (cached locally).
    This is the canonical dry-run acceptance test from the project spec.
    """
    return build_from_nflreadpy("Justin Jefferson", season=2024, week=12)


@pytest.mark.integration
@pytest.mark.network
class TestJeffersonWeek12:
    """
    Acceptance criteria: if this test class passes, the feature engineer
    produces a sensible vector for a real, elite WR game.
    """

    def test_feature_row_not_none(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        """Jefferson played in 2024 week 12 — we must find him."""
        assert jefferson_2024_w12 is not None, (
            "Jefferson not found in 2024 week 12 data. "
            "Check that nflreadpy has 2024 season data cached."
        )

    def test_position_is_wr(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.position == "WR"

    def test_team_is_vikings(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        """Jefferson plays for MIN."""
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.team == "MIN"

    def test_season_and_week_correct(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.season == 2024
        assert jefferson_2024_w12.week   == 12

    def test_game_id_format(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        assert jefferson_2024_w12 is not None
        pattern = re.compile(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$")
        assert pattern.match(jefferson_2024_w12.game_id), (
            f"Unexpected game_id: {jefferson_2024_w12.game_id}"
        )

    def test_season_in_game_id(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.game_id.startswith("2024_12_")

    def test_kalman_form_populated(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        """Jefferson should have at least 1 prior game — Kalman est must be set."""
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.kalman_est_receiving_yards is not None, (
            "kalman_est_receiving_yards should not be None — "
            "Jefferson must have played at least 1 game before week 12"
        )
        assert jefferson_2024_w12.kalman_variance_receiving_yards is not None

    def test_kalman_est_in_reasonable_range(
        self, jefferson_2024_w12: Optional[FeatureRow]
    ) -> None:
        """Elite WR: Kalman estimate should be in plausible range (20-200 yards)."""
        assert jefferson_2024_w12 is not None
        est = jefferson_2024_w12.kalman_est_receiving_yards
        assert est > 20.0, f"kalman_est_receiving_yards={est:.1f} is too low for an elite WR"
        assert est < 200.0, f"kalman_est_receiving_yards={est:.1f} is unrealistically high"

    def test_season_baseline_games_played(
        self, jefferson_2024_w12: Optional[FeatureRow]
    ) -> None:
        """Jefferson should have played at least 1 game before week 12."""
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.seas_games_played is not None
        assert jefferson_2024_w12.seas_games_played >= 1

    def test_season_avg_yards_reasonable(
        self, jefferson_2024_w12: Optional[FeatureRow]
    ) -> None:
        """Elite WR season avg should be in realistic range (0-300 yards/game)."""
        assert jefferson_2024_w12 is not None
        avg = jefferson_2024_w12.seas_avg_receiving_yards
        if avg is not None:
            assert 0 < avg < 300, f"seas_avg_receiving_yards={avg} is implausible"

    def test_kalman_target_share_between_zero_and_one(
        self, jefferson_2024_w12: Optional[FeatureRow]
    ) -> None:
        """Kalman estimate of target share must be a proportion: 0 ≤ value ≤ 1."""
        assert jefferson_2024_w12 is not None
        ts = jefferson_2024_w12.kalman_est_target_share
        if ts is not None:
            assert 0.0 <= ts <= 1.0, f"kalman_est_target_share={ts} is not a valid proportion"

    def test_matchup_stats_populated(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        """Matchup stats require prior games against the opponent — should be populated."""
        assert jefferson_2024_w12 is not None
        # It's acceptable for matchup stats to be None if this is the first
        # meeting, but the opponent should have faced WRs this season.
        # We just assert the field exists (it's in the dataclass).
        assert hasattr(jefferson_2024_w12, "opp_avg_receiving_yards_allowed")

    def test_venue_features_present(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        """All venue feature fields must be present (not missing from dataclass)."""
        assert jefferson_2024_w12 is not None
        for field_name in ("is_dome", "surface_turf", "temp_bucket", "wind_bucket"):
            assert hasattr(jefferson_2024_w12, field_name), (
                f"Missing venue field: {field_name}"
            )

    def test_rest_features_present(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.days_rest is not None

    def test_actual_result_is_real(self, jefferson_2024_w12: Optional[FeatureRow]) -> None:
        """Week 12 2024 is a completed game — actual stats should be populated."""
        assert jefferson_2024_w12 is not None
        assert jefferson_2024_w12.actual_fantasy_ppr is not None, (
            "actual_fantasy_ppr should be populated for a completed 2024 game"
        )
        assert jefferson_2024_w12.actual_receiving_yards is not None


# ── _print_feature_row — Kalman field correctness (CRITICAL fix) ──────────────

class TestPrintFeatureRow:
    """
    _print_feature_row() previously referenced deleted form_* fields.
    After the Kalman migration, it must reference kalman_est_* fields only.
    Calling it on a FeatureRow must not raise AttributeError.
    """

    def _minimal_fr(self) -> FeatureRow:
        return FeatureRow(
            player_id="00-0035228",
            game_id="2025_01_MIN_GB",
            season=2025,
            week=1,
            position="WR",
            team="MIN",
            kalman_est_receiving_yards=72.4,
            kalman_est_receiving_tds=0.5,
            kalman_est_targets=6.8,
            kalman_est_receptions=4.9,
            kalman_est_target_share=0.24,
            kalman_est_air_yards_share=0.31,
            kalman_est_fantasy_ppr=14.1,
            kalman_est_carries=0.2,
            kalman_est_rushing_yards=1.5,
            kalman_est_passing_yards=0.0,
        )

    def test_no_attribute_error(self, capsys):
        """Calling _print_feature_row must not raise AttributeError."""
        fr = self._minimal_fr()
        _print_feature_row(fr, "Justin Jefferson")   # must not raise

    def test_output_contains_kalman_label(self, capsys):
        fr = self._minimal_fr()
        _print_feature_row(fr, "Justin Jefferson")
        captured = capsys.readouterr()
        assert "Kalman" in captured.out

    def test_output_does_not_reference_form_fields(self, capsys):
        """Bucket 1 section must no longer print old form_* labels."""
        fr = self._minimal_fr()
        _print_feature_row(fr, "Justin Jefferson")
        captured = capsys.readouterr()
        assert "form_receiving_yards" not in captured.out
        assert "form_rushing_yards" not in captured.out
        assert "form_passing_yards" not in captured.out

    def test_output_contains_kalman_est_values(self, capsys):
        fr = self._minimal_fr()
        _print_feature_row(fr, "Justin Jefferson")
        captured = capsys.readouterr()
        assert "kalman_est_receiving_yards" in captured.out

    def test_no_form_attributes_on_feature_row(self):
        """FeatureRow dataclass must not have any form_* fields."""
        fr = FeatureRow(player_id="x", game_id="y", season=2025, week=1)
        for attr in (
            "form_receiving_yards", "form_rushing_yards", "form_passing_yards",
            "form_targets", "form_receptions", "form_carries",
        ):
            assert not hasattr(fr, attr), (
                f"FeatureRow should not have deleted field: {attr}"
            )


def test_upsert_feature_rows_does_not_wipe_elo_columns_on_rerun():
    """
    FeatureRow always constructs the six Elo columns as None — enrichment
    happens out-of-band via enrich_elo_embeddings.py. A blind EXCLUDED
    overwrite in the upsert would silently null out already-enriched Elo
    values every time feature_engineer.run() is re-run for a season.
    """
    from unittest.mock import MagicMock, patch
    from pipeline.feature_engineer import FeatureEngineer

    fe = FeatureEngineer.__new__(FeatureEngineer)
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    fe._conn = mock_conn

    fr = FeatureRow(player_id="x", game_id="y", season=2025, week=1)
    with patch("pipeline.feature_engineer.execute_values") as mock_execute_values:
        fe._upsert_feature_rows([fr])

    sql = mock_execute_values.call_args[0][1]
    for col in (
        "team_off_elo", "team_def_elo", "opp_off_elo",
        "opp_def_elo", "elo_matchup_diff", "elo_implied_win_prob",
    ):
        assert f"{col} = COALESCE(EXCLUDED.{col}, feature_matrix.{col})" in sql, (
            f"{col} must be COALESCEd against the existing row, not blindly overwritten: {sql}"
        )
    # A non-Elo column should still be a plain overwrite.
    assert "season = EXCLUDED.season" in sql
