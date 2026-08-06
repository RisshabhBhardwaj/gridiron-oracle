"""
backend/tests/test_team_elo.py

Unit tests for ml/team_elo.py.
All tests run without a database connection.
"""

import math
import pytest
import pandas as pd

from ml.team_elo import (
    TeamEloSystem,
    GameResult,
    EloConfig,
    INITIAL_ELO,
    _expected_score,
    _mov_multiplier,
    get_elo_system,
    reset_elo_system,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_game(home: str, away: str, home_score: int, away_score: int, season: int = 2024, week: int = 1) -> GameResult:
    return GameResult(season=season, week=week, home_team=home, away_team=away,
                      home_score=home_score, away_score=away_score)


def _make_games_df(games: list[tuple]) -> pd.DataFrame:
    """Build a DataFrame from (home, away, home_score, away_score) tuples."""
    rows = [
        {"season": 2024, "week": i + 1,
         "home_team": h, "away_team": a,
         "home_score": hs, "away_score": as_}
        for i, (h, a, hs, as_) in enumerate(games)
    ]
    return pd.DataFrame(rows)


# ── ELO MATH TESTS ────────────────────────────────────────────────────────────

class TestEloMath:
    def test_expected_score_equal_elo(self):
        """Equal Elo teams should each have ~0.5 expected score."""
        p = _expected_score(1500, 1500)
        assert abs(p - 0.5) < 0.01

    def test_expected_score_higher_elo_wins_more(self):
        """Higher Elo team should have expected score > 0.5."""
        p = _expected_score(1600, 1400)
        assert p > 0.6

    def test_expected_score_bounded(self):
        """Expected score must be in (0, 1)."""
        for elo_a, elo_b in [(1000, 2000), (2000, 1000), (1500, 1500)]:
            p = _expected_score(elo_a, elo_b)
            assert 0 < p < 1

    def test_expected_scores_sum_to_one(self):
        """P(A wins) + P(B wins) = 1."""
        for elo_a, elo_b in [(1500, 1600), (1200, 1800)]:
            assert abs(_expected_score(elo_a, elo_b) + _expected_score(elo_b, elo_a) - 1.0) < 1e-6

    def test_mov_multiplier_small_margin(self):
        """Small MOV → small multiplier."""
        mult = _mov_multiplier(3, 200)
        assert 1.0 <= mult <= 1.5

    def test_mov_multiplier_large_margin(self):
        """Large MOV → larger multiplier, capped at 3.0."""
        mult = _mov_multiplier(35, 200)
        assert mult <= 3.0
        assert mult > 1.5

    def test_mov_multiplier_cap(self):
        """Very large margin should be capped at max."""
        mult = _mov_multiplier(200, 200)
        assert mult <= 3.0

    def test_mov_multiplier_blowout_vs_close(self):
        """Blowout win → higher multiplier than close win."""
        mult_blowout = _mov_multiplier(28, 200)
        mult_close   = _mov_multiplier(3, 200)
        assert mult_blowout > mult_close


# ── GAME RESULT ────────────────────────────────────────────────────────────────

class TestGameResult:
    def test_home_won(self):
        r = _make_game("MIN", "GB", home_score=28, away_score=21)
        assert r.home_won is True

    def test_away_won(self):
        r = _make_game("MIN", "GB", home_score=14, away_score=21)
        assert r.home_won is False

    def test_tie(self):
        r = _make_game("MIN", "GB", home_score=17, away_score=17)
        assert r.is_tie is True
        assert r.home_won is False

    def test_not_tie(self):
        r = _make_game("MIN", "GB", home_score=14, away_score=17)
        assert r.is_tie is False

    def test_score_diff(self):
        r = _make_game("MIN", "GB", home_score=28, away_score=14)
        assert r.score_diff == 14


# ── TEAM ELO SYSTEM ───────────────────────────────────────────────────────────

class TestTeamEloSystem:
    def test_initial_all_teams_equal(self):
        """All 32 NFL teams start at default Elo."""
        elo = TeamEloSystem()
        cfg = EloConfig()
        off, def_ = elo.get_current_ratings("KC")
        assert off == cfg.initial_elo
        assert def_ == cfg.initial_elo

    def test_unknown_team_returns_default(self):
        """Unknown team returns default Elo (not error)."""
        elo = TeamEloSystem()
        off, def_ = elo.get_current_ratings("FAKE_TEAM")
        assert off == EloConfig().initial_elo

    def test_update_winner_elo_increases(self):
        """Winning team's offensive Elo should increase."""
        elo = TeamEloSystem()
        initial_off, _ = elo.get_current_ratings("MIN")
        elo.update_week([_make_game("MIN", "GB", 28, 14)])
        new_off, _ = elo.get_current_ratings("MIN")
        assert new_off > initial_off

    def test_update_loser_elo_decreases(self):
        """Losing team's offensive Elo should decrease."""
        elo = TeamEloSystem()
        initial_off, _ = elo.get_current_ratings("GB")
        elo.update_week([_make_game("MIN", "GB", 28, 14)])
        new_off, _ = elo.get_current_ratings("GB")
        assert new_off < initial_off

    def test_zero_sum_elo_change(self):
        """Aggregate Elo across all teams should remain constant."""
        elo = TeamEloSystem()
        games = [
            _make_game("KC", "BUF", 31, 17),
            _make_game("MIN", "GB",  28, 21),
            _make_game("SF", "DAL",  14, 28),
        ]
        all_teams = list(elo._off_elo.keys())
        total_before = sum(elo._off_elo[t] + elo._def_elo[t] for t in all_teams)
        elo.update_week(games)
        total_after = sum(elo._off_elo[t] + elo._def_elo[t] for t in all_teams)
        # Total should be approximately zero-sum (within rounding)
        assert abs(total_after - total_before) < 1.0

    def test_win_probability_equal_teams(self):
        """Equal Elo teams should have ~50% win probability."""
        elo = TeamEloSystem()
        wp = elo.win_probability("MIN", "GB")
        assert 0.4 < wp < 0.6  # home advantage gives MIN slight edge

    def test_rankings_returns_all_teams(self):
        """rankings() should return a row for every team."""
        elo = TeamEloSystem()
        r = elo.rankings()
        assert len(r) >= 32

    def test_fit_from_games_df(self):
        """fit() from DataFrame updates Elo correctly."""
        elo = TeamEloSystem()
        games_df = _make_games_df([
            ("KC", "BUF", 31, 17),  # KC wins
            ("MIN", "GB", 28, 14),  # MIN wins
        ])
        elo.fit(games_df)
        kc_off, _ = elo.get_current_ratings("KC")
        buf_off, _ = elo.get_current_ratings("BUF")
        assert kc_off > buf_off

    def test_season_regression(self):
        """apply_season_regression() should move Elo toward mean."""
        elo = TeamEloSystem()
        # Simulate KC winning a lot
        games = [_make_game("KC", "BUF", 35, 7, season=2024, week=i+1) for i in range(8)]
        elo.update_week(games)
        high_elo_before, _ = elo.get_current_ratings("KC")
        elo.apply_season_regression(2024)
        high_elo_after, _ = elo.get_current_ratings("KC")
        # KC's high Elo should regress toward mean
        assert high_elo_after < high_elo_before

    def test_enrich_features_adds_columns(self):
        """enrich_features() should add all 6 Elo columns to the DataFrame."""
        elo = TeamEloSystem()
        df = pd.DataFrame([
            {"player_id": "p1", "team": "MIN", "opponent_team": "GB", "season": 2024, "week": 5},
            {"player_id": "p2", "team": "KC",  "opponent_team": "BUF","season": 2024, "week": 5},
        ])
        enriched = elo.enrich_features(df, season=2024, week=5)
        for col in ["team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
                    "elo_matchup_diff", "elo_implied_win_prob"]:
            assert col in enriched.columns, f"Missing column: {col}"
            assert enriched[col].notna().all(), f"Column {col} has NaN values"

    def test_enrich_features_implied_prob_bounds(self):
        """elo_implied_win_prob should be in (0, 1)."""
        elo = TeamEloSystem()
        df = pd.DataFrame([
            {"player_id": "p1", "team": "KC", "opponent_team": "ARI", "season": 2024, "week": 1},
        ])
        games_df = _make_games_df([("KC", "ARI", 35, 7)] * 5)
        elo.fit(games_df)
        enriched = elo.enrich_features(df, season=2024, week=1)
        prob = enriched["elo_implied_win_prob"].iloc[0]
        assert 0.0 < prob < 1.0

    def test_persistence_roundtrip(self, tmp_path):
        """save() / load() should preserve Elo ratings."""
        elo = TeamEloSystem()
        elo.update_week([_make_game("KC", "BUF", 35, 7)])
        path = tmp_path / "elo.json"
        elo.save(str(path))

        elo2 = TeamEloSystem()
        elo2.load(str(path))
        kc_orig, _ = elo.get_current_ratings("KC")
        kc_loaded, _ = elo2.get_current_ratings("KC")
        assert abs(kc_orig - kc_loaded) < 0.01


# ── SINGLETON ─────────────────────────────────────────────────────────────────

class TestEloSingleton:
    def setup_method(self):
        reset_elo_system()

    def teardown_method(self):
        reset_elo_system()

    def test_get_elo_system_returns_same_instance(self):
        s1 = get_elo_system()
        s2 = get_elo_system()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        s1 = get_elo_system()
        reset_elo_system()
        s2 = get_elo_system()
        assert s1 is not s2
