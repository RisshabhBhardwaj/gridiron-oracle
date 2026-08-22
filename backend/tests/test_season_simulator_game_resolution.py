"""
Tests for SeasonSimulator's Phase 7 game-resolution rebuild: wins and Elo
updates now come from the real ml.team_game_model points Ridge model
(predicted for this simulation's own carried-forward Elo state), not the
fantasy_ppr-team-total proxy the module used before. This is the test
surface the Phase 7 plan noted didn't exist — see backend/tests/
test_season_coherence.py for the broader season/week disconnect this
rebuild does NOT (yet) close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.season_simulator import SeasonSimulator, load_real_schedule


def _fake_train_df(n_seasons: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for season in range(2019, 2019 + n_seasons):
        for i in range(40):
            rows.append({
                "season": season, "week": (i % 18) + 1,
                "team": "MIN", "opponent": "GB", "is_home": i % 2,
                "rest": 7.0, "opp_rest": 7.0,
                "team_off_elo": 1500.0 + rng.normal(0, 50),
                "team_def_elo": 1500.0 + rng.normal(0, 50),
                "opp_off_elo": 1500.0 + rng.normal(0, 50),
                "opp_def_elo": 1500.0 + rng.normal(0, 50),
                "prior_coach_pass_rate": 0.58,
                "is_dome": 0, "is_turf": 0,
                "points": 21.0 + rng.normal(0, 9),
            })
    return pd.DataFrame(rows)


def _fake_forward_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "game_id": "2026_05_MIN_GB", "team": "MIN", "opponent": "GB",
            "is_home": 1, "rest": 7.0, "opp_rest": 7.0,
            "team_off_elo": 1300.0, "team_def_elo": 1300.0,  # stale DB values
            "opp_off_elo": 1300.0, "opp_def_elo": 1300.0,
            "prior_coach_pass_rate": 0.58, "is_dome": 0, "is_turf": 0,
        },
        {
            "game_id": "2026_05_MIN_GB", "team": "GB", "opponent": "MIN",
            "is_home": 0, "rest": 7.0, "opp_rest": 7.0,
            "team_off_elo": 1300.0, "team_def_elo": 1300.0,
            "opp_off_elo": 1300.0, "opp_def_elo": 1300.0,
            "prior_coach_pass_rate": 0.58, "is_dome": 0, "is_turf": 0,
        },
    ])


class _FakeEloSystem:
    """Mimics the two-method surface SeasonSimulator actually uses."""

    def __init__(self, ratings: dict[str, tuple[float, float]]):
        self._ratings = dict(ratings)
        self.update_week_calls: list[list] = []

    def get_current_ratings(self, team: str) -> tuple[float, float]:
        return self._ratings.get(team, (1500.0, 1500.0))

    def update_week(self, results: list) -> None:
        self.update_week_calls.append(results)
        for r in results:
            ho, hd = self.get_current_ratings(r.home_team)
            ao, ad = self.get_current_ratings(r.away_team)
            if r.home_score > r.away_score:
                self._ratings[r.home_team] = (ho + 10, hd)
                self._ratings[r.away_team] = (ao - 10, ad)
            elif r.away_score > r.home_score:
                self._ratings[r.away_team] = (ao + 10, ad)
                self._ratings[r.home_team] = (ho - 10, hd)


class TestResolveWeekGamesOverridesElo:
    def test_uses_simulation_elo_not_db_elo(self, monkeypatch) -> None:
        """The forward frame from the DB carries stale Elo (no rows for
        unplayed weeks) — _resolve_week_games must override it with this
        simulation's own carried-forward ratings before predicting."""
        sim = SeasonSimulator(season=2026, start_week=5, end_week=5, n_simulations=4)

        monkeypatch.setattr(
            "pipeline.team_game_features.build_team_game_frame",
            lambda db_url, seasons: _fake_train_df(),
        )
        monkeypatch.setattr(
            "pipeline.team_game_features.build_team_game_forward_frame",
            lambda db_url, season, week: _fake_forward_frame(),
        )
        sim_elo = _FakeEloSystem({"MIN": (1700.0, 1400.0), "GB": (1350.0, 1550.0)})
        monkeypatch.setattr("ml.team_elo.load_elo_from_db", lambda db_url, seasons: sim_elo)

        captured = {}
        from ml.team_game_model import _prepare_x as real_prepare_x

        def spy_prepare_x(df, fill_values):
            captured["frame"] = df.copy()
            return real_prepare_x(df, fill_values)

        monkeypatch.setattr("ml.team_game_model._prepare_x", spy_prepare_x)

        resolved = sim._resolve_week_games(5)

        assert not resolved.empty
        frame = captured["frame"]
        min_row = frame[frame["team"] == "MIN"].iloc[0]
        assert min_row["team_off_elo"] == 1700.0  # sim state, not the stale 1300 from the DB frame
        assert min_row["team_def_elo"] == 1400.0
        assert min_row["opp_off_elo"] == 1350.0  # GB's sim off_elo
        assert "points_mean" in resolved.columns


class TestEnsurePointsModelFailsLoud:
    def test_raises_when_no_training_rows(self, monkeypatch) -> None:
        sim = SeasonSimulator(season=2026, start_week=5, end_week=5, n_simulations=4)
        monkeypatch.setattr(
            "pipeline.team_game_features.build_team_game_frame",
            lambda db_url, seasons: pd.DataFrame(),
        )
        with pytest.raises(RuntimeError, match="no team-game training rows"):
            sim._ensure_points_model()


class TestEnsureSimEloUsesDbBackedLoader:
    def test_calls_load_elo_from_db_not_flat_singleton(self, monkeypatch) -> None:
        """Regression guard: the old code called ml.team_elo.get_elo_system()
        with no db_url, which silently returns an unfit, flat-1500 system
        with zero real team-quality signal. _ensure_sim_elo must call
        load_elo_from_db(db_url, ...) instead."""
        sim = SeasonSimulator(season=2026, start_week=5, end_week=5, n_simulations=4,
                               database_url="postgresql://fake")
        calls = []

        def fake_load(db_url, seasons):
            calls.append((db_url, list(seasons)))
            return _FakeEloSystem({})

        monkeypatch.setattr("ml.team_elo.load_elo_from_db", fake_load)
        sim._ensure_sim_elo()
        assert calls == [("postgresql://fake", list(range(2019, 2027)))]


class TestWinsAndEloAgree:
    def test_favored_team_wins_more_and_gains_elo(self, monkeypatch) -> None:
        """Wins (_accumulate_week_wins) and Elo movement (_update_elo_from_week)
        must be derived from the SAME resolved points predictions — the whole
        point of the rebuild. A team predicted for more points should both
        win more often across paths and gain Elo."""
        sim = SeasonSimulator(season=2026, start_week=5, end_week=5, n_simulations=2000)
        sim._points_residual_std = 6.0
        sim._sim_elo = _FakeEloSystem({"MIN": (1500.0, 1500.0), "GB": (1500.0, 1500.0)})
        resolved = pd.DataFrame([
            {"game_id": "g1", "team": "MIN", "opponent": "GB", "is_home": 1, "points_mean": 27.0},
            {"game_id": "g1", "team": "GB", "opponent": "MIN", "is_home": 0, "points_mean": 17.0},
        ])
        accum = {}
        rng = np.random.default_rng(1)
        sim._accumulate_week_wins(resolved, accum, 2000, rng)
        sim._update_elo_from_week(resolved, 5)

        assert accum["MIN"].mean() > accum["GB"].mean()
        min_off, _ = sim._sim_elo.get_current_ratings("MIN")
        assert min_off > 1500.0  # MIN was favored and "won" the Elo update


class TestLoadRealSchedule:
    def test_builds_home_away_pairs_from_forward_frame(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "pipeline.team_game_features.build_team_game_forward_frame",
            lambda db_url, season, week: _fake_forward_frame(),
        )
        schedule = load_real_schedule("postgresql://fake", 2026, [5])
        assert list(schedule.columns) == ["week", "home_team", "away_team"]
        assert len(schedule) == 1
        assert schedule.iloc[0]["home_team"] == "MIN"
        assert schedule.iloc[0]["away_team"] == "GB"


class TestLiveDbIntegration:
    """Exercises the real DB + real Ridge fit end-to-end. Skips cleanly if
    no database is reachable, matching this repo's convention elsewhere
    (e.g. backend/tests/test_drive_engine.py) for tests that need real
    infrastructure rather than mocks."""

    def test_resolve_week_games_against_real_2026_schedule(self) -> None:
        try:
            from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
            import psycopg2
            conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
            conn.close()
        except Exception as exc:
            pytest.skip(f"database unavailable: {exc}")

        sim = SeasonSimulator(season=2026, start_week=5, end_week=5, n_simulations=10)
        resolved = sim._resolve_week_games(5)
        if resolved.empty:
            pytest.skip("no 2026 week 5 games in this database")
        assert "points_mean" in resolved.columns
        assert resolved["points_mean"].notna().all()
        # LA (Rams) is the one franchise whose ml.team_elo abbreviation
        # ("LAR") historically diverged from the pipeline's own ("LA") —
        # confirm the join didn't silently fall back to a flat default for it.
        if "LA" in set(resolved["team"]) | set(resolved["opponent"]):
            la_row = resolved[resolved["team"] == "LA"]
            if not la_row.empty:
                assert la_row.iloc[0]["team_off_elo"] != 1500.0 or la_row.iloc[0]["team_def_elo"] != 1500.0
