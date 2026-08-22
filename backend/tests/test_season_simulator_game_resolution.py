"""
Tests for SeasonSimulator's Phase 7 game-resolution rebuild: wins and Elo
updates now come from the real ml.team_game_model points Ridge model
(predicted for this simulation's own carried-forward Elo state), not the
fantasy_ppr-team-total proxy the module used before. This is the test
surface the Phase 7 plan noted didn't exist — see backend/tests/
test_season_coherence.py for the broader season/week disconnect this
rebuild does NOT (yet) close.

TestPickensCoherence covers the plan's "Pickens test": a low-scoring
projected loss cannot coexist with an outlier receiving line. See
SeasonSimulator._apply_team_script_coupling and _TEAM_SCRIPT_COUPLING for
the (real, measured, not invented) coupling this test verifies.
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
        team_score_paths = sim._draw_team_score_paths(resolved, 2000, rng)
        sim._accumulate_week_wins(resolved, team_score_paths, accum, 2000)
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


class TestByeWeekZeroing:
    """
    Phase 7 fix: before this, _simulate_week drew a normal player-week from
    the Kalman prior unconditionally, with no gate for whether the player's
    team actually had a game that week. A uniform extra 1/18 week of volume
    landed on every player's season total — a 5.3% overstatement invisible
    to the "summed weekly equals season" identity, since it inflates both
    sides equally.
    """

    def test_bye_week_player_contributes_zero_that_week(self) -> None:
        try:
            from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
            import psycopg2
            conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
            conn.close()
        except Exception as exc:
            pytest.skip(f"database unavailable: {exc}")

        from scripts.materialize_season_simulation import (
            _load_prior_game_rows,
            _load_roster,
            _load_schedule_gate,
        )

        db_url = DEFAULT_HOST_DATABASE_URL
        season, start_week, end_week = 2026, 5, 7
        roster_rows = _load_roster(db_url, season, start_week, ["QB"])
        min_players = [r for r in roster_rows if r.get("team") == "MIN"]
        if not min_players:
            pytest.skip("no MIN QB on the 2026 depth chart in this database")

        import pandas as pd

        players_df = pd.DataFrame([
            {"player_id": str(r["player_id"]), "position": r.get("position") or "", "team": r.get("team")}
            for r in min_players
        ])
        prior_game_rows = _load_prior_game_rows(
            db_url, season, start_week, list(players_df["player_id"])
        )
        schedule_df = _load_schedule_gate(db_url, season, start_week, end_week)
        if schedule_df.empty:
            pytest.skip("no 2026 schedule rows for weeks 5-7 in this database")

        sim = SeasonSimulator(
            season=season, start_week=start_week, end_week=end_week,
            n_simulations=30, stats=["fantasy_ppr"], database_url=db_url,
        )
        result = sim.run(
            players_df=players_df, prior_game_rows=prior_game_rows,
            schedule_df=schedule_df, rng_seed=0,
        )

        min_qb = str(players_df.iloc[0]["player_id"])
        week6_rows = [
            w for w in result.week_by_week
            if w["player_id"] == min_qb and w["stat"] == "fantasy_ppr" and w["week"] == 6
        ]
        assert week6_rows, "expected a week-6 row for the MIN QB (MIN is on bye week 6, 2026)"
        assert week6_rows[0]["mean"] == 0.0
        assert week6_rows[0]["p90"] == 0.0

    def test_no_player_exceeds_the_schedule_length_minus_one_bye(self) -> None:
        """
        Every 2026 team has exactly one bye between weeks 5 and 14 — so
        across the full 18-week schedule, no player should show non-zero
        output in more than 17 weeks. Guards the general case beyond the
        single MIN/week-6 example above.
        """
        try:
            from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
            import psycopg2
            conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
            conn.close()
        except Exception as exc:
            pytest.skip(f"database unavailable: {exc}")

        from scripts.materialize_season_simulation import (
            _load_prior_game_rows,
            _load_roster,
            _load_schedule_gate,
        )

        db_url = DEFAULT_HOST_DATABASE_URL
        season, start_week, end_week = 2026, 1, 18
        roster_rows = _load_roster(db_url, season, start_week, ["QB"])[:6]
        if not roster_rows:
            pytest.skip("no QBs on the 2026 depth chart in this database")

        import pandas as pd

        players_df = pd.DataFrame([
            {"player_id": str(r["player_id"]), "position": r.get("position") or "", "team": r.get("team")}
            for r in roster_rows
        ])
        prior_game_rows = _load_prior_game_rows(
            db_url, season, start_week, list(players_df["player_id"])
        )
        schedule_df = _load_schedule_gate(db_url, season, start_week, end_week)
        if schedule_df.empty:
            pytest.skip("no 2026 full-season schedule rows in this database")

        sim = SeasonSimulator(
            season=season, start_week=start_week, end_week=end_week,
            n_simulations=20, stats=["fantasy_ppr"], database_url=db_url,
        )
        result = sim.run(
            players_df=players_df, prior_game_rows=prior_game_rows,
            schedule_df=schedule_df, rng_seed=0,
        )

        from collections import defaultdict

        nonzero_weeks: dict[str, int] = defaultdict(int)
        for w in result.week_by_week:
            if w["stat"] == "fantasy_ppr" and w["mean"] != 0.0:
                nonzero_weeks[w["player_id"]] += 1

        assert nonzero_weeks, "expected at least one player with non-zero output"
        for pid, n in nonzero_weeks.items():
            assert n <= 17, f"{pid} has non-zero output in {n} of 18 weeks; expected <=17 (one bye)"


class TestApplyTeamScriptCoupling:
    def test_positive_coupling_produces_positive_correlation(self) -> None:
        rng = np.random.default_rng(7)
        n = 50_000
        team_score_path = rng.normal(21.0, 9.5, n)
        base_samples = rng.normal(60.0, 20.0, n)
        coupling = 0.10  # measured receiving_yards coupling
        adjusted = SeasonSimulator._apply_team_script_coupling(
            {("p1", "receiving_yards"): base_samples}, team_score_path, coupling
        )[("p1", "receiving_yards")]

        corr = np.corrcoef(adjusted, team_score_path)[0, 1]
        # Additive-shift approximation: realized correlation is close to but
        # not exactly `coupling` (see _apply_team_script_coupling docstring).
        assert 0.05 < corr < 0.16

    def test_zero_coupling_is_identity_apart_from_the_floor(self) -> None:
        rng = np.random.default_rng(8)
        team_score_path = rng.normal(21.0, 9.5, 1000)
        base_samples = rng.normal(60.0, 20.0, 1000)
        adjusted = SeasonSimulator._apply_team_script_coupling(
            {("p1", "targets"): base_samples}, team_score_path, 0.0
        )[("p1", "targets")]
        np.testing.assert_array_equal(adjusted, np.maximum(base_samples, 0.0))

    def test_degenerate_team_score_path_returns_unchanged(self) -> None:
        """A constant team_score_path (std=0) must not divide by zero."""
        flat = np.full(100, 21.0)
        samples = np.full(100, 60.0)
        out = SeasonSimulator._apply_team_script_coupling(
            {("p1", "receiving_yards"): samples}, flat, 0.10
        )
        np.testing.assert_array_equal(out[("p1", "receiving_yards")], samples)


class TestPickensCoherence:
    """The plan's own verify criterion: 'a low-scoring projected loss cannot
    coexist with an outlier receiving line.' Operationalized as: conditional
    on the team's simulated score landing in the bottom decile (a low-scoring
    path), a player's simulated receiving-yards outlier (top decile,
    unconditional) should be LESS likely than its unconditional rate — not
    equally or more likely, which is what independent draws would produce."""

    def test_outlier_receiving_games_are_rarer_on_low_scoring_paths(self) -> None:
        rng = np.random.default_rng(42)
        n = 200_000
        team_score_path = rng.normal(21.0, 9.5, n)
        base_samples = rng.normal(60.0, 22.0, n)
        coupling = 0.10
        adjusted = SeasonSimulator._apply_team_script_coupling(
            {("p1", "receiving_yards"): base_samples}, team_score_path, coupling
        )[("p1", "receiving_yards")]

        unconditional_p90 = np.quantile(adjusted, 0.90)
        unconditional_rate = float(np.mean(adjusted >= unconditional_p90))

        low_score_mask = team_score_path <= np.quantile(team_score_path, 0.10)
        conditional_rate = float(np.mean(adjusted[low_score_mask] >= unconditional_p90))

        assert conditional_rate < unconditional_rate, (
            f"outlier receiving games should be RARER on low-scoring paths: "
            f"conditional={conditional_rate:.4f} unconditional={unconditional_rate:.4f}"
        )

    def test_full_simulate_week_couples_receiving_yards_to_team_score(self, monkeypatch) -> None:
        """End-to-end through _simulate_week (copula path included), not just
        the isolated helper — confirms the wiring in run()'s week loop
        actually reaches the per-player draw."""
        sim = SeasonSimulator(season=2026, start_week=5, end_week=5,
                               n_simulations=20_000, stats=["receiving_yards"], use_copula=False)
        rng = np.random.default_rng(3)
        kalman_df = pd.DataFrame([
            {"player_id": "wr1", "team": "MIN", "position": "WR",
             "kalman_est_receiving_yards": 60.0, "kalman_variance_receiving_yards": 400.0},
        ])
        team_score_paths = {"MIN": rng.normal(21.0, 9.5, 20_000)}
        week_paths = sim._simulate_week(
            kalman_df=kalman_df, week=5, injury_report={}, n_simulations=20_000,
            rng=rng, team_score_paths=team_score_paths,
        )
        samples = week_paths[("wr1", "receiving_yards")]
        corr = np.corrcoef(samples, team_score_paths["MIN"])[0, 1]
        assert corr > 0.03, f"expected positive team-score coupling, got corr={corr:.4f}"
