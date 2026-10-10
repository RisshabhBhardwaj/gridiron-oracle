"""
backend/tests/test_ml_cov_run_backtest.py

Coverage tests for ml/run_backtest.py (DB replay backtest provider functions).

The SQLModel engine and session are replaced with in-test fakes: sessions
return preset rows in query order (game logs, then feature-matrix rows, then
projections). Nothing connects to a database. The trust policy loader is
stubbed. Expected baselines are worked out by hand on a three-player frame:

  - prev-season mean (naive baseline) for 2024 uses the 2023 mean.
  - trailing-3 mean (rolling baseline) uses only earlier 2024 weeks.
  - the cohort drops each player's first game of the season (no prior game).

The script's __main__ block (the full backtest run) is not executed here.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ml.run_backtest as rb


def _gl(player_id, season, week, position, **stats) -> SimpleNamespace:
    base = {
        "player_id": player_id, "game_id": f"{season}_{week:02d}_{player_id}", "season": season,
        "week": week, "position": position, "offense_pct": 0.8,
        "attempts": None, "completions": None, "passing_yards": None, "passing_tds": None,
        "passing_interceptions": None, "carries": None, "rushing_yards": None, "rushing_tds": None,
        "receptions": None, "targets": None, "receiving_yards": None, "receiving_tds": None,
        "fantasy_points_ppr": None, "rushing_fumbles": None, "receiving_fumbles": None,
        "sack_fumbles": None,
    }
    base.update(stats)
    return SimpleNamespace(**base)


def _history() -> list[SimpleNamespace]:
    rows = [
        # P1 WR: 2023 mean receiving yards = 60.
        _gl("P1", 2023, 1, "WR", receiving_yards=50.0),
        _gl("P1", 2023, 2, "WR", receiving_yards=70.0),
        # P1 WR 2024: week 1 is the cohort exclusion (no prior game this season).
        _gl("P1", 2024, 1, "WR", receiving_yards=80.0, receptions=5.0, targets=7.0, receiving_fumbles=1.0),
        _gl("P1", 2024, 2, "WR", receiving_yards=90.0, receptions=6.0, targets=8.0, receiving_fumbles=0.0),
        _gl("P1", 2024, 3, "WR", receiving_yards=100.0, receptions=7.0, targets=9.0, receiving_fumbles=2.0),
        # P2 RB: no 2023 rows, so naive baseline must fall back to the rolling baseline.
        _gl("P2", 2024, 1, "RB", carries=10.0, rushing_yards=40.0, rushing_fumbles=1.0),
        _gl("P2", 2024, 2, "RB", carries=12.0, rushing_yards=60.0, rushing_fumbles=0.0),
        # P3 QB: 2023 attempts = 30, with a 2024 game that has no kalman row.
        _gl("P3", 2023, 1, "QB", attempts=30.0, passing_yards=200.0, sack_fumbles=0.0),
        _gl("P3", 2024, 1, "QB", attempts=35.0, passing_yards=220.0, sack_fumbles=1.0),
        _gl("P3", 2024, 2, "QB", attempts=40.0, passing_yards=260.0, sack_fumbles=0.0),
    ]
    return rows


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _FakeSession:
    queue: list = []        # results returned in query order
    engines: list = []

    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def exec(self, statement):
        return _Result(_FakeSession.queue.pop(0))


@pytest.fixture
def fake_db(monkeypatch):
    import sqlmodel

    _FakeSession.queue = []
    _FakeSession.engines = []

    def fake_engine(url, **kwargs):
        _FakeSession.engines.append(url)
        return f"ENGINE<{url}>"

    monkeypatch.setattr(sqlmodel, "create_engine", fake_engine)
    monkeypatch.setattr(sqlmodel, "Session", _FakeSession)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/db")
    return _FakeSession


def _fm(player_id, game_id, **kalman):
    return SimpleNamespace(player_id=player_id, game_id=game_id, **kalman)


# ── _load_gamelog_history ─────────────────────────────────────────────────────

class TestLoadGamelogHistory:
    def test_maps_rows_and_normalises_offense_pct(self, fake_db) -> None:
        rows = [_gl("P9", 2024, 4, "TE", receiving_yards=33.0)]
        rows[0].offense_pct = 85.0  # percent scale -> converted to a fraction
        fake_db.queue.append(rows)

        df = rb._load_gamelog_history([2024])
        assert list(df.columns)[:5] == ["player_id", "game_id", "season", "week", "position"]
        assert df.loc[0, "offense_pct"] == pytest.approx(0.85)
        assert df.loc[0, "receiving_yards"] == pytest.approx(33.0)
        assert "receiving_fumbles" in df.columns
        assert fake_db.engines == ["postgresql://fake/db"]

    def test_missing_fumble_attributes_default_to_none(self, fake_db) -> None:
        row = SimpleNamespace(
            player_id="P9", game_id="g", season=2024, week=1, position="TE", offense_pct=None,
            attempts=None, completions=None, passing_yards=None, passing_tds=None,
            passing_interceptions=None, carries=None, rushing_yards=None, rushing_tds=None,
            receptions=None, targets=None, receiving_yards=None, receiving_tds=None,
            fantasy_points_ppr=None, rushing_fumbles=None,
        )
        fake_db.queue.append([row])
        df = rb._load_gamelog_history([2024])
        assert df.loc[0, "receiving_fumbles"] is None
        assert df.loc[0, "sack_fumbles"] is None


# ── real_data_provider ────────────────────────────────────────────────────────

class TestRealDataProvider:
    def test_receiving_yards_known_baselines(self, fake_db) -> None:
        fake_db.queue.append(_history())
        fake_db.queue.append([
            _fm("P1", "2024_02_P1", kalman_est_receiving_yards=88.0),
            _fm("P1", "2024_03_P1", kalman_est_receiving_yards=91.0),
        ])
        out = rb.real_data_provider(2024, ["WR", "RB", "QB"], ["receiving_yards"])

        p1 = out[out["player_id"] == "P1"].sort_values("week")
        assert list(p1["week"]) == [2, 3]
        assert list(p1["actual_value"]) == [90.0, 100.0]
        # Naive baseline = 2023 mean of [50, 70] = 60 for both weeks.
        assert list(p1["naive_baseline"]) == [60.0, 60.0]
        # Rolling baseline = mean of earlier 2024 weeks: [80] -> 80, then [80, 90] -> 85.
        assert list(p1["rolling_baseline"]) == [80.0, 85.0]
        assert list(p1["kalman_baseline"]) == [88.0, 91.0]
        assert set(p1["stat"]) == {"receiving_yards"}

    def test_missing_prior_season_uses_rolling_as_naive(self, fake_db) -> None:
        fake_db.queue.append(_history())
        fake_db.queue.append([])
        out = rb.real_data_provider(2024, ["RB"], ["rushing_yards"])
        p2 = out[out["player_id"] == "P2"]
        assert len(p2) == 1
        assert p2.iloc[0]["actual_value"] == pytest.approx(60.0)
        assert p2.iloc[0]["naive_baseline"] == pytest.approx(p2.iloc[0]["rolling_baseline"])
        # No feature-matrix row for this game, so the Kalman baseline is missing.
        assert np.isnan(p2.iloc[0]["kalman_baseline"])

    def test_non_matching_position_gives_empty_frame(self, fake_db) -> None:
        fake_db.queue.append(_history())
        out = rb.real_data_provider(2024, ["TE"], ["receiving_yards"])
        assert out.empty

    def test_empty_history_short_circuits(self, fake_db) -> None:
        fake_db.queue.append([])
        assert rb.real_data_provider(2024, ["WR"], ["receiving_yards"]).empty
        assert len(fake_db.queue) == 0  # no feature-matrix query was issued

    def test_stat_with_no_observed_values_is_skipped(self, fake_db) -> None:
        fake_db.queue.append(_history())
        fake_db.queue.append([])
        # Receptions are missing for the P3 QB and the P2 RB, so no row survives for them.
        out = rb.real_data_provider(2024, ["QB", "RB"], ["receptions"])
        assert out.empty

    def test_pass_attempts_resolve_through_gamelog_attribute(self, fake_db) -> None:
        fake_db.queue.append(_history())
        fake_db.queue.append([])
        out = rb.real_data_provider(2024, ["QB"], ["pass_attempts"])
        assert list(out["stat"].unique()) == ["pass_attempts"]
        qb = out.sort_values("week")
        assert list(qb["actual_value"]) == [40.0]  # week 2 only; week 1 is the cohort exclusion
        assert qb.iloc[0]["naive_baseline"] == pytest.approx(30.0)  # 2023 mean of attempts

    def test_multi_source_fumbles_sum_components(self, fake_db) -> None:
        # Every fumble component is filled in here; test_nan_fumble_component_does_not_poison_total
        # covers the NaN case that pandas conversion produces.
        history = _history()
        for row in history:
            for attr in ("rushing_fumbles", "receiving_fumbles", "sack_fumbles"):
                if getattr(row, attr) is None:
                    setattr(row, attr, 0.0)
        history[3].receiving_fumbles = 0.5  # P1 week 2 -> sum should be 0.5
        fake_db.queue.append(history)
        fake_db.queue.append([])
        out = rb.real_data_provider(2024, ["WR", "RB", "QB"], ["fumbles"])
        by_player = {pid: grp for pid, grp in out.groupby("player_id")}
        assert list(by_player["P1"].sort_values("week")["actual_value"]) == [0.5, 2.0]
        assert list(by_player["P2"]["actual_value"]) == [0.0]

    def test_nan_fumble_component_does_not_poison_total(self) -> None:
        from ml.stat_resolution import read_gamelog_stat

        row = {"receiving_fumbles": 1.0, "rushing_fumbles": float("nan"), "sack_fumbles": float("nan")}
        assert read_gamelog_stat(row, "fumbles") == pytest.approx(1.0)


# ── real_projection_provider ──────────────────────────────────────────────────

def _projection(**overrides) -> SimpleNamespace:
    base = {
        "player_id": "P1", "season": 2024, "week": 3, "position": "WR", "stat": "receiving_yards",
        "projection": 95.0, "floor": 70.0, "ceiling": 120.0, "p25": 84.0, "p75": 106.0,
        "pipeline_run_id": "run-1", "posterior_samples": [90.0, 95.0, 100.0],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRealProjectionProvider:
    @pytest.fixture
    def recorded_validation(self, monkeypatch):
        calls: dict = {}
        sentinel_policy = object()

        def fake_validate(df, policy, *, eval_season):
            calls.update(df=df.copy(), policy=policy, eval_season=eval_season)
            return df.assign(validated=True)

        monkeypatch.setattr(rb, "_get_trust_policy", lambda: sentinel_policy)
        monkeypatch.setattr(rb, "validate_projection_trust", fake_validate)
        calls["sentinel"] = sentinel_policy
        return calls

    def test_filters_positions_and_stats_then_validates(self, fake_db, recorded_validation) -> None:
        fake_db.queue.append([
            _projection(),
            _projection(player_id="P2", position="RB", stat="rushing_yards"),   # position not requested
            _projection(player_id="P3", position="WR", stat="targets"),         # stat not requested
        ])
        out = rb.real_projection_provider(2024, ["WR"], ["receiving_yards"])

        assert list(out["player_id"]) == ["P1"]
        assert bool(out.loc[0, "validated"]) is True
        assert recorded_validation["eval_season"] == 2024
        assert recorded_validation["policy"] is recorded_validation["sentinel"]
        row = recorded_validation["df"].iloc[0]
        assert row["projection"] == pytest.approx(95.0)
        assert row["pipeline_run_id"] == "run-1"
        np.testing.assert_array_equal(row["posterior_samples"], [90.0, 95.0, 100.0])

    def test_non_numeric_posterior_samples_become_none(self, fake_db, recorded_validation) -> None:
        fake_db.queue.append([_projection(posterior_samples=["a", "b"])])
        rb.real_projection_provider(2024, ["WR"], ["receiving_yards"])
        assert recorded_validation["df"].iloc[0]["posterior_samples"] is None

    def test_empty_samples_and_missing_max_train_season(self, fake_db, recorded_validation) -> None:
        fake_db.queue.append([_projection(posterior_samples=None)])
        rb.real_projection_provider(2024, ["WR"], ["receiving_yards"])
        frame = recorded_validation["df"]
        assert frame.iloc[0]["posterior_samples"] is None
        assert frame.iloc[0]["max_train_season"] is None

    def test_max_train_season_is_carried_when_present(self, fake_db, recorded_validation) -> None:
        fake_db.queue.append([_projection(max_train_season=2023)])
        rb.real_projection_provider(2024, ["WR"], ["receiving_yards"])
        assert recorded_validation["df"].iloc[0]["max_train_season"] == 2023

    def test_no_matching_projections_skips_validation(self, fake_db, monkeypatch) -> None:
        fake_db.queue.append([_projection(position="TE")])

        def must_not_run(*args, **kwargs):
            raise AssertionError("validation should not run on an empty projection frame")

        monkeypatch.setattr(rb, "validate_projection_trust", must_not_run)
        out = rb.real_projection_provider(2024, ["WR"], ["receiving_yards"])
        assert out.empty


# ── _get_trust_policy ─────────────────────────────────────────────────────────

def test_trust_policy_is_loaded_once_from_settings(monkeypatch) -> None:
    from backend.app.core.config import settings

    loads: list = []
    sentinel = object()

    def fake_load(baseline_path, invalidation_path):
        loads.append((baseline_path, invalidation_path))
        return sentinel

    monkeypatch.setattr(rb, "_TRUST_POLICY", None)
    monkeypatch.setattr(rb, "load_backtest_trust_policy", fake_load)
    first = rb._get_trust_policy()
    second = rb._get_trust_policy()
    assert first is sentinel and second is sentinel
    assert loads == [(settings.baseline_manifest_path, settings.artifact_invalidation_path)]
