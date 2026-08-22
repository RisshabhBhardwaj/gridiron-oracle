"""
Tests for the season_simulations read path wired into
ProjectionService.get_season_projections (Phase 7): prefer a materialized
SeasonSimulator run when an approved one exists for (season, start_week),
otherwise fall back to the flat-rate weekly-stack-rate path unchanged.
"""
from __future__ import annotations

from backend.app.services import projection as projection_mod
from backend.app.services.projection import ProjectionService


def _svc(monkeypatch) -> ProjectionService:
    svc = ProjectionService("postgresql://unused")
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run1"}))
    monkeypatch.setattr(svc, "_warn_if_depth_chart_stale", lambda season: None)
    return svc


class TestPrefersSimulatedRowsWhenPresent:
    def test_skips_flat_rate_loaders_when_simulation_rows_exist(self, monkeypatch) -> None:
        svc = _svc(monkeypatch)
        simulated = [{
            "player_id": "wr1", "player_name": "WR One", "position": "WR", "team": "MIN",
            "prior_games": 10, "prior_active_games": 9, "depth_rank": 1.0, "p_active": 0.9,
            "degraded": False, "interval_method": "season_simulator_mc",
            "fantasy_ppr": {"mean": 14.0, "p10": 8.0, "p50": 14.0, "p90": 20.0},
            "mean": 14.0,
        }]
        monkeypatch.setattr(svc, "_load_season_simulation_rows", lambda *a, **k: simulated)

        def boom(*a, **k):
            raise AssertionError("flat-rate loader should not run when simulation rows exist")

        monkeypatch.setattr(svc, "_load_season_feature_rows", boom)
        monkeypatch.setattr(svc, "_load_weekly_rates", boom)

        results = svc.get_season_projections(season=2026, start_week=5, stats=["fantasy_ppr"])
        assert len(results) == 1
        assert results[0]["player_id"] == "wr1"
        assert results[0]["interval_method"] == "season_simulator_mc"
        assert results[0]["ros_rank"] == 1

    def test_falls_back_to_flat_rate_when_no_simulation_rows(self, monkeypatch) -> None:
        svc = _svc(monkeypatch)
        monkeypatch.setattr(svc, "_load_season_simulation_rows", lambda *a, **k: [])
        monkeypatch.setattr(svc, "_load_season_feature_rows", lambda *a, **k: [{
            "player_id": "wr1", "player_name": "WR One", "position": "WR", "team": "MIN",
            "prior_games": 10, "seas_avg_fantasy_ppr": 14.0, "seas_avg_receiving_yards": 70.0,
        }])
        monkeypatch.setattr(svc, "_load_weekly_rates", lambda *a, **k: {})

        results = svc.get_season_projections(season=2026, start_week=5, stats=["receiving_yards"])
        assert len(results) == 1
        assert results[0]["interval_method"] == "playing_time_gaussian_mc"


class TestSimulationRowsApplyColdStartGuard:
    def test_cold_start_qb_degraded_even_from_simulated_rows(self, monkeypatch) -> None:
        svc = _svc(monkeypatch)
        rows = [
            {
                "player_id": f"wr{i}", "player_name": f"WR{i}", "position": "WR", "team": "MIN",
                "prior_games": 10, "prior_active_games": 9, "depth_rank": 1.0, "p_active": 0.9,
                "degraded": False, "interval_method": "season_simulator_mc",
                "fantasy_ppr": {"mean": 14.0, "p10": 8.0, "p50": 14.0, "p90": 20.0},
                "mean": 14.0,
            }
            for i in range(24)
        ]
        rows.append({
            "player_id": "cold-qb", "player_name": "Backup", "position": "QB", "team": "KC",
            "prior_games": 0, "prior_active_games": 0, "depth_rank": 2.0, "p_active": 0.02,
            "degraded": False, "interval_method": "season_simulator_mc",
            "fantasy_ppr": {"mean": 20.0, "p10": 5.0, "p50": 20.0, "p90": 40.0},
            "mean": 20.0,
        })
        monkeypatch.setattr(svc, "_load_season_simulation_rows", lambda *a, **k: rows)
        monkeypatch.setattr(svc, "_load_season_feature_rows", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not reach flat-rate path")))

        results = svc.get_season_projections(season=2026, start_week=5, stats=["fantasy_ppr"])
        cold = next(r for r in results if r["player_id"] == "cold-qb")
        assert cold["degraded"] is True
        assert cold["mean"] == 0.0


class TestLoadSeasonSimulationRowsPivot:
    def test_pivots_stat_rows_into_per_player_shape(self, monkeypatch) -> None:
        svc = ProjectionService("postgresql://unused")

        class _FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params):
                self.params = params

            def fetchall(self):
                return [
                    {
                        "player_id": "wr1", "player_name": "WR One", "position": "WR", "team": "MIN",
                        "stat": "receiving_yards", "mean": 900.0, "p10": 700.0, "p50": 900.0, "p90": 1100.0,
                        "p_active": 0.9, "prior_games": 10, "prior_active_games": 9, "depth_rank": 1.0,
                        "degraded": False, "interval_method": "season_simulator_mc",
                    },
                    {
                        "player_id": "wr1", "player_name": "WR One", "position": "WR", "team": "MIN",
                        "stat": "fantasy_ppr", "mean": 180.0, "p10": 140.0, "p50": 180.0, "p90": 220.0,
                        "p_active": 0.9, "prior_games": 10, "prior_active_games": 9, "depth_rank": 1.0,
                        "degraded": False, "interval_method": "season_simulator_mc",
                    },
                ]

        class _FakeConn:
            def cursor(self, cursor_factory=None):
                return _FakeCursor()

            def close(self):
                pass

        monkeypatch.setattr("psycopg2.connect", lambda *a, **k: _FakeConn())

        out = svc._load_season_simulation_rows(
            2026, 5, ["WR"], ["receiving_yards", "fantasy_ppr"], frozenset({"run1"})
        )
        assert len(out) == 1
        item = out[0]
        assert item["player_id"] == "wr1"
        assert item["receiving_yards"]["mean"] == 900.0
        assert item["fantasy_ppr"]["mean"] == 180.0
        assert item["mean"] == 180.0  # derived from fantasy_ppr, same convention as the flat-rate path

    def test_returns_empty_list_when_no_rows(self, monkeypatch) -> None:
        svc = ProjectionService("postgresql://unused")

        class _FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params):
                pass

            def fetchall(self):
                return []

        class _FakeConn:
            def cursor(self, cursor_factory=None):
                return _FakeCursor()

            def close(self):
                pass

        monkeypatch.setattr("psycopg2.connect", lambda *a, **k: _FakeConn())
        out = svc._load_season_simulation_rows(2026, 5, ["WR"], ["fantasy_ppr"], frozenset({"run1"}))
        assert out == []
