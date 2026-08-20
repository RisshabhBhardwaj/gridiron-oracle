from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.app.services.projection import (
    _interval_method,
    _interval_value,
    load_approved_pipeline_run_ids,
)


def test_stored_conformal_method_exposes_floor() -> None:
    row = {
        "interval_method": "causal_oof_conformal_90",
        "floor": 12.0,
        "ceiling": 40.0,
        "fantasy_floor": -3.0,
        "position": "WR",
        "posterior_samples": None,
    }
    assert _interval_method(row) == "causal_oof_conformal_90"
    assert _interval_value(row, "floor") == 12.0
    assert _interval_value(row, "ceiling") == 40.0
    assert _interval_value(row, "fantasy_floor") == 0.0


def test_unavailable_method_hides_legacy_bands() -> None:
    row = {"interval_method": "unavailable", "floor": 1.0, "ceiling": 99.0, "posterior_samples": None}
    assert _interval_method(row) == "unavailable"
    assert _interval_value(row, "floor") is None


def test_empty_approved_runs_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "baseline.json"
    manifest.write_text('{"projection_policy": {"approved_pipeline_run_ids": []}}')
    with pytest.raises(HTTPException) as exc:
        load_approved_pipeline_run_ids(manifest)
    assert exc.value.status_code == 503


def test_season_projections_db_error_is_fail_closed() -> None:
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService("postgresql://unused")
    with pytest.raises(HTTPException) as exc:
        svc.get_season_projections(season=2025, start_week=1)
    assert exc.value.status_code == 503


def test_season_projections_rank_by_playing_time(monkeypatch) -> None:
    from backend.app.services import projection as projection_mod
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService("postgresql://unused")
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run"}))

    def features(_season, _start_week, _positions):
        rows = []
        for i in range(24):
            rows.append({
                "player_id": f"wr{i}",
                "player_name": f"WR{i}",
                "position": "WR",
                "team": "MIN",
                "prior_snap_share": 0.75,
                "prior_games": 16,
                "seas_avg_fantasy_ppr": 14.0,
                "seas_avg_passing_yards": 0.0,
                "seas_avg_rushing_yards": 5.0,
                "seas_avg_receiving_yards": 70.0,
                "depth_rank": 1.0,
            })
        rows.append({
            "player_id": "cold-qb",
            "player_name": "Backup",
            "position": "QB",
            "team": "KC",
            "prior_snap_share": 0.0,
            "prior_games": 0,
            "prior_active_games": 0,
            "seas_avg_fantasy_ppr": 20.0,
            "seas_avg_passing_yards": 250.0,
            "seas_avg_rushing_yards": 0.0,
            "seas_avg_receiving_yards": 0.0,
            "depth_rank": 2.0,
        })
        rows.append({
            "player_id": "starter-qb",
            "player_name": "Starter",
            "position": "QB",
            "team": "KC",
            "prior_snap_share": 0.98,
            "prior_games": 17,
            "seas_avg_fantasy_ppr": 22.0,
            "seas_avg_passing_yards": 280.0,
            "seas_avg_rushing_yards": 20.0,
            "seas_avg_receiving_yards": 0.0,
            "depth_rank": 1.0,
        })
        return rows

    monkeypatch.setattr(svc, "_load_season_feature_rows", features)
    monkeypatch.setattr(svc, "_load_weekly_rates", lambda *_args, **_kwargs: {})
    results = svc.get_season_projections(season=2025, start_week=1)
    assert results
    assert all(row.get("interval_method") == "playing_time_enbpi" for row in results)
    assert all("degraded" in row for row in results)
    top24 = [row["player_id"] for row in results if int(row["ros_rank"]) <= 24]
    assert "cold-qb" not in top24
    assert "starter-qb" in [row["player_id"] for row in results]
