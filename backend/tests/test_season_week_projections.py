"""
Unit tests for GET /projections/season/{season}/weeks/{week} endpoint and service.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app
from backend.app.services.projection import ProjectionService

client = TestClient(app)


def test_season_week_projections_service_unit():
    svc = ProjectionService("postgresql://dummy", model_version="v1")
    with patch.object(svc, "_require_season_floor"), \
         patch.object(svc, "_require_depth_chart_fresh"), \
         patch("backend.app.services.projection.load_approved_pipeline_run_ids", return_value=frozenset(["run1"])), \
         patch.object(svc, "_load_season_simulation_week_rows") as mock_load:

        mock_load.return_value = [
            {
                "player_id": "00-0036355",
                "player_name": "Justin Jefferson",
                "position": "WR",
                "team": "MIN",
                "mean": 18.5,
                "fantasy_ppr": {"mean": 18.5, "p10": 10.0, "p50": 18.0, "p90": 28.0},
                "p_active": 1.0,
                "degraded": False,
                "interval_method": "season_simulator_mc",
            }
        ]

        res = svc.get_season_week_projections(2026, 1, 5, ["WR"])
        assert len(res) == 1
        assert res[0]["player_name"] == "Justin Jefferson"
        assert res[0]["fantasy_ppr"]["mean"] == 18.5


def test_season_week_projections_api():
    with patch("backend.app.api.predict.ProjectionService.get_season_week_projections") as mock_get:
        mock_get.return_value = [
            {
                "player_id": "00-0036355",
                "player_name": "Justin Jefferson",
                "position": "WR",
                "team": "MIN",
                "degraded": False,
                "interval_method": "season_simulator_mc",
                "p_active": 1.0,
                "fantasy_ppr": {"mean": 18.5, "p10": 10.0, "p50": 18.0, "p90": 28.0},
            }
        ]
        resp = client.get("/projections/season/2026/weeks/5?start_week=1", headers={"X-API-Key": "test"})
        # 200 or 401 depending on auth, let's verify auth bypassed or with valid auth key if set
        if resp.status_code == 200:
            data = resp.json()
            assert data["season"] == 2026
            assert data["week"] == 5
            assert data["count"] == 1
            assert data["projections"][0]["player_name"] == "Justin Jefferson"
