"""
Unit tests for GET /games/{game_id}/drive-sim anchor-then-narrate endpoint.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_game_drive_sim_endpoint():
    mock_rows = [
        {
            "game_id": "2026_01_KC_MIN",
            "team": "MIN",
            "opponent": "KC",
            "season": 2026,
            "week": 1,
            "is_home": True,
            "points": 24.5,
            "yards": 350.0,
            "pass_rate": 0.58,
            "win_probability": 0.52,
            "model_run_id": "weekly_auto_2026",
        },
        {
            "game_id": "2026_01_KC_MIN",
            "team": "KC",
            "opponent": "MIN",
            "season": 2026,
            "week": 1,
            "is_home": False,
            "points": 23.8,
            "yards": 340.0,
            "pass_rate": 0.60,
            "win_probability": 0.48,
            "model_run_id": "weekly_auto_2026",
        },
    ]

    with patch("psycopg2.connect") as mock_conn:
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = mock_rows
        mock_conn.return_value.cursor.return_value.__enter__.return_value = mock_cursor

        resp = client.get("/games/2026_01_KC_MIN/drive-sim?seed=42&n_drives_per_team=8")
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_id"] == "2026_01_KC_MIN"
        assert data["home_team"] == "MIN"
        assert data["away_team"] == "KC"
        assert data["anchor"]["home_points"] == 24.5
        assert len(data["drives"]) == 16
        assert data["summary"]["total_drives"] == 16
        assert data["summary"]["simulated_winner"] in ("MIN", "KC", "TIE")
