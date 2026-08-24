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


def test_drive_sim_narration_reconciles_to_the_fitted_anchor():
    """
    The whole point of "anchor, then narrate": the drive chain must never be
    allowed to compute its own score. Before reconciliation the free-running
    chain produced 6-36 narrations under a 23.5-19.8 forecast, with a declared
    winner that contradicted the win probability shown beside it.

    Points quantise to 7s and 3s, so the narrated total can sit a couple of
    points off a fractional forecast -- but never further, and never on the
    wrong side of the margin.
    """
    mock_rows = [
        {
            "game_id": "2026_05_DET_ARI", "team": "ARI", "opponent": "DET",
            "season": 2026, "week": 5, "is_home": True,
            "points": 19.77, "yards": 324.0, "pass_rate": 0.55,
            "win_probability": 0.39, "model_run_id": "weekly_auto_2026",
        },
        {
            "game_id": "2026_05_DET_ARI", "team": "DET", "opponent": "ARI",
            "season": 2026, "week": 5, "is_home": False,
            "points": 23.46, "yards": 351.0, "pass_rate": 0.53,
            "win_probability": 0.61, "model_run_id": "weekly_auto_2026",
        },
    ]

    for seed in (1, 42, 12345):
        with patch("psycopg2.connect") as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = mock_rows
            mock_conn.return_value.cursor.return_value.__enter__.return_value = mock_cursor

            resp = client.get(f"/games/2026_05_DET_ARI/drive-sim?seed={seed}")
            assert resp.status_code == 200
            data = resp.json()

            home_pts = data["summary"]["simulated_home_points"]
            away_pts = data["summary"]["simulated_away_points"]
            assert abs(home_pts - 19.77) <= 3, f"seed={seed} home {home_pts}"
            assert abs(away_pts - 23.46) <= 3, f"seed={seed} away {away_pts}"

            # The narrated winner must agree with the fitted margin, and with
            # the win probability rendered on the same screen.
            assert data["summary"]["simulated_winner"] == "DET", f"seed={seed}"

            # Per-drive points must sum to the headline, or the timeline and
            # the scoreboard tell different stories.
            drives = data["drives"]
            assert sum(d["points_scored"] for d in drives if d["possession_team"] == "ARI") == home_pts
            assert sum(d["points_scored"] for d in drives if d["possession_team"] == "DET") == away_pts

            # Running score must be monotone and end on the final score.
            assert drives[-1]["home_score_after"] == home_pts
            assert drives[-1]["away_score_after"] == away_pts
