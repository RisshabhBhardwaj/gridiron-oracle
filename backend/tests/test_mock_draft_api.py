"""
Unit tests for mock draft API endpoints (/mock-draft/profiles, /mock-draft/pick).
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_mock_draft_profiles_api():
    fake_payload = {
        "profiles": [
            {
                "owner_id": "owner_1",
                "display_name": "Manager Alpha",
                "draft_slot": 1,
                "drafts": 4,
                "picks": 68,
                "first_qb_pick_shrunk": 42.0,
                "first_te_pick_shrunk": 60.0,
                "adp_delta_mean_shrunk": -2.0,
                "adp_delta_sd_shrunk": 7.5,
                "early_rb_share_shrunk": 0.5,
                "early_wr_share_shrunk": 0.4,
            }
        ],
        "method": {"note": "Test note"},
    }
    with patch("backend.app.api.mock_draft.load_profiles", return_value=fake_payload):
        resp = client.get("/mock-draft/profiles?season=2026")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["profiles"][0]["owner_id"] == "owner_1"
        assert data["profiles"][0]["first_qb_pick_shrunk"] == 42.0


def test_mock_draft_pick_api():
    mock_pool = [
        {"player_id": "p1", "player_name": "Player 1", "position": "WR", "adp": 1.0, "blended_rank": 1},
        {"player_id": "p2", "player_name": "Player 2", "position": "RB", "adp": 2.0, "blended_rank": 2},
        {"player_id": "p3", "player_name": "Player 3", "position": "WR", "adp": 3.0, "blended_rank": 3},
    ]
    with patch("backend.app.api.mock_draft.build_mock_draft_pool", return_value=mock_pool), \
         patch("backend.app.api.mock_draft.load_profiles", return_value={"profiles": []}):

        req_body = {
            "season": 2026,
            "draft_order": ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8"],
            "picks_so_far": [],
            "user_slot": 3,  # User is slot 3, so picks 1 and 2 (o1 and o2) will simulate
            "seed": 12345,
        }
        resp = client.post("/mock-draft/pick", json=req_body)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["new_picks"]) == 2
        assert data["new_picks"][0]["slot"] == 1
        assert data["new_picks"][1]["slot"] == 2
        assert data["is_user_turn"] is True
        assert data["next_turn_slot"] == 3
