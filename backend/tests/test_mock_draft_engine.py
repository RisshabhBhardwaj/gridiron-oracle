"""
Unit tests for ml/mock_draft.py pick engine.
"""

import numpy as np
import pytest

from ml.mock_draft import pick


def test_pick_engine_timing_gate_filters_early_qb():
    rng = np.random.default_rng(42)
    available = [
        {"player_id": "qb1", "player_name": "Patrick Mahomes", "position": "QB", "blended_rank": 1},
        {"player_id": "wr1", "player_name": "Justin Jefferson", "position": "WR", "blended_rank": 2},
        {"player_id": "rb1", "player_name": "Bijan Robinson", "position": "RB", "blended_rank": 3},
    ]
    profile = {
        "first_qb_pick_shrunk": 20.0,
        "first_te_pick_shrunk": 30.0,
        "adp_delta_mean_shrunk": 0.0,
        "adp_delta_sd_shrunk": 5.0,
    }

    # At pick_no = 2 (< 20), QB should be filtered out by timing gate
    chosen = pick(available, roster=[], profile=profile, rng=rng, pick_no=2)
    assert chosen["position"] in ("WR", "RB")


def test_pick_engine_roster_cap():
    rng = np.random.default_rng(42)
    available = [
        {"player_id": "qb2", "player_name": "Josh Allen", "position": "QB", "blended_rank": 1},
        {"player_id": "wr2", "player_name": "CeeDee Lamb", "position": "WR", "blended_rank": 2},
    ]
    # Roster already has 3 QBs (cap is 3 for QB)
    roster = [
        {"player_id": "qba", "position": "QB"},
        {"player_id": "qbb", "position": "QB"},
        {"player_id": "qbc", "position": "QB"},
    ]
    profile = {
        "first_qb_pick_shrunk": 1.0,
        "adp_delta_mean_shrunk": 0.0,
        "adp_delta_sd_shrunk": 5.0,
    }

    chosen = pick(available, roster=roster, profile=profile, rng=rng, pick_no=50)
    assert chosen["position"] == "WR"


def test_pick_engine_never_drops_last_candidate():
    rng = np.random.default_rng(42)
    available = [
        {"player_id": "qb1", "player_name": "Patrick Mahomes", "position": "QB", "blended_rank": 1},
    ]
    profile = {
        "first_qb_pick_shrunk": 50.0,
        "adp_delta_mean_shrunk": 0.0,
        "adp_delta_sd_shrunk": 5.0,
    }
    # Even if pick_no = 1 < 50, never drop the last candidate
    chosen = pick(available, roster=[], profile=profile, rng=rng, pick_no=1)
    assert chosen["player_id"] == "qb1"
