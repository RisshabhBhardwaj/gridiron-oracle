"""
Unit tests for sleeper draft profile generation, ADP delta, and shrinkage.
"""

from scripts.generate_sleeper_draft_profiles import build_profiles, _shrink


def test_profiles_use_actual_overall_picks_and_owner_names() -> None:
    profiles = build_profiles([
        {"draft_id": "a", "owner_id": "me", "pick_no": 8, "position": "RB", "adp_delta": -2.0},
        {"draft_id": "a", "owner_id": "me", "pick_no": 25, "position": "QB", "adp_delta": -5.0},
        {"draft_id": "b", "owner_id": "me", "pick_no": 9, "position": "WR", "adp_delta": 1.0},
    ], {"me": "Risshabh734"})["profiles"]

    assert profiles[0]["display_name"] == "Risshabh734"
    assert profiles[0]["early_pick_positions"] == {"RB": 1, "WR": 1}
    assert profiles[0]["first_qb_pick_average"] == 25.0
    assert profiles[0]["n_adp_matched"] == 3
    assert profiles[0]["adp_delta_mean"] == -2.0
    assert "adp_delta_mean_shrunk" in profiles[0]
    assert "first_qb_pick_shrunk" in profiles[0]


def test_shrink_calculation() -> None:
    # weight = 30 / (30 + 30) = 0.5 -> 0.5 * 10 + 0.5 * 0 = 5.0
    shrunk = _shrink(10.0, 0.0, n=30, k=30.0)
    assert shrunk == 5.0

    # sample None -> returns league val
    assert _shrink(None, 48.0, n=0, k=30.0) == 48.0
