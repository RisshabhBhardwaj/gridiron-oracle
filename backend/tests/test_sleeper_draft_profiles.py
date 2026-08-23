from scripts.generate_sleeper_draft_profiles import build_profiles


def test_profiles_use_actual_overall_picks_and_owner_names() -> None:
    profiles = build_profiles([
        {"draft_id": "a", "owner_id": "me", "pick_no": 8, "position": "RB"},
        {"draft_id": "a", "owner_id": "me", "pick_no": 25, "position": "QB"},
        {"draft_id": "b", "owner_id": "me", "pick_no": 9, "position": "WR"},
    ], {"me": "Risshabh734"})["profiles"]
    assert profiles[0]["display_name"] == "Risshabh734"
    assert profiles[0]["early_pick_positions"] == {"RB": 1, "WR": 1}
    assert profiles[0]["first_qb_pick_average"] == 25.0
