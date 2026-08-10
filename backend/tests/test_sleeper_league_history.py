from scripts.import_sleeper_league_history import normalized_pick


def test_normalized_pick_keeps_roster_owner_and_vendor_player_id() -> None:
    row = normalized_pick(
        {"draft_id": "draft-1", "season": "2025", "league_id": "league-1"},
        {"pick_no": 5, "roster_id": 3, "round": 1, "draft_slot": 5,
         "metadata": {"player_id": "7564", "first_name": "Ja'Marr", "last_name": "Chase", "position": "WR", "team": "CIN"}},
        {3: "owner-1"},
    )
    assert row["owner_id"] == "owner-1"
    assert row["player_id"] == "7564"
    assert row["player_name"] == "Ja'Marr Chase"
