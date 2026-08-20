from ml.vor import attach_vor, league_wide_starters, replacement_projections


def test_league_starter_counts_are_eight_team_ppr() -> None:
    assert league_wide_starters("QB") == 8
    assert league_wide_starters("RB") == 16
    assert league_wide_starters("WR") == 16
    assert league_wide_starters("TE") == 8
    assert league_wide_starters("FLEX") == 8


def test_vor_ranks_flex_after_positional_starters() -> None:
    rows = (
        [{"position": "QB", "projection": 20.0 - i, "player_id": f"qb{i}"} for i in range(10)]
        + [{"position": "RB", "projection": 18.0 - i * 0.1, "player_id": f"rb{i}"} for i in range(24)]
        + [{"position": "WR", "projection": 17.0 - i * 0.1, "player_id": f"wr{i}"} for i in range(24)]
        + [{"position": "TE", "projection": 12.0 - i * 0.2, "player_id": f"te{i}"} for i in range(12)]
    )
    reps = replacement_projections(rows)
    ranked = attach_vor(rows)
    qbs = [row for row in ranked if row["position"] == "QB"]
    assert qbs[0]["vor"] > qbs[-1]["vor"]
    assert reps["QB"] == qbs[8]["projection"]
    assert all(row["vor"] is not None for row in ranked)
