"""Regression guards for pipeline/pbp_pipeline.py's opponent-scheme merge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.pbp_pipeline import _aggregate_pbp, _build_pbp_plays


def _dropback_rows(game_id: str, week: int, posteam: str, defteam: str, n: int, zone: bool) -> list[dict]:
    """n qb-dropback plays for one game, all zone or all man coverage."""
    return [
        {
            "play_type": "pass",
            "game_id": game_id,
            "week": week,
            "posteam": posteam,
            "defteam": defteam,
            "passer_player_id": f"{posteam}_QB",
            "receiver_player_id": None,
            "rusher_player_id": None,
            "pass_attempt": 1,
            "complete_pass": 1,
            "air_yards": 5.0,
            "yards_after_catch": 2.0,
            "xyac_mean_yardage": 2.0,
            "epa": 0.1,
            "qb_epa": 0.1,
            "yardline_100": 50,
            "pass_location": "middle",
            "rush_attempt": 0,
            "qb_dropback": 1,
            "sack": 0,
            "qb_hit": 0,
            "defense_man_zone_type": "ZONE_COVERAGE" if zone else "MAN_COVERAGE",
            "number_of_pass_rushers": 4,
        }
        for _ in range(n)
    ]


def test_opp_zone_pct_uses_the_players_opponent_not_their_own_team():
    """
    def_pressure is keyed by defteam. Merging it on features["team"] (the
    player's own team) instead of their opponent attaches a receiver's own
    defense's coverage rates to them. Build two teams with opposite
    defensive histories (OFF plays almost all zone on defense, DEF1 plays
    almost all man on defense), then have OFF's WR face DEF1's defense in a
    later game — the WR's opp_zone_pct must reflect DEF1's history, not
    OFF's own.
    """
    rows: list[dict] = []
    # Weeks 1-2: OFF plays defense against team X, almost always zone.
    rows += _dropback_rows("2024_01_X_OFF", 1, "X", "OFF", n=10, zone=True)
    rows += _dropback_rows("2024_02_X_OFF", 2, "X", "OFF", n=10, zone=True)
    # Weeks 1-2: DEF1 plays defense against team Y, almost always man.
    rows += _dropback_rows("2024_01_Y_DEF1", 1, "Y", "DEF1", n=10, zone=False)
    rows += _dropback_rows("2024_02_Y_DEF1", 2, "Y", "DEF1", n=10, zone=False)
    # Week 3: OFF (offense) faces DEF1 (defense). OFF's WR gets targeted.
    rows.append({
        "play_type": "pass",
        "game_id": "2024_03_OFF_DEF1",
        "week": 3,
        "posteam": "OFF",
        "defteam": "DEF1",
        "passer_player_id": "OFF_QB",
        "receiver_player_id": "REC1",
        "rusher_player_id": None,
        "pass_attempt": 1,
        "complete_pass": 1,
        "air_yards": 8.0,
        "yards_after_catch": 3.0,
        "xyac_mean_yardage": 3.0,
        "epa": 0.2,
        "qb_epa": 0.2,
        "yardline_100": 40,
        "pass_location": "right",
        "rush_attempt": 0,
        "qb_dropback": 1,
        "sack": 0,
        "qb_hit": 0,
        "defense_man_zone_type": "ZONE_COVERAGE",
        "number_of_pass_rushers": 4,
    })

    pbp = pd.DataFrame(rows)
    features, _ = _aggregate_pbp(pbp, season=2024)

    rec_row = features[(features["player_id"] == "REC1") & (features["game_id"] == "2024_03_OFF_DEF1")]
    assert len(rec_row) == 1
    opp_zone_pct = rec_row.iloc[0]["opp_zone_pct"]

    assert not np.isnan(opp_zone_pct), "opp_zone_pct should be populated from DEF1's prior-week history"
    # DEF1's history is all man coverage -> opp_zone_pct should be ~0, not ~1
    # (OFF's own history as a defense, which is what the bug would attach).
    assert opp_zone_pct < 0.5, (
        f"opp_zone_pct={opp_zone_pct} looks like it came from OFF's own defensive "
        "history (all zone) instead of DEF1's (all man) — the opponent merge key is wrong"
    )


def test_build_pbp_plays_renames_qtr_to_the_pbp_plays_schema_column():
    """
    Raw nflreadpy PBP uses "qtr"; the pbp_plays table (migration
    20260821_0016) declares the column "quarter". A silent mismatch here
    means every play in a season fails the INSERT and — because Postgres
    refuses further commands on an aborted transaction — cascades into
    every later season in the same run looking like an unrelated failure.
    """
    pbp = pd.DataFrame([
        {
            "game_id": "2024_01_X_OFF", "play_id": 1, "week": 1,
            "posteam": "OFF", "defteam": "X", "play_type": "pass",
            "down": 1, "ydstogo": 10, "yardline_100": 75, "qtr": 2,
            "game_seconds_remaining": 1800.0, "score_differential": 0,
        },
    ])
    plays = _build_pbp_plays(pbp, season=2024)
    assert "quarter" in plays.columns
    assert "qtr" not in plays.columns
    assert plays.iloc[0]["quarter"] == 2


def test_build_pbp_plays_drops_admin_rows_without_a_possession_team():
    pbp = pd.DataFrame([
        {"game_id": "2024_01_X_OFF", "play_id": 1, "week": 1, "posteam": "OFF", "play_type": "pass"},
        {"game_id": "2024_01_X_OFF", "play_id": 2, "week": 1, "posteam": None, "play_type": "timeout"},
    ])
    plays = _build_pbp_plays(pbp, season=2024)
    assert len(plays) == 1
    assert plays.iloc[0]["play_id"] == 1
