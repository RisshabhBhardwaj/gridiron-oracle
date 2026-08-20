from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.playing_time import (
    PlayingTimeModel,
    as_of_depth_rank,
    assert_cold_start_qb_not_in_top24,
    attach_playing_time,
    availability_calibration_table,
    build_playing_time_frame,
    rank_rest_of_season,
    simulate_season_paths,
)


def test_as_of_depth_rank_ignores_post_kickoff_rows() -> None:
    kickoff = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
    charts = pd.DataFrame([
        {"player_id": "qb1", "published_at": datetime(2025, 9, 6, tzinfo=timezone.utc), "depth_rank": 2.0},
        {"player_id": "qb1", "published_at": datetime(2025, 9, 8, tzinfo=timezone.utc), "depth_rank": 1.0},
    ])
    assert as_of_depth_rank(charts, player_id="qb1", kickoff_at=kickoff) == 2.0


def test_lagged_xfp_is_shift_not_same_week() -> None:
    history = pd.DataFrame([
        {"player_id": "w1", "season": 2024, "week": 1, "offense_pct": 80.0, "rec_fantasy_points_exp": 10.0},
        {"player_id": "w1", "season": 2024, "week": 2, "offense_pct": 80.0, "rec_fantasy_points_exp": 20.0},
    ])
    frame = build_playing_time_frame(history)
    assert pd.isna(frame.iloc[0]["lagged_rec_xfp"])
    assert frame.iloc[1]["lagged_rec_xfp"] == 10.0


def test_cold_start_qb_is_not_in_ros_top24() -> None:
    rows = []
    for i in range(30):
        rows.append({
            "player_id": f"wr{i}",
            "position": "WR",
            "prior_snap_share": 0.70,
            "prior_games": 14,
            "mean": 180.0 - i,
        })
    rows.append({
        "player_id": "cold-qb",
        "position": "QB",
        "prior_snap_share": 0.0,
        "prior_games": 0,
        "prior_active_games": 0,
        "depth_rank": 2.0,
        "mean": 400.0,
    })
    attached = attach_playing_time(rows)
    for row in attached:
        if row["player_id"] == "cold-qb":
            assert row["p_active"] <= 0.05
            row["mean"] = 12.0 * row["p_active"] * 17
    ranked = rank_rest_of_season(attached)
    assert_cold_start_qb_not_in_top24(ranked)


def test_simulate_season_paths_scales_with_availability() -> None:
    high = simulate_season_paths(15.0, 0.95, 17, residual_scale=0.01, n_sims=200, rng=1)
    low = simulate_season_paths(15.0, 0.05, 17, residual_scale=0.01, n_sims=200, rng=1)
    assert high["mean"] > low["mean"] * 5


def test_fitted_availability_and_role_are_bounded() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        prior = 0.15 + 0.7 * ((i % 10) / 9.0)
        active = 0 if i % 6 == 0 else 1
        snap = 0.0 if active == 0 else float(np.clip(0.35 * prior + 0.04 * rng.normal(), 0.05, 0.95))
        rows.append({
            "player_id": f"p{i % 16}",
            "season": 2023 + (i // 60),
            "week": 1 + (i % 17),
            "offense_pct": snap * 100.0,
            "rec_fantasy_points_exp": 4.0 + prior * 12,
        })
    history = pd.DataFrame(rows)
    model = PlayingTimeModel().fit(history)
    frame = build_playing_time_frame(history).dropna(subset=["prior_snap_share"])
    predicted_rows = model.predict(frame.to_dict("records"))
    p_active = np.array([row["p_active"] for row in predicted_rows])
    role = np.array([row["role_share"] for row in predicted_rows])
    assert p_active.min() >= 0.0 and p_active.max() <= 1.0
    assert role.min() >= 0.0 and role.max() <= 1.0
    table = availability_calibration_table(frame["active"].to_numpy(), p_active)
    assert not table.empty
    mae = float(np.mean(np.abs(frame["snap_share"].to_numpy(dtype=float) - role)))
    assert mae <= 0.25
