"""Regression guards for pipeline/team_game_features.py (Phase 4)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pipeline.team_game_features import _build_coach_tendency, build_team_game_frame

_DB_URL = os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle").replace(
    "postgresql+asyncpg://", "postgresql://"
)


def test_coach_tendency_lags_strictly_prior_weeks():
    """
    Week N's prior_coach_pass_rate must be computed only from weeks < N (or
    earlier seasons) — never from week N's own plays. If the trailing window
    ever included the target week, this would leak the outcome of the game
    the tendency feature is meant to help predict.

    Sample sizes here clear _MIN_PLAYS_FOR_CONFIDENCE (50) — below that, the
    rate is nulled as a confidence gate regardless of lag correctness, which
    is exercised separately in test_coach_tendency_below_confidence_floor_is_null.
    """
    plays = pd.DataFrame([
        # Week 1: coach passes on every neutral-script play (60/60).
        *[{"coach": "Andy Reid", "season": 2023, "week": 1, "play_type": "pass"} for _ in range(60)],
        # Week 2: coach runs on every neutral-script play (60/60).
        *[{"coach": "Andy Reid", "season": 2023, "week": 2, "play_type": "run"} for _ in range(60)],
    ])
    tendency = _build_coach_tendency(plays)

    week1 = tendency[(tendency["coach"] == "Andy Reid") & (tendency["week"] == 1)].iloc[0]
    assert pd.isna(week1["prior_coach_pass_rate"]), "no prior games exist yet — must be NaN, not defaulted"

    week2 = tendency[(tendency["coach"] == "Andy Reid") & (tendency["week"] == 2)].iloc[0]
    # Week 2's prior rate reflects ONLY week 1 (all-pass), not week 2 itself (all-run).
    assert week2["prior_coach_pass_rate"] == 1.0
    assert week2["prior_coach_neutral_plays"] == 60


def test_coach_tendency_travels_with_coach_across_teams():
    """
    The whole point of coach-indexed (not team-indexed) tendency: a coach's
    week-3 rate at a NEW team must still reflect their weeks 1-2 history at
    the OLD team, not reset to nothing.
    """
    plays = pd.DataFrame([
        *[{"coach": "Sean Payton", "season": 2023, "week": 1, "play_type": "pass"} for _ in range(40)],
        *[{"coach": "Sean Payton", "season": 2023, "week": 2, "play_type": "pass"} for _ in range(20)],
        # Week 3: different team, but same coach identity in the plays table.
        *[{"coach": "Sean Payton", "season": 2023, "week": 3, "play_type": "run"} for _ in range(1)],
    ])
    tendency = _build_coach_tendency(plays)
    week3 = tendency[(tendency["coach"] == "Sean Payton") & (tendency["week"] == 3)].iloc[0]
    # Weeks 1-2 combined: 60/60 pass, clears the confidence floor.
    assert week3["prior_coach_pass_rate"] == 1.0
    assert week3["prior_coach_neutral_plays"] == 60


def test_coach_tendency_below_confidence_floor_is_null():
    """
    A coach's rate from a single half-game's worth of neutral snaps is
    mostly sample noise — it must be nulled, not fed to the model as if it
    were a settled tendency.
    """
    plays = pd.DataFrame([
        *[{"coach": "Rookie Coach", "season": 2023, "week": 1, "play_type": "pass"} for _ in range(10)],
    ])
    tendency = _build_coach_tendency(plays)
    week2_plays = pd.DataFrame([
        *[{"coach": "Rookie Coach", "season": 2023, "week": 2, "play_type": "run"} for _ in range(10)],
    ])
    full = _build_coach_tendency(pd.concat([plays, week2_plays], ignore_index=True))
    week2 = full[(full["coach"] == "Rookie Coach") & (full["week"] == 2)].iloc[0]
    assert pd.isna(week2["prior_coach_pass_rate"]), "10 prior plays is below the confidence floor"
    assert week2["prior_coach_neutral_plays"] == 10


def test_coach_tendency_empty_input_returns_empty_frame():
    tendency = _build_coach_tendency(pd.DataFrame())
    assert tendency.empty
    assert list(tendency.columns) == [
        "coach", "season", "week", "prior_coach_pass_rate", "prior_coach_neutral_plays",
    ]


def test_build_team_game_frame_does_not_drop_raiders_rows_to_abbreviation_drift():
    """
    team_game_stats uses nflreadpy's current-franchise code ("LV") for every
    season, including 2019-2020 when games.home_team/away_team still say
    "OAK". An un-normalized join silently drops those rows (16 of 3,920 the
    first time this was built) instead of erroring — exactly the kind of
    quiet data loss that's easy to miss.
    """
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_URL)
        conn.close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc})")

    df = build_team_game_frame(_DB_URL, [2019, 2020])
    assert not df.empty
    assert "LV" in set(df["team"]) or "LV" in set(df["opponent"]), (
        "Raiders rows missing entirely — abbreviation join likely broken again"
    )
    assert "OAK" not in set(df["team"]), "team codes must be normalized to current franchise abbreviations"
