"""
ml/drive_path.py

Single-path Markov drive simulator in pure Python / NumPy.
Walks the 5D state-transition table from ml/oof/transitions_by_game_state.csv
to simulate play-by-play drive paths and outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.drive_engine import require_transitions
from ml.markov_simulator import N_DOWN, N_FP_BUCKETS, N_QUARTER, N_SCORE_BUCKETS, N_YTG_BUCKETS

logger = logging.getLogger(__name__)

# Cached transitions dataframe for microsecond lookups
_CACHED_TRANSITIONS: Optional[pd.DataFrame] = None
_CACHED_LOOKUP: Optional[dict[tuple[int, int, int, int, int], dict[str, float]]] = None


def _score_diff_to_bucket(score_diff: int | float) -> int:
    if score_diff <= -9:
        return 0
    elif score_diff <= -1:
        return 1
    elif score_diff == 0:
        return 2
    elif score_diff <= 8:
        return 3
    else:
        return 4


def get_transitions_lookup(csv_path: Path | None = None) -> dict[tuple[int, int, int, int, int], dict[str, float]]:
    global _CACHED_TRANSITIONS, _CACHED_LOOKUP
    if _CACHED_LOOKUP is not None:
        return _CACHED_LOOKUP

    artifact = require_transitions(csv_path)
    df = pd.read_csv(artifact)
    lookup: dict[tuple[int, int, int, int, int], dict[str, float]] = {}

    for _, row in df.iterrows():
        key = (
            int(row["fp_bucket"]),
            int(row["down_idx"]),
            int(row["ytg_bucket"]),
            int(row["score_diff_bucket"]),
            int(row["quarter_idx"]),
        )
        lookup[key] = {
            "mean_gain": float(row["mean_gain"]),
            "gain_std": float(row["gain_std"]),
            "p_turnover": float(row["p_turnover"]),
            "p_penalty_gain": float(row.get("p_penalty_gain", 0.04)),
            "p_penalty_loss": float(row.get("p_penalty_loss", 0.04)),
            "p_pass": float(row.get("p_pass", 0.58)),
            "count": float(row.get("count", 10)),
        }

    _CACHED_TRANSITIONS = df
    _CACHED_LOOKUP = lookup
    return lookup


@dataclass
class PlayStep:
    play_number: int
    down: int
    ytg: int
    field_pos: int
    play_type: str
    yards_gained: float
    is_turnover: bool
    is_first_down: bool
    is_touchdown: bool
    is_safety: bool
    end_field_pos: int


@dataclass
class SimulatedDrive:
    drive_number: int
    possession_team: str
    quarter: int
    start_field_pos: int
    end_field_pos: int
    plays_count: int
    yards_gained: float
    outcome: str
    points_scored: int
    plays: list[PlayStep] = field(default_factory=list)


def simulate_drive_path(
    start_fp: int = 25,
    score_diff: int = 0,
    quarter: int = 1,
    possession_team: str = "TEAM",
    drive_number: int = 1,
    rng: np.random.Generator | None = None,
    csv_path: Path | None = None,
    max_plays: int = 20,
) -> SimulatedDrive:
    """
    Simulate a single drive path starting at start_fp (0=own endzone, 100=opponent endzone).
    Returns step-by-step plays and terminal drive outcome.
    """
    if rng is None:
        rng = np.random.default_rng()

    lookup = get_transitions_lookup(csv_path)

    curr_fp = max(1, min(99, start_fp))
    curr_down = 1
    curr_ytg = 10
    total_yards = 0.0
    plays: list[PlayStep] = []
    outcome = "PUNT"
    points = 0

    for play_idx in range(1, max_plays + 1):
        # 4th down decision logic
        if curr_down == 4:
            # Field Goal attempt if in range (>= 62 yardline ≈ 55yd FG)
            if curr_fp >= 62:
                fg_dist = 100 - curr_fp + 17  # 17yd snap+endzone buffer
                # Empirical FG make curve: ~85% at 30yd, ~65% at 50yd
                fg_prob = max(0.20, min(0.98, 1.0 - (fg_dist - 20) * 0.015))
                is_made = bool(rng.uniform(0, 1) < fg_prob)
                if is_made:
                    outcome = "FIELD_GOAL"
                    points = 3
                else:
                    outcome = "MISSED_FIELD_GOAL"
                    points = 0
                break

            # Go for it on 4th down in desperation (4th quarter trailing with short yardage)
            go_for_it = (quarter == 4 and score_diff < 0 and curr_ytg <= 2) or (curr_fp >= 85 and curr_ytg <= 1)
            if not go_for_it:
                outcome = "PUNT"
                points = 0
                break

        # Map state to buckets
        # fp_bucket: in markov_simulator, yardline_100 = 100 - curr_fp
        yardline_100 = max(0, min(100, 100 - curr_fp))
        fp_bucket = int(np.clip(yardline_100 // 10, 0, N_FP_BUCKETS - 1))
        down_idx = int(np.clip(curr_down - 1, 0, N_DOWN - 1))
        ytg_bucket = int(np.clip((curr_ytg - 1) // 5, 0, N_YTG_BUCKETS - 1))
        sd_bucket = _score_diff_to_bucket(score_diff)
        quarter_idx = int(np.clip(quarter - 1, 0, N_QUARTER - 1))

        state_key = (fp_bucket, down_idx, ytg_bucket, sd_bucket, quarter_idx)
        cell = lookup.get(state_key)

        if cell is None:
            # Fallback to neutral game state
            cell = lookup.get((fp_bucket, down_idx, ytg_bucket, 2, 0), {
                "mean_gain": 4.5,
                "gain_std": 6.0,
                "p_turnover": 0.025,
                "p_pass": 0.55,
            })

        # Draw play call
        is_pass = bool(rng.uniform(0, 1) < cell.get("p_pass", 0.55))
        play_type = "pass" if is_pass else "run"

        # Turnover check
        p_to = cell.get("p_turnover", 0.025)
        if rng.uniform(0, 1) < p_to:
            plays.append(PlayStep(
                play_number=play_idx,
                down=curr_down,
                ytg=curr_ytg,
                field_pos=curr_fp,
                play_type=play_type,
                yards_gained=0.0,
                is_turnover=True,
                is_first_down=False,
                is_touchdown=False,
                is_safety=False,
                end_field_pos=curr_fp,
            ))
            outcome = "TURNOVER"
            points = 0
            break

        # Yardage draw
        mean_g = cell.get("mean_gain", 4.5)
        std_g = max(0.5, cell.get("gain_std", 6.0))
        raw_gain = float(rng.normal(mean_g, std_g))
        # Cap gain between -10 and remaining distance to goal
        gain = max(-10.0, min(float(100 - curr_fp), round(raw_gain, 1)))

        new_fp = curr_fp + gain
        total_yards += gain

        # Touchdown check
        if new_fp >= 100:
            plays.append(PlayStep(
                play_number=play_idx,
                down=curr_down,
                ytg=curr_ytg,
                field_pos=curr_fp,
                play_type=play_type,
                yards_gained=gain,
                is_turnover=False,
                is_first_down=True,
                is_touchdown=True,
                is_safety=False,
                end_field_pos=100,
            ))
            curr_fp = 100
            outcome = "TOUCHDOWN"
            points = 7
            break

        # Safety check
        if new_fp <= 0:
            plays.append(PlayStep(
                play_number=play_idx,
                down=curr_down,
                ytg=curr_ytg,
                field_pos=curr_fp,
                play_type=play_type,
                yards_gained=gain,
                is_turnover=False,
                is_first_down=False,
                is_touchdown=False,
                is_safety=True,
                end_field_pos=0,
            ))
            curr_fp = 0
            outcome = "SAFETY"
            points = 0
            break

        # First down conversion check
        is_first_down = gain >= curr_ytg
        plays.append(PlayStep(
            play_number=play_idx,
            down=curr_down,
            ytg=curr_ytg,
            field_pos=curr_fp,
            play_type=play_type,
            yards_gained=gain,
            is_turnover=False,
            is_first_down=is_first_down,
            is_touchdown=False,
            is_safety=False,
            end_field_pos=int(new_fp),
        ))

        curr_fp = int(new_fp)
        if is_first_down:
            curr_down = 1
            curr_ytg = min(10, 100 - curr_fp)
        else:
            if curr_down == 4:
                outcome = "TURNOVER_ON_DOWNS"
                points = 0
                break
            else:
                curr_down += 1
                curr_ytg = max(1, int(curr_ytg - gain))

    return SimulatedDrive(
        drive_number=drive_number,
        possession_team=possession_team,
        quarter=quarter,
        start_field_pos=start_fp,
        end_field_pos=curr_fp,
        plays_count=len(plays),
        yards_gained=round(total_yards, 1),
        outcome=outcome,
        points_scored=points,
        plays=plays,
    )
