"""
Unit tests for single-path drive simulation in ml/drive_path.py.
"""

import numpy as np
import pytest

from ml.drive_path import simulate_drive_path


def test_simulate_drive_path_basic():
    rng = np.random.default_rng(42)
    drive = simulate_drive_path(start_fp=25, score_diff=0, quarter=1, possession_team="MIN", rng=rng)

    assert drive.possession_team == "MIN"
    assert drive.quarter == 1
    assert drive.start_field_pos == 25
    assert drive.plays_count >= 1
    assert drive.outcome in (
        "TOUCHDOWN",
        "FIELD_GOAL",
        "MISSED_FIELD_GOAL",
        "PUNT",
        "TURNOVER",
        "TURNOVER_ON_DOWNS",
        "SAFETY",
    )
    assert len(drive.plays) == drive.plays_count


def test_simulate_drive_touchdown_reach():
    # Starting at 95 yardline (5 yards to goal)
    rng = np.random.default_rng(10)
    drive = simulate_drive_path(start_fp=95, score_diff=0, quarter=1, possession_team="KC", rng=rng)
    assert drive.end_field_pos >= 95
