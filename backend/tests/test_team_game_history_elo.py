"""
Why ETL step 6 (materialize_team_game_predictions) failed on the box, and why
restoring history fixes it without imputation.

Measured 2026-11-14 on the box: team_off_elo / team_def_elo / opp_*_elo are 0%
present for 2019-2023 training rows and 100% present for 2024-2025, because
``build_team_game_frame`` reads Elo from ``feature_matrix`` and the box cut
that table to 2024 onward. The Ridge fit then raised ``Input X contains NaN``.

The median fill in ``materialize_team_game_predictions._fit`` does not rescue
those rows: psycopg2 returns NUMERIC columns as ``Decimal``, pandas holds them
as ``object``, and ``DataFrame.median(numeric_only=True)`` skips object
columns, so no fill value exists for Elo. That is accidental, and correct:
filling a rating with a median would hide missing history.

These tests run the real ``build_team_game_frame`` and ``_fit`` over scripted
loader output (no database), once with the cut history and once with the
backfilled history. They pin the behavior: cut history fails loudly, full
history passes, and nothing is imputed.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import pipeline.team_game_features as tgf
from ml.team_game_model import FEATURE_COLS, _prepare_x
from scripts import materialize_team_game_predictions as mtgp

SEASONS = list(range(2019, 2026))
WEEKS = (1, 2, 3)
TEAMS = ("KC", "BAL")


def _games() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            rows.append({
                "game_id": f"{season}_{week:02d}_KC_BAL", "season": season, "week": week,
                "home_team": "KC", "away_team": "BAL",
                "home_score": 20 + week + (season % 5), "away_score": 17 + (season % 7),
                "home_rest": 7, "away_rest": 6 + (week % 2),
                "spread_line": -3.0, "total_line": 46.5, "roof": "outdoors",
                "surface": "grass", "home_coach": "A. Reid", "away_coach": "J. Harbaugh",
                "kickoff_at": datetime(season, 9, 7 + week),
            })
    return pd.DataFrame(rows)


def _team_game_stats() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            for i, team in enumerate(TEAMS):
                rows.append({
                    "team": team, "season": season, "week": week,
                    "pass_attempts": 30 + i + week, "rush_attempts": 25 - i,
                    "total_plays": 55 + week, "total_yards": 330.0 + 10 * week + 5 * i + season % 9,
                })
    return pd.DataFrame(rows)


def _elo(seasons) -> pd.DataFrame:
    """Elo as the loader returns it from a NUMERIC column: Decimal objects."""
    rows = []
    for season in seasons:
        for week in WEEKS:
            for i, team in enumerate(TEAMS):
                rows.append({
                    "season": season, "week": week, "team": team,
                    "team_off_elo": Decimal(1500 + 7 * i + week + season % 13),
                    "team_def_elo": Decimal(1490 - 5 * i + week + season % 11),
                })
    return pd.DataFrame(rows, columns=["season", "week", "team", "team_off_elo", "team_def_elo"])


def _coach_plays() -> pd.DataFrame:
    """Enough neutral-script plays per coach-week to clear the confidence floor."""
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            for coach in ("A. Reid", "J. Harbaugh"):
                rows += [{"season": season, "week": week, "play_type": "pass", "coach": coach}] * 35
                rows += [{"season": season, "week": week, "play_type": "run", "coach": coach}] * 30
    return pd.DataFrame(rows)


class _Conn:
    def close(self) -> None:
        pass


@pytest.fixture
def frame_for(monkeypatch):
    """Return build_team_game_frame(elo_seasons) with Elo present only for those seasons."""

    def build(elo_seasons) -> pd.DataFrame:
        monkeypatch.setattr(tgf.psycopg2, "connect", lambda *_a, **_k: _Conn())
        monkeypatch.setattr(tgf, "_load_games", lambda *_a: _games())
        monkeypatch.setattr(tgf, "_load_team_game_stats", lambda *_a: _team_game_stats())
        monkeypatch.setattr(tgf, "_load_team_elo", lambda *_a: _elo(elo_seasons))
        monkeypatch.setattr(tgf, "_load_coach_plays", lambda *_a: _coach_plays())
        monkeypatch.setattr(tgf, "_load_home_team_roof", lambda *_a: {"KC": "outdoors"})
        return tgf.build_team_game_frame("postgresql://unused", SEASONS)

    return build


def test_cut_history_reproduces_the_box_failure_and_is_not_imputed(frame_for):
    frame = frame_for([2024, 2025])
    elo_cols = ["team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo"]

    early = frame[frame["season"] < 2024]
    late = frame[frame["season"] >= 2024]
    assert early[elo_cols].isna().all().all(), "2019-2023 Elo must be absent in the cut state"
    assert late[elo_cols].notna().all().all()

    # The median fill has no value for an object (Decimal) column, so the rows stay NaN.
    assert frame["team_off_elo"].dtype == object
    fill = frame[FEATURE_COLS].median(numeric_only=True)
    assert "team_off_elo" not in fill.index

    with pytest.raises(ValueError, match="NaN"):
        mtgp._fit(frame, "points", alpha=10.0)


def test_full_history_passes_step_6_with_no_nan_and_no_median_fill(frame_for):
    frame = frame_for(SEASONS)

    elo_cols = ["team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo"]
    assert frame[elo_cols].notna().all().all(), "every 2019-2025 team-week carries real Elo"

    for target in ("points", "total_yards", "pass_rate"):
        model, fill_values = mtgp._fit(frame, target, alpha=10.0)
        X = _prepare_x(frame, fill_values)
        assert not np.isnan(X).any()
        assert np.isfinite(model.predict(X)).all()

    # Elo reaches the model as stored, not as a median stand-in.
    X = _prepare_x(frame, pd.Series(dtype=float))
    stored = pd.to_numeric(frame["team_off_elo"]).to_numpy(dtype=float)
    assert np.array_equal(X[:, FEATURE_COLS.index("team_off_elo")], stored)
