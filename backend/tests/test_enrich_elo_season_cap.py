"""enrich_elo must fit Elo only through LAST_COMPLETE_SEASON (audit C2).

Training, OOF and evaluation stop at LAST_COMPLETE_SEASON. Without a cap here,
2026 results would enter the Elo state that 2019-2025 rows are enriched from.
"""

from __future__ import annotations

import pandas as pd
import pytest

import ml.team_elo as team_elo
import pipeline.enrich_elo_embeddings as enrich
from backend.tests._pipeline_fakes import FakeConn
from ml.season_constants import LAST_COMPLETE_SEASON

GAME_COLS = ("game_id", "season", "week", "team", "opponent_team",
             "team_score", "opp_score", "home_away")
FM_COLS = ("player_id", "game_id", "season", "week", "team", "opponent_team")


class _RecordingElo:
    def __init__(self) -> None:
        self.fit_frames: list[pd.DataFrame] = []

    def fit(self, df: pd.DataFrame) -> "_RecordingElo":
        self.fit_frames.append(df.copy())
        return self

    def save(self, path: str) -> None:
        pass

    def enrich_features(self, grp: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
        return grp.copy()


def _game(game_id: str, season: int, week: int, home: str, away: str, hs: float, aw: float):
    return [
        (game_id, season, week, home, away, hs, aw, "home"),
        (game_id, season, week, away, home, aw, hs, "away"),
    ]


@pytest.fixture
def elo(monkeypatch) -> _RecordingElo:
    fake = _RecordingElo()
    monkeypatch.setattr(team_elo, "get_elo_system", lambda db_url=None: fake)
    return fake


def test_fit_excludes_seasons_after_the_cap(elo, monkeypatch):
    monkeypatch.setattr(enrich, "LAST_COMPLETE_SEASON", 2025)
    db = FakeConn()
    db.respond("FROM game_logs gl", rows=(
        _game("G2025", 2025, 17, "KC", "BAL", 27.0, 20.0)
        + _game("G2026", 2026, 1, "KC", "DEN", 31.0, 7.0)
    ), columns=GAME_COLS)
    db.respond("FROM feature_matrix WHERE team IS NOT NULL", rows=[], columns=FM_COLS)

    enrich.enrich_elo(db)

    fitted = elo.fit_frames[0]
    assert set(fitted["season"]) == {2025}
    assert len(fitted) == 1  # one game, from the two team rows


def test_query_receives_the_cap_as_a_parameter(elo, monkeypatch):
    monkeypatch.setattr(enrich, "LAST_COMPLETE_SEASON", 2025)
    db = FakeConn()
    db.respond("FROM game_logs gl", rows=[], columns=GAME_COLS)

    enrich.enrich_elo(db)

    (sql, params), = db.statements_containing("FROM game_logs gl")
    assert "gl.season <= %(last_season)s" in sql
    assert params == {"last_season": 2025}


def test_module_cap_is_the_project_constant():
    # Guards against the module drifting from ml/season_constants.py.
    assert enrich.LAST_COMPLETE_SEASON == LAST_COMPLETE_SEASON
