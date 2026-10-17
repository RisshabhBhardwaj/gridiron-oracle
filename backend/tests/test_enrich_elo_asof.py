"""enrich_elo end to end: the Elo values written for a feature row must not move
when that row's game or a later game changes (audit C1).

The real TeamEloSystem runs through enrich_elo; the DB is a FakeConn. save() is
patched out so the test never writes ml/elo_ratings.pkl.
"""

from __future__ import annotations

import psycopg2
import psycopg2.extras
import pytest

import ml.team_elo as team_elo
import pipeline.enrich_elo_embeddings as enrich
from backend.tests._elo_asof_fixture import (
    ASOF_EARLIER,
    ASOF_LATER,
    ASOF_TARGET,
    asof_game_id,
    asof_games_frame,
)
from backend.tests._pipeline_fakes import FakeConn
from ml.team_elo import TeamEloSystem

GAME_COLS = ("game_id", "season", "week", "team", "opponent_team",
             "team_score", "opp_score", "home_away")
FM_COLS = ("player_id", "game_id", "season", "week", "team", "opponent_team")
ELO_COLS = ("team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
            "elo_matchup_diff", "elo_implied_win_prob")
TARGET_GAME_ID = asof_game_id(*ASOF_TARGET)


def _game_log_rows(frame) -> list[tuple]:
    rows = []
    for g in frame.itertuples(index=False):
        hs, aw = float(g.home_score), float(g.away_score)
        rows.append((g.game_id, int(g.season), int(g.week), g.home_team, g.away_team, hs, aw, "home"))
        rows.append((g.game_id, int(g.season), int(g.week), g.away_team, g.home_team, aw, hs, "away"))
    return rows


def _target_elo_values(monkeypatch, flip=None) -> dict[str, float]:
    db = FakeConn()
    db.respond("FROM game_logs gl", rows=_game_log_rows(asof_games_frame(flip=flip)), columns=GAME_COLS)
    db.respond("FROM feature_matrix WHERE team IS NOT NULL",
               rows=[("P1", TARGET_GAME_ID, 2024, 5, "MIN", "GB")], columns=FM_COLS)
    monkeypatch.setattr(psycopg2.extras, "execute_batch",
                        lambda cur, sql, argslist, page_size=100: db.batches.append(list(argslist)))
    # A fresh system per run: the module singleton would carry state between calls.
    monkeypatch.setattr(team_elo, "get_elo_system", lambda db_url=None: TeamEloSystem())
    monkeypatch.setattr(TeamEloSystem, "save", lambda self, path: None)

    enrich.enrich_elo(db)

    written = [row for batch in db.batches for row in batch if row["game_id"] == TARGET_GAME_ID]
    assert len(written) == 1
    return {col: float(written[0][col]) for col in ELO_COLS}


@pytest.fixture
def base(monkeypatch) -> dict[str, float]:
    return _target_elo_values(monkeypatch)


def test_enrich_writes_pregame_values_for_the_target_row(base):
    assert base["team_off_elo"] != pytest.approx(1500.0)


def test_flipping_the_target_game_does_not_move_its_written_values(monkeypatch, base):
    assert _target_elo_values(monkeypatch, flip=ASOF_TARGET) == pytest.approx(base, rel=1e-12)


def test_flipping_a_later_game_does_not_move_the_target_row(monkeypatch, base):
    assert _target_elo_values(monkeypatch, flip=ASOF_LATER) == pytest.approx(base, rel=1e-12)


def test_flipping_an_earlier_game_moves_the_target_row(monkeypatch, base):
    # Positive control: the end-to-end path does see prior results.
    assert _target_elo_values(monkeypatch, flip=ASOF_EARLIER)["team_off_elo"] != pytest.approx(
        base["team_off_elo"]
    )
