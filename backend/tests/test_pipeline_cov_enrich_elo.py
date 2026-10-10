"""
Offline coverage tests for pipeline/enrich_elo_embeddings.py.

The Elo system is replaced with a recording fake, so nothing is saved to
ml/elo_ratings.pkl. The DB is a FakeConn. enrich_embeddings is disabled by
an unconditional RuntimeError in production; those tests assert that
contract and that nothing is written.
"""

from __future__ import annotations

import pandas as pd
import psycopg2
import psycopg2.extras
import pytest

import ml.team_elo as team_elo
import pipeline.enrich_elo_embeddings as enrich
from backend.tests._pipeline_fakes import FakeConn, FakeConnFactory

GAME_COLS = ("game_id", "season", "week", "team", "opponent_team",
             "team_score", "opp_score", "home_away")
FM_COLS = ("player_id", "game_id", "season", "week", "team", "opponent_team")


class FakeElo:
    """Records fit/save/enrich calls and returns deterministic Elo columns."""

    def __init__(self, enrich_cols: tuple[str, ...] = (
        "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo",
        "elo_matchup_diff", "elo_implied_win_prob",
    )) -> None:
        self.enrich_cols = enrich_cols
        self.fit_frames: list[pd.DataFrame] = []
        self.saved: list[str] = []
        self.enrich_calls: list[tuple[int, int, int]] = []

    def fit(self, df: pd.DataFrame) -> "FakeElo":
        self.fit_frames.append(df.copy())
        return self

    def save(self, path: str) -> None:
        self.saved.append(path)

    def enrich_features(self, grp: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
        self.enrich_calls.append((season, week, len(grp)))
        out = grp.copy()
        for col in self.enrich_cols:
            out[col] = 1500.0 + week
        return out


@pytest.fixture
def conn(monkeypatch) -> FakeConn:
    db = FakeConn()
    monkeypatch.setattr(psycopg2, "connect", FakeConnFactory(db))
    monkeypatch.setattr(psycopg2.extras, "execute_batch",
                        lambda cur, sql, argslist, page_size=100: db.batches.append(list(argslist)))
    return db


@pytest.fixture
def elo(monkeypatch) -> FakeElo:
    fake = FakeElo()
    monkeypatch.setattr(team_elo, "get_elo_system", lambda db_url=None: fake)
    return fake


def _game_rows() -> list[tuple]:
    """One game, KC 27 at home vs BAL 20, as two game_logs-derived rows."""
    return [
        ("G1", 2024, 1, "KC", "BAL", 27.0, 20.0, "home"),
        ("G1", 2024, 1, "BAL", "KC", 20.0, 27.0, "away"),
    ]


# ── enrich_elo ────────────────────────────────────────────────────────────────

def test_enrich_elo_fits_each_game_once_in_temporal_order_and_saves_ratings(conn, elo):
    conn.respond("FROM game_logs gl", rows=_game_rows() + [
        ("G2", 2023, 17, "KC", "DEN", 10.0, 3.0, "home"),
        ("G2", 2023, 17, "DEN", "KC", 3.0, 10.0, "away"),
    ], columns=GAME_COLS)
    conn.respond("FROM feature_matrix WHERE team IS NOT NULL", rows=[], columns=FM_COLS)

    enrich.enrich_elo(conn)

    fit = elo.fit_frames[0]
    assert list(fit["season"]) == [2023, 2024]  # sorted ascending: no temporal leakage
    assert fit.iloc[1][["home_team", "away_team", "home_score", "away_score"]].tolist() == [
        "KC", "BAL", 27, 20,
    ]
    assert elo.saved == ["ml/elo_ratings.pkl"]


def test_enrich_elo_with_no_game_logs_returns_without_fitting_or_writing(conn, elo):
    conn.respond("FROM game_logs gl", rows=[], columns=GAME_COLS)
    enrich.enrich_elo(conn)

    assert elo.fit_frames == []
    assert elo.saved == []
    assert conn.batches == []


def test_enrich_elo_with_empty_feature_matrix_skips_the_update(conn, elo):
    conn.respond("FROM game_logs gl", rows=_game_rows(), columns=GAME_COLS)
    conn.respond("FROM feature_matrix WHERE team IS NOT NULL", rows=[], columns=FM_COLS)

    enrich.enrich_elo(conn)

    assert len(elo.fit_frames) == 1  # ratings were still fitted and saved
    assert elo.enrich_calls == []
    assert conn.batches == []


def test_enrich_elo_updates_feature_matrix_per_season_week_and_defaults_missing_cols(
    conn, monkeypatch
):
    conn.respond("FROM game_logs gl", rows=_game_rows(), columns=GAME_COLS)
    conn.respond("FROM feature_matrix WHERE team IS NOT NULL", rows=[
        ("00-1", "G1", 2024, 1, "KC", "BAL"),
        ("00-2", "G1", 2024, 1, "KC", "BAL"),
        ("00-3", "G9", 2024, 2, "BUF", "NYJ"),
    ], columns=FM_COLS)
    # This fake omits elo_implied_win_prob, so the enrichment must default it.
    partial = FakeElo(enrich_cols=(
        "team_off_elo", "team_def_elo", "opp_off_elo", "opp_def_elo", "elo_matchup_diff",
    ))
    monkeypatch.setattr(team_elo, "get_elo_system", lambda db_url=None: partial)

    enrich.enrich_elo(conn)

    assert sorted(partial.enrich_calls) == [(2024, 1, 2), (2024, 2, 1)]
    by_player = {row["player_id"]: row for row in conn.batches[0]}
    assert set(by_player) == {"00-1", "00-2", "00-3"}
    assert by_player["00-3"]["team_off_elo"] == 1502.0
    # The column the enrichment never produced falls back to 0.0 in the frame,
    # so the UPDATE payload carries that value rather than a missing key.
    assert by_player["00-1"]["elo_implied_win_prob"] == 0.0
    assert by_player["00-1"]["elo_matchup_diff"] == 1501.0


def test_enrich_elo_flushes_in_batches_of_5000_and_commits_each(conn, elo):
    conn.respond("FROM game_logs gl", rows=_game_rows(), columns=GAME_COLS)
    fm_rows = [(f"00-{i}", "G1", 2024, 1, "KC", "BAL") for i in range(5001)]
    conn.respond("FROM feature_matrix WHERE team IS NOT NULL", rows=fm_rows, columns=FM_COLS)

    enrich.enrich_elo(conn)

    assert [len(b) for b in conn.batches] == [5000, 1]
    assert conn.commits == 2  # one commit per flushed batch
    assert len(conn.batches[0]) == 5000


def test_enrich_elo_treats_nan_outputs_as_null(conn, monkeypatch):
    class _NanElo(FakeElo):
        def enrich_features(self, grp, season, week):
            out = grp.copy()
            for col in self.enrich_cols:
                out[col] = float("nan")
            return out

    monkeypatch.setattr(team_elo, "get_elo_system", lambda db_url=None: _NanElo())
    conn.respond("FROM game_logs gl", rows=_game_rows(), columns=GAME_COLS)
    conn.respond("FROM feature_matrix WHERE team IS NOT NULL",
                 rows=[("00-1", "G1", 2024, 1, "KC", "BAL")], columns=FM_COLS)
    enrich.enrich_elo(conn)

    row = conn.batches[0][0]
    assert row["team_off_elo"] is None
    assert row["elo_implied_win_prob"] is None


# ── enrich_embeddings ─────────────────────────────────────────────────────────

def test_enrich_embeddings_refuses_to_run_and_touches_no_data(conn):
    with pytest.raises(RuntimeError, match="disabled by the as-of feature contract"):
        enrich.enrich_embeddings(conn)
    assert conn.statements == []
    assert conn.batches == []


# ── helpers, main and connection ──────────────────────────────────────────────

@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (float("nan"), None), ("abc", None), ("2.5", 2.5), (3, 3.0)],
)
def test_safe_float(value, expected):
    assert enrich._safe_float(value) == expected


def test_main_runs_elo_then_embeddings_and_closes_the_connection(conn, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(enrich, "_get_conn", lambda: conn)
    monkeypatch.setattr(enrich, "enrich_elo", lambda c: order.append("elo"))
    monkeypatch.setattr(enrich, "enrich_embeddings", lambda c: order.append("embeddings"))

    enrich.main()

    assert order == ["elo", "embeddings"]
    assert conn.closed is True


def test_main_closes_connection_and_propagates_the_embedding_contract_error(conn, monkeypatch):
    monkeypatch.setattr(enrich, "_get_conn", lambda: conn)
    ran: list[str] = []
    monkeypatch.setattr(enrich, "enrich_elo", lambda c: ran.append("elo"))

    with pytest.raises(RuntimeError, match="disabled by the as-of feature contract"):
        enrich.main()
    assert ran == ["elo"]
    assert conn.closed is True


def test_get_conn_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL not set"):
        enrich._get_conn()


def test_get_conn_strips_asyncpg_driver(monkeypatch, conn):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db.test/oracle")
    assert enrich._get_conn() is conn
    assert conn.connect_dsns == ["postgresql://u:p@db.test/oracle"]
