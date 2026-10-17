"""Elo and PBP enrichment write only the seasons they are given (H-5).

The weekly refresh used to rewrite Elo and PBP columns for 2019 to 2026 on every
run, with no run id. These tests check the scope at the statement level, with
fake connections. No database is used.

The Elo fit still reads every game (a running rating), but only the feature_matrix
rows for the requested seasons are updated.
"""

from __future__ import annotations

import psycopg2.extras
import pytest

import pipeline.enrich_elo_embeddings as elo_mod
import pipeline.pbp_pipeline as pbp_mod

GAME_ROWS = [
    # game_id, season, week, team, opponent_team, team_score, opp_score, home_away
    ("2025_01_KC_BAL", 2025, 1, "KC", "BAL", 24.0, 17.0, "home"),
    ("2025_01_KC_BAL", 2025, 1, "BAL", "KC", 17.0, 24.0, "away"),
    ("2026_01_KC_BAL", 2026, 1, "KC", "BAL", 20.0, 10.0, "home"),
    ("2026_01_KC_BAL", 2026, 1, "BAL", "KC", 10.0, 20.0, "away"),
]
FM_ROWS = [
    # player_id, game_id, season, week, team, opponent_team
    ("P1", "2025_01_KC_BAL", 2025, 1, "KC", "BAL"),
    ("P1", "2026_01_KC_BAL", 2026, 1, "KC", "BAL"),
    ("P2", "2026_01_KC_BAL", 2026, 1, "BAL", "KC"),
]


class _EloCursor:
    def __init__(self, conn: "_EloConn") -> None:
        self._conn = conn
        self._rows: list = []
        self.description: list = []

    def execute(self, sql: str, params=()) -> None:
        self._conn.executed.append((sql, tuple(params)))
        if "FROM game_logs gl" in sql:
            cols = ["game_id", "season", "week", "team", "opponent_team",
                    "team_score", "opp_score", "home_away"]
            self.description = [(c,) for c in cols]
            self._rows = list(GAME_ROWS)
        elif "FROM feature_matrix" in sql:
            cols = ["player_id", "game_id", "season", "week", "team", "opponent_team"]
            self.description = [(c,) for c in cols]
            scope = params[0] if params else None
            self._rows = [r for r in FM_ROWS if scope is None or r[2] in scope]
        else:
            raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def fetchall(self) -> list:
        return list(self._rows)


class _EloConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []

    def cursor(self) -> _EloCursor:
        return _EloCursor(self)

    def commit(self) -> None:
        pass


class _FakeElo:
    def fit(self, df) -> None:
        self.fitted_games = len(df)

    def save(self, path: str) -> None:  # the real fit writes ml/elo_ratings.pkl
        pass

    def enrich_features(self, grp, season: int, week: int):
        out = grp.copy()
        out["team_off_elo"] = 1500.0
        out["team_def_elo"] = 1500.0
        out["opp_off_elo"] = 1500.0
        out["opp_def_elo"] = 1500.0
        out["elo_matchup_diff"] = 0.0
        out["elo_implied_win_prob"] = 0.5
        return out


def _run_elo(monkeypatch, seasons):
    import ml.team_elo as team_elo

    updates: list[dict] = []

    def record_batch(cur, sql, rows, page_size=None):
        updates.extend(rows)

    monkeypatch.setattr(team_elo, "get_elo_system", lambda: _FakeElo())
    monkeypatch.setattr(psycopg2.extras, "execute_batch", record_batch)
    conn = _EloConn()
    elo_mod.enrich_elo(conn, seasons=seasons)
    return conn, {(r["player_id"], r["game_id"]) for r in updates}


def test_elo_update_touches_only_the_requested_season(monkeypatch):
    conn, updated = _run_elo(monkeypatch, seasons=[2026])

    fm_sql, fm_params = next((s, p) for s, p in conn.executed if "FROM feature_matrix" in s)
    assert fm_params == ([2026],), "the feature_matrix read is scoped in SQL"
    assert "season = ANY" in fm_sql
    assert updated == {("P1", "2026_01_KC_BAL"), ("P2", "2026_01_KC_BAL")}


def test_elo_update_without_a_season_list_keeps_the_full_etl_behaviour(monkeypatch):
    conn, updated = _run_elo(monkeypatch, seasons=None)

    fm_sql, fm_params = next((s, p) for s, p in conn.executed if "FROM feature_matrix" in s)
    assert fm_params == ()
    assert "season = ANY" not in fm_sql
    assert updated == {("P1", "2025_01_KC_BAL"), ("P1", "2026_01_KC_BAL"), ("P2", "2026_01_KC_BAL")}


def test_run_elo_enrichment_passes_the_seasons_through(monkeypatch):
    seen: dict = {}

    class _Conn:
        def close(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr(elo_mod, "_get_conn", lambda: _Conn())
    monkeypatch.setattr(elo_mod, "enrich_elo", lambda conn, seasons=None: seen.update(seasons=seasons))

    elo_mod.run_elo_enrichment([2026])

    assert seen == {"seasons": [2026], "closed": True}


# ── PBP ──────────────────────────────────────────────────────────────────────

class _PbpConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []

    def cursor(self) -> "_PbpConn":
        return self

    def __enter__(self) -> "_PbpConn":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def pbp_loads(monkeypatch):
    """Every season's PBP load fails, so each season is recorded as failed and dead-lettered."""
    loaded: list[int] = []
    conn = _PbpConn()

    def no_data(season: int):
        loaded.append(season)
        raise RuntimeError("no pbp")

    monkeypatch.setattr(pbp_mod, "_get_conn", lambda: conn)
    monkeypatch.setattr(pbp_mod, "_load_pbp_season", no_data)
    return loaded, conn


def test_pbp_main_runs_only_the_requested_seasons(pbp_loads):
    loaded, conn = pbp_loads

    failed = pbp_mod.main([2026])

    assert loaded == [2026]
    assert failed == [2026]
    assert any("INSERT INTO dead_letter" in sql for sql, _ in conn.executed)


def test_pbp_main_default_runs_every_season(pbp_loads):
    loaded, _ = pbp_loads

    failed = pbp_mod.main()

    assert loaded == list(pbp_mod.SEASONS)
    assert failed == list(pbp_mod.SEASONS)
