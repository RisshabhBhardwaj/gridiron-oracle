"""Backend tests for causal, ID-keyed draft board helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backend.app.api.draft import _fetch_db_projections_ppr
from ml.draft_projection import build_preseason_projections


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, _params):
        self.sql = sql

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, rows):
        self.cursor_obj = _Cursor(rows)

    def cursor(self, **_kwargs):
        return self.cursor_obj


def test_db_only_preseason_fallback_uses_projection_sum() -> None:
    conn = _Conn([{
        "player_id": "p1", "player_name": "Player One", "position": "WR", "team": "AAA",
        "fantasy_ppr": 220.0, "as_of": date(2026, 8, 1),
    }])
    rows, source, as_of = _fetch_db_projections_ppr(conn, 2026)
    assert rows["p1"]["fantasy_ppr"] == 220.0
    assert source == "preseason_projection_rows"
    assert as_of == date(2026, 8, 1)
    assert "SUM(pr.projection)" in conn.cursor_obj.sql
    assert "pr.week = 0" in conn.cursor_obj.sql
    assert "AVG(" not in conn.cursor_obj.sql


def test_default_adp_pool_is_the_fixed_projection_universe() -> None:
    from backend.app.api.draft import _fetch_adp
    conn = _Conn([])
    _fetch_adp(conn, 2026, "sleeper", "ppr", None)
    assert "UPPER(position) IN ('QB', 'RB', 'WR', 'TE')" in conn.cursor_obj.sql


def test_duplicate_db_projection_players_fail_closed() -> None:
    row = {"player_id": "p1", "player_name": "Player One", "position": "WR", "team": "AAA", "fantasy_ppr": 220.0, "as_of": date(2026, 8, 1)}
    with pytest.raises(ValueError, match="duplicate preseason projections"):
        _fetch_db_projections_ppr(_Conn([row, row]), 2026)


def test_preseason_rank_is_not_driven_by_target_season_availability() -> None:
    players = pd.DataFrame([
        {"id": f"p{i}", "full_name": f"P{i}", "position": "WR", "team": "AAA", "entry_year": 2020}
        for i in range(12)
    ])
    logs = pd.DataFrame([
        {"player_id": f"p{i}", "season": 2025, "week": 1, "season_type": "REG", "fantasy_points_ppr": float(30 - i)}
        for i in range(12)
    ])
    projections = build_preseason_projections(players, logs, season=2026).sort_values("projection", ascending=False)
    # Deliberately reverse realised 2026 appearances.  They are unavailable to
    # the builder, so the resulting draft rank cannot follow them.
    target_games = pd.Series([10, 2, 8, 12, 5, 4, 1, 9, 6, 3, 11, 7], index=[f"p{i}" for i in range(12)])
    rank_games = projections["player_id"].map(target_games).rank().to_list()
    rank_projection = projections["projection"].rank().to_list()
    assert abs(pd.Series(rank_projection).corr(pd.Series(rank_games), method="spearman")) < 0.25
