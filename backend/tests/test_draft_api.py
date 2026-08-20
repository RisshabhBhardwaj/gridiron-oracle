"""Backend tests for causal, ID-keyed draft board helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backend.app.api.draft import _fetch_db_projections_ppr, model_ranks_by_vor
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


def test_retired_players_are_dropped_from_preseason_universe() -> None:
    players = pd.DataFrame([
        {"id": "brady", "full_name": "Tom Brady", "position": "QB", "team": "TB", "entry_year": 2000},
        {"id": "mahomes", "full_name": "Patrick Mahomes", "position": "QB", "team": "KC", "entry_year": 2017},
    ])
    logs = pd.DataFrame([
        {"player_id": "brady", "season": 2022, "week": 1, "season_type": "REG", "fantasy_points_ppr": 20.0},
        {"player_id": "mahomes", "season": 2025, "week": 1, "season_type": "REG", "fantasy_points_ppr": 22.0},
    ])
    projections = build_preseason_projections(players, logs, season=2026)
    assert "brady" not in set(projections["player_id"])
    assert "mahomes" in set(projections["player_id"])


def test_rookies_with_no_history_are_unranked() -> None:
    adp = [
        {"player_id": "vet", "position": "WR"},
        {"player_id": "rook", "position": "WR"},
    ]
    model = {
        "vet": {"fantasy_ppr": 180.0, "position": "WR", "historical_games": 16},
        "rook": {"fantasy_ppr": 250.0, "position": "WR", "historical_games": 0},
    }
    ranks = model_ranks_by_vor(adp, model)
    assert "rook" not in ranks
    assert ranks["vet"] == 1


def test_practice_squad_without_entry_year_is_not_a_rookie() -> None:
    players = pd.DataFrame([
        {"id": "vet", "full_name": "Vet", "position": "WR", "team": "AAA", "entry_year": 2018},
        {"id": "ghost", "full_name": "Ghost", "position": "WR", "team": "AAA", "entry_year": 2015},
        {"id": "rook", "full_name": "Rook", "position": "WR", "team": "AAA", "entry_year": 2026, "draft_round": 2},
    ])
    logs = pd.DataFrame([
        {"player_id": "vet", "season": 2025, "week": 1, "season_type": "REG", "fantasy_points_ppr": 18.0, "team": "AAA"},
    ])
    projections = build_preseason_projections(players, logs, season=2026)
    assert "ghost" not in set(projections["player_id"])
    assert "rook" in set(projections["player_id"])
    assert projections.loc[projections["player_id"] == "vet", "projection_basis"].iloc[0] == "marcel_rate_x_playing_time"


def test_rookies_disperse_by_draft_capital() -> None:
    players = pd.DataFrame([
        {"id": "r1", "full_name": "R1", "position": "WR", "team": "AAA", "entry_year": 2026, "draft_round": 1},
        {"id": "r2", "full_name": "R2", "position": "WR", "team": "AAA", "entry_year": 2026, "draft_round": 7},
        {"id": "vet", "full_name": "Vet", "position": "WR", "team": "AAA", "entry_year": 2018},
    ])
    logs = pd.DataFrame([
        {"player_id": "vet", "season": 2025, "week": 1, "season_type": "REG",
         "fantasy_points_ppr": 18.0, "team": "AAA", "position": "WR"},
    ])
    projections = build_preseason_projections(players, logs, season=2026).set_index("player_id")
    assert float(projections.loc["r1", "projection"]) > float(projections.loc["r2", "projection"])
    assert projections.loc["r1", "projection_basis"] == "rookie_draft_capital_vacated_opportunity"


def test_prior_season_stack_rate_ignores_target_season_oof() -> None:
    from ml.draft_projection import apply_prior_season_stack_rate

    projections = pd.DataFrame([
        {
            "player_id": "p1", "player_name": "P", "position": "WR", "team": "AAA",
            "per_game_mean": 10.0, "games_played_prior": 16.0, "historical_games": 16,
            "projection": 160.0, "projection_basis": "historical_ppr_x_games_prior",
        }
    ])
    oof = pd.DataFrame([
        {"player_id": "p1", "season": 2024, "y_pred": 20.0},
        {"player_id": "p1", "season": 2025, "y_pred": 99.0},
    ])
    out = apply_prior_season_stack_rate(projections, oof, season=2025)
    assert out.loc[0, "per_game_mean"] == 20.0
    assert out.loc[0, "projection"] == 320.0
    assert out.loc[0, "projection_basis"] == "prior_season_stack_oof_x_playing_time"


def test_vor_does_not_fill_the_top_board_with_quarterbacks() -> None:
    adp = (
        [{"player_id": f"qb{i}", "position": "QB"} for i in range(12)]
        + [{"player_id": f"rb{i}", "position": "RB"} for i in range(20)]
    )
    model = {
        **{f"qb{i}": {"fantasy_ppr": 300.0 - i, "position": "QB", "historical_games": 16} for i in range(12)},
        **{f"rb{i}": {"fantasy_ppr": 180.0 - i, "position": "RB", "historical_games": 16} for i in range(20)},
    }
    ranks = model_ranks_by_vor(adp, model)
    top24 = [pid for pid, rank in ranks.items() if rank <= 24]
    assert sum(pid.startswith("qb") for pid in top24) <= 8


def test_rookies_with_a_fitted_basis_are_ranked() -> None:
    """The fitted rookie curve ranks rookies better than ADP, so they belong on the board.

    The companion test above still holds for a zero-history player carrying no
    basis: that projection is backed by nothing and stays unranked.
    """
    adp = [
        {"player_id": "vet", "position": "WR"},
        {"player_id": "rook", "position": "WR"},
    ]
    model = {
        "vet": {"fantasy_ppr": 180.0, "position": "WR", "historical_games": 16},
        "rook": {
            "fantasy_ppr": 250.0,
            "position": "WR",
            "historical_games": 0,
            "projection_basis": "rookie_draft_capital_vacated_opportunity",
        },
    }
    ranks = model_ranks_by_vor(adp, model)
    assert ranks["rook"] == 1
    assert ranks["vet"] == 2
