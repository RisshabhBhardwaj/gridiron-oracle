"""build_forward_week refuses a week that already has played games (Q3).

Forward rows are upserted over feature_matrix by (player_id, game_id) with
their actual_* labels cleared. Run for a played week, that would erase real
labels and historical features. The guard is the first statement the method
runs, so a played week never reaches the upsert.

These tests run without a database. A fake connection answers the guard
query with a chosen value and records every statement it sees.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.feature_engineer import FeatureEngineer

AS_OF = datetime(2026, 9, 9, tzinfo=timezone.utc)


class _Cursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn
        self._rows: list = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._conn.statements.append(sql)
        if "AS played" in sql:
            self._rows = [{"played": self._conn.played}]
        else:
            # Candidate and prior queries: no rows. Any row-level work after the
            # guard would show up as a statement in self._conn.statements.
            self._rows = []

    def fetchall(self) -> list:
        return list(self._rows)


class _FakeConn:
    def __init__(self, played: bool) -> None:
        self.played = played
        self.statements: list[str] = []

    def cursor(self, **_kwargs: object) -> _Cursor:
        return _Cursor(self)


def _engineer(played: bool) -> tuple[FeatureEngineer, _FakeConn, list]:
    fe = FeatureEngineer("postgresql://unused")
    conn = _FakeConn(played)
    fe._conn = conn
    upserts: list = []
    fe._upsert_feature_rows = lambda rows: (upserts.extend(rows), len(rows))[1]
    fe._fill_prior_routes_run = lambda season: None
    fe._fill_prior_pbp_ngs_features = lambda season: None
    fe._fill_lagged_depth_chart_rank = lambda season: None
    return fe, conn, upserts


def test_refuses_a_week_that_already_has_played_games() -> None:
    fe, conn, upserts = _engineer(played=True)
    with pytest.raises(ValueError, match="already has played games") as excinfo:
        fe.build_forward_week(season=2026, week=3, as_of=AS_OF)
    assert "season=2026 week=3" in str(excinfo.value)
    assert upserts == [], "a refused week must not reach the feature_matrix upsert"


def test_refusal_happens_before_any_candidate_or_prior_query() -> None:
    fe, conn, _ = _engineer(played=True)
    with pytest.raises(ValueError):
        fe.build_forward_week(season=2026, week=3, as_of=AS_OF)
    assert len(conn.statements) == 1, conn.statements
    assert "AS played" in conn.statements[0]


def test_guard_checks_final_scores_and_game_logs_for_the_requested_week() -> None:
    fe, conn, _ = _engineer(played=False)
    with pytest.raises(ValueError, match="No as-of depth-chart roster"):
        # Passes the guard (nothing played), then finds no candidates. That is
        # the existing behaviour for an unplayed week with no depth chart.
        fe.build_forward_week(season=2026, week=3, as_of=AS_OF)
    guard_sql = conn.statements[0]
    assert "FROM games" in guard_sql
    assert "home_score IS NOT NULL" in guard_sql
    assert "FROM game_logs" in guard_sql


def test_unplayed_week_is_not_refused() -> None:
    fe, conn, upserts = _engineer(played=False)
    # With no candidates the method raises, so the unplayed path is observed as
    # "the guard passed and the candidate query ran".
    with pytest.raises(ValueError, match="No as-of depth-chart roster"):
        fe.build_forward_week(season=2026, week=3, as_of=AS_OF)
    assert any("JOIN LATERAL" in sql for sql in conn.statements)
    assert upserts == []
