"""
Tests for ProjectionService.get_season_team_wins and the
/projections/season/{season}/team-wins route (Phase 7, the Vikings test):
season_team_wins is written by scripts/materialize_season_simulation.py from
SeasonSimulator.run()'s team_win_totals, which was computed but never
persisted before this fix — there was no served number to compare a served
season win total against.
"""
from __future__ import annotations

import os

import pytest

from backend.app.services import projection as projection_mod
from backend.app.services.projection import ProjectionService

_DB_URL = os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle").replace(
    "postgresql+asyncpg://", "postgresql://"
)


def _skip_if_db_unreachable():
    try:
        import psycopg2

        psycopg2.connect(_DB_URL).close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc})")


def test_get_season_team_wins_reads_only_approved_runs(monkeypatch) -> None:
    svc = ProjectionService(_DB_URL)
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run1"}))

    calls: list[tuple] = []

    class _FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            calls.append(params)

        def fetchall(self):
            return []

    class _FakeConn:
        def cursor(self, cursor_factory=None):
            return _FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(
        "psycopg2.connect", lambda *a, **k: _FakeConn()
    )
    svc.get_season_team_wins(season=2026, start_week=5)
    assert calls
    season, start_week, approved = calls[0]
    assert approved == ["run1"]


def test_materializer_writes_and_service_reads_win_totals() -> None:
    _skip_if_db_unreachable()
    import psycopg2

    from scripts.materialize_season_simulation import materialize

    season, start_week, end_week = 2026, 17, 18
    try:
        n = materialize(
            season=season, start_week=start_week, end_week=end_week,
            n_simulations=20, positions=["TE"], database_url=_DB_URL,
        )
    except ValueError as exc:
        pytest.skip(f"no data to materialize for this slice: {exc}")
    assert n > 0

    conn = psycopg2.connect(_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pipeline_run_id FROM season_simulations "
                "WHERE season=%s AND start_week=%s ORDER BY created_at DESC LIMIT 1",
                (season, start_week),
            )
            run_id_row = cur.fetchone()
            assert run_id_row is not None
            run_id = run_id_row[0]

            cur.execute(
                "SELECT team, wins_mean FROM season_team_wins "
                "WHERE season=%s AND start_week=%s AND pipeline_run_id=%s",
                (season, start_week, run_id),
            )
            win_rows = cur.fetchall()
        assert win_rows, "expected season_team_wins rows for this run"
        assert all(w[1] >= 0.0 for w in win_rows)

        svc = ProjectionService(_DB_URL)
        import backend.app.services.projection as pm
        orig = pm.load_approved_pipeline_run_ids
        pm.load_approved_pipeline_run_ids = lambda: frozenset({run_id})
        try:
            served = svc.get_season_team_wins(season=season, start_week=start_week)
        finally:
            pm.load_approved_pipeline_run_ids = orig

        assert served
        assert {r["team"] for r in served} == {r[0] for r in win_rows}
    finally:
        conn2 = psycopg2.connect(_DB_URL)
        try:
            with conn2.cursor() as cur:
                cur.execute("DELETE FROM season_simulations WHERE pipeline_run_id=%s", (run_id,))
                cur.execute("DELETE FROM season_simulation_weeks WHERE pipeline_run_id=%s", (run_id,))
                cur.execute("DELETE FROM season_team_wins WHERE pipeline_run_id=%s", (run_id,))
            conn2.commit()
        finally:
            conn2.close()
        conn.close()
