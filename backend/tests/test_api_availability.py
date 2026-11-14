"""The API serves only players expected to play, and shows Questionable/Doubtful (decision 0012 D3).

The week query must carry the served predicate so Out, Reserve, Suspended and Inactive
rows never reach a user, and the status of the rows that do must come through as an
optional field. No frontend change: the field is simply new and optional.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.projection import ProjectionService, ProjectionResult
from backend.tests._fake_db import FakeConn
from pipeline.availability import EXCLUDED_STATUSES

WEEK_COLUMNS = [
    "player_id", "season", "week", "stat", "position", "projection", "floor", "ceiling",
    "boom_probability", "bust_probability", "fantasy_projection", "fantasy_floor",
    "fantasy_ceiling", "posterior_samples", "pipeline_run_id", "interval_method",
    "availability_status", "full_name", "team",
]


def _row(pid: str, status):
    return (pid, 2026, 11, "receiving_yards", "WR", 70.0, 40.0, 110.0, None, None,
            None, None, None, None, "approved_run", "causal_oof_conformal_90",
            status, f"Player {pid}", "KC")


@pytest.fixture
def conn(monkeypatch) -> FakeConn:
    fake = FakeConn([
        ("FROM   projections p", (WEEK_COLUMNS, [_row("a", None), _row("b", "Questionable")])),
        ("MAX(computed_at)", (["max"], [(datetime(2026, 11, 17, tzinfo=timezone.utc),)])),
    ])
    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", lambda *a, **k: fake)
    monkeypatch.setattr(
        "backend.app.services.projection.load_approved_pipeline_run_ids",
        lambda *a, **k: frozenset({"approved_run"}),
    )
    return fake


def test_week_query_filters_out_excluded_statuses(conn: FakeConn) -> None:
    ProjectionService("postgresql://x").get_week_projections(11, 2026, ["WR"], "receiving_yards")
    sql = next(e[1] for e in conn.log if e[0] == "execute" and "FROM   projections p" in e[1])
    assert "p.availability_status IS NULL OR p.availability_status NOT IN (" in sql
    for status in EXCLUDED_STATUSES:
        assert f"'{status}'" in sql


def test_week_results_carry_the_status(conn: FakeConn) -> None:
    results = ProjectionService("postgresql://x").get_week_projections(11, 2026, ["WR"], "receiving_yards")
    assert {r.player_id: r.availability_status for r in results} == {"a": None, "b": "Questionable"}


def test_week_endpoint_exposes_optional_availability_status(conn: FakeConn) -> None:
    body = TestClient(app).get("/projections/week/11?season=2026&stat=receiving_yards").json()
    by_id = {p["player_id"]: p for p in body["projections"]}
    assert by_id["a"]["availability_status"] is None
    assert by_id["b"]["availability_status"] == "Questionable"


def test_single_player_row_query_filters_excluded_statuses(conn: FakeConn) -> None:
    ProjectionService("postgresql://x")._load_projection_row("a", 11, 2026, "receiving_yards")
    sql = next(e[1] for e in conn.log if e[0] == "execute" and "FROM   projections" in e[1])
    assert "availability_status IS NULL OR availability_status NOT IN (" in sql


def test_projection_result_status_defaults_to_none() -> None:
    fields = ProjectionResult.__dataclass_fields__
    assert fields["availability_status"].default is None
