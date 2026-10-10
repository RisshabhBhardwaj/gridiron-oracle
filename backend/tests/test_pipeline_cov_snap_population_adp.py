"""
Offline coverage tests for pipeline/test_snap_population.py (a DB script)
and pipeline/adp_resolution.py (the audit path of resolve_and_audit).

The script's DB and Normalizer are replaced by fakes. The ADP audit runs
against FakeConn with scripted catalogue rows.
"""

from __future__ import annotations

import json

import psycopg2
import pytest

import pipeline.adp_resolution as adp
import pipeline.test_snap_population as snap
from backend.tests._pipeline_fakes import FakeConn, FakeConnFactory
from pipeline.normalize import NormalizeSummary

# ── test_snap_population.main ─────────────────────────────────────────────────


def test_snap_population_main_exits_when_database_url_is_unset(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        snap.main()
    assert exc.value.code == 1
    assert "DATABASE_URL not set" in capsys.readouterr().out


class _FakeNormalizer:
    runs: list[list[int]] = []

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    def __enter__(self) -> "_FakeNormalizer":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def run(self, seasons=None) -> NormalizeSummary:
        _FakeNormalizer.runs.append(seasons)
        return NormalizeSummary(game_logs_upserted=9)


def test_snap_population_main_reports_snap_counts_before_and_after(monkeypatch, capsys):
    conn = FakeConn()
    monkeypatch.setattr(psycopg2, "connect", FakeConnFactory(conn))
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.test/oracle")
    monkeypatch.setattr(snap, "Normalizer", _FakeNormalizer)
    _FakeNormalizer.runs = []
    conn.respond("FROM game_logs WHERE offense_snaps IS NOT NULL", rows=[(2019, 4), (2024, 7)])

    snap.main()

    out = capsys.readouterr().out
    assert "Before normalize:" in out
    assert "Season 2019: 4 rows with snaps" in out
    assert "After normalize:" in out
    assert out.count("Season 2024: 7 rows with snaps") == 2  # counted before and after
    assert _FakeNormalizer.runs == [list(range(2019, 2026))]
    assert conn.connect_dsns == ["postgresql://u:p@db.test/oracle"] * 2  # before and after queries


# ── adp_resolution.resolve_and_audit ──────────────────────────────────────────


def _catalogue(conn: FakeConn, *, with_id_maps: bool) -> None:
    conn.respond("FROM players", rows=[
        ("00-1", "Patrick Mahomes", "QB", "KC"),
        ("00-2", "Travis Kelce", "TE", "KC"),
        ("00-3", "Travis Kelce", "WR", "BUF"),  # same name, different player
    ], columns=("id", "full_name", "position", "team"))
    conn.respond("information_schema.tables", rows=[(with_id_maps,)])
    conn.respond("FROM fantasy_player_ids", rows=[
        ("00-1", "1001", None, None, None, None),
    ], columns=("gsis_id", "sleeper_id", "espn_id", "yahoo_id", "mfl_id", "fantasypros_id"))


def test_resolve_and_audit_writes_one_audit_row_per_input_with_match_status(monkeypatch):
    conn = FakeConn()
    _catalogue(conn, with_id_maps=True)
    rows = [
        {"player_name": "Patrick Mahomes", "position": "qb", "team": "KC", "sleeper_id": "1001"},
        {"player_name": "Travis Kelce", "position": "TE", "team": "KC"},  # narrowed by team/position
        {"player_name": "Unknown Person", "position": "RB", "team": "NYJ"},  # no candidates
    ]

    resolved = adp.resolve_and_audit(conn, season=2025, source="fp", scoring="ppr", rows=rows)

    assert [r["match_status"] for r in resolved] == ["matched", "matched", "unmatched"]
    assert [r["match_method"] for r in resolved] == [
        "fantasy_player_ids:sleeper_id", "name_position_team", "unmatched",
    ]
    assert resolved[1]["player_id"] == "00-2"
    assert resolved[2]["player_id"] is None

    inserts = conn.statements_containing("INSERT INTO adp_player_matches")
    assert len(inserts) == 3
    first_params = inserts[0][1]
    assert first_params[:5] == (2025, "fp", "ppr", "Patrick Mahomes", "patrick mahomes")
    assert first_params[7] == "00-1"  # matched_player_id
    assert json.loads(first_params[10]) == ["00-1"]  # candidate ids are stored as JSON


def test_resolve_and_audit_creates_the_audit_tables_before_writing(monkeypatch):
    conn = FakeConn()
    _catalogue(conn, with_id_maps=False)
    adp.resolve_and_audit(conn, season=2025, source="fp", scoring="std", rows=[
        {"player_name": "Patrick Mahomes", "position": "QB", "team": "KC"},
    ])

    ddl = [sql for sql, _ in conn.statements]
    assert " ".join(adp.CREATE_FANTASY_ADP.split()) in ddl
    assert " ".join(adp.CREATE_ADP_PLAYER_MATCHES.split()) in ddl
    # No fantasy_player_ids table: the vendor crosswalk is skipped and name matching still works.
    assert conn.statements_containing("FROM fantasy_player_ids") == []
    inserts = conn.statements_containing("INSERT INTO adp_player_matches")
    assert inserts[0][1][7] == "00-1"


def test_resolve_and_audit_marks_ambiguous_name_matches_unmatched(monkeypatch):
    conn = FakeConn()
    _catalogue(conn, with_id_maps=False)
    resolved = adp.resolve_and_audit(conn, season=2025, source="fp", scoring="std", rows=[
        {"player_name": "Travis Kelce", "position": "RB"},  # no team, no RB: two candidates remain
    ])

    assert resolved[0]["match_status"] == "unmatched"
    assert resolved[0]["match_method"] == "ambiguous_name"
    assert resolved[0]["candidate_player_ids"] == ["00-2", "00-3"]
