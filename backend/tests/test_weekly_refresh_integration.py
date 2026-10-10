"""
backend/tests/test_weekly_refresh_integration.py

Integration tests for scripts/weekly_refresh.py against a real PostgreSQL database.

Only network-facing ingest is replaced:
  - step 2: Orchestrator.run (nflreadpy source adapters)
  - step 3: scraper.adapters.ff_playerids.load_playerids_frame (nflreadpy download)
Everything else runs for real: the step 3 upsert, the step 4 stack materialization
from the SHA-pinned artifacts, and the step 5-10 functions against the database.

Each module gets a throwaway database (template0, migrated to alembic head) that is
dropped at the end. The shared `oracle` database is never written.

Run:
    DATABASE_URL=postgresql://oracle:oracle@localhost:15439/oracle \
        pytest backend/tests/test_weekly_refresh_integration.py -m integration -q
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import psycopg2
import psycopg2.extras
import pytest

from pipeline.orchestrator import Orchestrator, OrchestratorSummary, SeasonResult, StepResult
from pipeline.schema import normalize_dsn, upgrade_head
from scripts.weekly_refresh import promote_run_id, run_weekly_refresh

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
APPROVED = "weekly_auto_2026"
PENDING = "weekly_auto_2026_pending"
# Approved in releases/current_baseline.json; present in the real keep set.
STACK_APPROVED = "stack_materialize_20260822T144443Z"

# Reports written by steps 4 and 7 that are pinned as evidence in
# releases/artifacts/MANIFEST.json. Snapshot and restore them around each test.
_PINNED_REPORTS = (
    ROOT / "reports" / "materialize_stack_projections.json",
    ROOT / "reports" / "season_acceptance.json",
)

_PLAYERIDS = pd.DataFrame([
    {"gsis_id": "00-0000001", "name": "Seed QB", "position": "QB", "team": "AAA", "sleeper_id": 1.0},
    {"gsis_id": "00-0000002", "name": "Seed WR", "position": "WR", "team": "BBB", "sleeper_id": 2.0},
])


# ── Database fixtures ────────────────────────────────────────────────────────

def _admin_url() -> str:
    return normalize_dsn(
        os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle")
    )


def _with_dbname(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


@pytest.fixture(scope="module")
def refresh_db():
    """A fresh database migrated to head, dropped when the module finishes."""
    admin_url = _admin_url()
    try:
        admin = psycopg2.connect(admin_url, connect_timeout=3)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres unavailable for weekly refresh integration: {exc}")
    admin.autocommit = True
    name = f"gor_refresh_it_{uuid.uuid4().hex[:10]}"
    admin.cursor().execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    url = _with_dbname(admin_url, name)
    # alembic/env.py prefers $DATABASE_URL over the URL passed to upgrade_head,
    # so point the environment at the throwaway database while migrating.
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        upgrade_head(url)
    finally:
        _restore_env("DATABASE_URL", previous)
    try:
        yield url
    finally:
        admin.cursor().execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


@pytest.fixture
def db(refresh_db: str) -> str:
    """Empty every public table (except alembic_version) before each test."""
    with closing(psycopg2.connect(refresh_db)) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename <> 'alembic_version'"
            )
            names = [row[0] for row in cur.fetchall()]
            if names:
                cur.execute(
                    "TRUNCATE " + ", ".join(f'"{n}"' for n in names) + " RESTART IDENTITY CASCADE"
                )
    return refresh_db


@pytest.fixture(autouse=True)
def _restore_pinned_reports():
    snapshot = {p: (p.read_bytes() if p.exists() else None) for p in _PINNED_REPORTS}
    yield
    for path, data in snapshot.items():
        if data is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(data)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _query(url: str, sql: str, params=()):
    with closing(psycopg2.connect(url)) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _scalar(url: str, sql: str, params=()):
    return _query(url, sql, params)[0][0]


def _execute(url: str, sql: str, params=()) -> None:
    with closing(psycopg2.connect(url)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def _dead_letters(url: str) -> list[tuple[str, str]]:
    return _query(url, "SELECT source, error_message FROM dead_letter ORDER BY source, id")


def _seed_schedule(url: str) -> None:
    """Teams AAA and BBB, two completed 2026 weeks and one unplayed week."""
    now = datetime.now(timezone.utc)
    games = []
    for week in (1, 2):
        games.append((f"2026_{week:02d}_BBB_AAA", week, now - timedelta(days=7 * (3 - week))))
    games.append(("2026_03_BBB_AAA", 3, now + timedelta(days=7)))
    with closing(psycopg2.connect(url)) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, "INSERT INTO teams (id) VALUES %s", [("AAA",), ("BBB",)]
            )
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO games (id, season, week, home_team, away_team, kickoff_at) VALUES %s",
                [(gid, 2026, wk, "AAA", "BBB", kick) for gid, wk, kick in games],
            )
        conn.commit()


def _seed_approved_projection(url: str, player: str, game: str, week: int) -> None:
    _execute(
        url,
        """
        INSERT INTO projections (player_id, game_id, season, week, stat, position,
                                 projection, pipeline_run_id, max_train_season)
        VALUES (%s, %s, 2026, %s, 'fantasy_ppr', 'QB', 12.0, %s, 2025)
        """,
        (player, game, week, APPROVED),
    )


def _clean_summary() -> OrchestratorSummary:
    steps = [StepResult(name=n, elapsed_s=0.0, ok=True) for n in ("ingest", "normalize", "feature_engineer")]
    return OrchestratorSummary(
        seasons=[2026],
        dry_run=False,
        season_results=[SeasonResult(season=2026, dry_run=False, steps=steps)],
    )


@contextmanager
def _ingest_mocked(summary: OrchestratorSummary):
    """Replace the two network calls. Step 3's upsert still runs against the database."""
    with patch.object(Orchestrator, "run", return_value=summary), patch(
        "scraper.adapters.ff_playerids.load_playerids_frame", return_value=_PLAYERIDS
    ):
        yield


def _first_stack_row(season: int) -> dict:
    """First pinned fantasy_ppr QB stack row for `season` (read from the pinned CSV)."""
    from ml.artifact_manifest import get_manifest

    frame = pd.read_csv(get_manifest().resolve_stack("fantasy_ppr", "QB"))
    row = frame[frame["season"] == season].iloc[0]
    return {"player_id": str(row["player_id"]), "game_id": str(row["game_id"]), "week": int(row["week"])}


# ── Tests ────────────────────────────────────────────────────────────────────

def test_failed_ingest_step_dead_letters_and_does_not_promote(db):
    _seed_schedule(db)
    _seed_approved_projection(db, player="SEED-APPROVED", game="2026_01_BBB_AAA", week=1)
    failed = StepResult(name="ingest", elapsed_s=0.1, ok=False, error="nflreadpy timed out")
    summary = OrchestratorSummary(
        seasons=[2026],
        dry_run=False,
        season_results=[SeasonResult(season=2026, dry_run=False, steps=[failed])],
    )

    with _ingest_mocked(summary):
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1
    assert _dead_letters(db) == [
        (
            "weekly_refresh/orchestrator_ingest",
            "Orchestrator run failed: season=2026 step=ingest: nflreadpy timed out",
        )
    ]
    assert _scalar(db, "SELECT COUNT(*) FROM fantasy_player_ids") == 0  # step 3 never ran
    assert _scalar(db, "SELECT COUNT(*) FROM projections WHERE pipeline_run_id = %s", (PENDING,)) == 0
    assert _scalar(db, "SELECT COUNT(*) FROM projections WHERE pipeline_run_id = %s", (APPROVED,)) == 1


def test_missing_training_history_fails_loudly_at_season_simulation(db):
    _seed_schedule(db)
    _seed_approved_projection(db, player="SEED-APPROVED", game="2026_01_BBB_AAA", week=1)

    with _ingest_mocked(_clean_summary()):
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1
    dead = _dead_letters(db)
    assert [source for source, _ in dead] == ["weekly_refresh/materialize_season_simulation"]
    assert "Refusing to materialize" in dead[0][1]
    # Steps 3 and 4 ran for real before the failure.
    assert _scalar(db, "SELECT COUNT(*) FROM fantasy_player_ids") == len(_PLAYERIDS)
    assert _scalar(db, "SELECT COUNT(*) FROM projections WHERE pipeline_run_id = %s", (PENDING,)) > 0
    # Nothing was promoted: the approved 2026 row is still the only APPROVED row.
    assert _scalar(db, "SELECT COUNT(*) FROM projections WHERE pipeline_run_id = %s", (APPROVED,)) == 1


@pytest.mark.xfail(
    strict=True,
    reason="weekly_refresh ignores the return code of step 4 (scripts/weekly_refresh.py:240); "
    "fixing it changes which rows are promoted, so it is reported, not fixed",
)
def test_stack_materialization_refusal_stops_the_refresh(db, tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "releases" / "current_baseline.json").read_text())
    stacks = manifest["artifacts"]["stacks"]
    kept = [p for p in stacks if "stack_fantasy_ppr_QB_" not in p]
    assert len(kept) == len(stacks) - 1
    manifest["artifacts"]["stacks"] = kept
    copy = tmp_path / "current_baseline.json"
    copy.write_text(json.dumps(manifest))
    monkeypatch.setenv("BASELINE_MANIFEST_PATH", str(copy))

    _seed_schedule(db)
    with _ingest_mocked(_clean_summary()):
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1
    assert [source for source, _ in _dead_letters(db)] == ["weekly_refresh/materialize_stack_projections"]


@pytest.mark.xfail(
    strict=True,
    reason="step 4 upserts over approved rows (scripts/materialize_stack_projections.py:216-229), "
    "so a refresh that later dead-letters still re-tags served rows; reported, not fixed",
)
def test_dead_lettered_refresh_leaves_approved_projection_rows_approved(db):
    key = _first_stack_row(season=2024)
    _execute(
        db,
        """
        INSERT INTO projections (player_id, game_id, season, week, stat, position,
                                 projection, pipeline_run_id, max_train_season)
        VALUES (%s, %s, 2024, %s, 'fantasy_ppr', 'QB', 12.0, %s, 2023)
        """,
        (key["player_id"], key["game_id"], key["week"], STACK_APPROVED),
    )
    _seed_schedule(db)

    with _ingest_mocked(_clean_summary()):
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1  # step 5 has no training history in this database
    run_id = _scalar(
        db,
        "SELECT pipeline_run_id FROM projections "
        "WHERE player_id = %s AND game_id = %s AND stat = 'fantasy_ppr'",
        (key["player_id"], key["game_id"]),
    )
    assert run_id == STACK_APPROVED


def test_acceptance_gate_fails_closed_for_an_unplayed_week_without_depth_charts(db, monkeypatch):
    """Characterization of step 7 on a forward week: it cannot pass on this data.

    The projection service's depth-chart freshness gate raises 503 before any
    accuracy check runs. Step 7 turns that into a dead letter (weekly_refresh.py:306-309).
    """
    import pipeline.db_defaults as db_defaults
    from fastapi import HTTPException

    from scripts.verify_season_acceptance import main as verify_main

    _seed_schedule(db)
    # verify_season_acceptance reads DEFAULT_HOST_DATABASE_URL, not the URL it is given
    # (see scripts/verify_season_acceptance.py:40), so point that default at this database.
    monkeypatch.setattr(db_defaults, "DEFAULT_HOST_DATABASE_URL", db)
    monkeypatch.setattr(sys, "argv", ["verify_season_acceptance.py", "--season", "2026", "--start-week", "3"])
    with pytest.raises(HTTPException) as excinfo:
        verify_main()
    assert excinfo.value.status_code == 503
    assert "Depth chart freshness gate failed for season=2026" in excinfo.value.detail


def test_promote_and_prune_steps_on_the_real_database(db):
    """Steps 8-10 composed from the real functions against seeded pending rows."""
    from scripts.prune_stale_pipeline_runs import (
        check_guard,
        delete_stale_rows,
        load_keep_run_ids,
        vacuum_tables,
    )
    from pipeline.staging_retention import purge_processed_staging, purge_stale_staging

    now = datetime.now(timezone.utc)
    _execute(
        db,
        """
        INSERT INTO season_simulations (season, start_week, player_id, stat, mean, interval_method, pipeline_run_id)
        VALUES (2026, 3, 'SEED-A', 'fantasy_ppr', 10, 'season_simulator_mc', %s),
               (2026, 3, 'SEED-OLD', 'fantasy_ppr', 4, 'season_simulator_mc', 'season_sim_old')
        """,
        (PENDING,),
    )
    _execute(
        db,
        "INSERT INTO season_simulation_weeks (season, start_week, week, player_id, stat, mean, pipeline_run_id) "
        "VALUES (2026, 3, 3, 'SEED-A', 'fantasy_ppr', 10, %s)",
        (PENDING,),
    )
    _execute(
        db,
        "INSERT INTO season_team_wins (season, start_week, team, wins_mean, pipeline_run_id) "
        "VALUES (2026, 3, 'AAA', 9.5, %s)",
        (PENDING,),
    )
    _execute(
        db,
        """
        INSERT INTO projections (player_id, game_id, season, week, stat, position,
                                 projection, pipeline_run_id, max_train_season)
        VALUES ('SEED-A', '2026_03_BBB_AAA', 2026, 3, 'fantasy_ppr', 'QB', 12.0, %s, 2025),
               ('SEED-JUNK', '2025_01_BBB_AAA', 2025, 1, 'fantasy_ppr', 'QB', 1.0, 'junk_run', 2024)
        """,
        (PENDING,),
    )
    _execute(
        db,
        """
        INSERT INTO staging_nflreadpy (source_type, season, raw_data, processed, ingested_at) VALUES
            ('depth_charts',   2026, '{}', true,  %s),
            ('player_stats',   2025, '{}', true,  %s),
            ('injury_reports', 2026, '{}', true,  %s),
            ('rosters',        2026, '{}', true,  %s),
            ('player_stats',   2026, '{}', false, %s)
        """,
        (now, now - timedelta(days=30), now - timedelta(days=30), now - timedelta(days=30), now - timedelta(days=30)),
    )

    with closing(psycopg2.connect(db)) as conn:
        with conn.cursor() as cur:
            promote_run_id(cur, 2026, 3, PENDING, APPROVED)
            # High-volume processed rows go regardless of age; the 14-day sweep takes the
            # old processed injury report. Rosters and unprocessed rows survive both.
            assert purge_processed_staging(cur) == 2
            assert purge_stale_staging(cur) == 1
            keep = load_keep_run_ids()
            assert APPROVED in keep
            assert check_guard(cur, keep) == []
            deleted = delete_stale_rows(cur, keep)
        conn.commit()
    vacuum_tables(db)

    assert _query(db, "SELECT player_id, pipeline_run_id FROM season_simulations ORDER BY 1") == [
        ("SEED-A", APPROVED)
    ]
    assert _scalar(db, "SELECT COUNT(*) FROM season_simulation_weeks WHERE pipeline_run_id = %s", (APPROVED,)) == 1
    assert _scalar(db, "SELECT COUNT(*) FROM season_team_wins WHERE pipeline_run_id = %s", (APPROVED,)) == 1
    assert _query(db, "SELECT pipeline_run_id FROM projections WHERE season = 2026") == [(APPROVED,)]
    assert deleted["projections (season >= 2024)"] == 1  # the junk row only
    assert _scalar(db, "SELECT COUNT(*) FROM staging_nflreadpy") == 2  # rosters + unprocessed
