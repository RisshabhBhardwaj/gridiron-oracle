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
import types
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import pytest
from fastapi import HTTPException

from backend.app.api import team_game as team_game_api
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
    upgrade_head(url)
    try:
        yield url
    finally:
        admin.cursor().execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


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


@contextmanager
def _season_sim_pass():
    """Step 5 succeeds without fitting models, so the run can reach step 8."""
    with patch("scripts.materialize_season_simulation.materialize") as sim:
        yield sim


@contextmanager
def _acceptance_pass():
    """Step 7 succeeds without reading realized points."""
    with patch("scripts.verify_season_acceptance.main", return_value=0) as acceptance:
        yield acceptance


def _team_game_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic training and forward frames for step 6.

    The real Ridge fit and the real team_game_predictions write run on these;
    only the frame builders (database reads) and the walk-forward residual are replaced.
    """
    from ml.team_game_model import FEATURE_COLS

    rng = np.random.default_rng(0)
    n = 40
    train = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    train["points"] = rng.normal(24.0, 3.0, n)
    train["total_yards"] = rng.normal(330.0, 40.0, n)
    train["pass_rate"] = rng.uniform(0.4, 0.7, n)
    train["season"] = 2025
    forward_rows = []
    for team, opponent, is_home in (("AAA", "BBB", 1), ("BBB", "AAA", 0)):
        row = {c: 0.0 for c in FEATURE_COLS}
        row.update(
            game_id="2026_03_BBB_AAA", team=team, opponent=opponent,
            season=2026, week=3, is_home=is_home,
        )
        forward_rows.append(row)
    return train, pd.DataFrame(forward_rows)


@contextmanager
def _team_game_step_runs():
    """Step 6 runs for real on the synthetic frames."""
    train, forward = _team_game_frames()
    with patch(
        "scripts.materialize_team_game_predictions.build_team_game_frame", return_value=train
    ), patch(
        "scripts.materialize_team_game_predictions.build_team_game_forward_frame", return_value=forward
    ), patch(
        "scripts.materialize_team_game_predictions.compute_oof_residual_std", return_value=10.0
    ):
        yield


def _seed_team_games(url: str, run_id: str, points: tuple[float, float]) -> None:
    """Week 3 team-game rows for 2026_03_BBB_AAA, tagged with run_id."""
    _execute(
        url,
        """
        INSERT INTO team_game_predictions
            (game_id, team, opponent, season, week, is_home, points, yards, pass_rate,
             win_probability, model_run_id)
        VALUES ('2026_03_BBB_AAA', 'AAA', 'BBB', 2026, 3, 1, %s, 330.0, 0.55, 0.6, %s),
               ('2026_03_BBB_AAA', 'BBB', 'AAA', 2026, 3, 0, %s, 320.0, 0.50, 0.4, %s)
        """,
        (points[0], run_id, points[1], run_id),
    )


def _served_team_games(url: str, monkeypatch) -> list[tuple[str, str, float]]:
    """(team, model_run_id, points) for week 3, read through the team-game API endpoint."""
    monkeypatch.setattr(team_game_api.settings, "database_url", url)
    try:
        response = team_game_api.team_game_week(2026, 3)
    except HTTPException as exc:
        if exc.status_code == 404:
            return []
        raise
    return [(g.team, g.model_run_id, g.points) for g in response.games]


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
    _seed_approved_projection(db, player="SEED-APPROVED", game="2026_01_BBB_AAA", week=1)
    with _ingest_mocked(_clean_summary()), _season_sim_pass() as sim, _team_game_step_runs(), _acceptance_pass() as acceptance:
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1
    assert [source for source, _ in _dead_letters(db)] == ["weekly_refresh/materialize_stack_projections"]
    sim.assert_not_called()
    acceptance.assert_not_called()
    # Step 8 would delete every non-pending 2026 projection; the refusal must stop it first.
    assert _query(db, "SELECT pipeline_run_id, projection FROM projections WHERE player_id = 'SEED-APPROVED'") == [
        (APPROVED, 12.0)
    ]


def test_successful_run_keeps_approved_2024_and_2026_projections(db):
    """Steps 4, 8, 9 and 10 run for real on a successful refresh.

    Step 4 writes the 2021-2025 stack artifacts. Step 9 prunes every projection
    row whose run id is not in the keep set. Both an approved 2024 stack row and
    an approved 2026 row must survive the whole run unchanged.
    """
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
    _seed_approved_projection(db, player="SEED-APPROVED", game="2026_01_BBB_AAA", week=1)

    with _ingest_mocked(_clean_summary()), _season_sim_pass(), _team_game_step_runs(), _acceptance_pass():
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 0
    assert _query(db, "SELECT pipeline_run_id, projection FROM projections WHERE player_id = 'SEED-APPROVED'") == [
        (APPROVED, 12.0)
    ]
    assert _query(
        db,
        "SELECT pipeline_run_id, projection FROM projections "
        "WHERE player_id = %s AND game_id = %s AND stat = 'fantasy_ppr'",
        (key["player_id"], key["game_id"]),
    ) == [(STACK_APPROVED, 12.0)]


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
    from scripts.verify_season_acceptance import main as verify_main

    _seed_schedule(db)
    monkeypatch.setattr(
        sys, "argv",
        ["verify_season_acceptance.py", "--season", "2026", "--start-week", "3", "--database-url", db],
    )
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


def test_unmet_season_acceptance_returns_nonzero(db, tmp_path, monkeypatch):
    """acceptance.met == False must fail the verifier (scripts/verify_season_acceptance.py)."""
    import pipeline.db_defaults as db_defaults
    from scripts.verify_season_acceptance import main as verify_main

    # Realized points equal the player index, every projection band is [0, 0.5]:
    # rank correlation is perfect but interval coverage is 1/30, outside 75-85%.
    _seed_schedule(db)
    _execute(
        db,
        "INSERT INTO players (id, full_name, position, team) "
        "SELECT 'P' || i, 'Player ' || i, 'WR', 'AAA' FROM generate_series(0, 29) AS i",
    )
    _execute(
        db,
        "INSERT INTO game_logs (player_id, game_id, season, week, fantasy_points_ppr) "
        "SELECT 'P' || i, '2026_03_BBB_AAA', 2026, 3, i FROM generate_series(0, 29) AS i",
    )
    projected = [
        {
            "player_id": f"P{i}", "player_name": f"Player {i}", "position": "WR", "degraded": False,
            "fantasy_ppr": {"mean": float(i), "p10": 0.0, "p90": 0.5},
        }
        for i in range(30)
    ]
    monkeypatch.setattr(db_defaults, "DEFAULT_HOST_DATABASE_URL", db)
    monkeypatch.setattr(sys, "argv", [
        "verify_season_acceptance.py", "--season", "2026", "--start-week", "3",
        "--out", str(tmp_path / "season_acceptance.json"),
    ])
    with patch("backend.app.services.projection.ProjectionService") as service:
        service.return_value.get_season_projections.return_value = projected
        rc = verify_main()

    assert rc == 1


def test_team_game_api_serves_only_approved_rows(db, monkeypatch):
    _seed_schedule(db)
    _seed_team_games(db, APPROVED, points=(24.0, 21.0))
    _execute(
        db,
        "UPDATE team_game_predictions SET model_run_id = %s WHERE team = 'BBB'",
        (PENDING,),
    )

    assert _served_team_games(db, monkeypatch) == [("AAA", APPROVED, 24.0)]


def test_failed_run_leaves_api_visible_team_game_rows_unchanged(db, monkeypatch):
    """Step 6 must not overwrite approved team-game rows, and a failed run must not promote anything."""
    _seed_schedule(db)
    _seed_team_games(db, APPROVED, points=(24.0, 21.0))
    before = _served_team_games(db, monkeypatch)
    assert before == [("AAA", APPROVED, 24.0), ("BBB", APPROVED, 21.0)]

    # Step 7 runs for real and fails on the unplayed week's depth-chart gate.
    with _ingest_mocked(_clean_summary()), _season_sim_pass(), _team_game_step_runs():
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1
    assert [source for source, _ in _dead_letters(db)] == ["weekly_refresh/verify_acceptance"]
    assert _served_team_games(db, monkeypatch) == before


def test_successful_run_promotes_team_game_rows_for_the_api(db, monkeypatch):
    _seed_schedule(db)

    with _ingest_mocked(_clean_summary()), _season_sim_pass(), _team_game_step_runs(), _acceptance_pass():
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 0
    served = _served_team_games(db, monkeypatch)
    assert [team for team, _, _ in served] == ["AAA", "BBB"]
    assert {run_id for _, run_id, _ in served} == {APPROVED}


def test_step7_reads_the_refresh_database_url_not_the_host_default(db, monkeypatch):
    """Step 7 must use the DSN the refresh was given, not the host default (scripts/verify_season_acceptance.py:40,55)."""
    import pipeline.db_defaults as db_defaults

    _seed_schedule(db)
    monkeypatch.setattr(
        db_defaults, "DEFAULT_HOST_DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/absent"
    )
    with _ingest_mocked(_clean_summary()), _season_sim_pass(), _team_game_step_runs():
        rc = run_weekly_refresh(db, season=2026)

    assert rc == 1
    dead = _dead_letters(db)
    assert [source for source, _ in dead] == ["weekly_refresh/verify_acceptance"]
    assert "Depth chart freshness gate failed for season=2026" in dead[0][1]


def test_upgrade_head_explicit_url_wins_over_database_url_env(monkeypatch):
    """An explicit URL passed to upgrade_head must beat DATABASE_URL (alembic/env.py:22-30)."""
    admin_url = _admin_url()
    admin = psycopg2.connect(admin_url, connect_timeout=3)
    admin.autocommit = True
    name = f"gor_refresh_it_{uuid.uuid4().hex[:10]}"
    admin.cursor().execute(f'CREATE DATABASE "{name}" TEMPLATE template0')
    url = _with_dbname(admin_url, name)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/absent")
    try:
        upgrade_head(url)
        assert _scalar(url, "SELECT COUNT(*) FROM alembic_version") == 1
    finally:
        admin.cursor().execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def test_season_simulation_write_never_changes_approved_rows(db, monkeypatch):
    """Step 5 (real materialize() and real upserts) must not overwrite rows served by an approved run.

    Only the model and roster reads are replaced. Each of the three season tables has an
    approved row on key p1 / AAA; the pending run must leave it alone and insert p2 / BBB.
    """
    import scripts.materialize_season_simulation as sim_mod

    _execute(
        db,
        "INSERT INTO season_simulations (season, start_week, player_id, stat, mean, interval_method, pipeline_run_id) "
        "VALUES (2026, 3, 'p1', 'fantasy_ppr', 10.0, 'season_simulator_mc', %s)",
        (APPROVED,),
    )
    _execute(
        db,
        "INSERT INTO season_simulation_weeks (season, start_week, week, player_id, stat, mean, pipeline_run_id) "
        "VALUES (2026, 3, 3, 'p1', 'fantasy_ppr', 10.0, %s)",
        (APPROVED,),
    )
    _execute(
        db,
        "INSERT INTO season_team_wins (season, start_week, team, wins_mean, pipeline_run_id) "
        "VALUES (2026, 3, 'AAA', 9.5, %s)",
        (APPROVED,),
    )

    def _player(player_id: str, name: str) -> dict:
        return {
            "player_id": player_id, "player_name": name, "position": "WR", "team": "AAA",
            "prior_snap_share": 0.5, "team_games_played": 3, "depth_rank": 1,
            "prior_active_games": 3, "prior_games": 3,
        }

    class _FakeResult:
        player_season_totals = {
            "p1": {"fantasy_ppr": {"mean": 99.0, "p10": 90.0, "p50": 99.0, "p90": 108.0}},
            "p2": {"fantasy_ppr": {"mean": 12.0, "p10": 6.0, "p50": 12.0, "p90": 18.0}},
        }
        week_by_week = [
            {"week": 3, "player_id": "p1", "stat": "fantasy_ppr", "mean": 99.0, "p10": 90.0, "p50": 99.0, "p90": 108.0},
            {"week": 3, "player_id": "p2", "stat": "fantasy_ppr", "mean": 12.0, "p10": 6.0, "p50": 12.0, "p90": 18.0},
        ]
        team_win_totals = {
            "AAA": {"wins_mean": 1.0, "wins_p10": 0.0, "wins_p90": 2.0},
            "BBB": {"wins_mean": 7.0, "wins_p10": 6.0, "wins_p90": 8.0},
        }

    class _FakeSimulator:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return _FakeResult()

    fake_playing_time = types.ModuleType("ml.playing_time")
    fake_playing_time.p_active_from_priors = lambda *a, **k: 0.9
    monkeypatch.setitem(sys.modules, "ml.playing_time", fake_playing_time)
    monkeypatch.setattr(sim_mod, "assert_training_window_present", lambda *a, **k: None)
    monkeypatch.setattr(sim_mod, "_load_roster", lambda *a, **k: [_player("p1", "P One"), _player("p2", "P Two")])
    monkeypatch.setattr(sim_mod, "_load_prior_game_rows", lambda *a, **k: [])
    monkeypatch.setattr(sim_mod, "_load_schedule_gate", lambda *a, **k: object())
    monkeypatch.setattr(sim_mod, "SeasonSimulator", _FakeSimulator)

    sim_mod.materialize(
        season=2026, start_week=3, end_week=18, n_simulations=1, positions=["WR"],
        database_url=db, pipeline_run_id=PENDING,
    )

    assert _query(db, "SELECT player_id, mean, pipeline_run_id FROM season_simulations ORDER BY player_id") == [
        ("p1", 10.0, APPROVED),
        ("p2", 12.0, PENDING),
    ]
    assert _query(
        db, "SELECT player_id, week, mean, pipeline_run_id FROM season_simulation_weeks ORDER BY player_id"
    ) == [("p1", 3, 10.0, APPROVED), ("p2", 3, 12.0, PENDING)]
    assert _query(db, "SELECT team, wins_mean, pipeline_run_id FROM season_team_wins ORDER BY team") == [
        ("AAA", 9.5, APPROVED),
        ("BBB", 7.0, PENDING),
    ]
