"""
Unit tests for scripts/weekly_refresh.py sequencer.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch
import pytest

from pipeline.orchestrator import Orchestrator, OrchestratorSummary, SeasonResult, StepResult
from scripts.weekly_refresh import (
    WEEKLY_SOURCES,
    promote_run_id,
    resolve_season_and_week,
    run_weekly_refresh,
)

# Every side effect that run_weekly_refresh reaches after step 2. Patched so the
# tests stop at step 2 or run through step 10 without touching a database.
_PATCHED_STEPS = {
    "dead_letter": "scripts.weekly_refresh._write_dead_letter",
    "upsert_playerids": "scripts.weekly_refresh.upsert_playerids",
    "stack": "scripts.materialize_stack_projections.main",
    "simulation": "scripts.materialize_season_simulation.materialize",
    "team_game": "scripts.materialize_team_game_predictions.materialize",
    "acceptance": "scripts.verify_season_acceptance.main",
    "purge_processed": "scripts.weekly_refresh.purge_processed_staging",
    "purge_stale": "scripts.weekly_refresh.purge_stale_staging",
    "keep_ids": "scripts.weekly_refresh.load_keep_run_ids",
    "guard": "scripts.weekly_refresh.check_guard",
    "delete_stale": "scripts.weekly_refresh.delete_stale_rows",
    "vacuum": "scripts.weekly_refresh.vacuum_tables",
    "db_size": "scripts.weekly_refresh.log_database_size",
}


def _summary(*steps: StepResult) -> OrchestratorSummary:
    """A genuine OrchestratorSummary for one season, as Orchestrator.run() returns."""
    return OrchestratorSummary(
        seasons=[2026],
        dry_run=False,
        season_results=[SeasonResult(season=2026, dry_run=False, steps=list(steps))],
    )


@contextmanager
def _weekly_refresh_with_orchestrator(summary: OrchestratorSummary):
    """Patch the world around run_weekly_refresh. Orchestrator.run returns `summary`."""
    with ExitStack() as stack:
        connect = stack.enter_context(patch("psycopg2.connect"))
        cur = MagicMock()
        cur.fetchone.return_value = (2,)  # completed week 2 -> start_week 3
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
        stack.enter_context(patch.object(Orchestrator, "run", return_value=summary))
        mocks = {name: stack.enter_context(patch(target)) for name, target in _PATCHED_STEPS.items()}
        mocks["upsert_playerids"].return_value = 5
        mocks["stack"].return_value = 0
        mocks["acceptance"].return_value = 0
        mocks["purge_processed"].return_value = 0
        mocks["purge_stale"].return_value = 0
        mocks["keep_ids"].return_value = set()
        mocks["guard"].return_value = []
        mocks["delete_stale"].return_value = {}
        yield mocks


def test_resolve_season_and_week():
    cur = MagicMock()
    cur.fetchone.return_value = (4,)
    season, completed_week, start_week = resolve_season_and_week(cur, 2026)
    assert season == 2026
    assert completed_week == 4
    assert start_week == 5


def test_resolve_season_and_week_none():
    cur = MagicMock()
    cur.fetchone.return_value = None
    season, completed_week, start_week = resolve_season_and_week(cur, 2026)
    assert season == 2026
    assert completed_week == 0
    assert start_week == 1


def test_promote_run_id_only_relabels_pending_rows():
    """Promotion is a relabel: no DELETE, one UPDATE per season table, projections and team games."""
    cur = MagicMock()
    promote_run_id(cur, 2026, 5, "pending_id", "approved_id")
    statements = [call.args[0] for call in cur.execute.call_args_list]
    assert not any("DELETE" in sql for sql in statements)
    assert len(statements) == 5  # 3 season tables + projections + team_game_predictions
    assert all("UPDATE" in sql for sql in statements)
    assert all(call.args[1][0] == "approved_id" for call in cur.execute.call_args_list)


def test_dry_run_mode():
    with patch("psycopg2.connect") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (2,)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur

        rc = run_weekly_refresh("postgres://dummy", season=2026, dry_run=True)
        assert rc == 0


_CLEAN_INGEST = StepResult(name="ingest", elapsed_s=1.0, ok=True, rows_out=100)
_CLEAN_NORMALIZE = StepResult(name="normalize", elapsed_s=1.0, ok=True, rows_out=50)
_CLEAN_FEATURES = StepResult(name="feature_engineer", elapsed_s=1.0, ok=True, rows_out=500)


def test_step2_clean_orchestrator_run_reaches_step3_without_dead_letter():
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert m["dead_letter"].call_args_list == [], m["dead_letter"].call_args_list
    m["upsert_playerids"].assert_called_once()
    assert rc == 0


def test_step2_failed_orchestrator_step_dead_letters_and_stops_before_step3():
    failed = StepResult(
        name="normalize", elapsed_s=1.0, ok=False, error="normalize exploded: bad row",
    )
    summary = _summary(_CLEAN_INGEST, failed)
    with _weekly_refresh_with_orchestrator(summary) as m:
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert rc == 1
    m["dead_letter"].assert_called_once()
    _, source, error, season = m["dead_letter"].call_args.args
    assert source == "weekly_refresh/orchestrator_ingest"
    assert season == 2026
    assert "normalize exploded: bad row" in error
    m["upsert_playerids"].assert_not_called()
