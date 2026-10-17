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
        orchestrator_run = stack.enter_context(patch.object(Orchestrator, "run", return_value=summary))
        mocks = {name: stack.enter_context(patch(target)) for name, target in _PATCHED_STEPS.items()}
        mocks["orchestrator_run"] = orchestrator_run
        mocks["cursor"] = cur
        mocks["upsert_playerids"].return_value = 5
        mocks["stack"].return_value = 0
        mocks["acceptance"].return_value = 0
        mocks["purge_processed"].return_value = 0
        mocks["purge_stale"].return_value = 0
        mocks["keep_ids"].return_value = {"weekly_auto_2026"}
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


def test_approved_run_ids_resolve_against_the_repo_not_the_working_directory(monkeypatch, tmp_path):
    """A cron job that starts outside the repo must still see the manifest's approved ids."""
    from pipeline.run_guard import approved_pipeline_run_ids

    monkeypatch.chdir(tmp_path)
    assert "weekly_auto_2026" in approved_pipeline_run_ids()


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


# ── Season is derived from ml/season_constants.py (H-6) ──────────────────────

@pytest.fixture
def season_constants_at(monkeypatch):
    from ml import season_constants

    def _set(last_complete: int, current: int) -> None:
        monkeypatch.setattr(season_constants, "LAST_COMPLETE_SEASON", last_complete)
        monkeypatch.setattr(season_constants, "CURRENT_SEASON", current)

    return _set


def test_current_season_is_read_from_the_season_constants(season_constants_at):
    from scripts.weekly_refresh import current_season

    season_constants_at(last_complete=2025, current=2026)
    assert current_season() == 2026
    season_constants_at(last_complete=2026, current=2027)
    assert current_season() == 2027


def test_current_season_refuses_constants_that_disagree(season_constants_at):
    from scripts.weekly_refresh import current_season

    season_constants_at(last_complete=2026, current=2026)
    with pytest.raises(RuntimeError, match=r"CURRENT_SEASON=2026 must equal LAST_COMPLETE_SEASON\+1"):
        current_season()


def test_resolve_season_and_week_defaults_to_the_derived_season(season_constants_at):
    season_constants_at(last_complete=2026, current=2027)
    cur = MagicMock()
    cur.fetchone.return_value = (0,)
    season, completed_week, start_week = resolve_season_and_week(cur)
    assert (season, completed_week, start_week) == (2027, 0, 1)
    assert cur.execute.call_args.args[1] == (2027,)


def test_weekly_run_for_2027_orchestrates_and_promotes_2027(season_constants_at):
    season_constants_at(last_complete=2026, current=2027)
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        m["keep_ids"].return_value = {"weekly_auto_2027"}
        rc = run_weekly_refresh("postgres://dummy")
    assert rc == 0
    assert m["orchestrator_run"].call_args.kwargs["seasons"] == [2027]
    promoted = [c.args[1] for c in m["cursor"].execute.call_args_list if "UPDATE" in c.args[0]]
    assert promoted and all(params[0] == "weekly_auto_2027" for params in promoted)


def test_season_missing_from_the_manifest_fails_before_any_write(season_constants_at):
    season_constants_at(last_complete=2026, current=2027)
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        m["keep_ids"].return_value = {"weekly_auto_2026"}
        rc = run_weekly_refresh("postgres://dummy")
    assert rc == 1
    m["orchestrator_run"].assert_not_called()
    m["stack"].assert_not_called()
    m["upsert_playerids"].assert_not_called()
    _, source, error, season = m["dead_letter"].call_args.args
    assert source == "weekly_refresh/approval_check"
    assert "weekly_auto_2027" in error
    assert season == 2027


def test_prune_is_refused_when_the_promoted_id_is_not_in_the_keep_set():
    """Step 9 re-checks the keep set. If the promoted id is missing, nothing is deleted."""
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        m["keep_ids"].side_effect = [{"weekly_auto_2026"}, set()]
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert rc == 1
    m["delete_stale"].assert_not_called()
    m["purge_processed"].assert_not_called()
    _, source, _, _ = m["dead_letter"].call_args.args
    assert source == "weekly_refresh/retention_and_pruning"


def test_step2_failed_run_level_enrichment_fails_the_weekly_run():
    """Elo and PBP run once for the whole run, not per season. Their failure must still fail step 2."""
    summary = OrchestratorSummary(
        seasons=[2026],
        dry_run=False,
        season_results=[SeasonResult(season=2026, dry_run=False, steps=[_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES])],
        global_steps=[StepResult(name="enrich_elo", elapsed_s=1.0, ok=False, error="elo broke")],
    )
    with _weekly_refresh_with_orchestrator(summary) as m:
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert rc == 1
    _, source, error, season = m["dead_letter"].call_args.args
    assert source == "weekly_refresh/orchestrator_ingest"
    assert "run-level step=enrich_elo: elo broke" in error
    assert season == 2026
    m["upsert_playerids"].assert_not_called()


# ── Step 7 routes forward weeks to the structural gate (C-1) ─────────────────

def _gate_argv(m):
    """The argv step 7 handed to verify_season_acceptance.main."""
    seen: dict = {}

    def spy():
        import sys

        seen["argv"] = list(sys.argv)
        return 0

    m["acceptance"].side_effect = spy
    return seen


def test_step7_forward_week_runs_the_structural_gate_on_the_pending_run():
    """No realized game_logs in the window: the statistical gate could never pass."""
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        seen = _gate_argv(m)
        # resolve_season_and_week reads completed week 2, then the step-7
        # realized check finds nothing from week 3 on.
        m["cursor"].fetchone.side_effect = [(2,), (False,)]
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert rc == 0
    assert "--structural-only" in seen["argv"]
    assert "weekly_auto_2026_pending" in seen["argv"]
    assert "--start-week" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--start-week") + 1] == "3"


def test_step7_window_with_realized_points_keeps_the_statistical_gate():
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        seen = _gate_argv(m)
        m["cursor"].fetchone.side_effect = [(2,), (True,)]
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert rc == 0
    assert "--structural-only" not in seen["argv"]


def test_step7_failed_structural_gate_aborts_before_promotion():
    summary = _summary(_CLEAN_INGEST, _CLEAN_NORMALIZE, _CLEAN_FEATURES)
    with _weekly_refresh_with_orchestrator(summary) as m:
        m["acceptance"].return_value = 1
        m["cursor"].fetchone.side_effect = [(2,), (False,)]
        rc = run_weekly_refresh("postgres://dummy", season=2026)
    assert rc == 1
    _, source, error, season = m["dead_letter"].call_args.args
    assert source == "weekly_refresh/verify_acceptance"
    assert "non-zero exit code: 1" in error
    assert season == 2026
