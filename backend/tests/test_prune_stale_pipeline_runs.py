"""
Unit tests for scripts/prune_stale_pipeline_runs.py's guard logic — the
"would zero out a currently-served (season, start_week)" refusal.

No live DB / network: this repo has no existing convention for mocking a
psycopg2 cursor for a script-level unit test (grepped backend/tests/ for
MagicMock/unittest.mock usage — none of the hits touch a DB cursor), so
this uses plain unittest.mock, scripted to answer each cur.execute() /
cur.fetchall() call in the exact sequence check_guard() makes them in.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prune_stale_pipeline_runs import (  # noqa: E402
    SEASON_START_WEEK_TABLES,
    check_guard,
    load_keep_run_ids,
)


def _make_cursor(kept_pairs: list[tuple[int, int]], violations_by_table: dict[str, list[tuple[int, int]]]):
    """
    A fake cursor whose fetchall() results are queued in the same order
    check_guard() issues queries: first the season_simulations kept-pairs
    query, then one query per SEASON_START_WEEK_TABLES entry.
    """
    cur = MagicMock()
    responses = [kept_pairs] + [
        violations_by_table.get(table, []) for table in SEASON_START_WEEK_TABLES
    ]
    cur.fetchall.side_effect = responses
    return cur


class TestGuardPasses:
    def test_no_kept_pairs_means_nothing_to_protect(self) -> None:
        cur = _make_cursor(kept_pairs=[], violations_by_table={})
        violations = check_guard(cur, keep_ids=["run-a"])
        assert violations == []

    def test_all_tables_cover_the_kept_pair_no_violation(self) -> None:
        cur = _make_cursor(
            kept_pairs=[(2026, 5)],
            violations_by_table={t: [] for t in SEASON_START_WEEK_TABLES},
        )
        violations = check_guard(cur, keep_ids=["run-a"])
        assert violations == []


class TestGuardCatchesDivergence:
    def test_weeks_table_missing_kept_coverage_is_flagged(self) -> None:
        # season_simulations says (2026, 5) is served by run-a, but
        # season_simulation_weeks has no row under run-a for that pair —
        # exactly the manifest/data divergence scenario the guard exists
        # to catch.
        cur = _make_cursor(
            kept_pairs=[(2026, 5)],
            violations_by_table={
                "season_simulation_weeks": [(2026, 5)],
                "season_simulations": [],
                "season_team_wins": [],
            },
        )
        violations = check_guard(cur, keep_ids=["run-a"])
        assert len(violations) == 1
        assert violations[0]["table"] == "season_simulation_weeks"
        assert violations[0]["season"] == 2026
        assert violations[0]["start_week"] == 5

    def test_multiple_tables_can_each_be_flagged(self) -> None:
        cur = _make_cursor(
            kept_pairs=[(2026, 5)],
            violations_by_table={
                "season_simulation_weeks": [(2026, 5)],
                "season_simulations": [],
                "season_team_wins": [(2026, 5)],
            },
        )
        violations = check_guard(cur, keep_ids=["run-a"])
        tables_flagged = {v["table"] for v in violations}
        assert tables_flagged == {"season_simulation_weeks", "season_team_wins"}


class TestLoadKeepRunIds:
    def test_missing_manifest_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_keep_run_ids(tmp_path / "does_not_exist.json")

    def test_empty_approved_list_raises(self, tmp_path) -> None:
        manifest = tmp_path / "baseline.json"
        manifest.write_text('{"projection_policy": {"approved_pipeline_run_ids": []}}')
        with pytest.raises(ValueError):
            load_keep_run_ids(manifest)

    def test_reads_approved_ids(self, tmp_path) -> None:
        manifest = tmp_path / "baseline.json"
        manifest.write_text(
            '{"projection_policy": {"approved_pipeline_run_ids": ["a", "b"]}}'
        )
        assert load_keep_run_ids(manifest) == ["a", "b"]
