"""
Unit tests for scripts/weekly_refresh.py sequencer.
"""

from unittest.mock import MagicMock, patch
import pytest

from scripts.weekly_refresh import (
    WEEKLY_SOURCES,
    promote_run_id,
    resolve_season_and_week,
    run_weekly_refresh,
)


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


def test_promote_run_id():
    cur = MagicMock()
    promote_run_id(cur, 2026, 5, "pending_id", "approved_id")
    # Should execute DELETE and UPDATE on each table
    assert cur.execute.call_count >= 8


def test_dry_run_mode():
    with patch("psycopg2.connect") as mock_conn:
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (2,)
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cur

        rc = run_weekly_refresh("postgres://dummy", season=2026, dry_run=True)
        assert rc == 0
