"""
Unit tests for pipeline/staging_retention.py's purge_stale_staging().

No live DB / network: following the same mocking convention as
backend/tests/test_prune_stale_pipeline_runs.py (this repo has no existing
DB-cursor test fixture, per that file's own note), this uses plain
unittest.mock.MagicMock to stand in for a psycopg2 cursor. This deliberately
never touches Neon — the DELETE predicate is asserted against the SQL text
and params passed to cur.execute(), never actually run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.staging_retention import (  # noqa: E402
    DEFAULT_RETENTION_DAYS,
    purge_stale_staging,
)


def _make_cursor(rowcount: int) -> MagicMock:
    cur = MagicMock()
    cur.rowcount = rowcount
    return cur


def _normalized_sql(cur: MagicMock) -> str:
    """Collapse whitespace so substring assertions aren't order/indent-fragile."""
    sql = cur.execute.call_args[0][0]
    return " ".join(sql.split())


class TestPurgeStaleStagingPredicate:
    def test_sql_requires_processed_true_not_negated(self) -> None:
        # Asserting the operator, not just the column name — "WHERE processed"
        # would also match a bug like "WHERE NOT processed", which is the
        # inverse (and catastrophic) predicate.
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur)
        sql = _normalized_sql(cur)
        assert "WHERE processed AND" in sql
        assert "NOT processed" not in sql

    def test_sql_excludes_rosters_source_type(self) -> None:
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur)
        sql = _normalized_sql(cur)
        assert "source_type <> 'rosters'" in sql

    def test_sql_deletes_rows_older_than_the_window_not_newer(self) -> None:
        # "ingested_at" in sql would also pass for the inverted (and
        # catastrophic) "ingested_at >" — assert the actual comparison
        # operator, which must point at OLDER rows being deleted.
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur)
        sql = _normalized_sql(cur)
        assert "DELETE FROM staging_nflreadpy" in sql
        assert "ingested_at <" in sql
        assert "ingested_at >" not in sql

    def test_default_retention_renders_as_literal_14_day_interval(self) -> None:
        # Matches the brief's literal SQL byte-for-byte for the default case:
        # DELETE FROM staging_nflreadpy WHERE processed AND source_type <>
        # 'rosters' AND ingested_at < now() - interval '14 days';
        assert DEFAULT_RETENTION_DAYS == 14
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur)
        sql = _normalized_sql(cur)
        assert "interval '14 days'" in sql
        # No bind params for retention_days: it's coerced to int() and
        # interpolated as a literal, not passed as a %s value (see the
        # comment in pipeline/staging_retention.py for why) — execute()
        # is called with a single positional arg (the SQL string only).
        assert len(cur.execute.call_args[0]) == 1

    def test_custom_retention_days_is_interpolated_as_literal(self) -> None:
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur, retention_days=30)
        sql = _normalized_sql(cur)
        assert "interval '30 days'" in sql

    def test_retention_days_is_coerced_to_int_before_interpolation(self) -> None:
        # Guards the injection-safety justification for interpolating
        # rather than binding: a non-int retention_days must still be
        # coerced, never spliced in raw.
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur, retention_days="7")
        sql = _normalized_sql(cur)
        assert "interval '7 days'" in sql

    def test_default_retention_sql_matches_exact_normalized_text(self) -> None:
        # A byte-for-byte pin (after whitespace normalization) so a mutation
        # like swapping the final AND for OR fails a test outright, instead
        # of merely failing to be caught by a substring check.
        cur = _make_cursor(rowcount=0)
        purge_stale_staging(cur)
        sql = _normalized_sql(cur)
        assert sql == (
            "DELETE FROM staging_nflreadpy "
            "WHERE processed "
            "AND source_type <> 'rosters' "
            "AND ingested_at < now() - interval '14 days'"
        )


class TestPurgeStaleStagingRetentionDaysGuard:
    def test_negative_retention_days_raises(self) -> None:
        # This is the core bug this guard exists to catch: interval '-1
        # days' flips `ingested_at < now() - interval '-1 days'` into
        # `ingested_at < now() + interval '1 days'`, which matches
        # essentially every row instead of only stale ones.
        cur = _make_cursor(rowcount=0)
        with pytest.raises(ValueError):
            purge_stale_staging(cur, retention_days=-1)
        cur.execute.assert_not_called()

    def test_zero_retention_days_raises(self) -> None:
        cur = _make_cursor(rowcount=0)
        with pytest.raises(ValueError):
            purge_stale_staging(cur, retention_days=0)
        cur.execute.assert_not_called()

    def test_non_numeric_retention_days_raises(self) -> None:
        cur = _make_cursor(rowcount=0)
        with pytest.raises(ValueError):
            purge_stale_staging(cur, retention_days="not-a-number")
        cur.execute.assert_not_called()

    def test_negative_numeric_string_retention_days_raises(self) -> None:
        cur = _make_cursor(rowcount=0)
        with pytest.raises(ValueError):
            purge_stale_staging(cur, retention_days="-5")
        cur.execute.assert_not_called()


class TestPurgeStaleStagingLogging:
    def test_string_retention_days_does_not_raise_typeerror_in_log_call(
        self, caplog
    ) -> None:
        # Reproduces the bug: logger.info(..., "%d"..., retention_days) with
        # an uncoerced string retention_days raises TypeError inside the
        # logging call ("%d format: a real number is required, not str").
        # Default logging error-handling swallows that TypeError silently
        # under a plain `pytest` run, which is why a naive test wouldn't
        # catch it — it only surfaces with --log-cli-level=INFO enabled, or
        # by asserting caplog captured the record (as done here).
        import logging

        cur = _make_cursor(rowcount=3)
        with caplog.at_level(logging.INFO, logger="pipeline.staging_retention"):
            result = purge_stale_staging(cur, retention_days="7")
        assert result == 3
        assert any("deleted 3 row(s)" in r.message for r in caplog.records)


class TestPurgeStaleStagingReturnValue:
    def test_returns_deleted_row_count(self) -> None:
        cur = _make_cursor(rowcount=42)
        result = purge_stale_staging(cur)
        assert result == 42

    def test_zero_rows_deleted_returns_zero(self) -> None:
        cur = _make_cursor(rowcount=0)
        result = purge_stale_staging(cur)
        assert result == 0


class TestPurgeStaleStagingUsesGivenCursor(object):
    def test_executes_exactly_once_on_the_passed_cursor(self) -> None:
        cur = _make_cursor(rowcount=5)
        result = purge_stale_staging(cur)
        assert result == 5
        cur.execute.assert_called_once()
