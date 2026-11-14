"""Migration 0030: the archive table behind ``--replace-week`` (review M1). Additive and reversible."""

from __future__ import annotations

from alembic import command

from backend.tests.test_availability_migration import _sql

PREV, REV = "20261114_0029", "20261114_0030"


def test_upgrade_creates_an_empty_archive_shaped_like_projections() -> None:
    sql = _sql(command.upgrade, f"{PREV}:{REV}")
    assert "CREATE TABLE IF NOT EXISTS projections_replaced (LIKE projections)" in sql
    assert "archived_at" in sql
    assert "CREATE INDEX IF NOT EXISTS ix_projections_replaced_week" in sql
    assert "SET LOCAL lock_timeout" in sql
    assert "DROP " not in sql and "ALTER TABLE projections " not in sql  # the served table is untouched


def test_downgrade_drops_only_the_archive() -> None:
    sql = _sql(command.downgrade, f"{REV}:{PREV}")
    assert "DROP TABLE IF EXISTS projections_replaced" in sql
    assert "projections;" not in sql.replace("projections_replaced", "")
