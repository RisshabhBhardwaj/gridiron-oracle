"""The availability migration (decision 0012 D3, addendum): additive, and it downgrades.

Offline SQL generation needs no database, so this runs in the normal gate. The ORM model
must also carry the column, or code that builds the table from the models would drop it.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]
PREV = "20261114_0027"
REV = "20261114_0028"
NEXT = "20261114_0029"
HEAD = "20261114_0030"


def _cfg() -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return cfg


@contextlib.contextmanager
def _logging_untouched():
    """alembic's env.py calls fileConfig(), which disables every existing logger.

    Left alone it silences later tests' caplog assertions, so put the state back.
    """
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    disabled = {n: lg.disabled for n, lg in logging.root.manager.loggerDict.items() if isinstance(lg, logging.Logger)}
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers
        for name, flag in disabled.items():
            logging.getLogger(name).disabled = flag


def _sql(fn, *args) -> str:
    buf = io.StringIO()
    cfg = _cfg()
    cfg.output_buffer = buf
    with _logging_untouched(), contextlib.redirect_stdout(io.StringIO()):
        fn(cfg, *args, sql=True)
    return buf.getvalue()


def test_single_head_and_chain() -> None:
    script = ScriptDirectory.from_config(_cfg())
    assert script.get_heads() == [HEAD]
    assert script.get_revision(HEAD).down_revision == NEXT
    assert script.get_revision(REV).down_revision == PREV
    assert script.get_revision(NEXT).down_revision == REV


def test_upgrade_sql_is_additive() -> None:
    sql = _sql(command.upgrade, f"{PREV}:{REV}")
    assert "ALTER TABLE projections ADD COLUMN IF NOT EXISTS availability_status TEXT" in sql
    assert "CREATE TABLE IF NOT EXISTS availability_snapshots" in sql
    assert "CREATE TABLE IF NOT EXISTS availability_reports" in sql
    # nullable: history rows stay NULL
    assert "availability_status TEXT NOT NULL" not in sql
    # only the stored values are allowed
    for status in ("Questionable", "Doubtful", "Out", "Reserve", "Suspended", "Inactive"):
        assert f"'{status}'" in sql
    assert "DROP " not in sql.replace("DROP CONSTRAINT IF EXISTS ck_projections_availability_status", "")


def test_downgrade_sql_removes_everything_it_added() -> None:
    sql = _sql(command.downgrade, f"{REV}:{PREV}")
    assert "DROP TABLE IF EXISTS availability_reports" in sql
    assert "DROP TABLE IF EXISTS availability_snapshots" in sql
    assert "DROP CONSTRAINT IF EXISTS ck_projections_availability_status" in sql
    assert "ALTER TABLE projections DROP COLUMN IF EXISTS availability_status" in sql
    # reports go before their snapshots, and the constraint before the column
    assert sql.index("availability_reports") < sql.index("DROP TABLE IF EXISTS availability_snapshots")
    assert sql.index("DROP CONSTRAINT") < sql.index("DROP COLUMN")


def test_orm_projection_has_the_column() -> None:
    from backend.app.models.production import Projection

    assert "availability_status" in Projection.__table__.columns
    assert Projection.__table__.columns["availability_status"].nullable is True


def test_0029_allows_the_unknown_marker_and_nothing_else_new() -> None:
    sql = _sql(command.upgrade, f"{REV}:{NEXT}")
    assert "DROP CONSTRAINT IF EXISTS ck_projections_availability_status" in sql
    for status in ("Questionable", "Doubtful", "Out", "Reserve", "Suspended", "Inactive", "Unknown"):
        assert f"'{status}'" in sql
    assert "DROP TABLE" not in sql and "DROP COLUMN" not in sql


def test_0029_downgrade_clears_unknown_rows_before_tightening_the_constraint() -> None:
    sql = _sql(command.downgrade, f"{NEXT}:{REV}")
    clear = sql.index("UPDATE projections SET availability_status = NULL WHERE availability_status = 'Unknown'")
    add = sql.index("ADD CONSTRAINT ck_projections_availability_status")
    assert clear < add
    assert "'Unknown'" not in sql[add:]


# ── L3: no long ACCESS EXCLUSIVE wait on the served table ─────────────────────────────────
@pytest.mark.parametrize("sql_for", [
    lambda: _sql(command.upgrade, f"{PREV}:{REV}"),
    lambda: _sql(command.upgrade, f"{REV}:{NEXT}"),
    lambda: _sql(command.downgrade, f"{NEXT}:{REV}"),
], ids=["0028-up", "0029-up", "0029-down"])
def test_the_check_constraint_is_added_without_a_long_lock_on_projections(sql_for) -> None:
    sql = sql_for()
    lock_at = sql.index("SET LOCAL lock_timeout = '5s'")
    first_alter = sql.index("ALTER TABLE projections")
    assert lock_at < first_alter  # a waiting ALTER gives up instead of queueing every reader behind it
    add = sql.index("ADD CONSTRAINT ck_projections_availability_status")
    assert "NOT VALID" in sql[add:sql.index(";", add)]  # no full-table scan under ACCESS EXCLUSIVE
    validate = sql.index("VALIDATE CONSTRAINT ck_projections_availability_status")
    assert validate > add  # validation scans under a lock that does not block reads or writes


# ── N4: alembic wraps a run in one transaction, so NOT VALID alone gives no lock relief ────────
@pytest.mark.parametrize("sql_for", [
    lambda: _sql(command.upgrade, f"{PREV}:{REV}"),
    lambda: _sql(command.upgrade, f"{REV}:{NEXT}"),
    lambda: _sql(command.downgrade, f"{NEXT}:{REV}"),
], ids=["0028-up", "0029-up", "0029-down"])
def test_the_validate_runs_after_the_exclusive_lock_was_released_with_its_own_lock_timeout(sql_for) -> None:
    sql = sql_for()
    add = sql.index("ADD CONSTRAINT ck_projections_availability_status")
    validate = sql.index("VALIDATE CONSTRAINT ck_projections_availability_status")
    between = sql[add:validate]
    # ADD CONSTRAINT holds ACCESS EXCLUSIVE until its transaction ends: that transaction must end first.
    assert "COMMIT" in between, "VALIDATE runs in the same transaction as ADD CONSTRAINT: no lock relief"
    after_commit = between[between.index("COMMIT"):]
    # SET LOCAL would vanish with the commit; the validate needs a session-level timeout of its own.
    assert "SET lock_timeout" in after_commit and "SET LOCAL" not in after_commit
    # And the session setting must not leak into the migrations that follow.
    assert "RESET lock_timeout" in sql[validate:]
