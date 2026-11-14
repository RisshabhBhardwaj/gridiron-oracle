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
    assert script.get_heads() == [REV]
    assert script.get_revision(REV).down_revision == PREV


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
