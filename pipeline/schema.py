"""
Schema migration helpers.

Prefer Alembic (`upgrade_head`) for all environments.
`bootstrap_schema` remains as the empty-database path used by the initial
revision and by local bootstraps when alembic_version is absent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema_ddl import ALL_DDL

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def normalize_dsn(db_url: str) -> str:
    """Convert SQLAlchemy/asyncpg URLs to a psycopg2-compatible DSN."""
    return (
        db_url.replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )


def database_url_from_env() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL)


def bootstrap_schema(conn: Any) -> None:
    """Apply idempotent CREATE TABLE IF NOT EXISTS statements."""
    cur = conn.cursor()
    for ddl in ALL_DDL:
        cur.execute(ddl)
    # Projections column backfills for older local DBs
    for col, typ in (
        ("posterior_samples", "JSONB"),
        ("p25", "FLOAT"),
        ("p75", "FLOAT"),
        ("max_train_season", "INTEGER"),
    ):
        cur.execute(f"ALTER TABLE projections ADD COLUMN IF NOT EXISTS {col} {typ}")
    conn.commit()
    logger.info("Schema bootstrap applied (%d DDL blocks).", len(ALL_DDL))


def ensure_schema(conn: Any) -> None:
    """
    Ensure schema exists on an open connection.

    This is the bootstrap path for empty databases. Prefer `upgrade_head`
    when Alembic is available; callers may invoke both safely.
    """
    bootstrap_schema(conn)


def upgrade_head(db_url: str | None = None) -> None:
    """Run `alembic upgrade head` against the given (or env) database URL."""
    from alembic import command
    from alembic.config import Config

    url = normalize_dsn(db_url or database_url_from_env())
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    logger.info("Alembic upgrade head complete.")


def stamp_head(db_url: str | None = None) -> None:
    """Mark an existing schema as current without re-running DDL."""
    from alembic import command
    from alembic.config import Config

    url = normalize_dsn(db_url or database_url_from_env())
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.stamp(cfg, "head")
    logger.info("Alembic stamp head complete.")
