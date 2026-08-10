"""Alembic migration helpers; Alembic is the sole schema authority."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

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
