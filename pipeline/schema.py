"""Alembic migration helpers; Alembic is the sole schema authority."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Key under which upgrade_head/stamp_head hand an explicit URL to alembic/env.py via Config.attributes.
EXPLICIT_DB_URL = "explicit_db_url"


def normalize_dsn(db_url: str) -> str:
    """Convert SQLAlchemy/asyncpg URLs to a psycopg2-compatible DSN."""
    return (
        db_url.replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )


def database_url_from_env() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL)


def _alembic_config(url: str):
    """Build the repo Alembic config carrying ``url`` both as the ini option and explicitly."""
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    # set_main_option goes through configparser interpolation, so a literal `%`
    # (a URL-encoded password, e.g. `%40`) must be doubled or it raises.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    # alembic/env.py prefers this explicit URL over DATABASE_URL.
    cfg.attributes[EXPLICIT_DB_URL] = url
    return cfg


def upgrade_head(db_url: str | None = None) -> None:
    """Run `alembic upgrade head` against the given (or env) database URL."""
    from alembic import command

    cfg = _alembic_config(normalize_dsn(db_url or database_url_from_env()))
    command.upgrade(cfg, "head")
    logger.info("Alembic upgrade head complete.")


def stamp_head(db_url: str | None = None) -> None:
    """Mark an existing schema as current without re-running DDL."""
    from alembic import command

    cfg = _alembic_config(normalize_dsn(db_url or database_url_from_env()))
    command.stamp(cfg, "head")
    logger.info("Alembic stamp head complete.")
