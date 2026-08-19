"""Database-backed schema authority checks (C-12 / NEW-04)."""

from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.integration


def test_feature_matrix_orm_matches_live_database() -> None:
    """ORM and Postgres must agree in both directions, not merely dataclass→ORM."""
    import psycopg2

    from backend.app.models.production import FeatureMatrix

    dsn = os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle")
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
    try:
        conn = psycopg2.connect(dsn, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable for schema authority check: {exc}")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'feature_matrix'
                """
            )
            database_columns = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    orm_columns = {column.name for column in FeatureMatrix.__table__.columns}
    assert orm_columns == database_columns, (
        "FeatureMatrix ORM/database drift:\n"
        f"missing from ORM: {sorted(database_columns - orm_columns)}\n"
        f"missing from database: {sorted(orm_columns - database_columns)}"
    )
