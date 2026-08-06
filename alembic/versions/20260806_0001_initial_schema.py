"""Initial schema — idempotent CREATE TABLE IF NOT EXISTS bootstrap.

Revision ID: 20260806_0001
Revises:
Create Date: 2026-08-06
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from pipeline.schema_ddl import ALL_DDL

revision: str = "20260806_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for ddl in ALL_DDL:
        op.execute(ddl)
    for col, typ in (
        ("posterior_samples", "JSONB"),
        ("p25", "FLOAT"),
        ("p75", "FLOAT"),
        ("max_train_season", "INTEGER"),
    ):
        op.execute(f"ALTER TABLE projections ADD COLUMN IF NOT EXISTS {col} {typ}")


def downgrade() -> None:
    # Initial schema is not reversed automatically — drop databases explicitly.
    raise NotImplementedError("Downgrade of the initial schema is not supported.")
