"""Restore source-acquisition provenance for depth charts.

Revision ID: 20260820_0012
Revises: 20260820_0011
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260820_0012"
down_revision: Union[str, None] = "20260820_0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE depth_charts ADD COLUMN IF NOT EXISTS ingest_at TIMESTAMPTZ")
    op.execute("UPDATE depth_charts SET ingest_at = NOW() WHERE ingest_at IS NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE depth_charts DROP COLUMN IF EXISTS ingest_at")
