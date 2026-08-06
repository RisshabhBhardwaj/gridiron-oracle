"""Add projections.max_train_season for causal provenance.

Revision ID: 20260806_0002
Revises: 20260806_0001
Create Date: 2026-08-06
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260806_0002"
down_revision: Union[str, None] = "20260806_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE projections ADD COLUMN IF NOT EXISTS max_train_season INTEGER"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE projections DROP COLUMN IF EXISTS max_train_season")
