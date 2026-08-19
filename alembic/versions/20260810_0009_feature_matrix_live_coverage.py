"""Reconcile FeatureMatrix with the live feature_matrix schema.

Revision ID: 20260810_0009
Revises: 20260810_0008
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260810_0009"
down_revision: Union[str, None] = "20260810_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    *(f"player_emb_{index}" for index in range(8, 32)),
    "routes_run_per_game",
    "slot_rate",
)


def upgrade() -> None:
    for column in _COLUMNS:
        op.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {column} DOUBLE PRECISION")


def downgrade() -> None:
    for column in reversed(_COLUMNS):
        op.execute(f"ALTER TABLE feature_matrix DROP COLUMN IF EXISTS {column}")
