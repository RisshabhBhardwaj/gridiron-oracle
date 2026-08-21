"""Persist opponent scheme columns (zone/man/blitz) into pbp_features.

They were computed in pbp_pipeline.py and used to update feature_matrix
directly, but never persisted to pbp_features itself — the audit table
that shows what pbp_pipeline actually derived for a player/game.

Revision ID: 20260820_0014
Revises: 20260820_0013
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260820_0014"
down_revision: Union[str, None] = "20260820_0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pbp_features ADD COLUMN IF NOT EXISTS opp_zone_pct FLOAT")
    op.execute("ALTER TABLE pbp_features ADD COLUMN IF NOT EXISTS opp_man_pct FLOAT")
    op.execute("ALTER TABLE pbp_features ADD COLUMN IF NOT EXISTS opp_blitz_rate FLOAT")


def downgrade() -> None:
    op.execute("ALTER TABLE pbp_features DROP COLUMN IF EXISTS opp_blitz_rate")
    op.execute("ALTER TABLE pbp_features DROP COLUMN IF EXISTS opp_man_pct")
    op.execute("ALTER TABLE pbp_features DROP COLUMN IF EXISTS opp_zone_pct")
