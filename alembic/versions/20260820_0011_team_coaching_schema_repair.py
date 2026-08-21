"""Restore the documented team-coaching schema repair migration.

Revision ID: 20260820_0011
Revises: 20260819_0010
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260820_0011"
down_revision: Union[str, None] = "20260819_0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE team_coaching ADD COLUMN IF NOT EXISTS source TEXT")
    op.execute("ALTER TABLE team_coaching ADD COLUMN IF NOT EXISTS verified_by TEXT")
    op.execute("ALTER TABLE team_coaching ADD COLUMN IF NOT EXISTS verified_on DATE")
    # These 2026 seeds were documented as contradictory. They must not be
    # consumed as caller truth.
    op.execute("DELETE FROM team_coaching WHERE season = 2026 AND team IN ('NE', 'WAS', 'SEA', 'CLE', 'TEN')")
    op.execute(
        "UPDATE team_coaching SET source = COALESCE(source, 'seed'), "
        "notes = COALESCE(notes, '') || ' [UNVERIFIED: do not consume for modeling]' "
        "WHERE season = 2026 AND verified_by IS NULL"
    )


def downgrade() -> None:
    # Do not restore invalid seed rows on downgrade.
    op.execute("ALTER TABLE team_coaching DROP COLUMN IF EXISTS verified_on")
    op.execute("ALTER TABLE team_coaching DROP COLUMN IF EXISTS verified_by")
    op.execute("ALTER TABLE team_coaching DROP COLUMN IF EXISTS source")
