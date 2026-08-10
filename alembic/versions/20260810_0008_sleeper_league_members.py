"""Store Sleeper member identities alongside historical draft picks.

Revision ID: 20260810_0008
Revises: 20260810_0007
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260810_0008"
down_revision: Union[str, None] = "20260810_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS sleeper_league_members (
            league_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            username TEXT,
            display_name TEXT,
            avatar TEXT,
            imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (league_id, owner_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_sleeper_league_members_owner
        ON sleeper_league_members (owner_id, imported_at DESC)
    """)


def downgrade() -> None:
    op.drop_index("idx_sleeper_league_members_owner", table_name="sleeper_league_members")
    op.drop_table("sleeper_league_members")
