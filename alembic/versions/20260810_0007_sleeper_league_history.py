"""Persist completed Sleeper league drafts for tendency analysis.

Revision ID: 20260810_0007
Revises: 20260810_0006
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260810_0007"
down_revision: Union[str, None] = "20260810_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS sleeper_league_draft_picks (
            draft_id TEXT NOT NULL,
            pick_no INTEGER NOT NULL,
            season INTEGER NOT NULL,
            league_id TEXT,
            roster_id INTEGER,
            owner_id TEXT,
            round INTEGER,
            draft_slot INTEGER,
            player_id TEXT,
            player_name TEXT,
            position TEXT,
            team TEXT,
            raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (draft_id, pick_no)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_sleeper_league_draft_picks_season
        ON sleeper_league_draft_picks (season, league_id, roster_id)
    """)


def downgrade() -> None:
    op.drop_index("idx_sleeper_league_draft_picks_season", table_name="sleeper_league_draft_picks")
    op.drop_table("sleeper_league_draft_picks")
