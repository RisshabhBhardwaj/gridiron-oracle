"""Adopt the sleeper tables already present in production.

Revision ID: 20260810_0008
Revises: 20260810_0007

The previous sleeper-history revisions were removed during a scope revert while
their database objects and version marker remained. Reintroducing this revision
with the exact live schema makes both fresh installs and the existing database
walk one authoritative Alembic chain again.
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
        CREATE TABLE IF NOT EXISTS sleeper_league_draft_picks (
            draft_id TEXT NOT NULL, pick_no INTEGER NOT NULL, season INTEGER NOT NULL,
            league_id TEXT, roster_id INTEGER, owner_id TEXT, round INTEGER,
            draft_slot INTEGER, player_id TEXT, player_name TEXT, position TEXT, team TEXT,
            raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (draft_id, pick_no)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_sleeper_league_draft_picks_season
        ON sleeper_league_draft_picks (season, league_id, roster_id)
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS sleeper_league_members (
            league_id TEXT NOT NULL, owner_id TEXT NOT NULL, username TEXT,
            display_name TEXT, avatar TEXT,
            imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (league_id, owner_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_sleeper_league_members_owner
        ON sleeper_league_members (owner_id, imported_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_sleeper_league_members_owner")
    op.execute("DROP TABLE IF EXISTS sleeper_league_members")
    op.execute("DROP INDEX IF EXISTS idx_sleeper_league_draft_picks_season")
    op.execute("DROP TABLE IF EXISTS sleeper_league_draft_picks")
