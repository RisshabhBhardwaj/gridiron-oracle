"""Add players.draft_round / draft_number and team_coaching table.

Revision ID: 20260806_0003
Revises: 20260806_0002
Create Date: 2026-08-06
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260806_0003"
down_revision: Union[str, None] = "20260806_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE players ADD COLUMN IF NOT EXISTS draft_round FLOAT")
    op.execute("ALTER TABLE players ADD COLUMN IF NOT EXISTS draft_number INTEGER")
    op.execute("ALTER TABLE players ADD COLUMN IF NOT EXISTS draft_club TEXT")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS team_coaching (
            season INTEGER NOT NULL,
            team TEXT NOT NULL REFERENCES teams(id),
            head_coach TEXT,
            offensive_coordinator TEXT,
            defensive_coordinator TEXT,
            scheme_pass_rate_prior FLOAT,
            notes TEXT,
            source TEXT,
            verified_by TEXT,
            verified_on DATE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (season, team)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS fantasy_adp (
            season INTEGER NOT NULL,
            source TEXT NOT NULL,
            scoring TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            team TEXT,
            adp FLOAT NOT NULL,
            player_id TEXT REFERENCES players(id),
            imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (season, source, scoring, player_name)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS fantasy_adp")
    op.execute("DROP TABLE IF EXISTS team_coaching")
    op.execute("ALTER TABLE players DROP COLUMN IF EXISTS draft_club")
    op.execute("ALTER TABLE players DROP COLUMN IF EXISTS draft_number")
    op.execute("ALTER TABLE players DROP COLUMN IF EXISTS draft_round")
