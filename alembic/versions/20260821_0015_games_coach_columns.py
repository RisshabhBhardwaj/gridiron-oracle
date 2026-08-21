"""Retain head-coach identity from nfl.load_schedules() on games.

home_coach/away_coach are present in the raw schedules feed but were
declared nowhere and discarded at the ScheduleRow validation boundary
(extra="ignore"). This is HC identity, not play-caller identity, but it's
free and is the Phase 4 coach-tendency-fingerprinting prerequisite.

Revision ID: 20260821_0015
Revises: 20260820_0014
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260821_0015"
down_revision: Union[str, None] = "20260820_0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE games ADD COLUMN IF NOT EXISTS home_coach TEXT")
    op.execute("ALTER TABLE games ADD COLUMN IF NOT EXISTS away_coach TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE games DROP COLUMN IF EXISTS away_coach")
    op.execute("ALTER TABLE games DROP COLUMN IF EXISTS home_coach")
