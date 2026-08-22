"""Add play-outcome columns to pbp_plays (Phase 5).

ml.markov_simulator.DriveMarkovModel.fit() needs yards_gained, interception,
fumble_lost, penalty, penalty_yards to fit transition distributions, but
pbp_plays (Phase 2) only carries game-state columns (down/ydstogo/
yardline_100/score_differential/quarter). Without these, the Markov fit
path could only run against a fresh nflreadpy pull
(scripts/export_drive_transitions.py --from-nflverse), duplicating a fetch
the pipeline has already done and leaving the drive-simulator fit outside
the project's DB-backed materialization pattern. All five columns are
already present in nflreadpy.load_pbp()'s frame and available in
pipeline/pbp_pipeline.py's per-season load — just not previously selected
into pbp_plays.

Revision ID: 20260822_0021
Revises: 20260822_0020
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0021"
down_revision: Union[str, None] = "20260822_0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pbp_plays
            ADD COLUMN IF NOT EXISTS yards_gained FLOAT,
            ADD COLUMN IF NOT EXISTS interception SMALLINT,
            ADD COLUMN IF NOT EXISTS fumble_lost SMALLINT,
            ADD COLUMN IF NOT EXISTS penalty SMALLINT,
            ADD COLUMN IF NOT EXISTS penalty_yards FLOAT
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE pbp_plays
            DROP COLUMN IF EXISTS yards_gained,
            DROP COLUMN IF EXISTS interception,
            DROP COLUMN IF EXISTS fumble_lost,
            DROP COLUMN IF EXISTS penalty,
            DROP COLUMN IF EXISTS penalty_yards
        """
    )
