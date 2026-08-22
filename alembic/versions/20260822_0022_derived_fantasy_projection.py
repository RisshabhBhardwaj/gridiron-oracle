"""Add projections.derived_fantasy_projection (Phase 6, L4 scoring).

fantasy_projection on the fantasy_ppr row stays the direct regression's own
output (the plan's "calibration anchor"). derived_fantasy_projection is the
NEW number, computed by scripts/materialize_stack_projections.py via
ml.scoring.score_fantasy from that player-game's full set of component-stat
projections (receiving_yards, receptions, receiving_tds, ...). Populated on
every row for a player-game (not just fantasy_ppr), fixing the "never-
written aggregation" the plan names: fantasy_projection = NULL for every
non-fantasy-ppr stat row.

Revision ID: 20260822_0022
Revises: 20260822_0021
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0022"
down_revision: Union[str, None] = "20260822_0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE projections ADD COLUMN IF NOT EXISTS derived_fantasy_projection DOUBLE PRECISION")


def downgrade() -> None:
    op.execute("ALTER TABLE projections DROP COLUMN IF EXISTS derived_fantasy_projection")
