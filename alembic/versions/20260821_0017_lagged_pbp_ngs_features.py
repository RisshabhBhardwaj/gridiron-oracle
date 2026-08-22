"""Lagged (strictly-prior-week) versions of PBP/NGS features (Phase 3).

epa_per_play, adot, drop_rate, avg_separation, and avg_cushion are all
computed from the target game's own plays, so they're permanently forbidden
as model inputs (ml/feature_contract.py FORBIDDEN_MODEL_FIELDS). This adds
their trailing, strictly-prior-games counterparts, following the same
pattern already used for depth_chart_rank and routes_run_per_game.

Revision ID: 20260821_0017
Revises: 20260821_0016
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260821_0017"
down_revision: Union[str, None] = "20260821_0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    "prior_epa_per_play",
    "prior_adot",
    "prior_drop_rate",
    "prior_avg_separation",
    "prior_avg_cushion",
)


def upgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION")


def downgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE feature_matrix DROP COLUMN IF EXISTS {col}")
