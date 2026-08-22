"""More lagged (strictly-prior-week) PBP feature families (Phase 3).

20260821_0017 added prior_epa_per_play/adot/drop_rate/avg_separation/
avg_cushion. This adds the rest of the PBP-sourced signals that were left
as permanently-cleared dead columns for no principled reason — they have a
real, live source in pbp_features (see pipeline/pbp_pipeline.py), same as
the ones already lagged. Only weather, player_emb_*, and the xFP family
(which has zero source data anywhere in the pipeline) stay excluded.

Revision ID: 20260822_0018
Revises: 20260821_0017
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0018"
down_revision: Union[str, None] = "20260821_0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = (
    "prior_epa_per_target",
    "prior_epa_per_rush",
    "prior_qb_epa_per_dropback",
    "prior_target_share_pbp",
    "prior_air_yards_share_pbp",
    "prior_red_zone_targets",
    "prior_end_zone_targets",
    "prior_red_zone_target_share",
    "prior_pass_left_rate",
    "prior_pass_middle_rate",
    "prior_pass_right_rate",
    "prior_ol_pressure_rate",
    "prior_ol_sack_rate",
    "prior_yac_per_reception",
    "prior_xyac_per_reception",
)


def upgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION")


def downgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE feature_matrix DROP COLUMN IF EXISTS {col}")
