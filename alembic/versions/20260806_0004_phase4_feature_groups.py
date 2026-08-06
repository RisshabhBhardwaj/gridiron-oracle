"""Add Phase 4 feature_matrix columns (usage / pace / progression).

Revision ID: 20260806_0004
Revises: 20260806_0003
Create Date: 2026-08-06
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260806_0004"
down_revision: Union[str, None] = "20260806_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PHASE4_COLS = [
    "carry_share",
    "snap_share_trailing",
    "snap_share_trend",
    "ts_vs_league",
    "rz_ts_vs_league",
    "snap_vs_pos_avg",
    "carry_share_vs_league",
    "opp_adj_target_share",
    "team_pace",
    "team_pass_rate",
    "expected_pass_attempts",
    "expected_pass_rate",
    "neutral_script_flag",
    "years_exp",
    "age",
    "career_games",
    "exp_bucket",
]


def upgrade() -> None:
    for col in _PHASE4_COLS:
        op.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {col} FLOAT")


def downgrade() -> None:
    for col in _PHASE4_COLS:
        op.execute(f"ALTER TABLE feature_matrix DROP COLUMN IF EXISTS {col}")
