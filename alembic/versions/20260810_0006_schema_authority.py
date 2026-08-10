"""Make the rebuilt feature schema explicit and remove runtime DDL ownership.

Revision ID: 20260810_0006
Revises: 20260809_0005
Create Date: 2026-08-10
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260810_0006"
down_revision: Union[str, None] = "20260809_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Literal, post-feature-contract FeatureRow additions.  The original columns
# and the Phase 4 group belong to 0001 and 0004 respectively.
_FEATURE_MATRIX_COLUMNS = (
    ("kalman_est_red_zone_target_share", "FLOAT"),
    ("kalman_variance_red_zone_target_share", "FLOAT"),
    ("seas_avg_passing_cpoe", "FLOAT"),
    ("seas_avg_passing_epa", "FLOAT"),
    ("seas_avg_receiving_epa", "FLOAT"),
    ("seas_avg_rushing_epa", "FLOAT"),
    ("seas_avg_racr", "FLOAT"),
    ("seas_avg_wopr", "FLOAT"),
    ("seas_avg_receiving_yac", "FLOAT"),
    ("opp_avg_carries_allowed", "FLOAT"),
    ("opp_avg_completions_allowed", "FLOAT"),
    ("opp_avg_pass_attempts_allowed", "FLOAT"),
    ("precipitation_bucket", "INTEGER"),
    ("epa_per_play", "FLOAT"),
    ("epa_per_target", "FLOAT"),
    ("epa_per_rush", "FLOAT"),
    ("qb_epa_per_dropback", "FLOAT"),
    ("adot", "FLOAT"),
    ("yac_per_reception", "FLOAT"),
    ("xyac_per_reception", "FLOAT"),
    ("target_share_pbp", "FLOAT"),
    ("air_yards_share_pbp", "FLOAT"),
    ("red_zone_targets", "INTEGER"),
    ("end_zone_targets", "INTEGER"),
    ("red_zone_target_share", "FLOAT"),
    ("pass_left_rate", "FLOAT"),
    ("pass_middle_rate", "FLOAT"),
    ("pass_right_rate", "FLOAT"),
    ("drop_rate", "FLOAT"),
    ("ol_pressure_rate", "FLOAT"),
    ("ol_sack_rate", "FLOAT"),
    ("opp_pressure_rate_pbp", "FLOAT"),
    ("opp_sack_rate_pbp", "FLOAT"),
    ("routes_run_pct", "FLOAT"),
    ("target_share_trend", "FLOAT"),
    ("wind_x_qb", "FLOAT"),
    ("wind_x_wr", "FLOAT"),
    ("precip_x_pass", "FLOAT"),
    ("team_pos_rank", "FLOAT"),
    ("depth_chart_rank", "FLOAT"),
    ("avg_separation", "FLOAT"),
    ("avg_cushion", "FLOAT"),
    ("player_emb_0", "FLOAT"),
    ("player_emb_1", "FLOAT"),
    ("player_emb_2", "FLOAT"),
    ("player_emb_3", "FLOAT"),
    ("player_emb_4", "FLOAT"),
    ("player_emb_5", "FLOAT"),
    ("player_emb_6", "FLOAT"),
    ("player_emb_7", "FLOAT"),
)


def upgrade() -> None:
    for column, sql_type in _FEATURE_MATRIX_COLUMNS:
        op.execute(f"ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS {column} {sql_type}")
    # These were formerly added by normalize.py at runtime.
    op.execute("ALTER TABLE game_logs ADD COLUMN IF NOT EXISTS receiving_fumbles INTEGER")
    op.execute("ALTER TABLE game_logs ADD COLUMN IF NOT EXISTS sack_fumbles INTEGER")
    op.execute("ALTER TABLE games ADD COLUMN IF NOT EXISTS precipitation_bucket SMALLINT")


def downgrade() -> None:
    op.execute("ALTER TABLE games DROP COLUMN IF EXISTS precipitation_bucket")
    op.execute("ALTER TABLE game_logs DROP COLUMN IF EXISTS sack_fumbles")
    op.execute("ALTER TABLE game_logs DROP COLUMN IF EXISTS receiving_fumbles")
    for column, _ in reversed(_FEATURE_MATRIX_COLUMNS):
        op.execute(f"ALTER TABLE feature_matrix DROP COLUMN IF EXISTS {column}")
