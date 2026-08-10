"""Record the source-time contract for recoverable feature inputs.

Revision ID: 20260809_0005
Revises: 20260806_0004
Create Date: 2026-08-09
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260809_0005"
down_revision: Union[str, None] = "20260806_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "player_season_profiles",
        sa.Column("player_id", sa.Text(), nullable=False),
        sa.Column("effective_season", sa.Integer(), nullable=False),
        sa.Column("height_inches", sa.Float(), nullable=True),
        sa.Column("weight_lbs", sa.Float(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("player_id", "effective_season"),
    )
    op.create_index(
        "idx_player_season_profiles_lookup",
        "player_season_profiles",
        ["player_id", "effective_season"],
    )
    # Existing installs retain one nflreadpy roster snapshot per season in
    # staging.  Promote that retained source data inside the migration instead
    # of reading today's mutable ``players`` table.
    op.execute(
        """
        INSERT INTO player_season_profiles
            (player_id, effective_season, height_inches, weight_lbs, source, source_captured_at)
        SELECT DISTINCT ON (raw_data->>'gsis_id', (raw_data->>'season')::INTEGER)
            raw_data->>'gsis_id',
            (raw_data->>'season')::INTEGER,
            CASE
                WHEN raw_data->>'height' ~ '^[0-9]+(\\.[0-9]+)?$'
                    THEN (raw_data->>'height')::FLOAT
                WHEN raw_data->>'height' ~ '^[0-9]+-[0-9]+$'
                    THEN SPLIT_PART(raw_data->>'height', '-', 1)::FLOAT * 12
                         + SPLIT_PART(raw_data->>'height', '-', 2)::FLOAT
                ELSE NULL
            END,
            CASE WHEN raw_data->>'weight' ~ '^[0-9]+(\\.[0-9]+)?$'
                 THEN (raw_data->>'weight')::FLOAT ELSE NULL END,
            'nflreadpy.load_rosters', ingested_at
        FROM staging_nflreadpy
        WHERE source_type = 'rosters'
          AND raw_data->>'gsis_id' IS NOT NULL
          AND raw_data->>'season' ~ '^[0-9]+$'
        ORDER BY raw_data->>'gsis_id', (raw_data->>'season')::INTEGER, ingested_at DESC, id DESC
        ON CONFLICT (player_id, effective_season) DO UPDATE SET
            height_inches = EXCLUDED.height_inches,
            weight_lbs = EXCLUDED.weight_lbs,
            source = EXCLUDED.source,
            source_captured_at = EXCLUDED.source_captured_at
        """
    )

    op.add_column(
        "depth_charts",
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "depth_charts",
        sa.Column("source", sa.Text(), nullable=True),
    )
    op.add_column(
        "games",
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "weather_forecasts",
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("forecast_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("temp_f", sa.Float(), nullable=True),
        sa.Column("wind_mph", sa.Float(), nullable=True),
        sa.Column("precipitation_bucket", sa.SmallInteger(), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("game_id", "provider", "captured_at"),
    )
    op.create_index(
        "idx_weather_forecasts_asof",
        "weather_forecasts",
        ["game_id", "captured_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_weather_forecasts_asof", table_name="weather_forecasts")
    op.drop_table("weather_forecasts")
    op.drop_column("games", "kickoff_at")
    op.drop_column("depth_charts", "source")
    op.drop_column("depth_charts", "published_at")
    op.drop_index("idx_player_season_profiles_lookup", table_name="player_season_profiles")
    op.drop_table("player_season_profiles")
