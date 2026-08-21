"""Play-grain PBP table (Phase 2).

pbp_pipeline.py loaded the full play-by-play frame every run and discarded
it after aggregating to player-game grain in _aggregate_pbp. Down,
distance, field position, score differential, and clock never survived
past that aggregation, so nothing could condition on game state — the
prerequisite for the SP5 drive/play simulator work.

Revision ID: 20260821_0016
Revises: 20260821_0015
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260821_0016"
down_revision: Union[str, None] = "20260821_0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS pbp_plays (
            game_id                TEXT     NOT NULL,
            play_id                BIGINT   NOT NULL,
            season                 INTEGER  NOT NULL,
            week                   INTEGER  NOT NULL,
            posteam                TEXT,
            defteam                TEXT,
            play_type               TEXT,
            down                   SMALLINT,
            ydstogo                SMALLINT,
            yardline_100           SMALLINT,
            quarter                SMALLINT,
            game_seconds_remaining FLOAT,
            score_differential     SMALLINT,
            offense_personnel      TEXT,
            defense_personnel      TEXT,
            PRIMARY KEY (game_id, play_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pbp_plays_season_week ON pbp_plays (season, week)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pbp_plays_game_state "
        "ON pbp_plays (down, ydstogo, yardline_100, score_differential)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pbp_plays_game_state")
    op.execute("DROP INDEX IF EXISTS idx_pbp_plays_season_week")
    op.execute("DROP TABLE IF EXISTS pbp_plays")
