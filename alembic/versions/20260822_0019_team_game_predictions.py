"""Team-game outcome predictions table (Phase 4).

Serves the first real game-outcome endpoint. Materialized (not computed
per-request) so the API reads a fixed, reproducible snapshot rather than
refitting Ridge on every call — consistent with the approval-gated,
SHA-pinned serving pattern used for player projections elsewhere.

Revision ID: 20260822_0019
Revises: 20260822_0018
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0019"
down_revision: Union[str, None] = "20260822_0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS team_game_predictions (
            game_id             TEXT     NOT NULL,
            team                TEXT     NOT NULL,
            opponent             TEXT     NOT NULL,
            season              INTEGER  NOT NULL,
            week                INTEGER  NOT NULL,
            is_home             SMALLINT NOT NULL,
            points              DOUBLE PRECISION,
            plays               DOUBLE PRECISION,
            yards               DOUBLE PRECISION,
            pass_rate           DOUBLE PRECISION,
            win_probability     DOUBLE PRECISION,
            model_run_id        TEXT     NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (game_id, team)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_game_predictions_season_week "
        "ON team_game_predictions (season, week, team)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_team_game_predictions_season_week")
    op.execute("DROP TABLE IF EXISTS team_game_predictions")
