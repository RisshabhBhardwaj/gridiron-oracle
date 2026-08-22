"""Season simulation team win totals (Phase 7, the Vikings test).

SeasonSimulator.run() already computes team_win_totals — a count of wins
per team, accumulated from the SAME per-path team score draws used for
player-stat coupling — but scripts/materialize_season_simulation.py never
persisted it, only the player rows. Without a served copy there is nothing
to compare a served season win total against, so the plan's "team win
totals match the game-by-game surface" verify criterion had no surface to
run against. This table gives it one, keyed by the same
(season, start_week, pipeline_run_id) as season_simulations so a served win
total and the underlying player/game rows are provably from the same run.

Revision ID: 20260822_0025
Revises: 20260822_0024
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0025"
down_revision: Union[str, None] = "20260822_0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS season_team_wins (
            season              INTEGER  NOT NULL,
            start_week          INTEGER  NOT NULL,
            team                TEXT     NOT NULL,
            wins_mean           DOUBLE PRECISION NOT NULL,
            wins_p10            DOUBLE PRECISION,
            wins_p90            DOUBLE PRECISION,
            pipeline_run_id     TEXT     NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (season, start_week, team)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_season_team_wins_lookup "
        "ON season_team_wins (season, start_week, pipeline_run_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_season_team_wins_lookup")
    op.execute("DROP TABLE IF EXISTS season_team_wins")
