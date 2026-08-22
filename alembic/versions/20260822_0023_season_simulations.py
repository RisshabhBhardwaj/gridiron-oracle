"""Season simulation player-projection table (Phase 7).

Serves SeasonSimulator's rest-of-season player totals through
/projections/season/{n}. Materialized like team_game_predictions
(20260822_0019) and projections generally — a full 14-week x ~320-skill-
player run takes tens of seconds (real Phase 4 model fit + per-week Ridge
predict + Monte Carlo), too slow for a synchronous request under the
endpoint's 20/minute rate limit. scripts/materialize_season_simulation.py
writes this table on a schedule; get_season_projections reads it when an
approved row exists for (season, start_week), falling back to the flat-rate
Monte Carlo path otherwise.

Carries p_active/degraded/interval_method so the cold-start guard
(ml.playing_time.assert_cold_start_qb_not_in_top24) and the "no forecast"
degraded-flag path both survive the materialize -> serve round trip exactly
as they do for the existing path.

Revision ID: 20260822_0023
Revises: 20260822_0022
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0023"
down_revision: Union[str, None] = "20260822_0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS season_simulations (
            season              INTEGER  NOT NULL,
            start_week          INTEGER  NOT NULL,
            player_id           TEXT     NOT NULL,
            stat                TEXT     NOT NULL,
            player_name         TEXT,
            position             TEXT,
            team                TEXT,
            mean                DOUBLE PRECISION NOT NULL,
            p10                 DOUBLE PRECISION,
            p50                 DOUBLE PRECISION,
            p90                 DOUBLE PRECISION,
            p_active            DOUBLE PRECISION,
            prior_games         DOUBLE PRECISION,
            prior_active_games  DOUBLE PRECISION,
            depth_rank          DOUBLE PRECISION,
            degraded            BOOLEAN  NOT NULL DEFAULT FALSE,
            interval_method     TEXT     NOT NULL,
            pipeline_run_id     TEXT     NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (season, start_week, player_id, stat)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_season_simulations_lookup "
        "ON season_simulations (season, start_week, pipeline_run_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_season_simulations_lookup")
    op.execute("DROP TABLE IF EXISTS season_simulations")
