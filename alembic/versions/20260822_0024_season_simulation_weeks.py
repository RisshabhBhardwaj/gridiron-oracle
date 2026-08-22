"""Season simulation per-week breakdown table (Phase 7 coherence).

Sibling to season_simulations (20260822_0023), which stores only the season
aggregate. SeasonSimulator.run() already computes a week_by_week breakdown
(mean/p10/p50/p90 per player per week) as part of the same run that
produces the season totals — this table persists it, so the plan's own
"summed weekly equals season" verify criterion can be written as an
executable test comparing two rows this codebase actually produces from the
SAME model run (same pipeline_run_id), rather than staying an assertion
about two different, deliberately-independent modeling pipelines (this
simulator vs. the stack models behind /projections/week/{n} — see
test_season_coherence.py for why those are NOT meant to reconcile).

Revision ID: 20260822_0024
Revises: 20260822_0023
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0024"
down_revision: Union[str, None] = "20260822_0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS season_simulation_weeks (
            season              INTEGER  NOT NULL,
            start_week          INTEGER  NOT NULL,
            week                INTEGER  NOT NULL,
            player_id           TEXT     NOT NULL,
            stat                TEXT     NOT NULL,
            mean                DOUBLE PRECISION NOT NULL,
            p10                 DOUBLE PRECISION,
            p50                 DOUBLE PRECISION,
            p90                 DOUBLE PRECISION,
            pipeline_run_id     TEXT     NOT NULL,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (season, start_week, week, player_id, stat)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_season_simulation_weeks_lookup "
        "ON season_simulation_weeks (season, start_week, pipeline_run_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_season_simulation_weeks_lookup")
    op.execute("DROP TABLE IF EXISTS season_simulation_weeks")
