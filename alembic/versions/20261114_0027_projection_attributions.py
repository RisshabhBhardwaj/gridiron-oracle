"""Per-row bucket attributions of each served projection (decision 0008, item 6).

The Settings page lets a user re-weight seven feature buckets. A re-weighted
number is the stored projection plus (ratio - 1) times that bucket's share of
it, so each served row needs its shares. Shares are exact TreeSHAP values from
the two base learners, blended with the stack's per-row meta weights
(ml/bucket_attribution.py), and written by scripts/materialize_stack_projections.py
under the same pipeline_run_id as the projection they explain.

``base + sum(seven buckets) + unweighted`` equals ``projections.projection``
for the row's (player_id, game_id, stat) to within 1e-4. The materializer
refuses to write a row where it does not, and the API refuses to use a row
whose run id differs from the served projection's or whose sum has drifted.

No foreign key to ``projections``: the pending-run overwrite guard handles
both tables with the same predicate, and the API joins on the run id.

Revision ID: 20261114_0027
Revises: 20260822_0026
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20261114_0027"
down_revision: Union[str, None] = "20260822_0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS projection_attributions (
            player_id         TEXT             NOT NULL,
            game_id           TEXT             NOT NULL,
            stat              TEXT             NOT NULL,
            pipeline_run_id   TEXT             NOT NULL,
            base              DOUBLE PRECISION NOT NULL,
            kalman_form       DOUBLE PRECISION NOT NULL,
            seasonal_baseline DOUBLE PRECISION NOT NULL,
            matchup           DOUBLE PRECISION NOT NULL,
            weather_venue     DOUBLE PRECISION NOT NULL,
            team_context      DOUBLE PRECISION NOT NULL,
            roster_injury     DOUBLE PRECISION NOT NULL,
            rule_meta         DOUBLE PRECISION NOT NULL,
            unweighted        DOUBLE PRECISION NOT NULL,
            max_residual      DOUBLE PRECISION NOT NULL,
            created_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
            PRIMARY KEY (player_id, game_id, stat)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_projection_attributions_run "
        "ON projection_attributions (pipeline_run_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_projection_attributions_run")
    op.execute("DROP TABLE IF EXISTS projection_attributions")
