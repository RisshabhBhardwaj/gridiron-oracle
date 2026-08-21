"""Add auditable source-time fields required for forward forecasts.

Revision ID: 20260820_0013
Revises: 20260820_0012
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "20260820_0013"
down_revision: Union[str, None] = "20260820_0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ``published_at`` is a source assertion; ``ingest_at`` records when we
    # obtained it.  They are deliberately distinct so an old snapshot cannot
    # masquerade as current roster truth.
    op.execute("ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS as_of TIMESTAMPTZ")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_matrix_forward_asof "
        "ON feature_matrix (season, week, as_of) WHERE as_of IS NOT NULL"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_runs (
            pipeline_run_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            as_of TIMESTAMPTZ NOT NULL,
            feature_rows INTEGER NOT NULL,
            source_summary JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (feature_rows > 0)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS forecast_runs")
    op.execute("DROP INDEX IF EXISTS idx_feature_matrix_forward_asof")
    op.execute("ALTER TABLE feature_matrix DROP COLUMN IF EXISTS as_of")
