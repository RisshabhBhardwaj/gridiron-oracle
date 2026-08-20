"""Prediction-surface rebuild tables and serving columns.

Revision ID: 20260819_0010
Revises: 20260810_0009
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260819_0010"
down_revision: Union[str, None] = "20260810_0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE projections ADD COLUMN IF NOT EXISTS interval_method TEXT")
    op.execute(
        "ALTER TABLE feature_matrix ADD COLUMN IF NOT EXISTS routes_run_per_game DOUBLE PRECISION"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS consensus_projections_weekly (
            source TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            projection FLOAT,
            scoring TEXT NOT NULL DEFAULT 'ppr',
            captured_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (source, season, week, player_id, captured_at)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS fantasy_player_ids (
            gsis_id TEXT PRIMARY KEY,
            mfl_id TEXT,
            sleeper_id TEXT,
            espn_id TEXT,
            yahoo_id TEXT,
            fantasypros_id TEXT,
            pff_id TEXT,
            sportradar_id TEXT,
            cbs_id TEXT,
            rotowire_id TEXT,
            fleaflicker_id TEXT,
            full_name TEXT,
            position TEXT,
            team TEXT,
            refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_fantasy_player_ids_sleeper
            ON fantasy_player_ids (sleeper_id)
            WHERE sleeper_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS injury_reports (
            player_id TEXT,
            player_name TEXT NOT NULL,
            team TEXT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            practice_status TEXT,
            injury_type TEXT,
            source TEXT NOT NULL,
            captured_at TIMESTAMPTZ NOT NULL,
            raw_payload JSONB,
            PRIMARY KEY (season, week, source, player_name, captured_at)
        )
        """
    )
    op.execute(
        "ALTER TABLE draft_preseason_projections ADD COLUMN IF NOT EXISTS historical_games INTEGER"
    )
    op.execute(
        "ALTER TABLE draft_preseason_projections ADD COLUMN IF NOT EXISTS projection_basis TEXT"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE draft_preseason_projections DROP COLUMN IF EXISTS projection_basis")
    op.execute("ALTER TABLE draft_preseason_projections DROP COLUMN IF EXISTS historical_games")
    op.execute("DROP TABLE IF EXISTS injury_reports")
    op.execute("DROP INDEX IF EXISTS idx_fantasy_player_ids_sleeper")
    op.execute("DROP TABLE IF EXISTS fantasy_player_ids")
    op.execute("DROP TABLE IF EXISTS consensus_projections_weekly")
    op.execute("ALTER TABLE projections DROP COLUMN IF EXISTS interval_method")
