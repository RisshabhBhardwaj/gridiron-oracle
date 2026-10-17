"""Causal line snapshots (decision 0008, item 3).

`games.total_line`/`spread_line` are overwritten by every normalize run, so
the history of what the line was before kickoff is lost. This append-only
table preserves it: pipeline/normalize.py inserts one row per game per run
that carries a line, keyed by when it was observed (`captured_at`).

Readers take the latest row captured strictly before as-of (and before
kickoff when known); see pipeline/line_snapshots.py:get_line_as_of.

Revision ID: 20260822_0026
Revises: 20260822_0025
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0026"
down_revision: Union[str, None] = "20260822_0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS game_line_snapshots (
            game_id           TEXT        NOT NULL REFERENCES games(id) ON DELETE CASCADE,
            captured_at       TIMESTAMPTZ NOT NULL,
            spread_line       DOUBLE PRECISION,
            total_line        DOUBLE PRECISION,
            source            TEXT        NOT NULL DEFAULT 'nflreadpy_schedule',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (game_id, captured_at)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_game_line_snapshots_lookup "
        "ON game_line_snapshots (game_id, captured_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_game_line_snapshots_lookup")
    op.execute("DROP TABLE IF EXISTS game_line_snapshots")
