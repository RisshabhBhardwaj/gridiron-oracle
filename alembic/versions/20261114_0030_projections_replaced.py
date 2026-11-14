"""Archive table for ``--replace-week`` (review M1).

``projections`` is unique on (player_id, game_id, stat) without the run id, so a new forward week
cannot sit beside the served rows until promotion. The replace copies the old week here inside the
producer's transaction. The promotion discards the copy; a failure after the producer puts it back.
The table has the same columns as ``projections`` (no constraints, no foreign keys: the rows are only
ever copied back into ``projections``, which enforces them) plus ``archived_at``. It is empty except
during a replace. Additive: nothing existing changes.

Revision ID: 20261114_0030
Revises: 20261114_0029
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20261114_0030"
down_revision: Union[str, None] = "20261114_0029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("CREATE TABLE IF NOT EXISTS projections_replaced (LIKE projections)")
    op.execute("ALTER TABLE projections_replaced ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NOT NULL DEFAULT now()")
    op.execute("CREATE INDEX IF NOT EXISTS ix_projections_replaced_week ON projections_replaced (season, week)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS projections_replaced")
