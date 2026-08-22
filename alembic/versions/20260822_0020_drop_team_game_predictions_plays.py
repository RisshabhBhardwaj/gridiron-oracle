"""Drop team_game_predictions.plays — the target has no signal (Phase 4).

On 2025 holdout, the plays Ridge model's MAE (6.897) is WORSE than the
training-mean constant baseline (6.877) — this feature set carries no
predictive signal for play count, likely because play count is driven by
in-game dynamics (turnovers, OT, blowout tempo) nothing pre-kickoff can
see. Serving a permanently-losing number next to points/yards/pass_rate/
win_probability (which do carry signal) would look authoritative without
being so. Dropped from ml.team_game_model.TARGET_COLS in the same change.

Revision ID: 20260822_0020
Revises: 20260822_0019
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260822_0020"
down_revision: Union[str, None] = "20260822_0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE team_game_predictions DROP COLUMN IF EXISTS plays")


def downgrade() -> None:
    op.execute("ALTER TABLE team_game_predictions ADD COLUMN IF NOT EXISTS plays DOUBLE PRECISION")
