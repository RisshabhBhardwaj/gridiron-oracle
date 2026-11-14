"""Allow the "Unknown" availability marker.

Coordinator answer 2 (2026-11-14): the manual ``--allow-no-availability`` override stores rows as
"availability unknown" so the UI can say so, instead of NULL (which means "no report against the
player"). Only the CHECK constraint changes. The downgrade first clears the marker, so it can
tighten the constraint again.

Revision ID: 20261114_0029
Revises: 20261114_0028
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20261114_0029"
down_revision: Union[str, None] = "20261114_0028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_BASE = ("Questionable", "Doubtful", "Out", "Reserve", "Suspended", "Inactive")


def _constraint(statuses: tuple[str, ...]) -> None:
    allowed = ", ".join(f"'{s}'" for s in statuses)
    op.execute("ALTER TABLE projections DROP CONSTRAINT IF EXISTS ck_projections_availability_status")
    op.execute(
        "ALTER TABLE projections ADD CONSTRAINT ck_projections_availability_status "
        f"CHECK (availability_status IS NULL OR availability_status IN ({allowed}))"
    )


def upgrade() -> None:
    _constraint(_BASE + ("Unknown",))


def downgrade() -> None:
    op.execute("UPDATE projections SET availability_status = NULL WHERE availability_status = 'Unknown'")
    _constraint(_BASE)
