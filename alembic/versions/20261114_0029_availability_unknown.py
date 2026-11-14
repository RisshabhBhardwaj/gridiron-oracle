"""Allow the "Unknown" availability marker.

Coordinator answer 2 (2026-11-14): the manual ``--allow-no-availability`` override stores rows as
"availability unknown" so the UI can say so, instead of NULL (which means "no report against the
player"). Only the CHECK constraint changes. The downgrade first clears the marker, so it can
tighten the constraint again. Each direction commits the running transaction once, before the
validation scan (see ``_validate_without_blocking``).

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
_CONSTRAINT = "ck_projections_availability_status"


def _validate_without_blocking(constraint: str) -> None:
    """VALIDATE CONSTRAINT in its own transaction, so ADD CONSTRAINT's lock is already gone.

    alembic runs the whole upgrade inside one transaction, and ACCESS EXCLUSIVE is held until it ends,
    so NOT VALID followed by VALIDATE in that same transaction scans the table under the strong lock.
    ``autocommit_block`` commits first (releasing it); VALIDATE then needs only SHARE UPDATE EXCLUSIVE,
    which lets reads and writes through. SET LOCAL would vanish with the commit, so the timeout is a
    session setting and is reset straight after.
    """
    with op.get_context().autocommit_block():
        op.execute("SET lock_timeout = '5s'")
        op.execute(f"ALTER TABLE projections VALIDATE CONSTRAINT {constraint}")
        op.execute("RESET lock_timeout")


def _constraint(statuses: tuple[str, ...]) -> None:
    # A waiting ALTER holds an ACCESS EXCLUSIVE request and queues every reader behind it: give up fast.
    op.execute("SET LOCAL lock_timeout = '5s'")
    allowed = ", ".join(f"'{s}'" for s in statuses)
    op.execute("ALTER TABLE projections DROP CONSTRAINT IF EXISTS ck_projections_availability_status")
    op.execute(
        "ALTER TABLE projections ADD CONSTRAINT ck_projections_availability_status "
        f"CHECK (availability_status IS NULL OR availability_status IN ({allowed})) NOT VALID"
    )
    # The table scan runs in its own transaction (see _validate_without_blocking), after the lock above is released.
    _validate_without_blocking(_CONSTRAINT)


def upgrade() -> None:
    _constraint(_BASE + ("Unknown",))


def downgrade() -> None:
    op.execute("UPDATE projections SET availability_status = NULL WHERE availability_status = 'Unknown'")
    _constraint(_BASE)
