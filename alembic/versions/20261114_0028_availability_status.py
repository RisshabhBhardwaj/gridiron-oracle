"""Availability status on forward projections, and the snapshots it comes from.

Decision 0012 D3 and the 2026-11-14 addendum: the forward producer stores a row for
every rostered depth-chart player with the status known at the feature snapshot's
``as_of``; the API and the weekly acceptance check serve only rows whose status is not
excluded. History rows leave the column NULL.

* ``projections.availability_status``: nullable. NULL means no report against the player.
  Allowed values: Questionable, Doubtful (served), Out, Reserve, Suspended, Inactive (not served).
* ``availability_snapshots`` / ``availability_reports``: each fetch of the injury and
  weekly-roster sources, stored whole with its fetch time, so a later run can show exactly
  what was known (``pipeline/availability_store.py``).

Additive only. The downgrade removes exactly what the upgrade added.

Revision ID: 20261114_0028
Revises: 20261114_0027
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20261114_0028"
down_revision: Union[str, None] = "20261114_0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUSES = ("Questionable", "Doubtful", "Out", "Reserve", "Suspended", "Inactive")


def upgrade() -> None:
    op.execute("ALTER TABLE projections ADD COLUMN IF NOT EXISTS availability_status TEXT")
    allowed = ", ".join(f"'{s}'" for s in _STATUSES)
    op.execute("ALTER TABLE projections DROP CONSTRAINT IF EXISTS ck_projections_availability_status")
    op.execute(
        "ALTER TABLE projections ADD CONSTRAINT ck_projections_availability_status "
        f"CHECK (availability_status IS NULL OR availability_status IN ({allowed}))"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS availability_snapshots (
            snapshot_id TEXT        PRIMARY KEY,
            season      INTEGER     NOT NULL,
            week        INTEGER     NOT NULL,
            fetched_at  TIMESTAMPTZ NOT NULL,
            source      TEXT        NOT NULL,
            n_reports   INTEGER     NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_availability_snapshots_week "
        "ON availability_snapshots (season, week, fetched_at)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS availability_reports (
            snapshot_id     TEXT NOT NULL REFERENCES availability_snapshots (snapshot_id),
            player_id       TEXT NOT NULL,
            team            TEXT,
            report_status   TEXT,
            practice_status TEXT,
            roster_status   TEXT,
            date_modified   TIMESTAMPTZ,
            PRIMARY KEY (snapshot_id, player_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS availability_reports")
    op.execute("DROP INDEX IF EXISTS idx_availability_snapshots_week")
    op.execute("DROP TABLE IF EXISTS availability_snapshots")
    op.execute("ALTER TABLE projections DROP CONSTRAINT IF EXISTS ck_projections_availability_status")
    op.execute("ALTER TABLE projections DROP COLUMN IF EXISTS availability_status")
