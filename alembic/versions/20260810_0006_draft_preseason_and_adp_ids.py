"""Add causal draft projections and make canonical player IDs the ADP key.

Revision ID: 20260810_0006
Revises: 20260809_0005
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "20260810_0006"
down_revision: Union[str, None] = "20260809_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS adp_player_matches (
            season INTEGER NOT NULL, source TEXT NOT NULL, scoring TEXT NOT NULL,
            player_name TEXT NOT NULL, normalized_name TEXT NOT NULL, position TEXT, team TEXT,
            matched_player_id TEXT REFERENCES players(id), match_method TEXT NOT NULL,
            match_status TEXT NOT NULL, candidate_player_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            audited_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (season, source, scoring, player_name)
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS fantasy_adp_unmatched (
            season INTEGER NOT NULL, source TEXT NOT NULL, scoring TEXT NOT NULL,
            player_name TEXT NOT NULL, position TEXT, team TEXT, adp FLOAT NOT NULL,
            reason TEXT NOT NULL, moved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (season, source, scoring, player_name)
        )
    """)
    # Preserve non-resolvable legacy rows for review before changing the key.
    op.execute("""
        INSERT INTO fantasy_adp_unmatched (season, source, scoring, player_name, position, team, adp, reason)
        SELECT season, source, scoring, player_name, position, team, adp, 'missing_player_id'
        FROM fantasy_adp WHERE player_id IS NULL
        ON CONFLICT (season, source, scoring, player_name) DO UPDATE SET
            position = EXCLUDED.position, team = EXCLUDED.team, adp = EXCLUDED.adp,
            reason = EXCLUDED.reason, moved_at = NOW()
    """)
    op.execute("DELETE FROM fantasy_adp WHERE player_id IS NULL")
    # A canonical player has one ADP value per source/scoring/season. Preserve
    # additional legacy aliases in the audit table before retaining lowest ADP.
    op.execute("""
        WITH duplicate_rows AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY season, source, scoring, player_id ORDER BY adp, player_name
            ) AS rn
            FROM fantasy_adp
        )
        INSERT INTO fantasy_adp_unmatched (season, source, scoring, player_name, position, team, adp, reason)
        SELECT season, source, scoring, player_name, position, team, adp, 'duplicate_player_id'
        FROM duplicate_rows WHERE rn > 1
        ON CONFLICT (season, source, scoring, player_name) DO UPDATE SET reason = EXCLUDED.reason
    """)
    op.execute("""
        DELETE FROM fantasy_adp a USING (
            SELECT ctid, ROW_NUMBER() OVER (
                PARTITION BY season, source, scoring, player_id ORDER BY adp, player_name
            ) AS rn FROM fantasy_adp
        ) d WHERE a.ctid = d.ctid AND d.rn > 1
    """)
    op.execute("ALTER TABLE fantasy_adp DROP CONSTRAINT IF EXISTS fantasy_adp_pkey")
    op.execute("ALTER TABLE fantasy_adp ALTER COLUMN player_id SET NOT NULL")
    op.execute("ALTER TABLE fantasy_adp ADD PRIMARY KEY (season, source, scoring, player_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS draft_preseason_projections (
            season INTEGER NOT NULL, as_of DATE NOT NULL, source TEXT NOT NULL,
            player_id TEXT NOT NULL REFERENCES players(id), player_name TEXT NOT NULL,
            position TEXT, team TEXT, per_game_mean FLOAT NOT NULL,
            games_played_prior FLOAT NOT NULL, projection FLOAT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (season, as_of, source, player_id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_draft_preseason_projection_lookup
        ON draft_preseason_projections (season, as_of DESC)
    """)


def downgrade() -> None:
    op.drop_index("idx_draft_preseason_projection_lookup", table_name="draft_preseason_projections")
    op.drop_table("draft_preseason_projections")
    # Audit records are intentionally retained; restoring name-keyed ADP would
    # silently reintroduce the collision that this migration removes.
