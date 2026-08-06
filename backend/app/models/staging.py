"""
backend/app/models/staging.py

SQLModel table definitions for raw staging data and the dead-letter queue.

These tables are written by scraper adapters and read by the pipeline's
normalize.py step. They are never exposed directly by the API.

Schema design:
  - staging_nflreadpy: one row per validated source record (JSONB payload)
  - dead_letter:        one row per validation failure — NEVER silently dropped
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class StagingNflReadPy(SQLModel, table=True):
    """
    Raw staging table for all nflreadpy data.

    source_type discriminates the record kind:
      "player_stats" | "rosters" | "schedules" | "snap_counts"

    raw_data stores the Pydantic-validated row as JSONB.
    processed=False until pipeline/normalize.py promotes to production tables.
    """

    __tablename__ = "staging_nflreadpy"
    __table_args__ = (
        Index("idx_staging_nflreadpy_lookup", "source_type", "season", "week"),
        Index(
            "idx_staging_nflreadpy_unprocessed",
            "processed",
            postgresql_where="NOT processed",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    source_type: str = Field(max_length=50)
    season: int
    week: Optional[int] = None
    raw_data: Any = Field(
        default=None,
        sa_column=Column(JSONB, nullable=False),
    )
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)


class DeadLetter(SQLModel, table=True):
    """
    Dead-letter queue for rows that failed Pydantic validation.

    CLAUDE.md §3: "Never silently drop bad data."
    Every failed row lands here with the full error message and raw payload.

    Fields:
      source          — e.g. "nflreadpy.player_stats"
      error_message   — full Pydantic ValidationError string
      raw_payload     — the raw row dict as JSONB
      ingested_at     — when the failure was recorded
    """

    __tablename__ = "dead_letter"

    id: Optional[int] = Field(default=None, primary_key=True)
    source: str = Field(max_length=100)
    error_message: str
    raw_payload: Any = Field(
        default=None,
        sa_column=Column(JSONB, nullable=False),
    )
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
