"""
backend/app/db.py

SQLModel / SQLAlchemy sync session factory.

All FastAPI route handlers receive a Session via the `get_db` dependency.
The session is opened at request start and closed (with rollback on error)
at request end via the generator.

Design choice: sync sessions (not asyncpg) to stay consistent with the
ML pipeline (train.py, backtest.py) which all use psycopg2 directly.
FastAPI runs sync dependencies in a threadpool automatically.
"""

from __future__ import annotations

from typing import Generator

from sqlmodel import Session, create_engine

from backend.app.core.config import settings

# Create a single engine shared across all requests.
# pool_pre_ping=True: silently reconnect dropped connections (important
# for long-running FastAPI processes overnight).
_engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a SQLModel Session per request.

    Usage:
        @router.get("/foo")
        def foo(db: Session = Depends(get_db)):
            ...
    """
    with Session(_engine) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise


def get_engine():
    """Return the shared SQLAlchemy engine (for raw queries)."""
    return _engine
