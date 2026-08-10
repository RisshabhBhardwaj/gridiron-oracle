"""
backend/app/models/__init__.py

Re-exports all SQLModel table definitions.
Import from here in migrations, pipeline code, and API services.
"""

from .production import FantasyADP, FeatureMatrix, Game, GameLog, Player, Projection, Team, TeamCoaching
from .staging import DeadLetter, StagingNflReadPy

__all__ = [
    # Production tables
    "Player",
    "Team",
    "Game",
    "GameLog",
    "FeatureMatrix",
    "FantasyADP",
    "Projection",
    "TeamCoaching",
    # Staging / ETL tables
    "StagingNflReadPy",
    "DeadLetter",
]
