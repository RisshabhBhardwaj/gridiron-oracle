"""
backend/app/services/exposure.py

Redundant ExposureManager that mirrors the C++ configuration logic.
Provides thread-safe (and async-safe) limits checking to pre-filter
signals in Python before pushing them to the IPC boundary.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ExposureDecision:
    APPROVED = "APPROVED"
    REJECTED_PLAYER = "REJECTED_PLAYER"
    REJECTED_GAME = "REJECTED_GAME"
    REJECTED_TOTAL = "REJECTED_TOTAL"


class PythonExposureManager:
    """
    Singleton exposure manager to replicate C++ limits (5% player, 20% game, 20% total).
    """
    _instance: "PythonExposureManager" | None = None
    _inst_lock = Lock()

    def __new__(cls, bankroll: float = 1000.0) -> "PythonExposureManager":
        with cls._inst_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.bankroll = bankroll
                cls._instance.max_per_player_fraction = 0.05
                cls._instance.max_per_game_fraction = 0.20
                cls._instance.max_total_in_play_fraction = 0.20
                
                # Tracking state
                cls._instance._player_exposure: Dict[str, float] = {}
                cls._instance._game_exposure: Dict[str, float] = {}
                cls._instance._total_in_play: float = 0.0
                
                # Mutex for state modifications
                cls._instance._mu = Lock()
        return cls._instance

    def try_place_bet(self, player_id: str, game_id: str, amount: float) -> str:
        """
        Check if an incoming bet violates any exposure limits.
        If it passes, reserve the exposure immediately.
        Returns an ExposureDecision string.
        """
        with self._mu:
            player_new = self._player_exposure.get(player_id, 0.0) + amount
            game_new = self._game_exposure.get(game_id, 0.0) + amount
            total_new = self._total_in_play + amount

            if player_new > self.max_per_player_fraction * self.bankroll:
                logger.warning(f"Python Exposure Reject: Player {player_id} limit hit")
                return ExposureDecision.REJECTED_PLAYER
            
            if game_new > self.max_per_game_fraction * self.bankroll:
                logger.warning(f"Python Exposure Reject: Game {game_id} limit hit")
                return ExposureDecision.REJECTED_GAME
            
            if total_new > self.max_total_in_play_fraction * self.bankroll:
                logger.warning("Python Exposure Reject: Total limit hit")
                return ExposureDecision.REJECTED_TOTAL
            
            # Commit the reservation
            self._player_exposure[player_id] = player_new
            self._game_exposure[game_id] = game_new
            self._total_in_play = total_new
            
            return ExposureDecision.APPROVED

    def release_bet(self, player_id: str, game_id: str, amount: float) -> None:
        """
        Release a bet (e.g. if C++ subsequently rejects it, or if it settles).
        """
        with self._mu:
            if player_id in self._player_exposure:
                self._player_exposure[player_id] = max(0.0, self._player_exposure[player_id] - amount)
            
            if game_id in self._game_exposure:
                self._game_exposure[game_id] = max(0.0, self._game_exposure[game_id] - amount)
                
            self._total_in_play = max(0.0, self._total_in_play - amount)

    def release_game(self, game_id: str) -> None:
        """Release an entire game's exposure."""
        with self._mu:
            if game_id in self._game_exposure:
                amount = self._game_exposure[game_id]
                self._total_in_play = max(0.0, self._total_in_play - amount)
                del self._game_exposure[game_id]
                # Note: exact per-player subtraction for a bulk game release is complex here.
                # In practice, live execution platforms will wipe all state cleanly or sync via 
                # a source-of-truth database on settlement. This mirrors C++'s simplified release_game.

    # Accessors for monitoring & API returns
    def get_exposures(self) -> Tuple[Dict[str, float], Dict[str, float], float]:
        with self._mu:
            return dict(self._player_exposure), dict(self._game_exposure), self._total_in_play

# Global instance
exposure_manager = PythonExposureManager()
