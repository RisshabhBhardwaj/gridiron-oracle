"""
backend/app/services/edge_coordinator.py

EdgeSignalCoordinator — decoupled orchestration for model edge signals.

Replaces the inline import / hardcoded-value pattern that was embedded in
AlertService.publish_edge().  By accepting its three dependencies at
construction time the coordinator is independently testable without
starting the IPC socket or the full alert bus.

Typical wiring (done once at application startup or in get_edge_coordinator()):

    coordinator = EdgeSignalCoordinator(
        alert_service=AlertService(),
        exposure_manager=exposure_manager,
        ipc_client=ipc_client,
    )

Call sites then invoke:

    coordinator.handle(
        player_id="...",
        player_name="...",
        stat="receiving_yards",
        edge_pct=0.12,
        body="12% edge vs market line",
        amount=50.0,
        game_id="2024_01_KC_BAL",
        position="WR",
        win_probability=0.60,
        decimal_odds=2.00,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.app.services.alert import AlertService
    from backend.app.services.exposure import PythonExposureManager
    from backend.app.services.ipc_client import IpcClient

logger = logging.getLogger(__name__)


class EdgeSignalCoordinator:
    """
    Orchestrates the three-step edge-signal pipeline:
      1. Pre-filter via Python ExposureManager (mirrors C++ limits).
      2. Forward to C++ engine via IPC.
      3. Broadcast alert to WebSocket clients via AlertService.

    Dependencies are injected so each can be replaced in tests.
    """

    def __init__(
        self,
        alert_service: "AlertService",
        exposure_manager: "PythonExposureManager",
        ipc_client: "IpcClient",
    ) -> None:
        self._alerts = alert_service
        self._exposure = exposure_manager
        self._ipc = ipc_client

    def handle(
        self,
        *,
        player_id: str,
        player_name: str,
        stat: str,
        edge_pct: float,
        body: str,
        # Parameters that callers should supply from real signal data.
        # Defaults match the original publish_edge() hardcoded values so
        # existing behaviour is preserved until callers are wired properly.
        amount: float = 50.0,
        game_id: str = "default_game_id",
        position: str = "WR",
        win_probability: float = 0.60,
        decimal_odds: float = 2.00,
    ) -> bool:
        """
        Process one edge signal end-to-end.

        Returns True if the signal was approved and forwarded, False if it
        was rejected by the exposure gate or the IPC queue was full.
        """
        from backend.app.services.exposure import ExposureDecision

        decision = self._exposure.try_place_bet(player_id, game_id, amount)
        if decision != ExposureDecision.APPROVED:
            logger.info(
                "Edge signal rejected by exposure gate (reason=%s) player=%s stat=%s",
                decision,
                player_id,
                stat,
            )
            return False

        event_dict = {
            "player_id": player_id,
            "game_id": game_id,
            "event_type": "projection",
            "position": position,
            "stat_delta": round(edge_pct, 4),
            "win_probability": win_probability,
            "decimal_odds": decimal_odds,
            "timestamp_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        }

        enqueued = self._ipc.send_event(event_dict)
        if not enqueued:
            logger.warning(
                "Failed to enqueue edge signal for IPC forwarding player=%s stat=%s",
                player_id,
                stat,
            )
            self._exposure.release_bet(player_id, game_id, amount)
            return False

        self._alerts.publish_edge_alert(
            player_id=player_id,
            player_name=player_name,
            stat=stat,
            edge_pct=edge_pct,
            body=body,
        )
        return True


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_coordinator: Optional[EdgeSignalCoordinator] = None


def get_edge_coordinator() -> EdgeSignalCoordinator:
    """
    Return the module-level EdgeSignalCoordinator, building it on first call.
    Separated from __init__ so tests can inject fakes via set_edge_coordinator().
    """
    global _coordinator
    if _coordinator is None:
        from backend.app.services.alert import AlertService
        from backend.app.services.exposure import exposure_manager
        from backend.app.services.ipc_client import ipc_client

        _coordinator = EdgeSignalCoordinator(
            alert_service=AlertService(),
            exposure_manager=exposure_manager,
            ipc_client=ipc_client,
        )
    return _coordinator


def set_edge_coordinator(coordinator: EdgeSignalCoordinator) -> None:
    """Replace the module-level instance (for testing)."""
    global _coordinator
    _coordinator = coordinator
