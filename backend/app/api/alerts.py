"""
backend/app/api/alerts.py

WebSocket /alerts/ws endpoint + REST /alerts endpoint.

WebSocket clients receive real-time broadcasts of:
  - Injury / practice report updates
  - Model edge signals from the C++ execution engine
  - System events (retrain complete, data freshness warnings)

REST /alerts returns the recent alert history for initial page load.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.app.services.alert import Alert, AlertService

router = APIRouter(prefix="", tags=["alerts"])
logger = logging.getLogger(__name__)

# Shared singleton alert service
_alert_svc = AlertService()


# ---------------------------------------------------------------------------
# REST response model
# ---------------------------------------------------------------------------

class AlertItem(BaseModel):
    id:          str
    severity:    str
    title:       str
    body:        str
    player_id:   Optional[str] = None
    player_name: Optional[str] = None
    stat:        Optional[str] = None
    value:       Optional[float] = None
    timestamp:   datetime


class AlertsResponse(BaseModel):
    count:  int
    alerts: list[AlertItem]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(n: int = Query(50, ge=1, le=200)) -> AlertsResponse:
    """
    Return the most recent n alerts (newest first).

    Used for initial page load of the Alerts feed before WebSocket connects.
    """
    recent = _alert_svc.recent(n)
    return AlertsResponse(
        count=len(recent),
        alerts=[_alert_to_item(a) for a in reversed(recent)],
    )


@router.websocket("/alerts/ws")
async def alerts_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for live alert streaming.

    Client connects and receives JSON alert objects as they are published.
    Connection stays open until client disconnects or server shuts down.

    Message format: JSON matching AlertItem schema.
    """
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    # Send recent history on connect so the client has context
    for alert in _alert_svc.recent(20):
        try:
            await websocket.send_text(alert.to_json())
        except Exception:
            break

    # Stream new alerts as they arrive
    try:
        async for alert in _alert_svc.subscribe():
            try:
                await websocket.send_text(alert.to_json())
            except WebSocketDisconnect:
                break
            except Exception as exc:
                logger.warning("WebSocket send error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("WebSocket client disconnected: %s", websocket.client)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _alert_to_item(a: Alert) -> AlertItem:
    return AlertItem(
        id=a.id,
        severity=a.severity.value,
        title=a.title,
        body=a.body,
        player_id=a.player_id,
        player_name=a.player_name,
        stat=a.stat,
        value=a.value,
        timestamp=a.timestamp,
    )


# ---------------------------------------------------------------------------
# Prometheus Alertmanager webhook receiver
# ---------------------------------------------------------------------------

@router.post("/alert-webhook", tags=["meta"], include_in_schema=False)
async def alertmanager_webhook(request: Request) -> dict:
    """
    Receive Prometheus Alertmanager webhook payloads and log them.

    Configured as Alertmanager's receiver so alerts are visible in backend
    logs within the Docker network without requiring an external notification
    service. Alertmanager posts here; the backend logs every firing/resolved
    alert at WARNING level.
    """
    try:
        payload = await request.json()
    except Exception:
        return {"status": "error", "detail": "invalid JSON"}

    status = payload.get("status", "unknown")
    for alert in payload.get("alerts", []):
        alert_status = alert.get("status", status)
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        alertname = labels.get("alertname", "unknown")
        summary = annotations.get("summary", "")
        description = annotations.get("description", "")
        severity = labels.get("severity", "unknown")
        log_level = logging.WARNING if alert_status == "firing" else logging.INFO
        logger.log(
            log_level,
            "Alertmanager %s — [%s] %s: %s %s",
            alert_status.upper(),
            severity,
            alertname,
            summary,
            description,
        )

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Expose the singleton for use by scheduled jobs and main.py
# ---------------------------------------------------------------------------

def get_alert_service() -> AlertService:
    """Return the shared AlertService singleton."""
    return _alert_svc
