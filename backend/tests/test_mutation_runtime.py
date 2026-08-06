from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from contextlib import suppress
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient


def test_settings_load_defaults_when_file_missing(monkeypatch, tmp_path):
    from backend.app.api import settings as settings_api

    config_path = tmp_path / "user_config.json"
    monkeypatch.setattr(settings_api, "_CONFIG_PATH", config_path)

    cfg = settings_api._load_config()

    assert cfg["weights"] == settings_api._DEFAULT_WEIGHTS
    assert cfg["fantasy_scoring"] == "ppr"
    assert cfg["engine_exposure_cap"] == 0.25


def _make_starlette_request(request_id: str = "test-req-id", path: str = "/settings"):
    """Real starlette.requests.Request — required by slowapi's isinstance check."""
    from starlette.requests import Request as StarletteRequest

    scope = {
        "type": "http",
        "method": "PUT",
        "path": path,
        "query_string": b"",
        "headers": [
            (b"x-request-id", request_id.encode()),
            (b"x-forwarded-for", b"127.0.0.1"),
        ],
    }
    return StarletteRequest(scope)


def test_settings_update_persists_changes(monkeypatch, tmp_path):
    from backend.app.api import settings as settings_api

    config_path = tmp_path / "user_config.json"
    monkeypatch.setattr(settings_api, "_CONFIG_PATH", config_path)

    req = settings_api.SettingsUpdateRequest(
        fantasy_scoring="half_ppr",
        engine_exposure_cap=0.2,
        weight_preset_name="Balanced",
    )

    result = settings_api.update_settings(_make_starlette_request(), req)

    assert result.fantasy_scoring == "half_ppr"
    assert result.engine_exposure_cap == 0.2
    assert result.weight_preset_name == "Balanced"
    persisted = json.loads(config_path.read_text())
    assert persisted["fantasy_scoring"] == "half_ppr"
    assert persisted["engine_exposure_cap"] == 0.2


def test_settings_update_rejects_invalid_scoring(monkeypatch, tmp_path):
    from fastapi import HTTPException

    from backend.app.api import settings as settings_api

    monkeypatch.setattr(settings_api, "_CONFIG_PATH", tmp_path / "user_config.json")

    try:
        settings_api.update_settings(
            _make_starlette_request(),
            settings_api.SettingsUpdateRequest(fantasy_scoring="draftkings"),
        )
    except HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("Expected invalid fantasy scoring to raise HTTPException")


def test_alerts_endpoint_returns_recent_alerts(monkeypatch):
    from backend.app.api import alerts as alerts_api
    from backend.app.services.alert import AlertService

    AlertService._instance = None
    svc = AlertService()
    svc.publish_system("First", "one")
    svc.publish_system("Second", "two")
    monkeypatch.setattr(alerts_api, "_alert_svc", svc)

    body = alerts_api.get_alerts(n=1)

    assert body.count == 1
    assert body.alerts[0].title == "Second"


def test_alert_to_item_preserves_fields():
    from datetime import datetime, timezone

    from backend.app.api.alerts import _alert_to_item
    from backend.app.services.alert import Alert, AlertSeverity

    alert = Alert(
        id="edge-1",
        severity=AlertSeverity.EDGE,
        title="Edge",
        body="12% edge",
        player_id="p1",
        player_name="Player One",
        stat="receiving_yards",
        value=0.1234,
        timestamp=datetime.now(timezone.utc),
    )

    item = _alert_to_item(alert)

    assert item.id == "edge-1"
    assert item.severity == "edge"
    assert item.player_name == "Player One"
    assert item.value == 0.1234


def test_settings_normalize_asyncpg_and_invalid_mode(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://oracle:oracle@localhost:15439/oracle",
    )
    monkeypatch.setenv("PRODUCT_MODE", "not-a-real-mode")
    monkeypatch.setenv("API_KEY", "  secret  ")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test,http://b.test")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "gridiron-audit")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console,otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer token,x-team=oracle")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "yes")

    settings = Settings()

    assert settings.database_url.startswith("postgresql://")
    assert settings.product_mode == "graceful_fallback"
    assert settings.api_key == "secret"
    assert settings.cors_origins == ["http://a.test", "http://b.test"]
    assert settings.otel_service_name == "gridiron-audit"
    assert settings.otel_traces_exporter == "console,otlp"
    assert settings.otel_exporter_otlp_endpoint == "http://collector:4318/v1/traces"
    assert settings.otel_exporter_otlp_protocol == "grpc"
    assert settings.otel_exporter_otlp_headers == "authorization=Bearer token,x-team=oracle"
    assert settings.otel_exporter_otlp_insecure is True


def test_settings_defaults_cover_all_runtime_fallbacks(monkeypatch):
    from backend.app.core.config import Settings

    for key in (
        "DATABASE_URL",
        "MLFLOW_TRACKING_URI",
        "MODEL_VERSION",
        "PRODUCT_MODE",
        "SETTINGS_FILE",
        "BASELINE_MANIFEST_PATH",
        "ARTIFACT_INVALIDATION_PATH",
        "LOG_LEVEL",
        "CORS_ORIGINS",
        "API_KEY",
        "OTEL_SERVICE_NAME",
        "OTEL_TRACES_EXPORTER",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_INSECURE",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.database_url == "postgresql://oracle:oracle@localhost:15439/oracle"
    assert settings.mlflow_tracking_uri == "http://localhost:5001"
    assert settings.model_version == "latest"
    assert settings.product_mode == "graceful_fallback"
    assert settings.settings_file == "user_config.json"
    assert settings.baseline_manifest_path == "releases/current_baseline.json"
    assert settings.artifact_invalidation_path == "releases/artifact_invalidations.json"
    assert settings.log_level == "INFO"
    assert settings.cors_origins == ["http://localhost:5173", "http://localhost:3000"]
    assert settings.api_key == ""
    assert settings.otel_service_name == "gridiron-oracle-api"
    assert settings.otel_traces_exporter == "none"
    assert settings.otel_exporter_otlp_endpoint == ""
    assert settings.otel_exporter_otlp_protocol == "http/protobuf"
    assert settings.otel_exporter_otlp_headers == ""
    assert settings.otel_exporter_otlp_insecure is False


def test_settings_respects_explicit_override_paths_and_modes(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://db.example.com:5432/oracle")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.internal:5001")
    monkeypatch.setenv("MODEL_VERSION", "2026.05.01")
    monkeypatch.setenv("PRODUCT_MODE", "artifact_backed")
    monkeypatch.setenv("SETTINGS_FILE", "custom_settings.json")
    monkeypatch.setenv("BASELINE_MANIFEST_PATH", "releases/custom_baseline.json")
    monkeypatch.setenv(
        "ARTIFACT_INVALIDATION_PATH",
        "releases/custom_invalidations.json",
    )
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "gridiron-prod")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318/v1/traces")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer prod-token")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "false")

    settings = Settings()

    assert settings.database_url == "postgresql://db.example.com:5432/oracle"
    assert settings.mlflow_tracking_uri == "http://mlflow.internal:5001"
    assert settings.model_version == "2026.05.01"
    assert settings.product_mode == "artifact_backed"
    assert settings.settings_file == "custom_settings.json"
    assert settings.baseline_manifest_path == "releases/custom_baseline.json"
    assert settings.artifact_invalidation_path == "releases/custom_invalidations.json"
    assert settings.log_level == "DEBUG"
    assert settings.otel_service_name == "gridiron-prod"
    assert settings.otel_traces_exporter == "otlp"
    assert settings.otel_exporter_otlp_endpoint == "http://otel-collector:4318/v1/traces"
    assert settings.otel_exporter_otlp_protocol == "http/protobuf"
    assert settings.otel_exporter_otlp_headers == "authorization=Bearer prod-token"
    assert settings.otel_exporter_otlp_insecure is False


def test_rate_limiter_uses_remote_address():
    from slowapi.util import get_remote_address

    from backend.app.core.rate_limit import limiter

    assert limiter._key_func is get_remote_address


def test_request_logging_middleware_sets_request_id_and_logs(caplog):
    from backend.app.middleware import RequestLoggingMiddleware

    app = FastAPI()

    @app.get("/ping")
    def ping():
        return Response(status_code=204)

    app.add_middleware(RequestLoggingMiddleware)

    with TestClient(app) as client, caplog.at_level(
        logging.INFO, logger="backend.app.middleware"
    ):
        response = client.get("/ping", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 204
    assert response.headers["X-Request-ID"] == "req-123"
    assert "req-123" in caplog.text
    assert "/ping" in caplog.text


def test_alert_service_publish_edge_alert_rounds_and_orders_recent():
    from backend.app.services.alert import AlertService

    AlertService._instance = None
    svc = AlertService()
    svc.publish_system("Older", "first")
    svc.publish_edge_alert("p1", "Player One", "targets", 0.12349, "edge body")

    recent = svc.recent(2)

    assert recent[0].title == "Edge Signal: Player One targets"
    assert recent[0].value == 0.1235
    assert recent[1].title == "Older"


def test_alert_service_publish_injury_records_alert_fields():
    from datetime import timezone

    from backend.app.services.alert import AlertService, AlertSeverity

    AlertService._instance = None
    svc = AlertService()

    svc.publish_injury("p1", "Player One", "Did not practice")

    recent = svc.recent(1)
    assert len(recent) == 1
    assert recent[0].severity == AlertSeverity.INJURY
    assert recent[0].id.startswith("inj-")
    assert recent[0].player_id == "p1"
    assert recent[0].player_name == "Player One"
    assert "Player One" in recent[0].title
    assert recent[0].body == "Did not practice"
    assert recent[0].timestamp is not None
    assert recent[0].timestamp.tzinfo is timezone.utc


def test_alert_service_recent_default_limit_is_fifty():
    from backend.app.services.alert import AlertService

    AlertService._instance = None
    svc = AlertService()

    for idx in range(60):
        svc.publish_system(f"Alert {idx}", f"body-{idx}")

    recent = svc.recent()

    assert len(recent) == 50
    assert recent[0].title == "Alert 59"
    assert recent[-1].title == "Alert 10"


def test_alert_service_start_persistence_sets_state_and_stops_cleanly(monkeypatch):
    from backend.app.services.alert import AlertService

    AlertService._instance = None
    svc = AlertService()

    calls: list[str] = []
    created_tasks: list[object] = []

    def fake_ensure_alerts_table(db_url: str) -> None:
        calls.append(db_url)

    class FakeTask:
        def cancel(self) -> None:
            calls.append("cancel")

        def __await__(self):
            async def _done():
                return None

            return _done().__await__()

    def fake_create_task(coro):
        created_tasks.append(coro)
        coro.close()
        return FakeTask()

    monkeypatch.setattr("backend.app.services.alert._ensure_alerts_table", fake_ensure_alerts_table)
    monkeypatch.setattr("backend.app.services.alert.asyncio.create_task", fake_create_task)

    async def scenario() -> None:
        await svc.start_persistence("postgresql://oracle:oracle@localhost:15439/oracle")
        assert svc._db_url == "postgresql://oracle:oracle@localhost:15439/oracle"
        assert svc._drain_task is not None
        assert calls[0] == "postgresql://oracle:oracle@localhost:15439/oracle"
        assert len(created_tasks) == 1
        await svc.stop_persistence()
        assert calls[-1] == "cancel"

    asyncio.run(scenario())


def test_alert_service_drain_loop_enters_before_cancellation(monkeypatch):
    from backend.app.services.alert import AlertService

    AlertService._instance = None
    svc = AlertService()

    async def fake_sleep(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr("backend.app.services.alert.asyncio.sleep", fake_sleep)

    async def scenario() -> None:
        with suppress(asyncio.CancelledError):
            await svc._drain_loop()

    asyncio.run(scenario())


def test_alert_service_subscribe_yields_published_alert():
    from backend.app.services.alert import AlertService

    AlertService._instance = None
    svc = AlertService()

    async def scenario() -> None:
        stream = svc.subscribe()
        next_alert = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0)
        svc.publish_system("Ready", "service online")
        alert = await asyncio.wait_for(next_alert, timeout=1)
        assert alert.title == "Ready"
        assert alert.body == "service online"
        await stream.aclose()

    asyncio.run(scenario())


def test_exposure_manager_enforces_player_game_and_total_limits():
    from backend.app.services.exposure import ExposureDecision, PythonExposureManager

    PythonExposureManager._instance = None
    manager = PythonExposureManager(bankroll=1000.0)

    assert manager.try_place_bet("p1", "g1", 40.0) == ExposureDecision.APPROVED
    assert manager.try_place_bet("p1", "g1", 20.0) == ExposureDecision.REJECTED_PLAYER
    assert manager.try_place_bet("p2", "g1", 45.0) == ExposureDecision.APPROVED
    assert manager.try_place_bet("p3", "g1", 45.0) == ExposureDecision.APPROVED
    assert manager.try_place_bet("p4", "g1", 45.0) == ExposureDecision.APPROVED
    assert manager.try_place_bet("p5", "g1", 30.0) == ExposureDecision.REJECTED_GAME

    assert manager.try_place_bet("p6", "g2", 20.0) == ExposureDecision.APPROVED
    assert manager.try_place_bet("p7", "g3", 10.0) == ExposureDecision.REJECTED_TOTAL

    manager.release_bet("p1", "g1", 10.0)
    player_exp, game_exp, total_in_play = manager.get_exposures()
    assert player_exp["p1"] == 30.0
    assert game_exp["g1"] == 165.0
    assert total_in_play == 185.0


def test_edge_coordinator_approved_path_publishes_and_enqueues():
    from backend.app.services.edge_coordinator import EdgeSignalCoordinator

    alerts = Mock()
    exposure = Mock()
    exposure.try_place_bet.return_value = "APPROVED"
    ipc = Mock()
    ipc.send_event.return_value = True

    coordinator = EdgeSignalCoordinator(
        alert_service=alerts,
        exposure_manager=exposure,
        ipc_client=ipc,
    )

    ok = coordinator.handle(
        player_id="p1",
        player_name="Player One",
        stat="targets",
        edge_pct=0.12,
        body="12% edge",
        game_id="2025_01_MIN_GB",
        position="WR",
    )

    assert ok is True
    ipc.send_event.assert_called_once()
    alerts.publish_edge_alert.assert_called_once()


def test_edge_coordinator_releases_exposure_on_ipc_failure():
    from backend.app.services.edge_coordinator import EdgeSignalCoordinator

    alerts = Mock()
    exposure = Mock()
    exposure.try_place_bet.return_value = "APPROVED"
    ipc = Mock()
    ipc.send_event.return_value = False

    coordinator = EdgeSignalCoordinator(
        alert_service=alerts,
        exposure_manager=exposure,
        ipc_client=ipc,
    )

    ok = coordinator.handle(
        player_id="p1",
        player_name="Player One",
        stat="targets",
        edge_pct=0.12,
        body="12% edge",
        game_id="2025_01_MIN_GB",
        position="WR",
    )

    assert ok is False
    exposure.release_bet.assert_called_once_with("p1", "2025_01_MIN_GB", 50.0)
    alerts.publish_edge_alert.assert_not_called()


def test_ipc_client_send_event_requires_running_state():
    from backend.app.services.ipc_client import IpcClient

    IpcClient._instance = None
    client = IpcClient("/tmp/gridiron-test.sock")

    assert client.send_event({"player_id": "p1"}) is False


def test_ipc_client_send_event_handles_queue_full_and_bad_payload():
    from backend.app.services.ipc_client import IpcClient

    IpcClient._instance = None
    client = IpcClient("/tmp/gridiron-test.sock")
    client._running = True
    client._queue = asyncio.Queue(maxsize=1)

    assert client.send_event({"player_id": "p1"}) is True
    assert client.send_event({"player_id": "p2"}) is False
    assert client.send_event({"bad": {1, 2, 3}}) is False
