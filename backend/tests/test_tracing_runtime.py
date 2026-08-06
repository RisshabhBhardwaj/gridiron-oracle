from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.tracing import (
    build_tracing_config,
    configure_tracing,
    describe_tracing,
    get_current_span_id,
    get_current_trace_id,
)
from backend.app.main import app
from backend.app.services.projection import ProjectionResult
from backend.app.services.runtime_status import RuntimeStatusService

client = TestClient(app, raise_server_exceptions=True)


def _mock_projection_result() -> ProjectionResult:
    return ProjectionResult(
        player_id="00-0035228",
        player_name="Justin Jefferson",
        position="WR",
        team="MIN",
        week=12,
        season=2025,
        stat="receiving_yards",
        projection=74.2,
        floor=38.5,
        ceiling=121.8,
        boom_probability=0.28,
        bust_probability=0.15,
        fantasy_projection=15.4,
        fantasy_floor=7.9,
        fantasy_ceiling=25.6,
        kalman_ability_estimate=88.1,
        kalman_uncertainty=6.4,
        confidence_score=0.72,
        prop_comparison=None,
        data_freshness=datetime(2025, 11, 19, 9, 14, tzinfo=timezone.utc),
        model_version="test-v1",
    )


def test_predict_response_exposes_trace_headers_and_service_sees_same_trace():
    captured: dict[str, str | None] = {}

    def fake_get_projection(self, player, week, season, stat="receiving_yards"):
        captured["trace_id"] = get_current_trace_id()
        captured["span_id"] = get_current_span_id()
        return _mock_projection_result()

    with patch(
        "backend.app.api.predict.ProjectionService.get_projection",
        autospec=True,
        side_effect=fake_get_projection,
    ), patch(
        "backend.app.api.predict._build_shap_factors",
        return_value=[],
    ):
        response = client.get("/predict?player=Jefferson&week=12&season=2025")

    assert response.status_code == 200
    assert captured["trace_id"] is not None
    assert captured["span_id"] is not None
    assert response.headers["X-Trace-ID"] == captured["trace_id"]
    assert response.headers["traceparent"].split("-")[1] == captured["trace_id"]


def test_predict_preserves_incoming_trace_id_from_traceparent():
    incoming_trace_id = "1" * 32
    incoming_span_id = "2" * 16
    captured: dict[str, str | None] = {}

    def fake_get_projection(self, player, week, season, stat="receiving_yards"):
        captured["trace_id"] = get_current_trace_id()
        captured["span_id"] = get_current_span_id()
        return _mock_projection_result()

    with patch(
        "backend.app.api.predict.ProjectionService.get_projection",
        autospec=True,
        side_effect=fake_get_projection,
    ), patch(
        "backend.app.api.predict._build_shap_factors",
        return_value=[],
    ):
        response = client.get(
            "/predict?player=Jefferson&week=12&season=2025",
            headers={"traceparent": f"00-{incoming_trace_id}-{incoming_span_id}-01"},
        )

    assert response.status_code == 200
    assert captured["trace_id"] == incoming_trace_id
    assert response.headers["X-Trace-ID"] == incoming_trace_id
    traceparent = response.headers["traceparent"].split("-")
    assert traceparent[1] == incoming_trace_id
    assert traceparent[2] != incoming_span_id


def test_build_tracing_config_normalizes_exporters_and_headers():
    cfg = build_tracing_config(
        SimpleNamespace(
            otel_service_name="gridiron-audit",
            otel_traces_exporter="none, console , otlp",
            otel_exporter_otlp_endpoint="http://collector:4318/v1/traces",
            otel_exporter_otlp_protocol="grpc",
            otel_exporter_otlp_headers="authorization=Bearer abc,x-team=oracle",
            otel_exporter_otlp_insecure=True,
            product_mode="artifact_backed",
            model_version="2026.05.02",
        )
    )

    assert cfg.service_name == "gridiron-audit"
    assert cfg.exporters == ("console", "otlp")
    assert cfg.otlp_endpoint == "http://collector:4318/v1/traces"
    assert cfg.otlp_protocol == "grpc"
    assert cfg.otlp_headers_dict == {
        "authorization": "Bearer abc",
        "x-team": "oracle",
    }
    assert cfg.otlp_insecure is True
    assert cfg.deployment_environment == "artifact_backed"
    assert cfg.service_version == "2026.05.02"


@pytest.mark.network
def test_configure_tracing_otlp_activates_exporter_when_dependency_present():
    settings = SimpleNamespace(
        otel_service_name="gridiron-audit",
        otel_traces_exporter="otlp",
        otel_exporter_otlp_endpoint="http://collector:4318/v1/traces",
        otel_exporter_otlp_protocol="http/protobuf",
        otel_exporter_otlp_headers="",
        otel_exporter_otlp_insecure=False,
        product_mode="graceful_fallback",
        model_version="test-v1",
    )

    state = configure_tracing(settings)

    assert state["service_name"] == "gridiron-audit"
    assert state["requested_exporters"] == ["otlp"]
    assert "otlp" in state["active_exporters"]
    assert state["last_error"] is None


def test_runtime_status_tracing_check_reports_local_only_mode():
    service = RuntimeStatusService(
        db_url="postgresql://example.invalid/db",
        mlflow_tracking_uri="http://example.invalid:5001",
        model_version="test-v1",
    )

    with patch(
        "backend.app.services.runtime_status.describe_tracing",
        return_value={
            "service_name": "gridiron-oracle-api",
            "requested_exporters": ["none"],
            "active_exporters": [],
            "otlp_protocol": "http/protobuf",
            "otlp_endpoint": None,
            "last_error": None,
        },
    ):
        result = service._check_tracing()

    assert result["status"] == "ok"
    assert result["detail"] == "Trace propagation enabled; exporter disabled"


def test_runtime_status_tracing_check_warns_when_collector_unreachable():
    service = RuntimeStatusService(
        db_url="postgresql://example.invalid/db",
        mlflow_tracking_uri="http://example.invalid:5001",
        model_version="test-v1",
    )

    with patch(
        "backend.app.services.runtime_status.describe_tracing",
        return_value={
            "service_name": "gridiron-oracle-api",
            "requested_exporters": ["otlp"],
            "active_exporters": ["otlp"],
            "otlp_protocol": "http/protobuf",
            "otlp_endpoint": "http://collector.invalid:4318/v1/traces",
            "last_error": None,
        },
    ), patch(
        "backend.app.services.runtime_status.socket.create_connection",
        side_effect=OSError("connection refused"),
    ):
        result = service._check_tracing()

    assert result["status"] == "warn"
    assert result["detail"] == "Tracing exporter configured but collector unreachable"
    assert result["observed"]["collector_reachability_error"] == "connection refused"
