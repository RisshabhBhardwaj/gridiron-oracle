"""
backend/app/core/tracing.py

OpenTelemetry-compatible tracing helpers for request and pipeline auditing.

Goals:
  - Preserve incoming W3C trace context (`traceparent`, `tracestate`) when present.
  - Generate valid trace/span identifiers even when no exporter is configured.
  - Expose lightweight helpers so API middleware and ML pipeline code can create
    nested spans without depending directly on OpenTelemetry setup details.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from opentelemetry import trace
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_propagator = TraceContextTextMapPropagator()
_provider_configured = False
_configured_signature: tuple[Any, ...] | None = None
_active_exporters: tuple[str, ...] = ()
_last_config_error: str | None = None

_DEFAULT_SERVICE_NAME = "gridiron-oracle-api"


@dataclass(frozen=True)
class TracingConfig:
    service_name: str
    exporters: tuple[str, ...]
    otlp_endpoint: str
    otlp_protocol: str
    otlp_headers: tuple[tuple[str, str], ...]
    otlp_insecure: bool
    deployment_environment: str
    service_version: str

    @property
    def otlp_headers_dict(self) -> dict[str, str]:
        return dict(self.otlp_headers)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_otlp_headers(raw: str | None) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()

    headers: list[tuple[str, str]] = []
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers.append((key, value))
    return tuple(headers)


def _normalize_exporters(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        tokens = ["none"]
    elif isinstance(raw, str):
        tokens = [part.strip().lower() for part in raw.split(",")]
    else:
        tokens = [str(part).strip().lower() for part in raw]

    cleaned = [token for token in tokens if token]
    if not cleaned:
        return ("none",)
    if len(cleaned) > 1:
        cleaned = [token for token in cleaned if token != "none"]
    return tuple(cleaned or ["none"])


def build_tracing_config(settings: Any) -> TracingConfig:
    """Build a normalized tracing config from a settings-like object."""
    return TracingConfig(
        service_name=(getattr(settings, "otel_service_name", "") or _DEFAULT_SERVICE_NAME).strip(),
        exporters=_normalize_exporters(getattr(settings, "otel_traces_exporter", "none")),
        otlp_endpoint=(getattr(settings, "otel_exporter_otlp_endpoint", "") or "").strip(),
        otlp_protocol=(
            getattr(settings, "otel_exporter_otlp_protocol", "http/protobuf") or "http/protobuf"
        ).strip().lower(),
        otlp_headers=_parse_otlp_headers(getattr(settings, "otel_exporter_otlp_headers", "")),
        otlp_insecure=_parse_bool(getattr(settings, "otel_exporter_otlp_insecure", False)),
        deployment_environment=(getattr(settings, "product_mode", "unknown") or "unknown").strip(),
        service_version=(getattr(settings, "model_version", "latest") or "latest").strip(),
    )


def _provider_signature(config: TracingConfig) -> tuple[Any, ...]:
    return (
        config.service_name,
        config.exporters,
        config.otlp_endpoint,
        config.otlp_protocol,
        config.otlp_headers,
        config.otlp_insecure,
        config.deployment_environment,
        config.service_version,
    )


def _build_resource(
    *,
    service_name: str,
    deployment_environment: str,
    service_version: str,
) -> Any:
    from opentelemetry.sdk.resources import Resource

    return Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": deployment_environment,
        }
    )


def _ensure_tracer_provider(
    *,
    service_name: str = _DEFAULT_SERVICE_NAME,
    deployment_environment: str = "unknown",
    service_version: str = "latest",
) -> None:
    """
    Install a local SDK tracer provider if only the proxy/no-op provider exists.

    This keeps trace IDs/span IDs valid even when no exporter is configured yet.
    If an external provider is already installed, it is left untouched.
    """
    global _provider_configured
    if _provider_configured:
        return

    provider = trace.get_tracer_provider()
    if provider.__class__.__name__ != "ProxyTracerProvider":
        _provider_configured = True
        return

    try:
        from opentelemetry.sdk.trace import TracerProvider
    except Exception:
        # Fallback to the active provider as-is. In that mode spans may be
        # non-recording, but propagation helpers still remain import-safe.
        _provider_configured = True
        return

    trace.set_tracer_provider(
        TracerProvider(
            resource=_build_resource(
                service_name=service_name,
                deployment_environment=deployment_environment,
                service_version=service_version,
            )
        )
    )
    _provider_configured = True


def configure_tracing(settings: Any) -> dict[str, Any]:
    """
    Configure exporters/processors for tracing from runtime settings.

    Safe to call multiple times. Repeated calls with the same config are no-ops.
    """
    global _configured_signature, _active_exporters, _last_config_error

    config = build_tracing_config(settings)
    signature = _provider_signature(config)
    if _configured_signature == signature:
        return describe_tracing(settings)

    _ensure_tracer_provider(
        service_name=config.service_name,
        deployment_environment=config.deployment_environment,
        service_version=config.service_version,
    )

    provider = trace.get_tracer_provider()
    added_exporters: list[str] = list(_active_exporters)
    errors: list[str] = []

    for exporter_name in config.exporters:
        if exporter_name == "none":
            continue
        if exporter_name in added_exporters:
            continue

        try:
            if exporter_name == "console":
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
                added_exporters.append("console")
            elif exporter_name == "otlp":
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                if config.otlp_protocol == "grpc":
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    exporter = OTLPSpanExporter(
                        endpoint=config.otlp_endpoint or None,
                        headers=config.otlp_headers_dict or None,
                        insecure=config.otlp_insecure,
                    )
                else:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    exporter = OTLPSpanExporter(
                        endpoint=config.otlp_endpoint or None,
                        headers=config.otlp_headers_dict or None,
                    )

                provider.add_span_processor(BatchSpanProcessor(exporter))
                added_exporters.append("otlp")
            else:
                errors.append(f"Unsupported tracing exporter '{exporter_name}'")
        except ModuleNotFoundError as exc:
            errors.append(
                f"Tracing exporter '{exporter_name}' requested but dependency is not installed: {exc}"
            )
        except Exception as exc:
            errors.append(f"Tracing exporter '{exporter_name}' failed to configure: {exc}")

    _configured_signature = signature
    _active_exporters = tuple(added_exporters)
    _last_config_error = "; ".join(errors) if errors else None

    if _last_config_error:
        logger.warning(_last_config_error)

    return describe_tracing(settings)


def force_flush_tracing(timeout_millis: int = 3000) -> bool:
    """Force-flush pending spans when the active provider supports it."""
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if callable(force_flush):
        try:
            return bool(force_flush(timeout_millis=timeout_millis))
        except TypeError:
            return bool(force_flush())
        except Exception as exc:
            logger.debug("Tracing force_flush failed: %s", exc)
            return False
    return False


def describe_tracing(settings: Any) -> dict[str, Any]:
    """Return the normalized tracing config plus exporter activation state."""
    config = build_tracing_config(settings)
    return {
        "service_name": config.service_name,
        "requested_exporters": list(config.exporters),
        "active_exporters": list(_active_exporters),
        "otlp_endpoint": config.otlp_endpoint or None,
        "otlp_protocol": config.otlp_protocol,
        "otlp_headers": dict(config.otlp_headers),
        "otlp_insecure": config.otlp_insecure,
        "deployment_environment": config.deployment_environment,
        "service_version": config.service_version,
        "provider_configured": _provider_configured,
        "last_error": _last_config_error,
    }


def bind_request_id(request_id: str) -> Any:
    """Bind a request ID into the current context and return the reset token."""
    return _request_id_var.set(request_id)


def reset_request_id(token: Any) -> None:
    """Restore the previous request ID context."""
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    """Return the current request ID, if any."""
    return _request_id_var.get()


def get_current_trace_id() -> str | None:
    """Return the active trace ID as 32 lowercase hex chars, if valid."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return f"{ctx.trace_id:032x}"


def get_current_span_id() -> str | None:
    """Return the active span ID as 16 lowercase hex chars, if valid."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None
    return f"{ctx.span_id:016x}"


def inject_current_context(carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Inject the current trace context into a mutable carrier."""
    inject(carrier)
    return carrier


def current_trace_headers() -> dict[str, str]:
    """Return the current W3C trace propagation headers."""
    carrier: dict[str, str] = {}
    inject_current_context(carrier)
    return carrier


def _coerce_attribute(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        coerced = [_coerce_attribute(v) for v in value]
        if all(isinstance(v, (bool, int, float, str)) for v in coerced):
            return coerced
    return str(value)


@contextmanager
def start_span(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    tracer_name: str = "gridiron_oracle",
) -> Iterator[Any]:
    """
    Start a span as current context and emit concise audit logs around it.

    `headers` is used only when starting a new root span from inbound request
    metadata. Internal callers should omit it so the current context is reused.
    """
    _ensure_tracer_provider()

    context = _propagator.extract(headers) if headers is not None else None
    tracer = trace.get_tracer(tracer_name)
    start = time.perf_counter()

    with tracer.start_as_current_span(
        name,
        context=context,
        kind=kind,
        record_exception=True,
        set_status_on_exception=True,
    ) as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, _coerce_attribute(value))

        trace_id = get_current_trace_id() or "-"
        span_id = get_current_span_id() or "-"
        request_id = get_request_id() or "-"
        logger.info(
            "trace.start %s trace_id=%s span_id=%s request_id=%s",
            name,
            trace_id,
            span_id,
            request_id,
        )
        try:
            yield span
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "trace.end %s trace_id=%s span_id=%s request_id=%s duration_ms=%.1f",
                name,
                trace_id,
                span_id,
                request_id,
                elapsed_ms,
            )
