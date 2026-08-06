"""
backend/app/middleware.py

Shared Starlette middleware for the Gridiron Oracle API.

Registered in main.py (outermost to innermost):
  1. RequestLoggingMiddleware — structured access log + X-Request-ID
  2. APIKeyMiddleware         — optional X-API-Key guard
  3. CORSMiddleware           — origin allow-list
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.app.core.tracing import (
    bind_request_id,
    current_trace_headers,
    get_current_trace_id,
    reset_request_id,
    start_span,
)

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured access log for every HTTP request.

    Adds X-Request-ID to every response (generated if the client didn't
    supply one). Logs at INFO level:
        METHOD /path → status  latency_ms ms  [request_id]

    Latency is wall-clock time from first byte received to response sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start = time.perf_counter()
        token = bind_request_id(request_id)

        try:
            with start_span(
                f"HTTP {request.method} {request.url.path}",
                headers=request.headers,
                attributes={
                    "http.method": request.method,
                    "http.route": request.url.path,
                    "http.scheme": request.url.scheme,
                    "request.id": request_id,
                },
                tracer_name="backend.app.middleware",
            ):
                request.state.request_id = request_id
                request.state.trace_id = get_current_trace_id()

                response: Response = await call_next(request)

                latency_ms = (time.perf_counter() - start) * 1000
                response.headers["X-Request-ID"] = request_id

                trace_headers = current_trace_headers()
                if "traceparent" in trace_headers:
                    response.headers["traceparent"] = trace_headers["traceparent"]
                if "tracestate" in trace_headers:
                    response.headers["tracestate"] = trace_headers["tracestate"]
                if request.state.trace_id:
                    response.headers["X-Trace-ID"] = request.state.trace_id

                logger.info(
                    "%s %s → %d  %.1fms  [%s] trace_id=%s",
                    request.method,
                    request.url.path,
                    response.status_code,
                    latency_ms,
                    request_id,
                    request.state.trace_id or "-",
                )

                return response
        finally:
            reset_request_id(token)
