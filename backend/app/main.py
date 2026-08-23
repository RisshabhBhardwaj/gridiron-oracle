"""
backend/app/main.py

FastAPI application entry point for Gridiron Oracle.

Registers:
  - All API routers (predict, explain, scenario, backtest, settings, alerts)
  - CORS middleware (configured via CORS_ORIGINS env var)
  - APScheduler background jobs (data freshness alert, retrain trigger)
  - Startup / shutdown lifespan events

Run locally (without Docker):
    uvicorn backend.app.main:app --reload --port 8000

Via Docker Compose:
    make up  →  http://localhost:18017
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.core.config import settings
from backend.app.core.rate_limit import limiter
from backend.app.core.tracing import configure_tracing, describe_tracing, force_flush_tracing
from backend.app.middleware import RequestLoggingMiddleware
from backend.app.services.runtime_status import RuntimeStatusService

# ---------------------------------------------------------------------------
# API key middleware
# ---------------------------------------------------------------------------

# Paths that never require authentication (health probes, OpenAPI schema, docs UI).
# /metrics is intentionally NOT exempt: when API_KEY is set it requires auth.
# Configure Prometheus scrape headers (prometheus.yml) to include X-API-Key.
# When API_KEY is unset (local dev) all endpoints are open, including /metrics.
#
# Only the two probe endpoints are exempt. Everything else that used to be open
# describes the deployment to anyone who asks, which was harmless while the API
# bound to localhost and is not now that it is on a public domain:
#   /integrity   names the deployed git commit, the baseline manifest's absolute
#                path, and the database host
#   /openapi.json, /docs, /redoc
#                publish the full route table, query parameters and response
#                schemas — a map of the API for anyone probing it
# Railway's liveness probe uses "/" and readiness uses /health, so nothing
# operational depends on the rest being open. Reaching /docs in a browser now
# requires a key the browser cannot attach; use a local instance (API_KEY unset)
# to read the docs, or curl /openapi.json with X-API-Key.
_AUTH_EXEMPT_PATHS = frozenset({"/", "/health"})

# Write paths that require X-Admin-Key (falls back to X-API-Key when ADMIN_API_KEY unset).
_ADMIN_PATHS = frozenset({"/settings"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Optional X-API-Key / X-Admin-Key guard.

    Active only when the API_KEY env var is set. When active:
      - Every non-exempt request must include X-API-Key: <API_KEY>
      - Requests to admin paths (PUT /settings) must also include
        X-Admin-Key: <ADMIN_API_KEY>. When ADMIN_API_KEY is unset,
        the regular API_KEY is accepted on admin paths (local dev).

    Exempt paths (always allowed): / /health /integrity /docs /redoc /openapi.json /metrics
    Admin paths (require X-Admin-Key): /settings
    """

    async def dispatch(self, request: Request, call_next):
        if settings.api_key and request.url.path not in _AUTH_EXEMPT_PATHS:
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided, settings.api_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-API-Key header"},
                )

            # Extra guard for write endpoints: require X-Admin-Key when configured.
            if request.url.path in _ADMIN_PATHS and settings.admin_api_key:
                provided_admin = request.headers.get("X-Admin-Key", "")
                if not hmac.compare_digest(provided_admin, settings.admin_api_key):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Invalid or missing X-Admin-Key header"},
                    )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Startup:
      - Log configuration summary
      - Start APScheduler background jobs
      - Publish a system startup alert
      - Start IPC client

    Shutdown:
      - Stop scheduler gracefully
      - Stop IPC client
    """
    tracing_state = configure_tracing(settings)
    logger.info(
        "Tracing configured — service=%s requested=%s active=%s",
        tracing_state["service_name"],
        ",".join(tracing_state["requested_exporters"]),
        ",".join(tracing_state["active_exporters"]) or "none",
    )
    if tracing_state["last_error"]:
        logger.warning("Tracing configuration issue: %s", tracing_state["last_error"])

    if not settings.api_key:
        logger.warning(
            "API_KEY is not set — authentication is DISABLED. "
            "Set the API_KEY environment variable before exposing this service."
        )
    logger.info(
        "Gridiron Oracle API starting — db=%s model_version=%s api_key=%s",
        settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured",
        settings.model_version,
        "enabled" if settings.api_key else "disabled",
    )

    scheduler = _start_scheduler()

    from backend.app.api.alerts import get_alert_service
    alert_svc = get_alert_service()
    await alert_svc.start_persistence(settings.database_url)

    from backend.app.services.ipc_client import ipc_client

    def _on_engine_disconnect(reason: str) -> None:
        alert_svc.publish_system(
            title="C++ Engine Disconnected",
            body=f"{reason}. Season projections may be unavailable. "
                 "Run `make engine && ./engine/build/engine engine/config.json` to restart.",
        )

    ipc_client.set_disconnect_callback(_on_engine_disconnect)
    await ipc_client.start()
    alert_svc.publish_system(
        title="Gridiron Oracle started",
        body=f"API online at {datetime.now(timezone.utc).isoformat()}",
    )

    yield  # ← server is running here

    logger.info("Gridiron Oracle API shutting down")
    if scheduler is not None:
        scheduler.shutdown(wait=False)
    await alert_svc.stop_persistence()
    await ipc_client.stop()
    if not force_flush_tracing():
        logger.debug(
            "Tracing force_flush unavailable — requested=%s active=%s",
            ",".join(describe_tracing(settings)["requested_exporters"]),
            ",".join(describe_tracing(settings)["active_exporters"]) or "none",
        )


def _start_scheduler():
    """
    Start APScheduler background jobs.

    Jobs:
      - data_freshness_check: hourly — warn if feature_matrix is stale (>24h)
      - (retrain trigger: only in production, gated by env flag)

    Returns the scheduler instance so lifespan can shut it down cleanly.
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler()

        # Hourly data freshness check
        scheduler.add_job(
            _check_data_freshness,
            "interval",
            hours=1,
            id="data_freshness_check",
            replace_existing=True,
        )

        # Daily alerts table pruning — delete rows older than 30 days
        scheduler.add_job(
            _prune_old_alerts,
            "interval",
            hours=24,
            id="alerts_pruning",
            replace_existing=True,
        )

        # Forecasts are useful only when their capture timestamp is retained.
        # The job is a no-op unless OPENWEATHER_API_KEY is configured.
        scheduler.add_job(
            _capture_pregame_weather_forecasts,
            "interval",
            hours=6,
            id="pregame_weather_forecasts",
            replace_existing=True,
        )

        scheduler.start()
        logger.info("APScheduler started with %d jobs", len(scheduler.get_jobs()))
        return scheduler

    except ImportError:
        logger.warning("apscheduler not installed — background jobs disabled")
        return None
    except Exception as exc:
        logger.error("Scheduler start failed: %s — continuing without scheduler", exc)
        return None


def _sync_check_data_freshness() -> None:
    """Synchronous DB work for the data freshness check — run via asyncio.to_thread."""
    import psycopg2
    from datetime import timedelta

    from backend.app.api.alerts import get_alert_service

    conn = psycopg2.connect(settings.database_url)
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(computed_at) FROM feature_matrix")
        row = cur.fetchone()
    finally:
        conn.close()

    if row and row[0]:
        latest: datetime = row[0]
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - latest
        if age > timedelta(hours=24):
            get_alert_service().publish_system(
                title="Data Freshness Warning",
                body=(
                    f"Feature matrix last updated {age.days}d "
                    f"{age.seconds // 3600}h ago. "
                    "Run `make ingest` to refresh."
                ),
            )
            logger.warning("Feature matrix is stale: last update %s", latest.isoformat())


async def _check_data_freshness() -> None:
    """
    Background job: check whether feature_matrix data is fresh (< 24h).

    Offloads the synchronous psycopg2 call to a thread pool via asyncio.to_thread
    so a DB hang cannot block the APScheduler event loop.
    """
    try:
        await asyncio.to_thread(_sync_check_data_freshness)
    except Exception as exc:
        logger.debug("Data freshness check failed (DB may not be ready): %s", exc)


def _sync_capture_pregame_weather_forecasts() -> int:
    """Persist timestamped forecast snapshots for games in the provider window."""
    if not os.environ.get("OPENWEATHER_API_KEY"):
        return 0
    from scraper.adapters.weather_adapter import capture_pregame_weather_forecasts

    return capture_pregame_weather_forecasts(settings.database_url)


async def _capture_pregame_weather_forecasts() -> None:
    """Run weather capture off the event loop; failure must not stop the API."""
    try:
        captured = await asyncio.to_thread(_sync_capture_pregame_weather_forecasts)
        if captured:
            logger.info("Captured %d pregame weather forecast snapshots", captured)
    except Exception as exc:
        logger.warning("Pregame weather forecast capture failed: %s", exc)


def _sync_prune_old_alerts() -> int:
    """Synchronous DB work for alert pruning — run via asyncio.to_thread."""
    import psycopg2

    conn = psycopg2.connect(settings.database_url)
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM alerts WHERE created_at < NOW() - INTERVAL '30 days'"
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


async def _prune_old_alerts() -> None:
    """
    Background job: delete alerts older than 30 days from the alerts table.

    Runs daily. Offloads the synchronous psycopg2 call to a thread pool via
    asyncio.to_thread so a DB hang cannot block the APScheduler event loop.
    """
    try:
        deleted = await asyncio.to_thread(_sync_prune_old_alerts)
        if deleted:
            logger.info("AlertService pruning: removed %d alert(s) older than 30 days", deleted)
    except Exception as exc:
        logger.debug("Alert pruning failed (DB may not be ready): %s", exc)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Gridiron Oracle API",
    description=(
        "NFL player projection platform. "
        "Kalman filter → XGB/LGB/TFT stacking → Bayesian uncertainty → Monte Carlo."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware (registration order: outermost first)
# ---------------------------------------------------------------------------

# Rate limiter state — must be set before SlowAPIMiddleware is added.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Structured access log + X-Request-ID on every response.
app.add_middleware(RequestLoggingMiddleware)

# IP-based rate limiting (limits defined per-endpoint in route modules).
app.add_middleware(SlowAPIMiddleware)

# API key guard — no-op when API_KEY env var is unset (local dev).
app.add_middleware(APIKeyMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health", "/docs", "/redoc", "/openapi.json"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus /metrics endpoint enabled")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed — /metrics unavailable")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from backend.app.api import (  # noqa: E402
    predict,
    explain,
    scenario,
    backtest,
    settings as settings_router,
    alerts,
    season as season_router,
    draft as draft_router,
    mock_draft as mock_draft_router,
    team_game as team_game_router,
    drive as drive_router,
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(scenario.router)
app.include_router(backtest.router)
app.include_router(settings_router.router)
app.include_router(alerts.router)
app.include_router(season_router.router)
app.include_router(draft_router.router)
app.include_router(mock_draft_router.router)
app.include_router(team_game_router.router)
app.include_router(drive_router.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> Response:
    """
    Readiness probe for Docker / load balancers.

    Returns:
      200 OK       — overall_status is "ok" or "degraded" (fallback allowed)
      503 Service  — overall_status is "blocked" (dependency unavailable)

    Docker healthcheck uses `curl -f`, which treats 5xx as failure, so a
    blocked instance is correctly removed from rotation.
    """
    report = RuntimeStatusService(
        db_url=settings.database_url,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        model_version=settings.model_version,
    ).build_report()
    body = {
        "status": report["overall_status"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": settings.model_version,
        "product_mode": settings.product_mode,
        "fallback_allowed": report["fallback_allowed"],
        "artifact_mode_ready": report["artifact_mode_ready"],
    }
    status_code = 503 if report["overall_status"] == "blocked" else 200
    return JSONResponse(content=body, status_code=status_code)


@app.get("/integrity", tags=["meta"])
def integrity() -> dict:
    """Operational integrity report for release readiness and runtime checks."""
    return RuntimeStatusService(
        db_url=settings.database_url,
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        model_version=settings.model_version,
    ).build_report()


@app.get("/", tags=["meta"])
def root() -> dict:
    """API root — redirect hint for browser users."""
    return {
        "message": "Gridiron Oracle API — see /docs for interactive documentation",
        "docs":    "/docs",
        "health":  "/health",
    }
