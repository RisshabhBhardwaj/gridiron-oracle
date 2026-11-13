"""
backend/app/core/config.py

Central settings loaded from environment variables / .env file.
All consumers import the singleton `settings`.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from ml.product_mode import resolve_product_mode

_log = logging.getLogger(__name__)

_ODDS_FLOOR_DEFAULT = 50


def _env(name: str, default: str) -> str:
    """Read a string setting: missing, empty, or whitespace-only values all
    fall back to the default. (A copied `.env.example` full of `KEY=` blanks
    must behave like those keys being absent, not like empty paths/URLs.)"""
    raw = os.environ.get(name, "")
    return raw if raw.strip() else default


def _parse_odds_floor(raw: str) -> int:
    """Parse ODDS_MIN_REMAINING. Garbage falls back to the default; a floor
    below zero is meaningless, so it becomes zero (never trip on unknown)."""
    try:
        return max(int(raw.strip() or str(_ODDS_FLOOR_DEFAULT)), 0)
    except (ValueError, AttributeError):
        return _ODDS_FLOOR_DEFAULT


class Settings:
    """
    Application settings read from environment variables.

    Required:
        DATABASE_URL: PostgreSQL DSN, e.g.
            postgresql://oracle:oracle@localhost:15439/oracle

    Optional:
        MLFLOW_TRACKING_URI: default http://localhost:15091 (host process)
        MODEL_VERSION:       tag stored in every /predict response
        PRODUCT_MODE:        artifact_backed (default) | graceful_fallback (explicit
                             opt-in). Any other value raises ValueError.
        SETTINGS_FILE:       path to user_config.json (weight overrides)
        LOG_LEVEL:           default "INFO"
        CORS_ORIGINS:        comma-separated allowed origins
                             (default: localhost dev servers only)
        API_KEY:             when set, all API requests must include
                             X-API-Key: <value>. Leave unset for local dev.
        OTEL_SERVICE_NAME:   tracing service.name resource attribute
        OTEL_TRACES_EXPORTER:none | console | otlp (comma-separated allowed)
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint
        OTEL_EXPORTER_OTLP_PROTOCOL: http/protobuf | grpc
        OTEL_EXPORTER_OTLP_HEADERS:  comma-separated k=v pairs
        OTEL_EXPORTER_OTLP_INSECURE: true/false for OTLP gRPC transport
        ODDS_API_KEY:        The Odds API key for GET /odds. Optional; absent
                             means /odds reports configured=false (no upstream call).
        ODDS_MIN_REMAINING:  Monthly quota floor for GET /odds. Upstream calls
                             stop (429, no spend) once the provider's
                             x-requests-remaining header falls below this.
                             Default 50. Unparseable values fall back to 50.
    """

    def __init__(self) -> None:
        self.database_url: str = _env(
            "DATABASE_URL",
            "postgresql://oracle:oracle@localhost:15439/oracle",
        )
        # Normalise: SQLModel/psycopg2 needs postgresql://, not asyncpg.
        self.database_url = self.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )

        self.mlflow_tracking_uri: str = _env(
            "MLFLOW_TRACKING_URI", "http://localhost:15091"
        )
        self.model_version: str = _env(
            "MODEL_VERSION", "latest"
        )
        # Fails closed: unknown values raise rather than falling back.
        self.product_mode: str = resolve_product_mode()  # pragma: no mutate
        self.settings_file: str = _env(
            "SETTINGS_FILE", "user_config.json"
        )
        self.baseline_manifest_path: str = _env(
            "BASELINE_MANIFEST_PATH", "releases/current_baseline.json"
        )
        self.artifact_invalidation_path: str = _env(
            "ARTIFACT_INVALIDATION_PATH", "releases/artifact_invalidations.json"
        )
        self.log_level: str = _env("LOG_LEVEL", "INFO").upper()  # pragma: no mutate
        self.cors_origins: list[str] = [
            o.strip()
            for o in _env(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:3000",
            ).split(",")
        ]
        # When set, every request must carry X-API-Key: <value>.
        # Empty string (the default) disables the check — safe for local dev.
        self.api_key: str = os.environ.get("API_KEY", "").strip()
        # When set, write endpoints (PUT /settings) additionally require
        # X-Admin-Key: <value>. Falls back to api_key if unset, so a single
        # key still works for local dev. Set a separate value in production.
        self.admin_api_key: str = os.environ.get("ADMIN_API_KEY", "").strip()
        # The Odds API key used by GET /odds. Read server-side only; never logged
        # or returned. Empty disables the upstream call (configured=false).
        self.odds_api_key: str = os.environ.get("ODDS_API_KEY", "").strip()
        # Monthly quota floor for GET /odds (ODDS_MIN_REMAINING, default 50).
        # The upstream answers every call with an x-requests-remaining header;
        # once the last seen value drops below this floor, /odds answers 429
        # without calling upstream, so the rest of the month's quota survives.
        self.odds_min_remaining: int = _parse_odds_floor(
            os.environ.get("ODDS_MIN_REMAINING", "50")
        )
        self.otel_service_name: str = _env(
            "OTEL_SERVICE_NAME",
            "gridiron-oracle-api",
        ).strip()
        self.otel_traces_exporter: str = _env(
            "OTEL_TRACES_EXPORTER",
            "none",
        ).strip().lower()
        self.otel_exporter_otlp_endpoint: str = _env(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "",
        ).strip()
        self.otel_exporter_otlp_protocol: str = _env(
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            "http/protobuf",
        ).strip().lower()
        self.otel_exporter_otlp_headers: str = _env(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "",
        ).strip()
        self.otel_exporter_otlp_insecure: bool = _env(
            "OTEL_EXPORTER_OTLP_INSECURE",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}

        self._warn_on_insecure_defaults()

    def _warn_on_insecure_defaults(self) -> None:
        """
        Emit warnings for insecure or missing configuration at startup.

        Secret rotation policy:
          - API_KEY:      rotate every 90 days or immediately on suspected leak.
                          Generate with: openssl rand -hex 32
          - DATABASE_URL: rotate the password component whenever DB credentials
                          are rotated in the PostgreSQL server.  Update .env and
                          restart the service.
          - All secrets should be stored in .env (never committed to git) or
            injected via the container orchestrator's secret management.
        """
        _DB_DEFAULT = "postgresql://oracle:oracle@localhost:15439/oracle"

        if not self.api_key:
            _log.warning(
                "API_KEY is not set — all endpoints are unauthenticated. "
                "Set API_KEY in your .env file before deploying."
            )

        if self.database_url == _DB_DEFAULT:
            _log.warning(
                "DATABASE_URL is using the default dev credential "
                "(oracle:oracle). Set a strong password before deploying."
            )

        if self.product_mode == "artifact_backed" and not self.api_key:
            _log.warning(
                "PRODUCT_MODE=artifact_backed with no API_KEY — "
                "production mode is active but the API is unauthenticated."
            )

        if self.api_key and not self.admin_api_key:
            _log.warning(
                "ADMIN_API_KEY is not set — write endpoints (PUT /settings) "
                "are protected only by API_KEY. Set ADMIN_API_KEY to separate "
                "read and write access before deploying."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()


# Convenience singleton — import this directly in most modules.
settings = get_settings()
