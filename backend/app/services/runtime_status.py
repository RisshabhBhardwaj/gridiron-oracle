"""
Runtime integrity checks for health and release-readiness reporting.
"""

from __future__ import annotations

import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from backend.app.core.config import settings
from backend.app.core.tracing import describe_tracing

logger = logging.getLogger(__name__)

_BACKTEST_RESULTS_DIR = Path(__file__).parents[3] / "ml" / "backtest_results"


class RuntimeStatusService:
    """Collect lightweight integrity checks for the running deployment."""

    def __init__(self, db_url: str, mlflow_tracking_uri: str, model_version: str) -> None:
        self._db_url = db_url
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._model_version = model_version

    def build_report(self) -> dict:
        """Return a serializable integrity report."""
        checks: dict[str, dict] = {
            "database": self._check_database(),
            "feature_matrix": self._check_feature_matrix(),
            "mlflow": self._check_mlflow(),
            "baseline_manifest": self._check_baseline_manifest(),
            "backtest_assets": self._check_backtest_assets(),
            "tracing": self._check_tracing(),
        }

        fallback_allowed = settings.product_mode == "graceful_fallback"
        required = ["database", "feature_matrix"]
        if settings.product_mode == "artifact_backed":
            required.extend(["mlflow", "baseline_manifest", "backtest_assets"])

        blocking = [name for name in required if checks[name]["status"] == "error"]
        issues = [name for name, result in checks.items() if result["status"] in {"warn", "error"}]

        if blocking:
            overall_status = "blocked"
        elif issues:
            overall_status = "degraded"
        else:
            overall_status = "ok"

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "product_mode": settings.product_mode,
            "model_version": self._model_version,
            "fallback_allowed": fallback_allowed,
            "artifact_mode_ready": all(
                checks[name]["status"] == "ok" for name in required
            ) if settings.product_mode == "artifact_backed" else None,
            "checks": checks,
        }

    def _check_database(self) -> dict:
        try:
            import psycopg2

            conn = psycopg2.connect(self._db_url)
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            finally:
                conn.close()
            return {"status": "ok", "detail": "Database reachable"}
        except Exception as exc:
            return {"status": "error", "detail": f"Database unavailable: {exc}"}

    def _check_feature_matrix(self) -> dict:
        try:
            import psycopg2

            conn = psycopg2.connect(self._db_url)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT season, week, computed_at
                    FROM feature_matrix
                    WHERE computed_at IS NOT NULL
                    ORDER BY computed_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
            finally:
                conn.close()
        except Exception as exc:
            return {"status": "error", "detail": f"feature_matrix check failed: {exc}"}

        if not row:
            return {"status": "error", "detail": "feature_matrix has no timestamped rows"}

        season, week, computed_at = row
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)
        age_hours = round((datetime.now(timezone.utc) - computed_at).total_seconds() / 3600.0, 2)
        status = "ok" if age_hours <= 24 else "warn"
        return {
            "status": status,
            "detail": "feature_matrix freshness checked",
            "observed": {
                "season": season,
                "week": week,
                "computed_at": computed_at.isoformat(),
                "age_hours": age_hours,
            },
        }

    def _check_mlflow(self) -> dict:
        if not self._mlflow_tracking_uri:
            return {"status": "error", "detail": "MLFLOW_TRACKING_URI is not configured"}

        parsed = urlparse(self._mlflow_tracking_uri)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return {
                "status": "warn",
                "detail": "MLflow URI is non-HTTP; socket reachability check skipped",
                "observed": {"uri": self._mlflow_tracking_uri},
            }

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((parsed.hostname, port), timeout=1.5):
                pass
            return {
                "status": "ok",
                "detail": "MLflow endpoint reachable",
                "observed": {"uri": self._mlflow_tracking_uri},
            }
        except OSError as exc:
            return {
                "status": "error",
                "detail": f"MLflow unreachable: {exc}",
                "observed": {"uri": self._mlflow_tracking_uri},
            }

    def _check_baseline_manifest(self) -> dict:
        path = Path(settings.baseline_manifest_path)
        if not path.exists():
            return {
                "status": "error",
                "detail": f"Baseline manifest missing at {path}",
            }

        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            return {
                "status": "error",
                "detail": f"Baseline manifest unreadable: {exc}",
            }

        observed = {
            "path": str(path),
            "model_version": payload.get("model_version"),
            "git_commit": payload.get("git_commit"),
            "created_at": payload.get("created_at"),
        }
        status = "ok" if payload.get("model_version") else "warn"
        detail = "Baseline manifest loaded" if status == "ok" else "Baseline manifest missing model_version"
        return {"status": status, "detail": detail, "observed": observed}

    def _check_backtest_assets(self) -> dict:
        latest_csv = None
        if _BACKTEST_RESULTS_DIR.exists():
            candidates = sorted(
                _BACKTEST_RESULTS_DIR.glob("backtest_*.csv"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                latest_csv = candidates[0]

        db_count = None
        try:
            import psycopg2

            conn = psycopg2.connect(self._db_url)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM backtest_results")
                db_count = int(cur.fetchone()[0])
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("backtest_results count check failed: %s", exc)

        if latest_csv or (db_count and db_count > 0):
            observed = {}
            if latest_csv is not None:
                observed["latest_csv"] = str(latest_csv)
            if db_count is not None:
                observed["db_rows"] = db_count
            return {
                "status": "ok",
                "detail": "Backtest assets available",
                "observed": observed,
            }

        return {
            "status": "error",
            "detail": "No backtest CSV artifacts or database rows found",
        }

    def _check_tracing(self) -> dict:
        state = describe_tracing(settings)
        requested = state["requested_exporters"]
        active = state["active_exporters"]

        observed = {
            "service_name": state["service_name"],
            "requested_exporters": requested,
            "active_exporters": active,
            "otlp_protocol": state["otlp_protocol"],
            "otlp_endpoint": state["otlp_endpoint"],
        }

        if state["last_error"]:
            observed["last_error"] = state["last_error"]
            return {
                "status": "warn",
                "detail": "Tracing configured with exporter issues",
                "observed": observed,
            }

        if requested == ["none"]:
            return {
                "status": "ok",
                "detail": "Trace propagation enabled; exporter disabled",
                "observed": observed,
            }

        if active:
            endpoint = state["otlp_endpoint"]
            if "otlp" in active and endpoint:
                parsed = urlparse(endpoint)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                if host:
                    try:
                        with socket.create_connection((host, port), timeout=1.5):
                            pass
                    except OSError as exc:
                        observed["collector_reachability_error"] = str(exc)
                        return {
                            "status": "warn",
                            "detail": "Tracing exporter configured but collector unreachable",
                            "observed": observed,
                        }
            return {
                "status": "ok",
                "detail": "Tracing exporter configured",
                "observed": observed,
            }

        return {
            "status": "warn",
            "detail": "Tracing exporter requested but not active",
            "observed": observed,
        }
