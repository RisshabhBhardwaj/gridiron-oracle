"""
Runtime integrity checks for health and release-readiness reporting.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from backend.app.core.config import settings
from backend.app.core.tracing import describe_tracing

logger = logging.getLogger(__name__)

_BACKTEST_RESULTS_DIR = Path(__file__).parents[3] / "ml" / "backtest_results"
_REPO_ROOT = Path(__file__).parents[3]

#: Paths a release commit may touch, on top of the manifest's pinned source
#: commit, without invalidating that commit as "what was verified".
#:
#: A baseline manifest necessarily records the commit it was frozen from
#: *before* that manifest is itself committed — committing it moves HEAD past
#: the commit it names. Requiring exact equality therefore made every
#: committed release reject itself. The fix is not to drop the check but to
#: scope it: HEAD may move ahead of the pinned commit only through changes
#: that are pure evidence about that commit (the manifest, reports about it,
#: locking tests, drift-detection bookkeeping) — never through changes to the
#: code that produced it or the artifacts it pins. `releases/candidates/` is
#: deliberately excluded even though it is under `releases/`: it holds the
#: SHA-pinned model artifacts themselves, and `load_manifest` does not verify
#: their digests at readiness-check time (only `resolve_stack` does, lazily,
#: at load time) — so this allowlist is the only thing standing between a
#: swapped artifact and a green readiness check.
_EVIDENCE_ONLY_PATHS = frozenset({
    "releases/current_baseline.json",
    "releases/artifacts/MANIFEST.json",
})
_EVIDENCE_ONLY_PREFIXES = ("reports/", "backend/tests/")


def _is_evidence_only_path(relpath: str) -> bool:
    if relpath in _EVIDENCE_ONLY_PATHS:
        return True
    return any(relpath.startswith(prefix) for prefix in _EVIDENCE_ONLY_PREFIXES)


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
        if not path.is_absolute():
            path = _REPO_ROOT / path
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

        try:
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True,
            ).strip()
            dirty = bool(subprocess.check_output(
                # An operator's untracked notes do not alter the checked-in
                # release.  Only tracked/untracked-index changes can make a
                # frozen code commit non-reproducible.
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO_ROOT, text=True,
            ).strip())
        except (OSError, subprocess.CalledProcessError) as exc:
            return {"status": "error", "detail": f"Cannot inspect release Git state: {exc}"}

        try:
            from ml.artifact_manifest import REQUIRED_SERVING_CELLS, load_manifest
            expected_cells = set(REQUIRED_SERVING_CELLS)
            manifest = load_manifest(path, root=_REPO_ROOT)
            actual_cells = set(manifest.stack_index)
            if actual_cells != expected_cells:
                return {
                    "status": "error",
                    "detail": "Baseline manifest does not declare the full serving cell matrix",
                    "observed": {"declared_cells": sorted(f"{s}/{p}" for s, p in actual_cells)},
                }
        except Exception as exc:
            return {"status": "error", "detail": f"Baseline artifact pins are invalid: {exc}"}

        observed = {
            "path": str(path),
            "model_version": payload.get("model_version"),
            "git_commit": payload.get("git_commit"),
            "head": head,
            "worktree_dirty": dirty,
            "created_at": payload.get("created_at"),
        }
        if not payload.get("model_version"):
            return {"status": "error", "detail": "Baseline manifest missing model_version", "observed": observed}
        if payload.get("git_dirty") is not False:
            return {"status": "error", "detail": "Baseline manifest was frozen from a dirty worktree", "observed": observed}
        if dirty:
            return {"status": "error", "detail": "Runtime worktree is dirty; refuse stale release manifest", "observed": observed}

        manifest_commit = payload.get("git_commit")
        exact_match = manifest_commit == head
        if not exact_match:
            lineage = self._check_commit_lineage(manifest_commit, head)
            if lineage is not None:
                return {"status": "error", "detail": lineage, "observed": observed}
        if payload.get("product_mode") != settings.product_mode:
            return {"status": "error", "detail": "Baseline manifest product_mode does not match runtime", "observed": observed}
        detail = (
            "Baseline manifest matches clean HEAD and full cell matrix"
            if exact_match else
            "Baseline manifest's pinned commit is a clean ancestor of HEAD via an "
            "evidence-only release commit, and declares the full cell matrix"
        )
        return {"status": "ok", "detail": detail, "observed": observed}

    def _check_commit_lineage(self, manifest_commit: object, head: str) -> str | None:
        """
        Return an error detail if *head* is not a legitimate evidence-only
        descendant of *manifest_commit*, else ``None``.

        Called only when the manifest's pinned commit differs from HEAD. HEAD
        is allowed to be ahead of the pinned commit — but only through commits
        that changed nothing except evidence about the pinned commit (see
        `_EVIDENCE_ONLY_PATHS`/`_EVIDENCE_ONLY_PREFIXES`). Anything else —
        code drift, an unrelated branch, an artifact swap — is rejected.
        """
        if not manifest_commit or not isinstance(manifest_commit, str):
            return "Baseline manifest commit does not match HEAD"

        try:
            subprocess.check_output(
                ["git", "merge-base", "--is-ancestor", manifest_commit, "HEAD"],
                cwd=_REPO_ROOT, stderr=subprocess.STDOUT, text=True,
            )
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return "Baseline manifest commit does not match HEAD and is not an ancestor of it"
            # Anything other than "definitively not an ancestor" (e.g. 128 for
            # an object git cannot resolve) is unknown lineage, not evidence
            # of safety — fail closed rather than guess.
            return f"Baseline manifest commit lineage could not be verified: {exc.output.strip()}"
        except OSError as exc:
            return f"Baseline manifest commit lineage could not be verified: {exc}"

        try:
            changed = subprocess.check_output(
                ["git", "diff", "--name-only", manifest_commit, "HEAD"],
                cwd=_REPO_ROOT, text=True,
            ).splitlines()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"Baseline manifest commit lineage could not be verified: {exc}"

        non_evidence = sorted(p for p in changed if p and not _is_evidence_only_path(p))
        if non_evidence:
            return (
                "Baseline manifest commit is an ancestor of HEAD, but HEAD "
                f"changes non-evidence paths since that commit: {non_evidence}"
            )
        return None

    def _check_backtest_assets(self) -> dict:
        latest_csv = None
        if _BACKTEST_RESULTS_DIR.exists():
            # By filename (which carries the run stamp), not st_mtime — readiness
            # must not name a different "latest" artifact just because a file was
            # touched or freshly checked out. Matches BacktestService._load_csv.
            candidates = sorted(
                _BACKTEST_RESULTS_DIR.glob("backtest_*.csv"),
                key=lambda path: path.name,
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
