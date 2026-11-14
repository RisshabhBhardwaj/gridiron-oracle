"""
backend/tests/test_ml_cov_backend_runtime_status.py

Coverage tests for backend/app/services/runtime_status.py (health and
release-readiness report).

Every external dependency is stubbed: psycopg2.connect (database and
feature_matrix checks), socket.create_connection (MLflow and collector
reachability), subprocess.check_output (git state), ml.artifact_manifest
.load_manifest (artifact pins) and describe_tracing (tracing state). The
manifest and backtest directory live under tmp_path. No network is used.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import ml.artifact_manifest as artifact_manifest
from backend.app.core.config import settings
from backend.app.services import runtime_status as rs
from backend.app.services.runtime_status import RuntimeStatusService, _is_evidence_only_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _service(mlflow_uri: str = "http://mlflow.test:5000") -> RuntimeStatusService:
    return RuntimeStatusService(
        db_url="postgresql://db/test", mlflow_tracking_uri=mlflow_uri, model_version="v-test",
    )


class _Cursor:
    def __init__(self, state):
        self._state = state

    def execute(self, sql, params=None):
        self._state["sql"].append(sql)
        if self._state.get("execute_error"):
            raise self._state["execute_error"]

    def fetchone(self):
        return self._state["one"]


class _Conn:
    def __init__(self, state):
        self._state = state

    def cursor(self):
        return _Cursor(self._state)

    def close(self):
        self._state["closed"] = True


@pytest.fixture
def db(monkeypatch):
    import psycopg2

    state = {"sql": [], "closed": False, "one": (1,), "connect_error": None, "execute_error": None}

    def connect(dsn):
        state["dsn"] = dsn
        if state["connect_error"]:
            raise state["connect_error"]
        return _Conn(state)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return state


@pytest.fixture
def reachable(monkeypatch):
    """socket.create_connection that records the address and succeeds."""
    seen: list = []

    class _Ok:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def connect(address, timeout=None):
        seen.append((address, timeout))
        return _Ok()

    monkeypatch.setattr(socket, "create_connection", connect)
    return seen


@pytest.fixture
def unreachable(monkeypatch):
    def connect(address, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(socket, "create_connection", connect)


# ── _check_database ───────────────────────────────────────────────────────────

class TestCheckDatabase:
    def test_reachable_database_is_ok(self, db) -> None:
        result = _service()._check_database()
        assert result == {"status": "ok", "detail": "Database reachable"}
        assert db["sql"] == ["SELECT 1"]
        assert db["closed"] is True

    def test_unreachable_database_is_error(self, db) -> None:
        db["connect_error"] = OSError("no route")
        result = _service()._check_database()
        assert result["status"] == "error"
        assert result["detail"] == "Database unavailable: no route"


# ── _check_feature_matrix ─────────────────────────────────────────────────────

class TestCheckFeatureMatrix:
    def test_fresh_rows_are_ok_with_observed_age(self, db) -> None:
        computed = datetime.now(timezone.utc) - timedelta(hours=2)
        db["one"] = (2025, 7, computed)
        result = _service()._check_feature_matrix()
        assert result["status"] == "ok"
        assert result["observed"]["season"] == 2025
        assert result["observed"]["week"] == 7
        assert result["observed"]["age_hours"] == pytest.approx(2.0, abs=0.05)

    def test_stale_rows_warn(self, db) -> None:
        computed = datetime.now(timezone.utc) - timedelta(hours=30)
        db["one"] = (2025, 1, computed)
        result = _service()._check_feature_matrix()
        assert result["status"] == "warn"
        assert result["observed"]["age_hours"] == pytest.approx(30.0, abs=0.05)

    def test_naive_timestamp_is_treated_as_utc(self, db) -> None:
        naive = datetime.utcnow() - timedelta(hours=1)
        db["one"] = (2025, 2, naive)
        result = _service()._check_feature_matrix()
        assert result["status"] == "ok"
        assert result["observed"]["computed_at"].endswith("+00:00")
        assert result["observed"]["age_hours"] == pytest.approx(1.0, abs=0.05)

    def test_no_timestamped_rows_is_error(self, db) -> None:
        db["one"] = None
        result = _service()._check_feature_matrix()
        assert result == {"status": "error", "detail": "feature_matrix has no timestamped rows"}

    def test_query_failure_is_error(self, db) -> None:
        db["execute_error"] = RuntimeError("relation missing")
        result = _service()._check_feature_matrix()
        assert result["status"] == "error"
        assert "feature_matrix check failed: relation missing" in result["detail"]


# ── _check_mlflow ─────────────────────────────────────────────────────────────

class TestCheckMlflow:
    def test_unset_uri_is_error(self) -> None:
        result = _service(mlflow_uri="")._check_mlflow()
        assert result == {"status": "error", "detail": "MLFLOW_TRACKING_URI is not configured"}

    def test_non_http_uri_only_warns(self) -> None:
        result = _service(mlflow_uri="file:///srv/mlruns")._check_mlflow()
        assert result["status"] == "warn"
        assert result["observed"] == {"uri": "file:///srv/mlruns"}

    def test_reachable_endpoint_is_ok_with_default_port(self, reachable) -> None:
        result = _service(mlflow_uri="https://mlflow.test")._check_mlflow()
        assert result["status"] == "ok"
        assert reachable == [(("mlflow.test", 443), 1.5)]

    def test_unreachable_endpoint_is_error(self, unreachable) -> None:
        result = _service()._check_mlflow()
        assert result["status"] == "error"
        assert result["detail"] == "MLflow unreachable: connection refused"
        assert result["observed"]["uri"] == "http://mlflow.test:5000"


# ── _check_backtest_assets: covered in test_release_evals.py (pinned causal evals) ──

# ── _check_tracing ────────────────────────────────────────────────────────────

def _tracing_state(**overrides) -> dict:
    state = {
        "service_name": "gridiron-api",
        "requested_exporters": ["otlp"],
        "active_exporters": ["otlp"],
        "otlp_protocol": "http/protobuf",
        "otlp_endpoint": "http://collector.test:4318",
        "last_error": None,
    }
    state.update(overrides)
    return state


class TestCheckTracing:
    def _patch(self, monkeypatch, state):
        monkeypatch.setattr(rs, "describe_tracing", lambda _settings: state)

    def test_exporter_error_warns(self, monkeypatch) -> None:
        self._patch(monkeypatch, _tracing_state(last_error="exporter failed to start"))
        result = _service()._check_tracing()
        assert result["status"] == "warn"
        assert result["observed"]["last_error"] == "exporter failed to start"

    def test_disabled_exporter_is_ok(self, monkeypatch) -> None:
        self._patch(monkeypatch, _tracing_state(requested_exporters=["none"], active_exporters=[]))
        result = _service()._check_tracing()
        assert result["status"] == "ok"
        assert result["detail"] == "Trace propagation enabled; exporter disabled"

    def test_reachable_collector_is_ok(self, monkeypatch, reachable) -> None:
        self._patch(monkeypatch, _tracing_state())
        result = _service()._check_tracing()
        assert result["status"] == "ok"
        assert reachable == [(("collector.test", 4318), 1.5)]

    def test_unreachable_collector_warns(self, monkeypatch, unreachable) -> None:
        self._patch(monkeypatch, _tracing_state())
        result = _service()._check_tracing()
        assert result["status"] == "warn"
        assert result["detail"] == "Tracing exporter configured but collector unreachable"
        assert result["observed"]["collector_reachability_error"] == "connection refused"

    def test_non_otlp_exporter_is_ok_without_probe(self, monkeypatch) -> None:
        def must_not_connect(*args, **kwargs):
            raise AssertionError("no collector probe expected")

        monkeypatch.setattr(socket, "create_connection", must_not_connect)
        self._patch(monkeypatch, _tracing_state(active_exporters=["console"], otlp_endpoint=None))
        result = _service()._check_tracing()
        assert result["status"] == "ok"
        assert result["detail"] == "Tracing exporter configured"

    def test_requested_but_inactive_warns(self, monkeypatch) -> None:
        self._patch(monkeypatch, _tracing_state(active_exporters=[]))
        result = _service()._check_tracing()
        assert result["status"] == "warn"
        assert result["detail"] == "Tracing exporter requested but not active"


# ── _is_evidence_only_path ────────────────────────────────────────────────────

@pytest.mark.parametrize("path, expected", [
    ("releases/current_baseline.json", True),
    ("releases/artifacts/MANIFEST.json", True),
    ("reports/audit.md", True),
    ("backend/tests/test_new_lock.py", True),
    ("releases/candidates/model.ubj", False),   # pinned artifacts are never evidence
    ("ml/train.py", False),
    ("backend/app/core/config.py", False),
])
def test_evidence_only_path_allowlist(path, expected) -> None:
    assert _is_evidence_only_path(path) is expected


# ── _check_commit_lineage ─────────────────────────────────────────────────────

def _git(monkeypatch, *, ancestor_rc=0, ancestor_error=None, diff_out="", diff_error=None):
    """Route subprocess.check_output calls for merge-base and diff."""
    calls: list = []

    def fake(args, **kwargs):
        calls.append(args)
        if args[:2] == ["git", "merge-base"]:
            if ancestor_error is not None:
                raise ancestor_error
            if ancestor_rc != 0:
                raise subprocess.CalledProcessError(ancestor_rc, args, output="fatal: bad object")
            return ""
        if args[:2] == ["git", "diff"]:
            if diff_error is not None:
                raise diff_error
            return diff_out
        raise AssertionError(f"unexpected git call {args}")

    monkeypatch.setattr(subprocess, "check_output", fake)
    return calls


class TestCommitLineage:
    def test_missing_manifest_commit_is_rejected(self) -> None:
        assert "does not match HEAD" in _service()._check_commit_lineage(None, "abc")
        assert "does not match HEAD" in _service()._check_commit_lineage(123, "abc")

    def test_not_an_ancestor(self, monkeypatch) -> None:
        _git(monkeypatch, ancestor_rc=1)
        detail = _service()._check_commit_lineage("old", "new")
        assert detail == "Baseline manifest commit does not match HEAD and is not an ancestor of it"

    def test_unresolvable_commit_fails_closed(self, monkeypatch) -> None:
        _git(monkeypatch, ancestor_rc=128)
        detail = _service()._check_commit_lineage("ghost", "new")
        assert detail.startswith("Baseline manifest commit lineage could not be verified: ")
        assert "bad object" in detail

    def test_git_not_runnable_fails_closed(self, monkeypatch) -> None:
        _git(monkeypatch, ancestor_error=OSError("git not installed"))
        assert _service()._check_commit_lineage("old", "new").endswith("git not installed")

    def test_diff_failure_fails_closed(self, monkeypatch) -> None:
        _git(monkeypatch, diff_error=subprocess.CalledProcessError(1, ["git", "diff"]))
        detail = _service()._check_commit_lineage("old", "new")
        assert detail.startswith("Baseline manifest commit lineage could not be verified")

    def test_code_drift_is_rejected_and_listed(self, monkeypatch) -> None:
        _git(monkeypatch, diff_out="releases/current_baseline.json\nml/train.py\nbackend/app/core/config.py\n")
        detail = _service()._check_commit_lineage("old", "new")
        assert "HEAD changes non-evidence paths since that commit" in detail
        assert "['backend/app/core/config.py', 'ml/train.py']" in detail

    def test_evidence_only_descendant_is_accepted(self, monkeypatch) -> None:
        calls = _git(monkeypatch, diff_out="releases/current_baseline.json\nreports/run.md\n")
        assert _service()._check_commit_lineage("old", "new") is None
        assert calls[0] == ["git", "merge-base", "--is-ancestor", "old", "HEAD"]
        assert calls[1] == ["git", "diff", "--name-only", "old", "HEAD"]


# ── _check_baseline_manifest ──────────────────────────────────────────────────

_CELLS = list(artifact_manifest.REQUIRED_SERVING_CELLS)


def _manifest_payload(**overrides) -> dict:
    payload = {
        "model_version": "v-test",
        "git_commit": "abc123",
        "git_dirty": False,
        "product_mode": settings.product_mode,
        "created_at": "2026-10-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def manifest_env(monkeypatch, tmp_path):
    """Writes a manifest file and stubs git plus load_manifest. Returns a setter helper."""
    state = {
        "path": tmp_path / "baseline.json",
        "head": "abc123",
        "dirty": "",
        "git_fails": False,
        "cells": {cell: None for cell in _CELLS},
        "load_error": None,
        "env_sha": None,
    }

    def fake_git(args, **kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            if state["git_fails"]:
                raise OSError("no .git directory")
            return state["head"] + "\n"
        if args[:3] == ["git", "status", "--porcelain"]:
            return state["dirty"]
        raise AssertionError(f"unexpected git call {args}")

    def fake_load(path, root=None):
        if state["load_error"]:
            raise state["load_error"]
        return SimpleNamespace(stack_index=state["cells"])

    monkeypatch.setattr(subprocess, "check_output", fake_git)
    monkeypatch.setattr(artifact_manifest, "load_manifest", fake_load)
    monkeypatch.setattr(settings, "baseline_manifest_path", str(state["path"]))
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)
    return state


def _write_payload(state, payload) -> None:
    state["path"].write_text(json.dumps(payload) if not isinstance(payload, str) else payload)


class TestBaselineManifest:
    def test_exact_match_with_clean_worktree_is_ok(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload())
        result = _service()._check_baseline_manifest()
        assert result["status"] == "ok"
        assert result["detail"] == "Baseline manifest matches clean HEAD and full cell matrix"
        assert result["observed"]["head"] == "abc123"
        assert result["observed"]["worktree_dirty"] is False

    def test_missing_manifest_is_error(self, manifest_env) -> None:
        result = _service()._check_baseline_manifest()
        assert result["status"] == "error"
        assert result["detail"].startswith("Baseline manifest missing at ")

    def test_unreadable_manifest_is_error(self, manifest_env) -> None:
        _write_payload(manifest_env, "{not json")
        result = _service()._check_baseline_manifest()
        assert result["status"] == "error"
        assert result["detail"].startswith("Baseline manifest unreadable")

    def test_dirty_runtime_worktree_is_error(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload())
        manifest_env["dirty"] = " M ml/train.py\n"
        result = _service()._check_baseline_manifest()
        assert result["status"] == "error"
        assert result["detail"] == "Runtime worktree is dirty; refuse stale release manifest"

    def test_missing_model_version_is_error(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload(model_version=""))
        result = _service()._check_baseline_manifest()
        assert result["detail"] == "Baseline manifest missing model_version"

    def test_manifest_frozen_from_dirty_tree_is_error(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload(git_dirty=True))
        result = _service()._check_baseline_manifest()
        assert result["detail"] == "Baseline manifest was frozen from a dirty worktree"

    def test_incomplete_cell_matrix_is_error(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload())
        manifest_env["cells"] = {}
        result = _service()._check_baseline_manifest()
        assert result["detail"] == "Baseline manifest does not declare the full serving cell matrix"

    def test_invalid_artifact_pins_are_error(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload())
        manifest_env["load_error"] = ValueError("digest mismatch")
        result = _service()._check_baseline_manifest()
        assert result["detail"] == "Baseline artifact pins are invalid: digest mismatch"

    def test_product_mode_mismatch_is_error(self, manifest_env, monkeypatch) -> None:
        _write_payload(manifest_env, _manifest_payload(product_mode="some_other_mode"))
        result = _service()._check_baseline_manifest()
        assert result["detail"] == "Baseline manifest product_mode does not match runtime"

    def test_no_git_with_deployed_sha_warns_when_commit_differs(self, manifest_env, monkeypatch) -> None:
        _write_payload(manifest_env, _manifest_payload(git_commit="abc123"))
        manifest_env["git_fails"] = True
        monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deadbeef")
        result = _service()._check_baseline_manifest()
        # The deployed SHA is taken from the platform env, and lineage cannot be checked without git.
        assert result["status"] == "warn"
        assert result["observed"]["head"] == "deadbeef"
        assert result["observed"]["worktree_dirty"] is False

    def test_no_git_and_no_sha_is_error(self, manifest_env) -> None:
        _write_payload(manifest_env, _manifest_payload())
        manifest_env["git_fails"] = True
        result = _service()._check_baseline_manifest()
        assert result["status"] == "error"
        assert result["detail"].startswith("Cannot inspect release Git state")

    def test_container_build_without_worktree_warns_on_different_commit(self, manifest_env, monkeypatch) -> None:
        _write_payload(manifest_env, _manifest_payload(git_commit="abc123"))
        manifest_env["git_fails"] = True
        monkeypatch.setenv("GIT_COMMIT_SHA", "feedface")
        result = _service()._check_baseline_manifest()
        assert result["status"] == "warn"
        assert "lineage cannot be verified without a git worktree" in result["detail"]
        assert result["observed"]["head"] == "feedface"

    def test_ancestor_with_evidence_only_changes_is_ok(self, manifest_env, monkeypatch) -> None:
        _write_payload(manifest_env, _manifest_payload(git_commit="old999"))
        manifest_env["head"] = "new111"
        monkeypatch.setattr(rs.RuntimeStatusService, "_check_commit_lineage", lambda self, c, h: None)
        result = _service()._check_baseline_manifest()
        assert result["status"] == "ok"
        assert "evidence-only release commit" in result["detail"]

    def test_lineage_rejection_is_error(self, manifest_env, monkeypatch) -> None:
        _write_payload(manifest_env, _manifest_payload(git_commit="old999"))
        manifest_env["head"] = "new111"
        monkeypatch.setattr(
            rs.RuntimeStatusService, "_check_commit_lineage",
            lambda self, c, h: "Baseline manifest commit does not match HEAD and is not an ancestor of it",
        )
        result = _service()._check_baseline_manifest()
        assert result["status"] == "error"
        assert "not an ancestor" in result["detail"]


# ── build_report ──────────────────────────────────────────────────────────────

def _stub_checks(service, monkeypatch, **statuses):
    """Replace each check with a fixed status. Missing names default to ok."""
    names = ["database", "feature_matrix", "mlflow", "baseline_manifest", "backtest_assets", "tracing"]
    method_for = {
        "database": "_check_database",
        "feature_matrix": "_check_feature_matrix",
        "mlflow": "_check_mlflow",
        "baseline_manifest": "_check_baseline_manifest",
        "backtest_assets": "_check_backtest_assets",
        "tracing": "_check_tracing",
    }
    for name in names:
        status = statuses.get(name, "ok")
        monkeypatch.setattr(service, method_for[name], lambda status=status: {"status": status, "detail": status})
    return service


class TestBuildReport:
    def test_artifact_backed_all_ok(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "product_mode", "artifact_backed")
        report = _stub_checks(_service(), monkeypatch).build_report()
        assert report["overall_status"] == "ok"
        assert report["artifact_mode_ready"] is True
        assert report["fallback_allowed"] is False
        assert report["model_version"] == "v-test"
        assert set(report["checks"]) == {
            "database", "feature_matrix", "mlflow", "baseline_manifest", "backtest_assets", "tracing",
        }
        assert datetime.fromisoformat(report["generated_at"]).tzinfo is not None

    def test_mlflow_outage_degrades_but_does_not_block_artifact_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "product_mode", "artifact_backed")
        report = _stub_checks(_service(), monkeypatch, mlflow="error").build_report()
        assert report["overall_status"] == "degraded"
        assert report["artifact_mode_ready"] is True

    def test_database_failure_blocks(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "product_mode", "artifact_backed")
        report = _stub_checks(_service(), monkeypatch, database="error").build_report()
        assert report["overall_status"] == "blocked"
        assert report["artifact_mode_ready"] is False

    def test_stale_feature_matrix_degrades_artifact_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "product_mode", "artifact_backed")
        report = _stub_checks(_service(), monkeypatch, feature_matrix="warn").build_report()
        assert report["overall_status"] == "degraded"
        assert report["artifact_mode_ready"] is False

    def test_graceful_fallback_only_requires_database_and_features(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "product_mode", "graceful_fallback")
        report = _stub_checks(
            _service(), monkeypatch, baseline_manifest="error", backtest_assets="error",
        ).build_report()
        assert report["overall_status"] == "degraded"
        assert report["fallback_allowed"] is True
        assert report["artifact_mode_ready"] is None

    def test_graceful_fallback_blocks_on_database(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "product_mode", "graceful_fallback")
        report = _stub_checks(_service(), monkeypatch, database="error").build_report()
        assert report["overall_status"] == "blocked"
