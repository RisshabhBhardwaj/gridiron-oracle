"""The materializer records the deployed commit. The container has no .git, so
the platform variable is used, and a missing git checkout must not raise."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from scripts import materialize_stack_projections as mod


def test_platform_commit_variable_wins(monkeypatch) -> None:
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "abc123")
    assert mod._git_commit() == "abc123"


def test_no_git_checkout_reports_unknown(monkeypatch) -> None:
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)
    with patch.object(mod.subprocess, "check_output", side_effect=subprocess.CalledProcessError(128, "git")):
        assert mod._git_commit() == "unknown"


def test_git_checkout_reports_head(monkeypatch) -> None:
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)
    with patch.object(mod.subprocess, "check_output", return_value="deadbeef\n"):
        assert mod._git_commit() == "deadbeef"
