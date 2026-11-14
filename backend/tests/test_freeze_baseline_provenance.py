"""freeze_baseline records the same commit the API's readiness check will compare."""
from __future__ import annotations

import pytest

from scripts import freeze_baseline as mod


@pytest.fixture()
def no_worktree(monkeypatch):
    monkeypatch.setattr(mod, "_has_worktree", lambda: False)
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)


def test_env_commit_is_used_without_a_worktree(no_worktree, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")
    assert mod._head_commit() == "abc123"
    assert mod._is_dirty() is False
    assert mod._branch() == "unknown"


def test_platform_variable_matches_the_runtime_lookup_order(no_worktree, monkeypatch):
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "railway1")
    monkeypatch.setenv("GIT_COMMIT_SHA", "other")
    assert mod._head_commit() == "railway1"


def test_no_worktree_and_no_variable_refuses(no_worktree):
    with pytest.raises(RuntimeError, match="cannot record"):
        mod._head_commit()


def test_a_worktree_wins_over_the_variable(monkeypatch):
    monkeypatch.setattr(mod, "_has_worktree", lambda: True)
    monkeypatch.setattr(mod, "_git", lambda args: "deadbeef")
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")
    assert mod._head_commit() == "deadbeef"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://oracle:s3cr%40t-pw@db:5432/oracle",
        "postgresql://oracle:plainpw@localhost/oracle",
    ],
)
def test_database_url_password_is_never_recorded(url):
    out = mod._redact_url(url)
    assert "s3cr" not in out and "plainpw" not in out
    assert "***" in out and "@" in out and "oracle" in out


def test_unparseable_url_is_redacted_entirely():
    assert mod._redact_url("not a url :pw@") == "<unparseable; redacted>"
    assert mod._redact_url(None) is None
