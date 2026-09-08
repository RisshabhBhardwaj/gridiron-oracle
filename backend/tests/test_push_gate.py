"""
Tests for the public-push gate (`.githooks/pre-push`) — audit findings C-07, C-30.

The gate previously scanned only the tip commit of a new ref and only the
endpoint diff of an existing ref, so a blocked path introduced in an ancestor
was invisible. These tests drive the *real* hook over its real stdin protocol.

`gitleaks` is an optional extra layer; every assertion here exercises the
built-in path/size/content scans so the suite behaves identically on a runner
without the binary. `CI` is scrubbed from the hook's environment except in the
one test that asserts the CI-requires-gitleaks policy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".githooks" / "pre-push"
ZERO = "0" * 40

# Assembled at runtime, never written as literals. The gate scans this file
# like any other, and a fixture containing a real-looking key would make the
# repository unpushable through its own gate — which is the correct behaviour,
# so the fixtures avoid triggering it rather than being exempted from it.
_FAKE_SECRETS = [
    "AWS_ACCESS_KEY_ID = '" + "AKIA" + "IOSFODNN7EXAMPLE" + "'\n",
    "-----BEGIN RSA " + "PRIVATE KEY" + "-----\nMIIEow==\n",
    "api" + "_key = '" + "abcdef0123456789" * 2 + "'\n",
]



def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    out = subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _commit(repo: Path, path: str, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _run_hook(
    repo: Path,
    stdin: str,
    *,
    remote: str = "origin",
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(HOOK), remote, "https://example.invalid/repo.git"],
        cwd=repo,
        input=stdin,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    return r


# ── C-07: new ref must be scanned across every introduced commit ─────────────


def test_new_ref_ancestor_bypass_is_blocked(repo: Path) -> None:
    """The exact reproduction from the audit: commit `.env`, then commit only
    README, feed the tip with a zero remote SHA. Old hook exited 0."""
    _commit(repo, ".env", "DATABASE_URL=postgresql://u:p@localhost/db\n", "add env")
    tip = _commit(repo, "README.md", "# hello\n", "add readme")

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n")

    assert result.returncode != 0, result.stdout + result.stderr
    assert ".env" in result.stderr


def test_new_ref_root_commit_is_scanned(repo: Path) -> None:
    """A blocked path in the very first commit of a new ref must be caught."""
    tip = _commit(repo, "secrets/token.txt", "hunter2\n", "root with secrets dir")

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "secrets/" in result.stderr


def test_clean_new_ref_passes(repo: Path) -> None:
    _commit(repo, "README.md", "# hello\n", "readme")
    tip = _commit(repo, ".env.example", "DATABASE_URL=\n", "public template")

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n")

    assert result.returncode == 0, result.stdout + result.stderr


# ── C-07: existing ref — add-then-delete within the range ────────────────────


def test_existing_ref_add_then_delete_is_blocked(repo: Path) -> None:
    """The endpoint diff hides a path added and removed inside the range."""
    base = _commit(repo, "README.md", "# hello\n", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)

    _commit(repo, "deploy.pem", _FAKE_SECRETS[1], "oops")
    (repo / "deploy.pem").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remove it")
    tip = _git(repo, "rev-parse", "HEAD")

    # Endpoint diff is clean — the file does not exist at either endpoint.
    endpoint_diff = _git(repo, "diff", "--name-only", base, tip)
    assert "deploy.pem" not in endpoint_diff

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {base}\n")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "deploy.pem" in result.stderr


def test_existing_ref_clean_range_passes(repo: Path) -> None:
    base = _commit(repo, "README.md", "# hello\n", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    tip = _commit(repo, "src/app.py", "print('hi')\n", "feature")

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {base}\n")

    assert result.returncode == 0, result.stdout + result.stderr


def test_branch_deletion_is_allowed(repo: Path) -> None:
    base = _commit(repo, "README.md", "# hello\n", "base")

    result = _run_hook(repo, f"(delete) {ZERO} refs/heads/main {base}\n")

    assert result.returncode == 0, result.stdout + result.stderr


# ── C-07: content-secret scanning ────────────────────────────────────────────


@pytest.mark.parametrize("payload", _FAKE_SECRETS)
def test_secret_content_in_innocuous_path_is_blocked(repo: Path, payload: str) -> None:
    """Path-name scanning alone cannot see this — the filename is unremarkable."""
    tip = _commit(repo, "config/settings.py", payload, "config")

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "possible secret" in result.stderr


def test_nested_dotenv_is_blocked(repo: Path) -> None:
    """`.env` patterns were root-anchored; a nested copy slipped through."""
    tip = _commit(repo, "infra/.env", "SECRET=1\n", "nested env")

    result = _run_hook(repo, f"refs/heads/main {tip} refs/heads/main {ZERO}\n")

    assert result.returncode != 0, result.stdout + result.stderr


# ── C-07: object-size limit ──────────────────────────────────────────────────


def test_oversized_object_is_blocked(repo: Path) -> None:
    tip = _commit(repo, "data/blob.csv", "x" * 4096, "bulk artifact")

    result = _run_hook(
        repo,
        f"refs/heads/main {tip} refs/heads/main {ZERO}\n",
        env_overrides={"PUSH_GATE_MAX_OBJECT_BYTES": "1024"},
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "too large" in result.stderr


# ── C-07: gitleaks policy ────────────────────────────────────────────────────


def test_ci_without_gitleaks_fails_closed(repo: Path, tmp_path: Path) -> None:
    """In CI the optional layer becomes mandatory: a missing binary blocks."""
    if shutil.which("gitleaks"):
        pytest.skip("gitleaks installed; the missing-binary policy cannot be exercised")
    tip = _commit(repo, "README.md", "# hello\n", "base")

    result = _run_hook(
        repo,
        f"refs/heads/main {tip} refs/heads/main {ZERO}\n",
        env_overrides={"CI": "true"},
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "gitleaks" in result.stderr


# ── C-30: the hook must be the one Git actually runs ─────────────────────────


def test_repo_configures_hooks_path_to_tracked_directory() -> None:
    """`make hooks` used to copy into .git/hooks, so clones had no gate and
    edits to the tracked hook did not take effect until reinstall."""
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "core.hooksPath .githooks" in makefile
    bootstrap = (REPO_ROOT / "scripts" / "bootstrap_local.sh").read_text()
    assert "core.hooksPath" in bootstrap


def test_hook_is_executable() -> None:
    assert os.access(HOOK, os.X_OK), f"{HOOK} must be executable"


def test_hook_is_executable_in_the_index() -> None:
    """A CI checkout takes the mode from the index, not the working tree."""
    mode = subprocess.run(
        ["git", "ls-files", "-s", ".githooks/pre-push"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]
    assert mode == "100755", f"tracked mode is {mode}, expected 100755"


def test_ci_refuses_to_pass_with_nothing_scanned(repo: Path) -> None:
    """A required check that can pass vacuously is not a check: if the object
    enumeration comes back empty under CI, that is a failure, not a pass."""
    base = _commit(repo, "README.md", "# hello\n", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)

    # Tip already present on the remote-tracking ref: nothing is introduced.
    result = _run_hook(
        repo,
        f"refs/heads/main {base} refs/heads/main {base}\n",
        env_overrides={"CI": "true", "PATH": os.environ.get("PATH", "")},
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "vacuously" in result.stderr


def test_nothing_to_push_is_fine_locally(repo: Path) -> None:
    base = _commit(repo, "README.md", "# hello\n", "base")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)

    result = _run_hook(repo, f"refs/heads/main {base} refs/heads/main {base}\n")

    assert result.returncode == 0, result.stdout + result.stderr
