"""scripts/sha256_files.sh must work on Linux (sha256sum only) and macOS (shasum only)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "sha256_files.sh"
BASH = shutil.which("bash")
REAL = {name: shutil.which(name) for name in ("sha256sum", "shasum")}


def _run(tmp_path: Path, tools: list[str]) -> subprocess.CompletedProcess:
    """Run the script with only the named hash tools on PATH (plus xargs and coreutils basics)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in [*tools, "xargs", "env"]:
        real = shutil.which(name)
        assert real, name
        (bin_dir / name).symlink_to(real)
    data = tmp_path / "a.csv"
    data.write_text("x,y\n1,2\n")
    return subprocess.run(
        [BASH, str(SCRIPT)], input=f"{data}\0".encode(), capture_output=True,
        env={"PATH": str(bin_dir)}, check=False,
    )


def _expected(tmp_path: Path) -> bytes:
    digest = hashlib.sha256((tmp_path / "a.csv").read_bytes()).hexdigest()
    return f"{digest}  {tmp_path / 'a.csv'}\n".encode()


@pytest.mark.skipif(not REAL["sha256sum"], reason="sha256sum not installed")
def test_uses_sha256sum_when_it_is_the_only_tool(tmp_path):
    result = _run(tmp_path, ["sha256sum"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == _expected(tmp_path)


@pytest.mark.skipif(not REAL["shasum"], reason="shasum not installed")
def test_falls_back_to_shasum_when_sha256sum_is_missing(tmp_path):
    result = _run(tmp_path, ["shasum"])
    assert result.returncode == 0, result.stderr
    assert result.stdout == _expected(tmp_path)


def test_fails_loudly_with_neither_tool(tmp_path):
    result = _run(tmp_path, [])
    assert result.returncode == 127
    assert b"neither sha256sum nor shasum" in result.stderr


def test_retrain_script_parses_and_uses_the_helper():
    subprocess.run([BASH, "-n", str(REPO / "scripts" / "retrain_causal_oof.sh")], check=True)
    text = (REPO / "scripts" / "retrain_causal_oof.sh").read_text()
    assert "shasum" not in text
    assert "scripts/sha256_files.sh" in text
