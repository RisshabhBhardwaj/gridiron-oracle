"""scripts/pin_artifact_group.py: add a new pinned group with computed digests, fail closed otherwise."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pin_artifact_group", Path(__file__).resolve().parents[2] / "scripts" / "pin_artifact_group.py"
)
pag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pag)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "releases" / "candidates" / "rel1" / "evals").mkdir(parents=True)
    (tmp_path / "releases" / "candidates" / "rel1" / "stack.csv").write_text("a\n")
    manifest = {
        "artifacts": {"stacks": ["releases/candidates/rel1/stack.csv"]},
        "artifact_digests": {"releases/candidates/rel1/stack.csv": "x" * 64},
    }
    (tmp_path / "releases" / "current_baseline.json").write_text(json.dumps(manifest))
    return tmp_path


def _file(root: Path, name: str, body: bytes) -> str:
    rel = f"releases/candidates/rel1/evals/{name}"
    (root / rel).write_bytes(body)
    return rel


def _manifest(root: Path) -> dict:
    return json.loads((root / "releases" / "current_baseline.json").read_text())


def test_pins_new_group_with_real_digests_and_leaves_others(root: Path) -> None:
    a = _file(root, "a.csv", b"one\n")
    b = _file(root, "b.json", b"{}\n")

    pag.pin_group("causal_evals", [b, a], root=root, check_git=False)

    m = _manifest(root)
    assert m["artifacts"]["causal_evals"] == [a, b]
    assert m["artifacts"]["stacks"] == ["releases/candidates/rel1/stack.csv"]
    assert m["artifact_digests"][a] == hashlib.sha256(b"one\n").hexdigest()
    assert m["artifact_digests"]["releases/candidates/rel1/stack.csv"] == "x" * 64


def test_dry_run_writes_nothing(root: Path) -> None:
    before = (root / "releases" / "current_baseline.json").read_text()
    pag.pin_group("causal_evals", [_file(root, "a.csv", b"1")], root=root, dry_run=True, check_git=False)
    assert (root / "releases" / "current_baseline.json").read_text() == before


@pytest.mark.parametrize(
    "setup, message",
    [
        (lambda r: ("stacks", [_file(r, "a.csv", b"1")]), "already pinned"),
        (lambda r: ("evals", ["releases/candidates/rel1/stack.csv"]), "another group"),
        (lambda r: ("evals", [_file(r, "p.bin", pag.LFS_POINTER_PREFIX + b"\noid sha256:ab\n")]), "LFS pointer"),
        (lambda r: ("evals", ["releases/current_baseline.json"]), "not under"),
        (lambda r: ("evals", ["releases/candidates/rel1/evals/missing.csv"]), "not a file"),
        (lambda r: ("bad/name", [_file(r, "a.csv", b"1")]), "bad group"),
    ],
)
def test_refuses(root: Path, setup, message: str) -> None:
    group, paths = setup(root)
    before = (root / "releases" / "current_baseline.json").read_text()
    with pytest.raises(pag.PinError, match=message):
        pag.pin_group(group, paths, root=root, check_git=False)
    assert (root / "releases" / "current_baseline.json").read_text() == before


def test_refuses_uncommitted_files(root: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    rel = _file(root, "a.csv", b"1")
    with pytest.raises(pag.PinError, match="not committed"):
        pag.pin_group("causal_evals", [rel], root=root)
