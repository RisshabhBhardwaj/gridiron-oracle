"""``scripts/check_no_lfs_pointers.py``: the box must hold model bytes, not Git LFS pointers.

A Mac checkout without ``git lfs pull`` has 130-byte pointer files where the models should
be. rsync would copy them, the box would fail its digest check only at serving time. This
stdlib-only script is run on the box with ``python3`` after the sync and exits non-zero.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_lfs_pointers.py"
POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 123\n"


def _run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), "--root", str(root)], capture_output=True, text=True)


def _models(root: Path, release: str = "causal_x") -> Path:
    d = root / "releases" / "candidates" / release / "models"
    d.mkdir(parents=True)
    return d


def test_real_model_bytes_pass(tmp_path):
    (_models(tmp_path) / "lgbm_a_QB.txt").write_bytes(b"tree\nversion=v4\n" + b"\x00" * 64)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr


def test_a_pointer_file_fails_and_is_named(tmp_path):
    models = _models(tmp_path)
    (models / "lgbm_a_QB.txt").write_bytes(b"real")
    (models / "catboost_a_QB.cbm").write_bytes(POINTER)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "catboost_a_QB.cbm" in result.stderr
    assert "git lfs pull" in result.stderr


def test_every_candidate_release_is_checked(tmp_path):
    (_models(tmp_path, "causal_a") / "m.txt").write_bytes(b"real")
    (_models(tmp_path, "causal_b") / "m.txt").write_bytes(POINTER)
    assert _run(tmp_path).returncode == 1


def test_no_candidates_directory_is_a_failure_not_a_pass(tmp_path):
    """An empty tree would silently pass otherwise."""
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "no model files" in result.stderr


def test_the_script_is_stdlib_only():
    import ast

    stdlib = set(sys.stdlib_module_names)
    tree = ast.parse(SCRIPT.read_text())
    imported = {
        n.split(".")[0]
        for node in ast.walk(tree)
        for n in ([a.name for a in node.names] if isinstance(node, ast.Import)
                  else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
    }
    assert imported <= stdlib | {"__future__"}, imported - stdlib
