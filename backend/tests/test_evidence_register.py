"""verify_evidence_manifest.register: adds only unlisted watched-directory files as provisional."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_evidence_manifest as vem  # noqa: E402


def _root(tmp_path: Path) -> Path:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "eval_causal_a_WR.csv").write_text("x\n")
    (tmp_path / "reports" / "eval_causal_b_TE.csv").write_text("y\n")
    (tmp_path / "ml" / "oof").mkdir(parents=True)
    (tmp_path / "ml" / "oof" / "stack.csv").write_text("z\n")
    return tmp_path


def test_registers_unlisted_files_as_provisional_with_hashes(tmp_path):
    root = _root(tmp_path)
    manifest = {"artifacts": [{"path": "reports/eval_causal_a_WR.csv", "sha256": "old", "status": "provisional", "produced_by": "p", "description": "d"}]}
    added = vem.register(manifest, ["reports/eval_causal_*.csv"], produced_by="p.py", description="d", root=root)
    assert added == ["reports/eval_causal_b_TE.csv"]
    entry = manifest["artifacts"][-1]
    assert entry["status"] == "provisional"
    assert entry["sha256"] == vem.sha256(root / "reports" / "eval_causal_b_TE.csv")
    assert entry["bytes"] == 2
    assert vem.validate_schema(manifest) == []


def test_registering_twice_adds_nothing(tmp_path):
    root = _root(tmp_path)
    manifest = {"artifacts": []}
    vem.register(manifest, ["reports/*.csv"], produced_by="p", description="d", root=root)
    assert vem.register(manifest, ["reports/*.csv"], produced_by="p", description="d", root=root) == []


def test_files_outside_watched_directories_are_refused(tmp_path):
    root = _root(tmp_path)
    with pytest.raises(ValueError, match="not under a watched directory"):
        vem.register({"artifacts": []}, ["ml/oof/*.csv"], produced_by="p", description="d", root=root)
