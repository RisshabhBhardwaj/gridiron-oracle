"""Behavioral locks intentionally runnable on audit-baseline-2026-08-09.

Unlike implementation tests, these only use modules and source files that
existed at the audit baseline. They therefore prove the bad behaviour is
rejected, rather than proving a new module imports.
"""

from __future__ import annotations

import ast
import re
import os
from pathlib import Path


# Allows this file to be executed from the audit worktree without copying it
# there.  Imports still resolve from that worktree when pytest is invoked from
# it, while source inspections use this explicit root.
ROOT = Path(os.environ.get("GRIDIRON_REPO_ROOT", Path(__file__).resolve().parents[2]))


def test_default_model_features_exclude_raw_target_game_snap_participation() -> None:
    forbidden = {"snap_pct_off", "offense_pct", "routes_run_pct"}
    tree = ast.parse((ROOT / "ml" / "utils.py").read_text())
    feature_cols = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "FEATURE_COLS"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    )
    assert not forbidden.intersection(feature_cols), (
        "Raw target-game participation reached the default model frame: "
        f"{sorted(forbidden.intersection(feature_cols))}"
    )


def test_reprojection_gate_does_not_select_candidate_evidence_by_glob() -> None:
    source = (ROOT / "scripts" / "reprojection_gate.py").read_text()
    assert "glob(" not in source
    assert "glob_pat" not in source
    assert "candidate-oof" in source


def test_serving_artifact_selection_does_not_use_mtime() -> None:
    source_roots = (ROOT / "backend" / "app", ROOT / "ml", ROOT / "scripts")
    offenders = []
    for root in source_roots:
        for path in root.rglob("*.py"):
            if path.name == "artifact_manifest.py":
                continue
            text = path.read_text()
            # ArtifactManifest uses mtime_ns only to invalidate an already
            # verified hash cache; it never selects an artifact. This lock
            # targets selection expressions, the C-06 failure shape.
            if re.search(r"(?:max|sorted)\([^\n]*(?:getmtime|key\s*=\s*mtime)", text):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"mtime-based artifact selection remains: {offenders}"
