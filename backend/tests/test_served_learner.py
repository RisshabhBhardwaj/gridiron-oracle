"""Served learner disclosure must match the pinned selection in current_baseline.json.

Expected values are read from the pinned files the baseline manifest names, not
hard-coded, so a re-freeze moves the test with the pin.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml import artifact_manifest
from ml.served_learner import default_selection_path, served_learner

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "releases" / "current_baseline.json"


def _baseline() -> dict:
    return json.loads(BASELINE.read_text())


def _pinned_selection_path() -> Path:
    return ROOT / _baseline()["artifacts"]["selection_policy"][0]


def _pinned_decisions() -> dict[str, dict]:
    return json.loads(_pinned_selection_path().read_text())["decisions"]


def _cells(selection: str) -> list[str]:
    return sorted(key for key, entry in _pinned_decisions().items() if entry.get("selection") == selection)


def test_default_disclosure_reads_the_pinned_selection() -> None:
    assert default_selection_path().resolve() == _pinned_selection_path().resolve()


def test_pinned_artifacts_match_their_recorded_digests() -> None:
    baseline = _baseline()
    pinned = baseline["artifacts"]["selection_policy"] + baseline["artifacts"]["ridge_coefficients"]
    for rel in pinned:
        digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert digest == baseline["artifact_digests"][rel], rel


@pytest.mark.parametrize("key", sorted(_pinned_decisions()))
def test_every_disclosed_cell_matches_pinned_selection(key: str) -> None:
    stat, position = key.split(":")
    expected = _pinned_decisions()[key]["selection"]
    assert served_learner(stat, position) == expected


def test_served_learner_discloses_lgbm_identity_cells() -> None:
    cells = _cells("lgbm_identity")
    assert cells, "pinned selection has no lgbm_identity cells to check"
    for key in cells:
        stat, position = key.split(":")
        assert served_learner(stat, position) == "lgbm_identity"


def test_served_learner_discloses_ridge_stack_cells() -> None:
    cells = _cells("ridge_stack")
    assert cells, "pinned selection has no ridge_stack cells to check"
    for key in cells:
        stat, position = key.split(":")
        assert served_learner(stat, position) == "ridge_stack"


def test_unpinned_cell_is_unknown() -> None:
    assert served_learner("not_a_stat", "QB") == "unknown"


def test_missing_manifest_discloses_unknown(monkeypatch, tmp_path) -> None:
    # Naming a learner from some other release would be a false disclosure.
    monkeypatch.setenv("BASELINE_MANIFEST_PATH", str(tmp_path / "absent.json"))
    artifact_manifest.clear_cache()
    try:
        assert default_selection_path() is None
        assert served_learner("carries", "RB") == "unknown"
    finally:
        monkeypatch.delenv("BASELINE_MANIFEST_PATH")
        artifact_manifest.clear_cache()
