"""Pinned bucket attributions: manifest resolution and the materializer's checks."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml import bucket_attribution as ba
from ml.artifact_manifest import (
    ArtifactIntegrityError,
    ManifestEntryMissing,
    load_manifest,
    sha256_of,
)
from ml.bucket_weights import BUCKET_KEYS, UNWEIGHTED_KEY

ROOT = Path(__file__).resolve().parents[2]
STACK_REL = "releases/candidates/t/stack_receiving_yards_WR_ab12cd34.csv"
ATTR_REL = "releases/candidates/t/attr_stack_receiving_yards_WR_ab12cd34.parquet"


def _load_materializer():
    spec = importlib.util.spec_from_file_location(
        "materialize_under_test", ROOT / "scripts" / "materialize_stack_projections.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stack_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "game_id": [f"g{i}" for i in range(n)],
        "projection": np.linspace(5.0, 60.0, n),
    })


def _attr_frame(stack: pd.DataFrame, *, break_row: bool = False) -> pd.DataFrame:
    attr = stack[["player_id", "game_id"]].copy()
    attr["base"] = 20.0
    for key in BUCKET_KEYS:
        attr[key] = 1.0
    attr[UNWEIGHTED_KEY] = stack["projection"].to_numpy() - 20.0 - len(BUCKET_KEYS)
    if break_row:
        attr.loc[0, "base"] += 0.5
    return attr


def _manifest(tmp_path: Path, *, pin_attr: bool = True, tamper: bool = False):
    stack_file = tmp_path / STACK_REL
    stack_file.parent.mkdir(parents=True)
    stack_file.write_text("player_id,game_id\n")
    attr_file = tmp_path / ATTR_REL
    _attr_frame(_stack_frame()).to_parquet(attr_file, index=False)
    digests = {STACK_REL: sha256_of(stack_file), ATTR_REL: sha256_of(attr_file)}
    if tamper:
        pd.concat([_attr_frame(_stack_frame())] * 2).to_parquet(attr_file, index=False)
    artifacts = {"stacks": [STACK_REL]}
    if pin_attr:
        artifacts["bucket_attributions"] = [ATTR_REL]
    else:
        digests.pop(ATTR_REL)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": artifacts, "artifact_digests": digests}))
    return load_manifest(path, root=tmp_path)


def test_resolve_attribution_returns_the_pinned_file(tmp_path):
    manifest = _manifest(tmp_path)
    assert manifest.resolve_attribution("receiving_yards", "WR") == tmp_path / ATTR_REL


def test_resolve_attribution_refuses_an_unpinned_file(tmp_path):
    manifest = _manifest(tmp_path, pin_attr=False)
    with pytest.raises(ManifestEntryMissing):
        manifest.resolve_attribution("receiving_yards", "WR")


def test_resolve_attribution_refuses_a_modified_file(tmp_path):
    manifest = _manifest(tmp_path, tamper=True)
    with pytest.raises(ArtifactIntegrityError):
        manifest.resolve_attribution("receiving_yards", "WR")


class _Manifest:
    def __init__(self, path):
        self.path = path

    def resolve_attribution(self, stat, position):
        return self.path


def _materializer_with(monkeypatch, tmp_path, attr: pd.DataFrame):
    module = _load_materializer()
    path = tmp_path / "attr.parquet"
    attr.to_parquet(path, index=False)
    monkeypatch.setattr(module, "get_manifest", lambda: _Manifest(path))
    return module


def test_materializer_accepts_attributions_that_add_up(monkeypatch, tmp_path):
    stack = _stack_frame()
    module = _materializer_with(monkeypatch, tmp_path, _attr_frame(stack))
    out = module._load_cell_attributions("receiving_yards", "WR", stack)
    assert len(out) == len(stack)
    assert set(out["stat"]) == {"receiving_yards"}
    assert out["max_residual"].max() <= ba.ATTRIBUTION_TOLERANCE


def test_materializer_refuses_attributions_that_do_not_add_up(monkeypatch, tmp_path):
    stack = _stack_frame()
    module = _materializer_with(monkeypatch, tmp_path, _attr_frame(stack, break_row=True))
    with pytest.raises(ba.AttributionError, match="do not add up"):
        module._load_cell_attributions("receiving_yards", "WR", stack)


def test_materializer_refuses_a_stack_row_with_no_attribution(monkeypatch, tmp_path):
    stack = _stack_frame()
    module = _materializer_with(monkeypatch, tmp_path, _attr_frame(stack).iloc[1:])
    with pytest.raises(ba.AttributionError, match="no attribution row"):
        module._load_cell_attributions("receiving_yards", "WR", stack)


def test_materializer_refuses_duplicate_attribution_rows(monkeypatch, tmp_path):
    stack = _stack_frame()
    attr = pd.concat([_attr_frame(stack), _attr_frame(stack).iloc[:1]])
    module = _materializer_with(monkeypatch, tmp_path, attr)
    with pytest.raises(ba.AttributionError, match="repeats"):
        module._load_cell_attributions("receiving_yards", "WR", stack)
