"""Pinned base models in the release manifest (decision 0012, commit 3, D1).

Each test builds a small self-consistent release in a temp directory, then breaks
exactly one thing. Loading must fail closed, and resolving a model must verify
its SHA-256 and refuse a Git LFS pointer with an instruction a person can follow.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.artifact_manifest import (
    ArtifactIntegrityError,
    LfsPointerError,
    ManifestEntryMissing,
    ManifestError,
    load_manifest,
    verify_file_digest,
)
from ml.season_constants import LAST_COMPLETE_SEASON

RELEASE = "causal_test"
LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:" + "0" * 64 + "\nsize 12345\n"
)

# (stat, position) -> learner weights. Covers every shape in the real release:
# both learners, lgbm only, catboost only (the other learner has weight 0.0).
CELLS = {
    ("targets", "TE"): {"lgbm": 0.4, "catboost": 0.6},
    ("passing_yards", "QB"): {"lgbm": 1.0, "catboost": 0.0},
    ("receiving_yards", "WR"): {"lgbm": 0.0, "catboost": 1.0},
}
EXT = {"lgbm": "txt", "catboost": "cbm"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(root: Path, relpath: str, data: bytes | str) -> str:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return relpath


def build_release(root: Path, *, with_base_models: bool = True) -> dict:
    """Write a consistent release under *root*; return the manifest dict."""
    rel_dir = f"releases/candidates/{RELEASE}"
    stacks, ridges, models = [], [], []
    digests: dict[str, str] = {}
    meta_models = []
    for (stat, pos), weights in CELLS.items():
        stack = _write(root, f"ml/oof/stack_{stat}_{pos}_20261114.csv", f"stack {stat} {pos}\n")
        stacks.append(stack)
        ridge = _write(
            root,
            f"{rel_dir}/ridge_{stat}_{pos}_coefs.json",
            json.dumps({"learner_order": ["lgbm", "catboost"], "weights": weights, "intercept": 0.0}),
        )
        ridges.append(ridge)
        for learner, weight in weights.items():
            if weight == 0.0:
                continue
            name = f"{learner}_{stat}_{pos}.{EXT[learner]}"
            body = f"model bytes {learner} {stat} {pos}".encode()
            relpath = _write(root, f"{rel_dir}/models/{name}", body)
            models.append(relpath)
            digests[relpath] = _sha(body)
            meta_models.append({
                "learner": learner, "stat": stat, "position": pos, "file": name,
                "sha256": _sha(body), "release_id": RELEASE,
                "train_seasons": list(range(2019, LAST_COMPLETE_SEASON + 1)),
                "max_train_season": LAST_COMPLETE_SEASON,
                "features": ["a", "b", "c"],
            })
    meta = {"schema_version": 1, "release_id": RELEASE, "models": meta_models}
    meta_rel = _write(root, f"{rel_dir}/models/MODELS.json", json.dumps(meta))
    for relpath in [*stacks, *ridges, meta_rel]:
        digests[relpath] = _sha((root / relpath).read_bytes())
    artifacts = {"stacks": stacks, "ridge_coefficients": ridges}
    if with_base_models:
        artifacts["base_models"] = [*models, meta_rel]
    manifest = {"artifacts": artifacts, "artifact_digests": digests}
    _write(root, "releases/current_baseline.json", json.dumps(manifest))
    return manifest


def _rewrite_meta(root: Path, mutate) -> None:
    """Change MODELS.json and re-pin its digest so only the content defect remains."""
    meta_rel = f"releases/candidates/{RELEASE}/models/MODELS.json"
    meta = json.loads((root / meta_rel).read_text())
    mutate(meta)
    _write(root, meta_rel, json.dumps(meta))
    manifest = json.loads((root / "releases/current_baseline.json").read_text())
    manifest["artifact_digests"][meta_rel] = _sha((root / meta_rel).read_bytes())
    _write(root, "releases/current_baseline.json", json.dumps(manifest))


def _rewrite_manifest(root: Path, mutate) -> None:
    path = root / "releases/current_baseline.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    build_release(tmp_path)
    return tmp_path


def _load(root: Path):
    return load_manifest(root / "releases/current_baseline.json", root=root)


def test_every_pinned_model_resolves_and_verifies(root: Path) -> None:
    m = _load(root)
    path = m.resolve_base_model("targets", "TE", "lgbm")
    assert path.read_bytes() == b"model bytes lgbm targets TE"
    assert m.resolve_base_model("targets", "te", "catboost").name == "catboost_targets_TE.cbm"
    assert m.resolve_base_model("passing_yards", "QB", "lgbm").exists()
    assert m.resolve_base_model("receiving_yards", "WR", "catboost").exists()
    assert len(m.model_index) == 4


def test_zero_weight_learner_has_no_model_and_resolving_it_raises(root: Path) -> None:
    m = _load(root)
    with pytest.raises(ManifestEntryMissing):
        m.resolve_base_model("passing_yards", "QB", "catboost")
    with pytest.raises(ManifestEntryMissing):
        m.resolve_base_model("receiving_yards", "WR", "lgbm")


def test_model_metadata_comes_from_the_pinned_models_json(root: Path) -> None:
    meta = _load(root).base_model_meta("targets", "TE", "catboost")
    assert meta["max_train_season"] == LAST_COMPLETE_SEASON
    assert meta["features"] == ["a", "b", "c"]
    assert meta["release_id"] == RELEASE


def test_models_are_protected_like_every_other_pinned_artifact(root: Path) -> None:
    m = _load(root)
    assert f"releases/candidates/{RELEASE}/models/lgbm_targets_TE.txt" in m.protected_relpaths()
    assert f"releases/candidates/{RELEASE}/models/MODELS.json" in m.protected_relpaths()


def test_manifest_without_base_models_still_loads_but_the_forward_producer_refuses(tmp_path: Path) -> None:
    build_release(tmp_path, with_base_models=False)
    m = _load(tmp_path)
    assert m.resolve_stack("targets", "TE", verify=True).exists()  # history serving is unaffected
    assert not m.has_base_models
    with pytest.raises(ManifestEntryMissing, match="base_models"):
        m.require_base_models()
    with pytest.raises(ManifestEntryMissing, match="base_models"):
        m.resolve_base_model("targets", "TE", "lgbm")


def test_missing_model_file_raises_file_not_found(root: Path) -> None:
    (root / f"releases/candidates/{RELEASE}/models/lgbm_targets_TE.txt").unlink()
    with pytest.raises(FileNotFoundError, match="lgbm_targets_TE"):
        _load(root).resolve_base_model("targets", "TE", "lgbm")


def test_modified_model_file_fails_its_digest(root: Path) -> None:
    (root / f"releases/candidates/{RELEASE}/models/lgbm_targets_TE.txt").write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        _load(root).resolve_base_model("targets", "TE", "lgbm")


def test_git_lfs_pointer_is_named_as_such_with_the_fix(root: Path) -> None:
    (root / f"releases/candidates/{RELEASE}/models/catboost_targets_TE.cbm").write_text(LFS_POINTER)
    with pytest.raises(LfsPointerError) as err:
        _load(root).resolve_base_model("targets", "TE", "catboost")
    assert "LFS file not fetched" in str(err.value)
    assert "git lfs pull" in str(err.value)
    assert isinstance(err.value, ArtifactIntegrityError)  # callers that catch integrity errors still catch it


def test_lfs_pointer_is_refused_even_when_the_digest_was_computed_from_the_pointer(root: Path) -> None:
    rel = f"releases/candidates/{RELEASE}/models/catboost_targets_TE.cbm"
    (root / rel).write_text(LFS_POINTER)
    _rewrite_manifest(root, lambda m: m["artifact_digests"].__setitem__(rel, _sha(LFS_POINTER.encode())))
    _rewrite_meta(root, lambda meta: [e.__setitem__("sha256", _sha(LFS_POINTER.encode()))
                                      for e in meta["models"] if e["file"] == "catboost_targets_TE.cbm"])
    with pytest.raises(LfsPointerError):
        _load(root).resolve_base_model("targets", "TE", "catboost")


def test_digest_check_names_an_lfs_pointer_instead_of_reporting_a_mismatch(tmp_path: Path) -> None:
    """A pointer must never pass a digest check, even against its own digest, and the
    error must say the file was not fetched rather than look like a drifted pin."""
    pointer = tmp_path / "lgbm_targets_TE.txt"
    pointer.write_text(LFS_POINTER)
    for expected in (_sha(LFS_POINTER.encode()), "0" * 64):
        with pytest.raises(LfsPointerError, match="LFS file not fetched"):
            verify_file_digest(pointer, expected)


def test_non_zero_weight_learner_without_a_pinned_model_fails_at_load(root: Path) -> None:
    rel = f"releases/candidates/{RELEASE}/models/catboost_targets_TE.cbm"

    def drop(m):
        m["artifacts"]["base_models"].remove(rel)
        m["artifact_digests"].pop(rel)

    _rewrite_manifest(root, drop)
    _rewrite_meta(root, lambda meta: meta.__setitem__(
        "models", [e for e in meta["models"] if e["file"] != "catboost_targets_TE.cbm"]))
    with pytest.raises(ManifestError, match=r"catboost.*targets.*TE|targets.*TE.*catboost"):
        _load(root)


def test_pinned_model_that_no_ridge_file_uses_fails_at_load(root: Path) -> None:
    # catboost for passing_yards/QB has weight 0.0 in its ridge file.
    rel = f"releases/candidates/{RELEASE}/models/catboost_passing_yards_QB.cbm"
    body = b"unused model"
    _write(root, rel, body)

    def add(m):
        m["artifacts"]["base_models"].append(rel)
        m["artifact_digests"][rel] = _sha(body)

    _rewrite_manifest(root, add)
    _rewrite_meta(root, lambda meta: meta["models"].append({
        "learner": "catboost", "stat": "passing_yards", "position": "QB", "file": "catboost_passing_yards_QB.cbm",
        "sha256": _sha(body), "release_id": RELEASE, "train_seasons": [2019, 2020],
        "max_train_season": 2020, "features": ["a"],
    }))
    with pytest.raises(ManifestError, match="no ridge file uses"):
        _load(root)


def test_model_trained_past_the_last_complete_season_fails_at_load(root: Path) -> None:
    def bump(meta):
        meta["models"][0]["max_train_season"] = LAST_COMPLETE_SEASON + 1
        meta["models"][0]["train_seasons"] = [LAST_COMPLETE_SEASON + 1]

    _rewrite_meta(root, bump)
    with pytest.raises(ManifestError, match="LAST_COMPLETE_SEASON"):
        _load(root)


def test_models_from_two_releases_fail_at_load(root: Path) -> None:
    _rewrite_meta(root, lambda meta: meta["models"][1].__setitem__("release_id", "causal_other"))
    with pytest.raises(ManifestError, match="release_id"):
        _load(root)


def test_models_json_modified_after_pinning_fails_at_load(root: Path) -> None:
    meta_rel = f"releases/candidates/{RELEASE}/models/MODELS.json"
    meta = json.loads((root / meta_rel).read_text())
    meta["models"][0]["features"] = ["x"]
    (root / meta_rel).write_text(json.dumps(meta))
    with pytest.raises(ArtifactIntegrityError):
        _load(root)


def test_models_json_digest_disagreeing_with_the_manifest_fails_at_load(root: Path) -> None:
    _rewrite_meta(root, lambda meta: meta["models"][0].__setitem__("sha256", "f" * 64))
    with pytest.raises(ManifestError, match="sha256"):
        _load(root)


def test_pinned_model_missing_from_models_json_fails_at_load(root: Path) -> None:
    _rewrite_meta(root, lambda meta: meta.__setitem__("models", meta["models"][:-1]))
    with pytest.raises(ManifestError, match="MODELS.json"):
        _load(root)


def test_models_json_entry_that_disagrees_with_its_file_name_fails_at_load(root: Path) -> None:
    _rewrite_meta(root, lambda meta: meta["models"][0].__setitem__("position", "RB"))
    with pytest.raises(ManifestError):
        _load(root)


def test_wrong_file_type_for_the_learner_is_rejected(root: Path) -> None:
    rel = f"releases/candidates/{RELEASE}/models/lgbm_targets_TE.cbm"
    _write(root, rel, b"x")
    _rewrite_manifest(root, lambda m: (m["artifacts"]["base_models"].append(rel),
                                       m["artifact_digests"].__setitem__(rel, _sha(b"x"))))
    with pytest.raises(ManifestError, match="not a recognised base-model file"):
        _load(root)


def test_pinned_model_without_a_digest_fails_at_load(root: Path) -> None:
    rel = f"releases/candidates/{RELEASE}/models/lgbm_targets_TE.txt"
    _rewrite_manifest(root, lambda m: m["artifact_digests"].pop(rel))
    with pytest.raises(ManifestError, match="SHA-256"):
        _load(root)


def test_ridge_file_changed_after_pinning_fails_the_load_cross_check(root: Path) -> None:
    ridge = root / f"releases/candidates/{RELEASE}/ridge_targets_TE_coefs.json"
    ridge.write_text(json.dumps({"learner_order": ["lgbm", "catboost"],
                                 "weights": {"lgbm": 1.0, "catboost": 0.0}, "intercept": 0.0}))
    with pytest.raises(ArtifactIntegrityError):
        _load(root)


def test_freeze_carries_base_models_forward_and_hashes_them(root: Path, monkeypatch) -> None:
    from scripts import freeze_baseline

    manifest_path = root / "releases/current_baseline.json"
    before = json.loads(manifest_path.read_text())
    model_rel = f"releases/candidates/{RELEASE}/models/lgbm_targets_TE.txt"
    before["artifact_digests"].pop(model_rel)
    manifest_path.write_text(json.dumps(before))
    monkeypatch.setattr(freeze_baseline, "ROOT", root)
    payload: dict = {}
    freeze_baseline._carry_forward_artifact_pins(payload, manifest_path)
    assert payload["artifacts"]["base_models"] == before["artifacts"]["base_models"]
    assert payload["artifact_digests"][model_rel] == _sha(b"model bytes lgbm targets TE")
