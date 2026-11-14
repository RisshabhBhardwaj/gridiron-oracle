"""
ml/artifact_manifest.py

Single authority for (a) which base learners may participate in a stack and
(b) which artifact file backs a given (stat, position) cell.

Why this module exists
----------------------
Two classes of silent-wrong-artifact bug motivated it:

1. **Learner policy lived only in the artifacts, not in the code.** Every
   shipped ``ridge_*_coefs.json`` is a clean ``lgbm`` + ``catboost`` two-learner
   fit, but the trainers and the stacker were perfectly willing to rebuild a
   four-learner ``xgb``/``tft`` stack. The policy was enforced by a CLI flag
   (``--exclude tft,xgb``) that callers kept forgetting. A forgotten flag must
   not be able to change which learners are served, so the allowlist is now a
   checked invariant inside the stacking code path (see
   :func:`assert_learner_policy`) rather than an argument.

2. **Artifact selection was by filesystem mtime.** ``max(paths, key=mtime)``
   picks whichever file was written or *touched* last, which is not a
   statement about correctness. Two reproduced consequences:

   * ``touch`` on a stale, collapsed stack makes every reader prefer it;
   * on a filesystem whose timestamp granularity is coarser than the checkout
     that produced the files (1-second granularity is common on ext3, HFS+ and
     a number of container and network mounts), the mtimes tie and ``max()``
     returns the *first* maximal element — the lexically earliest, i.e. the
     oldest-dated and in practice the poisoned file.

   Selection is therefore pinned to an explicit manifest entry and verified
   against a recorded SHA-256. There is deliberately **no glob fallback**: a
   missing or mismatched entry raises.

The manifest is ``releases/current_baseline.json``. It carries a path list
under ``artifacts`` and a flat ``artifact_digests`` map of repo-relative path →
SHA-256 hex digest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_RELPATH = "releases/current_baseline.json"

#: Every base-learner prefix the repo has ever written an OOF file for. Used to
#: *recognise* a learner OOF, not to authorise one.
LEARNER_PREFIXES: frozenset[str] = frozenset({"xgb", "lgbm", "catboost", "tft"})

#: The Phase-5 two-learner contract. TFT was dropped for MAE drag (~7.8) and
#: XGB for toxic causal-meta weights on short history plus the 2022 QB/RB/WR
#: OOF collapse. All 15 shipped coef files are exactly this set.
DEFAULT_ALLOWED_LEARNERS: frozenset[str] = frozenset({"lgbm", "catboost"})

#: Per-target overrides. Empty by design: the two-learner policy is global, and
#: an override here is the only legitimate way to widen it. Adding a key is a
#: reviewable change to the policy, not a runtime argument.
ALLOWED_LEARNERS_BY_TARGET: Mapping[str, frozenset[str]] = {}

#: Stacking needs at least two base learners for a meta-learner to mean
#: anything.
MIN_BASE_LEARNERS = 2

POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})

# The promoted serving matrix. A release may only omit a cell by making an
# explicit, reviewed manifest policy change; silent materializer drift is not
# an allowed state.
#
# Phase 6 (L4 scoring): the volume/yardage cells below were the only ones
# served for years even though receptions/TD/turnover models for every
# position had already cleared their promotion gate (serving_approved: true
# in ml/oof/candidate_20260821_fixed/promotion_*.json) and simply were never
# added here. Without them, ml.scoring.score_fantasy has no touchdown or
# reception input for any position — composing a "derived" fantasy total
# from yardage alone silently omits 6 pts/TD and 1 pt/reception for every
# player. This is the fix.
REQUIRED_SERVING_CELLS: tuple[tuple[str, str], ...] = (
    ("fantasy_ppr", "QB"), ("fantasy_ppr", "RB"),
    ("fantasy_ppr", "WR"), ("fantasy_ppr", "TE"),
    ("targets", "WR"), ("targets", "TE"), ("targets", "RB"),
    ("carries", "RB"), ("pass_attempts", "QB"), ("passing_yards", "QB"),
    ("receiving_yards", "WR"), ("receiving_yards", "TE"),
    ("receiving_yards", "RB"), ("rushing_yards", "QB"),
    ("rushing_yards", "RB"),
    # ── Phase 6 additions: TD/reception/turnover cells needed for score_fantasy ──
    ("receptions", "RB"), ("receptions", "TE"), ("receptions", "WR"),
    ("receiving_tds", "RB"), ("receiving_tds", "TE"), ("receiving_tds", "WR"),
    ("rushing_tds", "QB"), ("rushing_tds", "RB"), ("rushing_tds", "TE"), ("rushing_tds", "WR"),
    ("passing_tds", "QB"), ("interceptions", "QB"),
    ("fumbles", "QB"), ("fumbles", "RB"), ("fumbles", "TE"), ("fumbles", "WR"),
    ("completions", "QB"),
)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ManifestError(RuntimeError):
    """Base class for every fail-closed condition in this module."""


class ManifestEntryMissing(ManifestError):
    """The manifest does not pin an artifact for the requested cell."""


class ArtifactIntegrityError(ManifestError):
    """An artifact's SHA-256 does not match the digest the manifest pins."""


class LfsPointerError(ArtifactIntegrityError):
    """A pinned file is a Git LFS pointer, not the content: the checkout was never smudged."""


class LearnerPolicyError(ManifestError):
    """A learner outside the allowlist reached a stacking or coef-write path."""


# ── Learner policy ────────────────────────────────────────────────────────────


def allowed_learners(target: str) -> frozenset[str]:
    """Base-learner prefixes permitted to participate in *target*'s stack."""
    return frozenset(ALLOWED_LEARNERS_BY_TARGET.get(target, DEFAULT_ALLOWED_LEARNERS))


def killed_learners(target: str) -> frozenset[str]:
    """Known learner prefixes explicitly *not* permitted for *target*."""
    return LEARNER_PREFIXES - allowed_learners(target)


def learner_prefix_of(path: Path | str) -> Optional[str]:
    """
    Return the base-learner prefix encoded in an OOF filename, or None.

    ``lgbm_receiving_yards_WR_20260809.csv`` → ``"lgbm"``. Only recognised
    prefixes are returned, so ``stack_...`` and ``ridge_...`` yield None.
    """
    stem = Path(path).name.split("_", 1)[0].lower()
    return stem if stem in LEARNER_PREFIXES else None


def assert_learner_policy(
    target: str,
    learners: Iterable[str],
    *,
    context: str,
    require_minimum: bool = True,
) -> frozenset[str]:
    """
    Fail closed unless every learner in *learners* is allowed for *target*.

    This is the invariant that replaces ``--exclude tft,xgb``. It is called from
    inside the stacking code path so that omitting a CLI flag cannot widen the
    served learner set.

    Args:
        target:          Modelling target, e.g. ``"fantasy_ppr"``.
        learners:        Learner prefixes about to be used (or coef keys).
        context:         Human-readable call site, used in the error message.
        require_minimum: Also require at least :data:`MIN_BASE_LEARNERS`.

    Returns:
        The validated learner set.

    Raises:
        LearnerPolicyError: on any learner outside the allowlist, or too few.
    """
    seen = frozenset(str(item).lower() for item in learners)
    permitted = allowed_learners(target)
    violations = seen - permitted

    if violations:
        raise LearnerPolicyError(
            f"{context}: learner(s) {sorted(violations)} are not permitted for "
            f"target={target!r}. Allowed: {sorted(permitted)}. "
            f"Killed: {sorted(killed_learners(target))}. "
            "This is the Phase-5 two-learner contract; it is enforced here "
            "rather than via a CLI flag precisely so that a forgotten argument "
            "cannot resurrect a killed learner. To change the policy, edit "
            "ALLOWED_LEARNERS_BY_TARGET in ml/artifact_manifest.py."
        )

    if require_minimum and len(seen) < MIN_BASE_LEARNERS:
        raise LearnerPolicyError(
            f"{context}: need at least {MIN_BASE_LEARNERS} base learners for "
            f"target={target!r}; got {sorted(seen) or 'none'}."
        )

    return seen


def assert_coef_keys_allowed(
    target: str,
    coefs: Mapping[str, object],
    *,
    context: str,
    require_minimum: bool = True,
) -> None:
    """
    Reject a coefficient mapping carrying a killed learner's key.

    ``intercept`` is metadata, not a learner, so it is exempt. Everything else
    that names a recognised learner prefix must be on the allowlist; an
    unrecognised key is a schema problem and is also rejected.
    """
    learner_keys: set[str] = set()
    for raw in coefs:
        key = str(raw).lower()
        if key == "intercept":
            continue
        # Coef keys are written either bare ("lgbm") or as pred columns
        # ("lgbm_pred"); normalise both.
        prefix = key[:-5] if key.endswith("_pred") else key
        if prefix not in LEARNER_PREFIXES:
            raise LearnerPolicyError(
                f"{context}: coefficient key {raw!r} names no recognised base "
                f"learner. Recognised: {sorted(LEARNER_PREFIXES)}."
            )
        learner_keys.add(prefix)

    assert_learner_policy(
        target,
        learner_keys,
        context=context,
        require_minimum=require_minimum,
    )


# ── Digests ───────────────────────────────────────────────────────────────────


def sha256_of(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_digest(path: Path | str, expected: str) -> None:
    """
    Raise :class:`ArtifactIntegrityError` unless *path* hashes to *expected*.
    """
    actual = sha256_of(path)
    if actual.lower() != str(expected).lower():
        raise ArtifactIntegrityError(
            f"SHA-256 mismatch for {path}: manifest pins {expected}, file is "
            f"{actual}. Refusing to load. The artifact has been regenerated or "
            "modified without re-freezing the release manifest; restore the "
            "pinned artifact or re-freeze deliberately."
        )


#: First line of every Git LFS pointer file. A model checked out without
#: ``git lfs pull`` is this text, not the model (decision 0012, D1).
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path | str) -> bool:
    """True if *path* is a Git LFS pointer file (it starts with the LFS spec line)."""
    with open(path, "rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


# ── Manifest ──────────────────────────────────────────────────────────────────


#: Memoized successful digest verifications, keyed by
#: (path, expected digest, size, mtime_ns). Only ever holds *verified* entries,
#: so a hit is a statement that this exact file content already passed.
_VERIFIED: set[tuple[str, str, int, int]] = set()


@dataclass(frozen=True)
class ArtifactManifest:
    """A loaded release manifest, indexed for artifact lookup."""

    path: Path
    root: Path
    raw: Mapping[str, object]
    digests: Mapping[str, str]
    #: (stat, position) → repo-relative path, parsed from the pinned stack list.
    stack_index: Mapping[tuple[str, str], str]
    #: (stat, position) → repo-relative executable Ridge coefficient artifact.
    coefficient_index: Mapping[tuple[str, str], str]
    #: (learner, stat, position) → repo-relative fitted base-model file (decision 0012).
    #: Empty when the manifest has no ``artifacts.base_models`` (history serving only).
    model_index: Mapping[tuple[str, str, str], str] = field(default_factory=dict)
    #: (learner, stat, position) → that model's entry in the pinned MODELS.json.
    model_meta: Mapping[tuple[str, str, str], Mapping[str, object]] = field(default_factory=dict)

    # ── lookup ────────────────────────────────────────────────────────────

    def stack_relpath(self, stat: str, position: str) -> str:
        """Repo-relative path of the pinned stack artifact for a cell."""
        key = (stat, position.upper())
        try:
            return self.stack_index[key]
        except KeyError:
            raise ManifestEntryMissing(
                f"{self.path} pins no stack artifact for stat={stat!r} "
                f"position={position!r}. Pinned cells: "
                f"{sorted(self.stack_index)}. Selection is manifest-only by "
                "design — there is no mtime or glob fallback — so add the "
                "artifact to the manifest (with its digest) or drop the cell."
            ) from None

    def resolve_stack(self, stat: str, position: str, *, verify: bool = True) -> Path:
        """
        Absolute path to the pinned stack artifact for a cell, integrity-checked.

        Raises:
            ManifestEntryMissing:    cell is not pinned.
            FileNotFoundError:       pinned path is absent on disk.
            ArtifactIntegrityError:  content does not match the pinned digest.
        """
        relpath = self.stack_relpath(stat, position)
        absolute = self.root / relpath
        if not absolute.exists():
            raise FileNotFoundError(
                f"Manifest {self.path} pins {relpath} for {stat}/{position}, "
                "but the file is missing. Restore it from the release archive; "
                "do not regenerate it in place."
            )
        if verify:
            self.require_digest(relpath, absolute)
        return absolute

    def require_digest(self, relpath: str, absolute: Optional[Path] = None) -> str:
        """
        Verify *relpath* against its pinned digest and return that digest.

        Results are memoized on (path, expected digest, size, mtime_ns) for the
        life of the process. Without this, a request that resolves four position
        stacks re-hashes several MB every time — and `/draft/board` does exactly
        that. The cache key includes size and mtime, so a file replaced under a
        running process is re-verified rather than trusted.
        """
        expected = self.digests.get(relpath)
        if not expected:
            raise ManifestEntryMissing(
                f"{self.path} lists {relpath} but records no SHA-256 for it "
                "under 'artifact_digests'. Refusing to load an unpinned "
                "artifact."
            )
        target = absolute or (self.root / relpath)
        stat = target.stat()
        key = (str(target), expected.lower(), stat.st_size, stat.st_mtime_ns)
        if key in _VERIFIED:
            return expected
        verify_file_digest(target, expected)
        _VERIFIED.add(key)
        return expected

    def resolve_stacks(
        self, cells: Iterable[tuple[str, str]], *, verify: bool = True
    ) -> dict[tuple[str, str], Path]:
        """Resolve many cells, failing on the first problem."""
        return {
            (stat, position.upper()): self.resolve_stack(stat, position, verify=verify)
            for stat, position in cells
        }

    def resolve_attribution(self, stat: str, position: str, *, verify: bool = True) -> Path:
        """
        Absolute path to the pinned bucket-attribution file for a cell.

        The file sits beside the pinned stack as ``attr_<stack stem>.parquet``
        and must be listed under ``artifacts`` (so ``artifact_digests`` pins
        it). Missing, unpinned and modified files all raise; there is no
        fallback, because a number re-weighted from the wrong attributions
        would be a plausible wrong answer.
        """
        from ml.bucket_attribution import attribution_path_for

        relpath = attribution_path_for(self.stack_relpath(stat, position)).as_posix()
        if relpath not in self.digests:
            raise ManifestEntryMissing(
                f"{self.path} pins no bucket attributions for {stat}/{position} "
                f"(expected {relpath} under 'artifacts' and 'artifact_digests')."
            )
        absolute = self.root / relpath
        if not absolute.exists():
            raise FileNotFoundError(
                f"Manifest pins {relpath} for {stat}/{position}, but the file is missing."
            )
        if verify:
            self.require_digest(relpath, absolute)
        return absolute

    def resolve_coefficient(self, stat: str, position: str, *, verify: bool = True) -> Path:
        """Resolve the executable, SHA-pinned Ridge coefficient artifact for a cell."""
        key = (stat, position.upper())
        try:
            relpath = self.coefficient_index[key]
        except KeyError:
            raise ManifestEntryMissing(
                f"{self.path} pins no Ridge coefficient for {stat}/{position}. "
                "Artifact-backed inference cannot fall back to an unpinned coefficient file."
            ) from None
        absolute = self.root / relpath
        if not absolute.exists():
            raise FileNotFoundError(f"Manifest pins missing Ridge coefficient {relpath} for {stat}/{position}")
        if verify:
            self.require_digest(relpath, absolute)
        return absolute

    # ── base models (decision 0012) ───────────────────────────────────────

    @property
    def has_base_models(self) -> bool:
        return bool(self.model_index)

    def require_base_models(self) -> None:
        """The forward producer's entry check: refuse a manifest that pins no models."""
        if not self.model_index:
            raise ManifestEntryMissing(
                f"{self.path} has no 'artifacts.base_models'. The forward producer serves only "
                "fitted models pinned by this release; history serving is unaffected. "
                "Add the models and re-freeze (decision 0012)."
            )

    def resolve_base_model(
        self, stat: str, position: str, learner: str, *, verify: bool = True
    ) -> Path:
        """
        Absolute path to the pinned fitted base model for one cell and learner.

        Raises:
            ManifestEntryMissing:    no model is pinned (including a learner whose
                                     ridge weight is 0.0, which is never pinned).
            FileNotFoundError:       pinned path is absent on disk.
            LfsPointerError:         the file is a Git LFS pointer, not the model.
            ArtifactIntegrityError:  content does not match the pinned digest.
        """
        self.require_base_models()
        key = (learner, stat, position.upper())
        try:
            relpath = self.model_index[key]
        except KeyError:
            raise ManifestEntryMissing(
                f"{self.path} pins no base model for learner={learner!r} stat={stat!r} "
                f"position={position!r}. A learner with ridge weight 0.0 is never pinned."
            ) from None
        absolute = self.root / relpath
        if not absolute.exists():
            raise FileNotFoundError(
                f"Manifest pins {relpath} for {learner}/{stat}/{position}, but the file is missing."
            )
        if verify:
            if is_lfs_pointer(absolute):
                raise LfsPointerError(
                    f"{relpath} is a Git LFS pointer, not the model. Run `git lfs pull` on the "
                    "Mac, check that `git lfs ls-files` shows `*` for every model, then sync the "
                    "smudged files (docs/runbooks/vps4-causal-rerun.md)."
                )
            self.require_digest(relpath, absolute)
        return absolute

    def base_model_meta(self, stat: str, position: str, learner: str) -> Mapping[str, object]:
        """The pinned MODELS.json entry for one model (features, seasons, params, ...)."""
        self.require_base_models()
        try:
            return self.model_meta[(learner, stat, position.upper())]
        except KeyError:
            raise ManifestEntryMissing(
                f"{self.path} pins no base model for {learner}/{stat}/{position}."
            ) from None

    # ── protection ────────────────────────────────────────────────────────

    def protected_relpaths(self) -> frozenset[str]:
        """
        Every artifact the manifest pins, as repo-relative paths.

        Anything in this set is a *serving* artifact: no clean-slate or purge
        routine may delete it. See ``scripts/guard_release_artifacts.py``.
        """
        return frozenset(self.digests)

    def protects(self, path: Path | str) -> bool:
        """True if *path* (absolute or repo-relative) is a pinned artifact."""
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(self.root.resolve())
            except ValueError:
                return False
        return candidate.as_posix() in self.digests


def _parse_stack_relpath(relpath: str) -> Optional[tuple[str, str]]:
    """
    ``ml/oof/stack_fantasy_ppr_WR_20260809.csv`` → ``("fantasy_ppr", "WR")``.

    Returns None for anything that is not a positioned stack CSV, so
    non-stack manifest entries (reports, coef files) are simply skipped.
    """
    name = Path(relpath).name
    if not name.startswith("stack_") or not name.endswith(".csv"):
        return None
    body = name[len("stack_") : -len(".csv")]
    # Trailing token is the date/run stamp.
    body, _, stamp = body.rpartition("_")
    if not body or not stamp:
        return None
    # Next token from the right is the position; the remainder is the stat,
    # which may itself contain underscores (fantasy_ppr, receiving_yards).
    stat, _, position = body.rpartition("_")
    if not stat or position.upper() not in POSITIONS:
        return None
    return stat, position.upper()


def _parse_coefficient_relpath(relpath: str) -> Optional[tuple[str, str]]:
    """Parse ``ridge_fantasy_ppr_WR_coefs.json`` into its serving cell."""
    name = Path(relpath).name
    if not name.startswith("ridge_") or not name.endswith("_coefs.json"):
        return None
    body = name[len("ridge_") : -len("_coefs.json")]
    stat, _, position = body.rpartition("_")
    if not stat or position.upper() not in POSITIONS:
        return None
    return stat, position.upper()


MODELS_META_NAME = "MODELS.json"
MODELS_META_SCHEMA_VERSION = 1
#: File extension each learner's native model format uses.
BASE_MODEL_EXTENSIONS: Mapping[str, str] = {"lgbm": ".txt", "catboost": ".cbm"}


def _parse_base_model_relpath(relpath: str) -> Optional[tuple[str, str, str]]:
    """``.../lgbm_receiving_yards_WR.txt`` → ``("lgbm", "receiving_yards", "WR")``, else None."""
    name = Path(relpath).name
    for learner, ext in BASE_MODEL_EXTENSIONS.items():
        if name.startswith(learner + "_") and name.endswith(ext):
            body = name[len(learner) + 1 : -len(ext)]
            stat, _, position = body.rpartition("_")
            if stat and position in POSITIONS:
                return learner, stat, position
    return None


def _check_base_models(
    manifest_path_: Path,
    base: Path,
    digests: Mapping[str, str],
    base_model_relpaths: list[str],
    coefficient_index: Mapping[tuple[str, str], str],
) -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str, str], dict]]:
    """
    Load-time cross-check of ``artifacts.base_models`` (decision 0012, section 4).

    Everything checkable without reading the model files themselves is checked
    here, so a bad release fails at load and not at the first prediction. The
    model files are verified by digest when resolved. Returns the model index
    and the per-model MODELS.json entries.
    """
    from ml.season_constants import LAST_COMPLETE_SEASON

    where = f"Release manifest {manifest_path_}"
    if not base_model_relpaths:
        raise ManifestError(f"{where}: 'artifacts.base_models' is present but empty.")

    model_index: dict[tuple[str, str, str], str] = {}
    meta_relpaths: list[str] = []
    for relpath in base_model_relpaths:
        if Path(relpath).name == MODELS_META_NAME:
            meta_relpaths.append(relpath)
            continue
        key = _parse_base_model_relpath(relpath)
        if key is None:
            raise ManifestError(
                f"{where}: {relpath} is not a recognised base-model file "
                f"(expected <learner>_<stat>_<POS>{{.txt for lgbm, .cbm for catboost}})."
            )
        if key in model_index and model_index[key] != relpath:
            raise ManifestError(f"{where} pins two files for base model {'/'.join(key)}.")
        model_index[key] = relpath
    if len(meta_relpaths) != 1:
        raise ManifestError(
            f"{where}: 'artifacts.base_models' must list exactly one {MODELS_META_NAME}; "
            f"found {len(meta_relpaths)}."
        )
    meta_relpath = meta_relpaths[0]
    unpinned = sorted(r for r in [*model_index.values(), meta_relpath] if r not in digests)
    if unpinned:
        raise ManifestError(f"{where} records no SHA-256 for base-model file(s): {unpinned}.")

    # MODELS.json: pinned, then read. A changed file must not be trusted for its own checks.
    meta_path = base / meta_relpath
    if not meta_path.exists():
        raise ManifestError(f"{where} pins {meta_relpath}, but the file is missing.")
    verify_file_digest(meta_path, digests[meta_relpath])
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{meta_relpath} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("schema_version") != MODELS_META_SCHEMA_VERSION:
        raise ManifestError(f"{meta_relpath}: expected an object with schema_version {MODELS_META_SCHEMA_VERSION}.")
    release_id = meta.get("release_id")
    entries = meta.get("models")
    if not release_id or not isinstance(entries, list) or not entries:
        raise ManifestError(f"{meta_relpath}: needs a release_id and a non-empty 'models' list.")

    meta_dir = Path(meta_relpath).parent
    model_meta: dict[tuple[str, str, str], dict] = {}
    for entry in entries:
        for field_name in ("learner", "stat", "position", "file", "sha256", "release_id",
                           "train_seasons", "max_train_season", "features"):
            if not isinstance(entry, dict) or field_name not in entry:
                raise ManifestError(f"{meta_relpath}: a model entry lacks {field_name!r}.")
        key = (entry["learner"], entry["stat"], entry["position"])
        label = "/".join(key)
        relpath = (meta_dir / entry["file"]).as_posix()
        if _parse_base_model_relpath(relpath) != key:
            raise ManifestError(
                f"{meta_relpath}: entry {label} says file {entry['file']!r}, which names a different model."
            )
        if key in model_meta:
            raise ManifestError(f"{meta_relpath}: two entries for {label}.")
        if model_index.get(key) != relpath:
            raise ManifestError(
                f"{meta_relpath} lists {relpath} for {label}, but the manifest does not pin that file "
                "under 'artifacts.base_models'."
            )
        if str(entry["sha256"]).lower() != digests[relpath].lower():
            raise ManifestError(
                f"{meta_relpath}: sha256 for {label} differs from 'artifact_digests' ({relpath})."
            )
        if entry["release_id"] != release_id:
            raise ManifestError(
                f"{meta_relpath}: {label} has release_id {entry['release_id']!r}, "
                f"but the file's release_id is {release_id!r}. All models must come from one release."
            )
        seasons = entry["train_seasons"]
        if (not isinstance(seasons, list) or not seasons
                or not all(isinstance(x, int) for x in seasons)
                or max(seasons) != entry["max_train_season"]):
            raise ManifestError(f"{meta_relpath}: {label} train_seasons must be ints ending at max_train_season.")
        if entry["max_train_season"] > LAST_COMPLETE_SEASON:
            raise ManifestError(
                f"{meta_relpath}: {label} was fitted through {entry['max_train_season']}, past "
                f"LAST_COMPLETE_SEASON={LAST_COMPLETE_SEASON}."
            )
        features = entry["features"]
        if not isinstance(features, list) or not features or not all(isinstance(f, str) for f in features):
            raise ManifestError(f"{meta_relpath}: {label} needs a non-empty feature list.")
        model_meta[key] = entry
    unlisted = sorted("/".join(k) for k in set(model_index) - set(model_meta))
    if unlisted:
        raise ManifestError(f"{meta_relpath} (MODELS.json) has no entry for pinned model(s): {unlisted}.")

    # Cross-check against the pinned ridge files: every learner that carries weight has a
    # model, and no model is pinned that no ridge file uses.
    needed: set[tuple[str, str, str]] = set()
    for (stat, position), ridge_relpath in coefficient_index.items():
        ridge_path = base / ridge_relpath
        if not ridge_path.exists():
            raise ManifestError(f"{where} pins {ridge_relpath}, but the file is missing.")
        verify_file_digest(ridge_path, digests[ridge_relpath])
        try:
            coefs = json.loads(ridge_path.read_text())
            weights = coefs["weights"]
            order = coefs["learner_order"]
            for learner in order:
                if float(weights[learner]) != 0.0:
                    needed.add((learner, stat, position))
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"{ridge_relpath}: unreadable ridge coefficients: {exc!r}") from exc
    missing = sorted("/".join(k) for k in needed - set(model_index))
    if missing:
        raise ManifestError(
            f"{where} pins no base model for learner(s) with non-zero ridge weight: {missing} "
            "(learner/stat/position)."
        )
    extra = sorted("/".join(k) for k in set(model_index) - needed)
    if extra:
        raise ManifestError(
            f"{where} pins base model(s) that no ridge file uses (weight 0.0 or no ridge file): {extra}."
        )
    return model_index, model_meta


def _iter_artifact_relpaths(artifacts: object) -> Iterable[str]:
    """Flatten the ``artifacts`` block, which mixes strings and lists."""
    if isinstance(artifacts, str):
        yield artifacts
    elif isinstance(artifacts, Mapping):
        for value in artifacts.values():
            yield from _iter_artifact_relpaths(value)
    elif isinstance(artifacts, (list, tuple)):
        for value in artifacts:
            yield from _iter_artifact_relpaths(value)


def manifest_path(root: Optional[Path] = None) -> Path:
    """
    Resolve the manifest location.

    ``BASELINE_MANIFEST_PATH`` overrides the default so tests and alternate
    releases can point at their own manifest.
    """
    base = Path(root) if root else REPO_ROOT
    relpath = os.environ.get("BASELINE_MANIFEST_PATH") or DEFAULT_MANIFEST_RELPATH
    candidate = Path(relpath)
    return candidate if candidate.is_absolute() else base / candidate


def load_manifest(
    path: Optional[Path | str] = None,
    *,
    root: Optional[Path] = None,
) -> ArtifactManifest:
    """
    Load and index the release manifest.

    Raises:
        ManifestError: the manifest is missing, unparseable, or pins no
            artifacts. Every failure here is fail-closed; callers must not
            substitute a glob.
    """
    base = Path(root) if root else REPO_ROOT
    resolved = Path(path) if path else manifest_path(base)
    if not resolved.is_absolute():
        resolved = base / resolved

    if not resolved.exists():
        raise ManifestError(
            f"Release manifest not found at {resolved}. Artifact selection is "
            "manifest-pinned and fails closed; there is no mtime fallback."
        )

    try:
        raw = json.loads(resolved.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Release manifest {resolved} is not valid JSON: {exc}") from exc

    digests_raw = raw.get("artifact_digests")
    if not isinstance(digests_raw, Mapping) or not digests_raw:
        raise ManifestError(
            f"Release manifest {resolved} has no non-empty 'artifact_digests' "
            "map. Every served artifact must be pinned by SHA-256. Regenerate "
            "with scripts/freeze_baseline.py."
        )
    digests = {str(k): str(v) for k, v in digests_raw.items()}

    stack_index: dict[tuple[str, str], str] = {}
    coefficient_index: dict[tuple[str, str], str] = {}
    artifact_paths = list(_iter_artifact_relpaths(raw.get("artifacts", {})))
    for relpath in artifact_paths:
        cell = _parse_stack_relpath(relpath)
        if cell is not None:
            if cell in stack_index and stack_index[cell] != relpath:
                raise ManifestError(
                    f"Release manifest {resolved} pins two artifacts for "
                    f"{cell[0]}/{cell[1]}: {stack_index[cell]} and {relpath}. "
                    "A cell must have exactly one serving artifact."
                )
            stack_index[cell] = relpath

        coefficient = _parse_coefficient_relpath(relpath)
        if coefficient is not None:
            if coefficient in coefficient_index and coefficient_index[coefficient] != relpath:
                raise ManifestError(
                    f"Release manifest {resolved} pins two Ridge coefficients for "
                    f"{coefficient[0]}/{coefficient[1]}"
                )
            coefficient_index[coefficient] = relpath

    if not stack_index:
        raise ManifestError(
            f"Release manifest {resolved} pins no positioned stack artifacts "
            "under 'artifacts'."
        )

    # A pinned stack with no digest is an unpinned artifact wearing a pin.
    unpinned = sorted((set(stack_index.values()) | set(coefficient_index.values())) - set(digests))
    if unpinned:
        raise ManifestError(
            f"Release manifest {resolved} lists stack artifact(s) with no "
            f"SHA-256 in 'artifact_digests': {unpinned}."
        )

    model_index: dict[tuple[str, str, str], str] = {}
    model_meta: dict[tuple[str, str, str], dict] = {}
    artifacts_block = raw.get("artifacts", {})
    if isinstance(artifacts_block, Mapping) and "base_models" in artifacts_block:
        model_index, model_meta = _check_base_models(
            resolved,
            base,
            digests,
            list(_iter_artifact_relpaths(artifacts_block["base_models"])),
            coefficient_index,
        )

    return ArtifactManifest(
        path=resolved,
        root=base,
        raw=raw,
        digests=digests,
        stack_index=stack_index,
        coefficient_index=coefficient_index,
        model_index=model_index,
        model_meta=model_meta,
    )


# Module-level cache: the manifest is immutable for the life of a process, and
# every serving request would otherwise re-read and re-hash it.
_CACHED: dict[tuple[str, str], ArtifactManifest] = {}


def get_manifest(
    path: Optional[Path | str] = None,
    *,
    root: Optional[Path] = None,
    use_cache: bool = True,
) -> ArtifactManifest:
    """Cached :func:`load_manifest`. Pass ``use_cache=False`` in tests."""
    base = Path(root) if root else REPO_ROOT
    resolved = Path(path) if path else manifest_path(base)
    key = (str(base), str(resolved))
    if use_cache and key in _CACHED:
        return _CACHED[key]
    loaded = load_manifest(resolved, root=base)
    if use_cache:
        _CACHED[key] = loaded
    return loaded


def clear_cache() -> None:
    """Drop the manifest and verification caches (tests, deliberate re-freeze)."""
    _CACHED.clear()
    _VERIFIED.clear()
