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
from dataclasses import dataclass
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


# ── Exceptions ────────────────────────────────────────────────────────────────


class ManifestError(RuntimeError):
    """Base class for every fail-closed condition in this module."""


class ManifestEntryMissing(ManifestError):
    """The manifest does not pin an artifact for the requested cell."""


class ArtifactIntegrityError(ManifestError):
    """An artifact's SHA-256 does not match the digest the manifest pins."""


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


def assert_coef_keys_allowed(target: str, coefs: Mapping[str, object], *, context: str) -> None:
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

    assert_learner_policy(target, learner_keys, context=context)


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
    for relpath in _iter_artifact_relpaths(raw.get("artifacts", {})):
        cell = _parse_stack_relpath(relpath)
        if cell is None:
            continue
        if cell in stack_index and stack_index[cell] != relpath:
            raise ManifestError(
                f"Release manifest {resolved} pins two artifacts for "
                f"{cell[0]}/{cell[1]}: {stack_index[cell]} and {relpath}. "
                "A cell must have exactly one serving artifact."
            )
        stack_index[cell] = relpath

    if not stack_index:
        raise ManifestError(
            f"Release manifest {resolved} pins no positioned stack artifacts "
            "under 'artifacts'."
        )

    # A pinned stack with no digest is an unpinned artifact wearing a pin.
    unpinned = sorted(set(stack_index.values()) - set(digests))
    if unpinned:
        raise ManifestError(
            f"Release manifest {resolved} lists stack artifact(s) with no "
            f"SHA-256 in 'artifact_digests': {unpinned}."
        )

    return ArtifactManifest(
        path=resolved,
        root=base,
        raw=raw,
        digests=digests,
        stack_index=stack_index,
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
