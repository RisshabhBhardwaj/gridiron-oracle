"""
Regression locks for the artifact-containment failures (C-05, C-06, C-16,
C-17, C-20).

Nothing tested any of this before. Each class below locks one mechanism that
previously allowed the repo to ship a different model than the one that was
evaluated:

* ``TestLearnerPolicy``      — C-05: killed learners cannot re-enter a stack.
* ``TestArtifactPinning``    — C-06: selection is manifest-pinned, not mtime.
* ``TestManifestIntegrity``  — C-06: a digest mismatch refuses to load.
* ``TestDiscoveryHygiene``   — C-17: discovery refuses foreign learners and
                                inconsistent fold maps.
* ``TestNoPurge``            — C-16: no routine may delete release artifacts.

``TestArtifactPinning`` is written to fail against tag
``audit-baseline-2026-08-09``: it resolves whichever selector function the
module exposes, so the pre-fix mtime-based implementation is exercised and
returns the stale artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

import ml.artifact_manifest as am
from ml.artifact_manifest import (
    ArtifactIntegrityError,
    LearnerPolicyError,
    ManifestEntryMissing,
    ManifestError,
    assert_coef_keys_allowed,
    assert_learner_policy,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _stack_frame(y_pred: float, *, seasons=(2024,), n: int = 6) -> pd.DataFrame:
    """A minimal but schema-valid stack OOF frame with a recognisable y_pred."""
    rows = []
    for season in seasons:
        for i in range(n):
            rows.append(
                {
                    "player_id": f"p{i:02d}",
                    "game_id": f"{season}_w{i:02d}",
                    "season": season,
                    "week": i + 1,
                    "y_true": y_pred + 1.0,
                    "y_pred": y_pred,
                    "fold_idx": 0,
                    "lgbm_pred": y_pred,
                    "catboost_pred": y_pred,
                }
            )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pinned_repo(tmp_path, monkeypatch):
    """
    A throwaway repo root holding two candidate TE stacks, where the *stale*
    one has the newer mtime.

    ``_20260809`` is the pinned serving artifact (y_pred 20.0). ``_20260807``
    is the poisoned four-learner predecessor (y_pred 99.0) and is deliberately
    touched last, so any mtime-based selector prefers it. A correct selector
    returns the pinned file regardless.
    """
    oof = tmp_path / "ml" / "oof"
    oof.mkdir(parents=True)
    releases = tmp_path / "releases"
    releases.mkdir()

    pinned = oof / "stack_fantasy_ppr_TE_20260809.csv"
    stale = oof / "stack_fantasy_ppr_TE_20260807.csv"
    _stack_frame(20.0).to_csv(pinned, index=False)
    _stack_frame(99.0).to_csv(stale, index=False)

    manifest = {
        "model_version": "test_stack",
        "artifacts": {
            "stacks_fantasy_ppr": ["ml/oof/stack_fantasy_ppr_TE_20260809.csv"],
        },
        "artifact_digests": {
            "ml/oof/stack_fantasy_ppr_TE_20260809.csv": _sha256(pinned),
        },
    }
    manifest_path = releases / "current_baseline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Make the *stale* file the newest thing in the directory.
    stale_ns = pinned.stat().st_mtime_ns + 5_000_000_000
    os.utime(stale, ns=(stale_ns, stale_ns))
    assert stale.stat().st_mtime > pinned.stat().st_mtime, "fixture must invert mtimes"

    monkeypatch.setattr(am, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("BASELINE_MANIFEST_PATH", raising=False)
    am.clear_cache()
    yield tmp_path, pinned, stale
    am.clear_cache()


# ── Lock 1 — C-05: learner policy ─────────────────────────────────────────────


class TestLearnerPolicy:
    """The two-learner contract is enforced in code, not by a CLI flag."""

    def test_killed_learners_are_not_allowed(self):
        assert am.allowed_learners("fantasy_ppr") == {"lgbm", "catboost"}
        assert am.killed_learners("fantasy_ppr") == {"xgb", "tft"}

    @pytest.mark.parametrize("killed", ["xgb", "tft"])
    def test_assert_learner_policy_rejects_killed_learner(self, killed):
        with pytest.raises(LearnerPolicyError) as exc:
            assert_learner_policy(
                "fantasy_ppr", ["lgbm", "catboost", killed], context="test"
            )
        assert killed in str(exc.value)

    def test_assert_learner_policy_accepts_the_shipped_pair(self):
        assert assert_learner_policy(
            "fantasy_ppr", ["lgbm", "catboost"], context="test"
        ) == {"lgbm", "catboost"}

    def test_single_learner_is_not_a_stack(self):
        with pytest.raises(LearnerPolicyError):
            assert_learner_policy("fantasy_ppr", ["lgbm"], context="test")

    @pytest.mark.parametrize(
        "coefs",
        [
            {"lgbm": 0.4, "catboost": 0.6, "xgb": 0.1, "intercept": 0.0},
            {"lgbm_pred": 0.4, "catboost_pred": 0.5, "tft_pred": 0.1, "intercept": 0.0},
        ],
    )
    def test_no_coef_file_may_carry_a_killed_learner_key(self, coefs):
        """The coef file is what inference reads; it is gated independently."""
        with pytest.raises(LearnerPolicyError):
            assert_coef_keys_allowed("fantasy_ppr", coefs, context="test")

    def test_clean_two_learner_coefs_are_accepted(self):
        assert_coef_keys_allowed(
            "fantasy_ppr",
            {"lgbm": 0.42, "catboost": 0.58, "intercept": 0.1},
            context="test",
        )

    def test_unrecognised_coef_key_is_rejected(self):
        with pytest.raises(LearnerPolicyError):
            assert_coef_keys_allowed(
                "fantasy_ppr", {"lgbm": 0.5, "catboost": 0.5, "mystery": 1.0}, context="test"
            )

    def test_every_shipped_coef_file_satisfies_the_policy(self):
        """The artifacts and the code must agree, in the direction of the code."""
        coef_files = sorted((REPO_ROOT / "ml" / "oof").glob("ridge_*_coefs.json"))
        assert coef_files, "expected shipped ridge coef files"
        for path in coef_files:
            # ridge_{target}_{POS}_coefs.json — strip prefix, suffix, position.
            body = path.stem[len("ridge_") : -len("_coefs")]
            target = body.rsplit("_", 1)[0] if body.rsplit("_", 1)[-1] in am.POSITIONS else body
            assert_coef_keys_allowed(
                target, json.loads(path.read_text()), context=path.name
            )

    def test_stacking_against_a_dir_with_killed_learner_oofs_raises(self, tmp_path):
        """A --oof-dir restack must refuse a contaminated directory outright."""
        from ml.stacking_ensemble import _discover_oof_files

        oof = tmp_path / "oof"
        oof.mkdir()
        for prefix in ("lgbm", "catboost"):
            _stack_frame(5.0).to_csv(oof / f"{prefix}_receiving_yards_WR_20260809.csv", index=False)
        _stack_frame(5.0).to_csv(oof / "xgb_receiving_yards_WR_20260809.csv", index=False)

        with pytest.raises(LearnerPolicyError) as exc:
            _discover_oof_files(oof, "receiving_yards")
        assert "xgb_receiving_yards_WR_20260809.csv" in str(exc.value)

    def test_clean_dir_discovers_exactly_the_allowed_learners(self, tmp_path):
        from ml.stacking_ensemble import _discover_oof_files

        oof = tmp_path / "oof"
        oof.mkdir()
        for prefix in ("lgbm", "catboost"):
            _stack_frame(5.0).to_csv(oof / f"{prefix}_receiving_yards_WR_20260809.csv", index=False)

        found = {am.learner_prefix_of(p) for p in _discover_oof_files(oof, "receiving_yards")}
        assert found == {"lgbm", "catboost"}


# ── Lock 2 — C-06: artifact pinning beats mtime ────────────────────────────────


def _draft_stack_selector():
    """
    The draft module's stack selector, under either name.

    Resolving dynamically is what makes this lock fail against
    audit-baseline-2026-08-09 rather than erroring: the pre-fix module exposes
    ``_latest_stack_oof`` (max by mtime) and the assertion below catches it
    returning the stale artifact.
    """
    from backend.app.api import draft

    for name in ("_pinned_stack_oof", "_latest_stack_oof"):
        selector = getattr(draft, name, None)
        if selector is not None:
            return selector
    pytest.fail("backend.app.api.draft exposes no stack selector")


def _materialize_stack_selector():
    """The materializer's stack selector, under either name (see above)."""
    import importlib

    module = importlib.import_module("scripts.materialize_stack_projections")
    for name in ("_pinned_stack", "_latest_stack"):
        selector = getattr(module, name, None)
        if selector is not None:
            return selector
    pytest.fail("materialize_stack_projections exposes no stack selector")


class TestArtifactPinning:
    """
    With two candidate stacks present and mtimes inverted, every reader must
    still select the manifest-pinned artifact.

    Two reproduced pre-fix failure modes motivate this:

    a. ``touch`` on the stale file makes ``max(paths, key=mtime)`` prefer it;
    b. when the filesystem's timestamp granularity ties the two files (1-second
       granularity is common on ext3, HFS+ and various container/network
       mounts), ``max()`` returns the *first* maximal element — the lexically
       earliest, i.e. the older-dated poisoned file. Verified: with mtimes tied,
       the pre-fix draft selector returned ``stack_fantasy_ppr_TE_20260807.csv``.
    """

    def test_draft_selects_pinned_not_newest(self, pinned_repo):
        _, pinned, stale = pinned_repo
        selected = _draft_stack_selector()("TE")
        assert selected is not None, "pinned TE stack should resolve"
        assert Path(selected).name == pinned.name, (
            f"draft selected {Path(selected).name}; the manifest pins "
            f"{pinned.name}. {stale.name} has the newer mtime, which must not "
            "decide which artifact is served."
        )

    def test_materialize_selects_pinned_not_newest(self, pinned_repo):
        _, pinned, stale = pinned_repo
        selected = _materialize_stack_selector()("fantasy_ppr", "TE")
        assert selected is not None
        assert Path(selected).name == pinned.name, (
            f"materialize selected {Path(selected).name}; the manifest pins "
            f"{pinned.name} ({stale.name} is merely newer)."
        )

    def test_adp_eval_reads_pinned_artifact(self, pinned_repo):
        """
        ADP evidence must come from the pinned artifact.

        Asserted on the *values*: the pinned stack's y_pred is 20.0 and the
        stale one's is 99.0, so reading the wrong file is visible in the result
        rather than only in a path.
        """
        from ml.adp_eval import load_stack_season_ppr

        frame = load_stack_season_ppr(2024, manifest=load_manifest())
        assert not frame.empty
        # Six distinct players, one 2024 row each: every season sum is the
        # pinned file's y_pred. The stale file's 99.0 must appear nowhere.
        assert frame["fantasy_ppr"].tolist() == [pytest.approx(20.0)] * len(frame), (
            f"adp_eval read values from a file other than the pinned stack: "
            f"{frame['fantasy_ppr'].tolist()}"
        )
        assert 99.0 not in frame["fantasy_ppr"].values

    def test_tied_mtimes_still_resolve_to_the_pinned_artifact(self, pinned_repo):
        """Failure mode (b): equal mtimes must not make selection arbitrary."""
        _, pinned, stale = pinned_repo
        tied = pinned.stat().st_mtime_ns
        for path in (pinned, stale):
            os.utime(path, ns=(tied, tied))
        assert pinned.stat().st_mtime == stale.stat().st_mtime
        assert Path(_draft_stack_selector()("TE")).name == pinned.name

    @pytest.mark.parametrize(
        "relpath",
        [
            "backend/app/api/draft.py",
            "backend/app/api/predict.py",
            "scripts/materialize_stack_projections.py",
            "ml/adp_eval.py",
        ],
    )
    def test_no_serving_path_reaches_a_stack_by_glob(self, relpath):
        """
        Stronger than the mtime grep: a ``sorted(glob(...))[-1]`` selects an
        unpinned artifact while matching neither the mtime pattern nor ``ls -t``.
        No serving path may reach a stack artifact by pattern at all.
        """
        path = REPO_ROOT / relpath
        if not path.exists():
            pytest.skip(f"{relpath} not present")
        code = "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.strip().startswith("#")
        )
        for token in ("glob(", "rglob(", "iglob("):
            for line in code.splitlines():
                if token in line and "stack_" in line:
                    pytest.fail(
                        f"{relpath} selects a stack artifact by glob: {line.strip()!r}"
                    )

    def test_unpinned_cell_does_not_silently_fall_back_to_a_glob(self, pinned_repo):
        """
        A cell the manifest does not pin must fail, not be guessed at.

        The stale TE file is on disk and matches ``stack_fantasy_ppr_*`` — an
        implementation that fell back to globbing would happily serve it for an
        unpinned cell.
        """
        with pytest.raises(ManifestEntryMissing):
            load_manifest().resolve_stack("fantasy_ppr", "WR")


# ── Lock 3 — C-06: manifest integrity ─────────────────────────────────────────


class TestManifestIntegrity:
    def test_digest_mismatch_raises_rather_than_loading(self, pinned_repo):
        tmp_path, pinned, _ = pinned_repo
        # Modify the pinned artifact without re-freezing the manifest.
        pinned.write_text(pinned.read_text() + "p99,2024_w99,2024,99,1.0,1.0,0,1.0,1.0\n")
        am.clear_cache()
        with pytest.raises(ArtifactIntegrityError) as exc:
            load_manifest().resolve_stack("fantasy_ppr", "TE")
        assert "SHA-256 mismatch" in str(exc.value)

    def test_missing_pinned_file_raises(self, pinned_repo):
        _, pinned, _ = pinned_repo
        pinned.unlink()
        am.clear_cache()
        with pytest.raises(FileNotFoundError):
            load_manifest().resolve_stack("fantasy_ppr", "TE")

    def test_manifest_without_digests_is_refused(self, tmp_path, monkeypatch):
        releases = tmp_path / "releases"
        releases.mkdir()
        (releases / "current_baseline.json").write_text(
            json.dumps({"artifacts": {"s": ["ml/oof/stack_fantasy_ppr_TE_20260809.csv"]}})
        )
        monkeypatch.setattr(am, "REPO_ROOT", tmp_path)
        am.clear_cache()
        with pytest.raises(ManifestError, match="artifact_digests"):
            load_manifest()

    def test_pinned_stack_without_a_digest_is_refused(self, tmp_path, monkeypatch):
        """A listed artifact with no digest is an unpinned artifact wearing a pin."""
        oof = tmp_path / "ml" / "oof"
        oof.mkdir(parents=True)
        releases = tmp_path / "releases"
        releases.mkdir()
        a = "ml/oof/stack_fantasy_ppr_TE_20260809.csv"
        b = "ml/oof/stack_fantasy_ppr_WR_20260809.csv"
        for rel in (a, b):
            _stack_frame(1.0).to_csv(tmp_path / rel, index=False)
        (releases / "current_baseline.json").write_text(
            json.dumps(
                {
                    "artifacts": {"s": [a, b]},
                    "artifact_digests": {a: _sha256(tmp_path / a)},
                }
            )
        )
        monkeypatch.setattr(am, "REPO_ROOT", tmp_path)
        am.clear_cache()
        with pytest.raises(ManifestError, match="no\n?\\s*SHA-256|SHA-256"):
            load_manifest()

    def test_missing_manifest_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(am, "REPO_ROOT", tmp_path)
        am.clear_cache()
        with pytest.raises(ManifestError):
            load_manifest()

    def test_shipped_manifest_verifies_against_disk(self):
        """Every digest in the real manifest matches the real artifact."""
        am.clear_cache()
        manifest = load_manifest(REPO_ROOT / "releases" / "current_baseline.json", root=REPO_ROOT)
        assert manifest.stack_index, "manifest pins no stack cells"
        for (stat, position) in sorted(manifest.stack_index):
            manifest.resolve_stack(stat, position)  # raises on mismatch

    def test_shipped_manifest_pins_no_killed_learner_artifact(self):
        am.clear_cache()
        manifest = load_manifest(REPO_ROOT / "releases" / "current_baseline.json", root=REPO_ROOT)
        for relpath in manifest.protected_relpaths():
            prefix = am.learner_prefix_of(relpath)
            assert prefix not in am.LEARNER_PREFIXES - am.DEFAULT_ALLOWED_LEARNERS, (
                f"manifest pins a killed-learner artifact: {relpath}"
            )

    def test_shipped_manifest_pins_no_20260807_artifact(self):
        """The poisoned generation must not be reachable through the manifest."""
        am.clear_cache()
        manifest = load_manifest(REPO_ROOT / "releases" / "current_baseline.json", root=REPO_ROOT)
        offenders = [p for p in manifest.protected_relpaths() if "_20260807" in p]
        assert not offenders, f"manifest pins poisoned artifact(s): {offenders}"


# ── Lock 4 — C-17: discovery hygiene ──────────────────────────────────────────


class TestDiscoveryHygiene:
    def test_inconsistent_fold_to_season_mapping_raises(self, tmp_path):
        """
        Stacking keeps season/fold columns from the first file only, so inputs
        that disagree about what a fold index means would mislabel rows while
        the fold-ordering causality assert still passed.

        This is the shape of the real legacy data: the retired
        ``receiving_yards`` OOFs mapped fold 0 to 2022 (tft) and 2023
        (lgbm/xgb) while current runs map it to 2020.
        """
        from ml.stacking_ensemble import assert_consistent_fold_seasons

        a = _stack_frame(1.0, seasons=(2023,))
        b = _stack_frame(1.0, seasons=(2022,))  # same fold_idx, different season
        with pytest.raises(LearnerPolicyError, match="fold_idx"):
            assert_consistent_fold_seasons({Path("lgbm_a.csv"): a, Path("tft_b.csv"): b})

    def test_consistent_fold_map_is_accepted(self):
        from ml.stacking_ensemble import assert_consistent_fold_seasons

        a = _stack_frame(1.0, seasons=(2023,))
        b = _stack_frame(2.0, seasons=(2023,))
        assert_consistent_fold_seasons({Path("lgbm_a.csv"): a, Path("catboost_b.csv"): b})

    def test_discovery_rejects_mismatched_fold_maps_across_allowed_learners(self, tmp_path):
        """The check applies to allowed learners too, not just killed ones."""
        from ml.stacking_ensemble import _discover_oof_files

        oof = tmp_path / "oof"
        oof.mkdir()
        _stack_frame(1.0, seasons=(2023,)).to_csv(
            oof / "lgbm_receiving_yards_WR_20260809.csv", index=False
        )
        _stack_frame(1.0, seasons=(2020,)).to_csv(
            oof / "catboost_receiving_yards_WR_20260809.csv", index=False
        )
        with pytest.raises(LearnerPolicyError, match="fold_idx"):
            _discover_oof_files(oof, "receiving_yards")

    def test_no_legacy_receiving_yards_oofs_remain_in_the_repo(self):
        """
        The undated legacy OOFs are gone.

        ``{learner}_receiving_yards_{stamp}.csv`` with no position segment is the
        legacy shape (21-120 rows, WR-only, conflicting fold maps). Current
        artifacts always carry a position: ``lgbm_receiving_yards_WR_20260809.csv``.
        """
        offenders = []
        for path in (REPO_ROOT / "ml" / "oof").glob("*_receiving_yards_*.csv"):
            body = path.stem.split("_receiving_yards_", 1)[1]
            if body.split("_")[0].upper() not in am.POSITIONS:
                offenders.append(path.name)
        assert not offenders, (
            f"legacy position-less receiving_yards OOFs still present: {offenders}"
        )

    def test_no_killed_learner_oofs_are_tracked(self):
        """git must not carry xgb/tft OOF CSVs under the two-learner contract."""
        tracked = subprocess.run(
            ["git", "ls-files", "ml/oof"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        offenders = [
            p
            for p in tracked
            if p.endswith(".csv") and am.learner_prefix_of(p) in am.killed_learners("any")
        ]
        assert not offenders, f"killed-learner OOFs are still tracked: {offenders}"

    def test_no_20260807_artifacts_are_tracked(self):
        tracked = subprocess.run(
            ["git", "ls-files", "ml/oof"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        offenders = [p for p in tracked if "_20260807" in p and Path(p).name.startswith("stack_")]
        assert not offenders, f"poisoned stacks still tracked: {offenders}"

    def test_discovery_does_not_write_into_the_serving_directory(self, tmp_path):
        """
        Multi-file concatenations are staging artifacts and must not land beside
        the release artifacts, where every discovery glob would find them.
        """
        from ml.stacking_ensemble import _discover_oof_files

        oof = tmp_path / "oof"
        oof.mkdir()
        for prefix in ("lgbm", "catboost"):
            for pos in ("WR", "TE"):
                _stack_frame(1.0).to_csv(
                    oof / f"{prefix}_receiving_yards_{pos}_20260809.csv", index=False
                )
        _discover_oof_files(oof, "receiving_yards")
        assert not list(oof.glob("*_combined.csv")), (
            "combined blobs must not be written into the serving directory"
        )


# ── Lock 5 — C-16: no purge of release artifacts ──────────────────────────────


class TestNoPurge:
    """
    ``ml/train_all_models.sh`` used to run ``rm -f "$OOF_DIR"/*.csv`` on any
    non-``--resume`` start, deleting the tracked serving artifacts the manifest
    pins. Asserted at source level plus a unit test of the guard, because the
    script cannot safely be executed from a test (top-level ``cd``, ``exec >>LOG``,
    and it launches multi-hour trainers).
    """

    SCRIPT = REPO_ROOT / "ml" / "train_all_models.sh"

    def test_no_unconditional_oof_purge_in_train_all_models(self):
        src = self.SCRIPT.read_text()
        purge_lines = [
            line.strip()
            for line in src.splitlines()
            if 'rm -f "$OOF_DIR"' in line and not line.strip().startswith("#")
        ]
        for line in purge_lines:
            assert "*.csv" not in line or "PURGE_OOF" in src, (
                f"unguarded OOF purge still present: {line!r}"
            )
        # Any purge at all must be behind the explicit opt-in flag.
        if purge_lines:
            assert "--purge-oof" in src, (
                "an OOF purge exists but there is no --purge-oof opt-in flag"
            )
            assert "guard_release_artifacts.py" in src, (
                "an OOF purge exists but it does not consult the release-artifact guard"
            )

    def test_clean_start_no_longer_deletes_oof_csvs(self):
        """A non---resume start clears checkpoints only."""
        src = self.SCRIPT.read_text()
        clean_start = src.split('if [[ "$RESUME" == "false" ]]; then', 1)[1].split("fi", 1)[0]
        assert "$CHECKPOINT_DIR" in clean_start
        assert "$OOF_DIR" not in clean_start, (
            "clean start must not touch the OOF directory: "
            f"{clean_start.strip()!r}"
        )

    def test_guard_refuses_a_purge_that_would_delete_pinned_artifacts(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from guard_release_artifacts import protected_in

        am.clear_cache()
        protected = protected_in(
            REPO_ROOT / "ml" / "oof",
            manifest_path=REPO_ROOT / "releases" / "current_baseline.json",
        )
        assert protected, "guard found nothing to protect under ml/oof"
        assert all(p.startswith("ml/oof/") for p in protected)

    def test_guard_exits_nonzero_for_the_serving_directory(self):
        result = subprocess.run(
            [sys.executable, "scripts/guard_release_artifacts.py", "--check-purge", "ml/oof"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 3, (
            f"guard should refuse (exit 3); got {result.returncode}\n{result.stderr}"
        )
        assert "REFUSED" in result.stderr

    def test_guard_allows_a_directory_with_no_pinned_artifacts(self, tmp_path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from guard_release_artifacts import protected_in

        am.clear_cache()
        assert protected_in(
            tmp_path,
            manifest_path=REPO_ROOT / "releases" / "current_baseline.json",
        ) == []

    def test_guard_fails_closed_when_the_manifest_is_unreadable(self, tmp_path):
        bad = tmp_path / "broken.json"
        bad.write_text("{not json")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/guard_release_artifacts.py",
                "--check-purge",
                "ml/oof",
                "--manifest",
                str(bad),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 4, "unreadable manifest must fail closed"

    def test_required_cell_matrix_is_the_release_matrix_not_the_cross_product(self):
        """
        The required-cell check must use the manifest's release matrix.

        The trainer loops walk a stat×position cross product (37 combinations) in
        which many cells are irrelevant by design, while the release pins 15. An
        earlier version of this check required all 37, which made the script exit
        1 on a correct tree and left Step 4 unreachable — the same class of bug
        this session exists to remove, so it is locked.
        """
        src = self.SCRIPT.read_text()
        assert "--list-cells" in src, (
            "the required-cell matrix must come from the release manifest"
        )
        assert "REQUIRED_CELLS" in src
        # The fatal check must iterate REQUIRED_CELLS, not the STAT_SET loops.
        block = src.split("Expected-cell verification", 1)[1]
        assert 'for cell in "${REQUIRED_CELLS[@]}"' in block
        assert "STAT_SET" not in block, (
            "the fatal cell check must not iterate the stat×position cross product"
        )

    def test_release_matrix_is_satisfied_by_the_current_tree(self):
        """The 15 pinned cells all have a stack artifact, so the check can pass."""
        result = subprocess.run(
            [sys.executable, "scripts/guard_release_artifacts.py", "--list-cells"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        cells = [c for c in result.stdout.split() if c]
        assert cells, "release matrix is empty"
        missing = [
            c
            for c in cells
            if not list((REPO_ROOT / "ml" / "oof").glob(
                f"stack_{c.split(':')[0]}_{c.split(':')[1]}_*.csv"
            ))
        ]
        assert not missing, f"release matrix cells with no artifact: {missing}"

    def test_archive_step_moves_rather_than_copies(self):
        """
        ``cp -n`` left every "archived" four-learner stack in ml/oof/ — the direct
        cause of the poisoning. Archiving must remove the file from the serving
        directory.
        """
        script = (REPO_ROOT / "scripts" / "rebuild_fantasy_ppr_stack_phase5.sh").read_text()
        code = "\n".join(
            line for line in script.splitlines() if not line.strip().startswith("#")
        )
        assert "cp -n" not in code, "archive step must mv, not cp"
        assert 'mv "$f" "$ARCH/"' in code


# ── Trainer script hygiene (C-16) ─────────────────────────────────────────────


class TestTrainerScriptHygiene:
    TRAINERS = [
        "scripts/train_fantasy_ppr_oof.sh",
        "scripts/train_volume_oof.sh",
        "scripts/train_yardage_oof.sh",
        "ml/train_all_models.sh",
    ]

    @pytest.mark.parametrize("relpath", TRAINERS)
    def test_strict_mode(self, relpath):
        """Without -e, loops carry on after a failed cell and the script exits 0."""
        assert "set -euo pipefail" in (REPO_ROOT / relpath).read_text(), (
            f"{relpath} must use set -euo pipefail"
        )

    @pytest.mark.parametrize("relpath", TRAINERS)
    def test_syntax_is_valid(self, relpath):
        result = subprocess.run(
            ["bash", "-n", relpath], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("relpath", TRAINERS)
    def test_expected_cell_matrix_is_enforced(self, relpath):
        """A long job must exit non-zero when a declared cell is missing."""
        src = (REPO_ROOT / relpath).read_text()
        assert "MISSING" in src and "exit 1" in src, (
            f"{relpath} must fail when the expected-cell matrix is incomplete"
        )

    def test_checkpoint_contract_suffixes_match(self):
        """
        ``mark()`` wrote ``$1`` while ``done_ck()`` read ``$1.done``, so no
        checkpoint ever matched its own marker and every resume retrained
        everything.
        """
        src = (REPO_ROOT / "scripts" / "train_fantasy_ppr_oof.sh").read_text()
        assert 'mark()    { touch "$CK_DIR/$1.done"; }' in src or 'touch "$CK_DIR/$1.done"' in src
        assert '[[ -f "$CK_DIR/${ck}.done" ]]' in src

    @pytest.mark.parametrize(
        "relpath",
        ["scripts/train_fantasy_ppr_oof.sh", "scripts/train_volume_oof.sh", "scripts/train_yardage_oof.sh"],
    )
    def test_no_killed_learner_training_blocks(self, relpath):
        src = (REPO_ROOT / relpath).read_text()
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "ml.xgb_model" not in code, f"{relpath} still trains XGB"
        assert "ml.tft_model" not in code, f"{relpath} still trains TFT"

    def test_train_all_models_does_not_train_killed_learners(self):
        src = (REPO_ROOT / "ml" / "train_all_models.sh").read_text()
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "ml.xgb_model" not in code
        assert "ml.tft_model" not in code

    @pytest.mark.parametrize("relpath", TRAINERS + [
        "scripts/rebuild_fantasy_ppr_stack_phase5.sh",
        "scripts/retrain_fantasy_ppr_collapsed.sh",
        "scripts/train_passing_yards_qb.sh",
    ])
    def test_no_mtime_based_artifact_selection(self, relpath):
        """`ls -t` and `max(..., key=mtime)` are the same defect in shell clothing."""
        code = "\n".join(
            line
            for line in (REPO_ROOT / relpath).read_text().splitlines()
            if not line.strip().startswith("#")
        )
        assert "ls -t " not in code, f"{relpath} selects artifacts by mtime via `ls -t`"
        assert "st_mtime" not in code, f"{relpath} selects artifacts by mtime"


class TestFreezeBaselinePreservesPins:
    """
    A re-freeze must not drop the artifact pins.

    ``freeze_baseline.py`` never emitted ``artifacts`` at all, so running it
    would have overwritten the manifest without them — and since selection is
    now manifest-only and fails closed, that would break draft, materialize, ADP
    eval and conformal calibration at once. This is a hazard the pinning change
    introduces, so it is locked here.
    """

    def test_carry_forward_preserves_artifacts_and_recomputes_digests(self, tmp_path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from freeze_baseline import _carry_forward_artifact_pins

        payload: dict = {"model_version": "new"}
        _carry_forward_artifact_pins(
            payload, REPO_ROOT / "releases" / "current_baseline.json"
        )
        assert payload["artifacts"], "artifacts block must be carried forward"
        assert payload["artifact_digests"], "digests must be recomputed"
        assert "artifact_pins_missing" not in payload, (
            f"pinned artifacts absent from disk: {payload.get('artifact_pins_missing')}"
        )
        # The recomputed digests must agree with the committed manifest.
        committed = json.loads(
            (REPO_ROOT / "releases" / "current_baseline.json").read_text()
        )["artifact_digests"]
        assert payload["artifact_digests"] == committed

    def test_carry_forward_reports_missing_pins_rather_than_hiding_them(self, tmp_path):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from freeze_baseline import _carry_forward_artifact_pins

        manifest = tmp_path / "m.json"
        manifest.write_text(
            json.dumps({"artifacts": {"s": ["ml/oof/does_not_exist_20260809.csv"]}})
        )
        payload: dict = {}
        _carry_forward_artifact_pins(payload, manifest)
        assert payload["artifact_pins_missing"] == ["ml/oof/does_not_exist_20260809.csv"]

    def test_freeze_baseline_emits_the_keys_the_loader_requires(self):
        """The writer and the reader must agree on the manifest schema."""
        src = (REPO_ROOT / "scripts" / "freeze_baseline.py").read_text()
        assert "artifact_digests" in src
        assert "_carry_forward_artifact_pins(payload" in src


# ── C-20: conformal calibration is pinned ─────────────────────────────────────


class TestConformalCalibrationPinned:
    def test_conformal_does_not_glob_all_stacks(self):
        """
        Calibration concatenated *every* ``stack_{stat}_{pos}_*.csv`` match, so
        stale four-learner residuals fed the shipped intervals. One cell, one
        calibration source.
        """
        import inspect

        from ml.train import PipelineRunner

        src = inspect.getsource(PipelineRunner._apply_conformal_yardage_intervals)
        # Code-only patterns, so the explanatory docstring does not trip this.
        assert "glob.glob(" not in src, "conformal calibration must not glob stack files"
        assert "import glob" not in src
        assert "resolve_stack" in src, "conformal calibration must read the pinned artifact"
