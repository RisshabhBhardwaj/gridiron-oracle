"""
Repository-level invariants that documentation and evidence must satisfy.

Covers audit findings C-25 (PFR claims), C-29 (evidence manifest), and C-31
(metric labelling). These are drift guards: each one fails when code and prose
disagree, which is the failure mode all three findings share.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── C-25: PFR is retired as a scraper, live as an nflverse-derived source ────


def test_pfr_ingestion_is_documented_where_it_is_called() -> None:
    pbp = (REPO_ROOT / "pipeline" / "pbp_pipeline.py").read_text()
    if "load_pfr_advstats" not in pbp:
        pytest.skip("PFR ingestion has been retired; this guard no longer applies")

    data_sources = REPO_ROOT / "docs" / "DATA_SOURCES.md"
    assert data_sources.exists(), (
        "load_pfr_advstats is still called; docs/DATA_SOURCES.md must document it"
    )
    text = data_sources.read_text()
    assert "load_pfr_advstats" in text
    assert "drop_rate" in text


def test_retired_pfr_adapter_does_not_claim_the_project_is_pfr_free() -> None:
    """The retired-adapter docstring used to say "PFR is not on the critical
    path" while `drop_rate` was sourced from PFR advanced stats (C-25)."""
    source = (REPO_ROOT / "scraper" / "adapters" / "pro_football_ref.py").read_text()
    assert "not on the critical path" not in source
    if "load_pfr_advstats" in (REPO_ROOT / "pipeline" / "pbp_pipeline.py").read_text():
        assert "load_pfr_advstats" in source, (
            "the retired-scraper docstring must point at the live PFR-derived path"
        )


def test_retired_pfr_adapter_still_raises() -> None:
    from scraper.adapters.pro_football_ref import ProFootballRefAdapter

    with pytest.raises(RuntimeError, match="retired"):
        ProFootballRefAdapter()


# ── C-29: evidence under ignored paths is manifest-tracked ──────────────────


def test_releases_artifacts_tree_exists_with_a_policy() -> None:
    base = REPO_ROOT / "releases" / "artifacts"
    assert (base / "README.md").exists()
    assert (base / "MANIFEST.json").exists()


def test_releases_artifacts_is_not_gitignored() -> None:
    """The whole point is that new evidence shows up in `git status`."""
    result = subprocess.run(
        ["git", "check-ignore", "releases/artifacts/MANIFEST.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "releases/artifacts/ must not be ignored"


def test_evidence_manifest_schema() -> None:
    manifest = json.loads(
        (REPO_ROOT / "releases" / "artifacts" / "MANIFEST.json").read_text()
    )
    assert isinstance(manifest["artifacts"], list)
    assert manifest["artifacts"], "manifest must list the shipped evidence"

    # An entry carries either its own digest, or — when the serving manifest
    # already pins that path — a `sha256_authority` pointer to it. Storing a
    # second copy of a digest the serving manifest owns is what lets two
    # registries disagree, so the schema permits exactly one of the two.
    serving = json.loads(
        (REPO_ROOT / "releases" / "current_baseline.json").read_text()
    )
    pinned = set(serving["artifact_digests"])

    for entry in manifest["artifacts"]:
        for field in ("path", "status", "produced_by", "description"):
            assert entry.get(field), f"{entry.get('path')}: missing {field}"
        assert entry["status"] in {"frozen", "provisional"}

        path = entry["path"]
        assert not path.startswith("ml/oof/"), (
            f"{path}: serving artifacts belong to releases/current_baseline.json, "
            "not the evidence registry"
        )

        if path in pinned:
            assert entry.get("sha256_authority") == "releases/current_baseline.json", (
                f"{path}: serving-pinned, so it must defer via sha256_authority"
            )
            assert not entry.get("sha256"), (
                f"{path}: must not duplicate a digest the serving manifest owns"
            )
        else:
            assert entry.get("sha256"), f"{path}: missing sha256"
            assert len(entry["sha256"]) == 64
            assert not entry.get("sha256_authority"), (
                f"{path}: defers but is not pinned by the serving manifest"
            )


def test_every_feature_row_field_is_created_by_a_migration() -> None:
    """
    Alembic must be able to build a database that `_upsert_feature_rows` can
    write to (audit C-12).

    `pipeline/feature_engineer._FM_COLS` is generated from `FeatureRow`'s
    dataclass fields, and the INSERT names every one of them. So a FeatureRow
    field with no migration is not a cosmetic drift — on a fresh database built
    from `alembic upgrade head`, the first feature insert fails with
    UndefinedColumn.

    That is exactly how `prior_snap_share` shipped: session 02 added it to
    FeatureRow and to FEATURE_COLS, the live database acquired the column
    out-of-band, and no migration knew about it. The ORM/DDL check in
    `TestSchemaSync` did not catch it either, because it compares FeatureRow to
    the ORM rather than to the migrations.
    """
    import dataclasses
    import importlib.util
    import re

    from pipeline.feature_engineer import FeatureRow

    versions = REPO_ROOT / "alembic" / "versions"

    # The migrations declare columns four ways, all of which must be recognised:
    #   1. literal DDL bodies            "  is_home  SMALLINT,"
    #   2. explicit ALTER statements      "ADD COLUMN IF NOT EXISTS x FLOAT"
    #   3. (name, type) tuple lists       ('("prior_snap_share", "FLOAT")')
    #   4. bare name lists applied in a loop (0004's _PHASE4_COLS)
    _SQL_TYPE = (
        r"FLOAT|INTEGER|SMALLINT|BIGINT|NUMERIC|REAL|DOUBLE|TEXT|VARCHAR|"
        r"BOOLEAN|BOOL|JSONB|JSON|TIMESTAMPTZ|TIMESTAMP|DATE|SERIAL"
    )

    def _column_names(text: str) -> set[str]:
        found = set(
            re.findall(rf"^\s*([a-z][a-z0-9_]*)\s+(?:{_SQL_TYPE})\b", text, re.M)
        )
        found |= set(re.findall(r"ADD COLUMN IF NOT EXISTS ([a-z][a-z0-9_]*)", text))
        found |= set(
            re.findall(rf'\(\s*"([a-z][a-z0-9_]*)"\s*,\s*"(?:{_SQL_TYPE})"\s*\)', text)
        )
        # Bare-name lists: only trusted when the file actually applies them with
        # an ALTER ... ADD COLUMN loop, so an arbitrary string literal elsewhere
        # is not mistaken for a column.
        if re.search(r"ADD COLUMN IF NOT EXISTS \{col", text):
            found |= set(re.findall(r'^\s*"([a-z][a-z0-9_]*)",\s*$', text, re.M))
        return found

    declared: set[str] = set()
    for path in sorted(versions.glob("*.py")):
        declared |= _column_names(path.read_text())
        # 0001 stores its DDL as a gzip+base85 snapshot so migration history
        # cannot be rewritten by changing pipeline code. Use its own accessor
        # rather than trying to read the compressed blob.
        spec = importlib.util.spec_from_file_location(f"_mig_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - non-importable revision
            continue
        accessor = getattr(module, "_initial_schema_ddl", None)
        if accessor is not None:
            declared |= _column_names("\n".join(accessor()))

    fields = {f.name for f in dataclasses.fields(FeatureRow)}
    missing = sorted(fields - declared)
    assert not missing, (
        f"FeatureRow fields with no Alembic migration: {missing}\n"
        "`_upsert_feature_rows` will INSERT these column names, so a database "
        "built from `alembic upgrade head` would reject the first write. Add "
        "them to a migration, not to runtime DDL."
    )


def test_evidence_manifest_verifies_clean() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_evidence_manifest.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_evidence_manifest_detects_drift(tmp_path: Path) -> None:
    """A verifier that never fails is not a signal."""
    manifest_path = REPO_ROOT / "releases" / "artifacts" / "MANIFEST.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest["artifacts"][0]["status"] = "frozen"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        result = subprocess.run(
            [sys.executable, "scripts/verify_evidence_manifest.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "DRIFT" in result.stderr
    finally:
        manifest_path.write_text(original)


# ── C-31: report keys name the metric that was actually computed ────────────


def test_no_report_labels_a_count_metric_as_mae() -> None:
    """`reports/eval_causal_volume_summary.json` reported mean Poisson deviance
    under the key `pooled_mae` — wrong on both words."""
    from ml.stat_resolution import target_metric_family

    for path in sorted((REPO_ROOT / "reports").glob("*.json")):
        blob = path.read_text()
        assert "pooled_mae" not in blob, f"{path.name} still uses 'pooled_mae'"

    summary = json.loads(
        (REPO_ROOT / "reports" / "eval_causal_volume_summary.json").read_text()
    )
    for cell in summary["cells"]:
        family = target_metric_family(cell["target"])
        expected = "poisson_deviance" if family == "count" else "mae"
        assert cell["metric"] == expected, cell


def test_summary_metric_matches_the_source_csv() -> None:
    import pandas as pd

    for name in ("eval_causal_volume_summary.json", "eval_causal_yardage_summary.json"):
        summary = json.loads((REPO_ROOT / "reports" / name).read_text())
        assert summary["generated_by"], f"{name} must record its generator"
        for cell in summary["cells"]:
            df = pd.read_csv(REPO_ROOT / cell["source_report"])
            assert set(df["metric"].unique()) == {cell["metric"]}, cell


def test_summaries_are_reproducible_from_their_generator() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/summarize_causal_evals.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
