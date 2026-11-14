"""
The /backtest evidence is read from the served release's pinned causal evals.

Fixture manifests stand in for ``releases/current_baseline.json`` (which gets its
``causal_evals`` pin only when the manager re-freezes), so these tests do not
depend on the live manifest. One test builds the post-freeze manifest from the
real committed evals to check the headline numbers of decision 0011.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.core.runtime_mode import ArtifactRequiredError
from backend.app.services import backtest as bt
from ml.artifact_manifest import REPO_ROOT, clear_cache, load_manifest

COLUMNS = [
    "stat", "metric", "n", "n_model", "n_naive", "n_trailing3", "cohort_rule",
    "min_prior_games", "model_score", "naive_score", "trailing3_score",
    "beats_naive", "beats_trailing3", "max_train_season_ok", "marcel_score",
    "beats_marcel", "season", "position",
]
RULE = "player has at least this many completed games earlier in the same season"
EVALS = "releases/candidates/rel/evals"

# (stat, position, metric, season, model, naive, trailing3, marcel)
FIXTURE_ROWS = [
    ("receiving_yards", "WR", "mae", 2024, 20.0, 25.0, 24.0, 23.0),
    ("receiving_yards", "WR", "mae", 2025, 23.24, 25.17, 25.27, 25.72),
    ("receptions", "TE", "poisson_deviance", 2024, 1.00, 0.90, 1.20, 1.10),  # loses to naive
    ("receptions", "TE", "poisson_deviance", 2025, 1.00, 1.30, 1.20, 0.95),  # loses to marcel
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release(root: Path, *, with_key: bool = True, tamper: bool = False,
                   drop_file: bool = False, lie_in_beats: bool = False) -> Path:
    """Write a tiny release (2 cells, 4 cell-seasons) and return its manifest path."""
    evals = root / EVALS
    evals.mkdir(parents=True)
    stack = root / "releases/candidates/rel/stack_receiving_yards_WR_20260101.csv"
    stack.write_text("x\n")

    cells: dict[tuple[str, str], list] = {}
    for stat, pos, metric, season, model, naive, tr3, marcel in FIXTURE_ROWS:
        cells.setdefault((stat, pos, metric), []).append((season, model, naive, tr3, marcel))
    beats_cells, files = [], []
    for (stat, pos, metric), rows in cells.items():
        path = evals / f"eval_causal_{stat}_{pos}.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            n_both = 0
            for season, model, naive, tr3, marcel in rows:
                bn, bt3, bm = model < naive, model < tr3, model < marcel
                n_both += bn and bt3
                w.writerow([stat, metric, 1000, 1200, 1000, 1200, RULE, 1, model, naive, tr3,
                            bn, bt3, True, marcel, bm, season, pos])
        files.append(path)
        beats_cells.append({"stat": stat, "position": pos, "metric": metric,
                            "beat_both_count": n_both, "cell_seasons": len(rows)})
    beats = evals / "causal_baseline_beats.json"
    headline = {"beat_both": 3 if not lie_in_beats else 4, "cell_seasons": 4}
    beats.write_text(json.dumps({"schema_version": 1, "headline": headline, "cells": beats_cells,
                                 "metric_note": "count targets use Poisson deviance"}))
    files.append(beats)

    rel = [p.relative_to(root).as_posix() for p in files]
    digests = {stack.relative_to(root).as_posix(): _sha(stack)}
    digests.update({r: _sha(root / r) for r in rel})
    artifacts = {"stacks": [stack.relative_to(root).as_posix()]}
    if with_key:
        artifacts["causal_evals"] = rel
    manifest = root / "releases/current_baseline.json"
    manifest.write_text(json.dumps({"artifacts": artifacts, "artifact_digests": digests,
                                    "model_version": "latest"}))
    if tamper:
        (root / rel[0]).write_text((root / rel[0]).read_text() + "\n")
    if drop_file:
        (root / rel[0]).unlink()
    return manifest


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_cache()
    bt.clear_cache()
    yield
    clear_cache()
    bt.clear_cache()


def _service(tmp_path: Path, **kw) -> bt.BacktestService:
    manifest = load_manifest(_write_release(tmp_path, **kw), root=tmp_path)
    return bt.BacktestService(manifest=manifest)


# ── reader ────────────────────────────────────────────────────────────────────

def test_rows_carry_scores_metric_n_and_cohort_rule(tmp_path):
    s = _service(tmp_path).get_summary(stat="receiving_yards", positions=["WR"])
    row = next(r for r in s.rows if r.season == 2025)
    assert (row.model_score, row.naive_score, row.trailing3_score, row.marcel_score) == (
        23.24, 25.17, 25.27, 25.72)
    assert row.metric == "mae" and row.n == 1000
    assert row.cohort_rule == RULE
    assert row.beats_naive and row.beats_trailing3 and row.beats_marcel


def test_improvement_is_percent_below_the_simple_method(tmp_path):
    s = _service(tmp_path).get_summary(stat="receiving_yards", positions=["WR"])
    row = next(r for r in s.rows if r.season == 2024)
    assert row.improvement_vs_naive_pct == pytest.approx(20.0)  # (25-20)/25


def test_count_targets_are_labelled_deviance_never_average_miss(tmp_path):
    svc = _service(tmp_path)
    count = svc.get_summary(stat="receptions", positions=["TE"]).rows[0]
    yards = svc.get_summary(stat="receiving_yards", positions=["WR"]).rows[0]
    assert count.metric == "poisson_deviance"
    assert "deviance" in count.metric_label.lower()
    assert "average miss" not in count.metric_label.lower()
    assert "yards" in yards.metric_label.lower() and "average miss" in yards.metric_label.lower()


def test_release_verdict_counts_cell_seasons(tmp_path):
    v = _service(tmp_path).get_summary().release_verdict
    assert (v.cell_seasons, v.beat_naive, v.beat_trailing3, v.beat_marcel, v.beat_both) == (4, 3, 4, 3, 3)


def test_selection_verdict_follows_the_filter(tmp_path):
    s = _service(tmp_path).get_summary(stat="receptions", positions=["TE"])
    assert s.verdict.cell_seasons == 2 and s.verdict.beat_naive == 1
    assert s.release_verdict.cell_seasons == 4


def test_position_filter_is_applied_and_available_cells_are_unfiltered(tmp_path):
    s = _service(tmp_path).get_summary(stat="receiving_yards", positions=["TE"])
    assert s.rows == []
    assert {(c.stat, c.position) for c in s.available} == {("receiving_yards", "WR"), ("receptions", "TE")}


def test_unknown_stat_is_reported(tmp_path):
    with pytest.raises(bt.UnknownStat):
        _service(tmp_path).get_summary(stat="rushing_yards", positions=["QB"])


def test_unmeasured_things_are_named_not_invented(tmp_path):
    s = _service(tmp_path).get_summary()
    assert {"coverage", "crps", "brier", "pnl", "calibration"} <= set(s.not_measured)
    assert not hasattr(s, "calibration") and not hasattr(s, "overall_crps")


# ── fail closed ───────────────────────────────────────────────────────────────

def test_missing_manifest_key_fails_closed(tmp_path):
    with pytest.raises(bt.EvalsUnavailable, match="causal_evals"):
        _service(tmp_path, with_key=False).get_summary()


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(bt.EvalsUnavailable):
        _service(tmp_path, drop_file=True).get_summary()


def test_digest_mismatch_fails_closed(tmp_path):
    with pytest.raises(bt.EvalsUnavailable, match="SHA-256"):
        _service(tmp_path, tamper=True).get_summary()


def test_headline_that_disagrees_with_the_rows_fails_closed(tmp_path):
    with pytest.raises(bt.EvalsUnavailable, match="headline"):
        _service(tmp_path, lie_in_beats=True).get_summary()


def test_unavailable_is_an_artifact_required_error(tmp_path):
    assert issubclass(bt.EvalsUnavailable, ArtifactRequiredError)


# ── the real committed evals, post-freeze ─────────────────────────────────────

def _post_freeze_manifest(tmp_path: Path) -> Path:
    """Live manifest + the causal_evals pin, digests recomputed as freeze does."""
    from scripts import freeze_baseline as fb

    live = json.loads((REPO_ROOT / "releases/current_baseline.json").read_text())
    evals_dir = REPO_ROOT / "releases/candidates/causal_20261114/evals"
    live["artifacts"]["causal_evals"] = sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in evals_dir.iterdir()
    )
    src = tmp_path / "patched_current.json"
    src.write_text(json.dumps(live))
    payload: dict = {}
    fb._carry_forward_artifact_pins(payload, src)
    live["artifact_digests"] = payload["artifact_digests"]
    out = tmp_path / "post_freeze.json"
    out.write_text(json.dumps(live))
    return out


def test_real_evals_reproduce_decision_0011_counts(tmp_path):
    manifest = load_manifest(_post_freeze_manifest(tmp_path), root=REPO_ROOT)
    s = bt.BacktestService(manifest=manifest).get_summary()
    v = s.release_verdict
    assert (v.cell_seasons, v.beat_both, v.beat_trailing3, v.beat_marcel) == (160, 158, 160, 157)
    assert v.beat_naive == 158 and v.causal_check_passed == 160


def test_real_wr_receiving_yards_2025(tmp_path):
    manifest = load_manifest(_post_freeze_manifest(tmp_path), root=REPO_ROOT)
    s = bt.BacktestService(manifest=manifest).get_summary(stat="receiving_yards", positions=["WR"])
    r = next(r for r in s.rows if r.season == 2025)
    assert round(r.model_score, 2) == 23.24 and round(r.naive_score, 2) == 25.17
    assert r.improvement_vs_naive_pct > 0


def test_live_manifest_before_freeze_fails_closed_with_a_clear_error():
    """Until the re-freeze adds causal_evals, the live api must refuse, not guess."""
    live = json.loads((REPO_ROOT / "releases/current_baseline.json").read_text())
    if "causal_evals" in live["artifacts"]:
        pytest.skip("release already frozen with causal_evals")
    with pytest.raises(bt.EvalsUnavailable, match="causal_evals"):
        bt.BacktestService(manifest=load_manifest(root=REPO_ROOT)).get_summary()


# ── API ───────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from backend.app.main import app

    manifest = load_manifest(_write_release(tmp_path), root=tmp_path)
    monkeypatch.setattr(bt, "_default_manifest", lambda: manifest)
    return TestClient(app)


def test_api_returns_verdicts_rows_and_no_legacy_fields(client):
    body = client.get("/backtest?stat=receiving_yards&positions=WR").json()
    assert body["release_verdict"]["beat_both"] == 3
    assert body["verdict"]["cell_seasons"] == 2
    assert {r["season"] for r in body["rows"]} == {2024, 2025}
    for legacy in ("brier_score", "simulated_pnl", "sharpe_ratio", "max_drawdown",
                   "calibration", "overall_crps", "overall_rmse", "by_season", "data_source"):
        assert legacy not in body
    assert "coverage" in body["not_measured"]
    assert body["table_url"].endswith("/backtest/table.csv")


def test_api_unknown_stat_is_404(client):
    assert client.get("/backtest?stat=nope").status_code == 404


def test_api_unavailable_is_503_with_a_user_safe_message(tmp_path, monkeypatch):
    from backend.app.main import app

    manifest = load_manifest(_write_release(tmp_path, tamper=True), root=tmp_path)
    monkeypatch.setattr(bt, "_default_manifest", lambda: manifest)
    r = TestClient(app).get("/backtest")
    assert r.status_code == 503
    detail = json.dumps(r.json())
    assert "sha" not in detail.lower() and str(tmp_path) not in detail and "releases/" not in detail


def test_api_table_csv_lists_every_row_with_plain_headers(client):
    r = client.get("/backtest/table.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert len(lines) == 1 + 4
    assert "model_score" in lines[0] and "naive_score" in lines[0] and "metric" in lines[0]


# ── no public path reads the stale CSV ────────────────────────────────────────

def test_no_public_code_reads_ml_backtest_results():
    offenders = []
    for path in (REPO_ROOT / "backend/app").rglob("*.py"):
        text = path.read_text()
        if "backtest_results" in text or "backtest_2019" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []


# ── integrity check ───────────────────────────────────────────────────────────

def _status(manifest):
    from backend.app.services import runtime_status as rs

    return rs.RuntimeStatusService("postgresql://x", "http://m", "latest")._check_backtest_assets(manifest)


def test_integrity_check_ok_with_pinned_evals(tmp_path):
    manifest = load_manifest(_write_release(tmp_path), root=tmp_path)
    r = _status(manifest)
    assert r["status"] == "ok"
    assert r["observed"]["cell_seasons"] == 4


@pytest.mark.parametrize("kw", [{"with_key": False}, {"tamper": True}, {"drop_file": True}])
def test_integrity_check_errors_when_pin_is_missing_or_wrong(tmp_path, kw):
    manifest = load_manifest(_write_release(tmp_path, **kw), root=tmp_path)
    r = _status(manifest)
    assert r["status"] == "error"
    assert str(tmp_path) not in r["detail"]


# ── evidence verifier, post-freeze ────────────────────────────────────────────

def _run_verifier(env_manifest: Path | None):
    import os
    import subprocess
    import sys

    env = {**os.environ}
    env.pop("EVIDENCE_SERVING_MANIFEST", None)
    if env_manifest:
        env["EVIDENCE_SERVING_MANIFEST"] = str(env_manifest)
    return subprocess.run(
        [sys.executable, "scripts/verify_evidence_manifest.py"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )


def test_verifier_passes_on_the_committed_manifest_before_freeze():
    r = _run_verifier(None)
    assert r.returncode == 0, r.stdout + r.stderr


def test_verifier_passes_in_the_simulated_post_freeze_state(tmp_path):
    r = _run_verifier(_post_freeze_manifest(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Causal evals OK: 33 pinned files" in r.stdout


def test_verifier_fails_if_a_pinned_eval_drifts_after_freeze(tmp_path):
    manifest = _post_freeze_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    victim = next(k for k in data["artifact_digests"] if k.endswith("eval_causal_carries_RB.csv"))
    data["artifact_digests"][victim] = "0" * 64
    manifest.write_text(json.dumps(data))
    assert _run_verifier(manifest).returncode == 1
