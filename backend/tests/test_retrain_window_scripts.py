"""
retrain_causal_oof.sh pause/resume and train_window.sh completion bookkeeping.

The trainers are replaced by a fake interpreter that writes the files the real
ones would, so this checks only the shell logic: the STOP file pauses between
cells (exit 75) and resume skips finished cells; a subset run is never recorded
as a complete rebuild; `last_full_stamp` appears only with all 32 stacks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")

FAKE_PY = r"""#!/usr/bin/env bash
# Fake interpreter. Real python answers the two `-c` queries; trainers write files.
if [[ "$1" == "-c" ]]; then exec "$REAL_PY" "$@"; fi
if [[ "$1" == "-m" && "$2" == "ml.final_fit" ]]; then
  shift 2; echo "$*" >>"$FINAL_LOG"; exit 0
fi
if [[ "$1" == "-m" ]]; then
  MODULE="$2"; shift 2
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target) STAT="$2"; shift 2 ;;
      --position) POS="$2"; shift 2 ;;
      --out-dir) OUT="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  KIND="${MODULE#ml.}"; KIND="${KIND%_model}"
  [[ "$KIND" == "stacking_ensemble" ]] && KIND="stack"
  echo "$STAT:$POS" >>"$FAKE_LOG"
  printf 'x\n' >"$OUT/${KIND}_${STAT}_${POS}_t.csv"
  printf 'x' >"$OUT/attr_${KIND}_${STAT}_${POS}_t.parquet"
  # Optionally request a pause once the first cell's stack is written.
  if [[ "$KIND" == "stack" && -n "${FAKE_STOP_AFTER_FIRST:-}" ]]; then touch "$FAKE_STOP_AFTER_FIRST"; fi
  exit 0
fi
exit 0
"""


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    flock = bin_dir / "flock"  # the Mac has none; the box does
    flock.write_text("#!/bin/sh\nexit 0\n")
    flock.chmod(0o755)
    out_root = tmp_path / "train_out"
    out_root.mkdir()
    environ = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PY": str(fake),
        "REAL_PY": sys.executable,
        "FAKE_LOG": str(tmp_path / "calls.log"),
        "FINAL_LOG": str(tmp_path / "final_calls.log"),
        "TRAIN_OUT_ROOT": str(out_root),
        "SKIP_CONTRACT": "1",
        "PYTHONPATH": str(REPO),
        "SEASONS": "2019-2025",
    }
    environ.pop("ONLY_CELLS", None)
    return environ, out_root, tmp_path


def _run(script: str, args: list[str], environ: dict) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(REPO / "scripts" / script), *args], env=environ,
                          capture_output=True, text=True, check=False)


def _stacks(rebuild: Path) -> int:
    return len(list(rebuild.glob("stack_*.csv")))


def test_stop_file_pauses_between_cells_and_resume_finishes(env):
    environ, out_root, tmp_path = env
    stop = out_root / "STOP"
    environ["STAMP"] = "T1"
    environ["FAKE_STOP_AFTER_FIRST"] = str(stop)
    first = _run("retrain_causal_oof.sh", [], environ)
    assert first.returncode == 75, first.stderr
    rebuild = out_root / "oof" / "rebuild_T1"
    assert _stacks(rebuild) == 1
    assert "re-run with STAMP=T1" in first.stderr

    stop.unlink()
    environ.pop("FAKE_STOP_AFTER_FIRST")
    calls_before = len((tmp_path / "calls.log").read_text().splitlines())
    second = _run("retrain_causal_oof.sh", [], environ)
    assert second.returncode == 0, second.stderr
    assert _stacks(rebuild) == 32
    # Resume trained 31 cells x 3 steps; the finished cell was skipped.
    assert len((tmp_path / "calls.log").read_text().splitlines()) - calls_before == 31 * 3
    assert (rebuild / "SHA256SUMS").stat().st_size > 0


def test_train_window_full_ignores_a_cell_subset_and_marks_only_complete_runs(env):
    environ, out_root, _ = env
    environ["ONLY_CELLS"] = "targets:TE"  # must not shrink a full window
    stop = out_root / "STOP"
    environ["FAKE_STOP_AFTER_FIRST"] = str(stop)
    paused = _run("train_window.sh", ["full"], environ)
    assert paused.returncode == 75, paused.stderr
    assert not (out_root / "last_full_stamp").exists()
    assert (out_root / "full_stamp").exists(), "pointer must survive a pause so the next window resumes"

    stop.unlink()
    environ.pop("FAKE_STOP_AFTER_FIRST")
    done = _run("train_window.sh", ["full"], environ)
    assert done.returncode == 0, done.stderr
    stamp = (out_root / "last_full_stamp").read_text().strip()
    assert _stacks(out_root / "oof" / f"rebuild_{stamp}") == 32
    assert not (out_root / "full_stamp").exists()


def test_train_window_finishes_a_rebuild_that_was_started_as_a_subset(env):
    environ, out_root, _ = env
    stamp = "T9"
    (out_root / "full_stamp").write_text(stamp)
    rebuild = out_root / "oof" / f"rebuild_{stamp}"
    environ["STAMP"] = stamp
    environ["ONLY_CELLS"] = "targets:TE"
    assert _run("retrain_causal_oof.sh", [], environ).returncode == 0
    assert _stacks(rebuild) == 1

    # The window resumes the same stamp, trains the other 31 cells, then marks it.
    environ.pop("ONLY_CELLS")
    result = _run("train_window.sh", ["full"], environ)
    assert result.returncode == 0, result.stderr
    assert (out_root / "last_full_stamp").read_text().strip() == stamp
    assert _stacks(rebuild) == 32


def test_feature_contract_defaults_cover_every_training_season():
    from ml.season_constants import train_seasons

    assert train_seasons()[0] == 2019
    assert "train_seasons()" in (REPO / "scripts" / "verify_feature_contract.py").read_text()
    assert "train_seasons()" in (REPO / "scripts" / "retrain_causal_oof.sh").read_text()


# ── finalfit mode (decision 0012) ─────────────────────────────────────────────

def _final_calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "final_calls.log"
    return log.read_text().splitlines() if log.exists() else []


def test_finalfit_needs_a_release_id(env):
    environ, _, tmp_path = env
    done = _run("train_window.sh", ["finalfit"], environ)
    assert done.returncode == 2
    assert "RELEASE_ID" in done.stderr
    assert _final_calls(tmp_path) == []


def test_finalfit_needs_a_complete_rebuild_to_read_params_and_oof_from(env):
    environ, _, tmp_path = env
    environ["RELEASE_ID"] = "causal_20261114"
    done = _run("train_window.sh", ["finalfit"], environ)
    assert done.returncode == 1
    assert "last_full_stamp" in done.stderr
    assert _final_calls(tmp_path) == []


def test_finalfit_fits_every_model_then_assembles_models_json(env):
    environ, out_root, tmp_path = env
    (out_root / "last_full_stamp").write_text("STAMP1\n")
    environ["RELEASE_ID"] = "causal_20261114"
    environ["CODE_GIT_SHA"] = "abc"
    done = _run("train_window.sh", ["finalfit"], environ)
    assert done.returncode == 0, done.stderr
    fit, assemble = _final_calls(tmp_path)
    src = out_root / "oof" / "rebuild_STAMP1"
    out = out_root / "final" / "causal_20261114"
    assert fit.startswith("fit ") and assemble.startswith("assemble ")
    for arg in (f"--params-dir {src}", f"--oof-dir {src}", f"--out-dir {out}",
                "--release-dir releases/candidates/causal_20261114", "--release-id causal_20261114"):
        assert arg in fit, (arg, fit)
    assert "--cells" not in fit
    assert f"--out-dir {out}" in assemble and "--release-id causal_20261114" in assemble


def test_finalfit_with_a_cell_subset_fits_only_those_and_does_not_assemble(env):
    environ, out_root, tmp_path = env
    (out_root / "last_full_stamp").write_text("STAMP1\n")
    environ.update(RELEASE_ID="causal_20261114", ONLY_CELLS="passing_yards:QB targets:TE")
    done = _run("train_window.sh", ["finalfit"], environ)
    assert done.returncode == 0, done.stderr
    (fit,) = _final_calls(tmp_path)
    assert fit.startswith("fit ") and "--cells passing_yards:QB targets:TE" in fit
