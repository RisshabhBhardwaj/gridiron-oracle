"""scripts/run_causal_evals.py: per-cell eval files, stale-file refusal, honest failures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_causal_evals as rce  # noqa: E402
from scripts import summarize_causal_evals as summ  # noqa: E402


def _history() -> pd.DataFrame:
    rows = []
    for week, value in zip((1, 2, 3, 4), (80.0, 90.0, 100.0, 110.0)):
        rows.append({"player_id": "P1", "season": 2024, "week": week, "receiving_yards": value})
    for week, value in zip((1, 2), (50.0, 70.0)):
        rows.append({"player_id": "P1", "season": 2023, "week": week, "receiving_yards": value})
    return pd.DataFrame(rows)


def _stack(path: Path, *, max_train: int = 2023) -> Path:
    pd.DataFrame({
        "player_id": ["P1"] * 4,
        "season": [2024] * 4,
        "week": [1, 2, 3, 4],
        "position": ["WR"] * 4,
        "actual": [80.0, 90.0, 100.0, 110.0],
        "predicted": [80.0, 90.0, 100.0, 110.0],
        "max_train_season": [max_train] * 4,
    }).to_csv(path, index=False)
    return path


def test_writes_one_file_per_cell_and_summary_counts_only_those(tmp_path):
    stack = _stack(tmp_path / "stack.csv")
    out = tmp_path / "reports"
    written = rce.run(
        out_dir=out,
        history=_history(),
        resolve_stack=lambda s, p: stack,
        cells=[("receiving_yards", "WR")],
    )
    assert [p.name for p in written] == ["eval_causal_receiving_yards_WR.csv"]
    built = summ.build(out)
    assert built["headline"] == {"beat_both": 1, "cell_seasons": 1}
    assert built["inputs"] == ["eval_causal_receiving_yards_WR.csv"]


def test_stale_eval_files_are_refused(tmp_path):
    out = tmp_path / "reports"
    out.mkdir()
    (out / "eval_causal_old_WR.csv").write_text("x\n")
    with pytest.raises(rce.EvalRunError, match="already in"):
        rce.run(out_dir=out, history=_history(), resolve_stack=lambda s, p: tmp_path / "x", cells=[("receiving_yards", "WR")])


def test_non_causal_oof_is_refused(tmp_path):
    stack = _stack(tmp_path / "stack.csv", max_train=2024)
    with pytest.raises(ValueError, match="non-causal"):
        rce.evaluate_cell("receiving_yards", "WR", stack, _history())


def test_cell_with_no_evaluable_rows_fails(tmp_path):
    stack = _stack(tmp_path / "stack.csv")
    empty_history = _history().iloc[0:0]
    with pytest.raises(rce.EvalRunError):
        rce.evaluate_cell("receiving_yards", "WR", stack, empty_history)


def test_only_filter_limits_cells(tmp_path):
    stack = _stack(tmp_path / "stack.csv")
    written = rce.run(
        out_dir=tmp_path / "r",
        history=_history(),
        resolve_stack=lambda s, p: stack,
        cells=[("receiving_yards", "WR"), ("receiving_yards", "TE")],
        only={("receiving_yards", "WR")},
    )
    assert len(written) == 1
