"""The causal-eval summary is reproducible from committed per-cell CSVs.

No headline count may be invented: with no inputs the script fails loudly
and prints the rerun recipe instead of writing a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.summarize_causal_evals import build, main


def _cell_csv(path: Path, stat: str, position: str, beats: list[bool], metric: str = "mae") -> Path:
    rows = []
    for i, beat in enumerate(beats):
        rows.append(
            {
                "stat": stat,
                "season": 2021 + i,
                "position": position,
                "metric": metric,
                "n": 100,
                "model_score": 5.0 if beat else 9.0,
                "naive_score": 8.0,
                "trailing3_score": 7.0,
                "beats_naive": beat,
                "beats_trailing3": beat,
                "max_train_season_ok": True,
            }
        )
    out = path / f"eval_causal_{stat}_{position}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


class TestMissingInputsFailLoudly:
    def test_build_raises_with_producer_help(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="ml.eval_causal"):
            build(tmp_path)

    def test_main_writes_nothing_and_returns_one(self, tmp_path: Path, capsys) -> None:
        rc = main(["--eval-dir", str(tmp_path), "--out", str(tmp_path / "out.json")])
        assert rc == 1
        assert not (tmp_path / "out.json").exists()
        assert "ml.eval_causal" in capsys.readouterr().err


class TestHeadlineCount:
    def test_beat_both_total(self, tmp_path: Path) -> None:
        _cell_csv(tmp_path, "receiving_yards", "WR", [True, True, True, False])
        _cell_csv(tmp_path, "receiving_yards", "TE", [True, True])
        summary = build(tmp_path)
        assert summary["headline"] == {"beat_both": 5, "cell_seasons": 6}
        by_pos = {c["position"]: c["beat_both"] for c in summary["cells"]}
        assert by_pos == {"WR": "3/4", "TE": "2/2"}
        assert summary["generated_by"] == "python scripts/summarize_causal_evals.py"
        assert len(summary["inputs"]) == 2

    def test_metric_carried_through_not_assumed(self, tmp_path: Path) -> None:
        _cell_csv(tmp_path, "targets", "WR", [True], metric="poisson_deviance")
        summary = build(tmp_path)
        assert summary["cells"][0]["metric"] == "poisson_deviance"

    def test_mixed_metrics_rejected(self, tmp_path: Path) -> None:
        path = _cell_csv(tmp_path, "targets", "WR", [True, True], metric="poisson_deviance")
        df = pd.read_csv(path)
        df.loc[1, "metric"] = "mae"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="mixed metrics"):
            build(tmp_path)

    def test_missing_column_rejected(self, tmp_path: Path) -> None:
        path = _cell_csv(tmp_path, "targets", "WR", [True])
        df = pd.read_csv(path).drop(columns=["beats_trailing3"])
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            build(tmp_path)


class TestCheckMode:
    def test_round_trip_then_stale(self, tmp_path: Path) -> None:
        _cell_csv(tmp_path, "carries", "RB", [True, False])
        out = tmp_path / "causal_baseline_beats.json"
        assert main(["--eval-dir", str(tmp_path), "--out", str(out)]) == 0
        assert main(["--eval-dir", str(tmp_path), "--out", str(out), "--check"]) == 0
        blob = json.loads(out.read_text())
        blob["headline"]["beat_both"] += 1
        out.write_text(json.dumps(blob))
        assert main(["--eval-dir", str(tmp_path), "--out", str(out), "--check"]) == 1
