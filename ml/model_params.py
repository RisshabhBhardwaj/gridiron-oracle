"""Persisted base-learner hyperparameters (decision 0012).

Each trainer writes ``params_<learner>_<stat>_<pos>.json`` next to its OOF, so a
later final fit can reuse exactly the parameters that produced the OOF without
re-tuning (Optuna is unseeded). ``scripts/recover_optuna_params.py`` writes the
same files from the log lines of an earlier rebuild.

No heavy imports: the recovery script runs on the box with the stdlib plus this
module.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

LEARNERS: tuple[str, ...] = ("lgbm", "catboost")


class ParamsError(ValueError):
    """A params file is missing, malformed, or does not match its name."""


def params_filename(learner: str, stat: str, position: Optional[str]) -> str:
    return f"params_{learner}_{stat}_{position or 'all'}.json"


def write_params(
    out_dir: Path,
    learner: str,
    stat: str,
    position: Optional[str],
    params: dict,
    *,
    source: str,
) -> Path:
    """Write the params file atomically and return its path."""
    if learner not in LEARNERS:
        raise ParamsError(f"unknown learner {learner!r}; expected one of {LEARNERS}")
    if not params:
        raise ParamsError("refusing to write an empty params dict")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "learner": learner,
        "stat": stat,
        "position": position or "all",
        "source": source,
        "params": params,
    }
    path = out_dir / params_filename(learner, stat, position)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".params_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_params(path: Path) -> dict:
    """Read and validate a params file. The content must match the file name."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ParamsError(f"cannot read params file {path}: {exc}") from exc
    for key in ("learner", "stat", "position", "params", "source"):
        if key not in payload:
            raise ParamsError(f"{path.name}: missing key {key!r}")
    expected = params_filename(payload["learner"], payload["stat"], payload["position"])
    if expected != path.name:
        raise ParamsError(f"{path.name}: content says {expected}, which does not match the file name")
    if not payload["params"]:
        raise ParamsError(f"{path.name}: no params")
    return payload


def learners_in_use(coefs: dict) -> list[str]:
    """Learners of a ridge coefficient payload that carry weight (non-zero)."""
    weights = coefs["weights"]
    return [name for name in coefs["learner_order"] if float(weights[name]) != 0.0]


def required_models_from_ridge_dir(release_dir: Path) -> set[tuple[str, str, str]]:
    """``(learner, stat, position)`` for every base model the ridge files use.

    A learner whose weight is exactly 0.0 is neither fitted nor loaded.
    """
    out: set[tuple[str, str, str]] = set()
    for path in sorted(Path(release_dir).glob("ridge_*_coefs.json")):
        stem = path.name[len("ridge_"):-len("_coefs.json")]
        stat, _, position = stem.rpartition("_")
        for learner in learners_in_use(json.loads(path.read_text())):
            out.add((learner, stat, position))
    return out


def required_models(release_dirs: Iterable[Path]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for d in release_dirs:
        out |= required_models_from_ridge_dir(d)
    return out
