#!/usr/bin/env python3
"""Rebuild ``params_<learner>_<stat>_<pos>.json`` from a rebuild's training logs.

The causal OOF rebuild of 2026-11-14 ran before the trainers saved their
parameters. Each trainer's log (``<learner>_<stat>_<pos>.log`` under
``$TRAIN_OUT_ROOT/logs/rebuild_<STAMP>/``, written by
``scripts/retrain_causal_oof.sh``) holds one line

    ... Optuna best RMSE=<float>  params={...}     (ml/lgbm_model.py, ml/catboost_model.py)

which is the Optuna study's best parameters. The trainers then add fixed
parameters; this script applies the same completion, so the file holds exactly
what the OOF was trained with.

It fails, and writes nothing, if a base model the release needs has no line or
two different lines, or if a params file already exists with other content.

    python3 scripts/recover_optuna_params.py \\
        --log-dir /train_out/logs/rebuild_20261114T103140 \\
        --release-dir releases/candidates/causal_20261114 \\
        --out-dir /train_out/oof/rebuild_20261114T103140
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.model_params import (  # noqa: E402
    LEARNERS,
    ParamsError,
    params_filename,
    read_params,
    required_models_from_ridge_dir,
    write_params,
)

_LINE = re.compile(r"Optuna best RMSE=(?P<rmse>\S+)\s+params=(?P<params>.*?)\s*$")
_LOG_NAME = re.compile(r"^(?P<learner>" + "|".join(LEARNERS) + r")_(?P<stat>.+)_(?P<pos>[A-Z]+)\.log$")


class RecoveryError(RuntimeError):
    """The logs cannot give a complete, unambiguous set of parameters."""


def parse_log_line(line: str) -> tuple[float, dict] | None:
    """Return ``(best_rmse, best_params)`` for an Optuna line, else None."""
    match = _LINE.search(line)
    if not match:
        return None
    try:
        params = ast.literal_eval(match["params"])
        rmse = float(match["rmse"])
    except (ValueError, SyntaxError) as exc:
        raise RecoveryError(f"unparseable Optuna line: {line.strip()[:200]}") from exc
    if not isinstance(params, dict) or not params:
        raise RecoveryError(f"params field is not a plain dict: {line.strip()[:200]}")
    return rmse, params


def _complete(learner: str, best: dict) -> dict:
    """Apply the trainer's own fixed parameters (imported, so they cannot drift)."""
    if learner == "lgbm":
        from ml.lgbm_model import complete_optuna_params
    else:
        from ml.catboost_model import complete_optuna_params
    return complete_optuna_params(best)


def _scan(log_dir: Path) -> dict[tuple[str, str, str], dict]:
    found: dict[tuple[str, str, str], dict] = {}
    for path in sorted(log_dir.glob("*.log")):
        name = _LOG_NAME.match(path.name)
        if not name:
            continue
        key = (name["learner"], name["stat"], name["pos"])
        for line in path.read_text(errors="replace").splitlines():
            parsed = parse_log_line(line)
            if parsed is None:
                continue
            _, params = parsed
            if key in found and found[key] != params:
                raise RecoveryError(f"{path.name}: two different Optuna lines for {'_'.join(key)}")
            found[key] = params
    return found


def recover(log_dir: Path, out_dir: Path, release_dir: Path) -> list[Path]:
    """Write one params file per cell found in the logs. Returns the paths."""
    log_dir, out_dir = Path(log_dir), Path(out_dir)
    found = _scan(log_dir)
    needed = required_models_from_ridge_dir(release_dir)
    if not needed:
        raise RecoveryError(f"no ridge_*_coefs.json files under {release_dir}")
    missing = sorted(needed - set(found))
    if missing:
        raise RecoveryError(
            "no Optuna line for needed model(s): " + ", ".join("_".join(k) for k in missing)
        )
    completed = {key: _complete(key[0], params) for key, params in found.items()}

    # Check every target first so a failure leaves the directory untouched.
    for (learner, stat, pos), params in completed.items():
        target = out_dir / params_filename(learner, stat, pos)
        if target.exists():
            try:
                same = read_params(target)["params"] == params
            except ParamsError:
                same = False
            if not same:
                raise RecoveryError(f"{target} already exists with different content; not overwriting")

    written = []
    for (learner, stat, pos), params in sorted(completed.items()):
        written.append(write_params(out_dir, learner, stat, pos, params, source="recovered_from_log"))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path,
                        help="directory holding the ridge_*_coefs.json files that say which models are needed")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        written = recover(args.log_dir, args.out_dir, args.release_dir)
    except (RecoveryError, ParamsError) as exc:
        print(f"recover_optuna_params: FAILED: {exc}", file=sys.stderr)
        return 1
    needed = required_models_from_ridge_dir(args.release_dir)
    print(f"recovered {len(written)} params files into {args.out_dir} ({len(needed)} needed by the release)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
