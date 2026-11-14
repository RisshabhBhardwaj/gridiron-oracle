"""Final fit of the pinned base models (decision 0012, section 2).

The walk-forward trainers return the model of their last fold, fitted through
``LAST_COMPLETE_SEASON - 1``. A forward week needs a model fitted through
``LAST_COMPLETE_SEASON``. This module fits it, one model per (learner, stat,
position) that a ridge file gives non-zero weight, with exactly the
hyperparameters that produced the release's OOF. It never runs Optuna.

Replay gate. Before a model is kept, the last OOF fold is refitted with the
same code, params and data window, and its predictions are compared with the
base-learner OOF kept from the rebuild. A wrong parameter set, feature list,
data window or library version shows up here as a difference. A model whose
replay fails is not written.

Tree count. A fold model early-stops on its validation season (50 rounds). The
final fit has no season to validate on, so it is given the tree count the
replay early-stopped at (``trees_rule: replay_best_iteration``).

Reproducible means: same code sha, params, data and library versions give the
same predictions within the tolerance. It does not mean identical file bytes.

    python -m ml.final_fit fit --release-dir releases/candidates/causal_20261114 \\
        --params-dir /train_out/oof/rebuild_STAMP --oof-dir /train_out/oof/rebuild_STAMP \\
        --out-dir /train_out/final/causal_20261114 --release-id causal_20261114
    python -m ml.final_fit assemble --release-dir ... --out-dir ... --release-id ...

Driven on the box by ``scripts/train_window.sh finalfit``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import resource
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.artifact_manifest import (
    BASE_MODEL_EXTENSIONS,
    MODELS_META_NAME,
    MODELS_META_SCHEMA_VERSION,
)
from ml.model_params import params_filename, read_params, required_models_from_ridge_dir
from ml.reliability import build_training_manifest
from ml.season_constants import assert_seasons_within_cap, train_seasons

logger = logging.getLogger(__name__)

#: Replay tolerance: max absolute prediction difference, as a fraction of the
#: target's standard deviation. Threads can cause tiny float differences.
DEFAULT_TOLERANCE = 1e-4
TREES_RULE = "replay_best_iteration"
ENTRIES_DIRNAME = ".entries"
_TREE_PARAM = {"lgbm": "n_estimators", "catboost": "iterations"}
_ROOT = Path(__file__).resolve().parents[1]


class FinalFitError(RuntimeError):
    """The final fit cannot proceed or its inputs do not add up."""


class ReplayGateError(FinalFitError):
    """The replay of the last OOF fold does not reproduce the OOF."""


@dataclass(frozen=True)
class ReplayResult:
    passed: bool
    reason: str
    val_season: int
    n_rows: int
    max_abs_diff: float
    mean_abs_diff: float
    tolerance_rel: float
    tolerance_abs: float
    target_std: float
    best_iteration: int

    def to_dict(self) -> dict:
        return asdict(self)


# ── data ──────────────────────────────────────────────────────────────────────

def _module(learner: str):
    if learner not in _TREE_PARAM:
        raise FinalFitError(f"unknown learner {learner!r}; expected one of {sorted(_TREE_PARAM)}")
    return importlib.import_module(f"ml.{learner}_model")


def fit_window(df: pd.DataFrame) -> pd.DataFrame:
    """The final-fit data window: every season in *df*, which must not pass the cap."""
    assert_seasons_within_cap(sorted(int(s) for s in df["season"].dropna().unique()))
    return df


def _prepare(learner: str, df: pd.DataFrame, stat: str, position: str):
    """The trainers' own data prep: position filter, played-only rows, cohort check, features."""
    mod = _module(learner)
    target_col = mod.TARGET_COL_MAP[stat]
    data = df[df["position"] == position].copy()
    data = data[data[target_col].notna()].copy()
    mod.validate_training_cohort(
        data, target=stat, target_col=target_col, position_filter=position,
        source=f"final_fit/{learner}",
    )
    fit_window(data)
    features = [f for f in mod.FEATURE_COLS if f in data.columns]
    return mod, data, features, target_col


def _tree_count(learner: str, model) -> int:
    if learner == "lgbm":
        return int(model.best_iteration_ or model.booster_.current_iteration())
    return int(model.tree_count_)


def _runtime_params(learner: str, params: dict) -> dict:
    """Params plus settings that cannot change a prediction (no CatBoost log files)."""
    return {**params, "allow_writing_files": False} if learner == "catboost" else dict(params)


# ── replay gate ───────────────────────────────────────────────────────────────

def replay_gate(
    learner: str,
    stat: str,
    position: str,
    df: pd.DataFrame,
    params: dict,
    oof: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ReplayResult:
    """Refit the last OOF fold and compare its predictions with the kept OOF."""
    mod, data, features, target_col = _prepare(learner, df, stat, position)
    folds = mod._make_walk_forward_folds(sorted(int(s) for s in data["season"].unique()))
    fold_idx = len(folds) - 1
    train_seasons_, val_season = folds[-1]
    train_df = data[data["season"].isin(train_seasons_)]
    val_df = data[data["season"] == val_season]
    model, oof_rows, _ = mod._train_fold(
        fold_idx, train_df, val_df, features, target_col, _runtime_params(learner, params)
    )
    best_iteration = _tree_count(learner, model)
    target_std = float(np.std(val_df[target_col].to_numpy(dtype=float)))
    tolerance_abs = tolerance * (target_std if target_std > 0 else 1.0)

    def result(passed: bool, reason: str, max_d=float("nan"), mean_d=float("nan")) -> ReplayResult:
        return ReplayResult(passed, reason, int(val_season), int(len(oof_rows)), max_d, mean_d,
                            tolerance, tolerance_abs, target_std, best_iteration)

    needed = {"player_id", "game_id", "season", "y_pred"}
    if not needed <= set(oof.columns):
        return result(False, f"OOF lacks column(s) {sorted(needed - set(oof.columns))}")
    kept = oof[oof["season"] == val_season][["player_id", "game_id", "y_pred"]]
    merged = oof_rows[["player_id", "game_id", "y_pred"]].merge(
        kept, on=["player_id", "game_id"], how="inner", suffixes=("_replay", "_oof")
    )
    if not (len(merged) == len(oof_rows) == len(kept)):
        return result(
            False,
            f"rows differ: replay {len(oof_rows)}, OOF {len(kept)}, matched on player/game {len(merged)}",
        )
    diff = (merged["y_pred_replay"] - merged["y_pred_oof"]).abs().to_numpy(dtype=float)
    max_d, mean_d = float(diff.max()), float(diff.mean())
    if not np.isfinite(diff).all():
        return result(False, "non-finite prediction difference", max_d, mean_d)
    if max_d > tolerance_abs:
        return result(
            False,
            f"max |replay - OOF| {max_d:.3g} exceeds {tolerance_abs:.3g} ({tolerance:g} x target std {target_std:.3g})",
            max_d, mean_d,
        )
    return result(True, "replay reproduces the OOF", max_d, mean_d)


# ── fit ───────────────────────────────────────────────────────────────────────

def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _library_versions(learner: str) -> dict[str, str]:
    lib = "lightgbm" if learner == "lgbm" else "catboost"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        lib: importlib.import_module(lib).__version__,
    }


def model_filename(learner: str, stat: str, position: str) -> str:
    return f"{learner}_{stat}_{position}{BASE_MODEL_EXTENSIONS[learner]}"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _fit_and_save(learner: str, mod, data: pd.DataFrame, features: list[str], target_col: str,
                  params: dict, path: Path) -> None:
    sentinel = mod.MISSING_VALUE_SENTINEL
    tmp = path.with_name(f".{path.name}.tmp{path.suffix}")
    if learner == "lgbm":
        import lightgbm as lgb

        X = data[features].fillna(sentinel).infer_objects(copy=False)
        model = lgb.LGBMRegressor(**params)
        model.fit(X, data[target_col].values)
        model.booster_.save_model(str(tmp))
    else:
        import catboost as cb

        X = data[features].fillna(sentinel).values
        model = cb.CatBoostRegressor(**params)
        model.fit(X, data[target_col].values, verbose=False)
        model.save_model(str(tmp), format="cbm")
    os.replace(tmp, path)


def fit_cell(
    learner: str,
    stat: str,
    position: str,
    *,
    df: pd.DataFrame,
    params: dict,
    oof: pd.DataFrame,
    out_dir: Path,
    release_id: str,
    code_sha: Optional[str],
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Replay-gate, then fit one model on every season through the cap and write it.

    Returns the model's MODELS.json entry (also saved under ``.entries/``).
    Nothing is written if the gate fails.
    """
    sha = code_sha or os.environ.get("CODE_GIT_SHA") or _git_sha()
    if not sha:
        raise FinalFitError("code sha unknown: pass --code-sha or set CODE_GIT_SHA (the box is not a git checkout)")
    replay = replay_gate(learner, stat, position, df, params, oof, tolerance=tolerance)
    label = f"{learner}/{stat}/{position}"
    if not replay.passed:
        raise ReplayGateError(f"{label}: replay gate failed: {replay.reason}")
    mod, data, features, target_col = _prepare(learner, df, stat, position)

    final_params = {**params, _TREE_PARAM[learner]: replay.best_iteration}
    out_dir = Path(out_dir)
    (out_dir / ENTRIES_DIRNAME).mkdir(parents=True, exist_ok=True)
    name = model_filename(learner, stat, position)
    path = out_dir / name
    _fit_and_save(learner, mod, data, features, target_col, _runtime_params(learner, final_params), path)

    manifest = build_training_manifest(
        data, target=stat, position=position, target_col=target_col, feature_columns=features,
        source_contract={"learner": learner, "final_fit": True, "trees_rule": TREES_RULE},
    )
    seasons = sorted(int(s) for s in data["season"].unique())
    entry = {
        "learner": learner, "stat": stat, "position": position, "file": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "release_id": release_id,
        "train_seasons": seasons, "max_train_season": max(seasons),
        "features": features, "features_sha256": manifest.feature_schema_hash,
        "params": final_params, "params_oof": params,
        "trees": replay.best_iteration, "trees_rule": TREES_RULE,
        "n_rows": manifest.row_count, "training_data_sha256": manifest.training_data_hash,
        "target_summary": manifest.target_summary, "null_rates": manifest.null_rates,
        "code_git_sha": sha, "library_versions": _library_versions(learner),
        "replay_gate": replay.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(out_dir / ENTRIES_DIRNAME / f"{name}.json", (json.dumps(entry, indent=2, sort_keys=True) + "\n").encode())
    return entry


def load_cell_inputs(params_dir: Path, oof_dir: Path, learner: str, stat: str, position: str):
    """The params file and the OOF CSV of one model. A missing file is an error, never a default."""
    params_path = Path(params_dir) / params_filename(learner, stat, position)
    payload = read_params(params_path)
    if (payload["learner"], payload["stat"], payload["position"]) != (learner, stat, position):
        raise FinalFitError(f"{params_path} describes {payload['learner']}/{payload['stat']}/{payload['position']}")
    matches = sorted(Path(oof_dir).glob(f"{learner}_{stat}_{position}_*.csv"))
    if len(matches) != 1:
        raise FinalFitError(f"expected exactly one OOF CSV for {learner}/{stat}/{position} in {oof_dir}, found {len(matches)}")
    return payload["params"], pd.read_csv(matches[0]), matches[0]


# ── MODELS.json ───────────────────────────────────────────────────────────────

def assemble_models_json(out_dir: Path, release_dir: Path, release_id: str) -> Path:
    """Write MODELS.json from the per-model entries, only if they match the ridge files exactly."""
    out_dir = Path(out_dir)
    entries = [json.loads(p.read_text()) for p in sorted((out_dir / ENTRIES_DIRNAME).glob("*.json"))]
    if not entries:
        raise FinalFitError(f"no fitted models under {out_dir / ENTRIES_DIRNAME}")
    have = {(e["learner"], e["stat"], e["position"]) for e in entries}
    if len(have) != len(entries):
        raise FinalFitError("two entries for one model")
    need = required_models_from_ridge_dir(release_dir)
    missing = sorted("/".join(k) for k in need - have)
    if missing:
        raise FinalFitError(f"not fitted: {missing} (learner/stat/position); {len(need)} needed by {release_dir}")
    extra = sorted("/".join(k) for k in have - need)
    if extra:
        raise FinalFitError(f"fitted model(s) that no ridge file uses (weight 0.0 or no ridge file): {extra}")
    for e in entries:
        if e["release_id"] != release_id:
            raise FinalFitError(f"{e['file']} has release_id {e['release_id']!r}, expected {release_id!r}")
        path = out_dir / e["file"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != e["sha256"]:
            raise FinalFitError(f"{e['file']} is missing or changed since it was fitted")
    entries.sort(key=lambda e: (e["stat"], e["position"], e["learner"]))
    payload = {"schema_version": MODELS_META_SCHEMA_VERSION, "release_id": release_id, "models": entries}
    path = out_dir / MODELS_META_NAME
    _atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _load_frame() -> pd.DataFrame:
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise FinalFitError("DATABASE_URL not set")
    seasons = train_seasons()
    assert_seasons_within_cap(seasons)
    try:
        from ml.parquet_cache import FeatureCache

        df = FeatureCache().load_or_fetch(seasons, db_url, position_filter=None, no_cache=True)
    except ImportError:
        from ml.utils import load_feature_matrix

        df = load_feature_matrix(db_url, seasons, position_filter=None)
    if df.empty:
        raise FinalFitError("no feature_matrix rows loaded")
    return df


def _cmd_fit(args: argparse.Namespace) -> int:
    need = sorted(required_models_from_ridge_dir(args.release_dir), key=lambda k: (k[1], k[2], k[0]))
    if args.cells:
        wanted = {tuple(c.split(":", 1)) for c in args.cells}
        unknown = wanted - {(k[1], k[2]) for k in need}
        if unknown:
            raise FinalFitError(f"--cells not needed by the release: {sorted(unknown)}")
        need = [k for k in need if (k[1], k[2]) in wanted]
    df = _load_frame()
    failures = []
    for learner, stat, position in need:
        label = f"{learner}/{stat}/{position}"
        done = args.out_dir / ENTRIES_DIRNAME / f"{model_filename(learner, stat, position)}.json"
        if done.exists() and not args.force:
            print(f"{label}: already fitted")
            continue
        try:
            params, oof, _ = load_cell_inputs(args.params_dir, args.oof_dir, learner, stat, position)
            entry = fit_cell(learner, stat, position, df=df, params=params, oof=oof, out_dir=args.out_dir,
                             release_id=args.release_id, code_sha=args.code_sha, tolerance=args.tolerance)
        except FinalFitError as exc:
            failures.append(label)
            print(f"{label}: FAILED: {exc}", file=sys.stderr)
            continue
        gate = entry["replay_gate"]
        print(f"{label}: ok trees={entry['trees']} replay max|d|={gate['max_abs_diff']:.3g} "
              f"mean|d|={gate['mean_abs_diff']:.3g} (tol {gate['tolerance_abs']:.3g}) peak_rss={_peak_rss_mb():.0f}MB")
    if failures:
        print(f"{len(failures)} model(s) failed: {failures}", file=sys.stderr)
        return 1
    return 0


def _cmd_assemble(args: argparse.Namespace) -> int:
    path = assemble_models_json(args.out_dir, args.release_dir, args.release_id)
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--release-dir", required=True, type=Path, help="holds ridge_*_coefs.json")
        p.add_argument("--out-dir", required=True, type=Path)
        p.add_argument("--release-id", required=True)
        if name == "fit":
            p.add_argument("--params-dir", required=True, type=Path)
            p.add_argument("--oof-dir", required=True, type=Path)
            p.add_argument("--cells", nargs="*", help="only these stat:pos cells, e.g. passing_yards:QB")
            p.add_argument("--code-sha", default=None)
            p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
            p.add_argument("--force", action="store_true", help="refit models that already have an entry")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        return _cmd_fit(args) if args.cmd == "fit" else _cmd_assemble(args)
    except (FinalFitError, ValueError) as exc:
        print(f"final_fit: FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
