"""A tiny but real pinned release for the forward-serving tests (decision 0012, commit 5).

Trains small LightGBM and CatBoost models on synthetic rows that use the real ``FEATURE_COLS``,
writes them with ridge files, stack OOF CSVs and a ``MODELS.json``, and pins everything by
SHA-256 in a ``current_baseline.json`` under a temp root. Cells cover each real shape:

    targets:TE          both learners, non-zero intercept
    receiving_yards:WR  catboost only (lgbm weight 0.0, never pinned)
    receiving_tds:WR    lgbm only, weight 1.0 and intercept 0.0: the identity cell
    passing_yards:QB    lgbm only
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.season_constants import LAST_COMPLETE_SEASON
from ml.utils import FEATURE_COLS, MISSING_VALUE_SENTINEL

RELEASE = "causal_fwd_test"
SEASON, WEEK = 2026, 11
AS_OF = datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)
EXT = {"lgbm": "txt", "catboost": "cbm"}

CELLS: dict[tuple[str, str], dict] = {
    ("targets", "TE"): {"weights": {"lgbm": 0.4, "catboost": 0.6}, "intercept": 0.25},
    ("receiving_yards", "WR"): {"weights": {"lgbm": 0.0, "catboost": 1.0}, "intercept": 0.0},
    ("receiving_tds", "WR"): {"weights": {"lgbm": 1.0, "catboost": 0.0}, "intercept": 0.0},
    ("passing_yards", "QB"): {"weights": {"lgbm": 1.0, "catboost": 0.0}, "intercept": 0.0},
}
# players per position in the forward week (32 teams: inside the row-count bounds)
POSITION_COUNTS = {"QB": 40, "RB": 70, "WR": 110, "TE": 40}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(root: Path, relpath: str, data: bytes | str) -> str:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return relpath


def training_frame(n: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    X.iloc[:, 0] = np.where(rng.random(n) < 0.05, np.nan, X.iloc[:, 0])  # a little real missingness
    return X


def forward_frame(seed: int = 11) -> pd.DataFrame:
    """The feature_matrix rows of the forward week: one row per rostered depth-chart player."""
    rng = np.random.default_rng(seed)
    rows = []
    for position, count in POSITION_COUNTS.items():
        for i in range(count):
            rows.append({"player_id": f"{position}{i:03d}", "game_id": f"{SEASON}_{WEEK:02d}_{i % 16:02d}",
                         "season": SEASON, "week": WEEK, "position": position,
                         "team": f"T{i % 32:02d}", "as_of": AS_OF})
    frame = pd.DataFrame(rows)
    feats = pd.DataFrame(rng.normal(size=(len(frame), len(FEATURE_COLS))), columns=FEATURE_COLS)
    feats.iloc[:, 0] = np.where(rng.random(len(frame)) < 0.05, np.nan, feats.iloc[:, 0])
    return pd.concat([frame, feats], axis=1)


def _train(learner: str, X: pd.DataFrame, y: np.ndarray, path: Path) -> None:
    Xf = X.fillna(MISSING_VALUE_SENTINEL)
    if learner == "lgbm":
        model = lgb.LGBMRegressor(n_estimators=12, num_leaves=5, min_child_samples=5, verbose=-1, n_jobs=1)
        model.fit(Xf, y)
        model.booster_.save_model(str(path))
    else:
        model = cb.CatBoostRegressor(iterations=12, depth=3, verbose=False, thread_count=1,
                                     allow_writing_files=False, random_seed=3)
        model.fit(Xf.values, y)
        model.save_model(str(path), format="cbm")


def _stack_csv(seed: int) -> str:
    rng = np.random.default_rng(seed)
    parts = []
    for season in range(2021, LAST_COMPLETE_SEASON + 1):
        n = 150
        pred = np.abs(rng.normal(20, 6, n))
        parts.append(pd.DataFrame({
            "player_id": [f"h{i:03d}" for i in range(n)], "game_id": [f"{season}_{i:03d}" for i in range(n)],
            "season": season, "week": (np.arange(n) % 17) + 1,
            "y_pred": pred, "y_true": np.abs(pred + rng.normal(0, 4, n)),
            "max_train_season": season - 1,
        }))
    return pd.concat(parts).to_csv(index=False)


@dataclass
class Release:
    root: Path
    manifest_path: Path
    models_dir: Path
    meta_relpath: str

    def rewrite_meta(self, mutate) -> None:
        """Change MODELS.json and re-pin its digest, so only the content defect remains."""
        path = self.root / self.meta_relpath
        meta = json.loads(path.read_text())
        mutate(meta)
        path.write_text(json.dumps(meta))
        manifest = json.loads(self.manifest_path.read_text())
        manifest["artifact_digests"][self.meta_relpath] = sha(path.read_bytes())
        self.manifest_path.write_text(json.dumps(manifest))


def build_forward_release(root: Path) -> Release:
    rel_dir = f"releases/candidates/{RELEASE}"
    X = training_frame()
    rng = np.random.default_rng(5)
    digests: dict[str, str] = {}
    stacks, ridges, models, entries = [], [], [], []
    features_sha = sha(json.dumps(list(FEATURE_COLS)).encode())
    import lightgbm, catboost  # noqa: E401
    versions = {"lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "python": "3.11.0"}
    for i, ((stat, pos), cfg) in enumerate(CELLS.items()):
        stacks.append(write(root, f"ml/oof/stack_{stat}_{pos}_20261114.csv", _stack_csv(100 + i)))
        ridges.append(write(root, f"{rel_dir}/ridge_{stat}_{pos}_coefs.json", json.dumps({
            "learner_order": ["lgbm", "catboost"], "weights": cfg["weights"], "intercept": cfg["intercept"]})))
        for learner, weight in cfg["weights"].items():
            if weight == 0.0:
                continue
            name = f"{learner}_{stat}_{pos}.{EXT[learner]}"
            path = root / rel_dir / "models" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _train(learner, X, rng.normal(10, 3, len(X)), path)
            relpath = f"{rel_dir}/models/{name}"
            models.append(relpath)
            digests[relpath] = sha(path.read_bytes())
            entries.append({
                "learner": learner, "stat": stat, "position": pos, "file": name, "sha256": digests[relpath],
                "release_id": RELEASE, "train_seasons": list(range(2019, LAST_COMPLETE_SEASON + 1)),
                "max_train_season": LAST_COMPLETE_SEASON, "features": list(FEATURE_COLS),
                "features_sha256": features_sha, "params": {}, "params_oof": {}, "trees": 12,
                "trees_rule": "replay_best_iteration", "n_rows": len(X), "training_data_sha256": "0" * 64,
                "target_summary": {}, "null_rates": {c: float(X[c].isna().mean()) for c in FEATURE_COLS},
                "code_git_sha": "a" * 40, "library_versions": versions, "replay_gate": {"passed": True},
                "created_at": "2026-11-14T00:00:00+00:00",
            })
    meta_rel = write(root, f"{rel_dir}/models/MODELS.json",
                     json.dumps({"schema_version": 1, "release_id": RELEASE, "models": entries}))
    for relpath in [*stacks, *ridges, meta_rel]:
        digests[relpath] = sha((root / relpath).read_bytes())
    manifest = {
        "artifacts": {"stacks": stacks, "ridge_coefficients": ridges, "base_models": [*models, meta_rel]},
        "artifact_digests": digests,
        "projection_policy": {"approved_pipeline_run_ids": ["weekly_auto_2026"]},
    }
    manifest_path = root / "releases/current_baseline.json"
    write(root, "releases/current_baseline.json", json.dumps(manifest))
    return Release(root, manifest_path, root / rel_dir / "models", meta_rel)
