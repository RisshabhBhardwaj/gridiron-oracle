"""Forward producer (decision 0012): next-week projections from the pinned base models.

Reads one forward feature snapshot, runs the SHA-pinned LightGBM/CatBoost models through
the pinned Ridge stack, and writes the rows under the run's pending id (``weekly_auto_<season>_pending_<uuid>``) in one
transaction. Every guard runs before the first write:

    G1  PRODUCT_MODE=artifact_backed; the manifest loads; every model resolves and verifies
    G2  the week is forward: no scores, no game logs, no kickoff at or before as_of,
        previous week all final
    G3  a snapshot exists; its as_of and feature_rows match the rows read
    G4  the frame's feature columns equal each model's stored feature list exactly (no
        alignment, no reordering, no defaults); null rates stay near training's
    G5  after inference: a row set for every pinned cell, finite values, no degraded cell,
        one run id, per-position row counts inside bounds

Cohort (D3 and the 2026-11-14 addendum): rows are computed for every rostered
depth-chart player. Each row stores the availability status known at as_of; the API and
the weekly acceptance check serve only players expected to play.

Light imports on purpose: no PyMC, MLflow, torch or ``ml.monte_carlo`` (D5 keeps that code,
this path never loads it).
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import platform
import re
import resource
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ml.artifact_manifest import (
    REPO_ROOT,
    ArtifactManifest,
    ManifestError,
    get_manifest,
    sha256_of,
)
from ml.inference_client import InferenceClient
from ml.product_mode import ARTIFACT_BACKED, resolve_product_mode
from ml.utils import FEATURE_COLS, MISSING_VALUE_SENTINEL
from pipeline.availability import UNKNOWN, AvailabilityResult, is_served
from pipeline.availability_store import NoAvailabilitySnapshot, load_availability
from pipeline.week_archive import archive_week, archived_rows
from pipeline.week_lock import try_lock_week
from pipeline.serving_rows import (
    attach_causal_conformal_intervals,
    attach_derived_fantasy_projection,
    count_rows_served_by_another_approved_run,
    upsert_projections,
)

logger = logging.getLogger(__name__)

INTERVAL_METHOD = "causal_oof_conformal_90"
SNAPSHOT_KIND = "forward_feature_snapshot"
#: A feature may be this much more often missing forward than in training before G4 refuses.
#: A first-dry-run calibration point (handoff question), not a measured constant.
NULL_RATE_MARGIN = 0.25
#: Rows per position for a forward week (rostered depth-chart players, 32 teams). Same caveat.
DEFAULT_BOUNDS: dict[str, tuple[int, int]] = {
    "QB": (32, 100),
    "RB": (64, 200),
    "WR": (96, 300),
    "TE": (32, 160),
}
_LEARNER_LIBRARY = {"lgbm": "lightgbm", "catboost": "catboost"}


class ForwardGuardError(RuntimeError):
    """A guard refused the run. Nothing was written."""

    def __init__(self, guard: str, message: str) -> None:
        super().__init__(f"[{guard}] {message}")
        self.guard = guard
        self.message = message


class LibraryVersionMismatch(RuntimeError):
    """A model's learner library differs from the one that fitted it."""


def pending_run_id(season: int) -> str:
    """The season's shared pending name: the default when no per-run id is given."""
    return f"weekly_auto_{season}_pending"


# ── memory ────────────────────────────────────────────────────────────────────────────────
def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return _peak_rss_mb()


def _cgroup_mb() -> Optional[float]:
    try:
        return int(Path("/sys/fs/cgroup/memory.current").read_text().strip()) / (1024 * 1024)
    except (OSError, ValueError):
        return None


# ── models ────────────────────────────────────────────────────────────────────────────────
def load_booster(learner: str, path: Path | str):
    """Load one fitted model. Imports the learner library only when that learner is needed."""
    if learner == "lgbm":
        import lightgbm

        return lightgbm.Booster(model_file=str(path))
    if learner == "catboost":
        import catboost

        model = catboost.CatBoostRegressor()
        model.load_model(str(path))
        return model
    raise ValueError(f"unknown learner {learner!r}")


def check_library_versions(meta: dict, learner: str, label: str) -> None:
    """The learner library must be the version that fitted the model; others only warn."""
    pinned = meta.get("library_versions") or {}
    lib = _LEARNER_LIBRARY[learner]
    installed = importlib.import_module(lib).__version__
    if pinned.get(lib) != installed:
        raise LibraryVersionMismatch(
            f"{label}: fitted with {lib} {pinned.get(lib)!r}, installed {installed!r}. "
            "Run on the environment that fitted the models, or refit and re-freeze."
        )
    others = {"numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()}
    for name, version in others.items():
        fitted = pinned.get(name)
        if name == "python" and fitted is not None:  # a patch release does not change model behaviour
            fitted, version = fitted.rsplit(".", 1)[0], version.rsplit(".", 1)[0]
        if fitted not in (None, version):
            warnings.warn(f"{label}: fitted with {name} {pinned.get(name)}, installed {version}")
            logger.warning("%s: fitted with %s %s, installed %s", label, name, pinned.get(name), version)


def _predict(learner: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    if learner == "lgbm":
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X.to_numpy(dtype=float)), dtype=float)


def _ridge(manifest: ArtifactManifest, stat: str, position: str):
    manifest.resolve_coefficient(stat, position)  # pinned and verified; no unpinned fallback
    loaded = InferenceClient(manifest.root / "__no_unpinned_fallback__").load_ridge_coefs(
        stat, position, manifest=manifest
    )
    if loaded is None:
        raise ManifestError(f"no Ridge coefficients for {stat}/{position}")
    coefs, intercept, order = loaded
    return list(zip(order, coefs)), intercept


def predict_cell(
    stat: str,
    position: str,
    frame: pd.DataFrame,
    manifest: ArtifactManifest,
    *,
    run_id: str,
    loader: Callable[[str, Path], Any] = load_booster,
) -> pd.DataFrame:
    """One cell: ``sum(weight * base prediction) + intercept`` with the pinned models.

    A learner with weight 0.0 is never loaded. The identity cell (one learner, weight 1.0,
    intercept 0.0) returns that learner's prediction unchanged. One model is in memory at a
    time.
    """
    pairs, intercept = _ridge(manifest, stat, position)
    active = [(learner, weight) for learner, weight in pairs if weight != 0.0]
    if not active:
        raise ManifestError(f"{stat}/{position}: every ridge weight is 0.0")
    total: Optional[np.ndarray] = None
    train_seasons: list[int] = []
    for learner, weight in active:
        label = f"{learner}_{stat}_{position}"
        path = manifest.resolve_base_model(stat, position, learner)
        meta = dict(manifest.base_model_meta(stat, position, learner))
        check_library_versions(meta, learner, label)
        X = frame[list(meta["features"])].astype(float).fillna(MISSING_VALUE_SENTINEL)
        model = loader(learner, path)
        try:
            prediction = _predict(learner, model, X)
        finally:
            del model
            gc.collect()
        part = prediction if (len(active) == 1 and weight == 1.0) else weight * prediction
        total = part if total is None else total + part
        train_seasons.append(int(meta["max_train_season"]))
    if not (len(active) == 1 and active[0][1] == 1.0 and intercept == 0.0):
        total = total + intercept
    out = frame[["player_id", "game_id", "season", "week"]].copy()
    out["stat"] = stat
    out["position"] = position
    out["projection"] = total
    out["max_train_season"] = min(train_seasons)
    out["pipeline_run_id"] = run_id
    out["degraded"] = ~np.isfinite(total)
    return out.reset_index(drop=True)


# ── guards ────────────────────────────────────────────────────────────────────────────────
def _rows(conn, sql: str, params: tuple) -> tuple[list[str], list[tuple]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in (cur.description or [])]
        return columns, cur.fetchall()


def _plan(manifest: ArtifactManifest) -> list[dict]:
    """G1: every pinned cell, its ridge weights, and each active model resolved and verified."""
    try:
        manifest.require_base_models()
    except ManifestError as exc:
        raise ForwardGuardError("G1", str(exc)) from exc
    cells = []
    for stat, position in sorted(manifest.stack_index):
        try:
            pairs, intercept = _ridge(manifest, stat, position)
            manifest.resolve_stack(stat, position)
        except Exception as exc:
            raise ForwardGuardError("G1", f"{stat}/{position}: {exc}") from exc
        active = [learner for learner, weight in pairs if weight != 0.0]
        for learner in active:
            try:
                manifest.resolve_base_model(stat, position, learner)
                manifest.base_model_meta(stat, position, learner)
            except Exception as exc:
                raise ForwardGuardError("G1", f"{learner}_{stat}_{position}: {exc}") from exc
        cells.append({"stat": stat, "position": position, "learners": active})
    if not cells:
        raise ForwardGuardError("G1", "the manifest pins no stack cells")
    return cells


def _check_week_is_forward(conn, season: int, week: int) -> None:
    _, games = _rows(
        conn,
        "-- g2:target_week\nSELECT id, kickoff_at, home_score, away_score FROM games "
        "WHERE season = %s AND week = %s",
        (season, week),
    )
    if not games:
        raise ForwardGuardError("G2", f"no games for season={season} week={week}")
    if any(g[2] is not None or g[3] is not None for g in games):
        raise ForwardGuardError("G2", f"week {week} already has scores; it is not a forward week")
    _, logs = _rows(
        conn,
        "-- g2:logs\nSELECT COUNT(*) AS n FROM game_logs WHERE season = %s AND week = %s",
        (season, week),
    )
    if logs and int(logs[0][0]) > 0:
        raise ForwardGuardError("G2", f"week {week} already has {int(logs[0][0])} game logs")
    if week > 1:
        _, previous = _rows(
            conn,
            "-- g2:previous_week\nSELECT id, home_score, away_score FROM games "
            "WHERE season = %s AND week = %s",
            (season, week - 1),
        )
        if not previous or any(g[1] is None or g[2] is None for g in previous):
            raise ForwardGuardError("G2", f"previous week {week - 1} is not all final")


def _check_kickoffs_after(conn, season: int, week: int, as_of: datetime) -> None:
    _, games = _rows(
        conn,
        "-- g2:target_week\nSELECT id, kickoff_at, home_score, away_score FROM games "
        "WHERE season = %s AND week = %s",
        (season, week),
    )
    started = [g[0] for g in games if g[1] is not None and pd.Timestamp(g[1]) <= pd.Timestamp(as_of)]
    if started:
        raise ForwardGuardError(
            "G2",
            f"{len(started)} game(s) in week {week} have a kickoff at or before as_of "
            f"{as_of.isoformat()}: {sorted(started)[:5]}",
        )


def _read_snapshot(conn, season: int, week: int, feature_run_id: Optional[str]) -> Optional[dict]:
    sql = (
        "-- g3:snapshot\nSELECT pipeline_run_id, as_of, feature_rows, source_summary FROM forecast_runs "
        "WHERE season = %s AND week = %s AND source_summary->>'kind' = %s"
    )
    params: tuple = (season, week, SNAPSHOT_KIND)
    if feature_run_id:
        sql += " AND pipeline_run_id = %s"
        params += (feature_run_id,)
    sql += " ORDER BY as_of DESC, created_at DESC LIMIT 1"
    _, rows = _rows(conn, sql, params)
    if not rows:
        return None
    run_id, as_of, feature_rows, _summary = rows[0]
    return {"pipeline_run_id": run_id, "as_of": pd.Timestamp(as_of).to_pydatetime(), "feature_rows": int(feature_rows)}


def _read_frame(conn, season: int, week: int, snapshot: dict) -> pd.DataFrame:
    columns, rows = _rows(
        conn,
        "-- g3:frame\nSELECT * FROM feature_matrix WHERE season = %s AND week = %s AND as_of = %s",
        (season, week, snapshot["as_of"]),
    )
    return pd.DataFrame(rows, columns=columns)


def _check_frame(frame: pd.DataFrame, snapshot: dict) -> None:
    for column in ("player_id", "game_id", "position", "as_of"):
        if column not in frame.columns:
            raise ForwardGuardError("G3", f"feature rows have no {column!r} column")
    stamps = pd.to_datetime(frame["as_of"], utc=True)
    expected = pd.Timestamp(snapshot["as_of"]).tz_convert("UTC")
    wrong = int((stamps != expected).sum())
    if wrong:
        raise ForwardGuardError(
            "G3", f"{wrong} feature row(s) were built at an as_of other than the snapshot's {expected.isoformat()}"
        )
    if len(frame) != snapshot["feature_rows"]:
        raise ForwardGuardError(
            "G3", f"snapshot feature_rows={snapshot['feature_rows']} but {len(frame)} rows were read"
        )


def _check_features(frame: pd.DataFrame, cells: list[dict], manifest: ArtifactManifest, margin: float) -> None:
    """G4: exact feature-list equality per model, and null rates near training's."""
    present = [c for c in FEATURE_COLS if c in frame.columns]
    for cell in cells:
        stat, position = cell["stat"], cell["position"]
        rows = frame[frame["position"] == position]
        for learner in cell["learners"]:
            label = f"{learner}_{stat}_{position}"
            meta = manifest.base_model_meta(stat, position, learner)
            stored = list(meta["features"])
            missing = [c for c in stored if c not in frame.columns]
            if missing:
                raise ForwardGuardError("G4", f"{label}: feature column(s) missing from the frame: {missing[:8]}")
            if present != stored:
                extra = [c for c in present if c not in stored]
                raise ForwardGuardError(
                    "G4",
                    f"{label}: the frame's feature columns differ from the model's stored list "
                    f"(extra {extra[:8]}, order {'differs' if not extra else 'n/a'}); {stat}/{position}",
                )
            if rows.empty:
                continue
            train_rates = meta.get("null_rates") or {}
            worse = {
                c: (float(rows[c].isna().mean()), float(train_rates.get(c, 0.0)))
                for c in stored
                if float(rows[c].isna().mean()) - float(train_rates.get(c, 0.0)) > margin
            }
            if worse:
                shown = ", ".join(f"{c} forward {f:.0%} vs training {t:.0%}" for c, (f, t) in sorted(worse.items())[:8])
                raise ForwardGuardError("G4", f"{label}: null rate above the training margin ({margin:.0%}): {shown}")


def _check_predictions(
    preds: dict[tuple[str, str], pd.DataFrame],
    cells: list[dict],
    run_id: str,
    bounds: dict[str, tuple[int, int]],
) -> None:
    """G5: what the models returned, checked before anything is built for writing."""
    for cell in cells:
        key = (cell["stat"], cell["position"])
        out = preds.get(key)
        if out is None or out.empty:
            raise ForwardGuardError("G5", f"no rows for {key[0]}/{key[1]}")
        if out["degraded"].fillna(True).astype(bool).any():
            raise ForwardGuardError("G5", f"{key[0]}/{key[1]} returned degraded rows")
        if not np.isfinite(out["projection"].to_numpy(dtype=float)).all():
            raise ForwardGuardError("G5", f"{key[0]}/{key[1]} has non-finite projections; values must be finite")
        if out.duplicated(["player_id", "game_id", "stat"]).any():
            raise ForwardGuardError("G5", f"{key[0]}/{key[1]} repeats a (player, game, stat) row")
    ids = {str(v) for out in preds.values() for v in out["pipeline_run_id"].dropna().unique()}
    if ids != {run_id}:
        raise ForwardGuardError("G5", f"expected the single run id {run_id!r}, got {sorted(ids)}")
    for position in sorted({c["position"] for c in cells}):
        if position not in bounds:
            raise ForwardGuardError("G5", f"no row-count bounds for position {position}")
        low, high = bounds[position]
        counts = {len(preds[(c["stat"], position)]) for c in cells if c["position"] == position}
        for count in counts:
            if not low <= count <= high:
                raise ForwardGuardError("G5", f"{position}: {count} rows, outside the expected {low}-{high}")


# ── build rows ────────────────────────────────────────────────────────────────────────────
def _with_intervals(cell_rows: pd.DataFrame, manifest: ArtifactManifest, season: int) -> pd.DataFrame:
    stat, position = str(cell_rows["stat"].iloc[0]), str(cell_rows["position"].iloc[0])
    history = pd.read_csv(manifest.resolve_stack(stat, position))
    history = history.assign(projection=pd.to_numeric(history["y_pred"], errors="coerce"))
    history = history[history["season"].astype(int) < season][["player_id", "game_id", "season", "week", "projection", "y_true"]]
    forward = cell_rows[["player_id", "game_id", "season", "week", "projection"]].assign(y_true=np.nan)
    combined = pd.concat([history, forward], ignore_index=True)
    done = attach_causal_conformal_intervals(combined, only_seasons=[season]).iloc[len(history):]
    out = cell_rows.copy()
    out["floor"] = done["floor"].to_numpy()
    out["ceiling"] = done["ceiling"].to_numpy()
    out["interval_method"] = done["interval_method"].to_numpy()
    return out


def _build_rows(
    preds: dict[tuple[str, str], pd.DataFrame],
    manifest: ArtifactManifest,
    season: int,
    statuses: dict[str, Optional[str]],
) -> pd.DataFrame:
    parts = []
    for (stat, _position), out in sorted(preds.items()):
        cell = _with_intervals(out.drop(columns=["degraded"]), manifest, season)
        cell["p25"] = None
        cell["p75"] = None
        if stat == "fantasy_ppr":
            cell["fantasy_projection"] = cell["projection"]
            cell["fantasy_floor"] = cell["floor"]
            cell["fantasy_ceiling"] = cell["ceiling"]
        else:
            cell["fantasy_projection"] = None
            cell["fantasy_floor"] = None
            cell["fantasy_ceiling"] = None
        parts.append(cell)
    df = pd.concat(parts, ignore_index=True)
    df = attach_derived_fantasy_projection(df)
    df["availability_status"] = df["player_id"].map(statuses)
    df["availability_status"] = df["availability_status"].astype(object).where(df["availability_status"].notna(), None)
    bad = df[(df["interval_method"] != INTERVAL_METHOD) | ~np.isfinite(df[["floor", "ceiling"]].to_numpy(dtype=float)).all(axis=1)]
    if len(bad):
        raise ForwardGuardError(
            "G5", f"{len(bad)} row(s) have no causal conformal interval (pinned history too short?)"
        )
    return df


# ── run ───────────────────────────────────────────────────────────────────────────────────
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Written at image build from the CODE_GIT_SHA build arg. A file, not an ENV: compose `env_file: .env`
# overrides image ENV at runtime, so an env value could name a commit the code was not built from.
CODE_SHA_FILE = REPO_ROOT / "CODE_GIT_SHA"


def _valid_sha(value: str) -> bool:
    return bool(_SHA_RE.match(value))


def _code_sha() -> str:
    """The commit this code was built from, a 40-hex sha, or "unknown".

    In an image the baked file is authoritative and the environment is ignored. Outside an image
    (no file: a Mac or a dev shell) the env var is used when valid, then git, but only for a clean tree.
    """
    if CODE_SHA_FILE.exists():
        try:
            value = CODE_SHA_FILE.read_text().strip()
        except OSError:
            value = ""
        if _valid_sha(value):
            return value
        logger.warning("baked code sha %r is not a 40-hex commit; recording 'unknown'", value)
        return "unknown"
    for name in ("CODE_GIT_SHA", "GIT_SHA"):
        value = os.environ.get(name, "").strip()
        if _valid_sha(value):
            return value
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if head.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip() and _valid_sha(head.stdout.strip()):
            return head.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    logger.warning("code git sha unknown (no baked file, no valid env, no clean git tree); recording 'unknown'")
    return "unknown"


def _models_json_sha(manifest: ArtifactManifest) -> str:
    for relpath, digest in manifest.digests.items():
        if Path(relpath).name == "MODELS.json":
            return digest
    raise ForwardGuardError("G1", "the manifest pins no MODELS.json")


def run_forward(
    conn,
    season: int,
    week: int,
    *,
    feature_run_id: Optional[str] = None,
    manifest: Optional[ArtifactManifest] = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    product_mode: Optional[str] = None,
    predictor: Callable[..., pd.DataFrame] = predict_cell,
    availability: Optional[Callable[..., AvailabilityResult]] = None,
    dry_run: bool = False,
    allow_no_availability: bool = False,
    code_sha: Optional[str] = None,
    bounds: Optional[dict[str, tuple[int, int]]] = None,
    null_margin: float = NULL_RATE_MARGIN,
    replace_week: bool = False,
    pending_id: Optional[str] = None,
) -> dict:
    """Run G1-G5, compute every row, then write once. Returns a report.

    ``replace_week`` (manual, after a dump) copies the week's existing projection rows to
    ``projections_replaced``, deletes them and inserts the new ones in one transaction, so a failure
    inside the producer rolls back and leaves the old week served. After the producer, the weekly
    refresh restores the copy on any failure and discards it on promotion (``pipeline.week_archive``).

    ``pending_id`` is the run's own pending run id (``pipeline.pending_runs.new_pending_run_id``); the
    rows and the ``forecast_runs`` summary carry it, and step 8 promotes exactly that id. Without it the
    producer falls back to the season's shared name (manual runs and the guard tests).
    """
    started = now()
    mode = product_mode or resolve_product_mode()
    if mode != ARTIFACT_BACKED:
        raise ForwardGuardError("G1", f"PRODUCT_MODE must be artifact_backed, got {mode!r}")
    sha = code_sha or _code_sha()
    if not _valid_sha(sha) and not dry_run:
        raise ForwardGuardError(
            "D8", f"code_git_sha {sha!r} is not a 40-hex commit: refusing to write a forward run nobody can "
                  "trace to a commit. Set CODE_GIT_SHA in .env to the synced commit and rebuild the image "
                  "(it is baked to /app/CODE_GIT_SHA at build)."
        )
    manifest = manifest or get_manifest()
    cells = _plan(manifest)

    _check_week_is_forward(conn, season, week)
    snapshot = _read_snapshot(conn, season, week, feature_run_id)
    if snapshot is None:
        raise ForwardGuardError("G3", f"no forward feature snapshot for season={season} week={week}")
    as_of = snapshot["as_of"]
    _check_kickoffs_after(conn, season, week, as_of)
    frame = _read_frame(conn, season, week, snapshot)
    _check_frame(frame, snapshot)
    _check_features(frame, cells, manifest, null_margin)

    player_ids = sorted({str(p) for p in frame["player_id"]})
    if allow_no_availability:
        statuses: dict[str, Optional[str]] = {pid: UNKNOWN for pid in player_ids}
        avail = AvailabilityResult(statuses=statuses, as_of=as_of, counts={UNKNOWN: len(player_ids)})
    else:
        loader = availability or load_availability
        try:
            avail = loader(conn, season, week, player_ids, as_of)
        except NoAvailabilitySnapshot as exc:
            raise ForwardGuardError(
                "D3", f"availability unknown: {exc}. Capture a snapshot first, or pass allow_no_availability."
            ) from exc
    if allow_no_availability:
        logger.warning("availability NOT applied (explicit override): every row is stored as 'Unknown'")

    run_id = pending_id or pending_run_id(season)
    preds: dict[tuple[str, str], pd.DataFrame] = {}
    memory_cells = []
    for cell in cells:
        stat, position = cell["stat"], cell["position"]
        rows = frame[frame["position"] == position].reset_index(drop=True)
        before = _rss_mb()
        preds[(stat, position)] = predictor(stat, position, rows, manifest, run_id=run_id)
        after = _rss_mb()
        memory_cells.append({
            "cell": f"{stat}:{position}", "rss_before_mb": round(before, 1),
            "rss_after_mb": round(after, 1), "peak_rss_mb": round(_peak_rss_mb(), 1),
        })
        logger.info("cell %s:%s done, rss %.0f -> %.0f MB, peak %.0f MB", stat, position, before, after, _peak_rss_mb())
    _check_predictions(preds, cells, run_id, bounds or DEFAULT_BOUNDS)

    df = _build_rows(preds, manifest, season, avail.statuses)
    not_served = int(sum(0 if is_served(s) else 1 for s in df["availability_status"]))
    memory = {"peak_rss_mb": round(_peak_rss_mb(), 1), "cgroup_memory_current_mb": _cgroup_mb(), "cells": memory_cells}
    report = {
        "dry_run": dry_run, "season": season, "week": week, "pipeline_run_id": run_id,
        "feature_snapshot_id": snapshot["pipeline_run_id"], "as_of": as_of.isoformat(),
        "projection_rows": int(len(df)), "rows_served": int(len(df)) - not_served, "rows_not_served": not_served,
        "availability": {"snapshot_id": avail.snapshot_id, "counts": dict(avail.counts),
                         "newest_report_age": str(avail.newest_report_age), "override": allow_no_availability},
        "memory": memory,
    }
    if dry_run:
        logger.info("dry run: %d rows computed, nothing written", len(df))
        return report

    stamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = f"forward_projections_{season}_w{week:02d}_{stamp}"
    summary = {
        "kind": "forward_projections", "pipeline_run_id": run_id, "feature_snapshot_id": snapshot["pipeline_run_id"],
        "as_of": as_of.isoformat(), "availability_snapshot_id": avail.snapshot_id,
        "current_baseline_sha256": sha256_of(manifest.path), "models_json_sha256": _models_json_sha(manifest),
        "code_git_sha": sha, "projection_rows": int(len(df)),
        "availability_override": allow_no_availability, "replace_week": replace_week,
        "rows_not_served": not_served, "memory": memory,
    }
    if replace_week and getattr(conn, "autocommit", False) is True:
        raise ForwardGuardError(
            "M1", "--replace-week needs a transaction: this connection is in autocommit mode, so the "
                  "delete would commit on its own. Nothing was written."
        )
    try:
        # One writer per week: the Friday refresh updates these same rows. Held until the commit below.
        if not try_lock_week(conn, season, week):
            raise ForwardGuardError(
                "L2", f"another writer holds the lock for season={season} week={week} (the Friday availability "
                      "refresh or a second producer); nothing was written. Retry when it has finished."
            )
        with conn.cursor() as cur:
            if replace_week:
                leftover = archived_rows(cur, season, week)
                if leftover:
                    raise ForwardGuardError(
                        "M1", f"projections_replaced still holds {leftover} rows for season={season} week={week} "
                              "from an earlier replace. Restore them (scripts/restore_replaced_week.py) or "
                              "promote the week before replacing it again. Nothing was written."
                    )
                report["rows_replaced"] = report["rows_archived"] = archive_week(cur, season, week)
                summary["rows_replaced"] = summary["rows_archived"] = report["rows_replaced"]
            # The upsert silently skips a row another approved run already serves. For a new forward
            # week that must not happen: count first and refuse, so "N written" is never a guess.
            skipped = count_rows_served_by_another_approved_run(cur, df)
            if skipped:
                raise ForwardGuardError(
                    "L1", f"{skipped} of {len(df)} rows are already served by another approved run and would "
                          "be skipped; nothing was written. Use --replace-week (with a dump) to rewrite a served week."
                )
            report["rows_skipped_approved"] = summary["rows_skipped_approved"] = 0
            report["rows_written"] = summary["rows_written"] = int(len(df))
            cur.execute(
                "INSERT INTO forecast_runs (pipeline_run_id, season, week, as_of, feature_rows, source_summary) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (key, season, week, as_of, int(len(df)), json.dumps(summary, default=str)),
            )
        upsert_projections(conn, df)  # commits once; rolls the forecast_runs insert back on failure
    except Exception:
        conn.rollback()
        raise
    report["forecast_run_key"] = key
    logger.info("forward projections written: %s (%d rows, %d not served)", key, len(df), not_served)
    return report


__all__ = [
    "ForwardGuardError", "LibraryVersionMismatch", "load_booster", "pending_run_id",
    "predict_cell", "run_forward",
]
