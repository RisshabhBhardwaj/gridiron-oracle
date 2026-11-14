"""The forward producer (decision 0012, commit 5): guards G1-G5, the cohort, and the write.

Each guard test breaks exactly one thing and asserts two facts: the right guard named the
problem, and nothing was written. A fake connection records every statement in order, so
"no write before the last guard" is checked on the log, not assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from backend.tests._fake_db import FakeConn, install_batch_recorder
from backend.tests._forward_release import (
    AS_OF, CELLS, POSITION_COUNTS, RELEASE, SEASON, WEEK, Release, build_forward_release,
    forward_frame, sha,
)
from ml import forward_serving
from ml.artifact_manifest import load_manifest
from ml.forward_serving import ForwardGuardError, predict_cell, run_forward
from ml.utils import FEATURE_COLS, MISSING_VALUE_SENTINEL

NOW = AS_OF + timedelta(hours=1)
SNAPSHOT_ID = "forward_features_2026_w11_test"
CODE_SHA = "c" * 40
FRAME_COLUMNS = ["player_id", "game_id", "season", "week", "position", "team", "as_of", *FEATURE_COLS]


@pytest.fixture(scope="module")
def release(tmp_path_factory) -> Release:
    return build_forward_release(tmp_path_factory.mktemp("fwd_release"))


@pytest.fixture
def manifest(release: Release):
    return load_manifest(release.manifest_path, root=release.root)


def _games(kickoff: datetime, scored: bool, n: int = 16):
    return [(f"g{i}", kickoff, 20 if scored else None, 17 if scored else None) for i in range(n)]


def make_conn(
    *,
    frame: pd.DataFrame | None = None,
    kickoff: datetime | None = None,
    target_scored: bool = False,
    logs: int = 0,
    previous_final: bool = True,
    snapshot_rows: int | None = None,
    snapshot: bool = True,
    reports: list[tuple] | None = None,
    avail_fetched: datetime | None = None,
    lock: bool = True,
) -> FakeConn:
    frame = forward_frame() if frame is None else frame
    kickoff = kickoff or AS_OF + timedelta(days=5)
    rows = [tuple(r) for r in frame.itertuples(index=False, name=None)]
    columns = list(frame.columns)
    snapshot_rows = len(frame) if snapshot_rows is None else snapshot_rows
    if reports is None:
        reports = [
            ("WR000", "Out", None, None, None),
            ("WR001", "Questionable", None, None, None),
            ("RB000", None, None, "RES", None),
            ("TE000", "Doubtful", None, None, None),
        ]
    fetched = avail_fetched or AS_OF - timedelta(hours=1)
    prev = _games(AS_OF - timedelta(days=4), previous_final)
    return FakeConn([
        ("-- g2:target_week", (["id", "kickoff_at", "home_score", "away_score"], _games(kickoff, target_scored))),
        ("-- g2:logs", (["n"], [(logs,)])),
        ("-- g2:previous_week", (["id", "home_score", "away_score"], [(g[0], g[2], g[3]) for g in prev])),
        ("-- g3:snapshot", (
            ["pipeline_run_id", "as_of", "feature_rows", "source_summary"],
            [(SNAPSHOT_ID, AS_OF, snapshot_rows, {"kind": "forward_feature_snapshot"})] if snapshot else [],
        )),
        ("-- g3:frame", (columns, rows)),
        ("FROM availability_snapshots", (["snapshot_id", "fetched_at"], [("avail_1", fetched)])),
        ("FROM availability_reports", (
            ["player_id", "report_status", "practice_status", "roster_status", "date_modified"], reports)),
        ("-- wk:lock", (["locked"], [(lock,)])),  # last: tests address routes above by index
    ])


def go(conn: FakeConn, manifest, **overrides):
    kwargs = dict(manifest=manifest, now=lambda: NOW, code_sha=CODE_SHA, product_mode="artifact_backed")
    kwargs.update(overrides)
    return run_forward(conn, SEASON, WEEK, **kwargs)


@pytest.fixture(autouse=True)
def _recorder(monkeypatch) -> None:
    install_batch_recorder(monkeypatch)


def assert_refused(conn: FakeConn, guard: str, exc: ForwardGuardError, match: str | None = None) -> None:
    assert exc.guard == guard, f"expected {guard}, got {exc.guard}: {exc}"
    if match:
        assert match in str(exc)
    assert conn.writes == []
    assert conn.commits == 0


# ── G1 ────────────────────────────────────────────────────────────────────────────────────
def test_g1_refuses_graceful_fallback_mode(manifest) -> None:
    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest, product_mode="graceful_fallback")
    assert_refused(conn, "G1", e.value, "artifact_backed")
    assert conn.log == []  # refused before touching the database


def test_g1_refuses_a_manifest_without_base_models(release: Release, tmp_path: Path, manifest) -> None:
    other = json.loads(release.manifest_path.read_text())
    del other["artifacts"]["base_models"]
    path = tmp_path / "current_baseline.json"
    path.write_text(json.dumps(other))
    bare = load_manifest(path, root=release.root)
    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, bare)
    assert_refused(conn, "G1", e.value, "base_models")


def test_g1_refuses_a_model_whose_digest_changed(release: Release, tmp_path: Path) -> None:
    root = tmp_path / "copy"
    import shutil

    shutil.copytree(release.root, root)
    victim = root / "releases/candidates" / RELEASE / "models" / "lgbm_passing_yards_QB.txt"
    victim.write_bytes(victim.read_bytes() + b"\n# tampered")
    m = load_manifest(root / "releases/current_baseline.json", root=root)
    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, m)
    assert_refused(conn, "G1", e.value, "lgbm_passing_yards_QB")


# ── G2 ────────────────────────────────────────────────────────────────────────────────────
def test_g2_refuses_a_week_with_scores(manifest) -> None:
    conn = make_conn(target_scored=True)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G2", e.value, "scores")


def test_g2_refuses_a_week_with_game_logs(manifest) -> None:
    conn = make_conn(logs=3)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G2", e.value, "game logs")


def test_g2_refuses_a_kickoff_at_or_before_as_of_thursday_night(manifest) -> None:
    # A Thursday-night game has kicked off but nothing is scored yet: the week looks forward
    # to a score check, and is not (resolve_season_and_week would already call it complete).
    conn = make_conn(kickoff=AS_OF)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G2", e.value, "kickoff")


def test_g2_accepts_a_kickoff_just_after_as_of(manifest) -> None:
    conn = make_conn(kickoff=AS_OF + timedelta(seconds=1))
    assert go(conn, manifest, dry_run=True)["dry_run"] is True


def test_g2_refuses_when_the_previous_week_is_not_final(manifest) -> None:
    conn = make_conn(previous_final=False)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G2", e.value, "previous week")


# ── G3 ────────────────────────────────────────────────────────────────────────────────────
def test_g3_refuses_without_a_feature_snapshot(manifest) -> None:
    conn = make_conn(snapshot=False)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G3", e.value, "snapshot")


def test_g3_refuses_when_feature_rows_do_not_match_the_snapshot(manifest) -> None:
    conn = make_conn(snapshot_rows=len(forward_frame()) + 5)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G3", e.value, "feature_rows")


def test_g3_refuses_rows_built_at_another_as_of(manifest) -> None:
    frame = forward_frame()
    frame.loc[frame.index[:3], "as_of"] = AS_OF - timedelta(days=1)
    conn = make_conn(frame=frame)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G3", e.value, "as_of")


# ── G4 ────────────────────────────────────────────────────────────────────────────────────
def test_g4_refuses_a_missing_feature_column(manifest) -> None:
    frame = forward_frame().drop(columns=[FEATURE_COLS[3]])
    conn = make_conn(frame=frame)
    conn.routes[4] = ("-- g3:frame", (
        list(frame.columns), [tuple(r) for r in frame.itertuples(index=False, name=None)]))
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G4", e.value, FEATURE_COLS[3])


def test_g4_refuses_a_stored_feature_list_that_differs(release: Release, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(release.root, root)
    r = Release(root, root / "releases/current_baseline.json", root / "releases/candidates" / RELEASE / "models",
                release.meta_relpath)

    def reorder(meta):
        for entry in meta["models"]:
            if entry["stat"] == "targets" and entry["learner"] == "lgbm":
                entry["features"] = list(reversed(entry["features"]))

    r.rewrite_meta(reorder)
    m = load_manifest(r.manifest_path, root=root)
    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, m)
    assert_refused(conn, "G4", e.value, "targets")


def test_g4_refuses_a_column_that_is_empty_forward_but_not_in_training(manifest) -> None:
    frame = forward_frame()
    frame[FEATURE_COLS[5]] = np.nan  # a builder bug: 100% null forward, 0% in training
    conn = make_conn(frame=frame)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G4", e.value, FEATURE_COLS[5])


def test_g4_tolerates_null_rates_inside_the_margin(manifest) -> None:
    frame = forward_frame()
    mask = np.random.default_rng(1).random(len(frame)) < 0.08
    frame.loc[mask, FEATURE_COLS[6]] = np.nan
    assert go(make_conn(frame=frame), manifest, dry_run=True)["dry_run"] is True


# ── G5 ────────────────────────────────────────────────────────────────────────────────────
def _wrap(mutate):
    def predictor(stat, position, frame, manifest, **kw):
        out = predict_cell(stat, position, frame, manifest, **kw)
        return mutate(stat, position, out)
    return predictor


def test_g5_refuses_a_degraded_cell(manifest) -> None:
    def mark(stat, position, out):
        if (stat, position) == ("targets", "TE"):
            out["degraded"] = True
        return out

    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest, predictor=_wrap(mark))
    assert_refused(conn, "G5", e.value, "degraded")


def test_g5_refuses_two_run_ids(manifest) -> None:
    def other(stat, position, out):
        if (stat, position) == ("targets", "TE"):
            out["pipeline_run_id"] = "some_other_run"
        return out

    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest, predictor=_wrap(other))
    assert_refused(conn, "G5", e.value, "run id")


def test_g5_refuses_non_finite_values(manifest) -> None:
    def poison(stat, position, out):
        if (stat, position) == ("passing_yards", "QB"):
            out.loc[out.index[0], "projection"] = np.inf
        return out

    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest, predictor=_wrap(poison))
    assert_refused(conn, "G5", e.value, "finite")


def test_g5_refuses_a_cell_with_no_rows(manifest) -> None:
    def empty(stat, position, out):
        return out.iloc[0:0] if (stat, position) == ("targets", "TE") else out

    conn = make_conn()
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest, predictor=_wrap(empty))
    assert_refused(conn, "G5", e.value, "targets/TE")


def test_g5_refuses_row_counts_outside_the_bounds(manifest) -> None:
    frame = forward_frame()
    frame = frame[~((frame.position == "QB") & (frame.player_id.str[2:].astype(int) >= 10))]  # 10 QBs
    conn = make_conn(frame=frame)
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "G5", e.value, "QB")


# ── ordering: no write before the last guard ──────────────────────────────────────────────
def test_every_write_comes_after_every_read_and_every_guard(manifest) -> None:
    calls: list[str] = []

    def spy(stat, position, frame, manifest, **kw):
        calls.append(f"{stat}:{position}")
        return predict_cell(stat, position, frame, manifest, **kw)

    conn = make_conn()
    go(conn, manifest, predictor=spy)
    kinds = conn.kinds()
    first_write = next(i for i, entry in enumerate(conn.log) if entry in conn.writes)
    assert calls and all(e[0] != "commit" for e in conn.log[:first_write])
    reads_after = [e for e in conn.log[first_write:] if e[0] == "execute" and e[1].lstrip().upper().startswith("SELECT")]
    assert reads_after == []
    assert kinds.count("commit") == 1 and kinds[-1] == "commit"


# ── the write ─────────────────────────────────────────────────────────────────────────────
def test_one_transaction_with_a_unique_forecast_run_and_all_rows(manifest, release: Release) -> None:
    conn = make_conn()
    report = go(conn, manifest)
    assert conn.commits == 1 and conn.kinds().count("rollback") == 0
    run_insert = next(w for w in conn.writes if w[0] == "execute" and "forecast_runs" in w[1])
    key = run_insert[2][0]
    assert key == "forward_projections_2026_w11_" + NOW.strftime("%Y%m%dT%H%M%SZ")
    summary = json.loads(run_insert[2][5])
    assert summary["feature_snapshot_id"] == SNAPSHOT_ID
    assert summary["code_git_sha"] == CODE_SHA
    assert summary["current_baseline_sha256"] == sha(release.manifest_path.read_bytes())
    assert summary["models_json_sha256"] == sha((release.root / release.meta_relpath).read_bytes())
    batch = next(w for w in conn.writes if w[0] == "batch" and "INSERT INTO projections" in w[1])
    assert "skip" not in batch[1] and "ON CONFLICT (player_id, game_id, stat)" in batch[1]
    assert "pipeline_run_id = ANY(" in batch[1]  # the approved-run guard
    rows = batch[2]
    assert {r["pipeline_run_id"] for r in rows} == {"weekly_auto_2026_pending"}
    assert {r["interval_method"] for r in rows} == {"causal_oof_conformal_90"}
    assert all(r["p25"] is None and r["p75"] is None for r in rows)
    assert {r["max_train_season"] for r in rows} == {2025}
    # one row per cell per rostered player, excluded players included
    expected = sum(POSITION_COUNTS[p] for (_, p) in CELLS)
    assert len(rows) == expected == report["projection_rows"]
    assert "availability_status" in batch[1]
    by_player = {r["player_id"]: r["availability_status"] for r in rows if r["stat"] == "receiving_yards"}
    assert by_player["WR000"] == "Out" and by_player["WR001"] == "Questionable" and by_player["WR002"] is None


def test_excluded_players_are_stored_but_counted_as_not_served(manifest) -> None:
    conn = make_conn()
    report = go(conn, manifest)
    assert report["availability"]["counts"] == {"Out": 1, "Questionable": 1, "Doubtful": 1, "Reserve": 1}
    n_cells_wr = sum(1 for (_, p) in CELLS if p == "WR")
    # WR000 (Out) and RB000 (Reserve, but no RB cell here) -> only WR000 rows are unserved
    assert report["rows_not_served"] == n_cells_wr
    assert report["rows_served"] == report["projection_rows"] - n_cells_wr


def test_forward_rows_match_the_stack_arithmetic(manifest, release: Release) -> None:
    conn = make_conn()
    go(conn, manifest)
    rows = next(w for w in conn.writes if w[0] == "batch" and "INSERT INTO projections" in w[1])[2]
    got = {r["player_id"]: r["projection"] for r in rows if r["stat"] == "targets"}
    frame = forward_frame()
    te = frame[frame.position == "TE"]
    X = te[FEATURE_COLS].fillna(MISSING_VALUE_SENTINEL)
    models = release.models_dir
    lg = lgb.Booster(model_file=str(models / "lgbm_targets_TE.txt")).predict(X)
    import catboost as cb

    cbm = cb.CatBoostRegressor()
    cbm.load_model(str(models / "catboost_targets_TE.cbm"))
    expected = 0.4 * lg + 0.6 * cbm.predict(X.values) + 0.25
    assert np.allclose([got[p] for p in te.player_id], expected, rtol=0, atol=1e-9)


def test_identity_cell_equals_the_lgbm_prediction_and_never_loads_catboost(manifest, release: Release) -> None:
    loaded: list[str] = []
    real_loader = forward_serving.load_booster

    def spy(learner, path):
        loaded.append(f"{learner}:{Path(path).name}")
        return real_loader(learner, path)

    frame = forward_frame()
    wr = frame[frame.position == "WR"].reset_index(drop=True)
    out = predict_cell("receiving_tds", "WR", wr, manifest, run_id="r", loader=spy)
    assert loaded == ["lgbm:lgbm_receiving_tds_WR.txt"]
    X = wr[FEATURE_COLS].fillna(MISSING_VALUE_SENTINEL)
    direct = lgb.Booster(model_file=str(release.models_dir / "lgbm_receiving_tds_WR.txt")).predict(X)
    assert np.array_equal(out["projection"].to_numpy(), direct)


def test_library_version_mismatch_is_refused_when_the_model_loads(release: Release, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(release.root, root)
    r = Release(root, root / "releases/current_baseline.json", root / "releases/candidates" / RELEASE / "models",
                release.meta_relpath)

    def bump(meta):
        for entry in meta["models"]:
            if entry["learner"] == "lgbm" and entry["stat"] == "passing_yards":
                entry["library_versions"]["lightgbm"] = "0.0.1"

    r.rewrite_meta(bump)
    m = load_manifest(r.manifest_path, root=root)
    conn = make_conn()
    with pytest.raises(forward_serving.LibraryVersionMismatch, match="lightgbm"):
        go(conn, m)
    assert conn.writes == []


# ── availability ──────────────────────────────────────────────────────────────────────────
def test_availability_snapshot_after_as_of_is_not_used(manifest) -> None:
    conn = make_conn(avail_fetched=AS_OF + timedelta(minutes=1))
    with pytest.raises(ForwardGuardError) as e:
        go(conn, manifest)
    assert_refused(conn, "D3", e.value, "availability")


def test_unknown_availability_can_be_allowed_explicitly(manifest, caplog) -> None:
    """Coordinator answer 2: manual override, logged, and every row says "availability unknown"."""
    caplog.set_level("WARNING")
    conn = make_conn(avail_fetched=AS_OF + timedelta(minutes=1))
    report = go(conn, manifest, allow_no_availability=True)
    assert report["availability"]["snapshot_id"] is None
    assert report["availability"]["override"] is True
    rows = next(w for w in conn.writes if w[0] == "batch" and "INSERT INTO projections" in w[1])[2]
    assert {r["availability_status"] for r in rows} == {"Unknown"}
    assert any("availability NOT applied" in r.getMessage() for r in caplog.records)
    insert = next(w for w in conn.writes if w[0] == "execute" and "INSERT INTO forecast_runs" in w[1])
    assert json.loads(insert[2][5])["availability_override"] is True


def test_the_unknown_marker_is_served_not_excluded() -> None:
    from pipeline.availability import EXCLUDED_STATUSES, STORED_STATUSES, UNKNOWN, is_served

    assert UNKNOWN == "Unknown" and UNKNOWN in STORED_STATUSES
    assert UNKNOWN not in EXCLUDED_STATUSES and is_served(UNKNOWN)


# ── coordinator answers 7 and 8 ───────────────────────────────────────────────────────────
def _delete(conn):
    return [i for i, w in enumerate(conn.log) if w[0] == "execute" and w[1].lstrip().upper().startswith("DELETE")]


def test_replace_week_deletes_and_inserts_in_one_transaction(manifest) -> None:
    conn = make_conn()
    report = go(conn, manifest, replace_week=True)
    deletes = _delete(conn)
    assert len(deletes) == 1
    delete_sql, params = conn.log[deletes[0]][1:3]
    assert "FROM projections" in delete_sql and params == (SEASON, WEEK)
    insert_at = next(i for i, w in enumerate(conn.log) if w[0] == "batch" and "INSERT INTO projections" in w[1])
    assert deletes[0] < insert_at
    assert conn.commits == 1 and ("rollback",) not in conn.log
    assert "rows_replaced" in report


def test_a_failed_insert_after_the_delete_rolls_everything_back(manifest, monkeypatch) -> None:
    def boom(conn, df, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(forward_serving, "upsert_projections", boom)
    conn = make_conn()
    with pytest.raises(RuntimeError, match="disk full"):
        go(conn, manifest, replace_week=True)
    assert len(_delete(conn)) == 1
    assert conn.commits == 0 and ("rollback",) in conn.log  # the old week is still served


def test_without_replace_week_nothing_is_deleted(manifest) -> None:
    conn = make_conn()
    go(conn, manifest)
    assert _delete(conn) == []


def test_a_refused_guard_never_reaches_the_replace_delete(manifest) -> None:
    conn = make_conn(target_scored=True)
    with pytest.raises(ForwardGuardError):
        go(conn, manifest, replace_week=True)
    assert _delete(conn) == [] and conn.writes == []


def test_a_real_write_refuses_an_unknown_code_sha(manifest, monkeypatch) -> None:
    monkeypatch.setattr(forward_serving, "_code_sha", lambda: "unknown")
    conn = make_conn()
    with pytest.raises(ForwardGuardError, match="code_git_sha") as e:
        go(conn, manifest, code_sha=None)
    assert_refused(conn, "D8", e.value)


def test_a_dry_run_may_report_an_unknown_code_sha(manifest, monkeypatch) -> None:
    monkeypatch.setattr(forward_serving, "_code_sha", lambda: "unknown")
    report = go(make_conn(), manifest, code_sha=None, dry_run=True)
    assert report["dry_run"] is True


GOOD_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _no_git(monkeypatch) -> None:
    monkeypatch.setattr(forward_serving.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))


def test_the_baked_file_is_the_code_sha_and_no_env_value_can_override_it(tmp_path, monkeypatch) -> None:
    """env_file .env overrides image ENV at runtime, so the baked value lives in a file (review M3)."""
    sha_file = tmp_path / "CODE_GIT_SHA"
    sha_file.write_text(GOOD_SHA)
    monkeypatch.setenv("CODE_GIT_SHA", OTHER_SHA)
    monkeypatch.setenv("GIT_SHA", OTHER_SHA)
    monkeypatch.setattr(forward_serving, "CODE_SHA_FILE", sha_file)
    assert forward_serving._code_sha() == GOOD_SHA


@pytest.mark.parametrize("content", ["", "unknown", "latest", "abc123", GOOD_SHA.upper(), GOOD_SHA + "0", "main\n"])
def test_a_bad_baked_sha_is_unknown_even_when_the_env_has_a_good_one(tmp_path, monkeypatch, content) -> None:
    sha_file = tmp_path / "CODE_GIT_SHA"
    sha_file.write_text(content)
    monkeypatch.setenv("CODE_GIT_SHA", OTHER_SHA)
    monkeypatch.setattr(forward_serving, "CODE_SHA_FILE", sha_file)
    _no_git(monkeypatch)
    assert forward_serving._code_sha() == "unknown"


def test_without_a_baked_file_a_valid_env_sha_is_used_and_an_invalid_one_is_not(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forward_serving, "CODE_SHA_FILE", tmp_path / "absent")
    _no_git(monkeypatch)
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setenv("CODE_GIT_SHA", GOOD_SHA)
    assert forward_serving._code_sha() == GOOD_SHA
    for bad in ("", "unknown", "latest", "abc"):
        monkeypatch.setenv("CODE_GIT_SHA", bad)
        assert forward_serving._code_sha() == "unknown"


def test_the_git_fallback_refuses_a_dirty_tree(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forward_serving, "CODE_SHA_FILE", tmp_path / "absent")
    monkeypatch.delenv("CODE_GIT_SHA", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)

    def fake_run(cmd, **kw):
        out = {"rev-parse": GOOD_SHA + "\n", "status": " M ml/forward_serving.py\n"}[cmd[1]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(forward_serving.subprocess, "run", fake_run)
    assert forward_serving._code_sha() == "unknown"

    def clean_run(cmd, **kw):
        out = {"rev-parse": GOOD_SHA + "\n", "status": ""}[cmd[1]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(forward_serving.subprocess, "run", clean_run)
    assert forward_serving._code_sha() == GOOD_SHA


@pytest.mark.parametrize("bad", ["abc123", "latest", "main", GOOD_SHA[:39], GOOD_SHA.upper()])
def test_a_real_write_refuses_a_code_sha_that_is_not_40_hex(manifest, bad) -> None:
    conn = make_conn()
    with pytest.raises(ForwardGuardError, match="code_git_sha") as e:
        go(conn, manifest, code_sha=bad)
    assert_refused(conn, "D8", e.value)


def test_the_image_bakes_the_commit_sha_into_a_file_not_an_env_var() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "backend/Dockerfile").read_text()
    assert "ARG CODE_GIT_SHA" in dockerfile
    assert "> /app/CODE_GIT_SHA" in dockerfile
    assert "ENV CODE_GIT_SHA" not in dockerfile, "an ENV can be shadowed by env_file at runtime"
    compose = (root / "infra/hosts/vps1/compose.yml").read_text()
    # etl and training builds read their own .env variable. GIT_COMMIT_SHA stays pinned to the freeze
    # commit, so deriving the code sha from it would record false provenance after a later sync.
    assert compose.count("CODE_GIT_SHA: ${CODE_GIT_SHA:-}") >= 2
    assert "CODE_GIT_SHA: ${GIT_COMMIT_SHA" not in compose


def test_report_freshness_is_logged(manifest, caplog) -> None:
    caplog.set_level("INFO")
    go(make_conn(), manifest)
    assert any("newest report" in r.getMessage() for r in caplog.records)


# ── bookkeeping ───────────────────────────────────────────────────────────────────────────
def test_dry_run_computes_everything_and_writes_nothing(manifest) -> None:
    conn = make_conn()
    report = go(conn, manifest, dry_run=True)
    assert report["dry_run"] is True and report["projection_rows"] > 0
    assert conn.writes == [] and conn.commits == 0


def test_memory_is_logged_per_cell(manifest) -> None:
    report = go(make_conn(), manifest, dry_run=True)
    assert report["memory"]["peak_rss_mb"] > 0
    cells = {c["cell"] for c in report["memory"]["cells"]}
    assert cells == {f"{s}:{p}" for (s, p) in CELLS}
    assert all({"rss_before_mb", "rss_after_mb"} <= set(c) for c in report["memory"]["cells"])


def test_a_second_run_gets_its_own_forecast_run_key(manifest) -> None:
    first, second = make_conn(), make_conn()
    go(first, manifest)
    go(second, manifest, now=lambda: NOW + timedelta(seconds=5))
    keys = [next(w for w in c.writes if "forecast_runs" in w[1])[2][0] for c in (first, second)]
    assert keys[0] != keys[1]


def test_module_imports_no_bayesian_tracking_or_simulation_stack() -> None:
    code = (
        "import sys, ml.forward_serving\n"
        "bad = [m for m in ('pymc','pytensor','mlflow','torch','shap','onnxruntime','ml.monte_carlo',"
        "'ml.bayesian_layer','ml.train') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


# ── L1: rows skipped because another approved run serves them are counted, and are an error ──
def _skip_route(n):
    return ("-- sr:skipped", (["n"], [(n,)]))


def test_the_forward_write_counts_the_rows_it_skipped_and_refuses_when_any(manifest) -> None:
    conn = make_conn()
    conn.routes.append(_skip_route(3))
    with pytest.raises(ForwardGuardError, match="skipped") as e:
        go(conn, manifest)
    assert e.value.guard == "L1"
    assert ("rollback",) in conn.log and conn.commits == 0, "the forecast_runs row must roll back with the refusal"
    assert not any(e[0] == "batch" for e in conn.log), "nothing may be upserted once rows would be skipped"


def test_the_forward_write_reports_rows_written_and_zero_skipped(manifest) -> None:
    conn = make_conn()
    conn.routes.append(_skip_route(0))
    report = go(conn, manifest)
    assert report["rows_skipped_approved"] == 0
    assert report["rows_written"] == report["projection_rows"]
    insert = next(e for e in conn.log if e[0] == "execute" and "INSERT INTO forecast_runs" in e[1])
    summary = json.loads(insert[2][5])
    assert summary["rows_written"] == report["projection_rows"] and summary["rows_skipped_approved"] == 0


def test_the_skip_count_is_scoped_to_the_week_and_checks_only_approved_other_runs(manifest) -> None:
    conn = make_conn()
    conn.routes.append(_skip_route(0))
    go(conn, manifest)
    q = next(e for e in conn.log if e[0] == "execute" and "-- sr:skipped" in e[1])
    assert "season = %s AND p.week = %s" in q[1] or "p.season = %s AND p.week = %s" in q[1]
    assert "IS DISTINCT FROM" in q[1] and "= ANY(" in q[1]


# ── L2: one writer per (season, week) ─────────────────────────────────────────────────────
def test_the_producer_takes_the_week_lock_before_any_write(manifest) -> None:
    conn = make_conn()
    go(conn, manifest, replace_week=True)
    statements = [e[1] for e in conn.log if e[0] == "execute"]
    lock_at = next(i for i, q in enumerate(statements) if "-- wk:lock" in q)
    first_write = next(i for i, q in enumerate(statements) if q.lstrip().upper().startswith(("DELETE", "INSERT")))
    assert lock_at < first_write
    assert "pg_try_advisory_xact_lock" in statements[lock_at]
    params = next(e[2] for e in conn.log if e[0] == "execute" and "-- wk:lock" in e[1])
    assert params[1] == SEASON * 100 + WEEK


def test_a_producer_that_cannot_get_the_week_lock_refuses_and_writes_nothing(manifest) -> None:
    conn = make_conn(lock=False)
    with pytest.raises(ForwardGuardError, match="another writer") as e:
        go(conn, manifest, replace_week=True)
    assert e.value.guard == "L2"
    assert conn.writes == []


def test_a_dry_run_does_not_take_the_week_lock(manifest) -> None:
    conn = make_conn()
    go(conn, manifest, dry_run=True)
    assert not any("-- wk:lock" in e[1] for e in conn.log if e[0] == "execute")
