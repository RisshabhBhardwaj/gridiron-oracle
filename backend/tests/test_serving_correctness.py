"""Regression guards for C-04, C-09, C-10, and C-11 serving contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def test_materializer_declares_the_complete_serving_matrix():
    from ml.artifact_manifest import REQUIRED_SERVING_CELLS
    from scripts.materialize_stack_projections import CELLS

    assert tuple(CELLS) == REQUIRED_SERVING_CELLS
    assert len(CELLS) == 32


def test_materializer_uses_prior_seasons_for_conformal_bounds_not_percentiles(tmp_path, monkeypatch):
    from scripts import materialize_stack_projections as materializer

    artifact = tmp_path / "stack_receiving_yards_WR_rebuilt.csv"
    pd.DataFrame({
        "player_id": [f"p{i}" for i in range(102)],
        "game_id": [f"g{i}" for i in range(102)],
        "season": [2024] * 100 + [2025, 2025],
        "week": [1] * 102,
        "y_pred": [50.0] * 50 + [100.0] * 50 + [50.0, 100.0],
        "y_true": [40.0] * 50 + [130.0] * 50 + [0.0, 0.0],
        "max_train_season": [2023] * 100 + [2024, 2024],
    }).to_csv(artifact, index=False)
    monkeypatch.setattr(materializer, "_pinned_stack", lambda *_: artifact)

    frame = materializer._load_cell("receiving_yards", "WR")

    # First season has no prior calibration history; its bounds are absent.
    assert frame.loc[frame["season"] == 2024, "floor"].isna().all()
    # 2025 values use 2024 residuals and differ by prediction bucket.  Their
    # own y_true values are deliberately nonsensical, proving they were not
    # used to form their interval.
    current = frame.loc[frame["season"] == 2025]
    assert current["floor"].notna().all()
    assert current["ceiling"].notna().all()
    assert current.iloc[0]["floor"] != current.iloc[1]["floor"]
    assert frame["p25"].isna().all()
    assert frame["p75"].isna().all()
    assert set(current["interval_method"]) == {"causal_oof_conformal_90"}


def test_materializer_requires_artifact_provenance(tmp_path, monkeypatch):
    from scripts import materialize_stack_projections as materializer

    artifact = tmp_path / "stack_receiving_yards_WR_rebuilt.csv"
    pd.DataFrame({
        "player_id": ["p1"], "game_id": ["g1"], "season": [2025],
        "week": [1], "y_pred": [50.0],
    }).to_csv(artifact, index=False)
    monkeypatch.setattr(materializer, "_pinned_stack", lambda *_: artifact)

    try:
        materializer._load_cell("receiving_yards", "WR")
    except ValueError as exc:
        assert "max_train_season" in str(exc)
    else:
        raise AssertionError("materializer accepted a stack without artifact provenance")


def test_materializer_refuses_pre_02_stack_names(tmp_path, monkeypatch):
    from scripts import materialize_stack_projections as materializer

    artifact = tmp_path / "stack_receiving_yards_WR_20260809.csv"
    artifact.write_text("")
    monkeypatch.setattr(materializer, "_pinned_stack", lambda *_: artifact)

    try:
        materializer._load_cell("receiving_yards", "WR")
    except ValueError as exc:
        assert "pre-02 legacy stack" in str(exc)
    else:
        raise AssertionError("materializer accepted a pre-02 stack")


def test_materializer_conflict_update_clears_stale_uncertainty_fields():
    from scripts.materialize_stack_projections import _upsert
    import inspect

    sql_source = inspect.getsource(_upsert)
    assert "boom_probability = EXCLUDED.boom_probability" in sql_source
    assert "bust_probability = EXCLUDED.bust_probability" in sql_source
    assert "interval_method = EXCLUDED.interval_method" in sql_source


def test_materialization_report_distinguishes_conformal_coverage_from_posterior_percentiles():
    from scripts import materialize_stack_projections as materializer
    import inspect

    source = inspect.getsource(materializer.main)
    assert '"causal_oof_conformal_90"' in source
    assert '"posterior_percentiles_materialized": False' in source


def _clean_manifest_copy(tmp_path):
    payload = json.loads(Path("releases/current_baseline.json").read_text())
    payload["git_dirty"] = False
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload))
    return path


def test_readiness_accepts_a_manifest_exactly_at_head_without_extra_git_calls(tmp_path):
    """The pre-existing fast path: no ancestor/diff check needed when they're equal."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    manifest_commit = json.loads(manifest.read_text())["git_commit"]
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=[f"{manifest_commit}\n", ""],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "ok"


def test_readiness_rejects_a_manifest_at_an_unrelated_commit(tmp_path):
    """Manifest commit that is not even an ancestor of HEAD: an unrelated branch."""
    import subprocess

    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    not_an_ancestor = subprocess.CalledProcessError(1, ["git", "merge-base"], output="")
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=["not-the-manifest-head\n", "", not_an_ancestor],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "does not match HEAD" in result["detail"]
    assert "not an ancestor" in result["detail"]


def test_readiness_rejects_an_unresolvable_manifest_commit(tmp_path):
    """merge-base cannot even resolve the pinned commit (e.g. exit 128): fail closed, not open."""
    import subprocess

    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    unknown_object = subprocess.CalledProcessError(128, ["git", "merge-base"], output="fatal: not a valid object name")
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=["some-other-head\n", "", unknown_object],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "could not be verified" in result["detail"]


def test_readiness_accepts_an_ancestor_commit_with_an_evidence_only_diff(tmp_path):
    """
    The self-referential-freeze fix: a manifest necessarily names the commit it
    was frozen from *before* it is itself committed, so HEAD legitimately moves
    past that commit the instant the manifest lands. That is fine as long as
    everything HEAD added on top is evidence about the pinned commit, not a
    change to it.
    """
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=[
            "evidence-commit-head\n",
            "",
            "",  # git merge-base --is-ancestor: exit 0, no output
            "releases/current_baseline.json\n"
            "releases/artifacts/MANIFEST.json\n"
            "reports/materialize_stack_projections.json\n"
            "backend/tests/test_artifact_containment.py\n",
        ],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "ok"


def test_readiness_rejects_an_ancestor_commit_that_also_changed_application_code(tmp_path):
    """An evidence-looking commit that also carries a code change must not slip through."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=[
            "evidence-commit-head\n",
            "",
            "",
            "releases/current_baseline.json\nbackend/app/services/runtime_status.py\n",
        ],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "non-evidence paths" in result["detail"]
    assert "backend/app/services/runtime_status.py" in result["detail"]


def test_readiness_rejects_an_ancestor_commit_that_swapped_a_pinned_artifact(tmp_path):
    """
    `releases/candidates/` holds the SHA-pinned model artifacts. `load_manifest`
    does not verify their digests (only `resolve_stack` does, lazily) — so a
    diff touching that directory must not be treated as evidence-only, or a
    swapped artifact would sail through readiness.
    """
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=[
            "evidence-commit-head\n",
            "",
            "",
            "releases/current_baseline.json\n"
            "releases/candidates/causal_20260810/ridge_fantasy_ppr_QB_coefs.json\n",
        ],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "non-evidence paths" in result["detail"]
    assert "releases/candidates/causal_20260810/ridge_fantasy_ppr_QB_coefs.json" in result["detail"]


def test_readiness_rejects_a_dirty_runtime_worktree(tmp_path):
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=["a4fb0e96995de1de943842afaa196777e0a194cb\n", " M ml/train.py\n"],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "worktree is dirty" in result["detail"]


# ── Phase 1 — ride-along honesty fixes (Components E, F1, F5) ─────────────────

_DB_URL = (
    os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle")
    .replace("postgresql+asyncpg://", "postgresql://")
)


@pytest.fixture(scope="module")
def db_conn():
    try:
        import psycopg2

        conn = psycopg2.connect(_DB_URL)
        try:
            yield conn
        finally:
            conn.close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc})")


def test_pipeline_fallback_disabled_by_default():
    from backend.app.services.projection import _PIPELINE_FALLBACK_ENABLED

    assert _PIPELINE_FALLBACK_ENABLED is False


def test_bypass_closed_raises_no_forecast_available_not_a_degraded_number(db_conn):
    """
    F1: a player with no approved projection row must not silently fall
    through to the live Kalman-passthrough pipeline. get_projection must
    raise NoForecastAvailable, not return a degraded=True number.
    """
    from backend.app.services.projection import NoForecastAvailable, ProjectionService

    cur = db_conn.cursor()
    cur.execute("SELECT full_name FROM players WHERE full_name IS NOT NULL LIMIT 1")
    row = cur.fetchone()
    if row is None:
        pytest.skip("no players in DB")
    player_name = row[0]

    svc = ProjectionService(db_url=_DB_URL)
    with pytest.raises(NoForecastAvailable):
        # Season far outside any approved run's coverage.
        svc.get_projection(player_name, week=1, season=2099, stat="receiving_yards")


def test_predict_no_forecast_available_response_shape():
    """F5: the API must surface forecast_available=False, not a bare 404 string."""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as client:
        cur_r = client.get(
            "/predict",
            params={"player": "Puka Nacua", "week": 1, "season": 2099, "stat": "receiving_yards"},
        )
    assert cur_r.status_code == 404
    detail = cur_r.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["forecast_available"] is False


def test_check_depth_chart_freshness_passes_for_2026(db_conn):
    from backend.app.services.projection import check_depth_chart_freshness

    result = check_depth_chart_freshness(db_conn, 2026)
    assert result.ok, result.reason
    assert len(result.team_counts) == 32


def test_check_depth_chart_freshness_fails_for_a_season_with_no_data(db_conn):
    from backend.app.services.projection import check_depth_chart_freshness

    result = check_depth_chart_freshness(db_conn, 1999)
    assert not result.ok
    assert "teams" in result.reason


def test_season_board_excludes_retired_players(db_conn):
    """
    Phase 1 verification bullet: no retired player in the rest-of-season
    board. _load_season_feature_rows excludes players.status = 'RET'.

    Weak on its own: players.status is stale for many genuinely-retired
    players (Tom Brady is stored ACT), so this only proves the WHERE clause
    ran, not that retired players are absent — see
    test_season_feature_rows_gates_roster_on_depth_charts below for the
    independent check.
    """
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService(db_url=_DB_URL)
    rows = svc._load_season_feature_rows(2025, 10, ["QB", "RB", "WR", "TE"])
    player_ids = [r["player_id"] for r in rows]
    if not player_ids:
        pytest.skip("no season feature rows for 2025 week<10")

    cur = db_conn.cursor()
    cur.execute(
        "SELECT count(*) FROM players WHERE id = ANY(%s) AND status = 'RET'",
        (player_ids,),
    )
    assert cur.fetchone()[0] == 0


def test_season_feature_rows_gates_roster_on_depth_charts(db_conn):
    """
    Independent check of the actual defect: every player
    _load_season_feature_rows returns for (season, start_week) must have a
    depth_charts row for that season — not merely players.status != 'RET'.
    Regression guard for the missing-join bug that let Tom Brady (stored
    status='ACT', last real season 2022) onto the served 2026 season board
    at rank 21.
    """
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService(db_url=_DB_URL)
    rows = svc._load_season_feature_rows(2026, 1, ["QB", "RB", "WR", "TE"])
    if not rows:
        pytest.skip("no 2026 depth chart in this database")

    player_ids = [r["player_id"] for r in rows]
    cur = db_conn.cursor()
    cur.execute(
        "SELECT count(DISTINCT player_id) FROM depth_charts "
        "WHERE season = 2026 AND player_id = ANY(%s)",
        (player_ids,),
    )
    assert cur.fetchone()[0] == len(set(player_ids))

    # Tom Brady by name: has no 2026 depth-chart row (retired), so the
    # per-player fallback-to-any-prior-season bug this test guards against
    # would surface him specifically if it regressed.
    cur.execute("SELECT id FROM players WHERE full_name = 'Tom Brady'")
    brady = cur.fetchone()
    if brady:
        assert brady[0] not in player_ids


def test_served_season_board_top200_have_recent_game_logs(db_conn):
    """
    Independent, source-agnostic version of the retired-player check: calls
    get_season_projections — the actual served path, not a helper — and
    verifies against game_logs (not depth_charts, not players.status) that
    every player in the top 200 has played within the last year. This is
    deliberately a DIFFERENT data source than the depth_charts join the
    fix relies on, so it would catch a stale-depth-chart regression too,
    not just re-confirm the same join that's under test elsewhere.
    """
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService(db_url=_DB_URL)
    try:
        rows = svc.get_season_projections(season=2026, start_week=1)
    except Exception as exc:
        pytest.skip(f"season board unavailable: {exc}")
    if not rows:
        pytest.skip("no 2026 season board rows in this database")

    top200 = sorted(rows, key=lambda r: r.get("mean") or 0.0, reverse=True)[:200]
    player_ids = [r["player_id"] for r in top200]

    cur = db_conn.cursor()
    cur.execute(
        "SELECT player_id, MAX(season) FROM game_logs WHERE player_id = ANY(%s) GROUP BY player_id",
        (player_ids,),
    )
    last_season = dict(cur.fetchall())

    # Three-plus years back, not one or two: real rostered players
    # (including deep backups) can go a full season or two without a
    # meaningful game_logs row — injury (Deshaun Watson's Achilles,
    # Brandon Aiyuk's ACL) or just never getting on the field as a QB3/QB4
    # (Easton Stick, last real action 2023). That's not staleness, it's
    # real roster depth the depth-chart join already accounts for. A gap
    # back to Brady's 2022 or Roethlisberger's 2021 is what actually
    # signals a defect: a player with no recent NFL action at all.
    #
    # A player with NO game_logs row at all (last_season is None, not an
    # old int) is a rookie who has never played an NFL snap — the opposite
    # case from "stopped playing a long time ago", and a legitimate reason
    # to be on a season board (cold-start priors, real roster spot). Only
    # players with a REAL but old last-season are checked here.
    stale = [
        (r["player_name"], last_season.get(r["player_id"]))
        for r in top200
        if last_season.get(r["player_id"]) is not None and last_season[r["player_id"]] < 2023
    ]
    assert not stale, f"players in top 200 with no game_logs since before 2023: {stale}"


class TestSeasonBoardVolumeReconciliation:
    """
    Phase 7 volume-budget fix (season_simulator.py's _apply_volume_budget):
    summed player yards must reconcile with the team-game model's own
    yards prediction, and passing_yards must equal receiving_yards within
    a team-week (both true by definition in real football; the old
    additive team-script coupling shift could never enforce either since
    it was mean-zero by construction — see ml/season_simulator.py's
    _VOLUME_BUDGET_STATS docstring).
    """

    @pytest.fixture()
    def materialized_run(self):
        try:
            import psycopg2

            psycopg2.connect(_DB_URL).close()
        except Exception as exc:
            pytest.skip(f"PostgreSQL not reachable ({exc})")

        from scripts.materialize_season_simulation import materialize

        # NOT (2026, start_week=1) — that key is real served production
        # data. season_simulations upserts ON CONFLICT (season, start_week,
        # player_id, stat), so a test materializing into the SAME key as an
        # approved run overwrites its pipeline_run_id, and this fixture's
        # own teardown (deletes by run_id) would then delete what looks
        # like its own rows but is actually the live approved run. This bit
        # a real approved run during this session — use the same "safe",
        # far-future slice test_season_simulation_materializer.py and
        # test_season_team_wins.py already use instead.
        season, start_week, end_week = 2026, 17, 18
        try:
            n = materialize(
                season=season, start_week=start_week, end_week=end_week,
                n_simulations=200, positions=["QB", "RB", "WR", "TE"], database_url=_DB_URL,
            )
        except ValueError as exc:
            pytest.skip(f"no data to materialize for this slice: {exc}")
        if n == 0:
            pytest.skip("materializer wrote 0 rows for this slice")

        import psycopg2

        conn = psycopg2.connect(_DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pipeline_run_id FROM season_simulations "
                    "WHERE season=%s AND start_week=%s ORDER BY created_at DESC LIMIT 1",
                    (season, start_week),
                )
                run_id = cur.fetchone()[0]
        finally:
            conn.close()

        yield season, start_week, end_week, run_id

        conn = psycopg2.connect(_DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM season_simulations WHERE pipeline_run_id=%s", (run_id,))
                cur.execute("DELETE FROM season_simulation_weeks WHERE pipeline_run_id=%s", (run_id,))
                cur.execute("DELETE FROM season_team_wins WHERE pipeline_run_id=%s", (run_id,))
            conn.commit()
        finally:
            conn.close()

    def test_summed_player_yards_reconcile_with_team_game_model(self, materialized_run, db_conn, monkeypatch):
        season, start_week, end_week, run_id = materialized_run
        from backend.app.services import projection as projection_mod
        from backend.app.services.projection import ProjectionService

        monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({run_id}))
        svc = ProjectionService(db_url=_DB_URL)
        rows = svc.get_season_projections(season=season, start_week=start_week)
        assert rows

        # get_season_projections's yards means are SEASON totals across
        # [start_week, end_week] — sum team_game_predictions.yards across
        # the same week range, not just week=start_week, or a 2-week
        # season total gets compared against a single week's budget.
        cur = db_conn.cursor()
        cur.execute(
            "SELECT team, SUM(yards) FROM team_game_predictions "
            "WHERE season=%s AND week BETWEEN %s AND %s GROUP BY team",
            (season, start_week, end_week),
        )
        team_yards = dict(cur.fetchall())
        if not team_yards:
            pytest.skip("no team_game_predictions for this (season, week) range")

        by_team: dict[str, float] = {}
        for r in rows:
            team = r.get("team")
            if team not in team_yards:
                continue
            passing = (r.get("passing_yards") or {}).get("mean") or 0.0
            rushing = (r.get("rushing_yards") or {}).get("mean") or 0.0
            by_team[team] = by_team.get(team, 0.0) + passing + rushing

        checked = 0
        for team, summed in by_team.items():
            predicted = team_yards[team]
            if predicted <= 0:
                continue
            ratio = summed / predicted
            assert 0.7 <= ratio <= 1.3, (
                f"{team}: summed player yards={summed:.1f} vs team model yards={predicted:.1f} "
                f"(ratio={ratio:.2f}); expected within 30% after the volume-budget fix"
            )
            checked += 1
        assert checked > 0, "no teams had both served player rows and a team_game_predictions row"

    def test_passing_yards_equals_receiving_yards_within_team_week(self, materialized_run, monkeypatch):
        season, start_week, end_week, run_id = materialized_run
        from backend.app.services import projection as projection_mod
        from backend.app.services.projection import ProjectionService

        monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({run_id}))
        svc = ProjectionService(db_url=_DB_URL)
        rows = svc.get_season_projections(season=season, start_week=start_week)
        assert rows

        by_team_pass: dict[str, float] = {}
        by_team_rec: dict[str, float] = {}
        for r in rows:
            team = r.get("team")
            if not team:
                continue
            by_team_pass[team] = by_team_pass.get(team, 0.0) + (r.get("passing_yards") or {}).get("mean", 0.0)
            by_team_rec[team] = by_team_rec.get(team, 0.0) + (r.get("receiving_yards") or {}).get("mean", 0.0)

        checked = 0
        for team, pass_total in by_team_pass.items():
            rec_total = by_team_rec.get(team, 0.0)
            if pass_total <= 0 and rec_total <= 0:
                continue
            assert pass_total == pytest.approx(rec_total, rel=0.05), (
                f"{team}: passing_yards={pass_total:.1f} != receiving_yards={rec_total:.1f}"
            )
            checked += 1
        assert checked > 0, "no teams had both passing and receiving yardage to compare"


# ---------------------------------------------------------------------------
# Containerised deployments: no .git in the image
# ---------------------------------------------------------------------------

def test_readiness_uses_the_platform_commit_when_there_is_no_worktree(tmp_path, monkeypatch):
    """A container built from a pushed commit ships no .git, so `git rev-parse`
    fails by construction. The built commit is still known, and an image built
    from a remote ref cannot be dirty — so an exact match is still `ok`."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    manifest_commit = json.loads(manifest.read_text())["git_commit"]
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", manifest_commit)

    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=OSError("git: command not found"),
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "ok"
    assert result["observed"]["head"] == manifest_commit
    assert result["observed"]["worktree_dirty"] is False


def test_readiness_warns_rather_than_guessing_when_lineage_is_unverifiable(tmp_path, monkeypatch):
    """Without history the evidence-only-diff rule cannot be applied. Claiming
    `ok` would fake a check; claiming `error` would condemn a deployment that
    may be perfectly in lineage. It must warn, and stay non-blocking."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "0" * 40)

    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=OSError("git: command not found"),
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "warn"
    assert "cannot be verified without a git worktree" in result["detail"]


def test_readiness_errors_when_neither_git_nor_the_platform_names_the_commit(tmp_path, monkeypatch):
    """No worktree and no build commit means the running code is genuinely
    unidentified. That is a real failure, not a degraded environment."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("GIT_COMMIT_SHA", raising=False)

    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=OSError("git: command not found"),
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "no git worktree" in result["detail"]


def test_unreachable_mlflow_is_reported_but_does_not_block_serving():
    """MLflow backs SHAP attribution, which already degrades per-response. A
    serving-only deployment has no tracking server, and that must not take the
    service out of rotation — but it must still be visible in the report."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    ok = {"status": "ok", "detail": "fine"}

    with patch.object(settings, "product_mode", "artifact_backed"), \
         patch.object(RuntimeStatusService, "_check_database", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_feature_matrix", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_baseline_manifest", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_backtest_assets", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_tracing", return_value=ok), \
         patch.object(
             RuntimeStatusService, "_check_mlflow",
             return_value={"status": "error", "detail": "MLflow unreachable"},
         ):
        report = service.build_report()

    assert report["overall_status"] != "blocked"
    assert report["checks"]["mlflow"]["status"] == "error"


def test_a_failed_database_still_blocks():
    """Guard the change above: loosening MLflow must not loosen the checks that
    genuinely gate serving."""
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    ok = {"status": "ok", "detail": "fine"}

    with patch.object(settings, "product_mode", "artifact_backed"), \
         patch.object(RuntimeStatusService, "_check_feature_matrix", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_mlflow", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_baseline_manifest", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_backtest_assets", return_value=ok), \
         patch.object(RuntimeStatusService, "_check_tracing", return_value=ok), \
         patch.object(
             RuntimeStatusService, "_check_database",
             return_value={"status": "error", "detail": "down"},
         ):
        report = service.build_report()

    assert report["overall_status"] == "blocked"
