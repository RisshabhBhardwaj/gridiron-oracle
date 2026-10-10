"""
backend/tests/test_ml_cov_draft_adp.py

Coverage tests for:
  - ml/mock_draft.py  (pick engine: roster limits, timing gates, softmax; board builder)
  - ml/adp_eval.py    (Spearman agreement with ADP, DB loaders, evaluate, CLI)

Database access goes through an in-test psycopg2 fake. Rank and overlap
expectations are worked out by hand in the comments. Output files go to tmp_path.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml import adp_eval as ae
from ml import mock_draft as md
from ml.vor import FLEX_ELIGIBLE, STARTING_SLOTS, attach_vor


def _p(pid, pos, **extra) -> dict:
    row = {"player_id": pid, "position": pos, "blended_rank": None, "model_rank": None,
           "adp_rank": None, "adp": None}
    row.update(extra)
    return row


# ── mock_draft.pick ───────────────────────────────────────────────────────────

class _RecordingRng:
    """Stands in for numpy's Generator; returns a fixed index and records the probabilities."""

    def __init__(self, index: int = 0):
        self.index = index
        self.calls: list = []

    def choice(self, n, p):
        self.calls.append((n, np.asarray(p, dtype=float)))
        return self.index


PROFILE: dict = {}


class TestPick:
    def test_empty_pool_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty available player pool"):
            md.pick([], [], PROFILE, np.random.default_rng(0))

    def test_single_legal_candidate_is_returned_without_sampling(self) -> None:
        rng = _RecordingRng()
        only = _p("A", "WR", blended_rank=1)
        assert md.pick([only], [], PROFILE, rng) is only
        assert rng.calls == []

    def test_full_roster_position_is_skipped(self) -> None:
        max_wr = md.ROSTER_MAX_SLOTS["WR"] + STARTING_SLOTS["FLEX"]  # WR is flex eligible
        assert "WR" in FLEX_ELIGIBLE
        roster = [_p(f"W{i}", "WR") for i in range(max_wr)]
        available = [_p("WR1", "WR", blended_rank=1), _p("RB1", "RB", blended_rank=2)]
        assert md.pick(available, roster, PROFILE, np.random.default_rng(0))["player_id"] == "RB1"

    def test_all_positions_full_falls_back_to_best_ranked(self) -> None:
        max_qb = md.ROSTER_MAX_SLOTS["QB"]
        roster = [_p(f"Q{i}", "QB") for i in range(max_qb)]
        available = [_p("QB9", "QB", blended_rank=5), _p("QB8", "QB", blended_rank=3)]
        # Two candidates remain, so the softmax runs; the recording RNG takes the first (best) one.
        chosen = md.pick(available, roster, PROFILE, _RecordingRng(index=0), pick_no=100)
        assert chosen["player_id"] == "QB8"  # best rank wins when nothing is legal

    def test_quarterback_gated_before_its_earliest_pick(self) -> None:
        available = [_p("QB1", "QB", blended_rank=1), _p("WR1", "WR", blended_rank=2)]
        chosen = md.pick(available, [], {"first_qb_pick_shrunk": 48.0}, np.random.default_rng(0), pick_no=5)
        assert chosen["player_id"] == "WR1"

    def test_quarterback_allowed_after_gate_and_sampled(self) -> None:
        available = [_p("QB1", "QB", blended_rank=1), _p("WR1", "WR", blended_rank=2)]
        rng = _RecordingRng(index=0)
        chosen = md.pick(available, [], {"first_qb_pick_shrunk": 48.0}, rng, pick_no=60)
        assert chosen["player_id"] == "QB1"
        n, probs = rng.calls[0]
        assert n == 2
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] > probs[1]

    def test_tight_end_gate_uses_its_own_threshold(self) -> None:
        available = [_p("TE1", "TE", blended_rank=1), _p("RB1", "RB", blended_rank=2)]
        chosen = md.pick(available, [], {"first_te_pick_shrunk": 64.0}, np.random.default_rng(0), pick_no=30)
        assert chosen["player_id"] == "RB1"

    def test_gate_is_ignored_when_only_that_position_remains(self) -> None:
        available = [_p("QB1", "QB", blended_rank=1), _p("QB2", "QB", blended_rank=2)]
        rng = _RecordingRng(index=1)
        chosen = md.pick(available, [], {}, rng, pick_no=5)
        assert chosen["player_id"] == "QB2"
        assert rng.calls[0][0] == 2

    def test_rank_key_falls_back_through_available_fields(self) -> None:
        # Keys: model_rank 5, adp_rank 3, adp 1.5 -> the adp-only player is ranked first.
        available = [
            _p("model_only", "RB", model_rank=5.0),
            _p("adp_rank_only", "RB", adp_rank=3.0),
            _p("adp_only", "RB", adp=1.5),
            _p("unranked", "RB"),
        ]
        ordered = []
        for _ in range(len(available)):
            pool = [p for p in available if p["player_id"] not in ordered]
            ordered.append(md.pick(pool, [], {}, np.random.default_rng(0), top_k=1)["player_id"])
        assert ordered == ["adp_only", "adp_rank_only", "model_only", "unranked"]

    def test_reach_shift_and_top_k_truncation(self) -> None:
        available = [_p(f"R{i}", "RB", blended_rank=i + 1) for i in range(12)]
        rng = _RecordingRng(index=0)
        profile = {"adp_delta_mean_shrunk": -10.0, "adp_delta_sd_shrunk": 6.0}
        md.pick(available, [], profile, rng, top_k=8)
        n, probs = rng.calls[0]
        assert n == 8  # top_k truncation
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] > probs[-1]


# ── mock_draft.build_mock_draft_pool ──────────────────────────────────────────

class _DraftCursor:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._state["sql"].append((sql, params))

    def fetchall(self):
        return self._state["results"].pop(0)


class _DraftConn:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, cursor_factory=None):
        return _DraftCursor(self._state)


@pytest.fixture
def draft_db(monkeypatch):
    import psycopg2

    state = {"sql": [], "results": [], "dsn": None}

    def connect(dsn):
        state["dsn"] = dsn
        return _DraftConn(state)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return state


class TestBuildMockDraftPool:
    def _adp_rows(self):
        # Already in ascending ADP order, as the SQL ORDER BY adp ASC returns them.
        return [
            {"player_name": "Star RB", "position": "RB", "team": "SF", "adp": 1.0, "player_id": "RB1", "source": "fantasy_adp"},
            {"player_name": "Star WR", "position": "WR", "team": "DAL", "adp": 5.0, "player_id": "WR1", "source": "fantasy_adp"},
            {"player_name": "Star QB", "position": "QB", "team": "KC", "adp": 20.0, "player_id": "QB1", "source": "fantasy_adp"},
        ]

    def _proj_rows(self):
        return [
            {"player_id": "RB1", "player_name": "Star RB", "position": "RB", "team": "SF", "fantasy_ppr": 280.0},
            {"player_id": "WR1", "player_name": "Star WR", "position": "WR", "team": "DAL", "fantasy_ppr": 0.0},
            {"player_id": "QB1", "player_name": "Star QB", "position": "QB", "team": "KC", "fantasy_ppr": 300.0},
            # Two players missing from ADP: they form the VOR-ordered tail.
            {"player_id": "WRX", "player_name": "Tail WR", "position": "WR", "team": "NYJ", "fantasy_ppr": 220.0},
            {"player_id": "TEX", "player_name": "Tail TE", "position": "TE", "team": "BUF", "fantasy_ppr": 120.0},
        ]

    def test_board_orders_adp_then_vor_tail(self, draft_db) -> None:
        draft_db["results"] = [self._adp_rows(), self._proj_rows()]
        board = md.build_mock_draft_pool("postgresql+asyncpg://u@h/db", season=2026, min_required_picks=3)

        assert draft_db["dsn"] == "postgresql://u@h/db"
        assert [b["player_id"] for b in board[:3]] == ["RB1", "WR1", "QB1"]
        assert [b["adp_rank"] for b in board[:3]] == [1, 2, 3]
        assert board[0]["board_tail"] is False
        assert board[0]["model_fantasy_ppr"] == pytest.approx(280.0)
        assert board[1]["model_fantasy_ppr"] is None  # projection 0.0 is reported as no model value
        assert board[2]["model_fantasy_ppr"] == pytest.approx(300.0)

        tail = board[3:]
        assert all(entry["board_tail"] for entry in tail)
        assert all(entry["source"] == "draft_preseason_projections" for entry in tail)
        assert [entry["model_rank"] for entry in tail] == [4, 5]
        # Tail order follows VOR, highest first (the same ordering attach_vor produces).
        expected = [
            item["player_id"]
            for item in sorted(
                attach_vor([
                    {"player_id": "WRX", "position": "WR", "projection": 220.0},
                    {"player_id": "TEX", "position": "TE", "projection": 120.0},
                ]),
                key=lambda it: (-(it["vor"] if it.get("vor") is not None else -1e18), it["player_id"]),
            )
        ]
        assert [entry["player_id"] for entry in tail] == expected

    def test_queries_use_season_and_scoring(self, draft_db) -> None:
        draft_db["results"] = [self._adp_rows(), self._proj_rows()]
        md.build_mock_draft_pool("postgresql://u@h/db", season=2026, scoring="half_ppr", min_required_picks=1)
        adp_sql, adp_params = draft_db["sql"][0]
        assert "FROM fantasy_adp" in adp_sql
        assert adp_params == (2026, "half_ppr")
        proj_sql, proj_params = draft_db["sql"][1]
        assert "draft_preseason_projections" in proj_sql
        assert proj_params == (2026, 2026)

    def test_pool_too_small_is_refused(self, draft_db) -> None:
        draft_db["results"] = [self._adp_rows(), self._proj_rows()]
        with pytest.raises(ValueError, match="Draft pool size \\(5\\) is insufficient for 5 picks"):
            md.build_mock_draft_pool("postgresql://u@h/db", min_required_picks=5)

    def test_no_projection_tail_when_every_player_has_adp(self, draft_db) -> None:
        draft_db["results"] = [self._adp_rows(), [
            {"player_id": "RB1", "player_name": "Star RB", "position": "RB", "team": "SF", "fantasy_ppr": 10.0},
        ]]
        board = md.build_mock_draft_pool("postgresql://u@h/db", min_required_picks=2)
        assert len(board) == 3
        assert not any(entry["board_tail"] for entry in board)


# ── adp_eval: names, Spearman agreement ───────────────────────────────────────

class TestAdpEvalHelpers:
    @pytest.mark.parametrize("raw, expected", [
        ("Patrick Mahomes II", "patrick mahomes"),
        ("D'Andre  Swift", "dandre swift"),
        ("Odell Beckham Jr.", "odell beckham"),
        ("", ""),
    ])
    def test_name_normalisation(self, raw, expected) -> None:
        assert ae._normalize_name(raw) == expected

    def test_finite_maps_nan_to_none(self) -> None:
        assert ae._finite(float("nan")) is None
        assert ae._finite(2) == 2.0


def _ranked_pair(n: int, reverse: bool = False):
    players = [f"P{i:02d}" for i in range(n)]
    model = pd.DataFrame({
        "player_id": players,
        "position": (["WR", "RB", "QB", "TE", "WR", "RB"] * n)[:n],
        "fantasy_ppr": [float(300 - 10 * i) for i in range(n)],
    })
    adp_order = list(range(n))[::-1] if reverse else list(range(n))
    adp = pd.DataFrame({
        "player_id": players,
        "position": model["position"],
        "adp": [float(rank + 1) for rank in adp_order],
    })
    return model, adp


class TestSpearmanVsAdp:
    def test_identical_ordering_has_rho_one(self) -> None:
        model, adp = _ranked_pair(6)
        rho, pvalue, n, by_position = ae.spearman_vs_adp(model, adp)
        assert rho == pytest.approx(1.0)
        assert n == 6
        assert math.isfinite(pvalue)
        assert by_position == {}  # no position reaches the five-player minimum

    def test_reversed_ordering_has_rho_minus_one(self) -> None:
        model, adp = _ranked_pair(6, reverse=True)
        rho, _, n, _ = ae.spearman_vs_adp(model, adp)
        assert rho == pytest.approx(-1.0)
        assert n == 6

    def test_position_breakdown_when_groups_are_large_enough(self) -> None:
        model = pd.DataFrame({
            "player_id": [f"W{i}" for i in range(6)],
            "position": ["WR"] * 6,
            "fantasy_ppr": [60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        })
        adp = pd.DataFrame({"player_id": model["player_id"], "position": "WR",
                            "adp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        _, _, _, by_position = ae.spearman_vs_adp(model, adp)
        assert set(by_position) == {"WR"}
        assert by_position["WR"]["n"] == 6
        assert by_position["WR"]["spearman_rho"] == pytest.approx(1.0)

    def test_name_key_is_used_for_legacy_frames(self) -> None:
        model = pd.DataFrame({"name_key": [f"n{i}" for i in range(5)], "position": "RB",
                              "fantasy_ppr": [5.0, 4.0, 3.0, 2.0, 1.0]})
        adp = pd.DataFrame({"name_key": [f"n{i}" for i in range(5)], "position": "RB",
                            "adp": [1.0, 2.0, 3.0, 4.0, 5.0]})
        rho, _, n, _ = ae.spearman_vs_adp(model, adp)
        assert rho == pytest.approx(1.0)
        assert n == 5

    def test_too_few_matches_is_nan(self) -> None:
        model, adp = _ranked_pair(4)
        rho, pvalue, n, by_position = ae.spearman_vs_adp(model, adp)
        assert math.isnan(rho) and math.isnan(pvalue)
        assert n == 4 and by_position == {}

    def test_empty_inputs_return_nan_and_zero(self) -> None:
        rho, pvalue, n, by_position = ae.spearman_vs_adp(pd.DataFrame(), pd.DataFrame(columns=["player_id"]))
        assert math.isnan(rho) and math.isnan(pvalue)
        assert n == 0 and by_position == {}


# ── adp_eval: DB loaders and evaluate ─────────────────────────────────────────

class _EvalCursor:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._state["sql"].append((sql, params))
        self._current = (sql, params)

    def fetchall(self):
        sql, params = self._current
        if "fantasy_adp" in sql:
            return self._state["adp"]
        if "draft_preseason_projections" in sql:
            return self._state["projections"]
        if "game_logs" in sql:
            return self._state["actuals"].get(params[0], [])
        raise AssertionError(f"unexpected query: {sql[:60]}")


class _EvalConn:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, cursor_factory=None):
        return _EvalCursor(self._state)


@pytest.fixture
def eval_db(monkeypatch):
    import psycopg2

    state = {"sql": [], "adp": [], "projections": [], "actuals": {}, "dsn": None}

    def connect(dsn):
        state["dsn"] = dsn
        return _EvalConn(state)

    monkeypatch.setattr(psycopg2, "connect", connect)
    return state


def _season_fixture(n: int = 30):
    players = [f"P{i:02d}" for i in range(n)]
    ppr = {pid: float(300 - 10 * i) for i, pid in enumerate(players)}
    adp = [{"player_id": pid, "player_name": pid, "position": "WR", "team": "KC", "adp": float(i + 1)}
           for i, pid in enumerate(players)]
    projections = [{"player_id": pid, "player_name": pid, "position": "WR", "team": "KC",
                    "fantasy_ppr": ppr[pid]} for pid in players]
    actual_same = [{"player_id": pid, "player_name": pid, "position": "WR", "team": "KC",
                    "fantasy_ppr": ppr[pid]} for pid in players]
    actual_prev = [{"player_id": pid, "player_name": pid, "position": "WR", "team": "KC",
                    "fantasy_ppr": float(10 * i)} for i, pid in enumerate(players)]
    return adp, projections, actual_same, actual_prev


class TestAdpLoaders:
    def test_load_adp_passes_season_source_and_scoring(self) -> None:
        seen: dict = {}

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params):
                seen["params"] = params

            def fetchall(self):
                return [{"player_id": "P1", "adp": 3.0}]

        class _Conn:
            def cursor(self, cursor_factory=None):
                return _Cur()

        df = ae.load_adp(_Conn(), 2026, source="consensus", scoring="half_ppr")
        assert seen["params"] == (2026, "consensus", "half_ppr")
        assert list(df["player_id"]) == ["P1"]

    def test_empty_result_gives_empty_frame(self, eval_db) -> None:
        conn = _EvalConn(eval_db)
        assert ae.load_season_projected_ppr(conn, 2026).empty
        assert ae.load_season_actual_ppr(conn, 2025).empty


class TestEvaluate:
    def test_projection_mode_known_agreement(self, eval_db) -> None:
        adp, projections, actual_same, actual_prev = _season_fixture(30)
        eval_db.update(adp=adp, projections=projections)
        eval_db["actuals"] = {2024: actual_same, 2023: actual_prev}

        result = ae.evaluate(2024, database_url="postgresql+asyncpg://u@h/db")

        assert eval_db["dsn"] == "postgresql://u@h/db"
        assert isinstance(result, ae.AdpEvalResult)
        assert result.mode == "preseason_projections"
        assert result.n_matched == 30
        assert result.spearman_rho == pytest.approx(1.0)
        assert result.actuals_oracle_rho == pytest.approx(1.0)
        assert result.prev_season_baseline_rho == pytest.approx(-1.0)
        assert result.gap_vs_actuals_oracle == pytest.approx(0.0)
        assert result.gap_vs_prev_season_baseline == pytest.approx(2.0)
        assert result.top24_hit_rate == pytest.approx(1.0)
        assert result.points_lost_vs_optimal == pytest.approx(0.0)
        assert result.created_at.endswith("+00:00")

    def test_actuals_oracle_mode_scores_realised_points(self, eval_db) -> None:
        adp, projections, actual_same, actual_prev = _season_fixture(30)
        eval_db.update(adp=adp, projections=projections)
        eval_db["actuals"] = {2024: actual_same, 2023: actual_prev}
        result = ae.evaluate(2024, from_actuals=True)
        assert result.mode == "actuals_oracle"
        assert result.spearman_rho == pytest.approx(1.0)

    def test_stack_oof_is_forbidden_before_any_query(self, eval_db) -> None:
        with pytest.raises(ValueError, match="Target-season stack OOF is forbidden"):
            ae.evaluate(2024, from_stack_oof=True)
        assert eval_db["sql"] == []

    def test_small_field_has_no_hit_rate(self, eval_db) -> None:
        adp, projections, actual_same, actual_prev = _season_fixture(10)
        eval_db.update(adp=adp, projections=projections)
        eval_db["actuals"] = {2024: actual_same, 2023: actual_prev}
        result = ae.evaluate(2024)
        assert result.n_matched == 10
        assert result.top24_hit_rate is None
        assert result.points_lost_vs_optimal is None

    def test_no_projections_yields_missing_values(self, eval_db) -> None:
        adp, _, actual_same, actual_prev = _season_fixture(8)
        eval_db.update(adp=adp, projections=[])
        eval_db["actuals"] = {2024: actual_same, 2023: actual_prev}
        result = ae.evaluate(2024)
        assert math.isnan(result.spearman_rho)
        assert result.gap_vs_actuals_oracle is None
        assert result.gap_vs_prev_season_baseline is None
        assert result.top24_hit_rate is None


# ── adp_eval.main ─────────────────────────────────────────────────────────────

def _result(**overrides) -> ae.AdpEvalResult:
    base = dict(
        season=2024, source="historical", scoring="ppr", mode="preseason_projections", n_matched=0,
        spearman_rho=float("nan"), spearman_pvalue=float("nan"), by_position={},
        actuals_oracle_rho=None, prev_season_baseline_rho=None,
        gap_vs_actuals_oracle=None, gap_vs_prev_season_baseline=None,
        created_at="2026-10-10T00:00:00+00:00",
    )
    base.update(overrides)
    return ae.AdpEvalResult(**base)


class TestMain:
    def test_main_writes_json_with_null_for_nan_correlation(self, monkeypatch, tmp_path, capsys) -> None:
        seen: dict = {}

        def fake_evaluate(*args):
            seen["args"] = args
            return _result()

        monkeypatch.setattr(ae, "evaluate", fake_evaluate)
        monkeypatch.setattr(ae, "_OUT", tmp_path / "adp")

        assert ae.main(["--season", "2024", "--from-actuals"]) == 0
        assert seen["args"] == (2024, "historical", "ppr", True, False, ae.DEFAULT_HOST_DATABASE_URL)

        written = tmp_path / "adp" / "adp_2024_historical_preseason_projections.json"
        payload = json.loads(written.read_text())
        assert payload["spearman_rho"] is None
        assert payload["spearman_pvalue"] is None
        assert payload["season"] == 2024
        assert '"season": 2024' in capsys.readouterr().out  # the CLI echoes the payload it saved

    def test_main_passes_database_and_source_options(self, monkeypatch, tmp_path) -> None:
        seen: dict = {}

        def fake_evaluate(*args):
            seen["args"] = args
            return _result(season=2025, source="consensus", spearman_rho=0.42)

        monkeypatch.setattr(ae, "evaluate", fake_evaluate)
        monkeypatch.setattr(ae, "_OUT", tmp_path)
        ae.main(["--season", "2025", "--source", "consensus", "--scoring", "half_ppr",
                 "--database-url", "postgresql://alt/db"])
        assert seen["args"] == (2025, "consensus", "half_ppr", False, False, "postgresql://alt/db")
        payload = json.loads(next(tmp_path.glob("adp_2025_consensus_*.json")).read_text())
        assert payload["spearman_rho"] == pytest.approx(0.42)

    def test_stack_oof_flag_is_rejected_by_evaluate(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ae, "_OUT", tmp_path)
        with pytest.raises(ValueError, match="forbidden"):
            ae.main(["--season", "2024", "--from-stack-oof"])
        assert list(tmp_path.iterdir()) == []

    def test_season_is_required(self) -> None:
        with pytest.raises(SystemExit):
            ae.main([])
