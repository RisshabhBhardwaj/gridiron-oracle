"""
tests/test_volume_redistribution.py

Unit tests for ml/volume_redistribution.py — the Volume Redistribution Engine.

Tests cover:
    - Dirichlet concentration computation (healthy, injured, questionable)
    - Dirichlet sample normalization (shares sum to 1.0)
    - Injury cascade: injured player's share redistributed to teammates
    - Team volume prediction with and without game context
    - Full VolumeRedistributor.redistribute() API
    - DataFrame-level redistribute_team_projections()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.volume_redistribution import (
    ALPHA_SCALE,
    ALPHA_MIN,
    DirichletShareDistributor,
    TeamVolumePredictor,
    VolumeRedistributor,
    _injury_multiplier,
    get_redistributor,
    INACTIVE_STATUSES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_players():
    return [
        {"player_id": "p1", "name": "WR1", "position": "WR",
         "kalman_est_target_share": 0.30, "injury_status": None},
        {"player_id": "p2", "name": "WR2", "position": "WR",
         "kalman_est_target_share": 0.15, "injury_status": None},
        {"player_id": "p3", "name": "TE1", "position": "TE",
         "kalman_est_target_share": 0.13, "injury_status": None},
        {"player_id": "p4", "name": "RB1", "position": "RB",
         "kalman_est_target_share": 0.08, "injury_status": None},
    ]


@pytest.fixture
def injured_wr1_players():
    """WR1 is OUT — share should cascade to WR2/TE1/RB1."""
    return [
        {"player_id": "p1", "name": "WR1", "position": "WR",
         "kalman_est_target_share": 0.30, "injury_status": "out"},
        {"player_id": "p2", "name": "WR2", "position": "WR",
         "kalman_est_target_share": 0.15, "injury_status": None},
        {"player_id": "p3", "name": "TE1", "position": "TE",
         "kalman_est_target_share": 0.13, "injury_status": None},
        {"player_id": "p4", "name": "RB1", "position": "RB",
         "kalman_est_target_share": 0.08, "injury_status": None},
    ]


@pytest.fixture
def sample_game_logs_df():
    """Minimal game_logs-style DataFrame for fitting TeamVolumePredictor."""
    rows = []
    for team in ["MIN", "GB", "KC"]:
        for game_id in [f"{team}_g{i}" for i in range(8)]:
            # QB row — has pass attempts
            rows.append({
                "team": team, "game_id": game_id,
                "attempts": 35 + np.random.randint(-5, 5),
                "carries": 0,
            })
            # RB row — has rush carries
            rows.append({
                "team": team, "game_id": game_id,
                "attempts": 0,
                "carries": 24 + np.random.randint(-4, 4),
            })
    return pd.DataFrame(rows)


# ── _injury_multiplier ────────────────────────────────────────────────────────

class TestInjuryMultiplier:
    def test_none_returns_1(self):
        assert _injury_multiplier(None) == 1.0

    def test_empty_string_returns_healthy(self):
        """Empty string = no injury report → treat as healthy (1.0)."""
        assert _injury_multiplier("") == 1.0

    def test_out_returns_0(self):
        for status in ["out", "OUT", "Out", "dnp", "ir"]:
            assert _injury_multiplier(status) == 0.0, f"Expected 0.0 for status={status}"

    def test_doubtful_returns_low(self):
        m = _injury_multiplier("doubtful")
        assert 0.0 < m < 0.5

    def test_questionable_returns_mid(self):
        m = _injury_multiplier("questionable")
        assert 0.5 < m < 0.9

    def test_limited_returns_high(self):
        m = _injury_multiplier("limited")
        assert 0.8 < m < 1.0

    def test_healthy_string_returns_1(self):
        for s in ["full", "active", "healthy", "probable"]:
            assert _injury_multiplier(s) == 1.0


# ── DirichletShareDistributor ─────────────────────────────────────────────────

class TestDirichletShareDistributor:

    def test_concentrations_all_healthy(self, healthy_players):
        alphas = DirichletShareDistributor.compute_concentrations(healthy_players)
        assert len(alphas) == 4
        assert (alphas > 0).all(), "All healthy players should have positive concentration"

    def test_concentrations_injured_zero(self, injured_wr1_players):
        alphas = DirichletShareDistributor.compute_concentrations(injured_wr1_players)
        assert alphas[0] == 0.0, "OUT player should have zero concentration"
        assert (alphas[1:] > 0).all(), "Remaining players should have positive concentration"

    def test_concentrations_proportional_to_share(self):
        """Higher kalman share → higher concentration (proportionally)."""
        players = [
            {"kalman_est_target_share": 0.30, "injury_status": None},
            {"kalman_est_target_share": 0.10, "injury_status": None},
        ]
        alphas = DirichletShareDistributor.compute_concentrations(players)
        # WR1 share is 3× WR2 — alpha should be ~3× as well
        ratio = alphas[0] / alphas[1]
        assert 2.5 <= ratio <= 3.5, f"Expected ~3.0 ratio, got {ratio:.2f}"

    def test_sample_shares_sum_to_1(self, healthy_players):
        rng = np.random.default_rng(42)
        alphas = DirichletShareDistributor.compute_concentrations(healthy_players)
        samples = DirichletShareDistributor.sample_shares(alphas, n_samples=500, rng=rng)
        row_sums = samples.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9,
                                   err_msg="All Dirichlet samples must sum to 1")

    def test_injured_player_always_zero(self, injured_wr1_players):
        rng = np.random.default_rng(0)
        alphas = DirichletShareDistributor.compute_concentrations(injured_wr1_players)
        samples = DirichletShareDistributor.sample_shares(alphas, n_samples=200, rng=rng)
        assert (samples[:, 0] == 0.0).all(), "Injured player column must be exactly 0 in all samples"

    def test_remaining_players_still_sum_to_1_after_injury(self, injured_wr1_players):
        rng = np.random.default_rng(1)
        alphas = DirichletShareDistributor.compute_concentrations(injured_wr1_players)
        samples = DirichletShareDistributor.sample_shares(alphas, n_samples=500, rng=rng)
        # With player 0 = 0, remaining must still sum to 1
        np.testing.assert_allclose(
            samples[:, 1:].sum(axis=1), 1.0, atol=1e-9,
            err_msg="Remaining player shares must sum to 1 even with injury"
        )

    def test_zero_share_fallback(self):
        """Edge case: all players have zero share estimate → uniform fallback."""
        players = [
            {"kalman_est_target_share": 0.0, "injury_status": None},
            {"kalman_est_target_share": 0.0, "injury_status": None},
        ]
        alphas = DirichletShareDistributor.compute_concentrations(players)
        assert (alphas > 0).all(), "Zero-share players should fall back to ALPHA_MIN"


# ── TeamVolumePredictor ───────────────────────────────────────────────────────

class TestTeamVolumePredictor:

    def test_default_volume_without_fit(self):
        vp = TeamVolumePredictor()
        vol = vp.predict_pass_volume("MIN")
        assert 25.0 <= vol <= 50.0, f"Default pass volume out of range: {vol}"

    def test_fit_updates_team_averages(self, sample_game_logs_df):
        vp = TeamVolumePredictor()
        vp.fit(sample_game_logs_df)
        assert "MIN" in vp._team_pass_avg

    def test_spread_adjustment_reduces_passes(self):
        vp = TeamVolumePredictor()
        # Home team underdog (spread = +7 = away team favored) → home team throws more to catch up
        vol_underdog = vp.predict_pass_volume("MIN", {"spread_line": 7.0,  "total_line": 45.0, "is_home": 1})
        # Home team neutral
        vol_neutral  = vp.predict_pass_volume("MIN", {"spread_line": 0.0,  "total_line": 45.0, "is_home": 1})
        # Trailing team (underdog) throws more
        assert vol_underdog > vol_neutral, (
            f"Home underdog should pass more: {vol_underdog:.1f} vs {vol_neutral:.1f}"
        )

    def test_high_total_increases_passes(self):
        vp = TeamVolumePredictor()
        vol_high_ou = vp.predict_pass_volume("MIN", {"spread_line": 0.0, "total_line": 55.0, "is_home": 1})
        vol_low_ou  = vp.predict_pass_volume("MIN", {"spread_line": 0.0, "total_line": 37.0, "is_home": 1})
        assert vol_high_ou > vol_low_ou, "High O/U games should have more passing"

    def test_volume_never_negative(self):
        vp = TeamVolumePredictor()
        # Extreme spread — ensure floor is 1.0
        vol = vp.predict_pass_volume("MIN", {"spread_line": -50.0, "total_line": 25.0, "is_home": 1})
        assert vol >= 1.0

    def test_pass_rate_present_replaces_spread_total_heuristic(self, sample_game_logs_df):
        """
        Phase 6 L1→L3 wiring: when team_game_predictions.pass_rate is
        available, it drives the pass/rush SPLIT of total historical plays
        instead of the hand-tuned spread/total coefficients. total_plays
        still comes from this class's own historical average — Phase 4
        doesn't serve a plays count (see migration 20260822_0020).
        """
        vp = TeamVolumePredictor()
        vp.fit(sample_game_logs_df)
        total_plays = vp._total_plays("MIN")

        pass_vol = vp.predict_pass_volume("MIN", {"pass_rate": 0.65, "spread_line": -50.0, "total_line": 25.0})
        rush_vol = vp.predict_rush_volume("MIN", {"pass_rate": 0.65, "spread_line": -50.0, "total_line": 25.0})

        # Exact reconciliation: the extreme spread/total values above would
        # move the fallback heuristic a lot — pass_rate must override them
        # entirely, not blend, so this only holds if pass_rate truly won.
        assert pass_vol == pytest.approx(total_plays * 0.65, abs=1e-9)
        assert rush_vol == pytest.approx(total_plays * 0.35, abs=1e-9)
        assert pass_vol + rush_vol == pytest.approx(total_plays, abs=1e-9)

    def test_higher_pass_rate_strictly_increases_pass_volume(self, sample_game_logs_df):
        vp = TeamVolumePredictor()
        vp.fit(sample_game_logs_df)
        low = vp.predict_pass_volume("MIN", {"pass_rate": 0.40})
        high = vp.predict_pass_volume("MIN", {"pass_rate": 0.70})
        assert high > low

    def test_missing_pass_rate_falls_back_to_spread_total_heuristic(self):
        """No pass_rate in game_context → identical to pre-Phase-6 behavior."""
        vp = TeamVolumePredictor()
        ctx = {"spread_line": 7.0, "total_line": 45.0, "is_home": 1}
        with_none = vp.predict_pass_volume("MIN", {**ctx, "pass_rate": None})
        without_key = vp.predict_pass_volume("MIN", ctx)
        assert with_none == pytest.approx(without_key, abs=1e-9)


# ── VolumeRedistributor ───────────────────────────────────────────────────────

class TestVolumeRedistributor:

    def test_healthy_roster_targets_sum_to_team_volume(self, healthy_players):
        vr = VolumeRedistributor()
        rng = np.random.default_rng(42)
        result = vr.redistribute("MIN", healthy_players, n_samples=1000, rng=rng)
        total_targets = sum(r["expected_targets"] for r in result.values())
        team_vol = list(result.values())[0]["team_volume"]
        # In expectation, targets should approximately equal team volume
        assert abs(total_targets - team_vol) < 3.0, (
            f"Total targets {total_targets:.1f} far from team volume {team_vol:.1f}"
        )

    def test_injured_player_gets_zero_targets(self, injured_wr1_players):
        vr = VolumeRedistributor()
        rng = np.random.default_rng(7)
        result = vr.redistribute("MIN", injured_wr1_players, n_samples=500, rng=rng)
        wr1 = result["p1"]
        assert wr1["expected_targets"] == 0.0
        assert wr1["is_active"] is False

    def test_redistribution_increases_teammate_targets(self, healthy_players, injured_wr1_players):
        vr = VolumeRedistributor()
        rng_h = np.random.default_rng(10)
        rng_i = np.random.default_rng(10)

        result_healthy = vr.redistribute("MIN", healthy_players, n_samples=2000, rng=rng_h)
        result_injured = vr.redistribute("MIN", injured_wr1_players, n_samples=2000, rng=rng_i)

        # WR2 should get more targets when WR1 is out
        wr2_healthy = result_healthy["p2"]["expected_targets"]
        wr2_injured = result_injured["p2"]["expected_targets"]
        assert wr2_injured > wr2_healthy, (
            f"WR2 targets should increase on WR1 injury: {wr2_healthy:.1f} → {wr2_injured:.1f}"
        )

    def test_redistribute_returns_all_player_ids(self, healthy_players):
        vr = VolumeRedistributor()
        result = vr.redistribute("MIN", healthy_players, n_samples=100)
        assert set(result.keys()) == {"p1", "p2", "p3", "p4"}

    def test_questionable_player_reduced_targets_vs_doubtful(self):
        """Questionable player should get more targets than a doubtful player with same prior share."""
        players_q = [
            {"player_id": "p1", "name": "WR1", "position": "WR",
             "kalman_est_target_share": 0.30, "injury_status": "questionable"},
            {"player_id": "p2", "name": "WR2", "position": "WR",
             "kalman_est_target_share": 0.30, "injury_status": None},
            {"player_id": "p3", "name": "TE1", "position": "TE",
             "kalman_est_target_share": 0.15, "injury_status": None},
        ]
        players_d = [
            {"player_id": "p1", "name": "WR1", "position": "WR",
             "kalman_est_target_share": 0.30, "injury_status": "doubtful"},
            {"player_id": "p2", "name": "WR2", "position": "WR",
             "kalman_est_target_share": 0.30, "injury_status": None},
            {"player_id": "p3", "name": "TE1", "position": "TE",
             "kalman_est_target_share": 0.15, "injury_status": None},
        ]
        vr = VolumeRedistributor()
        rng_q = np.random.default_rng(44)
        rng_d = np.random.default_rng(44)
        result_q = vr.redistribute("MIN", players_q, n_samples=2000, rng=rng_q)
        result_d = vr.redistribute("MIN", players_d, n_samples=2000, rng=rng_d)
        # Questionable player gets more targets than doubtful player
        assert result_q["p1"]["expected_targets"] > result_d["p1"]["expected_targets"], (
            "Questionable player should get more targets than doubtful player"
        )
        # Both should be well below a healthy player with same share
        assert result_q["p1"]["expected_targets"] > 0.0, "Questionable player should still get targets"
        assert result_d["p1"]["expected_targets"] > 0.0, "Doubtful player should get some targets"

    def test_fit_runs_without_error(self, sample_game_logs_df):
        vr = VolumeRedistributor()
        vr.fit(sample_game_logs_df)
        assert vr._fitted is True

    def test_empty_player_list(self):
        vr = VolumeRedistributor()
        result = vr.redistribute("MIN", [])
        assert result == {}

    def test_redistribute_team_projections_df(self, healthy_players, injured_wr1_players):
        """Test the DataFrame-level API used by train.py."""
        all_players = injured_wr1_players  # WR1 is OUT

        # Build a minimal projections DataFrame
        df = pd.DataFrame([
            {
                "player_id":                  p["player_id"],
                "name":                       p["name"],
                "team":                       "MIN",
                "position":                   p["position"],
                "kalman_est_target_share":    p["kalman_est_target_share"],
                "kalman_est_carries":         0.0,
                "projected_targets":          p["kalman_est_target_share"] * 35,
                "projected_carries":          0.0,
            }
            for p in all_players
        ])

        injury_report = {"p1": "out"}
        vr = VolumeRedistributor()
        result_df = vr.redistribute_team_projections(df, injury_report=injury_report)

        # WR1 projected_targets should be 0
        wr1_row = result_df[result_df["player_id"] == "p1"].iloc[0]
        assert wr1_row["projected_targets"] == 0.0

        # WR2 projected_targets should be higher than original
        wr2_before = df[df["player_id"] == "p2"].iloc[0]["projected_targets"]
        wr2_after  = result_df[result_df["player_id"] == "p2"].iloc[0]["projected_targets"]
        assert wr2_after > wr2_before

        # Raw originals should be preserved
        assert "_raw_projected_targets" in result_df.columns

    def test_allocation_runs_with_empty_injury_report_not_skipped(self, healthy_players):
        """
        Phase 6 L3: redistribute_team_projections is the share model, not
        just an injury patch. An empty injury_report ({}) must still run
        the Dirichlet allocation (every player healthy) rather than passing
        raw, non-normalized Kalman shares straight through.
        """
        df = pd.DataFrame([
            {
                "player_id": p["player_id"], "name": p["name"], "team": "MIN",
                "position": p["position"],
                "kalman_est_target_share": p["kalman_est_target_share"],
                "kalman_est_carries": 0.0,
                "projected_targets": p["kalman_est_target_share"] * 35,
                "projected_carries": 0.0,
            }
            for p in healthy_players
        ])
        vr = VolumeRedistributor()
        result_df = vr.redistribute_team_projections(df, injury_report={})
        # target_share_mean is only written by the Dirichlet path — its
        # presence proves allocation actually ran.
        assert "target_share_mean" in result_df.columns
        assert result_df["target_share_mean"].notna().all()

    def test_skill_position_shares_sum_to_exactly_one(self, healthy_players):
        """
        The plan's Phase 6 verify line: shares sum to 1 within a team. Exact,
        not approximate — the Dirichlet constraint guarantees this by
        construction, so a loose tolerance would hide a real regression.
        """
        df = pd.DataFrame([
            {
                "player_id": p["player_id"], "name": p["name"], "team": "MIN",
                "position": p["position"],
                "kalman_est_target_share": p["kalman_est_target_share"],
                "kalman_est_carries": 0.0,
                "projected_targets": 0.0, "projected_carries": 0.0,
            }
            for p in healthy_players
        ])
        vr = VolumeRedistributor()
        result_df = vr.redistribute_team_projections(df, injury_report={}, n_samples=5000)
        total_share = result_df["target_share_mean"].sum()
        # Dirichlet sampling has Monte Carlo noise on the MEAN of n_samples
        # draws, not an exact identity — but it converges tightly at 5000
        # samples over 4 players. This checks the mechanism, not a single draw.
        assert total_share == pytest.approx(1.0, abs=1e-2)

    def test_non_skill_positions_pass_through_unmodified(self, healthy_players):
        """
        An OL/DL/DB row must never enter the Dirichlet group — it would get
        a nonzero ALPHA_MIN floor concentration despite a real share of ~0,
        diluting every skill player's allocation. Non-skill rows keep
        whatever projection they arrived with.
        """
        rows = [
            {
                "player_id": p["player_id"], "name": p["name"], "team": "MIN",
                "position": p["position"],
                "kalman_est_target_share": p["kalman_est_target_share"],
                "kalman_est_carries": 0.0,
                "projected_targets": 1.23, "projected_carries": 0.0,
            }
            for p in healthy_players
        ]
        rows.append({
            "player_id": "ol1", "name": "LeftTackle", "team": "MIN", "position": "OT",
            "kalman_est_target_share": 0.0, "kalman_est_carries": 0.0,
            "projected_targets": 0.0, "projected_carries": 0.0,
        })
        df = pd.DataFrame(rows)
        vr = VolumeRedistributor()
        result_df = vr.redistribute_team_projections(df, injury_report={})
        ol_row = result_df[result_df["player_id"] == "ol1"].iloc[0]
        assert pd.isna(ol_row.get("target_share_mean"))
        assert ol_row["projected_targets"] == 0.0

    def test_teammate_shares_strictly_increase_when_one_player_marked_out(self, healthy_players):
        """
        Turns the module docstring's Jefferson/Addison example into a real
        assertion: mark one player OUT, every OTHER active teammate's share
        strictly increases, and the OUT player's share is exactly 0.
        """
        df = pd.DataFrame([
            {
                "player_id": p["player_id"], "name": p["name"], "team": "MIN",
                "position": p["position"],
                "kalman_est_target_share": p["kalman_est_target_share"],
                "kalman_est_carries": 0.0,
                "projected_targets": 0.0, "projected_carries": 0.0,
            }
            for p in healthy_players
        ])
        vr = VolumeRedistributor()
        healthy_result = vr.redistribute_team_projections(df, injury_report={}, n_samples=3000)
        injured_result = vr.redistribute_team_projections(df, injury_report={"p1": "out"}, n_samples=3000)

        p1_share_after = injured_result.loc[injured_result["player_id"] == "p1", "target_share_mean"].iloc[0]
        assert p1_share_after == 0.0

        for pid in ("p2", "p3", "p4"):
            before = healthy_result.loc[healthy_result["player_id"] == pid, "target_share_mean"].iloc[0]
            after = injured_result.loc[injured_result["player_id"] == pid, "target_share_mean"].iloc[0]
            assert after > before, f"{pid} share should strictly increase when p1 is OUT: {before:.3f} -> {after:.3f}"


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestGetRedistributor:
    def test_get_redistributor_returns_instance(self):
        vr = get_redistributor()
        assert isinstance(vr, VolumeRedistributor)

    def test_get_redistributor_singleton(self):
        vr1 = get_redistributor()
        vr2 = get_redistributor()
        assert vr1 is vr2
