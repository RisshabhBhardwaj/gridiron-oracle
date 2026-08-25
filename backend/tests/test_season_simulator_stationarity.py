"""
Regression tests for the rest-of-season projection blow-up (see
ml/season_simulator.py: the block comment in run() where _advance_kalman
used to be called, and SeasonSimulator._ZERO_RATE_EPS).

Production symptoms these lock out, all from the materialized
(2026, start_week=1) run:
  * De'Von Achane (RB) projected for 2418.9 receiving yards — above the NFL
    single-season record, at a position that does not lead a team in it.
  * Stefon Diggs (WR) 1333.4 passing yards; MIA's entire passing budget
    sprayed across eight WR/TE with no passing role at ~130 yards each.
  * League-wide weekly fantasy_ppr mean decaying 3.74 -> 1.31 across the
    simulated season while the top WR's weekly receiving_yards climbed
    83.8 -> 158.9 — the same rich-get-richer drift seen from both ends.

No PostgreSQL required: run() only touches the DB when a non-empty
schedule_df asks it to resolve real games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.season_simulator import SeasonSimulator

_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "fantasy_ppr"]
_WEEKS = 18


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": "qb1", "position": "QB", "team": "AAA"},
        {"player_id": "wr1", "position": "WR", "team": "AAA"},
        {"player_id": "wr2", "position": "WR", "team": "AAA"},
    ])


def _priors() -> dict[str, list[dict]]:
    """17 prior games apiece, with realistic game-to-game spread."""
    rs = np.random.default_rng(11)

    def rows(pas: float, rush: float, rec: float, ppr: float) -> list[dict]:
        return [
            {
                "passing_yards": float(max(a, 0.0)),
                "rushing_yards": float(max(b, 0.0)),
                "receiving_yards": float(max(c, 0.0)),
                "fantasy_points_ppr": float(max(d, 0.0)),
            }
            for a, b, c, d in zip(
                rs.normal(pas, max(pas * 0.3, 1e-9), 17) if pas else np.zeros(17),
                rs.normal(rush, max(rush * 0.4, 1e-9), 17) if rush else np.zeros(17),
                rs.normal(rec, max(rec * 0.45, 1e-9), 17) if rec else np.zeros(17),
                rs.normal(ppr, max(ppr * 0.4, 1e-9), 17),
            )
        ]

    return {
        "qb1": rows(250.0, 15.0, 0.0, 18.0),
        "wr1": rows(0.0, 2.0, 80.0, 14.0),
        "wr2": rows(0.0, 0.0, 30.0, 7.0),
    }


def _run(p_active: float, n_simulations: int = 400):
    sim = SeasonSimulator(
        season=2026, start_week=1, end_week=_WEEKS,
        n_simulations=n_simulations, stats=_STATS,
    )
    return sim.run(
        players_df=_players(),
        prior_game_rows=_priors(),
        player_active_prob={pid: p_active for pid in ("qb1", "wr1", "wr2")},
        rng_seed=0,
    )


def _weekly(result, player_id: str, stat: str) -> np.ndarray:
    frame = pd.DataFrame(result.week_by_week)
    frame = frame[(frame.player_id == player_id) & (frame.stat == stat)]
    return frame.sort_values("week")["mean"].to_numpy()


# ── 1. A stat the player has no role in projects to exactly zero ────────────

@pytest.mark.parametrize("player_id,stat", [
    ("wr1", "passing_yards"),
    ("wr2", "passing_yards"),
    ("wr2", "rushing_yards"),
    ("qb1", "receiving_yards"),
])
def test_zero_rate_stat_projects_to_exactly_zero(player_id, stat):
    """
    A zero Kalman posterior mean must yield exact zeros, not a clipped
    half-normal. max(N(0, std), 0) has mean 0.4*std, which is how WRs came to
    be projected for four figures of passing yards once _apply_volume_budget
    allocated the team's passing budget in proportion to these draws.
    """
    result = _run(p_active=0.9)
    assert result.player_season_totals[player_id][stat]["mean"] == 0.0
    assert not _weekly(result, player_id, stat).any()


# ── 2. Rest-of-season rates are stationary (no compounding drift) ───────────

@pytest.mark.parametrize("p_active", [0.4, 0.9])
def test_weekly_means_do_not_drift_across_the_season(p_active):
    """
    Nothing is learned during a forward simulation, so week 18's expected
    output must match week 1's. Feeding each week's simulated median back in
    as a Kalman observation made the first and second halves of the season
    diverge badly in BOTH directions at once.
    """
    # Needs more paths than the other cases: the season-level role multiplier
    # (_SEASON_ROLE_LOG_SD) widens each week's spread, and at p_active=0.4 only
    # ~40% of paths are non-zero, so a 400-path run puts the sampling error on
    # a half-season mean at ~10% -- enough to trip this on noise alone. The
    # drift being measured is systematic, so it does not shrink with n; the
    # noise does (measured: -10.4% at n=400, +0.6% at n=1500, +1.7% at n=4000).
    result = _run(p_active=p_active, n_simulations=1500)
    for player_id, stat in (("qb1", "passing_yards"), ("wr1", "receiving_yards"),
                            ("wr1", "fantasy_ppr")):
        weekly = _weekly(result, player_id, stat)
        assert len(weekly) == _WEEKS
        first_half = float(weekly[: _WEEKS // 2].mean())
        second_half = float(weekly[_WEEKS // 2 :].mean())
        assert second_half == pytest.approx(first_half, rel=0.10), (
            f"{player_id}/{stat} drifted {first_half:.2f} -> {second_half:.2f} "
            f"across the season at p_active={p_active}"
        )


# ── 3. Availability discounts the total, it does not annihilate the rate ────

def test_p_active_below_one_half_does_not_collapse_the_projection():
    """
    The specific arithmetic that broke production: week_paths is already
    multiplied by the Bernoulli(p_active) mask, so for p_active < 0.5 the
    median across paths is exactly 0 and every week applied
    new_est = 0.70 * prior_est. Over 18 weeks that is 0.70^18 ~= 0.0016.
    72% of the production roster sat at p_active <= 0.4.

    A season total must be ~ p_active * weeks * per-game rate.
    """
    p_active = 0.4
    result = _run(p_active=p_active)
    for player_id, stat, rate in (("qb1", "passing_yards", 250.0),
                                  ("wr1", "receiving_yards", 80.0),
                                  ("wr1", "fantasy_ppr", 14.0)):
        expected = p_active * _WEEKS * rate
        actual = result.player_season_totals[player_id][stat]["mean"]
        assert actual == pytest.approx(expected, rel=0.15), (
            f"{player_id}/{stat}: {actual:.1f} vs expected ~{expected:.1f}"
        )


def test_season_total_scales_with_availability():
    """A 0.9-availability player must out-project the same player at 0.4."""
    low = _run(p_active=0.4).player_season_totals["qb1"]["passing_yards"]["mean"]
    high = _run(p_active=0.9).player_season_totals["qb1"]["passing_yards"]["mean"]
    assert high / low == pytest.approx(0.9 / 0.4, rel=0.15)


# ── 4. The volume budget cannot amplify a collapsed group without bound ─────

def test_volume_budget_scale_is_capped():
    """
    When everyone who owns a stat is inactive on a path, the budget must not
    be multiplied onto whatever marginal contributor is left. MIA's whole QB
    room decaying to ~0 is how eight MIA receivers each drew ~130 passing
    yards off a budget none of them had any claim to.
    """
    from ml.season_simulator import _MAX_BUDGET_SCALE

    sim = SeasonSimulator(season=2026, start_week=1, end_week=1, n_simulations=8)
    # One marginal contributor holding a sliver of signal against a 230-yard
    # budget: uncapped this would scale by 230.
    results = {("wr9", "passing_yards"): np.full(8, 1.0)}
    sim._apply_volume_budget(
        results, {"wr9": "MIN"}, {"MIN": np.full(8, 230.0 / 0.6640356828998991)}
    )
    assert results[("wr9", "passing_yards")].max() <= _MAX_BUDGET_SCALE + 1e-9


def test_team_passing_and_receiving_totals_agree_when_passers_are_scarce():
    """
    Every passing yard is a receiving yard, so a team's two group totals must
    match on every path. Handing both groups the same raw budget only holds
    that while both can absorb it: with the zero-rate gate and
    _MAX_BUDGET_SCALE in play, a team short of available passers falls short
    on passing while its much larger receiver group still soaks up the full
    budget. On the real 2026 roster that split the league season totals by
    ~11% (110.8k passing vs 122.8k receiving).
    """
    from ml.season_simulator import _PASS_YARDS_SHARE

    sim = SeasonSimulator(season=2026, start_week=1, end_week=1, n_simulations=64)
    # One passer holding only a sliver of signal (his team's QBs are inactive
    # on these paths) against a full receiving corps at normal volume.
    results = {
        ("qb1", "passing_yards"): np.full(64, 0.5),
        ("wr1", "receiving_yards"): np.full(64, 70.0),
        ("wr2", "receiving_yards"): np.full(64, 55.0),
        ("te1", "receiving_yards"): np.full(64, 40.0),
    }
    team_by_pid = {p: "MIN" for p in ("qb1", "wr1", "wr2", "te1")}
    sim._apply_volume_budget(results, team_by_pid, {"MIN": np.full(64, 400.0)})

    passing = results[("qb1", "passing_yards")]
    receiving = sum(results[(p, "receiving_yards")] for p in ("wr1", "wr2", "te1"))
    np.testing.assert_allclose(receiving, passing, rtol=1e-9)
    # And the cap really did bite here — otherwise this proves nothing.
    assert passing.max() < 400.0 * _PASS_YARDS_SHARE


# ── 5. A team's yardage budget survives its passers being unavailable ───────

def test_budget_is_allocated_even_when_every_passer_is_unavailable():
    """
    Availability decides WHO produces a team's yards, not WHETHER the team
    plays. Each player's Bernoulli(p_active) is drawn independently, so on
    10.1% of simulated team-weeks league-wide NO quarterback was available at
    all (MIA: 41%, with four QBs at 0.16-0.88) and that team's entire passing
    budget silently vanished. That is a pure downward bias on every passing
    and receiving projection -- the reason no QB reached 4,000 passing yards
    and no receiver 1,400, which happens every real NFL season.
    """
    sim = SeasonSimulator(season=2026, start_week=1, end_week=1, n_simulations=32)
    # Masked results: every passer zeroed on every path.
    results = {("qb1", "passing_yards"): np.zeros(32),
               ("qb2", "passing_yards"): np.zeros(32)}
    raw = {("qb1", "passing_yards"): np.full(32, 240.0),
           ("qb2", "passing_yards"): np.full(32, 60.0)}
    budget = np.full(32, 400.0)
    sim._apply_volume_budget(
        results, {"qb1": "MIN", "qb2": "MIN"}, {"MIN": budget}, raw_unmasked=raw
    )
    from ml.season_simulator import _PASS_YARDS_SHARE

    total = results[("qb1", "passing_yards")] + results[("qb2", "passing_yards")]
    np.testing.assert_allclose(total, budget * _PASS_YARDS_SHARE, rtol=1e-9)
    # Split still follows relative role (240 vs 60 -> 4x).
    assert results[("qb1", "passing_yards")][0] == pytest.approx(
        4 * results[("qb2", "passing_yards")][0]
    )


def test_available_players_are_unaffected_by_the_fallback():
    """The fallback must only fire on paths with no signal at all."""
    sim = SeasonSimulator(season=2026, start_week=1, end_week=1, n_simulations=16)
    results = {("qb1", "passing_yards"): np.full(16, 200.0)}
    raw = {("qb1", "passing_yards"): np.full(16, 999.0)}
    budget = np.full(16, 400.0)
    sim._apply_volume_budget(
        results, {"qb1": "MIN"}, {"MIN": budget}, raw_unmasked=raw
    )
    from ml.season_simulator import _PASS_YARDS_SHARE

    np.testing.assert_allclose(
        results[("qb1", "passing_yards")], budget * _PASS_YARDS_SHARE, rtol=1e-9
    )


# ── 6. Season intervals reflect role uncertainty, not just weekly bounce ────

def test_season_intervals_are_not_artificially_tight():
    """
    Two things used to collapse the 80% band to p90/mean ~= 1.08:
      * the weekly draw used only the Kalman posterior variance (uncertainty
        in the RATE), which shrinks toward zero for a consistent starter, and
        omitted the stat's own game-to-game spread entirely;
      * with 18 i.i.d. weeks, relative spread shrinks by sqrt(n_games), so no
        per-week variance alone can produce a realistic season band.
    A rest-of-season range is dominated by ROLE uncertainty, which persists
    across weeks -- see _SEASON_ROLE_LOG_SD.
    """
    result = _run(p_active=0.93, n_simulations=600)
    for player_id, stat in (("qb1", "passing_yards"), ("wr1", "receiving_yards"),
                            ("wr1", "fantasy_ppr")):
        d = result.player_season_totals[player_id][stat]
        ratio = d["p90"] / d["mean"]
        assert 1.12 <= ratio <= 1.35, f"{player_id}/{stat} p90/mean = {ratio:.3f}"
        assert d["p10"] < d["p50"] < d["p90"]
