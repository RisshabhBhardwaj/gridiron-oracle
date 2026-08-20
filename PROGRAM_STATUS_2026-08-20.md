# Where the five surfaces actually stand

**Date:** 2026-08-20 · **Suite:** 1204 passed, 19 skipped, 0 failed
**Goal being scored against:** real, useful, realistic predictions for play-by-play,
game-by-game, season-long, player-level across all three, and the fantasy/draft translation.

---

## Scorecard

| Surface | Real? | Useful? | Realistic? | Would I act on it |
|---|---|---|---|---|
| **Game-by-game (player-week)** | Yes | Yes | Yes | **Yes** |
| **Player-level, weekly** | Yes | Yes | Yes | **Yes** |
| **Draft / fantasy translation** | Yes | Yes | Yes | **Yes, via `blended_rank`** |
| **Season-long** | Yes | Partly | **No** | Ranks yes, ranges no |
| **Player-level, season** | Yes | Partly | No | Same |
| **Play-by-play** | **No** | — | — | **Nothing to act on** |

Three of five are there. One is half there. One does not exist.

---

## Game-by-game — the surface that works

Beats both naive baselines in **74 of 75** cell-seasons and Marcel in **15 of 16** on
fantasy points. Serving is bit-exact with evaluation across **80,404 rows**, so the
measured object is the served object. Conformal intervals cover inside the 75–85% band
in **15 of 15** cells.

Real, useful, realistic. The one thing it cannot claim is beating *weekly consensus* —
no historical weekly consensus exists to score against. Sleeper capture is running; the
first honest comparison lands after the 2026 season.

## Draft translation — usable, with a specific known weakness

Three defects closed and verified against the live API:

- **12 QBs in the top 24 → 3.** Market has 0–1.
- **Retired players gone.** Tom Brady had survived two rebuilds because materialization
  only upserted; it now replaces the run.
- **Rookies ranked 7/7, and better than the market ranks them** (Spearman 0.53 vs ADP's
  0.42 on 127 rookie seasons). Jeremiyah Love: 120th in the audit, now 21st on the blend.
- **Projected points are now honest.** Calibration ratio **0.712 → 1.006**. The board used
  to print numbers ~29% low; Gibbs now reads 312 points rather than ~220.

Against consensus, measured over **1030 player-seasons** with paired bootstrap CIs:

| | Model | ADP | Paired difference |
|---|---:|---:|---|
| Rank correlation, all players | **0.489** | 0.439 | +0.051 [−0.008, +0.113] ns |
| Precision **within position** | 0.486 | **0.556** | **−0.069 [−0.115, −0.031] significant** |

**That second row is the honest weakness.** Our positional framework is good — that is
where the nominal cross-position edge comes from. But *inside* a position, ADP picks the
right players significantly better than we do. That is the market's news advantage:
offseason role changes, camp reports, holdouts. It is exactly what a projection built only
from prior box scores cannot see.

**Correction to what I told you an hour ago.** I reported a significant win on top-24
precision. That metric ranked against the raw-points top 24, which in a 1-QB league is
9–11 quarterbacks — the same illegal-oracle bug I found yesterday, appearing a **third**
time. Under legal positional slots the comparison reverses: ADP wins. The metric is fixed,
the position-blind version is retained as a labelled diagnostic, and there is now a
regression test that fails if a quarterback-pile board scores better.

## Season-long — informative ranks, uninformative ranges

Nominal 80% intervals cover **37.6%** (was 10.5%). The board is no longer inverted —
top-24 correlation is +0.12, overall Spearman 0.64, and the top 24 average 71.6 realized
points against a field of 41.3, so it does discriminate.

Use the ordering. **Do not use the ranges.** Cause is diagnosed and recorded: the season
path takes a single week's projection and extrapolates it across every remaining week with
nothing modelling whether the player keeps that role.

## Play-by-play — does not exist

Transitions export and the C++ engine load, and that is all. No endpoints, no tracking
data, never backtested. Unchanged from the audit.

---

## The measurement problem, which was the real blocker

The draft board simulation produces one number per season and only six seasons of resolved
ADP exist. Season-to-season sd is 170.6; the SE on the mean gap is ~106. **Neither "beats
ADP" nor "loses to ADP" is supportable from it** — roughly 13–28 seasons would be needed.

Two component decisions had been made inside that noise before this was measured. Both were
wrong. `ml/draft_eval.py` and `scripts/draft_component_eval.py` now decide components over
1030 player-seasons with paired bootstrap CIs clustered by player; `draft_walkforward.py`
prints the gap with its interval and `underpowered: true` instead of a verdict.

Shipped on that instrument: rookie curve (Spearman +0.043, significant), games calibration
(calibration +0.302, significant). Age curve is **off** — nothing significant on any metric,
and an unresolvable component should lose to the simpler system.

---

## What remains, in order

1. **Within-position discrimination** — the one measured, significant deficit against
   consensus. Closing it needs information we do not ingest: depth charts as-of, injury
   and practice reports, offseason role change. This is the real work item.
2. **Season interval coverage** (37.6% → 75–85%) — needs role persistence, not wider bands.
3. **Play-by-play** — a build, not a fix.
4. FFC `teams=8` ADP; SP2 production artifact; SP5 endpoints.

**For your draft this month:** sort by `blended_rank`. The projected point totals are now
calibrated and can be read as points. Ignore season-long ranges.
