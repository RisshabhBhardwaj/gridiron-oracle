# Draft readiness review — 2026-08-20

## Decision

**Do not certify the repository as the coherent play → drive → game → week →
season prediction system described in the program plan.** That target is not
implemented. It is safe to use the current draft board as a *market-blended,
preseason ranking aid*, with the limitations below.

## Evidence checked

- The Phase 1 task list was delivered, but its own scope says that it is a
  roster/role and honesty foundation rather than the hierarchy build.
- The program plan's Phase 4 (team/game), Phase 5 (state-conditioned
  drive/play), Phase 6 (constrained allocation and one scoring function),
  Phase 7 (unified season simulation), and Phase 8 (external benchmarks) are
  still not all complete.
- Live DB: the current draft source is `preseason_historical_per_game`, as of
  2026-08-20; it contains 843 projections. The served Sleeper PPR board has
  127 players, 126 with `blended_rank`.
- Live DB: there are no 2026 rows in `projections`; `feature_matrix` ends at
  2025 week 22. Therefore 2026 weekly forecasts cannot be served from an
  approved forward-inference artifact.
- `scripts/verify_release_readiness.py` is **blocked** in artifact-backed
  mode: the baseline manifest is stale relative to current runtime changes,
  MLflow is unavailable, and feature data is 254 hours old.
- Focused tests covering drive engine, season simulator, draft API/evaluation,
  intervals, and serving correctness: **65 passed**. Frontend typecheck and
  production build passed. The complete backend suite could not be certified:
  it depends on a network fixture and the runner did not return a final summary
  in the available execution window.

## Findings

### P0 — Coherence architecture is absent

The plan requires a single generative process where player allocation is
bounded by a simulated team-game volume budget, fantasy points are derived
from component statistics, and season results are sums of the same game draws.
The current code still has independent surfaces: draft projections are a
historical preseason prior; weekly model stacks are separate; the season path
is not driven by a live team-game/drive/allocation process; and no play/drive
endpoint is served. The desired Pickens/Vikings consistency guarantees do not
hold by construction.

### P0 — No 2026 weekly, game, or play predictions

The database validates this directly: no 2026 projection rows exist, and the
feature matrix has no 2026 forward-inference rows. `ml/drive_engine.py` can
load a fitted transition artifact, but it has no serving endpoint and cannot
provide the requested play-by-play surface.

### P1 — Release gate is failing

The application is configured for `artifact_backed` serving, yet the release
readiness command reports `overall_status: blocked`. Until the baseline is
re-frozen after reviewing the post-baseline changes and the runtime services
are made available, it should not be represented as release-ready.

### P1 — There is no demonstrated independent edge over draft consensus

The current `blended_rank` deliberately combines model rank and ADP. It is the
appropriate board sort for today's draft, but it is not evidence that the
independent model beats consensus. The program status documents a significant
within-position deficit versus ADP; offseason role/news inputs remain missing.

### P2 — Full regression evidence is incomplete

Focused checks and frontend build passed. The full suite has a live nflverse
fixture; treat a final recorded successful suite summary as a release
requirement, rather than relying on earlier claims in walkthrough documents.

## Draft-day operating guidance

Use `/draft/board?season=2026` and sort by `blended_rank`. The current call
returned a 2026-08-20 historical-preseason board with 127 players. Treat model
point totals as calibrated estimates, not game-script-consistent simulations;
ignore season ranges; and use current depth-chart, injury, camp, holdout, and
league-specific roster news to override close calls.

## Required before claiming the original goal

1. Build approved 2026 feature rows and materialize causal weekly projections.
2. Ship and backtest the team-game model, then drive/play model, constrained
   allocation, a single scoring function, and a schedule-based season simulator
   from the same draws.
3. Add coherence tests: player components reconcile to fantasy points; team
   allocations stay within volume budgets; summed game results equal season
   totals/wins.
4. Capture and score historical weekly consensus, and measure against it
   separately from market-blended draft rankings.
5. Re-freeze the release manifest and pass readiness with a final full-suite
   result before marking artifact-backed serving ready.
