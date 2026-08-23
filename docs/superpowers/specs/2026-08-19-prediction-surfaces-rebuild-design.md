# Prediction surfaces rebuild — program design

**Date:** 2026-08-19 · **Revision 2** (integrates `OSS_SWEEP_2026-08-19.md` + league settings)
**Status:** design, awaiting review
**Evidence base:** `PREDICTION_SURFACE_AUDIT_2026-08-19.md`, `OSS_SWEEP_2026-08-19.md`, both same tree `f7eb25f` — held in the private archive (see `archive/INDEX.md`), not in this repository.

---

## 1. Purpose, bar, and league

Make all five prediction surfaces produce real, useful, realistic predictions: play-by-play, game-by-game,
season-long, player-level, and the fantasy/draft translation.

**Success bar: beat public consensus** — ADP and consensus rankings — not a naive baseline. The system must
be better than what you'd otherwise use, or it doesn't earn its place.

### League settings (decided)

| Setting | Value |
|---|---|
| Teams | **8** |
| Scoring | Standard **PPR** |
| Starters | 1 QB · 2 RB · 2 WR · 1 TE · 1 FLEX (RB/WR/TE) · 1 K · 1 DEF |

**This is a shallow league, and that changes the draft math more than anything else in this document.**
Only 8 QBs and 8 TEs start league-wide. Replacement-level QB is roughly **QB9** — a perfectly startable
player available on waivers. So **QB value-over-replacement is near zero at the top of the draft**, which
makes the current board's *12-QB top-24* not merely wrong but maximally wrong **for this specific league**.
Total relevant starters: 8 × 9 = **72 offensive slots**, so the ADP universe (148-224/season) is ample.

Two consequences that must be designed for, not discovered later:

- **VOR must be computed from these settings**, not generic 12-team assumptions (§7.1).
- **K and DST are startable slots the system does not project at all.** Decide explicitly: project them, or
  declare them out of scope and streamed. Recommendation: **out of scope**, documented — they are low-leverage
  and near-unpredictable, and pretending otherwise adds noise.

### What the bar is measurable against

| Surface | Consensus source | Measurable walk-forward? |
|---|---|---|
| Season-long / draft | `fantasy_adp` 2019-2025 (100% resolved) + **FFC REST API with `teams=8`** | **Yes** |
| Weekly | **None historical** | **No — capture starts now** |

**New capability from the sweep:** the Fantasy Football Calculator REST API accepts a `teams=N` parameter, so
you can pull **8-team-specific ADP** rather than generic 12-team ADP. For an 8-team league that is a materially
better baseline. Free, no key, commercial use permitted.

---

## 2. Decomposition and sequence

```
SP0  Enablers           → OpenMP fix, ID resolution, data refresh   (fast, unblocks everything)
SP1  Scoreboard         → make improvement measurable, serving honest
SP2  Playing-time/role  → the keystone; unblocks SP3 and SP4
SP3  Season-long        → rebuild on SP2
SP4  Draft / fantasy    → rebuild on SP2
SP5  DFS / play-by-play → now justified; correlated outcomes
```

SP0 is new — the sweep surfaced enabling work that is cheap and multiplies everything after it.
SP3 and SP4 may run in parallel once SP2 lands.

---

## 3. Corrections to the OSS sweep

The sweep is high quality and I'm adopting most of it. Three items must be corrected before anyone acts on
them, because two rest on the false headline my audit disproved.

### 3.1 ❌ "No demonstrated edge" is wrong — and it propagated

The sweep's constraint table (line 15) and its "uncomfortable part" #2 (line 184) both cite
*"MAE 35.43 vs constant-predictor 34.85"* from `reports/yardage_diagnostic.json`.

That artifact is **60 rows, one season, `temporal_ordered: false`, `feature_actual_columns: []`.** A proper
evaluation of the same cell over **7,374 rows across 5 seasons** gives `receiving_yards/WR` **MAE 24.43,
beating both baselines in all five seasons.** Across all cells the model beats both naive baselines in
**74 of 75 cell-seasons**.

**Do not gut the model stack on the strength of that number.** The sweep's strategic conclusion #2 — "model
complexity is far ahead of evidence" — survives in a *sharper* form: 5 of 15 cells are literally a single
LightGBM passthrough (`{lgbm: 1.0, catboost: 0.0}`), so for those cells the stack, the Bayesian layer, and
the Monte Carlo draws add nothing. That's the real complexity indictment, and it's better evidence than the
broken diagnostic.

### 3.2 🚨 Q4 (xFP as a "standalone causal projection") would manufacture a fake result

The sweep calls this "the highest information-per-hour test in this document." **It is a trap, and it is the
C-01 leak pattern exactly.**

`load_ff_opportunity` `*_exp` columns are computed from **the target game's own realized opportunity** —
targets, air yards, carries. Verified on 2023 weekly data:

```
corr(rec_fantasy_points_exp, rec_fantasy_points)  =  0.843   # same week
corr(receptions_exp,          rec_fantasy_points)  =  0.793
```

xFP is a **postgame** quantity. Scoring it "as a standalone causal projection under `ml/eval_causal`" would
compare a postgame measurement against pregame predictions. It would win enormously and mean nothing — the
same shape as `snap_pct_off`, which cost this project two audits and a full rebuild.

**xFP is still genuinely valuable — lagged.** See SP2.4. Any xFP column must enter `FORBIDDEN_MODEL_FIELDS`
in its contemporaneous form.

### 3.3 Minor

The `×3.0` sigma correction is at **`ml/train.py:939`**, not `ml/monte_carlo.py`. The sweep's read of *what
it is* — a calibration fudge with no guarantee — is exactly right.

---

## 4. SP0 — Enablers

Cheap, high-leverage, no modeling. Do these first.

| Task | Why | Evidence |
|---|---|---|
| **Fix the OpenMP conflict** (pixi + conda-forge unified `llvm-openmp`) | `OMP_NUM_THREADS=1` is a *performance* setting masquerading as a *stability* fix. You are training gradient boosters **single-threaded on a multi-core Mac.** Full run is 30-40 h; plausibly 5-12 h fixed. Every subsequent sub-project pays this tax on every iteration. | Sweep R3 / Q1 |
| **Adopt `load_ff_playerids`** | 12,470 players × 35 ID systems (`gsis_id`, `sleeper_id`, `fantasypros_id`, …), refreshed 2026-08-05. **This is the prerequisite for Sleeper weekly-consensus capture** (SP1.5) — you need `sleeper_id → player_id`. Also replaces the unresolved-name path in `adp_player_matches`. | Sweep Pass 2 |
| **Switch ADP to the FFC REST API with `teams=8`** | Replaces manual CSVs in `data/adp/historical/`; gives league-correct ADP. Free, no key, commercial-OK. | Sweep Pass 2 |
| **Build forward-only injury capture** | nflverse injuries **died after 2024, no ETA.** `injury_history` in your DB confirms it: 2009-2024 only. Capture nfl.com / ESPN injury + practice reports with `captured_at`, accept 2025 back-history is gone. | Sweep Pass 2, "still unbuilt" #1 |
| **Wire `participation.routes_run` into `feature_matrix`** | `feature_matrix.routes_run_per_game` and `slot_rate` are **100% NULL** across 129,128 rows, while `participation_player_game` holds **129,602 rows of real `routes_run`**. Wiring gap, not a data gap. | Audit + DB |

**Verification gate for the OpenMP work:** confirm exactly one `libomp.dylib` via `otool -L` across xgboost /
lightgbm / torch / sklearn, then time `ml/train_all_models.sh --seasons 2019-2020` before and after. Report
the wall-clock ratio. Keep `.venv_311` until parity is proven.

---

## 5. SP1 — The Scoreboard

**Goal:** the system can truthfully tell you whether a change made it better, against the consensus bar.

**Done when:** a deliberately-worsened candidate is rejected; tracked evidence matches the artifacts it
describes; nothing pre-remediation is servable; weekly consensus is accumulating.

### 5.1 Correct the evidence

- **Fix the units bug** in `scripts/write_post_leak_fix_comparison.py`: `_evaluate()` returns one verdict per
  **cell** (line 85), denominators `20/25/5/25` are hardcoded per **cell-season** (lines 131-134). Regenerate.
- **Delete** the 21 stale `reports/eval_causal_stack_*.csv` (13 tracked) that assert 71/75 from the leaky model.
- **Rewrite the README model-status section** and fix or delete `yardage_diagnostic.json` (§3.1).

### 5.2 Serving hygiene

- **Quarantine** the 5,469 rows from `stack_materialize_20260809T171811Z`, live and serving with
  `degraded: false`.
- **Enforce `approved_pipeline_run_ids` at serving.** Currently enforced only in `ml/backtest.py`;
  `_load_projection_row()` (`projection.py:588`) orders by `created_at` with no run filter. The manifest list
  is `[]`.
- **Fix interval plumbing.** `_interval_method()` (line 749) accepts only `posterior_samples`, NULL on all
  85,873 rows → always `"unavailable"` → `floor`/`ceiling` nulled, while `fantasy_floor`/`fantasy_ceiling`
  bypass the guard and leak the same numbers unlabeled. Teach it `"conformal"`; one guard for all four fields.
- **Fix the frontend.** `PlayerDetail.tsx:77-78` renders a fabricated `?? 0` floor; `Dashboard.tsx:156-158`
  calls `.toFixed()` on a null the types declare non-nullable, with **no error boundary anywhere in
  `frontend/src`**. Make the fields nullable; render an explicit "no interval" state.
- **Clamp** the 246 negative `fantasy_floor` rows (worst −3.53) at 0 for non-QB.

### 5.3 Make the gate capable of failing

Currently **75/75 passed** and structurally incapable of rejection:

- The "frozen prior release" is a **same-day bootstrap from the same training run**, not a prior release.
- `select_nonregressing_stacks.py` substitutes an **LGBM passthrough** wherever the Ridge lost in ≥1 season
  (5 of 15 cells, incl. `fantasy_ppr/QB` and `fantasy_ppr/WR`), so `candidate_mae == baseline_mae` to 16
  significant figures.

Required:

1. Frozen baseline must come from a **previously promoted release** with a distinct `release_id` and OOF hash.
2. **Refuse self-comparison:** `candidate_oof_sha256 == baseline_oof_sha256` → failure, not pass.
3. **Explicit first-release policy:** no valid prior → `no_prior_baseline_requires_human_override`, never
   auto-promote.
4. **Multiple-testing correction.** You select across (stat × position × learners × feature groups × alpha) —
   dozens to hundreds of trials, uncorrected. **Vendor** Deflated Sharpe Ratio + PBO + `effective_n_trials`
   (~60 lines) rather than depend on `purgedcv` (24★, single maintainer).
5. Replace the brittle "loses in *any* season → discard" rule with mean-performance-with-variance-penalty,
   recorded in the release.
6. Where a cell **is** a single-learner passthrough, the manifest and `/predict` must say so.

**Negative tests are the deliverable:** a 5%-worse candidate is rejected; an identical candidate is rejected
as self-comparison; a missing baseline entry is rejected, not skipped.

### 5.4 Baselines — the honest set

`ml/consensus_baseline.py`. Three baselines, not one:

1. **Consensus/ADP** — `fantasy_adp` 2019-2025 + FFC `teams=8`, restricted to the draftable universe.
2. **Marcel-style** — the sweep's sharpest methodological find. Multi-season weighted per-game rate (5/4/3),
   shrunk to position mean, **× projected snap/target share**, age-adjusted. Famously hard to beat in
   baseball and the honest bar here.
3. **Actuals-oracle + prev-season** — C-14 flagged both as missing; a Spearman number without them is
   uninterpretable.

> **Expect the measured edge to shrink.** The ~7% improvement in the audit is against *naive* and
> *trailing-3*. Marcel is a much stronger opponent, and its NFL form is essentially "rate × role" — which is
> what SP2 builds. **Run the Marcel comparison in SP1, before building SP2**, so you know the real starting
> point. If the stack cannot beat Marcel, that reframes the whole program, and it costs a day to find out.

Rank-weighted metrics (top-N hit rate, points-lost-vs-optimal per draft slot) alongside Spearman with
p-values — the top of the draft matters vastly more than the tail.

### 5.5 Proper distributional scoring

MAE cannot see whether an interval is right. Add **CRPS**. `scoringrules` requires **Python ≥3.12 and you are
on 3.11** — **vendor the ensemble-CRPS formula (~40 lines)** rather than upgrade the runtime mid-program.

### 5.6 Weekly consensus capture — the one deadline

**Before Week 1 2026.** Table `consensus_projections_weekly`
(`source, season, week, player_id, projection, scoring, captured_at`), append-only, **captured pregame** —
a snapshot taken after kickoff is as contaminated as any other postgame feature.

- **Primary: Sleeper** (your league is there; SP0's ID map handles `sleeper_id → player_id`).
- **Secondary: `load_ff_rankings`** (FantasyPros ECR via DynastyProcess). **Legal caveat:** DynastyProcess
  publishes no license for this redistribution. Keep it out of any promoted artifact until resolved
  (sweep Q7). Check whether it carries *historical* weekly ECR — if it does, part of the weekly bar becomes
  retroactively measurable and this deadline softens.

### 5.7 Test suite

Full suite: **4 failed, 1120 passed** (prior reports ran curated subsets showing 106).

- Three `TestStackingInferenceHelpers` tests are **hollow** — fixtures written to `tmp_path` are never read
  because `load_ridge_coefs` resolves through the manifest first (`inference_client.py:505-511`). Their
  result is decided by the production release. *(The validator itself is correct — verified directly.)*
- `test_normalize.py::test_count` hardcodes `19399` against live data now at `19400`.
- **Gate: full suite green, not a subset.**

---

## 6. SP2 — Playing-time and role model (the keystone)

**Goal:** predict causally (a) probability a player is active, (b) expected role share given active.

Fixes the season inversion, the rookie collapse, and per-player volatility — one component, three surfaces.

### 6.1 Data assets — verified present

| Asset | Rows | Coverage | Note |
|---|---|---|---|
| `depth_charts` | 195,539 | **2019-2026** weekly | Has `depth_rank` **and `published_at`** → as-of-correct |
| `participation_player_game` | 129,602 | 2019-2025 | **Real `routes_run`** (SP0 wires it in) |
| `injury_history` | 84,680 | **2009-2024 only** | Dead upstream; SP0 builds forward capture |
| `prior_snap_share` | 111,935 | verified 100% = previous game | Already causal |
| `load_ff_opportunity` | ~6,081/season | 2019+ | **xFP — lagged only, see 6.4** |
| `load_ftn_charting` | play-level | **2022+** | Blitzers, play-action, motion, RPO, screen |

### 6.2 Targets

- `p_active` — did the player record a snap.
- `snap_share | active` — bounded [0,1].
- `route_share` (pass-catchers), `carry_share` (RBs) — the opportunity channels that drive scoring.

### 6.3 Model

Two-stage, because availability and role are different questions:

1. **Availability classifier** → `p_active`, **probability-calibrated** (isotonic/Platt). SP3 multiplies by
   this; an uncalibrated classifier propagates straight into season totals.
2. **Role regressor** → share given active. Beta regression or quantile model; target is bounded.

### 6.4 xFP — lagged, and only lagged

Same-week xFP correlates **0.843** with realized points because it is derived from the target game's
opportunity (§3.2). Contemporaneous xFP columns go into `FORBIDDEN_MODEL_FIELDS`.

**Lagged, it is one of the strongest features available:**

- Prior-game `*_fantasy_points_exp` is a **de-noised measure of role quality** — it strips touchdown luck out
  of the usage signal, which raw prior points cannot do.
- **Actual-minus-expected on prior games is a regression-to-mean signal** — a player massively outscoring his
  opportunity is due for negative regression. That is a real, well-established fantasy edge and it is fully
  causal when lagged.

### 6.5 Causality constraints (non-negotiable, inherited from C-01)

- Every feature pregame; new columns pass `FORBIDDEN_MODEL_FIELDS`.
- Depth-chart features filtered by **`published_at <= kickoff`**, not merely by week.
- Walk-forward with real `max_train_season` provenance — `eval_causal.py` raises rather than inventing it,
  and that must stay true.
- Consider **Temporian** or Feast's point-in-time semantics **as a test oracle**, not a framework migration.
  You already built the as-of contract; re-implementing it deletes nothing (sweep's own verdict).

### 6.6 The defect this must kill

The uninformative Kalman prior is `est = 20.0, var = 1000.0` — higher per-game than every real 2025 starter
except four. A player with **no information** currently projects as a top-5 fantasy QB.

**Invariant test: a QB with no prior-season snaps must not appear in the top 24 of rest-of-season
projections.** Regression test on 2025 data. This is SP2's acceptance criterion.

### 6.7 Acceptance

- Backup QBs absent from the top 20 rest-of-season (2025 regression test).
- `p_active` calibration: predicted buckets match realized activity rate.
- Beats "same games played as last season" on availability.
- Role model beats `prior_snap_share` alone on held-out seasons.

---

## 7. SP3 — Season-long, rebuilt

**Current:** bypasses the trained models entirely — reads `kalman_est_*` and extrapolates across remaining
games with no role model. **80% intervals cover 10.5%.** Entire top-20 is backup QBs.

### 7.1 Replace the extrapolation

Simulate week by week from (a) the per-game stack projection, (b) expected role from SP2, (c) the
games-played distribution from SP2's availability model. Season totals become a distribution over simulated
paths, not `mean × remaining_games`.

### 7.2 Fix the uncertainty

The coverage failure is structural: the simulator propagates uncertainty about a player's *mean*, not
*outcome* variance. The same confusion exists on the weekly path — `ml/train.py:939` applies a hand-tuned
`std *= 3.0`, commented *"targeting ~80% empirical coverage,"* which correctly diagnoses the problem and then
papers over it with a constant that nothing tests.

**Replace both with conformal intervals (MAPIE, BSD-3, sklearn-contrib), using EnbPI/adaptive conformal for
the time-series structure.** This gives distribution-free coverage guarantees and — per the sweep — lets you
A/B the entire PyMC NUTS + 10k-draw Monte Carlo layer against a near-free alternative. **If conformal matches
or beats it on coverage and CRPS, the Bayesian layer is deletable.** That is a major simplification and it is
one experiment.

**Target: empirical 80% coverage inside [75%, 85%], tested, scored by CRPS.**

### 7.3 Serving

- Remove `except Exception: return []` (lines 330-332, 385-387) — silently converts failure into empty.
- Add `degraded` and `interval_method` to `SeasonPlayerProjection`; neither field exists.
- Gate under `artifact_backed`. Today `/health` reports 503 while this endpoint returns 200.
- **`nflseedR`** has real NFL tiebreaker/seeding logic but is **R-only** — port the rules or skip. Only
  matters for playoff-seeding scenarios, which are marginal for fantasy. Recommend skip.

### 7.4 Acceptance

- 80% coverage in [75%, 85%]; CRPS reported.
- Top-24 positively correlated with realized (currently anti-correlated at the top).
- **Beats ADP and Marcel** on the draftable universe.

---

## 8. SP4 — Draft and fantasy translation

**Current:** recency-weighted historical per-game PPR × a [8,17]-clamped games prior. Top 24 holds **12 QBs**;
market holds zero.

### 8.1 VOR from the real league settings

`value_vs_adp = adp_rank − model_rank` compares a raw-points rank against a market rank that embeds scarcity.
That one line is why the board says draft Mahomes ~95 picks early.

Replace with VORP computed from **8 teams · 1QB/2RB/2WR/1TE/1FLEX**:

- Starting slots league-wide: 8 QB, 16 RB, 16 WR, 8 TE, 8 FLEX (RB/WR/TE).
- Replacement level ≈ **QB9**, **TE9**, with RB/WR replacement set by the flex-inclusive pool.
- Rank by `projection − replacement_projection`.

**In an 8-team league QB VOR at the top is near zero** — QB9 is a startable player. Expect the corrected board
to look dramatically different from today's, and to agree with the market far more than it currently does.

**K and DST:** startable slots with no projections. Recommend declaring them **explicitly out of scope and
streamed**, documented in the board response, rather than emitting noise.

### 8.2 Scoring configurability — a decision to make explicitly

PPR is baked in: the target is `game_logs.fantasy_points_ppr`. Half-PPR / TE-premium / 6-pt passing TD cannot
be expressed. Your league is standard PPR, so this is **not urgent** — but it is the difference between a
tool for your league and a tool for any league.

- **(a) Model components and compose** — flexible, any scoring; compounds error across models.
- **(b) Keep modeling `fantasy_ppr` directly** — more accurate today; locks in one scoring system.

**Recommendation: keep (b) as primary, add (a) as a measured alternative.** If composed scoring is within
noise, switch and gain configurability free. That is a measurement the SP1 harness can answer — not an
assumption.

### 8.3 Rookie priors

7 rookies on the 2026 board share **3 distinct projections** and rank 120-126 — the board fades every rookie
by construction, including a real second-round ADP pick.

Minimum viable: **draft capital** (`load_draft_picks`) + **vacated opportunity share** on the landing team.
`combine` (1,571 rows, 2019-2026) adds athletic profile; `ml/player_embeddings.py` already computes speed
score. Anything with dispersion beats a constant.

Every projection carries an explicit `projection_basis` so the UI can label a rookie estimate as what it is.

### 8.4 Hygiene

Exclude retired players — **Tom Brady currently carries a 157-point 2026 projection** in
`draft_preseason_projections`. The ADP join hides it at serving; direct readers get it.

### 8.5 Acceptance

- Position mix defensible for an 8-team league; no 12-QB top-24.
- **Beats 8-team ADP and Marcel** on realized points per draft slot, walk-forward 2019-2025.
- Rookies show real dispersion.

---

## 9. SP5 — DFS and play-by-play (justified — you play DFS)

DFS changes the verdict. Keep `markov_simulator`, `gnn_matchup`, `rl_hedging_agent`,
`season_simulator_bridge` — but note that **none of them are currently reachable** (no importers, no
artifacts, no endpoints), so this is a build, not a revival.

**The DFS-specific value is correlation, not another marginal projection.** A per-play point estimate adds
nothing over the per-game model; the joint distribution is the entire point.

### 9.1 Correlated outcomes

Wire `DriveMarkovModel` → transitions artifact (`ml/oof/transitions.csv`, currently absent) → C++ `DriveMCMC`,
and expose **joint** outcomes for stacking (QB+WR same-game correlation, game-script effects, bring-backs).
`ml/copula_layer.py` already exists and is used by `season_simulator` — evaluate it as the cheaper path to
correlation before committing to drive simulation.

### 9.2 Formation / coverage — now unblocked

You've confirmed Big Data Bowl tracking data is usable. That unblocks `gnn_matchup`, which
`docs/ARCHITECTURE.md` lists as "blocked on formation data." Combine with `load_ftn_charting` (2022+:
blitzers, play-action, motion, RPO, screen, QB alignment).

**Caveat the sweep is right about:** FTN gives pressure/scheme, not **coverage shell**. `aDOT × coverage
shell` stays blocked unless tracking data fills it.

### 9.3 The exposure engine — the honest note

The C++ engine (lock-free SPSC ring buffer, Kelly sizing, exposure caps, 49 Catch2 tests) is the most
polished part of the system and has **never been backtested against real prop odds**. DFS justifies its
existence; it does not yet validate its parameters.

Historical player props are **paid-only, $30/mo, from 2023-05-03** — the sweep correctly identifies this as
the only line money can actually buy. Two paths: pay for history and backtest now, or start forward-only
capture on the free 500-credit tier and wait ~2 seasons. **Do not size real DFS exposure off an unbacktested
Kelly implementation** either way.

---

## 10. Cross-cutting invariants

Enforced by tests, not convention.

1. **Causality.** Every feature pregame. As-of data (depth charts, consensus, injuries, weather) filtered by
   publication timestamp, not by week. **xFP contemporaneous columns are forbidden.**
2. **Provenance.** Every artifact carries `max_train_season`; `eval_causal.py` raises rather than inferring.
3. **One population.** Model and baselines on the same declared finite cohort (`CohortSpec`).
4. **Honest degradation.** Every surface exposes `degraded` and `interval_method`; none returns an empty list
   to hide an error.
5. **The gate can fail.** Every sub-project adds a negative test proving rejection of a bad candidate.
6. **Multiplicity.** Any selection across many cells/learners/groups reports `effective_n_trials` and a
   multiplicity-corrected result.

---

## 11. What is already solid and must not regress

Verified independently this session:

- **C-01 leak closed** — `snap_pct_off` non-null on 0 rows; `prior_snap_share` = previous game on
  111,935/111,935 (100.0000%).
- **Walk-forward correct** — `max_train_season = season − 1` every fold, no leakage, no duplicates.
- **Serving matches evaluation bit-exact** — 15 cells, 80,404 rows, max diff `0.00e+00`.
- **Real per-game edge** — beats both baselines in 74/75 cell-seasons, ~7% better MAE than trailing-3
  *(pending the Marcel comparison in §5.4)*.
- **PPR never re-derived** — read from source; no scoring drift possible.
- **Ridge validator genuinely strict** — NaN and legacy schemas correctly rejected.

---

## 12. Open questions

| # | Question | Resolves how |
|---|---|---|
| Q1 | Does the stack beat **Marcel**? | §5.4 — do this before SP2. Highest information-per-hour test in this plan. |
| Q2 | Does **conformal** match/beat PyMC + `×3.0` on coverage and CRPS? | §7.2. If yes, delete the Bayesian + MC layer. |
| Q3 | Does `load_ff_rankings` carry **historical weekly** ECR? | If yes, the SP1.6 deadline softens materially. |
| Q4 | Is DynastyProcess's FantasyPros redistribution **licensed**? | Open an issue on `dynastyprocess/data`; keep out of promoted artifacts until answered. |
| Q5 | Actual **OpenMP speedup** on your hardware? | §4 — determines whether iteration costs 30 h or 8 h. |
| Q6 | Pay **$30/mo** for historical prop odds, or forward-capture and wait ~2 seasons? | Only if SP5 exposure sizing is a real goal. |
| Q7 | **K/DST** — project, or declare out of scope? | Recommendation: out of scope, documented. |
