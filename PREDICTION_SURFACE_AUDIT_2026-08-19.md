# Prediction-surface audit — can you trust this system for fantasy decisions?

**Date:** 2026-08-19 · **Tree:** `f7eb25f` (clean) · **DB:** `oracle@localhost:15439`, live
**Scope:** the four prediction surfaces — play-by-play, game-by-game, season-long, player-level — plus the
fantasy/draft translation. Prior audits (2026-08-09, 2026-08-11, 07 re-audit) covered plumbing: leakage,
artifact containment, gates, schema. **None of them evaluated the prediction surfaces themselves.** That gap
is what this audit fills.

Every number below is one I generated in this session against the live database and a running API, not one
copied from a prior report.

---

## Bottom line

**The per-game model is genuinely good, and the repo has been badly under-reporting it. The season-long and
draft surfaces are not trustworthy and should not drive decisions today.**

Three things are true at once and they need to be held separately:

1. **The core per-game model has real, broad, causal edge.** It beats both naive baselines in **74 of 75**
   cell-seasons and **15 of 15** cells. The repo's headline says "15/75". That is a reporting bug, not a
   result. The edge is **consistent but modest in size** — roughly **7% better MAE than a trailing-3-game
   average** on the yardage/points cells (more on count stats). Consistency that broad is strong evidence of
   real signal; a 7% weekly MAE improvement is not, on its own, going to decide a draft.
2. **The season-long surface is inverted.** Its entire top-20 rest-of-season board is backup quarterbacks.
   Nominal 80% intervals cover **10.5%** of outcomes.
3. **The draft board is not the model**, and its value column is positionally broken — it puts **12 QBs in
   its top 24** where the market has zero.

---

## 1. The headline correction: the model's edge was misreported by a units bug

This is the most consequential finding in the audit, and it runs *opposite* to what every prior document
concluded.

`reports/POST_LEAK_FIX_COMPARISON.md` reports the post-leak-fix edge as:

| Family | Reported |
|---|---|
| fantasy_ppr | 4/20 |
| volume | 5/25 |
| passing | 1/5 |
| yardage | 5/25 |

The 2026-08-11 re-audit built a central pillar of its DO-NOT-SHIP verdict on this: *"the model edge is now
small … 4/20, 5/25, 1/5, 5/25 … the product's value proposition needs restating."*

**The numerator and the denominator are counted in different units.**

In `scripts/write_post_leak_fix_comparison.py`, `_evaluate()` returns `int(model < naive and model < rolling)`
— a single 0-or-1 verdict **per cell**, computed on pooled rows (line 85). The family totals accumulate those
per-cell verdicts. But the denominators `20, 25, 5, 25` are **hardcoded literals** (lines 131-134) carried
over from the pre-fix table, where counting was **per cell-season**.

The families contain 4, 5, 1, and 5 cells respectively. So the reported numerators — 4, 5, 1, 5 — are each
**the maximum achievable value**. The model won *every cell*. Rendered against cell-season denominators, a
clean sweep prints as 20%.

I verified this two independent ways:

**(a) Re-running the script's own `_evaluate` on the promoted artifacts** — 15/15 cells win on pooled MAE
against both baselines:

```
volume       5/5 cells   -> report prints 5/25
fantasy_ppr  4/4 cells   -> report prints 4/20
passing      1/1 cells   -> report prints 1/5
yardage      5/5 cells   -> report prints 5/25
```

**(b) Scoring per cell-season through the hardened evaluator** (`ml.eval_causal.score_oof_against_baselines`,
which enforces `max_train_season < season`, applies `CohortSpec`, and scores model and both baselines on one
common finite cohort):

```
TOTAL cell-seasons: 75    beats_naive: 74    beats_trailing3: 75    beats BOTH: 74
```

The single loss is `rushing_yards/QB` 2024 against the prev-season baseline (12.335 vs 12.213) — it still
beats trailing-3.

The underlying predictions are sound. I confirmed the walk-forward structure directly: fold *i* has
`max_train_season = season − 1` in every fold, zero rows with `max_train_season >= season`, and zero duplicate
`(player_id, game_id)` pairs.

**This is not a small correction.** The repo currently tells its owner the model barely works. It works.

### How large is the edge, actually?

Correcting the count does not make the edge enormous, and it should not be oversold. Mean improvement in each
cell's primary metric (MAE; Poisson deviance for count stats):

| Cell | vs prev-season | vs trailing-3 |
|---|---:|---:|
| rushing_yards/QB | 3.4% | 4.6% |
| fantasy_ppr/QB | 8.8% | 6.0% |
| fantasy_ppr/RB | 9.4% | 6.4% |
| fantasy_ppr/WR | 7.5% | 6.4% |
| fantasy_ppr/TE | 5.6% | 8.6% |
| receiving_yards/WR | 6.3% | 6.9% |
| passing_yards/QB | 14.9% | 9.8% |
| targets/WR *(count)* | 40.9% | 17.0% |
| pass_attempts/QB *(count)* | 69.3% | 13.9% |

**MAE cells overall: 7.8% better than prev-season, 7.1% better than trailing-3.** For `fantasy_ppr`
specifically: **6.9% mean improvement over trailing-3** (range 3.7-10.5%).

So the honest statement is: **the 2026-08-11 audit's conclusion that the edge is "small" was defensible on
effect size, even though the number it cited was wrong because of the units bug.** What is not defensible is
"approximately baseline" or "no demonstrated edge" — the model beats both baselines in 74 of 75 cell-seasons,
and that consistency is the strong part of the result.

### The related consequence

The 2026-08-11 audit flagged as suspicious that the "recomputed legacy OOF" and "causal rebuild" columns were
*identical* in all four families, calling the count metric "nearly insensitive to the leak" and therefore
"weak evidence in either direction." The real explanation is simpler: **both columns are saturated at their
maximum.** Both the leaky and the causal model win every cell on pooled MAE. The metric wasn't insensitive —
it was clipped.

---

## 2. Per-surface trust verdicts

| Surface | Exists? | Backed by | Trustworthy today |
|---|---|---|---|
| **Play-by-play** | **No** | — | **N/A — the surface does not exist** |
| **Game-by-game (player-week)** | Yes | Promoted Ridge/LGBM stacks | **Yes for the point estimate. No for the uncertainty.** |
| **Season-long** | Yes | Kalman prior + C++ sim | **No — actively misleading** |
| **Player-level (`/predict`)** | Yes | Same as game-by-game | **Point estimate yes; intervals absent** |
| **Draft board** | Yes | Historical heuristic, *not* the model | **No — positionally broken** |

### 2.1 Play-by-play — does not exist

`ml/markov_simulator.py` (137 lines) extracts drive-level transition matrices and is intended to feed the C++
`DriveMCMC` engine. Verified:

- **No non-test importer** anywhere in `backend`, `ml`, `pipeline`, `scripts`, `e2e`.
- **No artifact**: `ml/oof/transitions.csv` does not exist; no `transitions*.csv` anywhere in the tree.
- **No endpoint**: no route serves drive- or play-level predictions.

The C++ side (`engine/src/drive_mcmc.cpp`) is real and compiled, but nothing connects Python to it. PBP data
*is* ingested (`pipeline/pbp_pipeline.py` feeds `features/buckets.py`), so play-by-play is a **feature
source, not a prediction surface**.

Same status — dead, no importers — for `ml/gnn_matchup.py` (553 lines), `ml/rl_hedging_agent.py` (173), and
`ml/season_simulator_bridge.py` (258).

**Verdict: the system does not make play-by-play predictions.** Nothing is wrong with the predictions because
there are none.

### 2.2 Game-by-game and player-level — the good surface, with a broken uncertainty layer

`/projections/week/{week}` and `/predict` both serve from the `projections` table, materialized from the
promoted stacks. All **15 declared cells are now served** (the 2026-08-11 finding C-10, five missing cells, is
genuinely closed).

**The evaluated object is the served object — verified, not assumed.** The edge in section 1 is measured on
OOF CSVs; what you consume comes from the `projections` table. I joined them on `(player_id, season, week)`
for all 15 cells:

```
cell                      joined   exact      maxdiff
fantasy_ppr/WR              9686    9686     0.00e+00
receiving_yards/WR          9686    9686     0.00e+00
passing_yards/QB            2733    2733     0.00e+00
        … all 15 cells …
cells with any mismatch: 0
```

**80,404 rows, bit-exact.** So the 74/75 result describes exactly the numbers the API returns. (C-21, the
never-implemented live-inference-vs-OOF parity check, is satisfied empirically here.) **The point-estimate
surface is sound.**

Four defects sit on top of it:

**(a) The uncertainty intervals are computed, then thrown away — except where they leak out unlabeled.**

`_interval_method()` (`backend/app/services/projection.py:749`) returns a usable method **only** when
`posterior_samples` is a non-empty list. `posterior_samples` is NULL on **all 85,873 rows**. So
`_interval_method` always returns `"unavailable"`, and `_interval_value()` (line 762) consequently nulls
`floor` and `ceiling` on every response.

But `fantasy_floor` and `fantasy_ceiling` are passed through raw `_opt()` with **no such guard** (lines
169-170, 252-253). Josh Allen, week 10 2025:

| | DB | API response |
|---|---|---|
| `floor` / `ceiling` | 7.837 / 36.769 | **null / null** |
| `interval_method` | — | **"unavailable"** |
| `fantasy_floor` / `fantasy_ceiling` | 7.837 / 36.769 | **7.837 / 36.769** |

The conformal bounds Wave 04 computed are suppressed on the fields designed to carry them and exposed,
unlabeled, on the fields that have no guard. The guard was written for posterior samples and never updated
when the project moved to conformal intervals.

**(b) The frontend substitutes `0` for the suppressed intervals — and one page will crash.**

- `frontend/src/pages/PlayerDetail.tsx:77-78`: `const floor = predictData?.percentiles.p10 ?? 0`. Since p10 is
  always null, the page renders **"Floor (p10): 0"** and **"Ceiling (p90): 0"** for every player, and draws a
  `PercentileFan` with p10=p90=0. A user reads that as the model predicting a zero floor.
- `frontend/src/pages/Dashboard.tsx:156-158` calls `row.floor.toFixed(0)` directly. `types/api.ts:66` declares
  `floor: number` (non-nullable) while the API returns `null` — so TypeScript does not catch it and this is a
  **runtime `TypeError` on every row** in artifact-backed mode. I grepped for a React error boundary
  (`ErrorBoundary`, `componentDidCatch`, `errorElement`) across `frontend/src`: **zero hits**, so nothing
  catches it. I did not load the page in a browser — this is derived from the type contract plus the observed
  `null` response.

**(c) Negative floors are served as fantasy floors.** 246 `fantasy_ppr` rows carry a negative
`fantasy_floor` (117 WR, 115 TE, 12 QB, 2 RB; worst −3.53). A PPR floor below zero is close to impossible for
a WR/TE — only a lost fumble scores negative. These are band artifacts presented in a percentile-named slot.

**(d) Uncertainty is not player-specific.** Across ~85k rows there are only **20-21 distinct**
`ceiling − projection` values per cell. The band is essentially constant within a calibration group, so the
system cannot distinguish a volatile boom/bust WR from a steady possession receiver — exactly the distinction
a fantasy manager wants.

### 2.3 Season-long — the most dangerous surface

`/projections/season/{season}` does **not use the trained models at all.** It reads `kalman_est_*` /
`kalman_variance_*` straight from `feature_matrix` (`projection.py:304-326`) and feeds them to the C++ season
simulator. The 74/75 edge result **does not apply here.**

I ran a calibration test: rest-of-season 2025 from week 10, projections vs realized PPR, scored over
**regular-season weeks 10-18 to match the simulator's own `end_week=18` horizon** (468 matched players):

```
Nominal 80% interval coverage (p10..p90):  10.5%     (should be ~80%)
median interval width:                     29.1 pts
median |actual − mean|:                    63.6 pts
```

(Including playoff weeks 10-22 gives 11.0% — the finding does not depend on the window.)

The intervals are roughly **half the width of the median error**. But the calibration failure is downstream
of a worse problem — **the point estimates are inverted at the top of the board:**

```
=== TOP 20 by projected rest-of-season PPR ===
     Joshua Dobbs  QB  294.1   actual   2.02
      Davis Mills  QB  292.7   actual  61.20
        Drew Lock  QB  290.0   actual  -0.40
     Kirk Cousins  QB  289.8   actual  95.26
  Jarrett Stidham  QB  288.3   actual   7.62
      Zach Wilson  QB  281.1   actual  -0.20
       Kyle Allen  QB  279.0   actual  -0.10
  Jimmy Garoppolo  QB  273.5   actual  -0.70
        … all 20 are backup quarterbacks …
```

**Mechanism, confirmed at the data level.** The Kalman filter's uninformative prior is `est = 20.0,
var = 1000.0`. 150 QB rows in 2025 sit at exactly 20.0. That prior is *higher* than every real starter except
four:

| Player | kalman_est | kalman_var |
|---|---|---|
| Patrick Mahomes | 24.82 | 5.50 |
| Lamar Jackson | 23.07 | 18.79 |
| Josh Allen | 22.14 | 12.26 |
| Jalen Hurts | 21.36 | 5.89 |
| **Drew Lock / Zach Wilson / Dobbs / Kyle Allen / Garoppolo** | **20.00** | **1000.00** |
| Aaron Rodgers | 16.97 | 6.29 |

The simulator then multiplies a per-game estimate by all remaining games with **no playing-time or role
model**, so a player with *no information* is projected as a top-5 fantasy QB every week for the rest of the
season. **Players with the least data receive the highest season projections.** The 1000-vs-5.5 variance gap
that should discount them is not used in the ranking.

Two supporting defects: `except Exception: return []` (lines 330-332, 385-387) silently converts any failure
into an empty result, and this path carries **no `artifact_backed` gate and no `degraded` flag** — the
response model has no field to report either.

**Verdict: do not use season-long projections for any decision.**

### 2.4 Draft board — honest about its source, broken in its advice

`/draft/board?season=2026` returns 127 players and correctly self-labels
`projection_source: "preseason_historical_per_game"`. That label is accurate: `ml/draft_projection.py` is a
recency-weighted historical PPR per-game mean (`0.70^years_ago`) times a games-played prior clamped to
[8, 17]. **It uses none of the trained models, no features, no stack.** The credit here is real — the source
is disclosed rather than dressed up as a model output. But the README's architecture narrative implies the
draft board is model-driven, and it is not.

Two defects make its output unsafe to act on:

**(a) No replacement-level adjustment — the value column is meaningless across positions.**

`value_vs_adp = adp_rank − model_rank`, a raw rank difference. The model ranks by *raw projected PPR points*
across all positions; ADP already embeds positional scarcity. QBs score the most raw points, so every QB looks
like an enormous value:

```
position mix, model top 24:   {QB: 12, RB: 7, WR: 5}
position mix, market top 24:  {RB: 12, WR: 10, TE: 2}

Patrick Mahomes   model_rank  10   adp_rank 105   → "95 slots of value"
Jalen Hurts       model_rank   7   adp_rank  65
Dak Prescott      model_rank  12   adp_rank  98
```

In a 1-QB league, drafting to this board means taking 12 quarterbacks in the first two rounds. There is no
VOR/VORP baseline anywhere in the draft path.

**(b) Rookies are unrankable by construction.** Players with zero prior games fall back to a position mean, so
the 7 rookies on the board share **3 distinct projections** (RB 62.5, WR 61.0, TE 46.1) and all land at
model_rank 120-126 — the bottom. Jeremiyah Love (ADP 20, a real second-round pick) is ranked 120th. The board
will tell you to fade every rookie, always. That is an artifact of having no rookie model, not a prediction.

**(c) Minor:** the underlying `draft_preseason_projections` table (2,538 rows) contains retired players — **Tom
Brady** carries a 157-point 2026 projection. The ADP join filters them out of the served board, but anything
reading the table directly gets them.

---

## 3. Release integrity: the gate cannot fail, by construction

The 07 re-audit states the candidate is **not** promoted and `current_baseline.json` is "deliberately not
re-frozen." **That is no longer true of this tree.** `releases/current_baseline.json` at HEAD reads:

```json
"model_version": "causal_20260819_zero_regression",
"release_status": "promoted_zero_regression",
"candidate": false
```

It was promoted by commits `53735fd` → `f7eb25f`, after that document was written. The promotion evidence is
`promotion_gates_zero_regression.json`: **75 gates, 75 passed.** Those gates cannot fail:

- The "frozen prior release" is `releases/baselines/20260819_bootstrap_causal_lgbm.json` — a bootstrap
  reference **created the same day from the same training run**, not a previously promoted release.
- `scripts/select_nonregressing_stacks.py` (policy: `zero_seasonal_regression_vs_walkforward_lgbm`) replaces
  the Ridge stack with an **LGBM passthrough** for any cell where the stack lost in ≥1 season.
- For those cells the candidate *is* the baseline, so the gate compares a number to itself:

```
fantasy_ppr/QB 2021:  baseline_mae 6.48355049407846
                      candidate_mae 6.48355049407846      → ok: true
```

Note this is not confined to the five substituted cells: because the "prior release" baseline was generated
from the *same training run* as the candidate, **all 75 gates are uninformative**, not just the 25 belonging to
substituted cells. The substitution merely makes five of them exactly tautological.

This is finding **C-08 from the original audit** — *"the frozen baseline is the candidate's own OOF,
bit-for-bit"* — reconstituted under a new name. 5 of 15 cells were substituted this way:

| Cell | Selection | Seasons the Ridge stack lost |
|---|---|---|
| fantasy_ppr:QB | **lgbm_identity** | 1/5 |
| fantasy_ppr:WR | **lgbm_identity** | 2/5 |
| passing_yards:QB | **lgbm_identity** | 2/5 |
| receiving_yards:WR | **lgbm_identity** | 1/5 |
| rushing_yards:QB | **lgbm_identity** | 1/5 |
| *(other 10)* | ridge_stack | 0/5 |

I confirmed the substitution is live: in `stack_fantasy_ppr_WR_20260819.csv`, `y_pred == lgbm_pred` for
**100.0%** of rows.

Two consequences worth stating plainly:

- **The advertised architecture is not what runs for those five cells.** The README describes four base
  learners stacked under a Bayesian layer; `fantasy_ppr/QB` and `fantasy_ppr/WR` — the two most important
  fantasy cells — are a **single LightGBM model** with `{lgbm: 1.0, catboost: 0.0, intercept: 0.0}`.
- **The selection rule is brittle.** "Discard the stack if it loses in *any* single season" optimizes against
  the worst season. `fantasy_ppr:WR` lost by 0.076 and 0.036 MAE in two seasons and the stack was dropped
  entirely.

To be fair to the design: the substitution is *recorded* in `CONSTRAINED_STACK_SELECTION.json`, and the
resulting model is never *worse* than LGBM. The problem is that a gate constructed this way provides **zero
evidence** and is labeled 75/75.

---

## 4. Stale and contradictory evidence still shipping

**(a) 21 pre-leak-fix evaluation CSVs are still in `reports/`, 13 of them git-tracked**, all dated Aug 6 or
Aug 9 13:xx — *before* the causal rebuild at `20260809T231500`. Aggregated, they assert **71/75 beats_naive**
from the contaminated model. The 07 re-audit deleted only two JSON summaries; these were left.

They are also internally inconsistent with the current artifacts — `eval_causal_stack_fantasy_ppr_WR.csv`
reports a 2021 model score of **2.662**, where the promoted candidate's own selection record gives **5.305**
for the same cell-season. They describe a different, contaminated object.

**(b) 5,469 pre-remediation projection rows are live and servable.** The `projections` table holds two runs:

| pipeline_run_id | rows | cells |
|---|---|---|
| `stack_materialize_20260809T171811Z` | **5,469** | 10 |
| `stack_materialize_20260819T151555Z` | 80,404 | 15 |

The 17:18 run predates the `rebuild_20260809T231500` causal rebuild and covers exactly the old 10-cell set,
so it is **almost certainly pre-leak-fix output** — that is an inference from timing and cell coverage, not
from a recorded provenance field, which is itself part of the problem. `_load_projection_row()`
selects `ORDER BY created_at DESC` with **no pipeline-run filter**; `projection_policy.approved_pipeline_run_ids`
is enforced only in `ml/backtest.py`, never in the serving path. Confirmed live:

```
GET /predict?player=Aaron Rodgers&week=1&season=2025&stat=fantasy_ppr
→ "fantasy_ppr": 14.51, "pipeline_run_id": "stack_materialize_20260809T171811Z", "degraded": false
```

A contaminated-model projection is served as authoritative, flagged `degraded: false`, distinguishable only by
reading the run id.

**(c) The README's model-status section is wrong in the pessimistic direction.** It headlines
`reports/yardage_diagnostic.json` — MAE 35.43 on **60 rows, one season, `temporal_ordered: false`,
`feature_actual_columns: []`** — and concludes the model is "slightly worse than a constant predictor." On the
same cell, my 5-season evaluation over 7,374 rows gives **receiving_yards/WR MAE 24.43**, beating both the
prev-season (25.3-26.8) and trailing-3 baselines in **all five seasons**. The README is drawing its headline
conclusion from a broken 60-row artifact.

**(d) The full test suite is red, and three of the four failures are hollow tests.**

The 07 re-audit reported "106 passed, 1 skipped" and "18 passed, 1 skipped" — both **curated subsets of six
files**. The full suite tells a different story:

```
4 failed, 1120 passed, 19 skipped
```

- `test_normalize.py::test_count` — asserts a hardcoded `19399` row count against live upstream data that now
  returns `19400`. Brittle, not a defect, but it keeps the suite red.
- **Three `TestStackingInferenceHelpers` tests are testing nothing at all.** They write malformed coefficient
  fixtures into a `tmp_path` `oof_dir` — but `InferenceClient.load_ridge_coefs` resolves through the *release
  manifest* first (`inference_client.py:505-511`), which points at the real promoted artifact and **ignores
  `oof_dir` entirely** for any declared cell. I confirmed this directly:

  ```
  manifest resolves receiving_yards/WR -> releases/candidates/causal_20260810/ridge_receiving_yards_WR_coefs.json
  malformed NaN fixture in oof_dir      -> ([1.0, 0.0], 0.0, ['lgbm','catboost'])   # fixture never read
  ```

  The tests' outcome is decided by the production release, not their fixtures — which is why promoting the
  candidate turned them red.

  **The validator itself is fine.** Pointed at a cell *not* in the manifest, so the fixture is actually read,
  it behaves correctly:

  ```
  NaN weight            -> correctly REJECTED ("JSON non-finite constant 'NaN' is forbidden")
  legacy 4-learner file -> correctly REJECTED ("Invalid Ridge coefficient schema")
  ```

  So the 2026-08-11 audit's praise of this validator stands. What does not stand is the **coverage**: the
  repo's malformation-rejection tests for this path are illusory, and they flip red/green with release state.

**(e) `/health` reports blocked while every prediction endpoint serves 200.**

```
503  /health            (overall_status: blocked, artifact_mode_ready: false)
200  /projections/week/10?season=2025
200  /projections/season/2025?start_week=10
200  /draft/board?season=2026
200  /predict?...
```

The fail-closed design covers `/health` and `/backtest` but not the surfaces that actually produce
predictions. (`artifact_mode_ready: false` is driven by MLflow being unreachable — an infrastructure
condition, not a model one.)

---

## 5. What is genuinely solid

Stated plainly, because it is substantial and it survived adversarial checking:

- **The C-01 leak fix holds.** Independently re-verified: `snap_pct_off` is non-null on **0** rows;
  `prior_snap_share` equals the *previous* game's `offense_pct` on **111,935/111,935 rows (100.0000%)**, and
  its 67.83% same-game match is exactly the column's own lag-1 autocorrelation.
- **The walk-forward structure is correct** — `max_train_season = season − 1` in every fold, no leakage rows,
  no duplicate player-games.
- **The evaluator is properly hardened.** `ml/eval_causal.py` now *raises* on missing `max_train_season`
  provenance rather than inventing it, applies an explicit `CohortSpec`, and scores model and both baselines
  on one common finite cohort. The old tautological `season − 1` default is gone.
- **PPR is never re-derived.** It is read from `game_logs.fantasy_points_ppr` and modeled directly as the
  `fantasy_ppr` target. There is no second scoring implementation to drift — the class of bug I went looking
  for in the fantasy translation **does not exist here.**
- **All 15 declared cells are served** (C-10 closed), and the Ridge coefficient artifacts load
  (NEW-01 closed).
- **The draft board discloses its own source** rather than implying model backing.

---

## 6. What to fix, in priority order

**Blocking for fantasy use:**

1. **Correct the headline.** Fix the denominators in `write_post_leak_fix_comparison.py` (count per
   cell-season, or label the numerator per-cell), regenerate the report, and update the README's model-status
   section. The system is materially better than its own documentation claims.
2. **Do not ship the season-long surface.** Add a playing-time/role model, or at minimum suppress players
   whose `kalman_variance` indicates the uninformative prior (`var ≈ 1000`). Until then the endpoint should
   fail closed rather than rank backups first.
3. **Add replacement-level (VOR) normalization to the draft board**, or remove `value_vs_adp`. Cross-position
   rank comparison against ADP is invalid without it.
4. **Delete or quarantine the 5,469 pre-leak-fix projection rows**, and filter the serving path by
   `approved_pipeline_run_ids` the way `ml/backtest.py` already does.

**Blocking for trusting the numbers:**

5. **Make the promotion gate capable of failing.** A baseline generated from the same run as the candidate,
   combined with identity substitution, cannot produce evidence. Either capture a genuine prior release or
   state explicitly that this is a first release with no regression evidence.
6. **Teach `_interval_method` about conformal bounds** so the computed intervals reach `floor`/`ceiling` with
   an honest label — and apply the same guard to `fantasy_floor`/`fantasy_ceiling`, which currently bypass it.
7. **Delete the 21 stale `reports/eval_causal_stack_*.csv`** asserting 71/75 from the contaminated model.
8. **Fix the three hollow `TestStackingInferenceHelpers` tests** so their fixtures are actually read (point
   them at an undeclared cell, or inject a manifest stub). As written they assert nothing and go red whenever
   a release is promoted. Replace the hardcoded `19399` in `test_normalize.py` with a tolerance or a
   pinned snapshot.

**Correctness:**

9. Fix `Dashboard.tsx` (`row.floor.toFixed(0)` on null) and `PlayerDetail.tsx` (`?? 0` rendering a fabricated
   zero floor); make `floor`/`ceiling` nullable in `types/api.ts`.
10. Clamp fantasy floors at 0 for non-QB positions, or expose the interval method alongside them.
11. Exclude retired players from `draft_preseason_projections`; give rookies a real prior or mark them
    explicitly unranked.
12. Either wire up or delete the dead modules — `markov_simulator`, `gnn_matchup`, `rl_hedging_agent`,
    `season_simulator_bridge` — and stop describing play-by-play as a prediction surface.

---

## 7. Answering the question directly

**Can you trust it for fantasy decisions today?**

- **Weekly player projections (point estimates): yes.** This is a real model with real, causally-validated
  edge over the baselines that matter — better than the repo says. Use the projection number.
- **Any uncertainty it shows you: no.** Floors and ceilings are either suppressed, rendered as 0, or
  constant-width; the negative floors are artifacts. Ignore the ranges.
- **Season-long projections: no.** Inverted at the top of the board, and nominal 80% intervals cover 10.5% of outcomes.
- **The draft board: not as ranked.** The underlying per-game history is reasonable, but the cross-position
  value column will systematically push you to draft quarterbacks far too early and to fade every rookie.
- **Play-by-play: there is nothing to trust or distrust.** It does not exist.

The 2026-08-11 "DO NOT SHIP" verdict still holds for the season and draft surfaces — but for **different
reasons than that audit gave**, and its central claim that the model lacks edge is wrong.
