# 02 · CLAUDE CODE · Feature contract + full retrain

| | |
|---|---|
| **Order** | Wave 2 — after wave 1 |
| **Concurrency** | ⛔ **SOLO** — blocks everything. Nothing else runs while this does. |
| **Depends on** | `01A` merged, `01B` report in hand |
| **Blocks** | `03A`, `03B`, `03C`, `04`, `05`, `06` |
| **Findings** | **C-01** (the finding that invalidates everything else), C-18, C-26 |
| **Runtime** | **hours.** Full feature rebuild + retrain of 15 cells. Start it when you can leave it. |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree remove ../go_01A_containment && git worktree remove ../go_01C_hygiene   # merge first
git worktree remove ../go_01B_audit
git checkout main && git pull                       # wave 1 merged
git checkout -b fix/02-feature-contract
```

**Run in the main tree** — this session needs `.venv_311`, MLflow and Postgres, and no one else is
running. Verify the antidote archive first:

```bash
cd /Users/risshabh/Projects/Active/_gridiron_archive/antidote_20260809_stacks && sha256sum -c MANIFEST.sha256
```

> **This prompt has a hole in it by design.** Paste the `01B` findings table where marked. Do not run this
> session without it — the scope of the rebuild is exactly what `01B` determines.

---

You are fixing the finding that invalidates everything else in `gridiron_oracle`: **C-01, target-game
feature leakage**. Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` §1 and the C-01 row first.

## The defect

`snap_pct_off` leaks the game being predicted, through two channels:

1. **Feature value** — `pipeline/feature_engineer.py:165`: `snap_pct = target_row.get("offense_pct")`,
   assigned at `:258`, member of default `FEATURE_COLS` at `ml/utils.py:169`.
2. **Cohort filter** — `ml/utils.py:382` and `:459` apply `df[snap.isna() | (snap >= 0.34)]`. The training
   *and* evaluation population is defined by target-game participation.

DB-confirmed: 125,425/125,425 non-null values equal same-game `game_logs.offense_pct`; correlation with
`actual_fantasy_ppr` = 0.637.

Channel 2 is the worse one. Three independent reviews "validated" the headline results by recomputing on
strict common cohorts — all drawn from this filtered population. Expanding-window season folds do not
cure within-game leakage.

## Additional leaks found by the 01B as-of audit

## 01B as-of audit findings — paste block

**Audit basis:** commit `191e151`, read-only DB/code trace.  Treat every
`UNPROVEN` item as illegal until the implementation proves a timestamped,
pre-kickoff source snapshot.  `LATENT LEAK` items are null/inert in the current
feature matrix but become illegal as soon as their existing refresh/enrichment
path populates them.

| Column / consumer / filter | Defined at | Source row is | Verdict | Evidence and required contract disposition |
|---|---|---|---|---|
| `temp_f` | `pipeline/features/buckets.py:301-350` | as-of unknown | UNPROVEN | `games.temp` has no capture-time/forecast provenance. Permit only a timestamped pre-kickoff forecast; otherwise remove. |
| `wind_mph` | `buckets.py:301-350` | as-of unknown | UNPROVEN | Same `games.wind` provenance gap. |
| `temp_bucket` | `buckets.py:301-350` | as-of unknown | UNPROVEN | Inherits `temp_f` provenance gap. |
| `wind_bucket` | `buckets.py:301-350` | as-of unknown | UNPROVEN | Inherits `wind_mph` provenance gap. |
| `injury_status_encoded` | `buckets.py:407-471` | as-of unknown | UNPROVEN | Same-week ESPN status has no stored report timestamp/selection rule. It is 0/129,128 non-null today because `FeatureEngineer.run()` does not pass `injury_df`. |
| `snap_pct_off` | `pipeline/feature_engineer.py:165,259` | target game | LEAK | Direct `target_row.offense_pct`. DB: 125,425/125,425 exact same-game matches; corr with `actual_fantasy_ppr` = 0.63746. Remove/replace only with an explicitly lagged feature. |
| `blitz_exposure` (non-default) | `buckets.py:506-584` | target game | LATENT LEAK | `snap_pct_off × opp_blitz_rate`; currently null solely because opponent blitz is null. Prohibit the target snap operand. |
| `height` | `feature_engineer.py:178-194` | as-of unknown | UNPROVEN | Reads current `players.height` without historical effective dates. It is null today due string-to-float coercion, not because it is safe. |
| `weight` | `feature_engineer.py:178-194` | as-of unknown | UNPROVEN | 129,103 FM values exactly equal current `players.weight`; no historical snapshot. Allow only a dated roster value or remove. |
| `epa_per_play` | `pipeline/pbp_pipeline.py:232-472,709-750` | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `epa_per_target` | same | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `epa_per_rush` | same | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `qb_epa_per_dropback` | same | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `adot` | same | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `yac_per_reception` | same | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `xyac_per_reception` | same | target game | LATENT LEAK | Same-game PBP aggregate written by `(player_id, game_id)` update. |
| `target_share_pbp` | same | target game | LATENT LEAK | Same-game player/team target share; source table already has 13,323 non-null values. |
| `air_yards_share_pbp` | same | target game | LATENT LEAK | Same-game player/team air-yards share. |
| `red_zone_targets` | same | target game | LATENT LEAK | Same-game outcome count. |
| `end_zone_targets` | same | target game | LATENT LEAK | Same-game outcome count. |
| `red_zone_target_share` | same | target game | LATENT LEAK | Same-game player/team red-zone share. |
| `drop_rate` | same | target game | LATENT LEAK | Same-game PBP/FTN/PFR drops divided by targets. |
| `ol_pressure_rate` | same | target game | LATENT LEAK | Same-game team hits/dropbacks. |
| `ol_sack_rate` | same | target game | LATENT LEAK | Same-game team sacks/dropbacks. |
| `pass_left_rate` | same | target game | LATENT LEAK | Same-game target direction/targets. |
| `pass_middle_rate` | same | target game | LATENT LEAK | Same-game target direction/targets. |
| `pass_right_rate` | same | target game | LATENT LEAK | Same-game target direction/targets. |
| `wind_x_qb` | `feature_engineer.py:204-212` | as-of unknown | UNPROVEN | Inherits `wind_bucket` provenance gap. |
| `wind_x_wr` | `feature_engineer.py:204-212` | as-of unknown | UNPROVEN | Inherits `wind_bucket` provenance gap. |
| `precip_x_pass` | `feature_engineer.py:204-212` | as-of unknown | UNPROVEN | `games.precipitation_bucket` has no as-of provenance; null today. |
| `depth_chart_rank` | `feature_engineer.py:793-803` | as-of unknown | UNPROVEN | Same-week join to `depth_charts`; 104,437 exact matches but no publication timestamp establishes a pregame snapshot. |
| `avg_separation` | `feature_engineer.py:804-814` | target game | LATENT LEAK | Same `(player,season,week)` Next Gen Stats postgame update; 4,207 source rows exist, FM is null today. |
| `avg_cushion` | `feature_engineer.py:804-814` | target game | LATENT LEAK | Same target-week NGS postgame join. |
| `player_emb_0` | `pipeline/enrich_elo_embeddings.py:205-344` | as-of unknown | LATENT LEAK | Global current-player embedding uses `CURRENT_DATE` age/current `years_exp`, then writes every historical row by player. |
| `player_emb_1` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `player_emb_2` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `player_emb_3` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `player_emb_4` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `player_emb_5` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `player_emb_6` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `player_emb_7` | same | as-of unknown | LATENT LEAK | Same global historical overwrite. |
| `years_exp` (opt-in `progression`) | `buckets.py:877-927` | as-of unknown | LEAK | Current `players.years_exp` is copied to all history: 129,115 rows; every one of 4,357 players has exactly one historical value. |
| `exp_bucket` (opt-in `progression`) | `buckets.py:909-918` | as-of unknown | LEAK | Direct deterministic descendant of leaked `years_exp`. |
| training snap cohort | `ml/utils.py:377-388` | target game | LEAK | `snap_pct_off >= .34` retains 37,268/129,128 FM rows and removes 91,860 on target participation. Delete this predicate. |
| backtest/eval snap cohort | `ml/eval_cohort.py:56-82`; `ml/run_backtest.py:117-121` | target game | LEAK | Target `game_logs.offense_pct >= .34` determines evaluation membership. Delete/replace with a pregame cohort rule. |
| `routes_run_pct` (non-default) | `feature_engineer.py:213-215,285` | target game | LEAK | Another direct copy of `target_row.offense_pct`; prohibit it or create an explicitly lagged substitute. |
| TFT `snap_share` consumer | `ml/tft_model.py:355-367` | target game | LEAK | Maps `snap_pct_off.fillna(0)` into TFT time-varying input. |
| GNN `snap_pct_off` node feature | `ml/gnn_matchup.py:85-103` | target game | LEAK | Explicit GNN feature-list consumer outside `FEATURE_COLS`. |

### Required additions to Part 1: TFT and GNN are not fixed by changing `FEATURE_COLS` alone

**No: “remove `snap_pct_off` from `FEATURE_COLS` or replace it with a lagged equivalent” is not sufficient by itself.**

Add these requirements to the feature contract and implementation plan:

1. Define exactly one approved lagged participation feature (for example
   `snap_pct_off_lag1` / `prior_snap_share`), computed only from completed
   player games before the prediction game.  It must have an explicit as-of
   registry declaration.
2. Prohibit the raw names `snap_pct_off`, `offense_pct`, and `routes_run_pct`
   from every model-input allowlist, not only `FEATURE_COLS`.  Add a contract
   assertion that fails before train/eval/inference if a target-game snap field
   reaches a model frame.
3. In `ml/tft_model.py`, remove the `snap_share ← snap_pct_off` mapping.  Map
   `snap_share` only from the approved lagged field, or omit it from TFT’s
   time-varying reals.  Do **not** silently fill a removed raw feature with zero:
   fail the contract or intentionally configure the field out.
4. In `ml/gnn_matchup.py`, remove `snap_pct_off` from `OFF_NODE_FEATURES` and
   substitute only the approved lagged feature (or reduce the feature dimension
   consistently).  Validate the GNN node dataframe against the same registry.
5. Search all model adapters, inference builders, serving feature maps, and
   derived aliases for the raw fields before training.  `routes_run_pct` and
   `blitz_exposure` are already-known aliases/descendants that must be covered.
6. Rebuild every existing TFT and GNN artifact that was trained or evaluated
   with either raw snap path: feature matrix first, then that model’s base OOF,
   then any dependent stack/evaluation/promotion/projection output.  A default
   `FEATURE_COLS` change cannot retrospectively cleanse their artifacts.

### Required additions to Part 1: default embedding path

The eight `player_emb_*` columns are in `FEATURE_COLS`; they are currently null
but an enrichment run would make the default set non-causal.  Their current
implementation is not trained on the model feature matrix; it instead learns a
global profile from the current `players` table, including `CURRENT_DATE` age
and current `years_exp`, and broadcasts one vector across historical rows.

Add these requirements:

1. Do not run `pipeline.enrich_elo_embeddings.enrich_embeddings()` until the
   embedding source contract is fixed.  Remove current-date and current-snapshot
   fields from its input, or version the embedding by `(player_id, season,
   as_of_week)` using only information available then.
2. Feature registration must mark every embedding input and the resulting
   `player_emb_*` columns with their as-of source.  A single player-level vector
   may be used historically only when all of its inputs are immutable and were
   known before the relevant game.
3. Delete/overwrite any existing embedding store and FM embedding values only
   as part of the new dated feature rebuild; do not reuse a global old store.
4. If any learned embedder (player profile, GNN, PCA, or another representation)
   was fit from a feature frame containing leaked snap/PBP/NGS fields, it is
   itself an invalid artifact even when its output is not named `snap_pct_off`.
   Refit it after the causal feature matrix is built and **before** every model
   that consumes its vectors.  Then regenerate that model’s base OOF, stacks,
   evals, promotion evidence, and projections.

Required order for an embedding-enabled run:

`causal source snapshots → causal feature matrix → causal embedding fit/store → embedding enrichment → base OOF → stacks → evals → promotion evidence → materialized projections`.

### C-18 scope narrowing

C-18 is **opt-in only**.  The `game_id`-only aggregation bug in
`build_season_feature_context()` corrupts the following opt-in columns:

`carry_share`, `carry_share_vs_league`, `opp_adj_target_share`, `team_pace`,
`team_pass_rate`, `expected_pass_attempts`, and `expected_pass_rate`.

It does **not** contaminate the default `FEATURE_COLS`, and it is not a
target-game time leak: its inputs are restricted to prior weeks.  Therefore it
does not enlarge the mandatory default base-model retrain beyond C-01 and the
default leaks above.  It *does* require rebuilding any experiment, OOF, stack,
evaluation, promotion evidence, or projection produced with `opp_adj_usage` or
`pace_script` enabled.  The separate opt-in `progression` group is likewise
outside the default scope, but its `years_exp` / `exp_bucket` leak invalidates
any run that enabled it.

### What must be rebuilt

The shipped default pipeline is invalid because its feature values and its
model/evaluation cohorts use target-game participation.  Strict-common-cohort
comparisons do not repair this, because those cohorts were drawn from the same
leaked population.

Required regeneration order:

1. **Features:** rebuild every affected `feature_matrix` season after removing
   target-game snap values and the target-game snap eligibility predicate.
   Keep PBP, NGS, and embeddings disabled until their causal contracts above
   are implemented; otherwise a rebuild promotes their latent leaks to live.
2. **Base OOF:** retrain every default base learner and regenerate all OOF files
   from the causal, unfiltered matrix.  Include TFT/GNN artifacts whenever
   those paths were used.
3. **Stacks:** rebuild every meta/stack model from replacement base OOF.
4. **Evaluations:** rerun causal, cohort, and baseline evaluations with a cohort
   independent of target-game participation.
5. **Promotion evidence:** regenerate comparison tables, model cards, manifests,
   thresholds, and all selection evidence derived from old OOF.
6. **Materialized projections:** refresh only from the newly promoted artifacts.

The C-18 and `years_exp`/`exp_bucket` findings do not require retraining the
default feature set unless an affected opt-in group was actually enabled.

## Guardrails

1. **Write new artifacts under a new date stamp. Do not clobber `_20260809`.** Those are the only
   reproducible record of current serving behaviour, archived at
   `/Users/risshabh/Projects/Active/_gridiron_archive/antidote_20260809_stacks/`.
2. **Confirm `01A` landed before running any trainer.** `ml/train_all_models.sh` must no longer purge OOF
   CSVs and the trainers must no longer train XGB/TFT. Check by reading, not by trusting. If either is
   still present, **stop** — you are about to destroy the artifact set.
3. Do not touch serving, schema, gates or the draft API. Those are waves 3 and 4.

## Part 1 — Define and enforce an as-of feature contract

This is the durable fix. Removing one column is not.

- Introduce an explicit **as-of** contract: every feature declares the latest point in time from which it
  may draw data, relative to the prediction game's kickoff. Represent it in the feature registry
  (`ml/utils.py` / `ml/feature_groups.py`) — not as a comment.
- Legal: prior games, prior seasons, pregame-known facts (schedule, venue, static player attributes as-of
  that season, injury/practice reports published before kickoff).
- Illegal: anything read from the target row or from a same-`game_id` postgame join.
- **Add an enforcement test** that fails the build when a feature correlates suspiciously with a same-game
  postgame field. Cheapest robust version: for every feature column, assert it is not exactly equal to any
  same-game `game_logs` column across the matrix. **That single assertion would have caught C-01 on day one.**

## Part 2 — Remove the leaks

- **`snap_pct_off` as a feature**: remove from `FEATURE_COLS`, or replace with a lagged equivalent
  (prior-game or trailing-N snap share). Prefer lagged — snap share carries real signal, it just has to
  come from games already played.
- **`snap_pct_off` as a filter** (`ml/utils.py:382`, `:459`): replace with a pregame-knowable eligibility
  rule. Options — prior-game snap share threshold, roster/depth-chart status at kickoff, games-played-to-date.
  Pick one and justify it. **Training and eval filters must use the same rule**, and the rule must be
  stated in the eval report.
- **Derived columns inheriting the leak**: `compute_scheme_interactions(form, matchup, snap_pct)` at
  `feature_engineer.py:170` receives the leaked value directly; `ml/utils.py:177` documents
  `blitz_exposure = snap_pct_off × opp_blitz_rate`. Trace every interaction term; fix or drop.
- **Everything the 01B table flagged, including LATENT items.** PBP (`pipeline/pbp_pipeline.py:232-275`,
  `:730-734`) and NGS (`feature_engineer.py:804-810`) are currently null-and-inert; if their joins are
  same-game, **fix them now**. Leaving a live trap that arms itself on the next data refresh is not
  acceptable.

## Part 3 — Fold in the two feature-definition bugs

Do these here, not later — you are rebuilding features anyway, and deferring means rebuilding twice.

- **C-18** — `pipeline/features/buckets.py:606-652` keys team carries/pass/rush/targets by `game_id` alone,
  combining both teams and repeatedly overwriting `game_team[gid]`; `compute_usage_shares:709-739` consumes
  that combined denominator. Re-key **all** team-game aggregates by `(game_id, team)`. The existing fixture
  (`backend/tests/test_phase4_phase6.py:49-74`) has only one team, which is why nothing caught it — **add a
  two-team fixture.**
- **C-26** — `years_exp` is a present-day roster snapshot applied to every historical row (2,363 players
  constant across seasons). Compute as-of season. Check the other roster-derived statics `01B` flagged
  (`age`, `career_games`, draft fields, height/weight) for the same bug.

## Part 4 — Rebuild, in order

Nothing downstream is valid until the step above it is redone.

1. Rebuild `feature_matrix` from scratch. **Do not reuse cached features.** Verify row count and season
   bounds (≤2025) afterward.
2. Retrain every base OOF — LGBM and CatBoost only, all 15 cells.
3. Restack — two-learner, via the `01A` manifest path, new date stamp.
4. Regenerate every eval report from code.
5. Regenerate gate/promotion evidence. The gates themselves are still broken (session `04`) — treat this
   output as data, not as a pass.
6. **Do not materialize to the DB.** That is `03A`, and it depends on manifest and uncertainty fixes that
   have not landed.

## Part 5 — Report honestly

Produce `reports/POST_LEAK_FIX_COMPARISON.md`:

- Old vs. new headline counts (fantasy_ppr 20/20, volume 25/25, passing 5/5, yardage 24/25) on the **new**
  cohort, with the cohort definition stated.
- Per-cell MAE before and after.
- The new eligibility rule and how many rows it admits vs. the old snap filter.
- A plain statement of how much of the previous edge was leakage.

### Do not use "identical counts" as the did-it-happen check

An earlier version of this prompt said: *if the counts come back identical, the retrain did not happen.*
**That check is now ambiguous and must not be used.** Since `01A` landed, artifact selection is pinned to
`releases/current_baseline.json` and fails closed — so **two opposite failures produce the same symptom:**

| Symptom | Cause A | Cause B |
|---|---|---|
| Old numbers reappear | Retrain silently reused cached features / stale artifacts | Retrain worked perfectly, but every reader still resolves the **old** `_20260809` artifacts because the manifest was not re-frozen (that is `03A`'s job, deliberately not yours) |

Cause B is the *expected* state at the end of this session. A number cannot distinguish them.

### Check the artifact, not the number

Verify the rebuild happened by reading the new artifacts directly. All three of these must hold, and each
is a property of the data rather than of a metric:

1. **`feature_matrix` actually changed.** Record row count, season bounds and column set before and after
   the rebuild, and diff them. Removing the snap-based cohort filter changes the row count — state the
   before/after numbers explicitly. An unchanged row count is the real signal that features were reused.
2. **The cohort changed size.** Report how many rows the new pregame-knowable eligibility rule admits vs.
   the old `snap_pct_off >= 0.34` filter, as an absolute count and a percentage. Identical cohort sizes
   mean the old filter is still in force somewhere.
3. **`snap_pct_off` is absent.** Assert it is in neither `FEATURE_COLS` nor any rebuilt artifact's columns,
   and that no remaining feature column equals a same-game postgame field. This is the Part 1 test, run
   against the rebuilt data rather than against the schema.

Additionally, confirm the new base OOFs and stacks carry the **new date stamp** and that their
`fold_idx → season` map matches the current plan (fold 0 → 2020), so the comparison is not silently
reading a `_20260809` file.

### Load new artifacts by explicit path, bypassing manifest selection

For the comparison itself, **do not** go through `ml.artifact_manifest` / `get_manifest()` — it resolves
only manifest-pinned paths and will hand you the old `_20260809` artifacts, or raise
`ManifestEntryMissing` for a cell whose new stamp is not pinned. Both are correct behaviour and neither is
what you want here.

Instead, read the new files by explicit path, the same way the stacker's `--oof` mode does:

```bash
# Enumerate the newly written artifacts explicitly; do not glob-and-take-newest.
ls -1 ml/oof/stack_*_<NEW_STAMP>.csv
```

```python
# Comparison harness: explicit paths on both sides, no manifest, no mtime.
import pandas as pd
old = pd.read_csv("ml/oof/stack_fantasy_ppr_WR_20260809.csv")   # pinned, pre-fix
new = pd.read_csv(f"ml/oof/stack_fantasy_ppr_WR_{NEW_STAMP}.csv")  # explicit, post-fix
```

Restacking is unaffected — `python -m ml.stacking_ensemble --oof <lgbm> <catboost>` and `--oof-dir` both
work without the manifest, and the two-learner allowlist is enforced regardless. Only *reader* paths are
manifest-gated.

**Do not re-freeze the manifest to make your new artifacts visible.** Registering them is `03A`'s step and
depends on the uncertainty and materialization fixes. State in your report that the new artifacts are
written and verified but not yet served, and list their paths and SHA-256 digests so `03A` can pin them
without recomputing anything.

Two `ml/oof/` hygiene notes for this session, both enforced by `01A`'s locks:

- A restack **raises** if any killed-learner OOF (`xgb_*`, `tft_*`) is present in `ml/oof/`. The directory
  was swept clean post-merge; if a trainer or test writes one back, remove it rather than working around
  the error.
- Trainers must be given `--out-dir`. `ml/{lgbm,catboost}_model.py` default to the real `ml/oof/`, which is
  how test fixtures previously landed beside the release artifacts.

**A worse result is the expected and acceptable outcome.** Do not tune to recover the old numbers — they
were not real. If the model no longer beats the baselines, that is the finding, and it is worth more than
a number that was never true.

## Done criteria

- No feature column equals any same-game postgame field — enforced by test, not inspection.
- No row filter anywhere in the train/eval path keys on a target-game quantity.
- Every `01B` LEAK / LATENT LEAK / UNPROVEN row resolved and marked.
- Team-game aggregates keyed by `(game_id, team)`, two-team fixture passing.
- `years_exp` and other statics computed as-of season.
- Full rebuild complete under a new date stamp; `_20260809` untouched; archive still verifies.
- `POST_LEAK_FIX_COMPARISON.md` written, including the honest delta.
- The did-it-happen evidence is **artifact-based, not metric-based**: `feature_matrix` row-count/column
  diff, cohort-size change under the new eligibility rule, and `snap_pct_off` absence — not "the counts
  changed".
- New artifacts listed with paths and SHA-256 digests, explicitly flagged as **written but not yet served**,
  for `03A` to pin. The manifest is not re-frozen in this session.
