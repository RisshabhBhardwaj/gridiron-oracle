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
