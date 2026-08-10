# As-of feature audit — commit `191e151`

**Scope.** Read-only audit of the default `FEATURE_COLS` (111 columns), the three
opt-in Phase-4 groups (17 columns), and every row-removing predicate in the
training/evaluation load paths.  `LEAK` means the shipped value or cohort uses
information from the game being predicted.  `LATENT LEAK` means the current DB
value is null/inert but the existing refresh path would write non-causal data.
`UNPROVEN` is intentional: the repository has no timestamped source snapshot
that can establish pre-kickoff availability.

## Result first

1. **Live, default leak:** `snap_pct_off`, plus the default training and
   evaluation cohort filters based on it.  DB evidence: all **125,425 / 125,425**
   non-null feature values equal `game_logs.offense_pct` for the same
   `(player_id, game_id)`; `corr(snap_pct_off, actual_fantasy_ppr) = 0.63746`.
   The filter retains only **37,268 / 129,128** rows and removes **91,860** using
   that target-game value.
2. **Default latent leaks, ranked by likely effect:**
   - The 18 direct player/team PBP outcome fields (highest).  `pbp_features` already
     contains 16,326 rows (15,876 non-null `epa_per_play`, 13,323 non-null
     `target_share_pbp`), and `pbp_pipeline._update_feature_matrix()` writes
     them to the matching target `(player_id, game_id)` feature row.
   - `avg_separation` and `avg_cushion` (Next Gen Stats) (high for receivers).
     The post-build update joins the target `(player_id, season, week)`;
     4,207 NGS source rows exist, while the target FM columns are presently null.
   - `blitz_exposure` (not default / not a registered Phase-4 group) (high once
     coverage PBP is populated): it directly multiplies leaked target snap share.
   - `player_emb_0`–`player_emb_7` (medium): the opt-in enrichment script builds
     a single current-player embedding using `CURRENT_DATE` age and current
     `years_exp`, then writes it to every historical row for that player.
3. **C-18 is confirmed and confined to opt-in groups.** `game_id`-only aggregate
   keys in `build_season_feature_context()` combine both teams' game totals and
   overwrite the team mapping.  This corrupts several opt-in usage/pace values,
   but the affected inputs are restricted to weeks `< target_week`; it is not a
   target-game time leak.  No equivalent accidental two-team key was found in
   the default-feature builders.

## Default features (exact `FEATURE_COLS` order)

### Buckets 1–2 — player history

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| kalman_est_receiving_yards | `feature_engineer.py:146` | Kalman over `receiving_yards` | prior game | SAFE | `_compute_player_features` passes `p_sorted[:i]` (`:585-593`). |
| kalman_est_receiving_tds | `:146` | Kalman over `receiving_tds` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_targets | `:146` | Kalman over `targets` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_receptions | `:146` | Kalman over `receptions` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_target_share | `:146` | Kalman over `target_share` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_red_zone_target_share | `:146` | Kalman state | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_air_yards_share | `:146` | Kalman over `air_yards_share` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_fantasy_ppr | `:146` | Kalman over `fantasy_points_ppr` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_carries | `:146` | Kalman over `carries` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_rushing_yards | `:146` | Kalman over `rushing_yards` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_rushing_tds | `:146` | Kalman over `rushing_tds` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_pass_attempts | `:146` | Kalman over `attempts` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_completions | `:146` | Kalman over `completions` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_interceptions | `:146` | Kalman over `passing_interceptions` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_fumbles | `:146` | Kalman over summed fumbles | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_passing_yards | `:146` | Kalman over `passing_yards` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| kalman_est_passing_tds | `:146` | Kalman over `passing_tds` | prior game | SAFE | Same strictly preceding `prior_rows`. |
| seas_games_played | `buckets.py:56-136` | `len(prior_rows)` | prior game | SAFE | Builder passes only `p_sorted[:i]`. |
| seas_avg_receiving_yards | `:56-136` | `receiving_yards` mean | prior game | SAFE | `compute_season_baseline(snap_prior_rows)`. |
| seas_avg_targets | `:56-136` | `targets` mean | prior game | SAFE | Same. |
| seas_avg_receptions | `:56-136` | `receptions` mean | prior game | SAFE | Same. |
| seas_avg_target_share | `:56-136` | `target_share` mean | prior game | SAFE | Same. |
| seas_avg_fantasy_ppr | `:56-136` | `fantasy_points_ppr` mean | prior game | SAFE | Same. |
| seas_yards_per_target | `:56-136` | prior yards / targets | prior game | SAFE | Same. |
| seas_yards_per_reception | `:56-136` | prior yards / receptions | prior game | SAFE | Same. |
| seas_avg_carries | `:56-136` | `carries` mean | prior game | SAFE | Same. |
| seas_avg_rushing_yards | `:56-136` | `rushing_yards` mean | prior game | SAFE | Same. |
| seas_yards_per_carry | `:56-136` | prior rush yards / carries | prior game | SAFE | Same. |
| seas_avg_attempts | `:56-136` | `attempts` mean | prior game | SAFE | Same. |
| seas_avg_passing_yards | `:56-136` | `passing_yards` mean | prior game | SAFE | Same. |
| seas_completion_pct | `:56-136` | prior completions / attempts | prior game | SAFE | Same. |
| seas_avg_passing_cpoe | `:56-136` | `passing_cpoe` mean | prior game | SAFE | Same. |
| seas_avg_passing_epa | `:56-136` | `passing_epa` mean | prior game | SAFE | Same. |
| seas_avg_receiving_epa | `:56-136` | `receiving_epa` mean | prior game | SAFE | Same. |
| seas_avg_rushing_epa | `:56-136` | `rushing_epa` mean | prior game | SAFE | Same. |
| seas_avg_racr | `:56-136` | `racr` mean | prior game | SAFE | Same. |
| seas_avg_wopr | `:56-136` | `wopr` mean | prior game | SAFE | Same. |
| seas_avg_receiving_yac | `:56-136` | prior YAC / receptions | prior game | SAFE | Same. |

### Buckets 3–8 — matchup, schedule, availability

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| opp_avg_receiving_yards_allowed | `buckets.py:143-250` | opponent-facing `receiving_yards` | prior game | SAFE | `int(r.week) < target_week`. |
| opp_avg_targets_allowed | `:143-250` | opponent-facing `targets` | prior game | SAFE | Same week restriction. |
| opp_avg_tds_allowed | `:143-250` | opponent-facing `receiving_tds` | prior game | SAFE | Same week restriction. |
| opp_avg_fantasy_ppr_allowed | `:143-250` | opponent-facing `fantasy_points_ppr` | prior game | SAFE | Same week restriction. |
| opp_avg_rushing_yards_allowed | `:143-250` | opponent-facing `rushing_yards` | prior game | SAFE | Same week restriction. |
| temp_f | `buckets.py:301-350` | `games.temp` | as-of unknown | UNPROVEN | Stored schedule row has no capture-time/forecast provenance; actual weather may have been backfilled. |
| wind_mph | `:301-350` | `games.wind` | as-of unknown | UNPROVEN | Same missing as-of proof. |
| is_dome | `:301-350` | `games.roof` | pregame-known | SAFE | Venue roof is static. |
| surface_turf | `:301-350` | `games.surface` | pregame-known | SAFE | Venue surface is static. |
| temp_bucket | `:301-350` | bucket(`games.temp`) | as-of unknown | UNPROVEN | Inherits `temp_f` provenance gap. |
| wind_bucket | `:301-350` | bucket(`games.wind`) | as-of unknown | UNPROVEN | Inherits `wind_mph` provenance gap. |
| game_total_line | `buckets.py:390-405` | `games.total_line` | pregame-known | SAFE | Market total is available before kickoff; no target stat is read. |
| spread_line | `:390-405` | `games.spread_line` | pregame-known | SAFE | Market spread is available before kickoff; no target stat is read. |
| is_home | `:390-405` | scheduled home/away teams | pregame-known | SAFE | Static schedule assignment. |
| days_rest | `buckets.py:355-385` | `games.home_rest` / `away_rest` | pregame-known | SAFE | Calendar-derived prior-game interval. |
| is_short_week | `:355-385` | `days_rest < 6` | pregame-known | SAFE | Derived only from schedule rest. |
| is_bye_prior | `:355-385` | `days_rest >= 12` | pregame-known | SAFE | Derived only from schedule rest. |
| rule_coeff | `buckets.py:477-501` | season rule table | pregame-known | SAFE | Pure deterministic function of `season`. |
| injury_status_encoded | `buckets.py:407-471` | same-week ESPN `practice_status` | as-of unknown | UNPROVEN | `FeatureEngineer.run()` never supplies `injury_df` (DB: 0/129,128 non-null); no report timestamp/selection rule proves a pre-kickoff snapshot. |
| games_missed_streak | `:438-449` | absent prior player game logs | prior game | SAFE | Iterates `target_week-1` down; uses only `prior_rows`. |
| snap_pct_off | `feature_engineer.py:165,259` | `target_row.offense_pct` | target game | LEAK | DB: exact same-game equality 125,425/125,425; corr PPR 0.63746. |

### Bucket 9 — defensive tendency and interactions

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| opp_zone_pct | `buckets.py:190-205` | latest prior populated opponent PBP tendency | prior game | SAFE | `opp_rows` is restricted to `< target_week`; currently 0/129,128 non-null. |
| opp_man_pct | `:190-205` | latest prior populated opponent PBP tendency | prior game | SAFE | Same restriction; currently null. |
| opp_blitz_rate | `:190-205` | latest prior populated opponent PBP tendency | prior game | SAFE | Same restriction; currently null. |
| opp_pressure_rate | `:190-205` | latest prior populated opponent PBP tendency | prior game | SAFE | Same restriction; currently null. |
| deep_matchup_score | `buckets.py:506-584` | prior Kalman air share × prior opponent tendency | prior game | SAFE | Both inputs are causal; currently null because tendency is null. |
| coverage_matchup_score | `:506-584` | prior Kalman target share × prior tendency | prior game | SAFE | Both inputs are causal; currently null. |
| separation_demand_score | `:506-584` | prior Kalman target share × prior tendencies | prior game | SAFE | Both inputs are causal; currently null. |

`blitz_exposure` is deliberately absent from `FEATURE_COLS` and all registered
Phase-4 groups, but is a sibling worth recording: `compute_scheme_interactions`
at `buckets.py:506-584` computes `snap_pct_off × opp_blitz_rate`.  It is null
today solely because `opp_blitz_rate` is null.  **Verdict: LATENT LEAK** (target
game snap share inherited from `snap_pct_off`; likely high effect once populated).

Two additional non-default consumers widen the C-01 artifact blast radius:

| Column / consumer | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| routes_run_pct | `feature_engineer.py:213-215,285` | `target_row.offense_pct` | target game | LEAK | Not in `FEATURE_COLS`, but the stored field is another direct copy of the confirmed postgame snap value. |
| TFT `snap_share` | `ml/tft_model.py:355-367` | `snap_pct_off.fillna(0)` | target game | LEAK | TFT maps the default leaked feature into its time-varying input; all TFT artifacts trained with this mapping are invalid. |
| GNN `snap_pct_off` node feature | `ml/gnn_matchup.py:85-103` | `feature_matrix.snap_pct_off` | target game | LEAK | GNN is outside `FEATURE_COLS`; any trained GNN using `OFF_NODE_FEATURES` is contaminated. |

### Buckets 10–11 and other default columns

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| team_off_elo | `enrich_elo_embeddings.py:47-202` | `TeamEloSystem` snapshot | prior game | SAFE | `get_snapshot()` selects `(season, week) < target`; current DB null. |
| team_def_elo | same | `TeamEloSystem` snapshot | prior game | SAFE | Same strict tuple comparison. |
| opp_off_elo | same | opponent Elo snapshot | prior game | SAFE | Same strict tuple comparison. |
| opp_def_elo | same | opponent Elo snapshot | prior game | SAFE | Same strict tuple comparison. |
| elo_matchup_diff | same | team prior Elo − opponent prior Elo | prior game | SAFE | Derived from safe snapshots. |
| elo_implied_win_prob | same | prior Elo + home flag | prior game | SAFE | `get_snapshot` excludes game week. (The enrichment omits `is_home`, an accuracy defect, not a leak.) |
| height | `feature_engineer.py:178-194` | current `players.height` | as-of unknown | UNPROVEN | Current snapshot lacks effective dating; DB value is null now because height strings do not coerce to float. |
| weight | `:178-194` | current `players.weight` | as-of unknown | UNPROVEN | DB: 129,103/129,103 non-null FM weights exactly equal current player record; no historical snapshot. |
| draft_round | `:178-194` | current `players.draft_round` | pregame-known | SAFE | Immutable after draft; DB 75,012 values exactly match player record. |
| epa_per_play | `pbp_pipeline.py:232-472,709-750` | same-game player PBP EPA | target game | LATENT LEAK | `main()` calls target `(player_id,game_id)` FM update; FM is null today, source has 15,876 values. |
| epa_per_target | same | same-game receiver EPA / target | target game | LATENT LEAK | Same direct target-game key. |
| epa_per_rush | same | same-game rusher EPA / carry | target game | LATENT LEAK | Same direct target-game key. |
| qb_epa_per_dropback | same | same-game QB EPA / dropback | target game | LATENT LEAK | Same direct target-game key. |
| adot | same | same-game air yards / targets | target game | LATENT LEAK | Same direct target-game key. |
| yac_per_reception | same | same-game YAC / receptions | target game | LATENT LEAK | Same direct target-game key. |
| xyac_per_reception | same | same-game expected YAC / receptions | target game | LATENT LEAK | Same direct target-game key. |
| target_share_pbp | same | same-game targets / team targets | target game | LATENT LEAK | Same direct target-game key; 13,323 PBP source values. |
| air_yards_share_pbp | same | same-game player / team air yards | target game | LATENT LEAK | Same direct target-game key. |
| red_zone_targets | same | same-game red-zone targets | target game | LATENT LEAK | Same direct target-game key. |
| end_zone_targets | same | same-game end-zone targets | target game | LATENT LEAK | Same direct target-game key. |
| red_zone_target_share | same | same-game player / team RZ targets | target game | LATENT LEAK | Same direct target-game key. |
| drop_rate | same | same-game PBP/FTN/PFR drops / targets | target game | LATENT LEAK | Same direct target-game key. |
| ol_pressure_rate | same | same-game team hits / dropbacks | target game | LATENT LEAK | Same direct target-game key. |
| ol_sack_rate | same | same-game team sacks / dropbacks | target game | LATENT LEAK | Same direct target-game key. |
| opp_pressure_rate_pbp | `pbp_pipeline.py:385-435` | shifted opponent defensive PBP | prior game | SAFE | `groupby(team).shift().cumsum()` occurs before rate; currently FM null. |
| opp_sack_rate_pbp | `:385-435` | shifted opponent defensive PBP | prior game | SAFE | Same shifted construction; currently FM null. |
| pass_left_rate | `:245-323,709-750` | same-game target direction / targets | target game | LATENT LEAK | Target PBP aggregate written by `(player_id,game_id)`. |
| pass_middle_rate | same | same-game target direction / targets | target game | LATENT LEAK | Same direct target-game key. |
| pass_right_rate | same | same-game target direction / targets | target game | LATENT LEAK | Same direct target-game key. |
| target_share_trend | `feature_engineer.py:197-202` | last two/three prior `target_share` values | prior game | SAFE | Reads `prior_rows[-3:]`, never `target_row`. |
| wind_x_qb | `:204-212` | `wind_bucket × position` | as-of unknown | UNPROVEN | Inherits `wind_bucket` provenance gap. |
| wind_x_wr | `:204-212` | `wind_bucket × position` | as-of unknown | UNPROVEN | Inherits `wind_bucket` provenance gap. |
| precip_x_pass | `:204-212` | `games.precipitation_bucket × position` | as-of unknown | UNPROVEN | Currently null; schedule weather source has no as-of snapshot. |
| team_pos_rank | `buckets.py:255-298` | team prior target/carry/attempt totals | prior game | SAFE | Explicit `week < target_week`. |
| depth_chart_rank | `feature_engineer.py:793-803` | same-week `depth_charts.depth_rank` | as-of unknown | UNPROVEN | DB exact match 104,437/104,437 on `(player,season,week)`; no publication timestamp establishes pregame source selection. |
| avg_separation | `feature_engineer.py:804-814` | same-week `nextgen_stats.avg_separation` | target game | LATENT LEAK | NGS is weekly game performance; exact target `(player,season,week)` update. 4,207 source rows; FM null. |
| avg_cushion | `:804-814` | same-week `nextgen_stats.avg_cushion` | target game | LATENT LEAK | Same target-week postgame NGS join; FM null. |
| player_emb_0 | `enrich_elo_embeddings.py:205-344` | current player profile embedding | as-of unknown | LATENT LEAK | Global current snapshot includes `CURRENT_DATE` age/current `years_exp`; updates every historical row by player. |
| player_emb_1 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same global historical overwrite; FM null. |
| player_emb_2 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same. |
| player_emb_3 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same. |
| player_emb_4 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same. |
| player_emb_5 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same. |
| player_emb_6 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same. |
| player_emb_7 | same | current player profile embedding | as-of unknown | LATENT LEAK | Same. |

## Opt-in Phase-4 groups (not in `FEATURE_COLS`)

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| carry_share | `buckets.py:709-753` | player prior carries / `team_carries_by_game[game_id]` | prior game | SAFE | Causal time restriction, **but C-18 corrupts denominator by combining both teams**. |
| snap_share_trailing | `:709-753` | player prior `offense_pct` mean | prior game | SAFE | Uses only `prior_rows`; not C-18 affected. |
| snap_share_trend | `:709-753` | player prior snap change | prior game | SAFE | Uses only `prior_rows`; not C-18 affected. |
| ts_vs_league | `:606-704,755-790` | trailing player / league target-share means | prior game | SAFE | Both per-player histories filter `< week`; not game-keyed. |
| rz_ts_vs_league | `:755-790` | no implemented source | as-of unknown | SAFE | Always returned `None`; inert, no leakage path. |
| snap_vs_pos_avg | `:606-704,755-790` | trailing player / league snap means | prior game | SAFE | Histories filter `< week`; not game-keyed. |
| carry_share_vs_league | `:606-704,755-790` | carry share / league carry mean | prior game | SAFE | Causal but **C-18 corrupts both game-keyed denominator and league reference**. |
| opp_adj_target_share | `:606-704,755-790` | prior season target share / matchup / league targets | prior game | SAFE | Causal but **C-18 corrupts `league_tgt_allowed` through both-team game totals**. |
| team_pace | `buckets.py:798-864` | prior team pass+rush totals | prior game | SAFE | Causal but **C-18 overwrites `game_team[game_id]` and combines both teams**. |
| team_pass_rate | `:798-864` | prior team pass / plays | prior game | SAFE | Same C-18 corruption. |
| expected_pass_attempts | `:798-864` | corrupted trailing volume + pregame spread/total | prior game | SAFE | Same C-18 corruption; odds part is pregame. |
| expected_pass_rate | `:798-864` | corrupted trailing pass/rush + odds | prior game | SAFE | Same C-18 corruption. |
| neutral_script_flag | `:798-864` | `abs(games.spread_line) <= 3` | pregame-known | SAFE | Not C-18 affected. |
| years_exp | `buckets.py:877-927` | current `players.years_exp` joined to target row | as-of unknown | LEAK | Confirmed snapshot leak: DB 129,115 values; all 4,357 players have exactly one value across all historical rows. |
| age | `:877-927` | target season − `players.birth_date` | pregame-known | SAFE | Uses target `season`, not current date; birth date is static. |
| career_games | `:907` | `len(prior_rows)` | prior game | SAFE | Strictly prior rows. |
| exp_bucket | `:909-918` | bucket(`years_exp`) | as-of unknown | LEAK | Direct deterministic descendant of the confirmed `years_exp` snapshot leak. |

### C-18 scope confirmation

The only problematic `game_id`-alone **team** aggregate is the block at
`buckets.py:606-652`: `team_carries_by_game`, `team_pass_by_game`,
`team_rush_by_game`, and `team_targets_by_game` are keyed only by `gid`, while
`game_team[gid] = team` overwrites one team with the other.  It affects
`carry_share`, `carry_share_vs_league`, `opp_adj_target_share`, `team_pace`,
`team_pass_rate`, `expected_pass_attempts`, and `expected_pass_rate` above.
`compute_matchup_stats` also groups by `game_id`, but its input contains only
the offensive players who faced the specified defending opponent, so that use
does not combine the two teams.

## Row-removing predicates in training/evaluation paths

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|
| training matrix season selection | `ml/utils.py:332-374` | `feature_matrix.season = ANY(seasons)` | pregame-known | SAFE | Train-fold membership only; does not inspect target outcome/feature. |
| training position selection | `ml/utils.py:348-371` | `feature_matrix.position` | pregame-known | SAFE | Static role selection. |
| **training snap cohort** | `ml/utils.py:377-388` | target-row `snap_pct_off >= .34` (or null) | target game | LEAK | Retains 37,268/129,128; `snap_pct_off` equals same-game `offense_pct`. |
| prior-history season/week selection | `ml/utils.py:438-454` | `(season, week) < target` | prior game | SAFE | Correctly excludes target and future history. |
| prior-history snap pruning | `ml/utils.py:458-460` | historical-row `snap_pct_off >= .34` | prior game | SAFE | Removes only rows before inference target; not target participation. |
| XGB/LGB/CatBoost position filter | `xgb_model.py:356`; `lgbm_model.py:380`; `catboost_model.py:259` | `position == requested` | pregame-known | SAFE | Static task partition. |
| XGB/LGB/CatBoost label-present filter | `xgb_model.py:360`; `lgbm_model.py:383`; `catboost_model.py:261` | target label non-null | target game | SAFE | Standard supervised-label availability; label is not an input or cohort quality proxy. |
| TFT label-present filter | `tft_model.py:455` | target label non-null | target game | SAFE | Same standard label-availability restriction. |
| walk-forward train/validation season splits | model files above | `season` | pregame-known | SAFE | Expanding folds separate seasons; not a within-game feature read. |
| eval causal position filter | `ml/eval_causal.py:64-66` | OOF `position` | pregame-known | SAFE | Static reporting subset. |
| backtest evaluation season filter | `ml/run_backtest.py:117` | `history.season == eval_season` | pregame-known | SAFE | Evaluation window selection only. |
| **backtest/eval snap cohort** | `ml/eval_cohort.py:56-82`; `ml/run_backtest.py:117-121` | target-game `game_logs.offense_pct >= .34` | target game | LEAK | Same participation-derived cohort, normalized from `offense_pct`. |
| backtest position subset | `ml/run_backtest.py:121` | `position.isin(positions)` | pregame-known | SAFE | Static reporting subset. |
| backtest actual-present filter | `ml/run_backtest.py:151-154` | target statistic non-null | target game | SAFE | Standard scoring-label availability, not used by prediction. |
| backtest baseline-available filter | `ml/run_backtest.py:176` | prior-season or trailing baseline exists | prior game | SAFE | Baselines themselves enforce `season-1` or `week < week` (`ml/baselines.py:31-70`). |
| feature-group train/holdout split | `ml/feature_groups.py:119-120` | `season < / == holdout` | pregame-known | SAFE | Holdout boundary only. |

## What must be rebuilt

The default pipeline is already invalid because the live `snap_pct_off` value
and its row filter were used in the model-ready matrix.  Treat all base-model
and downstream evidence built from that default cohort as invalid, regardless
of a later strict-common-cohort comparison.

Required regeneration order after remediation:

1. **Features:** rebuild every affected `feature_matrix` season with the
   target-game snap value removed from feature construction and no target-game
   snap eligibility filter.  Preserve or explicitly rebuild safe prior-history
   calculations.  Do not enable the PBP/NGS/embedding paths until they are made
   causal; otherwise their latent leaks become live during this step.
2. **Base OOF:** retrain every base learner and regenerate all OOF files on the
   unfiltered, causal feature matrix.
3. **Stacks:** rebuild all level-2/meta models from the new base OOF outputs.
4. **Evaluations:** rerun causal, cohort, and baseline evaluation with a cohort
   definition independent of target-game participation.
5. **Promotion evidence:** regenerate all comparison tables, model cards,
   artifact manifests, and any threshold/selection decision made from prior OOF.
6. **Materialized projections:** refresh projection tables/files from the
   promoted replacement artifacts only.

**Scope note:** C-18 and the confirmed `years_exp`/`exp_bucket` problem are
confined to opt-in Phase-4 groups, not `FEATURE_COLS`; they do not by themselves
expand the default base-model retrain scope.  They do invalidate any experiment
or promotion evidence that enabled `opp_adj_usage`, `pace_script`, or
`progression`.  In contrast, PBP, NGS, and embeddings are in `FEATURE_COLS`;
they are currently inert in the inspected feature matrix but must be repaired
before a refresh that populates them.
