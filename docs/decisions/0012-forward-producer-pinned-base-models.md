# 0012 Forward producer: pin fitted base models, serve them without MLflow

Date: 2026-11-14. Status: **proposed** (architect review; nothing built). Follows decision 0011, section "Scope of the forward producer". Extends decision 0002 (pinned, fail-closed serving). Author: Claude worker acting as architect, for the Claude Gridiron manager.

Every claim below was checked against the code on `main` at `9e15530`. Line numbers will drift; the function names will not.

## Context

The weekly refresh stops at step 7 because no forward-week `projections` rows exist (0011). The producer exists in pieces: `scripts/build_forward_features.py` (writes `feature_matrix` forward rows and a `forecast_runs` snapshot) and `scripts/materialize_forward_projections.py` (calls `PipelineRunner.run`). Neither is in `weekly_refresh.py`.

What the code does today, and why it cannot simply be wired in:

1. **No fitted models are pinned.** `releases/current_baseline.json` pins 5 artifact classes: stacks, ridge coefficients, bucket attributions, selection policy, drive transitions. The ridge files only hold weights.
2. **`artifact_backed` needs MLflow.** `ml/train.py:825-830` raises without `MLFLOW_TRACKING_URI`; `:849-854` raises if it is unreachable. Models then come from "latest finished run" (`ml/inference_client.py:404-471`), which is not pinned. Box B runs no MLflow.
3. **Even MLflow's latest model is the wrong fit.** Each trainer returns the model of its *last walk-forward fold* (`ml/lgbm_model.py:342` and `:472`; `ml/catboost_model.py:244` and `:351`). That model is fitted on seasons up to 2024, not `LAST_COMPLETE_SEASON` (2025). The 2026-11-14 rebuild also ran with `--no-mlflow` (`scripts/retrain_causal_oof.sh:99` and `:124`), so no model was saved at all.
4. **Hyperparameters were not saved.** Optuna is unseeded (`optuna.create_study(direction="minimize")`, `ml/lgbm_model.py:204`, `ml/catboost_model.py:145`), tunes on the second-to-last fold (`:451-462`), and one parameter set serves every fold of a cell. The result is only in the training log line `Optuna best RMSE=... params=...` (`ml/lgbm_model.py:206`, `ml/catboost_model.py:147`) under `$TRAIN_OUT_ROOT/logs/rebuild_<STAMP>/` on the box. Nothing in the release records it.
5. **An unpinned ONNX path is live.** `try_onnx_pred` (`ml/inference_client.py:473-491`) loads `ml/onnx/{learner}_{stat}_{pos}.onnx` if present and runs right after the MLflow lookup and wins over it (`:137-143`). The files are Git LFS objects from 2026-08-21 and `backend/Dockerfile` copies `ml/` into the image. Any `artifact_backed` run on a machine that has them mixes releases silently.
6. **Intervals would come from unpinned files.** `_load_oof_residuals` (`ml/train.py:1097-1152`) globs `ml/oof/{xgb,lgbm}_{stat}_*.csv` (the stale 2026-08-09 set, no `position` column, so it pools all positions). History rows use `causal_oof_conformal_90` from the pinned stack (`scripts/materialize_stack_projections.py:102-154`). Forward rows would get a different method with different inputs.
7. **The write is unguarded and the run id is random.** `PipelineRunner._write_db` (`ml/train.py:1284-1382`) upserts over any row, including rows served under an approved run id (compare `skip_rows_served_by_another_approved_run`, used at `scripts/materialize_stack_projections.py:386`). The id is `pipeline_<uuid4>` (`ml/train.py:346-348`), so it can never be in `approved_pipeline_run_ids`. It writes no `interval_method`, no `derived_fantasy_projection`, no attributions, and derives `max_train_season` from constants (`:1333-1346`), not from the model.
8. **The audit's guard finding is right and worse than stated.** The `degraded` and run-id checks (`materialize_forward_projections.py:53-57`) run after `runner.run` (`:50`), and `run` writes inside (`ml/train.py:470-473`). They can never stop a write.

Good: the forward feature builder already follows the causal rules. `build_forward_week` (`pipeline/feature_engineer.py:843-1000`) builds each row from the target season's completed games before `as_of`, takes lines, rest and venue from the target game, resolves missing lines causally, and fills routes, PBP/NGS priors and depth rank from strictly earlier weeks. Decision 0006's "copy the prior-season row" text is out of date.

## What "good" means

- A forward number is produced only by files pinned in the release and checked by SHA-256, and fails closed otherwise.
- Forward rows look like history rows to every consumer: same run-id rules, same interval method, same columns.
- A bad run never changes what is served.
- It fits the `etl` container (1.5 GiB) and the weekly cadence, and a person can re-run it.

## Options

**A. Keep `PipelineRunner.run`; add pinned loading to `InferenceClient`.** Smallest diff. But it keeps the Kalman/Bayesian/Monte Carlo layers, the unpinned residual glob, the random run id, the unguarded write, and imports PyMC and MLflow into the etl container. Fixing items 6-8 inside it means rewriting most of the path anyway.

**B (recommended). A small forward-serving module over pinned native model files.** New `ml/forward_serving.py`: load pinned models, predict, apply pinned ridge weights, calibrate with the pinned stack's earlier-season residuals, check, write in one transaction under an injected run id. History and forward writers share one code path for intervals, derived fantasy and upsert. `PipelineRunner`'s `artifact_backed` branch stops using MLflow and ONNX and points here; the MLflow path stays for `graceful_fallback` and local development.

**C. Same as B but ONNX files.** Smaller runtime dependencies (`onnxruntime` is already in `backend/requirements.txt`). Rejected for now: it adds a conversion step whose parity with the OOF models nobody has measured (`export_onnx`, `ml/utils.py:525`). It can be added later behind the same pin class without changing the design.

## Decision (recommended: B with native files)

### 1. Which models (answers Q1)

Derived from the 32 `ridge_*_coefs.json` files in `releases/candidates/causal_20261114/`:

| Cells | Learners in `learner_order` | Models |
|---|---|---|
| 16 | lgbm + catboost | 32 |
| 8 | lgbm only (`validated_best_base`) | 8 |
| 8 | catboost only (`validated_best_base`) | 8 |
| total 32 | | 48 listed, **47 needed** |

`receiving_tds:WR` serves `lgbm_identity`, but needs **no extra model**: `scripts/select_nonregressing_stacks.py:112-118` rewrites its ridge file to `lgbm 1.0, catboost 0.0, intercept 0.0`. Its lgbm model is already in the list. Its catboost model has weight 0.0 and is neither fitted nor loaded. The loader skips a learner whose weight is exactly 0.0 and the load-time check (section 4) allows only that case.

### 2. How final fits are made (Q2)

- New `ml/final_fit.py`, run through a new `scripts/train_window.sh finalfit` mode in the box's `training` profile (10 GiB, 5 CPU). The estimate is minutes, not hours: 47 single fits, no Optuna, no folds. Verify on first run.
- **Data:** `train_seasons()` (2019..`LAST_COMPLETE_SEASON`), guarded by `assert_seasons_within_cap` (`ml/season_constants.py`). Never 2026. This is one season more than the last OOF fold, which is exactly what the walk-forward rule gives for a 2026 target.
- **Same recipe:** reuse each trainer's own data prep and `_train_fold`-style fit so rows, sentinel fill (`MISSING_VALUE_SENTINEL`) and features match. Feature list is `[f for f in FEATURE_COLS if f in df.columns]` (`ml/lgbm_model.py:428`); the final fit stores the list it used.
- **Hyperparameters need a prerequisite commit:** trainers write `params_<learner>_<stat>_<pos>.json` next to the OOF. For the current release, recover from the box logs (open question 2). Without that, "same hyperparameters" cannot be shown.
- **Replay gate (proves parity, replaces trust).** Before a final model is pinned, `final_fit` refits the *last OOF fold* (seasons through 2024) with the same code and params, predicts 2025, and compares with the base-learner OOF kept in `train_out/oof/rebuild_<STAMP>/`. Report max and mean absolute difference per model. Fail above a tolerance set on the first run (start at 1e-4 relative to the target's std; threads can cause tiny float differences). This catches wrong params, wrong feature list, wrong data window and wrong library versions.
- **Reproducible means:** same code sha + params + data hash + library versions reproduce the same predictions within tolerance. It does not mean identical file bytes; the SHA-256 pins the artifact, the replay gate pins the behavior.

### 3. File format, size, storage (Q3)

- Native: LightGBM text (`Booster.save_model`) and CatBoost `.cbm`. Same library that produced the OOF, versions pinned in `backend/requirements.txt` (`lightgbm==4.6.0`, `catboost==1.2.10`) and recorded per model.
- Size is unknown until fitted. For scale, the 39 existing ONNX files total 237 MB (75 KB to 17 MB each). Plan for 100-250 MB.
- The push gate refuses git objects over 5 MiB (`.githooks/pre-push:56-59`). `ml/onnx` passes only because its files are LFS pointers. Recommend the same: Git LFS under `releases/candidates/<id>/models/`. The loader detects an LFS pointer file and fails with a clear message (its digest would also fail). Whether the box checkout smudges LFS is unverified (open question 1).
- `releases/candidates/<id>/models/MODELS.json` (also pinned) records per model: learner, stat, position, file, sha256, `train_seasons`, `max_train_season`, feature list and its hash, params, row count, training-data hash, code git sha, library versions, null rates per feature (`build_training_manifest`, `ml/reliability.py:60-95`), and the replay-gate result.

### 4. Manifest schema and pinning (Q3)

- `artifacts.base_models`: a new list in `releases/current_baseline.json` holding the 47 model files and `MODELS.json`. `load_manifest` (`ml/artifact_manifest.py:482`) already flattens every `artifacts` value (`_iter_artifact_relpaths`), so `artifact_digests` coverage and `protected_relpaths` (used by `scripts/guard_release_artifacts.py`) come for free.
- Add `ArtifactManifest.model_index` and `resolve_base_model(stat, position, learner)`, built like `resolve_coefficient` (`:383`): missing entry → `ManifestEntryMissing`, missing file → `FileNotFoundError`, bad digest → `ArtifactIntegrityError`, via `require_digest`.
- **Load-time cross-check (fail closed at load, not at first use):** for every pinned cell, every learner in its ridge file with weight ≠ 0.0 has a pinned model; no model is pinned that no ridge file uses; every model has `max_train_season <= LAST_COMPLETE_SEASON`; all models share one `release_id`. A manifest with no `base_models` key still loads (history serving is unaffected) but the forward producer refuses to run on it.
- `scripts/freeze_baseline.py` preserves the `artifacts` block and recomputes digests (`:76-116`), so the re-freeze needs no new logic. Write a one-line "accepted, extended by 0012" note into 0002 when this is accepted; do not rewrite 0002.

### 5. Loading without MLflow; the MLflow path (Q4)

- `ml/forward_serving.py` per cell: read pinned ridge (existing `load_ridge_coefs`, already manifest-first) → for each non-zero-weight learner call `resolve_base_model` and load natively → build features with `build_inference_features` → **assert the frame's columns equal the model's stored feature list exactly** → predict → apply weights → finite and shape checks (reuse the checks at `ml/inference_client.py:~150-175`).
- Do **not** use `align_features_to_model` (`ml/inference_client.py:311-363`, zero fill at `:343`): it fills missing columns with zeros and silently cuts to the first N columns. On the pinned path a mismatch is an error.
- Intervals: move `_attach_causal_conformal_intervals`, `_attach_derived_fantasy_projection` and `_upsert` into one shared module (`pipeline/serving_rows.py`) used by both writers. Forward calibration residuals are the pinned stack's seasons 2021-2025, all strictly before 2026. Result: `interval_method = causal_oof_conformal_90`, `p25/p75/boom/bust` NULL, exactly as history rows. No Bayesian or Monte Carlo layer, so no PyMC.
- `PipelineRunner`: in `artifact_backed`, replace the MLflow-latest and ONNX paths with a clear error that names `ml.forward_serving`. Keep MLflow for `graceful_fallback` and local work (it labels degraded output). Retiring `ml/onnx` is a separate cleanup (open question 5).

### 6. Feature parity (Q5)

The builder is causal by construction, but parity with the training columns is **unproven by any test**. Findings:

- All 113 `FEATURE_COLS` classify as PRIOR_GAMES, SEASON_PROFILE or PREGAME in `ml/feature_contract.py`; same-week xFP, snap share, weather, injury status and the like are in `FORBIDDEN_MODEL_FIELDS`. I found **no feature that cannot be known before kickoff**.
- Five groups depend on a runtime "as of" fact and need tests, not trust: market lines (`causal_line_for`), Elo features (state must hold only games completed before `as_of`), `days_rest`/`is_bye_prior`, lagged `depth_chart_rank`, and roster membership from depth charts with `published_at <= as_of`.
- The runner reads `feature_matrix` by `(season, week)` only (`ml/train.py:1532`) and ignores the `forecast_runs` snapshot except for an existence check (`materialize_forward_projections.py:34-42`). The new path filters on the snapshot's `as_of` and checks the row count against `feature_rows`.
- Null-rate gate: for each model feature, the forward null rate may not exceed the training null rate (stored in `MODELS.json`) by more than a set margin (calibrate on the first dry run). A column that is 100% null forward but 5% null in training is a builder bug, and with the -9999 sentinel it would not crash, it would just be wrong.
- Cohort caveat (not a blocker): training and the history rows only contain players who played (`df[target_col].notna()`, `ml/lgbm_model.py:419`). Forward rows are every depth-chart player, so players who end up inactive still get a projection. See open question 3.

### 7. Refresh wiring, run id, idempotency, rollback (Q6)

Place a new step **4b `forward_producer`** right after step 4 (`scripts/weekly_refresh.py:273-296`): it needs steps 1-3's data, runs before steps 5-7, and reuses `pending_run_id = weekly_auto_<season>_pending`. Same failure handling as the other steps (log, dead letter, return 1).

Guards, all before any write to `projections`:

| # | Guard |
|---|---|
| G1 | `PRODUCT_MODE=artifact_backed`; manifest loads; all 47 models resolve and verify |
| G2 | Week is forward: no game of the target week has scores or logs (exists, `feature_engineer.py:875-897`) **and** no kickoff of the target week is at or before `as_of` **and** every game of the previous week has a final score. This closes a real gap: `resolve_season_and_week` counts a week as complete once any kickoff has passed (`weekly_refresh.py:110-125`), so a Thursday-night kickoff on a Friday run flips the target to next week while Sunday games are unplayed |
| G3 | Feature snapshot exists; its `as_of` and `feature_rows` match the rows read |
| G4 | Column parity and null-rate gate (section 6) |
| G5 | After inference: rows for every pinned cell, no non-finite values, no degraded cell, exactly one run id, per-position row counts within bounds |

Then one transaction: upsert with `skip_rows_served_by_another_approved_run`, commit once. A failed guard leaves the database untouched. This fixes audit row 9 by construction.

- **Run ids.** Projection rows: `weekly_auto_<season>_pending`, relabeled by step 8 (`promote_run_id`). Provenance: a new `forecast_runs` row per run keyed `forward_projections_<season>_w<WW>_<UTC stamp>` (unique, so no primary-key clash on the next week; the current plain `INSERT` at `materialize_forward_projections.py:69` would fail on a re-run with a reused key), with feature snapshot id, `current_baseline.json` sha, `MODELS.json` sha and code sha in `source_summary`.
- **Idempotency.** Re-running the same week before promotion replaces its pending rows (same `(player_id, game_id, stat)` keys). After promotion the approved guard stops an overwrite; the producer then reports `already_current` and the step passes without writing. Replacing a served week is a manual, flagged action (`--replace-week`), documented in the runbook, taken after a DB dump. Reason: the table key allows one row per `(player_id, game_id, stat)`, so a pending version cannot sit beside the served one, and an automatic replace that then fails step 7 would pull the week out of serving.
- **Step 7 is too weak today and needs a change.** `verify_forward_week_structural` counts `projections` by season and run id only (`scripts/verify_season_acceptance.py:38-40`), not by week. After wiring, any 2026 pending row would satisfy it. Make it week-scoped (`AND week = start_week`), accept pending or approved ids for `already_current`, and require rows for every pinned cell and every position.
- **Rollback.** Failed run: nothing served changed; pending leftovers are pruned at step 9. After promotion: delete that week's forward rows (`season`, `week`, `pipeline_run_id = weekly_auto_<season>`; step 4 writes only 2021-2025, so 2026 rows are forward) or restore the pre-refresh dump (`~/rerun_backup/` pattern, `docs/runbooks/vps4-causal-rerun.md`). Bad models: revert the re-freeze commit. The old manifest has no `base_models`, so the producer refuses and step 7 fails closed while history keeps serving.

### 8. Memory (Q7)

Measured baseline: 1.16 GiB peak (`memory.peak`, page cache included) of 1.5 GiB, without the producer (0011). Estimate for the new step, **not measured**:

- Imports: `lightgbm` plus `catboost` about 0.1-0.2 GiB RSS. B imports no torch, PyMC, MLflow or SHAP in this step (`ml/train.py` and `ml/inference_client.py` import MLflow lazily, but `PipelineRunner` pulls in `ml.monte_carlo`; the new module must not).
- Feature frame: about 1,000-1,500 rows × 113 floats, under 1 MB.
- Models: loaded one cell at a time and freed. Largest existing file is 17 MB, so tens of MB in memory.

Expected addition 0.15-0.35 GiB on top of whatever the parent process still holds, so **1.3-1.5 GiB: it may not fit**. Measure, do not assume: (1) log `memory.current` before and after the step and per cell; (2) read `memory.peak` and `memory.events` (oom) for a full refresh in the real container, as for the 1.16 figure; (3) run the producer as a child process so the memory returns on exit. Decision rule: peak above about 1.35 GiB means raise `etl` to 2 GiB or run the producer as its own `jobs` one-shot ordered before steps 5-7. That is the owner's call.

### 9. Tests the implementer adds (Q8)

- **Guard order:** a fake connection records statements; each of G1-G5 failing yields zero writes. One test for `degraded` and for two run ids. The old order must fail the test.
- **Pinned load fails closed:** missing key, missing file, wrong digest, LFS pointer text, learner with non-zero weight but no model, pinned model unused by any ridge file, `max_train_season` above the cap, library-version mismatch. No MLflow or ONNX fallback is reachable in `artifact_backed` (patch both to raise).
- **Feature parity:** build forward rows for a synthetic week and for a past week replayed "as of" its Tuesday; assert the column set and order equal the model's list; assert no column is entirely null where training was not; assert the contract (`ml/feature_contract.py`) accepts every column. Property test: moving `as_of` earlier never adds information (rows built from a truncated history equal rows built as of that time).
- **Replay gate:** on a tiny synthetic frame, the gate passes for matching params and fails for changed params or a dropped feature.
- **Final-fit window:** raises for any season above `LAST_COMPLETE_SEASON`.
- **Identity cell:** `receiving_tds:WR` output equals the lgbm model's prediction exactly; its catboost model is not loaded.
- **Shared writer:** history output byte-identical before and after the extraction into `pipeline/serving_rows.py`.
- **Step 7 on forward rows:** passes with week-scoped rows for every cell; fails when only season-level pending rows exist; fails when one position is missing; passes `already_current`.
- **Refresh:** `test_weekly_refresh*.py` gains step 4b ordering (after 4, before 5), failure returns 1 and writes a dead letter, and the Thursday-kickoff case (G2).

### 10. Thin first slice and commit order (Q9)

Slice: three cells that cover every shape, the same three the nightly canary uses (`scripts/train_window.sh:small`): `passing_yards:QB` (lgbm only), `receiving_yards:WR` (catboost only), `targets:TE` (both). Run on the box in dry-run (no database write): final fit, replay gate, parity report, `memory.peak`. This settles the open measurements before any freeze.

Order (each commit passes the gate: pytest `-m "not integration and not slow and not network"`, ruff, evidence manifest):

1. Trainers persist `params_*.json` next to the OOF. *(prerequisite; also fixes future rebuilds)*
2. Extract the shared writer into `pipeline/serving_rows.py`; history output unchanged.
3. Manifest: `base_models`, `resolve_base_model`, load-time cross-check, tests.
4. `ml/final_fit.py` + `train_window.sh finalfit` + replay gate + tests.
5. `ml/forward_serving.py` with guards G1-G5 + tests.
6. `PipelineRunner` `artifact_backed` stops using MLflow/ONNX; `materialize_forward_projections.py` becomes a thin CLI over step 5's module.
7. `weekly_refresh.py` step 4b + week-scoped step 7 + tests. **Not enabled in the timer yet.**
8. *Box, manager:* slice dry-run on 3 cells, then final fit of all 47, replay gate, memory measurement.
9. *Manager:* add models (LFS), `MODELS.json`, `artifacts.base_models` to a new candidate directory.
10. *Manager:* **re-freeze** (`make freeze-baseline`), sync the box checkout, run one refresh, read `memory.peak`.
11. Slice 2 (before announcing forward numbers): per-row bucket attributions for forward rows. `PipelineRunner._write_db` writes none, so bucket weights would not move forward numbers (0011 made them live for history). TreeSHAP needs the `shap` library and the models at inference; it is the memory-heavy part and needs its own measurement.

The re-freeze (10) must come after the model files and `MODELS.json` exist and after the code that reads them is on the box; it changes `git_commit` and digests but not `approved_pipeline_run_ids`.

**Hard to undo:** the manifest key name and `MODELS.json` format (frozen releases and loaders depend on them); model blobs in public git history through LFS (cannot be removed without a history rewrite; storage and quota); the run-id and `forecast_runs` key conventions once forward rows are served and recorded. **Easy to undo:** the models (refit), the step wiring (switch the timer back or skip step 4b), forward rows (delete the week).

## Open questions for the owner

1. **Where do model files live?** Git LFS (precedent: `ml/onnx`) or outside git on a box volume with only digests in git? Estimated 100-250 MB. Does the box checkout run `git lfs pull`? (Unverified; I did not touch the box.)
2. **Hyperparameters.** If the box logs of rebuild `20261114` still have the `Optuna best ... params=` lines, recover them from there. If not: accept re-tuning (then the OOF no longer describes the final models exactly), or repeat the 51-minute rebuild with params saved.
3. **Players who do not play.** Show a forward projection for every depth-chart player (current builder), or only for players expected to play? Affects what the UI shows for inactive players.
4. **Replacing a served week.** Manual only (recommended), or automatic on a newer depth chart with a staged swap (more code)?
5. **Retire `ml/onnx` and MLflow-in-`artifact_backed`?** 39 stale LFS files (237 MB) that can be loaded without any pin today.
6. **`etl` memory.** If the measured peak is over about 1.35 GiB: raise the limit to 2 GiB, or split the producer into its own job.
7. **Forward attributions** are slice 2. Is it acceptable to serve forward numbers where Settings bucket weights do not apply until then?
8. **Security (resolved 2026-11-14 by the manager).** `releases/current_baseline.json` held the Box B database password inside `environment.DATABASE_URL`. That password was rotated the same day and no longer works; `scripts/freeze_baseline.py` now writes the URL with the password masked, and a gitleaks rule now catches passwords in connection URLs. The next re-freeze removes the dead value from the file; git history keeps it.
