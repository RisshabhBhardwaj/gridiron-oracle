# Runbook: scripts index

Type: Runbook
Status: Current
Last verified against code: 2026-10-10 (commands built from each script's argparse; none were run while writing this page)

Run from the repo root with `.venv_311/bin/python`. Scripts that take a database URL default to the local dev database (`localhost:15439`) unless noted. Pass `--database-url` to override.

- **WIRED**: has a Make target, or is called by `pipeline/run_full_etl.sh`.
- **DOC ONLY**: no target. Run by hand, as shown.
- **WARNING**: writes to a served table, a release file, or a tracked evidence file.

## Wired

### materialize-forward-projections (WIRED)
- Purpose: forward projections for one feature snapshot from the base models pinned in `releases/current_baseline.json`. A thin CLI over `ml/forward_serving.py` (decision 0012, built 2026-11-14).
- Inputs: the `forecast_runs` snapshot row (written by `build_forward_features.py`) and an availability snapshot fetched at or before its `as_of`. Needs `PRODUCT_MODE=artifact_backed`.
- Outputs: `projections` rows under the run's own pending id `weekly_auto_<season>_pending_<uuid>` (one row per rostered depth-chart player and stat; `availability_status` stored) and a `forecast_runs` row keyed `forward_projections_<season>_w<WW>_<UTC stamp>`. Served only after promotion to an approved run id.
- WARNING: writes `projections`. Guards G1-G5 all run before the single write; a refusal exits 1 and writes nothing. `--dry-run` runs everything except the write.
- Run: `make materialize-forward-projections SEASON=2026 WEEK=1 FEATURE_RUN_ID=<id>`

### materialize-preseason-draft (WIRED)
- Purpose: week-0 draft projections from seasons before `SEASON` only (`scripts/materialize_preseason_draft_projections.py`).
- Inputs: `players` and `game_logs` with `season < SEASON`.
- Outputs: `draft_preseason_projections` for `(SEASON, AS_OF)`. The draft API serves this table (`backend/app/api/draft.py:145-162`).
- WARNING: with `APPLY=1` it replaces the `SEASON` and `AS_OF` slice. It refuses an empty projection set. Without `APPLY=1` it is a dry run.
- Run: `make materialize-preseason-draft SEASON=2026 AS_OF=2026-09-01` (dry run). Add `APPLY=1` to write.

### verify-drive-transitions (WIRED)
- Purpose: checks the Q4 run-rate effect in raw `pbp_plays` and in the fitted drive model. `ml/markov_simulator.py:51` cites its thresholds.
- Inputs: `pbp_plays`, read only. Outputs: a printed report. No writes.
- Run: `make verify-drive-transitions`

### build_forward_features.py (WIRED)
- Purpose: one forward feature snapshot and its `forecast_runs` row.
- Wired in `pipeline/run_full_etl.sh:132`, only when `FORWARD_SEASON`, `FORWARD_WEEK` and `FORWARD_AS_OF` are all set. It is one step of the full ETL, not a separate command.
- Run: `FORWARD_SEASON=2026 FORWARD_WEEK=1 FORWARD_AS_OF=<iso-time> bash pipeline/run_full_etl.sh`

## Doc only

### compare_stack_vs_marcel.py (DOC ONLY)
- Purpose: walk-forward comparison of the stack's OOF against Marcel-style baselines, `fantasy_ppr` only.
- Inputs: `--oof-dir` (default `releases/candidates/causal_20260810`), optional `--history`. Outputs: `reports/marcel_vs_stack.json`. That file is gitignored and not in `releases/artifacts/MANIFEST.json`, so list it there before citing it.
- Run: `.venv_311/bin/python scripts/compare_stack_vs_marcel.py`

### draft_walkforward.py (DOC ONLY)
- Purpose: 8-team PPR draft walk-forward against ADP and Marcel.
- Inputs: DB `game_logs` and `fantasy_adp`. Outputs: `reports/draft_walkforward.json`, gitignored and not in MANIFEST.
- The served draft note no longer quotes results from this harness (`backend/app/api/draft.py`).
- Run: `.venv_311/bin/python scripts/draft_walkforward.py --start-season 2019 --end-season 2025`

### import_historical_adp.py (DOC ONLY)
- Purpose: loads historical ADP CSVs into `fantasy_adp`.
- Inputs: `data/adp/historical/adp_YYYY.csv` (see `data/adp/historical/README.md`). Outputs: `fantasy_adp` (source `historical`, scoring `ppr`) and `adp_player_matches`.
- WARNING: writes `fantasy_adp`, the production table the draft board reads.
- Run: `.venv_311/bin/python -m scripts.import_historical_adp --all`

### invalidate_artifacts.py (DOC ONLY)
- Purpose: records known-bad pipeline runs, or a position and stat pair. Used in `docs/runbooks/release-and-evidence.md` step 6.
- Inputs: `--reason` (required), and one or more of `--pipeline-run-id` or `--position` with `--stat`. Outputs: `releases/artifact_invalidations.json`.
- WARNING: writes a release file. It is not digest-pinned. Serving stops only through `approved_pipeline_run_ids` (`backend/app/services/projection.py:78-101`).
- Run: `.venv_311/bin/python scripts/invalidate_artifacts.py --pipeline-run-id <id> --reason "<why>"`

### regenerate_adp_diagnostics.py (DOC ONLY)
- Purpose: regenerates the ADP-agreement diagnostics for 2022 to 2026.
- Inputs: `draft_preseason_projections` for those seasons (from the preseason target) and `fantasy_adp`. Outputs: `reports/adp_spearman_stack_ranks.json`.
- NOTE: that file is tracked and listed in MANIFEST, so a rerun overwrites tracked evidence. Review the diff before committing.
- The default `--output` is relative to the current directory, so run from the repo root.
- Run: `.venv_311/bin/python scripts/regenerate_adp_diagnostics.py`

### regenerate_ridge_coefficients.py (DOC ONLY, stops on current inputs)
- Purpose: refits the serving Ridge coefficients from a causal base-OOF rebuild.
- Inputs: `--input-dir`, a rebuild directory with `README.md` and one LGBM and one CatBoost OOF CSV per cell. Outputs: `ridge_*_coefs.json` and `COEFFICIENTS_MANIFEST.json` in `--output-dir` (required).
- Status: `ml/oof/rebuild_20260809T231500` has 15 LGBM and 15 CatBoost files. The script stops at the first of the 32 cells with no file, which is the 16th.
- WARNING: writes into `--output-dir`. Served pins point at `releases/candidates/causal_20260821_fixed` (160 references in `releases/current_baseline.json`). Never write into that directory or any other pinned directory.
- Run: `.venv_311/bin/python scripts/regenerate_ridge_coefficients.py --input-dir ml/oof/<rebuild> --output-dir <new staging dir>`

### verify_openmp_runtime.py (DOC ONLY, owner runs)
- Purpose: reports the libomp binaries and a single- versus multi-thread gradient-boosting benchmark.
- Inputs: none. No DB or network. Outputs: `reports/openmp_runtime.json`, gitignored and not in MANIFEST. `ml/train_all_models.sh:46` cites a measurement from this script.
- The benchmark takes about a minute, so the owner runs it. Add `--no-benchmark` to skip it.
- Run: `.venv_311/bin/python scripts/verify_openmp_runtime.py`

### retrain_causal_oof.sh (DOC ONLY, owner runs, broken at HEAD)
- Purpose: rebuilds base OOFs and two-learner stacks for the release cells into `ml/oof/rebuild_<stamp>`. It does not touch serving artifacts, manifests, gates, or projections.
- Inputs: the DB (feature-contract check) and `scripts/guard_release_artifacts.py --list-cells`. Outputs: `ml/oof/rebuild_<stamp>/` and `ml/logs/rebuild_<stamp>/`.
- Broken at HEAD: line 26 requires 15 cells, but the guard prints 32. The script exits before training.
- Takes hours. Do not run it until line 26 is fixed.
- Run: `bash scripts/retrain_causal_oof.sh`

## Retired

- Moved to `archive/scripts-2026-10/` (see `archive/INDEX.md`): `backfill_pbp_plays_outcomes.py`, `build_candidate_release.py`, `draft_board_experiments.py`, `measure_conformal_coverage.py`, `promote_candidate_release.py`, `write_post_leak_fix_comparison.py`, `clear_stale_checkpoints.sh`, `rebuild_features_phase4.sh`.
- Deleted (still in git history): `quarantine_pre_leak_projections.py`, `refresh_depth_charts.py`. Weekly refresh already fetches depth charts (`scripts/weekly_refresh.py`, `WEEKLY_SOURCES`).

Not indexed yet: `scripts/draft_component_eval.py`. It has no caller, and served code cites it.
