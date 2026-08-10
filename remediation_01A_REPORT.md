# Session 01A — Containment report

Branch `fix/01a-containment`, worktree `../go_01A_containment`, based on tag
`audit-baseline-2026-08-09` (`191e151`). **Staged, not committed.**

Scope: C-05, C-06, C-16, C-17, C-20 + 5 regression locks. No trainer, restack or
materialize was run. Nothing written to Postgres or MLflow.

## Test results

| | Failed | Passed | Skipped |
|---|---|---|---|
| Baseline (tag, before changes) | 3 | 829 | 8 |
| After changes | 3 | 903 | 8 |

**The 3 failures are the same pre-existing ones and belong to other sessions:**

1. `test_kalman_tracker.py::TestSchemaSync::test_feature_row_fields_in_feature_matrix` — C-12, ORM/DB drift → **03B**
2. `test_train_pipeline.py::TestStackingInferenceHelpers::test_stacking_mlflow_with_mock_models_uses_their_predictions` — C-04, fail-open inference → **03A/05**
3. `test_xgb_model.py::TestParseSeasons::test_range_notation` — C-28, season cap warn-not-raise → **05**

Zero regressions introduced. +74 net new tests (73 containment locks, plus one pre-existing test inverted).

## Files deleted (10, all `git rm`)

Poisoned four-learner stack:
- `ml/oof/stack_fantasy_ppr_TE_20260807.csv` — header confirmed `xgb_pred,…,tft_pred`, 3783 rows
- `ml/oof/stack_fantasy_ppr_TE_20260807.csv.sha256`

Legacy position-less `receiving_yards` OOFs (conflicting fold maps):
- `ml/oof/xgb_receiving_yards_20260503.csv`, `ml/oof/xgb_receiving_yards_20260720.csv`
- `ml/oof/tft_receiving_yards_20260501.csv`, `ml/oof/tft_receiving_yards_20260503.csv`
- `ml/oof/lgbm_receiving_yards_20260503.csv`, `ml/oof/lgbm_receiving_yards_20260720.csv` — **not in the brief**; same defect and *more* dangerous, since `lgbm` is an allowed learner so the allowlist alone would not stop them

Dead killed-learner metadata:
- `ml/oof/manifests/xgb_receiving_yards_all_93eed99c6376e900.json`
- `ml/oof/manifests/xgb_receiving_yards_all_d4165abedad4881c.json`

### Deliberately KEPT, against the brief's wording

The brief said to remove "any other tracked `*_20260807.*`". Two such files are
**not** poison and were kept:

- `ml/oof/lgbm_fantasy_ppr_TE_20260807.csv`
- `ml/oof/catboost_fantasy_ppr_TE_20260807.csv`

Both are clean two-learner base OOFs (5576 rows, header `…,y_true,y_pred,fold_idx`)
and there is **no `_20260809` TE base OOF** — they are the only provenance for the
serving `stack_fantasy_ppr_TE_20260809.csv`. Deleting them would have destroyed the
inputs of a shipped artifact. Consequence: the done criterion "no `*_20260807.*` on
disk" is intentionally not met for these two.

## Selection sites changed (13)

Manifest-pinned + SHA-verified, fail-closed, no glob fallback:

| Site | Was |
|---|---|
| `backend/app/api/draft.py:81` | `max(paths, key=mtime)` |
| `scripts/materialize_stack_projections.py:61` | `max(paths, key=mtime)` |
| `ml/adp_eval.py:191` | `max(paths, key=mtime)` |
| `ml/train.py:1071` (C-20) | `glob.glob` + concatenate **all** matches |
| `ml/stacking_ensemble.py:799` `_discover_oof_files` | glob 4 learner prefixes, mtime sort |

Deterministic filename ordering (not manifest-pinned; these artifacts are not release-pinned):

| Site | Was | In brief? |
|---|---|---|
| `ml/stacking_ensemble.py:752` training-data hash | `getmtime` in hash input | no |
| `backend/app/services/backtest.py:144` | `key=st_mtime` | **no** |
| `backend/app/services/runtime_status.py:183` | `key=st_mtime` | **no** |
| `scripts/retrain_fantasy_ppr_collapsed.sh:49` | `max(key=mtime)` in heredoc | **no** |
| `scripts/train_volume_oof.sh` | `ls -t` ×2 | **no** |
| `scripts/train_yardage_oof.sh` | `ls -t` ×2 | **no** |
| `scripts/train_passing_yards_qb.sh:37` | `ls -t` | **no** |
| `backend/tests/test_api.py:520` | **asserted** `st_mtime` must be used | **no** |

`git grep -n 'key=.*mtime\|getmtime'` over `backend/ ml/ scripts/` returns nothing.
`ls -t` no longer appears in `ml/` or `scripts/`. The only surviving `st_mtime` is
`ml/parquet_cache.py:193`, a legitimate cache-age check, not selection.

## New files

- **`ml/artifact_manifest.py`** — the single authority. Learner allowlist
  (`{lgbm, catboost}`, global — verified against all 15 shipped coef files),
  `assert_learner_policy`, `assert_coef_keys_allowed`, SHA-pinned
  `resolve_stack(stat, position)`, `protected_relpaths()`.
- **`scripts/guard_release_artifacts.py`** — refuses a purge that would delete
  pinned artifacts. Exit 3 = refused, 4 = manifest unreadable (fail closed).
- **`backend/tests/test_artifact_containment.py`** — 73 locks.

## C-05 enforcement shape

The policy is a checked invariant at three points, so a forgotten CLI flag cannot
widen the learner set:

1. `stack()` immediately after `load_and_align_oofs` — covers `--oof`, `--oof-dir`, and direct Python callers
2. Coef-write site — no coef file may carry a killed learner's key
3. `_discover_oof_files` — **raises** on a killed learner's OOF in the directory (does not silently skip)

`--exclude` is now narrowing-only convenience. XGB/TFT training blocks removed from
`train_all_models.sh`, `train_fantasy_ppr_oof.sh`; `--exclude tft,xgb` removed from
all call sites as redundant.

## Manifest changes (`releases/current_baseline.json`)

Two additive changes; `git_commit`, `git_dirty` and `projection_policy` untouched
(re-freeze is 03A/04's job):

1. **`artifact_digests`** — new flat map, 16 entries. The manifest carried no SHAs
   at all, so "SHA-pinned manifest read" was not possible as written. All 15 stack
   digests **match their tracked `.sha256` sidecars** — independent confirmation.
2. **`artifacts.stacks_yardage`** — the 5 yardage stacks (`receiving_yards`×3,
   `rushing_yards`×2) that the manifest omitted (C-09/C-10). They are tracked,
   present, and globbed by the conformal path today; with fail-closed reads and no
   glob fallback, omitting them would have broken working calibration. **This is a
   scope note, not a re-freeze.**

`scripts/freeze_baseline.py` never emitted `artifacts` at all, so a re-freeze would
have silently dropped the pins my loader now requires and broken every reader. It
now carries the artifact list forward, recomputes digests, and reports (not hides)
any pinned artifact missing from disk.

## C-16

- Purge removed. Clean start clears checkpoints only. Purging OOF CSVs now needs
  `--purge-oof` **and** passes the guard.
- `mark()`/`done_ck()` suffix mismatch fixed (`$1` vs `$1.done`) — every resume
  previously retrained everything.
- All skip paths require the artifact, not just the marker; stale markers are
  removed and the cell re-runs.
- Order everywhere: write atomically → validate → *then* checkpoint. `_save_stack_oof`
  now writes via temp+rename.
- `set -euo pipefail` on all four trainers. Expected-cell matrix with non-zero exit
  on any missing cell.
- `cp -n` → `mv` in `rebuild_fantasy_ppr_stack_phase5.sh` (the direct cause of the
  poisoning), sidecars moved with their CSVs.
- Per-prefix `*_combined.csv` blobs now stage in `ml/oof/_combined_staging/` instead
  of the serving directory.

Portability verified on macOS `/bin/bash` 3.2: `compgen -G` and empty arrays under
`set -u` both behave correctly.

## Empirical findings that correct the brief

1. **The "fresh clone → mtimes tie → 0807 wins" mode does not reproduce on APFS.**
   Measured: git checkout wrote the two TE stacks 3.3 ms apart, in lexical order, so
   `_20260809` wins by accident. The mode is real but **filesystem-dependent**: with
   mtimes forced equal (simulating 1-second granularity — ext3, HFS+, many container
   and network mounts) the pre-fix draft selector returned
   `stack_fantasy_ppr_TE_20260807.csv`. Confirmed both modes directly.
2. **Legacy fold maps differ three ways, not two.** tft → fold 0 = 2022;
   legacy lgbm/xgb → fold 0 = 2023; current runs → fold 0 = **2020** (brief said 2023).
3. **NEW: the test suite poisoned the serving directory.** Running the suite wrote
   `xgb_receiving_yards_all_20260809.csv` and `lgbm_receiving_yards_all_20260809.csv`
   into `ml/oof/` with the *newest mtimes in the directory*, and modified 4 tracked
   manifest JSONs. Seven `train()` calls in `test_xgb_model.py` / `test_lgbm_model.py`
   omitted `out_dir`, which defaults to `Path(__file__).parent/"oof"` — the real
   serving directory. So **running the tests armed C-17**: a fresh xgb OOF, newest
   mtime, would win discovery. Fixed by passing `tmp_path`/`tmp_path_factory`. My own
   lock caught this.

## Locks — validated against the tag

73 locks pass on the branch. Against `audit-baseline-2026-08-09` (with the tag's
production selectors and only the test harness copied in), **36 of 63 then-existing
locks fail**, covering all five categories.

`TestArtifactPinning` fails 4 of 5 at the tag, but only **three** fail on the
poisoning itself — `test_draft_selects_pinned_not_newest`,
`test_materialize_selects_pinned_not_newest` and
`test_tied_mtimes_still_resolve_to_the_pinned_artifact`, each with:

```
assert 'stack_fantasy_ppr_TE_20260807.csv' == 'stack_fantasy_ppr_TE_20260809.csv'
```

The fourth, `test_adp_eval_reads_pinned_artifact`, fails at the tag on the
`manifest=` keyword not existing — a signature mismatch, not a demonstration of the
poisoning. Its value is on the branch, where it asserts on *values* (pinned 20.0 vs
stale 99.0) rather than on a path.

Lock 2 uses a synthetic `tmp_path` fixture rather than the real `_20260807` file, so
deleting that file does not make it vacuous, and resolves the selector dynamically
(`_pinned_stack_oof` or `_latest_stack_oof`) so the tag's implementation is genuinely
exercised instead of erroring on import.

## Verification

- All 58 `ml/oof/*_20260809.*` **byte-identical** to
  `_gridiron_archive/antidote_20260809_stacks/ml_oof/` (archive re-verified:
  `MANIFEST.sha256`, 0 failures, before any deletion).
- All 15 stack `.sha256` sidecars verify.
- All 15 manifest cells resolve and SHA-verify; no `_20260807` path reachable.
- `git status ml/oof/` shows only the intended deletions.

## HANDOFF — read this before session 02 (retrain)

**A newly trained artifact is invisible until the manifest is re-frozen.** That is
the intended fail-closed behaviour, but it will look like "my retrain didn't take":
a fresh `stack_fantasy_ppr_WR_20260810.csv` will be ignored by draft, materialize,
adp_eval and conformal calibration, because selection resolves only through
`releases/current_baseline.json`.

To make new artifacts visible, add them to `artifacts` and `artifact_digests` — run
`python scripts/freeze_baseline.py`, which now carries the artifact list forward and
recomputes digests, then update the `artifacts` block if the *paths* changed (a new
date stamp is a new path). `ml.artifact_manifest.clear_cache()` drops the in-process
manifest and verification caches.

Also relevant to 02: `_discover_oof_files` now **raises** if a killed learner's OOF
is present in the discovery directory. If a retrain leaves an `xgb_*`/`tft_*` OOF in
`ml/oof/`, the restack will refuse rather than silently include it.

## Correction applied late in the session

My first version of the `train_all_models.sh` expected-cell matrix required all 37
stat×position combinations the loops iterate, while the release pins 15 — it would
have made the script exit 1 on a correct tree and left Step 4 unreachable. The
required set is now the release matrix, read from the manifest via
`guard_release_artifacts.py --list-cells`; cells outside it are reported but not
fatal. Verified: 15 required, 0 missing on the current tree. Locked by
`test_required_cell_matrix_is_the_release_matrix_not_the_cross_product`.

## Beyond mtime: no serving path selects an unpinned stack

`sorted(glob(...))[-1]` matches neither the mtime grep nor `ls -t`, so that was
checked separately. `draft.py`, `predict.py`, `materialize_stack_projections.py` and
`adp_eval.py` contain **no** glob reaching a `stack_` artifact (the textual hits are
explanatory comments). `backtest.py` / `runtime_status.py` glob `backtest_*.csv`,
not stacks, and now order by filename. Locked by
`test_no_serving_path_reaches_a_stack_by_glob`.

Remaining stack globs are all non-serving and deliberately left: the post-training
eval heredocs in `train_yardage_oof.sh` and `train_passing_yards_qb.sh` (filename
order, reading the artifact they just produced), existence-check globs in the
checkpoint validation, and `scripts/reprojection_gate.py`'s `--oof-glob` (C-08 →
session 04).

## Left for other sessions

- The 3 pre-existing failures (03A/03B/05).
- Manifest re-freeze at a clean `191e151`, run-ID registration, posterior policy (C-09) → 03A/04.
- `_history_*.csv` and `promotion_*.json` remain unpinned (not read through the
  selection paths this session touched).
