# 01A · CLAUDE CODE · Containment

| | |
|---|---|
| **Order** | Wave 1, stream A — run first |
| **Concurrency** | ⇄ **CONCURRENT** with `01B` (Codex) and `01C` (Claude Code) |
| **Depends on** | nothing |
| **Blocks** | `02` |
| **Findings** | C-05, C-06, C-16, C-17, C-20 + 5 regression locks (part of C-19) |
| **Runtime** | one session, no long compute |

### Setup — run before pasting the prompt

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_01A_containment -b fix/01a-containment audit-baseline-2026-08-09
cd ../go_01A_containment
```

Point Claude Code at `../go_01A_containment`. Use the main tree's interpreter:
`/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`.
This session writes nothing to Postgres or MLflow.

---

You are working in `gridiron_oracle` in a worktree branched from tag `audit-baseline-2026-08-09`
(commit `191e151`). Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` first — it is the brief. This
session fixes **only** C-05, C-06, C-16, C-17, C-20: the mechanisms that let the repo silently ship
wrong artifacts or destroy the good ones. Nothing else.

## HARD GUARDRAILS — read before touching anything

1. **Do not run any trainer or restack.** Not `ml/train_all_models.sh`, not `scripts/train_*.sh`, not
   `python -m ml.stacking_ensemble`. `ml/train_all_models.sh:102-116` **purges every OOF CSV on a
   normal start, including tracked serving artifacts**, and the trainers rebuild the killed 4-learner
   stack. Running one is the exact failure mode you are here to prevent.
2. **Do not modify, move or regenerate any file matching `ml/oof/*_20260809.*`.** That is the current
   serving set. A verified copy exists at
   `/Users/risshabh/Projects/Active/_gridiron_archive/antidote_20260809_stacks/` (76 files,
   `MANIFEST.sha256`, all verified). If you damage the working set, restore from there — never regenerate.
3. **Do not fix C-01 (feature leakage).** That needs a full retrain and is session `02`. Do not touch
   `pipeline/feature_engineer.py` or the feature definitions in `ml/utils.py`.
4. Another Claude Code session (`01C`) is running concurrently in a different worktree on
   `.githooks/`, `scraper/`, `data/coaching/`, `e2e/`, `.gitignore`, `pipeline/pbp_pipeline.py` and
   `ml/season_constants.py`. **Stay out of those paths.**

## C-05 — Kill the XGB/TFT resurrection path

The Phase-5 two-learner policy exists only in the current artifacts, not in the code that generates
them. Every coef file and every `_20260809` stack is clean `lgbm`+`catboost`; the trainers rebuild
4-learner stacks.

- `scripts/train_fantasy_ppr_oof.sh:41-58` — delete the XGB and TFT training blocks.
- `scripts/train_fantasy_ppr_oof.sh:60-68` — the `--oof-dir` restack has no `--exclude`. Add
  `--exclude tft,xgb`.
- `ml/train_all_models.sh:148-368` (stack call at `:353-358`) — remove XGB/TFT from the Phase-5 target
  matrices and add the exclusion at the stack call site.
- **Primary fix:** make the allowed learner set a *checked invariant inside* `ml/stacking_ensemble.py`,
  keyed by target — a policy constant or manifest, not a CLI flag. The stacker must raise if discovery
  yields any learner outside the allowlist. A flag is exactly the thing that keeps getting omitted;
  the policy must not be bypassable by forgetting an argument.
- `backend/tests/test_stacking_ensemble.py:173-282` still asserts four-learner behaviour. Update to the
  two-learner contract. Update any docs or README text advertising 4-learner inference.

## C-06 — Remove poisoned artifacts and eliminate mtime selection

Two independently reproduced failure modes: on a fresh clone all mtimes tie and `max()` returns the
**0807** file; and `touch` on a stale collapsed file makes discovery prefer it.

- `git rm` the tracked poisoned stacks and sidecars — `ml/oof/stack_fantasy_ppr_TE_20260807.csv` and
  `.sha256`, plus any other tracked `*_20260807.*`. Sweep for untracked ones on disk (the WR `_20260807`
  stack, 2022 `y_pred_std`=0.059) and delete those. **List everything you find before deleting.**
- `scripts/rebuild_fantasy_ppr_stack_phase5.sh:16-20` — `cp -n` is a copy, so "archived" 4-learner
  stacks stay in `ml/oof/`. Change to `mv`. This is the direct cause of the poisoning.
- Replace **every** mtime-based selection with a manifest read pinned by SHA-256:
  - `backend/app/api/draft.py:81-87`
  - `scripts/materialize_stack_projections.py:61-69`
  - `ml/adp_eval.py:191-194`
  - `ml/stacking_ensemble.py:799-848` (`_discover_oof_files`)
  - **`ml/train.py:1071-1077` (C-20)** — a live defect, not a latent risk: conformal calibration
    *concatenates all* stack globs, so stale 4-learner residuals pollute the shipped bands today.
  `releases/current_baseline.json.artifacts` already lists the correct files. Read that, verify SHA-256
  on load, fail closed on mismatch or missing entry. **Do not fall back to globbing.**

## C-17 — Remove legacy learner OOFs from discovery

Tracked legacy files `ml/oof/xgb_receiving_yards_20260503.csv`, `xgb_receiving_yards_20260720.csv`,
`tft_receiving_yards_20260501.csv`, `tft_receiving_yards_20260503.csv` (21–120 rows, WR-only) map
`fold_idx 0 → 2022` while current runs map it to 2023, and `_discover_oof_files` keeps
`y_true`/`season`/`fold_idx` from the first file only. A `--oof-dir` restack of `receiving_yards`
silently reintroduces both killed learners *and* mismatched season mappings.

- `git rm` those four files and any siblings.
- Discovery must require an explicit manifest or `--learners` allowlist. Remove the bare
  `{prefix}_{target}_*.csv` glob across all four learner prefixes.

## C-16 — Stop scripts from succeeding while destroying or skipping work

- **`ml/train_all_models.sh:102-116` — remove the OOF purge.** It deletes tracked serving artifacts on a
  normal start. If a clean-slate mode is wanted, make it an explicit opt-in flag that refuses to touch
  anything listed in the release manifest.
- `scripts/train_fantasy_ppr_oof.sh:21-29` — `mark()` writes `$1` while `done_ck()` reads `$1.done`, so
  checkpoints never match and every resume retrains everything. Fix the suffix. The skip path also never
  verifies the output CSV exists — copy the CSV-verified pattern already at `scripts/train_volume_oof.sh:23`.
- Add `set -euo pipefail` to `train_fantasy_ppr_oof.sh`, `train_volume_oof.sh`, `train_yardage_oof.sh`,
  `train_all_models.sh`. Loops must exit nonzero when a cell fails.
- Add an explicit expected-cell matrix; exit nonzero if any expected cell is missing at the end.
- Order of operations everywhere: write output atomically → validate (SHA, row count, season range) →
  *then* checkpoint. Never checkpoint first.

## Regression locks to add — the point of the session

Nothing currently tests any of this.

1. **Learner policy** — stacking against an OOF dir containing xgb/tft files raises; no coef file can be
   written with a key outside the allowlist.
2. **Artifact pinning** — with two candidate stack files present and mtimes reversed (`touch` the stale
   one), draft / materialize / adp_eval still select the manifest-pinned file. **This test must fail
   against `audit-baseline-2026-08-09`.**
3. **Manifest integrity** — a SHA-256 mismatch raises rather than loading.
4. **Discovery hygiene** — `_discover_oof_files` refuses files outside the allowlist and refuses
   inconsistent `fold_idx→season` mappings across inputs.
5. **No-purge** — `train_all_models.sh` does not delete anything in the release manifest. Test the guard
   function; do not run the script.

## Done criteria

- `git grep -n 'key=.*mtime\|getmtime'` returns nothing in `backend/`, `ml/`, `scripts/`.
- No `*_20260807.*` tracked or on disk; no legacy `xgb_/tft_receiving_yards_*`.
- All four shell scripts start with `set -euo pipefail`; no unguarded OOF purge exists.
- The 5 locks pass, and lock 2 demonstrably fails when checked out against the tag.
- `ml/oof/*_20260809.*` untouched — byte-compare against
  `/Users/risshabh/Projects/Active/_gridiron_archive/antidote_20260809_stacks/ml_oof/`.
- Report the 3 pre-existing test failures separately; do not fix them here (they belong to `03B`/`05`).

**Do not commit until you have listed for me every file deleted and every selection site changed.**
