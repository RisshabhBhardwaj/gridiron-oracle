# Runbook: causal rerun on Box B (VPS-4)

Type: Runbook
Status: Run once on the box, 2026-11-14 (C1 `1810f51`, C2 `226e0ec`, C3 `c3dfc11`). Measured times and the fixes the run needed are in "Run log 2026-11-14" below; the Timeline table keeps the pre-run estimates for comparison.
Goal: restore 2019-2023 history, rebuild all 32 serving cells with per-row bucket attributions, re-freeze, evaluate, and end with `/api/health` reporting `artifact_mode_ready: true` and a clean `weekly_refresh.py`.
Who runs it: the Gridiron manager, with the owner's go-ahead for the production-DB steps (6, 7, 14).

## Conventions

- **Box commands** run on the box from `/srv/gridiron/infra/hosts/vps1` (`cd` there first). "In training" means `docker compose --profile training exec -T training <cmd>` (container `/app`, image `gridiron-etl:full`, 10 GB, 5 CPUs). "In db" means `docker compose exec -T db <cmd>`.
- **Mac commands** run in the merged repo checkout and are marked `[Mac]`. `$BOX` is `deploy@<box address>`, set in your shell. The address is never written in the repo.
- Commit names used below: **C1** = merged `main` that carries this work. **C2** = C1 plus the promoted release candidate (step 12). **C3** = C2 plus evidence only (step 18).
- `/srv/gridiron` has no `.git`. The images copy code and `releases/` at build time (`backend/Dockerfile` and `Dockerfile.serving` both `COPY releases`). `reports/` is gitignored and is **not** in the image.

## Decisions this runbook makes

| Question | Decision | Why |
|---|---|---|
| ETL timer (Tue 13:00 UTC) during the rerun | **Stop `gridiron-etl.timer` (and both train timers) before the code sync, re-enable after step 19.** | Promotion takes about a day, so ordering "promotion first" is not possible. After the sync, `materialize_stack_projections` needs pinned attributions for all 32 cells and fails closed until step 13 (`_load_cell_attributions`). A Tuesday ETL in that gap would fail and write a dead letter. The train timers would start the pinned recipes against a half-finished rerun. |
| Where `releases/` is written | **Box container, then copied to Mac and committed.** | The promote and freeze scripts write inside the training container. The Mac repo is the source of truth, and a later sync from it must carry the promoted release (step 18 checks this). |
| `GIT_COMMIT_SHA` on the box | **C2**, set in `.env` before the freeze and kept. | Readiness without a worktree needs the manifest's `git_commit` to equal this variable exactly (`runtime_status.py:272-289`). C3 adds only evidence files, so C2 stays the commit the release was frozen from. |
| Serving during the gap | **Accept a short gap** (step 14 to step 16). | See "Serving gap". |

## Readiness: what has to be true

`artifact_mode_ready` is `all(checks[name].status == "ok" for name in required)` (`backend/app/services/runtime_status.py:101-103`). In `artifact_backed` mode `required` is `database`, `feature_matrix`, `baseline_manifest`, `backtest_assets` (lines 72-83). MLflow and tracing are reported but not required.

| Check | Code | Needs | How this rerun satisfies it |
|---|---|---|---|
| `database` | `_check_database`, line 107 | `SELECT 1` works | The `db` service is up. |
| `feature_matrix` | `_check_feature_matrix`, line 122 | newest `computed_at` is at most **24 h** old, else `warn` | The final `weekly_refresh.py` (step 20) writes fresh rows. Until then this stays `warn`. No check is relaxed; run step 20 and read health within 24 h. |
| `baseline_manifest` | `_check_baseline_manifest`, line 191 | file exists and parses; no git worktree, so `head` = `RAILWAY_GIT_COMMIT_SHA` or `GIT_COMMIT_SHA` (lines 227-241, empty = error); `model_version` set; `git_dirty` is `false` (267); declared cells equal the 32 `REQUIRED_SERVING_CELLS` and every pin resolves (243-255); `git_commit` equals `head` exactly (273, otherwise `warn` at 281); `product_mode` equals the runtime's (293) | `freeze_baseline.py` records `GIT_COMMIT_SHA` (C2) and `git_dirty: false`. `promote_rebuild_candidate.py` pins all 32 cells with repo-relative paths and digests. The api `.env` carries the same C2. |
| `backtest_assets` | `_check_backtest_assets`, line 348 | a `ml/backtest_results/backtest_*.csv` or rows in `backtest_results` | Already satisfied today (health lists only the two other causes). Untouched. |

Artifact digests are verified lazily by `resolve_stack` when a stack is loaded, not by this check, so nothing here weakens the SHA-256 pinning.

## Serving gap (read before step 14)

The running `api` container has the old `releases/` baked in and approves the old run id `stack_materialize_20260822T144443Z`. Step 14 materializes the new stack rows under a new run id that overwrites those rows (the upsert replaces `pipeline_run_id`). Until the new `current_baseline.json` is in a rebuilt `api` image and the api is restarted (step 16), the old api does not serve the overwritten stack rows. Expect empty or 503 projections for stack cells for roughly 10-20 minutes (estimate: one image build plus one restart). Do steps 14-16 back to back, in a quiet hour. Rollback is step 14's.

Before promotion, with the new code and no pinned attributions, default weights serve exactly as before and non-default weights return the unchanged number with reason `no_attribution` (`backend/app/services/weighting.py`). The api image is untouched until step 16, so the site keeps serving the old release the whole time.

## Timeline (estimates, 5 vCPU / 10 GB training container)

| Step | Wall time | Peak memory | Basis |
|---|---|---|---|
| 1-3 timers, backup, sync | 10-20 min | n/a | pg_dump of a few hundred MB (the 2026-08 dump was 530 MB with staging; unmeasured here) |
| 4 image build | 10-20 min, CPU-heavy (engine compile uses `-j$(nproc)`) | build only | unmeasured; do it when ATS is quiet |
| 5-6 alembic | 2-5 min (scratch restore dominates) | < 0.5 GB | |
| 7 backfill + Elo | 15-30 min | 1-2 GB (Elo enrichment holds about 130k x 225 rows in pandas) | unmeasured |
| 8-9 OOF rebuild | **about 20-26 h** | **0.35 GB per LightGBM fold** (measured), about 2 GB with CatBoost (assumption), TreeSHAP unmeasured. Expect 2-4 GB peak, far under 10 GB | `docs/perf/VPS1_MEASUREMENTS.md`: one LightGBM cell 193 s on the Mac, x2.5 for 5 capped vCPUs is about 8 min; CatBoost 2-4x that; 32 cells instead of the 15 measured, plus about 25% for attributions. `TIMINGS.tsv` gives the real per-cell numbers after the first cells. |
| 10-11 select, promote | 5-10 min | < 1 GB | |
| 12-13 commit, sync, rebuild | 20-30 min | build only | |
| 14 materialize + freeze | 10-20 min | 1-2 GB | |
| 15 evals + summary | 15-40 min | 1-2 GB | |
| 16 api build + restart | 5-10 min | | |
| 20 `weekly_refresh.py` | 10-30 min | under 1.5 GB expected (ETL RSS is unmeasured; the `etl` limit is 1.5 GB, so this runbook runs it in training) | |

Whole rerun: about 1.5 days elapsed, of which about 22 h is unattended training. The Fri 22:00 to Mon 06:00 window is 56 h, so it fits, but the timers are stopped for this run, so start it by hand.

## Run log 2026-11-14 (measured)

| Step | Measured |
|---|---|
| 2 backup | dump 36 MB, 40 tables (STAMP `20261114T101322Z`) |
| 7 backfill + Elo | feature_matrix 91,299 rows; Elo on 139,723 rows, 0 nulls; db container peaked 614 of 700 MB |
| 8-9 OOF rebuild | **51 min** for 32 cells (10:39:57 to 11:30:54 UTC), 36-233 s per cell; the 20-26 h estimate was about 25x too high |
| 10 selection | 31 `ridge_stack`, 1 `lgbm_identity` (`receiving_tds:WR`) |
| 14 materialize + freeze | 173,843 rows, run `stack_materialize_20261114T115324Z`, max residual 5.7e-13 |
| 14-16 serving gap | 6 min 47 s (11:53:23 to 12:00:10 UTC); api build 1 min 27 s |
| 15 evals | 161 s; 158/160 cell-seasons beat both naive and trailing-3 (misses: `rushing_tds:TE` 2023, `rushing_yards:QB` 2024, both to naive only); Marcel 157/160. Pre- and post-freeze CSVs byte-identical |

What the run needed that this runbook did not foresee:

- **`libgomp1`** was missing from both images (LightGBM import failed on cell 1). Fixed in `backend/Dockerfile` and `infra/hosts/vps1/Dockerfile.serving`, each with an import check after the apt purge.
- **A `%` in the DB password** crashed Alembic's configparser. `pipeline/schema.py` now escapes it. A crash here prints the URL with its password, so pipe box output through a `://user:pass@` redaction filter.
- **Hand-pinned image digests** on the box were not in the repo; commit them before syncing, or the sync reverts them.
- **The sync must exclude `infra/hosts/vps1/cloudflared.config.yml`**: the box keeps its own copy with the real hostname.
- **`SHA256SUMS` paths are absolute** (`/train_out/...`); strip the prefix before `shasum -c` on the Mac.
- **Hard-coded candidate paths in tests** (`causal_20260821_fixed`) broke the gate on C2; tests and `ml/served_learner.py` now read the active candidate from the manifest.
- **`--register` skips paths already in the manifest**, so `reports/materialize_stack_projections.json` kept its old hash. Re-hash that one entry by hand on the Mac; do not run `--update`, which also absorbs unrelated drift.
- **`/api/health` is the Pages proxy path.** Inside the container, `/health` is the public probe, and the per-check report is `/integrity` with `X-API-Key: $API_KEY`.
- `feature_matrix` was already `ok` at step 17, because the step 7 backfill writes a fresh `computed_at`.
- **Step 20 stops at step 7, and that is expected for now.** `weekly_refresh.py` passed steps 1-6 in 8.5 min (12:21 to 12:29 UTC) and then hit the forward-week structural gate: 0 `projections` rows for the target week. Step 4 re-materializes only the out-of-fold history; the forward producer (`build_forward_features.py` then `materialize_forward_projections.py`) is not wired in. Nothing was promoted, and served rows kept their approved run id. Decision 0011 records it. Re-enable the ETL timer only when someone decides to.

## Phase A: preparation (box is still serving the old release)

### 1. Quiet the schedulers

```bash
sudo systemctl stop gridiron-etl.timer gridiron-train-small.timer gridiron-train-full.timer
systemctl list-timers 'gridiron-*' --all          # all three inactive
systemctl is-active gridiron-etl.service gridiron-train-small.service gridiron-train-full.service   # all inactive
```

Rollback: `sudo systemctl start gridiron-etl.timer gridiron-train-small.timer gridiron-train-full.timer`. Do not leave them stopped past step 19.

### 2. Back up first

```bash
mkdir -p ~/rerun_backup && cd /srv/gridiron/infra/hosts/vps1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
docker compose exec -T db pg_dump -U oracle -Fc oracle > ~/rerun_backup/oracle_pre_rerun_$STAMP.dump
docker compose exec -T db pg_restore -l < ~/rerun_backup/oracle_pre_rerun_$STAMP.dump | head -5   # lists TOC
ls -l ~/rerun_backup/
tar czf ~/rerun_backup/srv_gridiron_releases_$STAMP.tgz -C /srv/gridiron releases
docker tag gridiron-etl:full gridiron-etl:pre-rerun-$STAMP
docker tag gridiron-api:serving gridiron-api:pre-rerun-$STAMP
sudo systemctl start gridiron-backup.service        # the off-box (R2) copy; check: journalctl -u gridiron-backup -n 20
```

The R2 copy is configured and passed on 2026-11-14, but it is unencrypted until the owner sets `AGE_RECIPIENT`. If the unit fails, stop and tell the owner: the local dump above is then the only copy, and it is on the same disk. Copy it to the Mac as well: `[Mac] scp $BOX:rerun_backup/oracle_pre_rerun_*.dump ~/Projects/Archive/backups/gridiron_oracle/`.
Output: `~/rerun_backup/` on the box.
Rollback for the whole runbook's data steps: `docker compose exec -T db pg_restore -U oracle -d oracle --clean --if-exists < dump` after stopping `api`.

### 3. Sync the box to C1

`[Mac]`, from a clean checkout of C1 (`git status --porcelain` is empty and `git rev-parse HEAD` is C1). Model files are Git LFS objects: pull them first, or rsync copies 130-byte pointer files and the box fails its digest check only when it serves.

```bash
git lfs pull
git lfs ls-files        # every releases/candidates/*/models/* line must show "*" (content present), not "-" (pointer)
python3 scripts/check_no_lfs_pointers.py
git ls-files -z | rsync -a --from0 --files-from=- ./ $BOX:/srv/gridiron/
ssh $BOX 'cd /srv/gridiron && python3 scripts/check_no_lfs_pointers.py'    # same check on the box, plain python3
```

The second check must print `ok: <n> model files, none is an LFS pointer`. It exits 1 and names the files otherwise, or when no model files exist at all. Repeat this block before every later sync (steps 13 and 18).

This sends exactly the tracked files. It does not use `--delete`, never touches `infra/hosts/vps1/.env` (untracked), and does not send ignored paths (`reports/`, `ml/oof/`). On the box, set the provisional commit and confirm no secret is printed:

```bash
cd /srv/gridiron/infra/hosts/vps1
vim .env     # GIT_COMMIT_SHA=<C1>; TRAIN_SMALL_CMD / TRAIN_FULL_CMD stay as they are (timers are stopped)
grep -c '^GIT_COMMIT_SHA=' .env
```

Output: code in `/srv/gridiron`. The `releases/` copied here is still the old release (C1 has not promoted anything).
Rollback: re-run the same command from the previous commit; the pre-rerun images and the `srv_gridiron_releases_*.tgz` tarball restore the release files.

### 4. Rebuild the images (CPU torch)

```bash
docker compose build etl          # builds gridiron-etl:full, which `training` also uses
docker run --rm --entrypoint python3 gridiron-etl:full -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expect a `+cpu` version and `None`. Do not build `api` yet: it keeps serving the old image until step 16.
Rollback: `docker tag gridiron-etl:pre-rerun-<STAMP> gridiron-etl:full`.

### 5. Scratch database: `alembic upgrade head` first

```bash
docker compose exec -T db createdb -U oracle oracle_scratch_0027
docker compose exec -T db sh -c 'pg_dump -U oracle -Fc oracle | pg_restore -U oracle -d oracle_scratch_0027 --no-owner'
docker compose --profile training up -d training
docker compose --profile training exec -T training bash -c \
  'python3 scripts/migrate.py upgrade --db-url "${DATABASE_URL%/oracle}/oracle_scratch_0027"'
docker compose exec -T db psql -U oracle -d oracle_scratch_0027 -Atc \
  "select version_num from alembic_version; select count(*) from information_schema.tables where table_name='projection_attributions'"
```

Expect `20261114_0027` and `1`. Then `docker compose exec -T db dropdb -U oracle oracle_scratch_0027`.
Rollback: drop the scratch database; production is untouched.

### 6. Production migration

```bash
docker compose --profile training exec -T training python3 scripts/migrate.py upgrade
docker compose exec -T db psql -U oracle -d oracle -Atc "select version_num from alembic_version"
```

Expect `20261114_0027`. The migration only adds `projection_attributions` and an index, so the running old api is unaffected.
Rollback: `docker compose exec -T db psql -U oracle -d oracle -c "drop index if exists idx_projection_attributions_run; drop table if exists projection_attributions; update alembic_version set version_num='20260822_0026'"` (this is the migration's own `downgrade`), or restore the step 2 dump.

### 7. Backfill history (data decision: restore the history into `oracle`)

Why this and not a separate training DB or the parquet cache: the box needs 2019-2023 rows in the same tables the ETL, trainers and the contract check already read, with one schema and one migration history; disk is not a constraint (120 GB free); the parquet cache holds 10 MB of feature matrices only, with no `game_logs` for the Elo fit. Measured with read-only SELECTs on 2026-11-14: the `oracle` and `oracle_hist_20260827` `feature_matrix` tables have the same 225 columns, the same row counts and per-column fill (within 1 point) for 2024-2025, and the history copy holds every 2019-2025 team-week with an Elo value.

```bash
docker compose --profile training exec -T training bash -c '
  export BACKFILL_SOURCE_URL="${DATABASE_URL%/oracle}/oracle_hist_20260827"
  python3 scripts/backfill_history.py                    # dry run: prints counts, rolls back
'
```

Read the counts with the owner, then:

```bash
docker compose --profile training exec -T training bash -c '
  export BACKFILL_SOURCE_URL="${DATABASE_URL%/oracle}/oracle_hist_20260827"
  python3 scripts/backfill_history.py --apply
  python3 -c "from pipeline.enrich_elo_embeddings import run_elo_enrichment as r; r()"
  python3 scripts/backfill_history.py --verify
  python3 scripts/verify_feature_contract.py --seasons 2019 2020 2021 2022 2023 2024 2025
'
```

Use `run_elo_enrichment`, not `python -m pipeline.enrich_elo_embeddings`: its `main()` also calls the player-embedding step, which raises by design. The Elo fit reads all `game_logs` (now 2019-2025), so ratings are as-of values over full history. Then check that the six Elo columns are populated for every season (expect 0 nulls and 0 rows stuck at the 1500 start value after 2019 week 1):

```bash
docker compose exec -T db psql -U oracle -d oracle -c "
  select season, count(*) rows, count(*)-count(team_off_elo) null_off, count(*)-count(opp_def_elo) null_opp_def,
         count(*) filter (where team_off_elo=1500 and team_def_elo=1500) at_start
  from feature_matrix group by season order by season"
```

Expect `null_*` = 0 for 2019-2026. `at_start` should be non-zero only in 2019 (week 1 and any team's first appearance). Anything else means a team-name mismatch (the LA/LAR gap) fell through to the start rating; stop and report it.
Why step 6 of the ETL now passes: see `backend/tests/test_team_game_history_elo.py`. The training frame has real Elo for every team-week, so the walk-forward folds no longer train on an Elo-less 2019.
Output: rows in `oracle.feature_matrix`, `game_logs`, `depth_charts` and the other history tables for 2019-2023. Existing 2024+ rows are never changed (`ON CONFLICT DO NOTHING`); the Elo columns of every season are recomputed.
Rollback: restore the step 2 dump (the backfill is one transaction, so a failed `--apply` leaves nothing behind; the Elo update commits in batches, so a failure there needs the restore).

## Phase B: training (about a day, unattended, resumable)

### 8. OOF rebuild, per cell, resumable

Start it detached, so a dropped SSH session does not kill it:

```bash
docker compose --profile training exec -T -d training bash -c \
  'bash scripts/train_window.sh full > /train_out/logs/rerun_driver.log 2>&1'
```

This runs the pinned `full` recipe: the feature contract on 2019-2025, then, for each of the 32 cells, LightGBM OOF + attributions, CatBoost OOF + attributions, and the Ridge stack with blended attributions, 20 Optuna trials per learner. The pointer `/train_out/full_stamp` names the rebuild in progress. `flock` makes a second start exit 75.

Watch:

```bash
docker compose --profile training exec -T training bash -c \
  'S=$(cat /train_out/full_stamp); cat /train_out/oof/rebuild_$S/TIMINGS.tsv; ls /train_out/oof/rebuild_$S/stack_*.csv | wc -l'
docker stats --no-stream $(docker compose ps -q training)
```

**Stagger so ATS and statuta keep headroom.** The training container is already capped at 5 CPUs and `cpu_shares: 128`. Beyond that:

- Pause between cells (nothing lost): `docker compose --profile training exec -T training touch /train_out/STOP`. The script exits 75 before the next cell, after the one in progress finishes (one cell takes about 10-40 min, estimate). To resume: `exec -T training rm /train_out/STOP`, then run the same start command again; it picks up `full_stamp`, skips finished cells, and skips base learners whose OOF and attribution files exist.
- Freeze instantly (keeps memory, stops CPU): `docker compose --profile training pause training`; `unpause` to continue.
- Hard stop mid-cell: `docker compose --profile training restart training` loses only the cell in progress; the same start command resumes.

A paused or killed run is never recorded as complete: `last_full_stamp` is written only when all 32 stack files exist. `ONLY_CELLS` is ignored in the `full` recipe.

Output: `/train_out/oof/rebuild_<STAMP>/` (on the `train_out` volume): `stack_*.csv`, `attr_*.parquet`, ridge coefficient JSONs, `TIMINGS.tsv`, `SHA256SUMS`; logs in `/train_out/logs/rebuild_<STAMP>/`; `last_full_stamp` when complete.
Rollback: `rm -rf` the rebuild directory and `full_stamp`. Nothing served has changed.

### 9. Check the rebuild and archive it

```bash
docker compose --profile training exec -T training bash -c \
  'S=$(cat /train_out/last_full_stamp); cd /train_out/oof/rebuild_$S; ls stack_*.csv | wc -l; ls attr_stack_*.parquet | wc -l; sha256sum -c SHA256SUMS | grep -vc ": OK$"'
```

Expect `32`, `32`, `0`. Archive off the box: `[Mac] ssh $BOX 'cd /srv/gridiron/infra/hosts/vps1 && docker compose --profile training cp training:/train_out/oof/rebuild_<STAMP> -' | tar -x -C ~/Projects/Archive/backups/gridiron_oracle/`.

## Phase C: promote, commit, bring the release home

### 10. Choose non-regressing stacks (in place)

```bash
S=<STAMP>   # contents of last_full_stamp
docker compose --profile training exec -T training python3 scripts/select_nonregressing_stacks.py \
  --candidate-dir /train_out/oof/rebuild_$S \
  --baseline /train_out/oof/rebuild_$S/bootstrap_lgbm_reference/frozen_baseline.json
```

Cells that fall back to the LightGBM identity get their attributions rewritten to the LightGBM learner's. Output: `CONSTRAINED_STACK_SELECTION.json` and the reference directory in the rebuild.
Rollback: it edits the rebuild only; re-run step 8 for a clean copy if needed.

### 11. Promote the candidate

```bash
docker compose --profile training exec -T training python3 scripts/promote_rebuild_candidate.py \
  --rebuild-dir /train_out/oof/rebuild_$S --release-id causal_20261114 --dry-run
docker compose --profile training exec -T training python3 scripts/promote_rebuild_candidate.py \
  --rebuild-dir /train_out/oof/rebuild_$S --release-id causal_20261114
```

It validates all 32 cells (and that each attribution row adds up to its stack row), copies them to `/app/releases/candidates/causal_20261114/` **inside the container**, re-points `releases/current_baseline.json` `artifacts` at the copies with repo-relative paths and fresh `artifact_digests`, and removes the earlier stack run ids from `approved_pipeline_run_ids`. It prints the `freeze_env` line: keep it for step 14.
Do not recreate the training container before step 12 finishes, or these files are lost (they live in its writable layer; the OOF is safe on the volume and promote can be re-run).
Rollback: `docker compose --profile training exec -T training rm -rf /app/releases/candidates/causal_20261114` and recreate the container; the box's `releases/` on disk is unchanged until step 13.

### 12. Bring the release back to the Mac and commit it (C2)

Copy out of the container to the box's release tree, then to the Mac:

```bash
# box
docker compose --profile training cp training:/app/releases/candidates/causal_20261114 /srv/gridiron/releases/candidates/causal_20261114
docker compose --profile training cp training:/app/releases/current_baseline.json /srv/gridiron/releases/current_baseline.json
```

```bash
# [Mac] in the repo checkout, on the integration branch
rsync -a $BOX:/srv/gridiron/releases/candidates/causal_20261114/ releases/candidates/causal_20261114/
rsync -a $BOX:/srv/gridiron/releases/current_baseline.json releases/current_baseline.json
python3 - <<'EOF'
import json, hashlib, pathlib
d = json.load(open("releases/current_baseline.json"))
bad = [p for p, h in d["artifact_digests"].items() if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() != h]
print("digest mismatches:", bad); assert not bad
EOF
git add releases/candidates/causal_20261114 releases/current_baseline.json
git commit -m "release: causal_20261114 candidate (32 cells, with bucket attributions)"
git rev-parse HEAD     # this is C2
```

`current_baseline.json` here is the promote output (re-pointed pins, stale `git_commit` and `integrity`); the freeze in step 14 rewrites it. Committing the candidates first keeps the lineage rule satisfied: they sit at or below the commit the freeze records, and only evidence files come after it (`runtime_status.py:40-44` treats `releases/candidates/` as non-evidence on purpose).
Rollback: `git reset --hard HEAD~1` on the Mac (never pushed).

### 13. Sync C2, set `GIT_COMMIT_SHA`, rebuild, recreate training

`[Mac]`: run the step 3 block (`git lfs pull`, the pointer check, then `git ls-files -z | rsync -a --from0 --files-from=- ./ $BOX:/srv/gridiron/`). Then on the box:

```bash
cd /srv/gridiron/infra/hosts/vps1
vim .env                         # GIT_COMMIT_SHA=<C2>
docker compose build etl
docker compose --profile training up -d --force-recreate training
docker compose --profile training exec -T training bash -c 'echo $GIT_COMMIT_SHA; ls releases/candidates/causal_20261114 | wc -l'
```

The OOF rebuild stays on the `train_out` volume. This sync carries exactly the step 12 files, so it cannot overwrite the promoted release with an older one. Verify anyway: `[Mac] git show HEAD:releases/current_baseline.json | shasum -a 256` must equal `sha256sum /srv/gridiron/releases/current_baseline.json` on the box.
Rollback: `.env` back to the old SHA, `docker tag gridiron-etl:pre-rerun-<STAMP> gridiron-etl:full`, recreate training.

## Phase D: materialize, freeze, serve (the serving gap: do 14-16 back to back)

### 14. Materialize all 32 cells with attributions, then freeze

Get the freeze env from step 11's `freeze_env` line: `APPROVED_PIPELINE_RUN_IDS=<new materialize run id>,season_sim_2026_1_20260823T000324_8fb263f0,weekly_auto_2026`. The kept ids must include `weekly_auto_2026`, or `weekly_refresh.py` fails closed at its `approval_check`.

```bash
docker compose --profile training exec -T training python3 scripts/materialize_stack_projections.py
docker compose --profile training exec -T training bash -c \
  'python3 -c "import json;r=json.load(open(\"reports/materialize_stack_projections.json\"));print(r[\"pipeline_run_id\"],r[\"git_commit\"],r[\"attributions\"])"'
```

Expect all 32 cells, `git_commit` = C2, `max_residual` under the tolerance. Then freeze, approving the new run id plus the kept ids:

```bash
docker compose --profile training exec -T training bash -c \
  'APPROVED_PIPELINE_RUN_IDS="<run id from the line above>,season_sim_2026_1_20260823T000324_8fb263f0,weekly_auto_2026" python3 scripts/freeze_baseline.py'
docker compose --profile training exec -T training python3 -c \
  "import json;d=json.load(open('releases/current_baseline.json'));print(d['git_commit'],d['git_dirty'],d['git_provenance'],d['projection_policy']['approved_pipeline_run_ids'])"
```

Expect `git_commit` = C2, `git_dirty` False, `git_provenance` `env:GIT_COMMIT_SHA`. Copy the result to the box release tree (no Mac commit yet):

```bash
docker compose --profile training cp training:/app/releases/current_baseline.json /srv/gridiron/releases/current_baseline.json
```

Rollback: stop the api (`docker compose stop api`), restore the step 2 dump (`pg_restore --clean --if-exists`), `docker compose up -d api`. The old api image is still tagged `gridiron-api:pre-rerun-<STAMP>`.

### 15. Evals and summary

```bash
docker compose --profile training exec -T training python3 scripts/run_causal_evals.py
docker compose --profile training exec -T training python3 scripts/summarize_causal_evals.py --check
docker compose --profile training exec -T training python3 scripts/verify_evidence_manifest.py --register \
  'reports/eval_causal_*.csv' reports/causal_baseline_beats.json reports/materialize_stack_projections.json \
  --produced-by scripts/run_causal_evals.py --description "Causal rerun 2026-11-14: per-cell baseline comparison and materialization report"
docker compose --profile training exec -T training python3 scripts/verify_evidence_manifest.py
```

The run refuses stale `eval_causal_*.csv` files, fails on any cell with no scored rows, and the summary counts only files this run wrote. The headline beat-count comes from `reports/causal_baseline_beats.json`; the old 74/75 count is not reused.
Output (inside the container, `/app`): `reports/eval_causal_<stat>_<position>.csv` x 32, `reports/causal_baseline_beats.json`, `reports/materialize_stack_projections.json`, and `releases/artifacts/MANIFEST.json` (entries `provisional`).
Rollback: delete the `reports/` files; nothing served depends on them.

### 16. Rebuild the api image and restart it

```bash
# box; confirm the api sees C2 and the new release
grep -c '^GIT_COMMIT_SHA=' .env
docker compose build api
docker compose up -d api
sleep 45
curl -fsS http://localhost:8000/health >/dev/null 2>&1 || docker compose exec -T api curl -fsS http://localhost:8000/health
```

`api` has no published port; the tunnel reaches it over the internal network, so the `exec` form is the one to use. The api image now bakes the freeze's `current_baseline.json` and the 32-cell candidate. This ends the serving gap.
Rollback: `docker tag gridiron-api:pre-rerun-<STAMP> gridiron-api:serving && docker compose up -d api` (and the step 14 DB restore, since the old api needs the old run id's rows).

### 17. Health checks

```bash
docker compose exec -T api sh -c 'curl -fsS -H "X-API-Key: $API_KEY" http://localhost:8000/integrity' | python3 -m json.tool
curl -fsS https://gridiron-api.risshabh.com/health | python3 -m json.tool       # through the tunnel (public probe)
```

Expected now: `baseline_manifest.status` is `ok` ("matches clean HEAD and full cell matrix"), `database` ok, `backtest_assets` ok. `feature_matrix` is still `warn` if its newest `computed_at` is over 24 h old, so `artifact_mode_ready` is `false` here. That is the honest state until step 20. If `baseline_manifest` is anything but `ok`, stop: the likely causes are a `GIT_COMMIT_SHA` that differs from the manifest's `git_commit`, or a freeze run from a different commit.
Also check the weights path: with a non-default `bucket_weights` setting, `/api/predict` for a player should now move (reason is no longer `no_attribution`).

## Phase E: evidence home, then the weekly refresh

### 18. Commit the evidence (C3) and re-sync

```bash
# [Mac] copy the produced files out of the container via the box, then pull
# box:
mkdir -p /srv/gridiron/rerun_out && docker compose --profile training cp training:/app/reports /srv/gridiron/rerun_out/
docker compose --profile training cp training:/app/releases/artifacts/MANIFEST.json /srv/gridiron/rerun_out/MANIFEST.json
docker compose --profile training cp training:/app/releases/baselines /srv/gridiron/rerun_out/baselines
# [Mac]:
rsync -a $BOX:/srv/gridiron/rerun_out/reports/ reports/
rsync -a $BOX:/srv/gridiron/rerun_out/MANIFEST.json releases/artifacts/MANIFEST.json
rsync -a $BOX:/srv/gridiron/releases/current_baseline.json releases/current_baseline.json
python3 scripts/verify_evidence_manifest.py
git add releases/current_baseline.json releases/artifacts/MANIFEST.json
git commit -m "release: causal_20261114 frozen baseline and evidence manifest"
git rev-parse HEAD     # C3
```

Commit only those two files. `reports/` is gitignored (the manifest lists its hashes); the `releases/baselines/<ts>.json` snapshot stays in `rerun_out` (archive it with the OOF) because `releases/baselines/` is not an evidence-only path in `_EVIDENCE_ONLY_PATHS`, so committing it would make a local, worktree-based readiness check reject C3 as "changes non-evidence paths". The lineage rule then holds on the Mac: the manifest's pinned C2 is an ancestor of C3 and C2..C3 touches only `releases/current_baseline.json` and `releases/artifacts/MANIFEST.json`.
Then sync C3 and prove the release was not overwritten:

```bash
# [Mac]
git lfs pull && python3 scripts/check_no_lfs_pointers.py
git ls-files -z | rsync -a --from0 --files-from=- ./ $BOX:/srv/gridiron/
git show HEAD:releases/current_baseline.json | shasum -a 256
# box
sha256sum /srv/gridiron/releases/current_baseline.json        # must be identical
```

**Rule for every later sync:** only sync a Mac commit that contains C3 (a descendant). Syncing an older commit would put the old `current_baseline.json` and candidates back on disk; the next image build would then serve them. `GIT_COMMIT_SHA` stays C2 until the next re-freeze; do not bump it to C3 or readiness reports `warn` (manifest commit differs from the named commit and there is no worktree to check lineage).
Rollback: `git reset --hard C2` on the Mac and re-sync, then redo step 14 and 16 (or restore the step 2 dump).

### 19. Rebuild the images from C3 and re-enable the timers

```bash
docker compose build etl api
docker compose up -d api
docker compose --profile training up -d --force-recreate training
```

The `etl` one-shot and the `training` service now carry the frozen baseline in their image, so a new container sees it. Re-enable the schedulers only after step 20 succeeds.

### 20. `weekly_refresh.py`, end to end

Run it in `training` (10 GB; the `etl` limit of 1.5 GB is unmeasured for this job):

```bash
docker compose --profile training exec -T training python3 scripts/weekly_refresh.py 2>&1 | tee ~/rerun_backup/weekly_refresh_$(date -u +%Y%m%dT%H%M%SZ).log
echo "exit ${PIPESTATUS[0]}"
```

Expected: exit 0 and no `FAILED` line. Step 4 materializes with attributions under `weekly_auto_2026_pending`; step 6 (`materialize_team_game_predictions`) now passes because 2019-2025 Elo is real; step 8 relabels to `weekly_auto_2026`, which the manifest approves. A failure writes a dead letter (`select * from dead_letter order by id desc limit 5`) and promotes nothing.
Then, within 24 h of this run's `computed_at`:

```bash
docker compose exec -T api curl -fsS http://localhost:8000/health | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['artifact_mode_ready'], d['status'])"
```

Expect `True ok` (or `degraded` only from non-required checks such as MLflow, which this runbook does not touch). Then:

```bash
sudo systemctl start gridiron-etl.timer gridiron-train-small.timer gridiron-train-full.timer
vim .env     # confirm TRAIN_SMALL_CMD / TRAIN_FULL_CMD are the pinned recipes (see README "Training slice")
```

Rollback: the refresh only writes pending rows until step 8; to undo a promoted refresh restore the step 2 dump.

#### Forward producer in the weekly refresh (decision 0012)

Step 4b runs after the stack step and before the simulation. In this order, and the order matters:

1. Capture an availability snapshot (nflverse injuries plus weekly roster status), stored with its fetch time.
2. Build the forward features with `as_of` = now. The snapshot was fetched before `as_of`, so it is usable. A snapshot fetched after `as_of` would not be.
3. `ml.forward_serving.run_forward` runs guards G1-G5 and writes the week's projections and a `forecast_runs` row in one transaction under `weekly_auto_<season>_pending`.

Any refusal (a guard, a missing availability snapshot, a library-version mismatch) writes a dead letter named `weekly_refresh/forward_producer`, returns 1 and stops before the simulation. Step 7 then checks the target week only: rows for every pinned cell and every position, under the pending id, or under the approved id when the week was already promoted.

Code SHA: `GIT_COMMIT_SHA` from `.env` is passed as the `CODE_GIT_SHA` build arg for the `etl` and `training` images, so the producer records the commit the image was built from. A real forward run with `code_git_sha` `unknown` is refused (guard D8); only `--dry-run` may run unstamped. Rebuild the image after changing `GIT_COMMIT_SHA`.

`--allow-no-availability` (manual only, never in a unit) stores every row with status `Unknown` instead of refusing when no snapshot exists. The run is logged, and the report and summary carry `availability_override`. This needs migration 0029. The API returns the value in `availability_status`; showing it in the UI is a frontend change not made here.

Pinning the fitted base models: the producer needs `artifacts.base_models` in `releases/current_baseline.json`. Add them with `scripts/pin_artifact_group.py --group base_models ...` and the re-freeze steps in "Adding a pinned group to the served release (re-freeze)" below; do not edit the manifest by hand.

Inactives: the NFL publishes game-day inactives about 90 minutes before kickoff. The Tuesday refresh sees only injury reports, so a Questionable player is served with his status shown. Neither the weekly refresh nor the Friday job sees inactives; do not expect either to.

#### Friday availability refresh (`gridiron-availability.timer`, not enabled)

`scripts/refresh_availability.py` fetches a new availability snapshot and changes only `availability_status` on the next forward week's rows, in one transaction, recorded in `forecast_runs` as `availability_refresh_<season>_w<WW>_<stamp>`. It never changes a prediction. It works per game: rows of games that have kicked off (or have a score) are frozen, and only games that have not started are updated, so a Thursday game never blocks Friday's update of Sunday and Monday games. It refuses (exit 1, nothing written) only when no game of the week is still open or the week has no forward rows. The report and the `forecast_runs` summary carry `games_updated` and `games_frozen`. Run it by hand once, in the same container as step 20, before enabling the timer:

```bash
docker compose --profile jobs run --rm availability
echo "exit $?"
```

Then `sudo cp systemd/gridiron-availability.* /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now gridiron-availability.timer`. Fri 22:00 UTC is also the full retrain's start. The unit declares no `Conflicts=` and never stops training: the compose service takes the same `flock` as `scripts/train_window.sh` (`/train_out/.train.lock`) and waits up to `AVAILABILITY_LOCK_WAIT_SECONDS` (default 900) for it. On timeout it exits 75, the unit fails and no `HEALTHCHECKS_AVAILABILITY_URL` ping is sent, so the missing ping is the alert. The trainers wait up to `TRAIN_LOCK_WAIT_SECONDS` (default 120) for the lock instead of skipping their window. A retrain holds the lock for up to 56 h, so on a retrain weekend the refresh will time out unless the timer is moved earlier than the retrain start (open decision). Record the job's peak RSS on the first run and set `mem_limit` from it (`512m` is a guess).

#### `--replace-week` (manual only)

The producer never overwrites rows an approved run already serves. To rewrite a served week (for example after a data fix), take a dump first, then:

```bash
docker compose exec -T db pg_dump -U oracle -Fc oracle > ~/rerun_backup/pre_replace_$(date -u +%Y%m%dT%H%M%SZ).dump
docker compose --profile training exec -T training python3 scripts/weekly_refresh.py \
  --replace-week --dump-path /home/<user>/rerun_backup/pre_replace_<STAMP>.dump
```

`--dump-path` must be an existing, non-empty file, or the command refuses with exit 2 before touching anything. The delete and the new insert run in one transaction inside `run_forward`, so a failure leaves the old week served untouched; the dump is kept as a second safety net. It refuses a week that has scores or has kicked off. It is never in a systemd unit or compose file, and a test fails if one ever carries it.

## Adding a pinned group to the served release (re-freeze)

Used 2026-11-14 to pin `causal_evals` (C2 `a056891`, C3 `173cfe7`); decision 0012 will use it for `base_models`. A required runtime check that reads a group fails until the group has digests, and `freeze_baseline.py` refuses a blocked runtime, so pin first, then freeze.

1. On the Mac, commit the files under `releases/candidates/<release>/`, then pin them and commit (this is C2):
   `python scripts/pin_artifact_group.py --group <name> releases/candidates/<release>/<dir>/*`
   It refuses an existing group, paths outside `releases/candidates/`, uncommitted files and Git LFS pointers. Add `--dry-run` to check first.
2. Run the gate, secret-scan and push. Sync to the box (rule above: only descendants of the served C3).
3. On the box: check `training` is idle; back up `.env`, `current_baseline.json` and tag the `etl` and `api` images `pre-refreeze-<STAMP>`.
4. Set `.env` `GIT_COMMIT_SHA` to C2's full sha. `docker compose build etl`, then recreate `training`.
5. Freeze inside `training` with the current approved ids (read them from `projection_policy.approved_pipeline_run_ids`):
   `docker compose --profile training exec -T -e APPROVED_PIPELINE_RUN_IDS="<ids>" training python3 scripts/freeze_baseline.py`
   Check `git_commit` = C2, `git_dirty` false, the new group listed, `DATABASE_URL` masked.
6. `docker compose --profile training cp training:/app/releases/current_baseline.json /srv/gridiron/releases/current_baseline.json`; rsync it to the Mac, compare digests, commit (C3), gate, secret-scan, push.
7. Sync C3, `docker compose build api`, `docker compose up -d --no-deps api`. Check `/health` (`artifact_mode_ready` true), `/integrity` (the new group's check `ok`) and the endpoint that reads it.

The api image bakes the manifest, so the live frontend must already understand any new response shape before step 7; otherwise deploy the frontend in the same window. Rollback: restore the `.env` backup and retag the `pre-refreeze` images.

## Files that land

| Path | Where | Committed |
|---|---|---|
| `/train_out/oof/rebuild_<STAMP>/` | box volume | no (archived to `~/Projects/Archive/backups/gridiron_oracle/`) |
| `releases/candidates/causal_20261114/` (32 stacks, 32 attributions, 32 coefficient files, selection report, reference) | repo, image | yes, C2 |
| `releases/current_baseline.json` | repo, image | yes, C2 (promote output) then C3 (freeze output) |
| `releases/artifacts/MANIFEST.json` | repo | yes, C3 (new entries `provisional`) |
| `reports/eval_causal_*.csv`, `reports/causal_baseline_beats.json`, `reports/materialize_stack_projections.json` | repo (gitignored) | no, listed in `MANIFEST.json` |
| `releases/baselines/<ts>.json` | box `rerun_out/` | no (see step 18) |
| `projection_attributions` rows | DB | n/a |
