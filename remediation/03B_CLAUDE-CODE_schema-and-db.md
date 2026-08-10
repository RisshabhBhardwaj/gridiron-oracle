# 03B · CLAUDE CODE · Schema, DB and config

| | |
|---|---|
| **Order** | Wave 3, stream B |
| **Concurrency** | ⇄ **CONCURRENT** with `03A` and `03C` |
| **Depends on** | `02` merged |
| **Blocks** | `04`, `05` |
| **Findings** | C-12, C-33 |
| **Shared resources** | **Use a scratch Postgres.** `03A` owns writes to `:15439`. Never test migrations against the working DB. |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_03B_schema -b fix/03b-schema main
cd ../go_03B_schema

# scratch DB for migration testing — do NOT use :15439
docker run -d --name go_scratch_pg -e POSTGRES_USER=oracle -e POSTGRES_PASSWORD=oracle \
  -e POSTGRES_DB=oracle -p 15440:5432 postgres:15
```

Interpreter: `/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`.
Tear down with `docker rm -f go_scratch_pg` when done.

---

Read `CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md` rows C-12 and C-33. Do not touch
`backend/app/api/draft.py` — `03C` owns it, including the C-15 query fix.

## C-12 — Give Alembic actual ownership

Three reviews agreed there is no single schema authority. The ORM actively lies about the database, and
the repo's own guard for this is shipped failing.

- **Mutable history.** `alembic/versions/20260806_0001_initial_schema.py:14,22-31` dynamically imports
  today's `pipeline/schema_ddl.ALL_DDL`, so the meaning of a historical migration changes with the
  checkout. Freeze a **literal DDL snapshot** inside the revision.
- **Dual write path.** `pipeline/schema.ensure_schema:36-60` executes the same DDL directly and is called
  from ~10 runtime sites (materialize, train, normalize, feature engineering). Remove it from runtime;
  restrict to initial setup and test fixtures.
- **Incomplete DDL.** `ALL_DDL` lacks `fantasy_adp` and `team_coaching`. `0002` duplicates a column already
  added in `0001`. `0003`'s downgrade (`20260806_0003_draft_and_coaching.py:57-61`) drops `team_coaching`
  but leaks `fantasy_adp`. Fix all three.
- **ORM drift.** `FeatureMatrix` (`backend/app/models/production.py`) is missing **16** columns the database
  has — `age`, `career_games`, `carry_share`, `expected_pass_attempts`, `team_pace`, `years_exp` and others
  added by Alembic 0004 — and carries ~60 phantom ones. `backend/tests/test_kalman_tracker.py::TestSchemaSync`
  is **red on HEAD**. Reconcile the ORM against the post-`02` schema and get it green.
  - Note: `02` changed the feature set. Reconcile against the **rebuilt** schema, not the audit-era one.
- **Model the runtime tables.** `fantasy_adp` and `team_coaching` need ORM models and foreign keys.
- **Prove it.** Full `upgrade → downgrade → upgrade` cycle on the empty scratch DB at `:15440`. No review
  has ever done this; static analysis found drift and the local `0004` happy path proves nothing. Add this
  as a CI job.

## C-33 — MLflow port

`.env.example:10-11` documents MLflow on host port 5001; `infra/docker-compose.yml:19-31` publishes 15091.
Artifact readiness failed on connection refused at the documented port. Standardize host and container
URLs and document them in one place. Postgres 15439 is correct everywhere — leave it alone.

## Done criteria

- `upgrade → downgrade → upgrade` completes clean on an empty scratch DB, added as a CI job.
- `0001` contains frozen literal DDL; no runtime caller of `ensure_schema` outside setup/fixtures.
- `0002` duplicate removed; `0003` downgrade symmetric.
- `TestSchemaSync` green against the post-`02` schema; `fantasy_adp` and `team_coaching` modelled with FKs.
- MLflow reachable at the documented port; readiness gets past the artifact check.
