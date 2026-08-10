# 01B · CODEX · As-of feature audit (C-01 blast radius)

| | |
|---|---|
| **Order** | Wave 1, stream B — run first |
| **Concurrency** | ⇄ **CONCURRENT** with `01A` and `01C` (both Claude Code) |
| **Depends on** | nothing |
| **Blocks** | `02` — its output is pasted into that prompt |
| **Findings** | *diagnoses* C-01 scope, confirms C-18 siblings. **Fixes nothing — read-only by design.** |
| **Runtime** | one session |

### Setup — run before pasting the prompt

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_01B_audit audit-baseline-2026-08-09    # detached, read-only
```

Point Codex at `../go_01B_audit`. A pinned worktree is **required**: `01A` runs concurrently and
deletes files from `ml/oof/`, so the live tree shifts underneath an audit. Use
`/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python` for any read-only scripting.
Database is shared but read-only for this task, so no conflict. Tear down with
`git worktree remove ../go_01B_audit`.

---

You are auditing `gridiron_oracle` at commit `191e151`. This is a **read-only analysis task**. Do not
edit files. Do not run trainers — `ml/train_all_models.sh` and `scripts/train_*.sh` purge artifacts and
rebuild a killed model configuration.

## Context: one confirmed leak, unknown siblings

Three independent adversarial reviews were run on this repo. Two explicitly tested for leakage,
declared "no cross-season contamination," and passed the causal-eval claim. Both were wrong — they only
looked for leakage *across seasons* and never checked whether a feature reads from the game it is
predicting. The confirmed instance:

**`snap_pct_off` leaks the target game through two channels:**

1. **As a feature value** — `pipeline/feature_engineer.py:165` does
   `snap_pct = target_row.get("offense_pct")`, the postgame offensive snap share of the game being
   predicted. Assigned at `:258`; member of default `FEATURE_COLS` at `ml/utils.py:169`.
2. **As the cohort filter** — `ml/utils.py:382` (`load_feature_matrix`) and `ml/utils.py:459` (eval
   loader) both apply `df[snap.isna() | (snap >= 0.34)]`. The training and evaluation population is
   *defined by* target-game participation.

DB-confirmed: 125,425 of 125,425 non-null `snap_pct_off` values exactly equal same-game
`game_logs.offense_pct`; correlation with `actual_fantasy_ppr` = 0.637.

Channel 2 is the more damaging one and the one everyone missed: all three reviews "validated" the
headline results by recomputing on strict common cohorts, but every cohort was drawn from this filtered
population.

**Your job is to find every other instance of the same class.** I verified the snap path only. I did not
prove it is the only one.

## Deliverable 1 — `AS_OF_FEATURE_AUDIT.md`

One row per column in the default feature set, and one row per row-filtering predicate anywhere in the
training or eval load path:

| Column / filter | Defined at | Source field | Source row is | Verdict | Evidence |
|---|---|---|---|---|---|

`Source row is` ∈ **prior game** · **prior season** · **pregame-known** (scheduled, static, or published
before kickoff) · **target game** ← leak · **as-of unknown** ← treat as leak until proven otherwise.

`Verdict` ∈ `SAFE` · `LEAK` · `LATENT LEAK` (currently null or inert, but the code path would populate it
from the target game on a successful data refresh) · `UNPROVEN`.

Rank `LEAK` and `LATENT LEAK` rows by likely effect size — correlation with the target where you can
compute it read-only, otherwise by reasoning.

## Method

1. Enumerate the default feature set exactly: `FEATURE_COLS` in `ml/utils.py` (~line 121 onward). Include
   the Phase-4 group registries separately, marked opt-in, but audit them — they are promotion candidates.
2. For each column, trace backwards to the row it is read from. Builders: `pipeline/feature_engineer.py`
   (`build_feature_row` and bucket helpers in `pipeline/features/buckets.py`,
   `pipeline/features/feature_row.py`). The distinction that matters is `target_row` / same-`game_id`
   reads versus `prior_rows` / `all_season_rows` with a `week < current_week` or prior-season restriction.
3. Separately enumerate every predicate that *removes rows* in the training and eval load path
   (`ml/utils.py`, `ml/eval_causal.py`, `ml/eval_cohort.py`, `ml/baselines.py`). A filter keyed on a
   target-game quantity is a leak even when no feature carries the value.
4. Check enrichment passes that run *after* the batch build and write into the same rows —
   `FeatureEngineer.run()`'s `_enrich_*` steps, Elo population, and anything backfilling columns left
   `None` at build time.

## Known suspects — confirm or clear each explicitly

- **PBP enrichment**: `pipeline/pbp_pipeline.py:232-275` and `:730-734`, consumed at
  `pipeline/feature_engineer.py:804-810`, default features at `ml/utils.py:194-239`. Currently **all null
  in the database**, so they did not contaminate shipped artifacts — but the join looks same-game. **If a
  PBP refresh succeeds, does this become a live leak? This is the highest-value question in the audit.**
- **NGS fields** — same question, same shape.
- **Bucket 9 scheme interactions**: `compute_scheme_interactions(form, matchup, snap_pct)` at
  `feature_engineer.py:170` is passed the leaked `snap_pct` directly. Determine which derived columns
  inherit it — the comment at `ml/utils.py:177` names `blitz_exposure` as `snap_pct_off × opp_blitz_rate`.
- **Bucket 8 injury / availability** — `injury_status_encoded`, `games_missed_streak`. Practice
  participation is pregame; confirm the join does not pick up post-game status updates for the target week.
- **`years_exp`** — already confirmed corrupt by a different mechanism (present-day roster snapshot applied
  to all historical rows, 2,363 players constant). Include it, and check whether other roster-derived
  statics share the snapshot problem (`age`, `career_games`, draft fields, height/weight).
- **Rule coefficients** (`compute_rule_features(season)`) and **Elo** — Elo is populated lazily in bulk
  after the batch build (`feature_engineer.py:262-268`). Confirm ratings used for a game are as-of
  *before* that game, not end-of-season.
- **Phase-4 usage shares** — `pipeline/features/buckets.py:606-652` keys team-game aggregates by `game_id`
  alone, combining both teams and overwriting `game_team[gid]`; `compute_usage_shares:709-739` consumes
  that denominator. **Separately confirmed defect (C-18).** While in this file, check whether the same
  `game_id`-only keying appears anywhere else.

## Ground rules

- **Read-only.** No edits, no fixes, no refactors. Note things worth fixing in the table; do not do them.
- Database is read-only and useful: `postgresql://oracle:oracle@localhost:15439/oracle`. The strongest
  evidence is a join proving a feature column exactly equals a same-game postgame field, as was done for
  `snap_pct_off`. Produce that evidence where you can rather than reasoning from code alone.
- **Absence of a leak needs proof too.** "Looks fine" is not a verdict. If you cannot establish which row a
  value came from, mark it `UNPROVEN` and say what would settle it. **An honest `UNPROVEN` is more useful
  than a guessed `SAFE` — the last two reviews guessed `SAFE` on exactly this and that is why we are here.**
- Do not re-litigate `snap_pct_off`. Include it as row 1 for completeness and move on.

## Deliverable 2 — "What must be rebuilt"

Given your findings, state which artifacts are invalid and in what order they must be regenerated:
features → base OOF → stacks → evals → promotion evidence → materialized projections.

**If any leak is confined to an opt-in Phase-4 group and not in `FEATURE_COLS`, say so explicitly** — that
materially narrows the retrain scope and is worth a great deal.
