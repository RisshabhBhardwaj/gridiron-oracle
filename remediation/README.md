# Remediation programme — 35 findings, 10 sessions, 6 waves

Brief: `../CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md`. Read it before running anything.

## Filename convention

`<wave><stream>_<TOOL>_<topic>.md`

- **Same number, different letter → run concurrently.** `01A`, `01B`, `01C` are parallel.
- **Number with no letter → runs alone.** Wave 2, 4, 5, 6 block everything else.
- The tool is in the filename and in every prompt header.

| Order | Session | Tool | Concurrency | Findings |
|---|---|---|---|---|
| **01A** | Containment | Claude Code | ⇄ with 01B, 01C | C-05, C-06, C-16, C-17, C-20 |
| **01B** | As-of feature audit | **Codex** | ⇄ with 01A, 01C | *diagnoses* C-01, C-18 |
| **01C** | Hygiene + push gate | Claude Code | ⇄ with 01A, 01B | C-07, C-23, C-24, C-25, C-28, C-29, C-30, C-31, C-32 |
| **02** | Feature contract + retrain | Claude Code | ⛔ SOLO — blocks all | **C-01**, C-18, C-26 |
| **03A** | Serving correctness | Claude Code | ⇄ with 03B, 03C | C-04, C-09, C-10, C-11 |
| **03B** | Schema / DB / config | Claude Code | ⇄ with 03A, 03C | C-12, C-33 |
| **03C** | Draft product | Claude Code | ⇄ with 03A, 03B | C-02, C-03, C-14, C-15, C-22 |
| **04** | Gates + eval methodology | Claude Code | ⛔ SOLO | C-08, C-13, C-21, C-27, C-34 |
| **05** | Test hardening | Claude Code | ⛔ SOLO | C-19 (complete) |
| **06** | Independent re-audit | **Codex** | ⛔ SOLO | verifies all 35 |

C-35 is subsumed by C-11 and C-21 — no separate work.

```
 01A ─┐
 01B ─┼──► 02 ──┬─► 03A ─┐
 01C ─┘  (SOLO) ├─► 03B ─┼──► 04 ──► 05 ──► 06
                └─► 03C ─┘  (SOLO) (SOLO) (SOLO)
 ⇄ parallel              ⇄ parallel
```

## One-time setup before wave 1

```bash
cd ~/Projects/Active/gridiron_oracle

# clean up leftovers from the audit session
rm -f .git/index.lock .git/_writetest remediation/_deltest
git checkout -- data/adp/historical/          # spurious LF→CRLF rewrite, content identical

# tag the audited tree — every worktree below branches from this
git tag -a audit-baseline-2026-08-09 191e151 -m "Tree audited by three adversarial reviews"
git tag -a phase5-clean-stacks 30f997c -m "Phase-5 clean two-learner stacks"
git push origin main --follow-tags

# verify the antidote artifact set before anyone runs a trainer
cd /Users/risshabh/Projects/Active/_gridiron_archive/antidote_20260809_stacks
sha256sum -c MANIFEST.sha256
```

## Worktrees — required for every parallel wave

Two Claude Code sessions in one working tree will fight over the branch and the index. **Each
concurrent stream gets its own worktree.** Wave 1:

```bash
cd ~/Projects/Active/gridiron_oracle
git worktree add ../go_01A_containment -b fix/01a-containment audit-baseline-2026-08-09
git worktree add ../go_01C_hygiene     -b fix/01c-hygiene     audit-baseline-2026-08-09
git worktree add ../go_01B_audit                              audit-baseline-2026-08-09   # Codex, read-only, detached
```

Wave 3 (after 02 merges — branch these from `main`, not the tag):

```bash
git worktree add ../go_03A_serving -b fix/03a-serving main
git worktree add ../go_03B_schema  -b fix/03b-schema  main
git worktree add ../go_03C_draft   -b fix/03c-draft   main
```

Tear down as each merges: `git worktree remove ../go_01A_containment`.

**Worktrees do not carry `.venv_311`.** Point sessions at the main tree's interpreter by absolute
path: `/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`. Every prompt says this.

## Shared-resource rules for parallel waves

| Resource | Rule |
|---|---|
| Postgres `:15439` | Only **one** session writes at a time. Wave 1: nobody writes. Wave 3: 03A writes, 03B uses a **scratch DB**, 03C reads only. |
| `ml/oof/` artifacts | Only 02 writes. 01A deletes poisoned files; nothing else touches it. |
| MLflow | 03A only. |
| The antidote archive | Read-only, always. Nothing writes there. |

## Sequential fallback

Parallelism buys wall-clock and costs merge overhead and simultaneous usage burn. If you would rather
go one at a time, the order in the table is already correct — run 01A → 01B → 01C → 02 → 03A → 03B →
03C → 04 → 05 → 06 in a single tree, no worktrees needed. Only 01B and 06 need Codex.

## The gate between waves 2 and 3 — do not skip

Before starting wave 3, confirm 02 actually landed:

- Every feature column's source row is prior-game, prior-season or pregame-known — proven by re-running
  the 01B audit against the rebuilt `feature_matrix`.
- No row filter keys on a target-game quantity.
- Headline counts **recomputed from scratch**. Expect them to move.

**If 20/20, 25/25, 5/5 and 24/25 come back identical, the retrain did not happen** — cached features
or stale artifacts were reused. Treat an unchanged result as a failed retrain, not a success. This is
the single most likely way the whole programme quietly fails.
