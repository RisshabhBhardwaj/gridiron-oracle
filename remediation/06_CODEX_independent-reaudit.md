# 06 · CODEX · Independent re-audit

| | |
|---|---|
| **Order** | Wave 6 — last |
| **Concurrency** | ⛔ **SOLO** |
| **Depends on** | everything merged |
| **Blocks** | the ship decision |
| **Findings** | verifies all 35 |
| **Tool** | **Codex, not Claude Code** — model diversity is the point. Claude Code did the fixes; a different model family checks them. |

### Setup

```bash
cd ~/Projects/Active/gridiron_oracle
git checkout main && git pull
git tag -a post-remediation-$(date +%Y-%m-%d) -m "Post-remediation state for independent re-audit"
git worktree add ../go_06_reaudit post-remediation-$(date +%Y-%m-%d)
```

Point Codex at `../go_06_reaudit`. Read-only. Interpreter:
`/Users/risshabh/Projects/Active/gridiron_oracle/.venv_311/bin/python`. Database read-only.

---

You are re-auditing `gridiron_oracle` after a remediation programme that responded to a 35-finding
consolidated audit (`CONSOLIDATED_ADVERSARIAL_AUDIT_2026-08-09.md`) and a feature-leakage rebuild. Read
that audit and `reports/POST_LEAK_FIX_COMPARISON.md` first.

**Read-only.** Do not edit; do not run trainers.

Your job is to determine whether the fixes are real, and to find what the remediation introduced or missed.
**Assume the fixes are cosmetic until proven otherwise** — that assumption has been correct before in this
repo. The original audit exists because two of three prior reviews passed a claim that was false.

## Priorities, in order

1. **Did the leak fix actually land?** Re-run the as-of feature audit
   (`remediation/01B_CODEX_as-of-feature-audit.md` has the method) against the rebuilt `feature_matrix`.
   Every feature column and every row filter. Prove with database joins that no feature equals a same-game
   postgame field and no filter keys on a target-game quantity.
   **Check specifically whether the headline eval counts changed.** If they came back identical to the
   pre-fix 20/20, 25/25, 5/5, 24/25, that is evidence the retrain reused cached features — not evidence of a
   robust model. Treat an unchanged result as a red flag, not a pass.
2. **Are the gates falsifiable?** Construct a deliberately regressed candidate and confirm rejection.
   Confirm missing evidence fails closed. Confirm no promotion record can be committed as `false`.
3. **Is inference fail-closed?** Remove an artifact; malform a coef file (missing intercept, `NaN`, unknown
   key); confirm it raises rather than silently serving a Kalman estimate. Confirm any fallback is stamped
   `degraded`.
4. **Is artifact selection deterministic?** `touch` a stale artifact and confirm selection does not move.
   Simulate a fresh-clone mtime tie. Confirm no `*_20260807` or legacy learner OOF has returned.
5. **Is the draft output a prediction?** Confirm target-season OOF is unreachable, measure rank↔games-played
   correlation, confirm 2026 returns a real board, and confirm ADP joins resolve on `player_id`.
6. **Is the schema single-authority?** Run `upgrade → downgrade → upgrade` on a scratch DB. Confirm no
   runtime `ensure_schema` caller. Confirm the ORM matches the database.
7. **Regressions.** Anything the remediation broke — especially in the ORM/migration reconciliation and the
   draft rank rewrite, the two largest rewrites in the programme.
8. **Are the tests real locks?** Spot-check `backend/tests/README.md`'s claims: pick three locks and verify
   they actually fail against `audit-baseline-2026-08-09`. A claimed lock that passes on both commits is a
   finding.

## Deliverable

1. **A verdict per original finding**: `FIXED` / `PARTIAL` / `NOT FIXED` / `REGRESSED`, for all 35, with
   evidence for each. Not the fixer's word — your own check.
2. **A findings table for anything new**, in the consolidated audit's format:
   ID / severity / workstream / evidence / impact / repro / fix.
3. **An explicit list of attacks that failed** — what you tried to break and could not. This is as valuable
   as the findings; the original audit's non-findings section is what made it trustworthy.
4. **A ship verdict** for 2026 draft use, with the specific conditions attached.

## Ground rules

- Mark anything you could not verify as **UNVERIFIED** rather than passing it. The two reviews that guessed
  `SAFE` on feature provenance are why this programme exists.
- Do not accept a passing test as evidence the underlying behaviour is correct — check that the test would
  fail if the behaviour regressed.
- Do not accept a report artifact as evidence; regenerate it from code where you can. Several original
  findings were reports that had drifted from the code that supposedly produced them.
- If the headline model edge is now small or absent, **say so plainly.** That is a legitimate and expected
  outcome of removing a feature with 0.637 correlation to the target. An honest small number is the goal
  here; a large number that survived the rebuild unchanged is the thing to be suspicious of.
