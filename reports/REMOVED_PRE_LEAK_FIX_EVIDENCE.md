# Removed: pre-leak-fix causal evaluation CSVs

**Removed:** 2026-08-19 · **Recoverable from:** `f7eb25fc0c723aea12e786dde22cba91cd2e517b`

These 21 files were generated on 2026-08-06 and 2026-08-09 13:xx — *before* the
causal rebuild at `rebuild_20260809T231500` closed the C-01 `snap_pct_off` leak. Aggregated
they asserted **71/75 beats_naive**, a number produced by the contaminated model, and they
disagree with the promoted artifacts: `eval_causal_stack_fantasy_ppr_WR.csv` reported a 2021
model score of 2.662 where the promoted candidate's own selection record gives 5.305 for the
same cell-season.

They described a different object than the one being served, and left in `reports/` they
would keep being read as current evidence. The 2026-08-19 audit listed their deletion as
fix #7; the rebuild design lists it as SP1.1.

Current, non-contaminated evidence for the same question:

- `releases/candidates/causal_20260810/stack_*.csv` — the promoted OOF artifacts
- `reports/POST_LEAK_FIX_COMPARISON.md` — regenerated after the denominator fix
- `reports/marcel_vs_stack.json` — the weekly stack against a Marcel baseline

Regenerate with `scripts/train_yardage_oof.sh` / `scripts/summarize_causal_evals.py` if a
like-for-like historical comparison is ever needed.

| File | Bytes | SHA-256 (first 16) | Git |
|---|---:|---|---|
| `eval_causal_stack_carries_RB.csv` | 653 | `23ca08847b45623e` | tracked |
| `eval_causal_stack_fantasy_ppr_QB.csv` | 603 | `a992a77fab19f080` | untracked |
| `eval_causal_stack_fantasy_ppr_RB.csv` | 604 | `be32db264c19280b` | untracked |
| `eval_causal_stack_fantasy_ppr_TE.csv` | 604 | `207d32ef19baafd4` | untracked |
| `eval_causal_stack_fantasy_ppr_WR.csv` | 610 | `20191abb83ecce37` | untracked |
| `eval_causal_stack_fantasy_ppr_all.csv` | 2067 | `c497fa9807df7c25` | untracked |
| `eval_causal_stack_pass_attempts_QB.csv` | 685 | `6d42131f6d16adde` | tracked |
| `eval_causal_stack_passing_yards_QB.csv` | 613 | `bc2431d9b0efcd75` | tracked |
| `eval_causal_stack_phase5_fantasy_ppr_QB.csv` | 604 | `6c9ff48970058443` | untracked |
| `eval_causal_stack_phase5_fantasy_ppr_RB.csv` | 604 | `b33475d4a5dfd9c9` | untracked |
| `eval_causal_stack_phase5_fantasy_ppr_TE.csv` | 604 | `24aca83a94f9a50b` | untracked |
| `eval_causal_stack_phase5_fantasy_ppr_WR.csv` | 610 | `33ab4b05e2f46cbc` | untracked |
| `eval_causal_stack_phase5_fantasy_ppr_all.csv` | 2068 | `b31676127c848345` | tracked |
| `eval_causal_stack_receiving_yards_RB.csv` | 634 | `29a497193923c9d3` | tracked |
| `eval_causal_stack_receiving_yards_TE.csv` | 633 | `30895c94295b76ee` | tracked |
| `eval_causal_stack_receiving_yards_WR.csv` | 635 | `0e817e40ab3cdaef` | tracked |
| `eval_causal_stack_rushing_yards_QB.csv` | 622 | `ecd10f5d7c130dca` | tracked |
| `eval_causal_stack_rushing_yards_RB.csv` | 625 | `f8d8ab87aac269b5` | tracked |
| `eval_causal_stack_targets_RB.csv` | 657 | `47e6bdbd210cc035` | tracked |
| `eval_causal_stack_targets_TE.csv` | 661 | `baab28ac777b0a1b` | tracked |
| `eval_causal_stack_targets_WR.csv` | 662 | `640af635473e3cfa` | tracked |
