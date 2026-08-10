# ABLATION CONTROL — do not delete, do not serve

This run is **not** superseded output. It is the control condition that isolates
the two channels of the C-01 leak, and it exists nowhere else.

| Run | Features | Cohort |
|---|---|---|
| `rebuild_20260809T220500` (this one) | new — `snap_pct_off` removed | **OLD** target-snap filter |
| `rebuild_20260809T231500` | new | **NEW** pregame eligibility rule |

Verified: every one of the 15 stacks here has row counts identical to the old
serving stacks (`ml/oof/stack_*_20260809.csv`) cell for cell — QB 2929,
RB 3987, WR 8964, TE 4691. The cohort filter was still in force when this ran.

Why it matters: the audit's C-01 had two channels — the leaked feature (1) and
the leaked cohort filter (2). The consolidated audit called channel 2 "the worse
one" and all three original reviews missed it. This run holds channel 2 fixed
while removing channel 1, so scoring it answers the question the headline
collapse turns on:

* If this run still scores ~74/75, the cohort filter accounts for the entire
  collapse and `snap_pct_off`-as-a-feature contributed essentially nothing.
* If it scores ~15/75, the feature mattered after all and the cohort story is
  wrong.

Session 06 should score this run. It is the cheapest decisive experiment
available on the 15/75 result.
