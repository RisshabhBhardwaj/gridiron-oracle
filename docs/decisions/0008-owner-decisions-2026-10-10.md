# 0008: Owner decisions of 2026-10-10 and their status

Type: Decision record (index of decisions and what is done)
Status: Accepted by the owner on 2026-10-10. Implementation status is listed per item and is the source of truth for progress.
Last verified against code: 2026-10-11
Date: recorded 2026-10-11

## Decisions

| # | Decision | Status on 2026-10-11 |
|---|---|---|
| 1 | Elo fix (implied win probability from the pregame snapshot, cap at LAST_COMPLETE_SEASON), then rerun and refreeze | Fix committed on `agent/gor-ml-05-causal-fixes` (52e8cde, 3321b21). NOT gated, NOT merged. Rerun and refreeze not done. |
| 2 | Forward rows use real prior games cut at as-of; opp_* from the target opponent's prior games | WIP on `agent/gor-fwd-06-prior-games` (not gated). Not merged. |
| 3 | Missing lines: replace the median fallback | Design approved as decision 0008's line snapshots: `captured_at` snapshots, latest before as-of, causal Elo-implied fallback. Not started (the lines stream stopped before any commit). |
| 4 | In-season acceptance: gate on the approved window; the VPS run builds 2026 artifacts | Forward-week structural gate not implemented (ops stream stopped early). |
| 5 | League is eight teams | Verified from Sleeper's public API (La Hospitale, 2025): 8 rosters; QB, 2 RB, 2 WR, TE, FLEX, K, DEF, 8 bench. `ml/vor.py` skill slots match. K and DEF are not modelled (out of scope). No change needed. |
| 6 | Wire the Settings weights now | Only the `ml/bucket_weights.py` module exists (WIP on `agent/gor-weights-08-wire`). Not wired. |
| 7 | Hold availability commit 04033cf | Held. Preserved as tag `archive/fix-availability-inversion-04033cf-2026-10-10`. |
| 8 | Forward opponent/weather/injury nulling | Superseded by #2 (not adopted). |
| 9 | Odds: move to backend `odds.py` with a quota guard | Not done (security stream stopped before any commit). The Vercel handler still serves `/api/odds`; it now refuses non-GET. |
| 10 | Draft wording: remove unreproducible claims | Approved. Status not re-verified on this branch. |
| 11 | Delete leftover branches | Done. Their commits are kept as tags `archive/*-2026-10-10`. |
| 12 | Branch protection on `main` | Not set. Needs a green CI on a pushed head first. |
| 13 | Base UI, Tailwind 4, all deferred work done now | Base UI accepted (0004). Tailwind 4 and the remaining controls not done (frontend stream stopped before any commit). |

Owner decisions also made (not in the numbered list): merge to local `main` without push or deploy. Railway services `api` and `weekly-refresh` deploy from `main` on push, so any push is a production deploy.

## Still needs an owner decision
- Optuna validation on the last fold's season (`ml/catboost_model.py` ~L307-312): approved in principle, not implemented.
- Yardage conformal intervals (`ml/train.py` ~L1177): approved in principle, not implemented.
- NaN fumble training target (`pipeline/feature_engineer.py` ~L322): approved in principle, not implemented.
