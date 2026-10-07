Type: Design
Status: Proposed (user types are inferred from routes and copy, not from known customers)
Last verified against code: 2026-10-07 (routes and copy read; no user research exists in the repo)

# Users

## 1. Fantasy drafter (preseason)
- Sees: `/draft` board with rank, projected points, market rank (ADP), value versus ADP; mock draft.
- Least needed: a ranked list per position with "who is a bargain" and a note when data is a fallback.
- Next action: pick a player or run a mock draft.
- Credibility metrics (measurable today): rank agreement with later actual finishes (`ml/adp_eval.py`, `reports/adp_spearman_*.json`) against ADP. Not measurable: money won, since no league results are collected.

## 2. Weekly manager (in season)
- Sees: `/projections`, `/player/:id`, game detail; range (floor/ceiling), boom/bust.
- Least needed: projected points, a range, one-line reason, and a clear "no forecast yet" state.
- Next action: start or sit a player.
- Metrics: error versus last-three-games average on past weeks (`/backtest`). Not measurable: lineup wins.

## 3. Analyst / skeptic
- Sees: `/backtest`, `/season` review: model error versus naive baseline by season.
- Least needed: one plain statement "better / not better than the simple method, by how much", with a link to the data.
- Metrics: same baseline comparisons; note evidence is marked provisional.

## 4. Bettor (implied by "Book Line", edge, Kelly engine)
- Sees: player page edge versus a book line; the line is empty ("Book Line 0") when none exists.
- Status: in scope (owner). No bet slip in the UI; engine link to live odds unverified.

## 5. Operator/developer
- Sees: `/health`, `/integrity`, MLflow, logs. Needs the commands in AGENTS.md.

## Metrics the product cannot yet measure
Money or time saved, league outcomes, user retention. No usage tracking or result capture was found.

## Jargon in the UI (file:line, suggested wording)
| Where | Now | Suggest |
|---|---|---|
| `frontend/src/pages/Settings.tsx:13` | Kalman Form (Player Ability) | Recent form |
| `frontend/src/pages/PlayerDetail.tsx:267` | Factor Attributions (SHAP) | What drives this projection |
| `frontend/src/pages/PlayerDetail.tsx:302` | Kalman Ability | Underlying skill estimate |
| `frontend/src/pages/PlayerDetail.tsx:263` | Confidence Interval | Likely range |
| `frontend/src/pages/PlayerDetail.tsx:239-240` | Floor (p10) / Ceiling (p90) | Bad week (1 in 10) / Great week (1 in 10) |
| `frontend/src/pages/SeasonReview.tsx:161` | Mean Absolute Error (MAE) | Average miss, in points |
| `frontend/src/pages/SeasonReview.tsx:207` | Stack MAE / Naive MAE | Our average miss / Simple-method average miss |
| `frontend/src/components/BacktestPanel.tsx:79` | Brier Score | Probability accuracy (lower is better) |
| `frontend/src/components/BacktestPanel.tsx:93` | Sharpe Ratio | Return per unit of risk (or remove) |
| `frontend/src/pages/DraftBoard.tsx:186` | Ranked via 8-team VOR | Ranked by value over a replacement player in an 8-team league |
| `frontend/src/components/PercentileFan.tsx:13` | conformal bounds unavailable | Range not available for this player |
| `frontend/src/components/shared/NoForecastBanner.tsx` | hard-coded "2026 in-season projections are not yet available" | Derive year and state from the API |
