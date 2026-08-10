# Post-leak-fix comparison

## Status

The causal rebuild is **written and verified but not served**. `releases/current_baseline.json`, promotion state, and materialized projections were intentionally not changed.

## Feature and cohort evidence

- Feature matrix: 129,128 rows before and after; season bounds 2019–2025.
- Default model columns: 111 → 72. `snap_pct_off` is absent from every model allowlist and has 0 non-null feature-matrix values after rebuild.
- Eligibility: old target-snap cohort 37,268 rows; new pregame `seas_games_played >= 1` cohort 38,236 rows (+968, +2.6%).
- Same-game equality audit passed against `game_logs`; the raw target-game snap column is cleared.
- Stack OOF meta-folds begin at 2021 because fold 0 (2020) is consumed to train the first causal stack fold. Base OOF fold 0 maps to 2020.

## Headline counts

| Family | Reported pre-fix | Recomputed legacy OOF | Causal rebuild |
|---|---:|---:|---:|
| fantasy_ppr | 20/20 | 4/20 | 4/20 |
| volume | 25/25 | 5/25 | 5/25 |
| passing | 5/5 | 1/5 | 1/5 |
| yardage | 24/25 | 5/25 | 5/25 |

The reported pre-fix column preserves the audited historical headline. The two recalculated columns use the same strict, finite-baseline rule: a stack must beat both previous-season and trailing-three-game MAE. The legacy re-score is included only to make the comparison semantics explicit; it is not causal evidence because the old run was contaminated by target-game participation.

## Per-cell MAE

| Cell | Old rows | New rows | Old MAE | New MAE | Common baseline rows |
|---|---:|---:|---:|---:|---:|
| carries/RB | 3987 | 5048 | 3.7397 | 4.0246 | 4263 |
| fantasy_ppr/QB | 2929 | 2733 | 5.7433 | 6.2770 | 2353 |
| fantasy_ppr/RB | 3987 | 5048 | 5.4696 | 5.4759 | 4263 |
| fantasy_ppr/TE | 4691 | 5058 | 4.0057 | 3.8780 | 4382 |
| fantasy_ppr/WR | 8964 | 9686 | 5.3004 | 5.1650 | 8099 |
| pass_attempts/QB | 2929 | 2733 | 6.0789 | 7.3206 | 2353 |
| passing_yards/QB | 2929 | 2733 | 53.4890 | 62.2255 | 2353 |
| receiving_yards/RB | 3987 | 5048 | 13.9913 | 12.6883 | 4263 |
| receiving_yards/TE | 4691 | 5058 | 17.1488 | 16.5944 | 4382 |
| receiving_yards/WR | 8964 | 9686 | 24.8379 | 24.2104 | 8099 |
| rushing_yards/QB | 2929 | 2733 | 12.5764 | 12.1206 | 2353 |
| rushing_yards/RB | 3987 | 5048 | 24.6444 | 23.4323 | 4263 |
| targets/RB | 3987 | 5048 | 1.6197 | 1.5580 | 4263 |
| targets/TE | 4691 | 5058 | 1.6434 | 1.6168 | 4382 |
| targets/WR | 8964 | 9686 | 2.0705 | 2.1211 | 8099 |

## Artifact handoff (not served)

- `ml/oof/rebuild_20260809T231500/stack_carries_RB_20260810.csv` — `7cd3669293ff6060a4f7fac62c7f4532deebbe13286cc691b62f3e7bd3e92e63`
- `ml/oof/rebuild_20260809T231500/stack_fantasy_ppr_QB_20260810.csv` — `2a662afd674d513a3bd709bb9195ffcc33c82933d60a37fd0ff7f727d9d7cc4b`
- `ml/oof/rebuild_20260809T231500/stack_fantasy_ppr_RB_20260810.csv` — `5f666e5cbe6cf56419582f079dbf52e3dc8e118b5af58f3e9bf7e06a32c744a1`
- `ml/oof/rebuild_20260809T231500/stack_fantasy_ppr_TE_20260810.csv` — `644e33ddbd788bc7d68e36c8fefb7096db61cf623113093a94632716b7622860`
- `ml/oof/rebuild_20260809T231500/stack_fantasy_ppr_WR_20260810.csv` — `8c9eda44beafe795b4b3584ec5123e34876a4133a29b67911566e7728ec592de`
- `ml/oof/rebuild_20260809T231500/stack_pass_attempts_QB_20260810.csv` — `faf731e9f963176d68a5a59e659322002fa65c38249716fece8d7747e86a1b3a`
- `ml/oof/rebuild_20260809T231500/stack_passing_yards_QB_20260810.csv` — `d88b79a13f2d95135a78a17846ec887358bc536634acaadad552aecae14ec8b1`
- `ml/oof/rebuild_20260809T231500/stack_receiving_yards_RB_20260810.csv` — `bb9f90f3346a8457fb11e34f50005d4dc84f4fb6a2b4f8d9b06191bf02eb11f1`
- `ml/oof/rebuild_20260809T231500/stack_receiving_yards_TE_20260810.csv` — `086469c68b075ef3a4103f574d1f9622399b3b6d48e6941e85800142cc4d81e8`
- `ml/oof/rebuild_20260809T231500/stack_receiving_yards_WR_20260810.csv` — `4fdd7961a5323e58c69aeb331e73cfe5bde94036e1258bfc7a9df1824cb64eb8`
- `ml/oof/rebuild_20260809T231500/stack_rushing_yards_QB_20260810.csv` — `c0bfc3da341262c6971b9c3165f753ca29f9e2e2933452393a1a2f21e4260b2f`
- `ml/oof/rebuild_20260809T231500/stack_rushing_yards_RB_20260810.csv` — `77794d48cefacd9bd182d462b37978a72cf388d6c15b00d64fa70cd02dcd4d56`
- `ml/oof/rebuild_20260809T231500/stack_targets_RB_20260810.csv` — `c663c9f886adfd368bfbe786592b23506918142fa0d57b82607fe70ff9d2adaf`
- `ml/oof/rebuild_20260809T231500/stack_targets_TE_20260810.csv` — `8c3b2204fe9adb30036503201f5f79d68bfcb745dd95e510672519d5f889be76`
- `ml/oof/rebuild_20260809T231500/stack_targets_WR_20260810.csv` — `18d6020a76f692c4ccaf43636aefc5714e3881ad3d2f4572e194416be9cdfb0a`

Promotion/gate evidence is intentionally withheld. Wave 04 owns the broken gate implementation; these artifacts must be pinned by Wave 03A only after its manifest and uncertainty fixes.
