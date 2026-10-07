# Runbook: check evidence and freeze a baseline

Type: Runbook
Status: Accepted
Last verified against code: 2026-10-07 (`scripts/verify_evidence_manifest.py` run; gate and freeze not run)

1. `make verify-evidence` (same as `python scripts/verify_evidence_manifest.py`). Fails on content drift of a frozen entry; lists unlisted files in ignored evidence folders.
2. Gate a candidate: `make reprojection-gate HOLDOUT=2024 POSITION=WR TARGET=fantasy_ppr`. Note: stored gate reports compare against a prior LightGBM reference release, not naive baselines.
3. `make verify-artifact-prerequisites`, then `make verify-readiness`.
4. `make freeze-baseline` writes `releases/current_baseline.json`.
5. Serve with `PRODUCT_MODE=artifact_backed`, `BASELINE_MANIFEST_PATH=releases/current_baseline.json`, `ARTIFACT_INVALIDATION_PATH=releases/artifact_invalidations.json`.
6. Known-bad runs: `python scripts/invalidate_artifacts.py --pipeline-run-id <id> --reason "<why>"` (records, does not delete).

Policy for evidence files: `releases/artifacts/README.md`. Baselines: `releases/README.md`.
