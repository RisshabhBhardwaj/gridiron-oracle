# Release Baselines

`current_baseline.json` is the active promoted manifest consumed by runtime integrity checks.
`artifact_invalidations.json` is the denylist for known-bad pipeline runs or projection cohorts.

Create or refresh it with:

```bash
python scripts/freeze_baseline.py
```

Artifact-backed deployments should set:

```bash
PRODUCT_MODE=artifact_backed
BASELINE_MANIFEST_PATH=releases/current_baseline.json
ARTIFACT_INVALIDATION_PATH=releases/artifact_invalidations.json
```
