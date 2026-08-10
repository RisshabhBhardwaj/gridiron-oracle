# Shipped evidence — policy and manifest

Audit finding **C-29**: `.gitignore` ignores `reports/`, `ml/oof/` and
`ml/experiments/`, while roughly twenty artifacts inside them are force-added.
The result is that re-running an eval produces **no `git status` signal** for
anything it newly writes, so the evidence backing a release drifts silently.
That is not hypothetical — `ml/experiments/reprojection_gate/gate_*.json` was
written at 15:46 and `releases/current_baseline.json` frozen at 17:14, and the
two describe different states of the system.

`MANIFEST.json` in this directory is the missing signal.

## The policy

1. **Every artifact offered as evidence for a release is listed in
   `MANIFEST.json`,** with its SHA-256, its size, the command that produced it,
   and one sentence saying what it is.
2. **A file in an ignored evidence directory that is not in the manifest is a
   defect.** `verify_evidence_manifest.py` reports it as `UNLISTED`. Either add
   it or delete it; do not leave evidence in a directory where Git will not
   mention it.
3. **Content drift against a `frozen` entry fails the build.** Regenerating an
   artifact is fine; regenerating it without re-freezing the manifest is not.
4. **New evidence goes here** — `releases/artifacts/` is not ignored, so a new
   file shows up in `git status` without a force-add.

## Statuses

| Status | Meaning | Drift |
|---|---|---|
| `frozen` | Regenerated from a clean rebuild and accepted as release evidence | **Fails** verification |
| `provisional` | Pre-remediation. Carried for traceability; not valid evidence | Reported, does not fail |

Every entry is currently `provisional`. The audit's C-01 finding
(`snap_pct_off` leaks the target game, as both a feature and the cohort filter)
invalidates every post-release OOF, stack, gate record and eval report as
*causal* evidence, so freezing any of it now would freeze a known-bad baseline.
Entries flip to `frozen` as they are regenerated.

## Usage

```bash
python scripts/verify_evidence_manifest.py            # verify (CI runs this)
python scripts/verify_evidence_manifest.py --strict   # provisional drift and UNLISTED fail too
python scripts/verify_evidence_manifest.py --update   # re-hash after a deliberate regeneration
```

`make verify-evidence` runs the plain verification.

## Scope note — `ml/oof/`

Nothing under `ml/oof/` is listed yet, and nothing there has been moved. Those
artifacts are being rebuilt by the feature-contract/retrain work; the structure
and the policy land first so the rebuild has somewhere to land. `WATCHED_DIRS`
in the verifier covers `reports/` and `ml/experiments/` today — add `ml/oof` to
it once the rebuilt artifacts are registered.

## Entry shape

```json
{
  "path": "reports/eval_causal_stack_targets_WR.csv",
  "sha256": "…",
  "bytes": 1234,
  "status": "provisional",
  "produced_by": "python -m ml.eval_causal",
  "description": "Per-season causal eval of a stack cell against naive and trailing-3 baselines."
}
```

`produced_by` is `"unknown"` on artifacts that were committed without recording
the command that made them. That is itself a finding: an evidence file whose
provenance nobody wrote down cannot be reproduced, and should be regenerated
rather than trusted.
