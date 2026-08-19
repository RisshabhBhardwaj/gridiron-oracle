#!/usr/bin/env python3
"""Promote a fully gated candidate manifest into the active baseline."""
from __future__ import annotations
import argparse, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--gates', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'releases/current_baseline.json')
    args = p.parse_args()
    candidate = json.loads(args.candidate.read_text())
    gates = json.loads(args.gates.read_text())
    if gates.get('n_gates') != 75 or gates.get('n_passed') != 75 or any(not r.get('promote') for r in gates.get('reports', [])):
        raise SystemExit('Refusing promotion: all 75 zero-regression gates must pass')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit('Refusing promotion from tracked-dirty worktree')
    payload = {
        **candidate,
        'candidate': False,
        'release_status': 'promoted_zero_regression',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': head,
        'git_dirty': False,
        'model_version': 'causal_20260819_zero_regression',
        'product_mode': 'artifact_backed',
        'promotion_evidence': str(args.gates),
        'projection_policy': {'approved_pipeline_run_ids': [], 'require_posterior_samples': False, 'require_interval_columns': True},
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(args.output)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
