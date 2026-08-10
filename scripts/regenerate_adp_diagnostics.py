#!/usr/bin/env python3
"""Regenerate the complete ADP-agreement diagnostic from causal inputs.

The result contains every requested season, p-values, and the comparison gaps
needed to prevent an ADP-agreement number being cited as model skill.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.adp_eval import evaluate
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--source", default="historical")
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    parser.add_argument("--output", type=Path, default=Path("reports/adp_spearman_stack_ranks.json"))
    args = parser.parse_args()
    results = [asdict(evaluate(season, source=args.source, database_url=args.database_url)) for season in args.seasons]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "generated_by": "scripts/regenerate_adp_diagnostics.py",
        "method": "causal_preseason_projection_vs_adp_with_oracle_and_prior_baselines",
        "results": results,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
