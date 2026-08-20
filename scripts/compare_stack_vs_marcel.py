#!/usr/bin/env python3
"""Walk-forward Marcel vs stack comparison (SP1 Q1).

Reads causal OOF CSVs. Does not infer max_train_season. Exits 2 if OOF is
missing rather than fabricating a win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-dir", type=Path, default=ROOT / "releases" / "candidates" / "causal_20260810")
    parser.add_argument("--history", type=Path, default=None, help="Optional parquet/csv of prior game logs")
    parser.add_argument("--stat", default="fantasy_ppr")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "marcel_vs_stack.json")
    args = parser.parse_args()

    import pandas as pd

    from ml.eval_causal import score_oof_against_baselines

    stacks = sorted(args.oof_dir.glob(f"stack_{args.stat}_*.csv"))
    if not stacks:
        print(f"No stack OOF for {args.stat} under {args.oof_dir}; refusing to invent Marcel comparison", file=sys.stderr)
        return 2
    frames = []
    prefix = f"stack_{args.stat}_"
    for path in stacks:
        frame = pd.read_csv(path)
        if "position" not in frame.columns:
            rest = path.stem[len(prefix):]
            frame["position"] = rest.split("_")[0]
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if "y_true" in oof.columns:
        oof = oof.rename(columns={"y_true": "actual"}) if "actual" not in oof.columns else oof
        if "actual" not in oof.columns:
            oof["actual"] = oof["y_true"]
    if args.history is None:
        history = oof.copy()
        if args.stat not in history.columns:
            source = "y_true" if "y_true" in history.columns else "actual"
            history[args.stat] = pd.to_numeric(history[source], errors="coerce")
    elif args.history.suffix == ".parquet":
        history = pd.read_parquet(args.history)
    else:
        history = pd.read_csv(args.history)
    scored = score_oof_against_baselines(oof, history, stat=args.stat)
    if scored.empty:
        print("Marcel comparison produced no rows", file=sys.stderr)
        return 2
    defined = scored["beats_marcel"].dropna() if "beats_marcel" in scored.columns else pd.Series(dtype=bool)
    wins = int(defined.astype(bool).sum())
    n = int(len(defined))
    payload = {
        "stat": args.stat,
        "n_cell_seasons": n,
        "beats_marcel": wins,
        "share": None if n == 0 else wins / n,
        "undefined_marcel_rows": int(len(scored) - n),
        "rows": scored.to_dict(orient="records"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"beats_marcel": wins, "n": n, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
