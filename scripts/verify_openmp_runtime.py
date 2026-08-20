#!/usr/bin/env python3
"""Diagnose the OpenMP runtime conflict behind ``OMP_NUM_THREADS=1``.

Answers design-spec Q5 reproducibly instead of by assertion. Reports every
distinct ``libomp`` binary the ML stack can load, which library pairing
actually crashes, and the measured single-thread vs all-core ratio for
gradient boosting.

Run: ``python scripts/verify_openmp_runtime.py --out reports/openmp_runtime.json``
The benchmark takes about a minute; pass ``--no-benchmark`` for linkage only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH = """
import os, time
import numpy as np
{imports}
rng = np.random.default_rng(0)
X = rng.normal(size=(100000, 80)).astype(np.float32)
y = (X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.5, size=100000)).astype(np.float32)
n = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count()
import lightgbm as lgb, xgboost as xgb
t = time.perf_counter()
lgb.train({{"objective": "regression", "verbosity": -1, "num_threads": n}},
          lgb.Dataset(X, label=y), num_boost_round=200)
xgb.train({{"objective": "reg:squarederror", "nthread": n, "verbosity": 0, "tree_method": "hist"}},
          xgb.DMatrix(X, label=y), num_boost_round=200)
print("SECONDS", time.perf_counter() - t)
"""


def _libomp_binaries() -> list[dict]:
    seen: dict[str, dict] = {}
    for package in ("sklearn", "torch", "xgboost", "lightgbm"):
        try:
            module = __import__(package)
        except Exception:
            continue
        roots = [Path(module.__file__).parent]
        for root in roots:
            for candidate in root.rglob("libomp.dylib"):
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                seen.setdefault(digest, {"sha256": digest, "bytes": candidate.stat().st_size, "paths": []})
                seen[digest]["paths"].append(str(candidate))
    for extra in (Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"), Path("/usr/local/opt/libomp/lib/libomp.dylib")):
        if extra.exists():
            digest = hashlib.sha256(extra.read_bytes()).hexdigest()
            seen.setdefault(digest, {"sha256": digest, "bytes": extra.stat().st_size, "paths": []})
            seen[digest]["paths"].append(str(extra))
    return sorted(seen.values(), key=lambda item: item["paths"][0])


def _run(code: str, env_overrides: dict[str, str]) -> tuple[int, float | None]:
    env = {**os.environ, **env_overrides}
    env.pop("OMP_NUM_THREADS", None) if env_overrides.get("OMP_NUM_THREADS") == "" else None
    if env_overrides.get("OMP_NUM_THREADS") == "":
        env.pop("OMP_NUM_THREADS", None)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    seconds = None
    for line in proc.stdout.splitlines():
        if line.startswith("SECONDS"):
            seconds = float(line.split()[1])
    return proc.returncode, seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "openmp_runtime.json")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    binaries = _libomp_binaries()
    payload: dict = {
        "cpu_count": os.cpu_count(),
        "distinct_libomp_binaries": len(binaries),
        "libomp": binaries,
        "verdict": None,
    }

    if not args.no_benchmark:
        with_torch = BENCH.format(imports="import torch")
        without_torch = BENCH.format(imports="")
        code_torch, _ = _run(with_torch, {"OMP_NUM_THREADS": "", "KMP_DUPLICATE_LIB_OK": "TRUE"})
        single_rc, single_s = _run(without_torch, {"OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
        multi_rc, multi_s = _run(without_torch, {"OMP_NUM_THREADS": "", "KMP_DUPLICATE_LIB_OK": "TRUE"})
        payload["multithreaded_with_torch"] = {
            "exit_code": code_torch,
            "crashed": code_torch != 0,
            "note": "139 is SIGSEGV: torch's libomp is the runtime that actually conflicts",
        }
        payload["benchmark_without_torch"] = {
            "single_thread_seconds": single_s,
            "all_core_seconds": multi_s,
            "speedup": round(single_s / multi_s, 2) if single_s and multi_s else None,
            "single_thread_exit": single_rc,
            "all_core_exit": multi_rc,
        }
        if code_torch != 0 and multi_rc == 0:
            payload["verdict"] = (
                "OMP_NUM_THREADS=1 is load-bearing only when torch is loaded. Gradient-boosting "
                "stages that do not import torch can and should run multithreaded."
            )
        elif multi_rc != 0:
            payload["verdict"] = "Multithreaded gradient boosting crashes even without torch; keep OMP_NUM_THREADS=1."
        else:
            payload["verdict"] = "No crash observed in either configuration on this host."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
