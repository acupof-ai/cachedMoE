#!/usr/bin/env python3
"""Track MS: the ABAB harness for tools/ms_bench.py (docs/p4_multistream.md).

One process per cell, arms alternated, so a slow machine or a warming drive
moves both arms the same way.

    .venv/Scripts/python.exe bench/ms_abab.py --out bench/results/ms/abab \
        --arm serial --arm interleave --pairs 3 \
        --script bench/results/hitrate/y_turns.json \
        --script bench/results/hitrate/long_turns.json \
        --turns 4 --max-tokens 64 --cache-slots 5100 --warm-cache
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def cell(args, arm, i):
    out = os.path.join(args.out, f"{arm}_{i}")
    os.makedirs(out, exist_ok=True)
    cmd = [PY, os.path.join(REPO, "tools", "ms_bench.py"), "--out", out, "--sched", arm,
           "--turns", str(args.turns), "--max-tokens", str(args.max_tokens), "--no-stop"]
    for s in args.script:
        cmd += ["--script", s]
    if args.cache_slots:
        cmd += ["--cache-slots", str(args.cache_slots)]
    if args.warm_cache:
        cmd += ["--warm-cache"]
    for e in args.env:
        cmd += ["--env", e]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    open(os.path.join(out, "cell.log"), "w", encoding="utf-8").write(
        (p.stdout or "") + (p.stderr or ""))
    if p.returncode != 0:
        print((p.stdout or "")[-3000:], file=sys.stderr)
        print((p.stderr or "")[-3000:], file=sys.stderr)
        raise SystemExit(f"cell {arm}_{i} exited {p.returncode}")
    s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    s["cell_s"] = time.time() - t0
    print(f"  {arm}_{i}: aggregate {s['aggregate_tok_s']:.4f} tok/s  "
          + "  ".join(f"s{j}={d['tok_s']:.3f}/hit {d['hit_rate']:.4f}"
                      f"/stall {d['per_token_ms']['nvme_stall']:.0f}"
                      for j, d in enumerate(s["per_stream"]))
          + f"  ({s['cell_s']:.0f} s)", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--script", action="append", required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--cache-slots", type=int, default=5100)
    ap.add_argument("--warm-cache", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)
    rows = {a: [] for a in args.arm}
    for i in range(args.pairs):
        print(f"pair {i}", flush=True)
        for a in args.arm:
            rows[a].append(cell(args, a, i))
    table = {}
    for a, rs in rows.items():
        agg = [r["aggregate_tok_s"] for r in rs]
        table[a] = {
            "n": len(rs),
            "aggregate_tok_s": statistics.mean(agg),
            "sd_pct": (statistics.stdev(agg) / statistics.mean(agg) * 100) if len(agg) > 1 else 0.0,
            "per_stream": [
                {"tok_s": statistics.mean([r["per_stream"][j]["tok_s"] for r in rs]),
                 "hit_rate": statistics.mean([r["per_stream"][j]["hit_rate"] for r in rs]),
                 "nvme_mb_per_token": statistics.mean(
                     [r["per_stream"][j]["nvme_mb_per_token"] for r in rs]),
                 "stall_ms": statistics.mean(
                     [r["per_stream"][j]["per_token_ms"]["nvme_stall"] for r in rs]),
                 "gpu_ms": statistics.mean(
                     [r["per_stream"][j]["per_token_ms"]["attn"] +
                      r["per_stream"][j]["per_token_ms"]["moe_gpu"] for r in rs]),
                 "script": rs[0]["per_stream"][j]["script"]}
                for j in range(len(rs[0]["per_stream"]))],
        }
    base = args.arm[0]
    for a in args.arm:
        table[a]["x_vs_" + base] = table[a]["aggregate_tok_s"] / table[base]["aggregate_tok_s"]
    with open(os.path.join(args.out, "abab.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"table": table, "cells": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(table, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
