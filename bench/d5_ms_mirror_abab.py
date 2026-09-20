#!/usr/bin/env python3
"""Track D5: 2-stream aggregate with the mirror off vs on (docs/p4_dual_source.md §D5).

`bench/ms_abab.py` alternates *schedulers*; this one alternates the *mirror* with the
scheduler held at `pipeline` (the MS default), because the drive is the reason Track MS
stopped at +17.5%: the two-stream arm asks the same drive for 28% more bytes per second
and hits D:'s ceiling at 69%.

    .venv/Scripts/python.exe bench/d5_ms_mirror_abab.py --out bench/results/d5/ms \
        --script bench/results/hitrate/y_turns.json \
        --script bench/results/hitrate/long_turns.json \
        --mirror "E:\\models\\DeepSeek-V4.1-Flash" --pairs 2

One process per cell, arms alternated off/on/off/on so a warming drive moves both arms
the same way. The number reported is the aggregate tok/s of the two streams.
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
    tag = f"{arm}_{i}"
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    cmd = [PY, os.path.join(REPO, "tools", "ms_bench.py"), "--out", out,
           "--sched", args.sched, "--turns", str(args.turns),
           "--max-tokens", str(args.max_tokens), "--no-stop"]
    for s in args.script:
        cmd += ["--script", s]
    if args.cache_slots:
        cmd += ["--cache-slots", str(args.cache_slots)]
    if args.warm_cache:
        cmd += ["--warm-cache"]
    for sa in args.serve_arg:
        cmd += [f"--serve-arg={sa}"]
    if arm == "on":
        cmd += ["--serve-arg=--mirror", f"--serve-arg={args.mirror}"]
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
        raise SystemExit(f"cell {tag} exited {p.returncode}")
    s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    s["cell_s"] = time.time() - t0
    print(f"  {tag}: aggregate {s['aggregate_tok_s']:.4f} tok/s  "
          + "  ".join(f"s{j}={d['tok_s']:.3f}/hit {d['hit_rate']:.4f}"
                      f"/stall {d['per_token_ms']['nvme_stall']:.0f}"
                      f"/MB {d['nvme_mb_per_token']:.0f}"
                      for j, d in enumerate(s["per_stream"]))
          + f"  ({s['cell_s']:.0f} s)", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--script", action="append", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--sched", default="pipeline")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--cache-slots", type=int, default=0)
    ap.add_argument("--warm-cache", action="store_true", default=True)
    ap.add_argument("--serve-arg", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)

    rows = {"off": [], "on": []}
    for i in range(args.pairs):
        print(f"pair {i}", flush=True)
        for arm in ("off", "on"):
            rows[arm].append(cell(args, arm, i))

    table = {}
    for arm, rs in rows.items():
        agg = [r["aggregate_tok_s"] for r in rs]
        table[arm] = {
            "n": len(rs),
            "cells": agg,
            "aggregate_tok_s": statistics.fmean(agg),
            "sd_pct": (statistics.stdev(agg) / statistics.fmean(agg) * 100) if len(agg) > 1 else 0.0,
            "per_stream": [
                {"tok_s": statistics.fmean([r["per_stream"][j]["tok_s"] for r in rs]),
                 "hit_rate": statistics.fmean([r["per_stream"][j]["hit_rate"] for r in rs]),
                 "nvme_mb_per_token": statistics.fmean(
                     [r["per_stream"][j]["nvme_mb_per_token"] for r in rs]),
                 "stall_ms": statistics.fmean(
                     [r["per_stream"][j]["per_token_ms"]["nvme_stall"] for r in rs]),
                 "script": rs[0]["per_stream"][j]["script"]}
                for j in range(len(rs[0]["per_stream"]))],
        }
    table["on"]["x_vs_off"] = table["on"]["aggregate_tok_s"] / table["off"]["aggregate_tok_s"]
    table["on"]["delta_pct"] = 100.0 * (table["on"]["x_vs_off"] - 1.0)
    with open(os.path.join(args.out, "abab.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"mirror": args.mirror, "sched": args.sched, "table": table, "cells": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"\nmirror off {table['off']['aggregate_tok_s']:.4f} tok/s (sd {table['off']['sd_pct']:.2f}%)"
          f"  ->  on {table['on']['aggregate_tok_s']:.4f} (sd {table['on']['sd_pct']:.2f}%)"
          f"  = {table['on']['delta_pct']:+.2f}%", flush=True)
    print(f"-> {os.path.join(args.out, 'abab.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
