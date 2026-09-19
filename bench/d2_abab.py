#!/usr/bin/env python3
"""Track D2: ABAB over `--mirror` off/on (docs/p4_dual_source.md).

One `deepmoe serve` process per cell, arms alternated off/on/off/on..., so a
warming drive or a background process moves both arms the same way. Each cell is
a tools/hitrate_bench.py run of one chat script.

    .venv/Scripts/python.exe bench/d2_abab.py --out bench/results/d2/abab \
        --script bench/results/hitrate/y_turns.json \
        --script bench/results/hitrate/long_turns.json \
        --pairs 3 --cache-slots 5100 --mirror E:\\models\\DeepSeek-V4.1-Flash

The number reported per cell is decode tok/s and decode-only `nvme_stall`, both
straight out of the per-turn `done` events, plus the per-source byte split the
IoEngine wrote into status.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

SRC_RE = re.compile(
    r"src\[(\d+)\]\s+(\S+)\s+w\s+([\d.]+) GB/s\s+(\d+) req\s+([\d.]+) GiB \(([\d.]+)%\)"
    r"\s+mean lat ([\d.]+) ms")


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def cell_stats(out_dir):
    """Decode tok/s and decode-only per-token ms, averaged over the turns."""
    # serve writes `{"event": "done", ...}` -- NOT `"type"`. Reading the wrong key
    # made every cell report 0.0000 tok/s, which is why the D2 A/B never produced a
    # table even on the cells that ran to completion.
    events = [e for e in load_jsonl(os.path.join(out_dir, "events.jsonl"))
              if e.get("event") == "done" and e.get("decode_steps")]
    steps = sum(e["decode_steps"] for e in events)
    # `tok_s` on a done event is already decode-only; weight by that turn's steps
    # so a short turn does not count as much as a long one.
    ms = sum(e["decode_steps"] / e["tok_s"] * 1e3 for e in events if e.get("tok_s"))
    per = {}
    for e in events:
        for k, v in (e.get("per_token_ms") or {}).items():
            per[k] = per.get(k, 0.0) + v * e["decode_steps"]
    per = {k: v / steps for k, v in per.items()} if steps else {}
    hit = (sum(e.get("decode_hit_rate", 0.0) * e["decode_steps"] for e in events) / steps
           if steps else 0.0)

    sources = []
    sp = os.path.join(out_dir, "status.json")
    if os.path.exists(sp):
        blob = io.open(sp, encoding="utf-8", errors="replace").read()
        for m in SRC_RE.finditer(blob.replace("\\n", "\n")):
            sources.append({"i": int(m.group(1)), "root": m.group(2),
                            "weight": float(m.group(3)), "req": int(m.group(4)),
                            "gib": float(m.group(5)), "pct": float(m.group(6)),
                            "lat_ms": float(m.group(7))})
    return {"steps": steps,
            "tok_s": (steps * 1e3 / ms) if ms > 0 else 0.0,
            "per_token_ms": per, "hit": hit, "sources": sources}


def run_cell(args, arm, script, i):
    tag = f"{os.path.splitext(os.path.basename(script))[0]}_{arm}_{i}"
    out = os.path.join(args.out, tag)
    os.makedirs(out, exist_ok=True)
    cmd = [PY, os.path.join(REPO, "tools", "hitrate_bench.py"),
           "--script", script, "--out", out, "--exe", args.exe]
    if args.cache_slots:
        cmd += ["--cache-slots", str(args.cache_slots)]
    if args.max_context:
        cmd += ["--max-context", str(args.max_context)]
    for sa in args.serve_arg:
        cmd += [f"--serve-arg={sa}"]
    if arm == "on":
        cmd += ["--serve-arg=--mirror", f"--serve-arg={args.mirror}"]
        if args.weights:
            cmd += ["--env", f"DEEPMOE_MIRROR_WEIGHTS={args.weights}"]
    for e in args.env:
        cmd += ["--env", e]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    io.open(os.path.join(out, "cell.log"), "w", encoding="utf-8").write(
        (p.stdout or "") + (p.stderr or ""))
    if p.returncode != 0:
        print((p.stdout or "")[-3000:], file=sys.stderr)
        print((p.stderr or "")[-3000:], file=sys.stderr)
        raise SystemExit(f"cell {tag} exited {p.returncode}")
    s = cell_stats(out)
    s["cell_s"] = time.time() - t0
    s["tag"] = tag
    src = "  ".join(f"[{d['i']}]{d['pct']:.1f}%" for d in s["sources"])
    print(f"  {tag}: {s['tok_s']:.4f} tok/s  stall {s['per_token_ms'].get('nvme_stall', 0):.1f} ms"
          f"  hit {s['hit']:.4f}  {src}  ({s['cell_s']:.0f} s)", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--script", action="append", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--weights", default="", help="DEEPMOE_MIRROR_WEIGHTS, e.g. 4.6;1.0")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--cache-slots", type=int, default=5100)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--serve-arg", action="append", default=[],
                    help="extra `deepmoe serve` arg, added to BOTH arms (keep the A/B symmetric); "
                         "e.g. --serve-arg=--no-kv-disk so cell N's .pkv cannot leak into cell N+1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for script in args.script:
        for i in range(args.pairs):
            for arm in ("off", "on"):
                results.setdefault((script, arm), []).append(run_cell(args, arm, script, i))

    doc = {"mirror": args.mirror, "pairs": args.pairs, "cells": {}}
    print("\nscript                arm    tok/s     +-      d%    stall   hit     E: share")
    print("-" * 82)
    for script in args.script:
        base = None
        for arm in ("off", "on"):
            cells = results[(script, arm)]
            t = [c["tok_s"] for c in cells]
            mean = statistics.fmean(t)
            sd = statistics.pstdev(t)
            stall = statistics.fmean([c["per_token_ms"].get("nvme_stall", 0.0) for c in cells])
            hit = statistics.fmean([c["hit"] for c in cells])
            share = statistics.fmean(
                [next((d["pct"] for d in c["sources"] if d["i"] == 1), 0.0) for c in cells])
            if arm == "off":
                base = mean
            d = 100.0 * (mean / base - 1.0) if base else 0.0
            print(f"{os.path.basename(script):<20} {arm:<5} {mean:7.4f} {sd:6.4f} "
                  f"{d:+6.2f}% {stall:7.1f} {hit:6.4f} {share:9.1f}%")
            doc["cells"][f"{os.path.basename(script)}|{arm}"] = {
                "tok_s": t, "mean": mean, "sd": sd, "delta_pct": d,
                "nvme_stall_ms": stall, "hit": hit, "mirror_share_pct": share,
                "sources": cells[-1]["sources"]}
    with io.open(os.path.join(args.out, "abab.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"\n-> {os.path.join(args.out, 'abab.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
