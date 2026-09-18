#!/usr/bin/env python3
"""Track Q1: per-cell P0-queue report over a tools/hitrate_bench.py output dir.

    python tools/p0q_report.py bench/results/p0q/<cell> [...]

Reads

  events.jsonl  the per-turn `done` events -- `per_token_ms` there is already
                DECODE-only, which is the number the track is about
  profile.jsonl one record per token (prefill tokens of a short prompt go down
                the decode path too, so the records are sliced per turn into
                `prefill_tokens` + `decode_steps` and only the decode half is
                used); carries `p0_count` / `p0_mean_ms` from this track
  status.json   the IoEngine end-of-run counters, including the P0 block

Prints one block a cell and, for more than one, an A/B table.
"""
from __future__ import annotations

import io
import json
import os
import sys


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def pctile(v, q):
    if not v:
        return 0.0
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


def cell(d):
    prof = load_jsonl(os.path.join(d, "profile.jsonl"))
    ev = [e for e in load_jsonl(os.path.join(d, "events.jsonl")) if e.get("event") == "done"]
    steps = sum(int(e.get("decode_steps", 0)) for e in ev)
    dms = sum(float(e.get("decode_ms", 0.0)) for e in ev)
    hit = (sum(float(e.get("decode_hit_rate", 0.0)) * int(e.get("decode_steps", 0)) for e in ev)
           / steps) if steps else 0.0

    # The engine's own decode-only per-token breakdown, weighted by steps.
    keys = ("attn", "moe_gpu", "moe_host", "nvme_stall", "engram", "tail", "other")
    br = {k: 0.0 for k in keys}
    for e in ev:
        n = int(e.get("decode_steps", 0))
        for k in keys:
            br[k] += float(e.get("per_token_ms", {}).get(k, 0.0)) * n
    if steps:
        br = {k: v / steps for k, v in br.items()}

    # The sink is written with stdio buffering, so a run can leave a partly
    # flushed head in front of the real sequence. The real sequence is the one
    # that starts at token 0 and runs to the end, so take the LAST such start.
    zero = [i for i, r in enumerate(prof) if r.get("token") == 0]
    if zero:
        prof = prof[zero[-1]:]
    # Within it, each turn is `prefill_tokens` records then `decode_steps`.
    dec, i = [], 0
    for e in ev:
        i += int(e.get("prefill_tokens", 0))
        n = int(e.get("decode_steps", 0))
        dec.extend(prof[i:i + n])
        i += n
    if not dec:
        dec = prof
    stall = [r.get("nvme_stall_ms", 0.0) for r in dec]
    p0n = [r.get("p0_count", 0) for r in dec]
    p0m = [r.get("p0_mean_ms", 0.0) for r in dec if r.get("p0_count")]
    miss = [r.get("expert_misses", 0) for r in dec]
    mb = [r.get("miss_bytes", 0) / 1048576.0 for r in dec]

    st = {}
    sp = os.path.join(d, "status.json")
    if os.path.exists(sp):
        st = json.load(io.open(sp, encoding="utf-8", errors="replace"))

    n = max(1, len(dec))
    return dict(
        cell=os.path.basename(d.rstrip("/\\")),
        turns=len(ev), steps=steps, sliced=len(dec),
        tok_s=(steps * 1e3 / dms) if dms else 0.0,
        per_turn=[round(float(e.get("tok_s", 0.0)), 2) for e in ev],
        hit=hit, breakdown=br,
        stall=sum(stall) / n, stall_p50=pctile(stall, 0.5), stall_p95=pctile(stall, 0.95),
        p0=sum(p0n) / n, p0_lat=(sum(p0m) / len(p0m)) if p0m else 0.0,
        misses=sum(miss) / n, miss_mb=sum(mb) / n,
        gbps=(sum(mb) / 1024.0) / (sum(stall) / 1e3) if sum(stall) else 0.0,
    ), st


def main(argv):
    rows = []
    for d in argv:
        r, st = cell(d)
        rows.append(r)
        print(f"=== {r['cell']} ===")
        print(f"  {r['turns']} turns, {r['steps']} decode steps ({r['sliced']} profiler records)")
        print(f"  tok/s {r['tok_s']:.3f}   per turn {r['per_turn']}   decode hit {r['hit']:.4f}")
        b = r["breakdown"]
        print("  per decode token (ms): " + "  ".join(f"{k} {b[k]:.1f}" for k in b))
        print(f"  nvme_stall mean {r['stall']:.2f}  p50 {r['stall_p50']:.2f}  p95 {r['stall_p95']:.2f} ms")
        print(f"  misses {r['misses']:.2f}/tok  {r['miss_mb']:.0f} MiB/tok  -> {r['gbps']:.2f} GB/s "
              f"through the stall window")
        print(f"  P0 {r['p0']:.2f} requests/tok, mean latency {r['p0_lat']:.2f} ms")
        for k in ("io",):
            if st.get(k):
                for line in str(st[k]).rstrip().splitlines():
                    print("    " + line)
        print()
    if len(rows) > 1:
        print(f"{'cell':<22}{'tok/s':>8}{'d%':>7}{'hit':>8}{'stall':>9}{'s%':>7}"
              f"{'P0/tok':>8}{'P0 lat':>8}{'GB/s':>7}")
        base = rows[0]
        for r in rows:
            dt = 100.0 * (r["tok_s"] / base["tok_s"] - 1.0) if base["tok_s"] else 0.0
            ds = 100.0 * (r["stall"] / base["stall"] - 1.0) if base["stall"] else 0.0
            print(f"{r['cell']:<22}{r['tok_s']:8.3f}{dt:+7.1f}{r['hit']:8.4f}"
                  f"{r['stall']:9.2f}{ds:+7.1f}{r['p0']:8.2f}{r['p0_lat']:8.2f}{r['gbps']:7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
