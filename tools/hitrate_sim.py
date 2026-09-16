#!/usr/bin/env python3
"""Turn a tools/hitrate_bench.py run into docs/p4_hitrate.md's curve, and replay
its routing through tools/cache_sim.py's LRU (Track R1).

    .venv/Scripts/python.exe tools/hitrate_sim.py curve bench/results/hitrate/auto \
        [--window 128] [--capacities 4500,5711] [--heat store/static_heat.inc] [--json out.json]
    .venv/Scripts/python.exe tools/hitrate_sim.py heat --trace ../deepmoe/traces/mixed --out store/static_heat.inc

`curve` reads route.bin (runtime/engine.h's routing dump: u32 step, u32 position,
u16[40x6] ids in gate order, u8[40] hits), profile.jsonl and turns.json, and prints
per-window measured hit rate / stall / tok/s over the decode steps, next to what
cache_sim's LRU gives on exactly the same accesses (prefill-by-decode steps
included, since they touch the cache too) at the engine's own capacity and any
other. The engine's per-step, per-layer hit counts are checked against the
simulator's; a planner that is cache_sim's LRU agrees on every layer.

`heat` writes the static (layer, expert) routing frequency of a route trace as a
C++ include for the P3 backfill (store/planner.h), hottest first.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import OrderedDict

import numpy as np

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS, TOPK, EXPERTS = 40, 6, 384
REC = 8 + LAYERS * TOPK * 2 + LAYERS


def load_route(path):
    raw = open(path, "rb").read()
    n = len(raw) // REC
    a = np.frombuffer(raw[: n * REC], dtype=np.uint8).reshape(n, REC)
    step = a[:, 0:4].copy().view(np.uint32).ravel()
    pos = a[:, 4:8].copy().view(np.uint32).ravel()
    ids = a[:, 8:8 + LAYERS * TOPK * 2].copy().view(np.uint16).reshape(n, LAYERS, TOPK)
    hits = a[:, 8 + LAYERS * TOPK * 2:].astype(np.int32)
    return step, pos, ids, hits


def step_kinds(turns):
    """Per request: prefill steps then decode steps, in the order serve ran them."""
    kinds = []
    for i, t in enumerate(turns):
        kinds += [("prefill", i)] * int(t.get("prefill_tokens", 0))
        kinds += [("decode", i)] * int(t.get("decode_steps", 0))
    return kinds


class Lru:
    """cache_sim.LRU's demand path, exactly as Planner::plan_layer runs it."""

    def __init__(self, cap, warm=None):
        self.cap = cap
        self.od = OrderedDict()
        for k in (warm or []):
            if len(self.od) >= cap:
                break
            self.od[k] = None
        # warm keys are given hottest first; the coldest must be the oldest
        self.od = OrderedDict(reversed(list(self.od.items())))

    def layer(self, keys):
        h = 0
        for k in keys:
            if k in self.od:
                self.od.move_to_end(k)
                h += 1
            else:
                self.od[k] = None
                if len(self.od) > self.cap:
                    self.od.popitem(last=False)
        return h


def simulate(ids, cap, warm=None):
    lru = Lru(cap, warm)
    n = ids.shape[0]
    out = np.zeros((n, LAYERS), dtype=np.int32)
    keys = ids.astype(np.int64) + (np.arange(LAYERS, dtype=np.int64) * EXPERTS)[None, :, None]
    for s in range(n):
        for L in range(LAYERS):
            out[s, L] = lru.layer(keys[s, L].tolist())
    return out


def load_heat(path):
    """store/static_heat.inc: `layer, expert, count,` triples, hottest first."""
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = [p for p in line.replace("{", "").replace("}", "").split(",") if p.strip()]
            if len(parts) >= 3:
                order.append(int(parts[0]) * EXPERTS + int(parts[1]))
    return order


def cmd_curve(a):
    d = a.dir
    turns = json.load(open(os.path.join(d, "turns.json"), encoding="utf-8"))
    ready = turns.get("server", {})
    tlist = turns["turns"]
    step, pos, ids, hits = load_route(os.path.join(d, "route.bin"))
    prof = [json.loads(l) for l in open(os.path.join(d, "profile.jsonl"), encoding="utf-8") if l.strip()]
    kinds = step_kinds(tlist)
    n = len(step)
    if len(kinds) != n or len(prof) != n:
        print(f"warning: {n} route records, {len(prof)} profile records, {len(kinds)} steps from turns.json",
              file=sys.stderr)
    n = min(n, len(prof), len(kinds))
    dec = [i for i in range(n) if kinds[i][0] == "decode"]
    slots = int(ready.get("cache_slots", 0))
    caps = [slots] + [int(c) for c in a.capacities.split(",") if c and int(c) != slots]
    sims = {c: simulate(ids[:n], c) for c in caps}
    heat = load_heat(a.heat) if a.heat else None
    if heat:
        sims_heat = {c: simulate(ids[:n], c, heat) for c in caps}

    agree = int((sims[slots][:n] == hits[:n]).all(axis=1).sum()) if slots else 0
    rows = []
    W = a.window
    for w0 in range(0, len(dec), W):
        idx = dec[w0:w0 + W]
        req = len(idx) * LAYERS * TOPK
        r = {
            "steps": f"{w0}-{w0 + len(idx) - 1}",
            "n": len(idx),
            "hit": hits[idx].sum() / req,
            "stall_ms": float(np.mean([prof[i]["nvme_stall_ms"] for i in idx])),
            "ms_token": float(np.mean([prof[i]["wall_ms"] for i in idx])),
            "miss_mb": float(np.mean([prof[i]["miss_bytes"] for i in idx])) / 1e6,
        }
        r["tok_s"] = 1000.0 / r["ms_token"]
        for c in caps:
            r[f"sim_{c}"] = sims[c][idx].sum() / req
            if heat:
                r[f"heat_{c}"] = sims_heat[c][idx].sum() / req
        rows.append(r)
    allreq = len(dec) * LAYERS * TOPK
    tot = {
        "decode_steps": len(dec), "prefill_steps": n - len(dec), "cache_slots": slots,
        "hit": float(hits[dec].sum() / allreq) if dec else 0.0,
        "stall_ms": float(np.mean([prof[i]["nvme_stall_ms"] for i in dec])) if dec else 0.0,
        "ms_token": float(np.mean([prof[i]["wall_ms"] for i in dec])) if dec else 0.0,
        "steps_agreeing_with_sim": agree, "steps": n,
    }
    tot["tok_s"] = 1000.0 / tot["ms_token"] if tot["ms_token"] else 0.0
    for c in caps:
        tot[f"sim_{c}"] = float(sims[c][dec].sum() / allreq) if dec else 0.0
        if heat:
            tot[f"heat_{c}"] = float(sims_heat[c][dec].sum() / allreq) if dec else 0.0
    hdr = ["steps", "hit", "stall_ms", "ms_token", "tok_s", "miss_mb"] + [f"sim_{c}" for c in caps] + \
          ([f"heat_{c}" for c in caps] if heat else [])
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        cells = []
        for h in hdr:
            v = r[h]
            cells.append(v if isinstance(v, str) else (f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"))
        print("| " + " | ".join(cells) + " |")
    print(json.dumps(tot))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"windows": rows, "total": tot}, f, indent=1, default=float)
    return 0


def cmd_heat(a):
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import cache_sim  # noqa: E402
    t = cache_sim.Trace.load([a.trace], "pre0", quiet=True)
    keys = t.keys6.ravel()
    counts = np.bincount(keys, minlength=LAYERS * EXPERTS)
    order = np.argsort(-counts, kind="stable")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Generated by tools/hitrate_sim.py heat from the route trace "
                f"'{os.path.basename(os.path.normpath(a.trace))}' ({t.n_tokens} tokens).\n")
        f.write("// (layer, expert, routed count), hottest first: the P3 backfill order of\n")
        f.write("// store/planner.h. Do not edit by hand.\n")
        for k in order:
            f.write(f"{{{k // EXPERTS}, {k % EXPERTS}, {counts[k]}}},\n")
    print(f"wrote {len(order)} entries to {a.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("dir")
    c.add_argument("--window", type=int, default=128)
    c.add_argument("--capacities", default="")
    c.add_argument("--heat", default="")
    c.add_argument("--json", default="")
    h = sub.add_parser("heat")
    h.add_argument("--trace", required=True)
    h.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_curve(a) if a.cmd == "curve" else cmd_heat(a)


if __name__ == "__main__":
    sys.exit(main())
