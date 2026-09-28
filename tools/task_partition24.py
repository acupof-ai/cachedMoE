#!/usr/bin/env python3
"""Within-kind vs across-kind expert overlap for bench/results/hitrate/tasks24.json.

    .venv/bin/python tools/task_partition24.py bench/results/linux/tasks24 [--out DIR]

tasks24 runs 12 task kinds x 2 different prompts, every turn a fresh topic with a
GPU prefill (whose routing is not in route.bin), so route.bin is the 128-step
startup replay followed by exactly the decode steps of the 24 turns in order.
For every pair of turns this compares their per-layer expert frequency vectors
(cosine) and the measured decode hit rate per kind. If the model partitions its
experts by task, two prompts of the SAME kind overlap more than two of different
kinds; the gap per layer says where that partition lives.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hitrate_sim  # noqa: E402

L, K, E = 40, 6, 384
STARTUP = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="bench/results/linux/expert_hits")
    a = ap.parse_args()
    _, _, ids, hits = hitrate_sim.load_route(os.path.join(a.run, "route.bin"))
    turns = [x for x in json.load(open(os.path.join(a.run, "turns.json")))["turns"]
             if x.get("event") == "done"]
    script = json.load(open("bench/results/hitrate/tasks24.json"))["turns"]
    labels = [s["label"] for s in script]
    steps = [int(x["decode_steps"]) for x in turns]
    if len(ids) != STARTUP + sum(steps):
        sys.exit(f"route.bin has {len(ids)} records, expected {STARTUP} + {sum(steps)}")

    # freq[turn, layer, expert], hits per turn
    n = len(turns)
    freq = np.zeros((n, L, E), np.float64)
    hit = np.zeros(n); req = np.zeros(n)
    at = STARTUP
    for i, s in enumerate(steps):
        blk = ids[at:at + s]
        for l in range(L):
            np.add.at(freq[i, l], blk[:, l, :].ravel(), 1.0)
        hit[i] = hits[at:at + s].sum(); req[i] = s * L * K
        at += s
    unit = freq / np.maximum(np.linalg.norm(freq, axis=2, keepdims=True), 1e-12)

    kinds = sorted(set(labels), key=labels.index)
    same, diff = [], []
    for i, j in itertools.combinations(range(n), 2):
        cos_l = (unit[i] * unit[j]).sum(axis=1)          # per layer
        (same if labels[i] == labels[j] else diff).append(cos_l)
    same = np.array(same); diff = np.array(diff)

    # Noise floor: the same turn split into halves.
    halves = []
    at = STARTUP
    for i, s in enumerate(steps):
        h1 = np.zeros((L, E)); h2 = np.zeros((L, E))
        blk = ids[at:at + s]
        for l in range(L):
            np.add.at(h1[l], blk[: s // 2, l, :].ravel(), 1.0)
            np.add.at(h2[l], blk[s // 2:, l, :].ravel(), 1.0)
        u1 = h1 / np.maximum(np.linalg.norm(h1, axis=1, keepdims=True), 1e-12)
        u2 = h2 / np.maximum(np.linalg.norm(h2, axis=1, keepdims=True), 1e-12)
        halves.append((u1 * u2).sum(axis=1)); at += s
    halves = np.array(halves)

    # Kind x kind mean cosine (over layers and prompt pairs).
    km = np.zeros((len(kinds), len(kinds)))
    for a_, ka in enumerate(kinds):
        for b_, kb in enumerate(kinds):
            vals = [(unit[i] * unit[j]).sum(axis=1).mean()
                    for i in range(n) for j in range(n)
                    if labels[i] == ka and labels[j] == kb and i != j]
            km[a_, b_] = np.mean(vals)

    per_kind = {}
    for k in kinds:
        idx = [i for i in range(n) if labels[i] == k]
        per_kind[k] = dict(decode_steps=int(sum(steps[i] for i in idx)),
                           hit=float(hit[idx].sum() / req[idx].sum()),
                           misses_per_token=float((req[idx].sum() - hit[idx].sum()) / sum(steps[i] for i in idx)))

    gap = same.mean(axis=0) - diff.mean(axis=0)
    out = dict(
        run=a.run, kinds=kinds, turns=n, decode_steps=int(sum(steps)),
        within_kind_cos=float(same.mean()), across_kind_cos=float(diff.mean()),
        half_split_cos=float(halves.mean()),
        per_layer=[dict(layer=l, within=float(same[:, l].mean()), across=float(diff[:, l].mean()),
                        halves=float(halves[:, l].mean()), gap=float(gap[l])) for l in range(L)],
        kind_matrix=km.round(4).tolist(), per_kind=per_kind,
        turn_hits=[dict(turn=i, label=labels[i], steps=steps[i], hit=float(hit[i] / req[i])) for i in range(n)],
    )
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "tasks24_partition.json"), "w"), ensure_ascii=False, indent=1)

    print(f"{n} turns, {sum(steps)} decode steps, {len(kinds)} kinds")
    print(f"mean per-layer cosine: same kind {same.mean():.3f}  different kind {diff.mean():.3f}  "
          f"same turn split in halves {halves.mean():.3f}")
    order = np.argsort(-gap)
    print("most task-partitioned layers (within - across):",
          ", ".join(f"L{l} {gap[l]:+.3f}" for l in order[:6]))
    print("least:", ", ".join(f"L{l} {gap[l]:+.3f}" for l in order[-5:]))
    print("per kind (measured decode hit, misses/token):")
    for k in kinds:
        v = per_kind[k]
        print(f"  {k:13s} steps {v['decode_steps']:4d}  hit {v['hit']:.4f}  misses/token {v['misses_per_token']:.1f}")


if __name__ == "__main__":
    main()
