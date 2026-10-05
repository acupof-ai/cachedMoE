#!/usr/bin/env python3
"""Optimistic weighted-miss scan using cache_sim.LRU and recorded gate weights.

All demand misses are admitted instantly, as in the existing cache simulator.
This omits asynchronous fill lag and output feedback. Wait counts identify
candidates; elapsed-time estimates are bounds, not predicted GPU throughput.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cache_sim import LRU
from mask_freeze_sim import weighted_trace


def scan(ids, weights, capacity, taus):
    cache = LRU(capacity)
    waits = np.zeros((len(ids), len(taus)), dtype=np.int32)
    mass_sum = np.zeros(len(taus))
    worst = np.zeros(len(taus))
    hits = requests = 0
    for token, (row, weight) in enumerate(zip(ids, weights)):
        for layer, (experts, w) in enumerate(zip(row, weight)):
            miss = []
            for e, z in zip(experts, w):
                key = layer * 384 + int(e)
                requests += 1
                if cache.contains(key):
                    cache.touch(key)
                    hits += 1
                else:
                    miss.append(float(z))
                    cache.admit(key)
            total = sum(float(z) for z in w)
            lost = sum(miss) / total
            ordered = sorted(miss, reverse=True)
            for index, tau in enumerate(taus):
                remaining, count = lost, 0
                if tau < 1:
                    for z in ordered:
                        if remaining <= tau and not (len(miss) == 6 and count == 0):
                            break
                        remaining -= z / total
                        count += 1
                waits[token, index] += count
                remaining = max(0.0, min(1.0, remaining))
                mass_sum[index] += remaining
                worst[index] = max(worst[index], remaining)
    return dict(tokens=len(ids), capacity=capacity, hit_rate=hits / requests,
        configurations=[dict(tau=tau, wait_experts_mean=float(waits[:, i].mean()),
            wait_experts_p95=float(np.percentile(waits[:, i], 95)),
            mean_mass_lost=float(mass_sum[i] / (len(ids) * 40)),
            worst_layer_mass_lost=float(worst[i])) for i, tau in enumerate(taus)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--slots', type=int, default=5500)
    p.add_argument('--taus', default='1,.30,.20,.15,.10,.05')
    p.add_argument('--json', required=True, type=Path)
    args = p.parse_args()
    prompts, ids, weights, hashes = weighted_trace(args.trace)
    report = scan(ids, weights, args.slots, [float(t) for t in args.taus.split(',')])
    report.update(source=str(args.trace), source_sha256=hashes,
                  model='instant LRU admission; no fill lag, waiting budget or output feedback')
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
