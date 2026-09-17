#!/usr/bin/env python3
"""Track F5 step 3: the system model for 2-bit routed experts.

Two things have to be answered together, because they pull in the same direction:

  1. a miss costs half as many bytes, so the NVMe term halves at a fixed hit rate;
  2. a slot is half the size, so the same 100 GiB slab pool holds twice as many
     experts, and the hit rate itself goes up.

(2) needs the LRU hit-rate curve past 7,680 slots, which is where
`reports/cache_sweep.json` stops. This tool recomputes the exact curve from the
same 27,399-token routing trace with the same Fenwick-tree stack-distance method
(`tools/cache_sim.py:stack_distances`) and evaluates it at *every* capacity, so
the 2-bit capacities are read off the same curve as the FP4 ones rather than
extrapolated by eye.

The throughput model is design section 3.1's, unchanged:

    t_token = 42 ms                       # 8.5 GB resident @ 216 GB/s, rounded up
            + h       x B / 200 GB/s      # hit: read from LPDDR
            + (1 - h) x B / 4.5 GB/s      # miss: read from NVMe (section 9.2.1)

with B the routed-expert bytes per token: 6 experts x 40 layers x the expert size.

    python tools/quant2_model.py --trace <dir> --out bench/results/quant2
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# design section 2.2 / 5.2 / 3.1
EXPERT_ELEMS = 3 * 2304 * 5120          # w1 + w2 + w3, one expert
FP4_BITS = 4.25                          # 4 bits + one UE8M0 byte per 32
FP4_EXPERT_BYTES = 18_800_640
SLAB_BYTES = 100 * 2 ** 30               # the measured 100 GiB slab pool
N_ROUTED_TOTAL = 40 * 384
RESIDENT_MS = 42.0
LPDDR_GBPS = 200.0
NVME_GBPS = 4.5
ROUTED_PER_TOKEN = 6 * 40                # 6 experts per layer, 40 layers
MOE_KERNEL_MS_FP4 = 25.0                 # section 7.9.3, HQuant=3, 40 layers


def expert_bytes(bits_per_weight: float) -> int:
    """Bytes on disk and in a slot for one expert at `bits_per_weight`.

    Rounded up to the 4 KiB sector, because design section 5.1 reads experts as
    whole sectors and section 2.2 notes the FP4 expert is naturally 4590 pages.
    """
    raw = int(round(EXPERT_ELEMS * bits_per_weight / 8))
    return -(-raw // 4096) * 4096


def slots_for(bits_per_weight: float, slab_bytes: int = SLAB_BYTES) -> int:
    return slab_bytes // expert_bytes(bits_per_weight)


# --------------------------------------------------------------------------- #
# the exact LRU curve, at every capacity
# --------------------------------------------------------------------------- #

def stack_distance_hist(keys: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Exact reuse (stack) distances, as a histogram over distance.

    Same algorithm and therefore the same numbers as
    `tools/cache_sim.py:stack_distances` -- a Fenwick tree over access time, each
    key's most recent access marked 1, the distance being the number of marks
    strictly after the previous access. Only the return shape differs: a
    histogram, so the hit rate at capacity C is `hist[:C].sum() / n` for every C
    at once and the 6.6 M distances never have to be materialised.
    """
    n = len(keys)
    tree = np.zeros(n + 1, dtype=np.int64)
    hist = np.zeros(n + 2, dtype=np.int64)
    cold = 0

    def add(i: int, v: int):
        i += 1
        while i <= n:
            tree[i] += v
            i += i & (-i)

    def pref(i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    last: dict[int, int] = {}
    kl = keys.tolist()
    for t in range(n):
        k = kl[t]
        prev = last.get(k)
        if prev is None:
            cold += 1
        else:
            hist[pref(t - 1) - pref(prev)] += 1
            add(prev, -1)
        add(t, 1)
        last[k] = t
    return hist, cold, n


def curve_from_hist(hist: np.ndarray, n: int) -> np.ndarray:
    """hit(C) for C = 0 .. len(hist): the CDF of the distances over all accesses."""
    return np.cumsum(hist).astype(np.float64) / n


def load_keys(trace_dir: str) -> np.ndarray:
    import cache_sim
    t = cache_sim.Trace.load([trace_dir], "mean-hc", quiet=False)
    return t.keys6.reshape(-1).astype(np.int64)


# --------------------------------------------------------------------------- #
# throughput
# --------------------------------------------------------------------------- #

def tps(h: float, bytes_per_expert: int) -> dict:
    b = ROUTED_PER_TOKEN * bytes_per_expert
    hit_ms = h * b / (LPDDR_GBPS * 1e9) * 1e3
    miss_ms = (1.0 - h) * b / (NVME_GBPS * 1e9) * 1e3
    t = RESIDENT_MS + hit_ms + miss_ms
    return {"hit_rate": round(h, 4),
            "routed_gb_per_token": round(b / 1e9, 3),
            "resident_ms": RESIDENT_MS,
            "hit_read_ms": round(hit_ms, 1),
            "nvme_stall_ms": round(miss_ms, 1),
            "t_token_ms": round(t, 1),
            "tok_s": round(1000.0 / t, 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", default=r"C:\Users\Asus\code\deepmoe\traces\mixed")
    p.add_argument("--sweep", default=r"C:\Users\Asus\code\deepmoe\reports\cache_sweep.json")
    p.add_argument("--bits", default="4.25,2.5,2.35,2.25,2.125",
                   help="effective bits/weight to model (4.25 = today's FP4)")
    p.add_argument("--slab-gib", type=float, default=100.0)
    p.add_argument("--out", default="bench/results/quant2")
    p.add_argument("--curve-cache", default=None,
                   help="npz to reuse/write the stack-distance histogram")
    a = p.parse_args(argv)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    cache = a.curve_cache or os.path.join(a.out, "lru_curve.npz")
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(cache):
        z = np.load(cache)
        hist, n_acc, cold = z["hist"], int(z["n"]), int(z["cold"])
        print(f"reusing {cache}: {n_acc:,} accesses, cold {cold:,}")
    else:
        keys = load_keys(a.trace)
        print(f"{len(keys):,} accesses; computing exact stack distances "
              f"(Fenwick, single core)...", flush=True)
        hist, cold, n_acc = stack_distance_hist(keys)
        hist = hist[:N_ROUTED_TOTAL + 2]
        np.savez_compressed(cache, hist=hist, n=n_acc, cold=cold)
        print(f"  {time.perf_counter() - t0:.0f}s -> {cache}")
    curve = curve_from_hist(hist, n_acc)

    def hit_at(c: int) -> float:
        return float(curve[min(c, len(curve) - 1)])

    # cross-check against reports/cache_sweep.json's published capacities
    checks = []
    if os.path.exists(a.sweep):
        with io.open(a.sweep, encoding="utf-8") as f:
            published = json.load(f)["q2_reuse_distance"]["global_lru_hit_rate_by_capacity"]
        for c, v in sorted(published.items(), key=lambda kv: int(kv[0])):
            checks.append({"capacity": int(c), "published": v,
                           "recomputed": round(hit_at(int(c)), 4),
                           "delta": round(hit_at(int(c)) - v, 5)})

    slab = int(a.slab_gib * 2 ** 30)
    rows = []
    for b in [float(x) for x in a.bits.split(",")]:
        eb = expert_bytes(b)
        s = slab // eb
        h = hit_at(s)
        r = tps(h, eb)
        r.update({"bits_per_weight": b, "expert_bytes": eb, "slots": s,
                  "slot_frac_of_model": round(s / N_ROUTED_TOTAL, 4),
                  "all_experts_gb": round(eb * N_ROUTED_TOTAL / 1e9, 1),
                  "moe_kernel_ms": round(MOE_KERNEL_MS_FP4 * eb / FP4_EXPERT_BYTES, 1)})
        rows.append(r)

    # what the FP4 hit rate alone would buy if only the bytes halved (slots fixed)
    fp4_slots = slab // expert_bytes(FP4_BITS)
    h_fp4 = hit_at(fp4_slots)
    isolated = tps(h_fp4, expert_bytes(2.25))
    isolated["note"] = ("2-bit bytes at the FP4 hit rate: separates the "
                        "'halve the miss' effect from the 'double the slots' effect")

    blob = {"generator": "tools/quant2_model.py", "started": started,
            "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trace": a.trace, "accesses": n_acc,
            "cold_frac": round(cold / n_acc, 5),
            "slab_gib": a.slab_gib,
            "model": {"resident_ms": RESIDENT_MS, "lpddr_gbps": LPDDR_GBPS,
                      "nvme_gbps": NVME_GBPS, "routed_reads_per_token": ROUTED_PER_TOKEN,
                      "moe_kernel_ms_fp4": MOE_KERNEL_MS_FP4},
            "cross_check_vs_cache_sweep": checks,
            "curve": {str(c): round(hit_at(c), 4) for c in
                      (768, 1536, 2304, 3072, 3840, 4608, 4787, 5376, 5711, 6144,
                       7680, 8192, 9216, 9708, 10240, 10787, 11264, 12288, 15360)},
            "rows": rows, "isolated_bytes_only": isolated}
    path = os.path.join(a.out, "system_model.json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1)

    print(f"\ncold_frac {cold / n_acc:.5f}  (ceiling on h: {1 - cold / n_acc:.4f})")
    if checks:
        print("cross-check vs reports/cache_sweep.json: max |delta| "
              f"{max(abs(c['delta']) for c in checks):.5f}")
    print(f"\n{'bits':>6s} {'MB/expert':>10s} {'slots':>7s} {'h':>7s} "
          f"{'GB/tok':>7s} {'stall':>7s} {'ms/tok':>7s} {'tok/s':>7s} {'total GB':>9s}")
    for r in rows:
        print(f"{r['bits_per_weight']:6.3f} {r['expert_bytes'] / 1e6:10.2f} "
              f"{r['slots']:7d} {r['hit_rate']:7.4f} {r['routed_gb_per_token']:7.3f} "
              f"{r['nvme_stall_ms']:7.1f} {r['t_token_ms']:7.1f} {r['tok_s']:7.2f} "
              f"{r['all_experts_gb']:9.1f}")
    print(f"\nbytes-only control (2.25 bits at the FP4 hit rate {h_fp4:.4f}): "
          f"{isolated['t_token_ms']} ms, {isolated['tok_s']} tok/s")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
