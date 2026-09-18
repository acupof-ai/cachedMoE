# u(M) and the MISS account for a speculative verify batch, on real routings.
#
# docs/p4_dspark_runtime.md §6.5 rests on two numbers and this measures both.
#
#   u(M)     the expert union of M CONSECUTIVE positions. §6.5's first table.
#            docs/p3_dspark.md §12.6 had one point (~26 at M=6); this is all 40
#            layers over traces/mixed.
#
#   misses   what the union is actually worth. u(M)/6M counts a union member
#            once, but the unspeculated baseline runs with a WARM CACHE, and
#            what the cache already does is turn an expert's second use within a
#            few tokens into a hit -- precisely the reuse the union collapses.
#            So the two savings are the same money and the unique count is only
#            an upper bound. This walks a global LRU in decode order
#            (position-major, layer-minor) twice over the same trace at the same
#            capacity: once one position at a time, once M consecutive positions
#            as one batch that fetches its union, and reports miss bytes per
#            EMITTED token for each.
#
# Read-only over the trace directory. pyarrow only (no pandas in this env).
#
#   python tools/route_union.py [trace_dir] [--capacity 4500] [--max-pos 20000]
import argparse
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

EXPERTS_PER_LAYER = 384
TOPK = 6
SLOT_MB = 18.8            # one routed expert, docs/p4_hitrate.md
MS = [2, 3, 4, 5, 6]
# docs/p3_dspark.md §12.2, runtime replay, greedy, tree K=16 eal: E[tokens](k)
E_TOKENS = {1: 1.83, 2: 2.50, 3: 2.85, 4: 3.33, 5: 3.33}


def load(root, max_pos):
    """[layer][row] -> top6, plus the (prompt, pos) key shared by every layer."""
    files = sorted(glob.glob(os.path.join(root, "route_layer*.parquet")))
    if not files:
        sys.exit("no route_layer*.parquet under " + root)
    key = None
    layers = {}
    for f in files:
        layer = int(os.path.basename(f)[11:13])
        t = pq.read_table(f, columns=["prompt_id", "pos", "top6_ids"])
        pid = t.column("prompt_id").to_numpy()
        pos = t.column("pos").to_numpy()
        ids = np.asarray(t.column("top6_ids").combine_chunks().flatten()
                         .to_numpy()).reshape(-1, TOPK).astype(np.int32)
        order = np.lexsort((pos, pid))
        pid, pos, ids = pid[order], pos[order], ids[order]
        if max_pos and len(pos) > max_pos:
            pid, pos, ids = pid[:max_pos], pos[:max_pos], ids[:max_pos]
        if key is None:
            key = (pid, pos)
        elif len(pid) != len(key[0]) or not np.array_equal(pos, key[1]):
            # Layers may cover different row counts; intersect on the shorter.
            n = min(len(pid), len(key[0]))
            key = (key[0][:n], key[1][:n])
        layers[layer] = ids
    n = len(key[0])
    for L in layers:
        layers[L] = layers[L][:n]
    return key, layers


def u_of_m(key, layers):
    pid, pos = key
    out = {}
    for m in MS:
        n = len(pos)
        ok = np.ones(n - m + 1, dtype=bool)
        for j in range(1, m):
            ok &= pid[j:n - m + 1 + j] == pid[0:n - m + 1]
            ok &= pos[j:n - m + 1 + j] == pos[0:n - m + 1] + j
        idx = np.flatnonzero(ok)
        s = c = 0
        for L, ids in layers.items():
            take = idx if idx.size <= 20000 else idx[np.linspace(0, idx.size - 1, 20000).astype(np.int64)]
            for i in take:
                s += len(np.unique(ids[i:i + m]))
                c += 1
        out[m] = s / max(c, 1)
    return out


class Lru:
    """Global pool over (layer, expert). Access order is the fetch order."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.clock = 0
        self.when = {}            # key -> last use

    def touch(self, keys):
        """Returns how many of `keys` were misses. `keys` must be unique."""
        miss = 0
        for k in keys:
            if k not in self.when:
                miss += 1
            self.when[k] = self.clock
            self.clock += 1
        if len(self.when) > self.capacity:
            # Evict down to capacity, oldest first.
            order = sorted(self.when.items(), key=lambda kv: kv[1])
            for k, _ in order[:len(self.when) - self.capacity]:
                del self.when[k]
        return miss


def misses(key, layers, capacity, m, warmup_frac=0.2):
    """(sequential misses, batched misses, positions counted) over one pass each.

    `m == 1` is the unspeculated baseline. Both passes see the same trace, the
    same capacity and the same decode order (position-major, layer-minor).
    """
    pid, pos = key
    n = len(pos)
    L_ids = [layers[L] for L in sorted(layers)]
    n_layers = len(L_ids)
    warm = int(n * warmup_frac)

    cache = Lru(capacity)
    total = 0
    counted = 0
    i = 0
    while i < n:
        # How many consecutive positions of the same prompt this step covers.
        span = 1
        while span < m and i + span < n and pid[i + span] == pid[i] and pos[i + span] == pos[i] + span:
            span += 1
        for L in range(n_layers):
            ids = L_ids[L]
            if span == 1:
                keys = {(L, int(e)) for e in ids[i]}
            else:
                keys = {(L, int(e)) for e in ids[i:i + span].ravel()}
            mi = cache.touch(keys)
            if i >= warm:
                total += mi
        if i >= warm:
            counted += span
        i += span
    return total, counted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=r"C:\Users\Asus\code\deepmoe\traces\mixed")
    ap.add_argument("--capacity", type=int, default=4500,
                    help="expert cache slots; 4500 is the cache-4500 cell of "
                         "docs/p4_hitrate.md, where decode hit is 0.837")
    ap.add_argument("--max-pos", type=int, default=20000)
    ap.add_argument("--skip-misses", action="store_true")
    a = ap.parse_args()

    key, layers = load(a.trace, a.max_pos)
    print(f"{len(key[1])} positions x {len(layers)} layers from {a.trace}\n")

    u = u_of_m(key, layers)
    print("u(M): the expert union of M consecutive positions, all layers")
    print(f"{'M':>2} {'u(M)':>7} {'u/6M':>7}")
    for m in MS:
        print(f"{m:>2} {u[m]:>7.2f} {u[m] / (TOPK * m):>7.3f}")
    if a.skip_misses:
        return

    # The measurement: does fetching the UNION of M positions miss less often
    # than fetching them one position at a time, through the same LRU at the
    # same capacity? `miss/pos` is the answer and it is the only number here
    # that is a measurement rather than an inference.
    print(f"\nmisses through one global LRU of {a.capacity} slots, decode order "
          f"(position-major, layer-minor)")
    base_miss, base_n = misses(key, layers, a.capacity, 1)
    base_per_tok = base_miss / base_n
    print(f"{'M':>2} {'miss/pos':>9} {'vs M=1':>8} {'hit':>6}   "
          f"how a batch fetching its union compares")
    accesses = TOPK * len(layers)
    print(f"{1:>2} {base_per_tok:>9.3f} {1.0:>8.3f} {1 - base_per_tok / accesses:>6.3f}"
          f"   (the unspeculated baseline)")
    per_m = {1: base_per_tok}
    for m in MS:
        mm, nn = misses(key, layers, a.capacity, m)
        per_m[m] = mm / nn
        print(f"{m:>2} {per_m[m]:>9.3f} {per_m[m] / base_per_tok:>8.3f} "
              f"{1 - per_m[m] / accesses:>6.3f}")
    print(f"\n   A ratio of 1.000 means the union collapses NO miss the LRU was not")
    print(f"   already collapsing: u(M)/6M < 1 counts an expert's second use inside")
    print(f"   the batch once, and a warm cache was already scoring that use as a hit.")

    # What that implies per EMITTED token, between its two honest bounds.
    print(f"\nmiss per emitted token, relative to no speculation")
    print(f"{'k':>2} {'M':>2} {'E[tok]':>7} {'best':>7} {'worst':>7}")
    for k in (1, 2, 3, 4, 5):
        m = k + 1
        e = E_TOKENS[k]
        # best:  a rejected position's drafted token routes exactly where the
        #        true token will, so next cycle re-reads those experts as hits
        #        and speculation fetches the same set as sequential decode.
        # worst: a rejected position's drafted token routes elsewhere, so its
        #        whole fetch is wasted and the cycle pays for M positions to
        #        emit E of them.
        print(f"{k:>2} {m:>2} {e:>7.2f} {1.0:>7.3f} {per_m[m] * m / e / base_per_tok:>7.3f}")
    print(f"\n   Neither bound is an improvement. Miss counts are experts; "
          f"x {SLOT_MB} MB a slot for bytes.")


if __name__ == "__main__":
    main()
