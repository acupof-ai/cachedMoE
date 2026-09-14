#!/usr/bin/env python3
"""Offline expert-cache simulator: pick the policy before writing it in C++.

Design reference: section 9.2 (measurement tools), section 9.3 (residency) and
section 9.4 (lookahead prefetch). P1 deliverable (section 15). Section 16: "no
Planner policy code before the P-1 and P1 measurements".

Section 3.1 is the reason this exists. At 200 GB/s of LPDDR and the 4.5 GB/s section
9.2.1 actually measured off the NVMe, decode time per token is dominated by expert
misses:

    h = 0.30 -> ~670 ms/token (1.5 tok/s)
    h = 0.60 -> ~410 ms       (2.4 tok/s)
    h = 0.85 -> ~240 ms       (4.2 tok/s)
    h = 1.00 ->   65 ms       (15.3 tok/s, unreachable: 288 GB does not fit)

Ten points of hit rate are worth more than any kernel optimisation, so the policy is
chosen by simulation over a real trace, not by intuition.

What it sweeps:

  capacity   fraction of the 15,360 routed experts that fit (design section 5.2 puts
             the real cache at ~29%, i.e. 90 GB / 18.8 MB = 4787 slots):
             {20, 25, 30, 35}% plus the absolute slot count
  policy     lru | lfu-decay | arc | score-aware | static-pin+lru (section 9.3)
  allocation one global pool vs per-layer quotas (decided by Q2)
  prefetch   lookahead depth d and width K, using the predictions the trace recorded
             (section 9.4)

What it reports, per configuration: hit rate (total and per layer), miss bytes per
token, prefetch precision and recall, wasted prefetch bytes, hidden vs exposed
misses, and the implied tok/s from the section 3.1 model.

It also answers Q1 (static frequency coverage), Q2 (exact reuse-distance CDF, which
gives LRU's hit rate at every capacity without simulating each one) and Q3 (per-layer
Jaccard overlap between consecutive tokens, and the union growth over a speculative
window of k = 2..5, which feeds the schedule curve of section 10.3).

The winning configuration goes into design section 9 before P3 starts, and
store/planner.cpp implements exactly it.

Usage:
    uv run python tools/cache_sim.py --trace traces/mixed \\
        --capacities 20,25,30,35 --policies lru,score-aware \\
        --out reports/cache_sweep.json
"""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

# Design section 2.3 / 3.1 / appendix A. Kept in step with model/layout.h.
EXPERT_BYTES = 18_800_640
N_LAYERS = 40
N_ROUTED_PER_LAYER = 384
N_ROUTED_EXPERTS_TOTAL = N_LAYERS * N_ROUTED_PER_LAYER      # 15,360
HOT_BYTES_PER_TOKEN = 8.5e9          # section 2.3: everything pinned, read every token
RESIDENT_MS = 42.0                   # section 3.1: HOT_BYTES / 200 GB/s
CACHE_SLOTS_90GB = 4787              # 90 GB of slab pool / 18.8 MB


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cache_sim.py",
        description="simulate expert-cache policies over a recorded route trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--trace", required=True, action="append",
                   help="a directory of route_layer*.parquet, a glob, or a single "
                        "file; repeatable")
    p.add_argument("--capacities", default="20,25,30,35",
                   help="cache capacities as a percentage of all routed experts; "
                        "'abs:N' gives an absolute slot count")
    p.add_argument("--policies", default="lru,lfu-decay,arc,score-aware,static-pin+lru",
                   help="comma-separated policies to compare (design section 9.3)")
    p.add_argument("--allocation", default="global", choices=["global", "per-layer", "both"],
                   help="one pool or per-layer quotas; Q2 decides whether the latter "
                        "is worth it")
    p.add_argument("--prefetch-depths", default="0",
                   help="lookahead depth d; 0 disables prefetch (design section 9.4). "
                        "The full grid is capacities x policies x allocations x "
                        "depths x widths and each configuration is a full replay, so "
                        "sweep in two stages: policies at depth 0 first, then (d, K) "
                        "for the winner")
    p.add_argument("--prefetch-widths", default="16",
                   help="lookahead width K (design section 9.4)")
    p.add_argument("--lookahead-input", default="pre0", choices=["mean", "pre0"],
                   help="which of the two section 9.4 approximations to prefetch from")
    p.add_argument("--nvme-gbps", type=float, default=4.5,
                   help="measured effective NVMe bandwidth (section 9.2.1: 4.5-4.75)")
    p.add_argument("--lpddr-gbps", type=float, default=200.0)
    p.add_argument("--t-layer-ms", type=float, default=1.3,
                   help="GPU time for one layer's resident compute (section 3.4)")
    p.add_argument("--t-io-ms", type=float, default=4.0,
                   help="latency of one 18.8 MB expert read (section 9.2.1 measured "
                        "4.0-4.7 ms at QD 4-8)")
    p.add_argument("--io-qd", type=int, default=8,
                   help="target queue depth; section 9.2.1 fixed chunk = 4 MiB, QD = 8")
    p.add_argument("--lfu-decay", type=float, default=0.98)
    p.add_argument("--pin-frac", type=float, default=0.10,
                   help="static-pin+lru: fraction of each layer's experts to pin")
    p.add_argument("--pin-train-frac", type=float, default=0.25,
                   help="fraction of the trace used to pick the pinned set; the hit "
                        "rate is reported on the rest, so the pin is not an oracle")
    p.add_argument("--warmup-frac", type=float, default=0.05,
                   help="fraction of the trace excluded from the reported hit rate, "
                        "so a cold cache does not dominate a short trace")
    p.add_argument("--questions", default="q1,q2,q3",
                   help="which of the section 9.1 analyses to also emit")
    p.add_argument("--q2-max-accesses", type=int, default=6_000_000,
                   help="cap on the exact stack-distance pass (it is O(n log n) in "
                        "Python); beyond this the trace is truncated, not sampled")
    p.add_argument("--out", default=None, help="write the report as JSON here")
    p.add_argument("--quiet", action="store_true")
    return p


# --------------------------------------------------------------------------- #
# trace loading
# --------------------------------------------------------------------------- #

class Trace:
    """A route trace flattened into decode order.

    The simulator needs the request stream a decoder would make: for each token in
    sequence, for each layer 0..39, the six routed experts. `tools/route_trace.py`
    writes one file per layer, so the columns are re-sorted here into
    (prompt, position, layer) order once, and everything downstream is numpy.

    Cache keys are global: `layer * 384 + expert`, matching how design section 5.2
    treats the slab pool as one address space.
    """

    def __init__(self, top6: np.ndarray, top16: np.ndarray, top16_scores: np.ndarray,
                 layer: np.ndarray, prompt: np.ndarray, pos: np.ndarray,
                 preds: dict[int, np.ndarray], bias_applied: np.ndarray,
                 n_layers: int, n_experts: int):
        self.top6 = top6                  # [rows, 6] uint16, expert id within layer
        self.top16 = top16
        self.top16_scores = top16_scores
        self.layer = layer                # [rows] uint8
        self.prompt = prompt
        self.pos = pos
        self.preds = preds                # depth -> [rows, 16] uint16 (-1 where absent)
        self.bias_applied = bias_applied
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_rows = len(layer)
        self.n_tokens = self.n_rows // n_layers if n_layers else 0

    @property
    def keys6(self) -> np.ndarray:
        """[rows, 6] global cache keys."""
        return self.layer.astype(np.int32)[:, None] * self.n_experts + self.top6

    @staticmethod
    def load(paths: list[str], lookahead_input: str, quiet: bool = False) -> "Trace":
        import pyarrow.parquet as pq

        files: list[str] = []
        for spec in paths:
            if os.path.isdir(spec):
                files += sorted(glob.glob(os.path.join(spec, "route_layer*.parquet")))
            elif any(c in spec for c in "*?["):
                files += sorted(glob.glob(spec))
            else:
                files.append(spec)
        if not files:
            raise SystemExit(f"no trace files matched {paths}")

        import pyarrow as pa

        # Read only what is used: the trace carries both section 9.4 approximations
        # and the unused one is half the file.
        names = pq.ParquetFile(files[0]).schema_arrow.names
        wanted = [n for n in names
                  if not n.startswith("pred_d") or n.endswith("_" + lookahead_input)]
        table = pa.concat_tables([pq.read_table(f, columns=wanted) for f in files],
                                 promote_options="default")

        def fsl(name: str, dtype) -> np.ndarray | None:
            if name not in table.column_names:
                return None
            arr = table.column(name).combine_chunks()
            if arr.null_count == len(arr):
                return None
            width = arr.type.list_size
            flat = arr.flatten().to_numpy(zero_copy_only=False).astype(dtype)
            out = flat.reshape(-1, width)
            if arr.null_count:
                mask = np.asarray(arr.is_null())
                out = out.copy()
                out[mask] = -1
            return out

        layer = table.column("layer").to_numpy(zero_copy_only=False).astype(np.int32)
        prompt = table.column("prompt_id").to_numpy(zero_copy_only=False).astype(np.int32)
        pos = table.column("pos").to_numpy(zero_copy_only=False).astype(np.int64)
        order = np.lexsort((layer, pos, prompt))

        top6 = fsl("top6_ids", np.int32)[order]
        top16 = fsl("top16_ids", np.int32)[order]
        scores = fsl("top16_scores", np.float32)[order]
        bias_applied = table.column("gate_bias_applied").to_numpy(
            zero_copy_only=False).astype(bool)[order]
        preds: dict[int, np.ndarray] = {}
        for name in table.column_names:
            if name.startswith("pred_d") and name.endswith("_" + lookahead_input):
                d = int(name[len("pred_d"):name.rindex("_")])
                a = fsl(name, np.int32)
                if a is not None:
                    preds[d] = a[order]

        n_layers = int(layer.max()) + 1
        n_experts = N_ROUTED_PER_LAYER
        t = Trace(top6, top16, scores, layer[order], prompt[order], pos[order],
                  preds, bias_applied, n_layers, n_experts)
        if not quiet:
            print(f"trace: {t.n_rows} rows, {t.n_tokens} tokens x {n_layers} layers, "
                  f"lookahead depths {sorted(preds)} from '{lookahead_input}'")
        return t


# --------------------------------------------------------------------------- #
# cache policies (design section 9.3)
# --------------------------------------------------------------------------- #

class Policy:
    """Every policy exposes the same three operations.

    `touch` is called for each demand request *after* `contains` decided hit or miss;
    `admit` inserts and returns the evicted key (or None); `observe` is the
    score-aware hook -- design section 9.3's "near hit", where an expert that scored
    in the top-16 but was not selected still refreshes its heat.
    """

    name = "base"

    def __init__(self, capacity: int):
        self.capacity = capacity

    def contains(self, key: int) -> bool: raise NotImplementedError
    def touch(self, key: int) -> None: raise NotImplementedError
    def admit(self, key: int) -> int | None: raise NotImplementedError
    def observe(self, keys, scores) -> None: pass
    def size(self) -> int: raise NotImplementedError


class LRU(Policy):
    name = "lru"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.od: "OrderedDict[int, None]" = OrderedDict()

    def contains(self, key): return key in self.od

    def touch(self, key): self.od.move_to_end(key)

    def admit(self, key):
        self.od[key] = None
        if len(self.od) > self.capacity:
            return self.od.popitem(last=False)[0]
        return None

    def size(self): return len(self.od)


class HeatPolicy(Policy):
    """Shared machinery for the two heat-ordered policies.

    Both want "evict the coldest", with heat decaying over time. Scanning the
    resident set for the minimum is O(capacity) per eviction, and at 4,600 slots and
    millions of evictions that does not finish -- so heat is kept *monotonically
    increasing* (the standard aging trick: instead of decaying every counter by
    `decay`, inflate the increment by `1/decay`) and eviction uses a lazy heap. A
    popped entry whose stored heat is below the current one is simply re-pushed,
    which terminates because heat only ever goes up.

    `gain` is rescaled before it can overflow; the ordering is unaffected because
    every counter is divided by the same number.
    """

    RESCALE_AT = 1e100

    def __init__(self, capacity: int, decay: float):
        super().__init__(capacity)
        self.decay = decay
        self.gain = 1.0
        self.heat: dict[int, float] = {}
        self.last: dict[int, int] = {}
        self.resident: set[int] = set()
        self.heap: list = []
        self.clock = 0

    def contains(self, key): return key in self.resident

    def _bump(self, key: int, weight: float):
        self.clock += 1
        self.gain /= self.decay
        if self.gain > self.RESCALE_AT:
            inv = 1.0 / self.gain
            for k in self.heat:
                self.heat[k] *= inv
            self.heap = [(h * inv, c, k) for h, c, k in self.heap]
            heapq.heapify(self.heap)
            self.gain = 1.0
        self.heat[key] = self.heat.get(key, 0.0) + weight * self.gain
        self.last[key] = self.clock
        heapq.heappush(self.heap, (self.heat[key], self.clock, key))

    def admit(self, key):
        self.resident.add(key)
        if key not in self.heat:
            self._bump(key, 1.0)
        if len(self.resident) <= self.capacity:
            return None
        while self.heap:
            h, clk, k = heapq.heappop(self.heap)
            if k not in self.resident:
                continue
            cur = self.heat[k]
            if h < cur:                      # stale entry: this key got hotter
                heapq.heappush(self.heap, (cur, self.last[k], k))
                continue
            self.resident.discard(k)
            return k
        return None

    def size(self): return len(self.resident)


class LFUDecay(HeatPolicy):
    """Counts with exponential decay, so an expert that was hot an hour ago loses to
    one that is hot now. Ties break on last use, which makes it LRU at equal counts."""

    name = "lfu-decay"

    def __init__(self, capacity: int, decay: float = 0.98):
        super().__init__(capacity, decay)

    def touch(self, key): self._bump(key, 1.0)


class ARC(Policy):
    """Adaptive Replacement Cache (Megiddo & Modha).

    T1/T2 hold resident keys seen once / more than once, B1/B2 are ghost lists of
    recently evicted keys, and `p` moves the T1/T2 split toward whichever ghost list
    is being hit. Worth testing here because expert routing mixes a recency-driven
    working set with a frequency-driven one, which is exactly what ARC targets.
    """

    name = "arc"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        c = capacity
        self.c = c
        self.p = 0.0
        self.t1: "OrderedDict[int, None]" = OrderedDict()
        self.t2: "OrderedDict[int, None]" = OrderedDict()
        self.b1: "OrderedDict[int, None]" = OrderedDict()
        self.b2: "OrderedDict[int, None]" = OrderedDict()

    def contains(self, key): return key in self.t1 or key in self.t2

    def touch(self, key):
        if key in self.t1:
            del self.t1[key]
        self.t2[key] = None
        self.t2.move_to_end(key)

    def _replace(self, key_in_b2: bool) -> int | None:
        """ARC's REPLACE(x, p): take the victim from T1 when T1 is over its target
        size p, otherwise from T2. The victim becomes a ghost in the matching B."""
        if self.t1 and (len(self.t1) > self.p or (key_in_b2 and len(self.t1) == self.p)):
            victim = self.t1.popitem(last=False)[0]
            self.b1[victim] = None
        elif self.t2:
            victim = self.t2.popitem(last=False)[0]
            self.b2[victim] = None
        else:
            return None
        return victim

    def admit(self, key):
        """Cases II, III and IV of the published algorithm (Case I -- a hit in T1 or
        T2 -- is `touch`). The ghost lists are bounded by Case IV rather than by a
        blanket trim: trimming inside REPLACE can evict the very key being promoted,
        which is how this first went wrong."""
        c = self.c
        victim = None
        if key in self.b1:                                   # Case II
            delta = 1.0 if len(self.b1) >= len(self.b2) else len(self.b2) / len(self.b1)
            self.p = min(float(c), self.p + delta)
            victim = self._replace(False)
            self.b1.pop(key, None)
            self.t2[key] = None
        elif key in self.b2:                                 # Case III
            delta = 1.0 if len(self.b2) >= len(self.b1) else len(self.b1) / len(self.b2)
            self.p = max(0.0, self.p - delta)
            victim = self._replace(True)
            self.b2.pop(key, None)
            self.t2[key] = None
        else:                                                # Case IV
            l1 = len(self.t1) + len(self.b1)
            total = l1 + len(self.t2) + len(self.b2)
            if l1 == c:
                if len(self.t1) < c:
                    self.b1.popitem(last=False)
                    victim = self._replace(False)
                else:
                    victim = self.t1.popitem(last=False)[0]
            elif l1 < c and total >= c:
                if total >= 2 * c and self.b2:
                    self.b2.popitem(last=False)
                victim = self._replace(False)
            self.t1[key] = None
        return victim

    def size(self): return len(self.t1) + len(self.t2)


class ScoreAware(HeatPolicy):
    """Design section 9.3's baseline: LRU plus score-aware promotion.

    The router already produces the top-16 scores every layer. An expert that lands
    in the top-16 but outside the top-6 is a "near hit": the text is drifting toward
    it, so it refreshes `heat` even though it was never read. Eviction orders by
    heat, so a near-miss expert survives a sweep that plain LRU would not let it.

    A demand hit is worth a full unit; a near hit is worth its router score, which
    `sqrt(softplus(.))` keeps positive and on the order of 1.
    """

    name = "score-aware"

    def __init__(self, capacity: int, decay: float = 0.9):
        super().__init__(capacity, decay)

    def touch(self, key): self._bump(key, 1.0)

    def observe(self, keys, scores):
        for k, s in zip(keys, scores):
            if k in self.resident:
                self._bump(k, float(s))


class StaticPinLRU(Policy):
    """A pinned set chosen from the trace's first `pin_train_frac`, plus LRU for the
    rest. This is the policy Q1 either justifies or kills: if `noaux_tc` balances the
    experts well, the pinned set is just a smaller LRU."""

    name = "static-pin+lru"

    def __init__(self, capacity: int, pinned: set[int], keyspace: range | None = None):
        # Inside a per-layer pool, only this layer's share of the pinned set is
        # relevant -- pinning the global set into a 115-slot pool leaves nothing for
        # the LRU and the hit rate collapses to a few percent.
        if keyspace is not None:
            pinned = {k for k in pinned if k in keyspace}
        if len(pinned) >= capacity:
            pinned = set(sorted(pinned)[:max(0, capacity - 1)])
        super().__init__(capacity)
        self.pinned = pinned
        self.lru = LRU(max(1, capacity - len(pinned)))

    def contains(self, key): return key in self.pinned or self.lru.contains(key)

    def touch(self, key):
        if key not in self.pinned:
            self.lru.touch(key)

    def admit(self, key):
        if key in self.pinned:
            return None
        return self.lru.admit(key)

    def size(self): return len(self.pinned) + self.lru.size()


class PerLayerPool(Policy):
    """One independent pool per layer, each with `capacity // n_layers` slots.

    Q2 decides whether this beats a single global pool: it does only if the layers'
    reuse-distance distributions differ enough that a global LRU lets one layer's
    working set evict another's.
    """

    name = "per-layer"

    def __init__(self, capacity: int, n_layers: int, n_experts: int, make):
        super().__init__(capacity)
        self.n_experts = n_experts
        quota = max(1, capacity // n_layers)
        self.pools = [make(quota, range(L * n_experts, (L + 1) * n_experts))
                      for L in range(n_layers)]
        self.name = f"{self.pools[0].name}/per-layer"

    def _pool(self, key): return self.pools[key // self.n_experts]

    def contains(self, key): return self._pool(key).contains(key)
    def touch(self, key): self._pool(key).touch(key)
    def admit(self, key): return self._pool(key).admit(key)

    def observe(self, keys, scores):
        for k, s in zip(keys, scores):
            self._pool(k).observe([k], [s])

    def size(self): return sum(p.size() for p in self.pools)


def policy_factory(name: str, args, pinned: set[int] | None):
    """-> make(capacity, keyspace=None). `keyspace` is the range of global keys the
    pool will ever see, which only `static-pin+lru` needs (to keep one layer's pool
    from pinning every other layer's experts)."""
    base = {
        "lru": lambda c, ks=None: LRU(c),
        "lfu-decay": lambda c, ks=None: LFUDecay(c, args.lfu_decay),
        "arc": lambda c, ks=None: ARC(c),
        "score-aware": lambda c, ks=None: ScoreAware(c),
        "static-pin+lru": lambda c, ks=None: StaticPinLRU(c, pinned or set(), ks),
    }
    if name not in base:
        raise SystemExit(f"unknown policy {name!r}")
    return base[name]


# --------------------------------------------------------------------------- #
# the simulator
# --------------------------------------------------------------------------- #

def simulate(trace: Trace, policy: Policy, args, depth: int = 0, width: int = 0) -> dict:
    """Replay the trace through one cache configuration.

    Demand path: at each (token, layer) the six routed experts are looked up; a miss
    admits the expert, evicting per the policy.

    Prefetch path (design section 9.4): after layer L-d's FFN the planner has a
    predicted top-K for layer L, and issues the ones that are not resident. The
    timing model is deliberately coarse but explicit:

        * a layer takes `--t-layer-ms` of GPU time, so a prefetch issued d layers
          early has `d * t_layer` to land;
        * the NVMe serves at `--nvme-gbps` with a floor of `--t-io-ms` per request
          and `--io-qd` in flight, so a batch of m requests takes
          max(t_io, m / qd * t_io, m * EXPERT_BYTES / bw);
        * demand misses (P0) preempt: they are charged the same service time but the
          GPU stalls for it.

    A miss at layer L counts as *hidden* if the expert was prefetched at L-d and had
    time to land, and *exposed* otherwise. Exposed misses are what `stall_ms` is.

    `hit_rate` is section 9.8's definition -- resident hits over requests -- and a
    hidden miss is **not** one of them: the bytes still crossed the NVMe. What a
    hidden miss buys is that the GPU did not wait, which `effective_hit_rate` and
    `stall_ms_per_token` report instead.

    Prefetch only ever runs inside one token: the prediction on row L came from that
    token's own layer L-d, so a planner at layer 39 has nothing to say about the next
    token's layer 2 -- the next token does not exist yet.

    Prefetched slots enter as probes at the LRU tail (section 9.4's last line): they
    are admitted but not touched, so an unused prefetch is the first thing evicted.
    """
    n_layers, n_exp = trace.n_layers, trace.n_experts
    keys6 = trace.keys6
    scores16 = trace.top16_scores
    top16 = trace.top16
    pred = trace.preds.get(depth) if depth > 0 else None
    if depth > 0 and pred is None:
        return {"skipped": f"trace has no lookahead depth {depth}"}

    rows = trace.n_rows
    warm = int(rows * args.warmup_frac)
    hits = np.zeros(n_layers, dtype=np.int64)
    reqs = np.zeros(n_layers, dtype=np.int64)
    hidden = exposed = 0
    prefetch_issued = prefetch_used = 0
    stall_ms = 0.0
    bw_bytes_per_ms = args.nvme_gbps * 1e9 / 1e3
    t_io = args.t_io_ms
    qd = max(1, args.io_qd)

    # in_flight[key] = simulated completion time (ms since the start of the replay)
    in_flight: dict[int, float] = {}
    issued: set[int] = set()
    now = 0.0
    is_score_aware = isinstance(policy, ScoreAware) or (
        isinstance(policy, PerLayerPool) and policy.name.startswith("score-aware"))

    for i in range(rows):
        L = int(trace.layer[i])
        now += args.t_layer_ms
        counted = i >= warm

        # --- demand ------------------------------------------------------
        row_keys = keys6[i]
        misses = []
        for k in row_keys:
            k = int(k)
            if counted:
                reqs[L] += 1
            if policy.contains(k):
                policy.touch(k)
                if counted:
                    hits[L] += 1
                in_flight.pop(k, None)
                if k in issued:
                    prefetch_used += 1
                    issued.discard(k)
            else:
                arrival = in_flight.pop(k, None)
                if arrival is not None and arrival <= now:
                    # landed in time: the planner hid this one
                    if counted:
                        hidden += 1
                    if k in issued:
                        prefetch_used += 1
                        issued.discard(k)
                    policy.admit(k)
                    policy.touch(k)
                else:
                    misses.append(k)
        if misses:
            service = max(t_io, len(misses) / qd * t_io,
                          len(misses) * EXPERT_BYTES / bw_bytes_per_ms)
            if counted:
                exposed += len(misses)
                stall_ms += service
            now += service
            for k in misses:
                policy.admit(k)
                policy.touch(k)

        if is_score_aware:
            policy.observe([L * n_exp + int(e) for e in top16[i]], scores16[i])

        # --- issue the lookahead for the layer `depth` rows ahead ---------
        if pred is not None and i + depth < rows and L + depth < n_layers:
            tgt = i + depth
            if int(trace.prompt[tgt]) == int(trace.prompt[i]):
                tl = int(trace.layer[tgt])
                want = pred[tgt][:width]
                if want[0] >= 0:
                    fresh = list(dict.fromkeys(tl * n_exp + int(e) for e in want))
                    fresh = [k for k in fresh
                             if not policy.contains(k) and k not in in_flight]
                    if fresh:
                        service = max(t_io, len(fresh) / qd * t_io,
                                      len(fresh) * EXPERT_BYTES / bw_bytes_per_ms)
                        for k in fresh:
                            in_flight[k] = now + service
                            issued.add(k)
                        if counted:
                            prefetch_issued += len(fresh)

    total_req = int(reqs.sum()) or 1
    total_hit = int(hits.sum())
    tokens = max(1, (rows - warm) // n_layers)
    wasted = max(0, prefetch_issued - prefetch_used)
    # Every byte that crossed the NVMe: demand misses the planner did not hide,
    # prefetches that arrived in time, and prefetches nobody used.
    nvme_bytes = (exposed + hidden + wasted) * EXPERT_BYTES / tokens
    lpddr_bytes = total_hit * EXPERT_BYTES / tokens
    stall = stall_ms / tokens
    t_serial = (RESIDENT_MS + lpddr_bytes / (args.lpddr_gbps * 1e9) * 1e3
                + nvme_bytes / (args.nvme_gbps * 1e9) * 1e3)
    # With prefetch the NVMe runs alongside the GPU, so the floor is whichever of the
    # two is longer -- but nothing overlaps the bandwidth itself.
    t_overlap = max(RESIDENT_MS + lpddr_bytes / (args.lpddr_gbps * 1e9) * 1e3 + stall,
                    nvme_bytes / (args.nvme_gbps * 1e9) * 1e3)
    return {
        "policy": policy.name,
        "capacity": policy.capacity,
        "depth": depth,
        "width": width,
        "hit_rate": round(total_hit / total_req, 4),
        "effective_hit_rate": round((total_hit + hidden) / total_req, 4),
        "hit_rate_per_layer": [round(float(h) / max(1, r), 4)
                               for h, r in zip(hits, reqs)],
        "nvme_bytes_per_token": int(nvme_bytes),
        "miss_bytes_per_token": int((exposed + hidden) * EXPERT_BYTES / tokens),
        "ms_per_token_serial": round(t_serial, 1),
        "ms_per_token": round(t_overlap, 1),
        "tokens_per_s": round(1000.0 / t_overlap, 2),
        "tokens_per_s_serial": round(1000.0 / t_serial, 2),
        "hidden_misses": hidden,
        "exposed_misses": exposed,
        "stall_ms_per_token": round(stall, 2),
        "prefetch_issued": prefetch_issued,
        "prefetch_used": prefetch_used,
        "prefetch_precision": round(prefetch_used / max(1, prefetch_issued), 4),
        "prefetch_waste_bytes_per_token": int(wasted * EXPERT_BYTES / tokens),
    }


def run_one(trace: Trace, name: str, capacity: int, args, pinned, allocation: str,
            depth: int, width: int) -> dict:
    factory = policy_factory(name, args, pinned)
    policy = (PerLayerPool(capacity, trace.n_layers, trace.n_experts, factory)
              if allocation == "per-layer" else factory(capacity))
    out = simulate(trace, policy, args, depth, width)
    out["allocation"] = allocation
    return out


# --------------------------------------------------------------------------- #
# section 9.1 Q1 / Q2 / Q3
# --------------------------------------------------------------------------- #

def analyse_q1(trace: Trace, args) -> dict:
    """Static expert frequency: does top-x% cover y% of routes?

    Reported per layer and overall. If `noaux_tc` balances well the curve is nearly
    diagonal (top 10% of experts carry ~10% of routes) and a pinned set is not worth
    having; a knee means section 9.3's static-pin tier is real.
    """
    xs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    per_layer = []
    for L in range(trace.n_layers):
        sel = trace.layer == L
        if not sel.any():
            continue
        counts = np.bincount(trace.top6[sel].reshape(-1), minlength=trace.n_experts)
        s = np.sort(counts)[::-1].astype(np.float64)
        total = s.sum() or 1
        cum = np.cumsum(s) / total
        cover = {f"top{int(x * 100)}pct":
                 round(float(cum[min(len(cum) - 1,
                                     max(1, int(round(x * trace.n_experts))) - 1)]), 4)
                 for x in xs}
        p = s / total
        nz = p[p > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        per_layer.append({
            "layer": L,
            "distinct": int((counts > 0).sum()),
            "coverage": cover,
            "entropy_bits": round(entropy, 3),
            "max_entropy_bits": round(math.log2(trace.n_experts), 3),
            "gini": round(float(_gini(s)), 4),
        })
    agg = {f"top{int(x * 100)}pct":
           round(float(np.mean([l["coverage"][f"top{int(x * 100)}pct"]
                                for l in per_layer])), 4) for x in xs}
    return {"per_layer": per_layer, "mean_coverage": agg,
            "note": "coverage[topXpct] = share of all routes taken by the busiest "
                    "X% of that layer's experts; a diagonal curve means noaux_tc "
                    "left no static skew to exploit"}


def _gini(sorted_desc: np.ndarray) -> float:
    x = np.sort(sorted_desc)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def stack_distances(keys: np.ndarray) -> np.ndarray:
    """Exact reuse (stack) distance for every access: the number of *distinct* keys
    touched since this key was last touched, or -1 for a first touch.

    An access with stack distance < C is a hit under LRU at capacity C, so the CDF of
    these numbers is LRU's hit-rate curve at every capacity at once -- which is why
    design section 9.1 Q2 asks for it rather than for a sweep.

    Implemented with a Fenwick tree over access time: each key's most recent access
    is marked 1, and the distance is the count of marks strictly after the previous
    access. O(n log n), and the loop is Python, hence `--q2-max-accesses`.
    """
    n = len(keys)
    tree = [0] * (n + 1)

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
    out = np.full(n, -1, dtype=np.int64)
    for t in range(n):
        k = int(keys[t])
        prev = last.get(k)
        if prev is not None:
            out[t] = pref(t - 1) - pref(prev)
            add(prev, -1)
        add(t, 1)
        last[k] = t
    return out


def analyse_q2(trace: Trace, args) -> dict:
    """Reuse-distance distribution, globally and per layer."""
    flat = trace.keys6.reshape(-1)
    if len(flat) > args.q2_max_accesses:
        flat = flat[:args.q2_max_accesses]
    d = stack_distances(flat)
    finite = d[d >= 0]
    caps = [int(N_ROUTED_EXPERTS_TOTAL * f) for f in (0.05, 0.10, 0.15, 0.20, 0.25,
                                                      0.30, 0.35, 0.40, 0.50)]
    caps.append(CACHE_SLOTS_90GB)
    lru_curve = {str(c): round(float((finite < c).sum()) / len(d), 4) for c in sorted(caps)}

    per_layer = []
    for L in range(trace.n_layers):
        sel = trace.layer == L
        if not sel.any():
            continue
        kl = trace.top6[sel].reshape(-1)
        dl = stack_distances(kl)
        fl = dl[dl >= 0]
        per_layer.append({
            "layer": L,
            "p50": int(np.percentile(fl, 50)) if len(fl) else -1,
            "p90": int(np.percentile(fl, 90)) if len(fl) else -1,
            "cold_frac": round(float((dl < 0).mean()), 4),
        })
    return {
        "global_lru_hit_rate_by_capacity": lru_curve,
        "cold_frac": round(float((d < 0).mean()), 4),
        "p50": int(np.percentile(finite, 50)) if len(finite) else -1,
        "p90": int(np.percentile(finite, 90)) if len(finite) else -1,
        "per_layer_within_layer_distance": per_layer,
        "note": "global distances are over the interleaved (token, layer) stream, "
                "which is what a single global pool sees. The per-layer numbers are "
                "distances within one layer's own request stream, i.e. what a "
                "per-layer quota would see; if they are much smaller than "
                "global/40, quotas help. The CDF is the *sequential* LRU curve; the "
                "simulator issues a layer's six requests as one batch (a miss is not "
                "masked by a sibling's admit), so the two differ by a fraction of a "
                "point at capacities near 6",
    }


def _intersection_sizes(a: np.ndarray, b: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """|row of a ∩ row of b| for every row, assuming each row has distinct entries.

    Both `top6` and a predicted top-K come out of a `topk`, so entries within a row
    are distinct and a broadcast equality is exact. Chunked because the intermediate
    is [rows, |a|, |b|] bools.
    """
    out = np.empty(len(a), dtype=np.int32)
    for lo in range(0, len(a), chunk):
        hi = min(lo + chunk, len(a))
        out[lo:hi] = (a[lo:hi, :, None] == b[lo:hi, None, :]).any(-1).sum(-1)
    return out


def _union_sizes(rows: np.ndarray) -> np.ndarray:
    """|set(row)| for every row of a [n, w] integer array, vectorised via sort."""
    s = np.sort(rows, axis=1)
    return (np.diff(s, axis=1) != 0).sum(axis=1) + 1


def analyse_q3(trace: Trace, args, max_k: int = 5) -> dict:
    """Jaccard overlap of consecutive tokens' expert sets, and the union growth over a
    speculative window (design sections 9.5 and 10.3).

    `union_k / 6k` is what a verify batch of k tokens costs in NVMe traffic relative
    to k independent tokens: 1.0 means speculation buys nothing on the NVMe side,
    1/k means the batch reads what a single token would.
    """
    per_layer = []
    union_acc: dict[int, list[float]] = {k: [] for k in range(2, max_k + 1)}
    for L in range(trace.n_layers):
        sel = np.flatnonzero(trace.layer == L)
        if len(sel) < 2:
            continue
        top6 = trace.top6[sel]
        prompts = trace.prompt[sel]
        same = prompts[:-1] == prompts[1:]
        inter = _intersection_sizes(top6[:-1][same], top6[1:][same])
        # |A ∩ B| / |A ∪ B| with |A| = |B| = 6
        jac = inter / (12 - inter)
        u: dict[int, float] = {}
        for k in range(2, max_k + 1):
            if len(sel) < k:
                continue
            ok = prompts[:len(sel) - k + 1] == prompts[k - 1:]
            if not ok.any():
                continue
            win = np.concatenate([top6[i:len(sel) - k + 1 + i] for i in range(k)], axis=1)
            frac = _union_sizes(win[ok]) / (6.0 * k)
            u[k] = round(float(frac.mean()), 4)
            union_acc[k].append(u[k])
        per_layer.append({"layer": L,
                          "jaccard_mean": round(float(jac.mean()), 4) if len(jac) else 0.0,
                          "union_frac": u})
    return {
        "per_layer": per_layer,
        "mean_jaccard": round(float(np.mean([l["jaccard_mean"] for l in per_layer])), 4)
        if per_layer else 0.0,
        "mean_union_frac": {k: round(float(np.mean(v)), 4)
                            for k, v in union_acc.items() if v},
        "note": "union_frac[k] = |union of k consecutive tokens' experts| / 6k, per "
                "layer; it is the NVMe cost of a k-token verify batch relative to k "
                "separate tokens (section 10.3's T_nvme(k)). It assumes every draft "
                "token is accepted, so it is the conservative end: a rejected draft "
                "shortens the batch and shrinks the union",
    }


def analyse_q4(trace: Trace, args) -> dict:
    """Lookahead recall and precision over the (d, K) grid the trace recorded."""
    out = {}
    for d, pred in sorted(trace.preds.items()):
        valid = pred[:, 0] >= 0
        if not valid.any():
            continue
        actual = trace.top6[valid]
        p = pred[valid]
        row = {}
        for K in (6, 8, 10, 12, 16):
            if K > p.shape[1]:
                continue
            hit = _intersection_sizes(actual, p[:, :K]).astype(np.float64)
            row[K] = {"recall": round(float((hit / 6).mean()), 4),
                      "precision": round(float((hit / K).mean()), 4)}
        out[d] = row
    return {"by_depth": out,
            "rows": int(sum(int((p[:, 0] >= 0).sum()) for p in trace.preds.values())),
            "note": "recall@K = |predicted top-K ∩ actual top-6| / 6; precision is "
                    "the same intersection over K. Section 9.4 sizes the P1 queue "
                    "from these: d is how many layers of lookahead, K how wide"}


def analyse_gate_bias(trace: Trace) -> dict:
    """How often the `noaux_tc` correction bias changes the selected top-6.

    Not one of section 9.1's questions, but it is the cheapest available read on
    whether the load balancer is actively flattening the distribution, which is what
    Q1 is really asking.
    """
    per_layer = []
    for L in range(trace.n_layers):
        sel = trace.layer == L
        if sel.any():
            per_layer.append({"layer": L,
                              "frac": round(float(trace.bias_applied[sel].mean()), 4)})
    return {"per_layer": per_layer,
            "overall": round(float(trace.bias_applied.mean()), 4)}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def parse_capacities(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("abs:"):
            out.append((part, int(part[4:])))
        else:
            pct = float(part)
            out.append((f"{part}%", int(N_ROUTED_EXPERTS_TOTAL * pct / 100)))
    return out


def pick_pinned(trace: Trace, args) -> set[int]:
    """The busiest `pin_frac` of each layer's experts, chosen from the first
    `pin_train_frac` of the trace only -- a pinned set chosen from the whole trace
    would be an oracle and would flatter the policy."""
    cut = int(trace.n_rows * args.pin_train_frac)
    pinned: set[int] = set()
    per_layer = max(1, int(trace.n_experts * args.pin_frac))
    for L in range(trace.n_layers):
        sel = (trace.layer[:cut] == L)
        if not sel.any():
            continue
        counts = np.bincount(trace.top6[:cut][sel].reshape(-1),
                             minlength=trace.n_experts)
        for e in np.argsort(counts)[::-1][:per_layer]:
            pinned.add(L * trace.n_experts + int(e))
    return pinned


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace = Trace.load(args.trace, args.lookahead_input, args.quiet)

    report: dict = {
        "trace": {"rows": trace.n_rows, "tokens": trace.n_tokens,
                  "layers": trace.n_layers, "experts_per_layer": trace.n_experts,
                  "sources": args.trace},
        "model": {"expert_bytes": EXPERT_BYTES, "resident_ms": RESIDENT_MS,
                  "nvme_gbps": args.nvme_gbps, "lpddr_gbps": args.lpddr_gbps,
                  "t_layer_ms": args.t_layer_ms, "t_io_ms": args.t_io_ms,
                  "io_qd": args.io_qd},
    }

    qs = {q.strip() for q in args.questions.split(",") if q.strip()}
    if "q1" in qs:
        report["q1_static_frequency"] = analyse_q1(trace, args)
        report["gate_bias"] = analyse_gate_bias(trace)
    if "q2" in qs:
        report["q2_reuse_distance"] = analyse_q2(trace, args)
    if "q3" in qs:
        report["q3_jaccard"] = analyse_q3(trace, args)
    if "q4" in qs or trace.preds:
        report["q4_lookahead"] = analyse_q4(trace, args)

    caps = parse_capacities(args.capacities)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    depths = [int(d) for d in args.prefetch_depths.split(",") if d.strip()]
    widths = [int(w) for w in args.prefetch_widths.split(",") if w.strip()]
    allocations = (["global", "per-layer"] if args.allocation == "both"
                   else [args.allocation])
    pinned = pick_pinned(trace, args) if "static-pin+lru" in policies else set()

    runs = []
    for cap_label, cap in caps:
        for alloc in allocations:
            for name in policies:
                for d in depths:
                    for w in (widths if d else [0]):
                        if d and w < 6:
                            continue
                        r = run_one(trace, name, cap, args, pinned, alloc, d, w)
                        if "skipped" in r:
                            continue
                        r["capacity_label"] = cap_label
                        runs.append(r)
                        if not args.quiet:
                            print(f"  {cap_label:>6} {alloc:9} {name:15} "
                                  f"d={d} K={w or '-':>2}  hit {r['hit_rate']:.4f}  "
                                  f"eff {r['effective_hit_rate']:.4f}  "
                                  f"{r['ms_per_token']:7.1f} ms  "
                                  f"{r['tokens_per_s']:5.2f} tok/s  "
                                  f"stall {r['stall_ms_per_token']:6.1f} ms  "
                                  f"pf prec {r['prefetch_precision']:.3f}")
    report["runs"] = runs
    best = max(runs, key=lambda r: r["hit_rate"]) if runs else None
    report["best_by_hit_rate"] = best

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nreport -> {args.out}")
    elif args.quiet:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
