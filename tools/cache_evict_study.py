"""Track E1: eviction-policy study on the real routing trace (CPU, offline).

`docs/p4_cache_policy.md` measured prefetching and found every realizable
predictor net-negative, and named eviction as the one untested lever: Belady
demand-only is +42% at C=4,500 and +34% at C=5,711.  This file replays the same
request stream under the same timing model, but varies only the *eviction*
policy.  There is no prefetching anywhere: every byte moved is a demand miss.

Timing model (docs/p4_cache_policy.md section 1, with the drive rate set to the
4.6 GB/s effective figure measured on D: for this track):

    tok/s = 1000 / (80 + stall_ms_per_token)
    stall_ms_per_token = misses_per_token * EXPERT_BYTES / (GBPS * 1e9) * 1000

With no prefetch in flight the drive is empty whenever a layer starts, so the
work-conserving server of `tools/cache_policy_study.py` degenerates to exactly
this closed form.  `--cross-check` asserts that against that file's `replay()`.

Usage::

    .venv/Scripts/python.exe tools/cache_evict_study.py \
        --trace traces/mixed --out bench/results/cache_evict
"""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import os
import time
from collections import OrderedDict

import numpy as np

EXPERT_BYTES = 18_800_640
N_EXP = 384
N_LAYERS = 40
T_LAYER_MS = 2.0
COMPUTE_MS = N_LAYERS * T_LAYER_MS      # 80 ms
GBPS = 4.6                              # measured effective rate on D:
INF = float("inf")


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #

def load_trace(trace_dir, cache_npz=None):
    """-> keys6 [rows,6], keys16 [rows,16], scores16 [rows,16], prompt [rows]."""
    if cache_npz and os.path.exists(cache_npz):
        z = np.load(cache_npz)
        return z["keys6"], z["keys16"], z["scores16"], z["prompt"]
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(trace_dir, "route_layer*.parquet")))
    if not files:
        raise SystemExit("no route_layer*.parquet under " + trace_dir)
    cols = ["prompt_id", "pos", "layer", "top6_ids", "top16_ids", "top16_scores"]
    tbl = pa.concat_tables([pq.read_table(f, columns=cols) for f in files],
                           promote_options="default")

    def fsl(name, dt):
        a = tbl.column(name).combine_chunks()
        return (a.flatten().to_numpy(zero_copy_only=False).astype(dt)
                .reshape(-1, a.type.list_size))

    layer = tbl.column("layer").to_numpy(zero_copy_only=False).astype(np.int32)
    prompt = tbl.column("prompt_id").to_numpy(zero_copy_only=False).astype(np.int32)
    pos = tbl.column("pos").to_numpy(zero_copy_only=False).astype(np.int64)
    order = np.lexsort((layer, pos, prompt))
    layer, prompt = layer[order], prompt[order]
    keys6 = (layer[:, None] * N_EXP + fsl("top6_ids", np.int32)[order]).astype(np.int32)
    keys16 = (layer[:, None] * N_EXP + fsl("top16_ids", np.int32)[order]).astype(np.int32)
    scores16 = fsl("top16_scores", np.float32)[order]
    if cache_npz:
        d = os.path.dirname(cache_npz)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez(cache_npz, keys6=keys6, keys16=keys16, scores16=scores16,
                 prompt=prompt)
    return keys6, keys16, scores16, prompt


# --------------------------------------------------------------------------- #
# caches
# --------------------------------------------------------------------------- #

class LRU:
    """Baseline: rank on last use only -- what `evict_lru` does today."""

    def __init__(self, cap, **kw):
        self.cap = cap
        self.d = OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, k, w=1.0):
        self.d.move_to_end(k)

    def admit(self, k, w=1.0):
        self.d[k] = None
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


class MinRank:
    """Generic 'evict the lowest rank' cache with a self-correcting lazy heap.

    `observe_lo` / `observe_first` say where a subclass's near-miss hook sits:
    from which top-16 column, and whether it runs before or after the layer's
    six lookups.

    A key's rank may move in either direction (an EMA of router scores does).
    On pop: a stored rank below the current one means the key got better since
    that entry was pushed, so the entry is refreshed and the pop continues; a
    stored rank above the current one duplicates an entry still in the heap, so
    it is dropped.  Both branches terminate because every rank change pushes
    exactly one entry.
    """

    observe_lo = 6
    observe_first = False

    def __init__(self, cap, **kw):
        self.cap = cap
        self.rank = {}
        self.heap = []
        self.seq = 0

    def __contains__(self, k):
        return k in self.rank

    def _set(self, k, r):
        """Record a new rank.

        A heap entry is pushed only when the rank *falls* (or the key is new).
        The invariant the eviction loop needs is just "every resident key has
        some entry at or below its current rank": a rank that rose leaves the
        older, lower entry valid, and `_evict` refreshes it when it surfaces.
        Skipping those pushes is what keeps a 16-per-row heat update affordable
        -- it turns ~12 M pushes per replay into ~0.2 M.
        """
        old = self.rank.get(k)
        self.rank[k] = r
        if old is None or r < old:
            self.seq += 1
            heapq.heappush(self.heap, (r, self.seq, k))

    def _evict(self):
        while self.heap:
            r, s, k = heapq.heappop(self.heap)
            cur = self.rank.get(k)
            if cur is None:
                continue
            if r < cur:
                self.seq += 1
                heapq.heappush(self.heap, (cur, self.seq, k))
                continue
            if r > cur:
                continue
            del self.rank[k]
            self._forget(k)
            return k
        return None

    def _forget(self, k):
        pass


class ScoreExt(MinRank):
    """LRU whose rank is *extended* by heat: rank = last_use + alpha * heat.

    `heat` is an EMA over touches: a demand use is worth `w_demand` (or the
    demand's own router score when `w_demand` is None), and an expert that
    landed in the router's top-16 but outside the top-6 -- a "near miss" -- is
    worth `w_near * score` even though it was never read.  alpha is in units of
    touches: alpha = 400 means a fully hot expert outlives its own last use by
    400 further admissions.  This is the form a real planner can evaluate: one
    float per slot, updated on events the router already produces.
    """

    def __init__(self, cap, alpha=400.0, decay=0.9, w_demand=1.0, w_near=1.0,
                 near_k=16, **kw):
        super().__init__(cap)
        self.alpha, self.decay = alpha, decay
        self.w_demand, self.w_near, self.near_k = w_demand, w_near, near_k
        self.heat = {}
        self.last = {}
        self.clock = 0

    def _demand(self, k, w):
        self.clock += 1
        inc = w if self.w_demand is None else self.w_demand
        h = self.heat.get(k, 0.0) * self.decay + inc
        self.heat[k] = h
        self.last[k] = self.clock
        self._set(k, self.clock + self.alpha * h)

    def touch(self, k, w=1.0):
        self._demand(k, w)

    def admit(self, k, w=1.0):
        self._demand(k, w)
        if len(self.rank) > self.cap:
            self._evict()

    def observe(self, keys, scores):
        """Near-miss touch: refresh heat, but not last_use."""
        wn, a, dec = self.w_near, self.alpha, self.decay
        rank, heat, last = self.rank, self.heat, self.last
        for k, s in zip(keys, scores):
            if k in rank:
                h = heat.get(k, 0.0) * dec + wn * s
                heat[k] = h
                self._set(k, last[k] + a * h)

    def _forget(self, k):
        self.heat.pop(k, None)
        self.last.pop(k, None)


class EwmaExt(MinRank):
    """The engine's own shape: rank = last_use + alpha * heat, heat an EWMA.

    `store/expert_store.cpp::note_heat` already does ``h = (1-a)h + a*score``
    for every id in the router's top-16 (the top-6 included) and only for slots
    that are resident; a fill resets heat to 0.  `evict_lru` then ignores heat
    entirely.  This class is that code with the one missing line put in, so its
    parameters (`ewma` = note_heat's alpha, `alpha` = the rank weight, in units
    of expert accesses) transfer to the engine unchanged.
    """

    observe_lo = 0          # note_heat runs over the whole top-16 ...
    observe_first = True    # ... before the layer's lookups

    def __init__(self, cap, alpha=400.0, ewma=0.125, near_k=16, **kw):
        super().__init__(cap)
        self.alpha, self.a, self.near_k = alpha, ewma, near_k
        self.heat = {}
        self.last = {}
        self.clock = 0

    def touch(self, k, w=1.0):
        self.clock += 1
        self.last[k] = self.clock
        self._set(k, self.clock + self.alpha * self.heat.get(k, 0.0))

    def admit(self, k, w=1.0):
        self.clock += 1
        self.heat[k] = 0.0                       # a fill resets heat
        self.last[k] = self.clock
        self._set(k, float(self.clock))
        if len(self.rank) > self.cap:
            self._evict()

    def observe(self, keys, scores):
        a, one_a, al = self.a, 1.0 - self.a, self.alpha
        rank, heat, last = self.rank, self.heat, self.last
        for k, s in zip(keys, scores):
            if k in rank:
                h = heat.get(k, 0.0) * one_a + a * s
                heat[k] = h
                self._set(k, last[k] + al * h)

    def _forget(self, k):
        self.heat.pop(k, None)
        self.last.pop(k, None)


class Heat(MinRank):
    """Pure heat order -- `cache_sim`'s score-aware / lfu-decay family.

    rank = heat; last use enters only through the heap's tie-break.
    `w_near = 0` degenerates to lfu-decay.
    """

    def __init__(self, cap, decay=0.9, w_demand=1.0, w_near=1.0, near_k=16, **kw):
        super().__init__(cap)
        self.decay, self.w_demand = decay, w_demand
        self.w_near, self.near_k = w_near, near_k
        self.heat = {}

    def _demand(self, k, w):
        inc = w if self.w_demand is None else self.w_demand
        h = self.heat.get(k, 0.0) * self.decay + inc
        self.heat[k] = h
        self._set(k, h)

    def touch(self, k, w=1.0):
        self._demand(k, w)

    def admit(self, k, w=1.0):
        self._demand(k, w)
        if len(self.rank) > self.cap:
            self._evict()

    def observe(self, keys, scores):
        wn, dec = self.w_near, self.decay
        for k, s in zip(keys, scores):
            if k in self.rank:
                h = self.heat.get(k, 0.0) * dec + wn * s
                self.heat[k] = h
                self._set(k, h)

    def _forget(self, k):
        self.heat.pop(k, None)


class LRUK:
    """LRU-K (Kth-most-recent reference).  K=2 is O'Neil's LRU-2."""

    def __init__(self, cap, k=2, **kw):
        self.cap, self.k = cap, k
        self.hist = {}
        self.res = {}
        self.heap = []
        self.clock = 0
        self.seq = 0

    def __contains__(self, key):
        return key in self.res

    def _push(self, key):
        h = self.hist[key]
        r = float(h[0]) if len(h) >= self.k else -1e18 + h[0]
        self.res[key] = r
        self.seq += 1
        heapq.heappush(self.heap, (r, self.seq, key))

    def _ref(self, key):
        self.clock += 1
        h = self.hist.setdefault(key, [])
        h.append(self.clock)
        if len(h) > self.k:
            del h[0]

    def touch(self, key, w=1.0):
        self._ref(key)
        self._push(key)

    def admit(self, key, w=1.0):
        self._ref(key)
        self._push(key)
        while len(self.res) > self.cap:
            popped = False
            while self.heap:
                r, s, k = heapq.heappop(self.heap)
                cur = self.res.get(k)
                if cur is None or r != cur:
                    continue
                del self.res[k]
                self.hist.pop(k, None)
                popped = True
                break
            if not popped:
                break


class ARC:
    """Adaptive Replacement Cache (Megiddo & Modha), ghost-list driven."""

    def __init__(self, cap, **kw):
        self.c = cap
        self.p = 0
        self.t1, self.t2 = OrderedDict(), OrderedDict()
        self.b1, self.b2 = OrderedDict(), OrderedDict()

    def __contains__(self, k):
        return k in self.t1 or k in self.t2

    def touch(self, k, w=1.0):
        if k in self.t1:
            del self.t1[k]
            self.t2[k] = None
        elif k in self.t2:
            self.t2.move_to_end(k)

    def _replace(self, k):
        if self.t1 and (len(self.t1) > self.p or
                        (k in self.b2 and len(self.t1) == self.p)):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = None
        elif self.t2:
            old, _ = self.t2.popitem(last=False)
            self.b2[old] = None

    def admit(self, k, w=1.0):
        c = self.c
        if k in self.b1:
            self.p = min(c, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            self._replace(k)
            del self.b1[k]
            self.t2[k] = None
        elif k in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            self._replace(k)
            del self.b2[k]
            self.t2[k] = None
        else:
            if len(self.t1) + len(self.b1) >= c:
                if len(self.t1) < c:
                    self.b1.popitem(last=False)
                    self._replace(k)
                else:
                    self.t1.popitem(last=False)
            elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
                if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * c:
                    self.b2.popitem(last=False)
                self._replace(k)
            self.t1[k] = None
        while len(self.t1) + len(self.t2) > c:
            self._replace(k)


class S3FIFO:
    """S3-FIFO (Yang et al. 2023): small probationary FIFO, main FIFO, ghost."""

    def __init__(self, cap, small_frac=0.1, **kw):
        self.small_cap = max(1, int(cap * small_frac))
        self.main_cap = max(1, cap - self.small_cap)
        self.S, self.M, self.G = OrderedDict(), OrderedDict(), OrderedDict()
        self.ghost_cap = self.main_cap
        self.freq = {}

    def __contains__(self, k):
        return k in self.S or k in self.M

    def touch(self, k, w=1.0):
        self.freq[k] = min(3, self.freq.get(k, 0) + 1)

    def _trim_m(self):
        while len(self.M) > self.main_cap:
            k, _ = self.M.popitem(last=False)
            f = self.freq.get(k, 0)
            if f > 0:
                self.freq[k] = f - 1
                self.M[k] = None
            else:
                self.freq.pop(k, None)

    def _evict_s(self):
        while self.S:
            k, _ = self.S.popitem(last=False)
            if self.freq.get(k, 0) > 1:
                self.M[k] = None
                self.freq[k] = 0
                self._trim_m()
            else:
                self.G[k] = None
                self.freq.pop(k, None)
                if len(self.G) > self.ghost_cap:
                    self.G.popitem(last=False)
                return

    def admit(self, k, w=1.0):
        if k in self.G:
            del self.G[k]
            self.M[k] = None
            self.freq[k] = 0
            self._trim_m()
            return
        self.S[k] = None
        self.freq[k] = 0
        while len(self.S) > self.small_cap:
            self._evict_s()


class ScoreBypass:
    """TinyLFU-style admission on top of ScoreExt: an incoming expert whose
    router score is below `admit_thresh` is *used but not kept*.

    Physically legal on this engine: the expert is read into the layer's
    scratch slot and dropped, so a cold one-shot expert cannot evict a hot
    resident.  It costs nothing extra -- the byte had to move either way.
    """

    def __init__(self, cap, alpha=400.0, decay=0.9, w_demand=1.0, w_near=1.0,
                 near_k=16, admit_thresh=0.5, **kw):
        self.inner = ScoreExt(cap, alpha=alpha, decay=decay, w_demand=w_demand,
                              w_near=w_near, near_k=near_k)
        self.thresh = admit_thresh
        self.near_k = near_k

    def __contains__(self, k):
        return k in self.inner

    def touch(self, k, w=1.0):
        self.inner.touch(k, w)

    def observe(self, keys, scores):
        self.inner.observe(keys, scores)

    def admit(self, k, w=1.0):
        if w < self.thresh and len(self.inner.rank) >= self.inner.cap:
            # not kept: heat still learns, so a repeat offender eventually clears
            self.inner.heat[k] = self.inner.heat.get(k, 0.0) * self.inner.decay + w
            return
        self.inner.admit(k, w)


class Belady:
    """Clairvoyant demand-only ceiling: evict the furthest next demand use."""

    def __init__(self, cap, **kw):
        self.cap = cap
        self.res = {}
        self.heap = []

    def __contains__(self, k):
        return k in self.res

    def _set(self, k, nu):
        self.res[k] = nu
        heapq.heappush(self.heap, (-nu, k))

    def touch(self, k, nu=None):
        self._set(k, INF if nu is None else nu)

    def admit(self, k, nu=None):
        self._set(k, INF if nu is None else nu)
        while len(self.res) > self.cap:
            popped = False
            while self.heap:
                negnu, cand = self.heap[0]
                if cand not in self.res or self.res[cand] != -negnu:
                    heapq.heappop(self.heap)
                    continue
                heapq.heappop(self.heap)
                del self.res[cand]
                popped = True
                break
            if not popped:
                break


def next_use(flat):
    """per_slot[i,j] = flat request index of that key's next demand use."""
    lin = flat.ravel().astype(np.int64)
    order = np.argsort(lin, kind="stable")
    s = lin[order]
    nxt = np.full(len(s), INF)
    same = s[:-1] == s[1:]
    nxt[:-1] = np.where(same, order[1:].astype(np.float64), INF)
    out = np.empty(len(lin), dtype=np.float64)
    out[order] = nxt
    return out.reshape(flat.shape)


MAKE = {
    "lru": LRU,
    "belady": Belady,
    "lru-k": LRUK,
    "arc": ARC,
    "s3-fifo": S3FIFO,
    "heat": Heat,
    "score-ext": ScoreExt,
    "ewma-ext": EwmaExt,
    "score-bypass": ScoreBypass,
}


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #

def replay(keys6, keys16, scores16, capacity, policy="lru", params=None,
           warm_rows=0, label="", gbps=GBPS):
    """Demand-only replay.  Returns hit rate and the timing-model projection.

    `warm_rows` rows at the front prime the cache without being scored, which is
    how a train/test split is evaluated from a hot cache.
    """
    params = dict(params or {})
    cache = MAKE[policy](capacity, **params)
    per = next_use(keys6) if policy == "belady" else None
    near_k = int(params.get("near_k", 16))
    observe = getattr(cache, "observe", None)
    lo = getattr(cache, "observe_lo", 6)
    first = getattr(cache, "observe_first", False)
    if observe is not None and near_k <= lo:
        observe = None

    rows = keys6.shape[0]
    hit = req = 0
    t0 = time.time()
    for i in range(rows):
        row = keys6[i]
        srow = scores16[i]
        scored = i >= warm_rows
        if observe is not None and first:
            k16 = keys16[i]
            observe([int(x) for x in k16[lo:near_k]], srow[lo:near_k])
        for j in range(6):
            k = int(row[j])
            if scored:
                req += 1
            arg = per[i, j] if per is not None else float(srow[j])
            if k in cache:
                if scored:
                    hit += 1
                cache.touch(k, arg)
            else:
                cache.admit(k, arg)
        if observe is not None and not first:
            k16 = keys16[i]
            observe([int(x) for x in k16[lo:near_k]], srow[lo:near_k])

    hr = hit / max(1, req)
    s_ms = EXPERT_BYTES / (gbps * 1e9) * 1000.0
    n_tok = max(1, (rows - warm_rows) / N_LAYERS)
    miss_per_tok = (req - hit) / n_tok
    stall = miss_per_tok * s_ms
    return {
        "label": label or policy,
        "policy": policy,
        "params": params,
        "capacity": capacity,
        "gbps": gbps,
        "tokens": n_tok,
        "hit_rate": hr,
        "miss_per_token": miss_per_tok,
        "stall_ms_per_token": stall,
        "tok_s": 1000.0 / (COMPUTE_MS + stall),
        "seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

# The ridge found by the sweep in docs/p4_cache_policy.md section 11.  It is
# flat: everything from alpha 1,600 to 4,800 with decay 0.90-0.95 lands within
# 0.2 pt of hit, so these are representatives, not a fitted optimum.
FINALISTS = [
    ("score-ext", dict(alpha=3200, decay=0.90, w_near=2.0, near_k=16)),
    ("score-ext", dict(alpha=1600, decay=0.95, w_near=2.0, near_k=16)),
    ("score-ext", dict(alpha=3200, decay=0.90, w_near=0.0, near_k=6)),
    ("ewma-ext", dict(alpha=51200, ewma=0.125, near_k=16)),
]

_G = {}


def _init(npz):
    z = np.load(npz)
    _G["k6"], _G["k16"], _G["sc"] = z["keys6"], z["keys16"], z["scores16"]


def _job(a):
    lab, cap, pol, par, lo, hi, warm, gbps = a
    return replay(_G["k6"][lo:hi], _G["k16"][lo:hi], _G["sc"][lo:hi], cap,
                  policy=pol, params=par, warm_rows=warm, label=lab, gbps=gbps)


def name_of(pol, par):
    return pol + " " + " ".join("%s=%g" % kv for kv in sorted(par.items()))


def cross_check(k6, k16, sc, capacity, tokens=4000, gbps=GBPS):
    """Assert the closed form against `cache_policy_study`'s queueing replay."""
    import cache_policy_study as cps
    n = tokens * N_LAYERS
    a = replay(k6[:n], k16[:n], sc[:n], capacity, "lru", gbps=gbps)
    b = cps.replay(np.ascontiguousarray(k6[:n]).reshape(-1, N_LAYERS, 6),
                   capacity, evict="lru", nvme_gbps=gbps)
    print("closed form  : hit %.6f  stall %.4f  tok/s %.4f"
          % (a["hit_rate"], a["stall_ms_per_token"], a["tok_s"]))
    print("Drive replay : hit %.6f  stall %.4f  tok/s %.4f"
          % (b["hit_rate"], b["stall_ms_per_token"], b["tok_s"]))
    assert abs(a["tok_s"] - b["tok_s"]) < 1e-3, "timing models disagree"
    print("agree")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="traces/mixed")
    ap.add_argument("--out", default="bench/results/cache_evict")
    ap.add_argument("--capacities", default="4500,5100,5711")
    ap.add_argument("--gbps", type=float, default=GBPS)
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="prompt fraction held out of the test replay")
    ap.add_argument("--warm-tokens", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--cross-check", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    npz = os.path.join(args.out, "trace_cols.npz")
    k6, k16, sc, prompt = load_trace(args.trace, npz)
    if args.cross_check:
        cross_check(k6, k16, sc, 5100, gbps=args.gbps)
        return

    ids = np.unique(prompt)
    cut = int(np.searchsorted(prompt, ids[int(len(ids) * args.train_frac)]))
    warm = args.warm_tokens * N_LAYERS
    caps = [int(x) for x in args.capacities.split(",")]

    jobs = []
    for c in caps:
        for split, lo, hi, w in (("full", 0, None, 0),
                                 ("train", 0, cut, 0),
                                 ("test", cut - warm, None, warm)):
            jobs.append(("%s lru" % split, c, "lru", {}, lo, hi, w, args.gbps))
            jobs.append(("%s belady" % split, c, "belady", {}, lo, hi, w, args.gbps))
            for pol, par in FINALISTS:
                jobs.append(("%s %s" % (split, name_of(pol, par)), c, pol, par,
                             lo, hi, w, args.gbps))

    t0 = time.time()
    if args.procs > 1:
        import multiprocessing as mp
        with mp.Pool(args.procs, initializer=_init, initargs=(npz,)) as p:
            res = p.map(_job, jobs, chunksize=1)
    else:
        _init(npz)
        res = [_job(j) for j in jobs]

    path = os.path.join(args.out, "evict_study.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    res.sort(key=lambda r: (r["capacity"], -r["tok_s"]))
    for r in res:
        print("%-50s C=%-5d hit %.4f  stall %6.1f  tok/s %.3f"
              % (r["label"], r["capacity"], r["hit_rate"],
                 r["stall_ms_per_token"], r["tok_s"]))
    print("wall %.0fs -> %s" % (time.time() - t0, path))


if __name__ == "__main__":
    main()
