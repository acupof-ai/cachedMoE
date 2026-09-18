"""Track X: cache / prefetch policy study on the real routing trace.

Offline, CPU only.  Reads ``traces/mixed`` through ``tools/cache_sim.Trace``
(which this file never edits), then replays the decode request stream under an
explicit NVMe model and a set of cache / prefetch policies.  Every policy is
scored on the trace's *actual* routing, so the hit rates and stalls are exact
for this trace rather than estimates.

Timing model -- see docs/p4_cache_policy.md section 2:

* per token, layer L's routing becomes known when layer L starts computing;
* ``T_LAYER_MS`` = 2.0 ms of GPU compute per layer, 80 ms per token;
* one expert is 18.8 MB; the NVMe is a single work-conserving server at
  4.5 GB/s, so one expert occupies ``S_MS`` = 18_800_640 / 4.5e9 = 4.178 ms of
  drive time.  An isolated request completes 4.18 ms after issue (the measured
  4.0-4.7 ms single-expert latency); a saturated drive delivers one expert per
  4.178 ms (the measured 4.5 GB/s).  QD 8 and 4 MiB chunking are folded into
  that single number: at QD 8 the drive is bandwidth bound, so concurrency
  changes *which* request finishes when, not how many finish per second, and a
  layer stalls until all six of its experts have landed anyway.
* the drive is issued in 4 MiB chunks, so the server's quantum is
  4 MiB / 4.5 GB/s = 0.932 ms.  Demand requests outrank prefetches and a queued
  prefetch that turns into a demand is promoted, but a demand still waits for
  the chunk in service -- that is the standing head-of-line cost of prefetching.
  ``--quantum-ms 4.178`` reruns everything with a non-preemptible whole-expert
  server as the pessimistic bound;
* a planner may hold at most ``PF_QUEUE`` outstanding prefetches and re-plans at
  every token boundary, where prefetches that have not started are cancelled.
  Bytes are billed from work the drive actually performed, so a cancelled or
  never-scheduled prefetch costs nothing;
* a prefetch enters the cache (taking a slot, evicting something) when it
  arrives -- cache_sim's ``--probe-position head``.

Usage::

    .venv/Scripts/python.exe tools/cache_policy_study.py \
        --trace traces/mixed --out bench/results/cache_policy
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_sim import Trace, EXPERT_BYTES, N_ROUTED_PER_LAYER   # noqa: E402

T_LAYER_MS = 2.0
NVME_GBPS = 4.5
S_MS = EXPERT_BYTES / (NVME_GBPS * 1e9) * 1000.0
N_EXP = N_ROUTED_PER_LAYER
N_LAYERS = 40
COMPUTE_MS = N_LAYERS * T_LAYER_MS
CHUNK_BYTES = 4 * 1024 * 1024
CHUNK_MS = CHUNK_BYTES / (NVME_GBPS * 1e9) * 1000.0     # 0.932 ms per 4 MiB chunk
PF_QUEUE = 256         # outstanding prefetches a planner may hold (a queue
                       # entry is a descriptor, not a buffer; staleness is bounded
                       # by the per-token cancel, so this is effectively unlimited)

DEMAND, PREFETCH = 0, 1
INF = float("inf")


# --------------------------------------------------------------------------- #
# the drive
# --------------------------------------------------------------------------- #

class Drive:
    """One work-conserving server, priority queue, 4 MiB service quantum.

    The drive is issued in 4 MiB chunks at QD 8, so a demand request that
    arrives while a prefetch is being served waits at most one chunk, not one
    whole expert.  `quantum_ms` = 4 MiB / 4.5 GB/s = 0.932 ms models that;
    `quantum_ms = S_MS` degrades to a non-preemptible whole-expert server and is
    kept as the pessimistic sensitivity bound.
    """

    def __init__(self, s_ms=S_MS, quantum_ms=None):
        self.s = s_ms
        self.quantum = s_ms if quantum_ms is None else min(quantum_ms, s_ms)
        self.free = 0.0
        self.q = []
        self.seq = 0
        self.issue = {}
        self.cls = {}
        self.remain = {}
        self.inflight = set()
        self.n_demand = 0
        self.n_prefetch = 0
        self.served_demand = 0
        self.served_prefetch = 0
        self.cancelled = 0
        self.promoted = 0
        self.pf_work = {}
        self.busy = 0.0
        self.busy_prefetch = 0.0
        self.n_pending_pf = 0

    def submit(self, key, t, cls, urgency=0.0):
        if key in self.inflight:
            if cls < self.cls[key]:
                if self.cls[key] == PREFETCH:
                    self.n_pending_pf -= 1
                    self.promoted += 1
                self.cls[key] = cls
                heapq.heappush(self.q, (cls, urgency, self.seq, key))
                self.seq += 1
            return False
        self.inflight.add(key)
        self.issue[key] = t
        self.cls[key] = cls
        self.remain[key] = self.s
        if cls == DEMAND:
            self.n_demand += 1
        else:
            self.n_prefetch += 1
            self.n_pending_pf += 1
        heapq.heappush(self.q, (cls, urgency, self.seq, key))
        self.seq += 1
        return True

    def cancel_untouched_prefetches(self):
        """Drop queued prefetches that have not started; a planner re-plans per token."""
        keep = []
        for cls, urg, seq, key in self.q:
            if (key in self.inflight and self.cls[key] == PREFETCH
                    and abs(self.remain[key] - self.s) < 1e-9):
                self.inflight.discard(key)
                self.n_prefetch -= 1
                self.n_pending_pf -= 1
                self.cancelled += 1
                continue
            keep.append((cls, urg, seq, key))
        self.q = keep
        heapq.heapify(self.q)

    def _pop(self):
        while self.q:
            cls, urg, seq, key = heapq.heappop(self.q)
            if key not in self.inflight or self.cls[key] != cls:
                continue
            return cls, urg, seq, key
        return None

    def _quantum(self, on_done):
        """Serve one chunk of the best waiting request.  Returns (key, done_at)."""
        item = self._pop()
        if item is None:
            return None
        cls, urg, seq, key = item
        start = self.free if self.free > self.issue[key] else self.issue[key]
        w = self.remain[key]
        if w > self.quantum:
            w = self.quantum
        self.free = start + w
        self.busy += w
        if cls == PREFETCH:
            self.busy_prefetch += w
            self.pf_work[key] = self.pf_work.get(key, 0.0) + w
        self.remain[key] -= w
        if self.remain[key] > 1e-9:
            heapq.heappush(self.q, (cls, urg, seq, key))
            return key, None
        self.inflight.discard(key)
        if cls == PREFETCH:
            self.served_prefetch += 1
            self.n_pending_pf -= 1
        else:
            self.served_demand += 1
        on_done(key, self.free)
        return key, self.free

    def pump(self, now, on_done):
        while self.q and self.free < now:
            if self._quantum(on_done) is None:
                return

    def force(self, needed, on_done):
        want = set(k for k in needed if k in self.inflight)
        ready = 0.0
        while want:
            r = self._quantum(on_done)
            if r is None:
                break
            key, t = r
            if t is not None and key in want:
                want.discard(key)
                if t > ready:
                    ready = t
        return ready

    def depth(self):
        return len(self.inflight)

    def pf_pending(self):
        return self.n_pending_pf

    def take_pf_work(self, key):
        """Drive time already spent on `key` at prefetch priority, consumed once."""
        return self.pf_work.pop(key, 0.0)


# --------------------------------------------------------------------------- #
# caches -- all three expose contains / touch(k, nu) / admit(k, nu)
# --------------------------------------------------------------------------- #

class LRUCache:
    kind = "lru"

    def __init__(self, capacity):
        self.cap = max(0, capacity)
        self.d = OrderedDict()
        self.pinned = set()
        self.protect = set()

    def __contains__(self, k):
        return k in self.d

    def __len__(self):
        return len(self.d)

    def touch(self, k, nu=None):
        self.d.move_to_end(k)

    def _victim(self):
        if self.protect:
            for k in self.d:
                if k not in self.pinned and k not in self.protect:
                    return k
        for k in self.d:
            if k not in self.pinned:
                return k
        return None

    def admit(self, k, nu=None):
        if k in self.d:
            self.d.move_to_end(k)
            return None
        ev = None
        while len(self.d) >= self.cap:
            v = self._victim()
            if v is None:
                break
            del self.d[v]
            ev = v
        self.d[k] = None
        return ev


class QuotaCache:
    """LRU with a hard per-layer quota (cap / 40 slots each)."""

    kind = "quota"

    def __init__(self, capacity, n_layers=N_LAYERS):
        self.per = max(1, capacity // n_layers)
        self.pools = [OrderedDict() for _ in range(n_layers)]
        self.pinned = set()
        self.protect = set()
        self.n = 0

    def __contains__(self, k):
        return k in self.pools[k // N_EXP]

    def __len__(self):
        return self.n

    def touch(self, k, nu=None):
        self.pools[k // N_EXP].move_to_end(k)

    def admit(self, k, nu=None):
        p = self.pools[k // N_EXP]
        if k in p:
            p.move_to_end(k)
            return None
        ev = None
        while len(p) >= self.per:
            v = None
            for kk in p:
                if kk not in self.pinned:
                    v = kk
                    break
            if v is None:
                break
            del p[v]
            self.n -= 1
            ev = v
        p[k] = None
        self.n += 1
        return ev


class BeladyCache:
    """Clairvoyant: evict the resident key whose next demand use is furthest."""

    kind = "belady"

    def __init__(self, capacity):
        self.cap = max(0, capacity)
        self.res = {}
        self.heap = []
        self.pinned = set()
        self.protect = set()

    def __contains__(self, k):
        return k in self.res

    def __len__(self):
        return len(self.res)

    def _set(self, k, nu):
        self.res[k] = nu
        heapq.heappush(self.heap, (-nu, k))

    def touch(self, k, nu=None):
        if nu is not None:
            self._set(k, nu)

    def admit(self, k, nu=None):
        nu = INF if nu is None else nu
        if k in self.res:
            self._set(k, nu)
            return None
        ev = None
        while len(self.res) >= self.cap:
            v = None
            while self.heap:
                negnu, cand = self.heap[0]
                if cand not in self.res or self.res[cand] != -negnu:
                    heapq.heappop(self.heap)
                    continue
                if cand in self.pinned:
                    heapq.heappop(self.heap)
                    continue
                v = cand
                heapq.heappop(self.heap)
                break
            if v is None:
                break
            del self.res[v]
            ev = v
        self._set(k, nu)
        return ev


def make_cache(evict, capacity):
    if evict == "belady":
        return BeladyCache(capacity)
    if evict == "quota":
        return QuotaCache(capacity)
    return LRUCache(capacity)


# --------------------------------------------------------------------------- #
# next-use tables (Belady + clairvoyant prefetch)
# --------------------------------------------------------------------------- #

def next_use_tables(flat):
    """``per_slot[i, j]`` = flat request index of that key's next demand use.

    Also returns a CSR of every use index per key so a prefetch admitted at an
    arbitrary moment can be given an honest next-use value.
    """
    n, w = flat.shape
    lin = flat.ravel().astype(np.int64)
    order = np.argsort(lin, kind="stable")
    s = lin[order]
    nxt_sorted = np.full(len(s), INF)
    same = s[:-1] == s[1:]
    nxt_sorted[:-1] = np.where(same, order[1:].astype(np.float64), INF)
    per_slot = np.empty(len(lin), dtype=np.float64)
    per_slot[order] = nxt_sorted
    # CSR: uses of key k are order[start[k]:start[k+1]]
    nkeys = int(lin.max()) + 1
    counts = np.bincount(lin, minlength=nkeys)
    start = np.zeros(nkeys + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    return per_slot.reshape(n, w), order.astype(np.int64), start


class NextUse:
    def __init__(self, order, start):
        self.order = order
        self.start = start

    def after(self, key, idx):
        a, b = self.start[key], self.start[key + 1]
        if a == b:
            return INF
        seg = self.order[a:b]
        j = int(np.searchsorted(seg, idx, side="right"))
        return INF if j >= len(seg) else float(seg[j])


# --------------------------------------------------------------------------- #
# predictors
# --------------------------------------------------------------------------- #

class Predictor:
    name = "none"

    def token_start(self, ti):
        return ()

    def at_layer(self, ti, L, row):
        return ()


class Clairvoyant(Predictor):
    """Perfect knowledge of the token's remaining layers at layer 0."""

    def __init__(self, keys, lo=1, hi=N_LAYERS):
        self.keys = keys
        self.lo, self.hi = lo, hi
        self.name = "clairvoyant(L%d..%d)" % (lo, hi - 1)

    def token_start(self, ti):
        out = []
        blk = self.keys[ti]
        for L in range(self.lo, self.hi):
            for k in blk[L]:
                out.append((int(k), float(L)))
        return out


class NoisyClairvoyant(Predictor):
    """Clairvoyance degraded to a chosen precision -- what a predictor must reach.

    At token start it emits K candidates for every remaining layer, of which
    round(K * precision) are the layer's real experts (drawn from the six) and
    the rest are uniform random experts of that layer.  Byte cost is therefore
    fixed at K per layer and only the usefulness varies, which isolates the
    break-even precision under the bandwidth model.
    """

    def __init__(self, keys, precision, k=6, lo=1, seed=0):
        self.keys = keys
        self.p = precision
        self.k = k
        self.lo = lo
        self.rng = np.random.default_rng(seed)
        self.name = "noisy-clairvoyant(prec=%.2f,K=%d)" % (precision, k)

    def token_start(self, ti):
        m = int(round(self.k * self.p))
        out = []
        blk = self.keys[ti]
        for L in range(self.lo, N_LAYERS):
            base = L * N_EXP
            real = blk[L]
            sel = self.rng.permutation(6)[:min(m, 6)]
            for j in sel:
                out.append((int(real[j]), float(L)))
            for _ in range(self.k - len(sel)):
                out.append((base + int(self.rng.integers(N_EXP)), float(L)))
        return out


class Markov(Predictor):
    """P(expert at L+d | expert at L), trained on the train split."""

    def __init__(self, tables, leads, topk, min_p=0.0):
        self.tables = tables
        self.leads = leads
        self.topk = topk
        self.min_p = min_p
        self.name = "markov(d=%s,K=%d,p>=%.2f)" % (
            "-".join(str(d) for d in leads), topk, min_p)

    def at_layer(self, ti, L, row):
        out = []
        for d in self.leads:
            tgt = L + d
            if tgt >= N_LAYERS:
                continue
            T = self.tables[L].get(d)
            if T is None:
                continue
            sc = T[row % N_EXP].sum(0) / 6.0
            k = self.topk
            idx = np.argpartition(sc, -k)[-k:]
            base = tgt * N_EXP
            for e in idx:
                p = float(sc[e])
                if p >= self.min_p:
                    out.append((base + int(e), float(tgt)))
        return out


class TokenTable(Predictor):
    """Per (token id, layer) the most frequent experts, from the train split."""

    def __init__(self, index, ids, probs, topk, lo, min_p=0.0, tok_of=None):
        self.index = index
        self.ids = ids
        self.probs = probs
        self.topk = topk
        self.lo = lo
        self.min_p = min_p
        self.tok_of = tok_of
        self.name = "tokenid(K=%d,L>=%d,p>=%.2f)" % (topk, lo, min_p)

    def token_start(self, ti):
        r = self.index[self.tok_of[ti]]
        if r < 0:
            return ()
        out = []
        ids = self.ids[r]
        pr = self.probs[r]
        for L in range(self.lo, N_LAYERS):
            base = L * N_EXP
            for j in range(self.topk):
                if pr[L, j] >= self.min_p and ids[L, j] >= 0:
                    out.append((base + int(ids[L, j]), float(L)))
        return out


class WithinToken(Predictor):
    """Co-occurrence over the last H layers already routed in this token."""

    def __init__(self, tables, leads, topk, hist=3, min_p=0.0):
        self.tables = tables
        self.leads = leads
        self.topk = topk
        self.hist = hist
        self.min_p = min_p
        self.hold = []
        self.name = "cooc(H=%d,d=%s,K=%d)" % (
            hist, "-".join(str(d) for d in leads), topk)

    def at_layer(self, ti, L, row):
        self.hold.append((L, row % N_EXP))
        if len(self.hold) > self.hist:
            self.hold.pop(0)
        out = []
        for d in self.leads:
            tgt = L + d
            if tgt >= N_LAYERS:
                continue
            sc = None
            for (Ls, r) in self.hold:
                T = self.tables[Ls].get(tgt - Ls)
                if T is None:
                    continue
                v = T[r].sum(0) / 6.0
                sc = v if sc is None else sc + v
            if sc is None:
                continue
            k = self.topk
            idx = np.argpartition(sc, -k)[-k:]
            base = tgt * N_EXP
            for e in idx:
                out.append((base + int(e), float(tgt)))
        return out

    def reset(self):
        self.hold = []


class Lookahead(Predictor):
    """The trace's own hidden-state lookahead columns (design section 9.4)."""

    def __init__(self, preds, lead, topk):
        self.preds = preds
        self.lead = lead
        self.topk = topk
        self.name = "lookahead(d=%d,K=%d)" % (lead, topk)

    def at_layer(self, ti, L, row):
        tgt = L + self.lead
        if tgt >= N_LAYERS:
            return ()
        p = self.preds[ti, tgt]
        base = tgt * N_EXP
        out = []
        for j in range(self.topk):
            e = int(p[j])
            if e >= 0:
                out.append((base + e, float(tgt)))
        return out


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #

def replay(keys, capacity, *, evict="lru", predictor=None,
           warm=None, pinned_keys=None, protect_pred=None, label="",
           quantum_ms=None, nvme_gbps=NVME_GBPS, budget=PF_QUEUE):
    """Replay ``keys`` [N, 40, 6] under one policy.  Returns a metrics dict."""
    N = keys.shape[0]
    flat = np.ascontiguousarray(keys.reshape(N * N_LAYERS, 6))
    pinned_keys = set() if pinned_keys is None else set(int(x) for x in pinned_keys)
    eff_cap = capacity - len(pinned_keys)
    if eff_cap < 1:
        raise SystemExit("pinned set (%d) does not fit capacity %d"
                         % (len(pinned_keys), capacity))

    per_slot = nu = None
    if evict == "belady":
        per_slot, order, start = next_use_tables(flat)
        nu = NextUse(order, start)
    cache = make_cache(evict, eff_cap)

    if warm is not None and len(warm):
        wl = np.ascontiguousarray(warm.reshape(-1, 6))
        for r in range(wl.shape[0]):
            for k in wl[r]:
                k = int(k)
                if k in pinned_keys:
                    continue
                if k in cache:
                    cache.touch(k, INF)
                else:
                    cache.admit(k, INF)

    s_ms = EXPERT_BYTES / (nvme_gbps * 1e9) * 1000.0
    if quantum_ms is None:
        quantum_ms = CHUNK_BYTES / (nvme_gbps * 1e9) * 1000.0
    drive = Drive(s_ms=s_ms, quantum_ms=quantum_ms)
    pf_resident = set()
    st = dict(req=0, hit=0, late=0, miss=0, stall=0.0, pf_issue=0,
              pf_used=0, pf_wasted=0, pin_hit=0, pf_useful_ms=0.0)

    def on_done(key, t):
        ev = cache.admit(key, nu.after(key, cur_idx[0]) if nu else INF)
        if ev is not None and ev in pf_resident:
            pf_resident.discard(ev)
            st["pf_wasted"] += 1
        if drive.cls.get(key) == PREFETCH:
            pf_resident.add(key)

    cur_idx = [0]
    now = 0.0
    for ti in range(N):
        drive.cancel_untouched_prefetches()
        if hasattr(predictor, "reset"):
            predictor.reset()
        if protect_pred is not None:
            cache.protect = protect_pred(ti)
        if predictor is not None:
            for key, urg in predictor.token_start(ti):
                if key in pinned_keys or key in cache or key in drive.inflight:
                    continue
                if drive.pf_pending() >= budget:
                    break
                if drive.submit(key, now, PREFETCH, urg):
                    st["pf_issue"] += 1
        for L in range(N_LAYERS):
            i = ti * N_LAYERS + L
            cur_idx[0] = i
            row = flat[i]
            t_start = now
            drive.pump(t_start, on_done)

            need = []
            for j in range(6):
                k = int(row[j])
                st["req"] += 1
                if k in pinned_keys:
                    st["hit"] += 1
                    st["pin_hit"] += 1
                    continue
                if k in cache:
                    if k in pf_resident:
                        pf_resident.discard(k)
                        st["pf_used"] += 1
                        st["pf_useful_ms"] += drive.take_pf_work(k)
                    st["hit"] += 1
                    cache.touch(k, per_slot[i, j] if per_slot is not None else None)
                else:
                    st["miss"] += 1
                    if k in drive.inflight and drive.cls[k] == PREFETCH:
                        st["pf_used"] += 1
                        st["late"] += 1
                        st["pf_useful_ms"] += drive.take_pf_work(k)
                    drive.submit(k, t_start, DEMAND, 0.0)
                    need.append((k, j))

            if need:
                ready = drive.force([k for k, _ in need], on_done)
                if ready > t_start:
                    st["stall"] += ready - t_start
                    now = ready
                for k, j in need:
                    cache.admit(k, per_slot[i, j] if per_slot is not None else INF)
                    pf_resident.discard(k)

            cache.pinned = set(int(x) for x in row)
            now += T_LAYER_MS
            if predictor is not None:
                for key, urg in predictor.at_layer(ti, L, row):
                    if key in pinned_keys or key in cache or key in drive.inflight:
                        continue
                    if drive.pf_pending() >= budget:
                        break
                    if drive.submit(key, now, PREFETCH, urg):
                        st["pf_issue"] += 1
            cache.pinned = set()

    tok = float(N)
    req = max(1, st["req"])
    # bytes are billed from work the drive actually performed, so a prefetch that
    # was cancelled or never reached the head of the queue costs nothing.
    gb_s = nvme_gbps
    total_mb = drive.busy / 1000.0 * gb_s * 1e9 / 1e6
    pf_mb = drive.busy_prefetch / 1000.0 * gb_s * 1e9 / 1e6
    used_pf_mb = st["pf_useful_ms"] / 1000.0 * gb_s * 1e9 / 1e6
    fetched = drive.served_demand + drive.served_prefetch
    stall = st["stall"] / tok
    wall = COMPUTE_MS + stall                     # ms of wall clock per token
    busy = drive.busy / tok                       # ms of drive time per token
    # During a stall the drive is busy by construction (it is serving exactly the
    # requests the GPU is waiting for, back to back), so the drive work that
    # overlapped compute is busy - stall.  What is left of the token is idle drive.
    overlap = max(0.0, busy - stall)
    useful_frac = ((total_mb - pf_mb + used_pf_mb) / total_mb) if total_mb else 0.0
    return {
        "label": label,
        "capacity": capacity,
        "quantum_ms": drive.quantum,
        "nvme_gbps": nvme_gbps,
        "pinned_slots": len(pinned_keys),
        "tokens": N,
        # hit_rate is "resident when asked for", i.e. the GPU did not wait.  For a
        # prefetching policy that includes slots a prefetch had already filled, so
        # cold_hit_rate isolates what residency alone (no prefetch) delivered.
        "hit_rate": st["hit"] / req,
        "cold_hit_rate": (st["hit"] - (st["pf_used"] - st["late"])) / req,
        "pin_hit_rate": st["pin_hit"] / req,
        "stall_ms_per_token": stall,
        "wall_ms_per_token": wall,
        "experts_per_token": fetched / tok,
        "demand_per_token": drive.served_demand / tok,
        "prefetch_per_token": drive.served_prefetch / tok,
        "prefetch_issued_per_token": drive.n_prefetch / tok,
        "mb_per_token": total_mb / tok,
        "useful_mb_per_token": (total_mb - pf_mb + used_pf_mb) / tok,
        "wasted_mb_per_token": (pf_mb - used_pf_mb) / tok,
        "prefetch_cancelled": drive.cancelled,
        "prefetch_served": drive.served_prefetch,
        "prefetch_precision": (st["pf_used"] / drive.served_prefetch) if drive.served_prefetch else 0.0,
        "late_prefetch": st["late"],
        # "use more of what the machine has"
        "drive_busy_ms_per_token": busy,
        "gb_s_needed_for_zero_stall": total_mb * 1e6 / tok / (COMPUTE_MS / 1000.0) / 1e9,
        "drive_idle_ms_per_token": max(0.0, wall - busy),
        "drive_util": busy / wall,
        "gpu_stall_frac": stall / wall,
        "overlap_ms_per_token": overlap,            # drive work hidden under compute
        "overlap_frac_of_compute": overlap / COMPUTE_MS,
        "useful_overlap_ms_per_token": overlap * useful_frac,
        "tok_s": 1000.0 / wall,
    }


def per_layer_hits(keys, capacity, evict="lru", pinned_keys=None, warm=None):
    """Hit rate per layer for a demand-only replay (no timing) -- cheap."""
    N = keys.shape[0]
    flat = np.ascontiguousarray(keys.reshape(N * N_LAYERS, 6))
    pinned_keys = set() if pinned_keys is None else set(int(x) for x in pinned_keys)
    cache = make_cache(evict, capacity - len(pinned_keys))
    if warm is not None and len(warm):
        for r in np.ascontiguousarray(warm.reshape(-1, 6)):
            for k in r:
                k = int(k)
                if k not in pinned_keys:
                    (cache.touch if k in cache else cache.admit)(k, INF)
    hit = np.zeros(N_LAYERS, dtype=np.int64)
    tot = np.zeros(N_LAYERS, dtype=np.int64)
    for i in range(flat.shape[0]):
        L = i % N_LAYERS
        for k in flat[i]:
            k = int(k)
            tot[L] += 1
            if k in pinned_keys or k in cache:
                hit[L] += 1
                if k not in pinned_keys:
                    cache.touch(k, INF)
            else:
                cache.admit(k, INF)
    return hit / np.maximum(1, tot)


# --------------------------------------------------------------------------- #
# training the predictors
# --------------------------------------------------------------------------- #

def build_markov(top6_train, leads):
    """tables[L][d] = row-normalised (384, 384) float32, P(f at L+d | e at L)."""
    tables = [dict() for _ in range(N_LAYERS)]
    for L in range(N_LAYERS):
        a = top6_train[:, L, :].astype(np.int64)
        for d in leads:
            if L + d >= N_LAYERS:
                continue
            b = top6_train[:, L + d, :].astype(np.int64)
            idx = (a[:, :, None] * N_EXP + b[:, None, :]).ravel()
            M = np.bincount(idx, minlength=N_EXP * N_EXP).reshape(N_EXP, N_EXP)
            M = M.astype(np.float32)
            s = M.sum(1, keepdims=True)
            s[s == 0] = 1.0
            tables[L][d] = M / s
    return tables


def build_token_table(tok_train, top6_train, n_tok=8192, store=8):
    """Per (token id, layer) the most frequent experts on the train split."""
    vocab = int(tok_train.max()) + 1
    freq = np.bincount(tok_train, minlength=vocab)
    keep = np.argsort(freq)[::-1][:n_tok]
    keep = keep[freq[keep] >= 4]
    index = np.full(vocab, -1, dtype=np.int32)
    index[keep] = np.arange(len(keep), dtype=np.int32)
    ids = np.full((len(keep), N_LAYERS, store), -1, dtype=np.int32)
    probs = np.zeros((len(keep), N_LAYERS, store), dtype=np.float32)
    rows = index[tok_train]
    for r in range(len(keep)):
        sel = np.nonzero(rows == r)[0]
        n = len(sel)
        blk = top6_train[sel]                 # (n, 40, 6)
        for L in range(N_LAYERS):
            c = np.bincount(blk[:, L, :].ravel().astype(np.int64), minlength=N_EXP)
            top = np.argpartition(c, -store)[-store:]
            top = top[np.argsort(c[top])[::-1]]
            ids[r, L] = top
            probs[r, L] = c[top] / float(n)
    return index, ids, probs, keep, freq


def predictor_quality(pred_ids, actual6, k):
    """recall@K = |pred_K & actual6| / 6 and precision = same / K, per layer."""
    rec = np.zeros(N_LAYERS)
    pre = np.zeros(N_LAYERS)
    cnt = np.zeros(N_LAYERS)
    for L in range(N_LAYERS):
        p = pred_ids[:, L, :k]
        a = actual6[:, L, :]
        ok = np.zeros(len(p))
        for j in range(k):
            ok += (p[:, j:j + 1] == a).any(1) & (p[:, j] >= 0)
        valid = (p >= 0).any(1)
        if valid.sum() == 0:
            continue
        rec[L] = ok[valid].mean() / 6.0
        pre[L] = ok[valid].mean() / k
        cnt[L] = valid.sum()
    return rec, pre, cnt


def markov_predictions(tables, top6, lead, k):
    """Top-k prediction of layer L from layer L-lead, for every token."""
    N = top6.shape[0]
    out = np.full((N, N_LAYERS, k), -1, dtype=np.int32)
    for L in range(lead, N_LAYERS):
        T = tables[L - lead].get(lead)
        if T is None:
            continue
        src = top6[:, L - lead, :].astype(np.int64)      # (N, 6)
        sc = T[src].sum(1)                               # (N, 384)
        idx = np.argpartition(sc, -k, axis=1)[:, -k:]
        ord_ = np.take_along_axis(sc, idx, 1).argsort(axis=1)[:, ::-1]
        out[:, L, :] = np.take_along_axis(idx, ord_, 1)
    return out


def token_predictions(index, ids, tok, k):
    N = len(tok)
    out = np.full((N, N_LAYERS, k), -1, dtype=np.int32)
    r = index[tok]
    have = r >= 0
    out[have] = ids[r[have]][:, :, :k]
    return out


# --------------------------------------------------------------------------- #
# static layer / hot-set selections
# --------------------------------------------------------------------------- #

def layer_keys(layers):
    out = set()
    for L in layers:
        base = L * N_EXP
        out.update(range(base, base + N_EXP))
    return out


def hot_keys_per_layer(top6_train, k_per_layer):
    out = set()
    for L in range(N_LAYERS):
        c = np.bincount(top6_train[:, L, :].ravel().astype(np.int64), minlength=N_EXP)
        k = k_per_layer[L] if hasattr(k_per_layer, "__len__") else k_per_layer
        if k <= 0:
            continue
        top = np.argpartition(c, -k)[-k:]
        base = L * N_EXP
        out.update(base + int(e) for e in top)
    return out


def layer_skew(top6_train):
    """Fraction of a layer's requests covered by its 10% hottest experts."""
    sk = np.zeros(N_LAYERS)
    for L in range(N_LAYERS):
        c = np.bincount(top6_train[:, L, :].ravel().astype(np.int64), minlength=N_EXP)
        c = np.sort(c)[::-1]
        sk[L] = c[:N_EXP // 10].sum() / max(1, c.sum())
    return sk


def streaming_break_even(max_n=8, layer_bytes=N_EXP * EXPERT_BYTES,
                         window_ms=COMPUTE_MS):
    """Bandwidth needed to stream N whole layers inside one token's compute."""
    rows = []
    for n in range(1, max_n + 1):
        b = n * layer_bytes
        rows.append({
            "layers": n,
            "bytes_gb": b / 1e9,
            "gbps_for_80ms": b / (window_ms / 1000.0) / 1e9,
            "ms_at_4.5gbps": b / 4.5e9 * 1000.0,
            "ms_at_9gbps": b / 9e9 * 1000.0,
            "ms_at_30gbps": b / 30e9 * 1000.0,
        })
    return rows


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def load(trace_dir, tokids_path):
    t = Trace.load([trace_dir], "mean", quiet=False)
    N = t.n_tokens
    keys = t.keys6.reshape(N, N_LAYERS, 6).astype(np.int32)
    top6 = t.top6.reshape(N, N_LAYERS, 6).astype(np.int32)
    prompt = t.prompt.reshape(N, N_LAYERS)[:, 0].astype(np.int32)
    preds = {d: v.reshape(N, N_LAYERS, 16).astype(np.int32)
             for d, v in t.preds.items()}
    z = np.load(tokids_path, allow_pickle=True)
    tok = z["ids"].astype(np.int64)
    if len(tok) != N:
        raise SystemExit("token id file has %d rows, trace has %d" % (len(tok), N))
    return keys, top6, prompt, preds, tok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="traces/mixed")
    ap.add_argument("--token-ids", default="bench/results/cache_policy/token_ids.npz")
    ap.add_argument("--out", default="bench/results/cache_policy")
    ap.add_argument("--capacities", default="4500,5711,8000")
    ap.add_argument("--pin-capacities", default="4500,5500,5711,8000")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--quantum-ms", type=float, default=CHUNK_MS,
                    help="drive service quantum; %.3f ms = one 4 MiB chunk, "
                         "4.178 = non-preemptible whole expert" % CHUNK_MS)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    caps = [int(x) for x in args.capacities.split(",")]
    pin_caps = [int(x) for x in args.pin_capacities.split(",")]

    t0 = time.time()
    keys, top6, prompt, preds, tok = load(args.trace, args.token_ids)
    N = keys.shape[0]
    n_prompts = int(prompt.max()) + 1
    split_p = int(round(n_prompts * args.train_frac))
    train = prompt < split_p
    test = ~train
    ntr, nte = int(train.sum()), int(test.sum())
    print("split: %d prompts train (%d tok) / %d prompts test (%d tok)"
          % (split_p, ntr, n_prompts - split_p, nte))

    keys_tr, keys_te = keys[train], keys[test]
    top6_tr, top6_te = top6[train], top6[test]
    tok_tr, tok_te = tok[train], tok[test]
    warm = keys_tr[-2000:]

    results = {"model": {
        "t_layer_ms": T_LAYER_MS, "compute_ms_per_token": COMPUTE_MS,
        "nvme_gbps": NVME_GBPS, "expert_bytes": EXPERT_BYTES,
        "service_ms_per_expert": S_MS, "chunk_ms": CHUNK_MS,
        "quantum_ms": args.quantum_ms, "tokens_total": N,
        "tokens_train": ntr, "tokens_test": nte, "prompts": n_prompts,
        "train_prompts": split_p,
    }, "runs": [], "per_layer": {}, "quality": {}, "streaming": {}}

    def run(label, **kw):
        s = time.time()
        kw.setdefault("quantum_ms", args.quantum_ms)
        r = replay(keys_te, warm=warm, label=label, **kw)
        r["seconds"] = round(time.time() - s, 1)
        results["runs"].append(r)
        print("  %-46s C=%-5d hit %.4f stall %6.1f util %.3f idle %5.1f "
              "waste %5.0fMB tok/s %5.2f (%.0fs)"
              % (label, r["capacity"], r["hit_rate"], r["stall_ms_per_token"],
                 r["drive_util"], r["drive_idle_ms_per_token"],
                 r["wasted_mb_per_token"], r["tok_s"], r["seconds"]))
        return r

    # ---------------- 0. sanity: full-trace LRU demand-only ---------------- #
    if args.stage in ("all", "sanity"):
        print("\n[0] full-trace LRU demand-only (sanity vs the 0.905 figure)")
        for C in (4500, 5711):
            s = time.time()
            r = replay(keys, C, evict="lru", label="full-trace lru C=%d" % C,
                       quantum_ms=args.quantum_ms)
            r["seconds"] = round(time.time() - s, 1)
            r["scope"] = "full-trace"
            results["runs"].append(r)
            print("  full-trace LRU C=%-5d hit %.4f  stall %.1f ms  tok/s %.2f"
                  % (C, r["hit_rate"], r["stall_ms_per_token"], r["tok_s"]))

    # ---------------- 1. baselines + clairvoyant ceilings ------------------ #
    if args.stage in ("all", "base"):
        print("\n[1] baselines and ceilings (test split)")
        for C in caps:
            run("LRU demand-only", capacity=C, evict="lru")
            run("LRU+quota demand-only", capacity=C, evict="quota")
            run("Belady demand-only (capacity ceiling)", capacity=C, evict="belady")
            run("clairvoyant prefetch + LRU", capacity=C, evict="lru",
                predictor=Clairvoyant(keys_te))
            run("clairvoyant prefetch + Belady (streaming ceiling)", capacity=C,
                evict="belady", predictor=Clairvoyant(keys_te))
            for n in (2, 4, 8):
                run("clairvoyant prefetch last %d layers only" % n, capacity=C,
                    evict="lru", predictor=Clairvoyant(keys_te, N_LAYERS - n, N_LAYERS))

    # ---------------- 2. predictor quality --------------------------------- #
    print("\n[2] training predictors")
    leads = [1, 2, 3, 4]
    tm = time.time()
    tables = build_markov(top6_tr, leads)
    print("  markov tables: %.0fs" % (time.time() - tm))
    tm = time.time()
    tindex, tids, tprobs, tkeep, tfreq = build_token_table(tok_tr, top6_tr)
    print("  token table: %d tokens kept, %.0fs" % (len(tkeep), time.time() - tm))

    if args.stage in ("all", "quality", "base"):
        q = {}
        for d in leads:
            for k in (6, 8, 12, 16):
                p = markov_predictions(tables, top6_te, d, k)
                rec, pre, cnt = predictor_quality(p, top6_te, k)
                q["markov_d%d_K%d" % (d, k)] = {
                    "recall_per_layer": rec.round(4).tolist(),
                    "precision_per_layer": pre.round(4).tolist(),
                    "recall_mean": float(rec[d:].mean()),
                    "recall_last4": float(rec[-4:].mean()),
                    "precision_mean": float(pre[d:].mean()),
                }
        for k in (6, 8, 16):
            p = token_predictions(tindex, tids, tok_te, min(k, tids.shape[2]))
            rec, pre, cnt = predictor_quality(p, top6_te, min(k, tids.shape[2]))
            q["tokenid_K%d" % k] = {
                "recall_per_layer": rec.round(4).tolist(),
                "precision_per_layer": pre.round(4).tolist(),
                "recall_mean": float(rec.mean()),
                "recall_last4": float(rec[-4:].mean()),
                "precision_mean": float(pre.mean()),
                "coverage": float((tindex[tok_te] >= 0).mean()),
            }
        for d in sorted(preds):
            pa = preds[d][test]
            for k in (6, 8, 16):
                rec, pre, cnt = predictor_quality(pa, top6_te, k)
                q["lookahead_d%d_K%d" % (d, k)] = {
                    "recall_per_layer": rec.round(4).tolist(),
                    "precision_per_layer": pre.round(4).tolist(),
                    "recall_mean": float(rec[d:].mean()),
                    "recall_last4": float(rec[-4:].mean()),
                    "precision_mean": float(pre[d:].mean()),
                }
        results["quality"] = q
        print("  markov d=1 K=8 recall(mean/last4): %.3f / %.3f"
              % (q["markov_d1_K8"]["recall_mean"], q["markov_d1_K8"]["recall_last4"]))
        print("  markov d=4 K=8 recall(mean/last4): %.3f / %.3f"
              % (q["markov_d4_K8"]["recall_mean"], q["markov_d4_K8"]["recall_last4"]))
        print("  tokenid K=8   recall(mean/last4): %.3f / %.3f  coverage %.3f"
              % (q["tokenid_K8"]["recall_mean"], q["tokenid_K8"]["recall_last4"],
                 q["tokenid_K8"]["coverage"]))

    # ---------------- 3. predictive prefetch -------------------------------- #
    if args.stage in ("all", "pred"):
        print("\n[3] predictive prefetch (test split)")
        for C in caps:
            for k in (2, 4):
                run("markov d=1 K=%d" % k, capacity=C, evict="lru",
                    predictor=Markov(tables, [1], k))
            run("markov d=1-2 K=2", capacity=C, evict="lru",
                predictor=Markov(tables, [1, 2], 2))
            run("markov d=1-4 K=2", capacity=C, evict="lru",
                predictor=Markov(tables, [1, 2, 3, 4], 2))
            run("markov d=1 K=4 p>=0.05", capacity=C, evict="lru",
                predictor=Markov(tables, [1], 4, 0.05))
            run("markov d=1 K=4 p>=0.05 budget 4", capacity=C, evict="lru",
                predictor=Markov(tables, [1], 4, 0.05), budget=4)
            run("cooc H=3 d=1-2 K=2", capacity=C, evict="lru",
                predictor=WithinToken(tables, [1, 2], 2, 3))
            run("tokenid K=4 all layers p>=0.10", capacity=C, evict="lru",
                predictor=TokenTable(tindex, tids, tprobs, 4, 0, 0.10, tok_te))
            run("tokenid K=2 layers>=32 p>=0.10", capacity=C, evict="lru",
                predictor=TokenTable(tindex, tids, tprobs, 2, 32, 0.10, tok_te))
            for d in (2, 3, 4, 6, 8):
                if d not in preds:
                    continue
                for k in (4, 6, 8):
                    run("lookahead d=%d K=%d" % (d, k), capacity=C, evict="lru",
                        predictor=Lookahead(preds[d][test], d, k))
            if 3 in preds:
                for b in (2, 4, 8):
                    run("lookahead d=3 K=6 budget %d" % b, capacity=C, evict="lru",
                        predictor=Lookahead(preds[3][test], 3, 6), budget=b)
                run("lookahead d=3 K=6 + Belady evict", capacity=C, evict="belady",
                    predictor=Lookahead(preds[3][test], 3, 6))

    # ---------------- 4. prediction-aware eviction -------------------------- #
    if args.stage in ("all", "evict"):
        print("\n[4] prediction-aware eviction (no prefetch)")
        pred_k8 = token_predictions(tindex, tids, tok_te, 8)

        def protect_of(ti):
            p = pred_k8[ti]
            out = set()
            for L in range(N_LAYERS):
                base = L * N_EXP
                for e in p[L]:
                    if e >= 0:
                        out.add(base + int(e))
            return out

        for C in caps:
            run("LRU + tokenid-protected eviction", capacity=C, evict="lru",
                protect_pred=protect_of)

    # ---------------- 5. (g)(i) layer residency ----------------------------- #
    if args.stage in ("all", "layer"):
        print("\n[5] whole-layer pinning and static hot-set pinning")
        base_hits = per_layer_hits(keys_te, 4500, "lru", warm=warm)
        results["per_layer"]["lru_C4500"] = base_hits.round(4).tolist()
        skew = layer_skew(top6_tr)
        results["per_layer"]["skew_top10pct_train"] = skew.round(4).tolist()
        worst = list(np.argsort(base_hits)[:8])
        most_skewed = list(np.argsort(skew)[::-1][:8])
        results["per_layer"]["worst_lru_layers"] = [int(x) for x in worst]
        results["per_layer"]["most_skewed_layers"] = [int(x) for x in most_skewed]
        print("  worst-LRU layers:", [int(x) for x in worst])
        print("  most-skewed layers:", [int(x) for x in most_skewed])

        for C in pin_caps:
            run("LRU demand-only", capacity=C, evict="lru")
            for n in (1, 2, 4, 6, 8):
                if 384 * n >= C - 200:
                    continue
                run("pin last %d layers + LRU" % n, capacity=C, evict="lru",
                    pinned_keys=layer_keys(range(N_LAYERS - n, N_LAYERS)))
                run("pin %d worst-LRU layers + LRU" % n, capacity=C, evict="lru",
                    pinned_keys=layer_keys([int(x) for x in worst[:n]]))
                run("pin %d most-skewed layers + LRU" % n, capacity=C, evict="lru",
                    pinned_keys=layer_keys([int(x) for x in most_skewed[:n]]))
            for kpl in (8, 16, 32, 64):
                run("pin hot-%d per layer + LRU" % kpl, capacity=C, evict="lru",
                    pinned_keys=hot_keys_per_layer(top6_tr, kpl))

        for n in (0, 4):
            pk = layer_keys(range(N_LAYERS - n, N_LAYERS)) if n else None
            h = per_layer_hits(keys_te, 5711, "lru", pinned_keys=pk, warm=warm)
            results["per_layer"]["pin_last%d_C5711" % n] = h.round(4).tolist()

    # ---------------- 6. (h) whole-layer streaming break-even ---------------- #
    # ---------------- 7. sensitivity: non-preemptible drive ----------------- #
    if args.stage == "all":
        print("\n[7] sensitivity: non-preemptible whole-expert service "
              "(quantum = %.2f ms)" % S_MS)
        for C in caps:
            run("[np] LRU demand-only", capacity=C, evict="lru", quantum_ms=S_MS)
            run("[np] clairvoyant prefetch + LRU", capacity=C, evict="lru",
                predictor=Clairvoyant(keys_te), quantum_ms=S_MS)
            if 3 in preds:
                run("[np] lookahead d=3 K=4", capacity=C, evict="lru",
                    predictor=Lookahead(preds[3][test], 3, 4), quantum_ms=S_MS)

    # ---------------- 8. break-even precision and bandwidth ----------------- #
    if args.stage in ("all", "sweep"):
        print("\n[8] how good would a predictor have to be? "
              "(K=6/layer, full lead)")
        for C in caps:
            for prec in (0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                run("noisy-clairvoyant prec=%.1f K=6" % prec, capacity=C,
                    evict="lru", predictor=NoisyClairvoyant(keys_te, prec, 6))
            for prec in (0.6, 0.8, 1.0):
                run("noisy-clairvoyant prec=%.1f K=3" % prec, capacity=C,
                    evict="lru", predictor=NoisyClairvoyant(keys_te, prec, 3))

        print("\n[9] bandwidth sweep, demand-only LRU "
              "(how much drive would it take?)")
        for C in caps:
            for bw in (4.5, 6.0, 7.0, 9.0, 12.0, 14.0):
                r = replay(keys_te, C, evict="lru", warm=warm, nvme_gbps=bw,
                           label="LRU demand-only @ %.1f GB/s" % bw)
                results["runs"].append(r)
                print("  C=%-5d %5.1f GB/s  stall %6.1f ms  tok/s %5.2f  "
                      "(needs %.2f GB/s for zero stall)"
                      % (C, bw, r["stall_ms_per_token"], r["tok_s"],
                         r["gb_s_needed_for_zero_stall"]))

    results["streaming"] = {
        "layer_bytes": N_EXP * EXPERT_BYTES,
        "rows": streaming_break_even(8),
    }

    path = os.path.join(args.out, "policy_study.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("\nwrote %s  (%.0f s total)" % (path, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
