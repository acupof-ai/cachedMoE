#!/usr/bin/env python3
"""Track F6: idle-window prefetch, re-simulated under the CURRENT (post-Q2) regime.

Offline, CPU only.  No engine, no GPU process.  Reads ``traces/mixed`` directly
(the six routed experts a token asks each layer for, plus the trace's own
hidden-state lookahead columns) and replays decode under the timing model that
``docs/p4_p0_queue.md`` measured after Track Q2 landed, not the saturated-drive
model of ``docs/p4_cache_policy.md`` section 1.

What changed versus Track X's model
-----------------------------------
* the drive ceiling is **4.6 GB/s** (D:, ``p4_p0_queue.md`` section 9.1), not
  the 4.5 that came from a C: measurement;
* a layer's burst is not free to start: the **first P0 of a layer costs 1.3 ms**
  before any byte lands (``p4_p0_queue.md`` section 12: first-of-burst 1.27 ms
  after the submit thread pool).  That 1.3 ms x ~13 layers/token is ~17 ms of
  per-token stall that is *latency, not bandwidth* -- and a prefetch that
  removes a layer's last miss removes the ramp with it.  Track X's model had no
  ramp, so it could not see this term at all;
* the per-token wall clock is the measured decode breakdown, not 80 + stall.

Timing model (every constant has a citation)
--------------------------------------------
Per token, 40 layers.  For layer L::

    [ compute c = 2.0 ms ]  -> drive is free for prefetch in this window
    [ demand misses of L ]  -> GPU stalls until all of them land

* one expert is 18,800,640 B; at 4.6 GB/s that is ``S_MS`` = 4.087 ms of drive;
* the drive is ONE FIFO server with demand ahead of prefetch and **chunk
  granularity preemption**: a demand that arrives while a prefetch chunk is in
  service waits for that chunk (4 MiB / 4.6 GB/s = 0.911 ms), not for the whole
  expert.  This is the standing head-of-line cost of prefetching;
* a layer with at least one *fresh* miss pays ``RAMP_MS`` = 1.3 ms once.  A miss
  whose prefetch is already in flight does not re-pay it;
* a prefetch still in flight when its expert is demanded is **promoted**: the
  demand waits only for the bytes that are left (partial hide);
* a prefetch that lands takes a slot and evicts under LRU, exactly like a
  demand fill (``cache_sim``'s ``--probe-position head``);
* wrong prefetches cost drive time only where the drive would otherwise have
  been idle, plus the head-of-line chunk, plus the slot they stole.  All three
  fall out of the simulation; none of them is a fudge factor.

Per-token wall clock (decode-only, measured, ``bench/results/q2/d*_default``
``events.jsonl`` ``per_token_ms``)::

    attn 43.5 + moe_gpu 42.1 + moe_host 1.4 + engram 4.4 + tail 6.0 + other 5.3

The model's 40 x 2.0 ms = 80 ms stands in for attn + moe_gpu (measured 85.6);
``FIXED_MS`` = 17.0 ms stands in for moe_host + engram + tail + other (measured
17.0).  So ``tok/s = 1000 / (80 + 17 + stall)``, and the baseline has to land on
the measured stall ~101 ms / ~5.0 tok/s or the model is wrong.

Usage::

    .venv/Scripts/python.exe tools/idle_prefetch_sim.py \
        --trace traces/mixed --out bench/results/f6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from collections import OrderedDict, deque

import numpy as np

N_LAYERS = 40
N_EXP = 384
TOP_K = 6
EXPERT_BYTES = 18_800_640

NVME_GBPS = 4.6                                     # D:, p4_p0_queue.md section 9.1
S_MS = EXPERT_BYTES / (NVME_GBPS * 1e9) * 1000.0    # 4.087 ms per expert
CHUNK_BYTES = 4 * 1024 * 1024
CHUNK_MS = CHUNK_BYTES / (NVME_GBPS * 1e9) * 1000.0  # 0.911 ms
RAMP_MS = 1.3                                        # first-of-layer P0 latency
T_LAYER_MS = 2.0
COMPUTE_MS = N_LAYERS * T_LAYER_MS                   # 80.0
FIXED_MS = 17.0                                      # moe_host + engram + tail + other
PF_BUDGET = 64                                       # outstanding prefetch descriptors


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #

def load_trace(path: str, depths=(1, 2, 3, 4), approx="mean", quiet=False):
    """-> top6 [N, 40, 6] int32 (expert id within layer), preds {d: [N, 40, 16]}."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(path, "route_layer*.parquet")))
    if not files:
        raise SystemExit("no route_layer*.parquet under " + path)
    cols = (["prompt_id", "pos", "layer", "top6_ids", "top16_ids"]
            + [f"pred_d{d}_{approx}" for d in depths])

    top6 = {}
    top16 = {}
    preds = {d: {} for d in depths}
    prompt = pos = None
    for f in files:
        t = pq.read_table(f, columns=cols)
        L = int(np.asarray(t.column("layer"))[0])
        if prompt is None:
            prompt = np.asarray(t.column("prompt_id")).astype(np.int32)
            pos = np.asarray(t.column("pos")).astype(np.int32)
        top6[L] = (np.asarray(t.column("top6_ids").combine_chunks().flatten())
                   .reshape(-1, TOP_K).astype(np.int32))
        top16[L] = (np.asarray(t.column("top16_ids").combine_chunks().flatten())
                    .reshape(-1, 16).astype(np.int32))
        for d in depths:
            c = t.column(f"pred_d{d}_{approx}")
            if c.null_count == len(c):
                preds[d][L] = None
            else:
                preds[d][L] = (np.asarray(c.combine_chunks().flatten())
                               .reshape(-1, 16).astype(np.int32))
        if not quiet:
            print(f"  layer {L:02d} loaded", end="\r", flush=True)

    n = top6[0].shape[0]
    a6 = np.stack([top6[L] for L in range(N_LAYERS)], axis=1)          # [N, 40, 6]
    a16 = np.stack([top16[L] for L in range(N_LAYERS)], axis=1)        # [N, 40, 16]
    ap = {}
    for d in depths:
        arr = np.full((n, N_LAYERS, 16), -1, dtype=np.int32)
        for L in range(N_LAYERS):
            if preds[d][L] is not None:
                arr[:, L] = preds[d][L]
        ap[d] = arr
    if not quiet:
        print(f"  trace: {n} tokens x {N_LAYERS} layers                    ")
    return a6, a16, ap, prompt, pos


def split_by_prompt(prompt: np.ndarray, train_frac=0.7):
    ids = np.unique(prompt)
    cut = int(round(len(ids) * train_frac))
    tr = np.isin(prompt, ids[:cut])
    te = np.isin(prompt, ids[cut:])
    return tr, te


# --------------------------------------------------------------------------- #
# predictors -- all return a list of expert ids within the target layer
# --------------------------------------------------------------------------- #

class TracePredictor:
    """The trace's own hidden-state lookahead: layer ``tgt``'s routing predicted
    from the residual snapshot ``d`` layers earlier (tools/route_trace.py)."""

    def __init__(self, preds_d, k=TOP_K, name=""):
        self.p = preds_d
        self.k = k
        self.name = name

    def predict(self, ti, tgt):
        row = self.p[ti, tgt]
        return [int(e) for e in row[:self.k] if e >= 0]


class SyntheticPredictor:
    """A predictor of stated per-expert precision p, K = 6 candidates a layer.

    Of the six issued, ``round(6 p)`` are members of the layer's true top-6 and
    the rest are uniform random experts of that layer.  With K = 6 against a
    true set of 6, precision = recall = p, so one number parametrises both.
    This is the same degradation Track X used (p4_cache_policy.md section 7) so
    the two studies' precision axes are comparable.
    """

    def __init__(self, top6, p, k=TOP_K, seed=1234, name=""):
        self.top6 = top6
        self.p = p
        self.k = k
        self.lo = int(k * p)                 # floor
        self.frac = k * p - self.lo          # Bernoulli remainder, so p = 0.77
        self.rng = random.Random(seed)       # and p = 0.90 are not the same cell
        self.name = name

    def predict(self, ti, tgt):
        true = self.top6[ti, tgt]
        n_true = self.lo + (1 if self.rng.random() < self.frac else 0)
        out = [int(x) for x in true[:n_true]]
        s = set(out)
        while len(out) < self.k:
            e = self.rng.randrange(N_EXP)
            if e not in s:
                s.add(e)
                out.append(e)
        return out


class NearMissPredictor(SyntheticPredictor):
    """Same precision p, but the WRONG candidates are near misses.

    The uniform-random wrong candidate of ``SyntheticPredictor`` is cache-cold by
    construction: a random one of 384 experts is almost never resident, so every
    error costs a full 18.8 MB fetch *and* a hot slot.  A real predictor's errors
    are not like that -- they are the experts that *nearly* won the gate, which
    are exactly the ones LRU is already holding, so most of them cost nothing.
    This variant draws the wrong candidates from the layer's own top-16 (ranks
    6..15) to isolate the error *distribution* from the error *rate*.
    """

    def __init__(self, top6, top16, p, k=TOP_K, seed=1234, name=""):
        super().__init__(top6, p, k=k, seed=seed, name=name)
        self.top16 = top16

    def predict(self, ti, tgt):
        true = self.top6[ti, tgt]
        n_true = self.lo + (1 if self.rng.random() < self.frac else 0)
        out = [int(x) for x in true[:n_true]]
        s = set(out)
        pool = [int(x) for x in self.top16[ti, tgt][TOP_K:]]
        self.rng.shuffle(pool)
        for e in pool:
            if len(out) >= self.k:
                break
            if e not in s:
                s.add(e)
                out.append(e)
        while len(out) < self.k:
            e = self.rng.randrange(N_EXP)
            if e not in s:
                s.add(e)
                out.append(e)
        return out


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #

def replay(top6, *, capacity=5100, predictor=None, lead=0, warm_rows=None,
           budget=TOP_K, label=""):
    """Replay ``top6`` [N, 40, 6].  Returns a metrics dict.

    The cache is a plain LRU over global keys ``layer * 384 + expert``.
    """
    n = top6.shape[0]
    cache: "OrderedDict[int, int]" = OrderedDict()

    def touch(key):
        cache.move_to_end(key)

    def insert(key):
        if key in cache:
            cache.move_to_end(key)
            return
        cache[key] = 1
        if len(cache) > capacity:
            cache.popitem(last=False)

    if warm_rows is not None:
        for ti in range(warm_rows.shape[0]):
            for L in range(N_LAYERS):
                base = L * N_EXP
                for e in warm_rows[ti, L]:
                    insert(base + int(e))

    # drive state -------------------------------------------------------- #
    dt = 0.0                       # drive busy until this (absolute) time
    pq: "deque[int]" = deque()     # queued prefetch keys, FIFO
    rem: dict[int, float] = {}     # key -> bytes still to fetch (queued/in flight)
    started: set[int] = set()      # prefetch keys that have had >= 1 chunk served

    stall_total = 0.0
    n_miss = 0
    n_req = 0
    n_layers_stalled = 0
    pf_issued = 0
    pf_bytes = 0.0                 # bytes the drive actually spent on prefetch
    pf_wasted_bytes = 0.0          # prefetch bytes that were never demanded
    pf_full_hits = 0               # demanded expert already landed via prefetch
    pf_partial = 0                 # demanded expert was in flight
    pf_hidden_ms = 0.0             # drive time the partial hides saved
    pf_used_bytes = 0.0            # prefetch bytes that a demand later wanted
    ramp_saved = 0

    landed_from_pf: set[int] = set()   # keys currently resident thanks to prefetch

    def serve_prefetch_until(t_limit):
        """Run the drive on queued prefetch work up to ``t_limit``.

        A chunk that starts before the limit runs past it -- that is the
        head-of-line cost a demand pays for prefetching.
        """
        nonlocal dt, pf_bytes
        while pq and dt < t_limit:
            key = pq[0]
            take = min(CHUNK_BYTES, rem[key])
            dt += take / (NVME_GBPS * 1e9) * 1000.0
            pf_bytes += take
            rem[key] -= take
            started.add(key)
            if rem[key] <= 0:
                pq.popleft()
                del rem[key]
                started.discard(key)
                insert(key)
                landed_from_pf.add(key)
        if dt < t_limit:
            dt = t_limit               # drive idle
        return dt

    t = 0.0
    for ti in range(n):
        # a token boundary cancels prefetches that have not started
        keep = deque()
        for key in pq:
            if key in started:
                keep.append(key)
            else:
                del rem[key]
        pq = keep

        tok_start = t
        for L in range(N_LAYERS):
            base = L * N_EXP

            # --- issue prefetch for layer L + lead at the START of L ------ #
            if predictor is not None and lead > 0:
                tgt = L + lead
                if tgt < N_LAYERS and len(pq) < PF_BUDGET:
                    tbase = tgt * N_EXP
                    n_this = 0
                    for e in predictor.predict(ti, tgt):
                        if n_this >= budget:
                            break
                        key = tbase + e
                        if key in cache or key in rem:
                            continue
                        pq.append(key)
                        rem[key] = float(EXPERT_BYTES)
                        pf_issued += 1
                        n_this += 1
                        if len(pq) >= PF_BUDGET:
                            break

            # --- compute window: the drive is free -------------------------- #
            t_demand = t + T_LAYER_MS
            serve_prefetch_until(t_demand)

            # --- demand ---------------------------------------------------- #
            need_bytes = 0.0
            fresh = 0
            for e in top6[ti, L]:
                key = base + int(e)
                n_req += 1
                if key in cache:
                    touch(key)
                    if key in landed_from_pf:
                        pf_full_hits += 1
                        pf_used_bytes += EXPERT_BYTES
                        landed_from_pf.discard(key)
                    continue
                n_miss += 1
                if key in rem:                      # prefetch in flight -> promote
                    need_bytes += rem[key]
                    done = EXPERT_BYTES - rem[key]
                    pf_hidden_ms += done / (NVME_GBPS * 1e9) * 1000.0
                    pf_used_bytes += done
                    pf_partial += 1
                    pq.remove(key)
                    del rem[key]
                    started.discard(key)
                else:
                    need_bytes += EXPERT_BYTES
                    fresh += 1
                insert(key)
                landed_from_pf.discard(key)

            if need_bytes > 0:
                n_layers_stalled += 1
                ramp = RAMP_MS if fresh > 0 else 0.0
                if fresh == 0:
                    ramp_saved += 1
                finish = dt + ramp + need_bytes / (NVME_GBPS * 1e9) * 1000.0
                dt = finish
                stall_total += max(0.0, finish - t_demand)
                t = max(t_demand, finish)
            else:
                t = t_demand

        # bytes prefetched this token that nothing asked for
        _ = tok_start

    # prefetch bytes the drive moved that no demand ever wanted
    pf_wasted_bytes = max(0.0, pf_bytes - pf_used_bytes)

    stall = stall_total / n
    tok_ms = COMPUTE_MS + FIXED_MS + stall
    return {
        "label": label,
        "capacity": capacity,
        "tokens": int(n),
        "hit": 1.0 - n_miss / n_req,
        "miss_per_token": n_miss / n,
        "layers_stalled_per_token": n_layers_stalled / n,
        "stall_ms": stall,
        "token_ms": tok_ms,
        "tok_s": 1000.0 / tok_ms,
        "pf_issued_per_token": pf_issued / n,
        "pf_mb_per_token": pf_bytes / n / 1e6,
        "pf_wasted_mb_per_token": pf_wasted_bytes / n / 1e6,
        "pf_full_hits_per_token": pf_full_hits / n,
        "pf_partial_per_token": pf_partial / n,
        "pf_hidden_ms_per_token": pf_hidden_ms / n,
        "ramp_saved_per_token": ramp_saved / n,
        "lead": lead,
        "budget": budget if predictor is not None else 0,
    }


# --------------------------------------------------------------------------- #

def quality(preds_d, top6, d):
    """precision = recall of the trace's lookahead at K = 6, layers >= d only."""
    tgt = slice(d, N_LAYERS)
    p = preds_d[:, tgt, :TOP_K]
    a = top6[:, tgt, :]
    inter = 0
    for j in range(TOP_K):
        inter += (p[:, :, j:j + 1] == a).any(axis=2).sum()
    return float(inter) / (p.shape[0] * p.shape[1] * TOP_K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/mixed")
    ap.add_argument("--out", default="bench/results/f6")
    ap.add_argument("--capacity", type=int, default=5100)
    ap.add_argument("--tokens", type=int, default=0, help="cap the test set (0 = all)")
    ap.add_argument("--precisions", default="0.5,0.65,0.77,0.9,1.0")
    ap.add_argument("--leads", default="1,2,3")
    a = ap.parse_args()

    t0 = time.time()
    top6, top16, preds, prompt, pos = load_trace(a.trace)
    # prompt/pos come from one layer's file: one row per token, decode order
    tr, te = split_by_prompt(prompt)
    trn = top6[tr]
    tst = top6[te]
    t16 = top16[te]
    if a.tokens:
        tst = tst[:a.tokens]
        t16 = t16[:a.tokens]
    warm = trn[-2000:]
    print(f"  train {trn.shape[0]} / test {tst.shape[0]} tokens, warm {warm.shape[0]}")

    q = {d: quality(preds[d][te][:tst.shape[0]], tst, d) for d in (1, 2, 3, 4)}
    print("  trace lookahead precision@K=6: " +
          "  ".join(f"d={d} {v:.3f}" for d, v in q.items()))

    rows = []
    base = replay(tst, capacity=a.capacity, warm_rows=warm, label="demand-only LRU")
    rows.append(base)
    print(f"  BASE stall {base['stall_ms']:.1f} ms  {base['tok_s']:.3f} tok/s  "
          f"hit {base['hit']:.4f}  miss/tok {base['miss_per_token']:.1f}  "
          f"layers stalled {base['layers_stalled_per_token']:.1f}")

    leads = [int(x) for x in a.leads.split(",")]
    precs = [float(x) for x in a.precisions.split(",")]

    # The freshest residual a planner holds at the START of layer L is layer
    # L-1's output (layer L's own output does not exist until after layer L's
    # MoE, i.e. after its stall).  The trace's pred_dK column for target `tgt`
    # was built from the snapshot `K` layers earlier, so a lead of `d` layers
    # reads pred_d(d+1).  Getting this off by one is what makes a lookahead look
    # one layer cheaper than it is.
    for d in leads:
        k = d + 1
        pr = TracePredictor(preds[k][te][:tst.shape[0]], name=f"lookahead lead={d}")
        r = replay(tst, capacity=a.capacity, warm_rows=warm, predictor=pr, lead=d,
                   label=f"trace lookahead lead={d} K=6 (p={q[k]:.3f})")
        r["precision"] = q[k]
        r["kind"] = "trace"
        rows.append(r)
        print(f"  lead={d} trace p={q[k]:.3f}: stall {r['stall_ms']:.1f}  "
              f"{r['tok_s']:.3f} tok/s  ({100*(r['tok_s']/base['tok_s']-1):+.1f}%)  "
              f"pf {r['pf_mb_per_token']:.0f} MB/tok")

    for d in leads:
        for p in precs:
            pr = SyntheticPredictor(tst, p, seed=1000 + d)
            r = replay(tst, capacity=a.capacity, warm_rows=warm, predictor=pr, lead=d,
                       label=f"synthetic p={p:.2f} d={d} K=6")
            r["precision"] = p
            r["lead"] = d
            r["kind"] = "synthetic"
            rows.append(r)
            print(f"  d={d} p={p:.2f}: stall {r['stall_ms']:.1f}  "
                  f"{r['tok_s']:.3f} tok/s  ({100*(r['tok_s']/base['tok_s']-1):+.1f}%)  "
                  f"pf {r['pf_mb_per_token']:.0f} MB/tok  "
                  f"waste {r['pf_wasted_mb_per_token']:.0f} MB/tok")

    # --- budget sweep on the best implementable predictor ----------------- #
    for d in leads:
        pr0 = TracePredictor(preds[d + 1][te][:tst.shape[0]], name=f"lookahead lead={d}")
        for b in (2, 3, 4):
            r = replay(tst, capacity=a.capacity, warm_rows=warm, predictor=pr0,
                       lead=d, budget=b,
                       label=f"trace lookahead lead={d} K={b} (p={q[d + 1]:.3f})")
            r["precision"] = q[d + 1]
            r["kind"] = "trace"
            rows.append(r)
            print(f"  d={d} trace K={b}: stall {r['stall_ms']:.1f}  "
                  f"{r['tok_s']:.3f} tok/s  ({100*(r['tok_s']/base['tok_s']-1):+.1f}%)  "
                  f"pf {r['pf_mb_per_token']:.0f} MB/tok")

    # --- provisional gate: two-layer lead, precision is a free parameter ---- #
    # Predicting layer L+1 from layer L's PRE-MoE residual (h + attn out) makes
    # the prediction available one whole layer earlier than the real gate, i.e.
    # the same free-drive lead as the d=2 rows above, at a precision this trace
    # cannot measure (it recorded no pre-MoE residual snapshots).  So the cell is
    # the d=2 lead with p as the parameter -- reported here so the break-even p
    # can be read off directly.
    # Upper bound: give the 2-layer-lead slot the precision of the trace's own
    # pred_d1 (a gate run on the layer's true POST-MoE output).  A provisional
    # gate run on the PRE-MoE residual can only be worse, so this row is a
    # ceiling on the variant, not an estimate of it.
    pr = TracePredictor(preds[1][te][:tst.shape[0]], name="provisional ceiling")
    r = replay(tst, capacity=a.capacity, warm_rows=warm, predictor=pr, lead=1,
               label=f"provisional gate, 2-layer lead, CEILING (p={q[1]:.3f})")
    r["precision"] = q[1]
    r["kind"] = "provisional-ceiling"
    rows.append(r)
    print(f"  prov ceiling p={q[1]:.3f}: stall {r['stall_ms']:.1f}  {r['tok_s']:.3f} "
          f"tok/s  ({100*(r['tok_s']/base['tok_s']-1):+.1f}%)")
    for p in precs:
        pr = NearMissPredictor(tst, t16, p, seed=77)
        r = replay(tst, capacity=a.capacity, warm_rows=warm, predictor=pr, lead=1,
                   label=f"provisional gate, 2-layer lead, p={p:.2f}")
        r["precision"] = p
        r["kind"] = "provisional"
        rows.append(r)
        print(f"  prov p={p:.2f}: stall {r['stall_ms']:.1f}  {r['tok_s']:.3f} tok/s  "
              f"({100*(r['tok_s']/base['tok_s']-1):+.1f}%)")

    # --- the control: same precision, realistic (near-miss) error shape ---- #
    for p in precs:
        pr = NearMissPredictor(tst, t16, p, seed=31337)
        r = replay(tst, capacity=a.capacity, warm_rows=warm, predictor=pr, lead=1,
                   label=f"near-miss synthetic p={p:.2f} lead=1 K=6")
        r["precision"] = p
        r["kind"] = "nearmiss"
        rows.append(r)
        print(f"  nearmiss lead=1 p={p:.2f}: stall {r['stall_ms']:.1f}  "
              f"{r['tok_s']:.3f} tok/s  ({100*(r['tok_s']/base['tok_s']-1):+.1f}%)  "
              f"issued {r['pf_issued_per_token']:.1f}/tok")

    os.makedirs(a.out, exist_ok=True)
    out = {
        "model": {
            "nvme_gbps": NVME_GBPS, "s_ms": S_MS, "chunk_ms": CHUNK_MS,
            "ramp_ms": RAMP_MS, "t_layer_ms": T_LAYER_MS,
            "compute_ms": COMPUTE_MS, "fixed_ms": FIXED_MS,
            "expert_bytes": EXPERT_BYTES, "capacity": a.capacity,
        },
        "lookahead_precision": q,
        "rows": rows,
    }
    with open(os.path.join(a.out, "idle_prefetch.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {a.out}/idle_prefetch.json in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
