#!/usr/bin/env python3
"""Tests for tools/cache_sim.py, on synthetic traces with known answers.

    uv run python tools/tests/test_cache_sim.py      # no pytest needed
    uv run python -m pytest tools/tests/             # if pytest is installed

Design section 9.2 makes `cache_sim.py` the thing that picks the eviction policy that
goes into `store/planner.cpp`, so it has to be right on cases where the answer is
known independently:

  * a trace with perfect locality must come out at hit rate ~= 1
  * a uniform-random trace must come out at the analytic LRU hit rate, which for
    C slots over M equally likely keys is C/M
  * the exact stack-distance routine must agree with a brute-force definition
  * Jaccard and static-frequency must agree with hand-computed values

The tests build `Trace` objects directly rather than going through Parquet, so they
run in a second and do not need a real trace on disk.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_sim as cs  # noqa: E402


def make_trace(top6: np.ndarray, layer: np.ndarray, prompt=None, pos=None,
               n_experts: int = 8, preds=None) -> cs.Trace:
    n = len(layer)
    prompt = np.zeros(n, dtype=np.int32) if prompt is None else prompt
    pos = (np.arange(n) // max(1, int(layer.max()) + 1)) if pos is None else pos
    top16 = np.concatenate([top6, top6], axis=1)[:, :16] if top6.shape[1] < 16 else top6
    if top16.shape[1] < 16:
        top16 = np.pad(top16, ((0, 0), (0, 16 - top16.shape[1])), mode="edge")
    scores = np.ones((n, 16), dtype=np.float32)
    return cs.Trace(top6.astype(np.int32), top16.astype(np.int32), scores,
                    layer.astype(np.int32), prompt.astype(np.int32),
                    np.asarray(pos, dtype=np.int64), preds or {},
                    np.zeros(n, dtype=bool), int(layer.max()) + 1, n_experts)


def default_args(**over) -> argparse.Namespace:
    a = cs.build_parser().parse_args(["--trace", "x"])
    a.warmup_frac = 0.0
    for k, v in over.items():
        setattr(a, k, v)
    return a


# --------------------------------------------------------------------------- #

def test_perfect_locality():
    """Every token routes to the same six experts, so after the first token every
    request is a hit. With one layer and 6 distinct experts in an 8-slot cache the
    hit rate must be (n-1)/n for every policy."""
    n = 200
    top6 = np.tile(np.arange(6), (n, 1))
    trace = make_trace(top6, np.zeros(n, dtype=np.int32))
    for name in ("lru", "lfu-decay", "arc", "score-aware"):
        r = cs.run_one(trace, name, 8, default_args(), set(), "global", 0, 0)
        expect = (n - 1) / n
        assert abs(r["hit_rate"] - expect) < 1e-9, (name, r["hit_rate"], expect)
        assert r["exposed_misses"] == 6, (name, r["exposed_misses"])
    print("ok  perfect locality -> hit rate 1 - 1/n, exactly 6 cold misses")


def test_uniform_random_matches_analytic_lru():
    """For independent uniform draws over M keys, an LRU of C slots holds C keys
    that are, at steady state, a uniform random subset -- so the hit rate tends to
    C/M. Anything wildly off means the LRU bookkeeping is wrong.

    One expert per token (top-1 in a top-6 slot, repeated) keeps the analysis clean:
    with six distinct draws per token the six lookups inside one token are not
    independent of each other.
    """
    rng = np.random.default_rng(7)
    n, M, C = 40000, 64, 16
    draws = rng.integers(0, M, size=n)
    top6 = np.repeat(draws[:, None], 6, axis=1)
    trace = make_trace(top6, np.zeros(n, dtype=np.int32), n_experts=M)
    r = cs.run_one(trace, "lru", C, default_args(warmup_frac=0.1), set(), "global", 0, 0)
    expect = C / M
    assert abs(r["hit_rate"] - expect) < 0.03, (r["hit_rate"], expect)
    print(f"ok  uniform random: LRU hit {r['hit_rate']:.3f} vs analytic C/M "
          f"= {expect:.3f}")


def test_capacity_monotonic():
    """More slots can never hit less, for LRU (it has the stack property)."""
    rng = np.random.default_rng(3)
    n = 4000
    top6 = rng.integers(0, 40, size=(n, 6))
    trace = make_trace(top6, np.zeros(n, dtype=np.int32), n_experts=40)
    last = -1.0
    for cap in (4, 8, 16, 24, 32, 40):
        r = cs.run_one(trace, "lru", cap, default_args(), set(), "global", 0, 0)
        assert r["hit_rate"] >= last - 1e-9, (cap, r["hit_rate"], last)
        last = r["hit_rate"]
    assert last > 0.9, last
    print("ok  LRU hit rate is monotone in capacity, and saturates when it all fits")


def test_stack_distance_matches_brute_force():
    """The Fenwick implementation against the definition, on a small random trace."""
    rng = np.random.default_rng(11)
    keys = rng.integers(0, 12, size=600)
    fast = cs.stack_distances(keys)
    slow = np.full(len(keys), -1, dtype=np.int64)
    last: dict[int, int] = {}
    for t, k in enumerate(keys.tolist()):
        if k in last:
            slow[t] = len(set(keys[last[k] + 1:t].tolist()))
        last[k] = t
    assert np.array_equal(fast, slow), (fast[:20], slow[:20])
    print("ok  stack distances match the brute-force definition")


def test_stack_distance_gives_lru_hit_rate():
    """The whole point of Q2: the fraction of accesses with distance < C equals LRU's
    hit rate at capacity C.

    The CDF is the sequential-LRU curve while the simulator issues a layer's six
    requests as one batch, so they agree only up to the handful of accesses where a
    sibling's admit would have changed the recency order. With six distinct keys per
    row and a capacity well above six that is well under a point, which is what the
    tolerance here says.
    """
    rng = np.random.default_rng(5)
    n, M = 3000, 200
    top6 = np.stack([rng.permutation(M)[:6] for _ in range(n)]).astype(np.int32)
    trace = make_trace(top6, np.zeros(n, dtype=np.int32), n_experts=M)
    d = cs.stack_distances(trace.keys6.reshape(-1))
    for cap in (32, 64, 128):
        analytic = float((d[d >= 0] < cap).sum()) / len(d)
        r = cs.run_one(trace, "lru", cap, default_args(), set(), "global", 0, 0)
        assert abs(analytic - r["hit_rate"]) < 0.02, (cap, analytic, r["hit_rate"])
    print("ok  P(stack distance < C) tracks the simulated LRU hit rate at C")


def test_q1_and_q3_on_known_trace():
    rng = np.random.default_rng(2)
    n = 500
    # layer 0: always experts 0..5. layer 1: uniform over 0..7.
    t0 = np.tile(np.arange(6), (n, 1))
    t1 = np.stack([rng.permutation(8)[:6] for _ in range(n)])
    top6 = np.empty((2 * n, 6), dtype=np.int32)
    layer = np.empty(2 * n, dtype=np.int32)
    top6[0::2], layer[0::2] = t0, 0
    top6[1::2], layer[1::2] = t1, 1
    pos = np.repeat(np.arange(n), 2)
    trace = make_trace(top6, layer, pos=pos, n_experts=8)

    q1 = cs.analyse_q1(trace, default_args())
    l0 = q1["per_layer"][0]
    assert l0["distinct"] == 6, l0
    # six equally used experts out of eight: the busiest 50% (4) carry 4/6
    assert abs(l0["coverage"]["top50pct"] - 4 / 6) < 1e-3, l0["coverage"]
    q3 = cs.analyse_q3(trace, default_args())
    by_layer = {e["layer"]: e for e in q3["per_layer"]}
    assert abs(by_layer[0]["jaccard_mean"] - 1.0) < 1e-9, by_layer[0]
    assert by_layer[1]["jaccard_mean"] < 1.0, by_layer[1]
    assert abs(by_layer[0]["union_frac"][2] - 0.5) < 1e-9, by_layer[0]
    print("ok  Q1 coverage and Q3 Jaccard / union match the hand-computed values")


def test_prefetch_perfect_and_useless():
    """An oracle prediction should hide misses; a prediction that never overlaps
    should be pure waste.

    The cache is sized so probes actually survive the two layers between landing and
    being read: at `--probe-position head` (the default) a landed prefetch is a
    normal fill, so it lives as long as any other recently-filled slot.
    """
    rng = np.random.default_rng(13)
    n_tok, n_layers, M = 300, 4, 200
    rows = n_tok * n_layers
    top6 = np.stack([rng.permutation(M // 2)[:6] for _ in range(rows)])
    layer = np.tile(np.arange(n_layers), n_tok).astype(np.int32)
    pos = np.repeat(np.arange(n_tok), n_layers)
    good = np.pad(top6, ((0, 0), (0, 10)), mode="edge")          # contains the truth
    bad = np.full((rows, 16), M - 1, dtype=np.int32)             # never routed to
    tr_good = make_trace(top6, layer, pos=pos, n_experts=M, preds={2: good})
    tr_bad = make_trace(top6, layer, pos=pos, n_experts=M, preds={2: bad})
    a = default_args(t_layer_ms=1000.0)        # plenty of time for a prefetch to land

    g = cs.run_one(tr_good, "lru", 150, a, set(), "global", 2, 16)
    b = cs.run_one(tr_bad, "lru", 150, a, set(), "global", 2, 16)
    assert g["prefetch_used"] > 0 and g["hidden_misses"] > 0, g
    assert g["prefetch_precision"] > 0.5, g
    assert g["effective_hit_rate"] > g["hit_rate"], g
    assert b["prefetch_precision"] == 0.0, b
    assert b["prefetch_waste_bytes_per_token"] > 0, b
    assert b["effective_hit_rate"] == b["hit_rate"], b
    q4 = cs.analyse_q4(tr_good, a)["by_depth"][2]
    assert abs(q4[16]["recall"] - 1.0) < 1e-9, q4
    print(f"ok  prefetch: an oracle hides {g['hidden_misses']} misses at precision "
          f"{g['prefetch_precision']:.2f}; useless predictions are pure waste")


def test_probe_at_tail_is_self_defeating():
    """Section 9.4 says a landed prefetch goes in at the LRU tail. In a full cache
    that makes it the next victim, so it is usually gone before the demand read it
    was fetched for -- which is why `--probe-position` exists and why both settings
    are swept rather than one being assumed."""
    rng = np.random.default_rng(31)
    n_tok, n_layers, M = 300, 4, 200
    rows = n_tok * n_layers
    top6 = np.stack([rng.permutation(M // 2)[:6] for _ in range(rows)])
    layer = np.tile(np.arange(n_layers), n_tok).astype(np.int32)
    pos = np.repeat(np.arange(n_tok), n_layers)
    good = np.pad(top6, ((0, 0), (0, 10)), mode="edge")
    tr = make_trace(top6, layer, pos=pos, n_experts=M, preds={2: good})
    head = cs.run_one(tr, "lru", 150, default_args(t_layer_ms=1000.0,
                                                   probe_position="head"),
                      set(), "global", 2, 16)
    tail = cs.run_one(tr, "lru", 150, default_args(t_layer_ms=1000.0,
                                                   probe_position="tail"),
                      set(), "global", 2, 16)
    assert tail["prefetch_used"] < head["prefetch_used"], (tail, head)
    print(f"ok  probe at the LRU tail is worse than a normal fill "
          f"({tail['prefetch_used']} used vs {head['prefetch_used']})")


def test_static_pin_never_evicts_pinned():
    n = 500
    rng = np.random.default_rng(17)
    top6 = rng.integers(0, 30, size=(n, 6))
    trace = make_trace(top6, np.zeros(n, dtype=np.int32), n_experts=30)
    pinned = {0, 1, 2}
    p = cs.StaticPinLRU(10, pinned)
    for row in top6:
        for k in row:
            k = int(k)
            if p.contains(k):
                p.touch(k)
            else:
                p.admit(k)
    assert pinned <= {k for k in range(30) if p.contains(k)}
    assert p.size() <= 10, p.size()
    print("ok  static-pin+lru keeps the pinned set resident and respects capacity")


def test_tps_model_matches_section_3_1():
    """The reported ms/token must be section 3.1's model: 42 ms resident, plus hit
    bytes over LPDDR, plus miss bytes over NVMe."""
    n = 100
    top6 = np.tile(np.arange(6), (n, 1))
    trace = make_trace(top6, np.zeros(n, dtype=np.int32))
    a = default_args()
    r = cs.run_one(trace, "lru", 8, a, set(), "global", 0, 0)
    expect = (cs.RESIDENT_MS
              + r["nvme_bytes_per_token"] / (a.nvme_gbps * 1e9) * 1e3
              + (6 * cs.EXPERT_BYTES - r["nvme_bytes_per_token"])
              / (a.lpddr_gbps * 1e9) * 1e3)
    assert abs(r["ms_per_token_serial"] - expect) < 0.2, (r["ms_per_token_serial"], expect)
    # with no prefetch the overlapped model must agree with the serial one
    assert abs(r["ms_per_token"] - r["ms_per_token_serial"]) < 1.0, r
    assert r["effective_hit_rate"] == r["hit_rate"], r
    print(f"ok  tok/s model matches section 3.1 ({r['ms_per_token_serial']} ms/token)")


def test_arc_invariants():
    """ARC must never hold more than `capacity` resident keys, must never lose a key
    it reports as resident, and must not crash on a long adversarial stream.

    The crash it is guarding against is real: an earlier version trimmed the ghost
    lists inside REPLACE, which could evict the very key being promoted out of B2 and
    then KeyError on the delete. It survived a 4,000-access random trace and died on
    the smoke trace at 116,000.
    """
    rng = np.random.default_rng(23)
    cap, M = 40, 400
    p = cs.ARC(cap)
    resident = set()
    for _ in range(200000):
        # a mix of a small hot set and a long cold tail, which is what makes ARC
        # move its T1/T2 split around
        k = int(rng.integers(0, 20) if rng.random() < 0.5 else rng.integers(0, M))
        if p.contains(k):
            assert k in resident, k
            p.touch(k)
        else:
            v = p.admit(k)
            resident.add(k)
            if v is not None:
                resident.discard(v)
        assert p.size() <= cap, (p.size(), cap)
        assert len(p.t1) + len(p.b1) <= cap, (len(p.t1), len(p.b1))
        assert len(p.t1) + len(p.t2) + len(p.b1) + len(p.b2) <= 2 * cap
    assert {k for k in range(M) if p.contains(k)} == resident
    print(f"ok  ARC holds its invariants over 200k accesses (p = {p.p:.1f}, "
          f"|T1| = {len(p.t1)}, |T2| = {len(p.t2)})")


def test_heat_policies_survive_prefetch_probes():
    """A heat policy must not fall over when a prefetch probe is admitted.

    `admit` adds the key to `resident` and then calls `_bump`, which can trigger a
    heap rebuild over `resident` before the heat has been written -- the rebuild has
    to tolerate that.
    """
    rng = np.random.default_rng(41)
    n_tok, n_layers, M = 400, 4, 300
    rows = n_tok * n_layers
    top6 = np.stack([rng.permutation(M // 2)[:6] for _ in range(rows)])
    layer = np.tile(np.arange(n_layers), n_tok).astype(np.int32)
    pos = np.repeat(np.arange(n_tok), n_layers)
    pred = np.pad(top6, ((0, 0), (0, 10)), mode="edge")
    tr = make_trace(top6, layer, pos=pos, n_experts=M, preds={2: pred})
    for name in ("lfu-decay", "score-aware"):
        for pp in ("head", "tail"):
            a = default_args(t_layer_ms=1000.0, probe_position=pp)
            r = cs.run_one(tr, name, 120, a, set(), "global", 2, 16)
            assert 0.0 <= r["hit_rate"] <= 1.0, r
            assert r["effective_hit_rate"] >= r["hit_rate"] - 1e-9, r
    print("ok  heat policies survive prefetch probes at both probe positions")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
