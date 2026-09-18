# Expert routing / cache hit patterns (Track R1, 2026-09-16)

Analysis of the engine's own `route.bin` dumps (`tools/expert_patterns.py`).
Runs: `base_auto` (4500 slots, 8 turns, 2256 decode steps), `base_88g`
(5000 slots, 2 turns), `stall_off` (4500 slots), plus the 27k-token
`traces/mixed` corpus behind `store/static_heat.inc`.

## 1. The patterns

1. **Decode inherits most of its expert set from the prompt prefill.**
   `prefill_hit = 0.7607`; **86.6%** of decode expert selections were already
   touched during the prefill. Prefill→decode handoff is the single biggest
   hit-rate lever.
2. **The hit drop is a topic switch, not context length.**
   Adjacent-turn frequency cosine is 0.39 (8 unrelated prompts), first/last
   0.18; two related turns give 0.75. Per-128-step hit goes 0.94–0.955 inside a
   topic and 0.877–0.90 across a topic switch; stall moves 58 ms ↔ 151 ms.
3. **There is no small always-hot core.** Experts present in ≥50% of decode
   steps: 0.025/layer, 0.2% of events. In the 27k mixed trace, global top-16
   covers 3.0%, top-1024 40.8%, top-4096 73.4%; per layer top-16 covers 32.4%,
   top-64 60.8%, and ~380/384 experts are touched.
4. **Temporal locality is long-range and weak.** Reuse within the previous
   1/4/16/128 decode steps is 15.4% / 29.3% / 42.6% / 63.7%. LRU must retain
   hundreds of steps; adjacent-token prefetch is useless.
5. **Capacity is nearly linear in hit rate.** Same route replayed by cache_sim:
   2700→0.848, 3400→0.888, 4000→0.909, 4500→0.923, 5000→0.934,
   5711→0.945, 6500→0.953, 7680→0.962, 10000→0.974.
6. **Layer depth matters.** L0/L1 are the flattest and lowest (hit 0.887,
   top-16 share 0.22); later layers are more concentrated and higher
   (L24–L35 hit 0.94+, top-16 0.35–0.42). L19/L23 are local minima.
7. **Position is not a problem in the measured range.** Hit 0.901 (0–127) →
   0.929 (128–511) → 0.941 (512–2047): warming up improves hit.

## 2. Optimizations this implies

- **Capacity**: expose and use `serve --cache-slots N`. 5711 slots ≈ 100 GiB,
  expected +2.2 hit points over the auto 4500 and ~20% less miss stall.
- **Recency heat instead of global static heat**: `hitrate_bench.py
  --heat-recent N` generates the P3 backfill order from the last N routing
  records (a turn/topic), not the 27k global corpus. Global heat is too flat;
  topic-local sets are what repeats.
- **Prefill handoff**: keep the prompt's last-W experts and the experts routed
  by the prompt in the decode cache (`ExpertStore::adopt` /
  `DEEPMOE_PREFILL_HANDOFF`).
- **Layer-aware policy**: L0/L1 and boundary layers should prefer the recent
  set; late layers can trust heat/prefetch more.
- **A/B split**: report same-topic follow-ups and topic switches separately;
  averaging them hides the two regimes.

## 3. What it takes to reach 20 tok/s

M=1 decode moves ≥13.03 GB/token; at 217 GB/s that is **60 ms = 16.7 tok/s**
(a 256 GB/s theoretical ceiling is 19.6). So 20 tok/s sustained needs
speculative verify to amortize weight reads: accepted ≈3.3 tokens/verify and
C(M=6)/C(M=1) must be ≲2 while hit rate stays high enough that
stall(M=6) does not dominate. Capacity + handoff + recency heat are what make
the spec cycle competitive; DSpark supply of M>1 kernels (Track T) is already in
this PR.

## 4. Capacity is now a real knob (2026-09-16)

`serve --cache-slots N` is implemented. On this machine: 4500–5500 slots start;
**5600 fails** in path B's 20th slab (98.1 GiB). The same 8-turn script:

- 4500 slots: hit 0.9234 (cache_sim), old run ~4.5–5.2 tok/s
- **5500 slots: hit 0.9431, 6.05 tok/s, stall 46–71 ms**

This matches the simulated 5711-slot hit of 0.9451, so the gain is capacity, not
a policy change. Per-turn dips (0.912–0.916) remain at topic switches; capacity
does not remove that pattern, which is why recency heat and prefill handoff are
still needed.
