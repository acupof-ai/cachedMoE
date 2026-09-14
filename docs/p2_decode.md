# P2 step 2 — the first token

> A whole decode step through all forty layers on the GPU, and eight of them in
> a row, checked token by token against `inference/model.py`.
> Companion to [p2_attention.md](p2_attention.md) (the per-stage kernels and one
> layer), [kernel_p2_moe.md](kernel_p2_moe.md) (the expert dispatches) and
> [kernel_p1.md](kernel_p1.md) (the bandwidth matrix).

Status: **2026-09-14**. Everything here is the real `DeepSeek-V4.1-Flash`
checkpoint on the real machine. Raw data: `tests/data/l3/`,
`tests/test_decode.cpp`, and the `deepmoe run` transcripts quoted in §5.

---

## 0. What this says in one page

`Engine::decode_step` runs the model. Embedding lookup, forty layers of design
§7.14's dispatches 1–11 with the §7.1 residency gate sitting between 9 and 10,
the engram at layers 1 and 14 including its NVMe row fetch, the final collapse,
the 1.32 GB head, and a greedy argmax that returns four words instead of
129,280 logits.

Against the fp32 reference, at a 64-token prompt and position 64:

| | result |
|---|---|
| one decode step, logits vs L3 | **top-1 matches**; Spearman ρ = 0.97 over the reference's top 64; max \|Δlogit\| = 0.64 |
| eight steps, teacher-forced | **7 of 8 tokens match**; the one that differs does so at a reference margin of 0.95 |
| eight steps, free-running | **6 of 8 before divergence** |
| the stream, probed against L2 at seven layers | cosine ≥ **0.996** forty layers in; 6 of the 7 probed layers route to all six of the reference's experts |

design §12 L3 asks for 100% token agreement. This is 7/8 and 6/8, and §4 says
what the missing one is: at layer 2 the gate picks a different **sixth** expert
at a near-tie, and that one swap is the largest single term in the logit gap.

Speed: a step whose experts are all resident is **134 ms — 7.5 tok/s**, which is
this implementation's compute floor and 1.8× design §13.4's kernel budget. From
cold at a 0.36 hit rate it is 0.88 tok/s, 83% of it the GPU waiting for NVMe.
§5 breaks both down and §6 splits the gap: two-thirds of it is the MoE, and most
of that is launch and host overhead rather than the kernel.

**Three bugs only a forty-layer chain could find.** They are in §3, and all
three are the same shape: something that a per-layer or per-kernel test is
structurally unable to look at.

---

## 1. The L3 oracle

`tools/oracle.py --level l3` writes `tests/data/l3/`: 7.1 MB, 459 s, run once.

It prefills the 64-token L2 prompt through `inference/model.py` behind
`tools/dsref.py`'s CPU kernel shims, then decodes eight greedy steps. One pass
over the forty layers per step, each `Block` built from the shards and dropped
again, with only the ~200 KiB of per-layer KV buffers carried across — the same
structure `level2_layer` uses, nine times instead of twice. That is the opposite
of layer streaming, and deliberately so: a decode step needs every layer's
weights, so there is no ordering that reads less than the whole ~7.6 GB.

### 1.1 What is in it

| record | contents |
|---|---|
| `prefill` | every layer's window KV ring after the prompt (`L00.win_kv` … `L39.win_kv`, bf16 [128, 512]), the logits at the last prompt position, and the collapse inputs |
| `step00` … `step07` | per layer, the compressed KV (`Lnn.cmp_kv`) and the indexer's top-k list (`Lnn.topk_idxs`) that layer actually read; the step's logits; its input and output token |
| `engram` (in `index.json`) | the hash constants, plus `engram_token_map.bin` |

Logits are stored as the **top 64 (id, logit)** plus the max, the log-sum-exp
and the min of the full 129,280-wide vector. Storing all of it would be 4.7 MB
for nothing: every question design §12 L3 asks — does the argmax match, at what
margin, how far down does the ranking hold — is answered by the top of the
distribution, and the three whole-vector statistics catch a runtime whose top-64
agrees but whose tail does not.

Two index entries may share a byte offset. A compressed-KV cache is published by
its source layer and read verbatim by every reuse layer under it
(`model.py`'s `shared_attn`), so of forty entries only four are distinct; the
same for the eight index sources. De-duplicating takes the export from 22 MB to
7.1 MB and costs the reader nothing.

### 1.2 The engram tables are exported, the row ids are not

`inference/engram.py` derives the engram hash from the **tokenizer** and
`config.json`, not from the checkpoint: the compressed-vocabulary map is a
normalisation (NFKC, strip accents, lowercase, collapse whitespace) over all
129,280 decoded tokens; the per-(layer, lookback) multipliers come from
`numpy.random.default_rng(10007 * layer_id)`; the 24 bucket moduli are the next
24 unused primes above `engram_vocab_size - 1`.

Reproducing a PCG64 stream and a `tokenizers` normaliser in C++ would be a
second implementation to keep correct forever. So the export carries those
constants — 517 KB of token map and about a hundred integers — and
`runtime/engram.h` computes the **addresses** with them, on whatever trajectory
the runtime is actually on. What is loaded is a constant table; nothing about a
particular token is.

### 1.3 The one coincidence worth knowing

The model's greedy continuation at position 64 is the prompt's own next token
(3006). So L2's decode step and L3's step 0 are the **same forward pass**, and
`tests/data/l2`'s seven per-stage layers apply directly to step 0 of the decode
loop. That is what makes the per-layer probe in §4.1 possible without a second
export.

---

## 2. What runs, and what is handed to it

| piece | state |
|---|---|
| embedding lookup, the forty layers' dispatches 1–9, gate, MoE, final collapse, head, argmax | **real** |
| the engram at layers 1 and 14: hash, 48-read NVMe fetch, fp8 GEMV, gate, residual write | **real** |
| the window KV ring | **real** after the first step — seeded once from the prefill export, written by `wkv.slang` every step after |
| the routed experts | **real**, fetched through `store::Planner` on the gate's own ids, evicted by LRU |
| the shared expert | **real**, fp8, from the pinned set |
| the compressed KV and the indexer's top-k list | **LOADED**, per step, from the L3 export. design §7.4's kernels are another track's and were not there when this was written |
| the state the prompt leaves behind | **LOADED**. Prefill is design §11 / P5 |
| DSpark | not started (P4) |

`Engine::status()` prints the LOADED line on every run, so a transcript can
never be mistaken for a self-contained one.

### 2.1 The pinned set is 9.17 GiB, not 17.7 GB

design §9.3's 17.7 GB includes the three DSpark (mtp) blocks. Decode does not
touch them, so `init_gpu` pins layers 0–39 plus embed / norm / head: **884
tensors, 9.17 GiB, loaded in 3.4 s at 2.9 GB/s** through the IoEngine into path
A. The commit check of §5.2 runs once for the pinned set and the expert cache
together, before anything is allocated, so a machine that cannot hold both fails
with a sentence pointing at `docs/build.md` rather than with
`VK_ERROR_OUT_OF_DEVICE_MEMORY` 8 GB in.

### 2.2 The expert cache fills path A, then path B

`DualPathBacking` hands `SlabPool` path A until the allocator refuses and path B
afterwards, which is the difference between a 74 GiB cache and a 100 GiB one
(design §3.3, `bench/results/heap_capacity_idle.csv`). The ExpertStore never
learns which it got: both return a `(host_ptr, device_address)` pair. On the
runs in §5 path B is never reached — 12 GiB and 48 GiB both fit in path A.

---

## 3. Three bugs a chain finds and a layer cannot

### 3.1 The fused `hc_post` was folding in the wrong sublayer's output

design §7.7 defers a sublayer's `hc_post` into the **next** sublayer's first
`mega_mhc`. `DecodeLayer::bind` had the attention half's `hc_post` reading
`wob` — the attention output — where it must read `moe_y`, because the sublayer
before a layer's attention is the **previous layer's FFN**.

`tests/test_gpu_layer.cpp` cannot see this. It runs one layer from the oracle's
own input, which means it runs with `apply_hc_post` off, which means the slot is
never read. Forty layers read nothing else: every layer's attention output went
into the stream twice and every layer's MoE output was dropped. The model
decoded the input token over and over, at ρ = 0.14 and max |Δlogit| = 13.

### 3.2 `engram.slang` read the FP8 decode table before the barrier that fills it

`fp8_table_fill(tid)` writes `gTbl[tid]`; the staging pass then decodes bytes
through `gTbl[byte]`, which is some other thread's entry. Inside one Wave32 the
two are in lockstep, so entries 0..31 were always right and the rest were
whatever the other seven waves had reached.

The symptom was not a failure. It was **the same eight tokens coming out
differently from one run to the next**, and occasionally a step whose logits
were all NaN — which the sampler reports as token 0 at margin 0.0000, because
`v > best` is false for every NaN and the reduction keeps its −3e38 sentinel.
Two barriers instead of one.

### 3.3 Computing over write-combining memory, a float at a time

design §3.3 says CPU reads of the path-A mapping are uncached and very slow.
`GpuMoeBridge::run` did three element-wise loops straight off GPU memory — the
activation in, the routed `y` out, the shared `y` out. That is 25,600 uncached
accesses a layer at about 230 ns each:

| | ms/layer | ms/token | share of the step |
|---|---:|---:|---:|
| element-wise | 6.0 | 240 | **26%** |
| one `memcpy` per vector | 0.75 | 30 | 3% |

The write side had the same shape: the embedding expansion wrote the four hc
copies interleaved, four addresses 20 KiB apart per element, which flushes a
write-combining buffer per store instead of filling it. Widen once into host
memory, `memcpy` four times.

This is worth stating as a rule rather than a fix: **anything in `runtime/` that
touches a GPU-visible pointer with a scalar loop is a bug**, and the two places
it happened were both written as the obvious loop.

---

## 4. Correctness

`tests/test_decode.cpp`, `--cache-gb 8`, against `tests/data/l3`.

### 4.1 One step at position 64, probed layer by layer

The probe compares our `ffn_norm` output, our MoE output and our routing against
`tests/data/l2`'s export of the same forty-layer pass (§1.3). It is what turns
"the token is wrong" into "the stream is already off at layer 13".

| layer | `ffn_norm` cos | `moe_out` cos | gate |
|---|---:|---:|---:|
| 0 | 0.9999334 | 0.9997782 | 6/6 |
| 1 (engram) | 0.9998987 | 0.9991552 | 6/6 |
| 2 | 0.9997171 | **0.9883866** | **5/6** |
| 13 | 0.9969364 | 0.9975051 | 6/6 |
| 14 (engram) | 0.9976363 | 0.9968870 | 6/6 |
| 20 | 0.9987847 | 0.9974528 | 6/6 |
| 39 | 0.9959864 | 0.9995808 | 6/6 |

and the logits at the end of it:

```
top1 223 vs 223 MATCH, rho=0.965, max|dlogit|=6.4e-01, mean=2.0e-01
margin: reference 7.1360, ours 7.0927
```

**Layer 2 is the interesting row.** The stream entering it is at cosine
0.9997 — as good as layers 0 and 1 — and yet its MoE output is at 0.988, an
order of magnitude worse than every other layer. The gate row says why: five of
the six routed experts are the reference's and one is not. A 0.9997-accurate
score vector is enough to reorder a near-tie between the sixth and seventh
expert, and one swapped expert out of seven moves the sum by about 1%. That is
not drift, it is a discrete branch, and it is the single largest contribution to
the logit gap.

The bar in the test is therefore **not** design §12 L2's 0.999 — that bar is for
a layer run from the oracle's own input. What is asserted is that the stream
stays inside 1% forty layers deep and that the routing agrees on at least six of
the seven probed layers.

### 4.2 Eight steps, teacher-forced

Each step is fed the reference's own input token, so each step's error is
isolated and a disagreement is attributable.

| step | in | ours | reference | ρ | max \|Δlogit\| | ref margin |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3006 | 223 | 223 | 0.965 | 0.64 | 7.14 |
| 1 | 223 | 18 | 18 | 0.983 | 0.37 | 6.36 |
| 2 | 18 | 16 | 16 | 0.956 | 0.70 | 6.45 |
| 3 | 16 | 18 | 18 | 0.845 | 0.87 | 6.83 |
| 4 | 18 | 201 | 201 | 0.978 | 0.41 | 0.51 |
| 5 | 201 | 1 | 1 | 0.918 | 1.14 | 1.86 |
| 6 | 1 | **1** | **201** | 0.630 | 4.22 | **0.95** |
| 7 | 201 | 61 | 61 | 0.887 | 1.31 | 1.95 |

**7 of 8.** Two things are worth reading off this table rather than the total.

First, the error grows with the step, and it has to: every step writes its own
row into the window KV ring, so by step 7 eight of the 128 ring slots are ours
rather than the reference's. The reference's rows and ours differ by the fp8
quantisation of a hidden state that already differs by ~1e-3.

Second, step 6 is not a coin flip. The reference's own margin there is 0.95,
which is small but not a tie, and our max |Δlogit| at that step is 4.2 — four
times any other step's. The most likely explanation is the same one layer 2
shows in §4.1: somewhere in the forty layers a gate crossed a near-tie and a
different expert ran. This has not been isolated to a layer; doing so needs an
L2-style export at step 6, which the current oracle does not produce.

### 4.3 Eight steps, free-running

```
3006 -> 223 -> 18 -> 16 -> 18 -> 201 -> 1 -> [1] -> [0]
reference:  223    18    16    18    201    1    201     61
```

**6 of 8 before divergence.** Everything after the first mismatch is decoded
against compressed KV that belongs to the reference's trajectory and not to
ours, so those tokens are not evidence about anything and the test does not
count them.

### 4.4 Reproducibility

After the barrier fix of §3.2 the first fifteen layers are bit-stable run to
run. Layers 20 and 39 are not: their probed cosines move in the seventh decimal
and the third respectively, and the token choices are stable but the margins
move by a few percent. The divergence appears between layers 14 and 20 and is
consistent with a non-deterministic reduction order in one of the MoE
dispatches, which is another track's kernel. It is recorded here rather than
chased because it does not change a token.

---

## 5. Speed

### 5.1 One warm step: 134 ms

The cleanest number in this report is step 0 of the teacher-forced run in
`tests/test_decode.cpp`, because the probe pass before it has already fetched
every expert that step needs. **Nothing waits for NVMe, and the step is 134.0
ms:**

| bucket | ms | per layer |
|---|---:|---:|
| attention, dispatches 1–9 | 51.9 | 1.30 |
| MoE, dispatches 10–11 + the shared expert | 68.6 | 1.72 |
| engram (2 layers) | 4.9 | 2.45 |
| collapse + head + argmax + seeding the LOADED KV | 8.4 | — |
| NVMe stall | 0.2 | — |
| **total** | **134.0** | **7.5 tok/s** |

That is the compute floor of this implementation: what a token costs when the
expert cache holds everything it needs. design §13.4 models the same work as
`48.5 + 27.3 = 75.8 ms`, so the kernels are **1.8×** its budget; §6 splits that.

### 5.2 Eight steps from cold, and where the time actually goes

`deepmoe run --model … --prompt-ids … --steps 8 --cache-gb 12`. The machine was
running other work during these measurements — the submit round trip below moved
between 0.13 ms and 0.75 ms across runs of the same binary — so read the
**shape** of the breakdown, not the third digit.

```
step  in     -> out     wall      | attn    moe  (gpu   host)  stall   engram other | hit     nvme
  0    3006 ->    223  1469.9 ms |  70.3   98.5 ( 59.1  25.9) 1276.3    5.2   19.5 |   0/240 4514 MB
  1     223 ->     18  1557.9 ms |  76.9  114.6 ( 67.2  31.5) 1343.1    6.7   16.5 |  21/240 4119 MB
  2      18 ->     16  1223.4 ms |  70.1  105.8 ( 59.5  31.1) 1023.0    6.5   18.0 |  78/240 3047 MB
  3      16 ->     18   945.3 ms |  86.4  122.2 ( 74.4  32.3)  714.0    9.0   13.6 | 132/240 2031 MB
  4      18 ->    201   933.0 ms |  70.8  106.3 ( 63.8  30.4)  733.2    6.2   16.6 | 124/240 2182 MB
  5     201 ->      1  1034.1 ms |  73.7  103.2 ( 58.7  31.0)  836.8    6.7   13.7 | 109/240 2464 MB
  6       1 ->      1  1488.1 ms |  72.3  110.6 ( 65.1  31.3) 1284.5    8.2   12.6 |  34/240 3875 MB
  7       1 ->      0   482.4 ms |  60.5   90.5 ( 49.5  29.8)  312.9    5.9   12.6 | 193/240  884 MB

8 tokens, 9.13 s, 0.88 tok/s, hit rate 0.360, 23.1 GB read from NVMe
```

`--profile FILE.jsonl` writes the design §13.1 record, one line a token. Its
`hot_bytes` comes out at **8.52 GB**, which is design §2.3's resident-per-token
budget to three digits — summed from the manifest per layer rather than assumed,
so it is a check on §2.3 rather than a restatement of it.

One field in that record is wrong and is not ours: `nvme_util` reads above 1
(6.2 on the first token) because `IoEngine`'s busy counter sums each chunk's own
latency instead of the union of the windows in which at least one chunk was in
flight, and the queue depth is 8. `nvme_gbps` is understated by the same factor.
The IoEngine's own `effective_gbps` (3.0–3.9 GB/s in these runs) is the number
to read until that is fixed.

`--per-layer` prints the same breakdown forty rows deep, which is where design
§13.1 actually wants it — a layer stalling for 200 ms, or what layer 1 and 14's
engram costs, is invisible in the aggregate. A cold layer looks like
`engram 0.00  attn 1.10  stall 32.4  moe 2.51 (gpu 1.36 host 0.67)  0/6  112.9 MB`.

**NVMe is 83% of a cold step.** At a 0.36 hit rate a token misses about 150 of
its 240 experts, which is 2.8 GB. During a stall the drive moves 2.8–3.5 GB/s
against the 4.5 GB/s plateau of design §9.2.1 — the expected shape for six
misses issued per layer and then waited on: the queue depth peaks at 8, but only
inside one layer's burst, and between bursts the drive is idle. Overlapping the
next layer's fetch with this layer's compute is what design §9.4's Planner
thread is for, and it is not running.

**The submit round trip is not the problem.** `Engine::measure_submit_overhead`
times a dispatch that does essentially nothing: **0.13–0.15 ms** on an idle
machine. A decode step makes about 128 of them (40 attention + 80 MoE + 4 engram
+ 4 for the tail), so the one-command-buffer-per-dispatch-group shape costs
about 17 ms a token. Worth fixing — design §7.1 wants one pre-recorded command
buffer per token — but it is 13% of the warm step, not the 30% it looked like
before it was measured.

**The attention path costs 1.30 ms a layer against
[p2_attention.md](p2_attention.md) §5's 1.07 ms.** That benchmark cycled eight
layers, so `mega_mhc.mix` and `gate.score` sat in the 32 MB MALL and it said so.
A real token cycles forty and they do not.

**The MoE is 1.72 ms a layer against [kernel_p2_moe.md](kernel_p2_moe.md)'s
0.68 ms for the same seven slots**, of which about 0.75 ms is host work. Two
causes, both fixable and neither inside the kernel: the bridge asks for one
iteration per submit, so the record and the launch are not amortised the way a
benchmark's are; and `MoeBridgeConfig` still carries P1's specialisation
(`decode_mode = 0`, `h_quant = 1`) rather than the P2 winner the MoE track has
since measured. This is the cheapest 40 ms a token available.

### 5.3 A bigger cache does not help over eight steps

`--cache-gb 48` — 2,700 slots, more than the 1,920 (layer, expert) pairs eight
steps touch — gives a hit rate of **0.349**, no better than 12 GiB. That is not
a bug and not a contradiction of design §9.1.1: consecutive tokens share only
about half their routed experts, so a run eight tokens long never reaches the
steady state the 0.90 figure describes. Eight steps is a correctness harness,
not a cache benchmark; `tools/cache_sim.py` is where hit rate is measured.

---

## 6. The gap to design §13.4

§13.4 predicts 156 ms/token → 6.4 tok/s, as
`48.5 ms non-MoE + 27.3 ms MoE + 80 ms NVMe at h = 0.92`.

| term | §13.4 | warm step (§5.1) | ratio |
|---|---:|---:|---|
| non-MoE (attention + head + collapse) | 48.5 ms | 60.3 ms | 1.24× |
| engram | not in the model | 4.9 ms | — |
| MoE | 27.3 ms | 68.6 ms | **2.5×** |
| **compute** | **75.8 ms** | **133.8 ms** | **1.8×** |
| NVMe at h = 0.92 and the 3.2 GB/s this run gets | 80 ms | ~113 ms | 1.4× |
| **token** | **156 ms** | **≈ 247 ms → 4.0 tok/s** | 1.6× |

So the shortfall is two-thirds MoE and one-third the drive not reaching its
plateau, and the MoE two-thirds is mostly launch and host overhead rather than
the kernel (§5.2).

The engram is a term §13.4 forgot. It is small — 4.9 ms over two layers — but it
is not zero, and half of it is 48 random 4 KiB reads that a prefetcher should
have issued the moment the token was decided (design §9.5).

---

## 7. What is next, in order

1. **The MoE's 30 ms of host work and 35 ms of un-amortised launch.** Take
   `MoeBridgeConfig` to the P2 specialisation the MoE track measured, and record
   both dispatches plus the shared expert into one command buffer instead of
   two submits. Together this is worth roughly 65 ms a token, the largest
   single item on this list.
2. **The compressor and the indexer (§7.4).** They are the only reason this
   decode step is not self-contained, and the attention track has them in
   flight. Landing them deletes `runtime/decode_state.h`'s per-step half.
3. **Prefill (§11).** Then the window KV is ours too and the L3 export is a
   pure oracle rather than an input.
4. **The layer-2 near-tie.** Worth understanding before it is optimised away by
   accident: the question is whether a bf16 residual stream (which is what the
   reference carries, and p2_attention.md §3 already found matters in front of a
   quantiser) would put the sixth and seventh expert back in the reference's
   order. The stream is fp32 here on purpose; this is the first evidence that
   being more precise costs agreement somewhere that matters.
5. **A pre-recorded command buffer per token** (§7.1). 17 ms a token, and it
   needs the layer-indexed address table p2_attention.md §6.2 describes.
6. **Engram prefetch at P2 priority** (§9.5). The 24 row addresses are known the
   instant the token is, so the fetch should overlap the first layer instead of
   blocking layer 1.

---

## 8. Done / not done

**Done**

* `tools/oracle.py --level l3`: prefill state, eight greedy steps, top-64 logits
  per step, and the engram hash tables. 7.1 MB in `tests/data/l3/`.
* `gpu/shaders/engram.slang` (design §7.10, two dispatches) and
  `gpu/shaders/head.slang`'s stage 1, the greedy sampler (§7.11).
* `gpu/vulkan/decode_kernels.{h,cpp}`: a second small runner for those two.
* `runtime/engram.{h,cpp}`: the `NgramHashState` hash in C++, and the 48-read
  P2 fetch behind it.
* `runtime/decode_state.{h,cpp}`: the LOADED half, labelled as such everywhere.
* `runtime/engine.{h,cpp}`: `init_gpu` (both memory paths, the pinned set, the
  dual-path slab pool, every pipeline), `load_decode_state`, `decode_step`,
  `generate`, the §13.1 per-layer timeline, and `layer_probe`.
* `store::ExpertStore::reset`, so the allocator can outlive the store.
* `deepmoe run --prompt-ids … --steps …`.
* `tests/test_decode.cpp`, registered as `suite.decode` (`needs-model`,
  `needs-gpu`).

**Not done**

* Prefill (§11), the compressor and indexer (§7.4), DSpark (§7.12), sampling
  above temperature 0 (§7.11's Philox half).
* `KvCache::snapshot` / `rollback` (§10.2) and prefix persistence (§11.4).
* The eviction guard. `ExpertStore::set_guard` / `set_completed_timeline` exist
  and are never called, because every submit in this path is followed by a wait,
  so nothing in flight can have its slot recycled. The moment a command buffer
  spans a layer boundary that stops being true.
* Per-layer LRU stamps. The Planner is handed the **token** index, so all forty
  layers of one token share a timestamp and the tie-break between them is
  whatever the policy's sort does. It has not misbehaved; it is also not
  designed.
* `IoEngine`'s busy accounting, which makes the profiler's `nvme_util` and
  `nvme_gbps` wrong by roughly the queue depth (§5.2).
* A second L3 prompt. design §12 L3 asks for five; this is one, and the one
  disagreement it produces (§4.2 step 6) is a sample size of one.
