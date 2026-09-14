# P2 step 2 — the first token

> A whole decode step through all forty layers on the GPU, and eight of them in
> a row, checked token by token against `inference/model.py`.
> Companion to [p2_attention.md](p2_attention.md) (the per-stage kernels and one
> layer), [kernel_p2_moe.md](kernel_p2_moe.md) (the expert dispatches) and
> [kernel_p1.md](kernel_p1.md) (the bandwidth matrix).

Status: **2026-09-14** (P2 step 2, Track G), **updated 2026-09-15** (P2 step 3,
Track I: §9–§13). Everything here is the real `DeepSeek-V4.1-Flash`
checkpoint on the real machine. Raw data: `tests/data/l3/`,
`tests/test_decode.cpp`, and the `deepmoe run` transcripts quoted in §5 and
§9–§12.

> **Step 3 in five lines.** Nothing is loaded any more: design §7.4's
> compressor and indexer produce the compressed KV and the top-k list every
> step, and the prompt's state can come from our own slow prefill (§9). L3 is
> **8/8 teacher-forced and 8/8 free-running** on the export's prefill state and
> **7/8 and 6/8** on our own; the one miss is a near-tie that moves with the
> MoE's summation order (§11.3). The warm step is **81.5 ms, 41 submits**
> (from 134 ms, ~128) against §13.4's 75.8 (§10). Decode is bit-reproducible,
> and §4.4's non-reproducibility was a shared shader directory (§11.2). 128
> free-running steps on a 78.8 GiB cache ramp from 1.1 to 5.1 tok/s (§12).
> Sections 0–8 below are the step-2 report as written; where step 3 changed a
> conclusion the section says so.

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
| the compressed KV and the indexer's top-k list | ~~LOADED, per step, from the L3 export~~ **step 3: produced** by design §7.4's kernels, validated every step (§9) |
| the state the prompt leaves behind | ~~LOADED~~ **step 3: loaded from the export's prefill record, or produced by `Engine::slow_prefill`** (§9.5). Chunked prefill is still design §11 / P5 |
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

> **Step 3:** this section's non-reproducibility does not reproduce with this
> step's commit rebuilt on its own, and its numbers match neither kernel set
> that was committed at the time. §11.2 has the measurements and the
> explanation (a shared, concurrently rebuilt shader directory).

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

> **Step 3:** items 1 and 2 are done (§10, §9); item 3 is done the slow way
> (§9.5); item 4 is diagnosed (§11.3); item 5 is replaced by the gate-cut
> buffer of §10.2 (41 submits); item 6 is done (§10.2 item 5).

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

## 8. Done / not done (P2 step 2; superseded by §13)

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

---

# P2 step 3 — nothing loaded, and the warm step at the kernel floor

Status: **2026-09-15**, Track I. Same machine, same checkpoint, same prompt.
Everything below supersedes §2's LOADED rows, §5.1's 134 ms, §4.4 and §8.
Raw data: `tests/data/l3/` (regenerated, a superset), `tests/test_decode.cpp`,
and the `deepmoe run` transcripts quoted inline.

## 9. Nothing loaded

### 9.1 What runs now

| piece | step 2 | step 3 |
|---|---|---|
| compressed KV, per step | LOADED from the export | **ours**: `compressor.slang`, on the four `kv_source_layers` |
| indexer top-k, per step | LOADED | **ours**: window half by the host (`get_window_topk_idxs`), compressed half by `indexer.slang` on the eight `index_source_layers` |
| index-key cache | not held | **ours**, `indexer.slang` stage 2 on a source whose group completed |
| compressor carried group | not held | **ours**, `KvStore`'s `kv_state` / `score_state` |
| the prompt's state | LOADED | either the export's prefill record (**§9.4 (a)**) or **ours, by a slow prefill** (**§9.4 (b)**) |

`Engine::status()` prints one `LOADED` line and it now reads, for (b),
`LOADED none -- the prompt state came from slow_prefill and every decode-step
tensor from design 7.4's kernels`.

### 9.2 `shared_attn`, written down

`model.py` keeps one process-wide `SharedAttentionRuntime` that sources write
and every other layer reads. It has three fields, and they do **not** follow
the same rule, which is the one thing about the wiring that is easy to get
wrong (`Engine::build_ced_plan`, `Engine::run_layer`):

| field | who writes it | when | so layer L reads |
|---|---|---|---|
| `compress_kv` | a `kv_source_layer` (2, 8, 14, 20) | **every** pass, complete group or not | the last kv source ≤ L — static |
| `topk_idxs` | an `index_source_layer` (2, 8, 14, 20, 24, 28, 32, 36) | every pass | the last index source ≤ L — static |
| `index_k` | a kv source | **only when its group completed** | whichever source published LAST — **dynamic** |

The third row is why a ratio-2 source at an even position scores against layer
20's keys from the previous pass, and at an odd one against its own. The
reader's own ratio still decides how much of the cache it sees
(`end_pos // ratio`). Every layer group under an index source has that
source's ratio, so the top-k list can be shared as a buffer, not copied.

Order inside a source layer follows `Attention.forward`: wkv / wgate GEMVs →
state write + pooling + norm (every step) → index key (if the group
completed, off the **pre-RoPE** latent) → indexer q, weights, score, top-k →
the cache write that rotates and FP4-quantises the latent (last, because the
indexer needs it unrotated). Two RoPE tables per source layer: the query at
`pos`, the latent and its key at `pos + 1 − ratio`.

Design §2.1's candidate-block mask (layer 20 selects 2,048 blocks of 8,
layers 24–36 score inside them) is the identity below 16,384 compressed
positions and is not written; `prepare_ced` fails rather than silently
dropping it past that.

### 9.3 The export grew, and stayed the same

`tools/oracle.py --level l3` now also writes, in the prefill record, each kv
source's `cmp_cache`, `index_k` and `cmp_state_{kv,score}` (the index-key cache
cannot be recovered from `cmp_kv`: it comes off the pre-RoPE latent), and per
step and layer `ffn_in`, `gate_scores`, `gate_ids`, `gate_weights`, plus each
layer's `gate_bias` once. 7.1 → 11.4 MB. **All 749 tensors the previous export
had are byte-identical** in the new one — the reference is deterministic —
and the greedy tokens are the same nine.

### 9.4 L3, with every per-step tensor ours

Per step, `tests/test_decode.cpp` compares every layer's compressed KV against
the reference's `cmp_kv` and its top-k list against `topk_idxs`, through
`Engine::effective_kv` (the plane a layer actually read, not its own):

```
step 0  cmp_kv worst cos 0.9999267 (L20) over 38 layers, top-k 40/40 layers identical
step 7  cmp_kv worst cos 0.9983927 (L20) over 38 layers, top-k 40/40 layers identical
```

**The top-k list is identical on every layer at every step.** At this context
that is less than it sounds — `index_topk` is 512 and there are ≤ 72
compressed positions, so the indexer keeps all of them — but the window half,
the offsets, the -1 marks and the source routing are all in that equality.

**(a) prefill state loaded from the export, everything per step ours**

| build | teacher-forced | free-running |
|---|---:|---:|
| step-2 MoE path (two runners, `HQuant=1`) | 7/8 | 6/8 |
| **step-3 MoE path (one 7-slot runner, `HQuant=3`)** | **8/8** | **8/8** |

**(b) our own slow prefill, then the same eight steps — nothing loaded**

| build | prompt → token 3006 | teacher-forced | free-running |
|---|---|---:|---:|
| step-2 MoE path | MATCH, ρ 0.955 | **8/8** | **8/8** |
| **step-3 MoE path** | MATCH, ρ 0.934 | 7/8 | 6/8 |

The disagreement in all four cells is **the same token**: step 6, where the
reference's own margin is 0.95. It moves between cells with nothing but the
MoE's summation order, which is the evidence §11.3 turns into a diagnosis: it
is a near-tie on the precision floor, not a wiring error. Neither (a) nor (b)
is "the" number; the pair is.

### 9.5 The slow prefill

`Engine::slow_prefill` runs the prompt through the decode path, one token at a
time. It is 64 forward passes where design §11 wants one, so it is not how a
prompt will be consumed; it is how the prompt's state becomes ours before the
chunked kernels exist. It is exactly the reference's chunked prefill at this
geometry: at n ≤ window the ring puts token p in slot p and the decode top-k
list marks every slot above p as −1, which is the causal window
`get_window_topk_idxs` builds for query p at `start_pos` 0; a ratio-2 group
completes at odd p and pools tokens {p−1, p} into row p/2 rotated at p−1; and
query p sees `(p+1)/ratio` compressed rows, which is the chunked form's mask.

Against the export's prefill record:

| | worst cos | where |
|---|---:|---|
| window KV, 40 layers | 0.925–0.930 | L18 |
| compressed-KV cache, 4 sources | 0.967 | L14 |
| index-key cache, 4 sources | 0.975 | L14 |

That is well below a single step's 0.999, and **the shape says it is not a
bug**: `deepmoe run --slow-prefill` prints the window KV per prompt position,
and L0 is 0.99999 at every position, L2 0.9956 at p0 and 0.9993 at p63, L6
0.92 at p0 and 0.998 at p63, L39 0.72 at p0 and 0.99 at p63. The error grows
with **depth** and is worst where the context is **thinnest** — position 0
attends to one row and a sink — which is where a gate near-tie is likeliest
to break the other way (§11.3 counts 13 of 40 layers routing one expert
differently at a single well-conditioned step). A uniform error would have
been a bug; this one compounds exactly where routing is least decided.

The run found a real bug on the way: `token_` is the expert cache's LRU clock,
and `slow_prefill` wound it back to 0, which made everything already resident
look newer than what the next layer fetched. The policy evicted the slot it
had just filled and the MoE dispatch found `expert (4, 1) is not resident`.

## 10. The warm step: 134 → 81.5 ms

### 10.1 The breakdown

`deepmoe run --teacher-force --steps 1 --warm 6 --cache-gb 24`, idle machine
(nothing else on the CPU or GPU; the busy runs quoted elsewhere are labelled).
Step 0 repeated at the same position, so every expert is resident. GPU times
are timestamps, read once a token:

| bucket | step 2 (§5.1) | step 3, first cut | **step 3, final** | note |
|---|---:|---:|---:|---|
| attention, dispatches 1–9 + §7.4 | 51.9 | 35.8 | **36.0** | GPU; 0.90 ms/layer. Most of the drop is Track F's kernels, not this track |
| MoE, GPU (A + h-quant + B, 7 slots) | ~40 | 29.3 | **29.4** | 0.73 ms/layer; kernel_p2_moe's 7-slot pair is 0.60–0.73 |
| MoE, host | ~28 | 6.4 | **1.0** | x 0.0 + act_quant 0.4 + seven table rows 0.6 |
| engram (fetch + 2 dispatches) | 4.9 | 3.8 | **3.9** | |
| tail: collapse + head + argmax | 8.4 | 5.8 | **5.8** | the head alone is 5.65 |
| NVMe stall | 0.2 | 0.4 | **0.4** | 240 planner lookups on an all-hit step |
| other | — | 5.2 | **5.0** | record 0.4 + submit 1.6 + bind 1.0 + fence latency ≈ 2.0 |
| **total** | **134.0** | **86.7** | **81.5** | **12.3 tok/s warm** |
| **submits a token** | **~128** | **43** | **41** | |

"First cut" is the commit that introduced the gate-cut buffer and the 7-slot
MoE; "final" adds items 5 and 6 of §10.2. The final repeats measured 81.5,
81.9, 81.8, 83.2, 81.7 and 83.2 ms, then 89.6 and 91.3 as something else on the
machine woke up (submit time doubled; the GPU times did not move).

### 10.2 What changed

1. **The command buffer is cut only at the gate.** A layer's MoE and the next
   layer's attention travel in one submit — the host has to see the ids
   between a gate and its MoE, and nowhere else. 40 layers + the tail = 41
   submits; the two engram layers cost one more each until item 5.
   **The §7.1 gate is a semaphore wait inside that submit** on the residency
   timeline, which the host signals; completion signals a **second** timeline
   the host fences on, because one semaphore cannot be written by both
   (`gpu::Submission::signal_timeline`, additive). On an all-hit layer the
   host has already signalled before it submits, so the wait is free.
2. **One MoE runner, seven slots.** Six FP4 routed experts and the FP8 shared
   expert in the same two dispatches (`Fp8Slots`, the shared expert at a
   spare table index 384), `HQuant = 3` per kernel_p2_moe.md §12. Dispatch B
   sums all seven, so the host's fp32 add is gone, and `y` is
   device-addressable and read by the next layer's hc_post by address, so the
   two 20 KB write-combining reads a layer are gone (`MoeRunner::record_into`,
   `MoeRunner::y_address`, additive).
3. **The host act_quant is vectorised.** gpu/shaders/attn_common.slang's
   branch-free `fp8_round` transcribed, then F16C. `GpuMoeBridge::create`
   self-checks it against `cpu::act_quant_block` + `cpu::float_to_fp16` on
   5,120 values covering subnormals, ties and both overflow edges and refuses
   to start on a single differing bit.
4. **Three per-call allocations that never freed.** `DecodeLayer::run_attention`,
   `run_close` and `EngramRunner::run` each allocated a `VkCommandBuffer` per
   call from a pool that is only destroyed at shutdown — 40 a token and never
   returned. Each keeps one now. (`AttnRunner::dispatch_now` and
   `DecodeRunner::dispatch_now` still do this; neither is on the token path
   any more, and both belong to other tracks.)
5. **The engram rows are fetched at the start of the token** (design §9.5),
   overlapping layer 0's attention, into one pair of row planes per engram
   layer; the engram layer's close, its two dispatches and its attention then
   join the open buffer. `LayerStep::bind_close` keeps that layer's `bind`
   from overwriting the MhcClose slice the close has already been recorded
   against. And the RoPE tables are cached: a step needs three, and forty
   `bind`s were computing one each.
6. **The FFN input lives in cached host pages.** The one activation the host
   reads every layer -- the FFN half's normed input, which the gate scores and
   the host act_quants into the MoE's `x` -- was in path A, i.e.
   DEVICE_LOCAL|HOST_VISIBLE, i.e. write-combining, and its 20 KB read
   measured **155 µs a layer, 6.2 ms a token**: §3.3's rule again, one level
   up from the loop it was first written about. It now lives in a path-B
   buffer -- ordinary pages imported for the GPU -- where the same read is a
   cached memcpy (`DecodeLayer::set_ffn_input`). The expert table rows come
   out of the store under one lock each, and the shared expert's rows are
   resolved once per layer for the bridge's life.

### 10.3 The gap to design §13.4

design v0.8 §13.4 puts the kernel floor at 72–75 ms (75.8 with the real
shared expert). Against it:

| | floor | measured | |
|---|---:|---:|---|
| non-MoE (attention + §7.4 + tail) | 42.2 | 41.8 | at the floor |
| MoE GPU | 25.0–27.3 | 29.4 | the 7th slot is a real 23.6 MB fp8 expert, not a 12.5 MB FP4 stand-in |
| MoE host | — | 1.0 | the model has none |
| engram | 4.9 | 3.9 | |
| stall + other (submit / record / fence / bind) | — | 5.4 | the model has none |
| **total** | **75.8** | **81.5** | **+7.5%** |

What is left is structural: the host has to see every layer's gate, which is
40 host round trips a token (~1.6 ms of submit and ~2 ms of fence latency),
and the MoE's input still crosses to the host and back once a layer because
the kernels take fp16 `x` and no kernel produces the fp8 round trip of it.
A GPU-side act_quant into the MoE's `x`, and the pointer table moving onto
the GPU so the gate's ids index it directly, are what take those rows to
zero; both are kernel-track changes. The MoE GPU excess over the floor is
inside kernel_p2_moe's own 0.60–0.73 ms spread for a 7-slot pair.

## 11. Bugs

### 11.1 `nvme_util` > 1

The IoEngine's own `busy_ns` was already the union of in-flight windows. The
**profiler** was fed each request's latency from `IoEngine::finish`, which
counts overlapping requests once each — up to 8× the union at QD 8, which is
§5.2's `nvme_util` of 6.2 and its `nvme_gbps` at a sixth of the IoEngine's.
`handle_completion` now reports each in-flight window as it closes.
Separately, `EngramRunner` noted its bytes as miss bytes a second time on top
of the IoEngine's count.

### 11.2 Bit-reproducibility

`deepmoe run --determinism N` runs one warm step N times and hashes, at every
layer, the stream after attention, the FFN input, the gate scores and the MoE
output, plus all 129,280 logits.

| build | within a process | across processes |
|---|---|---|
| step 3 (`HQuant = 3`) | 4 runs, all 160 layer hashes and the logits identical | 3 processes, logits hash identical (`a919aaf6…`) |
| step 3 with `DEEPMOE_MOE_HQUANT=1` | 3 runs identical | 2 processes identical (`e82d620e…`) |
| **step 2's own commit (`c9bf0e1`), rebuilt in isolation** | 4 runs identical, including the cold first | 3 processes, 8 and 24 GiB caches, identical (`6fcdc1cb…`) |
| step 2's runtime + Track F's committed kernels (`0964e5e`) | 2 runs identical | — |

**Decode is bit-reproducible, and so was step 2's code.** The §4.4
observation — layers 20 and 39 moving in the 7th and 3rd decimal between runs —
does not reproduce with step 2's commit rebuilt on its own. Neither do its
numbers: step 2 reported a step-0 margin of **7.0927**; its commit, rebuilt,
gives **6.7477**, and with Track F's committed kernels **6.9389**. So the runs
§4 quotes executed neither kernel set. The explanation that fits all of it is
the shared `build/shaders` directory: three tracks were rebuilding `.spv` files
into it while step 2 ran its test, so two runs of the same binary could load
different kernels, and only the layers whose kernels had changed would move.
This track builds into its own `build-i/` for that reason, and a comparison
that means anything has to say which shader directory it loaded.

### 11.3 The step-6 token: precision at a near-tie, not a kernel

`deepmoe run --gate-report` runs the teacher-forced steps and, at every layer,
compares our gate against the reference's using the new per-step export:

```
step 6  in 1 -> 201 (ref 201 MATCH, ref margin 0.9485, ours 1.3957) | 40 layers, 11 route differently
        | worst in-cos 0.975695, |gpu-cpu(ours)| 2.36e-04, |cpu(ref)-ref| 1.43e-06
    L14 5/6: ours+210 ref+92   in-cos 0.999160  |dscore| 1.19e-02  6th-7th margin ref 3.18e-04
    L30 5/6: ours+49  ref+210  in-cos 0.989119  |dscore| 5.29e-02  6th-7th margin ref 1.75e-04
    L33 5/6: ours+67  ref+143  in-cos 0.985240  |dscore| 2.20e-01  6th-7th margin ref 8.35e-04
    L38 4/6: ...               in-cos 0.980238  |dscore| 3.06e-01  6th-7th margin ref 8.26e-03
    ...
```

Three measurements separate precision from a kernel bug:

* **the gate arithmetic is right.** `cpu/gate.cpp` on the **reference's** FFN
  input reproduces the reference's 384 scores to **1.4e-6**; our GPU gate on
  **our** input matches `cpu/gate.cpp` on the same input to ~1e-6 on every
  expert whose logit is not deeply negative (1.1e-6 at L0, 1.6e-6 at L20).
* **the input has drifted, by an amount larger than the ties.** At the
  layers that route differently, our FFN input is at cos 0.976–0.9992 to the
  reference's, which moves the scores by 1e-2 to 0.3. The reference's own gap
  between its sixth and seventh (score + bias) at those layers is
  **1.7e-4 to 8e-3**.
* **it happens at every step, including the ones that match.** 13 of 40
  layers route one expert differently at step 0, 25 at step 1, 11 at step 6.
  Whether the token survives it is decided by how the swaps accumulate against
  the reference's own logit margin — 0.95 at step 6, 7.1 at step 0.

So it is fp16/bf16-versus-fp32 activation error crossing near-ties, and
§9.4's table moving the one disagreement between cells with nothing but the
MoE's summation order is the same fact seen from the other side.

**One real kernel defect came out of it, and it is not the cause.** The
largest |gpu − cpu| is 1.5e-4 and it sits on experts the GPU scores **exactly
0**: `gate.slang` computes softplus as `log(1 + exp(z))`, which in fp32 is 0
below z ≈ −16 and loses digits from z ≈ −10, where `cpu/gate.cpp` and torch
use `log1p(exp(−|z|)) + max(z, 0)`. It only touches experts whose score is
below ~1e-3, far from any top-6 boundary, but it is a wrong number: the fix is
the stable form, in `gpu/shaders/gate.slang` (Track J's).

**Every compressed row written per step tracks the input.** The same report
prints the row each kv source wrote that step against the reference's: ratio 2
(layers 2/8/14) 0.987–0.998, ratio 1 (layer 20) 0.920–0.998. The ratio-1 path
is bit-exact on the reference's own input (p2_attention.md §9.2), so 0.92 is
what drift alone does to a row; the ratio-2 pooling, which that section could
only check against a CPU transcription, sits inside the same envelope. A
two-step export of `attn_norm_out` at a source layer would close that last gap
with an equality instead of an envelope.

## 12. Cache policy, and 64 steps

### 12.1 The Planner is the simulated global LRU

`tools/cache_sim.py`'s `LRU` is an `OrderedDict` that moves each key to the
end **as it is touched**, walking the trace token by token, layer by layer, in
the gate's order, and a miss is admitted (evicting the oldest if full) before
the next key is looked at. The Planner did two things differently: every
access of a token got the same stamp (the token index), so within a token the
slot index decided which of 240 equally-old experts left first; and all of a
layer's evictions were decided before its later hits had been touched.

It now keeps an access clock that ticks once per lookup and once per admit, in
gate order, and a miss evicts the single oldest resident (`ExpertStore::evict_lru`,
a scan under the lock instead of a copy and a sort) and is admitted before the
next key. A fill's slot is stamped with the tick it was asked for, not the
moment its last run lands. `IoRequest::issue_token` stays the token.

### 12.2 The cache sizes itself from both paths

Step 2 capped the auto-sized cache at path A's heap size (74 GiB) and forgot
that the pinned set lives there too, so it was ~64 GiB of experts and never
touched path B. `Engine::init_gpu` now sizes each path by what bounds it —
path A by its heap less the pinned set and a gigabyte of everything else
(commit, not physical memory), path B by physical memory free now less a 12 GiB
floor — and the sum by available commit less the pinned set and an 8 GiB
margin, and says so in one log line before allocating anything:

```
engine: cache budget auto -> 80.26 GiB (path A 64.26 GiB after 9.17 GiB pinned,
        path B 16.00 GiB of 51.19 GiB physical free; 136.19 GiB of commit available)
engine: expert cache 4500 slots, 78.83 GiB (36 slabs on path A, 9 on path B)
```

Path B's 16 GiB is a ceiling, and it is there because the obvious bound is
wrong. The first version sized B by physical memory alone and got 38 GiB with
50 GB free; the import of the slab at ~33 GiB returned
`VK_ERROR_INVALID_EXTERNAL_HANDLE`, the pool stopped cleanly -- and the **next
submit found the device lost**, which surfaced as the residency timeline
reading UINT64_MAX. A second run failed the same way at 15.8 GiB while another
track's GPU test was holding host-heap memory. `VK_EXT_memory_budget` reports
the host heap as 35.4 GiB with **0 B in use** whatever is running, so it
cannot see the other process either. B is now bounded by the host heap less
10 GiB, the budget less 4 GiB, physical memory less 12 GiB, and 16 GiB,
whichever is least; `--cache-gb` still asks for more and owns the risk; and a
lost device is reported as one. bench/heap_capacity's 100 GiB is a number for
an otherwise idle machine and a process that does nothing after allocating.

### 12.3 64 steps from the L3 prompt

`deepmoe run --steps 128 --cache-gb 0`, idle machine, free-running from the
L3 prompt with the prompt's state loaded and everything per step ours. The
first eight tokens are the reference's (8/8); after that the model repeats the
prompt, which is what a greedy decode of this prompt does and which flatters
the hit rate a little, since it re-routes through experts it has already
fetched. Every eight tokens:

| steps | ms/token | tok/s | hit rate | NVMe stall ms | `nvme_util` max | NVMe GB/s |
|---|---:|---:|---:|---:|---:|---:|
| 0–7 | 880 | 1.14 | 0.349 | 771 | 0.91 | 3.87 |
| 8–15 | 633 | 1.58 | 0.559 | 527 | 0.85 | 3.87 |
| 16–23 | 480 | 2.08 | 0.693 | 375 | 0.82 | 3.83 |
| 24–31 | 392 | 2.55 | 0.733 | 296 | 0.78 | 4.32 |
| 32–39 | 342 | 2.93 | 0.779 | 237 | 0.76 | 4.32 |
| 40–47 | 303 | 3.30 | 0.806 | 206 | 0.74 | 4.31 |
| 48–55 | 242 | 4.14 | 0.874 | 136 | 0.66 | 4.25 |
| 56–63 | 270 | 3.71 | 0.843 | 173 | 0.71 | 4.27 |
| 64–71 | 289 | 3.46 | 0.845 | 198 | 0.75 | 3.68 |
| 72–79 | 259 | 3.86 | 0.878 | 158 | 0.68 | 3.65 |
| 80–87 | 246 | 4.06 | 0.881 | 155 | 0.67 | 3.66 |
| 88–95 | 292 | 3.43 | 0.850 | 188 | 0.71 | 3.73 |
| 96–103 | 210 | 4.77 | 0.901 | 119 | 0.62 | 4.03 |
| 104–111 | 204 | 4.90 | 0.905 | 112 | 0.63 | 4.00 |
| 112–119 | 210 | 4.77 | 0.909 | 107 | 0.61 | 4.01 |
| 120–127 | 196 | 5.10 | 0.915 | 103 | 0.54 | 4.00 |

**The 64-step run is 2.26 tok/s at a hit rate of 0.705** (28.1 s); it is the
first half of this table, and three separate 64-step runs -- 5,100, 5,200 and
4,500 slots, idle and busy (the busy one 1.82 tok/s) -- gave hit rates
identical to the third decimal, because greedy routing and an exact LRU are
deterministic and 64 tokens fill none of those caches. Over all 128 it is
2.94 tok/s at 0.795, and over the last 32 **4.88 tok/s at 0.907**.

Against design §13.4's ~7 tok/s: that figure is for h ≈ 0.92–0.95 in steady
state with 5,711 slots, and this curve is still ramping at step 128. What the
run does show is that the token time follows the hit rate the way the model
says it should -- every token that reached h ≥ 0.94 took 105–159 ms, i.e.
**6.3–9.5 tok/s** -- so the gap at step 128 is the ramp, not the compute. The
stall per miss is ~5 ms (a 20 MB expert at the drive's ~4 GB/s), so 7 tok/s
(143 ms) needs about 12 misses a token, h ≈ 0.95.

Two things the profile says that it could not before: `nvme_util` is at most
0.91 (it read 6.2 in §5.2, see §11.1) and `nvme_gbps` agrees with the
IoEngine's own 3.7–4.3 GB/s. And **the MoE's GPU time rises on steps with
misses** -- 37–42 ms against 29.4 warm -- which is the first GPU touch of pages
the drive has just written; it is a cost of the stall that the stall bucket
does not see.

## 13. Done / not done, after P2 step 3

**Done**

* design §7.4's compressor and indexer on the decode path, with
  `shared_attn`'s three routing rules; the per-step compressed KV and top-k
  list validated every step (top-k identical on 40/40 layers at 8/8 steps).
* The prompt's state from the export's prefill record (window, compressed KV
  cache, index keys, compressor state) — or from `Engine::slow_prefill`, with
  nothing loaded at all. L3: (a) 8/8 and 8/8, (b) 7/8 and 6/8 on this build;
  the one disagreement is §11.3's near-tie.
* The warm step at **81.5 ms / 41 submits** (from 134 ms / ~128): one buffer
  per gate, the residency gate as a semaphore wait, a second fence timeline,
  one 7-slot MoE with `HQuant = 3` and `y` on the GPU, a vectorised host
  act_quant, the FFN input in cached host pages, the engram prefetched, GPU
  timestamps for the breakdown.
* Bugs: the profiler's NVMe busy time is the in-flight union; the engram's
  double-counted bytes; three `VkCommandBuffer` leaks on the token path; the
  LRU clock wound back by `slow_prefill`; the auto-sized cache ignoring path B
  and the pinned set, and then losing the device when path B was sized by
  physical memory; a failed expert read that released its slot silently.
* Bit-reproducibility shown (§11.2), and step 2's non-reproducibility traced
  to a shared shader directory rather than to a kernel.
* The step-6 token diagnosed as precision at a near-tie (§11.3), with a
  per-layer tool that will say so again for the next prompt.
* The Planner is `tools/cache_sim.py`'s global LRU, step for step; the cache
  sizes itself to 78.8 GiB; 128 free-running steps ramp from 1.1 to 5.1 tok/s
  as the hit rate goes from 0.35 to 0.92.
* `deepmoe run --slow-prefill --loaded-ced --warm --determinism --gate-report`;
  free-running past the export's last step.

**Not done**

* A GPU act_quant into the MoE's `x` and a GPU-side pointer table — the two
  rows of §10.3 the model does not have. Both need kernels.
* design §7.9.3's "split dispatch A only" when an expert is late. The buffer
  structure is ready for it (the MoE is the first thing in a gated submit);
  the Planner still drains all of a layer's I/O before signalling.
* The eviction guard. Still safe for the reason §8 gave — nothing is in flight
  when the Planner evicts — and it stops being safe the moment the item above
  lands.
* A 5,200-slot run once failed at step 117 with an expert not resident at the
  MoE dispatch, on a cache whose path B was 28 GiB -- near the import limit of
  §12.2. It did not recur at 4,500 slots over 128 steps with evictions, and a
  failed read now names itself, but the cause was not caught in the act.
* design §2.1's candidate-block mask past 16,384 compressed positions; chunked
  prefill (§11); a second L3 prompt; DSpark on this path.
* `gate.slang`'s softplus (§11.3) — Track J's file.
* `AttnRunner::dispatch_now` and `DecodeRunner::dispatch_now` allocate a command
  buffer per call; off the token path now, but still leaks for their callers.
