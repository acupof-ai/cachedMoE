# P2 — the non-MoE decode path

> The design §7.2–§7.8 and §7.11 kernels, validated stage by stage against
> `inference/model.py`, plus one whole decoder layer end to end.
> Companion to [kernel_p1.md](kernel_p1.md) (the MoE kernels and the bandwidth
> matrix) and [route_trace.md](route_trace.md) (routing and the cache).

Status: **2026-09-14**. Everything measured here is on the real
`DeepSeek-V4.1-Flash` checkpoint, one decode token at position 64 of a 64-token
prefill. Raw data: `tests/data/l2/`, `tests/data/l2x/`,
`bench/results/attn_p2.csv`.

**§0–§8 are step 1** and are left as they were written, because the numbers in
them are what the numbers in step 2 are measured against. **§9–§12 are step 2**:
the §7.4 compressor and indexer, which §6 listed as "loaded", and the bandwidth
items §7 opened. Where the two disagree — §5's table, §6's status table, §7's
list — step 2 is the current one. **§13 is P3 Track J** (the LDS bank-conflict
fix, the K-split GEMVs and the tiled sparse attention: one layer 893 → 696 µs)
and supersedes §10’s bandwidth table and §11’s list.

---

## 0. What this says in one page

Nine Slang kernels, eighteen pipelines, a pinned weight store, an oracle that
exports the reference's own per-stage tensors, and a runtime that executes one
decoder layer. Correctness first:

| stage (design §) | worst of 7 layers, cosine vs the oracle |
|---|---|
| mega-mHC mixes + Sinkhorn (§7.2) | 1.000000000 (pre/post/comb to 6e-7) |
| mega-mHC `attn_norm` output (§7.2) | **1.000000000, bit-exact on all seven** |
| `wq_a` (§7.3) | 0.999998624 |
| `wq_b` + `q_norm` + RoPE (§7.3) | 0.999998322 |
| `wkv` + `kv_norm` + RoPE (§7.4) | 0.999998342 |
| `wkv` after the fp8 cache write (§7.4) | 0.999910658 |
| sparse attention + inverse RoPE (§7.5) | 0.999997368 |
| `wo_a`, grouped (§7.6) | 0.999998577 |
| `wo_b` (§7.6) | 0.999998575 |
| `hc_post` into the stream (§7.7) | 0.999998535 |
| gate scores (§7.8) | 0.999999996, **6/6 experts on every layer** |
| `head` (§7.11), vs an fp64 CPU dot | 1.000000000 |

and the whole layer, chained, with only the block input and the prefill's KV
golden:

| | `ffn_norm` | gate | MoE out | **block out** |
|---|---|---|---|---|
| layer 0 | 0.999933 | 6/6 | 0.999778 | **0.999935** |
| layer 39 | 0.999986 | 6/6 | 0.999977 | **0.999980** |

design §12 L2 asks for cosine ≥ 0.999 on a layer output.

Speed, idle machine, `--layers 8 --iters 64`: **1.07 ms for one layer's
dispatches 1–9**, 48.5 ms for 40 layers plus the head. design §3.4's model for
the resident weights is 0.97–1.29 ms a layer, so the path is inside its budget,
but two kernels are well under the memory system's ceiling and §5 says which.

**Five places the reference forced a change to design §7.** They are in §2.
One of them (the activation quantisation) is not a precision detail — following
the design text computes a different function.

---

## 1. The L2 oracle

`tools/oracle.py --level l2` writes `tests/data/l2/`: 5.1 MB, seven layers,
one decode step at position 64.

### 1.1 How it is produced

It runs `inference/model.py` **unmodified**, behind the six CPU kernel shims
`tools/dsref.py` supplies, over a 64-token prefill and then one decode step.
Nothing is re-derived. Every exported tensor is either

* an `nn.Module` forward hook's input or output (`wq_a`, `q_norm`, `wq_b`,
  `wkv`, `kv_norm`, `attn_norm`, `ffn_norm`, `wo_b`, the compressor's norm, the
  indexer's `wq_b` / `wk` / `k_norm` / `weights_proj`), or
* an argument or result of one of the module-level functions the reference
  routes through: `act_quant`, `fp4_act_quant`, `sparse_attn`,
  `Block.hc_mixes`, `Block.hc_post`.

RoPE is the one stage with no capture point: `apply_rotary_emb` rewrites its
argument in place and returns nothing anyone stores. It is captured by
difference — the hook clones its output (pre-RoPE) and keeps the live tensor;
reading that same tensor again after `Attention.forward` returns gives the
post-RoPE value, because `unflatten` and `x[..., -64:]` are views of the very
storage the hook saw. The same trick gives the post-inverse-RoPE output.

### 1.2 Two passes, not prefill-then-decode per layer

`model.py`'s `shared_attn` is a process-wide singleton it deliberately never
resets ("every source writes before its consumers read"). During a decode step
a ratio-2 source layer whose compressed group is incomplete at that position
does not publish, and reads **the previous forward's** index keys — layer 20's,
not its own. That is what the reference does, so the export reproduces it: a
complete prefill pass runs first, and the decode pass starts where it ended.
Only the ~200 KiB of per-layer KV buffers are carried across; each `Block` is
rebuilt from the shards for the decode pass (5.1 GB of attention weights,
about 0.5 s a layer).

Cost of the whole export: 258 s, almost all of it the prefill MoE (~200
distinct experts a layer).

### 1.3 What is in it

Per layer, one token, in design §7.14 order. `dtype` is the dtype the
**reference** held, which matters — see §3.

| name | dtype | shape | what it is |
|---|---|---|---|
| `block_in` | bf16 | [4, 5120] | the residual stream entering the Block |
| `engram_out` | bf16 | [4, 5120] | after the engram write (layers 1, 14 only) |
| `attn_resid_in` | bf16 | [4, 5120] | `residual` for the attention `hc_post`; == `block_in` without engram |
| `pre_mix_in` | f32 | [4] | the previous sublayer's `pre`, which this one's `hc_pre` consumes |
| `attn_pre/post/comb` | f32 | [4], [4], [4,4] | `hc_mixes` on the attention half |
| `attn_hc_pre_out` | bf16 | [5120] | `hc_pre(x, pre_mix_in)`, before the norm |
| `attn_norm_out` | bf16 | [5120] | **the mega_mhc output**, and `wq_a`/`wkv`'s input |
| `wq_a_out` | bf16 | [1280] | before `q_norm` |
| `qr` | bf16 | [1280] | after `q_norm` |
| `q_pre_rope`, `q` | bf16 | [64, 512] | before and after RoPE |
| `wkv_out` | bf16 | [512] | before `kv_norm` |
| `kv_pre_rope` | bf16 | [512] | after `kv_norm`, before RoPE |
| `kv_pre_quant` | bf16 | [512] | after RoPE, before `act_quant` |
| `kv` | bf16 | [512] | **after** the fp8 quantisation, i.e. what the cache holds |
| `kv_fp8`, `kv_scale_e8m0` | u8 | [512], [16] | the same thing as the bytes the ring stores |
| `latent_*`, `index_*` | — | — | the compressor and indexer, on source layers |
| `win_kv` | bf16 | [128, 512] | the window ring, as `sparse_attn` sees it |
| `cmp_kv` | bf16 | [n, 512] | the compressed half, post-fp4-quantisation values |
| `topk_idxs` | i32 | [n_kv] | indices into `cat([window, compressed])`; −1 = empty |
| `attn_sink` | f32 | [64] | one logit a head |
| `attn_out` | bf16 | [64, 512] | `sparse_attn` output, before the inverse RoPE |
| `attn_out_irope` | bf16 | [64, 512] | after it; `wo_a`'s input |
| `wo_a_out` | bf16 | [8192] | `wo_b`'s input |
| `wo_b_out` | bf16 | [5120] | the attention sublayer's output `a` |
| `attn_block_out` | bf16 | [4, 5120] | after the attention `hc_post` |
| `ffn_pre/post/comb`, `ffn_hc_pre_out`, `ffn_norm_out` | | | the FFN half, same shape as the attention half |
| `gate_scores`, `gate_bias` | f32 | [384] | before and the `noaux_tc` bias |
| `gate_top6_ids`, `gate_top6_weights` | i32, f32 | [6] | after selection and normalisation × 1.5 |
| `moe_shared_out`, `moe_out` | f32 | [5120] | the shared expert alone, and routed + shared |
| `block_out` | bf16 | [4, 5120] | the stream leaving the Block |

The quantisation points are pinned, not assumed: the exporter recomputes the
fp8 E4M3 / UE8M0 bytes and the fp4 E2M1 / E4M3 bytes from the pre-quantisation
tensor and **asserts** they dequantise back to exactly what the reference
produced. That is what fixes the `fast_round_scale` rounding direction, the
E4M3 round-to-nearest-even rule and the FP4 nibble order in one place.

Layers exported: 0 (window only), 1 (engram, window only), 2 (ratio-2 kv and
index source), 13 (a plain reuse layer), 14 (engram plus source), 20 (the first
decoder — ratio 1, candidate source), 39 (the last).

---

## 2. What the reference forced on design §7

### 2.1 Every fp8 `Linear` quantises its activation. §6 only says the MoE does

`inference/model.py`'s `linear()` dispatches on the **weight** dtype, and for
both fp8 and fp4 weights it runs `act_quant(x, 32, "ue8m0", e8m0)` first — the
activation is rounded onto the FP8 E4M3 grid with a per-32-element
power-of-two scale before the GEMM. design §6 records that round trip only for
the MoE intermediate `h` (§7.9 v0.6). It applies equally to **`wq_a`, `wq_b`,
`wkv`, `wo_b` and the indexer's `wq_b`**.

This is not "a little less precise". E4M3 has three mantissa bits, so an
element moves by up to 6%; skipping it computes a different function.

Cost of doing it: zero. With `LanesPerRow = 32` and a 32-element UE8M0 block,
one lane already owns exactly one whole block of x, so the block amax is a
register reduction over values it has loaded anyway.

**`wo_a` is the exception.** Its module is declared bf16 (convert.py would have
dequantised it; "an fp8 grouped GEMM would halve the memory") and
`Attention.forward` reaches its weight through an `einsum`, not `linear`. So
the checkpoint stores fp8 bytes but the arithmetic is bf16 × bf16 with **no**
round trip. `gpu/vulkan/attn_kernels.cpp` compiles exactly one stage with
`ActQuant = 0`.

The full picture, from the manifest:

| weight | dtype stored | act_quant? |
|---|---|---|
| `wq_a`, `wq_b`, `wkv`, `wo_b`, `indexer.wq_b`, `shared_experts.w{1,2,3}` | fp8 E4M3 + 32×32 UE8M0 | **yes** |
| `wo_a` | fp8 E4M3 + 32×32 UE8M0 | **no** (einsum, declared bf16) |
| `compressor.{wkv,wgate,norm}`, `indexer.{wk,k_norm,weights_proj}`, `gate.weight`, every norm | bf16 | no |
| `head` | bf16, promoted to fp32 | no |

### 2.2 RoPE pairs ADJACENT elements, not (d, d+32)

design §7.3 says "rows with d ≥ 448 rotate in pairs (d, d+32)".
`apply_rotary_emb` does `view_as_complex(x.unflatten(-1, (-1, 2)))`, so within
the last `rope_head_dim` = 64 dims the pairs are **(448 + 2j, 448 + 2j + 1)**
and frequency j drives pair j. Following the design text rotates the wrong
pairs — invisible at position 0 and growing with the context.

### 2.3 `hc_post` contracts comb's FIRST index

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```
sums over `dim=2`, which is comb's first index, so
`out[j] = post[j] * a + Σ_i comb[i][j] * residual[i]`. comb transposed against
the obvious reading.

### 2.4 The RoPE base and YaRN are per layer, not global

A layer with `compress_ratio == 0` (layers 0, 1, and the DSpark blocks) uses
`rope_theta` = 10000 with **YaRN off** (`original_seq_len = 0` inside
`Attention.__init__`). Every other layer uses `compress_rope_theta` = 160000
with YaRN over 65536. `runtime/rope.h` takes the ratio, not a global.

### 2.5 Sinkhorn ends on a column normalise

Already in design §2.4 from route_trace.md §11.1, restated because it is a
one-line temptation: `gpu/shaders/mega_mhc.slang` must not gain a trailing row
normalise. The exported `attn_comb` agrees with the reference to 6e-7, which
is the check that it did not.

---

## 3. Precision: whose bf16 is it

`generate.py` runs under `torch.set_default_dtype(torch.bfloat16)`, so in the
reference the residual stream, every `Linear` output and the attention inputs
carry **eight** mantissa bits. Our kernels accumulate and carry fp32. design §6
sanctions being more precise ("一般情况下我们比参考实现精度更高"), and for most
stages the entire residual against the oracle is the reference's own bf16
rounding: relative L2 sits at 1.6e-3, which is half a bf16 ulp.

**But being more precise directly in front of a quantiser costs agreement and
buys nothing.** `RMSNorm.forward` ends in `.to(dtype)`, so its result is bf16,
and that bf16 value is what `act_quant` sees. Feeding `act_quant` a wider value
puts the block amax in a different power-of-two bracket, which re-rounds the
whole 32-element block one grid step away from the reference. Measured on
`wq_b`:

| | relative L2 vs the oracle | cosine |
|---|---|---|
| fp32 into `act_quant` | 1.5e-2 | 0.999896 |
| bf16-rounded, as the reference | **1.8e-3** | **0.999998** |

and `mega_mhc`'s norm output went from 2.3e-3 to **bit-exact**. So every
RMSNorm result is rounded to bf16 (`bf16_round` in `attn_common.slang`); the
residual stream stays fp32, where the extra bits feed no quantiser.

### 3.1 Why the criterion is cosine and relative L2, not max relative error

design §12 **L1** uses `max|Δy| / max|y|`, which is right there: L1 compares one
expert FFN against an fp32 oracle, so one outlying element is a real bug. Here
the reference is bf16 and fp8. One element landing on the other side of a
rounding boundary moves by up to half an ulp of **its own** magnitude — 0.4%
for bf16, 6% for E4M3 — and `max|Δ| / max|y|` reports that as though it were an
error in the whole vector. `wkv post-fp8` is the clearest case: 4 to 9 of its
512 E4M3 bytes differ from the reference's, `rel` reads 2–5e-2, relative L2 is
7e-3 and cosine is 0.99991. `tests/test_gpu_attn.cpp` prints both and gates on
cosine and relative L2.

The bytes themselves are reported, not asserted. `tests/data/l2`'s
`kv_fp8` / `kv_scale_e8m0` exist so the difference is visible: of 512 bytes,
4–9 differ and 0 of 16 scale bytes do. The re-encode check in the same test —
128 × 512 window values re-encoded and dequantised with `cpu::act_quant_block`,
`max|Δ| = 0.000e+00` — is the evidence that the rounding rule itself is right.

---

## 4. The kernels

Nine `.slang` files, eighteen pipelines. `gpu/shaders/attn_common.slang` holds
the FP8 E4M3 decode table and rounding, `fp8_gemv.slang` the GEMV core; both
are deliberately separate from `moe_common.slang`, which is the FP4 expert path
and another track's.

| stage | shader | dispatches | note |
|---|---|---|---|
| mega-mHC | `mega_mhc` | 3 × 3 | see §4.1 |
| `wq_a` | `wq_a` | 1 | the generic fp8 GEMV |
| `wq_b` | `wq_b` | 1 | `q_norm` and RoPE fused |
| `wkv` | `wkv` | 2 | `kv_norm`, RoPE, fp8 ring write |
| sparse attention | `sparse_attn` | 2 | see §4.2 |
| `wo_a` | `wo_a` | 1 | grouped, `ActQuant = 0` |
| `wo_b` | `wo_b` | 1 | the widest activation, K = 8192 |
| gate | `gate` | 2 | scores, then a workgroup argmax |
| head | `head` | 1 | bf16 |

**Weights arrive by `buffer_device_address`, not descriptors.** The pinned set
is 17.7 GB across many sub-2-GiB regions, and one descriptor per tensor per
layer would be thousands of them rewritten every token. Every kernel binds
exactly one descriptor: a 32-slot slice of a shared `uint64_t` address table.
Write addresses, push dimensions, dispatch — the same idea as design §5.3's
expert pointer table.

### 4.1 mega_mhc is three dispatches, not one

design §7.2 wants one. It cannot be:

* `hc_fn` is 1.97 MB and a single workgroup reads it at one CU's rate;
* both RMS statistics are whole-vector reductions, which need a grid-wide
  barrier, which is a dispatch boundary.

So: stage 0 over `dim/256` workgroups does `hc_post`, `hc_pre` and the two
partial sums of squares; stage 1 over 24 workgroups does one 20480-wide dot
product each; stage 2 back over `dim/256` does the RMSNorm, with Sinkhorn
riding along on workgroup 0. At the 0.66 µs per dispatch kernel_p1.md §3.4
measured, the extra dispatches are 1.3 µs a layer.

A layer runs mega_mhc **three** times — attention half, FFN half, and the
`hc_post` that closes the block — with different weights and buffers each time.
A stage owns one slice of the address table, so two dispatches of one stage in
a command buffer would both see whatever was written last; the FFN half and the
close have their own stages.

### 4.2 sparse_attn is two passes, on purpose

`sparse_attn_kernel` rounds the probabilities to bf16 **after** subtracting the
final row max (`acc_s_cast`), and takes that max over the KV scores **only** —
the per-head `attn_sink` enters the denominator as `exp(sink − mx)` but never
competes for the max. An online softmax rescales against a running max, so it
would round `p` against a different number and produce a different result.
Reproducing the reference costs one extra pass over 320 KiB. The running max
also starts at a finite −1e30, not −∞, so a query whose whole index row is −1
comes out zero rather than NaN.

### 4.3 Activation staging: bf16 in LDS, on both paths

The fp8 GEMVs stage x once per workgroup into LDS, already quantised, as
**bf16** — not as the E4M3 byte. The byte is half the LDS but costs a second
LDS access per element (one for the byte's word, one to index the decode
table), and LDS ports are what these kernels run out of. The evidence was
`wo_a`, which stages bf16 because its activation is not quantised at all,
reading at 260 GB/s while `wq_b` and `wo_b` — identical but for the byte
staging — managed 165 and 141. Storing the quantised **value** as bf16 is exact
(three mantissa bits) and makes the inner loop one shift and one table lookup.
The block scale stays out of the stored value and rejoins through `ldexp`,
which is exact and keeps the staged number in range whatever the block's
magnitude.

### 4.4 Two ways to lose 50× that have nothing to do with bandwidth

Both found by `bench/attn_bench` and both worth remembering:

1. **A local array indexed by a loop bounded by a push constant is in VRAM.**
   Slang cannot unroll the loop, the array stops being promotable to registers,
   and it goes to scratch memory. mega_mhc's 4×4 Sinkhorn matrix cost **216 µs
   a dispatch** for a 16-float working set; bounding the loops by the
   compile-time `kMaxHc` instead took it to **2.8 µs**. The gate's rank loop had
   the same shape: 55 µs → 2 µs.
2. **A serial scan in one lane is a serial scan in one lane.** The gate's
   top-k walked 384 experts per rank, 6144 dependent compares, **56 µs a layer
   — 2.2 ms a token** to pick six numbers out of 384. Sixteen rounds of a
   workgroup argmax over an LDS tree: **12.6 µs**.

---

## 5. Measured bandwidth

`bench/attn_bench --model … --layers 8 --iters 64 --rows 2`, idle machine,
2026-09-14. Raw: `bench/results/attn_p2.csv`.

**How the number is honest.** One `AttnRunner` owns one address table, so a
benchmark that cycles layers between *submissions* has every iteration inside a
command buffer reading the same layer's weights, and anything under the 32 MB
MALL reports a fantasy — the first version had `wq_a` at 374 GB/s and `wo_a` at
397 on a 217 GB/s memory system. There is now one runner per layer and
iteration *i* uses runner *i* mod layers. `bytes` is the weight plane plus its
UE8M0 scale plane; the ceiling to compare against is kernel_p1.md §2.2's
**216–218 GB/s**.

| kernel | µs | bytes | GB/s | % of ceiling |
|---|---:|---:|---:|---:|
| `mega_mhc.post` | 2.3 | 164 KB | 72 | latency |
| `mega_mhc.mix` | 5.1 | 1.97 MB | 385 | *MALL-resident, see below* |
| `mega_mhc.final` | 3.2 | 41 KB | 13 | latency |
| `wq_a` | 44.3 | 6.56 MB | 148 | 68% |
| `wq_b` | 268.3 | 41.98 MB | 157 | 72% |
| `wkv.gemv` | 20.8 | 2.62 MB | 126 | 58% |
| `wkv.finish` | 8.8 | 2.6 KB | — | latency |
| `sparse_attn.score` | 61.6 | 134 KB | — | latency |
| `sparse_attn.combine` | 80.6 | 134 KB | — | latency |
| `wo_a` | 201.0 | 33.59 MB | 167 | 77% |
| `wo_b` | 346.8 | 41.98 MB | 121 | 56% |
| `gate.score` | 16.3 | 3.93 MB | 242 | *partly MALL* |
| `gate.topk` | 12.6 | 3.1 KB | — | latency |
| **one layer, 1–9** | **1071** | **133 MB** | **124** | **57%** |
| `head` | 5643 | 1.32 GB | **235** | **108%** |
| **40 layers + head** | **48.5 ms** | | | |

Two caveats on the table. `mega_mhc.mix` and `gate.score` have working sets of
31 MB across eight layers, so they sit in the 32 MB MALL and their GB/s is a
cache number, not a memory one — a real token cycles 40 layers and would not.
And `head` reads 235 GB/s, above the raw-read figure, which is the same result
kernel_p1.md §3.2 got for the MoE kernel at 102%: a long sequential bf16 stream
with 16,160 independent workgroups keeps more requests in flight than the
raw-read shader does. design §7.1 rule 2 already says raw-read is a lower bound
on the ceiling, not the ceiling.

Against design §3.4's model — 0.97–1.29 ms a layer for the resident weights —
the path is inside budget. It is not near the memory system, and §7 says where
that is.

### 5.1 `rows_per_lane`, settled by sweep

Row blocking trades workgroups and register pressure for activation reuse, and
which side wins is not predictable from the shape. Sweeping 1 / 2 / 4 per
stage:

| kernel | rows 1 | rows 2 | rows 4 | cap set |
|---|---:|---:|---:|---:|
| `wq_a` (1280 rows) | 125 | **147** | 147 | 2 |
| `wq_b` (32768) | 114 | **160** | 157 | 2 |
| `wkv` (512) | **139** | 131 | 22 | 1 |
| `wo_a` (8192) | **166** | 132 | 132 | 1 |
| `wo_b` (5120) | **124** | 114 | 115 | 1 |
| `head` (129280) | **235** | 208 | 183 | 1 |

The two tall-and-narrow kernels gain; everything else loses. `wkv` at 4 is
sixteen workgroups on forty CUs. The head has nothing to reuse — its activation
is 20 KiB and already in LDS — so the register pressure is all cost. Caps live
in `kStages` in `gpu/vulkan/attn_kernels.cpp` next to the numbers.

---

## 6. What runs, and what is loaded

`tests/test_gpu_layer.cpp` runs one whole decoder layer the way decode will:
every stage fed the previous one's output, one command buffer for dispatches
1–9, the §7.1 timeline gate, then the MoE. Only the block input and the KV the
prefill left are golden.

| piece | state |
|---|---|
| mega-mHC, Q path, KV path, sparse attention, output projection, gate, head | **real**, from pinned weights read off NVMe by the IoEngine |
| the window KV ring | **real**, written by `wkv.slang` every step as `Attention._window_kv` writes it |
| the routed experts | **real**, fetched through `store::Planner` on the gate's ids and computed by the §7.9 kernels |
| the shared expert | **real**, fp8, from the pinned set through its own one-slot runner |
| the compressed KV and the indexer's top-k list | **LOADED** from the oracle's prefill when this was written; **produced** since — see §9. Everything downstream of them was real and measured either way |
| prefill, the engram write, DSpark | not started |

The compressed half is stored **bf16**, not design §11.3's packed FP4 E2M1 +
E4M3/16. The values are already on the fp4 grid — the compressor quantises them
— so this is a packing choice worth 18 MB at 64K context (66 vs 48 MB), and it
costs a nibble unpack in `sparse_attn`'s inner loop to take back. Worth
revisiting when the compressor kernel exists to write the packed form.

### 6.1 The §7.1 gate

`DecodeLayer::run_moe` reads the gate's ids out of host-coherent memory, calls
back to make the six experts resident, host-signals the timeline value the MoE
wait names, then dispatches. In P2 every expert would be resident and the wait
is a formality — but the test deliberately runs with an eight-slot cache, so
all six really are fetched from NVMe through `store::Planner` on every layer,
and the gate is the thing that unblocks them.

### 6.2 One command buffer per layer, not per token

design §7.1 wants one **pre-recorded** command buffer per token. This records
one per layer. The reason is the address table: a stage owns one slice of it,
so a pre-recorded token needs the table indexed by **layer inside the shader**,
the way the expert pointer table already is. That is a P3 change — the
dispatch-overhead budget (§3.4: 470 dispatches × 0.66 µs = 0.31 ms a token)
says it is worth about 0.5% either way, so it waits for the streaming runtime
that needs it for the timeline waits.

---

## 7. What is next, in order (step 1's list; superseded by §11)

1. **`wo_b` and `wq_b`**, 121 and 157 GB/s against 217. They are 84 MB of the
   133 MB a layer reads, so the layer is 57% of the ceiling largely because of
   them. The suspects, in order: the two barriers around the LDS staging (a
   workgroup retires 8–16 rows and pays them each time); K = 8192 meaning 16 KiB
   of LDS and low occupancy for `wo_b`; and the single accumulator per lane,
   which kernel_p1.md §3.3 found was not enough to hide latency on the FP4 path
   either. A K-split with a second reduction pass is the structural answer.
2. **`sparse_attn`, 142 µs a layer — 5.7 ms a token, 13% of the path.** 64
   workgroups each pull the whole 320 KiB of KV through L2, twice (score, then
   P·V), which is 42 MB of L2 traffic for 320 KiB of data. The fix design §7.5
   already describes is a KV tile in LDS shared across heads; the obstacle is
   that q for 64 heads is 64 KiB and does not fit, so it wants a head-group
   split with a partial-softmax combine. Not urgent at 2% of a decode token,
   but it is the largest pure waste in the path.
3. **The compressor and indexer kernels** (§7.4). Until they exist the
   compressed KV is loaded, which is the one gap between this and a decode step
   that stands on its own.
4. **A layer-indexed address table**, so §7.1's per-token command buffer is
   expressible (§6.2).
5. **`mega_mhc.post` at 72 GB/s.** 164 KB a dispatch, 80 times a token, so 2.3 µs
   × 80 = 0.18 ms — small, but it is a pure `hc_post` and `hc_pre` over 20480
   floats and should be near the ceiling.

## 8. Done / not done (step 1; §12 is the current one)

**Done**

* `tools/oracle.py --level l2`, and 5.1 MB of golden tensors for seven layers.
* `gpu/shaders/{attn_common,fp8_gemv,mega_mhc,wq_a,wq_b,wkv,sparse_attn,wo_a,wo_b,gate,head}.slang`,
  all through `slangc` and `spirv-val`.
* `gpu/vulkan/attn_kernels.{h,cpp}`: eighteen pipelines, the shared address
  table, `GpuScratch`.
* `store/pinned.{h,cpp}`: the §2.2 pinned set through the manifest and the
  IoEngine into GPU-addressable memory, with a `GlobalMemoryStatusEx` commit
  check that fails with a sentence pointing at build.md's pagefile section.
* `runtime/{rope.h,kvstore,decode_layer,moe_bridge}`: one decoder layer.
* `tests/test_gpu_attn.cpp` (per stage), `tests/test_gpu_layer.cpp` (chained),
  `bench/attn_bench.cpp`, `cpu::act_quant_block` and friends.

**Not done**

* The compressor and indexer kernels (§7.4) — the decode step reads their
  prefill output rather than producing it.
* The engram kernels (§7.10), prefill (§7.13), DSpark (§7.12).
* Sampling on the GPU (§7.11's Philox / Gumbel-max half).
* `KvCache::snapshot` / `rollback` (§10.2) and prefix persistence (§11.4) —
  still the interfaces they were.
* The pinned set is loaded a layer at a time by the tests; nothing has yet
  loaded all 17.7 GB in one go and measured it.

---

## 9. P2 step 2 — the compressor and the indexer, and where the bandwidth went

Status: **2026-09-14**, same machine, same checkpoint, same decode token.
Raw data: `tests/data/l2x/`, `bench/results/attn_p2.csv`.

Two things §7 asked for. The compressed KV and the top-k list are **produced**
now, not loaded — §6's one remaining gap is closed. And the two GEMVs §7 named
went 157 → 204 and 121 → 135 GB/s, with `sparse_attn` halving on the way past.

### 9.1 What is produced now

| piece | before | now |
|---|---|---|
| compressed KV (`cmp_kv`) | loaded from the oracle's prefill | `compressor.slang`, three dispatches |
| the index keys | loaded | `indexer.slang` stage 2 |
| the indexer's queries, weights, scores | loaded | `indexer.slang` stages 0, 1, 3, 4 |
| `topk_idxs`, compressed half | loaded | `indexer.slang` stage 5 |
| `topk_idxs`, window half | host, `get_window_topk_idxs` | unchanged |

Ten new pipelines on two new shaders, appended to `AttnStage`. A layer that is
not a `kv_source_layer` dispatches none of the `Cmp*`; one that is not an
`index_source_layer` dispatches none of the `Idx*`. Four layers of forty have a
compressor, eight an indexer.

### 9.2 Accuracy, against the reference's own tensors

`tests/test_gpu_attn.cpp` `gpu_attn.l2_compressor_indexer`, layers 2, 14 and 20
(a ratio-2 source, a second one with an engram, and the ratio-1 decoder
source). Every stage is fed the reference's own input:

| stage | layer 2 (ratio 2) | layer 14 (ratio 2) | layer 20 (ratio 1) |
|---|---|---|---|
| `compressor.wkv` | 1.000000000 | 1.000000000 | 0.999998848 |
| `compressor.wgate` | 1.000000000 | 1.000000000 | — (no gate) |
| `kv_state[slot]` write | **bit-exact** | **bit-exact** | — |
| `score_state[slot]` write | **bit-exact** | **bit-exact** | — |
| pooling + `compressor.norm` | **bit-exact** vs fp64 CPU | **bit-exact** | **bit-exact** vs the oracle |
| RoPE + FP4(16, E4M3) + cache write | — | — | **bit-exact**, 0/256 nibble bytes and 0/32 scale bytes differ |
| `indexer.wq_b` (+ `q_norm`, act_quant) | 0.999998575 | 0.999998615 | 0.999998719 |
| RoPE + FP4(32, UE8M0) on q | **bit-exact** | **bit-exact** | **bit-exact** |
| `wk` + `k_norm` | — | — | **bit-exact** |
| RoPE + FP4 on the key | — | — | **bit-exact**, 0/64 and 0/4 bytes differ |
| `weights_proj` × scale | **bit-exact** | **bit-exact** | **bit-exact** |
| `index_score` | **bit-exact** | **bit-exact** | **bit-exact** |
| `topk_idxs` | 32/32 | 32/32 | 65/65 |

Eleven of the sixteen are bit-exact against the reference, and the three that
are not are the three that end in an fp8 GEMV, sitting exactly on the 1.6e-3
bf16 noise floor §3 describes. The fp4 byte planes agree **byte for byte**,
which pins the E2M1 rounding rule (ties to even mantissa), the nibble order and
both scale formats — E4M3 for the compressed KV and UE8M0 for the indexer — at
the same time.

`index_score` being bit-exact is worth a sentence, because it was not at first:
0.99999 until the three bf16 roundings the reference performs went in (the
einsum's output, the multiply by `weights`, and the sum over the head axis).
Being more precise than the reference costs agreement here for the same reason
§3 gives about `act_quant`, one step removed.

### 9.3 What the reference does that is not obvious

Five things the L2 data made visible and which `compressor.slang` and
`indexer.slang` had to be written around:

1. **The pooling softmax is over the ratio axis, per element.** `kv_state` and
   `score_state` are [ratio, head_dim] and the pooling is
   `(kv_state * score_state.softmax(dim=1)).sum(dim=1)`, so each of the 512
   dims gets its own two-way softmax. It is not one weight per token.

2. **The dtype changes in different places on the two paths.** At ratio > 1 the
   reference runs `x.float()` and holds both projections and the pooling in
   fp32 — the checkpoint's bf16 weights are promoted at load — and rounds to
   bf16 exactly once, where `kv.to(dtype)` feeds the norm. At ratio 1 there is
   no gate and no fp32: `self.norm(self.wkv(x))` is a plain bf16 `Linear`. The
   table above shows it: `compressor.wkv` is exact to 2e-7 on the two fp32
   layers and to the bf16 floor on the ratio-1 one.

3. **A latent stands for the FIRST token of its group.** It is rotated at
   `start_pos + 1 - ratio`, not at `start_pos`, and so is the index key derived
   from it — while the indexer's *queries* are rotated at `start_pos`. Two RoPE
   tables per source layer, from the same per-layer configuration.

4. **Two FP4 formats in one layer.** The compressed KV is block 16 with an E4M3
   scale and an amax floor of 6·2⁻⁹; the indexer's q and k are block 32 with a
   power-of-two scale and a floor of 6·2⁻¹²⁶. Mixing them up is invisible in a
   cosine and visible in the bytes, which is why the test compares bytes.

5. **The key owner is the layer that compresses, not the layer that scores.**
   At decode position 64 a ratio-2 source's group does not complete, so it
   publishes nothing and scores against whatever key cache was published last —
   layer 20's, not its own. §1.2 already recorded that for the oracle; the
   kernel side of it is that the host, not the shader, decides which cache
   address goes in the slot.

### 9.4 Two tensors the oracle did not export, and one it cannot

`tools/oracle.py --level l2` exports the ONE index key this step produced, which
pins the key derivation and says nothing about the scoring, because the other 64
keys came out of the prefill. `tools/oracle_l2_extra.py` is a separate script —
`oracle.py` belongs to another track — that imports it, wraps `L2Capture.attach`
and `_l2_collect`, and writes a superset into `tests/data/l2x/`:
`index_k_all`, `index_score`, `index_weights_scaled`, the ratio-2 compressor's
`kv_state` / `score_state` and the raw `wkv` / `wgate` projections behind them.
2.35 MB for three layers, 305 s to produce.

Two gaps remain, both stated in the test:

* **`index_score` is recomputed, not captured.** `Indexer.forward` keeps it in a
  local and returns only indices, so the extra script recomputes it with torch
  from the reference's own post-fp4 q, its live `shared_attn.index_k` slice and
  its `weights_proj` output, in the reference's op order. A misreading of the
  formula would be reproduced on both sides; everything else is caught.
* **The ratio-2 pooling has no reference output at this position.**
  `(64 + 1) % 2 != 0`, so `Compressor.forward` returns None. What it *does* do —
  write slot 0 of both states — is checked against the reference. The pooling
  itself then runs on a synthetic complete group against an fp64 CPU
  transcription. Catching a wrong pooling against the reference needs a
  two-step decode export, which is the obvious next extension of that script.

And one that is not a gap so much as a fact about the context: **the top-k has
never yet been asked to select.** `index_topk` is 512, a 64-token prefill at
ratio 2 leaves 32 compressed positions and at ratio 1 leaves 65, so
`min(index_topk, n)` is `n` and the reference keeps everything. The comparison
against `topk_idxs` is real but degenerate. `gpu_attn.indexer_topk_select`
therefore drives the radix select directly — 512 of 4096 with a deliberate tie
cluster — against a CPU stable sort: **512/512**, ties included.

---

## 10. Bandwidth, after §7 items 1 and 2

`bench/attn_bench --layers 8 --iters 64 --rows 2`, idle machine (`head` at 234
GB/s in both halves of the A/B, which is what says so), 2026-09-14. The "before"
column is the *same binary and the same machine minutes apart*, running the
shaders as of the previous commit out of a separate `.spv` directory, so the
two columns are not from two different days.

| kernel | before µs | before GB/s | after µs | after GB/s | % of 217 |
|---|---:|---:|---:|---:|---:|
| `mega_mhc.post` | 2.38 | 69 | 2.39 | 69 | latency |
| `mega_mhc.mix` | 5.33 | 369 | 5.34 | 368 | *MALL* |
| `mega_mhc.final` | 2.96 | 14 | 3.01 | 14 | latency |
| `wq_a` | 44.5 | 147 | **38.5** | **171** | 79% |
| `wq_b` | 258.3 | 163 | **205.4** | **204** | **94%** |
| `wkv.gemv` | 19.9 | 132 | **17.1** | **154** | 71% |
| `wkv.finish` | 8.4 | — | 9.2 | — | latency |
| `sparse_attn.score` | 54.7 | — | **42.1** | — | latency |
| `sparse_attn.combine` | 85.9 | — | **27.3** | — | latency |
| `wo_a` | 199.2 | 169 | 201.4 | 167 | 77% |
| `wo_b` | 341.9 | 123 | **312.2** | **135** | 62% |
| `gate.score` | 16.7 | 236 | 16.3 | 241 | *MALL* |
| `gate.topk` | 12.4 | — | 12.5 | — | latency |
| **one layer, 1–9** | **1052.6** | 127 | **892.6** | **149** | 69% |
| `head` | 5618 | 236 | 5653 | 234 | 108% |
| **40 layers + head** | **47.7 ms** | | **41.4 ms** | | |

Plus the §7.4 kernels, which run on some layers only:

| kernel | µs | bytes | GB/s |
|---|---:|---:|---:|
| `compressor.wkv` | 13.9 | 5.24 MB | 377 *(4 layers: partly MALL)* |
| `compressor.wgate` | 13.9 | 5.24 MB | 378 *(same)* |
| `compressor.norm` | 3.2 | 8 KB | latency |
| `compressor.store` | 6.4 | 3 KB | latency |
| `indexer.wq_b` | 16.9 | 5.25 MB | 311 *(4 layers: partly MALL)* |
| `indexer.q_finish` | 9.7 | 25 KB | latency |
| `indexer.key` | 17.0 | 131 KB | latency |
| `indexer.weights` | 21.7 | 328 KB | latency |
| `indexer.score` | 16.1 | 17 KB | latency, and the only one that grows with context |
| `indexer.topk` | 0.8 | 260 B | latency |
| **compressor, 4 of 40 layers** | **37.3** | | |
| **indexer, 8 of 40 layers** | **82.2** | | |

**0.81 ms a token** for both together — 2% on top of the 41.4 ms. The GB/s
columns for the two 5 MB projections are cache numbers: only four layers have
them, so 21 MB of weights cycles inside the 32 MB MALL. The microseconds are
real; the bandwidth is not a memory-system number.

`indexer.score` is the one entry that will move. At 64K context and ratio 2 it
is 32K positions against 32 × 128 query dims — 134 MFLOP and 2 MB of keys
against the 16 KB and 8 keys it reads here — so its 16 µs is a floor, not a
measurement of the production case.

### 10.1 What actually made the difference, and what did not

Four changes, all in `fp8_gemv.slang` / `attn_common.slang` / `sparse_attn.slang`,
and the interesting part is which of the §7 suspects turned out to be wrong.

1. **The LDS budget, per shader rather than per family.** `gXQ` was sized for
   wo_b's K = 8192 unconditionally, so `wq_b` reserved 16 KiB to use 2.5 KiB.
   `DEEPMOE_GEMV_MAX_K` is now a `#define` in each including shader.
   It is **not** "make it as small as K": `wq_a` measured **152 GB/s at a
   5120 budget against 166 at 8192**, and `wo_a` the same way. More resident
   workgroups each re-staging a 4–5K-wide activation is more L2 traffic, not
   less. The two kernels that gain are the ones whose activation is already
   small — `wq_b` and the indexer's `wq_b`, both K = 1280 — where the occupancy
   is free. Both numbers are in the per-shader comment.

2. **The `act_quant` staging was running on a sixth of the workgroup.** One
   thread owned one 32-element block, so `wq_b` staged 1280 activations on 40
   of its 256 threads while 216 idled through the most expensive arithmetic in
   the kernel. Threads per block is now a compile-time function of the same K
   (4 at K = 1280, 2 at 4096, 1 at 8192) with the block amax an LDS reduction
   over them. Two smaller things came with it: `fp8_round` is now nine
   instructions and branchless (see `attn_common.slang`), and the threads with
   no block to quantise no longer run the rounding anyway — that one alone was
   8% of `wq_a`.

3. **`row_reduce` as one `WaveActiveSum` is not a free win.** Replacing the LDS
   tree's seven workgroup barriers per row with a subgroup reduction is worth
   **+30% on `wq_b` and +20% on `wkv` and −14% on `wq_a` and `wo_a`**. The two
   that lose are the two whose workgroups each re-stage a wide activation over
   many workgroups; the barriers evidently pace those workgroups against each
   other in a way the memory system likes. It is a per-stage specialisation
   constant now (`WaveReduce`), with the four numbers in `kStages`.

4. **`sparse_attn` was never bandwidth-bound; it was doing scalar byte loads.**
   Stage 0 gave one thread one (head, position) pair and walked 512 dims
   serially: 512 loads of q, 512 of the E4M3 byte and 512 of its scale, per
   pair. Across 64 heads and 640 positions that is 63 million load instructions
   for 320 KiB of data. A KV row is now covered by the 32 lanes of one wave,
   sixteen dims each, so the window half is one 16-byte load and one scale byte
   per lane and the dot product is a `WaveActiveSum`. Stage 1 does the same for
   P·V, with a lane owning sixteen output dims **whatever the head grouping**
   and the partials from the disjoint KV subsets meeting in LDS — giving a lane
   two dims instead, which is what a one-head workgroup used to do, turns every
   element back into a byte load and measured 106 µs against 27.

And the suspect §7 named that turned out not to be one:

> **The `act_quant` round trip costs `wo_b` nothing.** Compiling it with
> `ActQuant = 0` — wrong arithmetic, timing only — measures **150.2 GB/s
> against 151.7** with it on. So the pre-quantisation dispatch that would have
> been the structural fix buys nothing, and `wo_b`'s remaining gap is not the
> quantiser. Nor is it DRAM: 42 MB in 312 µs is 135 GB/s against a 217 GB/s
> memory system, and the 20 MB of re-staged activation is a 32 KB working set
> that lives in the MALL. What is left is LDS (19 KiB, three workgroups a CU)
> and the single accumulator per lane, which is the K-split kernel_p1.md §3.3
> wanted — see §11.

### 10.2 `sparse_attn`: 41 MB of L2 traffic for 320 KiB, still

§7 item 2 asked for the KV to be read once. The machinery is there —
`AttnSpec::heads_per_wg` puts G heads in a workgroup, which divides the L2
traffic by G — and it is **off**, because it is slower:

| heads per workgroup | workgroups | score + combine | L2 traffic |
|---|---:|---:|---:|
| **1** | **64** | **69 µs** | 41 MB |
| 2 | 32 | 100 µs | 20 MB |
| 4 | 16 | 169 µs | 10 MB |
| 8 | 8 | 288 µs | 5 MB |

64 heads at 8 a workgroup is eight workgroups on forty CUs. The traffic and the
occupancy are being traded along the wrong axis: what design §7.5 actually
describes is a head-group × KV-tile split, which keeps 64 workgroups AND reads
the KV eight times, at the cost of a partial-softmax combine across the tiles.
That is the remaining half of this item and it is in §11.

Meanwhile the vectorisation took the two stages from 141 µs to **69 µs a
layer** — 2.8 ms a token, 7% of the path instead of 13% — without touching the
traffic at all.

### 10.3 Interfaces: what changed, all of it additive

Track G calls these kernels. Nothing existing moved:

* **`AttnStage` gained ten enumerators** (`CmpKvGemv`, `CmpGateGemv`,
  `CmpNorm`, `CmpStore`, `IdxQGemv`, `IdxQFinish`, `IdxKey`, `IdxWeights`,
  `IdxScore`, `IdxTopK`), appended before `Count`. No existing value changed.
* **`AttnPush` gained a trailing `n_heads`, defaulted to 0**, and 0 means "one
  workgroup per head", which is the geometry every existing caller already
  dispatches. `runtime/decode_layer.cpp` needs no change and gets the
  vectorisation regardless; a caller that fills it in must dispatch
  `AttnRunner::attn_groups(n_heads)` workgroups instead of `n_heads`. Both
  geometries are checked against the oracle in `gpu_attn.l2_per_stage`, and
  they agree to the last bit.
* **`AttnSpec` gained `heads_per_wg` (default 1)**, which does nothing unless
  `AttnPush::n_heads` is set.
* **`CmpPush` and `IdxPush` are new**, as are `slot::kCmp*` / `slot::kIdx*` and
  `kIdxScoreTile`.
* `AttnRunner::create` now **rejects** `subgroup_size != 32` and
  `heads_per_wg` outside 1..8. Both `sparse_attn`'s score stage and the
  indexer's reduce across exactly one Wave32, and a silent wrong answer is the
  alternative. Every existing caller already passes 32.

---

## 11. What is next, in order (replacing §7)

1. **`wo_b`, 135 GB/s against 217.** Still 42 of the 133 MB a layer reads. §10.1
   rules out the activation quantisation and rules out DRAM; what is left is the
   19 KiB of LDS holding three workgroups to a CU and the single accumulator per
   lane. The structural answer is the K-split with a second reduction pass, and
   it needs one extra dispatch and therefore a runtime change, which is why it
   did not happen here.
2. **`sparse_attn`'s 41 MB of L2 traffic**, now that the instruction count is
   dealt with. The shape is a head-group × KV-tile grid: 64 workgroups, each
   covering 8 heads and an eighth of the KV, with a partial-softmax combine
   across the eight tiles of a head. It keeps the occupancy that §10.2 shows is
   worth more than the traffic, and takes the traffic too. Worth ~1 ms a token.
3. **`wq_a` at 171 and `wkv` at 154**, the two remaining GEMVs under 80%. Both
   are now limited by the same thing: a workgroup stages 5120 activations to
   retire 8 or 16 rows.
4. **A layer-indexed address table**, so §7.1's per-token command buffer is
   expressible (§6.2). Unchanged from §7.
5. **`mega_mhc.post` at 69 GB/s.** Unchanged from §7.
6. **A two-step decode L2 export**, so the ratio-2 compressor's pooling can be
   checked against the reference rather than against a CPU transcription of it
   (§9.4), and so the indexer's top-k has something to select from.

## 12. Done / not done, after P2 step 2

**Newly done**

* `gpu/shaders/compressor.slang` and `gpu/shaders/indexer.slang`, ten pipelines,
  through `slangc` and `spirv-val`; the compressed KV and the top-k list are
  produced, not loaded.
* `tools/oracle_l2_extra.py` and `tests/data/l2x/` (2.35 MB, three source
  layers): the index-key cache, the index scores and the compressor's carried
  state.
* `tests/test_gpu_attn.cpp` `gpu_attn.l2_compressor_indexer` and
  `gpu_attn.indexer_topk_select`.
* The §7 bandwidth items: one layer's dispatches 1–9 from 1.071 ms to
  **0.893 ms**, and 40 layers + head from 48.5 ms to **41.4 ms**.
* A run of the whole `gpu_attn` suite under the Khronos validation layers, with
  nothing reported. (It found one thing on the way: `Rig` left a `VkBuffer` and
  its memory alive past `vkDestroyDevice` because `PinnedStore` does not release
  its regions on destruction. Fixed in the test.)

**Still not done** — §8's list, minus the compressor and indexer:

* The engram kernels (§7.10), prefill (§7.13), DSpark (§7.12).
* Sampling on the GPU (§7.11's Philox / Gumbel-max half).
* `KvCache::snapshot` / `rollback` (§10.2) and prefix persistence (§11.4).
* The pinned set loaded all at once and measured.
* Prefill's side of the compressor and indexer: everything above is the decode
  step, where a group holds one token and a query is one row. The prefill forms
  (`start_pos == 0`, a whole chunk pooled at once, a per-query `compress_lens`
  mask) are in the reference and are not written.

---

## 13. P3 Track J — the attention path at 696 µs a layer

Status: **2026-09-14**, same machine and checkpoint. Raw data:
`bench/results/attn_p3.csv`. Commits: the kernels, then this section.

**How "idle" was established this time.** Two other tracks were running CPU
oracles and GPU benchmarks for most of the session. A CPU-only load turned out
not to matter — with a 12-core Python oracle running, `head` still read 233.4
GB/s and every kernel matched the idle numbers. GPU work from another process
does matter, and it comes in bursts short enough that `head` alone can miss one
(a run with `head` at 233 had every other kernel 3× slow). Every number below
is from a run with `head` ≥ 230 GB/s **and** `mega_mhc.post` ≤ 2.6 µs, taken
while no other GPU process was alive; runs that failed the gate were thrown away
and repeated. Run-to-run noise at that gate is ±3%.

### 13.0 The result in one table

`bench/attn_bench --layers 8 --iters 64 --rows 2`, the same binary for both
columns; "P2" is the shader directory of commit 0964e5e, "P3" the defaults this
section ends with.

| kernel | P2 µs | P2 GB/s | P3 µs | P3 GB/s | % of 217 | P3 stage(s) |
|---|---:|---:|---:|---:|---:|---|
| `wq_a` | 38.2 | 172 | **35.6** | **184** | 85% | `WqAKSplit` + `WqAKCombine` (2 slices) |
| `wq_b` | 207.5 | 202 | **194.7** | **216** | **99%** | unchanged stage, LDS fix only |
| `wkv` gemv + finish | 17.2 + 9.2 | 153 | **7.9 + 9.1** | 334 *(8 layers: partly MALL)* | — | `WkvKSplit` + `WkvKFinish` (2 slices, no extra dispatch) |
| `sparse_attn` | 42.2 + 27.4 | — | **17.4 + 26.9** | — | — | `AttnScoreT` + `AttnPvT` (32 × 1 tiles, no finish) |
| `wo_a` | 202.1 | 166 | **162.4** | **207** | **95%** | `WoAKSplit` + `WoAKCombine` (4 slices) |
| `wo_b` | 317.8 | 132 | **204.9** | **205** | **94%** | `WoBKSplit` + `WoBKCombine` (8 slices) |
| **one layer, 1–9** | **898.8** | 148 | **696.3** | **191** | 88% | |
| `head` | 5679 | 233 | 5650 | 234 | 108% | |
| **40 layers + head** | **41.6 ms** | | **33.5 ms** | | | |

(CSV runs `fin_p2` and `fin_p3`; a repeat of the defaults, `fin_tbl`, read
696.8. The P3 microseconds include each combine dispatch, 1.1–1.4 µs.)

Against §11's asks: `wo_b` **135 → 205 GB/s** (asked ≥ 185); one layer **893 →
696 µs** (asked ≤ 700); sparse attention **68 → 44 µs** with the KV read by 8
head groups instead of 64 heads (asked ≤ 40 — **not met**, §13.5 says where the
last 4 µs are); long context measured at 4096 and 32768 entries and exact
against fp64.

Two separate things did it, and it is worth keeping them apart because the one
§11 predicted is not the one that moved first:

1. an **LDS bank conflict** on every read of the staged activation, fixed by a
   layout change that is bit-identical (§13.1) — 899 → 800 µs on the stages
   the runtime already calls;
2. the **K-split** (§13.3) — 800 → 710 µs — which on its first measurement
   *lost* on every kernel but wkv, because it was measured at two rows a lane.

### 13.1 A 16-way bank conflict on every read of the staged activation

`fp8_gemv.slang` stored a 32-element block of the staged activation as 16 LDS
words (two bf16 to a word), block `blk` at `gXQ[16 blk .. 16 blk + 15]`. The
inner loop gives lane `sublane` the blocks `sublane, sublane + 32, …`, so the
32 lanes of one row group read word `i` at index `16 l + i`, `l = 0..31`.

LDS has 32 banks of one dword, and `(16 l + i) mod 32` takes exactly **two**
values over 32 lanes: every even lane lands in bank `i`, every odd lane in bank
`i + 16`. Sixteen lanes to a bank is sixteen LDS cycles where one would do, on
all sixteen reads `read_x_block` issues per block, per lane, per row.

The fix is one word of padding: a block is 17 words (`kXStride`). 17 is odd, so
`17 l mod 32` is a bijection and every lane owns a bank. It costs 1/16 more LDS
(wo_b's `gXQ` 16 → 17 KiB).

`sparse_attn.slang` had the same pattern twice: stage 0 staged q as
`[head][dim]` with a lane reading dims `16 l .. 16 l + 15`, and stage 1 stored
P.V partials at `16 tid + i`. q is now stored transposed,
`[head][dim-within-span][lane]`, so lane `l` reads `32 i + l`; the partials use
the odd 17 stride.

**It is a layout change and nothing else.** With `DEEPMOE_SKIP_P3=1`, the L2
suite pointed at the P2 shader directory and at the new one prints identical
lines — every cosine, `max|d|` and relative L2 of every classic stage on all
seven layers.

Idle, same binary: P2 shaders (`fin_p2`) against the new ones (`fin_tbl`,
`fin_p3`), classic stages only:

| kernel | P2 µs | new µs | Δ |
|---|---:|---:|---:|
| `wq_a` | 38.2 | 38.1 / 38.2 | 0 |
| `wq_b` | 207.5 | **193.6 / 194.7** | −7%, now **216–217 GB/s** |
| `wkv.gemv` | 17.2 | **9.5 / 9.8** | −44% |
| `sparse_attn.score` | 42.2 | **26.3 / 26.8** | −37% |
| `sparse_attn.combine` | 27.4 | 27.3 / 27.9 | 0 |
| `wo_a` | 202.1 | 196.0 / 195.7 | −3% |
| `wo_b` | 317.8 | **260.8 / 263.1** | −18%, 132 → 161 GB/s |
| `indexer.wq_b` | 20.5 | 15.2 / 17.5 | −20% |
| **one layer, 1–9** | **898.8** | **799.5 / 803.8** | **−11%** |
| **40 layers + head** | **41.6 ms** | **37.7 / 37.8 ms** | |

(A first pair of runs earlier in the session, not in the CSV, read 891.0 /
892.6 against 799.8 / 796.6.)

`runtime/decode_layer.cpp` calls exactly these classic stages, so it already has
this, with no interface change.

§10.1 item 1 recorded that `wq_a` and `wo_a` got *slower* with a tighter LDS
budget, and explained it as L2 traffic from more resident workgroups. With every
staged read queueing sixteen lanes to a bank, "more resident workgroups" also
meant more lanes stuck behind the same two banks; that explanation should be read
with this in mind.

### 13.2 Three things that did not help, measured

**The arithmetic E4M3 decode.** With the staged reads conflict-free, the
remaining LDS traffic in the inner loop is the 256-entry decode table, which 32
lanes index with 32 unrelated weight bytes. design §7.1 rule 5 measured "table
beats arithmetic" on a 16-entry *constant array* in registers, which is not this
situation, so it was measured again. `fp8_bits_decode` (attn_common.slang) is
six ALU instructions — `(b & 0x7F) << 20` already puts exponent and mantissa
where an fp32 wants them, `+ (120 << 23)` rebiases, exponent 0 is fixed up by
removing the implicit one and doubling — and it is exact:
`gpu_attn.fp8_arith_decode_matches_table` checks all 256 codes, and
`DEEPMOE_FP8_ARITH=1` gives output identical to the table on every stage of the
L2 suite. It loses:

| kernel | table µs (`fin_tbl`) | arithmetic µs (`fin_arith`) |
|---|---:|---:|
| `wq_a` | 38.1 | 57.5 (+51%) |
| `wq_b` | 193.6 | 221.5 (+14%) |
| `wkv.gemv` | 9.5 | 12.8 (+35%) |
| `wo_a` | 196.0 | 206.2 (+5%) |
| `wo_b` | 260.8 | 259.0 (−1%) |
| `wo_b.ksplit` ×8 | 204.7 | 211.5 (+3%) |
| one layer, classic / P3 | 799.5 / 696.8 | 861.8 / 748.1 |

Rule 5 now has an fp8 data point: a 256-entry LDS table indexed at random still
beats six ALU instructions. `AttnSpec::fp8_arith_decode` stays, off.

**Four accumulators per block** (`DEEPMOE_GEMV_ACC=4`). §10.1 named "the single
accumulator per lane" as a suspect: a block is a chain of 32 dependent FMAs and
wo_b walks eight of them per lane per row. Dealing them over four accumulators
that meet in a balanced tree makes the chain eight deep. wo_b 259.0 / 266.5 µs
(one accumulator) against 271.8 / 259.2 (four): nothing. Agreement with the oracle
is unchanged to nine digits, so it is only a performance knob, and it is off.

**Folding the UE8M0 weight scale into the staged block factor**
(`DEEPMOE_GEMV_FOLD_SCALE=1`). Every row a workgroup retires lies in one 32-row
band of the scale plane, so `2^(e−127)` can be multiplied into the activation's
power-of-two scale once per workgroup — exact, both being powers of two — which
removes a byte load and an `ldexp` per lane per block per row. wo_b 267.7 /
271.9 µs: nothing, and combined with the K-split it was *worse* (wo_b.ksplit at
2 slices 411 µs against 303). Off.

All three are compile-time or specialisation switches with the default being
the P2 arithmetic, so the measurements can be repeated.

### 13.3 The K-split: slices of 1024–2560, one row a lane

`gemv_ksplit.slang` is the fp8 GEMV cut along K. Stage 0 runs
`gemv_groups × KSplit` workgroups, `gid = row_group × KSplit + slice`; each
stages one K slice of the activation (so its LDS is the slice width, not K) and
writes one fp32 partial per (slice, row). Stage 1 adds the KSplit partials, one
thread a row. wkv's combine is `wkv.slang` stage 2, which adds the partials, takes
the RMS statistic in the same workgroup and runs the kv_norm / RoPE / fp8 ring
write, so for wkv the split costs no dispatch at all. `rows_per_group` carries
wo_a's block-diagonal structure.

The first measurement said the split loses on three of four kernels (wo_b 304
µs against 261). That run had `--rows 2`, and the split stages followed it. At
one row a lane the picture reverses — idle, `(µs / GB/s)`, the unsplit kernel in
the first column:

| kernel | unsplit | ×2 r1 | ×4 r1 | ×8 r1 | ×16 r1 | ×2 r2 | ×4 r2 | ×8 r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wq_a` | 37.7 / 174 | **34.7 / 189** | 36.2 / 181 | 36.3 / 181 | — | 54.9 / 120 | 41.4 / 158 | — |
| `wkv` (split stage only; finish ≈ 9 µs either way) | 9.2 | **7.4–8.1** | 9.9 | — | 17.2 | 7.8 | 10.0 | 9.8 |
| `wo_a` | 194.2 / 173 | 218.6 / 154 *(×1)* | **161.5 / 208** | 165.5 / 203 | — | 192.8 / 174 | 162.5 / 207 | 165.0 / 204 |
| `wo_b` | 259.0 / 162 | 283.2 / 148 | 218.3 / 192 | **204.0 / 206** | 216.3 / 194 | 304.3 / 138 | 232.0 / 181 | 211.0 / 199 |

(Each column is a different run, so a row's neighbours come from different
runs of the same idle machine; all are in the CSV as `ks*` and `k_*`. wo_a's ×1
is the split kernel with no split: the same arithmetic as `wo_a` in a
4096-wide LDS budget instead of 8192, and it is slower, which is §10.1's
observation again.)

So the defaults are **wq_a 2, wkv 2, wo_a 4, wo_b 8, one row a lane**
(`AttnSpec::ksplit_*`, and a rows cap of 1 on the four split stages in
`kStages`). Three repeats of the default: one layer 711.3 / 708.9 / 710.1 µs
with the 8-tile attention, 696–699 µs with the 32-tile one (§13.5).
`WaveReduce` flipped on or off per split stage changed nothing measurable.

Why it works where §11 said it would, and not at two rows a lane: every slice
retires the same rows against a 1/KSplit-wide staged activation, so the
activation re-read per retired weight byte is unchanged, while the LDS a
workgroup holds and the blocks each lane walks shrink by KSplit. Occupancy and
chain length go up and down together. Row blocking works the other way — it
amortises one staged activation over more rows by lengthening every lane's
work — and at slices this narrow there is nothing left for it to amortise.

**Re-association.** A split sums the same products in a different association:
the `WaveActiveSum` over a row group's 32 lanes happens per slice instead of per
row. Against the unsplit kernel, worst over seven layers:

| stage | relative L2 vs the kernel it replaces | max ULP | cosine vs oracle, worst layer |
|---|---:|---:|---:|
| `wq_a` K-split | 5.95e-08 | 1024 | 0.999998495 (= unsplit) |
| `wkv` K-split raw | 6.67e-08 | 64 | 0.999998342 (= unsplit) |
| `wkv` K-split after the fp8 ring write | — | — | 0.999859835 (= unsplit); **0 of 512 ring bytes differ** |
| `wo_a` K-split | 1.34e-07 | 8192 | 0.999998557 (= unsplit) |
| `wo_b` K-split | 8.26e-08 | 3072 | 0.999998527 (= unsplit) |

The ULP column looks large and is not the right measure: an output element that
is a small difference of 8192 large products has a large relative error and a
tiny absolute one. Relative L2 is at fp32 round-off (1e-7) on every stage,
four orders of magnitude under the 1.6e-3 bf16 floor the oracle sits on, and
the cosine against the oracle is the same to nine digits as the unsplit
kernel's. The fp8 ring bytes wkv writes are identical.

### 13.4 wo_b, specifically

| step | µs | GB/s | % of 217 |
|---|---:|---:|---:|
| P2 step 2 (§10) | 312–314 | 134 | 62% |
| + 17-word LDS stride (§13.1) | 259–266 | 158–162 | 73–75% |
| + arithmetic decode / 4 accumulators / folded scale (§13.2) | 258–272 | 154–163 | — |
| **+ K-split ×8, one row a lane (§13.3)** | **203.5–205.4** | **204–206** | **94–95%** |
| K-split ×16 | 216–220 | 191–194 | |

§10.1 said "what is left is LDS … and the single accumulator per lane". Of the
two, LDS — once its bank conflict was gone — was half the answer and the
workgroup geometry the K-split gives was the other half; the accumulator was
not measurably part of it.

### 13.5 sparse_attn on a head-group × KV-tile grid

`sparse_attn_t.slang`, three stages, `AttnTPush`:

* **`AttnScoreT`**, `(n_heads / tile_heads_per_wg) × n_tiles` workgroups: eight
  heads and one KV tile a workgroup. A wave takes a position, its 32 lanes hold
  sixteen dims of the KV row each, and the row is dotted against all eight
  heads' staged q — so the KV is read by 8 head groups, not 64 heads. Scores go
  to the score plane and one maximum per (head, tile) to `kAttnTileMax`.
* **`AttnPvT`**, on its own grid `(n_heads / pv_heads_per_wg) × pv_tiles`.
  Reads the head's `n_tiles` tile maxima — that is the whole reduction, no
  dispatch — takes the final row max (sink excluded, clamped at −1e30), and in
  one pass per wave computes `p = bf16(exp(s − mx))` exactly where
  `acc_s_cast` rounds it, adds it to the denominator partial and accumulates
  P.V. The classic combine does this in three steps with thirteen workgroup
  barriers; this is two.
* **`AttnFinishT`**, only when `pv_tiles > 1`: adds the tiles, adds
  `exp(sink − mx)` once, divides, inverse RoPE. At `pv_tiles == 1` AttnPvT
  finishes itself (`AttnRunner::attn_tiled_finish_needed`).

This is the two-pass structure F asked for — tile maxima first, exp/sum against
the *final* max second — so the bf16 rounding of p sees the reference's number.
It does not stage the index list in LDS, which is why it runs at any `n_kv`;
the classic kernel's 1024-entry index table now falls back to a load past its
end instead of walking off LDS (it lost the device at 4096 before).

**Why P.V has its own grid.** Scores can share a KV read across heads because a
dot product per head is one accumulator. P.V cannot: a lane owns sixteen
*output* dims of *one* head, so eight heads a workgroup is eight waves each
re-reading the tile for its own head — fewer workgroups and no traffic saved.
Measured: `pv_heads_per_wg` 8 is **120 µs**, 1 is 25–27.

**Accuracy at the L3 context** (193 entries), worst of seven layers: cosine
0.999997368 — identical to the classic kernel's — at one P.V tile and at four;
relative L2 against the classic kernel 5.2e-8 and 8.2e-8.

**Accuracy at long context**, against an fp64 CPU transcription of
`sparse_attn_kernel` (random fp8 window through the real E4M3/UE8M0 encode,
random bf16 compressed rows, a −1 hole every 401 positions, random sinks):

| n_kv | score tiles | P.V tiles | heads checked | cosine | max\|Δ\| | relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 1 | 64 of 64 | 1.000000000 | 2.8e-9 | 1.3e-7 |
| 4096 | 16 | 4 | 64 of 64 | 1.000000000 | 2.8e-9 | 1.0e-7 |
| 32768 | 32 | 16 | 4 | 1.000000000 | 9.3e-10 | 1.3e-7 |
| 32768 | 16 | 1 | 4 | 1.000000000 | 1.9e-9 | 3.9e-7 |

**Speed at the L3 context**, idle (µs; the classic kernel after §13.1 is 25.8–
27.2 + 26.8–27.8 ≈ 53):

| score tiles | P.V tiles | score | P.V | finish | total |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 28.0 | 23.2 | — | 51.2 |
| 8 | 1 | 18.6 | 25.1 | — | 43.7 |
| 8 | 2 | 18.0 | 26.1 | 2.2 | 46.3 |
| 16 | 1 | 20.1 | 24.8 | — | 44.9 |
| **32** | **1** | **16.6–17.4** | **25.9–26.9** | — | **42.5–44.3** |
| 32 | 4 | 16.8 | 29.5 | 3.6 | 49.9 |
| 64 | 1 | 29.4 | 26.7 | — | 56.1 |

So **32 × 1 is the default geometry, ≈ 43 µs a layer, not the ≤ 40 asked.**
The score stage is at 17 µs; P.V is where the rest is, and it is the same work
as the classic combine (≈ 26 µs) — the one-pass rewrite and a balanced
tile-max reduction each measured neutral. Getting P.V under that means reading
the KV once for several heads, which needs `16 × G` accumulators a lane; that is
the next step and it was not taken here.

**Speed at long context** (`--kv`, 64 score tiles, 8 P.V tiles, µs a layer):

| n_kv | score | P.V | finish | total |
|---:|---:|---:|---:|---:|
| 4096 | 200 | 336 | 5.8 | **542** |
| 32768 | 1585 | 3323 | 6.5 | **4914** |

These are synthetic lists: the real decode geometry never exceeds window 128 +
`index_topk` 512 = 640 entries whatever the context, which is exactly why
§9.4's indexer, not this kernel, is the one that grows with context. At 32768
entries the score stage reads 21 GB/s of KV and P.V 10 — both are latency, not
bandwidth — and `pv_tiles` 8 against 1 is worth 23% there (sweep: 3251 against
4237 µs), where at 193 entries it was a loss. (`fin_kv4k`, `fin_kv32k`; a sweep
pair earlier read 552 and 4829.)

**The n_heads-grouped classic geometry, re-measured for the runtime.**
`AttnPush::n_heads` with `heads_per_wg` G, after §13.1 (score + combine, µs):
**G = 1: 54, 2: 76, 4: 117, 8: 209.** Still a loss, for the same reason §10.2
gave. The runtime should not adopt it; it should adopt `AttnScoreT` + `AttnPvT`
(44 µs), which gets the traffic reduction grouping was for without giving up
the workgroup count.

### 13.6 The whole layer

| | µs a layer (1–9) | 40 layers + head |
|---|---:|---:|
| P2 step 2 (§10, doc) | 892.6 | 41.4 ms |
| P2 shaders, this session, same binary | 891.0–898.8 | 41.3–41.6 ms |
| + LDS stride fix, classic stages (what the runtime calls today) | 797–804 | 37.6–37.8 ms |
| + K-split, 8-tile attention | 709–711 | 34.0 ms |
| **+ K-split, 32 × 1 tile attention — the P3 defaults** | **696.3–696.8** | **33.5 ms** |

In GB/s: 133 MB in 696 µs is **191 GB/s, 88% of 217**. What is left, by kernel,
at the ceiling: wq_b is there; wo_b and wo_a are at 94–95%; wq_a is at 85%
(5 µs); the attention is 43 µs of latency; the gate's 27 µs and mega_mhc's 11
are outside this track's list.

### 13.7 Interfaces Track I has to adopt

All additive; every stage and push struct that existed is unchanged in layout
and in behaviour (§13.1's layout change is invisible to callers).

* **Eleven `AttnStage` enumerators**, appended before `Count`:
  `WqAKSplit, WqAKCombine, WkvKSplit, WkvKFinish, WoAKSplit, WoAKCombine,
  WoBKSplit, WoBKCombine, AttnScoreT, AttnPvT, AttnFinishT`.
* **Replacements, dispatch by dispatch:**

  | today | P3 | workgroups | push |
  |---|---|---|---|
  | `WqA` | `WqAKSplit` → `WqAKCombine` | `ksplit_groups(WqAKSplit, 1280)` → `combine_groups(1280)` | `KSplitPush{1280, 5120, 160, 0, 1280, 0}` for both |
  | `WkvGemv` → `WkvFinish` | `WkvKSplit` → `WkvKFinish` | `ksplit_groups(WkvKSplit, 512)` → 1 | `KSplitPush{512, 5120, 160, 0, 512, 0}`; `WkvPush{…, n_wg0 = 1, eps, part_stride = 512}` |
  | `WoA` | `WoAKSplit` → `WoAKCombine` | `ksplit_groups(WoAKSplit, 8192)` → `combine_groups(8192)` | `KSplitPush{8192, 4096, 128, 1024, 8192, 0}` |
  | `WoB` | `WoBKSplit` → `WoBKCombine` | `ksplit_groups(WoBKSplit, 5120)` → `combine_groups(5120)` | `KSplitPush{5120, 8192, 256, 0, 5120, 0}` |
  | `AttnScore` → `AttnCombine` | `AttnScoreT` → `AttnPvT` [→ `AttnFinishT` iff `pv_tiles > 1`] | `attn_tile_groups(64, n_tiles)` → `attn_pv_groups(64, pv_tiles)` [→ `attn_finish_groups(64, 512)`] | `AttnTPush{n_kv, 128, 512, 64, score_stride, scale, 64, 32, ceil(n_kv/32), 32768, 1, n_kv}` |

* **New push structs** `KSplitPush` and `AttnTPush`; **`WkvPush` gained a
  trailing `part_stride`** (defaulted 0, read only by `WkvKFinish`).
* **New slots:** `slot::kKsp{W,S,X,Y,Part}` (the first four are the same indices
  as `kGemv*`), `slot::kWkvPart = 10`, `slot::kAttnTileMax = 9, kAttnPartO = 10,
  kAttnPartD = 11` on top of the nine `kAttn*`.
* **New buffers the caller owns:** a partial plane of `ksplit × rows` floats per
  split GEMV (the four can share one of 8 × 8192 floats, since they run in
  sequence); `n_heads × n_tiles` floats of tile maxima; and, only for
  `pv_tiles > 1`, `pv_tiles × 32768` + `pv_tiles × 64` floats of partials.
* **`AttnSpec` gained** `tile_heads_per_wg` (8), `pv_heads_per_wg` (1),
  `ksplit_wq_a/wkv/wo_a/wo_b` (2/2/4/8), `rows_per_lane_ksplit`,
  `fp8_arith_decode` (0), and three `sweep_*` bitmasks that must stay 0 outside
  the bench. `AttnRunner` gained `ksplit()`, `ksplit_groups()`,
  `combine_groups()`, `attn_tile_groups()`, `attn_pv_groups()`,
  `attn_finish_groups()`, `attn_tiled_finish_needed()`.
* **Specialisation constants** 9 (`KSplit`) and 10 (`Fp8Arith`) are new in
  `attn_common.slang`; `PipelineSpec::extra` carries seven values now.
* **Limits:** `n_tiles ≤ 64`; `k / ksplit ≤ 4096` (`kGemvKSplitMaxSlice`);
  split factors are powers of two ≤ 16 (checked in `create`); head_dim 512.

The runtime gets §13.1 (−95 µs a layer) with no change. Adopting the table
above is another −100 µs a layer, 4 ms a token.

### 13.8 Accuracy, all stages, after this section's changes

`gpu_attn.l2_per_stage`, worst of the seven layers, with the P3 defaults; every
P3 stage fed the same golden input as the stage it replaces.

| stage | worst cosine vs the oracle | relative L2 there |
|---|---:|---:|
| mega-mHC pre / post / comb | 1.000000000 | ≤ 2.2e-7 |
| mega-mHC `attn_norm` output | **1.000000000, bit-exact** | 0 |
| `wq_a` / **`wq_a` K-split** | 0.999998495 / **0.999998495** | 1.7e-3 |
| `wq_b` + `q_norm` + RoPE | 0.999997972 | 2.0e-3 |
| `wkv` raw / **K-split** | 0.999998342 / **0.999998342** | 1.8e-3 |
| `wkv` after the fp8 ring write / **K-split** | 0.999859835 / **0.999859835** | 1.7e-2 |
| sparse attention + inverse RoPE / grouped / **tiled, 1 and 4 P.V tiles** | 0.999997368 all four | 2.3e-3 |
| `wo_a` / **K-split** | 0.999998557 / **0.999998557** | 1.7e-3 |
| `wo_b` / **K-split** | 0.999998527 / **0.999998527** | 1.7e-3 |
| `hc_post` into the stream | 0.999998535 | 1.7e-3 |
| mega-mHC `ffn_norm` output | 0.999994284 | 3.4e-3 |
| gate scores | 0.999999996, 6/6 experts on every layer | 9.1e-5 |
| `head`, vs fp64 CPU | 1.000000000 | 2.3e-7 |
| compressor / indexer (§9.2) | unchanged: 11 of 16 bit-exact, the rest ≥ 0.999998575 | |
| **`sparse_attn_t` at 4096 and 32768 entries, vs fp64** | **1.000000000** | ≤ 3.9e-7 |

The `wkv` post-fp8 line is 0.99986 on layer 2, below §0's 0.99991; it is not a
regression — the P2 shader directory gives the same number on the same layer
(§13.1's line-for-line comparison), and §0's table predates step 2. It is the
E4M3 re-rounding of §3.1: 11 of 512 bytes land on the other side of a boundary.

Every shader went through `slangc` and `spirv-val` (the two new ones, the
modified ones, and each sweep variant). The whole `gpu_attn` suite ran once
under the Khronos validation layers, logging to stdout, with the final defaults:
6 cases, 0 failures, **0 validation messages**.

### 13.9 Done / not done

**Done**

* The LDS bank-conflict fix in `fp8_gemv.slang` and `sparse_attn.slang`,
  bit-identical, already live for the runtime.
* `gemv_ksplit.slang` and `wkv.slang` stage 2: the K-split for all four fp8
  GEMVs; wo_b at 205 GB/s, wo_a at 207.
* `sparse_attn_t.slang`: head-group × KV-tile scores, P.V on its own grid,
  finish only when needed; exact at 193, 4096 and 32768 entries.
* One layer 899 → **696 µs**, 40 layers + head 41.6 → **33.5 ms**.
* Measured and switched off: arithmetic E4M3 decode, four accumulators a block,
  folded weight scale, row blocking on split stages, `WaveReduce` flips,
  eight heads a P.V workgroup, the n_heads-grouped classic geometry.
* `gpu_attn.sparse_attn_long_context`, `gpu_attn.fp8_arith_decode_matches_table`,
  P3 checks in `gpu_attn.l2_per_stage`, and `DEEPMOE_SKIP_P3` / the
  `DEEPMOE_KSPLIT_*` / `DEEPMOE_FP8_ARITH` / `DEEPMOE_TILE_HEADS` /
  `DEEPMOE_PV_HEADS` environment knobs on the test.
* `bench/attn_bench`: every P3 stage beside the stage it replaces, a P3 layer
  total, `--ksplit-*`, `--tiles`, `--pv-tiles`, `--pv-heads`, `--kv`,
  `--fp8-arith`, `--rows4`, `--wave-on/off`.

**Not done**

* **Sparse attention ≤ 40 µs**: 43. P.V needs to read the KV once for several
  heads (`16 × G` accumulators a lane).
* **`wq_a` at 85%**: 5 µs to the ceiling.
* The runtime adopting §13.7 (Track I).
* §11 items 4–6 (layer-indexed address table, `mega_mhc.post`, a two-step L2
  export) are unchanged.
