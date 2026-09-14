# P2 step 1 — the non-MoE decode path

> The design §7.2–§7.8 and §7.11 kernels, validated stage by stage against
> `inference/model.py`, plus one whole decoder layer end to end.
> Companion to [kernel_p1.md](kernel_p1.md) (the MoE kernels and the bandwidth
> matrix) and [route_trace.md](route_trace.md) (routing and the cache).

Status: **2026-09-14**. Everything measured here is on the real
`DeepSeek-V4.1-Flash` checkpoint, one decode token at position 64 of a 64-token
prefill. Raw data: `tests/data/l2/`, `bench/results/attn_p2.csv`.

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
| the compressed KV and the indexer's top-k list | **LOADED** from the oracle's prefill. The compressor and indexer kernels of §7.4 are not written. Everything downstream of them is real and is measured |
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

## 7. What is next, in order

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

## 8. Done / not done

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
