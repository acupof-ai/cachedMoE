# P4 / Track F5 — 2-bit routed experts: feasibility (`docs/p4_quant2.md`)

**Verdict: NO-GO at 2 bits, and NO-GO at 3 bits.** Section 5 has the end-to-end
numbers and neither is close. Teacher-forced perplexity on the 64-token L3
prompt, against the checkpoint's 29.26:

| | bits/wt | perplexity | vs FP4 | argmax agreement |
| --- | ---: | ---: | ---: | ---: |
| FP4 (checkpoint) | 4.25 | 29.26 | — | — |
| 3-bit `uni3_b32` | 3.25 | **13,251,329** | x452,957 | 1.6% (1 of 64) |
| 2-bit `int2_b32` | 2.25 | **14,290,394** | x488,475 | 0.0% (0 of 64) |

Section 8 has the recommendation and what to do instead.

**The acceptance criterion here is perplexity** -- whether the model still
produces normal output -- not agreement with FP4. Argmax agreement and KL are
reported alongside as diagnostics because they cost nothing and they say *how*
the distribution moved, but nothing in this document turns on them.

The trade that was on the table: 2-bit routed experts would take decode from
**7.09 to 15.83 tok/s** (section 6), the largest single speedup left in the
project and more than speculation or any kernel change can offer. It is
unavailable. Section 3 explains why that was decidable in 94 seconds, before any
scheme was implemented, and section 9 records what each rung cost.

## 1. The question

Design section 2.2: the 40 x 384 routed experts are **288.8 GB** of FP4 E2M1 +
UE8M0/32, 18,800,640 B each, and design section 3.1 says they are the whole
problem — 4.51 GB of them are read per decode token, the fraction that misses the
expert cache comes off NVMe at 4.5 GB/s, and that stall is what sets tok/s.

Track F1 closed off the alternatives: decode is NVMe-bound, and speculation
cannot reduce miss bytes because a draft that guesses the *token* still has to
read the *experts* the verify step routes to. That leaves exactly two levers,
and 2 bits pulls both:

1. a miss costs half the bytes, so the NVMe term halves at a fixed hit rate;
2. a slot is half the size, so the measured 100 GiB slab pool (design section
   5.2) holds **10,787** experts instead of 5,711 — and the hit rate itself goes
   up along the reuse-distance curve of design section 3.1, from 0.9219 to
   0.9823.

Section 6 prices exactly what that would buy. The rest of this document is about
whether the model survives it, because design section 6's standing rule is
"**use the checkpoint's own precision, one bit unchanged**" and this proposal is
the first serious argument for breaking it.

## 2. Method

**Ground truth is the checkpoint.** `tools/quant2_common.py` never sees a
hypothetical fp16 original — there isn't one. It dequantises an expert exactly
(`tools/oracle.py:dequant_fp4`, which design section 12's L0 pins bit for bit)
and re-quantises *that* fp32 array. Every error in this document is an error
against the bytes on disk.

**The ladder.**

| rung | tool | what it measures |
| --- | --- | --- |
| L1 | `tools/quant2_l1.py` | one expert: weight-space rel L2 / cosine per matrix (w1/w3 = gate/up, w2 = down), and the expert's own FFN output through `dsref.expert_ffn`, so the fp8 activation round trip the FP4 kernel performs is in the loop |
| L3 | `tools/quant2_l3.py` | the **unmodified reference** (`inference/model.py` behind `tools/dsref.py`'s six CPU kernel shims) with one change: a re-quantisation hook on `WeightStore.expert_weights_from_slot`. Teacher-forced over the L3 64-token prompt and 2,048 tokens of the mixed EN/ZH/code corpus: argmax agreement, mean KL vs FP4, perplexity |
| system | `tools/quant2_model.py` | the exact LRU curve at every capacity (recomputed from the same 27,399-token routing trace) and design section 3.1's `t_token` model |
| format | `tools/quant2_golden.py` | golden packed bytes + the dequant reference, `tests/data/quant2/` |

**What the hook does and does not change.** It intercepts the three matrices on
their way out of the weight store, after the exact FP4 dequantisation, and hands
back the re-quantised bf16 values a 2-bit kernel would produce. Routing, the
attention path, the shared expert, the fp8 activation round trip and the
accumulation order are all untouched, so a difference in the output is a
difference the quantisation caused. Two properties are asserted on every run
(`tools/quant2_l3.py --selftest`):

* `bf16(w) == w` exactly for a real dequantised expert — an E2M1 magnitude times
  a UE8M0 power-of-two scale has at most 3 mantissa bits — so the hook quantises
  the checkpoint's own values, not a rounded copy;
* the in-forward torch quantiser fits the same levels and scales as the slow
  numpy path in `quant2_common.py`, with per-block SSE excess ≤ 1e-6 relative.
  (They are not bit-identical: the two reconstruct level *i* with a different
  last-place rounding, and on 0.03% of blocks two power-of-two scale candidates
  tie in SSE exactly and are resolved differently. Both answers are optimal.)

**The harness's own self-test.** `fp4_exact` re-quantises with the fifteen E2M1
magnitudes as the codebook. Because the checkpoint's own UE8M0 scale is one of
the power-of-two candidates the fit searches, it must come back at **exactly
zero error** — and it does, on every expert sampled, weight space and function
space alike. That is what says the harness is measuring the schemes and not
itself. The packing round trip (`pack_codes` -> `dequant2_reference`) is
separately asserted exact for every tensor `tools/quant2_golden.py` writes.

## 3. Why the answer was visible before the first scheme ran

The FP4 weights are **QAT-trained at FP4**. They are not a 16-bit tensor rounded
down, and their code distribution is not concentrated. Over five experts spanning
layers 0/10/20/30/39 — 176,947,200 weights:

| code | value | share | code | value | share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0 | 11.670% | 8 | -0.0 | 0.000% |
| 1 | +0.5 | 7.360% | 9 | -0.5 | 7.361% |
| 2 | +1.0 | 5.287% | 10 | -1.0 | 5.286% |
| 3 | +1.5 | 6.263% | 11 | -1.5 | 6.265% |
| 4 | +2.0 | 8.637% | 12 | -2.0 | 8.636% |
| 5 | +3.0 | 7.625% | 13 | -3.0 | 7.627% |
| 6 | +4.0 | 5.662% | 14 | -4.0 | 5.660% |
| 7 | +6.0 | 3.328% | 15 | -6.0 | 3.333% |

**Entropy: 3.8375 bits per weight.** Fifteen codes carry mass (negative zero is
never emitted), so the ceiling is log2(15) = 3.9069 bits; the checkpoint's code
stream sits at **98.2% of it**. The most common code takes 11.7% of the mass and
the flattest imaginable distribution would take 6.7%. Per-expert entropy is
3.827–3.843 across every expert sampled, so this is a property of the
checkpoint, not of a lucky layer.

Two consequences, and they are the whole report:

* **No lossless recoding exists.** 3.84 bits is the floor for any entropy coder
  on these codes. A 2-bit format is not a recoding — it must discard
  information, and the only question is how much.
* **No codebook of four values can be nearly right.** The four heaviest codes
  together are 36.6% of the weights, so *at best* a 4-entry per-row LUT leaves
  63% of weights sitting on a value they were not trained to. Sections 4 and 5
  are the measurement of what that does; the sign of the answer was never in
  doubt.

This is the difference between this checkpoint and the literature that motivates
2-bit LLM weights. Those results quantise fp16 checkpoints whose weight
distributions are smooth and heavily over-parameterised in precision. Here the
training already spent the bits.

## 4. L1: one expert, every scheme

`tools/quant2_l1.py screen`, the full grid on 40 experts (one per layer). The
column that matters is `y rel L2`: the relative L2 error of the **expert's own
FFN output**, through `dsref.expert_ffn`, so the fp8 activation round trip the
FP4 kernel performs is inside the measurement. Weight-space error is reported
per matrix because a kernel LUT has to live with it, but it is not the quantity
that reaches the model.

| scheme | bits/wt | y rel L2 | p90 | y cos | w1 (gate) | w2 (down) | w3 (up) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `int2_b64` | 2.12 | 0.5749 | 0.5905 | 0.8198 | 0.3596 | 0.3598 | 0.3608 |
| `int2_b32` | 2.25 | 0.5685 | 0.5855 | 0.8229 | 0.3569 | 0.3574 | 0.3578 |
| `actw_int2_b32` | 2.25 | 0.5685 | 0.5860 | 0.8228 | 0.3574 | 0.3574 | 0.3584 |
| `int2_b16` | 2.50 | 0.5559 | 0.5748 | 0.8314 | 0.3489 | 0.3495 | 0.3494 |
| `lloyd_ten_b32` | 2.25 | 0.5520 | 0.5681 | 0.8340 | 0.3457 | 0.3461 | 0.3465 |
| `lloyd_row_b32` | 2.27 | 0.5504 | 0.5673 | 0.8349 | 0.3454 | 0.3457 | 0.3463 |
| `actw_lloyd_b32` | 2.27 | 0.5501 | 0.5684 | 0.8352 | 0.3457 | 0.3457 | 0.3467 |
| `int2_b32_e4m3` | 2.25 | 0.5461 | 0.5635 | 0.8379 | 0.3426 | 0.3433 | 0.3427 |
| `lloyd_row_b16` | 2.52 | 0.5391 | 0.5579 | 0.8429 | 0.3376 | 0.3378 | 0.3380 |
| `mix05_lloyd_b32` | 2.37 | 0.5289 | 0.5495 | 0.8487 | 0.3336 | 0.3359 | 0.3350 |
| `mix12_lloyd_b32` | 2.52 | 0.5029 | 0.5261 | 0.8642 | 0.3175 | 0.3213 | 0.3193 |
| `mix25_lloyd_b32` | 2.77 | 0.4607 | 0.4891 | 0.8873 | 0.2908 | 0.2961 | 0.2930 |
| `uni3_b32` (3-bit) | 3.25 | 0.3523 | 0.3633 | 0.9433 | 0.2051 | 0.2057 | 0.2042 |
| `lloyd3_row_b32` (3-bit) | 3.29 | 0.3197 | 0.3311 | 0.9479 | 0.1935 | 0.1941 | 0.1928 |
| `uni4_b32` (4-bit uniform) | 4.25 | 0.2575 | 0.2641 | 0.9807 | 0.1295 | 0.1296 | 0.1286 |
| `lloyd4_row_b32` (4-bit) | 4.32 | 0.1595 | 0.1655 | 0.9872 | 0.0940 | 0.0935 | 0.0930 |
| **`fp4_exact`** (self-test) | 4.16 | **0.0000** | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

Read down that table and the shape of the problem is clear.

**Every 2-bit scheme lands between 0.46 and 0.57.** Thirteen schemes spanning
block size, scale format, codebook fitting, mixed precision and activation
weighting move the output error by less than a quarter, and none gets near a
number a model could survive. Specifically:

* **Lloyd-Max buys 3%.** Fitting the codebook to each row's own distribution
  (`lloyd_row_b32`, 0.5504) instead of using the fixed symmetric grid
  (`int2_b32`, 0.5685) is worth 3% of the error. There is no codebook to find:
  four values cannot represent fifteen that all carry mass (section 3).
* **Activation weighting buys nothing.** `actw_lloyd_b32` (0.5501) against
  `lloyd_row_b32` (0.5504), and `actw_int2_b32` is identical to `int2_b32` to
  four decimals. Weighting the objective by E[x_c²] moves which weights are
  sacrificed, but with 63% of them being sacrificed there is nothing left to
  trade.
* **Finer blocks buy little and cost bits.** b16 over b32 is 0.5559 vs 0.5685
  for a quarter of a bit — the same exchange rate as everything else here.
* **A finer scale format buys little.** `int2_b32_e4m3` (0.5461) says the
  power-of-two scale is costing about 4% of the error, not the problem.
* **Mixed precision buys exactly what it pays for and no more.** `mix25`
  keeps the worst 25% of rows at FP4 — 0.52 extra bits, a 23% bigger expert —
  and returns 0.5504 -> 0.4607, a 16% error reduction. Section 4.1 says why
  that was the ceiling.

**And 3 bits is not a rescue either.** `uni3_b32` is 0.3523 and
`lloyd3_row_b32` 0.3197: a third of the expert's output, still gone. Section 5
measures what that does end to end.

**The one genuinely diagnostic control** is `uni4_b32` vs `fp4_exact`. Both spend
4 bits on the payload. The checkpoint's own E2M1 grid reproduces the weights
*exactly* (0.0000, which is the harness self-test); a **uniform** 4-bit grid at
the same cost gives 0.2575. The format matters more than the bit count, and the
checkpoint is already using the right format for its own weights. That is the
most compact statement of why this track ends where it does.

### 4.1 The error is not concentrated anywhere

`tools/quant2_concentration.py`, `int2_b32`, five experts spanning layers
0/10/20/30/39. Share of total squared error held by the worst fraction of units
(a perfectly uniform tensor puts the share **equal to** the fraction):

| unit | worst 5% | worst 12.5% | worst 25% | worst 50% |
| --- | ---: | ---: | ---: | ---: |
| rows | 0.0615 | 0.1445 | 0.2764 | 0.5284 |
| blocks of 32 | 0.1093 | 0.2290 | 0.3949 | 0.6611 |
| *uniform* | 0.05 | 0.125 | 0.25 | 0.50 |

Per-row relative L2 across all fifteen matrices: **mean 0.3583, sd 0.0053**, p1
0.3470, p99 0.3711. The spread from the first to the ninety-ninth percentile is
7% of the mean.

This is the result that closes off the mitigations, and it is worth being blunt
about what it means. Every proposal of the form "keep the important part in
higher precision" needs an important part to exist. Here there isn't one: no
row, no block, no matrix (gate, up and down are within 0.001 of each other), no
layer (0.551 at layers 0-9 rising only to 0.576 at 20-29) and no expert is
meaningfully better or worse than any other. Protecting x% of the rows removes
about x% of the error because that is all there is to remove. `mix25`'s measured
16% return on 25% of the rows is exactly this arithmetic, and it is why no
cleverer saliency metric would have changed the answer.

### 4.2 The same schemes on real activations

The screen above drives every expert with a synthetic input built from the
layer's own `ffn_norm.weight` (a unit-RMS Gaussian times `w` has the right
per-channel scale, which is the only property of `x` a per-channel objective can
use). The FP4 L3 control also dumped a calibration bank of **real** expert inputs
for all forty layers, so the finalists were re-run against those.

Stopped at **32 experts covering layers 0-5** on the owner's time budget (the
per-expert checkpoint is the record, `bench/results/quant2/l1_sweep_partial.json`).
It is therefore *not* layer-spread, and layers 0-9 are the best case in the
screen, so read these as the optimistic end:

| scheme | bits/wt | y rel L2 (real x) | sd | max | y cos min | screen, L00-09 (synthetic x) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `int2_b32` | 2.25 | 0.5021 | 0.0202 | 0.5343 | 0.8468 | 0.5513 |
| `lloyd_row_b32` | 2.27 | 0.4854 | 0.0192 | 0.5168 | 0.8567 | 0.5351 |
| `mix25_lloyd_b32` | 2.77 | 0.4039 | 0.0244 | 0.4450 | 0.8963 | 0.4495 |
| `uni3_b32` (3-bit) | 3.25 | 0.3146 | 0.0101 | 0.3331 | 0.9506 | 0.3423 |
| **`fp4_exact`** | 4.16 | **0.0000** | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

Real inputs are kinder than synthetic ones by about 0.05 — the activations
concentrate on directions the quantisation happens to damage slightly less — and
it changes nothing: 2 bits still loses half the expert's output and 3 bits still
loses a third. `fp4_exact` is exactly zero on real activations too, which is the
self-test passing on the path that actually matters.

## 5. L3: the model, end to end

`tools/quant2_l3.py`, teacher-forced. The **acceptance criterion is perplexity**
— does the model still produce normal output — not agreement with FP4. Agreement
and KL are reported alongside because they cost nothing to compute and they say
*how* the distribution moved, but they are diagnostics, not the bar.

The FP4 row is the control, and it is a real control: the unmodified path
reproduces the L3 golden exactly (argmax 3006, logit 28.719 against the golden's
28.712, a half-ULP fp16 storage difference), so the pipeline under test is known
to be the reference when nothing is quantised.

### 5.1 The 64-token L3 prompt (mixed EN / ZH / code)

| scheme | bits/wt | **perplexity** | vs FP4 | argmax agree | mean KL | run time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **FP4** (checkpoint) | 4.25 | **29.26** | — | — | — | 388 s |
| `uni3_b32` (3-bit) | 3.25 | **13,251,329** | **x452,957** | **1.6%** | 14.09 nats | 2,539 s |
| `int2_b32` (2-bit) | 2.25 | **14,290,394** | **x488,475** | **0.0%** | 14.26 nats | 1,777 s |

**3 bits fails too, and by the same margin.** This is the result that settles
the track. `uni3_b32` loses "only" a third of each expert's output at L1 (0.3146
on real activations) and still comes out at perplexity **13.25 million**, with
the argmax matching FP4 at **1 of 64** positions. Forty layers compose: a 31%
error per expert, applied six experts deep, forty times, does not stay a 31%
error. Design section 12's reason for having an L3 rung at all -- that a
per-kernel test cannot see what happens when forty layers of them are composed --
is exactly what this row demonstrates, and it means **L1 error of even 0.3 is
already fatal**, which is a far tighter constraint than anyone assumed going in.

**2 bits is not a degradation, it is a destruction.** Perplexity goes from 29 to
fourteen million. The argmax matches FP4 at **zero** of the 64 positions — not a
low agreement, no agreement — and the variant's own argmax is not even in FP4's
top 5 at any position. At the L3 golden's position the model wants token 104113
at logit 15.82 where the checkpoint wants 3006 at 28.72. A mean KL of 14.3 nats
against a distribution over 129,280 tokens means the two distributions have
essentially nothing to do with each other.

Both perplexities are far worse than the 129,280 a uniform distribution would
give, which is the signature of a model that is confidently wrong rather than
merely uncertain -- the collapsed network still produces large logits, just for
the wrong tokens.

No further measurement on either scheme was justified. On the owner's
instruction the 2,048-token passes and the free-running samples were cancelled
rather than run to completion, and the 3-bit run was stopped as soon as its
64-token perplexity failed the gate (within ~2x of FP4's 29.26). Section 3 and section 4.1 already say why no variant of it would
land anywhere else: the checkpoint's codes carry 3.84 bits of entropy, and the
error a 4-value codebook makes is spread perfectly evenly over every row, block,
matrix, layer and expert.

### 5.2 Free-running generation

`tools/quant2_gen.py` prefills a short prompt and greedily decodes with the model
reading its own output, so errors compound instead of being re-anchored on the
real token every step. The weights it runs go through the **packed 2-bit format
of section 7** on every use — codes, block scale, dequantisation — so the samples
exercise the proposed format rather than a float shortcut; the round trip is
asserted equal to the direct fit on the first expert of every run.

**Not run, on the time budget.** The measured cost is the reason, and it is worth
recording because it is a property of the harness, not of the schemes:

| stage | measured |
| --- | ---: |
| build 40 Blocks (once, 5.1 GB of attention weights) | 56 s |
| prefill a 19-token prompt, FP4 | > 4 min (killed) |
| one decode step, 240 experts | ~30–55 s |
| a 64-token sample, one prompt, one scheme | ~40 min |

A free-running sample through the CPU reference costs tens of minutes per prompt
per scheme, because every decode step re-reads and re-dequantises 4.5 GB of
experts in Python. For 2 bits the sample would have been a formality — a model at
perplexity 14 million and zero argmax agreement does not produce text — so it was
cancelled rather than spent. If 3 bits is ever revisited, this is the rung to run
first and it should be budgeted at roughly an hour for three prompts.

The tool, the packed codec and the block-prebuild restructure are committed and
tested, so that run is a single command when someone wants it.

## 6. System model: what 2 bits would have bought

`tools/quant2_model.py`. The LRU hit rate is the exact stack-distance curve
recomputed from the same 27,399-token routing trace (6,575,760 expert accesses,
Fenwick tree, 65 s), evaluated at every capacity so the 2-bit capacities are read
off the same curve as the FP4 one rather than extrapolated. It reproduces
`reports/cache_sweep.json`'s published capacities to within **0.0019** absolute
at every point. Cold misses are 0.231% of accesses, so the ceiling on any hit
rate is 0.9977.

`t_token` is design section 3.1's, unchanged:

```
t_token = 42 ms                    resident weights, 8.5 GB @ 216 GB/s
        + h       x B / 200 GB/s   hit: expert read from LPDDR
        + (1 - h) x B / 4.5 GB/s   miss: expert read from NVMe
```

with `B` = 6 experts x 40 layers x the expert size. Note the MoE kernel does not
appear as a separate term and should not: section 7.9.3 measures it at 25 ms for
FP4 and that time *is* the expert read at LPDDR bandwidth, which is the `h` term.
Halving the bytes halves it too (25 -> 13.2 ms), exactly as the model already
accounts for.

| bits/wt | MB/expert | slots in 100 GiB | hit rate | GB/token | NVMe stall | ms/token | **tok/s** | all experts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **4.25** (today) | 18.80 | 5,711 | 0.9219 | 4.512 | 78.3 ms | 141.1 | **7.09** | 288.8 GB |
| 2.50 | 11.06 | 9,709 | 0.9745 | 2.654 | 15.0 ms | 69.9 | **14.30** | 169.9 GB |
| 2.35 | 10.40 | 10,328 | 0.9791 | 2.495 | 11.6 ms | 65.8 | **15.20** | 159.7 GB |
| **2.25** (proposed) | 9.95 | 10,787 | 0.9823 | 2.389 | 9.4 ms | 63.2 | **15.83** | 152.9 GB |
| 2.125 | 9.40 | 11,422 | 0.9862 | 2.256 | 6.9 ms | 60.0 | **16.66** | 144.4 GB |

**2.23x**, 7.09 -> 15.83 tok/s. And the two effects are not equal: holding the
hit rate at FP4's 0.9219 and only halving the bytes gives 10.58 tok/s (1.49x),
so **roughly half the win comes from the extra cache slots**, not from the
narrower read. That matters for section 8, because it means anything that buys
slots without touching precision is chasing the same half of the prize.

It is worth being explicit that this is the largest single number anywhere in
the project's decode work, and that is precisely why the accuracy rungs had to be
run properly rather than argued about.

## 7. The storage format, if it were ever built

Per tensor `[rows, K]`, K contiguous along the reduction axis — the FP4 layout
with a narrower payload, so the addressing, the manifest arithmetic and the slot
packing are all unchanged:

```
codes   [rows, K/4]  u8   4 codes per byte, element j in bits 2*(j % 4),
                          element 0 in the LOWEST bits (the same
                          low-nibble-first convention as the FP4 unpack)
scales  [rows, K/32] u8   UE8M0, value = 2^(code - 127) -- bit-identical in
                          shape and meaning to what FP4 already stores
lut     [rows, 4]    f16  the per-row codebook, ascending

value(r, j) = lut[r][code(r, j)] * 2^(scale[r][j / 32] - 127)
```

Effective bits per weight: `2 + 8/32 = 2.25`, plus `4 x 16 / K` for the per-row
LUT (0.0125 bit at K = 5120, 0.0278 at K = 2304) — **2.25–2.28 bits** against
FP4's 4.25. An expert is 9,953,280 B against 18,800,640 B, still a whole number
of 4 KiB pages (2,430 against 4,590), so design section 5.1's whole-sector reads
need no change. The kernel change is the unpack step plus a 4-entry LUT in
registers or LDS; the scale plane, the K-major walk and the fp32 accumulation are
untouched.

`tests/data/quant2/` carries the golden bytes: 3 real experts (L00/E000,
L20/E191, L39/E383) x 2 schemes (`int2_b32`, `lloyd_row_b32`), 64 rows of each of
w1/w2/w3, 1.41 MB total. Each file has the packed codes, the UE8M0 scale plane,
the per-row LUT, a 2 x 256 window of the reference dequantisation for direct
value comparison, and hash64/sha256 over every part, so a C++ `test_dequant2.cpp`
could be written against it without re-deriving anything. (The golden LUT is
stored f32 for test convenience; the proposed on-disk format is f16, which is
where the 0.0125 bit/weight figure comes from.)

**Offline conversion.** Reading all 288.8 GB of FP4 at the measured 4.5 GB/s is
64 s and writing 152.9 GB back is a couple of minutes; the cost is the fit. At
the in-forward torch path's measured 0.41 s per expert (16 threads, 7 scale
candidates) the 15,360 routed experts are **~1.8 h**, or ~1 h with the
`{-2,-1,0,+1}` candidate set section 9 justifies — and it is embarrassingly
parallel across experts. **Disk: 152.9 GB.** D: has 195 GB free and already
holds the FP4 checkpoint, so the converted copy fits alongside it without
touching C: (257 GB free). (An earlier estimate of ~215 GB for the 2-bit
experts was wrong; 288.8 GB x 2.25/4.25 = 152.9 GB.)

## 8. Recommendation

**NO-GO on 2-bit routed experts.** Not "risky", not "needs a better scheme":
perplexity 29 -> 14,290,394 and zero argmax agreement on 64 positions. The
speedup it would have bought (7.09 -> 15.83 tok/s, 2.23x) is the largest number
in the project's decode work, and it is unavailable at any accuracy anyone would
accept. Thirteen schemes were tried; the spread between the best and worst of
them is smaller than the distance either one is from usable.

**The reason is structural, not a failure of scheme design.** The checkpoint is
QAT-trained at FP4: its code stream carries 3.8375 bits of entropy against a
log2(15) = 3.9069 ceiling, the top four codes hold 36.6% of the mass, and the
error a 4-value codebook makes is spread perfectly evenly across rows, blocks,
matrices, layers and experts (section 4.1). There is no redundancy to compress
and no salient subset to protect. Design section 6's "use the checkpoint's own
precision, one bit unchanged" survives its first serious challenge.

**On 3 bits: see section 5.1 for the measured number.** The L1 evidence is
already discouraging — `uni3_b32` loses a third of each expert's output (0.3146
rel L2 on real activations, 0.3523 on the 40-layer screen) — and the prize is
smaller: 10.49 tok/s rather than 15.83, a 1.48x rather than 2.23x, for 220.8 GB
that no longer fits on D: beside the FP4 checkpoint and would have to move to C:.

**What to do instead.** Section 6 contains the useful redirection. Holding the
hit rate at FP4's 0.9219 and only halving the bytes gives 10.58 tok/s, while
holding the bytes and only doubling the slots gives most of the rest — **about
half the 2-bit prize is the extra cache slots, not the narrower read**, and slots
are available without touching a single weight:

* more slab for the expert cache, which is a memory-budget decision and costs no
  accuracy at all;
* better replacement than global LRU — the curve in section 6 is LRU's, and the
  cold-miss floor is 0.231%, so there is real headroom between 0.9219 and 0.9977;
* admission and prefetch policy, which F1 already identified and which is not
  bounded by anything in this document.

Those are the tracks that inherit this one's speedup without inheriting its
accuracy cost.

## 9. Run log — what each rung cost

Wall clock on the 32-core machine, 2026-09-18. Several runs overlapped; where
they did, the time is not a clean single-job measurement and is marked.

| experiment | scope | cost | |
| --- | --- | ---: | --- |
| FP4 code entropy + error concentration | 5 experts, layers 0-39 | **94 s** | `quant2_concentration.py` |
| system model (LRU curve + t_token) | 6,575,760 accesses, every capacity | **65 s** | `quant2_model.py`, curve cached after |
| golden packed bytes | 3 experts x 2 schemes | **4 s** | `quant2_golden.py` |
| L1 screen | 40 experts x 18 schemes, synthetic x | 3,241 s | one expert per layer |
| L1 sweep (stopped) | 32 of 216 experts x 5 schemes, real x | ~1,900 s | ~59 s/expert, overlapped |
| L3 FP4 control | 64 + 2,048 tokens | 1,486 s | 388 s + 1,079 s |
| L3 `int2_b32` | 64 tokens | 1,777 s | 4.6x the FP4 pass, overlapped |
| L3 `int2_b32` 2,048 tokens | cancelled | — | 265 s/layer measured, ~2 h projected |
| L3 `uni3_b32` | 64 tokens | 2,539 s | 3-bit control, partly overlapped |
| L3 `uni3_b32` 2,048 tokens | cancelled | — | gate failed at 64 tokens |
| free-running sample | cancelled | — | ~40 min/prompt/scheme measured |

**The cheap experiments are the ones that decided this.** The code histogram and
the concentration analysis together cost **94 seconds** and contain the entire
argument: 3.84 bits of entropy and a perfectly uniform error. The L1 grid (54
minutes) confirmed it across eighteen schemes; the L3 passes (30+ minutes each)
confirmed it end to end. Nothing after the first 94 seconds changed the sign of
the answer.

**What would have made this faster.** The 64-token L3 pass is the experiment
that actually decides a format, and at ~30-40 min it is the wrong size for a
first look. Two things would fix that for the next one: run the gate on the
*smallest* prompt that composes all forty layers (a 16-token prompt touches far
fewer unique experts per layer and would have returned the same verdict in under
ten minutes), and screen on L1 against the 0.31 tolerance this track established
before scheduling any L3 at all.

**One control not run.** The FP4 path through this driver reproduces the L3
golden bit-exactly, and the in-forward quantiser is asserted against an
independent numpy implementation, so the two failing rows are trusted. The check
that would close the last gap is `uni4_b32` (uniform 4-bit, L1 error 0.2575)
through the same hook: it should land between FP4 and `uni3_b32`, confirming the
pipeline degrades monotonically rather than failing all-or-nothing. It is one
~30 min run and is the first thing to do if anyone doubts these numbers.

That ordering is worth keeping for the next format question. Entropy of the code
stream and concentration of the induced error are both one-expert measurements,
they need no forward pass, and they bound what any scheme in that family can
achieve before a single one is implemented.

**Why the L3 rung is slow.** It runs the reference on CPU: 40 layers, an 18.8 MB
expert per routed slot, re-quantised on the way out of the weight store. A
2,048-token teacher-forced pass touches ~15,360 experts. The in-forward
quantiser was made ~4x faster during this track (arithmetic level assignment
instead of a 94 MB gather per scale candidate, and a scale search narrowed to the
four candidates that ever win) and a pass is still half an hour. Any future
accuracy question that needs L3 should budget hours, or run on the GPU path.
