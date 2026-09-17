# P4 / Track F5 — 2-bit routed experts: feasibility (`docs/p4_quant2.md`)

**Verdict: NO-GO at 2 bits, and NO-GO at 3 bits.** Section 4 has the numbers;
section 8 has the recommendation and what to do instead.

> Status: RESULTS PENDING — this file is written as the runs complete. Every
> table marked `pending` has no data behind it yet.

## 1. The question

Design section 2.2: the 40 x 384 routed experts are **288.8 GB** of FP4 E2M1 +
UE8M0/32, 18,800,640 B each, and design section 3.1 says they are the whole
problem — 4.51 GB of them are read per decode token, the fraction that misses the
expert cache comes off NVMe at 4.5 GB/s, and that stall is what sets tok/s.

The owner's proposal: re-quantise them to **2 bits**. That does two things at
once, and both help.

1. A miss costs half the bytes, so the NVMe term halves at a fixed hit rate.
2. A slot is half the size, so the measured 100 GiB slab pool (design section
   5.2) holds **10,787** experts instead of 5,711 — and the hit rate itself goes
   up, along the reuse-distance curve of design section 3.1.

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
| L1 | `tools/quant2_l1.py` | one expert: weight-space rel L2 / cosine per matrix, and the expert's own FFN output through `dsref.expert_ffn` (so the fp8 activation round trip the FP4 kernel performs is in the loop) |
| L3 | `tools/quant2_l3.py` | the **unmodified reference** (`inference/model.py` behind `tools/dsref.py`'s six CPU kernel shims) with one change: a dequant hook on `WeightStore.expert_weights_from_slot`. Teacher-forced over the L3 64-token prompt and 2,048 tokens of the mixed EN/ZH/code corpus: argmax agreement, mean KL vs FP4, perplexity |
| system | `tools/quant2_model.py` | the exact LRU curve at every capacity (recomputed from the same 27,399-token trace, cross-checked against `reports/cache_sweep.json`) and design section 3.1's `t_token` model |
| format | `tools/quant2_golden.py` | golden packed bytes + the dequant reference, `tests/data/quant2/` |

**Self-tests.** `fp4_exact` in the scheme grid re-quantises with the fifteen E2M1
values as the codebook; because the checkpoint's own UE8M0 scale is one of the
power-of-two candidates the fit searches, it must come back at **exactly zero
error** — it does, on every expert sampled, which is what says the harness is
measuring the schemes and not itself. The packing round trip
(`pack_codes` -> `dequant2_reference`) is asserted exact for every tensor
`tools/quant2_golden.py` writes.

## 3. Why the answer was already visible before the first scheme ran

The FP4 weights are **QAT-trained at FP4**. They are not a 16-bit tensor rounded
down, and their code distribution is not concentrated: all sixteen E2M1 codes
carry real mass.

`pending: measured code histogram and entropy table`

The Shannon entropy of the code stream is the floor for any *lossless* recoding.
If it is near 4 bits, then no 2-bit format is a recoding of these weights — it
must discard information — and the only question left is how much that costs.

## 4. L1: scheme table

`pending: screen (40 experts, one per layer, full grid) and sweep (>= 200 experts)`

## 5. L3: teacher-forced agreement and KL

`pending`

## 6. System model: what 2 bits would have bought

`pending`

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
lut     [rows, 4]   f16   the per-row codebook, ascending

value(r, j) = lut[r][code(r, j)] * 2^(scale[r][j / 32] - 127)
```

Effective bits per weight: `2 + 8/32 = 2.25`, plus `4 x 16 / K` for the per-row
LUT (0.0125 bit at K = 5120, 0.0278 at K = 2304) — call it **2.25–2.27 bits**,
against FP4's 4.25. Kernel change: the unpack step and a 4-entry LUT held in
registers or LDS; the scale plane, the K-major walk and the fp32 accumulation are
untouched. `tests/data/quant2/` carries golden bytes plus the reference
dequantisation for three real experts under two schemes.

## 8. Recommendation

`pending`

## 9. Run log (CPU time)

`pending`
