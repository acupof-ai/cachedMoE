# Draft-head screening: FP8 eligible for GPU evaluation

The owner resumed serial GPU work on 2026-10-07. The production CPU-route
native capture completed 64 outputs / 25 cycles, with 38/49 accepted draft
positions. CPU BF16 reproduces all native top-1 proposals. FP8 preserves the
estimated acceptance; 16K / 32K / 64K vocabulary subsets lose 16.327 / 8.163 /
4.082 percentage points and are NO-GO. Only FP8 enters GPU evaluation.

An unprofiled balanced M=2 head micro-benchmark measures **8.903326 ms**,
148.689 effective GB/s: one warmup, one timestamped eight-call batch. The
bandwidth prediction is 4.448185 ms saved, **2.224093 ms after halving**.
This passes the 2 ms estimate gate; it is not measured FP8 speed. The capture
uses profiling only for correctness and is not a speed result. Target math,
final-token identity, actual FP8 speed and the copy's cache bill remain gates.

## GPU candidate (2026-10-07)

The row-FP8 kernel measures **3.539821 ms** / **187.137 effective GB/s** on the
same two captured rows: 5.363506 ms below the native head. Conversion bytes and
float32 scales have exactly the CPU-reference hashes. GPU-vs-CPU FP8 logit
maximum error is 0.0000153, RMS 0.00000160; both row argmax IDs agree.

`CACHEDMOE_DSPARK_HEAD_FP8=1` opts into a startup-only copy and pipeline.
The default is off; mega + FP8 and FP8 without speculation are rejected before
GPU initialization. ONECB snapshots the FP8 head's pointers with all other
operations; each candidate cycle still calls the target exactly once.

The **actual cache bill is 100 slots**, not the theoretical ceiling of 36:
the existing pool allocates 100-slot slabs. Reserve one 1,880,883,200-byte slab
from the same budget, leaving 5400 total / 5016 main-model slots after MTP pins.
The 662,430,720-byte copy leaves 1,218,452,480 bytes unused in that reservation.
Target BF16 stays resident. This rounding is explicit at startup; default
allocation is unchanged. Short whole-store hits are 85.388% native and 85.400%
FP8; these include draft accesses and do not establish a long-run hit cost.

Full native-5500 vs FP8-5400 64-output captures have identical IDs, 25 cycles,
38/49 accepted and one target forward/cycle. Both are instrumented correctness
runs, not a throughput comparison. CPU CTest 31/31 and tools 55/55 pass.
Off NLL, decode/DSpark baseline checks and the conditional same-engine eight-turn
comparison remain pending. The first quality-launch preflight refused to run
beside CPU CTest; it generated no output and its original log is preserved.

The complete power comparison selects **balanced**: 80.944 raw ms/token versus
113.801 for performance. That selection includes thermal pauses. See
[the power report](power_profile_comparison.md).

## What is implemented and checked

`draft_head_capture` exports the existing draft `head_norm_out` and target
verification rows on one greedy 64-output, k=2 path. It checks exactly one target
forward per cycle and both live checkpoint read sources. Its optional engine
observer copies logits after the existing fence. When unset, it adds no readback.
No shader, target-head tensor, acceptance rule or default numerical path changed.

The existing adaptive-k=0 GPU case passed with the observer: two cycles, one
forward each, recorded argmax equal to the existing GPU result, zero skips or
thermal pauses. **That test used GPU routing (route=1), not the production CPU
route.** Its tested worktree engine/test binary remains unchanged. The later native
capture initialized the GPU and read pinned weights, then rejected route=1
before session creation, prefill or generation: **zero output tokens, no hidden
capture**. Both original logs and the failed plan are preserved.

The cause was the production configuration factory using the generic route=1
when its argument was omitted, while the web launcher explicitly selected 0.
`profile_controls` now obtains the production default from `profile_default`;
explicit experiment overrides remain supported. The capture's policy check now
runs before GPU initialization, after the engine resolves its startup snapshot.

CPU validation: build successful, CTest **31/31**, standard gates **55/55**,
offline reference **11/11**, configuration checks **17/17**. All **52 SPIR-V
files** match the main checkout byte for byte. Six original user transcript/KV
files retain SHA, size and mtime. [Machine receipt](draft_head_screening_receipt.json).

## CPU weight check and memory bill

A read-only CPU pass over the actual BF16 `head.weight` (129280 × 5120) computes
per-row FP8 E4M3FN RNE with float32 `amax/448` scales. Zero rows use scale=1;
there are none in this tensor. The source is unmodified, all decoded weights are
finite, maximum absolute weight error is 0.0479911, and squared error/source
energy is **0.07005%**. Weight error does **not** establish token or acceptance
quality. CPU wall time was 2.575 s with eight threads; this is not GPU throughput.

| Draft copy | Bytes including scales/IDs | Slot equivalents |
|---|---:|---:|
| Full BF16 source | 1,323,827,200 | already resident for target |
| Row FP8 + float32 scales | 662,430,720 | 35.219 |
| 16,384 BF16 rows + uint32 IDs | 167,837,696 | 8.923 |
| 32,768 BF16 rows + uint32 IDs | 335,675,392 | 17.847 |
| 65,536 BF16 rows + uint32 IDs | 671,350,784 | 35.693 |

A derived draft copy is **additional** memory: the shared target BF16 head must
remain resident. These are byte equivalents at 18,808,832 bytes/slot, not a
measured allocator-headroom or cache-hit verdict. Allocation padding and any
required whole-slot reduction belong in the later GPU implementation bill.

The local independent frequency corpus contains 23,055 tokens / 5,123 unique
IDs, from long-context fixtures and earlier MMLU/questions/output traces. It is
not a representative training corpus and contains no new capture outputs.
Unseen-token ties use ascending ID. Its hashes and that limitation are recorded.

## Earlier preparation (historical)

A corrected, **unexecuted** CPU-route plan is prepared at
`bench/results/mask_quality/draft_head_cpu_route_prepared/plan.json`, with a new
output directory. It pins ONECB=1, GPU_ROUTE=0, dynamic mask, 5500 slots, k=2,
balanced, explicit primary/mirror and the worktree's binary/shaders. Original
failed inputs/results are not overwritten.

After the CPU-only stage ends and the normal GPU/dual-NVMe start gates are met,
run the prepared plan serially, with the web still stopped:

```bash
python3 bench/thermal_guard.py --plan \
  bench/results/mask_quality/draft_head_cpu_route_prepared/plan.json
```

Then use `bench/draft_head_offline.py` with `--model`, `--mirror`, `--capture`,
`--corpus-ids`, `--head-ms` and a fresh `--out` outside all checkpoint roots.
Supply the measured unprofiled draft-head cost; the historical 8.4 ms and an
instrumented capture wall time must not be relabelled as a new balanced
measurement. The capture's instrumentation is for correctness, not speed.

The offline tool reproduces the kernel's BF16 activation staging before CPU
FP32 dot products. Markov tensor names/rank come from checkpoint configuration;
each variant uses its own preceding proposal for the second row's Markov bias.
A **100% native BF16 top-1 match** gates the CPU reference. Partial/misaligned
captures, invalid IDs, nonfinite data and inconsistent native acceptance are
rejected. Agreement excludes k=0 and unused padded positions.

Candidate acceptance is a **fixed-trajectory estimate** against the existing
native target top-4 matrix. It stops at the first rejection and reports target
IDs outside the vocabulary subset as misses. Changed proposals change later
contexts, so that matrix does not prove counterfactual end-to-end quality.

Only a candidate with halved predicted savings **≥2 ms/cycle** and estimated
acceptance loss **≤3 percentage points** can enter GPU implementation. FP8 is eligible under the fresh cost above. GPU implementation, micro-benchmark, unchanged
final-token IDs, off NLL `.622784`, decode/DSpark baselines and same-engine
balanced eight-turn comparison remain pending. No new runtime quantization
switch or default was added.

The current top-4 acceptor emits accepted draft IDs directly. Changing proposals
can therefore change final output even with an untouched target head. The
owner's final-token identity gate remains mandatory; passing a short test would
not establish universal losslessness. See owner TODO §4.9.

Web restoration with balanced and actual `/api/config`/power-profile acceptance
also remains pending. User pages and transcripts were not refreshed or edited.
