# Verify cost and draft head controls — 2026-10-07

The corrected experiment confirms that the FP8 draft head can change final
IDs under top-4 acceptance. It stays opt-in and off by default. Reducing MoE
allocation width from six to three columns saves too little to pass the entry
gate; the runtime kernels and dynamic LRU defaults remain unchanged.

## Quality control

`bench/draft_head_compare.py --control exact-target` uses `off-spec`: the
main model waits for every routed expert. Both arms use the same engine,
BF16 target, k2/top4 and 5400-slot copy reservation. This control removes
asynchronous main-expert availability from the final-ID comparison; cache
carry-over still makes its elapsed times unsuitable for speed comparisons.

Native completed 64 outputs. FP8 changed output 22 from 76111 to 104505 and
was cancelled (26 outputs after buffered tokens). At cycle position 48,
root/first-draft input routes match in all 40 layers. The second draft token
changes: native target rank 3, FP8 rank 0; both satisfy rank < 4 and are emitted.
This is a head-sensitive output difference under the existing acceptance rule,
not the old experiment's cache-warmth ambiguity. Zero load failures.

## Frozen-cache cost control

An explicit heat file ranks all 15,360 main experts from the recorded native
814-cycle balanced run. The store fills before READY. The benchmark drains
queued IO outside the timed request and records every slot's layer/expert/pin
identity before and after each arm. Dynamic or partially filled caches are
rejected. All four snapshots have the same hash; fills stay 5400, with zero
replacements, pending reads and failed fills. This is an experimental control,
not a fixed-cache default or a production quality claim.

Both arms produce the same 64 IDs, with 35/54 accepted drafts and 28 cycles.
One rejected draft row differs in tokens/ranks/routes; the other 27 cycles
match these fields. Equal acceptance is not identical work in every row.

| Cost | Native BF16 | FP8 |
|---|---:|---:|
| Raw ms/decode token | 66.523251 | 71.453364 |
| Raw ms/cycle | 149.677314 | 160.770069 |
| Active ms/cycle estimate | 147.183277 | 142.574134 |
| Draft wall ms/cycle | 19.383639 | 14.247389 |
| Verify wall ms/cycle | 128.508599 | 144.858563 |
| Decode thermal pause ms | 69.833040 | 509.486198 |

Subtract cooling only from the whole decode window, not individual stage
buckets. The active estimate is not a default-selection score. The raw arm is
slower; neither this fixed-cache 64-output control nor the partial exact-target
arm passes the eight-turn dynamic production speed gate. The 5500→5400 cache
cost is still unmeasured. No production speed GO.

## Current dynamic verify trace

One fresh native-head dynamic-mask run: balanced, dual sources, 5500 slots,
k2/top4, CPU routing, ONECB, 64 outputs/25 cycles, zero thermal pauses or load
failures. Draft/verify/commit/CPU means: 22.181076 / 201.262970 / .838856 /
.125570 ms/cycle. Trace covers all 25 target forwards and contains no draft
GPU regions.

| Target trace | ms/cycle |
|---|---:|
| Attention | 45.667896 |
| CED | 1.506775 |
| MoE | 61.261392 |
| Engram GPU | 4.727241 |
| Tail | 10.900829 |
| Cross-submit gaps | 74.875007 |
| Same-submit gaps | 1.588214 |
| Target span | 200.527353 |

Engram host issue/landing averages 62.179328 ms, overlapping the cross-submit
waits. Expert blocking IO wait is zero. The host `record_layers_tail`,
`fence_wait` and `engram_issue_land` overlap: adding them invents extra cost.
The 814-cycle eight-turn host report independently puts verify at 200.533607
of 228.191530 accounted ms/cycle (87.88%). Even free draft would only give
about 1.13× at unchanged verify/acceptance, not 1.6×.

Engram is already issued early. Enabling the existing P2 bypass is not a new
safe optimisation: its cold dynamic-mask quality regression is recorded in
`mask_quality_plan.md`; leave the production policy unchanged.

## Column-width probe

The union runner retains six-column storage/specialisation for a three-row
batch. A new isolated probe holds reduction geometry, three live columns,
12 resident FP4 slots and a three-layer streaming cycle constant. One warmup
and one eight-iteration measurement per width; first-layer live outputs match
bit for bit. GPU per-layer 1.875508489→1.796428236 ms. Extrapolated across 40
layers: 3.163210 ms, halved **1.581605 ms < 2 ms**. Stop before changing runtime.

This resident micro contains no expert IO and no shared FP8 expert. It changes
array width with the existing dynamic loop; constant-loop variants are not
measured. It does not close every future MoE optimisation. Actual bounded
prefill workspaces also remain open: current `Prefill::plan` still sizes them
from the full prompt; splitting calls without preserving absolute position,
KV, CED/compressor and Engram state is not a correct implementation.

## Evidence and validation

[Machine receipt](verify_control_receipt.json) includes source hashes and
explicit measurement limits. Raw data stays under
`bench/results/mask_quality/compare_control_{exact_target,frozen_cache,prepared}/`,
`compare_verify_trace{,_prepared}/`, `verify_width_probe_prepared/`.
Actual main build, CPU gates 31/31 and tools 57/57 pass; new control checks
9/9 and nested-accounting checks 4/4 pass. Actual main GPU off NLL remains
.622784, with zero thermal pauses; 53 shader files match the tested worktree.
The implementation was pushed and its remote SHA verified. Balanced dynamic
mask web is restored with k2/top4/native BF16, 5500 slots and two sources;
config/status both return HTTP200. The browser was not refreshed and six user
files retain SHA/size/mtime. Tested binaries/shaders are retained in the raw
validation directory. Owned worktree and branch removed.
[Delivery receipt](verify_control_delivery_receipt.json).
