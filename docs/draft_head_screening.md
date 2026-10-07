# Draft-head result: keep native BF16 by default

**FP8 is NO-GO for the requested production change.** Its head kernel falls from
8.903326 to 3.539821 ms, but the same-engine eight-turn attempt fails final-ID
identity on output 7 of the first FP8 turn. The candidate is cancelled immediately;
its incomplete speed is not an eight-turn result. FP8 stays an explicit experiment,
and the web retains native BF16. Vocabulary subsets failed the offline gate.

## Measurements and gates

Balanced, dual RO checkpoint sources, CPU routing, ONECB, dynamic mask, k=2,
top-4 acceptance. Each speculative cycle uses one main-path target forward.

| Check | Result |
|---|---|
| Native 64-output capture / CPU top-1 reference | 25 cycles; 38/49 accepted; 100% agreement |
| FP8 fixed-trajectory acceptance estimate | 0 percentage-point loss |
| 16K / 32K / 64K subset acceptance loss | 16.327 / 8.163 / 4.082 points; all NO-GO |
| Halved FP8 predicted gain | 2.224093 ms/cycle; passes the 2 ms entry gate |
| Native / FP8 head M=2 micro | 8.903326 / 3.539821 ms |
| Native / FP8 effective read bandwidth | 148.689 / 187.137 GB/s |
| Encoded bytes / scales | CPU and runtime hashes identical |
| GPU-vs-CPU FP8 logits | max error 0.0000153; two row argmax IDs agree |
| Native-5500 / FP8-5400 greedy 64-output IDs | identical; both 38/49 accepted |
| Off NLL | 0.622784, unchanged |
| Decode | loaded 6/8, own-prefill 7/8: existing baseline, not strict 8/8 |
| DSpark | per-stage, full ONECB/mega and FP8 M=1..5 prefixes pass |
| CPU / tools | 31/31 / 56/56 |
| Eight-turn final-ID gate | fails; full candidate speed gate skipped |

Each micro configuration uses one warmup and one timestamped eight-call batch,
with no per-dispatch queries. Correctness captures use profiling; their wall
cost is not throughput. The older 8.4 ms estimate remains a separate receipt.

## Why the long gate is NO-GO

The native arm completes all eight original `long_turns.json` turns: 2178 outputs,
2170 decode steps, 814 cycles, acceptance **83.7338%**, hit **91.2558%**.
Raw decode cost is **85.991236 ms/token**; 18.744032 seconds of decode cooling
leave **77.353433 active ms/token**. This workload has sampled early stops and
is separate from the power comparison's fixed 8×512 outputs.

The FP8 arm emits the same first six IDs, then output 7 is **14643** versus
native **20968**. Cancellation leaves ten buffered outputs / three cycles;
no later turns run. Its 60.153859 raw ms/token is a partial diagnostic only.

This comparison does **not isolate quantization as the cause**. Both arms use
one engine and the same 5400-slot budget, but native starts with 384 pinned
residents and FP8 inherits a full warm cache. Target routes already differ at
326 of 720 positions in cycle 1. At the first changed output both respective
candidate tokens rank 0 in their target rows, so the target distributions also
differ. Dynamic mask residency and changed draft math are confounded. The
production combination fails the required identity gate; no universal FP8
quality claim or speed winner follows.

For the first two cycles only, both arms accept 4/4 and emit identical IDs:

| Common 100% acceptance | Native BF16 | FP8 |
|---|---:|---:|
| Draft host wall ms/cycle | 23.347569 | 17.664296 |
| Verify host wall ms/cycle | 240.296851 | 146.005231 |
| Accounted cycle ms | 264.649906 | 164.753605 |
| At three outputs/cycle, ms/token | 88.216635 | 54.917868 |

The 94.292 ms verify difference exceeds the 5.683 ms draft difference and cannot
be attributed to head conversion. These two cycles are a matched-acceptance
diagnostic, not the complete eight-turn ≥3% speed gate. A full acceptance-normalized
speed winner and the 5500→5400 long-run hit cost remain **not measured** after NO-GO.

## Memory and default isolation

The additional row-FP8 copy is **662,430,720 bytes** including float32 scales.
At 18,808,832 bytes/slot it is 35.219 slot equivalents, but the existing allocator
reserves a whole **100-slot slab** (1,880,883,200 bytes). Total slots fall from
5500 to 5400; after 384 MTP pins the main model has 5016. Reservation padding
is 1,218,452,480 bytes, not extra usable expert slots. Target BF16 stays resident.
The short whole-store hit rates, 85.388% / 85.400%, include draft accesses and
cannot establish the long-run cache cost.

`CACHEDMOE_DSPARK_HEAD_FP8=1` is opt-in, parsed once into RuntimeConfig. Without
speculation or with mega it is rejected before GPU initialization. The benchmark
head switch requires allocated resources, an empty default single-stream session
and a completed GPU fence. When FP8 is off, its new parameter slots and descriptors
are absent and all original 52 shaders stay byte-identical. No checkpoint is written.

An earlier candidate mega prefix test failed; original main and final candidate
ONECB/mega controls pass. The failure cause remains unproven and its log is kept.

## Evidence

Raw directories under `bench/results/mask_quality/`:
`draft_head_capture_cpu_route`, `draft_head_screen_current_cost`,
`draft_head_baseline_micro`, `draft_head_fp8_micro_arith`, `draft_head_fp8_capture`,
`draft_head_fp8_baseline_gates_r2`, `draft_head_fp8_golden_gates`,
`draft_head_fp8_dspark_controls`, `draft_head_fp8_default_layout_gates`,
`draft_head_fp8_e2e` and their prepared plans/thermal logs. Invalid preflights and
historical estimates were retained, not overwritten.
[Machine receipt](draft_head_screening_receipt.json) contains source hashes.
Final main build, CPU31/31, tools56/56, off NLL .622784 and ONECB pass.
The balanced guarded web is restored with native BF16, dynamic 5500-slot mask,
k2/top4, CPU routing, two sources, 1M and 4 GB disk KV. Six original files retain
hash/size/mtime; browser not refreshed. [Delivery receipt](draft_head_delivery_receipt.json).
