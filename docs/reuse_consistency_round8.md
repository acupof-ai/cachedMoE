# Reuse self-consistency — Round 8

2026-10-19, Linux RADV, production revision1017, two read sources. This round
fixes a demonstrated KV ownership defect and changes the LAN service to exact
unmasked/plain decode. **Self-consistency passes; the ≤15 s cold and ≥13 tok/s
decode targets do not.** Historical oracle certification is not claimed.

## 1. Reproduction and first differing state

The Python fixture uses a named session, temperature0, seed42, max256 output.
A is cold, B repeats the identical input, C appends the previous assistant
answer and asks the same user question again. Raw output token IDs and complete
SSE are saved for every invocation, not selected by output or speed.

Before the fix, A has111 IDs; B106 and C108. A/B first diverge at **index3**
(zero-based); A/C also at3. B reuses1470/1470 prompt tokens. All Python functions
in this round pass the prime checks, including3; the earlier Round6 semantic
failure does not recur, but raw-token inconsistency does. Semantic plausibility
is not a substitute for the token check.

Instrumentation uses the existing `CACHEDMOE_SPEC_DIAGNOSTICS` JSONL sink:
retained/before-restore/after-restore ring bytes, DSpark source-hidden bytes,
positions/capture masks, first two cycles' raw route weight bits/cache hits,
and proposal target ranks/argmax. It compares native bytes, without packing or
re-quantising them. Normal service has this sink disabled.

At retained boundary1470, the first required wrong row is **position1343,
layer0**. After generation,2,219,750 ring bytes differ in4,360 layer/position
rows. The old restore leaves **1,466,197 wrong bytes in2,880 rows**, all falsely
marked present by the host ownership map. DSpark source hidden state differs
by3,022,197 bytes/110 positions before restoration but has **zero byte,
position and capture-mask differences after restoration**. The surviving
first differing buffer is therefore the KV value/scale ring, not the restored
DSpark hidden window. These are byte mismatch counts, not floating-point norms.

Cause: a speculative `forward_batch` writes every verification row, but host
metadata previously tagged only the last row via `prepare_ced`.
`resolve_ring` repairs unknown ownership, not a known-but-stale tag.
`recover_prompt_ring` trusted `ring_holds(p)` and skipped saved bytes when the
stale tag claimed the prompt position was still present.

## 2. Cause-level fix and policy boundary

- Restore every required saved prompt row unconditionally. A saved boundary
  owns its bytes even when an old tag claims those bytes survived.
- Mark **all** successful batch writes with their actual positions.
- Resolve seeded unknown ownership before the pre-batch snapshot.
- Save ownership alongside ring bytes; restore rejected rows' saved ownership
  alongside their bytes, without undoing committed rows.
- A partial-layer restore invalidates store-wide ownership: it cannot claim a
  complete position while layers hold different generations.

With this fix, all three instrumented repeat restores have **zero ring bytes,
zero falsely-present bytes, zero DSpark hidden bytes/positions/masks differing**.
Yet the old mask/top4 policy still diverges at indices67,47 and28 in the
repeat1/repeat2/continuation rows (repeat3 happens to match). That configuration
is rejected for exact-default service, not presented as a completed fix.

A second, independent cause is approximate policy: after the ring fix, the first
raw layer0 route has identical expert24/weight bits1042341308 but cache hit
true in cold and false in repeat. Dynamic masking zeros different weights as
residency changes. There are397 hit differences among1,440 first-two-cycle
route records. Six accepted non-argmax proposals also occur in the fixed masked
matrix; relaxed top4 does not implement greedy exact sampling. These
observations do not uniquely attribute every later divergent token to one
mechanism, but they rule out cache-independent exact semantics for that policy.

The service now defaults to **off-plain**, no DSpark, no residency masking.
The approximate modes remain available explicitly; they are not relabelled
exact. Simply switching top4→top1 would not prove serial equivalence: historical
STATUS experiments already establish batch/serial numerical differences.

## 3. Exact proof and capacity candidate

At5000 slots, three repeats and C each match all91 cold IDs:
**364/364 compared tokens**, no first divergence. Each repeat reuses1470/1470
(**100%**); C reuses1560/1592 (**97.98995%**). At5500 slots the same four checks
again match91 each, **364/364**; cold IDs also match the5000-slot cold exactly.
Every generated Python response in the24-row matrix passes checks over
-10..1999 plus99991,999983 and1000000.

The retained result is only the prompt's first decode result and boundary
state, not a cached completed answer. Later output tokens are recomputed.

C/A equality is fixture-specific: C contains different input history. The
additional Chinese identical repeat matches104/104 IDs at100% reuse, but its
history continuation changes at index19 (91 IDs versus A104). This is recorded,
not hidden as a failed run. Fresh-session controls with **the identical complete
C input** match both reused continuations: Python91/91 and Chinese91/91,
zero reuse, GPU prefill,32.076 s and29.030 s respectively. This distinguishes
input-history effects from reuse errors; it is not a universal determinism
proof for all contexts or execution kernels.

## 4. Every matrix invocation

Client wall seconds include the complete request/stream. Divergence and equal
counts compare with that configuration's A. “none” means identical IDs/length;
a changed C context does not by itself establish a defect. Each configuration
is run once; the three requested repeats are one correctness matrix, not
repeated A/B speed selection. Instrumented timings are not speed baselines.

| Configuration / turn | IDs | First divergence | Equal positions | Client s | Engine tok/s | Reused / prompt |
|---|---:|---:|---:|---:|---:|---:|
| old / A |111|—|—|31.890|12.606|0/1470|
| old / B |106|3|3|6.821|15.682|1470/1470|
| old / C |108|3|5|12.664|12.470|1575/1607|
| old exact-policy control / A |91|—|—|36.582|6.261|0/1470|
| old exact-policy control / B |91|none|91|7.717|11.866|1470/1470|
| old exact-policy control / C |91|none|91|19.077|6.991|1560/1592|
| instrumented old / A |111|—|—|33.455|11.771|0/1470|
| instrumented old / B |106|3|3|8.825|12.311|1470/1470|
| instrumented old / C |108|3|5|14.849|10.002|1575/1607|
| ring fix, mask/top4 / A |111|—|—|33.485|11.717|0/1470|
| ring fix, mask/top4 / B1 |117|67|68|10.024|11.928|1470/1470|
| ring fix, mask/top4 / B2 |91|47|47|8.685|10.719|1470/1470|
| ring fix, mask/top4 / B3 |111|none|111|10.814|10.443|1470/1470|
| ring fix, mask/top4 / C |63|28|29|11.561|8.861|1580/1612|
| exact5000 / A |91|—|—|37.866|6.482|0/1470|
| exact5000 / B1 |91|none|91|8.637|10.567|1470/1470|
| exact5000 / B2 |91|none|91|8.929|10.218|1470/1470|
| exact5000 / B3 |91|none|91|9.248|9.865|1470/1470|
| exact5000 / C |91|none|91|17.627|7.573|1560/1592|
| exact5500 / A |91|—|—|37.605|6.711|0/1470|
| exact5500 / B1 |91|none|91|8.743|10.450|1470/1470|
| exact5500 / B2 |91|none|91|9.087|10.040|1470/1470|
| exact5500 / B3 |91|none|91|9.498|9.600|1470/1470|
| exact5500 / C |91|none|91|16.143|8.322|1560/1592|
| exact5500 Chinese / A |104|—|—|34.902|7.679|0/1463|
| exact5500 Chinese / B |104|none|104|10.611|9.834|1463/1463|
| exact5500 Chinese / C |91|19|20|13.501|9.305|1566/1591|

In addition: two same-full-history cold controls above; one5500 capacity request
(2000 prompt tokens, GPU,0 reused,9.671 s); and three client acceptance
attempts, whose individual wire files/tool requests are retained. These are
functional requests, not omitted latency retries. The5000 `acceptance-final/`
batch passes10/10. After restarting without profiling, `acceptance-5500/`
started before the listener: eight early checks fail ConnectionRefused, Codex
executes the exact write and finishes but its gate fails for a reconnect error,
and Claude Read passes. The whole batch is **FAIL**, not relabelled as passing.
A bounded engine-readiness wait is added; `acceptance-5500-ready/` then passes
all10 gates, including both real clients, all API schemas and2000-token GPU
prefill. This replacement is a transport-harness correction, not selection of
a faster or preferred model answer. The early `ring-fixed/`
attempt failed with ConnectionRefused before model invocation; readiness
polling was added and the full matrix has its own `ring-fixed-ready/` directory.
A build-plus-test shell used fish `and` under zsh and exited127 after a successful
build: the test had not run, and was then actually run. One offline aggregation
command had a Python quoting SyntaxError, then was replaced by `ledger.py`;
it made no model request. No output-producing model invocation was discarded.

## 5. Speed resumed after correctness — bounded result, not target success

The only new speed candidate is5000→5500 exact expert slots, using memory freed
by removing DSpark resources. Keep it for the measured read reduction, not for
sub-percent cold wall variation. It does not change arithmetic or cold IDs.

`decode_nvme_mb` is decimal MB; GiB below is MB×1e6/2^30. Wire throughput is
(N−1)/(last-token-time−first-token-time), excluding prefill like engine decode.
It is not N/client-total and not a new independent benchmark run.

| Python phase |5000 decode GiB / ms|5500 decode GiB / ms|5500 engine / wire tok/s|
|---|---:|---:|---:|
| cold |46.560 /13885.080|43.180 /13410.568|6.711125 /6.711476|
| B1 |0 /8517.293|0 /8612.062|10.450459 /10.450443|
| B2 |0 /8807.661|0 /8964.311|10.039812 /10.039708|
| B3 |0 /9123.159|0 /9374.885|9.600118 /9.600148|
| C |31.513 /11884.387|23.263 /10814.338|8.322285 /8.322364|

Cold reads drop7.26%, C26.18%; C decode wall drops1070.049 ms. Repeats already
read0 GiB and show no gain (1.1–2.7% slower), within the documented jitter
floor; no repeat acceleration is claimed. Whole cold wall37.866→37.605 s is
not a credible speed win. Chinese engine/wire A7.679124/7.679140,
B9.833861/9.833879, C9.305417/9.305543 tok/s (rounded wire values are available
in the full ledger); the code table is the exact-target comparison.

The1470-token cold prefill profile reads **165.4489 GiB** across9,445 experts:
156.8130 GiB required union plus8.6359 GiB unused blind reads. Profiled prefill
wall23877.966 ms, request-level prefill24052.137 ms (different measurement
boundaries). The bucket decomposition is:

| Prefill bucket | ms |
|---|---:|
| expert IO |17723.018|
| routed expert GPU |2588.453|
| attention |1272.307|
| Engram IO / compute |1016.530 /59.409|
| mHC / shared expert |232.515 /151.605|
| embed / gate / head |99.848 /38.325 /6.472|
| host |689.484|

Nested `io:experts ahead`3640.700 ms overlaps work; do not add it to total.
The5000 cold receipt also reads165.45 GiB: added capacity has not reduced cold
prefill IO. Already-tried read-ahead-off saved8.64 GiB but only1.5% wall and
was reverted in Round5; it is not rerun to seek a favourable sample.

Cold total37.605 s remains **22.605 s above15 s**. Even the measured expert-IO
bucket17.723 s exceeds that complete-request target, before decode. B1's
95.690 ms/token would need to fall below76.923 ms/token for13 tok/s despite
zero decode disk reads. Its per-token attention49.464 ms, MoE GPU29.286 ms,
Engram2.187 ms and tail6.432 ms identify GPU work, not HTTP buffering, as the
warm limit. Fence-wait91.824 ms is overlapping host wait, not an additive
second GPU cost. The bottlenecks require a new IO-volume/execution-path change,
not selecting a lossy mask/top4 result that happened to exceed13. No further
closed cache-policy/read-ahead/top1 experiments were reopened.

## 6. Gates and scope

| Gate | Result | Evidence under `~/cmoeapi/round8/` |
|---|---|---|
| Final build |PASS|`review-build.txt`|
| CPU CTest |31/31 PASS|`cpu-final.txt`|
| Tools |58/58 PASS|`tools-final.txt`|
| Vulkan KV regressions |7/7 PASS,0 assertions|`kvstore-gpu-final.txt`|
| Real wrapped batch/session policy |1/1 PASS,0 assertions|`batch-integration.txt`|
| Retained prompt bitwise regression |PASS, seven forced steps and bytes|`retained-current.txt`|
| Production strict decode counts |6/8 and7/8 UNCHANGED; suite exits1, six assertions|`decode-current.txt`, `gpu-commands.json`|
| Historical original0928/CM off64 |NLL0.622784,56/64; PASS historical reproduction only|`nll-original.json`|
| Python exact proof |3 repeats+C:364/364 at5000,364/364 at5500|`exact-final/`, `cache5500/`|
| Same-full-history controls |Python91/91, Chinese91/91|`continuation-controls/`|
| Exact2000 GPU capacity |PASS at5500,0 reuse|`cache5500-capacity.json`|
| Real Codex shell-write |PASS at5000 and final5500|`acceptance-final/`, `acceptance-5500-ready/`|
| Real Claude Read loop |PASS at5000 and final5500|same|
| Web exactHello |PASS at5000 and final5500|same|
| OpenAI and Anthropic JSON+SSE |PASS at5000 and final5500|same|
| ≤15 s cold / ≥13 engine+wire |NOT MET|all matrices and attribution above|

Strict decode is not called a passing suite: its child exit1 is preserved even
though the evidence-collection wrapper exits0. Production1017 has no trusted
matching historical exports. The original0928/CM NLL reproduction does not
certify1017, and no thresholds/fixtures/baselines were changed. New-executable
GPU validation includes the actual batch path, retained bytes, live service and
real client calls; no checkpoint write, repack or re-quantisation occurred.

New tiny tests cover stale tags, committed versus rejected slot ownership,
partial-layer restore and truncated short prompts. The real engine integration
crosses the128-position wrap and checks all live ownership tags and restored
prompt bytes after a real speculative batch.

## 7. Final service and evidence

`tools/web/run-web.sh` is authoritative: off-plain,5500slots,1M context,
GPU min16/speedup12, production model and read-only mirror, native BF16,
`build/web_exact/kv`,8GB disk-KV cap. Legacy `build/web_mask/kv` is preserved;
there is no user-state migration/merge/delete. `--mask-cache dynamic` is inert
while residency policy is off. Thermal guards and audit paths remain unchanged.
Temporary spec diagnostics/prefill profiling are removed from the service
manager before final acceptance.

Stop before any separate GPU job:

```bash
systemctl --user stop cachedmoe-lan.service
pgrep -af 'cachedmoe serve'   # verify it is gone before starting a GPU test
# after the single GPU job exits:
systemctl --user start cachedmoe-lan.service
```

Machine-only root: `~/cmoeapi/round8/`. Each matrix turn saves `.request.json`,
`.sse`, `.json` with raw IDs/timed frames, and `matrix.json`. Offline scripts:
`reproduce.py`, `analyze_diagnostics.py`, `summarize.py`, `ledger.py`;
`all-matrix-invocations.json` adds every wire rate/byte/time measurement.
`acceptance.py` saves API wire/schema checks and real CLI/tool-loop receipts;
credentials remain out of receipts and no global client settings were edited.
These machine artifacts, prompts and transcripts are not added to git.
Final commit/remote/service identity is recorded in `~/cmoeapi/report8.md` and
its private delivery receipt, avoiding a self-referential committed hash.
