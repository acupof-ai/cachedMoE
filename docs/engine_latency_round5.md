# Engine latency Round 5 (2026-10-19)

Historical round receipt. [Round 6](engine_latency_round6.md) subsequently found
repeat/continuation prime failures in the same retained implementation and
reproduced .622784 with the original checkpoint/attention settings. Its
checkpoint-provenance diagnosis supersedes the unexplained-drift interpretation
below; Round-5 measurements remain recorded as they occurred.

Exact repeated prompts now avoid rollback replay. Cold total ≤15 s and cold
code/Chinese decode ≥13 tok/s were **not achieved**. The historical numerical
baseline also fails on untouched current main; this is not full acceptance.

Source baseline: `203d3c47211d9ed3a384cc5f69b05b1caf353775`. Machine report:
`~/cmoeapi/report5.md`; raw receipts: `~/cmoeapi/round5/` (outside git), including
all 16 HTTP attempts, errors, rejected shortcuts, automatic retry, raw JSON,
SSE arrival times, engine logs, status and configuration snapshots.

## Measurements

flora Claude2.1.302, tools disabled, JSON/no-session-persistence, invocation-only
settings. One engine; default5000 slots/min16/speedup12, dynamic mask,
DSpark k2/top4, native checkpoint, two sources (mirror48/48). No launch flag,
sampling, quantization or shader arithmetic change. Current prompt counts are
1470/1478/1463 versus owner's Round4 1499/1505/1490, following existing billing
normalization; comparisons with supplied-before are descriptive, not a causal
matched-token A/B. Cold means zero KV reuse, not flushed expert/OS caches.

|Case|Before first /total s|After wire first /total s|Reuse|After engine /wire tok/s|
|---|---:|---:|---:|---:|
|cold code|22.4 /32.5|24.185 /33.491|0%|11.944 /11.846|
|cold arithmetic|22.8 /23.2|22.676 /25.338|0%|10.303 /9.410|
|cold Chinese|22.8 /34.5|22.960 /33.788|0%|8.861 /8.844|
|identical repeat|18.4 /32.8|0.057 /8.237|100%|12.978 /12.736|
|true continuation|4.27 /13.4|4.022 /14.364|98.009%|10.383 /10.366|

Claude's corresponding actual first-content frames are24.207/22.698/22.978/
0.079 seconds for code/math/Chinese/repeat. Its `ttft_ms` is full-message
completion, not first text. Final cold expert hits82.241–86.967%, repeat93.433%:
all exceed the81.78% floor derived from owner's83.78% minus2 percentage points.
Code/Chinese cold decode and continuation throughput did not improve.

## Attribution and scope

Prefill already reads each layer's selected expert union, not one expert copy
per token. New profile counters separate selected union from unused blind reads:

|Case|Union experts|Actual read GiB|Unused blind GiB|Expert IO ms|Expert GPU ms|
|---|---:|---:|---:|---:|---:|
|code|8952|165.45|8.64|18004.875|2544.798|
|arithmetic|8897|150.82|8.88|16325.432|2569.958|
|Chinese|9037|151.51|9.30|16489.476|2607.028|

Cold setup outside the prefill profile is about182/174/175 ms. No matched
before GPU/setup savings are established. At primary4.53+mirror3.73 GB/s,
actual bytes minus unused blind bytes imply20.38/18.45/18.49 s storage budgets.
These extrapolations are not universal hardware lower bounds; workload overlap
and probe rates differ. The observed kept-path prefill times are24.118/22.517/
22.731 s. Unique uncached expert demand through dual NVMe is the cold bottleneck.

**Read-ahead-off NO-GO:** code read165.45→156.81 GiB, expert IO18.005→18.760 s,
prefill24.118→23.752 s (~1.5%, below3% jitter). No repeated A/B. Default restored;
there is no demonstrated cold gain. Removing speculative reads lost overlap.

**Kept exact prompt boundary:** memory-only original window, compressor carry,
DSpark hidden means/positions/masks, token identity and first result. Greedy,
≤8192-token, exact prompt and current-history-prefix match only. Clear/unpack/
seed/reset/prefix mutation invalidate applicability; disk format unchanged.
Only the original first result is reused; remaining output is decoded normally.
Repeat replay18.128 s→0, restoration0.529 ms, snapshot capture about8.5 ms.
Replay GiB saved is unknown because old replay byte accounting was absent.
General divergent suffixes and disk-restored sessions retain normal replay.

**Rejected partial-ring shortcuts:** missing DSpark state caused an error after
`def`, then automatic cold non-stream retry; adding hidden state shortened
replay but repeat/continuation incorrectly rejected prime3. Both removed.

**Kept Engram row FIFO:**8192 immutable `(layer,row)` payloads (~2.1 MiB/stream),
original value/scale bytes, no numerical change. Final code/repeat/continuation/
math/Chinese hit counts1472/4800/5520/160/240 avoid at least5.75/18.75/21.56/
0.625/0.938 MiB of value-sector demand. Exact skipped aligned bytes and isolated
ms gain were not measured. Status exposes hit/read counts. Decode still spends
substantial time in verify/fence waits and Engram; those accounting buckets do
not establish a universal hardware throughput floor.

## Acceptance and numerical blocker

Final CPU31/31 and tools58/58 pass, run serially (initial concurrent CPU/tools
run raced a shared fixture and gave57/58). Real Codex shell-write passes with
6268 cached tokens on the next request; real flora Claude Read passes with
2241/2284 reused tokens. Web Hello, exactly2000-token GPU prefill (9873.093 ms),
OpenAI/Anthropic JSON and SSE pass. The Python answers pass an executed prime
oracle; arithmetic equals240 but still violates number-only formatting;
Chinese preserves the meaning/two sentences but uses generic HBM/PCIe wording,
unchanged from Round4 rather than literal Strix Halo topology.

New wrapped-ring GPU check passes: original ring bytes and7/7 forced
next-token/top1/top2 values bit-exact after restoration; mismatched prompt,
non-greedy request, clear and reset reject reuse.

**Historical numerical gates FAIL:** off NLL0.660034 rather than0.622784,
decode6/8 reference and7/8 own-prefill rather than8/8+8/8. GPU-prefill stage
comparison has85/113 failed checks. Untouched203d3c4 built in a separate cache
has the same failures/key values; final delivery has identical NLL, all64 token
IDs, top1 count53/64 and step count. See `nll-comparison.json` and
`baseline-comparison.json`. This is no observed regression against current main,
not historical-gate compliance; cause of drift is unproven. No gate weakened.
Legacy full kv_replay suite was bounded after >8 min: incomplete, not passed.
Reproduce/repair that historical baseline before calling further work fully
numerically accepted. Cold15 s, decode13 tok/s, general divergent replay and
strict answer-only formatting remain open.
