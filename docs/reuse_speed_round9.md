# Round 9 — pinned verification and exact speed boundary

2026-10-19, main. Independently executed at **f1892adbb09e23efd54109b119534f8319dd3045** before any change.
**Reuse verification PASS. Warm code/Chinese decode reaches 13 tok/s after restoring the already-selected balanced profile. Cold total and cold decode targets remain BLOCKED on the current native execution path. No runtime speed change is accepted.**

## Pinned verification

Fresh Python fixture, temperature 0, seed 42, max 256, named session:
A cold:91 IDs; B identical repeat:91; C true history extension:91.
A/B first divergence **none**, A/C **none** (zero-based indices if a mismatch had existed).
B reuse1470/1470=**100%**; C1560/1592=**97.989950%**.
A/B/C client totals **37.750157/8.088107/15.896911 s**.
These are fresh runs, not Round8 results. Fixture input is the captured real Claude code request, normalized through the pinned API adapter; its source SHA is saved privately. The historical environment text is fixture data.

- Build PASS; CPU CTest **31/31 PASS**.
- Tools **58/58 PASS**, checkpoint metadata gates actually run. Initial simultaneous CPU/tools invocation produced57/58 (suite.io). Both suites manipulate identical fixed scratch paths (`tests/test_io.cpp:117,477`); concurrent execution was a harness error. Serial rerun passes. Both attempts retained; no test threshold/source change.
- Production strict decode **6/8 and own-prefill7/8**, unchanged; **exit1**, six assertions. Own-prefill free-running is6/8. This is preservation of known failures, not an8/8 certification.
- Fresh acceptance **10/10 PASS**: real Codex shell-write and exact disk bytes, real Claude Read→tool_result→final (two requests), exact webHello, actual cold **2000/2000** prompt/prefill tokens on GPU, OpenAI/Anthropic non-stream and SSE schema checks, readiness/models.
- Actual Codex loop wall54.916987s; Claude Read loop47.432697s including SSH wrapper. These tool-enabled gates are separate from the tools-disabled speed clients.

Private evidence: `~/cmoeapi/round9/{build.txt,cpu.txt,tools.txt,tools-serial.txt,decode.txt,matrix.json,acceptance/}`. The acceptance harness was reviewed and reused, with new sessions, proof files, CLI processes and receipts. Its historical `round8` label in fresh UUID names does not make these old runs.

## Final-effect table

All numbers below are this round. Final configuration: exact off-plain, no DSpark,5500slots, native checkpoint1017, two sources,1M context, GPU min16/speedup12, balanced.
The Python continuation row is the final balanced consistency fixture. Real Claude second turn is a second independent CLI invocation repeating the exact code input; it is not relabelled as history continuation.

| Requested row | Number | Source / boundary |
|---|---|---|
| Service cold start → HTTP ready | **20.677s** final; pinned20.673s, balanced-client restart21.453s | client monotonic poll from systemctl start to /api/config, `start*.json`;0.1s polling |
| Cold code first content / total | **24.249/38.074s** | remote Claude tap wire; engine TTFT24.183s, total37.906s |
| Identical repeat reuse / first content | **100% / 0.075s** | engine1470/1470; remote Claude wire |
| True continuation reuse / first content | **97.98995% / 5.301s** | engine1560/1592; Python SSE wire, `balanced-consistency/C.json` |
| Decode code | cold **6.558/6.543**, repeat **13.592/13.609tok/s** | engine / remote client wire |
| Decode Chinese | session cold **7.862/7.867**, repeat **13.819/13.841tok/s** | engine / remote client wire; Chinese cold follows math, not a fresh engine |
| Expert decode hit | code cold**88.5880%**, Chinese cold**92.2735%**; both repeats**100%** | engine `decode_hit_rate` |
| Real Claude Code cold / second wall | **38.804/7.576s** | actual remote CLI process wall against192.168.31.111:8080; includes CLI overhead |

Wire throughput counts visible token deltas: `(delta_count-1)/(last_text-first_text)`. EOS is not a text delta. This is explicitly not output_tokens/client_total; original SSE timestamps and engine counters are retained. Code has90 text deltas/91 engine outputs; Chinese103/104. Engine includes its terminal decode step. No first-content time is taken from Claude's aggregate `ttft_ms`.

Each real speed invocation uses per-invocation settings, `--setting-sources ''`, `--tools ''`, `--output-format json --no-session-persistence`, and the three owner prompts verbatim. A remote loopback tap forwards to the exact LAN URL and adds only a stable `cache_session` to keep repeat sessions identifiable. The tap retains bodies/frames without auth headers; credentials stay in process memory. No global Claude settings are changed. Five CLI invocations total: code cold, code second, math, Chinese, Chinese second. Math returns the correct240 with unwanted explanatory text; this instruction-following shortfall is retained, not a numerical acceptance claim.

## Attribution and blocker

The fresh cold code profile counts **177,649,418,240 bytes =165.448914GiB** expert IO:
**168,376,664,064 bytes =156.812988GiB** required routed union (8952experts), plus
**9,272,754,176 bytes =8.635925GiB** unused blind reads (493experts).
Runner23,998.859ms; complete engine prefill24,174.469ms. Buckets:
expert IO**17,695.732ms**, expert GPU2751.601, attention1265.308,
Engram IO1034.195/compute63.607, mHC229.405, shared157.853,
embed85.702, gate29.208, head6.638, host679.610ms.
Nested read-ahead waits overlap and are not added again.
`prefill_nvme_mb` in the done event is0 for GPU prefill: it is not a complete IO counter. The actual prefill byte source is `prefill-ops.jsonl`, not that misleading zero.

**No cold IO saving is kept:0GiB removed,0ms attributed to a retained cold optimisation.**
Expert IO remains NVMe→native transit/cache slots; no checkpoint bytes are changed, repacked, quantised or substituted by an answer cache.
Read-ahead-off is already closed (8.64GiB/1.5% wall in Round5), so it is not rerun for a preferred sample. Even eliminating all blind reads leaves156.813GiB.
Fresh startup source probes are4.51+3.74=**8.25GB/s** (decimal).
At this measured bandwidth envelope, required-union transfer alone is
**168.376664064/8.25=20.409s**, before post-prefill decode; even a warmed6.622s code decode gives an optimistic current-path total of**27.03s**, before other unhidden work.
For15s with that warmed decode, IO would have only8.378s, requiring**20.10GB/s** for the union alone.
This is an **empirical floor for the present volume/execution path**, not a universal physical lower bound or proof that every new algorithm is impossible. Faster kernels cannot remove those required bytes; crossing the cold target needs a new, validated native IO-volume/execution path or materially different storage/memory capacity. No such path is implemented or certified here. The cold target is therefore unfinished; current counters establish the storage-volume blocker rather than a claimed completed speed fix.

The only environment correction restores the documented balanced winner from the observed performance setting. It changes no arithmetic/cache policy. Measured warm code stages are attention35.237 +MoE23.877 +Engram1.532 +tail5.870ms/token; fence_wait68.662ms overlaps GPU work. Cold code still reads**43.180GiB** in decode and stalls81.321ms/token; Chinese reads**33.458GiB**, stalls55.951ms/token. Repeats read0GiB. This accounts for why warm13tok/s does not establish cold13tok/s. Different cache states/clients prevent claiming the whole observed speed difference is a controlled causal power-profile gain; the prior power comparison is not reopened.

Final balanced A/B/C also matches91/91 and91/91; reuse100%/97.989950%. Its A matches pinned A exactly. The session-cold A runs after other clients in an already warm engine; its shorter wall is **not selected as the cold-start speed baseline**. No faster wrong runtime candidate was kept.

## Every invocation, including slower rows

All model invocations made by this round are below, with raw receipts in the private root. CPU/tool initial failure and serial correction are separately preserved; no output-producing model attempt was dropped. There were no model retries or speed A/B repeats. Additional balanced consistency calls and the Chinese warm repeat have distinct correctness/cache-state purposes.

| Invocation | Wall s | Source |
|---|---:|---|
| pinned/performance/A | 37.750157 | `A.json` |
| pinned/performance/B | 8.088107 | `B.json` |
| pinned/performance/C | 15.896911 | `C.json` |
| balanced consistency/session cold; engine already warm/A | 34.321450 | `balanced-consistency/A.json` |
| balanced consistency/session cold; engine already warm/B | 6.863619 | `balanced-consistency/B.json` |
| balanced consistency/session cold; engine already warm/C | 15.861943 | `balanced-consistency/C.json` |
| real Claude/code-cold | 38.804355 | `api_a1bdc21898544916a9c71d08ac23613e.response.json` |
| real Claude/code-second | 7.575986 | `api_30c1772e27ad410da483974de7bd2bc7.response.json` |
| real Claude/math | 26.229473 | `api_77a9cd7ccff3416b87e36aee1bc80ee9.response.json` |
| real Claude/zh | 35.298401 | `api_f31ef1d18b054a5ebe6c50697c66a5bb.response.json` |
| real Claude/zh-second | 8.370250 | `api_22b6cd7af7fb4993bed972edae3d44d0.response.json` |
| acceptance/openai-json | 2.778887 | `acceptance/openai-json.http.json` |
| acceptance/openai-sse | 0.131787 | `acceptance/openai-sse.http.json` |
| acceptance/anthropic-json | 3.041308 | `acceptance/anthropic-json.http.json` |
| acceptance/anthropic-sse | 0.129206 | `acceptance/anthropic-sse.http.json` |
| acceptance/web-hello | 0.875204 | `acceptance/web-hello.http.json` |
| acceptance/web-capacity | 9.359277 | `acceptance/web-capacity.http.json` |
| real tool-loop/api_9c3d840b80a67ee4b5d30db16d4a96db8bdc0c37 | 40.899632 | `acceptance-audits/api_64e9e92dcc6c403ebc965346e28ea401.response.json` |
| real tool-loop/api_9c3d840b80a67ee4b5d30db16d4a96db8bdc0c37 | 12.660254 | `acceptance-audits/api_bac27caff1e344a290fff6ed6ed6a642.response.json` |
| real tool-loop/api_827d953de67b250ccde5e32d5f979374197db3b5 | 34.101424 | `acceptance-audits/api_21423a7520d448719a5af2c376f55822.response.json` |
| real tool-loop/api_827d953de67b250ccde5e32d5f979374197db3b5 | 11.406640 | `acceptance-audits/api_e92e6e4936484d1dac7883518297f90c.response.json` |
| final liveness Hello | 4.714137 | `final-hello.json` |

Private `invocations.json` and `client-summary.json` retain engine/wire figures for each row. Acceptance functional rows are not speed baselines; tool-loop row wall is engine total for each model request. Complete CLI loop walls are reported above. Startup attempts are three successful serial boots,20.673/21.453/20.677s; no model invocation occurred during startup polling.

## Delivery

Only aggregate documentation changes are committed. Runtime, tests, fixture thresholds, checkpoint and API policy are unchanged. Machine prompts, raw IDs, transcripts, settings and profiles stay outside git. The existing systemd user unit remains enabled and active, balanced, one engine listening on8080. Temporary profiling is removed from the manager and a fresh final service is started without it; LAN health/status and exactHello are verified. Final HEAD/remote equality and binary/service identities are recorded in the private report after push.
