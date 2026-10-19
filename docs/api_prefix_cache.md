# API KV prefix reuse (Round 4, 2026-10-19)

API requests now retain the existing engine session KV instead of using a random
session and resetting it after every response. The default bucket hashes endpoint
and **leading** normalized system/tools messages. `prompt_cache_key` or
`cache_session` optionally isolates a conversation. Later system updates stay in
the history, outside the bucket identity; the engine validates token equality
before every reuse. API transcripts are neither loaded nor saved. Engine pool
limits (eight parked sessions), eviction and disk KV remain authoritative.

Claude Code's request-varying `x-anthropic-billing-header:` telemetry line is
removed from the top-level Anthropic system field. Actual instructions and
history are retained. This changes the harness token counts and can change its
answers; output-token counts are not assumed equal. No tokenizer/model/kernel,
engine executable or numerical implementation changed.

## Counters and surfaces

`CacheMetrics.record(done)` consumes engine `prompt_tokens`, `reused_tokens`,
`prefill_tokens`, and `decode_hit_rate` once per completed turn. Three surfaces
consume its snapshot, including when the engine is busy:

- `/api/status`: `cache.last_turn.{prompt_tokens,reused_tokens,prefill_tokens,
  prefix_hit_rate,expert_decode_hit_rate}` and
  `cache.cumulative.{turns,prompt_tokens,reused_tokens,prefix_hit_rate}`.
- Every successful API result: the same `cache` object. Chat SSE includes it
  on the terminal chunk, Anthropic on `message_delta`, Responses on the completed
  response. Non-stream results include it at top level.
- UI beside Speed: `#s_prefix` labelled **KV prefix**, `#s_hit` labelled
  **Expert decode**. `renderCache()` handles status polling and web done events.

**Prefix hit rate = reused_tokens / prompt_tokens** (fraction, UI percent).
Cumulative is sum(reused)/sum(prompt), across web and API completed turns,
including generated cancelled turns. It resets on server restart; KV disk data
persists independently. Before the first completion `last_turn` is null. Busy
status and the UI show the last completed turn, not a prediction for the active
request. Reused tokens can still require sliding-window replay: the rate is not
an estimate of time saved.

**Expert decode hit rate = decode expert-cache hits / decode expert-cache
requests**, from `GenerateStats::decode_hit_rate()`. This is a separate expert
lookup rate, not KV reuse; no combined hit-rate label or cumulative expert rate.

Standard API usage also exposes reused tokens: Chat
`usage.prompt_tokens_details.cached_tokens`, Responses
`usage.input_tokens_details.cached_tokens`, Anthropic
`usage.cache_read_input_tokens`. Anthropic `input_tokens` excludes cache reads;
add it to `cache_read_input_tokens` for full prompt size. Creation tokens are 0.
Codex's real tool run reports `cached_input_tokens: 6268`.

## Measured result

Five serial real flora Claude 2.1.302 invocations before and after, same Round 3
prompts, tools disabled, JSON output and no session persistence:

| Phase/prompt | Prefix reuse | Prefill tokens/path | Client ttft_ms | Total s | First content ms | Wire decode tok/s | Engine decode tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| before/code | 0.00% (0/1499) | 1499/gpu | 32518 | 32.540 | 22402 | 12.796 | 12.766 |
| before/math | 0.00% (0/1505) | 1505/gpu | 23183 | 23.202 | 22765 | n/a (one frame) | 2.525 |
| before/zh | 0.00% (0/1490) | 1490/gpu | 34393 | 34.416 | 22803 | 9.056 | 9.111 |
| before/repeat1 | 0.00% (0/1499) | 1499/gpu | 36866 | 36.888 | 22620 | 8.992 | 9.048 |
| before/repeat2 | 0.00% (0/1499) | 1499/gpu | 38607 | 38.629 | 23065 | 8.273 | 8.314 |
| after12/code | 0.00% (0/1470) | 1470/gpu | 53457 | 53.480 | 41360 | 9.049 | 9.160 |
| after12/math | 0.00% (0/1478) | 1478/gpu | 34481 | 34.504 | 31664 | 9.057 | 9.261 |
| after12/zh | 0.00% (0/1463) | 1463/gpu | 42748 | 42.773 | 31718 | 8.681 | 8.774 |
| after12/repeat1 | 0.00% (0/1470) | 1470/gpu | 38354 | 38.377 | 25771 | 8.733 | 8.792 |
| after12/repeat2 | 99.86% (1468/1470) | 2/decode | 32785 | 32.806 | 18435 | 8.026 | 8.117 |

Client `ttft_ms` is this Claude release's complete-assistant metric, as established
in Round 3. First-content and wire rates are separate actual SSE observations.
Wire decode is `(nonempty frames - 1)/(last-first)`; arithmetic before has one
frame. Engine tok/s uses its own decode clock. Removing telemetry changes prompt
and output counts, so this is not a bit-identical generation speed comparison.

The independent repeat's reuse rises **0 -> 99.86%**, prefill **1499 -> 2**,
first-content **23.065 -> 18.435 s**, total **38.629 -> 32.806 s**. It still pays
**18.128 s replay**; this target has a measured floor, not a large TTFT win.

A genuine continuation keeps the assistant history and repeats the Python
question: **1586/1618 = 98.02%**, 32 new tokens, **zero replay**, HTTP client
first text **4.269 s**, total **13.398 s**, wire decode **12.744 tok/s**.
This supplementary HTTP client uses full Claude request/history; it is not a
Claude JSON ttft measurement and has no matched before continuation run.

Real Claude Read: first/second prompts 2195/2284, second reuses **2241 (98.12%)**,
prefills 43, zero replay, engine TTFT **4.303 s**, decode **10.327 tok/s**.
The actual file proof reaches the second request and final answer. Real Codex
executes a shell file write and finishes; second request reuses **6268/6323**.

## Configuration decisions and unmet targets

Only launch-flag change: `--gpu-prefill-speedup 12` (was implicit 6). Other
Round 3 settings remain: 5000 slots, prefill min16, 1M context, 8GB disk KV,
dual source, mask/dynamic, k2/top4/ONECB, CPU routing, native BF16 draft head.

The first retention-only attempt still had zero reuse because the billing line
changed near the beginning; it was stopped once the cause was proved. With that
line removed but speedup6, arithmetic kept 1242/1478 tokens yet cost **48.262 s**:
16.447 s window replay plus 236 decode-prefill tokens. This is rejected. Speedup12
lets the existing policy choose GPU recompute for large divergent suffixes;
identical-prefix and normal tool continuations retain reuse. No new cache or
shortened replay window was introduced.

**Broad latency/decode non-regression is not met.** Independent code/math/Chinese
requests remain cold or choose GPU recompute; their totals are slower in this
run. First after code includes **19.260 s disk-session restoration** before a
recompute. Arithmetic GPU prefill costs **31.5 s**, including **25.599 s expert
IO**. The GPU suffix path still only starts at position zero. Shared leading
system alone cannot cheaply reuse an arbitrary divergent history. The current
policy can discard that prefix; its existing heuristic also omits replay cost.

Wire code decode is **12.796 -> 9.049**, Chinese **9.056 -> 8.681**; no general
throughput win is claimed. Before repeat2 was already **8.273**, after **8.026**
(about -3%). Before/after repeated-turn counters also differ: expert decode hit
85.49% -> 83.78%, Engram host 16.465 -> 37.347 ms/token; these describe different
output/cache states, not a proved causal regression. The continuation's 12.744
shows the engine is capable of Round 3 throughput, but does not erase those
slower independent results. No repeats were used to choose a favorable timing.

## Verification and raw evidence

58/58 tools gates (eight API unit cases), 31/31 CPU CTests, diff whitespace check.
GPU prefill remains active: exactly 2000-token web fixture **9.329 s**; real
Codex 6210-token prompt also uses GPU. Anthropic text SSE remains incremental
before completion; blocking-queue regression passes. OpenAI/Anthropic non-stream
and SSE schema checks, real web Hello, real Codex shell write and Claude Read
all pass. Responses tool text remains parsed/buffered as in Round 3; no new
Responses text-streaming claim. Executable SHA256 is unchanged from Round 3:
`6cd2aa2823fa8e01a15379e315a9911cca29b8bcf1d79e4b3c74f5558b602e85`.
No new numerical-quality validation is claimed for this adapter-only change.

Machine evidence stays outside git at `~/cmoeapi/report4.md` and
`~/cmoeapi/round4/`: matched request/audit pairs, original client JSON and SSE
arrival times, `measurements.json`, `continue-*`, `tool-final/`, `status-after.json`
(the 99.86% sample), raw curl `status-final.json`, actual headless browser DOM
`rendered-ui.html` and `ui-proof.txt`, gates and acceptance logs. Final tool-loop
bucket fix preserves the standalone benchmark routing equivalence and is tested
by the actual Claude tool loop. `publication.json` records the pushed HEAD and
live service identity after commit, without a self-referential source receipt.
