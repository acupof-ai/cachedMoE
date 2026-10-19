# Claude Code client latency (Round 3, 2026-10-19)

Real flora Claude Code 2.1.302 measurements confirm GPU prefill and incremental
Anthropic text SSE, but **no latency or decode speed improvement**. The adapter
now opens its text block with the first nonempty token, rather than before
prefill, so the client's `first_content_frame_ms` measures actual content. Audit
responses record first/last flushed text and completion times. Empty/EOS-only
text deltas are omitted; message framing and tool parsing remain unchanged.

The owner's original three JSON results and the after results use identical
prompts, tools disabled, JSON output and no session persistence:

| Prompt | Input/output tokens | Prefill path before/after | Prefill s before/after | JSON ttft_ms before/after | Total ms before/after | After first_content_frame_ms |
|---|---:|---|---:|---:|---:|---:|
| Python trial-division function | 1499/130 | gpu/gpu | 22.148/24.376 | 33070/35457 | 33092/35477 | 24451 |
| Train arithmetic | 1505/2 | gpu/gpu | 22.620/22.472 | 23185/23044 | 23204/23063 | 22597 |
| Chinese expert-cache explanation | 1490/106 | gpu/gpu | 22.586/22.616 | 34477/34734 | 34501/34757 | 22730 |

A flora loopback HTTP tap forwarded the same Claude requests to the LAN server
and recorded nonempty SSE arrival times. A supplementary Python before run
received 129 text frames from 21.994 to 32.072 seconds, response ending at 32.175;
after, 129 frames arrived from 24.428 to 35.340, ending at 35.434. Client-observed
inter-token rates `(frames-1)/(last-first)` are **12.702 → 11.731 tok/s**.
Chinese after is **8.731 tok/s** (105 frames); original Chinese wire timing is
unavailable. Arithmetic has one text frame, so no meaningful decode rate.
These are client-wire-derived rates, not native JSON decode-rate fields or
engine-only speed claims. No throughput gain is claimed.

`ttft_ms` in this Claude release timestamps a complete assistant message. Its
binary's assistant handler sets that timestamp; a separate stream timestamp
exists. Before, `first_content_frame_ms` was about 42–48 ms because it counted
an empty text-block start. After, real content arrives 11–12 seconds before the
long answers finish. JSON `ttft_ms` still approaches total duration: this target
is **not improved**, despite confirmed incremental SSE before and after.
The server preserves one text block rather than fragmenting messages to change
a client metric. JSON output itself produces a completed result; partial JSON
stream output is a different client mode, not used as the required benchmark.

Keep the existing launcher: **5000 slots**, prefill minimum **16**, context
**1048576**, 8 GB disk KV, dual source, dynamic mask, native BF16 draft head,
k2/top4/ONECB with CPU routing. Round 2 already reduced 5500 to 5000. The old
7.13/7.03 GiB workspace refusal concerned 6208 tokens at 5500 slots; none of
these three prompts fell back. No further slot change is justified by this
capacity evidence. Cold engine start to HTTP readiness is **21.586733 →
21.535373 s**, with OS page cache retained. A separate repetitive, exactly
2000-token web fixture uses GPU prefill in **10.299419 s**, proving capacity,
not harness performance. The actual three harness GPU prefills still cost
22.472–24.376 s, including 16.232–18.190 s of expert IO; this is the observed
floor for these requests, not a theoretical lower bound.

Acceptance: **58/58** tool gates, six API unit cases including a live blocking
queue incremental-flush regression, web Hello and 2000-token GPU chat,
OpenAI/Anthropic non-stream and SSE, and a real Codex shell file-write loop all
pass. A separate real Claude **Read-only** check passes in two turns / 52.910 s:
request two includes the actual file proof string as a tool result. All-tools
and general coding quality remain untested. Tool blocks still await parsing.
No binary, shader or numerical change; no new numerical-quality claim.

The complete report, exact launch flags, original and after client JSON,
request/SSE/response pairs, timestamp definitions, runner and checksums live
outside git at `~/cmoeapi/report3.md` and `~/cmoeapi/round3/`. The tap changes only
a per-invocation endpoint settings copy; all requests still reach the LAN
server. No global client configuration changes. The systemd service owns the
sole engine; all GPU work is serialized. Publication proof is
`~/cmoeapi/round3/publication.json`.
