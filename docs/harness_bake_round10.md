# Round 10 — bake exact Claude prompt KV once, restore in new sessions

2026-10-19–20, Linux `main`, RADV, exact off-plain, no DSpark, 5500 slots,
two verified 48/48 sources, balanced profile. No checkpoint write, repack,
re-quantisation, sudo, second engine or numerical-threshold change.

**Same captured prompt, new session after service restart: first content
23.952876 → 0.056063 s; total
37.678831 → 18.602763 s. PASS ≤1.5 s. Cold/restored outputs
match all 91 token IDs, zero divergence.** The other three clients miss this
bake; their actual harness differs at rendered line 11, well before cwd/shell.
This is a full, token-exact rendered-request boundary cache. A different user
message or different harness does not automatically match it; this report does
not claim arbitrary-prefix extension or universal sub-1.5-second cold text.

## What changed and what is actually stored

The old `.pkv` retains exact non-SWA rows (FP4 grid rows plus raw BF16 exceptions),
carry and token IDs, but omits the sliding-window ring. Restoring only that file
still replays the window and reads expert weights. Round8 fixed the in-memory
ring restore; that memory-only fast path did not survive process restart.

New `boundaries/<identity>-<token-hash>.bkv` files persist the exact prompt's raw
BF16 compressed/index rows, compressor carry, complete ring **bytes and ownership
tags**, and first-step token/sampling metadata. Subsequent output still runs the
real decoder; no completed answer is cached. The saved first result is the same
prompt-boundary first-step result used by the existing memory fast path.

The file has a versioned magic, bounded lengths, exact-token validation, shape
checks and a payload checksum. Identity covers the engine relink, checkpoint and
shader file metadata, and captured runtime options; receipt-only sinks are
excluded. Invalid/unsupported state falls back cold. Eligible requests are
single-stream, greedy, off-plain, no DSpark, 512–8192 tokens, with disk KV enabled.
The default API session hashing/normalisation is unchanged. A full hit also skips
legacy named-session replay before generation. Existing `.pkv` readers remain
compatible; both formats share the existing disk-byte quota. No transcript is
migrated or removed. CPU integrity cases cover exact fields, stale identity,
bit damage and truncation. Relinking invalidates old bakes by design.

## Real capture and method

Four real Claude CLI processes ran on flora, x, mac and cachyos. A local tap
captured their actual `/v1/messages?beta=true` bodies; no system prompt was
hand-written. The CLI user input was the same trial-division `is_prime` task.
Per-invocation `--setting-sources '' --tools '' --no-session-persistence`
reproduces the tools-disabled ~1500-token harness used in prior speed rounds;
the tool-enabled harnesses are tested separately by real CLI regressions.

The capture-only tap returned a placeholder JSON response and invoked no model.
Each CLI made two requests: original SSE (`request-0.json`) and JSON fallback
(`request-1.json`). Their normalised prompt tokens are identical. Capture command,
CLI exit0, wall, cwd, shell, hostname and both original bodies are retained in
`round10/{flora,x,mac,cachyos}-capture/`. No auth headers or credentials are saved.

Cold/ID checks replay the captured body through the existing API normaliser and
renderer via `/api/chat`, temperature0/seed42/max256. The output ends naturally
at91 IDs. The four-client wire table uses each captured **original SSE body**
through `/v1/messages`; only `cache_session` is added to force a distinct named
conversation. Original output limits are retained and produce the same completed
code. Timings are client monotonic first non-empty content and full response;
engine TTFT is identified separately. All calls are serial against one engine.

Final cold and restored boots take 20.771 and
20.874 s to ready.
**The sub-1.5s result starts with a ready service; it does not include startup.**

## Bake, park and restart restore

| State / invocation | Tokens in state | File bytes | Save/pack ms | Restore ms | First content s | Total s |
|---|---:|---:|---:|---:|---:|---:|
| Cold flora + exact full bake |1470|7546352 `.bkv`|36.426|—|23.952876|37.678831|
| Identical memory repeat |1470 prompt|same|no rewrite|0.205734|0.039828|6.896898|
| Restart, genuinely new session |1470|same|—|10.677461|0.056063|18.602763|
| Identical repeat after restore |1470 prompt|same|no rewrite|0.214420|0.053496|6.835314|
| Prompt-only legacy park, max1 size probe |1470|1339132 `.pkv`|59.09|8.011|0.125647|0.213847|
| Completed legacy park after cold reply |1560|1419592 `.pkv`|73.42|not the exact fast path|—|—|

Save means snapshot+serialization+flush/close+atomic rename, rather than a physical
power-loss/fsync latency guarantee. Legacy pack is CPU packing before background
write. The prompt-only size probe restores an already baked boundary and emits
one token without feeding it, leaving exactly1470 prompt positions for packing;
it is a size-accounting invocation, not a selected speed baseline.

Final cold, both full repeats and disk restore all have91 IDs; every compared
array is identical, including the initial candidate's cold array. No window
replay occurs on a full hit. Engine cold TTFT=23914.869ms,
restored=19.022ms; wire first content includes adapter scheduling.
The first restart leaves the expert cache cold: decode hit drops
88.587963%→79.972222%, decode
13.627→18.451s. KV restoration
improves TTFT enormously; it does not silently prewarm expert weights.

## Four clients: bake flora once, measure the other real prompts

| Client | Reused/prompt | Hit | Exact boundary bytes | Bake ms | Restore ms | Wire first s | Wire total s |
|---|---:|---:|---:|---:|---:|---:|---:|
|flora|1470/1470|100.00%|7546352|36.426|7.492388|0.108062|6.904540|
|x|0/1660|0.00%|8155112|39.289|miss|22.315975|35.198069|
|mac|0/1651|0.00%|8124356|39.296|miss|21.844421|35.468751|
|cachyos|0/1667|0.00%|8175620|37.992|miss|22.333277|35.645492|

Other-client files in this table are created **after their misses**, not present
before their measured request. Their own bakes cannot make a different client's
body match flora. Wire timing is measured on x against127.0.0.1:8080 with the captured real
client bodies. It excludes remote network/CLI overhead; these are not four
complete interactive CLI wall-time benchmarks.

All three misses share exactly221 leading tokens with flora. First difference:
**rendered line11, `Text inside <pasted_content> tags ...`**: absent on flora,
added on x/cachyos and expanded on mac. Memory-root paths differ later, followed
by cwd, shell, platform and environment lines; x also supplies git context.
Private `prefix-comparison.json`, `*.rendered.txt`, `*.tokens.json` and `*.diff`
retain the exact line evidence. Thus these actual captures differ by CLI harness
content as well as environment, not merely a late cwd line. A221-token overlap
cannot justify treating the remaining~1400 tokens as a hit. No instructions are
stripped or reordered to manufacture cache compatibility. Exact misses remain
cold GPU prefill; no general partial-boundary restore is claimed.

## Verify the arithmetic and bound the win

A pre-existing1566-position `.pkv` is **1,424,956 bytes**, read directly from
its header and filesystem size (`packed-size-check.json`). That is
**909.934 bytes/token = 0.909934 decimal KB/token**,
rather than an exact0.95KB/token. The new1470-position prompt-only park is
**1,339,132 bytes**;1500 tokens are indeed roughly1.4MB. Headers/carry and raw
exceptions mean the cost is not a perfectly constant per-token slope.

Fresh1470-token cold GPU prefill reads **177,649,418,240 bytes =
165.448914GiB** of expert weights, including
**156.812988GiB required routed union** and
**8.635925GiB unused blind reads**.
Expert IO bucket=17.581681s; complete engine prefill=
23.869770s. `prefill_nvme_mb=0` in the completion does
not count GPU-prefill reads; the byte source is the fresh per-op profile.

165.449GiB divided by1.425MB is **124,670×**; divided
by the actual complete7.546MB boundary it is still **23,541×**.
At8.3–9.2 decimal GB/s, the168.377GB required union takes18.30–20.29s;
all177.649GB takes19.31–21.40s. The measured~17.58s expert-IO bucket excludes
some overlapped read-ahead work, so it should not be equated to an unoverlapped
bytes/probe-bandwidth quotient. The rough151–167GiB/18s argument is correct in
scale; exact units and overlap matter. This final boot probes4.67+3.72GB/s
(as recorded in the retained engine log; use the actual log for individual boots).

**A full restored hit avoids all165.448914GiB of cold-prefill expert reads**,
of which156.812988GiB is the required union. It also avoids attention/Engram/
other prefill work; the timing improvement is not attributed exclusively to IO.
The full-request expert-IO saving is smaller: cold decode43.179626GiB versus
restored-cold decode75.778931GiB, so total expert reads avoided are
**132.849609GiB net**. This explains total37.679→18.603s rather than claiming
that restoration makes a91-token completion finish in0.056s. Warm repeats read
zero expert bytes in decode and retain their~0.1s first-content behavior.

## Arbitrary cold text: capacity, hit rate, bytes per token

One expert slot is exactly **18,808,832 bytes (18.808832MB)**.5500 slots =
**103,448,576,000 bytes =96.343994GiB**. The40×384 backbone routed experts
occupy**288,903,659,520 bytes =269.062500GiB** at this slot size. Thus “roughly270GB
of experts” means roughly270**GiB** (289 decimal GB), and the resident capacity
is35.8073% of expert count. DSpark experts are outside this off-plain capacity
comparison. All figures refer to original native weights, not re-quantisation.

The fresh cold code decode measures **88.587963% expert hits** and
**46,363.7709MB /90 steps =515.153010MB/token**. This agrees with
`240 routed selections/token ×18.808832MB ×(1−0.88587963)`.
After restart+KV restore,79.972222% hits imply904.077858MB/token; identical warm
repeats have100% hit and0MB/token. These are expert-cache hits, distinct from
`reused/prompt` KV hits. Arbitrary topics do not inherit the code fixture's hit
rate. Cold GPU prefill itself reads120.849944MB/prompt-token
(required union 114.541948MB/token); decode's hit-rate
formula must not be substituted for that union-prefill counter.

Capacity cost: another500 slots requires**8.758545GiB**,1000 requires
**17.517090GiB**. The earlier Linux Round8 matched5000→5500 result reduced this
code fixture's cold decode46.560→43.180GiB (**7.26%**), and continuation
31.513→23.263GiB (**26.18%**). Those are retained previous measurements, not
new RAM configurations run in this round. They establish real benefit from
capacity; they do not justify linear extrapolation past5500.

Sensitivity at240 selections:90% hit costs451.411968MB/token;95% costs
225.705984MB/token. Relative to the measured515.153MB/token that would save
63.741MB (12.37%) or289.447MB (56.19%) **if a larger resident set actually
achieves those hit rates**. This round has not measured which larger capacity
would achieve them on this1470-token fixture, nor converted those byte savings
into an unsupported end-to-end speed claim. The older Windows8-turn LRU curve
in `docs/p4_hitrate.md` gives5500:94.31%,6000:94.86%,6500:95.32% for its own
routing trace (256.853→232.026→211.261MB/token), with+8.759/+17.517GiB;
those are historical trace-specific simulations, not a fresh Linux prediction.

Full backbone expert residency needs**172.718506GiB more expert RAM** than now;
269.0625GiB expert RAM + current~9.17GiB pinned weights already exceeds
**278.23GiB**, before workspace, KV and OS. The current host reports121.49GiB
physical usable memory and onlyabout6–7GiB available with this engine; it cannot
fund even the next8.759GiB cache increment while preserving measured prefill
headroom. No cache-slot increase is recommended here. For unbakeable cold text,
reducing measured expert GiB/token (or changing storage/memory capability) is
the relevant lever; this patch does not speed a genuine miss by changing KV media.

## Regressions and delivery state

Final build PASS; **CPU31/31 PASS**, including the new boundary integrity case;
**tools58/58 PASS**, checkpoint metadata gates enabled. Production strict decode
is unchanged **6/8 teacher-forced,7/8 own-prefill**, own-prefill free-running6/8,
**suite exit1, six assertions**. Both candidate and final strict invocations
retain those failures. No threshold or fixture changed; this is not a claim
of strict8/8 or historicalNLL production certification.

Fresh client acceptance **10/10 PASS**: real Codex shell-write with exact disk
bytes, real Claude `Read`→tool_result→final, exact webHello, exact2000/2000
cold GPU-prefill request, OpenAI and Anthropic JSON and SSE schemas, readiness
and models. Original acceptance commands and proof files are in
`round10/acceptance/`; eight API model-request audits are retained separately.
Real codex complete CLI loop wall: **62.962178s**.
Real claude-remote complete CLI loop wall: **47.159395s**.

After removing profiling sinks and restarting again, an existing named `.pkv`
session is reopened with the same prompt: **0.053031s first,
18.592998s total,91/91 IDs equal,0 replay steps**. This independently
checks the legacy-session-replay bypass and that disabling profiling does not
invalidate the bake. Final HTTP config/status/models return200, exactHello
passes, unit enabled/active, one engine; final health and identities are in
`round10/final/final-health.json`. Service remains on0.0.0.0:8080 with profiling off.

## Every model invocation, including slower/failed measurements

Initial candidate runs preceded final relink-invalidation/shared-quota hardening.
They are not omitted or selected over final-build figures. The four initial
HTTP bodies were the captured JSON fallbacks (`stream:false`); HTTP200 was real,
but the helper wrongly asserted SSE first content and exited1. Their JSON first
content is therefore their complete-response time, not a sub-second TTFT.
The corrected final SSE matrix uses original `request-0.json`, a different wire
configuration, and a new build namespace with only flora baked before the other
clients. There was no repeated A/B search or cherry-picked fastest sample.

| Invocation | First content / identified engine TTFT s | Total s | Evidence / qualification |
|---|---:|---:|---|
|initial/cold|24.155769|37.821478|91 IDs; reuse 0/1470|
|initial/memory-repeat|0.042806|6.660835|91 IDs; reuse 1470/1470|
|initial/disk-restore|0.051974|18.610296|91 IDs; reuse 1470/1470|
|initial/restored-repeat|0.045343|6.679849|91 IDs; reuse 1470/1470|
|final/cold|23.952876|37.678831|91 IDs; reuse 0/1470|
|final/memory-repeat|0.039828|6.896898|91 IDs; reuse 1470/1470|
|final/disk-restore|0.056063|18.602763|91 IDs; reuse 1470/1470|
|final/restored-repeat|0.053496|6.835314|91 IDs; reuse 1470/1470|
|final/park-size|0.125647|0.213847|1 IDs; reuse 1470/1470|
|final/restart-existing|0.053031|18.592998|91 IDs; reuse 1470/1470|
|initial/cachyos-http|35.340556|35.340556|JSON; helper expected SSE and exited 1; HTTP 200 retained|
|initial/flora-http|6.872107|6.872107|JSON; helper expected SSE and exited 1; HTTP 200 retained|
|initial/mac-http|35.166400|35.166400|JSON; helper expected SSE and exited 1; HTTP 200 retained|
|initial/x-http|35.369080|35.369080|JSON; helper expected SSE and exited 1; HTTP 200 retained|
|final/final-cachyos|22.333277|35.645492|SSE|
|final/final-flora|0.108062|6.904540|SSE|
|final/final-mac|21.844421|35.468751|SSE|
|final/final-x|22.315975|35.198069|SSE|
|acceptance/openai-json|—|2.916029|functional/schema check|
|acceptance/openai-sse|—|0.133243|functional/schema check|
|acceptance/anthropic-json|—|1.010198|functional/schema check|
|acceptance/anthropic-sse|—|0.132121|functional/schema check|
|acceptance/web-hello|—|0.878834|functional/schema check|
|acceptance/web-capacity|—|9.786245|functional/schema check|
|tool-loop//v1/responses/5|27.413015|39.719054|engine times; api_bd5f217f27954d1d9d373499b29a25cc.response.json|
|tool-loop//v1/responses/6|6.895667|12.310015|engine times; api_0a2d6b7e8e8b46f4a8306de741bd1f0b.response.json|
|tool-loop//v1/messages/7|22.748442|33.904323|engine times; api_bd96fa6c89ec41bfa7813ea4deb3ea08.response.json|
|tool-loop//v1/messages/8|6.739519|11.043052|engine times; api_0ae0c29f33394ba098e86eac8339af91.response.json|
|delivery/Hello|—|3.215115|exact Hello|

Model invocations inside the two strict `decode.` suite executions are preserved
in `decode.txt` and `decode-delivery.txt` (two registered cases each, expected
exit1). CPU/tools are serial and use no inference service. Capture-only CLI
requests return the local placeholder and are not model inferences.

Build/gate attempts are also retained: build1/build2 PASS; build3 FAIL due to
missing `<fstream>` in the new CPU test; build4 and final build PASS. The CPU
command following failed build3 ran the previous binary and is not used as a
final gate. Earlier CPU31/31 and tools58/58 attempts remain alongside final
`cpu-delivery.txt`/`tools-delivery.txt`. A system-Python analysis command lacked
`tokenizers`; using the existing `.venv` corrected it without a model run.
An early audit collector read while a response was still in flight; it was
changed to collect completed receipts only, without repeating inference.
All service boots, model request/response bodies, SSE frames, raw IDs, profiles,
and slower attempts stay privately under `~/cmoeapi/round10/`.

Only implementation, tests and aggregate documentation enter git. Raw prompts,
KV files, client settings, CLI proofs and machine receipts stay out of git.
Final push/HEAD equality is appended to the private report and delivery receipt.
