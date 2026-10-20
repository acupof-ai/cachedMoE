# Round 11: exact expert-cache restart

An opt-in restart optimization, not a cold-prefill algorithm:

```bash
cachedmoe serve --model /path/to/checkpoint \
  --kv-dir /path/to/kv \
  --expert-cache-state /path/to/resident-ids.bin
```

`tools/web/server.py` accepts the same `--expert-cache-state FILE` flag.
Keep this file outside the checkpoint and its mirrors. The option requires an
initial single-stream, non-speculative, off-plain engine. It does not alter the
existing per-request decode-mode API. Without the option, startup and shutdown
retain their previous behavior.

## What is restored

At an orderly shutdown after a successful generation, the engine drains expert
IO and saves resident expert IDs in newest-first order. The file contains no
weights, generated answers, token predictions or KV state. It is bounded by the
current slot capacity and backbone expert count, and carries a version,
checkpoint metadata identity and checksum. Malformed, duplicate, out-of-range,
truncated, stale or capacity-mismatched hints are declined without preventing
ordinary demand loading.

At the next startup the existing free-slot backfill loader rereads those experts
from the **original native checkpoint** before READY. A successful restoration
suppresses the competing static-heat startup fill. Old process timestamps are
not reused: restored hints rank below new demand, preserving ordinary global LRU.
No expert is pinned, no route is masked, and no arithmetic kernel changes.

This deliberately pays IO and latency during startup. It can preserve an already
resident working set across restart, but does not remove its bytes from storage
traffic, improve an empty-cache cold prefill, or guarantee hits for a different
request. Other requests before shutdown can evict the measured prompt's experts.
The startup log reports restored/requested IDs, elapsed seconds, completed IO
bytes and the expected sum of manifest run lengths. Partial restoration retains
demand loading and is not reported as a full restore.

The metadata uses checkpoint path/file-size/mtime identity, not a weight-content
hash. This is adequate for cache hints: even an accepted hint always rereads
current native weights. It is not a numerical-state trust certificate. Disk KV
boundaries remain separately validated, including their existing build identity.

## Safe optional persistence

The writer uses exclusive temporary-file creation followed by same-filesystem
rename. A pre-existing temporary file or symlink is neither followed nor removed.
Only the temporary file created by this invocation is cleaned up on failure.
The final destination is checked against primary and configured mirror roots.
No-request and empty-cache shutdowns preserve previous hints; optional save
failures warn without changing the required conversation-KV shutdown status.
Atomic rename is not an fsync/power-loss durability claim. A stale `.tmp` can
prevent refreshing hints until its owner deals with it; it never authorizes
truncating that file or following a symlink.

## Measured result (2026-10-21)

Linux RADV, balanced, two sources,5500slots, native off-plain/no speculation.
Same captured1470-token Python prompt,91 output IDs, temperature0/seed42.
One baseline and one candidate configuration; the three repeats plus history
continuation are the364-ID consistency gate, not repeated A/B speed selection.

| Target | Before | After | Verdict |
|---|---:|---:|---|
| Cold total ≤15s |37.935224s|37.698884s|**FAIL**,22.698884s above target; within jitter, no cold gain claimed|
| Memory repeat ≥13tok/s |13.450301–13.490266|13.440664–13.454133|**PASS**, already passed before this patch|
| Disk-restored decode ≥13tok/s |4.873064|13.547080|**PASS for the measured restored working set**|
| Disk-restored first content |0.051421s|0.051121s|No material regression|
| Disk-restored request total |18.611983s|6.807471s|After READY only|
| Restart to READY |21.090055s|34.968634s|13.878579s added startup latency|
| Restart through response completion |39.702038s|41.776105s|**2.074067s slower** overall|

Snapshot22048B,5500/5500 IDs restored, original-weight reload13.799s.
Expected aligned manifest bytes103448576000B (96.343994GiB); IO completed-byte
counter delta103448568912B. The7088B difference reconciles exactly to12 selected
runs' sector padding beyond shard EOF (10×168+2904+2504B); actual returned bytes
exclude that allowed padding, not weight payload. Post-restore decode reads75.778931→0GiB;
expert hits79.972222%→100%, measured NVMe-stall132.551198→0.214487ms/token.
Decode90steps18468.872→6643.498ms. No overlapping GPU/fence buckets are added
again. Startup reload is **not** an IO saving; it reads more experts than this
request's demand misses would have required.

The cold prefill still reads exactly165.448914GiB, including156.812988GiB
required union and8.635925GiB unused blind reads. Its expert IO bucket is
17.455818→17.505389s, not an improvement. The current loader already deduplicates
experts and skips resident/read-ahead hits; padding is only8192B per18808832B
slot. No target-closing exact cold IO reduction is implemented. Closed
read-ahead-off, predictive-prefetch, non-LRU and approximate-mode experiments
were not rerun. This is not a proof that every future exact algorithm is impossible.

## Every measured matrix invocation

| Arm / request | First content s | Total s | Engine decode tok/s | Decode GiB |
|---|---:|---:|---:|---:|
|baseline/cold|24.162075|37.935224|6.590637|43.179626|
|baseline/repeat1|0.035334|6.823207|13.490266|0|
|baseline/repeat2|0.036238|6.841119|13.456502|0|
|baseline/repeat3|0.036716|6.844086|13.450301|0|
|baseline/continuation|5.317364|15.910871|8.573290|23.262695|
|baseline/disk-restored|0.051421|18.611983|4.873064|75.778931|
|baseline/restored-repeat|0.036690|6.846373|13.438966|0|
|candidate/cold|24.017214|37.698884|6.631527|43.179626|
|candidate/repeat1|0.036780|6.842858|13.454133|0|
|candidate/repeat2|0.038204|6.848509|13.445016|0|
|candidate/repeat3|0.038960|6.848585|13.440664|0|
|candidate/continuation|5.306743|15.855865|8.615575|23.262695|
|candidate/original after continuation|0.043224|9.206066|9.969461|14.661804|
|candidate/disk-restored|0.051121|6.807471|13.547080|0|
|candidate/restored-repeat|0.042198|6.779716|13.579282|0|

All15 completions match the same91-ID cold reference. Each arm preserves
364/364 IDs across three repeats+C. Cold equals the historical Round10 output.
The candidate's extra9.206s return to the original prompt is explicitly counted:
continuation changed residency, so the snapshot captures the original request's
latest working set. It is not an omitted warmup or a selected speed baseline.
Other intervening work can evict that working set; no usable residency snapshot
means ordinary cold expert demand loading, not guaranteed13tok/s.

## Validation and failures

Original and final executables both pass CPU31/31, tools58/58, Vulkan KV7/7
and wrapped-batch1/1. Strict decode remains6/8 teacher-forced and7/8 own-prefill,
own-prefill free-running6/8; suite exit1 with the same six assertions. No fixture
or threshold changes; no8/8 or fresh historical-NLL certification claimed.
All53 shader hashes are unchanged. New CPU cases cover identity/capacity,
duplicates/range/oversized/truncated/corrupt/trailing data and optional-file safety.

Failures retained: first candidate build exposed an include-order prerequisite
for uint32_t; corrected. The next CPU gate was30/31 because the new corruption
test wrote the same byte already present; corrected to XOR one bit without
changing its rejection assertion. Review caught a temporary-symlink truncation
hazard before candidate inference; exclusive creation and disposable-file tests
fixed it. Final build/gates above passed. No failed candidate GPU inference was
hidden or retried for a preferred speed sample.

## Real-client acceptance and production delivery

All required client gates pass: OpenAI and Anthropic JSON/SSE, exact web Hello,
actual2000-token GPU prefill with0 reused tokens, real Codex shell-write with
exact on-disk bytes, and real Claude model-issued Read→tool_result→final loop.
The original acceptance invocation was9/10: Codex inherited an open stdin and
waited without a model request; it was terminated after260.479s (exit−15).
A fresh Codex-only invocation with stdin=DEVNULL passed in53.220s. The successful
nine gates were not rerun. Claude's two-request loop took49.121s CLI wall.
No client settings or acceptance assertions were relaxed.

| Client/model invocation | Wire/HTTP completion s | Engine decode tok/s |
|---|---:|---:|
|OpenAI JSON Hello|4.844776|15.067880|
|OpenAI SSE Hello|0.129660|14.793041|
|Anthropic JSON Hello|1.003905|15.072350|
|Anthropic SSE Hello|0.129955|14.742771|
|Web Hello|0.888119|14.410633|
|Exactly2000 GPU prefill|10.574619|not applicable: one output token|
|Claude Read request|34.941015|7.365181|
|Claude tool-result continuation|10.479411|7.631761|
|Codex shell-write request|36.775602|7.464897|
|Codex tool-result continuation|6.189753|7.028302|
|Final fresh-session Hello after deployment restart|2.142972|9.010351|

Claude times are captured proxy wire completion; Codex times are server-side
API completion, not full CLI wall. Short Hello/one-step throughput is not the
91-token target. Tool prompts differ from that fixture and are explicitly not
claimed to reach13tok/s. The final fresh Hello also demonstrates that restoring
a different working set does not guarantee all expert hits.

Production first boot20.786128s (no resident hints); graceful shutdown saved5500
IDs to `build/web_exact/resident-ids.bin`. Final restart34.465141s,5500/5500
restored in13.658s, actual completed103448560736B/expected aligned103448576000B.
Final CPU31/31 (3.205103s) and tools58/58 (9.928054s) passed serially before restart.
Unit enabled/active, HTTP config/status/models healthy, listener0.0.0.0:8080,
exactly one engine, native off-plain/no speculation, two sources and profiling off.
Executable SHA256 remains `f231854773571fb1fbfc9bfb09b073d5d3535997efc623a970a08d7f45b4ec30`.

Raw requests, IDs, profiles, complete gate output, service boots and failed
attempts stay privately under `~/cmoeapi/round11/`; the detailed private report
and final git delivery receipt are `~/cmoeapi/report11.md`. Only implementation,
tests and aggregate documentation enter git.
