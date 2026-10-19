# Engine latency Round 6 — blocked acceptance (2026-10-19)

**No new runtime optimisation is accepted.** Starting main `01987e4` was clean;
Round 5 had already committed/pushed. The nine-request validation completes
without retries, but code repeat and continuation fail the executable prime
oracle. Cold total and cold code/Chinese decode targets remain unmet. This is a
hard-blocker closure, not an optimisation success or full numerical acceptance.

Detailed machine report: `~/cmoeapi/report6.md`; all raw artifacts/scripts stay
outside git in `~/cmoeapi/round6/`. No checkpoint writes, re-quantisation,
repacking, global client configuration, runtime flag, shader, sampling, oracle
or threshold change was made. No committed Round-5 implementation was blindly
reverted without establishing a causal defect.

## Complete request matrix

Existing production1017 checkpoint, two read sources, 5000 slots, dynamic mask,
DSpark k2/top4, GPU prefill min16/speedup12. Cold means zero KV reuse, not flushed
expert/OS caches. Six real flora Claude2.1.302 requests use invocation-only
settings, tools disabled, JSON/no-session-persistence. Each true continuation
uses the Round-4/5 HTTP method: preserve the real Claude request/history, append
its actual reply and the same verbatim question. Those three are **HTTP history
continuations, not CLI JSON TTFT measurements**. This inherited protocol
limitation is not represented as strict all-nine-CLI acceptance.

One cold/repeat/continuation per prompt, serial, no favourable-run selection.
**All nine: HTTP200, terminal message_stop, no errors, one HTTP attempt each.**
Wire decode retains prior rounds' nonempty-frame progression metric; one-frame
arithmetic continuation has no wire decode rate.

|Case|First / total wire s|Reuse %|Prefill ms|Replay ms|Engine / wire tok/s|Expert hit %|Answer|
|---|---:|---:|---:|---:|---:|---:|---|
|code cold|22.969 / 31.766|0|22870.242|0|12.580 / 12.528|84.584|prime oracle PASS|
|code repeat|.064 / 7.758|100|.509|0|14.090 / 13.965|93.106|**prime3 FAIL**|
|code continuation|3.727 / 13.380|98.011|3593.947|0|11.123 / 11.014|87.969|**prime3 FAIL**|
|arithmetic cold|22.313 / 24.971|0|22181.971|0|10.166 / 9.690|82.241|240; number-only FAIL|
|arithmetic repeat|.062 / 2.327|100|.473|0|12.472 / 11.629|95.435|240; number-only FAIL|
|arithmetic continuation|4.588 / 4.995|97.411|4423.262|0|3.003 / n/a|**77.429**|exact240 PASS|
|Chinese cold|22.771 / 34.516|0|22569.660|0|8.184 / 8.140|86.967|core meaning preserved|
|Chinese repeat|.075 / 11.060|100|.558|0|8.355 / 8.257|96.051|core meaning preserved|
|Chinese continuation|3.355 / 12.607|98.417|3277.954|0|9.269 / 9.120|94.752|core meaning preserved|

Claude actual first-content cold code/math/Chinese:22.992/22.336/22.793 s;
repeats:.086/.085/.097 s. JSON ttft_ms remains completion, not first text.
Python is executed against −10..1999 and99991/999983/1000000. Cold guards n<4;
repeat/continuation change it to n<3, then reject multiples of3. Both return
False for3. Chinese preserves cache→traffic→latency meaning, with generic
HBM/PCIe wording rather than literal Strix Halo topology; repeat's opening
wording is imprecise. No answer rewriting hides these failures.

## Before/after and attribution

|Goal|Before|Round6|Verdict|
|---|---|---|---|
|code identical first content|R4 18.435 s; R5 wire.057 / Claude.079 s|wire.064 / Claude.086 s|latency passes, answer FAIL|
|identical reuse / replay|R4 99.86% /18.128 s; R5 100% /0|all repeats100% /0|reuse passes, not quality-qualified|
|cold code/math/Chinese total|R5 33.491/25.338/33.788 s|31.766/24.971/34.516 s|all fail≤15 s|
|cold code engine/wire|R5 11.944/11.846 tok/s|12.580/12.528|both fail≥13|
|cold Chinese engine/wire|R5 8.861/8.844 tok/s|8.184/8.140|both fail≥13|
|expert hit floor|83.78−2=81.78%|cold82.241–86.967%; minimum77.429%|cold passes; per-invocation floor FAIL|

Existing exact-boundary restoration takes.508/.473/.558 ms. It removes the old
rollback replay for retained exact prompts, but avoided GiB is **unknown**: old
replay byte accounting is absent. Only the first result is retained; later
outputs still decode. No new Round6 code saving exists.

|Cold case|R5 expert GiB / IO ms|R6 expert GiB / IO ms|R6 expert GPU / setup ms|
|---|---:|---:|---:|
|code|165.45 /18004.875|157.43 /16795|2523 /170|
|arithmetic|150.82 /16325.432|150.94 /15952|2580 /172|
|Chinese|151.51 /16489.476|151.63 /16399|2601 /175|

Descriptive differences:−8.02/+.12/+.12 GiB, approximately−1210/−373/−90 ms IO.
**No optimisation attribution:** unchanged implementation/settings; code starts
with477 resident experts versus R5's0. Prompt expert unions are already
deduplicated; read-ahead-off's R5 NO-GO is not rerun. Prefill alone still costs
22.182–22.870 s. Verify/fence and Engram remain material decode costs; the raw
per-op/speculation counters are preserved, not a claim of a universal ceiling.
Halve any predicted gain; no credible correctness-qualified candidate was found.

## Historical numerical provenance recovered

The unchanged l3_64 oracle identifies the original model. Production manifest
is now DeepSeek-V4.1-Flash-1017; the preserved0928 checkpoint is original.
Bounded samples confirm **different actual weights**, not label-only changes:
whole layer0 attention norm10,240B, first4096B WKV, and first4096B expert0 W1
have different SHA256 values. Exact paths/offsets/digests are in machine
checkpoint-identity.json; no full-weight hash/scan or checkpoint write occurred.

After stopping the sole service, the actual main build succeeds. Three serial
64-step off runs use the same executable/shaders and unchanged oracle, remove
inherited CACHEDMOE_/DEEPMOE_ overrides, and explicitly pin attention settings:

|Checkpoint|Attention|Printed NLL|Top1|
|---|---|---:|---:|
|original0928|TILED=0, CM=1, KSPLIT=1|**.622784**|56/64|
|production1017|TILED=1, CM=1, KSPLIT=1 (default precedence)|**.660034**|53/64|
|production1017|TILED=0, CM=1, KSPLIT=1|**.693887**|53/64|

Historical reproduction therefore succeeds. Fixed-CM model change:+.071103;
1017 tiled versus CM:−.033853; net+.037250. These are **old-fixture-conditioned
diagnostics**, not fresh1017 quality rankings or permission to replace the
historical baseline. Early norm-stage mismatches precede tiled attention and
are consistent with changed norm weights. f10b23f also changed attention
precedence after historical acceptance. Switching only attention does not
repair mismatched checkpoint/oracle provenance.

Fresh current decode: **6/8 reference +7/8 own-prefill**, two cases/one failed,
six failed assertions, exit1. Fresh retained-boundary GPU test: **one case,
zero failures**, original ring bytes plus7/7 forced token/top1/top2 identical.
That narrow direct-decode test does **not** prove free-running API speculative
correctness. Full legacy kv_replay/all GPU CTests were not run or passed.

## Other fresh regression gates

- CPU: **100% tests passed out of31**; tools: **58/58 gates passed**, serial.
- Real Codex shell-write: executes/exit0, exact fresh fixture bytes, next request
  reports6268 cached input tokens. Existing model-metadata warning retained.
- Real Claude Read: actual tool_use/result in second request, fresh fixture proof
  in final answer, two turns/no denials.
- OpenAI/Anthropic JSON and SSE: **ALL HTTP SCHEMA/STREAM CHECKS PASS**.
- Web Hello passes; fresh exact2000-token preview/request has reused0, GPU
  prefill **9511.675460 ms**. Restored service returns exactHello after GPU gates.

## Blocker and disposition

No missing deterministic restore state was established. Existing code restores
ring/carry/DSpark hidden state and reseeds draft; Mask zeroes non-hit expert
weights and restoration does not rewind cache/LRU, so residency dependence is
possible, **not causal attribution** of the prime failure. Relevant existing
paths: runtime/session.cpp prompt-snapshot selection; runtime/engine.cpp retained
state and Mask/DSpark; runtime/kvstore.*; runtime/engram.*; prefill_kernels.cpp
expert union; decode_layer.cpp attention precedence; test_kv_replay.cpp's narrow
forced-step check.

Before further speed work: pair1017 with trusted matching exports, then isolate
production speculative/mask first divergence at identical residency/policy and
odd/even boundaries. No threshold weakened, old-model switch silently deployed,
DSpark/mask quality loss traded for speed, or claimed all-goals pass. Service
remains on existing1017 configuration; this report, STATUS, and a follow-up note
on the historical Round-5 report are the only changes. Publication/healthy enabled8080 receipt lives in round6/publication.json.
