# Correctness Round 7 — strict gate blocks further work (2026-10-19)

**BLOCKED; no correctness fix or speed claim.** Starting `main` was `95a9031`,
clean and published. The brief explicitly requires stopping when the numerical
acceptance gate cannot pass. One bounded check on the preserved, oracle-matching
checkpoint prints **6/8**, not the required **8/8**. It was terminated after
65.783 s once the strict failure was established. The remaining short-decode
sections were not completed. No candidate runtime change was made or deployed.

Machine report: `~/cmoeapi/report7.md`; raw logs, commands and closure receipts:
`~/cmoeapi/round7/`. These machine artifacts stay out of git.

## 1. What the provenance investigation establishes

`1017` is the checkpoint revision suffix, **not a count of 1,017 bad tensors**.

| Read source | Revision | Existing oracle relationship |
|---|---|---|
| `~/models/DeepSeek-V4.1-Flash` | production1017 | Old exports are not a matched validation pair |
| `/mnt/deepmoe2/models/DeepSeek-V4.1-Flash` | production1017 mirror | Same bounded samples as production, not an original-weight source |
| `/mnt/deepmoe2/models/DeepSeek-V4.1-Flash-0928` | preserved original | Historical NLL reproduced; strict short decode still fails |

Read-only checks of manifests and three bounded tensor samples confirm these
identities: the entire 10,240-byte layer-0 attention norm, 4,096 bytes of WKV,
and 4,096 bytes of expert-0 W1. Production and its mirror agree on these
samples; original differs in all three. This is **not a whole-checkpoint hash**.
Digests/offsets are retained in `~/cmoeapi/round6/checkpoint-identity.json`.
The mirror filesystem, including original0928, is mounted read-only.

Existing Round-6 evidence is reused, not rerun as another A/B:

| Weights / attention | Off64 NLL | Disposition |
|---|---:|---|
| original0928 / TILED=0, CM=1, KSPLIT=1 | **0.622784** | Reproduces the historical Linux baseline only |
| production1017 / default tiled precedence | 0.660034 | Conditioned on an unmatched old fixture; not a production quality baseline |
| production1017 / historical CM settings | 0.693887 | Same provenance problem; attention-only switching does not repair it |

Source: [Round 6 numerical provenance](engine_latency_round6.md#historical-numerical-provenance),
including exact commands in `~/cmoeapi/round6/gpu-commands.json`.
**Neither newer weights being corrupt nor a new acceptable baseline follows
from those mismatched comparisons.** The old exports remain usable for the
original revision's historical regression measurements; they are untrusted as
an oracle for production1017.

There is no weight/export identity guard in `runtime/decode_state.cpp`'s loader
or `tools/l3_ppl.py`'s independent model/state selection. Short-decode fixtures
remain `tests/data/l3` plus L2 probes in `tests/data/l2`, regardless of
`CACHEDMOE_MODEL_DIR`; NLL64 uses `traces/l3_64`.

The original revision could be selected through existing `--model` support,
without writing checkpoint files. It must **not** use the 1017 mirror:
`store/shard_set.h` checks names/lengths rather than content, and different
revisions share shard lengths. Original-only testing must disable mirror
auto-discovery/explicit inherited mirrors and isolate disk KV. This was not a
quality-qualified production candidate, so no service weight switch was made.

## 2. One fresh, bounded strict numerical check

After stopping `cachedmoe-lan.service` and verifying no live service engine:

```text
build/tests/cachedmoe_tests decode.
CACHEDMOE_MODEL_DIR=/mnt/deepmoe2/models/DeepSeek-V4.1-Flash-0928
CACHEDMOE_SHADER_DIR=<checkout>/build/shaders
CACHEDMOE_MODEL_MIRRORS=  CACHEDMOE_MIRROR_AUTO=0
CACHEDMOE_ATTN_TILED=0  CACHEDMOE_ATTN_CM=1  CACHEDMOE_ATTN_KSPLIT=1
```

All inherited `CACHEDMOE_*`/`DEEPMOE_*` overrides were removed first. This uses
the existing main executable, no new binary/shader or weight transformation.
The result is a correctness check, **not a speed comparison**.

```text
step 6  in      1  top1 1 vs 201 DIFFER
        reference margin 0.9485, ours 0.1018
step 7  in    201  top1 45324 vs 61 DIFFER
        reference margin 1.9522, ours 0.2091
teacher-forced: 6/8 tokens match, 1 miss by more than 0.5
```

SIGTERM stopped the verified test PID at **65.783 s**, exit **-15**, after the
strict verdict was decided. The own-prefill/free-running/Engram sections were
not completed or claimed to pass. Raw source: `trusted-decode.txt`, invocation
and outcome in `trusted-decode-command.json` / `trusted-decode-result.json`.
Historical original-era receipts already record **6/8 + 7/8**, so merely
restoring original weights does not satisfy the brief's **8/8 + 8/8** condition.
The tolerance-based CTest exit is not that strict gate: `test_decode.cpp`
explicitly allows near-tie discrepancies. No threshold, fixture or token was
changed to turn this into a pass.

## 3. Reuse is still unqualified; no unsupported causal claim

The previous code repeat/continuation returns False for `is_prime(3)`; that
failure is unresolved, not hidden by the provenance finding. The existing
saved-boundary test covers seven exact direct-decode steps, not the service's
free-running dynamic-mask/relaxed-speculative policy. Restoration does not
rewind expert residency, and masking/relaxed acceptance are additional sources
of output dependence. **No causal KV-corruption diagnosis was established.**
Blindly reverting saved-boundary reuse would both lack a causal proof and lose
the required reuse/latency behavior without fixing the strict numerical gate.
No such revert, output patch, or purported correctness fix was attempted.

Per the brief's stop rule, no fresh Claude cold/repeat/continuation matrix,
prefill experiment or decode optimisation was run. The last measured
100%-reuse repeats and failing answers remain historical Round-6 evidence,
not Round-7 acceptance. Avoided GiB, saved ms and speedup for Round 7 are
**not measured: no performance candidate exists**.

## 4. Gate accounting and restoration

| Gate | Fresh Round-7 disposition |
|---|---|
| CPU CTests | **31/31 PASS**, 2.97 s, service stopped |
| Tools gate runner | **58/58 PASS**, including metadata suites |
| Strict original-weight short decode | **FAIL 6/8**, bounded termination; remaining sections incomplete |
| NLL64 | Not rerun; Round-6 values explicitly identified above |
| Prime cold/repeat/continuation and reuse >=95% | **Not rerun; unresolved** |
| Real Codex shell-write / Claude Read loops | **SKIP after correctness stop**, not fresh passes |
| Exact 2000-token GPU-prefill request | **SKIP after correctness stop** |
| Full OpenAI + Anthropic JSON/SSE acceptance | **SKIP after correctness stop** |
| Web Hello | **PASS restoration smoke**, exact `Hello`, HTTP200 and terminal done |
| Service API health | `/api/config`, `/api/status`, `/v1/models` HTTP200 |

Only one fresh model-output request followed the failed test, after service
restoration: web Hello, **3.871 s** client wall / **3.810 s** engine total,
11 prompt tokens, two generated tokens, decode prefill, existing mask-spec
policy. All of its timings are retained, not selected from multiple attempts.
It demonstrates liveness **only**, not prime correctness, throughput acceptance,
or the skipped schema matrix.

The original production service is restored on **0.0.0.0:8080**, unit enabled
and active, exactly one service engine, 5000 slots, two production1017 read
sources, dynamic mask, k2/top4, prefill min16/speedup12. User transcripts/KV
were not migrated, deleted or replaced. No checkpoint writes, sudo, global
client configuration changes, performance-profile edits, branches or worktrees.
Only this report, STATUS and the current RUNNING note are changed. Publication
HEAD/remote equality and live-health evidence is in
`~/cmoeapi/round7/publication.json`.

## Required next input/work, not a success claim

1. Obtain independently trusted exports bound to production1017, or fix the
   original-revision numerical path until the strict gate genuinely passes;
   do not overwrite the historical fixtures/baseline to manufacture agreement.
2. On a valid numerical pair, isolate first cold/repeat divergence with the
   same exact policy/residency, then test full free-running production policy
   and the executable prime oracle. The saved-KV test is insufficient.
3. Only after both pass, resume GiB/ms-attributed prefill and >=13 tok/s work,
   followed by every skipped client/API/GPU acceptance gate.
