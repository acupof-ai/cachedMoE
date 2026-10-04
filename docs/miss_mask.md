# Miss masking with normal demand LRU (2026-10-04)

`deepmoe serve --resident-only mask` (also `run` and
`DEEPMOE_ROUTE_RESIDENT_ONLY=mask`) is an opt-in lossy mode. Default remains `off`.

The gate's original full top-k list and weights go to the existing demand planner.
Hits are touched, misses evict/admit via normal LRU, and P0 reads are issued as
usual. A miss already filling is stamped with the newest request time; its
completion preserves that stamp. There is no resident-only P3 miss queue,
renormalisation, replacement expert, or frozen cache.

Only the planner's hits contribute to the current single-position forward.
Miss weights become zero even if their read completes during planning; this
forward never waits for them. The bridge compacts its dispatch slot list, so
masked slots participate in neither gate/up, h quantisation, nor down. Shared
expert always contributes, including when every routed expert misses. Hit
weights remain unchanged. Masks therefore change output numerics and can change
future routes/cache contents, although the LRU algorithm is unchanged.

GPU prompt prefill stays exact. The experimental mask is supported by
single-position forward, including the slow prompt-feed path and multistream
single-position decode. The subsequent [DSpark integration](dspark_topk.md)
also supports masking in `forward_batch`: plan the full union normally, then
compute only each row's planner hits with their original weights.

## Validation

Sources: `bench/results/miss_mask/gates/`.

- CPU ctest 25/25; Python gates 30/30.
- `suite.decode` and `suite.decode_longctx` both pass (64/4K/17K reference gates).
- New GPU test `gpu_moe.masked_missing_slots_and_shared_only` passes: a masked
  slot has no resident pointer row; stale h from an earlier full call contributes
  nothing; an all-miss layer runs on shared alone.
- New CPU tests check that an async fill cannot overwrite a newer demand's
  LRU stamp, and a reused slot cannot inherit its evicted occupant's stamp.
- 64-step teacher-forced l3: off NLL **0.622784**, bit-for-bit platform baseline;
  mask **0.835581**. PPL **1.8641 -> 2.3062 (1.237x)**, top-1 **56/64 -> 50/64**.
  The l3 harness starts from static heat, hit 81.4%, and its **3.544 -> 6.187
  tok/s is not a conversation speed comparison**. These cells ran on one drive.

## Reproduce

Use the mounted read-only mirror and serialize all GPU commands. Conversation
arms use `tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json
--out OUT --require-sources 2 --serve-arg=--resident-only --serve-arg=MODE`,
with `DEEPMOE_MODEL_DIR` set and `DEEPMOE_MIRROR_AUTO=1`; MODE is `off` or `mask`.
The MMLU command is:

```bash
DEEPMOE_MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash" DEEPMOE_MIRROR_AUTO=1 \
  .venv/bin/python tools/mmlu_bench.py \
  --sample bench/results/miss_mask/mmlu_sample.json \
  --out bench/results/miss_mask/mmlu --limit 57 --require-sources 2
```

## Conversation and MMLU protocol

The dual-drive 8-turn conversation is complete:

| Metric | off | mask |
|---|---:|---:|
| Decode tok/s | 9.1095 | 13.0281 |
| Decode ms/token | 109.78 | 76.76 |
| End-to-end tok/s | 7.7552 | 10.5717 |
| Decode cache hit % | 94.183 | 94.048 |
| Planner/gate ms/token | 42.95 | 1.21 |
| Attention ms/token | 29.24 | 31.85 |
| MoE GPU ms/token | 25.90 | 24.90 |
| Engram ms/token | 1.90 | 6.12 |
| Prefill total seconds | 44.68 | 42.10 |
| Decode steps | 2,277 | 2,318 |
| Mirror max C / thermal rests | 74 / 0 | 74 / 0 |

Decode **+43.0%**, end to end **+36.3%**. Source errors/drops: zero in both
arms. The cache had 5,500 auto-sized slots in both. The mask arm skipped 34,537
of 565,920 routed requests across its single-position layer-steps (6.10%),
losing 4.83% mean gate mass. Background miss queue enqueued **0**: original
P0 demand loading is retained. Total source reads were 820.9 / 842.9 GiB;
**the mechanism saves the wait, not the bytes**. Faster generation changes
answers/routes/lengths, so the scripts match, not the token sequences.

Both speed cells used exe `e8a4b07b07df586b` and shaders `c3d403af6b51e18c`.
The later selected-logit protocol addition does not change inference. A final
slot-stamp reset prevents a newly admitted stale background expert from
inheriting its evicted occupant's stamp; normal monotonic demand admissions in
these off/mask cells already have newer stamps. The final build's off NLL
was rechecked at **0.622784**. No conversation cell was repeated.

Sources: `speed_{off,mask}/{provenance.json,perf.json,status.json,events.jsonl}`.
`thermal.jsonl` records GPU/CPU/NVMe temperatures and GPU clocks. Clocks stay
under normal DPM and differ with sustained load; they were not pinned high.
Each next dual-drive cell waits for GPU <=60 C and NVMe <=70 C. Runtime mirror
thermal gating remains **80 C pause / 72 C resume**.

Speed uses the existing `bench/results/hitrate/long_turns.json` script and
`tools/hitrate_bench.py`, once per mode with auto cache sizing, empty initial KV,
normal prompt prefill and cold process startup. Compare decode separately from
prefill/end-to-end time. `--require-sources 2` refuses a cell if the mirror health
check falls back to one source. Both arms must use the same source count.

The local sample is from https://huggingface.co/datasets/cais/mmlu, `all/test`
(14,042 questions, 57 subjects); parquet SHA-256
`74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`.
`mmlu_sample.json` contains three questions per subject, seed 42, ordered in
57-question balanced rounds. First round covers all 57 subjects.

`tools/mmlu_bench.py` evaluates zero-shot, taking the argmax of the raw logits
for A/B/C/D. It verifies each letter is one tokenizer token. `score_tokens`
reads only those selected rows of `Engine::last_logits()`; it refuses stale
prefill/uninitialised/reset/session-mismatched state. The evaluator also scores
the emitted greedy token and checks it is at least as high as all four choices.
This is a zero-shot sample, not the canonical full 5-shot benchmark. Each question resets KV but
keeps the expert cache. Exact GPU prefill processes all except the final prompt
token; a second request extends by that token through single-position forward,
so miss masking affects the answer distribution. The script validates that the
prefix really used GPU prefill and the final request reused the entire prefix
and fed exactly one token. Paired arms use the same questions and order. Each
answer, timing, routing trace, live source count and binary provenance is saved.

A discarded generation-format pilot completed only 12 off-mode questions:
8 correct, 3 outputs unparseable because the model began explaining and the
8-token limit truncated it. It was stopped once this protocol fault was clear,
not counted as a MMLU score, and preserved under `mmlu_generation_pilot/`.
The selected-logit evaluation uses the same question sample.

The completed **57-question / 57-subject zero-shot sample**:

| Metric | off | mask |
|---|---:|---:|
| Correct | 45/57 | 44/57 |
| Accuracy % | 78.95 | 77.19 |
| Evaluation wall seconds (excluding startup/cooling) | 646.6 | 890.0 |
| Sources / cache slots | 2 / 5,500 | 2 / 5,500 |

Paired changes: **1 correct -> wrong**, **0 wrong -> correct**;
2 answers changed in total. Difference **-1.75
percentage points**. With only one question per subject, this is an initial
quality probe; it does not establish the full MMLU score or losslessness.
The teacher-forced PPL degradation remains a separate observed quality cost.
MMLU's final-token mask skipped 1,381/13,680 requests
(10.10%), mean gate mass lost 7.33%.
Exact prompt prefill dominates this evaluation, so its wall-time ratio is not
a conversation decode speedup. Both arms used exe `14ca288dd5b944e0` and
shader `c3d403af6b51e18c`, with no source read errors or drops.

Sources: `mmlu/{off,mask}/{answers.jsonl,summary.json,status.json,provenance.json}`
and `mmlu/paired.json`. MMLU thermal telemetry is in `thermal_mmlu.jsonl`.


Both MMLU arms reached mirror max 74 C, rested 0 times, and finished with
no outstanding reads. The measured exact-prefix prefill totals were 629.1 / 877.8 s
(off/mask). Thus this sample took longer in the mask arm, in the exact prefill
portion; the logs show mirror P0 mean latency 141.22 / 203.63 ms. This
is an observed IO-time difference, not evidence that skipping routed decode
work speeds up short-prompt exact prefill. No extra run was made.
