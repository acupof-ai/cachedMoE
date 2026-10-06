# Miss masking with normal demand LRU (2026-10-04)

`cachedmoe serve --resident-only mask` (also `run` and
`CACHEDMOE_ROUTE_RESIDENT_ONLY=mask`) is an opt-in lossy mode. Default remains `off`.

2026-10-05: dynamic LRU is again mask's default. `--mask-cache fixed` freezes
the initial set explicitly. The legacy env flag `DEEPMOE_MASK_DYNAMIC_LRU=0`
also selects fixed; `1` selects dynamic, and the CLI takes priority. The
[adaptive-freeze scan](mask_freeze.md) did not pass its offline usefulness gate.

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

## Quality recovery: Phase A and repetition gates (2026-10-05)

**The loading regression is fixed; dynamic mask has not passed the full quality
gate.** `66d0f7a` allowed Engram P2 requests to bypass P0. Mask decode could then
run ahead of its asynchronous expert fills. Restoring P0 priority recovered the
historical teacher-forced result without changing routing mathematics.
`ce589e8` makes this the dynamic-mask default; fixed-cache experiments retain
their old behaviour. `DEEPMOE_IO_ENGRAM_DEADLINE=1` is an explicit experiment.
LRU planning and expert loads remain asynchronous. Engram can again wait behind
expert loads; removing that delay had also removed the cache's catch-up time.

Each cell ran once with 5,100 slots, static heat, no DSpark and the same l3 trace:

| Cell | NLL | top-1 / 64 | served | lost mass | teacher-forced tok/s |
|---|---:|---:|---:|---:|---:|
| Historical, one source | .835581 | 50 | .8135 | .1694 | 6.187 |
| Current, one source, P2 bypass | 1.610124 | 39 | .5326 | .4462 | 14.309 |
| One source, bypass disabled | .835581 | 50 | .8135 | .1694 | 6.120 |
| Two sources, bypass enabled | 1.383369 | 39 | .6511 | .3310 | 13.712 |
| Two sources, bypass disabled | .835581 | 50 | .8135 | .1694 | 7.597 |

This isolates the switch; a commit bisect was unnecessary. The single-position
l3 path does not use `BATCH_ENGRAM_EARLY`; MTP pins are absent, and the actual
cache has 5,100 slots. A higher context ceiling does not allocate that many KV
rows. The faster bypass cells computed substantially less routed expert work;
their throughput is not a gain at equal quality.

The final dual-source check reproduced off **.622784**, mask **.835581**,
50/64, served **.8135**, lost mass **.1694**. P0 averaged **62.82 ms**, split
into **60.42 ms queue** and **2.39 ms issue-to-land service**. Landing copy was
**.739 ms/request inside service**, not an additional serial cost. Source 0/1
P0 queue was **62.18/61.02 ms**, service **2.35/1.97 ms**, copy **.376/.415 ms**.
Service includes IO and landing; it does not isolate pure SSD time. P0/P2/P3
in-flight chunk peaks were **8/96/8**, with zero outstanding at shutdown and
zero reserve/submit/IO/failed-fill failures. These request latencies are not
decode joins and cannot be multiplied by miss count to predict stall time.

Quality validation used two healthy sources, 5,500 slots, exact GPU prefill,
dynamic decode masking, no speculation, power-saver and AC:

| Gate | Result | Decision |
|---|---|---|
| Chinese 64, T=0 / T=1 | no loops; 11.22 / 10.65 tok/s | pass |
| Three 512-token outputs | all no loops | structural gate passes |
| Follow-up repeated 4-grams | mask .117878 / off .049116 = 2.40x | **fails <=1.5x** |
| Generated-answer MMLU57 | **46/57**, two invalid formats | **fails >=48/57** |
| Eight-turn plain mask | **110.142 ms/token, 9.079226 tok/s**; hit .9390, lost mass .0483; no loops | Phase D baseline |

The other long-output repeated-4 fractions were off/mask **.033399/.033399**
(Chinese thinking) and **.003929/.001965** (English). This MMLU uses the
generated `Answer: X` protocol, not the older selected-logit result above;
format failures count as incorrect. Each eight-turn arm starts with empty KV,
keeps one session for its eight turns, and uses the same script and seeds.
The baseline's P0 mean was **27.89 ms = 24.67 queue + 3.22 service**; its fills
all completed successfully. All seven Phase A jobs had zero thermal pauses;
GPU peak at most **72 C**, external NVMe at most **74.85 C**.

`tools/repetition_metrics.py` now supplies token-level checks to both benchmark
drivers. It rejects the known fixed-cache `霓` loop and the k5 English period-2
loop, and accepts the two known normal Chinese outputs. Synthetic CPU cases
cover run-length, short-period and event-boundary edges. CPU/tool gates pass
25/25 and 33/33. The tests detect exact repetition, not semantic correctness.

The tested Phase A executable SHA-256 starts `0930bb7174b1a98a`; raw commands,
environments, outputs, source status and thermal samples are under
`bench/results/mask_quality/phase_a/{matrix,validation}/`. The recovery cells
are also recorded in `phase_a_matrix.json`, and repetition calibration in
`phase_b_calibration.json`. A passes its recovery thresholds; MMLU and long
repetition failures prohibit promoting plain mask as a newly qualified default.

## Repetition detector calibration: Phase B

**The known-output calibration and CPU boundary checks passed.** The detector
uses a 128-token window, periods up to eight, at least four cycles and at least
16 loop tokens. A same-token run longer than three also fails. The four saved
calibration outputs give:

| Output under `bench/results/` | Tokens | Longest run | Short-period loop | Repeated 4-grams | Distinct-2 | Result |
|---|---:|---:|---|---:|---:|---|
| `spec_e2e/fixed_final/fixed_mask` | 64 | 52 | period 1, indices [12,28) | .786885 | .206349 | fail |
| `spec_e2e/final_dual/mask` | 64 | 1 | none | .000000 | .968254 | pass |
| `spec_e2e/final64/mask` | 64 | 1 | none | .000000 | .952381 | pass |
| `web_spec5_long/spec5/turn2_events.jsonl` | 512 | 1 | period 2, indices [337,353) | .379175 | .508806 | fail |

The raw receipt is `bench/results/mask_quality/phase_b_calibration.json`.
`tools/tests/test_repetition_metrics.py` covers runs of three versus four;
14 versus 16 period-2 tokens; an embedded period-8 loop; period nine and only
three cycles; empty input and n-gram arithmetic; output boundaries, prompt
events and invalid boolean token IDs. Both benchmark drivers retain the four
metrics for each output. These checks detect exact repetition, not semantic
correctness.

The first tool run was **32/33**, with `suite.io` failing
(`bench/results/mask_quality/cpu_tools.log`); it was not a full pass. The final
CPU and tool receipts are **25/25** and **33/33** in `cpu_final.log` and
`tools_final.log` under the same raw root. A later concurrent tool run also
failed `suite.io`; its serial recovery is recorded in `final_tools_serial.log`.
The repetition test itself passed in those logs.

Subsequent numerical checks are accounted separately in
`bench/results/mask_quality/final_validation_summary.json`: six successful
`final_review/` jobs plus three corrected `final_review_remaining/` jobs,
with zero skips. The failed first off-NLL preparation is excluded. This is
nine completed validation jobs, including a CPU preflight rejection. The
receipt preserves the numerical binary's identity
and the current short-decode baseline; it does not claim strict token accuracy
of 8/8 or a fresh GPU rerun of the later build with clock metadata.

## Weighted miss waits: Phase C (2026-10-05)

`DEEPMOE_MASK_WAIT_TAU` is an experiment, **disabled by default**. For each
layer it selects the smallest descending miss-weight prefix needed to bound
lost routed mass. An all-miss layer waits for at least one expert, except at
tau=1, which preserves all-mask behaviour. The selected weights return unchanged
when their P0 reads land; the remaining misses retain zero weight. All misses
still enter normal LRU/P0 planning. `DEEPMOE_MASK_WAIT_BUDGET=experts,milliseconds`
limits actual joins per token; zero means unlimited, default is `8,20`.
Startup logs the values. Speculation and batched verification reject this
option: a decode-only experiment cannot silently stall or alter a verify batch.

Offline mixed-trace replay covered 27,399 weighted token rows at 5,500 slots.
Its LRU uses immediate fills and fixed routes; it omits IO lag, guards, prefill,
budgets and generation feedback. Its hit estimate is .9170. Mean selected
experts per token / remaining lost mass:

| tau | selected experts/token | mean lost mass |
|---|---:|---:|
| 1 | 0 | .070385 |
| .30 | 2.738 | .053426 |
| .20 | 5.521 | .041628 |
| .15 | 8.245 | .031037 |
| .10 | 15.819 | .007781 |
| .05 | 19.493 | .000373 |

These are selection counts, not serial IO costs. Multiplying them by P0 mean
latency would count overlapping or already queued requests several times.

The first GPU screen used two sources, 5,100 slots, static heat, l3_64,
power-saver, no speculation and **unlimited `0,0` budgets**. Each ran once:

| mode | NLL | NLL/off | top-1 / 64 | served | lost mass | teacher-forced tok/s |
|---|---:|---:|---:|---:|---:|---:|
| off | .622784 | 1 | 56 | 1 | 0 | 4.665 |
| tau0 | .622784 | 1 | 56 | 1 | 0 | 4.532 |
| tau.20 | .626679 | 1.006254 | 60 | .8960 | .0687 | 6.439 |
| tau.10 | .623711 | 1.001488 | 62 | .9666 | .0160 | 5.075 |

Tau0 also reproduced all 64 off greedy IDs. Both candidates pass the <=1.10
NLL ratio screen. More top-1 matches here do not prove a general quality gain.
These rates include the teacher-forced runtime's workload and are not the
eight-turn conversation comparison. Long-output, MMLU and conversation speed
decisions are still pending; the default `8,20` budget is not qualified by this
unlimited-budget test.

The final refactor build used executable SHA-256 `08a13390735e53ee...`, source
`4a262da`; all 52 SPIR-V outputs are identical to the earlier Phase A shaders.
The five existing GPU route/ONECB/zero-target/window-wrap/multistream-rejection
tests ran with no skips or failures. All four NLL jobs had AC, zero thermal
pauses, GPU peak <=69 C and external NVMe <=74.85 C. Raw receipts:
`bench/results/mask_quality/{phase_c,gpu_review}/` and
`gpu_review_build_provenance.json`; offline selection:
`phase_c_offline_final.json`.

## Cache capacity audit: Phase E

**No allocation change is justified by the >=150-slot reclaim rule.** The final
build's measured l3 decode allocation ledger (5,100 slots, no MTP pins) is:

| Allocation | GiB | Relationship |
|---|---:|---|
| Expert slots | 89.337 | inside allocator A |
| Pinned dense/attention tensors | 10.466 reserved / 9.170 payload | inside A; 1.295 padding |
| KV, 64-token imported state | .00346 | inside A |
| Decode scratch | .03125 | inside A |
| Other tracked A allocations | .00314 | remainder, not a free pool |
| Allocator A / B totals | 99.841 / .000061 | totals; do not add again |

The exact byte ledger is `phase_e_accounting.json` under the raw result root.
At 5,500 slots the expert payload is **96.344 GiB**. In speculative mode 384
MTP expert slots are inside this cache, leaving 5,116 replaceable main-model
slots; they are not an extra allocation. Batch and route scratch add at most
128+8 MiB for one stream. Engram table scales are not resident by default.
Prefill transit and its runner are destroyed before decode; bootstrap host
expert store is released before the GPU store is built (`store_.reset()`).
Neither is a second resident copy that can be freed again.

The earlier 5,500-slot live OS snapshot reported **121.49 GiB MemTotal**,
**5.65 GiB MemAvailable**, driver VRAM/GTT **3.41/104.71 GiB** and process RSS
about **.286 GiB**. Driver counters include desktop clients; mapped UMA and
runtime allocations overlap. They must not be summed as independent memory
consumers. OS, desktop and page-cache usage is not a demonstrated runtime
reclaim. Even reclaiming all measured pinned padding and all decode/batch/route
scratch would remain below the plan's roughly 3 GB threshold, and their safe
repacking has not been validated.

Immediate-fill LRU replay of the same 27,399-token mixed trace gives hits
**.9155/.9179/.9202/.9224** at 5,500/5,600/5,700/5,800 slots. Each additional
100 slots costs **1.881 GB** for about **.22-.24 percentage points** of hit rate.
This excludes prefill and async loading. The simulator's legacy disk bandwidth
and emitted tok/s are not hardware measurements or a speed forecast. Therefore
Phase E ends with an accounting result and skips allocation changes; there is
no changed prefill/long-context allocation path to qualify. Sources:
`phase_e_memory_live.json`, `phase_e_capacity.json`, the final l3 runtime log,
and `Engine::feed_gpu` cleanup.

## Power and thermal measurement correction (2026-10-06)

The owner's current policy is **performance for benchmarks and the web**,
with AC connected and the existing 80/72°C thermal thresholds. The earlier
Phase A speed of 110.142 ms/token was measured in power-saver. It is a quality
receipt and a historical speed observation, not a performance-mode baseline.
The owner accepts A's 46/57 MMLU and .118 follow-up repeated-fourgram result as
the web baseline; these remain failed original quality gates. Phase C retains
its stricter quality and speed requirements and may become an explicit option
only after passing them.

In the interrupted performance check, mask completed seven turns. Attention,
MoE and tail cost **31.147/24.566/6.649 ms/token**, against historical mask
**31.853/24.896/6.734**. The specified GPU compute regression disappeared,
so the conditional source bisect is skipped. Off finished eight turns; mask's
eighth turn was interrupted. These unequal, cooling-affected totals cannot
establish a speed ranking.

The old supervisor paused 243 times and spent **994.673 of 1500.48 seconds**
cooling. Its mean sample interval was .592 seconds, including a slow power
profile CLI call. The 89°C peak occurred on the first trigger sample;
subsequent samples while stopped did not exceed 72°C. It also held pauses
until an NVMe sensor that never reached 80°C fell below 72°C. Finally, it
counted cooling against the experiment's wall budget and killed the last turn.
Sources: `p0_power_resume/thermal_analysis.json` and
`evidence_audit/audit_20261006T092538.json` under the raw result root.

The new `bench/thermal_guard.py` samples temperature every 50 ms, moves slow
profile reads to a monitored thread, and latches each sensor at 80°C until
that sensor reaches 72°C. It enforces separate active and wall safety budgets,
records AC, actual profile, clocks/DPM and absolute pause intervals, and cleans
up the owned process group on exit. Thirteen CPU cases passed, including a
real ordinary-process stop/resume test. The thermal threshold is a trigger,
not a claim that already submitted GPU work can be instantly preempted.

New benchmarks retain raw wall time and also subtract the intersection of
cooling intervals with each decode interval. The current anchor is the engine's
`decode_finished_unix`, recorded at the end of `decode_ms` before reheat or KV
checkpoint work; the interval starts at that end minus `decode_ms`. Phase C r4
uses the frozen executable SHA-256 `b07f2480c8464681...`, built from `de42061`,
with all 52 recorded shaders. Host `done` receipt anchors in older logs remain
legacy estimates. Prefill cooling is excluded from the decode adjustment;
per-stage timers retain their original values without cooling subtraction.
Active time estimates CPU suspension within the engine interval, and does not
measure GPU compute time. Suspending the CPU cannot cancel submitted GPU work.

## Completed performance comparison: Phase C NO-GO (2026-10-06)

**Weighted waits close NO-GO.** Both candidates fail the strict repetition gate
and fall short of the required 20% observed throughput gain over off. The four
arms each completed eight turns, with supervisor rc=0, in one engine (PID
3641973), session `default`, in off/mask/tau.20/tau.10 order. KV resets between
arms; the dynamic 5,500-slot expert cache carries between them. Speculation is
off, with zero MTP pins. The runtime confirms both read sources and all 48
mirror shards. AC and both power profiles remain performance throughout.

The following values weight each turn by its timed `decode_steps`; the 32
outputs contain 9,003 timed decode tokens. Gains are `off_ms / arm_ms - 1`.

| Arm | Raw ms/token | Active estimate ms/token | Raw throughput gain | Active estimate gain |
|---|---:|---:|---:|---:|
| off | 197.219643 | 120.249564 | — | — |
| mask | 179.840496 | 80.264076 | 9.66% | 49.82% |
| tau .20 | 168.678853 | 86.390295 | 16.92% | 39.19% |
| tau .10 | 179.090454 | 111.956500 | 10.12% | 7.41% |

Raw time ranks tau .20 fastest; the active estimate ranks mask fastest. This
protocol difference is retained, and the larger estimated gain does not replace
the observed speed gate. Mask attention/MoE/tail average
31.115/24.729/6.723 ms/token, close to the earlier 31.853/24.896/6.734. These
original per-op counters support closing the GPU compute regression and its
conditional source bisect; they are not independently cooling-adjusted stages.

All 32 outputs have longest same-token run one and no detected short-period
loop. Across them, repeated-fourgram fractions range from 0 to .178674 and
distinct-2 from .613181 to .971831. The stricter 1.5-times-off repetition gate
still fails: tau .20's third turn has .004016 repeated fourgrams versus off's
zero; tau .10's third/fourth turns have .003311/.006734 versus zero. The mask
control also fails on its fourth turn (.003534 versus zero). A zero off value
requires zero candidate repetition under the unchanged rule. Every turn's
four metrics remain in the raw report; no qualitative loop claim replaces this
arithmetic gate. Final cumulative reserve/submit/IO/failed-fill counters are
all zero, including startup, and neither source was dropped.

The supervisor reports 1,850.350 s wall, 1,109.604 s active and 740.746 s
cooling, with 4,263 pauses. The saved absolute pause intervals sum to 741.236 s;
their endpoint accounting differs by .490079 s from the supervisor accumulator.
Decode adjustments use interval intersections; the job totals above retain the
supervisor's definitions. All 27,849 thermal samples report AC=1 and performance.
GPU/NVMe peaks are 84.0/74.85°C. The 80°C threshold triggers CPU suspension and
does not preempt queued GPU work. All 11,139 paused samples report GPU busy
above zero (maximum 62%); this is a sampled counter fact with possible windowing
or lag, not proof of uninterrupted GPU execution throughout each pause. Every
decode window has complete thermal coverage and an engine end marker.

The earlier cold 5,100-slot, power-saver NLL screens remain historical evidence:
off .622784, tau .20 .626679 (1.006254 times off), tau .10 .623711 (1.001488
times off); prior Chinese64 screens also remain under their original protocol.
They do not qualify the new performance configuration's missing quality gates.
After the decided NO-GO, new performance Chinese64, the three matched 512-token
quality cases and MMLU57 are skipped by the stop rule. They are not recorded as
passes. No weighted-wait web option is added; the owner's dynamic-mask plus
speculation policy and the pending Phase D selection remain unchanged.

Sources under `bench/results/mask_quality/`:
`phase_c/performance_recovered_r4/{final_report.json,decision_receipt.json,check_results.json}`,
each arm's `turns.json`/status and `arms/serve.log`, the recorded thermal JSONL,
and the frozen binary/shader manifest. The report SHA-256 is
`fdd52c0c8469e9e390e77a120b046eb8e9d1b9628625d8180fba5f6753d06afd`.
Prior screens are `phase_c/nll_{off,tau20,tau10}.json` and
`phase_c/conversation/`; interrupted comparisons contribute no speed average.
