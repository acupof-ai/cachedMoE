# P3 — decode at long context (Track Q)

> The decode step made correct at 4K and 17K tokens of context, and proven
> against Track M's exports with tie-aware top-k comparisons. Companion to
> [p3_longctx.md](p3_longctx.md) (the reference data and why top-k ties are
> normal), [p3_prefill.md](p3_prefill.md) §8.3 (the bug report this answers),
> [p3_chat.md](p3_chat.md) §5 (Track P's fix of the NaN logits) and
> [p2_decode.md](p2_decode.md) (the engine).

Status: **2026-09-15**. All numbers from the real checkpoint on the Strix Halo
machine. Tests: `tests/test_decode_longctx.cpp` (`ctest -R decode_longctx`,
needs `DEEPMOE_LONGCTX_DIR=<repo>/traces/longctx` when built outside the repo).

---

## 0. One page

**What was wrong, and who fixed it.**

| symptom (p3_prefill.md §8.3) | root cause | fix |
|---|---|---|
| NaN / zero logits past ~1K tokens, even from the reference's own state | `Engine::prepare_ced` gave sparse_attn a list of `window + n_cmp` entries instead of `window + min(index_topk, n_cmp)`. Past 512 compressed rows it read entries the indexer never wrote; past 1,024 entries it ran off the per-head score row into the next head | **Track P**, 16cf319 (p3_chat.md §5) |
| "> 4,096 positions" refused | `kMaxIndexPositions` doubled as the context cap | Track P raised it to 16,384; **Q** made it the indexer's dispatch limit, 524,280 (§3) |
| "> 16,384 positions needs the candidate-block mask" refused | design §2.1's two-level top-k was not implemented in decode | **Q**: indexer.slang stages 6–8 (§2) |
| the 17K export refused / under-sized | the KV store was sized from `max_compressed()` (the widest per-step run), not the prefill record's `max_seq_len // ratio` buffers | **Q**: `DecodeState::max_prefill_rows()` (§3) |
| 8 GB of host RAM to load the 17K export | every step's `Lnn.cmp_kv` widened to fp32 for 38 layers | **Q**: decoded on first use (§3) |

Track L's three suspects all turned out to be innocent: the GPU radix top-k is
tie-aware equal to the reference on the reference's own inputs (§4.1); the
classic sparse_attn's 1,024-entry index table is never reached because the list
is at most 640 entries; the key and compressed planes past 1,024 rows are
bit-for-bit right (compressed KV cos 1.00000 over 17,011 rows, §4.2).

**Acceptance.**

| | 4K (N = 4,133) | 17K (N = 17,010) |
|---|---|---|
| (a) indexer kernels on the reference's inputs, tie-aware | 24/24 (layer, step) pairs | 24/24, candidate blocks 8/8 steps |
| (a) one step from the export state: top-1 / margin (ref) | 416 ✓ 10.90 (10.78) | 416 ✓ 11.38 (11.38) |
| (a) per-layer top-k, kernel tie-aware on its own scores | 8/8 layers, every step | 8/8 layers, every step |
| (b) 8 steps teacher-forced | **8/8** | **8/8** |
| (b) 8 steps free-running — `kestrel-4471-amber"` | **8/8** | **8/8** |
| (c) from Track L's GPU prefill (oracle mode), free-running | **8/8** | **8/8** |
| (d) 64-token L3 from the export state | 8/8 teacher-forced, 8/8 free-running | |

**Speed** (quiet machine, all experts resident, GPU timestamps): attention is
36.9 ms a token at 64 tokens, **42.2 ms at 4K and 46.4 ms at 17K**. The
growth is the indexer: the eight index layers take 7.9 ms without it and 10.3 /
14.1 ms with it (§6). KV read per step, in the model's own
storage formats: **10.1 MB at 4K, 15.8 MB at 17K** — exactly Track M's figures.

---

## 1. Bisection

Track M's exports make the decode step checkable stage by stage. Three kinds of
evidence, in increasing distance from the kernel:

1. **Kernels on the reference's inputs** (`decode_longctx.indexer_vs_reference`,
   GPU only, no weights). Rebuild each probe indexer's key cache (prefill
   `index_k` plus the rows each step published), take the reference's
   `index_q` and `index_weights`, run `indexer.score`, `indexer.topk` and the
   candidate-block stages, and compare with a host recomputation that does the
   kernel's bf16 arithmetic element for element.
2. **The engine from the export state** (`decode_longctx.engine_vs_reference`):
   every probe layer (0, 2, 13, 14, 20, 39) of every step against the full L2
   export, and every index layer's list against the reference's.
3. **Tokens**: teacher-forced and free-running.

Before Track P's fix (1) and (2) would have shown the failure immediately: the
list length is checked by `record_attention` now (§3).

---

## 2. Design §2.1's candidate blocks in decode

`model.py`: the candidate source (layer 20) scores all `n` positions, takes the
max per block of 8, pins the block holding the newest position to +inf, keeps
the best 2,048 blocks; layers 24, 28, 32 and 36 mask their own scores to -inf
outside them before their top-512. Below `2,048 × 8 = 16,384` positions every
block is kept, so the mask only exists past 16,384.

Implemented in `gpu/shaders/indexer.slang` as three more stages of the same
.spv, dispatched by `DecodeLayer::record_ced` — the Engine sets nothing:

| stage | what | dispatch |
|---|---|---|
| 6 `IdxBlockKeys` | one radix key per block (the max of the positions' keys = the key of the max); the newest block gets `0xFFFFFFFF`, above +inf's key | `ceil(n_blocks / 256)` workgroups, one thread a block |
| 7 `IdxBlockSelect` | the 2,048 best blocks as keep flags; stage 5's 4-pass radix select, now `radix_threshold`, shared | 1 workgroup, chunked |
| 8 `IdxApplyCand` | a consumer's scores outside kept blocks → -inf (`0xFF800000`) | 1 workgroup, chunked |

The flags live in the decode scratch (`idx_cand`), written by layer 20 and read
by 24–36 in the same token. `record_ced` records which position and `n_cmp` the
flags were built for and refuses a consumer whose step differs — a
layer-at-a-time caller that skips layer 20 gets an error, not a stale mask.

**Ties.** At the 2,048th block score 4–9 blocks tie on every 17K step. Stage 7
keeps the lowest block indices; torch keeps an arbitrary subset. Over 8 steps
× 4 consumer layers × 512 picks, exactly one reference pick (step 1) lies in a
tied block the kernel dropped (§4.1). That is a legitimate tie difference, not
an error, and it cannot be removed.

Cost: stages 6–8 walk `n` scores a few times; at 17K they are inside the
≈ 6 ms that the indexer and compressor add to the eight index layers (§6).
Scoring only inside the candidate blocks for 24–36 (p3_longctx.md §5.1, exact)
is not done: the masked positions are still scored, as in the reference.

---

## 3. Limits, sizing and hardening

* **`kMaxIndexPositions` = 65,535 × 8 = 524,280** (`runtime/decode_layer.h`).
  It is what one `indexer.score` dispatch can cover, not a buffer size anyone
  picked; the score plane (2 MB) and the candidate planes (0.5 MB) are sized
  for it once. The working context limit is now the KV store's memory, which
  the caller sizes: 220 MB at 4K, 882 MB at 17K (all 40 layers carry a
  compressed and an index-key plane, §7).
* **KV store sizing from the prefill buffers.** `Engine::load_decode_state`
  sizes `max_context` from `max(max_compressed(), max_prefill_rows()) + 64`.
* **Lazy per-step `cmp_kv`.** `DecodeState` records offsets for the per-step
  compressed planes and decodes one on first `tensor()`; the produced-CED path
  never touches them.
* **Hardening against Track P's class of bug** (`runtime/decode_layer.cpp`):
  1. `record_attention` refuses any list longer than `window + index_topk` or
     the 1,024-entry score stride, and — whenever the layer's `n_cmp` is known —
     any list whose length is not exactly `window + min(index_topk, n_cmp)`.
  2. Before the indexer's top-k is recorded, the compressed half of the list is
     poisoned with -1. `DecodeLayer::verify_after_attention`, called by the token
     loop right after the wait that follows each layer's attention (and by
     `run_attention`), checks that all `min(index_topk, n_cmp)` entries were
     written, strictly increasing and in `[window, window + n_cmp)`. Cost: 512
     host reads on 8 layers a token.
  3. A candidate consumer refuses a mask built for another step (§2); an
     `n_cmp` beyond the dispatch limit is refused with the limit in the message.

---

## 4. Validation

### 4.1 Kernels on the reference's inputs (`indexer_vs_reference`)

Probe index layers 2, 14 and 20, all 8 steps of both exports (keys from layer
20's cache on the incomplete ratio-2 steps, as the reference does):

| | 4K | 17K |
|---|---|---|
| scores bit-equal to the host recomputation | 24/24 pairs, every position | 24/24 |
| reference's own top-k tie-consistent under those scores | 24/24 | 24/24 |
| **kernel top-k tie-aware equal to the reference** | **24/24** | **24/24** |
| … identical as sets | 5/24 | 9/24 |
| pairs with a tie at the 512th score | 21/24 | 22/24 |
| candidate blocks, kernel tie-aware equal to `select_candidate_blocks` | — | **8/8 steps** (4–9 blocks tied at the 2,048th) |
| reference picks at layers 24–36 outside the host mask | — | **0 of 16,384** |

The first two rows say the host emulation *is* the reference's arithmetic, so
the third is a real equality, not a tolerance. The last row checks the mask's
semantics (block max, pinned newest block, 2,048 blocks) against the
reference's own layers 24–36, which have no exported index queries.

### 4.2 The engine from the export state (`engine_vs_reference`)

Teacher-forced, all 8 steps probed (full traces L2). "Tie-aware on its own
scores" = the kernel's selection against a host top-k of the kernel's own score
plane (captured per layer): it must hold exactly, and does. Against the
reference the scores are not bit-equal (the stream is at cos 0.95–0.998), so the
comparison is overlap plus how far below our 512th score a missed reference
pick sits.

| | 4K | 17K |
|---|---|---|
| index layers tie-aware exact on their own scores | 64/64 (8 × 8 steps) | 64/64 |
| top-k in common with the reference, mean over layers × steps | 90.0–95.3% per step | 78.6–93.0% per step |
| worst layer | L32 426/512 (step 3) | L28 262/512 (step 1) |
| worst missed reference pick below our threshold (relative) | 0.71 | 1.45 |
| top-1 | 8/8 | 8/8 |

Per-layer overlaps are printed per step; at 17K step 1 (the first step that
completes a ratio-2 group, where layers 2–19 switch to their own keys) layers
24–36 drop to 262–363/512 because layer 20's input is at cos 0.955 there.

Worst stage cosine over the six probe layers, per stage, over the 8 steps:

| stage | 4K | 17K |
|---|---|---|
| `attn_norm_out` | 0.9704 (L20, step 7) | 0.9550 (L20, step 1) |
| `q` / `kv` | 0.9865 / 0.9750 | 0.9541 / 0.9275 |
| `attn_out` (post inverse RoPE) | 0.9561 | 0.9501 |
| `ffn_norm_out` / `moe_out` | 0.9443 / 0.9037 | 0.9454 / 0.9061 |
| window KV the layer read | ≥ 0.9994 | ≥ 0.9990 |
| compressed plane the layer read (all rows) | ≥ 0.99997 | ≥ 0.99999 |

Step 0 at 4K, one step from the reference's exact state, is at ≥ 0.9969 on
every stage of every probe layer except `index_q` (0.9905 at L14, 0.9958 at L2)
and the newest compressed row our step wrote (0.9940).

**Bisection control.** The same probed steps run again with the reference's
compressed KV and top-k lists *loaded* per step (`set_produce_ced(false)`):
at 17K step 1 layer 20's `attn_norm` input is 0.9557 loaded against 0.9550
produced, i.e. the worst dip is not the indexer's, the compressor's or the
mask's. At step 0 loaded is 0.9941 against 0.9878 produced (4K step 1: 0.9851
against 0.9883 worst), so near-tie selection differences account for some of
the drift on some steps and none on others. The rest is the compounding of MoE
routing near-ties through depth (gate top-6 agreement 3–6 of 6 at the probe
layers in both modes), the effect p2_decode.md §4 measures at 64 tokens.
The KV each layer reads is tight throughout: window KV ≥ 0.999, the compressed
plane cos 1.00000 over all rows (the rows our decode steps wrote: 0.98–0.9993).

### 4.3 From Track L's GPU prefill (`gpu_prefill.longctx`)

`DEEPMOE_PF_LONGCTX=traces/longctx/<name> DEEPMOE_PF_DECODE=free`, oracle mode:

| | 4K | 17K |
|---|---|---|
| prefill (validation run) | 175 s | 596 s |
| first token 77, margin ours / ref | 8.31 / 8.86 | 9.92 / 9.48 |
| handoff worst window / compressed / index keys | 0.960 / 0.968 / 0.977 | 0.948 / 0.967 / 0.978 |
| **free-running from OUR state** | **8/8**, margins 1.23–11.94 (ref 1.30–11.66) | **8/8**, margins 1.72–12.40 (ref 1.66–11.65) |

p3_prefill.md §9.3 "decode from these states could not be run" is closed: the
salt comes back from our own prefill at both lengths.

### 4.4 64 tokens (L3), regression

`decode.forty_layers_against_the_l3_oracle`: from the export state 8/8
teacher-forced and 8/8 free-running; from the slow prefill 7/8 and 6/8 — the
same as before this track.

---

## 5. Tokens

Both exports, from the export state, teacher-forced and free-running, and from
our GPU prefill free-running, produce the reference's greedy continuation
`416 4419 15 24391 19 15 13749 2701` = `est rel - 447 1 - amber "\n` after the
prefill's `k`. Margins (ours / reference), teacher-forced from the export:

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 4K | 10.90 / 10.78 | 12.18 / 11.66 | 9.60 / 9.51 | 9.10 / 8.83 | 10.83 / 10.77 | 8.45 / 8.41 | 10.14 / 9.65 | 0.98 / 1.30 |
| 17K | 11.38 / 11.38 | 11.43 / 11.65 | 10.73 / 10.59 | 10.13 / 10.10 | 10.91 / 11.06 | 8.59 / 9.12 | 11.34 / 11.33 | 1.69 / 1.66 |

Step 7 (`"\n` against `"`) is the only low-margin step, as in the reference.

---

## 6. Time and bytes per step

`deepmoe run --state <dir> --warm 3 --steps 8 --cache-gb 24`, one process after
another on an otherwise idle GPU; the first step after the warm-up has every
expert resident, so its wall time is the compute floor. Attention is the sum of
the forty layers' GPU-timestamped attention spans, identical on every step.

| context | wall, all experts resident | attention | MoE GPU | tail + engram + other |
|---:|---:|---:|---:|---:|
| 64 | 82.9 ms | 36.9 ms | 29.2 ms | 16.2 ms |
| 4,133 | 91.1 ms | **42.2 ms** | 29.5 ms | 18.2 ms |
| 17,010 | 94.2 ms | **46.4 ms** | 29.1 ms | 17.6 ms |

What the indexer costs, from `engine_vs_reference` (GPU timestamps of the eight
index layers' attention spans, produced vs the loaded-CED control, which runs
neither compressor nor indexer):

| context | 8 index layers, produced | same, loaded control | indexer + compressor + mask | all 40 layers |
|---:|---:|---:|---:|---:|
| 4,133 | 10.3 ms | 7.9 ms | **2.4 ms** | 43.0 ms |
| 17,010 | 14.1 ms | 7.9 ms | **6.2 ms** | 46.6 ms |

So attention grows by ≈ 0.3 ms per 1K tokens of context, all of it index
scoring (8 layers × T keys) — linear, as p3_longctx.md §5.1 predicts; the
window and the 512 compressed picks cost the same at every length. Extrapolated
to 64K: ≈ 22 ms of indexer on ≈ 60 ms of attention, which is where
candidate-restricted scoring for layers 24–36 starts to pay.

KV bytes read per step (`engine_vs_reference` (d)), in the model's formats as
p3_longctx.md §5.1 counts them — window 40 × 128 × 528 B, compressed rows
attended 38 × 512 × 288 B, index keys scored 8 × T × 68 B:

| | window | compressed | index keys | total | Track M | this store (bf16) |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 2.70 MB | 5.60 MB | 1.83 MB | **10.1 MB** | 10.1 MB | 29.5 MB |
| 17K | 2.70 MB | 5.60 MB | 7.52 MB | **15.8 MB** | 15.8 MB | 50.9 MB |

---

## 7. Done / not done

**Done**

* Design §2.1's candidate-block mask in decode, on the GPU, validated
  tie-aware against the reference at 17K (§2, §4.1).
* The >16,384 refusal and the 4,096/16,384 context cap removed; the KV store
  sized from the prefill buffers; the 17K export loads in bounded memory (§3).
* Hardening: list-length and stride checks, poisoned-and-verified top-k lists,
  mask provenance (§3).
* `tests/test_decode_longctx.cpp`; acceptance (a)–(d) at 4K and 17K (§0).
* `Engine::reseed_decode_state()` (a minimal engine edit, below).

**Engine edits (Track P's files, kept minimal)** — `runtime/engine.cpp`:
the `Unimplemented` refusal in `prepare_ced` removed; `load_decode_state` sizes
from `max_prefill_rows()`; one `verify_after_attention` call after
`cmd_wait()` in `run_layer`; `reseed_decode_state()` added (engine.h too).

**Not done**

* **Per-source planes.** `KvStore` gives all 40 layers a compressed plane and an
  index-key plane though only 4 write them: 882 MB at 17K, ≈ 3.3 GB at 64K where
  ≈ 0.35 GB would do. The loaded-CED path seeds every layer's own plane, so the
  change needs the Engine to redirect reuse layers in that mode too.
* Candidate-restricted scoring for layers 24–36 (exact; saves most of the index
  compute at 64K, p3_longctx.md §5.1).
* Past 524,280 positions the score stage needs a 2-D dispatch.
* The tiled `AttnScoreT/AttnPvT` kernels are not adopted in decode: the list is
  ≤ 640 entries, which the classic kernel handles; they matter only if the list
  grows.
* The stream drift at depth (§4.2) is MoE routing near-ties, not long-context
  arithmetic; nothing here addresses it.
