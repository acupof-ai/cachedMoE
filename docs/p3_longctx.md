# P3 Track M — long-context correctness data

> Reference data for the parts of design §7.4 / §7.5 / §11 that a 64-token prompt
> cannot exercise: a non-degenerate indexer top-k, a wrapped window ring, ratio-2
> pooling with a reference output, and layer 20's candidate-block pre-filter.
> Tool: `tools/oracle_longctx.py`. Small, committed: `tests/data/longctx/` (9.0 MB).
> Large, regenerable: `traces/longctx/` (1.3 GB, gitignored).

Status: **2026-09-14**, both prompts run to completion (prefill + 8 greedy steps).
Numbers: `tests/data/longctx/stats.json`; raw per-(step, layer) records:
`traces/longctx/<name>/stats_raw.json`.

---

## 0. Summary

* **Two prompts, 4,133 and 17,010 tokens**, mixed EN / ZH / code from local files.
  Each defines `LEDGER_SALT = "kestrel-4471-amber"` in its first 30 tokens and ends with
  `assert LEDGER_SALT == "`. **Both reference runs retrieve it exactly**:
  `k est rel - 447 1 - amber "\n`, 9/9 tokens, at greedy margins 8.4–11.7 for the
  eight salt tokens and 1.3 / 1.7 for the final `"\n` token.
* **The indexer is non-degenerate now**: 512 of 2,067–17,018 positions per layer. Its
  replay from the captured q / weights / keys reproduces the reference's top-k on
  **all 128 (layer, step) pairs of both runs**, 512/512.
* **Three things the reference does that contradict design §7.4 as written** (§5):
  1. Per decode step, the eight index layers read **7.5 MB of index keys and do 453 MFLOP at
     17K**. Extrapolated to 64K that is **29 MB and 1.74 GFLOP**, not §7.4's
     "2 MB / 134 MFLOP": the design figure counts one ratio-2 source, and five
     ratio-1 layers (20, 24, 28, 32, 36) each score all 64K keys.
  2. At every position where the ratio-2 group does not complete, **layers 2 / 8 / 14 score
     against layer 20's key cache** (`shared_attn.index_k` is never reset). §7.4
     mentions this mechanism; the data gives its size. At 17K that top-k shares only
     **8% (23–72 of 512)** with the top-k the layer's own keys would give. It also
     roughly halves those layers' compressed attention mass (layer 8: 0.71 → 0.37), and
     drops their mass on the salt definition to ~0.
  3. **Exact bf16 ties at the top-k boundary are the normal case.** In 43 of 48
     probe (layer, step) pairs the 512th score is shared by 2–29 positions. In 35 of them
     only some of the tied positions fit, and `torch.topk` keeps a subset that is neither
     the lowest-index nor the highest-index one in 29 of those 35. A radix select cannot be required to match the
     reference's top-k *set*; the comparison has to be tie-aware (§4.3).
* **Stability**: consecutive-step top-k overlap is **53–61%** on the ratio-1 index layers.
  On the ratio-2 layers it is **6–10% at 17K** (23–28% at 4K), because the key source
  alternates every step. Comparing steps two apart, i.e. with the same key source,
  gives 30–72% (53–86% at 4K).
* **Where attention goes at 17K**: 68% of selected compressed rows are ≥ 8K tokens
  back. They carry 41% of attention mass, and the window carries 54%. Sink mass is
  **0.28%** averaged over layers 2–39 (max head 1.6%), but **6.6% / 25%** at the
  window-only layers 0 and 1. The top 64 of the 512 selected rows carry 74% of the
  compressed mass, and the top 16 carry 57%.
* **Routing is unchanged by context length**: adjacent-token Jaccard is 0.201 at 17K and
  0.204 at 4K, against Q3's 0.234. Tokens before and after position 2,048 give 0.210
  vs 0.199. `union_frac[5]` is 0.674 vs Q3's 0.631.
* **KV bytes read per decode step**: 10.1 MB at 4K and 15.8 MB at 17K (window 2.7 +
  compressed 5.6 + index keys 1.8 / 7.5). Extrapolated to 64K it is ≈ 37 MB, **0.3% of
  §2.3's 13 GB/token**. KV reads do not matter for NVMe or bandwidth. Index-key *compute*
  is the part that grows (§5.1).

---

## 1. Why a 64-token prompt was not enough

| mechanism | at n = 64 (L2/L3) | here |
|---|---|---|
| indexer top-k | `min(index_topk = 512, n/ratio)` = all 32 / 65 positions; selection never tested | 512 of 2,067–2,070 / 4,134–4,141 (4K) and 8,505–8,509 / 17,011–17,018 (17K) |
| window ring | never wraps; slot p = token p | wrapped at a non-zero cutoff (4133 % 128 = 37, 17010 % 128 = 114); slot = `p % 128` is what is tested |
| ratio-2 pooling | pos 64 does not complete a group, so `Compressor.forward` returns None and there is no reference | 4K: N odd, so prefill leaves token 4132 in `kv_state` slot 0 and **step 0 (pos 4133)** pools it with the decode token into row 2066. 17K: N even, so **step 1 (pos 17011)** pools row 8505. Both are exported with the carried state before and after |
| layer-20 candidate blocks (`candidate_topk_blocks 2048 × candidate_block_size 8 = 16,384`) | all blocks kept | 17K: 2,127 blocks at decode; **632–640 positions (79–80 blocks) masked** for layers 24–36; prompt rows > 16,384 are pruned in prefill too |
| RoPE / YaRN | position 64 barely turns the low-frequency pairs | `q_pre_rope`/`q` and `attn_out`/`attn_out_irope` at pos 4134 and 17011 |

`config.json`: `window_size 128`; `compress_ratios` 0 (layers 0–1), 2 (2–19), 1 (20–39);
`kv_source_layers [2, 8, 14, 20]`; `index_source_layers [2, 8, 14, 20, 24, 28, 32, 36]`;
`index_topk 512`; `original_seq_len 65536`, `rope_factor 16`, `compress_rope_theta 160000`.
**V4.1-Flash has no ratio-4 layer.** Ratio 2 is the only pooling in the model, so "a
ratio-2 and a ratio-4 completion" reduces to the two consecutive steps above.

---

## 2. Prompts

`tests/data/longctx/prompts.json` holds, for each prompt, the text, the token ids, the
spans, the sources and the per-kind token counts. **The ids are the truth**: the text is
the ids decoded.

| | ctx4k | ctx16k |
|---|---:|---:|
| tokens (BOS included) | 4,133 | 17,010 |
| `LEDGER_SALT` literal (8 tokens) | 22–30 | 22–30 |
| `ledger_audit` usage block | 3,336 | 8,678 |
| filler EN / ZH / code | 1,109 / 1,071 / 1,723 | 5,116 / 5,811 / 5,853 |
| sources | 5 | 13 |

* Layout: BOS (id 0, what `encoding.py` prepends), then a `ledger.py` head (the salt, a
  function using it, and a Chinese glossary line). Then ~450-token segments of contiguous
  paragraphs rotating EN → ZH → code across documents, a `ledger.py, continued` block at
  the midpoint, and the tail `assert LEDGER_SALT == "`.
* Sources (no network): the checkpoint's `README.md` and `inference/{model,engram,kernel}.py`,
  and CPython's `argparse` / `json.encoder` / `dataclasses` / `textwrap`. Repo docs are read
  through `git show e41e54c:docs/...`, so concurrent edits to `docs/` cannot change a rebuild.
* 17,010 is deliberately **above** 16,384 so that the candidate pre-filter drops blocks. Neither
  N is a multiple of 128, and the two N have opposite parity (§1).

---

## 3. Runs

| | ctx4k | ctx16k |
|---|---|---|
| wall clock | 2026-09-14 **14:31:27 → 14:51:33** | 2026-09-14 **14:51:50 → 15:21:49** |
| prefill (40 layers) | 959 s (~24 s/layer) | 1,579 s (~39 s/layer; 42–51 s on 20–24) |
| decode, 8 steps | 28–34 s/step | 26–29 s/step |
| estimate before launch | 22 min | 34 min |
| other activity at start | `deepmoe_tests.exe` (GPU) | `deepmoe_tests.exe` (GPU) |
| waited for | Track L's `oracle_prefill.py` (181 s) | — |
| process peak | ~6 GiB | ~9 GiB (available 50.5 → 41.5 GiB) |

**Tracks measuring GPU timings between 14:31 and 15:22 on 2026-09-14 should label those
numbers**: a 16-thread CPU oracle was running, reading ~300 GB from NVMe per prefill.

Commit-limit hazard: the reference `ParallelEngramEmbedding` constructor does an untouched
`torch.empty(384M, 256, fp8)` (~98 GB of commit) at layers 1 and 14 on every pass. The tool
replaces it with a zero-parameter stub (`_NoEngramTable`) before any `Block` is built.
`dsref.make_block` swaps in its row reader immediately, so the stub is never read. The tool
also waits for ≥ 20 GiB commit headroom before starting.

### 3.1 What is and is not `model.py` as is

Everything is `inference/model.py` behind `tools/dsref.py`'s shims, with one addition:
`--index-chunk 1024`. The prefill index score
`einsum("bshd,btd->bsht") → relu_ → * weights → sum(dim=2)` is 17 GB of bf16 per copy
at 17K, and the statement holds two copies. `model.torch` is swapped for a proxy whose
`einsum` returns a lazy object for exactly that equation. The lazy object applies the same
three torch ops 1,024 query rows at a time, and raises on anything else. **The 4K run computed
both forms and asserted bit identity on all eight index layers: 8/8 identical**
(`stats.json → runs.ctx4k.index_chunk_checks`). The 17K run used the chunked form only.

Resumable: `ckpt_prefill.pt` every 4 layers, `ckpt_decode.pt` after every step. Rerun the same
command to resume. Once `index.json` exists, `traces/longctx/*/ckpt_decode.pt` (65 MB) can be
deleted.

---

## 4. Export schema, and how the engine should use it

### 4.1 `traces/longctx/<name>/` — the full export, L3 container

`index.json` has the same layout as `tests/data/l3/index.json`:
`prompt_ids`, `prefill_len = decode_pos = N`, `steps_exported = 8`, `greedy_tokens` (9),
`config`, `engram` (the hash constants, with `engram_token_map.bin` alongside), and
`steps[]` = `prefill`, `step00` … `step07`. **`runtime::DecodeState::load("traces/longctx/ctx16k")`
reads it unchanged.** Extensions over L3 are names only: `spans`, `expected_continuation`,
`run_log`, and the tensors marked *new* below.

`l3_prefill.bin` — the decode handoff state:

| tensor | dtype | shape (4K / 17K) | meaning |
|---|---|---|---|
| `Lnn.win_kv` | bf16 | [128, 512] | the ring **as the buffer holds it**: slot `p % 128` holds token p, for p in N−128..N−1 |
| `Lnn.cmp_cache` (2, 8, 14) | bf16 | [2074, 512] / [8513, 512] | whole `max_seq_len // 2` buffer; rows `0..N//2−1` filled (2066 / 8505) |
| `L20.cmp_cache` | bf16 | [4149, 512] / [17026, 512] | rows `0..N−1` filled |
| `Lnn.index_k` (2, 8, 14, 20) | bf16 | [rows, 128] | same row counts as `cmp_cache` |
| `Lnn.cmp_state_kv` / `_score` (2, 8, 14) | f32 | [2, 512] | 4K: slot 0 = token 4132, slot 1 score −inf. 17K: both −inf |
| `Lnn.topk_idxs_last` *new* | i32 | [640] or [128] | the index row the last prompt position attended to. Window part: absolute positions. Compressed part: row + N (prefill offset) |
| `Lnn.gate_bias`, `top_ids`, `top_logits`, `logit_stats`, `collapse_in`, `pre_mix_final`, `norm_out`, `argmax` | | | as L3 |

`l3_stepNN.bin` (44 MB each at 17K):

| tensor | shape | meaning |
|---|---|---|
| `Lnn.cmp_kv` | [T, 512] | the compressed rows this layer's `sparse_attn` saw, T = (pos+1)//ratio; de-duplicated by offset across reuse layers |
| `Lnn.topk_idxs` | [640] ([128] at layers 0–1) | window slots first (oldest first, `get_window_topk_idxs`), then compressed row **+ 128** in ascending order |
| `Lnn.cmp_row_new` *new* | [1, 512] | the row a kv source wrote this step (ratio 1 every step; ratio 2 when `(pos+1) % 2 == 0`) |
| `Lnn.ffn_in`, `gate_scores`, `gate_ids`, `gate_weights`, logits, `in_token`, `argmax` | | as L3 |

`l2/index.json` + `l2_Lnn_decode<pos>.bin` hold oracle.py's L2 per-stage set at layers 0, 2, 13,
14, 20, 39 for **every** step (win_kv and cmp_kv included). Names added on kv sources:
`cmp_state_{kv,score}_{before,after}` (f32 [2, 512]), `cmp_wkv_out` / `cmp_wgate_out` (f32 [512]),
`cmp_pooled` (bf16 [512], the `compressor.norm` input, completing steps only). Added on index
sources: `index_keys_from_layer` (i32 [1]). `routing.npz` holds `Lnn` → uint16 [N, 6], the prefill
top-6 of every token.

### 4.2 `tests/data/longctx/` — the committed subset (9.0 MB)

| path | size | contents |
|---|---:|---|
| `prompts.json` | 291 KB | §2 |
| `stats.json` | 185 KB | every number in this document |
| `<name>/index.json` + `l3s_prefill.bin`, `l3s_stepNN.bin` | 1.0 / 1.7 MB | the L3 container **without any KV buffer**. Prefill: logits, `Lnn.topk_idxs_last`, `Lnn.gate_ids_last`, ratio-2 `cmp_state_*`. Steps: every layer's `topk_idxs`, `gate_ids`, `gate_weights`, logits, and for kv sources 2/8/14/20 the pooling set (`cmp_state_*_before/after`, `cmp_wkv_out`, `cmp_wgate_out`, `cmp_pooled`, `latent_pre_rope`, `cmp_row_new`, `cmp_row_index`, `index_k_row_new`) |
| `<name>/l2/` | 3.1 / 4.4 MB | L2 stage tensors at the six probe layers for **steps 0 and 1 only**. Dropped: the KV buffers and the FFN half (`win_kv`, `cmp_kv`, the `[4, 5120]` stream tensors, `wo_a/wo_b`, `ffn_*`, `moe_*`, `gate_scores`, `gate_bias`). `q_pre_rope` and `attn_out_irope` are kept on the completing step only |

The small prefill record **cannot seed a KvStore**; load `traces/` for that. The engram tables
are identical to `tests/data/l3`'s and are not duplicated.

### 4.3 How Track I should compare

1. **Seed**: `DecodeState::load(traces/longctx/<name>)`, then `seed_prefill`. Size `KvStoreConfig`
   from the **prefill** record's `cmp_cache` rows (2074 / 4149 and 8513 / 17026), not from
   `max_compressed()`. The latter is T (≤ 4141 / 17018), which is smaller than the buffer
   `seed_compressed` is handed. The window ring is already in `p % 128` layout, so seed it
   verbatim. The first decode write goes to slot `N % 128` = 37 / 114, overwriting token
   N − 128.
2. **Teacher-force** `greedy_tokens[s]` at `N + s` with `produce_ced` on. The engine must
   reproduce `pub_index_k_` switching: at pos where `(pos+1) % 2 != 0`, layers 2/8/14 score
   `layer 20's cache[:T]`. Ground truth per (layer, step):
   `traces/longctx/<name>/stats_raw.json → index[].keys_from_layer`, and in L2 `index_keys_from_layer`.
   For 4K that means steps 1, 3, 5, 7; for 17K, steps 0, 2, 4, 6.
3. **Pooling** (closes design §12's "ratio-2 pooling unverified"): `tests/data/longctx/ctx4k/l3s_step00.bin`
   `L02/L08/L14.{cmp_state_kv_before, cmp_state_score_before, cmp_wkv_out, cmp_wgate_out}`
   → `cmp_pooled` → `latent_pre_rope` → `cmp_row_new` (row `cmp_row_index` = 2066), plus
   `index_k_row_new`. The 17K equivalent is `ctx16k/l3s_step01.bin`, row 8505. The probe
   layers 2 and 14 have the full L2 chain (`latent_pre_quant`, `latent`, `latent_fp4`,
   `latent_scale_e4m3`, `index_k_*`) in `<name>/l2/l2_L02_decode4133.bin` / `…17011.bin`.
4. **Top-k: compare tie-aware, not as sets.** Recompute the score for the engine's selection
   from the exported `index_q` / `index_weights` / keys. Accept if (a) every position whose
   score exceeds the reference's 512th score is selected, and (b) the remainder are drawn
   from positions whose bf16 score **equals** it. `stats.json → runs.*.ties` lists the
   threshold and tie count per probe (layer, step); `tie_analysis` in the tool is the
   reference implementation of that check. An exact set match can fail on the 35 of 48
   probe (layer, step) pairs with a partial tie through no arithmetic error.
5. **Attention**: to test `sparse_attn` in isolation, seed the reference `topk_idxs` for that
   step (it removes the tie ambiguity above) and compare `attn_out` at the probe layers. Near a
   tie, the engine's own top-k legitimately gives a different `attn_out`.
6. **Tokens**: the reference margins are large (8.4–11.7) on steps 0–6, so a mismatch there is a
   bug, not a near-tie. Step 7 (`"\n` vs top-2 id 4 `"` at 4K, margin 1.30; 1.66 at 17K) is
   the only low-margin step.

---

## 5. Statistics

### 5.1 Index-key bytes and compute per decode step

| | 4K (pos 4133) | 17K (pos 17010) | 64K (extrapolated) | design |
|---|---:|---:|---:|---|
| index keys scored, 8 layers | 26,871 | 110,570 | 425,984 | — |
| index-key bytes read (× 68 B) | 1.83 MB | 7.52 MB | **29.0 MB** | §7.4: "32K × 64 B = 2 MB"; §11.3's 28.97 MB is the *storage* of the same keys |
| of which distinct (layer 20's keys are re-read by 24–36) | 0.70 MB | 2.89 MB | 11.1 MB | |
| scoring FLOP (× 32 × 128) | 110 M | **453 M** | **1.74 G** | §7.4: 134 MFLOP |
| window KV (40 × 128 × 528 B) | 2.70 MB | 2.70 MB | 2.70 MB | |
| compressed rows attended (38 × 512 × 288 B) | 5.60 MB | 5.60 MB | 5.60 MB | §7.5: 640 × 512 B / layer |
| of which distinct rows (per kv source) | 0.77 MB (2,670 rows) | 0.82 MB (2,855 rows) | — | |
| **total KV read** | **10.1 MB** | **15.8 MB** | **≈ 37 MB** | §2.3: 13 GB/token |

Put differently, every decode step reads all of §11.3's index-key storage: the per-step
read equals the table's total.

The design undercounts because of how §7.4 is built. It sizes "the indexer" from one ratio-2
source, but all eight index layers score every compressed position on every step. Five of the
eight are ratio 1, and 24–36 score all of layer 20's keys **before** applying the candidate
mask. One exact equivalent saves most of this at 64K: score only inside the candidate blocks
for 24–36. The masked positions are −inf and can never be selected. That takes 5 × 64K
positions down to 64K + 4 × 16,384. Ties among masked positions are impossible, so the §4.3
caveat is unaffected.

### 5.2 Top-k stability (can compressed reads be cached or prefetched?)

Mean overlap |A_s ∩ A_{s+1}| / 512 over 7 step pairs:

| layer | ratio | 4K | 17K | 17K, s vs s+2 | 17K union over 8 steps |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0.26 | **0.07** | 0.47–0.71 | 1,832 |
| 8 | 2 | 0.28 | **0.06** | 0.30–0.72 | 2,193 |
| 14 | 2 | 0.23 | **0.10** | 0.31–0.66 | 2,192 |
| 20 | 1 | 0.69 | 0.60 | 0.34–0.79 | 1,259 |
| 24 | 1 | 0.65 | 0.61 | | 1,303 |
| 28 | 1 | 0.49 | 0.56 | 0.20–0.73 | 1,778 |
| 32 | 1 | 0.45 | 0.55 | | 1,796 |
| 36 | 1 | 0.61 | 0.53 | | 1,620 |

On the ratio-1 layers about half the set survives a step, and ~150–400 rows are new. On the
ratio-2 layers the set is almost entirely replaced every step at 17K, because the key source
alternates (§5.3). Neither pattern makes a cache worth building: all of it is < 1 MB of
distinct rows a step, sitting in GPU memory (§11.3). **Nothing here argues for compressed-KV
prefetch**, and the ratio-2 behaviour argues against any "last step's set" heuristic.

### 5.3 Ratio-2 layers: own keys vs layer 20's keys

Top-k from the layer's own keys vs the reference's (layer 20's keys), on the steps where they differ:
**27% overlap at 4K (113–154 of 512), 8% at 17K (23–72 of 512)**.

Layers 2–19, split by the key source their index layer used that step (4 steps each):

| 17K | own keys | layer 20's keys |
|---|---:|---:|
| L2 compressed mass | 0.67 | 0.50 |
| L8 compressed mass | 0.71 | 0.37 |
| L14 compressed mass | 0.65 | 0.41 |
| L8 mass on the salt definition | 0.061 | 0.000 |
| L14 salt rows selected (of 4 covering the literal) | 3.0 | 0.25 |

Layer 20's key j is a *token* (ratio 1), but layer 2's cache row j is the *group* of tokens
2j, 2j+1. A stale selection therefore points at the first half of the context with unrelated
content. The layer falls back on its window: L19's window mass rises from 0.52 to 0.71 at
4K and from 0.53 to 0.70 at 17K. The output tokens are unaffected here: the late retrieval
layers (§5.4) carry the salt. It is still the reference's behaviour, and the engine has to
copy it to agree with L3.

### 5.4 Attention mass, sinks, distance

Decode steps, averaged over 8 steps and over layers 2–39 unless named:

| | 4K | 17K |
|---|---:|---:|
| sink mass, mean (max head) | 0.27% (1.4%) | 0.28% (1.6%) |
| sink mass layer 0 / layer 1 (window only) | 6.6% / 25% (max head 40% / 65%) | 6.6% / 25% (40% / 67%) |
| window mass | 0.49 | 0.52 |
| compressed mass | 0.51 | 0.48 |
| selected rows at distance <128 / 128–512 / 512–2K / 2K–8K / ≥8K | 7 / 10 / 30 / 54 / 0 % | 5 / 2 / 5 / 21 / **68** % |
| mass at those distances | 0.51 / 0.02 / 0.05 / 0.42 / 0 | 0.54 / 0.01 / 0.01 / 0.03 / **0.41** |
| share of compressed mass in top 64 / top 16 of the 512 | 77% / 61% | 74% / 57% |
| heaviest salt-definition layers (mean mass) | L38 0.16, L37 0.15, L22 0.08 | L37 0.17, L38 0.16, L22 0.11 |

In prefill (sampled every N/32 positions) the sink mass is 34% at position 0, where the only
key is the token itself, and **0.7–1.4% everywhere after**, with no trend in position.
Mid-range (128–2K) rows get little mass at either length: the model attends to the local
window and to the far past. At 17K, **the top 16 rows carry 57%** of the compressed mass
(§6 Q4).

### 5.5 Routing (design §9.1.1 Q3)

| | Q3, 27K tokens / 40 prompts | 4K prefill | 17K prefill |
|---|---:|---:|---:|
| adjacent-token Jaccard, 40-layer mean | 0.234 | 0.204 | 0.201 |
| … tokens < 2,048 / ≥ 2,048 | — | 0.210 / 0.199 | 0.210 / 0.199 |
| per-layer range | 0.091 (L0) – 0.340 (L25) | 0.062 (L0) – 0.327 (L25) | 0.062 (L0) – 0.323 (L25) |
| `union_frac[2..5]` | 0.827 / 0.735 / 0.676 / 0.631 | 0.847 / 0.765 / 0.710 / 0.670 | 0.849 / 0.768 / 0.714 / 0.674 |
| decode steps (prefill-last → 8 steps) | — | 0.165 | 0.202 |

Context length has no effect on expert locality. The 0.03 gap to Q3 is the corpus: these
prompts are ~40% code and the salt head. It is not a position effect, since the before/after
2,048 split is flat. §9 and §10.3 need no change.

### 5.6 Greedy margins (top1 − top2 logit)

| record | 4K | 17K |
|---|---:|---:|
| prefill → `k` | 8.86 | 9.48 |
| step 0 → `est` | 10.78 | 11.38 |
| step 1 → `rel` | 11.66 | 11.65 |
| step 2 → `-` | 9.51 | 10.59 |
| step 3 → `447` | 8.83 | 10.10 |
| step 4 → `1` | 10.77 | 11.06 |
| step 5 → `-` | 8.41 | 9.12 |
| step 6 → `amber` | 9.65 | 11.33 |
| step 7 → `"\n` | **1.30** (vs `"`) | **1.66** |

---

## 6. Open questions for design §11 / §7.5 (and §7.4)

1. **Fidelity vs intent for the stale index keys (§5.3).** Replicating `shared_attn.index_k`
   is required for L3 agreement. It also means layers 2–19 attend to a poorly chosen set on
   half of all positions. Should deepMoE keep an option to use each source's own keys, and
   evaluate it only as "different model, measured"? The data to judge it is here: own-key
   top-k vs reference, per step.
2. **Top-k tie semantics (§4.3).** Design §7.4's radix select was validated against a CPU
   *stable* sort. The reference's tie choice is neither stable order nor its reverse. The
   proposal is to make "tie-aware equal" the §12 L2/L3 criterion for `topk_idxs`, and to
   state it in §7.4.
3. **Index compute at 64K (§5.1)**: 1.74 GFLOP / 29 MB per token as the reference does it.
   Adopt candidate-restricted scoring for 24–36 (exactly equivalent) before sizing the
   indexer kernel?
4. **§7.5 sizing**: the top 16 of the 512 rows carry 57% of the compressed mass, and the top
   64 carry 74%. `index_topk` is part of the model and must stay 512. A GPU kernel could still
   exploit this concentration (early-exit / tiling), which is a question for P3 kernels, not
   for correctness.
5. **§11.2 bounded replay** is not covered: both runs are oracle-mode prefill. Its error needs
   a replay run on the same prompts (only the decoder layers change). `traces/longctx`
   provides the oracle side, so only the approximate side needs running.
6. **§11.3 capacity**: the table is right for storage. It should gain a *per-step read* row
   (≈ 37 MB at 64K), and the `KvStore` sizing pitfall in §4.3 item 1 should be noted.
7. **§12 L3 "≥ 5 prompts"**: these add two long ones, both 9/9 on the reference. The engine
   comparison is Track I's next step.

---

## 7. Reproduce

```
.venv\Scripts\python.exe tools\oracle_longctx.py prompts
.venv\Scripts\python.exe tools\oracle_longctx.py run --name ctx4k  --index-chunk 1024 --index-verify
.venv\Scripts\python.exe tools\oracle_longctx.py run --name ctx16k --index-chunk 1024
.venv\Scripts\python.exe tools\oracle_longctx.py stats
```

`run --estimate-only` prints the wall-time model. `run` waits for other heavy Python oracles,
for physical memory, and for commit headroom, polling every 180 s. `prune` re-applies the
small-copy drop lists without rerunning.
