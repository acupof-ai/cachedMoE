# P3 — the model talks (Track P)

> `deepmoe serve`, sampling, a C++ tokenizer, the official prompt format,
> Track L's prefill wired into the engine, and a terminal chat.
> Companion to [p2_decode.md](p2_decode.md) (the decode step and the engine),
> [p3_prefill.md](p3_prefill.md) (the GPU prefill this hands off from) and
> design §7.11 (the head and the sampler).

Status: **2026-09-15**. Everything below ran on the real checkpoint on the real
machine. Raw data: `bench/results/chat/` (transcripts, per-turn stats JSON,
turn scripts), `tests/data/tokenizer/golden.json`.

---

## 0. One page

* **It talks.** A 3-turn Chinese/English conversation and a code question,
  greedy and then sampled (temperature 1.0, top_p 0.95), plus a
  thinking-mode exchange. All replies are coherent and on-topic, the
  translations are faithful, and the code is correct (§7).
* **`deepmoe serve`** keeps one engine alive: the pinned set, the 78.8 GiB
  expert cache and the KV state all persist across turns. Requests and events
  are line-delimited JSON on stdin/stdout. When a new prompt's ids extend the
  previous context, only the new ids are prefilled: turn 2 reused 153 tokens
  and prefilled 11 (§1).
* **Sampling**: `gpu/shaders/sample_topk.slang` returns the top set of logits
  (≈1024 candidates) plus the tail mass, so the host builds the full-vocabulary
  top-p nucleus exactly. Chat results:
  * 0 fallbacks in 650 sampled steps;
  * 226/226 kernel results bit-equal to the host emulation (`--check-topk`);
  * 0.1 ms of host time per sampled token.
  CPU tests against the L3 export's recorded logits:
  * 200,000 draws: χ² 54.1 on 49 df, worst |z| 2.2;
  * the top-set nucleus equals the full-vocab nucleus in 43 of 43 cases that
    claim exactness (§2).
* **Tokenizer**: C++ byte-level BPE, **100% id-equal to HF `tokenizers`**, and
  decode, skip-special decode and streaming decode are 100% equal too. Coverage:
  * 24,897 cases, 7.69 M ids;
  * every corpus file Track B used;
  * both long-context prompts;
  * every Unicode scalar value;
  * 20,000 fuzz strings.
  909 of these cases are committed as a golden file (§3).
* **Prompt format**: rendered by the checkpoint's own `encoding/encoding.py`,
  imported read-only by `tools/chat.py`. `tools/encoding_check.py` runs the
  official `test_encoding.py` without pytest: 50/50 pass, including the 5
  `encoding/tests` fixtures (§4).
* **Long-context decode fixed** (§5). The sparse-attention list was sized
  `window + n_cmp`; model.py uses `window + min(index_topk, n_cmp)`. This is
  why decode collapsed past ~1K tokens. After the fix:
  * decoding from Track M's 4,133-token reference state is 8/8 teacher-forced
    and 8/8 free-running;
  * `serve` with Track L's GPU prefill (97.9 s) continues the 4K prompt with
    `kestrel-4471-amber"` exactly.
* **Speed** (full auto cache, idle machine):

  | | TTFT | decode | expert-cache hit rate |
  |---|---:|---:|---:|
  | short prompts (11–46 tokens) | 5–23 s | 3.4 → 4.5 tok/s as the cache warms | 0.84 → 0.90 |

  Every step is ~90 ms of compute plus 120–190 ms of NVMe stall. That stall
  is the whole gap to the 12 tok/s warm floor (§7.2).

---

## 1. `deepmoe serve`

`cli/serve.cpp`, `runtime/session.{h,cpp}`, `Engine::begin_session / feed /
reset_context`.

```
deepmoe serve --model DIR [--cache-gb N] [--max-context 4096] [--engram-tables tests/data/l3]
              [--gpu-prefill-min N] [--replay 128] [--check-topk] [--profile F.jsonl]
```

| request | effect |
|---|---|
| `{"op":"generate","prompt_ids":[..] \| "text":"..","max_tokens":N,"temperature":T,"top_p":P,"seed":S,"stop_ids":[1],"reuse":true}` | streams `prefill` progress, then `token` events, then one `done` |
| `{"op":"reset"}` | KV state to position 0; the pinned set and expert cache stay warm |
| `{"op":"tokenize","text":..}` / `{"op":"detokenize","ids":[..]}` | the C++ tokenizer |
| `{"op":"status"}` | context length, store and planner counters |
| `{"op":"quit"}` | |

Every `token` event carries:
* `id` and `text` (the UTF-8 this token completed; partial characters are held
  back);
* `t_ms` (since the request arrived) and `step_ms`;
* `p` (softmax probability of the sampled token) and `margin`;
* `nucleus` (its size) and `hit` (the step's expert-cache hit rate).

`done` carries the §13.1 breakdown, averaged per decode token: attn, moe_gpu,
moe_host, nvme_stall, engram, tail, other, record/submit/bind/fence_wait,
submits and sample_host. It also carries TTFT, prefill and decode tok/s, hit
rates and NVMe MB for prefill and decode separately, fallback and top-k check
counts, mean nucleus size, and the finish reason (`stop`/`length`/`context`).

The engine prints its logs on fd 1. So `serve` duplicates the real stdout for
the protocol and points fd 1 at stderr: stdout carries only JSON.

**Sessions without an export.** `Engine::begin_session`:
* creates the KV store for `max_context` positions;
* loads the engram hash tables (constants, a pure function of the tokenizer;
  read from the L3 directory's `index.json` + token map);
* turns design 7.4's kernels on.

`feed(tokens)` runs one teacher-forced decode step per token at positions
`context_length()..`; only the last step samples. This is `slow_prefill`
without its 128-token guard. A decode query at p > 128 sees the ring's last
128 positions, which is exactly the band a chunked prefill's query p sees.
§5 and §7 show it holds past 128.

**KV continuation.** The KV store holds exactly `engine.history()`. If a prompt
p has `history == p[0..h)` with h < |p|, the session feeds `p[h..)`. Anything
else resets and feeds all of p, logging the common prefix. A generated token is
fed only when the next token is wanted. So after a turn ends on EOS, the
history is prompt + reply minus its last token, and the next rendered prompt
(which contains that EOS) extends it. Rolling back is not supported: the ring
overwrites slots, so a prompt that diverges k tokens back pays a full
re-prefill.

## 2. Sampling

`runtime/sampling.{h,cpp}`, `gpu/shaders/sample_topk.slang`
(`DecodeStage::SampleTopK`), `Engine::sample_into`.

A sampled step records one more dispatch into the tail buffer, after
head.slang's argmax. The argmax still runs, and `greedy_token` and the margin
are reported either way. The kernel runs in one workgroup of 256 threads; each
thread owns ids ≡ tid (mod 256):

1. global max M (reduction tree);
2. per-thread histograms of `(M − l)·16` over 256 bins (16 logits deep);
3. thread 0 picks the first bin J whose cumulative count reaches k = 1024;
4. every thread writes each (id, logit) in bins 0..J to its own 32-pair
   segment (a true top set), and sums `exp(−(M−l)/T)` over the rest (the
   tail);
5. thread 0 writes a header with M, the tail, J, the per-thread counts and an
   overflow flag.

The host sums the candidate weights in double: `S = Σ_cand exp((l−M)/T) + tail`
and `p_i = exp((l_i−M)/T)/S`. The candidates are then sorted by
(logit desc, id asc) and cut at the first cumulative mass ≥ top_p. That cut is
the full-vocabulary nucleus whenever the candidates hold ≥ top_p of the mass.
The host copies all 129,280 logits (`nucleus_from_full`) and reports
`topk_fallback` when any of these holds:
* the candidates hold less than top_p of the mass;
* `top_p = 1`;
* a thread overflowed (> 32 candidates).

The draw is inverse-CDF with `uniform01(seed, position+1)`, a counter-based
splitmix64. A run is reproducible from its seed, and a KV-continued turn
samples exactly what a fresh run would. Temperature ≤ 0 is greedy. The
defaults, temperature 1.0 and top_p 0.95, come from the model README.

**Tests** (`tests/test_decode.cpp`, suite `sampling`, CPU only, registered as
`unit`):

* `frequencies_match_top_p_softmax_on_l3_logits`. The L3 export records each
  step's top-64 logits and the full-vocab logsumexp. The test picks the widest
  provably exact nucleus: record 8 at top_p 0.7, 50 tokens holding 0.701 of the
  mass. It draws **200,000** samples and compares them with an independently
  written softmax(top_p): **χ² 54.09 on 49 df** (5σ bound 98.5), worst
  per-token |z| 2.22, 0 draws outside the nucleus. At top_p 0.95 the widest
  exact nucleus is 5 tokens (record 7), which is why 0.7 is also tried.
* `topk_nucleus_equals_full_vocabulary_nucleus`. Each record's recorded top-64
  goes into a seeded full vocabulary, over T ∈ {0.7, 1.0, 1.5} and
  top_p ∈ {0.5, 0.95, 0.99, 1.0}. `emulate_topk`, a float-for-float mirror of
  the shader, always returns a true top set. All **43** cases that claim
  exactness give the same ids as the full-vocab nucleus, with probabilities
  within 1e-5 relative. The other 65 fall back, including every top_p = 1 case.
  A flat distribution overflows and falls back.
* On the GPU, `serve --check-topk` compares the kernel's header and candidate
  set with `emulate_topk` on a host copy at every sampled step:
  **0 mismatches in 48 + 226 checked steps**. The chat runs had **0
  fallbacks in 650 sampled steps**; mean nucleus sizes were 1.2–2.7 tokens
  (a confident model). The kernel adds ~2–3 ms to the tail; the host half is
  0.1 ms.

## 3. Tokenizer

`text/tokenizer.{h,cpp}`, `text/unicode_tables.inc`.
Tooling: `tools/gen_unicode_tables.py`, `tools/tokenizer_golden.py`,
`deepmoe tokenize`. Tests: `tests/test_tokenizer.cpp` (suite `tokenizer`).

### 3.1 What tokenizer.json specifies

| stage | contents | implementation |
|---|---|---|
| normalizer | empty `Sequence` | identity |
| added tokens | 1,283 tokens; 1,236 with `normalized:false` (1,229 special), 47 with `normalized:true` | HF's two passes: split on the unnormalized trie, then the normalized trie, each leftmost-longest (byte tries) |
| pre-tokenizer | `Split(\p{N}{1,3})`, `Split([一-龥ぁ-ゟ゠-ヿ]+)`, `Split(<punct/letter/space alternation>)` (all `Isolated`), then ByteLevel with `use_regex:false` | regexes hand-compiled, each run inside every piece the previous pass produced |
| model | BPE: 128,000 tokens, 127,741 merges; no dropout, unk, byte fallback or ignore_merges | HF `Word::merge_all` priority queue: lowest rank, ties to the leftmost pair; stale entries skipped by new-id check |
| post-processor | ByteLevel (offsets only) | nothing; no BOS (`add_bos_token: false`) |
| decoder | ByteLevel: chars → bytes, or the token's raw UTF-8 if any char is outside the byte map | same rule, then Rust `from_utf8_lossy` (U+FFFD per maximal invalid subpart) |

`Tokenizer::load` refuses a tokenizer.json whose three regexes, normalizer or
ByteLevel flags differ from the ones hand-compiled here. It cannot silently
tokenize a different model's format.

### 3.2 The third regex, compiled by hand

The pattern is:

```
[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
```

It is a leftmost-first alternation. Each branch's backtracking is resolved in
closed form:
* The optional prefix of branch 2 is taken only if a letter or mark follows.
  Otherwise the branch retries from the same character, which matters when that
  character is itself a combining mark.
* `\s*[\r\n]+` ends right after the last CR/LF in the whitespace run.
* `\s+(?!\S)` takes the whole run at the end of the piece, and otherwise the
  run minus its last character when the run has ≥ 2 characters.

**Character classes.** `\p{N} \p{L} \p{M} \p{P} \p{S} \s` come from HF's own
regex engine, not from a Unicode database. `gen_unicode_tables.py` runs
`Split(Regex(class), "removed")` once over a string of every scalar value and
records what disappeared. Python's `unicodedata` or a different Oniguruma
version therefore cannot disagree with the reference. `\s` turns out to be 25
code points: 0x09–0x0D, 0x20, 0x85, 0xA0, U+1680, U+2000–200A, U+2028/2029,
U+202F, U+205F, U+3000.

### 3.3 Validation (`tools/tokenizer_golden.py`, HF tokenizers 0.23.2)

| set | cases | ids | ids equal | decode | skip-special | stream |
|---|---:|---:|---:|---:|---:|---:|
| corpus (tools/corpus.py `default_sources`: repo docs + C++, model card, tech report PDF text, reference Python, CPython stdlib) | 61 | 460,096 | 61/61 | 61/61 | 61/61 | 61/61 |
| longctx prompts (`tests/data/longctx/prompts.json`) | 2 | 21,118 | 2/2 | 2/2 | 2/2 | 2/2 |
| encoding: the 5 `encoding/tests` outputs + 10 renders (chat and thinking) | 15 | 4,136 | 15/15 | 15/15 | 15/15 | 15/15 |
| added tokens alone and glued to text | 2,566 | 11,547 | all | all | all | all |
| adversarial (emoji ZWJ/flags/skin tones, mixed scripts, every Unicode space, CRLF mixes, combining marks, digit runs, byte-level lookalikes, special-token fragments, 1K-char runs) | 81 | 1,540 | all | all | all | all |
| fuzz (seeded, 20,000 strings over those pools) | 20,000 | 765,264 | all | all | all | all |
| every Unicode scalar value, in 512-cp chunks and 4 contexts | 2,172 | 6,429,354 | all | all | all | all |
| **total** | **24,897** | **7,693,055** | **100%** | **100%** | **100%** | **100%** |

The C++ tokenizer loads in ~0.2 s and encodes at 5–12 MB/s.
`tests/data/tokenizer/golden.json` (1.5 MB, 909 cases) is the committed subset
`suite.tokenizer` checks: all adversarial, encoding and 128 added-token cases,
600 fuzz, 24 code-point chunks, and the first 1,500 characters of every corpus
file. `prompts.json`'s stored `ids` differ from `encode(text)` for HF too:
`oracle_longctx.py` builds them piecewise (BOS + head + filler + tail encoded
separately). That is not a tokenizer disagreement.

`StreamDecoder` holds back an incomplete trailing UTF-8 sequence until its
bytes arrive. The concatenation of its pushes equals `decode` on every case
above.

## 4. Prompt encoding: the Python path

The official `encoding.py` is 979 lines: tools in DSML, namespaces, images,
tasks, reasoning effort, drop_thinking, mid-conversation system messages. A C++
port would be a second implementation to keep in sync. The faster path to a
correct result was this:
1. `tools/chat.py` imports `encoding.py` from the model directory read-only
   (`sys.dont_write_bytecode`) and renders the text;
2. `serve`'s C++ tokenizer turns it into ids (proven identical to HF, §3);
3. `serve` accepts `prompt_ids` (or `text`).

`tools/encoding_check.py` runs the checkpoint's own `test_encoding.py` through a
50-line pytest shim (parametrize, raises, main); the `.venv` has no pytest and
there is no network. Result: **50 passed, 0 failed**, including the 5
`encoding/tests` input → output fixtures. Nothing was written under the model
directory.

**KV reuse with the official format.** `chat.py` keeps the previous turn's ids
(prompt + generated) and the text they render to. If the newly rendered
conversation starts with that text, the new prompt is those ids plus the
tokenization of the rendered suffix. The server then sees a strict extension.
The generated ids are used as generated, never re-tokenized: BPE-sampled ids
need not be the canonical tokenization of their own text.

The official default `drop_thinking=True` removes earlier turns' reasoning. The
rendered prompt then stops extending the context, and every turn re-prefills.
`chat.py` therefore defaults to `drop_thinking=False` (reasoning stays in
context, and reuse works: the thinking follow-up in §7 reused 130 tokens).
`/drop` switches to the official behaviour.

## 5. The long-context decode bug, found and fixed

p3_prefill.md §8.3 item 2 reported the failure: decoding past ~1K tokens gave
NaN or collapsed logits, even from the reference's own 4K state, and
`kMaxIndexPositions` capped context at 4,096.

**Cause (runtime, not a shader).** `Engine::prepare_ced` set every compressed
layer's list length to `n_kv = window + n_cmp`. model.py's indexer returns
`topk = min(index_topk, end_pos // ratio)` picks, so the list the reference
attends over is `window + min(512, n_cmp)`. The two agree up to 512 compressed
positions: 512 tokens at ratio 1 (layers 20–39), 1,024 at ratio 2. Past that:
* `sparse_attn` walked `n_cmp − 512` entries the indexer never wrote. They are
  zeros, which address window slot 0, so one KV row was attended hundreds of
  times: the "collapse into filler tokens".
* Past `n_kv = 1024`, `sparse_attn`'s per-head score stride (1024, sized for
  128 + 512 "with room") overlapped the next head's scores: the NaNs.

**Fix.** `n_sel = min(n_cmp, index_topk)`, passed to
`KvStore::set_decode_topk(layer, position, n_cmp, n_sel)` and `set_counts`.
`kMaxIndexPositions` is raised to 16,384 (a 64 KB score plane). Design 2.1's
candidate-block mask starts past 16,384 and is still refused there.

**Evidence** (`deepmoe run --state traces/longctx/ctx4k`, N = 4,133, 8 GiB cache):
* **8/8 teacher-forced and 8/8 free-running** against the fp32 reference, which
  is the salt `kestrel-4471-amber"`, at margins 8.4–12.2 (step 7's `"\n` has
  margin 0.98, the reference's own near-tie shape).
* `--topk-report` (new) checks every index source's list at every step. Every
  list is valid: count 512, strictly increasing, in range. It shares 426–503 of
  512 entries with the reference's list. For comparison, Track L's prefill picks
  share 87%: scores are not bit-equal when hidden states are at cos 0.96–0.99.
* Exact check of the kernel: on the last index source (L36), `indexer.slang`
  stage 5's radix select matches a host top-k of its own 4,134–4,137 scores
  with **0 of 512 entries different**. The `InterlockedAdd` histogram is not
  losing counts.
* **End to end with Track L's GPU prefill** (`serve --gpu-prefill-min 256
  --cache-gb 16`, the ctx4k prompt, greedy):

  | prefill | decode | decode hit rate |
  |---|---|---:|
  | 97.9 s: expert I/O 35.8 s, 10,544 experts / 184.7 GiB | `kestrel-4471-amber"` exactly, first token margin 8.34; 0.57–1.3 s/token with a 900-slot cache | 0.40 |

This touches `runtime/kvstore.*` and `runtime/decode_layer.h`. The coordinator
has since assigned them to Track Q, and the fix landed before that assignment
(commit 16cf319).

## 6. Prefill in the engine (Track L's API)

The `gpu::Prefill` API is stable (`create`, `run → PrefillHandoff`). p3_prefill.md
§8.3 item 1 asked for two engine entry points, and both exist now:
* `Engine::seed_from_prefill(const gpu::PrefillHandoff&)` seeds, in memory:
  * window rings (slot p % 128);
  * the kv sources' compressed cache and index keys;
  * the compressor state;
  * the history, with `pub_index_k_` = the last kv source.
  No directory round trip.
* `Engine::gpu_prefill(prompt, replay)` builds a `PrefillRunner` + `Prefill` on
  the engine's device, path-A allocator, shards, IoEngine and pinned set
  (engram tables from the session), runs it, seeds the state and frees the
  prefill's buffers. When sampling, it draws the first token from the handoff's
  host logits.

`serve --gpu-prefill-min N` sends fresh prompts of ≥ N tokens there. If it
fails (typically path A has no room beside an auto-sized 78.8 GiB cache), the
session logs a warning, resets, and falls back to the decode path. Short
prompts and every continuation use the decode path, which also warms the
decode cache.

~~**Default is off.**~~ **Default is ON at 512 since Track PF** (§8). At N = 64
the GPU prefill streams 94 GB of experts (43 s) and warms nothing, and the decode
path took 5–23 s for this document's 11–46-token prompts — so the threshold stays
at the ~500 tokens this paragraph named. What changed is everything above it:
F4's path-A reserve made the prefill allocatable beside a 5,000-slot cache
(`p4_hitrate.md` §5, so no explicit `--cache-gb` is needed any more), and
`PREFILL_HANDOFF` puts the streamed experts into the decode cache, so it no
longer "warms nothing". Measured at 1,118 tokens: **168.8 s → 28.6 s**.
And a CONTINUATION now reaches the GPU too, by throwing the reuse away when
that is cheaper (§8.3(b)) — the engine still cannot prefill a suffix at a
nonzero position (§8.4).

Still not done: §8.3 item 3 (`ExpertStore::adopt`, so the prefill's
experts fill the decode cache).

## 7. Smoke: the first conversations

`tools/chat.py --script bench/results/chat/smoke_turns.json` on an idle
machine: one `serve` process, auto cache 78.8 GiB (4,500 slots), cold at turn
0, context 4,096. Turns 0–3 are greedy, 4–7 sampled (T 1.0, top_p 0.95, seed
20260915). Turns 3, 4 and 7 reset the context; the cache stays warm.

### 7.1 Per turn

| # | turn | prefill (reused) | TTFT s | prefill tok/s | prefill hit | gen | decode tok/s | decode hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | greedy, zh intro (cold) | 13 (0) | 8.7 | 1.50 | 0.54 | 54 | 3.41 | 0.836 |
| 1 | greedy, en question | 24 (66) | 8.0 | 3.00 | 0.79 | 64 | 3.37 | 0.837 |
| 2 | greedy, zh translate | 11 (153) | 4.9 | 2.26 | 0.71 | 55 | 4.46 | 0.897 |
| 3 | greedy, code (reset) | 27 (0) | 11.4 | 2.36 | 0.72 | 96 | 3.51 | 0.843 |
| 4 | sampled, zh intro (reset) | 13 (0) | 6.0 | 2.16 | 0.68 | 46 | 3.70 | 0.857 |
| 5 | sampled, en question | 24 (58) | 8.1 | 2.97 | 0.80 | 66 | 3.38 | 0.836 |
| 6 | sampled, zh translate | 11 (147) | 5.0 | 2.20 | 0.69 | 56 | 4.26 | 0.889 |
| 7 | sampled, code (reset) | 27 (0) | 11.4 | 2.36 | 0.72 | 256 | 3.94 | 0.872 |
| T0 | thinking, sampled, zh (new process, cold) | 46 (0) | 23.4 | 1.96 | 0.65 | 85 | 4.25 | 0.893 |
| T1 | thinking follow-up, en | 15 (130) | 5.2 | 2.87 | 0.78 | 141 | 4.39 | 0.898 |

TTFT here is the prefill: 11–46 tokens at 1.5–3 tok/s through the decode path.

### 7.2 Where a decode token goes

Turn 7 (256 tokens, warm), mean per token:

| attn | moe gpu | moe host | nvme stall | engram | tail | other | submits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 39.3 ms | 41.4 ms | 1.1 ms | **151.9 ms** | 4.2 ms | 7.3 ms | 8.2 ms | 41 |

Over turns 0–7 the stall ranges 124–193 ms per token and NVMe reads
460–740 MB per token. Everything except the stall is ~100 ms, which is
p2_decode.md §10's 81.5 ms floor plus the sampler (tail 5.9 → 7–9 ms) and a
second first-touch effect in the MoE.

The model is p2_decode.md §12.3's: tok/s follows the hit rate, 3.4 tok/s at
h ≈ 0.84 and 4.5 at h ≈ 0.90. ~1,000 generated tokens across these turns do
not fill a 4,500-slot cache (§12.3 needed h ≥ 0.94 for 6–9 tok/s). Short
conversations like these stay in the ramp.

### 7.3 Transcript, abridged

Full transcripts: `bench/results/chat/smoke_transcript.md` and
`smoke_think_transcript.md`.

* **0** 你好！请用两三句话介绍一下你自己。
  → 你好！我是DeepSeek，由深度求索公司创造的AI助手，擅长回答各种问题、提供信息和帮助解决问题。我可以处理文本和图片输入，支持文件上传和联网搜索，而且完全免费使用。有什么我可以帮你的吗？😊
* **1** What is the capital of France, and what is one interesting fact about it? Answer in English.
  → The capital of France is **Paris**. One interesting fact about it is that the **Eiffel Tower was originally intended to be a temporary structure** built for the 1889 World's Fair, and it was almost demolished after 20 years—but it was saved because it proved useful as a radio antenna.
* **2** 把你上一个回答翻译成中文。
  → 法国的首都是**巴黎**。关于它的一个有趣事实是：**埃菲尔铁塔最初只是为1889年世界博览会建造的临时建筑**，原本计划在20年后拆除，但因为它被证明可以作为无线电天线使用，才得以保留下来。
* **3** Write a Python function that returns the n-th Fibonacci number iteratively…
  → a correct `fibonacci(n)` (`a, b = b, a + b` loop, ValueError for n < 0,
  doctest `fibonacci(10) → 55`).
* **4–6 sampled** give the same facts in different words ("…它之所以被保留下来，只是因为它的高度使它适合用作无线电天线。").
* **7 sampled** gives the same function, plus an explanation (O(n) time, O(1)
  space) and four example calls, all correct (`fibonacci(50) → 12586269025`).
* **T0 thinking** 9.11 和 9.9 哪个大？简短回答。
  → reasoning: "…需要注意陷阱：作为数字 9.11 < 9.9。但可能像版本号？…"
  → answer **9.9 大。**
* **T1** Why do people often get this wrong? One sentence.
  → reasoning in English, then: *People often compare the digits after the
  decimal point as if they were whole numbers (11 vs. 9), forgetting that 9.9
  equals 9.90.*

By eye, the checks pass:
* the model answers the question asked, in the requested language;
* translation preserves content;
* the code runs;
* thinking mode reasons, then closes `</think>` and answers;
* EOS stops every turn (no `length` finishes).

## 8. Track PF：prefill 为什么一直走 decode 路径（2026-09-19）

用户在 web UI 上看到的那一行：**`预填充 304 / 1,067 token · 199.6 ms/token`**。
`total` 是 `to_prefill`（tools/web/server.py `_run`），所以 **reuse = 0、整个 1,067 token
都在 decode 路径上重新 prefill**，一轮 TTFT ≈ 213 s。这一节是三个原因、两个改动和
一个没做的。

### 8.1 为什么 `--log` 文件里只有两行

`deepmoe serve` 把**真正的 stdout 复制成协议通道**，然后 `dup2(stderr, 1)`
（`cli/serve.cpp`）——所以 fd 1 指向 stderr，注释说"其他打到 fd 1 的都去 stderr"。
**但 `core/log.h` 写的是 `FILE* stdout`，而 pipe 上的 `stdout` 是全缓冲的**，
`emit` 只在 `>= Warn` 时 `fflush`。于是一个 serve 会话写的每一条 `log_info`
都躺在一个没人 flush 的 4 KiB 缓冲里，进程被 `Stop-Process` 杀掉时一起消失。
`build/web_serve.log` 里那两行是 `fprintf(stderr, ...)` **直接写的**
（`serve: restored 374 tokens from kv disk`），不是 logger 写的。

修：`core/log.h` 加一个进程级 sink（`set_log_stream(FILE*)`，默认 null = 旧行为），
并且**每一行都 flush**；`cli/serve.cpp` 在 dup 之后调 `set_log_stream(stderr)`。
同一份 `--log` 文件现在开机就有 37 行，其中包括本节所有的 session 决策行。

### 8.2 reuse 为什么是 0

**`Session::generate` 原来只会为 `reuse == 0` 的 prompt 走 GPU prefill**
（`runtime/session.cpp`），而 `gpu_prefill_min` **默认是 0**——所以两条路都关着：
一个全新的长 prompt 因为默认关而走 decode；一个续写因为 `reuse > 0` 而走 decode。
用户那一轮是前者（新会话 / `.pkv` 恢复的 history 与浏览器这一侧的 `ctx_ids` 对不上，
两种都让 `common prefix = 0`）。

**thinking 模式本身不是原因。** 官方模板 `drop_thinking=True` 会把更早轮次的
reasoning 从 prompt 里删掉，所以第二轮的 prompt 不是 history 的延长——但
Track R2 的回退已经覆盖了这条路径（`docs/p4_kv_ux.md` §6），它给出的是
**共同前缀那么多的 reuse，不是 0**。本轮实测复现（web UI 的渲染器、思考开）：

```
[INF] session: prompt 1514 tokens, history 1125, common prefix 1117 -> reuse 0 (gpu-recompute), ...
```

history 1,125、共同前缀 1,117——**差的 8 个 token 正是被 `drop_thinking` 删掉的那段**，
回退点 1,116，reuse 1,116 而不是 0。**渲染器不用改**。

新增的日志（每轮一行，`--log` 里）：

```
[INF] session: prompt N tokens, history H, common prefix C -> reuse R (cold|extend|rollback|reset|gpu-recompute), S to prefill (gpu_prefill_min M)
[INF] session: S tokens prefilled on the GPU in T s (X ms/token)
[INF] session: S tokens on the DECODE path (<为什么>)
```

### 8.3 两个改动

**(a) GPU prefill 默认开，阈值 512**（`runtime/session.h` `gpu_prefill_min = 512`；
serve 继承，web 什么都不传）。§6 当初关掉它的两条理由**今天都不成立了**：
F4 的 path-A 预留让它在 5,000 槽的 cache 旁边**能分配**（`p4_hitrate.md` §5），
`PREFILL_HANDOFF`（F4 之后默认开）把 prefill 流过的 expert 交给 decode cache，
所以它不再"什么都没热"。512 是 §6 自己给的阈值，本轮没有改它的数据。

**(b) 续写也能上 GPU——靠扔掉 reuse。** reuse 只值它**省下的那些 token 的价钱**：
一个续写的后缀走 decode（本机实测 151–172 ms/token），而一个冷 prompt 可以整个走
GPU prefill（23–26 ms/token）。于是 reuse 了 p 中 r 个 token 的一轮，代价是
`p − r` 个 decode token，而**把 reuse 扔掉、整个 prompt 重新 GPU prefill** 的代价是
`p / speedup` 个 GPU token。`speedup` 默认 **6.0**（实测热态比值 5.9–7.3，取下界留余量；
`--gpu-prefill-speedup`，`<= 1` 关掉这条规则）。规则在 `r < p × (1 − 1/6) = 83%` 时触发。

**输出不受影响**：从 0 重新 GPU prefill 整个 prompt，和这一轮碰上空 store 时会做的事
**逐字相同**。门是 Track L 的 `suite.gpu_prefill`：
`gpu_prefill.forty_layers` **教师强制 8/8 + 自由运行 8/8**，
`gpu_prefill.longctx`（`DEEPMOE_PF_LONGCTX=traces/longctx/ctx4k`，4,133 token）
**自由运行 8/8**，worst window KV cos 0.957 / compressed 0.969 / index keys 0.978。
回退路径不变：`kv_replay.l3_64` 场景 (5)(6) 仍然是 reused 60 / 44
（那些 prompt 只有 64–69 个 token，远在 512 之下，新规则够不着）。

一个已知的降级：`cold_wins` 先 `reset_context()` 再 prefill，**万一 GPU prefill 失败**，
回退的 decode 路径要喂整个 prompt 而不是后缀。`gpu_prefill` 失败在 F4 的 path-A 预留之后
没有再出现过，而它失败时本来也要 `reset_context()`。

### 8.4 没做：在非零位置 prefill 一个后缀

理想的形状是 `gpu_prefill(suffix, at = h)`：只算新的那 n 个 token，接在已有的 KV 后面。
**在 ~150 行的预算内做不到**，卡点是具体的五条，不是"复杂"：

1. **`gpu::Prefill::run(prompt)` 没有起始位置。** `run_layer` 里的两个位置基准
   （`fpos0` / `apos0`）只由本次 prompt 决定：`fpos0 = (replay && L > 20) ? N - R : 0`。
   那个非零分支是**给纯 SWA 层的有界 replay 用的**，KV 源（L0–L20）永远从 0 跑。
2. **attention 看不到旧的 KV。** `op_attention(q, kv, n_win, cmp, idx, ...)` 读的是
   prefill **自己为本次 token 建的**平面；`topk_rows(..., kv_pos0, ...)` 把绝对位置映射成
   行号 `pos - kv_pos0`，所以 kv 平面必须**真的存着** [h−127, h) 的窗口行、
   cmp 平面必须存着 [0, h/ratio) 的压缩行。这些行在引擎的 `KvStore` 设备缓冲里，
   布局不同，而 **`KvStore` 根本没有窗口环的读回口**——只有 `pack`，而 `pack`
   按设计 §11.2 **从不含 SWA 状态**。也就是说旧窗口得先用 128 步 decode replay 重建，
   而那正是这个改动想省掉的代价。
3. **index 这一侧同理。** `op_index_score(..., pos0)` 与 `candidate_blocks(b, pos0, ...)`
   需要 `g = (h + n)/ratio` 行旧的 index key 在 prefill 自己的 `idx` 平面里，
   还要复现 `pub_index_k_` 的发布顺序。
4. **`Engine::seed_from_prefill` 是"清空再整体播种"**（`cur_->kvs_.clear()`，
   然后逐层 `seed_*`，最后 `history_ = h.prompt`、`pub_index_k_` = "位置 0，每个源都发布过"）。
   续写要的是**按行偏移合并**。
5. **缓冲尺寸。** `PrefillConfig.max_tokens` 决定每一个激活与 KV 平面的大小，
   续写要按 `h + n` 开；h = 5,000 时光 cmp/idx 平面就是每层 ~13 MB 的 path-A 内存——
   而 path A 正是 F4 不得不加 4 GiB 预留的那块。

合起来要动 `gpu/vulkan/prefill_kernels.{h,cpp}`（run / run_layer / topk_rows /
candidate_blocks / op_index_score 的位置管线）、`runtime/engine.cpp`（一个会合并的
seed + 一条 KvStore 读回路径）和 `runtime/kvstore.{h,cpp}`（含环的逐层读回），
且全都落在 `gpu_prefill.forty_layers` 守着的数值敏感代码里。**没做，走 8.3(b)。**

### 8.5 测到的数（1,118-token prompt，5,000 槽，`--no-kv-disk`，一次一个引擎）

| | 之前（`gpu_prefill_min 0`） | 之后（默认 512） | |
|---|---|---|---|
| 新 prompt 1,118 token，**冷 cache** | **176.7 s**（158.1 ms/token，decode） | **39.2 s**（35.1 ms/token，gpu） | **4.5×** |
| 新 prompt 1,118 token，**热 cache** | **168.8 s**（151.0 ms/token，decode） | **28.6 s**（25.6 ms/token，gpu） | **5.9×** |
| 第二轮续写（+396 token → 1,514） | reuse **1,116**、prefill 398 个 decode token，**70.5 s**（171.5 ms/token） | reuse **扔掉**、1,514 整个走 GPU，**35.5 s**（23.4 ms/token） | **2.0×** |

预测对实测：预测"1,067 token 213 s → ~26 s（24 ms/token）"，实测 1,118 token
**168.8 → 28.6 s（25.6 ms/token）**；预测"续写 500 token 100 s → 若 GPU 续写不落地则不变"，
实测 **70.5 → 35.5 s**——比预测好，因为 8.3(b) 这条规则落地了。
热态 decode/GPU 比值实测 **151.0 / 25.6 = 5.9×**，正好是默认 `speedup = 6.0` 的来处。

顺带修掉的：`tools/web/server.py` 的 `if args.gpu_prefill_min:` 让 **`--gpu-prefill-min 0`
（关掉 GPU prefill 的唯一方式）变成空操作**；默认改成 `None`，并加了
`--gpu-prefill-speedup` 透传。


## 9. Done / not done

**Done**
* `deepmoe serve`: JSON over stdio, streamed tokens, §13.1 stats, KV
  continuation, reset, tokenize/detokenize/status.
* Temperature/top_p/greedy with a GPU top-k kernel, an exact host nucleus and a
  full-copy fallback. Tested: a CPU χ² test on the L3 logits, emulation vs full
  vocabulary, and kernel vs emulation on the GPU.
* C++ tokenizer, 100% equal to HF on 24,897 cases (7.69 M ids); a committed
  golden file and test suite.
* Official prompt format through `encoding.py`; its 50 tests pass.
* Long-context decode fixed and validated at 4,133 tokens.
* Track L's prefill integrated behind a threshold, with the decode path as
  fallback.
* `tools/chat.py`: interactive and scripted, with `/reset /think /drop /temp
  /top_p /greedy /max /seed /system /stats`.
* The smoke above.

**Not done / gaps**
* **Speed.** 3.4–4.5 tok/s at hit rate 0.84–0.90; the stall is 60% of every
  token. Prefetch (design §9.4) and a cache that starts warm are the levers.
  ~~The GPU prefill does not feed the decode cache (p3_prefill.md §8.3 item 3).~~
  Done in Track R1/F4: `PREFILL_HANDOFF` is on by default (`p4_hitrate.md` §8).
* **Short-prompt TTFT is the decode-path prefill** at 1.5–3 tok/s. The GPU
  prefill only pays above ~500 tokens, which is where its default threshold now
  is (§8); below that the decode path is still the whole TTFT.
* ~~**No KV rollback.**~~ Done in Track R2 (`p4_kv_ux.md` §6), thinking mode
  with the official `drop_thinking=True` included — and Track PF measured it
  working on the web UI's own renderer (§8.2). What is still missing is a
  prefill that can START at a nonzero position, so a reused context's suffix
  has only the decode path (§8.4); Track PF works around it by dropping the
  reuse when a whole-prompt GPU prefill is cheaper.
* **Context above 16,384 compressed positions** needs design 2.1's
  candidate-block mask (refused, not wrong). Between ~4K and 16K only the
  indexer's cost grows. Decode was validated at 4,133 tokens; 16K was not run.
* `serve` handles one conversation, and a request cannot be interrupted
  mid-generation.
* The engram tables are still read from `tests/data/l3`. They are constants,
  but deriving them in C++ (PCG64 + NFKC over the vocabulary) is not done.
