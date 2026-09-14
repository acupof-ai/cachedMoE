# P3 — real prefill (Track L)

> The prompt through the forty layers as one chunked pass, with the routed
> experts streamed expert-major off NVMe, ending in exactly the state
> `runtime::Engine`'s decode loop starts from. design §3.2, §7.13, §9.7, §11.
> Companion to [p2_decode.md](p2_decode.md) (the decode step this hands off to),
> [p2_attention.md](p2_attention.md) (the per-token kernels) and
> [kernel_p2_moe.md](kernel_p2_moe.md) (the GEMV the prefill GEMM is compared
> against).

Status: **v0.1, 2026-09-14** — design written before the kernels, then kept
true. §0 is the page to read; §11 is what is done and what is not.

---

## 0. What this says in one page

*(Filled in from the measurements in §§8–10 once they exist; the design
sections below are written first.)*

---

## 1. What the reference does at `start_pos == 0`

Everything below is read off `inference/model.py`, the prefill branch of every
module, and is what `tools/oracle_prefill.py` captures. It differs from the
decode step of p2_decode.md in six places, and each one is a kernel decision.

| module | decode (`start_pos > 0`, one query) | prefill (`start_pos == 0`, N queries) |
|---|---|---|
| `_window_kv` | one row into the ring at `start_pos % 128`; attend over the whole ring | all N rows quantised (fp8, block 32, over the whole post-RoPE vector); **attend over the chunk itself**, `window_kv = kv` (N rows, not 128); the ring is seeded afterwards with the last `min(N,128)` rows at slot `p % 128` |
| `get_window_topk_idxs` | the ring, oldest first, `-1` past `start_pos` | **a band**: query p sees `max(0,p-127) .. p`, one row of `min(N,128)` indices, `-1` past p |
| `Compressor` | fill slot `start_pos % r`, pool when the group completes | pool **all** `N // r` complete groups at once; the `N % r` tail goes into `kv_state/score_state[0..rem)`, the rest of the state stays zero / `-inf` |
| latent / index-key RoPE | `freqs[start_pos + 1 - r]` | `freqs[0 : N - N%r : r]` — group j at position `j·r`, its **first** token |
| `Indexer` | score against `end_pos // r` keys, no mask | score `[N][N//r]`, then **mask** key j for query p unless `j < (p+1) // r` — *a compressed block is visible only once complete* |
| top-k index offset | `offset = 128` (the ring) | `offset = kv.size(1) = N` (the chunk) — compressed rows sit after the N window rows in `cat([kv, compress_kv])` |

and one that is the same but matters more: `select_candidate_blocks` (layer 20
picks 2048 blocks of 8, layers 24..36 score inside them) is **the identity for
`N // r ≤ 16384`**. Every block that holds a visible position is kept, and a
block with none is `-inf` everywhere already. So the two-level top-k only has to
exist above 16K tokens at ratio 1 — the same bound `Engine::prepare_ced`
already refuses past.

The top-k row of query p, precisely: `k = min(512, N // r)` entries; the first
`min(k, c_p)` are the selected visible positions (all of them when
`c_p = (p+1)//r ≤ 512`) **sorted ascending** plus `N`; the rest are `-1`
(`topk` over `-inf` returns indices `≥ c_p`, `where(idxs < compress_lens)` turns
them into `-1`, and they sort last). At N = 64 every row selects every visible
position; the radix select only becomes live at 512 visible positions.

Everything else — mHC, Sinkhorn, the gate, the experts, the shared expert, the
engram — is the decode arithmetic applied row-wise, and `model.py`'s `MoE.forward`
is already expert-major: `for i in experts: y[idx] += expert(x[idx], w[idx])`.

### 1.1 CED: the layer split and the replay length, from `config.json`

`compress_ratios` is `[0,0, 2×18, 1×20, 0,0,0]` for layers 0..39 and the three
mtp blocks; `kv_source_layer_ids = [2, 8, 14, 20]`,
`index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`,
`candidate_source_layer_id = 20`. So:

* **encoder = layers 0–19**: window-only (0, 1) or ratio-2 compressed attention;
  every layer must run for every prompt token.
* **decoder = layers 20–39**: ratio 1, all reading layer 20's compressed KV and
  index keys; 20/24/28/32/36 re-run the indexer with their own queries.

**`model.py` has no bounded replay.** Its prefill runs every layer over every
token; design §11.2's replay is a deepMoE production approximation, and the
oracle mode is the reference. What each decoder layer contributes to the state
decode needs:

| decode needs, from decoder layer L | computed from | which prompt positions |
|---|---|---|
| its window ring (128 rows) | L's own hidden state | the last 128 |
| the compressed KV + index keys (layer 20) | layer 20's **`attn_norm` output** — i.e. the encoder output through layer 20's `hc_mixes`/`hc_pre`/`attn_norm` | **all N** |
| DSpark's `main_x` window (layers 37–39 inputs) | L's input stream | the last 128 |

So **replay length R = window = 128**, and layer 20 splits: its attention front
half (mHC, `attn_norm`, compressor, index keys) runs over all N; its queries,
attention, output projection and FFN — and all of layers 21–39 — run over the
last R. The approximation is exactly one thing: a replay position p sees window
KV rows `max(N-R, p-127) .. p` of decoder layers instead of `p-127 .. p`, because
decoder KV before `N-R` was never computed. Only position N−1 has a complete
window. **For N ≤ 128 the two modes are identical**, which is why the N = 64
validation is exact against the oracle. R is a parameter (`2·window` makes the
last 128 rows' windows complete at the cost of one more window of decoder
compute); the bench reports the production-vs-oracle difference at N = 512
(§10).

---

## 2. The schedule

N prompt tokens, replay R = min(N, 128), query block B (512 by default; §4 sizes
it). Activations live in GPU-addressable scratch and are addressed through the
same one-descriptor slot table every runner uses. Nothing below reads GPU memory
element-wise on the host (design §7.1 rule 10): the host reads the gate scores
(one `memcpy` of `[N][384]` per layer) and writes index lists.

```
embed N tokens (host: one row read per token, widen once, memcpy x4)       -> h [N][4][5120] fp32
engram hashes for all N tokens at layers 1, 14 (host, EngramTables)       -> 48·N row reads, issued at t=0

for L in 0..39:
    T = N if L < 20 else R          (layer 20's front half is N; see 1.1)
    if L in {1, 14}: engram(h[0..N))  — batched fp8 GEMM [25600 x 6144] + the gate
    -- attention half ------------------------------------------------------
    mhc.mix     : mixes[T][24] = hc_fn @ flatten(h)  x rsqrt(mean h^2)   GEMM fp32, K = 20480
    mhc.split   : Sinkhorn per token                                     -> pre/post/comb [T][24]
    mhc.pre     : u = bf16(sum_j pre_mix[j] h[j]); x = bf16(RMSNorm(u)) -> x [T][5120]
    wkv         : fp8 GEMM [512 x 5120], kv_norm, RoPE(p), act_quant    -> kv [T][512] (+ E4M3 bytes)
    if kv_source: compressor over all T rows (N at layer 20)           -> latents [T//r][512]
                  index keys (owns_k)                                  -> index_k [T//r][128]
    for each query block [b, b+B) of the rows that need queries:
        wq_a  fp8 GEMM [1280 x 5120]  -> q_norm -> wq_b fp8 GEMM [32768 x 1280] -> RoPE(p)
        if index_source: indexer.wq_b [4096 x 1280], RoPE, fp4(32); weights_proj [32 x 5120];
                         score [B][T//r] with the completion mask; top-k per query
        window band indices (host: arithmetic, no data)
        sparse_attn(q[B], kv=[kv rows | compressed rows], idx[B][128 + k])  — score, combine
        inverse RoPE; wo_a grouped [8x1024x4096]; wo_b fp8 GEMM [5120 x 8192]
    mhc.post    : h' = post·a + comb·h                                   (hc_post, attn half)
    -- FFN half ----------------------------------------------------------------
    mhc.mix / split / pre with the FFN weights, pre_mix = attn's pre   -> x [T][5120]
    act_quant(x) once for the layer (fp8, block 32)                    -> xq [T][5120] fp16 values
    gate GEMM bf16 [384 x 5120]; host: memcpy scores, sqrt(softplus), top-6 on score+bias, weights
    host: CSR token lists per expert; order experts by shard offset; issue reads (§3)
    shared expert: fp8 GEMM w1/w3 [2304 x 5120] + SwiGLU + h act_quant + w2 [5120 x 2304] over all T
    routed experts, expert-major as they arrive (§3):  gather-by-index GEMM, scatter-add into y
    mhc.post    : h'' = post·y + comb·h'                                (hc_post, ffn half)
    pre_mix = ffn pre
final: hc_pre(h[N-1]) -> norm -> head (1.32 GB bf16, one row) -> argmax      = decode step 0's input
```

**Which rows a decoder layer carries.** At layer 20 the stream enters with N
rows, the front half consumes all of them, and the stream is then **sliced to
the last R rows** before `wq_a`. The FFN half and every later layer see R rows.
The slice is a view — `h[N-R..N)` is already contiguous — so it costs nothing.

**Why query blocks.** The per-query attention state is wide: `q` is
`[64][512]` bf16 (64 KiB), the output `[64][512]` fp32 (128 KiB), the score
plane `[64][≤640]` fp32 (160 KiB), and at an index source the indexer's score
row is `[T//r]` fp32. B = 512 bounds that at 0.18 GB whatever N is, and at
N ≤ 512 it is one block. Everything that is per-row but narrow (`kv`, `x`,
`qr`) stays whole-layer so no row is computed twice.

### 2.1 Dispatch count

Per layer at one query block: mHC 3 × 2 halves + 2 closes, `wkv` 2, the
compressor 3 (4 source layers), the indexer 6 (8 layers), Q path 3, attention
2, O path 2, gate 1, shared expert 3, routed experts **3 per expert batch**
(§3.2), act_quant 1. About 25 + 3·(expert batches) per layer, so a 64-token
prefill is on the order of 1,000 dispatches and ~40 host round trips (one per
layer boundary — the gate readback — plus the expert batches). At the
0.13–0.15 ms submit round trip p2_decode.md §5.2 measured, the host overhead is
~0.1 s: not the problem.

---

## 3. The MoE, expert-major

### 3.1 Why expert-major, and how many experts a prompt really touches

A token-major MoE would read each expert once per token that routes to it. An
expert-major one reads each **distinct** expert once per layer and runs all of
its tokens as one GEMM — which is also exactly `model.py`'s loop.

design §3.2 says "a real prompt activates nearly every expert in every layer".
That is true at 4K and **not** at 64. Measured on the 27,399-token route trace
(`traces/mixed`, windows of N consecutive tokens):

| N | distinct experts / layer (mean, min–max over layers) | fraction of 384 | mean rows / expert | largest expert's rows |
|---:|---|---:|---:|---:|
| 16 | 41.9 (30–61) | 11% | 2.3 | 12 |
| 64 | 95.4 (68–143) | 25% | 4.0 | 42 |
| 128 | 136.3 (101–194) | 35% | 5.6 | 81 |
| 256 | 178.3 (136–244) | 46% | 8.6 | 159 |
| 512 | 235.4 (189–295) | 61% | 13.1 | 282 |
| 1024 | 289.1 (246–339) | 75% | 21.2 | 480 |
| 4096 | 358.0 (326–379) | 93% | 68.7 | 1,512 |
| 16384 | 377.2 (360–384) | 98% | 260.6 | 5,516 |

(The 64-token L2 prompt itself is denser — 180 experts at layer 0, 90–150 in
the middle — because it mixes English, Chinese and code in 64 tokens.)

Two consequences for the kernel: **the row count per GEMM is small and skewed** —
at 4K the mean expert sees 69 rows and the busiest sees 1,512, i.e. 37% of the
prompt — so the GEMM has to be good at n ∈ [1, 64] and must not fall over at
n ≈ 1,500 (§5 measures n ∈ {16, 64, 256, 1024}); and **short prompts stream a
quarter of each layer, not all of it**.

### 3.2 The streaming loop, per layer

```
host   top-6 for all T tokens  ->  per expert e: rows[e] (CSR), route weights
host   E = { e : rows[e] non-empty }, sorted by (shard file, run offset)   — shard-physical order
io     for e in E: submit its 2 aligned runs (18.8 MB) at P0 into a transit slot, QD 8, 4 MiB chunks
gpu    as experts land, in batches of up to K_b experts or ~512 MB:
         job table <- {w1,s1,w3,s3,w2,s2 addresses, row offset into the CSR, n_e}
         dispatch A: gate/up GEMM + SwiGLU x route weight  over x[rows[e]]   -> h[n_e][2304]
         dispatch H: act_quant(h) fp8/32                                    (HQuant = 3's shape)
         dispatch B: down GEMM, scatter-add into y[rows[e]]
       one submit per batch; the host keeps reading while it runs
host   release transit slots of experts not kept (§3.4)
```

**The job table is what makes one command buffer hold many experts.** A stage
owns one slice of the address table (p2_attention.md §4.1), so N dispatches of
the same stage in one buffer would all see the last addresses written. Instead
the three expert pipelines read their addresses from a GPU-side job array
indexed by a push-constant job number — the same trick design §5.3's expert
pointer table plays, one level up. The CSR token list and the concatenated
`h` plane are shared by the whole batch.

**The scatter-add has no race.** Top-6 ids are distinct per token, so inside
one expert's dispatch every output row is written by exactly one workgroup; two
experts touching the same token are two dispatches separated by a barrier.

### 3.3 I/O schedule against compute schedule

The drive is 4.5 GB/s (design §9.2.1: 4 MiB × QD 4–8 saturates; random and
sequential are the same above 2 MiB). One layer's expert set is read in shard
order as one queue; nothing waits on it except the GPU batches, so reads and
compute overlap by construction:

* **read-ahead** is bounded by transit memory, not by policy. As built, the
  transit is two halves of `transit_slots` = 32 slots each (1.2 GB together):
  batch i+1's reads are issued into one half *before* batch i computes on the
  other, so the drive fills the next batch while the GPU works, and a half is
  reused two batches later. (The first design had a free-running 64-slot queue
  consumed in arrival order; the two-half form needs no slot bookkeeping and
  measured the same, because the GPU is never the one waiting — §10.)
* **the next layer's reads start** as soon as the gate of the next layer has
  been read back — which is after this layer's attention half, i.e. while this
  layer's experts are still computing. Layer L+1's top-6 needs layer L's MoE
  output, so there is no earlier moment that knows the ids; a lookahead would be
  §9.4's predictor, which measured net negative.
* The GPU is the idle one. Expert compute at the measured GEMM rate (§5) is
  ~0.2–3 ms per expert against ~4 ms of NVMe per expert, so batches are small
  and frequent: one submit per 32 experts.

### 3.4 Keep or drop: handing the decode cache what decode will need

Every expert a prefill reads is already in GPU-visible memory; keeping one costs
a copy into (or better, reading directly into) an ExpertStore slot and nothing
else. The question is which of the ~E(N)·40 experts to keep in a cache of
~5,700 slots (100 GiB, design §5.2).

Measured on the same trace, keeping the experts used by the **last W prompt
tokens** and asking how many of the next 64 decode-time accesses they cover:

| prompt N | W = 16 | W = 64 | W = 128 | W = 256 |
|---:|---:|---:|---:|---:|
| 64 | 0.564 | 0.756 | — | — |
| 512 | 0.622 | 0.822 | 0.892 | 0.946 |

So the policy is **recency by position**: keep every expert used by the last
W = 128 prompt tokens (≈ 136 experts × 40 layers = 5,440 slots — the whole
cache, which is the point), in LRU order of their last position so the decode
LRU starts with the right ordering; drop the rest. That is 0.89 of the first 64
decode steps' accesses resident at step 0, against the ≈ 0.36 a cold decode
starts at (p2_decode.md §5.2). The decoder layers' experts are exactly the last
R = 128 tokens' by construction, so for them the policy is "keep everything".

**Interface this needs from Track I** (§8.3): the prefill must be able to ask
`store::ExpertStore` for a slot to read an expert into, and to mark it resident
with a last-access stamp. Until that exists the implementation here reads into
its own transit slots and drops everything; decode then refills on demand.

---

## 4. Memory budget per stage

Per token: the residual stream is `[4][5120]` fp32 = **80 KiB**; `hc_post`
writes a second copy. Per query block B the attention scratch is `q` 64 KiB +
output 128 KiB + score plane 160 KiB + indexer score `4·(T//r)` B per query.

| stage | formula | N = 64 | 512 | 4,096 | 16,384 |
|---|---|---:|---:|---:|---:|
| residual stream, in + out | 2 · 81,920 · N | 10 MB | 84 MB | 671 MB | 2.68 GB |
| attention input x, qr, kv (layer) | (20,480 + 5,120 + 2,576) · N | 1.8 MB | 14 MB | 116 MB | 462 MB |
| query block (q, o, scores) | 355 KiB · min(N, B) | 23 MB | 182 MB | 182 MB | 182 MB |
| indexer score, ratio 1 | 4 · min(N,B) · N | 16 KB | 1 MB | 8 MB | 33 MB |
| wo_a / wo_b out | (32,768 + 20,480) · min(N, B) | 3.4 MB | 27 MB | 27 MB | 27 MB |
| FFN x, xq, y | (20,480 + 10,240 + 20,480) · N | 3.3 MB | 26 MB | 210 MB | 839 MB |
| gate scores | 1,536 · N | 0.1 MB | 0.8 MB | 6.3 MB | 25 MB |
| expert `h` for one batch (fp16 + fp8 + scale) | 3.125 · 2304 · Σ n_e | < 1 MB | 7 MB | 55 MB | 220 MB |
| expert transit (K_ra = 64 slots) | 64 · 18.8 MB | 1.2 GB | 1.2 GB | 1.2 GB | 1.2 GB |
| compressed KV + index keys (4 sources, bf16) | (1,024 + 256) · (N/2·3 + N) | 0.2 MB | 1.6 MB | 12.8 MB | 51 MB |
| window rings (40 layers) | 40 · 128 · 528 | 2.7 MB | 2.7 MB | 2.7 MB | 2.7 MB |
| pinned weights (shared with decode) | p2_decode.md §2.1 | 9.17 GiB | 9.17 GiB | 9.17 GiB | 9.17 GiB |
| **prefill-only total** | | **≈ 1.25 GB** | **≈ 1.55 GB** | **≈ 2.5 GB** | **≈ 5.7 GB** |

Nothing here is a problem on a 110 GB machine; the expert transit is the only
line that is a choice, and it trades against the decode cache 1:1 (64 slots of
5,700). The residual stream at 16K is the largest activation and still fits one
2 GiB allocation as `[N][4][5120]`; above ~26K tokens it has to be split into
two allocations (design §1.1's 2 GiB `maxMemoryAllocationSize`).

---

## 5. The GEMM kernels, and how the choice is made

design §7.13 says: "all linear layers switch to cooperative matrix GEMM, weights
decoded FP4/FP8 → fp16 when loaded into LDS". That is a hypothesis. There are two
candidate shapes and they spend the machine differently:

**(a) cooperative matrix.** `linalg.CoopMat<half, Subgroup, 16, 16, …>`
(`VK_KHR_cooperative_matrix`, m16n16k16 half/half/float on this driver). A
wave's 32 lanes are the N axis: `W[R×K] · X[K×32] → C[R×32]` is 32 tokens per
wave, one `coopMatMulAdd` per (k-tile, output tile). The weight must be **fp16
in memory** — a 16×16 tile load is a raw half load — so an FP4 expert is first
decoded into a fp16 transit buffer (2304 × 5120 × 2 B = 23.6 MB per matrix, 4×
the FP4 bytes), and every wave re-loads all of it. Per wave:
`R/16 · K/16` tile loads (46,080 for `w1`) and `K/16 · 2 · R/16` MulAdds
(92,160).

**(b) GEMV, generalised to M ≥ 16 and tiled.** The decode kernel of
kernel_p2_moe.md with M activation columns per lane, where M is a tile of
tokens: a workgroup owns (row block, token tile), reads each FP4 weight block
once and FMAs it against the tile's M columns. Weight traffic is
`(n / M) × FP4 bytes` — no decode buffer, no 4× — and per (row, block, lane) the
instruction count is `2 + 2·M` for packed fp16 x.

The fp8 attention linears are the same comparison with an fp8 weight (1 B, not
0.5 B, per element), and `wo_a` is the no-act-quant exception.

Which wins is a question about this machine's matrix instruction versus its
VALU emission and memory — exactly the kind of thing kernel_p1.md §3.2 found
reverses intuition — so it is **measured, not reasoned**: `bench/prefill_bench`
runs both on the real layer-0 expert and `wq_b`, n ∈ {16, 64, 256, 1024}, and
reports GB/s (weight bytes per second, the design §7.1 rule 2 metric) and
rows/s. §9 has the numbers and the choice.

---

## 6. Prefill attention

`sparse_attn.slang` is M = 1 compiled (docs/p3_dspark.md §6), and so is every
other attention-path kernel; none of the decode kernels can be driven at M > 1
without a rebuild of `AttnRunner`, which is Track J's. The prefill therefore has
its own:

* **band attention** — query p against `idx[p]` = its causal window plus its
  top-k compressed rows, KV = `cat([kv rows of this layer, compressed rows])`.
  The same two passes as decode (the bf16 `acc_s_cast` of p against the final
  row max forbids an online softmax), over `[B][64][128 + k]`.
* **indexer score** — `[B][32]` queries against `T//r` keys, relu, weighted head
  sum with the reference's three bf16 roundings, the completion mask, and a
  per-query top-k.
* **compressor** — one dispatch pools all complete groups (the per-element
  softmax over the ratio axis), then RMSNorm, RoPE at the group's first
  position, FP4(16, E4M3).

---

## 7. Expected time

T(N) = T_nvme + T_compute + T_host.

**NVMe.** Encoder layers stream E(N) experts, decoder layers E(R), at 18.8 MB
and 4.5 GB/s, from §3.1's table:

| N | encoder GB | decoder GB (R = 128) | total | **T_nvme** | oracle mode (decoder over N) |
|---:|---:|---:|---:|---:|---:|
| 64 | 35.9 | 35.9 | 71.7 GB | **15.9 s** | same |
| 512 | 88.5 | 51.2 | 139.7 GB | **31.0 s** | 177 GB, 39.3 s |
| 4,096 | 134.6 | 51.2 | 185.9 GB | **41.3 s** | 269 GB, 59.8 s |
| 16,384 | 141.8 | 51.2 | 193.1 GB | **42.9 s** | 284 GB, 63.0 s |

with the cache holding a fraction h of them, `T_nvme ×(1−h)`. design §3.2's
"270 GB ≈ 50 s for 4K" is the oracle-mode row; with the replay it is 186 GB.
**At 64 tokens, the prefill is 16 s of NVMe for ~0.1 s of compute** — the
short-prompt case is the one where the cache, not the kernel, decides the wait.

**Compute** (§9 fills the measured rates): per token per layer the linears are
≈ 127 M weight elements of attention, 6 × 3 × 11.8 M of routed experts plus the
shared one, ≈ 42 M of attention FMAs at a full window, ≈ 0.5 M of mHC.

---

## 8. The handoff to the decode engine

### 8.1 What the decode step consumes

Read off `runtime/kvstore.h`, `runtime/decode_state.cpp::seed_prefill` and
`runtime/engine.cpp::load_decode_state` (not edited). After a prefill of N
tokens the engine needs, per layer L:

| buffer | shape, dtype | source in the prefill | notes |
|---|---|---|---|
| `win_kv` ring | `[128][512]`, values on the fp8 E4M3 / UE8M0-32 grid (`KvStore::seed_window` re-encodes exactly) | layer L's post-RoPE post-`act_quant` `kv` rows `max(0,N-128)..N-1` | slot `p % 128` holds position p (`_window_kv`'s `cutoff` split); slots ≥ N stay zero when N < 128 |
| `cmp_cache` (L ∈ 2, 8, 14, 20) | `[N//r][512]`, values on the FP4(16, E4M3) grid, post-RoPE | the compressor's cache rows | `seed_compressed` stores bf16 |
| `index_k` (L ∈ 2, 8, 14, 20) | `[N//r][128]`, FP4(32, UE8M0) grid, post-RoPE | the indexer key cache | cannot be recovered from `cmp_cache` (pre-RoPE latent) |
| `cmp_state_kv`, `cmp_state_score` (L ∈ 2, 8, 14) | `[2][512]` fp32 each | slot 0 = the last token's `wkv`/`wgate` when N is odd; score `-inf` in every unwritten slot | decode continues at slot `N % 2` |

plus, engine-wide:

* `history` = the prompt ids (the engram hashes on it);
* `pub_index_k_` = 20 (the last kv source that published — at `start_pos == 0`
  every source publishes, so the last one is layer 20);
* the **first token** = argmax of the head on position N−1, which is decode step
  0's input at position N;
* for DSpark (Track K, not consumed yet): layers 37–39's input streams
  `h.mean(hc)` at the last ≤ 128 positions.

### 8.2 How it is handed over today, without touching `runtime/`

`Engine::load_decode_state(dir)` reads an L3-format directory. So the prefill
writes one: `index.json` with `prompt_ids` / `prefill_len` / `config`, a
`prefill` record holding the table above under the L3 names
(`Lnn.win_kv`, `Lnn.cmp_cache`, `Lnn.index_k`, `Lnn.cmp_state_kv`,
`Lnn.cmp_state_score`), and the engram tables beside it. The L3 loader insists
on `steps_exported > 0`, so for validation the reference's own step records are
copied in beside ours — which is also exactly what the teacher-forced
comparison wants. `tests/test_gpu_prefill.cpp` does this and then calls
`decode_step`.

### 8.3 What Track I should add (specified, not done here)

1. `Result<void> Engine::seed_from_prefill(const PrefillState&)` taking the
   §8.1 buffers in memory — the directory round trip is ~6 MB of JSON and bf16
   for no reason — and replacing `Engine::prefill`'s `Unimplemented`.
   `PrefillState` is `{ std::vector<uint32_t> prompt; uint32_t first_token;
   per layer: std::vector<float> win_kv (rows·512), cmp_cache, index_k,
   cmp_state_kv, cmp_state_score; }`, i.e. `gpu/vulkan/prefill_kernels.h`'s
   `PrefillHandoff` (which already has that shape).
2. `KvStoreConfig::max_context` sized from N, and `kMaxIndexPositions`
   (`runtime/decode_layer.h`, 4,096) raised: a 4,096-token prompt at ratio 1 is
   already at the limit on its first decode step.
3. `store::ExpertStore::adopt(ExpertKey, slot, last_access)` (or a
   `reserve_for_read` returning a slot to read into) so §3.4's keep policy can
   fill the cache without a second read.
4. `Engine::decode_state()`-free generation: `generate` currently requires the
   L3 export; with (1) it can start from a prefill.

---

## 9. Validation

*(Filled from `tests/test_gpu_prefill.cpp`.)*

## 10. Bench

*(Filled from `bench/prefill_bench.cpp` and `bench/results/prefill_p3.csv`.)*

## 11. Done / not done

*(Kept current.)*
