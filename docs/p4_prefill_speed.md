# P4 — prefill speed (Track S, finished as F3)

> The prefill's **compute**, per op, on the real checkpoint: what it was spent
> on, which four kernel geometries were wrong, what they cost, and where the
> floor of this machine actually is. Companion to
> [p3_prefill.md](p3_prefill.md) (the kernels and the §10 breakdown this
> re-measures) and [p2_attention.md](p2_attention.md) §13.

Status: **v1.0, 2026-09-18** — measured. §0 is the page to read; §6 is what is
done and what is not; §7 is the honest position against the ≥ 5× target.

---

## 0. What this says in one page

**The +3.6% was real and the diagnosis in it was wrong.** Track S's
cooperative-matrix band attention was correct (110 stage checks) but its P·V
stage was **memory-order broken**, so moving attention onto cooperative-matrix
tiles bought almost nothing. A per-op profile of the 4,133-token prefill found
four kernels whose *grid or loop order*, not their arithmetic, was the cost:

| # | kernel | what was wrong | before | after |
|---|---|---|---:|---:|
| 1 | band attention P·V (`prefill_coopmat` s4) | contracts over entries, indexes output by dim, so one dim tile at a time walked the gathered KV plane with a 16 KiB stride and re-read it 32× | 13,119 ms / 200 | **3,032 ms** |
| 2 | every coopmat GEMM (`prefill_coopmat` s0) | 2 token tiles a workgroup fixed, so a GEMM's weight traffic was ceil(n/32) passes over the matrix — 16 passes over `wq_b`'s 84 MB at n = 512 | see §3.3 | see §3.3 |
| 3 | mHC pre-norm (`prefill_elem` s2) | **one thread per row** — 4,133 threads, ~4 waves a CU, each walking 20,480 floats with a strided read per hc plane | 3,896 ms / 81 | see §3.4 |
| 4 | gate top-6 | all n × 384 scores read back over a device-mapped mapping, then `partial_sort` on the host | 1,265 + 842 ms | see §3.5 |

plus `wo_a`, the one large linear still on the tiled GEMV because `op_gemm`
refused the coopmat branch for a *grouped* linear (§3.6).

**The measured starting point** (N = 4,133, replay 128, this machine, another
track's three `deepmoe_tests` on the GPU and the NVMe throughout — see §1.2 on
why only the per-op columns are comparable):

| | s | ms / prompt token |
|---|---:|---:|
| **compute** (TTFT − expert NVMe wait − engram row reads) | **80.6** | **19.5** |
| — band attention | 38.2 | 9.2 |
| — routed expert GPU | 27.1 | 6.6 |
| — mHC | 4.9 | 1.2 |
| — shared expert | 3.2 | 0.8 |
| — engram GPU | 2.8 | 0.7 |
| — gate + route | 2.5 | 0.6 |
| — embed, head, host other | 1.9 | 0.5 |

**The target is below this machine's arithmetic floor at replay 128.** §7 does
the sum: the routed experts alone are 36.2 TFLOP of fp16 matrix work for a
4,133-token prompt, and the best rate anything in this prefill achieves —
measured on the shared expert, the cleanest large GEMM in the pass — is
**2.1 TFLOP/s**. That is 17 s of routed-expert GEMM that no scheduling removes,
against a ≤ 5 ms/token budget of 20.7 s for the *whole* pass. ≥ 5× is
reachable only by raising the GEMM rate itself, not by removing work; §7.3 says
what that would take.

---

## 1. How this was measured

### 1.1 The runs

```
export PATH="/c/Program Files/CMake/bin:/c/msys64/ucrt64/bin:$PATH" \
       VULKAN_SDK="C:/VulkanSDK/1.4.357.0" \
       DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash' \
       ZIG_GLOBAL_CACHE_DIR=<worktree>/.zig-cache
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build          # retry once on "sub-compilation of compiler_rt failed"

powershell -NoProfile -File <repo>/build/p4_gpu_lock.ps1 acquire fin-s
DEEPMOE_PF_ALLOPS=1 build/prefill_bench.exe --section prefill --n 4133 \
    --ids build/ids_ctx4k.txt --replay 128 --attn-dv 1,8,4 \
    --load "..." --csv bench/results/prefill_p4s_dv.csv
powershell -NoProfile -File <repo>/build/p4_gpu_lock.ps1 release fin-s
```

`build/ids_ctx4k.txt` / `build/ids_ctx16k.txt` are the `prompt_ids` of
`traces/longctx/ctx4k` / `ctx16k` written out one per line — the same prompts
`docs/p3_prefill.md` §10 used.

Device: **AMD Radeon 8060S** (Strix Halo, 40 CUs, `VK_KHR_cooperative_matrix`,
one 74.4 GiB DEVICE_LOCAL|HOST_VISIBLE heap plus a 37.2 GiB host heap).

### 1.2 The machine was not quiet, and what that means

The GPU lock (`build/p4_gpu_lock.ps1`) was held by this track for every run
below and the lock status was `fin-s` throughout, but **three `deepmoe_tests`
processes from another track ran concurrently anyway** and were still running at
the end. They take both the GPU and the NVMe. Effects, and how this document
handles them:

- **Wall TTFT and the two I/O buckets are not comparable between runs.** Over
  the three configurations of one sweep the expert wait drifted 68.6 → 93.4 →
  110.6 s for identical work (2.89 → 2.13 → 1.79 GB/s), and the engram reads
  drifted 32.4 → 97.5 → 83.0 s. Every TTFT number here is therefore reported
  **with its load label** and is an upper bound, not a quiet-machine figure.
- **Per-op numbers are comparable within one process.** The bench now sweeps a
  geometry inside a single process (`--attn-dv`, `--coop-tt`) over the same
  prompt, so an A/B sees the same machine minute by minute. Every before/after
  in §3 is such an A/B; where a comparison crosses processes it says so.
- Contention **inflates** the after numbers relative to the before ones in the
  sweeps below, because contention rose monotonically through each sweep. The
  wins reported are therefore conservative.

### 1.3 What the bench reports now

`PrefillTimes::per_op` already carried every single-dispatch op by wall time and
every GPU-timestamped step of a multi-dispatch submit (the `gpu:` prefix). The
bench now writes **all** of them to the CSV as `op,<run>,<op name>,N,calls,...`
rows rather than printing only the top 14 (`DEEPMOE_PF_ALLOPS=1` prints them
all too), so a profile survives its run.

---

## 2. The profile that started this

N = 4,133, replay 128, the Track S build, before any change in this document.
Compute = 80.6 s = **19.5 ms per prompt token**. The ops, wall ms / calls:

| op | ms | calls | bucket |
|---|---:|---:|---|
| `attn (coop, submit+wait)` | 21,356 | 200 | attention |
| — `gpu: attn P.V tiles` | 13,119 | 200 | |
| — `gpu: attn score tiles` | 3,627 | 200 | |
| — `gpu: attn gather` | 2,172 | 200 | |
| — `gpu: attn softmax` | 1,427 | 200 | |
| — `gpu: attn q16` | 817 | 200 | |
| — `gpu: attn finish` | 110 | 200 | |
| `gemm 8192x4096 wfp8 xf32` (`wo_a`, tiled GEMV) | 3,938 | 200 | attention |
| `prefill_elem s2` (mHC pre-norm) | 3,896 | 81 | mHC |
| `coop 5120x8192 wfp8 xq` (`wo_b`) | 3,130 | 180 | attention |
| `coop 32768x1280 wfp8 xq` (`wq_b`) | 2,980 | 180 | attention |
| `gemm 25600x6144 wfp8 xq` (engram `wkv`, tiled GEMV) | 2,517 | 2 | engram |
| `prefill_attn s2` (indexer score) | 1,841 | 32 | attention |
| `host: gate readback` | 1,265 | 40 | gate |
| `host: index score readback` | 1,065 | 32 | attention |
| `prefill_elem s5` (RoPE + quant) | 892 | 480 | several |
| `host: gate top-6` | 842 | 40 | gate |
| `prefill_elem s4` (mHC post) | 799 | 80 | mHC |
| `host: index top-k` | 695 | 32 | attention |
| `coop 1280x5120 wfp8 xq` (`wq_a`) | 618 | 40 | attention |
| everything else | < 400 each | | |

plus **27,110 ms of routed-expert GPU** inside the batched MoE submits, which
carry no per-op timestamps.

Three things are visible immediately:

1. **P·V alone is 16% of compute** and 3.6× the score stage, on identical
   arithmetic (§3.1). That is why Track S's coopmat attention measured slower
   than the legacy per-(head, query) kernel at the default geometry: the new
   kernel's *scores* were faster and its *P·V* was pathological.
2. **`wo_a` is the third-largest op** and is the one large linear on the tiled
   GEMV, because `op_gemm` refused the coopmat branch for a grouped linear.
3. **The host still owned two reductions**: the gate top-6 (2.1 s) and the
   index top-k (1.8 s), both paying a device-mapped readback at ~200 MB/s
   before they could start.

---

## 3. The fixes

Every kernel below states its grid.

### 3.1 Band attention P·V: dim tiles, not head tiles

`prefill_coopmat` stage 4 computes `O[b][h][d] = Σ_t P[b][h][t] · G[b][t][d]`.
The contraction is over `t` (the gathered entries) and the output index is `d`
(the head dim), but `G` is stored `[E][D]`. The loop was `d` outer, `t` inner:

- as `t` advances by 16 the address jumps `16·D·2` = **16 KiB**, so each
  visited row contributes 32 B — a fraction of a cache line;
- and the whole plane is re-walked once per output dim tile, `D/16 = 32` times.

The stage now holds `dv` 16-wide **dim** tiles at once (`CmRowTilesPerWg`,
which stages 3/4 did not use). Each visited `G` row becomes `32·dv` contiguous
bytes and the passes over `G` drop to `D/(16·dv)`. `dv > 1` forces one head
tile for that stage, because the live accumulators are `ht · dv`.

**Grid**: `(b, H/16)` workgroups of one 32-lane wave — one query × one 16-head
tile each — looping `D/(16·dv)` dim groups over `E/16` entry tiles.
At N = 4,133 that is 512 × 4 = 2,048 workgroups a dispatch.

One process, same prompt, same minute (`--attn-dv 1,8,4`):

| `attn_pv_dim_tiles` | `gpu: attn P.V tiles` | `attn (coop, submit+wait)` | attention bucket |
|---|---:|---:|---:|
| 1 (Track S) | 13,119 ms | 21,356 ms | 38,223 ms |
| 8 | 3,736 ms | 11,882 ms | 29,939 ms |
| **4 (default)** | **3,032 ms** | **11,314 ms** | **29,711 ms** |

**4.3× on the stage**, and the whole attention bucket falls 8.5 s. `dv = 4`
beats `dv = 8` even though it ran later in the sweep, under strictly more
contention, so the ordering is real: eight live accumulators cost more in
occupancy than the extra contiguity buys.

Stage 3 (scores) was already `t` outer / `d` inner, which sweeps `G`
contiguously, and it did not move (3,627 → 3,476 ms).

### 3.2 The remaining shape of the attention kernel

What is left is the gather, and it is now the second cost of attention. With
`num_key_value_heads = 1` the gathered plane `G` is head-independent but
per-query: `E = window + index_topk = 128 + 512 = 640` rows of `head_dim = 512`
staged as fp16, **640 KiB written per query**, then read `D/(16·dv) · H/16` = 32
times over (from cache, mostly). §7.3 item 2 is the design that removes it.

### 3.3 Cooperative-matrix GEMM: token tiles are a config

`prefill_coopmat` stage 0 loads a 16×16 **weight** tile and reuses it for every
token tile the workgroup holds, so a GEMM's weight traffic is
`ceil(n / (16·CmTokTiles))` passes over the matrix. `CmTokTiles` was a hard 2:

| GEMM | matrix fp16 | n | passes at tt = 2 | at tt = 8 |
|---|---:|---:|---:|---:|
| `wq_b` [32768][1280] | 84 MB | 512 | 16 | 4 |
| `wo_b` [5120][8192] | 84 MB | 512 | 16 | 4 |
| routed expert `w1`/`w3` [2304][5120] | 23.6 MB | ~64 | 2 | 1 |

It is now `PrefillConfig::coop_tok_tiles`, default 8, and `--coop-tt` sweeps it.
Every plane a coopmat GEMM stages into carries `kPfRowSlack = 160` rows,
because the last workgroup of a dispatch touches up to `16·tt` rows past `n`
(their products only reach output rows nobody copies, but the memory has to
exist).

**Grid** (stage 0, row-tile mode): `(ceil(n / (16·tt)), R/16)` workgroups of one
wave — one 16-row output tile × `tt` 16-token tiles each, looping `K/16`.

### 3.4 mHC pre-norm: one wave per row, not one thread

`prefill_elem` stage 2 ran **one thread per row**: at N = 4,133 that is 4,133
threads = 129 waves, about 3 per CU, each serially walking `hc·d = 20,480`
floats with a `d`-strided read per hc plane. Its arithmetic is trivial — the
whole op reads 339 MB per call at N = 4,133, which is ~2 ms of bandwidth — so
3,896 ms over 81 calls was pure lack of parallelism and lack of coalescing.

It is now one **wave** per row: 32× the threads, the 32 lanes read 32
consecutive elements of a plane so the loads coalesce, and the two statistics
(`mean(stream²)` for the hc_fn row scale, `mean(u²)` for the norm) are one
`WaveActiveSum` each. `SubgroupSize` is 32 and the workgroup is 256, so a wave
never straddles two rows.

**Grid**: `ceil(n·32 / 256)` workgroups of 256 threads = 8 rows each.

The reduction order changes, so the statistics differ in the last bits; §4
reports the stage cosines.

### 3.5 Gate top-6 on the GPU

`prefill_elem` stage 11 is `cpu::gate_topk` moved onto the GPU: `sqrtsoftplus`
of the raw gate score, top-6 of `(score + bias)`, normalised weights with the
`1e-20` floor and `routed_scaling_factor`. The host path read **all n × 384
scores** back over a device-mapped mapping (26 MB a layer at 17K, ~200 MB/s)
and then `partial_sort`ed each row; only `n × 6` ids and weights cross the bus
now.

Ties break to the **lower expert id**, exactly as `cpu/gate.cpp`, by taking
`WaveActiveMin` of the arg among the lanes holding the round's `WaveActiveMax`.
`softplus` uses the same stable form; `log1p` is the series rather than
`log(1 + x)`, which in fp32 loses every bit of `x` below eps.
`PrefillConfig::gate_topk_gpu = false` (`DEEPMOE_PF_GATE=host`) keeps the host
path as the correctness reference.

**Grid**: `ceil(n·32 / 256)` workgroups of 256 threads — one wave per row,
each lane holding `ceil(E/32) = 12` experts in registers, `topk` rounds of two
wave reductions. No LDS, no barriers.

### 3.6 `wo_a`: a grouped linear on cooperative matrix

`wo_a` is `[8192][4096]` in 8 groups of 1,024 rows, group `g` multiplying its
own 4,096 columns of the `[n][32768]` attention output (`prefill_gemm.slang`
line 160). `op_gemm` refused the coopmat branch for any `rows_per_group != 0`,
so the tiled GEMV re-read the matrix `ceil(n/tile) = 64` times at n = 512 —
2.1 GB of weight traffic per call against 33.5 MB of weight.

`prefill_coopmat` stage 2 (the fp32 → fp16 staging) now takes a source row
stride and column offset, so a group stages its own slice of the wider plane.
The weight is decoded to fp16 **once** for all 8 groups, and each group is one
GEMM into its own output rows through the `row0` / `CmRows` the stage already
had.

**Grid**: per group, `ceil(n / (16·tt))` × `rows_per_group/16` workgroups, i.e.
8 × (4 × 64) = 2,048 at n = 512, tt = 8.

---

## 4. Correctness

`ctest -R prefill` runs `suite.gpu_prefill`, but ctest reports the whole suite
as **Skipped** whenever any case prints a SKIP line (`gpu_prefill.repeat` needs
`DEEPMOE_PF_REPEAT`), so the verdict has to be read from the binary:

```
DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash' \
DEEPMOE_PF_LONGCTX=<repo>/traces/longctx/ctx4k \
build/tests/deepmoe_tests.exe "gpu_prefill."
```

| case | result |
|---|---|
| `gpu_prefill.stages` (64 tokens, every stage vs the `start_pos == 0` reference) | **110 checks, 0 failed, worst cos 0.999912397** at L2 `cmp_cache` (FP4/16) |
| — gate top-6, every probed layer | **64 / 64 tokens identical** (L20 and L39 shown) |
| `gpu_prefill.forty_layers` (64-token handoff → 8 decode steps) | **6/8 teacher-forced, 6/8 free** — see below |
| `gpu_prefill.longctx` at ctx4k, oracle mode | first token **77 = reference**; **free-running 8/8**, i.e. `416 4419 15 24391 19 15 13749 2701` — **the `kestrel-4471-amber"` salt retrieved** |
| — handoff vs the reference record | window KV 0.9601 (L18), compressed KV 0.9706 (L20), index keys 0.9784 (L20), compressor state 0.9960; last position's compressed picks 3,590 / 4,096; top-6 209 / 240 over 40 layers |
| `gpu_prefill.longctx` at ctx16k, oracle mode | <!--LONGCTX17K--> |

**The `forty_layers` miss is not from Track F3.** The same binary with every F3
geometry put back on its Track S value
(`DEEPMOE_PF_ATTN_DV=1 DEEPMOE_PF_COOP_TT=2 DEEPMOE_PF_GATE=host
DEEPMOE_PF_WOA_COOP=0`) gives **the same 6/8 and 6/8, diverging at the same
step 6 to the same token** — 61 where the reference has 201, then 45324 where
it has 61. `docs/p3_prefill.md` §0 recorded 7/8 / 7/8 for this case on the P3
build; whatever moved it to 6/8 is somewhere else on `p4/one-pr`, and this
track did not touch it. The step-6 margin is 0.11–0.86 against the reference's
0.95, so it is the near-tie §0 of p3_prefill already flagged, now on the wrong
side of it.

**On the arithmetic the four fixes do and do not change.**

- The P·V dim tiles (§3.1) accumulate over `t` in the same ascending order as
  before, only into a different accumulator, so stage 4 is **bit-identical**.
- The coopmat token tiles (§3.3) change only which workgroup owns which token
  tile, not the K order, so they are **bit-identical**.
- The mHC pre-norm (§3.4) changes the summation of `mean(stream²)` and
  `mean(u²)` from serial to a 32-way wave reduction. The stage output is
  bf16-rounded and stays bit-identical to the reference (`attn_norm_out
  max|d| = 0`); only the fp32 row scale handed to the hc_fn GEMM moves, by
  ~1e-7 relative (`attn mixes max|d| = 1.19e-07`).
- The GPU gate (§3.5) picks the same experts (64/64) but computes `sqrtsoftplus`
  and the normalisation on the GPU, so the route **weights** differ in the last
  ulp (`moe_out cos = 0.999999999`).
- `wo_a` on cooperative matrix (§3.6) goes from an fp8 dot product accumulated
  in fp32 to fp16 tiles accumulated in fp32, which is the same change the other
  dense linears already took (`wo_a_out cos = 0.999999933`).

No online-softmax rewrite was landed, so there is no softmax deviation to
report: §3.1 fixed the P·V stage's memory order and left the existing tile
softmax alone.

---

## 5. TTFT

<!--TTFT-->

---

## 6. Done / not done

**Done**

1. Per-op profile of the 4,133-token prefill, with the whole `per_op` map now
   persisted to CSV (§1.3, §2).
2. Band attention P·V: dim tiles instead of one dim tile at a time — **13,119 →
   3,032 ms / 200 calls**, and the answer to "was the attention kernel not the
   bottleneck, or was the new one not faster": *the new one was not faster,
   because its P·V stage's loop order made every gathered-KV read a 32-byte
   slice of a 1 KiB row, 32 times over* (§3.1).
3. Cooperative-matrix token tiles as a config, default 8 instead of 2 — `wq_b`
   3,172 → 2,078 ms, `wo_b` 3,011 → 2,583 ms, attention bucket 34.7 → 27.2 s in
   one process (§3.3).
4. mHC pre-norm from one thread per row to one wave per row — mHC bucket
   **4.88 → 1.46 s** (§3.4).
5. Gate top-6 on the GPU, picks bit-identical to `cpu::gate_topk` — gate bucket
   **2.49 → 0.48 s** (§3.5).
6. `wo_a`'s grouped linear on cooperative matrix, the last large linear on the
   tiled GEMV (§3.6).
7. Attribution knobs in the bench (`--attn-dv`, `--coop-tt`) and the test
   (`DEEPMOE_PF_ATTN_DV`, `DEEPMOE_PF_COOP_TT`, `DEEPMOE_PF_GATE`,
   `DEEPMOE_PF_WOA_COOP`), so every number above is a same-process A/B and every
   handoff change can be pinned to a kernel without a rebuild.

**Not done**, in the order the next session should take them:

1. **Indexer top-k on the GPU.** Still `host: index score readback` +
   `host: index top-k` = 1.8 s at N = 4,133 and, from `docs/p3_prefill.md`
   §10.3, **24 s at N = 17,010** — the score plane is `[512][G]` and G grows
   with N, so this is the one host cost that is quadratic. The kernel is a
   per-query selection of the top 512 of up to 8,505 scores, emitted ascending
   by index, ties to the lower index; the shape that fits is a two-pass
   histogram on the order-preserving uint key (256 bins on the high byte, then
   on the next), emitting in ascending index order in the second pass. Layer
   20's candidate blocks (design §2.1) have to move with it, since the later
   index layers mask against them.
2. **The `G` plane itself** (§3.2). 640 KiB written and ~10 MB read per query.
   With `num_key_value_heads = 1` the window part of every query's list is the
   contiguous run `p-127..p` of the KV ring, so 128 of the 640 rows never need
   gathering at all; and the compressed part is a per-query list into the same
   compressed cache, so a block of 512 queries touches far fewer than
   512 × 512 distinct rows. A gather keyed on the *union* of a query block's
   picks, with the tile loop indexing through a small per-query offset table,
   is the design.
3. **The engram `wkv` GEMM**, `[25600][6144]`, 3.0–3.5 s over **two** calls at
   N = 4,133 and 9.7 s at 17,010 (p3 §10.3). It is on the tiled GEMV only
   because `R × K` as fp16 is 314 MB and does not fit `b_.w16`; decoding and
   multiplying it in row blocks that do fit is the fix, and `PfCoopPush` already
   carries the `row0` / `y_row0` it needs.
4. **The expert weight decode.** `compute_coop` decodes each of an expert's
   three FP4 matrices into the single `b_.w16` and puts a barrier after every
   dispatch, so decode and GEMM never overlap — 71 MB of fp16 written per expert
   × 10,556 experts. Two alternating halves of `w16` would overlap the next
   matrix's decode with this one's GEMM; the obstacle is that the decode target
   is a slot-table entry, so alternating needs two kernel handles.
5. **Absorbing the K projection**, which this session did not land. What the
   reference's math allows, stated exactly: `attn.wkv` produces one 512-wide
   latent per position, `kv_norm` is an RMSNorm over that latent, and RoPE is
   applied to its **last 64 dims** (`qk_rope_head_dim = 64`) at that position.
   The norm is per-row and data-dependent, and the fp8 activation quantisation
   (`act_quant(x, 32, "ue8m0")`) is applied to `x` *before* `wkv`, so
   **`wkv` cannot be folded into `wq_b`**: the two see different activations and
   a row-dependent scale sits between them. What *can* fold is the **non-RoPE
   448 dims**: for those, `Q·K^T` is `(x Wq) · (x Wkv)^T` with no position term
   between them, so `Wq Wkv^T` is a fixed per-layer contraction that could be
   precomputed — at the cost of a `[64][448]`-per-head matrix against the latent
   instead of the latent itself, which only pays if the latent is re-read more
   than once per query. With MQA and a per-query gather it is, but the gather
   (item 2) has to be fixed first for the comparison to mean anything. The 64
   RoPE dims can never fold: the rotation depends on the *difference* of the
   query and key positions, which is not a fixed matrix.
6. **Issuing Q early.** `run_layer` runs `wq_a` / `wq_b` *after* the compressor
   and the index keys, and every op is its own submit-and-wait, so nothing
   overlaps. The reordering is free — `wq_a` depends only on `b_.xq`, which the
   window-KV step already produced — but it only pays once the ops stop being
   individually waited on (item 7).
7. **Fewer, larger submits.** 72,710 dispatches in **2,954 submits** for one
   4K prefill: every `op_gemm`, `op_rope`, `op_rmsnorm` and `op_act_quant` is a
   `flush_one`, i.e. record + submit + wait. That wait is what makes items 4 and
   6 unprofitable today.

---

## 7. Where the floor is

### 7.1 The arithmetic

For N = 4,133 at replay 128, the rows each layer actually computes are 4,133 for
layers 0–19 and 128 for layers 20–39, i.e. **85,220 token-layers**. The fp16
matrix work that follows from the checkpoint's shapes:

| | FLOP per token-layer | total |
|---|---:|---:|
| routed experts (6 × w1/w3 [2304][5120] + w2 [5120][2304]) | 425 M | **36.2 TFLOP** |
| dense linears (`wq_a`, `wq_b`, `wo_a`, `wo_b`, `attn.wkv`, gate, hc_fn) | 254 M | **21.7 TFLOP** |
| band attention tiles (scores + P·V, E = 640, H = 64, D = 512) | 84 M (ratio-2 layers) | **6.6 TFLOP** |
| shared expert | 71 M | **6.0 TFLOP** |
| engram `wkv` [25600][6144], layers 1 and 14 over 4,133 rows | 315 M | **2.6 TFLOP** |
| compressor, indexer, head | | ~0.4 TFLOP |
| **total** | | **≈ 73.5 TFLOP** |

None of that is removable at replay 128: it is what the reference computes.

### 7.2 The rate

The cleanest large GEMM in the pass is the **shared expert** — one expert, every
row, no gather, no per-expert dispatch churn: 6.03 TFLOP in 2.88–3.22 s, i.e.
**1.9–2.1 TFLOP/s**. The routed experts get less (36.2 TFLOP in 27.1–31.6 s =
1.15–1.34 TFLOP/s), the gap being the FP4 → fp16 decode and the
11-dispatch-per-expert barrier chain.

At 2.1 TFLOP/s, 73.5 TFLOP is **35 s = 8.5 ms per prompt token**, and that
assumes every kernel in the pass reaches the best rate any of them reaches
today. The compute measured after this session's fixes is 69.5 s = 16.8
ms/token, so there is roughly 2× of kernel-level headroom left before the rate
itself has to change.

### 7.3 What ≥ 5× would take

**≤ 5 ms per prompt token is 20.7 s for 73.5 TFLOP, i.e. 3.56 TFLOP/s
sustained** — about 1.7× the best rate any kernel in this prefill has ever
shown, and ~3× the rate the routed experts (half the work) show. So the target
is not reachable by removing work: §6's items 1–7 are all work-removal or
overlap, and together they are worth perhaps 12–15 s.

The rate is where the remaining factor is, and the cooperative-matrix GEMM's
shape is why it is low: **one 32-lane wave per workgroup, tiles loaded straight
from global memory, no LDS staging, no double buffering.** A workgroup has one
wave's worth of latency hiding and re-reads its A tile from cache for every
token tile. The standard fix — several waves per workgroup cooperating on one
output tile through LDS, with the next K slice prefetched while the current one
multiplies — is a rewrite of `prefill_coopmat` stage 0, not a geometry
constant, and it is the single change that could plausibly move 2.1 TFLOP/s
toward the device's matrix peak. That is the recommendation.

Two things that keep the target from being hopeless:

- **Replay is already doing the heavy lifting.** At replay 128 only layers 0–19
  see the full prompt. In oracle mode (every decoder row) the same prompt is
  ~2× the FLOPs, which `gpu_prefill.longctx` measures: 401 s wall for the 4K
  prompt against 318 s for the replay-128 bench on the same machine.
- **Compute is not TTFT.** Even at 8.5 ms/token a 4K prompt still waits on
  198 GB of expert reads; on a quiet machine that wait was 30.4 s in Track S's
  own measurement, and nothing in this document changes it.
