# deepMoE

Run **DeepSeek-V4.1-Flash** — a 552B-parameter MoE with 196B of Engram tables, 510 GB of native
FP4/FP8 weights — on a single AMD Strix Halo machine (Ryzen AI Max+ 395, Radeon 8060S iGPU,
128 GB unified memory) by streaming routed experts from NVMe.

No re-quantisation and no repacking: the engine reads the checkpoint's 48 safetensors shards in
place, keeps ~9 GiB resident, caches ~96 GiB of experts and fetches the rest on demand.
C++20, Vulkan compute, shaders in [Slang](https://shader-slang.org).

<p align="center"><img src="docs/img/web_ui.jpg" width="540" alt="deepMoE web chat showing 11.96 tok/s in mask mode"></p>

The screenshot records a single-drive chat at **11.96 tok/s** with `--resident-only mask`,
5,500 cache slots, balanced power mode and DSpark off. It was captured from the existing
page using the earlier mask version, which kept LRU and asynchronous P0 fills.
Mask now defaults to dynamic LRU and asynchronous P0 fills. Use
`--resident-only mask --mask-cache fixed` to freeze the initial expert set explicitly.
Missing routed experts have zero weight; the shared expert always runs. Mask changes
the output distribution and can degrade long-context answers.

## Performance

Linux (Arch, kernel 7.2, Mesa 26.2 RADV), an internal NVMe (4.8 GB/s) plus a copy of the model
on a USB4 drive (3.7 GB/s) that `serve` finds under `/mnt` and reads striped per chunk, 5,500-slot
expert cache.

| | |
|---|---|
| **Chat decode** (8-turn script, cold start) | **9.20 tok/s** — per token 63 ms GPU compute + 42 ms waiting on the drives; expert cache hit rate 0.943 |
| … internal drive alone | **7.61 tok/s** — 66 ms waiting |
| Decode step, all experts resident | **66.0 ms** (the weight-read floor is 56.6 ms) |
| GPU prefill | 4,133 tokens in 29 s · 17,010 tokens in 45 s (both drives, cache at 4,900 slots, §7 0bb; 43 s with `--transit-ring 6` at 4,600 slots, §7 0bb; the internal drive alone last measured at 48.7 / 69.75 s, §7 0ag) — 4K waits on the drives, 17K on the GPU |
| Context | up to 1,048,576 tokens; index-score/KV boundaries checked, full 1M prompt quality not tested |
| Quality | 64-step teacher-forced NLL 0.623 against 0.598 for the fp32 reference; needle retrieval at 4K and 17K tokens 8/8 |

Every number above is machine-recorded with its commit in [docs/STATUS.md](docs/STATUS.md) §1
(ledger rows 19, 28 and 29), §7 0t–0bb (prefill, with its per-op cost model) and
[docs/p3_longctx_decode.md](docs/p3_longctx_decode.md) §4.3.

**Optional miss masking.** Historical single-drive, AC-connected, 64-token runs with no thermal
pause reached **11.88 tok/s in power-saver** and **13.23 tok/s in balanced mode**. These used
the earlier dynamic-cache mask and differ from the exact dual-drive chat above. The small generative
MMLU check scored **48/57** (one question per subject, zero-shot, at most 16 generated tokens);
this is not a full standard MMLU score or a guarantee for long conversations. Controlled
receipts and the speculation work are in [docs/dspark_topk.md](docs/dspark_topk.md).

The current experimental DSpark path keeps independent kernels in one draft command
buffer and verifies the main path with one GPU routing snapshot and one submission.
In the earlier dynamic-cache single-drive power-saver 64-token workload, ordinary mask measured
**12.33 tok/s (81.10 ms/output)** and DSpark k=2 measured **13.49 tok/s (74.12 ms/output)**.
Enable both `DEEPMOE_DSPARK_ONECB=1` and `DEEPMOE_BATCH_GPU_ROUTE=1` with
`--resident-only mask --dspark --spec-k 2 --spec-top-k 4`; both switches remain off
by default. Mask uses dynamic LRU by default; `--mask-cache` overrides the legacy
`DEEPMOE_MASK_DYNAMIC_LRU` switch (`0` selects fixed, `1` selects dynamic).
Dual-drive checks and the fixed-cache results are recorded separately. Conditions and quality
receipts are in [the execution report](docs/dspark_topk.md#14-按端到端方案执行草稿-onecb-与验证-gpu-快照路由2026-10-05).

The fixed-cache dual-drive mask measured **18.09 tok/s** and DSpark k=2 measured
**18.49 tok/s** with profiling off. The ordinary mask only served about **37.5%**
of routed expert requests and produced repeated text in the Chinese chat test.
This speed does not establish useful answer quality. Fixed-cache quality and the
projection/head experiment are recorded in [the latest report](docs/dspark_topk.md#15-双盘验收固定初始-cache-与实际-phase-4-尝试2026-10-05).
The final fixed-cache generated MMLU sample scored **48/57**, with one invalid answer;
it does not establish quality for arbitrary long conversations.
The proposed adaptive freeze did not pass its offline usefulness check: the
recorded eight-turn demand-LRU replay has no eligible freeze point at the requested
churn threshold. See [the calibration report](docs/mask_freeze.md). Automatic
freezing is not enabled; fixed cache remains an explicit experiment.
The latest dynamic-cache k=5 eight-turn run measured **9.52 tok/s (105.02 ms/decode token)**,
with one target submission per cycle. Draft / verify averaged 33.20 / 379.45 ms per
cycle. Its generated MMLU sample scored **48/57**, with two invalid answers; cold
l3 mask NLL was **1.360084**, so this is not a general quality pass. See the
[current validation](docs/mask_freeze.md#本机动态-mask-验证) for conditions and limits.

The web server accepts `--dspark --spec-k 5 --spec-top-k 4` with the same two
environment switches above. The page displays the engine's actual enabled state,
draft length and acceptance K; a default draft length alone does not mean speculation
is on. DSpark's native block has five draft tokens. Each full cycle verifies the
root plus that one draft path in one six-row forward; rejected suffix rows do not
become output tokens. Speculation stays off unless explicitly enabled.

Enable **思考** to choose the checkpoint's native reasoning effort: low 50,
high 75 (default), maximum 100, or a custom integer from 1 to 100. This budget is
rendered into the thinking prompt and saved with the conversation. Changing it
can change the reusable prompt prefix and require prefill again.

A sustained web test generated three 512-token turns per configuration. Fixed-cache
mask measured **13.53 tok/s**, versus **16.26 tok/s** for k=5: +20.2% in decode,
or +10.3% including prefill. Both outputs showed repetition; the speculative English
tail entered a two-token loop. This is not a speedup at established answer quality.
The [long-test report](docs/dspark_topk.md#17-k5-网页长测吞吐提升重复质量未通过2026-10-05)
records the workload, acceptance, cycle costs and thermal checks.

GPU prefill currently uses the exact streaming path, including in mask mode. The captured
chat's 2,835-token prefill took **39.6 s**; expert I/O took **29.7 s** and read **151.28 GiB**.
The decode speed shown in the screenshot does not apply to prefill.

**Where the time goes.** Each token routes to 6 of 384 experts in each of 40 layers. About 13.5 of
those 240 lookups miss the cache and cost one ~19 MB read each, so half of every token is disk.
The other half is compute, ~17% above its memory-bandwidth floor. The things that did
*not* help — 2/3-bit re-quantisation, prefetching, the earlier speculative decoding paths and resident-only routing policies,
persistent dispatch and some 60 others — are listed with their measurements in STATUS §3.

## How it works

- **Weights stay where they are.** `tools/manifest.py` writes one address book
  (`deepmoe_manifest.json`) next to the shards; experts are read straight into GPU-visible
  slabs with `io_uring` (Linux) or IOCP (Windows), `O_DIRECT`, at a per-request priority.
- **An exact global-LRU expert cache.** The planner is step-for-step identical to the offline
  simulator (`tools/cache_sim.py`) it was tuned with; the dense layers, shared experts and
  embeddings (9.2 GiB) stay pinned.
- **The model's own arithmetic.** FP4 E2M1 experts and FP8 E4M3 dense weights are decoded in the
  kernels (FP4 by bit placement straight into fp16); every layer type of V4.1 is implemented as
  in the reference `inference/model.py`: sliding-window-only, ratio-2 and ratio-1 compressed
  attention with a sparse indexer, compressed KV and index keys shared from four source layers,
  four-way hyper-connection residual streams, and Engram n-gram tables at layers 1 and 14.
- **Decode** is one command buffer per layer, cut only where the host reads the MoE gate;
  attention runs as two cooperative-matrix GEMMs over the 64 heads.
- **Prefill** runs on the GPU as batched GEMMs; for long prompts layers 21–39 replay only the
  final 128 rows, the ones decode reads. A layer over 1,024 rows or more starts reading its
  routed experts before its gate has run (nearly all 384 are used at that size), so the drive
  works through the attention instead of waiting for it.
- **Sessions.** The chat server keeps named sessions, reuses the common prefix across turns and
  can park a context's KV on disk.

## Quick start

**Build (Linux).** clang ≥ 22, `shader-slang`, `spirv-tools`, `vulkan-headers`, CMake, Ninja.

```bash
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/linux-clang-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j 1 -LE "needs-model|needs-gpu"      # CPU gate, 25 tests, ~3 s
```

**Build (Windows).** The same with `-DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake`
(zig as the compiler, Vulkan SDK 1.4); see [docs/build.md](docs/build.md).

**Model.** Download `deepseek-ai/DeepSeek-V4.1-Flash` from ModelScope (510 GB), then write the
address book — the only file deepMoE ever writes into the model directory:

```bash
modelscope download --model deepseek-ai/DeepSeek-V4.1-Flash --local_dir ~/models/DeepSeek-V4.1-Flash
python tools/manifest.py --src ~/models/DeepSeek-V4.1-Flash
```

**Chat.**

```bash
python tools/web/server.py        # web UI at http://127.0.0.1:8080, one session per tab
python tools/chat.py              # terminal chat: /reset /think /temp /top_p /max /stats ...
./build/deepmoe serve --model ~/models/DeepSeek-V4.1-Flash   # the JSON-lines engine both of them drive
```

## Correctness

Oracles come from the **unmodified** reference `inference/model.py`, run on the CPU behind small
kernel shims (`tools/dsref.py`): L1 per kernel, L2 per stage of one layer, L3 the whole model.
The engine is gated against them at 64 tokens and at 4K / 17K-token contexts, with tie-aware
top-k comparisons where the reference itself has exact score ties.

```bash
export DEEPMOE_MODEL_DIR=~/models/DeepSeek-V4.1-Flash
ctest --test-dir build -j 1                  # 46 tests; the GPU ones need the checkpoint
python tests/run_all.py                      # 30 Python gates (tokenizer, sampling, tools)
python tools/l3_ppl.py --modes off           # the 64-step NLL ruler
```

Run one GPU job at a time.

## Repository

| | |
|---|---|
| `runtime/` | engine and token loop, sessions, KV store, decode layer, MoE bridge, multi-stream decode |
| `store/`, `storage/` | expert slab pool and LRU planner; priority I/O engine, second read source (`--mirror`) |
| `gpu/vulkan/`, `gpu/shaders/` | Vulkan runners and Slang kernels: MoE, attention, indexer, prefill, sampling |
| `cpu/`, `text/`, `model/` | reference maths, the tokenizer (100% agreement with HF `tokenizers`), config and manifest |
| `cli/` | `deepmoe info / run / serve / tokenize / bench` |
| `tools/` | oracles, cache and hit-rate simulators, chat clients, trace and roofline tools |
| `bench/`, `tests/` | benchmarks and probes; unit, oracle and end-to-end tests |

## Documentation

- [docs/STATUS.md](docs/STATUS.md) — **start here**: today's numbers and their source, every
  optimisation attributed, every experiment tried and reverted, the next steps in order.
- [docs/design.md](docs/design.md) · [docs/architecture.md](docs/architecture.md) ·
  [docs/build.md](docs/build.md) — design, module structure and threading, build and environment.
- `docs/p*_*.md` — one report per track (kernels, attention, prefill, long context, cache policy, …).

Most of the documentation is written in Chinese.
