# cachedMoE

**Run DeepSeek-V4.1-Flash on a single 128 GB AMD Strix Halo PC.**

cachedMoE is a C++20 inference engine with Vulkan and Slang kernels. It runs the
**552B MoE model, plus 196B Engram parameters**, from its **510 GB native FP4/FP8
checkpoint**. Dense weights stay resident; a bounded expert cache streams the
remaining weights from NVMe. No re-quantisation, shard conversion, or repacking.

[Quick start](#quick-start) · [Performance](#measured-performance) ·
[Web chat](tools/web/README.md) · [Technical status](docs/STATUS.md) · [MIT license](LICENSE)

<p align="center"><img src="docs/img/web_ui.jpg" width="900" alt="English cachedMoE chat with a separate thought process and answer"><br>
<sub>Historical short demo: dynamic mask, DSpark k=5 / top-K=4, two drives. The 10.04 tok/s display belongs to that demo; it is neither the eight-turn benchmark nor the current default configuration.</sub></p>

## What it does

- **Native model arithmetic:** FP4 routed experts, FP8 dense weights, compressed
  sparse attention, hyper-connections, and Engram tables.
- **NVMe-backed inference:** priority reads into GPU-visible slabs, asynchronous
  expert loading, a global LRU cache, and an optional second checkpoint mirror.
- **GPU prefill and persistent sessions:** batched prompt processing, common
  prefix reuse, named conversations, and optional KV parking on disk.
- **Local web chat:** live throughput and hardware telemetry, Markdown answers,
  a separate collapsible thought process, and reasoning effort controls.
- **Experimental decode modes:** dynamic-cache miss masking and DSpark with one
  draft path verified in a single batched forward pass.

`main` is the Linux/RADV line. The older Windows implementation is maintained
separately on the [`windows` branch](https://github.com/acupof-ai/cachedMoE/tree/windows).
The tested target is AMD Strix Halo (`gfx1151`); portability to other GPUs is not established.

## Quick start

You need a 128 GB Strix Halo system, sufficient GPU-visible memory, and at least
510 GB for the checkpoint. A second NVMe with an identical checkpoint is optional.
Keep additional disk space for the build and optional KV cache.

Build on Linux with clang 22 or newer, CMake, Ninja, `shader-slang`, `spirv-tools`,
Vulkan headers, and a RADV driver with the required cooperative-matrix support.
See [build details](docs/build.md) for the environment and driver configuration.

```bash
git clone https://github.com/acupof-ai/cachedMoE.git
cd cachedMoE
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=cmake/linux-clang-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Download the checkpoint and create its address manifest. ModelScope downloads
must use a direct connection. The manifest is the only file written inside the
checkpoint directory; the original shards are read directly.

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
  modelscope download --model deepseek-ai/DeepSeek-V4.1-Flash \
    --local_dir "$HOME/models/DeepSeek-V4.1-Flash"
python3 tools/manifest.py --src "$HOME/models/DeepSeek-V4.1-Flash"
export DEEPMOE_MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash"
```

Start **one** of these clients:

```bash
python3 tools/web/launch_guarded.py   # web policy for this machine; requires AC and its mirror
python3 tools/chat.py                # terminal chat
./build/deepmoe serve --model "$DEEPMOE_MODEL_DIR"   # JSON-lines protocol
```

The guarded Linux web launcher uses the matching checkpoint mirror at
`/mnt/deepmoe2/models/DeepSeek-V4.1-Flash`, performance power mode and thermal
pause/resume at 80/72°C. For a manual exact-routing server, use
`python3 tools/web/server.py`. The web UI defaults to English; model replies
follow your prompt's language.
The initial engine load takes time; wait for its ready message. The guarded
recipe uses this machine's 5,500 total slots. For another memory layout, start
the manual server with automatic cache sizing before choosing an explicit budget.

## Measured performance

These historical measurements use Linux/RADV on a 128 GB Strix Halo PC. They
describe particular workloads, not a guaranteed rate for every prompt. The
two-drive results use an identical read-only mirror, with all 48 shards available.

| Mode and workload | Decode throughput | Conditions and evidence |
|---|---:|---|
| Exact routing, eight conversation turns | **9.20 tok/s** | Two drives, clean start without saved KV; [status ledger, run 29](docs/STATUS.md) |
| Exact routing, same eight turns | **7.61 tok/s** | One drive, clean start; [status ledger, run 28](docs/STATUS.md) |
| Dynamic mask + DSpark, eight long turns | **9.52 tok/s** | Historical power-saver run, two drives, 5,500 total cache slots, five draft tokens, target top-4 acceptance; 2,353 timed decode tokens; [long-run report](docs/mask_freeze.md) |
| Dynamic mask + DSpark, short instrumented run | **13.40 tok/s** | Historical power-saver run, two drives, same cache and speculation settings, 31 timed decode tokens; [trace report](docs/mask_async.md) |

The short and long speculative runs are different workloads. They do not establish
a speedup over exact routing at equal quality. They predate the P0-priority quality
repair and the current performance-mode selection. The later **110.142 ms/token**
plain-mask result also used power-saver; it is historical evidence, not the new
performance baseline. Current comparison costs and thermal qualifications are
in [STATUS](docs/STATUS.md#1-todays-numbers). Prompt processing is separate from
decode: an earlier exact GPU-prefill benchmark measured about **29 s at 4K** and
**45 s at 17K** tokens under its recorded cache and disk conditions. Follow-up
prefix reuse can reduce the work. See [STATUS](docs/STATUS.md) for attribution.

### Quality and experimental modes

The CLI starts with exact routing and speculation off when their flags are
omitted. On the Linux reference trace, the
current exact-path 64-step NLL is **0.622784**. The strict decode comparison is
currently **6/8**, or **7/8 with the engine's own prefill**; this is not a claim of
bitwise equality with the entire reference implementation.

`--resident-only mask --mask-cache dynamic` skips unavailable routed experts while
normal asynchronous loading and LRU updates continue. This changes the model's
output. Restoring strict P0 priority recovered mask NLL **0.835581** from the
earlier **1.360084** regression, with exact NLL unchanged. The latest Phase A
plain-mask screen scored **46/57** on generated-answer MMLU57, with two invalid
answers counted wrong. A follow-up output repeated fourgrams at **0.117878**
versus exact routing's **0.049116**. Both the 48/57 threshold and the 1.5-times
repetition gate failed. The earlier k5 score of 48/57 belongs to its historical
configuration. These small screens are not a full MMLU benchmark.

The owner accepts the current Phase A quality as the web baseline and selected
dynamic mask plus speculation. Before Phase D finishes, the guarded launcher
uses **k=2, top-K=4, ONECB on and GPU routing off**; its final k/routing choice is
pending. This acceptance does not turn failed quality gates into passes.
[Recovery evidence and policy](docs/miss_mask.md), [owner decision](docs/codex_todo.md).

`--dspark --spec-k 2 --spec-top-k 4` verifies the root and **one two-token draft
path** in one target forward. A main-path draft token is accepted when it is in
that target row's top-K. This approximate rule does not preserve the target
sampling distribution. Fixed-cache mask has produced
repetitive output; its roughly 18 tok/s result is excluded from the table above.
Weighted partial-miss waits closed **NO-GO**: both candidates failed repetition
and the required observed 20% speed gain. They add no web option or default.
[Completed comparison](docs/miss_mask.md#completed-performance-comparison-phase-c-no-go-2026-10-06).

The context capacity is **1,048,576 tokens**. A full 1M-token prompt has not passed
an end-to-end quality test, and GPU prefill also needs enough working memory.
[Adaptive-cache findings](docs/mask_freeze.md) and [DSpark results](docs/dspark_topk.md)
include the quality gates, cycle costs, and unsuccessful experiments.

## Web chat

```bash
python3 tools/web/launch_guarded.py
```

Open <http://127.0.0.1:8080/>. Enable **Thinking** to choose low (50), high (75),
maximum (100), or a custom effort from 1 to 100. Thinking streams into its own
panel; the answer appears below it. Completed panels collapse and remain available
in restored conversation history. **Ctrl+Enter** sends; **Stop** cancels.

Use **Decode** to select dynamic mask with speculation, dynamic mask with plain
decoding, or exact plain decoding for the next queued turn. Only the main path
is verified; each speculative cycle has one target forward. Existing KV is
reused when switching modes, so start **New chat** for a wholly exact history.
[Per-turn mode behavior and validation](docs/web_decode_modes.md).

The page has no CDN dependency. Conversations are stored locally, and requests
share one queued engine. A second read source is detected when its matching
manifest is present, or can be supplied with `--mirror`. Completed decode turns enqueue batched disk KV
checkpoints in the background when disk KV is enabled; [details](docs/kv_async.md).

The guarded launcher selects the interim mask/speculation policy described
above; the final Phase D choice is pending. Full configuration:
[web documentation](tools/web/README.md).

## Development and validation

```bash
ctest --test-dir build -j 1 -LE "needs-model|needs-gpu"   # CPU tests
python3 tests/run_all.py                               # Python tool gates
python3 tools/web/test_server.py                       # web settings/history
node --test tools/web/test_ui.cjs                      # streaming channel parser
```

GPU tests require the checkpoint and trace fixtures. **Run one GPU job at a time**;
the web engine also counts as a GPU job. Stop it before starting GPU benchmarks.
For numerical changes, run the oracle and long-context gates described in
[AGENTS.md](AGENTS.md) and [STATUS](docs/STATUS.md).

| Directory | Contents |
|---|---|
| `runtime/` | Token loop, sessions, KV storage, routing, speculation |
| `store/`, `storage/` | Expert cache, global LRU, priority I/O, mirror reads |
| `gpu/vulkan/`, `gpu/shaders/` | Vulkan runners and Slang kernels |
| `cpu/`, `text/`, `model/` | Reference maths, tokenizer, model config, manifest |
| `cli/`, `tools/` | Engine protocol, chat clients, oracles, trace tools, simulators |
| `bench/`, `tests/` | Benchmarks and validation suites |

Start technical work with [STATUS](docs/STATUS.md): it records measurements,
reverted experiments, and the next levers. Also see [design](docs/design.md),
[architecture](docs/architecture.md), and [build notes](docs/build.md).
Most detailed experiment reports are in Chinese. Contributions should include
the workload, hardware, quality checks, and evidence behind performance claims.

## License

The engine is [MIT licensed](LICENSE). The DeepSeek checkpoint and its bundled
reference code retain their own licenses.
