# deepMoE — working rules

Windows Strix Halo runtime for DeepSeek-V4.1-Flash (552B MoE, 510 GB, native FP4/FP8, no
re-quantisation). C++20 + Vulkan/Slang, cross-compiled with zig. Repo:
`git@github.com:acupof-ai/cachedMoE.git`.

## Read first

`docs/STATUS.md` is the single entry point: today's numbers with their source, the optimisation
path attributed step by step, **every experiment tried and reverted with its numbers**, the test
suites, and the next levers in order. Check it before trusting any number, and before proposing
anything — its §7 "不做" list exists so closed questions stay closed.

## Hard rules

- **One GPU job at a time.** `deepmoe serve`, `ctest`, any bench — serialise them. The web UI's
  engine counts as one: stop it (`tools/web/RUNNING.txt` has the PIDs and the launch command)
  before running anything else on the GPU, and start it again afterwards. Three concurrent
  engines froze the machine once.
- **Never write under `D:\models`** (the checkpoint). The one exception already made is
  `deepmoe_manifest.json`. `E:\models\...` is a read-only mirror of the same checkpoint.
- **Never route ModelScope through the proxy.** Clear `HTTP_PROXY`/`HTTPS_PROXY` for downloads.
- **No attribution lines** in commits or PR bodies.
- **Experiments answer in minutes, not hours.** Smallest N first; kill a run whose verdict is
  already decided. Pair A/B alternating (ABAB), ±3% jitter floor, and **halve every predicted
  gain** before believing it.
- Worktrees: one per track under `C:\Users\Asus\code\deepmoe-<track>`; merge → verify → push →
  **delete the worktree and branch**.

## Build and test

```bash
export PATH="/c/Program Files/CMake/bin:/c/msys64/ucrt64/bin:$PATH" \
       VULKAN_SDK="C:/VulkanSDK/1.4.357.0" \
       DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash' \
       DEEPMOE_LONGCTX_DIR=C:/Users/Asus/code/deepmoe/traces/longctx \
       ZIG_GLOBAL_CACHE_DIR=C:/Users/Asus/code/zig-cache-int
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j 1 -LE "needs-model|needs-gpu"   # CPU gate, ~4 s, 25/25
.venv/Scripts/python.exe tests/run_all.py                  # 29/29 gates
ctest --test-dir build -j 1                                # full, needs the checkpoint + GPU
```

- A zig `compiler_rt`/`libcxxabi` sub-compilation failure is a cache race — just rerun.
- `lld-link: failed to write output 'deepmoe.exe': Permission denied` means an engine is running.
- Per-worktree `ZIG_GLOBAL_CACHE_DIR`, or concurrent builds race.
- Quality gate for anything touching numerics: `tools/l3_ppl.py` in mode `off` must reproduce
  the platform's current NLL on `traces/l3_64` bit for bit — **0.630051** on the Windows zig build,
  **0.622784** on Linux RADV (2026-09-29; the latest STATUS §7 entry that moved it says why) —
  plus `suite.decode` 8/8 + 8/8.

## Web chat UI

`tools/web/server.py` on http://127.0.0.1:8080 (launch command + live PIDs in
`tools/web/RUNNING.txt`; the desktop app can also start it from `.claude/launch.json`). It
launches its own `deepmoe serve`, auto-detects the second read source on `E:`, keeps one named
session per browser tab, and persists each transcript under
`%LOCALAPPDATA%\deepmoe\web_chat\`. Context ceiling 524,280 tokens (indexer dispatch limit).

## Where things are

| | |
|---|---|
| `runtime/` | engine, token loop, sessions, KV store, streams (multi-stream decode), resident routing, speculation scaffolding |
| `store/`, `storage/` | expert slab pool + global-LRU planner; priority IO engine (P0 miss … P3 backfill), multi-source reads (`--mirror`) |
| `gpu/vulkan`, `gpu/shaders` | runners and Slang kernels: MoE, attention, decode chain, prefill, DSpark draft |
| `cpu/`, `text/`, `model/` | reference maths, tokenizer, manifest |
| `tools/` | oracles, simulators (`cache_sim`, `hitrate_sim`, …), `chat.py`, `web/`, `l3_ppl.py`, `trace_timeline.py` |
| `bench/`, `docs/` | benchmarks and probes; one report per track, all conclusions folded into `STATUS.md` |

Data that is gitignored and lives only in the main checkout: `traces/` (routing traces, long-context
and 64-step exports), `bench/results/` large runs, `.venv/`.
