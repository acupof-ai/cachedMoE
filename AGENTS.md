# deepMoE — working rules

Strix Halo runtime for DeepSeek-V4.1-Flash (552B MoE, 510 GB, native FP4/FP8, no
re-quantisation). C++20 + Vulkan/Slang. Repo: `git@github.com:acupof-ai/cachedMoE.git`.

`main` is the **Linux** line (machine x, Omarchy, RADV). The Windows mainline it came from is
kept on the `windows` branch and is not merged any more, so anything below that says "Windows"
is there to explain a number, not to be run here.

## Read first

`docs/STATUS.md` is the single entry point: today's numbers with their source, the optimisation
path attributed step by step, **every experiment tried and reverted with its numbers**, the test
suites, and the next levers in order. Check it before trusting any number, and before proposing
anything — its §7 "不做" list exists so closed questions stay closed.

## Hard rules

- **One GPU job at a time.** `deepmoe serve`, `ctest`, any bench — serialise them. The web UI's
  engine counts as one: stop it (`tools/web/RUNNING.txt` has the launch command and how to find
  the live PIDs) before running anything else on the GPU, and start it again afterwards. Three
  concurrent engines froze the machine once.
- **Never write under the checkpoint** — `~/models/DeepSeek-V4.1-Flash` here. The one exception
  already made is `deepmoe_manifest.json`. `/mnt/deepmoe2/models/...` is a read-only mirror of
  the same checkpoint.
- **Never route ModelScope through the proxy.** Clear `HTTP_PROXY`/`HTTPS_PROXY` for downloads.
- **No attribution lines** in commits or PR bodies.
- **Experiments answer in minutes, not hours.** Smallest N first; kill a run whose verdict is
  already decided. **One run per configuration, keep the best, and record it** (the owner,
  2026-09-29: no repeated A/B). An effect near the ±3% jitter floor is judged on the
  machine-recorded per-op numbers, not by repeating whole runs. **Halve every predicted gain**
  before believing it.
- Worktrees: one per track next to the checkout, `../deepmoe-<track>`; merge → verify → push →
  **delete the worktree and branch**.

## Build and test

```bash
export DEEPMOE_MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash" \
       DEEPMOE_LONGCTX_DIR="$PWD/traces/longctx"
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/linux-clang-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build -j 1 -LE "needs-model|needs-gpu"   # CPU gate, ~4 s, 25/25
.venv/bin/python tests/run_all.py                          # 30/30 gates
ctest --test-dir build -j 1                                # full, needs the checkpoint + GPU
```

`slangc` comes from the `shader-slang` package. The second read source is
`/mnt/deepmoe2/models/DeepSeek-V4.1-Flash`, mounted read-only; pass it as
`DEEPMOE_MODEL_MIRRORS` (or `--mirror`) for the two-drive numbers, which are the default
reporting basis.

- One worktree, one build directory, one toolchain cache — concurrent builds in a shared cache race.
- `failed to write output` on the engine binary means an engine is still running.
- Quality gate for anything touching numerics: `tools/l3_ppl.py` in mode `off` must reproduce
  the platform's current NLL on `traces/l3_64` bit for bit — **0.622784** here on Linux RADV
  (2026-09-29; the latest STATUS §7 entry that moved it says why; the Windows zig build's figure
  was **0.630051**) — plus `suite.decode` 8/8 + 8/8.
- The Windows recipe (zig cross-compile, `cmake/zig-toolchain.cmake`, the MSYS2 and Vulkan SDK
  paths) is on the `windows` branch.

## Web chat UI

`tools/web/server.py` on http://127.0.0.1:8080 (launch command + live PIDs in
`tools/web/RUNNING.txt`; the desktop app can also start it from `.Codex/launch.json`). It
launches its own `deepmoe serve`, auto-detects the second read source on `E:`, keeps one named
session per browser tab, and persists each transcript under
`%LOCALAPPDATA%\deepmoe\web_chat\`. Context ceiling 1,048,576 tokens (native checkpoint cap; index scores tile X/Z).

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
