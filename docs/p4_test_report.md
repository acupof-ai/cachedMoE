# P4 integration build/test report (2026-09-16)

Branch `p4/one-pr` (this PR). This is the first round in which the merged tree was
actually compiled and run. Environment: Windows Strix Halo, CMake 4.3 + Ninja,
zig toolchain (`cmake/zig-toolchain.cmake`), Vulkan SDK 1.4.357,
`DEEPMOE_MODEL_DIR=D:\models\DeepSeek-V4.1-Flash`.

## 1. Build

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Build passes. One compile error was found and fixed during this round:
`runtime/engine.cpp` used `store::ExpertKey`; `ExpertKey` lives in `deepmoe::`
(`core/types.h`).

## 2. Tests that pass

| Test | Result |
|---|---|
| `ctest -LE "needs-model\|needs-gpu"` | **21/21 passed** (unit; includes new `suite.kvdisk`) |
| `deepmoe_tests.exe kvdisk.roundtrip` | **pass** — SSD parked-context save/load/drop round-trip |
| `suite.tokenizer`, `suite.engram_tables` | **pass** |
| `gpu_layer.mgt1_layer_batch_vs_steps` | **pass** — M=2/4/6 batch vs M=1 steps, gate ids and ring/overflow agree |
| `gpu_layer.decode_layer_vs_oracle` | **pass** — chained block cos ≈0.9999, gate 6/6 |
| `gpu_prefill.stages` | **pass** — 110 checks, worst cos 0.999912 (L2 cmp_cache) |
| `kv_replay.l3_64` | **pass** — (1) 8/8; (3) restore ≈52 s + 8/8 / 8/8; (4) 0 raw rows |
| `decode.forty_layers_against_the_l3_oracle` | **pass** — from our own slow prefill: teacher-forced 8/8, free-running 8/8 |

## 3. Fixes that came out of this round

1. **compressor E4M3 scale round-trip** (`gpu/shaders/compressor.slang`,
   `gpu/shaders/mgt1_cmp.slang`). The shaders computed the FP4 scale with
   `fp8_round()` but wrote `fp8_encode_rn(scale)`; under the shader compiler the
   value and the byte could disagree in the E4M3 subnormal range. `KvStore::pack`
   then could not recover the grid: `kv_replay` reported 9 raw rows (merged) /
   6 raw rows (R2 baseline) instead of 0. Both shaders now use
   `fp8_decode(fp8_encode_rn(...))`, so the value is exactly the byte.
2. **Reverted the optional M=1 K-split/tiled attention adoption** from Track T's
   final WIP (`runtime/decode_layer.{h,cpp}` back to `b4f0e83`). With it merged,
   `kv_replay.l3_64` scenario (1) and `decode.forty_layers` both dropped to 7/8; Track T's actual
   deliverables (M>1 kernels, `MgtRunner`, batch path, gate softplus fix,
   `gpu_layer.mgt1_layer_batch_vs_steps`) are unchanged; only the unrequested
   adoption of Track J's M=1 interfaces was removed. `gpu_layer.mgt1_...` passes
   with the reverted file.
3. **`restore_context` timeout was contention, not a logic bug.** Under a quiet
   machine, scenario (3) replay [0,64) takes ~52 s and succeeds; with four tracks
   hammering the GPU it hit `cmd_wait`'s 120 s fence timeout. The replay path is
   still slow (~800 ms/token in the test, versus ~82 ms hot) because the expert
   cache is cold; this is exactly what R1 targets.

## 4. Still unverified / next measurements

- **S performance**: `bench/prefill_bench` N=4133 legacy vs coop on a quiet
  machine (`bench/results/prefill_p4.csv` still missing). `gpu_prefill.stages`
  correctness passes.
- **R1 A/B + auto-tune**: `MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL` A/B and
  the `--auto-tune N` cross-process curve still need a quiet-machine serve run.
  The Python helpers were unit-smoke-tested (`write_heat_from_route`,
  `round_stats`).
- **T scaling**: `bench/results/mgt1_p4.csv` (M=1/2/4/6 per-layer cost) still
  missing; `mgt1_layer_batch_vs_steps` correctness passes.
- **R2 SSD TTFT**: `suite.kvdisk` round-trips, but the end-to-end no-cache vs
  SSD-prefix-hit TTFT comparison (R=128/256) has not been run.
- **DSpark G1/G3**: after T's C(M) curve.
- `kv_replay.longctx` (4K/17K) was not run in this round.

## 5. End-to-end SSD KV TTFT (2026-09-16)

Same 4,133-token prompt, new process for the warm run (`serve --kv-dir`):

| | cold | warm SSD hit |
|---|---:|---:|
| reused tokens | 0 | **4132** |
| prefill tokens | 4133 | **1** |
| **TTFT** | **101.6 s** | **2.47 s** |

The clean-exit path parks and saves `<session>.pkv`; the fresh process calls
`SessionPool::restore_active_from_disk()` before serving. `load_s` (pinned set)
is 61 s for the warm process and is outside TTFT.

## 6. S prefill performance (N=4133, 2026-09-16)

| | legacy | coop |
|---|---:|---:|
| total | 103.67 s (40 tok/s) | 99.89 s (41 tok/s) |
| expert I/O | 30.44 s | 32.63 s |
| expert GPU | 21.76 s | 17.06 s |

Only **+3.6%**; the design target (≥5×) is not met and the coop attention
geometry is slower than legacy at the default head tiles. Correctness
(`gpu_prefill.stages`, 110 checks) passes. The 5× target needs tuning plus
GPU indexer top-k / host-readback removal / expert-I/O overlap.

## 7. R1 `--auto-tune` (2026-09-16)

2 fresh-server rounds on `chat3_turns.json` (187 decode steps/round). Round 1 did
load round 0's `heat_round_0.inc` via `DEEPMOE_HEAT_FILE` (loop verified), but the
short cold workload gave no gain: hit 0.8278 → 0.8275, tok/s 3.33 → 3.32.
A/B (`MOE_OVERLAP`/`PREFILL_HANDOFF`/`BACKFILL`) and longer warm-cache curves
remain to be measured.
