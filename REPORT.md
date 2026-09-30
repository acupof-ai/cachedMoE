# Tiled small-expert measurements

## Premise before changes

I dumped the RADV ISA for `prefill_gemm` stage 1, FP4, TileM 8. The weight instruction is `global_load_b128`. For each wave, `row = gid.x * 8 + wave` and `blk = lane + 32 * iteration`; the FP4 address is `W + row * 2560 + blk * 16`. Adjacent lanes therefore read 16-byte adjacent spans. One 32-lane weight instruction covers 512 consecutive bytes, or **four 128-byte cache lines** for these aligned rows. The eight rows belong to eight different waves, not to 32 lanes of one wave.

The decisive counter was **13,784,448 bytes of VRAM reads for 12.5 MB of gate/up FP4 weights** at n=8, TileM 8: **1.10 times the weight**, not the large multiplier that partly used cache lines would require. At n=16, TileM 16, it was 13,853,184 bytes (1.11 times); TileM 8 read 27,514,496 bytes (2.20 times) because it reads the weights in two token tiles. The L2 hit ratio of 36% at n=8, TileM 8 is therefore not evidence of poor weight coalescing. **The coalescing hypothesis was refuted; I made no shader change.**

The isolated n=16 gate/up took 0.136 ms at TileM 8 and 0.138 ms at TileM 16 even though the latter halved VRAM reads. GPU active cycles were 386,977 and 387,453 respectively. At n=8, TileM 8 took 0.074 ms; TileM 16 read essentially the same bytes but took 0.082 ms, with 120 versus 192 VGPRs in the ISA statistics. These numbers place the marginal limit in token arithmetic, register pressure and execution latency rather than weight-line utilization. They also explain why a blanket TileM 16 is unattractive for short jobs. Counter runs use the graphics queue, so their times should be compared within that run, not directly with the whole prefill's async-compute times. Instruction-count counters were not used to reach this conclusion.

## Change

`PrefillConfig::adaptive_moe_tile` now defaults on and can be disabled with `DEEPMOE_PF_ADAPTIVE_TILE=0` in `prefill_bench`. In `Prefill::run_moe`, each tiled expert uses the smallest compiled TileM in 4, 8, 16 that covers its row count, capped at 16. The gate/up and down dispatches use the same selected tile. The coop threshold and tiled accumulation order are unchanged. I restored the missing `bench/results/linux/prefill4k/ids4133.txt` from the tracked `tests/data/longctx/ctx4k/index.json` fixture so the specified command runs in this worktree.

One baseline 4K prefill and one adaptive 4K prefill, both with `--ops-json`, gave:

| Operation | Baseline, TileM 8 | Adaptive | Difference |
|---|---:|---:|---:|
| `gpu: moe tiled gate/up` | 902.9 ms | 843.2 ms | -59.7 ms (-6.6%) |
| `gpu: moe tiled down` | 401.0 ms | 376.6 ms | -24.4 ms (-6.1%) |
| `moe routed (gpu)` bucket | 5,092.0 ms | 4,977.3 ms | -114.7 ms (-2.3%) |

Against the prior §3 87 baseline of 882.1 / 397.5 / 5,071 ms, the adaptive run is 39 / 21 / 94 ms lower. Following the project's rule of one half, I would budget only about **42 ms** of tiled-path gain and **57 ms** of routed-bucket gain until this is seen across other prefill lengths. The 4K wall time was 46.82 versus 47.00 s; this run is dominated by expert IO, so that difference does not measure the GPU change.

After making adaptive the default, the exact required 4K command gave tiled gate/up **818 ms**, tiled down **380 ms**, routed GPU **5,003 ms**, and **`first token 77, margin 8.598`**. The adaptive `--ops-json` run also gave `first token 77, margin 8.598`. The exact output gate passed after each GPU-path edit.

## Gates

- `cmake --build build -j 8`: passed.
- `ctest --test-dir build -j 1 -LE "needs-model|needs-gpu"`: **25/25 passed**.
- `DEEPMOE_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash build/tests/deepmoe_tests "gpu_prefill."`: **7 cases, 0 failed, 0 assertion failures**. It returned 77 solely because three optional long-context cases reported skip notices.
- The exact 4K prefill command: **`first token 77, margin 8.598`**.

## Attempts that did not work

- The proposed coalescing rewrite had a false premise: the ISA already gives a wave four contiguous 128-byte weight lines, and n=8 reads only 1.10 times the weights from VRAM. No lane-layout rewrite was attempted.
- Blanket TileM 16 at n=8 was **0.082 versus 0.074 ms** for TileM 8, while reading essentially the same bytes. At n=16 it was **0.138 versus 0.136 ms** despite halving weight traffic. This is why the change selects the tile per expert.
- The first baseline full prefill stopped before GPU work because `ids4133.txt` was absent from this worktree. Regenerating it from the tracked fixture made the specified gate runnable.

The adaptive selection is worth taking for the tiled path. Its whole-prefill benefit is modest and the wall time is IO bound. Raw measurements and the stage-1 ISA dump are in `bench/results/tiled_coalesce/` and `bench/results/tiled_coalesce_baseline_gemm.txt` in this worktree.
