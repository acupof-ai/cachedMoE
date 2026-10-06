# cachedMoE announcement copy

## Project introduction

cachedMoE runs DeepSeek-V4.1-Flash on a single 128 GB AMD Strix Halo PC. The C++20
engine uses Vulkan and Slang to compute native FP4/FP8 weights, with an NVMe-backed
expert cache instead of fitting the entire 510 GB checkpoint in memory.

The Linux implementation includes asynchronous priority I/O, a global LRU cache,
two-drive checkpoint reads, GPU prefill, persistent sessions, and a local web UI
with separate thinking and answer panels. The exact-routing eight-turn benchmark
measured 9.20 tok/s with two drives. Experimental mask and single-path DSpark
modes are available with documented quality trade-offs.

Code, setup, and measurements: [acupof-ai/cachedMoE](https://github.com/acupof-ai/cachedMoE).
MIT licensed. Detailed reports include failed experiments and validation limits.

## Short post

Run a 510 GB DeepSeek-V4.1-Flash checkpoint on one 128 GB AMD Strix Halo PC:
Vulkan + Slang, native FP4/FP8, an asynchronous NVMe expert cache, and local chat.
Exact routing measured 9.20 tok/s on an eight-turn, two-drive run.
Code and reproducible reports: https://github.com/acupof-ai/cachedMoE

## Suggested contribution invitation

We welcome work on Vulkan kernel portability, prompt processing, cache behaviour,
and independent quality measurements. Please include hardware, driver, prompt,
cache and disk configuration when reporting speed. Experimental mask and top-K
speculation results must also include quality evidence.

## Claims to keep scoped

- The tested hardware is Strix Halo on Linux/RADV, not every AMD GPU.
- The 9.20 tok/s figure is the recorded exact-routing workload, not a guarantee.
- A short instrumented dynamic-mask/DSpark trace reached 13.40 tok/s; it is not a
  measured long-run speedup at equal answer quality.
- The full 1M context has not passed an end-to-end quality test.
- Fixed-cache results near 18 tok/s repeated tokens and are not usable-quality results.
