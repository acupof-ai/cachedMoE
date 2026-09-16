# P4 — prefill speed (Track S)

> **STATUS: BLOCKED — nothing was built, committed or measured.**
> This file is a **handoff stub**, not the measured document task 4 of
> `build/p4_tasks/sn.txt` asks for. Every number in it is *quoted from a named
> source* and is explicitly **not re-measured**. §4 is the ready-to-paste run
> list that produces the real document.

---

## 0. Why this file has no new numbers

The session could not spawn a single process. Every exec-capable tool
(`pwsh`, `grep`, `glob`, and a subagent used as a probe) aborts **before the
process starts**:

```
Error: SetNamedSecurityInfoW failed (Win32 5): grantWrite(C:\Users\Asus\code\deepmoe\build\p4-wt\s)
```

Win32 5 is `ERROR_ACCESS_DENIED`: the `workspace-write` sandbox cannot apply its
ACL grant to the worktree, so the command is refused at the harness layer. It is
deterministic — 7 attempts, including one from a delegated subagent, all
identical. Escalation was refused (`sandbox escalation to "danger-full-access"
requires approval, but no approval channel is available`), so there is no way to
run anything.

Consequences, mapped to `sn.txt`:

| task | what it needs | status |
|---|---|---|
| 1 build + re-run `test_gpu_prefill` / stage checks, commit in small steps | `cmake`, `ctest`, `git` | **not started** — no commit exists |
| 2 quiet N=4133 bench, legacy vs coop, per-stage table | GPU lock + bench binary | **not started** |
| 3 indexer top-k on GPU / drop `engram.h`+`rope.h` include dep | edit + build + test | **not started** |
| 4 `bench/results/prefill_p4.csv` + this doc | the run in task 2 | csv absent, doc = this stub |
| 5 done marker | file write | written, `status = "blocked"` |

The fallback marker `<worktree>\build\p4_done.json` was used; the primary
`C:\Users\Asus\code\deepmoe\build\p4_done\s.json` is outside this session's
writable workspace and its write was denied.

Minimal reproduction of the blocker (any one of these is enough):

```powershell
pwsh -Command "echo hi"                                   # -> SetNamedSecurityInfoW failed (Win32 5)
cmake --build build                                       # never reached
git -C C:\Users\Asus\code\deepmoe\build\p4-wt\s status    # never reached
```

What still works: read/write of files **inside** the worktree. Everything below
marked *verified* was checked that way.

---

## 1. Baseline — quoted, NOT re-measured

From the Track S handoff in `build/p4_tasks/sn.txt` (N=4133, replay 128, before
the cooperative-matrix work):

| bucket | s | share |
|---|---:|---:|
| **total** | **105.30 s = 39 tok/s** | 100% |
| expert I/O wait | 43.10 | 41% |
| attention | 27.19 | 26% |
| expert GPU | 16.71 | 16% |
| mHC | 4.91 | 5% |
| host readbacks | ~3.4 | 3% |
| (remainder: embed / engram / gate / head / host-other) | ~9.99 | 9% |

Hard floor quoted with it: **expert I/O 43.1 s / 198 GB at 4.30 GB/s**.

For reference, `docs/p3_prefill.md` §10 (pre-Track-S, contended) had N=4133 at
199.6 s ≈ 150 s quiet, so Track S is the change that got this to 105.30 s.

---

## 2. The uncommitted Track S work — verified present in the worktree

Per `sn.txt`: HEAD has **no own commits yet**, 6 dirty files, +500/−23:
`gpu/shaders/prefill_coopmat.slang`, `gpu/shaders/prefill_attn.slang`,
`gpu/vulkan/prefill_kernels.{h,cpp}`, `bench/prefill_bench.cpp`,
`tests/test_gpu_prefill.cpp`. (The dirty state itself is **not verifiable** in
this session — `git` cannot run — it is taken from the task file.)

Verified by reading the files:

- `gpu/shaders/prefill_coopmat.slang` exists (227 lines). Its header documents
  the four stages: 0 the GEMM, 1 the quantised-activation → fp16 plane, 2 the
  fp32-activation → fp16 plane, 3 band-attention scores and 4 P·V, one
  `CoopMat<half, Subgroup, 16, 16, use>` 16×16 tile per wave, 32 tokens a
  workgroup.
- `gpu/vulkan/prefill_kernels.h` carries the switches:
  `PrefillConfig::attn_coop = true` (band attention on cooperative-matrix tiles,
  `false` = the per-(head, query) kernel of `docs/p3_prefill.md` §6),
  `coopmat_min_rows = 16` (routed experts ≥ 16 rows on coopmat),
  `coopmat_dense_min_rows = 64` (dense linears ≥ 64 rows on coopmat), and
  `attn_head_tiles` (16-head tiles per workgroup, 1/2/4) — i.e. the three
  specialisation knobs the bench is meant to sweep.
- The header comment already points at `docs/p4_prefill_speed.md` §3 for the
  band-attention design, which is why this stub keeps that section number.
- `bench/results/` exists; `bench/results/prefill_p4.csv` does **not**.
- `build/CMakeCache.txt` does **not** exist — the worktree build directory is
  not configured, so task 1 starts with a full `cmake -S . -B build`.
- Validation quoted from `sn.txt` (not reproducible here): coop vs legacy stage
  cos ≈ 0.99999, gate top-6 64/64 identical (`stages_coop.log` /
  `stages_legacy.log`). Also quoted: the final integrated 4K run died at layer 9
  with `vkQueueSubmit2 failed (-4)` while another track's job ran concurrently
  and free RAM was 57.5/66.7 GB — an invalid measurement, not a code result,
  which is exactly why task 2 demands the GPU lock and a quiet machine.

---

## 3. Band attention on cooperative-matrix tiles

*(design section referenced by `prefill_kernels.h`; described here only from the
shader's own header comment — no measurement was made)*

`gpu/shaders/prefill_coopmat.slang` stages 3 and 4 re-express
`prefill_attn.slang`'s band attention as two cooperative-matrix products per
query block:

- stage 3 — scores `S[b][h][t] = Σ_d Q[b][h][d] · G[b][t][d]`, with `Q` the fp16
  queries `[B][H][D]` and `G` the fp16 KV rows each query's index list gathers
  `[B][E][D]`; A = heads × dims tile, B = dims × entries, C = heads × entries,
  `CmRows = E` (multiple of 16), `CmCols = D`;
- stage 4 — `O[b][h][d] = Σ_t P[b][h][t] · G[b][t][d]` with the bf16
  probabilities staged as fp16 `[B][H][E]`; A = heads × entries, B = entries ×
  dims, C = heads × dims;
- plus gather, tile softmax and a finish pass, all in **one submit**.

`attn_coop = false` selects the pre-existing per-(head, query) kernel, which is
the correctness fallback the stage comparison in task 1 checks against.

---

## 4. The runs that still have to happen (ready to paste)

Build (identical to `build/p4_tasks/r2.txt`, which `sn.txt` names as the build
recipe; run from `C:\Users\Asus\code\deepmoe\build\p4-wt\s`):

```powershell
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build          # retry once on "sub-compilation of compiler_rt failed"
```

Correctness first (task 1), then commit in small steps — shaders, kernels,
bench/test — e.g. `git add gpu/shaders/prefill_coopmat.slang
gpu/shaders/prefill_attn.slang` → commit, then `gpu/vulkan/prefill_kernels.*`,
then `bench/prefill_bench.cpp tests/test_gpu_prefill.cpp`. Compare the coop and
legacy stage logs (`stages_coop.log` vs `stages_legacy.log`): the acceptance
bar already used by this track is stage cos ≈ 0.99999 and gate top-6 64/64
identical.

Quiet N=4133 bench (task 2) — heavy runs only after the lock says ACQUIRED, and
release in all cases:

```powershell
powershell -NoProfile -File C:\Users\Asus\code\deepmoe\build\p4_gpu_lock.ps1 acquire s
# coop (default)
build\bench\prefill_bench.exe --section prefill --n 4133 --ids ids.txt --replay 128 `
    --csv bench/results/prefill_p4.csv
# legacy attention, same binary
$env:DEEPMOE_PF_ATTN="legacy"
build\bench\prefill_bench.exe --section prefill --n 4133 --ids ids.txt --replay 128 `
    --csv bench/results/prefill_p4.csv
Remove-Item Env:\DEEPMOE_PF_ATTN
powershell -NoProfile -File C:\Users\Asus\code\deepmoe\build\p4_gpu_lock.ps1 release s
```

The bench already prints and CSV-rows exactly the buckets task 2 asks for
(`bench/prefill_bench.cpp`, `run_prefill`): `total`, `embed`, `engram_io`,
`engram_gpu`, `mhc`, `attention`, `gate_route`, `shared_expert`, `expert_io`,
`expert_gpu`, `head`, `host_other`, plus experts streamed, GB read, GB/s against
the wait, engram reads, dispatches/submits, first token and margin. Knobs worth
sweeping in the same session: `--coop-min {-1,0,16}`, `--coop-dense {-1,64}`,
`--transit`, `--replay {128,0}`, and `DEEPMOE_PF_ATTN_HT` for
`attn_head_tiles`.

---

## 5. Provisional reading of the ≤ 20 s target (arithmetic only)

Clearly **not** a measurement, and to be replaced by §4's table:

- The quoted expert-I/O floor is 198 GB; at the observed ~4.3–4.6 GB/s that is
  ~43–46 s of NVMe traffic per 4K prompt. So ≤ 20 s total is unreachable on
  bandwidth alone: it would need **≥ ~10 GB/s** effective expert bandwidth, or
  substantially fewer bytes read (better replay/coopmat reuse, cache hits), or
  a much smaller resident working set.
- Conversely, 43.10 s of 105.30 s is 41% of the baseline total, so even
  perfectly overlapping *all* expert I/O leaves ~62 s of non-I/O time at the
  old compute speed. Attention 27.19 s + expert GPU 16.71 s + mHC 4.91 s is
  where the coopmat work is supposed to pay.
- The honest first milestone is therefore **~55–65 s** (overlap expert I/O with
  the coopmat compute), with ≤ 20 s only after the expert-read path itself is
  ~2.5–3× cheaper. Task 2's revised-target paragraph is the place to confirm or
  refute this with the real per-stage table.

---

## 6. What remains, in `sn.txt` order

1. configure + build, run `test_gpu_prefill` and the coop-vs-legacy stage checks;
2. commit the validated coop code in small steps (shaders → kernels →
   bench/test), **no attribution lines, no push**;
3. quiet N=4133 legacy-vs-coop bench under the GPU lock; fill in §1/§5 here with
   the measured table, plus `bench/results/prefill_p4.csv`;
4. extras if time permits, in this order: indexer top-k fully on GPU (removes
   the host score readback + host top-k, ~2.2 s baseline), then design §15 open
   issue 28 (drop `prefill_kernels.cpp`'s include dependency on
   `runtime/engram.h` + `runtime/rope.h` by sinking what is needed);
5. rewrite `build\p4_done.json` with `status:"done"`, the commit hashes, the
   test command, and the numbers.
