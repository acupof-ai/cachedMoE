# P4 / Track R1 — MoE hit-rate auto-tuning and A/B results (`docs/p4_hitrate.md`)

**STATUS: BLOCKED — NO MEASUREMENTS TAKEN (2026-09-15).**
This file is a status record written by the Track R1 subagent. Every shell command in that
session failed during sandbox setup, so no build, no `serve`/`chat` run, no bench and no git
commit was possible. Nothing below section 2 is measured; the result tables are intentionally
empty. Do not read this file as a results document — re-run Track R1 after fixing the blocker
in section 6, then replace this file with the real report.

## 1. What this document is supposed to contain

Method, the per-round/per-turn hit curve, stall (ms/token) and throughput (tok/s), the
MOE_OVERLAP / PREFILL_HANDOFF / BACKFILL A/B table, a cache-size recommendation
(measured 4200–4500 vs. the 5711-slot design value), conclusions and open items.

## 2. Method (as designed — not executed)

- One `deepmoe serve` process at a time, started only under the GPU lock
  (`powershell -NoProfile -File C:\Users\Asus\code\deepmoe\build\p4_gpu_lock.ps1 acquire r1`),
  `--cache-gb 60–70` (never 80+), released in all cases.
- Scripts: `bench/results/hitrate/long_turns.json` (8 turns) and the 3-turn `chat3` script.
- Auto-tuning loop (`--auto-tune N` to be added to `tools/hitrate_bench.py`):
  serve + chat with `DEEPMOE_ROUTE_DUMP` on → per-expert/per-layer/per-window heat →
  `tools/hitrate_sim.py` → new LRU stamps / prefetch order / backfill order → next round via
  the existing `DEEPMOE_BACKFILL` + static-heat / NVMe-prefetch path → repeat for N rounds.
- Acceptance target: decode hit ≥ 0.92, stall ≤ 100 ms/token (≥ 3 rounds or ≥ 8 turns).
- If the target is missed, localise: cache slots, union miss, backfill timing, P0/P1/P3 priority.
- A/B: `MOE_OVERLAP=0/1` with `stall_turns --repeat 2` (mandatory); `PREFILL_HANDOFF=0/1` with
  `handoff_4k` (512 optional); `BACKFILL=0/1` with `backfill_turns`. Same binary, same shader
  dir, rotated order, raw logs saved.

Historical context only (from the track notes, **not** re-measured here): an early smoke run of
`base_auto` / `base_88g` reached decode 3.40–5.72 tok/s with decode hit 0.885–0.946.

## 3. Hit curve — NOT MEASURED

| round | turns | decode hit | stall ms/token | tok/s | notes |
| ----- | ----- | ---------- | -------------- | ----- | ----- |
| —     | —     | —          | —              | —     | not run |

## 4. A/B table — NOT MEASURED

| A/B | variant | turns | decode hit | stall ms/token | tok/s | result |
| --- | ------- | ----- | ---------- | -------------- | ----- | ------ |
| MOE_OVERLAP=0/1 (`stall_turns --repeat 2`) | — | — | — | — | — | not run |
| PREFILL_HANDOFF=0/1 (`handoff_4k`, 512 optional) | — | — | — | — | — | not run |
| BACKFILL=0/1 (`backfill_turns`) | — | — | — | — | — | not run |

## 5. Cache-size recommendation — NOT MEASURED

Open: measured 4200–4500 slots vs. the 5711-slot design value — no data collected.

## 6. Blocker (evidence)

Every `pwsh` invocation in the session (including a trivial `Get-Date` and a read-only
`git status`) failed **before the command ran**:

```
Error: SetNamedSecurityInfoW failed (Win32 5): grantWrite(C:\Users\Asus\code\deepmoe\build\p4-wt\r1)
```

The command sandbox cannot set a write DACL on the session workspace root, so no process can be
started: no `cmake`/`ninja` build, no `deepmoe serve`, no `tools/hitrate_bench.py` /
`tools/hitrate_sim.py`, no tests, no GPU-lock acquisition, no `git commit`. An independent
subagent probe reproduced the identical error, and escalation to `danger-full-access` was
rejected because no approval channel is available. File tools (read/write/edit) still work, so
this record could be written; the worktree however received **no commits and no source edits**.

## 7. Remediation / next steps

1. Fix the workspace permission so the sandbox can grant write, e.g. from an elevated shell:
   `icacls C:\Users\Asus\code\deepmoe\build\p4-wt\r1 /grant "%USERNAME%":(OI)(CI)F /T`
   (or recreate the worktree from the unelevated account so that account owns it).
2. Remove the leftover probe file created while diagnosing: `Remove-Item .sandbox_probe.tmp`.
3. Re-run Track R1 from task 1: commit the uncommitted `tools/hitrate_bench.py` diff and
   `bench/results/hitrate/` results, harden the bench, add `--auto-tune N`, then run the
   locked A/Bs and rewrite this document with real numbers.

## 8. Open items

- Tasks 1–5 of `p4_tasks/r1n.txt` are untouched (no commit, no build, no runs, no tests).
- No hit-vs-round curve, no stall/tok-s numbers, no A/B verdicts, no cache-size recommendation.

## 2026-09-16 追加：多轮 auto-tune

- `tools/hitrate_bench.py --auto-tune N`：每轮起一个全新 `serve`，跑完脚本后从
  `route.bin` 生成 `heat_round_r.inc`（`layer, expert, count,`，最热在前），下一轮通过
  `DEEPMOE_HEAT_FILE` 传给引擎；结果写 `auto_tune.json`（每轮 hit / tok/s / prefill tokens）。
- `--write-heat PATH`：单轮运行后直接导出热度文件。
- 引擎侧：`store/planner.cpp` 新增 `static_heat_order(path)` 文件解析；
  `runtime/engine.cpp` 在 P3 backfill 启动时读取 `DEEPMOE_HEAT_FILE`，为空时回退内置
  `static_heat.inc`。
- `--repeat N` 仍是在同一个 serve 进程里连续跑 N 遍（expert cache 变热，用于 warm 曲线）；
  `--auto-tune` 是跨进程重新调 P3 backfill 顺序。
- **未验证**：本环境不能编译/起 serve；Python 侧已过 `py_compile`，C++ 侧需在能构建的机器上复验。
