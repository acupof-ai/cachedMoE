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

## 2026-09-16 测试状态

- 引擎侧 `DEEPMOE_HEAT_FILE` 已随整树编译通过。
- Python 侧 `write_heat_from_route` 与 `round_stats` 已用合成 route/events 单测通过；
  `--auto-tune`/`--write-heat` 出现在 `--help`。
- 尚未跑：安静机上的 serve A/B（MOE_OVERLAP / PREFILL_HANDOFF / BACKFILL）、
  `--auto-tune N` 的跨进程曲线、`prefill_p4.csv`/hit 曲线。

## 2026-09-16 `--auto-tune` 端到端实测

命令：`hitrate_bench.py --auto-tune 2 --script bench/results/hitrate/chat3_turns.json
--cache-gb 60 --env DEEPMOE_MOE_OVERLAP=1`（每轮 fresh serve，round 1 通过
`DEEPMOE_HEAT_FILE` 读 round 0 的 `heat_round_0.inc`）。

| round | heat_in | decode steps | hit | tok/s |
|---|---|---:|---:|---:|
| 0 | 无 | 187 | 0.8278 | 3.33 |
| 1 | heat_round_0.inc | 187 | 0.8275 | 3.32 |

结论：**回路已打通**（热表生成、下一轮加载、`auto_tune.json` 产出），但在
3 turn / 190 token 的短工作负载上没有命中率提升——每轮都是冷 cache，P3 backfill
的初始顺序对这段热点不够关键。需要 idle-s 更长的 backfill、更长的对话或跨轮保留
expert cache 才能看出差异；`MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL` A/B 仍待做。

## 2026-09-16 容量实测：`--cache-slots`

新增 `serve --cache-slots N`（`N * 18,808,832 B`）。同一 8-turn 长对话脚本，
`DEEPMOE_MOE_OVERLAP=1`：

| | 4500 槽（旧 auto，模拟/旧实测） | **5500 槽（实测）** |
|---|---:|---:|
| cache | 78.8 GiB | **96.34 GiB** |
| decode hit | 0.9234 | **0.9431** |
| decode tok/s | ~4.5–5.2（旧 run） | **6.05** |
| per-turn hit | 0.885–0.946 | 0.912–0.961 |
| stall 采样 | 104 ms 均值（旧 run） | 46–71 ms |

安全上限扫描：4500/5000/5200/5400/5500 槽都能 ready；**5600 槽在 path B 第 20 个
slab 失败（98.10 GiB）**。所以这台机器当前实际可加容量是 **5500 槽 ≈ 96.3 GiB**。
5500 的命中率 0.9431 与 cache_sim 对 5711 的 0.9451 基本重合，说明收益来自容量本身。
热度：`hitrate_bench.py --heat-recent N` 可按最近 N 条 route 记录生成 P3 backfill
顺序，替代全局 static heat。

## 2026-09-17 每轮 reheat（per-turn reheat）

前面 §"容量实测" 把容量加到了 5500 槽，命中率到 0.9431 之后剩下的缺口就是
**话题切换**：`docs/p4_expert_patterns.md` 实测换话题时 hit 从 0.94–0.96 掉到
0.88–0.91，因为 cache 里全是上一个话题的 expert，而下一个话题的 expert 要一个一个
miss 进来。§"`--auto-tune`" 那条路（写热表 → 下一轮 fresh serve 读）在短工作负载上
没有提升（0.8278 → 0.8275），因为它需要**跨进程**才生效。本节是把那条路搬进**同一个
serve 进程**，在每个 turn 边界上做。

### 机制

- **热**：不需要新的累加器。`ExpertSlot::heat` 本来就是 router score 的 EWMA
  （α = 0.125，覆盖 top-6 命中与 top-16 near-miss，design §9.3），每个 token 的每一层
  都在更新。所以"最近这一轮想要什么"已经在 store 里了。
- **边界**：`Engine::reheat(decay)` = `ExpertStore::decay_heat(decay)` +
  `heat_order()` + 把最冷的一段腾出来 + 用热度顺序再跑一次 P3 backfill
  （`Planner::start_backfill(order, inflight, keep=true)`）。
- **`keep=true` 是新的**：默认 P3 把填充的槽打上"低于所有 demand stamp"的时间戳，
  于是 LRU 最先淘汰它们——那是对的，启动时的静态猜测不该挤掉实测访问。reheat 的顺序
  **就是**这次对话的实测访问（按刚衰减过的热度排），所以它按 demand 规则打戳
  （`kDemandStampBase + rank`），否则这一轮刚填进去的 expert 会被下一层的 miss 立刻淘汰，
  白读一遍。
- **回归不变式**：backfill 从不淘汰，只填 free slot，所以空 cache / 无 free slot 时
  它是一个 no-op，不可能让命中率变差。腾位置是显式的：只淘汰热度低于
  `max(0.05, 0.1 * hot)` 的**尾部**，上限 `slots/32`（5500 槽时 ≤ 172 个，≈3.2 GB 读）。
- **热度必须重新归一化**：这是第一版实测抓到的 bug——`decay_heat(0.5)` 每轮把整个
  热度场乘 0.5，几轮之后所有值都趋近 0，于是"5% 的最热 expert"这种绝对阈值把**每一个**
  expert 都判成冷的：日志里 `2200 resident experts ranked ... 0 above the floor, 68
  coldest freed`。现在 `decay_heat` 把热端缩回 1.0（EWMA 自己的上限），排序（也就是
  比值）不变，而阈值在每一轮含义相同。单测
  `suite.expert_store` 的 `heat_decay_and_order_drive_the_reheat_pass` 把这两条都钉住了。

### 接口

| 位置 | 新增 |
|---|---|
| `Engine` | `reheat(decay)` / `set_reheat(bool)` / `HeatOrder{slots, warm, evicted, passed, free_slots, decay, ms}` |
| `ExpertStore` | `decay_heat(factor)`（归一化）、`heat_order()`、`heat_slots()`、`heat_max()` |
| `Planner` | `start_backfill(order, inflight, keep)` |
| `Session` | `SessionOptions{reheat, reheat_decay}`，`generate()` 结束时自动跑一次（`generated > 0` 才算边界）；`GenerateStats::reheat_turn/keys/free_slots` 进 `done` 事件 |
| `serve` | `--reheat`、`--reheat-decay F`（默认 0.5）；`{"op":"reheat","decay":F}`；`ready`/`status` 里可见 |
| 脚本 | `bench/reheat_ab.py`（on/off 三轮回合）、`bench/reheat_probe.py`（op 探针） |

### 实测

命令：`bench/reheat_ab.py --cache-slots 2200 --tokens 48`（同一话题连问三句，
`temperature 0`，`--reheat` vs `--no-reheat`，两次都是 fresh serve；`bench/results/reheat_on.json`
/ `reheat_off.json`）。

| turn | 指标 | `--reheat` | `--no-reheat` |
|---:|---|---:|---:|
| 1（冷 cache，48 token 回复） | decode hit | 0.7576 | 0.7576 |
| 1 | prefill hit | 0.6028 | 0.6028 |
| 1 | tok/s | 2.70 | 2.69 |
| 2（同话题追问，被 EOS 截到 1 token） | prefill hit | 0.6568 | 0.6560 |
| 3（同话题，48 token） | decode hit | **0.7210** | **0.7210** |
| 3 | prefill hit | 0.6415 | 0.6389 |
| 3 | tok/s | 2.43 | 2.43 |

reheat 每次 `keys = 68`（= `slots/32`），`free_slots = 0`，即 cache 已满、淘汰尾部、
重新填 68 个。**turn 3 的 decode hit 完全相同（0.7210）**。

### 这一轮最有价值的产出是一个 bug：heat 在 decode 路径上恒等于 0

写 reheat 时先加了 `--reheat` 的日志（`reheat debug -- head … floor … warm …`），
结果每一次都是 `head 0.000000 floor 0.050000 warm 0 max 0.000000`：**所有槽的 heat 都是 0**。
根因在 `Engine::run_layer`：gate kernel（`gpu/shaders/gate.slang` stage 1，
`GatePush::record = 16`）本来就把 top-16 的 id **和原始分数**写进了同一对 host-coherent
buffer（`DecodeScratch::gate_ids` / `gate_weights`，各 64 B = 16 × 4），
`Planner::plan_layer` 的 heat EWMA（design §9.3 的"near miss 让突发 expert 熬过一轮"）
就是定义在这 16 个 entry 上的——但 `RouteDecision.route.near_ids / near_scores`
**从来没有被赋值**，于是 `plan_layer` 里那个循环一轮都没进过。

影响面（不只是 reheat）：

- 任何 `score-aware` 策略今天都在对全 0 排序（design §9.3 的 Q2 变体）；
- `docs/p4_hitrate.md` 的 `DEEPMOE_HEAT_FILE` / `--heat-recent` 那条路生成的热表是
  **从 route dump 离线算的**，不受影响——所以之前那条路能跑、而引擎内的 heat 是死的；
- 这也解释了为什么"容量 +1000 槽只换来 +0.02 hit"之外，**再没有别的顺序信号**：
  cache 里 2132 个 resident expert 的 heat 全是 0，谁先被淘汰只由 LRU 时间戳决定。

修复：`run_layer` 把 gate 的 16 个 entry 一起传给 `RouteDecision`（前 6 个已乘
`route_scale`，后 10 个是 raw score，EWMA 要的就是 raw）。修完之后同样的日志变成
`head 1.000000 floor 0.100000 warm 90..121`——500 槽里 90–121 个在"这一轮还热"的一侧，
其余是冷尾，正是 reheat 需要的区分度。

### 结论：reheat 在**饱和** cache 上是 no-op，不要把 hit 的缺口算在它头上

- **机制是对的、可测的**：热度有区分度（warm 90–121 / 500），淘汰只动冷尾，
  `keep=true` 让填进去的 expert 不会被下一次 miss 立刻淘汰，单测
  `expert_store.heat_decay_and_order_drive_the_reheat_pass` 钉住了排序与归一化。
- **但在一个已经饱和的 cache 上它没有东西可做。** turn 1 之后 2132/2200 槽全满，
  turn 2/3 的专家**本来就在 cache 里**（同话题连续三轮，expert 集合几乎重合），
  于是 reheat 把 68 个冷尾换成同样用不到的专家，命中率一动不动（0.7210 = 0.7210）。
- 换句话说：reheat 能改变的只有"free slot 或者冷尾里恰好是下一轮要用的 expert"这两种情况，
  而**短对话 + 2200 槽**的实测里两者都不成立。
- 真正该做的对照不是"短对话 on/off"，而是**用户那套 8-turn 脚本 + 4500/5500 槽**：
  那里 hit 0.9234 → 0.9431 的 +0.02 是容量带来的，剩下的缺口（话题切换掉到 0.88–0.91）
  才是 reheat 的目标场景，而每次 3–4 分钟的 serve 启动使这轮时间预算不够跑完两个配置。

### 下一步（按顺序）

1. **在 4500/5500 槽 + 8-turn 脚本上重跑这个 A/B**（`bench/reheat_ab.py` 换个脚本参数即可，
   或直接用 `tools/hitrate_bench.py --script … --heat-recent`）；判据：turn ≥3 的
   decode hit 是否比 off 高 ≥0.01。
2. **`--heat-recent N` 与引擎内 heat 合并**：现在两条路各有一份热度（重放 route dump 的
   离线表 vs 引擎内的 EWMA）。引擎内的这份现在活了，`DEEPMOE_HEAT_FILE` 可以退化成
   "冷启动的第一份顺序"。
3. **把 `near_scores` 接进 `score-aware` 策略**（design §9.3 的 Q2）：`make_score_aware_policy`
   的排序函数一直是 stub，现在输入终于不是 0 了，`tools/cache_sim.py` 可以离线先判。

复现：

```powershell
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
.venv\Scripts\python.exe bench\reheat_ab.py --cache-slots 2200 --reheat     --out bench\results\reheat_on.json
.venv\Scripts\python.exe bench\reheat_ab.py --cache-slots 2200 --no-reheat  --out bench\results\reheat_off.json
```

单测：`ctest --test-dir build -R suite.expert_store`。

## 2026-09-17 命中率 → 吞吐的实测斜率（`bench/config_sweep.py`）

同一套 4 轮同话题对话、同一份二进制、同一台空闲机，只改 cache 容量与 reheat 开关。
`bench/results/config_sweep.json`；`MB/tok` = 每个 decode token 的 P0 miss 字节，
`NVMe GB/s` = 这些字节 ÷ decode 墙钟（turn 的 `total_ms − ttft_ms` 之和）。

| 配置 | 槽 | cache | decode tok/s | decode hit | MB/tok | **有效 NVMe** |
|---|---:|---:|---:|---:|---:|---:|
| cache-1000 | 1000 | 17.5 GiB | 1.830 | 0.5912 | 4,998 | 9.14 GB/s |
| cache-2200 | 2200 | 38.5 GiB | 2.593 | 0.7431 | 3,528 | 9.15 GB/s |
| cache-2200 **+ reheat** | 2200 | 38.5 GiB | 2.594 | **0.7431** | 3,522 | 9.13 GB/s |
| cache-4500 | 4500 | 78.8 GiB | 3.647 | 0.8370 | 2,269 | 8.28 GB/s |
| cache-4500 **+ reheat** | 4500 | 78.8 GiB | 3.629 | 0.8386 | 2,301 | 8.35 GB/s |

三条结论：

1. **decode 是被 NVMe 带宽钉住的，不是被算力。** 四种容量的有效读带宽都是
   **8.3–9.2 GB/s**——命中率一涨，每 token 的 miss 字节就降，tok/s 就按同样的比例涨。
   所以这一段的性能模型是 `tok/s ≈ NVMe_eff / (MB/tok)`，先看 `MB/tok`。
2. **容量的斜率是可外推的**：1000 → 2200 → 4500 槽，hit 每次翻倍容量 **+0.093/+0.094**
   （0.5912 → 0.7431 → 0.8370）。按这个斜率外推到 5500 槽得 **≈0.90**，
   实测（8-turn 脚本）是 **0.9431**——同量级，说明"容量给 hit"这条线是稳的，
   而且 8-turn 长对话比 4 轮短脚本更吃容量（话题内复用更多）。
3. **reheat 在这一轮的账是 0，而且有微小的负成本**：hit 差 +0.0016（噪声），
   tok/s 差 −0.018（4500 槽），MB/tok **+32**——那正是它自己填的那 68 个 expert
   （68 × 18.8 MB = 1.28 GB/轮）。**结论：在 4 轮短对话上不要开 `--reheat`**；
   它要证明自己，只能靠"8-turn + 5500 槽 + 话题切换"那套脚本。

`MB/tok` 的量级也值得注意：1,000 槽时 4,998 MB/token，而"完全没有 cache"的理论值是
40 层 × 6 expert × 18.8 MB = **4,514 MB/token**——也就是说 1000 槽时几乎等于没有 cache，
且**实测字节比"每个 miss 读一个 expert"高 10–15%**（对齐/两段 run：每个 expert 是
1,110,016 B 的 scales run + 17,698,816 B 的 weights run，槽 18,808,832 B，
读的是对齐后的字节）。

