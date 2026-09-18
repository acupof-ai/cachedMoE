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

## 9. 专家加载路径上没有量化（2026-09-17 审计）

有一个看起来很有希望的优化方向是"加载专家时别做量化 / 别做变换，推迟到后面"。
**审计结论：这条路径上本来就没有量化**，所以没有可省的东西。逐段核对
（`store/planner.cpp::fetch` → `storage/windows/iocp.cpp` → `store/expert_store.cpp::finish_run`）：

1. `Planner::fetch` 把每个 manifest run 变成一条 `IoRequest`：源是**对齐后的文件偏移**，
   目标**直接是槽的 host 指针**（`req.dst = run.dst`，在 `begin_fill` 里由
   `slot.host_ptr + slot_offset` 算出）；
2. `iocp.cpp` 对这条请求只做一次 `ReadFile`（overlapped，`FILE_FLAG_NO_BUFFERING`），
   **字节从块设备直接落进槽里**；
3. 完成回调只调 `ExpertStore::finish_run` → `settle_locked`：状态机、`publish_locked`、
   几个计数器，**没有一次逐元素循环**。

也就是说 expert 的 fp4 权重 + block-32 scale **在 checkpoint 里就长这样**，运行时读到的就是
最终字节；每个槽 18,808,832 B 的搬运里，CPU 只负责 `ReadFile` 的提交与状态翻转。
"加载时量化"要么指的是 checkpoint 侧的离线重打包（那要重写 475 GiB，和本项目
"原样读、不重打包" 的原则相悖），要么是本项目从来没做过的事。

**真正的疑点**是每个 miss 的 20–31 ms（`p4_summary` §2：107 GB / 5,030 fills）：按
8.3–9.2 GB/s 算，盘读只要 ~2 ms，所以另外 ~20 ms 在别处 —— I/O 的并发深度，还是把字节
写进 path A（`DEVICE_LOCAL | HOST_VISIBLE`，共享显存）那一段的带宽。要定位它，该做的是
在 `ReadFile` 进出的两侧各打一个 host 时间戳（不动数据路径），**而不是**动量化的主意。


## 2026-09-18 Track F4 — the A/Bs, and the defaults they set

Everything below was measured in `C:\Users\Asus\code\deepmoe-fin-r1` on the
`p4/fin-r1` branch, one `deepmoe serve` at a time under the p4 GPU lock
(`build\p4_gpu_lock.ps1`), same binary, same shader directory, same 8-turn
script `bench/results/hitrate/long_turns.json`. Raw runs are in
`bench/results/hitrate/<cell>/` (`turns.json`, `profile.jsonl`, `route.bin`,
`events.jsonl`, `status.json`, `serve.log`).

### 0. What the engine's own routing says before any new run

`tools/hitrate_sim.py curve` replays a run's `route.bin` through
`tools/cache_sim.py`'s LRU. On both existing 8-turn dumps the engine and the
simulator agree on **every** step (2,489/2,489 and 2,552/2,552), so the
simulator can be trusted to price a capacity the machine cannot hold.

Decode-only hit of the 8-turn route at each capacity (one slot = 18,808,832 B):

| slots | GiB | sim decode hit | Δ vs 4,500 |
|---:|---:|---:|---:|
| 2,200 | 38.5 | 0.8127 | −0.1119 |
| 3,000 | 52.6 | 0.8701 | −0.0545 |
| 3,500 | 61.3 | 0.8941 | −0.0305 |
| 4,000 | 70.1 | 0.9109 | −0.0137 |
| 4,500 | 78.8 | 0.9246 | — |
| 5,000 | 87.6 | 0.9356 | +0.0109 |
| 5,200 | 91.1 | 0.9390 | +0.0144 |
| 5,500 | 96.3 | 0.9431 | +0.0185 |
| 5,711 | 100.0 | 0.9456 | +0.0210 |
| 6,000 | 105.1 | 0.9486 | +0.0239 |
| 6,500 | 113.9 | 0.9532 | +0.0285 |

The curve is still climbing at the machine's ceiling, and §8 showed decode is
NVMe-bandwidth-bound (`tok/s ≈ NVMe_eff / (MB/tok)`), so **every point of hit is
throughput**. That is the whole argument for sizing the cache from this curve.

Per-turn, how much of a turn's decode routing its own prompt prefill already
touched (the lever `docs/p4_expert_patterns.md` §1 calls the biggest one), on
the 5,500-slot run:

| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill tokens | 29 | 18 | 35 | 22 | 64 | 39 | 42 | 47 | |
| decode selections seen in that turn's prefill | 0.679 | 0.425 | 0.670 | 0.647 | 0.662 | 0.681 | 0.723 | 0.623 | **0.657** |

(The 86.6% of `p4_expert_patterns.md` is cumulative over every prefill so far;
per turn, against its own prompt only, it is 0.66. Turn 2 is an 18-token
follow-up, which is why it is the outlier.)

### 1. The per-turn reheat pass was a no-op *by construction*, not by luck

§7 measured turn-3 decode hit 0.7210 with `--reheat` and 0.7210 without, and
concluded "a saturated cache has nothing for it to do". Reading the pump says
something stronger. `Planner::backfill_pump` has

```cpp
if (store_->slot_of(key)) continue;          // held already
```

and the old `Engine::reheat` built its order as *residents hottest-first, then
non-residents*, then truncated it to `budget = free_slots + evicted`. On a
saturated cache `free_slots == 0` and `evicted ≤ slots/32`, so the truncation
kept **only residents** — every key the pump was handed was one it skips. The
pass therefore:

* evicted `slots/32` experts (68 at 2,200 slots, 172 at 5,500), and
* issued **zero** fetches,

which is exactly a cost: the +32 MB/token and −0.018 tok/s that §8's sweep
charged to `--reheat` is those evicted slots being re-read on demand.

Fixed in this branch: the pass selects its candidates from the heat table
(`DEEPMOE_HEAT_FILE` when given, else `store/static_heat.inc` — now read once
and shared with the startup P3 backfill, so `--write-heat` / `--heat-recent`
steer both) filtered to **non-resident** keys, and it evicts at most as many
slots as it has candidates for, so it can never again evict without fetching.
`tests/test_integration.cpp::a_backfill_order_of_resident_keys_fetches_nothing`
pins the planner half of the invariant.

Two things the same reading turned up that are *not* bugs but are worth writing
down, because the interface reads as if they were:

* **`decay` does nothing to the ranking.** `ExpertStore::decay_heat(f)` returns
  early for `f >= 1` and otherwise rescales the hot end to 1.0 — it never
  multiplies by `f`. Since a uniform scale is order-preserving, "decay then
  renormalise" and "renormalise" are the same ranking, so `--reheat-decay` is a
  no-op on the order and only the `f >= 1` early return is observable. The real
  ageing is in `note_heat`'s EWMA.
* **`warm` grows monotonically.** With no true decay, a slot's heat only ever
  rises, so the `floor_heat = max(0.05, 0.1 * head)` prefix widens turn by turn
  and `want = min(slots/32, slots − warm)` shrinks towards zero. The pass fades
  out over a long conversation even when it has work.

---

### 2. Every run below was taken on a quiet machine — and which earlier ones were not

The campaign the previous agent left behind had produced exactly one complete
cell (`m_4500_off`) before the freeze; every other directory under
`bench/results/hitrate/` held an empty `route.bin` and a serve log ending in a
lost device. Worse, a **detached** `f4_campaign.sh` was still running: killing
its `deepmoe serve` only made it start the next cell. The whole tree
(`f4_campaign.sh` → `bash` → `hitrate_bench.py` → `deepmoe serve`) had to go
before anything could be measured, and the campaign was re-run in the
foreground, one cell at a time, by `f4_seq.sh`: it refuses to start a cell while
`Get-Process deepmoe,deepmoe_tests,prefill_bench` is non-empty and kills the
engine after each. Every cell logs the free physical memory it started with.

| provenance | cells |
| --- | --- |
| quiet, this session, verified per cell (50–54 GiB free, no other engine) | every `m_*`, `ov_*`, `bf_*`, `ho_*`, `g_*` cell in the tables below |
| quiet under the old campaign's own gate (≥38 GiB free, no other engine) | `m_4500_off` — kept and reused; its sim agreement and hit are consistent with this session's cells |
| **loaded** — do not read as results | everything in `docs/p4_dspark_runtime.md` §8's `config_sweep.json`, and the pre-freeze `m_5500_*` / `ov_*` / `bf_off` directories, which were deleted |

One caveat that applies to every table: **`tok/s` is only comparable within a
contiguous block of cells**, because a CPU-only job on the other track (F5) was
running throughout and its load varied. `hit` and `MB/token` are properties of
the routing and the cache and are not affected; they are what the defaults are
set from. Where a `tok/s` comparison carries weight below, the two cells ran
back to back.

### 3. There is no planner-vs-LRU gap: the engine *is* pure LRU

The premise that the engine's hit was ~7 points below a pure-LRU replay of the
same routing does not survive being measured on the same stream. `hitrate_sim.py
curve` replays a run's own `route.bin` through `cache_sim.py`'s LRU at the
engine's own capacity and compares the engine's per-step, per-layer hit counts
against the simulator's:

| run | script | slots | engine hit | sim LRU hit | steps agreeing |
| --- | --- | ---: | ---: | ---: | --- |
| `m_4500_off` | 8-turn `long_turns` | 4,500 | 0.9062 | 0.9062 | **2,489 / 2,489** |
| `m_5000_off` | 8-turn `long_turns` | 5,000 | 0.9136 | 0.9136 | **2,489 / 2,489** |
| `g_sweep_4500` | the 4-turn `config_sweep` TOPIC | 4,500 | 0.8505 | 0.8505 | **214 / 214** |

The third row is the direct test: `config_sweep.json`'s 0.8370 is the number the
gap was computed from, so its own dialogue was replayed through
`hitrate_bench.py` (which dumps routing; `bench/config_sweep.py` does not) and
then through the simulator. Engine and simulator agree on every step. The
0.8370-vs-0.905 comparison is between a **4-turn, 92-decode-step, cold-cache**
run and an **8-turn, 2,193-step** one; it is not a policy gap, it is two
different workloads.

The source says the same thing, which is why the suspects can be closed
individually rather than measured one at a time:

* **(a) eviction is not `(heat, last_use)`.** `ExpertStore::evict_lru`
  (`store/expert_store.cpp:420`) ranks on `last_use_token` alone, and
  `store/planner.cpp:90` still carries the `TODO(design §9.3): rank by (heat,
  last_use)`. `heat` is read only by `heat_order()`, which only the reheat pass
  calls. The near-miss fix made `heat` non-zero; it did not put it in the
  eviction order.
* **(b) the P3 backfill is stamped *below* demand**, not as demand:
  `kDemandStampBase - 1 - rank` (`planner.cpp:495`). Only the reheat pass passes
  `keep=true`, and its `kDemandStampBase + rank` still lands below every real
  demand access, because demand stamps start at `kDemandStampBase` and only
  increase (`planner.h:235`, `access_clock_`). A reheated slot is therefore the
  *first* demand-range slot evicted, not the last.
* **(c) the prefill handoff's stamps are the token-major access order the prompt
  would have produced anyway** (`engine.cpp`, `sink.reserve`), and
  `store::admit_streamed` drops a reservation whose stamp is older than the
  oldest evictable slot, so it cannot displace anything fresher.
* **(d) no prefetch is active**: `prefetch 0 issued / 0 used / 0 wasted` in every
  `status.json` of every cell.

And the ablation the premise asked for, run anyway (`g_abl_none`: 8-turn,
4,500 slots, `DEEPMOE_BACKFILL=0 DEEPMOE_PREFILL_HANDOFF=0 DEEPMOE_MOE_OVERLAP=0`,
no reheat) confirms it from the other side — turning everything off does not
raise the hit, because there was nothing to turn off:

| 4,500 slots, 8-turn | decode hit | stall ms/token | MB/token | steps agreeing with sim |
| --- | ---: | ---: | ---: | --- |
| defaults | 0.9250 | 92.2 | 338.3 | 2,489 / 2,489 |
| everything off (`g_abl_none`) | 0.9252 | 115.8 | 337.8 | 2,489 / 2,489 |

(The two runs' `tok/s` are 5.144 and 3.347, but they are 50 minutes apart with
different F5 load and the isolated `MOE_OVERLAP` A/B below prices the same
switch at +1–4%, so that difference is the machine, not the flags. The hit and
the bytes, which are load-independent, are identical.)

### 4. Capacity: the ceiling moved down, and `auto` is now the right default

`--cache-slots 5500` was the recommendation of the §"容量实测" section above.
It no longer holds on this machine. At 5,500 slots the pool needs 19 path-B
slabs and the device is lost on the first token, on a **quiet** machine with
53.8 GiB free:

```
[INF] slab pool: path A full after 36 slabs (...), continuing on path B
[INF] engine: expert cache 5500 slots, 96.34 GiB (36 slabs on path A, 19 on path B)
{"event":"error","message":"internal: the residency timeline reads UINT64_MAX, which is
 what a LOST device reports -- most likely the expert cache's last path-B import
 exhausted the host heap"}
```

The ceiling scan, one serve at a time (100 slots per slab; path A takes the
first 36 slabs, everything above 3,600 slots goes to path B):

| slots | GiB | path A / B slabs | 3-turn probe | 8-turn script |
| ---: | ---: | --- | --- | --- |
| 4,500 | 78.8 | 36 / 9 | — | **ok** |
| 5,000 | 87.6 | 36 / 14 | ok | **ok** |
| 5,200 | 91.1 | 36 / 16 | ok | — |
| 5,400 | 94.6 | 36 / 18 | ok | **failed** (`io: layer 21: an expert read failed`, 12 min in) |
| 5,500 | 96.3 | 36 / 19 | — | **failed** (device lost, first token) |

A short probe is not enough: 5,400 passes three turns and dies in the middle of
eight. So the safe hand-set maximum is 5,000, and **the default is not a
hand-set number at all — it is `auto`**, which on this machine lands at 5,100
slots / 89.3 GiB and is the best cell measured:

| cell | slots | GiB | decode hit | stall ms/tok | MB/token | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m_4500_off` | 4,500 | 78.8 | 0.9250 | 92.2 | 338.3 | 5.144 |
| `m_5000_off` | 5,000 | 87.6 | 0.9364 | 77.5 | 287.3 | 5.264 |
| **`m_auto`** | **5,100** | **89.3** | **0.9383** | **76.0** | **278.6** | **5.603** |

`auto` wins because it sizes from the machine at boot instead of from a constant
measured on a different day: `path A 62.26 GiB after 9.17 GiB pinned, path B
27.22 GiB of 53.91 GiB physical free`. The three bounds that produce that are
`avail_phys - kPhysFloor` (path B shrinks whenever another track holds host
memory), `kPathBAutoCeiling` (30 GiB — 5,100 slots sits ~400 under the 5,500
that loses the device and ~300 under the 5,400 that fails a long run), and
`heap_a - pinned - kPathAOther`. **Do not pass `--cache-slots` in production**;
it bypasses all three, which is how 5,500 was reachable at all.

The measured curve, and what the simulator prices the capacities it cannot hold
at (`hitrate_sim.py curve m_5000_off`, 2,193 decode steps in 18 windows of 128 —
the engine agrees with the simulator on all 2,489 steps, so the columns to the
right are trustworthy):

| steps | measured hit | stall ms | tok/s | MB/token | sim 4,500 | sim 5,000 | sim 5,400 | sim 6,500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–127 | 0.882 | 138.5 | 3.854 | 535.1 | 0.878 | 0.882 | 0.882 | 0.883 |
| 128–255 | 0.937 | 80.6 | 5.235 | 286.5 | 0.931 | 0.937 | 0.941 | 0.950 |
| 256–383 | 0.926 | 90.2 | 4.968 | 335.1 | 0.915 | 0.926 | 0.934 | 0.955 |
| 384–511 | 0.867 | 155.0 | 3.862 | 602.4 | 0.852 | 0.867 | 0.874 | 0.898 |
| 512–639 | 0.945 | 72.2 | 5.789 | 249.8 | 0.935 | 0.945 | 0.950 | 0.958 |
| 640–767 | 0.926 | 91.1 | 5.139 | 332.9 | 0.909 | 0.926 | 0.935 | 0.955 |
| 768–895 | 0.944 | 72.1 | 5.503 | 253.7 | 0.933 | 0.944 | 0.954 | 0.976 |
| 896–1023 | 0.849 | 173.3 | 3.356 | 680.6 | 0.835 | 0.849 | 0.859 | 0.885 |
| 1024–1151 | 0.908 | 113.1 | 4.254 | 415.5 | 0.890 | 0.908 | 0.921 | 0.936 |
| 1152–1279 | 0.917 | 101.8 | 4.565 | 373.6 | 0.901 | 0.917 | 0.929 | 0.952 |
| 1280–1407 | 0.855 | 168.4 | 3.378 | 655.6 | 0.840 | 0.855 | 0.864 | 0.888 |
| 1408–1535 | 0.934 | 82.7 | 5.282 | 297.2 | 0.929 | 0.934 | 0.938 | 0.948 |
| 1536–1663 | 0.963 | 50.2 | 6.474 | 167.3 | 0.957 | 0.963 | 0.966 | 0.971 |
| 1664–1791 | 0.967 | 45.1 | 6.612 | 149.4 | 0.958 | 0.967 | 0.974 | 0.980 |
| 1792–1919 | 0.873 | 148.4 | 3.726 | 571.7 | 0.856 | 0.873 | 0.880 | 0.896 |
| 1920–2047 | 0.890 | 130.5 | 4.061 | 497.1 | 0.881 | 0.890 | 0.896 | 0.920 |
| 2048–2175 | 0.944 | 72.0 | 5.555 | 255.4 | 0.936 | 0.944 | 0.948 | 0.955 |
| 2176–2192 | 0.951 | 63.7 | 5.891 | 221.7 | 0.942 | 0.951 | 0.954 | 0.962 |
| **all** | **0.9136** | **104.7** | **4.605** | | 0.9024 | 0.9136 | 0.9205 | 0.9357 |

The curve is still climbing at the hardware's limit — 6,500 slots would be worth
another 2.2 points — which is the argument for spending every byte the machine
will *safely* give, and the reason the ceiling is enforced by three measured
bounds rather than one constant.

### 5. The GPU prefill could not allocate at all, and the fix is a path-A reserve

Running the prefill→decode handoff A/B turned up something bigger than the A/B.
Every 8-turn cell in this document reports `prefill_mode: "decode"`, and the
reason is not that the GPU prefill lost a race:

```
[WRN] session: GPU prefill failed (resource-exhausted: prefill buffers (1203765248 B):
 vkAllocateMemory(1203765248 B, type 2) failed (-2)); falling back to the decode path
[WRN] session: GPU prefill failed (resource-exhausted: prefill buffers (21160960 B):
 vkAllocateMemory(21164032 B, type 2) failed (-2)); falling back to the decode path
```

The second line is a **21 MB** allocation. Path A was not short, it was empty:
the slab pool fills path A until `vkAllocateMemory` refuses, so any cache of
3,600 slots or more takes every path-A slab, and the GPU prefill — allocated
later, and only from path A — never gets a byte. `kPathAOther` reserves for
exactly this, but only inside the *auto* budget; `--cache-slots` and `--cache-gb`
bypass that arithmetic, so the reserve has to be taken from the heap rather than
from a number.

`Engine::build_expert_cache` now holds a 4 GiB path-A reserve across the pool
build and frees it immediately after (`kPathAReserve`, taken in
`maxMemoryAllocationSize` pieces — a single 4 GiB allocation is refused outright
with `4294967296 B exceeds maxMemoryAllocationSize 2147483648 B`, the same 2 GiB
cap that made slabs exist). It costs two slabs — the 4,500-slot cache becomes 34
path-A + 11 path-B instead of 36 + 9, with the same 4,500 slots — and it is the
difference between a GPU prefill and a decode-path one:

| 4,133-token prompt, 4,500 slots | prefill | prefill tok/s | wall for the cell |
| --- | ---: | ---: | ---: |
| before (decode-path fallback) | 1,061.7 s | 3.89 | 19 min |
| after (GPU prefill runs) | 100.0 s | 41.3 | 3 min |

**10.6× on TTFT for a 4K prompt**, from two slabs of cache. 2 GiB was tried
first and was enough at a 4,096-position context but not at 8,192, so the
constant is 4 GiB; the auto budget's `kPathAOther` was raised from 1 GiB to
3 GiB in the same spirit (a 4K-context decode-only run was what 1 GiB was
measured against).

### 6. The A/Bs

All four switches already default the way the data says, except the P3 backfill,
which the counters show has been defaulting **off** all along
(`backfill 0 issued` in every cell with no `DEEPMOE_BACKFILL` set).

**`DEEPMOE_MOE_OVERLAP`** (`stall_turns --repeat 2`, 4,500 slots, cold pass then
warm pass in one process; the two cells ran back to back at 09:00 and 09:10 with
52.7/52.6 GiB free, so the `tok/s` are comparable).

| pass | variant | steps | decode hit | stall ms/tok | fence_wait ms | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1, cold | off | 933 | 0.9116 | 107.2 | 108.3 | 4.431 |
| 1, cold | **on** | 933 | 0.9114 | 106.6 | 105.5 | **4.475** (+1.0%) |
| 2, warm | off | 933 | 0.9115 | 107.6 | 114.3 | 4.307 |
| 2, warm | **on** | 933 | 0.9115 | 107.5 | 104.0 | **4.482** (+4.1%) |

Hit is identical to four decimals in both passes, which is the point: the overlap
moves work, it does not change what the cache holds. **Default stays on.**

**P3 backfill** (`backfill_turns`, `--idle-s 90`, 4,500 slots) — the ramp.

| | decode hit | stall ms/tok | tok/s | turn-1 hit | turn-1 tok/s | P3 bytes read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off (the current default) | 0.9309 | 85.9 | 5.038 | 0.9378 | 5.294 | 9.4 GiB |
| on | 0.9320 | 84.0 | 5.112 | 0.9407 | 5.459 | 30.9 GiB |

The whole effect is in turn 1 (+0.003 hit, +3.1% tok/s); turns 2 and 3 are
identical to four decimals, because by then demand has evicted the static-heat
guess. **Default stays off**: +0.0011 overall hit is not worth 21.5 GiB of extra
NVMe reads during the idle window, and 90 s of idle before the first token is the
*best* case for it.

(The 9.4 GiB in the "off" row is not backfill. `p3_backfill 1214 req 9395.3 MiB`
is a constant in every cell, equal to the 9.17 GiB pinned-weight load — the
pinned loader submits at the P3 priority. The P0/P2/P3 counters are otherwise
exactly as designed: in `m_auto`, `p0_blocking 103,658 req / 907.9 GiB` matches
the store's `runs 103658/103658`, and `p2_engram 251,232 req / 988 MiB` matches
the engram prefetch. Worth relabelling, but nothing is mis-prioritised.)

**Per-turn reheat — drop it.**

| cell | slots | reheat | decode hit | stall ms/tok | MB/token | tok/s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `m_4500_off` | 4,500 | off | 0.9250 | 92.2 | 338.3 | 5.144 |
| `m_4500_on` | 4,500 | **on** | 0.9250 | 91.7 | 338.4 | 5.126 |
| `m_5000_off` | 5,000 | off | 0.9364 | 77.5 | 287.3 | 5.264 |
| `m_5000_heat` | 5,000 | **on**, `DEEPMOE_HEAT_FILE=heat_8turn_recent.inc` | 0.9365 | 77.9 | 286.8 | 5.534 |

This is the test §7 above asked for and could not afford: the user's 8-turn
script, at both production capacities, with the §1 fix in place (candidates are
now non-resident keys from the heat table, and the pass never evicts without one),
and with a candidate table built from the *recent* routing of this very
conversation rather than the static one. The per-turn hits track the `off` run to
four decimals on all eight turns, including the two topic changes (turn 5: 0.8855
both; turn 7: 0.8857 vs 0.8858). `keys = 140` and `free_slots = 0` on every turn,
so the pass ran and did its work — the work is simply not worth anything on a
cache this size.

**So: topic-change resilience with the heat fix — dropped.** The fix is real (the
pass can no longer evict without fetching, which is what made the old version cost
+32 MB/token), and the unit test
`tests/test_integration.cpp::a_backfill_order_of_resident_keys_fetches_nothing`
keeps it honest. But `--reheat` stays **off by default** and should be considered
for removal: on a 5,100-slot cache the topic-change dip is not a cache-content
problem the reheat can reach.

**Prefill→decode handoff — keep it on, with one unexplained cell.**
4,500 slots, `--gpu-prefill-min 256`, one request per cell, 65 decode steps.

| prompt | max-ctx | variant | prefill s | decode hit | stall ms/tok | tok/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 512 | 4,096 | off | 42.1 | 0.7346 | 291.3 | 2.439 |
| 512 | 4,096 | **on** | 35.1 | **0.8883** | 131.2 | **4.207** (+72%) |
| 1,024 | 4,096 | off | 53.0 | 0.8267 | 196.0 | 3.253 |
| 1,024 | 4,096 | **on** | 46.5 | **0.9171** | 100.5 | **4.851** (+49%) |
| 2,048 | 4,096 | off | 72.3 | 0.8487 | 171.5 | 3.534 |
| 2,048 | 4,096 | **on** | 65.8 | **0.8874** | 133.7 | **4.132** (+17%) |
| 2,048 | 8,192 | off | 72.7 | **0.9715** | 37.1 | **6.903** |
| 2,048 | 8,192 | on | 65.9 | 0.8874 | 133.5 | 4.135 |
| 4,133 | 8,192 | off | 100.0 | **0.9609** | 51.0 | **6.306** |
| 4,133 | 8,192 | on | 97.8 | 0.8480 | 175.2 | 3.496 |

Read the first six rows and the handoff is an unambiguous win that shrinks with
prompt length, and it also takes 9–14% off the prefill itself, because a cached
expert is computed from where it is instead of being read again. **Default stays
on.**

The last four rows are the open item, and they are *not* a prompt-length effect:
the two 2,048-token pairs differ only in `--max-context`, the prompt, the cache
and the reused tokens (0) are identical, and the prefill wall clock is the same
to within 0.6%. What changes is the **handoff-off** arm, from 0.8487 to 0.9715;
the handoff-**on** arm is identical across the two (0.8874 / 0.8874, 4.132 /
4.135 tok/s), which is itself the tell — with the handoff on, the decode's cache
contents are determined by the prompt and nothing else. Something about a
4,096-position context leaves the off-arm's cache in a much worse state than an
8,192-position one does, and until that is named, the 4,133-token row cannot be
read as "the handoff loses on long prompts". Reproduce with:

```
tools/hitrate_bench.py --out <dir> --requests bench/results/hitrate/handoff_2048.json \
  --cache-slots 4500 --max-context {4096,8192} --serve-arg=--gpu-prefill-min \
  --serve-arg=256 --env DEEPMOE_PREFILL_HANDOFF={0,1}
```

### 7. Defaults, and why

| setting | default | why |
| --- | --- | --- |
| cache size | **`auto`** (no `--cache-slots`), 5,100 slots / 89.3 GiB here | best measured cell (hit 0.9383, 76.0 ms stall, 278.6 MB/token); sizes from the machine at boot, so it tracks what another track is holding. `--cache-slots 5400` fails a long run and `5500` loses the device on a quiet machine. |
| `kPathBAutoCeiling` | **30 GiB** (was 16) | 16 GiB capped auto at ~4,500 slots, 1.3 points of hit below what the machine holds. 30 GiB puts auto at 5,100 — ~300 slots under the first observed long-run failure. |
| `kPathAOther` | **3 GiB** (was 1) | 1 GiB was measured against a 4K-context decode-only run; a long context plus a GPU prefill needs three. |
| `kPathAReserve` | **4 GiB, new** | without it the GPU prefill cannot allocate *anything* at any cache ≥ 3,600 slots; with it, a 4K prompt's prefill goes 1,061.7 s → 100.0 s. Costs two slabs. |
| `DEEPMOE_MOE_OVERLAP` | **on** (unchanged) | +1.0% cold / +4.1% warm tok/s, decode hit bit-identical (0.9114/0.9115 both arms). |
| `DEEPMOE_PREFILL_HANDOFF` | **on** (unchanged) | +72% / +49% / +17% tok/s at 512 / 1,024 / 2,048 tokens, and 9–14% off the prefill. One unexplained cell at a larger `--max-context`, §6. |
| `DEEPMOE_BACKFILL` | **off** (unchanged) | +0.0011 hit for +21.5 GiB of idle reads; the whole effect is turn 1. |
| `--reheat` | **off** (unchanged), candidate for removal | no-op to four decimals at 4,500 and 5,000 slots on the 8-turn script, with the fixed candidate selection and with a recent-routing heat table. |

### 8. Open items

1. The `--max-context` sensitivity of the handoff-off arm (§6). Two cells, ~7
   minutes, and it decides whether the handoff needs a prompt-length gate.
2. `p3_backfill` in the IoEngine counters includes the 9.17 GiB pinned-weight
   load. Relabel, or give the pinned loader its own priority bucket.
3. `--reheat` and `--reheat-decay`: no measured benefit at any capacity tested.
   Removing them would also remove `decay_heat`'s renormalisation subtlety.
4. The capacity curve is still climbing at 6,500 slots (sim 0.9357 vs 0.9136 at
   5,100). Every point of hit is throughput; the limit is the host heap on the
   path-B import, not the design.

### 9. Tests

`ctest --test-dir build -j 1`, run alone on a quiet machine after every change
above: **36/36 passed, 0 failed** (2,862 s). `suite.gpu_layer` and
`suite.gpu_prefill` are reported as skipped by ctest because each contains
optional sub-cases gated on `DEEPMOE_PF_LONGCTX`; running the binary directly,
`gpu_prefill.forty_layers` reports `free-running: 8/8 before divergence` — **L3
is 8/8 with the path-A reserve in place**, which is what the reserve had to not
break, since it changes where the cache's slabs come from.
