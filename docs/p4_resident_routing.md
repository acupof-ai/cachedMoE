# P4 — Resident-only routing：只路由到已在缓存里的 expert（Track Y）

设计对应：[design.md](design.md) §3.1（时间模型）、§5.2（slab 池容量）、§9.2/§9.3（cache 策略）、§10（DSpark）、§12（判据）。

**一句话**：把 decode 的 stall 换成 quality loss。gate 的 6 个 expert 里只用**已经常驻**的那些，
缺的直接跳过并对剩下的重新归一化，miss 交给后台补，任何一步都不等盘。
tok/s 从 5.2 → 12.5（100 GiB 槽位下 verify-only 到 19.5），代价是**平均丢掉 6.7–7.2% 的 gate 权重质量**。
这个量级 ≈ 「平均每行少用 0.63 个第 6 名 expert」，**值得上 GPU 做一次 64-token L3 PPL**，
但 90 GB（4,500 槽）配置下**不行**（丢 12–21%）。

产物：

| 东西 | 路径 |
|---|---|
| 模拟器（CPU，全 trace 一遍 ≈ 10 s） | `tools/resident_sim.py` |
| 结果 JSON（主表 + 容量扫 + 带宽扫） | `bench/results/resident/` |
| trace | `traces/mixed`（27,399 token × 40 层 × top-6，`tools/route_trace.py` 的格式） |

方法：`tools/cache_sim.py` 的 demand 路径原样保留作基线，本工具只改「miss 时做什么」。
LRU **只被真正用到的 expert touch**——resident-only 模式下策略永远学不到被跳过的那些，这一点在模拟里是建模了的，
不是用 exact 的 LRU 状态去读 resident-only 的命中率。权重质量按 gate 的定义算：
`inference/model.py` 的 `Gate` 用**无 bias 的原始分数** gather 后归一化（`norm_topk_prob = true`），
trace 的 `top16_scores[:6]` 正是这六个分数，所以
`mass_lost = (缺席 expert 的分数和) / (六个分数之和)`，与 `routed_scaling_factor = 1.5` 无关。
时间模型：计算 2 ms/层 × 40 = 80 ms/token，一个 expert 18.8 MB / 4.5 GB/s = 4.18 ms，QD 8。

---

## 1. 校准：这个 gate 很平

| rank | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 平均权重占比 | 0.290 | 0.200 | 0.156 | 0.131 | 0.116 | 0.106 |

第 1 名只占 29%，第 6 名占 10.6%，第 5+6 名合计 **0.223**。
所以后面所有 `mass_lost` 都应该拿这三个数当尺子读：
丢 0.034 ≈ 「1/3 个末位 expert」，丢 0.067 ≈ 「0.63 个末位 expert」，丢 0.22 ≈ **等于把 top-6 变成 top-4**。

## 2. 主表（全 trace，warmup 5%，26,029 个计数 token）

`served` = 请求到的 expert 中真正用上的比例；`mass_lost` = 平均丢掉的 gate 权重质量（resident-only 的那些 step 上）。

| C（槽） | 模式 | served | mass_lost | 6/6 常驻的行 | 丢 top-1 的行 | 0/6 的行 | ms/token | tok/s |
|---|---|---|---|---|---|---|---|---|
| 4,500 (90 GB) | exact（基线） | 0.8881 | 0.0961 | 0.533 | 0.061 | 0.0003 | 192.3 | **5.20** |
| 4,500 | verify-only | 0.8623 | 0.1207 | 0.469 | 0.082 | 0.0004 | 65.4 | 15.29 |
| 4,500 | backfill（全程） | 0.7760 | **0.2071** | 0.322 | 0.167 | **0.0047** | 80.0 | 12.50 |
| 5,711 (100 GiB) | exact（基线） | 0.9204 | 0.0674 | 0.640 | 0.041 | 0.0001 | 159.8 | **6.26** |
| 5,711 | verify-only | 0.9209 | **0.0670** | 0.641 | 0.041 | 0.0001 | 51.4 | **19.46** |
| 5,711 | backfill（全程） | 0.9159 | **0.0719** | 0.630 | 0.045 | 0.0002 | 80.0 | **12.50** |
| 8,000 (150 GB，装不下) | exact | 0.9581 | 0.0344 | 0.790 | 0.018 | 0.0000 | 122.0 | 8.20 |
| 8,000 | verify-only | 0.9585 | 0.0341 | 0.790 | 0.018 | 0.0000 | 43.3 | 23.12 |
| 8,000 | backfill | 0.9581 | 0.0344 | 0.790 | 0.018 | 0.0000 | 80.0 | 12.50 |

基线校验：C = 4,500 的 exact 命中 **0.8881**（已知 0.8903）、C = 5,711 **0.9204**（已知 0.9219），
与 `tools/cache_sim.py` 的 LRU 一致，差值来自 warmup 口径。

常驻个数的分布（C = 5,711）：

| 常驻 0..6 个 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| exact | .0001 | .0007 | .0033 | .0160 | .0726 | .2670 | .6404 |
| backfill | .0002 | .0012 | .0047 | .0191 | .0769 | .2678 | .6301 |
| backfill @ C=4,500 | **.0047** | .0198 | .0516 | .1071 | .1943 | .3009 | .3217 |

**分层看基本是平的**：10 层一组的 `mass_lost`（C = 5,711 backfill）为
L00–09 0.0762 / L10–19 0.0749 / L20–29 0.0656 / L30–39 0.0708；
最差的层是 **1、0、13、19、5、15**（0.083–0.096），最好的一组是 20–29。
没有哪一组烂到需要单独给配额，所以**不需要 per-layer quota**，全局池就行。

「整个 token 的 240 个 expert 全常驻」的比例极低（C = 5,711 时 0.0011），
所以「只在全常驻时才走 resident-only」这种保守开关是没意义的——它永远不触发。

## 3. 为什么 4,500 会塌：后台补盘是有预算的

盘一秒能补 4.5 GB / 18.8 MB = **239 个 expert**；一步 80 ms，所以**每步 19.2 个**。
而 exact 基线的 miss 需求是 `240 × (1 − hit)`：

| C | 需求（expert/token） | 预算 19.2 | 结果 |
|---|---|---|---|
| 4,500 | 26.9 | 不够（140%） | 队列饱和（丢弃 19,019 次），mass_lost 0.096 → **0.207** |
| 5,711 | 19.1 | 刚好（99.5%） | 0.067 → 0.072 |
| 8,000 | 10.0 | 富余 | 0.034 → 0.034（与 exact 逐项相同） |

这是本 track 最重要的一条：**resident-only 把 NVMe 的带宽缺口从「延迟」线性翻译成「质量」**。
在 C = 5,711 上它正好卡在刀锋上——带宽扫（C = 5,711，全程 backfill）：

| 每步能补的 expert | 15 | 17 | 19.2（4.5 GB/s） | 22 | 26 |
|---|---|---|---|---|---|
| mass_lost | 0.1456 | 0.1107 | **0.0719** | 0.0674 | 0.0674 |

也就是说：盘只要掉到 4.0 GB/s（≈17/步），质量损失就从 7.2% 涨到 11%；
反过来 22/步以上就贴到 exact-LRU 的地板 0.0674，再快也没用。
**这条曲线同时意味着：把 tok/s 目标提到 12.5 以上会直接吃掉质量**，因为每步的补盘预算按比例变小。

容量扫（全程 backfill，找拐点）：

| C | 4,500 | 5,000 | 5,711 | 6,500 | 8,000 | 9,500 | 11,000 |
|---|---|---|---|---|---|---|---|
| mass_lost | 0.2071 | 0.1538 | 0.0719 | 0.0535 | 0.0344 | 0.0216 | 0.0122 |

拐点在 **≈ 5,500 槽**（96 GiB）。今天的 100 GiB slab 池（5,711）刚好在拐点右边一点点，没有余量。

## 4. verify-only（DSpark block = 5）

每块：position 1 exact（会取盘、会 warm cache），position 2..5 resident-only。
盘在 verify step 上是空闲的，所以模拟里后台补盘**在两种 step 上都跑**。
tok/s = (1 + 3.7) / (80 + stall_exact + 80) ms。

| C | stall / exact step | block | tok/s | verify 位置的 mass_lost |
|---|---|---|---|---|
| 4,500 | 147.3 ms（基线 112.3） | 307.3 ms → 4.7 tok | 15.29 | 0.1207 |
| 5,711 | 81.5 ms（基线 79.8） | 241.5 ms → 4.7 tok | **19.46** | **0.0670** |

C = 5,711 上 verify-only 有个漂亮的性质：**verify 位置的 mass_lost 与 exact 基线一样（0.0670 vs 0.0674）**，
因为每 5 步一次的 exact step 加上后台补盘，已经足够把 cache 维持在 exact 的状态。
换句话说这里**质量损失不是 verify-only 带来的，它就是 LRU 在这个容量下本来的缺席率**——
差别只在于 exact 模式会为此等盘，verify-only 不等。
C = 4,500 上则相反：warm 的机会少了 5 倍，exact step 自己的 stall 反而从 112 涨到 147 ms。

对比（C = 5,711）：demand-only 基线 6.26 tok/s → verify-only **19.5 tok/s（×3.1）**，全程 backfill 12.5 tok/s（×2.0）。
verify-only 比全程 backfill **又快又准**，因为那一次 exact step 既 warm 了 cache 又给了一个精确 token。

## 5. Router group：V4.1-Flash 没有

`D:\models\DeepSeek-V4.1-Flash\config.json` 的 `text_config` 里：
`topk_method = "noaux_tc"`、`scoring_func = "sqrtsoftplus"`、`num_experts_per_tok = 6`、
`n_routed_experts = 384`、`norm_topk_prob = true`、`routed_scaling_factor = 1.5`、`n_shared_experts = 1`，
**没有 `n_group`、也没有 `topk_group`**。
`inference/model.py` 的 `Gate.forward` 是平的一句
`indices = (scores + bias).topk(self.topk, dim=-1)[1]`——全 384 个里直接取 6，没有任何 group mask。
所以 V3/R1 那套 group-limited routing 在这个模型上**不存在**，没有「按 group 缓存 / 淘汰」这回事。

作为反证，把 384 个 id 按连续块切成 G 组，量一个 (token, 层) 的六个 expert 落在几个不同块里：

| G | 每块 expert 数 | 平均落在几块 | 六个全分开的比例 | 若以块为取盘单位的放大倍数 |
|---|---|---|---|---|
| 8 | 48 | 4.44 | 0.083 | ×35.6 |
| 16 | 24 | 5.17 | 0.361 | ×20.7 |
| 32 | 12 | 5.58 | 0.633 | ×11.2 |
| 64 | 6 | 5.81 | 0.818 | ×5.8 |

六个 expert 几乎总是散在不同块里，所以即使人为造一个 group 作为取盘/淘汰单位，
也要多读 5.8–35.6 倍的字节。**group 粒度这条路关掉。**

## 6. 判决与下一步（需要 GPU）

**判决：值得做一次 GPU PPL 实验，只在 C = 5,711 上，优先 verify-only。**
理由是三个数：(a) C = 5,711 上 verify-only 丢的质量 0.0670 与 exact LRU 的缺席率 0.0674 **相同**——
这不是新的近似，是把已有的缺席从「等盘」改成「跳过」；
(b) 0.067 ≈ 0.63 个末位 expert，而这个 gate 的末位只占 10.6%，远没到 top-4 那种 0.22 的量级；
(c) 丢掉 top-1 的行只有 4.1%，0/6 的行 1e-4。
**不做**的是 C = 4,500：0.207 恰好等于「top-6 → top-4」，而且每 213 行就有一行 6 个 expert 全缺
（约每 5 个 token 就有一层只剩 shared expert），这不像还是原来那个模型。

需要 GPU 的，按顺序：

1. **`DEEPMOE_ROUTE_RESIDENT_ONLY` 引擎开关**（三档：`off` / `verify` / `all`）。
   实现面很小：`ExpertStore` 的取用点在 miss 时返回「不常驻」而不是发起同步取盘，
   MoE dispatch 在剩下的子集上重新归一化 gate 权重（缺席数 = 6 时退化为只有 shared expert），
   miss 进后台队列（按 gate 分数优先，不是 FIFO——模拟里 C = 4,500 的 FIFO 丢了 19,019 次，分数优先应当更好，**未测**）。
   估：约 100–150 行 + 一个 CPU 单测（给定常驻集合，权重和为 1）。
2. **64-token L3 PPL**：同一个 L3 prompt，teacher-forced 64 步，比 `off` / `verify` / `all` 三档的 PPL 与 top-1 一致率。
   定价：resident-only 64 × 80 ms ≈ 5 s/次，exact 64 × 160 ms ≈ 10 s/次，加 prefill 与 cache warm，
   **一次 ≈ 1–2 分钟，三档 + 两个容量 ≈ 10 分钟一个独占 GPU 槽**。
   判据建议：PPL 相对 `off` 上升 < 5%，且 §1.3 (a) 的自由运行分歧仍只落在参考的近似平局上。
3. 只有 (2) 过了才谈 **runtime 的后台补盘队列**（今天 `store/engram_prefetch.cpp` 那条路的复用）与
   §10 的 DSpark 集成；(1)(2) 之前不写 planner 策略代码。

未做 / 已知的空白：

* 后台队列用的是 FIFO + 容量 2,048 的丢弃，没试过按 gate 分数优先或按层优先；
* 模拟的是「跳过并归一化」，没有模拟「用第 7..16 名里已常驻的 expert 顶替」这一变体（trace 里有 top-16，做得了，没做）；
* PPL 只能上 GPU 量——本 track 全部在 CPU 上，任何「输出还正常」的说法都还没有证据；
* C = 8,000 只是参照量，150 GB 在这台机器上装不下。

---

## 7. GPU results（2026-09-18，Track Y step 2，实测）

**一句话：NO-GO。** `all` 档在真机上跑出来的 `mass_lost` 是 **0.52**，不是模拟的 0.072；
输出直接坏掉。模拟第 3 节那条「后台补盘是有预算的」是对的，但它低估了**反馈**：
resident-only 让 LRU 只被服务过的 expert 触摸，cache 于是偏离真实路由，miss 变多，
跳过变多，再偏离更多。C = 5,711 上模拟算的是「静态 19.1 个 miss/token」，
真机量到的是 **129 个/token**（64,948 skipped / 20,075 layer-step × 40 层），盘的预算差 6.7 倍。

实现（本 track 写的，全部 additive、默认关）：

| 东西 | 路径 |
|---|---|
| 归一化 + 跳过的纯函数 | `runtime/resident_route.h` |
| 引擎接线、计数器、`DEEPMOE_ROUTE_RESIDENT_ONLY=off\|all` | `runtime/engine.cpp` / `.h` |
| `--resident-only off\|all`（run / serve）、`--warm-cache`（run） | `cli/deepmoe_main.cpp` / `cli/serve.cpp` |
| CPU 单测（6 例，变异可杀） | `tests/test_resident_route.cpp` |
| 结果 | `bench/results/resident/chat_off`、`chat_all` |

MoE runner 的槽位是**固定七个**（`moe_bridge.cpp` 断言 `topk + 1 == slots`），所以被跳过的槽
不是删掉，而是借用第一个常驻 expert 的 id + 权重**恰好 0**；
`gpu/shaders/moe_gateup.slang` 在 dispatch A 里乘 `RouteW[m * num_slots + slot]`，
所以那一槽的 h 是精确的 0，dispatch B 加 0。代价是多算一个 expert 的算术——本 track 要省的是**等盘**，不是 FLOPs。

### 7.1 64-token L3 PPL：没有这个 harness，实际做的是 8 步

F5 的 `tools/quant2_l3.py` 是**参考实现侧**的（`dsref` + CPU shim），一遍 64 token 要 388 s，
而且它根本不过引擎，量不到引擎的路由。`tests/data/l3` 的导出只有 **8 个 greedy step**（`steps_exported = 8`），
所以「64-token teacher-forced」在这条路上不存在。实际做的是：
`run --state tests/data/l3 --steps 8 --teacher-force --warm-cache`，
PPL = `exp(mean −log p(参考的下一个 token))`，从 head 的 logits 在主机侧算（`Engine::last_logits`，每步都有）。
因为被 teacher-force 的序列**就是 fp32 参考自己的 greedy 续写**，绝对值很小（1.57），
能读的是**同样 8 步上两档之间的比值**——这正是判据要的东西。`off ≈ 29.26` 没有对应物，不要再引用。

`--warm-cache`（新增）在解码前用 static heat 表把 5,100 槽全部填满（20.8 s），
否则 `run` 是冷 cache，两档都没有意义。

| 5,100 槽，L3，teacher-forced 8 步 | off | all |
|---|---|---|
| PPL | **1.5688** | **4.1273**（×2.63） |
| 对 fp32 参考的 top-1 一致 | 8/8 | **6/8** |
| tok/s | 2.71 | 9.68（×3.6） |
| served | 0.739（hit） | **0.7552** |
| mass_lost | — | **0.2232** |
| 只剩 shared expert 的 layer-step | — | 2 / 320 |
| nvme_stall 占比 | 73.5% | 4.8% |

判据是 `all ≤ 1.05 × off`，实测 **×2.63**。**不过。**
注意这里的 0.2232 恰好是第 1 节那把尺子上的「top-6 变成 top-4」。
这一档的 cache 是 static heat 先验填的（不是这段对话自己的 LRU），所以它对 `all` 偏不利——
下面 7.2 用对话自己的 warm cache 又量了一遍，结果**更差**。

### 7.2 4 轮对话（`bench/results/hitrate/y_turns.json`，每轮 64 token，5,100 槽，backfill 开）

`tools/hitrate_bench.py --env DEEPMOE_BACKFILL=1 --env DEEPMOE_ROUTE_RESIDENT_ONLY=off|all`，一次一个进程。

| 轮 | off tok/s | off decode hit | all tok/s | all decode hit |
|---|---|---|---|---|
| y0 | 4.29 | 0.885 | 9.81 | 1.000 |
| y1 | 4.74 | 0.907 | 9.34 | 1.000 |
| y2 | 4.89 | 0.915 | 9.13 | 1.000 |
| y3 | 4.86 | 0.916 | **2.34** | 1.000 |

`all` 的 hit 恒等于 1.000 是定义使然（只路由到常驻的），**它不是质量指标**，真正的指标是：

| 全程（20,075 个 layer-step） | off | all |
|---|---|---|
| experts requested | 120,450 | 120,450 |
| served | 104,286 (0.866) | **55,502 (0.4608)** |
| skipped | 0 | **64,948** |
| gate mass lost（均值） | 0 | **0.5195** |
| 只剩 shared expert 的 layer-step | 0 | **1,753**（8.7%） |
| 后台补盘 enqueued / refused | — | 19,356 / **45,592** |
| 盘：busy / 有效带宽 | 85.8 s / 3.99 GB/s | 91.5 s / **4.12 GB/s** |
| 盘的流量分布 | P0 290 GB | **P3 359 GB** |
| 盘的平均延迟 | 24.3 ms | **5,594 ms** |
| cache eviction | 12,548 | 14,401 |

**盘一直是满的**——两档都把 NVMe 跑到 4 GB/s。区别只是 `off` 在等它、`all` 不等。
P3 的平均延迟 5.6 s 说明补盘**完全跟不上**：一个 100 ms 的 step 要的 expert 56 步之后才到。
`refused` 的 45,592 是本实现的在飞上限（48 个 chunk）挡掉的，但放开也没用——盘已经饱和。

### 7.3 输出还正常吗：不正常

off（y2 开头，连贯）：

> \# Host Waits vs. Device Waits on a Timeline Semaphore
> \#\# 1. The Two Kinds of Wait
> A timeline semaphore can be waited on from two very different places: …

all（y0 全部，退化）：

> 2022027: 2022027: 2027 2027 2027 2027 202\# 202 202 202 202 202 202 202 …

all（y1，半连贯但词都烂了）：

> A timeline semaphore is a **monfon** (counter that advances monotononly, not only via signal/w wait operations. …

y0 和 y3 是纯噪声，y1/y2 是「看起来像但词是碎的」。这与 0.52 的 mass_lost 和 8.7% 的
「整层只剩 shared expert」完全一致。

### 7.4 为什么模拟乐观了 7 倍

模拟把 miss 需求当成**外生**的：`240 × (1 − hit_exact)`，在 C = 5,711 上是 19.1/token，正好等于盘的预算。
真机上它是**内生**的：跳过的 expert 不被 touch → LRU 里它老化 → 下一次更可能不在 → 跳过更多。
静态的 19.1 变成实测的 129。第 3 节那张带宽扫表的 15 → 0.1456 那一格已经指向这个方向
（预算掉 20% 质量就翻倍），但它没有把「质量掉 → 需求涨」这一环接回去，
所以它算的是不动点之外的一次迭代，不是不动点。

第 4 节「verify-only 在 C = 5,711 上 mass_lost 与 exact 相同」的论证依赖同一个假设
（每 5 步一次的 exact step 足以把 cache 维持在 exact 状态），**因此也不再可信**，
在 p4/fin-t 上照搬之前必须先量。

### 7.5 试过 / 回退

* **压低在飞上限**（48 个 chunk 的 P3 闸门，`Engine::rr_inflight_cap_`）：挡掉 45,592 次入队，
  但盘本来就满，放开只会把平均延迟从 5.6 s further 推高。留着，没调。
* **让被跳过的 expert 也 touch LRU**：没试。它会把 LRU 变回 exact 的状态（好），
  但那样 cache 就会为从不被计算的 expert 腾位置（坏）。模拟里明确建模了「不 touch」这一支，
  换一支要重跑模拟，不是一个引擎改动。
* **按 gate 分数给后台队列排优先级**（第 6 节的未做项）：在 0.52 的 mass_lost 面前是二阶的，没做。
* **C = 4,500 的 all 档**（step 2c）：没跑。5,100 已经塌了，4,500 只会更塌，机时省下来。

### 7.6 判决

* **`all` 作为默认：NO-GO。** ×2.63 PPL、8.7% 的层只剩 shared expert、两轮纯噪声。
* **在 DSpark 上做 verify-only 变体（p4/fin-t）：NO-GO / 暂缓。**
  它的全部论证来自第 4 节，而第 4 节和第 2 节用的是同一个「补盘跟得上」的假设，
  这个假设已经被 7.2 证伪。要做的话，先量一次**真机上的 verify 位置 mass_lost**，
  而不是拿模拟的 0.0670 当前提。
* 代码留下（默认 off）：开关、计数器和单测是量这件事的唯一方式，删了下次还要重写。
