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

> **第 8 节接着把 7.4 诊断的两条都修了**：mass lost 0.5195 → 0.2587，补盘延迟 5,594 ms → 97 ms，
> 四轮对话都连贯；判据仍不过。下面 7.5 的「没试」项已经被第 8 节取代。

---

## 8. 修两个 bug 之后再量一遍（2026-09-18，Track Y step 3）

**一句话：反馈螺旋修好了，判据还是不过。**
第 7 节诊断的两条都是真的，也都修得动：`all` 的 gate mass lost 从 **0.5195 → 0.2587**，
「整层只剩 shared expert」从 8.7% → 4.4%，后台补盘延迟从 **5,594 ms → 97 ms**，
4 轮对话从「两轮纯噪声」变成**四轮都连贯**。
但 L3 8 步 PPL 仍是 **×2.56**（判据 ×1.05 / ×1.3）。
新加的 `stall1` 档把质量几乎修满（mass lost **0.0478**，0 个 shared-only 层，
输出逐字跟着 `off` 走），代价是**它不快**：4.81 → 5.14 tok/s，只有 **×1.07**。

### 8.1 改了什么（全部 additive，默认 off）

| 修 | 路径 | 内容 |
|---|---|---|
| A. 需求戳 | `runtime/engine.cpp`、`store/planner.h` | 每个**被请求**的 top-k expert 都拿一个 LRU 戳，常驻与否都算 |
| B. 有界队列 | `runtime/engine.cpp`、`runtime/resident_route.h` | 后台 miss 队列只留最近 N 步，最新优先，在飞上限 24 个 expert |
| C. `stall1` | `runtime/engine.cpp`、`cli/*.cpp` | 每层最多阻塞在**一个**门控权重最高的缺席 expert 上（1 次 P0） |

**A（需求戳）**。第 7.5 节把「让被跳过的 expert 也 touch LRU」列为未试，理由是
「cache 会为从不被计算的 expert 腾位置」。真实的坑比这个细：常驻的被请求 expert 其实**一直**被
`Planner::plan_layer` 的 `lookup` 戳着；漏的是**非常驻的那一半**——它没有槽位可戳，
所以 `Planner::fetch` 用了**提交时刻**的 `next_stamp()`。于是一个晚了 5.6 s 才落盘的 expert
**一进来就是 cache 里最新的东西**，转头把真正在用的踢掉。
改法：请求时取一个 demand 戳存进 `rr_demand_`（key → stamp），交给 P3 fetch 当 `stamp_in`，
`ExpertStore::settle_locked` 把它写进 `last_use_token`。
落地的 expert 于是**按「什么时候被要过」计龄**，不是按「盘什么时候轮到它」。
`Planner::demand_stamp()` 是为此开的一行公开访问器（和 `plan_layer` 的命中用同一把时钟）。

**B（有界队列）**。第 7.2 节说 `refused` 的 45,592 是在飞上限挡的、「放开也没用」——
这话对了一半。挡住的是**提交**，没挡住的是 **IoEngine 自己的 P3 队列**，那条队列没有上限：
19,356 个 expert 排进去，盘满速 4.12 GB/s 地读，平均 5,594 ms 才落一个。
一步 100 ms，所以盘读的是 **56 步以前**想要的东西。
改法：miss 先进 Engine 自己的 `rr_queue_`（带步号），每层做三件事——
把早于 `resident_queue_cutoff(token, N)` 的整条丢掉（不提交）、
采一次队列深度、**从队尾（最新）**往外发，直到在飞 24 个 expert 为止。
`DEEPMOE_RESIDENT_QUEUE_STEPS` 默认 **2**（`DEEPMOE_RESIDENT_QUEUE_EXPERTS` 默认 24）。
盘搬的字节数不变，**变的是搬哪些字节**。

**C（`stall1`）**。`DEEPMOE_ROUTE_RESIDENT_ONLY=stall1` / `--resident-only stall1`：
一层里缺席的 expert 里挑**门控权重最高**的那一个，发一次 P0 并等它（18.8 MB / 4.5 GB/s ≈ 4 ms），
其余照 `all` 跳过。实测每次 P0 花 **5.6 ms**。约 40 行。

CPU 单测 `tests/test_resident_route.cpp` 加到 8 例（窗口截断的边界、0 窗口夹到 1、
起步不下溢、窗口单调滑动；队列深度与 stall1 计数器的口径）。

### 8.2 L3，teacher-forced 8 步，5,100 槽，`--warm-cache`

`run --model ... --state tests/data/l3 --steps 8 --teacher-force --warm-cache --resident-only off|all|stall1`

| | off | all（step 2） | **all（step 3）** | **stall1** |
|---|---|---|---|---|
| PPL | **1.5688** | 4.1273 (×2.63) | **4.0143 (×2.56)** | **5.3336 (×3.40)** |
| 对 fp32 参考的 top-1 一致 | 8/8 | 6/8 | 6/8 | 6/8 |
| tok/s | 2.72 | 9.68 | **10.20 (×3.75)** | 3.59 (×1.32) |
| served | 0.739 (hit) | 0.7552 | 0.7516 | **0.8656** |
| gate mass lost | — | 0.2232 | 0.2258 | **0.1061** |
| 只剩 shared expert 的 layer-step | — | 2 / 320 | **0 / 320** | **0 / 320** |
| 后台队列 深度均值 / 峰值 | — | 无界 | 25.0 / 95 | 5.9 / 45 |
| 后台补盘延迟（P3 提交→落盘） | — | — | **67 ms** | 99 ms |
| 丢弃的过期 miss | — | — | 22 | 0 |
| stall1 的 P0 | — | — | — | 244 次 / 1,336 ms（5.5 ms 一次） |

**这张表上 A/B 几乎没动 PPL**，因为这个 harness 的 cache 是 static heat 先验填满的、只跑 8 步：
缺席集合由先验决定，反馈螺旋根本来不及转起来。它量到的是**先验有多准**，不是**策略有多稳**。

重复性：`stall1` 三次跑出同一个 5.3336（确定性）；`all` 三次是 4.0143 / 4.5306 / 4.7539。
两档**都在同样的第 6、7 步上分叉**（`off` 自己在第 6 步只有 90/240 命中，是最难的一步），
所以 4.0 和 5.3 的差别是**两个已经分叉的位置上分配了多少概率质量**，
8 步分不开 ×2.5 和 ×3.4。能读的只有一句：**都远在 ×1.05 和 ×1.3 之外。**
`stall1` 的 mass lost 只有 `all` 的一半却 PPL 更高，就是这个 harness 分辨率不够的直接证据。

### 8.3 4 轮对话（`y_turns.json`，每轮 64 token，5,100 槽，backfill 开）

`tools/hitrate_bench.py --script bench/results/hitrate/y_turns.json --cache-slots 5100
--env DEEPMOE_BACKFILL=1 --env DEEPMOE_ROUTE_RESIDENT_ONLY=off|all|stall1`，一次一个进程。
结果在 `bench/results/resident/y3_off`、`y3_all`、`y3_stall1`。

| 全程（≈20,000 个 layer-step） | off | all（step 2） | **all（step 3）** | **stall1** |
|---|---|---|---|---|
| decode tok/s（四轮） | 4.75 4.71 4.88 4.88 | 9.81 9.34 9.13 **2.34** | **9.06 9.02 8.96 8.92** | 4.88 5.22 5.36 5.09 |
| 均值 / 相对 off | 4.81 | — | **8.99 (×1.87)** | 5.14 (**×1.07**) |
| experts requested | 120,480 | 120,450 | 117,822 | 120,240 |
| served | 105,284 (**0.874**) | 55,502 (0.4608) | 86,280 (**0.7323**) | 112,629 (**0.9367**) |
| gate mass lost | 0 | **0.5195** | **0.2587** | **0.0478** |
| 只剩 shared expert 的 layer-step | 0 | 1,753 (8.7%) | **859 (4.4%)** | **0** |
| 后台 enqueued / refused / 过期丢弃 | — | 19,356 / 45,592 / — | 10,150 / 0 / **21,151** | 4,354 / 0 / 3,158 |
| 后台队列 深度均值 / 峰值 | — | 无界 | 67.9 / 419 | 10.0 / 297 |
| 后台补盘延迟（P3 提交→落盘） | — | — | **97.1 ms** | 172.3 ms |
| 盘 平均延迟（全类，IoStats） | 24.6 ms | **5,594 ms** | **55.2 ms** | 43.0 ms |
| 盘 有效带宽 | 3.97 GB/s | 4.12 GB/s | 3.91 GB/s | 3.69 GB/s |
| 盘 搬的总字节 | 322 GB | 377 GB | **213 GB** | 292 GB |
| 其中 P0 / P3 | 273 / 34 GB | 0.3 / 359 GB | 0.3 / 203 GB | **180 / 98 GB** |
| cache eviction | 11,469 | 14,401 | **5,703** | 9,887 |

三件事值得单独拎出来：

1. **螺旋确实断了。** `all` 的 served 从 0.4608 回到 0.7323，eviction 从 14,401 掉到 5,703，
   盘搬的字节少了 44%（377 → 213 GB）而质量翻倍好转。这正是 A+B 想要的：
   盘不再为 56 步以前的需求做无用功，落地的 expert 也不再顶着假的「最新」戳踢掉在用的。
2. **过期丢弃 21,151 vs 入队 10,150**：`all` 的需求仍然是盘的 2 倍——
   第 3 节「后台补盘是有预算的」这条**没被推翻**，只是现在超出预算的部分是**明着丢最旧的**，
   而不是排在队里把盘堵死。这也是为什么 `all` 的 mass lost 停在 0.26 而不是回到 0.067。
3. **`stall1` 的账是反的。** 它把质量买回来了（0.0478 已经低于模拟里 exact-LRU 自己的缺席率
   0.0674），但一层一次 P0 让 P0 流量回到 180 GB，tok/s 只剩 **×1.07**。
   花 56.5 s 的 P0（10,055 次 × 5.6 ms）换 7% 速度，这笔交易本身就不成立，和 PPL 判据无关。

### 8.4 输出：`all` 恢复连贯，`stall1` 逐字跟着 `off`

第 7.3 节的两轮纯噪声没有了。四轮都成句、都有正确的 Markdown 结构。
和 `off` 的贪心输出做公共前缀：

| 轮 | all 与 off 的公共前缀 | stall1 与 off 的公共前缀 |
|---|---|---|
| y0 | 0 字符（标题就不同） | **137** |
| y1 | 2 | **122** |
| y2 | 15 | **152** |
| y3 | 50 | **92** |

`stall1` 四轮的**标题和小节结构与 `off` 完全一致**（y0「# Vulkan Timeline Semaphores:
Cross-Queue Ordering and Out-of-Order Waits / ## 1. What a Timeline Semaphore Is」、
y3「# Sizing an Expert Cache for a Long Conversation / ## 1. What the Cache Is Actually Holding」），
之后才在措辞上分岔——这是贪心解码在近似平局上正常的分歧。
`all` 则从第一句就换了骨架，y3 里还出现了 `runtime (exture, ...` 这种碎词，
和 0.26 的 mass lost、4.4% 的 shared-only 层一致。

### 8.5 试过 / 回退

* **在飞上限 24（`DEEPMOE_RESIDENT_QUEUE_EXPERTS`）**：留着。`refused` 从 45,592 变成 **0**，
  因为压力现在由窗口（丢最旧）承担，不再由闸门承担。
* **窗口 N=2**：默认。没扫 N=1/4——`all` 的过期丢弃已经是入队的 2 倍，
  N 再大只会把更旧的东西塞进同一条满带宽，N=1 只会多丢；这条曲线的两端都已经被 8.3 的账算死了。
* **`stall1` 挑「权重最高的缺席者」**：没试过别的挑法（按层、按缺席个数）。
  在 ×1.07 的 tok/s 面前是二阶的。
* **让 `all` 在 4,500 槽上跑**：仍然没跑。5,100 的 mass lost 还有 0.26，更小只会更差。
* **给 8 步以上的 teacher-forced PPL**：`tests/data/l3` 只导出了 8 个 greedy step，
  这是现在**唯一**挡在判决前面的东西（见 8.6）。

### 8.6 判决

* **(i) `all` 作为默认：NO-GO。** PPL ×2.56（判据 ×1.05 / ×1.3），
  gate mass lost 0.2587 ≈ 第 1 节尺子上的「top-6 变成 top-4」，4.4% 的 layer-step 只剩 shared expert。
  比 step 2 好一倍，但好一倍的 0.52 还是 0.26。**两个 bug 不是全部原因**：
  剩下的是第 3 节那条硬预算——需求仍是盘的 2 倍（21,151 丢弃 vs 10,150 入队）。
* **(ii) `stall1`：NO-GO，但理由和 (i) 不同。**
  质量这一侧它基本是干净的（mass lost 0.0478、0 个 shared-only 层、输出结构与 `off` 一致）；
  **它输在收益**：4.81 → 5.14 tok/s，×1.07，换来 180 GB 的 P0 和 56.5 s 的等待。
  L3 8 步给的 ×3.40 PPL **不该当作它的判据**——同一张表上它的 mass lost 只有 `all` 的一半，
  8 步、两个分叉位置的 harness 分不开这两档（8.2）。
  代码留着（默认 off）：它是目前唯一一个把 resident-only 的质量损失压到 exact-LRU 缺席率以下的档。
* **(iii) 在 DSpark 上做 verify-only：仍然 NO-GO / 暂缓，但卡点换了。**
  step 2 的卡点是「第 4 节的前提被证伪」；现在前提的**因果链**（补盘跟不上 → 质量掉）
  已经被 A+B 修掉一半，verify-only 的 cache 状态会落在 `stall1`（0.048）和 `all`（0.259）之间。
  真正挡路的已经不是路由策略，而是**没有能分辨 ×1.05 和 ×1.3 的 PPL harness**：
  `tests/data/l3` 只有 8 个 teacher-forced step，两档在同样的第 6、7 步分叉，
  PPL 比值是两个已分叉位置上的概率质量，噪声比信号大。
  **下一步只有一件事**：导出 ≥ 64 个 teacher-forced step（或换一个过引擎的 PPL 路径），
  在上面重量 `off` / `stall1` / verify-only。在那之前不要再写 planner 策略代码。

> **已完成，见第 9 节。** 64 步的 harness 做出来并量完了：`all` 的判决不变（×1.82 – ×2.29），
> `stall1` 的质量判决**翻了**（×1.09 – ×1.11，在 ≤1.3 的 verify-only 带里）。
> **本节 8.2 给 `stall1` 的 ×3.40、以及上面 (ii)/(iii) 的理由，以第 9 节为准。**

---

## 9. 64 步的 teacher-forced harness，判决重量（2026-09-18，Track Y step 4）

**一句话：8.6 要的尺子做出来了，量完之后 `all` 的判决不变，`stall1` 的质量判决翻了。**
64 个 teacher-forced 位置上，`all` 是 **×1.82–2.29**（仍然远在两条线外），
而 `stall1` 是 **×1.09–1.11**——落在 **≤1.3 的 verify-only 带里**，不到 ≤1.05 的 GO 线。
8.2 那张表给 `stall1` 的 **×3.40 是 8 步 harness 的假象**，本节取代它。

### 9.1 harness：`traces/l3_64` + `tools/l3_ppl.py`

| 件 | 路径 | 内容 |
|---|---|---|
| 导出器 | `tools/oracle_l3_ppl.py` | 64 token 自然语料 prompt + **64 个参考自己的 greedy 续写**，逐步导出 |
| 驱动 | `tools/l3_ppl.py` | 三档各起一个进程**串行**跑 `run --teacher-force`，出表 + 判据 |
| ctest | `tests/CMakeLists.txt` `bench.l3_ppl64` | `needs-model;needs-gpu;bench`，`RUN_SERIAL`，没有导出集就 skip |
| 数据 | `traces/l3_64`（**.gitignore**，5.83 MB + 0.52 MB engram 表） | 65 条记录 |
| 结果 | `bench/results/resident/ppl64/{off,all,stall1}.log` + `table.json`，`rep2/`、`rep3/` | |

导出集的形状和 `tests/data/l3` 是同一个容器（`runtime/decode_state.h` 原样读），只改了一件事：
**每步只存 logit 证据**（top-64 的 (id, logit)、整向量的 max/logsumexp/min、输入 token、argmax
和 argmax 自己的 log 概率），**不存每步的 compressed KV 和 top-k**。
Track Q 之后引擎自己产这两样（`Engine::produce_ced()`），从来不读它们，
去掉之后 64 步的导出从 ~45 MB 变成 5.83 MB。prefill 记录一字没动——
引擎仍然从它 seed 整个 prompt 的注意力状态。

参考的续写是**参考自己逐步解码出来的**（`phase greedy`，和 `oracle.py --level l3` 同一个循环，
所以每步的注意力状态是**增量**建起来的，和引擎一样）：
prefill 179 s + 64 × 23.4 s = **1,687.8 s**。

**试过并放弃的捷径，以及它牵出来的一条真事**（`phase prefill` + `phase steps` 留着当探针）：
teacher forcing 本来一遍 128 token 的 prefill 就能一次拿到全部 65 个 next-token 分布
（实测 **226.5 s**，比逐步快 7.4 倍，也是 F5 `quant2_l3.py` 的论证）。
但它要求续写**事先已知**，而续写按定义就是参考的 greedy 路径。
于是让**引擎**先自由跑一版候选（GPU 上 21 s），再用这一遍去**证明**它
（位置 63 的 argmax 必须是 c[0]，位置 64+s 的必须是 c[s+1]）。实测 **5/64**。

**但这不是引擎的锅。** 把同一遍原样跑在**参考自己逐步解码出来的**那 64 个 token 上
（`traces/l3_64_verify`，同样 227.1 s），分叉的位置和分叉的 token **一模一样**：
index 5，一遍式给 ` either`，参考自己逐步给的是 ` about`。两次对照合起来只能读成一句话：

> **参考的「一遍 prefill 的第 j 个位置」和「逐步解码到第 j 个位置」不是同一个分布。**

（compressor 结尾那个不完整分组的进位、indexer 在整段 compressed cache 上的 top-k 选择，
在一遍式里都不是逐位置因果的。）所以一遍式在**中间位置**上给的 argmax
**不能**当 greedy 路径的定义，它作为 teacher-forced PPL 的参考分布也是错的靶子——
引擎是**逐步**解码的。顺带量到的对照：
**引擎自由跑跟住参考逐步 greedy 12 个 token**，而一遍式只跟住 5 个——
这一局**引擎比捷径准**。导出必须走 `phase greedy` 的理由就在这里。

### 9.2 harness 自己的门：`off` 对参考

导出集里存了参考自己在这 64 个位置上的平均 NLL（`reference_nll` = `max − logsumexp` 的均值）。
比值全部是对 `off` 量的，所以先看 `off` 离参考多远：

| | 参考（fp32 / CPU） | `off`（fp4 / GPU） |
|---|---|---|
| NLL | **0.597555** | **0.630051** |
| PPL | **1.8177** | **1.8777**（**×1.0330**） |
| top-1 | 64/64（定义如此） | **61/64** |

**×1.0330 就是这把尺子的噪声底。** 下面的 `stall1` 是 ×1.09，只有噪声底的三倍——
读得出来，但不要把 1.09 和 1.05 的差别当成什么大数字。

### 9.3 结果（5,100 槽，`--warm-cache`，teacher-forced 64 步，一次一个引擎）

`tools/l3_ppl.py --state traces/l3_64 --steps 64`

| | 参考 | **off** | **all** | **stall1** |
|---|---|---|---|---|
| NLL | 0.597555 | **0.630051** | 1.456643 | **0.732431** |
| PPL | 1.8177 | **1.8777** | **4.2915** | **2.0801** |
| 对 off 的比 | 0.968 | 1.000 | **×2.286** | **×1.108** |
| 对参考的 top-1 一致 | — | **61/64** | 42/64 | 51/64 |
| tok/s | — | 3.19 | 8.75（×2.74） | 3.87（×1.21） |
| cache hit | — | 0.8150 | 1.0000 | 1.0000 |
| served | — | — | 10,536 (0.6859) | **13,825 (0.9001)** |
| gate mass lost | — | — | **0.3054** | **0.0760** |
| 只剩 shared expert 的 layer-step | — | — | 52 / 2,560 | **0 / 2,560** |
| 后台 入队 / 过期丢弃 | — | — | 2,687 / 2,072 | 1,218 / 309 |
| 后台补盘延迟 | — | — | 66.5 ms | 100.4 ms |
| stall1 的 P0 | — | — | — | 1,624 次 / 8,937 ms |
| nvme_stall 占比 | — | 64.5% | 3.0% | 1.0% |

重复性（同一个导出集，三次独立进程）：

| | 1 | 2 | 3 | 对 off |
|---|---|---|---|---|
| off | 1.8777 | 1.8777 | 1.8777 | 确定性 |
| all | 4.2915 | 3.4223 | 4.0564 | **×1.82 – ×2.29** |
| stall1 | 2.0801 | 2.0704 | 2.0453 | **×1.089 – ×1.108** |

**尺子确实变准了**，三处可以直接看出来：

1. **两档不再撞在同一个分叉点上。** 8 步时三档都是 6/8 或 8/8，分不开；
   64 步时 top-1 是 **61 / 42 / 51**，`stall1` 和 `all` 中间隔了 9 个位置。
2. **PPL 终于和 mass lost 同向了。** 8.2 里 `stall1` 的 mass lost 只有 `all` 的一半、PPL 却更高，
   那是 harness 的自相矛盾；64 步上 0.0760 对 0.3054，PPL 是 ×1.11 对 ×2.29，**单调**。
3. **`stall1` 三次跑进 0.0035 的带里**，`all` 三次跨 0.87 个 PPL——
   后者的方差是真实的（后台补盘的时序），不是测量噪声，而且**三次都在 ×1.3 外**。

### 9.4 判决（取代 8.6 的 (ii) 和 (iii)）

* **(i) `all` 作为默认：NO-GO，不变。** 64 步上 **×1.82 – ×2.29**（判据 ×1.05 / ×1.3），
  mass lost 0.3054，52 个 layer-step 只剩 shared expert，top-1 掉到 42/64。
  8 步给的 ×2.56 落在这个区间里，**结论没有被 harness 改变**——第 3 节那条硬预算仍然是原因
  （2,072 个过期丢弃 vs 2,687 个入队）。
* **(ii) `stall1` 的质量判决翻了：从「×3.40，比 `all` 还差」变成「×1.09 – ×1.11，在 verify-only 带里」。**
  8.6 已经怀疑过这一点（「同一张表上它的 mass lost 只有 `all` 的一半」），现在是量出来的。
  但**作为默认仍然是 NO-GO**，理由和 8.6 的 (ii) 一样、只是现在两侧都有数：
  质量差 ×1.09 > ×1.05 的线，速度这条 harness 上是 ×1.21（8.3 的四轮对话里是 ×1.07），
  而买来的代价是 1,624 次 P0、8.9 s 的等待。
* **(iii) 在 DSpark 上做 verify-only：卡点解除，判据这一侧过了。**
  8.6 说「真正挡路的是没有能分辨 ×1.05 和 ×1.3 的 harness」——现在有了，
  而且 `stall1` 这一档的 cache 状态**落在 ≤1.3 带内**（×1.09，mass lost 0.076，0 个 shared-only 层）。
  也就是说 verify-only 的质量前提**成立**；下一步该量的是它的**收益**
  （verify 批 M 上的 tok/s），不再是它的质量。
  `all` 那一档仍然不能当 verify 的 cache 状态（×1.8 – ×2.3）。

### 9.5 重跑

```
.venv/Scripts/python.exe tools/oracle_l3_ppl.py greedy \
    --model D:/models/DeepSeek-V4.1-Flash --out traces/l3_64 --steps 64   # ~28 min, CPU
.venv/Scripts/python.exe tools/l3_ppl.py --state traces/l3_64 --steps 64  # ~3 min, GPU
ctest --test-dir build -R bench.l3_ppl64                                  # 同一件事
```

`index.json` 每步重写一次，所以中途杀掉留下的是一个**更短但完整、能加载**的导出集；
`tools/l3_ppl.py` 会照实说它只有几步。导出集在 `traces/`（`.gitignore`），
5.83 MB，可再生；committed 的只有两个脚本、ctest 一条和 `bench/results/resident/ppl64/` 的抄本。

---

## 10. verify-only 在真机上量出来了：判据这一侧**没过**（2026-09-18，Track Z）

§9.4 的 (iii) 说「verify-only 的质量前提成立，下一步该量的是它的收益」。
本节把这两件事都量了，结论是 **(iii) 的前提被推翻**：
`stall1` 那一档的质量（×1.09、mass lost 0.076、0 个只剩 shared expert 的层）
**不是 verify-only 的 cache 状态**，因为 `stall1` 每一步都做一次 P0 取盘，
而 verify-only 的四个 draft 位置一次也不做。把 P0 的刷新率降到 1/5 之后，
质量落到 **×1.376**——连 ≤1.3 那条带都出去了。

### 10.1 先说清楚**没能**量的那一半：投机解码在引擎里不存在

任务要的是「接受率 / 每 block 接受的 token 数」和「spec-on 的 decode tok/s」。
这两个数今天**量不了**，而且原因在 §6 的 5 与 §3 的 34b 里已经写着：

* `Engine::generate` 的 `speculative` 是 `unimplemented`（`runtime/engine.cpp`）；
* `Engine::forward_batch` **不存在**——`runtime/speculate.h` 的注释自己说「which does not exist yet」；
* `GpuMoeBridge::record_batch_union` **没有调用者**，`run_batch_union` 只被 `tests/test_gpu_moe.cpp` 调用；
* `--spec` **没有接进 CLI**；`runtime::Speculator` 唯一的 `SpecModel` 是 `tests/test_speculate.cpp` 里
  回放 token 流的 `ReplayModel`。

`p4_dspark_runtime.md` §3 的实施顺序里，第 3（`forward_batch`）、5（贪心循环 + `generate`）、
6（草稿链进 runtime）三步都没做，自己估的工期是 1–2 天 + 2 天 + 3–5 天。
**所以本节量的不是 DSpark，是 DSpark 的 verify 那一遍在 M = 1 上的等价物。**

### 10.2 `DEEPMOE_ROUTE_RESIDENT_ONLY=verify`：block-5 的形状，一次一个位置

新增的第四档（`runtime/engine.h` 的 `ResidentOnly::Verify`，`--resident-only verify`，
两个 CLI 都收）：**每五个 decode step 里有一个精确路由**（block 的第一个位置——
它的 miss 照常 P0 取、照常 warm cache），**另外四个走 resident-only**，
和 `all` 完全一样：跳过不驻留的 expert、按 `resident_route()` 重新归一化、
盖 LRU 需求戳、把 miss 交给 Y3 那条有界后台队列。相位是 `token_ % 5`，
哪一个余数是精确的那个无所谓，因为 block 边界本来就是任意的。
落地是引擎里的两行：

```cpp
ResidentOnly ro = resident_only_;
if (ro == ResidentOnly::Verify)
    ro = (token_ % kVerifyBlock) == 0 ? verify_first_ : verify_draft_;
```

两个环境变量把这两半拆开，便于扫：`DEEPMOE_VERIFY_FIRST`（`exact` 默认 / `stall1` / `all`）、
`DEEPMOE_VERIFY_DRAFT`（`all` 默认 / `stall1` / `exact`）。

**这个 M = 1 等价物在哪里偏乐观**：真正的并集 verify 批里，五个位置的 gate 是**同一次 forward**
算出来的，所以 2–5 位的驻留判断用的是**取盘之前**的 cache 状态；
而 M = 1 逐位跑时，第 1 位的 P0 取盘**先落地**，2–5 位看到的是一个更热的 cache。
也就是说真机上的 verify 批只会比下面这张表**更差**，不会更好。
计数器证实了相位：64 步里 51 步走 resident-only（51 × 40 = 2,040 层-步），
四轮对话里 16,040 / 20,080 = 79.9%。

### 10.3 质量：64 步 teacher-forced，5,100 槽，`--warm-cache`（`tools/l3_ppl.py`）

导出集在本机上重生成过一次（`traces/l3_64` 不在仓库里），
reference NLL **0.597555**，与 §9 那次**逐位相同**；`off` 也复现到小数点后六位
（NLL 0.630051 / PPL 1.8777 / top-1 61/64），所以下面的比值与 §9.3 同尺。

| mode | NLL | PPL | ×off | top-1 | tok/s | cache hit | served | mass lost | 只剩 shared 的层-步 | 判据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `off` | 0.630051 | 1.8777 | 1.000× | 61/64 | 3.22 | 0.815 | — | 0 | 0 | baseline |
| **`verify`（draft = `all`）** | 0.948936 | 2.5830 | **1.376×** | 49/64 | 5.85 | 0.946 | 9,449 (0.7720) | **0.2099** | **17** | **NO-GO** |
| `verify`（draft = `stall1`） | 0.662515 | 1.9397 | **1.033×** | 54/64 | 3.82 | 0.958 | 11,177 (0.9132) | 0.0649 | 0 | 在 ≤1.05 带内 |
| `stall1`（§9.3 抄本） | 0.732431 | 2.0801 | 1.108× | 51/64 | 3.87 | — | 0.9001 | 0.076 | 0 | — |
| `all`（§9.3 抄本） | 1.456643 | 4.2915 | 2.286× | 42/64 | 8.75 | — | 0.6859 | 0.3054 | 52 | — |

**这张表的中心一行是第二行。** 任务要的那个设计——verify 那一遍**一个字节都不为 draft 位置去盘上取**——
量出来是 **×1.376**，比 `stall1` 差、比 `all` 好，而且**在 ≤1.3 的带外**。
mass lost 0.2099 与 `all` 的 0.3054 同一个量级，不是 `stall1` 的 0.076。

**为什么**，一句话：`stall1` 的 0.076 是**每一步都掏一次 P0** 买来的。
`verify` 把 P0 的机会从 5/5 降到 1/5，cache 的刷新率就掉到 1/5，
而后台那条有界队列补不上（§3 那条硬预算没被推翻）：
64 步里入队 —— 四轮对话里 enqueued 9,026 / 过期丢弃 5,422，仍然是**需求超出盘的预算**。
第三行是同一件事的反证：只要把四个 draft 位置也给一次 P0（`draft = stall1`），
质量立刻回到 **×1.033**、mass lost 0.0649、**0 个只剩 shared expert 的层**——
但那样 verify 那一遍**就又在等盘了**，整个方案的前提没了（收益见 10.4）。

### 10.4 速度：4 轮对话，每轮 64 token，5,100 槽，backfill 开（`tools/hitrate_bench.py --script y_turns.json`）

| 全程（20,080 层-步） | `off` | **`verify`（draft = `all`）** | `verify`（draft = `stall1`） |
|---|---|---|---|
| decode tok/s（四轮） | 4.77 4.76 4.90 4.86 | **7.48 8.10 7.49 7.22** | 4.71 5.13 5.44 5.11 |
| 均值 / 相对 `off` | 4.82 | **7.57（×1.57）** | 5.10（×1.06） |
| cache hit | 0.876 | 0.966 | 0.972 |
| resident-only 的层-步 | 0 | 16,040 (79.9%) | 16,040 (79.9%) |
| experts requested / served | 120,480 / 全部 | 96,240 / 81,621 (**0.8481**) | 96,240 / 91,242 (0.9481) |
| gate mass lost | 0 | **0.1377** | 0.0385 |
| 只剩 shared expert 的层-步 | 0 | **123** | **0** |
| 后台 enqueued / 过期丢弃 | — | 9,026 / 5,422 | 3,409 / 1,488 |
| 盘 搬的总字节 | 317.5 GB | **258.5 GB** | 293.0 GB |
| 其中 P0 / P3 | **262.5 / 33.0 GiB** | **62.5 / 178.2 GiB** | 191.2 / 81.6 GiB |
| 盘 有效带宽 / 平均延迟 | 3.98 GB/s / 24.8 ms | 3.87 GB/s / 51.0 ms | 3.75 GB/s / 38.7 ms |
| cache eviction | 11,247 | 8,116 | 9,950 |
| `stall1` P0 次数 / 等待 | — | 0 | 7,687 / **42.9 s** |

`verify` 这一档**确实把盘从 decode 的关键路径上搬走了**：P0 从 262.5 GiB 掉到 62.5 GiB（−76%），
总字节少 19%，tok/s ×1.57，四轮输出仍然成句、Markdown 结构正确
（转写在 `bench/results/resident/z_verify/transcript.md`）。
**但这 ×1.57 买不回 ×1.376 的质量**，而唯一能把质量买回来的那一档（draft = `stall1`）
只剩 **×1.06**，代价是 7,687 次 P0、42.9 s 的等待——和 §8.3 说 `stall1` 的那笔账**一模一样**，
只是次数少了一半。

### 10.5 判决

* **把 `verify` 当作「投机开着时的默认路由」：NO-GO。** 质量 **×1.376 > 1.30**，
  连 §9 给 verify-only 留的那条宽带都出去了。
* **§9.4 (iii) 的前提被推翻。** 「`stall1` 的 cache 状态就是 verify-only 的 cache 状态」
  这个读法是错的：`stall1` 的质量是每步一次 P0 买的，verify-only 不买。
  要改写 §9.4 (iii) 的话：verify-only 的 cache 状态落在 `stall1`（0.048）和 `all`（0.259）之间，
  **实测 0.1377**，对应 PPL **×1.376**。
* **任务问的那个变体（block 首位用 `stall1` 而不是 exact）：不用跑也知道更差**，
  因为它在首位取的字节是 exact 的真子集（每层最多一个，而不是最多六个），
  cache 刷新只会更少。开关已经在（`DEEPMOE_VERIFY_FIRST=stall1`），但没有理由花一次 run。
* **真正的中间档是反过来的那个**（首位 exact + 四个 draft 位 `stall1`，
  `DEEPMOE_VERIFY_DRAFT=stall1`）：质量 **×1.033 落在 ≤1.05 的 GO 带内**，
  0 个只剩 shared expert 的层——但它的 verify 那一遍**仍然等盘**（每层一个 expert），
  速度只有 **×1.06**。**作为 DSpark 的 verify 路由，它把方案的全部意义都抵消了。**
* **对 DSpark 的净结论**：§3 的 34b 说「并集相对热缓存省下的 miss 是 0」，
  本节说「把 verify 的取盘全砍掉能省 76% 的 P0，但要付 ×1.376 的质量」。
  两条合起来，**verify-only 不是 34b 那道门的钥匙**。在投机解码本身落地之前
  （`forward_batch` + 贪心循环，见 10.1），不值得再在这条路上写 planner 或路由代码。

### 10.6 重跑

```
# 导出（~29 min，纯 CPU；traces/ 在 .gitignore 里）
.venv/Scripts/python.exe tools/oracle_l3_ppl.py greedy --out traces/l3_64 --steps 64

# 质量（每档 ~60 s GPU，一次一个引擎）
.venv/Scripts/python.exe tools/l3_ppl.py --state traces/l3_64 --steps 64 --modes off,verify
DEEPMOE_VERIFY_DRAFT=stall1 .venv/Scripts/python.exe tools/l3_ppl.py --state traces/l3_64 \
    --steps 64 --modes verify

# 速度（每档 ~4 min GPU）
.venv/Scripts/python.exe tools/hitrate_bench.py --script bench/results/hitrate/y_turns.json \
    --cache-slots 5100 --env DEEPMOE_BACKFILL=1 \
    --env DEEPMOE_ROUTE_RESIDENT_ONLY=off|verify --out bench/results/resident/z_<mode>
```
