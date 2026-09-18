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
