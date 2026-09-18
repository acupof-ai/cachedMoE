# Track F6 — 空闲窗口预取，与 MoE / attention 的 roofline（2026-09-18）

一句话：**两个都不到判据，但两个的"为什么"都被换掉了。**

* **预取**：Track X 说盈亏平衡精度是 **1.00**。那是**饱和盘假设**加上一个
  **均匀随机的错误模型**一起造出来的。在 Q2 之后的真实时间常数上重算
  （4.6 GB/s、5,100 槽、每层 1.3 ms 的 first-of-layer 斜坡、按 chunk 抢占），
  盈亏平衡精度是 **≈ 0.60**，不是 1.00。但**可实现的 lead 拿不到那个精度**：
  一个诚实的 planner 在第 L 层开头手里只有第 L−1 层的输出，对应 trace 的
  `pred_d2`，precision **0.588**——恰好压在平衡点上，实测 **−1.9%**。
  把提前量再买一层（"provisional gate"，用第 L 层 pre-MoE 残差预测 L+1）
  的**天花板**是 **+3.5%**，砍半 **+1.8%**。**< +5%，NO-GO。**
* **kernel**：MoE 每 token 4.512 GB / `moe_gpu` 42.1 ms = **107 GB/s = UMA 的 50%**，
  看上去有 16 ms 可救。但 `kernel_p2_moe.md` §3.5 量到 **M=1 的 MoE kernel 本身是
  222.6 GB/s = 上限的 102%**——**kernel 没有余量，那 16–22 ms 全在 dispatch/间隙里**。
  attention 侧 8.523 GB / 43.5 ms = **196 GB/s = UMA 的 91%**，**已经在 80% 之上**，
  连 4 ms 都拿不满。**按"kernel 跑到 80% UMA"的口径：attention 是负的，MoE 是 +8.9%
  raw / +4.5% 砍半——但它不是 kernel 的活。**

全部离线、纯 CPU，**没有启动任何 GPU / engine 进程**。总机时 ≈ 4 min。

出处：`tools/idle_prefetch_sim.py`（新增）、`bench/results/f6/idle_prefetch.json`、
`traces/mixed`（27,399 token × 40 层）、既有的
`bench/results/q2/d1_default`（`events.jsonl` 的 decode-only `per_token_ms`）、
`D:\models\DeepSeek-V4.1-Flash\deepmoe_manifest.json`（只读，逐张量字节数）。

---

## 1. 时间模型：和 Track X 差在哪

`docs/p4_cache_policy.md` §1 的模型是在 **Q1/Q2 之前**写的，它假设：
盘 4.5 GB/s（其实那是 **C:** 盘的读数，见 §3 的 46）、一层的 miss 没有起步代价、
每 token 墙钟 = 80 + stall。Q2 落地之后这三条都要换。

| | Track X（`p4_cache_policy.md` §1） | **本文** | 出处 |
|---|---|---|---|
| 盘 | 4.5 GB/s | **4.60 GB/s**（D:） | `p4_p0_queue.md` §9.1 |
| 一个 expert | 4.178 ms | **4.087 ms**（18,800,640 B） | 同上 |
| 抢占量子 | 4 MiB / 4.5 = 0.932 ms | **0.911 ms** | 同上 |
| 一层 burst 的起步 | **没有这一项** | **1.3 ms**（first-of-layer P0） | `p4_p0_queue.md` §12（提交线程池之后 `first-of-burst` 1.27 ms） |
| 每 token 墙钟 | 80 + stall | **80 + 17 + stall** | `bench/results/q2/d1_default` `per_token_ms`：moe_host 1.4 + engram 4.4 + tail 6.0 + other 5.3 = **17.0** |
| 容量 | 4,500 / 5,711 / 8,000 | **5,100**（本机实际） | `STATUS.md` §2.4 |

那个 **1.3 ms 的斜坡**是本文和 Track X 最重要的差别，而且它是**对预取有利的**：
它是**延迟**不是**带宽**，Track X 的模型里根本没有这一项。一个把某层最后一个 miss
提前取到的预取，除了省下 4.087 ms 的传输，**还顺带省掉那一层的 1.3 ms 起步**。
基线里这一项是 15.6 层 × 1.3 = **20.3 ms/token**，占 stall 的 19%。

每层的形状（`tools/idle_prefetch_sim.py`）：

```
第 L 层：[ 计算 c = 2.0 ms，盘在这个窗口里是空的 ] -> [ 本层 demand miss，GPU 原地停 ]
盘 = 一个 FIFO 服务器，demand 优先，按 4 MiB chunk 抢占（队头代价 0.911 ms）
预取在飞时被 demand 命中 -> 升级，只等剩下的字节（partial hide），不重付 1.3 ms
预取落地 -> 占一个槽，LRU 淘汰（= cache_sim 的 --probe-position head）
错的预取只在盘本来空着的时候花盘时间，外加队头那一个 chunk，外加它偷走的槽
```

**基线复现**（测试集 7,201 token，训练集末 2,000 token 预热，C=5,100）：

| | 模型 | 引擎实测（`q2/d1_default`，decode-only） |
|---|---:|---:|
| miss / token | **21.3** | **21.3**（`p4_p0_queue.md` §2） |
| hit | 0.9113 | 0.9161 |
| 有 miss 的层 / token | 15.6 | 13.4 |
| `nvme_stall` ms/token | **107.2** | **93.1–100.9** |
| tok/s | **4.897** | **4.998–5.098** |

stall 高 6–14%、tok/s 低 2–4%，来源是 **trace 不同**：引擎跑的是 `y_turns.json`
四轮同话题聊天（重用高、有 miss 的层只有 13.4），trace 是 40 个互不相干的 prompt。
**同一个模型内的相对比较是可比的**，这是本文所有 Δ 的口径。

---

## 2. 提前量到底有多少：一个必须说清楚的 off-by-one

`tools/route_trace.py` 的 `snapshots[L] = t.snapshot(H)` 是在 `run_layer(L)`
**之后**取的，所以 `pred_dK` 对目标层 `tgt` 用的是 `snapshot[tgt-K]` =
**第 tgt−K 层的输出**。

一个真的 planner 在**第 L 层开头**手里有什么？**第 L−1 层的输出**——第 L 层自己的
输出要等它的 MoE 算完，而 MoE 要等它的 miss 落地，那已经是本层的 stall 之后了。
所以"在第 L 层开头为第 L+d 层发预取"能用的列是 **`pred_d(d+1)`**，不是 `pred_dd`。

**这一位错了，lookahead 就白拿一层的提前量，也白拿一档精度。**

| trace 列 | precision@K=6（测试集，层 ≥ K） | 谁能用它 |
|---|---:|---|
| `pred_d1` | **0.674** | 没有人。它要第 L 层的**输出**，那时第 L+1 层马上就要算了 |
| `pred_d2` | **0.588** | 诚实的 lead = 1 |
| `pred_d3` | **0.532** | 诚实的 lead = 2 |
| `pred_d4` | **0.491** | 诚实的 lead = 3 |

`pred_d1` 的 0.674 **不是不能用，而是要换一条路才能用**——那就是 §4 的
provisional gate：如果能用第 L 层的 **pre-MoE 残差**（h + attn 输出，在本层 stall
**之前**就有）跑一遍 gate 去预测 L+1，提前量就回到了"第 L 层开头"。
`pred_d1` 是那条路的**天花板**（它用的是真的 post-MoE 输出，pre-MoE 残差只会更差）。

---

## 3. 表一：idle-window 预取，当前 regime（C = 5,100，测试集 7,201 token）

判据带 **±3%**（`p4_p0_queue.md` §3），"二分之一法则"后要 **≥ +5%** 才值得动 GPU。

### 3.1 诚实的 lookahead（lead = d 用 `pred_d(d+1)`）

| lead | K | precision | stall ms | tok/s | Δ | 预取 MB/token | 其中浪费 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| — | — | — | **107.2** | **4.897** | — | 0 | 0 |
| 1 | 2 | 0.588 | 109.8 | 4.836 | **−1.2%** | 326 | 163 |
| 1 | 3 | 0.588 | 111.0 | 4.808 | −1.8% | 328 | 171 |
| 1 | 6 | 0.588 | 111.1 | 4.805 | −1.9% | 328 | 172 |
| 2 | 2 | 0.532 | 114.5 | 4.727 | −3.5% | 339 | 208 |
| 2 | 6 | 0.532 | 115.4 | 4.709 | −3.8% | 340 | 215 |
| 3 | 2 | 0.491 | 117.4 | 4.665 | −4.7% | 346 | 226 |
| 3 | 6 | 0.491 | 118.3 | 4.645 | −5.1% | 346 | 231 |

**全部为负，而且 lead 越长越负。** 机制：提前量买到的窗口是每层 2 ms，
而一个 expert 要 4.087 ms，**lead=1 连一个 expert 都藏不完**；lead 拉长以后
窗口是多了，但精度掉得更快（0.588 → 0.491），错的那些既吃盘又吃槽。

### 3.2 合成预测器：盈亏平衡精度（K=6/层，错的候选**均匀随机**——Track X 的口径）

| precision | lead 1 | lead 2 | lead 3 |
|---:|---:|---:|---:|
| 不预取 | **4.897** | **4.897** | **4.897** |
| 0.50 | 4.172（−14.8%） | 4.165（−14.9%） | 4.168（−14.9%） |
| 0.65 | 4.240（−13.4%） | 4.241（−13.4%） | 4.250（−13.2%） |
| 0.77 | 4.337（−11.4%） | 4.351（−11.1%） | 4.366（−10.8%） |
| **0.90** | **4.998（+2.1%）** | **5.062（+3.4%）** | **5.113（+4.4%）** |
| 1.00 | 7.210（+47.2%） | 7.384（+50.8%） | 7.453（+52.2%） |

**这一张就订正了 Track X §7 的"盈亏平衡 = 1.00"**：在当前 regime 下平衡点掉到
**0.85 左右**（0.77 还是 −11%，0.90 已经 +2…+4%）。把 1.00 变成 0.85 的是那 1.3 ms
的斜坡和 5,100 槽，不是别的。

### 3.3 但这个合成模型对预测器不公平——错误的**形状**比错误的**多少**更重要

均匀随机的错候选**按构造就是 cache-cold 的**：384 选 1，几乎从不驻留，
所以每一个错都要整整 18.8 MB 加一个热槽。
**真预测器的错不是这样的**——它错在那些"差一点就进 top-6"的 expert，
而那些**正是 LRU 手里已经有的**，于是绝大多数错**一个字节都不花**。

控制实验：同样的 precision，错的候选改从**本层 top-16 的第 6–15 名**里抽
（`NearMissPredictor`），lead = 1：

| precision | 均匀随机错（§3.2） | **近似错（真实形状）** | 实际入队的预取/token |
|---:|---:|---:|---:|
| 0.50 | 4.172（−14.8%） | **4.792（−2.1%）** | 35.8 |
| 0.65 | 4.240（−13.4%） | **4.996（+2.0%）** | 30.8 |
| 0.77 | 4.337（−11.4%） | **5.385（+10.0%）** | 27.2 |
| 0.90 | 4.998（+2.1%） | **6.179（+26.2%）** | 23.7 |
| 1.00 | 7.210（+47.2%） | 7.210（+47.2%） | 20.6 |

**盈亏平衡精度 ≈ 0.60，不是 0.85，更不是 1.00。**
校验：真的 `pred_d1`（precision 0.674）在同一个 lead 上给 **+3.5%**，
而近似错模型在 0.674 附近插值也是 **+3 %** 左右——**模型和真预测器对上了**，
这是这张表可信的理由。

> 这一条同时是对 `p4_cache_policy.md` §7 那个 "+78% 悬崖" 的解释：
> 悬崖不是预测器的性质，**是那个降级方法的性质**。

---

## 4. 表二：provisional gate（两层提前量）

设想：用第 L 层的 **pre-MoE 残差**（`h + attn_out`，在本层 MoE / stall **之前**
就在手上）过一遍 `ffn_norm` + gate，得到第 L+1 层的**临时路由**，
在第 L 层开头就发预取。**提前量 = 第 L 层的 2 ms + 第 L 层的 stall + 第 L+1 层的 2 ms。**

**它的 precision 这条 trace 量不出来**——`route_trace.py` 只存了 post-MoE 的
`snapshot`，没有存 pre-MoE 残差。所以它在下表里是**参数 p**。

| | precision | stall ms | tok/s | Δ | 砍半后 |
|---|---:|---:|---:|---:|---:|
| 不预取 | — | 107.2 | **4.897** | — | — |
| **天花板：用 `pred_d1`（真 post-MoE gate）** | **0.674** | 100.4 | **5.067** | **+3.5%** | **+1.8%** |
| 参数 p = 0.50 | 0.50 | 111.9 | 4.788 | −2.2% | — |
| 参数 p = 0.65 | 0.65 | 103.3 | 4.993 | +2.0% | +1.0% |
| **参数 p = 0.77** | 0.77 | 88.8 | **5.383** | **+9.9%** | **+5.0%** |
| 参数 p = 0.90 | 0.90 | 64.7 | 6.184 | +26.3% | +13.2% |
| 参数 p = 1.00 | 1.00 | 41.7 | 7.210 | +47.2% | +23.6% |

天花板那一行是**用真的 post-MoE gate 冒充 provisional gate**：pre-MoE 残差少了本层
MoE 的贡献，**只会更差**，所以 **+3.5% 是这条路的上界，不是估计值**。

预取字节（天花板那一行）：入队 **27.1 个/token**（不是 6 × 39 = 234——绝大多数预测
的 expert 已经驻留），盘真正搬了 **313 MB/token**，其中 **176 MB 是浪费**；
完整藏住 6.1 个 miss、部分藏住 10.5 个，省掉 6.2 次 1.3 ms 的层起步。

---

## 5. 表一/表二的判决

| 问题 | 答案 |
|---|---|
| 当前 regime 下有没有可实现的 (d, p) 能拿 ≥ +5%（砍半后）？ | **没有。** 最好的可实现格是 provisional gate 的**天花板** +3.5%，砍半 **+1.8%**；诚实的 lookahead（lead=1，`pred_d2`，p=0.588）是 **−1.2 … −1.9%** |
| Track X 的"盈亏平衡 = 1.00"还成立吗？ | **不成立。** 在真实的错误形状下平衡点是 **≈ 0.60**；在均匀随机错误模型下是 **≈ 0.85**。1.00 是**饱和盘 + 均匀随机错**两个假设叠出来的 |
| 那为什么还是 NO-GO？ | **提前量和精度是互斥的，而可用的提前量只有 2 ms/层。** 诚实的 lead=1 拿到的是 `pred_d2` 的 0.588，刚好压在 0.60 的平衡点上；要越过 +5% 的线需要**在两层提前量上做到 p ≥ 0.77**，而那条路上最乐观的读数（真 post-MoE gate）也只有 **0.674** |
| 门槛是什么（可以写成一条验收线） | **两层提前量、K ≤ 6、per-expert precision ≥ 0.77。** 低于 0.60 一定是负的 |

**要不要上 GPU 实验：不要。**
如果哪天要做，它是这样一条（**先做离线那半**，它不占 GPU）：
给 `route_trace.py` 加一列 `pred_prov`——在 `run_layer` 里把 **pre-MoE 残差**
（`H` 在 attention 之后、MoE 之前的那一份）也 snapshot 一遍，过 `ffn_norm` + gate，
记 top-16。**一次 trace 重跑，纯 CPU，没有任何引擎改动**，出来的就是上表里那个
**参数 p** 的真值。只有当它 **≥ 0.77** 时，才轮到 runtime 那一半
（`Planner` 在 `plan_layer` 之前多发一批 P1 优先级的 fetch）。
**预测收益（已砍半）：p = 0.77 时 +5.0%，p = 0.674 的天花板时 +1.8%。**
按 STATUS §3 的口径，**+1.8% 低于 ±3% 的抖动带，写出来也测不出来**。

---

## 6. 表三：MoE kernel 的 roofline

每 token 的 routed expert 字节是确定的：**40 层 × 6 expert × 18,800,640 B = 4.512 GB**
（shared expert 不在这里，它是 pinned 的，算在 `attn` / `hot_gemv` 那一格，见 §7）。

UMA 上限取 **216 GB/s**（`p2_attention.md` §10 的 raw-read 216–218，
`kernel_p2_moe.md` 的 216.6–217.9）。

| 口径 | ms/token | 有效 GB/s | % of 216 | 出处 |
|---|---:|---:|---:|---|
| **引擎 decode `moe_gpu`** | **42.14** | **107.1** | **50%** | `q2/d1_default` `per_token_ms` |
| runtime `MoeRunner` 单层 × 40（M=1，全驻留，加列掩码后 1.044 ms/层） | 41.76 | 108.0 | 50% | `p4_mgt1.md` §4 `mgt1.moe_m_curve` |
| **MoE kernel 本身（M=1 最优变体 `L32 R1 xglob`，7 个 FP4 槽，0.5912 ms/dispatch 对）** | **23.65** | **190.8** | **88%** | `kernel_p2_moe.md` §3.5 |
| 同上按 6 槽线性折算（去掉 shared 那一槽） | 20.27 | **222.6** | **103%** | 同上（222.6 GB/s 就是它自己报的速率） |
| 假想：kernel 跑到 80% UMA（172.8 GB/s） | 26.09 | 172.8 | 80% | — |

**读这张表的唯一正确方式**：

1. **kernel 没有余量。** `kernel_p2_moe.md` §3.5 白纸黑字：
   *"M=1 上一个字节也没省——那里 kernel 已经在内存系统的上限上"*，**222.6 GB/s = 102%**
   （超过 100% 是 MALL 的贡献）。所以"把 MoE kernel 提到 80% UMA"这个提法在 M=1 上
   **是反的**：它已经在 103%。
2. **那 50% 是 dispatch 和间隙，不是 kernel。**
   引擎 42.14 ms ÷ 40 层 = **1.053 ms/层**，而 kernel 对只要 **0.507–0.591 ms**。
   **每层丢了 0.46–0.55 ms，每 token 18.5–21.9 ms。**
   `MoeRunner` 的 1.044 ms/层和引擎的 1.053 几乎相等，**说明这段不在引擎的调度里，
   在 `MoeRunner` 这一层**（staging、`set_live_columns`、每层的 dispatch 对之间的
   barrier / 空隙）。
3. host 侧的提交**不是**这 18.5 ms：`per_token_ms` 里 `submit` 只有 **3.73 ms**
   （56.5 次提交）、`record` 0.75、`bind` 0.99，`fence_wait` 90.0 ms 基本等于
   `nvme_stall` 93.1。**所以缺口在 GPU 时间轴上的 dispatch 间隙，不在 CPU 上。**

**能省多少：**

| 目标 | 省下 ms/token | token ms | tok/s | Δ raw | **Δ 砍半** |
|---|---:|---:|---:|---:|---:|
| kernel 跑到 80% UMA（172.8 GB/s） | 16.0 | 179.7 | 5.565 | +8.9% | **+4.5%** |
| 把间隙压到 kernel 的 7 槽速率（190.8 GB/s） | 18.5 | 177.2 | 5.642 | +10.4% | **+5.2%** |
| 把间隙压到 kernel 的 6 槽速率（222.6 GB/s） | 21.9 | 173.8 | 5.752 | +12.6% | **+6.3%** |

（基准 token = 195.7 ms = 5.109 tok/s，`q2/d1_default` 第 4 轮 decode-only。）

---

## 7. 表四：attention / dense 权重的 roofline

**每 token 真正读的 dense 字节**，逐张量从 `deepmoe_manifest.json` 加起来
（`store::pinned_layer_tensors` 的那张名单 + `head.weight`，每个都含 `.scale`）：

| 组 | MB/token | 说明 |
|---|---:|---|
| `wq_b` | 1,679.4 | 40 层 × 41.98 |
| `wo_b` | 1,679.4 | 40 层 × 41.98 |
| `ffn.shared_experts.w1/w2/w3` | 1,417.0 | 40 层 × 35.42，**pinned，所以它在这一格不在 MoE 那一格** |
| `wo_a` | 1,343.5 | 40 层 × 33.59 |
| `head.weight` | 1,323.8 | 129,280 × 5,120 × 2 B |
| `engram.wkv / q / k` | 315.0 | 只在第 1、14 层 |
| `wq_a` | 262.4 | 40 层 × 6.56 |
| `ffn.gate.weight/bias`（router） | 157.4 | |
| mHC 六张表 | 157.3 | |
| `wkv` | 105.0 | |
| indexer | 45.1 | 8 层 |
| compressor | 36.7 | 4 层 |
| norms / `attn_sink` | 1.0 | |
| **合计** | **8,522.8** | = 引擎 `hot_bytes` **8,522,849,728**，**逐字节相等** |

| 口径 | ms/token | 有效 GB/s | % of 216 |
|---|---:|---:|---:|
| **引擎 decode `attn`（= `hot_gemv` 相位）** | **43.51** | **195.9** | **91%** |
| kernel 地板：`attn_bench` P3 的 "40 层 + head" 33.5 ms（6,790.9 MB，203 GB/s = 94%）+ shared expert 1,417 MB @ 216.9 GB/s（`kernel_p2_moe.md` §0 实测）6.53 ms | **40.0** | 213 | 99% |
| 假想：跑到 80% UMA（172.8 GB/s） | 49.3 | 172.8 | 80% |
| 假想：跑到 100% UMA（216 GB/s） | 39.5 | 216.0 | 100% |

**这一格已经没有东西了：**

* 91% 已经**在 80% 之上**，"提到 80% UMA"会让它**慢 5.8 ms**；
* 提到 **100%** 也只省 **4.0 ms/token = +2.1% raw / +1.1% 砍半**；
* 而 kernel 地板是 **40.0 ms**，引擎是 43.51——**整段差距只有 3.5 ms**，
  即所有 dispatch 间隙、mHC 的三个小 kernel、sparse attention 的延迟项加起来。
  P3 那一轮（`p2_attention.md` §13）已经把 `wq_b` 做到 216 GB/s = **99%**、
  `wo_b` 205、`wo_a` 207。

---

## 8. 判决

| | 达成 | 上限 | "kernel 跑到 80% UMA" 能省 | Δ raw | **Δ 砍半** | 判决 |
|---|---:|---:|---:|---:|---:|---|
| **MoE（routed，4.512 GB/token）** | 42.1 ms / **107 GB/s（50%）** | kernel 本身 **222.6 GB/s（103%）** | **16.0 ms** | +8.9% | **+4.5%** | **有 ≥5%，但不在 kernel 里**：kernel 已经打满，缺口 18.5–21.9 ms 全在 **dispatch / 间隙**。压到 kernel 自己的速率是 **+5.2% … +6.3%（砍半后）** |
| **attention / dense（8.523 GB/token）** | 43.5 ms / **196 GB/s（91%）** | kernel 地板 40.0 ms（213 GB/s） | **−5.8 ms（更慢）** | −2.9% | −1.4% | **没有**。打满 100% UMA 也只有 +1.1%（砍半）。**这一格关掉** |

**在哪个 kernel：一个都不是。** 唯一 ≥5% 的那笔钱在 **MoE 的 dispatch 路径**上，
不在 shader 里。它正好就是 `STATUS.md` §7 第 4 项（per-dispatch GPU trace）和
第 5 项（persistent-dispatch decode）指着的东西，而且本文第一次给它标了价：
**18.5–21.9 ms/token = 每 token 的 9–11%，砍半后 +5.2% … +6.3%**。

**要不要上 GPU：这一条值得，但它不是 F6 的范围。**
探针就是 §7 第 4 项那条命令（per-dispatch trace，把每层 MoE 的两个 dispatch 的
busy / gap 拆开）。**预测（已砍半）：+5.2%**。证伪条件很干净——
如果 trace 显示两个 dispatch 之间的 gap 加起来 **< 0.2 ms/层**，
那 42.1 vs 23.7 的差就不在间隙里，而在 `MoeRunner` 送进 kernel 的**形状**
（槽数、staging、`live_columns` 的写入）上，那时要量的是 `stage_batch` 而不是 gap。

---

## 9. 重跑

```powershell
# 纯 CPU，不占 GPU，约 65 s（含 trace 载入）
.venv\Scripts\python.exe tools\idle_prefetch_sim.py `
    --trace traces\mixed --out bench\results\f6 --capacity 5100
# 表三 / 表四的算术全部来自已有数据，没有新测量：
#   bench\results\q2\d1_default\events.jsonl   per_token_ms（decode-only）
#   bench\results\q2\*\profile.jsonl           hot_bytes / expert_requests
#   D:\models\DeepSeek-V4.1-Flash\deepmoe_manifest.json  （只读）逐张量字节
```

## 10. 留下什么

| | 结论 |
|---|---|
| `tools/idle_prefetch_sim.py` | **保留**。它是当前 regime 的预取模型，Track X 的那个已经过期（盘、斜坡、容量三项都换了） |
| 任何预测式预取 | **仍然不做**（`STATUS.md` §3 的 35），但**理由换了**：不是"要 precision 1.00"，是"**两层提前量上要 p ≥ 0.77，而最乐观的读数是 0.674**" |
| provisional gate（pre-MoE 残差跑 gate） | **不做，除非先把它的 precision 量出来**。量它的代价是一次纯 CPU 的 trace 重跑（§5），不是 GPU 实验 |
| attention / dense kernel 的带宽 | **关掉这条线**。91% of UMA，打满也只有 +1.1%（砍半） |
| MoE 的 dispatch 间隙 | **接到 `STATUS.md` §7 第 4 项**，标价 +5.2%（砍半） |
