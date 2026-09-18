# P5 — 测量协议、实验台账、下一阶段的有序实验表

[STATUS.md](STATUS.md) 说的是"今天在哪里"。这一份说的是**怎么量**和**接下来按什么顺序试**。

两条纪律，取自 `fleet-mi300x`，它们是这个项目 34 条退掉的实验换来的：

> **一、一个绿色的数字什么也不证明，除非它来自一个安静机上的成对交替运行。**
> `p2_decode.md` §4.4 那次"不可复现"不是 kernel——是三条 track 往同一个 `build/shaders`
> 并发重建 `.spv`，引用的几次 run **两套 kernel 都没执行**。
> `kernel_p2_moe.md` §6.2 那个"+0.193 ms/层、§7.9 错了 340×"的结论，
> 在轮转配置下重测变成 +0.021…+0.047 ms/层——**差 8 倍，全是热漂移**。
>
> **二、基于字节数的估计先砍一半。**
> fleet 那边三次都偏高一倍。这里同构的三次：MoE live-column mask microbench 1.7×、
> 端到端 **0**（§3 的 20）；reheat 预测有收益、实测 **0**（21）；
> auto-tune 预测有收益、实测 **−0.0003 hit**（22）。

---

## 1. 测量协议

### 1.1 安静机

**一台机器上同时只能有一个引擎进程。** 不是建议：

- 几条 track 同时压 GPU 时 `restore_context` 撞上 `cmd_wait` 的 120 s fence 超时，
  被误诊成逻辑 bug 整整一轮（`p4_test_report.md` §3 第 3 条）。
- 并发重建 shader 让两轮的数字指向从未执行过的 kernel（`p2_decode.md` §4.4）。
- 热漂移：同一段内 8%、run 之间 7%（`kernel_p2_moe.md` §11）。

清单，跑任何一个数字之前：

```powershell
# 1. 只有一个 deepmoe / bench 进程
Get-Process deepmoe,*bench,*probe -ErrorAction SilentlyContinue
# 2. 每条 track 自己的 build 目录（shaders 不共享）
# 3. 先跑 2 次丢掉（热机），再开始记
```

### 1.2 成对交替运行

一个改动的 A/B **永远是 `ABABAB`，不是 `AAABBB`**。三对是最小值，每对之间不重启。
报告的形式是**两个区间**，不是两个均值：

> `3.647 / 3.629 / 3.641` vs `3.536 / 3.551 / 3.544`，区间不重叠 → 真的
> `3.647 / 3.629 / 3.641` vs `3.652 / 3.618 / 3.660`，区间重叠 → 噪声，记成 0

本机的 within-session 抖动是 **±3%**（`p4_mgt1.md` §4）。**任何小于 3% 的单次差值都是 0。**
要声明 <3% 的收益，就需要更多的交替对，并且把区间写出来。

### 1.3 每一个改动都要带"预测 vs 实测"这一列

台账（§2）的每一行必须有：

| 机制 | 预测（已砍半） | 实测 | 差 |
|---|---|---|---|

写下预测**再**去测。`p4_hitrate.md` 里那两个 bug（heat 恒等于 0、`decay_heat` 把整场压成 0）
是因为一个"应该有收益"的实验实测为 0 才被翻出来的——**预测与实测不符本身就是产出**。

### 1.4 砍半规则

任何"这里有 X 字节，带宽是 B，所以省 X/B"的估计，**先乘 0.5** 再跟成本比。
理由在 §0 的纪律二。反向也成立：如果砍半之后仍然值得做，那它大概率真的值得。

### 1.5 两道闸，每一个被保留的改动都要过

1. **正确性**：`suite.decode`（L3 oracle 四十层八步）**教师强制 8/8 且自由运行 8/8**。
   Track T 就是在这道闸上被回退的（8/8 → 7/8，`p4_test_report.md` §3 第 2 条）。
   动到长上下文的改动再加 `suite.decode_longctx`（4K / 17K）。
2. **逐位确定性**：同一个 prompt、同一个 seed，跑三个**独立进程**，
   logits hash 必须完全相同（今天是 `a919aaf6…`，`p2_decode.md` §4.4）。
   一个只在某些运行里出现的差异，比一个稳定的错误更贵。

改动如果**本该**逐位相同（layout 改动、K-split 的重结合除外），就说出来并验证它。
LDS bank-conflict 那次修复（797 µs，−10%）是**逐位相同**的，这一点是它可信的原因。

### 1.6 CPU 侧的闸

```powershell
.venv\Scripts\python.exe tests\run_all.py --mutate
```

一条命令、一个退出码：所有 CPU 单元套件（glob 找到，新文件不用改列表）、
`tools/trace_timeline.py --self-test`、变异注入这道闸。**买机器时间之前先跑它。**

---

## 2. 实验台账

每一个实验——留下的和退掉的——在 [STATUS.md §3](STATUS.md#3-试过并退掉的编号共-34-条) 里有编号。
新实验从 **35** 开始编号，写进那张表，**包括失败的**。格式：

```
| 35 | 名字 | 机制一句话 | 预测（砍半后） | 实测（成对交替，区间） | 保留/退掉 + 原因 |
```

**退掉的实验必须留下它的编号和数字。** 这份记录是这个项目最有价值的部分：
34 条里至少有 6 条是别人会重新提出来的（VGM、lookahead、树采样、heads_per_wg、
LDS x-tiling、CPU 分担 GEMV），每一条都已经花过一次钱了。

---

## 3. 下一阶段的有序实验表

顺序的依据是 [STATUS.md §4](STATUS.md#4-为什么-decode-是-nvme-bound而不是-kernel-慢)：
**decode 被 NVMe 钉死，每个"算得更快"的改动都要先除以它在 token 里的占比。**
所以 (a) 之前的三件事（miss 插桩、SSD KV 默认开、三个 A/B）在 STATUS.md §7 的 1–3 项，
不在这里——它们不需要新机制，只需要有人去跑。

这里是**需要新机制**的那些。

---

### (a) persistent-dispatch decode — 一层或一个 token 一次 dispatch

**机制**。今天一个 token 是 41 次 submit，每次 submit 的 host 往返实测
**0.13–0.15 ms**（`Engine::measure_submit_overhead`），加上每个 dispatch 之间一个全局
barrier。`fleet-mi300x` 的做法是把 ~800 次 launch 压成**一次**：304 个 workgroup 常驻不退出，
host 预先建好一张任务图，workgroup 之间通过 device 内存上的事件计数器互相递工作。
这里对应的形状是：一层（或一个 token）一次 dispatch，device 侧任务队列 + 自旋等待，
gate 那个必须回 host 的点用一个 device 可见的 fence 代替。

**能省的**。

| 项 | 今天 | 消掉之后 |
|---|---|---|
| submit 往返 | 41 × 0.13–0.15 ms = **5.3–6.2 ms/token** | 1–2 次 |
| barrier | 每层 16–29 个 dispatch，每个一个全局 barrier；单个 dispatch+barrier 实测 **0.56–0.66 µs** | 层内 barrier 变成 device 事件 |
| `Engine` 记的 "other" | **5.0 ms/token**（record 0.4 + submit 1.6 + bind 1.0 + fence ≈2.0） | 大部分 |

**预测（砍半后）**：热步 81.5 → **77–78 ms**（−4%），因为 41 次 submit 里有相当一部分
已经与 GPU 工作重叠了（`sub_ms_` 记的是 1.6 ms，不是 6 ms）。
对话里的 token（215–290 ms）**−2%**。
**这就是为什么它排在 §7 的第 5 项而不是第 1 项**——按砍半规则，它的收益比 hit-rate 那条线小一个量级。

**成本**：高。一个 device 侧任务队列、事件协议、以及 gate 的重新设计
（host 今天必须在 gate 之后读 ids 才能决定取哪些 expert；这正是 command buffer 在那里被切开的原因）。

**探针（先跑这个，不要先写 kernel）**：`build/residency_probe`。三个问题：

1. **常驻数**。8060S 上 256 线程的 workgroup 同时能跑几个?
   一个 workgroup N 等 workgroup M 的 megakernel，如果 M 从来没被调度，就**死锁**。
   Vulkan 没有 cooperative launch，也没有 workgroup 之间的前进保证。
   这是**合法性问题，不是性能问题**。
2. **跨 workgroup 握手延迟**。fleet 在 MI300X 上是 idle 1.36 µs / 满载 5.88 µs。
   如果这里是几十微秒，40 层 × 每层几个事件就把省下来的 submit 吃回去了。
3. **前进保证**。一个自旋的 workgroup 会不会饿死它等的那个？
   探针在**低于**常驻上限的 group 数上报超时 = 共存不蕴含前进 = **自旋等待在这台机器上 NO-GO**，
   方案退回"一张 dispatch 图"而不是"一次 dispatch"。

探针里**每一个等待都有硬上限**（`--spin`），撞上上限就报超时。没有任何一条路径会不终止。
这台机器是 owner 的桌面，TDR 意味着重启。

---

**结果（2026-09-18，Track PD）——关闭。预测 −4%，实测上限 −0.8%，而且机制本身不合法。**

两侧独立地倒下，任何一侧都够：

**一、合法性。`build/residency_probe --max-groups 2048 --spin 200000 --rounds 1000`**
（`bench/results/p4pd/residency_probe.csv`，STATUS §3 的 48）：

| 段 | 实测 | 判据 |
|---|---|---|
| residency census | 32 / 64 / 128 / 256 组全活；512 起 peak 停在 **406–413** | 常驻上限 **~406** 个 256-线程 workgroup |
| ping-pong | **1,000 次往返里第 3 次就 TIMEOUT**（两个 workgroup） | §5.1 的第 2 条**不过**。fleet 的 1.36 µs 在这里量不出来，因为两个 group 根本不保证同时前进 |
| forward progress | 2 / 8 / 32 / 128 `progressed`；512 / 2048 `TIMEOUT`（超上限，预期内） | 上限之内前进，但握手已经判了死刑 |

**共存不蕴含前进 ⇒ 自旋等待 NO-GO ⇒ device 侧任务队列 + 事件计数器的形状不成立。**
§5.1 说这时 (a) 退化成「一张 dispatch 图」——那就是下面这一半。

**二、收益。热步的 per-dispatch trace**（`--steps 8 --warm 8 --slow-prefill --trace`，
7 个稳态 token，741 dispatch / 41 submit 每 token）：

```
span 102.07 ms  =  GPU busy 85.23 ms  +  gap 16.84 ms (16.5%)
```

16.84 ms 看起来远在 8 ms 的门槛之上，**但它的分布不是平的**：

| gap 在哪 | ms/token | 机制 |
|---|---|---|
| 每层 MoE dispatch **之前** | **16.04**（40 × 0.40 ms） | 命令缓冲在 gate 处被切开的那次 **host 往返**（fence → 读 ids → 选 expert → bind → record → submit） |
| 其余 **700 个** dispatch 的层内 barrier | **0.80** | 0.4–2.2 µs 一个，与 §3 的 27 的 0.56–0.66 µs 一致 |

逐层（`--layer`，一个稳态 token）：

| 层 | dispatch | busy | 层内 gap | MoE 前的 gap |
|---|---|---|---|---|
| 21（普通） | **17** | 1,895 µs | **18 µs** | 335 µs |
| 20（kv+index 源层） | **26** | 2,112 µs | **24 µs** | 271 µs |

**所以「把一层的非 MoE 链录进一个 command buffer」不是一个待做的改动——它是今天的实现**
（STATUS §2.1，~128 → **41** submit，只在 gate 处切开；submit 数 41 与 §3(c) 数出来的
17 / 26 dispatch 一起，把「一个普通层 20 / 源层 33」这个计数也更正了）。
再融合任何东西的天花板是 **0.80 ms/token = 0.8%**，**四分之一个 ±3% 抖动带**。

| 机制 | 预测（已砍半） | 实测 | 差 |
|---|---|---|---|
| (a) persistent dispatch，热步 | **−4%**（81.5 → 77–78 ms） | **上限 −0.8%**，且自旋形不合法 | 预测偏高 5×，并且方向错：预测把钱押在 **submit 往返**上，实际 41 次 submit 的往返**已经全部被 GPU 工作盖住了**——省下来的只有 gate 那一次，而那一次的 device 侧替代品被探针关掉了 |

**要重开它，先得有一条不用自旋的 device 侧 gate**（indirect dispatch + device 侧写
`VkDispatchIndirectCommand`，不需要 workgroup 互等）。那是一条新路，不是这一条。
代价：一次 trace + 两个探针，约一小时。

**顺带**：`sharing_probe` 的行是平的（STATUS §3 的 50），共享读免费；
`--trace` 的开销实测为 **0**（带 102.07 ms vs 不带 102.3–106.2 ms），trace 机制在真机上验收通过。

---

### (b) absorbed-K attention — **已经关闭，不要做**

结论先写：**V4.1-Flash 不是 MLA，没有 `W_UK` / `W_UV` 可以吸收。这条路不存在。**

代数（读 `D:\models\DeepSeek-V4.1-Flash\inference\model.py`，只读）：

| 事实 | 出处 |
|---|---|
| `self.wkv = Linear(dim=5120, head_dim=512)` —— 唯一一个 KV 矩阵，是**降维**，不是升维 | `model.py:643` |
| attention kernel 只拿到**一个** `kv` 张量：score 用 `T.gemm(q_shared, kv_shared, transpose_B=True)`，value 用 `T.gemm(acc_s_cast, kv_shared)`——**K 和 V 是同一个张量** | `kernel.py:365`、`kernel.py:380` |
| 全文没有 `wkv_b` / `kv_b_proj`，也没有 `.split([qk_nope_head_dim, v_head_dim])` | — |
| 448/64 只是**一个 512 向量的 RoPE / 非 RoPE 划分**，不是 K/V 划分；`nope_head_dim` 赋值之后再没被读过 | `model.py:632` |
| cache 里存的就是 kernel 消费的那 512 维，64 个 head 共享（对 latent 做 MQA） | `model.py:718` |

也就是说：**吸收已经烤进 checkpoint 了**。`wq_b` 是 `[32768, 1280]`，在 MLA 的语言里
它已经等于 `W_UQ @ W_UK^T` 的乘积形式；`wo_a` 是 `[8192, 4096]`，直接吃 512 维的 per-head 输出。
MLA 的吸收技巧之所以存在，是因为它存一个小的 `c_KV` 却在一个大的空间里做 attention；这里**没有这个落差**。

顺带确认了几条设计里写着的事，以及几个**阻断折叠的点**（将来任何"把两个矩阵合起来"的提议都先看这张表）：

| 阻断点 | 什么 | 出处 |
|---|---|---|
| `q_norm` | RMSNorm(1280)，夹在 `wq_a` 和 `wq_b` 之间——两者不能合并 | `model.py:641` / 应用于 `:770` |
| `kv_norm` | RMSNorm(512, eps=1e-20)，在 `wkv` 之后、RoPE 之前；数据相关的非线性 | `model.py:644` / `:705` |
| fp8 假量化 | 存进 window KV 之前对**整个 512 维（含 RoPE 尾）**做 block-32 / UE8M0 的 quant→dequant | `model.py:707` |
| 逆 RoPE | attention 输出的最后 64 维要**反旋**回去再进 `wo_a`；位置相关，挡住把 `wo_a` 折回 attention | `model.py:781` |
| 每个量化 Linear 自己量化输入 | `wq_a` / `wq_b` / `wkv` / `wo_b` 都是 act_quant(block 32, UE8M0) 之后再 GEMM；`wo_a` 是 bf16 不量化 | `model.py:186-204`、`:648` |

**唯一代数上干净的折叠是 `wo_a`→`wo_b`**（两者之间什么都没有，`model.py:787-788`），
但它会把 `[8192,4096] + [5120,8192]` 展开成 per-group `[5120,4096]×8` = 完整的 `[5120, 32768]`，
**参数量 ×4**——而 LoRA 分解存在的全部理由就是避免这个。**不做。**

RoPE 的配对约定值得单独记一笔，因为它容易写错：`apply_rotary_emb` 用的是
**相邻两维交错**（`view_as_complex(x.unflatten(-1, (-1, 2)))`，`model.py:392-397`），
不是 split-half。

**这一条的产出是一个"不做"**，代价是一次只读的代数检查。这正是探针文化要买的东西。

---

### (c) prologue / epilogue 融合 —— 把 norm / residual / act_quant 折进 GEMV

**机制**。今天一层的 dispatch 数（`runtime/decode_layer.cpp`，逐条数过）：

| | dispatch 数 |
|---|---|
| 非源层的 attention 半边（`record_attention`） | **16** |
| `record_close`（MoE 输出折回流） | 1 |
| §7.4 的 compressor + indexer（源层，`record_ced`） | 最多 **13** |
| MoE（gate/up + h-quant + down） | 3 |
| engram（只在 engram 层） | 2 |
| tail（MhcClose / MhcFinal / Head / Argmax [+ SampleTopK]） | 4–5 |
| **一个普通层** | **20** |
| **一个 kv+index 源层（含 candidate block）** | **33** |

其中**纯粹是"把一个向量搬一下"的**：`MhcPost` / `MhcMix` / `MhcFinal` 三件事在
attention 半边和 FFN 半边各做一遍（6 个 dispatch），`WkvFinish` 是 `WkvGemv` 的尾巴，
`MhcClose` 是下一层 `MhcPost` 的前半。

**候选融合**（每一条都要单独 A/B，不要打包——v0.11 那次 fleet 把三件事打包上，整体变慢，
花了一个 per-layer timeline 才知道是哪一件）：

| # | 融合 | 省掉的 dispatch | 预测（砍半） |
|---|---|---|---|
| c1 | `WkvGemv` + `WkvFinish` 合一（K-split 版本已经是这个形状：`WkvKSplit`+`WkvKFinish` 替换而不是追加） | 1 | 0.6 µs × 40 = 0.02 ms |
| c2 | `MhcClose` 折进下一层的 `MhcPost`（design §7.7 说它本来就是同一件事，只是为了"逐层验证器"才单独 dispatch） | 1 | 0.02 ms |
| c3 | `GateScore` + `GateTopK` 合一（top-k 已经是单 workgroup） | 1 | 0.02 ms |
| c4 | `WoB` 的 epilogue 直接写出 act_quant 好的 FFN 输入，省掉 host 的一次 20 KB 读 + 一次 `act_quant` | 0 个 dispatch，但省 **MoE host 1.0 ms/token** | **0.5 ms/token** |

**注意 c1–c3 加起来只有 0.06 ms**。dispatch+barrier 实测 **0.56–0.66 µs**（STATUS.md §3 的 27），
design §3.4 曾经猜 5–20 µs 并因此恐慌了一轮——**融合 dispatch 本身几乎什么也买不到**。
真正值钱的是 **c4**，它省的不是 dispatch，是那次 host 往返。

**先量再做**：跑 §4 的 trace，看 `--layer 20` 里这些 dispatch 各自的 busy。
如果 `MhcPost`/`MhcMix`/`MhcFinal` 三件加起来不到一层的 5%，c1–c3 直接划掉。

**结果（Track PD）：c1–c3 划掉。** 层内 barrier 实测是**普通层 17 个 dispatch 一共 18 µs、
源层 26 个一共 24 µs** —— 40 层 = **0.8 ms/token**，比这里预测的 0.06 ms 大一个量级，
但仍然只有抖动带的四分之一。`MhcPost`/`MhcMix`/`MhcFinal` 六件在一个普通层里是
7.6+14.0+6.2+3.8+13.6+13.6 = **58.8 µs / 1,895 µs = 3.1%**，低于这里写的 5% 线。
**c4 没有被这次 trace 否掉**：它作用在 host 往返上（MoE host 实测 1.6–1.9 ms/token），不在 dispatch 上。

---

### (d) Q 提前发出 / 每个 phase 的 K-split

**机制**。两件独立的事，绑在一起是因为都要 trace 才能判断：

- **Q 提前**：`WqA` / `WqB` 只依赖这一层的 block 输入，不依赖 KV 路径、不依赖 indexer。
  今天它们排在 `MhcFinal` 之后、`WkvGemv` 之前，串行。如果 trace 显示 `WkvGemv` +
  `WkvFinish` + 13 个 ced dispatch 期间 GPU 有空闲，Q 可以挪到它们**之前**并与之重叠。
  fleet 的 v0.18（q_c 由空闲 worker 提前发布）就是这个形状，实测 **−1.5%**——
  而它之前三次同类"提前发布"的尝试全部失败，**唯一的区别是等待者有没有东西可以重叠**。
- **每个 phase 各自的 K-split**：今天 K-split 的档位是全局默认
  （`wq_a` 2 / `wkv` 2 / `wo_a` 4 / `wo_b` 8，一行一 lane，`p2_attention.md` §13.3）。
  这些是在 Track J 的 microbench 上定的，**而 Track J 的接口至今没有被 runtime 采纳**
  （STATUS.md 限制 6，且它的 LDS 修复在真机上看不到）。

**预测（砍半）**：Q 提前 −1%（热步 −0.8 ms）；K-split 重新定档 **未知**，要 trace 说话。

**成本**：Q 提前是 `record_attention` 里换个顺序，低；K-split 要先把 Track J 的接口接进 runtime，中。

**判据**：这两条**都不要在 trace 上 GPU 之前动**。没有 per-dispatch 的 busy/gap，
"哪里有空闲可以重叠"完全是猜的——而 fleet 的记录说，三次猜错、一次猜对。

**结果（Track PD）：Q 提前也划掉，理由和 c1–c3 是同一个数。**
trace 说 `WkvGemv`+`WkvFinish`+ced 期间 GPU **没有空闲可以重叠**——
源层 20 的 26 个 dispatch 之间一共只有 **24 µs** 的 gap，ced 那 9 个各 0.4–1.4 µs。
`WqA`/`WqB` 挪到它们前面**没有东西可以填**（fleet v0.18 之所以赚 1.5%，是因为等待者那边有空闲）。
这一层唯一的空闲是 MoE 前那 271–335 µs，而 Q 在那之前就已经算完了。
**每 phase 各自的 K-split 仍然未测**（要先把 Track J 的接口接进 runtime）。

---

### (e) 2-bit expert —— 等 F5 的精度判定

**机制**。expert 权重今天是原生 FP4（18.8 MB 一个 expert）。
2-bit 会把 MB/token 从 2,269 降到大约 **1,200**，按 §4 的公式
`tok/s ≈ NVMe_eff / MB_per_token`，在 hit 0.837 上直接是 **3.65 → ~6.9 tok/s**。

**预测（砍半）**：**+50%**，是这张表上最大的一项——因为它作用在正确的那一侧（字节，不是指令）。

**前置**：F5 的精度判定。判据用 design §12 的 ≤5e-3，并且必须过 §1.5 的两道闸。
参考点：int8 激活量化在 expert (39,383) 上是 **8.85e-3**，超标（STATUS.md §3 的 2）。
权重侧的 2-bit 比激活侧的 int8 更危险。

**成本**：如果 F5 说不行，成本是 0（不做）。如果行，是一条新的 dequant 路径 + 一次重新打包。

---

### (f) 命中率杠杆 —— 等 F4

**机制**。§4 的公式的另一半。今天已知的三件事：

1. **容量每翻一倍，hit +0.093**，5500 槽（96.34 GiB）是本机上限，给 0.9431 / 6.05 tok/s。
   **容量这条路走到头了。**
2. **decode 的 86.6% expert 在 prompt prefill 里已经出现过**（`prefill_hit = 0.7607`）。
   prefill→decode 交接（R1 的 `handoff_`，默认开）**从未单独 A/B 过**。
3. 掉命中率的原因是**换话题**，不是上下文长度（同话题 0.94–0.955，跨话题 0.877–0.90）。

**预测（砍半）**：把 hit 从 0.84 提到 0.92 = MB/token 2,269 → ~1,400 = **+30%**。

**前置**：F4。以及 STATUS.md §7 的第 1 项——**每次 miss 20–31 ms 对着 ~2 ms 的真实盘时间**，
这 10× 没有解释；如果它是一个可修的 stall，那它比命中率本身更大。

**不要再试的**（都有编号）：lookahead 预取（23，每一个 (d,K) 都净负）、
score-aware / 每层配额 / 静态 pin（24）、饱和 cache 上的 reheat（21）、
顺序学习 auto-tune（22）。

---

### 汇总：预测表

| # | 实验 | 作用在 | 预测（砍半后） | 成本 | 前置 |
|---|---|---|---|---|---|
| (e) | 2-bit expert | 字节 | **+50%** | 中 | F5 的精度判定 |
| (f) | 命中率 0.84 → 0.92 | 字节 | **+30%** | 中 | F4；先解释 miss 的 20–31 ms |
| (c4) | `WoB` epilogue 写出量化好的 FFN 输入 | host 往返 | 热步 −0.5 ms（−0.6%） | 低 | trace |
| (a) | persistent-dispatch decode | submit + barrier | 热步 −4%，对话 token −2% | **高** | `residency_probe` |
| ~~(d)~~ | ~~Q 提前~~ **关闭** | — | **0，没有空闲可以重叠**（层内 gap 24 µs） | 已付 | 同上 |
| ~~(c1–c3)~~ | ~~dispatch 融合~~ **划掉** | — | **上限 0.8 ms/token = 0.8%** | 已付 | 同上 |
| (b) | absorbed-K attention | — | **0，机制不存在** | 已付（一次只读检查） | **关闭** |

**顺序的含义**：前两项作用在字节上，后面全部作用在 81.5 ms 里的那 7% 余量上，
而那 7% 对着 130–190 ms 的 stall。**(a) 是这张表上最贵的一项，收益排第四。**
它值得做的唯一情形是 hit 已经上到 0.95+、stall 降到 ~40 ms、热步重新变成主项之后。

---

## 4. per-dispatch trace：GPU owner 要跑的命令

机制已经落地（`runtime/trace.*`、`tools/trace_timeline.py`、`tests/test_trace.cpp` 6/6 过），
**但没有在 GPU 上验证过**——写它的 track 不允许起 GPU 进程。下面是要跑的东西和预期的输出。

### 4.1 采一份 trace

```powershell
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"

# 热步的 trace：先 warm 让 expert 全驻留，再采 8 步
.\build\deepmoe.exe run --model $env:DEEPMOE_MODEL_DIR `
    --steps 8 --warm 8 --slow-prefill `
    --trace bench\results\trace_hot.bin `
    --profile bench\results\trace_hot.jsonl
```

### 4.2 读它

```powershell
# 一个 kv+index 源层（layer 20），逐 dispatch
.venv\Scripts\python.exe tools\trace_timeline.py bench\results\trace_hot.bin --layer 20

# 一个普通层，对比
.venv\Scripts\python.exe tools\trace_timeline.py bench\results\trace_hot.bin --layer 21

# 整个 token 按 phase 汇总
.venv\Scripts\python.exe tools\trace_timeline.py bench\results\trace_hot.bin --summary
```

### 4.3 预期输出，以及每一种"不对"的含义

`--layer 21`（普通层）应该给出 **20 行**，形如：

```
token N, layer 21: 20 dispatches, span ~2000 us
    # phase      stage                       start        end      busy      gap  sub
    0 attention  mhc_post                    0.0us      2.1us     2.1us    0.0us   21
    1 attention  mhc_mix                     2.6us      3.9us     1.3us    0.5us   21
   ...
  busy XXX us in 20 dispatches, gaps YY us (Z% of the layer), span ~2000 us
```

判据，按重要性排：

| 看什么 | 期望 | 不对的话意味着 |
|---|---|---|
| dispatch 数 | 普通层 **20**，源层 **33**（含 candidate block） | 与 §3(c) 的计数不符 → `record_ced` 的分支和这份文档对不上，先改文档 |
| `--summary` 的 attention busy | ≈ **0.90 ms/层 × 40 = 36 ms**（`p2_decode.md` §10） | 差 >10% → trace 和 profiler 量的不是同一件事，先修 trace |
| `--summary` 的 moe busy | ≈ **0.73 ms/层 = 29.4 ms** | 同上 |
| gaps 占 span 的比例 | **未知，这是这次要拿的数** | 这个数直接决定 §3(a) 值不值得做 |
| `WARNING: N dispatches lost a stamp` | **不该出现** | query pool 太小。`trace::Tracer::suggested_pool(40)` 给 2,784 槽；如果还不够，说明每层的 dispatch 数比数出来的多 |
| `host time not on the GPU` | 对照 profiler 的 `dispatch` + `cpu_sync` | 差很多 → 有一段 host 时间没有被任何一个桶接住 |

**一个具体的风险**：`Engine::cmd_open` 只在 token 的**第一次** open 时 reset query pool，
`tsq_used_` 在 token 内单调增长。40 层 × ~20 dispatch × 2 stamp ≈ 1,600 槽，
源层再多一些——`suggested_pool(40) = 2 × (40 × 34 + 8) + 64 = 2,784`，有余量。
但**如果 `--trace` 打开之后引擎报 `no GPU timestamps`**，说明 2,784 槽的 `VkQueryPool`
创建失败了；那就退回 `--trace` 只采前几层，或者把 `suggested_pool` 调小。
**这是唯一一个只能在真机上发现的问题。**

### 4.4b 真机上跑过之后（2026-09-18，Track PD）

**trace 机制验收通过**：8 步 × 40 层、64,282 条记录，**没有一条 `lost a stamp`**，
`suggested_pool(40) = 2,784` 槽够用，`host time not on the GPU` = 0.000 ms，
开销实测 **0**（带 `--trace` 102.07 ms/热步 vs 不带 102.3–106.2 ms）。

两处要改的是**文档**，不是 trace：

1. **每层的 dispatch 数**。§3(c) 数的是「普通层 20 / 源层 33」，真机是
   **普通层 17 / 源层 26**（MoE 的 gate/up + h-quant + down 是**一个**融合 dispatch
   `moe_gateup+hquant+down`，不是三个；`MhcClose` 没有单独出现）。
   741 dispatch/token、41 submit。
2. **一个 trace token id 可能装着好几遍前向**。`Tracer::token_begin` 用的键是**位置**，
   而 `--warm N` 把同一个位置解码 N+1 遍，所以 `--warm 8` 的 9 遍全部落在同一个 token id 上，
   每一遍的 `seq` 又都从 0 开始。**按 seq 排序会把它们交错**，gap 于是变成「两遍之间的等待」
   （一个 token 读出 6,733 个 dispatch、`host time not on the GPU` = **−753,222 ms**）。
   `tools/trace_timeline.py` 已修：按 seq 重置切分，默认显示**最后一遍**（最热的那遍），
   `--pass N` 选别的，并在多遍时打印一行说明。

### 4.4 然后把结果写回来

`--summary` 那张表（每个 phase class 的 dispatch 数 / busy / % / gaps）**直接进
STATUS.md §2**，替换今天那张只有七个桶的分项表。
`--layer 20` 的逐 dispatch 表进 `p2_attention.md`。

---

## 5. 探针：GPU owner 要跑的命令

三个都已经编译（`build/residency_probe.exe`、`sharing_probe.exe`、`capacity_probe.exe`），
shader 过 `slangc` + `spirv-val`，**每一个等待都有硬上限**，`--help` 路径不碰 GPU。
**都没有在 GPU 上跑过。**

### 5.1 residency_probe —— §3(a) 的可行性

```powershell
.\build\residency_probe.exe --max-groups 2048 --spin 200000 --rounds 1000 `
    --csv bench\results\residency_probe.csv
```

三段输出，三个判据：

| 段 | 输出 | 判据 |
|---|---|---|
| residency census | `groups` / `peak` / verdict | `peak` 在某个 group 数之后不再跟着涨 = 常驻上限。**persistent kernel 不能超过它**，超过就死锁 |
| ping-pong | `X.XXX us a round trip` | 对照 fleet 的 1.36 µs（idle）。一次 submit 0.156 ms ÷ 这个数 = 一次省下的 submit 能换多少个 device 侧事件。如果 <10，方案不成立 |
| forward progress | `progressed` / `TIMEOUT` | **在低于常驻上限的 group 数上 TIMEOUT = 共存不蕴含前进 = 自旋等待 NO-GO**。这时 §3(a) 退化成"一张 dispatch 图"，收益只剩 barrier，不剩 submit |

**先跑这个，在写任何 persistent kernel 代码之前。** 三个判据里任何一个不过，(a) 就关闭，
像 (b) 一样写进台账，代价是十分钟。

### 5.2 sharing_probe —— 共享是不是免费的（fleet 的 microbench (g)）

```powershell
.\build\sharing_probe.exe --buffer-mb 256 --max-groups 320 --disjoint `
    --csv bench\results\sharing_probe.csv
```

输出是一张矩阵：行是 working set，列是并发 reader 数，格子是**每个 reader 的 GB/s**。

- **行是平的** → 共享免费，"读一次再发布"这一类设计**全部不可能赚**。
  这会一次性解释 STATUS.md §3 的 10（`heads_per_wg` 1/2/4/8 = 69/100/169/288 µs）
  和 11（`pv_heads_per_wg=8` = 120 µs vs 25），并且封掉将来每一个同形状的提议。
  fleet 用这一个探针关掉了两个优化，每个都值一次 kernel 重写。
- **行在下降** → L2/MALL 在并发下是瓶颈，共享读的设计值得写。

### 5.3 capacity_probe —— 容量悬崖，以及它是容量还是分页

```powershell
.\build\capacity_probe.exe --readers 40 --buffer-mb 512 --csv bench\results\capacity_probe_a.csv
.\build\capacity_probe.exe --readers 40 --buffer-mb 512 --path b --csv bench\results\capacity_probe_b.csv
```

先验：8060S 有 32 MB MALL，raw-read 上限 215–218 GB/s（两条路差 <0.5%）。

- 悬崖之后**平** = 容量边界。低于它的 working set 值得安排，高于它 tile 大小无所谓
  → STATUS.md §3 的 17（"更紧的 LDS 预算反而更慢"）就有了解释。
- 悬崖之后**继续跌** = TLB/分页 → §3 的 30（path B 的 12% 是 GART 4 KiB page walk）
  有了支持，大页（§3 的 9，被 `SeLockMemoryPrivilege` 挡住）才值得去争权限。
- 两条路的悬崖在同一个大小、但 path B 的平台更低 = **把 page-walk 代价从容量里分离出来了**，
  这正是 30 断言但从未单独量过的东西。

### 5.4 三个探针的共同纪律

- 一次一个，安静机（§1.1）。
- 三个都带 `--csv`，结果进 `bench/results/`，并把结论写回 STATUS.md §3 的编号表——
  **包括"什么也没测出来"**。一个平的矩阵和一个陡的矩阵一样值钱。
