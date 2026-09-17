# P4 — DSpark runtime 验证循环：方案、现状、缺口与实施计划

设计对应：design §7.12 / §10，规格 `docs/p3_dspark.md`（算法）与 `docs/p4_mgt1.md`（M>1 kernel）。
本文是 **DSpark 从"离线 oracle"走到"runtime 里能跑"** 的交接：循环本身长什么样、
今天哪些部件已经存在、还缺什么、按什么顺序补，以及每个缺口的判据。

**状态（2026-09-17）：循环本身未实现。** 本文是缺口分析 + 实施计划，不是完成报告。
之所以先写这一篇而不是直接写循环：把草稿链和 verify batch 逐 dispatch 拆开对过之后，
**runtime 侧缺的不是"循环"这段胶水，而是三个 kernel 能力**；在它们存在之前写出来的
循环只能对着一半的路径打桩，而打桩的投机解码恰好会破坏 design §10.2 的唯一硬不变式
（温度 0 下与无投机逐 token 相同）。先把缺口钉死，比先交一个跑不起来的循环有用。

---

## 0. 选定的方案：K = 16 格 → 一次草稿 forward → 一次 verify forward → 最长路径

这一节把方案本身钉死（`docs/p3_dspark.md` §3 的 K2 规格），后面的缺口分析都是它的推论。

```
每周期：
  1) 草稿 forward（M = 5，只跑一次）
       main_proj + main_norm → 3 个 mtp 块 → head → 5 × 129280 的 base logits B[5][V]
     每位取 top-K = 16 个候选（anchor 规则：位 0 用输入 token，位 i 用位 i−1 的 top-1；
     GPU 仍按参考顺序做 5 次 Markov GEMV + 加偏置，只是 argmax 换成 top-16 并读回
     候选 id、候选上的 base logit、该行全词表 logsumexp）
  2) CPU 建格（5 层 × 每层 16 候选，K × K 个 256 维点积），选**一条路径**：
       eal  = 最大期望接受长度 q0(1 + q1(1 + q2(1 + q3(1 + q4))))   ← 默认
       viterbi = 最大联合 log q
       chain   = 格内逐位 argmax（= 参考单链）
     5 层 × 16 的精确 DP 是 < 0.2 ms（§13.1）
  3) verify forward（M = k + 1 ≤ 6，只跑一次）：[last, path[:k]]
     与"单链方案"同一批 token 数、同一份 GPU 字节（§3.3）
  4) 采样 + 精确接受（温度 1，`accept_sampling_exact`）或贪心前缀匹配（温度 0）
     每个 verify 行要：候选 C_j 上的 logit、全行 logsumexp、把 C_j 屏蔽掉的样本、
     最后一行的整行样本 → 产出 a 个接受 token + 1 个修正/bonus
  5) 回滚被拒位置（窗口环 ≤ k 个槽）、把 a + 1 个已接受位置的 main_x 写进 mtp 环
```

**关键结论：树不改变 GPU 的账。** §3.3 与 §6.1 写得很直白——草稿的 GPU 字节与单链
方案**完全相同**，verify 也是同一次 M = k+1 forward；树只让 CPU 上的采样与比较变丰富，
代价是微秒级。所以：

* "改成树" **不能**绕过下面任何一个 kernel 缺口（M=6 的 MoE、M=5 的 head、bf16 输入的
  GEMV 都需要），它只是让**同一份** M=6 前向的产出更值钱（接受长度从 2.6–3.3 往上走）。
* 反过来，缺口补齐之后，树是**没有额外 GPU 代价**的升级——所以两条路可以按同一个顺序做。

**温度 0 与温度 1 是两条不同的验证判据**，这一点决定了测试怎么写：

| 模式 | 跑什么路径 | 接受规则 | 判据 |
|---|---|---|---|
| 温度 0（贪心） | `eal` / `viterbi` / `chain` 都可以（贪心时三者对"接受长度"等价，§15.4） | `accept_greedy` | **逐 token 与无投机输出相同**（design §10.2 的硬不变式）+ 接受长度 |
| 温度 1（采样） | `sample`（格上祖先采样）+ K = 16 | `accept_sampling_exact`（top-256 当词表、Gumbel-max 屏蔽样本） | **分布无损**：`dspark_tree.py lossless` 的频率检验（§13.3：TV 0.0033 / 噪声底 0.0039，χ² 53.7/45），不能要求 token 相同 |

`accept_sampling_exact` 的离线无损检验已经做了（4 组规则全在噪声底内）；runtime 要补的是
它需要的那几个数（下面 §2.4）。

---

## 1. 一个周期的完整 dispatch 序列（规格 → 部件）

命题：verify 从位置 `p` 开始，草稿给出 `path[0..k-1]`（k ≤ 5），verify batch 是
`[last, path[:k]]`，M = k + 1 ≤ 6。

### 1.1 主模型前向，M = k+1（verify）

| # | 做什么 | 今天的部件 | 状态 |
|---|---|---|---|
| 1 | 40 层的 M=6 批量前向 | `runtime::DecodeLayer::record_attention_batch` + `gpu::MgtRunner`（`mgt1_*.slang`，8 个 shader / 31 个 stage） | ✅ 逐阶段对过参考（`gpu_layer.mgt1_layer_batch_vs_steps`，M=2/4/6） |
| 2 | 第 37/38/39 层的 block 输入 `h.mean(hc)` → `main_hidden[15360]` | 无 | ❌ 见 §2.1 |
| 3 | 每行的 top-1（贪心接受要 argmax，采样接受要 top-Kv + lse） | `MgtStage::Head` + `HeadArgmax` / `HeadTopK`（`DecodeLayer::record_tail_batch`） | ✅ |
| 4 | 一层的 MoE，6 个 token × 各自的 top-6 | `MoeRunner`（`moe_gateup` / `moe_down`） | ❌ 见 §2.2（**最大的一个**） |
| 5 | 拒绝位置的窗口环回滚（≤ k 个槽 × 40 层 ≈ 200 KB） | `KvStore` 的 ring 视图 + `runtime::session::rollback_context` | ⚠️ 机制在（R2），但没有"快照 ≤k 个槽再恢复"的入口；`docs/p3_dspark.md` §3.5 |

### 1.2 草稿链，M = 5（`DsparkRunner` / `dspark_*.slang`）

| # | 做什么 | 今天的部件 | 状态 |
|---|---|---|---|
| 6 | `main_proj`(15360→5120) + `main_norm` | `DsparkStage::Gemv` + `RmsNorm` | ⚠️ 权重是 fp8，输入是 **bf16** → §2.3 |
| 7 | 3 个 mtp 块：`mega_mhc`×3 → `hc_pre` → `attn_norm` → attention → `hc_post`，然后 FFN 半边 | `DsparkStage::Gemv / RmsNorm / RopeQuant / WoA / AttnScore / AttnCombine` | ⚠️ attention 半边齐（M=5 已测）；**FFN 半边的 MoE 是 128 routed / top-3**，同样卡 §2.2 |
| 8 | mtp 窗口环写入（每个**已接受**位置一格，共 a+1 格） | `DsparkRunner` 的 ring 参数 | ❌ 需要 §4.6 的 `a+1` 接口（不是 1 格） |
| 9 | `head`（M=5，129280×5120 bf16） | `mgt1_head.slang` 的 `Head`（M 已支持） | ⚠️ mgt1 的 `Head` 用得到，但草稿的头挂在 `DsparkRunner` 的地址表上；要么加 stage、要么让 `MgtRunner` 认得 mtp 的权重 |
| 10 | 5 次**严格顺序** Markov GEMV（129280×256 bf16，每位 66.2 MB）+ argmax | `DsparkStage::MarkovBias` + `AddBiasArgmax`（M=1 链，`kDsFlagTokenFromBuf` 让 5 步在 GPU 上串起来） | ✅ |
| 11 | confidence head（5 × 5376 点积）→ 选 k | `DsparkStage::Confidence` + `cpu::dspark::k_from_confidence` | ✅ |
| 12 | 单链草稿（`chain` 规则）或 K=16 格（`eal` / `sample`） | `cpu::dspark::Lattice`（逐位对齐 Python，`suite.dspark_tree` 过） | ✅ |

### 1.3 接受与回滚（CPU）

| # | 做什么 | 今天的部件 | 状态 |
|---|---|---|---|
| 13 | 贪心：`a = 最长前缀使 path[j] == argmax[j]`，产出 `path[:a] + [argmax[a]]` | `cpu::dspark::accept_greedy` | ✅ |
| 14 | 采样：`accept_sampling_exact`（对温度 1 无损） | `cpu::dspark::accept_sampling_exact` | ✅ |
| 15 | 回滚被拒位置 + 写 (a+1) 格 mtp 环 | 无 | ❌ §1.1 #5 同一个入口 |

**一句话**：13 个环节里，部件的 6 个已经存在且验证过（#1、#3、#10、#11、#12、#13/14），
2 个缺机制，5 个缺 kernel 能力。缺的全在"把 M 维从 1 推到 5/6"这条线上。

---

## 2. 三个 kernel 缺口（按代价排序）

### 2.1 主模型第 37/38/39 层的 `main_hidden` 捕获

规格（`docs/p3_dspark.md` §1.1）：取的是**进入 block 之前**的残差流（engram 之后、
`hc_pre`/`attn_norm` 之前）在 hc 维上的均值，`[b,s,5120]`，三层拼成 15360。

今天 `Engine::run_layer` 只跑 M = 1；batch 路径 `record_attention_batch` 直接吃
`bb.x.host`（block 输入）。所以需要的是一句 host 侧拷贝：

```
if (L == 37 || L == 38 || L == 39)
    for m in 0..M-1: sum_hc(x[m][hc][dim]) / hc  ->  main_hidden[m][(L-37)*dim ...]
```

成本：M × 4 × 5120 floats 的读 + 求和（≈0.5 MB），可忽略。**这不是 kernel 问题，
是 Engine 的一小段代码**；列在这里是因为它必须在 batch 循环里做，而不是 trace 里。

### 2.2 MoE：6 个 token × 各自 top-6（最大缺口）

`MoeRunner`（`gpu/vulkan/moe_kernels.h`）的几何写得很清楚：

```cpp
uint32_t* ids();            // [slots]: which expert sits in each slot
float*    route_weights();  // [m][slots]
uint32_t  slots = 7;        // 6 routed + shared
```

- `ids()` 是**一组**专家（7 个槽），`route_weights()` 才是 `[M][slots]`。
- 也就是说：**一个 dispatch 里所有 M 个 token 必须路由到同一组专家**。
  M = 1 时这没问题；verify batch 里 6 个 token 的 top-6 是 6 个不同的集合
  （`docs/p3_dspark.md` §12.6 实测 6 个 token 的并集 ≈ 26 个 expert，
  `docs/p4_mgt1.md` §7 缺口 1 也点过：`MoeRunner` 是 7 槽、`route_weights` 是稠密
  `[M][slots]`，M>1 verify 需要 >7 槽或分组 dispatch）。
- `docs/p4_mgt1.md` §7 已经把这个列为 DSpark 的第 1 号缺口。

三条出路，按代价：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A · 逐 token dispatch**（不改 kernel） | 每个 token 一次 `stage` + `record`（6 次/层，240 次/verify batch） | 每层 6× 的 A/B 启动与 host 侧 staging；**这就是 20 t/s 要跨过的坎** |
| **B · 多组并行（union 表）** | 把 6 个 token 的并集摊进 n 个 7 槽分组（`set_accumulate` 已经支持多组，见 `MoeRunner::set_accumulate` 与 `docs/kernel_p2_moe.md` §6.3 的 `deferred` 调度），每组一个 dispatch | 需要 kernel 侧支持"每 token 每槽的权重矩阵"（今天的 `route_weights` 是 `[m][slots]`，正好够用——缺的是"哪些 token 参与哪一组"） |
| **C · 真正 batch MoE** | kernel 里按 token 查自己的槽表 | 新 kernel 族，最大工作量 |

**先做 A，用它测出 C(M) 的实际形状**——A 的每层代价就是 6 × M=1 的 MoE 代价，
是 C(M) 的下界还是上界要测了才知道（启动与 staging 可能被 6 个 token 摊薄）。
`docs/p4_mgt1.md` 判定 G2 的判据（M=6 ≤ ×3.06）就是围绕这条。

**2026-09-17 实测：A 已经做出来了，而且它不能作为 DSpark 的落地方案。**
`GpuMoeBridge::run_batch`（`runtime/moe_bridge.{h,cpp}`）接受 M ≤ 6 列、每列各自的
top-6 id/权重/激活，正确性判据是"第 m 列与 M=1 路径对该 token 的结果**逐位相同**"
（`tests/test_gpu_moe.cpp::gpu_moe.the_verify_batch_takes_one_expert_set_per_column`）。
逐列 dispatch 的代价，层 0、每列互不相交的 6 个 expert、全部驻留
（`tests/test_gpu_moe.cpp::mgt1.moe_m_curve`，`ctest -R bench.mgt1_moe_m_curve`）：

| M | ms/层 | ms/token | C(M)/C(1) |
|---:|---:|---:|---:|
| 1 | 1.786 | 1.786 | 1.00× |
| 2 | 3.615 | 1.808 | 2.02× |
| 4 | 7.484 | 1.871 | 4.19× |
| 6 | 11.674 | 1.946 | **6.54×** |

也就是说 **A 等价于把 6 次 M=1 的 MoE 原样跑一遍**（每 token 1.79–1.95 ms，与 M=1 的
1.786 相同）：没有摊薄，只有多出来的 dispatch 开销。把它和 C(M) 加起来：

- verify 一批（M=6）每层 ≈ C(M) 4.7 ms + MoE 11.7 ms ≈ **16 ms** → 40 层 ≈ **640 ms**；
- 同样的 6 个 token 顺序走 M=1：40 × (1.3 + 1.8) = **124 ms**。

**在"每列不同 expert"的假设下，逐列 dispatch 的 verify 比顺序 M=1 慢约 5×**，
所以 A 只适合当作正确性基线（它证明了这个接口的语义），**不是性能路径**。
真正的出路是 §2.2 的 B（按 expert 归并成多组 dispatch）或 C（kernel 侧 `ids[m][slots]`）；
在那之前，DSpark 的 TPS 投影应该按**小 k**（k=1–2）算——C(M) 那一半在 M=2 时每 token
只要 0.6–0.8×，而 MoE 那一半在 M=2 时只有 2 列。

### 2.3 草稿链的输入是 bf16，而 fp8 GEMV 只吃 fp32/bf16 **激活**、**fp8 权重**

`mgt1_gemv.slang` 的 stage 0/1 是 fp8 权重 × fp32 激活（内部 ActQuant 到 E4M3）。
草稿链里：

- `main_proj` / `wq_a` / `wq_b` / `wkv` / `wo_b`：fp8 权重 ✅，输入是 bf16 的
  `main_x` / 残差流 → **需要把 bf16 激活当作输入**（今天 stage 0 从 `Ptr[kX]` 按 fp32
  读；bf16 输入要么加一个 flag，要么 host 上转 fp32，要么让 forward 产出 fp32）。
- 反例：`wkv(main_x, M=1)` 的 `main_x` 是 `main_norm(main_proj(main_hidden))` 的输出，
  参考里它是 **bf16**；`DsparkRunner` 的 `RmsNorm` 有 `kDsFlagInBf16`，所以这一段
  可以走 `DsparkRunner` 自己的 gemv（它本来就是 fp8 权重 + 可配激活类型）。

**判据**：`main_proj` 的输入若被舍入到 E4M3（ActQuant 的固有行为），草稿的
`wkv` 会偏；草稿偏一点**只会降低接受率，不会破坏正确性**（verify 主模型仍然对）。
所以这一条可以接受"先跑起来，再看接受率掉多少"，但必须在文档里写清楚是近似。

### 2.4 verify 行的读回：`accept_sampling_exact` 要的四个数

温度 1 的精确接受（`docs/p3_dspark.md` §3.3 规则 2）对**每个 verify 行 j** 要：

| 要什么 | 干什么用 | 今天的部件 | 状态 |
|---|---|---|---|
| 候选 `C_j` 上的 logit（草稿阶段就知道候选 id） | 算 `p(c) = exp(l_c − lse)`、残差 `max(0, p − q)` | `HeadTopK` 的候选段（每线程 ≤ 256，`kCap`）里**可能**有，但要在主机上做集合成员判断 | ⚠️ 需要一个"按 id gather"的读回 |
| 整行 `logsumexp` | 归一化 | `HeadTopK` 的直方图 + `tail`（`out[2]`），是**分箱近似**；`ovf` 时整行拷回 | ⚠️ 精确版要么用直方图 + 溢出尾部，要么最后一行做一次整行扫描 |
| 屏蔽掉 `C_j` 的 Gumbel-max 样本 | 拒绝时的残差采样 | 无 | ❌ 新的一遍整行扫描（head 已经产出那一行，只是一次 reduce） |
| 最后一行的整行样本（bonus） | 全接受时的 bonus token | M=1 路径的 `sample_topk`；batch 路径 `HeadTopK` 不解 mask | ❌ 同上 |

这四个数都**不需要新的 kernel 族**，是 `mgt1_head` 的一个附加 stage 加一段主机侧读回；
但它们必须在 §3 的第 5 步之前就位，否则温度 1 只能退化成"top-Kv 截断"那版
（`accept_sampling`，§13.3 实测尾部质量均值 3.3%、最坏 42%，**不是** `generate.py` 的分布）。
贪心路径（温度 0）不需要这一节，per-row argmax 已经在 `MgtStage::HeadArgmax` 里。

---

## 3. 实施顺序（每一步都有可测判据）

1. **`main_hidden` 捕获（§2.1）** — 半天。判据：batch 跑 M=6，把
   `main_hidden[m][0:5120]` 与 oracle `--mgt1` 导出的同一位置的 `main_hidden` 比 cos ≥ 0.9999。
2. **MoE 逐 token dispatch（§2.2 方案 A）** — 1–2 天。判据：
   (a) M=6 时每 token 的 `moe_out` 与 oracle 的逐 token MoE 输出 cos ≥ 0.9999；
   (b) 同一层 batch 的 6 行 MoE 结果与 6 次 M=1 step 的逐行结果**逐位相同**
   （`mgt1_layer_batch_vs_steps` 已经是这个判据的形式）；
   (c) 计时：M=6 每层 MoE 的 ms，与 6 × M=1 的比值 —— **这个数就是 G2 的答案**。
3. **`Engine::forward_batch(p0, tokens[M])`** — 1–2 天。把 1 的捕获 + 40 层 batch +
   MoE（2）+ `record_tail_batch` 串起来，返回每行 argmax/top-Kv/lse。判据：
   M=6 的 `[last, path]` 跑出来的逐行 argmax 与 6 次 M=1 decode step 的 argmax 相同
   （`docs/p3_dspark.md` §14 的批边界效应要先量出来，`DEEPMOE_MGT1_CTX` 已有 l3/ctx4k 两个上下文）。
4. **回滚入口（§1.1 #5）** — 半天。`KvStore` 上"快照 ≤k 个 ring 槽 / 恢复"的两个函数，
   单测：写 k 个位置 → 快照 → 再写 → 恢复 → 与快照前逐字节相同。
5. **贪心投机循环 + `generate(speculative=true)`** — 2 天。判据（design §10.2 的硬不变式）：
   同一 prompt、温度 0、`--dspark` 开/关，**输出 token 序列逐 token 相同**；
   并记录每周期接受长度 `E[tokens]`，与 `docs/p3_dspark.md` §12.2 的 2.6–3.3 对照。
6. **草稿链进 runtime（§2.3 / §1.2）** — 3–5 天。判据：草稿 5 个 token 与 oracle 的
   `forward_head` 输出逐位相同（`docs/p3_dspark.md` §3.1 的 705/705 是参考链的判据）。
7. **采样接受（温度 1 无损）** — 1 天。`accept_sampling_exact` 已经在 CPU 侧，
   需要的是 verify 行的 `cand_logit / lse / masked / full`（`HeadTopK` 的输出格式
   `kMgtTopKRecordWords` 已经带了 top-K + 直方图，够不够要核）。

**到第 5 步结束**，就有了"单链草稿 + 精确接受 + 可回滚"的完整 verify 循环，
即使草稿链还是 M=1 的慢版本 —— 那时 `T_draft` 大约 42 ms（`docs/p4_mgt1.md` §7 缺口 2：
head 的 M=5 从 42 ms 到 19 ms 是 Track T 的下一项），接受长度 2.6–3.3，
于是可以拿**实测**的 `E[accept] / (T_draft + T_verify(M=6))` 回答 20 t/s 那个问题，
而不是继续用 design §10.1.2 的投影。

---

## 4. 现在就能测、不依赖上面任何一步的三件事

1. **C(M) 曲线（`bench/results/mgt1_p4.csv`，缺）** — `MgtRunner` 的 M=1/2/4/6 每层代价。
   `gpu_layer.mgt1_layer_batch_vs_steps` 已经把 batch 路径跑通了，差的是计时循环。
   判据就是 `docs/p4_mgt1.md` §5 的 G2（M=6 ≤ ×3.06）。
2. **验证 batch 的 expert 并集** — `docs/p3_dspark.md` §12.6 有离线数（≈26），
   runtime 侧要在真实 KV 上量：M=6 的并集是否真的落在 7 槽的 3–4 倍以内。
3. **批边界隔离（G3）** — M=6 的 per-row logits 与 6 次 M=1 的逐行 logits 差多少。
   `DEEPMOE_MGT1_CTX=l3,ctx4k` 已经把数据准备好（`traces/mgt1/`），
   `gpu_layer.mgt1_layer_batch_vs_steps` 已经有逐阶段 cos 的框架。

这三件都不需要 MoE 改动，也不需要在 runtime 里插桩，是 20 t/s 判定的**前置输入**。

---

## 5. 与其它 track 的接口约定

- **P4-T（M>1 kernel）**：`MgtStage` 的枚举是 append-only（`runtime/` 按枚举名找 stage）；
  新增 stage 只能加在 `Count` 之前，并同步 `gpu/vulkan/decode_kernels.cpp` 的
  `{stage, "shader", …}` 表与 `stage_name`。
- **P4-R1（hit rate）**：投机解码把"N 个 token 的访问"变成"一个 batch 的访问"，
  expert 并集（§12.6）比逐 token 的 6 个多，对 expert cache 是**更差**的局部性。
  reheat（`docs/p4_hitrate.md` §7）与这里是同一条线上的两件事，谁先落地都要重测 hit。
- **P4-R2（KV）**：§1.1 #5 的回滚入口应该和 `runtime/session.h` 的
  `rollback_context` / `park_context` 放在一起，而不是新开一套。

---

## 附：M=6 MoE 逐列 dispatch 的根因（2026-09-17）

**一句话**：kernel 没有错，错在 host 把"哪一列"这件事配错了——`run_batch` 每次 dispatch
前把**全部 6 列** x 刷成当前列的激活，dispatch 后却从 **y 的第 0 列**拷回
`call.y[m*hidden]`；而第 m 列的 h/y 是 kernel 用 `x[m]` 和 `route_weights()[m]` 算的。
只有第 0 列三者自洽，所以第 0 列逐位正确、其余列差 15–22% of |y|max。
修的是 host 的列号配对，不是 kernel；`XLayout` 这个开关本身是死的（见 5）。

### 1. 根因与证据

**(a) kernel 侧根本没有"活列数"这个东西**（这是原 `moe_bridge.h`/`create()` 注释里
"`min(M, pc.m)`"那句话的错处，先自行核实过）：

- `gpu/shaders/moe_gateup.slang:52-60`：`GateUpPush{layer, experts_per_layer, num_slots,
  n_rows, k, swiglu_limit}`，**没有 m**；
- `gpu/shaders/moe_down.slang:38-47`：`DownPush{layer, experts_per_layer, num_slots,
  list_count, n_rows, k, flags}`，也没有 m；
- 于是 M 是特化常量（`runtime/moe_bridge.cpp:130` 的 `spec.m = kMoeBatchMax`，即 6），
  **每次 dispatch 都会重算全部 6 列**：x 按列读（`moe_gateup.slang:161`，XMode 0；
  `:396/:463/:515` 是其它 XMode），路由权重按列读（`moe_gateup.slang:552`
  `RouteW[m * pc.num_slots + slot]`），h 写到 `H[(m*num_slots+slot)*n_rows+row]`
  （`:554-555`），第三个 dispatch 整片量化 M 列 h（`moe_hquant.slang:46,71`，
  `gpu/vulkan/moe_kernels.cpp:331-332` 的 groups 是 `spec_.m * list_count * …`），
  dispatch B 从 `H[(m*num_slots+slot)]` 读（`moe_down.slang:106,136,156`）并写
  `Y[m*pc.n_rows + rowBase + r]`（`moe_down.slang:468-469`）。
- 结论：**"让第 m 列可见"完全靠 host 自己对齐列号**，kernel 不看任何 live count。
  一次 dispatch 只表达一组 expert（`ids()` 是 `[slots]`），所以第 m 次 dispatch
  （ids = token m）之后，只有**第 m 列**的 (x, 权重行, y) 三元组同属 token m。

**(b) host 侧三处列号不一致**（诊断开始时那版未提交的工作树，行号为当时版本）：

- `runtime/moe_bridge.cpp:366` `run_batch(call, XLayout layout)` 的 `layout`
  **全程没有被读过**；`column_x(m)`（原 `:381-386`）无条件把 6 列 x 都写成第 m 列的
  激活，然后在 `:414` 调用——于是 `XLayout::Staged` 与 `OneColumn` 完全等价，
  "别的列是陈旧的"这个假设从一开始就测不了（旧输出里 `x the batch ran on`
  第 0、1 列 4971/4854 个 word 不同、只有最后一列 0，就是这个原因）；
- `stage_batch`（今 `runtime/moe_bridge.cpp:354-358`）把每列权重写进
  `route_weights()[m*slots + s]`——**这一步是对的**；
- 但 `:420-421` 拷的是 `runner_.y()`，也就是 **y 第 0 列**，而 y[0] 的 h 来自
  `x[0]` 与 `RouteW[0*slots + s]`（第 0 行权重）。
  于是第 m 列实际算的是 `f(x_m, w_0, ids_m)`，M=1 参考是 `f(x_m, w_m, ids_m)`：
  `w_m ≠ w_0` 时必然差，`m = 0` 时因为 `w_0 == w_0` 而"恰好逐位对"。

**(c) kernel 的逐列机制本身是对的**，静态、动态各验过一次：

- 静态：`spirv-dis build/shaders/moe_gateup.spv` 里 `RouteW` 的索引是
  `m * pc.num_slots + slot`（`OpAccessChain … %RouteW`，索引 = `OpIMul %m, pc[2]` + slot，
  m 是 `OpULessThan %m %M` 的循环变量），`X` 的索引是 `m * (pc.k/8) + blk*4 + w`；
  不存在"永远读第 0 列 / 第 0 行"。
- 动态：下面实验里 runner 的 y[2]（唯一一列 x 没被 `column_x` 覆盖掉的）与它自己的
  M=1 参考**逐位相同**（`0.000e+00`），即第 2 列的 `x[2]` 与 `RouteW` 第 2 行都被
  kernel 正确采用了。

### 2. 最小实验（证明它）

**实验**：一批 3 列，**三列用同一组 expert**——这样"dispatch 了几次"不可能改变答案，
只有 x 和路由权重按列不同；然后**直接读 runner 的 y 缓冲的每一列**
（`GpuMoeBridge::y_host()` 返回 y 的基址，列 m 在 `[m*hidden]`），与"该列自己的 x +
自己的权重行"的 M=1 运行逐位比。这段现在留在测试里，名字是 `column pairing`。

```powershell
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
Set-Location C:\Users\Asus\code\deepmoe\build\p4-integ
cmake --build build
.\build\tests\deepmoe_tests.exe "gpu_moe.the_verify_batch_takes_one_expert_set_per_column"
```

修复**前**（同一次运行里这几行；当时这组诊断的标签是 `E1 column …`，数值不动）:

```
       column pairing 0 (same experts, per-column x+w): runner y[0] vs M=1 2.812e+00 | run_batch's column 0.000e+00
       column pairing 1 (same experts, per-column x+w): runner y[1] vs M=1 2.547e+00 | run_batch's column 2.532e+00
       column pairing 2 (same experts, per-column x+w): runner y[2] vs M=1 0.000e+00 | run_batch's column 5.497e+00
       per-column weights only    columns spread 0.000e+00, col 0 vs the M=1 run 0.000e+00
       per-column x only          columns spread 2.812e+00, col 0 vs the M=1 run 0.000e+00
       column 1: run_batch(M=1) vs M=1 0.000e+00 | batch(M=3) vs run_batch(M=1) 1.864e+00
       column 2: run_batch(M=1) vs M=1 0.000e+00 | batch(M=3) vs run_batch(M=1) 2.778e+00
       x the batch ran on, column 0: 4971/5120 words differ from the staged x
       x the batch ran on, column 1: 4854/5120 words differ from the staged x
       x the batch ran on, column 2: 0/5120 words differ from the staged x
```

怎么读：

- `run_batch's column 0 = 0` 而 `runner y[0] = 2.812`：最后一次 dispatch 之后，
  **y[0] 已经是 `(x_2, w_0)` 的答案**（x 被 `column_x(2)` 整片刷成第 2 列），
  host 把每列都从 y[0] 拷出来，所以第 0 列"碰巧对"、第 1/2 列不对。
  三个拷贝差 `0 / 2.532 / 5.497` 就是 `|ref[0] − ref[m]|`，即"每列都在读 y[0]"。
- `runner y[2] = 0`：第 2 列是唯一 x 没被覆盖、又能配到自己权重行的一列，
  kernel 算得**逐位正确**——把 kernel 摘干净了。
- `per-column weights only` 的 spread 是 **0**：这正是 bug 的指纹（每列的路由权重
  都从第 0 行读，所以按列给的权重根本没起作用）；它在修复后变成非 0。
- 对照组（每列各自的 expert，即原始判据）：`1.44e-01 / 2.14e-01 of |y|max`。

### 3. 修复（host 侧，已做）

| 文件 | 改了什么 |
|---|---|
| `runtime/moe_bridge.cpp` | `run_batch` 去掉 `column_x` 与 `layout` 参数：每列 dispatch 只设 ids/list/list_count，不再重刷 x（`stage_batch` 已经把第 m 列激活和权重行都放好了）；回读改成 `runner_.y() + m*dim → call.y + m*dim`（今 `:428`）。`create()` 里那句"每个 shader 循环都被 `min(M, pc.m)` 限制"的注释改成事实（没有 `pc.m`，每次 dispatch 重算 6 列），并给出缓冲的真实大小（x 60 KB / h 295 KB / y 120 KB）。 |
| `runtime/moe_bridge.h` | 删掉 `enum class XLayout` 与 `run_batch(call, layout)` 的默认参数（见 5），把 batch 段注释改成"列 m 的 x、`route_weights()[m]`、`y[m]` 必须同一列号"的契约。 |
| `tests/test_gpu_moe.cpp` | 保留 `column pairing`（就是第 2 节的最小实验，并加 `CHECK(d_y <= 1e-7)` / `CHECK(d_out <= 1e-7)`，防止再退化）、隔离表、逐列对照、x 逐字校验、计时；删掉只服务于已被否掉假设的打印（"重跑同一批是否一致"、"每列 expert 落在哪个 slot"、以及"陈旧列"那版的措辞）。隔离表加了两个判据：`col 0 vs M=1 ≤ 1e-7` 恒成立，`per-column weights/x` 的 spread **必须 > 0**。 |

修复后同一条命令的输出（`1e-7` 相对判据、逐列逐位）：

```
       batch: 3 columns: stage 0.04 ms, expert rows 0.00 ms, dispatches 4.83 ms, wall 5.49 ms (1.83 ms a column)
       column pairing 0 (same experts, per-column x+w): runner y[0] vs M=1 0.000e+00 | run_batch's column 0.000e+00
       column pairing 1 (same experts, per-column x+w): runner y[1] vs M=1 0.000e+00 | run_batch's column 0.000e+00
       column pairing 2 (same experts, per-column x+w): runner y[2] vs M=1 0.000e+00 | run_batch's column 0.000e+00
       identical columns          columns spread 0.000e+00, col 0 vs the M=1 run 0.000e+00
       per-column weights only    columns spread 5.235e+00, col 0 vs the M=1 run 0.000e+00
       per-column x only          columns spread 2.812e+00, col 0 vs the M=1 run 0.000e+00
       column 0: run_batch(M=1) vs M=1 0.000e+00 | batch(M=3) vs run_batch(M=1) 0.000e+00
       column 1: run_batch(M=1) vs M=1 0.000e+00 | batch(M=3) vs run_batch(M=1) 0.000e+00
       column 2: run_batch(M=1) vs M=1 0.000e+00 | batch(M=3) vs run_batch(M=1) 0.000e+00
       x column 0 ran on: 0/5120 words differ from the staged x
       x column 1 ran on: 0/5120 words differ from the staged x
       x column 2 ran on: 0/5120 words differ from the staged x
       column 0: max |batch - one| 0.000e+00 (0.00e+00 of |y|max)
       column 1: max |batch - one| 0.000e+00 (0.00e+00 of |y|max)
       column 2: max |batch - one| 0.000e+00 (0.00e+00 of |y|max)
       columns differ by 0.376 of |y|max
[  ok ] gpu_moe.the_verify_batch_takes_one_expert_set_per_column
1 case(s) run, 0 failed, 0 assertion failure(s)
```

"columns differ by" 从 0.290 涨到 0.376 也是修复的一部分：以前三列都读第 0 行权重，
列间差异被压小了；现在每列用自己的权重。相邻的
`gpu_moe.the_verify_batch_computes_one_answer_per_column`（M=6、10 个变体、逐列对 oracle）
照样全过，说明改的只是 host 的配对。

### 4. 现在的每列代价

```
batch: 3 columns: stage 0.04 ms, expert rows 0.00 ms, dispatches 4.83 ms, wall 5.49 ms (1.83 ms a column)
```

- **和修复前一样**（修前同一条命令：4.74 ms / 5.55 ms / 1.85 ms 一列；修后另外两次跑：
  4.88/5.53/1.84 与 4.83/5.49/1.83 —— 跑与跑之间的抖动就有这个量级）：修的是正确性，
  不是速度。省掉的只是 host 侧每列 6×10 KB 的 x 覆盖（≈ 0.02 ms/列），
  GPU 该算的 6 列一列没少。
- 每列 ~1.8–1.9 ms 就是"一次 M=6 形状的 A + hquant + B"的价钱，与 §2.2 表里
  `mgt1.moe_m_curve` 的 M=6 = 11.674 ms / 6 列 = 1.946 ms/token 一致。
  也就是说：**逐列 dispatch 每个 token 花的是 M=6 的钱**，摊薄为零，这仍然是
  §2.2 判定"方案 A 只是正确性基线、不是性能路径"的那条结论。
- 想把这 ~1.9 ms/列 降下来只有 kernel 侧两条路：(i) 给两个 push struct 加一个真正的
  活列数（`pc.m`）并把 `for (m = 0; m < M; ++m)` 收成 `min(M, pc.m)` 甚至"只算第 m 列"，
  于是每次 dispatch 的 FMA/带宽按列数缩；(ii) §2.2 的 B/C（按 expert 归并成多组 dispatch、
  或 kernel 侧 `ids[m][slots]`）。在当前 kernel 下，host 怎么摆都不可能便宜。

### 5. `XLayout` 该不该留：不留，已删

删掉了（`enum class XLayout`、`run_batch(call, layout)` 的第二个参数）。理由：

1. 它**从来没生效过**：`run_batch` 没读 `layout`，`column_x` 两种模式都刷满 6 列，
   所谓 `Staged` 只是文档里的说法。留着它比删掉更容易误导下一个人。
2. 正确的形状只有一种，而且不是开关：第 m 列的 x 在第 m 列、权重在第 m 行、y 从第 m 列读
   （`stage_batch` + `run_batch` 现在就是这么做的）。这就是"每列一份 activations、
   整批只 act_quant 一次"的便宜形状，`OneColumn`（把每列都塞进 slot 0 再读 y[0]）
   没有任何剩余价值——它唯一"多"出来的好处是让 kernel 少读几列，而 kernel 无论如何
   都会算满 M 列。
3. `debug_x()` 保留（校验用），注释里写清 `>= m` 的列是上一次调用留下的。
