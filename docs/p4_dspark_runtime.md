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
