# P4 — DSpark runtime 验证循环：方案、现状、缺口与实施计划

设计对应：design §7.12 / §10，规格 `docs/p3_dspark.md`（算法）与 `docs/p4_mgt1.md`（M>1 kernel）。
本文是 **DSpark 从"离线 oracle"走到"runtime 里能跑"** 的交接：循环本身长什么样、
今天哪些部件已经存在、还缺什么、按什么顺序补，以及每个缺口的判据。

**状态（2026-09-18，Track F1 更新）：§2.2 与 §2.4 的两个 kernel 缺口已关闭，循环的 CPU 半边
（`runtime/speculate.h`）已实现并有判据；仍然跑不起来的唯一原因是 `Engine::forward_batch`
不存在——见 §6。下面 §0–§5 是 2026-09-17 的缺口分析，保留原样，§6 是执行结果与修正。**

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
**2026-09-17 状态**：`mgt1_gemv.slang` 已加 `kGemmFlagInBf16`（`MgtGemvPush::flags`
的 bit 0，host 侧 `gpu::kGemmFlagInBf16`）：打开后 stage 0 从激活指针按 **bf16** 读
（两个 half 一个 word，低 half 在前，与 `dspark_gemv.slang` 一致），amax / ActQuant /
LDS 布局都不变，默认 0 走原来的 fp32 分支。

- 编译过、过 `spirv-val`；**默认分支未变**，`gpu_layer.mgt1_layer_batch_vs_steps`
  在 l3 仍然逐阶段通过（batch vs steps、gate 35/36、top-k 405/405 全同）。
- **已测**（`mgt1.gemv_bf16_activations_match_the_fp32_path`）：同一组 bf16 舍入后的值，
  bf16 输入与 fp32 输入 **max|d| = 0.000e+00（逐位相同）**；拿未舍入的 fp32 值则差
  **7.659e-02（|y|max 的 2.06%）** —— 前者证明 flag 只改了加载，后者证明它真的在读 bf16。
  第一步失败的原因是测试自己的 rig 少了 `cmd.end()`（`MgtRunner::record` 只 bind，
  `dispatch_now` 用的是 begin/record/end/submit 这一对），错误码是 `vkQueueSubmit2 (-13)`。

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

---

## 6. Track F1（2026-09-18）：并集 MoE、verify 读回、循环本身，以及它们改了什么账

本节是 §2/§3 的执行结果。**三件事落地了，一件没有**，下面按"能跑什么 / 数字 / 还缺什么"写。

### 6.1 §2.2 的缺口关闭了：一次 dispatch 吃整批的 expert 并集（方案 C，没有新 kernel）

§2.2 的三条出路里，方案 C（"kernel 侧 `ids[m][slots]` 的两级索引"）当时被判为"最大工作量"。
**这个判断是错的，而且错在一个具体的地方**：`MoeRunner` 的槽轴本来就不是 7。

* `MoeDims::slots` 决定 `ids` / `slot_list` / `route_weights` / `h` 的大小；
* 两个 shader 都把它当 **push constant**（`GateUpPush::num_slots` / `DownPush::num_slots`）读；
* `route_weights` 已经是稠密的 `[m][slots]`——**这正是并集需要的那张矩阵**。

所以并集不需要两级索引，只需要把槽轴放宽：

```
ids[u]               = 整批第 u 个互不相同的 expert
route_weights[m][u]  = token m 对它的路由权重，没路由到就是 0
最后一个槽            = fp8 shared expert，每列权重 1
```

一次 A + hquant + B 就算完所有列，**每个 expert 的权重整批只读一遍**，而不是"每个路由到它的
列读一遍"。新接口（`runtime/moe_bridge.h`）：

| | |
|---|---|
| `union_experts(call)` | 去重后的 expert 列表——**Planner/ExpertStore 必须先按它 P0 取齐**，与 M=1 的 `stage_rows` 是同一个契约 |
| `stage_batch_union(call)` | 主机侧一次：每列 x + act_quant、并集槽表、`[m][slots]` 权重矩阵、shared 行、每个 routed expert 的指针表行 |
| `run_batch_union(call)` | stage + 一对 dispatch + 等待 + 逐列读回 |
| `record_batch_union(cmd)` | 只记录 dispatch，让 verify 的 MoE 跟着层的 submit 走（M=1 的 `record` 那样） |
| `union_info()` | 列数、并集大小、stage/table/gpu/wall 分项 |

**正确性**（`gpu_moe.the_verify_batch_runs_its_expert_union_once`，层 0，M=6，
对照的是逐列路径——它已经被 `..._takes_one_expert_set_per_column` 钉死为逐位等于 M=1）：

| 路由 | 并集 | 最差列 |
|---|---:|---:|
| 互不相交 | 36 / 36 | **0.000e+00** |
| 有重叠 | 21 / 36 | **0.000e+00** |
| 故意反序（每列同样 6 个 expert，奇数列倒着排） | 6 / 36 | 1.775e-08 of \|y\|max |
| 权重全 0 的一列 vs 只跑 shared expert | — | **0.000e+00** |

读法：**只要并集的槽序与该列自己的顺序一致就逐位相同**——gate 的输出顺序是一致的，所以
真实路由落在这一格；不一致时差的只是 fp32 槽和的结合律（`reordered` 那行是故意造出来的）。
权重 0 的槽贡献**恰好 +0.0**：dispatch A 把 h 乘上权重，而 `moe_hquant` 把块 amax 下限钉在
1e-4，所以全零的 h 量化成零而不是 NaN。

**代价**（`ctest -R bench.mgt1_moe_union_m_curve`，层 0，全部驻留，40 次迭代；
⚠️ 本轮机器上同时有另外四个 worktree 在编译/跑测，绝对毫秒数比 §4 的表高 1.5–1.8×，
**只有同一次运行里背靠背测出来的比值是可信的**）：

| 列的 expert 集 | M | 并集 | 逐列 ms/层 | 并集 ms/层 | 并集 ms/token | 并集/逐列 |
|---|---:|---:|---:|---:|---:|---:|
| 互不相交（最坏） | 6 | 36/36 | 15.04 | 15.36 | 2.560 | 0.98× |
| 有重叠（真实形状） | 2 | 9/12 | 3.80 | 2.80 | 1.400 | **1.36×** |
| 有重叠 | 4 | 15/24 | 8.36 | 6.07 | 1.517 | **1.38×** |
| 有重叠 | 6 | 21/36 | 19.63 | 9.69 | 1.615 | **2.03×** |

并集**正好只花并集那么多钱**：列之间没有共享时它与逐列打平（没有可省的），
有重叠时省下的就是重叠的那部分。这是 MoE(M) **第一次在 M 上次线性**。

### 6.2 §2.4 的缺口关闭了：`dspark_verify.slang`

温度 1 的精确接受要的四个数，现在是一个 stage：`gpu::DsparkStage::VerifyRows`
（每行一个 workgroup，M 个 workgroup，整批一次 mapped 读回，每行 68 个 word——
把整行拷回来是每行 3.1 MB）。抽样用 Gumbel-max，
`argmax_i(l_i/T + g_i)`，`g_i = -log(-log(u_i))`：它**精确等于** `softmax(l/T)`，
限制在一个子集上就精确等于 softmax 限制在那个子集上——所以"屏蔽掉 C_j 的样本"
就是同一次扫描跳过候选，不需要重新归一化。均匀数是 `(seed, row, id)` 的
counter-based hash 而不是流，因为每个线程按自己的顺序走自己那部分行。

判据（`gpu_dspark.verify_rows_give_exact_acceptance_its_four_numbers`，**不需要 checkpoint**）：

* 6 行 × 129,280：候选 logit 与主机镜像 `runtime::emulate_verify_row` **逐位相同**，
  lse 与镜像逐位相同、与 float64 重算差 < 1e-6，**两个采样 token 全部相同**
  （77.6 万个 Gumbel key 里一个近似平局都没有）；
* 屏蔽样本 4,000 次抽样**一次都没抽到候选**；
* 它的分布：total variation **0.1139**，而精确采样器在这个抽样数下的噪声底是 **0.1184**，
  χ² 210.8 / 222 格——与精确采样器不可区分。这一条不依赖镜像是对的。

### 6.3 循环本身：`runtime/speculate.h`

循环与两次 forward 是可分的，这里就在那里切开：`SpecModel` 是循环向模型要的四件事
（`draft_forward` / `verify_forward` / `snapshot_ring` + `restore_ring` / `commit`），
`Speculator::cycle` 是位置对齐、k 规则、选路径、接受、回滚记账。

位置对齐（§3.5，也是最容易错的一段）：周期 `(p0, last_token)` 验证
`[last_token, path[:k]]` 于 p0 .. p0+k，接受长度 a，**发出 a+1 个 token** 于
p0+1 .. p0+a+1（接受的草稿 + 修正 token，修正就是第 a 行自己的答案）；
p0+a+1 .. p0+k 是被拒位置，环槽要恢复——**p0+a+1 也在里面**，因为修正 token 不是草稿的那个，
下一周期的第 0 行会重写那一格。confidence 规则给出 k = 0 时下钉到 1，
所以一个周期至少提交模型本来就会产出的那一个 token。

`ctest -R suite.speculate`（无 GPU、无 checkpoint）：

* `greedy_reproduces_the_unspeculated_stream`：草稿质量（全错 → 全对）× k ∈ {1,2,3,5}，
  20 组配置、每组 64+ token，**每一组发出的 token 流都与无投机逐 token 的流完全相同**，
  接受长度从 0.00 扫到 4.91 / verify、每周期 1.00 → 5.91 个 token。
  这是 design §10.2 的不变式**在循环这一层**的判据（前提是"verify 行的 argmax = 该位置 M=1 的
  argmax"，那是 §14 单独量的批边界效应）；
* `a_rejection_restores_exactly_the_rejected_slots`：每周期一次快照、只有拒绝时才恢复、
  且只恢复被拒位置的槽；
* `confidence_chooses_k_by_the_prefix_rule`：θ = 0.3/0.5/0.7/0.95 → k = 4/3/2/1；
* `a_misconfigured_cycle_refuses`：`--spec off`、K > 64 都是报错而不是静默空转。

### 6.4 还缺的一件：`Engine::forward_batch`（循环跑不起来的唯一原因）

**M>1 的批量前向今天只存在于 DecodeLayer 这一层**，由 `tests/test_gpu_layer.cpp` 从导出的
trace 驱动；Engine 里没有 `gpu::MgtRunner`、没有 batch scratch，40 层从来没有在 Engine 里
串起来跑过。所以 §6.3 的循环有 `SpecModel` 的测试实现，没有真实现，
**§3 第 5 步（贪心投机循环）和 §3 第 3 步都没有完成**。

写 `Engine::forward_batch(p0, tokens[M])` 具体要补的（已逐行核对过）：

1. **Engine 侧的成员**：`gpu::MgtRunner mgt_`、第二个 `gpu::GpuScratch bscratch_`（测试用 64 MB）、
   `layer_.create_batch(mgt_, bscratch_, gpu::kMgtMaxM, max_context)`，
   以及 batch 的 logits `[M][129280]` fp32（M=6 是 3.10 MB）、sample `[M][4]`、
   topk_out `[M][kMgtTopKRecordWords]`（66.6 KB/行）、topk_hist `[M][256][256]`（256 KB/行）。
2. **`BatchStep` 的两个新字段**（M=1 的 `LayerStep` 没有对应物）：
   * `list` —— `BatchScratch::lists` 的下标：窗口层是 0，否则是 `1 + ced_[L].idx_src 在
     index source 里的名次`（就是 `test_gpu_layer.cpp` 的 `ced_src()`）。要在 bring-up 时
     跟 `build_ced_plan()` 一起建一张 `layer -> list` 表；
   * `key_sel` —— 位 mm 表示"查询 mm 自己那一组在这一批里完成了"，即
     `run_compressor && ((p0+mm+1) % ratio) == 0`。它取代了 M=1 的 `cmp_complete`。
   `idx_key_own = view->idx_key`；`idx_key_pub = kvs_.layer(pub_index_k_)->idx_key`，
   而 `pub_index_k_` 要按批的**最后一个**位置更新，并在回滚时恢复。
   `kv` 里唯一要覆盖的平面是 `kv.cmp_kv = kvs_.layer(ced_[L].cmp_src)->cmp_kv`；
   `kv.top_idx` 批路径不读（列表来自 `BatchScratch`）。
3. **每批一次（不是每层）**：`prepare_ced(p0 + M - 1)`，然后
   `write_batch_window_lists(layer_.batch(), c.sliding_window, p0, M)`。
4. **M 个 token 的 embedding**：`Engine::embed_token` 写的是 M=1 的 `DecodeScratch`；
   批路径要把同样的东西写进 `bb.x`（`[M][hc_mult][hidden]`，行距 `hc_mult*hidden` 个 float）
   和 `bb.mix_a`（**每行 128 字节**，清零后第 0 个 float = 1.0）。
5. **MoE**：`gate_ids`/`gate_weights` 是 `[M][16]` 跨距 16，而 `BatchCall::ids/weights` 是
   `[m][topk]` 紧排，要重排；`x = bb.u.host`（`[M][hidden]` 连续）。
   然后 `union_experts` → Planner 取齐 → `stage_batch_union` → `record_batch_union`。
   批路径**没有** `set_moe_output`，所以要么让 bridge 往 `bb.moe_y.host` 写，
   要么每次 `bind_batch` 之后把 `MhcPost/MhcMix/MhcFinal/MhcClose` 的 `slot::kA`
   改指到 `union_y_address()`（注意 `MhcPostB/MhcMixB/MhcFinalB` 的 `kA` 是 `b.wob.addr`，不能动）。
6. **engram 层（1 和 14）在 M>1 上根本没有 runtime 实现**。kernel 侧有
   `MgtStage::EngramGemv/EngramGate` 和 `mgt1_engram.slang`（它要 `[M][24][256]` 的行平面），
   但 C++ 里**从来没有人 bind 或 record 过这两个 stage**；`EngramRunner` 的
   `fetch`/`record`/`Planes` 全是单 token 的，每层只有一份行平面。
   今天唯一可行的做法是**每行各跑一遍 M=1 的 `fetch` + `record`**，x_in/x_out 指到批流的第 m 行；
   真正的批 engram 是 Track T 的活，没做。
7. **`record_tail_batch` 从来没有被调用过**（全仓库只有 docs 提到它）。它自己会跑
   `MhcClose` + `MhcFinal`，所以第 39 层**不要**再调 `record_close_batch`；
   `norm_w`/`head_w` 就是 `Engine` 的那两个。逐行 argmax 从 `sample[m*4]` 读
   （与 `gpu::SampleOut` 同布局）；逐行 top-k 的解析与 `Engine::sample_into` 同形，
   但常数是 mgt1 的（`kHdr=8`、256 线程、`kCap=32`、`kBins=256`），不是 `sample_topk.slang` 的。
8. **回滚**：`KvStore::snapshot_ring(slots, layers)` / `restore_ring`，
   `slots = {(p0+m) % window}`、`layers = 0..39`，M=6 时 40×6×(512+16) B ≈ **127 KB**。
   批的 wkv 尾巴自己写 M 个环槽（`mgt1_gemv.slang:308-325`，被逐出的行只落到
   **临时的** `ovf_val/ovf_scale`，下一层就被覆盖，不是备份）。
   压缩行和组内进位**不用回滚**（§3.5）。主机侧还要自己恢复的：`history_`、`pub_index_k_`、
   `DecodeLayer::cand_position_/cand_n_cmp_`。
9. **CLI**：`--spec {off,greedy,sample} --spec-k` 没有接。`runtime::parse_spec_mode` /
   `SpecConfig` 已经在，`GenerateOptions::speculative` 也已经在，
   但 `cli/` 属于另一条 track 的文件，本轮没有动它，留给合并的人一行 wiring。

### 6.5 把账重算一遍：并集之后，k 应该取多少（go / no-go）

§2.2 当时的结论是"逐列 dispatch 的 verify 比顺序 M=1 慢约 5×"。那个数是**加列掩码之前**的
（MoE M=6 = 11.674 ms/层）。掩码之后是 8.711（`docs/p4_mgt1.md` §4），并集再把它按并集大小缩。
用 §4 的两张表 + 本节的比值，**按 kernel 时间**重算一层：

| | 每层 ms |
|---|---:|
| 6 个 token 顺序走 M=1 | 6 × (1.283 + 1.044) = **13.96** |
| verify M=6，逐列 MoE | 3.384 + 8.711 = **12.10** |
| verify M=6，并集 MoE（实测并集 21.6/36，见下） | 3.384 + 8.711 × 21.6/36 = **8.61** |

也就是说**kernel 时间上 verify 已经便宜了**（并集后 0.62×）。但这不是 decode 的瓶颈：
`docs/p4_mgt1.md` §4 的端到端 A/B 量得很清楚——一个 token 是 **274 ms**，其中
**2,269 MB ÷ ~8.3 GB/s ≈ 273 ms 是 NVMe 读**，kernel 只占 ~2%。所以决定 tok/s 的是
**每个发出的 token 读多少字节**，而字节 ∝ **并集大小**（每个 expert 整批读一遍）。

**u(M) 实测了**（`tools/route_union.py`，`traces/mixed` 的 40 层真实路由，每层每个 M
取至多 20,000 个"同一 prompt 的 M 个连续位置"窗口）。它比 §12.6 那个单点好：

| M | u(M) | u(M) / 6M | 上一版外插 6+4(M−1) |
|---:|---:|---:|---:|
| 2 | **9.88** | 0.824 | 10 |
| 3 | **13.22** | 0.734 | 14 |
| 4 | **16.23** | 0.676 | 18 |
| 5 | **19.02** | 0.634 | 22 |
| 6 | **21.61** | 0.600 | 26 |

层间差别不大（u(6) 从层 25 的 17.3 到层 0 的 28.6，中位数 ~21）。
每周期发出的 token 数用 §12.2 的 **runtime 回放**（树 K=16 `eal`，贪心）
1.83 / 2.50 / 2.85 / 3.33 / 3.33：

| k | M=k+1 | u(M) | E[tokens](k) | expert **unique** / 发出 token | 相对无投机（6） |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 9.88 | 1.83 | 5.40 | 0.900× |
| 2 | 3 | 13.22 | 2.50 | 5.29 | 0.881× |
| 3 | 4 | 16.23 | 2.85 | 5.69 | 0.949× |
| 4 | 5 | 19.02 | 3.33 | 5.71 | 0.952× |
| 5 | 6 | 21.61 | 3.33 | 6.49 | 1.082× |

看起来 k=2 能省 12%。**但这张表数的是 unique，不是 miss，而它们不是一回事——
量完之后，并集省下的 miss 是 0。**

#### 并集相对热缓存的收益：实测 0

无投机的顺序 decode 是**带热缓存**的，而缓存做的事情正是"同一个 expert 在相邻几个
token 里被用第二次 → 命中"——**那正是并集省下的那一部分**。所以两者省的是同一笔钱，
重叠多少必须量。`tools/route_union.py` 在同一条 trace、同一个全局 LRU、同一个容量
（4,500 槽，`docs/p4_hitrate.md` 的 cache-4500 格）、同一个 decode 顺序
（position-major、layer-minor）下走两遍：一遍逐位置取，一遍把 M 个连续位置当一批取并集。

| M | miss / 位置 | 相对 M=1 | hit |
|---:|---:|---:|---:|
| 1 | 25.553 | 1.000 | 0.894 |
| 2 | 25.541 | 1.000 | 0.894 |
| 3 | 25.542 | 1.000 | 0.894 |
| 4 | 25.541 | 1.000 | 0.894 |
| 5 | 25.590 | 1.001 | 0.893 |
| 6 | 25.545 | 1.000 | 0.894 |

**逐位相同到第三位小数。并集没有省下任何一次 LRU 本来就会命中的访问。**
`u(6)/36 = 0.585` 那个"省 41%"是**相对于一个没有缓存的世界**说的；
在有缓存的世界里它恰好是 0。

于是每个**发出的** token 的 miss，落在两个界之间：

| k | M | E[tokens](k) | 最好情况 | 最坏情况 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 1.83 | 1.000 | **1.092** |
| 2 | 3 | 2.50 | 1.000 | 1.200 |
| 3 | 4 | 2.85 | 1.000 | 1.403 |
| 4 | 5 | 3.33 | 1.000 | 1.504 |
| 5 | 6 | 3.33 | 1.000 | **1.801** |

* **最好情况 1.000**：被拒位置上草稿 token 的路由**恰好等于**真 token 的路由，
  于是这些 expert 下一周期被重读时是命中，整个运行取的 expert 集合与顺序 decode 相同。
* **最坏情况 M / E[tokens](k)**：被拒位置的草稿 token 路由到别处，那一位的取数全白费——
  一个周期为 M 个位置付钱，只发出 E 个 token。

**两个界都不是改善。**

#### 结论 / go-no-go

1. **NO-GO，不要把投机设成默认，而且理由比上一版硬**：在这台机器上 decode 的
   99.6% 是 NVMe 读（274 ms 里 273 ms），而投机对 miss 字节的影响区间是
   **[1.00×, 1.09×]（k=1）到 [1.00×, 1.80×]（k=5）**——最好打平，通常变差。
2. 剩下能赚的只有 kernel 那 ~2%：verify 一批 40 层比 6 个 token 顺序走便宜 **0.62×**
   （并集之后，上面第一张表），也就是每 6 个 token 省 ~5.4 ms，在 1,644 ms 里。
   再减去 `T_draft` 的 19–42 ms/周期，**净值为负**。
3. 模型给的 **×1.18–1.32 在这台机器上不会出现**，原因不是接受率
   （接受率就是 §12.2 那些数，已实测），而是**代价模型**：×1.18–1.32 假设 verify 的
   代价随 M 摊薄。在一个 NVMe 受限、带热缓存的 MoE 上，**verify 的代价随位置数长，
   既不随 M 摊薄，也不随并集缩**——缓存已经把并集那部分赚走了。
4. **值得留下的是开关和这张表**，不是默认值。`--spec greedy --spec-k 2` 应该存在，
   因为这个结论完全挂在"decode 是 I/O 受限"这一条上：一旦 R1 的 cache/hit 或容量把
   MB/token 压下来，kernel 那 0.62× 就会浮出水面，那时要重跑 `tools/route_union.py`
   和这一节。**投机在这个系统里是一个等待前提条件的优化，不是一个被证伪的想法。**

### 6.6 下一步，按价值排序

**先说排序本身变了。** §3 那份实施顺序是在"接受率决定成败"的假设下排的；
§6.5 把它推翻了——决定成败的是 I/O 账，而 I/O 账已经算完了，是 NO-GO。
所以第 2、3 项**现在不值得做**，除非第 1 项先变。它们留在这里是因为
一旦前提变了它们就是下一步，不是因为现在该排队。

1. ~~量真实的 u(M)，然后量 miss~~ **两件都做了**（`tools/route_union.py`，§6.5）：
   u(6) = 21.6 而不是外插的 26，而并集相对热缓存省下的 miss 是 **0**。
   go / no-go 因此不再挂在一个未测的数上。
   **接下来唯一能翻盘的是"让 decode 不再 I/O 受限"**——那是 P4-R1（容量 / 命中率 /
   reheat）的事，不是这条 track 的。R1 每把 MB/token 压下去一档，就该重跑
   `tools/route_union.py` 和 §6.5：kernel 那 0.62× 什么时候浮出水面，
   什么时候投机才重新是一个问题。
2. **`Engine::forward_batch`**（§6.4 的 1–8），2–3 天。**前提变了再做。**
   没有它循环跑不起来，§6.5 的账也只能按 kernel 表算而不能实测。
   engram 的 M>1 先走"每行一遍 M=1"。
3. **草稿链进 runtime**（§3 第 6 步），3–5 天，而且要 7.2 GB 的 mtp expert 常驻。
   **前提变了再做**，而且它比 2 更贵：那 7.2 GB 是从 expert cache 里拿走的，
   直接把 §6.5 的 hit 0.894 往下压——投机会先付这笔钱。
4. 采样模式的端到端频率检验（§3 第 7 步）——CPU 半边（`accept_sampling_exact`）和
   GPU 半边（§6.2）都齐了，缺的只是把它们接到 2 上。

**本轮没有测到的东西**（免得下一个人以为测过了）：

* `GpuMoeBridge::record_batch_union`（把并集的 dispatch 记进调用方的命令缓冲，
  给 §6.4 第 5 步用）**没有调用者，因此没有被执行过**。它只是
  `union_runner_.record_into(cmd, Both)`，与 M=1 的 `record` 同一个调用，
  风险低，但它是未测代码。
* `--spec {off,greedy,sample} --spec-k` **没有接进 CLI**：`runtime::parse_spec_mode`、
  `SpecConfig`、`GenerateOptions::speculative` 都在，但 `cli/` 属于另一条 track，
  本轮没有动它。
* §3 第 5 步的判据（同一 prompt、温度 0、`--dspark` 开/关、输出逐 token 相同）
  **在真实模型上没有跑过**——跑过的是 §6.3 的循环层判据。
* L3 8/8、4K/17K 等既有判据在本分支上与 base 相同（投机默认关闭，
  M=1 路径一行没改）。

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
