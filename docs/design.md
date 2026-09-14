# deepMoE 设计方案

> Windows Strix Halo 专用超大 MoE 本地推理 Runtime
> 目标模型：DeepSeek-V4.1-Flash（552B backbone + 196B Engram，decode 激活 16B）
> C++20 · Zig 工具链 · Vulkan Compute (Slang) · AVX-512 · Unified Memory · NVMe Expert Streaming · DSpark 投机解码

**核心目标：** 在一台 Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / 单 NVMe 的机器上，把 DeepSeek-V4.1-Flash 的本地 decode 打到这台机器的物理上限，并且能用数据解释"上限在哪、为什么没到"。

文档状态：**v0.7（2026-09-14/15）**。v0.1 为初始 docx；v0.2 按开发机实测修订内存模型；v0.3 锁定目标模型为 V4.1-Flash，据其真实权重布局重写内存/存储/kernel/prefetch/投机解码设计；v0.4 落地代码骨架，写回 Q6/Q7 实测（§9.2.1），更正 §11.3 的 indexer KV 体积，固定 §14.1 目录；v0.5 取消 repack，改为直读原始 safetensors 分片（§5.1 重写）；v0.6 写回 P1 全部实测（内存系统是单一 ~217 GB/s 共享上限、dispatch 开销低一个数量级、cache 容量受 commit 限额约束、27,399 token 路由 trace、lookahead 降级为不做）；**v0.7 写回 P2 step 1 全部实测：容量问题已解决——扩 pagefile 后 slab 池 100 GiB = 5,711 个 expert 槽、h ≈ 0.92（§5.2/§9.2.2/§3.1）；非 MoE decode 路径九个 kernel 逐级对齐参考实现并跑通整层（§7.15），参考实现强制了八处改动（§2.4/§6/§7.2–§7.7/§7.14）；MoE kernel 的 M 扫描、packed fp16 / int8 dot4、fp8 shared expert 与 `h` 的 fp8 量化（§7.9.2），并**推翻**了两条 v0.6 的结论：§7.1 rule 6 的「x 分块进 LDS」与 §7.9 的「拆分 dispatch 免费」**。修订记录见 [附录 C](#附录-c-修订记录)。

> **本文的实测数字** 一律标注测量日期与来源文件。四份原始报告是
> [docs/kernel_p1.md](kernel_p1.md)（P1 Track A：带宽矩阵、kernel 变体、内存路径、dispatch 开销）、
> [docs/route_trace.md](route_trace.md)（P1 Track B：trace 方法、faithfulness、与参考实现的出入）、
> [docs/kernel_p2_moe.md](kernel_p2_moe.md)（P2 Track D：M 扫描、fp8 shared expert、`h` 量化、分组 dispatch）
> 与 [docs/p2_attention.md](p2_attention.md)（P2 Track E：L2 oracle、九个非 MoE kernel、整层链路）；
> 原始数据在 `bench/results/*.csv` 与 `reports/*.json`。

---

## 目录

1. [项目定义](#1-项目定义)
2. [目标模型解剖：DeepSeek-V4.1-Flash](#2-目标模型解剖deepseek-v41-flash)
3. [核心判断](#3-核心判断)
4. [总体架构](#4-总体架构)
5. [内存与存储布局](#5-内存与存储布局)
6. [精度选择](#6-精度选择)
7. [GPU Kernel 设计](#7-gpu-kernel-设计)
8. [CPU 路径](#8-cpu-路径)
9. [NVMe 冷层与智能 Prefetch](#9-nvme-冷层与智能-prefetch)
10. [DSpark 投机解码](#10-dspark-投机解码)
11. [Prefill、CED 与 KV Cache](#11-prefillced-与-kv-cache)
12. [正确性 Oracle](#12-正确性-oracle)
13. [评测与验收](#13-评测与验收)
14. [工程技术栈](#14-工程技术栈)
15. [开发阶段](#15-开发阶段)
16. [明确不做](#16-明确不做)
- [附录 A：Tensor 清单与字节预算](#附录-atensor-清单与字节预算)
- [附录 B：权重下载](#附录-b权重下载)
- [附录 C：修订记录](#附录-c修订记录)

---

## 1. 项目定义

deepMoE 是一个针对 Windows Strix Halo 的本地 MoE 推理 Runtime。第一阶段聚焦单机、单用户、batch=1 场景，解决的问题是：**一个 510 GB 的模型放不进 128 GB，传统 runtime（mmap + 通用 kernel）在 CPU + GPU + NVMe 三层上都跑不到硬件上限。**

### 1.1 目标硬件（开发机实测）

| 项目 | 规格 | 实测备注 |
|---|---|---|
| APU | AMD Ryzen AI Max+ 395 (Strix Halo) | |
| GPU | Radeon 8060S, gfx1151, 40 CU (RDNA 3.5) | Vulkan 1.4.349, driver 32.0.31041.1004 |
| CPU | Zen 5, 16C/32T, AVX-512 / VNNI | `_mm512_dpbusd_epi32` 已验证可用 |
| 内存 | 128 GB LPDDR5X-8000, 256-bit | 理论 ≈ 256 GB/s，CPU 与 GPU 共用 |
| 内存划分 | BIOS VGM 当前 = 64 GB | **Windows 只看到 63.6 GB**；Vulkan device-local heap 74.4 GiB，host heap 37.2 GiB |
| 存储 | 1 × WD SN740 2 TB NVMe (PCIe 4.0 x4) | C: 383 GB 空闲（**v0.7：其中 96 GiB 已给固定 pagefile**，§5.2），D: 671 GB 空闲；**只有一块盘**，D: 上不需要 pagefile |
| OS | Windows 11 Pro 26200 | |

Vulkan 关键能力（`vulkaninfo` + `tools/envcheck`）：

| 能力 | 值 | 对设计的影响 |
|---|---|---|
| `maxMemoryAllocationSize` / `maxBufferSize` | **2 GiB** | 权重必须分片；expert 级分配，slab 池 |
| `maxStorageBufferRange` | 4 GiB | 单 descriptor 可覆盖一个 2 GiB slab |
| `VK_EXT_external_memory_host` | 支持，对齐 4 KiB | CPU 内存导入 GPU；与 NVMe 直读对齐一致 |
| memoryType `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | 4 种（heap 1） | GPU 内存可被 CPU 直接写入（NVMe 读入落地） |
| `VK_KHR_timeline_semaphore` | 支持 | host-signal 让 GPU 等待 expert 就位，不打断命令流 |
| `VK_KHR_buffer_device_address` | 支持 | expert 指针表，避免海量 descriptor |
| `VK_KHR_shader_integer_dot_product` | 4×int8 packed 加速 | int8 dot4 GEMV 备选路径 |
| `VK_KHR_cooperative_matrix` | 支持 | prefill GEMM |
| subgroup | 默认 64，可控 32–64 | Wave32/64 A/B |
| `shaderFloat16` / `Int8` / `8bit_storage` / `bfloat16` | 支持 | |
| `maxComputeSharedMemorySize` | 32 KiB | attention tile、LUT 放 LDS 的上限 |

### 1.2 第一阶段边界

- 只支持 Windows 11 + Strix Halo。
- 只支持 DeepSeek-V4.1-Flash（文本路径；vision encoder 权重下载但第一阶段不实现）。
- 单用户、batch=1 decode，投机解码的 verify batch ≤ 6。
- 不做 OpenAI Server / Web UI；提供 CLI 与 benchmark harness。
- 正确性优先：所有 kernel 与端到端输出都要过 oracle（§12）。

### 1.3 成功标准

**MVP 成功** 需同时满足：

1. 端到端 greedy 输出与 fp32 oracle 一致（§12），投机解码开关不改变 greedy 输出。
2. decode TPS 达到 §3.1 模型给出的、由实测带宽与实测 hit rate 推出的上限的 ≥ 70%（v0.6：该模型的每一项输入现在都是实测值），并能按 LPDDR / kernel / dispatch / NVMe 四类分解每 token 时间。
3. 在同类可比模型（llama.cpp 能跑的 DeepSeek-V4-Flash GGUF 或 Qwen3-30B-A3B）上，deepMoE 显著快于 llama.cpp Vulkan / HIP。

绝对参照指标：

```
LPDDR 利用率 = hot_bytes_per_token / (T_token − T_nvme_stall) / BW_measured
NVMe 利用率  = miss_bytes_per_token / T_nvme_busy / BW_nvme_measured
```

---

## 2. 目标模型解剖：DeepSeek-V4.1-Flash

来源：ModelScope `deepseek-ai/DeepSeek-V4.1-Flash`（config.json、48 个 safetensors 分片、`inference/model.py`、技术报告）。以下全部数字来自对 safetensors header 的实际统计（附录 A）。

### 2.1 结构要点

| 项目 | 值 |
|---|---|
| 层数 | 40（0–19 causal encoder，20–39 decoder；CED）+ 3 层 DSpark 草稿（`mtp.0–2`） |
| hidden / hc_mult | 5120；残差流为 **4 份并行拷贝**（mHC），即 4 × 5120 |
| MoE | 384 routed + 1 shared，top-6，`sqrtsoftplus` 打分，`noaux_tc` 偏置选路，路由权重归一化 × 1.5 |
| expert 维度 | 5120 → 2304 → 5120，SwiGLU，up 双侧 clamp ±10、gate 上侧 clamp 10 |
| Attention | 64 头 × head_dim 512（含 rope 64），**1 个 KV 头**（MQA over latent 512）；Q 低秩 1280；O 分 8 组低秩 1024 |
| Attention 范围 | 每层 sliding window 128 + 压缩 KV 的 top-512（indexer）；层 0/1 与 mtp 仅 window |
| 压缩 KV 来源 | `kv_source_layer_ids = [2, 8, 14, 20]`，ratio：2–19 层为 2，20–39 层为 1；其他层复用 |
| Indexer | 8 个 index source 层，32 头 × 128，FP4 QK；层 20 生成候选块池（2048 块 × 8） |
| Engram | 层 1、14；每层 384M 行 × 256 fp8 的 n-gram hash 表；每 token 查 3 阶 × 8 头 = 24 行 |
| DSpark | 3 个草稿块（window 128 attention + 128 expert top-3 MoE），一次前向出 5 个草稿位；Markov head + confidence head |
| 词表 | 129,280；embed 与 head 各 1.32 GB bf16 |
| 上下文 | 1M（YaRN），第一阶段目标 64K |

### 2.2 权重体积（510.3 GB）

| 组件 | 大小 | 格式 | 访问模式 | 驻留策略 |
|---|---|---|---|---|
| routed experts（40 层 × 384） | **288.8 GB** | FP4 E2M1 + UE8M0/32 | 每 token 每层随机 6 个 | **Cache + NVMe 流式**（核心问题） |
| Engram 表（2 层） | **203.1 GB** | FP8 E4M3 + UE8M0/32 | 每 token 每层 24 行随机 4 KiB | **永驻 NVMe**，按 token id 确定性预取 |
| DSpark（mtp.0–2） | 7.9 GB | 同上（其中 128×3 个 expert 7.2 GB） | 每个草稿周期 | 全部 pin |
| attention（40 层） | 5.1 GB | FP8 E4M3 + UE8M0（32×32 块） | 每 token 全读 | pin |
| shared experts | 1.4 GB | FP8 | 每 token 全读 | pin |
| embed / head | 1.32 + 1.32 GB | BF16 | embed 查 1 行；head 每 token 全读 | pin |
| vision + aligner | 0.97 GB | BF16 | 不用 | 不加载 |
| router / mHC / norm / engram wkv | ~0.6 GB | BF16 / FP32 / FP8 | 每 token 全读 | pin |

**pin 集合 ≈ 17.7 GB**；其余 ~90 GB 内存做 routed expert cache，只能装下约 **30%** 的 routed experts。

### 2.3 每 token 字节预算（decode，M=1，无投机）

| 项 | 每层 | × 40 |
|---|---|---|
| attention fp8（wq_a 6.6 + wq_b 41.9 + wkv 2.6 + wo_a 33.6 + wo_b 41.9 MB） | 126.6 MB | 5.07 GB |
| shared expert fp8 | 36.5 MB | 1.46 GB |
| routed experts fp4，6 × 18.80 MB | 112.8 MB | **4.51 GB** |
| router bf16 + mHC fp32 | 7.9 MB | 0.31 GB |
| head bf16 | — | 1.32 GB |
| engram wkv fp8 ×2 + 48 行 | — | 0.31 GB |
| **合计** | | **≈ 13.0 GB / token**，其中 routed 4.5 GB 可能来自 NVMe |

单个 routed expert = 3 × 5,898,240 B 权重 + 3 × 368,640 B scale = **18,800,640 B = 4590 × 4 KiB**，天然扇区对齐。

### 2.4 参考实现中必须复刻的细节

从 `inference/model.py` 读出、容易做错的点：

- 残差流是 `[hc=4, 5120]`。每个子层前 `hc_mixes`：把 4×5120 展平做 RMS 归一，乘 `hc_fn [24, 20480]`，经 Sinkhorn（20 轮）得到 `pre[4]`、`post[4]`、`comb[4×4]`。**本子层算出的 pre 给下一个子层用**（attention 用上一层 FFN 的 pre，FFN 用本层 attention 的 pre）。
- **Sinkhorn 的输出不是双随机的**（实测 2026-09-14，`traces/verify/verify.json`，见 route_trace.md §5 检查 4 与 §11.1）。`hc_split_sinkhorn_kernel` 的循环是 `softmax(-1)+eps → /colsum → (iters−1)×(/rowsum → /colsum)`，**以列归一化结尾**：layer 0 真实系数上列和 `|Σ−1| = 1.2e-06`，行和 `|Σ−1| = **8.5e-02**`。参考实现只是收敛到双随机附近，并不落在那里。**§7.2 的 Mega-mHC kernel 绝不能"顺手"在末尾补一次行归一化**，否则输出与参考实现不同。
- **压缩 KV 与 indexer 的激活量化格式不同**（route_trace.md §11.3）：压缩 KV 是 `fp4_act_quant(latent, block=16, scale_dtype=E4M3)`，amax 下限 `6 × 2^-9`；indexer 的 q/k 是 `fp4_act_quant(x, block=32, scale_dtype=E8M0)`，amax 下限 `6 × 2^-126`。§11.3 的容量表（256 B + 32 B）与前者一致。
- `act_quant` 的 scale 是**向上取到 2 的幂**（`fast_round_scale` 用 fp32 位模式算 `2^ceil(log2(amax/448))`，不是浮点 `ceil(log2(x))`——后者在 x 恰好是 2 的幂时会算错），amax 有 1e-4 下限。§6、§7.9 都依赖这个舍入方向。
- Gate：`scores = sqrt(softplus(x_f32 · W))`；选路用 `scores + bias`，权重用未加 bias 的 `scores`，归一化后 × 1.5。图像 token 用 `bias_vl`。
- Expert：`gate = clamp(w1 x, max=10)`，`up = clamp(w3 x, ±10)`，`silu(gate) * up`，乘路由权重后再过 w2；6 个 expert 输出与 shared 输出在 fp32 累加。
- Attention：`q = wq_b(rmsnorm(wq_a x))`，RoPE 只作用最后 64 维；`kv = rmsnorm(wkv x)` 后 RoPE，参考实现写入 window cache 前 **量化为 fp8**；输出 `o` 先做 **逆 RoPE** 再进分组 `wo_a`（8 组各 1024×4096）再 `wo_b`。attention 带 `attn_sink`（每头一个 logit）。
- **`linear()` 在每一次 fp8 / fp4 GEMM 之前都把激活量化成 fp8 E4M3 / block-32 / UE8M0**（v0.7 新增，p2_attention.md §2.1）。
  `inference/model.py` 的 `linear()` 按**权重** dtype 分派，fp8 与 fp4 两支的第一步都是
  `act_quant(x, 32, "ue8m0", e8m0)`。v0.6 只把这件事记在 MoE 的中间激活 `h` 上（§7.9），
  实际上它同样作用于 **`wq_a`、`wq_b`、`wkv`、`wo_b`、`indexer.wq_b`** 与 `shared_experts.w{1,2,3}`。
  E4M3 只有 3 位尾数，一个元素最多动 6%——**跳过它算的是另一个函数**，不是"精度略低"。
  **`wo_a` 是唯一的例外**：它的 module 声明为 bf16，`Attention.forward` 是用 `einsum` 取它的权重
  而不是走 `linear()`，所以 checkpoint 里存的是 fp8 字节、算术却是 bf16 × bf16，**没有这一次往返**。
- **RMSNorm 的输出必须先舍入到 bf16 再进 `act_quant`**（v0.7 新增，p2_attention.md §3）。
  `generate.py` 跑在 `torch.set_default_dtype(torch.bfloat16)` 下，`RMSNorm.forward` 以 `.to(dtype)` 结尾，
  所以量化器看到的是 8 位尾数的值。给量化器喂更宽的数会把块 amax 推进另一个 2 的幂区间，
  **整块 32 个值一起挪一格**。实测 `wq_b`：fp32 进 `act_quant` 相对 L2 1.5e-2 / cos 0.999896，
  bf16 进则 1.8e-3 / 0.999998；`mega_mhc` 的 norm 输出直接从 2.3e-3 变成**逐位相同**。
  **"我们比参考更精确"这条（§6）在量化器正前方是负收益**，残差流仍保持 fp32。
- **RoPE 配对的是相邻元素，不是 (d, d+32)**（v0.7 更正，p2_attention.md §2.2）。
  `apply_rotary_emb` 做的是 `view_as_complex(x.unflatten(-1, (-1, 2)))`，所以在最后
  `rope_head_dim = 64` 维里配对是 **(448 + 2j, 448 + 2j + 1)**，频率 j 驱动第 j 对。
  按旧写法转会转错对——**在 position 0 上看不出来，随上下文增长**。§7.3 已更正。
- **RoPE base 与 YaRN 是按层的，不是全局的**（v0.7 新增，p2_attention.md §2.4）。
  `compress_ratio == 0` 的层（层 0、1 与 DSpark 块）用 `rope_theta` = 10000 且 **YaRN 关闭**
  （`Attention.__init__` 里 `original_seq_len = 0`）；其余层用 `compress_rope_theta` = 160000
  且 YaRN 覆盖 65536。`runtime/rope.h` 因此取 ratio 作参数，而不是读一个全局常量。
- **`hc_post` 收缩的是 comb 的第一个下标**（v0.7 新增，p2_attention.md §2.3）。
  `y = post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)`
  沿 `dim=2` 求和，也就是 comb 的第一个下标，即
  `out[j] = post[j] * a + Σ_i comb[i][j] * residual[i]`——**comb 相对于"顺手的读法"是转置的**。
- Engram：token id 经"压缩词表"映射（NFKC/小写/空白归一）后与前 3 个 token 做乘法-XOR hash，24 个桶各自取模不同素数；查出 24 × 256 fp8 行，拼成 6144 维过 `wkv [25600, 6144]`，得到 4 份 key + 1 份 value，门控 = 归一化点积的符号平方根过 sigmoid。
- DSpark：草稿输入 = [上一个 token, noise_token × 4]，主模型第 37/38/39 层**进入 block 之前**的残差流（engram 之后、`hc_pre` / `attn_norm` **之前**）在 hc 维上的均值拼接（15360 维）经 `main_proj` 成为草稿 attention 的 KV。参考实现是 `if i in target_layer_ids: main_hiddens.append(h.mean(dim=2))`——取的**不是** attention 那个 `[dim]` 输入的均值（route_trace.md §11.5；旧措辞两种读法都读得出来，实现会差很多）；输出 5 个位置的 logits，Markov head 逐位加偏置并采样，confidence head 输出每位接受概率。参考 `generate.py` **未使用**投机路径，验证循环要我们自己实现（§10）。
- 数值：`rms_norm_eps = 1e-20`，gate 归一化加 `1e-20`；hc 相关全部 fp32。

---

## 3. 核心判断

### 3.1 这台机器上 V4.1-Flash 的真实瓶颈是 NVMe，不是 LPDDR

decode 每 token 需读 13.0 GB 权重，其中 8.5 GB 常驻内存，4.5 GB routed expert 有 (1−h) 的比例要从 NVMe 读（h 为 expert cache 命中率）。模型：

```
t_token = 42 ms                      # 常驻 8.5 GB @ 实测 216 GB/s 为 39.4 ms，取 42 ms 保守
        + h × 4.51 GB / 200 GB/s     # 命中读
        + (1−h) × 4.51 GB / 4.5 GB/s # NVMe 读（§9.2.1 实测 4.5–4.75）
```

**v0.6：h 不再是假设。** 下表的 h 来自 27,399 token / 40 prompt（中/英/代码）的真实路由 trace 上
全局 LRU 的**解析解**（重用距离 CDF，Fenwick 树精确计数，一遍扫描给出所有容量），
每 token 时间与 TPS 由上式直接算出。实测 2026-09-14，来源 `reports/cache_sweep.json`
的 `q2_reuse_distance.global_lru_hit_rate_by_capacity`，方法见 [route_trace.md](route_trace.md)。

| cache 容量（slot） | 占 15,360 的比例 | **实测 h（全局 LRU）** | 每 token | TPS | 其中 NVMe stall |
|---|---|---|---|---|---|
| 768 | 5% | 0.5521 | 503 ms | 1.99 | 449 ms |
| 1536 | 10% | 0.6934 | 365 ms | 2.74 | 307 ms |
| 2,056（v0.6 时的真实容量，4 GiB pagefile） | 13% | ≈0.748（内插） | ≈312 ms | ≈3.2（估算） | ≈253 ms |
| 2304 | 15% | 0.7735 | 286 ms | 3.49 | 227 ms |
| 3072 | 20% | 0.8281 | 233 ms | 4.29 | 172 ms |
| 3840 | 25% | 0.8646 | 197 ms | 5.07 | 136 ms |
| 4608 | 30% | 0.8916 | 171 ms | 5.86 | 109 ms |
| 4,787（90 GB slab 池，v0.6 的目标） | 31% | 0.8969 | 166 ms | 6.04 | 103 ms |
| 5376 | 35% | 0.9124 | 150 ms | 6.65 | 88 ms |
| **5,711（今天的真实容量：100 GiB slab 池，§5.2）** | **37%** | **≈0.920（内插）** | **≈143 ms** | **≈7.0** | **≈80 ms** |
| 6144 | 40% | 0.9293 | 134 ms | 7.47 | 71 ms |
| 7680 | 50% | 0.9536 | 110 ms | 9.09 | 47 ms |

（**内部一致性**：把整条 trace 真跑一遍的模拟器给出 0.8266 / 0.8638 / 0.8914 / 0.8968 / 0.9125
（3072 / 3840 / 4608 / 4787 / 5376 槽，`reports/cache_sweep.json` 的 `runs`），
对上面的解析解差 **0.3–0.5 个千分点**。两条完全独立的实现——一条是 Fenwick 树数 distinct key，
一条是真跑一遍 LRU——差别只来自"一层 6 个请求是一批发出的，一个 miss 不会被同批兄弟的 admit
掩盖"。这是整套工具最强的自检，见 route_trace.md §8。
**2,056 与 5,711 两行是估算**：容量按 Q2 曲线内插得 h，再代入上式，没有单独跑模拟器。）

结论（**v0.7 修订**）：

1. **容量问题已经解决，hit rate 不再是"下一步"。** v0.6 的第一优先级是"把 pagefile 调大"；
   它已经做完了（C: 固定 96 GiB，commit 限额 **159.6 GiB**），实测 slab 池
   **100 GiB = 5,711 个 expert 槽 = 15,360 的 37%**，比 v0.6 的目标（4,787 槽 / 31%）还多
   924 个槽（§5.2、§9.2.2）。**代价接近零、收益 ≈3.2 → ≈7.0 tok/s 的那一步已经兑现。**
   往上再要容量就只能换硬件了（§5.2 末）。
2. 常驻部分实测 8.5 GB / 216 GB/s = **39.4 ms**（kernel_p1.md §2.2），比原来按 200 GB/s
   估的 42 ms 略好；表里仍用 42 ms 保守。
   **v0.7 补充**：这一项现在有了逐 kernel 的实测替代品——非 MoE 路径 40 层 + head
   实测 **48.5 ms**（§7.15），MoE 40 层实测 **27.3 ms**（§7.9.2），合计 **75.8 ms**，
   比表里的"42 + 命中读 20.7 = 62.7 ms"**多 13 ms**。差额全部来自 `wo_b` / `wq_b` /
   `sparse_attn` 还没打到内存上限（§7.15 的 57%）。**上表保留模型口径，§13.4 用实测口径。**
3. **投机解码是唯一能把常驻 8.5 GB 摊到多个 token 上的手段**，但它对 NVMe 项**只能部分
   摊薄**：实测相邻 token 的 expert 集合 Jaccard 只有 0.10–0.36（逐层，40 层均值 0.239），
   `union_frac[5] = 0.52–0.82`（均值 0.634）——**一个 k=5 的 verify batch 要碰
   2.6–4.1 倍（均值 3.2 倍）于单 token 的 expert**。也就是说常驻部分被完全摊薄（÷5），
   expert 流量只降到 1/1.6（5 / 3.2）。量化见 §10.1。
4. 第二块 NVMe 直接把 NVMe 项减半，是性价比最高的硬件升级，设计上预留 stripe。
5. **lookahead 预取在这条 trace 上全程净负，已降级为不做**（§9.4）。

### 3.2 Prefill 同样被 NVMe 主导

prompt ≥ 64 token 时，每层几乎所有 384 个 expert 都会被激活。encoder 20 层 × 7.2 GB + decoder（bounded replay 128 token，约 86% expert）≈ **270 GB 流式读取 ≈ 50 s**（cache 命中 30% 时约 38 s）。计算侧 4K prompt 仅需 ~3 s。因此 prefill 必须是 **expert-major 顺序流式**（每层 7.2 GB 连续读满 NVMe 顺序带宽），并且需要 **prefix KV 持久化** 避免多轮重复 prefill。

### 3.3 UMA 的实际形态与两条路径（v0.6 按实测重写）

Windows 上 GPU 可用内存 = BIOS VGM（CPU 不可见）+ WDDM shared（约系统内存一半）。当前 VGM=64 GB：Windows 可见 63.65 GiB，GPU device-local heap 74.4 GiB，GPU host-visible heap 37.2 GiB。

| 路径 | 机制 | expert cache 位置 |
|---|---|---|
| **A** | `DEVICE_LOCAL \| HOST_VISIBLE` 类型分配（memory type 2, heap 1），CPU 映射写入 | GPU heap |
| **B** | `VK_EXT_external_memory_host` 导入 `VirtualAlloc` 的 CPU 内存 | 系统内存 |

**实测结论（2026-09-14，`bench/results/bw_matrix.csv` + `kernel_p1.csv` + `heap_capacity.csv`，详见 [kernel_p1.md](kernel_p1.md) §2–§4）：**

| 维度 | 路径 A | 路径 B | 结论 |
|---|---|---|---|
| GPU raw-read 带宽 | **216.4 GB/s** | **215.2 GB/s** | 差 0.5%，**流式读上完全等价** |
| 纯 `DEVICE_LOCAL`（host 不可见）对照 | 216.0 GB/s | — | **这颗 APU 上不存在"真 VRAM 更快"** |
| **MoE kernel 有效带宽** | **222.1 GB/s** | **195.2 GB/s** | **B 慢 12%**，六轮交替测完全可复现 |
| CPU 写入（NVMe 落地） | 21.5 GB/s（写合并） | 32.7 GB/s（可缓存 + 非临时store） | B 快 1.5× |
| NVMe 直读落地 | 3.7–4.1 GB/s | **4.8 GB/s**（打满盘） | B 快 22% |
| 单次分配 / 导入上限 | 2 GiB（`maxMemoryAllocationSize`） | **> 2 GiB 可用**，实测 3 / 6 / 12 / 24 GiB 导入全部成功 | B 不受 2 GiB 约束 |
| 容量（v0.6，4 GiB pagefile） | **两条路径共用同一个 commit 限额**（§5.2） | 同 | commit 限额是**当时**唯一的约束 |
| **容量（v0.7，96 GiB pagefile）** | **74 GiB**（撞 device-local heap 的 74.4 GiB） | **叠加 26 GiB**（撞可用物理内存下限；单独跑是 40 GiB） | **合计 100 GiB**；commit 已不是约束（§5.2） |

三点必须记住：

1. **raw-read 看不出路径 B 的 12%。** raw-read 每个 workgroup 顺序走一整片连续内存，GART TLB
   命中率接近 100%；MoE kernel 同时在 7 个 expert × 2304 行上推进，每行只有 2560 B，一个
   workgroup 瞬间横跨几十个 4 KiB 页。**导入内存的页表走查成本只在真实访问模式下显形。**
   每 token 的代价 = 40 层 × 0.081 ms = **+3.2 ms/token**。
2. **2 MiB 大页可能抹掉这 12%，但本账户拿不到**（`VirtualAlloc(MEM_LARGE_PAGES)` 需要
   `SeLockMemoryPrivilege`，要 secpol.msc 授予"锁定内存页"并重新登录）。代码里的尝试是完整的，
   失败时回落到 4 KiB 页并把原因记进 CSV。**未测**。
3. **BIOS VGM 不是一个可调的旋钮，也不要再提议调它。** 两个 heap 是同一条 LPDDR5X，
   GPU 在当前设置下已经能寻址两种内存；raw-read 三条路径（A / B / 纯 DEVICE_LOCAL）
   落在同一个 214.9–217.0 GB/s 区间。**调小 VGM 换不来带宽，也换不来容量**。
   v0.3–v0.5 里"在两种 VGM 下各测一遍 / 需要一次重启"的条目全部作废。
   **v0.7 补一句机制上的更正**：commit 限额扩大之后，容量的约束换成了
   "VGM carve-out + 可见物理内存 − 安全余量"（§5.2），VGM 因此**决定 A / B 的划分**
   （路径 A 的前 60 GiB 正好落在 VGM 里、一个字节物理内存都不占，再往上 1:1 吃可见内存），
   但**不改变总量**——总量守恒，所以"调 VGM 换不来容量"这句话仍然成立。
   （"更大的 VGM 能把更多槽挪到快 12% 的路径 A 上"是一个**未测**的推论，
   代价是一次重启与一次全套重测，优先级低于 §15 的任何一项。）

**取舍：热 expert 优先走路径 A（kernel 快 12%），slab 池在当前 BIOS 设置下横跨两种内存**
（§5.2/§5.3）。设计对两条路径都成立：ExpertStore 只暴露 `(host_ptr, device_address)` 对，
`RuntimeConfig::memory_path` 一个字段切换。

### 3.4 dispatch 开销：比估计低一个数量级（v0.6 更正）

每层常驻 + 命中权重 ≈ 210–280 MB，按实测 217 GB/s 为 **≈ 0.97–1.29 ms**（原按 200 GB/s 估 1.0–1.4 ms，吻合）。

**v0.1–v0.5 猜的"每次 dispatch + barrier 约 5–20 µs"高了一个数量级。** 实测
（2026-09-14，`bench/kernel_bench`，一个 command buffer 里连发 1024 个空 dispatch，
每两个之间一个全局 shader-write → shader-read barrier，见 kernel_p1.md §3.4）：

| 指标 | 实测 |
|---|---|
| GPU 侧，每 dispatch + barrier | **0.56–0.66 µs** |
| CPU 侧，录制一对 A+B dispatch（bind + push + dispatch + barrier ×2） | **1.1 µs** |
| CPU 侧，提交 + 等待整个 command buffer | 占 GPU 时间 < 5% |

重算：每 token ~470 个 dispatch × 0.66 µs = **0.31 ms**，占 65 ms 的 **0.5%**，而不是原来担心的 5–14%。后果：

- "每 token 一个预录制的 command buffer"仍然值得做（它同时解决 timeline wait 的表达问题），
  但**不再是性能上的必需品**：即使每 token 重新录制，CPU 侧 470 × 0.55 µs ≈ 0.26 ms 也可接受。
  **v0.7：P2 step 1 实现的是每层一个 command buffer**，因为预录制整个 token 需要
  地址表在 shader 里按层索引（§7.15.3），那是 P3 的改动，按这里的账只值 ~0.5%。
- ~~**§7.9 的"按 expert 拆两组、先到的先算"从"P3 用数据决定"变成默认方案**：翻倍的代价是
  0.66 µs × 2 × 40 层 ≈ 0.05 ms/token，可以忽略~~ —— **v0.7 作废，这句话错了 340 倍。**
  实测拆 3+4 每层多花 **M=1 0.193 ms / M=6 0.108 ms**，40 层就是 **7.7 / 4.3 ms/token**
  （2026-09-14，`bench/results/kernel_p2_moe.csv`，kernel_p2_moe.md §6.2）。
  多出来的**不是启动开销**（那确实是 0.56 µs），而是 dispatch B 的 640 个 workgroup
  每多发一次就要重跑一遍跨 lane 的 LDS 树归约、重读重写 `y`、重新 ramp 起 8 个 wave——
  **这部分工作量与槽数无关、与 M 成正比**。结论改为：**只在真的要等 I/O 时才分组**，
  见 §7.9 的"驻留检查的粒度"。**教训：把一个"每 dispatch 固定开销"的微基准直接外推到
  "多发一次 dispatch"的代价上是错的**，后者还包含被重做的那部分工作。

router 结果仍留在 GPU，用 timeline semaphore 的 host signal 表达"expert 已就位"，CPU 不在层间回读。

---

## 4. 总体架构

```
┌───────────────────────────────────────────────────────────────────────┐
│                            deepMoE Runtime                            │
├───────────────────────────────────────────────────────────────────────┤
│ Model Runtime: tokenizer / engram hash / embed / 40 × Block / head    │
│                DSpark draft + verify loop / sampling                  │
└───────────────┬───────────────────────────────────┬───────────────────┘
                │ expert ids (GPU buffer)           │ token ids
                ▼                                   ▼
   ┌────────────────────────┐          ┌─────────────────────────┐
   │      ExpertPlanner     │          │   EngramPrefetcher      │
   │ residency / lookahead  │          │ hash → row addrs → I/O  │
   │ gating / priority queue│          └─────────────┬───────────┘
   └───────┬───────┬────────┘                        │
           │       │ misses / predictions             │
           │       ▼                                  ▼
           │  ┌────────────────────────────────────────────────┐
           │  │  IoEngine: IOCP + FILE_FLAG_NO_BUFFERING       │
           │  │  priority queue, QD control, direct-to-GPU     │
           │  └────────────────────┬───────────────────────────┘
           │                       │ 4 KiB-aligned DMA
           ▼                       ▼
   ┌──────────────────┐   ┌────────────────────────────────────┐
   │  GPU Executor    │   │  ExpertStore (slab pool ≤ 2 GiB)   │
   │  Vulkan / Slang  │◀──│  pinned | cached | free            │
   │  1 cmd buf/token │   │  (host_ptr, device_addr) per expert│
   └──────────────────┘   └────────────────────────────────────┘
   ┌──────────────────┐   ┌────────────────────────────────────┐
   │  CPU Executor    │   │  NVMe: 48 个原始 safetensors 分片   │
   │  AVX-512 oracle/ │   │  + deepmoe_manifest.json + kvcache/│
   │  fallback / A-B  │   └────────────────────────────────────┘
   └──────────────────┘
   ┌──────────────────────────────────────────────────────────────┐
   │ Profiler: per-token/per-layer bytes, hit/miss, I/O wait,     │
   │           dispatch time, predictor precision, accept length  │
   └──────────────────────────────────────────────────────────────┘
```

### 4.1 模块职责

| 模块 | 职责 |
|---|---|
| Model Runtime | 结构解析、层调度、KV cache、采样；只看到逻辑 expert id |
| ExpertStore | 所有权重块的物理位置、slab 分配、pin / cache / free 三态、`(host_ptr, device_addr)` |
| ExpertPlanner | 命中判断、lookahead 预测、优先级排序、cache 淘汰、向 IoEngine 下单 |
| EngramPrefetcher | token 一旦确定（含草稿 token）即计算 24 个 hash 行地址并下单 |
| IoEngine | Win32 overlapped I/O + IOCP，请求切分、队列深度控制、直接写入 GPU 可见内存 |
| GPU Executor | Slang kernel、command buffer 录制、timeline semaphore |
| CPU Executor | fp32 oracle 前向、AVX-512 GEMV（fallback 与 A/B）、lookahead gate 计算 |
| Profiler | 每 token 时间分解与所有在线统计；输出 JSONL |

---

## 5. 内存与存储布局

### 5.1 NVMe 布局：**不 repack，直接读原始分片**（v0.5 改写）

**决定：磁盘上只增加一个文件 `deepmoe_manifest.json`，它是原始 48 个 safetensors 分片的纯地址簿。** 由 `tools/manifest.py` 生成（替代已删除的 `tools/repack.py`）。

**为什么改**

1. **磁盘放不下第二份。** checkpoint 510.3 GB 在 D:，而 D: 只剩 ~196 GB 空闲。`hot.bin/experts.bin/engram.bin/mtp.bin` 合计仍是 510 GB，物理上不存在。
2. **权重只有一份。** 没有"repack 输出与原始分片不一致"这类只会在 L2/L3 才暴露的 bug 面。
3. **不承担重下风险。** repack 一旦写坏要重跑；真删错了原始分片就是 1 小时重新下载（附录 B）。

**代价（下面量化）：一个 expert 从 1 次读变成 2 次读**，字节数多 8,192 B（+0.044%）。

#### 5.1.1 实测的分片结构（`tools/manifest.py` 扫过全部 48 个 header、96,085 个 tensor）

| 事实 | 值 | 后果 |
|---|---|---|
| 分片数据区起点 `8 + header_len` | 每片不同，`% 4096 ∈ {96, 184, 352, 664, 672, 1056, 2432, 2696, 2768, 2808, 3016, 3240, 3400, 3704}` | **tensor 的绝对偏移是 8 的倍数，永远不是 4096 的倍数** |
| expert tensor 的**长度** | w1/w2/w3.weight 各 5,898,240 B；各 .scale 368,640 B，全是 4096 的倍数 | 只有起点要对齐，长度天然对齐 |
| 一个 expert 的 3 个 `.scale` | 在分片内**连续**，共 1,105,920 B | 合成 1 个 run |
| 一个 expert 的 3 个 `.weight` | 在分片内**连续**，共 17,694,720 B | 合成 1 个 run |
| 一个 expert 是否跨分片 | **0 个**（15,744 个 expert 全部不跨） | 一个 expert 永远只涉及一个文件 |
| 一层的 384 个 expert 是否跨分片 | **0 层**（layer L 全在 `model-{L+3:05d}`，mtp.N 全在 `model-{44+N:05d}`） | §9.7 的 expert-major 顺序流仍然是单文件顺序读 |
| engram 表 | `[rows,256] F8_E4M3` 与 `[rows,8] F8_E8M0` 是**两个独立平面**（分片 47/48），不是交错的 264 B 行 | 取一行 = 2 次 4 KiB 读；一次 4 KiB 覆盖 16 个 value 行或 512 个 scale 行 |

#### 5.1.2 Run 与 skew

FILE_FLAG_NO_BUFFERING 要求 offset / 长度 / 目标指针都是扇区倍数（§9.6）。**对齐纪律整体搬进 manifest，I/O 层一个字节都不用改。** 对每组字节连续的 tensor：

```
aligned_off   = floor(off / 4096) * 4096
aligned_bytes = ceil((off + len) / 4096) * 4096 − aligned_off
skew          = off − aligned_off                    // 0 ≤ skew < 4096
```

这样的一段就是一个 **run**，也就是一个 `IoRequest`。ExpertStore 把一个 expert 的所有 run **首尾相接**填进一个槽，kernel 拿到的每个 part 地址是 `slot_base + run.slot_offset + part.skew`。

实测每个 expert **恰好 2 个 run**（15,744/15,744）：

| run | payload | aligned_bytes | 说明 |
|---|---|---|---|
| scales | 1,105,920 B | **1,110,016 B** | w1/w2/w3.scale |
| weights | 17,694,720 B | **17,698,816 B** | w1/w2/w3.weight |
| 合计 | 18,800,640 B (`kExpertBytes`) | **18,808,832 B (`kExpertSlotBytes`)** | = 4592 × 4 KiB |

`kExpertSlotBytes` 是所有 15,744 个 expert 的最大值，写在 `model/layout.h`，`Manifest::validate()` 与 `tests/test_model.cpp`、`tests/test_integration.cpp` 三处交叉校验；`tools/manifest.py` 的 summary 直接打印这个数。

#### 5.1.3 代价量化（对照 §9.2.1 实测）

| 项 | 数字 |
|---|---|
| 每个 expert 多读的字节 | 8,192 B / 18,800,640 B = **+0.044%** |
| slab 槽变大 | 18,800,640 → 18,808,832 B；100 GiB cache 少装 **约 3 个** expert（5,711 → 5,708，v0.7 实测容量，§5.2） |
| 每个 expert 的 I/O 次数 | 1 → 2 |
| 切成 4 MiB chunk 后的 chunk 数 | 5 → 6（weights run 4+4+4+4+1.88 MiB，scales run 1 个 1.06 MiB） |
| 新增的那个 chunk 大小 | 1.06 MiB，**低于 §9.2.1 的 2 MiB 带宽平台**（1 MiB QD8 随机 = 2.30 GB/s，2 MiB = 4.54 GB/s，4 MiB = 4.68 GB/s） |
| 单个 expert 冷取的最坏情况 | weights 17.70 MB @ 4.68 GB/s = 3.78 ms；scales 1.11 MB 若**单发**则 0.48 ms（而非 0.24 ms）→ 4.26 ms vs 原来 18.36 MiB 单发实测 4.02 ms，**+6%** |
| 实际情况 | P0 一层 6 个 miss = 12 个 request、36 个 chunk 同时在途，聚合 QD ≥ 8，盘仍在 4.5–4.75 GB/s 平台上。§9.2.1 结论 3（≥2 MiB 时随机读≈顺序读）说明**这块盘在 expert 粒度上不在乎局部性**，所以 6 个 miss 的一层仍是 ~23.7 ms，40 层全 miss 仍是 ~0.95 s/token |

结论：**代价在噪声里**。真正的收益（省下不存在的 510 GB、不承担重下风险）是决定性的。若将来 §9.2.1 的小请求带宽塌得更厉害，补救办法是把同一层 6 个 expert 的 scales run 合并下发（它们在分片内并不相邻，所以只能靠 QD 而不能靠合并），或把 scales 常驻内存——15,360 × 1.11 MB = 17.0 GB，超出 pin 预算，**不做**。

#### 5.1.4 EOF 尾部：I/O 层唯一必要的让步

分片的**最后一个** tensor 结束在文件字节长度处，而文件长度不是 4096 的倍数。把它的读扩到扇区边界必然越过 EOF 几百字节。实测有 **43 个 run**（每个含 expert 的分片各一个，即每层的 expert 99 的 weights run）与 **5 个非 expert tensor** 属于这种情况，最多越界 3,336 B。

Windows 对这种读返回 EOF 之前的有效字节并报告"短读"，这是正确行为而不是错误。因此 `ChunkRequest` 增加 `min_bytes`（落在文件内的那部分），三个 backend 与 `IoEngine::finish` 都按它判定短读。**这是 I/O 层为直读付出的全部代价。**

#### 5.1.5 磁盘上的东西

| 路径 | 内容 |
|---|---|
| `model-000NN-of-00048.safetensors` × 48 | 原始权重，510.3 GB，**只读，绝不修改** |
| `model.safetensors.index.json`、`config.json`、`inference/` | 原样保留；`manifest.py --verify` 用 index 校验大小与 sha256 |
| `deepmoe_manifest.json` | 唯一新增文件，9.9 MB（紧凑 JSON）。schema v2 见 `model/manifest.h` 头注释 |
| `kvcache/` | 持久化 prefix KV（§11.4），按 prompt hash 分文件 |

manifest v2 的四张表：

- `files[]`：分片路径、字节数、`data_start`（= `8 + header_len`）。**数组下标就是别处引用的 file id。**
- `tensors{}`：每个非 expert tensor 的 `{file, offset, bytes, dtype, shape, scale{...}}`，绝对偏移，`.scale` 平面折叠进所属 weight 条目。
- `experts[layer][expert]`：run 列表，每个 run `{file, aligned_off, aligned_bytes, slot_offset, parts:[{tensor, skew, bytes, slot_offset}]}`。逻辑层号沿用 `ExpertKey`：0–39 主模型（各 384），40–42 是 DSpark 的 3 块（各 128）。
- `engram[]`：两个平面的基址与行宽 + 行数；384M 行不可能枚举，`Manifest::engram_row(layer, row)` 按算术生成那两个对齐读。

**expert 在槽内的布局不再由我们选择**，它就是分片里的物理顺序（scales run 在前，weights run 在后）。原设计中 `w1/w3 行交错` 的 A/B 方案随之作废——要做这件事必须 repack，而 repack 已被否决。fused gate/up kernel（§7.9）改为靠 `w1`/`w3` 两个 buffer_device_address 并行取址。

### 5.2 内存布局：commit 限额已解除，现在的上限是 VGM + 可见内存（v0.7 按实测重写）

> **一句话**：v0.6 的行动项（把 pagefile 调到 96–128 GB）**已经做完**，
> slab 池从 36 GiB / 2,056 槽变成 **100 GiB / 5,711 槽（15,360 的 37%）**，
> 超过了 v0.6 定的 90 GB / 4,787 槽的目标。**§15 的第一优先级项已关闭。**

#### 5.2.1 v0.6 的状态：commit 限额（历史，保留因为它解释了修法）

`bench/heap_capacity.exe` 逐个分配 2 GiB slab 并**逐页触碰**（WDDM 只有在页被触碰时才决定一次
commitment 真正值多少钱），每个 slab 后读一次 `GlobalMemoryStatusEx`。实测 2026-09-14，
原始数据 `bench/results/heap_capacity.csv`：

| 现象 | 实测 |
|---|---|
| 路径 A 的 slab 对**物理内存**的影响 | **几乎为零**：`availPhys` 从 48.76 GB 掉到 48.44 GB，18 个 slab（36 GiB）一共只动了 0.3 GB |
| 路径 A 的 slab 对 **commit charge** 的影响 | **1:1**：`availCommit` 每个 slab 掉 2.02 GiB，45.02 → 8.14 GiB |
| 路径 B 的 slab | 物理与 commit **都是 1:1**（`availPhys` 46.58 → 9.83 GB） |
| 本机 commit 限额 | **67.65 GiB = 63.65 GiB RAM + 4 GiB pagefile**（系统托管的默认 pagefile） |
| 路径 A 停在哪里 | **36 GiB**（18 个 slab），原因 `available commit 7.58 GiB < slab 2.00 + floor 6.00 GiB` |
| 先 A 后 B 的混合分配 | A 拿到 36 GiB 之后，**B 一个 slab 都拿不到**（同样的 commit 判据） |
| **合计可用 cache** | **36 GiB = 2,056 个 expert 槽 = 15,360 的 13.4%** |
| 分配后的 raw-read 验证 | 路径 A 216.0、路径 B 208.5、混合 211.1 GB/s——**最后那批字节是真的** |

**结论：cache 容量既不受 BIOS VGM 约束，也不受两个 heap 的大小约束，而是受
`RAM + pagefile` 这一个数约束。** 路径 A 的 VGM carve-out 对 Windows 的物理内存是隐形的，
但它**照样按 1:1 吃 commit**。于是今天这台机器上：

```
commit 限额 67.65 GiB − OS 与其他进程 − 6 GiB 安全余量  →  36 GiB slab  →  2,056 个 expert
```

**修复办法：把 pagefile 调大。** 这是纯粹的 commit 记账——**VGM 支撑的页永远不会被写进
pagefile**（它们不在物理内存的分页池里），所以这不会带来任何换页 I/O，只是把限额抬高。
操作步骤见 [build.md](build.md#windows-虚拟内存pagefile已在开发机上做完)。

#### 5.2.2 v0.7 实测：pagefile 调大之后（2026-09-14/15）

开发机的 C: 已设成**固定 98,304 MB = 96 GiB** 的 pagefile，
commit 限额从 67.65 GiB 变成 **159.6 GiB**（= 63.65 GiB RAM + 96 GiB pagefile）。
重测 `bench/heap_capacity.exe --slab-gib 2 --min-free-gib 6`，
原始数据 **`bench/results/heap_capacity_idle.csv`**（空闲机）与
`bench/results/heap_capacity_pagefile128.csv`（同一设置，但有并发基准，见下面的量测卫生说明）：

| | 路径 A 单独 | 路径 B 单独 | **先 A 后 B（混合，实际布局）** |
|---|---|---|---|
| 拿到的 slab | 37 × 2 GiB = **74 GiB** | 20 × 2 GiB = 40 GiB | A **74 GiB** + B **26 GiB** = **100 GiB** |
| 停下的原因 | `vkAllocateMemory(2 GiB, type 2)` 返回 **`-2` (OUT_OF_DEVICE_MEMORY)** | `availPhys 6.68 GiB < slab 2 + floor 6` | A 撞 heap 上限，B 撞物理内存下限 |
| 停下时还剩多少 commit | **61.8 GiB** | 96.2 GiB | **35.6 GiB** |
| 满载后 GPU raw-read | 214.4 GB/s | 206.4 GB/s | **216.1（A）/ 207.5（B）GB/s** |
| **折成 expert 槽** | | | **5,711 个**（按 `kExpertBytes`；按 `kExpertSlotBytes` 是 5,708），**= 15,360 的 37%** |

**三条新事实**：

1. **commit 不再是约束。** 混合分配拿满 100 GiB 之后还剩 35.6 GiB commit。
   约束换成了两个物理上限：路径 A 撞 **device-local heap 的 74.4 GiB**
   （37 × 2 GiB = 74 GiB，只差最后半个 slab），路径 B 撞**可用物理内存的 6 GiB 安全下限**。
2. **路径 A 的前 60 GiB 一个字节物理内存都不占，第 61 GiB 起 1:1 吃可见内存。**
   `availPhys` 在前 30 个 slab 上只掉 0.5 GB，从第 31 个 slab 起每个掉 2.02 GiB。
   30 × 2 GiB = 60 GiB **正好是 BIOS VGM 的 64 GB = 59.6 GiB**。
   这解释了为什么混合分配里路径 B 只剩 26 GiB 而单独跑有 40 GiB：路径 A 越过 VGM 之后
   吃掉的就是路径 B 要用的那块可见内存。**总量 = VGM + (可见物理内存 − 安全余量)，守恒**（§3.3 第 3 点）。
3. **装满 100 GiB 之后带宽没有任何惩罚**：A 216.1、B 207.5 GB/s，与空盘时的
   216–218 GB/s 上限（kernel_p1.md §2.2）在漂移范围内。**最后那批字节是真的，而且是满速的。**

> **量测卫生（一条要记住的教训）**：`heap_capacity_pagefile128.csv` 是同一套设置下先跑的一轮，
> 满载 raw-read 只有 **151.9（A）/ 121.4（B）GB/s**，看起来像"装满之后带宽掉一半"。
> **那不是容量的代价，是争用**——那一轮有另一个基准在并发跑。空闲机重测（`heap_capacity_idle.csv`）
> 给出 216.1 / 207.5。build.md 的"测量纪律"一节说的就是这件事，**容量测量同样适用**：
> 一个会让你改设计的负面结果，先确认机器是空的。

#### 5.2.3 今天的布局

| 区域 | 位置 | 大小 |
|---|---|---|
| pinned 权重（hot + mtp + embed） | GPU heap（路径 A） | 17.7 GB |
| routed expert cache | **slab 池，横跨两种内存**（路径 A 优先，溢出走路径 B） | **100 GiB（5,711 个 expert，37%）** |
| KV cache（64K ctx） | GPU heap | < 0.5 GB |
| I/O staging、engram 行、hash 表、OS | 系统内存 | 其余 |

**slab 池在当前 BIOS 设置下就横跨两种内存**，不需要重启、不需要改 VGM：路径 A 的 slab 在
kernel 上快 12%（§3.3），所以**热 expert 优先落在路径 A 的 slab 上**，路径 A 用满之后
继续用路径 B 的 slab；两者对 ExpertStore 是同一种 `(host_ptr, device_address)`。
**注意 pinned 的 17.7 GB 也要从这 100 GiB 里出**（它同样占 heap 与可见内存），
所以 ExpertStore 启动时实测到的槽数会比 5,711 少——§9.8 要求报告里必须带上实测槽数，
原因就在这里。

**再要更多容量只能换硬件。** 两个约束都已经顶到物理量：heap 是 VGM 决定的，
可见内存是 128 GB 这颗料决定的。**不要再提出"调 pagefile"类的软办法**，它已经用完了。

### 5.3 Slab 池

- slab = 1.88 GB = 100 个 expert 槽；**~57 个 slab**（v0.7：100 GiB 池，§5.2）。**2 GiB 上限只约束路径 A**
  （`maxMemoryAllocationSize`，`SlabConfig::slots_per_slab` 的硬上界因此是
  `2 GiB / 18,808,832 = 114`，默认 100 安全）。**路径 B 的导入不受这个限制**：
  实测 `external_memory_host` 导入 3 / 6 / 12 / 24 GiB 全部成功
  （`memory=ok buffer=ok addr=ok`，`bench/results/heap_capacity.csv` 的 `oversize_import` 段，
  2026-09-14；48 GiB 那次失败是撞上物理内存下限，不是 API 拒绝）。
  **路径 B 的 slab 因此不必切成 2 GiB**——少一些 slab 就少一些 GART 表项，正是 §3.3 第 1 点
  说的那个成本来源。当前实现仍统一用 2 GiB，改大是 P2 的一个低成本实验。
- 每个槽的 `(slab_id, slot)` → `device_address = slab_base + slot × 18,800,640`。
- GPU 侧 **expert 指针表** `uint64 addr[40][384]`（123 KB，host-coherent）；未驻留为 0。kernel 用 `buffer_device_address` 直接取址，不需要 per-expert descriptor。
- 槽状态机：`Free → Filling(I/O in flight) → Resident → (Evictable)`；驻留槽被淘汰前必须确认没有 in-flight command buffer 引用它（用 timeline 值做 generation 校验）。

### 5.4 数据模型

```cpp
enum class Tier : uint8_t { Pinned, Cached, Cold };
enum class SlotState : uint8_t { Free, Filling, Resident };

struct ExpertKey { uint16_t layer; uint16_t expert; };   // layer 40..42 = mtp

struct ExpertSlot {
    ExpertKey    key;
    SlotState    state;
    uint16_t     slab;      uint32_t slot;
    void*        host_ptr;                 // 路径 A: CPU 映射的 device 内存(只写)
    VkDeviceAddress dev_addr;
    uint64_t     last_use_token;           // LRU
    float        heat;                     // EWMA of router score (含未选中的近似命中)
    uint64_t     guard_timeline;           // 淘汰前需 GPU timeline ≥ 此值
};

struct IoRequest {
    ExpertKey key;  uint8_t priority;      // 0=当前层缺失 1=lookahead 2=engram 3=后台回填
    uint64_t file_off, bytes;  void* dst;  // dst 4 KiB 对齐
    uint64_t issue_token, deadline_layer;
};
```

---

## 6. 精度选择

**原则：使用 checkpoint 自带的精度，一位不改。** V4.1-Flash 的 FP4 expert 与 FP8 attention 是 QAT 训练出来的，没有"再量化"的空间；GGUF 对 V4.1 不存在，自造量化没有 reference 可对。

| 部分 | 存储精度 | 计算精度 | 说明 |
|---|---|---|---|
| routed / mtp experts | FP4 E2M1，UE8M0 scale / 32 元素（按 K） | 解码到 fp16，fp32 累加；备选 int8 dot4 | 已是最小精度 |
| attention / shared / engram wkv / main_proj | FP8 E4M3，UE8M0 scale / 32×32 块 | 解码到 fp16，fp32 累加 | |
| embed / head / router W | BF16 | fp32 | head 1.32 GB 占每 token 10%，**保持 bf16**；int8 head 仅作为标注清楚的 A/B 实验 |
| mHC、gate bias、sink、scale 向量 | FP32 | fp32 | |
| 激活 | — | fp16 / fp32 传递，fp32 归约 | 一般情况下我们比参考实现精度更高，oracle 用容差而非逐位。**两类例外见下面的"激活量化清单"（v0.7）**，它们不是精度选择而是复刻要求 |
| `act_quant` 的 scale | — | UE8M0，**向上取到 2 的幂** | `fast_round_scale` 用 fp32 位模式算 `2^ceil(log2(amax/448))`；amax 有 1e-4 下限。浮点写法的 `ceil(log2(x))` 在 x 恰好是 2 的幂时会算错（§2.4） |
| SWA KV cache | fp8 E4M3 块量化（同参考） | | 保持与参考一致，便于逐层对齐 |
| 压缩 KV | oracle 模式 fp8（同 `model.py`）；生产模式 FP4 E2M1 + E4M3/16（同官方部署） | | 两种都实现，后者需通过 §12 容差 |

**激活量化清单（v0.7 新增，p2_attention.md §2.1 / §3；§2.4 有完整论证）**

`inference/model.py` 的 `linear()` 按**权重** dtype 分派，凡是 fp8 或 fp4 权重，
第一步都是 `act_quant(x, 32, "ue8m0", e8m0)`。v0.6 只记了 MoE 的 `h`，**清单其实是这样的**：

| 权重 | 存储 | 进 GEMM 前量化激活？ |
|---|---|---|
| `wq_a`、`wq_b`、`wkv`、`wo_b`、`indexer.wq_b`、`shared_experts.w{1,2,3}` | FP8 E4M3 + 32×32 UE8M0 | **是** |
| routed experts 的 `w1`/`w3`（输入 `x`）与 `w2`（输入 `h = silu(gate)*up`） | FP4 E2M1 + UE8M0/32 | **是**（§7.9 v0.6 记的就是后者） |
| **`wo_a`** | FP8 E4M3 + 32×32 UE8M0 | **否**——module 声明 bf16，`Attention.forward` 走 `einsum` 不走 `linear()`，算术是 bf16 × bf16 |
| `compressor.*`、`indexer.{wk,k_norm,weights_proj}`、`gate.weight`、所有 norm | BF16 | 否 |
| `head` | BF16，提升到 fp32 | 否 |

**代价是零**：`LanesPerRow = 32` 时一个 lane 正好持有一个完整的 32 元素 UE8M0 块，
块 amax 就是它已经加载的值上的一次寄存器归约。

**配套的一条**：**每个 RMSNorm 的输出必须先舍入到 bf16 再交给 `act_quant`**。
参考的 `RMSNorm.forward` 以 `.to(dtype)` 结尾，而 `generate.py` 的默认 dtype 是 bf16；
喂给量化器更宽的值会把块 amax 推进另一个 2 的幂区间，整块 32 个值一起挪一格
（`wq_b` 相对 L2 1.5e-2 → **1.8e-3**，`mega_mhc` 的 norm 输出 2.3e-3 → **逐位相同**）。
**残差流仍然是 fp32**——那里没有下游量化器，多出来的位不会伤人。
实现是 `gpu/shaders/attn_common.slang` 的 `bf16_round`。

int8 dot4 备选路径：E2M1 的取值 {0, .5, 1, 1.5, 2, 3, 4, 6} × 2 全是整数，权重可无损映射为 int8 {0,1,2,3,4,6,8,12}；激活按 32 元素块对称量化到 int8。这与参考的 fp8 激活精度同量级但不相同，属于 oracle 容差内的实现选择，用于 M≥4 时 ALU 压力（§7.6）。**v0.7 实测：它只在 dispatch B 上划算，而且精度刚好超出 §12 的 5e-3 判据（5.4e-3），见 §7.9.2。**

---

## 7. GPU Kernel 设计

### 7.1 通用规则

1. **每 token 一个 command buffer**，40 层 × ~14–15 dispatch 预录制（v0.7 更正，原写 ~11，
   §7.14），expert 地址来自 GPU 侧指针表。CPU 只在 MoE 层前通过 timeline semaphore
   host-signal "该层 6 个 expert 已驻留"。
   **v0.7 实现状态：P2 step 1 录的是每层一个 command buffer，不是每 token 一个。**
   拦路的是地址表——一个 stage 拥有表里的一个 slice，所以预录制整个 token 需要
   **在 shader 里按层索引地址表**（expert 指针表已经是这么做的）。这是 P3 的改动，
   按 §3.4 的账值 ~0.5%，等需要它来表达 timeline wait 的流式 runtime 时一起做（§7.15.3）。
2. **所有 GEMV 以有效 GB/s 为唯一指标**。~~raw-read 作为上限~~ —— **v0.6 作废**：实测最好的
   MoE kernel 是 218.5 GB/s = raw-read 的 **102%**（kernel_p1.md §3.2），因为它同时有
   w1 / w3 / scale 三条独立地址流，在途请求比 raw-read 多。**"216–218 GB/s 是这颗 APU 的
   LPDDR 流式读上限"才是那个上限**，raw-read shader 只是它的一个下界估计。raw-read 还有一个
   盲区：它对两条内存路径给出一样的数字，而 MoE kernel 在路径 B 上慢 12%（§3.3）。
3. **权重读法：一个 wave 协同读一行**。FP4 一行 5120 元素 = 2560 B；scale 160 B 随行读入。
   实测 `LanesPerRow ∈ {16, 32, 64}` 之间只差 4%，**32 最好**（M=1）；v0.3 猜的
   "16 lane 一行、读粒度 256 B 更好"没有兑现——在 220 GB/s 这个水平上读粒度已不是限制。
   **M=6 时反过来，16 最好**（§7.9）。
4. **Wave32 作为默认，但没有数据支持"优先"**：同变体下 Wave32/Wave64 互有胜负，差 1–5%，
   无系统性差异（kernel_p1.md §3.2）。保留 32 的唯一理由是它让 `LanesPerRow=32` 的行组
   正好落在一个 subgroup 内。
5. **解码必须用常量表**：FP8 E4M3 → fp16 用 256 项 LDS 表（512 B）；**FP4 用
   `static const float kE2M1[16]` 常量数组下标**。实测这是整个 sweep 里唯一"必须选对"的旋钮：
   常量表 218.5 GB/s > 位运算算术构造 164.2（77%）> 显式 select 树 151.4 GB/s（69%），
   **同样的字节数差 1.4 倍**。直觉（"查表要访存、算术更快"）与实测完全相反，AMD 的着色器
   编译器把 16 项常量数组变成了比 5–7 条 ALU 指令更便宜的东西。**这条只能靠实测。**
   E8M0 scale 用 `ldexp(acc_block, e − 127)`，即整数加到指数位。
6. **激活放 LDS：只在它能省下指令的地方，绝不为了省字节而加 per-K-chunk barrier。**
   （**v0.7 整条重写**；v0.6 写的是"M=6 时 x 要按 K 分块进 LDS，这是 P2 的头号 kernel 待办"，
   **实测净负收益，那句话已作废**，见 §7.9.2。）
   - `h`（2304 个元素 = 4.5 KiB）整块进 LDS：一次搬入、被 5120 行复用，**没有 per-chunk barrier**，
     这是正确的用法。
   - x 在 M=1 时不必进 LDS（每 lane 每块只读 4 个 `uint4`，L0 完全吸收）。
   - **x 按 K 分块进 LDS 在每一个 M 上都更慢**（M=6 时 A+B 从 119.0 掉到 95.5 GB/s，
     dispatch B 掉 27%）。原因不是 LDS 慢：`ds_read_b128` 和 `global_load_dwordx4`
     都是一条指令读 16 B，**分块换不来指令，只换来每个 K-chunk 两个
     `GroupMemoryBarrierWithGroupSync()`**，于是权重那条 DRAM 读的延迟
     从"被下一块的计算掩盖"变成"每 chunk 暴露一次"。
   - **唯一的例外是 `xldsi8`（LDS 分块 + int8 dot4）在 dispatch B 上**——它之所以赢，
     是因为分块让**一次激活量化被 16 个 row group 共用**，省下的指令盖过了 barrier。
     **判据因此是"这次分块净省了多少条指令"，不是"省了多少字节"。**
   - 同一条规则在 attention 路径上的表现：fp8 GEMV 把量化后的 x 暂存进 LDS 时
     **存 bf16 值而不是 E4M3 字节**（§7.15.2）。字节省一半，却要每个元素多一次 LDS 访问
     （一次取字节所在的 word、一次查解码表），而 LDS 端口正是这些 kernel 用光的东西。
7. **M ≤ 6 时仍按 GEMV 结构**，每 lane 持有 M 个累加器，权重读一次复用 M 次；M > 16（prefill）切换到 cooperative matrix GEMM。
   **v0.7：M ≥ 3 时 kernel 已经是 VALU 发射受限而不是访存受限**（dispatch A 的时间是
   `0.33 + 0.055·M` ms 的直线，§7.9.2），所以那里的衡量指标要换成 `ms/token`，
   省指令的设计（packed fp16、int8 dot4）有用，省访存的设计全都没用。
8. 每个 kernel 是 Slang 泛型：`Gemv<WeightFmt, M, LanesPerRow>`，`WeightFmt ∈ {Fp4E2M1, Fp8E4M3, Bf16}`。
9. **三条"和带宽无关、但能白掉 50 倍"的 Slang 陷阱**（v0.7 新增，p2_attention.md §4.4，
   都是 `bench/attn_bench` 抓到的，每一条都值一条规则）：
   - **被 push constant 作上界的循环里的局部数组会落到 VRAM。** Slang 展不开循环，
     数组就不再可提升为寄存器，直接进 scratch memory。`mega_mhc` 的 4×4 Sinkhorn 矩阵
     （16 个 float 的工作集！）因此**每 dispatch 花 216 µs**；改用编译期常量 `kMaxHc`
     作上界之后是 **2.8 µs**。gate 的 rank 循环同一个形状：55 µs → 2 µs。
     **凡是循环上界来自 push constant 而循环体索引局部数组，就要检查一遍。**
   - **单 lane 里的串行扫描就是单 lane 里的串行扫描。** gate 的 top-k 原来每个 rank
     走 384 个 expert、6144 次依赖比较，**56 µs 一层 = 2.2 ms 一个 token**，只为从 384 个数里挑 6 个。
     改成 16 轮 LDS 树上的 workgroup argmax：**12.6 µs**。**任何 argmax / top-k 都不许写成串行扫描。**
   - 激活进 LDS 存 bf16，不存 E4M3 字节（见 rule 6 最后一条）。

### 7.2 Mega-mHC（`hc_mixes + hc_pre + RMSNorm`）

- 输入：残差流 `x[4][5120]` fp32（80 KiB），`hc_fn[24][20480]` fp32（1.97 MB），`hc_base[24]`，`hc_scale[3]`，前一子层的 `pre[4]`，norm 权重。
- 输出：本子层输入 `u[5120]` fp16，以及供下一子层用的 `pre'[4], post[4], comb[4][4]`。
- 访存：`hc_fn` 1.97 MB 是主流量；24 个输出各是一个 20480 维点积 → 24 个 wave 各算一个，然后 Sinkhorn 20 轮在一个 wave 内做 4×4 矩阵行列归一（纯寄存器）。
- 计算注意点：整流 rsqrt 在整个 20480 维上做一次；Sinkhorn 与 comb 严格 fp32；`pre` 加权求和后再 RMSNorm。
- ~~融合：1 个 dispatch~~ —— **v0.7 更正：是 3 个 dispatch，做不到 1 个**（p2_attention.md §4.1）。
  两个理由都是硬的：(a) `hc_fn` 是 1.97 MB，**一个 workgroup 只能以一个 CU 的速率读它**；
  (b) 两个 RMS 统计量都是整向量归约，需要 grid-wide barrier，而 grid-wide barrier 就是 dispatch 边界。
  实际的切法是：
  **stage 0**（`dim/256` 个 workgroup）做 `hc_post`、`hc_pre` 与两个平方和的部分和 →
  **stage 1**（24 个 workgroup）各算一个 20480 维点积 →
  **stage 2**（回到 `dim/256` 个）做 RMSNorm，Sinkhorn 搭在 workgroup 0 上。
  按 §3.4 实测的 0.66 µs/dispatch，多出来的两个 dispatch 是 **1.3 µs 一层**，买得起。
- **每层跑 3 次 mega_mhc，不是 2 次**：attention 半、FFN 半，以及**收尾 block 的那次 `hc_post`**，
  三次的权重和缓冲都不同。**一个 stage 只拥有地址表的一个 slice**，所以同一个 command buffer 里
  两次同 stage 的 dispatch 会都看到最后写进去的那份地址——FFN 半与收尾各有自己的 stage。
  这是 §7.14 的 dispatch 数从 11 涨到 14–15 的主要来源。
- 输出 `attn_norm` 与参考实现**在七个层上全部逐位相同**（§7.15），前提是 rule "RMSNorm 输出先舍入 bf16"（§6）。

### 7.3 Attention Q 路径（`wq_a → q_norm → wq_b → RoPE`）

- `wq_a`: fp8 [1280 × 5120]，6.6 MB → `qr[1280]`；`q_norm`（RMS over 1280）；`wq_b`: fp8 [32768 × 1280]，41.9 MB → `q[64][512]`；RoPE 作用每头最后 64 维。
- 2 个 dispatch（`wq_b` 需要完整的 `qr`）。`q_norm` 融入第二个 kernel 的输入加载：每 workgroup 先在 LDS 上算 `rsqrt(mean(qr²))` 再读 `qr`。
- fp8 块 scale 索引：`scale[row/32][k/32]`，一行的 K 方向 40 个块 → 每 32 个 K 元素一次 `ldexp`。
- **两个 GEMV 的输入都要按 §6 的清单做 fp8 E4M3 / block-32 激活量化**（v0.7），
  并且送进量化器的 `qr` 必须先舍入到 bf16。
- RoPE 融合进 `wq_b` 的写出：输出行 `h*512 + d`。
  ~~`d ≥ 448` 的行成对（d, d+32）旋转~~ —— **v0.7 更正：配对的是相邻元素
  `(448 + 2j, 448 + 2j + 1)`，频率 j 驱动第 j 对**（p2_attention.md §2.2）。
  `apply_rotary_emb` 做的是 `view_as_complex(x.unflatten(-1, (-1, 2)))`。
  按旧写法转会转错对，**position 0 上看不出来，误差随上下文增长**。
  让同一 lane 负责一对**相邻**输出元素。
- **RoPE 的 base 与 YaRN 按层取**（§2.4）：`compress_ratio == 0` 的层用 `rope_theta` = 10000 且 YaRN 关闭，
  其余层用 `compress_rope_theta` = 160000 且 YaRN 覆盖 65536。`runtime/rope.h` 的入参是 ratio。

### 7.4 KV 路径与 Compressor / Indexer

- `wkv`: fp8 [512 × 5120]，2.6 MB → `kv[512]`，`kv_norm`，RoPE 后 64 维，fp8 块量化写入该层 window cache（环形 128 槽 × 512 B + scale）。1 个 dispatch，极小。
- kv source 层（2, 8, 14, 20）额外：Compressor 每 `ratio` 个 token 输出一个压缩 latent（`wkv_c [512 × 5120]` fp32 + softmax 门控累积到状态），写入该源层的压缩 KV cache；索引 K（fp4，128 维）由 latent 派生。
- index source 层（8 层）：`indexer.wq` 把 `qr` 投到 32 头 × 128（fp4 激活量化），对压缩位置的 K 打分（`ReLU` 后按头加权求和），取 top-512。**这是唯一随上下文长度增长的 decode kernel**：64K 上下文、ratio 2 → 32K 个位置 × 32 头 × 128 = 134 MFLOP，K 数据 32K × 64 B = 2 MB，带宽和算力都很小；top-512 选择用两级 radix select（先每 workgroup 局部 top-512，再合并），层 20 另输出候选块池供 24–36 层限定范围。
- 这些每层只有一两个的 kernel 允许各自独立 dispatch；正确性优先。

### 7.5 稀疏 Attention（decode，单 query）

- 输入：`q[64][512]` fp16，KV 条目 = window 128（fp8）+ 压缩 top-512（fp8 或 fp4）= 640 条 × 512，`attn_sink[64]`，索引数组。
- 数据量：640 × 512 B = 320 KiB；算力 64 × 640 × 512 × 2 ≈ 42 MFLOP。**纯延迟型 kernel**，目标是 < 30 µs。
- 组织：MQA 意味着 64 个头共享同一 KV，所以 **一个 workgroup 处理全部 64 头**，KV 分 tile 进 LDS（32 条 × 512 B = 16 KiB 一个 tile，双缓冲 32 KiB），每个 tile 对 64 头做 QKᵀ、~~在线 softmax（含 sink 项作为初始 max/sum）~~、PV 累加。workgroup 256 线程 = 4 wave；每 wave 16 头，lane 负责 1 头的 1/4 维度（128 维 = 64 个 packed fp16 寄存器），跨 4 lane 用 subgroup shuffle 归约点积。
- **在线 softmax 不行，必须两遍（v0.7 更正，p2_attention.md §4.2）**：
  `sparse_attn_kernel` 把概率舍入到 bf16（`acc_s_cast`）是在**减去最终那一行的 max 之后**，
  而且那个 max **只取 KV 的分数**——每头的 `attn_sink` 以 `exp(sink − mx)` 进分母，
  但**不参与竞争 max**。在线 softmax 会拿一个滚动的 max 去舍入 `p`，**算出来的是另一个数**。
  复刻它的代价是多扫一遍 320 KiB 的 KV，换成 **2 个 dispatch**（score / combine）。
  滚动 max 的初值也要用有限的 −1e30 而不是 −∞，否则整行索引全是 −1 的 query 会出 NaN 而不是 0。
- 融合：q 的 RoPE 已在上游做；输出 `o[64][512]` 的 **逆 RoPE** 融合在写出阶段。
- 输出直接以 8 组 × 4096 的布局写出，供分组 `wo_a`。

### 7.6 输出投影（分组 `wo_a` → `wo_b`）

- `wo_a`: fp8，块对角 8 组，每组 [1024 × 4096]，33.6 MB；输入 `o` 按组切 4096 维。一个 dispatch，workgroup 索引的高位选组。
- `wo_b`: fp8 [5120 × 8192]，41.9 MB。
- 两个 dispatch，输出 `a[5120]` fp16 进 Mega-mHC 的 `hc_post`（§7.7）。

### 7.7 `hc_post`（融合进下一个 Mega-mHC）

`x' = post ⊗ a + comb · x`（4 份拷贝各自加权）。与下一子层的 `hc_mixes` 在同一 kernel：读 `x`、`a`，写 `x'`，同时计算 `x'` 的 mixes。残差流始终 fp32。

**comb 的下标顺序（v0.7 更正，p2_attention.md §2.3）**：参考实现沿 `dim=2` 求和，
那是 comb 的**第一个**下标，所以逐元素写出来是

```
out[j] = post[j] * a + Σ_i comb[i][j] * residual[i]
```

**comb 相对于"顺手的读法"是转置的**。§7.2 的 Sinkhorn 输出本来就不是对称的
（它以列归一化结尾，行和偏差 8.5e-2，§2.4），所以**转置写反不会被任何守恒律挡住**，
只会安静地算错。

**收尾的那次 `hc_post`（block 出口）也是一次 mega_mhc dispatch**，见 §7.2 与 §7.14。

### 7.8 Router（Gate）

- `W [384 × 5120]` bf16，3.9 MB；`scores = sqrt(softplus(x·W))`；`+ bias` 取 top-6；权重归一化 × 1.5。
- 1 个 workgroup：384 个点积（6 个 wave × 64 行），top-6 用 wave 内 6 轮 `WaveActiveMax` + 屏蔽。
- 输出到 **host-coherent 缓冲**：`ids[6]`、`weights[6]`，以及 `top16_ids/scores` 供 Planner 的 heat 统计（近似命中）。同时 GPU 侧写一个 `layer_done` 计数器（原子 +1），CPU 自旋等待，不用 fence。
- CPU 拿到 ids → 检查指针表 → 缺失的下 P0 I/O → 全部驻留后 host-signal timeline 值 `token_base + layer`；command buffer 中下一 dispatch 前有对应的 wait。**GPU 在 6 个 expert 就位前不会跑 MoE kernel，其他情况零等待。**

### 7.9 融合 FP4 MoE FFN（每层 2 个 dispatch）

**Dispatch A：`gate/up + SwiGLU`**，7 个 expert（6 routed + 1 shared，shared 为 fp8 路径的模板实例）。

- workgroup 索引 → `(expert_slot, row_block)`；expert 基址 = 指针表 `addr[layer][ids[slot]]`。
- 每 wave 读 `w1` 行 i 与 `w3` 行 i（§5.1 v0.5：两者在分片里相隔 5,898,240 B，靠两个 buffer_device_address 并行取址），K=5120：每 lane 32 个 K 元素 = 16 B 的 FP4 + 1 个 E8M0 scale；解码 → 与 LDS 中 `x` 的对应 32 个 fp16 做 FMA（packed fp16 或 fp32），块内和 `ldexp` 后 fp32 累加；wave 归约得 `gate_i`, `up_i`。
- 尾处理：`gate = min(gate, 10)`, `up = clamp(up, ±10)`, `h_i = silu(gate) × up × route_weight`；写 `h[slot][2304]` fp16（7 × 4.5 KiB）。
- **中间激活必须量化（v0.6 新增，route_trace.md §11.2）**：参考实现 `Expert.forward` 的最后一步是
  `self.w2(x.to(dtype))` → `linear()` → `act_quant(x, 32, "ue8m0")`，也就是
  `silu(gate) * up` 这个中间结果在进 `w2` 之前要**按行每 32 个元素量化成 fp8 E4M3、
  block scale 是向上取到 2 的幂的 UE8M0**（§2.4 的 `fast_round_scale`）。
  v0.3–v0.5 的"每层 2 个 dispatch"漏了这一步——**它不是可选的数值细节，省掉会改变输出**。
  实现上它融合在 Dispatch A 的写出阶段（每 32 个元素一次 amax + `ldexp`），
  `h` 的存储因此是 fp8 + scale 而不是 fp16；§12 的 L1/L2 判据据此重算。
- ALU 预算：每字节权重（2 个 FP4）≈ 2 次 nibble 提取 + 2 次查表 + 2 次 FMA ≈ 6 op；200 GB/s → 1.2 Top/s，gfx1151 fp32 峰值 ~30 Top/s（fp16 packed 翻倍）；**M=1 时 ALU 富余 20×，M=6 时 3×**，int8 dot4（每指令 4 个 MAC）作为 M≥4 的备选。
- 备注：`w1`、`w3` 的 scale 布局 `[2304][160]`，每行 160 B，与权重行同步读。

**Dispatch B：`down + 归约`**

- 7 个 expert 的 `w2 [5120 × 2304]`，输入 `h[slot]` 在 LDS；每 wave 一行；输出 `y_slot[5120]`。
- 归约：同一 row_block 的 7 个 slot 由同一 workgroup 处理（workgroup 索引只按 row_block，内部循环 7 个 expert），fp32 累加后直接写 `y[5120]`，无需原子、无需第三个 kernel。代价是 workgroup 数减少为 5120/rows_per_wg，用 rows_per_wg=8 → 640 个 workgroup，对 40 CU 足够。
- `y` 进入下一层的 Mega-mHC（`hc_post`）。

**驻留检查的粒度（v0.7 推翻 v0.6 的定案）**：

> v0.6 写的是："**数据出来了：翻倍的代价是 0.66 µs × 2 × 40 层 ≈ 0.05 ms/token（§3.4），
> 可以忽略。因此'按到达顺序分组计算'是默认方案**"。

**这句话错了 340 倍。** 实测同一轮、同样 7 个 expert、同样的字节，把一层拆成 3+4 两组：

| M | 一次 7 槽 | 拆 3 + 4 | 差 | × 40 层 |
|---|---|---|---|---|
| 1 | 0.596 ms | 0.343 + 0.446 = 0.789 ms | **+0.193 ms（+32%）** | **+7.7 ms/token** |
| 6 | 1.237 ms | 0.584 + 0.762 = 1.345 ms | **+0.108 ms（+9%）** | +4.3 ms/token |

（2026-09-14，`bench/results/kernel_p2_moe.csv`，kernel_p2_moe.md §6.2。）

**多出来的不是启动开销**——§3.4 的 0.56 µs 是对的，只是它衡量的是空 dispatch。
多出来的是 **dispatch B 的 640 个 workgroup（5120 行 / 8）每多发一次就要重跑一遍
跨 lane 的 LDS 树归约、重读并重写 `y`、重新 ramp 起 8 个 wave**。
这部分工作量**与槽数无关、与 M 成正比**，所以 M=6 的绝对代价反而比 M=1 小（分母大了）。

**定案改为：默认一次算完；只有当本层确实有 expert 未就位、且预计等待 > 0.2 ms 时才分组。**
`Planner` 需要这个判据（§9.4 把 lookahead 降级为不做，是同一类判断：别为一个假想的收益
无条件付一个确定的代价）。kernel 侧的支持不变：indirection list（`SlotList` + `list_count`）
短一点即可，另加 `DownPush.flags` 的 bit 0 = 累加（`MoeRunner::set_accumulate()`），
让一层可以拆成任意多组。

**分组之后不是逐位相同，是 1–2 ULP。** 归约被重新结合了（一次算完时 lane 的 fp32 累加器按
`slot → block` 顺序加；拆开之后每组各做一次跨 lane 的 LDS 树归约，最后两个树相加）。
实测三种切法（3+4 / 1+6 / 6+1）的 max|Δy| ≤ **8e-8 的 |y|max（相对 2.4e-7）**，
与 kernel_p1.md §3.1 里 M=6 各列之间的差异同一量级。
**判据因此定为 1e-6，不是逐位相等**，`tests/test_gpu_moe.cpp::a_partial_dispatch_reduces_to_the_same_y`。

#### 7.9.1 实测（2026-09-14，`bench/results/kernel_p1.csv`，60 个变体；详见 [kernel_p1.md](kernel_p1.md) §3）

测的是上面两个 dispatch，7 个 expert，每次迭代读 87,736,320 + 43,868,160 = **131.6 MB**
（与 §7.14 的清单一致），工作集 1053 MB（8 层轮转，MALL 装不下）。

| | 最优变体 | 有效带宽 | 每 token MoE 时间 |
|---|---|---|---|
| **decode（M=1）** | `LanesPerRow=32, subgroup=32, DecodeMode=常量表, h=fp16, RowsPerLane=1` | **218.5 GB/s = raw-read 的 102%** | 0.602 ms |
| **投机验证（M=6）** | `LanesPerRow=16, subgroup=32, DecodeMode=常量表, h=fp16, RowsPerLane=2` | 135.1 GB/s = 上限的 **63%** | 0.974 ms / 6 token = 0.162 ms/token |

> **v0.7：上表的 0.602 / 0.974 ms 不是真实的每层时间，别再引用它们做端到端账。**
> P1 的第七个槽是拿一个 FP4 routed expert（12.5 MB）顶替 shared expert 的，
> 而真正的 fp8 shared expert 是 23.6 MB。**换成真权重之后每层是 0.683 ms（M=1）/
> 1.310 ms（M=6，= 0.218 ms/token）**，见 §7.9.2。§9.4 Q5、§10.1、§10.3、§13.4 已按新数字更新。

- **§15 P2 的"每 kernel ≥ raw-read 的 80%"在 M=1 上已达成并超出**（102%），而且
  raw-read 本身已不是有意义的上限（§7.1 rule 2）。
- **M=6 是现在最大的一块空地**：有效带宽只有 63%。~~瓶颈不是 ALU 而是激活 x 的读取指令~~
  （每个 32 元素块一个 lane 要发 6 条 `uint4` 读拿 96 B 的 x，却只读 32 B 的权重，
  L0 请求量是权重的 12 倍）。~~修法是 §7.1 rule 6 的 x 分块进 LDS；这是 P2 的头号 kernel 待办。~~
  **v0.7 推翻：瓶颈是 VALU 发射，不是访存；x 分块进 LDS 在每一个 M 上都更慢**（§7.9.2）。
  P1 的诊断"L0 请求量是权重的 12 倍"本身没错，但**那不是限制项**——L0 的带宽和发射端口都还有余量。
  `RowsPerLane=2` 在 M=6 上的 13% 收益也和"省访存"无关，它同时省了指令。
- `h` 用 fp16 而不是 fp32：fp32 把输出误差从 1.37e-4 降到 1.15e-4（改善 16%），代价是慢 9%
  （198.5 GB/s / 0.663 ms）。**决定用 fp16**，误差主线是输入的 fp16 量化而不是 `h`。
  （`HPrecision` 是 spec 常量，原地可切。）
- **正确性（§12 L1）**：十四个变体对 `tools/oracle.py` 的 torch fp32 结果全部
  **cos = 0.99999996、max|Δy| = 1.37e-4**（占 |y|max），远在 1e-3 判据内，而且**彼此完全一致**。
  误差几乎全部来自 `x` 从 fp32 舍入到 fp16（相对 L2 误差 2.04e-4），kernel 自身的累加误差可忽略。
- **路径 A / B**：同一个变体路径 A 221.8–222.1、路径 B 193.6–198.2 GB/s（§3.3）。

**实测钉死的三条约束**（之前只是纸面约定，写在这里免得被改坏）：

1. **part 地址只有 8 字节对齐，不是 16。** 全量 manifest 扫描：94,464 个 part 里
   87,552 个（92.7%）的槽内偏移 `mod 16 = 8`。kernel 每 lane 每块读 16 B（`uint4`），
   所以这些读**必然**是 8 对齐；Slang 发出的是 `OpLoad … Aligned 4`（SPIR-V 合法），
   RDNA 的 `global_load_dwordx4` 只要求 dword 对齐——**实测零代价**。
   这是 §5.1 v0.5"不 repack"决定的全部 kernel 侧代价，账已结清。
2. **slab 必须同时 host 可写、device 可寻址。** 纯 `DEVICE_LOCAL` 内存（§3.3 那条 216.0 GB/s）
   **不能**用作 expert slab，尽管它和路径 A 一样快——IoEngine 会往空指针写。
   `VulkanSlabBacking::allocate()` 对没有 `host_ptr` 的 slab 直接返回 `Internal` 错误。
3. kernel 需要 `experts_per_layer`（指针表第二维的步长）作为 push constant；
   `gpu/vulkan/device.cpp` 额外申请 `shaderInt16` / `shaderInt64` /
   `storageBuffer16BitAccess` / `uniformAndStorageBuffer16BitAccess` / `synchronization2`，
   §1.1 的能力表没列，gfx1151 全部支持。

#### 7.9.2 P2 实测：M 扫描、fp8 shared expert、`h` 的 fp8 量化（2026-09-14，`bench/results/kernel_p2_moe.csv`，261 行一次完整 sweep；详见 [kernel_p2_moe.md](kernel_p2_moe.md)）

同一台机器、同样的方法（`--iters 48 --layer-cycle 8 --repeats 3 --sweeps 2`），路径 A。
**先记一条量测纪律：这一轮的 run 间漂移是 ~7%（P1 记的是 1.5%），同一轮不同小节之间也有 8%。
所以下面所有"提升了多少"都取自同一轮内部的对照，绝对值给区间**（kernel_p2_moe.md §2）。
**CI 上的 kernel 带宽回归必须同轮对照，不能比绝对值。**

**(a) M 扫描：真正买到的是 `ms/token`，不是有效带宽**

| M | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 最优变体 | `L32 R1 xglob` | `L32 R1 xgf16` | `L16 R1 xglob` | `L16 R2 xgf16` | `L16 R2 xgf16` | `L16 R2 xgf16/ldsi8` |
| **A+B GB/s** | **222.6** | 206.4 | 178.2 | 162.0 | 149.7 | **135.3** |
| % 上限 | 102% | 95% | 82% | 74% | 69% | 62% |
| **ms/token** | **0.5912** | 0.3187 | 0.2461 | 0.2031 | 0.1758 | **0.1621** |

**M=1 → M=6 是 3.65 倍**。M ≥ 2 上 P2 的新变体比 P1 的 kernel 稳定快 **5–10%**
（保守口径，即拿全轮最好的 `xglob` 读数比；节内口径下 M=6 是 +14%）。
M=1 上一个字节也没省——那里 kernel 已经在内存系统的上限上。

**(b) 瓶颈在 M ≥ 3 上换了位置：VALU 发射，不是访存。** dispatch A 的时间是
`t_A(M) ≈ 0.33 + 0.055·M` ms 的**直线**。把斜率折成指令数：每列每次迭代
`7 × 2304 × 5120 × 2 = 165 M` 次 fp32 FMA + `7 × 2304 × 5120 = 82.6 M` 次 `f16tof32`，
除以机器的 VALU 发射能力（40 CU × 2 SIMD32 × 32 lane × 2.9 GHz ≈ 7.4 T op/s）得 0.033 ms，
对实测的 0.055 ms 是 **61% 的 VALU 占用**。
**直接后果：省访存的设计全都没用，省指令的设计都有用**，而且
**"有效 GB/s ÷ 内存上限"在 M ≥ 3 上不再是一个有意义的效率指标**——
分子是常数、分母被算术撑大，数字必然难看。§15 P2 的准出条件据此改写。

**(c) 三个设计的成败，都能用 (b) 解释**

| 设计 | M=6 的结果（L16 R2，同一轮） | 判决 |
|---|---|---|
| **x 按 K 分块进 LDS**（§7.1 rule 6 v0.6） | A+B 119.0 → **95.5**；L32 R1 上 110.4 → 42.6 | **作废**，见 §7.1 rule 6 |
| **packed fp16（`v_pk_fma_f16`，全局 x）** | dispatch A 117.2 → **133.9（+14%）** | **采纳，M ≥ 4 的 A 默认** |
| **int8 dot4（LDS 分块，`dot4add_i8packed`）** | dispatch B 100.0 → **125.7（+26%）**；A 上无收益 | **采纳，但只用在 B** |
| **A / B 分别特化**（§7.9.1 未解决问题 3） | 最优是 A packed fp16 + B int8 dot4 = **135.3** | **必要**，已实现（`MoeSpec::x_mode_b`） |
| `RowsPerLane` 放宽到 1..16 | **A 在 R ≥ 4 上塌方**（寄存器溢出），**B 在 R = 4 上还在涨** | A/B 的 R 也该分开，收益 < 2%，优先级低 |

**int8 的精度是它的定价，不是 bug**：`xgf16` 的 max\|Δy\| 是 **4.52e-4**（远在 1e-3 判据内，
可无条件采用）；int8 dot4 在 A 与 B 都量化时是 **5.38e-3**（M=6 逐列 7.36e-3），
**比 §12 的 5e-3 高 8%**。来源是 8 位激活量化本身：x 在块内近似高斯，amax ≈ 2.5σ，
步长 2.5σ/127，每元素相对误差 ~0.5%，两次 GEMV 叠到输出约 1%。
**所以 int8 只用在 dispatch B**（只有 B 用时误差约减半），并且 §12 的 int8 判据要按 L2 的实测重定，
而不是让 kernel 去迁就一个没有实测依据的 5e-3（§12、§15 的未解决项）。

**(d) 想让 M=6 摸到 85%，唯一已知的设计是"把 x 预量化成 int8"——这是一个待决的问题**

按 (b) 的模型，要到 185 GB/s（A+B 0.71 ms）需要把每 (列, 块, lane) 的指令数压到 FMA 数的 2 倍以内。
**唯一能做到的是每 token 做一次 x 的 int8 量化、40 层复用**：消费侧每 (列, 块) 变成
2 条 `uint4` 读 + 8 条 `dot4add_i8packed` × `RowsPerLane`，模型算下来 dispatch A 可以到 **~187 GB/s**。
代价是：一个新 buffer（`x_i8[M][5120]` + `[M][160]` 的 scale）、一个极小的前置 dispatch
（或融进上一个 kernel 的写出阶段），**以及 ~1% 的输出误差**。

> **未决问题（需要 L2 来定）**：~1% 的输出误差能不能接受？
> 参照系：`h` 的 fp8 量化**单独**就把输出改动 **1.0–3.3%**（§7.9 v0.6，实测见下面的 (e)），
> 而那是**必须做**的复刻要求。也就是说 int8 激活的量级**和参考实现自己引入的量级相同**。
> 判断不能在 kernel 层做，要等 §12 L2 的逐层曲线。**在那之前 M=6 用 packed fp16 + B int8。**

**(e) fp8 shared expert：模板成立，M=1 打满内存**

第七个槽换成真的 `layers.L.ffn.shared_experts.{w1,w2,w3}`（FP8 E4M3 + **32×32 tile scale**，
一层 35,424,000 B，是一个 routed expert 的 1.88 倍，与 §7.14 的清单吻合）。
同一对 dispatch 同时处理两种格式：`Ids[slot]` 的 bit 31（`gpu::kSlotFp8`）告诉 kernel 这个槽是 fp8，
kernel 据此切换字节步长（32 B/块 对 16 B/块）、scale 布局与解码表；
`Fp8Slots` 为 0 时整条 fp8 路径被折叠掉，**routed-only 的 pipeline 一个指令也没多**。

| 变体（7 槽 = 6 routed + 1 真 shared，148,227,840 B） | A+B GB/s | % 上限 | **每层 ms** | ms/token |
|---|---|---|---|---|
| **M=1 `L32 R1 xglob fp8`** | **216.9** | **100%** | **0.683** | **0.683** |
| M=6 `L16 R2 xgf16 fp8` | 113.2 | 52% | **1.310** | **0.218** |

**fp8 模板在 M=1 上和 FP4 一样打满内存系统。** 正确性 cos = **0.999999947**、
max\|Δy\| = 1.37e-4，与 FP4 routed expert 完全同一个量级（误差仍由 x 的 fp16 舍入主导）。
顺带确认 §7.9.1 的约束 1 在 shared expert 上也成立：六个 part 的设备地址 `mod 16 = 8`。

> **这两个数（0.683 / 1.310 ms）替换 §7.9.1 的 0.602 / 0.974**，§9.1.1 Q5、§10.1、§10.3、§13.4 已更新。

**(f) `h` 的 fp8 量化（§7.9 v0.6 的复刻要求）：正确性零疑问，decode 的代价比预期大**

两种实现：`HQuant = 1`（`hqB`，dispatch B 内现算）与 `HQuant = 2`（`hq8`，dispatch A 写出
fp8 + UE8M0 scale 平面，B 直接消费）。**`hq8` 明显更优**（`hqB` 为了拿块 amax 必须把 `h` 读两遍，
而且每 lane 每权重行重算一次同一个 scale；`hq8` 每块只算一次，并且把 `h` 的流量从 2 B/元素
降到 1.125 B/元素）。

- **数值上与参考逐位一致**：对"x 和 `h` 都先舍入到 fp16 再量化 `h`"这个 kernel 真正在算的向量，
  `hq8` 吻合到 **5.03e-8**、`hqB` 到 3.0e-7。也就是说 kernel 里的 fp8 E4M3 舍入
  （round-to-nearest-even）与 `fast_round_scale` 和 torch 的 `.to(torch.float8_e4m3fn)` **逐位一致**。
- **一个对 §12 L2 很重要的副作用**：`fast_round_scale` 是块 amax 的**阶跃函数**。
  kernel 的 `h` 是 fp16 而参考是 fp32，两者只差 2e-4，但只要某个块的 amax 落在 2 的幂两侧，
  整块 32 个值就落到**粗一倍的网格**上——实测这把 2e-4 放大成 **2.78e-3**
  （Python 侧同一实验给 2.763e-3，**与 GPU 的 2.776e-3 吻合**，所以这是量化器的敏感性而不是 kernel 的误差）。
  **`h` 量化之后，逐层的容差不能再按 1e-3 定**（§12）。
- **量化本身把输出改动 1.0–3.3%**（三个 expert：shared 层 0 为 1.00e-2、routed (0,0) 1.36e-2、
  routed (39,383) 3.31e-2）。§7.9 v0.6 说"省掉会改变输出"，实测证实，**幅度比 §12 的任何判据都大两个数量级**。
- **代价：M=1 +27%（0.591 → 0.749 ms），M=6 只有 +6%（0.1621 → 0.1721 ms/token）。**
  而且这 27% **不是量化本身的代价，是被迫换 workgroup 形状**：`hq8` 要求一个 workgroup
  拥有 32 的整数倍行（`(256/L) × R % 32 == 0`，否则一个 fp8 块会跨 workgroup），
  这把 M=1 从冠军形状 `L32 R1` 赶到 `L16 R2`，而 `L16 R2` 不量化本身就只有 179.3 GB/s；
  在它之上 `hq8` 几乎不要钱（175.6）。

> **这是 P2 之后 kernel 侧收益最大的一件事**：把 `h` 的写出阶段与计算的 workgroup 形状解耦
> （让 A 的 workgroup 覆盖 32 行的写出而保持 L32 R1 的计算形状，需要一次 LDS 转置），
> 或者把 `h` 的量化拆成第三个极小的 dispatch（2304 × 7 个元素）。
> 量级是 **40 层 × 0.158 ms = 6.3 ms/token**。

**(g) 接口变化（全部是加法，没有删改现有字段的语义）**

- `MoeSpec` 新增 `x_mode` / `x_mode_b` / `h_quant` / `fp8_slots`，默认值都是 P1 的行为（0）；
  `x_mode_b` 默认 `kFollowA`。`rows_per_lane` 的合法范围从 {1,2,4} 放宽到 1..16 的幂次。
- `MoeDims` 新增 `fp8_slot_count`（**只用于字节记账**，不影响 kernel 行为）；
  新常量 `gpu::kSlotFp8 = 0x80000000` 写进 `ids()[slot]` 的高位；新方法 `set_accumulate(bool)`。
- **唯一的 descriptor 布局变更**：dispatch A 的 storage buffer 从 7 个变成 8 个
  （binding 7 是 `h` 那块 allocation 的 `RWStructuredBuffer<uint>` 别名，`HQuant = 2` 用）。
  dispatch B 的 `DownPush` 从 24 B 变成 28 B（末尾多一个 `uint flags`）。
- **shared expert 不在 `ExpertStore` 里**：它是常驻的，单独分配一块 host-visible + device-address
  的缓冲，六个 part 的地址写进指针表的一个备用 expert 下标
  （`MoeDims::experts_per_layer = 385`，下标 384 留给 shared）。

**推荐的默认特化**

```
decode（M=1）        : L32 R1, sg32, dec0（常量表）, h=fp16, XMode=0
                       —— 开 HQuant=2 则必须换成 L16 R2（见 (f)）
投机验证（M=6）      : L16 R2, sg32, dec0, h=fp16, XMode=4（A packed fp16）+ x_mode_b=3（B int8 dot4）
shared expert 所在的 dispatch: Fp8Slots=1，槽的 ids 带 kSlotFp8
`h` 的 fp8 量化      : HQuant=2
```

### 7.10 Engram（层 1、14）

- CPU 侧：token id → 压缩 id → 24 个 hash → 行地址；行数据 24 × 264 B 由 EngramPrefetcher 提前读到 GPU 可见的小环形缓冲。
- GPU：`wkv [25600 × 6144]` fp8 157 MB GEMV（输入 6144 = 24 行 × 256，解码 fp8 + 每 32 元素 scale 时融合在输入加载中）→ `key[4][5120]`, `value[5120]`；门控 + 残差更新融合在写出阶段（每份拷贝各自做 RMS 归一化点积）。
- 2 个 dispatch（GEMV 需完整输入；门控需完整 key）。

### 7.11 LM Head 与采样

- `norm → head [129280 × 5120] bf16`（1.32 GB）→ `logits` fp32 → 采样。
- GEMV 结构同上（bf16 模板），129280 行 / 40 CU 负载均衡良好。**这是单个最大的常驻读取（10%）**，也是投机解码收益最大的地方（M=5 位置一次读完）。
- 采样在 GPU 上：温度缩放 + Gumbel-max（与参考一致）或 argmax；随机数用 Philox 计数器，seed 可复现。logits 只回传需要的 top-k 与采样结果。

### 7.12 DSpark 草稿 kernel

- `main_proj [5120 × 15360]` fp8（79 MB）+ norm → `main_x`（草稿 attention 的 KV 来源）。
- 3 个草稿块 × (Mega-mHC、Q 路径、KV、window attention（M=5，KV = 128 窗口 + 5 个块内位置）、O 路径、gate(128 expert, top-3)、FP4 MoE(M=5))：与主模型同一套模板，`M=5`。mtp expert 全部 pinned，无 I/O 等待。
- 头：`head` GEMM（M=5，读 1.32 GB 一次）；Markov head 5 步顺序：`embed_m [129280 × 256]` 查 1 行 + `head_m [129280 × 256]` GEMV 33 MB + 加偏置 + 采样 → 5 个小 dispatch；confidence head：`[5][5376] × [5376 × 1]` fp32。
- 草稿周期总流量 ≈ 2.2 GB ≈ 11 ms。

### 7.13 Prefill kernel（M ≥ 16）

- 所有线性层切换为 cooperative matrix（`VK_KHR_cooperative_matrix`，fp16 累加 fp32）GEMM，权重解码 FP4/FP8 → fp16 在加载到 LDS 时完成。
- MoE prefill 按 **expert-major**：对每个 expert，收集路由到它的 token 行（gather），做 M=n_e 的 GEMM，scatter-add 回去。expert 顺序按分片内的物理顺序（§5.1 实测：一层的 384 个 expert 全在同一个分片里），和 NVMe 顺序流式一致（§9.7）。
- Attention prefill：window 部分为 band attention；压缩部分需先完成 Compressor 的全序列压缩与 Indexer 打分 — 分块（block 128）实现，正确性对齐参考的因果可见性规则（压缩块只有在完整后才可见）。

### 7.14 每层 dispatch 清单（Reuse 模式层，decode）——**v0.7 按实现更正为 14–15 个，不是 11 个**

| # | kernel（shader.stage） | 主要字节 |
|---|---|---|
| 1–3 | **Mega-mHC(attn)**：`mega_mhc.post` / `.mix` / `.final` + attn_norm（§7.2：3 个 dispatch，不是 1 个） | 2 MB |
| 4 | `wq_a` | 6.6 MB |
| 5 | `wq_b` + q_norm + RoPE | 41.9 MB |
| 6–7 | `wkv.gemv` + `wkv.finish`（kv_norm、RoPE、fp8 写 cache；source 层附加 compressor / indexer 1–3 个） | 2.6 MB |
| 8–9 | `sparse_attn.score` + `.combine` + 逆 RoPE（§7.5：两遍，不能在线 softmax） | 0.3 MB |
| 10 | `wo_a`（分组，`ActQuant = 0`） | 33.6 MB |
| 11 | `wo_b` | 41.9 MB |
| 12–14 | **Mega-mHC(ffn)** 的三个 stage，其中 stage 0 融合 attention 半的 `hc_post` + ffn_norm | 2 MB |
| 15 | `gate.score` + `gate.topk` → host-coherent（两个 dispatch，但共用一个 §7.14 行） | 3.9 MB |
| — | **timeline wait**（CPU 确认 6 个 expert 驻留） | |
| 16 | FP4/FP8 gate/up + SwiGLU（7 expert） | 6 × 12.5 + 23.6 MB |
| 17 | down + 归约（7 expert） | 6 × 6.3 + 11.8 MB |

**合计 ≈ 14–15 个 dispatch**（block 出口那次 `hc_post` 与下一子层的 mega_mhc stage 0 融合，
所以每层三次 mega_mhc 只体现为两组 ×3 加一次融合；带 compressor / indexer 的 source 层再多 1–3 个）。
~~与 DeepSeek 官方 Reuse 模式 decode 的 11 个 kernel 一致~~ —— **v0.7 作废**：官方那 11 个是
逻辑 kernel，我们多出来的四个全部是**grid-wide barrier 的代价**（两个 RMS 归约、一次 softmax 的
两遍、一次 top-k 的两遍），不是多算了什么。

**对 §3.4 的影响很小**：每 token 从 470 个 dispatch 变成约 **600 个**，
按实测 0.66 µs 是 **0.40 ms/token**（原 0.31 ms），仍然是 65 ms 的 0.6%。

### 7.15 非 MoE decode 路径的实测（P2 step 1，2026-09-14，`bench/results/attn_p2.csv`；详见 [p2_attention.md](p2_attention.md)）

九个 Slang kernel、十八个 pipeline、一个 pinned 权重集、一个导出参考实现自身逐级张量的 oracle，
外加一个能跑完整一层 decoder 的 runtime。**全部在真实的 `DeepSeek-V4.1-Flash` checkpoint 上，
64 token prefill 之后位置 64 的那一个 decode token。**

#### 7.15.1 正确性（§12 L2）

逐级对 oracle（七个层里最差的一层）：

| stage（design §） | cos |
|---|---|
| mega-mHC mixes + Sinkhorn（§7.2） | 1.000000000（pre/post/comb 到 6e-7） |
| mega-mHC 的 `attn_norm` 输出（§7.2） | **1.000000000，七层全部逐位相同** |
| `wq_a`（§7.3） | 0.999998624 |
| `wq_b` + `q_norm` + RoPE（§7.3） | 0.999998322 |
| `wkv` + `kv_norm` + RoPE（§7.4） | 0.999998342 |
| `wkv` 经 fp8 cache 写入之后（§7.4） | 0.999910658 |
| 稀疏 attention + 逆 RoPE（§7.5） | 0.999997368 |
| `wo_a`（分组，§7.6） | 0.999998577 |
| `wo_b`（§7.6） | 0.999998575 |
| `hc_post` 进残差流（§7.7） | 0.999998535 |
| gate 打分（§7.8） | 0.999999996，**每一层都 6/6 个 expert 一致** |
| `head`（§7.11），对 fp64 CPU 点积 | 1.000000000 |

**整层链起来**（只有 block 输入与 prefill 留下的 KV 是 golden）：

| | `ffn_norm` | gate | MoE 输出 | **block 输出** |
|---|---|---|---|---|
| 层 0 | 0.999933 | 6/6 | 0.999778 | **0.999935** |
| 层 39 | 0.999986 | 6/6 | 0.999977 | **0.999980** |

**§12 L2 的判据是层输出 cos ≥ 0.999，两端都超出两个数量级。**

**判据用 cos 与相对 L2，不用 max 相对误差**（p2_attention.md §3.1）：§12 **L1** 用
`max|Δy| / max|y|` 是对的——L1 拿一个 expert FFN 对 fp32 oracle，一个离群元素就是真 bug。
**L2 不一样**：参考自己就是 bf16 和 fp8，一个元素落到舍入边界的另一侧就会动**它自己量级**的
半个 ulp（bf16 0.4%、E4M3 6%），而 `max|Δ| / max|y|` 会把它报成整个向量的误差。
`wkv` 过 fp8 之后是最清楚的例子：512 个 E4M3 字节里有 4–9 个与参考不同，
`rel` 读 2–5e-2，而相对 L2 是 7e-3、cos 是 0.99991。
`tests/test_gpu_attn.cpp` 两个都打印，但**只对 cos 与相对 L2 设门**。

#### 7.15.2 带宽（空闲机，`--layers 8 --iters 64 --rows 2`）

**先说这个数为什么是诚实的**：一个 `AttnRunner` 拥有一份地址表，所以"在**提交之间**轮转层"的
基准会让一个 command buffer 里的每一次迭代都读同一层的权重，**任何小于 32 MB MALL 的东西
都会报出幻觉**——第一版就把 `wq_a` 报成 374 GB/s、`wo_a` 报成 397，在一个 217 GB/s 的内存系统上。
现在是**每层一个 runner，第 i 次迭代用第 i mod layers 个**。对照的上限是 kernel_p1.md §2.2 的 216–218 GB/s。

| kernel | µs | bytes | GB/s | % 上限 |
|---|---:|---:|---:|---:|
| `mega_mhc.post` | 2.3 | 164 KB | 72 | 延迟型 |
| `mega_mhc.mix` | 5.1 | 1.97 MB | 385 | *MALL 驻留，见下* |
| `mega_mhc.final` | 3.2 | 41 KB | 13 | 延迟型 |
| `wq_a` | 44.3 | 6.56 MB | 148 | 68% |
| `wq_b` | 268.3 | 41.98 MB | 157 | 72% |
| `wkv.gemv` | 20.8 | 2.62 MB | 126 | 58% |
| `wkv.finish` | 8.8 | 2.6 KB | — | 延迟型 |
| `sparse_attn.score` | 61.6 | 134 KB | — | 延迟型 |
| `sparse_attn.combine` | 80.6 | 134 KB | — | 延迟型 |
| `wo_a` | 201.0 | 33.59 MB | 167 | 77% |
| `wo_b` | 346.8 | 41.98 MB | 121 | 56% |
| `gate.score` | 16.3 | 3.93 MB | 242 | *部分 MALL* |
| `gate.topk` | 12.6 | 3.1 KB | — | 延迟型 |
| **一层，dispatch 1–9** | **1071** | **133 MB** | **124** | **57%** |
| `head` | 5643 | 1.32 GB | **235** | **108%** |
| **40 层 + head** | **48.5 ms** | | | |

两条 caveat：`mega_mhc.mix` 与 `gate.score` 在 8 层上的工作集是 31 MB，**坐在 32 MB 的 MALL 里**，
它们的 GB/s 是 cache 数字不是内存数字——真 token 轮转 40 层，不会。
`head` 的 235 GB/s 高于 raw-read，与 kernel_p1.md §3.2 的 MoE kernel 102% 是同一回事
（一条很长的顺序 bf16 流 + 16,160 个独立 workgroup，在途请求比 raw-read shader 多），
§7.1 rule 2 已经说过 raw-read 只是上限的下界。

**对 §3.4 的模型**（常驻权重 0.97–1.29 ms 一层）：**路径在预算之内**（1.07 ms），
但它离内存系统还远，缺口在哪见 §7.15.4。

**`rows_per_lane` 由 sweep 定死，不能靠形状猜**（每个 stage 的 cap 写在
`gpu/vulkan/attn_kernels.cpp` 的 `kStages` 里，紧挨着这些数字）：

| kernel | rows 1 | rows 2 | rows 4 | cap |
|---|---:|---:|---:|---:|
| `wq_a`（1280 行） | 125 | **147** | 147 | 2 |
| `wq_b`（32768） | 114 | **160** | 157 | 2 |
| `wkv`（512） | **139** | 131 | 22 | 1 |
| `wo_a`（8192） | **166** | 132 | 132 | 1 |
| `wo_b`（5120） | **124** | 114 | 115 | 1 |
| `head`（129280） | **235** | 208 | 183 | 1 |

只有"高而窄"的两个赚了；其余都亏。`wkv` 在 4 上是十六个 workgroup 对四十个 CU。
head 没有东西可复用（它的激活 20 KiB 本来就在 LDS 里），寄存器压力纯是成本。

**权重靠 `buffer_device_address` 到位，不靠 descriptor。** pinned 集合是 17.7 GB、
散在许多小于 2 GiB 的区域里，每 tensor 每层一个 descriptor 就是每 token 重写几千个。
**每个 kernel 只绑一个 descriptor：一张共享 `uint64_t` 地址表的 32 槽切片**——
和 §5.3 的 expert 指针表是同一个办法。

#### 7.15.3 什么是真的，什么是加载进来的

`tests/test_gpu_layer.cpp` 按 decode 将来的样子跑完整一层：每个 stage 吃上一个的输出，
dispatch 1–9 一个 command buffer，§7.1 的 timeline gate，然后 MoE。

| 部件 | 状态 |
|---|---|
| mega-mHC、Q 路径、KV 路径、稀疏 attention、输出投影、gate、head | **真的**，权重由 IoEngine 从 NVMe 读进 pinned 集合 |
| window KV 环 | **真的**，`wkv.slang` 每步按 `Attention._window_kv` 的写法写 |
| routed experts | **真的**，经 `store::Planner` 按 gate 的 ids 取，由 §7.9 的 kernel 算 |
| shared expert | **真的**，fp8，从 pinned 集合经它自己的单槽 runner |
| **压缩 KV 与 indexer 的 top-512 列表** | **加载进来的**（来自 oracle 的 prefill）。**§7.4 的 compressor / indexer kernel 还没写**；它们下游的一切都是真的、也都测了 |
| prefill、engram 写入、DSpark | 未开始 |

**压缩的那一半现在存 bf16，不是 §11.3 的 FP4 E2M1 + E4M3/16。** 值本来就在 fp4 网格上
（compressor 量化过），所以这只是一个打包选择，64K 上下文上值 18 MB（66 对 48 MB），
拿回来的代价是 `sparse_attn` 内层循环里一次 nibble 解包。**等 compressor kernel 能写出打包形式时再定。**

**§7.1 的 gate 是真跑的**：`DecodeLayer::run_moe` 从 host-coherent 内存读出 gate 的 ids，
回调让六个 expert 驻留，host-signal MoE 那个 wait 命名的 timeline 值，然后 dispatch。
P2 里每个 expert 本来都会驻留、这个 wait 是走形式——**但测试故意只给八个槽的 cache**，
所以六个 expert 每层都真的从 NVMe 经 `store::Planner` 取回来，gate 就是放行它们的那个东西。

**command buffer 是每层一个，不是每 token 一个**（§7.1 rule 1）。拦路的是地址表：
一个 stage 拥有表里的一个 slice，预录制一个 token 需要**地址表在 shader 里按层索引**，
就像 expert 指针表那样。按 §3.4 的账值 ~0.5%，所以它等 P3 的流式 runtime。

#### 7.15.4 缺口，按大小排序

1. **`wo_b` 与 `wq_b`，121 和 157 GB/s（对 217）。** 它们是一层 133 MB 里的 84 MB，
   一层只有 57% 主要就是因为它们。嫌疑按顺序：LDS staging 两侧的两个 barrier
   （一个 workgroup 退 8–16 行，每次都要付）、`wo_b` 的 K = 8192 意味着 16 KiB LDS 与低占用率、
   以及每 lane 只有一个累加器（kernel_p1.md §3.3 在 FP4 路径上也发现这不足以掩盖延迟）。
   **结构性的答案是 K-split 加第二遍归约。**
2. **`sparse_attn`，142 µs 一层 = 5.7 ms 一个 token，占这条路径的 13%。**
   64 个 workgroup 每个都把整个 320 KiB 的 KV 拉过 L2 两遍（打分、再 P·V），
   **320 KiB 的数据产生 42 MB 的 L2 流量**。§7.5 已经描述了修法（KV tile 进 LDS 跨头共享），
   障碍是 64 个头的 q 是 64 KiB 放不下，所以要按头分组 + 部分 softmax 合并。
   在一个 decode token 的 2% 上不算急，但**它是这条路径上最大的一块纯浪费**。
3. **compressor 与 indexer kernel**（§7.4）。在它们存在之前压缩 KV 是加载进来的，
   **这是这条路径与一个能独立成立的 decode step 之间唯一的缺口**。
4. **按层索引的地址表**，让 §7.1 的每 token command buffer 可表达（§7.15.3）。
5. **`mega_mhc.post` 只有 72 GB/s。** 一次 164 KB、一个 token 八十次，2.3 µs × 80 = 0.18 ms，
   不大，但它是 20480 个 float 上的纯 `hc_post` / `hc_pre`，本该贴着上限。

---

## 8. CPU 路径

### 8.0 实测：内存控制器是单一共享上限，"CPU 分担 expert GEMV"从带宽上就不可能（v0.6）

`bench/bw_matrix.exe` 在 GPU 跑 raw-read（~200 ms 窗口）的同时让 32 个 CPU 线程顺序读主机内存。
实测 2026-09-14，`bench/results/bw_matrix.csv`（方法与两个踩过的坑见 [kernel_p1.md](kernel_p1.md) §1–§2.3）：

| 场景 | GPU GB/s | CPU GB/s | **合计** |
|---|---|---|---|
| GPU 单独（路径 A） | **216.4** | — | 216 |
| CPU 单独（32 线程峰值） | — | **100.9** | 101 |
| 并发，GPU 读路径 A | 185.8 | 28.6 | **214.4** |
| 并发，GPU 读路径 B | 184.4 | 28.7 | **213.1** |
| 并发，GPU 读纯 DEVICE_LOCAL | 186.4 | 29.6 | **216.0** |

**三条路径完全一样，而且并发合计正好等于 GPU 单独时的 216 GB/s。内存控制器的总读带宽就是
~217 GB/s，CPU 和 GPU 只是在分它**（仲裁偏向 GPU：CPU −71%，GPU −14%）。
CPU 峰值 100.9 GB/s 只有理论 256 的 39%、GPU 的 47%；3–8 线程之间有一个平台
（53 → 55 GB/s），16 线程才继续爬。

**直接后果：删掉"CPU 分担 expert GEMV"（原 §15 P6）。** 系统总带宽固定，把一部分权重挪给
CPU 算不会增加总吞吐，只会 (a) 让 GPU 少拿 14%，(b) 由一个每字节更慢的执行体处理这部分。
这与 h 有多高无关——它不是"等 NVMe 不再主导才有意义"的实验，而是**带宽上不成立**。
已移入 §16 明确不做。

**反过来这对 IoEngine 与 Planner 是好消息**：它们只需要几 GB/s（NVMe 上限 4.7），
而 CPU 在 GPU 满载时仍有 **29 GB/s** 可用，绰绰有余。

### 8.1 CPU 的角色

在 V4.1-Flash 的量级下，CPU 的价值是**编排、gate/预测数学、I/O**，不是算 expert。角色按优先级：

1. **IoEngine 与 Planner 的宿主**：IOCP 完成处理、请求切分、cache 管理、预测排序。绑定 2–4 个物理核，避免抢 GPU 带宽。
2. **Engram hash**：int64 乘法-XOR-取模，每 token 24 × 2 次，µs 级。
3. ~~**Lookahead gating**（§9.4）~~ —— **v0.6 取消**：§9.4 的预取已降级为不做，这条 GEMV 没有消费者了。
   若将来换一个更好的预测器再启用（§9.4）。
4. **fp32 oracle 前向**（§12）：完整参考路径，慢但精确；也是 P0 的第一个可运行产物。
5. **AVX-512 / VNNI 量化 GEMV**：FP4 nibble 解码与 `vpdpbusd` 融合的 int8 路径；
   **只作为 GPU kernel 的 A/B 参照与 fallback**，不再是"CPU 分担 expert"的执行体（§8.0）。
6. **CPU 拷贝不是落地瓶颈**：实测单线程写入 21.5 GB/s（路径 A 的写合并映射，非临时存储无收益）
   / 32.7 GB/s（路径 B 的可缓存内存，非临时存储省掉 RFO，快 1.8 倍），
   都远超 NVMe 的 4.7 GB/s。一个 18.8 MB expert 拷贝 0.6–0.9 ms。
   **不过真正的零拷贝路径连这一次拷贝都没有**（§9.6，已实测跑通）。

CPU/GPU 协同分担 expert（v0.1 的核心假设）**在 v0.6 被实测否决，见 §8.0 与 §16**。

---

## 9. NVMe 冷层与智能 Prefetch

这是本项目的核心难点。设计原则：**先测量，再设计策略；所有策略先在离线模拟器上用真实路由 trace 评估，再进 C++。**

### 9.1 要回答的问题

| # | 问题 | 决定什么 |
|---|---|---|
| Q1 | 每层 expert 的静态频率分布：top-x% expert 覆盖 y% 的路由？ | pinned 集合是否值得存在；`noaux_tc` 负载均衡可能让静态偏斜很小 |
| Q2 | 重用距离（stack distance）分布 | 任意容量下 LRU 命中率的解析解；cache 容量按层分配 |
| Q3 | 相邻 token 的 expert 集合 Jaccard 重叠（每层） | 时间局部性；投机解码 verify batch 的 expert 并集增长 |
| Q4 | Lookahead 路由预测：用第 L−d 层的残差流预测第 L 层 top-6，recall@K（K=6..16，d=1..8） | prefetch 的提前量 d 与宽度 K；预取浪费比例 |
| Q5 | 一个 token 内，MoE 层间的 GPU 计算时间 T_layer 与单 expert 读取时间 T_io 的比值 | 需要多少层提前量才能隐藏一次 miss |
| Q6 | NVMe 微基准：顺序/随机吞吐 vs 请求大小 (64 KiB–18.8 MB) × QD (1–64)；读入 GPU 可见内存 vs 普通内存 | 请求切分大小、目标 QD、是否需要 staging |
| Q7 | Engram 行读取的延迟分布（4 KiB 随机，QD 48） | 预取需要提前多少 token |

### 9.1.1 Q1–Q5 的实测答案（2026-09-14，P1 完成）

**trace**：`tools/route_trace.py` 直接 import 参考实现 `inference/model.py`，只替换六个跑不了
CPU 的 tilelang kernel；语料 `tools/corpus.py`，**40 个 prompt / 27,399 token**（中 38.6% /
英 34.2% / 代码 27.2%，19 个源文件，长度 p50 606 / p90 1324）。faithfulness 检查 6 项全过
（Engram 压缩词表 99,092 = config、manifest run/skew 与 `safetensors` 库逐字节相同、
FP4→bf16 解码逐位相同、我们的 top-6 与 `Gate.forward` 每层每 token 完全一致、路由权重和恒为 1.5000、
teacher forcing 下 3/4 argmax 命中语料自身的下一个 token，唯一不匹配处 margin 只有 0.032）。
方法、局限与四条需要回写的参考实现出入见 [route_trace.md](route_trace.md)。
模拟结果 `reports/cache_sweep.json`、`reports/cache_prefetch_{head,tail}.json`。

**Q1 静态频率偏斜：真实存在，但 LRU 自己就吃掉了，不值得做静态 pin。**

| | 实测（40 层） |
|---|---|
| 最忙 10% 的 expert 覆盖的路由 | **34%（层 0）– 63%（层 25）**，均值 **49%** |
| 最忙 5% / 20% / 50%（均值） | 35% / 66% / 90% |
| 每层激活的不同 expert | 366–384（共 384） |
| Gini | 0.478（层 0）– 0.754（层 25） |
| `noaux_tc` bias 改变 top-6 的 token 比例 | **总体 37%**，逐层 **20%（层 38）– 59%（层 12）** |

偏斜曲线离对角线很远——v0.3 猜的"`noaux_tc` 负载均衡可能让静态偏斜很小"**不成立**。
但偏斜的形态是"层 0 最平、中后段（层 24–28）最偏、末尾略回落"，而且
**`static-pin + LRU` 在每个容量上都不如纯 LRU**（见 Q2 下方的策略表）：LRU 自己就抓住了那份偏斜，
把 10% 的槽冻结成静态集合只是减少了 LRU 能用的空间。**结论：§9.3 的 routed expert
"静态 pin"子档删掉**，只留 attention/shared/router 那类真正永驻的。
`gate_bias` 改变 37% 的选路这一条说明**负载均衡器在主动改选路**，预测器不能忽略 bias。

**Q2 重用距离：全局 LRU 的解析解见 §3.1 的表。**

| | 实测 |
|---|---|
| 首次访问（冷）占比 | **0.25%** |
| 重用距离 p50 / p90（全局交错流） | **600 / 4,815** |
| 层内重用距离 p50 / p90 | 8–48 / 63–197 |

**策略对比**（无预取；`reports/cache_sweep.json` 的 `runs`，命中率为模拟器实测）：

| 容量 | 分配 | LRU | LFU-decay | ARC | score-aware | static-pin+LRU |
|---|---|---|---|---|---|---|
| 3072 (20%) | global | 0.8266 | 0.8266 | 0.8266 | **0.8288** | 0.7999 |
| 3840 (25%) | global | 0.8638 | 0.8638 | 0.8638 | **0.8661** | 0.8503 |
| 4608 (30%) | global | 0.8914 | 0.8914 | 0.8914 | **0.8934** | 0.8841 |
| **4787（90 GB）** | global | 0.8968 | 0.8968 | 0.8968 | **0.8988** | 0.8904 |
| 5376 (35%) | global | 0.9125 | 0.9125 | 0.9125 | **0.9145** | 0.9086 |
| 4787 | per-layer | 0.8899 | 0.8902 | 0.8899 | 0.8920 | 0.8840 |

1. **LRU、LFU-decay、ARC 在四位小数上不可分。** 这条 trace 里没有 ARC 的 scan-resistance
   能吃到的结构，也没有 LFU 能吃到的长期频率结构——重用距离太短，**近期性就是全部信号**。
2. **score-aware 每个容量都赢，但只赢 0.20–0.23 个点。** 代价是每层 16 个 heat 更新 +
   一个按 heat 排序的淘汰堆。
3. **global 稳定胜过 per-layer 0.6–0.8 个点**（每个容量都是）。按层配额把一层用不完的槽锁死了，
   而层内 p50（8–48）与 global p50 / 40 层（15）是同一量级——各层的工作集没有互相挤压。
4. **static-pin + LRU 差 0.4–2.7 个点**（容量越小差得越多）。

**Q3 相邻 token 的 expert 重叠：投机解码只能部分摊薄 expert 流量。**

| | 实测（40 层） |
|---|---|
| 相邻 token 的 Jaccard | 均值 **0.239**，逐层 **0.104（层 19）– 0.359（层 25）** |
| `union_frac[2] / [3] / [4] / [5]` （均值） | 0.824 / 0.734 / 0.676 / **0.634** |
| `union_frac[5]` 逐层范围 | **0.518（层 25）– 0.818（层 0）** |

`union_frac[k] = |k 个相邻 token 的 expert 并集| / 6k`。k=5 的 verify batch 读
`0.634 × 30 ≈ 19` 个 expert 而不是 30 个，**NVMe 流量省 37%**；换个说法，
**它碰的 expert 是单 token 的 2.6–4.1 倍（均值 3.2 倍）而不是 5 倍**。
这就是 §10.3 调度曲线里 `T_nvme(k)` 的输入（假设草稿全被接受，是保守端）。

**Q4 Lookahead 路由预测：recall 不够，预取净负收益。** `pre-identity` 近似
（`hc_pre(x, pre_identity)` 就是残差流的第 0 份拷贝，**零计算**，比 `mean_hc` 高 6–8 个点）：

| d \ K | 6 | 8 | 12 | 16 |
|---|---|---|---|---|
| **1** | **0.658** | 0.728 | 0.797 | 0.833 |
| **3** | 0.518 | **0.578**（precision 0.434） | 0.652 | 0.697 |
| 4 | 0.475 | 0.531 | 0.602 | 0.649 |
| 8 | 0.363 | 0.408 | 0.471 | 0.515 |

深度每加一层 recall 掉 4–5 个点，一直到 d=8 都没有拐点。**接进模拟器之后每个 (d, K) 都是净负**
——见 §9.4。

**Q5 `T_layer` / `T_io`：实测确认 `d ≥ 4`，但这个结论现在没有消费者。**
`T_layer(MoE, M=1) = 0.683 ms`（**v0.7 实测，含真 fp8 shared expert**，§7.9.2；v0.6 写的 0.602 是
第七个槽用 FP4 顶替时的数），非 MoE 的那一层实测 **1.071 ms**（§7.15.2），
整层合计 **≈ 1.75 ms**（§3.4 的模型说 0.97–1.29 ms，那是按字节 ÷ 上限算的下界；
实测高出来的部分是 `wo_b` / `wq_b` / `sparse_attn` 还没打满，§7.15.4）；
`T_io(18.8 MB) ≈ 4.0–4.26 ms`（§9.2.1、§5.1.3）。
隐藏 1 个 miss 需要 `d ≥ 3`（按 1.75 ms 的实测整层；v0.6 按 0.97–1.29 ms 算得 `d ≥ 4`）；
隐藏一层 6 个全 miss（并发 23.7 ms）需要 `d ≥ 14`，**仍然超过 40 层里能用的窗口**。
这从另一个角度重述了 §3.1 的结论：**提前量救不了低命中率，只有命中率本身能救**——
而命中率已经靠容量修到 0.92 了（§5.2），所以这一条更没有消费者了。

### 9.2 测量工具（P1 产物，均已完成）

- `tools/route_trace.py`：用 P0 的 CPU 前向在语料（中英代码混合，≥ 20K token，多 prompt）上记录 `(token, layer, top6_ids, top16_ids, top16_scores, hidden_norm_input)`；同时记录每个 d 的 lookahead 预测集。输出 Parquet。
- `tools/cache_sim.py`：读 trace，模拟容量 C ∈ {20%, 25%, 30%, 35%}、策略 ∈ {LRU, LFU-decay, ARC, score-aware, static-pin + LRU}，输出 hit rate、miss bytes/token、prefetch 浪费；同时模拟 lookahead 预取（给定 d、K、precision）。
- `bench/nvme_bench.cpp`：Win32 overlapped I/O 微基准，回答 Q6/Q7 → §9.2.1。
- `bench/bw_matrix.cpp`：CPU/GPU 单独与并发读带宽（P-1）→ §8.0、§3.3。
- `bench/kernel_bench.cpp`：§7.9 两个 dispatch 的变体 sweep、dispatch 开销、两条路径对照
  → §7.9.1（P1 的 60 变体，`--p1`）、§7.9.2（P2 的 261 行 sweep，默认）、§3.3、§3.4。
- `bench/attn_bench.cpp`（**P2 产物**）：§7.2–§7.8/§7.11 逐 kernel 的 µs / bytes / GB/s，
  每层一个 `AttnRunner` 以避开 MALL 幻觉 → §7.15.2。
- `bench/heap_capacity.cpp`：两个 heap 的**实际可分配上限**（逐 slab 分配 + 触页 +
  `GlobalMemoryStatusEx`）→ §5.2、§9.2.2。

`tools/` 的单元测试：`uv run python tools\tests\test_cache_sim.py`，12 个**答案可独立算出**的
合成用例（完美局部性 → `1 − 1/n`、均匀随机 → `C/M`、stack distance 对暴力定义逐元素相等、
`P(distance < C)` 与模拟出的 LRU 命中率互证等）。**模拟器的每一条结论都只和它的实现一样可信**，
这是那些用例存在的理由。

### 9.2.1 Q6 / Q7 实测结果（2026-09-14，P-1 部分完成）

`bench/nvme_bench.exe`（经 `storage::IoEngine`，FILE_FLAG_NO_BUFFERING + IOCP，
2 GB 测试文件在 C: 上，48 次请求/点）。完整矩阵见
`bench/results/nvme_q6_q7.csv`。

**Q6：吞吐 vs 请求大小 × 队列深度**（GB/s，WD SN740，PCIe 4.0 x4）

| chunk | QD1 seq | QD4 seq | QD8 seq | QD32 seq | QD8 rand | QD32 rand |
|---|---|---|---|---|---|---|
| 64 KiB | 0.09 | 1.13 | 1.71 | 2.35 | 0.34 | 0.53 |
| 256 KiB | 0.94 | 2.01 | 2.64 | 3.43 | 0.87 | 1.57 |
| 1 MiB | 2.03 | 3.77 | 4.17 | 4.09 | 2.30 | 2.94 |
| 2 MiB | 2.60 | 4.29 | 4.41 | 4.58 | 4.54 | 4.40 |
| 4 MiB | 3.19 | **4.52** | **4.54** | 4.67 | **4.68** | 4.46 |
| 8 MiB | 2.44 | 3.57 | 3.68 | 3.93 | 4.62 | 4.61 |
| 18.36 MiB（整个 expert） | 2.62 | 4.04 | 4.15 | 4.02 | 4.69 | 4.75 |

结论：

1. **饱和带宽 ≈ 4.5–4.75 GB/s**，比 §3.1 假设的 5 GB/s 低约 8%。§3.1 的 TPS 表因此
   略微乐观，但结论（NVMe 主导）不变。
2. **§9.6 猜的 2–4 MiB chunk 是对的。** 4 MiB × QD 4–8 已达 4.5 GB/s，且延迟只有
   3.6–6.9 ms；再往上只换来更高延迟（4 MiB × QD64 = 4.56 GB/s 但 37.7 ms 平均延迟）。
   **定为 chunk = 4 MiB、目标 QD = 8**（在途 32 MB，而非 §9.6 写的 ≥ 64 MB——64 MB
   在途只增加延迟，不增加吞吐）。
3. **≥ 2 MiB 时随机读与顺序读几乎无差别**（4 MiB：4.54 seq vs 4.68 rand）。这块盘在
   expert 粒度上不在乎局部性。**§9.7 的 expert-major 顺序流式因此不是为了带宽，而是
   为了一次读进整层后能按到达顺序算**；prefill 若改成按路由顺序随机读，带宽代价可忽略。
4. 一个 expert（18.8 MB）单发 ≈ 4.0–4.7 ms，与 §9.4 估的 T_io ≈ 3.8 ms 吻合，
   故 **lookahead 提前量 d ≥ 3–4 的推导成立**。
5. 6 个 expert 全 miss 的一层 ≈ 23.7 ms；40 层全 miss ≈ 0.95 s/token。

**Q7：engram 行读取（4 KiB 随机，QD 48）**

| 指标 | 实测 |
|---|---|
| IOPS | 83,700 |
| 吞吐 | 0.343 GB/s |
| 平均延迟 | 0.55 ms |
| 最大延迟 | 0.99 ms |

每 token 每 engram 层 24 行、两层共 48 行，一批 < 1 ms。**EngramPrefetcher 只需要
一个 token 的提前量**，草稿 token 一确定就下单绰绰有余（§9.5）。

**P-1 已全部完成**（v0.6）：带宽矩阵的 GPU 与并发部分见 §8.0 / §3.3，容量见 §9.2.2，
kernel 见 §7.9.1，dispatch 开销见 §3.4。原计划的"两种 VGM 设置下各测一遍"**作废**——
实测证明 BIOS VGM 既不影响带宽也不影响容量（§3.3 第 3 点、§5.2）。

### 9.2.2 容量实测（2026-09-14 → **2026-09-14/15 复测，已解决**）

`bench/heap_capacity.exe`。完整解读在 §5.2，这里是两轮的摘要：

**第一轮（v0.6，4 GiB 系统托管 pagefile，`bench/results/heap_capacity.csv`）**

| | 路径 A | 路径 B | 先 A 后 B |
|---|---|---|---|
| 拿到的 slab | 18 × 2 GiB = **36 GiB** | 18 × 2 GiB = 36 GiB | A 36 GiB + B **0** |
| 停下的原因 | `available commit < slab + floor` | 同 | 同 |
| 对 `availPhys` 的影响 | **≈0**（48.76 → 48.44 GB） | 1:1（46.58 → 9.83 GB） | |
| 对 `availCommit` 的影响 | **1:1** | 1:1 | |
| 分配后 raw-read 复核 | 216.0 GB/s | 208.5 GB/s | 211.1 GB/s |

commit 限额 = 67.65 GiB = 63.65 GiB RAM + 4 GiB pagefile，
**cache 停在 36 GiB = 2,056 槽**。

**第二轮（v0.7，C: 固定 98,304 MB = 96 GiB pagefile，commit 限额 159.6 GiB，
`bench/results/heap_capacity_idle.csv`）**

| | 路径 A | 路径 B | **先 A 后 B（实际布局）** |
|---|---|---|---|
| 拿到的 slab | 37 × 2 GiB = **74 GiB** | 20 × 2 GiB = 40 GiB | A **74** + B **26** = **100 GiB** |
| 停下的原因 | `vkAllocateMemory` 返回 `-2`（**device-local heap 的 74.4 GiB 到顶**） | `availPhys 6.68 GiB < 2 + floor 6` | 两个物理上限各自到顶 |
| 停下时剩余 commit | 61.8 GiB | 96.2 GiB | **35.6 GiB**——commit 已不是约束 |
| 满载后 raw-read | 214.4 GB/s | 206.4 GB/s | **216.1（A）/ 207.5（B）** |
| **折成 expert 槽** | | | **5,711 个 = 15,360 的 37%** |

**行动项已关闭。** `external_memory_host` 导入 3 / 6 / 12 / 24 GiB 全部成功，
**路径 B 不受 2 GiB 限制**（§5.3）。

复测命令（`--csv` 指向一个新文件，不要覆盖历史那一份）：

```powershell
.\build\heap_capacity.exe --slab-gib 2 --min-free-gib 6 --csv bench\results\heap_capacity_idle.csv
```

> **必须在空闲机上跑。** 同一套设置下先跑的一轮（`heap_capacity_pagefile128.csv`，
> 有另一个基准在并发）满载 raw-read 只有 **151.9 / 121.4 GB/s**，看起来像"装满之后带宽掉一半"；
> 空闲重测是 216.1 / 207.5。**容量测量和带宽测量适用同一条纪律**（build.md 的"测量纪律"）。

### 9.3 分层驻留策略

```
Pinned : attention / shared expert / router / mHC / norm / engram wkv
         / embed / head / mtp                （~17.7 GB，永不淘汰）
Cached : routed experts，slab 池           （**v0.7 实测 100 GiB / 5,711 槽 = 37%**，§5.2）
Cold   : 其余 routed expert + engram 表    （NVMe，原始分片）
```

**v0.6 定案（依据 §9.1.1 Q1/Q2）：**

1. **策略 = 全局 LRU。** LRU / LFU-decay / ARC 在四位小数上不可分，global 稳定胜过 per-layer
   0.6–0.8 个点。`store/planner.cpp` 的 LRU 基线就是最终形态：最简单、最好写、最好验。
   **不做按层配额。**
2. **score-aware 作为可选开关，默认关。** 它每个容量都赢，但只赢 0.20–0.23 个点，
   代价是每层 16 次 heat 更新加一个按 heat 排序的淘汰堆。做成编译期/配置开关，
   在 P3 的真机上再用 §9.8 的指标决定要不要打开。
   （还没试过的变体：只提升 top-16 里排名 7–10 的，以及把近似命中的权重降成一个远小于 1 的常数。）
3. **删掉 routed expert 的"静态 pin"子档。** Q1 的偏斜是真的，但 LRU 自己就吃掉了；
   `static-pin + LRU` 在每个容量上都差 0.4–2.7 个点。Pinned 层只留真正永驻的那些。
4. **`heat` 字段保留在 `ExpertSlot` 里**（§5.4），因为 score-aware 的开关要用它；
   默认路径不写它。

### 9.4 Lookahead 预取：**第一阶段不做**（v0.6 降级，依据 Q4 实测）

原方案是：CPU 在第 L 层 FFN 之后对层 L+1..L+d 的 gate 做 GEMV，取预测 top-K，未驻留的进 P1 队列，
预取落地的槽以"探针"身份插入 LRU 尾部。**把 Q4 的预测集接进模拟器之后，每一个 (d, K) 都是净负收益。**

基线（4,787 槽、global LRU、无预取）：hit 0.8968，165.7 ms/token，**6.03 tok/s**。
实测 2026-09-14，`reports/cache_prefetch_head.json` / `cache_prefetch_tail.json`：

| 探针位置 | 最好的 (d, K) | hit_rate | effective | ms/token | **tok/s** | precision |
|---|---|---|---|---|---|---|
| — | 基线 | **0.8968** | 0.8968 | **165.7** | **6.03** | — |
| `head`（当普通填充） | d=4, K=8 | 0.8399 | **0.9052** | 282.3 | **3.54** | 0.348 |
| `head` | d=4, K=16（最差） | 0.7761 | 0.8737 | 608.6 | 1.64 | 0.202 |
| `tail`（原 §9.4 字面） | d=2, K=8 | 0.8968 | 0.9015 | 393.9 | 2.54 | **0.016** |
| `tail` | d=6, K=16（最差） | 0.8968 | 0.8969 | 1036.3 | **0.96** | 0.000 |

**预取从两头伤害**：K 个预测里只有 2–3 个是对的，剩下的 (a) 烧 NVMe 带宽，
(b) 在 `head` 模式下还把真正有用的数据挤出去（hit 0.897 → 0.776）。
最好的一档把 effective hit rate 抬了 0.8 个点，代价是 NVMe 字节 +60%——**换不回来**。

**"探针插 LRU 尾部"这个写法是自毁的**：在满的 cache 里，LRU 尾部就是下一个被淘汰的位置，
下一次 admit 就把它踢掉，通常远早于它被取用。实测 precision 掉到 0.000–0.016，字节全浪费，
hit_rate 纹丝不动（污染不了 cache，也帮不上忙）。合成用例上这个差距是 **32 倍**，
已固定为 `tools/tests/test_cache_sim.py:test_probe_at_tail_is_self_defeating`，
**不要照抄进 `store/planner.cpp`**。

**决定：lookahead 预取不在第一阶段范围内。** `store/predictor.h` 的接口保留（它是纯函数、
零成本），`PrefetchConfig` 默认 `lookahead_depth = 0`。§8.1 的 CPU lookahead GEMV 一并取消。

**什么时候重新考虑**：这个结论的前提是**基线命中率已经 0.90**，每 token 只剩 ~25 个 miss
可抢。两种情况会推翻它——

1. **换一个更好的预测器**（当前 recall@6 在 d=1 只有 0.66、d=3 只有 0.52，且到 d=8
   都没有拐点）。最值得试的是**用 trace 训一个**：Q3 说相邻 token 的 Jaccard 是 0.239，
   而 d=8 的 recall@6 是 0.363，**同一量级**——"上一个 token 在 L 层选了谁"这个零成本基线
   就能和 8 层 lookahead 打平，说明现在的近似几乎没抓到跨层信息。
2. **命中率被拉回 0.6–0.7**（更大的模型、更小的 cache、跨语域的语料）。那才是原 §9.4
   针对的场景，届时整个 (d, K) 网格要重扫。

若将来重启，实现上还要先解决"探针该放在 recency 序列的哪里"：`head` 与 `tail` 都不对，
合理的中间方案是插在 LRU 栈中部，或给探针一个固定的保护期（比如 d 层）再降级——两者都还没实现。

### 9.5 与投机解码的耦合

- verify batch 的 M=k+1 个 token 在每层同时路由，一层的 expert 并集一次性告知 Planner，I/O 并行度天然提高。
- 草稿 token 一旦生成，其 **Engram 行地址完全确定**，立即预取（Q7 保证 ~ms 级到达）。
- **这是 §9.4 被砍掉之后唯一剩下的"预取"，而且它是精确的**：verify batch 的 expert 集合不是预测，
  是草稿 token 真实路由出来的结果，precision = 1。
- Q3 的 Jaccard 已实测（§9.1.1）：`union_frac[5] = 0.634`，k=5 的 batch 只多读 3.2 倍而不是 5 倍，
  **NVMe 流量省 37%**。纳入 §10.3 的调度曲线。

### 9.6 IoEngine

- `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)` + IOCP，专用 2 个完成线程。
- 一个 expert = 18.8 MB 切成 `chunk` 并发下发。**Q6 已实测（§9.2.1）：chunk = 4 MiB、目标 QD = 8（在途 32 MB）** 即达饱和 4.5 GB/s；更高的在途字节只增加延迟（4 MiB × QD64 仍是 4.56 GB/s，但平均延迟从 6.9 ms 涨到 37.7 ms）。默认值见 `core/config.h` 的 `IoConfig`。
- 目标缓冲直接是 slab 槽（路径 A：CPU 映射的 device 内存；路径 B：导入的 host 内存），**零拷贝**。
  **v0.6：已从假设变成事实**——`bench/kernel_bench` 每次启动都用真正的 `IoEngine` + IOCP +
  `FILE_FLAG_NO_BUFFERING` 把 56 个真实 expert（1053 MB）直接读进 slab，中间没有任何缓冲；
  实测路径 A **3.7–4.1 GB/s**、路径 B **4.8 GB/s**（路径 B 打满 §9.2.1 的盘上限；路径 A 差的那一截
  来自写合并内存的写入路径，与 §8.1 第 6 条同源）。`vkMapMemory` 与 `VirtualAlloc` 返回的指针
  天然页对齐，满足扇区对齐要求。`tests/test_gpu_moe.cpp` 每次运行都再验证一遍
  （读进去的字节要能算出 oracle 的答案）。这六轮的下单还是**串行**的（一个 expert 的两个 run
  发完才发下一个），Planner 的并发队列会更快。
- 优先级队列：P0 当前层缺失（阻塞 GPU）> P1 lookahead > P2 engram 行 > P3 后台回填（空闲时按静态热度填 free 槽）。P0 到达时可抢占：暂停下发 P1–P3 的新 chunk。
- 每次完成更新 Profiler：字节、延迟、队列深度；每 token 输出 `nvme_busy_ms`、`stall_ms`（GPU 在 timeline wait 上的时间）。
- 双盘：按各盘实测带宽比例分配 expert（静态哈希到盘），而非按盘数平均。**直读模式下这要求把分片复制到第二块盘**（每个分片整体 7.4 GB，复制粒度就是分片），manifest 的 `files[]` 记各自路径。
- **EOF 尾部**：manifest 的 run 允许越过文件末尾不足一个扇区（§5.1.4）。`ChunkRequest::min_bytes` 告诉 backend 真正必须到达的字节数；短读只在这一种情况下合法。

### 9.7 Prefill 流式模式

- prompt ≥ 阈值（由 Q1 与 prompt 长度推算的"预计激活 expert 数 > 容量"）时切换为 **expert-major 顺序流**：按分片内物理顺序读整层 7.22 GB（§5.1 实测一层不跨分片，仍是单文件顺序读，满带宽），每到达一个 expert 就对路由到它的 token 做 GEMM，然后决定它留在 cache 还是丢弃（按 decode 阶段的预测热度）。
- 短 prompt 走 decode 同一套按需路径。
- 无论哪种模式，encoder 20 层先做完，decoder 用 bounded replay（§11.2）只算最后 128 个 token。

### 9.8 验收指标（NVMe 部分）

| 指标 | 定义 |
|---|---|
| hit rate（每层 / 总） | 驻留命中 / 请求（**不含** hidden，与 `cache_sim` 的定义一致） |
| effective hit rate | （驻留命中 + hidden）/ 请求，即"GPU 没等"的比例 |
| miss bytes / token | 每 token 从 NVMe 读的 expert 字节 |
| ~~prefetch precision / recall~~ | §9.4 降级后无消费者；投机解码的 verify batch 预取 precision 恒为 1（§9.5） |
| stall_ms / token | GPU 等 expert 的时间。**P3 的对照值**：`cache_sim` 在 4,787 槽上给出 103 ms/token；**今天的 5,711 槽（h ≈ 0.920）按 §3.1 的模型是 ≈80 ms**（§13.4） |
| nvme_util | nvme_busy / T_token |
| 有效 NVMe GB/s | miss bytes / nvme_busy |
| **cache 槽数** | 启动时实测（**v0.7：上限是 device-local heap 与可用物理内存，不再是 commit 限额**；而且 pinned 的 17.7 GB 也要从同一个池子里出，§5.2.3）——**报告里必须带上它，否则 hit rate 无法对照 §3.1 的表** |

---

## 10. DSpark 投机解码

### 10.1 为什么关键

decode 每周期常驻 8.5 GB 读一次，若平均接受 a 个 token，则常驻部分的每 token 成本降为 8.5/a GB；head 1.32 GB 同理。在 NVMe 主导的区间，投机解码 + 高 hit rate 是唯二的杠杆。

**v0.6 量化（Q3 与 §7.9.1 实测）：摊薄是不对称的。**

| 项 | k=5 的 verify batch | 摊薄倍数 |
|---|---|---|
| 常驻 8.5 GB（attention / shared / head / router / mHC） | 读 1 次 | **÷5（完全）** |
| MoE kernel 时间（**v0.7 实测，7 槽含真 fp8 shared expert**） | **1.310 ms / 6 token = 0.218 ms/token**（M=1 是 **0.683**） | **÷3.1（实测）** |
| routed expert 流量（并集） | `union_frac[5] = 0.634` → 碰 3.2 倍于单 token 的 expert（逐层 2.6–4.1 倍） | **÷1.6（5 / 3.2）** |

也就是说：**常驻部分被完全摊薄，expert 流量只降到 1/1.6。**

**v0.7 的形势变了**：pagefile 已经修好，h ≈ 0.920，NVMe 项从每 token 的 81% 掉到
**≈51%**（80 ms / 156 ms，§13.4）。所以：

- **投机解码现在是第一杠杆**，因为它摊薄的那 75.8 ms 计算已经是每 token 的一半了；
- M=6 的 kernel 效率仍然是投机解码收益最大的一块 kernel 工作（它同时乘在上面两行上），
  但 §7.9.2 已经说清楚 **M=6 不是带宽受限而是 VALU 受限**，
  所以目标要换成 `ms/token`（现在 0.218，含真 shared expert），
  而不是"有效带宽 ≥ 80% 上限"。

注意 `union_frac` 假设草稿全被接受，是**保守端**：接受率低时并集只会更小。

### 10.2 周期流程

```
1. 主模型 forward(M=k+1: 上一接受 token + k 个草稿)
   ← 记录第 37/38/39 层**进入 block 之前**的残差流在 hc 维上的均值（§2.4 v0.6 更正）
2. 逐位验证 → 接受前缀 a ∈ [0, k]，得到 a+1 个新 token（含修正 token）
3. 草稿: forward_embed(main_hidden, last_token) → 3 个 DSparkBlock(M=5) → head → Markov 逐位采样 → confidence
4. 调度器按 confidence 选下一周期的 k ∈ [0, 5]
```

- **取的是 `h.mean(dim=2)`**（engram 之后、`hc_pre` / `attn_norm` **之前**），不是 attention 那个
  `[dim]` 输入的均值。参考 `generate.py` 没有调用 `forward_spec`，所以这一条只能靠读
  `model.py` 的 `if i in target_layer_ids: main_hiddens.append(h.mean(dim=2))` 确认
  （route_trace.md §11.5）。**实现前先对着这行代码核一遍。**
- 验证规则：greedy 模式下逐位比较 argmax；采样模式下用标准的 target/draft 概率拒绝采样（草稿 logits 由 DSpark head + Markov 偏置给出，需保存 5 位的完整分布或 top-k 近似 — 先实现 greedy，采样模式作为 P4 后半段）。
- **不变量**：`temperature=0` 时开关投机解码输出必须逐 token 相同，这是内建的正确性检查。
- 主模型的 window KV / 压缩 KV 状态在拒绝后要回滚：window cache 按位置覆盖即可；Compressor 的部分组状态需保存快照（ratio ≤ 2，状态很小）。

### 10.3 Confidence-scheduled 验证长度

调度器最大化 `E[accepted tokens] / T_cycle(k)`：

- `E[accepted](k) = Σ_{i≤k} Π_{j≤i} c_j`，`c_j` 来自 confidence head（每位条件接受概率）。
- `T_cycle(k)` 来自 **Profiler 在线测得的曲线**：`T_hot(M)`（常驻权重 GEMV/GEMM 随 M 的时间，M=1..6，近似平坦）+ `T_nvme(k)`（并集 expert 的期望 miss 字节 / 有效带宽，使用 Q3 的重叠统计与当前 hit rate）+ `T_draft`（≈ 11 ms）。
- **`T_nvme(k)` 的离线初值已有**（Q3，§9.1.1）：`union_frac[k] = 0.824 / 0.734 / 0.676 / 0.634`（k=2..5，40 层均值），
  即 `miss_bytes(k) ≈ (1−h) × k × union_frac[k] × 112.8 MB × 40 层`。逐层差别很大
  （层 0 的 `union_frac[5]` 是 0.818，层 25 是 0.518），所以在线 EWMA 仍然要按层维护。
- **`T_hot(M)` 的 MoE 部分已实测（v0.7 用含真 fp8 shared expert 的数）**：
  M=1 **0.683 ms/层**、M=6 **1.310 ms/层**（§7.9.2）——不是平坦的，但增长远慢于 M。
  中间值可按 §7.9.2 (a) 的 M 扫描内插，注意那张表的 7 个槽全是 FP4，
  换成真 shared expert 之后每层约再加 0.09（M=1）–0.34（M=6）ms。
- **`T_hot(M)` 的非 MoE 部分也已实测**：M=1 时 40 层 + head = **48.5 ms**（§7.15.2）。
  M > 1 的曲线还没测——它主要是权重复用，预期接近平坦，**但这是 P4 要实测的第一件事**。
- 曲线在运行中持续更新（EWMA），因此调度随 cache 状态和文本类型自适应。这正是技术报告中"profiled engine throughput curves"在单机上的对应物。

---

## 11. Prefill、CED 与 KV Cache

### 11.1 CED 的含义

- 层 0–19（encoder）对全部 prompt token 计算。层 20 是 kv source（ratio 1）：其 Compressor 直接把 **第 20 层的输入**（即 encoder 输出）投成 decoder 的全局 KV；层 21–39 复用这份 KV 与索引（Reuse / Reindex 模式由 `index_source_layer_ids` 决定）。
- decoder 各层仍有自己的 window KV，由各层自己的隐状态产生 → 需要 decoder 前向。

### 11.2 Decoder SWA Bounded Replay

- 生产模式：decoder 层只对 prompt 的最后 128 个 token 前向，window 截断到 replay 段；得到的 decoder window KV 仅用于 decode。这是近似（官方报告"对质量影响可忽略"）。
- Oracle 模式：decoder 对全部 token 前向（与 `model.py` 精确一致）。
- 两种模式都实现；benchmark 用生产模式并报告与 oracle 模式的输出差异率。

### 11.3 KV cache 结构（64K 上下文）

| 结构 | 每层 / 每源层 | 64K 总量 |
|---|---|---|
| window KV（40 + 3 层） | 128 × (512 B + 16 B scale) | 2.8 MB |
| 压缩 KV（源层 2, 8, 14 ratio 2；20 ratio 1） | C/ratio × 512 B（fp8）或 256 B + 32 B（fp4） | 84 MB fp8 / 48 MB fp4 |
| indexer K（8 源层） | C/ratio × (128 × 0.5 B + 128/32 B) = C/ratio × 68 B | **~29 MB**（v0.4 更正：原记 ~2 MB，那是*单个* ratio=2 源层的量；8 个源层中 5 个 ratio=1，合计 28.97 MB） |
| 候选块池（层 20 → 24..36） | 2048 块索引 | KB 级 |
| Engram hash cache | C × 8 B | 0.5 MB |

合计 ≈ 116 MB（fp8 压缩 KV）/ 80 MB（fp4）。KV 不是内存问题；全部放 GPU heap。上表由 `runtime/kvcache.h` 的 `KvGeometry` 按 config.json 实算，`tests/test_model.cpp` 回归。

### 11.4 Prefix KV 持久化

多轮对话中重复 prefill 一次要读 ~270 GB。因此把 encoder 输出（第 20 层输入，5120 × 4 份 fp16 = 40 KB/token）与压缩 KV、indexer K 按 prompt 前缀 hash 持久化到 `kvcache/`；命中时 encoder 跳过，只跑 decoder bounded replay（128 token）。这与官方"Encoder SWA Bounded Replay + 全局 KV 持久化"一致。64K 上下文的持久化体积 ≈ 2.7 GB，可接受。

---

## 12. 正确性 Oracle

没有可用的官方 CUDA 环境，也没有 GGUF/llama.cpp 参考。oracle 自建，分四层：

| 层级 | 内容 | 判据 |
|---|---|---|
| L0 解码 | FP4/FP8/E8M0 → fp32 解码函数与参考逐位相同。`oracle.py --level l0` 把三张表导出到 `tests/data/l0_dequant.bin`（2,132 B），`tests/test_dequant.cpp` 逐位比对 | 逐位 |
| L1 kernel | 每个 GPU kernel 对随机输入与 fp32 CPU 实现比较；**外加 expert FFN 的真权重版**（下）。**v0.6 实测**：`tests/test_gpu_moe.cpp` 把真实 expert `(0,0)` 与 `(39,383)` 经 IoEngine 读进 GPU slab，十四个 kernel 变体全部 cos = 0.99999996、max\|Δy\| = 1.37e-4（占 \|y\|max），且彼此完全一致；误差下界就是 `x` 的 fp16 舍入（相对 L2 2.04e-4），kernel 自身累加误差可忽略（§7.9.1）。**v0.7 补充**：fp8 shared expert cos = 0.999999947；`h` 的 fp8 量化对参考吻合到 5.0e-8；分组 dispatch 差 1–2 ULP（判据 1e-6，不是逐位）；packed fp16 4.52e-4；**int8 dot4 5.38e-3，略超判据**（§7.9.2） | 相对误差 ≤ 1e-3（fp16 传递）；~~int8 路径 ≤ 5e-3~~ **v0.7：5e-3 这个数没有实测依据，而 int8 激活的定价就是 ~5.4e-3。判据待 L2 重定（§15 未解决项）；在那之前 int8 只用在 dispatch B** |
| L2 逐层 | 用真实权重跑单层：`tools/oracle.py`（纯 torch fp32，从 safetensors 直接解码权重，逐函数对照 `model.py` 移植）vs deepMoE 每层输出。**v0.7：`--level l2` 已实现并跑通**——它跑的是**未经修改**的 `inference/model.py`（经 `tools/dsref.py` 的六个 CPU kernel shim），把七个层（0 / 1 / 2 / 13 / 14 / 20 / 39）每层 ~40–53 个张量导出到 `tests/data/l2/`（5.1 MB），逐级与整层的实测见 §7.15.1 | 每层输出余弦相似度 ≥ 0.999（**实测层 0 / 39 的 block 输出 0.999935 / 0.999980**），最大相对误差记录并画曲线。**判据用 cos 与相对 L2，不用 max 相对误差**——参考自己是 bf16/fp8，一个元素越过舍入边界会动它自己量级的半个 ulp（§7.15.1） |
| L3 端到端 | 多 prompt × 64 token greedy：deepMoE 与 `oracle.py` 全模型 CPU fp32 前向（每 token 数分钟，跑 ≥ 5 个 prompt）逐 token 一致；投机开/关一致；logits KL 记录 | token 一致率 = 100%（允许在极低 margin 处出现分歧并记录 margin） |

**L0 的三张表分别来自哪里**（信任锚点必须写清楚）：

- **FP8 E4M3 / UE8M0** 由 torch 本身生成（`torch.float8_e4m3fn` / `torch.float8_e8m0fnu`），这正是 `inference/kernel.py` 里 `T.Cast` 走的路径。
- **FP4 E2M1** 在 CPU 上 torch 无法转换（`copy_kernel not implemented for Float4_e2m1fn_x2`），所以 16 项表由 OCP E2M1 定义构造，并在 `ml_dtypes.float4_e2m1fn` 可导入时逐位交叉校验——实测**完全一致**：`{0, .5, 1, 1.5, 2, 3, 4, 6}` 加符号位，也与 `inference/kernel.py` 的 `fp4_max = 6.0` 吻合。
- **nibble 顺序**（低半字节 = K 方向的偶数元素）取自 PyTorch `float4_e2m1fn_x2` 的打包约定，`inference/kernel.py` 的 `B: [N, K//2] FP4, logical [N, K]` 依赖它。**L0 以下没有任何实验能证伪这个顺序**——在一个字节内交换两个 nibble，不会改变任何 32 元素 scale 块内的取值多重集，所以每块统计量完全相同。它作为**假设**记录在 `tools/oracle.py` 里，是 L2 出现偏差时第一个要翻的石头。

**L1 的真权重扩展（v0.5 新增）**：`oracle.py --level l1` 对给定 `(layer, expert)` 把 6 个 tensor **各读两遍**——一遍走 `deepmoe_manifest.json` 的 run/skew 算术，一遍走 `safetensors` 库——要求字节完全一致，然后在 torch fp32 里算 `w2(silu(clamp(w1 x,max=10)) * clamp(w3 x,±10))`，把 `x`、`y`、每个 part 的 64 位校验和与 sha256 写进 `tests/data/l1_layer{L}_expert{E}.bin`（各 41,528 B）。`tests/test_integration.cpp` 用**真正的 IoEngine + IOCP + ExpertStore** 把同一个 expert 填进槽，按指针表的六个地址重算校验和，再用 `cpu/gemv_fp4_ref` 复算 FFN 与 oracle 比对。这一条同时验证了：manifest 的 run/skew 算术、槽内布局、指针表的 skew 加法、FP4/E8M0 解码。当前实测见 §15 的 P0 行。

**L2 oracle 的产生方式（v0.7 新增，p2_attention.md §1）** —— 这一条决定了 L2 有多可信：

- **跑的是未经修改的 `inference/model.py`**，只借 `tools/dsref.py` 的六个 CPU kernel shim。
  **没有任何张量是重新推导出来的**：每一个导出的张量要么是某个 `nn.Module` forward hook 的
  输入或输出（`wq_a`、`q_norm`、`wq_b`、`wkv`、`kv_norm`、`attn_norm`、`ffn_norm`、`wo_b`、
  compressor 的 norm、indexer 的 `wq_b`/`wk`/`k_norm`/`weights_proj`），
  要么是参考实现路过的那几个模块级函数的实参或返回值
  （`act_quant`、`fp4_act_quant`、`sparse_attn`、`Block.hc_mixes`、`Block.hc_post`）。
- **RoPE 是唯一没有捕获点的阶段**：`apply_rotary_emb` 就地改写实参、不返回任何人会存的东西。
  它是**用差分捕的**——hook 克隆输出（pre-RoPE）并留住活的那个张量，
  `Attention.forward` 返回之后再读同一个张量就拿到 post-RoPE 的值，
  因为 `unflatten` 与 `x[..., -64:]` 都是 hook 看到的那块存储的视图。逆 RoPE 同理。
- **量化点是钉死的，不是假设的**：导出器从**量化前**的张量重算 fp8 E4M3 / UE8M0 与
  fp4 E2M1 / E4M3 的字节，并**断言**它们反量化回去正好等于参考产出的值。
  `fast_round_scale` 的舍入方向、E4M3 的 round-to-nearest-even、FP4 的 nibble 顺序
  **在这一个地方一起被钉死**。
- **两遍，不是逐层的"先 prefill 再 decode"**：`model.py` 的 `shared_attn` 是一个
  进程级单例，它**故意从不重置**（"每个源在消费者读之前都写过"）。decode 时一个 ratio-2 的源层
  如果它的压缩组在那个位置还不完整就不会发布，于是读到的是**上一次前向**的 index key
  （层 20 的，不是它自己的）。**参考就是这么做的，所以导出照样复现**：先跑完整一遍 prefill，
  decode 那一遍从它结束的地方开始。只有每层 ~200 KiB 的 KV 缓冲跨遍携带，
  每个 `Block` 都从分片重建（5.1 GB attention 权重，约 0.5 s 一层）。
- 整个导出 **258 s**，几乎全部花在 prefill 的 MoE 上（一层 ~200 个不同的 expert）。

- `oracle.py` 的正确性靠代码评审对照 `model.py` + 小规模合成模型的双实现一致性测试保证；这是无法绕开的信任锚点，要写清楚。
- 额外 sanity：与 DeepSeek 官方 API 同 prompt 的 greedy 输出比较，只作参考不作判据。

---

## 13. 评测与验收

### 13.1 每 token 时间分解（Profiler 必须输出）

`T_token = T_hot_gemv + T_expert_hit + T_attn_misc + T_dispatch_overhead + T_nvme_stall + T_cpu_sync`

每一项都有对应的上限模型（字节 / 实测带宽），报告实际 / 上限。

### 13.2 必测指标

| 指标 | 定义 |
|---|---|
| 正确性 | §12 四层 |
| 带宽矩阵 | CPU 单独 / GPU 单独 / 并发；device-local 与 host-import |
| kernel 有效 GB/s | 每个 GEMV 类 kernel vs raw read |
| hit rate / miss bytes / stall | §9.8 |
| 投机解码 | 平均接受长度、每周期时间、有效 TPS 增益 |
| decode TPS | 稳态，≥ 512 token，多种文本 |
| TTFT | 512 / 4K / 32K prompt，冷 cache 与 prefix 命中 |
| P50 / P95 token 延迟 | |

### 13.3 对照组

- llama.cpp Vulkan / HIP：在 `unsloth/DeepSeek-V4-Flash-GGUF`（同家族、284B、mmap 流式）和 Qwen3-30B-A3B 上对比；V4.1 本身无外部 runtime 可比。
- deepMoE 自身：GPU-only vs 投机开/关 vs prefetch 开/关 vs 各 cache 策略。

### 13.4 预期数字（**v0.7：三项输入现在全部是实测的**）

v0.6 的每 token 时间还是 §3.1 那个三项模型（42 ms 常驻 + 命中读 + NVMe）。
**v0.7 把前两项换成实测的 kernel 时间**，于是 decode 的账第一次是这样算出来的：

```
T_token ≈ T_非MoE + T_MoE + T_stall
        = 48.5 ms          # 40 层 dispatch 1–9（40 × 1.071）+ head，实测 §7.15.2
        + 40 × 0.683 ms    # MoE 两个 dispatch，7 槽含真 fp8 shared expert，实测 §7.9.2
          = 27.3 ms
        + (1−h) × 4.51 GB / 4.5 GB/s     # NVMe，用 §3.1/`cache_sim` 的方法
        + 0.40 ms          # ~600 个 dispatch × 0.66 µs（§7.14），可忽略
```

代入 h ≈ **0.920**（5,711 槽，§3.1 的 Q2 曲线内插）：
`0.080 × 240 个 expert × 18.80 MB = 361 MB`，**÷ 4.5 GB/s = 80 ms**，
合计 **≈ 156 ms/token → ≈ 6.4 tok/s**。

**这个算式的四条假设，必须和数字一起读**：

1. **完全串行，零重叠。** 和 §3.1 的模型一样是加法。真机上 Planner 会让下一层的取数
   与本层的计算重叠，所以这是**保守端**；但 §9.1.1 Q5 说提前量最多藏住一层的 1 个 miss
   （`d ≥ 3`），藏不住一层 6 个全 miss，所以重叠能拿回的不多。
2. **NVMe 用 4.5 GB/s 的平台值**，即假定 miss 的请求足够多、聚合 QD ≥ 8
   （§9.2.1 结论 5 的口径：一层 6 个 miss = 12 个 request / 36 个 chunk 同时在途）。
   换一个口径——一个 expert 单发 **4.26 ms**（§5.1.3，两个 run）——
   `0.080 × 240 × 4.26 ms = 82 ms`，**两个口径差 2%**，说明 expert 粒度上盘基本已经在平台上。
   真机若 QD 起不来，这一项会按比例变差。
3. **h = 0.920 是内插的**，没有单独跑模拟器（§3.1 表下的注）。
4. **h 有外推风险**：trace 的中文语料全部来自本仓库的设计文档，只有一种语域
   （route_trace.md §6"已知的语料局限"），跨语域的 h 可能更低。

| 场景 | 预期 | 依据 |
|---|---|---|
| **decode，无投机，今天的 5,711 槽（h≈0.920）** | **5.5–6.5 tok/s** | 上面的算式给 156 ms / 6.4 tok/s；留 10% 给真机开销。**§3.1 的旧模型口径给 143 ms / 7.0 tok/s**——差的 13 ms 是实测 kernel 比"字节 ÷ 上限"慢的那一截（§7.15.4） |
| decode，无投机，h 掉到 0.85（跨语域） | 4.0–4.5 tok/s | 同式，stall 从 80 ms 涨到 150 ms |
| decode，DSpark 平均接受 2.5，5,711 槽 | **9–13 tok/s** | 常驻 76 ms ÷2.5 = 30 ms（§10.1；**M>1 的非 MoE 曲线未测**，按平坦假设），expert 流量 ÷1.4（`union_frac[3]` = 0.734）→ stall ≈ 57 ms；MoE 按 §7.9.2 的 M=3 内插 |
| MoE kernel 有效带宽（M=1，7 个 FP4 槽） | ≥ 210 GB/s | 实测 222.6（§7.9.2），真机管线不应比它差 5% 以上 |
| MoE 每层时间（M=1，含真 shared expert） | ≤ 0.72 ms | 实测 0.683 ms；开 `HQuant=2` 会变成 0.749（+27%），**那个 27% 是一个已知的待修项**（§7.9.2 (f)） |
| 投机验证批的 MoE（M=6） | ≤ 0.23 ms/token | 实测 0.218（含真 shared expert）。**判据换成 ms/token，不再是"≥ 80% 上限"**（§7.9.2 (b)） |
| 非 MoE 路径，一层 | ≤ 1.1 ms | 实测 1.071 ms（§7.15.2），在 §3.4 模型的 0.97–1.29 ms 之内 |
| 每 token dispatch 开销 | < 0.5 ms | 实测 0.40 ms（~600 × 0.66 µs，§7.14） |
| stall_ms / token（5,711 槽） | 75–90 ms | 上面的两个口径给 80 / 82 ms |
| TTFT 4K prompt，冷 cache，单盘 | 40–55 s | §3.2，未重测 |
| TTFT 4K prompt，prefix 命中 | 1–2 s | 未重测 |
| 第二块 NVMe | 上述 NVMe 项 ×0.5 | 未验证 |

这些数字若实测偏离 30% 以上，必须在文档中解释原因。

**与 §1.3 成功标准 2 的关系**：那条要求"达到模型给出的上限的 ≥ 70%"。
上面的算式现在**就是**那个模型，而且它的每一项都有实测来源，
所以 P3 的验收变成一件很具体的事：把 Profiler 的四项分解贴到这四行上。

---

## 14. 工程技术栈

| 层 | 技术 | 原因 |
|---|---|---|
| 语言 | C++20 | RAII、`std::format`、协程可选、零开销 |
| 编译器 / 构建 | **Zig 0.16 (`zig c++`) + CMake + Ninja** | 单文件下载得到 clang 21 + lld + libc++；`-march=znver5`；可交叉编译 Linux 做 CI；MSVC 14.44 作为 A/B |
| GPU | Vulkan 1.3+ Compute | Windows gfx1151 主路径 |
| Shader | Slang → SPIR-V（Vulkan SDK 1.4.357 自带 `slangc`） | 泛型 kernel 模板、`WaveActive*`、int8 dot |
| CPU | AVX-512 / VNNI intrinsics | |
| I/O | Win32 Overlapped + IOCP | |
| 工具 | Python 3.12 + uv（numpy, safetensors, torch-cpu, pyarrow；`ml_dtypes` 可选，仅用于 L0 的 FP4 交叉校验） | manifest、oracle、trace、模拟、报告 |

构建、环境与验证命令见 [docs/build.md](build.md)。

### 14.1 目录

骨架已落地（2026-09-14）。顶层即模块目录，没有 `src/`；平台代码只存在于
`storage/windows/` 与 `storage/linux/`。模块依赖 DAG 与线程模型见
[architecture.md](architecture.md)。

```
deepmoe/
├─ CMakeLists.txt          根；选项 DEEPMOE_BUILD_TESTS / _ENABLE_VULKAN / _ENABLE_DIRECTSTORAGE
├─ cmake/
│  ├─ zig-toolchain.cmake  zig-{cc,cxx,ar,ranlib,rc}.cmd.in
│  └─ deepmoe_options.cmake  选项与各模块源文件清单
├─ docs/                   design.md  build.md  architecture.md
├─ core/                   types.h status.h align.h bytes.h log.h config.h
│                          json.{h,cpp}      自写的最小 JSON 读取器（无第三方依赖）
│                          profiler.{h,cpp}  每 token 时间线 → JSONL（§13.1）
├─ model/                  layout.h          编译期常量（kExpertBytes=18800640、kExpertSlotBytes=18808832 等）
│                          v41_config.{h,cpp}  config.json → struct（§2.1 全字段）
│                          manifest.{h,cpp}    deepmoe_manifest.json v2（run/skew）读取与自洽校验
├─ runtime/                engine.{h,cpp}    token 循环编排（仅编排）
│                          block.h attention.h moe.h engram.h dspark.h
│                          sampler.h kvcache.h
│                          rope.{h}          RoPE / YaRN，按层取 ratio（§7.3，v0.7）
│                          kvstore.{h,cpp}   window 环 + 压缩 KV 的持有者（v0.7）
│                          decode_layer.{h,cpp}  一整层 decoder 的编排（§7.15.3，v0.7）
│                          moe_bridge.{h,cpp}    gate ids → Planner → timeline（§7.1，v0.7）
├─ store/                  slab.{h,cpp}         slab 池 + SlabBacking 接口
│                          expert_store.{h,cpp} 槽状态机 + GPU 指针表（§5.3/§5.4）
│                          planner.{h,cpp}      LRU 基线已实现，其余策略待 P1
│                          predictor.h/.cpp     lookahead 接口（§9.4，待 Q4）
│                          engram_prefetch.h/.cpp  接口（§7.10）
│                          shard_set.h          48 个原始分片的打开与索引（§5.1，header-only）
│                          pinned.{h,cpp}       §2.2 的 17.7 GB pinned 集合：manifest + IoEngine
│                                               → GPU 可寻址内存，带 commit 检查（v0.7）
├─ storage/                file.h file_common.cpp  打开的文件抽象（unbuffered/扇区）
│                          backend.h               抽象后端 {submit, poll, caps}
│                          io_engine.{h,cpp}       优先级队列 + 切分 + QD 控制（§9.6）
│                          windows/ file_win.cpp iocp.cpp directstorage.cpp
│                          linux/   file_posix.cpp io_uring.cpp   （仅 CI 编译）
├─ gpu/
│  ├─ vulkan/              device.{h,cpp} memory.{h,cpp} timeline.{h,cpp}
│  │                       cmdbuf.{h,cpp} pipeline.{h,cpp} descriptor.{h,cpp}
│  │                       moe_kernels.{h,cpp}   §7.9 的两个 dispatch（§7.9.1/§7.9.2）
│  │                       attn_kernels.{h,cpp}  §7.2–§7.8/§7.11 的十八个 pipeline
│  │                                             + 共享地址表 + GpuScratch（§7.15，v0.7）
│  │                       rawread.{h,cpp}       上限对照 shader
│  └─ shaders/             §7.14 dispatch 清单，每个都过 slangc + spirv-val：
│                          attn_common  fp8_gemv                     （v0.7 新增，FP8 路径的公共部分）
│                          mega_mhc  wq_a  wq_b  wkv  sparse_attn  wo_a  wo_b
│                          gate  head
│                          moe_common  moe_gateup  moe_down  moe_gemv_fp4  rawread
│                          （attn 侧与 moe 侧的公共头**故意分开**：一个是 FP8 attention，
│                            一个是 FP4 expert，两条 track 各自演进）
├─ cpu/                    dequant.{h,cpp}      FP4/FP8/E8M0 解码（L0 oracle，已实现）
│                          gemv_avx512.{h,cpp}  标量参考已实现，AVX-512 为桩
│                          gate.{h,cpp}         router 数学（已实现）
├─ bench/                  nvme_bench.cpp     Q6/Q7 微基准（§9.2.1）
│                          bw_matrix.cpp      带宽矩阵：CPU / GPU / 并发（§8.0、§3.3）
│                          kernel_bench.cpp   §7.9 kernel sweep + dispatch 开销 + 路径 A/B
│                                             （§7.9.1 的 P1 旋钮用 `--p1`；默认是 §7.9.2 的 P2 sweep）
│                          attn_bench.cpp     §7.15.2 非 MoE 路径逐 kernel 带宽（v0.7）
│                          heap_capacity.cpp  两个 heap 的实际可分配上限（§9.2.2）
│                          results/           实测 CSV
├─ reports/                cache_sweep.json  cache_prefetch_{head,tail}.json（§9.1.1，gitignore）
├─ tools/                  envcheck/vkinfo.cpp
│                          manifest.py oracle.py route_trace.py cache_sim.py
│                          oracle_shared.py     fp8 shared expert 与 `h` 量化的四个参考答案（§7.9.2，v0.7）
│                          dsref.py corpus.py   参考实现的 kernel shim 与语料（route_trace.md §4/§6）
│                          tests/test_cache_sim.py  12 个可独立验算的模拟器用例
├─ tests/                  test_framework.h  自写的 header-only 测试框架
│                          fake_backend.h    确定性的 Backend 假实现
│                          l1_golden.h  l2_golden.h   黄金数据的读取（v0.7）
│                          test_core / test_dequant / test_store / test_io / test_model
│                          test_integration.cpp   真 checkpoint 端到端（DEEPMOE_MODEL_DIR 门控）
│                          test_gpu_core / test_gpu_moe（§7.9）
│                          test_gpu_attn.cpp      §7.2–§7.11 逐 stage 对 L2 oracle（v0.7）
│                          test_gpu_layer.cpp     一整层链起来（§7.15.1/§7.15.3，v0.7）
│                          data/v41_config.json   ModelScope 上的真实 config.json
│                          data/manifest_v2_slice.json  真 manifest 的切片（48 个 file 条目 + 9 个 expert）
│                          data/l0_dequant.bin    oracle L0 黄金表（2,132 B）
│                          data/l1_layer*.bin     oracle L1 黄金向量（各 41,528 B）
│                          data/l1_shared_layer0.bin  fp8 shared expert 的黄金向量（v0.7）
│                          data/l2/               oracle L2：七个层 × ~40–53 个张量，5.1 MB（v0.7）
└─ cli/                    deepmoe_main.cpp  `info` / `bench nvme` / `run`
```

**C++ 标准：代码写作 C++20，但必须用 `-std=c++23` 编译。** `std::expected`
在 `<expected>` 里，libc++ 把它整体挡在 `_LIBCPP_STD_VER >= 23` 之后。
`cmake/deepmoe_options.cmake` 里的 `DEEPMOE_CXX_STANDARD=23` 是唯一的开关。

## 15. 开发阶段

| 阶段 | 范围 | 产物 | 准出条件 |
|---|---|---|---|
| **P-1 可行性**（1–2 周）**已完成** | 带宽矩阵（CPU / GPU / 并发 × 两条路径）、NVMe 微基准、slab 分配 + `external_memory_host` 导入 + NVMe 直读进 GPU 内存、容量上限、kernel sweep、dispatch 开销 | [kernel_p1.md](kernel_p1.md)、`bench/results/*.csv` | ✅ 三组带宽、Q6/Q7、容量全部有数；直读零拷贝跑通；结论已写回 §3.3 / §3.4 / §5.2 / §7.9.1 / §8.0 / §9.2.2 |
| **P0 权重与 oracle**（2–3 周） | `manifest.py`（直读地址簿，替代 repack）、`model/manifest`、`oracle.py` L0/L1、C++ fp32 CPU 前向（无 GPU） | 能用 CPU 产出正确 token（慢） | L0–L3 全过；trace 工具可跑。**已完成：manifest v2、oracle L0/L1、`tests/test_integration.cpp` 对真实 checkpoint 的端到端读路径校验** |
| **P1 测量**（1 周，与 P0 后半并行）**已完成** | `route_trace` 跑 27,399 token / 40 prompt；`cache_sim` 出策略/容量/(d,K) 曲线 | [route_trace.md](route_trace.md)、`reports/*.json` | ✅ Q1–Q5 全部回答，写回 §9.1.1；cache 策略定为全局 LRU（§9.3）；lookahead 降级为不做（§9.4） |
| **P2 GPU 常驻路径**（3–4 周）**step 1 已完成** | §7 全部 decode kernel，M=1，所有 expert 假设驻留 | [kernel_p2_moe.md](kernel_p2_moe.md)（Track D）、[p2_attention.md](p2_attention.md)（Track E）、`bench/results/{kernel_p2_moe,attn_p2}.csv`、`tests/data/l2/` | ✅ **step 1**：九个非 MoE kernel 逐级过 L2（整层 block 输出 cos 0.999935 / 0.999980，§7.15.1），MoE 的 M 扫描 / fp8 shared expert / `h` 量化 / 分组 dispatch 全部有数（§7.9.2）；M=1 MoE kernel 222.6 GB/s（判据 ≥ 210）。**未完**：compressor / indexer kernel、40 层串起来、Engram 的 GPU 路径、head/sampler、L3。**M=6 的准出条件改写**：~~有效带宽 ≥ 上限的 80%~~ 在 VALU 受限的 kernel 上不是有意义的指标（§7.9.2 (b)），改为 **`ms/token` ≤ 0.23**（实测 0.218） |
| **P3 流式 decode**（3 周） | ExpertStore、Planner（全局 LRU）、IoEngine、Profiler。**不含 lookahead** | 端到端 decode，hit/miss/stall 报告 | h、stall 与 `cache_sim` 预测一致（±5 点）；NVMe 利用率 > 80% 在 miss 期；**报告里必须带上实测槽数** |
| **P4 DSpark**（2 周） | 草稿、greedy 验证、confidence 调度、KV 回滚 | 投机 decode | greedy 输出与非投机一致；TPS 增益可测 |
| **P5 Prefill/CED**（2–3 周） | cooperative matrix GEMM、expert-major 流式、bounded replay、prefix 持久化 | 完整对话 CLI | TTFT 数字；oracle 模式与生产模式差异率 |
| **P6 实验** | int8 dot4 路径、Wave64、双盘 stripe、路径 B 的 2 MiB 大页（需 `SeLockMemoryPrivilege`）、路径 B 的 >2 GiB slab | A/B 报告 | 只保留有数据支持的改动。~~CPU 分担 expert~~ 已被 §8.0 否决，移入 §16 |

**P-1、P1 与 P2 step 1 的结论已写回本文档（v0.7）。**

**v0.6 的四项优先级，现在的状态：**

| # | v0.6 的事项 | 状态 |
|---|---|---|
| 1 | 调大 pagefile 并重跑 `heap_capacity` | ✅ **已完成**：100 GiB / **5,711 槽**（比目标的 4,787 还多 924），h ≈ 0.920（§5.2） |
| 2 | DSpark 投机解码，其中"M=6 kernel 的 x 分块进 LDS"是最高优先级 kernel 项 | ⚠️ **那个 kernel 项已作废**（x 分块进 LDS 在每个 M 上都更慢，§7.9.2）。M=6 换成了 packed fp16 + B int8 dot4，0.162 ms/token。DSpark 本身仍未开始 |
| 3 | 端到端流式 decode，Planner 用全局 LRU | 未开始（P3） |
| 4 | 不做 lookahead 预取 | ✅ 维持 |

**P2 step 2 起的优先级：**

| # | 事项 | 为什么排这里 |
|---|---|---|
| **1** | **compressor 与 indexer kernel（§7.4）** | 它们是"一个能独立成立的 decode step"与今天之间**唯一的缺口**：压缩 KV 与 top-k 现在是从 oracle 的 prefill 加载进来的（§7.15.3） |
| **2** | **把 40 层串起来 + Engram 的 GPU 路径（§7.10）+ head/sampler（§7.11）+ L3 token 级 oracle** | 这就是 **"第一个 token"** 那个里程碑。到这里为止 deepMoE 才第一次自己产出一个 token |
| **3** | **DSpark 投机解码（P4）** | h 修到 0.92 之后 NVMe 只占每 token 的一半（§13.4），**投机解码变成第一杠杆**（§10.1）。前置是 2 |
| **4** | `HQuant=2` 的 32 行约束（§7.9.2 (f)） | **kernel 侧收益最大的一项**：M=1 因此损失 27% = **6.3 ms/token**。修法是把 `h` 的写出与计算的 workgroup 形状解耦，或拆第三个极小的 dispatch |
| **5** | `wo_b` / `wq_b` 的 K-split（§7.15.4 第 1 项） | 一层 133 MB 里的 84 MB，一层只有 57% 上限主要因为它们 |
| **6** | **端到端流式 decode，Planner 用全局 LRU**（score-aware 作为默认关闭的开关） | 策略已由 §9.1.1 定案，剩下的是接到真机上并用 §9.8 的指标对照 `cache_sim`。**分组 dispatch 的启用判据在这里落地**（§7.9） |

**v0.7 留下的未解决问题**

| # | 问题 | 需要谁来定 |
|---|---|---|
| 1 | **`HQuant=2` 的 32 行约束**，M=1 损失 27%（6.3 ms/token） | kernel（见上表第 4 项） |
| 2 | **x 在 kernel 之外预量化成 int8**（§7.9.2 (d)）：模型算下来 dispatch A 能到 ~187 GB/s，是唯一能让 M=6 摸到 85% 的设计，代价是 ~1% 输出误差 | **L2**：这个误差能不能接受，kernel 层判断不了 |
| 3 | **§12 的 int8 判据 5e-3 没有实测依据**，而 int8 激活的定价就是 ~5.4e-3 | **L2**：要么按实测重定判据，要么放弃 int8 走 packed fp16 |
| 4 | **分组 dispatch 的启用条件**（§7.9）：需要 Planner 的一个"预计等待 > 0.2 ms"判据 | P3 |
| 5 | A 与 B 的 `RowsPerLane` 也该分开（§7.9.2 (c)），预估收益 < 2% | 低优先级 |
| 6 | **run 间 ~7% 的漂移**（§7.9.2 开头），比 P1 记的 1.5% 大 | CI：kernel 带宽回归必须同轮对照，不能比绝对值 |
| 7 | 压缩 KV 现在存 bf16 而不是 §11.3 的打包 FP4，64K 上值 18 MB | 等 compressor kernel 能写出打包形式 |
| 8 | M > 1 时非 MoE 路径的时间曲线未测（§10.3 按平坦假设） | P4 的第一件事 |

**P0 已完成的部分（2026-09-14，v0.5）**

| 项 | 结果 |
|---|---|
| `tools/manifest.py` 扫全部 48 个分片 | **0.9 s**（header 0.10 s，建表 0.43 s），输出 9.9 MB manifest；一致性：分片字节数、header 覆盖范围、`index.json` 的 `weight_map` 与 `total_size = 510,286,023,000` 全部吻合 |
| runs/expert 直方图 | **2 : 15,744**（无一例外），最大 run 17,698,816 B，`kExpertSlotBytes = 18,808,832` |
| 跨分片 | expert 0 个、层 0 层 |
| oracle L1 `(0,0)` / `(39,383)` | manifest 路径与 safetensors 路径**六个 tensor 字节完全一致**；`‖y‖₂ = 168.288` / `180.752` |
| `tests/test_integration.cpp`（真 checkpoint） | 六个 part 的校验和与 oracle 一致；C++ `gemv_fp4_ref` 复算 FFN 对 oracle：`(0,0)` cos = 1.000000000000、`max\|Δy\| = 8.58e-6`（输出尺度的 4.5e-7）；`(39,383)` cos = 0.999999999999、`1.10e-5`（1.2e-6）。差异只来自 2304/5120 项求和顺序 |

**P2 step 1 已完成的部分（2026-09-14/15，v0.7）**

| 项 | 结果 |
|---|---|
| `tools/oracle.py --level l2` | 跑**未修改**的 `inference/model.py`，七个层 × ~40–53 个张量 → `tests/data/l2/`（5.1 MB，258 s）；量化字节是**断言**出来的不是假设的（§12） |
| `tools/oracle_shared.py` | fp8 shared expert 与 `h` 量化的四个参考答案（`y_ref` / `y_hq` / `y_full` / `y_hq16`），§7.9.2 (e)(f) |
| `gpu/shaders/`：九个 attention 侧 kernel + `attn_common` / `fp8_gemv` | 全部过 `slangc` + `spirv-val`；`gpu/vulkan/attn_kernels.{h,cpp}` 十八个 pipeline + 共享地址表 |
| `store/pinned.{h,cpp}` | §2.2 的 pinned 集合经 manifest + IoEngine 进 GPU 可寻址内存，带 `GlobalMemoryStatusEx` 的 commit 检查（失败时直接指向 build.md 的 pagefile 一节） |
| `runtime/{rope.h,kvstore,decode_layer,moe_bridge}` | 一整层 decoder |
| 逐 stage 正确性（`tests/test_gpu_attn.cpp`） | 十二个 stage，最差 cos 0.99991（`wkv` 过 fp8 之后）；`attn_norm` **七层逐位相同**（§7.15.1） |
| 整层正确性（`tests/test_gpu_layer.cpp`） | 层 0 / 39 的 block 输出 cos **0.999935 / 0.999980**，gate **6/6** |
| 速度（`bench/attn_bench`） | 一层 dispatch 1–9 = **1.071 ms / 124 GB/s**；40 层 + head = **48.5 ms**（§7.15.2） |
| MoE 的 P2 sweep（`bench/kernel_bench`，261 行） | M 扫描、`XMode` 六个变体、`HQuant` 两种实现、fp8 shared expert、分组 dispatch（§7.9.2） |
| MoE 每层真实时间 | **M=1 0.683 ms / M=6 1.310 ms**（7 槽含真 fp8 shared expert），替换 §7.9.1 的 0.602 / 0.974 |
| 容量（`bench/heap_capacity`） | **100 GiB = 5,711 槽**，满载后 raw-read 216.1 / 207.5 GB/s（§5.2） |

**还没做的**（p2_attention.md §8）：compressor / indexer kernel（§7.4）、Engram kernel（§7.10）、
prefill（§7.13）、DSpark（§7.12）、GPU 上的采样（§7.11 的 Philox / Gumbel-max 那一半）、
`KvCache::snapshot` / `rollback`（§10.2）与 prefix 持久化（§11.4）；
**pinned 集合到现在都是测试一层一层加载的，17.7 GB 一次性加载还没人测过。**

---

## 16. 明确不做

- 不兼容其他模型架构；不做通用 GGUF 加载器。
- 不做 OpenAI API / Web UI / Agent。
- 不做多用户、高并发、EP/TP。
- 不做通用 CUDA/HIP/Vulkan 抽象；HIP 只作 kernel 对照。
- 不重新量化任何权重；不改 router 语义、不改 expert 精度。
- 不接受无法复现的"快很多"；所有优化保留 A/B 与原始数据。
- 不在 P-1、P1 的测量结论出来之前写 Planner 策略代码。（v0.6：结论已出，见 §9.3。）
- 第一阶段不实现 vision encoder、1M 上下文、采样模式的投机解码（greedy 先行）。
- **不做"CPU 分担 expert GEMV"**（v0.6 新增，原 §15 P6 的一项）。实测内存控制器是单一
  ~217 GB/s 共享上限，CPU 与 GPU 并发时合计仍是 214–216 GB/s（§8.0）。把权重挪给 CPU 算
  不增加总吞吐，只会让 GPU 少拿 14% 并交给一个每字节更慢的执行体。除非换成双内存控制器
  或别的硬件前提，否则这条不再讨论。
- **不做 lookahead 路由预取**（v0.6 新增）。实测在 27,399 token 的 trace 上每个 (d, K)
  都是净负收益（§9.4）。重启的条件写在 §9.4，不要在没有更好的预测器之前重做一遍。
- **不提议调 BIOS VGM**（v0.6 新增，**v0.7 理由更新**）。两个 heap 是同一条 LPDDR5X，带宽相同。
  v0.6 的理由是"容量受 commit 限额约束"；commit 解除之后（§5.2）理由换成
  **"容量 = VGM + (可见物理内存 − 余量)，总量守恒"**——调 VGM 只改变 A/B 的划分，不改变总量。
  任何"最小 VGM 下重测"的条目仍然作废。
- **不再提"调 pagefile"**（v0.7 新增）。它已经做完了（96 GiB），
  两个约束（device-local heap 74.4 GiB、可见物理内存）都顶到了物理量，**软办法用完了**。

---

## 附录 A：Tensor 清单与字节预算

按组件统计（来自 48 个 safetensors header，`total_size = 510,286,023,000`）：

| 组件 | 字节 | dtype |
|---|---|---|
| main/routed_experts | 288.78 GB | I8（打包 FP4）+ F8_E8M0 |
| main/engram | 203.07 GB | F8_E4M3 + F8_E8M0（wkv、q/k 权重 bf16 极小） |
| mtp（DSpark 3 块） | 7.93 GB | 同主模型 |
| main/attn | 5.11 GB | F8_E4M3 + F8_E8M0 |
| main/shared_experts | 1.42 GB | F8_E4M3 + F8_E8M0 |
| embed / head | 1.32 / 1.32 GB | BF16 |
| vision / aligner | 0.82 / 0.15 GB | BF16 |
| main/router (gate) | 0.16 GB | BF16 + F32 bias |
| main/hc | 0.16 GB | F32 |
| main/indexer | 0.05 GB | |
| dtype 合计 | I8 278.6 / F8_E4M3 204.0 / F8_E8M0 23.6 / BF16 4.0 / F32 0.2 GB | |

单层典型 tensor（layer 5）：

| tensor | shape | dtype | 字节 |
|---|---|---|---|
| attn.wq_a.weight / .scale | [1280, 5120] / [40, 160] | F8_E4M3 / E8M0 | 6,553,600 / 6,400 |
| attn.wq_b.weight / .scale | [32768, 1280] / [1024, 40] | | 41,943,040 / 40,960 |
| attn.wkv.weight / .scale | [512, 5120] / [16, 160] | | 2,621,440 / 2,560 |
| attn.wo_a.weight / .scale | [8192, 4096] / [256, 128] | | 33,554,432 / 32,768 |
| attn.wo_b.weight / .scale | [5120, 8192] / [160, 256] | | 41,943,040 / 40,960 |
| ffn.experts.E.w1 / w3 .weight | [2304, 2560] | I8 = 2×FP4 | 5,898,240 |
| ffn.experts.E.w2.weight | [5120, 1152] | I8 = 2×FP4 | 5,898,240 |
| ffn.experts.E.w{1,3}.scale / w2.scale | [2304, 160] / [5120, 72] | E8M0 | 368,640 |
| ffn.gate.weight / bias / bias_vl | [384, 5120] / [384] / [384] | BF16 / F32 | 3,932,160 |
| hc_attn_fn / hc_ffn_fn | [24, 20480] | F32 | 1,966,080 |
| hc_*_base / hc_*_scale | [24] / [3] | F32 | |
| attn.attn_sink | [64] | F32 | |
| engram.embed.weight / .scale（层 1） | [384,006,168, 256] / [384,006,168, 8] | F8_E4M3 / E8M0 | 98.3 GB / 3.07 GB |
| engram.wkv.weight / .scale | [25600, 6144] / [800, 192] | F8_E4M3 / E8M0 | 157,286,400 |
| engram.q_weight / k_weight | [4, 5120] | BF16 | |
| mtp.N.ffn.gate.bias | [128] | F32 | DSpark 128 expert |

FP4 打包：`I8 [rows, K/2]`，每字节 2 个 E2M1（低 nibble = 偶数元素，按 PyTorch `float4_e2m1fn_x2` 约定，P0 用 oracle L0 验证），scale `[rows, K/32]` E8M0 = 2^(e−127)。FP8 块 scale `[rows/32, K/32]`。

## 附录 B：权重下载

- 源：ModelScope `deepseek-ai/DeepSeek-V4.1-Flash`，48 个 safetensors 共 510.3 GB，含 `mtp.*`（DSpark）、`inference/`、技术报告。
- 目标：`D:\models\DeepSeek-V4.1-Flash`（D: 671 GB 空闲）。
- **必须直连，不走代理**：开发机 shell 环境有 `HTTP_PROXY/HTTPS_PROXY=127.0.0.1:7078`，下载前清空；WinHTTP 系统代理为"直接访问"。实测直连 ModelScope CDN 8 并发 ≈ 115 MB/s。
- 命令见 [docs/build.md](build.md#模型权重下载)；完成后 `tools/manifest.py` 用 `model.safetensors.index.json` + 各分片 header 校验大小并生成 `deepmoe_manifest.json`（`--verify` 另外并行算 48 个分片的 sha256）。**原始分片不动，只往目录里写这一个文件。**

## 附录 C：修订记录

| 版本 | 变更 | 依据 |
|---|---|---|
| v0.1 | 初始 docx 方案 | |
| v0.2 | CPU+GPU 协同降为待验证假设；补 Windows UMA 拓扑、2 GiB 分配限制、GGUF 对照、oracle 定义、attention/KV/prefill、cost model 查表、单盘、带宽利用率、MTP、工具链 | 开发机实测 |
| v0.4 | 代码骨架落地（§14.1 目录重写、新增 [architecture.md](architecture.md)）；**Q6/Q7 实测写回 §9.2.1**，据此把 chunk 定为 4 MiB、目标 QD 定为 8（原 §9.6 写的"在途 ≥ 64 MB"改为 32 MB：实测 64 MB 在途只增延迟不增吞吐）；**更正 §11.3 indexer K 体积 ~2 MB → ~29 MB**（原值是单个 ratio=2 源层的量）；记录 C++20 代码必须以 `-std=c++23` 编译（libc++ 把 `std::expected` 挡在 C++23 之后） | `bench/nvme_bench` 实测、`tests/test_model.cpp` 回归 |
| v0.5 | **取消 repack：§5.1 整节重写为"`deepmoe_manifest.json` 是原始 48 个分片的地址簿"**。理由：D: 只剩 196 GB，第二份 510 GB 放不下；权重只有一份；不承担重下风险。代价已量化（每 expert 2 次读而非 1 次、+8,192 B = +0.044%、新增的 1.06 MiB scales chunk 低于 §9.2.1 的 2 MiB 平台，最坏 +6%、实际在噪声里）。连带：`tools/repack.py` 删除并由 `tools/manifest.py` 取代；manifest schema 升到 v2（run/skew/slot_offset）；`kExpertSlotBytes = 18,808,832` 进 `model/layout.h`；ExpertStore 指针表变成每 expert 6 项；`ChunkRequest::min_bytes` 让越过 EOF 不足一扇区的读合法（§5.1.4）；§7.9 的 `w1/w3 行交错` A/B 作废；§12 的 L0 golden 表与 L1 真权重 expert FFN 落地，并新增 `tests/test_integration.cpp` | 48 个分片 header 全量统计、`tools/manifest.py` 实测、`tools/oracle.py` L0/L1、`tests/test_integration.cpp` |
| **v0.6** | **P1 两条 track 的实测全部写回。**（a）**内存系统**：CPU 与 GPU 共用一个 ~217 GB/s 的读带宽上限（GPU 单独 216、CPU 单独 101、并发 GPU 185 + CPU 29 = 214），raw-read 在路径 A / 路径 B / 纯 DEVICE_LOCAL 上完全相同 → §3.3 重写、§8 新增 §8.0 并**否决"CPU 分担 expert GEMV"**（移入 §16）。（b）**BIOS VGM 与本项目无关**：同一条 LPDDR5X，GPU 已能寻址两个 heap；§3.3 / §9.2.1 / §15 里所有"最小 VGM 下重测 / 需要重启"的条目删除，改为"slab 池在当前设置下横跨两种内存"（§5.2/§5.3）。（c）**容量的真实约束是 Windows commit 限额**：路径 A 的 slab 不吃物理内存（`availPhys` 基本不动）却 1:1 吃 commit，限额 = RAM + pagefile = 67.65 GiB，于是路径 A 停在 36 GiB、路径 B 随后一个槽都拿不到，合计 **36 GiB = 2,056 个 expert**（13.4%）→ 新增 §9.2.2、§5.2 重写，修复办法（调大 pagefile，VGM 支撑的页永不写入 pagefile）写进 build.md。`external_memory_host` 导入实测支持 >2 GiB（3/6/12/24 GiB），路径 B 的 slab 不必切成 2 GiB（§5.3）。（d）**kernel**：M=1 最优变体 218.5 GB/s = raw-read 的 **102%**、`T_layer(MoE) = 0.602 ms`；FP4 常量表解码是唯一"必须选对"的旋钮（算术解码只有 77%）；`h` 用 fp16（fp32 慢 9%）；M=6 只有 63%、瓶颈是激活读取，x 分块进 LDS 是头号待办；正确性 cos 0.99999996、max\|Δy\| 1.37e-4（误差下界就是 x 的 fp16 舍入）→ §7.1 rule 2–6 修订、新增 §7.9.1；**dispatch + barrier 实测 0.56–0.66 µs 而不是猜的 5–20 µs**，每 token 0.31 ms = 0.5% → §3.4 重写，"按到达顺序分组算 expert"从备选变成默认；raw-read shader 不再是有意义的上限。（e）**参考实现的四处出入**（route_trace.md §11）：Sinkhorn **以列归一化结尾、不是双随机**（行和偏差 8.5e-2）→ §2.4 更正并警告 §7.2 的 kernel；MoE 中间激活进 `w2` 前**必须按行 block-32 量化成 fp8 E8M0** → §7.9 补上；压缩 KV（FP4/block-16/E4M3 scale）与 indexer（FP4/block-32/E8M0）格式不同 → §2.4；DSpark 取的是**进入 block 之前**的残差流均值 → §2.4/§10.2。（f）**路由 trace（27,399 token / 40 prompt）**：§3.1 的假设表换成实测的容量-命中率曲线（768→0.552 … 7,680→0.954），冷 miss 0.25%，重用距离 p50 600 / p90 4,815；LRU = LFU-decay = ARC（四位小数不可分），score-aware 只赢 0.2 点，global 胜 per-layer 0.7 点，static-pin 差 0.4–2.7 点 → §9.3 定为**全局 LRU**、删掉静态 pin 档；Q1 偏斜真实（top-10% 覆盖 34–63%）但 LRU 自己吃掉了；`gate_bias` 改变 37% 的 top-6 选择；Q3 相邻 token Jaccard 0.10–0.36、`union_frac[5] = 0.634` → 投机解码对常驻完全摊薄、对 expert 流量只降到 1/1.6（§10.1、§3.1）；**Q4 lookahead 预取在每个 (d, K) 上都是净负**（最好 3.54 tok/s vs 基线 6.03，"探针插 LRU 尾部"更是自毁）→ **§9.4 降级为不做**，移入 §16。（g）§13.4 的预期数字全部换成有实测基础的值；§15 给出 P2/P3 的优先级：pagefile → 投机解码（M=6 kernel）→ 端到端 LRU → 不做预取 | `bench/results/{bw_matrix,kernel_p1,kernel_p1_patha,kernel_p1_pathb,heap_capacity}.csv`、`reports/cache_sweep.json`、`reports/cache_prefetch_{head,tail}.json`、`traces/verify/verify.json`；报告 [kernel_p1.md](kernel_p1.md)、[route_trace.md](route_trace.md) |
| **v0.7** | **P2 step 1 两条 track 的实测全部写回。**（a）**容量问题已解决**：C: 的 pagefile 设成固定 96 GiB，commit 限额 67.65 → **159.6 GiB**，slab 池从 36 GiB / 2,056 槽变成 **100 GiB / 5,711 槽（37%）**——路径 A 拿到真正的 **74 GiB**（撞 device-local heap 的 74.4 GiB，不再是 commit），路径 B 在它之上叠 **26 GiB**（撞可用物理内存下限；单独跑是 40 GiB）；满载后 GPU raw-read **216（A）/ 207（B）GB/s，没有任何惩罚**。顺带钉死机制：路径 A 的**前 60 GiB 正好落在 BIOS VGM 的 64 GB 里、一个字节物理内存都不占**，再往上 1:1 吃可见内存，所以总量 = VGM + (可见内存 − 余量)，**守恒**（§3.3 第 3 点仍成立，但 VGM 现在决定 A/B 的划分）。**不需要 D: 上的 pagefile。** h 随之到 ≈0.920 → §3.1 的"今天"行、§5.2 重写、§9.2.2、§13.4、build.md。**一条量测卫生**：同设置下先跑的一轮满载 raw-read 只有 152 / 124 GB/s，看起来像"装满就掉一半"——那是并发基准的争用，不是容量的代价（`heap_capacity_pagefile128.csv` vs `heap_capacity_idle.csv`）。（b）**Track D（MoE kernel，[kernel_p2_moe.md](kernel_p2_moe.md)）**：M 扫描 M=1 222.6 → M=6 135.3 GB/s，`ms/token` 0.591 → 0.162；**§7.1 rule 6 的"x 分块进 LDS"实测在每一个 M 上都更慢**（M=6 时 dispatch B 掉 27%，代价是每 K-chunk 两个 barrier 把访存流水排空）→ 规则重写为"LDS 只用在能省指令的地方，绝不带 per-K-chunk barrier"；**M ≥ 3 是 VALU 发射受限**（dispatch A = `0.33 + 0.055·M` ms，对得上指令数），所以尺子换成 `ms/token`，§15 P2 的 M=6 准出条件据此改写；packed fp16 赢 dispatch A（+14%）、int8 dot4 赢 dispatch B（+26%），**int8 给 A 用会超出 §12 的 5e-3 判据（5.4e-3），所以 B-only**；要摸到 ≥85% 只剩"每 token 把 x 预量化成 int8"一条路（~1% 误差，**记为待决问题**，需要 L2 定）；fp8 shared expert 在 M=1 上打满 100% 上限；**每层 MoE 的真实时间 0.683（M=1）/ 1.310（M=6）ms 替换 §7.9.1 的 0.602 / 0.974**（第七个槽从 12.5 MB 的 FP4 顶替换成 23.6 MB 的真 fp8）→ §7.9.1/§9.4 Q5/§10.1/§10.3/§13.4；**§7.9 的"拆分 dispatch 免费"错了 340 倍**（每层 +0.193 ms = 7.7 ms/token，因为 dispatch B 的 640 个 workgroup 要重跑一遍归约）→ 改为"只在真的要等 I/O 时才分组"，§3.4 同步更正；`HQuant=2` 在 M=1 上要 +27%（0.591 → 0.749，根因是被迫从 L32 R1 换到 L16 R2），**是 kernel 侧剩下收益最大的一项**；分组 dispatch 是 1–2 ULP 而不是逐位相同（判据 1e-6）。（c）**Track E（非 MoE decode 路径，[p2_attention.md](p2_attention.md)）**：L2 oracle 落地（七层 × ~40–53 个张量，hook 挂在**未修改**的 `inference/model.py` 上，fp8/fp4 字节是**断言**出来的；因为 `shared_attn` 是进程级单例所以必须跑两遍）；逐 kernel 的精度/带宽表；一层（stage 1–9）**1.071 ms / 124 GB/s**，40 层 + head **48.5 ms/token**；端到端层 0/39 的 block 输出 cos **0.999935 / 0.999980**、gate 6/6；什么是真的、什么是加载进来的（压缩 KV 与 indexer top-k 是加载的，compressor/indexer kernel 还没写）。**参考实现强制的八处更正**：(1) `linear()` 在**每一次** fp8/fp4 GEMM 前把激活量化成 fp8 E4M3 block-32（`wq_a`/`wq_b`/`wkv`/`wo_b`/`indexer.wq_b`），§6 补全清单；(2) **`wo_a` 是例外**（bf16 einsum）；(3) **RoPE 配对的是相邻元素 (448+2j, 448+2j+1)**，不是 (d, d+32) → §7.3 更正；(4) `hc_post` 收缩的是 comb 的**第一个**下标 → §7.7；(5) RoPE base / YaRN **按层**取，ratio 0 → YaRN 关 + `rope_theta`；(6) **RMSNorm 的输出必须先舍入到 bf16 再进 `act_quant`**（在量化器正前方"比参考更精确"是负收益）；(7) **mega_mhc 是 3 个 dispatch、sparse_attn 是 2 个**（`p` 的 bf16 舍入要对最终行最大值、sink 不参与竞争 max）→ §7.2/§7.5/§7.14 的 dispatch 数从 11 改为 **14–15**，§3.4 的算术同步；(8) command buffer 现在是**每层一个**，每 token 预录制需要地址表在 shader 里按层索引（P3，值 ~0.5%）。另有三个值得升格为 §7.1 规则的性能陷阱：被 push constant 作上界的循环里的局部数组会落 VRAM（216 µs → 2.8 µs）、激活进 LDS 要存 bf16 而不是 E4M3 字节、**任何 argmax 都不许写成单 lane 串行扫描**（56 µs → 12.6 µs）。（d）§13.4 的端到端预期改用实测 kernel 时间：48.5 + 27.3 + stall 80 ms ≈ **156 ms/token ≈ 6.4 tok/s**，四条假设写在算式旁边。（e）§15：P2 step 1 关闭，下一步是 compressor/indexer → 40 层 + Engram + head/sampler + L3（"第一个 token"）→ DSpark；八个未解决问题成表 | `bench/results/{kernel_p2_moe,attn_p2,heap_capacity_idle,heap_capacity_pagefile128}.csv`、`tests/data/l2/`；报告 [kernel_p2_moe.md](kernel_p2_moe.md)、[p2_attention.md](p2_attention.md) |
| v0.3 | 目标模型锁定 DeepSeek-V4.1-Flash 并解剖；结论改为 **NVMe 主导**；pinned/cached/cold 三层与 slab 池；精度锁定原生 FP4/FP8；按 V4.1 结构逐 kernel 设计（mHC、MQA-latent 稀疏 attention、分组 O 投影、FP4 融合 MoE、Engram、DSpark、head）；每 token 单 command buffer + timeline semaphore；NVMe 测量问题 Q1–Q7 与 trace/模拟工具；lookahead gating 预取；expert-major 流式 prefill；DSpark 验证循环与 confidence 调度；CED / bounded replay / prefix 持久化；四层自建 oracle；阶段重排（P-1、P0、P1 测量前置）；CPU 协同降为 P6；写下预期数字 | ModelScope 权重 header 统计、`inference/model.py`、V4.1 技术报告、工具链实测 |
