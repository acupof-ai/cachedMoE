# deepMoE 设计方案

> Windows Strix Halo 专用超大 MoE 本地推理 Runtime
> 目标模型：DeepSeek-V4.1-Flash（552B backbone + 196B Engram，decode 激活 16B）
> C++20 · Zig 工具链 · Vulkan Compute (Slang) · AVX-512 · Unified Memory · NVMe Expert Streaming · DSpark 投机解码

**核心目标：** 在一台 Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / 单 NVMe 的机器上，把 DeepSeek-V4.1-Flash 的本地 decode 打到这台机器的物理上限，并且能用数据解释"上限在哪、为什么没到"。

文档状态：**v0.8（2026-09-15）**。v0.1 为初始 docx；v0.2 按开发机实测修订内存模型；v0.3 锁定目标模型为 V4.1-Flash，据其真实权重布局重写内存/存储/kernel/prefetch/投机解码设计；v0.4 落地代码骨架，写回 Q6/Q7 实测（§9.2.1），更正 §11.3 的 indexer KV 体积，固定 §14.1 目录；v0.5 取消 repack，改为直读原始 safetensors 分片（§5.1 重写）；v0.6 写回 P1 全部实测（内存系统是单一 ~217 GB/s 共享上限、dispatch 开销低一个数量级、cache 容量受 commit 限额约束、27,399 token 路由 trace、lookahead 降级为不做）；**v0.7 写回 P2 step 1 全部实测：容量问题已解决——扩 pagefile 后 slab 池 100 GiB = 5,711 个 expert 槽、h ≈ 0.92（§5.2/§9.2.2/§3.1）；非 MoE decode 路径九个 kernel 逐级对齐参考实现并跑通整层（§7.15），参考实现强制了八处改动（§2.4/§6/§7.2–§7.7/§7.14）；MoE kernel 的 M 扫描、packed fp16 / int8 dot4、fp8 shared expert 与 `h` 的 fp8 量化（§7.9.2），并**推翻**了两条 v0.6 的结论：§7.1 rule 6 的「x 分块进 LDS」与 §7.9 的「拆分 dispatch 免费」**；**v0.8 写回 P2 step 2 三条 track 的实测：**deepMoE 第一次自己产出了 token**——四十层、engram、head、采样全在 GPU 上，对 L3 oracle 一步 top-1 一致、教师强制 7/8、自由运行 6/8，热步 **134 ms = 7.5 tok/s**（§7.16、§13.4、§15）；§7.4 的 compressor 与 indexer **已产出、不再是加载进来的**，非 MoE 一层从 1.053 降到 **0.893 ms**（§7.15.5）；MoE 的 `HQuant=3` 把 decode 的量化税从 6.3 降到约 1.0 ms/token，int8 `x` 过不了 §12 判据因而默认关闭（§7.9.3）；并且**推翻了 v0.7 自己的一条结论**：§3.4 / §7.9 的「拆分 dispatch 每层 +0.193 ms」是测量漂移，真实代价 ≤ 0.026 ms/层且逐位相同，「先到的先算」恢复为默认（§7.9.3、§3.4）。修订记录见 [附录 C](#附录-c-修订记录)。

> **本文的实测数字** 一律标注测量日期与来源文件。五份原始报告是
> [docs/kernel_p1.md](kernel_p1.md)（P1 Track A：带宽矩阵、kernel 变体、内存路径、dispatch 开销）、
> [docs/route_trace.md](route_trace.md)（P1 Track B：trace 方法、faithfulness、与参考实现的出入）、
> [docs/kernel_p2_moe.md](kernel_p2_moe.md)（P2 Track D/H：M 扫描、fp8 shared expert、`h` 量化、
> 分组 dispatch；v0.1 是 step 1，**v0.2 是 step 2**）、
> [docs/p2_attention.md](p2_attention.md)（P2 Track E/F：L2 oracle、九个非 MoE kernel、整层链路；
> **§9–§12 是 step 2 的 compressor / indexer 与第二轮带宽**）
> 与 [docs/p2_decode.md](p2_decode.md)（P2 Track G：L3 oracle、整个 decode step、第一个 token）；
> 原始数据在 `bench/results/*.csv`、`tests/data/l3/` 与 `reports/*.json`。
> **报告之间有冲突时以后写的那一份为准**，本文在每一处都注明是哪一份。

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
   **v0.8 更新**：非 MoE 降到 **41.4 ms**（§7.15.5）、MoE（`HQuant=3`）**25.0 ms**（§7.9.3），
   加上 v0.7 漏掉的 engram 4.9 ms 与 compressor/indexer 0.8 ms，kernel 地板是 **≈ 72–75 ms**。
   **但真机上一个热步是 134 ms**（§7.16.2）——**这一段说的仍然是 kernel 的地板，不是 runtime 的现状**，
   两者差 1.8 倍，逐项在 §13.4。
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
- **§7.9 的"按 expert 拆两组、先到的先算"是默认方案。** v0.6 的理由（0.66 µs × 2 × 40 层
  ≈ 0.05 ms/token）口径不对，但**结论是对的**；v0.7 在这里写的"这句话错了 340 倍、
  每层 +0.193 ms = 7.7 ms/token"**是 v0.8 收回的一条结论**——那三个数是**依次**测出来的，
  中间隔着同一轮内 7–8% 的热漂移，而要量的差只有 1–3%。
  v0.2 改成**九个配置轮转测量**之后（`bench/results/kernel_p2b_moe.csv`，kernel_p2_moe.md §11），
  真实代价是**只拆 dispatch A 时每层 0.006–0.026 ms**（两对 A+B 是 0.014–0.046 ms），
  40 层 **0.2–1.0 ms/token**，而且**只拆 A 与一次算完逐位相同**（五种切法全部 5120/5120 个字一致）。
  结论：**默认只拆 dispatch A、dispatch B 最后跑一次**，Planner 只要判断"本层是不是真有 expert
  没就位"，不需要再估等待时长（§7.9）。
  **真正的教训不是外推错了，是量测纪律**：v0.7 的那个数字之所以站得住三周，是因为没有自检——
  轮转测量里 `A + B` 必须对得上 `whole`，而 v0.1 那一节里量到过
  `whole 0.652 = A 0.520 + B 0.290`，**物理上不可能**。
  **凡是要量 1–5% 的差，对照必须和被测量的东西轮转着测**（§7.9.3、build.md 的量测纪律一节）。

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
10. **CPU 侧绝不逐元素访问 GPU 可见的内存**（v0.8 新增，p2_decode.md §3.3）。
    §3.3 说过路径 A 的映射是写合并的、CPU 读它未缓存；实测**一次访问约 230 ns**。
    `GpuMoeBridge::run` 原来用三个逐元素循环搬激活与两个 `y`，一层 25,600 次访问
    = **6.0 ms/层 = 240 ms/token，占整个 step 的 26%**；换成每个向量一次 `memcpy`
    是 0.75 ms/层 = 30 ms/token。写的一侧同样：embedding 展开成 4 份 hc 拷贝时按元素交错写
    （四个相隔 20 KiB 的地址），**每次存都刷一次写合并缓冲**；正确的写法是在主机内存里展宽一次、
    再 `memcpy` 四次。
    **规则：`runtime/` 里任何用标量循环碰 GPU 可见指针的代码都是 bug**，
    只允许整块 `memcpy` 或非临时 store 流（§8.1 第 7 条是同一条规则的 CPU 侧表述）。
    这一条两次都是被写成"显然的循环"才出现的，所以它必须是规则而不是一次修复。

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

**v0.8 实测（P2 step 2，Track F，2026-09-15，`tests/data/l2x/`、`bench/results/attn_p2.csv`；
详见 [p2_attention.md](p2_attention.md) §9）：这两个 kernel 已经存在，压缩 KV 与 top-k
不再是加载进来的。** `gpu/shaders/compressor.slang`（**3 个 stage**：`wkv` / `wgate` 两个投影、
`norm`（含池化）、`store`）与 `gpu/shaders/indexer.slang`（**6 个 stage**：`wq_b` / `q_finish` /
`key` / `weights` / `score` / `topk`），十个 pipeline 接在 `AttnStage` 后面。
四十层里四层有 compressor、八层有 indexer，其余层一个都不 dispatch。

- **精度：十六个比较点里十一个逐位相同**（层 2 / 14 / 20，每个 stage 吃参考自己的输入）。
  不逐位的三个都以一个 fp8 GEMV 收尾，坐在 §3 的 1.6e-3 bf16 噪声地板上（cos 0.9999986）。
  fp4 字节平面**逐字节相同**，这一下同时钉死了 E2M1 的舍入规则（ties to even mantissa）、
  nibble 顺序，以及**两种 scale 格式**（压缩 KV 的 E4M3 与 indexer 的 UE8M0，§2.4）。
- **`index_score` 逐位相同是补上三次 bf16 舍入之后才有的**（einsum 的输出、乘 `weights`、
  沿头轴求和）；补之前是 0.99999。**又一次"比参考更精确"要付代价**，和 §6 的
  "RMSNorm 输出先舍入 bf16"是同一回事，只隔了一步。
- **三个还没被证伪的地方**，都写在测试里：(1) `index_score` 是**重算**的不是捕获的
  （`Indexer.forward` 只返回 indices），公式读错会在两侧同时复现；
  (2) **ratio-2 的池化在 pos 64 上没有参考输出**（`(64+1) % 2 != 0`，`Compressor.forward`
  返回 None），现在是对一个合成的完整组比 fp64 CPU 转写——**要真正验它需要一次两步的 decode 导出**，
  记为 §12 的未决项；(3) **top-k 在这个上下文上是退化的**：`index_topk = 512` 而候选只有
  32（ratio 2）或 65（ratio 1）个，`min(index_topk, n) = n`，参考把全部保留。
  所以 radix select 是**单独**测的——4096 选 512、故意造一簇并列值，对 CPU 稳定排序
  **512/512 含并列全中**。
- **带宽**：compressor 四层合计 **37.3 µs**、indexer 八层合计 **82.2 µs**，
  两者一起 **0.81 ms/token**，是 §7.15.5 那 41.4 ms 之上的 **2%**。
  两个 5 MB 投影报出的 311–378 GB/s 是 **cache 数字**（只有四层有它们，21 MB 在 32 MB MALL 里循环），
  µs 是真的、GB/s 不是内存系统的数。
  **`indexer.score` 是唯一随上下文增长的那个**：这里它读 16 KB / 8 个 key，16 µs 是地板不是测量值；
  64K 上下文 ratio 2 时是 32K 个位置 × 2 MB 的 key。
- **参考实现在这里又逼出五件事**（p2_attention.md §9.3，实现前逐条核）：池化的 softmax
  **沿 ratio 轴逐元素**做（512 维各有自己的两路 softmax，不是一个 token 一个权重）；
  ratio > 1 与 ratio = 1 两条路的 **dtype 变化点不同**（前者 fp32 全程、只在 `kv.to(dtype)`
  处舍入一次 bf16，后者是一个朴素的 bf16 `Linear`）；**一个 latent 代表它那一组的第一个 token**，
  旋转在 `start_pos + 1 − ratio`，而 indexer 的 **query** 旋转在 `start_pos`——一层两张 RoPE 表；
  一层里**两种 FP4 格式**（§2.4）；**key 的所有者是压缩它的那一层、不是给它打分的那一层**
  （decode 位置 64 上 ratio-2 源的组没完成，于是打分打在上一次发布的 cache 上，
  即层 20 的），**由主机而不是 shader 决定哪个 cache 地址进槽**。

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
- **"一个 workgroup 处理全部 64 头、KV 只读一次"这条实测是输的**（v0.8 更正，
  p2_attention.md §10.2）。机制是有的（`AttnSpec::heads_per_wg` 把 G 个头放进一个 workgroup，
  L2 流量除以 G），实测**越省流量越慢**：1 头/wg 64 个 workgroup 69 µs / 41 MB，
  2 头 100 µs / 20 MB，4 头 169 µs / 10 MB，8 头 288 µs / 5 MB——
  **64 头按 8 个一组就是 8 个 workgroup 对 40 个 CU**，占用率换流量换错了轴。
  §7.5 真正描述的是 **head-group × KV-tile** 的二维切分：**保持 64 个 workgroup**，
  每个覆盖 8 个头和八分之一的 KV，再跨 tile 做一次**部分 softmax 合并**——
  占用率和流量一起拿。这是这一项剩下的一半，排在 §15。
  与此同时，**向量化本身已经把两个 stage 从 141 µs 降到 69 µs/层**（2.8 ms/token，
  从这条路径的 13% 降到 7%），一个字节的流量都没动：原来一个线程负责一个 (head, position)
  串行走 512 维，64 头 × 640 位置 = **6300 万条 load 指令去读 320 KiB**；
  现在一整行 KV 由一个 wave 的 32 个 lane 覆盖（每 lane 16 维），点积是一次 `WaveActiveSum`。

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

**驻留检查的粒度（v0.8 恢复 v0.6 的定案，撤回 v0.7 的"340 倍"）**：

v0.7 在这里写过"翻倍的代价是 0.193 ms/层 = 7.7 ms/token，所以只在预计等待 > 0.2 ms 时才分组"。
**那三个数是测量漂移，不是代价**（kernel_p2_moe.md §11，`bench/results/kernel_p2b_moe.csv`）：
`whole` / `first` / `rest` 是**依次**测的，中间隔着同一轮内 7–8% 的热漂移，
而要量的差只有 1–3%；同一节还量到过 `whole 0.652 < A 0.520 + B 0.290` 这种物理上不可能的读数。

改成**九个配置轮转测量、各取自己的最好值**之后，八行冷热对照给出的真实代价是：

| 切法 | 每层 | × 40 层 | 与一次算完的数值关系 |
|---|---|---|---|
| **只拆 dispatch A，B 最后跑一次** | **+0.006 … +0.026 ms**（有一行是 −0.041，那是漂移在往回走） | **0.2–1.0 ms/token** | **逐位相同**（五种切法全部 5120/5120 或 30720/30720 个字一致） |
| 两对 A+B（3 + 4） | +0.014 … +0.046 ms | 0.6–1.8 ms/token | 1–2 ULP（max\|Δy\| ≤ 8e-8 的 \|y\|max，相对 2.4e-7） |

**定案：默认"按到达顺序分组计算"，而且只拆 dispatch A。** 三个理由：
dispatch A 占一层的 2/3（M=1 时 0.391 / 0.592）且工作量与 `list_count` **严格成正比**
（`gid.y` 就是槽），所以能和 I/O 等待重叠的正是它、拆开不浪费任何字节；
dispatch B **本来就要等所有槽的 `h`**，推到最后不损失任何重叠机会，还省掉第二次
640 个 workgroup 的跨 lane 归约与 `y` 读改写；而且这条路**逐位相同**，
不需要把判据从"逐位"放松到 1e-6。
**Planner 因此只要判断"本层是不是真有 expert 没就位"，不需要再估等待时长**——
v0.7 那条 "> 0.2 ms" 的门槛可以拿掉。每层 0.01–0.03 ms 的代价意味着**只要真有一个 expert 迟到，
拆就是划算的**。（§9.4 把 lookahead 降级为不做那条结论不受影响：那是为一个假想收益付确定代价，
这里是为一个真实的等待付一个已量到的、接近零的代价。）

kernel 侧的支持不变：indirection list（`SlotList` + `list_count`）短一点即可，
另加 `DownPush.flags` 的 bit 0 = 累加（`MoeRunner::set_accumulate()`），让一层可以拆成任意多组；
**两对 A+B 的切法仍然只有 1–2 ULP**，判据 1e-6，
`tests/test_gpu_moe.cpp::a_partial_dispatch_reduces_to_the_same_y` 两种切法都测。

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

> **v0.8：这一段的"唯一能到 85% 的设计"被实现之后推翻了。** `XMode = 6` 写出来了，
> dispatch A 确实稳定快 15–30%（M=6 的 A 是全场最好的），但 **A+B 仍然只有上限的 63–67%**，
> 因为**瓶颈换到了 dispatch B，而 B 的激活是 `h` 不是 `x`**，这条路对它无能为力；
> 端到端 M ≥ 2 只有 0–7% 且不稳定，M=1 是负收益；精度在第二个 golden 上是 **8.9e-3**，
> 过不了 §12。**默认关闭**，详见 §7.9.3。上面那句"这是唯一能让 M=6 摸到 80/85% 线的设计"**撤回**。

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
> —— **v0.8：做了后者，`HQuant = 3`，代价从 +23…29% 降到中位数约 +3%，
> 6.3 → 约 1.0 ms/token。LDS 转置那条路是做不到的，理由见 §7.9.3。`HQuant = 2` 从此没有理由再用。**

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

**v0.8：`h` 的量化改用 `HQuant=3`，分组 dispatch 改为只拆 A，见 §7.9.3 末的新清单。**

#### 7.9.3 P2 step 2 的 MoE（Track H，2026-09-15，`bench/results/kernel_p2b_moe.csv`，336 行；详见 [kernel_p2_moe.md](kernel_p2_moe.md) §8–§12）

同一台机器、同样的方法与口径。**这一轮只动了 §7.9.2 留下的三件事**，
而且三件都改变了 v0.7 写进本文的结论。v0.1 的 `kernel_p2_moe.csv` 原样保留，
**两个 CSV 之间不要比绝对值**——这一轮的 M 扫描（不量化的最优变体，上限 211.1–217.2 GB/s）
与 §7.9.2 (a) 是同一个形状、同一个量级，M=1 220.7 GB/s / 0.596 ms，M=6 146.7 GB/s /
0.1495 ms/token，**没有让任何一个 M 变慢**。

**(a) `HQuant = 3`：把 `h` 的量化拆成 A 与 B 之间的第三个 dispatch——(f) 那 27% 拿回来了**

LDS 转置那条路**做不到**：fp8 的块是 `h` 沿 2304 维连续的 **32 行**，而 `L32 R1` 的一个
workgroup 只拥有 `256/32 × 1 = 8` 行。LDS 转置只能在 workgroup 内部换轴，换不来它根本没有的
另外 24 行；跨 workgroup 交换块 scale 需要一次全局 barrier，**那就退化成第三个 dispatch**。
所以直接拆：`gpu/shaders/moe_hquant.slang`，一个线程一个 32 元素块
（读 64 B fp16 → amax → `fast_round_scale` → 写 32 B fp8 + 一个 scale word），
M=1 时是 504 个线程 2 个 workgroup。`SlotList` 是它的第一个 binding，所以"先到的先算"照样切得开。
唯一的代价是它**不能就地量化**（线程 t 要写的 fp8 字节落在线程 t/2 的读区间里），
所以 fp8 平面放在 fp16 平面之后，`h` 的 allocation 从 2 B/元素变成 **3.125 B/元素**
（M=6、7 槽时 302 KB，对 132 MB 的权重流可忽略）。

每个 M 在**它真正会用的形状**上成对测量（相邻测量，§7.9.3 (c) 说明为什么这很重要）：

| M | 形状 | 不量化 ms/pair | `hqP` ms/pair | 代价 | `hqP` ms/token |
|---|---|---|---|---|---|
| **1** | `L32 R1 xglob` | 0.600 | **0.625** | **+4.2%** | **0.625** |
| 2 | `L16 R2 xgf16/ldsi8` | 0.841 | 0.851 | +1.2% | 0.425 |
| 3 | `L16 R2 xgf16/ldsi8` | 0.780 | 0.780 | 0.0% | 0.260 |
| 4 | `L16 R2 xgf16/ldsi8` | 0.806 | 0.814 | +1.0% | 0.204 |
| 5 | `L16 R2 xgf16/ldsi8` | 0.853 | 0.874 | +2.5% | 0.175 |
| **6** | `L16 R2 xgf16/ldsi8` | 0.897 | **0.945** | **+5.4%** | **0.158** |

**代价是 0.0% … +5.4%**，M ∈ {2,3,4,5} 在 ±2.5%（也就是 §2 的漂移）以内。
M=1 量了四轮：**−0.5% / +2.2% / +3.9% / +4.2%，中位数约 +3%**——
对比 `hq8` 在 M=1 上的 **+23…29%**：**decode 的量化税从 6.3 ms/token 降到约 1.0 ms/token。**
M=1 那 +4.2% 拆得出来：`ms_a`（0.401 对 0.401）和 `ms_b`（0.226 对 0.223）与不量化**一模一样**，
差额整个 0.025 ms 出在**多出来的那个 dispatch 与它的全局 barrier**，不是任何一段计算。
再往下只能减 barrier，那是 `gpu/vulkan/cmdbuf.cpp` 的事。

两件顺带的事：**B 的 scale 每块只乘一次**——第一版 `load_h8q` 对 fp8 `h` 的每个元素做
`sFp8[byte] * s`（每 (列, 块) 32 次乘法），而块 scale 在整块里是常数，
提到块外写成 `acc += ldexp(p * hs[m], e2)` 之后，**每 (行, 列, 块) 一次乘法**取代 32 次，
在 M=1 的 `RowsPerLane = 1` 上一分钱都摊不掉，正是那 5.2% 里的 0.040 ms；
**块 scale 是 2 的幂，提取逐位精确**，`vs y_hq16` 的 5.030e-08 改动前后一个数字都没变。
**数值上 `HQuant = 3` 与 `HQuant = 2` 是同一段算术换了个 dispatch**：对 `y_hq16` 都是
**5.030e-08**，`L32 R1` / `L16 R2` / `L64 R1` / M=6 四种形状给出同一个数——
这正是 `hq8` 做不到的（它对形状有约束）。
**`HQuant = 2` 从此没有理由再用**；而 `runtime/moe_bridge.h` 的默认还是 `h_quant = 1`（`hqB`），
在它自己的默认形状 `L32 R1` 上那是 **+45.5%**，改成 3 是一行的事（§7.16 的 P1 特化那一条）。
**限制**：`h_quant = 3` 要求 `h_precision == 0`（它量化的就是 A 写出的 fp16 `h`）。

**(b) `XMode = 6`（x 在 kernel 外预量化成 int8）：实现了，两条判据都不达标，默认关闭**

`gpu/shaders/moe_xquant.slang`，每 token 一次、40 层复用（就地量化在这里是安全的）。
§7.9.2 (d) 的模型**漏了权重侧**：`fp4_nibbles_to_i8x4` 每 (块, 行, 矩阵) 要调 8 次，
粗算每块 512 条指令，而 M=6 个列一共才 150 条——**权重解码占 3/4**。
修法与 §7.1 rule 5 同一套（256 项 LDS 表，一个 FP4 字节直接给出两个 `2×E2M1` 的 int8），
M=6 的 dispatch A 从 147 走到 **168.1 GB/s**。

- **速度**：dispatch A 的收益是真的也是稳定的（M=6 三轮 137.5→158.8、129.6→168.1、
  131.3→152.1 GB/s，**+15…30%**，每一轮都是全场最好的 A）；
  **端到端不是**：M ≥ 2 三轮方向一致但幅度在 0–11% 之间摆（和 §2 的漂移同一量级），
  诚实的说法是 **0–7% 且不稳定**，**M=1 是明确的负收益**（+1.2…+15.9%）。
- **A+B 在 M=6 上只有 137–147 GB/s = 上限的 63–67%**，离 80% 还很远。原因很直接：
  **瓶颈换到了 dispatch B（M=6 时 99–108 GB/s），而 B 的激活是 `h` 不是 `x`。**
- **精度：过不了 §12。** 每 32 元素一个 scale 时 expert (0,0) 是 **2.90e-3**、
  (39,383) 是 **8.85e-3**（GPU 侧 2.902e-3 / 8.856e-3，与 torch 吻合到三位有效数字），
  **两者差 3 倍**。权重侧在 int8 路径上是**精确的**（`2×E2M1` 就是 int8），
  所以误差全部是 `x` 自己的。**每行一个 scale 更糟**（4.45e-3 / 1.40e-2：块内 amax ≈ 2.5σ，
  整行 amax ≈ 4σ，步长粗 1.6 倍）；int8 + fp16 残差确实能修回 1.4e-4 / 3.7e-4，
  但那只是把 fp16 的乘加拆成两段做，**比直接用 `XMode = 4` 更贵**——它证明的是
  "误差确实全部来自 x 的量化"，不是一条可用的实现。
- **处置**：实现完整、有测试、**spec 常量默认 0（关闭）**，**绝不设成 decode 的默认**。
  唯一说得通的用法是 §10 的**投机验证批**（M ≥ 2，而且接受检查本来就要对比 draft 和 target），
  **但即使在那里也要先由 L2 判定 8.9e-3 能不能接受**——0–7% 不值得拿一个没被批准的数值代价去换。
- **对 §12 的后果**：**判据不该为 int8 `x` 放宽。** 错误是 `x` 的、不是 kernel 的，
  而且**逐 expert 变化 2 倍以上**——一个判据不能靠挑 expert 来满足。

**(c) §7.9.2 开头那条量测纪律要加一句：对照必须轮转着测**

v0.7 写进 §3.4 与 §7.9 的"拆分 dispatch 每层 +0.193 ms（340 倍）"**整个是漂移**（§3.4 已改）。
根因不是外推，是**顺序测量**：`whole` / `first` / `rest` 隔着几秒钟依次跑，
而同一轮不同小节之间有 7–8% 的热漂移，要量的差只有 1–3%。
**规则：凡是要量 1–5% 的差，对照必须和被测量的东西轮转着测，并且用一个物理自检把它钉住**
（这里是 `A + B` 必须对得上 `whole`；轮转之后 0.391 + 0.201 = 0.591 对 0.592，
顺序测量时量到过 0.520 + 0.290 = 0.810 > 0.652 这种不可能的读数）。
`bench/kernel_bench.cpp` 的分组 dispatch 一节现在就是轮转的，`h fp8` 与 `int8 x`
两节也各自带上了同节的不量化对照。

**v0.8 推荐的默认特化（取代 §7.9.2 末的那一段）**

```
decode（M=1）        : L32 R1, sg32, dec0（常量表）, h=fp16, XMode=0, HQuant=3
投机验证（M=6）      : L16 R2, sg32, dec0, h=fp16, XMode=4, x_mode_b=3, HQuant=3
shared expert 的 dispatch : Fp8Slots=1，槽的 ids 带 kSlotFp8
分组 dispatch        : 只拆 dispatch A，dispatch B 最后跑一次（§7.9）
x 的 int8 预量化      : XMode=6，默认关闭
```

**接口变化仍然全是加法**（kernel_p2_moe.md §12）：`MoeSpec::h_quant` 多一个取值 3、
`x_mode` 多一个取值 6（`x_mode_b` 取 `kFollowA` 时遇到 6 会落回 0，因为 B 的激活是 `h`）；
dispatch A 的 storage buffer 从 8 个变成 9 个（binding 8 是 x allocation 的别名，
`XMode = 6` 用），**这是本轮唯一的 descriptor 布局变更**；
`h` 在 `h_quant == 3` 时是 3.125 B/元素、x 在 `x_mode == 6` 时是 3.125 B/元素，
两块都由 `MoeRunner::create` 自己分配，**调用方一个字都不用改**；
两条新 pipeline 由 `run()` 自动录进同一个 command buffer。

### 7.10 Engram（层 1、14）

- CPU 侧：token id → 压缩 id → 24 个 hash → 行地址；行数据 24 × 264 B 由 EngramPrefetcher 提前读到 GPU 可见的小环形缓冲。
- GPU：`wkv [25600 × 6144]` fp8 157 MB GEMV（输入 6144 = 24 行 × 256，解码 fp8 + 每 32 元素 scale 时融合在输入加载中）→ `key[4][5120]`, `value[5120]`；门控 + 残差更新融合在写出阶段（每份拷贝各自做 RMS 归一化点积）。
- 2 个 dispatch（GEMV 需完整输入；门控需完整 key）。
- **v0.8 实测（Track G，§7.16）**：`gpu/shaders/engram.slang` 两个 dispatch 已实现并跑通，
  **两层合计 4.9 ms/token（2.45 ms 一层）**——不大，但 §13.4 v0.7 把它整个漏掉了。
  其中约一半是 48 次随机 4 KiB 读，**本该在 token 定下来的那一刻就发出去**（§9.5 的 P2 预取），
  现在它阻塞层 1。
- **hash 常量表是导出的，行 id 不是**（§12）：压缩词表映射（NFKC / 去重音 / 小写 / 空白归一，
  129,280 个 token）、逐 (层, lookback) 的乘子（`numpy.random.default_rng(10007 * layer_id)`）、
  24 个桶的模数（`engram_vocab_size − 1` 之上的 24 个素数）**全部由 tokenizer 与 `config.json` 决定，
  与 checkpoint 无关**。在 C++ 里复现一个 PCG64 流和一个 `tokenizers` 归一化器是第二份要永远维护对的实现，
  所以导出的是那些常量（517 KB 的 token map + 约一百个整数），`runtime/engram.h` 用它们在
  **runtime 真正走的轨迹上**算地址。**加载进来的是一张常量表，不是关于某个 token 的任何东西。**

### 7.11 LM Head 与采样

- `norm → head [129280 × 5120] bf16`（1.32 GB）→ `logits` fp32 → 采样。
- GEMV 结构同上（bf16 模板），129280 行 / 40 CU 负载均衡良好。**这是单个最大的常驻读取（10%）**，也是投机解码收益最大的地方（M=5 位置一次读完）。
- 采样在 GPU 上：温度缩放 + Gumbel-max（与参考一致）或 argmax；随机数用 Philox 计数器，seed 可复现。logits 只回传需要的 top-k 与采样结果。
- **v0.8：`head.slang` 的 stage 1（greedy argmax）已实现**，返回四个词而不是 129,280 个 logit（§7.16）。
  **Philox / Gumbel-max 那一半仍未做。** 注意 argmax 的归约必须保住它的 −3e38 哨兵语义：
  `v > best` 对 NaN 永远是假，所以**一步全 NaN 的 logits 会被报成 token 0 / margin 0.0000**
  而不是报错——§7.16.4 的第二个 bug 就是这么显形的，**这不是采样器的 bug，但它是一个静默的失败模式**。

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

> **v0.8：这张表是 step 1 的，已被 §7.15.5 取代**（一层 1.053 → **0.893 ms**，
> 40 层 + head 48.5 → **41.4 ms**）。留着是因为 step 2 的"before / after"对照是对着它量的。
> **端到端的账请用 §7.15.5 和 §7.16，不要引这一张。**

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
| **压缩 KV 与 indexer 的 top-512 列表** | **加载进来的**（来自 oracle 的 prefill）。**§7.4 的 compressor / indexer kernel 还没写**；它们下游的一切都是真的、也都测了。**v0.8：kernel 已经有了（§7.4），但还没接进 `Engine`**（§7.16、§15.2 的里程碑 (i)） |
| prefill、engram 写入、DSpark | 未开始。**v0.8：engram 的 GPU 路径已完成**（§7.10、§7.16） |

~~**压缩的那一半现在存 bf16，不是 §11.3 的 FP4 E2M1 + E4M3/16。**~~ ——
**v0.8 已解决**：compressor 写出来了，而且**就是按 §11.3 的打包形式写的**
（FP4 block-16 的 nibble 平面 + E4M3 的 scale 平面，逐字节对上参考），
64K 上那 18 MB 的差额已经拿回来（§11.3）。

**§7.1 的 gate 是真跑的**：`DecodeLayer::run_moe` 从 host-coherent 内存读出 gate 的 ids，
回调让六个 expert 驻留，host-signal MoE 那个 wait 命名的 timeline 值，然后 dispatch。
P2 里每个 expert 本来都会驻留、这个 wait 是走形式——**但测试故意只给八个槽的 cache**，
所以六个 expert 每层都真的从 NVMe 经 `store::Planner` 取回来，gate 就是放行它们的那个东西。

**command buffer 是每层一个，不是每 token 一个**（§7.1 rule 1）。拦路的是地址表：
一个 stage 拥有表里的一个 slice，预录制一个 token 需要**地址表在 shader 里按层索引**，
就像 expert 指针表那样。按 §3.4 的账值 ~0.5%，所以它等 P3 的流式 runtime。

#### 7.15.4 缺口，按大小排序（step 1 的清单；**v0.8 的状态见 §7.15.5**）

> **v0.8 逐条勾账**：第 1 项做了一半（`wq_b` 到 94%，`wo_b` 仍 62% 且已排除两个嫌疑犯）；
> 第 2 项的**指令数**解决了（142 → 69 µs/层），**流量**没有，而且"KV 只读一次"的那个形状实测是输的（§7.5）；
> 第 3 项**已完成**（§7.4）；第 4、5 项不变。

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

#### 7.15.5 第二轮带宽（P2 step 2，Track F，2026-09-15，`bench/results/attn_p2.csv`；详见 [p2_attention.md](p2_attention.md) §10）

§7.15.4 的第 1、2 两项做了一半。**"before" 列是同一个二进制、同一台机器、相隔几分钟**，
跑的是上一个 commit 的 `.spv`（所以不是两天的两个数）。

| kernel | before µs / GB/s | after µs / GB/s | % 上限 |
|---|---|---|---|
| `wq_a` | 44.5 / 147 | **38.5 / 171** | 79% |
| `wq_b` | 258.3 / 163 | **205.4 / 204** | **94%** |
| `wkv.gemv` | 19.9 / 132 | **17.1 / 154** | 71% |
| `sparse_attn.score` | 54.7 | **42.1** | 延迟型 |
| `sparse_attn.combine` | 85.9 | **27.3** | 延迟型 |
| `wo_b` | 341.9 / 123 | **312.2 / 135** | 62% |
| `wo_a` | 199.2 / 169 | 201.4 / 167 | 77% |
| **一层，dispatch 1–9** | **1052.6 / 127** | **892.6 / 149** | **69%** |
| `head` | 5618 / 236 | 5653 / 234 | 108% |
| **40 层 + head** | **47.7 ms** | **41.4 ms** | |

加上 §7.4 的 compressor（四层 37.3 µs）与 indexer（八层 82.2 µs）= **0.81 ms/token**，
即这 41.4 ms 之上的 2%。

**四件事起了作用，而 §7.15.4 点名的嫌疑犯有一个不是：**

1. **LDS 预算要按 shader 定，不是按 kernel 家族定。** `gXQ` 原来无条件按 `wo_b` 的
   K = 8192 开，于是 `wq_b` 预留 16 KiB 只用 2.5 KiB。**但不是"越小越好"**：
   `wq_a` 在 5120 的预算下是 152 GB/s、在 8192 下是 166——更多常驻 workgroup 各自
   重新 stage 一个 4–5K 宽的激活是**更多**的 L2 流量。赚的是激活本来就小的那两个
   （`wq_b` 与 `indexer.wq_b`，K = 1280），那里占用率是白给的。
2. **`act_quant` 的 staging 原来只用了六分之一的 workgroup。** 一个线程一个 32 元素块，
   于是 `wq_b` 用 256 个线程里的 40 个去 stage 1280 个激活，216 个线程在最贵的那段算术上空转。
   每块的线程数现在是 K 的编译期函数（K=1280 时 4、4096 时 2、8192 时 1），块 amax 在它们之间做 LDS 归约。
   顺带：`fp8_round` 改成九条无分支指令，**没有块要量化的线程也不再白跑舍入**——光这一条就是 `wq_a` 的 8%。
3. **`row_reduce` 换成一次 `WaveActiveSum` 不是普遍的胜利。** 用 subgroup 归约替掉 LDS 树的
   七个 workgroup barrier，值 **`wq_b` +30%、`wkv` +20%，而 `wq_a` 与 `wo_a` 各 −14%**。
   输的两个正是"每个 workgroup 都要重新 stage 一个宽激活"的那两个——**barrier 显然在替它们
   彼此定速，而内存系统喜欢那个节奏**。所以它是一个**逐 stage 的 specialisation 常量**
   （`WaveReduce`），四个数字就写在 `kStages` 旁边。**不要把它当成一条普适规则。**
4. **`sparse_attn` 从来不是带宽受限，它在做标量字节读**（见 §7.5 的更正）。

**而 §7.15.4 点名的那个嫌疑犯不是嫌疑犯**：把 `wo_b` 用 `ActQuant = 0` 编一遍
（算错的算术，只看时间）量到 **150.2 GB/s 对开着时的 151.7**——
**激活量化的往返对 `wo_b` 一分钱都不收**，那个"结构性的前置量化 dispatch"什么也买不到。
也不是 DRAM：42 MB / 312 µs = 135 GB/s 对 217 GB/s 的内存系统，
而 20 MB 的重复 stage 是一个 32 KB 的工作集、住在 MALL 里。
**剩下的只有 LDS（19 KiB，一个 CU 三个 workgroup）与每 lane 一个累加器——
也就是 kernel_p1.md §3.3 要的那个 K-split**，它需要多一个 dispatch、因而需要 runtime 改动，
所以这一轮没做（§15）。

### 7.16 整个 decode step 的实测（P2 step 2，Track G，2026-09-15；详见 [p2_decode.md](p2_decode.md)）

**这是 deepMoE 第一次自己产出 token。** `Engine::decode_step` 跑的是：embedding 查表、
四十层 §7.14 的 dispatch 1–11（§7.1 的驻留 gate 夹在 9 和 10 之间）、层 1 和 14 的 engram
（含它的 NVMe 行取数）、最后的 collapse、1.32 GB 的 head、以及一个返回四个词而不是
129,280 个 logit 的 greedy argmax。

**仍然是加载进来的只剩两样**（`Engine::status()` 每次运行都打印 LOADED 那一行，
所以一份 transcript 不可能被误当成自足的）：**prompt 留下的状态**（prefill 是 §11 / P5），
以及**每步的压缩 KV 与 indexer top-k**——§7.4 的 kernel 已经存在（§7.4 v0.8），
但写这一条时还没有接进 Engine，接上就删掉 `runtime/decode_state.h` 的逐步那一半。

#### 7.16.1 正确性（§12 L3）

| | 结果 |
|---|---|
| 一步 decode，logits 对 L3 | **top-1 一致**；对参考 top-64 的 Spearman ρ = **0.97**；max \|Δlogit\| = 0.64 |
| 八步，教师强制 | **7 / 8 一致**；不一致的那一步，参考自己的 margin 是 **0.95** |
| 八步，自由运行 | **6 / 8**（此后解码的是参考没走过的序列，不计） |
| 残差流，七层逐层探针对 L2 | 四十层下去 cos ≥ **0.996**；七层里六层的路由与参考 6/6 一致 |

**§12 L3 要的是 100% token 一致，这里是 7/8 和 6/8**，而且**知道缺的那一个是什么**：
层 2 的 gate 在一个近似平局上选了**不同的第六个 expert**，
那一次交换是 logit 差里最大的单项。层 2 那一行值得单独读：进它的流是 cos 0.9997
（和层 0、1 一样好），而它的 MoE 输出只有 **0.9884**，比其他每一层都差一个数量级——
**这不是漂移，是一个离散的分支**：一个 0.9997 精度的分数向量足以把第六和第七个 expert
的近似平局排反，七个里换掉一个就把和挪动约 1%。
自由运行那一步（step 6）同理：参考的 margin 是 0.95、我们的 max |Δlogit| 是 4.2
（是其他任何一步的四倍），**但还没有定位到层**——那需要一次 step 6 的 L2 导出，
现在的 oracle 不产出（§12 的未决项）。

**层级探针因此不按 §12 L2 的 0.999 收**：那条判据是给"从 oracle 自己的输入跑一层"用的。
这里断言的是**四十层之后流仍在 1% 以内**，且七个探针层里至少六层的路由一致。

#### 7.16.2 速度：热步 134 ms = 7.5 tok/s，是 §13.4 kernel 预算的 1.8 倍

一个所有 expert 都已驻留的步（探针 pass 已经把它们取回来了，**没有任何东西在等 NVMe**）：

| 桶 | ms | 每层 |
|---|---:|---:|
| attention，dispatch 1–9 | 51.9 | 1.30 |
| MoE，dispatch 10–11 + shared expert | 68.6 | 1.72 |
| engram（2 层） | 4.9 | 2.45 |
| collapse + head + argmax + 播种 LOADED 的 KV | 8.4 | — |
| NVMe stall | 0.2 | — |
| **合计** | **134.0** | **7.5 tok/s** |

**这就是这个实现今天的计算地板**，是 §13.4 那 75.8 ms kernel 预算的 **1.8 倍**。
差额三条，**全部可修，而且都不在 kernel 里面**（Track I）：

1. **MoE 每层 1.72 ms 对 §7.9.2 同样七个槽的 0.68 ms**，其中约 0.75 ms 是主机侧的工作。
   两个原因：`MoeBridgeConfig` 还带着 **P1 的特化（`decode_mode = 0`、`h_quant = 1`）**
   而不是 §7.9.3 定下的 P2 赢家——`h_quant = 1` 在它自己的默认形状 `L32 R1` 上是 **+45.5%**，
   改成 3 是一行的事；以及桥每次 submit 只要一次迭代，**录制与启动一点都没有被摊薄**。
   **这是一个 token 上最便宜的 40 ms。**
2. **21 ms 的主机桥**（见 §7.1 rule 10）：那三个逐元素循环换成 `memcpy` 之后从 240 ms/token
   掉到 30 ms/token；剩下的是仍未消除的往返。
3. **两次分开的 submit**：`Engine::measure_submit_overhead` 量到一次几乎不做事的 dispatch
   是 **0.13–0.15 ms**（空闲机），一个 decode step 大约做 128 次
   （40 attention + 80 MoE + 4 engram + 4 收尾），也就是 **约 17 ms/token**。
   **值得修**（§7.1 rule 1 要的是每 token 一个预录制的 command buffer），
   **但它是热步的 13%，不是量之前看上去的 30%。**

**attention 每层 1.30 ms 对 §7.15.5 的 1.07 ms（step 1 的那一轮）不是回归**：
那个基准循环八层，`mega_mhc.mix` 与 `gate.score` 因此坐在 32 MB 的 MALL 里，
§7.15.2 的两条 caveat 早就写明了这一点。**真 token 循环四十层，它们不在。**

#### 7.16.3 冷启动：八步 0.88 tok/s，83% 是 NVMe

`deepmoe run --steps 8 --cache-gb 12`，从零命中开始：**8 个 token、9.13 s、0.88 tok/s
（逐步 0.88–1.15）、命中率从 0 爬到 0.80（总计 0.360）、从 NVMe 读 23.1 GB**。
**NVMe 是一个冷步的 83%。** 0.36 的命中率下一个 token 错过 240 个 expert 里的约 150 个 = 2.8 GB；
stall 期间盘跑 2.8–3.5 GB/s（IoEngine 自己的 `effective_gbps` 是 **3.0–3.9 GB/s**），
对 §9.2.1 的 4.5 GB/s 平台——**正是"每层发六个 miss 然后等"该有的形状**：
队列深度峰值是 8，但只在一层的突发内部，突发之间盘是闲的。
**把下一层的取数与本层的计算重叠正是 §9.4 Planner 线程要做的事，而它还没在跑。**

- **`--profile FILE.jsonl` 写 §13.1 的记录，一个 token 一行。它的 `hot_bytes` 是 8.52 GB**，
  与 §2.3 的每 token 驻留预算**吻合到三位有效数字**——而且是按层从 manifest **加出来的**，
  不是把 §2.3 抄一遍，所以它是对 §2.3 的一次检验。
- **那条记录里有一个字段是错的，而且不是我们的**：`nvme_util` 读到大于 1（第一个 token 是 6.2），
  因为 `IoEngine` 的忙碌计数器把每个 chunk 自己的延迟**相加**，而不是取"至少有一个 chunk 在途"
  的窗口的并集，队列深度是 8。`nvme_gbps` 被同一个因子低估。**在这个修好之前读
  `effective_gbps`。**
- **`--cache-gb 48` 的命中率是 0.349，并不比 12 GiB 好**——2,700 个槽多过八步碰到的
  1,920 个 (层, expert) 对。这不是 bug 也不与 §9.1.1 矛盾：相邻 token 只共享约一半的
  routed expert（§3.1 第 3 点），**八个 token 长的运行永远到不了 0.90 描述的稳态**。
  八步是一个正确性 harness，不是 cache 基准；命中率在 `tools/cache_sim.py` 里量。

#### 7.16.4 三个只有四十层的链路才能发现的 bug

三个都是同一个形状：**一个逐层或逐 kernel 的测试在结构上看不到的东西。**

1. **融合的 `hc_post` 折进了错的那个子层的输出。** §7.7 把一个子层的 `hc_post` 推迟到
   **下一个**子层的第一个 `mega_mhc`；`DecodeLayer::bind` 让 attention 那一半的 `hc_post`
   读 `wob`（attention 的输出），而它必须读 `moe_y`——**一层 attention 之前的那个子层
   是上一层的 FFN**。`tests/test_gpu_layer.cpp` 看不到：它从 oracle 自己的输入跑一层，
   于是 `apply_hc_post` 是关的，那个槽从来不被读。四十层什么都读：
   每层的 attention 输出进流两次、每层的 MoE 输出被丢掉，模型把输入 token 一遍遍解码出来
   （ρ = 0.14，max |Δlogit| = 13）。
2. **`engram.slang` 在填表的 barrier 之前就读 FP8 解码表。** `fp8_table_fill(tid)` 写 `gTbl[tid]`，
   紧接着的 staging pass 通过 `gTbl[byte]` 解码，那是**别的线程**的条目。
   一个 Wave32 内部两者是同步的，所以 0..31 项永远是对的、其余是另外七个 wave 恰好走到哪儿。
   **症状不是失败，是同样八个 token 每次运行出来不一样**，偶尔一步 logits 全是 NaN——
   采样器把它报成 token 0、margin 0.0000，因为 `v > best` 对 NaN 永远是假、
   归约保住了它 −3e38 的哨兵。修法是两个 barrier 而不是一个。
3. **在写合并内存上逐个 float 地算。** 见 §7.1 rule 10。

**三条一起说的是同一件事**：§12 的 L1/L2 判据管得住"一个 kernel 算得对不对"，
管不住"把它们接起来时接对了没有"。**L3 不是 L2 的加强版，它是另一个维度。**

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
7. **CPU 碰 GPU 可见内存只能整块搬，不能逐元素算**（v0.8 新增，p2_decode.md §3.3，
   与 §7.1 rule 10 是同一条规则的两侧）。第 6 条的 21.5 / 32.7 GB/s 是**流式**数字；
   **随机的标量访问是另一个量级：路径 A 的映射上实测约 230 ns 一次**。
   `runtime/` 里任何在 GPU 可见指针上跑标量循环的代码都是 bug——
   一层 25,600 次就是 6.0 ms/层 = **240 ms/token，占一个热 decode step 的 26%**。
   允许的只有整块 `memcpy` 与非临时 store 流：先在主机内存里把向量拼好，再一次搬过去。
   **这条被违反过两次，两次都是写成"显然的循环"。**

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
`T_layer(MoE, M=1) = 0.625 ms`（**v0.8 实测，含真 fp8 shared expert 与 `HQuant=3`**，§7.9.3；
v0.7 写的 0.683 是不做 `h` 量化的数，v0.6 写的 0.602 是第七个槽用 FP4 顶替时的数），
非 MoE 的那一层实测 **0.893 ms**（v0.8，§7.15.5；v0.7 是 1.071），
整层合计 **≈ 1.52 ms**（§3.4 的模型说 0.97–1.29 ms，那是按字节 ÷ 上限算的下界；
实测高出来的部分是 `wo_b` 还没打满，§7.15.5）；
`T_io(18.8 MB) ≈ 4.0–4.26 ms`（§9.2.1、§5.1.3）。
隐藏 1 个 miss 需要 `d ≥ 3`（按 1.52 ms 的实测整层；v0.6 按 0.97–1.29 ms 算得 `d ≥ 4`）；
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
| MoE kernel 时间（**v0.8 实测，7 槽 + `HQuant=3`**） | **0.945 ms / 6 token = 0.158 ms/token**（M=1 是 **0.625**） | **÷4.0（实测）** |
| routed expert 流量（并集） | `union_frac[5] = 0.634` → 碰 3.2 倍于单 token 的 expert（逐层 2.6–4.1 倍） | **÷1.6（5 / 3.2）** |

也就是说：**常驻部分被完全摊薄，expert 流量只降到 1/1.6。**

**v0.7 的形势变了**：pagefile 已经修好，h ≈ 0.920，NVMe 项从每 token 的 81% 掉到
**≈51%**（80 ms / 156 ms，§13.4）。所以：

- **投机解码现在是第一杠杆**，因为它摊薄的那 75.8 ms 计算已经是每 token 的一半了；
- M=6 的 kernel 效率仍然是投机解码收益最大的一块 kernel 工作（它同时乘在上面两行上），
  但 §7.9.2 已经说清楚 **M=6 不是带宽受限而是 VALU 受限**，
  所以目标要换成 `ms/token`（v0.8 的 `HQuant=3` 口径是 0.158），
  而不是"有效带宽 ≥ 80% 上限"。
- **v0.8 的形势又变了一次**：真机上的热步是 **134 ms**（§7.16.2），不是 §13.4 的 75.8 ms
  kernel 预算。**在 Track I 把 134 拉回 ~75 之前，投机解码摊薄的是一个虚高的分子**——
  §15 因此把"热步 ≤ 90 ms"排在 DSpark 前面。

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
- **`T_hot(M)` 的 MoE 部分已实测**：v0.8 开着 `HQuant=3`（产线上真正要跑的东西）是
  M=1 **0.625 ms/层**、M=6 **0.945 ms/层 = 0.158 ms/token**（§7.9.3 (a)）；
  不开量化是 0.600 / 0.897。中间值按同一张表内插。
  **注意口径**：那张表的 7 个槽全是 FP4，换成真 fp8 shared expert 之后每层约再加
  0.09（M=1）–0.34（M=6）ms（§7.9.2 (e)）；v0.7 记的 0.683 / 1.310 是"真 shared、不量化"的那一对。
  **三个口径不要混用**，用哪个都要说是哪个。
- **`T_hot(M)` 的非 MoE 部分也已实测**：M=1 时 40 层 + head = **41.4 ms**（v0.8，§7.15.5）。
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

**v0.8：打包的那一行现在是真的。** §7.15.3 记的"压缩的那一半现在存 bf16，不是 §11.3 的
FP4 E2M1 + E4M3/16，等 compressor kernel 能写出打包形式时再定"——compressor 写出来了
（§7.4），**它就是按打包形式写的：FP4 block-16 的 nibble 平面 + E4M3 的 scale 平面，两块分开的 buffer**，
`tests/test_gpu_attn.cpp` 对参考**逐字节**比（0/256 个 nibble 字节、0/32 个 scale 字节不同）。
所以 64K 上下文取 **48 MB 那一栏**，18 MB 的差额已经拿回来了；
代价是 `sparse_attn` 内层循环里一次 nibble 解包，已经包含在 §7.15.5 的 69 µs/层里。
**indexer 的 q/k 是另一种格式**（FP4 block-32 + 2 的幂 scale，§2.4）；
一层里两种 FP4 格式，**混掉在 cosine 上看不见、在字节上看得见**，所以测试比的是字节。

### 11.4 Prefix KV 持久化

多轮对话中重复 prefill 一次要读 ~270 GB。因此把 encoder 输出（第 20 层输入，5120 × 4 份 fp16 = 40 KB/token）与压缩 KV、indexer K 按 prompt 前缀 hash 持久化到 `kvcache/`；命中时 encoder 跳过，只跑 decoder bounded replay（128 token）。这与官方"Encoder SWA Bounded Replay + 全局 KV 持久化"一致。64K 上下文的持久化体积 ≈ 2.7 GB，可接受。

---

## 12. 正确性 Oracle

没有可用的官方 CUDA 环境，也没有 GGUF/llama.cpp 参考。oracle 自建，分四层：

| 层级 | 内容 | 判据 |
|---|---|---|
| L0 解码 | FP4/FP8/E8M0 → fp32 解码函数与参考逐位相同。`oracle.py --level l0` 把三张表导出到 `tests/data/l0_dequant.bin`（2,132 B），`tests/test_dequant.cpp` 逐位比对 | 逐位 |
| L1 kernel | 每个 GPU kernel 对随机输入与 fp32 CPU 实现比较；**外加 expert FFN 的真权重版**（下）。**v0.6 实测**：`tests/test_gpu_moe.cpp` 把真实 expert `(0,0)` 与 `(39,383)` 经 IoEngine 读进 GPU slab，十四个 kernel 变体全部 cos = 0.99999996、max\|Δy\| = 1.37e-4（占 \|y\|max），且彼此完全一致；误差下界就是 `x` 的 fp16 舍入（相对 L2 2.04e-4），kernel 自身累加误差可忽略（§7.9.1）。**v0.7 补充**：fp8 shared expert cos = 0.999999947；`h` 的 fp8 量化对参考吻合到 5.0e-8；分组 dispatch 差 1–2 ULP（判据 1e-6，不是逐位）；packed fp16 4.52e-4；**int8 dot4 5.38e-3，略超判据**（§7.9.2）。**v0.8**：`HQuant=3` 与 `HQuant=2` 对 `y_hq16` **同为 5.030e-08**，四种 workgroup 形状给出同一个数（§7.9.3 (a)）；**kernel 外预量化的 int8 `x` 在两个真权重 expert 上是 2.90e-3 与 8.85e-3**（§7.9.3 (b)） | 相对误差 ≤ 1e-3（fp16 传递）；~~int8 路径 ≤ 5e-3~~ **v0.8 定案：判据不为 int8 `x` 放宽。** 误差是 `x` 的、不是 kernel 的，而且**逐 expert 变化 3 倍**（2.9e-3 对 8.9e-3）——一个判据不能靠挑 expert 来满足。`XMode=6` 因此**默认关闭**，只留一个"投机验证批可选"的开关，且**要开也得先由 L2 判定 8.9e-3 能不能接受**（§7.9.3 (b)、§15 未解决项）。dispatch B 的 `xldsi8` 不受影响（它量化的是 `h`，不是这条路） |
| L2 逐层 | 用真实权重跑单层：`tools/oracle.py`（纯 torch fp32，从 safetensors 直接解码权重，逐函数对照 `model.py` 移植）vs deepMoE 每层输出。**v0.7：`--level l2` 已实现并跑通**——它跑的是**未经修改**的 `inference/model.py`（经 `tools/dsref.py` 的六个 CPU kernel shim），把七个层（0 / 1 / 2 / 13 / 14 / 20 / 39）每层 ~40–53 个张量导出到 `tests/data/l2/`（5.1 MB），逐级与整层的实测见 §7.15.1 | 每层输出余弦相似度 ≥ 0.999（**实测层 0 / 39 的 block 输出 0.999935 / 0.999980**），最大相对误差记录并画曲线。**判据用 cos 与相对 L2，不用 max 相对误差**——参考自己是 bf16/fp8，一个元素越过舍入边界会动它自己量级的半个 ulp（§7.15.1） |
| L3 端到端 | 多 prompt × 64 token greedy：deepMoE 与 `oracle.py` 全模型 CPU fp32 前向（每 token 数分钟，跑 ≥ 5 个 prompt）逐 token 一致；投机开/关一致；logits KL 记录。**v0.8：`--level l3` 已实现并跑通**——`tests/data/l3/`（7.1 MB，459 s，一个 prompt）：prefill 之后每层的 window KV、八个 greedy step 的逐步压缩 KV 与 top-k、每步 **top-64 (id, logit)** 加上全 129,280 维的 max / log-sum-exp / min，以及 engram 的 hash 常量表。实测见 §7.16.1 | token 一致率 = 100%（允许在极低 margin 处出现分歧并记录 margin）。**实测：一步 top-1 一致、ρ = 0.97；八步教师强制 7/8、自由运行 6/8**，缺的那一个在**参考自己 margin 0.95** 处、原因是层 2 的 gate 在近似平局上选了不同的第六个 expert（§7.16.1）。**判据未达成，差额已定位到机制而不是未知**；`--level l3` 只跑了 **1 个 prompt，不是 5 个**，所以这一个分歧是样本量为 1 |

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

**L3 抓到了什么，而 L1/L2 结构上抓不到（v0.8 新增，p2_decode.md §3）** —— 这一条是**为什么四层
都要有**的证据，不是一段故事：

| bug | 为什么逐层 / 逐 kernel 的测试看不见 | 症状 |
|---|---|---|
| 融合的 `hc_post` 折进了**错的那个子层**的输出（读 `wob` 而不是 `moe_y`，§7.7） | `tests/test_gpu_layer.cpp` 从 oracle 自己的输入跑一层，于是 `apply_hc_post` 是**关的**，那个槽从来不被读 | 四十层里每层 attention 输出进流两次、MoE 输出被丢掉；ρ = 0.14，max \|Δlogit\| = 13 |
| `engram.slang` 在填 FP8 解码表的 barrier **之前**读它 | 一个 Wave32 内部填与读是同步的，所以**表的前 32 项永远是对的**；单 dispatch 的用例只会命中它们 | 不是失败，是**同样八个 token 每次运行不一样**，偶尔整步 NaN → 采样器报 token 0 / margin 0.0000 |
| 在写合并内存上逐个 float 地算（§7.1 rule 10 / §8.1 第 7 条） | 是性能不是正确性，**任何正确性判据都不会触发** | 6.0 ms/层 = 240 ms/token = 一个热 step 的 26% |

**三条的共同形状：一个逐层或逐 kernel 的测试在结构上无法观察的东西。**
所以 **L3 不是"L2 的加强版"，它是另一个维度**：L1/L2 管"一个 kernel 算得对不对"，
L3 管"把它们接起来时接对了没有"。
**推论（对 §15 的排期）**：L1/L2 全绿不构成把 L3 往后排的理由。

**L3 还留着的两个洞**（都写在测试里）：

1. **只有一个 prompt。** §12 要的是 ≥ 5；现在那一个分歧（§7.16.1 的 step 6）是样本量为 1。
2. **step 6 没有定位到层。** 要定位需要一次 **step 6 的 L2 式导出**，现在的 oracle 不产出。
   同一个工具缺口还挡着 §7.4 的另一件事：**ratio-2 的池化在 pos 64 上没有参考输出**
   （`(64+1) % 2 != 0`），要真正验它需要一次**两步的 decode 导出**。
   **这两件事是同一个扩展**，排在 §15。

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

### 13.4 预期数字（**v0.8：kernel 地板重算，并且第一次有了一个真机数与它对照**）

v0.6 的每 token 时间还是 §3.1 那个三项模型（42 ms 常驻 + 命中读 + NVMe）；
v0.7 把前两项换成实测的 kernel 时间，给出 `48.5 + 27.3 = 75.8 ms` 的计算项。
**v0.8 有两件新的东西**：(1) 两条 track 都变快了，kernel 地板要重算；
(2) **有了一个真机的热步（134 ms，§7.16.2）可以和这个地板对照**，
而这两个数**差 1.8 倍**，差在哪已经逐项定位（§7.16.2）。

**今天的 kernel 地板：**

```
T_kernel ≈ 41.4 ms         # 40 层 dispatch 1–9（40 × 0.893）+ head，实测 §7.15.5
         + 40 × 0.625 ms   # MoE 两个 dispatch + HQuant=3 的第三个，实测 §7.9.3 (a)
           = 25.0 ms
         + 4.9 ms          # engram 两层，实测 §7.16.2（§13.4 v0.7 漏了这一项）
         + 0.8 ms          # compressor(4 层) + indexer(8 层)，实测 §7.4
         + ≈2.8 ms         # collapse + argmax + 写 KV（§7.16.2 的 8.4 ms 尾巴减去已计入 41.4 的 head）
         + 0.4 ms          # ~600 个 dispatch × 0.66 µs（§7.14），可忽略
         ≈ 72–75 ms
```

**三条必须和数字一起读的口径：**

1. **head 只算一次。** §7.15.5 的 41.4 ms 里已经含 head（5.65 ms），
   而 §7.16.2 的 8.4 ms 尾巴也含它——**加起来会重复计一次 head**，上式已经减掉。
2. **MoE 的 0.625 ms 是 v0.2 那一轮 7 个 FP4 槽 + `HQuant=3` 的数**（§7.9.3 (a)）。
   按 §7.9.2 (e)，第七个槽换成**真的** fp8 shared expert 每层还要再加约 0.09 ms，
   地板会往上挪约 3.6 ms 到 **76–79 ms**。**两个口径都记在这里**——
   同轮的"真 shared + `HQuant=3`"那一对还没量，是 §15 的一个小缺口。
3. **engram 是 §13.4 v0.7 漏掉的一项。** 它不大（两层 4.9 ms），但不是零，
   而且其中一半是 48 次随机 4 KiB 读，本该在 token 定下来的那一刻就发出去（§9.5）。

**真机对照（§7.16.2，同一台机器、所有 expert 已驻留）：**

| 项 | 地板 | 热步实测 | 比 |
|---|---:|---:|---|
| 非 MoE（attention + head + collapse） | 44.2 ms | 60.3 ms | 1.4× |
| MoE | 25.0 ms | 68.6 ms | **2.7×** |
| engram | 4.9 ms | 4.9 ms | 1.0× |
| **计算合计** | **≈ 74 ms** | **134.0 ms** | **1.8×** |

**差额三分之二在 MoE，而且几乎全部不在 kernel 里**：P1 的特化还挂着
（`h_quant = 1` 在 `L32 R1` 上 +45.5%）、每次 submit 只跑一次迭代、以及 21 ms 的主机桥。
**"把 134 拉回 ~75"是 Track I 的目标，也是 §15 的第二个里程碑（热步 ≤ 90 ms）。**

**加上 NVMe**，代入 h ≈ **0.920**（5,711 槽，§3.1 的 Q2 曲线内插）：
`0.080 × 240 个 expert × 18.80 MB = 361 MB`，**÷ 4.5 GB/s = 80 ms**。

| 口径 | 计算 | stall | 每 token | TPS |
|---|---:|---:|---:|---:|
| **kernel 地板（Track I 之后）** | 74 ms | 80 ms | **154 ms** | **6.5** |
| **今天的实现（§7.16.2 的热步）** | 134 ms | 80 ms | **214 ms** | **4.7** |
| v0.7 写的 | 75.8 ms | 80 ms | 156 ms | 6.4 |

**注意 stall 那一栏也还没兑现**：真机冷跑时盘只到 **3.0–3.9 GB/s** 而不是 4.5（§7.16.3），
按 3.2 GB/s 算这一项是 ~113 ms，八步那一轮实测每 token ≈ 247 ms → 4.0 tok/s。
**§9.4 的 Planner 线程（重叠下一层的取数）没有在跑**，这是 P3。

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
5. **（v0.8 新增）两个 kernel 口径都是 bench 的，不是真机的。** §7.16.2 量到真机比地板
   慢 1.8 倍，原因逐项已知；**在 Track I 之前，"预期"那一列要读 214 ms 那一行，不是 154 ms 那一行。**

| 场景 | 预期 | 依据 |
|---|---|---|
| **decode，无投机，今天的实现（h≈0.920）** | **4.3–4.9 tok/s** | §13.4 的表给 214 ms / 4.7 tok/s（热步 134 + stall 80）。**八步冷跑实测 0.88 tok/s**，因为那一轮的 h 只有 0.36（§7.16.3）——它是正确性 harness，不是稳态测量 |
| **decode，无投机，Track I 之后（热步 ≤ 90 ms）** | **5.8–6.5 tok/s** | 地板 74 + stall 80 = 154 ms / 6.5 tok/s，留 10% 给真机开销 |
| decode，无投机，h 掉到 0.85（跨语域） | 3.6–4.1 tok/s（今天）/ 4.4–4.8（地板） | 同式，stall 从 80 ms 涨到 150 ms |
| decode，DSpark 平均接受 2.5，5,711 槽 | **9–13 tok/s**（**按地板算**） | 常驻 74 ms ÷2.5 = 30 ms（§10.1；**M>1 的非 MoE 曲线未测**，按平坦假设），expert 流量 ÷1.4（`union_frac[3]` = 0.734）→ stall ≈ 57 ms；MoE 按 §7.9.3 (a) 的 M=3 内插。**前置是 Track I**，否则摊薄的是一个虚高的分子 |
| MoE kernel 有效带宽（M=1，7 个 FP4 槽） | ≥ 210 GB/s | 实测 222.6（§7.9.2）/ 220.7（§7.9.3 同口径），真机管线不应比它差 5% 以上 |
| MoE 每层时间（M=1，7 槽 + `HQuant=3`） | ≤ 0.65 ms | 实测 **0.625**（§7.9.3 (a)）。`HQuant=2` 的 +27% 已经解决；**`runtime/moe_bridge.h` 的默认还是 `h_quant = 1`（+45.5%），改成 3 是一行**（§7.16.2） |
| 投机验证批的 MoE（M=6，7 槽 + `HQuant=3`） | ≤ 0.17 ms/token | 实测 0.158（§7.9.3 (a)）。**判据是 ms/token，不是"≥ 80% 上限"**（§7.9.2 (b)） |
| 非 MoE 路径，一层 | ≤ 0.95 ms | 实测 **0.893 ms**（§7.15.5），在 §3.4 模型的 0.97–1.29 ms 之下。**真 token 上是 1.30 ms**——那不是回归，是 bench 的八层坐在 MALL 里而四十层不在（§7.16.2） |
| **一个热 decode step（全部 expert 驻留）** | **≤ 90 ms（Track I 的目标）** | 今天 **134.0 ms**（§7.16.2）；差额三分之二在 MoE，且几乎全在 kernel 之外 |
| **L3 token 一致率** | 100%（见 §12 的 margin 例外） | 今天 **7/8 教师强制、6/8 自由运行**，缺的一个在参考 margin 0.95 处（§7.16.1） |
| 每 token dispatch 开销 | < 0.5 ms | 实测 0.40 ms（~600 × 0.66 µs，§7.14）。**但真机上每 dispatch 组一次 submit 是 17 ms/token**（128 次 × 0.13–0.15 ms，§7.16.2）——那是 submit 不是 dispatch，两件事 |
| stall_ms / token（5,711 槽） | 75–90 ms | 上面的两个口径给 80 / 82 ms。**真机冷跑时盘只到 3.0–3.9 GB/s**，按 3.2 算是 ~113 ms（§7.16.3） |
| `hot_bytes` / token | 8.5 GB | 实测 **8.52 GB**（`--profile` 按层从 manifest 加出来，§7.16.3），对 §2.3 吻合到三位有效数字 |
| TTFT 4K prompt，冷 cache，单盘 | 40–55 s | §3.2，未重测 |
| TTFT 4K prompt，prefix 命中 | 1–2 s | 未重测 |
| 第二块 NVMe | 上述 NVMe 项 ×0.5 | 未验证 |

这些数字若实测偏离 30% 以上，必须在文档中解释原因。

**与 §1.3 成功标准 2 的关系**：那条要求"达到模型给出的上限的 ≥ 70%"。
上面的算式现在**就是**那个模型，而且它的每一项都有实测来源，
所以 P3 的验收变成一件很具体的事：把 Profiler 的四项分解贴到这四行上。
**v0.8 第一次能算这个比值**：74 / 134 = **55%**，还不到 70%。
`--profile` 与 `--per-layer` 已经在输出 §13.1 的分解（§7.16.3），
**所以现在缺的不是测量，是把 §7.16.2 的三项修掉。**

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
| **P2 GPU 常驻路径**（3–4 周）**已完成（在加载进来的 prefill 状态之上）** | §7 全部 decode kernel，M=1，所有 expert 假设驻留 | [kernel_p2_moe.md](kernel_p2_moe.md)（Track D/H）、[p2_attention.md](p2_attention.md)（Track E/F）、[p2_decode.md](p2_decode.md)（Track G）、`bench/results/{kernel_p2_moe,kernel_p2b_moe,attn_p2}.csv`、`tests/data/{l2,l2x,l3}/` | ✅ **step 1**：九个非 MoE kernel 逐级过 L2（整层 block 输出 cos 0.999935 / 0.999980，§7.15.1），MoE 的 M 扫描 / fp8 shared expert / `h` 量化全部有数（§7.9.2）。✅ **step 2**：§7.4 的 compressor 与 indexer **已产出**（十六个比较点里十一个逐位相同，§7.4）；**四十层 + engram + head + 采样串起来，deepMoE 自己产出了 token**——L3 一步 top-1 一致 / ρ 0.97，八步教师强制 **7/8**、自由运行 **6/8**（§7.16.1）；热步 **134 ms = 7.5 tok/s**（§7.16.2）；`HQuant=3` 把 decode 的量化税降到约 +3%，M=1 MoE **0.625 ms/层**、M=6 **0.158 ms/token**（§7.9.3）。**M=6 的准出条件改写**：~~有效带宽 ≥ 上限的 80%~~ 在 VALU 受限的 kernel 上不是有意义的指标（§7.9.2 (b)），改为 **`ms/token` ≤ 0.17**（实测 0.158）。**这一阶段的边界要说清楚：prefill 的状态（window KV）与每步的压缩 KV / top-k 仍是从 L3 导出加载的**，`Engine::status()` 每次运行都打印 LOADED；**"没有加载状态的 decode"是 P3 的第一件事** |
| **P3 流式 decode**（3 周） | ExpertStore、Planner（全局 LRU）、IoEngine、Profiler。**不含 lookahead** | 端到端 decode，hit/miss/stall 报告 | h、stall 与 `cache_sim` 预测一致（±5 点）；NVMe 利用率 > 80% 在 miss 期；**报告里必须带上实测槽数** |
| **P4 DSpark**（2 周） | 草稿、greedy 验证、confidence 调度、KV 回滚 | 投机 decode | greedy 输出与非投机一致；TPS 增益可测 |
| **P5 Prefill/CED**（2–3 周） | cooperative matrix GEMM、expert-major 流式、bounded replay、prefix 持久化 | 完整对话 CLI | TTFT 数字；oracle 模式与生产模式差异率 |
| **P6 实验** | int8 dot4 路径、Wave64、双盘 stripe、路径 B 的 2 MiB 大页（需 `SeLockMemoryPrivilege`）、路径 B 的 >2 GiB slab | A/B 报告 | 只保留有数据支持的改动。~~CPU 分担 expert~~ 已被 §8.0 否决，移入 §16 |

**P-1、P1 与 P2（step 1 + step 2）的结论已写回本文档（v0.8）。**

### 15.1 里程碑：第一个 token（2026-09-15）

**deepMoE 第一次自己产出了 token。** 从 embedding 查表到 greedy argmax 全在 GPU 上，
四十层、engram、head、采样都是真的，routed expert 由 gate 自己的 ids 经 `store::Planner`
从 NVMe 取回、LRU 淘汰。**对 fp32 参考：一步 top-1 一致、ρ 0.97、八步 7/8（教师强制）与
6/8（自由运行）；一个热步 134 ms = 7.5 tok/s。**（§7.16、[p2_decode.md](p2_decode.md)）

**还不是什么**，一条一条说清楚，因为这决定了下一步排什么：

1. **prompt 留下的状态是加载的**（prefill 是 §11 / P5），**每步的压缩 KV 与 top-k 也是**——
   §7.4 的 kernel 已经存在但还没接进 `Engine`。这两样让 decode step **不自足**。
2. **没有 tokenizer**：`deepmoe run` 吃的是 `--prompt-ids` 里的 token id。
3. **热步是 §13.4 kernel 地板的 1.8 倍**，差额三分之二在 MoE、几乎全在 kernel 之外（§7.16.2）。
4. **冷启动是 NVMe 的形状**：八步 0.88 tok/s，83% 在等盘，**§9.4 的 Planner 线程没在跑**（§7.16.3）。
5. **L3 只有一个 prompt**，§12 要五个。

### 15.2 接下来的里程碑，按顺序

| # | 里程碑 | 判据 | 前置 |
|---|---|---|---|
| **(i)** | **没有 LOADED 状态的 decode** | `Engine::status()` 不再打印 LOADED 那一行；compressor / indexer 接进 `Engine`，prefill 哪怕很慢也由我们自己跑 | §7.4 的 kernel 已有（Track F），缺的是接线 + §7.13 之外的一条慢路径 |
| **(ii)** | **热步 ≤ 90 ms**（Track I） | 全部 expert 驻留时一个 step ≤ 90 ms（今天 134）；§13.4 的"实际 / 上限"≥ 70%（今天 55%） | 三件事，全在 kernel 之外：`MoeBridgeConfig` 换成 §7.9.3 的 P2 特化（`h_quant = 3`）、一个 command buffer 录完 MoE 的三个 dispatch、消掉剩下的主机桥往返 |
| **(iii)** | **DSpark**（P4） | greedy 输出与非投机逐 token 一致；TPS 增益可测 | **Track K 交付 kernel + 已验证的算法**（§7.12 / §10）；前置是 (ii)，否则摊薄的是虚高的分子 |
| **(iv)** | **真正的 prefill**（P5） | TTFT 有数；oracle 模式与生产模式的差异率有数 | §7.13 的 cooperative matrix GEMM + §9.7 的 expert-major 流式；compressor / indexer 的 prefill 形态（整块池化、`compress_lens` 掩码）还没写 |
| **(v)** | **tokenizer** | `deepmoe run --prompt "…"` | 与 §1.2 的 CLI 边界一起做 |

**(i) 和 (ii) 可以并行**：一个是接线，一个是 runtime 的三处改动。
**(iii) 严格排在 (ii) 之后。**

### 15.3 各模块的完成度估计（v0.8）

粗估，只为排期用；"实测"一栏指的是"有没有一个数字支撑这个百分比"。

| 模块 | 完成度 | 缺什么 |
|---|---|---|
| `core` / `model` / `storage` | **95%** | `IoEngine` 的忙碌记账（`nvme_util` 大于 1，§7.16.3）；DirectStorage 后端 |
| `store`（slab / ExpertStore / Planner） | **80%** | 淘汰守卫与逐层 LRU 时间戳（两者都还没有消费者，§7.16 的 Done/Not done）；Planner 的预取线程（§9.4）；分组 dispatch 的启用判据接线（§7.9） |
| `gpu/shaders`（decode 路径） | **90%** | `wo_b` 的 K-split、`sparse_attn` 的 head-group × KV-tile；两者合计值 ~2 ms/token |
| `gpu/shaders`（prefill / DSpark） | **0%** | §7.13、§7.12 |
| `runtime`（decode 循环） | **75%** | LOADED 的两半；预录制的每 token command buffer；主机桥的 21 ms |
| `runtime`（prefill / KV 回滚 / prefix 持久化） | **10%** | §11.2 / §10.2 / §11.4 |
| oracle（L0–L3） | **85%** | L3 的第二到第五个 prompt；两步 decode 导出（同时解掉 step 6 的定位与 ratio-2 池化的验证，§12） |
| CLI | **40%** | tokenizer；对话循环 |
| **端到端 decode** | **60%** | (i) 与 (ii) |

**v0.6 的四项优先级，现在的状态：**

| # | v0.6 的事项 | 状态 |
|---|---|---|
| 1 | 调大 pagefile 并重跑 `heap_capacity` | ✅ **已完成**：100 GiB / **5,711 槽**（比目标的 4,787 还多 924），h ≈ 0.920（§5.2） |
| 2 | DSpark 投机解码，其中"M=6 kernel 的 x 分块进 LDS"是最高优先级 kernel 项 | ⚠️ **那个 kernel 项已作废**（x 分块进 LDS 在每个 M 上都更慢，§7.9.2）。M=6 换成了 packed fp16 + B int8 dot4，0.162 ms/token。DSpark 本身仍未开始 |
| 3 | 端到端流式 decode，Planner 用全局 LRU | 未开始（P3） |
| 4 | 不做 lookahead 预取 | ✅ 维持 |

**v0.7 的六项优先级，现在的状态：**

| # | v0.7 的事项 | 状态 |
|---|---|---|
| 1 | compressor 与 indexer kernel（§7.4） | ✅ **已完成**（Track F，§7.4）。**但还没接进 `Engine`**——这是里程碑 (i) 剩下的那一半 |
| 2 | 40 层 + Engram + head/sampler + L3（"第一个 token"） | ✅ **已完成**（Track G，§15.1、§7.16） |
| 3 | DSpark 投机解码（P4） | 未开始。**v0.8 把它排到里程碑 (ii) 之后**，理由见 §10.1 末 |
| 4 | `HQuant=2` 的 32 行约束（6.3 ms/token） | ✅ **已解决**：`HQuant = 3`，中位数约 +3%，6.3 → 约 1.0 ms/token（§7.9.3 (a)） |
| 5 | `wo_b` / `wq_b` 的 K-split | **一半**：`wq_b` 已经到 **94% 上限**（§7.15.5），靠的是 LDS 预算与 `act_quant` staging，不是 K-split。**`wo_b` 仍是 62%，而且已经排除了激活量化与 DRAM**（`ActQuant = 0` 量到 150.2 对 151.7）——**只剩 K-split**，它要多一个 dispatch、因而要 runtime 改动 |
| 6 | 端到端流式 decode，Planner 用全局 LRU | 未开始（P3）。§7.16.3 已经给出它的基线：0.36 命中率下 83% 在等盘，盘只到 3.0–3.9 GB/s |

**P2 step 2 之后的优先级**（里程碑见 §15.2）：

| # | 事项 | 为什么排这里 |
|---|---|---|
| **1** | **Track I：把热步从 134 拉回 ~90 ms**（里程碑 (ii)） | 三件事全在 kernel 之外，合计约 65 ms/token，是一个 token 上最便宜的一块（§7.16.2） |
| **2** | **compressor / indexer 接进 `Engine` + 一条慢 prefill**（里程碑 (i)） | 删掉 `runtime/decode_state.h` 的 LOADED 两半，decode step 第一次自足 |
| **3** | **端到端流式 decode，Planner 用全局 LRU + 预取线程**（P3） | §7.16.3 说 83% 在等盘而重叠没在跑；**分组 dispatch 的启用判据在这里落地**（§7.9，判据已简化为"本层有没有 expert 没就位"） |
| **4** | **DSpark（P4）** | 前置是 1。**Track K 要交付的是 kernel + 已验证的算法**，不只是 kernel |
| **5** | `wo_b` 的 K-split + `sparse_attn` 的 head-group × KV-tile | 合计约 2 ms/token（§7.15.5、§7.5）；两者都要多一个 dispatch |
| **6** | **两步的 decode 导出**（L2 式，在 step 6 上） | 一次解开两个悬案：L3 step 6 的分歧定位，与 ratio-2 池化的参考验证（§12、§7.4） |
| **7** | **真正的 prefill（P5）+ tokenizer** | 里程碑 (iv)、(v) |

**v0.8 留下的未解决问题**

| # | 问题 | 需要谁来定 |
|---|---|---|
| 1 | **L3 只有一个 prompt**（§12 要五个），所以那一个分歧是样本量为 1 | oracle：再跑四个 prompt（459 s 一个） |
| 2 | **L3 step 6 的分歧没有定位到层**；同一个工具缺口还挡着 **ratio-2 池化的参考验证**（§7.4） | 一次两步的 L2 式 decode 导出，两件事一起解 |
| 3 | **层 2 的近似平局**（§7.16.1）：bf16 的残差流会不会把第六 / 第七个 expert 排回参考的顺序？ | **在优化之前先弄清楚**——残差流现在是 fp32 是有意为之，这是"更精确反而更不一致"的第一个真实例子（§6） |
| 4 | **x 预量化成 int8**：`XMode = 6` 实现了，**两条判据都不达标**（M=6 仍只有 63–67% 上限；精度 8.9e-3）。§7.9.2 (d) 的"唯一能到 85%"已撤回 | **L2**：投机验证批里 8.9e-3 能不能接受；不接受就永久关闭 |
| 5 | **§12 的 int8 判据**：v0.8 定为**不放宽**（误差逐 expert 变化 3 倍，§7.9.3 (b)） | 已定案，除非 4 改变 |
| 6 | A 与 B 的 `RowsPerLane` 也该分开（§7.9.2 (c)），预估收益 < 2% | 低优先级，仍未做 |
| 7 | **run 间 ~7% / 节间 8% 的漂移**。v0.7 有一条结论整个是它造成的（§3.4） | CI：**同轮对照 + 轮转测量 + 一个物理自检**（§7.9.3 (c)） |
| 8 | M > 1 时非 MoE 路径的时间曲线未测（§10.3 按平坦假设） | P4 的第一件事 |
| 9 | **`IoEngine` 的忙碌记账**让 `--profile` 的 `nvme_util` / `nvme_gbps` 差一个约等于队列深度的因子（§7.16.3） | P3：在它修好之前读 `effective_gbps` |
| 10 | **同轮的"真 fp8 shared expert + `HQuant=3`"那一对还没量**，§13.4 的地板因此有 72–75 与 76–79 两个口径 | kernel：一次 `--only` 就够 |
| 11 | **§7.15.5 的 `WaveReduce` 是逐 stage 的**（+30% 到 −14%），没有一条能预测输赢的规则 | 观察项：新加 GEMV 时两种都量一遍 |

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

**P2 step 2 已完成的部分（2026-09-15，v0.8）**

| 项 | 结果 |
|---|---|
| `tools/oracle.py --level l3`（Track G） | prefill 状态、八个 greedy step、每步 top-64 logits、engram 的 hash 表 → `tests/data/l3/`（**7.1 MB，459 s**）。同一个字节偏移被多个层共享（压缩 KV 与 index 源），去重把导出从 22 MB 压到 7.1 MB |
| `gpu/shaders/{compressor,indexer}.slang`（Track F） | 3 + 6 个 stage、十个 pipeline；十六个比较点里**十一个逐位相同**，fp4 字节平面**逐字节**相同（§7.4） |
| `tools/oracle_l2_extra.py` + `tests/data/l2x/` | 2.35 MB / 三个源层：index key cache、index scores、compressor 的携带状态（§7.4） |
| `gpu/shaders/{engram,head}.slang` + `gpu/vulkan/decode_kernels.{h,cpp}`（Track G） | §7.10 的两个 dispatch 与 §7.11 的 greedy sampler |
| `runtime/{engram,decode_state,engine}.{h,cpp}`（Track G） | `NgramHashState` 的 C++ 版与它背后的 48 次 P2 读；LOADED 的那一半（到处都标着）；`init_gpu`（两条内存路径、pinned 集合、双路径 slab 池、全部 pipeline）、`load_decode_state`、`decode_step`、`generate`、§13.1 的逐层时间线、`layer_probe` |
| **pinned 集合一次性加载**（v0.7 的"还没人测过"） | ✅ decode 用的是 **884 个 tensor / 9.17 GiB**，3.4 s / 2.9 GB/s（§9.3 的 17.7 GB 含三个 DSpark 块，decode 不碰）。commit 检查对 pinned 集合与 expert cache 一起做一次，**在分配任何东西之前** |
| `deepmoe run --prompt-ids … --steps …`、`tests/test_decode.cpp`（`suite.decode`） | §7.16 的全部数字 |
| `gpu/shaders/moe_{hquant,xquant}.slang`（Track H） | `HQuant = 3`（采纳，默认）与 `XMode = 6`（实现完整，默认关闭）（§7.9.3） |
| `bench/kernel_bench` 的轮转测量（Track H） | 分组 dispatch 一节改成九个配置轮转，`A + B` 与 `whole` 对得上——**这是 §3.4 那条结论被撤回的原因**（§7.9.3 (c)） |

**还没做的**：prefill（§7.13 / §11）、DSpark（§7.12）、GPU 上温度 > 0 的采样
（§7.11 的 Philox / Gumbel-max 那一半）、`KvCache::snapshot` / `rollback`（§10.2）与
prefix 持久化（§11.4）；compressor / indexer 的 **prefill 形态**（`start_pos == 0`、
整块池化、逐 query 的 `compress_lens` 掩码）；**淘汰守卫**（`ExpertStore::set_guard` /
`set_completed_timeline` 存在但从没被调用过——这条路径上每次 submit 后面都跟一个等待，
所以在途的东西不可能被回收；**command buffer 一旦跨层边界，这句话就不成立了**）；
**逐层的 LRU 时间戳**（Planner 拿到的是 **token** 下标，所以一个 token 的四十层共享一个时间戳，
层与层之间的 tie-break 是排序恰好怎么排的——没出过问题，也不是设计出来的）。

**一个已知的非确定性，记录而不追**（p2_decode.md §4.4）：修掉 §7.16.4 的 barrier 之后
**前十五层逐次运行逐位稳定**，层 20 与 39 不是（探针 cosine 在第七位 / 第三位小数上动，
token 选择稳定但 margin 动几个百分点）。分歧出现在层 14 与 20 之间，
与某个 MoE dispatch 里归约顺序的非确定性一致。**它不改变一个 token**，所以记在这里。

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
| **v0.8** | **P2 step 2 三条 track 的实测全部写回。**（a）**里程碑：第一个 token**（Track G，[p2_decode.md](p2_decode.md)）。四十层 + engram + head + greedy 采样全在 GPU 上；对 L3 oracle **一步 top-1 一致、ρ = 0.97、max\|Δlogit\| = 0.64**，八步**教师强制 7/8**、**自由运行 6/8**，缺的那一个在**参考自己 margin 0.95** 处、根因是层 2 的 gate 在近似平局上选了不同的**第六个** expert（那一层进来 cos 0.9997、MoE 输出 0.9884，比别的层差一个数量级——**是离散分支不是漂移**）→ 新增 §7.16、§12 的 L3 行、§15.1。**热步 134 ms = 7.5 tok/s**（attn 51.9 / MoE 68.6 / engram 4.9 / 尾 8.4），是 §13.4 kernel 预算的 **1.8 倍**，差额逐项已知且**都不在 kernel 里**：P1 的 MoE 特化还挂着（`h_quant = 1` 在 `L32 R1` 上 +45.5%）、每次 submit 只跑一次迭代、21 ms 的主机桥——**Track I 在修**。**冷启动**八步 0.88–1.15 tok/s、**83% 在等 NVMe**、命中率 0→0.80、盘的 `effective_gbps` 3.0–3.9（§7.16.3）；`--profile` 的 **`hot_bytes` = 8.52 GB，对 §2.3 吻合到三位**（按层从 manifest 加出来的，所以是对 §2.3 的检验）。**三个只有四十层才找得到的 bug 升格为规则**：融合的 `hc_post` 读错了子层（逐层测试结构上看不见它，因为那里 `apply_hc_post` 是关的）、`engram.slang` 在填表 barrier 之前读 FP8 解码表（一个 Wave32 内同步，所以前 32 项永远是对的；症状是**每次运行结果不同**与偶发全 NaN）、**在写合并内存上逐个 float 地算**（230 ns 一次 → 6.0 ms/层 = 一个热 step 的 26%）→ **新增 §7.1 rule 10 与 §8.1 第 7 条：`runtime/` 里任何在 GPU 可见指针上跑的标量循环都是 bug，只许整块 `memcpy` / 非临时 store**；§12 新增"L3 抓到了什么而 L1/L2 结构上抓不到"。（b）**Track H（MoE kernel，[kernel_p2_moe.md](kernel_p2_moe.md) v0.2，`kernel_p2b_moe.csv`）**：**`HQuant = 3`** 把 `h` 的量化拆成第三个极小的 dispatch，M=1 回到 `L32 R1`，代价从 +23…29% 降到**四轮中位数约 +3%**（6.3 → 约 **1.0 ms/token**），数值上与 `HQuant = 2` **逐位相同**（对 `y_hq16` 5.030e-08，四种形状同一个数）——**`HQuant = 2` 从此没有理由再用**；顺带把 dispatch B 的块 scale 从每元素一次乘法提到**每块一次**（2 的幂，逐位精确）。**`XMode = 6`（x 在 kernel 外预量化成 int8）实现了但两条判据都不达标**：dispatch A 稳定快 15–30%，**A+B 在 M=6 上仍只有 63–67% 上限**（瓶颈换到了 dispatch B，而 **B 的激活是 `h` 不是 `x`**），端到端 M ≥ 2 只有 0–7% 且不稳定、M=1 负收益；精度 **2.9e-3 / 8.9e-3（逐 expert 差 3 倍）**，**每行一个 scale 更糟**（1.6 倍粗的步长）→ **默认关闭**，§7.9.2 (d) 的"这是唯一能让 M=6 摸到 80/85% 的设计"**撤回**，§12 的 int8 判据定为**不为它放宽**（误差是 `x` 的、不是 kernel 的，而且一个判据不能靠挑 expert 来满足）。**并且撤回 v0.7 自己的一条结论**：§3.4 / §7.9 的「拆分 dispatch 每层 +0.193 ms = 7.7 ms/token（错了 340 倍）」**是测量漂移**——那三个数是**依次**测的，隔着同一轮内 7–8% 的热漂移而要量的差只有 1–3%，同一节还量到过 `whole 0.652 < A 0.520 + B 0.290` 这种物理上不可能的读数；轮转测量给出的真实代价是**只拆 dispatch A 时 ≤ 0.026 ms/层**（40 层 0.2–1.0 ms/token）**而且与一次算完逐位相同** → **「先到的先算」恢复为默认，且只拆 A、B 最后跑一次**，Planner 不再需要"预计等待 > 0.2 ms"的门槛；**量测纪律加一条：要量 1–5% 的差，对照必须轮转着测，并且用一个物理自检钉住**（§7.9.3 (c)、build.md）。（c）**Track F（[p2_attention.md](p2_attention.md) §9–§12）**：**§7.4 的 compressor（3 个 stage）与 indexer（6 个 stage）已产出、不再是加载进来的**，十六个比较点里**十一个逐位相同**，三个不逐位的都以 fp8 GEMV 收尾、坐在 1.6e-3 的 bf16 噪声地板上；**`index_score` 逐位相同是补上参考的三次 bf16 舍入之后才有的**（又一次"比参考更精确"要付代价）；fp4 字节平面**逐字节**相同，一次钉死 E2M1 舍入、nibble 顺序与两种 scale 格式 → **§11.3 的打包压缩 KV 现在是真的**（FP4 block-16 + E4M3 scale，两块分开的 buffer），64K 取 48 MB 那一栏。**三个没被证伪的地方**：`index_score` 是重算的不是捕获的；**ratio-2 的池化在 pos 64 上没有参考输出**（需要两步导出，记为 §12 未决项）；**top-k 在这个上下文上退化**（512 ≥ n），所以 radix select 是单独测的（4096 选 512 带并列，512/512）。**带宽第二轮**：`wq_b` 163 → **204 GB/s（94% 上限）**、`wq_a` → 171、`wkv` → 154、`sparse_attn` 两个 stage 141 → **69 µs**，**一层 dispatch 1–9 从 1.053 降到 0.893 ms，40 层 + head 从 47.7 降到 41.4 ms**；compressor + indexer 另加 **0.81 ms/token（+2%）**。**两个否证**：`wo_b` 的 62% **不是激活量化造成的**（`ActQuant = 0` 量到 150.2 对 151.7）**也不是 DRAM**——只剩 K-split；**`sparse_attn` 的"KV 只读一次"（`heads_per_wg`）是输的**（8 头/wg = 8 个 workgroup 对 40 个 CU，288 µs 对 69），要的是 §7.5 的 **head-group × KV-tile + 部分 softmax 合并**。**`WaveActiveSum` 不是普适的胜利**（逐 stage +30% 到 −14%）。（d）**§13.4 重算**：kernel 地板 = 41.4（非 MoE）+ 25.0（MoE，`HQuant=3`）+ 4.9（engram，v0.7 漏掉的一项）+ 0.8（compressor/indexer）+ ≈2.8（尾巴减去已计入的 head）≈ **72–75 ms**；对真机热步 **134 ms** 是 1.8 倍，**"134 → ~75"写成 Track I 的目标**；口径注写清楚了 head 不能算两次、以及 0.625 是 7 个 FP4 槽的数（换真 shared expert 地板挪到 76–79）。（e）**§15 重写**：P2 标为完成（**在加载进来的 prefill 状态之上**，边界写死）；新增 §15.1 里程碑与"还不是什么"的五条、§15.2 的五个下一里程碑（(i) 没有 LOADED 状态 → (ii) 热步 ≤ 90 ms → (iii) DSpark（Track K 交付 kernel + 已验证的算法）→ (iv) 真正的 prefill → (v) tokenizer）、§15.3 的逐模块完成度；未解决问题从八条变成十一条 | `bench/results/{kernel_p2b_moe,attn_p2}.csv`、`tests/data/{l2x,l3}/`；报告 [p2_decode.md](p2_decode.md)、[kernel_p2_moe.md](kernel_p2_moe.md) v0.2、[p2_attention.md](p2_attention.md) §9–§12 |
| v0.3 | 目标模型锁定 DeepSeek-V4.1-Flash 并解剖；结论改为 **NVMe 主导**；pinned/cached/cold 三层与 slab 池；精度锁定原生 FP4/FP8；按 V4.1 结构逐 kernel 设计（mHC、MQA-latent 稀疏 attention、分组 O 投影、FP4 融合 MoE、Engram、DSpark、head）；每 token 单 command buffer + timeline semaphore；NVMe 测量问题 Q1–Q7 与 trace/模拟工具；lookahead gating 预取；expert-major 流式 prefill；DSpark 验证循环与 confidence 调度；CED / bounded replay / prefix 持久化；四层自建 oracle；阶段重排（P-1、P0、P1 测量前置）；CPU 协同降为 P6；写下预期数字 | ModelScope 权重 header 统计、`inference/model.py`、V4.1 技术报告、工具链实测 |
