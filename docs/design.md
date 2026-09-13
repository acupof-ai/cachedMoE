# deepMoE 设计方案

> Windows Strix Halo 专用超大 MoE 本地推理 Runtime
> 目标模型：DeepSeek-V4.1-Flash（552B backbone + 196B Engram，decode 激活 16B）
> C++20 · Zig 工具链 · Vulkan Compute (Slang) · AVX-512 · Unified Memory · NVMe Expert Streaming · DSpark 投机解码

**核心目标：** 在一台 Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / 单 NVMe 的机器上，把 DeepSeek-V4.1-Flash 的本地 decode 打到这台机器的物理上限，并且能用数据解释"上限在哪、为什么没到"。

文档状态：**v0.4（2026-09-14）**。v0.1 为初始 docx；v0.2 按开发机实测修订内存模型；v0.3 锁定目标模型为 V4.1-Flash，据其真实权重布局重写内存/存储/kernel/prefetch/投机解码设计；v0.4 落地代码骨架，写回 Q6/Q7 实测（§9.2.1），更正 §11.3 的 indexer KV 体积，固定 §14.1 目录。修订记录见 [附录 C](#附录-c-修订记录)。

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
| 存储 | 1 × WD SN740 2 TB NVMe (PCIe 4.0 x4) | C: 383 GB 空闲，D: 671 GB 空闲；**只有一块盘** |
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
2. decode TPS 达到 §3.3 模型给出的、由实测带宽与 hit rate 推出的上限的 ≥ 70%，并能按 LPDDR / kernel / dispatch / NVMe 四类分解每 token 时间。
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

- 残差流是 `[hc=4, 5120]`。每个子层前 `hc_mixes`：把 4×5120 展平做 RMS 归一，乘 `hc_fn [24, 20480]`，经 Sinkhorn（20 轮）得到 `pre[4]`、`post[4]`、`comb[4×4]`（双随机）。**本子层算出的 pre 给下一个子层用**（attention 用上一层 FFN 的 pre，FFN 用本层 attention 的 pre）。
- Gate：`scores = sqrt(softplus(x_f32 · W))`；选路用 `scores + bias`，权重用未加 bias 的 `scores`，归一化后 × 1.5。图像 token 用 `bias_vl`。
- Expert：`gate = clamp(w1 x, max=10)`，`up = clamp(w3 x, ±10)`，`silu(gate) * up`，乘路由权重后再过 w2；6 个 expert 输出与 shared 输出在 fp32 累加。
- Attention：`q = wq_b(rmsnorm(wq_a x))`，RoPE 只作用最后 64 维；`kv = rmsnorm(wkv x)` 后 RoPE，参考实现写入 window cache 前 **量化为 fp8**；输出 `o` 先做 **逆 RoPE** 再进分组 `wo_a`（8 组各 1024×4096）再 `wo_b`。attention 带 `attn_sink`（每头一个 logit）。
- Engram：token id 经"压缩词表"映射（NFKC/小写/空白归一）后与前 3 个 token 做乘法-XOR hash，24 个桶各自取模不同素数；查出 24 × 256 fp8 行，拼成 6144 维过 `wkv [25600, 6144]`，得到 4 份 key + 1 份 value，门控 = 归一化点积的符号平方根过 sigmoid。
- DSpark：草稿输入 = [上一个 token, noise_token × 4]，主模型第 37/38/39 层 attention **输入**的 hc 均值拼接（15360 维）经 `main_proj` 成为草稿 attention 的 KV；输出 5 个位置的 logits，Markov head 逐位加偏置并采样，confidence head 输出每位接受概率。参考 `generate.py` **未使用**投机路径，验证循环要我们自己实现（§10）。
- 数值：`rms_norm_eps = 1e-20`，gate 归一化加 `1e-20`；hc 相关全部 fp32。

---

## 3. 核心判断

### 3.1 这台机器上 V4.1-Flash 的真实瓶颈是 NVMe，不是 LPDDR

decode 每 token 需读 13.0 GB 权重，其中 8.5 GB 常驻内存，4.5 GB routed expert 有 (1−h) 的比例要从 NVMe 读（h 为 expert cache 命中率）。以 LPDDR 有效带宽 200 GB/s、NVMe 5 GB/s 估算：

| h | 常驻读 | 命中读 | NVMe 读 | 每 token | TPS |
|---|---|---|---|---|---|
| 0.30（无局部性、LRU） | 42 ms | 7 ms | 630 ms | 679 ms | 1.5 |
| 0.60 | 42 ms | 14 ms | 360 ms | 416 ms | 2.4 |
| 0.80 | 42 ms | 18 ms | 180 ms | 240 ms | 4.2 |
| 0.90 | 42 ms | 20 ms | 90 ms | 153 ms | 6.5 |
| 1.00（全驻留，不可能） | 42 ms | 23 ms | 0 | 65 ms | 15.3 |

结论：

1. **hit rate 每提高 10 个点，价值超过任何 kernel 优化。** §9 的 prefetch / cache 策略是第一优先级。
2. 常驻部分 42 ms 是 LPDDR 上限，kernel 目标是接近它，而不是超越它。
3. **投机解码是唯一能把常驻 8.5 GB 摊到多个 token 上的手段**，也是把 NVMe 读摊薄的手段（相邻 token 的 expert 重叠度决定摊薄程度，需测量）。
4. 第二块 NVMe 直接把 NVMe 项减半，是性价比最高的硬件升级，设计上预留 stripe。

### 3.2 Prefill 同样被 NVMe 主导

prompt ≥ 64 token 时，每层几乎所有 384 个 expert 都会被激活。encoder 20 层 × 7.2 GB + decoder（bounded replay 128 token，约 86% expert）≈ **270 GB 流式读取 ≈ 50 s**（cache 命中 30% 时约 38 s）。计算侧 4K prompt 仅需 ~3 s。因此 prefill 必须是 **expert-major 顺序流式**（每层 7.2 GB 连续读满 NVMe 顺序带宽），并且需要 **prefix KV 持久化** 避免多轮重复 prefill。

### 3.3 UMA 的实际形态与两条路径

Windows 上 GPU 可用内存 = BIOS VGM（CPU 不可见）+ WDDM shared（约系统内存一半）。当前 VGM=64 GB：CPU 63 GB，GPU device-local 74 GiB，GPU host-visible 37 GiB。

| 路径 | 机制 | expert cache 位置 | 需要 P-1 验证 |
|---|---|---|---|
| **A** | `DEVICE_LOCAL \| HOST_VISIBLE` 类型分配，CPU 映射写入 | GPU heap（74 GiB） | CPU 顺序写入带宽（NVMe 落地）；GPU 读带宽；CPU 读很慢（uncached），CPU 不能算 |
| **B** | 调小 VGM，`VK_EXT_external_memory_host` 导入 CPU 内存 | 系统内存（~110 GB） | GPU 读导入内存的带宽是否打折；单次导入上限；大页支持 |

P-1 在 VGM ∈ {64 GB, 最小值} 两种设置下各测一次，选带宽 × 容量最优者。设计对两条路径都成立：ExpertStore 只暴露 `(host_ptr, device_address)` 对。

### 3.4 单层计算极短，dispatch 开销不可忽视

每层常驻 + 命中权重 ≈ 210–280 MB，200 GB/s 下 **≈ 1.0–1.4 ms**。Vulkan 每次 dispatch + barrier 约 5–20 µs，每层 ~11 次、每 token ~470 次 → 3–9 ms，占 65 ms 的 5–14%。因此：每 token 一个预录制的 command buffer，router 结果留在 GPU，用 timeline semaphore 的 host signal 表达"expert 已就位"，CPU 不在层间回读。

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
   │  CPU Executor    │   │  NVMe layout: hot.bin experts.bin   │
   │  AVX-512 oracle/ │   │  engram.bin mtp.bin kvcache/        │
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

### 5.1 NVMe 文件布局（由 `tools/repack.py` 从 safetensors 生成一次）

| 文件 | 内容 | 布局 |
|---|---|---|
| `hot.bin` | attention、shared expert、router、mHC、norm、engram wkv、embed、head | 按层顺序，每 tensor 4 KiB 对齐；启动时整体顺序读入（~10.5 GB，≈2 s） |
| `experts.bin` | 40 × 384 routed experts | `[layer][expert]`，每个 18,800,640 B；层内连续 7.22 GB → prefill 顺序流式 |
| `mtp.bin` | DSpark 3 块全部权重 | 启动时读入并 pin（7.9 GB） |
| `engram.L1.bin` / `engram.L14.bin` | 384M 行 | 每行 **264 B 交错**（256 B fp8 + 8 B E8M0），一次 4 KiB 读取同时拿到值和 scale |
| `manifest.json` | 每 tensor 的 offset、shape、dtype、块 scale 布局、sha256 | 校验与寻址 |
| `kvcache/` | 持久化 prefix KV（§11.4） | 按 prompt hash 分文件 |

expert 块内部布局：`[w1 rows | w3 rows | w2 rows | s1 | s3 | s2]`，另提供 `w1/w3 行交错` 变体（fused gate/up kernel 一个 wave 同时读到两行）；P2 A/B 后固定一种。

### 5.2 内存布局（以路径 A、VGM=64 GB 为例）

| 区域 | 位置 | 大小 |
|---|---|---|
| pinned 权重（hot + mtp + embed） | GPU heap | 17.7 GB |
| routed expert cache | GPU heap，slab 池 | ~50 GB（2,650 个 expert） |
| routed expert cache（二级） | 系统内存，`external_memory_host` 导入 | ~35 GB（1,860 个） |
| KV cache（64K ctx） | GPU heap | < 0.5 GB |
| I/O staging、engram 行、hash 表、OS | 系统内存 | 其余 |

路径 B 下两级 cache 合并为一级 ~90 GB。两种情况下 cache 容量 ≈ 4,500 个 expert ≈ 29% of 15,360。

### 5.3 Slab 池

- slab = 1.88 GB = 100 个 expert 槽（受 2 GiB 分配上限约束）；~45 个 slab。
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
| 激活 | — | fp16 传递，fp32 归约 | 参考实现把激活量化到 fp8；我们精度更高，oracle 用容差而非逐位 |
| SWA KV cache | fp8 E4M3 块量化（同参考） | | 保持与参考一致，便于逐层对齐 |
| 压缩 KV | oracle 模式 fp8（同 `model.py`）；生产模式 FP4 E2M1 + E4M3/16（同官方部署） | | 两种都实现，后者需通过 §12 容差 |

int8 dot4 备选路径：E2M1 的取值 {0, .5, 1, 1.5, 2, 3, 4, 6} × 2 全是整数，权重可无损映射为 int8 {0,1,2,3,4,6,8,12}；激活按 32 元素块对称量化到 int8。这与参考的 fp8 激活精度同量级但不相同，属于 oracle 容差内的实现选择，用于 M≥4 时 ALU 压力（§7.6）。

---

## 7. GPU Kernel 设计

### 7.1 通用规则

1. **每 token 一个 command buffer**，40 层 × ~11 dispatch 预录制，expert 地址来自 GPU 侧指针表。CPU 只在 MoE 层前通过 timeline semaphore host-signal "该层 6 个 expert 已驻留"。
2. **所有 GEMV 以有效 GB/s 为唯一指标**。每个 kernel 的 benchmark 同时输出 raw-read（同样字节数的纯读 shader）作为上限。
3. **权重读法：一个 wave 协同读一行**。64 lane × 16 B = 1 KiB 连续读；FP4 一行 5120 元素 = 2560 B = 2.5 次；scale 160 B 随行读入。A/B `lanes_per_row ∈ {16, 32, 64}`（16 lane 一行时一个 wave 同时处理 4 行，读粒度 256 B）。
4. **Wave32 优先**（`subgroup_size_control`），GEMV 类 kernel 对比 Wave64。
5. **解码用查表**：FP8 E4M3 → fp16 用 256 项 LDS 表（512 B；E4M3 的 denormal 在 fp16 中是 normal，算术转换要处理特例，查表最简单且精确）；FP4 用寄存器内 16 项表（`select` 链或 `v_perm`）。E8M0 scale 用 `ldexp(acc_block, e − 127)`，即整数加到指数位。
6. **激活放 LDS**：x（5120 fp16 = 10 KiB）或 h（2304 fp16 = 4.5 KiB）整块进 LDS，所有 wave 广播读。
7. **M ≤ 6 时仍按 GEMV 结构**，每 lane 持有 M 个累加器，权重读一次复用 M 次；M > 16（prefill）切换到 cooperative matrix GEMM。
8. 每个 kernel 是 Slang 泛型：`Gemv<WeightFmt, M, LanesPerRow>`，`WeightFmt ∈ {Fp4E2M1, Fp8E4M3, Bf16}`。

### 7.2 Mega-mHC（`hc_mixes + hc_pre + RMSNorm`）

- 输入：残差流 `x[4][5120]` fp32（80 KiB），`hc_fn[24][20480]` fp32（1.97 MB），`hc_base[24]`，`hc_scale[3]`，前一子层的 `pre[4]`，norm 权重。
- 输出：本子层输入 `u[5120]` fp16，以及供下一子层用的 `pre'[4], post[4], comb[4][4]`。
- 访存：`hc_fn` 1.97 MB 是主流量；24 个输出各是一个 20480 维点积 → 24 个 wave 各算一个，然后 Sinkhorn 20 轮在一个 wave 内做 4×4 矩阵行列归一（纯寄存器）。
- 计算注意点：整流 rsqrt 在整个 20480 维上做一次；Sinkhorn 与 comb 严格 fp32；`pre` 加权求和后再 RMSNorm。
- 融合：1 个 dispatch，2 个 workgroup 阶段之间用 LDS，不写回中间量。每层 2 次（attn 前、ffn 前）。

### 7.3 Attention Q 路径（`wq_a → q_norm → wq_b → RoPE`）

- `wq_a`: fp8 [1280 × 5120]，6.6 MB → `qr[1280]`；`q_norm`（RMS over 1280）；`wq_b`: fp8 [32768 × 1280]，41.9 MB → `q[64][512]`；RoPE 作用每头最后 64 维。
- 2 个 dispatch（`wq_b` 需要完整的 `qr`）。`q_norm` 融入第二个 kernel 的输入加载：每 workgroup 先在 LDS 上算 `rsqrt(mean(qr²))` 再读 `qr`。
- fp8 块 scale 索引：`scale[row/32][k/32]`，一行的 K 方向 40 个块 → 每 32 个 K 元素一次 `ldexp`。
- RoPE 融合进 `wq_b` 的写出：输出行 `h*512 + d`，`d ≥ 448` 的行成对（d, d+32）旋转 → 让同一 lane 负责一对行。

### 7.4 KV 路径与 Compressor / Indexer

- `wkv`: fp8 [512 × 5120]，2.6 MB → `kv[512]`，`kv_norm`，RoPE 后 64 维，fp8 块量化写入该层 window cache（环形 128 槽 × 512 B + scale）。1 个 dispatch，极小。
- kv source 层（2, 8, 14, 20）额外：Compressor 每 `ratio` 个 token 输出一个压缩 latent（`wkv_c [512 × 5120]` fp32 + softmax 门控累积到状态），写入该源层的压缩 KV cache；索引 K（fp4，128 维）由 latent 派生。
- index source 层（8 层）：`indexer.wq` 把 `qr` 投到 32 头 × 128（fp4 激活量化），对压缩位置的 K 打分（`ReLU` 后按头加权求和），取 top-512。**这是唯一随上下文长度增长的 decode kernel**：64K 上下文、ratio 2 → 32K 个位置 × 32 头 × 128 = 134 MFLOP，K 数据 32K × 64 B = 2 MB，带宽和算力都很小；top-512 选择用两级 radix select（先每 workgroup 局部 top-512，再合并），层 20 另输出候选块池供 24–36 层限定范围。
- 这些每层只有一两个的 kernel 允许各自独立 dispatch；正确性优先。

### 7.5 稀疏 Attention（decode，单 query）

- 输入：`q[64][512]` fp16，KV 条目 = window 128（fp8）+ 压缩 top-512（fp8 或 fp4）= 640 条 × 512，`attn_sink[64]`，索引数组。
- 数据量：640 × 512 B = 320 KiB；算力 64 × 640 × 512 × 2 ≈ 42 MFLOP。**纯延迟型 kernel**，目标是 < 30 µs。
- 组织：MQA 意味着 64 个头共享同一 KV，所以 **一个 workgroup 处理全部 64 头**，KV 分 tile 进 LDS（32 条 × 512 B = 16 KiB 一个 tile，双缓冲 32 KiB），每个 tile 对 64 头做 QKᵀ、在线 softmax（含 sink 项作为初始 max/sum）、PV 累加。workgroup 256 线程 = 4 wave；每 wave 16 头，lane 负责 1 头的 1/4 维度（128 维 = 64 个 packed fp16 寄存器），跨 4 lane 用 subgroup shuffle 归约点积。
- 融合：q 的 RoPE 已在上游做；输出 `o[64][512]` 的 **逆 RoPE** 融合在写出阶段。
- 输出直接以 8 组 × 4096 的布局写出，供分组 `wo_a`。

### 7.6 输出投影（分组 `wo_a` → `wo_b`）

- `wo_a`: fp8，块对角 8 组，每组 [1024 × 4096]，33.6 MB；输入 `o` 按组切 4096 维。一个 dispatch，workgroup 索引的高位选组。
- `wo_b`: fp8 [5120 × 8192]，41.9 MB。
- 两个 dispatch，输出 `a[5120]` fp16 进 Mega-mHC 的 `hc_post`（§7.7）。

### 7.7 `hc_post`（融合进下一个 Mega-mHC）

`x' = post ⊗ a + comb · x`（4 份拷贝各自加权）。与下一子层的 `hc_mixes` 在同一 kernel：读 `x`、`a`，写 `x'`，同时计算 `x'` 的 mixes。残差流始终 fp32。

### 7.8 Router（Gate）

- `W [384 × 5120]` bf16，3.9 MB；`scores = sqrt(softplus(x·W))`；`+ bias` 取 top-6；权重归一化 × 1.5。
- 1 个 workgroup：384 个点积（6 个 wave × 64 行），top-6 用 wave 内 6 轮 `WaveActiveMax` + 屏蔽。
- 输出到 **host-coherent 缓冲**：`ids[6]`、`weights[6]`，以及 `top16_ids/scores` 供 Planner 的 heat 统计（近似命中）。同时 GPU 侧写一个 `layer_done` 计数器（原子 +1），CPU 自旋等待，不用 fence。
- CPU 拿到 ids → 检查指针表 → 缺失的下 P0 I/O → 全部驻留后 host-signal timeline 值 `token_base + layer`；command buffer 中下一 dispatch 前有对应的 wait。**GPU 在 6 个 expert 就位前不会跑 MoE kernel，其他情况零等待。**

### 7.9 融合 FP4 MoE FFN（每层 2 个 dispatch）

**Dispatch A：`gate/up + SwiGLU`**，7 个 expert（6 routed + 1 shared，shared 为 fp8 路径的模板实例）。

- workgroup 索引 → `(expert_slot, row_block)`；expert 基址 = 指针表 `addr[layer][ids[slot]]`。
- 每 wave 读 `w1` 行 i 与 `w3` 行 i（交错布局时相邻），K=5120：每 lane 32 个 K 元素 = 16 B 的 FP4 + 1 个 E8M0 scale；解码 → 与 LDS 中 `x` 的对应 32 个 fp16 做 FMA（packed fp16 或 fp32），块内和 `ldexp` 后 fp32 累加；wave 归约得 `gate_i`, `up_i`。
- 尾处理：`gate = min(gate, 10)`, `up = clamp(up, ±10)`, `h_i = silu(gate) × up × route_weight`；写 `h[slot][2304]` fp16（7 × 4.5 KiB）。
- ALU 预算：每字节权重（2 个 FP4）≈ 2 次 nibble 提取 + 2 次查表 + 2 次 FMA ≈ 6 op；200 GB/s → 1.2 Top/s，gfx1151 fp32 峰值 ~30 Top/s（fp16 packed 翻倍）；**M=1 时 ALU 富余 20×，M=6 时 3×**，int8 dot4（每指令 4 个 MAC）作为 M≥4 的备选。
- 备注：`w1`、`w3` 的 scale 布局 `[2304][160]`，每行 160 B，与权重行同步读。

**Dispatch B：`down + 归约`**

- 7 个 expert 的 `w2 [5120 × 2304]`，输入 `h[slot]` 在 LDS；每 wave 一行；输出 `y_slot[5120]`。
- 归约：同一 row_block 的 7 个 slot 由同一 workgroup 处理（workgroup 索引只按 row_block，内部循环 7 个 expert），fp32 累加后直接写 `y[5120]`，无需原子、无需第三个 kernel。代价是 workgroup 数减少为 5120/rows_per_wg，用 rows_per_wg=8 → 640 个 workgroup，对 40 CU 足够。
- `y` 进入下一层的 Mega-mHC（`hc_post`）。

**驻留检查的粒度**：一层 6 个 routed expert 全部就位才能发 Dispatch A。为减少等待，Dispatch A/B 也可以按 expert 拆成 2 组（先到的 3 个先算），但会翻倍 dispatch 数；P3 用数据决定。

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
- MoE prefill 按 **expert-major**：对每个 expert，收集路由到它的 token 行（gather），做 M=n_e 的 GEMM，scatter-add 回去。expert 顺序按 `experts.bin` 的物理顺序，和 NVMe 顺序流式一致（§9.7）。
- Attention prefill：window 部分为 band attention；压缩部分需先完成 Compressor 的全序列压缩与 Indexer 打分 — 分块（block 128）实现，正确性对齐参考的因果可见性规则（压缩块只有在完整后才可见）。

### 7.14 每层 dispatch 清单（Reuse 模式层，decode）

| # | kernel | 主要字节 |
|---|---|---|
| 1 | Mega-mHC(attn) + attn_norm | 2 MB |
| 2 | wq_a | 6.6 MB |
| 3 | wq_b + q_norm + RoPE | 41.9 MB |
| 4 | wkv + kv_norm + RoPE + fp8 写 cache（source 层附加 compressor / indexer 1–3 个） | 2.6 MB |
| 5 | sparse attention + 逆 RoPE | 0.3 MB |
| 6 | wo_a（分组） | 33.6 MB |
| 7 | wo_b | 41.9 MB |
| 8 | hc_post + Mega-mHC(ffn) + ffn_norm | 2 MB |
| 9 | gate + top-6 → host-coherent | 3.9 MB |
| — | **timeline wait**（CPU 确认 6 个 expert 驻留） | |
| 10 | FP4/FP8 gate/up + SwiGLU（7 expert） | 6 × 12.5 + 23.6 MB |
| 11 | down + 归约（7 expert） | 6 × 6.3 + 11.8 MB |

合计 ≈ 11 个 dispatch，与 DeepSeek 官方 Reuse 模式 decode 的 11 个 kernel 一致。

---

## 8. CPU 路径

在 V4.1-Flash 的量级下，CPU 的最高价值不是"分担 GEMV"，而是**当 I/O 与预测引擎**。角色按优先级：

1. **IoEngine 与 Planner 的宿主**：IOCP 完成处理、请求切分、cache 管理、预测排序。绑定 2–4 个物理核，避免抢 GPU 带宽。
2. **Engram hash**：int64 乘法-XOR-取模，每 token 24 × 2 次，µs 级。
3. **Lookahead gating**（§9.4）：对预测层 L+d 的 gate 做 bf16 GEMV（3.9 MB），用 AVX-512 在共享内存中的残差流副本上算，40 层 × 3.9 MB = 157 MB/token，与 GPU 重叠。
4. **fp32 oracle 前向**（§12）：完整参考路径，慢但精确；也是 P0 的第一个可运行产物。
5. **AVX-512 / VNNI 量化 GEMV**：FP4 nibble 解码与 `vpdpbusd` 融合的 int8 路径；作为 GPU kernel 的 A/B 参照，以及 P6 "CPU 分担 expert" 实验的执行体。

CPU/GPU 协同分担 expert（v0.1 的核心假设）降级为 **P6 实验**：只有在 hit rate 已高到 NVMe 不再主导（h > 0.9）且 GPU 常驻部分仍未吃满带宽时才有意义。cost model 保持 v0.2 的形式，但 `BW_cpu(load)`、`BW_gpu(load)` 必须来自 P-1 的并发带宽矩阵。

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

### 9.2 测量工具（P1 产物）

- `tools/route_trace.py`：用 P0 的 CPU 前向在语料（中英代码混合，≥ 20K token，多 prompt）上记录 `(token, layer, top6_ids, top16_ids, top16_scores, hidden_norm_input)`；同时记录每个 d 的 lookahead 预测集。输出 Parquet。
- `tools/cache_sim.py`：读 trace，模拟容量 C ∈ {20%, 25%, 30%, 35%}、策略 ∈ {LRU, LFU-decay, ARC, score-aware, static-pin + LRU}，输出 hit rate、miss bytes/token、prefetch 浪费；同时模拟 lookahead 预取（给定 d、K、precision）。
- `bench/nvme_bench.cpp`：Win32 overlapped I/O 微基准，回答 Q6/Q7。
- `bench/bw_matrix.cpp`：CPU/GPU 单独与并发读带宽（P-1）。

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

**仍未完成的 P-1 项**：带宽矩阵的 GPU 与 CPU+GPU 并发部分（需要 §7 的 Vulkan 内存与
dispatch 路径，P2）；两种 VGM 设置下的重测。

### 9.3 分层驻留策略

```
Pinned : hot.bin + mtp.bin + embed        （~17.7 GB，永不淘汰）
Cached : routed experts，slab 池           （~90 GB，策略淘汰）
Cold   : experts.bin 其余 + engram 表       （NVMe）
```

Cache 策略基线为 **LRU + score-aware 提升**：router 每层输出 top-16 分数，未被选中但分数高的 expert（"近似命中"）也刷新 `heat`，淘汰按 `(heat, last_use)` 排序。是否优于纯 LRU 由 `cache_sim` 决定。按层的容量分配先用全局 LRU，若 Q2 显示各层重用距离差异大则改为按层配额。

### 9.4 Lookahead 预取（对 Q4/Q5 的回答决定形态）

- 在第 L 层的 FFN 之后（残差流更新完成），CPU 对层 L+1..L+d 的 gate 权重做 GEMV（输入用 `ffn_norm(mean_hc(x))` 与 `ffn_norm(hc_pre(x, pre_identity))` 两种近似，trace 阶段选 recall 高者），取每层预测 top-K。
- 预测集中未驻留的 expert 按 `score × layer_proximity` 排序进入 P1 队列；每层预取上限 K−6 个"多余"expert 以控制浪费。
- 提前量 d 的选择：`d × T_layer_measured ≥ T_io(18.8 MB) × expected_misses / QD_effective`。以 T_layer ≈ 1.3 ms、T_io ≈ 3.8 ms、6 个 miss 并发估算，d ≥ 3–4 才能完全隐藏；实际值由 Profiler 在线维护的 EWMA 给出。
- 在线自适应：Planner 维护每层的预测 precision（预取后 3 层内被使用的比例）；precision < 阈值时缩小 K，NVMe 队列空闲时放大 K。预取落地的槽以"探针"身份插入 LRU 尾部，用了才提升到头部。

### 9.5 与投机解码的耦合

- verify batch 的 M=k+1 个 token 在每层同时路由，一层的 expert 并集一次性告知 Planner，I/O 并行度天然提高。
- 草稿 token 一旦生成，其 **Engram 行地址完全确定**，立即预取（Q7 保证 ~ms 级到达）。
- Q3 的 Jaccard 决定 verify 长度 k 对 NVMe 流量的边际影响，纳入 §10.3 的调度曲线。

### 9.6 IoEngine

- `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)` + IOCP，专用 2 个完成线程。
- 一个 expert = 18.8 MB 切成 `chunk` 并发下发。**Q6 已实测（§9.2.1）：chunk = 4 MiB、目标 QD = 8（在途 32 MB）** 即达饱和 4.5 GB/s；更高的在途字节只增加延迟（4 MiB × QD64 仍是 4.56 GB/s，但平均延迟从 6.9 ms 涨到 37.7 ms）。默认值见 `core/config.h` 的 `IoConfig`。
- 目标缓冲直接是 slab 槽（路径 A：CPU 映射的 device 内存；路径 B：导入的 host 内存），**零拷贝**。4 KiB 对齐由布局保证。
- 优先级队列：P0 当前层缺失（阻塞 GPU）> P1 lookahead > P2 engram 行 > P3 后台回填（空闲时按静态热度填 free 槽）。P0 到达时可抢占：暂停下发 P1–P3 的新 chunk。
- 每次完成更新 Profiler：字节、延迟、队列深度；每 token 输出 `nvme_busy_ms`、`stall_ms`（GPU 在 timeline wait 上的时间）。
- 双盘：按各盘实测带宽比例分配 expert（静态哈希到盘），而非按盘数平均；`experts.bin` 允许分成 `experts.0.bin`/`experts.1.bin`。

### 9.7 Prefill 流式模式

- prompt ≥ 阈值（由 Q1 与 prompt 长度推算的"预计激活 expert 数 > 容量"）时切换为 **expert-major 顺序流**：按 `experts.bin` 物理顺序读整层 7.22 GB（顺序读，满带宽），每到达一个 expert 就对路由到它的 token 做 GEMM，然后决定它留在 cache 还是丢弃（按 decode 阶段的预测热度）。
- 短 prompt 走 decode 同一套按需路径。
- 无论哪种模式，encoder 20 层先做完，decoder 用 bounded replay（§11.2）只算最后 128 个 token。

### 9.8 验收指标（NVMe 部分）

| 指标 | 定义 |
|---|---|
| hit rate（每层 / 总） | 驻留命中 / 请求 |
| miss bytes / token | 每 token 从 NVMe 读的 expert 字节 |
| prefetch precision / recall | 预取后被用 / 预取总数；被用的 miss 中有多少已在预取中 |
| stall_ms / token | GPU 等 expert 的时间 |
| nvme_util | nvme_busy / T_token |
| 有效 NVMe GB/s | miss bytes / nvme_busy |

---

## 10. DSpark 投机解码

### 10.1 为什么关键

decode 每周期常驻 8.5 GB 读一次，若平均接受 a 个 token，则常驻部分的每 token 成本降为 8.5/a GB；head 1.32 GB 同理。NVMe 部分的收益取决于 verify batch 中 expert 并集的增长（Q3）。在 NVMe 主导的区间，投机解码 + 高 hit rate 是唯二的杠杆。

### 10.2 周期流程

```
1. 主模型 forward(M=k+1: 上一接受 token + k 个草稿)   ← 记录第 37/38/39 层 attention 输入的 hc 均值
2. 逐位验证 → 接受前缀 a ∈ [0, k]，得到 a+1 个新 token（含修正 token）
3. 草稿: forward_embed(main_hidden, last_token) → 3 个 DSparkBlock(M=5) → head → Markov 逐位采样 → confidence
4. 调度器按 confidence 选下一周期的 k ∈ [0, 5]
```

- 验证规则：greedy 模式下逐位比较 argmax；采样模式下用标准的 target/draft 概率拒绝采样（草稿 logits 由 DSpark head + Markov 偏置给出，需保存 5 位的完整分布或 top-k 近似 — 先实现 greedy，采样模式作为 P4 后半段）。
- **不变量**：`temperature=0` 时开关投机解码输出必须逐 token 相同，这是内建的正确性检查。
- 主模型的 window KV / 压缩 KV 状态在拒绝后要回滚：window cache 按位置覆盖即可；Compressor 的部分组状态需保存快照（ratio ≤ 2，状态很小）。

### 10.3 Confidence-scheduled 验证长度

调度器最大化 `E[accepted tokens] / T_cycle(k)`：

- `E[accepted](k) = Σ_{i≤k} Π_{j≤i} c_j`，`c_j` 来自 confidence head（每位条件接受概率）。
- `T_cycle(k)` 来自 **Profiler 在线测得的曲线**：`T_hot(M)`（常驻权重 GEMV/GEMM 随 M 的时间，M=1..6，近似平坦）+ `T_nvme(k)`（并集 expert 的期望 miss 字节 / 有效带宽，使用 Q3 的重叠统计与当前 hit rate）+ `T_draft`（≈ 11 ms）。
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
| L0 解码 | FP4/FP8/E8M0 → fp32 解码函数与 numpy 参考逐位相同 | 逐位 |
| L1 kernel | 每个 GPU kernel 对随机输入与 fp32 CPU 实现比较 | 相对误差 ≤ 1e-3（fp16 传递）；int8 路径 ≤ 5e-3 |
| L2 逐层 | 用真实权重跑单层：`tools/oracle.py`（纯 torch fp32，从 safetensors 直接解码权重，逐函数对照 `model.py` 移植）vs deepMoE 每层输出 | 每层输出余弦相似度 ≥ 0.999，最大相对误差记录并画曲线 |
| L3 端到端 | 多 prompt × 64 token greedy：deepMoE 与 `oracle.py` 全模型 CPU fp32 前向（每 token 数分钟，跑 ≥ 5 个 prompt）逐 token 一致；投机开/关一致；logits KL 记录 | token 一致率 = 100%（允许在极低 margin 处出现分歧并记录 margin） |

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

### 13.4 预期数字（写在前面，事后对照）

| 场景 | 预期 |
|---|---|
| decode，无投机，h≈0.6 | 2–2.5 tok/s |
| decode，无投机，h≈0.85 | 4–5 tok/s |
| decode，DSpark 平均接受 2.5，h≈0.85 | 8–10 tok/s |
| TTFT 4K prompt，冷 cache，单盘 | 40–55 s |
| TTFT 4K prompt，prefix 命中 | 1–2 s |
| 第二块 NVMe | 上述 NVMe 项 ×0.5 |

这些数字若实测偏离 30% 以上，必须在文档中解释原因。

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
| 工具 | Python 3.12 + uv（numpy, safetensors, torch-cpu, pyarrow） | repack、oracle、trace、模拟、报告 |

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
├─ model/                  layout.h          编译期常量（EXPERT_BYTES=18800640 等）
│                          v41_config.{h,cpp}  config.json → struct（§2.1 全字段）
│                          manifest.{h,cpp}    manifest.json 读取与自洽校验
├─ runtime/                engine.{h,cpp}    token 循环编排（仅编排）
│                          block.h attention.h moe.h engram.h dspark.h
│                          sampler.h kvcache.h
├─ store/                  slab.{h,cpp}         slab 池 + SlabBacking 接口
│                          expert_store.{h,cpp} 槽状态机 + GPU 指针表（§5.3/§5.4）
│                          planner.{h,cpp}      LRU 基线已实现，其余策略待 P1
│                          predictor.h/.cpp     lookahead 接口（§9.4，待 Q4）
│                          engram_prefetch.h/.cpp  接口（§7.10）
├─ storage/                file.h file_common.cpp  打开的文件抽象（unbuffered/扇区）
│                          backend.h               抽象后端 {submit, poll, caps}
│                          io_engine.{h,cpp}       优先级队列 + 切分 + QD 控制（§9.6）
│                          windows/ file_win.cpp iocp.cpp directstorage.cpp
│                          linux/   file_posix.cpp io_uring.cpp   （仅 CI 编译）
├─ gpu/
│  ├─ vulkan/              device.{h,cpp} memory.{h,cpp} timeline.{h,cpp}
│  │                       cmdbuf.{h,cpp} pipeline.{h,cpp}
│  └─ shaders/             §7.14 dispatch 清单，每个都过 slangc + spirv-val：
│                          mega_mhc  wq_a  wq_b  wkv  sparse_attn  wo_a  wo_b
│                          gate  moe_gateup  moe_down  head  moe_gemv_fp4
├─ cpu/                    dequant.{h,cpp}      FP4/FP8/E8M0 解码（L0 oracle，已实现）
│                          gemv_avx512.{h,cpp}  标量参考已实现，AVX-512 为桩
│                          gate.{h,cpp}         router 数学（已实现）
├─ bench/                  nvme_bench.cpp   Q6/Q7 微基准（P-1 产物，已实现）
│                          bw_matrix.cpp    带宽矩阵（CPU 部分已实现）
│                          results/          实测 CSV
├─ tools/                  envcheck/vkinfo.cpp
│                          repack.py oracle.py route_trace.py cache_sim.py
├─ tests/                  test_framework.h  自写的 header-only 测试框架
│                          fake_backend.h    确定性的 Backend 假实现
│                          test_core / test_dequant / test_store / test_io / test_model
│                          data/v41_config.json   ModelScope 上的真实 config.json
└─ cli/                    deepmoe_main.cpp  `info` / `bench nvme` / `run`
```

**C++ 标准：代码写作 C++20，但必须用 `-std=c++23` 编译。** `std::expected`
在 `<expected>` 里，libc++ 把它整体挡在 `_LIBCPP_STD_VER >= 23` 之后。
`cmake/deepmoe_options.cmake` 里的 `DEEPMOE_CXX_STANDARD=23` 是唯一的开关。

## 15. 开发阶段

| 阶段 | 范围 | 产物 | 准出条件 |
|---|---|---|---|
| **P-1 可行性**（1–2 周） | 带宽矩阵（两种 VGM × 两条路径）、NVMe 微基准、2 GiB slab 分配 + `external_memory_host` 导入 + NVMe 直读进 GPU 内存的原型 | `bench/` 数字与报告；选定路径 A/B | 三组带宽、Q6/Q7 全部有数；直读零拷贝跑通 |
| **P0 权重与 oracle**（2–3 周） | `repack.py`、`manifest`、`oracle.py`、C++ fp32 CPU 前向（无 GPU） | 能用 CPU 产出正确 token（慢） | L0–L3 全过；trace 工具可跑 |
| **P1 测量**（1 周，与 P0 后半并行） | `route_trace` 跑 ≥ 20K token；`cache_sim` 出策略/容量/d/K 曲线 | 回答 Q1–Q5；选定 cache 策略与预取参数 | 报告写回本文档 §9 |
| **P2 GPU 常驻路径**（3–4 周） | §7 全部 decode kernel，M=1，所有 expert 假设驻留（用小 prompt + 固定 expert 集） | 单 token GPU decode，kernel benchmark 表 | 每 kernel 有效 GB/s ≥ raw read 的 80%；L1/L2 通过；dispatch 开销测出 |
| **P3 流式 decode**（3 周） | ExpertStore、Planner、IoEngine、lookahead、Profiler | 端到端 decode，hit/miss/stall 报告 | h、stall 与 `cache_sim` 预测一致（±5 点）；NVMe 利用率 > 80% 在 miss 期 |
| **P4 DSpark**（2 周） | 草稿、greedy 验证、confidence 调度、KV 回滚 | 投机 decode | greedy 输出与非投机一致；TPS 增益可测 |
| **P5 Prefill/CED**（2–3 周） | cooperative matrix GEMM、expert-major 流式、bounded replay、prefix 持久化 | 完整对话 CLI | TTFT 数字；oracle 模式与生产模式差异率 |
| **P6 实验** | CPU 分担 expert、int8 dot4 路径、Wave64、双盘 stripe | A/B 报告 | 只保留有数据支持的改动 |

**P-1 与 P1 的结论必须写回本文档后才开始 P2 / P3。**

---

## 16. 明确不做

- 不兼容其他模型架构；不做通用 GGUF 加载器。
- 不做 OpenAI API / Web UI / Agent。
- 不做多用户、高并发、EP/TP。
- 不做通用 CUDA/HIP/Vulkan 抽象；HIP 只作 kernel 对照。
- 不重新量化任何权重；不改 router 语义、不改 expert 精度。
- 不接受无法复现的"快很多"；所有优化保留 A/B 与原始数据。
- 不在 P-1、P1 的测量结论出来之前写 Planner 策略代码。
- 第一阶段不实现 vision encoder、1M 上下文、采样模式的投机解码（greedy 先行）。

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
- 命令见 [docs/build.md](build.md#模型权重下载)；完成后 `tools/repack.py --verify` 用 `model.safetensors.index.json` + header 校验大小并生成 `manifest.json`（sha256）。

## 附录 C：修订记录

| 版本 | 变更 | 依据 |
|---|---|---|
| v0.1 | 初始 docx 方案 | |
| v0.2 | CPU+GPU 协同降为待验证假设；补 Windows UMA 拓扑、2 GiB 分配限制、GGUF 对照、oracle 定义、attention/KV/prefill、cost model 查表、单盘、带宽利用率、MTP、工具链 | 开发机实测 |
| v0.4 | 代码骨架落地（§14.1 目录重写、新增 [architecture.md](architecture.md)）；**Q6/Q7 实测写回 §9.2.1**，据此把 chunk 定为 4 MiB、目标 QD 定为 8（原 §9.6 写的"在途 ≥ 64 MB"改为 32 MB：实测 64 MB 在途只增延迟不增吞吐）；**更正 §11.3 indexer K 体积 ~2 MB → ~29 MB**（原值是单个 ratio=2 源层的量）；记录 C++20 代码必须以 `-std=c++23` 编译（libc++ 把 `std::expected` 挡在 C++23 之后） | `bench/nvme_bench` 实测、`tests/test_model.cpp` 回归 |
| v0.3 | 目标模型锁定 DeepSeek-V4.1-Flash 并解剖；结论改为 **NVMe 主导**；pinned/cached/cold 三层与 slab 池；精度锁定原生 FP4/FP8；按 V4.1 结构逐 kernel 设计（mHC、MQA-latent 稀疏 attention、分组 O 投影、FP4 融合 MoE、Engram、DSpark、head）；每 token 单 command buffer + timeline semaphore；NVMe 测量问题 Q1–Q7 与 trace/模拟工具；lookahead gating 预取；expert-major 流式 prefill；DSpark 验证循环与 confidence 调度；CED / bounded replay / prefix 持久化；四层自建 oracle；阶段重排（P-1、P0、P1 测量前置）；CPU 协同降为 P6；写下预期数字 | ModelScope 权重 header 统计、`inference/model.py`、V4.1 技术报告、工具链实测 |
