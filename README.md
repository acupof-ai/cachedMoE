# deepMoE

Windows Strix Halo（Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / NVMe）专用的超大 MoE 本地推理 Runtime。目标模型：**DeepSeek-V4.1-Flash**（552B backbone + 196B Engram，decode 激活 16B，510 GB 权重）。

- 设计方案：[docs/design.md](docs/design.md)
- 模块依赖与线程模型：[docs/architecture.md](docs/architecture.md)
- 构建与环境：[docs/build.md](docs/build.md)

核心思路：权重原生 FP4/FP8 不再量化；**运行时直接读 ModelScope 下下来的 48 个 safetensors 分片，不 repack**（`deepmoe_manifest.json` 是一份纯地址簿，design §5.1）；常驻部分 pin 在统一内存，routed expert 在 NVMe 与内存之间由数据驱动的 Planner 流式调度；Vulkan Compute（Slang）每 token 一个 command buffer；DSpark 投机解码摊薄带宽；所有优化以 oracle 与可分解的每 token 时间线为准。

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\deepmoe.exe info
ctest --test-dir build --output-on-failure
```

## 当前状态

**P-1、P1 与 P2 step 1 已完成，结论写回 [design.md](docs/design.md) v0.7。**
四份原始报告：[docs/kernel_p1.md](docs/kernel_p1.md)（P1：GPU 微内核与内存路径）、
[docs/route_trace.md](docs/route_trace.md)（P1：路由 trace 与 cache 模拟器）、
[docs/kernel_p2_moe.md](docs/kernel_p2_moe.md)（P2：MoE kernel 为投机解码做准备）、
[docs/p2_attention.md](docs/p2_attention.md)（P2：非 MoE decode 路径，逐级对齐参考实现）。
**一整层 decoder 已经能在 GPU 上跑通并对上参考实现**（层 0/39 的 block 输出 cos
0.999935 / 0.999980）；40 层串起来、compressor/indexer kernel、Engram GPU 路径、
head/sampler 与 L3 是 P2 step 2 的工作。阶段划分与优先级见 design.md §15。

**P2 step 1 改变了设计的四件事**：

1. **容量问题已经解决。** C: 的 pagefile 设成固定 96 GiB 之后，expert cache 从 2,056 槽
   变成 **5,711 槽（100 GiB = 15,360 的 37%）**，超过设计目标（4,787），**h ≈ 0.92**。
   约束不再是 commit 限额，而是 device-local heap（路径 A 74 GiB）与可用物理内存
   （路径 B 再叠 26 GiB）；**装满之后 GPU raw-read 仍是 216 / 207 GB/s，没有惩罚**（§5.2）。
2. **一整层 decoder 是真的了。** 九个 Slang kernel、十八个 pipeline、17.7 GB 的 pinned 权重集、
   一个导出参考实现自身逐级张量的 L2 oracle。一层（dispatch 1–9）**1.071 ms**，
   40 层 + head **48.5 ms/token**；MoE 每层 **0.683 ms**（M=1，含真 fp8 shared expert）。
   **参考实现强制了八处设计更正**（§2.4 / §6 / §7.2–§7.7 / §7.14），其中
   "每一次 fp8 GEMM 之前都要量化激活"不是精度细节——按旧文字写算的是另一个函数。
3. **两条 v0.6 的结论被实测推翻**：(a) §7.1 rule 6 的"M=6 时 x 要分块进 LDS"
   **在每一个 M 上都更慢**（barrier 把访存流水排空），而且 M ≥ 3 的瓶颈是 **VALU 发射**
   不是访存，所以尺子要换成 `ms/token`；(b) §7.9 的"拆分 dispatch 是免费的"
   **错了 340 倍**（每层 +0.193 ms = 7.7 ms/token），改为只在真的要等 I/O 时才分组（§7.9.2）。
4. **NVMe 不再是压倒性的那一项。** h 从 0.75 到 0.92 之后，stall 只占每 token 的
   ≈51%（80 / 156 ms），**投机解码变成第一杠杆**（§10.1 / §13.4）。

| 已实现 / 已实测 | 桩 |
|---|---|
| `core/`：`Result<T>`、4 KiB 对齐、JSON 读取器、Profiler（每 token JSONL） | — |
| `model/`：`config.json` 与 `deepmoe_manifest.json` v2（run/skew 地址簿）解析 + 与 `layout.h` 的交叉校验 | — |
| `tools/manifest.py`：扫 48 个分片 header → `deepmoe_manifest.json`（0.9 s，9.9 MB） | — |
| `tools/oracle.py`：L0 解码黄金表、L1 真权重 expert FFN、**L2 逐层（七个层 × ~40–53 个张量，跑未修改的 `inference/model.py`）** | L3 端到端（§12） |
| `tools/oracle_shared.py`：fp8 shared expert 与 `h` 量化的四个参考答案 | — |
| `tools/route_trace.py` + `dsref.py` + `corpus.py`：**27,399 token / 40 prompt 的真实路由 trace 已跑完**，faithfulness 6 项全过 | — |
| `tools/cache_sim.py`：**Q1–Q4 已回答**（§9.1.1）；12 个可独立验算的单元测试 | — |
| `cpu/dequant` + `cpu::act_quant_block`：FP4 E2M1 / FP8 E4M3 / E8M0 解码与块量化（L0 oracle 与 §6 的复刻判据） | AVX-512 / VNNI 路径 |
| `cpu/gate`：router 数学（sqrtsoftplus + noaux_tc top-k） | ~~lookahead 预测~~（§9.4 已取消） |
| `cpu/gemv`：标量 fp32 参考（L1 oracle 的对照） | AVX-512 GEMV |
| `storage/`：优先级队列 + 切分 + QD 控制；IOCP 后端（`FILE_FLAG_NO_BUFFERING\|OVERLAPPED`）；io_uring 后端（仅交叉编译验证）；**NVMe 直读进 GPU slab 的零拷贝已跑通**（路径 A 3.9 / 路径 B 4.8 GB/s） | DirectStorage（§9.6，需要 SDK） |
| `store/`：slab 池、`Free→Filling→Resident`（每 run 计数）状态机、每 expert 6 项的 GPU 指针表、timeline 淘汰保护；**LRU 基线即最终策略**（§9.3 定案）；`ShardSet`；**`pinned`：17.7 GB pinned 集合经 manifest + IoEngine 进 GPU 可寻址内存** | score-aware（可选开关，默认关）；engram prefetch；**17.7 GB 一次性加载还没测过** |
| `runtime/`：`rope`（按层的 RoPE/YaRN）、`kvstore`、**`decode_layer`（一整层 decoder）**、`moe_bridge`（gate ids → Planner → timeline） | 40 层的 token 循环、engram、dspark、sampler |
| `gpu/vulkan`：device + 能力查询、timeline semaphore、command pool、pipeline、descriptor；**两条内存路径的分配与导入已实测**（§3.3）；`moe_kernels`；**`attn_kernels`：十八个 pipeline + 共享地址表** | 每 token 一个预录制 command buffer（需要按层索引的地址表，P3） |
| `gpu/shaders`：**十六个 kernel，全部过 `slangc` + `spirv-val`**；MoE 侧（`moe_gateup` / `moe_down` / `moe_common`）与 attention 侧（`mega_mhc` / `wq_a` / `wq_b` / `wkv` / `sparse_attn` / `wo_a` / `wo_b` / `gate` / `head` / `attn_common` / `fp8_gemv`）**主体全部实现并 sweep 过** | compressor / indexer（§7.4）、engram（§7.10）、prefill（§7.13）、DSpark（§7.12） |
| `bench/nvme_bench`：Q6/Q7（§9.2.1） | — |
| `bench/bw_matrix`：**CPU / GPU / 并发全部完成**（§8.0、§3.3） | — |
| `bench/kernel_bench`：**P1 的 60 变体 + P2 的 261 行 sweep（M 扫描、`XMode`、`HQuant`、fp8 shared expert、分组 dispatch）**（§7.9.1、§7.9.2、§3.4） | — |
| `bench/attn_bench`：**非 MoE 路径逐 kernel 的 µs / bytes / GB/s**（§7.15.2） | — |
| `bench/heap_capacity`：**100 GiB / 5,711 槽已实测**（§5.2、§9.2.2） | — |
| `cli`：`deepmoe info` | `deepmoe run`（等 P2 step 2 / P3） |

单元测试覆盖：解码表逐位（含 `oracle.py` 导出的黄金表）、对齐、manifest v2 的 run/skew 解析与
自洽校验、slab/ExpertStore 的多 run 填充状态机、LRU、I/O 切分与优先级抢占（含假后端与真实无缓冲
I/O）、Vulkan 内存路径 A/B 的 `(host_ptr, device_address)` 契约、以及用 ModelScope 上真实
`config.json` 做的 §2.1 全字段回归。

另有**需要真 checkpoint** 的测试，默认自动跳过；给出 `DEEPMOE_MODEL_DIR` 后：

- `suite.integration` 用真正的 `IoEngine` + IOCP + `ExpertStore` 把 `(0,0)` 与 `(39,383)` 填进槽，
  比对六个 part 的校验和，并用 `cpu/gemv_fp4_ref` 复算 FFN 与 `oracle.py` 的 torch fp32 结果对照
  （cos = 1.000000000000、最大绝对偏差为输出尺度的 4.5e-7）。
- `suite.gpu_moe` 把同样两个 expert 读进 **GPU 可见的 slab**，跑 §7.9 的两个 kernel 的十四个变体
  对照同一个 oracle（cos = 0.99999996、max|Δy| = 1.37e-4，误差下界就是 `x` 的 fp16 舍入），
  外加 **fp8 shared expert**（cos = 0.999999947）、**`h` 的 fp8 量化**（对参考吻合到 5.0e-8）
  与**分组 dispatch**（1–2 ULP，判据 1e-6）。
- `suite.gpu_attn` 把 §7.2–§7.11 的十二个 stage 逐个对 L2 oracle（最差 cos 0.99991；
  `attn_norm` 在七个层上**逐位相同**）。
- `suite.gpu_layer` 把一整层链起来跑，只有 block 输入与 prefill 的 KV 是 golden
  （层 0 / 39 的 block 输出 cos **0.999935 / 0.999980**，gate **6/6**）；
  它故意只给八个槽的 cache，所以六个 expert 每层都真的从 NVMe 取回来。

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure
```
