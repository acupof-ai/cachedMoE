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

**P-1 与 P1 测量已全部完成，结论写回 [design.md](docs/design.md) v0.6。**
两份原始报告：[docs/kernel_p1.md](docs/kernel_p1.md)（GPU 微内核与内存路径）、
[docs/route_trace.md](docs/route_trace.md)（路由 trace 与 cache 模拟器）。
代码仍是骨架 + 已验证的测量路径：GPU kernel 与 Planner 策略的主体是 P2/P3 的工作。
阶段划分与优先级见 design.md §15。

**P1 改变了设计的四件事**：

1. **内存系统是单一 ~217 GB/s 的共享上限**（GPU 216 / CPU 101 / 并发合计 214）——
   "CPU 分担 expert GEMV"从带宽上就不成立，已移入"明确不做"（§8.0 / §16）。
2. **expert cache 的真实约束是 Windows 的 commit 限额，不是内存大小也不是 BIOS VGM**：
   今天只有 **2,056 个槽**（设计要 4,787）。修法是**调大 pagefile**，
   见 [build.md](docs/build.md#windows-虚拟内存pagefile必须先调大)。这是收益最大的一步：
   **≈3.2 → 6.0 tok/s**（§3.1 / §5.2）。
3. **MoE kernel 已经打到内存上限**（218.5 GB/s = raw-read 的 102%，0.602 ms/层），
   **dispatch 开销比估计低一个数量级**（0.6 µs 而不是 5–20 µs）。M=6（投机验证批）
   只有上限的 63%，是唯一的大缺口（§7.9.1）。
4. **lookahead 预取在 27,399 token 的真实 trace 上全程净负收益，已降级为不做**（§9.4）。

| 已实现 / 已实测 | 桩 |
|---|---|
| `core/`：`Result<T>`、4 KiB 对齐、JSON 读取器、Profiler（每 token JSONL） | — |
| `model/`：`config.json` 与 `deepmoe_manifest.json` v2（run/skew 地址簿）解析 + 与 `layout.h` 的交叉校验 | — |
| `tools/manifest.py`：扫 48 个分片 header → `deepmoe_manifest.json`（0.9 s，9.9 MB） | — |
| `tools/oracle.py`：L0 解码黄金表、L1 真权重 expert FFN（manifest 路径 vs `safetensors` 库） | L2 逐层 / L3 端到端（§12） |
| `tools/route_trace.py` + `dsref.py` + `corpus.py`：**27,399 token / 40 prompt 的真实路由 trace 已跑完**，faithfulness 6 项全过 | — |
| `tools/cache_sim.py`：**Q1–Q4 已回答**（§9.1.1）；12 个可独立验算的单元测试 | — |
| `cpu/dequant`：FP4 E2M1 / FP8 E4M3 / E8M0 解码（design §12 的 L0 oracle） | AVX-512 / VNNI 路径 |
| `cpu/gate`：router 数学（sqrtsoftplus + noaux_tc top-k） | ~~lookahead 预测~~（§9.4 已取消） |
| `cpu/gemv`：标量 fp32 参考（L1 oracle 的对照） | AVX-512 GEMV |
| `storage/`：优先级队列 + 切分 + QD 控制；IOCP 后端（`FILE_FLAG_NO_BUFFERING\|OVERLAPPED`）；io_uring 后端（仅交叉编译验证）；**NVMe 直读进 GPU slab 的零拷贝已跑通**（路径 A 3.9 / 路径 B 4.8 GB/s） | DirectStorage（§9.6，需要 SDK） |
| `store/`：slab 池、`Free→Filling→Resident`（每 run 计数）状态机、每 expert 6 项的 GPU 指针表、timeline 淘汰保护；**LRU 基线即最终策略**（§9.3 定案）；`ShardSet` | score-aware（可选开关，默认关）；engram prefetch |
| `gpu/vulkan`：device + 能力查询、timeline semaphore、command pool、pipeline；**两条内存路径的分配与导入已实测**（§3.3） | 完整 decode 管线的 dispatch / submit（§7，P2） |
| `gpu/shaders`：§7.14 的 12 个 kernel 签名齐全，全部过 `slangc` + `spirv-val`；**`moe_gateup` / `moe_down` 主体已实现并 sweep 过 60 个变体** | 其余 kernel 主体（P2）；M=6 的 x 分块进 LDS |
| `bench/nvme_bench`：Q6/Q7（§9.2.1） | — |
| `bench/bw_matrix`：**CPU / GPU / 并发全部完成**（§8.0、§3.3） | — |
| `bench/kernel_bench`：**60 变体 sweep、dispatch 开销、路径 A/B 对照**（§7.9.1、§3.4） | — |
| `bench/heap_capacity`：**两个 heap 的实际可分配上限 = commit 限额**（§5.2、§9.2.2） | 调大 pagefile 后重测 |
| `cli`：`deepmoe info` | `deepmoe run`（等 P2/P3） |

74 个单元测试覆盖：解码表逐位（含 `oracle.py` 导出的黄金表）、对齐、manifest v2 的 run/skew 解析与
自洽校验、slab/ExpertStore 的多 run 填充状态机、LRU、I/O 切分与优先级抢占（含假后端与真实无缓冲
I/O）、Vulkan 内存路径 A/B 的 `(host_ptr, device_address)` 契约、以及用 ModelScope 上真实
`config.json` 做的 §2.1 全字段回归。

另有**需要真 checkpoint** 的测试，默认自动跳过；给出 `DEEPMOE_MODEL_DIR` 后：
`suite.integration` 用真正的 `IoEngine` + IOCP + `ExpertStore` 把 `(0,0)` 与 `(39,383)` 填进槽，
比对六个 part 的校验和，并用 `cpu/gemv_fp4_ref` 复算 FFN 与 `oracle.py` 的 torch fp32 结果对照
（cos = 1.000000000000、最大绝对偏差为输出尺度的 4.5e-7）；`suite.gpu_moe` 把同样两个 expert
读进 **GPU 可见的 slab**，跑 §7.9 的两个 kernel 的十四个变体对照同一个 oracle
（cos = 0.99999996、max|Δy| = 1.37e-4，误差下界就是 `x` 的 fp16 舍入）。

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure
```
