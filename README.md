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

骨架阶段：每个模块都有真实的接口、类型、所有权与线程模型；便宜的部分已完整实现，
昂贵的部分（GPU kernel、模型数学、预测策略）是返回 `Err::Unimplemented` 的桩，
每个都带 `TODO(design §x.y)` 指回设计文档。阶段划分见 design.md §15。

| 已实现 | 桩 |
|---|---|
| `core/`：`Result<T>`、4 KiB 对齐、JSON 读取器、Profiler（每 token JSONL） | — |
| `model/`：`config.json` 与 `deepmoe_manifest.json` v2（run/skew 地址簿）解析 + 与 `layout.h` 的交叉校验 | — |
| `tools/manifest.py`：扫 48 个分片 header → `deepmoe_manifest.json`（0.9 s，9.9 MB） | — |
| `tools/oracle.py`：L0 解码黄金表、L1 真权重 expert FFN（manifest 路径 vs `safetensors` 库） | L2 逐层 / L3 端到端（§12） |
| `cpu/dequant`：FP4 E2M1 / FP8 E4M3 / E8M0 解码（design §12 的 L0 oracle） | AVX-512 / VNNI 路径（§8.5） |
| `cpu/gate`：router 数学（sqrtsoftplus + noaux_tc top-k） | lookahead 预测（§9.4，等 Q4） |
| `cpu/gemv`：标量 fp32 参考（L1 oracle 的对照） | AVX-512 GEMV |
| `storage/`：优先级队列 + 切分 + QD 控制；IOCP 后端（`FILE_FLAG_NO_BUFFERING\|OVERLAPPED`）；io_uring 后端（仅交叉编译验证） | DirectStorage（§9.6，需要 SDK） |
| `store/`：slab 池、`Free→Filling→Resident`（每 run 计数）状态机、每 expert 6 项的 GPU 指针表、timeline 淘汰保护；LRU 基线；`ShardSet` |  score-aware / LFU / ARC（§9.3，等 Q2）；predictor、engram prefetch |
| `gpu/vulkan`：device + 能力查询、timeline semaphore、command pool、pipeline（.spv 加载 + 特化常量） | 内存路径 A/B 分配与导入、dispatch、submit（§7，P2） |
| `gpu/shaders`：§7.14 的 12 个 kernel，签名与绑定齐全，全部过 `slangc` + `spirv-val` | kernel 主体（P2） |
| `bench/nvme_bench`：Q6/Q7 微基准（P-1 产物，结果已写回 design §9.2.1） | `bw_matrix` 的 GPU 与并发部分（等 P2） |
| `cli`：`deepmoe info` | `deepmoe run`（等 P2/P3） |

66 个单元测试覆盖：解码表逐位（含 `oracle.py` 导出的黄金表）、对齐、manifest v2 的 run/skew 解析与
自洽校验、slab/ExpertStore 的多 run 填充状态机、LRU、I/O 切分与优先级抢占（含假后端与真实无缓冲
I/O）、以及用 ModelScope 上真实 `config.json` 做的 §2.1 全字段回归。

另有一条**需要真 checkpoint** 的集成测试，默认自动跳过；给出 `DEEPMOE_MODEL_DIR` 后它用真正的
`IoEngine` + IOCP + `ExpertStore` 把 `(0,0)` 与 `(39,383)` 两个 expert 填进槽，比对六个 part 的
校验和，并用 `cpu/gemv_fp4_ref` 复算 FFN 与 `oracle.py` 的 torch fp32 结果对照（实测
cos = 1.000000000000、最大绝对偏差为输出尺度的 4.5e-7）。

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure
```
