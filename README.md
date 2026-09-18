# deepMoE

Windows Strix Halo（Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / NVMe）专用的超大 MoE 本地推理 Runtime。目标模型：**DeepSeek-V4.1-Flash**（552B backbone + 196B Engram，decode 激活 16B，510 GB 权重）。

- **单一入口：[docs/STATUS.md](docs/STATUS.md)** — 今天量到了什么、什么被试过并退掉（编号）、测试套件、下一步顺序。**相信任何数字之前先看这里。**
- 设计方案：[docs/design.md](docs/design.md)（v0.9）
- 模块依赖与线程模型：[docs/architecture.md](docs/architecture.md)
- 构建、环境、对话与 worktree 工作流：[docs/build.md](docs/build.md)

核心思路：权重原生 FP4/FP8 不再量化；**运行时直接读 ModelScope 下下来的 48 个 safetensors 分片，不 repack**（`deepmoe_manifest.json` 是一份纯地址簿，design §5.1）；常驻部分 pin 在统一内存，routed expert 在 NVMe 与内存之间由 Planner（精确全局 LRU）流式调度；Vulkan Compute（Slang），command buffer 只在 gate 处切开（每 token 41 次 submit）；prompt 在 GPU 上整块 prefill；**KV 按模型格式算账，sliding-window KV 从不持久化**（design §11.2）；DSpark 投机解码是**一个有门槛的决策**——按实测预测能赚 18–32%，但要先有 M > 1 的 kernel（design §10）；所有优化以 oracle 与可分解的每 token 时间线为准。

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\deepmoe.exe info
ctest --test-dir build --output-on-failure
```

## 当前状态：**它会说话了**（design v0.9，2026-09-15）

```powershell
.venv\Scripts\python.exe tools\chat.py             # 交互对话；/reset /think /drop /temp /top_p /greedy /max /seed /system /stats
.venv\Scripts\python.exe tools\chat.py --think     # 思考模式
```

`tools/chat.py` 起一个 `deepmoe serve`：一个进程常驻 pinned 集合（9.17 GiB）、78.8 GiB expert cache 与 KV；
对话由 checkpoint 自己的 `encoding/encoding.py` 渲染，C++ tokenizer 转 id，流式出 token；新一轮只 prefill 延伸出来的那部分。

| | 今天 |
|---|---|
| 对话 decode | **3.4–4.5 tok/s**（expert cache 命中率 0.84 → 0.90）；每 token ≈ 90–100 ms 计算 + **124–193 ms NVMe stall** |
| 热步（expert 全部驻留） | **81.5 ms = 12.3 tok/s**（64 token 上下文）；91.1 ms（4K）；94.2 ms（17K） |
| 短 prompt TTFT | 5–23 s（11–46 token，走 decode 路径，1.5–3 tok/s） |
| 长 prompt TTFT（GPU prefill，冷 cache） | 64 / 4,133 / 17,010 token：**43 s / ≈ 150 s / ≈ 445 s**（长的由 compute 主导：24 ms / token） |
| 正确性（对 fp32 参考） | 64 token 8/8；**4K / 17K 教师强制 8/8、自由运行 8/8**，取回 prompt 开头埋的 `kestrel-4471-amber"`——从我们自己的 GPU prefill 出发也是 |
| tokenizer | C++ byte-level BPE，对 HF `tokenizers` 24,897 个用例 / 7.69 M id **100% 一致** |
| 采样 | 温度 / top-p：GPU 取 top 集合 + 主机精确核；650 个采样步 0 次回退 |
| KV（按模型格式） | 64K 上下文 61 MB、1M 上下文 0.94 GB；今天的 `KvStore` 17K 就分配 882 MB |

**还缺什么**——就是 design §15.2 里正在并行的四条 track：

- **慢在 NVMe stall**（每 token 的 60%）：GPU prefill 读进来的 expert 不交给 decode cache，Planner 不做层间重叠 → **R1**。
- **KV 不会回退**：prompt 在 k 个 token 前分叉就从 0 重新 prefill（官方 `drop_thinking=True` 每轮都这样，所以 `chat.py` 默认保留思考）；
  `KvStore` 给 40 层都开 bf16 平面、又是单块分配，上下文实际上限 ≈ 41.7K → **R2**（原则：只存 ids + 压缩 KV + index key + compressor 状态，window 靠 bounded replay 重建）。
- **长 prompt 的 prefill 由 compute 主导** → **S**（目标 ≥ 5×）。
- **DSpark 有条件 GO**：单遍单路径的树采样方案预测 ×1.18–1.32，前提是 M > 1 的 attention / head / engram kernel → **T**。
- 另外：`serve` 单会话、不能中途打断；GPU prefill 默认关（~500 token 以上才划算，要给路径 A 留空间）；热步离 §1.3 (c) 的 75 ms 还差 6.5 ms。

**走到这里的里程碑**：P-1 / P1 测量（内存系统是一个 ~217 GB/s 的共享上限、全局 LRU、不做 lookahead）→ P2 常驻路径 kernel 逐级对齐参考 →
**第一个 token**（v0.8：134 ms 热步）→ **nothing loaded + 81.5 ms**（Track I）→ **GPU prefill、4K / 17K 长上下文、对话**（Track J / K2 / L / M / P / Q，v0.9）。

**原始报告**（结论都写回了 design.md，冲突时以后写的为准，v0.9 发现的冲突在 design 附录 C）：
P1 [kernel_p1.md](docs/kernel_p1.md)、[route_trace.md](docs/route_trace.md)；
P2 [kernel_p2_moe.md](docs/kernel_p2_moe.md)、[p2_attention.md](docs/p2_attention.md)（§13 是 Track J）、[p2_decode.md](docs/p2_decode.md)（§9–§13 是 Track I）；
P3 [p3_dspark.md](docs/p3_dspark.md)（K / K2）、[p3_prefill.md](docs/p3_prefill.md)（L）、[p3_longctx.md](docs/p3_longctx.md)（M）、
[p3_longctx_decode.md](docs/p3_longctx_decode.md)（Q）、[p3_chat.md](docs/p3_chat.md)（P）。

| 已实现 / 已实测 | 桩 / 没做 |
|---|---|
| `core/`：`Result<T>`、4 KiB 对齐、JSON 读写、Profiler（每 token JSONL） | — |
| `model/`：`config.json` 与 `deepmoe_manifest.json` v2（run/skew 地址簿）+ 与 `layout.h` 的交叉校验 | — |
| `text/`：**C++ tokenizer**（对 HF 100%，`deepmoe tokenize`） | engram hash 常量在 C++ 里推导（今天读 `tests/data/l3`） |
| `storage/`：优先级队列 + 切分 + QD；IOCP 后端（`FILE_FLAG_NO_BUFFERING\|OVERLAPPED`）；NVMe 直读进 GPU slab 零拷贝；忙碌记账按在途窗口并集 | DirectStorage |
| `store/`：slab 池（路径 A + B，自动定大小，实测 4,500 槽）、每 expert 6 项的 GPU 指针表、**Planner = 与 `cache_sim` 逐步相同的全局 LRU**、pinned 集合（9.17 GiB / 3.4 s） | `ExpertStore::adopt`（prefill → cache，R1）；层间重叠；淘汰守卫没有消费者 |
| `cpu/`：FP4 / FP8 / E8M0 解码、`act_quant_block`、router 数学、**`dspark_tree`**（树采样的格 / 路径 / confidence / 温度 1 精确接受，对 Python 参考逐位） | AVX-512 GEMV |
| `gpu/vulkan`：两条内存路径、timeline（驻留 + 完成两条）、`moe_kernels`、`attn_kernels`、`decode_kernels`（engram / head / **sample_topk**）、**`dspark_kernels`**、**`prefill_kernels`（`gpu::Prefill`）** | M > 1 的 attention 族 / head / engram（T）；Track J 的 K-split / tiled 接口还没被 runtime 采纳 |
| `gpu/shaders`：decode 全链（mega_mhc / wq_a / wq_b / wkv / sparse_attn / wo_a / wo_b / gate / head / compressor / indexer 含 candidate block / engram / MoE 三件 / sample_topk）、Track J 的 `gemv_ksplit` / `sparse_attn_t`、DSpark 草稿四个、prefill 五个；全部过 `slangc` + `spirv-val` | GPU 上的 indexer top-k（prefill）；`gate.slang` 的 softplus 数值 |
| `runtime/`：`engine`（token 循环、`shared_attn` 路由、在 gate 处切开的 buffer、`slow_prefill`、`gpu_prefill` / `seed_from_prefill`、会话）、`kvstore`、`decode_layer`（列表长度加固）、`moe_bridge`（一个 7 槽 runner）、`engram`（token 开始时预取）、`decode_state`、**`session`**、**`sampling`** | KV 回退 / 持久化、per-source 平面（R2）；DSpark 验证循环（等 T） |
| `cli`：`info`、`bench nvme`、`run`（`--slow-prefill --loaded-ced --warm --determinism --gate-report --topk-report --per-layer --profile`）、`tokenize`、**`serve`** | 回退、中断、多会话 |
| `tools/`：`manifest.py`；oracle L0–L3 + `oracle_l2_extra` / `oracle_prefill` / `oracle_longctx` / `oracle_dspark` / `dspark_tree`；`route_trace` + `cache_sim`；**`chat.py`**、`encoding_check.py`、`tokenizer_golden.py` | 64 token 的第 2–5 个 L3 prompt |
| `bench/`：`nvme_bench`、`bw_matrix`、`kernel_bench`、`attn_bench`、`heap_capacity`、**`prefill_bench`**、**`dspark_bench`** | 安静机上同口径的 TTFT |

需要真 checkpoint 的测试默认自动跳过；给出 `DEEPMOE_MODEL_DIR` 后跑（命令与每个 suite 验什么见 build.md）：

- `suite.integration` / `suite.gpu_moe` / `suite.gpu_attn` / `suite.gpu_layer`：读路径、MoE kernel、attention 逐 stage 与整层对 L1 / L2 oracle。
- **`suite.decode`**：四十层、八步对 L3（从导出状态 8/8 / 8/8，从慢 prefill 7/8 / 6/8，差的是一个参考 margin 0.95 的近似平局）。
- **`suite.decode_longctx`**：4K / 17K 的 indexer kernel 在参考输入上 tie-aware、引擎逐层逐步、8 步教师强制 + 8 步自由运行（需要 `traces/longctx/`，见 `DEEPMOE_LONGCTX_DIR`）。
- **`suite.gpu_prefill`**：prefill 逐 stage（110 项）、64 token 进引擎；`DEEPMOE_PF_LONGCTX` 给出长上下文导出时跑 4K / 17K 交接与之后的 decode。
- **`suite.gpu_dspark`**：DSpark 草稿 kernel 逐阶段对参考。
- 纯 CPU：**`suite.tokenizer`**（909 个黄金用例）、**`suite.sampling`**（L3 logits 上 200,000 次抽样的 χ²）、**`suite.dspark_tree`**（对 Python 参考逐位）。

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure
```
