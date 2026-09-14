# deepMoE

Windows Strix Halo（Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / NVMe）专用的超大 MoE 本地推理 Runtime。目标模型：**DeepSeek-V4.1-Flash**（552B backbone + 196B Engram，decode 激活 16B，510 GB 权重）。

- 设计方案：[docs/design.md](docs/design.md)
- 模块依赖与线程模型：[docs/architecture.md](docs/architecture.md)
- 构建与环境：[docs/build.md](docs/build.md)

核心思路：权重原生 FP4/FP8 不再量化；**运行时直接读 ModelScope 下下来的 48 个 safetensors 分片，不 repack**（`deepmoe_manifest.json` 是一份纯地址簿，design §5.1）；常驻部分 pin 在统一内存，routed expert 在 NVMe 与内存之间由数据驱动的 Planner 流式调度；Vulkan Compute（Slang）每 token 一个 command buffer；DSpark 投机解码是**一个有门槛的决策**——它是唯一能摊薄常驻读的手段，但要实测证明在这台机器上赚钱才接入（design §10.1）；所有优化以 oracle 与可分解的每 token 时间线为准。

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\deepmoe.exe info
ctest --test-dir build --output-on-failure
```

## 当前状态：**第一个 token → 设计 v0.9-draft（结构调整，数字待填）**

**design.md v0.9-draft（2026-09-15）改了计划的结构，不是数字**（详见 design §15.2 与附录 C）：

1. **成功标准换成四条可验证的绝对判据**（design §1.3 / §13.3）。外部对照删除：llama.cpp / Colibri / 任何 GGUF runtime
   **都跑不了 V4.1-Flash**（没有 GGUF，FP4/FP8、Engram、DSpark、CED 都不支持），"显著快于通用 runtime"没有对手。
   新判据：**(a)** L3 每 prompt 教师强制 8/8，短上下文与 4K / 16K 都要，自由运行只在参考 margin < 1.0 处分歧；
   **(b)** 实测 decode TPS 对模型 ±15%，kernel bench 解释非 stall 时间的 ≥ 90%；
   **(c)** 常驻路径有效带宽 ≥ 80% × 217 GB/s（≈ 热步 ≤ 75 ms）；**(d)** 内部 A/B：无投机 vs DSpark、冷 vs 热 cache、路径 A vs A+B。
2. **DSpark 挂到一个 go / no-go 门槛上**（design §10.1）。Track K 实测（[docs/p3_dspark.md](docs/p3_dspark.md)）：
   接受 1.93 tokens / verify（单条退化轨迹）、`T_draft` ≈ 19 ms（M = 5 head）、只有 MoE 支持 M > 1、
   fp8 投影 M = 5 时 ×2.2–4.8 → **h = 0.92 下最好 +5%，按实测 M 缩放每个 k 都亏**；stall 在 k = 5 从 80 涨到 286 ms。
   **今天 NO-GO**，等 G1（≥ 5 个正常 prompt 的接受率）、G2（M > 1 的 fp8 投影 kernel）、G3（参考实现的批边界依赖）。
3. **prefill 升格为一等章节**（design §7.13 / §9.7）：已有的慢 prefill、真正 prefill 的五个部件、
   TTFT 模型（冷 cache：64 / 512 / 4K / 16K token ≈ 34 s / 58 s / 62–65 s / 78–88 s，NVMe 主导；kernel 速率 `TBD(Track L)`）、五条验收。
4. **长上下文进入计划**：迄今全部验证在 64–72 token 上（indexer top-k 退化、window 环不回绕、ratio-2 池化几乎没跑），
   **4K / 16K 的 L3 是必需的里程碑**（design §11.5 / §12.1，oracle 与统计 `TBD(Track M)`）。
5. **里程碑顺序**：无 LOADED 状态 → 热步 ≤ 90 ms → **真正的 prefill + 长上下文 L3** → DSpark（过门之后）→ Planner 重叠 → tokenizer。
   前两项 Track I 的提交信息报告已达成（LOADED 已去、热步 **86.7 ms**），**正式写回在下一次集成**。

**以下是 v0.8 的"第一个 token"快照，数字未更新。**

**deepMoE 第一次自己产出了 token。** 四十层、engram、head、greedy 采样全在 GPU 上跑，
routed expert 由 gate 自己的 ids 经 Planner 从 NVMe 取回。对 fp32 参考（`tools/oracle.py --level l3`）：

| | 结果 |
|---|---|
| 一步 decode，logits 对 L3 | **top-1 一致**，对参考 top-64 的 Spearman ρ = **0.97**，max \|Δlogit\| = 0.64 |
| 八步，教师强制 | **7 / 8 一致**（不一致的那一步，参考自己的 margin 是 **0.95**） |
| 八步，自由运行 | **6 / 8**（此后解码的是参考没走过的序列，不计） |
| 残差流，四十层下去 | cos ≥ **0.996**；七个探针层里六层路由与参考 6/6 一致 |
| 一个热步（expert 全部驻留） | **134 ms = 7.5 tok/s**（attn 51.9 / MoE 68.6 / engram 4.9 / 尾 8.4） |
| 八步冷启动（12 GiB cache，命中率 0 → 0.80） | 0.88 tok/s，**83% 在等 NVMe** |

**还不是什么**，因为这决定了下一步排什么：**prompt 留下的状态是加载的**（prefill 是 P5），
**每步的压缩 KV 与 top-k 也是**——§7.4 的 kernel 已经存在但还没接进 `Engine`；
**没有 tokenizer**（`deepmoe run` 吃 token id）；**热步是 kernel 地板（≈75 ms）的 1.8 倍**，
差额三分之二在 MoE 而且几乎全在 kernel 之外；**L3 只跑了一个 prompt**，design §12 要五个。

```
$ deepmoe run --model D:\models\DeepSeek-V4.1-Flash --prompt-ids p.txt --steps 8 --cache-gb 12
state     LOADED from tests/data/l3: window KV after 64 prompt tokens, and the
          compressed KV + indexer top-k of 8 steps (design 7.4's kernels are not ours yet)

step  in     -> out     wall      | attn    moe  (gpu   host)  stall   engram other | hit     nvme
  0    3006 ->    223  1469.9 ms |  70.3   98.5 ( 59.1  25.9) 1276.3    5.2   19.5 |   0/240 4514 MB
  ...
  7       1 ->      0   482.4 ms |  60.5   90.5 ( 49.5  29.8)  312.9    5.9   12.6 | 193/240  884 MB

8 tokens, 9.13 s, 0.88 tok/s, hit rate 0.360, 23.1 GB read from NVMe
6/8 tokens match the fp32 reference before divergence
```

**P-1、P1 与 P2（step 1 + step 2）已完成，结论写回 [design.md](docs/design.md) v0.8；v0.9-draft 是其上的结构调整。**
六份原始报告（第六份是 P3 的 DSpark 规格与实测 [docs/p3_dspark.md](docs/p3_dspark.md)，Track K）；前五份：[docs/kernel_p1.md](docs/kernel_p1.md)（P1：GPU 微内核与内存路径）、
[docs/route_trace.md](docs/route_trace.md)（P1：路由 trace 与 cache 模拟器）、
[docs/kernel_p2_moe.md](docs/kernel_p2_moe.md)（P2 Track D/H：MoE kernel 为投机解码做准备）、
[docs/p2_attention.md](docs/p2_attention.md)（P2 Track E/F：非 MoE decode 路径、compressor/indexer）、
[docs/p2_decode.md](docs/p2_decode.md)（P2 Track G：整个 decode step 与第一个 token）。
下一批里程碑与逐模块完成度见 design.md §15.1–§15.3（v0.9-draft 已重排）。

**P2 step 2 改变了设计的五件事**：

1. **compressor 与 indexer 是真的了**（design §7.4）。十六个比较点里**十一个逐位相同**，
   fp4 字节平面**逐字节**相同——一次钉死 E2M1 舍入、nibble 顺序与一层里的**两种** scale 格式。
   §11.3 的打包压缩 KV 随之成真（FP4 block-16 + E4M3 scale），64K 上省回 18 MB。
   `index_score` 的逐位相同是**补上参考的三次 bf16 舍入之后**才有的：
   又一次"比参考更精确"要付代价。
2. **非 MoE 路径快了 15%**：一层 dispatch 1–9 从 1.053 到 **0.893 ms**，40 层 + head
   从 47.7 到 **41.4 ms**，`wq_b` 到 **94% 上限**。两个否证同样有用：`wo_b` 的 62%
   **不是激活量化**（`ActQuant = 0` 量到 150.2 对 151.7）**也不是 DRAM**，只剩 K-split；
   `sparse_attn` 的"KV 只读一次"**实测是输的**（8 头/workgroup = 8 个 workgroup 对 40 个 CU）。
3. **`HQuant = 3` 把 decode 的量化税从 6.3 降到约 1.0 ms/token**（§7.9.3），
   与 `HQuant = 2` 逐位相同却不挑 workgroup 形状。而 **kernel 外把 x 预量化成 int8 两条判据都不达标**
   （M=6 仍只有 63–67% 上限，精度 2.9e-3 / 8.9e-3、**逐 expert 差 3 倍**）→ **默认关闭**，
   v0.7 那句"这是唯一能让 M=6 摸到 85% 的设计"撤回。
4. **v0.7 自己的一条结论被撤回**：「拆分 dispatch 每层 +0.193 ms（错了 340 倍）」**是测量漂移**。
   轮转测量给出的真实代价是**只拆 dispatch A 时 ≤ 0.026 ms/层，而且与一次算完逐位相同**，
   所以"先到的先算"恢复为默认。**量测纪律因此加一条：要量 1–5% 的差，
   对照必须和被测量的东西轮转着测，并且用一个物理自检钉住**（这里是 `A + B` 必须对得上 `whole`）。
5. **三个只有四十层的链路才找得到的 bug**（design §7.16.4 / §12）：融合的 `hc_post` 读错了子层
   （逐层测试结构上看不见——那里 `apply_hc_post` 是关的）；`engram.slang` 在填表 barrier 之前
   读 FP8 解码表（一个 Wave32 内同步，所以**前 32 项永远是对的**，症状是每次运行结果不同）；
   **在写合并内存上逐个 float 地算**（230 ns 一次 = 一个热 step 的 26%）。
   最后一条升格为规则：**`runtime/` 里任何在 GPU 可见指针上跑的标量循环都是 bug**，
   只许整块 `memcpy` 或非临时 store。

| 已实现 / 已实测 | 桩 |
|---|---|
| `core/`：`Result<T>`、4 KiB 对齐、JSON 读取器、Profiler（每 token JSONL） | — |
| `model/`：`config.json` 与 `deepmoe_manifest.json` v2（run/skew 地址簿）解析 + 与 `layout.h` 的交叉校验 | — |
| `tools/manifest.py`：扫 48 个分片 header → `deepmoe_manifest.json`（0.9 s，9.9 MB） | — |
| `tools/oracle.py`：L0 解码黄金表、L1 真权重 expert FFN、**L2 逐层（七个层 × ~40–53 个张量，跑未修改的 `inference/model.py`）**、**L3 端到端（prefill 状态 + 八个 greedy step + 每步 top-64 logits + engram hash 表，7.1 MB / 459 s）** | L3 的第二到第五个 prompt；两步 decode 导出（§12） |
| `tools/oracle_shared.py`：fp8 shared expert、`h` 量化的四个参考答案、**int8 `x` 的逐 expert 误差对照** | — |
| `tools/oracle_l2_extra.py` + `tests/data/l2x/`：index key cache、index scores、compressor 的携带状态（2.35 MB / 三个源层） | ratio-2 池化的参考输出（要两步导出） |
| `tools/route_trace.py` + `dsref.py` + `corpus.py`：**27,399 token / 40 prompt 的真实路由 trace 已跑完**，faithfulness 6 项全过 | — |
| `tools/cache_sim.py`：**Q1–Q4 已回答**（§9.1.1）；12 个可独立验算的单元测试 | — |
| `cpu/dequant` + `cpu::act_quant_block`：FP4 E2M1 / FP8 E4M3 / E8M0 解码与块量化（L0 oracle 与 §6 的复刻判据） | AVX-512 / VNNI 路径 |
| `cpu/gate`：router 数学（sqrtsoftplus + noaux_tc top-k） | ~~lookahead 预测~~（§9.4 已取消） |
| `cpu/gemv`：标量 fp32 参考（L1 oracle 的对照） | AVX-512 GEMV |
| `storage/`：优先级队列 + 切分 + QD 控制；IOCP 后端（`FILE_FLAG_NO_BUFFERING\|OVERLAPPED`）；io_uring 后端（仅交叉编译验证）；**NVMe 直读进 GPU slab 的零拷贝已跑通**（路径 A 3.9 / 路径 B 4.8 GB/s） | DirectStorage（§9.6，需要 SDK） |
| `store/`：slab 池、`Free→Filling→Resident`（每 run 计数）状态机、每 expert 6 项的 GPU 指针表、timeline 淘汰保护；**LRU 基线即最终策略**（§9.3 定案）；`ShardSet`；**`pinned`：decode 用的 884 个 tensor / 9.17 GiB 一次性加载已实测（3.4 s / 2.9 GB/s）** | score-aware（可选开关，默认关）；engram prefetch；**淘汰守卫与逐层 LRU 时间戳都还没有消费者** |
| `runtime/`：`rope`（按层的 RoPE/YaRN）、`kvstore`、`decode_layer`、`moe_bridge`（gate ids → Planner → timeline）、**`engram`（hash + 48 次 P2 读）**、**`engine`（`init_gpu` / `decode_step` / `generate` / §13.1 的逐层时间线 / `layer_probe`）**、`decode_state`（LOADED 的那一半，到处都标着） | prefill；把 compressor/indexer 接进 `Engine`；dspark；温度 > 0 的采样 |
| `gpu/vulkan`：device + 能力查询、timeline semaphore、command pool、pipeline、descriptor；**两条内存路径的分配与导入已实测**（§3.3）；`moe_kernels`、`attn_kernels`（十八 + 十个 pipeline + 共享地址表）、`decode_kernels` | 每 token 一个预录制 command buffer（需要按层索引的地址表，P3） |
| `gpu/shaders`：**二十一个 kernel，全部过 `slangc` + `spirv-val`**；MoE 侧（`moe_gateup` / `moe_down` / `moe_common` / **`moe_hquant`** / **`moe_xquant`**）、attention 侧（`mega_mhc` / `wq_a` / `wq_b` / `wkv` / `sparse_attn` / `wo_a` / `wo_b` / `gate` / `head` / `attn_common` / `fp8_gemv` / **`compressor`** / **`indexer`**）与 **`engram`** | prefill（§7.13）；**v0.9-draft：DSpark 草稿 kernel 已由 Track K 提交（`dspark_{common,gemv,attn,head}.slang`）**，缺的是 verify batch 要的 M > 1 attention 族 / `head` / `engram`（design §10.1.5 的 G2）；compressor/indexer 的 **prefill 形态** |
| `bench/nvme_bench`：Q6/Q7（§9.2.1） | — |
| `bench/bw_matrix`：**CPU / GPU / 并发全部完成**（§8.0、§3.3） | — |
| `bench/kernel_bench`：**P1 的 60 变体 + P2 的两轮 sweep（261 + 336 行：M 扫描、`XMode` 含 int8 预量化、`HQuant` 三种实现、fp8 shared expert、分组 dispatch 的轮转测量）**（§7.9.1–§7.9.3、§3.4） | — |
| `bench/attn_bench`：**非 MoE 路径逐 kernel 的 µs / bytes / GB/s，两轮**（§7.15.2、§7.15.5） | — |
| `bench/heap_capacity`：**100 GiB / 5,711 槽已实测**（§5.2、§9.2.2） | — |
| `cli`：`deepmoe info`、**`deepmoe run --prompt-ids … --steps …`（§13.1 的每 token 分解，`--per-layer` 是四十行的版本，`--profile` 写 JSONL）** | tokenizer（现在吃的是 token id）；对话循环 |

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
  外加 **fp8 shared expert**（cos = 0.999999947）、**`h` 的 fp8 量化**（`HQuant` 1/2/3 对参考
  吻合到 5.0e-8，且 `HQuant=3` 在四种 workgroup 形状上给出同一个数）、**int8 `x` 预量化的
  逐 expert 误差**（2.902e-3 / 8.856e-3，**差 3 倍**，所以它默认关闭）
  与**分组 dispatch**（只拆 A **逐位相同**；两对 A+B 是 1–2 ULP，判据 1e-6）。
- `suite.gpu_attn` 把 §7.2–§7.11 的十二个 stage 逐个对 L2 oracle（最差 cos 0.99991；
  `attn_norm` 在七个层上**逐位相同**），外加 **`l2_compressor_indexer`**（层 2 / 14 / 20，
  十六个比较点里十一个逐位相同，fp4 字节平面逐字节相同）与 **`indexer_topk_select`**
  （4096 选 512、带一簇并列值，**512/512**——因为在 64 token 的上下文上 top-k 是退化的）。
- `suite.gpu_layer` 把一整层链起来跑，只有 block 输入与 prefill 的 KV 是 golden
  （层 0 / 39 的 block 输出 cos **0.999935 / 0.999980**，gate **6/6**）；
  它故意只给八个槽的 cache，所以六个 expert 每层都真的从 NVMe 取回来。
- **`suite.decode`** 把四十层、engram、head 与采样串起来跑八步，对 `tests/data/l3/`：
  一步 top-1 一致 / ρ 0.97、教师强制 **7/8**、自由运行 **6/8**，外加七个层的逐层探针
  （四十层下去 cos ≥ 0.996）。**层 2 那一行是最值得看的**：进它的流 cos 0.9997，
  它的 MoE 输出却只有 0.9884——gate 在一个近似平局上选了不同的**第六个** expert，
  **那是一个离散分支，不是漂移**。

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure
```
