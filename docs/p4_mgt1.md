# P4 Track T — M>1 decode kernel：状态、证据与交接

> ⚠️ **阻塞骨架（2026-09-15 会话）。本文件不含任何本次实测数字。**
> 本会话在 DSH 沙箱 setup 阶段即失败：所有需要启动进程的工具（`pwsh` / `git` / `glob` / `grep`）
> 一律返回 `SetNamedSecurityInfoW failed (Win32 5): grantWrite(C:\Users\Asus\code\deepmoe\build\p4-wt\t)`，
> 命令从未执行。因此 **构建、4K 测试、bench 全部未跑，任务 1–3 未完成，无 commit**。
> 下面第 3 节的"证据"**转述自任务文件 `p4_tasks/tn.txt` 记录的上一轮结果**（`mgt1_layer_4k.log`），本次无法复核；
> 所有标 `TBD(未测)` 的格子等环境恢复后按第 6 节补齐。最小复现与恢复步骤见第 6 节。
> marker：`build/p4_done.json`（共享路径 `build/p4_done/t.json` 因越界写被沙箱拒绝）。

---

## 1. 本 track 的任务（tn.txt）

| # | 任务 | 本次状态 |
|---|---|---|
| 1 | 把未提交的 5 个文件小步提交（softplus 修复、`attn_common`、`decode_layer` batch 收尾）+ 构建 | ❌ 未做（无 shell） |
| 2 | 4K 跑 `gpu_layer.mgt1_layer_batch_vs_steps`（M = 2/4，支持则 5/6）+ `decode.forty_layers`；确认 7/8 是已知 step-7 近似平局、无新回归 | ❌ 未做 |
| 3 | M = 1/2/4/6 每层代价 bench，报缩放，判定 design §10.1.5 G2（M=6 ≤ ×3.06 @ M=5 线性外推）是否成立；存 `bench/results/mgt1_p4.csv` | ❌ 未做（csv 未生成） |
| 4 | 写 `docs/p4_mgt1.md`（kernel 清单 / 正确性证据 / M 缩放表 / 缺口） | ⚠️ 仅本骨架 |
| 5 | 写 done marker（共享路径或 worktree fallback） | ✅ 已写（status = blocked） |

## 2. Kernel 清单（来源：tn.txt 现状段；本次无法列目录核对，入口与行号待补）

| 部件 | 位置（已知） | 说明 | 本次核对 |
|---|---|---|---|
| M>1 decode kernel | `gpu/shaders/mgt1_*.slang`（其中 `mgt1_gate.slang` 明确在列） | batch（M>1）路径的 shader 族 | 未核对 |
| gate softplus 修复 | `gpu/shaders/gate.slang`、`gpu/shaders/mgt1_gate.slang` | `log1p(exp(-|z|)) + max(z,0)` 数值稳定写法 | 未核对（文件在 dirty 列表） |
| 共用 attention 定义 | `gpu/shaders/attn_common.slang` | dirty 列表内 | 未核对 |
| batch runner | `MgtRunner`（头/实现路径待补） | M>1 的 kernel 调度 | 未核对 |
| batch decode 路径 | `runtime/decode_layer.{h,cpp}` | DecodeLayer batch 路径（dirty 列表内） | 未核对 |
| 层内批量对照测试 | `gpu_layer.mgt1_layer_batch_vs_steps`（ctest 名） | batch vs 逐 step 逐步对照 | 未跑 |
| 40 层端到端 | `decode.forty_layers`（ctest 名） | 教师强制 + 自由运行 | 未跑 |
| indexer 权重 | 已把 scale 折进权重（HEAD 三个提交之一） | — | 未核对 |

HEAD（据 tn.txt）：`b4f0e83`，3 个提交（`tools/oracle_dspark.py --mgt1 verify-batch` 参考、M>1 decode kernel、indexer scale 折叠）。
dirty（据 tn.txt，本次未 `git status` 复核）：`gpu/shaders/attn_common.slang`、`gate.slang`、`mgt1_gate.slang`、`runtime/decode_layer.{h,cpp}`。

## 3. 正确性证据

**上一轮（转述 tn.txt，摘自 `mgt1_layer_4k.log`，本次未复核）**

| 项 | 结果 |
|---|---|
| L14 M=2 / M=4 batch-vs-steps 余弦 | ≈ 0.99999（`wq_a` / `q` / `kv` / `attn` / `wo_a` / `wo_b` / `ffn_norm`） |
| gate 集合 | M=2 **2/2**、M=4 **4/4** |
| ring overflow | equal |
| compressed rows / index keys | bit-equal |
| top-k 列表 | identical |
| `decode.forty_layers` | 教师强制 **7/8**、自由运行（自跑 prefill）**7/8**；缺的一个在 step 7，已知近似平局 |

**本次新增证据**：无（未跑任何测试）。

## 4. M 缩放表（2026-09-17 实测，`bench/results/mgt1_p4.csv`）

`tests/test_gpu_layer.cpp::gpu_layer.mgt1_m_curve`：每个 (上下文, 层, M) 跑 30 次计时迭代，
一次 = `bind_batch` + `run_attention_batch`，即一条完整的**非 MoE** attention 链（mhc ×3、
wq_a/b、q_norm、wkv、sparse_attn、wo_a/b、gate score/topk，含提交与栅栏）。
层 2 是纯窗口层，层 20 是压缩/索引源层。复现：

```powershell
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
$env:DEEPMOE_MGT1_CTX="l3,ctx4k"; $env:DEEPMOE_LONGCTX_DIR="C:\Users\Asus\code\deepmoe\traces\longctx"
$env:DEEPMOE_MGT1_ITERS="30"; $env:DEEPMOE_MGT1_CSV="bench\results\mgt1_p4.csv"
.\build\tests\deepmoe_tests.exe "gpu_layer.mgt1_m_curve"
```

| 上下文 | 层 | M=1 | M=2 | M=4 | M=6 | **C(6)/C(1)** |
|---|---|---:|---:|---:|---:|---:|
| l3（64 token） | 2 | 1.283 | 1.664 | 2.512 | 3.384 | **2.64×** |
| l3 | 20 | 1.332 | 1.630 | 2.421 | 3.206 | **2.41×** |
| 4K | 2 | 1.523 | 2.039 | 3.180 | 4.234 | **2.78×** |
| 4K | 20 | 1.566 | 2.177 | 3.481 | 4.669 | **2.98×** |

（ms/层；每 token 的摊薄在 CSV 的 `ms_per_token`：4K 层 20 是 1.566 → 0.778。）

**判定：G2 的前半段成立**（判据 M=6 ≤ ×3.06）：最坏一格（4K、压缩源层 20）**2.98×**，
短上下文 2.4–2.6×——投机 verify 的 6 个 token 在非 MoE 半边只花 2.4–3.0 个 token 的钱，
M=2 就已经把每 token 代价压到 0.6–0.8×。

**但这条曲线只覆盖 MoE 之外的一半。** l3 的 M=1 是 1.28–1.33 ms/层 → 40 层 ≈ **51 ms**，
而实测完整热步是 **81.5 ms**，差额 ~30 ms 是 MoE + head。M=6 那次前向要跑 6 个 token 的
MoE，而 MoE 是**按需从 NVMe 取 expert** 的（docs/p4_hitrate.md §8：decode 被 8.3–9.2 GB/s
的读带宽钉住），6 个 token 的 expert 并集 ≈26 个/层（docs/p3_dspark.md §12.6）——
**verify 的 MoE 代价不随 M 摊薄，它随并集线性长。**
所以下一块要量的不是 C(M) 而是 **MoE(M)**：`GpuMoeBridge::run_batch` 已经能把 M 个
不同 expert 集的 token 当一批跑（逐列 dispatch，见 docs/p4_dspark_runtime.md §2.2），
要补的是并集驻留后每层 MoE 的 M=1/2/4/6 时间。

### MoE(M)：不摊薄，是线性的（2026-09-17 实测）

`tests/test_gpu_moe.cpp::mgt1.moe_m_curve`（`ctest -R bench.mgt1_moe_m_curve`，
`DEEPMOE_MOE_M_ITERS` 设迭代数）：层 0 的一层 MoE，**每列 6 个互不相交的 expert**
（M=6 时并集 36/384），全部先驻留，所以量到的是 kernel 代价、不含取 expert 的 I/O。

**加列掩码（`pc.m`）之前**——每个 dispatch 都按编译期 M=6 算全部 6 列：

| M | ms/层 | ms/token | C(M)/C(1) |
|---:|---:|---:|---:|
| 1 | 1.786 | 1.786 | 1.00× |
| 2 | 3.615 | 1.808 | 2.02× |
| 4 | 7.484 | 1.871 | 4.19× |
| 6 | 11.674 | 1.946 | **6.54×** |

**加列掩码之后**（`GateUpPush::m` / `DownPush::m`，shader 的每个 `for (m = 0; m < M; …)`
改成 `m < pc.m`，`MoeRunner::set_live_columns`）：

| M | ms/层 | ms/token | C(M)/C(1) |
|---:|---:|---:|---:|
| 1 | **1.044** | 1.044 | 1.00× |
| 2 | 2.206 | 1.103 | 2.11× |
| 4 | 5.114 | 1.279 | 4.90× |
| 6 | 8.711 | 1.452 | **8.34×** |

（`gpu_moe` 10 个用例全过，含 M=6 的 10 个变体对 oracle、以及分两次 submit 的
split-A 逐位一致；所以掩码没有改变任何数值。）

两个结论：

1. **M=1 的 MoE 快了 1.7×（1.786 → 1.044 ms/层）**，纯粹因为不再算 6 列里用不到的 5 列。
   这是 decode 热步的直接收益，不需要投机解码 —— 40 层省下约 30 ms/token。
2. **MoE 仍然不随 M 摊薄**（8.34×），因为每列的 6 个 expert 是**不同**的：
   `ids()` 是一组专家，所以一批不同路由的 token 只能逐列 dispatch，每列都要完整读一遍
   那 6 个 expert 的权重。每条 token 的 MoE 代价基本恒定（1.04 → 1.45 ms/token），
   多出来的部分是每列多一次 dispatch 的固定开销。**要让它摊薄，只能让一次 dispatch
   同时吃多列的不同 expert 集**（`docs/p4_dspark_runtime.md` §2.2 的方案 C），
   而那条路需要"每 token 一个物理槽行"的两级索引（共享 expert 那一行是关键），
   目前的 `Ids[slot]` 单级索引表达不了。

### 端到端 A/B：掩码的收益在墙钟上看不见（2026-09-17 实测）

同一套脚本（`bench/one_config.py`，等价于 `config_sweep` 的 cache-4500 格：4 轮同话题、
24 token 回复、4500 槽）、同一台机器、同一份代码除掩码外的一切：

| | 掩码前（`77601fa`） | 掩码后（`fb53514`） |
|---|---:|---:|
| decode tok/s | 3.647 | 3.536 |
| decode hit | 0.8370 | 0.8370 |
| decode MB/token | 2,269.3 | 2,269.3 |
| 每轮 ttft | 14.96 / 8.24 / 6.31 / 6.74 s | 15.04 / 8.24 / 6.31 / 6.74 s |

**没有可测的差别**（3.647 vs 3.536 是 ±3% 的机器抖动；命中率与读字节逐位相同）。
理由就是上面的账：掩码省的是 kernel 的 0.74 ms/层 ≈ 30 ms/token，而端到端一个 token 是
**274 ms**（2,269 MB ÷ ~8.3 GB/s ≈ 273 ms 是 I/O 地板），kernel 只占 **~2%**。
所以：

- 掩码是**真的**（microbench 里 1.7×，40 层 30 ms/token），但在一个 I/O 受限的 decode 里
  它被 273 ms 的读淹没了；
- **要让它变成墙钟收益，得先让 MB/token 降下来**（容量 / 命中率 / 投机），
  那时 30 ms/token 才会浮出来；
- 这也是为什么本轮所有"算得快一点"的改动都该排在"少读一点"后面。

### 并集 dispatch：MoE 终于在 M 上次线性（2026-09-18，Track F1）

上面那句"要让它摊薄，只能让一次 dispatch 同时吃多列的不同 expert 集……而那条路需要
'每 token 一个物理槽行'的两级索引，目前的 `Ids[slot]` 单级索引表达不了"**是错的**，
而且错在一个具体的地方：**`MoeRunner` 的槽轴本来就不是 7**。`MoeDims::slots` 决定
`ids`/`slot_list`/`route_weights`/`h` 的大小，两个 shader 都把它当 push constant
（`num_slots`）读，而 `route_weights` 已经是稠密的 `[m][slots]`——**这正是并集要的那张矩阵**。
所以并集只要把槽轴放宽到整批的 expert 并集、没路由到的格子填 0，一次 dispatch 就吃完所有列，
**不需要任何 kernel 改动**（`runtime/moe_bridge.h` 的 `stage_batch_union` / `run_batch_union`，
`docs/p4_dspark_runtime.md` §6.1）。

`ctest -R bench.mgt1_moe_union_m_curve`（层 0、全部驻留、40 次迭代；⚠️ 这一轮机器上另有四个
worktree 在编译/跑测，绝对毫秒比上面两张表高 1.5–1.8×，**只有同一次运行里背靠背的比值可信**）：

| 列的 expert 集 | M | 并集 | 逐列 ms/层 | 并集 ms/层 | 并集 ms/token | 并集/逐列 |
|---|---:|---:|---:|---:|---:|---:|
| 互不相交（最坏） | 6 | 36/36 | 15.04 | 15.36 | 2.560 | 0.98× |
| 有重叠（真实形状） | 2 | 9/12 | 3.80 | 2.80 | 1.400 | **1.36×** |
| 有重叠 | 4 | 15/24 | 8.36 | 6.07 | 1.517 | **1.38×** |
| 有重叠 | 6 | 21/36 | 19.63 | 9.69 | 1.615 | **2.03×** |

并集**正好只花并集那么多钱**：没有重叠时与逐列打平（没有可省的），有重叠时省下的就是重叠。
数值上，只要并集的槽序与该列自己的顺序一致（gate 的输出顺序是一致的，真实路由落在这一格）
就**逐位等于 M=1**；不一致时差的只是 fp32 槽和的结合律（实测 1.775e-08 of |y|max）。

**但这没有改变 go / no-go**：decode 是 NVMe 受限的（下面那条 274 ms/token 的账），
而字节 ∝ **并集大小**，并集随 M 几乎线性长（每多一列多 ~4 个 expert）。
重算见 `docs/p4_dspark_runtime.md` §6.5：k=1–2 省 ~10% 的 expert 字节，k=5 仍然是亏的。

**（下面是 2026-09-17 的结论，保留原文；"只能……两级索引"那一句已被上面推翻。）**

**结论与 C(M) 相反：MoE 在 M 上不摊薄，是"每 token 一份"的 6.5×。**
原因是几何：`MoeRunner::ids()` 是 `[slots]`——一组专家——所以一批不同路由的 token
只能逐列 dispatch（方案 A），每列的权重读取（6 routed × 18.8 MB + shared）都是完整的
一份。所以：

- **verify 一批的每层代价** = C(M) 链（3.4–4.7 ms）+ MoE(M)（11.7 ms）≈ **15–16 ms/层**
  → 40 层 ≈ **620 ms**；同样的 6 个 token 逐个 M=1 走是 40 × (1.3 + 1.8) = 124 ms。
  **在"每列不同 expert"的假设下，逐列 dispatch 的 verify 比顺序 M=1 慢 5 倍**，
  这是必须在 runtime 之前解决的算术。
- 它同时是一个**权重带宽**问题：M=6 时每层 MoE 要读 36 个 routed expert × 18.8 MB
  ≈ 677 MB（加上 shared 与两段 run 的对齐），40 层 ≈ 27 GB/批。
  对比 docs/p4_hitrate.md §8 实测的 8.3–9.2 GB/s，光权重就是 ~3 s/批。
- 所以 §12.6 的并集数（≈26/层）不是"稍微多一点"，它是**决定性的**：
  MoE 的代价 ≈ 并集大小 × 单 expert 读取，与 M 无关。三条出路，按代价：
  1. **按 expert 分组**：把 6 列按 expert 归并成 ≤7 槽的组（`set_accumulate` 的多组
     dispatch），并集 36 仍需 6 组；但 M=2/3 时并集 12 → 2 组，收益立现；
  2. **kernel 侧真 batch MoE**：让 `ids` 变成 `[m][slots]`，一次 dispatch 吃所有列
     （设计 §10.1.3 缺口 1 的"多组 dispatch"就是这条）；
  3. **缩小 k**：k=5（M=6）在这个算术下不划算，k=1–2 时 verify 的 MoE 只有 2–3 列，
     而 C(M) 那一半仍然便宜（M=2 的每 token 代价 0.6–0.8×）。**在 (1)/(2) 落地前，
     DSpark 的 TPS 投影应该按小 k 算，而不是按 k=5。**

## 5. G2 判据（design §10.1.5 / §10.1.2）

- G2 要求：M>1 的 fp8 投影 kernel（`wq_a` / `wq_b` / `wkv` / `wo_a` / `wo_b`）**逐 M 代价实测**，外加 **M=5 的 `head`**；MoE 的**并集形态**（>7 槽或多组 dispatch）也要逐 M 实测。
- 判定：M=6 的每层代价若落在 **≤ ×3.06（M=5 实测倍率）线性外推**以内 → G2 的"非 MoE 在 M 上接近平坦"的前提成立；否则按场景 B 代入 §10.1.2，DSpark 保持 NO-GO。
- **本次状态：前半段 ✅（§4：最坏 2.98×）；MoE 的 M 曲线仍待测。**

## 6. 阻塞与恢复

**阻塞**：DSH 沙箱在每次进程启动前尝试对 workspace 授予写 ACL 失败（`Win32 5 = ERROR_ACCESS_DENIED`），
因此 `pwsh`（进而 `git` / `cmake` / `ctest` / bench）、`glob`、`grep` 全部不可用；子代理环境同样失败（已实测）。
向 `danger-full-access` 升级被拒：*"requires approval, but no approval channel is available"*。
文件读写工具（read/write/edit）正常，故只能写文档与 marker。

**最小复现**：

```powershell
powershell -NoProfile -Command "git -C C:\Users\Asus\code\deepmoe\build\p4-wt\t status --short"
# 期望：仓库状态；实际：SetNamedSecurityInfoW failed (Win32 5): grantWrite(C:\Users\Asus\code\deepmoe\build\p4-wt\t)
```

**环境恢复后按序执行**：

```powershell
powershell -NoProfile -File C:\Users\Asus\code\deepmoe\build\p4_gpu_lock.ps1 acquire t   # 仅在 ACQUIRED 后继续
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"    # 只读；绝不写 D:\models
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
# 1) 小步提交 dirty 5 文件；2) ctest -R "gpu_layer.mgt1_layer_batch_vs_steps|decode.forty_layers"（4K）
# 3) M=1/2/4/6 每层 bench -> bench/results/mgt1_p4.csv；4) 回填本文第 2–5 节
powershell -NoProfile -File C:\Users\Asus\code\deepmoe\build\p4_gpu_lock.ps1 release t       # 无论成败都 release
```

## 7. 缺口（交给下一阶段）

1. ~~**MoE 并集 / 多 runner**~~ **已关闭（2026-09-18，Track F1）**：`MoeDims::slots` 不是常数，
   `num_slots` 是 push constant，`route_weights` 已经是 `[M][slots]`——所以"> 7 槽"就是把
   `slots` 放宽到整批的 expert 并集，没有 kernel 改动。见上一节与
   `docs/p4_dspark_runtime.md` §6.1（`gpu_moe.the_verify_batch_runs_its_expert_union_once`）。
2. **`head` / `engram` 的 M=5**：DSpark 草稿链需要 M=5 的 head；今天 head 是 M=1 循环五次（≈42 ms vs ≈19 ms）。
3. **DSpark G1**：接受率需在 ≥5 个正常 prompt × ≥64 token、空闲机上重测（先修 dsref 层 1/14 的 98 GB 未触碰占位）。
4. **G3 长上下文的批边界**：M>1 verify 与逐 token decode 的 logits 分歧机制未隔离（§10.2 不变量），runtime 对齐哪一侧未定。
5. **本 track 自身**：dirty 5 文件未提交、`mgt1_p4.csv` 未生成、`decode.forty_layers` 无回归结论（本次环境阻塞导致）。

## 2026-09-16 测试状态与范围调整

- `gpu_layer.mgt1_layer_batch_vs_steps` 已通过：M=2/4/6 batch vs steps，gate ids、
  ring/overflow、compressed rows/index keys/top-k 均一致。
- `compressor.slang` 与 `mgt1_cmp.slang` 都改为 `fp8_decode(fp8_encode_rn(scale))`，
  两者才重新逐位一致（此前 M=1 用非 E4M3 网格、M>1 用另一网格）。
- Track T 收尾 WIP 里顺带把 **M=1** 也换成 Track J 的 K-split/tiled attention；
  该采纳使 `kv_replay.l3_64` 的 (1) 从 8/8 变成 7/8，已回退（`decode_layer.{h,cpp}`
  回到 b4f0e83），M>1 交付不受影响。
- 尚未测：M=1/2/4/6 每层性能曲线 `bench/results/mgt1_p4.csv`、DSpark G1/G3。
