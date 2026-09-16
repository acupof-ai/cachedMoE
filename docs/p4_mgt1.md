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

## 4. M 缩放表（本次未测，全部待填）

| M | 每层 ms | 相对 M=1 | vs M=5 线性外推 | 备注 |
|---:|---:|---:|---|---|
| 1 | TBD(未测) | 1.00 | — | |
| 2 | TBD(未测) | TBD | — | |
| 4 | TBD(未测) | TBD | — | |
| 6 | TBD(未测) | TBD | 判据 ≤ 3.06 | G2 判定格 |

数据落盘位置：`bench/results/mgt1_p4.csv`（**本次未生成**）。

## 5. G2 判据（design §10.1.5 / §10.1.2）

- G2 要求：M>1 的 fp8 投影 kernel（`wq_a` / `wq_b` / `wkv` / `wo_a` / `wo_b`）**逐 M 代价实测**，外加 **M=5 的 `head`**；MoE 的**并集形态**（>7 槽或多组 dispatch）也要逐 M 实测。
- 判定：M=6 的每层代价若落在 **≤ ×3.06（M=5 实测倍率）线性外推**以内 → G2 的"非 MoE 在 M 上接近平坦"的前提成立；否则按场景 B 代入 §10.1.2，DSpark 保持 NO-GO。
- 本次状态：**未测**（需要 M 扫描 bench，见第 6 节）。

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

1. **MoE 并集 / 多 runner**：今天的 `MoeRunner` 是 7 槽、`route_weights` 是稠密 `[M][slots]`；M>1 verify 需要 >7 槽或分组 dispatch（design §10.1.3 缺口 1）。
2. **`head` / `engram` 的 M=5**：DSpark 草稿链需要 M=5 的 head；今天 head 是 M=1 循环五次（≈42 ms vs ≈19 ms）。
3. **DSpark G1**：接受率需在 ≥5 个正常 prompt × ≥64 token、空闲机上重测（先修 dsref 层 1/14 的 98 GB 未触碰占位）。
4. **G3 长上下文的批边界**：M>1 verify 与逐 token decode 的 logits 分歧机制未隔离（§10.2 不变量），runtime 对齐哪一侧未定。
5. **本 track 自身**：dirty 5 文件未提交、`mgt1_p4.csv` 未生成、`decode.forty_layers` 无回归结论（本次环境阻塞导致）。
