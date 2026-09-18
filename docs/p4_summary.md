# P4 单 PR 总览：R1 / R2 / S / T

本 PR 把 P4 四条 track 的全部工作合成一个集成分支 `p4/one-pr`，合并顺序 R2 → R1 → S → T。
**重要：本 PR 尚未编译、未跑测试、未重跑 benchmark。** 合并工作是在一个嵌套沙箱里完成的，
该环境无法创建 `cmake`/编译器子进程；所有数字都是合并前既有 Claude 会话的实测/记录，
状态逐条列在下面。请把本 PR 当 **draft / 待复验**，并按 §4 的清单跑完再合入 main。

## 1. 四条 track 的交付

| Track | 主要交付 | 报告 | 状态 |
|---|---|---|---|
| R2 | per-source KV 平面、engram 常量 C++ 推导、session rollback/park/restore、serve 中断、kv_replay 测试 | `docs/p4_kv_ux.md` | 部分完成；restore 超时未修，SSD KV 未实现 |
| R1 | prefill→decode expert 交接、MOE_OVERLAP、P3 backfill、淘汰守卫、路由 dump、hitrate bench/sim、结果 | `docs/p4_hitrate.md` | 实现已提交；A/B 未全部跑完 |
| S | coopmat band attention、dense coopmat、prefill kernel/bench/test 接口 | `docs/p4_prefill_speed.md` | 代码/正确性对齐完成；4K 安静机数据缺失 |
| T | M>1 decode kernels、MgtRunner、batch 路径、gate softplus 修复、gpu_layer mgt1 测试 | `docs/p4_mgt1.md` | 正确性已验证；C(M) 缩放曲线缺失 |

合并时的文件级重叠只有两处，均已用 3-way 自动合并，无冲突标记：

- `runtime/engine.{h,cpp}`（R1 × R2）
- `gpu/vulkan/prefill_kernels.{h,cpp}`（R1 × S）

## 2. 既有实测数字（合并前，未在本 PR 重跑）

- decode 热步：**81.5 ms**（64 token）、91.1 ms（4K）、94.2 ms（17K），41 submit；
  地板 75.8 ms，有效带宽 160 GB/s = 217 GB/s 的 74%。
- 对话：**3.4–4.5 tok/s**，命中率 0.84–0.90，每 token ≈90–100 ms 计算 + **124–193 ms NVMe stall**。
- prefill：64 / 4,133 / 17,010 token TTFT **43 s / ≈150 s / ≈445 s**，
  compute ≈ **24 ms/prompt token**；4K baseline 分项：attention 27.2 s、
  expert I/O 43.1 s（198 GB @ 4.6 GB/s）、expert GPU 16.7 s、mHC 4.9 s、host readback ≈3.4 s。
- DSpark：接受 2.6–3.3 tokens / verify，预测 ×1.18–1.32，条件 GO；
  **M>1 的 C(M)（尤其 M=3/6）是决定性未知数**。
- T 的 mgt1 4K 正确性：M=2/4 各 stage cos ≈0.99999，gate 集合 2/2、4/4 相同，
  compressed rows / index keys 逐位相同，top-k 列表完全一致。
- S 的 coop vs legacy 正确性：stage cos ≈0.99999，gate top-6 64/64 相同。

## 3. 构建/测试状态（2026-09-16 更新）

- 本 PR 已在 Strix Halo（CMake 4.3 + Ninja + zig/Vulkan SDK 1.4.357）上完成编译。
- 已通过：CPU 单元 **21/21**、`kvdisk.roundtrip`、`suite.tokenizer`、
  `suite.engram_tables`、`gpu_layer.mgt1_layer_batch_vs_steps`、
  `gpu_prefill.stages`（110 checks）、`kv_replay.l3_64`。
- 完整命令与结果见 **`docs/p4_test_report.md`**。
- 本轮修复：`compressor.slang` / `mgt1_cmp.slang` 的 E4M3 scale 与字节不一致
  （raw rows 9 → 0，M>1 batch vs steps 重新逐位一致）；回退 Track T 的 M=1
  K-split 采纳（`kv_replay` (1) 恢复 8/8）；确认 restore timeout 是四线并发导致。
- R2 SSD TTFT 已实测：同一 4,133-token prompt，冷 **101.6 s** → 新进程 SSD 命中
  **2.47 s**（reused 4132 / prefill 1），详见 `docs/p4_kv_ux.md` §8 与
  `docs/p4_test_report.md` §5。
- S 的 N=4133 已实测：legacy 103.67 s / coop 99.89 s，只快 **3.6%**，5× 目标未达成；
  瓶颈与下一步见 `docs/p4_prefill_speed.md`。
- R1 `--auto-tune` 回路已端到端验证（round 1 读 round 0 的 heat 文件），但短工作负载下
  hit 0.8278 → 0.8275（无提升）；A/B 与更长 warm 曲线仍待补。
- 仍待补：T 的 C(M) 曲线、DSpark G1/G3、`kv_replay.longctx`。

## 4. 合入前必须完成的复验

```powershell
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
$env:DEEPMOE_LONGCTX_DIR="C:\Users\Asus\code\deepmoe\traces\longctx"
ctest --test-dir build --output-on-failure
```

按顺序补：

1. **R2**：`kv_replay.l3_64` 全绿（先把 restore 超时修掉）；SSD prefix cache 落地并量 TTFT。
2. **R1**：安静机串行跑 MOE_OVERLAP 0/1、PREFILL_HANDOFF 0/1、BACKFILL 0/1；
   hit 目标 ≥0.92、stall ≤100 ms/token；把结果写回 `docs/p4_hitrate.md`。
3. **T**：跑 `gpu_layer.mgt1_layer_batch_vs_steps` 与 `decode.forty_layers`；
   产出 `bench/results/mgt1_p4.csv`（M=1/2/4/6），确认 `decode.forty_layers` 仍是已知的 7/8 near-tie。
4. **S**：安静机跑 N=4133 legacy vs coop，写 `bench/results/prefill_p4.csv`；
   若 20 s 目标达不到，给出 expert I/O 43 s 地板与 host readback 的修正目标。
5. **DSpark**：T 的 C(M) 出来后再做 G1（GPU runtime、5 prompt）与 G3（4K/17K 批边界）。

## 5. 本环境的限制（为什么这里是 draft）

合并所在的会话沙箱禁止嵌套进程创建：dsh 子 agent 的 shell 报 `spawn EPERM`，
我们自己的 shell 里 `cmake` 配置以 `0xC0000005` 崩溃，`zig cc` 同样被挡；只有 git 和文件读写可用。
因此无法在本 PR 内编译或跑测试。代码/文档/合并已完成；构建与测量需要在普通终端
（或 CI / Claude Code 限额恢复后）执行。

## 6. 容量与专家命中模式（2026-09-16）

- `serve --cache-slots N` 已实现。本机安全上限实测：**5500 槽（96.34 GiB）**；
  5600 槽在 path B 第 20 个 slab 失败（98.10 GiB）。
- 同一 8-turn 脚本：4500 槽模拟 hit 0.9234、旧 run ~4.5–5.2 tok/s；
  **5500 槽实测 hit 0.9431、6.05 tok/s、stall 46–71 ms**。
- 命中模式（`docs/p4_expert_patterns.md`）：decode 的 86.6% expert 在 prompt
  prefill 里已出现；换话题时 hit 从 0.94–0.96 掉到 0.88–0.91；全局 static heat
  很平（top1024 只覆盖 40.8%），时间复用是 128 步尺度（63.7%），不是相邻 token。
- KV：17,010 位置当前引擎实测 **54.96 MiB**；0.89 GB 是 R2 之前旧 40 层
  KvStore 的数字（见 `docs/p4_kv_ux.md` §9）。

## 7. 2026-09-17：每轮 reheat、一个 heat bug、DSpark runtime 方案

三条，细节分别在各自的文档里：

1. **每轮 reheat（`docs/p4_hitrate.md` §7）** — turn 边界衰减热度并重跑 P3 backfill。
   接口：`Engine::reheat` / `serve --reheat [--reheat-decay F]` / `{"op":"reheat"}` /
   `SessionOptions.reheat`。实测（2200 槽、同话题三轮、on vs off）：**decode hit
   0.7210 = 0.7210，没有提升**——因为 2200 槽在 turn 1 之后就饱和（2132 resident），
   同话题的专家本来就在 cache 里，reheat 只能把冷尾换成同样用不到的专家。
   结论：reheat 在饱和 cache 上是 no-op；要验它得上"8-turn 脚本 + 4500/5500 槽"那套
   （§6 的 0.9234 → 0.9431 就是容量给的），这轮时间预算不够跑两个配置。
2. **一个真 bug：decode 路径的 expert heat 恒等于 0** — `Engine::run_layer` 从来没有把
   gate 的 top-16（`GatePush::record = 16`，id 和 raw score 都已在 host-coherent buffer 里）
   传给 `RouteDecision.near_ids / near_scores`，而 `Planner::plan_layer` 的 heat EWMA
   （design §9.3）正是定义在它们上面的。已修；修完 500 槽里 90–121 个 expert 在"这一轮
   还热"的一侧。影响：任何 score-aware 策略此前都在对全 0 排序；
   `DEEPMOE_HEAT_FILE`（离线 route dump 那条路）不受影响。
3. **DSpark runtime 方案与缺口（`docs/p4_dspark_runtime.md`）** — 方案钉死为
   **K = 16 格 → 一次草稿 forward → 一次 verify forward（M = k+1 ≤ 6）→ 最长路径
   （`eal` / `viterbi`）→ 采样接受（温度 1 `accept_sampling_exact`）/ 贪心前缀匹配（温度 0）**。
   逐 dispatch 对过之后，runtime 侧缺的是**三个 kernel 能力**而不是循环本身：
   (a) M=6 的 MoE —— `MoeRunner` 是 7 槽 / 一组专家，6 个 token 各有各的 top-6；
   (b) 草稿链的 bf16 输入 GEMV；(c) `accept_sampling_exact` 要的四个读回。
   树方案**不改变** GPU 的账（规格 §3.3 / §6.1：与单链同一批 token、同一份字节），
   所以它不能绕过 (a)/(b)，只是让同一份 M=6 前向更值钱。
   §4 列了三件不依赖 MoE 改动、现在就能测的事（C(M) 曲线 / expert 并集 / 批边界 G3），
   它们是 20 t/s 判定的前置输入。

## 8. 优化项的现状与账（2026-09-17 实测，`bench/config_sweep.py`）

同一套 4 轮同话题对话、同一份二进制、同一台空闲机，只改 cache 容量与 reheat 开关；
`bench/results/config_sweep.json`。按"拿到了多少"排：

| 优化项 | 状态 | 实测 |
|---|---|---|
| **expert cache 容量**（`--cache-slots`） | 已用，仍在加 | 1000 → 2200 → 4500 槽：**1.83 → 2.59 → 3.65 tok/s**，hit 0.5912 → 0.7431 → 0.8370，**hit 每翻倍容量 +0.093**；外推 5500 ≈ 0.90，与 8-turn 实测 0.9431 同量级 |
| **SSD KV 前缀复用**（R2） | 已用，需 `--kv-dir` | 4,133-token prompt **101.6 s → 2.47 s（41×）** |
| prefill→decode expert 交接（R1） | 已用（`handoff_` 默认开） | 未单独 A/B |
| MoE overlap（R1） | 已用（默认开，`DEEPMOE_MOE_OVERLAP=0` 关） | 未单独 A/B（P4-T 清单第 2 项） |
| **每轮 reheat**（R1 round 2） | 已实现、可用 | **4 轮短对话上收益 0**：2200 槽 hit 0.7431 = 0.7431；4500 槽 0.8370 → 0.8386，但 tok/s −0.018、**MB/tok +32**（它自己填的 68 × 18.8 MB）。别在短对话上开 |
| 顺序学习（`--auto-tune` / `DEEPMOE_HEAT_FILE`） | 已实现 | 无提升（0.8278 → 0.8275）；本轮修好引擎内 heat 之后才具备前提 |
| 热步 fence_wait / mHC 提交（Track J 的 K-split/tiled 接口） | **未做** | 热步 81.5 ms，地板 75.8 ms → 余量只有 **~7%** |
| prefill chunked / coopmat（Track L / S） | S 已合入，默认 legacy | N=4133 legacy 103.67 s vs coop 99.89 s，**只快 3.6%**（5× 目标未达成）；`--gpu-prefill-min` 默认 0 = 不启用 |
| **投机解码（DSpark）** | **未做** | `Engine::generate` 的 `speculative` 仍是 `unimplemented`；缺三个 kernel 能力（见 `docs/p4_dspark_runtime.md` §2） |

**最重要的一条性能事实**：decode 被 NVMe 带宽钉住。四种容量的有效读带宽都是
**8.3–9.2 GB/s**，于是 `tok/s ≈ NVMe_eff / (MB/tok)`：命中率 0.59 → 0.84 让
MB/tok 从 4,998 降到 2,269，tok/s 就跟着翻倍。**这段的性能完全由"每个 token 要读多少字节"决定**
——容量、顺序（heat）、投机解码（同一批字节换多个 token）是同一件事的三种做法。
另外 prefill 是 **≈24 ms/prompt token**（4,133 token ≈ 100 s），在对话里比 decode 还贵，
而它已经有 41× 的复用手段（SSD KV）没接进 `serve` 的默认路径。
