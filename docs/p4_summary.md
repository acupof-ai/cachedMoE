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

## 3. 已知问题 / 不要当成已完成

1. `kv_replay.l3_64` 第 3 段 restore 失败：`timeline wait for 5579 timed out`（详见 `docs/p4_kv_ux.md`）。
2. 本 PR 没有编译产物，也没有测试结论；见 §4 复验清单。
3. R1 后段的 stall/chat3 A/B 在 4 条线并发时 device lost / 超时，数据无效；需安静机串行重跑。
4. KV 落 SSD prefix cache 与 R1 `--auto-tune` 已于 2026-09-16 追加（见 p4_kv_ux.md / p4_hitrate.md），但本环境无法编译验证。
5. S 的 N=4133 端到端提速、T 的 M=1..6 缩放、prefill_p4.csv 都还没产出。

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
