# 动态 mask 的异步等待检查（2026-10-05）

专家 miss 已经异步换入。动态 LRU 恢复后，当前前向仅计算 hit、miss 权重为零，
不调用 `Planner::wait_layer`。GPU snapshot verify 在唯一 target fence 后提交
LRU/P0 请求，后续 cycle 可以使用完成的填充；并不等待该批专家读完。
普通 mask 也跳过 wait_layer，命中槽的 guard 保护到 GPU 使用完为止。

此前 planner 的 `stalls` 在 plan_layer 发现 miss 时递增，没有测量等待。
`45ee3ab` 改为分别报告 miss layers、实际 expert wait calls、miss joins 和
wait_layer 的耗时；超时也计入耗时。DSpark diagnostics 新增每次 target 的
`expert_io_wait`。它是计数修复，不是算子优化；LRU、异步 P0、采样、数值和
一个主路径／一次 target 前向的规则保持原样。原 `nvme_stall_ms` / `gate_ms`
仍包含规划／记账成本，不可当作纯读盘等待。

## 同 prompt 的短 trace

主模型为 native FP4/FP8、双盘 48/48、5500 槽、power-saver、AC 在线。
中文夜跑 prompt / seed41001，每格生成 32 token，31 个 decode step。
每配置仅一次运行，target GPU trace 开；k5 有 host diagnostics，draft GPU
profiling 关闭。采样输出／异步命中时机不同，不能当逐 token 数值对拍，也
不能将此短测的速度推广到 8 轮或不同上下文。0 读源丢失、0 热暂停、0 AC 改变。

| 项目 | 普通动态 mask | 动态 k5 / top-K4 / ONECB / GPU route |
|---|---:|---:|
| decode tok/s | 12.595869 | 13.398560 |
| decode ms/token | 79.391 | 74.635 |
| target 周期 | 31 | 9 |
| draft 接受 | — | 22/41 |
| emitted / cycle | 1 | 3.444444 |
| target attention busy ms/cycle | 31.495 | 86.515 |
| target MoE busy ms/cycle | 31.147 | 106.808 |
| target Engram GPU busy ms/cycle | 1.793 | 9.833 |
| target CED / tail busy ms/cycle | 1.653 / 7.528 | 2.450 / 11.364 |
| 跨 submit gap ms/cycle | 3.309 | 0 |
| 同 submit gap ms/cycle | 1.757 | 1.927 |
| target GPU span ms/cycle | 78.682 | 218.896 |

k5 的平均 draft / verify / commit / CPU 是 25.782 / 228.715 / 1.134 /
.160 ms/cycle。唯一 target submit 的 fence_wait 219.072 ms 与 GPU span
218.896 ms 对齐，等待的是 GPU 执行完成。fence 后 finish_routes 为 1.312 ms，
其中 planner/LRU/P0 issue .863 ms；它包含请求提交，没有专家读完的 join。
输入与 Engram issue .600 ms，record_layers_tail 7.065 ms，setup_snapshot
.403 ms；Engram issue+land 5.267 ms 是重叠分项，已经包含在准备／录制中，
不能再次相加。这份 trace 排除了“verify 主要在等专家 IO”的解释。
七个满 k5 周期的平均 verify 为 240.804 ms；末尾两个周期 k=4/2。

短测 k5 比普通 mask 快约 6.37%，只作本配置的小样本结果。主模型多行
验证仍有 106.808 ms 的 MoE 和 86.515 ms 的 attention，异步 IO 无法消除
这些计算。没有把 GPU fence_wait 算成 NVMe 等待。

## 为什么 8 轮的 9.52 不等于短测的 13.40

已有 8 轮记录为 105.022 ms/token、9.521827 tok/s，594 周期，draft
33.199 ms、verify 379.451 ms/cycle。每周期实际计算的 routed union 从
本次短测的 546.667 增到 740.562 个 layer/expert 项；上下文和路由也不同。
每周期新提交 miss 字节反而从约 2.497 GB 降到 1.489 GB。8 轮 Engram
实际 land 等待总计 3929.596 ms，即使全部记入 decode 也仅占其 1.59%。
因此不能把 P0 请求的队列延迟直接加到 token 耗时上。长测没有 per-op trace，
不能拿这份短测反推它的 attention/MoE 精确分桶。

之前固定 cache 的 18 tok/s 只计算约 37.5% routed 请求，并出现循环；动态
8 轮 snapshot 服务约 92.07%。两者的算量和质量不同，不能将全部速度差
归到 LRU 或同步加载，也没有在本次声称恢复 18 tok/s 或达到 1.6 倍。

## 验证与收据

CPU gates 32/32；新增测试将一个 outstanding FetchGroup 保持 pending，
要求 wait_layer 超时被计入实际等待，再验证 900 个 miss layer 不会产生
任何虚假的 wait call。原始配置、exe / shader SHA、trace、逐周期和温控
记录均在 `bench/results/mask_async/`；机器摘要为
`docs/mask_async_receipt.json`。短 trace 仅用于本次等待归因，质量边界仍见
[动态 mask 验收](mask_freeze.md)，当前冷 l3 NLL 未达历史 mask 目标。

新计数在同配置的 16-token 检查中，4 个 target cycle 的
`expert_io_wait.calls / miss_joins / ms` 全部为 **0 / 0 / 0**。
状态仍记录启动阶段 MTP exact 加载的 3 次 wait，逐 cycle 的差量排除了
启动／prefill。这一格只验计数与异步路径，不计为速度优化结果。
