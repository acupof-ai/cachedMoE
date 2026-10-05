# Mask 冻结标定与动态 LRU 恢复（2026-10-05）

按本次提出的规则，**自动冻结判为离线 NO-GO，mask 默认恢复动态 LRU**。
此判定适用于下列回放和这组门槛。
冻结前必须同时满足权重门槛、至少 32 个预热 token、最近 16 个 token
淘汰少于 1% 槽位。现有 8 轮动态对话的 demand-LRU 回放没有满足淘汰门槛的点。
因此停止自动冻结这条实施分支，没有将预热／冻结控制器、P3 补热或按层解冻接入运行时。
固定初始集合保留为 `--mask-cache fixed` 显式实验。

## 数据与边界

`tools/mask_freeze_sim.py` 复用 `tools/cache_sim.py` 的 LRU。
原始输出位于 `bench/results/adaptive_mask/offline.json`；输入 SHA-256 和限制
写在输出中。它检查完整记录、专家编号、turn 对齐及 40 层的权重／位置对齐。

原生 `route.bin` 记录专家 id 和命中个数，**没有 gate 权重或输出 token id**。
不能从命中个数推算 `1 - mass_lost`，所以原生回放仅测试必要的淘汰门槛，
权重占比字段为 null。GPU prefill 不写这类记录；decode prefill 的 40 条记录
根据 `turns.json` 剔出 decode 统计。模拟从静态 heat 开始，miss 立即完成，
不模拟异步 IO、P3、GPU prefill 对缓存的修改及冻结后的输出反馈。
以下数字是回放结果，不是运行时实测的冻结比例。

| 动态输入 | 可用主模型槽 | decode token | 通过淘汰门槛的点 | 16-token 淘汰数：最小／中位／最大 | 门槛 |
|---|---:|---:|---:|---:|---:|
| `miss_mask/speed_mask`，双盘 8 轮 | 5500 | 2318 | 0 | 69 / 199 / 773 | <55 |
| 同一输入，扣除 MTP 的槽数近似 | 5116 | 2318 | 0 | 78 / 224.5 / 811 | <51.16 |
| `spec_e2e/final_dual/mask`，中文 64 输出 | 5500 | 63 | 0 | 272 / 395.5 / 493 | <55 |
| 同一输入，扣除 MTP 的槽数近似 | 5116 | 63 | 0 | 282 / 406 / 507 | <51.16 |

5116 是 5500 减去草稿常驻 384 个专家的容量近似，未模拟具体 pinned key。
这些必要条件与 .93/.88、.95/.90、.97/.92 的权重阈值无关，放宽其中的
权重门槛不能改变这组回放的 0 个候选点。也不能把历史的“18 tok/s、只算
37.5% 专家”作为本方案的收益。

这个淘汰门槛比 .95 权重门槛更苛刻：稳态满 cache、每个 miss 都需淘汰一槽
且没有 joined fill 时，16 token 有 3840 次专家请求，允许淘汰少于 55 次
相当于**命中率高于约 98.57%**。少量免费槽、异步 joined fill、prefill 和
prefetch 会改变此近似，不能用它推断真实 eviction 计数，但它说明限制
来自低 churn 条件，而不只是进入／离开的 mass 数值。

## 权重扫描

补充使用已有 `traces/mixed/route_layer00..39.parquet`：40 个 prompt，27,399
个有真实 gate 权重的 teacher-forced token，包含中／英文和代码。
这是旧参考模型的 prompt 路由，**不是这次 Linux 生成的输出**，用于交叉检查，
不能代替生成质量或真实吞吐。每个新 prompt 重置控制器，LRU 保留跨 prompt 的集合。

控制器用每层保留 gate mass 的 EWMA（α=1/8），取其均值和最差层；
预热≥32、冻结≥16、平均单 token 硬下限 .80；重复 4 次或同一 3-gram
在最近 32 token 出现 3 次立即解冻。重复仅在有输出 token id 时可评估，
本次输入没有 id，因此这项只通过合成边界测试，没有用于真实数据统计。
预热期信号是 **P0 换入之前的 resident 权重占比**；若用 P0 等待后的已计算
权重占比，其值会恒为 1，无法判断当前集合能否冻结。

| 主模型槽 | 进入／离开 | 冻结 token 比例 | 全程 pre-admission 权重占比 | 冻结期权重占比 | 阈值切换次数 |
|---|---|---:|---:|---:|---:|
| 5500 | .93 / .88 | 2.748% | .929024 | .962675 | 6 |
| 5500 | .95 / .90 | 2.748% | .929024 | .962675 | 6 |
| 5500 | .97 / .92 | 2.697% | .929128 | .966129 | 7 |
| 5116 | .93 / .88 | .788% | .921274 | .969292 | 7 |
| 5116 | .95 / .90 | .788% | .921274 | .969292 | 7 |
| 5116 | .97 / .92 | .774% | .921277 | .971266 | 5 |

切换次数不包括新 prompt 的强制重置。冻结不 touch、不换入，解冻后的下一
token 恢复 LRU；本模拟未实现 P3 补热。所有 routed 原始权重按其总量
归一后统计保留比例，不改推理中的 routed scaling factor。

假设各 token 成本相同，甚至把冻结 token 的**全部成本降为零**，2.748%
占比也只给出 `1/(1-.02748)-1 = 2.826%` 的模型化加速上限；MTP 容量近似
的上限为 .795%。实际冻结仍要做 attention、head、hit expert 计算。
按项目规则把预测收益减半后，分别只有约 1.41% 和 .40%。这不构成超出
3% 抖动门槛的收益证据；它不是本机测出的速度上限。

## 决策与验证范围

按用户的“冻结占比太低就放弃”规则，在离线阶段停止自动冻结。
不为没有通过用途检查的三组阈值跑重复 GPU 质量／速度格。
因此也没有自动冻结与动态 mask 的同-session GPU 对照，不能宣称该方案
达到 MMLU≥48/57、接近 .835581 NLL 或快≥3%。

保留的运行时变更只有恢复默认动态 LRU，以及显式 `--mask-cache dynamic|fixed`
选择（run、serve、网页均支持）。动态 mask 沿用原机制：正常 LRU/P0 请求，
当前前向只计算 planner hit、miss 权重置零，不等待 miss。GPU prefill 仍精确。
`DEEPMOE_MASK_DYNAMIC_LRU=0` 兼容旧固定配置，`1` 为动态；显式 CLI 优先。
普通路由默认仍 off；GPU 路由和 DSpark 仍是显式实验。

离线控制器 11 项 CPU 测试覆盖停留、硬下限、回差、滚动淘汰、逐层 EWMA、
重复、预热时阻止重复输出进入冻结和坏 trace；运行时测试覆盖默认动态、legacy 显式固定、CLI 覆盖及
离开 mask 后关闭固定 cache。CPU ctest 25/25，综合 gates 32/32。
动态 mask 的本机输出、NLL、MMLU 与 8 轮速度验证另记在本文件后续结果段。

## 本机动态 mask 验证

原始日志与传感器记录在 `bench/results/adaptive_mask/`，GPU 任务按
`jobs.json` 串行执行。AC 接通、power-saver、双盘 48/48；所有数字分开记录
配置，未重跑完整 A/B。

| 64-step l3，静态 heat 开始，5100 槽 | NLL | top-1 | 命中比例 | 平均丢失 gate mass |
|---|---:|---:|---:|---:|
| 本次 off | .622784 | 56/64 | 不用于 mask 比较 | 0 |
| 本次动态 mask，双盘、Engram deadline on | 1.360084 | 40/64 | .6513 | .3310 |
| 历史动态 mask，单盘（`miss_mask/gates/mask.txt`） | .835581 | 50/64 | .814 | .1694 |

off 与 Linux 平台基线 bit-for-bit 一致。**当前 mask 未达到接近历史
.835581 的质量目标。** 历史与本次同为 5100 槽，但运行配置／加载节奏不同，
不能以换了默认值为由把历史质量贴到当前版本上。本次从静态 heat 起步的
首 token 只命中 51/240；P0 异步加载的平均 request 延迟为 1233.72 ms。
这与质量下降相伴，尚未通过受控拆分证明是哪个加载／调度改动造成的。
13.848 tok/s 是该 teacher-forced 数列的速度，不是正常对话质量下的收益。

`suite.decode` / `suite.decode_longctx` 的现有 ctest 两套均通过。
严格 L3 teacher-forced 是 6/8，自有 prefill 7/8，与先前基线相同；这不满足
AGENTS 的严格 8/8+8/8，不能报作该严格门槛通过。4K、16K teacher-forced
各为 8/8。masked/shared-only GPU 检查与 k5 committed-prefix 回滚检查各 1/1。
回滚检查使用 2000 槽，记录到 reserve 拒绝，因此不能对所有测试
宣称“加载失败为零”；性能／质量实际运行的失败数另从 status 报告提取。

中文 turn64：5500 槽、动态 LRU、无投机，同一夜跑 prompt / seed 41001。
63 个 decode step 为 5169.07191 ms，**82.049 ms/token / 12.1878745 tok/s**；
命中 .890542、丢失 mass .0882。64 个输出 token 中最大同-token 连串长度为 1、
最大 3-gram 计数为 1，没有短周期循环。P0 reserve/submit/IO 与 failed fill
均为 0；cache_fixed/cache_frozen 均 false。短测不代表长对话已通过。

MMLU57，动态 LRU / k5 / top-K4 / ONECB / GPU snapshot route：**48/57
（84.21%），2 个格式无效答案按错误计**。65 个投机 cycle，197/317 草稿接受，
2 个读源、5500 槽。协议是每科 1 题的零样本生成 `Answer: X`，最多 16 token，
exact prefix + 最后 prompt token 单步；不是完整标准 5-shot MMLU。
cache_fixed/cache_frozen false；P0 reserve/submit/IO 和 failed fill 为 0。
本轮达到 ≥48/57 的样本门槛，但冷启动 l3 NLL 仍未达到 .835581。

57 次精确 prefix prefill 使本轮超出最初的 900 秒监督预算。
同一引擎继续完成，模型设置未变；临时 continuation guard 接管 80/72°C
温控，然后恢复原 supervisor。原监督日志和 continuation 的传感器日志
共同保存，均为 0 次热暂停。复跑计划预算改为 1300 秒，未重复 GPU 格。

双盘 `long_turns.json` 的 8 轮动态 mask / k5 在一个 session 引擎中完成。
5500 总槽、384 个 MTP pinned expert；ONECB=1、GPU snapshot route=1、
draft profiling=0、max_context=4096。它是本次动态基线，没有自动冻结对照格。

| 本次 8 轮 | 结果 |
|---|---:|
| 输出／decode step | 2361 / 2353 |
| decode 总时长／ms per token | 247116.431 ms / **105.022 ms/token** |
| decode tok/s | **9.521827** |
| prefill 总时长 | 47381.379 ms |
| 周期／target submit | **594 / 594** |
| 平均输出 per cycle | 3.961279 |
| 草稿接受 | 1764 / 2953，59.736% |
| 平均 draft / verify / commit / CPU per cycle | **33.199 / 379.451 / 1.834 / .197 ms** |
| P0 reserve / submit / IO、failed fill | 0 / 0 / 0、0 |
| 热暂停／AC 改变／镜像健康 | 0 / 0 / 48 of 48 |
| 8 轮末 128 token 的精确短周期循环 | 0 轮 |

草稿处于先前约 31 ms 的量级；本格主要成本是 verify。按相同接受率，
即使把草稿的 33.199 ms 完全删掉，其余 381.482 ms/周期也只允许约
10.38 tok/s。这个账不能支持“只优化草稿便获得 1.6×”。profiling off 时
attention、MoE、tail 分桶没有采集，不能把零值当零耗时，未根据本格猜算子原因。
每轮速度 7.716～11.545 tok/s；没有计算与历史 18 tok/s 或不同长度短测的加速比。

重复检查只排除持续的精确短周期，不能证明语义质量。代码轮和杭州行程轮
均到 450-token 上限；原文、token id、逐轮统计在 raw 中。输入并不是 1M，
也没有为 mask 做新的 4K/17K 输入质量验收。固定 cache 与自动冻结均不默认启用。

网页已恢复为动态 mask，双盘、5500 总槽、k5/top-K4、1M 上限与 80/72°C
温控保留。`/api/config` 和 `/api/status` 均返回 200，确认一个主路径、
`cache_fixed=false`、`cache_frozen=false`。浏览器没有刷新，聊天记录保留。
启动后配置／状态快照在 raw 的 `web_config.json`、`web_status.json`；
当前启动与停止方式见 `tools/web/RUNNING.txt`。这不是新的网页速度测量。

## 复跑

```bash
.venv/bin/python tools/mask_freeze_sim.py \
  --native-run bench/results/miss_mask/speed_mask \
  --native-run bench/results/spec_e2e/final_dual/mask \
  --weighted-trace traces/mixed \
  --out bench/results/adaptive_mask/offline.json

python3 tools/web/server.py --resident-only mask --mask-cache dynamic \
  --cache-slots 5500 --max-context 1048576 \
  --mirror /mnt/deepmoe2/models/DeepSeek-V4.1-Flash --no-kv-disk
```

GPU 任务需先停止网页，由同一温控 supervisor 串行运行，80°C 暂停、72°C
恢复；速度格出现任何热暂停即无效。离线工具不占 GPU，可在网页运行时执行。
