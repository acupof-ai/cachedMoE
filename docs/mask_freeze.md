# Mask 冻结标定与动态 LRU 恢复（2026-10-05）

按本次提出的规则，**自动冻结判为离线 NO-GO，mask 默认恢复动态 LRU**。
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

| 动态输入 | 可用主模型槽 | decode token | 可冻结候选点 | 16-token 淘汰数：最小／中位／最大 | 门槛 |
|---|---:|---:|---:|---:|---:|
| `miss_mask/speed_mask`，双盘 8 轮 | 5500 | 2318 | 0 | 69 / 199 / 773 | <55 |
| 同一输入，扣除 MTP 的槽数近似 | 5116 | 2318 | 0 | 78 / 224.5 / 811 | <51.16 |
| `spec_e2e/final_dual/mask`，中文 64 输出 | 5500 | 63 | 0 | 272 / 395.5 / 493 | <55 |
| 同一输入，扣除 MTP 的槽数近似 | 5116 | 63 | 0 | 282 / 406 / 507 | <51.16 |

5116 是 5500 减去草稿常驻 384 个专家的容量近似，未模拟具体 pinned key。
这些必要条件与 .93/.88、.95/.90、.97/.92 的权重阈值无关，放宽其中的
权重门槛不能改变这组回放的 0 个候选点。也不能把历史的“18 tok/s、只算
37.5% 专家”作为本方案的收益。

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

离线控制器 10 项 CPU 测试覆盖停留、硬下限、回差、滚动淘汰、逐层 EWMA、
重复和坏 trace；运行时测试覆盖默认动态、legacy 显式固定、CLI 覆盖及
离开 mask 后关闭固定 cache。CPU ctest 25/25，综合 gates 32/32。
动态 mask 的本机输出、NLL、MMLU 与 8 轮速度验证另记在本文件后续结果段。

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
