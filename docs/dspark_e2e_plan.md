# DSpark 投机解码端到端优化方案（交 Codex 依次执行）

基线来源：`docs/dspark_topk.md` §13、`bench/results/draft_attribution/`、`bench/results/engram_deadline/`。
最新 64-token 测试（power-saver、单盘、AC、0 次热暂停）：spec k=2 **10.72 tok/s**，plain mask **11.88 tok/s**。投机路径仍然更慢。

---

## 0. 先算账：目标与止损线

当前 k=2 每个 cycle（power-saver，T=1 采样，接受 35/55）：

| 项 | ms |
|---|---|
| draft | 38.8 |
| verify | 163.6（GPU 忙：attention 57.9 + MoE 66.1 + tail 11.4 + engram 5.3 + CED 1.9；空隙 20.1） |
| commit | 3.65 |
| 合计 | 207.8 ms / 2.25 token = **92.3 ms/token** |
| plain mask | **84.2 ms/token** |

- **盈亏平衡点**：cycle ≤ 2.25 × 84.2 ≈ **189 ms**，至少还要省 19 ms。
- **乐观上限**（各项都做到位）：draft −17、空隙 −15、attention −13、MoE −8，cycle 约 155 ms，即 68.8 ms/token，约 1.22×。按规则收益减半后约 **1.04×**。
- **硬件下限参考**（8060S，40 CU，约 216 GB/s）：
  - draft 每次约 2.8 GB，下限约 13 ms（实测 serial 26–39 ms）。
  - verify M=3 下限约 77 ms（实测约 161 ms）。
  - MoE 在 3 个位置上的 union 平均 12.2 个 expert，基本没有摊薄效果，所以 traffic 理论上限约 1.47×。
- **结论**：空间很窄。必须按 §8 的决策点止损，不能无限投入。

---

## 1. 通用规则（每个阶段都适用）

1. **工作树**：用 `../deepmoe-spec`，分支 `spec-e2e`，独立 build 目录。主 checkout 上有别的 agent 在改，不要在主树上动。合并后按 AGENTS.md：merge → verify → push → 删除工作树和分支。
2. **GPU 一次只跑一个任务**：跑之前停掉 web UI 的引擎（见 `tools/web/RUNNING.txt`），跑完再恢复。
3. **测量协议固定**：
   - 迭代用 `bench/results/draft_attribution/turn16.json`，阶段收尾用 `turn64.json`。
   - 命令模板：`tools/hitrate_bench.py --max-turns 1 --cache-slots 5500 --require-sources 1 --serve-arg=--resident-only --serve-arg=mask [--serve-arg=--dspark --serve-arg=--spec-k --serve-arg=2 --serve-arg=--spec-top-k --serve-arg=4]`。
   - 环境：power-saver、接 AC、0 次热暂停。出现 pause 的结果作废。可复用 `draft_attribution/run_checks.py` 的热记录逻辑。
   - 迭代时用单盘；阶段最终结果补跑双盘（`--mirror /mnt/deepmoe2/models/DeepSeek-V4.1-Flash`，这是默认报告口径）。
   - **同一个 session 内同时跑 plain mask 基线**。不同电源模式、不同 session 的数字不能互相比较。
   - 每种配置只跑一次，保留最好结果并记录，不做重复 A/B。接近 ±3% 抖动的差异，以机器记录的 per-op 数字为准。
4. **报告指标**：以 **ms/output token** 为主（不能只报 tok/s），另外报：
   - tokens/cycle、接受率、命中率；
   - cycle 分解（draft / verify / commit / other）；
   - verify trace（各类 busy + gap）；
   - draft 每个 stage 的 GPU 时间。
5. **对比口径**：两组输出的 token 序列不同，结论只能是"同一 workload 的对比"，不能当逐 token 对比。
6. **新开关必须证明真的生效**（吸取 deadline 开关那次的教训）：启动时打印一次生效配置，`provenance.json` 里也要能看到对应 env。
7. **预测收益一律减半**后再决定是否继续。
8. **每项实验都要记录**：结果写进 `docs/dspark_topk.md` 新的一节，NO-GO 写进 `docs/STATUS.md` §7。commit 里不加 attribution 行。
9. 新功能先用开关控制、默认关闭；通过 §2 的全部质量门、并且 turn64 确认有收益之后才改成默认开启。

---

## 2. 质量门（凡是动数值的改动都要过）

- `tools/l3_ppl.py` mode `off`：NLL **0.622784**，逐位一致；top-1 56/64。
- Batch gate：63 个位置 cos ≥ 0.88（当前 0.9466，不能明显下降）。
- 生成式 MMLU 57 题：≥ 48/57。
- `suite.decode` 8/8 + longctx 8/8。
- CPU gate：`ctest -LE "needs-model|needs-gpu"` 25/25；`tests/run_all.py` 30/30。
- 投机路径专项：
  - wrap rollback 正确；
  - k=0 时只调用一次 target；
  - draft 与 golden chain 逐位一致（near-tie 规则沿用现有测试）；
  - ring / carry / hidden 回滚后，与非投机路径状态一致。

---

## Phase 0：补测量（不改性能）

**目标**：后面每一阶段都能按 stage 归因。

1. 在 draft 链（`runtime/dspark_runtime.cpp`，非 mega 路径）的每个 op 前后写 GPU timestamp。按 stage 汇总：
   - 3 个 MTP stage × {q/kv 投影 `wq_a`/`wq_b`、attn core、o 投影、MoE、mHC}；
   - head（bf16，1.32 GB）、Markov、readout。
2. 每个 stage 同时记录 **host 墙钟**，用来算出 GPU 时间与墙钟之差，也就是 submit/fence 往返的开销。
3. 结果写入 `DEEPMOE_SPEC_DIAGNOSTICS` 的 cycle 行；`tools/spec_diagnostics.py` 汇总输出 draft stage 表和 verify busy/gap 表。
4. 在同一 session 内跑一组 turn16 基线（mask、spec k=2、spec k=3），作为后续所有阶段的对照，并写进 `dspark_topk.md`。

**验收**：所有 stage 的 GPU 时间加总，与 draft 墙钟的差距可以解释（即 dispatch 往返开销）。打开 timestamp 后性能变化 < 1%。

---

## Phase 1：draft 合成一个 command buffer（预计 −8～−12 ms，减半后 −5）

**现状**：非 mega 路径每个 op 都走一次 `dispatch_now`，约 87 次 submit + fence 往返。MoE 走 host 的 `run_batch_union`，还要做 WC readback 和舍入。mega kernel（VGPR 120、12 waves、93 stage）测得 30.19 ms，比 serial 的 26.06 ms 还慢，**不要再走 mega 方向**。

**做法**：
1. 保持每个 kernel 各自独立的 pipeline（保住各自的 occupancy），把整条 draft 链录进**一个** command buffer，op 之间用 `vkCmdPipelineBarrier`（compute→compute，只对相关 buffer 做 memory barrier）。只在最后提交一次、等一次 fence。
2. draft 的 expert 全部常驻（每个 MTP stage 128 个 × 3 个 stage = `draft_pins 384`），所以 draft MoE 的路由可以**全在 GPU 上做**：gate → top-k → 用常驻 slot 表间接索引 → MoE kernel。不需要 host 读回。
3. 原来 host 端 WC readback 时做的舍入（包括 act_quant），要在 GPU 上**逐位复刻**。先写一个 CPU vs GPU 的逐位对拍测试。
4. 用开关 `DEEPMOE_DSPARK_ONECB=1` 控制，默认关闭。

**验收**：
- draft 输出与 golden chain 逐位一致；接受率不变（同 prompt、同 seed）。
- draft 墙钟 ≤ 各 stage GPU 时间之和 + 1 ms。

**注意事项**：
- draft attention 是**双向的，没有 causal mask**（见 `gpu/shaders/dspark_attn.slang`）。3 个 MTP stage 必须保持 M=5，不能缩成 k 列。`TRIM_TAIL` 只能裁 head 和 Markov。
- barrier 不要用全局的 `ALL_COMMANDS`→`ALL_COMMANDS`。先用正确的粗粒度 barrier 保证结果对，再按 profile 收紧。

---

## Phase 2：draft 的 tile 修正（预计 −5～−8 ms，减半后 −3）

根据 Phase 0 的 stage 表，按耗时从大到小处理：

1. **`wq_a`**：现在只有 20 个 workgroup，带宽利用率约 8%。改成 split-K，让 workgroup 数量 ≥ 2×CU（≥ 80），再加一个小的归约 pass，或者用 subgroup 原子以外的确定性归约。**归约顺序必须固定**，保证逐位可复现。
2. **`wq_b`**：K 很短，但每行分配的 lane 太多。减少每行 lane 数，提高每个 workgroup 处理的行数。
3. **head**（bf16，1.32 GB，带宽利用率 64%）：调 tile 和每 lane 的加载宽度（128-bit load），目标 ≥ 85%。
4. **mHC**：把相邻的小 kernel 合并，减少 barrier 数量。
5. 每个 kernel 用 RADV shader stats 检查 VGPR 和 waves，并记录。

**验收**：每个 kernel 单独做 micro-bench，带宽利用率有提升；draft 输出逐位一致；draft 总时间目标 ≤ 22 ms。

**需要 owner 决定，Codex 不要自己动手**：head 的 vocab 子集或 FP8 head，会违反"不重新量化"的硬规则。只做估算，写进文档，等 owner 决定。

---

## Phase 3：verify 去掉每层 host 往返（最大的一项，预计空隙 −15 ms，减半后 −8）

**现状**（`runtime/engine.cpp` 中的 `run_layer_batch`）：每层 attention 之后都要：
`cmd_flush` → 读回 gate → CPU 上做 `union_experts` → `plan_layer` / guard → 读回 x + act_quant（`stage_batch_union`）→ 录制 union MoE。
这已经从 128.5 ms 降到 20.1 ms/cycle，但仍有 40 次 submit+fence。k=3 时空隙还有 60.6 ms，其中 engram wait 42.7 ms。

**做法**（必须与已关闭的 host-flag / persistent dispatch 机制不同。那个是 STATUS §7 0g/5 的 NO-GO，**不要重开**）：
1. batch 开始时，host 上传一张**常驻快照表**：`expert → slot`，不在常驻里的写 -1。
2. 在 GPU 上逐层做：gate → top-k → 查快照表 → miss 置 0（保持 mask 语义）→ 压缩生成 union 列表 → 写 indirect dispatch 参数 → act_quant → union MoE（`vkCmdDispatchIndirect`）。
3. 整个 batch（40 层 + tail + GPU rank readout）录进一个 command buffer，只提交一次。
4. batch 结束后，host 读回各层的路由结果，再**异步补做** LRU touch、miss 的 P0/P2 提交和 planner 统计，保证 planner 看到的访问序列与原来一致。
5. 用开关 `DEEPMOE_BATCH_GPU_ROUTE=1` 控制，默认关闭。

**必须先写清楚并验证的设计点**：
- **Slot 安全**：batch 运行期间，快照表里的任何 slot 都不能被驱逐或复用。最简单的做法是在 batch 期间冻结驱逐（P0/P2 只写空闲 slot），在 fence 之后再放开。hit-guard 必须在 store 的锁内设置（这个 race 之前修过，不能退回）。
- **Publish 可见性**：P0 加载完成的 slot，只有在 fence 之后的下一个 batch 才能出现在快照表里。
- **语义差异**：原来按层实时查看常驻状态，batch 中途才到的 expert 也能命中；快照方式会稍微降低命中率。要测量命中率差异，并重新过 §2 的质量门（cos ≥ 0.88，MMLU ≥ 48/57）。
- 共享 expert、miss-mask 语义、正常的 LRU / P0 都**保持不变**。
- k=3 的 engram wait（42.7 ms）是 IO 问题，不是 dispatch 问题。单独列为子任务：查清楚 P2 ≤ 8 KiB 绕过 P0 和 in-flight cap 96 是否也覆盖 batch 的 engram 行，再决定是否处理。

**验收**：verify 空隙 ≤ 5 ms/cycle；质量门全部通过；同一 session 的 turn64 结果中 ms/token 下降。

**顺带评估**：plain decode 每 token 也有约 16 ms 的 gate 往返。同样的机制也能降低 decode 基线。要单独报告这一项。这会**抬高投机路径的门槛**：如果 decode 降到约 68 ms/token，盈亏平衡点就变成约 153 ms/cycle。但这对整个项目是净收益，应该独立推进。

---

## Phase 4：verify attention 的投影（预计 −10～−15 ms，减半后 −6）

**现状**：attention busy 57.9 ms，旧归因中 Q/out 投影占 45.3 ms 里的 29.8 ms，score/PV 只占 6.5 ms。投影受权重带宽限制：M=3 时权重读一遍就够，理论上应该接近 M=1 的耗时。

**做法**：
1. 对比 batch 路径和 decode 路径的投影 kernel。如果 batch 路径用的是旧 kernel，就把 decode 已经优化过的 GEMV 移植过来，扩展成 M 列（权重加载一次，在寄存器里累加 M 列）。
2. 检查每个 kernel 的 VGPR / waves，确认加列之后不会掉 occupancy。M 最大取 k+1 = 4。
3. CM attention（`DEEPMOE_MGT_ATTN_CM`）因为 63 个位置 cos 只有 0.874 < 0.88，保持关闭。除非先查明是哪一步精度出问题（比如 fp16 累加）并修好，否则**不要重测**。
4. scale-fold（`DEEPMOE_MGT_FOLD_SCALE`）没有稳定收益，保持关闭，不重开。

**验收**：attention busy 降到 ≤ 45 ms/cycle；batch gate cos 不变；NLL 逐位一致。

---

## Phase 5：union MoE kernel（预计 −5～−8 ms，减半后 −3；优先级最低）

**现状**：MoE busy 66.1 ms。`moe_gateup.slang` 用标量 FMA `acc[RowsPerLane][M]`，带宽利用率约 70%。union 平均 12.2 个 expert，字节量约是 decode 的 2 倍，这部分是结构性的，省不掉。

**做法**：
1. 先用 profile 判断是 ALU bound 还是带宽 bound。只有在 dequant 或指令数是瓶颈时才做 WMMA。
2. WMMA 方案：把 M 补齐到 16 列，参考 prefill 的 coopmat FP4 路径；针对约 20 个 slot 的 union 做参数扫描。
3. 之前"按 M 移植 champion kernel"的方案是 NO-GO（STATUS §3 64），**不要重做**。

**验收**：MoE busy 下降 ≥ 5 ms，且 NLL 逐位一致。达不到就记为 NO-GO 并停止。

---

## Phase 6：k 与接受率调优

1. 用 `DEEPMOE_SPEC_DIAGNOSTICS` 的 `confidence` 和 `target_rank` 数据，离线拟合 `--spec-confidence-min` 的阈值，以及动态 k（包括 k=0）的策略，目标是最小化期望 ms/token。
2. 拟合时要用**优化后**的 cycle 成本模型：draft(k)、verify(M=k+1)、gap。每加一行 verify 的成本要用 Phase 3/4 之后重新测的值（原来约 45–60 ms/行）。
3. 在 engram wait 解决之前，不要默认开 k=3。
4. 分别报告贪心解码和 T=1 采样下的接受率；最终口径按产品默认的采样设置。
5. 取消 MTP expert 的 pin 只能提高 0.22 个百分点命中率，不做。除非 address table 改成动态的，否则保留 384 个 pin。

---

## 7. 已关闭的方向（不要重开）

| 方向 | 结论来源 |
|---|---|
| DSpark mega kernel（30.19 vs 26.06 ms） | `dspark_topk.md` |
| host-flag / persistent dispatch | STATUS §7 0g/5 |
| union kernel 按 M 移植 champion | STATUS §3 64 |
| CM attention（cos 0.874） | `dspark_topk.md`，修好精度前不重测 |
| scale-fold | 没有稳定收益 |
| 取消 MTP pin | +0.22 pt，收益太小 |
| draft / target 重叠执行 | 被 root token 依赖卡住 |
| 把 draft attention 缩成 k 列 | attention 是双向的，会改变数值 |

---

## 8. 决策点（止损）

- **Phase 1 + 2 + 3 完成后**：在同一 session 用 turn64、双盘口径测量。
  - cycle ≤ 175 ms，且 ms/token 比 plain mask 低 ≥ 5%：继续 Phase 4/5。
  - cycle 在 175～189 ms 之间：只做 Phase 4，然后再评估。
  - cycle > 189 ms：投机路径在本机记为 NO-GO，写进 STATUS §7。Phase 3 的 GPU 路由机制移植到 decode 单独推进。
- **任何一个阶段**：减半后的预计收益 < 2 ms/cycle 就跳过，不做。
- 最后的 1.04× 是乐观估计。如果 decode 基线也被 Phase 3 的机制降低，就要用新基线重新判断投机路径是否还值得保留。

---

## 9. 每个阶段的交付物

1. 分支上的小 commit（一个 commit 一件事），新功能都放在开关后面。
2. `dspark_topk.md` 新的一节，包括：改了什么、per-stage 前后对比表、质量门结果、热数据、命令行和 env。
3. 结果目录 `bench/results/spec_e2e/<phase>/`，含 `provenance.json`、cycles jsonl、thermal jsonl。
4. 结论是 GO 时：改成默认开启，更新 STATUS 的主表；结论是 NO-GO 时：写进 STATUS §7，代码留在开关后面或者回退。

## 10. 执行收据（2026-10-05）

实现和阶段判断见 `dspark_topk.md` §14–15，原始结果在
`bench/results/spec_e2e/`。用户本次明确要求停止网页，因此本次不恢复 web。
追加要求“mask 不做 LRU，只用初始满槽专家”替代 Phase3 原文的动态缓存语义；
历史动态格保留作对照，不能替代固定 cache 的质量验收。

| 阶段 | 执行情况 |
|---|---|
| 0 测量 | 分离 GPU timestamp 与 host wall；序列化调度和 ONECB 的统计均可读。校准变化 −3.33%，未证明 <1%，性能格关闭 draft profiling。 |
| 1 草稿 ONECB | 完成，独立 pipeline + compute barrier + GPU 常驻路由/量化；golden 全链、输出前缀和 KV wrap 逐位一致；k=2 从 81 次提交降到 1 次。 |
| 2 草稿 tile/head | 小算子不扩大；按追加要求实际尝试逐对 weight decode，head 8.403→8.319 ms，低于止损线，保持关闭。 |
| 3 验证 GPU 路由 | 完成，锁内原子快照/guard，immutable 参数、indirect union、GPU act_quant；40 层和 tail 一次提交；旧动态格在 fence 后补 LRU/P0；按追加要求，默认 mask 填满后冻结，禁止淘汰/新增换入。Engram 在提交前到位。 |
| 4 attention 投影 | 实际实现/测试 pair-dot，同输出/同接受率/同 union；cycle 121.731→119.806 ms，减半仅 .963 ms，NO-GO，开关0。未达到 ≤45 ms，未重开 CM/fold。 |
| 5 union WMMA | 本次不进入：Phase 1–3 周期约 180 ms，属于方案的 175–189 ms 区间，只检查 Phase 4。固定121ms格伴随低专家服务率，不能用它重设该决策条件。 |
| 6 动态 k | 完成成本拟合工具与实际 k=0 单前向测试。旧动态样本最佳阈值仅估计 +0.47%；固定轨迹最佳仍 k=2、估计额外收益0。固定 k=2，保留 384 pin。 |

同窗口单盘、power-saver、AC、无热暂停的 turn64：普通 mask
**12.33 tok/s**，serial k=2 **12.43**，ONECB k=2 **11.90**，
ONECB + GPU snapshot route **13.49**。最后一项相对普通 mask 吞吐约
**+9.4%**，ms/output 从 **81.10 降到 74.12**。文本与路由不同，这是任务对照。

镜像随后恢复挂载，双盘三格已完成。用户追加固定初始 cache、host 残差追踪和实际 Phase 4/head 优化；最新收据和启用判断以报告 §15 为准。

最终固定5500槽、双盘生成式 MMLU **48/57**（1 invalid，60周期、115/120草稿接受），满足此项门槛。off64 **.622784/56**，batch63 **.9537048/55**；普通短oracle严格8/8未复现，故新GPU路由仍默认0。精确普通4K/17K TF/free8/8不代替固定mask质量；独立端到端检查见报告§15。

固定mask+ONECB/GPU route的独立4K/17K端到端检索也已完成：各9/9输出前缀与参考一致、各5/5草稿接受；4900槽、双盘、0淘汰/加载失败/温控暂停。它是两次检索，不证明任意长对话无损。

最终双盘5500槽、profiling-off速度：固定mask **18.091544**、固定投机 **18.492909 tok/s**（+2.22%，在抖动内），投机cycle **121.668256 ms**，draft **20.817178**、verify **98.694461**。28周期35/54接受，2.25output/cycle，与profiling对照全部tokens/ranks/routes一致。主目录52个shader hash一致、CPU25/工具30/streams拒绝通过；精简收据 `dspark_e2e_receipt.json`，完整日志留主目录 `bench/results/spec_e2e/`。网页停止、performance已恢复。
