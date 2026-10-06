# 动态 mask 的命中率与质量方案（交 Codex 依次执行）

> 执行记录：先读 [codex_todo.md](codex_todo.md) 的当前进度和
> [miss_mask.md](miss_mask.md#quality-recovery-phase-a-and-repetition-gates-2026-10-05)
> 的实测结果。下文保留制定方案时的假设；Phase A 已证明主要变化是 P2
> 绕过 P0 后的排队和缓存追赶，不是 SSD 设备读取突然慢六倍。恢复历史 NLL
> 并不代表全部质量门通过。未执行、止损和按规则跳过的事项须分开记录。

> 2026-10-06 owner 决定优先于下文的旧电源和默认值规则：所有后续作业与网页使用
> performance，80/72°C 温控暂停后继续并记录。owner 接受 A 的 MMLU46/57 和续轮重复
> .118 作为网页基线，默认动态 mask + 投机；D 完成前用 k2，ONECB 开，GPU route
> 由分开测量决定。C 仍须通过原质量/速度门，GO 后也只作为显式选项。
> 详见 [codex_todo.md §0](codex_todo.md#0-owner-决定2026-10-06优先于下文任何旧表述)。

背景来源：`docs/miss_mask.md`、`docs/mask_freeze.md`、`docs/dspark_topk.md` §17–19、
`docs/STATUS.md` §3 第 21/23/35/38–40/47/51 行。

---

## 0. 现状与目标

| 项 | 历史动态 mask（2026-10-04） | 当前动态 mask（2026-10-05） |
|---|---|---|
| l3 64 步，静态 heat 起步，5100 槽 | NLL **.835581**，top-1 50/64，命中 .814，丢失 mass .1694 | NLL **1.360084**，top-1 40/64，命中 **.6513**，丢失 mass **.3310** |
| P0 异步加载平均延迟 | 约 141–204 ms（`miss_mask.md` MMLU 段） | **1233.72 ms** |
| 第一个 token 的命中 | — | 51/240 |
| 8 轮对话，k5 | — | 9.52 tok/s，verify **379 ms/cycle** |

结论：**当前版本的动态 mask 比历史版本明显变差，原因还没有查明。** 质量差主要来自
miss 读盘慢了约 6 倍：miss 的 expert 迟迟不进 cache，命中率掉下来，丢失的权重翻倍。
这比任何新策略都重要，必须先查清楚。

目标（按顺序）：
1. 找回历史质量：l3 mask 回到命中 ≥ .80、NLL ≤ .86。
2. 补上能抓到"重复、乱码"的质量门。
3. 在此基础上，用"按权重决定是否等待 miss"来调节质量和速度。
4. 重新选投机的 k 和网页默认配置。

---

## 1. 通用规则

沿用 `docs/dspark_e2e_plan.md` §1–2 的全部规则（独立工作树、一次一个 GPU 任务、
power-saver + AC + 0 热暂停、同 session 对照、按 ms/token 报告、新开关先默认关闭、
预测收益减半、每项记录、NO-GO 写进 STATUS §7）。另外：

- 本方案主要改的是**质量**，所以每个阶段都要报质量门（§2），不能只报速度。
- 网页正在运行。跑 GPU 任务前先停网页，跑完再恢复（`tools/web/RUNNING.txt`）。
- 离线工具（`cache_sim`、`mask_freeze_sim.py` 等）不占 GPU，可以随时跑。

---

## 2. 质量门（本方案每个阶段都要跑）

| 门 | 命令 / 数据 | 要求 |
|---|---|---|
| off 基线 | `tools/l3_ppl.py --modes off` | NLL **.622784** 逐位一致 |
| mask NLL | `tools/l3_ppl.py --modes mask`，5100 槽，静态 heat 起步 | 记录 NLL、top-1、命中、丢失 mass；Phase A 之后不得比 Phase A 的结果变差 |
| 中文 turn64 | `bench/results/draft_attribution/turn64.json`，T=0 和 T=1 各一次 | 无循环（见下） |
| 长生成 | `bench/web_longtest.py` 的三组 512 token（中文思考、续轮、英文） | 无循环；重复指标不高于 plain off 的 1.5 倍 |
| MMLU57 | `tools/mmlu_bench.py`（现有协议） | ≥ 48/57 |
| decode | `suite.decode`、`suite.decode_longctx` | 与当前基线相同（6/8 + 7/8，4K/16K 8/8） |

**新增重复指标**（Phase B 实现，之后所有长输出都要报）：
- 最长同 token 连串长度；
- 最近 128 token 内是否存在精确短周期（周期 ≤ 8）；
- 重复 4-gram 占比（`dspark_topk.md` §17 已经在用）；
- distinct-2。

"无循环"的定义：同 token 连串 ≤ 3，且没有周期 ≤ 8 的精确短周期。

---

## Phase A：查清动态 mask 的退化（最优先）

**目标**：找出命中从 .814 掉到 .651、P0 延迟从约 0.2 s 涨到 1.23 s 的原因，并修好。

1. **先复现历史格**：用 `bench/results/miss_mask/gates/mask.txt` 记录的配置（单盘、5100 槽、
   静态 heat、无 DSpark），在当前 HEAD 上跑 l3 mask。每次约 1 分钟。
2. **逐个开关拆分**，每次只改一项，一项一格：
   - 单盘 vs 双盘；
   - `DEEPMOE_IO_ENGRAM_DEADLINE` 0 / 1（P2 绕过 P0、in-flight 上限 96）；
   - `DEEPMOE_BATCH_ENGRAM_EARLY` 0 / 1；
   - DSpark 关闭（不占 384 个 MTP pin）；
   - 实际槽数：确认 ready 里真的是 5100，1M 上下文改动（`3ba80ab`）有没有挤占内存或 KV。
3. **拆不出来就二分**：在历史 exe（`miss_mask.md` 记录的 commit）和 HEAD 之间做
   `git bisect`，判据是 l3 mask 的命中率 ≥ .78。
4. **给 P0 加分段计时**：排队等待、设备读取、落地拷贝，按读源分开统计；同时记录
   P0 和 P2/P3 的在途数量。要回答"1.2 s 花在哪"。
5. 修复后重跑 §2 全部质量门，并跑一次 8 轮 `long_turns.json`（双盘、动态 mask、
   不开投机），作为后续阶段的新基线。

**验收**：l3 mask 命中 ≥ .80、NLL ≤ .86；P0 平均延迟回到 ≤ 300 ms；off NLL 不变。

**止损**：如果拆分和二分都找不到单一原因，把各格数据写清楚，交 owner 决定是否回退到
历史版本的 IO 配置。不要在退化没查清的情况下进入 Phase C。

---

## Phase B：补质量门（可与 Phase A 并行，纯 CPU）

1. 在 `tools/` 下新增重复指标脚本，读 `events.jsonl` 的 token id 计算 §2 的四项指标。
   离线跑一遍已有结果做标定：
   - `spec_e2e/fixed_final/fixed_mask`（"霓"循环）必须判为失败；
   - `spec_e2e/final_dual/mask`、`final64/mask`（正常成文）必须判为通过；
   - `web_spec5_long`（`the *the *` 周期 2）必须判为失败。
2. 把它接进 `hitrate_bench.py` 和 `web_longtest.py` 的报告输出。
3. 加一个 CPU 测试，用合成序列覆盖边界。

**验收**：上面三组已知结果判定正确；CPU 和工具门禁通过。

---

## Phase C：按权重决定是否等待 miss（"部分 miss 等待"）

**思路**：现在的 mask 是所有 miss 一律不等、权重归零。改成按 gate 权重分开处理：
权重大的 miss 等它读完（精确计算），权重小的照旧 mask。用一个参数在"全 mask"和
"全等待（off）"之间调节。

**和已关闭方案的区别**（不要做成它们）：
- §3 第 39 行 `stall1` 按**个数**降级（只降级一个 expert），质量 ×1.09，速度 ×1.07，NO-GO。
- §3 第 38/40 行只路由到常驻 expert（resident-only all / verify），NO-GO。
- 本方案按**每层丢失的权重**决定，有每个 token 的等待预算，miss 的读盘和 LRU 不变。

**规则**（每层、每个 token）：
1. 算出全部 mask 时这一层会丢失的权重比例 `lost`。
2. 如果 `lost > τ`，就按权重从大到小挑 miss，等待它们的 P0（用现有的 wait_layer / join），
   直到 `lost ≤ τ`。
3. 如果这一层所有 routed expert 都 miss（只剩 shared），至少等待权重最大的那一个。
4. 每个 token 有总等待预算：最多等 `W` 个 expert，或者累计最多等 `T_ms` 毫秒。
   预算用完后，剩下的层全部按 mask 处理。
5. τ = 1 时等价于现在的 mask，τ = 0 且预算无限时等价于 off。

**步骤**：
1. **先离线扫描**：用 `traces/mixed` 里带 gate 权重的路由（27,399 token），在 `cache_sim`
   的 LRU 上模拟 τ ∈ {0.30, 0.20, 0.15, 0.10, 0.05}，统计每个 token 平均要等几个 expert、
   丢失的平均和最差权重。按 Phase A 修好后的 P0 延迟估算每个 token 的等待时间。
2. 从离线结果里挑 2～3 个 τ（等待代价最小、丢失权重明显下降的点），在 GPU 上每个 τ
   跑一次：l3 mask NLL + 中文 turn64 + 8 轮 `long_turns.json`（双盘，不开投机），
   同一 session 里带上 off 和全 mask 两个对照。
3. 投机模式下，verify 是一次算多行。等待会让整个 batch 停下，所以先只在不开投机的
   decode 上做。投机路径的处理放到 Phase D 之后另议。
4. 新开关 `DEEPMOE_MASK_WAIT_TAU`（默认关闭，等价于现在的 mask）和
   `DEEPMOE_MASK_WAIT_BUDGET`，启动时打印生效值。

**验收**（全部满足才能考虑设为默认）：
- l3 mask NLL / off NLL ≤ 1.10（历史 mask 是 1.34 倍，stall1 是 1.09 倍）；
- §2 全部质量门通过，长生成无循环；
- 8 轮对话 decode 速度比 off 快 ≥ 20%（历史 mask 是 +43%）。

**止损**：如果质量达到 ≤ 1.10 倍时，速度相对 off 的提升已经低于 10%（减半后约 5%），
记 NO-GO，写进 STATUS §7。

---

## Phase D：动态 mask 下重新选投机配置

当前动态 mask + k5 的 8 轮是 9.52 tok/s，verify 379 ms/cycle。k5 的 verify 是 6 行，
动态 LRU 下 union 很大，代价很高。

1. 用 Phase A 修好后的版本，在同一 session 里跑 8 轮 `long_turns.json`（双盘）：
   不开投机、k=2、k=3、k=5。如果 Phase C 通过了，再加一组"最好的 τ + 不开投机"。
2. 按 ms/token 报告，同时报告 cycle 分解、接受率、命中率和 §2 的重复指标。
3. 记录 384 个 MTP pin 对主模型命中率的影响（投机格和不开投机格的命中率之差）。
4. 选 ms/token 最低、且质量门全部通过的那一组作为网页默认值。如果不开投机最快，
   网页就关掉投机。

**注意**：
- ONECB 和 GPU 路由在双盘下没有降低 cycle 成本（`dspark_topk.md` §15），这一阶段要在
  开和关两种情况下各测一次 k=2，确认它们在动态 mask 下是否值得保留。
- 每种配置只跑一次；接受率波动大，按相同接受率换算 cycle 成本后再比较。

---

## Phase E：腾出更多 cache 槽位（最后做，可选）

STATUS 第 1481 行的结论是：decode 侧剩下的杠杆只有容量和磁盘。

1. 做一次内存账：121 GB 中 expert cache、pinned dense/attention 权重、MTP pin、transit、
   各种 scratch、KV、Engram 表、host 端副本、OS 各占多少。
2. 列出 decode 期间可以释放或缩小的部分（比如 decode 时不用的 transit、重复的 host 拷贝）。
3. 用 `cache_sim` 在真实路由上量出"每加 100 槽能提高多少命中"，按每 GB 约 53 槽换算收益。
4. 只有能腾出 ≥ 150 槽（约 3 GB）时才动手，并且要证明不会让 prefill 或长上下文出问题。

---

## 3. 不做的方向（已经关闭，不要重开）

| 方向 | 结论来源 |
|---|---|
| 任何预测预取（lookahead、Markov、hidden-state 等） | STATUS §3 第 23、35、51 行 |
| 非 LRU 淘汰策略（score-aware、ARC、LFU 等） | §3 第 47 行 |
| 饱和 cache 上的 reheat | §3 第 21 行 |
| resident-only 作为默认、verify-only resident-only | §3 第 38–40 行 |
| 固定初始 cache | `dspark_topk.md` §15、§17：只服务 37.5%，输出循环 |
| 自动冻结 | `mask_freeze.md`：离线判定冻结占比太低 |
| 用 CPU 算 miss 的 expert | miss 的 expert 在 SSD 上，不在内存里；瓶颈是读盘，不是计算 |
| 只读一部分的 expert 先算 | 缺行的 expert 输出是错的，不比 mask 好 |
| miss 权重重新归一化、替补 expert | 会改变路由数学；要做需另开方案并先过质量门 |

---

## 4. 决策点

- **Phase A 之后**：如果找回了历史质量，Phase A 的版本成为新基线，继续 Phase C。
  如果没找回，停下来交 owner 决定，不进入 Phase C。
- **Phase C 之后**：通过就设为显式可选模式，并在网页上提供选项；是否设为默认交 owner 决定。
  不通过就记 NO-GO。
- **Phase D 之后**：按测得的最快且质量合格的配置更新网页默认值。
- 任何阶段：减半后预计收益 < 3% 就跳过。

---

## 5. 每个阶段的交付物

1. 分支上的小 commit，一个 commit 一件事，新功能都在开关后面。
2. 报告写进 `docs/miss_mask.md` 新的一节（Phase A/C/E）或 `docs/dspark_topk.md`（Phase D），
   包括：改了什么、前后对比表、§2 质量门结果、热数据、命令行和 env。
3. 原始结果放 `bench/results/mask_quality/<phase>/`。
4. GO 的结论更新 STATUS 主表；NO-GO 写进 STATUS §7。
