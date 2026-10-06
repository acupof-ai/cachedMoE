# DSpark main-path verification with top-K acceptance (2026-10-04)

The checkpoint's three MTP stages now run in `cachedmoe serve`, enabled by
`--dspark`. The requested rule is **keep the draft's original token when it is
in the target row's top-K**; default K=4. This is approximate acceptance and
does not preserve greedy output or the target sampling distribution.

```bash
build/cachedmoe serve --model "$HOME/models/DeepSeek-V4.1-Flash" \
  --resident-only mask --dspark --spec-k 2 --spec-top-k 4
```

`--spec-k` is the maximum draft prefix length (1..5). `--spec-top-k` controls
acceptance rank, independently of draft length. Without `--dspark`, normal
single-position generation remains the default. Only one GPU stream is supported.

## One matrix, one path

Each cycle evaluates the five bidirectional draft positions in the real MTP
chain. Its output head/Markov suffix now produces only the needed k decisions
by default. It calls `Engine::forward_batch` **once** on `[root, draft[:k]]`.
The resulting `[k+1,129280]` target matrix supplies every acceptance decision,
the first rejection's correction, or the all-accepted bonus token.

Acceptance stops at the first row whose original draft token is outside top-K.
That row uses the requested greedy/nucleus sampler, and no later rows are
accepted. Accepted tokens stay unchanged, so every retained row was evaluated
under the prefix actually emitted. EOS/stop and length boundaries can truncate
the accepted prefix. The target's window KV, compressed-state carry, history,
CED and draft hidden window retain only the committed input prefix. The final
emitted token is still uncomputed, matching the ordinary session convention.

Top-K creates **one actual speculative path**. A depth-d top-K matrix has up to
K^d mathematical token combinations, but its later rows are conditioned on
the original path. Replacing an earlier token invalidates those later rows.
No tree nodes or alternative paths are forwarded or treated as verified.

## Cache and model data

Normal main-model demand planning/LRU/P0 loading also runs for the verify batch.
With `--resident-only mask`, misses have zero compute weight and are not waited
for; hits keep their original weights and the shared expert always runs.
The draft's 384 MTP experts are loaded and pinned in the existing expert store;
they consume slots inside the configured budget. At 5,500 total slots this
leaves 5,116 for the main model. This is a measured part of DSpark's cost.
Checkpoint and mirror files remain read-only.

The draft gathers BF16 means of main blocks 37/38/39, including exact GPU
prefill's last 128 positions. Only committed positions enter its 128-slot KV.
Draft attention/GEMV use BF16 rounding at reference boundaries; MoE retains
the engine's existing FP16 intermediate activation policy.

## Short speed cells

Sources: `bench/results/dspark_topk/short_{mask_only,k2_top4,k5_top4}/`.
One run per configuration, same 13-token Chinese prompt, 32 greedy output
tokens, empty initial KV/cold process, 5,500 total slots, GPU prefill minimum
16, two live read sources, DPM auto. These are cold short cells and must not
replace or be compared directly with the eight-turn conversation headline.

| Configuration | Decode tok/s | Accepted / verified drafts | Cycles | Emitted per cycle |
|---|---:|---:|---:|---:|
| miss-mask only | 7.1059 | — | — | — |
| DSpark k=2, top-4 | 6.7577 | 20 / 20 | 11 | 2.818 |
| DSpark k=5, top-4 | 5.8543 | 23 / 37 | 8 | 3.875 |

Thus k=2 is -4.90% and k=5 is -17.61% against the paired short baseline.
For k=2, draft/verify/commit wall totals were 1544.54/2998.27/24.76 ms;
for k=5, 1305.47/3945.79/23.23 ms. Initial draft-window seeding is included.
The draft always produces five positions even when only two are verified.
These cells exposed the first implementation's readback bottleneck and are
historical measurements, not the current executable's result.

These first cells predate the final hidden-window rollback/selected-logit
scoring corrections. Their executable hashes are preserved in each
`provenance.json`; the changes affect bookkeeping and exceptional boundaries,
not the recorded one-shot chat's accepted decisions. The final batch report
charges GPU time without individual timestamps to `other`, rather than
interpreting unmeasured per-kernel zeros as zero work.

## Performance corrections and matched conversation

The draft originally converted Q from a write-combining GPU mapping with scalar
CPU loads. Its one golden-chain profile spent **109.21 ms** in that conversion,
plus **11.79 ms** copying logits. Q now lands directly on its BF16 grid on the
GPU, and streaming AVX2 loads copy the output into cached CPU RAM. The same
profile went from **153.08 to 31.75 ms**, with identical draft decisions; logits
readback fell to **0.67 ms**. Source: `draft_profile_comparison.json`.
The corrected cold k=2 cell reached **7.7846 tok/s**, +9.55% over the paired
short miss-mask baseline; its 20/20 accepted draft decisions were unchanged.

Three turns of `bench/results/hitrate/long_turns.json` then reached **11.6640
tok/s**, versus the previous mask-only first three turns' **13.2110**. This
comparison changes the emitted text and expert routes; it is a practical
throughput comparison, not an identical-work kernel comparison. Three-position
target batches averaged 175.72 ms in the trace: attention 45.30 ms, MoE union
75.01 ms, Engram 4.59 ms, CED 1.48 ms, tail 9.79 ms, and 39.55 ms of gaps.
Draft/verify/commit wall totals were 7.828/38.843/0.633 seconds for 220 cycles,
440 verified drafts and 347 accepted drafts.

The union had deduplicated the expert weights, but still multiplied every
expert by every live activation column, including zero-route columns. Gate/Up
and Down now skip arithmetic for those columns while keeping all workgroup
barriers and zero output writes uniform. The M=1 specialization folds this
condition away. Sources: `gpu_union_sparse.log`, `gpu_mask_sparse.log`.

One first-turn cell after this correction produced exactly the same 234 tokens,
92 cycles, 184 verified drafts, 141 accepted drafts and 0.913674594 hit rate as
the first turn before it. Decode improved **11.6263 -> 12.2314 tok/s (+5.20%)**;
verification **15.877 -> 14.871 s**. Matched first-turn GPU traces show MoE busy
**71.924 -> 60.716 ms/batch**, total span **171.657 -> 160.807 ms/batch**.
Source directories: `warm3_k2_top4_wc/`, `warm1_k2_top4_sparse/`, including
`trace_first_turn.json`. The mirror peaked at 74 C, with zero rests/drops/errors.

The batch index checker now streams GPU-visible index lists into cached CPU
RAM before scanning entries and poisoned padding, including unaligned list
strides. The unchanged 234-token first-turn cell was **12.2907 tok/s** versus
12.2314 before (+0.49%, within throughput jitter); GPU span was 160.51 ms.
This is not credited as a significant speed gain. Source:
`warm1_k2_top4_indexwc/trace_first_turn.json`.

The final k=5 first-turn cell must **not** be used as a two-drive speed result:
at **2026-10-04 09:51:31 +0800** the kernel recorded Thunderbolt disconnect and
PCIe Link Down. Source 1 was dropped after 9 errors, reread from primary, at a
maximum 74 C with no thermal rests. `warm1_k5_top4_final/validation.json` marks
it invalid; the mount and external NVMe disappeared. No A/B verdict is taken
from its 8.97 tok/s. A valid dual-source k=5 cell remains pending the mirror's return. Generated
MMLU was instead checked as a separately labelled primary-only smoke sample.

Its accounting still illustrates the cost model: 78 cycles submitted **468
target rows** and emitted 245 tokens; only **168/390 drafts** were accepted.
There was only one path, not six accepted tokens per cycle. The active routed
expert union summed to 56,221 over 78 x 40 layer steps, or **18.02 experts per
layer/batch**, although each position requests six. Even deduplicated weights
must cover that larger union. These are this invalid cell's counters, not a
claim of what an uninterrupted dual-source benchmark would achieve.

Weight reuse does not make a batch free: different positions route to an
expert union larger than one position's set; that union still streams from
LPDDR into the GPU. Activation work, attention, draft computation and host
synchronization remain. The 384 pinned draft experts also reduce main-model
cache capacity. For each cycle, throughput depends on
`(accepted + 1) / (draft_ms + verify_ms + commit_ms + other_ms)`. A 1.6x gain
requires that entire denominator to fit the corresponding budget, not merely
one target forward call.

The comparison harness waits for GPU <=60 C and mirror <=70 C before startup.
The existing mirror's 80 C pause / 72 C resume gate stays active throughout.

## Validation and quality

`bench/results/dspark_topk/` preserves build, CPU/Python, GPU and quality logs.
The CPU acceptance test checks first rejection, unchanged prefixes, stable
lower-ID tie ordering and invalid matrix/K handling. The store test checks
pinning preserves LRU stamps and prevents recycling. The GPU mask test covers
masked absent slots and a batch with an empty routed union/shared-only output.

The complete MTP chain's golden test has all 45 router IDs identical, per-stage
cosine >=0.999, and head-norm cosine 0.9990497. Four of five draft tokens match
exactly. The fifth changes from reference 671 to its second-ranked 3098 at a
0.062426-logit near-tie; the test permits only a reference top-two tie <0.1,
with explicit activation/confidence bounds. Draft equality is not a claim of
lossless target generation.

The final off-mode L3 check reproduces **NLL 0.622784** (64 steps, 56/64 top-1).
Both required decode suites pass again. CPU ctest is 25/25 and the serialized
Python gate is 30/30. The streamed readback test sweeps all 32 source alignments
and lengths covering both heads and tails, with destination guards.

The wrap regression verifies a six-row target batch at position 128 truncated
to one retained input by a stop token: rejected window slots and untouched
compressor carry slots restore exactly. Resetting draft KV then reseeding
checks the restored main hidden window, followed by context reset/restart.

The earlier miss-mask result remains a separate **57-question/57-subject
zero-shot logit MMLU sample**: off 45/57, mask 44/57. It is not a DSpark score:
an immediate first-token answer would bypass speculative acceptance. Generated
MMLU uses `Answer: X` output and counts invalid formats as incorrect, recording
speculative cycles to prove the new rule participated. Its small pilot is a
functional quality check, not a full MMLU estimate.

The final generated smoke sample ran the first six questions on **one primary
source** (`DEEPMOE_MIRROR_AUTO=0`), 5,500 slots, temperature 0, max 16 tokens,
DSpark k=2/top-4. Mask was **5/6**, mask+DSpark **6/6**, both with **zero invalid
formats**. There were 6 speculative cycles and 12/12 accepted drafts; all six
answers used verification. Source:
`bench/results/dspark_topk/mmlu_generation_primary/summary.json`, with every
answer and provenance retained. This sample is too small to infer an accuracy
gain, uses a different protocol from the earlier 57-question logit sample, and
is not a full MMLU result or a dual-source benchmark.

A final primary-only wrap regression also passes after the index readback
change (`gpu_wrap_final_primary.log`); the independent readback/alignment test
and the final CPU/Python logs pass. No engine was left running. External NVMe
recovery is still required before resuming source-matched performance work.


## 单 kernel 草稿链与缓存复用（2026-10-04）

`DEEPMOE_DSPARK_MEGA=1` 启用真正的融合草稿 kernel。默认保持原路径。
这次融合覆盖三个 MTP 阶段、MoE、完整输出头、五步 Markov 采样和置信度。
每次草稿只有 **一次 `vkCmdDispatch`、一次提交和等待**。
原路径有 87 次 GPU dispatch（此前的 81 是提交次数，其中三个 MoE 提交各含三个 dispatch）。
最终版本在同一个 kernel 内执行 **93 个阶段**，包括 93 次有界网格同步。
它仍同时计算五个草稿位置；没有缩小双向注意力的输入。

中间结果复用创建时分配的同一块 GPU scratch。三个 MTP KV 环在 GPU 上跨轮保留。
提交主模型 hidden 时，每五行用一个融合 kernel 更新三个 KV 环，包括回绕写入。
草稿 KV 不再在每轮从 CPU 环拷回。CPU 只准备输入和参数，并读取最终结果。
MoE 路由去重在 GPU 上完成；384 个已固定的专家通过地址表读取。
主模型的 LRU、P0、miss mask 和单路径 top-K 验证规则保持原有实现。

共享内存按阶段的调用图确定存活范围。同一阶段会同时使用的数组互不重叠；
不同阶段的 float、uint、int 数组按位复用同一块存储。
源码分配从 40,704 B 降到 20,480 B；驱动实际分配从 **40 KiB 降到 21 KiB**。
VGPR 从 **256 降到 120**；每个 SIMD 的 wave 上限从 **4 升到 12**，没有寄存器溢出。
最终使用 120 个固定工作组。输出中的 BF16 舍入直接在 MoE Down 的最终写入完成；
草稿 KV 的 RoPE 直接写入 BF16 KV 尾部。这样去掉了六个转换/搬运阶段及其网格同步。

网格使用 GPU 设备范围原子计数和 epoch，并限制等待次数。
SPIR-V 使用 Vulkan device memory model；可写中间结果的读写显式声明可见性和可用性。
不可变参数表与标记过的权重读取保持缓存策略。实现仅允许 RADV Radeon 8060S，
超时会返回错误。它不依赖 Vulkan 提供标准的协作式网格启动保证。

一次 kernel 不代表权重都留在片上。**1.323 GB 的 BF16 输出头仍要从 LPDDR 读取**，
三个 MTP 阶段的权重也不同。融合保留已有的五列权重复用，并没有消除这些字节。
低并行度阶段、93 次网格同步及中间结果的设备可见性仍有成本。
目前没有内部阶段计时或最终 DRAM 流量计数，不能把所有剩余差距归给某一个原因。

每个配置只运行一次。下表用同一输入、无比对探针的完整 `wall.draft` 计时。
每一行只与该行的原路径相比；各行有独立的供电和时钟状态。
这些是常驻草稿链测试，不能替代双盘对话的端到端速度。

| 配置 | 原路径 ms | 融合路径 ms | 融合 kernel ms | 内部阶段 | 结论 |
|---|---:|---:|---:|---:|---|
| 可缓存权重，40 工作组，40 KiB LDS | 25.760854 | 47.013863 | 46.262917 | 99 | 慢 82.5% |
| 参数表改为 descriptor，40 工作组 | 26.666594 | 36.201638 | 35.506776 | 99 | 慢 35.8% |
| 分开各阶段任务循环，40 工作组 | 26.701373 | 34.890772 | 34.103476 | 99 | 慢 30.7% |
| 不同类型复用 LDS，80 工作组，28 KiB | 25.800576 | 30.718066 | 29.855221 | 99 | 慢 19.1% |
| 按阶段存活范围复用，120 工作组，21 KiB | 27.435793 | 30.799296 | 29.982803 | 99 | 慢 12.3% |
| 转换并入输出写入，120 工作组，电池供电 | 40.263823 | 42.635535 | 41.478518 | 93 | 同行慢 5.9%；不能与前面的外接供电结果横比 |
| 最终确定性布局，主 checkout，电池供电 | 45.475171 | 43.177682 | 42.135241 | 93 | 同行耗时减少 5.1%；这是最终回归检查，不是外接供电结果 |
| 最终确定性布局，主 checkout，稳定外接供电 | 26.062215 | 30.188503 | 28.872565 | 93 | 同行慢 15.8%；完整草稿链和 KV 回绕逐位一致 |

转换并入输出写入的那一行运行时外接供电断开。运行前传感器仍显示 USB-C 输入 3 A，
运行后 `AC0/online=0`、USB-C 输入 0 A，风扇停止、空闲 GPU PPT 约 5 W。
这是供电状态变化，不是数值失败；不会将 30.80 → 42.64 ms 解释成该改动的退化。
确定性布局排序固定了等大小数组的放置顺序，最终主 checkout 的完整草稿链已重新逐位验证通过。
14:01 的一次稳定外接供电检查补齐了最终配置：原路径 26.062215 ms，融合 30.188503 ms，
融合 kernel 28.872565 ms，仍慢 15.8%。这是常驻草稿测试，模型仅从主盘加载，不能替代双盘对话测速。
供电从开始到结束保持 AC，GPU 最高 63 C、主 NVMe 45.85 C、CPU 67.75 C，无温控暂停。
来源：`bench/results/dspark_topk/mega/final_ac_golden.{log,json}` 和对应 `_thermal.jsonl`。
这格重新验证全部中间结果、646,400 个 logits、token/置信度及五行 KV 回绕。

全部已有融合配置均通过 16 处中间结果、45 个路由 ID、646,400 个 logits、
五个草稿 token 和置信度的逐位比较。五行提交跨越 128 槽 KV 环回绕也逐位一致。
目标 `suite.decode` 和 `suite.decode_longctx` 通过；后者 4K/17K 的 TF、自由生成均 8/8。
最终主 checkout 的 CPU gate 25/25，Python gate 30/30。
64 步 off 模式 NLL **0.622784**、top-1 **56/64**，保持平台基线。
质量门禁温控日志：GPU 最高 51 C，主 NVMe 最高 56.85 C，CPU 最高 63.75 C，无暂停。
融合改变调度，不据此声称新的 MMLU 分数或 1.6x 加速。
外接供电的已测配置未超过原路径；最终电池检查耗时只减少 5.1%，故保留为显式实验开关。
已有的 MMLU 样本见上文，未重跑完整 MMLU。

实现入口：`runtime/dspark_runtime.cpp`、`gpu/shaders/dspark_mega.slang`、
`gpu/vulkan/dspark_mega.cpp`。生成器复用原有 Slang 算术，文件位于 `tools/fuse_dspark.py`；
内存语义修正由 `tools/dspark_spirv_memory.py` 完成，并经过 `spirv-val`。
结果和驱动统计位于 `bench/results/dspark_topk/mega/`，各配置的日志名对应上表。
最终主 checkout 与工作树的 generated arithmetic 相同，去除调试信息后的 SPIR-V SHA256 相同。

## 七项优化路线的核对（2026-10-04）

本节记录实现前的核对；下节记录落实结果。这里的预测不能作为已取得的收益。

先固定同一格的周期账。`warm1_k2_top4_indexwc/turns.json`：92 个周期、233 个输出位置，
每周期输出 **2.532609 token**，耗时 **206.059224 ms**，速度 **12.2907 tok/s**。
其中 draft **34.086501 ms**（包含首次 KV 播种）、verify **161.316928 ms**、commit **2.830825 ms**；
余下 **7.824969 ms** 包含周期前后准备和 CPU 接受/采样，原日志没有单独分开。
不能把余项全部算成 top-K 接受成本。

同一格 target GPU trace：attention **43.650532 ms**、MoE **61.036365 ms**、Engram **4.501642 ms**、
CED **1.395166 ms**、tail **9.658433 ms**，busy 合计 **120.242138 ms**，gap 合计 **40.269228 ms**。
其中 MoE 前后的 gap 是 **35.691023 ms**，不是全部 fence：还含路由回读、planner/P0 发起、
激活量化、并集与地址表组装。下面的收益都必须由新配置的同口径测量给出。

| 原方向 | 核对结果 | 下一步与依赖 |
|---|---|---|
| 1. 缩短 verify 间隙 | 路径每层仍执行 `cmd_flush`，随后 CPU 路由、planner 和 `stage_batch_union`。全部 40.27 ms gap 不能直接算作可省时间。CPU 写标志、GPU 等待的旧机制在 STATUS §7 0g/5 已关闭；目前没有新 Linux 测量能推翻它。 | GPU 驻留筛选/紧凑并集是新机制候选；必须处理发布可见性、槽位 guard 和 mask 时刻的一致性，CPU 的正常 LRU/P0 仍执行。先量各段，再做最小探针。 |
| 2. 移植 decode 优化 | `record_attention_batch` 仍是 MGT 的 score/PV，确实没有 ATTN_CM。但旧三轮 trace 的 45.30 ms attention 中，Q/输出投影三项占 29.83 ms，score/PV 仅 6.48 ms。只移植 ATTN_CM 不足以省 10–15 ms。旧 Windows M=5 的 25.3 ms floor 不适用于这格 Linux M=3。 | 优先投影的批内权重复用、MHC 调度与 ATTN_CM；分别归因，验证数值边界。保留当前并集 MoE 对多列的权重复用。 |
| 3. draft 降本 | typed_overlay80 的 29.3/31.2 ms 来自带比对探针的旧版本。最终稳定 AC 的无探针格是串行 26.06、mega 30.19 ms，不能计融合收益。 | 五个位置在三个 MTP 阶段做双向注意力，不能把 M 缩到 k 而声称同一草稿。可单独研究输出头/Markov 尾部截断、按需 logits 回读；需要改变接口并验证前 k 个决策。 |
| 4. draft 与 target 尾部重叠 | 完整下一轮 draft 除 hidden 外，还需要下一轮 root token。这个 token 在第一次拒绝时是 correction，全接受时是 bonus，均依赖 target 输出头和接受/采样。只做 KV 前缀回滚不能解除这个 token 依赖。 | 可研究 KV 投影/环更新的准备与尾部重叠，但须暂存未决定的前缀并处理回绕/回滚。不能直接记 5–10 ms 完整 draft 重叠收益；同一 compute queue 上的异步提交也不自动形成 GPU 执行重叠。 |
| 5. GPU 接受与采样 | k=5 的 12.02 ms CPU 数据来自掉盘的无效速度格，不能套到 k=2。当前 CPU 区间含 `accept_topk_prefix`、`nucleus_from_full` 的全词表 exp/候选排序及结果复制。只读取 top-4 ID 不足以执行 top-p=0.95 采样。 | 复用已有 MGT `HeadTopK` 的候选和尾部质量接口（当前 target 未启用）；GPU 判定接受前缀，校验候选质量，不足时仅回读所选拒绝/bonus 行。精确保留采样约定，仍只对主路径做一次 target 前向。 |
| 6. 动态 k，包括 0 | live `speculative_step` 按固定 max_draft 取 k，没有消费 draft confidence。旧 `Speculator` 有 sigmoid 累乘规则，但它不是当前 live 路径。单条 golden 链的五个 confidence 不能证明主模型 top-4 接受概率或 +3–6% 收益。 | 先联合记录 confidence、target rank、首拒位置并校准。取得本轮 confidence 前已经支付了完整 draft；本轮 k=0 只能省 verify 扩展行，跳过 draft 要有上一轮/更便宜的预测。 |
| 7. 384 MTP 专家加入 LRU | 槽位占用确实是 5500→5116。但当前 mega 的专家地址表只在 create 时建立，依赖永久 pin；直接 unpin 后槽位回收会使地址指向别的专家。现有 hitrate_sim route.bin 仅包含 40 层主模型路由。 | 先加入三个 MTP 阶段与主模型交错的访问 trace，再离线模拟。运行时必须更新地址表/驻留快照并加执行期 guard，才可讨论 draft miss mask；主模型正常 LRU 保留。 |

优先顺序：**2 的投影归因/移植和 5 的接受加采样；1 做不同于旧 host-flag 的最小机制探针；
3 保留原路径并只改可截断的尾部；6/7 先记录与离线校准；4 暂不计收益。**
项 4 与项 5 涉及同一段尾部，且异步工作仍争用 GPU/内存；不可直接相加。

原汇总的算术须区分基线。假设输出数不变且周期确实从 206.059 降到 160–165 ms，
速度是 **15.35–15.83 tok/s**，相对当前 DSpark 是 **1.249–1.288x**；
**1.158–1.195x** 是相对所引用的约 13.25 tok/s mask-only 成绩，它的文本/路由不同，不能当相同工作的加速比。
按项目规则减半的是预计节省的毫秒：周期变为 **183.030–185.530 ms**，
即 **13.65–13.84 tok/s**，相对当前 DSpark **1.111–1.126x**，相对 13.25 则约 **1.030–1.044x**。
这些只是条件算式，并不证明上述时间目标能实现。

同样固定 2.532609 token/周期，做到当前 DSpark 的 1.6x 要 **≤128.79 ms/周期**；
做到约 13.25 tok/s mask-only 的 1.6x 要 **≤119.46 ms/周期**，比当前少 **86.60 ms**。
即便完整 draft 免费，verify 自己仍有 161.32 ms，所以单独优化草稿不能实现这个目标。


## 本轮端到端落实与验收（2026-10-04）

### 已实现的路径与开关

- 主模型仍执行正常 LRU/P0；mask 只令缺失专家的计算权重为零，不重归一化，shared expert 始终计算。
- target 仍只对 `[root, draft[:k]]` 做一次前向，生成一个验证矩阵。GPU 计算原草稿 token 的精确全词表 rank，使用较小 token ID 打破同值并列。
- `DEEPMOE_SPEC_GPU_READOUT=1` 是默认。GPU 根据这些 rank 选择首拒行或 bonus 行，只为该行生成采样候选和尾部质量。最终随机抽样仍使用原 host sampler；候选不足时只回读所选行。所选 token 评分接口仍保留最后实际输出行的全词表 logits。
- `DEEPMOE_DSPARK_TRIM_TAIL=1` 是默认。三个双向 MTP 阶段仍计算五个位置；输出 head 的列数和顺序 Markov 后缀缩到实际 k，省掉未用 logits 的回读。前 k 个 logits/token/confidence 已与原五行版本逐位比较，串行和 mega 均一致。
- Engram 现在为每层、每个 batch row 保留独立落地区、GPU planes 和指针表。所有主路径 token 的 P2 请求提前发起，各行的原 decode 算子放进同一提交；不改变其算术，也不改变 P0/LRU。
- `DEEPMOE_MGT_ATTN_CM=1` 可复用普通 decode 的五个 CM 阶段，支持逐行因果表、overflow、compressed KV、RoPE 和独立 scratch。`DEEPMOE_MGT_FOLD_SCALE=1` 可在 GEMV staging 中复用同一 32-row scale band 的 UE8M0 权重 scale；此选项保持实验开关。
- `--spec-confidence-min X` 明确启用原始置信度的连续前缀策略，包括 k=0；遇到第一项低于 X 就停止扩展，上限仍由 `--spec-k` 限定。默认不设阈值，仍固定 `--spec-k 2`。本轮 draft 已经计算，所以 k=0 不能计作省掉 draft；k=0 仍只做一个 root 行的 target 前向。CPU 边界门禁及真实引擎两轮 k=0 均通过，后者每轮恰好调用 40 层 target 一次。
- 真正单 kernel 的 `DEEPMOE_DSPARK_MEGA=1` 继续可用，默认关闭。前一节稳定 AC 的实测仍是串行 26.06、mega 30.19 ms；不会把减少 dispatch 当作加速。

### 可比较的短速度格

`engram_early/{stock,early}/`：同一 29-token 中文 prompt，T=1/P=0.95/seed=41001，64 个输出，5500 总槽，主盘一读源，临时 power-saver 模式；均无温控暂停或 AC 切换。两格的 64 个 token、10617 个 union 请求、31862161408 miss bytes、38/49 接受/验证完全一致。

| 配置 | decode tok/s | 完整周期 ms | draft ms/周期 | verify ms/周期 | CPU ms/周期 |
|---|---:|---:|---:|---:|---:|
| GPU readout + trim，逐行 Engram | 7.745165 | 325.364270 | 34.650040 | 285.917372 | 0.179158 |
| 同配置 + Engram 提前取数/合批 | 8.006094 | 314.760218 | 33.544575 | 276.526986 | 0.144144 |

这是 **+3.369%** 的实测短格收益。target trace 的平均 gap **146.001490 → 127.461793 ms/周期**，Engram host 取数/landing **130.634663 → 112.652490 ms/周期**；busy **139.231083 → 148.200360 ms**。合批减少等待，但更连续的 GPU 工作在低功耗模式下并未保持相同 busy 时间，所以不能把 18.54 ms gap 的减少直接当成整个周期的节省。完整周期还包含约 **1.45 ms/周期** 的准备/接口余项，不能只加四个计时字段冒充完整 wall。

原 readout 的同文本 234-token 对比，CPU **12.232663 → 0.139881 ms/周期**，但两个格各有 28 次温控暂停，**不作速度比较**。所选行/并行 histogram 版本的 16-token 无暂停单测格，GPU rank/candidates 合计 **0.096223 + 0.367645 ms/周期**；CPU **5.703400 → 0.157258 ms/周期**。两个完整速度为 6.303/6.241 tok/s，在抖动内，不宣称端到端增益。来源：`gpu_readout/comparison.json`、`gpu_readout_selected/comparison.json`。

### CM、投影与热状态的结论

CM 对每个 row 复用原 decode 算术的合成门禁 M=1…6 全部逐位一致。真实引擎 12 个 teacher-forced 位置，M=3：最差 logits cosine **0.9949362**，top-1 **11/12**，M=1/batch NLL **0.700711/0.683117**；通过原回归线。首个 batch 的 score/PV busy 原路径约 **4.1–4.7 ms**，CM 五阶段约 **1.59 ms**。这不是 10–15 ms 的 attention 整体节省。

扩大到导出的 63 个有效位置后，CM 最差 cosine **0.8737936**，低于既有 **0.88** 门槛：**失败**；top-1 **55/63**，M=1/batch NLL **0.627547/0.647910**、PPL 比 **1.0206x**。来源 `final_gate/batch_quality_64.log`。不降低门槛；CM 恢复默认关闭，只留显式实验开关。其无暂停 64-token 组合格为 **7.198512 tok/s**，35/55 接受、28 轮；原路径 + Engram 的格是 38/49 接受、25 轮、8.006094 tok/s。输出和路由不同，不能把两格当相同工作的算子加速比；实际吞吐也没有支持默认启用 CM。

UE8M0 scale-fold 对五种投影、M=1/3/6、ActQuant 开/关和 WoA 分组 stride 的所有 partials 都逐位一致。小探针 M=3 多数降低几十微秒，但 M=6 WqA 出现反向变化；真实 trace 的投影总时间未给出稳定、超出抖动的收益。因此保留实验开关，不默认启用。

`end_to_end/` 的 balanced 格中 readout、fold、CM+fold 各触发一次 80°C 暂停；它们的 tok/s **不可用于收益结论**。`batch_cm/cm_unit.log` 的旧二进制执行了 0 个用例，明确无效；`cm_unit_fixed.log` 才是通过的门禁。Engram 探针最初用了空 TextConfig 而失败，修正为加载真实 config 后才通过，失败收据另存。最终长测试切换到临时 power-saver，结束后恢复原 performance 模式。

### 置信度与 LRU 离线账

`DEEPMOE_SPEC_DIAGNOSTICS=FILE` 输出原草稿 confidence、target rank、三个 MTP 阶段与 40 层 target 的交错专家请求，以及首次周期前的主模型 LRU。只有 serial draft 记录真实 MTP 路由，mega 诊断明确标为不可用。写入/flush 失败会报错；日志耗时计入公开 wall。

`tools/spec_diagnostics.py` 校验几何和连续性，按“前缀仍可到达”分别统计接受率，并模拟 pinned / 全 LRU / draft miss-mask。模拟使用固定轨迹、同步填充，忽略实际 P0 完成时刻、backfill、执行期 guard、同 stamp 槽位 tie 和 mask 对后续路由的反馈，**不能当运行时命中率、MMLU 或速度预测**。

首条 64-token、k=5 轨迹有 18 轮/90 个草稿。五个位置的前缀条件接受率为 **16/18、14/16、8/14、4/8、3/4**。负置信度的可达草稿仍有 **22/36 = 61.1%** 在 target top-4，不能直接用旧 sigmoid 代替 top-4 接受概率。没有独立校准集，因此动态阈值保持显式实验参数，未计 +3–6% 收益。

同一短轨迹尚未填满可回收容量：5116 + 固定 384 与 5500 全 LRU 都是 main **1993/14053 misses**、MTP **0/302 misses**，没有出现取消 pin 的收益。运行时仍保留 pin；mega 的地址表依赖槽位不回收，不能直接 unpin 而不更新地址、驻留快照和 guard。

后续 246-token 轨迹含 78 个投机周期、390 个草稿、168 个接受草稿，诊断覆盖 245 个投机输出（首个输出来自普通单位置路径）。来源 `selected_gate/diagnostics_long.jsonl` 和 `selected_gate/diagnostic_analysis.json`。五个位置的前缀条件接受率为 **64/78、52/64、31/52、12/31、9/12**；负置信度的可达接受率仍有 **79/142 = 55.6%**。这支持继续保留固定 k=2，不能从这一条轨迹确定泛化阈值。

该轨迹已出现容量压力。固定 MTP pin 的离线 main misses 为 **4976/61198**；全 LRU 为 **4842/61198**，命中率 **91.869% → 92.088%（+0.219 个百分点）**，同时引入 MTP **6/1327** misses。main 填充量减少 **2.519 GB**，MTP 增加 **0.113 GB**，净少 **2.406 GB**；draft miss-mask 模型是 main **4840**、MTP **8** misses。正反 union 顺序结果相同。上述固定路由模型不能表示异步填充、mask 反馈或实际阻塞，不能把这 2.406 GB 直接换算为加速。此次没有取消运行时 pin；取消 pin 需要重做 draft 地址快照、槽位执行保护和数值验收，当前收益证据不足以启用它。

完整下一轮 draft 仍依赖 correction/bonus root；同一 compute queue 上提前提交 KV 投影不证明执行重叠。旧 host-flag/persistent 方向的已关闭结论不变。此次真正省掉的是 Engram 的逐行 fence，没有取消 CPU 路由/LRU 的每层依赖，也没有计入原预测中的 15–25 ms 或 5–10 ms 完整 draft 重叠收益。

### 为什么没有 1.6x

保留配置的同 token 格，每周期实际输出 **2.52** 个 token，完整 wall 为 **314.760 ms**，即 **124.905 ms/token**。其中 draft **13.311 ms/token**、verify **109.733 ms/token**，其余为接受处理、提交和准备。`--spec-k 2` 不等于每周期稳定输出三个 token；本格只有 **38/49** 个草稿通过，仍支付被拒绝后缀的批计算。

批 MoE 已走 miss-mask 和专家 union：平均每周期 **424.68** 个 layer/expert union 项，即每个输出 **168.52** 项。专家权重在这一批的各 row 间复用，但不同 row 会选出不同专家，union 不是普通 decode 的固定六个专家。该格仍有 **505.75 MB/output** 的 P0 miss 读取，以及每周期约 **112.65 ms** 的 Engram host 取数/落地开销；mask 没有取消正常 LRU/P0，也没有取消 Engram。单 kernel draft 不会消除这些 target 成本。

同 prompt/seed、单盘、power-saver、64-token 的普通 miss-mask 参考格是 **8.473850 tok/s**（`final_gate/mask/`），投机保留格 **8.006094 tok/s**。两者实际输出和专家轨迹不同，而且投机占用 384 个 pin；这说明本短任务尚未出现投机加速，不能用于隔离一个算子的变化。**+3.369%** 只指前述同 token 的 Engram 合批前后，不代表投机相对普通 decode 的加速。此次没有第二块读盘，不能把这些数据替代历史双盘无损约 9 tok/s 的结果。

### 最终保留配置与门禁

七项方向的最终状态如下。收益列只采用有效实测，原预测不计入交付收益。

| 方向 | 本轮处置 | 收益与限制 |
|---|---|---|
| 1 缩短 verify 空隙 | Engram 独立落地区、指针绑定和同次提交；提前发起 P2 | 同 token wall 每周期少 **10.604 ms**；保留每层 CPU 路由/LRU 依赖。旧 host-flag / persistent NO-GO 不重开 |
| 2 decode 优化移植 batch | CM、scale-fold 已实现并测量 | CM 长数值门禁失败，默认关闭；scale-fold 无稳定端到端收益，默认关闭 |
| 3 draft 降本 | 真正单 kernel mega、实际 k 的输出尾部裁剪 | mega 比串行更慢，默认关闭；串行裁剪仅约 **0.217 ms**，不计超出抖动的速度收益；三个双向阶段仍计算五个位置 |
| 4 draft / target 尾部重叠 | 核对数据和 compute queue 依赖 | 下一轮完整 draft 依赖 correction/bonus root；没有实测重叠，不计预测收益 |
| 5 GPU 接受 / 采样 | GPU 精确 rank，首拒/bonus 行候选，保留 host 随机抽样与所选行 fallback | CPU 显著下降；同文本长格有温控暂停，仅用于 CPU 归因。16-token 无暂停格未显示超出抖动的吞吐变化 |
| 6 动态 k 含零 | 连续置信度前缀参数和真实 k=0 门禁 | 可显式试验；置信度尚未独立校准、选择时 draft 已支付，默认固定 k=2 |
| 7 draft pin / mask | 交错请求诊断和离线三种 LRU 模型 | 246-token trace 的全 LRU 净少 **2.406 GB** 填充；不代表阻塞减少。运行时保留 pin，未引入未验证的地址回收 |

| 功能 | 最终设置 | 退出或试验方式 |
|---|---|---|
| 主模型 miss-mask | `--resident-only mask`；正常 LRU/P0 保留 | `--resident-only off` 回普通路径 |
| 主路径投机 | 显式 `--dspark`，默认 k=2、接受 top-4 | 不传 `--dspark`；长度用 `--spec-k` |
| GPU rank / 所选行候选 | 默认开启 | `DEEPMOE_SPEC_GPU_READOUT=0` |
| 草稿输出尾部裁剪 | 默认开启 | `DEEPMOE_DSPARK_TRIM_TAIL=0` |
| Engram 提前取数/合批 | 默认开启 | `DEEPMOE_BATCH_ENGRAM_EARLY=0` |
| 批 CM、投影 scale-fold、mega | 默认关闭 | 分别用 `DEEPMOE_MGT_ATTN_CM=1`、`DEEPMOE_MGT_FOLD_SCALE=1`、`DEEPMOE_DSPARK_MEGA=1` |
| 动态 k | 默认不设阈值 | 显式 `--spec-confidence-min X`，不节省已支付的 draft |
| MTP 专家 pin | 保留 384 个 | 仅有离线敏感度分析，无运行时 unpin 开关 |

`selected_gate/` 在上述保留配置、主盘一读源和临时 power-saver 下逐项串行运行。CM 关闭后，63 个有效 teacher-forced 位置的最差 cosine **0.9466123**、top-1 **55/63**，M=1/batch NLL **0.627547/0.645183**、PPL 比 **1.0178x**，通过原门槛。这是批路径一致性门禁，与下面 64 步 off 基准的协议不同。

| 验收 | 结果 | 收据 |
|---|---|---|
| CPU / 工具门禁 | **30/30** | `selected_gate/cpu_gates.log` |
| 普通 decode | 通过 | `selected_gate/decode.log` |
| 4K / 16K 长上下文 | free / teacher-forced 均 **8/8** | `selected_gate/decode_longctx_detail.log` |
| 64 步 off 数值基准 | NLL **0.622784**，top-1 **56/64**，保持平台基线 | `selected_gate/l3_off.json` |
| 拒绝后缀 / 窗口回滚 | 通过，执行了一个真实用例 | `selected_gate/wrap_selected.log` |
| k=0 只做一次 target | 两周期均恰好调用 40 层一次 | `final_gate/adaptive_zero.log` |
| 57 科目生成式 MMLU | **48/57 = 84.2105%**；2 个格式无效计错 | `selected_gate/mmlu_generated_57/summary.json` |

GPU rank/所选行候选、草稿前缀裁剪、Engram 独立 row planes 的专门门禁均已执行；CM 的合成逐位门禁通过不替代其失败的长回归门禁。每个作业的 rc、峰值温度、温控暂停次数和供电状态保存在相应 `check_results.json`；0-case 的旧收据明确作废。

### MMLU 生成式验收协议

最终样本取 `bench/results/miss_mask/mmlu_sample.json` 的前 57 行，每个科目一题，seed=42。源是 `cais/mmlu` 的 test 数据，文件 hash `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`。这是 **57 科目零样本抽样**，不是完整约 14K 题的标准五样本分数。

每题重置 KV，保留专家缓存；除最后一个 prompt token 外，前缀用精确 GPU prefill。随后通过单位置路径处理最后一个 prompt token，再生成最多 16 个 token，T=0、k=2、accept top-4。要求输出 `Answer: X`；未按格式作答或截断前没有答案均计错误。记录全部原始输出、采样 token 和投机统计，确认投机参与答案生成，避免只评分首个答案 token 而绕过投机。

早期 57 题 off **45/57**、mask **44/57**（`bench/results/miss_mask/mmlu/`）采用另一条指令和单位置 A/B/C/D logit 评分，不能与本生成式协议直接相减来归因投机质量。此前同协议六题 pilot 中 mask **5/6**、mask-spec **6/6**，样本太小；临时 CM 配置的六题 pilot 也不替代最终关闭 CM 的 57 题测试。

最终关闭 CM、保留 GPU readout/trim/Engram 合批的 mask-spec 结果为 **48/57 = 84.2105%**，其中 **2** 个格式无效计错。全部 57 题均至少执行一轮投机，共 **66** 轮、**130** 个验证草稿、**129** 个接受草稿；每题都验证了精确 GPU 前缀和只扩展最后一个 prompt token 的路径。原始答案、token 和逐题统计见 `selected_gate/mmlu_generated_57/mask-spec/answers.jsonl`，总表见 `selected_gate/mmlu_generated_57/summary.json`。这给出当前协议下的质量样本，不证明 top-4 接受保留原模型分布或质量无损。

### 收尾与温控

最终 `selected_gate` 七个 GPU 作业均 **rc=0**、执行了实际用例或样本、**0 次温控暂停**、AC 一直在线。峰值 **GPU 72°C、NVMe 71.85°C、CPU 73.5°C**；MMLU 的完整作业 wall 为 **1137.118 s**，仅用于识别作业耗时，不作 decode 速度比较。`selected_gate/check_results.json` 保存各项收据，同目录的 `*_thermal.jsonl` 保存温控轨迹。最后已恢复 **performance → power-saver → performance**，见 `selected_gate/profile_receipt.json`；没有遗留 `deepmoe` / `deepmoe_tests` 进程，开始验收时也没有需恢复的 web engine。

`selected_gate/source_manifest.json` 收录 259 个源码文件（含未追踪的新实现）、实际二进制和 50 个 shader 的 hash，已与 MMLU provenance 匹配；验收期间源码未变。二进制 SHA-256 前 16 位 **0d782e23e71c6124**，shader 集合 **214a8d1c095876be**。各速度格仍以各自启动时的 provenance 为准；没有把文档更新后的 diff hash 冒充当时运行版本。

本轮保留有效实现，记录失败和无效格。没有达到 1.6x，也没有重新宣称双盘速度、完整标准 MMLU 或近似接受无损。

## 13. 性能回退定位与修复（2026-10-04）

本节解释上一节单盘速度偏低，并记录最终源码上的修复和验收。所有证据位于 `bench/results/engram_deadline/`；当前机器只有主盘读源，外接镜像未挂载。历史双盘普通 mask 首轮是 **13.250358 tok/s**，DSpark k=2 首轮是 **12.290684 tok/s**；“13”对应普通 mask，不能与本次单盘、power-saver、64-token 投机格直接比较。

### 已确认的两处问题

**1. 必要的 Engram 读请求被异步专家填充堵住。** miss-mask 不等待 P0 专家填充，但原 I/O dispatcher 仍严格先发 P0；P0 的队深满后直接停止发请求，P2 小读请求不能借用其自身队深。GPU 到达 Engram 时必须等待这些 P2，因而把“不等待 miss”变成了“间接等待 P0 队列”。新统计将 P2 总延迟拆成排队和服务，并把 Engram landing 拆成 future 等待与 CPU staging。

最终普通 mask 格的 Engram bucket 从 **39.643 → 4.791 ms/token**。P2 请求平均延迟 **30.64 → 3.94 ms**，其中排队 **27.95 → 2.43 ms**。126 次 landing 的 staging 总共只有 **1.800 ms**，不是几十毫秒的复制瓶颈。投机 target trace 的平均 gap 从 **128.531 → 20.084 ms/周期**；相应 target span **277.347 → 162.681 ms/周期**。这里的 trace 只包括 target，不含 draft。`comparison.json`、`verify_trace.json`、各格 `trace.bin` 与结束时 I/O 状态提供原始证据。

修复为 `EngramRunner::land` 的限时作用域：仅在 mask 模式、host 正在等待 Engram 时，允许不超过 8 KiB 的 P2 请求绕过排队的 P0。额外在途 P2 上限 **96** 个 chunk，总队深不超过 `min(engram_qd, p0_qd + 96)`，继续受原字节上限约束。作用域结束后恢复原严格优先级；P2 仍是 P2、仍走主盘，未把它变成可镜像条带的 P0。普通 off 模式不启用此策略。主模型 P0 请求、正常 LRU、miss mask 和 shared expert 继续执行。`DEEPMOE_IO_ENGRAM_DEADLINE=0` 可关闭新策略。

第一次候选只在 engine 初始化时绑定策略，但 CLI 在初始化后才设置 mask，实际 **0 deadline chunks**。该候选没有收益，收据保留于 `measure/` 和 `inactive_candidates.json`。随后将绑定移到真实 `Engine::set_resident_only`，CPU 集成测试也通过该 setter 启用，最终普通格计数 **5987**、投机 k=2 格 **7680**。不能只凭“编译了新代码”判断策略已经执行。

**2. 同一轮规划中，后来的 miss 可以淘汰前面已选中的 hit。** 在异步填充占满其他槽时，一个 hit 可能是唯一可淘汰槽。原逻辑在完整规划结束后才添加 GPU guard，存在命中地址失效窗口。最终将 hit guard 与 `ExpertStore::lookup` 在同一 store 锁内设置；普通 mask 和 batch mask 均在规划前分配本层 guard，按已有 GPU fence 释放。LRU 排名规则保留，正在执行的专家受生命周期保护。

两槽 CPU 用例确定性复现这一边界，并验证保护后 miss 无可用槽时被 mask、填充完成后可继续正常淘汰。首个 2000 槽窗口回滚 GPU 验收曾在 `REQUIRE(first)` 失败；原收据没有具体 Result 错误，因此不宣称已确定该失败唯一由此窗口导致。保留失败收据，改为 `REQUIRE_OK` 输出实际错误；诊断版曾通过，最终带原子保护的版本也通过。没有把一次诊断重跑通过当作修复证明，CPU 复现与最终验收分别保存。

### 有效端到端速度

均使用 native FP4/FP8、相同 prompt/seed、5500 总槽、64 个生成 token（63 个 decode 输出）、单读源、AC 在线。投机另占 384 个 MTP pin，k=2、接受 top-4，CM/scale-fold/mega 均关闭；GPU readout、trim、Engram 合批开启。下表前三行均为 power-saver，无温控暂停。缓存完成时间改变后，实际文本、路由和 mask 命中率也会改变，因此这些是任务端到端测量，不是逐位同输出的 kernel A/B。 普通 mask 的请求服务率从 **89.85% → 83.69%**、平均丢弃 gate mass **7.92% → 13.54%**；投机 batch 的逐 row 服务率 **89.27% → 84.15%**、丢弃 mass **8.69% → 13.23%**。更快生成给填充留下的时间减少，且文本改变了路由。I/O 等待改善有独立统计，但总吞吐收益不能全部记作等量计算变快；57 题质量样本相同也不等于其他输入无损。

| 配置 | 修复前 tok/s | 最终 tok/s | 实测变化 | 原始格 |
|---|---:|---:|---:|---|
| 普通 miss-mask，power-saver | 8.480823 | **11.880114** | **+40.08%** | `baseline_mask/` → `final_power/fix_mask/`，ledger #42 / #44 |
| mask + DSpark k=2，power-saver | 7.938441 | **10.830064** | **+36.43%** | `measure/baseline_spec/` → `final_power/fix_spec/` |
| mask + DSpark k=3，power-saver | — | **9.967502** | 比 k=2 慢 | `final_power/spec_k3/` |
| 普通 miss-mask，balanced | 无同模式基准 | **13.228557** | 不作跨功耗加速比 | `balanced/fix_mask/`，ledger #45 |

`final_power/check_results.json` 中三格均 rc=0、零暂停。balanced 普通 mask 的 GPU 峰值 **75°C**，也无暂停。performance 的普通/投机格分别有 **5/7 次**温控暂停，balanced 投机在 **80°C** 暂停后主动终止；这些格保留记录，**不能作为速度或回退证据**。本次有效 13.23 是单盘 balanced 普通 mask，不冒充此前双盘成绩。

### 为什么投机仍未超过普通 mask

| power-saver 每周期成本 | k=2 | k=3 |
|---|---:|---:|
| 实际输出 token | 2.250 | 2.739 |
| target row 数（含 root） | 2.964 | 3.957 |
| draft ms | 38.847 | 37.866 |
| verify ms | 163.574 | 231.611 |
| commit ms | 3.653 | 3.505 |
| CPU 接受/采样 ms | 0.207 | 0.177 |
| 完整周期 wall ms | 207.755 | 274.806 |
| Engram future 等待 ms | 4.594 | 42.744 |

k=2 的 28 轮共验证 **55** 个草稿、接受 **35** 个、输出 **63** 个 token。按实际输出算，完整周期约 **92.34 ms/token**，仍高于普通 mask 的 **84.17 ms/token**。接受数量不等于 target 计算数量，被拒绝的后缀已经计算。草稿约占本周期 **18.7%**，CPU 接受已不是主要成本。

批 MoE 已走 mask/union，并在一批内复用同一专家权重；这一格仍有每周期约 **397.93** 个 layer/expert union 项，约 **176.86** 项/output。不同 row 的专家集合扩大了工作量；复用权重不等于整个批与一个 token 同价。P0 miss 请求总量仍为 **31.636 GB / 63 输出 ≈ 502.17 MB/output**，代表正常 LRU 填充量，不代表 GPU 等待了全部这些字节。

k=3 增加接受输出，但验证成本增加更快，且 Engram host 等待重新增大。trace 中 target GPU busy 也增加，gap 从 k=2 的 **20.084** 增到 **60.596 ms/周期**；不能把新增验证耗时全部归因于算术，也不能凭这两格认定另一个唯一 I/O 根因。保留实测较快的默认固定 k=2。真正单 kernel mega 的 powered resident 测量仍是 **30.1885 vs 串行 26.0622 ms**，默认关闭；CM 的长质量失败也保持关闭，不用这些已失败方向换取表面速度。

每轮依旧只把 `[root, draft[:k]]` 的**主路径**送入一次 target 前向，得到一个验证矩阵。原草稿 token 命中该行 top-K 才接受；首次拒绝后采样并结束本轮，回滚未接受后缀。没有额外候选路径前向。新修复解决了实际排队回退，没有得到投机相对普通 mask 的 1.6x。

### 最终源码验收

二进制 SHA-256 前 16 位 **1423d089d6464fb8**；259 个源码文件和 50 个 shader 的完整 hash 见 `source_manifest.json`。MMLU、最终速度格的 provenance 均匹配实际二进制；验收期间编译源码未变。

| 验收 | 最终结果 | 收据 |
|---|---|---|
| P2 等待作用域/正常优先级恢复/96-chunk 上限 | CPU 用例通过；真实 setter 已覆盖 | `cpu_guarded.log`、`tests/test_io.cpp` |
| 同轮 hit 生命周期 | 两槽压力复现与保护用例通过 | `cpu_guarded.log`、`tests/test_store.cpp`、`tests/test_integration.cpp` |
| CPU gate / Python 工具 | **25/25；30/30** | `cpu_guarded.log`、`final_cpu_gates.log` |
| 普通 decode | **2 用例通过现有门禁**；短 oracle teacher/free 为 **6/8**，自主 prefill 为 **7/8**，未调整原阈值 | `final_decode_gate/{check_results.json,decode_detail.log}` |
| 4K / 16K 长上下文 decode | **free / teacher-forced 均 8/8**，两用例通过 | `final_decode_gate/{check_results.json,decode_longctx_detail.log}` |
| 64 步 off 基准 | **NLL 0.622784，top-1 56/64**，保持平台基线 | `guarded_gate/l3_off.json` |
| 2000 槽拒绝后缀/窗口回滚 | 通过 | `guarded_gate/wrap_selected.log` |
| 512 槽动态 k=0 | 两轮均只调用 target 40 层一次 | `guarded_gate/adaptive_zero.log` |
| 同版本新策略关/开，生成式 MMLU 6 题 | 均 **6/6**，答案及全部 emitted token 一致 | `guarded_gate/{mmlu6_baseline,mmlu6_fix}/` |
| 57 科目生成式 MMLU | **48/57 = 84.2105%**，2 个无格式答案计错；67 轮、128/130 接受/验证 | `mmlu57/evaluation/summary.json` |

57 题沿用 §12 的零样本生成协议、同样本 hash、每科一题、最多 16 token、精确 GPU 前缀和最后一个 prompt token 的单位置处理。**不是完整标准五样本 MMLU。** 每题均实际执行投机。与上一节最终收据逐题比较，**57/57 答案和正确性相同，56/57 完整 emitted token 序列相同**；差异只在同一个原本格式无效的截断答案中。总分相同不证明 mask/top-K 分布无损。比较见 `mmlu57_comparison.json`。

此次 MMLU 作业 **1045.868 s**、rc=0、零暂停，GPU/NVMe/CPU 峰值 **71/69.85/72.625°C**，AC 一直在线；该作业总耗时不当作 decode 吞吐。各 wrapper 结束后恢复原 performance，温控轨迹与 `profile_receipt.json` 全部保留。失败/暂停格没有进入有效速度表。最终验收汇总见 `final_acceptance.json`。

最终二进制上的普通 decode / 长上下文作业分别 **142.783 / 126.848 s**，均 rc=0、零暂停；两格 GPU/NVMe/CPU 峰值 **63/64.85/67.375°C**。4K、16K 的自由生成和 teacher-forced 均 **8/8**。短 oracle 的 6/8 与自主 prefill 的 7/8 属于 STATUS §7 已记录的普通 ATTN_CM 平台门禁，本次没有修改 `tests/test_decode.cpp` 的任何阈值。普通 off64 的 NLL **0.622784** 基线也保持不变。

## 14. 按端到端方案执行：草稿 ONECB 与验证 GPU 快照路由（2026-10-05）

执行方案为 `docs/dspark_e2e_plan.md`。独立工作树 `../deepmoe-spec`，分支
`spec-e2e`，从当前已验证代码的快照 `66d0f7a` 开始，独立 Ninja/Clang Release
build。网页按用户本次要求停止。没有写入 checkpoint，没有并行 GPU 作业。
结果根目录为 `bench/results/spec_e2e/`；下面的路径均相对该目录。

### 实际改动与安全边界

`DEEPMOE_DSPARK_ONECB=1` 将原串行草稿的独立 pipeline 放在同一 command
buffer，使用 compute→compute barrier，只在末尾 submit/fence 一次。三阶段
双向 attention 仍计算 M=5，只裁 head/Markov/readout 的输出行。GPU 实现常驻
expert union、act_quant、BF16 舍入和 KV append；384 个 MTP expert 继续 pin。
复用既有 GEMV、attention、head 和 Markov pipeline；helper source 由同一
`fuse_dspark.py` 数学模板生成。这里没有 persistent grid、spin wait 或 mega
单 kernel。k=2 由 **81 次提交降到 1 次**，仍有 90 个独立 dispatch。

`DEEPMOE_BATCH_GPU_ROUTE=1` 仅允许在 `resident-only mask` 下使用。
store 锁内复制已发布地址表并给所有 Resident slot 设置 guard；Filling 行不
进入快照。后续 P0 完成的地址只能被下一批看见。共享 expert 继续执行。
GPU 逐层生成 union 与 indirect 参数，计算所有活动行的量化、gate/up、hquant
和 down；miss 权重归零，不重新归一化。40 层、head、rank 共 **一次提交**。
每层参数、RoPE、隐藏状态和 compressor carry 有独立保存空间，参数从录制到
fence 不可变；fence 后按原层序和 union 首次出现顺序执行 planner/LRU/P0。
Engram P2 在 GPU 提交前到位，继续覆盖已有 ≤8 KiB/96-chunk deadline 策略。
不提供与这一机制冲突的实时逐层 CPU probe；专项测试用实际 forward/layer/
submission 计数确认 k=0 恰好一次 target。

CPU 两槽快照测试覆盖 guard、晚到 publication 和 fence 后可驱逐。
GPU 对拍覆盖 M=1/3/6 × disjoint/overlap/reordered × 无 miss/部分 miss/
全 miss，共 **27 组**，act_quant 和 MoE 输出逐位一致，并使用第二层地址表
验证 Vulkan storage offset 对齐。新资源在 allocator/device 关闭前释放。

### 同窗口 turn64：端到端结果（旧动态 LRU）

同一 29-token prompt、seed=41001、T=1/top_p=.95、5500 总槽，native FP4/FP8，
AC 在线、power-saver、单读源。draft profiling 关闭，CM/fold/mega 关闭。
共生成 64 token，其中 decode 为 63 token。每配置一次运行，没有重复 A/B。

| 配置 | tok/s | ms/output | hit | 周期 | 接受/验证 | token/cycle |
|---|---:|---:|---:|---:|---:|---:|
| 普通 mask | 12.329986 | 81.103091 | .8345 | — | — | — |
| serial draft k=2 | 12.430016 | 80.450422 | .7840 | 25 | 38/50 | 2.5200 |
| ONECB draft k=2 | 11.897429 | 84.051773 | .8016 | 28 | 35/55 | 2.2500 |
| ONECB + GPU snapshot route k=2 | **13.491914** | **74.118469** | .7646 | 26 | 37/50 | 2.4231 |

最后一项吞吐相对本窗口普通 mask **+9.42%**，ms/output **−8.61%**。
相对 serial spec 吞吐 +8.54%。输出文本、路由和异步填充完成时机不同，
这是同 workload 的实际对照，不能当逐 token 同输出的算子 A/B。ONECB 单独
降低了草稿时间，却因接受率从 76% 变成 63.6% 而没有带来本格端到端收益。
新快照路径服务率降低，也不能将跳过的计算称作无损提速。

`final64/{comparison.json,check_results.json}` 保存准确数字和热收据，每格
`turns.json`、`provenance.json`、`transcript.md`、`trace.bin` 和 cycles jsonl
保存原始证据。四格均 rc=0、**0 热暂停、0 AC 切换**；GPU/NVMe/CPU 的
最高温为 **68/61.85/67.875°C**。测量二进制 SHA256
`1aeb2426e1f6c7898655aedcbee39dd91311ada5ff4bb4774aa8d39cb3ee8af7`。

### 时间分解：省去了什么，仍在付什么

下面为 diagnostics 的每周期平均；包含首轮单独列出的草稿 seed 工作，
结束余量可出现 k=0，因此不把这些数字当固定 M=3 微基准。

| 项目，ms/cycle | serial k=2 | ONECB k=2 | ONECB + GPU route |
|---|---:|---:|---:|
| draft | 37.163905 | 24.536414 | **23.077655** |
| verify wall | 160.278824 | 161.823334 | **153.736131** |
| commit/rollback | 3.450352 | .991055 | **1.000300** |
| 接受 CPU | .211057 | .197293 | **.183502** |
| 已计量合计 | 201.104138 | 187.548096 | **177.997588** |
| 其他：decode wall 减已计量项 | 1.630926 | 1.568393 | **1.597164** |
| 完整 decode wall / cycle | 202.735064 | 189.116489 | **179.594752** |

其他项是时间残差，未按函数单独剖析；不擅自全部归因于 diagnostics 或 token
loop。按当前 2.4231 output/cycle 和普通 mask 81.1031 ms/output，盈亏平衡约
**196.52 ms/cycle**；原方案的 189 ms 使用的是旧 2.25 output/cycle。
新完整周期约180 ms，依旧处在原方案 175–189 ms 的决策区间。

主模型 trace 的 per-cycle 平均：

| 项目，ms | serial | GPU route |
|---|---:|---:|
| attention busy | 56.753422 | 56.200066 |
| MoE busy | 63.966965 | 63.536584 |
| Engram busy | 5.268651 | 5.130932 |
| CED busy | 1.870431 | 1.829058 |
| tail busy | 11.191728 | 11.171137 |
| 跨提交 gap | 18.738114 | **0** |
| 同提交 gap | 1.665117 | **2.140941** |
| target GPU span | 159.454429 | **140.008718** |

GPU route 的 verify wall 比 span 多 **13.73 ms**，包含 CPU 录制、快照与参数
准备、提交/等待和 fence 后的路由回放；不能把这部分称为 GPU kernel 时间。
attention/MoE 的 busy 基本未变。批中不同 row 的 expert union 仍需读取更多
权重，target 仍是最大项。这组证据支持约 9% 的任务收益，不支持 1.6×。

本格计算的 routed union 为 **9685 个 layer/expert 项 / 26 周期**，即
**372.50 项/cycle、153.73 项/output**。按 native FP4 完整 slot 18,800,640 B
估算，主模型 routed 权重约 **2.890 GB/output**，不含 shared、dense 与
activation traffic，也不是硬件 DRAM counter。正常 P0 填充总量
**32.859 GB / 63 output = .522 GB/output**；mask 没有等待这些填充。

### 逐阶段判断和停止条件

Phase 0 补齐每个 draft op 的 GPU stamp，并保留 host wall。`phase0/` 的
16-token 格为 mask **12.90**、未 profile 的 k2 **11.28**、profile k2 **10.40**、
profile k3 **11.88 tok/s**。k2 draft GPU sum **18.435 ms**、wall **23.975 ms**；
head **10.005**、MoE **3.176**、Q/KV/O **3.674**、Markov/confidence **.956**。
初版 profiler复用 command buffer，改变了 normal path 的分配成本；已改为
与正常 dispatcher 一样 acquire。最终同状态校准 off **29.797** / on
**28.804 ms**，变化 **−3.33%**。这未证明方案要求的 <1%；profile 只作为
显式诊断，关闭它后的 final64 才用于速度结论。不得把其加速算成优化收益。

Phase 1 初版错误沿用了 mega helper 的每次访问可见性 SPIR-V 修饰，ONECB
full5 **32.232 vs serial 26.797 ms**。独立 dispatch 已有 API barrier，不需
这些 mega 修饰；去掉后修正。后续修复只读已写 timestamp 的范围，并在
runtime create 中初始化 ONECB，避免首次 seed 承担约 **210 ms** 的 pipeline
初始化。`phase3_aligned/` 的 profile k2 草稿 wall **23.213**、GPU sum
**22.012**，余量 **1.202 ms**，接近但没有满足严格 +1 ms 目标。

Phase 2 最初按方案止损规则跳过 tile 改动；后续按用户要求实际尝试了 head/投影改动，见 §15。实测三阶段 wq_a 共 **.165 ms**、
wq_b 共 **1.354 ms**；mHC 小算子的全部成本也很小。head 将带宽利用率从
64% 提到 85% 的乐观节省为 `10.005 × (1 − .64/.85) = 2.47 ms`，减半仅
**1.24 ms**，低于 2 ms 继续线。没有做 FP8 head 或 vocab 子集。

Phase 3 完成并达到 **gap ≤5 ms**。第一轮端到端被主动停止，因为多层
快照表的 descriptor offset 未补齐 storage alignment；该格在
`phase3/k2_interrupted.json` 标为无效，不能取速度。补齐 stride、增加 layer-1
对拍并修正资源销毁后，27 组全部通过。`phase3_aligned/` 短格 **12.21 tok/s**，
首次 cold seed 和小样本不作最终收益判断；final64 才采用当前实现。

Phase 4 初步检查确认 **178 ms 属于 175–189 ms** 的区间；后续实际优化尝试和验收见 §15，不进入 Phase 5。
`mgt1_gemv.slang` 已在 weight-outer / batch-inner 的循环内复用一次读取，
使用 128-bit `load16`，M 列在寄存器内累加；WqA/B、WoA/B 已采用 split-K
和 decode 相同的 wave/LDS reduction 选择。不存在“把旧逐列 GEMV 换掉”
这一未实现部分。现格 projection busy 为 wq_a **3.328**、wq_b **17.970**、
wo_a **8.353**、wo_b **11.951 ms**。驱动统计 M=3 split 为 **96 VGPR、
9216 B LDS、0 spill、16 subgroups/SIMD 的寄存器上限**，不是运行时活动
occupancy 测量。完整统计在 `quality_complete/pipeline_stats/index.tsv`。
方案的 attention ≤45 ms 未达到；不重开已经质量失败的 CM、scale-fold，
也不把既有 split-K/weight-once 再计为新收益。

Phase 5 本次按决策点不进入。union WMMA / 按 M champion / mega / host-flag
均未重开。普通 decode 的单提交机制也单独检查：本格结束余量的 GPU-route
M=1 verify **113.08 ms**，普通 mask 平均 **81.10 ms/output**，工作状态不同，
不能当精确 A/B，但足以说明直接调用旧 batch M=1 不能作为 decode 加速实现。
普通 decode 保留专用 attention/MoE 路径。

Phase 6 新工具 `tools/dspark_cost_fit.py` 用优化后的实际 verify 成本拟合
confidence-prefix 策略，包括 k=0。M=1 只有一次末尾样本，M=3 有 25 次，
拟合每多一行约 **21.143 ms**；M=2 是插值，没有实测校准。k=0 仍支付已经
执行的 draft，首次拒绝后不继续读后面行。训练轨迹最佳阈值 **−2**，只
把一轮 k=2 裁成 k=1，估算相对固定 k=2 **+0.47%**，低于抖动范围，更
没有 held-out 验证。`final64/confidence_fit.json` 保存全表和限制；保持固定
k=2，不默认启用动态阈值，不推测 k=3 的收益。

### 质量验收与启用方式

| 门禁 | 当前结果 | 收据 |
|---|---|---|
| CPU / 工具 | **25/25；30/30** | `final_cpu.log` / `final_tools.log` |
| 原生 MoE union、量化对拍 | **27/27 逐位一致** | `phase3_aligned/gpu_unit.log` |
| DSpark golden / 前缀 / KV wrap | **逐位一致**，沿用原 near-tie 规则 | `quality_complete/profile_calibration.log` |
| committed-prefix 窗口回滚 | **通过** | `quality/wrap.log` |
| k=0 只调用一次 target | **通过：40 层、1 提交** | `quality_fixed/k0.log` |
| 63-position batch mask | **cos .9537048，top-1 56/63，PPL 比 1.0072** | `quality_fixed/batch63.log` |
| off64 平台基准 | **NLL .622784，top-1 56/64** | `quality_complete/l3_off.json` |
| 普通 decode | **通过现有两用例门槛**；短 oracle teacher/free **6/8**、自身 prefill **7/8**，方案的严格 8/8 未复现，数值与平台基线一致，未改阈值 | `quality_complete/decode.log` |
| 4K / 17K decode | **free / teacher-forced 均 8/8，两用例通过** | `quality_complete/longctx.log` |
| 生成式 MMLU57（旧动态 LRU） | **48/57，2 个无效答案计错；67 轮、128/130 接受/验证** | `quality_complete/mmlu57/summary.json` |

新开关默认 **off**。镜像随后恢复只读挂载，48/48 shard 的文件大小匹配；
双盘三格已完成，ready 为 sources=2，status 和盘计数均证明两个源参与读取。
固定专家 cache、双盘成本和后续 Phase 4 结果见 §15。网页保持停止。

复现单盘速度（先停 web，AC/power-saver，并记录温控；profiling 保持关闭）：

```bash
DEEPMOE_MODEL_DIR="$HOME/models/DeepSeek-V4.1-Flash" \
DEEPMOE_MIRROR_AUTO=0 DEEPMOE_DSPARK_MEGA=0 \
DEEPMOE_DSPARK_ONECB=1 DEEPMOE_BATCH_GPU_ROUTE=1 \
DEEPMOE_DSPARK_PROFILE=0 DEEPMOE_MGT_ATTN_CM=0 DEEPMOE_MGT_FOLD_SCALE=0 \
DEEPMOE_MASK_DYNAMIC_LRU=1 \
python tools/hitrate_bench.py --script bench/results/draft_attribution/turn64.json \
  --max-turns 1 --exe build/deepmoe --cache-slots 5500 --require-sources 1 \
  --out bench/results/spec_e2e/local --serve-arg=--resident-only --serve-arg=mask \
  --serve-arg=--dspark --serve-arg=--spec-k --serve-arg=2 --serve-arg=--spec-top-k --serve-arg=4

python tools/dspark_cost_fit.py bench/results/spec_e2e/final64/gpu_route_k2_cycles.jsonl \
  --plain-ms-per-token 81.103091 --out bench/results/spec_e2e/final64/confidence_fit.json
```

测量 wrapper 的完整 argv/env、thermal、AC 和 power profile 恢复收据在各阶段
`jobs.json` / `check_results.json` / `*_thermal.jsonl` / `profile_receipt.json`。
mask 和 top-K 接受改变输出分布；本次不宣称长上下文无损或修复所有内容退化。

## 15. 双盘验收、固定初始 cache 与实际 Phase 4 尝试（2026-10-05）

本节覆盖用户追加的五项任务。§14 的速度和 MMLU 均为旧动态 LRU；不能替代固定 cache 的质量结论。网页始终停止，首页图片没有刷新网页后重拍。

### 同接受率的周期成本

固定采用 **2.25 output/cycle** 进行成本换算。这只是消除接受率差异的算术对照，路由/union 仍可能不同，不能称作相同 kernel workload。

| 读源 / 配置 | 完整 cycle，ms | 按 2.25 output/cycle 的 ms/output |
|---|---:|---:|
| 单盘 serial k2 | 202.735064 | 90.104473 |
| 单盘 ONECB k2 | 189.116489 | 84.051773 |
| 单盘 ONECB + GPU route | 179.594752 | 79.819890 |
| 双盘 serial k2 | 211.316586 | 93.918483 |
| 双盘 ONECB + GPU route | 211.771057 | 94.120470 |

单盘相同接受率下 cycle 成本下降 **11.4%**。双盘这格 cycle 成本没有下降（+0.22%）；实际 tok/s 的改善来自 2.52→2.625 output/cycle，不能归为计算加速。

双盘 `final_dual/` 三格分别为普通 mask **12.06**、serial k2 **11.93**、ONECB + GPU route **12.40 tok/s**。每格一次、64 token、5500 总槽、AC/power-saver、0 暂停。镜像只读且 **holds 48 of 48**，ready `sources=2`。组合格两个盘实际读取 **60,368.2 / 50,818.3 MB**，不是只登记了 mirror。

### 查清 verify 的 host 成本

新增 `verify_host_ms`，将 setup/snapshot、输入与 Engram 发起、层/tail 录制、submit/fence、验证后路由、trace/readout 分开；嵌套子项不能重复相加。`phase4_controls/dynamic_host_trace/` 的首轮层/tail 录制 **468.274 ms**，是新 SPIR-V 的延迟 pipeline 创建；后续平均 **7.100 ms**（含 Engram 等待 **5.477 ms**）。验证后路由平均 **5.206 ms**，其中 hidden/carry 回读 **4.250 ms**，planner **.636 ms**，地址/索引检查 **.040 ms**。snapshot **.277 ms**、其他输入准备 **.415 ms**、trace/readout **.166 ms**。这说明残差不属于单一 GPU 算子。

验证流水线改为生成前准备；37/38/39 层 hidden 的 BF16 mean 改在 GPU 上按相同加法顺序计算，回读量缩为四分之一。独立 GPU 对拍逐位一致，k=0 一次 target 和 committed-prefix 窗口回滚均通过。GPU 路由仍是每轮一次主路径前向，一份验证矩阵。

新增 P0 reserve / submit / IO 与 failed-fill 计数。上述动态 trace **四类均为 0**，其完整 IO status 同样是 0 failed。小缓存测试的 ResourceExhausted 不再混为磁盘失败。`set_streams`、`forward_batch` 和 serve 参数解析均拒绝 GPU route + streams>1；serve 在加载模型前返回错误，CPU 专项测试覆盖拒绝后 stream 数和 forward 计数未变化。

### 固定 cache 的实现边界

mask 先用静态热表填满空槽，然后锁住初始专家。启用 DSpark 时保留 384 个 MTP pin，主模型用剩余 5116 槽。store 不更新 LRU/heat，不允许淘汰；满槽后的 decode miss 不再提交 P0/P3；精确 prefill 的 transit 读盘仍可能用 P0，但不进入 cache。KV/session 重置保留这一专家集合。prefill 仍精确计算：resident 直接复用，miss 走临时 transit，禁止用 prefill 替换固定槽。显式退出 mask 才恢复普通缓存策略。

初版 serve 先 begin_session 后应用 mask，短格只达到 3979/4494 resident，未形成满槽 cache；这两格在 `phase4_controls/partial_cache_verdict.json` 标为不能用于固定 cache 速度判断，并取消待跑的同类 pair 格。已将 CLI mode 提前到 session 前，并要求初始填充全部 settle；部分加载则拒绝开始 session。旧动态行为仅供历史复现用 `DEEPMOE_MASK_DYNAMIC_LRU=1`，默认 mask 不启用它。

### Phase 4 和 draft head 的实际尝试

已实现 `DEEPMOE_MGT_PAIR_DOT=1`：保留 128-bit 读取和 weight-once，将一次保留 32 个解码浮点权重改为逐对解码并供 M 列共同累加。每列加法顺序相同，FP8 量化、归约和 head 精度不变。15 个 projection 对拍覆盖 M=1/3/6 和 Q/KV/out 五种形状，全部逐位一致；DSpark golden 全链、所有输出前缀和 KV wrap 也逐位一致。

驱动统计 M=3 projection **96 VGPR / 9216 B LDS / 0 spill**，head **96 VGPR / 7168 B LDS / 0 spill**，前后寄存器占用相同。实际固定满槽的成本对照和启用判断在后续收据中；不将源码写法变化视作收益。CM、scale-fold、mega、union WMMA、champion 和 MTP unpin 的原停止结论保持不变。

### 提交和 main 的处理

Phase 1：`81826bd`；Phase 3：`c796e56`，是两个独立 commit。`7955423` 是共同依赖的不可变参数/indirect 基础设施，`faf20a4` 是专项验证。固定 cache、host 计时、提前准备与 GPU hidden mean 在 `dc5d870`。

`66d0f7a` 是以 `200dc3f` 为父节点的已有工作快照，包含本轮开始前的 mask/DSpark 及文档。它不是可丢弃的测试提交。本机 main 的内容经核对与快照一致后，先保存 index/patch/ref 收据，再对齐快照并 fast-forward 整条链；已合入 main。其他 main 若仍在 `200dc3f` 且没有这些内容，先合快照/整条链；若已用其他 commit 纳入同一内容，应核对后仅 cherry-pick `2cd38a3` 起的优化提交，不能重复 cherry-pick `66d0f7a`。合回记录在 `integration_backup/`。

### 固定满槽成本与 Phase 4 判定

`fixed_final/` 的三格均为双盘、5500 槽、64 token、power-saver/AC、0 温控暂停。ready 前 5500 槽全部 settle。普通 mask **18.091544 tok/s**；投机两格 profiling 同时开启：

| 项目 | pair-dot off | pair-dot on |
|---|---:|---:|
| tok/s | 18.483370 | 18.780369 |
| 完整 cycle，ms | 121.731049 | 119.805954 |
| output/cycle | 2.25 | 2.25 |
| 接受/验证 | 35/54 | 35/54 |
| Q/out 投影合计，ms/cycle | 36.791031 | 35.745392 |
| wq_a / wq_b | 2.955411 / 16.033172 | 2.877120 / 15.505799 |
| wo_a / wo_b | 7.471029 / 10.331419 | 7.312811 / 10.049662 |
| draft head，ms/cycle | 8.403404 | 8.319407 |
| target attention，ms/cycle | 48.631910 | 47.375583 |

全部 tokens、target ranks、draft/target routes、接受数均一致；这次是同接受率、同输出和同 union 的实际对照。cycle 仅少 **1.925095 ms**，按规则减半为 **.962547 ms < 2 ms**；projection 自身只少 **1.045639 ms**、head **.083997 ms**。**NO-GO：pair-dot 保持 0，不做更多整轮 A/B。** attention ≤45 ms 仍未达到。

提前准备 + GPU hidden mean 后的固定路径：host setup **.278 ms**、输入准备 **.334 ms**、层/tail 录制 **3.608 ms**（其中 Engram issue/land **2.092 ms**）、验证后路由 **.604 ms**、trace/readout **.175 ms**。hidden/carry 部分 **.034 ms**；trace span **92.750 ms**、verify wall **98.307 ms**，残差 **5.557 ms**。这是固定 cache 的新状态，不能把相对动态格的全部差额归为代码优化；逐位相同的 GPU mean 对拍单独证明数学不变。

两格的 cache_fixed/cache_frozen 全为 true、cache 5500 resident/0 filling/0 free、evictions=0，全部 cycle miss_bytes=0，P0 reserve/submit/IO 和 failed-fill 均为0。prefill transit 仍有读盘，不能把其 P0 字节混为 cache 换入。`fixed_final/comparison.json` 保存各字段与同路径核对；`projection_comparison.json` 保存逐算子结果。

k=2 的常规验证矩阵是3行；每行每层选6个 routed 专家，单层原始 union 最多18个，shared另算。mask 后只对驻留 union 读取权重，各行复用同一份专家权重，跨层仍需读取不同权重。固定格合计 **3539** 个驻留 layer/expert 项，平均 **126.393 项/cycle（3.160 项/层）**。按每项 **18,800,640 B** 计，routed 权重逻辑量为 **2.376 GB/cycle、1.056 GB/output**。这不含 shared/dense/激活，也不是 DRAM 硬件计数；0 miss_bytes 仅说明 decode 不换入专家，不表示无访存。收据 `fixed_final/memory_account.json`。

固定专家集合的质量已出现明显退化：普通 mask route served 约 **37.5%**，投机约 **26.7%**；输出有反复“霓”字/重复句。上述速度是这种近似模型的硬件测量，**不能作为可用质量下的加速结论**。MMLU 使用相同固定初始化单独重跑，不能拿旧动态 LRU 的 48/57 代替。

Phase 6 在该轨迹上重算：最优观察策略仍是固定 k=2（阈值 −inf），动态阈值估计净收益为0，继续不设默认。固定 cache 的121ms周期伴随严重服务率下降；它不能代替 Phase1–3 的原动态 LRU 180ms条件去触发 Phase5。更改 union 数学前先需要可用的质量门，故本轮不进入 WMMA。

### 最终固定 cache 质量门

固定 cache 的最终 MMLU57 为 **48/57 = 84.2105%**，**1** 个格式无效计错。全部57题都有投机参与，共 **60** 个周期，**115/120** 个草稿接受/验证。5500槽、双盘 `sources=2`、k=2/accept top-4、ONECB/GPU route=1、pair-dot/CM/fold/mega/profiling=0。协议与§14相同：每科一题、零样本、精确前缀、最后prompt token单位置处理、最多生成16个token、T=0/seed42。不能称为完整标准五样本 MMLU。

该分数满足用户的 **≥48/57** 条件，但它不是唯一的启用条件。普通短 oracle 的严格8/8没有复现，固定中文 turn64 也有重复输出；因此 **ONECB/GPU route 仍默认0，mask仍显式近似模式**。不把旧动态LRU的48/57当作固定缓存的证明；这次是独立重跑的收据。

最终源码 `dc5d870` 的 off64 仍为 **NLL .622784 / top1 56/64**；batch63 为 **min cosine .9537048 / top1 55/63 / PPL ratio 1.0053**。新 mean GPU对拍、27组union、k=0单前向和窗口回滚均通过。这里的普通精确4K/17K TF/free8/8来自 `quality_complete/longctx.log`；固定mask长上下文另列，不能混用。

`fixed_quality/mmlu57/summary.json`、`mask-spec/answers.jsonl` 保存全部答案和计分。样本 SHA256 **74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7**；所有诊断周期的fixed/frozen均true，四类load failures均0。三个最终质量作业均rc0、双盘48/48、AC保持在线、0温控暂停；峰值GPU **72°C**、内盘 **62.85°C**、外盘 **74.85°C**。MMLU作业827.775s包含启动等耗时，不用它算decode速度。最后恢复原performance模式。

### 固定 mask 的长上下文端到端检查

新增独立 raw prompt 检查，实际经过精确 GPU prefill、固定 mask、ONECB draft 和 GPU route 验证。两个session分别为 **4133 / 17010** token，T=0/seed42、k=2/top-4，各生成9个token。总cache用 **4900** 槽，为17K prefill保留工作区；不是5500槽速度格。

| prompt | 输出 | 与参考相同的连续前缀 | 投机接受/验证 | 精确 prefill |
|---|---|---:|---:|---:|
| 4K | `kestrel-4471-amber"` | **9/9** | **5/5** | 28.211 s |
| 17K | `kestrel-4471-amber"` | **9/9** | **5/5** | 52.233 s |

cache全程fixed/frozen，最终4900 resident/0 filling/0 free、384pin、0evictions、0failed-fill；六个投机周期miss_bytes与四类load failures均0。双盘48/48、AC在线、0温控暂停，GPU峰值71°C、外盘74.85°C；结束恢复performance。原始输出/IDs/引用IDs/温控见 `fixed_longctx/{quality.json,needle4k17k/events.jsonl,check_results.json}`。两次检索通过不证明任意长对话或mask数学无损，也不替代普通teacher-forced质量门。

### 最终关闭 profiling 的速度格与交付

`final_fixed_speed/` 是独立配置：pair-dot/CM/fold/mega/draft profiling全部0、ONECB/GPU route=1，双盘5500槽、AC/power-saver、64个输出。最终 **18.492909 tok/s、54.074781 ms/output、121.668256 ms/cycle**；draft **20.817178**、verify **98.694461**、commit **1.014235**、CPU **.181888 ms/cycle**。28周期、35/54草稿接受、2.25output/cycle、82target rows、3539resident union项，与profiling-on对照的全部tokens/ranks/draft+target routes逐项相同。

相对固定普通mask **18.091544** 的观察差为 **+2.22%**，在±3%抖动内；普通格输出/路由不同，不能隔离投机收益。没有达到1.6×，也不据此启用默认GPU路由。该格verify trace跨提交gap **0**、同提交gap **1.898 ms**、span **93.173 ms**，wall/span残差 **5.521 ms**；attention **48.980**、MoE **26.695**、Engram **4.423**、CED **1.648**、tail **9.529 ms/cycle**。four load-failure counters、cache evictions、decode miss_bytes全0，0温控暂停，GPU最高66°C、外盘74.85°C，AC稳定；结束恢复performance。源码与最终质量格均为 `dc5d870`。

主目录已用自己的toolchain/shader路径编译；52个SPIR-V与被测工作树逐文件hash一致。主目录CPU **25/25**、工具 **30/30**、`--streams 2` GPU-route提前拒绝均通过。工作树被测exe SHA256 **62a42a90a128e4dd0de9c0a47b93529349d0cc8e9aa68aa667c484164f879942**；主目录exe **9e4d44c75ccc6dcdcbb48fcefa808f788c8aa880df7756f30976074c84054542**，路径嵌入不同，未声称exe hash相同。`integration_backup/final_provenance.json`保留逐shader hash与主目录编译收据。

精简机器收据为 [`dspark_e2e_receipt.json`](dspark_e2e_receipt.json)，完整原始收据归档在主目录 `bench/results/spec_e2e/`。日志记录原始执行路径；清理工作树后，以主目录归档位置读取。Phase1/3分开提交；文档随整条链合入main。网页保持停止、浏览器没有刷新，系统回到原performance模式。

## 16. 网页启用原生 k=5 与思考 effort（2026-10-05）

用户先要求尝试16，随后改为5；没有扩展模型原生五位置草稿块。此前网页
`draft_tokens=5` 只是默认值，实际 `enabled=false`。新增 server 的显式投机参数，
重启后 READY 为 `enabled=true / draft_tokens=5 / accept_top_k=4 / main_paths=1`。
ONECB/GPU route=1，profiling/pair-dot/CM/fold/mega=0，固定5500槽、384 MTP pin、
双盘 sources=2、power-saver，温控仍80°C暂停/72°C恢复。网页显示实际投机状态。

一次独立冷会话短测，提示词 `简短解释 GPU 的作用。`（10 tokens），T=0、seed42、
生成32 tokens（首个来自prefill，decode输出31）。实测 **15.2335805 tok/s**；
**10 cycles、10 verify submits、57 target rows、21/47 草稿接受、3.1 decode outputs/cycle**。
满周期将 root 和唯一五位置主路径合成六行验证，最后一轮按剩余输出长度缩短。
没有多树、多次验证前向。所有decode专家miss_bytes为0；固定cache保持冻结。

每周期 draft **23.3931 ms**、verify **177.7626 ms**、commit **1.2334 ms**、
CPU **.0933 ms**，合计 **202.4824 ms**。更长的验证矩阵仍需要更多计算，且被拒绝的
后缀不计入输出；按本次3.1输出摊销约65.32ms/output。前一节k=2的提示词和输出
轨迹不同，不能把15.23与18.49直接当作配对加速比。这是功能短测，不是标准吞吐基准。
GPU峰值62°C、外盘74.85°C，0温控暂停。

思考 effort 使用checkpoint自身renderer，支持低50、高75、最大100与自定义1–100整数；
不思考时不渲染预算前缀。预览和发送使用同一预算，随会话保存；预算改变可能使前缀
不可复用并重新prefill。原默认75的思考prompt逐字一致；真实renderer预算1/50/75/100、
API保存50/100/33与拒绝101、旧会话兼容、浏览器自定义输入均通过。CPU/工具门禁31/31。

机器收据：[`dspark_web_receipt.json`](dspark_web_receipt.json)；原始events、READY、done、
状态和派生统计：主目录 `bench/results/web_spec5/`。网页保持运行，新页面含控件，
旧页面没有刷新。恢复历史消息不表示重启后的GPU KV仍驻留。

## 17. k=5 网页长测：吞吐提升，重复质量未通过（2026-10-05）

按用户“长测、提交好”要求，各配置只运行一次。直接使用web API和checkpoint原生
renderer：中文思考high75、同会话中文续轮、重置后的英文chat，各生成512 token。
T=1、top_p=.95、seed51001/51002/51003，固定5500槽、双盘48/48、AC/power-saver、
max_context4096、精确GPU prefill minimum16、无磁盘KV。k5/top4显式启用ONECB和GPU
route；plain无384个MTP pin，spec的384pin占用总5500槽。每个配置输出1536 token，
其中1533个为decode输出。输入最多675 token，实际context最多1186；这是长生成和
多轮测试，不替代4K/17K输入质量门。

**本次不启用draft GPU时间戳、per-op trace、route dump、cycle diagnostics或engine
profile。** 同一引擎exe和shader，完整构建信息、环境和启动参数保存在机器收据。
网页先停止；测试只有一个GPU作业，结束后恢复k5网页与原温控80°C暂停/72°C恢复。

| 工作负载 | plain decode tok/s | k5 decode tok/s | k5输出/周期 | k5周期ms |
|---|---:|---:|---:|---:|
| 中文思考512 | 13.759750 | 17.786256 | 4.7315 | 264.8210 |
| 中文续轮512 | 13.527413 | 16.816002 | 4.7315 | 280.2592 |
| 英文chat512 | 13.307152 | 14.543124 | 4.2231 | 289.2599 |
| 加权decode总计 | **13.528916** | **16.264449** | **4.5490** | **278.5433** |

加权decode按总decode_steps/总decode_ms计算，观察提升 **20.2199%**。包含prefill的
三轮输出速度为 **10.772035 / 11.879256 tok/s，+10.2787%**；总prefill分别
**29.278 / 35.046 s**。含启动、HTTP驱动与关闭的完整进程作业耗时
**184.632 / 172.173 s**，观察比率1.0724。不要将sidebar的decode速度当成全流程速度。

续轮的web start事件均只匹配到99个prefix token，而完整prompt是675；引擎选择整段精确
GPU prefill，最终reused_tokens=0。plain/spec该轮prefill分别 **13.278 / 17.194 s**。
未闭合的截断思考输出经官方renderer重新编码，不能假设全部已生成文本都复用KV。

投机共 **337 cycles、337 target forwards/verify submits、2013 target rows**；
**1196/1676 = 71.3604%** 草稿接受，输出 **1533/337 = 4.5490/cycle**。
每轮仍只有一条主路径，完整轮root加五个草稿为六行一次前向，尾轮按剩余长度缩短。
context计数逐轮满足prompt+generated−1，token流数量、范围、时间单调与数值有限
检查均通过；每轮readout cycles与cycles相同，未发现多次target提交。

周期成本为draft **27.1986**、verify **248.8669**、commit **2.1115**、CPU
**.3664 ms**，合计 **278.5433 ms/cycle**。按当前成本摊销约61.23ms/output；
plain实测约73.92ms/output。固定当前成本，要达到1.6×约需 **6.03个输出/周期**，
已经接近六行全部接受的极限；这只是成本敏感性计算，未来轨迹和成本不会保持不变。
不能把新接受率计作kernel降本，也不能据此承诺1.6×。

**质量未通过。** 两种配置的持续输出都出现重复；spec英文最后128个token为精确
period2，IDs交替 **1805/982**，文本为 `the *the *the *...`。英文spec首/末四分段
的有效速度 **12.99→20.28 tok/s**，尾段加速与重复退化同时出现。中文spec首轮
重复四元token片段比例 **39.88%**（plain **16.50%**）；此指标不能替代语义评分。
最后累计gate mass lost plain/spec为 **.3365/.3729**，两条轨迹和常驻专家不同，
不能仅凭本次确定固定miss-mask与top-K接受各自的质量责任。未重跑MMLU，也不声称
通过长对话质量或无损门。ONECB/GPU route默认仍为0；网页k5保持用户显式实验配置。

两配置都0温控暂停、AC未变化、48/48镜像健康；GPU峰值plain/spec为 **72/75°C**，
外盘峰值均 **74.85°C**。全部逐轮status为cache_fixed/cache_frozen=true、0evictions、
0failed-fill、P0 reserve/submit/IO失败均0，decode专家miss_bytes为0。

驱动及输入：[`web_longtest.py`](../bench/web_longtest.py)、
[`web_longtest_prompts.json`](../bench/web_longtest_prompts.json)；读取已有结果：
[`web_longtest_report.py`](../bench/web_longtest_report.py)。驱动需要先停止其他engine并
通过温控监督执行。机器收据：[`dspark_longtest_receipt.json`](dspark_longtest_receipt.json)。
原始逐token events、完整输出、status、provenance、温控与监督计划归档到主目录
`bench/results/web_spec5_long/`。执行时的工作树路径是历史记录，读取时用主目录归档。

## 18. 动态 LRU 恢复与自动冻结离线判定（2026-10-05）

本次按用户指定的 mass EWMA / 32-token 预热 / 16-token 冻结 / <1% churn
规则扫描 .93/.88、.95/.90、.97/.92。原生 8 轮 demand-LRU 回放的 2318
个 decode token 没有通过 churn 门槛的点；有权重的 mixed 参考 trace 冻结
占比在 5500 槽为 2.697～2.748%，5116 主模型槽近似为 .774～.788%。
这些是回放，缺少 GPU prefill、异步 IO 与输出反馈，不能作为运行时实测比例。
按“占比太低就放弃”规则，自动冻结未接入运行时；mask 默认恢复动态 LRU。
固定初始集合只保留显式 `--mask-cache fixed`，`--mask-cache dynamic` 为默认。

新动态 k5 双盘 8 轮：**105.022 ms/decode token，9.521827 tok/s**；
2361 输出，2353 decode step，**594 cycle / 594 target submit**，1764/2953
草稿接受。平均 draft / verify / commit / CPU 是
**33.199 / 379.451 / 1.834 / .197 ms/cycle**。0 source/load failure，0 热暂停，
48/48 镜像健康；8 轮没有末 128 token 的精确短周期循环。

动态 k5 的零样本生成 MMLU57 是 **48/57，2 invalid**，达到样本门槛。
off l3 仍 **.622784**；冷启动动态 mask l3 是 **1.360084**，未接近历史
.835581。因此不能将本次标为整体质量通过，也不把旧固定 cache 的 18 tok/s
作为可用质量速度。没有自动冻结对照格；GPU route 与 ONECB 在全局仍显式启用。

具体输入、控制器边界测试、原始数据限制和决策见
[`mask_freeze.md`](mask_freeze.md)、[`mask_freeze_offline.json`](mask_freeze_offline.json)。
raw 归档 `bench/results/adaptive_mask/`，工作树路径只表示执行时来源。

## 19. 动态 mask 的异步等待核对（2026-10-05）

同 prompt 的 32 输出短 trace：普通动态 mask 12.595869 tok/s，k5/top4
13.398560 tok/s，9 cycle / 9 target submit，22/41 草稿接受。k5 平均 verify
228.715 ms，target GPU span 218.896 ms，attention / MoE busy 86.515 /
106.808 ms；跨 submit gap 0。fence 后 finish_routes 1.312 ms，其中规划／
异步 P0 提交 .863 ms，没有等待专家 miss 完成。此为有 trace 的短测，
不能将 +6.37% 推广为 8 轮加速，也不是逐 token 同输出的算子对拍。

旧 planner 在出现 miss 时加 `stalls`，并没有测量 IO join。`45ee3ab`
将 miss layers 和实际 wait_layer 分开，diagnostics 输出每个 target 的
`expert_io_wait`；16-token 验证的 4 个 cycle 全部 0 calls / 0 miss joins /
0 ms，启动 MTP 的 3 次 exact wait 另计。异步策略本身已经存在，这次只
修正计数，没有新增 kernel 加速。长测9.52的上下文、union及接受率边界
与固定cache18不同，说明与限制见 [报告](mask_async.md) 和
[收据](mask_async_receipt.json)。

## 20. D：同一引擎五臂长测，重复质量未通过（2026-10-06）

**没有合格赢家。** 五臂都完整跑完8轮、无短周期循环、无加载失败，但都未通过
与 C/off 匹配的严格四元重复门。保留 owner 指定的临时 **k=2、ONECB=1** 基线；
它没有因此获得最终默认资格。GPU route 的独立判定为 **NO-GO，采用 CPU route**：
配对 k2 中 GPU 的 raw/active 成本分别高 **9.69% / 25.25%**，未满足二者都降低
超过3%的条件。**默认候选必须 ONECB=1**；串行臂只用于比较。完整质量结果仍未补齐。

来源：[最终报告](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/phase_d/final_report_r1.json)
（SHA256 `99e8c66f6413803412b45a07f97829ccb9783ad0ad3183bba1c6f4f90ae751fd`）、
[实际比较记录](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/phase_d/same_engine_fresh/arms/comparison.json)、
[监督收据](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/phase_d/same_engine_fresh/check_results.json)。
冻结时钟版引擎 `b07f2480…`、52个shader `185109f0…`；同一个 PID **3701458**、
default session，按下表顺序运行，臂间重置KV并保留专家缓存。双盘实际48/48、
动态5500槽、AC在线、performance/platform performance、80/72°C温控；8轮输入、
T=1/top-p=.95/seed41001～41008一致。top-K=4、主路径=1，正式长测不开 profiling
或 per-op trace。监督 rc0，作业 wall **2137.537 s**，扣CPU暂停估算为
**1271.703 s**，暂停 **865.834 s**；这些作业时间包含启动与prefill。

| 臂 | ONECB / route | raw ms/decode输出 | active估算 ms/decode输出 | 命中率 | 接受/验证（比例） |
|---|---|---:|---:|---:|---:|
| k2_gpu | 1 / GPU | 145.491 | 85.091 | .90282 | 1369/1636（.83680） |
| k2_onecb_cpu_route | 1 / CPU | 132.634 | 67.936 | .91835 | 1344/1606（.83686） |
| k2_serial | 0 / CPU | 156.207 | 84.394 | .91536 | 1441/1728（.83391） |
| k3_best | 1 / CPU | 161.270 | 74.258 | .90664 | 1543/2007（.76881） |
| k5_best | 1 / CPU | 167.004 | 82.089 | .90149 | 1733/2943（.58885） |

以下均为 **ms/cycle**。stage是原始wall计时，暂停可能发生在任一stage内，不能
逐stage扣除或把raw draft劣化全归为kernel变慢。active只扣记录到的CPU暂停，
已经提交的GPU工作仍可继续；它不是GPU时间戳。未覆盖项为完整cycle减四stage。

| 臂 | cycles | raw cycle | active估算 cycle | draft | verify | commit | CPU | 未覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k2_gpu | 819 | 387.975 | 226.910 | 50.915 | 333.812 | 1.980 | .192 | 1.076 |
| k2_onecb_cpu_route | 804 | 353.856 | 181.248 | 42.171 | 308.937 | 1.487 | .269 | .991 |
| k2_serial | 866 | 415.230 | 224.337 | 65.472 | 339.216 | 8.971 | .379 | 1.192 |
| k3_best | 670 | 531.470 | 244.717 | 51.080 | 476.003 | 1.169 | .135 | 3.083 |
| k5_best | 592 | 654.759 | 321.839 | 46.763 | 603.003 | 2.429 | .178 | 2.385 |

| 臂 | resident routed union项/cycle | miss GB/cycle | 固定2.25输出：raw / active ms/输出 | 接受/验证=.60：raw / active ms/输出 |
|---|---:|---:|---:|---:|
| k2_gpu | 465.093 | .847982 | 172.433 / 100.849 | 176.470 / 103.210 |
| k2_onecb_cpu_route | 472.189 | .786836 | 157.269 / 80.555 | 160.953 / 82.441 |
| k2_serial | 471.939 | .816990 | 184.547 / 99.705 | 188.979 / 102.100 |
| k3_best | 571.363 | 1.084091 | 236.209 / 108.763 | 189.993 / 87.483 |
| k5_best | 746.529 | 1.464515 | 291.004 / 143.039 | 164.398 / 80.808 |

union按层累加；miss是planner计数（GB=10⁹字节），不是硬件访存或完成IO计数。
两种归一化都复用实测cycle成本；.60是总接受/总验证比例，分母为
`1 + .60 × 实测验证数/cycle`，未加独立逐位置接受假设，也未模拟新路由。
保留实测停止边界修正后，.60的raw/active成本依次为 **176.863/103.439、
161.226/82.582、189.477/102.369、190.501/87.717、164.677/80.945 ms/输出**。
这说明扩大k的收益依赖接受率；不能把观察到的更高输出/cycle当作算子降本。

严格门逐轮要求重复四元片段占比≤同label/seed的 C/off 的1.5倍；off为0时只容0。
下表保留全部失败行，数值为 **候选百分比 / 允许上限百分比**；轮号从1起。

| 臂 | 未通过的轮与占比/上限 |
|---|---|
| k2_gpu | 第4轮 en follow-up：.3584 / 0 |
| k2_onecb_cpu_route | 第4轮 en follow-up：.3571 / 0 |
| k2_serial | 第3轮 en explain：2.0958 / 0 |
| k3_best | 第2轮 zh follow-up：11.9048 / 2.1739；第5轮 code：21.7002 / 15.1478 |
| k5_best | 第3轮 en explain：.3165 / 0；第4轮 en follow-up：1.6835 / 0；第5轮 code：19.4631 / 15.1478 |

40个输出的 `no_loop=true` 与上述严格门失败可以同时成立。五臂的 failed-fill、
IO failed、P0 reserve/submit/IO 失败 **before、after与差值全0**，sources无DROPPED；
这些加载结果已验证。未生成质量赢家收据；本轮没有新增l3、MMLU57或3×512质量结论。

D的384个MTP pin包含在总5500槽内，target可用 **5116**；C/plain无MTP pin。
C动态mask参考为 raw/active **179.840 / 80.264 ms/decode输出**，跨 C/D 的
速度或命中率差无法单独归因于投机或pin成本。D固定顺序、缓存续用与输出轨迹变化
也限制归因。新的 GPU route trace / single-target-call 诊断已完成，见§21。
本节stage wall值保持原口径，不据此补推draft kernel耗时。


## 21. 新双盘 k2 verify trace：host账与异步等待（2026-10-06）

冻结b07引擎，同PID3751491/default session、相同中文prompt/seed；ONECB1，route1→0，
各32输出。GPU route开始时只有384个MTP resident、5116个free，CPU组继承已升温cache。
这组用于拆账；接受路径、union、miss量和温控不同，不作路由kernel的因果A/B或新速度排名。

| 每cycle均值 | GPU route | CPU route |
|---|---:|---:|
| cycles / target calls | 15 / 15 | 14 / 14 |
| queue submits / target | 1 | 41 |
| draft host wall ms | 21.901 | 23.362 |
| verify wall ms | 301.300 | 183.295 |
| target GPU span ms | 106.571 | 173.257 |
| wall减GPU span ms | 194.730 | 10.039 |
| host未覆盖残差 ms | .002337 | .002613 |

CPU组约10ms已对上：record+final fence减完整GPU span **9.576974ms**，其它顶层host区间
**.459017ms**，残差 **.002613ms**。这是跨41次submit的区间差，span包含submit间空隙；
不能把它解释成一个额外10ms kernel，也无法用未记录的每cycle绝对时钟精确扣除冷却。

GPU组span外194.729531ms主要是 **Engram issue/landing 190.660680ms**；其它input/record
1.650867、submit/fence减span1.053285、setup .307086、route finish .910896、trace .144381、
残差 .002337ms。Engram桶含发起异步读取、等待数据落地及录制，是主模型Engram依赖，
不是MoE专家阻塞，也不是纯SSD设备时间。六个顶层host桶互斥；Engram、queue/fence、
route细项嵌套在父桶内，不可重复相加。这也不能证明旧+13ms全部来自同一因素。

29个target region/28063 records逐条与combined文件顺序匹配，每region40层+head、flags0；
**每cycle恰好一个主路径target forward**，queue submit数与forward数不同。
专家wait calls/joins/ms及加载失败before/after均0；mask的专家miss仍异步，Engram保留精确依赖。
监督rc0，55.861s wall/55.129 active/.733 cooling、9次暂停；GPU最高80°C，NVMe49.85/74.85°C，
AC1/performance稳定。CPU组decode重叠冷却663.847ms，所有stage表保留raw；不逐stage扣暂停。
Draft GPU profiling关闭，上表是host wall（CPU正k均值25.159ms），不能与旧31ms kernel地板等同比较。

来源 `phase_d/diagnostics/{fresh_trace_analysis.json,fresh_trace_analysis_hashes.json}`、
拆分 `run/partition_receipt.json`、每臂 `timing.json/spec.jsonl/target_verify.bin` 与监督收据。
GPUroute不设默认的决策沿用§20完整八轮结果；本项仅关闭测量/归因，不改优先级或kernel。

## 22. 电源长测完成；draft head 暂做CPU工作（2026-10-06）

§4.8同引擎双盘三臂各8×512全部完成：balanced原始 **80.944ms/token**，
power-saver107.830、performance113.801；短/长两组也都选balanced。
三臂4096ID逐位一致、接受率85.8902%、命中91.4210%，无循环或加载失败。
performance的70.601 active估算不用于默认决策；其decode冷却176.603秒。
完整报告 [power_profile_comparison.md](power_profile_comparison.md)。

§4.9目前只完成工具和CPU权重参考：FP8行缩放副本662.431MB，35.219槽等价，
原target BF16仍需常驻。真实权重平方误差比0.07005%，**不是接受率/输出质量结果**。
64-output捕获第一次因route=1在session/prefill/生成前被拒绝，0输出、无hidden；
先前k=0 observer的GPUroute=1案例通过，不能代替生产CPUroute的head捕获。
配置工厂的生产默认已统一为route=0，准备新目录重做捕获，旧失败保留。

owner要求散热暂不可增加，先做CPU工作；之后新GPU作业0，网页保持停止。
实际main构建、CPU31/31、标准55/55通过，52shader未变。没有合格head候选，
减半收益/接受率、条件GPU实现、最终ID/NLL/decode/DSpark和八轮速度门仍待执行。
当前top4接受器直接输出接受的草稿ID，不能将target head未改推断成最终ID必然不变。
[准备情况与后续命令](draft_head_screening.md)，[机器收据](draft_head_screening_receipt.json)。
