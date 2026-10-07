# Codex 接续任务

更新：2026-10-05 23:53（Asia/Shanghai），review 纠正 2026-10-06。这是接续任务与验收状态，不是全部完成报告。
先读 [STATUS](STATUS.md)、[质量方案](mask_quality_plan.md) 和本文件，再核对进程及 Git 状态。

**本文件规则：只追加和勾选，不删除条目，不整篇覆盖。** 关闭某项须写明证据或 owner 决定，保留原文。
§4「review 补充/纠正」由 review 维护，Codex 只能勾选或在条目下追加收据。

## 0. owner 决定（2026-10-06，优先于下文任何旧表述）

1. **默认电源模式 performance。** 所有测速、质量作业和网页都用 performance。
   监督脚本（`phase_*/run_checks.py` 里 `job.get('profile','power-saver')` 及开头的 `set power-saver`）
   默认值改为 `performance`；provenance 记 `powerprofilesctl get`。power-saver 下已有的速度数只作参考，
   需要速度结论的配置在 performance 下补测一次。
   温控 80°C 暂停/72°C 恢复和 AC 检查保持不变；若 performance 下频繁热暂停，记录暂停次数后照常跑完。
   **§4.8 更新收据（2026-10-06）：未来默认 balanced。** 完整v4三臂原始墙钟
   80.944 / 107.830 / 113.801 ms/token（balanced / power-saver / performance），
   前两轮与后两轮也均由balanced胜出。按§0.5/§4.8规则替换暂定performance默认；
   原文及历史结果保留，当前温控仍按§0.7六项设备阈值。来源`docs/power_profile_comparison.md`。
2. **网页默认：动态 mask + 投机。** owner 接受 A 当前质量（MMLU 46/57、续轮重复 .118）作为默认。
   - k 由 D 决定：在 performance 下测 k2/k3/k5（mask），选 ms/token 最低且无循环/重复门不失败的 k。
   - D 完成前网页先用 mask + k=2，不用 k=5。
   - ONECB 开；GPU route 按 §4.2 的 D 结果决定，无收益则关。
   - 网页仍保留 `off`/plain 显式选项。
3. 以上不影响 C：C 继续，按原门判断，GO 也只加显式选项，不改默认。
4. **项目全面改名为 cachedMoE。** 展示名已在 main `0f637f9` 改完；程序名、CMake、namespace、
   环境变量、数据路径全部改，旧名保留兼容。步骤和约束见 §4.7。
5. **（2026-10-06 晚）重新比较电源模式，§0.1 的 performance 默认暂定。** C r4 在 performance 下
   热暂停 4263 次、冷却占墙钟 40%，动态 mask 实际 ~180ms/token；A 在 power-saver 下 110ms/token、0 暂停。
   用 §4.8 的实测决定网页和测速的默认电源模式。结果出来前网页保持现状。
6. **批准 draft head 的 FP8 / vocab 子集实验。** 这是“不重新量化”硬规则的**唯一例外**，
   只限 DSpark draft 的 head；目标模型的权重、head 和输出数学一律不动。步骤和门槛见 §4.9。
7. **（2026-10-06 晚）温控阈值改为按设备分开，优先于下文所有“80/72”。**
   - GPU（amdgpu edge）：**85°C 暂停 / 77°C 恢复**（原 80/72）。C r4 的 GPU 峰值 84°C，暂停主要由 GPU 触发。
   - NVMe（Composite）：**保持 80°C / 72°C**。内盘自身告警线 `temp1_max` 83.85°C，外盘 89.85°C，
     85°C 会越过内盘告警，盘会自行降速。
   - 每臂/作业起跑门：GPU ≤60°C，NVMe ≤65°C（外盘空闲约 61°C，§21 的 60°C 门因此卡了 900 秒）。
   - 适用于 `bench/thermal_guard.py` 等监督脚本和网页 `launch_guarded.py`。阈值做成参数，默认取上面的值；
     每份结果的 provenance 记录实际阈值。旧结果仍按 80/72 标注，不追溯改写。
   - 改完跑温控相关 CPU 测试（`tools/tests/test_thermal_guard.py` 等），然后按新阈值继续 §4.8，用新输出目录；
     网页按新阈值重启，核对日志里的阈值和 `/api/config`。

## 当前现场

2026-10-06追加现场：C/D正式对照与GPU trace均已结束；`0ed4cb4`已把owner main `0f637f9`
展示名改动合入本工作树，合入后CPU27/27、工具41/41，52 SPIR-V与两个exe hash不变。
本地main已fast-forward到同一`0ed4cb4`，main目录正串行验证；推送、own worktree/branch清理、
完整兼容改名和网页恢复仍待完成。
下文现场描述保留其原始时点，不当作当前正在运行的作业。

- 主仓库：`/home/chenkailun/projects/cachedMoE`，main 已推到 `41cc44c`。
- 当前工作树：`/home/chenkailun/projects/deepmoe-mask-quality`，分支 `codex/mask-quality`。
  **在这里继续，勿新开重复工作树。** 构建目录是该工作树的 `build/`。
- 用户网页已停止，聊天记录保留。结束后须恢复；不要刷新用户的浏览器页。
- Phase A 的 MMLU 与八轮 plain 已结束。`phase_c/run_checks.py`（四项 NLL）已结束；
  review 时唯一 GPU 作业为 `phase_c/conversation/run_checks.py`（tau.20 中文64等）。
  接续时重新检查进程，不得重复启动。
- 原始结果均在主仓库 `bench/results/mask_quality/`；监督脚本、命令、环境和温度记录也在里面。
- **一次一个 GPU 任务**；电源模式 **performance**（owner 2026-10-06，见 §0），必须接 AC 电源，
  80°C 暂停/72°C 恢复。镜像须为 `holds 48 of 48`。
  不得写 checkpoint。每种配置只跑一次，已有结果要复用。
  **注意：监督脚本把作业切到 power-saver（ACPI `quiet`），`profile_receipt.json` 里的
  `performance` 只是恢复值。** power-saver 下的 ms/token 不能和历史 STATUS 速度直接比较，见 §4.1。

## 1. 优先完成本轮代码审查修复

### KV 写盘

已提交 `210cb0f`，尚未合回 main。改动：队列超限改为背压，不再丢其他会话快照；
内存淘汰及退出保存等待原子写盘成功，失败返回错误并保留内存副本；普通轮末仍后台写盘。
三项新增 CPU 测试已通过：队列压力下所有会话落盘、必要保存返回错误、旧磁盘快照与新提示词的共同前缀检查。

仍需：

- [x] 审核 coalesce/reset/失败路径：同名较新快照继承确认；reset 取消排队确认并等待在途写盘。
  activate 的必要保存失败会返回错误，但目标会话已激活，超预算旧副本保留；已写进文档。
- [x] 最终 CPU 25/25，包括 `suite.kvdisk` 8 cases。工具门禁因与 CTest 同时跑固定临时文件
  出现一次 suite.io 失败；其余32项通过，suite.io 串行复验通过。最终交付前仍须串行跑完整门禁。
- [x] 更新 `docs/kv_async.md` 的背压、必要保存与错误保留语义。

### GPU 路由 / ONECB 整理

`4a262da` 已独立提交，CPU 与 GPU 验收通过：

- `RuntimeConfig::gpu` 集中解析 11 个 GPU 路由、ONECB、MGT 和 readout 开关；热路径读取配置。
- 捕获层号读取 `dspark_target_layer_ids`，噪声 token 读取 `dspark_noise_token_id`。
- 每层 readback 容器按配置定长；给路由记录和 host/shader ABI 偏移命名并交叉断言。
- 重排 `finish_gpu_routes`、`snapshot_with_shared`、`record_gpu_route` 和 DSpark 文件，补 snapshot/fence/union 顺序注释。
- 单线程 union 保留原顺序；**没有并行化 kernel，也没有宣称性能优化**。

仍需：

- [x] 审核默认值及旧环境变量语义，保持 union 顺序与数学不变。
- [x] A 完成后完整构建；52个 SPIR-V shader 与整理前 hash 全部一致。
- [x] 五项已有 GPU case 均实际执行通过：route quant/hidden mean、ONECB 对 serial、
  prefix KV 回绕、k=0 单次 target、GPU route 拒绝多 streams。0 skips、0 热暂停。
- [x] 独立整理 commit；构建和逐项收据在 `gpu_review/`、`gpu_review_build_provenance.json`。
  （review：`4a262da` 还带了 `tests/test_io.cpp` +41 行，与整理无关，不算完全独立，见 §4.4。）

## 2. 完成 mask_quality_plan（顺序执行）

### A：根因已定位，质量门还没全部通过

`ce589e8` 恢复动态 mask 的 P0 严格优先；`c59fc95` 补计时单测。
`66d0f7a` 引入的 Engram P2 绕过 P0，使 decode 跑到异步填充前面。
恢复优先级即可在单盘、双盘找回历史 l3 NLL；不是磁盘设备读取突然慢六倍。

已测：off NLL `.622784` 逐位一致；mask `.835581`、服务率 `.8135`、丢失 mass `.1694`。
最终双盘 P0 平均 `62.82ms = 60.42ms 排队 + 2.39ms issue-to-land`。
落地拷贝在 service 内；service 是墙钟区间，不是纯 SSD 时间。
T=0/T=1 中文 64 token 均无循环；off/mask 的三组各 512 token 也无短周期循环。
**但 mask 中文续轮重复 4-gram 为 `.117878`，off 为 `.049116`，超过 1.5 倍门槛。**
不得把 A 写成“全质量通过”。

- [x] MMLU57 **46/57、2 invalid，不达48/57**。八轮 plain **110.142ms/token，9.079226tok/s**，
  served `.9390`、mass lost `.0483`、八轮无循环、P0失败0。全部 A 作业热暂停0。
- [x] 执行 `phase_a/validation/additional/run_checks.py` 的 decode 与 longctx 门禁（尚未启动）。
  收据：旧脚本未再启动，改用同覆盖的新80/72监督 `final_review_remaining/jobs.json`。
  decode两个case与longctx两个case均rc0；短decode保留当前6/8、7/8基线，ctx4k/ctx16k均8/8。
- [x] mask 的 MMLU 与续轮重复门失败，不能设置为新的质量合格默认值。
- [x] `docs/miss_mask.md` 已记录矩阵、分段计时和质量结果；STATUS 待最终决策更新。
- [x] **A 未通过（MMLU 46/57、续轮重复 .118 > 1.5×off）。按方案此时应停下请 owner 决定，
  而不是直接进入 C。** C 已经开跑，结果可以保留作为数据，但在 owner 决定前不得据此改默认或改网页。
  - [x] owner 2026-10-06 已决定：接受当前质量，动态 mask + 投机作网页默认，C 继续（见 §0）。
    STATUS 里要如实写 A 的两项未达标及 owner 接受。

> review 注：八轮 plain 110.1ms/token 比历史动态 mask 76.7ms/token 慢 43%，命中率却相同，
> 原因未查清，见 §4.1。在查清前不要用这个数判断 C/D 的速度。

### B：重复质量工具已实现，待最终交付

`bbcad74` 已接入 `hitrate_bench.py`、`web_longtest.py` 和工具门禁。
已知“霓”与英文周期 2 判失败，历史正常输出判通过。
短周期定义为滚动 128 token 内周期 ≤8、至少四轮且至少 16 token；同 token 连串 >3 另行判失败。

- [x] 将标定与边界测试收据写进报告；最终所有候选输出均列四项重复指标。
  - [x] 标定与边界测试收据已写入 `miss_mask.md` 的 Phase B（`11af3d1`）。四个已知输出
    判定正确；来源 `phase_b_calibration.json`、`cpu_final.log`、`tools_final.log`。
    首次工具门禁32/33的失败保留，最终25/25与33/33另列，未冒充首次全过。
  - [x] C/D 最终所有候选逐轮列出最长连串、短周期、重复4-gram和distinct-2；原合并条目暂不关闭。
    C r4的32轮已在 `phase_c/performance_recovered_r4/final_report.json` 完整列出；本项仍等D全部候选。
    最终追加收据：C32轮、D40轮共72输出四项均齐全，独立读取 `phase_d/final_report_r1.json`
    的逐轮candidate字段核对；D五臂全失败严格重复门，指标齐全不代表质量通过。B合并条目结案。

### C：加权部分 miss 等待，GPU 未验证

2026-10-06 r4追加状态：完整对照结束，C结案NO-GO；后续三类候选质量作业按止损跳过，详见下方收据。

已提交实验开关 `33962a2`，默认关闭；仅 plain decode，拒绝投机/批验证。
按原始 gate 权重选择 miss；P0/LRU 不变、不重新归一化。支持每 token 专家数/时间预算。
离线 mixed 27,399 token 已扫完，结果是理想即时 LRU，**不含填充延迟或生成反馈**。
`.20` 与 `.10` 的理想平均等待为 `5.52/15.82` expert/token，仅用于选候选。

- [x] 已保存最终整理构建 exe，SHA256 `08a13390735e53ee…`；来源 `4a262da`，52 shaders 不变。
- [x] 串行执行 `phase_c/run_checks.py`：off、tau0、tau.20、tau.10 的 l3 NLL。
  环境含双盘、`DEEPMOE_MASK_WAIT_BUDGET=0,0` 和工作树 shader 目录。
  （review 读数：off `.622784`；tau0 `.622784`，served 1.0；tau.10 `.623711`，served .9666，
  mass lost .016；tau.20 `.626679`，served .896，mass lost .0687。NLL/off 均 ≤1.007。
  l3 的 tok/s 含 prefill 且在 power-saver 下，不能当速度结论。）
- [x] off 必须 `.622784`；tau0 无限预算核对 off 等价性（两者逐位一致）。
- [x] 候选 NLL/off ≤1.10 才继续长生成、中文64、MMLU和八轮速度。
  已经判失败的配置按止损规则停止，写明省略哪些测试及理由，不冒充完成。
  收据：旧power-saver/5100槽NLL筛查通过；r4完整八轮后两候选均判NO-GO。
  按止损跳过后续performance中文64、三组匹配512输出、MMLU57；历史筛查保留原scope，不写全质量通过。
- [x] 最好候选同工作负载带 off/全 mask 对照，**三者同一会话、同一电源模式、连续跑**，
  provenance 写电源模式。速度 ≥20% 且全部质量通过才 GO；
  质量通过但速度 <10% 则 NO-GO。GO 才添加网页显式选项，是否默认仍由 owner 决定。
  收据：`performance_recovered_r4` rc0，四臂各8/8，同PID3641973/default session、performance/AC、
  双盘48/48、动态5500槽。tau.20实际raw增益16.92%/active估算39.19%；tau.10为10.12%/7.41%。
  两候选均失败严格重复门，且raw增益未达20%，C结案NO-GO；没有据active估算提升默认。
- [x] 报告写 `miss_mask.md` / STATUS；NO-GO 不进默认，可移除没有价值的运行时策略。
  收据：C结案及完整对照已写 `miss_mask.md`；`final_report.json`、`decision_receipt.json` 与
  `check_results.json` 在 `phase_c/performance_recovered_r4/`。冻结b07f二进制来源 `de42061`，
  engine decode边界/温控覆盖完整，加载失败0；不添加weighted-wait网页选项。
  本勾选为C实验/判定结案；STATUS统一汇总、D、网页GPU验收和合并发布仍在§3等后续项中待完成。

### D：重新选 k，未启动

- [x] 复用 A 的八轮 plain；同双盘5500总槽、同脚本/seed/单会话测 k2/k3/k5。
- [x] k2 再测 ONECB+GPU_ROUTE 同时关闭的一格；每配置只跑一次。
- [x] 报 ms/token、draft/verify/commit/CPU 每周期、接受率、union、命中和重复指标。
  用相同接受率换算 cycle 成本；不要把接受率变化称作 kernel 加速。
- [x] 标明 384 个 MTP pin 在5500槽内（主模型可替换槽5116），命中差只是不同生成路由的观测，
  不能完全归因于 pin。
- [x] 只对最有希望且未被重复门判失败的候选补全部质量门；最快且合格才可作为网页默认。
  若 plain 最快就关投机；若无 mask 配置合格，明确说明，不选一个伪“通过”配置。
- [x] 写进 `docs/dspark_topk.md` 与 STATUS，完成后按决策恢复网页。

收据（2026-10-06）：五臂各8/8同PID3701458/default session，performance/AC、双盘48/48，
KV重置而专家cache按固定顺序延续；plain参照复用完整C r4 performance矩阵，不把旧A power-saver速度混入。
ONECB+GPU/ONECB+CPU/serial/k3/k5的raw成本依次145.491/132.634/156.207/161.270/167.004ms/token，
active估算85.091/67.936/84.394/74.258/82.089；C/D不同PID且MTP pin容量不同，不称等质量加速。
40轮无精确循环、加载失败delta全0，但五臂全失败严格重复门，决策NO-ELIGIBLE-CANDIDATE，
未生成winner。后续winner中文64、三组512、MMLU57按已决定的质量失败SKIP，未冒充完成。
独立GPUroute无收益：k2 raw/active成本比CPU高9.69%/25.25%，保留ONECB、route默认关。
owner已经授权的mask+k2基线保留，不升级成D质量合格默认。来源 `phase_d/final_report_r1.json`、
`quality_skip_receipt.json`、`same_engine_fresh/check_results.json`；细表与归一化在 `dspark_topk.md` §20。
最终STATUS/机器收据及网页恢复仍按各自条目结案。

### E：可选内存账，未结案

`38d730a` 加了 allocator/cache/pinned/KV/scratch 状态账。
离线真实路由容量命中：5500/5600/5700/5800槽 = `.9155/.9179/.9202/.9224`。
勿采用模拟器默认旧磁盘模型的 tok/s 当实测。
已有现场数据在 `phase_e_memory_live.json`；prefill transit 在 decode 前已释放，
Engram 全量 scale 默认没有常驻，也没有可再释放一次的整套 host 权重副本。

- [x] 从最终构建的 status 补完整内存账，避免把共享 UMA 的 RSS/GTT/cache 重复相加。
  收据：`phase_e_accounting.json`、`miss_mask.md` 的 Phase E。A 99.8408 GiB、B 64 KiB；
  5100 槽占89.3372 GiB，pinned reserve 10.4658 GiB（payload 9.1704 GiB、padding 1.2954 GiB）。
  KV/decode scratch 0.00346/0.03125 GiB；这些是同一分配账的组成，不能再加 RSS/GTT。
- [x] 只有确证可安全腾出 ≥150槽（约3GB）才改分配，并验证 prefill/长上下文。
  没有足够安全空间则按方案跳过，记录依据。
  决策：SKIP。padding 与 scratch 不足3GB，且没有证明能安全挪作完整专家槽；未修改分配。
  追加收据：账由 `38d730a` 导出，`11af3d1` 补明释放的是 bootstrap host expert store
  （`store_.reset()`）；来源 `phase_e_accounting.json`、`phase_e_capacity.json` 与 `miss_mask.md` Phase E。

## 3. 收尾与交付

- [x] 提交本方案和简洁结论报告、机器收据。原始大数据保持 gitignored。
  收据：方案 `mask_quality_plan.md`、报告 `miss_mask.md` / `dspark_topk.md` / STATUS，
  公开聚合机器收据 `mask_quality_receipt.json` 已提交；C NO-GO、D无合格候选、E按规则SKIP明确保留。
  源报告 `phase_c/performance_recovered_r4/`、`phase_d/final_report_r1.json` 与trace归因仍在主raw目录，未加入Git。
- [x] 合回 main，验证、推送，再删除**本任务自己的**工作树与分支；不得删用户 untracked 数据。
  收据：51e8451 已合并并推送；main 构建、CPU27/27、工具41/41、52shader逐位一致。
  原始输入及两套54文件工具链均留在主仓库，随后仅删除本任务干净工作树。
  追加进度：owner main→工作树合并 `0ed4cb4` 已完成，合入后CPU27/27、工具41/41，
  52 shader及runtime/tests两个exe逐位不变；`post_main_merge_{cpu,tools}.log` 与
  `post_main_merge_shader_receipt.json` 已核实。main已fast-forward到`0ed4cb4`，main目录验证/推送仍交付中，本合并条目保持未勾；
  尚未推送、删own工作树或宣称网页恢复，不动owner untracked副本。
- [x] 更新本清单为最终未完成事项；先核对主仓库同名 untracked 副本是否有用户新编辑，再合并。
- [x] 用主仓库 `build/web_mask/launch.py` 恢复网页（磁盘KV开启、1M上下文、5500槽、双盘、80/72温控），
  验证 `/api/config` 与引擎日志。用户 transcript 不动，浏览器页不刷新。
- [x] 完成当前 goal 前确认以上必要项已处理；NO-GO / 按决策跳过必须有证据。

## 已完成的前置工作：不要重做

英文 README/UI、思考与正文分开、实际截图、异步轮末 KV 保存均已在 main。
Strata 可选 DeepMoE 文本后端 [PR #943](https://github.com/Niko1221/Strata/pull/943) 已提交并附到任务，
维护者尚未合并；本账号只有上游 READ 权限。等待维护者不是本地可自主完成的工作。
宣传稿在 `docs/launch_announcement.md`，GitHub About/topics 已更新；未自动发布社交媒体消息。
不要重新开启预测预取、非LRU、固定cache默认、自动冻结、重新归一化或已关闭 kernel 实验。

## 4. review 补充/纠正（23:53 版覆盖时丢失或遗漏的条目，勿删）

### 4.1 P0：速度回退未解释（新发现）

2026-10-06 r4追加状态：完整双盘八轮performance对照已完成，计算回退归因与本节验收结案。

同为动态 mask、双盘、5500 槽、八轮 plain：

| | ms/token | tok/s | 命中 | nvme_stall | expert_hit_ms | hot_gemv_ms |
|---|---|---|---|---|---|---|
| 历史 `miss_mask/speed_mask` | 76.7 | 13.04 | .94 | 1.22 | 24.9 | 38.6 |
| Phase A `long8_plain` | 110.1 | 9.08 | .94 | 1.31 | 35.3 | 58.7 |

IO 没变，变慢的是 GPU 计算（+40~50%）。最可能是 power-saver/DPM：监督脚本把作业切到 power-saver，
历史数据大概率在 performance 下测。同会话 512 token 网页测：off 8.52/9.13/8.34，mask 10.19/8.91/7.84，
mask 只比 off 快约 4%（历史 +43%）。

- [x] 在 performance 模式下同会话连续跑 off 与动态 mask 八轮（已有脚本，每配置一次），
  看能否回到 ~77ms/token。provenance 写 `powerprofilesctl get`、GPU 时钟/DPM 状态。
  收据：C r4同引擎完整off/mask八轮，raw为197.219643/179.840496ms/token，扣CPU暂停的
  active估算120.249564/80.264076；两种口径单独保留，raw包含热暂停。
  mask原始attention/MoE/tail为31.115/24.729/6.723ms/token，计算回退已消失；不做逐stage冷却扣减。
  来源 `phase_c/performance_recovered_r4/{final_report.json,check_results.json}` 与每臂 `turns.json`。
- [x] 若 performance 下仍慢，按提交二分（`41cc44c` → `4a262da`），以 per-op 的 `expert_hit_ms`/`hot_gemv_ms` 判断。
  收据：条件未触发，SKIP 二分。performance 七轮 mask attention/MoE/tail 为
  31.147/24.566/6.649 ms/token，历史八轮为31.853/24.896/6.734；指定算子退化已消失。
  原八轮对照尚未完整，墙钟受温控暂停污染，不能从这些算子数推断最终速度。
  追加来源：`p0_power_resume/thermal_analysis.json`、`evidence_audit/audit_20261006T092538.json`。
  仅关闭条件二分；§4.1完整八轮与C/D的速度决策仍保持未完成。
- [x] 查清前，C 的「比 off 快 ≥20%」门和 D 的 k 选择都不判；温控仍按 80/72，
  若 performance 下温度无法跑完，写明并请 owner 决定测速用哪个模式。
  （owner 已定 performance，见 §0。）
  收据：r4在performance/AC、80/72继续协议下完整结束；监督wall/active/cooling为
  1850.350/1109.604/740.746秒、4263暂停，GPU/NVMe峰值84.0/74.85°C。
  active仅是CPU暂停估算，暂停不取消已提交GPU任务；原始与估算排名有差异，C两者公开且按规则NO-GO。
  §4.1对照/归因结案，D可继续；这不代表D的k选择已完成。

### 4.2 P1：双盘 GPU route 让 verify 变慢（原 #4，被删）

`final_dual`：verify 169.7 → 182.8ms/周期（+13ms），同接受率周期 211.3 vs 211.8，无收益；
按 `dspark_e2e_plan.md` §8 是 >189 档。

- [x] D 里 GPU_ROUTE 开/关分开测；双盘下若仍无收益，只保留 ONECB，GPU route 默认关，记为 NO-GO。
  收据：正式D两个k2 ONECB控制均8/8；独立raw/active双3%门选CPU，GPUroute结案NO-GO，见§D追加。
- [x] 查 +13ms 来源（双盘时 snapshot/union 与 IO 完成的同步点），只查不改，结论写进 `dspark_topk.md`。
  收据：新双盘32输出trace已完成，CPU约10.039ms由host区间对上、未覆盖.002613ms；
  GPU组190.661ms Engram issue/landing是本次主项，缓存热度/union/接受率不同不作旧+13ms完整因果归因。
  29个target regions每cycle单forward，专家wait/加载失败0；详 `dspark_topk.md` §21和新分析hash收据。

### 4.3 P1：网页投机配置

- [x] 网页之前跑 k=5（约 65ms/token，可能比不投机还慢）。恢复网页时用 D 的结论；
  ~~D 未完成就用 plain~~ → owner 改为：D 未完成就用 mask + k=2，不用 k=5（§0）。
- [x] ~~mask 默认值、是否开网页投机都由 owner 决定~~ → 已决定：动态 mask + 投机（§0）。
  恢复网页后验证 `/api/config` 显示 mask、投机 k、performance 模式。

### 4.4 P2：代码质量（原 #10、#13 等，被删或只做了一部分）

- [x] **一个 commit 一件事**：`4a262da` 混入了 `tests/test_io.cpp` +41 行；以后拆开。
  收据：后续状态抽离 `b154180`、配置 `60d0458`、ABI测试尺寸 `34eede7` 和格式提交分别独立；
  保留 `4a262da` 的混合提交事实，不重写历史，后续仍须遵守本规则。
- [x] GPU route 状态移出 `Engine`：`4a262da` 只把 `saved_*` 改成按层数的 vector，
  `route_snapshot_`/`route_steps_`/`finish_gpu_routes` 仍在 `Engine` 里，未完成。
  收据：`b154180` 将状态与finish移入 `runtime/gpu_route_state.{h,cpp}`，Engine仅持有该状态对象；
  顺序核对与最终数值验证见 `gpu_state_draft/cleanup_receipt.json`、`final_validation_summary.json`。
- [x] 环境变量收进 `RuntimeConfig`：`engine.cpp` 的 `getenv` 由 47 降到 38，剩余项列清单，热路径上的优先。
  收据：`4a262da` 集中GPU配置，`60d0458` 移除decode热路径8键/9处读取；剩29处、27键是
  启动、session或prefill边界，逐项清单在 `gpu_state_draft/cleanup_receipt.json`，未声称全部getenv归零。
- [x] 魔数：核对 `L>=37`、`6*16*2`、`128799`、`opidx*336+80` 是否都已命名，未命名的列出。
  收据：`4a262da` 使用配置层号/噪声ID、`layout::kSavedRouteWords` 和共享plan ABI常量；
  `7513dd0` 补prefill捕获层号，`34eede7` 补测试尺寸。上述四个旧模式在审查范围内无剩余，
  host/shader尺寸有交叉断言；来源 `model/layout.h`、`gpu/shaders/dspark_plan_layout.h` 与最终验证收据。
- [x] 新代码不要只追加到文件末尾；超长行与一行多语句按周边风格整理（与功能改动分 commit）。
  收据：GPU route/ONECB相关实现重排与必要顺序注释在 `4a262da`、`b154180`；格式专用
  `d5b7913`、`723d218`、`74eb84c` 与功能提交分开，可执行token核对和最终数值门另有收据。
  本项覆盖此次审查范围，不宣称全仓库已统一格式；来源 `gpu_state_draft/cleanup_receipt.json`、
  `final_static_review_20261006_094804/review.json`、`final_validation_summary.json`。

### 4.5 P3：先测量再决定

- [x] 单线程 union kernel：先用 per-op 计时看占周期多少，<2ms 就不动。
  收据：`64b0443` 的27组单层fixture实测，M=1/3/6均值乘40约 .533/.970/1.754ms，
  是算术外推，非完整cycle实测；M6最大外推4.080ms保留。按均值减半收益不足3%不改kernel。
  来源 `final_review/main_union_timing.json`、`union_profile_receipt.json` 与 `final_validation_summary.json`。
- [x] profiling 开销 3.33%，目标 <1%；查是哪些计时点，默认关闭或降采样。
  收据：旧serial校准是 -3.33%，不等于开销；同状态ONECB fixture on/off为 +4.131%。
  处理为生产draft profiling默认off，主模型测试observer为空时不录额外query；不声称已达到<1%。
  来源 `final_review/onecb_serial.log`、`final_validation_summary.json` 的profiling段与 `core/config.h`。
- [x] draft head 约 10ms：FP8 head / vocab 子集只有估算，不实现，等 owner 决定。
  owner 2026-10-06 已批准实验，见 §0.6 和 §4.9。

### 4.6 不做（已关闭，勿重开）

预测预取（含 CPU 预测预取）、非 LRU 淘汰、reheat、resident-only 默认、固定 cache 默认、自动冻结、
mega kernel、host-flag、champion port、CM attention、scale-fold、pair-dot、MTP unpin、缩小 draft attention、
CPU 计算 miss 专家、部分专家、重新归一化、streams>1 的 GPU route。

### 4.7 P2：全面改名 deepmoe → cachedMoE（owner 2026-10-06）

现状：README、AGENTS/CLAUDE、文档、注释、网页标题已在 main `0f637f9` 改成 cachedMoE。
剩余约 290 个文件：`DEEPMOE_*` 178 个不同变量（1800+ 处）、`namespace deepmoe` 171 个文件、
可执行文件 `build/deepmoe`、`project(deepmoe)`、`cmake/deepmoe_options.cmake`、网页/KV 数据目录。

**时机：** 等当前 GPU 作业结束、本分支在途改动全部提交并合回 main 之后再开始；
单独开 `../cachedmoe-rename` 工作树，改名期间不做别的功能改动，避免和其他分支大面积冲突。

**命名：** 展示名 `cachedMoE`；程序、namespace、目录用小写 `cachedmoe`；环境变量 `CACHEDMOE_*`；
CMake 选项 `CACHEDMOE_*`。

按以下顺序，每步一个 commit：

- [x] **环境变量**：加一个统一的读取函数，先读 `CACHEDMOE_X`，没有再读 `DEEPMOE_X`；
  两者都设且不同时以新名为准并打印一次警告。C++、Python 工具、bench、网页都走这个规则。
  文档和脚本里的写法全部换成新名。旧名只在兼容函数和一条说明里出现。
- [x] **CMake**：`project(cachedmoe)`、`cmake/cachedmoe_options.cmake`、选项改 `CACHEDMOE_*`；
  已有 build 目录里旧缓存变量 `DEEPMOE_*` 仍能生效（读到就映射并提示）。
- [x] **可执行文件**：产物改为 `build/cachedmoe`，构建时同时生成 `build/deepmoe` 符号链接；
  网页 `server.py`、`launch.py`、`launch_guarded.py`、`RUNNING.txt`、bench 和 tests 默认用新名。
- [x] **C++ namespace**：`deepmoe` → `cachedmoe`，纯机械替换，单独 commit，不夹带格式或逻辑改动。
- [x] **数据路径**：网页 transcript、KV 目录等改到 `cachedmoe` 下。新目录不存在而旧目录存在时继续用旧目录
  或原子迁移；**不得删除或覆盖用户已有 transcript 和 KV 快照**，迁移前先备份并记录。
- [x] **工作规则**：AGENTS.md / CLAUDE.md 的工作树命名改成 `../cachedmoe-<track>`，构建、测试命令用新变量名。

**不改：**

- checkpoint 里的 `deepmoe_manifest.json`：checkpoint 不许写，只有这一个文件是已批准的例外。
  继续读这个文件名；不要在 checkpoint 里新建 `cachedmoe_manifest.json`。
- 二进制格式的 magic（trace、KV 快照、缓存文件等）：保持原值，否则旧数据读不了。
- `tests/data/` 的 tokenizer golden 数据，以及 `tools/oracle_dspark.py` 里引用固定 git 版本的原文。
- STATUS 历史记录、过去报告里的旧名、外部 Strata PR 标题。

**验收（全部在 performance 模式、一次一个 GPU 任务）：**

- [x] 全新 build 目录和已有 build 目录都能构建；52 个 SPIR-V shader hash 与改名前一致。
- [x] CPU 25/25、`tests/run_all.py` 全过；l3 off NLL 逐位 `.622784`；`suite.decode` 不低于改名前。
- [x] 只设旧 `DEEPMOE_*` 变量、只设新变量、两者都设，三种情况各跑一次 CPU 门，行为一致。
- [x] 网页用新程序名恢复，`/api/config` 正常，旧 transcript 和 KV 快照能打开。
- [x] `rg -i deepmoe` 剩余命中逐条列进报告，每条写明为什么保留。

### 4.8 P1：电源模式实测比较（owner 2026-10-06，先于 §4.9 做）

问题：80/72 温控下，performance 的冷却暂停可能把墙钟拖得比 power-saver 还慢。
已有数字不是同条件（A 是 power-saver plain mask，C r4 是 performance 同引擎多臂），不能直接下结论。

- [x] 配置固定为当前网页默认：动态 mask、5500 槽、双盘 48/48、k=2、ONECB1、GPU route 0、磁盘 KV 开、AC。
  用网页同一套启动参数和 `web_longtest.py`（或等价脚本）跑**八轮、每轮 512 token**，同一 prompt 集、seed 和顺序。
- [x] 三臂：power-saver、balanced、performance，每臂只跑一次。同一引擎进程中途切
  `powerprofilesctl`；每臂开始前等 GPU 和 NVMe 降到 ≤60°C 再开始，避免上一臂余热影响下一臂。
  （owner 2026-10-06 晚改为 GPU ≤60°C、NVMe ≤65°C，见 §0.7。）
  KV 每臂重置；专家 cache 按固定顺序延续，并在报告里写明顺序。
- [x] 温控照常生效（按 §0.7：GPU 85/77、NVMe 80/72），网页实际就是这样跑的。**主指标是原始墙钟 ms/token（含暂停）**，
  不用扣除暂停后的 active 估算做决定，active 只作附表。
- [x] 每臂报：总 ms/token、第 1–2 轮与第 7–8 轮各自 ms/token（区分短对话和持续生成）、暂停次数与冷却秒数、
  GPU/NVMe 峰值温度、GPU 平均频率（`pp_dpm_sclk` 或 amdgpu 传感器）、接受率、命中、四项重复指标。
- [x] 决策：原始 ms/token 最低者为默认；与最快者差距 ≤3% 的取更低功耗的那个。
  若短对话最快和持续生成最快不是同一模式，两组数都写清楚，交 owner 决定，不自行做按长度切换。
- [x] 结论写进 STATUS 和本文件 §0.1；需要改时更新网页启动器和监督脚本的默认，重启网页后核对 `/api/config`
  和 `powerprofilesctl get`。不刷新用户浏览器，不动 transcript。
  v4结论及共享balanced默认已写入；网页/API恢复在§4.9串行GPU步骤之后进行，暂不勾选此项。
  最新owner要求暂做CPU工作；网页引擎也属于GPU作业，当前不启动。恢复/API验收继续待办。
  2026-10-07补充：owner恢复GPU，现已完成main验收和实际balanced网页/API恢复，见§33；上述待办为历史时点。

### 4.9 P2：draft head FP8 / vocab 子集（owner 2026-10-06 批准，§4.8 之后做）

现状：draft head 是 bf16、约 1.32 GB，带宽利用率约 64%，每 cycle 约 8.4ms（profile 下 10.0ms）。
tile 调优已判过（减半 1.24ms < 2ms）；pair-dot NO-GO。draft 只产生候选，target verify 决定输出，
所以 **draft head 变化只能影响接受率和速度，不应改变任何最终 token**；这一点必须实测证明。

范围约束：

- 只改 draft head 的副本。若 draft 与 target 共享同一个 head 张量，新建 draft 专用副本，target 继续读原 bf16。
  副本占用的内存要计入槽位账（约 18.8MB/槽），写明少了多少槽及其命中代价。
- 不写 checkpoint：FP8/子集权重在加载时由 bf16 现场生成，或放在 checkpoint 外的派生缓存，记录生成方式和 hash。
- 默认关闭，用显式开关；GO 后才考虑默认值，由 owner 定。

步骤（先离线，后 GPU，每步一个 commit）：

- [x] **离线估算**：在一次短运行（64 token、k=2）里导出 draft 最终 hidden，用 CPU 分别算 bf16、FP8（per-row scale）、
  vocab 子集（按语料词频取前 N，N=16K/32K/64K）的 top-1/top-4，对比 bf16 的一致率，
  并用已记录的 target token 估算接受率变化。子集外的 token 记为 draft 必然不中。
- [x] **预期收益先算再减半**：FP8 字节减半、子集按 N/129280 缩小，按实测带宽换算每 cycle 省的 ms，减半后
  **≥2ms/cycle 且估算接受率下降 ≤3 个百分点**才进入 GPU 实现；否则 NO-GO，写明数字。
- [x] **GPU 实现**（只做通过离线门的那一种，或两者组合）：新 kernel 与 bf16 版并存，micro-bench 报带宽利用率和 ms。
- [ ] **正确性**：同 prompt 下开关前后最终输出 token 逐位相同（target verify 不变）；l3 off NLL `.622784` 不变；
  `suite.decode`、DSpark golden 不低于当前基线。
- [ ] **速度**：同引擎、同电源模式（用 §4.8 的结论）、八轮 k=2，与 bf16 head 对照，各跑一次。
  按同接受率换算 cycle 成本；墙钟 ms/token 提升 <3% 判 NO-GO，开关保留默认关或删除。
- [x] 结论写进 `dspark_topk.md` 与 STATUS；GO 才让 owner 决定是否进网页默认。
  2026-10-07收据：生产组合最终ID门NO-GO，完整候选速度门SKIP，FP8默认继续关；见§32。

## 5. Codex 追加收据（2026-10-06 00:23）

- 已遵守 §0 的新决定：九份监督脚本的未来默认值改为 performance；旧脚本及 hash 保存在
  `bench/results/mask_quality/profile_supervisor_legacy/`，历史 power-saver 结果仍按原模式标注。
- P0 的第一次 performance 对照在旧 `stop_on_pause` 开关下提前终止，不作为速度结果。
  `p0_power_resume/` 正在同一引擎/PID/default session 连续运行 off/mask 八轮；
  80/72°C 暂停恢复，逐样本保存实际 power profile、ACPI 和 GPU 时钟。
- D 尚未启动。已补独立 `k2_onecb_cpu_route`：ONECB=1、GPU_ROUTE=0，另保留0/0和1/1，
  以及 k3/k5。所有配置 performance，保留384个 MTP pin，不改变主模型 LRU。
- 草稿单线程 union 的旧逐算子数据已复核：三个阶段合计0.02445/0.02841 ms每周期，
  最大0.03066/0.03855 ms，来源 `union_profile_receipt.json`。主模型 union 尚未隔离计时。
- GPU 路由状态抽取和热路径配置整理目前只有未应用草案；本轮测量的源码和二进制没有改动。

## 6. Codex 追加收据（2026-10-06，继续执行）

- `p0_power_resume` 已结束：off 八轮完整，mask 七轮完整，第八轮被旧1500秒墙钟预算中断。
  243次暂停累计994.673秒。89°C均是旧慢采样首次触发暂停时的温度；停止后的后续采样GPU最高72°C。
  旧逻辑还等待未达到80°C的NVMe降到72°C。详见原始 `thermal_analysis.json` 与 `evidence_audit/`。
- `9f61ab3` 增加逐传感器80/72锁存、50ms温控采样、AC检查和实际电源模式记录；
  执行预算与冷却墙钟分开，保存绝对暂停区间。13项CPU测试通过，含真实普通进程组暂停恢复。
  新报告同时保留原墙钟与扣暂停的active时间；后者使用host done边界估计，不能冒充GPU计时。
- C 的 `phase_c/performance_recovered` 已启动：同一个冻结 `4079180` 引擎、default session，
  依次 off/mask/tau.20/tau.10 各八轮，performance、双盘5500槽；KV重置，专家cache延续。
  二进制hash、shader来源和新温控协议另存，完成前不判C/D。
- `b154180` 抽离GPU route state/finish，`60d0458` 移除decode热路径8键/9处getenv；
  新CPU配置suite与最终GPU对拍尚待统一构建执行。路由数学、union和fence顺序保留。
- `64b0443` 为主模型union增加测试专用计时点，生产未传observer时无额外query；
  `34eede7` 命名测试ABI尺寸，后续ONECB/union测试格式提交经可执行token核对不变。
  主模型union实测尚待执行；不改变单线程kernel。
- `2d7e786` 的网页配置API显式报告mask/cache/spec-k及当前实际电源模式，CPU网页检查8项通过。
  网页仍停止，后续按D结果恢复；用户transcript和浏览器页未动。

## 7. Codex 追加收据（2026-10-06，恢复执行）

- `phase_c/performance_recovered` 以 SIGINT 结束（rc99），off 八轮完成、mask 四轮完成，
  tau 两臂未开始。保留原始文件与 interruption receipt；这不是完整四臂速度结论。
  `performance_recovered_r2` 将用同一新引擎重测完整四臂，避免跨引擎复用旧 off。
- `5d757af` 在网页关闭时先取消生成，再排队 quit，等待必要 KV 保存；CPU 网页10项通过。
  `3c3a2c6` 注册投机配置选择测试（4项通过）。
- 新构建来源 `3c3a2c6`，exe SHA256 `d4bf3184a3885660…`，52 shaders；
  串行 CPU CTest 26/26、工具门禁38/38通过，收据在 `final_review/`。
- 最终 GPU 前六项实际执行通过：27组路由/量化/hidden mean、ONECB 对 serial、
  k=0 单次 target、KV 窗口回绕、多 streams 拒绝、新增同引擎投机配置切换。
  后者验证各路径草稿前缀逐位一致、每轮 target calls=1；窗口测试一次热暂停后正常继续。
- off NLL 准备脚本错误传入了仅 serve 支持的 `--cache-slots`，未开始数值测试。
  原失败保留；纠正后只执行剩余 off64/decode/longctx，不重复已经通过的六项。
- `c39ea20` 修复温控异常退出后的失败收据：终态读数失败仍保留清理与失败原因，
  未知 AC/末尾样本记 null，不伪造成功。17项 CPU 温控检查通过。

## 8. Codex 追加收据（2026-10-06，完整数值门与计时）

- `final_review/` 前六项与 `final_review_remaining/` 后三项实际完成，无skip。
  off为打印精度的 `.622784 / 64`，自动缓存实际5100槽；不是声称内部double完整精度已导出。
  short decode仍为loaded 6/8、own-prefill 7/8，与方案所列当前基线一致；
  4K/16K free与teacher均8/8。case通过与严格逐token8/8分开记录。
- `de42061` 在真正decode结束点记录Unix时间（17位有效数），先于reheat与KV checkpoint。
  `4228778`/`a5133cd` 使用该边界并拒绝用legacy host估计选择新默认；
  覆盖不足的温控日志保留raw、active记null。新JSON两个C++case实际通过。
- 计时版exe SHA256 `b07f2480c8464681…`，52shader不变；工具门禁39/39。
  热计时23项、投机选择6项、报告11项CPU检查通过。
- `023eb6f` 保留实际raw端到端成本作D选择，并另报扣CPU暂停的active估算排名。
  GPU route须在raw和active都超过3%收益；SIGSTOP不能撤回已提交的GPU工作。
- `a9e1027` 修复网页启动期退出发现late engine的竞态，19项CPU检查通过；
  `101e686` 将closing标记与cancel/quit放在同一次io_lock内，之后拒绝新生成，15项检查通过。
- C `performance_recovered_r2` 已开始四臂完整对照，使用上述真实时间标记版。
  原准备manifest与旧无marker二进制收据均备份，旧中断结果保留；完成前不判C/D。

## 9. Codex 追加收据（2026-10-06，AC中断与网页显式模式）

- C `performance_recovered_r2` 在10:15因AC从1变为0终止（rc99），ACPI随之变为balanced。
  off八轮、mask五轮完成，tau两臂未开始；保留`ac_loss_receipt.json`和全部原始记录。
  这不是C的完整速度判定，也不是温控首次暂停后主动停止。
- 原计时版exe、测试exe与52个shader已复制至`frozen_runtime_de42061/`并逐文件核对SHA256。
  C完整重测准备为`performance_recovered_r3/jobs.json`，D正式对比和trace也绑定该冻结版本。
  只有AC恢复才能启动GPU；独立代码和文档工作继续，goal保持active。
- `11af3d1`补齐B四样本标定、CPU边界与历次门禁收据；E仍按安全空间不足判SKIP。
- `6171b71`补网页每轮显式`mask-spec`/`mask-plain`/`off-plain`选择、排队模式固定及实际模式历史。
  新网页CPU11/11、原网页15/15通过；C++请求策略与GPU验收尚未完成，不宣称部署已生效。


## 10. Codex 追加收据（2026-10-06，网页模式验收与双盘恢复）

- `0fe94ac` 实现每轮 C++ decode policy，`c1e173d`/`2678a79` 修复 mask 切 off 的缓存边界。
  显式 off 在 fence 后清除尚未提交的旧 background 意图，只等足够槽可用；已提交 IO 继续。
  512槽压力 fixture 实测 evictable 3→6，filling 125→122，384个MTP pin与guard/timeline不变。
  这证明只等3个完成，不是排空122个在途填充；该小缓存测试不是生产P0零失败或速度验收。
- `final_web_policy/recovered_boundary/validation_receipt.json` 的真实 GPU case rc0：精确feed/KV逐位、
  129 plain steps、命名session恢复、每投机cycle单target、错误/取消恢复全部通过。
  `complete_validation_receipt.json` 补最新CPU CTest27/27、工具41/41（串行执行）；
  54个测试版本文件备份在 `frozen_web_boundary_df316001/receipt.json`。
  `b6401f4` 说明每轮模式、旧KV复用和新chat精确历史；网页仍未部署，待D、合并及改名后恢复。
- 外接盒在USB4已授权时仍缺NVMe端点；owner授权重扫描后已恢复PCIe NVMe，镜像ntfs3只读挂载。
  C r3启动前缺源失败独立保留，不混入速度；r4实际两源48/48、完整32轮、加载失败0。
  C r4 NO-GO及跳过后续质量作业见 `phase_c/performance_recovered_r4/decision_receipt.json`，
  D五臂串行测量继续，期间不启动第二个GPU任务。
- README/STATUS最新说明在 `f127018`：旧截图与power-saver速度保持历史标记；
  A质量由owner接受，未写成全部质量门通过；C NO-GO、D/改名/网页恢复仍分别待结案。


## 11. Codex 追加收据（2026-10-06，D完整判定）

- D五臂完整监督rc0，wall/active/cooling为2137.537/1271.703/865.834秒。
  五臂全strict-repeat失败，原始与估算控制组排名保留；没有伪造qualified winner。
- `0bd3d42` 强制报告实际ONECB1、formal route、完整比较后才可能选默认，16项CPU检查通过。
  最新完整工具门禁41/41在 `final_web_policy/recovered_boundary/tools_after_policy_report.log`。
- C/D每轮正式7个输入在 `input_snapshot/receipt.json` 中按hash备份，以便合并/改名后审计。
  冻结b07测试工具链与df316001网页边界实现另存，未覆盖任何历史原始结果。
- 新双盘k2 GPU/CPU短trace已监督rc0，55.861秒wall、55.129 active、.733秒冷却；
  每臂32输出、拆分收据已生成。当前仅说明采集结束，额外verify耗时归因待独立分析。
- Strata PR #943最新只读核对仍OPEN，无comments/reviews；原题与兼容接口保留，维护者合并待外部处理。


## 12. Codex 追加收据（2026-10-06，trace结案）

- 新verify trace完成独立host分区核对，残差GPU/CPU .002337/.002613ms；嵌套桶不重复相加。
  mask专家miss异步不等，Engram保留精确数据依赖。温控9暂停/.733s、AC/performance，所有目标时间戳有效。
- `f368b2e`修掉CPU控制loop可绕过GPU双3%门的selector边界，8项CPU测试；
  变更后完整工具41/41通过。原C/D/trace输入与收据保留，不回填新源hash，也不重跑GPU。
- `623a744`、`a120e35`与本次trace报告记录D失败判定和curated机器收据；
  当前所有必要GPU作业已结束，准备保留owner main展示名后合并、验证、push，再单独执行§4.7。

## 14. Codex 追加收据（2026-10-06，main交付）

- main 实际构建、27/27 CPU、41/41 工具通过；52 SPIR-V 与冻结测量版逐位相同。
  实收据 `bench/results/mask_quality/main_delivery_receipt.json`，51e8451 已 push。
- 原始 C/D/trace、输入备份、数值门与网页边界工具链全部保留。主仓库 untracked 数据未删。
- §4.7 六步兼容改名开始；网页仍停止，改名验收后恢复，无浏览器刷新。

## 15. Codex 追加收据（2026-10-06，配置唯一来源与删除式整理）

- 六步兼容改名分别提交，源实现到 `43a2ab9`。原生87键的解析与owned快照集中，
  75个Python文件/433个parser defaults/113处配置调用均有逐项清单，热路径不再读取环境。
- 15项C++/Python事实共享一份typed定义；回放128、RADV16/会话512阈值分别命名，
  39处模型可选字段fallback读字段本身默认。模型shape与实验工作量不因同值而混合。
- 修复两项真实CLI问题并单独提交：GPU route检查与引擎同一exact-one解析；
  no-KV在两种参数顺序都有效。删除从未生效的run缓存开关。
- Python最终启动环境、encoder、provenance一致；空shader目录原样保留，
  不把cwd里的无关SPV计入。两项独立review发现均关闭。
- 最终真实构建通过；旧/新/混合前缀分别31/31 CPU；工具52/52，元数据suite实际执行，
  原漏label的3组unit已纳入。15事实由实际C++ header reader对拍，不只比手写字典。
- 52 SPIR-V逐个hash不变；新library读旧default/web KV（396/4710 positions），
  原文件hash/mtime/size未改；旧namespace qualified client实际编译通过。
- 配置说明：`docs/runtime_configuration.md`。机器证据：
  `bench/results/mask_quality/rename_prepared/final_config_validation/`及各source delta/review收据。
- 主目录旧的ignored `build/web_mask/launch.py`已备份，改为调用tracked温控启动器，
  不再复制电源/k/route等设置；dry-run通过，未启动服务，聊天与KV目录未改。
- 尚未关闭：已有main build/合入推送清理、改名后GPU数值门及网页恢复。
  本次preflight实际AC=0、外置NVMe未枚举；已请求接电/插盘，GPU未启动，网页仍停止。
  此处CPU/格式验收不当作新GPU数值验收；owner接受的mask质量限制与C/D NO-GO保持。

## 16. Codex 追加收据（2026-10-06，实际main目录验收）

- 本地main已fast-forward到`bec29fd`；已有build的170步完成，旧程序/library/tests
  三个入口均为正确的canonical产物符号链接。原11个owner untracked结果目录全部保留。
- 实际main的旧/新/混合前缀各31/31 CPU；工具52/52（checkpoint元数据启用），
  52 SPIR-V与冻结版逐个hash相同。不是拿worktree的通过代替main验收。
- main实际binary的5个CLI边界例全部通过；旧namespace client实际编译；
  旧default/web KV读取396/4710 positions且hash/mtime/size均未改。
- committed-tree逐命中审计`remaining_committed_cpu_delivery.json`覆盖4842处内容，
  全部有RETAIN角色/理由，0 NEEDS_REVIEW；旧module文件名另列兼容理由。
- ignored启动器在main重新dry-run通过，指向tracked温控入口和新程序；未启动网页。
- 最终9项GPU验收计划已改为实际main路径，之前未执行的准备版完整备份。
  验收适配器拒绝skip、0/wrong case数、NLL不足64步、baseline漂移及非48/48镜像；
  9+6个CPU适配器边界例通过，包括防止短decode/longctx以宽松退出码掩盖退化。
  这些只验证验收脚本，不代表GPU已运行。
- 仍待实际硬件：AC=0、外置NVMe未枚举。新binary数值门、push/own工作树清理、
  网页/API/旧session继续生成验收保持未勾；不刷新浏览器，不重跑C/D已关闭方向。
  本段追加当前事实，不覆盖§15及更早时点的快照。

## 17. Codex 追加收据（2026-10-06，最终待交付项审计）

- 阶段25份与配置31份原始引用均按SHA复核；A/B、C/D止损与独立路由选择、E容量边界已核对。
  完成审计为`rename_prepared/final_config_validation/goal_completion_audit_e5131ab.json`。
- 当前只剩：①接AC/恢复完整只读双盘后跑新main的9项数值验收；②通过后push并清理本任务工作树/分支；
  ③温控恢复网页并实际验证API、旧聊天和KV续接，不刷新浏览器。此三项仍未完成。
- 最终清单已核对主目录同名tracked文件与工作树，没有用户新编辑被覆盖；原11个untracked结果目录仍在。
  当前remote main仍`a016a78`，8080拒绝连接，AC=0且无外置NVMe；目标未标为完成。

## 18. Codex 追加收据（2026-10-06，硬件恢复与实际main数值门）

- 外置PCIe NVMe恢复，镜像只读挂载；48/48 shard尺寸与header核对，AC/performance。
- 实际main `825ee1a` 的9项串行数值作业全rc0，10个registered cases、0 skip；
  off/mask64 NLL `.622784/.835581`，短decode 6/8、own-prefill 7/8，4K/16K两种检查均8/8。
  原严格短decode门仍未达8/8；本次证明不低于既有baseline，未重写A质量例外或C/D止损。
- 358源文件、55产物、92输入逐hash与冻结版一致；52 SPIR-V不变。80/72温控4次暂停共.543秒，
  不把验收用时当新速度跑分。tiny-cache switch fixture的P0资源告警有单独范围说明。
- 原旧web KV的隔离副本GPU恢复4710 positions、回放128；正常rollback2，复用4708、补prefill2、生成2，
  正常退出排空disk writer。原KV和8份聊天的hash/size/mtime均未改。
  首次smoke脚本漏算正常rollback，失败记录及修正后的验收都保留，没有改runtime逻辑。
- 原始收据：`rename_prepared/final_config_validation/actual_main_numerical_summary.json`、
  `legacy_live_smoke/legacy_live_result.json`；仍待push/own清理与网页/API恢复。

## 19. Codex 追加收据（2026-10-06，推送与网页恢复结案）

- `cd5f1ea` 已push，远端main逐SHA核对相同；只删除本任务干净工作树与branch，11个owner结果目录保留。
- 网页已用canonical程序名启动；guard/server/engine的实时身份核对，双盘48/48、AC/performance、80/72温控，
  dynamic5500、KV4GB、上下文1M、k2/top4/ONECB1/CPUroute0与 `/api/config` 一致。
- 4个旧会话history实际API读取正常；原旧KV和8份聊天文件hash/size/mtime仍不变，浏览器未刷新。
  旧web KV4710与当前web transcript3313不是同一prefix，未把隔离恢复宣称成当前聊天cache复用。
- 新隔离命名网页smoke生成24 tokens，9 cycles、18草稿验证/14接受，effort75；
  38-position后台KV已落盘并由实际library的CPU reader读回。这里只验部署和持久化，不作性能/质量跑分。
- 收据 `docs/web_restore_receipt.json`；原始live配置、状态、日志与thermal已在final_config_validation按hash冻结。
  原A质量例外、C/D NO-GO、E/Phase4/5条件SKIP保持；draft FP8 head仍按原文等owner决定，不新增实验。

## 20. Codex 追加收据（2026-10-06，新owner任务开始）

- owner `1db5c99` 新增§4.8/4.9；独立工作树 `../cachedmoe-power-draft`、分支 `codex/power-draft`。
  先电源三臂，再做draft-only离线筛选；不重开旧C/D/union/target量化方向。
- `d73bb7f`新增实际网页同参数的三臂driver、八轮512-token固定prompt和原始墙钟报告。
  同一engine、KV逐臂reset、专家cache按power-saver→balanced→performance延续，diskKV开启。
  临时端口8081和私有KV/transcript；原网页已正常退出，原6份state文件备份，5份transcript未改。
- 准备时镜像48/48尺寸/header核对、AC=1、只读挂载；真实engine已ready，两源/dynamic5500/k2/ONECB1/CPUroute0/1M一致。
- 第一臂尚未开始：GPU约48°C、内盘约33°C，外置NVMe空闲61.85°C，仍等所有设备≤60°C。
  60°C门不放宽；80/72温控持续。已请求外接盒辅助散热，当前没有有效速度结果或新默认。
- 原脚本/输入逐hash冻结在 `power_profiles_prepared/input_snapshot/`。CPU报告门新增短/长winner分歧检查，
  原始reader在进程中已加载，不把后处理修正追溯成已运行的新源；会保留旧报告并只重算raw数据，不重跑GPU。

## 21. Codex 追加收据（2026-10-06，电源比较冷启动门阻塞）

- 第一臂冷却完整等待900.289秒，11889份样本；外置NVMe最低60.85°C，未达§4.8的≤60°C。
  GPU/内盘最低44/31.85°C；三臂完成0，生成0，属于PREFLIGHT_BLOCKED，不是电源模式NO-GO。
- 启动2076次IO全部完成、context0；APST参数0、外盘runtime active只读记录，未改稳定性设置。
  已请求外接盒增加散热；清单的60°C门没有放宽。
- 临时引擎正常退出，KV必要写盘排空，web server rc0；用户网页已恢复原performance策略，
  实际 `/api/config` 验dynamic5500/k2/top4/ONECB/CPUroute/双盘/diskKV/1M；5份transcript hash/mtime未改，未刷新浏览器。
- 新报告工具CPU门4/4，初始driver commit `d73bb7f`、报告校核 `5387be7`；工作树保留，尚未合入main。
  原始输入、失败冷却、恢复收据在 `bench/results/mask_quality/power_profiles_prepared/`，
  原始thermal与临时engine日志在 `power_profiles_v1/`。测量样本为空，不更新默认或填造ms/token。
- §4.8未完成；§4.9按owner顺序依赖仍未启动，不宣称离线/FP8质量或速度通过。
  待外盘散热达标（或owner明确修改起跑条件）后继续，使用新输出目录；本次未开始任何配置测量。

## 22. 按 owner §0.7 更新温控（2026-10-06）

- [x] `RuntimeDefaults.ThermalPolicy` 是六项设备阈值的唯一默认源：GPU 85/77、起跑60；NVMe80/72、起跑65。
  监督器/网页启动器/三臂脚本共享参数解析，逐设备锁存；CLI 可覆盖且拒绝非有限或错误回差。
- [x] 网页命令显式携带实际策略，state、thermal、结果 provenance 和 `/api/config` 报告实际值；直接裸跑server不声称有监督器。
- [x] 温控、网页监督器、配置和三臂报告 CPU 检查73项通过；旧80/72行为测试改为显式旧策略，旧结果不改。
- 历史尝试：新目录`power_profiles_v2`按§4.8执行；只在正常停网页并排空KV后启动，恢复时核对新阈值API。
  已在§23因USB4掉线终止，后续完整比较见§27；不是待启动的作业。

## 23. §4.8 v2 被 USB4 掉线中断（2026-10-06）

- `bfb7fa0` 新阈值下实际API/manifest匹配，61项冻结输入未变；八轮首臂只完成3轮各512输出，
  第4轮prefill时内核PCIe Link Down、Ugreen设备断开，引擎镜像源连续3次IO错误被移除。
  0完整臂，不能选电源默认；§4.8/§4.9继续未完成。原始数据保留`power_profiles_v2/`。
- 断开前GPU67、内盘49.85、外盘74.85°C，0温控暂停。已证实链路丢失，温度/线材因果未证实。
- [x] 之前授权的PCIe端口rescan恢复NVMe枚举和既有RO挂载；APST等稳定性设置不改。
- [x] 修复监督脚本错误遮蔽：HTTP EOF后先查controller；primary/controller/cleanup分别记收据，
  清理失败也恢复原电源；紧急清理只用拥有的进程身份。77项CPU检查及新增真实CPU所属进程清理通过。
- 临时benchmark退出为非graceful，私有KV排空不能确认；用户原网页已先正常排空，六份原文件保持不变。
- [x] 等盒子散热/USB4连接检查后，完整双盘长测用新目录；保留失败样本，不将三轮补充成已完成臂。
  由完整v4三臂完成，见§27；v2失败样本不改。
- [x] 恢复新温控网页并核对日志/state/API，不刷新浏览器、不改transcript。

## 24. 新温控网页恢复与CPU集成收据（2026-10-06）

- [x] main 标准CPU门54/54，含新三臂报告与失败处理9个CPU案例。没有改动引擎、权重或shader。
- [x] 实际8080 API/state/log共同确认GPU85/77、NVMe80/72、起跑60/65；performance未改变。
  动态mask5500、双盘48/48、k2/top4、ONECB1/CPU路由、磁盘KV和1M容量均保持。
- [x] 六份旧文件SHA/mtime保持，未刷新浏览器。公开摘要`docs/power_profile_web_receipt.json`，
  完整本机收据`power_profiles_v2_prepared/web_restore_new_thermal_acceptance.json`。
- 保留工作树继续§4.8/§4.9。外接盒散热/USB4连接检查问题已发出，等待答复；不是“全部任务完成”。

## 25. 继续执行与双盘监督补强（2026-10-06）

- 外接盒换到domain1后，曾约17秒在USB4与USB/UAS之间重连。新接口PCIe端口rescan已执行；
  之后实际NVMe恢复为`63:00.0`，RO镜像48/48的大小/头部匹配，六份用户文件SHA/mtime仍一致。
- [x] 生产网页与三臂启动前必须找到两块NVMe Composite传感器；缺失时拒绝启动，
  不能以仅内盘温度代替双盘温控。默认所需读源数在RuntimeDefaults只有一处。
- [x] 三臂除了READY初始读源数，还检查引擎source-dropped日志；温度仍可读的瞬时掉盘也使臂无效。
- [x] 相关CPU80项、main标准门54/54通过；新原始目录`power_profiles_v3/`和`power_profiles_v3_prepared/`。
- 历史尝试：v3同引擎三臂测量，61项输入已冻结、实际API阈值/双盘配置匹配。
  已在§26因owner手动换接口终止；仅两个完整臂，完整比较由v4替代，不再启动v3。
  §4.8/§4.9未完成；原网页已停止，只有私有benchmark引擎在工作。

## 26. v3 中断原因与 v4（2026-10-06）

- owner 确认20:51手动换接口，导致v3第五轮performance中断；两个八轮完整臂保留，
  performance仅四轮有效，不能补成三臂比较。原始数据与partial_receipt都保留，不改默认。
- owner 已授权重测，并确认接稳。新目录`power_profiles_v4/`，原接口RO镜像48/48，
  61输入逐hash冻结、实际API匹配；六份原网页文件SHA/mtime不变。
- [x] v4完整三臂测量，收据见§27；随后§4.9仍按原顺序。独立工作树准备只读hidden/既有verify矩阵捕获和CPU离线工具，
  捕获每cycle严格仅一个target forward，不将准备中的工具写成已完成实验。

## 27. §4.8 完整 v4 收据（2026-10-06）

- [x] 同引擎三臂各八轮512输出全部完成；正常退出并排空私有disk KV，两源始终在线。
  power-saver / balanced / performance 原始107.830 / 80.944 / 113.801ms/token；
  全程、前两轮和末两轮都选balanced，不需要owner裁定短/长分歧。
- [x] 三臂4096输出ID逐位一致，接受率85.8902%、命中91.4210%，0循环/加载失败。
  performance的decode温控1567次、冷却176.603秒；其70.601ms/token active估算不用于默认决策。
- [x] 共享电源默认改为balanced，实际本机profile也为balanced；报告和机器收据已push于3b7d763。
  来源`power_profile_comparison.md`、`power_profile_result.json`及`power_profiles_v4/`原始数据。
- [ ] 网页重启/API验收：owner最新要求暂做CPU工作，网页也计作GPU作业，保持停止。
  本节完成测量/默认选择，不冒充网页已恢复；§4.8最后一项仍未勾。

## 28. §4.9 CPU准备与原始失败收据（2026-10-06）

- [x] 新capture工具与只读verify观察点已构建；每cycle检查一个target forward，
  observer未设置时无额外读回。CPU CTest31/31、标准门55/55、离线参考11/11，52shader逐位不变。
- [x] 先前GPU k=0 observer案例通过：两轮各单forward、argmax一致，0 skip/暂停。
  此例是GPUroute=1；不将其写成CPUroute或64-token head捕获验收。
- [x] 真实head的CPU行FP8校核：副本662,430,720 bytes，35.219槽等价；
  权重平方误差/源能量0.07005%。仅权重误差，不是输出质量或接受率结果。
  原BF16 target仍必须常驻，副本属于额外内存，未实际削减专家槽。
- [x] 第一次native capture因route=1被工具拒绝：GPU初始化/pinned读取已发生，
  但session/prefill/生成尚未开始，0输出、无hidden。原失败plan/log/thermal完整保留。
- [x] 生产配置工厂默认路由修为统一CPUroute=0，17项配置CPU检查通过；显式实验覆盖保留。
  捕获策略校核提前到init_gpu之前。新CPUroute计划/新目录/输入hash已准备，尚未执行。
- [x] 64-token k2实际捕获及CPU BF16逐位候选校核、FP8/16K/32K/64K接受率估算。
  2026-10-07结案：见§29～32，FP8过离线门、子集NO-GO；此后原准备描述保持历史原文。
  独立词频输入23,055 tokens、5,123 unique，语料代表性限制已记；没有合格候选或GO结论。
- [ ] 原§4.9减半收益/接受率门、条件GPU实现、最终ID/off NLL/decode/DSpark质量门、八轮速度门。
  全部保持未勾；top4接受器直接发草稿ID，不能凭target未改就宣称最终输出必然不变。
- [x] 双盘散热条件满足且CPU-only阶段结束后，再继续串行GPU任务及balanced网页恢复。
  2026-10-07实际恢复完成，见§33。
  最新owner要求之后新GPU作业0；六份用户文件SHA/size/mtime不变，浏览器未刷新。
- 报告`draft_head_screening.md`、机器收据`draft_head_screening_receipt.json`；
  原始CPU/准备数据`draft_head_cpu_weights/`、`draft_head_cpu_route_prepared/`，旧失败`draft_head_prepared/`。

## 29. §4.9 真实捕获与离线入口门（2026-10-07）

- owner已恢复GPU工作；网页继续停止，所有GPU任务串行，balanced和双NVMe温控不变。
- 64输出、25cycle、49验证位置、38接受；CPU BF16 top1逐位100%。每cycle一个target forward。
- FP8固定轨迹接受率下降0pp；16K/32K/64K下降16.327/8.163/4.082pp，子集NO-GO。
- fresh M=2 head微测8.903326ms、148.689GB/s，一次预热/一次8调用批；
  FP8减半预测2.224093ms，只有FP8符合GPU试做门。仪器化capture时间不当速度。
- GPU实现/实际微测/最终ID/off NLL/decode/DSpark/条件八轮速度仍未完成；副本662,430,720B必须另计。
- 原始`draft_head_capture_cpu_route/`、`draft_head_screen_current_cost/`、`draft_head_baseline_micro/`；
  8.4ms历史估算收据保留，不覆盖。公开聚合`draft_head_screening_receipt.json`。

## 30. §4.9 FP8 GPU短门（2026-10-07）

- head微测8.903326→3.539821ms，有效带宽148.689→187.137GB/s；原始温控收据0暂停。
- 全量编码/scale hash与CPU相同；FP8 logits最大误差0.0000153，两行argmax相同。
- 显式`CACHEDMOE_DSPARK_HEAD_FP8=1`，默认关；启动配置唯一注册，拒绝mega/无投机组合。
- 当前100槽slab使实际预算从5500降到5400，不只少理论36槽；384 pin后主模型5016槽。
  副本662,430,720B；slab预算1,880,883,200B，多出的1,218,452,480B不冒充可用专家。
- 真实64输出逐位相同，25cycle和38/49接受不变；每cycle一次target，target BF16未改。
- CPU31/31、工具55/55。一次质量启动被并行CPU CTest的idle检查拒绝，0GPU输出；旧日志保留。
- off NLL/decode/DSpark及八轮速度仍待完成，不能把短门写成GO或网页默认。

## 31. §4.9 默认布局与数值门（2026-10-07）

- off teacher-force NLL 保持 0.622784；decode loaded 6/8、own-prefill 7/8 保持当前基线，未冒称 strict 8/8。
- DSpark per-stage、ONECB full runtime、mega full runtime 已通过。原候选 mega 的一次失败及 main 控制通过都保留，原因未证实。
- FP8 1～5 行 serial/ONECB 共10个组合逐位一致。关闭 FP8 时不分配新增 stage，保持默认参数布局。
- CPU 31/31，tools 56/56。证据：`bench/results/mask_quality/draft_head_fp8_default_layout_gates/`。
- 同引擎八轮驱动及受控空上下文 head 切换已准备；两臂都固定5400槽，隔离 head 成本，不把它写成对5500槽默认的完整增益。
- 最终 ID 若发生差异立即取消候选臂；八轮结果和默认判定仍待完成。

## 32. §4.9 条件八轮判定结案（2026-10-07）

- [x] 同引擎balanced双盘八轮已启动、按首次最终ID差异止损。Native完整8轮2178输出，85.991236 raw ms/decode token；接受83.7338%、命中91.2558%。
- [x] 最终ID门验收结案但**未通过**：FP8第1轮第7输出14643≠20968，取消后10缓冲输出/3cycle，后7轮不再跑。§4.9原正确性“逐位相同”要求未满足，不冒勾成通过。64-output greedy短门曾通过仍保留。
- [x] 速度条件门**SKIP AFTER QUALITY NO-GO**，不是≥3%通过或测得无收益；不重复A/B。前2cycle相同4/4接受的cycle账写进报告，不能当完整速度结果。
- [x] 归因边界：native起点384 pin、FP8起点5400满缓存；第1cycle target routes已有326/720不同，差异处各自target rank0。因此不能把最终ID变化单独归因FP8。保留动态LRU、top4接受规则和一次target forward。
- [x] FP8保留显式实验默认关，子集不实现；副本100槽预算和长期命中成本“未测”写明。原始结果`draft_head_fp8_e2e/e2e_receipt.json`、私有KV正常退出rc0，149温控暂停/19.230911秒，GPU/NVMe峰88/74.85°C。
- [x] STATUS与dspark_topk结论已更新，当前§4.9决策结案；先前§28～31“待”描述是原始时点，现以本节为准。
- [x] 实际main最终构建/CPU/GPU验收、push、balanced网页/API恢复、own工作树清理。
  2026-10-07全部交付步骤完成，见§33～34；质量NO-GO不改写为通过。

## 33. 实际main验收与balanced网页恢复（2026-10-07）

- [x] 合入main `8157f73`，实际main构建、CPU31/31、tools56/56；53个shader与已测工作树逐字节一致。
- [x] main新程序off NLL .622784、ONECB full runtime实际GPU通过，0热暂停；未并行网页或其他GPU作业。原始`draft_head_main_validation/`。
- [x] 网页实际恢复，HTTP config/status都200：balanced/AC、动态mask、5500槽、双盘、1M、4GB disk KV、k2/top4、ONECB1、CPU route0、单主路径target。FP8明确为0，保持原生BF16。
- [x] guard/server/engine身份已核实为79283/79305/79306；只作为本次观察，禁止用保存PID直接发信号。实际阈值GPU85/77、NVMe80/72、起跑60/65，guard未暂停。
- [x] 六份原transcript/KV的SHA、size、mtime全部不变；浏览器未刷新。本次只做恢复/API验收，不新增网页速度成绩。收据`docs/draft_head_delivery_receipt.json`。
- [x] 推送并核对origin/main、删除本次`cachedmoe-power-draft`工作树与`codex/power-draft`分支。
  `b3986c5`已推送、remote SHA一致；工作树/分支已实际删除。

§4.9最终ID“逐位相同”原验收项保持未勾，表示门未通过；速度项保持未勾，表示按质量NO-GO条件SKIP，**不是待跑任务**。本轮只保留FP8显式实验且默认关，不宣称质量/速度GO。

## 34. 本轮发布及清理收据（2026-10-07）

- `37e6367`离线/真实测量，`d5f0ef8`显式GPU副本，`80c5a04`默认布局隔离/比较控制，`8157f73`门槛判定分别提交；`b3986c5`实际main验收和网页恢复已push，origin SHA完全一致。
- own工作树`../cachedmoe-power-draft`及`codex/power-draft`分支已删除，只剩主工作树；主目录11份原有未追踪结果目录保留。原始GPU/CPU/失败/温控/配置收据均在main结果目录。
- 本轮§4.8/§4.9按owner门槛结案：balanced默认已验收；FP8生产质量NO-GO、候选完整速度条件SKIP、子集NO-GO。不把未通过项、未测长期命中成本或上游Strata等待合并写成完成/GO。

## 35. 受控草稿对照与当前 verify 拆解（2026-10-07）

- [x] 对照工具区分 exact-target 质量和 frozen-cache 成本；后者核对逐槽专家/pin 映射，IO 排空后测量，拒绝部分缓存、失败或替换。默认动态 LRU 不变。
- [x] exact-target：native64 输出；FP8 第22输出76111→104505，按首次差异取消。差异前两行 target routes 一致，候选分别 rank3/rank0，均被 top4 接受；明确 FP8 最终ID门不通过。
- [x] 冻结控制：5400槽全程同映射/同5400 fills，64最终ID一致、35/54接受和28cycle相同。raw66.523251→71.453364ms/token，温控69.833→509.486ms；active估算不当生产赢家，5500→5400长期成本仍未测。
- [x] balanced 双盘当前动态 trace：25 target/64输出，0加载失败/热暂停；verify201.262970ms，GPUbusy124.064133、跨提交gap74.875007，Engram host62.179328ms嵌套其中，专家阻塞等待0。不把 record/fence/Engram 重复相加。
- [x] 未关闭的 MoE 数组宽度探针：6→3列、同3 live/12 FP4专家/3层，逐位一致，1.875508489→1.796428236ms/层；40层减半1.581605ms<2ms入口，止损，不加默认内核。不重开§4.6关闭项。
- [x] 代码/测试和实验记录分开提交；报告`verify_control.md`和`verify_control_receipt.json`保留原混杂实验历史，不覆盖旧收据。
- [x] 实际main最终验收、push、恢复balanced网页、六份原用户文件核对、own工作树清理；完成后追加交付收据。
- 后续有边界的开放工作：真正限制 prefill activation workspace；须保留绝对位置/KV/CED/compressor/Engram连续状态，不把简单分段喂入冒充正确分块。M3常量循环尚未测，不把本次数组宽度止损扩成全部MoE无空间。


## 36. 本轮实际main与发布收据（2026-10-07）

- [x] `a644bdc`对照控制、`eabb11c`列宽探针、`eeb3323`决策记录已合main并push，origin/main SHA核对一致。
- [x] 实际main CPU31/31、tools57/57、off NLL .622784，0热暂停；53 shader与受测工作树完全一致。
- [x] balanced网页API config/status均200；动态mask/5500槽/双盘/k2/top4/ONECB1/CPUroute0/BF16，FP8明确0。
- [x] 六份原用户文件SHA/size/mtime不变、浏览器未刷新；此次恢复不新增速度数字。
- [x] 已保存受测工作树二进制和53 shaders到`compare_main_validation/tested_artifacts/`，删除own工作树`cachedmoe-compare-control`和分支`codex/compare-control`。
- 报告`verify_control.md`、方法/测量收据`verify_control_receipt.json`、实际交付`verify_control_delivery_receipt.json`。没有把新64-token控制写成完整MMLU或八轮生产加速GO；FP8和列宽候选按门槛止损。

## 37. MoE 常量循环探针（2026-10-07）

- [x] 六列存储、同三行验证只改常量循环：1.986774897→2.029320735ms/层，逐位一致，0热暂停，无收益。
- [x] 三列存储加常量循环这一新组合：1.987997219ms/层；不重复计时基线。没有达到减半2ms/cycle入口，不启动完整速度/MMLU门，也不声称提升。
- [x] 撤掉本轮准备的启动环境开关；只保留显式GPU探针的MoeSpec选项，默认false，没有配置绑定。生产mask/LRU、k2/top4、native BF16与CPU路由不变。
- 待收尾：实际main编译、CPU及GPU数值门、push、网页恢复和本轮工作树清理。完成后追加收据，不改写owner条目。
- 开放工作仍是有限prefill activation workspace；本轮没有实现，不把简单分段喂入当作正确分块。
- 报告：`docs/moe_static_columns.md`；机器收据：`docs/moe_static_columns_receipt.json`。
