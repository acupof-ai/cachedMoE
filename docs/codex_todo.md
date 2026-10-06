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
2. **网页默认：动态 mask + 投机。** owner 接受 A 当前质量（MMLU 46/57、续轮重复 .118）作为默认。
   - k 由 D 决定：在 performance 下测 k2/k3/k5（mask），选 ms/token 最低且无循环/重复门不失败的 k。
   - D 完成前网页先用 mask + k=2，不用 k=5。
   - ONECB 开；GPU route 按 §4.2 的 D 结果决定，无收益则关。
   - 网页仍保留 `off`/plain 显式选项。
3. 以上不影响 C：C 继续，按原门判断，GO 也只加显式选项，不改默认。

## 当前现场

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
- [ ] 执行 `phase_a/validation/additional/run_checks.py` 的 decode 与 longctx 门禁（尚未启动）。
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

- [ ] 将标定与边界测试收据写进报告；最终所有候选输出均列四项重复指标。

### C：加权部分 miss 等待，GPU 未验证

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
- [ ] 候选 NLL/off ≤1.10 才继续长生成、中文64、MMLU和八轮速度。
  已经判失败的配置按止损规则停止，写明省略哪些测试及理由，不冒充完成。
- [ ] 最好候选同工作负载带 off/全 mask 对照，**三者同一会话、同一电源模式、连续跑**，
  provenance 写电源模式。速度 ≥20% 且全部质量通过才 GO；
  质量通过但速度 <10% 则 NO-GO。GO 才添加网页显式选项，是否默认仍由 owner 决定。
- [ ] 报告写 `miss_mask.md` / STATUS；NO-GO 不进默认，可移除没有价值的运行时策略。

### D：重新选 k，未启动

- [ ] 复用 A 的八轮 plain；同双盘5500总槽、同脚本/seed/单会话测 k2/k3/k5。
- [ ] k2 再测 ONECB+GPU_ROUTE 同时关闭的一格；每配置只跑一次。
- [ ] 报 ms/token、draft/verify/commit/CPU 每周期、接受率、union、命中和重复指标。
  用相同接受率换算 cycle 成本；不要把接受率变化称作 kernel 加速。
- [ ] 标明 384 个 MTP pin 在5500槽内（主模型可替换槽5116），命中差只是不同生成路由的观测，
  不能完全归因于 pin。
- [ ] 只对最有希望且未被重复门判失败的候选补全部质量门；最快且合格才可作为网页默认。
  若 plain 最快就关投机；若无 mask 配置合格，明确说明，不选一个伪“通过”配置。
- [ ] 写进 `docs/dspark_topk.md` 与 STATUS，完成后按决策恢复网页。

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

## 3. 收尾与交付

- [ ] 提交本方案和简洁结论报告、机器收据。原始大数据保持 gitignored。
- [ ] 合回 main，验证、推送，再删除**本任务自己的**工作树与分支；不得删用户 untracked 数据。
- [ ] 更新本清单为最终未完成事项；先核对主仓库同名 untracked 副本是否有用户新编辑，再合并。
- [ ] 用主仓库 `build/web_mask/launch.py` 恢复网页（磁盘KV开启、1M上下文、5500槽、双盘、80/72温控），
  验证 `/api/config` 与引擎日志。用户 transcript 不动，浏览器页不刷新。
- [ ] 完成当前 goal 前确认以上必要项已处理；NO-GO / 按决策跳过必须有证据。

## 已完成的前置工作：不要重做

英文 README/UI、思考与正文分开、实际截图、异步轮末 KV 保存均已在 main。
Strata 可选 DeepMoE 文本后端 [PR #943](https://github.com/Niko1221/Strata/pull/943) 已提交并附到任务，
维护者尚未合并；本账号只有上游 READ 权限。等待维护者不是本地可自主完成的工作。
宣传稿在 `docs/launch_announcement.md`，GitHub About/topics 已更新；未自动发布社交媒体消息。
不要重新开启预测预取、非LRU、固定cache默认、自动冻结、重新归一化或已关闭 kernel 实验。

## 4. review 补充/纠正（23:53 版覆盖时丢失或遗漏的条目，勿删）

### 4.1 P0：速度回退未解释（新发现）

同为动态 mask、双盘、5500 槽、八轮 plain：

| | ms/token | tok/s | 命中 | nvme_stall | expert_hit_ms | hot_gemv_ms |
|---|---|---|---|---|---|---|
| 历史 `miss_mask/speed_mask` | 76.7 | 13.04 | .94 | 1.22 | 24.9 | 38.6 |
| Phase A `long8_plain` | 110.1 | 9.08 | .94 | 1.31 | 35.3 | 58.7 |

IO 没变，变慢的是 GPU 计算（+40~50%）。最可能是 power-saver/DPM：监督脚本把作业切到 power-saver，
历史数据大概率在 performance 下测。同会话 512 token 网页测：off 8.52/9.13/8.34，mask 10.19/8.91/7.84，
mask 只比 off 快约 4%（历史 +43%）。

- [ ] 在 performance 模式下同会话连续跑 off 与动态 mask 八轮（已有脚本，每配置一次），
  看能否回到 ~77ms/token。provenance 写 `powerprofilesctl get`、GPU 时钟/DPM 状态。
- [x] 若 performance 下仍慢，按提交二分（`41cc44c` → `4a262da`），以 per-op 的 `expert_hit_ms`/`hot_gemv_ms` 判断。
  收据：条件未触发，SKIP 二分。performance 七轮 mask attention/MoE/tail 为
  31.147/24.566/6.649 ms/token，历史八轮为31.853/24.896/6.734；指定算子退化已消失。
  原八轮对照尚未完整，墙钟受温控暂停污染，不能从这些算子数推断最终速度。
- [ ] 查清前，C 的「比 off 快 ≥20%」门和 D 的 k 选择都不判；温控仍按 80/72，
  若 performance 下温度无法跑完，写明并请 owner 决定测速用哪个模式。
  （owner 已定 performance，见 §0。）

### 4.2 P1：双盘 GPU route 让 verify 变慢（原 #4，被删）

`final_dual`：verify 169.7 → 182.8ms/周期（+13ms），同接受率周期 211.3 vs 211.8，无收益；
按 `dspark_e2e_plan.md` §8 是 >189 档。

- [ ] D 里 GPU_ROUTE 开/关分开测；双盘下若仍无收益，只保留 ONECB，GPU route 默认关，记为 NO-GO。
- [ ] 查 +13ms 来源（双盘时 snapshot/union 与 IO 完成的同步点），只查不改，结论写进 `dspark_topk.md`。

### 4.3 P1：网页投机配置

- [ ] 网页之前跑 k=5（约 65ms/token，可能比不投机还慢）。恢复网页时用 D 的结论；
  ~~D 未完成就用 plain~~ → owner 改为：D 未完成就用 mask + k=2，不用 k=5（§0）。
- [x] ~~mask 默认值、是否开网页投机都由 owner 决定~~ → 已决定：动态 mask + 投机（§0）。
  恢复网页后验证 `/api/config` 显示 mask、投机 k、performance 模式。

### 4.4 P2：代码质量（原 #10、#13 等，被删或只做了一部分）

- [ ] **一个 commit 一件事**：`4a262da` 混入了 `tests/test_io.cpp` +41 行；以后拆开。
- [ ] GPU route 状态移出 `Engine`：`4a262da` 只把 `saved_*` 改成按层数的 vector，
  `route_snapshot_`/`route_steps_`/`finish_gpu_routes` 仍在 `Engine` 里，未完成。
- [ ] 环境变量收进 `RuntimeConfig`：`engine.cpp` 的 `getenv` 由 47 降到 38，剩余项列清单，热路径上的优先。
- [ ] 魔数：核对 `L>=37`、`6*16*2`、`128799`、`opidx*336+80` 是否都已命名，未命名的列出。
- [ ] 新代码不要只追加到文件末尾；超长行与一行多语句按周边风格整理（与功能改动分 commit）。

### 4.5 P3：先测量再决定

- [ ] 单线程 union kernel：先用 per-op 计时看占周期多少，<2ms 就不动。
- [ ] profiling 开销 3.33%，目标 <1%；查是哪些计时点，默认关闭或降采样。
- [ ] draft head 约 10ms：FP8 head / vocab 子集只有估算，不实现，等 owner 决定。

### 4.6 不做（已关闭，勿重开）

预测预取（含 CPU 预测预取）、非 LRU 淘汰、reheat、resident-only 默认、固定 cache 默认、自动冻结、
mega kernel、host-flag、champion port、CM attention、scale-fold、pair-dot、MTP unpin、缩小 draft attention、
CPU 计算 miss 专家、部分专家、重新归一化、streams>1 的 GPU route。

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
