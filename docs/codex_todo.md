# Codex 接续任务

更新：2026-10-05 23:39（Asia/Shanghai）。这是未完成事项清单，不是完成报告。
先读 [STATUS](STATUS.md)、[质量方案](mask_quality_plan.md) 和本文件，再核对进程及 Git 状态。

## 当前现场

- 主仓库：`/home/chenkailun/projects/cachedMoE`，main 已推到 `41cc44c`。
- 当前工作树：`/home/chenkailun/projects/deepmoe-mask-quality`，分支 `codex/mask-quality`。
  **在这里继续，勿新开重复工作树。** 构建目录是该工作树的 `build/`。
- 用户网页已停止，聊天记录保留。结束后须恢复；不要刷新用户的浏览器页。
- GPU 监督进程当时为 PID `2782881`，命令为
  `python3 bench/results/mask_quality/phase_a/validation/run_checks.py`（主仓库目录）。
  正在跑动态 mask 的生成式 MMLU57，之后自动跑双盘八轮 plain mask。
  PID 会变化，接续时重新检查，不能直接再启动一次。
- 原始结果均在主仓库 `bench/results/mask_quality/`；监督脚本、命令、环境和温度记录也在里面。
- **一次一个 GPU 任务**；power-saver、AC、80°C 暂停/72°C 恢复。镜像须为 `holds 48 of 48`。
  不得写 checkpoint。每种配置只跑一次，已有结果要复用。

## 1. 优先完成本轮代码审查修复

### KV 写盘

已提交 `210cb0f`，尚未合回 main。改动：队列超限改为背压，不再丢其他会话快照；
内存淘汰及退出保存等待原子写盘成功，失败返回错误并保留内存副本；普通轮末仍后台写盘。
三项新增 CPU 测试已通过：队列压力下所有会话落盘、必要保存返回错误、旧磁盘快照与新提示词的共同前缀检查。

仍需：

- [ ] 审核队列 coalesce/reset/失败路径，以及 `SessionPool::activate` 返回保存错误后的状态。
- [ ] 在最终构建上复用全部 `suite.kvdisk` 和工具门禁；必要的实机回归与下面 GPU 测试串行。
- [ ] 更新 `docs/kv_async.md`：队列满时允许等待；淘汰/退出必须确认写盘，不能再写“永不阻塞”。

### GPU 路由 / ONECB 整理

当前有未提交修改，不能宣称已验收：

- `RuntimeConfig::gpu` 集中解析 11 个 GPU 路由、ONECB、MGT 和 readout 开关；热路径读取配置。
- 捕获层号读取 `dspark_target_layer_ids`，噪声 token 读取 `dspark_noise_token_id`。
- 每层 readback 容器按配置定长；给路由记录和 host/shader ABI 偏移命名并交叉断言。
- 重排 `finish_gpu_routes`、`snapshot_with_shared`、`record_gpu_route` 和 DSpark 文件，补 snapshot/fence/union 顺序注释。
- 单线程 union 保留原顺序；**没有并行化 kernel，也没有宣称性能优化**。

仍需：

- [ ] 审核 diff，确保只是配置/常量/排版整理；核对参数默认值和原环境变量语义。
- [ ] 看 `gpu_review_build.log`、`gpu_review_cpu.log`；完成 CPU 与 Python 工具门禁。
- [ ] 当前 A 实验结束前不要覆盖 `build/deepmoe` 或 `build/shaders`，它们属于已跑/待跑 A 格。
- [ ] A 结束后完整构建；运行已有 GPU 路由 quant/hidden mean、ONECB 对 serial、KV 回绕、k=0
  单次 target 的逐位对拍。每个 case 必须实际执行，不能把 skip 当 pass。
- [ ] 整理成独立 commit，和数值策略/报告分开。

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

- [ ] 收尾正在跑的 MMLU57 和八轮 plain 基线，记录实际分数/invalid/速度/命中/重复率。
- [ ] 执行 `phase_a/validation/additional/run_checks.py` 的 decode 与 longctx 门禁（尚未启动）。
- [ ] 对 baseline 不合格项明确作判定；不能擅自降低门槛或设置未合格默认值。
- [ ] 把矩阵、P0 分段/读源计时、温控、命令和质量门写入 `docs/miss_mask.md`，更新 STATUS。

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

- [ ] 最终整理构建后更新 `phase_c/deepmoe-tested`；当前该 raw exe 是整理前版本，勿混用来源。
- [ ] 串行执行 `phase_c/run_checks.py`：off、tau0、tau.20、tau.10 的 l3 NLL。
  环境含双盘、`DEEPMOE_MASK_WAIT_BUDGET=0,0` 和工作树 shader 目录。
- [ ] off 必须 `.622784`；tau0 无限预算核对 off 等价性。
- [ ] 候选 NLL/off ≤1.10 才继续长生成、中文64、MMLU和八轮速度。
  已经判失败的配置按止损规则停止，写明省略哪些测试及理由，不冒充完成。
- [ ] 最好候选同工作负载带 off/全 mask 对照。速度 ≥20% 且全部质量通过才 GO；
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

- [ ] 从最终构建的 status 补完整内存账，避免把共享 UMA 的 RSS/GTT/cache 重复相加。
- [ ] 只有确证可安全腾出 ≥150槽（约3GB）才改分配，并验证 prefill/长上下文。
  没有足够安全空间则按方案跳过，记录依据。

## 3. 收尾与交付

- [ ] 提交本方案和简洁结论报告、机器收据。原始大数据保持 gitignored。
- [ ] 合回 main，验证、推送，再删除**本任务自己的**工作树与分支；不得删用户 untracked 数据。
- [ ] 更新本清单为最终未完成事项；本文件主仓库副本是任务新增文件，合并前处理同名 untracked 副本。
- [ ] 用主仓库 `build/web_mask/launch.py` 恢复网页（磁盘KV开启、1M上下文、5500槽、双盘、80/72温控），
  验证 `/api/config` 与引擎日志。用户 transcript 不动，浏览器页不刷新。
- [ ] 完成当前 goal 前确认以上必要项已处理；NO-GO / 按决策跳过必须有证据。

## 已完成的前置工作：不要重做

英文 README/UI、思考与正文分开、实际截图、异步轮末 KV 保存均已在 main。
Strata 可选 DeepMoE 文本后端 [PR #943](https://github.com/Niko1221/Strata/pull/943) 已提交并附到任务，
维护者尚未合并；本账号只有上游 READ 权限。等待维护者不是本地可自主完成的工作。
宣传稿在 `docs/launch_announcement.md`，GitHub About/topics 已更新；未自动发布社交媒体消息。
不要重新开启预测预取、非LRU、固定cache默认、自动冻结、重新归一化或已关闭 kernel 实验。
