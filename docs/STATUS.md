# STATUS — deepMoE 今天在哪里

DeepSeek-V4.1-Flash（552B backbone + 196B Engram，decode 激活 16B，510 GB 权重）在一台
Windows Strix Halo（Ryzen AI Max+ 395 / Radeon 8060S / 128 GB LPDDR5X / NVMe）上的本地 decode。

这份文件是**唯一入口**。设计在 [design.md](design.md)（v0.9），但设计说的是"应该怎样"；
这里说的是**量到了什么、什么被扔掉了、下一步按什么顺序做**。任何数字在相信它之前，先看它右边的出处。
凡是没有出处的一律标 `未实测`。

> 方法论取自同一个 owner 的 `fleet-mi300x`：每一项优化由一份可分解的 per-token 时间线归因；
> 每一个实验——留下的和退掉的——都带编号记在这里；测试用变异注入审计；
> 每一个被保留的改动都要过"正确性 gate + 逐位确定性"两道闸。

---

## 目录

1. [今天的数字](#1-今天的数字)
2. [优化路径：每一步与它的归因](#2-优化路径每一步与它的归因)
3. [试过并退掉的（编号，共 58 条）](#3-试过并退掉的编号共-58-条)
4. [为什么 decode 是 NVMe-bound，而不是 kernel 慢](#4-为什么-decode-是-nvme-bound而不是-kernel-慢)
5. [测试套件](#5-测试套件)
6. [已知限制与未决风险](#6-已知限制与未决风险)
7. [Next, in order](#7-next-in-order)

---

## 1. 今天的数字

**它会说话。** 一个 `deepmoe serve` 进程常驻 pinned 集合、expert cache 与 KV；
`tools/chat.py` 起对话，流式出 token，新一轮只 prefill 延伸出来的那部分。
正确性对 fp32 参考在 64 token / 4K / 17K 三个上下文上都是 8/8。

| 指标 | 今天 | 出处 |
|---|---|---|
| **对话 decode** | **3.4–4.5 tok/s**（hit 0.84–0.90）；每 token ≈ 90–100 ms 计算 + **124–193 ms NVMe stall** | `design.md` §15.1.1 |
| 对话 decode（5500 槽 / 96.34 GiB cache，8-turn 脚本） | **6.05 tok/s，hit 0.9431，stall 46–71 ms** | `p4_summary.md` §6 |
| **热步**（expert 全驻留，64 token 上下文） | **81.5 ms = 12.3 tok/s**（`--cache-gb 24`，全 path A）；4K 91.1 ms；17K 94.2 ms | `p2_decode.md` §10 |
| **热步，2026-09-18 复测（Track PD）** | **102.0–106.2 ms**，41 submit，`--trace` 开与关一致。分项 attn **40** / moe gpu **38–40** / engram 4.9 / tail 6.0 / other 11。per-dispatch trace：**busy 85.23 ms + gap 16.84 ms = span 102.07 ms**，其中 **16.04 ms 是每层 MoE 前的 gate host 往返**，700 个非 MoE dispatch 的 barrier 一共 **0.80 ms** | 本节 §3 的 49；`bench/results/p4pd/trace_hot.bin` |
| **热步，2026-09-19（Track K1a 之后）** | **97.0 ms**，moe gpu **34.33**（`auto`，ABAB 三对，A 臂 sd 2.1%）。全 path A（`--cache-gb 24`）同日是 **92.7 / 30.97** | §3 的 55；`plan_p5.md` §3(h) 9.2 |
| **path 放置策略的两个方向都输了（Track K1b）** | 假设（§3 的 30：MoE 从 path B **读**慢）→ 让新 expert 落 path A：四轮对话 **−5.4%**。镜像（落 path B，因为往 path A **写**贵——Track Q2 的 4 MiB `ReadFile` **704 µs vs 70 µs**）：四轮 **+7.0%**（stall −6.78 ms）、八轮换题脚本 **−20.5%**（hit 0.914 → 0.879，fills 53,957 → 76,102，两对 ABAB）——因为它顺带把 path A 的 3,400 槽冻成一个 first-touch pin。**两个都 NO-GO，默认保持单一全局 LRU**。留下的是仪器（`hits_path_a/b`、`fills_path_a/b`，默认开）和两个数：path-B 的读今天值 **0.036 ms/expert-read**，而 path-A 的写值 **~6.8 ms/token** | §3 的 55；`plan_p5.md` §3(i)；`bench/results/k1b/` |
| **staged fill 关在 bench 上（Track S1）** | 假设：P0 的 miss 填进 path A 要付 **704 µs/chunk** 的锁页（Q2），换成「读进一个 GPU 可读的 pinned 主机环 + 一次 `vkCmdCopyBuffer` 进 path A」能省 5 ms/token 的 stall。`io_dst_bench` 新的 `stage` 档（`--gpu-copy`，端到端墙钟含拷贝，n=5）：QD 8 **4.7633 → 4.7689 GB/s（+0.12%）**、QD 24 **4.7040 → 4.7202（+0.35%）**，而 A 臂噪声底是 **0.69% / 0.40%**——判据 ≥5%，**差一个半数量级，运行时一行没动**。机制：Q2 的 8 条提交线程**已经**把 704 µs 挡在关键路径外——**`patha` 与 `pathb` 在发出时的队列深度一模一样**（6.87/6.87、22.02/22.04）。顺带补齐 Q2 §12.2 欠的那个数：`vkCmdCopyBuffer` path B → path A 18.8 MB = **0.145 ms（129 GB/s）**，8 个 region 一次 submit 反而 **0.201 each（93.5）**——**带宽受限不是提交受限**，「折进 MoE 命令缓冲」买不到东西。并且 **`gpu::Device` 只建一个 compute 队列**，拷贝没有第二条队列可藏：21.3 次 miss × 0.145 = **3.1 ms/token 的队列占用** | §3 的 56；`p4_p0_queue.md` §16–§21；`bench/results/s1/` |
| **decode 一直在跑 M=6 的 MoE kernel（Track K1a）** | **引擎从来没有创建过 M=1 的 pipeline**：`moe_bridge.cpp` 用 `spec.m = kMoeBatchMax`（=6）建 runner 好让同一个 runner 也能跑 verify 批，decode 只把 `live_columns` 设成 1。于是每个 decode token 用六个累加器去算一列。`MoeRunner` 现在带一套 M=1 特化（gate/up + h 量化 + down 必须一起换，fp8 h 平面的偏移是 M 的函数）：**moe gpu 37.87 → 34.33（−9.3%，六个格子无一交叠）、热步 101.8 → 97.0（−4.7%）**。`fb53514` 的 live-column mask 只是其中的 1.6 ms（microbench 0.6547 → 0.6028，−7.9%），**M=6 的形状是另外的 ~3.2** | `plan_p5.md` §3(h)；`DEEPMOE_MOE_STATIC_M1=0` 是 A 臂 |
| ✅ 上面两行差 **25%**，**已解释**（Track G） | **不是一件事，是四件。** ① **命令不同**：Track I 跑 `--cache-gb 24`，今天默认 `auto` = 5,100 槽，**51 个 slab 里 17 个在 path B** → 同一 binary 同一 session：`--cache-gb 24` **96.4 ms / moe gpu 33.8**、`--cache-gb 48` 96.6 / 33.8、`auto` **102.1 / 38.4**，**+5.5 ms**（§3 的 30 第一次在引擎上显形）。② **Track T 的 live-column mask**（`fb53514`，Track I 之后唯一动过 MoE kernel 的 commit）：`kernel_bench` 同 session、raw-read 相同，`fb53514^` **0.6035 ms/pair** → HEAD **0.6426**，M=1 **+6.5% = +1.6 ms/token**；同一 commit 把 M=6 压 1.56×，**而 M=6 是 §3 的 41/42 已经退掉的投机批**。③ `vkQueueSubmit2` 的 host 成本：同样 41 次 submit，`submit` **1.6 → 4.2 ms/token = +2.6**。④ 剩下 **+7.2 ms**（attn 36.0 → 39.0、moe 扣掉 kernel 漂移仍多 2.9、engram +0.7、moe host +0.6）**仍然没有解释**，与限制 6.5 是同一条线。合计 **+16.9**，对 81.5 → 96.4（`--cache-gb 24`）/ 102.1（`auto`）。⚠️ **2026-09-19 订正（Track K1a）**：②不是 1.6 而是 **4.8**——mask 只是它的三分之一，另外 ~3.2 是「decode 跑的一直是 M=6 的 kernel」；①因此不是 5.5 而是 **2.90 ms**（同日重测）。K1a 之后 `auto` 是 **97.0 / 34.33**，`--cache-gb 24` 是 **92.7 / 30.97** | `plan_p5.md` §3(g) 8.7 + §3(h)/(i)；`bench/results/p4g/ab_summary.csv` |
| **第二个读源落地（Track D2）** | 同一份 checkpoint 在一块 USB SSD（E:）上的逐字节副本，`--mirror DIR` / `DEEPMOE_MODEL_MIRRORS` 打开，**默认关**。路由是**加权最小在飞字节**（权重 = 启动时 1 s / 4 MiB / QD 8 的随机读探针），只路由 P0 与 P3。拷贝 **510 GB / 529 s = 964 MB/s**，94 个文件长度全对、5 个抽样 SHA-256 全对。聚合随机读（同一个 6.8 GB shard，4 MiB，QD 16）：D: 单盘 **3.453** → D:+E: **4.555 GB/s（+31%）**，分流 **78 : 22**，**D: 的份额一点没掉**（3.53 vs 3.47）。⚠️ **端到端没测成：这块 U 盘在引擎的负载下掉出总线**，见 §3 的 58 | §3 的 58；`p4_dual_source.md` |
| **gate 往返的 0.40 ms 花在哪（Track G）** | 热步全驻留、40 个层步、µs/层：**GPU 侧 gap 386.1**（trace）＝ host 侧 fence 之后 **220.9** ＋ 量不到的 **165.2**（fence 唤醒 + submit → GPU 起跑）。host 那 220.9 的分项：**`vkQueueSubmit2` 这一次调用 99.5**、MoE staging 44.7、下一层序言（bind + record_attention，它在同一个空窗里）32.6、`verify` + 读 top-16 **26.0**、录 MoE 9.7、planner 8.1、planner wait 0.35。**top-k 没有 readback**——gate kernel 早就写进 host-coherent 内存了。miss 层完全是另一回事：fence 之后 8,780 µs，其中 `pwait` **8,493 = NVMe** | §3 的 53；`DEEPMOE_GATE_PROBE=1`、`tools/trace_timeline.py --gate` |
| 热步分项（64 token） | attention 36.0 / MoE GPU 29.4 / MoE host 1.0 / engram 3.9 / tail 5.8 / stall 0.4 / other 5.0 ms | `p2_decode.md` §10 |
| 热步对设计地板 | 地板 **75.8 ms**，实测 81.5 ms = **+7.5%**；余量只剩 ~7% | `design.md` §13.4；`p2_decode.md` §10 |
| **每 token submit 数** | **41**（P2 step 2 是 ~128）；一次 submit 往返 **0.13–0.15 ms** | `p2_decode.md` §5.2/§10 |
| 每 token 常驻字节 | **8.52 GB**（`hot_bytes`，与 design §2.3 三位一致），在 216 GB/s 上 = **39.4 ms** | `p4_expert_patterns.md` §3 |
| 内存带宽上限（实测） | **215–218 GB/s**（path A 216.4 / path B 215.2 / DEVICE_LOCAL 216.0）= LPDDR5X-8000 理论 256 的 **84%** | `kernel_p1.md` §2.2 |
| 热步有效带宽 | **160 GB/s = 217 的 74%** | `p4_summary.md` §2 |
| **NVMe 有效读带宽** | **8.3–9.2 GB/s**，四种 cache 容量下都一样 | `p4_summary.md` §8 |
| 短 prompt TTFT | 5–23 s（11–46 token，走 decode 路径，1.5–3 tok/s） | README |
| 长 prompt TTFT（GPU prefill，冷 cache） | 64 / 4,133 / 17,010 token：**43 s / ≈150 s / ≈445 s**，compute ≈ **24 ms/prompt token** | `p3_prefill.md` §10.2 |
| **SSD KV 前缀复用** | 同一 4,133-token prompt：冷 **101.6 s** → 新进程 SSD 命中 **2.47 s（41×）** | `p4_test_report.md` §5 |
| 正确性（对 fp32 参考） | 64 token 8/8；4K / 17K 教师强制 8/8、自由运行 8/8 | `p4_test_report.md` §2 |
| tokenizer | 对 HF `tokenizers` 24,897 用例 / 7.69 M id **100% 一致** | README |
| 采样 | GPU top 集合 + 主机精确核；650 个采样步 **0 次回退** | README |
| KV（按模型格式） | 17,010 位置实测 **54.96 MiB**；64K 上下文 61 MB、1M 上下文 0.94 GB | `p4_summary.md` §6 |
| decode 逐位确定性 | 3 个进程同一 hash `a919aaf6…` | `p2_decode.md` §4.4 |
| **cache 默认（`auto`）** | **5,100 槽 / 89.3 GiB**：hit 0.9383、stall 76.0 ms/token、**5.603 tok/s**（八轮脚本，安静机） | `p4_hitrate.md` §4（F4） |
| cache 容量上限（本机） | **安全上限 5,000 槽**。5,400 过三轮、八轮中途死；**5,500 第一个 token 就丢设备**（36 A + 19 B slab，53.8 GiB 空闲）。~~5500 槽 = 96.34 GiB 可用~~ 作废 | `p4_hitrate.md` §4（F4） |
| 长 prompt TTFT（4,133 token，4,500 槽） | GPU prefill 被路径 A 饿死时 **1,061.7 s**；加 4 GiB 路径 A 预留后 **100.0 s（10.6×）** | `p4_hitrate.md` §5（F4） |
| **resident-only 路由的四档（4 轮对话，5,100 槽）** | `off` 4.82 tok/s（mass lost 0，P0 262.5 GiB）／`stall1` 5.14（0.048）／**`verify` 7.57 ×1.57（0.1377，P0 62.5 GiB）**／`all` 8.99 ×1.87（0.2587）。质量（64 步 teacher-forced PPL，×`off`）依次 1.00 / 1.11 / **1.376** / 2.29——**四档全部 NO-GO 作默认** | `p4_resident_routing.md` §8.3 / §9.3 / **§10** |
| **投机解码（DSpark），引擎实测** | 顺序 decode **102.6 ms/位置**；`forward_batch` M=6 **75.9（0.74×）**、M=1 123.7（1.21×）。草稿**白送**时每发出 token：`chain` k=5 **0.86×（更慢）**、k=1 1.06–1.13×；`longest` k=5 **≤1.07×**，而 `longest` **不无损**。加上 `T_draft` 19–42 ms/周期全部打平或更差——**本机 NO-GO** | `p4_dspark_runtime.md` §7；§3 的 41 / 42 |
| **M1 gate（`forward_batch` vs M=1）** | **不逐位**：前三层 cos = 1.000000000，L07 第一次门控翻转；60 个位置最差 cos **0.9398**、top-1 **54/60**。质量不变（teacher-forced PPL 1.8589 vs 1.8857 = 0.986×）。`suite.spec_forward` 因此**以 WARN 通过**，只在 cos < 0.93 或 top-1 < 50/60 时失败 | §3 的 41 |
| **P0 的盘时间是怎么花的（Track Q1）** | 4 轮对话、5,100 槽、backfill off：`nvme_stall` **105.2 ms/token**、21.3 个 miss = **383 MiB/token** ⇒ **3.55 GB/s**，而同一块盘 `nvme_bench` 是 **5.16 GB/s**。**0.0% 的 P0 在发出时有非 P0 的 chunk 在飞**（backfill 开也只有 8.2%、平均 1.02 个）——**队列争用 = 0**。一层这一批的第一个 P0 是 **2.83 ms**，后面的每个 **8.46 ms**；发出时平均在飞 chunk **3.80**（上限 8） | `p4_p0_queue.md` §2 |
| **那 30% 的缺口在哪（Track Q2）** | **不是盘，是 `ReadFile` 本身。** 往 path A（`DEVICE_LOCAL\|HOST_VISIBLE`、uncached）读，一次 4 MiB 的同步 `ReadFile` 要 **704 µs**，读进普通主机内存只要 **70 µs**（path B 导入的 69.6 µs）——内核要先 probe 住再锁住目的地的 1,024 个页。它过去**只在 dispatcher 一条线程上**发生：42.81 s 的 issue 里 **42.71 s 在 `Backend::submit`**，完成回调只占 0.13 s（1.6 µs/次）。**提交搬到 8 条线程 + P0 队列 24 / 96 MiB：4.828 → 4.998 tok/s（+3.5%）**，stall 108.1 → 100.9 ms，busy 窗口 3.90 → 4.10 GB/s。**并且 Q1 的参照系是错的**：`nvme_bench` 默认测的是 `%TEMP%`（C:），而 shard 在 **D:**，D: 的天花板是 **4.60–4.65 GB/s 而不是 5.16**，且对请求大小**不平** | `p4_p0_queue.md` §9–§13 |

| **两条并发对话的系统吞吐（Track MS）** | 一个引擎进程里两条 decode 流，按层交错（`Engine::decode_step_multi` 的 `pipeline` 档）：合计 **4.6474 → 5.4602 tok/s（+17.5%）**，ABAB 三对、cell 间 sd ≤0.83%，每路延迟 4.415/4.906 → 2.730 tok/s（**0.62×，吞吐换延迟**）。**token 级乒乓（D2）是 −5.1%**，所以收益确实来自重叠。盘的聚合速率 **2.21 → 2.83 GB/s（+28%）**，但两个工作集抢一个 5,100 槽的 LRU 让 `y_turns` 的 hit 掉 **1.6 pt**、MB/token 涨 **13.5%**，把一半收益吃回去；**N≥3 是负的**（5,100 槽装不下三个工作集） | `p4_multistream.md` §5 |

| **淘汰策略这个杠杆（Track E1）** | **关掉了。** 在 4.6 GB/s 的 demand-only 模型上扫了 127 种配置：最好的可实现策略是 `last_use + α × heat`（α ≈ 1,600–4,800），C=5,100 测试集 **+2.3%**（hit 0.9113 → 0.9151），折半后 **+1.2%**，低于 ±3% 的抖动带。**top-16 的"近似命中"分数只贡献其中的 0.06 pt（16%）**。`cache_sim` 里那条 `score-aware` 原样跑是 **−68%**（按 heat 排序会退化）。Belady 在同一口径下是 **+40%**——那 40% 在"未来"里，不在分数里 | `p4_cache_policy.md` §12；§3 的 47 |

**一句话结论**：`tok/s ≈ NVMe_eff / (MB per token)`。四种容量下有效读带宽恒定在 8.3–9.2 GB/s，
hit 0.59 → 0.84 把 MB/token 从 4,998 降到 2,269，tok/s 就翻倍。
**这一段的性能完全由"每个 token 要读多少字节"决定**，而不是由 kernel 决定（§4）。

---

## 2. 优化路径：每一步与它的归因

### 2.1 decode 热步：134 ms → 81.5 ms（Track I，`p2_decode.md` §10）

| 步骤 | 每 token 前 → 后 | 归因它的那次测量 |
|---|---|---|
| 基线（P2 step 2） | **134.0 ms**，~128 submits | 分项：attention 51.9 / MoE ~68 / tail 8.4 |
| **command buffer 只在 gate 处切开**（一层的 MoE + 下一层的 attention 同一次 submit） | ~128 → **41 submits** | `Engine::measure_submit_overhead` = **0.13–0.15 ms/往返** × ~128 = **~17 ms/token = 热步的 13%**。residency gate 变成 submit 内部的 semaphore wait，host fence 走第二条 timeline |
| **一个 MoE runner、七个槽**（6 个 FP4 routed + fp8 shared 在备用 index 384，`HQuant=3`） | MoE 1.72 → 0.73 ms/层 | §5.2 量到 MoE 是 1.72 ms/层，而 `kernel_p2_moe` 的 kernel 只要 0.68——**其中 ~0.75 ms 是 host 的 fp32 加法和两次 20 KB write-combining 读**。"最便宜的 40 ms" |
| 向量化 host `act_quant`（branch-free `fp8_round` + F16C） | MoE host ~28 → 6.4 → **1.0 ms** | 对 `cpu::act_quant_block` 5,120 个值逐位自检 |
| 修掉三处每次调用都分配、从不释放的 `VkCommandBuffer` | — | 每 token 泄 40 个（`run_attention` / `run_close` / `EngramRunner::run`） |
| **engram 行在 token 开始时取**，与 layer 0 的 attention 重叠；RoPE 表缓存 | engram 4.9 → **3.9 ms** | 每 engram 层少一次 submit；40 次 `bind` 每次都在算一张 RoPE 表，一步只需要三张 |
| **FFN 输入搬到 cached host 页（path B）** | −6.2 ms/token | 那一个 host 每层都要读的 20 KB 激活原本在 path A 的 write-combining 内存里，**读它实测 155 µs/层** |
| **合计** | **134.0 → 86.7 → 81.5 ms = 12.3 tok/s** | 重复：81.5 / 81.9 / 81.8 / 83.2 / 81.7 / 83.2 ms |
| **decode 走 M=1 特化的 MoE pipeline**（Track K1a，2026-09-19，在今天的 `auto` 基线上） | 热步 **101.8 → 97.0 ms**，moe gpu **37.87 → 34.33** | 引擎的 runner 是 `spec.m = 6`（为了 verify 批），decode 只设 `live_columns = 1`——六个累加器算一列。ABAB 三对，A = `DEEPMOE_MOE_STATIC_M1=0`；`plan_p5.md` §3(h) |

对地板：非 MoE 42.2 地板 vs 41.8 实测（**已在地板上**）；MoE GPU 25.0–27.3 vs 29.4
（第 7 个槽是真的 23.6 MB fp8 expert，不是 12.5 MB 的 FP4 替身）；engram 4.9 vs 3.9。
**剩下的是结构性的**：每 token 40 次 host 往返读 gate（~1.6 ms submit + ~2 ms fence 延迟），
以及 MoE 的 `x` 每层往 host 走一趟再回来。

### 2.2 attention：1,071 → 696 µs/层

| 阶段 | 一层（dispatch 1–9） | 40 层 + head | 做了什么 |
|---|---|---|---|
| P2 step 1 | **1,071 µs / 124 GB/s** | 48.5 ms | 基线 |
| P2 step 2 | **892.6 µs / 149 GB/s** | 41.4 ms | per-shader LDS 预算；act_quant staging 摊到整个 workgroup；per-stage `WaveReduce`；`sparse_attn` 向量化（141 → 69 µs） |
| + LDS bank-conflict 修复 | **797–804 µs** | 37.6–37.8 ms | **一个字的 padding**（`kXStride = 17`）——纯 layout 改动，输出逐位相同 |
| + K-split、8-tile attention | 709–711 µs | 34.0 ms | |
| **+ K-split、32×1 tile（P3 默认）** | **696.3–696.8 µs / 191 GB/s = 217 的 88%** | **33.5 ms** | K-split 默认：wq_a 2 / wkv 2 / wo_a 4 / wo_b 8，**一行一 lane** |

`wo_b` 的逐步账：312–314 µs（62%）→ +17 字 LDS stride **259–266（73–75%）**
→ +算术解码 / 4 累加器 / 折叠 scale **258–272（没变化）** → **+K-split ×8、一行一 lane：203.5–205.4 µs / 204–206 GB/s（94–95%）**；
K-split ×16 又退回 216–220。

路上修掉两个"亏 50×"的 bug：**被 push constant 界定大小的局部数组会落到 VRAM**
（Sinkhorn **216 µs → 2.8 µs**，gate rank 循环 55 → 2 µs）；**单 lane 串行 argmax**（gate top-k **56 µs → 12.6 µs**）。

⚠️ **Track J 的这条路径尚未被 runtime 采纳**（`design.md` §15.2）。
而且 LDS bank-conflict 修复**在真机上看不到**：attention 在 Track I 是 36.0 ms，Track Q 之后是 36.9 ms，
而 J 声称 −3.8 ms 且"已经生效"。这是一个未解决的矛盾（限制 6.5）。

### 2.3 MoE kernel（`kernel_p1.md` / `kernel_p2_moe.md`）

| M | 最佳变体 | A+B GB/s | 占上限 | ms/pair | **ms/token** |
|---|---|---|---|---|---|
| 1 | `L32 R1 xglob` | 222.6 | **102%** | 0.591 | **0.5912** |
| 2 | `L32 R1 xgf16` | 206.4 | 95% | 0.637 | 0.3187 |
| 4 | `L16 R2 xgf16` | 162.0 | 74% | 0.813 | 0.2031 |
| 6 | `L16 R2 xgf16/ldsi8` | 135.3 | 62% | 0.972 | **0.1621** |

ms/token **0.5912 → 0.1621 = 3.65×**——**这就是投机解码真正买到的东西**（每份权重换更多 token）。

保留的三项：**packed fp16（`XMode=4`）只上 dispatch A**（M=6 A 117.2 → 133.9 GB/s，+14%；误差 4.52e-4，远在 1e-3 内，"可以无条件采纳"）；
**int8 dot4（`xldsi8`）只上 dispatch B**（M=6 B 100.0 → 125.7，+26%）；
**`HQuant=3`**（h 量化挪进第三个小 dispatch）——把 decode 的量化税从 **6.3 ms/token 降到 ≈1.0 ms/token**，
对 `HQuant=2` 逐位相同（5.030e-08），M=1 的 +4.2% 全是那个额外 dispatch 和它的全局 barrier。

**诊断的转折**：P1 说"M=6 被激活载入指令卡住"是**错的**。`t_A(M) ≈ 0.33 + 0.055·M` ms
对得上一个指令计数（165 M FMA + 82.6 M `f16tof32`，7.4 Top/s，61% VALU 占用）：
**M≥3 是 VALU-issue bound，不是 memory bound**。所有省内存的设计都失败了，省指令的都成功了（§3 的 1/2/9）。
"≥80% raw-read"这个出口判据本身对一个 VALU-bound kernel 不合适，已换成 `ms/token ≤ 0.17`（实测 0.158）。

### 2.4 命中率与容量——decode 真正的杠杆

| 配置 | 槽数 | cache | decode tok/s | hit | MB/token | 有效 NVMe |
|---|---|---|---|---|---|---|
| cache-1000 | 1000 | 17.5 GiB | 1.830 | 0.5912 | 4,998 | 9.14 GB/s |
| cache-2200 | 2200 | 38.5 GiB | 2.593 | 0.7431 | 3,528 | 9.15 GB/s |
| cache-4500 | 4500 | 78.8 GiB | 3.647 | 0.8370 | 2,269 | 8.28 GB/s |
| **8-turn，5500 槽** | 5500 | **96.34 GiB** | **6.05** | **0.9431** | — | stall 46–71 ms |

**容量每翻一倍，hit +0.093**（0.5912 → 0.7431 → 0.8370）。
5500 槽实测的 0.9431 与 `cache_sim` 在 5711 槽的 0.9451 对上——**收益来自容量，不是策略**。

**F4（2026-09-18，`p4_hitrate.md` §0/§3/§4）把"策略"这一侧彻底关掉了：engine 就是纯 LRU。**
把一次 run 自己的 `route.bin` 回放进 `cache_sim` 的 LRU，engine 与模拟器在**每一步**上一致
（4,500 槽 2,489/2,489；5,000 槽 2,489/2,489；4 轮 `config_sweep` 214/214）。
所谓"engine 比纯 LRU 低 6.8 点"的前提**不存在**——那是拿一个 4 轮 92 步冷 cache 的 run
去比一个 8 轮 2,193 步的 run。把 `BACKFILL` / `PREFILL_HANDOFF` / `MOE_OVERLAP` 全关掉的
ablation 从另一侧确认：hit 0.9250 → 0.9252，MB/token 338.3 → 337.8，**没有东西可关**。
容量曲线在本机上限处**仍在爬**（sim：5,100 → 0.9383，6,500 → 0.9532），所以**每一点 hit 都是吞吐**。

路由事实（`p4_expert_patterns.md`）：`prefill_hit = 0.7607`，decode 选中的 expert **86.6% 在 prompt prefill 里已经出现过**
→ prefill→decode 交接是最大的单一杠杆。掉命中率的原因是**换话题，不是上下文变长**
（同话题每 128 步 hit 0.94–0.955，跨话题 0.877–0.90；stall 58 ms ↔ 151 ms）。
**没有小的热核**：出现在 ≥50% decode 步里的 expert 是 0.025/层 = 事件的 0.2%；~380/384 个 expert 都被碰过。
时间复用是 128 步尺度（63.7%），相邻 token 只有 15.4%——**相邻 token 预取没有意义**。

### 2.5 prefill

| N | TTFT | expert NVMe 等待 | compute | design §7.13.3 模型 |
|---|---|---|---|---|
| 64 | **43.2 s** | 33.7 s（94 GB） | 8.2 s | ≈34 s |
| 4,133 | **199.6 s（≈150 s 安静机）** | 44.5 s（198 GB） | 100 s | ≈62–65 s |
| 17,010 | **628.6 s（≈445 s 安静机）** | 28.2 s（203 GB） | 396 s | ≈78–88 s |

**模型漏掉的是 compute：24 ms/prompt token，线性，先验只给了 0.7 ms**；
§7.13.4 判据 4（±15%）在重拟成 `T_compute(N) ≈ 8 s + 0.024·N` 之前都是 fail。
17K 的 prefill 是 **compute-bound，不是 NVMe-bound**。

**prefill 唯一一个大的保留项**：host 的 top-6 每 token 通过 device-mapped 内存读 gate bias ~3,000 次——
**4K prefill 里 58 s（1.45 s/层）→ 1.1 s**，只是把 384 个 float 每层拷一次。

**F4 找到的更大一件事：GPU prefill 一个字节都分配不到（`p4_hitrate.md` §5）。**
slab 池会把路径 A 填到 `vkAllocateMemory` 拒绝为止，所以 **≥ 3,600 槽的 cache 之后，
晚一步分配、且只认路径 A 的 GPU prefill 连 21 MB 都要不到**，静默退回 decode 路径——
这就是为什么本文档此前每一格都报 `prefill_mode: "decode"`。
修法是 `Engine::build_expert_cache` 在建池期间持住 **4 GiB 路径 A 预留**（`kPathAReserve`，
按 2 GiB 一块，因为一次性 4 GiB 会撞 `maxMemoryAllocationSize`），建完立刻释放。
代价两个 slab，收益：**4,133-token prompt 的 prefill 1,061.7 s → 100.0 s（10.6×），
prefill 吞吐 3.89 → 41.3 tok/s，整格 19 min → 3 min**。

Track S 的 coopmat：N=4133 **legacy 103.67 s vs coop 99.89 s，只快 3.6%**，5× 目标未达成。

**F3（`p4_prefill_speed.md` v1.0）说清楚了那 3.6% 为什么这么小，以及 5× 为什么够不着。**
那次 coopmat 是**对的**（110 项 stage 全过），但它的 **P·V 那一段访存顺序是坏的**——
它按 dim 索引输出、按 entry 归约，于是一个 dim tile 以 16 KiB 的步长走一遍聚合后的 KV 平面、
**把它重读 32 遍**：13,119 ms → **3,032 ms**。另外三处同样是**网格/循环顺序**而不是算术：
每个 coopmat GEMM 固定 2 个 token tile/workgroup（n=512 时对 `wq_b` 的 84 MB 走 16 遍）、
mHC pre-norm **一行一个线程**（4,133 个线程，每个走 20,480 个 float）、
gate top-6 把 n × 384 个分数经 device-mapped 映射读回主机再 `partial_sort`；
再加上 `wo_a`——唯一还留在 tiled GEMV 上的大 linear，因为 `op_gemm` 对**分组** linear 拒绝走 coopmat 分支。

修完之后 4,133 token 的 **compute 是 80.6 s = 19.5 ms/prompt token**
（band attention 38.2 / routed expert GPU 27.1 / mHC 4.9 / shared expert 3.2 / engram 2.8 / gate+route 2.5 / 其它 1.9）。

**≥ 5× 在这台机器上低于算术地板，这是一句可以据此停手的话**：replay 128 下这个 prompt 是
**73.5 TFLOP**（光 routed expert 就 36.2），而这趟 prefill 里**任何** kernel 达到过的最好速率是
**2.1 TFLOP/s**（shared expert，最干净的大 GEMM；routed expert 只有 1.15–1.34）。
5 ms/prompt token = 20.7 s = **需要 3.56 TFLOP/s 持续**，是最好速率的 1.7 倍。
所以**目标不是靠"少算"能到的**——§6 那七项去work/重叠加起来值 12–15 s。
唯一能动速率的是把 `prefill_coopmat` stage 0 重写成
**多 wave 经 LDS 协作一个输出 tile + 下一个 K 切片预取**（今天是一个 workgroup 一个 32-lane wave、
tile 直接从全局内存读、没有 LDS 暂存也没有双缓冲）。**这是 F3 的建议，也是它没做的事。**

---

## 3. 试过并退掉的（编号，共 58 条）

这一节是这份文件里最有用的部分。**估计值系统性偏高**（fleet 那边是"二分之一法则"；
这里的同类现象见 23、25、30），所以任何基于字节数的估计**先砍一半**再决定要不要花一天。

### 3.1 MoE kernel

| # | 实验 | 实测 | 结论 |
|---|---|---|---|
| **1** | **LDS x-tiling**（`XMode=1/2`，把一块 K 的激活搬进 LDS 给所有 M 列共享） | M=6 L16R2：`xglob` **119.0 GB/s / 0.1844 ms/token** → `xlds` **95.5 / 0.2297**；dispatch B 单看 **100.0 → 72.5（−27%）**；`xldsf16` 更差 76.3。L32R1：110.4 → **42.6** | **退掉，规则删除**。原因：每个 K-chunk 两次 `GroupMemoryBarrierWithGroupSync()` 把内存流水抽干（A 每 wg 10 块，B 35 块）；`ds_read_b128` 和 `global_load_dwordx4` 都是读 16 B，指令数并没有少。design §7.1 规则 6 改写为"LDS 只在省指令时用，绝不配 per-K-chunk barrier"。**唯一例外**：`xldsi8` 只上 dispatch B（125.7 vs 100.0），保留 |
| **2** | **int8 预量化 x（`XMode=6`）**——§3.6 曾断言"唯一能到 85% 的设计" | dispatch A 稳定 **+15…+30%**（M=6 A 129.6 → **168.1 GB/s**）；但端到端 ms/token：M=1 **0.597 → 0.692（+15.9%）**、M=2 −6.5%、M=6 三轮 −7.0% / +0.7% / −0.6%。A+B 在 M=6 仍只有 137–147 GB/s = 上限的 63–67%（瓶颈搬到了 dispatch B，它的激活是 `h` 不是 `x`）。精度：expert (39,383) **8.85e-3**，超 design §12 的 ≤5e-3 | **实现了，spec constant 默认 0（关）**；"永远不要让它成为 decode 默认"。§7.9.2(d) 的断言撤回 |
| **3** | **"拆分 dispatch 每层要 +0.193 ms"** | v0.1：7 槽一次 vs 3+4，M=1 **0.596 → 0.789（+32%）** ⇒ "每 token 7.7 ms，§7.9 的『可忽略』错了 340×"。v0.2 用**轮转配置**重测：真实代价是 **+0.021…+0.047 ms/层**，"只拆 A" 是 **+0.006…+0.026**（有一行还是 −0.041）。40 层 = **0.2–1.0 ms/token，不是 7.7** | **v0.1 结论撤回**。自曝的证据：旧章节量到 `whole 0.652 < A 0.520 + B 0.290`，物理上不可能——三个数来自三个热态（段内漂移 8%、run-to-run 7%）。**"先到的先算"恢复为默认，只拆 dispatch A，B 最后跑一次**（逐位相同：5120/5120、30720/30720 个字） |
| **4** | **`HQuant=2`（`hq8`，dispatch A 直接写 fp8 h）** | 数值正确（5.030e-08），但强制 32 行的 workgroup，把 M=1 从 `L32 R1` 挤到 `L16 R2`：**0.591 → 0.749 ms/pair（+27%）= 6.3 ms/token**。`hqB`（`HQuant=1`）更差：M=1 **+45.5%**，M=6 `L32R1 xgf16 hqB` **+248%** | 两者都被 `HQuant=3` 取代（§2.3）。"`HQuant=2` 现在没有任何使用理由"。配套的 LDS 转置方案**被证明不可能**（`L32 R1` 的 workgroup 只拥有 8 行，一个 fp8 块要 32 行） |
| **5** | fp32 `h` | 输出误差 1.37e-4 → 1.15e-4（好 16%），但**慢 9%**（198.5 GB/s vs 218.5） | 退掉，h 保持 fp16 |
| **6** | **算术 / select-tree 的 FP4 解码**（`dec1` / `dec2`） | M=1 A+B：常量表 **218.5 GB/s / 0.602 ms**、算术 **164.2 / 0.802**、select 树 **151.4 / 0.869** —— 差 1.4× | 反直觉，"只有测量能说话"。**表解码是必须选对的那个旋钮** |
| **7** | `RowsPerLane ≥ 4`（M=1 / dispatch A） | M=1：R1 218.5 > R2 197.2 > R4 189.8。M=6 dispatch A 在 R≥4 崩溃：L16 R2 130.6 → R4 87.6 → R8 38.4 → **R16 11.0 GB/s**（寄存器 spill） | 早先"R 对 dispatch B 有帮助"的结论在干净重测下**被推翻**。A 和 B 共用 R；给它们各自一个 spec constant 估计 <2%，**从未做**（design §15 issue 6） |
| **8** | `LanesPerRow` 16/32/64 与 Wave32/Wave64 | LanesPerRow 只差 4%（32 最好）；Wave32 vs Wave64 **没有系统性差别（1–5%，方向混杂）** | design §7.1 规则 3（"16 lanes/row、256 B 粒度更好"）**未被证实**；规则 4（"优先 Wave32"）**既无支持也无反证** |
| **9** | 2 MiB 大页给 path B | `VirtualAlloc(MEM_LARGE_PAGES)` 需要 `SeLockMemoryPrivilege`，账户没有；代码回退 4 KiB 并记录原因 | **未实测 / 被环境挡住**。潜在奖品是 path B 那 12% 的 MoE kernel 赤字 |

### 3.2 attention kernel

| # | 实验 | 实测 | 结论 |
|---|---|---|---|
| **10** | **`heads_per_wg`（"KV 只读一次"）** | heads/wg = 1 / 2 / 4 / 8 → score+combine **69 / 100 / 169 / 288 µs**（L2 流量 41 / 20 / 10 / 5 MB）。LDS 修复后重测：**54 / 76 / 117 / 209 µs** | **永不采纳**。8 heads/wg = 8 个 workgroup 摊在 40 个 CU 上——沿着错误的轴换流量。用 head-group × KV-tile 的网格代替 |
| **11** | tiled attention 里的 `pv_heads_per_wg = 8` | **120 µs** vs 1 的 25–27 µs | 拒绝。P·V 无法共享一次 KV 读（一个 lane 拥有**一个** head 的 16 个**输出**维） |
| **12** | attention GEMV 里的算术 fp8 E4M3 解码（`DEEPMOE_FP8_ARITH`） | 精确（256 个码全同，L2 输出逐位相同）但更慢：`wq_a` 38.1 → **57.5 µs（+51%）**、`wq_b` +14%、`wkv.gemv` +35%；一层 799.5 → 861.8 µs | 关。**一张 256 项的 LDS 表随机索引，仍然打得过六条 ALU 指令** |
| **13** | 每块四个累加器（`DEEPMOE_GEMV_ACC=4`） | `wo_b` 259.0 / 266.5 µs（一个）vs 271.8 / 259.2（四个）：**没有差别** | 关 |
| **14** | 把 UE8M0 权重 scale 折进 staged block factor（`DEEPMOE_GEMV_FOLD_SCALE=1`） | `wo_b` 267.7 / 271.9：**没有差别**；和 K-split 合起来**更差**（`wo_b.ksplit×2`：**411 µs vs 303**） | 关 |
| **15** | **"`act_quant` 往返是 `wo_b` 的瓶颈"——被证伪** | 编译 `ActQuant=0`（算术是错的，只看时间）：**150.2 GB/s vs 开着的 151.7** | 那个本该是结构性修复的预量化 dispatch **什么也买不到** |
| **16** | 两行一 lane 的 K-split | 首测**四个 kernel 里输三个**（`wo_b` **304 µs vs 261**）；`wo_a` ×1（拆分 kernel 但不拆）**218.6 µs / 154 GB/s vs 194.2 / 173** | 只有在**一行一 lane**时才赢（§2.2 保留） |
| **17** | 更紧的 LDS 预算（"和 K 一样小"） | `wq_a` **5120 预算 152 GB/s vs 8192 的 166**；`wo_a` 同向 | 更多常驻 workgroup 各自重新 stage 一份宽激活 = 更多 L2 流量。只有 `wq_b` / `indexer.wq_b`（K=1280）受益 |
| **18** | `row_reduce` 改成一次 `WaveActiveSum` | `wq_b` **+30%**、`wkv` **+20%**，但 `wq_a` 和 `wo_a` **−14%** | 做成 per-stage 的 specialisation constant（`WaveReduce`）；**没有规则能预测谁赢**（design §15 issue 11）。在四个 K-split stage 上"没有可测变化" |

### 3.3 runtime / decode

| # | 实验 | 实测 | 结论 |
|---|---|---|---|
| **19** | Track T 收尾时顺手把 M=1 也换成 Track J 的 K-split / tiled attention | `kv_replay.l3_64` 场景 (1) 从 **8/8 变成 7/8** | **按正确性回退**（`decode_layer.{h,cpp}` 回到 `b4f0e83`）。Track T 自己的交付物不受影响 |
| **20** | **MoE live-column mask（`pc.m` / `set_live_columns`）** | microbench 是真的：M=1 MoE **1.786 → 1.044 ms/层（1.7×，40 层 ≈30 ms/token）**。端到端 A/B（4 轮、4500 槽）：tok/s **3.647 → 3.536**、hit 0.8370 = 0.8370、MB/token 2,269.3 = 2,269.3、TTFT 逐轮相同 | **没有可测差别（±3% 抖动）**。原因：30 ms/token 的 kernel 对着一个 **274 ms** 的 token（2,269 MB ÷ 8.3 GB/s ≈ 273 ms I/O 地板）——kernel 只占 **~2%**。kernel 里保留，但**"算得更快"的每一个改动都排在"读得更少"后面** |
| **21** | **每轮 reheat**（turn 边界衰减热度 + 重跑 P3 backfill） | 2200 槽、同话题三轮、on vs off：turn 3 **decode hit 0.7210 = 0.7210**。4500 槽：hit **+0.0016（噪声）**、tok/s **−0.018**、**MB/token +32**（它自己抓的 68 × 18.8 MB = 1.28 GB/轮） | **无收益，且略负**。在**饱和 cache 上按构造就是 no-op**（每次 reheat `free_slots = 0`）。别在短对话上开 |
| **21b** | （21 的副产品，两个真 bug） | (a) **decode 路径的 expert heat 恒等于 0**——`RouteDecision.near_ids/near_scores` 从来没被赋值，design §9.3 的 EWMA 循环从没跑过，"任何 score-aware 策略此前都在对全 0 排序"。已修。(b) `decay_heat(0.5)` 几轮后把整个 heat 场压到 0，绝对 5% 地板于是把**所有** expert 判成冷。已修（热端重归一到 1.0） | 记在这里是因为：**一个"无收益"的实验的真正产出，是它暴露的两个 bug** |
| **22** | 多轮 auto-tune / `DEEPMOE_HEAT_FILE` 顺序学习 | round 0 hit **0.8278** / 3.33 tok/s → round 1（读 round 0 的 heat）**0.8275** / 3.32 | 回路已打通，但 3 turn / 190 token 的短冷负载上**没有提升**（每轮都是冷 cache 开始）。需要更长的 idle backfill / 更长对话 / 跨轮 cache 保留 |
| **23** | **Lookahead 路由预取（design §9.4）** | 27,399-token trace 上**每一个 (d,K) 都是净负**。基线（4,787 槽、全局 LRU、无预取）**hit 0.8968 / 165.7 ms/token / 6.03 tok/s**。最好的一档（`head`, d=4,K=8）：effective hit 0.9052（+0.8 点）但 **282.3 ms/token / 3.54 tok/s**，NVMe 字节 **+60%**。最差（`tail`, d=6,K=16）：**1036.3 ms/token / 0.96 tok/s，precision 0.000** | **移到 design §16 明确不做**，`PrefetchConfig::lookahead_depth = 0`。"插在 LRU 尾部的探针是自毁的"，已钉成测试 `test_probe_at_tail_is_self_defeating`，**不要抄进 `store/planner.cpp`**。预测器 recall@6 在 d=1 是 0.66、d=3 是 0.52，到 d=8 没有拐点；而"上一个 token 选了什么"这个零成本基线（Jaccard 0.239）就等于 d=8 的 lookahead |
| **24** | 静态 pin routed expert / 每层配额 / score-aware / LFU-decay / ARC | LRU = LFU-decay = ARC 到小数点后四位都分不出；score-aware 只赢 **0.20–0.23 点**；全局比每层好 **0.6–0.8 点**；`static-pin + LRU` 在**每一个容量上都差 0.4–2.7 点** | 全局 LRU、无每层配额、静态 pin 删除、score-aware 做成默认关的开关 |
| **25** | **CPU 分担 expert GEMV（旧 P6）** | 内存控制器是一个共享的 ~217 GB/s 上限：GPU 单独 216、CPU 单独 100.9、并发 **GPU 185.8 + CPU 28.6 = 214.4** —— 是两者单独之和的 **67–68%** | 移到 design §16 明确不做 |
| **26** | **"在 expert 载入时做量化/变换"**（2026-09-17 审计） | 审计 `Planner::fetch` → `iocp.cpp` → `ExpertStore::finish_run`：一次 `ReadFile`（overlapped、`FILE_FLAG_NO_BUFFERING`）直接进槽的 host 指针，完成回调只做状态机，**整条路径上没有任何逐元素循环** | **这条路上本来就没有量化，所以没有可省的东西**。真正的嫌疑是**每次 miss 20–31 ms**（107 GB / 5,030 次填充）对着 8.3–9.2 GB/s 下 ~2 ms 的真实盘时间——下一步是给 `ReadFile` 两侧插桩，不是优化一个不存在的循环 |
| **27** | 预录制的 per-token command buffer（"dispatch 开销恐慌"） | design §3.4 猜每个 dispatch + barrier 要 5–20 µs；实测 **0.56–0.66 µs**（GPU），host 录一个 A+B 对 **1.1 µs**。~470 dispatch × 0.66 µs = **0.31 ms/token = 65 ms 的 0.5%**，不是 5–14% | 预录制 buffer **不再是性能必需**（作为 timeline 表达手段仍然想要） |
| **28** | 8 步内换更大的 cache | `--cache-gb 48`（2,700 槽，比这 8 步碰到的 1,920 对还多）hit **0.349**，不比 12 GiB 的 0.360 好 | 八步永远到不了稳态。**任何 cache 实验的最短长度是 128 步**（§2.4 的 ramp） |

### 3.4 内存路径 / 容量

| # | 实验 | 实测 | 结论 |
|---|---|---|---|
| **29** | **BIOS VGM（可变显存）调整** | `kernel_p1.md` §4.1 原本建议"最小 VGM + path B"（path B cache ~90–95 GB vs path A 在 VGM=64 时的 ~56 GB，代价 40 层 × 0.081 ms = **3.2 ms/token**）。撤回理由：**容量 = VGM + (可见物理内存 − 余量)，总量守恒**——调 VGM 只改变 A/B 的划分，不改变总量。而且两个堆是同一片 LPDDR5X（path A 216.4 / path B 215.2 / DEVICE_LOCAL 216.0 GB/s，raw read 相差 <0.5%） | **计划撤回，重启从未需要**。"任何『最小 VGM 下重测』的条目仍然作废"（design §16）。**不要再提这一条** |
| **30** | path B 给 MoE kernel 用 | raw-read 看不见的那 12%：A **221.8–222.1 GB/s / 0.592–0.593 ms**，B **193.6–198.2 / 0.664–0.680**，六轮交替完全可复现，而 raw-read 在两条路上都是 214.8–217.6。原因：GART 4 KiB page-walk 在 MoE 的 7-expert × 2304-row 访问模式下。反向：path B 的 NVMe 落地 **+22%**（4.8 vs 3.9 GB/s） | **kernel 用 path A，path B 只做容量溢出**。path B 只按物理内存定大小会炸：50 GB 空闲时要到 38 GiB，slab 导入在 ~33 GiB 返回 `VK_ERROR_INVALID_EXTERNAL_HANDLE`，**下一次 submit 发现设备丢失**；`VK_EXT_memory_budget` 不论跑什么都报 host heap 35.4 GiB / **0 B in use**。**path B 封顶 16 GiB** |
| **31** | 非临时存储（non-temporal store）进 path A | 20.1 → 21.5 GB/s，**噪声内**（path A 映射本来就是 write-combining）。只有 path B 受益（18.3 → **32.7 GB/s，1.8×**） | 两者都远高于 NVMe 的 4.7 GB/s，所以**不相关** |

### 3.5 投机解码（DSpark）

| # | 实验 | 实测 | 结论 |
|---|---|---|---|
| **32** | **树采样（K=16 格）作为提速手段** | 贪心，runtime replay tokens/verify：单链 k=1..5 **1.83 / 2.59 / 2.89 / 3.33 / 3.33**；树 K=16 `eal` **1.83 / 2.50 / 2.85 / 3.33 / 3.33**。每事件 E[tokens] 在 k=5：单链 **3.93**、K=4 3.93、K=8 3.87、**K=16 3.82**、K=32 3.87。采样（温度 1）：单链 1.78/2.38/2.74/3.05/3.22，树 1.77/2.39/2.70/3.02/3.22。TPS：**贪心 −0…−4%，采样 ±1%** | **作为提速手段 NO-GO**。只保留为温度 1 的 CPU 侧无损比较器（`accept_sampling_exact`）与 confidence 计算；路径目标退回 `chain`（= 单链）。配套否掉的：`base` 候选（接受率 **1.65 vs 2.93**，试跑 6 分钟就停）；`union2`/`union4`（覆盖率升到 3.99/4.13 但选中路径**并不更好**：2.87/2.85 vs 2.93）；精确全词表归一（改变 1–13% 的事件路径但 **≤0.1 token** 的接受长度，"要 1+4K 次 66 MB 的 GEMV，不值"） |
| **33** | **MoE 列式 dispatch（方案 A）作为 DSpark 的落地路径** | 每层 MoE（layer 0，每列 6 个不相交 expert，全驻留）：M=1 **1.786 ms/层**、M=2 3.615（2.02×）、M=4 7.484（4.19×）、M=6 **11.674（6.54×）**——**零摊薄**。verify 一层 ≈ C(M) 4.7 + MoE 11.7 ≈ **16 ms**，40 层 ≈ **640 ms**；同样 6 个 token 顺序跑 M=1 是 40 × (1.3+1.8) = **124 ms**。**列式 verify 比顺序 M=1 慢约 5×**。字节侧同样：M=6 每层读 36 × 18.8 MB ≈ 677 MB，40 层 ≈ **27 GB/batch** ≈ 3 s | 方案 A 只是正确性基线，**不是性能路径**。DSpark 的 TPS 预测应该用小 k（1–2），不是 k=5 |
| **34b** | **verify 批的并集能省下多少 miss（F1 实测，`tools/route_union.py`）** | **u(6) = 21.6**（不是外插的 26），而并集相对热缓存省下的 miss 是 **0** | 34 的那个"有条件 GO"现在挂在一个**实测为 0** 的数上。**go/no-go 不再是未知数**：只要 decode 还是 I/O 受限的，投机就先亏。能翻盘的只有"让 decode 不再 I/O 受限"，那是容量/命中率的事。顺带记下**没测过的**：`GpuMoeBridge::record_batch_union` 没有调用者因而从未被执行；`--spec` 没接进 CLI；"同 prompt、温度 0、`--dspark` 开关输出逐 token 相同"这条判据在真模型上没跑过 |
| **34** | **在一个 NVMe-bound 的 decode 上做投机，总体** | l3 的 M=1 是 1.28–1.33 ms/层 → 40 层 ≈ **51 ms**，但热步实测 **81.5 ms**；差的 ~30 ms 是 MoE+head，而 MoE 按需从 NVMe 取 expert。**verify 的 MoE 代价不随 M 摊薄，它随并集线性增长**——6 个 token 的并集 ≈26 expert/层。C(M) 曲线本身**过了**（M=6 最差 2.98×，判据 ≤3.06×），但 MoE(M) 没过。整体：**有条件 GO ×1.18–1.19**（confidence θ=0.5），刚过 1.15 门槛；固定 k 过不了（最好 ×1.14）；如果 head 退化成 M=1 循环（`T_draft` 42 ms）就掉到 **×1.11 = NO-GO** | `Engine::generate` 的 `speculative` 至今是 `unimplemented`。缺三个 kernel 能力（`p4_dspark_runtime.md` §2）。**投机解码在"权重读取"这一侧才有意义（§2.3 的 3.65×），但它的 MoE 并集把这个收益又吃回去了**——这是 P5 必须先关掉的那道门 |

### 3.6 其它记下来的

- **`XLayout` enum 从来没生效过**：`run_batch` 从不读 `layout`，`column_x` 两种模式都填满 6 列，所以 `Staged ≡ OneColumn`。**已删除**。底下压着的真 bug：host 把每一列都从 **y 的第 0 列**拷出来，而权重是按列 stage 的——第 0 列"恰好"逐位正确，1/2 列差 15–22% 的 |y|max。指纹：`per-column weights only` 的离散度修前是 **0**，修后是 **5.235e+00**
- **bf16 残差流实验**（会不会把第 6/7 个 expert 放回参考的顺序）——design §15 issue 3，**从未做**。第 2 层的近似平局：流入 cos 0.9997，MoE 输出 **0.9883866**，gate 5/6
- **`gate.slang` 的 softplus**：`log(1+exp(z))` 在 z≈−16 以下归零，max |gpu−cpu| **1.5e-4**，只影响分数 <1e-3 的 expert。一行的修复，**仍然开着**（design §15 issue 20）
- **`KvStore::clear()`** 每次 `reset_context` 在 path A 映射上逐元素写 163,840 个 `−inf`——§7.1 规则 10 的违反，**未修**（design §15 issue 26）
- **压缩 KV 存 bf16 而不是打包 FP4**：64K 上省 18 MB（66 vs 48 MB），代价是 `sparse_attn` 内循环里一次 nibble 解包。**故意推迟**
- **P2 step 2 的不可复现不是 kernel**——是**三条 track 往同一个 `build/shaders` 目录并发重建 `.spv`**。证据：step 2 报 step-0 margin **7.0927**，它自己那个 commit 隔离重建给 **6.7477**，带 Track F 的 kernel 给 **6.9389**——引用的那几次 run **两套 kernel 都没执行**。**这是"安静机"规则的由来**（`plan_p5.md` §1）
- **`MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL` 的 A/B**：设计好的表**从未跑过**（沙箱 `SetNamedSecurityInfoW failed (Win32 5)`）。`p4_hitrate.md` §4 至今是空的。**未实测**

### 3.7 P4 收尾四条 track 的否定（2026-09-18）+ Track Q1（43、44）

| # | 实验 | 实测 | 结论 |
|---|---|---|---|
| **35** | **预测式预取**（Markov / token-id 表 / token 内共现 / 隐状态 lookahead d=1..8），真实路由 trace、诚实盘模型、80 余种配置 × 253 次完整 replay | 在 4,500 / 5,711 槽下**全部净负或持平**，最好的一个 +0.8%。翻正需要 **precision = 1.00**：精度 0.9 仍比不预取**慢 15%**（5.47 vs 6.45 @ 5,711）。"最后四层"不特殊（逐层 LRU hit 0.856–0.918 几乎是平的，36–39 层 0.884–0.889；全知只预取最后 4 层只值 +5.5%）。整层 pin **一律变差**（pin 最后 4 层 −14%），每层静态热集 −0.5%，逐层配额 −3.9%，整层 streaming 要 90–361 GB/s。预测感知淘汰 ±0.1% | **不要实现**（`p4_cache_policy.md` §9.1）。根因：demand miss 本身每 token 就要 95–106 ms 盘时间，而计算窗口只有 80 ms——**盘已经欠着 25 ms 的债，浪费预算是负数**。天花板确实很高（全知预取 +50%、配全知淘汰 +101%），只是**够不着** |
| **36** | **第二块 NVMe（stripe 到 9 GB/s）vs 任何软件策略** | C=5,711 下 **+32%**，C=4,500 下 +40%，无预测、无风险 | **不是否定，是排序**：它比本项目测过的每一种软件策略都大。design §3.1 已经预留了 stripe。同表里**唯一没跑过的软件候选是 score-aware 淘汰**（用 top-16 的原始分数刷 heat）——Belady 上限是 +42%，它是最有可能吃到其中一部分的那个，**至今未测** |
| **37** | **2-bit routed expert** | 13 种方案。L3 端到端 **PPL 29 → 14,290,394，64 个位置上 argmax 0 次一致**；最好与最坏方案之间的差**小于任一个离"可用"的距离**。**3-bit 也过不了**（PPL 13.25M，top-1 1/64）。根因是结构性的：checkpoint 是 QAT 到 FP4 的，码流熵 **3.8375 bit** 对 log2(15)=3.9069 的上限，误差在行 / 块 / 矩阵 / 层 / expert 之间**摊得完全均匀**——**没有冗余可压，也没有显著子集可保护**。判据侧：**L1 上 expert 输出相对误差 0.31 端到端就已经致命** | **NO-GO，两个位宽都是**（`p4_quant2.md` §8）。它本来是本项目 decode 工作里最大的一个数（7.09 → 15.83 tok/s，×2.23）。**重定向**：那个 ×2.23 里**约一半其实是"多出来的槽"而不是"更窄的读"**——保住 FP4 的 hit 只把字节减半是 10.58 tok/s，而槽是不用动一个权重就能拿的。design §6"用 checkpoint 自己的精度，一个 bit 都不改"**第一次被认真挑战，活下来了** |
| **38** | **resident-only 路由（`all`）作为默认**：只路由到已在 cache 里的 expert | 64 步 teacher-forced L3：PPL **×1.82 – ×2.29**（判据 ×1.05 / ×1.30，三次独立进程），gate mass lost **0.3054**，**52 / 2,560 个 layer-step 只剩 shared expert**，对参考 top-1 42/64（`off` 是 61/64）。速度确实是 ×2.74，输出**不连贯** | **NO-GO**（`p4_resident_routing.md` §9.4）。原因是第 3 节那条硬预算：后台补盘 2,687 入队 / **2,072 过期丢弃** |
| **39** | **resident-only 路由（`stall1`，只在会 stall 时降级一个 expert）作为默认** | 质量 **×1.09 – ×1.11**（mass lost 0.076，**0 个只剩 shared expert 的层**，top-1 51/64），速度 harness 上 ×1.21、四轮对话里只有 ×1.07，代价 **1,624 次 P0 / 8.9 s 等待** | **作为默认 NO-GO**（质量 ×1.09 > ×1.05 的线，速度买不回来）。~~**但它的质量落在 ≤ ×1.3 带内，所以"只在 DSpark 的 verify 那一遍上用"的前提成立**~~——
**这一句被 40 推翻**：`stall1` 的 0.076 是**每一步都掏一次 P0** 买来的，而 verify-only 的四个 draft 位置一次也不掏。它同时推翻了 step 3 的判决（当时读成 ×3.40、比 `all` 还差），差别全在尺子上（§5.5） |
| **40** | **verify-only 的 resident-only 路由（`DEEPMOE_ROUTE_RESIDENT_ONLY=verify`）作为"投机开着时的默认"**：block-5 里第 1 位精确路由、4 个 draft 位只路由到已驻留的 expert，verify 那一遍一个字节都不为它们去盘上取 | 64 步 teacher-forced L3：PPL **×1.376**（判据 ×1.05 / ×1.30），gate mass lost **0.2099**，**17 个只剩 shared expert 的 layer-step**，top-1 49/64。速度这一侧是真的：四轮对话 **×1.57**（4.82 → 7.57 tok/s），**P0 字节 262.5 → 62.5 GiB（−76%）**，总字节 −19%，输出仍然连贯。反向的中间档（4 个 draft 位改成 `stall1`）质量回到 **×1.033 / mass lost 0.0649 / 0 个 shared-only**，但**速度只剩 ×1.06**，代价 7,687 次 P0 / **42.9 s** 等待 | **NO-GO**（`p4_resident_routing.md` §10）。它**推翻了 39 的那句前提**与 §7 第 3 项的读法：`stall1` 的质量是每步一次 P0 买的，把 P0 的机会从 5/5 降到 1/5，质量就从 ×1.09 掉到 ×1.376。能把质量买回来的那一档**让 verify 那一遍重新等盘**，方案的全部意义随之抵消。**注意口径**：投机解码在引擎里不存在（`Engine::forward_batch` 没写、`generate(speculative)` 是 `unimplemented`、`--spec` 没接进 CLI），所以量的是 verify 那一遍在 M = 1 上的等价物，而且该等价物**偏乐观**（真并集批的 2–5 位用的是取盘前的 cache 状态） |
| **41** | **`Engine::forward_batch` 与 M=1 decode 的逐位一致性（M1 的 gate）** | 尺子先立住：同一个 reseed 状态上 M=1 重跑 **64/64 逐位相同**。然后两条路径：前三层 **cos = 1.000000000（逐位相同）**，L03 差开 **3e-9**，L07 第一次**门控翻转**，L39 cos 0.962。64 步（60 个位置、块 5、warm cache、routing off）最差 cos **0.9398**、top-1 **54/60**、max\|dlogit\| 8.26。块 = 1 时最差 cos 0.9928——**所以这不是批边界，是 kernel 家族差 + 门控近似平局的放大**。质量没受影响：teacher-forced PPL **1.8589 vs M=1 的 1.8857（0.986×）**，参考 1.8177 | **gate NO，实现 GO**。`docs/p4_dspark_runtime.md` §7.2。直接后果：design §10.2 的硬不变式（温度 0、投机开/关逐 token 相同）**用这个 verify 前向做不到**。同时把 `p4_mgt1.md` §7 缺口 4（G3）关掉一半：机制隔离出来了。**ctest 口径**：`suite.spec_forward` 不在这条 gate 上失败——它把 cos / top-1 作为 **WARN** 打出来并通过，只在**回归地板**之下失败（cos < 0.93 或 top-1 < 50/60；实测 0.9398 / 54/60）；`DEEPMOE_SPEC_STRICT=1` 才断言原来的那条 |
| **43** | **"P0 独占盘"（非 P0 让路）作为提速手段** | 插桩量到 decode 里 **0.0%** 的 P0 在发出时有任何非 P0 的 chunk 在飞（backfill 开也只有 8.2%、平均 1.02 个 = `kBackgroundOpsWhileBusy` 那一个）。`DEEPMOE_IO_BG_CAP_BUSY=0`（P0 活跃时后台**完全停**）：4.846 → **4.830 tok/s（−0.3%）**，stall 105.8 → 105.4。把 engram（P2）也拉进节流（`DEEPMOE_IO_BG_THROTTLE_P2=1`）**是反的**：**4.846 → 3.500（−27.8%）**，且分项干净地指认原因——`nvme_stall` 不变（105.8 → 103.9）而 **`engram` 4.5 → 85.8 ms/token** | **NO-GO**（`p4_p0_queue.md` §3）。预测是「stall 111 → 75、tok/s +25%」，**实测 105.2 → 105.4、−0.3%**：能让的路今天的节流早就让了。附带一条**永久的「不做」**：**P2 engram 绝不节流**，它的 264 B 行读在关键路径上 |
| **44** | **给 P0 更深的队列 / 更大的 chunk** | `DEEPMOE_IO_P0_QD=32` + 256 MiB 在飞：发出时平均 QD 3.80 → 6.04，但**每个 P0 更慢**（6.70 → 7.70 ms），tok/s +0.6%（抖动带 ±3%，实测四次基线 ±0.5%）。`P0_CHUNK_MB=16`（一个 run 一个 op，正是 `nvme_bench` 里最快的形状）：**−8.4%**，stall +17%。反方向：2 MiB +0.9%、**1 MiB +1.4%**（P0 延迟 6.70 → 5.76，一层头一个 2.83 → **1.30 ms**） | **全部不作默认**（`p4_p0_queue.md` §3、§6）。1 MiB 方向一致、三次 B 对四次 A 全部为正，但 **+1.4% < ±3% 的判据带**。真正的发现在 §4：**盘对请求大小是平的**（1 MiB–18.4 MiB，QD ≥ 4 都是 5.07–5.16 GB/s），**引擎对它不平**——所以那 30% 的缺口在引擎侧（目的地是 GPU 可见内存，或 dispatcher 的补队速度），**不在盘上，也不在排队上**。旋钮（`DEEPMOE_IO_*` 五个）留着，默认等于今天 |
| **42** | **投机解码，用实测的 verify 代价重算（不再用 kernel 表）** | `bench.spec_forward_m_curve`（l3_64、5,100 槽、warm、先跑一遍不计时的顺序 decode）：顺序 decode **102.6 ms/位置**；`forward_batch` M=1 **123.7**（慢 1.21×）、M=6 **75.9（0.74×）**。并集实测 u(2..6) = 9.90/13.35/16.53/19.35/22.13，对 `route_union.py` 的离线值**误差 ≤ 2%**。接受长度（`tools/spec_longest.py`，71 个贪心事件）：`chain` E[tokens](k=5) **3.82**、`longest` **4.77**，链外 top-16 命中 **0.5419**。**草稿完全白送**时每发出 token：`chain` k=5 **0.86×（更慢）**、k=1 1.06–1.13×；`longest` k=5 1.06–1.07×。`T_draft` 19–42 ms/周期 = 每 token +4…+23 ms，**全部打平或更差** | **NO-GO，第三次，这次是引擎实测**。而且这还是 **compute-bound 的最好情形**（P0 = 0）；NVMe-bound 下 §6.5 的 [1.00×, 1.80×] miss 区间只会更差。`longest` 另外**不无损**：verify 第 i+1 行条件在链的 token 上，不是被替换进去的那个 |
| **45** | **「目的地是 GPU 可见内存所以 DMA 慢」**（Q1 §6 留下的候选 i），以及为它做 staging（读进 pinned 主机内存再拷进 path A） | `bench/io_dst_bench`：同一个 `IoEngine`、同一个 chunk、同一个 D: 上的 shard，**只换目的地**。QD 8 的吞吐 `ram` 4.722 / `pinned` 4.758 / **`patha` 4.511** / `pathb` 4.818 GB/s——**path A 只慢 5%，不是 30%**。真正差 8–10 倍的是 `Backend::submit`：**704 µs vs 70–122 µs**。staging 的代价实测：`memcpy` 18.8 MB 要 **0.88 ms（21.5 GB/s）**，每 token 21.3 次 miss = **18.7 ms**，比要救的 ~8 ms 还贵。「path B 优先」同样不做：提交并行之后 path A 的 4.21 → **4.80** 已追平 path B 的 4.82 | **候选 i 成立但读错了方向；staging / path B 优先 NO-GO**（`p4_p0_queue.md` §9、§12.2）。目的地内存**确实**是原因，但它贵在**提交**（锁页）而不是在传输，所以正确的解法是并行提交而不是换内存 |
| **46** | ~~**`nvme_bench` 的 5.07–5.16 GB/s 是这块盘的天花板，且盘对请求大小是平的**~~（Q1 §4） | `nvme_bench` 不带 `--file` 时把测试文件建在 `%TEMP%`——**C: 盘**，而 runtime 读的 shard 在 **D:**。同一条命令指到 D: 的一个 shard：**4.60–4.65 GB/s 封顶**，而且**对大小不平**（1 MiB @ QD4 只有 3.518，4 MiB 4.383，18.4 MiB 4.647） | **作废，Q1 的缺口要重读**：3.55 vs **4.60** = **23%**，不是 30%。Q1 §3 里「1 MiB chunk +1.4%」也换了解释——不是盘喜欢小请求（D: 对 1 MiB 更慢），是小 chunk 让那条**单线程的提交路**更快开始下一个 |
| **48** | **persistent-dispatch decode（plan_p5 §3(a)）的合法性**：`build/residency_probe`，8060S / Vulkan | 三段。**常驻上限 ~406 个 256-线程 workgroup**（32/64/128/256 全活，512 起 peak 卡在 406–413）。**跨 workgroup 握手：1,000 次往返里第 3 次就 TIMEOUT**——两个 workgroup（远低于 406 的上限）都不能保证同时前进，所以 fleet 的 1.36 µs 参照值**根本量不出来**。前进保证：2 / 8 / 32 / 128 组 `progressed`，512 / 2048 组 `TIMEOUT`（超上限，预期内） | **NO-GO，和 (b) 一样关闭**（`bench/results/p4pd/residency_probe.csv`）。plan_p5 §5.1 的三个判据里第 2 条直接不过：**自旋等待在这台机器上不可用**，device 侧任务队列 + 事件计数器的形状不成立。(a) 因此只剩「一张 dispatch 图」，而 49 说明那张图的收益是 **0.8 ms/token**。代价：十分钟 |
| **49** | **dispatch 融合 / 合并 command buffer 能拿到的上限（plan_p5 §3(c1–c3)、§3(a) 的退化形）** | 热步的 per-dispatch trace（`--steps 8 --warm 8 --trace`，741 dispatch/token、41 submit，7 个稳态 token）：**span 102.07 ms = GPU busy 85.23 + gap 16.84（16.5%）**。而那 16.84 ms 的**分布不是平的**：**16.04 ms 全部坐在每层 MoE dispatch 前面**（40 × 0.40 ms，命令缓冲在 gate 处被切开的那个 host 往返），attention / ced / tail 的 **700 个 dispatch 加起来只有 0.80 ms**（层内 barrier 0.4–2.2 µs 一个，与 §3 的 27 的 0.56–0.66 µs 一致）。逐层：普通层 17 dispatch、busy 1,895 µs、层内 gap **18 µs** + MoE 前 335 µs；源层 26 dispatch、busy 2,112 µs、层内 gap **24 µs** + MoE 前 271 µs | **c1–c3 划掉，(a) 的便宜形已经在树里**。「把一层的非 MoE 链录进一个 command buffer」**是今天的实现**（§2.1，~128 → 41 submit，只在 gate 处切开）；再融合任何 dispatch 的天花板是 **0.80 ms/token = 0.8%**，四分之一个 ±3% 抖动带。剩下的 16.0 ms 只有一个机制够得着——device 侧的 gate——**而 48 把它关掉了**。trace 本身是**免费的**：带 `--trace` 102.07 ms vs 不带 102.3–106.2 ms |
| **50** | **「读一次再发布」这一类共享读设计（plan_p5 §5.2，fleet 的 microbench (g)）** | `build/sharing_probe --buffer-mb 256 --max-groups 320 --disjoint`。**SHARED 的行是平的**：4 MiB working set 在 1 → 16 个并发 reader 上是 96.2 / 97.7 / 98.6 / 98.8 / 97.2 GB/s **每个 reader**，8 MiB 102.4 → 99.1，16 MiB 90.1 → 97.3；32 个 reader 才开始掉（92.0），80 个 83.9，320 个 31.4。DISJOINT（rawread 的切片）**同一行从 16 个 reader 起就崩**：4 MiB 91.4 / 94.3 / 93.8 / 91.6 / **59.0 / 29.7 / 20.5 / 10.2 / 5.1**——聚合带宽封顶，每个 reader 线性摊薄 | **共享是免费的（到 ~16–32 个 workgroup 为止），所以「读一次再发布」不可能赚**（`bench/results/p4pd/sharing_probe.csv`）。这一次性解释了 §3 的 10（`heads_per_wg` 1/2/4/8 = 69/100/169/288 µs）和 11（`pv_heads_per_wg=8` 120 µs vs 25）：它们慢不是因为重复读，重复读本来就不要钱。**封掉将来每一个同形状的提议** |
| **47** | **score-aware 淘汰**（§7 第 1 项的 (a)，本项目「唯一没跑过」的淘汰策略），以及 ARC / LRU-K / LFU-decay / S3-FIFO / TinyLFU 式准入，全部放进 Track X 的时间模型（`tools/cache_evict_study.py`，4.6 GB/s、demand-only、127 种配置） | 最好的可实现形状是 **`rank = last_use + α × heat`**（α ≈ 1,600–4,800，岭很平）：全量 trace **hit 0.9073 → 0.9120 @ C=5,100**，**测试集 +2.3%**（4,500 上 +2.7%，5,711 上 +1.8%）。**把 top-16 的原始分数整个关掉只掉 0.06 pt**。`cache_sim` 的 `score-aware` 原样（按 heat 排序）**hit 0.5345 = −68%**；LRU-2 / LFU-decay **−68%**；ARC **−1.5%**；按分数拒绝准入 −0.1%…−33%。Belady 同口径 **+40%** | **NO-GO**（`p4_cache_policy.md` §12）。折半后 **+1.2%**，低于 `p4_p0_queue.md` §3 的 **±3% 抖动带**——写出来也测不出来。**§9.4 ablation #3 的证伪条件实测命中**：top-16 分数没有可用信息，Belady 的 40% 全在「未来」里。机制：细粒度 LRU **已经就是**「最早可能的下一次使用」排序（第 L 层的 expert 最早也要等 40 个 layer-step，对每个 key 是同一个常数偏移），周期结构里没有免费信息。`store/planner.cpp` 的 `ScoreAwarePolicy` 保持回退 LRU，那条 TODO 改成「不做」 |
| **51** | **idle-window 预取，在 Q2 之后的 regime 上重算**（Track F6，`tools/idle_prefetch_sim.py`：4.6 GB/s、C=5,100、每层 1.3 ms 的 first-of-layer 斜坡、按 chunk 抢占、真 trace 的隐状态 lookahead + 两种合成降级） | 基线复现 stall **107.2 ms / 4.897 tok/s**（引擎 93–101 / 4.998–5.098，同模型内可比）。**诚实的 lookahead 全部为负**：lead=1（只能用 `pred_d2`，p=0.588）**−1.2…−1.9%**，lead=2 −3.5%，lead=3 −4.7…−5.1%。**Track X 的「盈亏平衡 precision = 1.00」被证伪**：均匀随机错误模型下平衡点是 **≈0.85**，改成真实的「近似错」形状（错的候选来自本层 top-16，而那些本来就驻留）后是 **≈0.60**——**那个 +78% 的悬崖是降级方法的性质，不是预测器的性质**。`provisional gate`（用第 L 层 pre-MoE 残差预测 L+1，买两层提前量）的**天花板**是 **+3.5%** | **NO-GO**（`p4_idle_prefetch.md` §5）。砍半后 **+1.8%**，低于 ±3% 的抖动带。**验收线从此是一个数**：两层提前量上 per-expert precision **≥ 0.77**（p=0.77 时 +9.9% raw / +5.0% 砍半），而最乐观的读数（真 post-MoE gate）只有 **0.674**。顺带订正一个 off-by-one：`route_trace.py` 的 `snapshot[L]` 是 `run_layer(L)` **之后**取的，所以在第 L 层开头能用的是 `pred_d(d+1)` 不是 `pred_dd`——搞错这一位会白拿一层提前量和一整档精度 |
| **52** | **「MoE / attention kernel 还有带宽余量」**（Track F6 的 roofline，全部由既有数据算出，没有新测量） | MoE：4.512 GB/token ÷ `moe_gpu` 42.14 ms = **107 GB/s = UMA(216) 的 50%**，看着有 16 ms；但 `kernel_p2_moe.md` §3.5 的 M=1 最优变体是 **222.6 GB/s = 上限的 102%**（0.5912 ms/dispatch 对，7 个 FP4 槽）——**kernel 已经打满**，`MoeRunner` 单层 1.044 ms 与引擎的 1.053 几乎相等，**缺口 18.5–21.9 ms/token 全在 dispatch / 间隙**（host 侧 `submit` 只有 3.73 ms / 56.5 次）。attention/dense：逐张量加出来 **8,522.8 MB/token = 引擎 `hot_bytes` 8,522,849,728 逐字节相等**，÷ `attn` 43.51 ms = **196 GB/s = 91% of UMA**，kernel 地板（`attn_bench` P3 33.5 ms + shared expert 6.53 ms）**40.0 ms** | **attention 这条线关掉**（`p4_idle_prefetch.md` §8）：已经在 80% 之上，「提到 80%」是**更慢 5.8 ms**，打满 100% 也只有 **+1.1%（砍半）**。**MoE 那 ≥5% 是真的，但它不在 shader 里**：压到 kernel 自己的速率是 **+5.2%…+6.3%（砍半）**，这笔钱记在 §7 第 4 项（per-dispatch trace）名下 |
| **53** | **「把 gate 的 host 往返本身压小」**（Track G，`plan_p5.md` §3(g)）——49 留下的那 16.0 ms 的最后一条路。新仪器两件：`DEEPMOE_GATE_PROBE=1`（host 侧按段计时，命中/未命中分行）和 `tools/trace_timeline.py --gate`（GPU 侧按层打印 MoE 前的那个 gap；在 PD 自己的 trace 上复现了 PD 的 102.07 / 16.04 / 0.80） | 热步全驻留、40 个层步、**µs/层**：GPU 侧 gap **386.1** ＝ host 侧 fence 之后 **220.9** ＋ 量不到的 **165.2**（fence 唤醒 + submit → GPU 起跑）。220.9 的分项：**`vkQueueSubmit2` 这一次调用 99.5**、MoE staging 44.7、**下一层的 bind + record_attention 32.6**（承载第 L 层 MoE 的那次 submit 在 `run_layer(L+1)` 里，所以 L+1 的序言坐在 L 的空窗里）、`verify` + 读 top-16 **26.0**、录 MoE 9.7、planner 8.1、planner wait 0.35。**(d) 不存在**：gate kernel 早已把 top-16 写进 host-coherent 内存，没有 readback 可省。攻击 (a) 落地为 `DEEPMOE_FENCE_SPIN_US`（轮询 timeline 计数器，超预算退回阻塞）：gap **386.1 → 325.2（−60.9）**，**其中只有 ≈15 µs 是唤醒延迟，45.4 µs 是唤醒之后整条 host 往返变快**（同一个 `vkQueueSubmit2` 99.5 → 76.0 µs，同一段 `vkCmd*` 9.7 → 4.1）——**park 的代价不主要是唤醒，是唤醒之后**。ABAB 三对：热步 100.3 → **98.0 ms（−2.3%）**；4 轮对话 5.0096 → **4.9594 tok/s（−1.0%，六个格子无一交叠）**。闸：`l3_ppl` off **NLL 0.630051** 逐位复现、`suite.decode` 8/8+8/8 | **NO-GO**（`bench/results/p4g/ab_summary.csv`）。理由是一个数：**0.40 ms/层里 265 µs 是驱动的**（submit 调用 99.5 + submit→起跑 ≈150 + 唤醒 ≈15），我们自己的代码只有 95 µs，其中 45 是 `act_quant`（归 §7 第 7 项的 c4）。**可动的三件加起来 103 µs/层 = 4.1 ms/token**，而判据要 **≥250 µs/层**——差 2.4 倍，还没砍半。`DEEPMOE_FENCE_SPIN_US` 留在树里、**默认 0**：热步 −2.3% 在 ±3% 之下，对话 −1.0% 方向是反的（自旋线程和 Q2 的 8 条 IO 提交线程抢核，而对话每 token 有 ~105 ms stall）。**要重开，前提和第 5 项一样：一条不用自旋的 device 侧 gate**（间接 dispatch + device 写 `VkDispatchIndirectCommand` + device 可见的 expert-id → slot 表），它能一次拿掉那 265 µs，因为它根本不 submit |
| **54** | **「热步 102 vs 81.5 那 25% 没有解释」**（Track G 顺带，§1 的 ✅ 行） | 四件事，前两件有 commit 有数：① **命令不同**——Track I 是 `--cache-gb 24`，今天默认 `auto` = 5,100 槽，**51 个 slab 里 17 个在 path B**；同 binary 同 session：24 GB **96.4 ms / moe gpu 33.8**、48 GB 96.6 / 33.8、`auto` **102.1 / 38.4** = **+5.5 ms**。② **`fb53514`（Track T 的 live-column mask）是 Track I 之后唯一动过 MoE kernel 的 commit**；`kernel_bench --quick --layer-cycle 8 --iters 48` 同 session、raw-read 216.3 / 216.7：`fb53514^` **0.6035 ms/pair（218.08 GB/s）** → HEAD **0.6426（204.80）**，M=1 **+6.5% = +1.6 ms/token**。③ `vkQueueSubmit2` 的 host 成本 1.6 → **4.2 ms/token**（同样 41 次 submit）= +2.6。④ 余下 **+7.2 ms**（attn 36.0 → 39.0、MoE 扣掉 kernel 漂移仍多 2.9、engram +0.7、moe host +0.6）**仍未解释**，与限制 6.5 同源 | ①**不是 bug，是一次做对的取舍**（`auto` 的 5,100 槽买的是 hit 0.673 vs 0.583，而 §2.4 说 hit 才是杠杆）——但 §1 的两行**必须标明它们是两条不同的命令**。②**是一笔该退的税**：同一个 commit 把 M=6 压 1.56×，而 M=6 是 §3 的 41/42 已经判 NO-GO 的投机批——**decode 每 token 为一个已经关掉的方向付 1.6 ms**。进 §7 第 7 项 |
| **55** | **「让热工作集迁移到 path A」的三个做法**（Track K1b，`plan_p5.md` §3(i)）——前提是 §3 的 30：MoE 从 path B 读慢。新仪器：`ExpertStoreStats` 的 `hits_path_a / hits_path_b / fills_path_a / fills_path_b`（引擎在建完 slab pool 后把 `a_slabs()` 交给 store） | **先给 path B 定价**（K1a 之后，热步 ABAB 三对）：`auto`（34 A + 17 B）**96.43 ms / moe gpu 33.87**，`--cache-gb 24`（全 path A）**92.67 / 30.97** = **−2.90 ms（−8.6%）**，即每次 path-B 的 expert 读 **≈0.036 ms**。**再看命中怎么分**：4 轮对话上 **path A 命中占 0.7861**，而 path A 的槽只占 0.6667——**LRU 本来就偏 A**，可动的只有 21.4% = **1.85 ms/token 的天花板，对着 240 ms 的 token = 0.8%**。**(i) 按原假设做是负的**：`DEEPMOE_EVICT_PATH=a`（先淘汰 path A，新 expert 就落在 path A）把 fills 从 10,197/5,305 推到 **14,093/1,700**，而 path A 的命中占比只动 **1.5 个点**（命中落在长寿常驻上，不落在刚进来的那批）；`expert_hit` −0.64 ms，`nvme_stall` **+3.94 ms**，四轮 decode **5.083 → 4.808 tok/s（−5.4%，三对）** | **四个做法全部 NO-GO，默认保持单一全局 LRU**。**(i) 的镜像先赢后输**：`DEEPMOE_EVICT_PATH=b`（新 expert 落 path B）在 4 轮对话上 **+7.0%**（5.0942 → 5.4517 tok/s，三对，A 臂 sd 0.06%；`nvme_stall` 141.14 → **134.36**、`expert_hit` +1.79、hit 0.8713 → 0.8760）——机制是**往 path A 写比读贵得多**（Track Q2：4 MiB 同步 `ReadFile` 落 path A **704 µs**、落普通主机内存 **70 µs**），而 decode 每 token 写 21–31 个 expert、只读它们几次。**但它顺带把 path A 冻成了 pin**：`fills path A = 3,400` 正好是 path A 的槽数——冷启动填满一次之后再也不淘汰，全部 churn 挤进 path B 的 1,700 槽。换一个会换话题的脚本（`long_turns.json`，8 轮中英混合，2,617 步）：hit **0.914 → 0.879**、fills **53,957 → 76,102（miss ×1.41）**、`nvme_stall` 96.99 → **130.88**、逐轮 decode **5.894 → 4.684 tok/s（−20.5%）**。它是 §3 的 24 / 35（静态 pin / 逐层配额）同一个形状，先验换成 first-touch 而已。**(ii) 迁移拷贝关掉**：18.8 MB 一次 ≈0.087 ms，要吃掉那 1.85 ms 得每步搬 6–7 个 = 1.15 ms，净 **+0.35%**。**(iii) 抬 path A 关掉**：`auto` 日志 `path A 62.26 GiB after 9.17 GiB pinned`，74 GiB 的 `DEVICE_LOCAL\|HOST_VISIBLE` 堆减 pinned 减 `kPathAOther` 3 GiB 减 `kPathAReserve` 4 GiB = 34 个 slab，而 commit 还剩 **133 GiB**——**不是 commit 在卡是堆在卡**；再要就得动 F4 量过的那两个预留（TTFT 10.6×），而 VGM 是 §3 的 29 已封的条目 |
| **56** | **staged fill**（Track S1，`p4_p0_queue.md` §16–§21）——45 判过一次「staging NO-GO」，理由是 CPU `memcpy` 0.88 ms；这次换成**不用 memcpy 的形状**：读落进一个 **path B**（`VK_EXT_external_memory_host`）的环（提交 70 µs 而不是 704），再用 **`vkCmdCopyBuffer`** 把它搬进 path A 槽。新仪器：`io_dst_bench --gpu-copy` / `--dst stage`，报的 `GB/s` 是**端到端**（墙钟含每一次拷贝退休），一条独占命令池的拷贝线程，**有未退休拷贝的槽不还给读** | 同一个 D: shard、请求 9,184 KiB、chunk 4 MiB、`patha` 与 `stage` 各 **n=5**（一次混跑 + 两次 ABAB 两对）：QD 8 **4.7633（sd 0.69%）→ 4.7689（+0.12%）**、QD 24 **4.7040（sd 0.40%）→ 4.7202（+0.35%）**；每请求 mean 7.775 → 7.771 ms、21.021 → 20.880，**尾延迟反而变差**（max 10.07–10.46 → 10.5–12.5，拷贝挂在尾上）。拷贝定价：`vkCmdCopyBuffer` path B → path A **18.8 MB = 0.145 ms（129 GB/s）**，8 region 一次 submit **0.201 each（93.5 GB/s）**——**批量更慢 ⇒ 带宽受限，折进已有命令缓冲省不到 submit**；CPU `memcpy` 同机今天是 0.72 ms（26.0 GB/s）。`ctest -LE needs-model` **25/25** | **NO-GO，闸在 bench 就关，运行时一行没动**（`storage/io_engine.*`、`runtime/engine.cpp` 未改，默认状态 = 改动前）。机制：**Q2 的 8 条提交线程已经把 704 µs 从关键路径上拿走了**——证据是 `patha` 与 `pathb` **发出时的队列深度相同**（6.87/6.87、22.02/22.04），锁页的 CPU 时间一个字没少（648–722 µs）但盘看不见它。**K1b 的 6.8 ms/token 因此是提交成本在突发里的残留**（`first-of-burst` 1.27 ms），不是提交成本本身，而本节的 bench 是稳态背靠背，量不到它。要重开需要两件：① **一条真正的 transfer 队列**——`gpu/vulkan/device.h` 今天只建**一个 compute 队列**，每次拷贝都和 decode 抢它（21.3 miss × 0.145 = **3.1 ms/token**，对着 moe gpu 34.33）；② 一个突发形状的 bench。留在树里的是仪器 |
| **57** | **多路 decode 的三件事**（Track MS，`p4_multistream.md`）——前提是 §4 的时间线：一个 token 是 ~97 ms 计算 + ~100 ms stall，两段不重叠，而两条独立对话之间没有数据依赖。新仪器：`runtime::Stream`（引擎按「进程的」/「序列的」切开）、`Engine::decode_step_multi`、`StepBreakdown` 的 per-step `requests/hits/miss_bytes`、`serve --streams N --ms-sched`、`tools/ms_bench.py` + `bench/ms_abab.py` | **(a) 按相分组的交错（`interleave`）：4.584 → 4.992（+8.9%）**——走到第一个 MoE 的时候这一轮所有 submit 都退休了，**GPU 在整个 stall 里是空的**；被 (b) 取代。**(b) 流水线（`pipeline`，留下来的那个）：4.6474 → 5.4602（+17.5%，三对，sd ≤0.83%）**。**(c) token 级乒乓（D2）：4.4123（−5.1%）**——没有重叠，只把「两个工作集抢一个 LRU」这项成本原样付了。**(d) N≥3：负的**——N=2→3 第一路 hit 0.861 → 0.830、MB/token 626 → 768（+23%），合计 5.199 → 4.807 | **(b) 落地并默认开**（`MsSched::Pipeline`，`--streams` 不给就是单流，单流路径与 main 逐字相同）；**(a)(c)(d) 保留为对照臂不作默认**。**预测 1.6–1.8× 没达到，差在哪是量到的**：盘的聚合速率确实涨了 28%（2.21 → 2.83 GB/s），但同时 hit 掉 1.6 pt、MB/token 涨 13.5%，而 2.83 只到 D: 突发天花板 4.10 的 69%——**剩下的缺口是每次 `wait_layer` 之前只压得进一条 attention 链（~2.1 ms）对着 ~3.4 ms 的 stall**，要盖满就得更多条流，而 (d) 把那条路关了。**下一次开这一条的前提是 MB/token 先降下来**（§7 第 1 项：第二块盘） |
| **58** | **第二个读源**（Track D2，`p4_dual_source.md`）——§7 第 1 项 (b) 的前半段。这台机器上没有第二块 NVMe，有的是一块 **USB 3.2 Gen2 的外置 SSD（E:，1.0 GB/s）**，所以量的不是「stripe 到 9 GB/s」而是「4.6 + 1.0」。新东西：`storage/source_router.h` 的**加权最小在飞字节** `argmin_s (outstanding[s] + bytes) / weight[s]`、`ShardSet::open_mirror`、`IoEngine::set_sources/add_mirror/probe_source_gbps`、`--mirror DIR` 与 `DEEPMOE_MODEL_MIRRORS`、`IoStats` 的 per-source 行（进 `status.json`）、`nvme_bench --mirror` | **拷贝** 510 GB / 529 s = **964 MB/s**，94 个文件长度全对 + 5 个抽样 SHA-256 全对。**聚合随机读**（同一个 6.8 GB shard，4 MiB）：D: 单盘 3.453（QD 16）→ D:+E: **4.555 GB/s（+31%）**，分流 **78 : 22**，**D: 自己的份额没掉**（3.53 vs 3.47）——慢盘的带宽是**净加**的。E: 单盘 QD 8 只有 0.494、QD 16 才 1.038。CPU `ctest -LE "needs-model;needs-gpu"` **34/34** | **默认关，端到端没测成，而且理由不是「没测到 3%」**。第一次是机器不是我的（另一条 track 的 serve 从 10:22 起常驻，commit 只剩 30/172 GB）；机器空了之后第二次，**E: 自己掉了**：off 臂正常跑完 144 s，第一个 on 臂 `pinned load of 'norm.weight': overlapped read failed`，第二个 on 臂**卡在 `48 shards open` 之后，12 分钟只用 2 秒 CPU / 8 线程 / 73 MB——全在等 I/O**；之后 `ls E:\` 挂住、`nvme_bench` 对 E: 挂住、`Get-Process`/`Get-Counter` 挂住，**`taskkill /F` 报成功但进程还在**。**这块 USB 外置盘在 decode 的负载形状（510 GB 持续写之后、48 个 `NO_BUFFERING\|OVERLAPPED` 句柄上的并发随机读）下会掉出总线**，要物理拔插才回来。**所以默认关的理由比判据更硬：一个会把整台机器拖进不可中断 I/O 等待的读源不能进默认路径。** 代码侧没查出问题（E: 还活着时 `nvme_bench --mirror` 四个 QD 点全跑满，路由/分流/计数全对；`io.` 11/11；CPU ctest 34/34）；`iocp.cpp` 的那条错误现在带 Win32 码 + 长度 + 偏移。harness 已提交：`bench/d2_abab.py`。预测留在 `p4_dual_source.md` §4.1（+14%，砍半 +7%，大概率落进 ±3% 带）。**§7 第 1 项那句「+32–40%」只对第二块真 NVMe 成立——读路径的代码已经就位，它等的是盘** |

---

## 4. 为什么 decode 是 NVMe-bound，而不是 kernel 慢

> **Track Q1/Q2 的订正（`p4_p0_queue.md` §4、§10）**：本节那句「没有 CPU 开销、没有共享显存写入的代价、没有排队，就是盘」在**一次 miss 的量级**上成立，在 **decode 的聚合速率**上不成立。排队这一条是实测排除的（0.0%）；**CPU 开销这一条是错的**——往 path A 内存发一个 4 MiB 的 `ReadFile` 要 704 µs 的锁页，过去全压在 dispatcher 一条线程上，一个 token ≈ 87 ms，占 105 ms stall 的绝大部分。并行提交之后引擎的 busy 窗口是 **4.10 GB/s = D: 天花板 4.60 的 89%**，那时才轮到「就是盘」。

这是这个项目最容易搞错的一件事，所以单独一节。

**热步（expert 全驻留）是 81.5 ms = 12.3 tok/s；对话里的真实 token 是 215–290 ms = 3.4–4.5 tok/s。**
差的那 130–190 ms 全是 NVMe stall。四种 cache 容量下有效读带宽**恒定在 8.3–9.2 GB/s**，于是：

```
tok/s  ≈  NVMe_eff (8.3–9.2 GB/s)  /  MB per token
```

- hit 0.5912 → MB/token 4,998 → 1.83 tok/s
- hit 0.8370 → MB/token 2,269 → 3.65 tok/s
- hit 0.9431 → stall 46–71 ms → 6.05 tok/s

**推论 1：每一个"算得更快"的改动都要先除以它在 token 里的占比。**
MoE live-column mask 在 microbench 上是 1.7×（30 ms/token），端到端**测不出来**（§3 的 20），
因为它是 274 ms 里的 2%。

**推论 2：容量、顺序（heat）、投机解码是同一件事的三种做法**——都是在降 MB/token。
其中容量是唯一一个被证明有效的（§2.4：每翻倍 +0.093 hit）；顺序（§3 的 22、23）无效；
投机解码（§3 的 33、34）在 MoE 并集上把收益吃回去了。

**推论 3：热步那 7% 的余量（81.5 vs 地板 75.8）不值得先做。**
它是每 token 的 5.7 ms，对着 130–190 ms 的 stall。
但**一旦 hit 上到 0.95+，stall 降到 ~40 ms，热步就重新变成主项**——那时候 §7 的第 3、4 项才有意义。

~~**没量过的那一块**：每次 expert miss 花 20–31 ms，而 8.3–9.2 GB/s 下 18.8 MB 的真实盘时间是 ~2 ms，
这 10× 的差没有解释。~~ **已解释，就是盘本身**（`bench/nvme_bench`，`p4_hitrate.md` §10）：
18.8 MB 的**随机**读 QD=4 是 **5.19 GB/s / 13.6 ms 每请求**、QD=1 是 4.47 GB/s / 4.05 ms，
一层 6 个 expert 全 miss 就是 **21.7 ms**——和 20–31 ms 对得上。
那个"~2 ms"是拿 decode 跨多个并发 fill 的聚合带宽去除单个请求算的，口径错了。
**没有 CPU 开销、没有共享显存写入的代价、没有排队，就是盘。**
顺带否掉三件看起来可行的事：离线重打包（checkpoint 已经是量化格式，而且 17.7 MB 的请求已经跑在 5.2 GB/s，
说明没有按请求的固定开销可省）、减少对齐浪费（18.8 MB 里 3 KB，0.02%）、提高 QD（1→4 只有 +16%）。
**结论**：decode 的每 token 时间 = `MB/token ÷ 5.2 GB/s`。**只有少读字节这一条路。**

⚠️ **2026-09-18 收窄（Track Q1，`p4_p0_queue.md`）**：上面这句里的「没有排队」**现在是实测的**
（decode 里 **0.0%** 的 P0 在发出时有非 P0 的 chunk 在飞），但「就是盘」**只成立于一次 miss 的量级**。
聚合速率上不成立：引擎在 stall 窗口里拿到的是 **3.55 GB/s**（383 MiB / 105.2 ms），
而 `nvme_bench` 在同一块盘、同一请求大小上是 **5.16 GB/s**，而且**盘对请求大小是平的**
（1 MiB–18.4 MiB，QD ≥ 4 都是 5.07–5.16），**引擎对它不平**（16 MiB 慢 8.4%、1 MiB 快 1.4%）。
所以那 30% 的缺口在**引擎侧**——目的地是 GPU 可见内存，或 `issue_ready_chunks` 的补队速度
（上限 QD 8，实测发出时平均只有 3.80）。这两条**本轮没有分开**，是 §7 里新的一项。

---

## 5. 测试套件

一条命令跑完所有不需要 GPU / 不需要 checkpoint 的 gate：

```powershell
.venv\Scripts\python.exe tests\run_all.py            # 全部 CPU gate，一个退出码
.venv\Scripts\python.exe tests\run_all.py --mutate   # 再加变异注入这一道闸
```

需要 checkpoint / GPU 的另一半由 `ctest` 跑（命令见 §5.3）。

### 5.1 纯 CPU（`run_all.py` 覆盖）

| 套件 | 验什么 | 规模 |
|---|---|---|
| `suite.align` / `suite.layout` / `suite.types` / `suite.json` / `suite.status` / `suite.bytes` | `core/`：4 KiB 对齐、`Result<T>`、JSON 读写、字节记账 | 单元 |
| `suite.profiler` | per-token JSONL 记录与汇总 | 单元 |
| `suite.dequant` | FP4 E2M1 / FP8 E4M3 / E8M0 解码表 | 单元 |
| `suite.gate` | router 数学（noaux_tc top-k） | 单元 |
| `suite.slab` / `suite.expert_store` | slab 池、每 expert 6 项的 GPU 指针表 | 单元 |
| `suite.planner` | 全局 LRU（与 `cache_sim` 逐步相同），含"尾部探针自毁"的钉子 | 单元 |
| `suite.io` | 优先级队列、切分、QD、忙碌记账 | 单元 |
| `suite.v41_config` / `suite.manifest` / `suite.block_info` | `config.json`、`deepmoe_manifest.json` v2（run/skew）、与 `layout.h` 的交叉校验 | 单元 |
| `suite.kvcache` / `suite.kvstore` | KV 平面几何、ring 解析 | 单元 |
| `suite.dspark_tree` | 树采样的格 / 路径 / confidence / 精确接受，**对 Python 参考逐位** | 单元 |
| `suite.sampling` | L3 logits 上 **200,000 次抽样的 χ²** | 单元 |
| `suite.kvdisk` | SSD parked-context 存/取/丢 round-trip | 单元 |
| `suite.resident_route` | resident-only 路由的选择逻辑（`runtime/resident_route.h`） | 单元 |
| `suite.speculate` | 投机周期的算术：位置对齐、接受、回滚记账 | 单元 |
| `suite.trace` | per-dispatch trace 的格式与解析 | 单元 |

### 5.2 需要 checkpoint 或 GPU

| 套件 | 验什么 | 今天 |
|---|---|---|
| `suite.tokenizer` | 909 个黄金用例（全量 24,897 个见 `tools/tokenizer_golden.py`） | pass，对 HF 100% |
| `suite.engram_tables` | engram hash 常量在 C++ 里的推导 | pass |
| `suite.integration` | 读路径端到端 | pass |
| `suite.gpu` / `suite.gpu_moe` / `suite.gpu_attn` | MoE kernel、attention 逐 stage 对 L1/L2 oracle | pass |
| `suite.gpu_layer` | 整层对 oracle（chained block cos ≈0.9999，gate 6/6）；`mgt1_layer_batch_vs_steps`（M=2/4/6 批 vs 逐步） | pass |
| `suite.decode` | 四十层、八步对 L3 oracle | pass：从自己的慢 prefill 教师强制 8/8、自由运行 8/8 |
| `suite.decode_longctx` | 4K / 17K 的 indexer kernel tie-aware、引擎逐层逐步、8 教师强制 + 8 自由运行 | pass（需要 `DEEPMOE_LONGCTX_DIR`） |
| `suite.gpu_prefill` | prefill 逐 stage **110 项**，worst cos 0.999912 | pass |
| `suite.gpu_dspark` | DSpark 草稿 kernel 逐阶段对参考 | pass |
| `suite.kv_replay` | KV 回放/回退记账 | pass（l3_64：(1) 8/8；(3) restore ≈52 s + 8/8；(4) 0 raw rows） |
| `bench.l3_ppl64` | **64 步教师强制 L3 PPL 尺**，三档路由各起一个进程串行跑，出 NLL / PPL / top-1 / served / mass lost 与判据 | 有导出集（`traces/l3_64`，.gitignore）才跑，否则 skip；§5.5 |
| `bench.mgt1_m_curve` / `bench.mgt1_moe_m_curve` | C(M) 曲线 | **未产出**（`bench/results/mgt1_p4.csv` 缺） |
| `suite.spec_forward` | **M1**：`Engine::forward_batch` 对 M=1 decode（64 步 teacher-forced、块 5、逐层 bisect、环回滚逐字节） | pass **带 WARN**（gate 不逐位：打印 cos 0.9398 / top-1 54/60；默认判据是回归地板 **cos ≥0.93、top-1 ≥50/60**，加 PPL 比 ≤1.05×；原 gate 在 `DEEPMOE_SPEC_STRICT=1` 后面，**它不过**，§3 的 41）；要 `traces/l3_64` |
| `bench.spec_forward_m_curve` | verify 前向在引擎里的 M=1..6 代价、并集大小、P0 字节，对顺序 decode | 出表（§3 的 42）；要 `traces/l3_64` |

**本轮合并后的全量 gate（2026-09-18，`p4/one-pr`，安静机，`ctest -j 1`）**：

```
100% tests passed, 0 tests failed out of 41
Total Test time (real) = 3514.80 sec
The following tests did not run:  25 - suite.gpu_layer (Skipped)
                                  34 - suite.gpu_prefill (Skipped)
                                  41 - bench.l3_ppl64 (Skipped)
```

三条 Skipped **都不是失败，是可选子用例的 skip 正则命中了整条 ctest 项**，直接跑二进制可以看到：

- `gpu_layer`：2 个用例过（层 0 / 39 的 chained `block_out` cos 0.99993 / 0.99998，gate 6/6），
  只有 `mgt1_layer_batch_vs_steps` 的两档因为这棵工作树里没有 `traces/mgt1` 而 skip。
- `gpu_prefill`：5 个用例过——**110 项 stage 检查 0 失败、worst cos 0.999912**，
  `forty_layers` 自由运行 **8/8**，`longctx` **8/8**（这是 F3 改了 prefill kernel 几何之后的复核）；
  只有 opt-in 的 `gpu_prefill.repeat` 要 `DEEPMOE_PF_REPEAT=1` 才跑。
- `bench.l3_ppl64`：导出集 `traces/l3_64` 只在主 checkout 里（`traces/` 是 .gitignore 的），
  按设计 skip。

`tests/run_all.py`：**28/28 gates passed**。

### 5.3 命令

```powershell
# 纯 CPU（无 checkpoint、无 GPU）
ctest --test-dir build -LE "needs-model|needs-gpu" --output-on-failure

# 全部
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'
$env:DEEPMOE_LONGCTX_DIR='C:\Users\Asus\code\deepmoe\traces\longctx'
ctest --test-dir build --output-on-failure
```

### 5.4 套件本身的审计（变异注入）

一个全绿的套件本身什么也不证明。`tests/mutate.py` 往测试**声称覆盖**的代码里注入真实的回归，
重新编译，跑那一个套件，**要求它失败**。今天 14 条，**14/14 被抓到**：

| 注入 | 套件 | 结果 |
|---|---|---|
| 交换 FP4 E2M1 表里两个幅值 | `dequant` | caught |
| FP8 E4M3 指数偏置差一 | `dequant` | caught |
| 丢掉 run 的 skew（张量在对齐 run 内的偏移） | `manifest` | caught |
| `min_bytes` 允许读过文件尾 | `io` | caught |
| 后台 op 节流放宽一个 | `io` | caught |
| 淘汰最近用过的而不是最久没用的 | `planner` | caught |
| 反转 LRU 的确定性 tie-break | `planner` | caught |
| BPE merge 平局取最右而不是最左 | `tokenizer` | caught |
| 核采样多吃 2% 的质量 | `sampling` | caught |
| 在 top 集合证明不了的时候声称 nucleus 精确 | `sampling` | caught |
| 窗口环的 slot→position 映射偏一代 | `kvstore` | caught |
| 把序列从未到过的槽当成已占用 | `kvstore` | caught |
| engram 乘子改成偶数（参考强制奇数） | `engram_tables` | caught |
| 改 engram 的每层种子 | `engram_tables` | caught |

**这次审计抓到的真问题**（第一轮是 10/14，四条活了下来）：

1. **`suite.kvstore` 对窗口环的 slot→position 算术零覆盖。** 唯一在跑它的是
   `suite.kv_replay`——而那条同时要 checkpoint **和** GPU。
   也就是说，一个"replay 用错误的位置重建窗口"的 bug 可以通过每一道 CPU 闸。
   **已修**：算术抽成纯函数 `KvStore::ring_slot_position`，`resolve_ring` 调用它，
   并加了一个 CPU 用例把它钉在"按顺序写入 n 个位置之后每个槽拿到什么"这个定义上。
2. 另外两条活下来的是**变异本身太弱**，不是测试弱（一个浮点和上的 `>=` vs `>`、
   一个 1e-9 的 CDF 亏空，两者都是测度零），已换成真的会改变行为的版本。
   **记在这里是因为它是这套方法的失败模式**：一个"survived"要先怀疑变异，再怀疑测试。

**这套方法第二次抓到同一类缺陷**：合并 Track Y 时 `suite.resident_route` 被加进了 suite 列表
却**没有加进 `unit` 标签那个 foreach**——于是它在 `ctest -LE needs-model` 里会跑（它没有任何标签），
但 `run_all.py` 和 `ctest -L unit` 都**看不见它**。本轮补上（27 → 28 个 gate）。
**教训和 `suite.kvdisk` 那次是同一条**：新套件要改两处，而只改一处的症状是"全绿但少跑了一个"。

**已知的标签缺陷**：`deepmoe_tests`（整个二进制这一条 ctest 项）挂着 `needs-model`，
所以 `ctest -LE needs-model` 会把它整条跳掉。
（`suite.kvdisk` 原本完全没有 `set_tests_properties`——没有标签也没有 skip 正则；本轮补上 `unit`。）
这也是 `run_all.py` 问 ctest 要 `unit` 标签、而不是自己维护一张列表的原因：
列表会把这种缺陷藏起来。

### 5.5 尺子本身：8 步分不开的东西，64 步分得开

Track Y 的判决在同一份代码上**翻过一次**，翻的不是代码是 harness——这条值得单列，
因为它是这份文件里唯一一次"结论被尺子决定"。

- **8 步的 teacher-forced L3 分不开档位**：三档都落在 6/8 或 8/8，而且出现过自相矛盾
  （`stall1` 的 gate mass lost 只有 `all` 的一半，PPL 却更高）。
- **64 步（`traces/l3_64` + `tools/l3_ppl.py`）就分得开**：top-1 **61 / 42 / 51**，
  PPL ×1.00 / ×2.29 / ×1.11，**与 mass lost 单调同向**；`stall1` 三次独立进程跑进 0.0035 的带里，
  `all` 三次跨 0.87 个 PPL（那是后台补盘时序的真实方差，不是测量噪声）。
- **这把尺子自己的噪声底是 ×1.0330**（`off` 对 fp32 参考：NLL 0.5976 → 0.6301，top-1 61/64）。
  所以 ×1.09 只有噪声底的三倍——**读得出来，但不要把 1.09 和 1.05 的差别当大数字**。
- **导出必须逐步解码，不能用一遍 prefill 抄近路。** 一遍 128 token 的 prefill 能一次拿到全部 65 个
  next-token 分布，快 7.4 倍（226.5 s vs 1,687.8 s），但它只能**证明** 5/64 个位置。
  把同一遍跑在参考**自己逐步**解码出的 token 上，分叉位置和分叉 token 一模一样（index 5）——
  **"一遍 prefill 的第 j 个位置"和"逐步解码到第 j 个位置"不是同一个分布**
  （compressor 结尾不完整分组的进位、indexer 在整段 compressed cache 上的 top-k，在一遍式里都不是逐位置因果的）。
  顺带的对照：**引擎自由跑跟住参考 12 个 token，一遍式只跟住 5 个——这一局引擎比捷径准。**

一句话：**任何跨 routing / cache 策略的质量判决，尺子至少要 64 步，而且参考续写必须是逐步产的。**

---

## 6. 已知限制与未决风险

1. **decode 被 NVMe 钉死**：每 token 60% 是 stall。**默认 `auto` = 5,100 槽 / 89.3 GiB，6.05 → 5.60 tok/s / hit 0.9383**，
   而且这已经是本机安全上限附近（5,400 八轮中途死、5,500 第一个 token 丢设备，§3 的 30 / `p4_hitrate.md` §4）。
   容量曲线还在爬（sim 6,500 槽 +2.2 点），但**本机没有字节了**。20 tok/s 需要把 MB/token 再砍 3×：
   2-bit 已经否掉（§3 的 37），只剩投机解码，而它今天做不到（§3 的 34）。
2. ~~**每次 expert miss 20–31 ms，真实盘时间 ~2 ms，10× 的差没有解释。**~~ **已关闭**：
   那就是盘对 expert 尺寸随机读的表现（QD=4 时 5.19 GB/s / 13.6 ms 每请求，一层 6 个 = 21.7 ms），§4。
3. **热步只剩 ~7% 余量**（81.5 vs 地板 75.8 ms）。剩下的是结构性的：每 token 40 次 host 往返读 gate
   （~1.6 ms submit + ~2 ms fence），以及 MoE 的 `x` 每层往 host 走一趟。
4. **prefill 是 19.5 ms/prompt token 的 compute**（F3 实测，4,133 token = 80.6 s），
   在对话里比 decode 还贵。**≥ 5× 的目标低于这台机器的算术地板**：73.5 TFLOP ÷ 本趟最好的
   2.1 TFLOP/s = 35 s，而 5 ms/token 只有 20.7 s——**要 3.56 TFLOP/s 持续**。
   唯一能动速率的是重写 `prefill_coopmat` stage 0（多 wave + LDS + 双缓冲），**未做**（§2.5）。
   **另外它已经有一个 41× 的复用手段（SSD KV）没接进 `serve` 的默认路径。**
5. ~~**`Engine::generate` 的 `speculative` 是 `unimplemented`**，缺三个 kernel 能力~~ **verify 那一半做完了，draft 那一半没有**（Track SP，`p4_dspark_runtime.md` §7）：
   `Engine::forward_batch(p0, tokens[M<=6])` + `snapshot_batch_ring` / `restore_batch_ring` 已经是 `SpecModel` 的三个方法，
   CPU 比较器早就有。**`Engine::generate` 的 `speculative` 仍然是 `unimplemented`**，因为 `SpecModel::draft_forward`
   缺 runtime 侧的 DSpark 草稿链（37/38/39 层 hc-mean → 三个 `DSparkBlock` → top-16 矩阵；kernel 齐了，`runtime/` 里一行没有），
   而且它要 **7.2 GB 的 mtp expert 常驻**，那是从 expert cache 里拿走的。`--spec` / `--accept` 的 CLI **没有接**，因为没有可接的东西。
   **接不接得下去现在是个已答的问题**：§3 的 42 说，就算草稿白送，投机在这台机器上也只是打平。
6. **Track J 的接口（K-split / tiled attention，696 µs/层）没有被 runtime 采纳**，
   而且**它的 LDS 修复在真机上看不到**：attention 在 Track I 是 36.0 ms、Track Q 后是 36.9 ms，J 声称 −3.8 ms 且"已生效"。**这个矛盾未解决。**
   （有一份**从未编译、从未验证**的采纳尝试，是 P4 合并时在 `deepmoe-t` 工作树里捡到的未提交改动，
   保存在 tag `wip/track-t-ksplit-decode` 上，免得删工作树时丢掉。**不要当成已验证的东西用。**）
7. ~~**`MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL` 从未 A/B 过。**~~ **已做（F4，`p4_hitrate.md` §6）**：
   `MOE_OVERLAP` +1–4%（hit 到小数点后四位不变——它搬运工作，不改 cache 内容）**默认 on**；
   `PREFILL_HANDOFF` 在 512 / 1,024 / 2,048 token prompt 上 **+72% / +49% / +17%**，
   还顺带削掉 prefill 自己的 9–14%，**默认 on**；`BACKFILL` 整体只值 +0.0011 hit 却多读 21.5 GiB，
   **默认 off**（而且计数器显示它一直就是 off 的）。**留下一个未解释的格子**：两对 2,048-token
   run 只差 `--max-context`（4,096 vs 8,192），handoff-**off** 那一侧从 0.8487 跳到 0.9715，
   而 handoff-on 两侧完全相同（0.8874/0.8874）。在这条被命名之前，4,133-token 那一行**不能**读成"长 prompt 上 handoff 输了"。
8. **C(M) 曲线缺失**（`bench/results/mgt1_p4.csv`），DSpark 的 20 tok/s 判定挂在它上面。
9. **`serve` 单会话、不能中途打断**；GPU prefill 默认关（`--gpu-prefill-min` 默认 0）。
10. **口径分裂：本文件 §1 / §2.4 里 5,500 槽那一行（6.05 tok/s / 0.9431）是在一台当时能撑住 5,500 槽的机器上量的**，
    而 F4 在安静机上复现不出来（第一个 token 就丢设备）。两个数都留着，但**可依赖的默认是 5,100 槽那一行**。
11. **路径 B 封顶 16 GiB**，因为按物理内存定大小会让下一次 submit 发现设备丢失（§3 的 30）。
12. **没有 per-dispatch 的时间线**（本轮补上：`runtime/trace.*` + `tools/trace_timeline.py`，但**尚未在 GPU 上验证**，命令见 `plan_p5.md` §4）。
13. **F3（prefill kernel 几何）与 F2（KV / session）都是部分工作。** F3 的四处几何修正 + `wo_a` 的 coopmat
    已合入，每个 pre-F3 几何都留了一个环境变量开关用于归因（`tests/test_gpu_prefill.cpp`），
    它的报告与 per-op CSV 也已合入（本轮从 `p4/fin-s` 的工作树里捡回来的，**当时没提交**）。
    但 **F3 自己的那次测量是在另一条 track 的三个 `deepmoe_tests` 同时占着 GPU/NVMe 时做的**——
    只有 per-op 那几列可比，墙钟不可比；§2.5 的 TTFT 仍是 Track L 的数，**没有新的安静机端到端对照**。
    F2 的 KV 多 slab、`clear()` 批量化、fence 等待改成预算（不再是 120 s 死线）、`.pkv`
    字段表与三会话 park/spill 演示已合入，但 **SSD KV 前缀复用仍然不是 `serve` 的默认路径**（见 §7 的 2）。
14. **`p4/one-pr` 合进 main 时，整个 P4 的数字没有在合并后的这棵树上重跑**——
    合并后跑的是 build + 全量 ctest，不是 bench。任何性能数字的出处仍然是它自己那一行指的报告。

---

## 7. Next, in order

顺序的依据是 §4：**先降 MB/token 和 stall，再降 kernel 时间**。
每一项的机制、预测（已按"二分之一法则"砍半）、成本与探针在 [plan_p5.md](plan_p5.md)。

0b. **2026-09-19：多路 decode 落地（Track MS，`p4_multistream.md`，§3 的 57）。**
   引擎按「进程拥有的」（expert cache、planner、pinned、IO、device）和「序列拥有的」
   （KV、激活、**shader 在执行时才读的那几张地址表**、command buffer、两条 timeline）
   切成 `Engine` 与 `runtime::Stream`；`run_layer` 拆成 begin / gate / moe 三段，
   交错的规则是**「进 stall 之前队列里必须有活」**。
   **两条并发对话：4.6474 → 5.4602 tok/s（+17.5%，ABAB 三对）**，每路延迟 0.62×。
   **预测的 1.6–1.8× 没达到**，而且**缺口是量到的**：盘的聚合速率 +28%，但两个工作集
   抢一个 5,100 槽的 LRU 让 MB/token 涨 13.5%，且 2.83 GB/s 只到 D: 天花板的 69%。
   **N≥3 是负的**，所以「再加一条流」这条路是关的。**接这一位的仍然是第 1 项**：
   MB/token 这一侧唯一还开着的杠杆（第二块盘）——它同时是多路这条路的解锁条件。
   单流路径与 main 逐字相同（`l3_ppl` 的 `off` 臂 **NLL 0.630051** 逐位复现）。
0. **2026-09-19：K1a 落地并默认开，K1b 两个方向都退掉**（`plan_p5.md` §3(h) / §3(i)）：
   **K1a**——decode 一直在跑 M=6 的 MoE kernel（引擎的 runner 是 `spec.m = kMoeBatchMax`，decode 只设 `live_columns = 1`）。
   M=1 特化的 pipeline：热步 **101.8 → 97.0 ms**、moe gpu **37.87 → 34.33**。第 7 项点名的 live-column mask 只是其中 1.6 ms。
   **K1b 是 NO-GO，但它量到了一个新系数**：path 放置的两个方向都试了——新 expert 落 path A 四轮 **−5.4%**；
   落 path B 四轮 **+7.0%**（`nvme_stall` −6.78 ms，因为 Track Q2 的 704 µs vs 70 µs）却在八轮换题脚本上 **−20.5%**
   （它顺带把 path A 的 3,400 槽冻成 first-touch pin，miss 涨 41%）。**默认保持单一全局 LRU**，记在 §3 的 55。
   留下的是仪器（`hits_path_a/b`、`fills_path_a/b`）和两个数：path-B 的读 **0.036 ms/expert-read**、path-A 的写 **~6.8 ms/token**——
   **后者比前者大一个量级**，下一次开这一条要的是"只改放置、不顺带变成 pin"的形式，且必须在两个以上的脚本上同时为正。
1. ~~**给 `ReadFile` 两侧插桩，解释每次 miss 的 20–31 ms。**~~ **已完成，答案是"就是盘"**（§4）。
   接这一位的是那条式子的两个因子。**(a) score-aware 淘汰已经做完了，答案是 NO-GO**
   （§3 的 47，`p4_cache_policy.md` §12）：127 种淘汰配置在 4.6 GB/s 的 demand-only 模型上
   跑完，最好的可实现形状 `last_use + α × heat` 在 C=5,100 的测试集上只有 **+2.3%**，
   折半后 **+1.2%**，低于 ±3% 的抖动带；top-16 的原始分数只贡献其中 0.06 pt。
   Belady 的 **+40%** 在"未来"里，当前状态的任何函数都够不着它。
   **所以第 1 项现在只剩 (b)：第二块盘。**
   ⚠️ **2026-09-19 收窄（Track D2，§3 的 58，`p4_dual_source.md`）**：读路径这一侧**已经做完了**——
   `--mirror DIR` 把 manifest 引用的每个 shard 在第二个根下也开一份，
   按**加权最小在飞字节**分流（权重 = 启动探针），默认关。
   **但「stripe 到 9 GB/s，+32–40%」这个预测在本机不成立**：本机没有第二块 NVMe，
   手上的第二块盘是 **1.0 GB/s 的 USB 外置**，加起来是 **+22% 的带宽**而不是 ×2。
   实测聚合随机读 **3.453 → 4.555 GB/s（+31%，分流 78 : 22，主盘份额没掉）**，
   折成 decode 的预测是 **+14%，砍半 +7%**，而且两条打折理由已经点名
   （只路由 P0/P3；decode 是突发形状）——**很可能落进 ±3% 的带**。
   **端到端 ABAB 本轮没跑成，而且不是因为没排上机器**：机器空了之后
   **E: 在引擎的负载形状下掉出了总线**（48 个 `NO_BUFFERING|OVERLAPPED` 句柄上的并发随机读，
   跟在 510 GB 的持续写后面），把一个 `deepmoe` 留在 `taskkill /F` 都杀不掉的 I/O 等待里。
   harness 是 `bench/d2_abab.py`，等一块**插得住**的盘。
   **这一项要重新写成两条**：(b1) **第二块真 NVMe** ——+32–40% 的那句只对它成立，
   而读路径的代码**已经就位**，插上盘、拷一份、`--mirror` 就行；
   (b2) **这块 U 盘** —— **不要再试**：带宽是真的（净加 1.0 GB/s），可靠性不是。
   **它仍然是 MB/token 这一侧唯一还开着的大杠杆**——另一个是容量，已经顶到本机的 5,100 槽。
2. **把 SSD KV 前缀复用接进 `serve` 的默认路径。**
   已实测 41×（101.6 s → 2.47 s），已实现，只是没默认开。这是当前性价比最高的一项。
3. ~~**量 verify-only 的 resident-only 路由在 DSpark 上的收益。**~~ **已做，答案是 NO-GO**（§3 的 40，
   `p4_resident_routing.md` §10）。`DEEPMOE_ROUTE_RESIDENT_ONLY=verify` 已经实现（第四档，
   两个 CLI 都收，`DEEPMOE_VERIFY_FIRST` / `DEEPMOE_VERIFY_DRAFT` 拆开两半）：
   速度这一侧过了（四轮对话 ×1.57，P0 字节 −76%），**质量这一侧 ×1.376 出了 ≤1.3 的带**。
   本条原本的前提——「`stall1` 的 ≤×1.3 就是 verify-only 的 cache 状态」——**是错的**：
   那 0.076 是每步一次 P0 买的。
   ~~**接这一位的是投机解码本身**~~ **也做完了，也是 NO-GO**（§3 的 41 / 42，`p4_dspark_runtime.md` §7）：
   `Engine::forward_batch` 落地了，于是 verify 的代价第一次是**实测**的——M=6 每位置 0.74× 顺序 decode，
   并集实测对离线值误差 ≤2%，接受长度 `chain` 3.82 / `longest` 4.77。把这些代进去，
   **草稿白送时 k=5 反而更慢（0.86×），k=1 最好 1.13×，加上 19–42 ms 的草稿就打平**。
   **不要再往投机上写代码**，除非第 1 项（score-aware 淘汰 / 第二块盘）先把 MB/token 压下来；
   那时要重跑的是 `bench.spec_forward_m_curve` 和 `tools/spec_longest.py`，不是重写循环。
   **唯一还开着、方向为正的问题**：verify 批的草稿行用 resident-only 路由把 P0 打到 0、
   ms/位置从 102.6 压到 70.9，而草稿行掉质量的代价是"接受率低一点"而不是"输出差一点"——
   这条要有真实草稿链才测得了。
3b. ~~**把「引擎只拿到盘的 69%」这件事的两个候选分开**~~ **已做，答案是两个候选是同一件事**（Track Q2，`p4_p0_queue.md` §9–§13）。**目的地内存**确实是原因，但它贵在**提交**不在传输：往 path A 读一个 4 MiB chunk，同步的 `ReadFile` 要 **704 µs**（普通主机内存 70–122 µs），因为内核要 probe 并锁住 1,024 个写合并的设备映射页。而这个调用过去**只在 dispatcher 一条线程上**发生，所以它同时也是**补队速度**：42.81 s 的 issue 里 42.71 s 是它，一个 token ~124 个 chunk × 0.7 ms ≈ **87 ms**，而 `nvme_stall` 才 105 ms。完成回调是清白的（0.13 s，1.6 µs/次）。**已落地并默认开**：`IoEngine` 的提交线程池（`kDefaultSubmitThreads = 8`，`DEEPMOE_IO_SUBMIT_THREADS=1` 回到旧行为）+ runtime `IoConfig` 的 P0 队列 24 / 96 MiB（P1–P3 仍是 8 / 32 MiB）。**4.828 → 4.998 tok/s（+3.5%）**，stall 108.1 → 100.9 ms，issue 42.81 → **0.51 s**，queue wait 2.01 → 0.09 ms，busy 窗口 3.90 → **4.10 GB/s = D: 天花板的 89%**。**预测（+13%）偏乐观，机制说对了**：参照系本来就该是 D: 的 4.60 而不是 C: 的 5.16（§3 的 46），而且那还是**连续流**的天花板——引擎是突发的，一层只有 ~9.6 个 chunk，每层都要重新爬坡再排空。**剩下的不是「填队列」能拿的**，所以这一条到此为止；接下去仍然回到第 1 项（score-aware 淘汰 / 第二块盘）。
   **2026-09-19 追记（Track S1，§3 的 56）**：这条的最后一个疑问——「704 µs 还在，把它换成 staged fill 是不是还能再拿一次」——
   **也关掉了**。`io_dst_bench` 新的 `stage` 档（读进 path B 环 + `vkCmdCopyBuffer` 进 path A，端到端墙钟含拷贝，n=5）：
   QD 8 **+0.12%**、QD 24 **+0.35%**，A 臂噪声底 **0.69% / 0.40%**，判据 ≥5%。
   **因为 Q2 自己已经把那 704 µs 拿走了**：`patha` 和 `pathb` **在发出时的队列深度相同**（6.87/6.87、22.02/22.04）。
   **运行时一行没动。** 顺带定价：`vkCmdCopyBuffer` 18.8 MB = **0.145 ms（129 GB/s）**，
   而 `gpu::Device` **只有一个 compute 队列**——拷贝要和 decode 抢它（21.3 miss × 0.145 = 3.1 ms/token）。
   要重开需要一条真正的 transfer 队列，外加一个**突发**形状的 bench（K1b 那 6.8 ms/token 是 `first-of-burst` 的残留，稳态 bench 量不到）。

4. ~~**per-dispatch trace 上 GPU**~~ **已完成**（Track PD，§3 的 49）。trace 在真机上工作、没有丢 stamp、开销为 0。
   拿到的数：热步 **busy 85.23 + gap 16.84 = 102.07 ms**，而 gap 的分布是 **16.04 ms 在 gate 往返、0.80 ms 在全部 700 个非 MoE dispatch 的 barrier 上**。
   **这个分布就是第 5、7 项的判决**。
   **Track F6 的 roofline（§3 的 52）把这 16 ms 标了价**：MoE 每 token 读 4.512 GB，`moe_gpu` 42.14 ms = 107 GB/s = UMA 的 50%，而 kernel 本身 222.6 GB/s；缺口 18.5–21.9 ms/token 全在 dispatch / 间隙，与 trace 的 16.04 ms gate 往返一致。
   **压到 kernel 速率 = +5.2%…+6.3%（砍半）**，但唯一够得着它的机制（device 侧 gate）被第 5 项关掉。
   ~~剩下的路是把 gate 往返本身从 0.40 ms 压小（host 侧轮询 / 预录 MoE 命令缓冲 / 间接参数），未测。~~ **已测，也是 NO-GO**（Track G，§3 的 53，`plan_p5.md` §3(g)）：
   那 0.40 ms/层 里 **265 µs 是驱动的**——`vkQueueSubmit2` 这一次调用 **99.5 µs**、submit 返回到 GPU 起跑 **≈150 µs**、fence 唤醒 **≈15 µs**；
   我们自己的代码（planner + staging + record + 下一层序言）一共 **95 µs**，而其中 45 是 `act_quant`（归第 7 项的 c4）。
   三件可动的（自旋 60.9 + 提前 submit 32.6 + 预录 9.7）**合计 103 µs/层 = 4.1 ms/token**，而可测的门槛是 **250 µs/层**。
   已落地的 `DEEPMOE_FENCE_SPIN_US`：热步 **−2.3%**、对话 **−1.0%（反向）**，**默认 0**。
   **attention 这条线关掉**：dense 8,522.8 MB/token ÷ 43.51 ms = 196 GB/s = 91% of UMA，打满也只有 +1.1%（砍半）。
5. ~~**persistent-dispatch decode**~~ **关闭**（§3 的 48 + 49）。两侧同时倒：
   合法性——`residency_probe` 说**两个 workgroup 的握手 1,000 次里第 3 次就超时**，自旋等待在这台机器上不可用（常驻上限 ~406 组，但共存不蕴含前进）；
   收益——「一张 dispatch 图」的退化形**已经是今天的实现**（41 submit，只在 gate 处切开），再融合的天花板是 **0.8 ms/token**。
   要重开它，先得有一条**不用自旋**的 device 侧 gate。
6. **absorbed-K attention（V4.1）**：给定 `wkv` latent、`kv_norm`、只在最后 64 维上的 RoPE、fp8 量化点，哪些矩阵可以折叠。
   代数在 `plan_p5.md` §3(b)，读 `D:\models\DeepSeek-V4.1-Flash\inference\`（只读）。
7. ~~**prologue/epilogue 融合**（c1–c3）~~ **划掉**（§3 的 49）。层内 barrier 实测：普通层 17 个 dispatch 一共 **18 µs**、源层 26 个一共 **24 µs**——40 层 = 0.8 ms/token，四分之一个抖动带。
   ~~**新开一项（Track G，§3 的 54）**：把 Track T 的 live-column mask 在 M=1 路径上特化掉~~
   **已做完，而且比预测大三倍**（Track K1a，2026-09-19，`plan_p5.md` §3(h)）。
   mask 那一半对上了：`kernel_bench` M=1 ABAB 四对 **0.6547 → 0.6028 ms/pair（−7.9%）**，B 臂正好落在 `fb53514^` 的 0.6035。
   但**引擎从来没有创建过 M=1 的 pipeline**：`moe_bridge.cpp` 用 `spec.m = kMoeBatchMax`（=6）建 runner，decode 只把 `live_columns` 设成 1，
   于是每个 decode token 用六个累加器算一列。`MoeRunner` 现在带一套 M=1 特化（gate/up + h 量化 + down 三个必须一起换）：
   **热步 101.8 → 97.0 ms（−4.7%）、moe gpu 37.87 → 34.33（−9.3%）**，预测是 −1.6 ms，实测 **−4.8**。
   闸：`l3_ppl` off **NLL 0.630051** 逐位复现、`suite.gpu_moe` / `suite.decode` / `suite.decode_longctx` 全过。**默认开**，`DEEPMOE_MOE_STATIC_M1=0` 是退路。
   **c4 还开着**（`WoB` 的 epilogue 直接写出量化好的 FFN 输入），它省的是 host 往返不是 dispatch：MoE host 今天 1.6–1.9 ms/token。
8. ~~**2-bit expert**（等 F5 的精度判定）~~ **已否决**（§3 的 37，2 bit 与 3 bit 都是 NO-GO）；
   ~~**命中率杠杆**（等 F4）~~ **已交付**（默认 `auto`、路径 A 预留、三个 A/B，§2.4 / §2.5 / §6 的 7）。
   接这一位的是 **`prefill_coopmat` stage 0 重写**（多 wave 经 LDS 协作一个输出 tile + 下一个 K 切片预取）。
   它是 F3 点名的、唯一能动 2.1 TFLOP/s 这个速率的改动（§2.5），但它**只动 prefill**，
   所以按 §4 的排序它排在最后——除非对话里的 prompt 变长到 prefill 压过 decode。

**不做**（有编号的理由，不要再提）：BIOS VGM（§3 的 29）、lookahead 预取（23、**35**、**51**——
**48 换掉了 35 的理由**：盈亏平衡不是 precision 1.00 而是 **≈0.60**，NO-GO 的原因是
**可用的提前量只有 2 ms/层**，诚实的 lead=1 只拿得到 `pred_d2` 的 0.588；验收线是
**两层提前量上 precision ≥ 0.77**）、**attention / dense kernel 的带宽（49：已经 91% of UMA）**、
**节流 P2 engram（43：−27.8%，engram 4.5 → 85.8 ms/token）**、
**为 P0 加深队列或加大 chunk（44）**、
CPU 分担 GEMV（25）、树采样作为提速手段（32）、静态 pin / 每层配额（24、**35**）、
LDS x-tiling 配 per-K-chunk barrier（1）、饱和 cache 上的 reheat（21、**F4 §6**）、
任何预测式预取（**35**）、**score-aware / ARC / LFU-decay / LRU-K / S3-FIFO 任何非 LRU 的淘汰策略（47）**、2-bit / 3-bit expert（**37**）、resident-only 作为默认路由（**38、39**）、verify-only 的 resident-only 路由（**40**）、
**投机解码本身（41、42：`chain` 0.86×、`longest` ≤1.07× 且不无损）**、
**把 gate 的 host 往返压小（53：265 µs/层 是驱动的，可动的只有 103）**、手写 `--cache-slots`（它绕过三条实测边界，5,500 就是这么够得着的）。
