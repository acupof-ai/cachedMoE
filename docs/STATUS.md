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
3. [试过并退掉的（编号，共 44 条）](#3-试过并退掉的编号共-40-条)
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
| **热步**（expert 全驻留，64 token 上下文） | **81.5 ms = 12.3 tok/s**；4K 91.1 ms；17K 94.2 ms | `p2_decode.md` §10 |
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

## 3. 试过并退掉的（编号，共 44 条）

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

---

## 4. 为什么 decode 是 NVMe-bound，而不是 kernel 慢

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

1. ~~**给 `ReadFile` 两侧插桩，解释每次 miss 的 20–31 ms。**~~ **已完成，答案是"就是盘"**（§4）。
   接这一位的是那条式子的两个因子，按性价比：**(a) score-aware 淘汰**——把 `cache_sim` 里
   用 top-16 原始分数刷 heat 的那个策略接进真 planner，是本项目测过的淘汰策略里**唯一没跑过**、
   而 Belady 上限有 **+42%** 的那个（§3 的 36）；**(b) 第二块 NVMe**（stripe 到 9 GB/s，
   **+32–40%**，无预测、无风险，design §3.1 已预留 stripe）。
   **两条都比任何"算得更快"的改动大一个量级。**
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
3b. **把「引擎只拿到盘的 69%」这件事的两个候选分开**（Track Q1 的唯一遗留，`p4_p0_queue.md` §4、§6）。
   已知：`nvme_stall` 105.2 ms/token 里，**队列争用是 0**（实测 0.0%），
   **队列深度不是因**（抬到 32 反而更慢），而引擎拿到 3.55 GB/s、盘能给 5.16 GB/s。
   剩下两个候选：**(i) 目的地内存**（expert slot 是 GPU 可见的，`nvme_bench` 写的是普通页）、
   **(ii) dispatcher 的补队速度**（`issue_ready_chunks` → `poll(1 ms)` → `handle_completion` 一圈；
   上限 QD 8，实测发出时平均 3.80，而 chunk 越碎实测 QD 越高、速率越高）。
   **最便宜的探针**：给 `bench/nvme_bench` 加一个「写进 GPU 可见 slab」的目的地开关，
   一个 cell、纯盘、不占 GPU。若那样仍是 5.1 GB/s，缺口就全在 dispatcher 那一圈。
   **值 105 ms 里的 ~30 ms = 每 token 近 30%**，比第 4–7 项都大，且不动一个权重。

4. **per-dispatch trace 上 GPU**（`plan_p5.md` §4 的命令），把每层 16–29 个 dispatch 的 busy / gap 拆开。
   §2.1 之后所有归因都靠它——没有它，第 5、6 项只能猜。
5. **persistent-dispatch decode**：一层或一个 token 一次 dispatch，device 侧任务队列 + 自旋等待。
   能省的是 **41 × 0.156 ms 的 submit 往返 + barrier 尾巴**。
   先跑可行性探针（`bench/probes/residency_probe`）：8060S 上 Vulkan 能常驻几个 workgroup、跨 workgroup 原子/事件延迟、前进保证风险。
6. **absorbed-K attention（V4.1）**：给定 `wkv` latent、`kv_norm`、只在最后 64 维上的 RoPE、fp8 量化点，哪些矩阵可以折叠。
   代数在 `plan_p5.md` §3(b)，读 `D:\models\DeepSeek-V4.1-Flash\inference\`（只读）。
7. **prologue/epilogue 融合**：把 norm / residual / act_quant 折进 GEMV。
   今天一个非源层 16 个 dispatch、源层最多 29 个；融合能删掉其中 5–7 个和它们的 barrier。
8. ~~**2-bit expert**（等 F5 的精度判定）~~ **已否决**（§3 的 37，2 bit 与 3 bit 都是 NO-GO）；
   ~~**命中率杠杆**（等 F4）~~ **已交付**（默认 `auto`、路径 A 预留、三个 A/B，§2.4 / §2.5 / §6 的 7）。
   接这一位的是 **`prefill_coopmat` stage 0 重写**（多 wave 经 LDS 协作一个输出 tile + 下一个 K 切片预取）。
   它是 F3 点名的、唯一能动 2.1 TFLOP/s 这个速率的改动（§2.5），但它**只动 prefill**，
   所以按 §4 的排序它排在最后——除非对话里的 prompt 变长到 prefill 压过 decode。

**不做**（有编号的理由，不要再提）：BIOS VGM（§3 的 29）、lookahead 预取（23、**35**）、
**节流 P2 engram（43：−27.8%，engram 4.5 → 85.8 ms/token）**、
**为 P0 加深队列或加大 chunk（44）**、
CPU 分担 GEMV（25）、树采样作为提速手段（32）、静态 pin / 每层配额（24、**35**）、
LDS x-tiling 配 per-K-chunk barrier（1）、饱和 cache 上的 reheat（21、**F4 §6**）、
任何预测式预取（**35**）、2-bit / 3-bit expert（**37**）、resident-only 作为默认路由（**38、39**）、verify-only 的 resident-only 路由（**40**）、
**投机解码本身（41、42：`chain` 0.86×、`longest` ≤1.07× 且不无损）**、
手写 `--cache-slots`（它绕过三条实测边界，5,500 就是这么够得着的）。
