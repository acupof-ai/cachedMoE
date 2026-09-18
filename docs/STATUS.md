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
3. [试过并退掉的（编号，共 34 条）](#3-试过并退掉的编号共-34-条)
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
| cache 容量上限（本机） | **5500 槽 = 96.34 GiB 可用；5600 在 path B 第 20 个 slab 失败（98.10 GiB）** | `p4_hitrate.md` 容量实测 |

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

Track S 的 coopmat：N=4133 **legacy 103.67 s vs coop 99.89 s，只快 3.6%**，5× 目标未达成（§3 的 32）。

---

## 3. 试过并退掉的（编号，共 34 条）

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
| **34** | **在一个 NVMe-bound 的 decode 上做投机，总体** | l3 的 M=1 是 1.28–1.33 ms/层 → 40 层 ≈ **51 ms**，但热步实测 **81.5 ms**；差的 ~30 ms 是 MoE+head，而 MoE 按需从 NVMe 取 expert。**verify 的 MoE 代价不随 M 摊薄，它随并集线性增长**——6 个 token 的并集 ≈26 expert/层。C(M) 曲线本身**过了**（M=6 最差 2.98×，判据 ≤3.06×），但 MoE(M) 没过。整体：**有条件 GO ×1.18–1.19**（confidence θ=0.5），刚过 1.15 门槛；固定 k 过不了（最好 ×1.14）；如果 head 退化成 M=1 循环（`T_draft` 42 ms）就掉到 **×1.11 = NO-GO** | `Engine::generate` 的 `speculative` 至今是 `unimplemented`。缺三个 kernel 能力（`p4_dspark_runtime.md` §2）。**投机解码在"权重读取"这一侧才有意义（§2.3 的 3.65×），但它的 MoE 并集把这个收益又吃回去了**——这是 P5 必须先关掉的那道门 |

### 3.6 其它记下来的

- **`XLayout` enum 从来没生效过**：`run_batch` 从不读 `layout`，`column_x` 两种模式都填满 6 列，所以 `Staged ≡ OneColumn`。**已删除**。底下压着的真 bug：host 把每一列都从 **y 的第 0 列**拷出来，而权重是按列 stage 的——第 0 列"恰好"逐位正确，1/2 列差 15–22% 的 |y|max。指纹：`per-column weights only` 的离散度修前是 **0**，修后是 **5.235e+00**
- **bf16 残差流实验**（会不会把第 6/7 个 expert 放回参考的顺序）——design §15 issue 3，**从未做**。第 2 层的近似平局：流入 cos 0.9997，MoE 输出 **0.9883866**，gate 5/6
- **`gate.slang` 的 softplus**：`log(1+exp(z))` 在 z≈−16 以下归零，max |gpu−cpu| **1.5e-4**，只影响分数 <1e-3 的 expert。一行的修复，**仍然开着**（design §15 issue 20）
- **`KvStore::clear()`** 每次 `reset_context` 在 path A 映射上逐元素写 163,840 个 `−inf`——§7.1 规则 10 的违反，**未修**（design §15 issue 26）
- **压缩 KV 存 bf16 而不是打包 FP4**：64K 上省 18 MB（66 vs 48 MB），代价是 `sparse_attn` 内循环里一次 nibble 解包。**故意推迟**
- **P2 step 2 的不可复现不是 kernel**——是**三条 track 往同一个 `build/shaders` 目录并发重建 `.spv`**。证据：step 2 报 step-0 margin **7.0927**，它自己那个 commit 隔离重建给 **6.7477**，带 Track F 的 kernel 给 **6.9389**——引用的那几次 run **两套 kernel 都没执行**。**这是"安静机"规则的由来**（`plan_p5.md` §1）
- **`MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL` 的 A/B**：设计好的表**从未跑过**（沙箱 `SetNamedSecurityInfoW failed (Win32 5)`）。`p4_hitrate.md` §4 至今是空的。**未实测**

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

**没量过的那一块**：每次 expert miss 花 **20–31 ms**（107 GB / 5,030 次填充），
而 8.3–9.2 GB/s 下 18.8 MB 的真实盘时间是 **~2 ms**。
**这 10× 的差没有解释**（§3 的 26 排除了"载入时量化"这个嫌疑）。
这是今天最大的一个未知数，也是 §7 第 1 项。

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
| `bench.mgt1_m_curve` / `bench.mgt1_moe_m_curve` | C(M) 曲线 | **未产出**（`bench/results/mgt1_p4.csv` 缺） |

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

**已知的标签缺陷**：`deepmoe_tests`（整个二进制这一条 ctest 项）挂着 `needs-model`，
所以 `ctest -LE needs-model` 会把它整条跳掉。
（`suite.kvdisk` 原本完全没有 `set_tests_properties`——没有标签也没有 skip 正则；本轮补上 `unit`。）
这也是 `run_all.py` 问 ctest 要 `unit` 标签、而不是自己维护一张列表的原因：
列表会把这种缺陷藏起来。

---

## 6. 已知限制与未决风险

1. **decode 被 NVMe 钉死**：每 token 60% 是 stall。5500 槽（96.34 GiB，本机上限）能到 6.05 tok/s / hit 0.9431，
   再往上没有容量了。20 tok/s 需要把 MB/token 再砍 3×，只有投机解码能做到，而它今天做不到（§3 的 34）。
2. **每次 expert miss 20–31 ms，真实盘时间 ~2 ms，10× 的差没有解释。**（§4）
3. **热步只剩 ~7% 余量**（81.5 vs 地板 75.8 ms）。剩下的是结构性的：每 token 40 次 host 往返读 gate
   （~1.6 ms submit + ~2 ms fence），以及 MoE 的 `x` 每层往 host 走一趟。
4. **prefill 是 24 ms/prompt token**，在对话里比 decode 还贵（4,133 token ≈ 100 s）。
   Track S 的 coopmat 只买到 3.6%，5× 目标未达成。**它已经有一个 41× 的复用手段（SSD KV）没接进 `serve` 的默认路径。**
5. **`Engine::generate` 的 `speculative` 是 `unimplemented`**，缺三个 kernel 能力（M=6 的 MoE、草稿链的 bf16 输入 GEMV、`accept_sampling_exact` 的四个读回）。
6. **Track J 的接口（K-split / tiled attention，696 µs/层）没有被 runtime 采纳**，
   而且**它的 LDS 修复在真机上看不到**：attention 在 Track I 是 36.0 ms、Track Q 后是 36.9 ms，J 声称 −3.8 ms 且"已生效"。**这个矛盾未解决。**
7. **`MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL` 从未 A/B 过。** 三个默认开着的开关，没有一个有自己的数字。
8. **C(M) 曲线缺失**（`bench/results/mgt1_p4.csv`），DSpark 的 20 tok/s 判定挂在它上面。
9. **`serve` 单会话、不能中途打断**；GPU prefill 默认关（`--gpu-prefill-min` 默认 0）。
10. **`p4/one-pr` 是 draft**：合并在一个禁止创建子进程的沙箱里完成，多数数字是合并前会话的记录，不是在这棵树上重跑的。
11. **路径 B 封顶 16 GiB**，因为按物理内存定大小会让下一次 submit 发现设备丢失（§3 的 30）。
12. **没有 per-dispatch 的时间线**（本轮补上：`runtime/trace.*` + `tools/trace_timeline.py`，但**尚未在 GPU 上验证**，命令见 `plan_p5.md` §4）。

---

## 7. Next, in order

顺序的依据是 §4：**先降 MB/token 和 stall，再降 kernel 时间**。
每一项的机制、预测（已按"二分之一法则"砍半）、成本与探针在 [plan_p5.md](plan_p5.md)。

1. **给 `ReadFile` 两侧插桩，解释每次 miss 的 20–31 ms。**
   这是唯一一个 10× 量级的未知数，而且它按定义在 token 的 60% 那一侧。
   便宜：两个时间戳 + 一条 JSONL 字段。**在做任何 kernel 工作之前做这个。**
2. **把 SSD KV 前缀复用接进 `serve` 的默认路径。**
   已实测 41×（101.6 s → 2.47 s），已实现，只是没默认开。这是当前性价比最高的一项。
3. **跑完 §6 第 7 项的三个 A/B**（`MOE_OVERLAP` / `PREFILL_HANDOFF` / `BACKFILL`），安静机、成对交替。
   三个默认开着但没有数字的开关，其中任何一个可能是负的（参见 §3 的 20、21、22：三个"显然有用"的东西都是 0）。
4. **per-dispatch trace 上 GPU**（`plan_p5.md` §4 的命令），把每层 16–29 个 dispatch 的 busy / gap 拆开。
   §2.1 之后所有归因都靠它——没有它，第 5、6 项只能猜。
5. **persistent-dispatch decode**：一层或一个 token 一次 dispatch，device 侧任务队列 + 自旋等待。
   能省的是 **41 × 0.156 ms 的 submit 往返 + barrier 尾巴**。
   先跑可行性探针（`bench/probes/residency_probe`）：8060S 上 Vulkan 能常驻几个 workgroup、跨 workgroup 原子/事件延迟、前进保证风险。
6. **absorbed-K attention（V4.1）**：给定 `wkv` latent、`kv_norm`、只在最后 64 维上的 RoPE、fp8 量化点，哪些矩阵可以折叠。
   代数在 `plan_p5.md` §3(b)，读 `D:\models\DeepSeek-V4.1-Flash\inference\`（只读）。
7. **prologue/epilogue 融合**：把 norm / residual / act_quant 折进 GEMV。
   今天一个非源层 16 个 dispatch、源层最多 29 个；融合能删掉其中 5–7 个和它们的 barrier。
8. **2-bit expert**（等 F5 的精度判定）、**命中率杠杆**（等 F4）。这两项依赖别的 track。

**不做**（有编号的理由，不要再提）：BIOS VGM（§3 的 29）、lookahead 预取（23）、
CPU 分担 GEMV（25）、树采样作为提速手段（32）、静态 pin / 每层配额（24）、
LDS x-tiling 配 per-K-chunk barrier（1）、饱和 cache 上的 reheat（21）。
