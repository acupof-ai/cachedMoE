# P1：GPU 微内核与内存路径实测

本文是 design.md §3.3（UMA 两条路径）、§7.1/§7.9（MoE kernel）、§9.2（带宽矩阵）、
§3.4（dispatch 开销）几个问题的**实测答案**。原始 CSV 在 `bench/results/`。
design.md 与 README 由本文的结论另行整合。

状态：v0.1（2026-09-14）。测量机器：Ryzen AI Max+ 395 / Radeon 8060S (gfx1151) /
128 GB LPDDR5X / **BIOS VGM = 64 GB**。Vulkan 1.4.349，Adrenalin 32.0.31041.1004。

---

## 0. 先看结论

| 问题 | design 的假设 | 实测 | 结论 |
|---|---|---|---|
| GPU 读带宽上限 | §3.1 假设 200 GB/s 有效 | **215–218 GB/s** | 假设略保守，成立 |
| 路径 A vs B 的 raw-read 带宽 | §3.3 待验证 | A 216.4 / B 215.2 GB/s，**差 0.5%** | 流式读上完全等价 |
| 路径 A vs B 的 **MoE kernel** | §3.3 待验证 | A 222.1 / B 195.2 GB/s，**B 慢 12%** | GART 4 KiB 页的代价，只有真实访问模式才暴露 |
| MoE kernel 有效带宽 | §15 P2 要求 ≥ raw-read 的 80% | **218.5 GB/s = raw-read 的 102%** | 达标，已到内存系统上限 |
| 每 dispatch 启动 + barrier | §3.4 猜 5–20 µs | **0.56–0.66 µs**（GPU 侧） | **低一个数量级**，§3.4 的 5–14% 顾虑取消 |
| CPU 读带宽 | §3.1 隐含与 GPU 同量级 | **100.9 GB/s**（32 线程峰值） | 只有 GPU 的 47% |
| CPU + GPU 并发总带宽 | §8 cost model 需要 | **213–216 GB/s，与 GPU 单独时相同** | 内存控制器是单一共享上限 |
| NVMe 直读进 GPU 内存 | §9.6 零拷贝 | **跑通**，路径 A 3.9 / 路径 B 4.8 GB/s | 零拷贝成立，B 打满盘 |
| MoE 每层时间 T_layer | §9.4 估 1.3 ms（整层） | **0.602 ms**（MoE 部分，M=1） | §9.4 的 `d ≥ 4` 成立 |

三个**出乎意料**的结果，下面各有一节：

1. **FP4 解码方式决定一切**：常量表解码 218.5 GB/s，算术构造 164.2 GB/s，显式 select 树
   151.4 GB/s。同一个 kernel、同样的字节数，差 **1.4 倍**（§3.2）。
2. **CPU 和 GPU 一起读，总带宽不增加**：GPU 单独 216 GB/s，加上 32 线程 CPU 之后
   GPU 185 + CPU 29 = 214 GB/s。**内存控制器就是那 217 GB/s，谁读都一样**（§2.3）。
   这直接否掉了"CPU 分担 expert GEMV"。
3. **M=6 时瓶颈从权重变成激活读取**：有效带宽掉到 135.1 GB/s，但每 token 的 MoE 时间
   从 0.602 ms 降到 **0.162 ms**（快 3.7 倍），投机解码的收益比 §10.1 估的还大（§3.3）。

---

## 1. 测量方法（先说清楚噪声）

**LPDDR 是 CPU 和 GPU 共用的，任何吃内存带宽的 CPU 进程都会污染 GPU 的数字。**
测量期间本机上另一个 agent 在跑 `tools/route_trace.py`（torch CPU 前向），CPU 约 55%，
同一个 kernel 变体在忙 / 闲两种状态下相差最多 **50%**
（`M1 L32 R1 sg32 dec1`：闲时 164.6 GB/s，忙时 94.5 GB/s）。因此：

- 本文所有数字都来自**机器空闲窗口**。判据是 raw-read 上限在 sweep 前后一致：
  随 CSV 一起提交的那一轮 **sweep 前 214.5 GB/s、sweep 后 214.4 GB/s，差 0.05%**。
  另一轮独立的完整 sweep（上限 217.8 / 217.2）给出的每个结论都一致，最佳变体
  218.5 vs 221.6 GB/s、`T_layer` 0.602 vs 0.594 ms，**差异 ≤ 1.5%**；
  带宽矩阵连跑四次，GPU 三条路径的数字都落在 214.9–217.0 GB/s 之间。
- `bench/kernel_bench` 的方法学：每个变体先 warm-up 8 次迭代，取 N 次测量中**最快**的
  一次；整张变体表**扫两遍**取每个变体的最好值，避免"先跑的变体占便宜"；
  sweep 之前先用 raw-read 跑满 1 秒把 GPU 时钟拉上去。
- GPU 时间一律用 timestamp query（`timestampPeriod = 10.0 ns`，`timestampValidBits = 64`），
  墙钟时间并排打印，两者之差就是提交开销。
- 工作集用 `--layer-cycle N` 在 N 层 × 7 个 expert 之间轮转（用 8 层 = 1053 MB），
  这样 32 MB 的 MALL 装不下，读的确实是 DRAM。只用 1 层（132 MB）会把带宽**虚高约 25%**。

**踩过的两个坑，记下来免得重犯**：

- 并发那一栏最初用 20 ms 的测量窗口 + 每遍完整扫描才计一次字节，而 32 个线程走完各自的
  134 MB 切片要 40 ms —— CPU 那一侧的数字完全是调度噪声（同一配置在 5.7 和 76 GB/s
  之间跳）。改成每 4 MiB 记一次、窗口拉到 ~200 ms 之后，三条路径的结果才稳定在 29–34 GB/s。
- 编译和测量**不能同时跑**。一次 `cmake --build` 和 `bw_matrix` 撞车，CPU 峰值从
  100.9 掉到 97.7，并发那一栏更是废掉。

---

## 2. 带宽矩阵（design §9.2 / §3.3，VGM = 64 GB）

原始数据：`bench/results/bw_matrix.csv`。

### 2.1 CPU 单独顺序读（普通主机内存，4 GiB 缓冲）

| 线程 | GB/s |
|---|---|
| 1 | 24.4 |
| 2 | 43.5 |
| 3 | 53.4 |
| 4 | 53.0 |
| 8 | 55.3 |
| 16 | 84.7 |
| 32 | **100.9** |

**峰值 100.9 GB/s，是理论 256 GB/s 的 39%、GPU 的 47%。** 3–8 线程之间有一个平台
（53 → 55 GB/s），16 线程才继续爬——单个 CCX 的 L3/IF 端口先饱和，跨 CCX 之后才拿到更多。

### 2.2 GPU 单独读（`gpu/shaders/rawread.slang`，1 GiB，320 workgroup）

| 内存 | GB/s | 备注 |
|---|---|---|
| **路径 A** `DEVICE_LOCAL\|HOST_VISIBLE`（memory type 2, heap 1） | **216.4** | vkMapMemory 映射，CPU 可写 |
| **路径 B** `VK_EXT_external_memory_host`（VirtualAlloc，4 KiB 页） | **215.2** | 比 A 低 0.5% |
| 路径 B（2 MiB 大页） | — | **不可用**：本账户没有 SeLockMemoryPrivilege |
| 纯 `DEVICE_LOCAL`（memory type 0，host 不可见） | **216.0** | 与 A 无差别 |

四次独立运行，三条线始终落在 214.9–217.0 GB/s。**在这颗 APU 上不存在"真 VRAM 更快"**——
device-local-only 和 device-local+host-visible 是同一块 LPDDR，驱动只是换了缓存属性标签；
路径 B 的 0.5% 差距在测量噪声边缘。

`216 / 256 = 84%` 的理论带宽利用率，对 LPDDR5X-8000 的流式读是正常水平。
**这是 §3.1 里"常驻 42 ms"那一项的真实分母**：8.5 GB / 216 GB/s = **39.4 ms**，
比 §3.1 用 200 GB/s 算的 42 ms 略好。

### 2.3 CPU / GPU 并发：总带宽是固定的

GPU 跑 raw-read（~200 ms 窗口）的同时，32 个 CPU 线程顺序读主机内存：

| GPU 读的内存 | GPU GB/s | 同时 CPU GB/s | **合计** | 占"各自单独"之和 |
|---|---|---|---|---|
| 路径 A | 185.8 | 28.6 | **214.4** | 68% |
| 路径 B | 184.4 | 28.7 | **213.1** | 67% |
| 纯 DEVICE_LOCAL | 186.4 | 29.6 | **216.0** | 68% |

**三条路径完全一样，而且合计正好等于 GPU 单独时的 216 GB/s。**
也就是说：**内存控制器的总读带宽就是 ~217 GB/s，CPU 和 GPU 只是在分它**。
仲裁偏向 GPU：CPU 从单独时的 100.9 掉到 29（−71%），GPU 从 216 掉到 185（−14%）。

对 design §8 的直接后果：**"CPU 分担 expert GEMV"（§8 的 P6 实验）从带宽上就不可能成立。**
系统总带宽固定，把一部分权重挪给 CPU 算不会增加总吞吐，只会：
(a) 让 GPU 少拿 14%，(b) 由一个每字节更慢的执行体来处理这部分。
**建议把它从 P6 改为"明确不做"**（design §16），除非双盘或别的改变动摇了前提。

反过来，这个数字对 **IoEngine 与 Planner 是好消息**：它们只需要几 GB/s（NVMe 上限 4.7），
而 CPU 在 GPU 满载时仍有 29 GB/s 可用，绰绰有余。

### 2.4 CPU 写入（NVMe 落地模拟，design §9.6）

单线程，4 × 256 MB：

| 目标 | memcpy | 非临时存储（`_mm512_stream_si512`） | 一个 18.8 MB expert |
|---|---|---|---|
| 路径 A 的映射内存（写合并、不可缓存） | 20.1 GB/s | 21.5 GB/s | 0.9 ms |
| 路径 B 的主机内存（普通可缓存） | 18.3 GB/s | **32.7 GB/s** | 0.6 ms |

- 路径 A 的映射本来就是写合并的，普通 store 已被硬件合并，**非临时存储没有收益**
  （20.1 → 21.5，噪声内）。
- 路径 B 是可缓存内存，非临时存储省掉 RFO，**快 1.8 倍**。
- 两条路都远超 NVMe 的 4.7 GB/s（§9.2.1）：**CPU 拷贝不会成为落地瓶颈**。
  这只在需要 staging 的路径上才有意义；真正的零拷贝路径（下节）连这一次拷贝都没有。

### 2.5 NVMe 直读进 GPU 内存：零拷贝成立

`bench/kernel_bench` 每次启动都做一次：用**真正的** `IoEngine` + IOCP +
`FILE_FLAG_NO_BUFFERING`，把 56 个真实 expert（1053 MB）直接读进 slab，
中间没有任何缓冲。

| 路径 | 56 个 expert（112 个 run） | 有效 GB/s |
|---|---|---|
| A（映射的 device 内存，写合并） | 0.260–0.286 s | 3.7–4.1 |
| B（导入的主机内存，可缓存） | 0.218–0.221 s | **4.8** |

**两条路径的指针都能直接作为 `ReadFile(OVERLAPPED \| NO_BUFFERING)` 的目标缓冲**：
`vkMapMemory` 与 `VirtualAlloc` 返回的指针天然页对齐，满足扇区对齐要求。
design §9.6 的"目标缓冲直接是 slab 槽、零拷贝"从假设变成事实，
`tests/test_gpu_moe.cpp` 每次运行都会再验证一遍（读进去的字节要能算出 oracle 的答案）。

路径 B 打满了 §9.2.1 的 4.5–4.7 GB/s 盘上限；路径 A 的 3.7–4.1 GB/s 差的那一截来自
写合并内存的写入路径（与 §2.4 的 21 vs 33 GB/s 同源）。两者都受限于 bench 的
**串行下单**（一个 expert 的两个 run 发完才发下一个），而不是 Planner 的并发队列。

### 2.6 大页：这台机器上拿不到，但它比看起来重要

`VirtualAlloc(MEM_LARGE_PAGES)` 需要 `SeLockMemoryPrivilege`，本账户没有，
运行时也无法自行授予（需要 secpol.msc → "锁定内存页" + 重新登录）。
代码里的尝试是完整的（`gpu::alloc_host_pages(bytes, try_large_pages=true)` 会先试着
提权），失败时把原因写进 `HostAllocInfo::note`、回落到 4 KiB 页，CSV 记为
`large pages unavailable: SeLockMemoryPrivilege is not held by this account`。

看 §2.2 的 raw-read 数字（路径 B 只慢 0.5%）会以为大页无关紧要。
**§3.5 推翻了这个印象**：同样的两块内存，MoE kernel 在路径 B 上慢 **12%**。
raw-read 每个 workgroup 顺序走一整片连续内存，GART TLB 只需要很少的表项；
MoE kernel 同时在 7 个 expert × 2304 行上前进，每行 2560 B，
在途的 4 KiB 页数量高一两个数量级。**这正是 2 MiB 大页该解决的问题。**

因此本文把大页从"优先级很低"改判为 **P2 之前值得做一次的实验**：
如果它能把路径 B 的 12% 差距抹掉，§4.1 的容量取舍就变成白拿的。

---

## 3. MoE kernel（design §7.9）

原始数据：`bench/results/kernel_p1.csv`（60 个变体 × 路径 A）。

测的是 §7.9 的两个 dispatch，7 个 expert（6 routed + 1 个用 FP4 routed expert 顶替的
shared——真正的 shared 是 fp8，属于另一个模板实例）：

- **Dispatch A** `moe_gateup`：w1/w3 的 FP4 GEMV + E8M0 块 scale + §2.4 的 clamp +
  SwiGLU + 路由权重 → `h`。每次迭代读 **87,736,320 B**。
- **Dispatch B** `moe_down`：w2 的 FP4 GEMV + 7 个 slot 的 fp32 归约（无原子）→ `y`。
  每次迭代读 **43,868,160 B**。
- 合计 **131,604,480 B ≈ 131.6 MB**，与 design §7.14 的"6 × 12.5 + 23.6 MB"一致。

有效 GB/s = 这些字节 ÷ GPU timestamp 时间。

### 3.1 正确性（design §12 L1）

`tests/test_gpu_moe.cpp`（`ctest -R suite.gpu_moe`，需要 `DEEPMOE_MODEL_DIR`）把真实
expert `(0,0)` 与 `(39,383)` 用 IoEngine 读进 **GPU 可见的 slab**，按指针表取址，
跑两个 kernel，与 `tools/oracle.py` 的 torch fp32 结果比对。

| 变体 | cos | max\|Δy\| | 占 \|y\|max |
|---|---|---|---|
| M=1，wave32，常量表解码，fp16 h（基准） | 0.999999961 | 2.61e-3 | 1.37e-4 |
| 16 / 32 / 64 lanes per row | 同上 | 2.61e-3 | 1.37e-4 |
| wave32 / wave64 | 同上 | 2.61e-3 | 1.37e-4 |
| 三种 FP4 解码变体 | 同上 | 2.61e-3 | 1.37e-4 |
| RowsPerLane = 1 / 2 / 4 | 同上 | 2.61e-3 | 1.37e-4 |
| **fp32 h** | 0.999999971 | **2.19e-3** | **1.15e-4** |
| M=6（每列同一个 x） | 同上 | 2.61e-3 | 1.37e-4 |
| `(39,383)`，两个 slot 里选一个 | 0.999999935 | 3.26e-3 | 3.67e-4 |
| 同一个 expert 分 0.25 + 0.75 两个 slot | 0.999999962 | 2.71e-3 | 1.42e-4 |

十四个变体全部 ≤ 1e-3（§12 L1 的判据），**而且彼此完全一致**。

**误差来自哪里**：`x` 从 fp32 舍入到 fp16 的相对 L2 误差是 **2.04e-4**，而输出误差
1.37e-4——**几乎全部误差都是输入量化，kernel 自身的累加误差可以忽略**
（同一个 oracle 下 CPU fp32 路径是 4.5e-7，见 design §15 P0 行）。
这正是 §6 定的"激活 fp16 传递、fp32 归约"的代价，符合预期。

**`h` 用 fp16 还是 fp32**：fp32 h 把输出误差从 1.37e-4 降到 1.15e-4（改善 16%），
代价是 `h` 流量翻倍和 dispatch B 的读取指令翻倍，实测**慢 9%**
（最好的 fp32 变体 198.5 GB/s / 0.663 ms，对最好的 fp16 变体 218.5 GB/s / 0.602 ms）。
**决定：`h` 用 fp16。** 两个误差都远在 1e-3 判据内，误差主线是输入的 fp16 量化而不是
`h`，把 `h` 加宽只是在小数点后第五位上花 9% 的时间。
（若 L2 逐层比对显示误差累积超预期，`HPrecision = 1` 这个 spec 常量原地可切。）

M=6 时各列之间有 1–2 ULP 的差别（相对 2.4e-7），来自编译器对 M 条独立累加链做 FMA
收缩的方式不同，不是错误；测试对列间一致性的判据因此是 1e-6 而不是逐位相等。

### 3.2 M=1（decode）：FP4 解码方式是最大的变量

工作集 1053 MB，路径 A。**raw-read 上限 214.4–214.5 GB/s**（sweep 前后）。

| 变体 | A GB/s | B GB/s | **A+B GB/s** | A+B ms |
|---|---|---|---|---|
| **L32 R1 sg32 dec0 fp16h** | 222.0 | 213.2 | **218.5** | **0.602** |
| L16 R1 sg32 dec0 fp16h | 212.5 | 207.9 | 211.1 | 0.623 |
| L64 R1 sg32 dec0 fp16h | 222.8 | 185.7 | 210.8 | 0.624 |
| L64 R1 sg64 dec0 fp16h | 210.3 | 198.8 | 207.2 | 0.635 |
| L32 R1 sg64 dec0 fp16h | 205.8 | 202.3 | 204.1 | 0.645 |
| L16 R1 sg32 dec0 **fp32h** | 211.6 | 170.6 | 198.5 | 0.663 |
| L32 R2 sg32 dec0 fp16h | 197.3 | 194.1 | 197.2 | 0.667 |
| L32 R4 sg32 dec0 fp16h | 185.6 | 193.1 | 189.8 | 0.693 |
| L16 R1 sg64 **dec1** fp16h（算术解码） | 159.8 | 143.3 | 164.2 | 0.802 |
| L32 R1 sg32 dec1 fp16h | 163.2 | 121.7 | 158.5 | 0.830 |
| L32 R1 sg64 **dec2** fp16h（select 树） | 155.4 | 122.2 | 151.4 | 0.869 |
| L32 R1 sg32 dec2 fp16h | 138.6 | 113.7 | 134.3 | 0.980 |

完整 60 行见 CSV。读法：

1. **解码方式差 1.4 倍。** `dec0`（`static const float kE2M1[16]` 常量数组下标）
   218.5 GB/s > `dec1`（位运算直接拼 fp32 的指数/尾数）164.2 > `dec2`（显式 select 树）151.4。
   直觉会说"查表要访存、算术更快"，实测完全相反：AMD 的着色器编译器把 16 项常量数组变成了
   比 5–7 条 ALU 指令更便宜的东西，而手写的算术链它只能照做。
   **这条只能靠实测，推不出来**，也是整个 sweep 里唯一"必须选对"的旋钮。
2. **LanesPerRow 16 / 32 / 64 之间只差 4%**，32 最好。design §7.1 rule 3 猜的
   "16 lane 一行、读粒度 256 B 更好"没有兑现——在 220 GB/s 这个水平上粒度已不是限制。
3. **Wave32 vs Wave64 没有系统性差异**（同变体下互有胜负，差 1–5%）。
   §7.1 rule 4 的"Wave32 优先"**既无数据支持也无反证**；建议保留 32 作为默认，
   理由只是它让 LanesPerRow=32 的行组正好落在一个 subgroup 内。
4. **RowsPerLane 在 M=1 时没有收益**（R1 218.5 > R2 197.2 > R4 189.8）。
   最初在有背景负载的测量里看起来"R 能帮 dispatch B"，干净重测之后**这个结论被推翻**——
   M=1 时每个 lane 每块只读 4 个 `uint4` 的 x，L0 完全吸收，不是瓶颈。
   这个旋钮留着是因为它在 M=6 时确实有用（§3.3）。

**离上限还有多远**：218.5 GB/s vs raw-read 214.4–214.5 GB/s = **102%**。
**MoE kernel 比 raw-read shader 还快**，所以 §15 P2 的"≥ 80% of raw read"不但达成，
而且 raw-read shader 本身已经不是有意义的上限了——两者都撞在内存系统的
~217 GB/s 上。原因：raw-read 每 lane 每次迭代发 4 条 `uint4` 读，MoE kernel 同时有
w1 / w3 / scale 三条独立地址流，在途请求更多。
**"216–218 GB/s"应读作这颗 APU 的 LPDDR 流式读上限，而不是某个 shader 的上限。**

顺带：A+B 合起来测得的 218.5 GB/s 高于 A 单独（222.0）与 B 单独（213.2）的加权——
不是测量错误。单独测 B 时 `h` 是冷的，合并测时 `h` 刚被 A 写过还在 cache 里。
**合并的数字才是 decode 真实会经历的。**

### 3.3 M=6（投机验证批）：瓶颈换成了激活读取

| 变体 | A GB/s | B GB/s | A+B GB/s | A+B ms（6 个 token） | ms / token |
|---|---|---|---|---|---|
| **L16 R2 sg32 dec0 fp16h** | 135.1 | 107.9 | **135.1** | **0.974** | **0.162** |
| L16 R2 sg64 dec0 fp16h | 144.1 | 105.9 | 132.7 | 0.992 | 0.165 |
| L16 R2 sg32 dec0 fp32h | 132.1 | 108.0 | 131.4 | 1.002 | 0.167 |
| L32 R2 sg64 dec0 fp16h | 135.4 | 102.6 | 125.1 | 1.052 | 0.175 |
| L32 R1 sg64 dec0 fp16h | 132.2 | 91.7 | 120.9 | 1.088 | 0.181 |
| L16 R1 sg32 dec0 fp16h | 135.0 | 90.5 | 119.1 | 1.105 | 0.184 |
| L16 R1 sg64 dec1 fp16h | 88.7 | 83.3 | 87.1 | 1.511 | 0.252 |
| L16 R1 sg32 dec2 fp16h | 76.0 | 74.0 | 74.8 | 1.759 | 0.293 |
| （对照）M=1 最佳 | 222.0 | 213.2 | 218.5 | 0.602 | 0.602 |

- 有效带宽掉到 **135.1 GB/s = 上限的 63%**：M=6 **不再是带宽受限**。
- 但每 token 的 MoE 时间 **0.602 → 0.162 ms，快 3.7 倍**。§10.1 说"投机解码是唯一能把
  常驻部分摊到多个 token 上的手段"——实测比它估的还好，因为权重只读一次而不是六次。
- **瓶颈是激活的读取指令，不是 ALU。** M=6 时每个 32 元素块，一个 lane 要发 6 条
  `uint4` 读拿 x（96 B），却只读 32 B 的权重：**L0 请求量是权重的 12 倍**。
  证据是 `RowsPerLane = 2` 在 M=6 时**有 13% 的收益**（119.1 → 135.1），
  而它在 M=1 时是负收益（§3.2 第 4 点）——R 省的正是这些重复的 x 读，
  只有在 x 读真的成为瓶颈时才有意义。这两个方向相反的结果互相印证了瓶颈的位置。
- **改进方向（P2/P4 之前）**：把 x 的一个 K 分块搬进 LDS
  （`M × LanesPerRow × 32` 个 half，M=6 / L=16 时 6 KiB），全 workgroup 共享。
  design §7.1 rule 6 本来就写了"激活放 LDS"，本实现当初因为 `x[6][5120] = 60 KiB`
  放不下 32 KiB 而放弃；**分块之后放得下**。
  §6 的 int8 dot4 备选路径是第二条路，但**先做 LDS 分块**——数据说瓶颈在访存不在 ALU。

### 3.4 每 dispatch 的启动 + barrier 开销（design §3.4）

在一个 command buffer 里连发 1024 个空 dispatch（1 个 workgroup，读循环不执行），
每两个之间放一个 §7.1 的全局 shader-write → shader-read barrier：

| 指标 | 实测 |
|---|---|
| GPU 侧，每 dispatch + barrier | **0.56–0.66 µs** |
| CPU 侧，录制一对 A+B dispatch（bind + push + dispatch + barrier ×2） | **1.1 µs** |
| CPU 侧，提交 + 等待整个 command buffer | 48 次迭代下占 GPU 时间的 < 5% |

**design §3.4 的 5–20 µs 高了一个数量级。** 按实测重算它那笔账：
每 token ~470 个 dispatch × 0.66 µs = **0.31 ms**，占 65 ms 的 **0.5%**，
而不是 §3.4 担心的 5–14%。

后果：

- "每 token 一个预录制的 command buffer"仍然值得做（它同时解决了 timeline wait 的
  表达问题），但**不再是性能上的必需品**：即使每 token 重新录制，
  CPU 侧 470 × 0.55 µs ≈ 0.26 ms 也可以接受。
- §7.9 末尾"把 Dispatch A/B 按 expert 拆成两组、先到的 3 个先算，代价是 dispatch 数翻倍"
  ——**翻倍的代价是 0.66 µs × 2 × 40 层 ≈ 0.05 ms/token，可以忽略**。
  这个优化现在是纯收益，应该在 P3 做。kernel 侧已经支持：
  indirection list（`SlotList` + `list_count`）短一点即可，不用改 kernel。

### 3.5 路径 A vs 路径 B 下的 kernel：**raw-read 看不出来的 12%**

同一个最佳变体（`M1 L32 R1 sg32 dec0 fp16h`）在两条路径上交替测，机器空闲：

| 轮次 | 路径 | raw-read GB/s | **kernel A+B GB/s** | A+B ms | NVMe → 内存 GB/s |
|---|---|---|---|---|---|
| 1 | A | 217.6 | **222.1** | 0.593 | 3.69 |
| 2 | B | 216.1 | **195.2** | 0.674 | 4.76 |
| 3 | B | 214.8 | 198.2 | 0.664 | 4.77 |
| 4 | A | 217.4 | 222.1 | 0.592 | 3.87 |
| 5 | B | 215.7 | 193.6 | 0.680 | 4.82 |
| 6 | A | 217.3 | 221.8 | 0.593 | 3.97 |

完全可复现：**路径 A 221.8–222.1，路径 B 193.6–198.2，B 慢 11–13%**，
而两条路径的 **raw-read 一模一样**（214.8–217.6）。

**为什么 raw-read 看不出来**：raw-read 的每个 workgroup 顺序走自己那片连续内存，
在途只涉及很少几个 4 KiB 页，GART TLB 命中率接近 100%。
MoE kernel 则同时在 7 个 expert 的 2304 行上推进，每行只有 2560 B，
一个 workgroup 在一个瞬间就横跨几十个 4 KiB 页——
**导入内存的页表走查成本只在真实访问模式下才会显形**。
这条同时说明 §7.1 rule 2 的"用 raw-read 当上限"有个盲区：
它能告诉你 kernel 离带宽有多远，但不能替 kernel 回答"这块内存好不好用"。

每 token 的代价：40 层 × (0.674 − 0.593) ms = **+3.2 ms/token**（§4.1 用这个数做取舍）。
（这六轮是 `--quick` 的单变体重复测，所以路径 A 读到 222.1 而不是 60 变体 sweep 里的
218.5——同一个变体、不同的测量批次，差 1.6%，见 §1 的复现性说明。）

反方向：**路径 B 的 NVMe 落地反而快 22%**（4.8 vs 3.9 GB/s），
因为目标是普通可缓存内存而不是写合并的 device 内存（与 §2.4 的 CPU 写入一致）。

---

## 4. 决定

### 4.1 路径 A vs 路径 B：**仍然建议最小 VGM + 路径 B，但代价现在有数字了**

| 维度 | 路径 A | 路径 B | 胜者 |
|---|---|---|---|
| raw-read 带宽 | 216.4 GB/s | 215.2 GB/s（−0.5%） | 平 |
| **MoE kernel 有效带宽** | **222.1 GB/s** | **195.2 GB/s（−12%）** | **A** |
| 并发时 GPU / CPU 各拿多少 | 185.8 / 28.6 | 184.4 / 28.7 | 平 |
| CPU 写入（staging 场景） | 21.5 GB/s | 32.7 GB/s | **B** |
| NVMe 直读落地 | 3.9 GB/s | 4.8 GB/s（+22%） | **B** |
| 分配上限 | 2 GiB / 次 | 同 | 平 |
| 需要的权限 | 无 | 无（大页才需要） | 平 |
| **容量** | 受 BIOS VGM 约束的 GPU heap | 系统内存 | **看 VGM** |

**取舍的算术**。路径 B 的 kernel 代价是 40 层 × 0.081 ms = **3.2 ms/token**。
换来的是 cache 容量：

- **VGM = 64 GB（当前）**：GPU device-local heap 74.4 GiB，减去 §2.2 的 pin 集合
  17.7 GB，路径 A 的 expert cache 约 **56 GB**；Windows 只看到 63.6 GB 系统内存，
  减掉 OS / I/O staging / engram 行缓冲，路径 B 只剩约 **45–50 GB**。
  **这个设置下路径 A 在容量和带宽上都赢，应该用路径 A。**
- **VGM = 最小值（待实测）**：device-local heap 缩到几 GB，系统内存涨到 ~120 GB，
  路径 B 的 cache 可达 **~90–95 GB**，比路径 A 在 VGM=64 下多 60%
  （≈2,980 → ≈4,780 个 expert，19% → 29% 的驻留率）。

换算成时间：按 §9.2.1 的 4.7 GB/s，**hit rate 每提高 1 个百分点省
`0.01 × 4.51 GB / 4.7 GB/s = 9.6 ms/token`**。
路径 B 的 3.2 ms 代价因此只相当于 **0.33 个百分点的 hit rate**——
只要多出来的 34–45 GB cache 能换来哪怕半个百分点的命中率，路径 B 就赢，
而按 §3.1 的表它应该能换来 5–10 个百分点。

**结论：**

1. **现在（VGM = 64 GB）用路径 A**：它在带宽和容量上都更好。
   `RuntimeConfig::memory_path` 的默认值 `Auto` 已经会选到它。
2. **调到最小 VGM 之后改用路径 B**，接受 3.2 ms/token 的 kernel 代价换 cache 容量。
3. **先试大页**（§2.6）：如果它抹掉了那 12%，第 2 条就变成白拿。

这个建议还缺最后一块拼图——最小 VGM 下两个 heap 的实际大小——**需要一次重启**；
但由于 raw-read 已经证明两条路的上限相同，那次重测只需要回答"最小 VGM 下系统内存是不是
真的到了 ~120 GB"，外加在路径 B 上重跑一次 `kernel_bench --quick` 确认那 12% 不变。

设计上不需要任何改动：`ExpertStore` 通过 `store::SlabBacking` 只看到
`(host_ptr, device_address)`，`gpu::MemoryAllocator::allocate_slab` 按 `MemoryPath` 分派，
`RuntimeConfig::memory_path` 一个字段就能切换，
`tests/test_gpu_core.cpp` 对两条路径都验证了这对指针。

### 4.2 最佳 kernel 变体

```
decode（M=1）   : LanesPerRow=32, subgroup=32, DecodeMode=0（常量表）, h=fp16, RowsPerLane=1
投机验证（M=6） : LanesPerRow=16, subgroup=32, DecodeMode=0（常量表）, h=fp16, RowsPerLane=2
```

`DecodeMode=0` 在两个 M 下都最优且优势很大（1.4–1.8 倍），是唯一"必须选对"的旋钮；
`RowsPerLane` 随 M 变（M=1 用 1，M=6 用 2）；其余旋钮的影响都在 ±5% 以内。

### 4.3 T_layer 与 design §9.4 的 `d`

实测（路径 A，7 个 expert 全驻留）：

```
T_layer(MoE 部分, M=1) = 0.602 ms      （131.6 MB @ 218.5 GB/s）
T_layer(MoE 部分, M=6) = 0.974 ms      （0.162 ms / token）
```

把 §7.14 的整层清单按实测的 217 GB/s 折算：一层常驻 + 命中权重 ≈ 210–280 MB
→ **T_layer(整层) ≈ 0.97–1.29 ms**，与 §3.4 估的 1.0–1.4 ms 吻合。

§9.4 的提前量公式 `d × T_layer ≥ T_io(18.8 MB) × expected_misses / QD_effective`，
用 §9.2.1 实测的 `T_io ≈ 4.0 ms`：

| 要隐藏的 | 需要的 d（整层 1.13 ms） | 需要的 d（只算 MoE 0.594 ms） |
|---|---|---|
| 1 个 miss | **3.5 → d ≥ 4** | 6.6 → d ≥ 7 |
| 一层 6 个全 miss（§9.2.1 实测并发 23.7 ms） | 21 → **不现实** | 40 |

**§9.4 写的 `d ≥ 3–4` 在"每层至多一个 miss"的前提下成立，实测确认；
`PrefetchConfig::lookahead_depth = 4` 保持不变。**
但一层 6 个全 miss 需要 21 层提前量，超过模型总共 40 层里能用的窗口——
这从另一个角度重述了 §3.1 结论 1：**提前量救不了低命中率，只有命中率本身能救。**

M=6 时 T_layer 只涨到 0.974 ms（不是 6 倍），所以**投机解码同时放大了每次 I/O 能被
隐藏的窗口**：同样 4 ms 的 T_io，M=6 下 `d ≥ 4.0 / 0.974 = 4.1 → d ≥ 5`，
和 M=1 几乎一样，但这几层掩护的是 6 个 token 的进度。

---

## 5. 代码层面：store / 指针表契约

**没有需要修改的契约。** design §5.3 / §5.4 定的指针表结构原样可用：

- `ExpertStore::pointer_table()` 的 `uint64[layer][expert][6]` 直接 `memcpy` 进 GPU 的
  storage buffer，kernel 用 `table[(layer * experts_per_layer + expert) * 6 + part]` 取址。
  kernel 需要 `experts_per_layer`（表第二维的步长）作为 push constant——
  这是新增的一个 push constant 字段，不是契约变更。
- `publish_locked()` 里"有 `dev_addr` 就用 `dev_addr`，否则用 host 指针"的写法，
  让同一份表在主机 backing（测试 / CPU oracle）和 Vulkan backing（GPU）下都成立，
  实测无需改动。`tests/test_gpu_core.cpp` 的
  `gpu.the_expert_store_runs_on_a_vulkan_slab_backing` 固定了这个行为。

有三条**被实测钉死、之前只是纸面约定**的约束，写在这里以免将来被改坏：

1. **part 地址只有 8 字节对齐，不是 16。** 扫全量 manifest（15,744 个 expert、
   94,464 个 part）：**87,552 个（92.7%）的槽内偏移 mod 16 = 8**，只有 7.3% 是 16 对齐；
   所有 skew 都是 8 的倍数（94,464 / 94,464）。kernel 每 lane 每块读 16 B（`uint4`），
   所以这些读**必然**是 8 对齐而非 16 对齐。Slang 的指针解引用发出的是
   `OpLoad … Aligned 4`（SPIR-V 合法），RDNA 的 `global_load_dwordx4` 只要求 dword 对齐
   ——**实测零代价**（102% of raw read）。
   这是 §5.1 v0.5"不 repack"决定的全部 kernel 侧代价，账已结清。
2. **slab 必须同时是 host 可写和 device 可寻址的。** `VulkanSlabBacking::allocate()`
   拿到一个没有 `host_ptr` 的 slab 时直接返回 `Internal` 错误而不是默默接受——
   否则 IoEngine 会往空指针写。纯 `DEVICE_LOCAL` 内存（§2.2 里那条 216.0 GB/s）
   **不能**用作 expert slab，尽管它和路径 A 一样快。
3. **每个 slab ≤ `maxMemoryAllocationSize` = 2 GiB。** `SlabConfig::slots_per_slab`
   的上界因此是 `2 GiB / 18,808,832 = 114`；默认的 100（1.88 GB）安全。
   `MemoryAllocator::allocate_from_type()` 显式检查并给出指向 §5.3 的错误消息，
   `tests/test_gpu_core.cpp` 回归这一条。

另外，为了让 kernel 能编译和运行，`gpu/vulkan/device.cpp` 现在额外申请：
`shaderInt16`（storage buffer 里的 `half`）、`shaderInt64`（PhysicalStorageBuffer 寻址）、
`storageBuffer16BitAccess` / `uniformAndStorageBuffer16BitAccess`、
`synchronization2`（`vkCmdPipelineBarrier2` / `vkCmdWriteTimestamp2`）。
`DeviceCaps::check_required()` 相应加了检查。这些在 design §1.1 的能力表里没列，
gfx1151 全部支持。**注意 `VkPhysicalDeviceVulkanNNFeatures` 与被它取代的扩展结构体
不能同时出现在 pNext 链里**（VUID-VkDeviceCreateInfo-pNext-02830），
这是 validation layer 抓到的第一个错误。

**一处偏离 architecture.md §1.2**：`gpu/vulkan/memory.cpp` 里有一个 `#if defined(_WIN32)`
块（`VirtualAlloc` / 大页 / `SeLockMemoryPrivilege`）。架构文档说只有 `storage/windows/`
与 `storage/linux/` 可以有平台分支。理由是 design §3.3 明确要求路径 B 在 VirtualAlloc
内存和大页上测量，而 gpu/ 在依赖 DAG 上不能引用 storage/windows/；
块内有可移植的 aligned-new 回落，Linux 交叉编译通过。
要消除这个偏离，正确做法是把"页分配器"下沉到 `core/`。

---

## 6. 未解决的问题

1. **最小 VGM 下的容量重测**（§4.1）。需要进 BIOS 改 UMA Frame Buffer Size 并重启。
   只需回答"系统内存是否到 ~120 GB"，带宽部分不必重跑。**这是 P2 之前唯一的阻塞项。**
2. **M=6 的 x 分块进 LDS**（§3.3）。M=6 现在只有上限的 63%，瓶颈是 x 的 L0 请求量。
   这是投机解码收益最大的一块，应在 P4 之前做。
3. **两个 dispatch 分别特化**（§3.2 第 4 点、§3.3）。`MoeSpec` 目前对 A 和 B 用同一份
   spec，但 M=6 时 A 和 B 的最优 `RowsPerLane` 未必相同。一行改动。
4. **shared expert 的 fp8 模板还没写**。本文 7 个 expert 里第 7 个是 FP4 顶替的；
   真正的 shared expert 是 fp8 E4M3 + 32×32 块 scale，是 §7.9 的另一个模板实例
   （23.6 MB 而不是 12.5 MB），有效带宽需要单独测。
5. **大页下的路径 B**（§2.6、§3.5）。需要 secpol.msc 授予"锁定内存页"并重新登录。
   潜在收益是路径 B 上 MoE kernel 的那 **12%**（不是 raw-read 的 0.5%），
   如果兑现，§4.1 的取舍就变成白拿。**优先级从"低"上调为"P2 之前做一次"。**
6. **raw-read shader 不再是有意义的上限**（§3.2 末）。MoE kernel 已经超过它。
   若还想要一个"机器上限"，应写一个多地址流版本的 rawread；
   或者接受"216–218 GB/s 就是 LPDDR 的实际上限"这个结论，本文取后者。
7. **测量环境的隔离**（§1）。同机的 CPU 负载会让 GPU 数字浮动 25–50%。
   P2/P3 的回归测量需要一个"安静窗口"检查，否则 CI 上的 kernel 带宽数字没有意义。

---

## 附：如何复现

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'

# 正确性：Vulkan 管道（不需要权重）+ 十四个变体对照 oracle（需要权重）
ctest --test-dir build -R "suite.gpu$" --output-on-failure
ctest --test-dir build -R suite.gpu_moe --output-on-failure

# 带宽矩阵（design §9.2 / §3.3）
.\build\bw_matrix.exe --cpu-size-gb 4 --size-gb 1 --repeats 3

# kernel sweep（design §7.9 / §7.1）
.\build\kernel_bench.exe --csv bench\results\kernel_p1.csv `
    --iters 48 --layer-cycle 8 --repeats 4 --sweeps 2
.\build\kernel_bench.exe --path b --quick        # 路径 B 对照
```

带 validation layer 跑一遍（两个 bench 与两个 GPU 测试当前都是干净的）：

```powershell
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
$env:VK_LOADER_LAYERS_ENABLE='VK_LAYER_KHRONOS_validation'
```

**测量时不要同时编译，也不要跑任何吃内存带宽的进程**（§1）。
