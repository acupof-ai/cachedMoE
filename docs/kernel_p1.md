# P1：GPU 微内核与内存路径实测

本文是 design.md §3.3（UMA 两条路径）、§7.1/§7.9（MoE kernel）、§9.2（带宽矩阵）、
§3.4（dispatch 开销）几个问题的**实测答案**。所有数字来自本机，原始 CSV 在
`bench/results/`。design.md 与 README 由本文的结论另行整合。

状态：v0.1（2026-09-14）。测量机器：Ryzen AI Max+ 395 / Radeon 8060S (gfx1151) /
128 GB LPDDR5X / **BIOS VGM = 64 GB**。Vulkan 1.4.349，Adrenalin 32.0.31041.1004。

---

## 0. 先看结论

| 问题 | design 的假设 | 实测 | 结论 |
|---|---|---|---|
| GPU 读带宽上限 | §3.1 假设 200 GB/s 有效 | **216–218 GB/s**（raw-read shader） | 假设略保守，成立 |
| 路径 A vs B 的 GPU 读带宽 | §3.3 待验证 | A 216.6 / B 214.2 GB/s，**差 1.1%** | **带宽不是选择依据**，容量是 |
| MoE kernel 有效带宽 | §15 P2 要求 ≥ raw-read 的 80% | **216.1 GB/s = raw-read 的 99%**（M=1 最佳变体） | 达标，且已到内存系统上限 |
| 每 dispatch 启动 + barrier | §3.4 猜 5–20 µs | **0.56–0.66 µs**（GPU 侧），CPU 录制 ~1.1 µs/两个 dispatch | **低一个数量级**，§3.4 的 5–14% 顾虑取消 |
| CPU 读带宽 | §3.1 隐含与 GPU 同量级 | **101.9 GB/s**（32 线程峰值） | 只有 GPU 的一半，CPU 不可能分担 GEMV |
| NVMe 直读进 GPU 内存 | §9.6 零拷贝 | **跑通**，3.0–4.0 GB/s，与 §9.2.1 的盘上限一致 | 零拷贝成立 |

两个**出乎意料**的结果，下面各有一节：

1. **FP4 解码方式决定一切**：常量表解码 216 GB/s，算术构造 155 GB/s，显式 select 树
   138 GB/s。同一个 kernel，同样的字节数，差 1.6 倍（§3.2）。
2. **M=6 时瓶颈是激活读取而不是权重**，有效带宽掉到 125 GB/s；但每 token 的 MoE 时间
   从 0.609 ms 降到 0.175 ms，投机解码的收益比 §10.1 估计的还大（§3.4）。

---

## 1. 测量方法（先说清楚噪声）

**LPDDR 是 CPU 和 GPU 共用的，所以任何吃内存带宽的 CPU 进程都会污染 GPU 的数字。**
测量期间本机上另一个 agent 在跑 `tools/route_trace.py`（torch CPU 前向），CPU 占用
约 55%，同一个 kernel 变体在忙/闲两种状态下相差最多 **50%**（如
`M1 L32 R1 sg32 dec1`：闲时 155 GB/s，忙时 94.5 GB/s）。因此：

- 本文的所有数字都来自**机器空闲窗口**，判据是 raw-read 上限在 sweep 前后一致
  （最终一轮：sweep 前 218.0 GB/s，sweep 后 216.7 GB/s，差 0.6%）。
- `bench/kernel_bench` 的方法学固定为：每个变体先 warm-up 8 次迭代，取 N 次测量中**最快**
  的一次；整张变体表**扫两遍**取每个变体的最好值，避免"先跑的变体占便宜"。
- GPU 时间一律用 timestamp query（`timestampPeriod = 10.0 ns`），墙钟时间并排打印，
  两者之差就是提交开销。
- 工作集用 `--layer-cycle N` 在 N 层 × 7 个 expert 之间轮转（默认 8 层 = 1053 MB），
  这样 32 MB 的 MALL 装不下，读的确实是 DRAM。用 1 层（132 MB）会把带宽**虚高约 25%**。

顺带得到一个 design §8 需要的数字：**一个吃内存的 CPU 进程会让 GPU 损失 25–30% 的读带宽**。
这比 §9.2 的"CPU+GPU 并发"矩阵里那条受控的并发读更贴近真实负载。

---

## 2. 带宽矩阵（design §9.2 / §3.3，VGM = 64 GB）

原始数据：`bench/results/bw_matrix.csv`。

### 2.1 CPU 单独顺序读（普通主机内存）

| 线程 | GB/s |
|---|---|
| 1 | 24.6 |
| 2 | 43.0 |
| 3 | 52.3 |
| 4 | 52.9 |
| 8 | 54.9 |
| 16 | 77.0 |
| 32 | **101.9** |

**峰值 101.9 GB/s，只有理论 256 GB/s 的 40%，也只有 GPU 的 47%。** 8 线程到 16 线程
之间有一个平台（54.9 → 77.0），说明 4 个 CCX/线程组之外才继续拿到带宽。

对 design §8 的直接后果：**"CPU 分担 expert GEMV"（§8 P6 实验）从带宽上就不成立**——
把一份权重交给 CPU 算，它的读带宽只有 GPU 的一半，还要和 GPU 抢同一条内存总线。
§8 把它降级为 P6 是对的，本文建议直接标记为**不做**，除非双盘/更高 VGM 改变前提。

### 2.2 GPU 单独读（`gpu/shaders/rawread.slang`，1 GiB，320 workgroup）

| 内存 | GB/s | 备注 |
|---|---|---|
| **路径 A** `DEVICE_LOCAL\|HOST_VISIBLE` (memory type 2, heap 1) | **216.6** | vkMapMemory 映射，CPU 可写 |
| **路径 B** `VK_EXT_external_memory_host`（VirtualAlloc 4 KiB 页） | **214.2** | 比 A 低 1.1% |
| 路径 B（大页） | — | **不可用**：本账户没有 SeLockMemoryPrivilege |
| 纯 `DEVICE_LOCAL`（memory type 0，host 不可见） | **216.7** | 与 A 完全一致 |

三条线在 1% 以内。**在这颗 APU 上不存在"真 VRAM 更快"**——device-local-only 和
device-local+host-visible 是同一块 LPDDR，驱动只是换了缓存属性标签。
路径 B 的 1.1% 折扣是 GART/页表的代价（1 GiB = 262,144 个 4 KiB 页）。

`216.7 / 256 = 85%` 的理论带宽利用率，对 LPDDR5X-8000 的流式读是正常水平。
**这就是 §3.1 里"常驻 42 ms"那一项的真实分母**：8.5 GB / 216 GB/s = **39.4 ms**，
比 §3.1 用 200 GB/s 算出的 42 ms 略好。

### 2.3 CPU / GPU 并发

| GPU 读的内存 | GPU GB/s | 同时 CPU GB/s | 合计 | 占"各自单独"之和 |
|---|---|---|---|---|
| 路径 A | 186.9 | 75.6 | 262.6 | 82% |
| 路径 B | 184.9 | 49.6 | 234.4 | 74% |
| 纯 DEVICE_LOCAL | 186.2 | 34.4 | 220.6 | 69% |

三种情况下 GPU 都稳定在 ~186 GB/s（单独时的 86%），**被牺牲的是 CPU**：
路径 A 下 CPU 还能拿 75.6 GB/s，纯 device-local 下只剩 34.4 GB/s。
也就是说仲裁器偏向 GPU，而且**路径 A 对 CPU 最友好**——这正好是 IoEngine 完成线程
和 Planner 需要的（它们在 GPU 满负荷时仍要推进 I/O）。

### 2.4 CPU 写入（NVMe 落地模拟，design §9.6）

单线程 256 MB memcpy：

| 目标 | memcpy | 非临时存储（`_mm512_stream_si512`） | 18.8 MB 一个 expert |
|---|---|---|---|
| 路径 A 的映射内存 | 18.5 GB/s | 18.9 GB/s | 1.0 ms |
| 路径 B 的主机内存 | 18.9 GB/s | **31.6 GB/s** | 0.6 ms |

- 路径 A 的映射是**写合并且不可缓存**的，普通 store 本来就被硬件合并，所以非临时存储
  没有任何收益（18.5 → 18.9，噪声内）。
- 路径 B 是普通可缓存内存，非临时存储省掉 RFO，**快 1.7 倍**。
- 两条路都远超 NVMe 的 4.7 GB/s（§9.2.1），**CPU 拷贝不会成为落地瓶颈**。这只在需要
  staging 的路径上有意义；真正的零拷贝路径（下节）连这一次拷贝都没有。

### 2.5 NVMe 直读进 GPU 内存：零拷贝成立

`bench/kernel_bench` 每次启动都做一次：用**真正的** `IoEngine` + IOCP +
`FILE_FLAG_NO_BUFFERING`，把 56 个真实 expert（1053 MB）直接读进
路径 A 的映射 slab，不经过任何中间缓冲。

| 路径 | 56 个 expert | 有效 GB/s |
|---|---|---|
| A（映射的 device 内存） | 0.266–0.406 s | 2.6–4.0 |
| B（导入的主机内存） | 0.300–0.329 s | 3.2–3.5 |

**两条路径都能直接作为 `ReadFile(OVERLAPPED | NO_BUFFERING)` 的目标缓冲**，
`vkMapMemory` 返回的指针天然页对齐，满足扇区对齐要求。design §9.6 的"目标缓冲直接是
slab 槽、零拷贝"从假设变成事实。

带宽 2.6–4.0 GB/s 低于 §9.2.1 的 4.5–4.7 GB/s 平台，原因是这里是 56 个 expert ×
2 个 run = 112 个请求依次下发、每个 expert 之间没有重叠下单（bench 的填充循环是
串行的，不是 Planner 的并发队列）。不是路径 A/B 的问题。

### 2.6 大页：这台机器上拿不到

`VirtualAlloc(MEM_LARGE_PAGES)` 需要 `SeLockMemoryPrivilege`，本账户没有，
运行时也无法自行授予（需要 secpol.msc → "锁定内存页" + 重新登录）。
代码里的尝试是完整的（`gpu::alloc_host_pages(bytes, try_large_pages=true)`），
拿不到时会把原因写进 `HostAllocInfo::note` 并回落到 4 KiB 页，CSV 里记为
`large pages unavailable: SeLockMemoryPrivilege is not held by this account`。

**这是一个未回答的问题**：路径 B 相对路径 A 的 1.1% 差距和它在并发下对 CPU 更不友好，
都指向 GART 页表；2 MiB 大页很可能把这 1.1% 抹平。要回答它需要改系统策略并重启。

---

## 3. MoE kernel（design §7.9）

原始数据：`bench/results/kernel_p1.csv`（60 个变体 × 路径 A）、
`bench/results/kernel_p1_pathb.csv`。

测的是 §7.9 的两个 dispatch，7 个 expert（6 routed + 1 个用 FP4 routed expert
顶替的 shared——真正的 shared 是 fp8，是另一个模板实例）：

- **Dispatch A** `moe_gateup`：w1/w3 的 FP4 GEMV + E8M0 块 scale + §2.4 的 clamp +
  SwiGLU + 路由权重 → `h`。每次迭代读 **87,736,320 B**。
- **Dispatch B** `moe_down`：w2 的 FP4 GEMV + 7 个 slot 的 fp32 归约（无原子）→ `y`。
  每次迭代读 **43,868,160 B**。
- 合计 **131,604,480 B ≈ 131.6 MB**，与 design §7.14 的"6 × 12.5 + 23.6 MB"一致。

有效 GB/s = 这些字节 / GPU timestamp 时间。

### 3.1 正确性（design §12 L1）

`tests/test_gpu_moe.cpp`（`ctest -R suite.gpu_moe`，需要 `DEEPMOE_MODEL_DIR`）
把真实 expert `(0,0)` 与 `(39,383)` 用 IoEngine 读进 GPU 可见的 slab，
按指针表取址，跑两个 kernel，和 `tools/oracle.py` 的 torch fp32 结果比。

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

**误差来自哪里**：`x` 从 fp32 舍入到 fp16 的相对 L2 误差是 **2.04e-4**，而输出误差是
1.37e-4——也就是说**几乎全部误差都是输入量化，kernel 自身的累加误差可以忽略**
（CPU fp32 路径对同一个 oracle 是 4.5e-7，见 design §15 P0 行）。这正是 §6 定的
"激活 fp16 传递、fp32 归约"的代价，符合预期。

**fp16 vs fp32 h 的判据**：fp32 h 把输出误差从 1.37e-4 降到 1.15e-4（改善 16%），
代价是 `h` 的流量翻倍（7 × 2304 × 2 B → 4 B）和 dispatch B 的读取指令翻倍。
实测 fp32 h 在最佳变体上略慢（0.696 ms vs 0.609 ms，慢 14%）。
**决定：`h` 用 fp16。** 1.15e-4 与 1.37e-4 都远在 1e-3 的判据里，误差主线是输入的
fp16 量化而不是 `h`，把 `h` 加宽只是在小数点后第五位上花 14% 的时间。
（若将来 L2 逐层比对显示误差累积超预期，`HPrecision = 1` 这个 spec 常量原地可切。）

M=6 时各列之间有 1–2 ULP 的差别（fp32，2.4e-7 相对），来自编译器对 M 条独立累加链
做 FMA 收缩的方式不同，不是错误；测试对列间一致性的判据因此是 1e-6 而不是逐位相等。

### 3.2 有效带宽：FP4 解码方式是最大的变量

M=1（decode 场景）、路径 A、工作集 1053 MB。**raw-read 上限 216.7–218.0 GB/s。**

| 变体 | A GB/s | B GB/s | **A+B GB/s** | A+B ms | 占上限 |
|---|---|---|---|---|---|
| **L32 R1 sg32 dec0 fp16h** | 213.0 | 178.5 | **216.1** | **0.609** | **99%** |
| L16 R1 sg32 dec0 fp16h | 207.4 | 179.2 | 214.2 | 0.614 | 98% |
| L64 R1 sg32 dec0 fp16h | 215.2 | 174.2 | 206.1 | 0.639 | 95% |
| L32 R2 sg32 dec0 fp16h | 196.8 | 190.4 | 198.3 | 0.664 | 91% |
| L32 R4 sg32 dec0 fp16h | 181.1 | 188.1 | 188.7 | 0.697 | 87% |
| L32 R1 sg64 dec1 fp16h（算术解码） | 175.3 | 128.5 | 162.1 | 0.812 | 74% |
| L16 R1 sg32 dec1 fp16h | 152.0 | 139.0 | 154.6 | 0.851 | 71% |
| L16 R1 sg64 dec2 fp16h（select 树） | 135.8 | 130.1 | 146.5 | 0.898 | 67% |
| L32 R1 sg32 dec2 fp16h | 138.8 | 106.8 | 130.1 | 1.012 | 60% |
| L32 R1 sg32 dec0 **fp32h** | 221.9 | 131.4 | 183.3 | 0.718 | 84% |

完整 60 行见 CSV。读法：

1. **解码方式差 1.6 倍。** `dec0`（`static const float kE2M1[16]` 常量数组下标）
   216 GB/s > `dec1`（位运算直接拼 fp32）155 GB/s > `dec2`（显式 select 树）138 GB/s。
   直觉会说"查表要访存、算术更快"，实测完全相反：AMD 的着色器编译器把 16 项常量数组
   变成了比 5–7 条 ALU 指令更便宜的东西（很可能是 `v_perm_b32` 或标量寄存器里的表），
   而手写的算术/select 链它只能照做。**这条只能靠实测，不能靠推理。**
2. **LanesPerRow 16/32/64 之间只差 5%**，32 最好。design §7.1 rule 3 猜的
   "16 lane 一行读 256 B 粒度更好"没有兑现——在 216 GB/s 这个水平上，粒度已经不是
   限制因素了。
3. **Wave32 vs Wave64 没有系统性差异**（同一变体下互有胜负，差 1–5%）。
   design §7.1 rule 4 的"Wave32 优先"**没有数据支持，也没有反证**；建议保留
   `subgroup_size = 32` 作为默认只是因为它让 LanesPerRow=32 的行组正好落在一个
   subgroup 内。
4. **RowsPerLane（每 lane 多算几行权重、激活只读一次）帮了 dispatch B，害了 A。**
   B 单独：R1 178.5 → R2 190.4 → R4 188.1（+7%）；A 单独：R1 213.0 → R2 196.8（−8%）。
   B 的激活流量（`h`，每行每 slot 都要读一遍）是它权重流量的 3.7 倍，所以 R>1 有收益；
   A 的寄存器压力更敏感。**统一用一个 spec 的话 R=1 胜出；但把两个 dispatch 分别
   特化（A 用 R=1，B 用 R=2）应当能再拿到 dispatch B 的 7%**——`MoeSpec` 目前对两个
   pipeline 用同一份 spec，这是留给 P2 的一行改动。

**离上限还有多远**：最佳变体 216.1 GB/s vs raw-read 216.7–218.0 GB/s = **99%**。
§15 的 P2 准出条件（≥ 80%）达成，而且**没有余量可挖了**：kernel 已经在内存系统的
极限上，任何进一步的 kernel 优化都不会变快。

注意 A+B 合并测得的 216.1 GB/s 高于 A 单独（213.0）与 B 单独（178.5）——不是测量
错误：单独测 B 时 `h` 是冷的，合并测时 `h` 刚被 A 写过还在 cache 里。**合并的数字才是
decode 真实会经历的**。

**raw-read 本身不是硬上限**：早期一次测量里 MoE kernel（220.7 GB/s）**超过**了
raw-read shader（216.5 GB/s）。raw-read 每 lane 每次迭代发 4 条 `uint4` 读，
MoE kernel 同时有 w1/w3/scale 三条流，未完成请求更多。所以"216–218 GB/s"应该读作
**这颗 APU 的 LPDDR 实际流式读上限**，而不是某个 shader 的上限。

### 3.3 M=6（投机验证批）：瓶颈换成了激活读取

| 变体 | A+B GB/s | A+B ms（6 个 token） | ms / token |
|---|---|---|---|
| **M6 L16 R2 sg64 dec0 fp16h** | **125.5** | **1.049** | **0.175** |
| M6 L16 R2 sg32 dec0 fp16h | 124.6 | 1.057 | 0.176 |
| M6 L32 R1 sg64 dec0 fp16h | 117.3 | 1.122 | 0.187 |
| M6 L16 R1 sg32 dec0 fp16h | 115.9 | 1.135 | 0.189 |
| M6 L16 R1 sg64 dec1 fp16h | 85.5 | 1.540 | 0.257 |
| M6 L16 R1 sg32 dec2 fp16h | 72.9 | 1.805 | 0.301 |
| （对照）M=1 最佳 | 216.1 | 0.609 | 0.609 |

- 有效带宽掉到 **125.5 GB/s = 上限的 58%**，M=6 明显**不再是带宽受限**。
- 但每 token 的 MoE 时间 **0.609 → 0.175 ms，快 3.5 倍**。§10.1 说"投机解码是唯一能把
  常驻部分摊到多个 token 上的手段"——实测比它估的还好，因为权重只读一次而不是六次。
- **瓶颈不是 ALU，是激活的读取指令。** M=6 时每个 32 元素块，一个 lane 要发
  6 次 `uint4` 读拿 x（96 B），却只读 32 B 的权重；即 **L0 请求量是权重的 12 倍**。
  这也解释了为什么 RowsPerLane 在 M=6 只能帮一点（它省的是同样的 x 读，但寄存器不够）。
- **改进方向（P2/P6）**：把 x 的一个 K 分块搬进 LDS（`M × LanesPerRow × 32` 个 half，
  M=6/L=16 时 6 KiB），全 workgroup 共享。design §7.1 rule 6 本来就写了"激活放 LDS"，
  当时因为 `x[6][5120] = 60 KiB` 放不下 32 KiB 而被本实现放弃；分块之后放得下。
  §6 的 int8 dot4 备选路径（`vpdpbusd` 类指令，每指令 4 个 MAC）是第二条路，
  但**先做 LDS 分块**，因为数据说瓶颈在访存不在 ALU。

### 3.4 每 dispatch 的启动 + barrier 开销（design §3.4）

在一个 command buffer 里连发 1024 个空 dispatch（1 个 workgroup，读循环不执行），
每个之间放一个 §7.1 的全局 shader-write → shader-read barrier：

| 指标 | 实测 |
|---|---|
| GPU 侧 每 dispatch + barrier | **0.56–0.66 µs**（机器空闲），忙时 2.2–2.5 µs |
| CPU 侧 录制一对 A+B dispatch | **1.1 µs**（bind + push constants + dispatch + barrier ×2） |
| CPU 侧 提交 + 等待整个 command buffer | 与 GPU 时间之差，48 次迭代下 < 5% |

**design §3.4 的 5–20 µs 高了一个数量级。** 按实测重算它那笔账：
每 token ~470 个 dispatch × 0.66 µs = **0.31 ms**，占 65 ms 的 **0.5%**，
而不是 §3.4 担心的 5–14%。

后果：

- "每 token 一个预录制的 command buffer"仍然值得做（它同时解决了 timeline wait 的
  表达问题），但**不再是性能上的必需品**。即使每 token 重新录制，CPU 侧
  470 × 0.55 µs ≈ 0.26 ms 也可以接受。
- §7.9 末尾"把 Dispatch A/B 按 expert 拆成两组，先到的 3 个先算，代价是 dispatch 数翻倍"
  —— **翻倍的代价是 0.66 µs × 2 × 40 层 = 0.05 ms/token，可以忽略**。
  这个优化现在是纯收益，应该在 P3 做。kernel 侧已经支持：indirection list
  （`SlotList` + `list_count`）短一点就行，不用改 kernel。

### 3.5 路径 A vs 路径 B 下的 kernel

同一个最佳变体（M1 L32 R1 sg32 dec0 fp16h），交替测四轮（机器有背景负载，
所以绝对值偏低，看的是同一轮内的相对关系）：

| 轮次 | 路径 | raw-read | kernel A+B | 占比 |
|---|---|---|---|---|
| 1 | A | 208.5 | 156.0 | 75% |
| 2 | B | 179.2 | 125.7 | 70% |
| 3 | A | 195.0 | 148.7 | 76% |
| 4 | B | 183.5 | 187.2 | 102% |

背景负载让绝对值在 125–187 GB/s 之间跳，**两条路径的差别被噪声完全盖住**。
空闲窗口下路径 A 的干净数字是 216.1 GB/s；路径 B 没有拿到同等质量的空闲窗口，
但它的 raw-read（214.2 GB/s，§2.2）说明**上限相同**。

**结论：kernel 对路径 A/B 无感，两条路径在 GPU 侧等价。**

---

## 4. 决定

### 4.1 路径 A vs 路径 B：**用路径 B 做主 expert cache，路径 A 只放 pin 集合**

理由按权重排序：

| 维度 | 路径 A | 路径 B | 胜者 |
|---|---|---|---|
| GPU 读带宽 | 216.6 GB/s | 214.2 GB/s（−1.1%） | 平 |
| 并发时 CPU 还剩多少 | 75.6 GB/s | 49.6 GB/s | **A** |
| CPU 写入（staging 场景） | 18.5 GB/s | 31.6 GB/s（非临时存储） | **B** |
| NVMe 直读落地 | 跑通 | 跑通 | 平 |
| **容量（VGM=64 GB）** | GPU heap 74.4 GiB，但要和 pin 集合、KV、activations 共享 | 系统内存 ~110 GB 中可用的部分 | **B** |
| 分配上限 | `maxMemoryAllocationSize` 2 GiB，两条路都一样 | 同 | 平 |
| 需要的权限 | 无 | 无（大页才需要） | 平 |

带宽既然只差 1.1%，**选择就完全由容量决定**，而容量取决于 VGM 设置：

- **VGM = 64 GB（当前）**：Windows 只看到 63.6 GB 系统内存。路径 B 的 expert cache
  受限于这 63.6 GB 减去 OS、I/O staging、engram 行缓冲；实际能拿到大约 45–50 GB。
  路径 A 的 74.4 GiB device-local heap 减去 17.7 GB pin 集合，剩 ~56 GB。
  **这个设置下路径 A 容量更大。**
- **VGM = 最小值（待重启实测）**：device-local heap 缩到几 GB，系统内存涨到 ~120 GB，
  路径 B 的 cache 可达 ~90–95 GB，**比路径 A 在 VGM=64 下多约 60%**。
  按 §3.1 的表，cache 从 56 GB（≈2,980 个 expert，19% 命中不到）涨到 90 GB
  （≈4,780 个，29%）——**hit rate 每 10 个点价值超过任何 kernel 优化（§3.1 结论 1）**。

所以：**建议把 VGM 调到最小，走路径 B。** 这个建议的最后一块拼图（最小 VGM 下的
heap 大小与两条路径的重测）**需要一次重启，尚未完成**——但由于带宽已经证明两条路等价，
这次重测只需要回答"最小 VGM 下系统内存是不是真的到了 ~120 GB"，
不再需要重跑整个 kernel sweep。

设计上不需要改动：`ExpertStore` 通过 `store::SlabBacking` 只看到
`(host_ptr, device_address)`，`gpu::MemoryAllocator::allocate_slab` 按
`MemoryPath` 分派，`RuntimeConfig::memory_path` 一个字段就能切换。

### 4.2 最佳 kernel 变体

```
decode（M=1）      : LanesPerRow=32, subgroup=32, DecodeMode=0（常量表）, h=fp16, RowsPerLane=1
投机验证（M=6）    : LanesPerRow=16, subgroup=64, DecodeMode=0, h=fp16, RowsPerLane=2
```

`DecodeMode=0` 在两个 M 下都是最优，而且优势很大（1.3–1.6 倍），是唯一一个
"必须选对"的旋钮。其余旋钮的影响都在 ±8% 以内。

### 4.3 T_layer 与 design §9.4 的 `d`

实测（M=1，最佳变体，7 个 expert 全驻留，路径 A）：

```
T_layer(MoE 部分, M=1) = 0.609 ms      （131.6 MB @ 216.1 GB/s）
T_layer(MoE 部分, M=6) = 1.049 ms      （0.175 ms / token）
```

把 §7.14 的整层清单按实测的 216 GB/s 折算：一层常驻 + 命中权重
≈ 210–280 MB → **T_layer(整层) ≈ 0.97–1.30 ms**，与 §3.4 估的 1.0–1.4 ms 吻合
（因为带宽比假设的 200 GB/s 高一点）。

§9.4 的提前量：`d × T_layer ≥ T_io(18.8 MB) × expected_misses / QD_effective`。
用 §9.2.1 实测的 `T_io ≈ 4.0 ms`（一个 expert 单发）：

| 要隐藏的 | 需要的 d（按整层 1.15 ms） | 需要的 d（只算 MoE 0.609 ms） |
|---|---|---|
| 1 个 miss | **3.5 → d ≥ 4** | 6.6 → d ≥ 7 |
| 一层 6 个 miss（并发，§9.2.1 实测 23.7 ms） | 20.6 → **d ≥ 21，不现实** | 39 |

**§9.4 写的 "d ≥ 3–4" 在"每层至多 1 个 miss"的前提下成立，实测确认。**
但 6 个全 miss 的一层需要 21 层提前量，超过了模型的 40 层里可用的窗口——
这从另一个角度重述了 §3.1 结论 1：**提前量救不了低命中率，只有命中率本身能救。**
`d = 4`（`PrefetchConfig::lookahead_depth` 的默认值）应保持不变。

M=6 时 `T_layer` 只涨到 1.049 ms（不是 6 倍），所以**投机解码同时放大了每次 I/O
可以被隐藏的窗口**：同样 4 ms 的 T_io，M=6 下 `d ≥ 4.0/1.049 = 3.8 → d ≥ 4`，
和 M=1 一样，但这 4 层掩护的是 6 个 token 的进度。

---

## 5. 代码层面：store / 指针表契约

**没有需要修改的契约。** design §5.3 / §5.4 定的指针表结构原样可用，具体地：

- `ExpertStore::pointer_table()` 的 `uint64[layer][expert][6]` 布局直接 `memcpy`
  进 GPU 的 storage buffer，kernel 用
  `table[(layer * experts_per_layer + expert) * 6 + part]` 取址。
  kernel 需要 `experts_per_layer` 作为 push constant（表的第二维步长），
  这是新增的一个 push constant 字段，不是契约变更。
- `publish_locked()` 里"有 `dev_addr` 用 `dev_addr`，否则用 host 指针"的写法，
  让同一份表在主机 backing（测试/oracle）和 Vulkan backing（GPU）下都成立，
  实测无需改动。

有三条**被实测钉死、之前只是纸面约定**的约束，写在这里以免将来被改坏：

1. **part 地址只有 8 字节对齐，不是 16。** 扫全量 manifest（15,744 个 expert，
   94,464 个 part）：**87,552 个（92.7%）的槽内偏移 mod 16 = 8**，只有 7.3% 是 16 对齐；
   所有 skew 都是 8 的倍数（`skew mod 8 = 0`，94,464/94,464）。
   kernel 每 lane 每块读 16 B（`uint4`），因此这些读**必然**是 8 对齐而非 16 对齐。
   Slang 的指针解引用发出的是 `OpLoad ... Aligned 4`，SPIR-V 合法，
   RDNA 的 `global_load_dwordx4` 只要求 dword 对齐——**实测零代价**（99% of raw read）。
   这是 §5.1 v0.5"不 repack"决定的全部 kernel 侧代价，账已结清。
2. **slab 必须同时是 host 可写和 device 可寻址的。** `VulkanSlabBacking::allocate()`
   在拿到一个没有 `host_ptr` 的 slab 时直接返回 `Internal` 错误而不是默默接受——
   否则 IoEngine 会往空指针写。纯 `DEVICE_LOCAL` 内存（§2.2 里那条 216.7 GB/s）
   **不能**用作 expert slab，尽管它最快。
3. **每个 slab ≤ `maxMemoryAllocationSize` = 2 GiB。** `SlabConfig::slots_per_slab`
   的上界因此是 `2 GiB / 18,808,832 = 114`；默认的 100（1.88 GB）是安全的。
   `MemoryAllocator::allocate_from_type()` 显式检查并给出指向 §5.3 的错误消息。

另外，为了让 kernel 编译通过，`gpu/vulkan/device.cpp` 现在额外申请：
`shaderInt16`（storage buffer 里的 `half`）、`shaderInt64`（PhysicalStorageBuffer
寻址）、`storageBuffer16BitAccess` / `uniformAndStorageBuffer16BitAccess`、
`synchronization2`（`vkCmdPipelineBarrier2` / `vkCmdWriteTimestamp2`）。
`DeviceCaps::check_required()` 相应加了检查。这些在 design §1.1 的能力表里没列，
但 gfx1151 全部支持。

**一处偏离 architecture.md §1.2**：`gpu/vulkan/memory.cpp` 里有一个
`#if defined(_WIN32)` 块（`VirtualAlloc` / 大页 / `SeLockMemoryPrivilege`）。
架构文档说只有 `storage/windows/` 和 `storage/linux/` 可以有平台分支。
理由是 design §3.3 明确要求路径 B 在 VirtualAlloc 内存和大页上测量，而 gpu/ 在依赖
DAG 上不能引用 storage/windows/；块内有可移植的 aligned-new 回落，Linux 交叉编译通过。
如果要消除这个偏离，正确的做法是把"页分配器"下沉到 `core/`。

---

## 6. 未解决的问题

1. **最小 VGM 下的重测**（§4.1）。需要进 BIOS 改 UMA Frame Buffer Size 并重启。
   只需要回答"系统内存是否到 ~120 GB"，带宽部分已经不需要重跑。
2. **大页下的路径 B**（§2.6）。需要 secpol.msc 授予"锁定内存页"并重新登录。
   预期能抹平路径 B 那 1.1% 的差距，并改善它在并发下对 CPU 的挤压。
3. **两个 dispatch 分别特化 RowsPerLane**（§3.2 第 4 点）。预期 dispatch B +7%，
   整体 +2%。一行改动，留给 P2。
4. **M=6 的 x 分块进 LDS**（§3.3）。现在 M=6 的有效带宽只有上限的 58%，
   瓶颈是 x 的 L0 请求量。这是投机解码收益最大的一块，应该在 P4 之前做。
5. **shared expert 的 fp8 模板还没写**。本文的 7 个 expert 里第 7 个是 FP4 顶替的，
   真正的 shared expert 是 fp8 E4M3 + 32×32 块 scale，是 §7.9 的另一个模板实例
   （23.6 MB 而不是 12.5 MB）。它的有效带宽需要单独测。
6. **raw-read shader 不是真正的上限**（§3.2 末）。MoE kernel 曾超过它。
   如果要一个可信的"机器上限"，应该写一个多流版本的 rawread（同时读 2–3 个独立区域），
   或者接受"216–218 GB/s 就是 LPDDR 的实际上限"这个结论。
7. **测量环境的隔离**（§1）。同机的 CPU 负载会让 GPU 数字浮动 25–50%。
   P2/P3 的回归测量需要一个"安静窗口"检查，否则 CI 上的 kernel 带宽数字没有意义。

---

## 附：如何复现

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'

# 正确性（十四个变体 × 两个真实 expert，对照 oracle）
ctest --test-dir build -R suite.gpu_moe --output-on-failure

# 带宽矩阵（design §9.2 / §3.3）
.\build\bw_matrix.exe --cpu-size-gb 4 --size-gb 1 --repeats 3

# kernel sweep（design §7.9 / §7.1）
.\build\kernel_bench.exe --csv bench\results\kernel_p1.csv `
    --iters 48 --layer-cycle 8 --repeats 4 --sweeps 2
.\build\kernel_bench.exe --path b --quick        # 路径 B 对照
```

带 validation layer 跑一遍（两个 bench 与 gpu_moe 测试当前都是干净的）：

```powershell
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
$env:VK_LOADER_LAYERS_ENABLE='VK_LAYER_KHRONOS_validation'
```

**测量时不要同时跑编译或任何吃内存带宽的进程**（§1）。
