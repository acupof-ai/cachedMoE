# P2：MoE FFN kernel 为投机解码做好准备

本文是 design.md §7.9 / §7.9.1 在 **投机验证批（M = k+1 ≤ 6）**、**fp8 shared expert**、
**`h` 的 fp8 量化**、**按到达顺序分组 dispatch** 四件事上的实测答案，接在
[kernel_p1.md](kernel_p1.md) 之后。原始 CSV 在 `bench/results/kernel_p2_moe.csv`
（261 行，一次完整 sweep）。

状态：v0.1（2026-09-14）。测量机器与 P1 相同：Ryzen AI Max+ 395 / Radeon 8060S
(gfx1151) / 128 GB LPDDR5X / BIOS VGM = 64 GB，路径 A，Vulkan 1.4，
`--iters 48 --layer-cycle 8 --repeats 3 --sweeps 2`。

---

## 0. 先看结论

| 问题 | P1 的状态 / design 的假设 | P2 实测 | 结论 |
|---|---|---|---|
| M=6 有效带宽 | 135.1 GB/s = 上限的 63%，§15 P2 要求 ≥ 80% | **135.3 GB/s = 62%**（同一节里 P1 kernel 119.0，全轮最好的 P1 读数 128.8） | **目标未达成**；同节相对提升 **+13.7%**，保守口径 +5%，`ms/token` 0.184 → **0.162** |
| §7.1 rule 6「x 分块进 LDS」 | design 写明这是 P2 的头号 kernel 待办 | **净负收益**：M=6 A+B 从 119.0 掉到 95.5 GB/s | **规则作废**（§3.2），barrier 破坏访存流水的代价大于省下的 L0 请求 |
| M=6 的真实瓶颈 | P1 说是「激活的读取指令」 | **是 VALU 发射，不是访存**（§3.1 的拆解） | 有效的修法是减少每列的指令数，不是减少访存 |
| packed fp16（`v_pk_fma_f16`） | §7.9 只提了 int8 dot4 | **dispatch A 最优**：M=6 A 117.2 → 133.9 GB/s（+14%） | 采纳，是 M ≥ 4 的默认 |
| int8 dot4 | §7.9 的「M ≥ 4 备选」 | **dispatch B 最优**：M=6 B 100.0 → 125.7 GB/s（+26%）；A 上无收益 | 采纳，但只用在 B |
| A / B 分别特化 | §7.9.1 未解决问题 3 | **必要**：M=6 最优是 A packed fp16 + B int8 dot4 | 已实现（`MoeSpec::x_mode_b`） |
| int8 路径的精度 | §12 判据 ≤ 5e-3 | **5.4e-3**（M=1）/ 7.4e-3（M=6 逐列） | **略微超标**，见 §3.4 |
| fp8 shared expert | §7.9「另一个模板实例，带宽需单独测」 | **M=1 216.9 GB/s = 上限的 100%**，cos 0.999999947 | 与 FP4 同样打满内存，模板成立 |
| `h` 的 fp8 量化（§7.9 v0.6） | 「不是可选的数值细节」 | 与参考实现**数值上吻合到 5.0e-8**；代价 **M=1 +27% 时间，M=6 +6%** | 正确性零疑问，**decode 的代价比预期大** |
| 分组 dispatch 的代价 | §3.4/§7.9：「0.66 µs × 2 × 40 层 ≈ 0.05 ms/token，可忽略」 | **M=1 每层 +0.193 ms，M=6 +0.108 ms** | **§7.9 的这句话必须改**，见 §6 |

三个**出乎意料**的结果：

1. **LDS 是错的方向。** design §7.1 rule 6、kernel_p1.md §3.3、design §15 的 P2 优先级表
   都把「x 分块进 LDS」列为 M=6 的头号修法。实测它在**每一个 M 上都更慢**，M=6 时
   dispatch B 掉 27%。原因不是 LDS 慢，是每个 K-chunk 两个 workgroup barrier 把
   访存流水排空了（§3.2）。
2. **M=6 不是访存受限，是 VALU 发射受限。** 把 M 从 1 扫到 6，dispatch A 的时间是
   `0.33 + 0.055·M` ms 的直线；那个斜率对得上 fp32 FMA + `f16tof32` 的指令数
   （§3.1）。这解释了为什么省访存的设计全都没用，而省指令的设计（packed fp16、
   int8 dot4）有用。
3. **把 7 个 expert 拆成两次 dispatch 不是免费的。** §3.4 量到「每 dispatch + barrier
   0.56 µs」，于是 §7.9 下结论说分组计算是纯收益。实测拆 3+4 在 M=1 上每层多花
   **0.193 ms**——**是那个 0.56 µs 的 340 倍**，因为多出来的不是启动开销，而是
   dispatch B 的 640 个 workgroup 又跑了一遍归约和写出（§6）。

---

## 1. 这一轮改了什么

四个新的 specialisation 常量（`gpu/shaders/moe_common.slang`，id 接在 P1 的 0–6 之后）：

| id | 常量 | 取值 |
|---|---|---|
| 7 | `XMode` | 激活怎么到 FMA：0 全局（P1 kernel）、1 LDS K 分块、2 LDS + packed fp16、3 LDS + int8 dot4、**4 全局 + packed fp16**、**5 全局 + int8 dot4** |
| 8 | `HQuant` | `h` 的 fp8 量化：0 不做（v0.5）、1 在 dispatch B 内复现、2 dispatch A 写 fp8 + UE8M0 scale |
| 9 | `Fp8Slots` | 编译 FP8 E4M3 权重路径（shared expert） |
| — | `MoeSpec::x_mode_b` | dispatch B 的 `XMode`，默认跟随 A（§7.9.1 未解决问题 3） |

`RowsPerLane` 的上界从 4 放宽到 16（幂次），因为 §3.1 一开始怀疑的是激活流量，而
`RowsPerLane` 是唯一能整除它的旋钮。结论见 §3.3：它救不了 A，但**能救 B**。

---

## 2. 测量方法

与 kernel_p1.md §1 相同，外加两条：

- **fp8 shared expert 是真的。** `bench/kernel_bench` 启动时用 IoEngine 把
  `layers.L.ffn.shared_experts.{w1,w2,w3}.weight` 与它们的 scale 平面直读进一块
  带 device address 的 GPU 可见缓冲（8 层共 283.4 MB），六个 part 地址塞进指针表的
  一个备用 expert 下标。`Ids[slot]` 的 bit 31（`gpu::kSlotFp8`）告诉 kernel 这个槽
  是 fp8。**因此 A+B 一对 dispatch 的字节数从全 FP4 的 131,604,480 B 变成
  148,227,840 B**，有效 GB/s 的分子按槽的格式分别算。
- **run 与 run 之间有 ~7% 的漂移。** 同一个变体在四轮完整 sweep 里落在
  134.0–144.7 GB/s（M=6 最优变体），而 raw-read 上限四轮都是 216.5–217.9 GB/s；
  **同一轮内部、不同小节之间也有 8%**（§3.4 末）。
  所以**本文所有「提升了多少」都取自同一轮内部的对照**，绝对值给区间。
  测量期间 CPU 负载 1–11%，没有并行编译。

---

## 3. M 扫描：投机验证批

### 3.1 瓶颈在哪：先把 P1 的诊断推翻

P1（kernel_p1.md §3.3）说 M=6 的瓶颈是「激活的读取指令：每个 32 元素块一个 lane 要发
6 条 `uint4` 读拿 96 B 的 x，却只读 32 B 的权重，L0 请求量是权重的 12 倍」。
把 M 从 1 扫到 6（`xglob`、L32 R1），dispatch A 的时间是：

| M | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| A ms | 0.391 | 0.452 | 0.454 | 0.557 | 0.581 | 0.680 |

拟合成 `t_A(M) ≈ 0.33 + 0.055·M` ms（`bench/results/kernel_p2_moe.csv` 的
`section = M x RowsPerLane`、L32 R1 行）。**0.33 ms 那一项是 87.7 MB 的访存
（≈ 266 GB/s，比 A+B 合起来的上限高，因为 A 单独测时 `h` 还没被写脏），
0.055·M 是每多一列的固定代价。** 把这个斜率折成指令数：

- 每列每次迭代的 fp32 FMA：`7 × 2304 × 5120 × 2 = 165 M`
- 每列的 `f16tof32`：`7 × 2304 × 5120 = 82.6 M`
- 机器的 VALU 发射能力：40 CU × 2 SIMD32 × 32 lane × 2.9 GHz ≈ **7.4 T op/s**

`(165 + 82.6) M / 7.4 T = 0.033 ms`，对 0.055 ms 的实测斜率是 **61% 的 VALU 占用**
（剩下的是 FP4 解码、块 `ldexp` 与发射空档）。
也就是说 **M 增加的那部分完全是算术与格式转换的指令发射，不是访存**。
P1 的「L0 请求量是 12 倍」是对的，但那不是限制项——L0 的带宽和发射端口都还有余量。

这一条决定了后面所有设计的成败：**省访存的没用，省指令的有用。**

### 3.2 设计 1、2：x 分块进 LDS（`xlds` / `xldsf16` / `xldsi8`）——净负收益

design §7.1 rule 6 要求的形式：整个 workgroup 协作把 `LanesPerRow` 个 K 分块的 x
（M 列全部）搬进 LDS，然后所有 row group 从 LDS 读。M=6 / L=16 时是 6 KiB，放得下。

L16 R2、M=6，同一轮：

| XMode | A GB/s | B GB/s | A+B GB/s | ms/token |
|---|---|---|---|---|
| 0 `xglob`（P1 kernel） | 117.2 | 100.0 | 119.0 | 0.1844 |
| 1 `xlds` | 114.3 | **72.5** | 95.5 | 0.2297 |
| 2 `xldsf16`（LDS + packed fp16） | 91.0 | 60.4 | 76.3 | 0.2873 |
| 3 `xldsi8`（LDS + int8 dot4） | 116.6 | **125.7** | 126.4 | 0.1736 |

L32 R1 更极端：`xglob` 110.4 → `xlds` 42.6 GB/s。

**为什么会更慢**：LDS 分块并不减少指令数——`ds_read_b128` 和 `global_load_dwordx4`
都是一条指令读 16 B，只是换了一条流水。它换来的是每个 K-chunk **两个
`GroupMemoryBarrierWithGroupSync()`**。在这个 kernel 里权重读是 DRAM 读，barrier
之后全 workgroup 的权重读必须重新发起，**内存延迟从「被下一块的计算掩盖」变成
「每 chunk 暴露一次」**。dispatch A 每个 workgroup 有 160/16 = 10 个 chunk，
dispatch B 有 7 个 slot × ⌈72/16⌉ = 35 个。B 的 chunk 更多、而且 72 不是 16 的倍数
（最后一块只有一半 lane 有活干），所以 B 掉得更狠。

**`xldsi8` 是唯一的例外**，而且只在 dispatch B 上：int8 把每 (列, 块) 的激活读从
4 条 `uint4` 变成 2 条 `ds_read_b128`，把 32 条 FMA + 32 条转换变成 8 条
`dot4add_i8packed`，省下的指令数足够盖过 barrier 的代价（§3.4 讲精度代价）。

> **对 design 的直接后果**：§7.1 rule 6 的「M=6 时 x 要按 K 分块进 LDS」和
> §15 P2 优先级表第 2 项里的同一句话，**实测不成立，应当删掉**。取而代之的是
> §3.5 的那一句：M=6 要省的是指令，不是字节。

### 3.3 设计 3：`RowsPerLane` —— 救不了 A，但能救 B

一个 workgroup 每个 row group 都要把整条 x 读一遍，所以跨 L2 的 x 字节是
`2304 × slots × M × 10 KiB / RowsPerLane`，**只有 `RowsPerLane` 能整除它**。
M=6、`xglob`：

| R | L16 A / B | L32 A / B | L64 A / B |
|---|---|---|---|
| 1 | 134.4 / 88.1 | 129.0 / 78.0 | 109.1 / 67.8 |
| 2 | 130.6 / **107.8** | 126.0 / 90.8 | 100.9 / 84.0 |
| 4 | **87.6** / 113.8 | 79.7 / 110.9 | 64.7 / 97.5 |
| 8 | 38.4 / 107.4 | 39.3 / 93.3 | 34.3 / 73.1 |
| 16 | 11.0 / 51.3 | 11.4 / 63.3 | 11.1 / 52.5 |

**A 在 R ≥ 4 上塌方**（寄存器溢出：R × M 个 fp32 累加器 × 2 个矩阵 + R × 8 个解码值），
**B 在 R = 4 上还在涨**（B 只有一个矩阵，寄存器压力是 A 的一半）。这是
§7.9.1 未解决问题 3「A 和 B 的最优 `RowsPerLane` 未必相同」的实测确认，
只不过差别比预期大得多。本实现因此让 A 和 B 用同一个 `RowsPerLane`
（改起来要再加一个 spec 常量，收益 < 2%），而把 A/B 的分别特化用在 `XMode` 上。

### 3.4 设计 4、5：不进 LDS 的 packed fp16 与 int8 dot4 —— 这两个有用

既然 §3.2 说 barrier 是代价、§3.1 说指令是瓶颈，正确的做法是**不碰 LDS，只换算术**：

- **`XMode = 4`（`xgf16`）**：x 仍然从全局按 `uint4` 读，但**不做 `f16tof32`**，
  直接 `bit_cast<float16_t2>` 成 half2，权重也解码成 half2（16 项 `float16_t` 常量表，
  E2M1 的十六个值在 fp16 里都是精确的），用 `v_pk_fma_f16` 累加。
  **half2 累加器活满整个 32 元素块**——按 8 个元素归约一次的话，每
  (行, 列, 组) 要 2 次 `f16→f32` + 2 次加法，这部分本身就和它省下的 FMA 一样大
  （第一版这么写，M=6 只快了 1%）。
- **`XMode = 5`（`xgi8`）**：lane 自己把它那 32 个 x 量化成 int8（amax → scale →
  round），权重解码成 packed int8（`2 × E2M1` 是精确的 int8，那个 2 折进块指数），
  用 `dot4add_i8packed`。

L16 R2、M=6，同一轮：

| XMode | A GB/s | B GB/s | A+B GB/s | ms/token |
|---|---|---|---|---|
| 0 `xglob` | 117.2 | 100.0 | 119.0 | 0.1844 |
| 4 `xgf16` | **133.9** | 108.8 | 132.0 | 0.1662 |
| 5 `xgi8` | 77.2 | 53.3 | 67.6 | 0.3245 |
| 0 / 3 `xglob/ldsi8` | 120.0 | 120.4 | 128.5 | 0.1708 |
| 4 / 3 `xgf16/ldsi8`（A packed fp16、B int8 dot4） | 132.0 | **121.8** | **135.3** | **0.1621** |

同一节里 `xglob` 是 119.0，而同一个特化在同一轮的 `M x RowsPerLane` 一节里读到
128.8 GB/s——**同轮不同节之间有 8% 的漂移**（后跑的节更热）。
所以上表的 +13.7% 是节内对照，**保守口径（拿全轮最好的 `xglob` 读数 128.8 比）
是 +5%**。两种口径都记在这里，不要只引一个。

`xgi8` 在 A 上很差是可以预期的：lane 自己量化要 32 次 abs/max + 32 次乘加取整，
**在 R = 2 时这笔开销和它省下的 FMA 差不多**；只有当同一份量化被更多权重行复用
（R ≥ 4，而 A 在 R ≥ 4 上塌方）或者 x 在 kernel 之外就已经是 int8 时才划算。
`xldsi8` 在 B 上好，正是因为 LDS 分块让**一次量化被 16 个 row group 共用**。

### 3.5 M ∈ {1..6} × 变体总表

每个 M 的最优变体（`bench/results/kernel_p2_moe.csv`，`section = M sweep`，
7 个 FP4 槽，上限 216.6–217.9 GB/s）：

| M | P1 kernel `L16 R2 xglob`（同轮） | P2 最优变体 | A GB/s | B GB/s | **A+B GB/s** | %上限 | ms/dispatch 对 | **ms/token** |
|---|---|---|---|---|---|---|---|---|
| 1 | 179.3 | `L32 R1 xglob`（P1 的冠军不变） | 224.3 | 217.4 | **222.6** | **102%** | 0.591 | **0.5912** |
| 2 | 186.9 | `L32 R1 xgf16` | 215.3 | 184.7 | **206.4** | 95% | 0.637 | **0.3187** |
| 3 | 171.4 | `L16 R1 xglob` | 193.4 | 129.6 | **178.2** | 82% | 0.738 | **0.2461** |
| 4 | 153.0 | `L16 R2 xgf16` | 157.5 | 148.0 | **162.0** | 74% | 0.813 | **0.2031** |
| 5 | 138.5 | `L16 R2 xgf16` | 144.2 | 130.5 | **149.7** | 69% | 0.879 | **0.1758** |
| 6 | 128.8 | `L16 R2 xgf16/ldsi8` | 132.0 | 121.8 | **135.3** | 62% | 0.972 | **0.1621** |

（对照列取的是全轮里 `L16 R2 xglob` 最好的读数，即对 P2 最不利的口径。M=1 和 M=3
的赢家仍然是 P1 的形状——M ≤ 3 时 packed fp16 与 fp32 的差别在漂移范围内，
因为那里 §3.1 的 M-正比项还不够大。）

**收益在 M ≥ 2 上稳定在 5–10%**（保守口径：M=2 +10%、M=4 +6%、M=5 +8%、M=6 +5%；
§3.4 的节内口径下 M=6 是 +14%），因为被砍掉的正是 §3.1 里那条与 M 成正比的指令流。
M=1 上一个字节也没省——那里 kernel 已经在内存系统的上限上，没有东西可省。

**`ms/token` 才是投机解码真正买到的东西**：0.5912（M=1）→ 0.1621（M=6），
**快 3.65 倍**（P1 是 3.7 倍，但那是 0.602 / 0.162，分子分母各来自不同的最优形状）。

### 3.6 M=6 的目标为什么没达成，以及怎么才能达成

§15 P2 的准出条件是「M=6 有效带宽 ≥ 上限的 80%」（本任务加码到 85% / 185 GB/s）。
实测 62%。**这个指标本身在 M=6 上已经不合适**：kernel 在那里不是带宽受限的
（§3.1），拿「读了多少字节 ÷ 时间」去衡量一个 VALU 受限的 kernel，分子是常数、
分母被算术撑大，数字必然难看。**应该换成 `ms/token`**。

如果仍然要 185 GB/s（= A+B 0.71 ms），按 §3.1 的模型需要把每 (列, 块, lane) 的
指令数压到 FMA 数的 2 倍以内。已知唯一能做到的设计是
**把 x 在 kernel 之外量化成 int8**（每 token 一次，40 层复用）：

- 消费侧每 (列, 块) 变成 2 条 `uint4` 读 + 8 条 `dot4add_i8packed` × `RowsPerLane`，
  按 §3.1 的模型 A 可以到 ~187 GB/s；
- 代价是一个新的 buffer（`x_i8[M][5120]` + `[M][160]` 的 scale）、一个极小的
  前置 dispatch（或由上一个 kernel 的写出阶段融合），
- **以及精度**：见下。

**int8 路径的精度（design §12 判据 ≤ 5e-3）**：

| 变体 | cos | max\|Δy\| 占 \|y\|max |
|---|---|---|
| fp32 基准（`xglob`） | 0.999999961 | 1.37e-4 |
| packed fp16（`xgf16`） | 0.999999569 | **4.52e-4** |
| packed fp16（`xldsf16`，每 8 元素归约一次） | 0.999999846 | 3.04e-4 |
| int8 dot4（`xldsi8` / `xgi8`，A 与 B 都量化） | 0.999931569 | **5.38e-3** |
| int8 dot4，M=6 逐列对 CPU 参考 | — | 7.36e-3 |

**int8 比 §12 的 5e-3 高 8%**，来源是激活的 8 位量化本身：x 在块内近似高斯，
amax ≈ 2.5σ，量化步长 2.5σ/127，每个元素的相对误差 ~0.5%，两次 GEMV 叠加
到输出约 1%（cos 1 − 6.8e-5）。这不是实现问题，是 int8 激活的定价。
**建议**：`xldsi8` 只用在 dispatch B（上表的 5.4e-3 是 A 和 B 都用 int8 的数字；
只有 B 用时误差约减半），并且把 §12 的 int8 判据按 L2 的实测重新定，
而不是让 kernel 去迁就一个拍脑袋的 5e-3。
`xgf16` 的 4.52e-4 远在 1e-3 判据内，**可以无条件采用**。

---

## 4. fp8 shared expert（design §7.9 的另一个模板实例）

P1 的第七个槽是拿 FP4 routed expert 顶替的。现在是真的：

- `layers.L.ffn.shared_experts.{w1,w2,w3}.weight`：**FP8 E4M3**，
  `[2304, 5120]` / `[5120, 2304]`，每个矩阵 **11,796,480 B**；
- scale：**UE8M0**，`[72, 160]` / `[160, 72]`，`block = [32, 32]`，每个 **11,520 B**
  ——注意这是 **32 × 32 的 tile scale**，不是 FP4 的「每行每 32 个 K 元素一个」，
  kernel 里索引是 `scale[(row >> 5) * (K/32) + blk]`；
- 一层的 shared expert 共 **35,424,000 B**，是一个 routed expert（18,800,640 B）的
  1.88 倍，与 §7.14 清单的「6 × 12.5 + 23.6 MB / 6 × 6.3 + 11.8 MB」吻合。

**同一对 dispatch 同时处理两种格式**：`Ids[slot]` 的 bit 31 置位表示该槽是 fp8，
kernel 据此切换权重字节步长（32 B/块 对 16 B/块）、scale 布局和解码表。
`Fp8Slots` 这个 spec 常量为 0 时整条 fp8 路径被折叠掉，所以 **routed-only 的
pipeline 一个指令也没多**。解码用 §7.1 rule 5 要求的 **256 项 LDS 表**
（1 KiB，kernel 入口由 256 个线程各算一项填好）。

### 4.1 正确性

`tools/oracle_shared.py --shared 0` 写出 `tests/data/l1_shared_layer0.bin`
（torch fp32，权重经 manifest 的 run/skew 读出后按 tile scale 解码），
`tests/test_gpu_moe.cpp` 的 `the_fp8_shared_expert_runs_in_the_same_dispatches`
用真正的 IoEngine 把六个 tensor 直读进 GPU 可见缓冲再跑 kernel：

| 变体 | 对照 | cos | 占 \|y\|max |
|---|---|---|---|
| M=1，全局 x | `y_ref` | 0.999999947 | 1.367e-4 |
| M=1，16 lane / 2 行 | `y_ref` | 0.999999947 | 1.364e-4 |
| M=6，LDS 分块（fp8 槽仍走全局 x） | `y_ref` | 0.999999947 | 1.364e-4 |
| M=1，`h` 在 dispatch B 内 fp8 量化 | `y_hq16`（§5.2） | 1.000000000 | **2.966e-7** |

前三行与 FP4 routed expert 完全同一个量级（1.37e-4），误差仍然由 x 的 fp16 舍入主导。
最后一行说明 fp8 权重路径和 fp8 `h` 量化叠在一起时**没有额外误差**。

**顺带确认了一条 §5.1 的约束在 shared expert 上也成立**：六个 part 的设备地址
`mod 16 = 8`，和 routed expert 一样只有 8 字节对齐，kernel 的 `uint4` 读照样合法。

### 4.2 带宽

7 个槽 = 6 routed + 1 真 shared，一对 dispatch 读 **148,227,840 B**：

| 变体 | A GB/s | B GB/s | A+B GB/s | %上限 | ms/token |
|---|---|---|---|---|---|
| M=1 `L32 R1 xglob fp8` | 216.5 | 210.8 | **216.9** | **100%** | 0.683 |
| M=1 `L16 R2 xgf16 fp8` | 184.4 | 211.9 | 193.8 | 89% | 0.765 |
| M=6 `L32 R1 xglob fp8` | 108.4 | 82.8 | 101.7 | 47% | 0.243 |
| M=6 `L16 R2 xgf16 fp8` | 109.4 | 117.3 | 113.2 | 52% | 0.218 |

**fp8 模板在 M=1 上打满内存系统（100% 上限）**，与 FP4 完全一样。
它每字节的 ALU 比 FP4 少（一个字节一个元素，不用拆 nibble），所以 M=6 时
它反而是 7 个槽里压力最小的那个；M=6 整体掉到 52% 是 6 个 FP4 槽拖的。

**每层 MoE 的真实时间**（7 槽含真 shared expert）：**M=1 0.683 ms，M=6 1.310 ms
= 0.218 ms/token**。这两个数比 §7.9.1 的 0.602 / 0.974 ms 大，
因为 §7.9.1 的第七个槽只有 12.5 MB 而不是 23.6 MB。**§9.4 的 `T_layer` 和
§10.3 的 `T_hot(M)` 应该用这里的数字。**

---

## 5. `h` 在进 `w2` 之前的 fp8 量化（design §7.9 v0.6）

参考实现 `Expert.forward` 的最后一步是 `self.w2(x.to(dtype))`，而 `linear()`
对 fp4/fp8 权重的第一步是 `act_quant(x, 32, "ue8m0", e8m0)`：
`silu(gate) * up` 要按行每 32 个元素量化成 fp8 E4M3，块 scale 是
`fast_round_scale`（向上取到 2 的幂的 UE8M0）。

### 5.1 两种实现，数值上等价

| `HQuant` | 做法 | 约束 |
|---|---|---|
| 1 `hqB` | dispatch B 从它本来就要读的 fp16 `h` 里现算：一个 lane 正好持有一个块的 32 个元素，**amax 不需要任何额外访存** | 无 |
| 2 `hq8` | dispatch A 写出 fp8 `h` + UE8M0 scale 平面，dispatch B 直接消费 | **一个 workgroup 必须拥有 32 的整数倍行**（`(256/L) × R % 32 == 0`），否则一个 fp8 块会跨 workgroup |

`hq8` 的存储布局（`h` 那一块 allocation 内部）：前 `M × slots × 2304` 字节是
fp8 值（4 个一个 word），紧接着 `M × slots × 72` 个 uint 是每块的 scale
（存的是 `asuint(float)` 形式的 2 的幂，直接乘）。没有新的 descriptor——
dispatch A 多绑一个 binding 7（同一块内存的 `RWStructuredBuffer<uint>` 别名），
dispatch B 本来读的就是 raw word。

### 5.2 正确性：与参考吻合到 5.0e-8，并且**离 v0.5 的 golden 更远了**

`tools/oracle.py --level l1` 的 `expert_ffn` 明写着「in fp32 with no activation
quantisation」，所以 `tests/data/l1_*.bin` 是**没有量化**的答案。
一个正确实现 §7.9 v0.6 的 kernel **必须离它更远**。
`tools/oracle_shared.py` 对同一个 x 写出四个答案：

| 向量 | 定义 |
|---|---|
| `y_ref` | 全 fp32（= `l1_*.bin` 里的 y，两者数值一致：‖y‖₂ = 168.288 / 180.752） |
| `y_hq` | `h` 量化，x 仍 fp32 |
| `y_full` | x 和 `h` 都量化（= 参考实现） |
| `y_hq16` | x 和 `h` 先舍入到 fp16（design §6）再量化 `h` —— **kernel 真正在算的东西** |

expert (0, 0) 上的实测（`tests/test_gpu_moe.cpp::the_fp8_h_quantisation_matches_the_reference`）：

| kernel 变体 | 对 `y_ref` | 对 `y_hq` | 对 `y_hq16` |
|---|---|---|---|
| 不量化（v0.5） | **1.371e-4** | 1.359e-2 | 1.232e-2 |
| `hqB`（B 内做，L32 R1） | 1.229e-2 | 2.776e-3 | **3.018e-7** |
| `hqB`（B 内做，L16 R2） | 1.229e-2 | 2.776e-3 | **3.081e-7** |
| `hq8`（A 写 fp8 h） | 1.229e-2 | 2.776e-3 | **5.030e-8** |
| `hq8` + LDS x 分块 | 1.229e-2 | 2.776e-3 | 5.030e-8 |
| M=6，`hq8` | 1.229e-2 | 2.776e-3 | 5.030e-8 |

**两个实现彼此一致到 1e-4 以内，对 `y_hq16` 吻合到 3e-7 / 5e-8**——
也就是说 kernel 里的 fp8 E4M3 舍入（round-to-nearest-even，正规数走位运算、
次正规数走 `(t + 2^23) − 2^23` 的魔数加法）和 `fast_round_scale`
与 torch 的 `.to(torch.float8_e4m3fn)` **逐位一致**。

**为什么对 `y_hq` 还差 2.8e-3**：`fast_round_scale` 是块 amax 的**阶跃函数**。
kernel 的 `h` 是 fp16（design §6），参考的是 fp32，两者相差 2e-4；
只要某个块的 amax 落在 2 的幂两侧，整块 32 个值就落在**粗一倍的网格**上。
Python 侧同一个实验（`y_hq16` vs `y_hq`）给出 2.763e-3，**与 GPU 的 2.776e-3 吻合**
——差异来自量化器的敏感性，不是 kernel 的误差。这条对 L2 逐层比对很重要：
**`h` 量化之后，逐层的容差不能再按 1e-3 定**，量化器会把上游的 2e-4 放大 14 倍。

量化本身对输出的影响（`tools/oracle_shared.py`，三个 expert）：

| expert | `y_hq` vs `y_ref` | `y_full` vs `y_ref` | `y_full` vs `y_hq`（fp8 x 的代价） |
|---|---|---|---|
| shared (layer 0) | 1.002e-2 | 1.559e-2 | 1.845e-2 |
| routed (0, 0) | 1.355e-2 | 1.680e-2 | 2.188e-2 |
| routed (39, 383) | 3.314e-2 | 4.642e-2 | 5.537e-2 |

**`h` 的 fp8 量化单独就把输出改动 1–3.3%。** design §7.9 v0.6 说「省掉会改变输出」，
实测证实，而且幅度比 §12 的任何判据都大两个数量级。

### 5.3 代价：M=1 上比预期贵

| 变体 | A+B GB/s | ms/dispatch 对 | ms/token | 对同 M 最优的代价 |
|---|---|---|---|---|
| M=1 `L32 R1 xglob`（不量化，M=1 冠军） | 222.6 | 0.591 | 0.591 | — |
| M=1 `L16 R2 xglob hq8` | 175.6 | 0.749 | 0.749 | **+27%** |
| M=1 `L32 R4 xglob hq8` | 172.7 | 0.762 | 0.762 | +29% |
| M=1 `L32 R1 xglob hqB` | 149.7 | 0.879 | 0.879 | +49% |
| M=1 `L16 R2 xglob hqB` | 160.3 | 0.821 | 0.821 | +39% |
| M=6 `L16 R2 xgf16/ldsi8`（不量化，M=6 冠军） | 135.3 | 0.972 | 0.1621 | — |
| M=6 `L16 R2 xgf16/ldsi8 hq8` | 127.4 | 1.033 | **0.1721** | **+6%** |
| M=6 `L16 R2 xgf16 hqB` | 65.6 | 2.006 | 0.3345 | +106% |

两条结论：

1. **`hq8` 明显优于 `hqB`**，而且 M 越大差得越多（M=6 时 `hqB` 的 dispatch B
   掉到 32 GB/s）。原因是 `hqB` 为了拿块 amax 必须把 `h` **读两遍**
   （一遍求 amax，一遍逐组用），并且每个 lane、每个权重行都重算一次同一个 scale；
   `hq8` 只在 dispatch A 里每块算一次，而且把 `h` 的流量减半（fp8 + scale
   = 1.125 B/元素，对 fp16 的 2 B）。
2. **`hq8` 的 32 行约束把 M=1 从 L32 R1 赶到 L16 R2，这就是那 27% 的大头。**
   M=1 的 `L16 R2 xglob`（不量化）本身就只有 179.3 GB/s，而 `hq8` 在它之上
   几乎不要钱（175.6）。**所以代价不是量化，是被迫换 workgroup 形状。**

   **建议**：要拿回这 27%，正确的做法是给 dispatch A 的 `h` 写出阶段解耦
   workgroup 形状——例如让 A 的 workgroup 覆盖 32 行的写出而保持 L32 R1 的
   计算形状（需要一次 LDS 转置），或者把 `h` 的量化拆成第三个极小的 dispatch
   （2304 × 7 个元素，一次全局 barrier）。**这是 P2 之后最值得做的一件事**，
   量级是 40 层 × 0.158 ms = **6.3 ms/token**。

---

## 6. 分组 dispatch（design §7.9「先到的先算」）

kernel 侧本来就支持（`SlotList` + `list_count`）。P2 补了两件事：

- `DownPush.flags` 的 bit 0 = **累加**：dispatch B 把它这一组 slot 的和**加进** `y`
  而不是覆盖，于是一层可以拆成任意多组。`MoeRunner::set_accumulate()`。
- `tests/test_gpu_moe.cpp::a_partial_dispatch_reduces_to_the_same_y` 把 7 个槽
  拆成 3+4、1+6、6+1 三种切法，对照一次算完的 `y`。

### 6.1 正确性：不是逐位相同，但差 1 ULP 量级

| 变体 | 切法 | 逐位相同的字 | max\|Δy\| 占 \|y\|max |
|---|---|---|---|
| M=1 `L32 R1 xglob` | 3 + 4 | 1349 / 5120 | 5.95e-8 |
| M=6 `L16 R2 xgf16` | 3 + 4 | 26964 / 30720 | 7.24e-8 |
| M=6 `L16 R2 xgf16 hq8` | 4 + 3 | 13952 / 30720 | 4.49e-8 |
| M=1 `L32 R1 xglob` | 1 + 6 | 1241 / 5120 | 7.93e-8 |
| M=1 `L32 R1 xglob` | 6 + 1 | 1559 / 5120 | 3.96e-8 |

**不是逐位相同**，因为归约被重新结合了：一次算完时 lane 的 fp32 累加器按
`slot → block` 的顺序加，拆开之后每组各自做一次跨 lane 的 LDS 树归约、
最后两个树的结果相加。差异 **≤ 8e-8 的 \|y\|max（相对 2.4e-7，1–2 ULP）**，
和 kernel_p1.md §3.1 里 M=6 各列之间的差异同一量级。
**判据因此定为 1e-6 而不是逐位相等**，和 §3.1 的处理一致。

### 6.2 代价：§7.9 的「可忽略」是错的

同一轮、同样 7 个 expert、同样的字节：

| M | 一次 7 槽 | 拆 3 + 4 | 差 |
|---|---|---|---|
| 1 | 0.596 ms | 0.343 + 0.446 = 0.789 ms | **+0.193 ms（+32%）** |
| 6 | 1.237 ms | 0.584 + 0.762 = 1.345 ms | **+0.108 ms（+9%）** |

拆开之后两段的时间相加**不等于**一次算完的时间：3 槽那一段占了 7 槽的 58%
（对 43% 的权重字节），4 槽那一段占 75%（对 57%）。**多出来的是一个与
`list_count` 无关的按次固定项**——它不是启动开销（§3.4 实测 0.56 µs，
比这里小两个数量级），而是 **dispatch B 的 640 个 workgroup（5120 行 / 8）
每多一次 dispatch 就要重跑一遍跨 lane 的 LDS 树归约、重读并重写 `y`、
重新 ramp 起 8 个 wave**。这部分工作量与槽数无关，与 M 成正比，
所以 M=6 的绝对代价反而比 M=1 小（分母大了）。

**对 design §7.9 的直接后果**：

> 「**数据出来了：翻倍的代价是 0.66 µs × 2 × 40 层 ≈ 0.05 ms/token，可以忽略。
> 因此"按到达顺序分组计算"是默认方案**」

这句话要改。实测每层拆一次的代价是 **M=1 时 0.193 ms、M=6 时 0.108 ms**，
40 层就是 **7.7 / 4.3 ms/token**。按 §3.1 的 cost model，
只有当它**真的换来了 I/O 等待的重叠**（即一层里确实有 expert 迟到）才划算，
而不是无条件默认开启。**建议**：
`Planner` 只在「本层已有 ≥ 1 个 expert 未就位且预计等待 > 0.2 ms」时才分组，
其余情况一次算完。§9.4 已经把 lookahead 预取降级为不做，这一条是同一类判断。

---

## 7. Track E 需要知道的接口变化

全部是**加法**，没有删改现有字段的语义。

**`gpu/vulkan/moe_kernels.h`**

- `MoeSpec` 新增 `x_mode`、`x_mode_b`、`h_quant`、`fp8_slots`，默认值都是
  「P1 的行为」（0）。`x_mode_b` 默认 `kFollowA`，即跟随 `x_mode`；
  它让 dispatch A 和 dispatch B 拿到**不同特化的 pipeline**（§7.9.1 问题 3）。
- `MoeSpec::rows_per_lane` 的合法范围从 {1,2,4} 放宽到 1..16 的幂次。
- `MoeDims` 新增 `fp8_slot_count`：**只用于字节记账**（有效 GB/s 的分子），
  不影响 kernel 行为；哪个槽是 fp8 由 `ids()` 决定。
- 新常量 `gpu::kSlotFp8 = 0x80000000`：写进 `ids()[slot]` 的高位表示
  「这个槽的权重是 FP8 E4M3 + `[rows/32][K/32]` scale 平面」，低位仍是
  指针表里的 expert 下标。
- 新方法 `set_accumulate(bool)`：dispatch B 累加进 `y` 而不是覆盖（§6）。

**descriptor / push constant**

- **dispatch A 的 storage buffer 从 7 个变成 8 个**：binding 7 是 `h` 那块
  allocation 的 `RWStructuredBuffer<uint>` 别名（`HQuant = 2` 时写 fp8 字节和
  scale 用）。binding 0–6 不变。**这是唯一的 descriptor 布局变更。**
- **dispatch B 的 `DownPush` 末尾多了一个 `uint flags`**（bit 0 = 累加），
  结构从 24 B 变成 28 B。`GateUpPush` 不变。
- dispatch B 的 storage buffer 仍是 5 个，binding 不变。

**缓冲大小**

- `h` 的 allocation 大小不变（`M × slots × inter × (h_precision ? 4 : 2)`）：
  fp8 布局是 1.125 B/元素，比 fp16 小，装得下。
- shared expert **不在 `ExpertStore` 里**：它是常驻的，P2 的做法是单独
  `allocate(bytes, host_visible, device_address=true)` 一块，把六个 part 的
  设备地址写进指针表的**一个备用 expert 下标**（`MoeDims::experts_per_layer`
  设成 385，下标 384 留给 shared）。`ExpertStore` 的表是 `[layers][384][6]`，
  所以要**按层重新 stride 地拷贝**，`bench/kernel_bench.cpp` 和
  `tests/test_gpu_moe.cpp` 里各有一份十行的实现可以照抄。

**推荐的默认特化**

```
decode（M=1）        : L32 R1, sg32, dec0（常量表）, h=fp16, XMode=0
                       —— 若开 HQuant=2 则必须换成 L16 R2（见 §5.3）
投机验证（M=6）      : L16 R2, sg32, dec0, h=fp16, XMode=4（A packed fp16）
                       + x_mode_b=3（B int8 dot4）
shared expert 所在的 dispatch: Fp8Slots=1，槽的 ids 带 kSlotFp8
`h` 的 fp8 量化      : HQuant=2（数值上与参考一致，且比 HQuant=1 快得多）
```

---

## 8. v0.1 未解决的问题，与 v0.2 的处置

v0.2（2026-09-14，Track H）只动了 §8 自己列出的头两项，加上 §6.2 的那个测量，
机器与口径和 v0.1 相同（Ryzen AI Max+ 395 / Radeon 8060S / VGM 64 GB，路径 A，
`--iters 48 --layer-cycle 8 --repeats 3 --sweeps 2`）。
**新的 CSV 是 `bench/results/kernel_p2b_moe.csv`**，v0.1 的
`kernel_p2_moe.csv` 原样保留；§2 的漂移规则照旧，
**两个文件之间不要比绝对值**，§9–§11 的每一个对照都在同一节、同一轮里。

| # | v0.1 的问题 | v0.2 |
|---|---|---|
| 1 | `hq8` 的 32 行约束让 M=1 损失 27%（6.3 ms/token） | **解决**（§9）。`HQuant = 3` 把量化搬进第三个 dispatch，M=1 回到 `L32 R1`，代价从 +23% 降到 **−0.5% / +2.2% / +3.9%（三轮，落在 §2 的漂移里）** |
| 2 | x 在 kernel 外量化成 int8 能不能到 187 GB/s | **实现了，A 快 15–30%，但两条都不达标**（§10）。A+B 在 M=6 上仍只有 65–67% 上限，端到端收益 0–7%（漂移量级）；精度在第二个 golden 上 **8.9e-3**，超 §12 的 5e-3。`XMode = 6` **默认关闭** |
| 3 | §15 P2 的 M=6 准出条件应该改写 | 不变，§3.6 的论据仍然成立 |
| 4 | A 和 B 的 `RowsPerLane` 应该分开 | 未做，优先级仍然低（预估 < 2%） |
| 5 | 分组 dispatch 的启用条件 | **前提错了**（§11）。真实代价是**每层 +0.006…0.025 ms**（只拆 A）或 **+0.014…0.046 ms**（两对 A+B），不是 §6.2 写的 0.193 ms；那个数字是漂移，不是代价 |
| 6 | §12 的 int8 判据 | §10 给出两个 expert 的实测。**判据不该为 int8 x 放宽**：错误是 x 的，不是 kernel 的，而且逐 expert 变化 2 倍以上 |
| 7 | run 间 ~7% 的漂移 | 不变，而且 §11 说明它比想象的更能伤人 |

v0.2 那一轮的 M 扫描（不量化的最优变体，上限 215.4–217.8 GB/s，
`bench/results/kernel_p2b_moe.csv`）——**和 §3.5 是同一个形状、同一个量级，
v0.2 没有让任何 M 变慢**：

| M | 最优变体 | A+B GB/s | %上限 | ms/pair | **ms/token** |
|---|---|---|---|---|---|
| 1 | `L32 R1 xglob` | 221.9 | 102% | 0.593 | **0.5930** |
| 2 | `L32 R1 xprei8`（int8 x，§10） | 207.6 | 95% | 0.634 | **0.3170** |
| 3 | `L32 R2 xgf16` | 178.1 | 82% | 0.739 | **0.2463** |
| 4 | `L32 R2 xgf16` | 164.1 | 75% | 0.802 | **0.2004** |
| 5 | `L16 R2 xprei8`（int8 x，§10） | 153.7 | 71% | 0.857 | **0.1713** |
| 6 | `L16 R2 xgf16/ldsi8` | 145.2 | 67% | 0.906 | **0.1511** |

M=2 和 M=5 的赢家是 `XMode = 6`，但那两格的领先都在 §2 的漂移里，
而且 `XMode = 6` 因为 §10.4 的精度默认关着；**不含 int8 x 的最优是
M=2 `L32 R1 xgf16`、M=5 `L16 R2 xgf16/ldsi8`，ms/token 各高 2–6%。**
开着 `HQuant = 3`（也就是 design §7.9 v0.6 要求的 `h` 量化）的同一张表在 §9.4。

---

## 9. `HQuant = 3`：把 `h` 的量化拆成第三个 dispatch（§8 第 1 项）

### 9.1 为什么不是 LDS 转置

§5.3 给了两条路：给 dispatch A 的写出阶段解耦 workgroup 形状（LDS 转置），
或者拆成第三个极小的 dispatch。选了后者，理由是前者做不到：
fp8 的块是 **`h` 沿 2304 维连续的 32 行**，而 `L32 R1` 的一个 workgroup 只拥有
`256/32 × 1 = 8` 行。LDS 转置只能在 workgroup 内部换轴，换不来它根本没有的
另外 24 行；要凑够 32 行就得 `L16 R2` 或 `L32 R4`——正是要躲开的那个约束。
跨 workgroup 交换块 scale 则需要一次全局 barrier，也就退化成了第三个 dispatch。

### 9.2 实现：`gpu/shaders/moe_hquant.slang`

一个线程一个 32 元素块：读 64 B 的 fp16 `h`，求 amax，`fast_round_scale`，
写 32 B 的 fp8 + 一个 scale word。整个 `h` 是 `M × slots × 2304` 个值——
M=1 时 32 KB，M=6 时 194 KB——所以 M=1 是 504 个线程 2 个 workgroup。
`SlotList` 是它的第一个 binding，所以 §7.9 的「先到的先算」照样切得开。

**唯一的代价是它不能就地量化。** 线程 t 读 fp16 字节 `[64t, 64t+64)`，
要写的 fp8 字节 `[32t, 32t+32)` 落在线程 t/2 的读区间里。所以 fp8 平面放在
fp16 平面**之后**（`moe_common.slang` 的 `hq_value_words`），`h` 的 allocation
从 2 B/元素变成 **3.125 B/元素**（fp16 2 + fp8 1 + scale 0.125）。
M=6、7 槽时是 302 KB，和 132 MB 的权重流比可以忽略。
`HQuant = 2` 的就地布局一个字节没改。

### 9.3 还顺手修掉了 dispatch B 的一笔冤枉钱

第一版 `HQuant = 3` 在 M=1 上仍然贵 5.2%，拆开看有 0.040 ms 在 **dispatch B**：
`load_h8q` 对 fp8 `h` 的每个元素做一次 `sFp8[byte] * s`，也就是每 (列, 块)
32 次乘法，而块 scale 在整块里是常数。把它提到块外——
`acc += ldexp(p * hs[m], e2)`——**每 (行, 列, 块) 一次乘法**取代 32 次。
块 scale 是 2 的幂，提取是逐位精确的，`tests/test_gpu_moe.cpp` 里
`vs y_hq16` 的 5.030e-08 改动前后一个数字都没变。
`RowsPerLane = 1` 时这 32 次乘法完全没有被摊薄，正是 M=1 decode 形状的那笔钱。

### 9.4 M 扫描，HQuant 开着（同一节内成对测量，`--only "h fp8"`，机器空闲）

每一行的「不量化」和「hqP」是**相邻测量**的（§11.1 说明为什么这很重要）。
两种形状都列出来，因为 M ≤ 2 的赢家是 `L32 R1`、M ≥ 3 的是 `L16 R2`：

| M | 形状 | 不量化 ms/pair | `hqP` ms/pair | 代价 | `hqP` ms/token |
|---|---|---|---|---|---|
| 1 | `L32 R1 xglob` | 0.608 | **0.632** | **+3.9%** | **0.632** |
| 1 | `L16 R2 xgf16/ldsi8` | 0.781 | 0.795 | +1.8% | 0.795 |
| 2 | `L32 R1 xglob` | 0.692 | 0.833 | +20.4% | 0.416 |
| 2 | `L16 R2 xgf16/ldsi8` | 0.865 | **0.872** | **+0.8%** | 0.436 |
| 3 | `L32 R1 xglob` | 0.805 | 0.984 | +22.2% | 0.328 |
| 3 | `L16 R2 xgf16/ldsi8` | 0.808 | **0.788** | **−2.5%** | **0.263** |
| 4 | `L16 R2 xgf16/ldsi8` | 0.837 | **0.848** | **+1.3%** | **0.212** |
| 5 | `L16 R2 xgf16/ldsi8` | 0.901 | **0.871** | **−3.3%** | **0.174** |
| 6 | `L16 R2 xgf16/ldsi8` | 0.950 | **0.992** | **+4.4%** | **0.165** |

**在每个 M 真正会用的形状上，`hqP` 的代价落在 −3.3% … +4.4% 之间，
也就是落在 §2 的漂移里。** 对比 v0.1 的 `hq8`（M=1 +27%）：
M=1 量了三轮，分别是 **−0.5% / +2.2% / +3.9%**（`--only` 两轮 + 全 sweep 一轮），
中位数 +2.2%——**decode 的 6.3 ms/token 量化税降到 0.4–0.9 ms/token**。

`L32 R1` 在 M ≥ 2 上那 20% 不是 `hqP` 的问题、也不是第三个 dispatch 的问题：
是 **dispatch B 在 R=1 上给 M 个列做 fp8 反量化**（M=2 时 B 从 169.7 掉到
116.2 GB/s），而 `L32 R1` 在 M ≥ 2 上本来就不是赢家。

### 9.4.1 和另外两种实现比（同一节，全 sweep 的那一轮）

| 变体 | A GB/s | B GB/s | A+B GB/s | ms/pair | 对同节对照 |
|---|---|---|---|---|---|
| M=1 `L32 R1 xglob`（不量化，对照） | 218.4 | 196.8 | 218.2 | **0.603** | — |
| M=1 `L32 R1 xglob hqP` | 217.3 | 190.4 | 213.6 | **0.616** | **+2.2%** |
| M=1 `L16 R2 xglob hq8` | 167.7 | 202.4 | 176.9 | 0.744 | +23.4% |
| M=1 `L32 R4 xglob hq8` | 178.9 | 156.3 | 172.4 | 0.763 | +26.5% |
| M=1 `L32 R1 xglob hqB` | 222.9 | 84.8 | 151.1 | 0.871 | +44.4% |
| M=6 `L16 R2 xgf16/ldsi8`（不量化，对照） | 133.2 | 122.2 | 137.2 | **0.959** | — |
| M=6 `L16 R2 xgf16/ldsi8 hqP` | 139.2 | 118.3 | 139.2 | **0.945** | **−1.5%** |
| M=6 `L16 R2 xgf16/ldsi8 hq8` | 136.0 | 118.5 | 135.8 | 0.969 | +1.0% |
| M=6 `L16 R2 xgf16 hqB` | 142.6 | 36.2 | 72.8 | 1.807 | +88.4% |

**M=6 上 `hqP` 不比 `hq8` 慢**（两者都在对照的 ±1.5% 内），
所以换过去没有任何代价。`hqB` 在 `L32 R1` 上 dispatch B 只有 84.8 GB/s：
它要为**每个权重行**重算同一个块 amax，R=1 时一个都摊不掉。
`runtime/moe_bridge.h` 现在的默认就是 `h_quant = 1`，那是 **+44%**（§12）。

### 9.5 数值

`HQuant = 3` 与 `HQuant = 2` 是同一段算术换了个 dispatch，所以
`tests/test_gpu_moe.cpp::the_fp8_h_quantisation_matches_the_reference` 里
两者对 `y_hq16` 都是 **5.030e-08**，和 §5.2 的表逐位相同；
`L32 R1` / `L16 R2` / `L64 R1` / M=6 四种形状给出同一个数，
这正是 §5.3 说 `hq8` 做不到的事。

---

## 10. `XMode = 6`：x 在 kernel 外量化成 int8（§8 第 2 项）

### 10.1 实现

`gpu/shaders/moe_xquant.slang`，一个线程一个 32 元素块：
x `[M][5120]` fp16 → int8 + 每块一个 fp32 scale，追加在同一块 x allocation 的
fp16 之后（`xq_value_words`）。写区间从 `M·k/2` 个 word 开始、读区间到它为止，
所以这里就地量化是安全的（`h` 不行，见 §9.2）。
**这是每 *token* 一次，不是每层一次**——同一个 x 喂 40 层 MoE——但 bench 每次
迭代都重录一遍，所以下面的 `ms_a` 只会高估它。

dispatch A 侧 (`XMode = 6`) 每 (列, 块) 是 2 条 `global_load_dwordx4` + 1 个
scale + 每 `RowsPerLane` 行 16 条 `dot4add_i8packed`，没有 `f16tof32`、
没有 amax、没有取整。

### 10.2 §3.6 的模型漏了权重解码

第一版量到 dispatch A 在 M=6 上只有 147 GB/s，和 packed fp16 打平。拆开看：
§3.6 的模型只数了**激活侧**的指令，而 int8 路径的瓶颈在**权重侧**——
`fp4_nibbles_to_i8x4` 每 (块, 行, 矩阵) 要调 8 次，每次 4 个 nibble 查表 +
移位 + 拼装，粗算是每块 512 条，而 M=6 个列一共才 150 条。**权重解码占 3/4。**

修法和 §4 的 fp8 解码同一套：**256 项 LDS 表**，一个 FP4 字节（两个元素）
直接给出两个 `2×E2M1` 的 int8，于是一个输出 word = 2 次 `ds_read` + 一次
移位或。加上把 8 条标量 word 读换成 2 条 `dwordx4`，M=6 的 dispatch A 从
147 走到 **168.1 GB/s**。

### 10.3 速度：dispatch A 快 15–30%，端到端在漂移里

同一节内对照（`int8 x` 那一节带着自己的 fp16 对照），全 sweep 的那一轮：

| M | 最优 fp16 对照 ms/token | 最优 int8-x ms/token | 变化 |
|---|---|---|---|
| 1 | `L32 R1 xglob` **0.596** | `L32 R1 prei8` 0.603 | +1.2% |
| 2 | `L32 R1 xglob` 0.337 | `L32 R1 prei8` **0.317** | **−5.9%** |
| 3 | `L16 R2 xgf16/ldsi8` 0.255 | `L32 R1 prei8` **0.249** | −2.4% |
| 4 | `L16 R2 xgf16/ldsi8` **0.202** | `L16 R2 prei8` 0.206 | +2.1% |
| 5 | `L16 R2 xgf16/ldsi8` 0.173 | `L16 R2 prei8` **0.171** | −1.2% |
| 6 | `L16 R2 xgf16/ldsi8` **0.153** | `L16 R2 prei8` 0.154 | +0.7% |

第二轮（`--only "int8 x"`，独立运行）在 M=6 上给出 0.161 → **0.150（−7.0%）**，
M=2 给出 −11.5%。**两轮的方向一致、幅度差一倍**，所以诚实的说法是：
**端到端 M ≥ 2 的收益在 0…7% 之间，和 §2 的漂移同一量级。**

拆开看，dispatch A 上的收益是真的、也是稳定的：M=6 的 A 从 137.5 走到
**158.8 GB/s（+15%）**，另一轮是 129.6 → **168.1（+30%）**，
两轮里它都是 M=6 全场最好的 A。

**但 §3.6 的结论仍然不成立**：M=6 的 A+B 只有 **142.6–146.6 GB/s = 上限的
65–67%**，离 §15 P2 的 80% 还差得远。原因很直接——瓶颈换到了 dispatch B
（M=6 时 105–108 GB/s），而 **B 的激活是 `h` 不是 x**，这条路对它无能为力。
§3.6 写的「这是唯一能让 M=6 摸到 80% 线的设计」，实测是错的。

`RowsPerLane` 这次确实能往上开一格——int8 路径不持有 `float d[R][8]`，
寄存器压力只有 fp32 路径的一半，M=1 时 `L16 R4` 甚至是最好的 int8 变体
（0.689–0.700 ms）——但 R=8 依然塌方（M=6 的 A 只有 47.4 GB/s），
而且 R=4 在 M ≥ 3 上没赢过 R=2。§3.3 的结论往后挪了一格，没有被推翻。

### 10.4 精度：**不达标**，而且逐 expert 差两倍

`tools/oracle_shared.py` 现在对每个 routed golden 跑一遍 `x_quant_study`：
权重侧在 int8 路径上是**精确的**（`2×E2M1` 就是 int8，那个 2 折进块指数），
所以误差全部是 x 自己的，torch 模型和 GPU 应该对得上——实测对得上到三位有效数字。

| x 的形式 | (0, 0) 占 \|y\|max | (39, 383) 占 \|y\|max |
|---|---|---|
| fp16（`XMode 0/4`，design §6） | 1.37e-4 | 3.65e-4 |
| **int8 + 每 32 元素一个 scale（`XMode 6`）** | **2.90e-3** | **8.85e-3** |
| int8 + 每行一个 scale | 4.45e-3 | 1.40e-2 |
| int8 + fp16 残差 | 1.36e-4 | 3.67e-4 |

GPU 侧（`tests/test_gpu_moe.cpp::the_int8_x_pre_pass_is_expert_dependent`）：
(0, 0) **2.902e-3**、(39, 383) **8.856e-3**，与上表吻合。

三条结论，都是 §8 第 2 项直接问的：

1. **过不了 §12 的 5e-3。** (0, 0) 的 2.9e-3 过得去，(39, 383) 的 8.9e-3 过不去。
   一个判据不能靠挑 expert 来满足。
2. **每行一个 scale 只会更糟**，不是更好：块内 amax ≈ 2.5σ，整行 amax ≈ 4σ，
   量化步长粗 1.6 倍，实测误差就是 1.5 倍。它省的是每 (列, 块) 一条标量读，
   不是精度。
3. **int8 + fp16 残差确实能修好**（回到 1.4e-4 / 3.7e-4），但那只是把 fp16 的
   乘加拆成两段做：要多一个 fp16 残差平面、多一条 FMA 流，**比直接用
   `XMode = 4` 的 packed fp16 更贵**。它证明的是「误差确实全部来自 x 的量化」，
   不是一条可用的实现。

**处置**：`XMode = 6` 实现完整、有测试、**spec 常量默认 0（关闭）**。
它唯一说得通的用法是 §10 的**投机验证批**：那里 M ≥ 4（收益最大的区间），
而且接受检查本来就要拿 draft 和 target 对比，对 MoE 输出的近似有容忍度。
**不要把它设成 decode 的默认**——M=1 上它本来就是负收益（+3.2%）。

---

## 11. §6.2 的 0.193 ms 是漂移，不是代价

### 11.1 v0.1 怎么测错的

§6.2 的三个数是**依次**测出来的：先 `whole`（7 槽），再 `first`（3 槽），
再 `rest`（4 槽），每个都是「预热 + 3 次取最快」。§2 自己写着
「同一轮不同小节之间有 8% 的漂移（后跑的节更热）」——
这三个数就隔着几秒钟，而要测的差是 1–3%。于是后跑的 `first + rest` 被系统性
地拉高，差额被记成了「分组 dispatch 的代价」。

一个能自我检查的信号在 v0.2 的第一次全 sweep 里露了出来：
分组 dispatch 那一节跑在 25 分钟 sweep 的最后，量到
`one pair 0.652 ms (A 0.520 + B 0.290)`——**A + B = 0.810 > 0.652**，
物理上不可能，纯粹是三个数取自三个热状态。

v0.2 把这一节改成**九个配置轮转测量、各取自己的最好值**
（`bench/kernel_bench.cpp`），同一次运行里 `A + B` 就和 `whole` 对得上了
（0.391 + 0.201 = 0.591 对 0.592）。

### 11.2 真实的代价

机器空闲、同一节内轮转：

| 轮次 | M | 一次 7 槽 | 两对 A+B（3 + 4） | split A + 一次 B |
|---|---|---|---|---|
| 冷（`--only`） | 1 | 0.592（A 0.391 + B 0.201） | 0.613（**+0.021**, +3.5%） | 0.605（**+0.013**, +2.2%） |
| 冷（`--only`） | 6 | 0.960（A 0.628 + B 0.330） | 0.985（**+0.025**, +2.6%） | 0.970（**+0.010**, +1.1%） |
| 冷（第二轮） | 1 | 0.634（A 0.407 + B 0.229） | 0.668（**+0.034**, +5.4%） | 0.652（**+0.018**, +2.9%） |
| 冷（第二轮） | 6 | 0.942（A 0.622 + B 0.326） | 0.987（**+0.045**, +4.7%） | 0.966（**+0.023**, +2.5%） |
| 热（25 分钟 sweep 之后） | 1 | 0.728（A 0.449 + B 0.276） | 0.743（**+0.014**, +1.9%） | 0.734（**+0.006**, +0.8%） |
| 热（25 分钟 sweep 之后） | 6 | 1.018（A 0.666 + B 0.359） | 1.063（**+0.046**, +4.5%） | 1.043（**+0.025**, +2.5%） |

（单位 ms/层。热的那两行绝对值高 10–20%——这正是 §2 的漂移——
但**差额**在冷热两种状态下是一致的，因为现在是轮转测量的。
`A + B` 与 `whole` 在每一行都对得上，这是这一节以前没有的自检。）

**六行里没有一行超过 0.05 ms/层，`split A + 一次 B` 全部 ≤ 0.025 ms/层。**
40 层是 **0.2–1.0 ms/token**，不是 §6.2 写的 7.7 ms/token。

### 11.3 推荐的调度：只拆 dispatch A

两种切法差别不大，但**「只拆 A、B 在最后跑一次」更好，而且更省事**：

- dispatch A 占一层的 **2/3**（M=1 时 0.391 / 0.592），而且它的工作量与
  `list_count` **严格成正比**（`gid.y` 就是槽）——所以能和 I/O 等待重叠的
  正是它，拆开不浪费任何字节。
- dispatch B **本来就要等所有槽的 `h`**，推迟到最后不损失任何重叠机会，
  而且省掉了第二次 640 个 workgroup 的跨 lane 归约与 `y` 读改写。
- **它与一次算完逐位相同**：归约没有被重新结合，
  `tests/test_gpu_moe.cpp::a_partial_dispatch_reduces_to_the_same_y` 对五种切法
  都量到 `5120/5120`（M=1）/ `30720/30720`（M=6）个字完全一致。
  两对 A+B 的切法仍然是 §6.1 的 1–2 ULP。

写法（`MoeRunner`，无接口变化）：

```cpp
runner.set_list_count(3);  runner.set_accumulate(false);
runner.run(1, gpu::MoePhase::GateUpOnly);     // 先到的三个
/* ... 等 I/O ... */
runner.slot_list()[0..3] = {3,4,5,6};
runner.set_list_count(4);
runner.run(1, gpu::MoePhase::GateUpOnly);     // 后到的四个
runner.slot_list()[0..6] = {0..6};
runner.set_list_count(7);  runner.set_accumulate(false);
runner.run(1, gpu::MoePhase::DownOnly);       // 一次归约
```

**对 §6.2 建议的修正**：那条「只在预计等待 > 0.2 ms 时才分组」的门槛可以拿掉。
每层 0.01–0.03 ms 的代价意味着**只要真有一个 expert 迟到，拆就是划算的**；
Planner 需要判断的只是「这一层是不是真的有 expert 没就位」，
不需要再去估等待时长。§9.4 把 lookahead 预取降级为不做的那条结论不受影响。

---

## 12. Track G 需要知道的接口变化（v0.2）

仍然全部是**加法**，v0.1 §7 列的东西一个语义都没改。

**`gpu/vulkan/moe_kernels.h`**

- `MoeSpec::h_quant` 多一个取值 **3**：`h` 的 fp8 量化由 A 与 B 之间的第三个
  dispatch 做。**这是现在应该用的那个值**——数值上与 2 逐位相同，而且不对
  dispatch A 的 workgroup 形状提任何要求。
  `runtime/moe_bridge.h` 的 `MoeBridgeConfig::h_quant` 现在默认 1（`hqB`），
  在 `L32 R1` 上那是 **+40%**；改成 3 是一行的事，代价 −0.5%。
  **限制**：`h_quant = 3` 要求 `h_precision == 0`（它量化的就是 A 写出的 fp16 `h`）。
- `MoeSpec::x_mode` 多一个取值 **6**：dispatch A 消费预量化的 int8 x。
  **默认仍是 0，不要打开它做 decode**（§10）。`x_mode_b` 取 `kFollowA` 时
  遇到 `x_mode == 6` 会落回 0，因为 dispatch B 的激活是 `h` 不是 x。
- `set_accumulate` 的注释里写清了两种分组调度和该选哪个（§11.3）。
- **`MoeRunner` 内部多了两条 pipeline**（`hquant_` / `xquant_`）和两个
  descriptor set，只在 `h_quant == 3` / `x_mode == 6` 时创建。
  `run()` 会自动把它们录进同一个 command buffer，**调用方什么都不用做**。
  `MoePhase::GateUpOnly` 的计时包含 `moe_hquant`（它是「产出 h」的一部分），
  `MoePhase::DownOnly` 不包含。

**descriptor / buffer**

- **dispatch A 的 storage buffer 从 8 个变成 9 个**：binding 8 是 x 那块
  allocation 的 `StructuredBuffer<uint>` 别名（`XMode = 6` 读 int8 与 scale 用）。
  binding 0–7 不变。这是 v0.2 唯一的 descriptor 布局变更。
- **`h` 的 allocation 在 `h_quant == 3` 时是 3.125 B/元素**（其余情况不变）；
  **x 的 allocation 在 `x_mode == 6` 时是 3.125 B/元素**（其余情况 2 B/元素）。
  两者都由 `MoeRunner::create` 自己分配，`h()` 和 `x_fp16()` 指向的还是
  各自平面的开头，**调用方的写法一个字都不用改**。
- 新 shader 两个：`gpu/shaders/moe_hquant.slang`、`gpu/shaders/moe_xquant.slang`，
  已经进 `cmake/deepmoe_options.cmake` 的 `DEEPMOE_SHADERS`，
  和其余 kernel 一样每次编译都过 `spirv-val --target-env vulkan1.3`。

**推荐的默认特化（取代 v0.1 §7 的那一段）**

```
decode（M=1）        : L32 R1, sg32, dec0, h=fp16, XMode=0, HQuant=3
投机验证（M=6）      : L16 R2, sg32, dec0, h=fp16, XMode=4, x_mode_b=3, HQuant=3
                       —— 若接受检查容得下 x 的近似，XMode=6 再快 7%（§10）
shared expert 的 dispatch : Fp8Slots=1，槽的 ids 带 kSlotFp8
分组 dispatch        : 只拆 dispatch A，dispatch B 最后跑一次（§11.3）
```

---

## 附：如何复现

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'

# 先生成 P2 需要的三个 golden（fp8 shared expert + 量化 h 的三个参考答案）
uv run python tools/oracle_shared.py --model D:/models/DeepSeek-V4.1-Flash `
    --shared 0 --expert 0:0 --expert 39:383 --out tests/data

# 正确性：八个用例，含 M>1 逐列、fp8 shared expert、h 量化（含 HQuant=3）、
# int8 x 的逐 expert 误差、分组 dispatch 的两种切法
ctest --test-dir build -R suite.gpu_moe --output-on-failure

# v0.1 的 sweep（§0–§7 的表）
.\build\kernel_bench.exe --csv bench\results\kernel_p2_moe.csv `
    --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2

# v0.2 的 sweep（§9–§11 的表）。同一条命令，因为 v0.2 的变体是加上去的
.\build\kernel_bench.exe --csv bench\results\kernel_p2b_moe.csv `
    --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2

# 只重测一节 —— 一整轮 25 分钟，足够让芯片热几度（§2 / §11.1）。
# --only 过滤 section 名；不匹配任何东西就只剩分组 dispatch 那一节
.\build\kernel_bench.exe --only "h fp8"  --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2
.\build\kernel_bench.exe --only "int8 x" --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2
.\build\kernel_bench.exe --only none     --iters 48 --layer-cycle 8 --repeats 3

# P1 的旋钮 sweep（解码方式 / lane 数 / wave 宽度）仍然可以跑
.\build\kernel_bench.exe --p1 --csv bench\results\kernel_p1.csv
```

`tools/oracle_shared.py` 现在顺带打印 §10.4 的 x 量化对照
（fp16 / int8 每块 / int8 每行 / int8 + fp16 残差），不需要 GPU。

带 validation layer 跑一遍（两个 GPU 测试与 bench 当前都是干净的，
`spirv-val --target-env vulkan1.3` 由 `add_slang_shader()` 每次编译都跑）：

```powershell
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
$env:VK_LOADER_LAYERS_ENABLE='VK_LAYER_KHRONOS_validation'
```

**测量时不要同时编译**（kernel_p1.md §1），并且**跨轮只比同轮内的相对值**（§2）。
§11.1 是这条规矩被违反一次的代价：v0.1 §6.2 的「分组 dispatch 每层 +0.193 ms」
整个是漂移。凡是要量 1–5% 的差，**对照必须和被测量的东西轮转着测**——
`bench/kernel_bench.cpp` 的分组 dispatch 那一节现在就是这么做的，
`h fp8` 与 `int8 x` 两节也各自带上了同节的不量化对照。
