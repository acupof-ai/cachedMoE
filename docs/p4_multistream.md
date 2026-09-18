# Track MS — 一个引擎进程里的多路 decode（2026-09-19）

一句话：**decode 的 per-token 时间线是「~97 ms 计算 + ~100 ms NVMe stall，两段不重叠」
（`p4_p0_queue.md`）。两条互相独立的对话之间没有数据依赖，所以 A 的 stall 就是 B 的计算窗口。**
本轮把引擎按「进程拥有的」和「序列拥有的」切开，落地了 **D1：层级交错的双流 decode**
（`Engine::decode_step_multi`），并把 **D2：token 级乒乓**作为同一套代码里的对照臂。

出处：`C:\Users\Asus\code\deepmoe-ms`（分支 `p4/ms-multistream`）。
脚本 `bench/results/hitrate/y_turns.json` 与 `long_turns.json`，`--cache-slots 5100`，
`--warm-cache`，一次一个 `deepmoe serve` 进程。原始结果在 `bench/results/ms/`。

---

## 1. 切在哪里：进程的 vs 序列的

`Engine` 原来是一个对象，既拿着 510 GB 的东西，也拿着一个 token 的激活。
本轮引入 `runtime::Stream`（`runtime/engine.h`），把后者整块拿出来：

| 留在 `Engine`（进程的，共享） | 移进 `Stream`（序列的，每路一份） |
|---|---|
| `ShardSet` / `IoEngine` / `PinnedStore`（9.2 GiB） | `KvStore`（4,096 位置 ≈ 15 MB） |
| **`ExpertStore` + `Planner`（89 GiB 的 expert cache）** | `GpuScratch`（32 MB）、`DecodeLayer` |
| `gpu::Device` / 两条 `MemoryAllocator` | **`AttnRunner` / `DecodeRunner` / `GpuMoeBridge` / `EngramRunner`** |
| `weights_`（pinned 权重地址，不可变）、`ced_` | 两条 timeline（residency gate + 完成 fence）、command pool/buffer、query pool |
| LRU 时钟 `clock_`、eviction guard 时钟 | logits / sample / top-k / FFN-input 缓冲、`history_`、`token_` |

**为什么 runner 必须每路一份**（这是本轮第一个不显然的约束）：
`AttnRunner::slots(stage)` 返回的是一块 **host-visible 的地址表，shader 在 dispatch
真正执行的时候才去读它**，不是 record 时烤进 command buffer 的常量。
`decode_layer.h` 里那条 `bind_close` 注释就是单流版本的同一个坑
（"`bind` for L-1 left MhcClose pointing at L-1's hc_ffn weights … `bind_close`
stops this layer's `bind` from overwriting it before the buffer runs"）。
两条流共享一个 runner，B 的 `bind` 会在 A 已提交、未执行的 dispatch 底下把表改掉。
代价很便宜：pipeline 是已经编译好的 SPIR-V 模块，表是几 KB；**一路流合计 ~50 MB**，
对着 89 GiB 的 cache 可以忽略——**cache 是唯一大的东西，而它是共享的**。

一个 `Stream` ≈ 50 MB，所以 N 路的内存不是问题；**问题是 N 个工作集共用一个 LRU**（§5）。

## 2. `run_layer` 切成三段

交错要求「发出 P0 之后不要马上等」。所以 `run_layer` 拆成：

| 段 | 做什么 | 结束时这条流在等什么 |
|---|---|---|
| `layer_begin` | engram、`bind`、录 attention 链、`cmd_submit(prev_gate)` | GPU 在跑它的 attention |
| `layer_gate` | 等 fence、读 gate 的 ids、Track Y 路由、`plan_layer`——**发出 P0 miss 后立刻返回** | 盘在读它的 expert |
| `layer_moe` | `wait_layer`（= NVMe stall）、signal timeline、stage、录 MoE | — |

单流的 `run_layer` 就是这三句连写，**语句顺序一个字没变**，所以单流路径是同一份代码。

### 2.1 第一版（`interleave`）：把三段按相分组——**只买到 +9%**

```
for L in 0..40:
    for s in streams: layer_begin(s, L)     # 每条流的 attention 依次进同一个队列
    for s in streams: layer_gate(s, L)      # 先提交的先拿到 fence，先把 miss 发给盘
    for s in streams: layer_moe(s, L)
```

直觉是「两条流的 miss 都发出去了再一起等」。**实测 4.584 → 4.992 tok/s（+8.9%）**
（`bench/results/ms/abab1/`，serial vs interleave，一对）。

失败的原因在时间线上一眼可见：`layer_gate(s_{n-1})` 会等 `s_{n-1}` 的 fence，
而那是这一轮**最后一次提交**——所以走到第一个 `layer_moe` 的时候，
**这一轮所有的 submit 都已经退休了，GPU 上一个 dispatch 都没有**。
两条流的盘读确实同时在飞，但它们谁也没有和计算重叠。

### 2.2 第二版（`pipeline`，默认）：**进 stall 之前队列里必须有活**

```
layer_begin(s0, 0)
for L in 0..40:
    for i in 0..n-1:
        layer_gate(s_i, L)                       # 发出 s_i 的 miss
        if i+1 < n:   layer_begin(s_{i+1}, L)    # 下一条流这一层的 attention
        else:         layer_begin(s_0, L+1)      # 或者 s_0 的下一层
        layer_moe(s_i, L)                        # 这时才等盘，队列里有一条链在跑
```

`layer_begin(s_0, L+1)` 之所以合法，是因为 `s_0` 这一层的 MoE 在 `i == 0` 时已经录并提交、
residency 值已经 signal 过了，所以这次 submit 不会在信号量上把队列堵住。

多流时 MoE 记完就**立刻单独提交**（`DEEPMOE_MS_EAGER_MOE`，默认开）：
单流里 L 层的 MoE 是搭 L+1 层 attention 的便车走的，但交错时 L+1 层的 attention
要等另一条流跑完一整层才录得到，那 MoE 就白等了。代价是每层多一次 submit（~0.15 ms）。

### 2.3 第三版（`pingpong`）：D2 的对照臂

整个 token 跑完 A 再跑 B。**没有任何重叠**，只有「两条对话共用一个 cache」这一项
（它是负的）和调度本身的开销。它存在的意义是把 §5 里的收益归因到「重叠」而不是别的。

## 3. 三个必须做对的共享点

1. **eviction guard 跨流**（`Engine::advance_store_guard`）。单流时「我刚等到的这个
   buffer」就是最新的 reader，把它的 guard 号发布给 store 是对的。两条流时 A 的
   buffer 可能比 B **还在飞**的 buffer 更新，直接发布 A 的号会放掉 B 正在读的槽。
   现在发布的是 A 自己的号**被别的流还没证明完成的最小 guard 减一夹住**之后的值，
   **单流时这个夹子是空的**，所以单流路径与 main 一字不差。
   这一条踩了两次，两次都值得记：
   - **第一版发布的是「没有在飞就发布时钟本身」**，比 main 更激进——
     guard 已经发出、buffer 还没提交的那个窗口里，它会把正在被录进去的槽判成可淘汰。
     **改回「用自己的号，被别的流夹住」之后单流与 main 逐字相同。**
   - **第二版又夹得太紧**（连「已发号、buffer 还在录」的 `layer_guard_pending_` 也夹），
     400 槽的 `suite.multistream` 直接 `cache is full and nothing is evictable`。
     最终只夹 `open_guard_`（已录未提交）和 `inflight_guard_`（已提交未等）。
2. **LRU 时钟是进程级的**（`Engine::clock_`）。每条流有自己的 `token_`（它给
   **自己那条 residency timeline** 编号，timeline 是每路一条），但交给
   `planner_.plan_layer` 的必须是同一个时钟，否则淘汰会按「哪条流最近 tick 过」排序。
3. **每步的命中率必须是这条流自己的**。`ExpertStore` 的计数器是进程级的，两条流同时
   跑的时候对它取差得到的是两条流的合计。`StepBreakdown` 新增 `requests / hits /
   miss_bytes`，从这一步自己的 40 个 `LayerTiming` 汇总——`GenerateStats` 现在用它。

## 4. 接口

- `Engine::set_streams(n)` / `select_stream(i)` / `begin_session_on(i, sc)`
- `Engine::decode_step_multi(steps, out)`、`feed_multi(steps, out)`
- `Engine::set_ms_sched(Pipeline | Interleave | PingPong)`（默认 `Pipeline`）
- `runtime::generate_multi(...)`（`runtime/session.h`）：每路一个 turn，prompt 逐路
  单独 prefill（prefill 本身就是 decode 形状的循环，交错它没有收益、状态却全是坑），
  然后 generation loop 一起跑；先结束的那路退出，剩下的继续，所以「N 路满」的轮数要单独报。
- `deepmoe serve --streams N [--ms-sched pipeline|interleave|pingpong] [--warm-cache]`，
  新 op `{"op":"generate_multi","requests":[{...,"stream":i}]}`，
  事件带 `"stream"`，末尾多一条 `done_multi`。
- `tools/ms_bench.py`（每路一个 chat 脚本）+ `bench/ms_abab.py`（ABAB 谐波）。
  `--sched serial` 是**基线臂**：同样 N 条流、同样每路一个 KV、同样共享的 cache，
  **只有调度不同**——所以 serial→pipeline 的差就是调度本身的差。

---

## 5. 结果

### 5.1 主表：三种调度，ABAB 三对（4 轮 × 64 token，两条脚本，5,100 槽，`--warm-cache`）

`bench/results/ms/abab2/`。每个 cell 一个进程，三臂轮转，`bench/ms_abab.py`。

| 臂 | 合计 tok/s（3 cell 均值） | cell 间 sd | × vs serial | 每路 tok/s | 每路 hit | 每路 stall ms/token | 每路 MB/token |
|---|---|---|---|---|---|---|---|
| **`serial`（基线）** | **4.6474** | 0.55% | 1.000 | 4.415 / 4.906 | 0.8841 / 0.9048 | 127.2 / 105.1 | 523 / 430 |
| **`pipeline`（D1）** | **5.4602** | 0.12% | **1.175** | 2.730 / 2.730 | 0.8685 / 0.9022 | 136.9 / 107.5 | 594 / 442 |
| `pingpong`（D2） | 4.4123 | 0.83% | **0.949** | 2.206 / 2.206 | 0.8690 / 0.9019 | 142.3 / 107.8 | 591 / 443 |

逐 cell：serial `4.6221 / 4.6729 / 4.6474`、pipeline `5.4680 / 5.4556 / 5.4570`、
pingpong `4.4487 / 4.3758 / 4.4124`。**三臂互不重叠，全部远在 ±3% 的抖动带之外。**

三件事：

1. **D1 是 +17.5%，不是预测的 1.6–1.8×。** 预测按「compute 97 / stall 100，完全重叠」
   算出 ~1.7×，砍半是 1.3–1.4；**实测 1.175 连砍半后的下沿都没到**。为什么见 §5.3。
2. **D2（token 级乒乓）是 −5.1%，是个负数。** 它证明 §5.1 的收益**确实来自重叠**，
   不是来自「两条对话共用一个热 cache」——后者本身是负的（hit 掉 1.6 个点，
   MB/token 涨 13.5%），乒乓把这一项原样付了而一点重叠都没买到。
3. **每路延迟掉到 0.62×**（4.415 → 2.730）。两条对话同时说话时每一条都更慢，
   这是**吞吐换延迟**，不是免费的。

### 5.3 那 1.175 和 1.7 之间差在哪：盘的聚合速率

把主表换算成盘的速率（每对 token 的字节 ÷ 每对 token 的墙钟）：

| | serial | pipeline | pingpong |
|---|---|---|---|
| 合计 tok/s | 4.6474 | 5.4602 | 4.4123 |
| 每对 token 读的字节 | 953 MB | 1,035 MB | 1,034 MB |
| **墙钟上的盘聚合速率** | **2.21 GB/s** | **2.83 GB/s（+28%）** | 2.28 GB/s |

**机制是对的**：两条流的 P0 同时在飞，盘的聚合速率涨 **28%**。
**但它只变成 17.5% 的吞吐**，因为字节数同时涨了 **8.6%**——两个工作集抢一个 5,100 槽的 LRU，
`y_turns` 的 hit 从 0.8841 掉到 0.8685（**−1.6 pt**）、MB/token 从 523 涨到 594（**+13.5%**）。
**这正是事前点名的风险，量级也对上了**（预测 2–4 pt / −10–15%，实测 1.6 pt / −13.5% 的字节）。

**剩下的一半在哪**：D: 在引擎突发形状下的天花板是 **4.10 GB/s**（Track Q2），
2.83 只到它的 **69%**。缺口不是调度的形状而是**每次 stall 前能压进队列的 GPU 活太少**：
一层的 stall 平均 **3.4 ms**（137 ms ÷ 40），而 `pipeline` 在每次 `wait_layer` 之前
只压得进**一条 attention 链（~2.1 ms）**。要盖满就得有更多条流——而 §5.4 说那条路是关的。

### 5.4 N 的扫描：**加流不会继续买到东西**

`bench/results/ms/nscan/`。**基线臂 `serial` 与实验臂 `pipeline` 用的是同一个进程形状**
（同样 N 条流、同样每路一个 KV、同样共享的 cache），只有调度不同。

| N | 脚本 | serial 合计 tok/s | pipeline 合计 tok/s | ×  | pipeline 的每路 hit | pipeline 的每路 MB/token |
|---|---|---|---|---|---|---|
| 2 | y, long | **4.395** | **5.199** | **1.183** | 0.849 / 0.900 | 682 / 453 |
| 3 | + smoke | 4.297 | 4.807 | 1.119 | 0.830 / 0.886 / 0.853 | 768 / 516 / 662 |
| 4 | + chat3 | 5.037 | 5.652 | 1.122 | 0.830 / 0.886 / 0.853 / **1.000** | 768 / 516 / 662 / **0** |

N=4 那一格的第四路是 `chat3_turns`，与 `y_turns` 同题，**hit 1.000 / stall 0 / 0 MB**——
它不是第四个工作集，是同一个工作集的第二份，所以那两行的合计都被它抬高了，
**不要拿 N=4 的 5.652 去和 N=2 的 5.199 比**。可比的是每一行内部的 serial→pipeline。

**加流不会继续买到东西**：N 从 2 到 3，第一路的 hit 从 0.861 掉到 0.830、
MB/token 从 626 涨到 768（**+23%**），合计 tok/s 反而从 5.199 掉到 4.807。
**5,100 槽装不下三个工作集**，这和 `p4_hitrate.md` §2.4 的容量曲线是同一件事。


---

## 6. 闸

| 闸 | 结果 |
|---|---|
| **单流不许退**（`tools/hitrate_bench.py --script y_turns --cache-slots 5100`，ABAB 三对，基线是**当场从 `5bafc98` 重新编出来的**二进制） | base `5.0808 / 5.0377 / 5.1230` → 均值 **5.0805**；本分支 `5.0962 / 5.0701 / 5.0712` → 均值 **5.0792**。**−0.03%**；hit 两边都是 **0.9111**、stall 98.1–98.4 ms、gpu 80–83 ms——**远在 ±3% 的地板之内** |
| `l3_ppl` 的 `off` 臂（单流） | **NLL 0.630051 / PPL 1.8777 / 61 of 64 top-1**——与 main 的**同一个数**，逐位复现 |
| `suite.multistream`（新增，三例） | 两条流一起跑时，每条流的 token id、`top1`/`top2` 的**原始 bit pattern**、以及 40 层 window ring 的**原始字节**，与它单独跑时完全相同。三种调度（`pipeline` / `interleave` / `pingpong`）各一例 |
| `ctest -LE needs-model` | **25/25** |
| `suite.decode` / `suite.kv_replay` / `suite.spec_forward` | 见 §6.2 |
| device-lost | 主表 9 个 cell + N 扫描 6 个 cell + 单流 6 个 cell，**0 次** |

### 6.1 一个必须说清楚的坑：基线二进制是陈的

第一次跑「单流不许退」用的是主工作树里现成的 `build/deepmoe.exe`，工作树是干净的
`5bafc98`——但那个 **exe 的时间戳早于 Track K1a 的 commit**。于是量出来本分支
「单流快 5.8%」，分项是 `moe_gpu` **42.94 → 39.31**——**正好是 K1a 报的
37.87 → 34.33（−9.3%）那一项**，和本轮一点关系都没有。把 `5bafc98` 重新编一遍
（`git worktree add --detach`，用完删掉）之后差值回到 **−0.03%**。
**规矩：跨分支的 A/B，两臂都要当场编。**

### 6.2 其余套件

`ctest -R "suite.decode$|suite.kv_replay|suite.multistream|suite.spec_forward"`：
**4 个里 3 个 Passed，1 个 Skipped**（`suite.spec_forward` 要的导出这台机器上没有，
与本轮无关）。`suite.kv_replay` 918 s 全过，是最重要的一条——
KV rollback / parking / window replay 全部在 `Stream` 里换了一遍地址，它一个都没碎。

---

## 7. 判决

**D1（`pipeline`）留下并默认开，但它不是为它写的那张支票。**

- **要什么有什么的部分**：正确性是硬的（逐位、含 ring 字节）；单流 **−0.03%**；
  接口是加法（`--streams` 不给就是今天的引擎）；内存 **~50 MB/流**。
- **没兑现的部分**：**+17.5% 而不是 1.6–1.8×**。
  **原因不是实现而是这台机器的账**：
  ① 两个工作集抢一个 5,100 槽的 LRU → MB/token **+13.5%**（事前点名的风险，量级也对）；
  ② 盘的聚合速率只到突发天花板的 **69%**，因为每次 `wait_layer` 之前
  只压得进一条 attention 链（~2.1 ms）对着 ~3.4 ms 的 stall；
  ③ 把 ② 修掉的唯一办法是更多条流，而 ④ **N≥3 因为 ① 直接变成负的**。
- **所以它和 §7 第 1 项是同一件事的两面**：`tok/s ≈ NVMe_eff / (MB per token)`。
  多路把 `NVMe_eff` 抬了 28%，但同时把 `MB per token` 也抬了；
  **要让多路真正变成 1.6×，得先有第二块盘或者更大的 cache**——
  那时 ① 消失、③ 可行，②也就跟着开了。
- **延迟这一侧要诚实**：每路 **0.62×**。这是给「两个人同时在用」优化的，
  不是给「一个人等回答」优化的。单人场景应该继续走单流，而它一个字没变。
