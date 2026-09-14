# 模块依赖与线程模型

本文描述 deepMoE 代码骨架的**模块边界、依赖 DAG 与线程模型**。设计意图见
[design.md](design.md)；本文只回答"哪个模块能 include 哪个模块""哪段代码跑在哪个线程上"。

状态：**v0.4（2026-09-15）**。v0.1 对应骨架提交；v0.2 跟进 design §5.1 v0.5 的直读改动
（新增 `store/shard_set.h`，`ExpertStore` 改为每 run 计数、每 expert 6 项指针表）；
**v0.3 跟进 P2 step 1：`store/pinned`（常驻权重集）、`gpu/vulkan/attn_kernels`（十八个 pipeline
与共享地址表）、`runtime/{rope,kvstore,decode_layer,moe_bridge}`（一整层 decoder），
以及由此多出来的第三个跨层接口反转（§1.3）。**
**v0.4 跟进 P2 step 2/3（Track G / Track I，[p2_decode.md](p2_decode.md)）：补上 G 加的
`runtime/{engine 的 token 循环,decode_state,engram}` 与 `gpu/vulkan/decode_kernels`；
command buffer 从"每层一个"改成"在 gate 处切开"（§2.1）；完成信号走第二条 timeline（§2.3）；
§7.4 的 compressor / indexer 的状态进了 `kvstore`，`shared_attn` 的路由进了 `Engine`（§1.1）；
Planner 的 LRU 时钟改为按访问计（§2.4）。**
实现进度见 design.md §15。

---

## 1. 模块依赖 DAG

依赖只允许从左向右（下层不知道上层存在）：

```
core ← model ← store ← storage ← gpu / cpu ← runtime ← cli
```

准确的边（`A → B` 表示 A include B）：

```
                    ┌──────────────────────────────────────┐
                    │ core/  (types, status, align, bytes, │
                    │        json, log, profiler, config)  │
                    └───▲────▲────▲────▲─────▲────▲────────┘
                        │    │    │    │     │    │
          ┌─────────────┘    │    │    │     │    └──────────────┐
          │                  │    │    │     │                   │
     ┌────┴─────┐     ┌──────┴──┐ │ ┌──┴─────┴──┐         ┌──────────────┐
     │  model/  │     │ storage/│ │ │   cpu/    │         │  gpu/vulkan  │
     │ layout   │     │ file    │ │ │ dequant   │         │  device      │
     │ v41_confg│     │ backend │ │ │ act_quant │         │  memory ─────┼──┐
     │ manifest │     │io_engine│ │ │ gemv gate─┼──┐      │  timeline    │  │
     └────▲─────┘     └────▲────┘ │ └───────────┘  │      │  cmdbuf      │  │
          │                │      │                │      │  pipeline    │  │
          │  ┌─────────────┘      │                │      │  descriptor  │  │
          │  │                    │                │      │  moe_kernels │  │
     ┌────┴──┴───────────────┐    │                │      │  attn_kernels│  │
     │                       │    │                │      │decode_kernels│  │
     │        store/         │    │                │      └──────▲───────┘  │
     │ slab ← expert_store   │◄───┼────────────────┼─────────────┼──────────┘
     │ pinned                │    │   (slab.h 被 gpu/memory.h 引用：
     │ planner  predictor    │    │    SlabBacking 是两条路径的公共接口；
     │ engram_prefetch       │    │    pinned 走同一条路)
     │ shard_set             │    │
     └────────────▲──────────┘    │
                  │               │
          ┌───────┴───────────────┴──────────────────────────┐
          │                  runtime/                        │
          │ engine  block  attention  moe  engram            │
          │ dspark  sampler  kvcache  rope  kvstore          │
          │ decode_layer  moe_bridge  decode_state           │
          └───────────────────────▲──────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │  cli/   bench/   tests/    │
                    └────────────────────────────┘
```

### 1.1 每个模块的职责与它**不**做的事

| 模块 | 做 | 明确不做 |
|---|---|---|
| `core/` | 值类型、`Result<T>`、4 KiB 对齐、JSON 读取、日志、Profiler、运行时配置 | 不知道模型、不做 I/O、不碰 GPU |
| `model/` | `layout.h` 编译期常量；解析 `config.json` 与 `deepmoe_manifest.json`（v2：原始分片的 run/skew 地址簿，design §5.1） | 不读权重字节，不分配内存 |
| `storage/` | 平台无关的异步 I/O：`IoRequest` 优先级队列、切分、队列深度；`File` 抽象 | 不知道 expert 是什么（`IoRequest::key` 只是给 Profiler 归因用的标签），也不知道 manifest —— 4 KiB 对齐由 manifest 保证（design §5.1），I/O 层只搬对齐好的块 |
| `store/` | slab 池、`Free→Filling→Resident` 状态机（每 expert 的所有 run 到齐才 Resident）、`(host_ptr, dev_addr)`、每 expert 6 项的 GPU 指针表、`ShardSet`（48 个原始分片的打开与索引）；**`pinned`：design §2.2 的 17.7 GB 常驻集合，经 manifest + IoEngine 落进 GPU 可寻址内存**；Planner 的淘汰与取数策略 | 不做前向计算；ExpertStore 不选淘汰对象（那是 Planner 的事）。**`pinned` 也不解释权重的语义**——它只知道"哪些 tensor 要常驻、落在哪个地址"，形状与用法是 `gpu/vulkan/attn_kernels` 的事 |
| `cpu/` | FP4/FP8/E8M0 解码（L0 oracle）、**`act_quant_block`（fp8 E4M3 / UE8M0 的块量化，design §6 的复刻判据）**、标量与 AVX-512 GEMV、router 数学 | 不做调度、不做 I/O |
| `gpu/vulkan/` | device/queue、内存路径 A/B、timeline、command buffer、pipeline、descriptor；**`moe_kernels`（design §7.9 的两个 dispatch）与 `attn_kernels`（§7.2–§7.11 的十八个 pipeline + 共享地址表 + `GpuScratch`）** | ~~不知道层结构~~ —— **v0.3 更正**：`attn_kernels` 知道 stage 的顺序与每个 stage 的形状（`kStages`），因为 pipeline 特化与 `rows_per_lane` 的 cap 就长在那里。它仍然**不知道"第几层"**——层号是 push constant 与地址表的内容，不是代码里的分支 |
| `runtime/` | token 循环编排：block/attention/moe/engram/dspark/sampler/kvcache；**`rope`（按层的 RoPE/YaRN 参数）、`kvstore`（window 环与压缩 KV 的持有者）、`decode_layer`（一整层 decoder 的编排）、`moe_bridge`（gate ids → Planner → timeline signal）** | 不实现 I/O 策略，不直接 `ReadFile`；**不决定 kernel 的特化**（那在 `attn_kernels` 的 `kStages` 里） |
| `runtime/engine`（v0.4） | **token 循环本身**：embedding、四十层、engram、collapse + head + argmax；**`shared_attn` 的路由**（`build_ced_plan`：一个复用层读哪个 source 的压缩 KV 平面、哪个 source 的 top-k、以及——**另算**——最后一次发布的是谁的 index key）；**在 gate 处切开的 command buffer**、第二条 timeline 做完成栅栏、每 token 读一次 GPU 时间戳；`slow_prefill`（把 prompt 一个 token 一个 token 走 decode 路径）；`status()` 说清楚哪些输入是算出来的、哪些是交进来的 | 不做 chunked prefill（design §11）；不选 kernel 特化；不在 GPU 可见指针上跑标量循环（design §7.1 rule 10） |
| `runtime/kvstore`（v0.4 扩充） | window 环；**§7.4 的压缩 KV 平面、index-key cache、compressor 跨步携带的 `kv_state`/`score_state`（`score_state` 初值 −inf）**；`set_decode_topk`（model.py 的 `get_window_topk_idxs`，写 top-k 列表的 window 半截） | **按层分配、不按 source 分配**：只有四层会写压缩平面，但"第 l 层的地址"保持为 l 的纯函数，谁读谁的平面由 `Engine` 决定 |
| `runtime/decode_state` | L3 导出的读入器：prefill 记录（window KV + 四个 kv source 的压缩 KV cache / index key / compressor 状态）与每步的参考 logits、压缩 KV、top-k、gate 捕获。**有了 §7.4 的 kernel 之后只在 prompt 的状态上被使用**，每步的张量只拿来比对 | 不在 decode 路径上"代替"任何 kernel（`Engine::set_produce_ced(false)` 保留成对照开关） |
| `runtime/engram` | design §7.10：哈希（用导出的常量表，行号在 runtime 里算）、48 次 4 KiB 读、两个 dispatch；**每个 engram 层一对行平面**，所以两层的行可以在 token 一开始就取（design §9.5），`fetch` 与 `record` 分开 | 不推导哈希常量（那是 tokenizer 的函数，导出一次） |
| `runtime/moe_bridge`（v0.4 重写） | **一个 runner、七个槽**：6 个 FP4 routed + 1 个 FP8 shared（`Fp8Slots`，shared 占表里的第 385 项）；`HQuant=3`；主机侧只剩 x 的一次 memcpy、向量化 act_quant（启动时与 `cpu::act_quant_block` 自检逐位相同）和七行地址；`record` 进调用方的 buffer，`y` 留在 GPU 上由下一层的 hc_post 按地址读 | 不碰 `MoeRunner` 的 push constant 布局以外的东西；不做 host 侧的 fp32 加法 |
| `gpu/vulkan/decode_kernels` | engram 的两个 stage 与 head 的 greedy argmax（design §7.10、§7.11）；与 `attn_kernels` 同一个"共享地址表"办法 | 不含采样的 Philox 半截（§7.11） |
| `cli/` `bench/` | 入口与测量 | 不含可复用逻辑 |

### 1.2 平台代码的边界

**只有** `storage/windows/` 与 `storage/linux/` 允许 `#if defined(_WIN32)`。

- `storage/windows/`：`file_win.cpp`（`CreateFileW` + `FILE_FLAG_NO_BUFFERING|OVERLAPPED`）、
  `iocp.cpp`（IOCP 后端）、`directstorage.cpp`（可选，缺 `dstorage.h` 时报 `Unavailable`）。
- `storage/linux/`：`file_posix.cpp`（`O_DIRECT`）、`io_uring.cpp`（裸 io_uring 系统调用）。
  Linux 不是目标平台；它存在是为了让可移植的一半能在 CI 上编译并跑单元测试（design §14）。

CMake 按 `if(WIN32)` 选择源文件列表（`cmake/deepmoe_options.cmake`）。其余所有模块两边都编译。

### 1.3 三个跨层的"接口反转"

1. **`gpu/vulkan/memory.h` 实现 `store::SlabBacking`。** store 在依赖图上位于 gpu 之下，
   却定义了 gpu 要实现的接口。这是有意的：design §3.3 要求 ExpertStore 对路径 A / 路径 B
   完全无感，只看到 `(host_ptr, device_address)`。`HostSlabBacking` 是同一接口的纯主机实现，
   测试、CPU oracle 与 `bench/` 用它。
   **`store/pinned` 走同一条路**：它要的也只是"一块 host 可写、device 可寻址的内存"。
2. **`gpu/shaders/*.slang` 不是 C++ 依赖。** kernel 通过
   `buffer_device_address` 读 ExpertStore 的指针表（design §5.3），而不是 per-expert
   descriptor；C++ 侧只传一个 buffer。
3. **`gpu/vulkan/attn_kernels` 的共享地址表，是同一个办法的第二个实例**（v0.3 新增）。
   pinned 集合是 17.7 GB、散在许多小于 2 GiB 的区域里，"每 tensor 每层一个 descriptor"
   就是每 token 重写几千个 descriptor。所以**每个 kernel 只绑一个 descriptor：
   一张共享 `uint64_t` 地址表的 32 槽切片**；`store/pinned` 只往表里写地址，
   `attn_kernels` 只从表里读，两边都不认识对方的数据结构（design §7.15.2）。

   **这条反转今天有一个代价**：一个 stage 拥有表里的一个 slice，
   所以**同一个 command buffer 里两次同 stage 的 dispatch 会都看到最后写进去的那份地址**。
   后果有两个：(a) 一层里三次 mega_mhc 必须是三个不同的 stage（design §7.2）；
   (b) **command buffer 只能每层录一个，不能每 token 录一个**——后者需要地址表
   在 shader 里按层索引，那是 P3 的改动（design §7.1 rule 1、§7.15.3）。
   **v0.4 的补充**：这条约束决定了 token 循环里 buffer 能在哪里切。一个 buffer 里是
   "第 L−1 层的 MoE + 第 L 层的 attention"，所以绑第 L 层时**会覆盖** `MhcClose` 的切片——
   而 engram 层要在同一个 buffer 里先用 L−1 的权重做一次 `MhcClose`。
   `LayerStep::bind_close = false` 让 engram 层的 `bind` 跳过那一片。
   **`bench/attn_bench` 也被这一条约束**：它必须每层一个 `AttnRunner`，
   否则一个 command buffer 里每次迭代读的都是同一层的权重，
   小于 32 MB MALL 的东西就会报出幻觉（第一版把 `wq_a` 报成 374 GB/s）。

---

## 2. 线程模型

design §7.1 与 §9.6 决定了线程数：Vulkan 队列提交不是线程安全的，所以**提交只在一个线程**；
I/O 策略需要一个独立线程才能在 GPU 计算时推进；IOCP 需要自己的完成线程。

| 线程 | 数量 | 归属 | 做什么 | 不能做什么 |
|---|---|---|---|---|
| **engine / submit** | 1 | `runtime::Engine` / `runtime::DecodeLayer` | 录制并提交 command buffer（**v0.4：在 gate 处切开，一个 token 41 次提交**，§2.1）；从 host-coherent 内存读 gate 的 ids；host-signal 驻留 timeline；在完成栅栏上等；token 一开始发出 engram 的 96 次读；采样 | 不能阻塞在 I/O 上——**例外**：P0 缺失今天仍是 `io_.drain()` 同步等，§7.9.3 的"只拆 dispatch A"还没接 |
| **io dispatcher** | 1 | `storage::IoEngine` | 优先级排队、请求切分、下发 chunk、收完成、调用完成回调 | 回调里不能做重活（见下） |
| **iocp completion** | 2（`IoConfig::completion_threads`） | `IocpBackend` | `GetQueuedCompletionStatus` 循环，把完成推给 dispatcher 的队列 | 不直接碰 ExpertStore |
| **planner** | 1（尚未启动） | `store::Planner` | token 间隙的 lookahead 预取与空闲回填（design §9.4、§9.6 P3） | — |

### 2.1 每 token 的时序（design §7.8）

**v0.4：下面这张图是设计意图，今天实际跑的是这样（`runtime/engine.cpp` 的 `run_layer`）：**

```
submit 线程                                             GPU
  embed → 打开 buffer
  L=0:  [attn 0 + §7.4] ─ submit ────────────────────────► 跑
        发出 engram 层 1 与 14 的 96 次读（与上面重叠）
        等完成栅栏 ◄──────────────────────────────────────── 完成，signal fence
        读 ids → Planner.plan_layer（每个 expert 一个 LRU tick）→ P0 读 → drain
        signal 驻留 timeline(token, 0)
        MoE stage（x memcpy、act_quant、七行地址）
  L=1:  [MoE 0][MhcClose 0][engram 1][attn 1]
          ─ submit，wait 驻留 timeline(token, 0) ───────────► 跑
        等栅栏 …… 同上
  ...
  尾:   [MoE 39][collapse][head][argmax] ─ submit ─────────► 跑；读 4 个字 + 时间戳
```

每层一次 submit、一次栅栏等待；**每个 buffer 的头一个 dispatch 就是被 gate 住的那个 MoE**，
所以 §7.1 的门是提交里的 semaphore wait，不是主机等完再重新提交。
驻留全命中时主机在提交前已经 signal 过，这个 wait 立即满足。

```
submit 线程                     io dispatcher              iocp 线程
   │
   ├─ 提交本 token 的 cmdbuf ───────────────────────────────────►  GPU 开跑
   │
   │  (每层) gate kernel 写 host-coherent ids[6] + layer_done++
   ├─ 自旋等 layer_done ◄─────────────────────────────────────────
   │
   ├─ Planner::plan_layer()
   │    ├─ ExpertStore::lookup() × 6        命中/缺失
   │    ├─ Planner::reclaim()               按 LRU 淘汰腾槽
   │    └─ IoEngine::submit() × (miss × run) ►│   每个 expert 2 个 run（§5.1）
   │                                         ├─ 切成 4 MiB chunk
   │                                         ├─ ReadFile(OVERLAPPED) ──►│
   │                                         │                          ├─ 完成包
   │                                         │◄─────────────────────────┘
   │                                         ├─ ExpertStore::finish_run()
   │                                         │   （在 dispatcher 线程上！全部 run
   │                                         │    到齐才 Filling→Resident）
   │◄─ 最后一个 fill 完成后 ──────────────────┘
   ├─ Timeline::signal(token_base + layer)  ──────────────────────►  GPU 继续跑 MoE
   │
   └─ 全层结束 → head → 采样 → Profiler::token_end() → JSONL 一行
```

**v0.3：中间那一段（读 gate 的 ids → `Planner` → host-signal timeline → 发 MoE）
就是 `runtime/moe_bridge`，已经实现并在 `tests/test_gpu_layer.cpp` 里跑通**——
那个测试**故意只给八个槽的 cache**，所以六个 expert 每层都真的从 NVMe 取回来，
这条时序是真跑的而不是走形式（design §7.15.3）。

### 2.2 回调契约

`IoEngine` 的完成回调**跑在 dispatcher 线程上**，而 dispatcher 线程同时负责给所有其他在途
请求下发 chunk。回调里做实事就会卡住整条 I/O 流水线。允许做的只有：

- `ExpertStore::finish_run()`（拿一次锁，计一个数；最后一个 run 到达时改几个指针）
- `Timeline::signal()`
- 通知一个条件变量

### 2.3 谁持有哪把锁

| 数据 | 保护方式 | 争用面 |
|---|---|---|
| `ExpertStore` 的 slot 表、free list、指针表 | 一把 `std::mutex`，只在指针记账期间持有，绝不跨 I/O | submit 线程 lookup / dispatcher 线程 finish_run / planner 线程 reclaim |
| `IoEngine` 的四条优先级队列与 `chunk_owner_` | 一把 `std::mutex` | 任意线程 submit / dispatcher |
| `IoEngine` / `Planner` 的统计 | 各自一把 mutex，与热路径分开 | 读取方是报告代码 |
| `Profiler` 的分段累加器与计数器 | relaxed 原子；token 边界是同步点 | 所有线程 |
| `Profiler` 的 JSONL sink | mutex，每 token 取一次 | — |
| Vulkan queue | **无锁——靠只有一个提交线程** | — |
| 两条 timeline | Vulkan 内部同步。**驻留 timeline 只由主机 signal，完成栅栏只由 GPU signal**——同一条 semaphore 不能两边都写，否则双方的值会互相越过（`gpu::Submission::signal_timeline`） | — |
| `CommandPool` | **无锁——Vulkan 规范要求外部同步，只有 submit 线程碰** | — |

### 2.4 淘汰安全（design §5.3）与 LRU 时钟

**v0.4**：`ExpertSlot::last_use_token` 装的不再是 token 号，而是 **Planner 的访问时钟**——
每次 lookup、每次 miss 的 admit 各走一格，按 gate 列出的顺序。这样才与
`tools/cache_sim.py` 的全局 LRU 一步一步相同（design §9.3）；按 token 打戳会让一个 token 的
240 次访问同龄，谁先走由槽号决定。`IoRequest::issue_token` 仍是 token 号。

**淘汰守卫今天仍然没有接上**（`set_guard` / `set_completed_timeline` 从未被调用）。
今天它是安全的：淘汰只在 `plan_layer` 里发生，而那时唯一在途的 buffer 已经跑完（栅栏刚等过）。
**一旦 §7.9.3 的"先到的先算"让 buffer 跨过一次 I/O 等待，这就不再成立。**

驻留槽被回收前必须确认没有已提交的 command buffer 还会读它：

```
Planner 决定淘汰 slot
  └─ ExpertStore::evict(slot)
       └─ 若 slot.guard_timeline > completed_timeline_ → FailedPrecondition
```

`completed_timeline_` 由 submit 线程用 `Timeline::value()` 推进。`ExpertStore::evictable()`
一开始就滤掉被 guard 住的槽，所以策略永远看不到它们。

### 2.5 一个 Win32 细节

一个文件 handle **一生只能关联一个 IOCP 端口**，且无法解除关联——关掉端口也不行。因此一个
`storage::File` 交给过一个 `IoEngine` 之后，就不能再交给另一个。`bench/nvme_bench.cpp` 每个
测量点重开文件，正是这个原因。此约束写在 `storage/backend.h` 的头注释里。

---

## 3. 错误处理

- 所有可失败的 API 返回 `Result<T> = std::expected<T, Status>`，异常不跨模块边界。
- `Err::Unimplemented` 是"设计里有、代码里还没写"的专用码，消息里带 `design §x.y`。
  `deepmoe run` 会把它和真正的失败区分开。
- OS 错误带 `Status::os_code`（`GetLastError()` / `errno`）。

## 4. 构建产物

| 目标 | 内容 |
|---|---|
| `deepmoe_core`（静态库） | core + model + cpu + storage + store + gpu + runtime |
| `deepmoe` | CLI：`info` / `bench nvme` / `run` |
| `nvme_bench` | design §9.2.1 Q6/Q7 的微基准（P-1 产物） |
| `bw_matrix` | design §8.0 / §3.3 的带宽矩阵（CPU / GPU / 并发 × 两条路径，全部已实测） |
| `kernel_bench` | design §7.9.1（`--p1`）/ §7.9.2（默认）的 MoE kernel sweep + dispatch 开销 + 路径 A/B |
| `attn_bench` | design §7.15.2 非 MoE decode 路径的逐 kernel 带宽。**每层一个 `AttnRunner`**（§1.3 第 3 条） |
| `heap_capacity` | design §5.2 / §9.2.2 两条路径的实际可分配上限 |
| `envcheck` | 环境自检，见 docs/build.md |
| `deepmoe_tests` | 单元测试；`ctest` 另按 suite 注册一遍。`suite.{integration,gpu_moe,gpu_attn,gpu_layer,decode}` 需要 `DEEPMOE_MODEL_DIR`（标签 `needs-model`），不给就自动跳过。**`suite.decode` 是 L3：四十层对参考，含 slow prefill，二十多分钟** |
| `deepmoe run` 的诊断开关（v0.4） | `--slow-prefill`（自己算 prompt 状态并逐层逐位置对参考）、`--loaded-ced`（对照：每步加载压缩 KV）、`--warm K`（热步地板）、`--determinism N`（逐层四个张量 + logits 的哈希）、`--gate-report`（逐层逐步把 gate 拆成"输入漂移 / kernel / 近似平局"三块） |
| `shaders` | `gpu/shaders/*.slang` → SPIR-V，每个都过 `spirv-val`（由 `add_slang_shader()` 每次编译都跑） |
