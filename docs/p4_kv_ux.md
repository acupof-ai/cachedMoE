# Track R2 / F2：KV 账、回退、持久化与 serve UX（P4）

状态：**完成**（F2 收尾）。per-source KV 平面、engram 常量、session 回退/挂起、serve 中断、
SSD prefix KV（design §11.4）都已实现并实测；R2 留下的 `restore_context` 超时已定位并修掉；
KV 记账从"设计估算"换成"真实分配的字节"；KvStore 不再是**一块**分配。

本文件记录：已验证事实、每个数字是怎么测的、复验命令。

---

## 1. 已实现（R2 + F2）

- `runtime/kvstore.*`：KvStore 平面只开在实际写 KV 的 source 层上，按上下文增长；
  **多 slab 分配**（§3）；`clear()` 只用整块 memset/memcpy（§3.3）；
  FP4 pack/unpack 与非 SWA 行 backup/restore 逐位精确。
- `runtime/engram_tables.*` + `tools/gen_engram_norm.py`：engram hash 常量从 tokenizer.json /
  config.json 推导，对 L3 导出的 129,280 个 token-map 项一致。
- `runtime/session.*`、`cli/serve.cpp`、`tools/chat.py`：rollback、park/restore、
  SSD `.pkv`、serve 中断/命名会话/LRU 挂起。
- `tests/test_kv_replay.cpp`、`tests/test_kvstore.cpp`、`tests/test_engram_tables.cpp`、
  `tools/serve_demo.py`。

---

## 2. 修复：`restore_context` 超时（R2 的"已知 bug"）

**症状**（R2 记录）：restore 导出状态，replay [0,64) 跑了 102 s 之后
`runtime::restore_context(...) -> cancelled: timeline wait for 5579 timed out`。

**根因不是 replay 逻辑，是那条等待的期限本身。**
`Engine::cmd_wait()`（design §7.1：每层一次 host 等待）用的是**固定 120 s** 的
`fence_.wait(...)`。在信号量这一端，"提交被别的进程排在队列后面"和"设备卡死"长得一模一样；
而这台机器上同时跑四条 track 时，一次提交排队超过 120 s 是常态。于是：replay 没有卡住，
它只是在排队，却被当成失败整体丢弃——64 步里前 63 步的结果也一起没了。

**修法**（`runtime/engine.cpp` `cmd_wait()`，`runtime/engine.h` 一个静态声明）：

- 等待切成 15 s 一片，总预算 `DEEPMOE_GPU_WAIT_S`（默认 **900 s**）；
- 第一片超时就 `log_warn` 一次，慢机器会说自己慢，而不是看上去挂死；
- 真放弃时报出 fence 目标值、token、这一步的 submit 数和实际等了多久，并明确
  "排队和卡死在这里无法区分，机器是共享的就把 `DEEPMOE_GPU_WAIT_S` 调大"。

这样"安静机器才能跑绿"变成"忙机器也能跑绿，只是慢"。

**这次实测**（2026-09-18，四条 track 同时抢 GPU）：`kv_replay.l3_64` 全绿，replay
[0,64) 用了 **139 s**（2,110 ms/token，安静机器上是 ~810 ms/token）；整个过程**没有一片
15 s 的等待超时**（一条 `log_warn` 都没有），所以这一次单次提交没有排到 120 s 以上——
`kv_replay.longctx`（4K）则直接抓到了证据——新加的那条警告真的响了：

```
[WRN] engine: still waiting for GPU fence 24705 after 15 s (token 4439, 9 submits this step)
      -- the queue is shared; giving it 900 s (DEEPMOE_GPU_WAIT_S)
```

一次提交在队列里等了 15 s 以上，而这一步**本身没有任何问题**：这条 track 随后
8/8 教师强制 + 8/8 自由运行全对。旧代码在这种机器上只是"还没到 120 s"而已。
现在只有"GPU 900 s 没有信号"才会放弃，而那已经不是排队能达到的量级。

> 附带一条：R2 那次还顺手修掉了 compressor 的 E4M3 scale/字节不一致
> （`gpu/shaders/compressor.slang`、`mgt1_cmp.slang` 用 `fp8_decode(fp8_encode_rn(...))`），
> 所以 `kv_replay` 场景 (4) 的 `raw_rows()` 是 **0**——打包路径是无损的，不是"大部分无损"。

---

## 3. KV 账：实测字节

### 3.1 实测表（`gpu.kvstore_measured_bytes_and_64k_slabs`）

每一行都是**真的 `KvStore::create()` 之后问分配器要到的字节**，不是公式：

| positions | 实际分配 | slab 数 | 最大 slab | 布局合计（bf16） | design §11.3 模型格式 | 倍数 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | **3.07 MB** | 1 | 3.1 MB | 3.07 MB | 2.76 MB | 1.1× |
| 4,096 | **16.04 MB** | 1 | 16.0 MB | 16.04 MB | 6.37 MB | 2.5× |
| 17,010 | **57.37 MB** | 1 | 57.4 MB | 57.37 MB | 17.91 MB | 3.2× |
| 65,536 | **212.65 MB** | 1 | 212.7 MB | 212.65 MB | 61.29 MB | 3.5× |

（MB = 10⁶ B。"模型格式"= design §11.3 的 window 2,703,360 B + 894 B/token。
17,010 那一行就是引擎打印的 **54.96 MiB ≈ 57.4 MB**。）

拆开看 17,010：window 2.70 + 压缩 KV 43.55 + index key 10.89 + compressor 状态 0.13 +
top-k 列表 0.10 = 57.37 MB。对照 design §11.3 里 R2 之前的 **878 MB**（40 层每层一个平面、
按 ratio-1 最坏情况开满）：**少 15.3 倍**。

### 3.2 bf16 工作集：为什么还是 3.2×，上界是多少

差的这 3.2× 全在**压缩 KV 与 index key 以 bf16 存活**：`sparse_attn.slang` 读的是 bf16 平面、
`indexer.slang` 打分读的是 bf16 key，选行的 top-k 是同一个 command buffer 里在 GPU 上算的，
**没有一个 host 侧的点可以把打包后的行收拢成有界工作集**（`runtime/kvstore.h` 头注释、
design §11.3 的"两处更正"第 2 条）。所以：

- **live（bf16）每 token**：压缩 2.5 行 × 1,024 B + key 2.5 行 × 256 B = **3,200 B**；
- **模型格式每 token**：720 + 170 + 4 = **894 B**；比值 3.58×，加上与 C 无关的 2.70 MB
  window 之后在 4K 是 2.5×、64K 是 3.5×，**随 C 单调趋近 3.58×，这就是上界**。
- **不在 GPU 上的每一份副本都是打包的**：挂起的会话、回退保存的行、`.pkv` 文件，
  走 `KvStore::pack` / `unpack`，逐位精确（bf16 的值本来就在 FP4 网格上，
  恢复的是 (nibble, scale) 这一对，不是重新量化）。64K 的打包态是 **58.35 MB**。

把 live 也换成 nibble 平面要改 `sparse_attn` / `indexer` 的读法，代价未测——**仍是待办**，
但它现在是一个 3.2× 的问题，不是一个 49× 的问题。

### 3.3 `clear()` 只用整块搬运（design §7.1 rule 10）

`clear()` 里 `score_state` 要填 `-inf`。`-inf` 不是一个字节模式，原来是**逐元素 for 循环**写
GPU 可见内存：4 个 owner × max_ratio 8 × 512 = 16,384 次未缓存写 × ~230 ns ≈ **3.8 ms**，
而且每次 context reset / restore / 回退重置都付一遍。现在 `-inf` 只在**普通 host 内存**里
构造一次（`ninf_`），然后一次 `memcpy` 进去。`runtime/` 里再没有标量循环碰 GPU 指针。

### 3.4 多 slab：2 GiB 上限不再是 `max_context` 上限

design §11.3 写过：`KvStore` 是**一块** host 可写分配，受 2 GiB `maxMemoryAllocationSize`
约束，所以 `max_context ≲ 41.7K`。现在 KvStore 的各个区域排在**一个扁平地址空间**里，
按 `KvStoreConfig::slab_bytes`（默认 2 GiB）切成若干 slab，**每个区域整块落在一个 slab 内**，
所以"平面地址 + 行偏移"仍然是一个合法地址；驱动拒绝这个尺寸时 `create()` 自动折半重排。
`reserve()` 增长时按区域逐块 memcpy（两个 layout 的切法可能不同）。

实测：64K 只要 212.65 MB，**一个 slab 就够**，2 GiB 的墙在 per-source 平面下要到
~600K positions 才会碰到。为了真的走到切分路径，测试把上限压到 80 MiB
（刚好高于最大的单个区域——层 20 的 ratio-1 压缩平面 65,536 × 1,024 B = 67.1 MB，
单个区域不会被切开）：

```
    forced 80 MiB cap at 65536 positions: 4 slabs, 212.65 MB, largest 83.89 MB
```

然后在 64K 上用合成的量化行把四个 source 平面**从头写到尾**，检查最后一行（最后一个 slab 的
末端）逐位正确，再 `pack` → `clear` → `unpack` round-trip 逐位相同。

---

## 4. SWA 从不落盘：验证

design §11.2 的原则：跨过"当前这一步"要保存的状态**只有四样**——token ids、压缩 KV、
index key、compressor 携带状态；**window 环不存**，恢复时用 ≤ 128 token 的 bounded replay 重建。

代码层面这是结构性的：`ParkedContext` = `tokens` + `KvPacked`，而 `KvPacked` / `KvPackedPlane`
里**没有 window 字段**。测试层面：

`kv_replay.l3_64`（L3 导出，64 token 上下文）与 `kv_replay.longctx`（Track M 的 4K 导出），
2026-09-18，**四条 track 同时抢 GPU** 的情况下跑：

**L3 64（全绿）**

| 场景 | 结果 |
|---|---|
| (1) 从导出状态连续 decode | **8/8** 教师强制 |
| (2) 回退 72→68 / 72→66（窗口内） | 0 个重放位置；重解码 4/4、6/6 条 logits（top1/top2）**逐位相同** |
| (3) restore 导出状态 | replay [0,64) **139.0 s**；环 vs 参考实现 mean cos **0.9688**（最差层 L37 0.9328，最差 slot 0.6505）；随后 **8/8 教师强制 + 8/8 自由运行** |
| (4) park + restore 我们自己的 slow prefill | 打包 **0.082 MB**、**0 raw rows**；replay [0,64) 135.1 s；**环逐位相同、压缩行 + index key + carry 逐位相同**；随后 8 步 **8/8 逐位相同** |
| (5) Session：回退 / 取消 / 命名会话 | 分叉于 61 → reused 60、dropped 9、prefilled 3；取消后 finish=`cancel`、context 一致；切到 'other' 再切回，'default' 重放 64 个位置后 history 完全一致 |
| (6) drop_thinking 回退 | 见 §6 |

场景 (4) 是这条原则最强的证据：**只存四样 + replay**，重建出来的环和参考的环**逐位相同**，
接下来的 8 步 logits 也逐位相同。场景 (3) 是从**参考实现的**状态恢复，环是近似的
（mean cos 0.969），但 8/8 教师强制 + 8/8 自由运行都对——近似发生在 replay 段开头窗口不完整
的地方，沿 replay 衰减（design §11.2）。

**4K（Track M 的 ctx4k 导出，N = 4,133，全绿）**

| 场景 | 结果 |
|---|---|
| (1) 连续 decode | **8/8** 自由运行（margin 10.90 … 0.98） |
| (2) park 4,133 个 token | 打包 **3.72 MB**、**0 raw rows**（同一状态的 live bf16 平面是 **13.23 MB**，3.6×）；replay [4006, 4133) = **127 步 / 379.1 s**（2,984.8 ms/token；行备份 535.6 KB、5.8 ms）；环 vs 参考 mean cos **0.876**（最差层 L18 0.733）；随后 **8/8 教师强制 + 8/8 自由运行** |
| (3) 回退 4141→4134 | replay [4006, 4013) = **7 步 / 24.0 s**；重解码 7 个位置 **7/7** 与回退前相同 |
| (4) 回退 4133→4000 | replay [3872, 4000) = **128 步**（满窗）/ 486.0 s；再喂回 133 个 prompt token（458.8 s）后，位置 4133 的环 vs 参考 mean cos **0.946**，下一个 token **77（参考 77）**，随后 **8/8 自由运行** |

注意 (2)：**只用 3.72 MB 的非 SWA 状态 + 127 步 replay**，就把一个 4,133-token 的上下文
恢复到"接下来 8 个 token 与参考完全一致"。环本身只有 cos 0.876（replay 段开头窗口不完整），
**但 8/8 仍然成立**——这正是 design §11.2 说的"误差沿 replay 衰减"。

**L3 64 / 4K / 17K 的 decode 没有回退**（`decode.` 与 `decode_longctx.`，同一次 sweep）：
64 token 教师强制 8/8 + 自由运行 8/8（从参考状态与从我们自己的 prefill 各一遍）；
ctx4k 8/8 + 8/8（salt token 7/7）；ctx16k 8/8 + 8/8（salt token 7/7），KV store **57.6 MB**。

**replay 代价（实测）**：L3 64 是 2,110 ms/token、4K 是 2,985 ms/token（**四到五个进程同时抢
GPU + NVMe**，expert cache 冷）；安静机器上 R2 记录的是 ~810 ms/token（restore ≈52 s / 64 步）；
热 cache 的 decode 步是 ~82 ms。
replay 的步数 = 真正丢掉的环位置数，不是固定 128：回退 d < 128 个 token 只 replay d 步
（场景 (3) 是 7 步），窗口内的回退（L3 场景 2）是 **0 步**。

---

## 5. SSD prefix KV：`.pkv` 文件格式

`runtime/session.h` 的 `KvDiskOptions` / `save_parked_context` / `load_parked_context` /
`drop_parked_context`；`SessionPool` 在 park、LRU 淘汰、干净退出时落盘，在新进程
`activate(name)` / `restore_active_from_disk()` 时回读。
文件 `<dir>/<session>.pkv`（原子写 `.tmp` → rename，目录级 LRU 上限 `--kv-max-gb`）。

**每个位置存了什么、以及为什么里面没有 window**——字段表就是全部内容：

| 偏移 | 字段 | 字节 |
|---|---|---:|
| 0 | magic `DMOEKV01` | 8 |
| 8 | 格式版本（当前 1） | 4 |
| 12 | `model_tag` 的 FNV-1a 指纹 | 8 |
| 20 | token 数 n，然后 n 个 u32 token id | 4 + 4n |
| — | `positions`（非 SWA 状态覆盖的位置数） | 4 |
| — | plane 数（= kv source 数，4） | 4 |
| — | 每 plane：`layer`、`ratio`、`rows`（3 × u32），然后 9 个 (u64 字节数 + 负载) 向量：<br>`cmp_fp4`（FP4 E2M1，256 B/行）、`cmp_scale`（E4M3/16，32 B/行）、<br>`key_fp4`（FP4，64 B/行）、`key_scale`（UE8M0/32，4 B/行）、<br>`raw_rows`/`raw_cmp`/`raw_key`（不在 FP4 网格上的行，实测为 **0 行**）、<br>`carry_kv`/`carry_score`（ratio > 1 的 [2][512] fp32 携带状态） | 12 + 72 + 负载 |
| 末尾 | trailer `KVEND001` | 8 |

**每个位置 = 压缩 KV 288 B（÷ratio）+ index key 68 B（÷ratio）+ id 4 B**，四个 source 合起来
就是 design §11.2 的 894 B/token；**没有任何 window 字段**。
`kvdisk.roundtrip` 现在断言文件的**精确字节数**等于上表逐项之和——多存一份 window 会多
40 × 128 × 528 B = 2.70 MB，断言必然挂，所以"window 绝不落盘"是被测出来的，不是被声明的。

指纹不符 / 版本不符 / 截断都按 miss 或 Corrupt 处理，回退到重 prefill。

### 5.1 跨进程 TTFT（SSD 命中 vs 冷 prefill）

{{SEC51}}

---

## 6. 回退：分叉 prompt 与 thinking 模式

规则（`runtime/session.h` 头注释）：prompt 与 KV 里的 history 求最长公共前缀 c，
回退点 k = c 向下取偶（且 ≤ |p| − 1）；压缩行、index key 与计数截到 k，window 环靠
replay 重建，然后喂 p[k..)。**k 取偶**是"半满的 ratio-2 组"的处理方式：偶数点上没有未完成的
组，携带状态是死状态（下一次 pooling 之前两个 slot 都会被重写），奇数分叉点的最后一个 token
再喂一遍即可——那一组永远不需要被重建。

**thinking 模式是这条路径的日常**：官方模板的 `drop_thinking=True` 会把更早轮次的 reasoning
从 prompt 里删掉，所以下一轮的 prompt **不是** KV 里 history 的延长，而是"公共前缀 + 截断"。
`tools/chat.py` 的默认值就是 `drop_thinking = True`（官方默认），`/drop` 可以切。

验证（`kv_replay.l3_64` 场景 (6)）：

```
(6) drop_thinking turn: history 69, prompt diverges at 45 -> reused 44, dropped 25,
                        replayed 0 in 0 ms, prefilled 3
    vs the same turn from an empty store: tokens identical,
                                          compressed rows + keys + carry bit-identical
```

- 分叉点 45 是**奇数**，落在一个 ratio-2 组的中间；回退点取 44。
- 与"同一个 prompt 从空 store 跑一遍"对比：**生成的 token 完全一致**，
  并且结束时 store 里的**压缩行、index key、compressor carry 逐位相同**。
  这就是"半满的 ratio-2 组 + compressor 状态精确"的判据——不是声明，是对照。
- 25 个 token 被丢掉（上一轮的 reasoning + 回复），只重喂 3 个。

---

## 7. serve：取消、命名会话、LRU 挂起

- **取消**：reader 线程独占 stdin，`{"op":"cancel"}` 立刻置 `cancel_upto` 并回
  `{"event":"cancel","upto":n}`；`Session::generate` 在**每个 token 之间**和**每个 prefill
  分块之间**轮询，停下来时 KV 状态与 `history()` 一致（finish = `cancel`）。
- **命名会话**：`{"op":"generate","session":"name"}` / `reset` / `drop` / `sessions`。
  一个会话在 KV 里活着，其余以 `ParkedContext`（打包的非 SWA 状态 + ids）挂起；
  expert cache、pinned set、planner 三者共享。
- **LRU**：超过 `--max-parked` 或 `--park-budget-mb` 时淘汰最久未用的挂起会话，
  淘汰前先落 `.pkv`，所以"淘汰"只丢内存、不丢会话。

### 7.1 三会话脚本 demo

`tools/serve_demo.py`（`--max-parked 1`，专用的空 `--kv-dir`）：一轮首次、一轮续写、
一轮分叉编辑、一次生成中取消、三个命名会话（第三个触发 LRU 淘汰）、再依次切回。

```
.venv\Scripts\python.exe tools\serve_demo.py --cache-gb 8
```

2026-09-18，**机器同时被另外三个 track 占着**（所以 tok/s 只有 0.3–0.45，不是这条路径的速度指标；
要看速度去 p4_hitrate.md）。`--max-parked 1` 意味着**三个会话里只有一个能挂在内存**，
第三个一出现就把最久未用的那个淘汰到 `.pkv`，再切回来时从盘上读回来：

| 步骤 | wall s | prompt | reused | dropped | 切换时 replay | prefill | gen | finish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 alice 首轮 | 47.82 | 8 | 0 | 0 | — | 8 | 12 | length |
| 2 alice 续写 | 38.05 | 23 | **19** | 0 | — | 4 | 12 | length |
| 3 alice 分叉编辑 | 43.74 | 19 | **12** | **22** | 0 步 | 7 | 12 | length |
| 4 alice 第 5 个 token 时取消 | 15.62 | 31 | 30 | 0 | — | 1 | 6 | **cancel** |
| 5 bob 新会话 | 45.08 | 5 | 0 | 0 | — | 5 | 12 | length |
| 6 carol 新会话（触发 LRU 淘汰） | 51.62 | 6 | 0 | 0 | — | 6 | 12 | length |
| 7 alice 从挂起回来（**从 SSD**） | 149.57 | 37 | **36** | 0 | **36 步** | 1 | 12 | length |
| 8 bob 从挂起回来（**从 SSD**） | 103.14 | 19 | **16** | 0 | **16 步** | 3 | 12 | length |
| 9 carol 从挂起回来（**从 SSD**） | 89.34 | 19 | **17** | 0 | **17 步** | 2 | 12 | length |

服务端日志把这条链路写全了：

```
session pool: dropped least recently used session 'alice'
session pool: loaded 'alice' from disk (36 tokens, 0.06 MB packed)
session pool: dropped least recently used session 'bob'
session pool: loaded 'bob' from disk (16 tokens, 0.04 MB packed)
```

`build/serve_demo_kv/` 里是 `alice.pkv` 67,864 B、`bob.pkv` 51,772 B、`carol.pkv` 51,772 B
——**这就是一个会话的全部持久状态**（ids + 打包的非 SWA 行 + carry），没有 window。

三件事因此是被演示过的，不是被声明的：

1. **取消**：步骤 4，`finish=cancel`，只生成了 6 个 token（请求的是 200），
   context 与 history 一致，下一轮（步骤 7）在它上面继续。
2. **回退**：步骤 3，prompt 在 12 处分叉 → reused 12、dropped 22、只重喂 7 个 token，
   窗口内所以 replay **0 步**。
3. **LRU 挂起 + SSD**：步骤 6 把 alice 淘汰出内存（`evicted` 从 0 变 1，最终 4），
   步骤 7/8/9 依次从盘上读回来，每次只用 **36 / 16 / 17 步**的 bounded replay 重建窗口，
   prefill 只喂新加的 1 / 3 / 2 个 token。

---

## 8. 复验命令

```powershell
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
$env:DEEPMOE_LONGCTX_DIR="C:\Users\Asus\code\deepmoe\traces\longctx"
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build

.\build\tests\deepmoe_tests.exe kvstore.        # 账（CPU）
.\build\tests\deepmoe_tests.exe kvdisk.         # .pkv 格式与精确字节数（CPU）
.\build\tests\deepmoe_tests.exe gpu.kvstore     # 实测分配 + 64K 多 slab（GPU，不要权重）
.\build\tests\deepmoe_tests.exe kv_replay.      # 回退 / park / restore / drop_thinking
.\build\tests\deepmoe_tests.exe decode.         # L3 8/8
.\build\tests\deepmoe_tests.exe decode_longctx. # 4K / 17K
.venv\Scripts\python.exe tools\serve_demo.py --cache-gb 16
```

机器被别的构建占着时把 `DEEPMOE_GPU_WAIT_S` 调大（默认 900 s）；
`kv_replay` 的 replay 步在 expert cache 冷的时候是 ~800 ms/token，热的时候 ~82 ms。

---

## 9. 遗留

1. **live 平面仍是 bf16**（§3.2）：3.2× 的开销，换成 nibble 平面要改
   `sparse_attn` / `indexer` 的读法，代价未测。
2. **R ∈ {128, 256} 的取舍**没测（design §15 未决问题 24）；现在一律 R = 128。
3. **前缀的前缀命中**：`.pkv` 是按会话名存的，design §11.4 说的"按 prompt 前缀 hash 分文件、
   前缀的前缀也能命中"还没做（字段本身都是按位置可截的，缺的是索引）。
