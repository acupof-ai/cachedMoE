# Track D2 — 第二块盘：同一份权重的第二个读源（2026-09-19）

一句话：**第二个读源落地了，路由按测出来的速率分流，聚合读 3.47 → 4.56 GB/s（+31%）。**
但这台机器上的第二块盘是 **USB 外置（1.0 GB/s）而不是第二块 NVMe**，
所以 STATUS §7 第 1 项里那句「stripe 到 9 GB/s，+32–40%」**在这里不成立**：
拿到的是 **+22% 的带宽**（4.6 + 1.0），按 §4 的式子折成 decode 是**一位数**。

出处：本文所有数字来自 `C:\Users\Asus\code\deepmoe-d2`（分支 `p4/d2-dual-source`），
同一个二进制、同一个 shader 目录，`--cache-slots 5100`，一次一个 `deepmoe serve`。

---

## 1. 硬件与拷贝

| | D:（主） | E:（镜像） |
|---|---|---|
| 设备 | WD SN740 NVMe（内置） | J.ZAO KP 2TB，USB 3.2 Gen2 UAS，新格式化 NTFS |
| 4 MiB 随机读 | 4.6 GB/s（STATUS §4 的天花板） | **1.0 GB/s**（QD 4–16） |
| 4 KiB 随机读 | 0.44 GB/s（本轮实测，QD 48） | 0.17 GB/s |

**拷贝**：`robocopy /E /J /MT:8`，整份 checkpoint（不只是带 expert 的 43 个 shard——
写速率够快，全拷比挑文件省事，而且镜像完整时 `open_mirror` 不用处理缺口）。

```
94 files, 475.3 GiB (510 GB)   529 s   = 964 MB/s
```

**校验**：94 个文件逐个比对字节长度，**0 个不符**；
`model-00003` / `model-00024` / `model-00046` / `deepmoe_manifest.json` / `config.json`
五个文件的 SHA-256 **全部 MATCH**。

`D:\models` 下**没有写入任何东西**（manifest 是唯一一次写过的文件，本轮没碰）。

---

## 2. 实现

镜像是**只读、可选、默认关**的。没给 `--mirror` 时 `submit()` 多一个 `bool` 判断，
其余逐字不变，`IoStats` 也不多一行。

**`storage/source_router.h`** —— 纯函数 `pick_source()`，**加权最小在飞字节**：

```
argmin_s  (outstanding_bytes[s] + bytes) / weight[s]
```

被最小化的是**时间**不是字节。两块盘速率差 4.6 : 1.0，所以：

* 单纯的 least-outstanding-bytes（不除权重）**会在 D: 一有队列时就把请求扔给 U 盘**；
* 加权之后，**D: 队列到 3.6 个 run（≈32 MiB）之前，排在 D: 后面仍然比去 E: 快**。

这个 **3.6 倍的交叉点**就是变异测试的靶子（§5）。
稳态的**字节比例**不是判据——不除权重的规则也会收敛到大致按速率分流
（慢盘排得更长，于是被选得更少），两条规则真正分得开的是**单次选择**。

**`store/shard_set.h`** —— `open_mirror(dir, manifest)`：按 manifest 的 `files` 顺序在
镜像根下开同名文件，**字节长度不等就跳过这一个 shard**（不是失败，是不镜像它）。
一个 shard 也开不出来才返回错误，而 `Engine` 把这个错误降级成 warning：
**镜像是优化，永远不是正确性输入**，盘拔了就回到单盘跑。

**`storage/io_engine.cpp`** —— `set_sources()` / `add_mirror()` 建表（启动时写一次，
热路径无锁读），`submit()` 在**校验之前**选源，于是对齐与 EOF 检查落在真正要读的那个句柄上。
**一个请求整体走一个源**：9 MiB 的一个 run 拆到两块盘上，延迟会变成两者的 max 而不是任一个。
只路由 **P0 与 P3**（`DEEPMOE_MIRROR_CLASSES` 可改）：
P1 是投机的、量小，P2 是 264 B 的 engram 行、延迟敏感——把它扔到慢盘上是净亏
（`p4_p0_queue.md` §3 的 (a) 已经为 P2 付过一次学费）。

**权重**从哪来：启动时对每个源的**同一个 shard** 做 1 秒、4 MiB、QD 8 的随机读探针
（`IoEngine::probe_source_gbps`，自己开句柄再关掉——Win32 句柄一辈子只能绑一个完成端口）。
`DEEPMOE_MIRROR_WEIGHTS=4.6;1.0` 跳过探针，A/B 需要每个 cell 权重一致时用它。

**接口**：`--mirror DIR`（`run` 与 `serve` 都收，可重复）或 `DEEPMOE_MODEL_MIRRORS=DIR;DIR`。
`status.json` 的 `io` 段每个源一行：

```
  src[0] D:\models\DeepSeek-V4.1-Flash  w 4.60 GB/s  N req  X GiB (78.6%)  mean lat ... ms  inflight ...
  src[1] E:\models\DeepSeek-V4.1-Flash  w 1.00 GB/s  M req  Y GiB (21.4%)  mean lat ... ms  inflight ...
```

---

## 3. 聚合读：盘这一侧拿到了什么

`bench/nvme_bench` 新增 `--mirror FILE`：给一个字节相同的副本，每个测量点就走
`IoEngine` 的路由，报出来的 GB/s 是**两块盘的聚合**。
同一个 shard（`model-00024`，6.8 GB），4 MiB 随机读：

| QD | D: 单盘 | E: 单盘 | D:+E: 聚合 | 分流 |
|---:|---:|---:|---:|---|
| 8  | 3.468 | 0.494 | **4.337** | D: 80.7% / E: 19.3% |
| 16 | 3.453 | 1.038 | **4.555** | D: 77.6% / E: 22.4% |
| 24 | 3.473 | — | **4.508** | D: 78.1% / E: 21.9% |
| 32 | 3.492 | — | **4.490** | D: 78.6% / E: 21.4% |

**+31%**（3.47 → 4.56），而且 **D: 的份额没有掉**（3.50–3.53 GB/s，和单盘的 3.47 一样），
即 E: 的 1.0 GB/s 是**净加上去的**，路由没有拖慢主盘。分流 78 : 22，
和权重 4.6 : 1.0 想要的 82 : 18 同一个量级。

> 注意这里 D: 单盘是 **3.47 GB/s** 而不是 STATUS §4 引用的 5.16——
> 那是在 C: 上的专用测试文件（§3 的 46 已经订正过参照系应该是 D: 的 4.60），
> 本轮量的是 D: 上真实 shard 的随机读，而且盘刚被 510 GB 的拷贝走过一遍。
> **预测 5.6 GB/s 的那个式子用的是 4.6 + 1.0；实际的基数是 3.5，所以聚合落在 4.5 而不是 5.6。**

---

## 4. ABAB：端到端 —— **没跑成，而且理由换了一个**

两次尝试，两种失败，**第二种是这一整条 track 最重要的结果**。

**第一次（11:00–12:30）：机器不是我的。** 另一条 track 的 `deepmoe serve` 从 10:22 起常驻
（pid 43564，101 GB private，90 分钟只涨 6 s CPU），Windows commit 只剩 30/172 GB，
第二个引擎起不来（`feedback-one-gpu-job`：一次只能一个）：

```
gpu init: resource-exhausted: pinned weights + expert cache needs 105.77 GB
but only 30.07 GB of Windows commit is available
```

**第二次（12:30–13:00，机器空了之后）：E: 自己掉了。** 三个 cell，三种症状，一条线：

| # | 臂 | 结果 |
|---|---|---|
| 1 | off（纯 D:） | 正常跑完，144 s |
| 2 | on | `gpu init: io: pinned load of 'norm.weight': overlapped read failed` |
| 3 | on | **卡死在 `48 shards open` 之后**：12 分钟里只用掉 **2 秒 CPU**、8 条线程、73 MB——**全在等 I/O** |

之后这块盘就**没再回来**：`ls E:\models\...` 挂住、`nvme_bench` 对 E: 的 32 个 4 MiB 读挂住、
`Get-Process` / `Get-Counter`（要枚举磁盘）也挂住，
**`taskkill /F` 报成功但进程还在**——卡在不可中断的 I/O 等待里的进程杀不掉。

**结论：这块 USB 外置盘在 decode 的负载形状下不可用。**
不是带宽不够（§3 已经量到它净加 1.0 GB/s），是**它撑不住**：
先是 510 GB 的持续写，再是 48 个 `FILE_FLAG_NO_BUFFERING|OVERLAPPED` 句柄上的并发随机读，
UASP 桥接芯片掉出总线。要恢复得**物理拔插**（或重启）。

**所以默认值只能是关，而且理由比「没测到 3%」更硬**：
**一个会把整台机器拖进不可中断 I/O 等待的读源，不能进默认路径。**

代码这一侧没有发现问题：`nvme_bench --mirror` 在 E: 还活着的时候
（§3，拷贝完成后约一小时内）跑满了四个 QD 点，路由、分流、per-source 计数全对；
`io.` 单元套件 11/11；CPU `ctest` 34/34。
第 2 个 cell 那条 `overlapped read failed` 现在会带上 Win32 错误码、长度和偏移
（`storage/windows/iocp.cpp`），下一次就不用猜了。

### 4.1 预测（留在这里等着被打脸）

按 §4 的式子 `tok/s ≈ NVMe_eff / (MB per token)`，用**本轮实测**的聚合而不是任务书里的 5.6：

| | 单盘 | 双源 | 比 |
|---|---:|---:|---:|
| 聚合随机读（4 MiB，QD 16） | 3.453 | 4.555 | **×1.319** |
| 引擎 busy 窗口（Q2 之后是天花板的 89%） | ~4.10 | ~5.4 | ×1.32 |
| `nvme_stall` | 105 ms | ~80 ms | −24% |
| decode（97 ms 计算 + stall） | 4.95 tok/s | ~5.65 | **+14%** |

**砍半之后 +7%。** 两个前提都可能不成立，而且方向相反：

* 只有 **P0/P3** 被路由，P1/P2 还在 D: 上——聚合的 +31% 不会整份落到 stall 上；
* decode 的形状是**突发**（一层 ~9.6 个 chunk，每层重新爬坡），而 §3 的 bench 是背靠背的稳态。
  Track Q2 在这个差别上已经吃过一次亏（预测 +13%，实测 +3.5%）。

所以**真正该预期的是个位数**，而 ±3% 的判据带就在旁边——**这一条很可能是 NO-GO**，
而它 NO-GO 的话，STATUS §7 第 1 项那句「第二块盘 +32–40%」就要改写成
「**第二块 NVMe** +32–40%；一块 1 GB/s 的 USB 盘不是它」。
### 4.2 还欠的

机器和盘都好了之后，一条命令：

```
.venv/Scripts/python.exe bench/d2_abab.py --out bench/results/d2/abab \n    --script bench/results/hitrate/y_turns.json \n    --script bench/results/hitrate/long_turns.json \n    --mirror "E:\models\DeepSeek-V4.1-Flash" --pairs 3 --cache-slots 5100
```

一个 cell 一个进程、off/on 交替，每个 cell 报 decode tok/s、decode-only `nvme_stall`、hit，
以及 `status.json` 里那两行 `src[i]` 的字节占比，最后打 A/B 表并写 `abab.json`。
**跑之前先确认 E: 是新插上的**，并且考虑 `DEEPMOE_MIRROR_PROBE_MS=0` +
`DEEPMOE_MIRROR_WEIGHTS=4.6;1.0` 把启动探针关掉——每个 cell 少两秒，权重也不受当时盘况影响。

---

## 5. 变异测试

`io.source_router_respects_weights` 的靶子是「把除以权重那一步删掉」：

```cpp
const double eta = (outstanding[s] + bytes) / w;   // 原
const double eta = (outstanding[s] + bytes);       // 变异
```

变异后 **2 个断言失败**（`{1.0, 0}` 与 `{3.0, 0}` 两个 case：D: 上排着 1 个和 3 个 run 时，
正确的规则仍然选 D:，变异后的规则选了 E:）。恢复后 `io.` 套件 11/11 通过。

`io.no_mirror_leaves_the_stats_untouched` 是另一侧的闸：不配置源时
`mirrors_enabled()` 为假、`stats().sources` 为空——**「默认关就是真的关」**。

---

## 6. 结论与默认值

**默认关。** `--mirror` 不给就是今天的单盘行为，逐字相同：
`submit()` 多一次 `bool` 判断，`IoStats::sources` 为空，`status.json` 不多一行
（`io.no_mirror_leaves_the_stats_untouched`）。

**立住的三件**：

1. **拷贝可行且便宜**：510 GB / 529 s / 964 MB/s，校验干净。
2. **路由是对的**：聚合 **+31%**，分流 78 : 22，**主盘的份额一点没掉**——
   慢盘的带宽是**净加上去**的，不是从快盘那儿挪过来的。
3. **变异测试抓得住**：去掉权重这一步，两个断言立刻红。

**倒过来的一件**：**这块 U 盘本身不能用**（§4）——在 decode 的负载形状下它掉出总线，
把一个 `deepmoe` 进程留在杀不掉的 I/O 等待里。带宽是真的，**可靠性不是**。
读路径的代码已经就位，它等的是**一块真 NVMe**。

**闸的状态**：CPU 全量 `ctest -LE "needs-model;needs-gpu"` **34/34 通过**（`suite.io` 11/11）。
`suite.decode` 8/8+8/8、`suite.integration`、`tools/l3_ppl.py`（NLL 0.630051）
**都要引擎，本轮没跑**——它们要验的是「打开 mirror 之后仍然逐位相同」，
而 mirror 现在没有一块能用的盘。默认（关）这一侧与 main 逐字相同，由
`io.no_mirror_leaves_the_stats_untouched` 和「`submit()` 只多一个 `bool`」两件事担保。
