# Track D2–D6 — 第二块盘：同一份权重的第二个读源（2026-09-19 / 20）

> **2026-09-20 晚，Track D6（§10）：“镜像睡着了”这个假说被证伪。**
> §9.5 那句“`src[1]` 的 mean lat 是 D: 的 5–6 倍（149 vs 26 ms）”是我们自己的口径：
> 它 **97% 是 P3 backfill 在引擎队列里等 P0 的时间**（两块盘的 backfill 都是 1.1–1.3 秒）。
> 把 **P0 单独量出来：E: 4.5 ms，D: 3.8 ms，差 1.19 倍**；一整个 cell 最慢的 P0 是 **19.84 ms**，
> 一次 1 秒的唤醒藏不进去。keep-alive（3,052 次 poke）**+0.15%**，
> 按带宽比开环压到 44% 分流 **−2.68%** —— **两个都 NO-GO，默认一个字不改**。
>
> **2026-09-20，Track D5（§9）：成了。** 同一块 SSD 换进 **USB4 盒子**（`BusType NVMe`，
> UAS 桥整层消失），十二个 cell **零错误**，`y_turns` **+9.4%**、`long_turns` **+4.63%**、
> 两条流 **+13.5%**，`l3_ppl` **NLL 0.630051 逐位不变** ⇒ **GO**。
> 挡在最后的不是盘而是**我们自己的启动探针**：它 1 秒的窗口正好是一块刚睡醒的 USB4 盒子的
> 第一个读，于是把 3.77 GB/s 的盘测成 **0.03**，路由只给它 0.0% 的字节。
> 修法是**热身之后再取基线**。下面的 §0–§8 是 D2/D3/D4 的原始记录，**一个字没改**——
> 它们对那个 UAS 盒子的判决仍然成立（§9.8）。

## 0. Track D2（2026-09-19）

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

---

## 7. Track D3：把 §4.2 欠的那条命令跑了（2026-09-19 下午）

一句话：**A/B 表不存在，因为 B 臂起不来。**
`--mirror` 那一侧在**引擎初始化时**就死了，连第一个 token 都没到：

```
gpu init: io: pinned load of 'norm.weight': overlapped read failed
          (win32 1117, 12288 B at off 1323827200)
```

**`win32 1117 = ERROR_IO_DEVICE`** —— 这正是 D2 在 `iocp.cpp` 里加的那个错误码要回答的问题
（§4「下一次就不用猜了」）。**答案是：不用猜了，是设备本身报的 I/O 错误。**
和 §4 第 2 个 cell 是**同一条错误、同一个 tensor（`norm.weight`）**，
只是这一次带着码。这是这块盘在**同一个位置**的第二次复现（跨两个会话）。

### 7.1 这一轮和 §4 不一样的地方

| | §4（上午） | §7（下午） |
|---|---|---|
| E: 插拔状态 | 拷完 510 GB 之后连续用 | **已物理重插**，本轮第一次加载 |
| E: 轻 I/O | `ls` 挂住 | **正常**（`ls` / 小读 RC=0） |
| E: `nvme_bench` | 挂住 | **正常跑完：0.962 GB/s**（4 MiB，QD 8，32 reads，rand），和 §1 的 1.0 一致 |
| 镜像完整性 | 94 文件校验干净 | 顶层 61 项与 D: **逐项相同**（`comm` 两侧皆空） |
| 引擎带 `--mirror` | 一次 `overlapped read failed`，一次卡死 | **`overlapped read failed`（win32 1117）** |
| 失败之后的 E: | 掉出总线，要拔插 | **活着**（轻 I/O 仍然正常） |

**所以带宽这一侧再一次确认是真的，可靠性这一侧再一次确认不是。**
盘能扛住 `nvme_bench` 的 32 个 4 MiB 随机读（QD 8），
扛不住引擎 pinned load 的形状——**在 `norm.weight` 这个偏移上、48 个句柄都开着的时候**。
这一次它没有把机器拖进不可中断等待（比 §4 好），但**结论一个字不变**。

### 7.2 off 臂：今天的单盘基线

B 臂起不来，A/B 没有意义，所以**没有跑满 12 个 cell**（`feedback-fast-experiments`：
判决已定的跑就该杀掉）。跑完的那个 off cell（`y_turns.json`，4 轮，252 个 decode step）：

| | 值 |
|---|---:|
| decode | **4.5693 tok/s** |
| `nvme_stall` | **113.5 ms/token** |
| hit | **0.8957** |
| `sources` | **空**（镜像关，`status.json` 不多一行——`io.no_mirror_leaves_the_stats_untouched` 的运行时确认） |

分项 ms/token：attn 46.5 / moe_gpu 43.2 / nvme_stall 113.5 / engram 5.2 / tail 6.2 / moe_host 1.2 / other 2.6。

### 7.3 预测 vs 实测

| | 预测（§4.1） | 实测 |
|---|---:|---|
| 聚合随机读 | ×1.319 | **没测**（本轮只测了 E: 单盘 0.962，与 §1 一致） |
| decode | +14% | **无法测量——B 臂不启动** |
| **砍半后** | **+7%** | **无法测量** |

**§4.1 那句「留在这里等着被打脸」没有被打脸，也没有被证实**——
它需要的那个数**这块盘给不出来**。打脸的是另一件事：§4.2 写「机器和盘都好了之后，一条命令」——
**盘「好了」是假的**：轻 I/O 恢复 ≠ 能跑引擎，`ls` 和 `nvme_bench` 都过了，
引擎还是死在同一个 tensor 上。**下一次别拿 `ls` 当健康检查。**

### 7.4 判据

任务书的判据是「两个脚本都 ≥3%、闸全过、所有 cell 零挂零错」。
**第一个 on cell 就是硬错误 ⇒ NO-GO，不用看 3%。**
**没有重试**：D2 已经记过一次「重试的代价是整台机器进不可中断 I/O 等待、`taskkill /F` 都杀不掉」，
而这一轮的下游是要把网页 UI 起回来——**拿用户的机器去换一个 n=3 不划算**。

**闸（`suite.decode` / `l3_ppl` / `suite.integration`）本轮没跑**，理由和 D2 §6 一样且更强：
它们要验的是「打开 mirror 之后仍然逐位相同」，而 **mirror 打不开**。
默认（关）这一侧与 main 逐字相同，担保没变。

### 7.5 顺带修掉的两个东西

**(a) `bench/d2_abab.py` 的 `cell_stats` 读错了 key。** serve 写的是 `{"event": "done"}`，
harness 找的是 `e.get("type") == "done"`——**于是每个 cell 都报 0.0000 tok/s**，
包括**跑完的那个**。§4 那次 off 臂「正常跑完 144 s」也正是这个原因没能留下一个数。
修好之后同一份 `events.jsonl` 立刻解析出 4.5693 tok/s / 113.5 ms / 0.8957。
**这个 bug 比盘更早挡在路上，而且它是静默的**——0.0000 看起来像「没跑」而不像「解析错了」。

**(b) `--serve-arg` 现在对称地加到两个臂上。** 原来只有 on 臂能加 serve 参数（`--mirror`），
于是想给**两个臂**都加一个开关（比如 `--no-kv-disk`）没有入口。

### 7.6 两件挡路的机器事实（都不是镜像的锅）

**(i) `--cache-slots 5100` 今天在这台机器上跑不起来。** 第一个 decode submit 就
`vkQueueSubmit2 failed (-2)`（`VK_ERROR_OUT_OF_DEVICE_MEMORY`），slab 布局是 **34 A + 17 B**。
**5000 槽（34 A + 16 B）干净跑完**四轮，4.08 / 4.60 / 4.82 / 4.89 tok/s。
STATUS §1 写的「安全上限 5,000 槽」是对的，而 **`auto` 今天选的 5,100 在它上面**——
本轮 A/B 因此整体降到 **5,000 槽，两臂对称**。

**(ii) `.pkv` 恢复失败是致命的，不是降级的。** path A 被 expert cache 占满之后，
KV 盘恢复拿不到显存：`serve: kv-disk restore failed: internal: vkQueueSubmit2 failed (-2)`，
而这个错误**杀掉了那一轮**（`tools/chat.py:186` 抛出去）。
镜像那一侧的规矩是「优化永远不是正确性输入，开不出来就降级成 warning」（§2）——
**KV 盘这一侧没有这条规矩**。它挡着 STATUS §7 第 2 项（SSD KV 前缀复用 41×）默认开。

### 7.7 结论

**§6 一个字不改，而且现在有第三次复现撑着它。**
`(b2) 这块 U 盘 —— 不要再试` 升级成：**不要再试，包括在它看起来已经好了的时候。**
`(b1) 第二块真 NVMe` 仍然是 §7 第 1 项唯一开着的大杠杆，读路径的代码仍然就位。

网页 UI 因此**以 mirror 关**重新启动 —— 和 main 的默认行为逐字相同。

---

## 8. Track D4：闸加上了，盘还是死在同一个 tensor 上（2026-09-19 晚）

一句话：**DX §5.1 的验收门 PASS 了，镜像侧的健康闸也加上了，A/B 表还是不存在——
第一个 `on` cell 依旧是 `win32 1117 / norm.weight`，这是跨三个会话的第四次复现。**

出处：本文所有数字来自 `C:\Users\Asus\code\deepmoe`（分支 `p4/d4-dual-ab`），
`bench/results/d4/`（on cell 的 `serve.log`、harness 日志、失败时段的系统事件）。
一次一个引擎；网页 UI 的三个 PID 在动工之前停掉，`Get-Process deepmoe*` 为空。

### 8.1 前提：这一次盘是「好的」，而且是按 DX 的门验的

用户把盒子换到另一个 USB 口，NTFS 用 `chkdsk /spotfix` 修过，丢掉的 2 个文件重拷；
`E:\models\DeepSeek-V4.1-Flash` 与 D: **逐文件相同**（48 shard + manifest，名字与长度）。

**DX §5.1 gate: PASS after port change** —— `dx_probe`，48 句柄 × QD 24：

| 臂 | 时长 | 结果 |
|---|---:|---|
| 纯 4 MiB 随机读 | 300 s | ✅ 1.041 GB/s，**零错误** |
| + 每 8 个插一个 12 KiB | 300 s | ✅ 1.040 GB/s，**零错误** |
| 48 个句柄开销 | — | ✅ **158 ms**（换口之前 **17 s**） |
| 本轮开跑前的复核（同一条命令，60 s） | 60 s | ✅ 1.033 GB/s，**句柄 0 ms**，零错误 |

**这不是「`ls` 过了」那种健康检查**——它正是 D2 §7.3 付学费之后写下的那道门。
**它还是不够**（§8.4）。

### 8.2 先落地 DX §5.2 建议的那条闸（三件，全做了）

| # | DX §5.2 的建议 | 落地 |
|---|---|---|
| ① | 启动时从镜像试读 pinned 权重 | `Engine::probe_mirror_health()`（`runtime/engine.cpp`）：48 个 shard 句柄**都开着**的时候，从镜像读 **8 个 pinned 尺寸的小 tensor**（≤256 KiB，按名字排序、每个 shard 至多一个，所以 A/B 的两个 cell 读同一批字节），用自己新开的句柄——Win32 句柄一辈子只能绑一个完成端口 |
| ② | 失败 → 摘镜像，降级成 warning | 探针失败就 `io_.drop_source(s)` + 一条 warning，引擎继续**单盘**跑。摘源发生在 `set_sources` **之后**，所以失败的镜像**仍然出现在 `status.json` 里**、标着 `DROPPED`，而不是凭空消失。`DEEPMOE_MIRROR_HEALTH=0` 关掉这道门 |
| ③ | 运行期连续 N 次 I/O 错误 → 动态摘源 | `storage/source_router.h` 新增 `SourceHealth`：**连续**（不是累计）`DEEPMOE_MIRROR_ERROR_BUDGET`（默认 **3**）次失败就把源从候选 mask 里摘掉、权重归零、`status.json` 行尾追加 `DROPPED after N errors`。**源 0 永不摘**——它是唯一保证存在的那份拷贝，把它的失败藏起来就是把硬错误变成静默的错读 |

**热路径上多出来的是一个 AND**（`mask = src_health_.live_mask(mask)`），
而且它整个在 `if (mirrors_on_ && ...)` 里面——**不给 `--mirror` 的那条路逐字不变**。

**变异测试**（`io.source_health_drops_a_mirror_that_keeps_failing`）：

```cpp
void note_success(uint32_t s) { consecutive_[s] = 0; }   // 原：连续
void note_success(uint32_t s) { (void)s; }               // 变异：累计
```

变异后 **5 个断言失败**（两次「错、错、成功、错、错 ⇒ 还不该摘」的 case）。
恢复后 `io.` 套件通过，CPU 全量 `ctest -LE "needs-model;needs-gpu"` **46/46**。

**靶子为什么是这一条**：掉出总线的盘，**第一次失败之后每一次都失败**；
偶尔报一个错的盘是**能用的盘**，而累计计数器早晚会在一条足够长的 run 上把它摘掉。
「连续」这两个字就是这条规则的全部内容，所以它就是唯一值得打的靶。

### 8.3 ABAB：**12 个 cell 跑了 2 个**

命令（`--cache-slots 0` = auto，D3 §7.6(i) 之后 auto 自己封顶到 5,000 槽；
`--serve-arg=--no-kv-disk` 对**两个臂**都加，免得 cell N 的 `.pkv` 漏进 cell N+1）：

```
.venv/Scripts/python.exe bench/d2_abab.py --out bench/results/d4/abab \
    --script bench/results/hitrate/y_turns.json \
    --script bench/results/hitrate/long_turns.json \
    --mirror "E:\models\DeepSeek-V4.1-Flash" --pairs 3 \
    --cache-slots 0 --serve-arg=--no-kv-disk
```

| cell | 臂 | 结果 |
|---|---|---|
| `y_turns_off_0` | off | ✅ **4.5607 tok/s**，`nvme_stall` **113.0 ms**，hit **0.8957**，114 s，`sources` 空 |
| `y_turns_on_0` | on | ❌ `gpu init: io: pinned load of 'norm.weight': overlapped read failed (**win32 1117**, 12288 B at off **1323827200**)` |

**off 臂和 D3 §7.2 对得上**（4.5693 / 113.5 / 0.8957），而且这一轮是 **auto 选的 5,000 槽**，
不是 `--cache-slots 5000`——H1 的自动封顶 + 探测在 A/B 的形状下也是对的。

**`norm.weight`、12288 B、偏移 1323827200 —— 和 D3 §7 逐字相同。**

### 8.4 闸做了它该做的，然后不够

`on` cell 的 `serve.log`，按发生顺序：

```
engine: 48 shards open, 475.25 GiB
engine: mirror 'E:\...' holds 48 of 48 shards, 475.25 GiB
engine: source 'D:\...' probes at 4.78 GB/s     <- 主盘探针
engine: source 'E:\...' probes at 1.03 GB/s     <- 镜像探针：盘是活的
IoEngine: 2 read sources ..., routing classes 0x9
engine: mirror 'E:\...' health probe: 8 pinned-sized reads, all served   <- ① 过了
...
IoEngine: source 1 'E:\...' dropped after 3 consecutive I/O errors
          (last: overlapped read failed (win32 433, 8192 B at off 8146944))   <- ③ 触发
gpu init: io: pinned load of 'norm.weight': overlapped read failed (win32 1117, ...)
```

**四件要读出来的事：**

1. **① 启动健康探针过了。** 8 个 pinned 尺寸的读，48 个句柄都开着，全部送到。
   **所以「开得出来 ≠ 能用」这条闸拦不住这块盘**——它在被探的那一刻确实是好的。
2. **③ 运行期摘源触发了，而且它读出来的码是 `win32 433 = ERROR_NO_SUCH_DEVICE`**，
   不是 1117。**盘不是读失败，是不在总线上了。** DX §4 把桥排第一，这是第五条证据。
3. **摘晚了一步。** `PinnedStore::load` 把**第一个失败的请求**直接抛出去
   （`store/pinned.cpp:183`），所以摘源和「引擎决定失败」是同一秒的两件事。
   **还差一条**：被摘掉的源上那个失败的请求应该**在主盘上重放**，而不是成为 `load()` 的返回值。
   **没有写**——判决已定，而且写了也没有盘能验它（`feedback-fast-experiments`）。
4. **失败之后 `Get-Process` 又挂住了**（超过 120 秒无返回）——要枚举磁盘的调用卡在
   不可中断 I/O 等待里，D2 §4 那个签名原样回来。

同一时刻的系统日志（`bench/results/d4/winevent_d4.txt`），19:51:30 起：

```
39 x disk      153   已在磁盘 1 ... 重试 IO 操作
 7 x UASPStor  129   发出了对设备 \Device\RaidPort4 的重置
```

**19:51:30 正是 `on` cell 起引擎的那一秒。**

### 8.5 判据与判决

任务书：**两个脚本都 ≥3% + 闸全过 + 零 E: 错误**。

| 条件 | 结果 |
|---|---|
| y_turns ≥3% | **无法测量**（B 臂不启动） |
| long_turns ≥3% | **没跑到** |
| 零 E: 错误 | ❌ **win32 1117 + win32 433 + 46 条系统盘事件** |

**⇒ NO-GO。** 而且任务书写明「E: 出 win32 1117 就停、记、不带 mirror 起网页、报告」——
**照做，没有重试。** D2 已经记过重试的代价（整台机器进不可中断 I/O 等待，
`taskkill /F` 都杀不掉），这一轮的下游是把用户的网页 UI 起回来。

**§4.1 那个预测（+14%，砍半 +7%）第三次没有拿到它要的数。**
这不是「预测错了」，是**这块盘三次都没让人测**。它要的仍然是 **STATUS §7 第 1 项的 (b1)：一块真 NVMe**
（或者 DX §4.1 那个还没试过的 USB4/雷电盒子——**它把 UAS 桥那一层整个拿掉**）。

### 8.6 闸（`suite.decode` / `l3_ppl` / `suite.integration`）

**跑了，mirror 关**——理由和 D2 §6 / D3 §7.4 一样：它们要验的是「打开 mirror 之后仍然逐位相同」，
而 **mirror 打不开**。但这一轮**动了默认路径上的一行**（`ready` 事件多了一个 `"sources"` 字段），
所以默认这一侧必须自己验一遍，不能靠「逐字相同」的担保。**跑了，全过：**

| 闸 | 结果 |
|---|---|
| `suite.decode` | ✅ **Passed**（141.5 s） |
| `tools/l3_ppl.py --modes off`（`traces/l3_64`） | ✅ **NLL 0.630051 / PPL 1.8777 / top-1 61/64** —— 与 §7 之前逐位相同 |
| `suite.integration` | ✅ **Passed**（1.5 s） |
| device lost | ✅ **0**（三个闸的全部输出里一次都没有） |
| CPU 全量 `ctest -LE "needs-model;needs-gpu"` | ✅ **46/46** |

**默认（关）这一侧仍然逐字不变的担保有三件**：
`io.no_mirror_leaves_the_stats_untouched`、
新加的那个 AND 整个在 `if (mirrors_on_)` 里面、
以及 `finish()` 里的健康计数整个在 `if (p->routed)` 里面。

### 8.7 顺带修掉的一个东西

**`tools/web/server.py` 根本没有 `--mirror` 透传。** D3（以及本轮任务书）都写着「透传存在」——
**不存在**，`Serve.__init__` 的命令行里没有这个分支。现在有了（可重复，默认不给），
同时 `ready` 事件与就绪横幅多出 **`read sources N`** ——
N 是**过了健康闸之后真正在读的源数**，不是「要来的」源数。
镜像判 NO-GO 之后网页 UI 仍然不带 `--mirror` 起，所以这条透传今天是**空转的**；
它值得留下的理由是：**下一块盘到货时，验它的那个人不必先发现文档说了谎。**

---

## 9. Track D5：换成 USB4 盒子之后，第二个读源**成了**（2026-09-20）

一句话：**盘这一次没有掉线，而挡在收益前面的最后一样东西是我们自己的启动探针——
它测的是盘的「醒过来」，不是盘。**

出处：`bench/results/d5/`（`abab2/` 是判据用的那 12 个 cell，`abab/` 是探针修好之前的那两个，
`probe_w/` 是用 `DEEPMOE_MIRROR_WEIGHTS` 手动压过权重的那一个）。
分支 `p4/d5-usb4`，主 checkout，一次一个引擎；动工之前网页 UI 已经停掉，`Get-Process deepmoe*` 为空。

### 9.1 硬件：UAS 桥整个消失了

同一块 J.ZAO KP 2TB SSD，换进 **USB4（40 Gbps）盒子**。这不是「换了个口」——
DX §4.1 预测的那件事发生了：**Windows 把它枚举成 `BusType NVMe`**
（`SCSI\DISK&VEN_NVME&PROD_J.ZAO_KP_SERIES`，走 PCIe 隧道），
**没有 `uaspstor.sys`，没有 RTL9210，没有 SCSI 翻译**。失效的那一层不在了。

| | D2/D3/D4 的 USB 3.2 盒子 | D5 的 USB4 盒子 |
|---|---|---|
| 枚举 | `USB\VID_0BDA&PID_9210`，UAS | **`BusType NVMe`**，PCIe 隧道 |
| 48 句柄开销 | 17 s（换口后 158 ms） | **4 ms** |
| `dx_probe` 48h × QD 24 | 1.04 GB/s（换口后，300 s 零错误） | **3.658 GB/s，60 s 零错误** |
| 引擎带 `--mirror` | **四次都死在 `norm.weight` / win32 1117** | ✅ **起来了**，四轮跑完，零错误 |

**`win32 1117` 和 `win32 433` 本轮一次都没有出现**，跨 12 个 cell、6 个带镜像的引擎进程。
D4 §8.4 那句「摘晚了一步」欠的那条「在主盘上重放失败请求」**仍然没写，而且现在也没有盘能验它**——
判决的方向反了，这条从「欠的」变成「用不上的」。

### 9.2 挡路的是探针：**0.03 GB/s vs 3.77**

第一轮 ABAB（`bench/results/d5/abab/`）跑出来的 `on` cell 是 **4.7935 tok/s**，
对 `off` 的 4.7258 是 **+1.4%**——在噪声里。原因在 `serve.log` 的一行：

```
engine: source 'D:\...' probes at 4.72 GB/s
engine: source 'E:\...' probes at 0.03 GB/s      <-- 盘实际是 3.77
```

于是加权最小在飞字节把 **99.9% 的字节留在 D:**，E: 只拿到 **775 个请求 / 0.1 GiB（0.0%）**，
而且那 775 个的 **mean lat 是 1315 ms**——因为权重低所以请求稀疏，请求稀疏所以盘一直在睡，
盘一直在睡所以每个请求都付一次唤醒。**这是个自锁的反馈环**，不是「权重调得不准」。

**0.03 这个数不是噪声，是一次测量的全部内容。** 三条独立的测量把它钉死：

| 测法 | E: |
|---|---|
| `dx_probe`，1 句柄，QD 8，25 s（本轮实测，逐 5 s 全平） | **3.772 GB/s**，零错误 |
| 引擎探针，1 s 窗口，**盘刚被用过（热）** | **3.74 GB/s** |
| 引擎探针，1 s 窗口，**盘空闲了一阵（冷）** | **0.03 GB/s** |

**一块空闲过的 USB4 NVMe 盒子，第一个读要一秒左右。**
探针的窗口正好是 1 秒，所以整个窗口就是那一个读。
`dx_probe` 从第一个 5 秒桶起就是 3.77，是因为它跑 25 秒，那一秒被摊掉了。

**修法是让探针先热身，而且热身的字节不算数**（`storage/io_engine.cpp`）：

```cpp
if (warmup_ms) std::this_thread::sleep_for(...);   // 同样的线程在读，只是不计
const uint64_t base = moved.load();                 // 基线取在热身之后
const TimePoint t0  = Clock::now();
```

默认 **1000 ms 热身 + 1000 ms 测量**（`DEEPMOE_MIRROR_PROBE_WARMUP_MS` 可改，0 = 回到旧行为），
每个源多一秒启动。修完之后，**同一台机器、同样冷的盘**：

```
engine: source 'D:\...' probes at 4.82 GB/s (4 MiB, QD 8, random, 1000 ms after a 1000 ms warmup)
engine: source 'E:\...' probes at 3.30 GB/s (...)
```

**0.03 → 3.30。** 权重 4.82 : 3.30 ≈ **59 : 41**，和任务书预测的 56 : 44 是同一个量级。

> 这条值得单独记一句：**D2 到 D4 三轮都在问「盘行不行」，而 D5 真正的答案是
> 「盘行了，然后我们自己的尺子不行」。** 上一块盘的带宽太低（1.0 GB/s），
> 1 秒窗口里它也能读出 ~1 GB，探针从来没有暴露过这个缺陷——
> **是 USB4 盒子的「睡得更深、醒得更慢」把它顶出来的。**

### 9.3 变异测试

`io.probe_does_not_count_its_warmup` 的靶子是基线那一行：

```cpp
const uint64_t base = moved.load();   // 原
const uint64_t base = 0;              // 变异：热身的字节也算进去
```

测试在一个本地临时文件上测两次——`warmup 0` 和 `warmup = 3 × 窗口`——本地文件没有唤醒，
所以两次必须是同一个稳态速率。变异之后热身那次把 ~4 个窗口的字节除以 1 个窗口的时间，
**落在 4× 上**，而断言的带是 2.5×。

### 9.4 ABAB：**判据用的那 12 个 cell**（`bench/results/d5/abab2/`）

命令（**没有任何权重覆盖**——这是默认路径本身，探针自己测出来的权重）：

```
.venv/Scripts/python.exe bench/d2_abab.py --out bench/results/d5/abab2 \
    --script bench/results/hitrate/y_turns.json \
    --script bench/results/hitrate/long_turns.json \
    --mirror "E:\models\DeepSeek-V4.1-Flash" --pairs 3 \
    --cache-slots 0 --serve-arg=--no-kv-disk
```

| 脚本 | 臂 | tok/s（3 cell 均值） | sd | Δ | `nvme_stall` | hit | E: 份额 |
|---|---|---:|---:|---:|---:|---:|---:|
| `y_turns` | off | 4.6547 | 0.1869 | — | 113.9 ms | 0.8957 | 0% |
| `y_turns` | **on** | **5.2368** | **0.0064** | **+12.50%** | **95.4 ms** | 0.8957 | **27.1%** |
| `long_turns` | off | 5.8764 | 0.0043 | — | 72.7 ms | 0.9356 | 0% |
| `long_turns` | **on** | **6.1488** | **0.0070** | **+4.63%** | **65.1 ms** | 0.9356 | **22.8%** |

逐 cell：
`y_turns` off **4.8027 / 4.3911 / 4.7703**、on **5.2449 / 5.2293 / 5.2361**；
`long_turns` off **5.8805 / 5.8783 / 5.8704**、on **6.1431 / 6.1446 / 6.1586**。
**两个脚本的 on 臂都和自己的 off 臂完全不交叠。**

> ⚠️ **`y_turns` 的 off 臂 sd 是 0.1869（4.0%），而这是我自己弄脏的。**
> `y_turns_off_1`（4.3911）跑的时候我在同一台机器上编译（`cmake --build`）。
> 另外两个 off cell 是 **4.8027 / 4.7703**（相差 0.3%），和 D3 的 4.5693、D4 的 4.5607
> 同一条线。**所以 +12.50% 里有一部分是那次污染压低了基线。**
> **诚实的数是拿 on 臂对那两个干净的 off cell：5.2368 / 4.7865 = +9.40%。**
> 下面所有结论用 **+9.4%**，不用 +12.5%。`long_turns` 的两臂都没被碰过（sd 0.07% / 0.11%），
> **+4.63% 原样成立**。

**零 E: 错误。** 12 个 cell、6 个带镜像的引擎进程，`serve.log` 里
`win32` / `dropped after` / `failed its health probe` **一条都没有**。

### 9.5 预测 vs 实测

任务书的预测：miss ~400 MB/token、stall ~100 ms，8.3 GB/s 聚合 + 56:44 分流
⇒ stall 55–60 ms、decode 5.1 → **~6.3 tok/s（+25%）**，砍半 **+12%**。

| | 预测 | 实测（`y_turns`） | 实测（`long_turns`） |
|---|---:|---:|---:|
| 分流 | 56 : 44 | **73 : 27** | **77 : 23** |
| `nvme_stall` | 100 → 55–60 ms | 113.9 → **95.4**（−16%） | 72.7 → **65.1**（−10%） |
| decode | +25%，砍半 +12% | **+9.4%** | **+4.63%** |

**砍半法则这一次是对的**（+12% 的下沿 vs 实测 +9.4%），**而分流没到 56:44**，
两件事是同一件：**E: 在引擎的负载形状下不是 3.3–3.8 GB/s。**
`status.json` 里 `src[1]` 的 **mean lat 是 D: 的 5–6 倍**
（`y_turns` 149 ms vs 26 ms，`long_turns` 51 ms vs 8.5 ms），
所以加权最小在飞字节**自己**就把它压到 23–27%——**路由是对的，它在如实反应一块更慢的盘**。
探针量的是 QD 8 的稳态；decode 给它的是**突发**（一层 ~9.6 个 chunk，每层重新爬坡），
而 USB4 隧道对突发的反应比内置 NVMe 差。**这正是 D2 §4.1 点名过的第二条打折理由**，
它这一次终于有数据了。

聚合读这一侧：`y_turns` 的 `io: eff` **4.11 → 5.03 GB/s（+22%）**。
**不是 8.3。** 8.3 是两块盘各自跑满的和；实际拿到的是 5.03，**因为只路由 P0/P3，
而且 decode 不是背靠背的稳态**。

### 9.6 两条流：盘确实是 Track MS 停在 +17.5% 的那堵墙

`bench/d5_ms_mirror_abab.py`（新增：`ms_abab.py` 轮转的是**调度**，这一个轮转的是**镜像**，
调度固定在 `pipeline`），4 轮 × 64 token，两条脚本，auto 槽，ABAB 两对：

| | off | on | |
|---|---:|---:|---:|
| 合计 tok/s | **5.4507**（sd 0.12%） | **6.1867**（sd 0.34%） | **+13.50%** |
| 每路 tok/s | 2.723 / 2.723 | **3.086 / 3.086** | ×1.13 |
| 每路 `nvme_stall` | 140 / 111 ms | **116 / 94 ms** | |
| 每路 MB/token | 606 / 453 | **606 / 453** | **一模一样** |
| 盘的聚合速率 | 4.16 GB/s | **5.29 GB/s** | **+27%** |
| 分流 | — | D: 71.0% / E: 29.0% | 零错误 |

**MB/token 一个字没变**（606/453 两臂相同），所以这 +13.5% **整个来自盘**，
不是来自 cache 行为的变化。`p4_multistream.md` §5.3 把 MS 只拿到 +17.5% 归因到两件事：
①两个工作集抢一个 LRU 让 MB/token 涨 13.5%，②盘的 2.83 GB/s 只到 D: 天花板的 69%。
**第二件现在部分松开了**：4.16 → 5.29 GB/s。①没动，它要的是更多的槽，不是更多的盘。

对照 MS 当初的主表（`serial` 4.6474 → `pipeline` 5.4602 = ×1.175）：
**off 臂 5.4507 和它对得上**（0.17%），所以这一轮是接在 MS 后面量的，不是重量了一遍 MS。

### 9.7 闸

全部**在镜像开着的情况下**跑（`DEEPMOE_MODEL_MIRRORS=E:\...`），
而且 `l3_ppl` 的 `off.txt` 里有 `IoEngine: 2 read sources ...` + `health probe: 8 pinned-sized reads, all served`
**证明镜像真的在读**——不是「设了环境变量但没生效」：

| 闸 | 结果 |
|---|---|
| `suite.decode` | ✅ **Passed**（106.8 s） |
| `suite.integration` | ✅ **Passed**（1.4 s） |
| `tools/l3_ppl.py --modes off`（`traces/l3_64`） | ✅ **NLL 0.630051 / PPL 1.8777 / top-1 61/64** —— 与 D4 §8.6、MS、K1a **逐位相同** |
| device lost | ✅ **0** |
| CPU 全量 `ctest -LE "needs-model;needs-gpu"` | ✅ **46/46**（`suite.io` 31 例，含新增的 `io.probe_does_not_count_its_warmup`） |

**`NLL 0.630051` 这个数是这一节最重要的一行**：
镜像把 27% 的 expert 字节换成从另一块盘读来的同样的字节，**输出逐位不变**。

### 9.8 判据与判决

任务书：**两个脚本都 ≥3% + 闸全过 + 零 E: 错误**。

| 条件 | 结果 |
|---|---|
| `y_turns` ≥3% | ✅ **+9.40%**（用干净的 off cell；名义 +12.50%） |
| `long_turns` ≥3% | ✅ **+4.63%**（两臂都干净，sd ≤0.11%） |
| 闸全过 | ✅ 四条全过，`l3_ppl` 逐位相同 |
| 零 E: 错误 | ✅ **12 个 cell 零错误零摘源** |

**⇒ GO。** 跨四个会话、五次尝试之后的第一个 A/B 表。

**默认值分两层，故意不一样：**

* **引擎默认仍然关。** `deepmoe serve` 不给 `--mirror` 就是单盘，逐字不变
  （`io.no_mirror_leaves_the_stats_untouched`）。它是一个库/CLI，不该替调用者猜盘。
* **网页 UI 默认开。** `tools/web/server.py` 现在 `find_mirrors()` 自动找
  「另一个盘上同样路径、且有 manifest」的那份拷贝（`--no-mirror-auto` 关，
  显式 `--mirror` 覆盖）。理由是它是用户走到机器前时**正在跑的那个进程**，
  而引擎侧的闸（启动 pinned 试读 / 失败摘镜像 / 运行期连续 3 次错误摘源）
  已经让「盘不在了」是一条 warning 而不是一次崩溃，
  就绪横幅的 `read sources N` 报的是**过闸之后真正在读的源数**。

**STATUS §7 第 1 项的 (b1) 结清了**：它说的是「第二块真 NVMe，+32–40%」。
**拿到的是 +4.6% 到 +9.4%（单流）/ +13.5%（两流）**，
而差距的原因写在 §9.5：**只路由 P0/P3、decode 是突发形状、E: 在突发下比稳态慢 5–6 倍**。
`(b2) 这块 U 盘不要再试` 仍然成立——**成立的是那个 UAS 盒子，不是这块 SSD**：
同一块 J.ZAO 在 USB4 盒子里跑完了 12 个 cell 零错误。**坏的一直是桥**（DX §4 的一号假说），
**换盒子确实把那一层整个拿掉了**。

### 9.9 没做的两件（都是故意的）

* **权重 / P0 队列深度的进一步调参。** 任务书允许「分流离带宽比例太远就试
  `DEEPMOE_MIRROR_WEIGHTS` 覆盖和 QD 24 → 32/48」。**试了第一个**：
  `DEEPMOE_MIRROR_WEIGHTS=4.72;3.66` 的那个 cell（`bench/results/d5/probe_w/`）
  是 **5.1046 tok/s / stall 97.1 / 分流 71.4 : 28.5**——和修好探针之后的默认
  （5.2368 / 95.4 / 73 : 27）**在同一个点上**。**手压权重买不到东西**，
  因为压不动的是 §9.5 那个 5–6 倍的延迟差，而路由已经在如实反应它。
  **QD 32/48 没试**：判据已经过了，而每个格子要 2–8 分钟（`feedback-fast-experiments`）。
* **被摘源上失败请求在主盘的重放**（D4 §8.4 的第 3 条）。本轮零错误，**没有盘能验它**。

## 10. Track D6：镜像的「唤醒」假说被证伪，而 149 ms 从来不是盘（2026-09-20）

一句话：**§9.5 那句「`src[1]` 的 mean lat 是 D: 的 5–6 倍」是我们自己的口径造成的——
它 97% 是 P3 backfill 在引擎队列里等 P0 的时间。把 P0 单独量出来，E: 是 4.5 ms，D: 是 3.8 ms，
差 1.19 倍。** 于是 keep-alive **+0.15%（否）**，按带宽比强压 44% 分流 **−2.68%（否）**，
**默认一个字不改**。

出处：`bench/results/d6/`（`abab_keepalive/` 与 `abab_static/` 各 4 个 cell，`gates/`）。
分支 `p4/d6-mirror-keepalive`，主 checkout，一次一个引擎；动工之前网页 UI 已经停掉，
`Get-Process deepmoe*` 为空。

### 10.1 假说，以及它在 D5 的数据里就已经被否掉了

本轮的假说（来自 `nvme_bench` 的独立测量：E: 热的时候每请求延迟**等于** D:——
4 MiB QD1 1.39 ms 两边相同，QD4 4.4 vs 3.5，QD8 8.0 vs 6.7——但**空闲几百毫秒之后
第一个读要到 1,093 ms**）是：

> 引擎里镜像拿到的字节少 → 突发之间进入空闲 → 链路/NVMe 低功耗态 →
> 每个突发付一次唤醒 → mean lat 149 ms → 路由给它更少。**自锁**，和 §9.2 探针那个环同构。

**第一条证据是 D5 自己的 `status.json`，不需要新跑任何东西**：

```
P0: 25092 req, 225043.8 MiB, lat mean 3.99 / p50 4.62 / p95 9.68 / max 19.84 ms
```

**整整一个 cell 里，最慢的一个 P0 是 19.84 ms**（六个 on cell 的 max 是 17.66–39.89 ms）。
一次 1 秒的唤醒**不可能**藏在一个 max 19.84 ms 的分布里。而 P0 是 routed 的两个类之一，
**E: 服务的 P0 也在这个 max 里面**。所以：**引擎的 P0 路径上没有唤醒惩罚**，假说在动工前就该死。

那 149 ms 是什么？**routed 的类有两个（`route_classes 0x9` = P0 | P3），而这个平均把它们混在一起。**
本轮给每个源加了 P0 专属的三个计数器，一量就清楚了（`y_turns`，keep-alive off 臂）：

| | `src[0]` D: | `src[1]` E: |
|---|---:|---:|
| routed 请求 | 18,386 | 7,920 |
| 其中 **P0** | **18,086** | **7,006** |
| 其中 P3 backfill | 300 | **914**（1,214 个里的 **75.3%**） |
| **混合** mean lat（§9.5 引用的那个数） | 22.32 ms | **150.87 ms** |
| **P0** mean lat | **3.81 ms** | **4.52 ms** |
| 反解出来的 **P3** mean lat | 1,138 ms | **1,272 ms** |

**两块盘的 backfill 都是 1.1–1.3 秒**，因为 P3 被 P0 抢占（`kBackgroundQuiet` 内最多 1 个在飞），
**它量的是引擎队列，不是盘**。E: 的延迟质量里 **97.3%** 来自那 914 个 backfill。

> **为什么 backfill 偏偏堆在 E: 上**：路由是加权最小**在飞字节**，而 backfill 是在 P0 突发
> *之后*发出的——那一刻 D: 的在飞字节正是刚发出的那一层 P0，所以 `(queue+me)/w` 选 E:。
> 于是 E: 扛下 75% 的 backfill，**而这些 backfill 的在飞字节又要挂在 E: 头上 1.2 秒**，
> 反过来把后面的 P0 推回 D:。**这才是把分流压在 28.5% 而不是权重比 43% 的那个环**，
> 和「盘睡着了」没有关系。§9.5 把这件事归因给「E: 对突发更慢」，**那一半是错的**。

空闲本身也量了（`idle gaps >= 15 ms` = 这个源在一个请求到达之前**完全没有任何在飞请求**的时长）：

| | 次数 | 平均 | 最大 |
|---|---:|---:|---:|
| D: | 296–309 | 54 ms | 8.0 s |
| E: | 1,002–1,010 | 59 ms | 8.0 s |

E: 的空闲**确实**比 D: 多三倍多，**但典型只有 59 ms**，远在 `nvme_bench` 量到惩罚的那个
「几百毫秒」之下；那个 8 秒的最大值**两块盘都有**，是脚本两轮之间的 prefill 停顿，
一轮只发生一次（4 次 / 7,006 个 P0），在平均里看不见。

### 10.2 落地的东西

**keep-alive**（`storage/source_router.h` 的 `keepalive_due` + `IoEngine::tick_keepalive`）：
非主源空闲超过 `DEEPMOE_MIRROR_KEEPALIVE_MS` 就发一个 **4 KiB、轮转偏移**的读。
它**直接走 backend**，不是 `IoRequest`：没有 `Pending`、没有队列槽、不进 `bytes_completed` /
`busy_ns` / 延迟和 / 每源字节分流——**所有 A/B 表的口径一个字不变**，它自己的三个计数器
（`keepalive N reads, N done, N refused`）说它做了什么。每个源**最多一个在飞**，而且
**永远不在该源有真实请求在飞时发**（盘本来就醒着，这时候插一个 4 KiB 只是抢 backend 队列槽）。
最后那一条是变异测试 `io.keepalive_never_races_a_real_request` 盯的那一行。**默认关。**

**每源 P0 计数器 + 空闲间隔计数器**：上面那两张表就是它们。
空闲间隔**两个臂都测**——它是判据，不是修法的副产品。

**`DEEPMOE_MIRROR_STATIC_SPLIT=f`**：把 P0 按累计字节开环压到固定份额。
加权最小在飞字节是个**闭环**（答得慢 → 在飞字节挂得久 → 分得更少），
所以它**结构上不可能**把一个源推到它「挣到」的份额之上；这一档是那个开环对照。

### 10.3 ABAB：两个都输了

`y_turns`，**两个臂都带 `--mirror`**（新的 `bench/d2_abab.py --arm-env/--off-env/--on-env`），
auto 槽，`--no-kv-disk`，ABAB 两对：

| 臂 | tok/s | sd | Δ | `nvme_stall` | E: 份额 | E: 的 P0 | E: P0 lat | 空闲 ≥15 ms | poke |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| keep-alive **off** | 5.2566 | 0.0098 | — | 95.5 ms | 28.5% | 7,006 | 4.52 ms | 1,006 | 0 |
| keep-alive **on**（15 ms） | 5.2643 | 0.0180 | **+0.15%** | 94.7 ms | 28.5% | 7,006 | 4.44 ms | 1,006 | **3,052** |
| 默认路由 **off** | 5.2503 | 0.0183 | — | 95.1 ms | 28.5% | 7,006 | 4.49 ms | 1,010 | 0 |
| **static split 0.44** | 5.1094 | 0.0115 | **−2.68%** | 100.9 ms | **44.0%** | 12,547 | 3.90 ms | 828 | 0 |

**keep-alive：3,052 次 poke，零 refused，零错误，然后什么都没发生。**
分流逐位相同（28.5%），E: 的 P0 延迟 4.52 → 4.44 ms（−1.8%，在噪声里），
tok/s **+0.15%**，判据 ±3% ⇒ **否**。**假说证伪**——不是「效果小」，是**那个机制不在这里**。

**static split：分流真的到了 44.0%**（开环所以它当然到得了），**然后慢了 2.68%**。
机制在同一份 `status.json` 里：

| | 默认路由 | static 0.44 |
|---|---:|---:|
| `io: busy`（同样的字节） | 47.795 s | **50.234 s** |
| `io: eff` | 5.15 GB/s | **4.90 GB/s** |
| P0 mean / p95 | 3.99 / 9.66 ms | **4.09 / 10.45 ms** |
| P0 first-of-burst | 1.15 ms | **1.27 ms** |
| D: P0 lat | 3.80 ms | 4.29 ms |
| E: P0 lat | 4.47 ms | **3.90 ms** |

**两块盘各自的 P0 延迟都没变坏到能解释它**（E: 甚至更快了——它拿的多了、更热了）。
变坏的是**层的那个 max**：一层 ~9.6 个 chunk 的突发被摊得更平之后，
**每一层都要在两块盘上各付一次爬坡**，而 `first-of-burst` 1.15 → 1.27 ms 就是它。
decode 等的是一层里最慢的那个 expert，不是平均。

> **所以 §9.9「手压权重买不到东西」的结论要加强一句**：
> 不只是手压 `DEEPMOE_MIRROR_WEIGHTS` 没用（D5 试过，5.1046 落在同一点），
> **连绕开闭环、直接按带宽比开环压，也是负的**。加权最小在飞字节给出的 28.5%
> **不是它够不到 44%，是 44% 本来就更差**。稳态带宽比不是 decode 的正确目标函数，
> **每层突发的完成时间才是**。

### 10.4 Windows 的盘电源设置

任务书允许「顺带看一眼能不能按设备设 idle timeout / 链路电源管理」。**看了，没改，也不建议改**：
这台机器上 E: 枚举成 `BusType NVMe`（PCIe 隧道），它的 idle timeout 属于 NVMe 电源状态机
（APST）+ PCIe ASPM，**不在 SetupAPI 能改的那一层**；Windows 侧能动的是电源方案里
`disk idle timeout` 和 `PCI Express / Link State Power Management`，**两个都是系统级的**。
而且 §10.1 已经说明**根本不需要**：P0 的 max 是 19.84–39.89 ms，唤醒惩罚从来没有到过 P0。
**不动系统电源设置**（`feedback-vgm-irrelevant` 那条规矩的同类：不要为了一个没被证实的机制
去改机器的全局状态）。

### 10.5 闸

全部**在镜像开着**的情况下跑（`DEEPMOE_MODEL_MIRRORS=E:\models\DeepSeek-V4.1-Flash`）：

| 闸 | 结果 |
|---|---|
| `suite.decode` | ✅ Passed（108.2 s） |
| `suite.decode_longctx` | ✅ Passed（270.9 s） |
| `suite.integration` | ✅ Passed（1.4 s） |
| `tools/l3_ppl.py --modes off`（`traces/l3_64`） | ✅ **NLL 0.630051 / PPL 1.8777 / top-1 61/64** —— 与 §9.7、D4 §8.6、MS、K1a **逐位相同**（跑了两次，两次都是） |
| CPU 全量 `ctest -LE "needs-model;needs-gpu"` | ✅ **46/46**（`suite.io` 33 例，含新增的两条 D6 测试） |
| 变异 `io.keepalive_never_races_a_real_request` | ✅ **caught**（27 s） |
| 8 个 A/B cell 的 E: 错误 | ✅ **零**（`0 failed`，无 `DROPPED`，`serve.log` 无 `win32`） |

⚠️ **一次瞬时的 E: 掉线，记下来**：`l3_ppl` 的**第一次**运行里，
5,000 槽的 warm-cache backfill 途中 E: 报了 `win32 433 / win32 55`，
运行期闸按设计**连续 3 次就摘源**，那一轮在单盘上跑完，
`backfill 5005 issued / 5000 done / 5 failed`，**NLL 仍然是 0.630051**。
**同一条命令立刻重跑：零错误**，E: 拿 39.4% / 5,909 req，和 D5 §9.7 的 5,917 req / 38.1 GiB 对得上。
本会话 11 个带镜像的引擎进程里**只有这一次**。记在这里是因为
**D2–D4 的 `win32 433` 是同一个码**：USB4 盒子把那一层的概率压得很低，**但没有压到零**。
闸做了它该做的事——**一条 warning 加一次降级，不是一次崩溃**。

### 10.6 判决

| 项 | 判据 | 实测 | 判决 |
|---|---|---|---|
| keep-alive | ≥3% 且零错误 | **+0.15%**，零错误 | **NO-GO**，默认关，代码与计数器留下 |
| static split 0.44 | ≥3% | **−2.68%** | **NO-GO**，默认关 |
| 默认路径 | 与 D5 逐字相同 | 是 | **不变** |

**留下的是三样东西，都不是性能**：
① **每源 P0 计数器**——`src[]` 那行的混合 mean lat 再也不会被读成「盘慢」；
② **空闲间隔计数器**——下一次有人提「盘是不是睡着了」，这是一个 `status.json` 就能回答的问题；
③ **两个关着的开关**（`DEEPMOE_MIRROR_KEEPALIVE_MS`、`DEEPMOE_MIRROR_STATIC_SPLIT`），
和一个能让两个臂都带 `--mirror` 的 ABAB harness。

**没做的**：`long_turns`（判据写的是「`y_turns` 赢了才加」，它没赢）；
`DEEPMOE_MIRROR_KEEPALIVE_MS` 的其它窗口（机制已经证伪，扫窗口是在扫噪声）。
