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
