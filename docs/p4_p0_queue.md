# Track Q1 — P0 在盘上到底等了什么（2026-09-18）

一句话：**假设错了。** stall 里那 ~30 ms 的缺口**不是**队列争用——decode 里
**0.0% 的 P0 在发出时有任何非 P0 的 chunk 在飞**。缺口是**这台盘在引擎的
访问形状下就只跑到 3.5 GB/s**，而同一块盘在 `nvme_bench` 里跑 5.16 GB/s。
四个 A/B 的三个是 NO-GO，第四个（P0 用 1 MiB chunk）**+1.4%，低于 ±3% 的判据带，
不作默认**。

出处：本文所有数字来自 `C:\Users\Asus\code\deepmoe-q1`（分支 `p4/q1-p0-queue`），
同一个二进制、同一个 shader 目录、同一个脚本
`bench/results/hitrate/y_turns.json`（4 轮 × 64 token，贪心，逐 token 可复现），
`--cache-slots 5100`，一次一个 `deepmoe serve`。原始结果在
`bench/results/p0q/<cell>/`（`profile.jsonl` / `events.jsonl` / `status.json` /
`turns.json` / `route.bin`），报表用 `tools/p0q_report.py`。

---

## 1. 插的桩

三处，全是加法，默认不改任何行为。

| 在哪 | 记什么 |
|---|---|
| `storage/io_engine.h` `IoStats` | 每个 **P0 请求**：提交→完成的延迟、字节、**排队等待**（提交→第一个 chunk 交给 backend）与**服务时间**（第一个 chunk→最后一个 chunk 落地）、**发出时在飞的非 P0 chunk 数**、**提交时前面还有几个 P0**（= 是不是这一层这一批的头一个）、发出时的 chunk 队列深度 |
| `storage/io_engine.cpp` `IoStats::to_string()` | 两行 P0 汇总：req / MiB / mean / p50 / p95 / max，以及 first-of-burst vs behind、带 bg 的比例、平均 QD。`serve` 的 `{"op":"status"}` 已经把它写进 `status.json`，所以**每个 cell 自带这份汇总** |
| `core/profiler.h/.cpp` | 每 token 的 `p0_count` / `p0_mean_ms`，进 `profile.jsonl`；`RunSummary` 多一行 P0 |

percentile 用保留的每请求样本（每个 P0 一个 `uint32` 微秒），`stats()` 时排序，
不在热路径上做任何事。

另外四个**只用环境变量**的旋钮（默认全部等于今天的行为，所以 A/B 不用重新编译）：

```
DEEPMOE_IO_BG_CAP_BUSY     P0 近期活跃时后台类允许在飞的 chunk 数（默认 1）
DEEPMOE_IO_BG_THROTTLE_P2  engram 类是否也让路（默认 0）
DEEPMOE_IO_P0_QD           P0 的 chunk 队列深度（默认 = cfg.max_inflight_ops = 8）
DEEPMOE_IO_P0_INFLIGHT_MB  P0 的在飞字节上限（默认 32 MiB）
DEEPMOE_IO_P0_CHUNK_MB     P0 请求的 chunk 大小（默认 4 MiB）
```

`IoEngine::widen_for_env` 在 `make_default_backend` 之前把 IoConfig 抬到 P0 要的
高度（backend 的队列深度是构造时定死的），`IoEngine` 自己再把 P1–P3 压回出厂值，
所以「给 P0 更深的队列」不会偷偷变成「给 backfill 更深的队列」。
`bench/nvme_bench` 每个点自建 IoConfig，不受影响。

---

## 2. 基线：一个 token 的 stall 是怎么花掉的

`a1/a2/a3/a4_bfoff`（backfill 默认 off，四次重复）：

| | a1 | a2 | a3 | a4 | 均值 |
|---|---:|---:|---:|---:|---:|
| decode tok/s | 4.855 | 4.863 | 4.812 | 4.857 | **4.847**（±0.5%） |
| `nvme_stall` ms/token | 104.8 | 105.0 | 105.9 | 105.3 | **105.2** |
| decode hit | 0.9111 | | | | |

每 decode token 的分项（引擎自己的 `per_token_ms`，decode-only）：
`attn 42.1 / moe_gpu 42.4 / moe_host 1.2 / nvme_stall 105.2 / engram 4.5 / tail 6.1 / other 4.4`。

字节与请求（decode-only，来自 `profile.jsonl`）：

* **21.3 个 expert miss / token = 383 MiB / token**（假设里写的是 19 × 18.8 MB = 360 MB，
  差 6%，方向一致）。
* **42.7 个 P0 请求 / token**：**一个 expert 恰好两个 run，两个 `IoRequest`**
  （`integration.every_run_is_a_legal_unbuffered_read`：15,744 个 expert / 31,488 个 run，
  `max 2 per expert`）。每个 run 8.97 MiB。
* 383 MiB ÷ 105.2 ms = **3.55 GB/s**，而 `bench/nvme_bench` 同一块盘是 **5.16 GB/s**。
  **这就是那个缺口，30%。**

全程 P0 汇总（`status.json`，含 prefill；31,148 个 P0 请求 / 272.8 GiB）：

```
P0: lat mean 6.70 / p50 5.36 / p95 16.11 / max 31.43 ms   (queue wait 2.00 + service 4.69)
P0: first-of-burst 9769 at 2.83 ms, behind 21379 at 8.46 ms;
    0 (0.0%) issued with a non-P0 chunk in flight;  mean QD at issue 3.80
```

三件事直接读出来：

1. **争用是零。** backfill off 时 **0.0%** 的 P0 在发出时有非 P0 的 chunk 在飞；
   backfill on 也只有 **8.2%**，且平均只有 **1.02** 个（就是 `kBackgroundOpsWhileBusy`
   那一个）。**今天的 P1/P3 节流已经把这条路堵死了**，假设里的「P3/P2 争用」不存在。
2. **QD ramp 是真的，但它不是账单。** 一层这一批的**第一个** P0 只要 **2.83 ms**
   （8.97 MiB ⇒ 3.17 GB/s），**后面的每个要 8.46 ms**。但一层等的是**最后一个**，
   所以决定 stall 的是 `behind` 那一栏。
   `plan_layer` 里 9,769 次 stall（= 有 miss 的层数）对 31,148 个 P0 请求，
   **每个有 miss 的层平均 3.19 个请求 = 1.59 个 expert**；
   105.2 ms ÷ (13.4 个有 miss 的层/token) = **每层 7.8 ms**，和 `behind` 的 8.46 对得上。
3. **引擎跑不到它自己的队列深度。** 上限 8，**发出时的平均在飞 chunk 数只有 3.80**。

### 2.1 一层里的 miss 是并发发出的，不是串行

`Planner::plan_layer` 把这一层**所有** miss 先 `fetch` 完再 `wait_layer`，
`Planner::fetch` 在同一个循环里把一个 expert 的**两个 run 背靠背**提交，中间不等。
所以到 `IoEngine` 时它们就在同一个 P0 队列里排着。
真正的限制在 `issue_ready_chunks`：**QD 8 / 32 MiB 在飞**，一个 run 是 3 个 4 MiB chunk，
一层 3.19 个 run = 9.6 个 chunk，**放不下**，于是最后几个 chunk 要等前面的退休。
这不是「串行发出」，是「队列装不下一层」。
把它放大（§3 的 b）**没有变快**——见下。

### 2.2 backfill 开不开，不影响 decode

| | tok/s | stall | 带 bg 的 P0 |
|---|---:|---:|---:|
| `e_bfon`（`DEEPMOE_BACKFILL=1`） | 4.846 | 105.8 | 8.2%，平均 1.02 个 |
| 基线均值 | 4.847 | 105.2 | 0.0% |

**−0.0%。** 和 `p4_hitrate.md` §7 说 backfill 默认 off 的理由一致。

---

## 3. 四个干预，ABAB 交替，同一个脚本

判据：**±3% 的抖动带**。实测抖动比这窄得多（四次基线 4.812–4.863，**±0.5%**），
但判据不改。

| cell | 变量 | tok/s | Δ | stall ms | P0 lat | first | behind | 发出时平均 QD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **a1–a4** | 基线（4 MiB × QD 8） | **4.847** | — | 105.2 | 6.70 | 2.83 | 8.46 | 3.80 |
| `b_qd32` | **(b)** `P0_QD=32`，`P0_INFLIGHT_MB=256` | 4.875 | **+0.6%** | 104.7 | **7.70** | 4.10 | 9.33 | 6.04 |
| `c_ck16` | **(c)** `P0_CHUNK_MB=16`（一个 run 一个 op） | **4.442** | **−8.4%** | 123.3 | 7.57 | 3.48 | 9.42 | 2.38 |
| `bc_qd32ck16` | (b)+(c) | 4.429 | −8.6% | 123.6 | 7.92 | 3.77 | 9.81 | 2.83 |
| `d_ck2`/`d2_ck2` | **(c′)** `P0_CHUNK_MB=2` | 4.893 | +0.9% | 102.1 | 6.15 | 2.28 | 7.91 | 4.67 |
| `h_ck1`/`h2`/`h3` | **(c″)** `P0_CHUNK_MB=1` | **4.916** | **+1.4%** | 102.5 | **5.76** | **1.30** | 7.77 | 5.40 |
| `g_bfon_yield0` | **(a)** backfill on + `BG_CAP_BUSY=0` | 4.830 | −0.3%（对 `e_bfon` 的 4.846） | 105.4 | 6.59 | 2.81 | 8.33 | 3.80 |
| `f_bfon_yield` | (a) + `BG_THROTTLE_P2=1` | **3.500** | **−27.8%** | 103.9 | 6.57 | 2.74 | 8.34 | 3.83 |

顺序是 ABAB：基线四次穿插在中间（a1 → b/c/bc → a2 → d → e/f → d2 → a3 → h → g → h2 → a4 → h3）。

逐条：

**(a) 非 P0 让路 —— NO-GO，而且其中一半是有害的。**
`BG_CAP_BUSY=0`（P0 活跃时后台**完全停**）对 `e_bfon` 是 **−0.3%**，在抖动里。
原因 §2 已经说了：**能让的路早就让了**，带 bg 的 P0 只有 8.2% 且只有 1 个 chunk。
把 engram（P2）也拉进节流（`BG_THROTTLE_P2=1`）是**这一整轮唯一的大变化，而且是反的**：
tok/s 4.846 → **3.500（−27.8%）**。分项干净地指认了原因——**`nvme_stall` 没变
（105.8 → 103.9），`engram` 从 4.5 ms 炸到 85.8 ms/token**。
engram 的 264 B 行读也在关键路径上，节流它等于把它扔到 P0 后面排队。
**结论：P2 永远不节流**（`io_engine.h` 的注释本来就这么写，现在有数了）。

**(b) 给 P0 更深的队列 —— NO-GO。**
`P0_QD=32` 把发出时的平均在飞 chunk 从 3.80 抬到 **6.04**，
但**每个 P0 反而更慢**（6.70 → 7.70 ms，`behind` 8.46 → 9.33），
tok/s +0.6% 在抖动里。**队列深度不是那 30% 缺口的因**。

**(c) 更大的 chunk —— 明确 NO-GO，方向是反的。**
16 MiB（一个 8.97 MiB 的 run 整个作一个 op，正是 `nvme_bench` 里最快的形状）
**慢 8.4%**，stall +17%。发出时平均 QD 掉到 2.38。
**反过来**：2 MiB **+0.9%**、1 MiB **+1.4%**，`first-of-burst` 从 2.83 ms 掉到 **1.30 ms**
（8.97 MiB / 1.30 ms = **6.9 GB/s**，一个 run 独占盘时它是够快的）。
但 `behind` 只从 8.46 降到 7.77，所以端到端只买到 1.4%。

**(d) expert 的两个 run 是不是一起发的 —— 是。**
`Planner::fetch` 一个循环里提交两个 `IoRequest`，中间不等；
`plan_layer` 把一层所有 miss 提交完才 `wait_layer`。
计数对得上：`p0_blocking` 请求数 / `fills` = **31,148 / 15,574 = 2.00**，
`first-of-burst` 数 = planner 的 `stalls` 数 = **9,769**（一层一个）。
**这里没有可省的串行。**

---

## 4. 那 30% 到底在哪：`nvme_bench` 的对照

同一台机器，同一块盘，写进**普通主机内存**（`bench/results/p0q/nvme_chunk_qd.csv`）：

```
.\build\nvme_bench.exe --size-gb 4 --chunk-kb 1024,2048,4096,9184,18360 --qd 1,4,8 --pattern rand --reads 48

pattern  chunk_kb   qd     GB/s   mean_ms
rand         1024    1    2.935     0.356
rand         1024    4    5.067     0.808
rand         1024    8    4.974     1.584
rand         2048    4    5.107     1.603
rand         4096    1    4.011     1.044
rand         4096    4    5.156     3.173
rand         4096    8    5.130     6.126
rand         9184    4    5.118     7.132      <- 正好是一个 run 的大小
rand        18360    4    5.155    14.163
```

**盘对请求大小是平的**：1 MiB 到 18.4 MiB，只要 QD ≥ 4 就是 **5.07–5.16 GB/s**。
而引擎在 stall 窗口里是 **3.55 GB/s**（全程 `IoStats` 的 busy 窗口是 4.02 GB/s）。

所以：

* **缺口不是队列争用**（0.0%），
* **不是队列深度**（抬到 32 反而更慢），
* **不是请求太碎**（盘对大小是平的；而引擎对大小**不平**，16 MiB 比 4 MiB 慢 17%、
  1 MiB 比 4 MiB 快 2.6%）。

引擎对 chunk 大小敏感而盘不敏感，这两件事只能同时成立于**引擎侧**：
要么是**目的地内存**（expert slot 是 GPU 可见的，`nvme_bench` 写的是普通页），
要么是 **dispatcher 的补队速度**（`issue_ready_chunks` → `backend_->poll(1 ms)` →
`handle_completion` → 再 `issue_ready_chunks` 这一圈；上限 8 而实测只到 3.80，
说明队列没被填满，而更碎的 chunk 恰好给了它更多可填的东西：
16 MiB → QD 2.38、4 MiB → 3.80、1 MiB → 5.40）。
**本轮没有把这两者分开**，见 §6。

`STATUS.md` §4 那句「没有 CPU 开销、没有共享显存写入的代价、没有排队，就是盘」
**要收窄**：「就是盘」成立于**一次 miss 的量级**（13.6 ms vs 20–31 ms 对得上），
但**不成立于 decode 的聚合速率**——引擎拿到的是 3.55 GB/s，盘能给 5.16 GB/s。
**排队这一条现在是实测排除的**（0.0%），**另外两条（目的地内存、补队速度）没有排除**。

---

## 5. 预测对实测

| | 预测 | 实测 |
|---|---|---|
| 每 token miss 字节 | 19 expert × 18.8 MB = 360 MB | **21.3 × 18.8 = 383 MiB**（差 6%） |
| 纯传输时间 @5.2 GB/s | 69 ms | 74 ms |
| `nvme_stall` | 100–111 ms | **105.2 ms** ✓ |
| 缺口 | ~40 ms 是队列争用 / QD ramp / 每 miss 延迟 | **~31 ms，其中队列争用 = 0** |
| 「P0 独占盘」后的 stall | 111 → ~75 ms | **105.2 → 105.4**（`BG_CAP_BUSY=0`，**没有变化**） |
| tok/s | 4.8 → ~6（+25%），砍半 +12% | **4.847 → 4.830（−0.3%）** |

**假设的机制被证伪**：P0 已经独占着盘，而 stall 没有因此变小。
唯一活着的一条是「每 miss 延迟」，而它不是排队造成的。

---

## 6. 留下什么、退掉什么

| 改动 | 结论 |
|---|---|
| P0 队列插桩（`IoStats` 两行 + `profile.jsonl` 的 `p0_count`/`p0_mean_ms`） | **保留，默认开**。零热路径成本，`status.json` 自带 |
| `DEEPMOE_IO_*` 五个旋钮 | **保留，默认全部等于今天的行为**。它们是这一节能重跑的原因 |
| (a) `BG_CAP_BUSY=0` | **退掉**（−0.3%，在抖动里）。已有的节流已经够了 |
| (a′) `BG_THROTTLE_P2=1` | **退掉，并且写进「不做」**：engram 4.5 → 85.8 ms/token，−27.8% |
| (b) `P0_QD=32` / `P0_INFLIGHT_MB=256` | **退掉**（+0.6%，在抖动里，且 P0 延迟更差） |
| (c) `P0_CHUNK_MB=16` | **退掉**（−8.4%） |
| (c″) `P0_CHUNK_MB=1` | **不作默认**：+1.4%，**低于 ±3% 的判据带**。三次 B（4.932/4.917/4.899）对四次 A（4.812–4.863）方向一致、P0 延迟 6.70 → 5.76 ms、first-of-burst 2.83 → 1.30 ms，**但端到端买不到判据要求的那么多**。旋钮留着，重跑一条命令 |

**下一步（不在本轮范围内，但这一轮把它变成了唯一活着的问题）**：
把「目的地内存」和「dispatcher 补队速度」分开。最便宜的探针是给 `nvme_bench`
加一个「写进 GPU 可见 slab」的目的地开关，一个 cell、纯盘、不占 GPU；
如果那样它还是 5.1 GB/s，缺口就全在 dispatcher 那一圈，
而那是 `issue_ready_chunks`/`poll` 的代码问题，不是盘的问题，
**值 105 ms 里的 ~30 ms = 每 token 30%**，比 `STATUS.md` §7 第 4–7 项都大。

---

## 7. 重跑

```powershell
# 每个 cell ~2 min 20 s，一次一个引擎
foreach ($c in @(
  @{n='a1_bfoff'; e=@{}},
  @{n='b_qd32';   e=@{DEEPMOE_IO_P0_QD='32'; DEEPMOE_IO_P0_INFLIGHT_MB='256'}},
  @{n='c_ck16';   e=@{DEEPMOE_IO_P0_CHUNK_MB='16'; DEEPMOE_IO_P0_INFLIGHT_MB='128'}},
  @{n='h_ck1';    e=@{DEEPMOE_IO_P0_CHUNK_MB='1'}},
  @{n='e_bfon';   e=@{DEEPMOE_BACKFILL='1'}},
  @{n='g_bfon_yield0'; e=@{DEEPMOE_BACKFILL='1'; DEEPMOE_IO_BG_CAP_BUSY='0'}}
)) {
  $ea=@(); foreach ($k in $c.e.Keys) { $ea+='--env'; $ea+="$k=$($c.e[$k])" }
  .venv\Scripts\python.exe tools\hitrate_bench.py --script bench\results\hitrate\y_turns.json `
      --cache-slots 5100 --out "bench\results\p0q\$($c.n)" --exe build\deepmoe.exe `
      --shader-dir build\shaders @ea
}
.venv\Scripts\python.exe tools\p0q_report.py bench\results\p0q\*     # 表 §3

# 盘的对照（纯盘，~4 min，不占 GPU）
.\build\nvme_bench.exe --size-gb 4 --chunk-kb 1024,2048,4096,9184,18360 --qd 1,4,8 `
    --pattern rand --reads 48 --csv bench\results\p0q\nvme_chunk_qd.csv
```

## 8. 测试

`build\tests\deepmoe_tests.exe io.` **8/8**（含
`io.background_is_throttled_while_p0_is_recent`，它按 `kBackgroundOpsWhileBusy`
断言，默认值没动所以仍然过）；
`build\tests\deepmoe_tests.exe integration.` **5/5**（`every_run_is_a_legal_unbuffered_read`
就是 §3 (d) 的出处：15,744 expert / 31,488 run / `max 2 per expert`）。
