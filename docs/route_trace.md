# 路由 trace 与 cache 模拟器（design §9.1 Q1–Q4）

> 本文是 `tools/route_trace.py`、`tools/cache_sim.py`、`tools/dsref.py`、
> `tools/corpus.py` 的使用与方法说明。设计依据：[design.md](design.md) §9.1（要回答的
> 问题）、§9.2（测量工具）、§9.3（分层驻留）、§9.4（lookahead 预取）、§16（测量出来
> 之前不许写 Planner 策略）。
>
> 结论性数字**不在这里**。本文只给出 2,000 token 的 **smoke 切片**，用来证明流水线
> 端到端跑通并给出全量运行的耗时外推。Q1–Q4 的答案要等全量 ≥ 20K token 的 trace 跑完
> 之后才写进 design §9。

---

## 1. 为什么是 prefill 而不是 decode

teacher forcing 下，每个位置的路由决策就是 decode 同一段文本时会做出的决策——模型是因果
的，位置 p 之后的内容不会回流。区别只在权重的读法：

| | 每 token 读权重 | 20K token 总计 |
|---|---|---|
| decode 形状的 trace | 13.0 GB（§2.3） | 260 TB |
| prefill 形状的 trace | — | **510 GB，全语料共一遍** |

所以语料是**按层流式**处理的：装载第 L 层的权重 → 把所有 prompt 的隐状态推过去 → 释放
→ 下一层。20K token 的 `[hc=4, 5120]` bf16 残差流是 0.8 GB，一层的 routed expert 是
7.2 GB，两者都装得下；40 层的 expert 一起装不下。

层内 FFN 再按 **expert-major** 跑：一层的 384 个 expert 各解码一次，应用到路由到它的
那些 token 上。这正是 §9.7 给 prefill 规定的顺序，所以这个脚本的形状就是 runtime
prefill 路径的形状。

**decoder（层 20–39）跑全部位置**，即 §11.2 的 "oracle 模式" 而不是生产用的 bounded
replay。bounded replay 是 prefill 的近似；trace 要的是每个位置**被 decode 时**会看到的
路由，那就得让 decoder 前向全部位置。

---

## 2. 环境

```powershell
# 仓库里已经有 .venv（P0 的 oracle.py 用的那个）；没有就先建
uv venv
uv pip install numpy safetensors --index-url https://pypi.org/simple
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install pyarrow tokenizers
uv pip install pypdf        # 可选：把技术报告 PDF 收进语料
```

shell 里有 `HTTP_PROXY/HTTPS_PROXY=127.0.0.1:7078`。**PyPI 走代理是通的**（实测
`uv pip install pyarrow tokenizers` 直接成功），和附录 B 里"下权重必须直连"的规则不
冲突——那条规则只针对 ModelScope CDN。

需要的其他东西：

- `D:\models\DeepSeek-V4.1-Flash\deepmoe_manifest.json`（`tools/manifest.py` 生成）
- `D:\models\DeepSeek-V4.1-Flash\tokenizer.json`
- `D:\models\DeepSeek-V4.1-Flash\inference\`（`model.py`、`kernel.py`、`engram.py`、
  `config.json`）

**只读**：这些脚本从不往模型目录写任何东西。

---

## 3. 跑法

### 3.1 先跑 faithfulness 检查

```powershell
uv run python tools/route_trace.py --model D:\models\DeepSeek-V4.1-Flash `
    --out traces\verify --verify
```

一次 64 token 的完整 40 层前向，约 5 分钟，结果写 `traces\verify\verify.json`。
结果见 §5。

### 3.2 全量 trace

```powershell
uv run python tools/route_trace.py --model D:\models\DeepSeek-V4.1-Flash `
    --tokens 20000 --out traces\mixed
```

**单命令、可断点续跑。** 每层结束写一次 `traces\mixed\state.pt`（残差流 + 跨层
attention 状态 + lookahead 快照），再跑同一条命令就从下一层接着算。改
`--checkpoint-every N` 可以少写几次（state 大约 100 KB/token，20K token ≈ 2 GB）。

> ⚠️ 这是个吃满 16 线程的 CPU 任务。**不要和 Track A 的带宽基准同机并跑**——
> `bench/bw_matrix` 测的是内存/NVMe 带宽，被这个任务污染就白测了。

### 3.3 跑模拟器

```powershell
uv run python tools/cache_sim.py --trace traces\mixed `
    --capacities 20,25,30,35,abs:4787 `
    --policies lru,lfu-decay,arc,score-aware,static-pin+lru `
    --allocation both --prefetch-depths 0,2,3,4,6 --prefetch-widths 6,8,12,16 `
    --out reports\cache_sweep.json
```

`--trace` 吃目录、glob 或单个文件，可重复。

### 3.4 单元测试

```powershell
uv run python tools\tests\test_cache_sim.py     # 不需要 pytest
uv run python -m pytest tools\tests\            # 有 pytest 也行
```

9 个用例，全部是**答案可独立算出**的合成 trace：完美局部性 → 命中率
`1 − 1/n`；均匀随机 → LRU 命中率 = C/M（解析解）；stack distance 对暴力定义逐元素
相等；stack distance 的 CDF 就是 LRU 的命中率曲线；Q1 覆盖率与 Q3 Jaccard 对手算值；
oracle 预取能隐藏 miss、无用预取纯浪费；static-pin 不淘汰 pinned；tok/s 模型等于
§3.1 的公式。

---

## 4. 实现结构：为什么信得过

design §12 说得很直白：`oracle.py` 是手工移植，因而是信任链上最弱的一环。这套工具走
相反的路线——**直接 import `inference/model.py` 本身**，只替换它在 CPU 上跑不了的六个
tilelang kernel：

```
act_quant   fp4_act_quant   fp8_gemm   fp4_gemm   hc_split_sinkhorn   sparse_attn
```

于是 mHC/Sinkhorn、gate、expert、带 compressor/indexer/candidate 两级筛选的 attention、
RoPE 与 YaRN、Engram 的 n-gram hash，全都是参考实现自己的代码，跑在它自己的
`inference/config.json` 上。**需要评审的只有四件事**：

1. **六个 kernel shim**（`tools/dsref.py` §3）。每个都把它对应的 tilelang 源码摘在
   docstring 里。
2. **manifest 权重加载器**（`WeightStore`），就是 `tools/oracle.py` L1 已经证明与
   `safetensors` 库逐字节相同的那套 run/skew 算术。
3. **`StreamingMoE` / `StreamingBlock`**，把 `Block.forward` 按 FFN 切成两半，好让
   driver 一次只流一层 expert。数学一行没改，原版 `Block.forward` 抄在 docstring 里对照。
4. **Engram 行读取器**，按需取 24 行，而不是把 98 GB 的表读进内存。

### 数值约定

`generate.py` 里 `torch.set_default_dtype(torch.bfloat16)`，所以残差流、每个 `Linear`
的输出、attention 的输入都是 **bf16**，而 mHC 系数、gate、expert 累加是 fp32。这里
完全照搬。

权重一次性解码成 bf16，而且**这是无损的，不是近似**：

- FP4 E2M1 的取值是 `{0, .5, 1, 1.5, 2, 3, 4, 6}` 加符号——一位尾数——UE8M0 scale 是精确
  的 2 的幂，所以 `value × scale` 永远落在 bf16 的 8 位尾数上。
- FP8 E4M3 三位尾数，块 scale 同样是 UE8M0，同理。

这台机器上 bf16 × bf16 的 matmul 用 fp32 累加、输出舍入到 bf16，正是 `fp8_gemm` /
`fp4_gemm` 做的事（fp32 累加器，`out_dtype = BF16`）。**激活侧的量化也没有跳过**：
`act_quant_dequant` 逐位复刻了 `act_quant_kernel`，包括 `fast_round_scale` 的
"向上取到 2 的幂"（用 fp32 位模式算，不是 `ceil(log2(x))`——后者在恰好是 2 的幂时会
算错）。

所以与参考实现**唯一有意的差异是 fp32 的累加顺序**，外加下面这条：

- `sparse_attn` 的 QK 与 PV 在我们这里是 fp32 matmul。因为输入本来就是 bf16，升到 fp32
  是精确的，fp32 累加等价于 kernel 的 "bf16 gemm + fp32 累加器"；概率矩阵按 kernel 的
  `acc_s_cast` 舍到 bf16，输出按 kernel 舍到 bf16。等价性靠推理，不靠实测——这台机器上
  没有能跑 tilelang 的 GPU，无法做 A/B。**这是本工具最大的未验证假设。**

### 性能

| | 每 expert |
|---|---|
| 从分片读 18.8 MB（Python 缓冲读，单线程） | ~16 ms（1.2 GB/s） |
| FP4 → bf16 解码 3 个矩阵 | ~35 ms |

解码用 torch 而不是 numpy：torch 把 gather 和乘法摊到 16 个核上，实测比 numpy 快 7×
（240 ms → 35 ms/expert）。读取由一个后台线程预取，正好被解码盖住（§9.6 P1 队列的离线
影子）。

一层的 FFN 输入只做一次 `act_quant`，而不是每个 routed expert 各做一次：`act_quant`
只跟激活有关、与权重无关，所以这不是近似，是**同样的字节少算 6 遍**。

---

## 5. Faithfulness 检查与结果

`--verify` 跑一个 64 token 的英文 + 中文 + Python 混合 prompt，走完整 40 层。

<!-- VERIFY_RESULTS -->

---

## 6. 语料

`tools/corpus.py`。约束两条：

- **不下载任何东西。** 附录 B 的"不走代理"规则针对权重，但同样的纪律适用于语料：每个
  来源都是机器上已有的文件，默认来源要么是本仓库（MIT）、要么是 checkpoint 目录里自带
  的 model card 与参考实现（MIT）、要么是 CPython 标准库（PSF）。`--source PATH:KIND`
  可以追加。
- **长度分布要真实。** 重用距离（§9.1 Q2）以 token 计，40 个 500 token 的 prompt 和
  10 个 2000 token 的 prompt 尾部完全不同。prompt 长度在 `[--min-len, --max-len]` 上
  按对数均匀抽样，且一个 prompt 永远是**单一来源的连续文本**。

token 预算在有来源的 kind（zh / en / code）之间均分，kind 内部各文档轮转，所以一个大
文件不会把预算吃光。

<!-- CORPUS_RESULTS -->

### 已知的语料局限

**中文几乎全部来自本仓库自己的设计文档**，是真实中文，但只有一种语域（系统工程技术
散文）。如果 Q1/Q3 在中文上的结论要外推到对话或文学体裁，需要另找语料重跑。这是
trace 的**已知局限，写在这里，不是疏忽**。

---

## 7. Parquet schema

每层一个文件 `route_layer{L:02d}.parquet`（zstd）。一行 = 一个 (token, layer)。

| 列 | 类型 | 含义 |
|---|---|---|
| `prompt_id` | uint16 | 语料里的 prompt 序号 |
| `pos` | uint32 | prompt 内位置 |
| `layer` | uint8 | 0–39 |
| `top6_ids` | fixed_size_list\<uint16\>[6] | `Gate.forward` 选出的 6 个 expert，按 `score+bias` 降序 |
| `top6_weights` | fixed_size_list\<float32\>[6] | 实际用的路由权重（归一化 × `route_scale`=1.5） |
| `top16_ids` | fixed_size_list\<uint16\>[16] | 按 `score+bias` 的 top-16；前 6 个恒等于 `top6_ids` |
| `top16_scores` | fixed_size_list\<float32\>[16] | **未加 bias** 的原始 `sqrt(softplus(xW))`，§9.3 score-aware 淘汰要用的就是它 |
| `gate_bias_applied` | bool | `noaux_tc` 的 bias 是否改变了 top-6 的集合（即"按原始分数选"与"按 score+bias 选"是否不同） |
| `pred_d{d}_mean` | fixed_size_list\<uint16\>[16] | 从第 L−d 层残差流、用 `ffn_norm(mean_hc(x))` 近似预测的本层 top-16（§9.4） |
| `pred_d{d}_pre0` | fixed_size_list\<uint16\>[16] | 同上，近似换成 `ffn_norm(hc_pre(x, pre_identity))` |

`pre_identity` 是 `make_identity_pre_mix` 的 one-hot，所以
`hc_pre(x, pre_identity)` 恰好等于残差流的第 0 份拷贝。两个近似都是 `[n, 5120]` 的
投影，快照因此是 `[n, 2, 5120]` 而不是整条 `[n, 4, 5120]` 流——这才让保留 8 个深度的
快照变得可负担（8 × 2 × 20K × 5120 × 2 B = 3.3 GB）。

L < d 的层没有第 L−d 层，对应列整列为 null。

**bias 为什么不逐行存**：bias 是 `(layer, expert)` 的常量，逐行存 16 个 float 是纯浪费
（800K 行 × 64 B = 51 MB）。它单独落在 `gate_bias.parquet`（`layer, expert, bias`，
15,360 行）里，`top16_scores + bias` 就能复原选路分数。`gate_bias_applied` 保留的是
真正有信息量的那一位。

### 附带文件

| 文件 | 内容 |
|---|---|
| `gate_bias.parquet` | `layer, expert, bias`，40 × 384 行 |
| `engram_rows.parquet` | `layer, prompt_id, pos, rows[24]`，层 1 与层 14 每 token 查的 24 个行号——§9.5 的 EngramPrefetcher 要下单的就是这些地址（Q7 规划用） |
| `meta.json` | 语料成分、逐层耗时、配置 |
| `state.pt` | 断点续跑用；trace 跑完可以删 |
| `token_map.npz` | 压缩词表映射的缓存（纯函数，删了会重算 ~20 s） |

### 没有的东西：DSpark 草稿位置

任务里写的是"**如果便宜的话**顺便记 DSpark 的草稿位置"。不便宜：

- 草稿输入要主模型第 37/38/39 层 attention **输入**的 hc 均值（15360 维）过
  `main_proj`，而且 `DSparkAttention` 走的是 `start_pos > 0` 的 decode 分支——它要求
  window KV cache 是逐步填起来的，正好和这里的按层流式结构相反。
- 参考实现的 `generate.py` **根本没调用** `forward_spec`，验证循环要我们自己写
  （design §10 明确说了），那是 §10 的工作，不是 §9 的。

替代：Q3 的 `union_frac[k]`（k = 2..5 个相邻 token 的 expert 并集 / 6k）已经给出了
verify batch 在 NVMe 上的边际成本，也就是 §10.3 调度曲线里 `T_nvme(k)` 要的量。差别是
它假设草稿全被接受；接受率低时并集只会更小，所以这是**保守**估计。

---

## 8. 模拟器方法

`tools/cache_sim.py`。cache key 是全局的 `layer * 384 + expert`，和 §5.2 把 slab 池当
成一个地址空间的做法一致。请求流按 `(prompt, pos, layer)` 排序，就是 decoder 会发出的
顺序。

### 容量

| 记法 | slot 数 | 由来 |
|---|---|---|
| `20,25,30,35` | 3072 / 3840 / 4608 / 5376 | 15,360 个 routed expert 的百分比 |
| `abs:4787` | 4787 | 90 GB slab 池 / 18.8 MB，§5.2 的实际容量 |

### 策略（§9.3）

- `lru`
- `lfu-decay`：计数带指数衰减，同分按 last-use 破平（即等计数时退化成 LRU）
- `arc`：标准 ARC，T1/T2 + B1/B2 ghost list，`p` 自适应
- `score-aware`：§9.3 的基线。router 每层都产出 top-16 分数；落在 top-16 却不在 top-6
  的 expert 是"近似命中"，即使没被读也刷新 `heat`。淘汰按 `(heat, last_use)`
- `static-pin+lru`：pinned 集合从 trace 的**前 25%** 选（`--pin-train-frac`），命中率
  在其余部分上报——用全量选 pinned 会变成 oracle，把策略吹得虚高

`--allocation per-layer` 把容量按层均分成 40 个独立池。Q2 决定它值不值：只有当各层的
重用距离分布差异大到"全局 LRU 会让一层的工作集把另一层挤掉"时才值。

### 时序模型（§9.4 / §9.2.1）

刻意做得粗但写明白：

- 一层占 `--t-layer-ms`（默认 1.3 ms，§3.4）的 GPU 时间，所以提前 d 层发出的预取有
  `d × t_layer` 的时间落地；
- NVMe 以 `--nvme-gbps`（默认 4.5，§9.2.1 实测 4.5–4.75）服务，每个请求下限
  `--t-io-ms`（默认 4.0 ms，§9.2.1 实测 4.0–4.7），在途 `--io-qd`（默认 8，§9.2.1 定的
  目标）。一批 m 个请求耗时
  `max(t_io, m/qd × t_io, m × 18.8 MB / bw)`；
- demand miss（P0）抢占：同样的服务时间，但 GPU 停在那里等。

第 L 层的一次 miss，如果该 expert 在第 L−d 层被预取过且来得及落地，记为 **hidden**，
否则记为 **exposed**。`stall_ms` 就是 exposed 的那部分。

预取落地的槽以"探针"身份 admit 但不 touch（§9.4 最后一行），所以没被用到的预取是下一个
被淘汰的。

一层的 6 个请求是**一批**发出的：一个 miss 不会被同批兄弟的 admit 掩盖。这和 stack
distance CDF 假设的顺序访问差一点点（容量接近 6 时差不到一个百分点），`analyse_q2` 的
`note` 里写了。

### tok/s

就是 §3.1 的模型，NVMe 换成 §9.2.1 实测的 4.5 GB/s：

```
t_token = 42 ms                      # 常驻 8.5 GB / 200 GB/s
        + hit_bytes  / lpddr_gbps
        + miss_bytes / nvme_gbps
```

### Q1 / Q2 / Q3 / Q4

- **Q1** `analyse_q1`：逐层把 expert 按频次降序，报 top-{5,10,20,30,40,50}% 的覆盖率，
  外加 Gini 与熵。曲线接近对角线 = `noaux_tc` 没留下静态偏斜 = pinned 集合不值得做。
  另附 `gate_bias`：bias 改变 top-6 的 token 比例，这是"负载均衡器是不是在主动压平分布"
  最便宜的读数。
- **Q2** `analyse_q2`：**精确**重用距离，Fenwick 树按访问时刻计数，O(n log n)。
  `P(distance < C)` 就是容量 C 下 LRU 的命中率——所以一遍扫描给出所有容量的解析解，
  不用逐个容量模拟。循环在 Python 里，因此有 `--q2-max-accesses`（默认 4M 次访问；
  20K token × 40 层 × 6 = 4.8M，会截断，不是抽样）。
- **Q3** `analyse_q3`：逐层相邻 token expert 集合的 Jaccard，以及 k = 2..5 的
  `union_frac[k] = |k 个相邻 token 的 expert 并集| / 6k`——k token verify batch 相对
  k 个独立 token 的 NVMe 成本，即 §10.3 的 `T_nvme(k)`。
- **Q4** `analyse_q4`：对 trace 记下的每个 d，算 K ∈ {6,8,10,12,16} 的
  recall@K = `|预测 top-K ∩ 实际 top-6| / 6` 与 precision = 同样的交集 / K。

---

## 9. Smoke 切片结果

<!-- SMOKE_RESULTS -->

---

## 10. 全量运行的耗时外推

<!-- WALLTIME -->

---

## 11. 留给 design §9 的开放问题

<!-- OPEN_QUESTIONS -->
