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

```
The expert cache holds about thirty percent of the routed experts, so the decode
loop is bounded by NVMe rather than by memory bandwidth.
这台机器的真实瓶颈是 NVMe，而不是 LPDDR 带宽。
def hit_rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0
```

结果（`traces/verify/verify.json`，2026-09-14）：

| # | 检查 | 结果 | 说明 |
|---|---|---|---|
| 1 | Engram 压缩词表大小 | **99,092 = config** | `NgramHashState.__init__` 自己 assert 的。所有 hash 乘子都由这个数派生，normalizer（NFKC → NFD → StripAccents → Lowercase → 空白归一）错一点就对不上。这是 tokenizer 适配器唯一也是最强的独立验证 |
| 2 | manifest run/skew vs `safetensors` 库 | **逐字节相同** | layer 0 expert 0 的六个 part 两条路各读一遍。复用了 `tools/oracle.py` L1 的对照组 |
| 3 | FP4 → bf16 快速解码 vs 浮点参考 | **逐位相同** | 真实 expert 权重，2304 × 5120 |
| 4 | Sinkhorn（20 轮，layer 0 真实系数） | 列和 \|Σ−1\| = **1.2e-06**，行和 \|Σ−1\| = **8.5e-02** | 列精确、行不精确，是因为 `hc_split_sinkhorn_kernel` 的循环**以列归一化结尾**。design §2.4 写的是"双随机"；参考 kernel 只是收敛到那里，并不落在那里。**这条要回写 design §2.4** |
| 5 | 我们的 top-6（按 `score+bias`）vs `Gate.forward` | **完全一致** | 每层每 token 都断言，不只 verify 时 |
| 5b | 路由权重之和 | **1.5000 – 1.5000** | 归一化 × `route_scale` ✓ |
| 5c | `noaux_tc` bias 改变 top-6 的 token 比例（layer 0） | **53.1%** | bias 在**主动**改选路，不是摆设。Q1 要认真回答 |
| 6 | teacher forcing 下的 argmax | **3/4 命中语料自身的下一个 token** | 见下 |
| — | 64 token × 40 层完整前向 | **320 s** | |

逐位置的预测：

| pos | 上文（尾部） | 语料的下一个 token | argmax | margin |
|---|---|---|---|---|
| 16 | `…he routed experts, so the decode loop is` | `' bounded'` | `' only'` | **0.032** |
| 32 | `…这台机器的真实瓶颈是 NV` | `'Me'` | `'Me'` ✓ | 8.688 |
| 48 | `… LPDDR 带宽。\ndef hit_rate(hits: int` | `','` | `','` ✓ | 3.78 |
| 62 | `…return hits / total if` | `' total'` | `' total'` ✓ | 6.242 |
| 63 | `…return hits / total if total` | — | `' else'` | 4.043 |

三种语域（英文散文、中文散文、Python）都给出连贯且自信的续写；`return hits / total
if total` → `' else'` 是唯一正确的 Python 续写。唯一一处不匹配（pos 16，`' only'` 而不是
`' bounded'`）margin 只有 **0.032**——两个候选在 logit 上几乎并列，而且 `' only'` 在英文
里同样成立。这正是 design §12 L3 说的"允许在极低 margin 处出现分歧并记录 margin"。

### 为什么没有跟 `inference/` 逐层对拍

不能。`inference/model.py` 的六个 kernel 是 tilelang 写的，需要 CUDA；这台机器上没有能跑
它们的 GPU（design §12 开篇就说了"没有可用的官方 CUDA 环境"）。所以 **kernel shim 的
等价性靠源码评审 + 上面这些不变量，不靠 A/B**。已知的未验证假设两条：

1. `sparse_attn` 的 fp32 累加等价于 kernel 的 "bf16 gemm + fp32 累加器"（推理如
   §4，但没测）；
2. FP4 的 nibble 顺序（低半字节 = K 方向偶数元素）。design §12 把它记为 L2 以下无法证伪
   的假设，这里同样成立——但如果它错了，上面 6 的连贯续写不可能出现，所以这算一次**弱但
   真实的**验证。

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

默认来源（`tools/corpus.py:default_sources`，本机实际命中 55 个文件）：

| kind | 来源 | 许可 |
|---|---|---|
| zh | `docs/design.md`、`docs/architecture.md`、`docs/build.md`、`README.md` | 本仓库，MIT |
| en | checkpoint 里的 `README.md`（model card）、`inference/README.md`、`encoding/README.md`、`evaluation/README.md`、`DeepSeek_V41_Tech_Report.pdf`（51 页，`pypdf` 抽文本） | DeepSeek-V4.1-Flash，MIT |
| code | 本仓库 `core/ storage/ store/ model/ gpu/vulkan/` 的 `.h`/`.cpp`；checkpoint 的 `inference/*.py`、`encoding/*.py`；CPython 标准库 10 个模块 | MIT / PSF |

默认参数（`--tokens 20000 --min-prompts 40 --min-len 128 --max-len 2048`）实际产出：

| | 值 |
|---|---|
| prompt 数 | **40** |
| token 数 | **27,399** |
| 长度 min / p50 / p90 / max | 107 / 606 / 1324 / 1580 |
| 长度分布（128–256 / 256–512 / 512–1024 / 1024–2048） | 5 / 9 / 14 / 10 |
| zh / en / code 的 token 占比 | 0.386 / 0.342 / 0.272 |
| 用到的不同源文件 | 19 |

token 数超过 20K 是因为"每个 kind 至少 ⌈40/3⌉ 个 prompt"这条下限比 token 预算更紧；
多出来的数据只有好处。

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
| `pred_d{d}_mean` | fixed_size_list\<int16\>[16] | 从第 L−d 层残差流、用 `ffn_norm(mean_hc(x))` 近似预测的本层 top-16（§9.4）。L < d 时整行填 **−1** |
| `pred_d{d}_pre0` | fixed_size_list\<int16\>[16] | 同上，近似换成 `ffn_norm(hc_pre(x, pre_identity))` |

> 预测列用 **int16 + −1 哨兵**而不是 null：全 null 的 `FixedSizeListArray` 过不了
> Parquet 往返（子数组是空的，reader 直接报
> `Expected all lists to be of size=16 but index 1 had size=0`）。expert id ≤ 383，
> int16 装得下，−1 不会与任何合法 id 混淆。

`pre_identity` 是 `make_identity_pre_mix` 的 one-hot，所以
`hc_pre(x, pre_identity)` 恰好等于残差流的第 0 份拷贝。两个近似都是 `[n, 5120]` 的
投影，快照因此是 `[n, 2, 5120]` 而不是整条 `[n, 4, 5120]` 流——这才让保留 8 个深度的
快照变得可负担（8 × 2 × 20K × 5120 × 2 B = 3.3 GB）。

快照取在**第 L−d 层 block 输出之后、第 L−d+1 层 engram（如果有）之前**，这就是 §9.4
描述的在线时刻：Planner 在第 L−d 层 FFN 算完时能看到的，正好是这条流。

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

> ## ⚠️ 这一节全部是 smoke 输出，**不是结论**
>
> 切片：**2,900 token / 6 个 prompt**（zh 43.6%、en 33.2%、code 23.2%），40 层全跑。
> 六个文档撑起来的语料自相似度远高于真实工作负载，**命中率几乎肯定被高估**。这里的数字
> 只用来证明"流水线端到端跑通、各个数量级对得上、内部一致"。Q1–Q4 的答案等
> ≥ 20K token / ≥ 40 prompt 的全量 trace，那时候才回写 design §9。

### Q1 静态频率偏斜

每层按频次降序，top-x% 的 expert 覆盖多少路由（40 层平均）：

| top 5% | top 10% | top 20% | top 30% | top 40% | top 50% |
|---|---|---|---|---|---|
| 0.388 | **0.534** | 0.704 | 0.806 | 0.877 | 0.926 |

- 每层激活的不同 expert：317–380（共 384）
- 熵 6.41–7.94 bit（均匀分布是 8.585 bit）
- Gini 0.50–0.80
- `noaux_tc` bias 改变 top-6 的 token 比例：**35.0%**（layer 0 是 53.1%）

**初步读数**：§9.1 Q1 猜的"`noaux_tc` 负载均衡可能让静态偏斜很小"在这份语料上**不成立**
——最忙的 10% expert 吃掉一半以上的路由，曲线离对角线很远。所以 §9.3 的 static-pin 档位
**值得保留**，别提前砍掉。但要小心：6 个文档的偏斜里有多少是**领域性的**而不是模型固有的，
只有全量语料能分开。

### Q2 重用距离

全局（(token, layer) 交错流，单一全局池看到的就是这个）：

| 容量（slot） | 768 | 1536 | 2304 | **3072 (20%)** | 3840 (25%) | **4608 (30%)** | **4787 (90 GB)** | 5376 (35%) | 7680 (50%) |
|---|---|---|---|---|---|---|---|---|---|
| LRU 命中率（解析） | 0.563 | 0.707 | 0.784 | **0.835** | 0.868 | **0.894** | **0.898** | 0.912 | 0.946 |

- 首次访问（冷）占 2.0%；重用距离 p50 = 562，p90 = 4,244
- **层内**重用距离（按层配额会看到的）：p50 = 8–48，p90 = 63–197

**内部一致性检查**：解析解 0.835 / 0.894 / 0.898 对上模拟器实测的 0.838 / 0.898 / 0.903
（20% / 30% / 4787 slot，global LRU）。两条完全独立的代码路径——一条是 Fenwick 树数
distinct key，一条是真跑一遍 OrderedDict——差 0.3–0.5 个千分点，就是"一层六个请求是一批
发出的"那点差别。这是本工具最强的自检。

**per-layer 配额值不值**：层内 p50 只有 8–48，而 global p50 / 40 层 ≈ 14，两者同一量级，
说明各层的工作集没有互相挤压到需要隔离。模拟器也是这么说的（下表）。

### Q3 相邻 token 的 expert 重叠

| | 值 |
|---|---|
| 相邻 token 的 Jaccard（40 层平均） | **0.234**（6 个里约 1.4 个相同） |
| 逐层范围 | 0.091（层 0）– 0.340（层 25） |
| `union_frac[2]` | 0.827 |
| `union_frac[3]` | 0.735 |
| `union_frac[4]` | 0.676 |
| `union_frac[5]` | **0.631** |

`union_frac[k] = |k 个相邻 token 的 expert 并集| / 6k`。k = 5 的 verify batch 读
0.631 × 30 ≈ 19 个 expert，而不是 30 个——**NVMe 流量省 37%**。这就是 §10.3 调度曲线里
`T_nvme(k)` 要的输入。注意它假设草稿全被接受，是保守端。

### Q4 Lookahead recall（`pre-identity` 近似）

recall@K = |预测 top-K ∩ 实际 top-6| / 6：

| d \ K | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|
| **1** | 0.658 | 0.728 | 0.770 | 0.797 | **0.834** |
| **2** | 0.576 | 0.642 | 0.686 | 0.717 | 0.760 |
| **3** | 0.521 | 0.581 | 0.623 | 0.655 | **0.700** |
| **4** | 0.479 | 0.535 | 0.575 | 0.606 | **0.652** |
| 5 | 0.442 | 0.494 | 0.533 | 0.563 | 0.608 |
| 6 | 0.413 | 0.463 | 0.499 | 0.528 | 0.573 |
| 7 | 0.390 | 0.437 | 0.472 | 0.501 | 0.545 |
| 8 | 0.374 | 0.420 | 0.454 | 0.482 | 0.526 |

- 深度每加一层，recall 掉 **4–5 个点**，一直到 d = 8 都没有拐点。
- §9.4 推出的 **d ≥ 3–4**（`d × T_layer ≥ T_io × misses / QD`）落在 recall@16 = 0.65–0.70，
  precision（= recall × 6/16）只有 0.24–0.26。也就是说 **K = 16 的预取里四分之三是浪费**。
  在 4.5 GB/s 的盘上，每层多读 10 个没用的 expert = 188 MB = 42 ms，比它能省下的还多。
  §9.4 的"precision < 阈值就缩小 K"不是可选的自适应，而是**默认就该开着**。
- 两个近似的对比（在早期 10 层切片上测的）：`ffn_norm(hc_pre(x, pre_identity))`
  比 `ffn_norm(mean_hc(x))` 好 6–8 个点（d=1、K=16：0.72 vs 0.62），而且它就是残差流的
  第 0 份拷贝，**零计算**。§9.4 应该直接选它。

### 策略对比（d = 0，无预取）

| 容量 | 分配 | LRU | LFU-decay | ARC | score-aware | static-pin+LRU |
|---|---|---|---|---|---|---|
| 20% (3072) | global | **0.8381** | 0.8381 | 0.8381 | 0.8161 | 0.8117 |
| 20% | per-layer | 0.8304 | 0.8314 | 0.8304 | 0.8338 | — |
| 25% (3840) | global | **0.8715** | 0.8715 | 0.8715 | 0.8398 | 0.8592 |
| 30% (4608) | global | **0.8979** | 0.8978 | 0.8979 | 0.8604 | 0.8917 |
| 30% | per-layer | 0.8915 | 0.8917 | 0.8915 | 0.8940 | 0.8862 |
| 35% (5376) | global | **0.9169** | 0.9168 | 0.9169 | 0.8747 | 0.9145 |
| **4787（90 GB，§5.2 的真实容量）** | global | **0.9027** | 0.9026 | 0.9027 | 0.8658 | 0.8985 |

对应的 §3.1 tok/s（42 ms 常驻 + 命中走 LPDDR + miss 走 4.5 GB/s NVMe）：
20% → 4.48，30% → 6.07，4787 slot → **6.25**，35% → 6.85。

**初步读数（重申：smoke）**：

1. **LRU、LFU-decay、ARC 三者在四位小数上几乎不可分。** 这份 trace 里没有 ARC 的
   scan-resistance 能吃到的结构。如果全量 trace 也是这样，§9.3 应该直接定 LRU——最简单的
   那个，`store/planner.cpp` 也最好写。
2. **score-aware 反而更差**（global 下 0.866 vs 0.903）。"top-16 近似命中刷新 heat" 把
   热度摊平了：384 个 expert 里有 16 个每层都在刷，真正的 LRU 信号被淹没。per-layer 下它
   略胜（0.8940 vs 0.8915），因为池小、16 个近似命中占比更大。**§9.3 把 score-aware 当基线
   是可疑的**，全量 trace 要重点看这一项。
3. **global 略胜 per-layer**（0.8979 vs 0.8915 @ 30%），和 Q2 的层内重用距离一致。
   §9.3 说"先用全局 LRU，若 Q2 显示各层差异大再改按层配额"——目前看不需要改。
4. **static-pin + LRU 不如纯 LRU**（0.8985 vs 0.9027 @ 4787）。Q1 有明显偏斜，但那份偏斜
   **LRU 自己就抓住了**；把 10% 的槽冻结成静态集合，只是减少了 LRU 能用的空间。
5. **命中率远高于 §3.1 表里假设的 0.30。** 如果全量 trace 守得住 0.85–0.90，§3.1 的
   TPS 表要整体上修，第一阶段目标（§1.3）也要重估。**但先别信** —— 6 个文档的语料
   自相似度太高，这个数字最可能是被高估的那个。

---

## 10. 全量运行的耗时外推

smoke 切片实测（2,900 token × 40 层，`--checkpoint-every 1`，
`meta.json` 里有逐层数据）：

| 阶段 | 总计 | 每层均值 | 随 token 数变化？ |
|---|---|---|---|
| 权重装载 + 绑定 | 21.7 s | 0.56 s | 否 |
| attention（含层 1/14 的 engram 行读取） | 104.7 s | 2.68 s | 是 |
| gate（384 个分数，全 token） | 2.2 s | 0.06 s | 是 |
| lookahead 预测（8 深度 × 2 近似） | 20.3 s | 0.52 s | 是 |
| **expert（I/O + 解码 + GEMM）** | **649.5 s** | **16.65 s** | **主要不变** |
| checkpoint（`state.pt` 665 MB） | 32.0 s | 0.82 s | 是 |
| 合计 | 839 s（14 分钟） | 21.5 s | |

关键结构：**expert 阶段是固定成本，跟 token 数几乎无关。** 2,900 token 时每层已经激活
350/384 个 expert；20K token 时是 384/384，多不了多少。每个 expert ≈ 16 ms 读
（1.2 GB/s，被预取线程盖住）+ 35 ms 解码（3 个矩阵），GEMM 只有几 ms。

### 外推到默认全量语料（27,399 token，9.45×）

| 项 | 每层 | 依据 |
|---|---|---|
| expert 固定（384 × 45 ms） | 17.3 s | 实测每 expert 成本 |
| expert 随 token（GEMM + act_quant + 累加） | ~9.5 s | 2,900 时约 1 s，线性 |
| attention（非 engram 层） | ~17 s | 1.8 s × 9.45 |
| lookahead | ~4.9 s | 0.52 × 9.45 |
| gate + 装载 | ~1.2 s | |
| checkpoint（`--checkpoint-every 4`，state ≈ 6.3 GB） | ~2 s 摊薄 | |
| **非 engram 层小计** | **~52 s** | |
| engram 层（1、14）额外 | **+63 s** 每层 | 658K 行 / 10,435 行每秒（见下） |

**总计 ≈ 38 层 × 52 s + 2 × 115 s ≈ 2,200 s ≈ 37 分钟。**
留出余量，报 **35–50 分钟**。`--checkpoint-every 1` 再加 ~4 分钟。

### 资源占用

| | 全量运行 |
|---|---|
| 峰值 RSS | ~8 GB（残差流 1.1 + 8 个 lookahead 快照 4.5 + y/ffn_in ~1.1 + 权重 ~0.7） |
| NVMe 读 | **~300 GB**（40 层 × 384 × 18.8 MB = 289 GB，加 attention 6.4 GB 与 engram 行） |
| 输出 Parquet | ~430 MB（切片 45.6 MB × 9.45），**远在 1 GB 以内** |
| `state.pt` | ~6.3 GB（跑完可删） |

### engram 行读取：已修，但还没进 smoke 数字

切片跑的时候 `WeightStore` 用一把全局锁保护单个文件句柄，24 行 × token 的 4 KiB 随机读
被压成 QD 1，层 1/14 的 attention 因此是 19–23 s 而不是 1.8 s。**已改成 per-thread 句柄**
（seek + read 不是原子的，所以共享句柄必须加锁；每线程一个句柄就不用锁），实测：

| 线程数 | 14,400 行 | IOPS |
|---|---|---|
| 1 | 7.15 s | 4,026 |
| 16 | 1.41 s | 20,395 |
| **32（默认）** | **1.38 s** | **20,866** |
| 48 | 1.42 s | 20,220 |

5.2× 提升。上面 §10 的外推用的是修好之后的数字（10,435 行/秒）；切片 `meta.json` 里
层 1/14 的 19–23 s 是**修之前**的，不要拿它外推。

仍然只有 §9.2.1 实测 83,700 IOPS 的四分之一——差的是 Python 每次调用的开销，不是盘。
runtime 的 `EngramPrefetcher` 走 IOCP，不受这个限制。

---

## 11. 参考实现与 design 的出入 / 需要回写的点

跑通这套东西的过程中，从 `inference/` 读出来的、和 design 现有措辞对不上或没写到的地方。
**我没有改 design.md**，列在这里等整合。

### 会影响 kernel 正确性的

1. **§2.4 "Sinkhorn（20 轮）得到 …（双随机）"——参考 kernel 出来的不是双随机。**
   `hc_split_sinkhorn_kernel` 的循环结构是
   `softmax(-1) + eps → /colsum → (iters−1)×(/rowsum → /colsum)`，**以列归一化结尾**。
   实测 layer 0 真实系数：列和 |Σ−1| = 1.2e-06，行和 |Σ−1| = **8.5e-02**。
   §7.2 的 Mega-mHC kernel 如果"顺手"在最后补一次行归一化，输出会和参考实现不同。

2. **§7.9 融合 FP4 MoE FFN 漏了中间激活的量化。** `Expert.forward` 最后一步是
   `self.w2(x.to(dtype))`，走 `linear()` → `act_quant(x, 32, "ue8m0", e8m0)`。也就是说
   `silu(gate) * up` 这个中间结果在进 `w2` 之前要**按行每 32 个元素量化成 fp8 E4M3、
   scale 向上取到 2 的幂**。§7.9 只写了"每层 2 个 dispatch"，没写这一步——它不是可选的
   数值细节，省掉会改变输出。

3. **§2.4 没写压缩 KV 与 indexer 的量化格式不同。** 压缩 KV 是
   `fp4_act_quant(latent, block=16, scale_dtype=E4M3)`，amax 下限 `6 × 2^-9`；
   indexer 的 q/k 是 `fp4_act_quant(x, block=32, scale_dtype=E8M0)`，amax 下限
   `6 × 2^-126`，scale 向上取 2 的幂。§11.3 的容量表（256 B + 32 B）和前者是一致的，
   但 §2.4 的"必须复刻的细节"清单里没有这条。

4. **`act_quant` 的 scale 是"向上取到 2 的幂"，且 amax 有 1e-4 下限。**
   `fast_round_scale` 用 fp32 位模式算 `2^ceil(log2(amax/448))`。§6（精度选择）没写这个
   舍入方向；用 `ceil(log2(x))` 的浮点写法在 x 恰好是 2 的幂时会错。

### 措辞需要收紧的

5. **§10.2 / §2.4 "第 37/38/39 层 attention 输入的 hc 均值"**：参考实现是
   `if i in target_layer_ids: main_hiddens.append(h.mean(dim=2))`，取的是**进入 block
   之前**的残差流（engram 之后、`hc_pre`/`attn_norm` 之前）在 hc 维上的均值，不是
   attention 那个 `[dim]` 输入的均值。两种读法都能从现在的措辞里读出来，实现会差很多。

6. **§9.4 的两个近似里，`ffn_pre(x, pre_identity)` 就是残差流的第 0 份拷贝。**
   `make_identity_pre_mix` 是 one-hot on index 0，所以
   `hc_pre(x, pre_identity) = x[:, 0]`，不需要任何计算。直接这么写更清楚，也说明这个近似
   比 `mean_hc` 便宜。

7. **§9.3 "router 每层输出 top-16 分数"**：参考 `Gate.forward` 只取 top-6。384 个分数
   本来就全算出来了（`sqrt(softplus(x·W))` 是一次 GEMV），所以 top-16 是**免费的**，
   但它是我们加的，不是参考实现的输出。

### 已经对上的（记一下，免得重复怀疑）

- 残差流 `[hc=4, 5120]`、pre 给下一个子层用 ✓
- gate 选路用 `score+bias`、权重用无 bias 的 score、归一化 `+1e-20`、× 1.5 ✓
  （实测权重和恒为 1.5000）
- expert 双侧/单侧 clamp 10 ✓；window KV 写入前 fp8 量化 ✓；输出先逆 RoPE 再进分组
  `wo_a` ✓；`attn_sink` 每头一个 logit ✓
- Engram 压缩词表 99,092 ✓；每 token 3 阶 × 8 头 = 24 行 ✓
- `compress_ratios` 中层 0/1 与 mtp 为 0 ✓
- §9.2.1 已经把 §3.1 的 5 GB/s 修正成 4.5 GB/s ✓（本文的 tok/s 模型用的是 4.5）

---

## 12. 留给 design §9 的开放问题

<!-- OPEN_QUESTIONS -->
