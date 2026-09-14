# P3 — DSpark 投机解码：验证过的算法规格（Track K → K2）

设计对应：[design.md](design.md) §2.4（DSpark 条）、§7.12、§9.5、§10（全部）、§12、§13.4。
本文是 **Track K / K2 交给 Track I 的集成规格**：草稿周期的每一步算什么、树采样与验证怎么写、
哪些 buffer 要存在、今天哪些 kernel 支持 M > 1、实测的接受率 / expert 并集 / 批边界 / CPU 代价，以及 TPS 投影。

> **K2 更正（2026-09-15）**：Track K 的验证循环用单条贪心链 + 精确前缀接受，这个采样方案对本模型是错的，
> 它在一条退化轨迹上得出的接受率、confidence 与加速结论（旧 §7.2、§7.5、§9）**作废**。
> 新方案与实测见 §3（规格）、§11–§15（实测与 go / no-go）。§1、§2、§4–§6、§7.1 / §7.3 / §7.4 / §7.6 / §7.7、§8 的 kernel 与规格部分仍然有效。
> **一句话结论**：树采样不比单链快（贪心 −0…−4%、温度 1 ±1%）；正常文本上单链接受率 2.6–3.3 tokens / verify，
> confidence 选 k 在实测 M 缩放下预测 ×1.18–1.19，**前提是 M > 1 的 attention / head / engram kernel 做出来**（§15.4）。

产物：

| 东西 | 路径 |
|---|---|
| oracle + 验证循环（Track K）+ 树采样轨迹驱动（K2，`--tree`） | `tools/oracle_dspark.py` |
| 树采样 / 比较的参考实现 + 离线评估 / 黄金数据 / 无损检验 / TPS | `tools/dspark_tree.py` |
| 黄金数据（position 64 全阶段 + 两个薄记录）；K2：`tree_golden.{bin,json}`、`tree_stats.json` | `tests/data/dspark/` |
| 草稿 kernel | `gpu/shaders/dspark_{common,gemv,attn,head}.slang`、`gpu/vulkan/dspark_kernels.{h,cpp}` |
| 树采样 CPU 实现（K2） | `cpu/dspark_tree.{h,cpp}`，`tests/test_dspark_tree.cpp`（`ctest -R suite.dspark_tree`） |
| 逐阶段正确性 | `tests/test_gpu_dspark.cpp`（`ctest -R suite.gpu_dspark`） |
| 带宽 / 时间 | `bench/dspark_bench.cpp` → `bench/results/dspark_p3.csv` |
| K2 轨迹与草稿矩阵（大，不入库） | `traces/dspark_tree/` |

复现（K2）：

```
.venv\Scripts\python.exe tools\oracle_dspark.py --model D:\models\DeepSeek-V4.1-Flash --tree --threads 16
.venv\Scripts\python.exe tools\dspark_tree.py analyse      # -> tests/data/dspark/tree_stats.json
.venv\Scripts\python.exe tools\dspark_tree.py tps
.venv\Scripts\python.exe tools\dspark_tree.py lossless
.venv\Scripts\python.exe tools\dspark_tree.py golden       # -> tests/data/dspark/tree_golden.bin
```

**参考实现从不跑投机路径**：`generate.py` 没有一处调用 `Transformer.forward_spec`，
`ModelArgs` 的注释也直说了（`model.py:129-130`：*"Only the forward pass is implemented here --
nothing calls forward_spec, so the speculative-decoding loop itself is out of scope for this repo."*）。
所以验证循环是我们写的，本文 §3 是它的规格，§4 是为了跑它必须对参考做的推广。

---

## 1. 草稿周期，按参考实现逐行

形状（全部来自 `config.json`，`tools/oracle_dspark.py` 启动时 assert）：
`dim` 5120、`hc_mult` 4、`n_heads` 64、`head_dim` 512、`rope_head_dim` 64、
`q_lora_rank` 1280、`o_lora_rank` 1024、`o_groups` 8、`window_size` 128、
`vocab_size` 129280、`rms_norm_eps` 1e-20、
**`dspark_block_size` = 5**、`dspark_noise_token_id` = 128799、
**`dspark_target_layer_ids` = [37, 38, 39]**、`dspark_markov_rank` = 256、
`num_nextn_predict_layers` = 3、mtp 的 MoE 是 **128 routed / top-3**。
三个 mtp 块的 `compress_ratios` 都是 **0** —— 纯滑动窗口 attention，
`rope_theta` = 10000 且 **YaRN 关闭**（`Attention.__init__` 里 `original_seq_len = 0`）。

### 1.1 输入：主模型的 37/38/39 层 block 输入

```python
# model.py Transformer.forward
for i, layer in enumerate(self.layers):
    if layer.engram is not None:
        h = layer.engram(...)
    # the MTP head reads the attention input of its target layers, not their output
    if i in self.target_layer_ids:
        main_hiddens.append(h.mean(dim=2))      # [b, s, 5120]
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
main_hidden = torch.cat(main_hiddens, dim=-1)   # [b, s, 15360]
```

取的是 **进入 block 之前** 的残差流（engram 之后、`hc_pre`/`attn_norm` 之前）在 hc 维上的均值。
层 37/38/39 都没有 engram（engram 只在 1 和 14），所以实现上就是"第 37 层 kernel 链开始之前
把 `[4, 5120]` 的残差流按 hc 求平均，写进 15360 维缓冲的第 0/5120/10240 段"。
design §2.4 的更正（对照 route_trace.md §11.5）在这里被再次确认：**不是** attention 那个
`[dim]` 输入的均值。两者差一次 `hc_pre` 加一次 `attn_norm`。

### 1.2 `forward_embed`

```python
main_x = self.main_norm(self.main_proj(main_hidden))       # [b, s, 5120]
draft_input_ids = full([b, 5], noise_token_id); draft_input_ids[:, 0] = input_ids
x = self.embed(draft_input_ids).unsqueeze(2).repeat(1, 1, 4, 1)   # [b, 5, 4, 5120]
```

* `main_proj` 是 fp8 `Linear` **[5120, 15360]**（manifest `mtp.0.main_proj.weight`，
  `fp8_e4m3`，scale plane `[160, 480]`）→ 激活要先 act_quant（block 32、UE8M0、2 的幂）。
  78.6 MB 权重 + 76.8 KB scale。
* `main_norm` 是 `RMSNorm(5120, 1e-20)`，bf16 权重。
* 草稿的 5 个输入 token 是 **[上一个已接受 token, noise × 4]**；embed 用主模型的
  `embed.weight`（`mtp[-1].embed = self.embed`），不是单独一份。
* 残差流一开始是 embed 的 **4 份拷贝**，`pre_mix` 是 one-hot on 0。

### 1.3 三个 `DSparkBlock`

每个块是普通 `Block.forward`，只是 attention 换成 `DSparkAttention` 并多收一个 `main_x`：

```
hc_mixes(attn) → hc_pre → attn_norm → DSparkAttention(x, start_pos, main_x) → hc_post
hc_mixes(ffn)  → hc_pre → ffn_norm  → MoE(128 routed, top-3, + 1 shared)     → hc_post
```

`DSparkAttention.forward`（start_pos > 0）：

```python
bsz, seqlen, _ = main_x.size()                 # 参考里恒为 1，见 §4.6
main_freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
main_kv = self.kv_norm(self.wkv(main_x))       # wkv fp8 [512, 5120]；kv_norm RMSNorm(512)
apply_rotary_emb(main_kv[..., -64:], main_freqs_cis)
act_quant(main_kv, 32, "ue8m0", e8m0, inplace=True)      # 环里存的是 fp8 舍入后的值

freqs_cis = self.freqs_cis[start_pos+seqlen : start_pos+seqlen+5]
qr = self.q_norm(self.wq_a(x))                 # wq_a fp8 [1280, 5120]
q  = self.wq_b(qr).unflatten(-1, (64, 512))    # wq_b fp8 [32768, 1280]
apply_rotary_emb(q[..., -64:], freqs_cis)
kv = self.kv_norm(self.wkv(x))
apply_rotary_emb(kv[..., -64:], freqs_cis)
act_quant(kv, 32, "ue8m0", e8m0, inplace=True)

topk_idxs = get_dspark_topk_idxs(128, bsz, 5, start_pos)
self.window_kv_cache[:bsz, start_pos % 128] = main_kv.squeeze(1)
kv = cat([self.window_kv_cache[:bsz], kv], dim=1)        # 128 + 5 = 133 行 × 512
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, 512 ** -0.5)
apply_rotary_emb(o[..., -64:], freqs_cis, inverse=True)
o = o.view(bsz, 5, 8, -1)                                # 8 组 × 4096
o = einsum("bsgd,grd->bsgr", o, self.wo_a.weight.view(8, 1024, 4096))
x = self.wo_b(o.flatten(2))                              # wo_b fp8 [5120, 8192]
```

### 1.4 `forward_head`（只在 stage 2）

```python
x = self.hc_pre(x, pre_mix)                              # [b, 5, 5120]
logits = self.head(self.norm(x), full_logits=True)       # head bf16 [129280, 5120]，fp32 算
output_ids[:, 0] = input_ids
for i in range(5):
    logits_bias, markov_embed = self.markov_head(output_ids[:, i])
    logits[:, i].add_(logits_bias)
    output_ids[:, i + 1] = sample(logits[:, i], temperature)      # temperature = 0 → argmax
confidence = self.confidence_head(x, stack(markov_embeds, dim=1))
```

* `markov_head.embed` bf16 **[129280, 256]**（查 1 行 = 512 B）；
  `markov_head.head` bf16 **[129280, 256]** = **66.2 MB**，每个草稿位读一遍。
  **design §7.12 写的是 33 MB（相当于每元素 1 字节）；checkpoint 里这两张表都是 bf16，66.19 MB**（manifest 实测）。
* `logits[:, i]` 用的是 **position i 的 head 输出** 加上 **position i 的输入 token（即上一位的采样结果）**
  的 Markov 偏置。所以五步是**严格顺序**的：`output_ids[i+1]` 要等 `output_ids[i]` 出来。
* `confidence_head.proj` 在 checkpoint 里是 **bf16 [1, 5376]**，模块声明是 fp32，
  输入 `cat(x, markov_embed)` 先 `.float()`。

### 1.5 每个草稿周期真正读的字节

全部按 `deepmoe_manifest.json` 的实际字节算（权重 + scale plane）：

| 项 | 大小 | 说明 |
|---|---|---|
| `main_proj` + `main_norm` | 78.73 MB | fp8，每周期 1 次 |
| 3 × attention（wq_a 6.56 + wq_b 41.98 + wkv 2.62 + wo_a 33.59 + wo_b 41.98 MB） | 380.2 MB | fp8，含 scale plane |
| 3 × shared expert（w1/w2/w3 fp8，11.81 MB 各） | 106.3 MB | |
| 3 × gate(1.31) + hc_fn(2 × 1.97) + norms | 15.8 MB | `hc_*_fn` 是 24 × 20480 fp32 = 1.966 MB，一块两份 |
| `head` | 1323.8 MB | bf16，M = 5 只读一次 |
| `markov_head.head` × 5 位 | 331.0 MB | bf16 66.19 MB × 5，**严格顺序**，不能合并 |
| `markov_head.embed` × 5 行 | 2.5 KB | 66.19 MB 的表只查 5 行 |
| `confidence_head.proj` | 10.8 KB | |
| **不含 routed expert 合计** | **2236 MB = 2.24 GB** | |
| 3 × routed expert，5 token × top-3 的并集 | **见 §7.3**（实测 489 MB） | fp4，每个 expert 18.81 MB |

mtp 的非 expert 权重一共 **713.4 MB**（全部 pin），routed expert **7.22 GB**（384 个）。
design §2.2 的 7.9 GB = 7.22 + 0.71，对得上。

design §7.12 估的是"草稿周期总流量 ≈ 2.2 GB ≈ 11 ms"，固定部分对得上 ——
但那个估计里 Markov head 按 **33 MB × 5** 记（"`head_m [129280 × 256]` GEMV 33 MB"），
checkpoint 里这张表是 **bf16 66.19 MB**，五次就是 331 MB 而不是 165 MB。
差的 166 MB 在 §8 的 `T_draft` 里按实测带宽折算。

mtp 的 384 个 routed expert 一共 **7.2 GB**，design §2.2 要求全部 pin。这是对的：
每周期最多碰 3 × 15 = 45 个（§7.3 实测 26 个），但它们是随机的 —— 从 NVMe 取一个
18.8 MB 的 expert 要 4.26 ms（§5.1.3），而整个草稿周期的预算是 11 ms。

---

## 2. 参考实现里九个容易做错的点（Track K 八个 + K2 一个）

1. **`main_hidden` 取的是 block 输入，不是 attention 输入**（§1.1）。已在 design §2.4 更正，
   这里只是确认：`model.py:1265-1266` 的 `if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))`
   在 `h, pre_mix = layer(...)` **之前**。

2. **草稿块内部没有因果掩码。** `get_dspark_topk_idxs` 对 5 个 query 返回**同一行**索引：

   ```python
   matrix = cat([arange(min(window_size, start_pos + 1)), window_size + arange(block_size)])
   return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1)
   ```

   即第 0 个草稿位也能看到第 4 个草稿位的 KV。这不是 bug，是 MTP 的设计（5 个位置里有 4 个是
   noise token，块内是双向的）；**kernel 不要"顺手"加下三角掩码**，否则输出和参考不同。

3. **窗口 KV 环里存的是 `main_x` 推出来的 KV，不是草稿 token 的 KV。**
   每个 mtp 块有自己的 `window_kv_cache[128, 512]`，prefill 时由 `main_x` 的全部位置填满，
   decode 时每个**已接受**位置写一格。草稿位自己的 5 行 KV 是**每周期重算、不入环**的。
   → 三个 mtp 块各 128 × 512 × 2 B = **131 KB**，共 393 KB 常驻状态，
   而且它 **属于 KV cache，拒绝草稿时要一起回滚**（§3.4）。

4. **confidence head 没有激活函数。** `DSparkConfidenceHead.forward` 是
   `self.proj(hidden.float()).squeeze(-1)` —— 一个裸的 fp32 线性层。
   design §10.3 把 `c_j` 当作条件接受概率用，所以中间缺一个映射。
   单 logit 头唯一自洽的映射是 logistic，`tools/oracle_dspark.py::conf_to_prob` 就这么做，
   并且**把它标成假设**；`stats.confidence_vs_accept` 是支持/反对它的证据（§7.5）。

5. **Markov head 的两张表都是 bf16，不是 fp8**（§1.4），design §7.12 的 33 MB 要改成 66.2 MB。

6. **`wo_a` 依然是那个例外**：checkpoint 里是 fp8 字节，`convert.py` 反量化成 bf16，
   `DSparkAttention.forward` 用 `einsum` 取权重而不走 `linear()`，所以**它前面没有 act_quant**。
   其余 `main_proj`/`wq_a`/`wq_b`/`wkv`/`wo_b`/`shared_experts.w{1,2,3}` 全都有（design §2.4 v0.7）。

7. **`main_x` 的位置是 `start_pos`，草稿位是 `start_pos + seqlen ...`。**
   `main_freqs_cis = freqs_cis[start_pos : start_pos+seqlen]`，
   `freqs_cis = freqs_cis[start_pos+seqlen : start_pos+seqlen+5]`。
   在参考的 `seqlen == 1` 下就是：主位置 `p`，草稿位置 `p+1 .. p+5`。
   而 `output_ids[0] = input_ids` 是**位置 `p+1` 上的那个 token**，
   `logits[:, i]` 预测的是位置 `p+2+i`。也就是说 **5 个草稿 token 覆盖位置 `p+2 .. p+6`**。

8. **`DSparkAttention` 的代码本身已经是按 `seqlen > 1` 写的，只有环写入那一行不是。**
   `main_freqs_cis`、草稿位的频率偏移、`n_win` 的计数全都从 `seqlen` 推出来；
   卡住的只有 `window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)` 和
   `get_dspark_topk_idxs` 里的 `min(window_size, start_pos + 1)`。§4.6 给出推广。
9. **（K2）Markov 偏置不是小修正，base logits 单独不含序列信息。** `forward_head` 的 `logits` 在循环前一次算出、与采样无关（§3.1），
   但链 token 在未加偏置的行里中位排名 1–13、p90 138–268（§12.1）。任何"只看 B 选候选"的做法都会失败。

---

## 3. 验证循环：单遍、单路径的树采样方案（Track K2，取代 Track K 的单链方案）

> **更正（Track K2）**：Track K 的循环在草稿里按 `forward_head` 逐位 argmax 出一条链，verify 时做精确前缀比较。
> 对这个模型这个采样方案是错的：温度 0 下它只是本节方案的一个特例（锚链），温度 1（`generate.py` 默认）下它根本不是无损的投机采样。
> 旧 §7.2（接受长度）、§7.5（confidence）、§9（调度与加速）的结论**作废**，由 §11–§15 取代。

方案一句话：**草稿一次 forward、verify 一次 forward，两者的 GPU 代价与单链方案完全相同；只有 CPU 上的采样与比较变丰富，代价是微秒级**。

### 3.1 草稿是一个矩阵（代码证据 + 实测）

* `DSparkBlock.forward_head`（`model.py:1137-1156`）：`logits = self.head(self.norm(x), full_logits=True)` 在采样循环**之前**一次算出 `[5, V]`；
  循环里只有 `logits[:, i].add_(logits_bias)` 与 `sample(...)`。
* 草稿输入恒为 `[上一个已接受 token, noise × 4]`（`forward_embed`），不含任何草稿 token。
* 草稿块内**没有因果掩码**：`get_dspark_topk_idxs` 对 5 个 query 返回同一行（§2 第 2 点），
  所以 5 个位置的残差流 x 与 base logits `B[5, V]` 与采样结果无关。
* 唯一与 token 有关的是 Markov 偏置 `bias(prev) = H · E[prev]`（`DSparkMarkovHead`：`embed` 与 `head` 都是 `[129280, 256]`，`ParallelHead` 无 bias）
  和 confidence head 的输入 `cat(x_i, E[prev_i])`。
* 实测：在 CPU 上用 `B + H·E[prev]` 逐位 argmax 重放参考链，**全部 705 个草稿 token 与 `forward_head` 的 `output_ids` 相同**（§12.1）。

**但 B 本身几乎不含序列信息**（K2 第一次试跑就撞上，§12.1）：参考链的 token 在 B 行里的中位排名（位 0–4）是 1 / 3 / 8 / 13 / 11、p90 是 138–268，
落在 B 的 top-16 里的只有 74% / 67% / 62% / 55% / 54%，而且"前缀全在候选里"才有用。Markov 偏置（std ≈ 3–4，与 B 同量级）不是小修正。
所以"每位直接取 B 的 top-K"这个候选规则**不可用**（`base` 规则，§12.2：贪心 k = 5 的 `E[tokens]` 从 3.93 掉到 2.65；K2 第一次试跑用它，两个周期接受 0）。

### 3.2 候选与格（CPU）

**候选规则 `anchor`（默认）**：GPU 仍按参考顺序做 5 次 Markov GEMV + 加偏置（锚 = 位 0 用输入 token，位 i 用位 i−1 的 top-1，**正是参考的贪心链**），
只是每位把 argmax 换成 top-K，并读回：候选 id、**这些 id 上的 base logit**（加偏置之前的 B）、该行（锚行）的全词表 logsumexp。
草稿读的 GPU 字节与单链方案**完全相同**（§13.2）。候选 0 永远是链 token。

对比规则（离线）：`base`（B 的 top-K，零 Markov GEMV）、`unionM`（位 i 的行分别用位 i−1 的 top-M 个候选加偏置，轮转合并到 K 个；
一次 M 列 GEMM，读的还是那 66 MB，算力 ×M）。

**格**：位 i 的偏置矩阵 `bias[p][c] = E[C_{i−1}[p]] · H[C_i[c]]`（K × K 个 256 维点积，只用 gather 出来的行；两张表在主机上各 66 MB）。

```
score[i][p][c] = B_i[C_i[c]] + bias[p][c]            位 0 只有一行 p = 输入 token
tail_i         = log( exp(lse_anchor_i) − Σ_c exp(score[i][anchor][c]) )    锚行扣掉 K 个候选，精确
lse_i(p)       = logsumexp( score[i][p][0..K−1], tail_i )                   tail 对所有 prev 共用（近似）
log q_i(c|p)   = score[i][p][c] − lse_i(p)
```

"每个 (位置, prev) 都做一次全词表 logsumexp"要 1 + 4K 次 GEMV；§12.3 实测它改变所选路径的频率与对接受率的影响。

**路径**（都输出**一条** ≤ 5 的路径）：

| 目标 | 定义 | 代价 |
|---|---|---|
| `viterbi` | 最大联合 log q（5 层精确 DP） | O(5K²) |
| `eal` | 最大期望接受长度 `q0(1 + q1(1 + q2(1 + q3(1 + q4))))`（后向精确 DP） | O(5K²) |
| `chain` | 格内逐位 argmax（= 参考链） | O(5K) |
| `sample`（温度 1） | 格上祖先采样，条件分布在 K 个候选上归一化 | O(5K) |

beam 不需要：K ≤ 32 的 5 层格精确 DP 就是 C++ 里 < 0.2 ms（§13.1）。

**confidence 与 k**：`conf_i = proj(cat(x_i, E[path_{i−1}]))`（`path_{−1}` = 输入 token），x 与格无关、只换 256 维那一段。
k 规则：**累计 σ(conf) ≥ θ 的最长前缀**（前缀规则：位 j 是否被验证只取决于位 < j，温度 1 的耦合评估因此是精确的）。θ ∈ {0.3, 0.5, 0.7}。

### 3.3 比较（CPU，verify 矩阵）

verify 一次 forward `[last, path[:k]]`，M = k + 1，**与单链方案同一批 token 数、同一份 GPU 字节**。

**贪心（温度 0）**：每行只要 argmax。`a = 最长的 j 使 path[j] == argmax[j]`，产出 `path[:a] + [argmax[a]]`。

**采样（温度 1）——两种截断规则，都实现了（Python 与 C++ 逐位一致）**：

1. **`accept_sampling`：top-Kv 截断。** 目标 `p~_j = softmax(verify 行的 top-Kv logits)`，提议 `q_j(·|prev)` = 格的条件分布（K 个候选上归一化，这对 q 是精确的——采样器不会提议别的）。
   接受 `u·q(x) < p~(x)`；拒绝时从 `max(0, p~ − q)`（支撑在 top-Kv 上）采样；全接受时从 `p~_k` 采 bonus。
   **对 p~ 严格无损**（§13.3 频率检验），但 p~ **不是** `generate.py` 的分布：top-32 之外的尾部质量实测均值 3.3%、p90 12.6%、最坏 42%（§13.3），
   而且截断会抬高同一行上的接受率。**不推荐作为默认。**
2. **`accept_sampling_exact`：对温度 1 严格无损（默认）。** 每个 verify 行多要三样东西（都在 head 已经算出的那一行上一遍扫描得到，候选 id 在 verify 之前就已知）：
   K 个候选 `C_j` 上的 logit、全行 logsumexp、一个把 `C_j` 屏蔽掉的 Gumbel-max 样本（最后一行再要一个不屏蔽的样本）。
   ```
   p(c) = exp(l_c − lse) (c ∈ C_j)，   q = 格条件分布（C_j 外为 0）
   接受 x      ⟺ u_acc · q(x) < p(x)
   拒绝：w_c = max(0, p(c) − q(c))，Z = Σw + (1 − Σ_c p(c))，t = u_res · Z
         t < Σw → 在 C_j 里按 w 取；否则 → 取屏蔽样本（它正是 p 在 C_j 之外的条件分布）
   全接受      → bonus = 最后一行的整行样本
   ```
   残差 `max(0, p − q)` 在 `C_j` 之外就是 p 本身，所以这是标准投机采样的逐字实现，**对温度 1 精确无损**。

### 3.4 离线评估所有方案（一次 forward 都不多跑）

所有方案共用一条轨迹（每个 prompt × 模式一条），由真实投机循环产生（驱动方案：K = 16，贪心 `eal` / 采样 `sample` + `accept_sampling_exact`，k = 5）。

* **贪心**：给定已接受前缀，verify 行的 argmax 就是轨迹的下一个 token，所以任一路径的接受长度 = 它与轨迹逐位相同的最长前缀。
  （前提是批边界不改变 argmax——§14 实测。）
* **采样**：对任何无损方案，`P(接受, 输出 y) = min(p(y), q(y))`、`P(输出 y) = p(y)`，所以沿一条从 p 采出来的轨迹，
  `P(接受 j | y_j, 前面都接受) = min(1, q(y_j | y_{j−1}) / p(y_j))`，
  `E[接受 | 轨迹] = Σ_j Π_{i≤j} r_i` 精确（Rao-Blackwell，比实际掷骰的方差小）。确定性路径（q = 点质量）时 `r_i = [y_i == path_i]`。
  p 用 verify 行里被发出 token 的精确 logit 与全行 lse（`emit_logit`）。
* 草稿在**每个被接受的位置**都跑一次（参考的 M = 1 顺序：`forward_spec(p+1 处的 token, main_hidden[p], start_pos = p)`），
  所以每个轨迹位置都有一个草稿矩阵；只统计后面 5 个 token 都已知的草稿事件。

### 3.5 位置对齐与回滚

对齐与旧 §3.2 相同：verify 从 `p` 开始、接受 a，则草稿用 `main_hidden[p + a]`、`start_pos = p + a`、输入是 `p + a + 1` 处的 token。
runtime 每周期只跑**一次**草稿，并把 `a + 1` 个已接受位置的 `main_x` 写进 mtp 环（§4.6）；oracle 为了数据在每个位置各跑一次（同样写满环）。

**不需要重跑主模型。** 被拒位置写脏的状态分三类：
ratio-2 的 `kv_state` 槽在所属组 pool 之前一定被逐 token 重写；压缩行只在组完成后可见，完成时重写同一行——这两类**不用回滚**。
**窗口环要回滚**：环写满（位置 ≥ 128）之后，被拒位置 `p' ∈ (p+a, p+k]` 写进的槽 `p' mod 128` 原本存着 `p' − 128` 的 KV，
而 `p' − 128` 仍在后续 query 的窗口内；若下一批没有覆盖到 `p'`（下一周期 k 更小），它就被读错。
所以 runtime 在 verify 前快照这 ≤ k 个槽（40 层 × 5 × 1024 B ≈ 200 KB）、拒绝后恢复；mtp 环只写已接受位置，不涉及。
K2 的 oracle 恒为 k = 5、位置 < 128，下一批总是先覆盖全部脏槽再读，所以没有快照、没有重跑（旧 §3.4 的"快照 + 重跑前缀"不再使用）。
唯一跨 forward 的状态是 `shared_attn.index_k`，它正是批边界效应的来源（§14）。

---

## 4. 参考实现跑不了 verify batch：`seqlen == 1` 的五处硬编码

这是本轮唯一一处我们对参考做的推广，也是 Track I 要在 kernel 里实现的语义。
`tools/oracle_dspark.py::install_batched_decode` 把它们逐个替换，
每一处都写成 **`seqlen == 1` 时逐位等于参考、`start_pos == 0` 时原样调用参考的 prefill 分支**。

### 4.1 `Attention._window_kv` 的环写入

参考：`self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`
推广：`for j in range(seqlen): cache[(start_pos + j) % win] = kv[:, j]`

### 4.2 `get_window_topk_idxs` 的逐 query 可见性

参考（decode 分支）是一行、全部槽位：

```python
oldest = start_pos % window_size + 1
idxs = cat([arange(oldest, window_size), arange(oldest)])
idxs = where(idxs > start_pos, -1, idxs)        # 环还没填满
```

推广要按 query 回答同一个问题。批量写入 `seqlen` 个之后，最新位置是
`q = start_pos + seqlen - 1`，于是槽 `s` 持有的是**不超过 q 的、模 win 同余于 s 的最大位置**：

```
p(s) = q - ((q - s) mod win)          # < 0 表示从没写过
槽 s 对位置 P 的 query 可见  ⟺  0 <= p(s) <= P
```

`p(s) > P` 这一支是新的，也正是全部要害：**同一批里更晚 token 的槽在更早的 query 跑的时候
已经写进去了，因果性要求它们不可见**。`seqlen == 1` 时这个集合与参考逐元素相同
（顺序不同，但 `sparse_attn` 独立处理每个槽，`model.py` 的 docstring 自己也这么说）。

### 4.3 `Compressor.forward` 的分组

参考：`slot = start_pos % ratio`，`should_compress = (start_pos+1) % ratio == 0`。
推广：对 `j = 0..seqlen-1` 逐个填 `(start_pos+j) % ratio`，
每当 `(start_pos+j+1) % ratio == 0` 就 pool 一组，最后 `stack` 起来。
M = 6、ratio = 2 时一批最多完成 3 组。ratio == 1 的分支参考本来就是全批的。

### 4.4 压缩 latent 的 RoPE 频率

参考：`freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)`
推广：`freqs_cis[[start_pos + j + 1 - ratio for j in range(seqlen) if (start_pos+j+1) % ratio == 0]]`
—— 一个 latent 代表它那组的**第一个** token。cache 的写入下标 `start_pos // ratio` 参考的写法已经是对的。

### 4.5 `Indexer` 的逐 query `compress_lens`

参考 decode 分支用标量 `end_pos // ratio` 且**不做掩码**（因为只有一个 query，
`index_score` 的列数正好等于它能看到的组数）。推广：

```python
compress_lens = ((start_pos + arange(1, seqlen + 1)) // ratio).unsqueeze(-1)   # [M, 1]
index_score.masked_fill_(arange(index_score.size(-1)) >= compress_lens, -inf)
```

`seqlen == 1` 时掩码恒为 False，与参考相同；`start_pos == 0` 时它就是参考 prefill 分支的
`(arange(1, seqlen+1) // ratio)`。`select_candidate_blocks` 和末尾的
`where(idxs < compress_lens, idxs + offset, -1)` 都直接吃这个 `[M, 1]` 张量（prefill 分支本来就是）。

### 4.6 `DSparkAttention` 的同一处

草稿块也要吃 `seqlen` 个已接受位置的 `main_x`（否则 mtp 的窗口环会漏掉被接受的中间位置）：

```python
n_win = min(win, start_pos + seqlen)                       # 参考是 start_pos + 1
matrix = cat([arange(n_win), win + arange(block_size)])
for j in range(seqlen): window_kv_cache[(start_pos + j) % win] = main_kv[:, j]
```

**这是 Track I 必须实现的接口变化**：`forward_spec` 一次吃 `a+1` 个已接受位置的 `main_hidden`，
而不是只吃最后一个。参考写不出来只是因为 `generate.py` 只有 M=1。

### 4.7 这套推广的锚点

`--crosscheck` 把同一个 verify batch 当成**整条序列的一次 prefill** 再跑一遍
（prefill 分支是原封不动的参考代码，而且 prompt 短于 128 槽窗口时两条路径语义完全相同），
比较最后 M 个位置的 logits。结果见 §7.1。
另外 Track K 每个周期的"恢复快照 + 重跑已接受前缀"也是同一件事的弱化版自检（K2 不再重跑，§3.5；批边界见 §14）。

### 4.8 批边界隔离：逐 token 语义的 indexer（K2）

`install_batched_decode` 的 `Indexer.forward` 多了一个开关 `ref._dm_emulate_decode`（`oracle_dspark.py::forward_emulated`）：
每个 query 记一个"它现在该用哪份 index key"。forward 开始时全部指向上一次 forward 留下的 `shared_attn.index_k`（最后发布者是层 20）；
key 源层（2 / 8 / 14 / 20）只对**本 query 的位置恰好完成一组**（`(P+1) % ratio == 0`）的 query 发布自己的 `k_cache`；
其余 query 保留原来的源。这就是"把这一批逐 token decode"时每个 query 看到的 key，一次 forward 算出来。
按源分组各做一次 einsum，行数与掩码不变。M = 1 时它与参考逐位相同（开关只在 `seqlen > 1` 生效）。

### 4.9 窗口环在回绕之后的一个缺口（K2 读代码发现，未实测）

§4.1 / §4.2 的推广是"先把 M 个槽全部写进环，再按 `p(s) ≤ P` 给每个 query 掩掉更晚的槽"。
**位置 < 128 时这是对的**（Track K 与 K2 的全部数据都在这个范围）。回绕之后，批里较晚的 token 写进的槽 `P' mod 128`
原本存着 `P' − 128`，而批里较早的 query（位置 `P < P'`）的窗口仍包含 `P' − 128`——被覆盖了，掩码只能把新值藏起来，旧值已经没了。
每个 query 最多丢 M − 1 个最老的窗口位置。runtime 的 M > 1 窗口 attention 要么先 attend 再写环，要么把被覆盖的 ≤ M − 1 行放进旁路缓冲。
§3.5 的"verify 前快照 ≤ k 个槽"正好就是这份旁路缓冲。

---

## 5. Buffer 与数据流（给 `runtime/`）

草稿周期新增的常驻状态：

| buffer | 大小 | 谁写 | 谁读 |
|---|---|---|---|
| `main_hidden[M][15360]` fp32/bf16 | 6 × 30 KB | 主模型层 37/38/39 的 `hc` 均值 | `main_proj` |
| `main_x[M][5120]` bf16 | 6 × 10 KB | `main_proj` + `main_norm` | 三个 mtp 块的 `wkv` |
| mtp `window_kv_cache[3][128][512]` bf16 | 393 KB | 每个已接受位置一格 | mtp 的 `sparse_attn` |
| 草稿残差流 `[5][4][5120]` bf16 | 205 KB | mtp 块 | mtp 块 |
| `draft_logits[5][129280]` fp32 | 2.59 MB | `head`（M=5） | Markov 加偏置 + argmax |
| `markov_embed[5][256]` bf16 | 2.5 KB | Markov embed 查表 | confidence head |
| `output_ids[6]` u32 | 24 B | 顺序 argmax | 下一周期的 verify batch |
| `confidence[5]` fp32 | 20 B | confidence head | 调度器 |
| 主模型 KV 快照（回滚用） | 见 §3.4 | 周期开始 | 拒绝时 |

dispatch 顺序（一个周期）：

```
main_proj(GEMV K=15360, M=1) → main_norm(RMSNorm)
for stage in 0..2:
    mega_mhc(attn) ×3 → hc_pre → attn_norm
    wkv(main_x, M=1) → kv_norm → RoPE(pos p..p+a) → act_quant → 写环
    wq_a(M=5) → q_norm → wq_b(M=5) → RoPE(pos p+a+1..p+a+5)
    wkv(x, M=5) → kv_norm → RoPE → act_quant
    dspark_attn(M=5, 64 头, n_kv = 133)
    inverse RoPE → wo_a(grouped bf16, M=5) → wo_b(M=5)
    hc_post
    mega_mhc(ffn) ×3 → hc_pre → ffn_norm
    gate(128 expert, top-3, M=5) → MoE gateup/down(M=5) + shared
    hc_post
hc_pre(final) → norm → head(M=5)
for i in 0..4:                      # 严格顺序
    markov embed 查 1 行 → markov head GEMV(129280 × 256 bf16) → 加偏置 → argmax
confidence head(5 × 5376 fp32 点积)
```


---

## 6. 今天哪些 kernel 支持 M > 1

这是 Track I 集成时最硬的约束，逐条查过：

| kernel | M > 1？ | 证据 |
|---|---|---|
| `moe_gateup.slang` / `moe_down.slang` | **支持，M = 1..6** | `MoeRunner::create` 里 `if (spec.m == 0 \|\| spec.m > 6) return fail(...)`；累加器是 `float accGate[RowsPerLane][M]`，activation 按 `X[m * (pc.k/8) + ...]` 取；`kernel_p2_moe.md` §3.5 有完整 M 扫描 |
| `moe_hquant.slang` / `moe_xquant.slang` | 支持 | `idx / blocks` → m |
| `mega_mhc` / `wq_a` / `wq_b` / `wkv` / `sparse_attn` / `compressor` / `indexer` / `wo_a` / `wo_b` / `gate` | **不支持** | `AttnRunner::make` 无条件 `ps.m = 1`，`AttnSpec` **根本没有 `m` 字段**；`attn_common.slang` 的 `[vk::constant_id(0)] const uint M = 1; // batch; 1 until DSpark (§7.12)` |
| `engram.slang` / `head.slang` | **不支持** | `DecodeRunner::make` 同样 `ps.m = 1`；`head.slang` 的 `groupshared float gXf[5120]` 是单 token 的 |

也就是说 **verify batch 的 M = k+1 今天只有 MoE 那一段能跑**，
非 MoE 的十二个 dispatch 全是 M = 1 编译进去的。三条路：

1. 把 `AttnSpec` 加上 `m`，让 `ps.m = spec.m`，并给 `fp8_gemv.slang` 的
   `stage_x_fp8` / `gemv8_rows` 加 m 下标（`gXQ`/`gXS` 现在只有一列）。
   这是 **Track A（attention）的工作**，不是 Track K 的；
   `p2_attention.md` §11 的 next-steps 里没有它，要加进去。
2. 对 M = k+1 循环调用 M = 1 的 kernel。正确，但把 §10.1 的"常驻 8.5 GB 摊薄"
   直接作废 —— 那正是投机解码的全部收益。**不可接受。**
3. `head.slang` 是单点最大的一块（1.32 GB，§2.3 预算的 10%），
   而且 `head.slang` 自己的注释就说了 *"at M = 5 the same 1.32 GB yields logits for five positions.
   This is the M = 1 form."* —— 从它开始。

`gpu/shaders/dspark_gemv.slang` 里的 M 循环（每个 K 块解码一次权重、复用给 M 列，
就是 `moe_gateup.slang` 的做法）**是给 Track A 的模板**：它和 `fp8_gemv.slang` 的
`gemv8_rows` 是同一套算术，只多了最内层的 m 维。


### 6.1 K2：树采样方案要的 kernel 清单（相对单链方案的增量）

| 位置 | 要什么 | 今天 | 备注 |
|---|---|---|---|
| 草稿 head（M = 5） | 与单链相同 | `head.slang` 是 M = 1（循环 5 次 ≈ 28 ms，§8） | 不是 K2 引入的；树方案不改变它 |
| 草稿 Markov 5 位 | GEMV + 加偏置，**argmax 换成 top-K**，并 gather 候选上的 base logit + 该行 lse | `dspark_head` 有 GEMV + argmax | 新增一个 top-K/gather/lse 的小 kernel（V 次标量扫描），读回 5 × K × 8 B |
| verify 的主模型（M = k + 1） | attention 族（`mega_mhc` / `wq_a` / `wq_b` / `wkv` / `sparse_attn` / compressor / indexer / `wo_a` / `wo_b` / `gate`）、`head`、`engram` 的 M > 1 | **全部不存在**（§6 表） | 与单链方案完全相同的需求；indexer 若要逐位对齐逐 token 参考还要 §4.8 的逐 query key 源（§14） |
| verify 的 MoE | M 个 token 的 expert **并集**（> 7 槽或多组 dispatch） | 7 槽 runner | 同上 |
| verify 读回 | 贪心：M 个 argmax；温度 1 精确方案：每行 K 个候选 logit + lse + 1 个屏蔽样本（+ 末行 1 个整行样本） | 只有 argmax | 一个 V 次标量扫描的小 kernel，读回 < 1 KB |
| CPU | `cpu/dspark_tree`（格、路径、confidence、接受） | **已实现、逐位对齐 Python** | §13.1：K = 16 全程 ≈ 0.1 ms |

---

## 7. Track K 的实测（`tools/oracle_dspark.py`，L3 prompt，64 token prefill）

**运行时间（供其他 track 核对计时标签）**：本机 CPU torch 12 线程，
run 1 **2026-09-14 12:54:35 – 13:28**（含 306 s crosscheck，第 13 周期因 assert 退出），
run 2 **13:29:59 – 14:05**（第 14 周期因共享机器 commit 耗尽退出，见 §7.6）。
两次都是 ~3.5 GB 常驻、12 个 CPU 线程满载；GPU 在 13:30–13:56 之间另跑了 `dspark_bench` 与 `gpu_dspark`。

### 7.1 锚点

| 检查 | 结果 |
|---|---|
| prefill 64 token 后的 argmax | **3006**，与 `tests/data/l3` 的 `greedy_tokens[0]` 相同 |
| position 64 decode 的 argmax | **223**，与 L3 `greedy_tokens[1]` 相同（两条独立导出路径） |
| position 64 的草稿 `[18, 16, 18, 201, 671]` | L3 的真实续写是 `18, 16, 18, 201, 1` → **前 4 个全对**，cycle 0 实测接受 4 |
| `--crosscheck`：cycle 0 的 M=6 verify batch vs 整条 71 token 的参考 prefill | argmax **6/6 一致**；logits 余弦 0.9968 / 0.9964 / 0.9932 / **0.9453** / 0.9940 / 0.9981 |
| 每周期"恢复快照 + 只重跑已接受前缀"与 verify batch 同位置 logits 的余弦 | a=4（M=5 重跑）**1.000001**；a≤2（M≤3 重跑）0.973–0.99999，中位 0.998 |
| 同上的 argmax | 13/14 周期一致；**cycle 13（位置 91）不一致**：verify 选 63954（margin 0.026），重跑选 1300（margin 0.174） |

**这不能证明 `install_batched_decode` 有 bug，也没能证明没有**，所以如实写：
余弦 0.945–0.998 只出现在 M 不同的两次 forward 之间，说明参考的 decode 路径**本身依赖批边界**。
最可能的机制（**假设，未单独隔离验证**）：ratio 2 的 KV 源层只在"本次 forward 内有压缩组完成"时
才发布 `shared_attn.index_k`，否则层 2–19 的 indexer 读的是上一次 forward 留下的 key
（`oracle.py` L2 导出说明里记录的那个"故意不重置的单例"）。M = 6 的 verify 必然完成至少一组，
M = 1 的重跑在偶数位置不完成，两者给 2–19 层的 top-512 就不同。prefill vs decode 的 0.945 同理。

**后果写进规格**：design §10.2 的不变量"`temperature=0` 时开关投机逐 token 相同"
**对参考实现本身不成立**，在 margin 小的位置会翻转（本次 14 个周期 1 次，margin 0.026）。
Track I 的内建检查要改成"margin > ε 时逐 token 相同，否则记录"，并决定 runtime 对齐谁（§10 问题 1）。
oracle 在分歧时沿用重跑（M = a+1）的 token，因为携带下去的状态是重跑的。

**K2 补记**：上面那个 key 源机制在短上下文被排除（index top-512 退化成全选），剩下的差是 CPU torch 的批形状数值差，见 §14。

### 7.2 接受长度 —— **作废**（K2）

Track K 在 L3 prompt 的一条退化轨迹（两个 EOS 之后进入无关中文）上用单链方案量到平均接受 0.929、1.929 tokens / verify。
方案错（§3 开头）、轨迹退化、14 个周期，三条都使它不能代表模型。新数据见 §11–§12：正常文本上 runtime 回放 2.6–3.3 tokens / verify。

### 7.3 expert 并集（design §9.5 / §10.3 的 `union_frac`）

每个 verify batch 按前缀 M = 1..6 统计每层不同 routed expert 数，除以 `6M`，40 层 × 14 周期平均：

| M | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 并集 expert 数 / 层 | 6.00 | 9.96 | 13.15 | 15.85 | 18.87 | 21.39 |
| `union_frac[M]` | 1.000 | **0.830** | **0.730** | **0.660** | **0.629** | **0.594** |
| Q3（route_trace，相邻真实 token） | | 0.824 | 0.734 | 0.676 | 0.634 | — |

**和 Q3 几乎重合**（差 ≤ 0.016）。design §9.1.1 担心的"Q3 假设草稿全被接受"没有表现出来：
被拒草稿 token 路由到的 expert 与真实 token 的重叠程度差不多。
逐层 M = 6：最小 **层 18 0.482**、层 16 0.496、层 6 0.500；最大 **层 0 0.802**、层 19 0.692、层 1 0.671
（全表在 `tests/data/dspark/stats.json` 的 `union_frac_by_layer`），与 design §10.3 的"层 0 0.818 / 层 25 0.518"同形。

mtp 自己的 routed expert（position 64 的草稿周期）：stage 0/1/2 分别碰 **10 / 9 / 7** 个（上限 15），
即每周期 **26 × 18.81 MB = 489 MB**。

### 7.4 草稿周期的字节

§1.5 的固定 2.24 GB + §7.3 的 0.49 GB = **每周期 2.73 GB**（design §7.12 估 2.2 GB）。

### 7.5 confidence head —— **作废**（K2）

对齐方式（一个周期记录的 `confidence` 属于它的 `next_drafts`）仍然正确；统计结论由 §12.5 取代
（246 个位置：σ(conf) 大致单调、低端偏保守，σ ≈ 0.3 时实际接受 0.60）。

### 7.6 run 2 为什么停在第 14 周期

`dsref.make_block` 构造层 1/14 的 `Block` 时，参考的 `Engram.__init__` 先
`torch.empty(384M, 256, dtype=fp8)` —— **98 GB 的未触碰占位**（随后被 `EngramRowEmbedding` 换掉）。
Windows 上它要 commit，共享机器上其他进程占着 commit 时就失败。这是 dsref 的既有行为；
每周期写的检查点让已测的 14 个周期没有丢。**建议**（本轮没改 dsref）：`make_block` 里临时把
`ref.ParallelEngramEmbedding` 换成 meta-device 占位。
（K2 已按这个建议做了：`oracle_longctx._NoEngramTable` 在构造期替换 `ParallelEngramEmbedding`，并在开跑前等 commit 余量，§11。）

### 7.7 kernel 正确性（`tests/test_gpu_dspark.cpp`，`ctest -R suite.gpu_dspark` 通过；开 validation layers 跑过一次，0 条消息）

每个阶段喂参考自己的输入、比参考自己的输出；stage 1/2 的 attention 没有 golden，改为链式（RoPE → attention → 逆 RoPE → wo_a）在 wo_a 输出处比较：

| 阶段 | cos（三个 stage 最差） | rel L2 | 说明 |
|---|---|---|---|
| `main_proj`（K=15360, M=1） | 0.9999986 | 1.68e-3 | fp8 GEMV + act_quant；rel L2 是 bf16 地板 |
| `main_norm` | 0.99999998 | 1.9e-4 | |
| `wq_a` / `wq_b` / `wkv`(M=1) / `wkv`(M=5) / `wo_b` | 0.9999985–0.9999992 | 1.60–1.75e-3 | bf16 地板（p2_attention.md 同一数） |
| `q_norm` / `kv_norm` | 1.000000000 | 0（stage 1 的 q_norm 7e-6） | 逐位 |
| RoPE(q) | 1.000000000 | 1.3e-6 | |
| RoPE + act_quant（main_kv 与 draft kv，环里存的值） | 1.000000000 | ≤ 3.4e-7，多数逐位 | |
| attention（score + combine）+ 逆 RoPE | 0.999999998 | 5.7e-5 | |
| `wo_a`（stage 0 golden 输入） / 链式（stage 1、2） | 0.9999986 / 0.9999986 | 1.65e-3 | 链式不比单级差 |
| Markov bias + 加偏置 + argmax，5 位 | top-64 logits cos 1.000000000 | ≤ 1.2e-7 | **5/5 草稿 token 一致**（位 4 margin 0.062 也对上） |
| Markov embed（5 行） | 1.000000000 | 0 | 逐位 |
| confidence（5 位） | 1.000000000 | 4.8e-7 | 7.459416 vs 7.459421 |

全部过 cos ≥ 0.9999；最差的 0.9999985 就是参考 bf16 输出的舍入地板。head GEMV 不是 DSpark kernel（§6），测试在 CPU 上算 logits 喂给 Markov 阶段。

---

## 8. `T_draft`

`bench/dspark_bench.cpp` → `bench/results/dspark_p3.csv`，`--iters 16`，best of 3，L32 sg32，
**机器不空闲**（CPU oracle 12 线程在跑，另两个 agent 共用）：

| kernel | ms | 字节 | GB/s | 同 kernel M=1（p2_attention §5） |
|---|---:|---:|---:|---:|
| `main_proj`（K=15360, M=1） | 0.664 | 78.7 MB | 118.5 | — |
| `wq_a`（M=5） | 0.131 | 6.56 MB | 50 | 0.044（×3.0） |
| `wq_b`（M=5） | **1.282** | 42.0 MB | 33 | 0.268（**×4.8**） |
| `wkv`（M=1 / M=5） | 0.046 / 0.080 | 2.62 MB | | 0.021 |
| attention score + combine（M=5，70 KV） | 0.338 | | | 0.142（×2.4） |
| `wo_a`（M=5） | 0.520 | 33.6 MB | 65 | 0.201（×2.6） |
| `wo_b`（M=5） | 0.753 | 42.0 MB | 56 | 0.347（×2.2） |
| norm / RoPE / quant（合计） | 0.21 | | | |
| Markov bias（1 位） | 0.448 | 66.2 MB | 148 | |
| 加偏置 + argmax | 0.292 | | | |
| confidence | 0.072 | | | |
| **整条 DSpark kernel 链，一个命令缓冲（77 个 dispatch）** | **11.26** | 798 MB | 71 | |
| 同上，开 validation layers | 16.4 | | | |

`T_draft` 还要加上复用已有 kernel 的部分（**借自已有测量，不是本次测的**）：

| 项 | ms | 来源 |
|---|---:|---|
| DSpark kernel 链 | 11.3 | 本表 |
| 3 × (mega_mhc × 2 + gate) | ~0.3 | p2_attention §5 M=1 × 5 的上界 |
| 3 × MoE（3 routed + 1 shared，M=5） | ~2.0 | kernel_p2_moe §3.5 M=5 的 0.879 ms（7 槽）按 4 槽缩 + 真 shared expert |
| `head`（1.32 GB，M=5） | **5.6 – 28.2** | 摊满权重读是 5.6；今天 head.slang 是 M=1，循环 5 次 28.2 |
| **`T_draft`** | **≈ 19（M=5 head）/ ≈ 42（循环 M=1 head）** | design §10.3 假设 11 |

**M 维几乎没有摊薄 GEMV。** `wq_b` 在 M=5 花 M=1 的 4.8 倍 —— 与 kernel_p2_moe §3.6 对 MoE M=6 的诊断相同：
VALU 受限（每个权重块和 M 列激活各做一次点积），不是带宽受限。`wo_a`/`wo_b`/attention 好一些（×2.2–2.6）。
这否定了 design §10.3 的"`T_hot(M)` 非 MoE 部分预期接近平坦"，§9 的场景 B 用它。

---

## 9. 调度与预计加速 —— **作废**（K2）

旧 §9 用 §7.2 的退化轨迹接受率与 Track K 的 `union_frac` 算出"最多 +5%、按实测 M 缩放每个 k 都亏"。
接受率与方案都已更正，调度（confidence 选 k 的规则）也换了。**新的 TPS 模型、输入与 go / no-go 在 §15。**
`oracle_dspark.py::schedule_k` 保留作 Track K 数据的复现用，不再是推荐调度。

---

## 10. 未解决的问题（K2 更新）

1. **长上下文的批边界**（§14）。短上下文 key 源机制已排除（top-512 退化），4K / 17K 时它一定起作用。
   runtime 的 M > 1 indexer 要不要实现 §4.8 的逐 query key 源？需要在 Track M 的 4K / 17K 状态上跑一次"逐 token 语义 vs 普通 verify"的 top-k 与接受比较。
2. **接受率样本**（§15.4 条件 3）：30 个周期 / 模式、56 token prompt。×1.18 离 1.15 门槛只有 3%；要 ≥ 64 decode token × 5 prompt × 两模式、空闲机、最好直接用 GPU runtime。
3. **M > 1 kernel**（§6、§15.3）不是 K / K2 的文件；它们的实测曲线替换 §15 的场景 A / B 才是真正的决定。
4. **窗口环回绕后的批写入**（§4.9）：oracle 的推广在位置 ≥ 128 时会让批里较早的 query 丢掉最多 M − 1 个最老窗口位置；runtime 与 oracle 都要改成"先 attend 后写"或旁路缓冲。
5. **温度 1 的树截断**（§12.4）：anchor 候选在采样出非链 prev 时覆盖不足（位 4 的 r 0.30 vs 单链 0.51）。
   `union2` 一次 2 列 GEMM 可能追回，未在采样模式测。只有在"比较必须在 CPU 上"这一约束成立时才值得做。
6. **confidence θ**：θ = 0.5–0.7 在本轮最好，是在同一批数据上选的（有过拟合）；换数据要重选。σ 映射低端偏保守（§12.5）。
7. **stall 的 miss 口径**（design §10.1.3 缺口 2）未变：§15 用 ∝ `M · union_frac[M]`；Planner 在 verify 前一次下单并集 miss 的收益未建模。
8. **DsparkRunner 地址表**、**mtp KV 回滚**、**多位置 `main_x` 写环**（Track K 的问题 6–7）未变。

---

## 11. Track K2 的运行：五个正常 prompt × {贪心, 采样}

**运行时间（供其他 track 核对计时标签）**：本机 CPU torch 16 线程，~3.5–4 GB 常驻，Engram 的 98 GB 占位已用 `oracle_longctx._NoEngramTable` 桩掉，
开跑前等 commit 余量 ≥ 28 GiB（实测 97–112 GiB 可用），每周期写检查点。

| 段 | 起止 | 说明 |
|---|---|---|
| 试跑 1 | **2026-09-14 23:19:18 – 23:25** | 候选取 B 的 top-K（`base`），接受率 ≈ 0，停掉（§3.1）；en_prose 的 prefill 保留复用 |
| 试跑 2 | **23:27:52 – 23:43** | `anchor` 候选；en_prose 6 周期 + zh_prose 5 周期；为换温度 1 的精确接受而停掉，从检查点续跑 |
| 正式 | **23:43:33 – 2026-09-15 01:00:26** | 贪心 23:43:40–00:16:21，采样 00:16:21–00:48:01，批边界 00:48:01–01:00:19 |

**CPU 总时长 ≈ 1 h 40 min**（23:19–01:00，含两次停机）。GPU 在这段时间里有其他 track 的 `deepmoe_tests.exe`（23:19、23:27 两次开跑时可见）。

**prompt**（`tools/oracle_dspark.py::TREE_PROMPTS`，BOS + 56 token，都在句中/行中截断；轨迹原文在 `traces/dspark_tree/<prompt>/<mode>/log.json`）：

| 名字 | 来源 | 贪心续写开头 |
|---|---|---|
| `en_prose` | 模型 README 的 "**Architecture.**" 段 | " attention is replaced by cross-attention over the encoder's hidden states…" |
| `zh_prose` | `docs/design.md`（`0170339`）开头段 | " + page cache…SSD…" |
| `python` | `tools/dsref.py::expert_stream`（`0170339`） | "q.put(e) / except Exception: pass" |
| `cpp` | `cpu/gate.cpp::gate_topk`（`0170339`） | "if (n == 0 \|\| k == 0) { return GateResult{}; }" |
| `markdown` | `docs/p2_decode.md` §10.3 的编号列表（`0170339`） | ". The 30 ms is the host-side cost of building the dispatch…" |

没有一条轨迹出 EOS 或复读退化（与 Track K 的 L3 轨迹不同）。

**规模**：**60 个真实 verify 周期**（5 × 2 × 6，恒 k = 5、M = 6），产出 181 个 token；
每个被接受位置各跑一次草稿，共 **141 个有完整 5 token 前瞻的草稿事件**（贪心 71、采样 70）。

**驱动方案自己的实测**（真实 M = 6 verify，不是离线评估）：

| prompt | 贪心每周期接受 a | 采样每周期接受 a |
|---|---|---|
| cpp | 5 2 2 5 2 5 | 5 5 1 0 5 5 |
| python | 5 1 5 5 1 0 | 5 1 1 5 5 0 |
| en_prose | 1 1 2 3 2 1 | 2 1 4 0 2 3 |
| markdown | 1 0 4 0 3 1 | 0 0 0 3 0 0 |
| zh_prose | 0 0 2 0 0 2 | 0 2 0 2 2 1 |
| **平均** | **2.03 → 3.03 tokens / verify** | **2.00 → 3.00 tokens / verify** |

Track K 那条退化轨迹是 0.93 → 1.93。代码 prompt 高、中文低，差一个数量级的方差——**30 个周期一个模式，只够定量级**。

---

## 12. 接受率

所有数都来自 `tests/data/dspark/tree_stats.json`（`dspark_tree.py analyse`）。两种口径：
**每事件**（每个有前瞻的草稿位置一个样本，n = 71 / 70）与 **runtime 回放**（只在每周期最后一个接受位置起草，§3.4 的轨迹上模拟真实周期序列；
采样模式按 r 掷骰 400 次取平均）。前者样本多、偏向容易的段落（全接受的段落贡献更多事件），后者就是 runtime 的 tokens / verify，**TPS 用后者**。

### 12.1 草稿矩阵的两条事实

* `B + H·E[prev]` 逐位 argmax 重放参考链：**705 / 705 个草稿 token 与 `forward_head` 完全一致**。
* 参考链 token 在**不加偏置**的 B 行里的排名：位 0–4 中位数 1 / 3 / 8 / 13 / 11，p90 176 / 268 / 138 / 164 / 236，
  落在 B 的 top-16 里的只有 74% / 67% / 62% / 55% / 54%。

### 12.2 贪心（温度 0）：`E[tokens](k) = 1 + E[min(a, k)]`，每事件，n = 71

| 方案 | k=1 | k=2 | k=3 | k=4 | k=5 | 平均 a（k=5） | a = 0..5 分布 |
|---|---:|---:|---:|---:|---:|---:|---|
| **旧单链（参考 `forward_head`）** | **1.83** | **2.52** | **3.08** | **3.55** | **3.93** | **2.93** | 12 10 9 7 6 27 |
| 树 K=4 `eal`（tail） | 1.83 | 2.52 | 3.08 | 3.55 | 3.93 | 2.93 | 与单链相同 |
| 树 K=8 `eal` | 1.83 | 2.52 | 3.06 | 3.51 | 3.87 | 2.87 | |
| 树 K=16 `eal` | 1.83 | 2.51 | 3.03 | 3.46 | 3.82 | 2.82 | |
| 树 K=32 `eal` | 1.83 | 2.52 | 3.06 | 3.51 | 3.87 | 2.87 | |
| 树 K=4 / 8 / 16 / 32 `viterbi` | 1.83 / 1.80 / 1.82 / 1.82 | 2.51 / 2.46 / 2.49 / 2.51 | 3.06 / 2.99 / 3.03 / 3.04 | 3.52 / 3.44 / 3.46 / 3.51 | 3.90 / 3.80 / 3.82 / 3.87 | 2.90 / 2.80 / 2.82 / 2.87 | |
| 树 K=16 `eal`，全词表精确归一化 | 1.83 | 2.52 | 3.07 | 3.54 | 3.92 | 2.92 | |
| 候选 `base` K=16（`eal` / `chain`） | 1.69 / 1.68 | 2.18 / 2.15 | 2.45 / 2.42 | 2.58 / 2.55 | 2.65 / 2.62 | 1.65 / 1.62 | 22 14 16 10 4 5 |
| 候选 `union2` / `union4` K=16 `eal` | 1.83 / 1.83 | 2.52 / 2.51 | 3.06 / 3.03 | 3.51 / 3.48 | 3.87 / 3.85 | 2.87 / 2.85 | |

runtime 回放（tokens / verify）：旧单链 k = 1..5 **1.83 / 2.59 / 2.89 / 3.33 / 3.33**（24–40 个周期）；树 K=16 `eal` 1.83 / 2.50 / 2.85 / 3.33 / 3.33。

**结论：温度 0 下树搜索不比单链好，只会更差或持平。** 这是必然的：贪心接受要求每一位都等于主模型 argmax，
草稿在"前缀全对"条件下的逐位 argmax 就是对这件事的最好点估计；`viterbi` / `eal` 为了联合概率会在早位置让步，而早位置错一个就全丢。
候选规则里 `union` 让格的覆盖从 anchor K=16 的"前缀全在候选里"平均 3.44 位提高到 3.99 / 4.13，但被选中的路径并没有因此更好。

### 12.3 归一化：锚行 tail 近似 vs 每个 (位置, prev) 的全词表 logsumexp

71 个事件里所选路径与"精确归一化"不同的次数：

| K | `viterbi` tail | `viterbi` 只在 K 内归一化 | `eal` tail | `eal` 只在 K 内 |
|---:|---:|---:|---:|---:|
| 4 | 6 | 17 | 1 | 10 |
| 8 | 7 | 20 | 2 | 8 |
| 16 | 7 | 18 | 3 | 8 |
| 32 | 9 | 21 | 1 | 7 |

**会改变路径**：只在 K 内归一化时 10–30% 的事件换了路径；锚行 tail 近似把它压到 1–13%，`eal` 基本不受影响（≤ 3/71）。
对接受率的影响 ≤ 0.1 token（§12.2 表中 tail vs 精确）。精确版要 1 + 4K 次 66 MB 的 GEMV，不值；tail 近似零成本。

### 12.4 采样（温度 1，`generate.py` 默认；精确无损接受）：`E[tokens](k)`，每事件，n = 70

| 方案 | k=1 | k=2 | k=3 | k=4 | k=5 | 各位平均 r（位 0..4） |
|---|---:|---:|---:|---:|---:|---|
| **旧单链（整词表采样，`forward_head` 温度 1）** | 1.78 | 2.39 | 2.87 | 3.24 | **3.52** | 0.78 0.75 0.66 0.56 0.51 |
| 树 K=4 祖先采样 | 1.79 | 2.35 | 2.77 | 3.12 | 3.38 | 0.79 0.60 0.45 0.38 0.29 |
| 树 K=8 | 1.78 | 2.37 | 2.81 | 3.16 | 3.42 | |
| 树 K=16 | 1.78 | 2.38 | 2.84 | 3.21 | 3.48 | 0.78 0.65 0.52 0.41 0.30 |
| 树 K=32 | 1.78 | 2.39 | 2.85 | 3.22 | 3.49 | |
| 树 K=16 `eal` 确定路径（用采样接受） | 1.74 | 2.30 | 2.70 | 3.06 | 3.31 | |

runtime 回放：旧单链 k = 1..5 **1.78 / 2.38 / 2.74 / 3.05 / 3.22**；树 K=16 1.77 / 2.39 / 2.70 / 3.02 / 3.22。

**结论：温度 1 下树采样与单链持平（−1%），K 越大越接近。** 截断到 anchor 候选的代价在后几位：
采样出来的 prev 不是链 token 时，下一位真正该出的 token 常不在"按链 token 加偏置"选出的候选里（位 4 的 r 0.30 vs 单链 0.51）。
K 从 4 到 32 只追回一半。**树方案在温度 1 的真正价值不是接受率，而是比较步骤能在 CPU 上做**：
单链的精确无损接受需要整词表的 q 行（2.6 MB，只能留在 GPU 上做残差），格的 q 只支撑在 K 个候选上，残差所需的全部东西 < 1 KB 就能读回（§3.3）。

### 12.5 confidence 选 k

规则：累计 σ(conf) ≥ θ 的最长前缀（§3.2）。runtime 回放的 tokens / verify 与 k 分布（k = 0..5 的周期数）：

| 方案 | θ = 0.3 | θ = 0.5 | θ = 0.7 |
|---|---|---|---|
| 贪心 旧单链 | 3.04（平均 k 2.88） | 2.74（k 2.15；k 分布 1 10 9 1 3 3） | 2.17（k 1.23） |
| 贪心 树 K=16 `eal` | 3.04 | 2.74 | 2.11 |
| 采样 旧单链 | 2.88（k 2.71） | 2.58（k 1.91） | 2.09（k 1.24） |
| 采样 树 K=16 | 2.92 | 2.57 | 2.09 |

confidence 的 tokens / verify 比固定 k = 5 低，但 verify 的 M 小得多——§15 里它是**最好的调度**。

**校准**（贪心，树 K=16 `eal` 路径上前缀全对时每一位，n = 246）：

| σ(conf) 区间 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---:|---:|---:|---:|---:|
| 样本 | 20 | 47 | 67 | 47 | 65 |
| 平均 σ | 0.32 | 0.51 | 0.70 | 0.85 | 0.97 |
| 实际接受率 | **0.60** | **0.57** | **0.81** | **0.89** | **1.00** |

logistic 映射**大致单调（前两档持平）、低端偏保守**（σ = 0.3 时实际一半以上被接受）。这比 Track K 的"没有被否定"强一档；θ 应按 TPS 选（§15），不必按概率意义选。

### 12.6 verify batch 的 expert 并集

60 个 M = 6 verify batch，40 层平均（`union_frac[M]` = 并集 / 6M）：

| M | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 并集 expert 数 / 层 | 6.00 | 9.96 | 13.46 | 16.67 | 19.77 | 22.50 |
| `union_frac[M]` | 1.000 | **0.830** | **0.748** | **0.695** | **0.659** | **0.625** |

比 Track K（0.830 / 0.730 / 0.660 / 0.629 / 0.594）略高：正常文本的相邻 token 路由更分散。§15 用新值。

---

## 13. CPU 代价、GPU 字节、无损性

### 13.1 CPU 时间（步骤 2 与 4）

C++：`cpu/dspark_tree`，`tests/test_dspark_tree.cpp::cpu_cost_microseconds`（zig clang `-march=znver5`，AVX-512，单线程，机器上有别的 track；K = 4/8/16 取 2000 次、K = 32 取 400 次平均）：

| K | 格（含 tail） | `eal` 路径 | confidence ×5 | 贪心比较 | 格（无 tail） | 祖先采样 | top-32 截断接受 | **精确接受** | **贪心一周期合计** | **采样一周期合计** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5.5 µs | 2.3 µs | 1.3 µs | 0.06 µs | 4.5 µs | 0.7 µs | 3.1 µs | 0.35 µs | **≈ 9 µs** | **≈ 7 µs** |
| 8 | 21.8 | 8.3 | 1.6 | 0.07 | 20.5 | 1.2 | 3.6 | 0.68 | **≈ 32 µs** | **≈ 24 µs** |
| 16 | 53.0 | 30.1 | 2.3 | 0.08 | 48.0 | 2.4 | 4.1 | 1.30 | **≈ 85 µs** | **≈ 54 µs** |
| 32 | 197.5 | 123.1 | 5.2 | 0.19 | 192.3 | 5.6 | 5.5 | 2.54 | **≈ 0.33 ms** | **≈ 0.20 ms** |

（K = 4–32 的"格/路径/精确接受"列来自最后一次构建；前一次构建在把 Horner 的除法换成倒数常数之前，K = 16 的格是 93 µs、`eal` 61 µs。）
格的主要代价是 5K² 个 256 维 lane 点积与 5K(K+1) 次 `dm_exp`；K = 16 一个周期 < 0.1 ms，对 150 ms 量级的周期可以忽略。

Python（`dspark_tree.py bench`，同一批草稿事件，numpy）：K = 16 格 6.5–8.4 ms、`eal` 0.6 ms、confidence 1.3 ms、截断接受 2.4 ms——**比 C++ 慢约 100 倍**，只作参考实现。
另外 5 次 Markov 行 + top-K 在 numpy 里是 ≈ 22 ms（整词表 GEMV），runtime 里它在 GPU 上，属于草稿本来就有的那部分。

**逐位一致**：`suite.dspark_tree`（`ctest -R dspark_tree`）用 `tests/data/dspark/tree_golden.bin`（2 个真实周期：en_prose 采样 cycle 0、cpp 贪心 cycle 0；431 KiB）
对 K ∈ {4, 8, 16, 32} × {tail, 无 tail} 比较：全部 logq / bias 数组的 FNV-1a 哈希、三种目标的路径、`eal` 值、每条路径的 5 个 confidence 与 k、
祖先采样路径、两种采样接受与贪心接受的结果——**全部 `==` 通过**。做法：float32 点积按 16 lane 累加再按 lane 顺序求和（AVX-512 一个寄存器一行），
exp / log 用只含 `+ − × ÷ floor frexp ldexp` 的双精度级数（不碰 libm），所有求和按下标顺序，C++ 文件关闭浮点收缩（`-march=znver5` 有 FMA）。

### 13.2 GPU 字节与单链方案相同

| | 单链方案 | 树方案（`anchor`） |
|---|---|---|
| 草稿 forward | DSpark kernel 链 + head（M = 5） | **相同** |
| Markov | 5 次 66.2 MB GEMV + 加偏置 + argmax | **相同**的 5 次 GEMV + 加偏置；argmax 换成 top-K（同一行一次扫描），另读回候选上的 base logit 与 lse |
| 草稿读回 | 6 个 token id | 5 × K 个（id + logit）+ 5 个 lse：K = 16 时 660 B |
| CPU 查表 | — | 主机上的 `markov_head.embed` / `head` 两张 bf16 表各 66 MB，每周期 gather (4K + 1) + 5K 行 × 256 维：K = 16 时 ≈ 83 KB 的缓存读 |
| verify forward | `[last, drafts[:k]]`，M = k + 1 | **相同**（同样 M 个 token） |
| verify 读回（贪心） | M 个 argmax | **相同** |
| verify 读回（温度 1，精确） | 单链要整词表 q 与 p 做残差（留在 GPU 上） | 每行 K 个 logit + lse + 1 个屏蔽样本（+ 末行 1 个整行样本）：K = 16、M = 6 时 < 0.5 KB |

`base` 规则会把草稿的 5 次 Markov GEMV（331 MB，§1.5）省掉，但接受率掉到 1.65（§12.2）。`unionM` 读同样 66 MB/位、算力 ×M，覆盖更好但路径不更好。

### 13.3 无损性（温度 1）

`dspark_tree.py lossless`：固定一个真实周期（en_prose 采样 cycle 0），抽 60,000 次"祖先采样一条路径 + 接受"，统计第一个输出 token 的频率，对比目标分布：

| 接受规则 | K | 位 0 接受率（实测 / 理论 Σ min(p, q)） | TV（频率 vs 目标） | 同样本量的 TV 噪声底 | χ² / 自由度 |
|---|---:|---|---:|---:|---:|
| `accept_sampling`（目标 = top-32 截断的 p~） | 4 | 0.8421 / 0.8443 | 0.0018 | 0.0022 | 22.7 / 31 |
| 同上 | 16 | 0.8412 / 0.8403 | 0.0017 | 0.0022 | 19.6 / 31 |
| `accept_sampling_exact`（目标 = 该行 top-256 当作整个词表的 softmax，屏蔽样本用 Gumbel-max） | 4 | 0.8460 / 0.8440 | 0.0035 | 0.0039 | 32.8 / 45 |
| 同上 | 16 | 0.8386 / 0.8402 | 0.0033 | 0.0039 | 53.7 / 45 |

四组都在噪声底以内、χ² 与自由度同量级：**两条规则都对各自的目标无损**，接受率与理论值 `Σ min(p, q)` 吻合到 0.2%。
（精确规则的检验用 top-256 当词表，只是为了在离线数据上能抽屏蔽样本；规则本身对任何分布都成立。
一个更尖的周期——cpp 采样 cycle 0，位 0 接受率 0.987——同样通过：χ² 3.4 / 4 与 10.0 / 4。）

verify 行 top-Kv 之外的尾部质量（60 个周期里所有被接受相关的行，每行 1 − Σ top-Kv p）：

| Kv | 8 | 16 | 32 | 64 | 256 |
|---:|---:|---:|---:|---:|---:|
| 均值 | 0.077 | 0.050 | **0.033** | 0.022 | 0.009 |
| p90 | 0.310 | 0.205 | **0.126** | 0.073 | 0.027 |
| 最大 | 0.631 | 0.524 | **0.418** | 0.343 | 0.184 |

**top-32 截断的目标与 `generate.py` 的温度 1 分布之间的 TV 距离就是这个尾部质量**：均值 3.3%、十分之一的行超过 12.6%、最坏 42%。
所以默认用 `accept_sampling_exact`；截断版只在"本来就想做 top-k 采样"时才是对的。

---

## 14. 批边界效应

Track K 在 M = 6 verify 与 M ≤ 3 重跑之间看到 logit 余弦低到 0.945、14 个周期里一次 argmax 翻转，猜测机制是
"ratio-2 的 key 源只在本次 forward 内有组完成时才发布 `index_k`，否则层 2–19 读上一次 forward 留下的层 20 的 key"（旧 §7.1）。K2 把它隔离了：

**实验**（`oracle_dspark.py --tree` 第三阶段，每个 prompt 贪心的前 2 个周期，共 10 个 verify batch，从 verify 之前的完整状态快照重放）：

1. 同一 batch 用 §4.8 的**逐 token 语义 indexer**（每个 query 用它逐 token decode 时会看到的 key 源）再跑一遍；
2. 其中 en_prose cycle 0 再**真的逐 token decode** 6 次（M = 1）；
3. 同一 batch 的普通 verify 再跑一遍（确定性）。

| 比较 | 结果 |
|---|---|
| 普通 verify 重跑 vs 轨迹里的那次 | top-256 logits **逐位相同** |
| 逐 token 语义 indexer vs 普通 verify（10 个 batch × 6 行） | top-32 logits **逐位相同**（最大差 0.0），argmax 60/60 相同，接受长度 10/10 相同 |
| 逐 token decode（M = 1 × 6）vs 普通 verify | logits 余弦 **0.9992 / 0.9994 / 0.9995 / 0.9992 / 0.9992 / 0.9982**；共有 top-32 id 上的 logit 绝对差均值 **0.18**、最大 **0.58**；argmax **6/6 相同**，接受长度 1 = 1 |

**机制**：

* **Track K 猜的 key 源机制在这个上下文长度下不起作用。** 位置 57–90 时每个 ratio-2 indexer 能看到的压缩行只有 `(P+1)/2 ≤ 45` 行，
  而 `index_topk` = 512：top-k 退化成"全选"，key 用谁的都选出同一个集合（design §11.5 的"top-k 退化"）。所以 1 的结果逐位相同。
  **长上下文（≥ 1024 token）时它一定起作用**（p3_longctx.md §5.3：17K 时两种 key 的 top-k 只重叠 8%），那时 runtime 要么实现 §4.8 的逐 query 源、要么接受与逐 token 参考不同。
* **剩下的差（余弦 0.998–0.9995）是 CPU torch 的批形状数值差**：`F.linear` 对同一行在 batch 1 与 batch 6 里给出不同的值——
  float32 权重时单次投影最大差 7.4e-5，bf16 权重时 0.25（本机实测，`torch.equal` 为 False），40 层 fp8/bf16 量化边界把它放大。
  这不是模型语义，是参考实现在 CPU 上的非批不变性；**GPU kernel 若按行独立累加（lane 顺序固定）就没有它**。
  Track K 的 0.945 出现在 prefill（M = 71）对 decode 的比较里，形状差更大，同一机制。

**对接受决定的影响**：本轮 10 个 batch 零翻转。暴露面按 margin 估：60 个贪心周期里决定接受的行（每周期第 0..a 行，n = 91），
top-1 − top-2 margin 中位数 2.2，**< 0.25 的占 8.8%、< 0.5 的占 16.5%、< 1.0 的占 34%**。批形状扰动最大 0.58，
所以大约每 10 行有 1 行**可能**被翻转（翻转只改变周期边界处的哪个 token 被选，贪心轨迹本身在参考的近似平局处本来就不稳定，design §1.3 (a) 已按 margin < 1.0 放行）。
**结论**：(a) 树方案下批边界不改变接受决定的机制是数值的，不是语义的；(b) design §10.2 的"贪心开关投机逐 token 相同"只能写成 margin 门限；
(c) G3（design §10.1.5）在短上下文已关闭，长上下文要在 Track M 的 4K / 17K 数据上重做一次（§10 问题 1）。

---

## 15. TPS 投影与 go / no-go

### 15.1 模型与输入

design v0.9 §10.1.2 的式子，按本轮实测更新输入（`dspark_tree.py tps`，每个常数在代码里注明出处）：

```
TPS(k)   = tokens_per_verify(k) / ( T_draft + T_CPU + T_verify(M = k+1) + stall(M) )     k = 0：无草稿
T_verify = T_attn+head(M) + T_MoE(M) + (engram 3.9 + MoE host 1.0) · M + other 5.0 + 0.4
T_MoE(M) = 29.4 ms · max( (6·M·uf[M]·18.81 + 23.6) / (6·18.81 + 23.6),  t_pair(M)/t_pair(1) )
stall(M) = (1 − h) · 6·M·uf[M] · 18.81 MB · 40 / 4.5 GB/s
```

| 输入 | 值 | 来源 |
|---|---|---|
| tokens / verify | §12 的 **runtime 回放**（不是每事件均值） | 本文 |
| `uf[M]` | 1 / 0.830 / 0.748 / 0.695 / 0.659 / 0.625 | §12.6 |
| attention + head，M = 1 | **33.5 ms** | p2_attention.md §13（P3 默认） |
| MoE GPU，M = 1 | 29.4 ms（7 槽，含 fp8 shared） | p2_decode.md §10.1 |
| engram / MoE host / other / 命中步 stall | 3.9 / 1.0 / 5.0 / 0.4 ms | p2_decode.md §10.1 |
| `t_pair(M)` | 0.625 / 0.851 / 0.780 / 0.814 / 0.874 / 0.945 ms | kernel_p2_moe.md M 扫描（design §10.1.2 引用） |
| MoE 按并集字节缩放 | M = 6 读 3.27 倍字节 → **96 ms**（29.4 ms 在 M = 1 就是带宽受限的 186 GB/s） | design §10.1.3 缺口 1 的口径 |
| 场景 A / B | A：attention 族 + head 在 M 上平坦；**B：按 §8 实测 ×3.06 @ M = 5 线性增长** | §8 |
| `T_draft` | 19 ms（M = 5 head）/ 42 ms（head 循环 M = 1） | §8 |
| `T_CPU` | 0.2 ms（K = 16 树 + 比较，§13.1 的上界） | 本文 |
| h | 0.92（§3.1 内插）、0.95、1.0 | design §3.1 / p2_decode.md §12.3 |

一个 M = 1 的步：33.5 + 29.4 + 10.3 + stall 80.3 = **153.5 ms（6.52 tok/s，h = 0.92）**。
M = 6 的 verify（场景 A）：33.5 + 96.2 + 34.8 + stall 300.9 + 草稿 19.2 = 484.6 ms；场景 B 的 attention 是 119.8 ms，合计 570.9 ms。

### 15.2 结果（h = 0.92，`T_draft` = 19 ms；tok/s，括号里是对无投机 6.52 的倍数）

| 调度 | 贪心 旧单链 | 贪心 树 K=16 `eal` | 采样 旧单链 | 采样 树 K=16 |
|---|---:|---:|---:|---:|
| **场景 A**：固定 k = 1 / 2 / 3 / 4 / 5 | 7.40 / **8.27** / 7.74 / 7.71 / 6.88 | 7.40 / 8.00 / 7.64 / 7.71 / 6.88 | 7.20 / **7.63** / 7.33 / 7.05 / 6.65 | 7.19 / 7.64 / 7.22 / 6.99 / 6.65 |
| 场景 A：confidence θ = 0.5 | **8.63（×1.32）** | 8.57（×1.32） | **8.55（×1.31）** | 8.48（×1.30） |
| **场景 B**：固定 k = 1 / 2 / 3 / 4 / 5 | 6.92 / **7.45** / 6.80 / 6.65 / 5.84 | 6.92 / 7.20 / 6.71 / 6.65 / 5.84 | 6.73 / **6.87** / 6.44 / 6.08 / 5.64 | 6.72 / 6.88 / 6.34 / 6.03 / 5.64 |
| 场景 B：最好的固定 k | ×1.14（k = 2） | ×1.10 | ×1.05 | ×1.06 |
| 场景 B：confidence θ = 0.5 / 0.7 | **7.73（×1.19）/ 7.79（×1.20）** | 7.66 / 7.65（×1.17） | **7.71（×1.18）**/ 7.48（×1.15） | 7.65（×1.17）/ 7.49（×1.15） |
| 场景 B、head 循环（`T_draft` 42）、θ = 0.5 | 7.26（×1.11） | 7.20（×1.11） | 7.21（×1.11） | 7.16（×1.10） |

其他 h（场景 B、`T_draft` 19）：h = 0.95 时固定 k 最好 ×1.05–1.14、confidence θ = 0.5 为 ×1.17–1.18（贪心单链 9.59 对 8.11 tok/s）；h = 1.0 时 **16.0 tok/s 对 13.66（×1.17）**；
场景 A 在 h = 1.0 最好 **×1.45–1.50**（固定 k = 4 或 confidence）。完整表：`dspark_tree.py tps --json`。

**读法**：

1. **树采样对 TPS 没有贡献**：贪心 −0…−4%，采样 ±1%。它在温度 1 的价值是"比较可以在 CPU 上以 < 1 KB 读回完成"（§12.4），不是速度。
2. **接受率不再是瓶颈**：正常文本上 runtime 回放 2.6–3.3 tokens / verify，比 Track K 的 1.93 高 40–70%。§10.1.4 的"k = 3 需要 `E[tokens]` ≥ 2.1–2.2 才打平"已经过线。
3. **confidence 调度是最大的杠杆**：场景 B 下固定 k 最好只有 ×1.05–1.14，confidence θ = 0.5–0.7 是 **×1.15–1.20**——它把平均 k 压到 ≈ 2（M ≈ 3），
   避开 stall 与 MoE 并集随 M 的增长，同时在高置信的段落（代码）照样 k = 5。
4. **stall 仍是主导项**：M = 6 的 stall 301 ms 占 verify 周期的 62%。h = 0.92 → 1.0 让场景 A 的倍数从 ×1.32 升到 ×1.50。

### 15.3 仍需 M > 1 的 kernel

与单链方案完全相同（§6 表、§6.1）：**attention 族**（`mega_mhc` / `wq_a` / `wq_b` / `wkv` / `sparse_attn` / compressor / indexer / `wo_a` / `wo_b` / `gate`，今天全部 M = 1）、
**head**（verify 的 M 行 + 草稿的 M = 5，今天 M = 1）、**engram**（M 行，今天 M = 1）、**MoE 的并集形态**（> 7 槽或多组 dispatch）。
树方案只额外要一个"top-K + gather + lse（+ 屏蔽 Gumbel 样本）"的一次扫描小 kernel。

### 15.4 Go / no-go

design §10.1.5 的决定规则：某个调度的预测 TPS ≥ 1.15 × 无投机 → GO。

* **树采样（本轮要评估的方案本身）：NO-GO 作为加速手段**——在两个温度下都不比单链快（§15.2 第 1 条）。
  保留 `cpu/dspark_tree` 作为温度 1 的 CPU 侧无损比较器（`accept_sampling_exact`）与 confidence 计算；路径目标用格内 `chain`（= 单链）。
* **DSpark 投机本身（单链草稿 + confidence 选 k + 精确接受）：有条件 GO。**
  在**实测的** M 缩放（场景 B）、M = 5 的草稿 head、h = 0.92 下，confidence θ = 0.5 预测 **×1.18–1.19**（贪心 7.73、采样 7.71 tok/s 对 6.52），刚过 1.15 门槛；
  固定 k 过不了（最好 ×1.14）。条件，缺一不可：
  1. **G2**：M > 1 的 attention 族 / head / engram kernel 实际做出来，**每层 M = 6 的代价不劣于 §8 的 ×3.06 @ M = 5 线性外推**，且草稿 head 是 M = 5（head 循环时只剩 ×1.11，NO-GO）；
  2. MoE 并集的实际代价不劣于"按字节缩放"（M = 6 ≈ 96 ms）；
  3. **G1 的样本要扩大**：30 个周期 / 模式、56 token 的 prompt、5 个 prompt 里代码占两个；×1.18 与门槛只差 3%，
     把 zh_prose / markdown 那样的低接受文本比例提高就会掉到门槛下。至少要 ≥ 64 个 decode token × 5 个 prompt × 两个模式、并在空闲机上用 GPU runtime 实测；
  4. **G3**：短上下文已关闭（§14）；长上下文的逐 query key 源要在 4K / 17K 上重测一次。

今天 G2 的 kernel 都不存在，**所以状态仍是"不集成"，但门槛已经从"每个 k 都亏"变成"做出 M > 1 kernel 后按 confidence 调度可以赚 15–20%"**。
