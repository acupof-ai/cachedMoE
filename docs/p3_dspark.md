# P3 — DSpark 投机解码：验证过的算法规格

设计对应：[design.md](design.md) §2.4（DSpark 条）、§7.12、§9.5、§10（全部）、§12、§13.4。
本文是 **Track K 交给 Track I 的集成规格**：草稿周期的每一步算什么、验证循环怎么写、
哪些 buffer 要存在、今天哪些 kernel 支持 M > 1、以及实测的接受率 / expert 并集 / `T_draft`。

产物：

| 东西 | 路径 |
|---|---|
| oracle + 验证循环模拟器 | `tools/oracle_dspark.py` |
| 黄金数据（position 64 全阶段 + 两个薄记录 + 统计） | `tests/data/dspark/` |
| 草稿 kernel | `gpu/shaders/dspark_{common,gemv,attn,head}.slang`、`gpu/vulkan/dspark_kernels.{h,cpp}` |
| 逐阶段正确性 | `tests/test_gpu_dspark.cpp`（`ctest -R suite.gpu_dspark`） |
| 带宽 / 时间 | `bench/dspark_bench.cpp` → `bench/results/dspark_p3.csv` |

**参考实现从不跑投机路径**：`generate.py` 没有一处调用 `Transformer.forward_spec`，
`ModelArgs` 的注释也直说了（`model.py:129-130`：*"Only the forward pass is implemented here --
nothing calls forward_spec, so the speculative-decoding loop itself is out of scope for this repo."*）。
所以 §10 的验证循环是我们写的，本文 §3 是它的规格，§4 是为了跑它必须对参考做的唯一一处推广。

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

## 2. 参考实现里八个容易做错的点（本轮新读出来的）

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

---

## 3. 验证循环（design §10.2），实现并验证过的版本

### 3.1 状态

```
p        : 当前已确认 token `cur` 的绝对位置
cur      : 位置 p 上的 token（主模型还没吃过它）
drafts[] : 上一周期草稿出来的 k 个 token，覆盖位置 p+1 .. p+k
mh       : 位置 p-? 的 main_hidden —— 见下，实际用的是本周期算出来的
```

### 3.2 一个周期

```
1.  batch = [cur, drafts[0..k-1]]                       # M = k+1 个 token，位置 p .. p+k
2.  logits[0..M-1], main_hidden[0..M-1], routing = MainModel.forward(batch, start_pos = p)
3.  preds[i] = argmax(logits[i])                        # preds[i] 是位置 p+i+1 的预测
4.  a = 最长的 a 使得 preds[j] == batch[j+1] 对所有 j < a  （a ∈ [0, k]）
5.  本周期产出 a+1 个 token：drafts[0..a-1] 被确认，外加修正 token preds[a]
6.  若 a < k：回滚（§3.4），把 batch[0..a] 重新过一遍主模型拿到干净状态
7.  草稿：forward_spec(input_ids = preds[a], main_hidden = mh[a], start_pos = p + a)
8.  p ← p + a + 1;  cur ← preds[a];  drafts ← 新的 5 个
9.  调度器按 confidence 选下一周期的 k（§9）
```

第 7 步的位置对齐是整件事里最容易错的地方，所以写死在这里：
`forward_spec` 的 `start_pos` 是**产生 `preds[a]` 的那个输入 token 的位置**，即 `p + a`；
`main_hidden` 取 verify batch 的第 `a` 行；`input_ids` 是 `preds[a]`，它位于 `p + a + 1`。
于是草稿覆盖位置 `p + a + 2 .. p + a + 6`，而下一周期的 batch
`[preds[a], d1..dk]` 从 `p + a + 1` 开始 —— 首尾正好接上。

### 3.3 贪心下的不变量

`temperature == 0` 时投机解码产生的 token 序列与不投机**逐 token 相同**，
这是 design §10.2 写死的内建检查，也是本文所有统计的前提：
接受长度 `a` 只改变周期边界，不改变输出。
`tools/oracle_dspark.py` 每个周期都比较第 6 步重跑出来的 argmax 与 `preds[a]`（不一致时记录 `cycles[].divergence` 并沿用重跑的 token），
并记录两者 logits 的余弦（`cycles[].commit_logit_cos`，§7.1）。

### 3.4 回滚

被拒绝的草稿在位置 `p+a+1 .. p+k` 上写脏了四处状态：

| 状态 | 粒度 | 回滚方式 |
|---|---|---|
| 主模型 window KV 环（40 层 × 128 槽 × 512 × 2 B = 5.2 MB） | 逐位置 | 把 `(p+a+1..p+k) % 128` 这几格从周期开始的快照恢复 |
| Compressor 的 `kv_state` / `score_state`（ratio 2 的 18 层，每层 2 × 512 fp32） | 逐 slot | 快照整份（ratio ≤ 2，一层 8 KB）恢复，再按已接受位置重放 |
| `compress_kv_cache` / indexer `k_cache` | 逐组 | 只恢复 `> (p+a)/ratio` 的组 |
| mtp 三块的 window KV 环（393 KB） | 逐位置 | 同上；注意它写的是 `main_x` 推的 KV，位置与主模型一致 |
| `NgramHashState.cache`（engram 的 token 历史） | 逐位置 | **不需要回滚**：它只向后看（`positions - shift`），脏的是未来位置，下一周期原地覆盖 |

**因果性是这件事的支点**：位置 `p..p+a` 在 M=k+1 的 forward 里算出来的值，
和在干净的 M=a+1 forward 里算出来的**完全相同**（window attention 是因果的，
compressor 只 pool 已完成的组），所以"恢复快照 + 只重放已接受前缀"和"外科手术式撤销"等价。

`tools/oracle_dspark.py` 用的是前者（快照 + 重跑 `batch[:a+1]`），因为它同时充当
§4 那套推广的每周期自检。**runtime 不能这么干**（等于把每个周期的主模型跑两遍）。
Track I 要实现的是后者，按上表逐项撤销；`commit_logit_cos` 的实测值（§7.1）
就是"两者等价"这条断言的证据。

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
另外每个周期的"恢复快照 + 重跑已接受前缀"也是同一件事的弱化版自检（§3.3）。

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

---

## 7. 实测（`tools/oracle_dspark.py`，L3 prompt，64 token prefill）

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

### 7.2 接受长度

轨迹：prompt 结尾是 `return hits / total if total else`，贪心续写 `" else 0.0\n"` 之后连出两个 EOS，
再进入一段与语境无关的中文 —— **这是一条退化的轨迹**，接受率偏低是它的性质，不一定是 DSpark 的。
数字照实给，外推见 §9 的敏感度。

14 个周期（k 恒为 5，M = 6），verify 循环产出 27 个 token：

| 接受 a | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 周期数 | 6 | 5 | 2 | 0 | 1 | 0 |

* **平均接受 0.929，tokens / verify = 1.929**
* 任意 k 下 `E[tokens](k) = 1 + E[min(a, k)]`：k = 1..5 → **1.571 / 1.786 / 1.857 / 1.929 / 1.929**
  （由 k = 5 的观测截断得到；贪心下 token 序列与 k 无关、只有周期边界不同，所以是近似）
* 退化之前的唯一一个代码续写周期（golden）接受 4/5。

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

### 7.5 confidence head

**对齐**：一个周期记录的 `confidence` 属于它的 `next_drafts`，由**下一个**周期验证；cycle 0 验证的草稿来自 golden 草稿。
第一版统计配错了一位，已改（`summarise` 的 `conf_of`）。

| 草稿位 | 被测次数（前面全接受才测到） | 实际接受率 | 接受时平均原始分 | 拒绝时平均原始分 | 平均 sigmoid |
|---|---|---|---|---|---|
| 0 | 14 | 0.571 | **1.69** | **0.53** | 0.637 |
| 1 | 8 | 0.375 | **1.45** | **−0.62** | 0.429 |
| 2 | 3 | 0.333 | 5.68 | −0.43 | 0.600 |
| 3 | 1 | 1.0 | 1.62 | — | 0.835 |
| 4 | 1 | 0.0 | — | 0.46 | 0.612 |

位 0、1 上接受与拒绝的原始分分得开，sigmoid 均值与实际接受率同量级（0.64 vs 0.57，0.43 vs 0.38）。
样本太少，只能说"logistic 映射没有被否定"。golden 周期分数 `[7.46, 5.80, 5.68, 1.62, 0.46]`
对应 4 接受 1 拒绝，被拒的恰是分最低的那位。

### 7.6 run 2 为什么停在第 14 周期

`dsref.make_block` 构造层 1/14 的 `Block` 时，参考的 `Engram.__init__` 先
`torch.empty(384M, 256, dtype=fp8)` —— **98 GB 的未触碰占位**（随后被 `EngramRowEmbedding` 换掉）。
Windows 上它要 commit，共享机器上其他进程占着 commit 时就失败。这是 dsref 的既有行为；
每周期写的检查点让已测的 14 个周期没有丢。**建议**（本轮没改 dsref）：`make_block` 里临时把
`ref.ParallelEngramEmbedding` 换成 meta-device 占位。

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

## 9. 调度与预计加速（design §10.3 / §13.4）

模型（§13.4 的三项 + `T_draft`，h = 0.920，NVMe 用 §7.3 实测 `union_frac`）：

```
T_cycle(k) = T_非MoE(M) + 40 · T_MoE(M) + (1−h) · M · union_frac[M] · 112.8 MB · 40 / 4.5 GB/s + T_draft
tok/s(k)   = 1000 · E[tokens](k) / T_cycle(k)          M = k + 1，k = 0 时无 T_draft
```

`E[tokens](k)` 用 §7.2 实测 1.000 / 1.571 / 1.786 / 1.857 / 1.929 / 1.929：

| 场景 | k=0（无投机） | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|---|
| A：非 MoE 随 M 平坦（design 假设），T_draft 19 | **6.41** | **6.74** | 6.36 | 5.77 | 5.25 | 4.75 |
| A'：同上，T_draft 42（head 循环） | 6.41 | 6.16 | 5.90 | 5.40 | 4.96 | 4.51 |
| B：非 MoE 按 §8 实测倍率（×3.06/层 @M=5）线性增长，T_draft 19 | 6.41 | 6.09 | 5.40 | 4.68 | 4.13 | 3.63 |

**这条轨迹上投机解码最多 +5%（A, k=1），在实测的 M 缩放下是净亏。**
design §13.4 的"平均接受 2.5 → 9–13 tok/s"对不上，原因按大小排：

1. **NVMe 项随 M 涨得最快。** k=5 时 `T_nvme` 从 80 ms 涨到 286 ms：常驻读被摊薄，
   但 miss 字节按并集算 `M · union_frac[M]`（6 × 0.594 = 3.6 倍单 token），而一次 verify 只确认 1.93 个 token。
   §10.1 表里"expert 流量 ÷1.6"成立的前提是 5 个草稿**全被接受**。
2. **接受率低**（这条退化轨迹上 0.93，§7.2）。
3. **非 MoE 路径不平坦**（§8）。

敏感度（场景 A、T_draft 19）：k = 3（T_cycle 322 ms）**`E[tokens]` ≥ 2.07 才打平，≥ 2.9 到 9 tok/s，
全接受（4.0）也只有 12.4 tok/s**；k = 5（406 ms）打平要 2.6。
**需要在正常文本上重测接受率**（§10 问题 2），不能凭本轨迹下结论。

影子调度器（`schedule_k`，logistic 映射后的 confidence + 场景 A 曲线）14 个周期选的 k：
0 ×2、1 ×8、2 ×3、4 ×1 —— 低置信时少猜，方向对。

---

## 10. 未解决的问题

1. **贪心不变量对参考不成立**（§7.1）。runtime 对齐"逐 token decode"还是"verify batch"？
   前者要在 M>1 kernel 里复现"组未完成时读上一次 forward 的 index key"；后者要把 §10.2 的内建检查降级为 margin 门限。
   先要在 oracle 里把机制隔离（M=1 连续两步 vs M=2 一步，比较层 2 的 `topk_idxs`），本轮没做。
2. **接受率只有一条退化轨迹、14 个周期。** 需要 ≥ 5 个正常 prompt × ≥ 64 token（design §12 L3 的规模），
   CPU 上每周期 ~90 s，约 2–3 小时；应在机器空闲时跑，并先修 §7.6 的 98 GB 占位。
3. **非 MoE kernel 的 M>1**（§6）不是 Track K 的文件。`dspark_gemv.slang` 可作模板，但 §8 说明
   "加一个 m 循环"只买到 ×2.2–4.8 而不是 ×1 —— M>1 的 GEMV 需要自己的设计，而这正是投机解码能否赚钱的分界（§9 A vs B）。
4. **confidence → 概率的映射**是假设（§2 第 4 点），样本不足以校准。
5. **`T_nvme` 与投机的相互作用**：h = 0.92 时 NVMe 是投机的主要成本（§9 第 1 条）。
   verify batch 的 expert 集合精确已知（§9.5），Planner 可在 verify 开始前把并集 miss 全部下单；
   §13.4 的"完全串行"假设在这里偏保守得更多，值得单独建模。
6. **DsparkRunner 的地址表**：一个周期里 Gemv 每 stage 发 5 次、RmsNorm 3 次、RopeQuant 4 次，
   单表一次只挂一组 slot。bench 用了 16 个 runner；runtime 要么同样多表，要么在提交之间改 slot
   （那就不能一个命令缓冲录完整个周期）。按"dispatch 序号"而非"阶段"分片 slot 表更合适 —— 给 Track I 的接口问题。
7. **mtp KV 回滚**（§3.4）与**多位置 `main_x` 写环**（§4.6）在参考里没有对应代码，只由 oracle 的推广定义；
   GPU 端还没有它们的测试（本轮 golden 是 seqlen = 1 的周期）。
8. **`install_batched_decode` 本身**只由 crosscheck（argmax 6/6，但 cos 低至 0.945）和逐周期重跑间接验证；
   问题 1 的机制隔离清楚之前，不能排除其中还有一处与参考不同。
