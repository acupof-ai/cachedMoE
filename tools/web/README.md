# tools/web — 本地网页对话界面

一行启动（然后打开 <http://127.0.0.1:8080/>）：

```bash
python3 tools/web/server.py --max-context 1048576
```

（`--max-context 1048576` 就是引擎硬上限；不给 `--cache-gb` / `--cache-slots` 就是 `auto`。
启动约 75 秒。当前跑着的实例见 `RUNNING.txt`。）

**`auto` 是安全的（Track H1a，2026-09-19）**：算出来的预算先封顶到 **5,000 槽 ≈ 87.6 GiB**
（`DEEPMOE_CACHE_SLOT_CAP` 可改，0 = 关），然后**建完 slab 池探一次提交**——
这正是 over-size 被发现的那一刻——被拒就退 200 槽（`DEEPMOE_CACHE_BACKOFF_SLOTS`）重建，最多 5 次。
在这之前 `auto` = 5,100 槽 / 89.3 GiB，而 5,100 在这台机器上第一个 decode submit 就丢设备，
所以 `RUNNING.txt` 里的命令一直手写着 `--cache-slots 5000`。
**修改引擎后需要重新编译 `build/deepmoe` 并重启 server.py**；
手写的 `--cache-slots` / `--cache-gb` 一样会被探，但**不会被悄悄改小**，失败就是致命错并报下一个该试的数。

`server.py` 把 `deepmoe serve`（docs/p3_chat.md §1）当子进程拉起来，自己不碰 GPU；
提示词用 checkpoint 自带的 `encoding/encoding.py` 渲染，和 `tools/chat.py` 逐字一样，
所以同一个 seed 在网页和命令行得到同一条回复。

常用参数：

| 参数 | 说明 |
|---|---|
| `--max-context N` | 引擎 KV 位置数，默认 1048576；**硬上限 1048576**（见下）。KV 从 4096 位置起按需翻倍；解码 score scratch 预留 1M，GPU prefill 工作集另受显存余量限制 |
| `--cache-gb N` / `--cache-slots N` | 专家缓存 |
| `--resident-only mask` | miss 专家的权重置零；有损模式，默认仍为 off |
| `--mask-cache dynamic\|fixed` | mask 默认走动态 LRU/P0 异步加载；fixed 显式冻结初始集合，会降低长对话质量 |
| `--kv-dir DIR` | SSD 上的 `.pkv` 前缀/挂起 KV 缓存（不给就用 serve 的默认目录） |
| `--kv-max-gb N` | 该目录的预算 |
| `--max-parked N` | 内存里最多挂起几个会话，其余落盘 |
| `--port 8080` / `--host 127.0.0.1` | 监听地址 |
| `--think` | 默认开思考模式（网页上也能勾） |
| `--exe build\deepmoe.exe` | 引擎二进制 |

## 上下文上限：1,048,576 token

网页默认及引擎容量上限现在为 `1 << 20`，与 checkpoint 的
`max_position_embeddings` 一致。之前的 524,280 是索引评分的一维 dispatch
限制（65,535 workgroup × 8 位置）。现在 decode 和投机 verify 的索引评分都将
超长位置分到 X/Z 轴；1M 的网格为 `(32768, M, 4)`，Y 仍为验证行。
整批评分仍只有一次 dispatch、一次主模型前向，没有重复验证树。

最后一个 KV source（第 20 层）的 `compress_ratio == 1`，因此它的压缩行数
也按 token 数增长。引擎将 score/candidate 平面按 1M 配置，KV 从 4K 容量
按需增长并分 slab；活跃 bf16 KV 约 3,200 B/token，1M 约 3.13 GiB，
不含 window、scratch 和挂起会话。SWA/top-k 列表仍最多 640 行。

`server.py` 在启动时拒绝高于 1M 的参数，发送前仍校验本次引擎实际
`max_context`；提示词必须小于该值，以留出输出位置。

## 长文档

粘贴框接受多 MB 文本，`/api/preview` 返回提示词、复用和待预填充 token 数。
容量上限与一次 GPU prefill 的可运行长度是不同条件：当前 GPU prefill 工作集
按完整提示词分配，超出专家 cache 旁的 GPU heap 余量会在提交前拒绝，
不能靠提高 `--max-context` 消除。1M 完整输入尚未做端到端质量验证。
长文预填充耗时使用当前运行实测估计；未预填充满 1M 的速度不作承诺。
需预填充 ≥ 20,000 token 时发送前仍二次确认。

## 并发

serve 的主循环是串行的：一次只从 inbox 里取一个请求，`Session::generate` 跑完才取下一个。
Track MS 的多流调度（docs/p4_multistream.md）只能通过 `generate_multi` 走，而它要求
一轮里所有 turn 一次性交进去——那是离线批量的形状，不是「谁先按发送」。
所以这里**排队**：一次一个在跑，等待的标签页看到「排队中：前面还有 N 个请求」。

真正干活的是 serve 的**命名会话**（docs/p4_kv_ux.md §7）：一个标签页 = 一个会话名，
serve 让一个活在 KV 里、其余挂起，切回来只 replay ≤ 128 个 token。
改会话名字段就是换会话。

## 界面

状态栏：上下文 / max_context、tok/s、命中率、盘等待 ms/token、首字延迟、预填充 ms/token、队列。
控件：温度、top_p、最长 token、种子、思考模式、逐 token 概率（hover 显示 p / hit / step_ms / id）、
新对话（= `{"op":"reset"}`）、停止（= `{"op":"cancel"}`）。
Ctrl+Enter 发送。转写用 markdown-lite：```` ``` ```` 代码块、`` ` `` 行内代码、换行。

无 CDN 依赖，断网可用。
