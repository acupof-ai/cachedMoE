# tools/web — 本地网页对话界面

一行启动（然后打开 <http://127.0.0.1:8080/>）：

```powershell
.venv\Scripts\python.exe tools\web\server.py --max-context 524280
```

（`--max-context 524280` 就是引擎硬上限；不给 `--cache-gb` / `--cache-slots` 就是 `auto`。
启动约 75 秒。当前跑着的实例见 `RUNNING.txt`。）

**`auto` 是安全的（Track H1a，2026-09-19）**：算出来的预算先封顶到 **5,000 槽 ≈ 87.6 GiB**
（`DEEPMOE_CACHE_SLOT_CAP` 可改，0 = 关），然后**建完 slab 池探一次提交**——
这正是 over-size 被发现的那一刻——被拒就退 200 槽（`DEEPMOE_CACHE_BACKOFF_SLOTS`）重建，最多 5 次。
在这之前 `auto` = 5,100 槽 / 89.3 GiB，而 5,100 在这台机器上第一个 decode submit 就丢设备，
所以 `RUNNING.txt` 里的命令一直手写着 `--cache-slots 5000`。
**要让它生效得重新链接 `build\deepmoe.exe` 并重启 server.py**；
手写的 `--cache-slots` / `--cache-gb` 一样会被探，但**不会被悄悄改小**，失败就是致命错并报下一个该试的数。

`server.py` 把 `deepmoe serve`（docs/p3_chat.md §1）当子进程拉起来，自己不碰 GPU；
提示词用 checkpoint 自带的 `encoding/encoding.py` 渲染，和 `tools/chat.py` 逐字一样，
所以同一个 seed 在网页和命令行得到同一条回复。

常用参数：

| 参数 | 说明 |
|---|---|
| `--max-context N` | 引擎 KV 位置数，默认 65536；**硬上限 524280**（见下）。只是上限，KV 实际从 4096 位置起按需翻倍，所以直接给满不花钱 |
| `--cache-gb N` / `--cache-slots N` | 专家缓存 |
| `--kv-dir DIR` | SSD 上的 `.pkv` 前缀/挂起 KV 缓存（不给就用 serve 的默认目录） |
| `--kv-max-gb N` | 该目录的预算 |
| `--max-parked N` | 内存里最多挂起几个会话，其余落盘 |
| `--port 8080` / `--host 127.0.0.1` | 监听地址 |
| `--think` | 默认开思考模式（网页上也能勾） |
| `--exe build\deepmoe.exe` | 引擎二进制 |

## 上下文硬上限：524,280 token

`Engine::max_context() = min(KvStoreConfig::max_context, kMaxIndexPositions)`，
而 `kMaxIndexPositions = 65535 * kIdxScoreTile(8) = 524,280`
（`runtime/decode_layer.h`）：`indexer.score` 一个 workgroup 覆盖 8 个压缩位置，
一次 dispatch 最多 65,535 个 workgroup。

这条限制**以 token 计**而不只是以压缩位置计，因为 checkpoint 最后一个 KV source
（第 20 层）的 `compress_ratio == 1`——它每个 token 存一行压缩 KV，所以那个平面上
`n_cmp == positions`。前三个 source（2/8/14 层）是 ratio 2，不是瓶颈。

其余的账：

- 活跃 KV（bf16）**3,200 B/token**（2,560 压缩 + 640 index key，按 ratio 2/2/2/1 加权），
  跑满 524,280 位置是 **1.68 GB**；打包成 `.pkv` 是 design §11 的 **894 B/token** = 469 MB。
- `KvStore` 切 slab（每块 ≤ 2 GiB），**~600K 位置以后才需要第二块**，所以
  524,280 这个上限先于内存生效。
- SWA 窗口 128，`topk_rows = 128 + min(512, max_context) = 640`，在
  `kAttnScoreStride = 1024` 之内，与 max_context 无关。
- 解码路径里没有残留的 16,384 / 65,536 常量：`idx_score` / 候选块平面都是按
  `kMaxIndexPositions` 一次性分配的（2 MB + 0.5 MB，在 32 MB scratch 里）。

`server.py` 在启动时拒绝 `--max-context > 524280`，并在**发送前**用 serve 的
`tokenize` 算出提示词 token 数，超过本次启动的 `max_context` 就直接拒绝并说明原因。

## 长文档

- 粘贴框接受多 MB 文本；输入停下 0.35 秒（超长时 0.9 秒）后调 `/api/preview`，
  显示「提示词 N token（复用 R，需预填充 P）· 预计 T」。
- 预填充走 decode 路径，**约 24 ms/token**（会用实测值自动校正）：
  10 万 token ≈ 40 分钟，100 万 token ≈ 6.7 小时（但 100 万超上限）。
- 需预填充 ≥ 20,000 token 时发送前会二次确认。

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
