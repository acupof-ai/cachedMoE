# Track R2：KV 账、回退、持久化与 serve UX（P4）

状态：**部分完成**。per-source KV 平面、engram 常量、session 回退/挂起、serve 中断已实现；
SSD prefix KV 持久化（design §11.4）已实现（本环境无法编译验证，测试见 `suite.kvdisk`）。本文件同时记录已验证事实、已知失败和复验命令。

## 1. 已实现（R2 分支提交）

- `runtime/kvstore.*`：KvStore 平面只开在实际写 KV 的 source 层上，按上下文增长；window 用
  bounded replay 重建；FP4 pack/unpack 与非 SWA 行 backup/restore 逐位。
- `runtime/engram_tables.*` + `tools/gen_engram_norm.py`：engram hash 常量从 tokenizer.json /
  config.json 推导，对 L3 导出的 129,280 个 token-map 项一致。
- `runtime/session.*`、`cli/serve.cpp`、`tools/chat.py`：rollback、park/restore、
  serve 中断/回退的最小 UX。
- `tests/test_kv_replay.cpp`、`tests/test_kvstore.cpp`、`tests/test_engram_tables.cpp`。
- `tools/serve_demo.py`。

## 2. 已验证结果（来自 kv_replay.l3_64，2026-09-15）

- (1) 从导出状态连续 decode：**8/8 教师强制 OK**。
- (2) rollback 72→68、72→66：0 个重放位置，重解码 logits 与连续运行**逐位相同**（4/4、6/6）。
- (3) restore 导出状态：replay [0,64) 耗时 **102 s**，ring 平均 cos **0.969**（最差层 L37 0.934，
  最差 slot 0.650），随后 **FAIL**：
  `runtime::restore_context(...) -> cancelled: timeline wait for 5579 timed out`。

## 3. SSD prefix KV：设计已定，代码未落

按 design §11.4：

- 目录 `kvcache/`，按 prompt 前缀 hash 分文件；CLI/serve 用 `--kv-dir` 或 `DEEPMOE_KV_CACHE`。
- 只存四样：token ids、compressed KV（只含真正写过的 source 层行）、index keys、compressor state。
  **绝不存 SWA window，也不存 encoder 输出**；四样都按位置可截断，所以前缀的前缀也能命中。
- 命中时：读 SSD -> 重建 KvStore -> 只 replay 最后 R 个 token（全 40 层，R ∈ {128, 256}）
  建回 window -> 继续 decode。
- 文件 header 带版本与模型指纹；损坏/指纹不符回退到重 prefill；目录大小上限 + LRU 清理。

## 4. 下一步

1. 先修 `restore_context` 超时（timeline 依赖/allocator/并发），让 kv_replay 第 3 段全绿——
   SSD restore 复用同一条路径。
2. 实现 §3 的文件格式与 CLI；用 kv_replay 增加 park→SSD→restore、截断、指纹不符、SWA 不落盘断言。
3. 在同一 4K prompt 上量 TTFT：无缓存 vs SSD prefix hit（只加载 + replay R），记录
   TTFT、prefill token 数、SSD 读字节、首步 hit；对比 R=128/256。
4. 收尾 serve 多会话/中断 UX，写进本文件。

## 5. 复验命令

```powershell
$env:PATH="C:\Program Files\CMake\bin;C:\msys64\ucrt64\bin;$env:PATH"
$env:VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
$env:DEEPMOE_MODEL_DIR="D:\models\DeepSeek-V4.1-Flash"
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\tests\deepmoe_tests.exe kv_replay.l3_64
```

第 3 段当前失败；修复前 PR 不得标记 R2 完成。

## 6. 2026-09-16 追加：SSD prefix cache 实现

- API：`runtime/session.h` 的 `KvDiskOptions` / `save_parked_context` /
  `load_parked_context` / `drop_parked_context`；`SessionPool` 在 park、LRU 淘汰时落盘，
  在新进程 `activate(name)` 时回读。文件 `<dir>/<session>.pkv`（原子写 `.tmp` -> rename，
  目录级 LRU 上限）。
- 文件内容：token ids + `KvPacked`（compressed rows / index keys / FP4 打包 / carry）；
  **window 不写**，恢复仍走 `restore_context()` 的 bounded replay。header 带版本和
  `model_tag` 的 FNV-1a 指纹，指纹不符按 miss 处理。
- CLI：`deepmoe serve --kv-dir DIR [--kv-max-gb N]`；`tools/chat.py --kv-dir DIR`。
- 测试：`tests/test_kv_replay.cpp` 新增纯 CPU `DEEPMOE_TEST(kvdisk, roundtrip)`，
  CMake 注册 `suite.kvdisk`（无模型依赖）。
- **未验证**：无法在本环境编译；TTFT 对照（无缓存 vs SSD hit、R=128/256）需在能构建的机器上跑。

## 7. 2026-09-16 构建/测试更新

- `runtime/session.{h,cpp}` 的 SSD API 与 `kvdisk.roundtrip` 已编译并通过（CPU，无模型）。
- `kv_replay.l3_64` 在安静机上**全绿**：restore [0,64) 约 52 s、随后 teacher-forced 8/8、
  free-running 8/8；此前 `timeline wait for 5579 timed out` 是四条 track 并发抢 GPU 导致。
- 顺带修掉 compressor 的 E4M3 scale/字节不一致：`kv_replay` 场景 (4) 的 `raw_rows()`
  从 9 变为 **0**。SSD persist 沿同一 packed 格式，受益于该修复。
- 仍未做：无缓存 vs SSD prefix hit 的端到端 TTFT 对照（R=128/256）。
