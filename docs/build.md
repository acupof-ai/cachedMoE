# 构建与开发环境

目标平台：Windows 11 + Strix Halo。所有命令在 PowerShell 中执行。

## 工具链

| 工具 | 版本 | 来源 | 位置 |
|---|---|---|---|
| Zig（`zig cc` / `zig c++`，内置 clang 21 + lld + libc++） | 0.16.0 | `winget install zig.zig` | `%LOCALAPPDATA%\Microsoft\WinGet\Packages\zig.zig_*\zig-x86_64-windows-0.16.0\`（已加入用户 PATH） |
| Vulkan SDK（`slangc`、`glslc`、`spirv-val`、`spirv-dis`、validation layers、`vulkan-1.lib`） | 1.4.357.0 | `winget install KhronosGroup.VulkanSDK` | `C:\VulkanSDK\1.4.357.0`，`VULKAN_SDK` 已设为机器环境变量 |
| CMake | 4.x | 已安装 | `C:\Program Files\CMake\bin` |
| Ninja | | msys2 ucrt64 | `C:\msys64\ucrt64\bin\ninja.exe` |
| MSVC Build Tools（A/B 对照编译器） | 14.44 | 已安装 | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| Python + uv | 3.12 | 已安装 | `uv` 管理虚拟环境 |
| modelscope CLI | 1.37 | `pip install modelscope` | 权重下载 |
| AMD Adrenalin 驱动 | 32.0.31041.1004 (Vulkan 1.4.349) | | |

新开的终端才会看到 PATH / `VULKAN_SDK` 的变更。

## 为什么用 Zig 做编译器

- 一个 ~90 MB 的下载得到完整的 C/C++20 工具链，无需 Visual Studio；`zig c++` 就是 clang，`-march=znver5` 直接可用。
- 目标三元组可切换：`x86_64-windows-gnu`（默认，libc++ + mingw CRT，已验证）或 `x86_64-windows-msvc`（链接 MSVC CRT，用于 A/B）；`x86_64-linux-gnu` 交叉编译给 CI 做纯 CPU 单元测试。
- 与 CMake/Ninja 配合通过 `cmake/zig-toolchain.cmake` 与几个 `.cmd` 包装脚本完成（CMake 要求编译器是单个可执行文件）。

已知噱头：Zig 0.16 首次以 gnu 目标构建时会从源码编译 libc++，并输出大量 `-Wnullability-completeness` 警告；只发生一次（结果缓存在 `%LOCALAPPDATA%\zig`），工具链文件里已加 `-Wno-nullability-completeness`。

## C++ 标准：代码是 C++20，编译用 `-std=c++23`

设计写的是 C++20，代码也按 C++20 写（除 `std::expected` 外不用 C++23 特性）。但
`std::expected` 定义在 `<expected>` 里，libc++ 把整个头文件挡在 `_LIBCPP_STD_VER >= 23`
之后，`-std=c++20` 下 `core/status.h` 直接编不过。因此 `cmake/deepmoe_options.cmake` 里
`DEEPMOE_CXX_STANDARD = 23`。这是骨架阶段唯一被迫偏离设计文档的地方。

另一个 libc++/mingw 的小坑：带 size 的对齐 `operator delete(void*, size_t, align_val_t)`
在这个目标下没有声明，`core/align.h` 用的是两参数的 `operator delete(void*, align_val_t)`。

## 配置与构建

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

选项（`cmake/deepmoe_options.cmake`）：

| 选项 | 默认 | 作用 |
|---|---|---|
| `DEEPMOE_BUILD_TESTS` | ON | 构建 `deepmoe_tests` 并注册到 ctest |
| `DEEPMOE_ENABLE_VULKAN` | ON | Vulkan 后端；关掉后 `gpu/` 各类返回 `Unavailable`，其余照常编译 |
| `DEEPMOE_ENABLE_DIRECTSTORAGE` | OFF | DirectStorage 后端；还要 `dstorage.h` 在 include 路径上（SDK 不在仓库里，也不下载） |

产物：

```
build/deepmoe.exe              CLI: info / bench nvme / run
build/nvme_bench.exe           design §9.2.1 Q6/Q7 微基准
build/bw_matrix.exe            design §8.0 / §3.3 带宽矩阵（CPU / GPU / 并发 × 两条路径）
build/kernel_bench.exe         design §7.9.1（--p1）/ §7.9.2（默认）MoE kernel sweep
                               + dispatch 开销 + 路径 A/B
build/attn_bench.exe           design §7.15.2 非 MoE decode 路径逐 kernel 带宽
build/heap_capacity.exe        design §9.2.2 两条路径的实际可分配上限
build/envcheck.exe             环境自检
build/tests/deepmoe_tests.exe  单元测试
build/shaders/*.spv            §7.14 的 16 个 kernel，每个都过 spirv-val
```

Git Bash 下（首次 zig 构建 libc++ 会刷一屏 `-Wnullability-completeness`，过滤掉再看）：

```bash
export PATH="/c/Program Files/CMake/bin:/c/msys64/ucrt64/bin:$PATH" VULKAN_SDK="C:/VulkanSDK/1.4.357.0"
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build 2>&1 | grep -vE 'warning:|note:|_Nullable|_Nonnull|In file included'
```

## 测试

```powershell
.\build\tests\deepmoe_tests.exe            # 全部
.\build\tests\deepmoe_tests.exe io         # 只跑名字含 "io" 的用例
.\build\tests\deepmoe_tests.exe --list
ctest --test-dir build --output-on-failure  # 整体 + 按 suite 各注册一遍
```

测试框架是 `tests/test_framework.h`，约 150 行，自写（设计 §14：不引第三方依赖）。
`tests/data/v41_config.json` 是 ModelScope 上 V4.1-Flash 的真实 `config.json` 原样拷贝，
`tests/test_model.cpp` 拿它逐字段核对 §2.1 的每一个数字。

## Linux 交叉编译（CI）

Linux 不是目标平台；交叉编译只是为了让可移植的一半（core/model/cpu/storage/store）
在没有 Windows 的 CI 上也能编译并跑单元测试，并且保证 `storage/linux/io_uring.cpp`
不腐烂：

```bash
zig c++ -target x86_64-linux-gnu -std=c++23 -I. -c storage/linux/io_uring.cpp -o /tmp/io_uring.o
```

## 基准

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'

# NVMe 微基准（Q6/Q7）→ design §9.2.1
.\build\nvme_bench.exe --reads 48 --csv bench\results\nvme_q6_q7.csv

# 带宽矩阵：CPU 单独 / GPU 单独 / 并发 × 路径 A/B → design §8.0、§3.3
.\build\bw_matrix.exe --cpu-size-gb 4 --size-gb 1 --repeats 3

# P2 的 MoE kernel sweep（261 行：M 扫描 × XMode × HQuant × fp8 shared expert × 分组 dispatch）
#   → design §7.9.2、docs/kernel_p2_moe.md v0.1
.\build\kernel_bench.exe --csv bench\results\kernel_p2_moe.csv `
    --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2

# P2 step 2 的那一轮（336 行：加上 HQuant=3、XMode=6、轮转测量的分组 dispatch）
#   同一条命令，因为 step 2 的变体是加上去的 → design §7.9.3、docs/kernel_p2_moe.md v0.2
.\build\kernel_bench.exe --csv bench\results\kernel_p2b_moe.csv `
    --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2

# 只重测一节。一整轮 25 分钟，足够让芯片热几度（见下面的量测纪律）
.\build\kernel_bench.exe --only "h fp8"  --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2
.\build\kernel_bench.exe --only "int8 x" --iters 48 --layer-cycle 8 --repeats 3 --sweeps 2
.\build\kernel_bench.exe --only none     --iters 48 --layer-cycle 8 --repeats 3  # 只剩分组 dispatch 那一节

# P1 的旋钮 sweep（解码方式 / lane 数 / wave 宽度，60 变体）+ dispatch 开销 → design §7.9.1、§3.4
.\build\kernel_bench.exe --p1 --csv bench\results\kernel_p1.csv `
    --iters 48 --layer-cycle 8 --repeats 4 --sweeps 2
.\build\kernel_bench.exe --path b --quick        # 路径 B 对照

# 非 MoE decode 路径的逐 kernel 带宽 → design §7.15.2、docs/p2_attention.md
#   --layers 8 不是可选的：每层一个 AttnRunner，否则 32 MB 的 MALL 会给出幻觉
#   （第一版把 wq_a 报成 374 GB/s，在一个 217 GB/s 的内存系统上）
.\build\attn_bench.exe --layers 8 --iters 64 --rows 2 --csv bench\results\attn_p2.csv

# 容量上限 → design §5.2 / §9.2.2。**必须在空闲机上跑**（见上面的 pagefile 一节）
.\build\heap_capacity.exe --slab-gib 2 --min-free-gib 6 --csv bench\results\heap_capacity_idle.csv
```

`bench/results/` 里的 CSV 与它们对应的文档：

| CSV | 文档 |
|---|---|
| `nvme_q6_q7.csv` | design §9.2.1 |
| `bw_matrix.csv` | design §8.0 / §3.3 |
| `kernel_p1.csv`、`kernel_p1_path{a,b}.csv` | design §7.9.1、[kernel_p1.md](kernel_p1.md) |
| `kernel_p2_moe.csv` | design §7.9.2、[kernel_p2_moe.md](kernel_p2_moe.md) v0.1（step 1） |
| `kernel_p2b_moe.csv` | design §7.9.3、[kernel_p2_moe.md](kernel_p2_moe.md) v0.2（step 2）。**不要和上一行比绝对值** |
| `attn_p2.csv` | design §7.15.2 / §7.15.5、[p2_attention.md](p2_attention.md) |
| `heap_capacity.csv` | design §9.2.2 第一轮（4 GiB pagefile，历史） |
| `heap_capacity_idle.csv` | design §5.2 / §9.2.2 第二轮（96 GiB pagefile，**这是当前的那一份**） |
| `heap_capacity_pagefile128.csv` | 同上设置但**有并发污染**，作为量测卫生的反面教材保留 |

`nvme_bench` 默认在 `%TEMP%` 建一个 2 GB 测试文件，跑完删除（`--keep` 保留，`--file`
指定已有文件）。实测结果写回 design.md §9.2.1。

**测量纪律（否则数字没有意义）**：LPDDR 是 CPU 和 GPU 共用的，**任何吃内存带宽的进程都会
污染 GPU 的数字**——实测同一个 kernel 变体在机器忙 / 闲两种状态下相差最多 **50%**
（164.6 vs 94.5 GB/s）。所以：

- **不要同时编译**（一次 `cmake --build` 撞上 `bw_matrix`，CPU 峰值从 100.9 掉到 97.7，
  并发那一栏直接作废），也不要同时跑 `tools/route_trace.py`。
- `kernel_bench` 的工作集要用 `--layer-cycle 8`（1053 MB）跨过 32 MB 的 MALL；
  只用 1 层（132 MB）会把带宽**虚高约 25%**。`attn_bench` 的对应旋钮是 `--layers 8`
  **加上每层一个 `AttnRunner`**（否则一个 command buffer 里每次迭代读的都是同一层，
  小于 MALL 的东西会报出幻觉）。
- **容量测量同样适用**：`heap_capacity` 在有并发基准时满载 raw-read 报 152 / 121 GB/s，
  空闲时是 216 / 207（design §5.2）。
- 判据是 raw-read 上限在 sweep 前后一致。完整的方法与两个踩过的坑见
  [kernel_p1.md](kernel_p1.md) §1。
- **跨轮只比同轮内的相对值。** P2 那一轮的 run 间漂移是 **~7%**（P1 记的是 1.5%），
  同一轮不同小节之间也有 8%（[kernel_p2_moe.md](kernel_p2_moe.md) §2）。
  **CI 上的 kernel 带宽回归必须做同轮对照，不能比绝对值。**
- **要量 1–5% 的差，对照必须和被测量的东西轮转着测**（v0.8 新增，
  [kernel_p2_moe.md](kernel_p2_moe.md) §11、design §7.9.3 (c)）。
  "同一节内"还不够：**顺序测量里后跑的那个系统性地更热**。
  design v0.7 有一整条结论（"拆分 dispatch 每层 +0.193 ms，比启动开销大 340 倍"）
  就是这么来的——`whole` / `first` / `rest` 隔着几秒钟依次跑，而要量的差只有 1–3%。
  `bench/kernel_bench.cpp` 的分组 dispatch 一节现在是**九个配置轮转、各取自己的最好值**，
  `h fp8` 与 `int8 x` 两节也各自带上了同节的不量化对照。
- **每一节都要有一个物理自检。** 上面那条结论之所以能站住三周，是因为没有人能从数字上看出它错了。
  轮转之后有了：**`A + B` 必须对得上 `whole`**（0.391 + 0.201 = 0.591 对 0.592）。
  顺序测量那一节量到过 `whole 0.652 ms = A 0.520 + B 0.290`——**A + B > whole，物理上不可能**，
  那就是三个数取自三个热状态的签名。**设计基准时先想好这个自检是什么。**

带 validation layer 跑一遍（`kernel_bench` / `attn_bench` 与四个 GPU 测试 suite 当前都是干净的；
`spirv-val --target-env vulkan1.3` 由 `add_slang_shader()` 每次编译都跑）：

```powershell
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
$env:VK_LOADER_LAYERS_ENABLE='VK_LAYER_KHRONOS_validation'
```

MSVC 目标 A/B：

```powershell
cmake -S . -B build-msvc -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DZIG_TARGET=x86_64-windows-msvc -DCMAKE_BUILD_TYPE=Release
```

Shader：CMake 规则对 `vulkan/shaders/*.slang` 调用

```powershell
& "$env:VULKAN_SDK\Bin\slangc.exe" kernel.slang -target spirv -profile spirv_1_6 -entry main -O2 -o kernel.spv
```

随后 `spirv-val --target-env vulkan1.3 kernel.spv`。

## 环境自检

`tools/envcheck/vkinfo.cpp` 是环境自检（C++20 + Vulkan + AVX-512 VNNI）。原来与它放在一起的 `gemv.slang` 已移到 `gpu/shaders/moe_gemv_fp4.slang`（它是 §7.9 所有 routed-expert kernel 的模板，不再只是一个冒烟测试），由 `shaders` target 统一编译。手工验证：

```powershell
zig c++ -std=c++20 -O2 -march=znver5 -Wno-nullability-completeness -I"$env:VULKAN_SDK\Include" tools\envcheck\vkinfo.cpp -L"$env:VULKAN_SDK\Lib" -lvulkan-1 -o build\vkinfo.exe; .\build\vkinfo.exe
```

```powershell
& "$env:VULKAN_SDK\Bin\slangc.exe" gpu\shaders\moe_gemv_fp4.slang -target spirv -profile spirv_1_6 -entry main -O2 -o build\gemv.spv; & "$env:VULKAN_SDK\Bin\spirv-val.exe" --target-env vulkan1.3 build\gemv.spv
```

2026-09-13 在开发机上的输出：

```
device: AMD Radeon(TM) 8060S Graphics  api 1.4.349
  maxStorageBufferRange = 4095 MiB
  heap[0] 37.2 GiB host
  heap[1] 74.4 GiB DEVICE_LOCAL
  type[2] DEVICE_LOCAL|HOST_VISIBLE (heap 1)   (另有 type 6/10/14)
  VK_EXT_external_memory_host                   yes
  VK_KHR_shader_integer_dot_product             yes
  VK_KHR_cooperative_matrix                     yes
  VK_KHR_timeline_semaphore                     yes
  VK_KHR_buffer_device_address                  yes
avx512-vnni dpbusd lane0 = 24 (expect 24)
```

## 模型权重下载

源：ModelScope `deepseek-ai/DeepSeek-V4.1-Flash`（510.3 GB，48 个 safetensors，含 DSpark `mtp.*`）。目标目录 `D:\models\DeepSeek-V4.1-Flash`。

**不能走代理。** 开发机 shell 里有 `HTTP_PROXY` / `HTTPS_PROXY` 指向本地代理，先清掉再下：

```powershell
$env:HTTP_PROXY=$null; $env:HTTPS_PROXY=$null; $env:ALL_PROXY=$null; $env:NO_PROXY='*'; modelscope download --model deepseek-ai/DeepSeek-V4.1-Flash --local_dir D:\models\DeepSeek-V4.1-Flash --max-workers 8
```

支持断点续传；重复执行即可补齐。验证连接是否直连：

```powershell
Get-NetTCPConnection -State Established | ? { (Get-Process -Id $_.OwningProcess).ProcessName -match 'python' } | Group-Object RemoteAddress | % Name
```

应看到 ModelScope CDN 的公网 IP（如 `116.136.x.x`、`47.92.x.x`），而不是 `127.0.0.1`。

下载完成后**不要 repack**（design §5.1 v0.5：D: 放不下第二份 510 GB）。只生成地址簿：

```powershell
uv run python tools/manifest.py --src D:\models\DeepSeek-V4.1-Flash
```

它读 `model.safetensors.index.json` 与 48 个分片 header，校验大小与 `total_size`，把
`deepmoe_manifest.json` 写进**同一个目录**（这是 deepMoE 往 checkpoint 目录里写的唯一文件；
原始分片只读）。实测 **0.9 s**，输出 9.9 MB。常用选项：

| 选项 | 作用 |
|---|---|
| `--out PATH` | 换个输出路径（默认 `<src>/deepmoe_manifest.json`） |
| `--verify` | 另外并行算 48 个分片的 sha256 并记进 manifest（要读 510 GB，几十分钟） |
| `--workers N` | 并行读 header / 算 hash 的线程数，默认 8 |
| `--dry-run` | 只扫描与打印 summary，不写文件 |
| `--indent N` | 输出可读的缩进 JSON（默认紧凑） |

summary 会打印 runs/expert 直方图、对齐后的 run 大小、**`kExpertSlotBytes`**、跨分片的层，
以及一致性问题列表。`kExpertSlotBytes` 必须与 `model/layout.h` 里的常量相同——
`Manifest::validate()` 和 `tests/test_model.cpp` 会因为不一致而失败，`manifest.py` 打印的就是要填的值。

## Python 环境

```powershell
uv venv
uv pip install numpy safetensors pyarrow
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install ml_dtypes      # 可选：oracle L0 的 FP4 表交叉校验
```

torch 只用 CPU 版。注意 **PyTorch 的 wheel index 里没有 `safetensors` / `numpy`**，所以要分两条
命令：普通包走 PyPI，torch 走 `download.pytorch.org`。开发机 shell 里的
`HTTP_PROXY` / `HTTPS_PROXY` 对 PyPI 通常是需要的（与 ModelScope 相反），先带着代理试，
不通再清掉重试。

## Oracle（design §12）

```powershell
uv run python tools/oracle.py --model D:\models\DeepSeek-V4.1-Flash --level l0 --out tests/data
uv run python tools/oracle.py --model D:\models\DeepSeek-V4.1-Flash --level l1 --out tests/data

# L2：七个层的逐级黄金张量（约 258 s，5.1 MB → tests/data/l2/）
uv run python tools/oracle.py --model D:\models\DeepSeek-V4.1-Flash --level l2 --out tests/data

# P2 需要的三个额外 golden：fp8 shared expert + 量化 h 的四个参考答案
uv run python tools/oracle_shared.py --model D:/models/DeepSeek-V4.1-Flash `
    --shared 0 --expert 0:0 --expert 39:383 --out tests/data

# L3：端到端。prefill 状态 + 八个 greedy step（约 459 s，7.1 MB → tests/data/l3/）
uv run python tools/oracle.py --model D:\models\DeepSeek-V4.1-Flash --level l3 --out tests/data

# compressor / indexer 需要的 L2 超集（约 305 s，2.35 MB → tests/data/l2x/）
uv run python tools/oracle_l2_extra.py --model D:\models\DeepSeek-V4.1-Flash --out tests/data
```

- `--level l0` 导出 FP4 E2M1 / FP8 E4M3 / UE8M0 三张解码表到 `tests/data/l0_dequant.bin`（2,132 B），
  `tests/test_dequant.cpp` 逐位比对。
- `--level l1` 默认跑 `(layer 0, expert 0)` 与 `(layer 39, expert 383)`（用 `--expert L:E` 指定，可重复）：
  六个 tensor 各读两遍（一遍走 manifest 的 run/skew，一遍走 `safetensors` 库）要求字节一致，
  再在 torch fp32 里算 expert FFN，把 `x`/`y`/校验和写进 `tests/data/l1_layer{L}_expert{E}.bin`。
  加 `--report out.json` 可以把数字存下来。
- **`--level l2`（P2 新增）**导出 `tests/data/l2/`：**5.1 MB，七个层
  （0 / 1 / 2 / 13 / 14 / 20 / 39），每层 ~40–53 个张量**，64 token prefill 之后位置 64 的
  那一个 decode step。它跑的是**未经修改**的 `inference/model.py`（只借 `tools/dsref.py`
  的六个 CPU kernel shim），每个张量要么是 forward hook 的输入/输出，要么是参考路过的
  模块级函数的实参/返回值——**没有任何东西是重新推导的**。量化点是**断言**出来的：
  导出器从量化前的张量重算 fp8/fp4 字节并要求它们反量化回参考的值。
  **约 258 s，几乎全部花在 prefill 的 MoE 上**（一层 ~200 个不同的 expert，要读 ~5 GB/层的
  attention 权重重建 Block）。解读见 design.md §12 与 [p2_attention.md](p2_attention.md) §1。
- `oracle_shared.py` 写出 `tests/data/l1_shared_layer0.bin` 与
  `y_ref` / `y_hq` / `y_full` / `y_hq16` 四个向量（design §7.9.2 (e)(f)）。
  它现在**顺带打印 int8 `x` 的量化对照**（fp16 / int8 每块 / int8 每行 / int8 + fp16 残差，
  design §7.9.3 (b)），不需要 GPU。
- **`--level l3`（P2 step 2 新增）**导出 `tests/data/l3/`：**7.1 MB，459 s**。
  内容是 prefill 之后每层的 window KV 环、八个 greedy step 的**逐步压缩 KV 与 indexer top-k**、
  每步的 logits，以及 engram 的 hash 常量表（`engram_token_map.bin`，517 KB）。
  **logits 存的是 top 64 个 (id, logit) 加上全 129,280 维的 max / log-sum-exp / min**——
  存全部要 4.7 MB 而且没有用：design §12 L3 问的每一个问题（argmax 对不对、margin 多大、
  排序往下还成不成立）都由分布的顶部回答，而那三个整向量统计量抓得住"top-64 一致但尾巴不一致"的实现。
  **两个索引条目可以共用同一个字节偏移**（一份压缩 KV 由它的源层发布、被它下面每个复用层逐字节读），
  所以四十条里只有四条是不同的——去重把导出从 22 MB 压到 7.1 MB，读的一侧一分钱不花。
  **engram 导出的是常量表，不是行 id**：hash 由 **tokenizer 与 `config.json`** 决定而不是 checkpoint，
  在 C++ 里复现一个 PCG64 流和一个 `tokenizers` 归一化器等于第二份要永远维护对的实现；
  `runtime/engram.h` 拿这些常量在 runtime 真正走的轨迹上算地址。
- **一个值得知道的巧合**：模型在位置 64 的 greedy 续写正好是 prompt 自己的下一个 token（3006），
  所以 **L2 的那个 decode step 与 L3 的 step 0 是同一次前向**——
  `tests/data/l2` 的七个逐级层因此可以直接用在 decode 循环的 step 0 上，
  不需要第二次导出（design §7.16.1 的逐层探针就是这么来的）。
- `oracle_l2_extra.py` 是**另一个脚本**（`oracle.py` 属于另一条 track）：它 import `oracle.py`、
  包住 `L2Capture.attach` 与 `_l2_collect`，写出 `index_k_all` / `index_score` /
  `index_weights_scaled` 与 ratio-2 compressor 的 `kv_state` / `score_state` 和它们背后的
  原始 `wkv` / `wgate` 投影。**注意两个已知的洞**：`index_score` 是**重算**的不是捕获的
  （`Indexer.forward` 只返回 indices），公式读错会在两侧同时复现；
  **ratio-2 的池化在位置 64 上没有参考输出**（`(64+1) % 2 != 0`，`Compressor.forward` 返回 None），
  现在比的是一个合成完整组对 fp64 CPU 转写——**要真正验它需要一次两步的 decode 导出**（design §12）。

## 路由 trace 与 cache 模拟器（design §9.1.1）

P1 的 Track B。完整说明、方法与结论在 [route_trace.md](route_trace.md)；这里只放命令：

```powershell
uv pip install pyarrow tokenizers pypdf     # pypdf 可选（把技术报告 PDF 收进语料）

# faithfulness 检查（64 token 完整前向，约 5 分钟）
uv run python tools/route_trace.py --model D:\models\DeepSeek-V4.1-Flash --out traces\verify --verify

# 全量 trace（27,399 token / 40 prompt，35–50 分钟，~300 GB NVMe 读，可断点续跑）
uv run python tools/route_trace.py --model D:\models\DeepSeek-V4.1-Flash `
    --tokens 20000 --out traces\mixed --checkpoint-every 4

# 策略 × 容量 × 分配
uv run python tools/cache_sim.py --trace traces\mixed `
    --capacities 20,25,30,35,abs:4787 `
    --policies lru,lfu-decay,arc,score-aware,static-pin+lru `
    --allocation both --prefetch-depths 0 --out reports\cache_sweep.json

uv run python tools\tests\test_cache_sim.py   # 12 个可独立验算的用例
```

**不要和 `bench/` 的带宽基准同机并跑**（见上一节的测量纪律）。
`traces/` 与 `reports/` 在 `.gitignore` 里；`reports/*.json` 用 utf-8 打开。

## 测试

```powershell
ctest --test-dir build --output-on-failure
```

需要真 checkpoint 的那几条默认**自动跳过**并说明原因。要跑它们：

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure

# 只跑其中一个 suite
ctest --test-dir build -R suite.gpu_moe   --output-on-failure   # §7.9 / §7.9.2 / §7.9.3
ctest --test-dir build -R suite.gpu_attn  --output-on-failure   # §7.2–§7.11 逐 stage 对 L2 + §7.4
ctest --test-dir build -R suite.gpu_layer --output-on-failure   # 整层链起来
ctest --test-dir build -R suite.decode    --output-on-failure   # 四十层，八步，对 L3
```

| suite | 它验的是什么 |
|---|---|
| `suite.integration` | 真正的 `IoEngine` + IOCP + `ExpertStore` 把 `(0,0)` 和 `(39,383)` 填进槽，比对六个 part 的校验和与 `cpu/gemv_fp4_ref` 复算的 FFN（对照 `oracle.py` 的 torch fp32） |
| `suite.gpu_moe` | §7.9 两个 kernel 的十四个变体 + **fp8 shared expert + `h` 的 fp8 量化 + 分组 dispatch**（design §7.9.2）。需要先跑 `oracle_shared.py` |
| `suite.gpu_attn` | §7.2–§7.11 的十二个 stage 逐个对 `tests/data/l2/`（design §7.15.1）。需要先跑 `oracle.py --level l2` |
| `suite.gpu_layer` | 一整层 decoder 链起来，只有 block 输入与 prefill 的 KV 是 golden；**故意只给八个槽的 cache**，所以六个 expert 每层都真的从 NVMe 取回来（design §7.15.3） |
| `suite.decode` | **四十层 + engram + head + 采样，八步，对 `tests/data/l3/`**（design §7.16.1）。需要先跑 `oracle.py --level l3` |

按标签筛选：`ctest -L needs-model` 只跑需要 checkpoint 的，`ctest -LE needs-model` 完全不跑。

## 跑一个 token：`deepmoe run`（design §7.16、[p2_decode.md](p2_decode.md)）

```powershell
deepmoe run --model DIR [--prompt-ids FILE] [--steps N]
            [--state DIR] [--teacher-force] [--per-layer]
            [--cache-gb N] [--profile FILE.jsonl] [--chunk-kb N] [--qd N]
```

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'

# 八步，12 GiB 的 routed-expert cache，逐 token 打印 design §13.1 的分解
.\build\deepmoe.exe run --model D:\models\DeepSeek-V4.1-Flash `
    --prompt-ids prompt_ids.txt --steps 8 --cache-gb 12

# 教师强制（每步喂参考自己的输入 token），每步的误差因此是独立可归因的
.\build\deepmoe.exe run --model D:\models\DeepSeek-V4.1-Flash --steps 8 --teacher-force

# 四十行一层的版本 + JSONL 的 profile
.\build\deepmoe.exe run --model D:\models\DeepSeek-V4.1-Flash --steps 8 `
    --per-layer --profile run.jsonl
```

- **`--prompt-ids` 是一个 token id 的文件**：runtime 里**还没有 tokenizer**（design §15.2 (v)）。
  `tests/data/l3/index.json` 的 `prompt_ids` 就是一份现成的。
- **`--state` 是 oracle 的 L3 导出**（默认 `tests/data/l3`），它提供的是**prefill 将来会提供的东西**：
  prompt 之后每层的 window KV，以及每步的压缩 KV 与 indexer top-k（design §7.4）。
  **`Engine::status()` 每次运行都打印 `state LOADED from …` 那一行**，
  所以一份 transcript 不可能被误当成自足的。
- **`--cache-gb 0`（默认）按机器算 cache 大小。** 注意八步是**正确性 harness 不是 cache 基准**：
  `--cache-gb 48` 的命中率（0.349）不比 12 GiB 好，因为相邻 token 只共享约一半的 routed expert，
  八个 token 长的运行到不了稳态（design §7.16.3）。命中率在 `tools/cache_sim.py` 里量。
- **`--profile` 写 design §13.1 的记录，一个 token 一行。** 其中 **`hot_bytes` 是 8.52 GB**，
  按层从 manifest 加出来，对 design §2.3 吻合到三位有效数字。
  **但 `nvme_util` 和 `nvme_gbps` 现在是错的**（前者会大于 1）：`IoEngine` 的忙碌计数器
  把每个 chunk 自己的延迟相加而不是取并集，队列深度是 8。**在它修好之前读 `effective_gbps`。**
- **`--per-layer` 打四十行**，这才是 design §13.1 真正想要的粒度——
  一层 stall 了 200 ms、或者层 1 和 14 的 engram 花了多少，在聚合里是看不见的。
  一个冷层长这样：`engram 0.00  attn 1.10  stall 32.4  moe 2.51 (gpu 1.36 host 0.67)  0/6  112.9 MB`。
- **量测纪律**：热步的数字（design §7.16.2 的 134 ms）要在空闲机上取，而且要在一次探针 pass
  **之后**——否则量到的是 NVMe。上面那份八步 transcript 里 submit 的往返在同一个二进制的
  不同次运行之间在 0.13–0.75 ms 之间动，**所以读分解的形状，不要读第三位数字**。

## Windows 虚拟内存（pagefile）：**已在开发机上做完**

> **状态：DONE（2026-09-14/15）。** C: 已设成**固定 98,304 MB = 96 GiB** 的 pagefile，
> commit 限额从 67.65 GiB 变成 **159.6 GiB**，expert cache 从 2,056 槽变成
> **5,711 槽（100 GiB = 15,360 的 37%）**，超过设计目标的 4,787。
> **新机器上这仍然是第一件要做的事**，所以步骤留在下面。

### 为什么

P-1 实测（2026-09-14，`bench/results/heap_capacity.csv`，解读见 design.md §5.2 / §9.2.2）：
默认设置下 expert slab 的真实上限**不是** BIOS VGM、也不是两个 Vulkan heap 的大小，而是 Windows 的
**commit 限额 = 物理内存 + pagefile**。路径 A（`DEVICE_LOCAL|HOST_VISIBLE`）的 slab
几乎不占物理内存（`availPhys` 在 36 GiB 的分配里只动了 0.3 GB），但**照样按 1:1 吃 commit**：

```
commit 限额 = 63.65 GiB RAM + 4 GiB 系统托管 pagefile = 67.65 GiB
→ 路径 A 拿到 36 GiB 就停（留 6 GiB 余量），路径 B 随后一个 slab 都拿不到
→ expert cache 只有 2,056 个槽
```

**调大 pagefile 不会带来换页 I/O**：VGM 支撑的页不在分页池里，**永远不会被写进 pagefile**，
这纯粹是 commit 记账。

### 怎么做

```
系统属性 → 高级 → 性能 [设置] → 高级 → 虚拟内存 [更改]
→ 取消"自动管理所有驱动器的分页文件大小"
→ 选 C: → 自定义大小 → 初始大小 = 最大值 = 98304 MB（96 GiB）
→ [设置] → [确定] → 重启
```

C: 需要相应的空闲空间（固定大小会立刻占掉那么多磁盘）。
**只需要 C: 一块**——实测 96 GiB 就足以让两条路径把物理上限跑满，**D: 上不需要 pagefile**。

### 做完之后的实测（2026-09-14/15，`bench/results/heap_capacity_idle.csv`）

```powershell
.\build\heap_capacity.exe --slab-gib 2 --min-free-gib 6 --csv bench\results\heap_capacity_idle.csv
```

| | 路径 A | 路径 B | **先 A 后 B（实际布局）** |
|---|---|---|---|
| 拿到的 slab | 37 × 2 GiB = **74 GiB** | 20 × 2 GiB = 40 GiB | A **74** + B **26** = **100 GiB** |
| 停下的原因 | `vkAllocateMemory` 返回 `-2`，**device-local heap 的 74.4 GiB 到顶** | `availPhys 6.68 GiB < slab 2 + floor 6` | 两个**物理**上限各自到顶 |
| 停下时剩余 commit | 61.8 GiB | 96.2 GiB | **35.6 GiB** |
| 满载后 GPU raw-read | 214.4 GB/s | 206.4 GB/s | **216.1 / 207.5 GB/s** |
| 折成 expert 槽 | | | **5,711 个** |

**判断有没有生效看哪一列**：`avail_commit_bytes`（应该一路都还剩几十 GiB）与
`status` 那一列的停止原因——**如果还写着 `available commit < slab + floor`，说明 pagefile 没生效。**
`--min-free-gib` 是 `availPhys` 的安全下限，不要调到 6 以下（Ctrl+C 有清理路径，但别指望它）。

> **必须在空闲机上跑。** 同一套设置下先跑的一轮（`heap_capacity_pagefile128.csv`）满载 raw-read
> 只有 **151.9 / 121.4 GB/s**，看起来像"装满之后带宽掉一半"；空闲重测是 216.1 / 207.5。
> **那是并发基准的争用，不是容量的代价。** 下一节的"测量纪律"同样适用于容量测量——
> **一个会让你改设计的负面结果，先确认机器是空的。**

## BIOS：不需要动

**不要改 UMA Frame Buffer Size（VGM）。** 实测两个 heap 是同一条 LPDDR5X：
GPU raw-read 在路径 A（216.4 GB/s）、路径 B（215.2）、纯 DEVICE_LOCAL（216.0）上没有差别，
GPU 在当前设置下已经能寻址两种内存。
design.md v0.3–v0.5 里"在两种 VGM 下各测一遍 / 需要一次重启"的条目已作废（design.md §3.3、§16）。

**容量上也不用动它。** 扩完 pagefile 之后实测：路径 A 的**前 60 GiB 正好落在 VGM 的 64 GB 里、
一个字节物理内存都不占**，第 61 GiB 起 1:1 吃可见内存。所以
**总量 = VGM + (可见物理内存 − 安全余量)，是守恒的**——调 VGM 只改变 A / B 的划分，不改变总量。

## 大页（可选，未测）

路径 B 的 MoE kernel 比路径 A 慢 12%（GART 4 KiB 页的表走查成本；raw-read 看不出来，
见 design.md §3.3）。2 MiB 大页可能抹掉它，但 `VirtualAlloc(MEM_LARGE_PAGES)` 需要
`SeLockMemoryPrivilege`：`secpol.msc` → 本地策略 → 用户权限分配 → **锁定内存页** → 加入当前账户
→ **重新登录**。代码会先尝试提权，失败时回落到 4 KiB 页并把原因写进 CSV 的 `note`
（`large pages unavailable: SeLockMemoryPrivilege is not held by this account`）。
这是 design.md §15 P6 的一个 A/B 实验，不是必需项。
