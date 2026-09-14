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
build/kernel_bench.exe         design §7.9.1 MoE kernel sweep + dispatch 开销 + 路径 A/B
build/heap_capacity.exe        design §9.2.2 两个 heap 的实际可分配上限（commit 限额）
build/envcheck.exe             环境自检
build/tests/deepmoe_tests.exe  单元测试
build/shaders/*.spv            §7.14 的 12 个 kernel，每个都过 spirv-val
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

# MoE kernel sweep（60 变体）+ dispatch 开销 → design §7.9.1、§3.4
.\build\kernel_bench.exe --csv bench\results\kernel_p1.csv `
    --iters 48 --layer-cycle 8 --repeats 4 --sweeps 2
.\build\kernel_bench.exe --path b --quick        # 路径 B 对照

# 容量上限（commit 限额）→ design §9.2.2。调大 pagefile 之后要重跑
.\build\heap_capacity.exe --slab-gib 2 --min-free-gib 6 --csv bench\results\heap_capacity.csv
```

`nvme_bench` 默认在 `%TEMP%` 建一个 2 GB 测试文件，跑完删除（`--keep` 保留，`--file`
指定已有文件）。实测结果写回 design.md §9.2.1。

**测量纪律（否则数字没有意义）**：LPDDR 是 CPU 和 GPU 共用的，**任何吃内存带宽的进程都会
污染 GPU 的数字**——实测同一个 kernel 变体在机器忙 / 闲两种状态下相差最多 **50%**
（164.6 vs 94.5 GB/s）。所以：

- **不要同时编译**（一次 `cmake --build` 撞上 `bw_matrix`，CPU 峰值从 100.9 掉到 97.7，
  并发那一栏直接作废），也不要同时跑 `tools/route_trace.py`。
- `kernel_bench` 的工作集要用 `--layer-cycle 8`（1053 MB）跨过 32 MB 的 MALL；
  只用 1 层（132 MB）会把带宽**虚高约 25%**。
- 判据是 raw-read 上限在 sweep 前后一致。完整的方法与两个踩过的坑见
  [kernel_p1.md](kernel_p1.md) §1。

带 validation layer 跑一遍（两个 bench 与两个 GPU 测试当前都是干净的）：

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
```

- `--level l0` 导出 FP4 E2M1 / FP8 E4M3 / UE8M0 三张解码表到 `tests/data/l0_dequant.bin`（2,132 B），
  `tests/test_dequant.cpp` 逐位比对。
- `--level l1` 默认跑 `(layer 0, expert 0)` 与 `(layer 39, expert 383)`（用 `--expert L:E` 指定，可重复）：
  六个 tensor 各读两遍（一遍走 manifest 的 run/skew，一遍走 `safetensors` 库）要求字节一致，
  再在 torch fp32 里算 expert FFN，把 `x`/`y`/校验和写进 `tests/data/l1_layer{L}_expert{E}.bin`。
  加 `--report out.json` 可以把数字存下来。

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

需要真 checkpoint 的那一条（`suite.integration`）默认**自动跳过**并说明原因。要跑它：

```powershell
$env:DEEPMOE_MODEL_DIR='D:\models\DeepSeek-V4.1-Flash'; ctest --test-dir build --output-on-failure
```

它用真正的 `IoEngine` + IOCP + `ExpertStore` 把 `(0,0)` 和 `(39,383)` 填进槽，
比对六个 part 的校验和与 `cpu/gemv_fp4_ref` 复算的 FFN 输出（对照 `oracle.py` 的 torch fp32 结果）。
按标签筛选：`ctest -L needs-model` 只跑它，`ctest -LE needs-model` 完全不跑。

## Windows 虚拟内存（pagefile）：必须先调大

**这是这台机器上唯一一个必须手工做的系统设置，不做的话 expert cache 只有设计容量的 43%。**

P-1 实测（2026-09-14，`bench/results/heap_capacity.csv`，解读见 design.md §5.2 / §9.2.2）：
expert slab 的真实上限**不是** BIOS VGM、也不是两个 Vulkan heap 的大小，而是 Windows 的
**commit 限额 = 物理内存 + pagefile**。路径 A（`DEVICE_LOCAL|HOST_VISIBLE`）的 slab
几乎不占物理内存（`availPhys` 在 36 GiB 的分配里只动了 0.3 GB），但**照样按 1:1 吃 commit**。
本机默认：

```
commit 限额 = 63.65 GiB RAM + 4 GiB 系统托管 pagefile = 67.65 GiB
→ 路径 A 拿到 36 GiB 就停（留 6 GiB 余量），路径 B 随后一个 slab 都拿不到
→ expert cache 只有 2,056 个槽（设计要 4,787）
```

**调大 pagefile 不会带来换页 I/O**：VGM 支撑的页不在分页池里，**永远不会被写进 pagefile**，
这纯粹是 commit 记账。设置路径：

```
系统属性 → 高级 → 性能 [设置] → 高级 → 虚拟内存 [更改]
→ 取消"自动管理所有驱动器的分页文件大小"
→ 选 C: → 自定义大小 → 初始大小 = 最大值 = 98304 MB（96 GB；128 GB 更稳妥）
→ [设置] → [确定] → 重启
```

C: 需要相应的空闲空间（96 GB 固定大小会立刻占掉 96 GB 磁盘）。重启后重测：

```powershell
.\build\heap_capacity.exe --slab-gib 2 --min-free-gib 6 --csv bench\results\heap_capacity.csv
```

期望看到 `path_a` + `path_b` 合计 ≥ 90 GiB（4,787 个 expert 槽）。CSV 的
`avail_commit_bytes` 列是判断有没有生效的那一列；`--min-free-gib` 是 `availPhys` 的安全下限，
不要调到 6 以下（Ctrl+C 有清理路径，但别指望它）。

## BIOS：不需要动

**不要改 UMA Frame Buffer Size（VGM）。** 实测两个 heap 是同一条 LPDDR5X：
GPU raw-read 在路径 A（216.4 GB/s）、路径 B（215.2）、纯 DEVICE_LOCAL（216.0）上没有差别，
GPU 在当前设置下已经能寻址两种内存，而容量受 commit 限额而不是 VGM 约束（见上一节）。
design.md v0.3–v0.5 里"在两种 VGM 下各测一遍 / 需要一次重启"的条目已作废（design.md §3.3、§16）。

## 大页（可选，未测）

路径 B 的 MoE kernel 比路径 A 慢 12%（GART 4 KiB 页的表走查成本；raw-read 看不出来，
见 design.md §3.3）。2 MiB 大页可能抹掉它，但 `VirtualAlloc(MEM_LARGE_PAGES)` 需要
`SeLockMemoryPrivilege`：`secpol.msc` → 本地策略 → 用户权限分配 → **锁定内存页** → 加入当前账户
→ **重新登录**。代码会先尝试提权，失败时回落到 4 KiB 页并把原因写进 CSV 的 `note`
（`large pages unavailable: SeLockMemoryPrivilege is not held by this account`）。
这是 design.md §15 P6 的一个 A/B 实验，不是必需项。
