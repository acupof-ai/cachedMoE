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
build/nvme_bench.exe           design §9.1 Q6/Q7 微基准
build/bw_matrix.exe            design §9.2 带宽矩阵
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
.\build\nvme_bench.exe --help
.\build\nvme_bench.exe --reads 48 --csv bench\results\nvme_q6_q7.csv
.\build\bw_matrix.exe --size-gb 4
```

`nvme_bench` 默认在 `%TEMP%` 建一个 2 GB 测试文件，跑完删除（`--keep` 保留，`--file`
指定已有文件）。实测结果写回 design.md §9.2.1。

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

## BIOS 提示

`P-1` 需要在两种 VGM（GPU 专用显存）设置下测带宽：当前 64 GB，以及最小值。VGM 在 BIOS → Advanced → GFX Configuration → UMA Frame Buffer Size。改动后 Windows 可见内存与 `vulkaninfo` 的 heap 大小都会变化，把两组数字都记进 `bench/results/`。
