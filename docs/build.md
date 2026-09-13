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

## 配置与构建

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
```

```powershell
cmake --build build
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

`tools/envcheck/` 里有两份文件：`vkinfo.cpp`（C++20 + Vulkan + AVX-512 VNNI）与 `gemv.slang`（FP4 解码 GEMV 骨架，用到 `WaveActiveSum` 与 int8 存储）。手工验证：

```powershell
zig c++ -std=c++20 -O2 -march=znver5 -Wno-nullability-completeness -I"$env:VULKAN_SDK\Include" tools\envcheck\vkinfo.cpp -L"$env:VULKAN_SDK\Lib" -lvulkan-1 -o build\vkinfo.exe; .\build\vkinfo.exe
```

```powershell
& "$env:VULKAN_SDK\Bin\slangc.exe" tools\envcheck\gemv.slang -target spirv -profile spirv_1_6 -entry main -O2 -o build\gemv.spv; & "$env:VULKAN_SDK\Bin\spirv-val.exe" --target-env vulkan1.3 build\gemv.spv
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

下载完成后：

```powershell
uv run python tools/repack.py --src D:\models\DeepSeek-V4.1-Flash --dst D:\models\deepmoe-v41 --verify
```

（`repack.py` 为 P0 产物；`--verify` 用 `model.safetensors.index.json` 与各分片 header 校验大小并写出 `manifest.json`。）

## Python 环境

```powershell
uv venv; uv pip install numpy safetensors pyarrow torch --index-url https://download.pytorch.org/whl/cpu
```

torch 只用 CPU 版，供 `tools/oracle.py`。

## BIOS 提示

`P-1` 需要在两种 VGM（GPU 专用显存）设置下测带宽：当前 64 GB，以及最小值。VGM 在 BIOS → Advanced → GFX Configuration → UMA Frame Buffer Size。改动后 Windows 可见内存与 `vulkaninfo` 的 heap 大小都会变化，把两组数字都记进 `bench/results/`。
