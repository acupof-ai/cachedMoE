# deepMoE

Windows Strix Halo（Ryzen AI Max+ 395 / Radeon 8060S / 128 GB / NVMe）专用的超大 MoE 本地推理 Runtime。目标模型：**DeepSeek-V4.1-Flash**（552B backbone + 196B Engram，decode 激活 16B，510 GB 权重）。

- 设计方案：[docs/design.md](docs/design.md)
- 构建与环境：[docs/build.md](docs/build.md)

核心思路：权重原生 FP4/FP8 不再量化；常驻部分 pin 在统一内存，routed expert 在 NVMe 与内存之间由数据驱动的 Planner 流式调度；Vulkan Compute（Slang）每 token 一个 command buffer；DSpark 投机解码摊薄带宽；所有优化以 oracle 与可分解的每 token 时间线为准。

```powershell
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\envcheck.exe
```
