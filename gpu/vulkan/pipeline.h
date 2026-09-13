// Compute pipeline: load a .spv produced by add_slang_shader(), specialise it,
// create the layout (design §7.1, §7.14).
//
// Every GEMV kernel in design §7 is one Slang generic
// `Gemv<WeightFmt, M, LanesPerRow>`; slangc emits one .spv per entry point and
// the remaining knobs (M, lanes_per_row, subgroup size) arrive as
// specialisation constants, so the A/B sweeps of §7.1 need no recompilation
// between runs.
//
// Ownership/threading: a Pipeline owns its VkShaderModule, VkPipelineLayout,
// VkDescriptorSetLayout and VkPipeline. Created once at startup on the submit
// thread; VkPipeline is immutable and safe to bind from any recording thread.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/device.h"

namespace deepmoe::gpu {

// design §7.1 knobs. `subgroup_size` is applied through
// VK_EXT_subgroup_size_control, the rest through specialisation constants.
struct PipelineSpec {
    uint32_t m             = 1;    // batch: 1 decode, 5-6 speculative verify, >=16 prefill
    uint32_t lanes_per_row = 64;   // A/B over {16, 32, 64}
    uint32_t subgroup_size = 0;    // 0 = driver default, else 32 or 64
    uint32_t rows_per_wg   = 8;    // design §7.9 dispatch B
    std::vector<uint32_t> extra;   // kernel-specific constants, ids continue from 4
};

// How many storage buffers a kernel binds. The expert weights themselves are
// NOT descriptors -- they are reached through buffer_device_address from the
// pointer table (design §5.3), which is why this number stays small.
struct PipelineLayoutSpec {
    uint32_t storage_buffers   = 0;
    uint32_t push_constant_size = 0;
};

class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    // Reads `spv_path` (build/shaders/<name>.spv) and creates the pipeline.
    Result<void> create(Device& device, const std::string& spv_path,
                        const PipelineLayoutSpec& layout, const PipelineSpec& spec = {});
    void destroy();
    bool valid() const { return valid_; }

    const std::string& name() const { return name_; }

#if defined(DEEPMOE_ENABLE_VULKAN)
    VkPipeline            handle() const { return pipeline_; }
    VkPipelineLayout      layout() const { return layout_; }
    VkDescriptorSetLayout set_layout() const { return set_layout_; }
#endif

private:
    Device*     device_ = nullptr;
    std::string name_;
    bool        valid_ = false;
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkShaderModule        module_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      layout_     = VK_NULL_HANDLE;
    VkPipeline            pipeline_   = VK_NULL_HANDLE;
#endif
};

// Reads a SPIR-V file into words, checking the 0x07230203 magic. Implemented
// regardless of DEEPMOE_ENABLE_VULKAN so the build can validate shader output
// without a GPU.
Result<std::vector<uint32_t>> load_spirv(const std::string& path);

// The dispatch list of design §7.14, in order. The shader files in
// gpu/shaders/ are named after these.
inline constexpr const char* kDecodeKernels[] = {
    "mega_mhc",      // 1  mHC mixes + hc_post + norm              (§7.2, §7.7)
    "wq_a",          // 2  fp8 [1280 x 5120]                        (§7.3)
    "wq_b",          // 3  fp8 [32768 x 1280] + q_norm + RoPE       (§7.3)
    "wkv",           // 4  fp8 [512 x 5120] + kv_norm + RoPE + cache(§7.4)
    "sparse_attn",   // 5  MQA over window 128 + top-512            (§7.5)
    "wo_a",          // 6  grouped fp8 8 x [1024 x 4096]            (§7.6)
    "wo_b",          // 7  fp8 [5120 x 8192]                        (§7.6)
    "gate",          // 9  bf16 [384 x 5120] + top-6                (§7.8)
    "moe_gateup",    // 10 fused FP4 gate/up + SwiGLU, 7 experts    (§7.9)
    "moe_down",      // 11 FP4 down + 7-way reduction               (§7.9)
    "head",          // -- bf16 [129280 x 5120]                     (§7.11)
    "moe_gemv_fp4",  // the standalone FP4 GEMV template / benchmark
};

}  // namespace deepmoe::gpu
