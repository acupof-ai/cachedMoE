// Host side of the two FP4 MoE dispatches of design §7.9.
//
// MoeRunner owns everything the two kernels need except the expert weights
// themselves: those live in ExpertStore slabs and are reached through the
// GPU-side pointer table (design §5.3), which is the only "weight" buffer bound
// here. Upload the ExpertStore's table, say which slots to compute, set x, run.
//
// The A/B knobs of design §7.1 (M, LanesPerRow, subgroup size, FP4 decode
// variant, fp16 vs fp32 h) are specialisation constants, so sweeping them costs
// a pipeline creation and no recompilation. bench/kernel_bench sweeps them;
// tests/test_gpu_moe.cpp checks each against the oracle.
//
// Ownership/threading: one MoeRunner is created, recorded and submitted from a
// single thread. It owns its pipelines, descriptor pool, command pool, query
// pool and every small buffer; the expert slabs belong to the ExpertStore.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/pipeline.h"
#include "model/layout.h"

namespace deepmoe::gpu {

// The specialisation sweep of design §7.1.
struct MoeSpec {
    uint32_t m             = 1;    // accumulators per lane: 1 decode, 6 verify
    uint32_t lanes_per_row = 32;   // {16, 32, 64}
    uint32_t subgroup_size = 0;    // 0 = driver default, else 32 or 64
    uint32_t decode_mode   = 1;    // 0 = const table, 1 = arithmetic, 2 = select tree
    uint32_t h_precision   = 0;    // 0 = fp16 h, 1 = fp32 h
    std::string name() const;
};

struct MoeDims {
    uint32_t layer             = 0;
    uint32_t experts_per_layer = layout::kRoutedExperts;
    uint32_t slots             = 7;   // 6 routed + shared (design §7.9)
    uint32_t hidden            = layout::kHiddenSize;         // 5120
    uint32_t inter             = layout::kMoeIntermediate;    // 2304
    uint32_t table_layers      = layout::kTotalLogicalLayers;
    // Iteration `i` of a repeated benchmark uses logical layer
    // `layer + i % layer_cycle`. With one layer the same 7 experts are re-read
    // every iteration and a 132 MB working set partly lives in the 32 MB MALL,
    // which flatters the measured GB/s; cycling over several layers restores
    // the streaming behaviour decode actually has (design §2.3).
    uint32_t layer_cycle       = 1;
    float    swiglu_limit      = layout::kSwigluLimit;
};

// Which of the two dispatches to time. Running them separately is how
// bench/kernel_bench gets a per-dispatch effective GB/s; Both is what the
// decode loop actually issues.
enum class MoePhase : uint8_t { Both = 0, GateUpOnly, DownOnly };

struct MoeTiming {
    double   seconds_a = 0.0;      // dispatch A (gate/up) GPU seconds per iteration
    double   seconds_b = 0.0;      // dispatch B (down)
    double   seconds_total = 0.0;  // both, including the barrier between them
    double   wall_seconds = 0.0;   // submit to queue-idle, for the launch-overhead number
    bool     gpu_timed = false;
    uint32_t iterations = 0;
};

class MoeRunner {
public:
    MoeRunner() = default;
    ~MoeRunner() { destroy(); }

    MoeRunner(const MoeRunner&) = delete;
    MoeRunner& operator=(const MoeRunner&) = delete;

    Result<void> create(Device& device, MemoryAllocator& alloc, const std::string& shader_dir,
                        const MoeSpec& spec, const MoeDims& dims);
    void destroy();

    const MoeSpec& spec() const { return spec_; }
    const MoeDims& dims() const { return dims_; }

    // --- inputs (host-visible, written directly) --------------------------

    // [table_layers][experts_per_layer][6] device addresses, exactly the layout
    // of store::ExpertStore::pointer_table().
    uint64_t* pointer_table();
    size_t    pointer_table_entries() const;

    uint32_t* ids();          // [slots]: which expert sits in each slot
    uint32_t* slot_list();    // [slots]: the indirection list of design §7.9
    float*    route_weights();// [m][slots]
    uint16_t* x_fp16();       // [m][hidden] activations (design §6)
    float*    y();            // [m][hidden] output of dispatch B
    void*     h();            // [m][slots][inter], fp16 or fp32 per spec

    void set_list_count(uint32_t n);
    uint32_t list_count() const { return list_count_; }

    // --- execution ---------------------------------------------------------

    // Records `iterations` back-to-back A+B pairs into one command buffer, with
    // timestamps around each phase, and submits it. `iterations` > 1 amortises
    // submit cost; launch overhead is measured by comparing 1 and N.
    Result<MoeTiming> run(uint32_t iterations = 1, MoePhase phase = MoePhase::Both);

    // Bytes of weights + scales one A+B pair touches, the numerator of the
    // effective GB/s of design §7.1 rule 2.
    uint64_t bytes_per_iteration() const;
    uint64_t bytes_dispatch_a() const;
    uint64_t bytes_dispatch_b() const;

private:
    Result<void> record(uint32_t iterations, MoePhase phase);

    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    MoeSpec          spec_{};
    MoeDims          dims_{};
    uint32_t         list_count_ = 0;
    uint32_t         recorded_   = 0;

    Pipeline       gateup_, down_;
    DescriptorPool descriptors_;
    CommandPool    pool_;
    CommandBuffer  cmd_{};
    QueryPool      queries_;

    GpuBuffer table_{}, ids_{}, list_{}, routew_{}, x_{}, h_{}, y_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet set_a_ = VK_NULL_HANDLE;
    VkDescriptorSet set_b_ = VK_NULL_HANDLE;
#endif
};

// Where the build put the .spv files: DEEPMOE_SHADER_DIR, overridable with the
// environment variable of the same name so a moved build tree still runs.
std::string default_shader_dir();

}  // namespace deepmoe::gpu
