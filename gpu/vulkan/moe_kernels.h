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
    uint32_t rows_per_lane = 1;    // {1, 2, 4}: weight rows per lane
    // --- P2 knobs (docs/kernel_p2_moe.md) -------------------------------
    // How the activation reaches the FMA: 0 global (the P1 kernel), 1 LDS
    // K-tile, 2 LDS + packed fp16, 3 LDS + int8 dot4, 4 global + packed fp16,
    // 5 global + in-kernel int8 dot4 (design §7.9), 6 dispatch A consumes an
    // int8 x quantised once per token by a tiny pre-pass. Mode 6 approximates x
    // and is therefore never a default; docs/kernel_p2_moe.md §8 item 2.
    uint32_t x_mode        = 0;
    // The fp8 quantisation of h before w2 (design §7.9 v0.6): 0 off,
    // 1 reproduced inside dispatch B, 2 fp8 h written by dispatch A,
    // 3 fp8 h written by a third tiny dispatch between A and B. 3 is the one to
    // use: it is numerically identical to 2 and, unlike 2, does not force a
    // 32-row workgroup on dispatch A (docs/kernel_p2_moe.md §8 item 1).
    uint32_t h_quant       = 0;
    // Compile the FP8 E4M3 weight path so a slot flagged with kSlotFp8 can be
    // the fp8 shared expert.
    uint32_t fp8_slots     = 0;
    // Dispatch B's staging mode, when it should differ from dispatch A's.
    // design §7.9.1 open item 3: A and B have different shapes (5120 K-elements
    // against 2304, w1+w3 against w2) and P1 already found their best
    // RowsPerLane differ, so they get separate pipelines from one MoeSpec.
    // ~0u means "whatever x_mode says".
    static constexpr uint32_t kFollowA = ~0u;
    uint32_t x_mode_b      = kFollowA;
    // Mode 6 is a property of x, and dispatch B's activation is h, so a
    // following B falls back to the plain global path rather than to 6.
    uint32_t b_mode() const {
        if (x_mode_b != kFollowA) return x_mode_b;
        return x_mode == 6 ? 0u : x_mode;
    }
    std::string name() const;
};

// Set in MoeRunner::ids()[slot] to say "this slot's weights are FP8 E4M3 with a
// [rows/32][K/32] scale plane" (the shared expert of design §7.9). The low bits
// stay the expert index into the pointer table.
inline constexpr uint32_t kSlotFp8 = 0x80000000u;

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
    // How many of `slots` hold an fp8 shared expert. Byte accounting only: an
    // fp8 expert reads 2x the weight bytes of an FP4 one, so the effective
    // GB/s of a mixed dispatch needs to know the mix.
    uint32_t fp8_slot_count    = 0;
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
    double   record_seconds = 0.0; // CPU cost of recording the command buffer (design §3.4)
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

    // design §7.9 partial dispatch: when set, dispatch B adds its slot sum into
    // y instead of overwriting it, so a layer can be computed as several
    // dispatches over disjoint subsets of the slot list.
    //
    // Two schedules are possible and they do NOT cost the same
    // (docs/kernel_p2_moe.md §6.3):
    //
    //   pairs     run(Both) over slots [0, c), then set_accumulate(true) and
    //             run(Both) over [c, n). Every group produces its own y
    //             contribution, so nothing has to be held back -- but dispatch
    //             B pays a second time for 640 workgroups' worth of ramp,
    //             cross-lane reduction and y read-modify-write, and the answer
    //             is 1-2 ULP from the one-shot one because the reduction is
    //             re-associated.
    //   deferred  run(GateUpOnly) over [0, c), run(GateUpOnly) over [c, n),
    //             then one run(DownOnly) over the whole list with accumulate
    //             off. Dispatch A is ~2/3 of a layer and its work is exactly
    //             proportional to the slots it is handed, so this is what
    //             overlaps with the I/O wait; dispatch B needs every slot's h
    //             anyway, so deferring it loses nothing. Measurably cheaper and
    //             bit-identical to the one-shot run.
    //
    // Prefer `deferred` unless a partial y is needed before the last expert
    // lands, which decode never needs.
    void set_accumulate(bool on) { accumulate_ = on; }
    bool accumulate() const { return accumulate_; }

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
    bool             accumulate_ = false;

    // The optional third and pre-dispatches: `hquant_` quantises h between A
    // and B when spec.h_quant == 3, `xquant_` quantises x before A when
    // spec.x_mode == 6. Both are tiny and both are absent otherwise.
    Pipeline       gateup_, down_, hquant_, xquant_;
    DescriptorPool descriptors_;
    CommandPool    pool_;
    CommandBuffer  cmd_{};
    QueryPool      queries_;

    GpuBuffer table_{}, ids_{}, list_{}, routew_{}, x_{}, h_{}, y_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet set_a_ = VK_NULL_HANDLE;
    VkDescriptorSet set_b_ = VK_NULL_HANDLE;
    VkDescriptorSet set_hq_ = VK_NULL_HANDLE;
    VkDescriptorSet set_xq_ = VK_NULL_HANDLE;
#endif
};

// Where the build put the .spv files: DEEPMOE_SHADER_DIR, overridable with the
// environment variable of the same name so a moved build tree still runs.
std::string default_shader_dir();

}  // namespace deepmoe::gpu
