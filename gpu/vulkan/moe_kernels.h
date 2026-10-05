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
#include "gpu/shaders/dspark_plan_layout.h"

namespace deepmoe::gpu {
static_assert(layout::kMoeBatchColumns == DM_ROUTE_MAX_COLUMNS);
static_assert(layout::kGateRecordCount == DM_ROUTE_GATE_RECORDS);
static_assert(layout::kExpertAddressWords == DM_ROUTE_ADDRESS_WORDS);
static_assert(layout::kMoeIntermediate / 32 == DM_ROUTE_INTER_BLOCKS);
static_assert(layout::kDsparkBlockSize == DM_DS_DRAFT_COLUMNS);
static_assert(layout::kDsparkTopK == DM_DS_TOPK);
static_assert(layout::kDsparkExperts == DM_DS_EXPERTS);

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
    // Dispatch B's own shape (0 = follow A). On RADV the decode shape L32 R1
    // reads w2 at ~152 GB/s against ~204 for w1+w3, and L16 R2 reads w2 at
    // ~185 -- but costs A as much as it saves B when both have to agree.
    uint32_t lanes_b       = 0;
    uint32_t rows_b        = 0;
    uint32_t b_lanes() const { return lanes_b ? lanes_b : lanes_per_row; }
    uint32_t b_rows()  const { return rows_b  ? rows_b  : rows_per_lane; }
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
    // The h allocation as words (fp16 h, or the fp8 value + scale planes under
    // h_quant 2/3; moe_common.slang hq_value_words). For tests.
    const uint32_t* h_words() const { return static_cast<const uint32_t*>(h_.host_ptr); }
    void*     h();            // [m][slots][inter], fp16 or fp32 per spec

    void set_list_count(uint32_t n);
    uint32_t list_count() const { return list_count_; }

    // Track T / DSpark: activation columns this dispatch computes (the shaders'
    // `pc.m`). The kernels are specialised on `M`, so without this every
    // dispatch paid for all M columns whatever the caller's live count --
    // docs/p4_mgt1.md §4 measured a verify batch's dispatching at 1.79-1.95 ms a
    // column at M=6 against 1.786 at M=1, i.e. the shape, not the work.
    // 1 is a decode token; the batch size is a verify batch. Clamped to [1, M].
    void set_live_columns(uint32_t n) {
        live_columns_ = n < 1 ? 1u : (n > spec_.m ? spec_.m : n);
    }
    uint32_t live_columns() const { return live_columns_; }

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

    // ADDITIVE (Track I, docs/p2_decode.md §9): the same dispatch sequence as
    // one iteration of `run` -- x-quant if spec.x_mode == 6, dispatch A,
    // h-quant if spec.h_quant == 3, dispatch B -- recorded into a command
    // buffer the CALLER has begun and will submit. Barriers between the
    // dispatches, none before the first (the caller's last dispatch decides
    // that) and one after the last, so whatever the caller records next can
    // read `y`. No timestamps and no submit. This is what lets a decode layer
    // put its MoE and the next layer's attention into ONE submit instead of
    // three.
    Result<void> record_into(CommandBuffer& cmd, MoePhase phase = MoePhase::Both);

    // ADDITIVE (Track R1, docs/p4_hitrate.md §4): design §7.9's "compute the
    // experts that arrived first" across SUBMITS. The slot list is read when a
    // buffer executes, so dispatch A over a subset and dispatch B over the whole
    // list cannot share one list buffer inside one command buffer. This second
    // list is bound only to dispatch A and the h quantisation: `record_gateup_alt`
    // records x-quant (x_mode 6), A and h-quant over `slot_list_alt()[0, count)`,
    // leaving `slot_list()` and `list_count()` to dispatch B. Deferred split-A
    // schedule of kernel_p2_moe.md §11.3, so y is bit-identical to one shot.
    uint32_t*    slot_list_alt();
    Result<void> record_gateup_alt(CommandBuffer& cmd, uint32_t count);

    // Track SE: the shared expert ahead of routing. `record_shared_early`
    // records moe_xact (x's act_quant from the fp32 ffn_norm output at
    // `x_src_address` into column 0 of x), then dispatch A and the h
    // quantisation over the LAST slot only, through a third one-entry slot list
    // of its own -- so it can be in flight while the host rewrites the other
    // two lists for the routed slots, and dispatch B over the whole list later
    // is bit-identical to one shot (the Track R1 split). Needs spec.m's decode
    // twin or M = 1, x_mode 0..5.
    Result<void> record_shared_early(CommandBuffer& cmd, uint64_t x_src_address);
    // Which page of the pointer table the dispatches recorded from now on
    // index (the `layer` push constant, `(layer * experts_per_layer + expert)
    // * 6`). Track SE alternates two pages by layer parity so the shared row of
    // layer L can be written while layer L-1's MoE is still queued.
    void         set_table_layer(uint32_t layer) { dims_.layer = layer; }

    // Device address of `y`, so the caller's next dispatch can read the MoE
    // output through buffer-device-address instead of the host copying it.
    uint64_t y_address() const { return y_.dev_addr; }
    Result<void> record_gpu_copy(CommandBuffer&,uint64_t src,uint64_t dst,uint32_t words,
                                 bool hc_mean=false,uint32_t hidden=0);
    Result<void> init_gpu_route(uint32_t layers,const std::string& dir);
    Result<void> upload_snapshot(std::span<const uint64_t>);
    // snapshot is [layers][experts][6], each layer includes its shared row.
    Result<void> record_gpu_route(CommandBuffer&,uint32_t layer,uint32_t m,
                                  uint32_t topk,uint64_t ids,uint64_t weights,uint64_t x,
                                  uint64_t saved,uint32_t* trace_unused=nullptr);

    // Bytes of weights + scales one A+B pair touches, the numerator of the
    // effective GB/s of design §7.1 rule 2.
    uint64_t bytes_per_iteration() const;
    uint64_t bytes_dispatch_a() const;
    uint64_t bytes_dispatch_b() const;

private:
    Pipeline gpu_copy_,gpu_mean_,gpu_route_,gpu_up_,gpu_down_,gpu_hq_;
    GpuBuffer gpu_snapshot_,gpu_args_,gpu_indirect_;
    DescriptorPool gpu_descriptors_;uint32_t gpu_layers_=0;uint64_t gpu_table_stride_=0,gpu_arg_stride_=256;
#if defined(DEEPMOE_ENABLE_VULKAN)
    std::vector<VkDescriptorSet> gpu_route_sets_,gpu_up_sets_,gpu_down_sets_;
#endif

    Result<void> record(uint32_t iterations, MoePhase phase);

    Device*          device_ = nullptr;
    MemoryAllocator* alloc_  = nullptr;
    MoeSpec          spec_{};
    MoeDims          dims_{};
    uint32_t         list_count_ = 0;
    uint32_t         live_columns_ = 1;
    uint32_t         recorded_   = 0;
    bool             accumulate_ = false;

    // The optional third and pre-dispatches: `hquant_` quantises h between A
    // and B when spec.h_quant == 3, `xquant_` quantises x before A when
    // spec.x_mode == 6. Both are tiny and both are absent otherwise.
    Pipeline       gateup_, down_, hquant_, xquant_, xact_;
    // Track K1a: the same three pipelines specialised on M == 1 (and StaticM
    // == 1), created only when the runner itself is specialised on M > 1 and
    // used whenever `live_columns_ == 1`. The decode path of the engine runs on
    // a runner created with M = kMoeBatchMax so that one MoeRunner can also
    // serve a verify batch (runtime/moe_bridge.cpp), which meant every decode
    // token paid the M = 6 shape: six accumulators a lane a row, an M = 6 LDS
    // tile, and -- since fb53514 -- a push-constant trip count the shader
    // compiler cannot fold. These are empty when spec.m == 1 (the M = 1 runner
    // is already the specialised one) or when x_mode == 6, whose int8 x plane
    // offsets are M-dependent and written by a separate pipeline.
    Pipeline       gateup_m1_, down_m1_, hquant_m1_;
    // Whether this dispatch should take them: every M-dependent buffer offset
    // (the fp8 h plane of moe_common.slang hq_value_words, the LDS tile) is
    // consistent only if A, the h quantiser and B agree, so it is one decision
    // for the whole chain.
    bool           use_m1() const { return live_columns_ == 1 && gateup_m1_.valid(); }
    uint32_t       effective_m() const { return use_m1() ? 1u : spec_.m; }
    DescriptorPool descriptors_;
    CommandPool    pool_;
    CommandBuffer  cmd_{};
    QueryPool      queries_;

    GpuBuffer table_{}, ids_{}, list_{}, routew_{}, x_{}, h_{}, y_{};
    GpuBuffer list_alt_{}, list_sh_{};
#if defined(DEEPMOE_ENABLE_VULKAN)
    VkDescriptorSet set_a_sh_ = VK_NULL_HANDLE;
    VkDescriptorSet set_hq_sh_ = VK_NULL_HANDLE;
    VkDescriptorSet set_xact_ = VK_NULL_HANDLE;
    VkDescriptorSet set_a_alt_ = VK_NULL_HANDLE;
    VkDescriptorSet set_hq_alt_ = VK_NULL_HANDLE;
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
