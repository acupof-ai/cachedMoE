// The one file that knows about the design §7.9 MoE kernels.
//
// runtime/decode_layer.h asks for `MoeCall{layer, ids, weights, x, y}`. This
// turns that into MoeRunner, MoeSpec, the expert pointer table, the fp8
// shared-expert flag and the routed-expert fetch. The MoE kernels are owned by
// another track and are being changed while this is written, so the coupling
// is deliberately one file wide: a change to their push constants or their
// specialisation constants is a change here and nowhere else.
//
// One runner, seven slots
// -----------------------
// Six routed FP4 experts and the FP8 shared expert go through the SAME two
// dispatches, which is what design §7.9 describes and what
// docs/kernel_p2_moe.md §4 measured (`Fp8Slots = 1`, the shared slot's id
// carrying kSlotFp8). The routed experts live in ExpertStore slots and the
// shared one in the pinned set, which has no expert id, so the runner's table
// is one expert wider than the model's: index 384 of row 0 is rewritten with
// the shared expert of whatever layer is being run. Dispatch B then sums all
// seven into one fp32 `y`, which is `MoE.forward`'s `y += shared(x)` without
// the host doing the add.
//
// P2 step 2 used two runners, which cost a second submit, a second set of
// timestamp reads and a host-side fp32 add per layer (docs/p2_decode.md §5.2).
//
// The specialisation
// ------------------
// docs/kernel_p2_moe.md §12's decode champion: `L32 R1, sg32, dec0, h=fp16,
// XMode=0, HQuant=3`. HQuant 3 is the h fp8 round trip of design §7.9 v0.6 as
// its own tiny dispatch between A and B; the previous default here, HQuant 1,
// reproduces it inside dispatch B at +45% on this shape.
//
// The activation round trip
// ------------------------
// `linear()` quantises the input of an fp4 or fp8 weight to fp8 E4M3, block
// 32, UE8M0 scale before the GEMM (see gpu/shaders/attn_common.slang). The MoE
// kernels take x as fp16, so the round trip happens here, on the host, once
// per layer. That is not an approximation: `act_quant` is a property of the
// activation alone, so quantising once instead of once per expert produces the
// identical bytes seven times less often.
//
// Two ways to run it
// ------------------
//   `run`           stage + submit + wait + copy y into `call.y`. The
//                   MoeBridge interface; what a layer-at-a-time validator uses.
//   `stage`+`record` the host half, then the dispatches recorded into a command
//                   buffer the caller owns. `y` stays on the GPU at
//                   `y_address()`, which is where the next layer's hc_post
//                   reads it. What the token loop uses.
//
// Ownership/threading: borrows the store, the planner and the pinned set; owns
// its runner. Single-threaded, on the GPU submit thread.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/moe_kernels.h"
#include "model/v41_config.h"
#include "runtime/decode_layer.h"
#include "store/expert_store.h"
#include "store/pinned.h"
#include "store/planner.h"

namespace deepmoe::runtime {

// The staging activation conversion, exposed so a validator can reproduce the
// bytes the kernel will read (moe_bridge.cpp's act_quant_to_fp16).
void debug_act_quant_to_fp16(const float* x, uint16_t* out, float* scratch, uint32_t n);

struct MoeBridgeConfig {
    // Passed straight to MoeSpec: docs/kernel_p2_moe.md §12's M = 1 champion.
    uint32_t lanes_per_row = 32;
    uint32_t subgroup_size = 32;
    uint32_t decode_mode   = 0;   // 0 = the constant table
    uint32_t rows_per_lane = 1;
    uint32_t x_mode        = 0;   // global fp16 x; 6 approximates x and is never a default
    // 3 = the fp8 round trip on h as a third dispatch between A and B, which
    // is numerically identical to 2 and to the reference (kernel_p2_moe §9.5).
    uint32_t h_quant       = 3;
};

class GpuMoeBridge final : public MoeBridge {
public:
    GpuMoeBridge() = default;
    ~GpuMoeBridge() override { destroy(); }

    Result<void> create(gpu::Device& device, gpu::MemoryAllocator& alloc,
                        const std::string& shader_dir, store::ExpertStore& store,
                        store::Planner& planner, const store::PinnedStore& pinned,
                        const TextConfig& cfg, const MoeBridgeConfig& bc = {});
    void destroy();

    const char* name() const override { return "gpu(moe_gateup+moe_hquant+moe_down, 6 fp4 + 1 fp8)"; }
    Result<void> run(const MoeCall& call) override;

    // The host half of one layer: x out of GPU-visible memory and through
    // act_quant into the runner's fp16 input, the seven table rows, the ids,
    // the slot list and the routing weights. `call.y` is ignored.
    Result<void> stage(const MoeCall& call);
    // The dispatches, into a command buffer the caller has begun. Ends with a
    // barrier, so the caller's next dispatch may read `y_address()`.
    Result<void> record(gpu::CommandBuffer& cmd);

    // --- Track R1: "compute the experts that arrived first" (design §7.9,
    // docs/kernel_p2_moe.md §11.3, docs/p4_hitrate.md §4) ------------------
    //
    //   stage_input(call)          x, act_quant, ids, weights, the shared
    //                              expert's row; the full slot list for B.
    //   stage_rows(call, slots)    the table rows of routed slots now resident.
    //   record_gateup(cmd, slots)  dispatch A (+ h quantisation) over `slots`
    //                              only -- the shared expert is slot `topk`.
    //   record_down(cmd)           dispatch B over all seven.
    //
    // A over early slots in one submit, A over the late ones and B in the
    // next: bit-identical to `stage` + `record` (the deferred schedule).
    Result<void> stage_input(const MoeCall& call);
    Result<void> stage_rows(const MoeCall& call, std::span<const uint32_t> slots);
    Result<void> record_gateup(gpu::CommandBuffer& cmd, std::span<const uint32_t> slots);
    Result<void> record_down(gpu::CommandBuffer& cmd);

    uint64_t     y_address() const { return runner_.y_address(); }
    const float* y_host() { return runner_.y(); }
    const gpu::MoeSpec& spec() const { return runner_.spec(); }

    // --- Track T / DSpark: the verify batch (docs/p4_dspark_runtime.md §2.2) ---
    //
    // A speculative verify batch is M = k+1 <= 6 tokens whose top-k expert sets
    // are DIFFERENT (docs/p3_dspark.md §12.6 measures a union of ~26 over the six
    // columns), and MoeRunner can express exactly one expert set per dispatch --
    // `ids()` is `[slots]`, the `[m][slots]` axis of `route_weights` is not an
    // expert axis (docs/p4_mgt1.md §7 open item 1). So the batch is M dispatches
    // over the same activations buffer, one column each.
    //
    // `stage_batch` does the host half ONCE for all columns: x out of
    // GPU-visible memory, act_quant, the shared expert's table row, and each
    // column's routing weights into its own row of `route_weights()`.
    // `run_batch` then makes one column's expert set visible per dispatch, and
    // pairs that dispatch with ITS OWN column: dispatch m reads x[m] and
    // `route_weights()[m]`, and `call.y` is filled from y[m]. That is the whole
    // contract -- the kernel has no live-count mask, so every dispatch rewrites
    // all M columns of h and y and only the column whose x and routing weights
    // the host actually fed is meaningful (docs/p4_dspark_runtime.md: the
    // appendix added 2026-09-17).
    //
    // What is copied per column is the same activation the M = 1 path copies, in
    // the same block layout, so a column's result is bit-for-bit the M = 1 result
    // for that token (tests/test_gpu_moe.cpp checks exactly that).
    struct BatchCall {
        uint32_t layer = 0;
        uint32_t m = 0;                      // 1..kMoeBatchMax
        const uint32_t* ids = nullptr;       // [m][topk] routed expert ids, row m = token m
        const float*    weights = nullptr;   // [m][topk] routing weights, already x route_scale
        uint32_t        topk = 0;
        const float*    x = nullptr;         // [m][hidden] ffn_norm outputs, host or GPU-visible
        float*          y = nullptr;         // [m][hidden], written on the host
        uint32_t        hidden = 0;
    };
    // The largest batch this interface accepts: gpu::kMgtMaxM, i.e. one verify
    // batch of the DSpark cycle (k = 5 drafts + the last accepted token).
    static constexpr uint32_t kMoeBatchMax = 6;

    // Human-readable per-column cost of the last `run_batch`, so a caller can
    // see how much of the batch is the per-dispatch overhead the MoE batching
    // gap costs. `gpu_ms` is the sum over columns.
    struct BatchTiming {
        double   stage_ms = 0.0;    // the one-off host half (x read + act_quant + shared row)
        double   table_ms = 0.0;    // the per-column expert table rows
        double   gpu_ms = 0.0;      // sum of the columns' dispatch time
        double   wall_ms = 0.0;     // everything, as the caller experiences it
        uint32_t columns = 0;
        std::string to_string() const;
    };

    // Which column a dispatch's x comes from is NOT a knob.
    //
    // A dispatch reads x[column], `route_weights()[column]` and slots its h and
    // y in that same column, so column m of the batch has to be staged in column
    // m of every one of those buffers -- which is what `stage_batch` and
    // `run_batch` do. An `XLayout` enum used to live here whose `OneColumn` mode
    // pushed each column into slot 0 instead; it was dead weight and worse than
    // that: `run_batch` clobbered all six x columns with the current column's
    // activation whatever the enum said, so the enum was inert *and* every
    // column was evaluated against `route_weights()[0]`, which is what made
    // columns 1..M-1 disagree with the M = 1 run by 15-22% of |y|max
    // (that appendix). Deleted rather than repaired: the
    // per-column shape costs one act_quant for the whole batch instead of one
    // per column and is bit-exact.
    Result<void> stage_batch(const BatchCall& call);
    // One column a dispatch; column m's dispatch is read back from y[m].
    Result<void> run_batch(const BatchCall& call);
    const BatchTiming& batch_timing() const { return batch_timing_; }

    // The x the runner ended up holding, for a validator that wants to compare
    // what was staged with what the kernel reads. `[m][hidden]`, fp16. Columns
    // >= the last call's `m` still hold the call before's activations.
    const uint16_t* debug_x() { return runner_.x_fp16(); }

    // Whether the shared expert is being computed. False when the layer's
    // shared-expert weights are not in the pinned set, in which case `stage`
    // fails rather than returning only the routed half and looking right.
    bool shared_ready() const { return shared_ok_; }

    // What the last `stage` / `run` spent, split so the §13.1 MoE bucket can
    // be read as "kernel" against "everything around it".
    struct Timing {
        double gpu_ms   = 0.0;      // `run` only; the token loop times the GPU itself
        double wall_ms  = 0.0;      // `run` only: submit to fence
        double host_ms  = 0.0;      // `stage` (and `run`'s y copy)
        double x_read_ms = 0.0;     //   of which: x out of write-combining memory
        double quant_ms  = 0.0;     //   of which: act_quant + fp16, 5120 values
        double table_ms  = 0.0;     //   of which: residency check + 7 table rows
    };
    const Timing& timing() const { return timing_; }

private:
    Result<void> bind_shared(uint32_t layer);

    store::ExpertStore*       store_   = nullptr;
    store::Planner*           planner_ = nullptr;
    const store::PinnedStore* pinned_  = nullptr;
    const TextConfig*         cfg_     = nullptr;
    gpu::MoeRunner            runner_;
    uint32_t                  shared_index_ = 0;          // == n_routed_experts
    uint64_t                  shared_addr_[kExpertPartCount]{};
    std::vector<std::array<uint64_t, kExpertPartCount>> shared_rows_;
    uint32_t                  shared_layer_ = 0xFFFFFFFFu;
    bool                      shared_ok_    = false;
    // Ordinary host staging: computing over GPU-visible memory a float at a
    // time is 60x slower than one memcpy each way (docs/p2_decode.md §3.3).
    std::vector<float>        xf_;
    std::vector<uint16_t>     xq_;
    Timing                    timing_{};
    // The verify batch's staging buffers, sized to kMoeBatchMax columns.
    std::vector<float>        xf_batch_;
    std::vector<uint16_t>     xq_batch_;
    BatchTiming               batch_timing_{};
};

}  // namespace deepmoe::runtime
