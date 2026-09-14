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

#include <cstdint>
#include <memory>
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

    uint64_t     y_address() const { return runner_.y_address(); }
    const float* y_host() { return runner_.y(); }
    const gpu::MoeSpec& spec() const { return runner_.spec(); }

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
    uint32_t                  shared_layer_ = 0xFFFFFFFFu;
    bool                      shared_ok_    = false;
    // Ordinary host staging: computing over GPU-visible memory a float at a
    // time is 60x slower than one memcpy each way (docs/p2_decode.md §3.3).
    std::vector<float>        xf_;
    std::vector<uint16_t>     xq_;
    Timing                    timing_{};
};

}  // namespace deepmoe::runtime
