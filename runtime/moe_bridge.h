// The one file that knows about the design §7.9 MoE kernels.
//
// runtime/decode_layer.h asks for `MoeCall{layer, ids, weights, x, y}`. This
// turns that into MoeRunner, MoeSpec, the expert pointer table, the fp8
// shared-expert flag and the routed-expert fetch. The MoE kernels are owned by
// another track and are being changed while this is written, so the coupling
// is deliberately one file wide: a change to their push constants or their
// specialisation constants is a change here and nowhere else.
//
// Two runners, not one
// --------------------
// The routed experts are FP4 and live in ExpertStore slots addressed by
// `(layer, expert)`; the shared expert is FP8 and lives in the pinned set,
// which has no expert id. Rather than widen the pointer table by one column
// and re-derive every stride, the shared expert gets its own single-slot
// runner whose table this file fills directly, and the two outputs are summed
// in fp32 -- which is what `MoE.forward` does anyway (`y += shared(x)`).
//
// The activation round trip
// ------------------------
// `linear()` quantises the input of an fp4 or fp8 weight to fp8 E4M3, block
// 32, UE8M0 scale before the GEMM (see gpu/shaders/attn_common.slang). The MoE
// kernels take x as fp16, so the round trip happens here, on the host, once
// per layer. That is not an approximation: `act_quant` is a property of the
// activation alone, so quantising once instead of once per expert produces the
// identical bytes six times less often -- the same argument tools/dsref.py
// makes for `input_is_quantised`.
//
// Ownership/threading: borrows the store, the planner and the pinned set; owns
// its two runners. Single-threaded, on the GPU submit thread.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/moe_kernels.h"
#include "model/v41_config.h"
#include "runtime/decode_layer.h"
#include "store/expert_store.h"
#include "store/pinned.h"
#include "store/planner.h"

namespace deepmoe::runtime {

struct MoeBridgeConfig {
    // Passed straight to MoeSpec. The defaults are kernel_p1.md §3.2's winner
    // at M=1 plus the h quantisation design §7.9 v0.6 added.
    uint32_t lanes_per_row = 32;
    uint32_t subgroup_size = 32;
    uint32_t decode_mode   = 0;   // 0 = the constant table, which P1 measured fastest
    uint32_t rows_per_lane = 1;
    // 1 reproduces `act_quant(silu(gate) * up)` inside dispatch B, which is
    // what the reference does before w2 (route_trace.md §11.2).
    uint32_t h_quant       = 1;
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

    const char* name() const override { return "gpu(moe_gateup+moe_down)"; }
    Result<void> run(const MoeCall& call) override;

    // Whether the shared expert is being computed. False when the layer's
    // shared-expert weights are not in the pinned set, in which case `run`
    // returns only the routed half and says so rather than looking right.
    bool shared_ready() const { return shared_ok_; }

    // What the last `run` spent, split so the §13.1 MoE bucket can be read as
    // "kernel" against "everything around it". The host half is the activation
    // round trip and the two accumulations, all of which read write-combining
    // memory (design §3.3) and are therefore not free.
    struct Timing {
        double routed_gpu_ms = 0.0, routed_wall_ms = 0.0;
        double shared_gpu_ms = 0.0, shared_wall_ms = 0.0;
        double host_ms       = 0.0;
    };
    const Timing& timing() const { return timing_; }

private:
    Result<void> bind_shared(uint32_t layer);

    store::ExpertStore*       store_   = nullptr;
    store::Planner*           planner_ = nullptr;
    const store::PinnedStore* pinned_  = nullptr;
    const TextConfig*         cfg_     = nullptr;
    gpu::MoeRunner            routed_, shared_;
    uint32_t                  shared_layer_ = 0xFFFFFFFFu;
    bool                      shared_ok_    = false;
    std::vector<uint16_t>     xq_;
    // Ordinary host staging for x and the two y vectors: see the memcpy note in
    // `run`. Computing over the GPU-visible originals is 60x slower.
    std::vector<float>        xf_, yf_, sf_;
    Timing                    timing_{};
};

}  // namespace deepmoe::runtime
