#include "runtime/moe_bridge.h"

#include <cstring>
#include <format>

#include "cpu/dequant.h"
#include "model/layout.h"

namespace deepmoe::runtime {

Result<void> GpuMoeBridge::create(gpu::Device& device, gpu::MemoryAllocator& alloc,
                                  const std::string& shader_dir, store::ExpertStore& store,
                                  store::Planner& planner, const store::PinnedStore& pinned,
                                  const TextConfig& cfg, const MoeBridgeConfig& bc) {
    destroy();
    store_ = &store;
    planner_ = &planner;
    pinned_ = &pinned;
    cfg_ = &cfg;

    gpu::MoeSpec spec;
    spec.m             = 1;
    spec.lanes_per_row = bc.lanes_per_row;
    spec.subgroup_size = bc.subgroup_size;
    spec.decode_mode   = bc.decode_mode;
    spec.rows_per_lane = bc.rows_per_lane;
    spec.h_quant       = bc.h_quant;

    gpu::MoeDims rd;
    rd.experts_per_layer = cfg.n_routed_experts;
    rd.slots             = cfg.num_experts_per_tok;
    rd.hidden            = cfg.hidden_size;
    rd.inter             = cfg.moe_intermediate_size;
    rd.table_layers      = layout::kTotalLogicalLayers;
    rd.swiglu_limit      = static_cast<float>(cfg.swiglu_limit);
    if (auto r = routed_.create(device, alloc, shader_dir, spec, rd); !r) return r;

    // The shared expert: one slot, one "layer", one "expert", fp8 weights whose
    // six part addresses this file writes into the table by hand.
    gpu::MoeSpec ss = spec;
    ss.fp8_slots = 1;
    gpu::MoeDims sd;
    sd.layer             = 0;
    sd.experts_per_layer = 1;
    sd.slots             = 1;
    sd.hidden            = cfg.hidden_size;
    sd.inter             = cfg.moe_intermediate_size;
    sd.table_layers      = 1;
    sd.fp8_slot_count    = 1;
    sd.swiglu_limit      = static_cast<float>(cfg.swiglu_limit);
    if (auto r = shared_.create(device, alloc, shader_dir, ss, sd); !r) return r;

    xq_.assign(cfg.hidden_size, 0);
    return {};
}

void GpuMoeBridge::destroy() {
    routed_.destroy();
    shared_.destroy();
    store_ = nullptr;
    planner_ = nullptr;
    pinned_ = nullptr;
    shared_ok_ = false;
    shared_layer_ = 0xFFFFFFFFu;
}

Result<void> GpuMoeBridge::bind_shared(uint32_t layer) {
    if (shared_layer_ == layer) return {};
    shared_ok_ = false;
    const std::string pre = std::format("layers.{}.ffn.shared_experts", layer);
    // The order is model/manifest.h's ExpertPart: w1.weight, w1.scale,
    // w2.weight, w2.scale, w3.weight, w3.scale. Getting it wrong swaps the
    // gate and up branches, which is silent and wrong.
    const char* mats[3] = {"w1", "w2", "w3"};
    uint64_t* t = shared_.pointer_table();
    for (uint32_t i = 0; i < 3; ++i) {
        auto p = pinned_->require(std::format("{}.{}.weight", pre, mats[i]));
        if (!p) return std::unexpected(p.error());
        if ((*p)->scale == kNoDeviceAddress)
            return fail(Err::FailedPrecondition,
                        std::format("{}.{} has no fp8 scale plane", pre, mats[i]));
        t[i * 2 + 0] = (*p)->data;
        t[i * 2 + 1] = (*p)->scale;
    }
    shared_.ids()[0]       = 0u | gpu::kSlotFp8;
    shared_.slot_list()[0] = 0;
    shared_.set_list_count(1);
    shared_.route_weights()[0] = 1.0f;      // the shared expert is not routed
    shared_layer_ = layer;
    shared_ok_ = true;
    return {};
}

Result<void> GpuMoeBridge::run(const MoeCall& call) {
    if (!store_ || !planner_) return fail(Err::FailedPrecondition, "MoE bridge is not created");
    const uint32_t dim = call.hidden;

    // `act_quant(x, 32, ue8m0)` once for the whole layer. See the header.
    for (uint32_t b = 0; b < dim / 32; ++b) {
        uint8_t bytes[32];
        float back[32];
        cpu::act_quant_block(call.x + b * 32, 32, bytes, back);
        for (uint32_t i = 0; i < 32; ++i)
            xq_[b * 32 + i] = cpu::float_to_fp16(back[i]);
    }

    // The routed half. design §7.1's residency gate has already run by the
    // time we get here (DecodeLayer::run_moe), so this only has to confirm it.
    for (uint32_t s = 0; s < call.topk; ++s) {
        const ExpertKey key{static_cast<uint16_t>(call.layer),
                            static_cast<uint16_t>(call.ids[s])};
        if (!store_->resident(key))
            return fail(Err::FailedPrecondition,
                        std::format("expert ({}, {}) is not resident at the MoE dispatch",
                                    key.layer, key.expert));
    }
    // MoeDims::layer is fixed when a MoeRunner is created and the kernel
    // indexes `(layer * experts_per_layer + expert) * 6`, so a runner made for
    // layer 0 only ever reads row 0. Rather than create forty runners (forty
    // pipeline pairs) or reach into MoeRunner's push constants, row 0 of this
    // runner's table is rewritten each call with the six addresses of the
    // layer we are actually on. Six entries of six words.
    for (uint32_t s = 0; s < call.topk; ++s) {
        const ExpertKey key{static_cast<uint16_t>(call.layer),
                            static_cast<uint16_t>(call.ids[s])};
        for (uint32_t part = 0; part < kExpertPartCount; ++part) {
            auto a = store_->table_entry(key, static_cast<ExpertPart>(part));
            if (!a) return std::unexpected(a.error());
            routed_.pointer_table()[size_t(call.ids[s]) * kExpertPartCount + part] = *a;
        }
        routed_.ids()[s] = call.ids[s];
        routed_.slot_list()[s] = s;
        routed_.route_weights()[s] = call.weights[s];
    }
    routed_.set_list_count(call.topk);
    std::memcpy(routed_.x_fp16(), xq_.data(), size_t(dim) * sizeof(uint16_t));
    if (auto r = routed_.run(1); !r) return std::unexpected(r.error());
    const float* ry = routed_.y();
    for (uint32_t i = 0; i < dim; ++i) call.y[i] = ry[i];

    // The shared expert, added in fp32 exactly as `MoE.forward` does.
    if (auto r = bind_shared(call.layer); !r) return r;
    std::memcpy(shared_.x_fp16(), xq_.data(), size_t(dim) * sizeof(uint16_t));
    if (auto r = shared_.run(1); !r) return std::unexpected(r.error());
    const float* sy = shared_.y();
    for (uint32_t i = 0; i < dim; ++i) call.y[i] += sy[i];
    return {};
}

}  // namespace deepmoe::runtime
