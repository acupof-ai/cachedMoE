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
    xf_.assign(cfg.hidden_size, 0.0f);
    yf_.assign(cfg.hidden_size, 0.0f);
    sf_.assign(cfg.hidden_size, 0.0f);
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
    timing_ = Timing{};
    const TimePoint t_call = Clock::now();
    auto ms = [](TimePoint a, TimePoint b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Copy x out of GPU-visible memory BEFORE computing over it.
    //
    // design §3.3 is explicit that CPU reads of the path-A mapping are uncached
    // and very slow, and it means it: `call.x` is 5,120 fp32 in a
    // DEVICE_LOCAL|HOST_VISIBLE allocation, and reading it a float at a time
    // costs about 230 ns per access. The three loops this function used to run
    // straight off GPU memory -- x in, the routed y out, the shared y out --
    // were 25,600 such accesses a layer, which measured 6.0 ms a layer, 240 ms
    // a token, and 26% of the whole decode step. One `memcpy` per vector issues
    // wide loads instead and takes the same three loops to 0.1 ms a layer.
    std::memcpy(xf_.data(), call.x, size_t(dim) * sizeof(float));

    // `act_quant(x, 32, ue8m0)` once for the whole layer. See the header.
    for (uint32_t b = 0; b < dim / 32; ++b) {
        uint8_t bytes[32];
        float back[32];
        cpu::act_quant_block(xf_.data() + b * 32, 32, bytes, back);
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
    const TimePoint t_r0 = Clock::now();
    auto rt = routed_.run(1);
    if (!rt) return std::unexpected(rt.error());
    const TimePoint t_r1 = Clock::now();
    timing_.routed_gpu_ms  = rt->seconds_total * 1e3;
    timing_.routed_wall_ms = ms(t_r0, t_r1);
    std::memcpy(yf_.data(), routed_.y(), size_t(dim) * sizeof(float));

    // The shared expert, added in fp32 exactly as `MoE.forward` does.
    if (auto r = bind_shared(call.layer); !r) return r;
    std::memcpy(shared_.x_fp16(), xq_.data(), size_t(dim) * sizeof(uint16_t));
    const TimePoint t_s0 = Clock::now();
    auto stm = shared_.run(1);
    if (!stm) return std::unexpected(stm.error());
    const TimePoint t_s1 = Clock::now();
    timing_.shared_gpu_ms  = stm->seconds_total * 1e3;
    timing_.shared_wall_ms = ms(t_s0, t_s1);
    std::memcpy(sf_.data(), shared_.y(), size_t(dim) * sizeof(float));
    for (uint32_t i = 0; i < dim; ++i) yf_[i] += sf_[i];
    std::memcpy(call.y, yf_.data(), size_t(dim) * sizeof(float));
    timing_.host_ms = ms(t_call, Clock::now()) - timing_.routed_wall_ms - timing_.shared_wall_ms;
    return {};
}

}  // namespace deepmoe::runtime
