#include "runtime/moe_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <immintrin.h>

#include "core/log.h"
#include "cpu/dequant.h"
#include "model/layout.h"

namespace deepmoe::runtime {
namespace {

double ms_between(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// `act_quant(x, 32, ue8m0)` followed by the fp16 the MoE kernels take, for a
// whole activation at once.
//
// The scalar form -- cpu::act_quant_block, then fp8_e4m3_to_float of the byte,
// then cpu::float_to_fp16 -- is three branchy calls per element, and at 5,120
// elements a layer it was most of the "host" half of the MoE bucket. This is
// gpu/shaders/attn_common.slang's `fp8_round` transcribed: round-to-nearest-
// even onto the E4M3 grid as an integer add and a mask on the float's bit
// pattern for a normal, and `(a + 1.5 * 2^14) - 1.5 * 2^14` for a subnormal,
// both branch-free so the compiler vectorises the block loop; then one F16C
// conversion per sixteen values. `self_check` below proves it is the same
// function as the scalar one before the bridge will use it.
inline float fp8_round_value(float v) {
    const float a = std::fabs(v);
    uint32_t b;
    std::memcpy(&b, &a, 4);
    const uint32_t r = (b + 0x7FFFFu + ((b >> 20) & 1u)) & 0xFFF00000u;
    float rf;
    std::memcpy(&rf, &r, 4);
    const float sub = (a + 24576.0f) - 24576.0f;
    const float mag = (a < 0.015625f) ? sub : rf;
    return std::copysign(mag, v);
}

void act_quant_to_fp16(const float* x, uint16_t* out, float* scratch, uint32_t n) {
    const uint32_t blocks = n / 32;
    for (uint32_t blk = 0; blk < blocks; ++blk) {
        const float* v = x + size_t(blk) * 32;
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float s = cpu::fp8_round_scale(amax);
        const float inv = 1.0f / s;
        float* q = scratch + size_t(blk) * 32;
        for (uint32_t i = 0; i < 32; ++i) {
            const float c = std::clamp(v[i] * inv, -448.0f, 448.0f);
            q[i] = fp8_round_value(c) * s;
        }
    }
    uint32_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        const __m256i h = _mm512_cvtps_ph(_mm512_loadu_ps(scratch + i),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), h);
    }
#endif
    for (; i < n; ++i) out[i] = cpu::float_to_fp16(scratch[i]);
}

// The fast path against the scalar reference, over values that cover every
// range the rounding distinguishes: subnormal E4M3, the normal grid, ties, the
// clamp at 448, and fp16's own subnormal and overflow boundaries after the
// scale. Run once at create; a mismatch is a refusal to start, not a warning.
Result<void> self_check() {
    const uint32_t n = 5120;
    std::vector<float> x(n), scratch(n), back(32);
    std::vector<uint16_t> fast(n), slow(n);
    uint32_t seed = 0x9E3779B9u;
    for (uint32_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const float u = float(seed >> 8) / float(1u << 24);
        // Log-uniform magnitudes from 2^-30 to 2^16, both signs, with a block
        // every so often that is exactly on the E4M3 tie points.
        float mag = std::ldexp(1.0f, int(u * 46.0f) - 30) * (1.0f + u);
        if ((i / 32) % 7 == 3) mag = std::ldexp(1.0f + 0.0625f * float(i % 16), int(i % 9) - 4);
        x[i] = (seed & 0x100u) ? -mag : mag;
    }
    act_quant_to_fp16(x.data(), fast.data(), scratch.data(), n);
    for (uint32_t b = 0; b < n / 32; ++b) {
        uint8_t bytes[32];
        cpu::act_quant_block(x.data() + b * 32, 32, bytes, back.data());
        for (uint32_t i = 0; i < 32; ++i) slow[b * 32 + i] = cpu::float_to_fp16(back[i]);
    }
    for (uint32_t i = 0; i < n; ++i)
        if (fast[i] != slow[i])
            return fail(Err::Internal,
                        std::format("moe bridge: the vectorised act_quant disagrees with "
                                    "cpu::act_quant_block at element {} (x={}, {:#06x} vs "
                                    "{:#06x})", i, x[i], fast[i], slow[i]));
    return {};
}

}  // namespace

Result<void> GpuMoeBridge::create(gpu::Device& device, gpu::MemoryAllocator& alloc,
                                  const std::string& shader_dir, store::ExpertStore& store,
                                  store::Planner& planner, const store::PinnedStore& pinned,
                                  const TextConfig& cfg, const MoeBridgeConfig& bc) {
    destroy();
    if (auto r = self_check(); !r) return r;
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
    spec.x_mode        = bc.x_mode;
    spec.h_quant       = bc.h_quant;
    spec.fp8_slots     = 1;          // slot 6 is the fp8 shared expert

    // One row of table: MoeDims::layer is fixed when a runner is created and
    // the kernel indexes `(layer * experts_per_layer + expert) * 6`, so row 0
    // is rewritten each call with the addresses of the layer being run. One
    // expert wider than the model, so index `n_routed_experts` is free for the
    // shared expert.
    gpu::MoeDims d;
    d.layer             = 0;
    d.experts_per_layer = cfg.n_routed_experts + 1;
    d.slots             = cfg.num_experts_per_tok + 1;
    d.hidden            = cfg.hidden_size;
    d.inter             = cfg.moe_intermediate_size;
    d.table_layers      = 1;
    d.fp8_slot_count    = 1;
    d.swiglu_limit      = static_cast<float>(cfg.swiglu_limit);
    if (auto r = runner_.create(device, alloc, shader_dir, spec, d); !r) return r;
    shared_index_ = cfg.n_routed_experts;

    xf_.assign(cfg.hidden_size, 0.0f);
    xq_.assign(cfg.hidden_size, 0);
    log_info("moe bridge: {} ({} slots: {} fp4 routed + 1 fp8 shared)",
             spec.name(), d.slots, cfg.num_experts_per_tok);
    return {};
}

void GpuMoeBridge::destroy() {
    runner_.destroy();
    store_ = nullptr;
    planner_ = nullptr;
    pinned_ = nullptr;
    shared_ok_ = false;
    shared_layer_ = 0xFFFFFFFFu;
}

Result<void> GpuMoeBridge::bind_shared(uint32_t layer) {
    if (shared_layer_ != layer) {
        shared_ok_ = false;
        const std::string pre = std::format("layers.{}.ffn.shared_experts", layer);
        // The order is model/manifest.h's ExpertPart: w1.weight, w1.scale,
        // w2.weight, w2.scale, w3.weight, w3.scale. Getting it wrong swaps the
        // gate and up branches, which is silent and wrong.
        const char* mats[3] = {"w1", "w2", "w3"};
        for (uint32_t i = 0; i < 3; ++i) {
            auto p = pinned_->require(std::format("{}.{}.weight", pre, mats[i]));
            if (!p) return std::unexpected(p.error());
            if ((*p)->scale == kNoDeviceAddress)
                return fail(Err::FailedPrecondition,
                            std::format("{}.{} has no fp8 scale plane", pre, mats[i]));
            shared_addr_[i * 2 + 0] = (*p)->data;
            shared_addr_[i * 2 + 1] = (*p)->scale;
        }
        shared_layer_ = layer;
        shared_ok_ = true;
    }
    // Written every call, not only on a layer change: row 0 is shared by every
    // layer and nothing else guarantees the previous writer left it alone.
    std::memcpy(runner_.pointer_table() + size_t(shared_index_) * kExpertPartCount,
                shared_addr_, sizeof shared_addr_);
    return {};
}

Result<void> GpuMoeBridge::stage(const MoeCall& call) {
    if (!store_ || !planner_) return fail(Err::FailedPrecondition, "MoE bridge is not created");
    const uint32_t dim = call.hidden;
    const uint32_t slots = runner_.dims().slots;
    if (call.topk + 1 != slots)
        return fail(Err::InvalidArgument,
                    std::format("{} routed experts, but the runner has {} slots for "
                                "routed + shared", call.topk, slots));
    timing_ = Timing{};
    const TimePoint t0 = Clock::now();

    // Copy x out of GPU-visible memory BEFORE computing over it: design §3.3's
    // uncached write-combining read, one memcpy instead of 5,120 loads
    // (docs/p2_decode.md §3.3).
    std::memcpy(xf_.data(), call.x, size_t(dim) * sizeof(float));
    const TimePoint t1 = Clock::now();

    // `act_quant(x, 32, ue8m0)` once for the whole layer, straight into the
    // runner's fp16 input. The scratch is the runner-independent host copy;
    // quantising in place over it is safe because each block reads its 32
    // values before writing them.
    act_quant_to_fp16(xf_.data(), xq_.data(), xf_.data(), dim);
    std::memcpy(runner_.x_fp16(), xq_.data(), size_t(dim) * sizeof(uint16_t));
    const TimePoint t2 = Clock::now();

    // The routed half. design §7.1's residency gate has already run by the
    // time we get here, so this only has to confirm it.
    uint64_t row[kExpertPartCount];
    uint64_t* table = runner_.pointer_table();
    for (uint32_t s = 0; s < call.topk; ++s) {
        const ExpertKey key{static_cast<uint16_t>(call.layer),
                            static_cast<uint16_t>(call.ids[s])};
        if (!store_->resident(key))
            return fail(Err::FailedPrecondition,
                        std::format("expert ({}, {}) is not resident at the MoE dispatch",
                                    key.layer, key.expert));
        for (uint32_t part = 0; part < kExpertPartCount; ++part) {
            auto a = store_->table_entry(key, static_cast<ExpertPart>(part));
            if (!a) return std::unexpected(a.error());
            row[part] = *a;
        }
        std::memcpy(table + size_t(call.ids[s]) * kExpertPartCount, row, sizeof row);
    }
    if (auto r = bind_shared(call.layer); !r) return r;

    uint32_t ids[16];
    uint32_t list[16];
    float    w[16];
    for (uint32_t s = 0; s < call.topk; ++s) {
        ids[s]  = call.ids[s];
        list[s] = s;
        w[s]    = call.weights[s];
    }
    // The shared expert: not routed, weight 1, fp8.
    ids[call.topk]  = shared_index_ | gpu::kSlotFp8;
    list[call.topk] = call.topk;
    w[call.topk]    = 1.0f;
    std::memcpy(runner_.ids(), ids, slots * sizeof(uint32_t));
    std::memcpy(runner_.slot_list(), list, slots * sizeof(uint32_t));
    std::memcpy(runner_.route_weights(), w, slots * sizeof(float));
    runner_.set_list_count(slots);
    runner_.set_accumulate(false);
    const TimePoint t3 = Clock::now();

    timing_.x_read_ms = ms_between(t0, t1);
    timing_.quant_ms  = ms_between(t1, t2);
    timing_.table_ms  = ms_between(t2, t3);
    timing_.host_ms   = ms_between(t0, t3);
    return {};
}

Result<void> GpuMoeBridge::record(gpu::CommandBuffer& cmd) {
    return runner_.record_into(cmd, gpu::MoePhase::Both);
}

Result<void> GpuMoeBridge::run(const MoeCall& call) {
    if (auto r = stage(call); !r) return r;
    const Timing staged = timing_;
    const TimePoint t0 = Clock::now();
    auto rt = runner_.run(1);
    if (!rt) return std::unexpected(rt.error());
    const TimePoint t1 = Clock::now();
    if (call.y) std::memcpy(call.y, runner_.y(), size_t(call.hidden) * sizeof(float));
    timing_ = staged;
    timing_.gpu_ms  = rt->seconds_total * 1e3;
    timing_.wall_ms = ms_between(t0, t1);
    timing_.host_ms += ms_between(t1, Clock::now());
    return {};
}

}  // namespace deepmoe::runtime
