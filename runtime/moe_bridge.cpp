#include "runtime/moe_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <immintrin.h>

#include "core/log.h"
#include "cpu/dequant.h"
#include "model/layout.h"

namespace deepmoe::runtime {

void debug_act_quant_to_fp16(const float* x, uint16_t* out, float* scratch, uint32_t n);

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

void debug_act_quant_to_fp16(const float* x, uint16_t* out, float* scratch, uint32_t n) {
    act_quant_to_fp16(x, out, scratch, n);
}

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
    // The batch axis the kernels are specialised to: `m` is a specialisation
    // constant, so the M = 1 token loop and the M = 6 verify batch share one
    // pipeline. There is no live-count push constant -- every shader loop runs
    // `for (m = 0; m < M; ++m)` over the whole buffer (see `run_batch` below), so
    // the buffers carry all six columns: x 60 KB, h 295 KB (fp16 + fp8 + scale
    // planes), y 120 KB.
    spec.m             = kMoeBatchMax;
    spec.lanes_per_row = bc.lanes_per_row;
    spec.subgroup_size = bc.subgroup_size;
    spec.decode_mode   = bc.decode_mode;
    spec.rows_per_lane = bc.rows_per_lane;
    spec.x_mode        = bc.x_mode;
    spec.h_quant       = bc.h_quant;
    // An experiment knob, not a setting: docs/p2_decode.md §8.2 uses it to
    // A/B the h quantisation's placement for bit-reproducibility.
    if (const char* e = std::getenv("DEEPMOE_MOE_HQUANT"); e && *e)
        spec.h_quant = static_cast<uint32_t>(std::atoi(e));
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

    // Track F1: the union runner. Same pipelines' shape, a wider slot axis --
    // `MoeDims::slots` is what sizes ids/list/route_weights/h and the shaders
    // take it as `num_slots` in a push constant, so nothing about the kernels
    // changes. h is the one buffer this grows: [6][49][2304] fp16 + the fp8
    // plane is ~2.1 MB against 295 KB at seven slots, which is the whole cost of
    // making a verify batch one dispatch instead of six.
    gpu::MoeDims du = d;
    du.slots = kUnionSlotsMax;
    if (cfg.num_experts_per_tok * kMoeBatchMax + 1 < du.slots)
        du.slots = cfg.num_experts_per_tok * kMoeBatchMax + 1;
    if (auto r = union_runner_.create(device, alloc, shader_dir, spec, du); !r) return r;
    union_ids_.assign(du.slots, 0);
    union_slot_of_.assign(size_t(cfg.n_routed_experts) + 1, ~0u);

    xf_.assign(cfg.hidden_size, 0.0f);
    xq_.assign(cfg.hidden_size, 0);
    xf_batch_.assign(size_t(kMoeBatchMax) * cfg.hidden_size, 0.0f);
    xq_batch_.assign(size_t(kMoeBatchMax) * cfg.hidden_size, 0);
    log_info("moe bridge: {} ({} slots: {} fp4 routed + 1 fp8 shared; union runner {} slots)",
             spec.name(), d.slots, cfg.num_experts_per_tok, du.slots);
    return {};
}

void GpuMoeBridge::destroy() {
    union_runner_.destroy();
    runner_.destroy();
    store_ = nullptr;
    planner_ = nullptr;
    pinned_ = nullptr;
    shared_ok_ = false;
    shared_layer_ = 0xFFFFFFFFu;
}

Result<void> GpuMoeBridge::bind_shared(uint32_t layer) {
    return bind_shared_into(layer, runner_.pointer_table());
}

Result<void> GpuMoeBridge::bind_shared_into(uint32_t layer, uint64_t* dest_table) {
    // Resolved once per layer for the life of the bridge: the pinned set never
    // moves, and three formatted-name lookups a layer were most of the
    // "table" line of the host breakdown.
    if (layer < shared_rows_.size() && shared_rows_[layer][0]) {
        std::memcpy(dest_table + size_t(shared_index_) * kExpertPartCount,
                    shared_rows_[layer].data(), sizeof(uint64_t) * kExpertPartCount);
        shared_ok_ = true;
        return {};
    }
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
        if (layer >= shared_rows_.size()) shared_rows_.resize(layer + 1);
        std::memcpy(shared_rows_[layer].data(), shared_addr_, sizeof shared_addr_);
    }
    // Written every call, not only on a layer change: row 0 is shared by every
    // layer and nothing else guarantees the previous writer left it alone.
    std::memcpy(dest_table + size_t(shared_index_) * kExpertPartCount,
                shared_addr_, sizeof shared_addr_);
    return {};
}

Result<void> GpuMoeBridge::stage(const MoeCall& call) {
    if (auto r = stage_input(call); !r) return r;
    uint32_t all[16];
    for (uint32_t s = 0; s < call.topk; ++s) all[s] = s;
    return stage_rows(call, std::span<const uint32_t>(all, call.topk));
}

Result<void> GpuMoeBridge::stage_rows(const MoeCall& call, std::span<const uint32_t> slots) {
    const TimePoint t0 = Clock::now();
    uint64_t row[kExpertPartCount];
    uint64_t* table = runner_.pointer_table();
    for (uint32_t s : slots) {
        if (s >= call.topk) return fail(Err::InvalidArgument, "stage_rows: not a routed slot");
        const ExpertKey key{static_cast<uint16_t>(call.layer),
                            static_cast<uint16_t>(call.ids[s])};
        if (auto r = store_->table_row(key, row); !r)
            return fail(r.error().code,
                        std::format("at the MoE dispatch: {}", r.error().message));
        std::memcpy(table + size_t(call.ids[s]) * kExpertPartCount, row, sizeof row);
    }
    const double ms = ms_between(t0, Clock::now());
    timing_.table_ms += ms;
    timing_.host_ms  += ms;
    return {};
}

Result<void> GpuMoeBridge::record_gateup(gpu::CommandBuffer& cmd, std::span<const uint32_t> slots) {
    if (slots.empty() || slots.size() > runner_.dims().slots)
        return fail(Err::InvalidArgument, "record_gateup: 1..slots slots");
    std::memcpy(runner_.slot_list_alt(), slots.data(), slots.size() * sizeof(uint32_t));
    return runner_.record_gateup_alt(cmd, static_cast<uint32_t>(slots.size()));
}

Result<void> GpuMoeBridge::record_down(gpu::CommandBuffer& cmd) {
    return runner_.record_into(cmd, gpu::MoePhase::DownOnly);
}

Result<void> GpuMoeBridge::stage_input(const MoeCall& call) {
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

    // The routed half's table rows are `stage_rows`'s: design §7.1's residency
    // gate decides when each one may be written.
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

// --- the verify batch (docs/p4_dspark_runtime.md §2.2) -----------------------

std::string GpuMoeBridge::BatchTiming::to_string() const {
    return std::format("{} columns: stage {:.2f} ms, expert rows {:.2f} ms, "
                       "dispatches {:.2f} ms, wall {:.2f} ms ({:.2f} ms a column)",
                       columns, stage_ms, table_ms, gpu_ms, wall_ms,
                       columns ? wall_ms / columns : 0.0);
}

Result<void> GpuMoeBridge::stage_batch(const BatchCall& call) {
    if (!store_ || !planner_) return fail(Err::FailedPrecondition, "MoE bridge is not created");
    if (call.m == 0 || call.m > kMoeBatchMax)
        return fail(Err::InvalidArgument,
                    std::format("a verify batch of {} columns; the interface takes 1..{}",
                                call.m, kMoeBatchMax));
    const uint32_t dim = call.hidden;
    const uint32_t slots = runner_.dims().slots;
    if (call.topk + 1 != slots)
        return fail(Err::InvalidArgument,
                    std::format("{} routed experts, but the runner has {} slots for routed + shared",
                                call.topk, slots));
    const TimePoint t0 = Clock::now();
    batch_timing_ = BatchTiming{};
    batch_timing_.columns = call.m;

    // Every column's activation into its own column of the runner's x buffer,
    // once for the whole batch. `run_batch` reads each column back from the same
    // index it staged, so nothing re-stages x per dispatch.
    for (uint32_t m = 0; m < call.m; ++m) {
        const float* xin = call.x + size_t(m) * dim;
        float*    xf = xf_batch_.data() + size_t(m) * dim;
        uint16_t* xq = xq_batch_.data() + size_t(m) * dim;
        std::memcpy(xf, xin, size_t(dim) * sizeof(float));
        act_quant_to_fp16(xf, xq, xf, dim);
        std::memcpy(runner_.x_fp16() + size_t(m) * dim, xq, size_t(dim) * sizeof(uint16_t));
    }
    if (auto r = bind_shared(call.layer); !r) return r;
    const TimePoint t1 = Clock::now();

    // Every column's routing weights, once: the kernel reads
    // `route_weights[m * slots + s]`, which is the one axis of this runner that
    // is already a batch axis.
    for (uint32_t m = 0; m < call.m; ++m) {
        float* w = runner_.route_weights() + size_t(m) * slots;
        for (uint32_t s = 0; s < call.topk; ++s) w[s] = call.weights[size_t(m) * call.topk + s];
        w[call.topk] = 1.0f;                       // the shared expert
    }
    const TimePoint t2 = Clock::now();
    batch_timing_.stage_ms = ms_between(t0, t1);
    batch_timing_.table_ms += ms_between(t1, t2);
    return {};
}

Result<void> GpuMoeBridge::run_batch(const BatchCall& call) {
    const TimePoint t_all = Clock::now();
    if (auto r = stage_batch(call); !r) return r;
    const uint32_t slots = runner_.dims().slots;
    const uint32_t dim = call.hidden;

    // There is no live-count mask on the kernel side. M is a specialisation
    // constant and BOTH shaders loop `for (m = 0; m < M; ++m)` over the whole
    // buffer -- GateUpPush/DownPush carry layer, experts_per_layer, num_slots,
    // n_rows, k (, list_count, flags) and NO column count -- so one dispatch
    // recomputes every column of h and of y from x[m'] and RouteW[m'],
    // whatever the caller's live count is. A dispatch expresses one expert set
    // (`ids()` is `[slots]`), and after dispatch m the only column whose
    // (x, routing weight, y) triple is token m's is COLUMN m:
    //
    //     dispatch m:  ids = token m's experts
    //                  h[m]  from x[m] and RouteW[m * num_slots + slot]
    //                  y[m]  = token m's MoE output
    //
    // So a column's activation, its routing weights and the y it is read back
    // from must all carry the SAME column index. Pairing column m's x with
    // RouteW row 0 -- or copying y[0] for every column, as this did -- mixes one
    // token's activation with another token's routing and is off by 15-22% of
    // |y|max (docs/p4_dspark_runtime.md: the appendix added 2026-09-17, last
    // section of the file).
    //
    // Columns >= call.m still hold the previous call's activations; their h and
    // y are computed and thrown away, which is what M being a compile-time
    // constant costs (that appendix's "per-column cost" item).
    for (uint32_t m = 0; m < call.m; ++m) {
        // One column at a time: MoeRunner expresses one expert set per dispatch
        // (ids() is [slots]), so token m's six experts are made visible here.
        // The pointer table rows are the residency gate's, as at M = 1.
        uint64_t row[kExpertPartCount];
        uint64_t* table = runner_.pointer_table();
        for (uint32_t s = 0; s < call.topk; ++s) {
            const ExpertKey key{static_cast<uint16_t>(call.layer),
                                static_cast<uint16_t>(call.ids[size_t(m) * call.topk + s])};
            if (auto r = store_->table_row(key, row); !r)
                return fail(r.error().code,
                            std::format("at the verify batch's column {}: {}", m,
                                        r.error().message));
            std::memcpy(table + size_t(call.ids[size_t(m) * call.topk + s]) * kExpertPartCount,
                        row, sizeof row);
        }
        uint32_t ids[16];
        uint32_t list[16];
        for (uint32_t s = 0; s < call.topk; ++s) {
            ids[s]  = call.ids[size_t(m) * call.topk + s];
            list[s] = s;
        }
        ids[call.topk]  = shared_index_ | gpu::kSlotFp8;
        list[call.topk] = call.topk;
        std::memcpy(runner_.ids(), ids, slots * sizeof(uint32_t));
        std::memcpy(runner_.slot_list(), list, slots * sizeof(uint32_t));
        runner_.set_list_count(slots);
        runner_.set_accumulate(false);
        // The kernels are specialised on M and loop `m < pc.m`, so a dispatch
        // pays only for the columns it is TOLD to compute. Column m's activation
        // is in x[m] and its weights are in route_weights()[m], so `pc.m = m+1`
        // makes this dispatch compute exactly column m of both -- the shape of
        // one token, instead of six columns of work for it. docs/p4_mgt1.md §4
        // measured the difference: 1.79-1.95 ms a column at M=6 against 1.786 at
        // M=1, which was all shape.
        runner_.set_live_columns(m + 1);
        auto rt = runner_.run(1);
        if (!rt) return std::unexpected(rt.error());
        batch_timing_.gpu_ms += rt->seconds_total * 1e3;
        // Column m, not column 0: y[m] is the column the dispatch above computed
        // from x[m] and RouteW[m], so it is bit-for-bit the M = 1 result for
        // token m (tests/test_gpu_moe.cpp checks exactly that).
        if (call.y)
            std::memcpy(call.y + size_t(m) * dim, runner_.y() + size_t(m) * dim,
                        size_t(dim) * sizeof(float));
    }
    batch_timing_.wall_ms = ms_between(t_all, Clock::now());
    return {};
}

// --- the union batch (Track F1) ---------------------------------------------

std::string GpuMoeBridge::UnionInfo::to_string() const {
    return std::format("{} columns over {} distinct experts (+ shared, {} slots): "
                       "stage {:.2f} ms, expert rows {:.2f} ms, dispatch {:.2f} ms, "
                       "wall {:.2f} ms ({:.2f} ms a column)",
                       columns, routed, slots, stage_ms, table_ms, gpu_ms, wall_ms,
                       columns ? wall_ms / columns : 0.0);
}

std::vector<uint32_t> GpuMoeBridge::union_experts(const BatchCall& call) const {
    std::vector<uint32_t> out;
    if (!call.ids || call.m == 0 || call.m > kMoeBatchMax || call.topk == 0) return out;
    out.reserve(size_t(call.m) * call.topk);
    for (uint32_t m = 0; m < call.m; ++m)
        for (uint32_t s = 0; s < call.topk; ++s) {
            const uint32_t e = call.ids[size_t(m) * call.topk + s];
            if (std::find(out.begin(), out.end(), e) == out.end()) out.push_back(e);
        }
    return out;
}

Result<void> GpuMoeBridge::stage_batch_union(const BatchCall& call) {
    if (!store_ || !planner_) return fail(Err::FailedPrecondition, "MoE bridge is not created");
    if (call.m == 0 || call.m > kMoeBatchMax)
        return fail(Err::InvalidArgument,
                    std::format("a verify batch of {} columns; the interface takes 1..{}",
                                call.m, kMoeBatchMax));
    const uint32_t dim   = call.hidden;
    const uint32_t slots = union_runner_.dims().slots;
    const TimePoint t0 = Clock::now();
    const uint32_t prev_routed = union_info_.routed;
    union_info_ = UnionInfo{};
    union_info_.columns = call.m;

    // 1. The activations, once for the whole batch, column m into column m --
    //    the same contract `stage_batch` has, and the reason the union pays one
    //    act_quant for six tokens instead of six.
    for (uint32_t m = 0; m < call.m; ++m) {
        const float* xin = call.x + size_t(m) * dim;
        float*    xf = xf_batch_.data() + size_t(m) * dim;
        uint16_t* xq = xq_batch_.data() + size_t(m) * dim;
        std::memcpy(xf, xin, size_t(dim) * sizeof(float));
        act_quant_to_fp16(xf, xq, xf, dim);
        std::memcpy(union_runner_.x_fp16() + size_t(m) * dim, xq,
                    size_t(dim) * sizeof(uint16_t));
    }
    const TimePoint t1 = Clock::now();

    // 2. The union. `union_slot_of_` is an expert-id -> slot scratch cleared by
    //    walking the ids we set, not by clearing 385 entries a layer.
    for (uint32_t u = 0; u < prev_routed; ++u)
        if (union_ids_[u] < union_slot_of_.size()) union_slot_of_[union_ids_[u]] = ~0u;
    uint32_t n_routed = 0;
    for (uint32_t m = 0; m < call.m; ++m)
        for (uint32_t s = 0; s < call.topk; ++s) {
            const uint32_t e = call.ids[size_t(m) * call.topk + s];
            if (e >= union_slot_of_.size())
                return fail(Err::InvalidArgument,
                            std::format("expert {} is out of range for layer {}", e, call.layer));
            if (union_slot_of_[e] != ~0u) continue;
            if (n_routed + 1 >= slots)
                return fail(Err::InvalidArgument,
                            std::format("the batch's expert union is larger than the union "
                                        "runner's {} slots", slots));
            union_slot_of_[e] = n_routed;
            union_ids_[n_routed] = e;
            ++n_routed;
        }
    union_info_.routed = n_routed;
    union_info_.slots  = n_routed + 1;

    // 3. The weight matrix. Dense `[m][slots]`, zero wherever token m does not
    //    route to that slot's expert: dispatch A multiplies h by this, so a zero
    //    makes the slot contribute exactly +0.0 to that column in dispatch B.
    float* w = union_runner_.route_weights();
    std::fill(w, w + size_t(call.m) * slots, 0.0f);
    for (uint32_t m = 0; m < call.m; ++m)
        for (uint32_t s = 0; s < call.topk; ++s) {
            const uint32_t u = union_slot_of_[call.ids[size_t(m) * call.topk + s]];
            w[size_t(m) * slots + u] = call.weights[size_t(m) * call.topk + s];
        }
    for (uint32_t m = 0; m < call.m; ++m) w[size_t(m) * slots + n_routed] = 1.0f;

    // 4. The slot table and the expert pointer rows. The shared expert is the
    //    last live slot, exactly as at M = 1.
    uint32_t* ids  = union_runner_.ids();
    uint32_t* list = union_runner_.slot_list();
    uint64_t* table = union_runner_.pointer_table();
    uint64_t row[kExpertPartCount];
    for (uint32_t u = 0; u < n_routed; ++u) {
        ids[u]  = union_ids_[u];
        list[u] = u;
        const ExpertKey key{static_cast<uint16_t>(call.layer),
                            static_cast<uint16_t>(union_ids_[u])};
        if (auto r = store_->table_row(key, row); !r)
            return fail(r.error().code,
                        std::format("at the verify batch's expert union (expert {}): {}",
                                    union_ids_[u], r.error().message));
        std::memcpy(table + size_t(union_ids_[u]) * kExpertPartCount, row, sizeof row);
    }
    if (auto r = bind_shared_into(call.layer, table); !r) return r;
    ids[n_routed]  = shared_index_ | gpu::kSlotFp8;
    list[n_routed] = n_routed;
    union_runner_.set_list_count(n_routed + 1);
    union_runner_.set_accumulate(false);
    union_runner_.set_live_columns(call.m);
    const TimePoint t2 = Clock::now();
    union_info_.stage_ms = ms_between(t0, t1);
    union_info_.table_ms = ms_between(t1, t2);
    return {};
}

Result<void> GpuMoeBridge::record_batch_union(gpu::CommandBuffer& cmd) {
    return union_runner_.record_into(cmd, gpu::MoePhase::Both);
}

Result<void> GpuMoeBridge::run_batch_union(const BatchCall& call) {
    const TimePoint t_all = Clock::now();
    if (auto r = stage_batch_union(call); !r) return r;
    const TimePoint t0 = Clock::now();
    auto rt = union_runner_.run(1);
    if (!rt) return std::unexpected(rt.error());
    union_info_.gpu_ms = rt->seconds_total * 1e3;
    if (call.y) {
        const uint32_t dim = call.hidden;
        std::memcpy(call.y, union_runner_.y(), size_t(call.m) * dim * sizeof(float));
    }
    (void)t0;
    union_info_.wall_ms = ms_between(t_all, Clock::now());
    return {};
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
