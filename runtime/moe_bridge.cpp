#include "core/env.h"
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

namespace cachedmoe::runtime {

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

// Track BF: reading x back out of GPU-visible memory.
//
// `call.x` points into a device-local, host-visible allocation: on this APU
// that mapping is UNCACHED WRITE-COMBINING, and an ordinary load from WC memory
// is not allowed to use the cache hierarchy, so `std::memcpy` walks it at a few
// hundred MB/s. docs/p2_decode.md §3.3 already knew to do ONE memcpy rather
// than 5,120 scalar loads; what it did not do is use the instruction the ISA
// provides for exactly this, `vmovntdqa` (`_mm256_stream_load_si256`), which
// pulls a whole WC line into a fill buffer instead of issuing an uncached read
// per access. The M = 1 path pays this once for 20 KB a layer; a verify batch
// pays it M times, which the Track BF trace found to be 33.3 of the 34.7 ms of
// host time an M=5 batch spends between its attention fence and its MoE
// dispatch -- the biggest single item of GPU-idle gap in the batch.
//
// CACHEDMOE_MOE_WC_READ=0 goes back to plain memcpy (the A/B).
void wc_read(void* dst, const void* src, size_t bytes) {
    static const bool on = [] {
        const char* e = ::cachedmoe::environment::get("CACHEDMOE_MOE_WC_READ");
        return !(e && *e == '0');
    }();
    // Only the SOURCE has to be 32-byte aligned: `vmovntdqa` is the load, and
    // the store below is an unaligned one into ordinary cached memory.
    const uintptr_t sa = reinterpret_cast<uintptr_t>(src);
    if (!on || (sa & 31u) || bytes < 64) {
        static bool said = false;
        if (on && !said) {
            said = true;
            log_info("moe bridge: x read falls back to memcpy (src {:#x}, {} B)", sa, bytes);
        }
        std::memcpy(dst, src, bytes);
        return;
    }
    const auto* s32 = reinterpret_cast<const __m256i*>(src);
    auto*       d32 = reinterpret_cast<__m256i*>(dst);
    const size_t n = bytes / 32;
    for (size_t i = 0; i < n; ++i) _mm256_storeu_si256(d32 + i, _mm256_stream_load_si256(s32 + i));
    // `vmovntdqa` on WC memory is weakly ordered against everything else.
    _mm_mfence();
    const size_t tail = bytes - n * 32;
    if (tail)
        std::memcpy(static_cast<std::byte*>(dst) + n * 32,
                    static_cast<const std::byte*>(src) + n * 32, tail);
}

// Track BF: the union runner's shape knobs, so one binary can A/B them.
uint32_t env_u32(const char* name, uint32_t dflt) {
    const char* e = ::cachedmoe::environment::get(name);
    if (!e || !*e) return dflt;
    char* end = nullptr;
    const unsigned long v = std::strtoul(e, &end, 10);
    return (end && end != e) ? static_cast<uint32_t>(v) : dflt;
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
    // Experiment knobs, the same shape as the union runner's below, so the
    // decode shape can be re-swept per driver instead of re-derived.
    spec.lanes_per_row = env_u32("CACHEDMOE_MOE_L", spec.lanes_per_row);
    spec.rows_per_lane = env_u32("CACHEDMOE_MOE_R", spec.rows_per_lane);
    spec.x_mode        = env_u32("CACHEDMOE_MOE_XMODE", spec.x_mode);
    spec.x_mode_b      = env_u32("CACHEDMOE_MOE_XMODE_B", spec.x_mode_b);
    // The FP4 decode is compiler-specific. On the AMD proprietary driver the
    // constant table (DecodeMode 0) is the measured M = 1 champion; Mesa's ACO
    // (RADV) lowers the same `kE2M1[nib]` into a per-element branchy select
    // tree, which held the whole MoE to 53 ms/token against 30.9 for the
    // arithmetic decode (DecodeMode 1). DecodeMode 3 places the nibble straight
    // into fp16 bits (moe_common.slang fp4_pair_bits): the driver's own
    // statistics put DecodeMode 1's gate/up loop at 840 instructions a block,
    // 700 of them decode (bench/results/linux/isa/), and kernel_bench's engine
    // shape goes 0.848 -> 0.710 ms per 7-slot pair (A 191 -> 212 GB/s, B 161
    // -> 203). Every product and partial sum is the old one times 2^-14, which
    // the block ldexp undoes exactly, so l3_ppl off is bit-identical (NLL
    // 0.621814, 59/64, both ways). See docs/build.md, "Linux".
    if (bc.decode_mode == 0 && device.caps().driver_id == VK_DRIVER_ID_MESA_RADV)
        spec.decode_mode = 3;
    spec.decode_mode   = env_u32("CACHEDMOE_MOE_DEC", spec.decode_mode);
    // Dispatch B (w2) gets its own shape on RADV. With A at the decode champion
    // L32 R1, ACO's B reads w2 at ~152 GB/s; B alone at L16 R2 reads it at ~185
    // (kernel_bench "fp8 dec1 B": 0.807 -> 0.757 ms per 7-slot pair). Engine
    // hot step, four alternating pairs: ~79.9 -> ~76.7 ms (-3.9%), moe gpu
    // 32.3 -> 30.0. Numerically it is a different fp32 reduction width only:
    // the L1 golden is unchanged (1.37e-4 of |y|max, cos 0.999999961), but
    // l3_ppl off moves 0.601884 -> 0.623007 (top-1 62 -> 60/64) because two
    // near-tie positions flip -- PPL 1.021x off, inside the harness's 1.05x
    // bar; L16 R4 lands on the identical 0.623007 and L32 R2 on the identical
    // 0.601884, i.e. the number follows the lane count and nothing else.
    // STATUS §7 0h. CACHEDMOE_MOE_LB / CACHEDMOE_MOE_RB override (LB=32 RB=1 is
    // the old shape).
    if (spec.lanes_b == 0 && spec.rows_b == 0 && spec.lanes_per_row == 32 &&
        spec.rows_per_lane == 1 && device.caps().driver_id == VK_DRIVER_ID_MESA_RADV) {
        spec.lanes_b = 16;
        spec.rows_b  = 2;
    }
    spec.lanes_b       = env_u32("CACHEDMOE_MOE_LB", spec.lanes_b);
    spec.rows_b        = env_u32("CACHEDMOE_MOE_RB", spec.rows_b);
    // An experiment knob, not a setting: docs/p2_decode.md §8.2 uses it to
    // A/B the h quantisation's placement for bit-reproducibility.
    if (const char* e = ::cachedmoe::environment::get("CACHEDMOE_MOE_HQUANT"); e && *e)
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
    d.table_layers      = 2;   // two pages: Track SE alternates them by layer parity
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
    du.table_layers = 1;
    du.slots = kUnionSlotsMax;
    if (cfg.num_experts_per_tok * kMoeBatchMax + 1 < du.slots)
        du.slots = cfg.num_experts_per_tok * kMoeBatchMax + 1;
    // Track BF: the union runner's kernel shape -- MEASURED, and the answer is
    // "the decode champion, unchanged". docs/kernel_p2_moe.md §3.5 says the
    // winner moves at M >= 4 (`L16 R2 xgf16`, 149.7 GB/s at M=5 against 138.5
    // for `L16 R2 xglob`), so the obvious change here was to give the union
    // runner that shape. It LOSES in the engine: a full `bench_spec` A/B on a
    // warm, all-resident cache is 68.2 -> 72.9 ms/position (+6.9%), and a
    // per-knob sweep puts every alternative behind `L32 R1 xglob`. See
    // docs/p4_dspark_runtime.md §8.2 -- the kernel bench's per-M table is for a
    // SEVEN-slot dispatch, and the union is ~20 slots, which is a different
    // occupancy/working-set point. The knobs stay as env overrides so the next
    // person can re-run the sweep instead of re-deriving it.
    gpu::MoeSpec uspec = spec;
    uspec.lanes_per_row = env_u32("CACHEDMOE_MOE_UNION_L", spec.lanes_per_row);
    uspec.rows_per_lane = env_u32("CACHEDMOE_MOE_UNION_R", spec.rows_per_lane);
    uspec.x_mode        = env_u32("CACHEDMOE_MOE_UNION_XMODE", spec.x_mode);
    // Dispatch B's own shape is a 7-slot decode measurement; the union (~20
    // slots) was never measured with it, so it keeps following A.
    uspec.lanes_b       = env_u32("CACHEDMOE_MOE_UNION_LB", 0);
    uspec.rows_b        = env_u32("CACHEDMOE_MOE_UNION_RB", 0);
    if (auto r = union_runner_.create(device, alloc, shader_dir, uspec, du); !r) return r;
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
    return bind_shared_into(layer, page_table());
}

uint32_t GpuMoeBridge::debug_check_x(const MoeCall& call, std::string* first) {
    const uint32_t dim = call.hidden;
    std::vector<float> xf(dim), scratch(dim);
    std::vector<uint16_t> q(dim);
    wc_read(xf.data(), call.x, size_t(dim) * sizeof(float));
    act_quant_to_fp16(xf.data(), q.data(), scratch.data(), dim);
    const uint16_t* g = runner_.x_fp16();
    uint32_t bad = 0;
    for (uint32_t i = 0; i < dim; ++i) {
        if (g[i] == q[i]) continue;
        if (bad++ == 0 && first)
            *first = std::format("elem {} x={:.9g} host {:#06x} gpu {:#06x}", i, xf[i], q[i], g[i]);
    }
    return bad;
}

Result<void> GpuMoeBridge::record_shared_early(gpu::CommandBuffer& cmd, uint32_t layer,
                                               uint64_t x_addr) {
    if (!store_ || !planner_) return fail(Err::FailedPrecondition, "MoE bridge is not created");
    set_page(layer & 1u);
    if (auto r = bind_shared(layer); !r) return r;
    // The shared slot's id and route weight never change, but dispatch A reads
    // both (h = swiglu * RouteW[slot]) and this runs BEFORE this layer's
    // stage_input -- on the process's first layer the weight was still the
    // zero the runner was created with, which zeroed that layer's shared
    // expert (found by l3_ppl: 0.646705 against 0.623007).
    const uint32_t slots = runner_.dims().slots;
    runner_.ids()[slots - 1] = shared_index_ | gpu::kSlotFp8;
    runner_.route_weights()[slots - 1] = 1.0f;
    return runner_.record_shared_early(cmd, x_addr);
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
        const std::string pre = layer < layout::kNumLayers
                ? std::format("layers.{}.ffn.shared_experts", layer)
                : std::format("mtp.{}.ffn.shared_experts", layer - layout::kNumLayers);
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
    uint64_t* table = page_table();
    for (uint32_t s : slots) {
        if (s >= call.topk) return fail(Err::InvalidArgument, "stage_rows: not a routed slot");
        if (call.weights[s] == 0.0f) continue;
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

Result<void> GpuMoeBridge::stage_input(const MoeCall& call, bool x_on_gpu) {
    if (!store_ || !planner_) return fail(Err::FailedPrecondition, "MoE bridge is not created");
    const uint32_t dim = call.hidden;
    const uint32_t slots = runner_.dims().slots;
    if (call.topk + 1 != slots)
        return fail(Err::InvalidArgument,
                    std::format("{} routed experts, but the runner has {} slots for "
                                "routed + shared", call.topk, slots));
    timing_ = Timing{};
    const TimePoint t0 = Clock::now();

    if (!x_on_gpu) {
        // Copy x out of GPU-visible memory BEFORE computing over it: design
        // §3.3's uncached write-combining read, one memcpy instead of 5,120
        // loads (docs/p2_decode.md §3.3).
        wc_read(xf_.data(), call.x, size_t(dim) * sizeof(float));
    }
    const TimePoint t1 = Clock::now();

    if (!x_on_gpu) {
        // `act_quant(x, 32, ue8m0)` once for the whole layer, straight into the
        // runner's fp16 input. The scratch is the runner-independent host copy;
        // quantising in place over it is safe because each block reads its 32
        // values before writing them.
        act_quant_to_fp16(xf_.data(), xq_.data(), xf_.data(), dim);
        std::memcpy(runner_.x_fp16(), xq_.data(), size_t(dim) * sizeof(uint16_t));
    }
    const TimePoint t2 = Clock::now();

    // The routed half's table rows are `stage_rows`'s: design §7.1's residency
    // gate decides when each one may be written. With x on the GPU the shared
    // row is already in this page (record_shared_early).
    if (!x_on_gpu)
        if (auto r = bind_shared(call.layer); !r) return r;

    uint32_t ids[16];
    uint32_t list[16];
    float    w[16];
    uint32_t live = 0;
    for (uint32_t s = 0; s < call.topk; ++s) {
        ids[s]  = call.ids[s];
        if (call.weights[s] != 0.0f) list[live++] = s;
        w[s]    = call.weights[s];
    }
    // The shared expert: not routed, weight 1, fp8.
    ids[call.topk]  = shared_index_ | gpu::kSlotFp8;
    list[live++] = call.topk;
    w[call.topk]    = 1.0f;
    std::memcpy(runner_.ids(), ids, slots * sizeof(uint32_t));
    std::memcpy(runner_.slot_list(), list, live * sizeof(uint32_t));
    std::memcpy(runner_.route_weights(), w, slots * sizeof(float));
    runner_.set_list_count(live);
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
    reset_page();
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
        wc_read(xf, xin, size_t(dim) * sizeof(float));
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
            if (call.weights && call.weights[size_t(m) * call.topk + s] == 0.0f) continue;
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
        wc_read(xf, xin, size_t(dim) * sizeof(float));
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
            if (call.weights && call.weights[size_t(m) * call.topk + s] == 0.0f) continue;
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
            if (call.weights[size_t(m) * call.topk + s] == 0.0f) continue;
            const uint32_t u = union_slot_of_[call.ids[size_t(m) * call.topk + s]];
            // `+=`, not `=`: the gate's own top-k has no duplicates, but Track Y's
            // resident-only routing fills a skipped slot with a BORROWED resident
            // id at weight 0 (runtime/resident_route.h), so one column can name the
            // same expert twice. At M = 1 those are two runner slots and the sum is
            // the kernel's; collapsed onto one union slot, an assignment would let
            // the borrowed 0 clobber a real weight.
            w[size_t(m) * slots + u] += call.weights[size_t(m) * call.topk + s];
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
    reset_page();
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

std::vector<uint64_t> GpuMoeBridge::snapshot_with_shared(std::span<const uint64_t> table,
                                                         uint32_t layers) const {
    constexpr auto words = layout::kExpertAddressWords;
    const uint32_t routed = shared_index_, experts = routed + 1;
    std::vector<uint64_t> out(size_t(layers) * experts * words);
    // Copy routed experts from the guarded snapshot. The shared expert is
    // pinned separately and occupies the last row in each layer's GPU table.
    for (uint32_t l = 0; l < layers; ++l) {
        std::memcpy(out.data() + size_t(l) * experts * words,
                    table.data() + size_t(l) * routed * words, routed * words * sizeof(uint64_t));
        const auto prefix = std::format("layers.{}.ffn.shared_experts.", l);
        for (uint32_t j = 0; j < words / 2; ++j) {
            const auto *tensor = pinned_->find(prefix + std::format("w{}.weight", j + 1));
            const auto row = (size_t(l) * experts + routed) * words + j * 2;
            out[row] = tensor ? tensor->data : 0;
            out[row + 1] = tensor ? tensor->scale : 0;
        }
    }
    return out;
}

} // namespace cachedmoe::runtime
