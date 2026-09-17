#include "gpu/vulkan/moe_kernels.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>
#include <format>

#include "core/profiler.h"

namespace deepmoe::gpu {

std::string MoeSpec::name() const {
    static const char* kXMode[] = {"glob", "lds", "ldsf16", "ldsi8", "gf16", "gi8", "prei8"};
    static const char* kHQuant[] = {"", " hqB", " hq8", " hqP"};
    std::string s = std::format("M{} L{} R{} sg{} dec{} h{} x{}", m, lanes_per_row, rows_per_lane,
                                subgroup_size ? std::to_string(subgroup_size) : std::string("auto"),
                                decode_mode, h_precision ? "fp32" : "fp16",
                                kXMode[x_mode < 7 ? x_mode : 0]);
    if (b_mode() != x_mode) s += std::format("/{}", kXMode[b_mode() < 7 ? b_mode() : 0]);
    if (h_quant)   s += kHQuant[h_quant < 4 ? h_quant : 0];
    if (fp8_slots) s += " fp8";
    return s;
}

std::string default_shader_dir() {
#if defined(_MSC_VER)
    size_t n = 0; char* buf = nullptr;
    if (_dupenv_s(&buf, &n, "DEEPMOE_SHADER_DIR") == 0 && buf) {
        std::string v(buf); free(buf); return v;
    }
#else
    if (const char* e = std::getenv("DEEPMOE_SHADER_DIR")) return e;
#endif
#if defined(DEEPMOE_SHADER_DIR)
    return DEEPMOE_SHADER_DIR;
#else
    return "build/shaders";
#endif
}

uint64_t MoeRunner::bytes_dispatch_a() const {
    // w1 and w3: packed FP4 rows plus their E8M0 scales, per computed expert.
    // An fp8 slot reads a whole byte per element and a 32x32-tiled scale plane.
    const uint64_t elems   = uint64_t(dims_.inter) * dims_.hidden;
    const uint64_t fp4_mat = elems / 2 + elems / layout::kFp4ScaleBlock;
    const uint64_t fp8_mat = elems + elems / (layout::kFp8ScaleBlockM * layout::kFp8ScaleBlockK);
    const uint32_t fp8 = dims_.fp8_slot_count < list_count_ ? dims_.fp8_slot_count : list_count_;
    return 2 * (fp4_mat * (list_count_ - fp8) + fp8_mat * fp8);
}

uint64_t MoeRunner::bytes_dispatch_b() const {
    const uint64_t elems   = uint64_t(dims_.hidden) * dims_.inter;
    const uint64_t fp4_mat = elems / 2 + elems / layout::kFp4ScaleBlock;
    const uint64_t fp8_mat = elems + elems / (layout::kFp8ScaleBlockM * layout::kFp8ScaleBlockK);
    const uint32_t fp8 = dims_.fp8_slot_count < list_count_ ? dims_.fp8_slot_count : list_count_;
    return fp4_mat * (list_count_ - fp8) + fp8_mat * fp8;
}

uint64_t MoeRunner::bytes_per_iteration() const {
    return bytes_dispatch_a() + bytes_dispatch_b();
}

void MoeRunner::set_list_count(uint32_t n) { list_count_ = n; }

uint64_t* MoeRunner::pointer_table() { return static_cast<uint64_t*>(table_.host_ptr); }
size_t    MoeRunner::pointer_table_entries() const {
    return size_t(dims_.table_layers) * dims_.experts_per_layer * layout::kExpertParts;
}
uint32_t* MoeRunner::ids()           { return static_cast<uint32_t*>(ids_.host_ptr); }
uint32_t* MoeRunner::slot_list()     { return static_cast<uint32_t*>(list_.host_ptr); }
float*    MoeRunner::route_weights() { return static_cast<float*>(routew_.host_ptr); }
uint16_t* MoeRunner::x_fp16()        { return static_cast<uint16_t*>(x_.host_ptr); }
float*    MoeRunner::y()             { return static_cast<float*>(y_.host_ptr); }
void*     MoeRunner::h()             { return h_.host_ptr; }

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> MoeRunner::create(Device&, MemoryAllocator&, const std::string&,
                               const MoeSpec&, const MoeDims&) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void MoeRunner::destroy() {}
Result<void> MoeRunner::record(uint32_t, MoePhase) { return fail(Err::Unavailable, "no vulkan"); }
Result<MoeTiming> MoeRunner::run(uint32_t, MoePhase) { return fail(Err::Unavailable, "no vulkan"); }
Result<void> MoeRunner::record_into(CommandBuffer&, MoePhase) { return fail(Err::Unavailable, "no vulkan"); }
uint32_t* MoeRunner::slot_list_alt() { return nullptr; }
Result<void> MoeRunner::record_gateup_alt(CommandBuffer&, uint32_t) { return fail(Err::Unavailable, "no vulkan"); }

#else

namespace {

// design §7.9 dispatch A push constants; mirrors GateUpPush in the shader.
struct GateUpPush {
    uint32_t layer, experts_per_layer, num_slots, n_rows, k;
    float    swiglu_limit;
    // Track T / DSpark: live activation columns. `M` is the specialisation
    // constant (what the shader's arrays are sized for), this is what the
    // dispatch computes -- 1 for a decode token, the batch size for a verify
    // batch. It is the tail word so the first six keep their offsets, and every
    // shader loop is `for (m = 0; m < pc.m; ++m)`.
    uint32_t m = 1;
};
// design §7.9 dispatch B; mirrors DownPush.
struct DownPush {
    uint32_t layer, experts_per_layer, num_slots, list_count, n_rows, k, flags;
    uint32_t m = 1;
};
// The two tiny pre/post passes; mirror HQuantPush and XQuantPush.
struct HQuantPush {
    uint32_t num_slots, list_count, k;
};
struct XQuantPush {
    uint32_t k;
};

}  // namespace

Result<void> MoeRunner::create(Device& device, MemoryAllocator& alloc,
                               const std::string& shader_dir,
                               const MoeSpec& spec, const MoeDims& dims) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    if (spec.m == 0 || spec.m > 6) return fail(Err::InvalidArgument, "M must be 1..6 (design §1.2)");
    if (spec.lanes_per_row != 16 && spec.lanes_per_row != 32 && spec.lanes_per_row != 64)
        return fail(Err::InvalidArgument, "lanes_per_row must be 16, 32 or 64 (design §7.1 rule 3)");
    // Powers of two only: rows_per_lane is what divides the activation traffic,
    // and design §7.9's measured M=6 bottleneck is exactly that traffic
    // (docs/kernel_p2_moe.md §3). 16 is the largest that still divides 2304.
    if (spec.rows_per_lane == 0 || spec.rows_per_lane > 16 ||
        (spec.rows_per_lane & (spec.rows_per_lane - 1)) != 0)
        return fail(Err::InvalidArgument, "rows_per_lane must be a power of two, 1..16");
    const uint32_t rows_per_group = (256 / spec.lanes_per_row) * spec.rows_per_lane;
    if (dims.inter % rows_per_group || dims.hidden % rows_per_group)
        return fail(Err::InvalidArgument, "row count must divide by the workgroup's rows");
    if (spec.x_mode > 6 || spec.b_mode() > 5)
        return fail(Err::InvalidArgument,
                    "x_mode must be 0..6 and dispatch B's 0..5 (mode 6 is about x)");
    if (spec.h_quant > 3) return fail(Err::InvalidArgument, "h_quant must be 0..3");
    if (spec.h_quant == 3 && spec.h_precision)
        return fail(Err::InvalidArgument,
                    "h_quant=3 quantises the fp16 h dispatch A wrote; h_precision must be 0");
    // The int8 x plane and its scales are appended to the x allocation at word
    // granularity, and moe_xquant writes one block per thread.
    if (spec.x_mode == 6 && dims.hidden % (4 * layout::kFp4ScaleBlock))
        return fail(Err::InvalidArgument, "x_mode=6 needs hidden to be a multiple of 128");
    if ((spec.x_mode == 3 || spec.b_mode() == 3) && spec.m * spec.lanes_per_row > 256)
        return fail(Err::InvalidArgument,
                    "the int8 x path stages one (column, block) per thread: M * lanes <= 256");
    if (spec.x_mode >= 1 && spec.x_mode <= 3) {
        // The LDS tile is M x LanesPerRow x 32 activations, plus the 1 KiB
        // reduction scratch and the optional 1 KiB fp8 decode table.
        const uint64_t tile = uint64_t(spec.m) * spec.lanes_per_row *
                              (spec.x_mode == 3 ? 32u : 64u);
        const uint64_t lds  = tile + 1024 + (spec.fp8_slots || spec.h_quant == 2 ? 1024 : 0)
                            + (spec.h_quant == 2 ? uint64_t(spec.m) * rows_per_group * 4 : 0);
        const uint64_t cap  = device.caps().max_compute_shared_memory ? device.caps().max_compute_shared_memory : 32768u;
        if (lds > cap)
            return fail(Err::InvalidArgument,
                        std::format("the x tile needs {} B of LDS, the device allows {}", lds, cap));
    }
    if (spec.h_quant == 2 && rows_per_group % layout::kFp8ScaleBlockK)
        return fail(Err::InvalidArgument,
                    std::format("h_quant=2 writes whole fp8 blocks, so a workgroup must own a "
                                "multiple of 32 rows; this one owns {}", rows_per_group));
    device_ = &device;
    alloc_  = &alloc;
    spec_   = spec;
    dims_   = dims;
    list_count_ = dims.slots;

    PipelineSpec ps;
    ps.m             = spec.m;
    ps.lanes_per_row = spec.lanes_per_row;
    ps.rows_per_wg   = 256 / spec.lanes_per_row;
    ps.subgroup_size = spec.subgroup_size;
    ps.extra = {spec.decode_mode, spec.h_precision, spec.rows_per_lane,
                spec.x_mode, spec.h_quant, spec.fp8_slots};
    PipelineSpec ps_b = ps;
    ps_b.extra[3] = spec.b_mode();

    PipelineLayoutSpec la;
    la.storage_buffers   = 9;   // + the raw-word aliases of h (§7.9 v0.6) and x (XMode 6)
    la.push_constant_size = sizeof(GateUpPush);
    if (auto r = gateup_.create(device, shader_dir + "/moe_gateup.spv", la, ps); !r) {
        destroy(); return r;
    }
    PipelineLayoutSpec lb;
    lb.storage_buffers   = 5;
    lb.push_constant_size = sizeof(DownPush);
    if (auto r = down_.create(device, shader_dir + "/moe_down.spv", lb, ps_b); !r) {
        destroy(); return r;
    }
    if (spec.h_quant == 3) {
        PipelineLayoutSpec lh;
        lh.storage_buffers   = 2;
        lh.push_constant_size = sizeof(HQuantPush);
        if (auto r = hquant_.create(device, shader_dir + "/moe_hquant.spv", lh, ps); !r) {
            destroy(); return r;
        }
    }
    if (spec.x_mode == 6) {
        PipelineLayoutSpec lx;
        lx.storage_buffers   = 1;
        lx.push_constant_size = sizeof(XQuantPush);
        if (auto r = xquant_.create(device, shader_dir + "/moe_xquant.spv", lx, ps); !r) {
            destroy(); return r;
        }
    }
    if (auto r = descriptors_.create(device, 8, 64); !r) { destroy(); return r; }

    const uint64_t h_elem = spec.h_precision ? 4 : 2;
    // h_quant 3 cannot quantise in place (gpu/shaders/moe_hquant.slang), so the
    // fp8 plane and its UE8M0 scales sit past the fp16 h: 3.125 B/element.
    // x_mode 6 appends the same way: fp16 x, then int8 x, then one fp32 scale
    // per 32 elements = 3.125 B/element as well.
    const uint64_t h_elems = uint64_t(spec.m) * dims.slots * dims.inter;
    const uint64_t h_bytes = spec.h_quant == 3
        ? h_elems * 3 + h_elems / layout::kFp4ScaleBlock * 4
        : h_elems * h_elem;
    const uint64_t x_elems = uint64_t(spec.m) * dims.hidden;
    const uint64_t x_bytes = spec.x_mode == 6
        ? x_elems * 3 + x_elems / layout::kFp4ScaleBlock * 4
        : x_elems * 2;
    struct { GpuBuffer* b; uint64_t bytes; } bufs[] = {
        {&table_,  uint64_t(pointer_table_entries()) * sizeof(uint64_t)},
        {&ids_,    uint64_t(dims.slots) * sizeof(uint32_t)},
        {&list_,   uint64_t(dims.slots) * sizeof(uint32_t)},
        {&routew_, uint64_t(spec.m) * dims.slots * sizeof(float)},
        {&x_,      x_bytes},
        {&h_,      h_bytes},
        {&y_,      uint64_t(spec.m) * dims.hidden * sizeof(float)},
        {&list_alt_, uint64_t(dims.slots) * sizeof(uint32_t)},
    };
    for (auto& e : bufs) {
        // `y` alone is device-addressable, so a caller can read the MoE output
        // from its next dispatch without a host copy (record_into).
        auto b = alloc.allocate(e.bytes, /*host_visible=*/true,
                                /*device_address=*/e.b == &y_);
        if (!b) { destroy(); return std::unexpected(b.error()); }
        *e.b = *b;
        std::memset(e.b->host_ptr, 0, static_cast<size_t>(e.bytes));
    }

    std::vector<BufferBinding> ba(9);
    ba[0] = {0, 0, 0, table_.buffer};
    ba[1] = {1, 0, 0, ids_.buffer};
    ba[2] = {2, 0, 0, list_.buffer};
    ba[3] = {3, 0, 0, routew_.buffer};
    ba[4] = {4, 0, 0, x_.buffer};
    ba[5] = {5, 0, 0, h_.buffer};
    ba[6] = {6, 0, 0, h_.buffer};        // H16, H32 and HU alias one allocation
    ba[7] = {7, 0, 0, h_.buffer};
    ba[8] = {8, 0, 0, x_.buffer};        // X and XU alias one allocation
    auto sa = descriptors_.allocate(gateup_, ba);
    if (!sa) { destroy(); return std::unexpected(sa.error()); }
    set_a_ = *sa;
    {
        std::vector<BufferBinding> alt = ba;
        alt[2] = {2, 0, 0, list_alt_.buffer};
        auto s2 = descriptors_.allocate(gateup_, alt);
        if (!s2) { destroy(); return std::unexpected(s2.error()); }
        set_a_alt_ = *s2;
    }

    std::vector<BufferBinding> bb(5);
    bb[0] = {0, 0, 0, table_.buffer};
    bb[1] = {1, 0, 0, ids_.buffer};
    bb[2] = {2, 0, 0, list_.buffer};
    bb[3] = {3, 0, 0, h_.buffer};
    bb[4] = {4, 0, 0, y_.buffer};
    auto sb = descriptors_.allocate(down_, bb);
    if (!sb) { destroy(); return std::unexpected(sb.error()); }
    set_b_ = *sb;

    if (hquant_.valid()) {
        std::vector<BufferBinding> bh(2);
        bh[0] = {0, 0, 0, list_.buffer};
        bh[1] = {1, 0, 0, h_.buffer};
        auto sh = descriptors_.allocate(hquant_, bh);
        if (!sh) { destroy(); return std::unexpected(sh.error()); }
        set_hq_ = *sh;
        bh[0] = {0, 0, 0, list_alt_.buffer};
        auto sh2 = descriptors_.allocate(hquant_, bh);
        if (!sh2) { destroy(); return std::unexpected(sh2.error()); }
        set_hq_alt_ = *sh2;
    }
    if (xquant_.valid()) {
        std::vector<BufferBinding> bx(1);
        bx[0] = {0, 0, 0, x_.buffer};
        auto sx = descriptors_.allocate(xquant_, bx);
        if (!sx) { destroy(); return std::unexpected(sx.error()); }
        set_xq_ = *sx;
    }

    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    auto cb = pool_.acquire();
    if (!cb) { destroy(); return std::unexpected(cb.error()); }
    cmd_ = *cb;
    // Four timestamps: start, after A, after B, end.
    (void)queries_.create(device, 4);
    return {};
}

void MoeRunner::destroy() {
    queries_.destroy();
    pool_.destroy();
    descriptors_.destroy();
    gateup_.destroy();
    down_.destroy();
    hquant_.destroy();
    xquant_.destroy();
    if (alloc_) {
        for (GpuBuffer* b : {&table_, &ids_, &list_, &routew_, &x_, &h_, &y_, &list_alt_})
            if (b->valid()) alloc_->free(*b);
    }
    table_ = ids_ = list_ = routew_ = x_ = h_ = y_ = list_alt_ = GpuBuffer{};
    set_a_ = set_b_ = set_hq_ = set_xq_ = set_a_alt_ = set_hq_alt_ = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_  = nullptr;
    recorded_ = 0;
}

Result<void> MoeRunner::record(uint32_t iterations, MoePhase phase) {
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter  / rows_per_wg;
    const uint32_t groups_b = dims_.hidden / rows_per_wg;

    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                  dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    DownPush   pb{dims_.layer, dims_.experts_per_layer, dims_.slots, list_count_,
                  dims_.hidden, dims_.inter, accumulate_ ? 1u : 0u, live_columns_};
    // One thread per 32-element block of the thing being quantised.
    HQuantPush ph{dims_.slots, list_count_, dims_.inter};
    XQuantPush px{dims_.hidden};
    const uint32_t hq_groups =
        (spec_.m * list_count_ * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
    const uint32_t xq_groups =
        (spec_.m * (dims_.hidden / layout::kFp4ScaleBlock) + 255) / 256;

    const bool run_a = phase != MoePhase::DownOnly;
    const bool run_b = phase != MoePhase::GateUpOnly;
    const bool timed = queries_.count() >= 4;
    bool first = true;

    if (auto r = cmd_.begin(); !r) return r;
    if (timed) {
        (void)cmd_.reset_queries(queries_, 0, 4);
        (void)cmd_.write_timestamp(queries_, 0, /*bottom=*/false);
        // Slots 1 and 2 bracket the first iteration's two dispatches; when a
        // phase is skipped they collapse onto their neighbour, so seconds_a or
        // seconds_b comes out zero rather than wrong.
        if (!run_a) (void)cmd_.write_timestamp(queries_, 1, true);
    }
    const uint32_t cycle = dims_.layer_cycle ? dims_.layer_cycle : 1;
    for (uint32_t it = 0; it < iterations; ++it) {
        pa.layer = pb.layer = dims_.layer + (it % cycle);
        if (run_a) {
            if (!first) { if (auto r = cmd_.barrier(); !r) return r; }
            // XMode 6: x becomes int8 + per-block scales before dispatch A
            // reads it. It is a per-*token* cost, not a per-layer one -- the
            // same x feeds all 40 MoE layers -- but it is recorded every
            // iteration here so the measured ms_a can only overstate it.
            if (xquant_.valid()) {
                if (auto r = cmd_.bind(xquant_, set_xq_); !r) return r;
                if (auto r = cmd_.push(xquant_, &px, sizeof(px)); !r) return r;
                if (auto r = cmd_.dispatch(xq_groups); !r) return r;
                if (auto r = cmd_.barrier(); !r) return r;
            }
            if (auto r = cmd_.bind(gateup_, set_a_); !r) return r;
            if (auto r = cmd_.push(gateup_, &pa, sizeof(pa)); !r) return r;
            if (auto r = cmd_.dispatch(groups_a, list_count_); !r) return r;
            // HQuant 3: the fp8 round trip of design §7.9 v0.6, as its own
            // dispatch, so dispatch A above is free to keep the workgroup shape
            // that is fastest for it (docs/kernel_p2_moe.md §8 item 1).
            if (hquant_.valid()) {
                if (auto r = cmd_.barrier(); !r) return r;
                if (auto r = cmd_.bind(hquant_, set_hq_); !r) return r;
                if (auto r = cmd_.push(hquant_, &ph, sizeof(ph)); !r) return r;
                if (auto r = cmd_.dispatch(hq_groups); !r) return r;
            }
            first = false;
            if (timed && it == 0) (void)cmd_.write_timestamp(queries_, 1, true);
        }
        if (run_b) {
            if (!first) { if (auto r = cmd_.barrier(); !r) return r; }
            if (auto r = cmd_.bind(down_, set_b_); !r) return r;
            if (auto r = cmd_.push(down_, &pb, sizeof(pb)); !r) return r;
            if (auto r = cmd_.dispatch(groups_b); !r) return r;
            first = false;
        }
        if (timed && it == 0) (void)cmd_.write_timestamp(queries_, 2, true);
    }
    if (timed) (void)cmd_.write_timestamp(queries_, 3, true);
    if (auto r = cmd_.end(); !r) return r;
    recorded_ = iterations;
    return {};
}

Result<void> MoeRunner::record_into(CommandBuffer& cmd, MoePhase phase) {
    if (!device_ || !gateup_.valid()) return fail(Err::FailedPrecondition, "runner is not created");
    if (list_count_ == 0 || list_count_ > dims_.slots)
        return fail(Err::InvalidArgument, "list_count must be 1..slots");
    // One iteration of `record`, minus the begin/end and the timestamps. Kept
    // as a separate function rather than a flag on `record` so the measured
    // path in bench/kernel_bench is byte-for-byte what it was.
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter  / rows_per_wg;
    const uint32_t groups_b = dims_.hidden / rows_per_wg;
    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                        dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    const DownPush   pb{dims_.layer, dims_.experts_per_layer, dims_.slots, list_count_,
                        dims_.hidden, dims_.inter, accumulate_ ? 1u : 0u, live_columns_};
    const HQuantPush ph{dims_.slots, list_count_, dims_.inter};
    const XQuantPush px{dims_.hidden};
    const uint32_t hq_groups =
        (spec_.m * list_count_ * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
    const uint32_t xq_groups =
        (spec_.m * (dims_.hidden / layout::kFp4ScaleBlock) + 255) / 256;

    if (phase != MoePhase::DownOnly) {
        if (xquant_.valid()) {
            if (auto r = cmd.bind(xquant_, set_xq_); !r) return r;
            if (auto r = cmd.push(xquant_, &px, sizeof(px)); !r) return r;
            if (auto r = cmd.dispatch(xq_groups); !r) return r;
            if (auto r = cmd.barrier(); !r) return r;
        }
        if (auto r = cmd.bind(gateup_, set_a_); !r) return r;
        if (auto r = cmd.push(gateup_, &pa, sizeof(pa)); !r) return r;
        if (auto r = cmd.dispatch(groups_a, list_count_); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
        if (hquant_.valid()) {
            if (auto r = cmd.bind(hquant_, set_hq_); !r) return r;
            if (auto r = cmd.push(hquant_, &ph, sizeof(ph)); !r) return r;
            if (auto r = cmd.dispatch(hq_groups); !r) return r;
            if (auto r = cmd.barrier(); !r) return r;
        }
    }
    if (phase != MoePhase::GateUpOnly) {
        if (auto r = cmd.bind(down_, set_b_); !r) return r;
        if (auto r = cmd.push(down_, &pb, sizeof(pb)); !r) return r;
        if (auto r = cmd.dispatch(groups_b); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
    }
    return {};
}

uint32_t* MoeRunner::slot_list_alt() { return static_cast<uint32_t*>(list_alt_.host_ptr); }

Result<void> MoeRunner::record_gateup_alt(CommandBuffer& cmd, uint32_t count) {
    if (!device_ || !gateup_.valid()) return fail(Err::FailedPrecondition, "runner is not created");
    if (count == 0 || count > dims_.slots) return fail(Err::InvalidArgument, "count must be 1..slots");
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter / rows_per_wg;
    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                        dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    const HQuantPush ph{dims_.slots, count, dims_.inter};
    const XQuantPush px{dims_.hidden};
    const uint32_t hq_groups =
        (spec_.m * count * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
    const uint32_t xq_groups =
        (spec_.m * (dims_.hidden / layout::kFp4ScaleBlock) + 255) / 256;
    if (xquant_.valid()) {
        if (auto r = cmd.bind(xquant_, set_xq_); !r) return r;
        if (auto r = cmd.push(xquant_, &px, sizeof(px)); !r) return r;
        if (auto r = cmd.dispatch(xq_groups); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
    }
    if (auto r = cmd.bind(gateup_, set_a_alt_); !r) return r;
    if (auto r = cmd.push(gateup_, &pa, sizeof(pa)); !r) return r;
    if (auto r = cmd.dispatch(groups_a, count); !r) return r;
    if (auto r = cmd.barrier(); !r) return r;
    if (hquant_.valid()) {
        if (auto r = cmd.bind(hquant_, set_hq_alt_); !r) return r;
        if (auto r = cmd.push(hquant_, &ph, sizeof(ph)); !r) return r;
        if (auto r = cmd.dispatch(hq_groups); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
    }
    return {};
}

Result<MoeTiming> MoeRunner::run(uint32_t iterations, MoePhase phase) {
    if (!device_ || !gateup_.valid()) return fail(Err::FailedPrecondition, "runner is not created");
    if (iterations == 0) return fail(Err::InvalidArgument, "iterations must be >= 1");
    if (list_count_ == 0 || list_count_ > dims_.slots)
        return fail(Err::InvalidArgument, "list_count must be 1..slots");
    const auto t_rec = Clock::now();
    if (auto r = record(iterations, phase); !r) return std::unexpected(r.error());
    const double record_s = std::chrono::duration<double>(Clock::now() - t_rec).count();

    const auto t0 = Clock::now();
    if (auto r = submit_and_wait(*device_, cmd_); !r) return std::unexpected(r.error());
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();

    MoeTiming t;
    t.iterations     = iterations;
    t.wall_seconds   = wall;
    t.record_seconds = record_s;
    t.seconds_total = wall / iterations;
    if (queries_.count() >= 4) {
        auto a = queries_.elapsed_seconds(0, 1);
        auto b = queries_.elapsed_seconds(1, 2);
        auto all = queries_.elapsed_seconds(0, 3);
        if (a && b && all) {
            t.seconds_a = *a;
            t.seconds_b = *b;
            t.seconds_total = *all / iterations;
            t.gpu_timed = true;
        }
    }
    return t;
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu
