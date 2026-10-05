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
    if (b_lanes() != lanes_per_row || b_rows() != rows_per_lane)
        s += std::format(" B:L{}R{}", b_lanes(), b_rows());
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
void MoeRunner::destroy() {
    gpu_descriptors_.destroy();gpu_copy_.destroy();gpu_mean_.destroy();gpu_route_.destroy();gpu_up_.destroy();gpu_down_.destroy();gpu_hq_.destroy();
    if(alloc_){for(auto* b:{&gpu_snapshot_,&gpu_args_,&gpu_indirect_})if(b->valid())alloc_->free(*b);}
    gpu_layers_=0;
}
Result<void> MoeRunner::record(uint32_t, MoePhase) { return fail(Err::Unavailable, "no vulkan"); }
Result<MoeTiming> MoeRunner::run(uint32_t, MoePhase) { return fail(Err::Unavailable, "no vulkan"); }
Result<void> MoeRunner::record_into(CommandBuffer&, MoePhase) { return fail(Err::Unavailable, "no vulkan"); }
uint32_t* MoeRunner::slot_list_alt() { return nullptr; }
Result<void> MoeRunner::record_gateup_alt(CommandBuffer&, uint32_t) { return fail(Err::Unavailable, "no vulkan"); }
Result<void> MoeRunner::record_shared_early(CommandBuffer&, uint64_t) { return fail(Err::Unavailable, "no vulkan"); }

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
struct XActPush {
    uint64_t src;
    uint32_t k;
    uint32_t pad;
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
    if (spec.b_lanes() != 16 && spec.b_lanes() != 32 && spec.b_lanes() != 64)
        return fail(Err::InvalidArgument, "dispatch B's lanes must be 16, 32 or 64");
    if (spec.b_rows() == 0 || spec.b_rows() > 16 || (spec.b_rows() & (spec.b_rows() - 1)) != 0)
        return fail(Err::InvalidArgument, "dispatch B's rows per lane must be a power of two, 1..16");
    const uint32_t rows_per_group_b = (256 / spec.b_lanes()) * spec.b_rows();
    if (dims.hidden % rows_per_group_b)
        return fail(Err::InvalidArgument, "hidden must divide by dispatch B's workgroup rows");
    if ((spec.b_lanes() != spec.lanes_per_row || spec.b_rows() != spec.rows_per_lane) &&
        (spec.h_quant == 1 || spec.h_quant == 2))
        return fail(Err::InvalidArgument,
                    "a separate dispatch-B shape needs h_quant 0 or 3 (1 and 2 tie A to B)");
    if (spec.b_mode() == 3 && spec.m * spec.b_lanes() > 256)
        return fail(Err::InvalidArgument,
                    "the int8 h path stages one (column, block) per thread: M * lanes <= 256");
    if (spec.b_mode() >= 1 && spec.b_mode() <= 3) {
        const uint64_t tile = uint64_t(spec.m) * spec.b_lanes() * (spec.b_mode() == 3 ? 32u : 64u);
        const uint64_t lds  = tile + 1024 + (spec.fp8_slots ? 1024 : 0);
        const uint64_t cap  = device.caps().max_compute_shared_memory ? device.caps().max_compute_shared_memory : 32768u;
        if (lds > cap)
            return fail(Err::InvalidArgument,
                        std::format("dispatch B's h tile needs {} B of LDS, the device allows {}", lds, cap));
    }
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
    // Track K1a: constant id 10. `pc.m` is opaque to the shader compiler, so
    // Track T's live-column mask turned all 26 column loops into dynamic-trip
    // loops even on the decode path, where the count is always 1 -- 6.5% of the
    // M = 1 kernel (docs/plan_p5.md §3(g) 8.7 / §3(h)). When this runner is
    // specialised on M == 1 the count is 1 for every dispatch it can ever
    // issue, so hand it to the pipeline as a specialisation constant and the
    // loops fold away again. M > 1 (the verify batch) keeps the mask.
    // DEEPMOE_MOE_STATIC_M1=0 is the A arm of the A/B; default on.
    bool m1_default = true;
    if (const char* e = std::getenv("DEEPMOE_MOE_STATIC_M1"); e && *e == '0') m1_default = false;
    const uint32_t static_m = (spec.m == 1 && m1_default) ? 1u : 0u;
    ps.extra = {spec.decode_mode, spec.h_precision, spec.rows_per_lane,
                spec.x_mode, spec.h_quant, spec.fp8_slots, static_m};
    PipelineSpec ps_b = ps;
    ps_b.extra[3] = spec.b_mode();
    ps_b.lanes_per_row = spec.b_lanes();
    ps_b.rows_per_wg   = 256 / spec.b_lanes();
    ps_b.extra[2]      = spec.b_rows();

    PipelineLayoutSpec la;
    la.storage_buffers   = 9;   // + the raw-word aliases of h (§7.9 v0.6) and x (XMode 6)
    la.push_constant_size = sizeof(GateUpPush);
    if (auto r = gateup_.create(device, shader_dir + "/moe_gateup.spv", la, ps); !r) {
        destroy(); return r;
    }
    PipelineLayoutSpec lb;
    lb.storage_buffers   = 6;
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
    if (spec.x_mode != 6) {
        PipelineLayoutSpec lx;
        lx.storage_buffers    = 1;
        lx.push_constant_size = sizeof(XActPush);
        if (auto r = xact_.create(device, shader_dir + "/moe_xact.spv", lx, ps); !r) {
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

    // Track K1a: the decode-shaped twin of the three dispatches above. The
    // engine's runner is specialised on M = kMoeBatchMax so that one runner can
    // also serve a verify batch, so every decode token ran the M = 6 kernel
    // with one live column: six accumulators a lane a row of register pressure
    // for one column of work, plus (since fb53514) a trip count the compiler
    // cannot fold. `live_columns_ == 1` now picks a pipeline that is M = 1 all
    // the way down. x_mode 6 is excluded because its int8 x plane offsets are
    // M-dependent and written by moe_xquant, which would have to agree too.
    if (spec.m > 1 && spec.x_mode != 6 && m1_default) {
        PipelineSpec ps1 = ps;
        ps1.m = 1;
        ps1.extra[6] = 1;               // StaticM
        PipelineSpec ps1_b = ps1;
        ps1_b.extra[3] = spec.b_mode();
        ps1_b.lanes_per_row = spec.b_lanes();
        ps1_b.rows_per_wg   = 256 / spec.b_lanes();
        ps1_b.extra[2]      = spec.b_rows();
        if (auto r = gateup_m1_.create(device, shader_dir + "/moe_gateup.spv", la, ps1); !r) {
            destroy(); return r;
        }
        if (auto r = down_m1_.create(device, shader_dir + "/moe_down.spv", lb, ps1_b); !r) {
            destroy(); return r;
        }
        if (spec.h_quant == 3) {
            PipelineLayoutSpec lh;
            lh.storage_buffers   = 2;
            lh.push_constant_size = sizeof(HQuantPush);
            if (auto r = hquant_m1_.create(device, shader_dir + "/moe_hquant.spv", lh, ps1); !r) {
                destroy(); return r;
            }
        }
    }
    if (auto r = descriptors_.create(device, 12, 96); !r) { destroy(); return r; }

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
        {&list_,   uint64_t(dims.slots+1) * sizeof(uint32_t)},
        {&routew_, uint64_t(spec.m) * dims.slots * sizeof(float)},
        {&x_,      x_bytes},
        {&h_,      h_bytes},
        {&y_,      uint64_t(spec.m) * dims.hidden * sizeof(float)},
        {&list_alt_, uint64_t(dims.slots) * sizeof(uint32_t)},
        {&list_sh_,  uint64_t(dims.slots) * sizeof(uint32_t)},
    };
    for (auto& e : bufs) {
        // `y` alone is device-addressable, so a caller can read the MoE output
        // from its next dispatch without a host copy (record_into).
        auto b = alloc.allocate(e.bytes, /*host_visible=*/true,
                                /*device_address=*/true);
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
        alt[2] = {2, 0, 0, list_sh_.buffer};
        auto s3 = descriptors_.allocate(gateup_, alt);
        if (!s3) { destroy(); return std::unexpected(s3.error()); }
        set_a_sh_ = *s3;
        // The shared expert is always the last slot.
        static_cast<uint32_t*>(list_sh_.host_ptr)[0] = dims.slots - 1;
    }

    std::vector<BufferBinding> bb(6);
    bb[0] = {0, 0, 0, table_.buffer};
    bb[1] = {1, 0, 0, ids_.buffer};
    bb[2] = {2, 0, 0, list_.buffer};
    bb[3] = {3, 0, 0, h_.buffer};
    bb[4] = {4, 0, 0, y_.buffer};
    bb[5] = {5, 0, 0, routew_.buffer};
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
        bh[0] = {0, 0, 0, list_sh_.buffer};
        auto sh3 = descriptors_.allocate(hquant_, bh);
        if (!sh3) { destroy(); return std::unexpected(sh3.error()); }
        set_hq_sh_ = *sh3;
    }
    if (xact_.valid()) {
        std::vector<BufferBinding> bx(1);
        bx[0] = {0, 0, 0, x_.buffer};
        auto sx = descriptors_.allocate(xact_, bx);
        if (!sx) { destroy(); return std::unexpected(sx.error()); }
        set_xact_ = *sx;
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
    gpu_descriptors_.destroy();gpu_copy_.destroy();gpu_mean_.destroy();gpu_route_.destroy();gpu_up_.destroy();gpu_down_.destroy();gpu_hq_.destroy();
    if(alloc_){for(auto* b:{&gpu_snapshot_,&gpu_args_,&gpu_indirect_})if(b->valid())alloc_->free(*b);}
    gpu_snapshot_=gpu_args_=gpu_indirect_={};gpu_layers_=0;
    gpu_route_sets_.clear();gpu_up_sets_.clear();gpu_down_sets_.clear();
    queries_.destroy();
    pool_.destroy();
    descriptors_.destroy();
    gateup_.destroy();
    down_.destroy();
    hquant_.destroy();
    xquant_.destroy();
    gateup_m1_.destroy();
    down_m1_.destroy();
    hquant_m1_.destroy();
    if (alloc_) {
        for (GpuBuffer* b : {&table_, &ids_, &list_, &routew_, &x_, &h_, &y_, &list_alt_, &list_sh_})
            if (b->valid()) alloc_->free(*b);
    }
    table_ = ids_ = list_ = routew_ = x_ = h_ = y_ = list_alt_ = list_sh_ = GpuBuffer{};
    set_a_ = set_b_ = set_hq_ = set_xq_ = set_a_alt_ = set_hq_alt_ = VK_NULL_HANDLE;
    set_a_sh_ = set_hq_sh_ = set_xact_ = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_  = nullptr;
    recorded_ = 0;
}

Result<void> MoeRunner::record(uint32_t iterations, MoePhase phase) {
    // Track K1a: one decision for the whole chain -- A, the h quantiser and B
    // must agree on M or the fp8 h plane offsets do not line up.
    const Pipeline& pipe_a  = use_m1() ? gateup_m1_ : gateup_;
    const Pipeline& pipe_b  = use_m1() ? down_m1_   : down_;
    const Pipeline& pipe_hq = use_m1() ? hquant_m1_ : hquant_;
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter  / rows_per_wg;
    const uint32_t groups_b = dims_.hidden / ((256 / spec_.b_lanes()) * spec_.b_rows());

    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                  dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    DownPush   pb{dims_.layer, dims_.experts_per_layer, dims_.slots, list_count_,
                  dims_.hidden, dims_.inter, accumulate_ ? 1u : 0u, live_columns_};
    // One thread per 32-element block of the thing being quantised.
    HQuantPush ph{dims_.slots, list_count_, dims_.inter};
    XQuantPush px{dims_.hidden};
    const uint32_t hq_groups =
        (effective_m() * list_count_ * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
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
            if (auto r = cmd_.bind(pipe_a, set_a_); !r) return r;
            if (auto r = cmd_.push(pipe_a, &pa, sizeof(pa)); !r) return r;
            if (auto r = cmd_.dispatch(groups_a, list_count_); !r) return r;
            // HQuant 3: the fp8 round trip of design §7.9 v0.6, as its own
            // dispatch, so dispatch A above is free to keep the workgroup shape
            // that is fastest for it (docs/kernel_p2_moe.md §8 item 1).
            if (pipe_hq.valid()) {
                if (auto r = cmd_.barrier(); !r) return r;
                if (auto r = cmd_.bind(pipe_hq, set_hq_); !r) return r;
                if (auto r = cmd_.push(pipe_hq, &ph, sizeof(ph)); !r) return r;
                if (auto r = cmd_.dispatch(hq_groups); !r) return r;
            }
            first = false;
            if (timed && it == 0) (void)cmd_.write_timestamp(queries_, 1, true);
        }
        if (run_b) {
            if (!first) { if (auto r = cmd_.barrier(); !r) return r; }
            if (auto r = cmd_.bind(pipe_b, set_b_); !r) return r;
            if (auto r = cmd_.push(pipe_b, &pb, sizeof(pb)); !r) return r;
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
    // Track K1a: one decision for the whole chain -- A, the h quantiser and B
    // must agree on M or the fp8 h plane offsets do not line up.
    const Pipeline& pipe_a  = use_m1() ? gateup_m1_ : gateup_;
    const Pipeline& pipe_b  = use_m1() ? down_m1_   : down_;
    const Pipeline& pipe_hq = use_m1() ? hquant_m1_ : hquant_;
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter  / rows_per_wg;
    const uint32_t groups_b = dims_.hidden / ((256 / spec_.b_lanes()) * spec_.b_rows());
    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                        dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    const DownPush   pb{dims_.layer, dims_.experts_per_layer, dims_.slots, list_count_,
                        dims_.hidden, dims_.inter, accumulate_ ? 1u : 0u, live_columns_};
    const HQuantPush ph{dims_.slots, list_count_, dims_.inter};
    const XQuantPush px{dims_.hidden};
    const uint32_t hq_groups =
        (effective_m() * list_count_ * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
    const uint32_t xq_groups =
        (spec_.m * (dims_.hidden / layout::kFp4ScaleBlock) + 255) / 256;

    if (phase != MoePhase::DownOnly) {
        if (xquant_.valid()) {
            if (auto r = cmd.bind(xquant_, set_xq_); !r) return r;
            if (auto r = cmd.push(xquant_, &px, sizeof(px)); !r) return r;
            if (auto r = cmd.dispatch(xq_groups); !r) return r;
            if (auto r = cmd.barrier(); !r) return r;
        }
        if (auto r = cmd.bind(pipe_a, set_a_); !r) return r;
        if (auto r = cmd.push(pipe_a, &pa, sizeof(pa)); !r) return r;
        if (auto r = cmd.dispatch(groups_a, list_count_); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
        if (pipe_hq.valid()) {
            if (auto r = cmd.bind(pipe_hq, set_hq_); !r) return r;
            if (auto r = cmd.push(pipe_hq, &ph, sizeof(ph)); !r) return r;
            if (auto r = cmd.dispatch(hq_groups); !r) return r;
            if (auto r = cmd.barrier(); !r) return r;
        }
    }
    if (phase != MoePhase::GateUpOnly) {
        if (auto r = cmd.bind(pipe_b, set_b_); !r) return r;
        if (auto r = cmd.push(pipe_b, &pb, sizeof(pb)); !r) return r;
        if (auto r = cmd.dispatch(groups_b); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
    }
    return {};
}

uint32_t* MoeRunner::slot_list_alt() { return static_cast<uint32_t*>(list_alt_.host_ptr); }

Result<void> MoeRunner::record_gateup_alt(CommandBuffer& cmd, uint32_t count) {
    if (!device_ || !gateup_.valid()) return fail(Err::FailedPrecondition, "runner is not created");
    if (count == 0 || count > dims_.slots) return fail(Err::InvalidArgument, "count must be 1..slots");
    // Track K1a: one decision for the whole chain -- A, the h quantiser and B
    // must agree on M or the fp8 h plane offsets do not line up.
    const Pipeline& pipe_a  = use_m1() ? gateup_m1_ : gateup_;
    const Pipeline& pipe_hq = use_m1() ? hquant_m1_ : hquant_;
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter / rows_per_wg;
    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                        dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    const HQuantPush ph{dims_.slots, count, dims_.inter};
    const XQuantPush px{dims_.hidden};
    const uint32_t hq_groups =
        (effective_m() * count * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
    const uint32_t xq_groups =
        (spec_.m * (dims_.hidden / layout::kFp4ScaleBlock) + 255) / 256;
    if (xquant_.valid()) {
        if (auto r = cmd.bind(xquant_, set_xq_); !r) return r;
        if (auto r = cmd.push(xquant_, &px, sizeof(px)); !r) return r;
        if (auto r = cmd.dispatch(xq_groups); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
    }
    if (auto r = cmd.bind(pipe_a, set_a_alt_); !r) return r;
    if (auto r = cmd.push(pipe_a, &pa, sizeof(pa)); !r) return r;
    if (auto r = cmd.dispatch(groups_a, count); !r) return r;
    if (auto r = cmd.barrier(); !r) return r;
    if (pipe_hq.valid()) {
        if (auto r = cmd.bind(pipe_hq, set_hq_alt_); !r) return r;
        if (auto r = cmd.push(pipe_hq, &ph, sizeof(ph)); !r) return r;
        if (auto r = cmd.dispatch(hq_groups); !r) return r;
        if (auto r = cmd.barrier(); !r) return r;
    }
    return {};
}

Result<void> MoeRunner::record_shared_early(CommandBuffer& cmd, uint64_t x_src_address) {
    if (!device_ || !gateup_.valid() || !xact_.valid() || !set_a_sh_)
        return fail(Err::FailedPrecondition, "runner has no shared-early path (x_mode 6?)");
    if (!x_src_address) return fail(Err::InvalidArgument, "record_shared_early: no x address");
    const Pipeline& pipe_a  = use_m1() ? gateup_m1_ : gateup_;
    const Pipeline& pipe_hq = use_m1() ? hquant_m1_ : hquant_;
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter / rows_per_wg;
    const XActPush px{x_src_address, dims_.hidden, 0};
    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                  dims_.inter, dims_.hidden, dims_.swiglu_limit, live_columns_};
    const HQuantPush ph{dims_.slots, 1, dims_.inter};
    const uint32_t xa_groups = (dims_.hidden / layout::kFp4ScaleBlock + 255) / 256;
    const uint32_t hq_groups =
        (effective_m() * 1 * (dims_.inter / layout::kFp4ScaleBlock) + 255) / 256;
    // Whatever the caller recorded before may still be reading x (the previous
    // layer's dispatch A): order against it.
    if (auto r = cmd.barrier(); !r) return r;
    if (auto r = cmd.bind(xact_, set_xact_); !r) return r;
    if (auto r = cmd.push(xact_, &px, sizeof(px)); !r) return r;
    if (auto r = cmd.dispatch(xa_groups); !r) return r;
    if (auto r = cmd.barrier(); !r) return r;
    if (auto r = cmd.bind(pipe_a, set_a_sh_); !r) return r;
    if (auto r = cmd.push(pipe_a, &pa, sizeof(pa)); !r) return r;
    if (auto r = cmd.dispatch(groups_a, 1); !r) return r;
    if (auto r = cmd.barrier(); !r) return r;
    if (pipe_hq.valid()) {
        if (auto r = cmd.bind(pipe_hq, set_hq_sh_); !r) return r;
        if (auto r = cmd.push(pipe_hq, &ph, sizeof(ph)); !r) return r;
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

Result<void> MoeRunner::init_gpu_route(uint32_t layers, const std::string &dir) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    if (gpu_layers_) return {};
    if (spec_.x_mode == 6 || spec_.h_quant != 3)
        return fail(Err::Unavailable, "GPU union needs exact fp16 x and h_quant=3");
    PipelineSpec ps;
    ps.m = spec_.m;
    ps.lanes_per_row = spec_.lanes_per_row;
    ps.rows_per_wg = 256 / spec_.lanes_per_row;
    ps.subgroup_size = spec_.subgroup_size;
    ps.extra = {spec_.decode_mode,
                spec_.h_precision,
                spec_.rows_per_lane,
                spec_.x_mode,
                spec_.h_quant,
                spec_.fp8_slots,
                0,
                1};
#define ROUTE_TRY(e)                                                                               \
    do {                                                                                           \
        auto r = (e);                                                                              \
        if (!r) return r;                                                                          \
    } while (0)
    ROUTE_TRY(gpu_up_.create(*device_, dir + "/moe_gateup.spv", {9, sizeof(GateUpPush)}, ps));
    auto pb = ps;
    pb.lanes_per_row = spec_.b_lanes();
    pb.rows_per_wg = 256 / spec_.b_lanes();
    pb.extra[2] = spec_.b_rows();
    pb.extra[3] = spec_.b_mode();
    ROUTE_TRY(gpu_down_.create(*device_, dir + "/moe_down.spv", {6, sizeof(DownPush)}, pb));
    ROUTE_TRY(gpu_hq_.create(*device_, dir + "/moe_hquant.spv", {2, sizeof(HQuantPush)}, ps));
    PipelineSpec rp;
    rp.subgroup_size = 32;
    rp.extra = {0};
    ROUTE_TRY(gpu_route_.create(*device_, dir + "/batch_route.spv", {1, 48}, rp));
    rp.extra = {1};
    ROUTE_TRY(gpu_copy_.create(*device_, dir + "/batch_route.spv", {1, 48}, rp));
    rp.extra = {2};
    ROUTE_TRY(gpu_mean_.create(*device_, dir + "/batch_route.spv", {1, 48}, rp));
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device_->physical(), &props);
    const auto align = std::max<uint64_t>(256, props.limits.minStorageBufferOffsetAlignment);
    gpu_arg_stride_ = align;
    gpu_table_stride_ = (uint64_t(dims_.experts_per_layer) * 6 * 8 + align - 1) / align * align;
    for (auto [b, bytes] : std::initializer_list<std::pair<GpuBuffer *, uint64_t>>{
             {&gpu_snapshot_, uint64_t(layers) * gpu_table_stride_},
             {&gpu_args_, uint64_t(layers) * gpu_arg_stride_},
             {&gpu_indirect_, uint64_t(layers) * 24}}) {
        auto r = alloc_->allocate(bytes, true, true);
        if (!r) return std::unexpected(r.error());
        *b = *r;
    }
    ROUTE_TRY(gpu_descriptors_.create(*device_, layers * 3, layers * 16));
    for (uint32_t l = 0; l < layers; ++l) {
        auto rs = gpu_descriptors_.allocate(
            gpu_route_, {{0, uint64_t(l) * gpu_arg_stride_, 256, gpu_args_.buffer}});
        if (!rs) return std::unexpected(rs.error());
        gpu_route_sets_.push_back(*rs);
        std::vector<BufferBinding> ba{{0, uint64_t(l) * gpu_table_stride_,
                                       uint64_t(dims_.experts_per_layer) * 6 * 8,
                                       gpu_snapshot_.buffer},
                                      {1, 0, 0, ids_.buffer},
                                      {2, 0, 0, list_.buffer},
                                      {3, 0, 0, routew_.buffer},
                                      {4, 0, 0, x_.buffer},
                                      {5, 0, 0, h_.buffer},
                                      {6, 0, 0, h_.buffer},
                                      {7, 0, 0, h_.buffer},
                                      {8, 0, 0, x_.buffer}};
        auto as = gpu_descriptors_.allocate(gpu_up_, ba);
        if (!as) return std::unexpected(as.error());
        gpu_up_sets_.push_back(*as);
        std::vector<BufferBinding> bb{ba[0],
                                      ba[1],
                                      ba[2],
                                      {3, 0, 0, h_.buffer},
                                      {4, 0, 0, y_.buffer},
                                      {5, 0, 0, routew_.buffer}};
        auto bs = gpu_descriptors_.allocate(gpu_down_, bb);
        if (!bs) return std::unexpected(bs.error());
        gpu_down_sets_.push_back(*bs);
    }
    gpu_layers_ = layers;
    return {};
#else
    return fail(Err::Unavailable, "no Vulkan");
#endif
}
Result<void> MoeRunner::upload_snapshot(std::span<const uint64_t> table) {
    if (!gpu_layers_ || table.size() != size_t(gpu_layers_) * dims_.experts_per_layer * 6)
        return fail(Err::InvalidArgument, "GPU snapshot size mismatch");
    const auto row_bytes = uint64_t(dims_.experts_per_layer) * 6 * 8;
    for (uint32_t l = 0; l < gpu_layers_; ++l)
        std::memcpy(static_cast<std::byte *>(gpu_snapshot_.host_ptr) + l * gpu_table_stride_,
                    table.data() + size_t(l) * dims_.experts_per_layer * 6, row_bytes);
    return {};
}
Result<void> MoeRunner::record_gpu_copy(CommandBuffer &cmd, uint64_t src, uint64_t dst, uint32_t n,
                                        bool mean, uint32_t hidden) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    if (mean && (!hidden || n % hidden))
        return fail(Err::InvalidArgument, "hidden mean requires complete rows");
    struct Push {
        uint32_t l, m, k, s, e, u, h, n;
        uint64_t src, dst;
    } p{0, 0, 0, 0, hidden, 0, 0, n, src, dst};
    auto &pipe = mean ? gpu_mean_ : gpu_copy_;
    ROUTE_TRY(cmd.bind(pipe, gpu_route_sets_[0]));
    ROUTE_TRY(cmd.push(pipe, &p, sizeof p));
    ROUTE_TRY(cmd.dispatch((n + 255) / 256));
    return cmd.barrier();
#else
    return fail(Err::Unavailable, "no Vulkan");
#endif
}
Result<void> MoeRunner::record_gpu_route(CommandBuffer &cmd, uint32_t layer, uint32_t m,
                                         uint32_t topk, uint64_t ids, uint64_t weights, uint64_t x,
                                         uint64_t saved, uint32_t *) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    // Each layer has immutable arguments. The routing dispatch selects from
    // the guarded snapshot and writes two VkDispatchIndirectCommand records.
    // Keep the indirect dependency before any dispatch consumes those counts.
    constexpr uint32_t indirect_bytes = 3 * sizeof(uint32_t);
    constexpr uint32_t indirect_stride = 2 * indirect_bytes;
    if (layer >= gpu_layers_ || m < 1 || m > spec_.m || m * topk + 1 > dims_.slots)
        return fail(Err::InvalidArgument, "GPU route dimensions");
    auto *a = reinterpret_cast<uint64_t *>(static_cast<std::byte *>(gpu_args_.host_ptr) +
                                           uint64_t(layer) * gpu_arg_stride_);
    uint64_t ptr[]{ids,
                   weights,
                   gpu_snapshot_.dev_addr + uint64_t(layer) * gpu_table_stride_,
                   ids_.dev_addr,
                   list_.dev_addr,
                   routew_.dev_addr,
                   saved,
                   gpu_indirect_.dev_addr + layer * indirect_stride};
    std::memcpy(a, ptr, sizeof ptr);
    uint32_t ga = dims_.inter / ((256 / spec_.lanes_per_row) * spec_.rows_per_lane);
    uint32_t gb = dims_.hidden / ((256 / spec_.b_lanes()) * spec_.b_rows());
    struct Push {
        uint32_t l, m, k, s, e, u, h, n;
        uint64_t src, dst;
    } p{layer, m, topk, dims_.slots, dims_.experts_per_layer, ga, 0, 0, 0, 0};
    ROUTE_TRY(cmd.bind(gpu_route_, gpu_route_sets_[layer]));
    ROUTE_TRY(cmd.push(gpu_route_, &p, sizeof p));
    ROUTE_TRY(cmd.dispatch(1));
    ROUTE_TRY(cmd.indirect_barrier());
    XActPush xa{x, dims_.hidden, m};
    ROUTE_TRY(cmd.bind(xact_, set_xact_));
    ROUTE_TRY(cmd.push(xact_, &xa, sizeof xa));
    ROUTE_TRY(cmd.dispatch((m * dims_.hidden / 32 + 255) / 256));
    ROUTE_TRY(cmd.barrier());
    GateUpPush up{
        0, dims_.experts_per_layer, dims_.slots, dims_.inter, dims_.hidden, dims_.swiglu_limit, m};
    ROUTE_TRY(cmd.bind(gpu_up_, gpu_up_sets_[layer]));
    ROUTE_TRY(cmd.push(gpu_up_, &up, sizeof up));
    ROUTE_TRY(cmd.dispatch_indirect(gpu_indirect_, layer * indirect_stride));
    ROUTE_TRY(cmd.barrier());
    HQuantPush hq{dims_.slots, dims_.slots, dims_.inter};
    ROUTE_TRY(cmd.bind(gpu_hq_, set_hq_));
    ROUTE_TRY(cmd.push(gpu_hq_, &hq, sizeof hq));
    ROUTE_TRY(cmd.dispatch_indirect(gpu_indirect_, layer * indirect_stride + indirect_bytes));
    ROUTE_TRY(cmd.barrier());
    DownPush down{
        0, dims_.experts_per_layer, dims_.slots, dims_.slots, dims_.hidden, dims_.inter, 0, m};
    ROUTE_TRY(cmd.bind(gpu_down_, gpu_down_sets_[layer]));
    ROUTE_TRY(cmd.push(gpu_down_, &down, sizeof down));
    ROUTE_TRY(cmd.dispatch(gb));
    return cmd.barrier();
#else
    return fail(Err::Unavailable, "no Vulkan");
#endif
#undef ROUTE_TRY
}
} // namespace deepmoe::gpu
