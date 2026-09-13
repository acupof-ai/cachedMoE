#include "gpu/vulkan/moe_kernels.h"

#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>
#include <format>

#include "core/profiler.h"

namespace deepmoe::gpu {

std::string MoeSpec::name() const {
    return std::format("M{} L{} R{} sg{} dec{} h{}", m, lanes_per_row, rows_per_lane,
                       subgroup_size ? std::to_string(subgroup_size) : std::string("auto"),
                       decode_mode, h_precision ? "fp32" : "fp16");
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
    const uint64_t per_mat = uint64_t(dims_.inter) * dims_.hidden / 2
                           + uint64_t(dims_.inter) * dims_.hidden / layout::kFp4ScaleBlock;
    return 2 * per_mat * list_count_;
}

uint64_t MoeRunner::bytes_dispatch_b() const {
    const uint64_t per_mat = uint64_t(dims_.hidden) * dims_.inter / 2
                           + uint64_t(dims_.hidden) * dims_.inter / layout::kFp4ScaleBlock;
    return per_mat * list_count_;
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

#else

namespace {

// design §7.9 dispatch A push constants; mirrors GateUpPush in the shader.
struct GateUpPush {
    uint32_t layer, experts_per_layer, num_slots, n_rows, k;
    float    swiglu_limit;
};
// design §7.9 dispatch B; mirrors DownPush.
struct DownPush {
    uint32_t layer, experts_per_layer, num_slots, list_count, n_rows, k;
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
    if (spec.rows_per_lane != 1 && spec.rows_per_lane != 2 && spec.rows_per_lane != 4)
        return fail(Err::InvalidArgument, "rows_per_lane must be 1, 2 or 4");
    const uint32_t rows_per_group = (256 / spec.lanes_per_row) * spec.rows_per_lane;
    if (dims.inter % rows_per_group || dims.hidden % rows_per_group)
        return fail(Err::InvalidArgument, "row count must divide by the workgroup's rows");
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
    ps.extra = {spec.decode_mode, spec.h_precision, spec.rows_per_lane};

    PipelineLayoutSpec la;
    la.storage_buffers   = 7;
    la.push_constant_size = sizeof(GateUpPush);
    if (auto r = gateup_.create(device, shader_dir + "/moe_gateup.spv", la, ps); !r) {
        destroy(); return r;
    }
    PipelineLayoutSpec lb;
    lb.storage_buffers   = 5;
    lb.push_constant_size = sizeof(DownPush);
    if (auto r = down_.create(device, shader_dir + "/moe_down.spv", lb, ps); !r) {
        destroy(); return r;
    }
    if (auto r = descriptors_.create(device, 4, 32); !r) { destroy(); return r; }

    const uint64_t h_elem = spec.h_precision ? 4 : 2;
    struct { GpuBuffer* b; uint64_t bytes; } bufs[] = {
        {&table_,  uint64_t(pointer_table_entries()) * sizeof(uint64_t)},
        {&ids_,    uint64_t(dims.slots) * sizeof(uint32_t)},
        {&list_,   uint64_t(dims.slots) * sizeof(uint32_t)},
        {&routew_, uint64_t(spec.m) * dims.slots * sizeof(float)},
        {&x_,      uint64_t(spec.m) * dims.hidden * 2},
        {&h_,      uint64_t(spec.m) * dims.slots * dims.inter * h_elem},
        {&y_,      uint64_t(spec.m) * dims.hidden * sizeof(float)},
    };
    for (auto& e : bufs) {
        auto b = alloc.allocate(e.bytes, /*host_visible=*/true, /*device_address=*/false);
        if (!b) { destroy(); return std::unexpected(b.error()); }
        *e.b = *b;
        std::memset(e.b->host_ptr, 0, static_cast<size_t>(e.bytes));
    }

    std::vector<BufferBinding> ba(7);
    ba[0] = {0, 0, 0, table_.buffer};
    ba[1] = {1, 0, 0, ids_.buffer};
    ba[2] = {2, 0, 0, list_.buffer};
    ba[3] = {3, 0, 0, routew_.buffer};
    ba[4] = {4, 0, 0, x_.buffer};
    ba[5] = {5, 0, 0, h_.buffer};
    ba[6] = {6, 0, 0, h_.buffer};        // H16 and H32 alias one allocation
    auto sa = descriptors_.allocate(gateup_, ba);
    if (!sa) { destroy(); return std::unexpected(sa.error()); }
    set_a_ = *sa;

    std::vector<BufferBinding> bb(5);
    bb[0] = {0, 0, 0, table_.buffer};
    bb[1] = {1, 0, 0, ids_.buffer};
    bb[2] = {2, 0, 0, list_.buffer};
    bb[3] = {3, 0, 0, h_.buffer};
    bb[4] = {4, 0, 0, y_.buffer};
    auto sb = descriptors_.allocate(down_, bb);
    if (!sb) { destroy(); return std::unexpected(sb.error()); }
    set_b_ = *sb;

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
    if (alloc_) {
        for (GpuBuffer* b : {&table_, &ids_, &list_, &routew_, &x_, &h_, &y_})
            if (b->valid()) alloc_->free(*b);
    }
    table_ = ids_ = list_ = routew_ = x_ = h_ = y_ = GpuBuffer{};
    set_a_ = set_b_ = VK_NULL_HANDLE;
    device_ = nullptr;
    alloc_  = nullptr;
    recorded_ = 0;
}

Result<void> MoeRunner::record(uint32_t iterations, MoePhase phase) {
    const uint32_t rows_per_wg = (256 / spec_.lanes_per_row) * spec_.rows_per_lane;
    const uint32_t groups_a = dims_.inter  / rows_per_wg;
    const uint32_t groups_b = dims_.hidden / rows_per_wg;

    GateUpPush pa{dims_.layer, dims_.experts_per_layer, dims_.slots,
                  dims_.inter, dims_.hidden, dims_.swiglu_limit};
    DownPush   pb{dims_.layer, dims_.experts_per_layer, dims_.slots, list_count_,
                  dims_.hidden, dims_.inter};

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
            if (auto r = cmd_.bind(gateup_, set_a_); !r) return r;
            if (auto r = cmd_.push(gateup_, &pa, sizeof(pa)); !r) return r;
            if (auto r = cmd_.dispatch(groups_a, list_count_); !r) return r;
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
