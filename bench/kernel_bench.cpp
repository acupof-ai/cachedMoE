// kernel_bench -- effective GB/s of the two FP4 MoE dispatches of design §7.9,
// against the raw-read ceiling of design §7.1 rule 2, on the real checkpoint.
//
// What it measures and why:
//
//   * Effective GB/s  = (weight + scale bytes the dispatch must read) / GPU
//     time. The denominator comes from timestamp queries, so driver and submit
//     overhead are excluded; the wall clock is printed next to it so the gap is
//     visible.
//   * The same number for gpu/shaders/rawread.slang over the same byte count
//     and the same memory. A kernel at 80% of raw read is at the machine's
//     limit (design §15 P2 exit criterion); a kernel at 30% is not.
//   * Per-dispatch launch + barrier cost, which design §3.4 guessed at 5-20 us
//     and turned into 5-14% of a token.
//
// The experts are real: `layer_cycle` layers x 7 experts are streamed off NVMe
// through IoEngine + ExpertStore into GPU-visible slabs, exactly as decode
// would, so the working set is 132 MB x layer_cycle and does not sit in the
// 32 MB MALL. The shared expert is stood in for by a seventh FP4 routed expert
// (the real one is fp8 -- design §7.9 -- which is a separate template).
//
// Usage:
//   kernel_bench --model-dir D:\models\DeepSeek-V4.1-Flash --csv out.csv
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "core/log.h"
#include "core/profiler.h"
#include "core/status.h"
#include "cpu/dequant.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/rawread.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/planner.h"
#include "store/shard_set.h"

using namespace deepmoe;

namespace {

struct Options {
    std::string model_dir;
    std::string csv;
    uint32_t    iters       = 32;
    uint32_t    layer_cycle = 4;
    uint32_t    slots       = 7;
    uint32_t    repeats     = 3;
    uint32_t    sweeps      = 2;
    MemoryPath  path        = MemoryPath::DeviceLocalHostVisible;
    bool        quick       = false;
    bool        p1          = false;   // the design §7.1 knob sweep of P1
    bool        no_fp8      = false;   // skip the shared expert (saves the NVMe read)
    std::string only;                  // run only the sections whose name contains this
};

const char* env(const char* name) {
#if defined(_MSC_VER)
    static std::string v; char* b = nullptr; size_t n = 0;
    if (_dupenv_s(&b, &n, name) == 0 && b) { v = b; free(b); return v.c_str(); }
    return nullptr;
#else
    return std::getenv(name);
#endif
}

int usage() {
    std::puts(
        "kernel_bench -- design section 7.9 MoE kernels vs the raw-read ceiling\n"
        "  --model-dir DIR   checkpoint directory (default: $DEEPMOE_MODEL_DIR)\n"
        "  --csv PATH        write the table as CSV\n"
        "  --iters N         timed A+B iterations per variant (default 32)\n"
        "  --layer-cycle N   distinct layers of experts to rotate over (default 4)\n"
        "  --path a|b        memory path of design section 3.3 (default a)\n"
        "  --quick           only the default variant\n"
        "  --p1              the P1 knob sweep (decode mode, lanes, wave size)\n"
        "                    instead of the P2 sweep (M x activation staging)\n"
        "  --no-fp8          skip the fp8 shared expert section\n"
        "  --only SUBSTR     only variants whose section name contains SUBSTR\n"
        "                    (none of them = the partial-dispatch section alone)\n");
    return 2;
}

struct Row {
    std::string variant;
    const char* section = "";
    uint32_t    m = 0, lanes = 0, subgroup = 0, decode = 0, hprec = 0, rpl = 0;
    uint32_t    xmode = 0, hquant = 0, fp8 = 0, fp8_slots = 0, list_count = 0;
    uint64_t    bytes_a = 0, bytes_b = 0;
    double      gbps_a = 0, gbps_b = 0, gbps_total = 0;
    double      ms_a = 0, ms_b = 0, ms_total = 0, ms_wall = 0, ms_per_token = 0;
    double      record_us = 0, submit_us = 0;
};

// One entry of the sweep: a specialisation plus how the slot list is made up.
// `fp8_slots` slots at the end of the list carry the fp8 shared expert, which
// is the only thing that changes the byte count of a dispatch.
struct Variant {
    gpu::MoeSpec spec{};
    uint32_t     fp8_slots = 0;
    const char*  section = "M sweep";
};

// The fp8 shared expert of one layer, read straight off NVMe into a GPU-visible
// buffer with a device address (design §7.9; it is pinned for the life of the
// process, so it is not an ExpertStore slot). The six device addresses go into
// the spare expert index of the runner's pointer table.
struct SharedExpert {
    gpu::GpuBuffer buf{};
    uint64_t       addr[kExpertPartCount] = {};
    uint64_t       value_bytes = 0, scale_bytes = 0;
};

Result<SharedExpert> load_shared_expert(gpu::MemoryAllocator& alloc, const Manifest& mf,
                                        const store::ShardSet& shards, storage::IoEngine& io,
                                        uint32_t layer) {
    static const char* kMats[3] = {"w1", "w2", "w3"};
    AlignedRead reads[kExpertPartCount];
    for (uint32_t i = 0; i < 3; ++i) {
        const std::string name = std::format("layers.{}.ffn.shared_experts.{}.weight",
                                             layer, kMats[i]);
        auto v = mf.tensor_read(name);
        if (!v) return std::unexpected(v.error());
        auto sc = mf.tensor_scale_read(name);
        if (!sc) return std::unexpected(sc.error());
        reads[2 * i]     = *v;
        reads[2 * i + 1] = *sc;
    }
    uint64_t total = 0, off[kExpertPartCount] = {};
    for (uint32_t i = 0; i < kExpertPartCount; ++i) {
        off[i] = total;
        total += align_up(reads[i].aligned_bytes, kPageSize);
    }
    auto b = alloc.allocate(total, /*host_visible=*/true, /*device_address=*/true);
    if (!b) return std::unexpected(b.error());
    SharedExpert out;
    out.buf = *b;
    if (reinterpret_cast<uintptr_t>(out.buf.host_ptr) % kPageSize)
        return fail(Err::Internal, "the shared-expert buffer is not page aligned");
    std::vector<std::future<storage::IoResult>> futures;
    for (uint32_t i = 0; i < kExpertPartCount; ++i) {
        auto file = shards.require(reads[i].file);
        if (!file) return std::unexpected(file.error());
        storage::IoRequest req;
        req.priority = IoPriority::BlockingMiss;
        req.file     = *file;
        req.file_off = reads[i].aligned_off;
        req.bytes    = reads[i].aligned_bytes;
        req.dst      = static_cast<std::byte*>(out.buf.host_ptr) + off[i];
        auto fut = io.submit_future(req);
        if (!fut) return std::unexpected(fut.error());
        futures.push_back(std::move(*fut));
        out.addr[i] = out.buf.dev_addr + off[i] + reads[i].skew;
    }
    for (auto& f : futures)
        if (!f.get().ok()) return fail(Err::Io, "shared expert read failed");
    out.value_bytes = reads[0].bytes;
    out.scale_bytes = reads[1].bytes;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    set_log_level(LogLevel::Warn);
    Options o;
    if (const char* e = env("DEEPMOE_MODEL_DIR")) o.model_dir = e;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fputs("missing value\n", stderr); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model-dir")        o.model_dir = next();
        else if (a == "--csv")         o.csv = next();
        else if (a == "--iters")       o.iters = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--layer-cycle") o.layer_cycle = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--repeats")     o.repeats = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--sweeps")      o.sweeps = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--path")        o.path = next() == "b" ? MemoryPath::ExternalMemoryHost
                                                              : MemoryPath::DeviceLocalHostVisible;
        else if (a == "--quick")       o.quick = true;
        else if (a == "--p1")          o.p1 = true;
        else if (a == "--no-fp8")      o.no_fp8 = true;
        else if (a == "--only")        o.only = next();
        else return usage();
    }
    if (o.model_dir.empty()) {
        std::fputs("no checkpoint: pass --model-dir or set DEEPMOE_MODEL_DIR\n", stderr);
        return 2;
    }
    if (o.layer_cycle == 0) o.layer_cycle = 1;

    // --- device and memory ---------------------------------------------------
    gpu::Device dev;
    gpu::DeviceOptions dopts;
    dopts.enable_validation = env("VK_INSTANCE_LAYERS") != nullptr;
    if (auto r = dev.create(dopts); !r) {
        std::puts(std::format("vulkan unavailable: {}", r.error().str()).c_str());
        return 1;
    }
    if (auto r = dev.caps().check_required(); !r) {
        std::puts(r.error().str().c_str());
        return 1;
    }
    std::puts(std::format("device: {}  subgroup {} ({}..{})  timestampPeriod {:.1f} ns",
                          dev.caps().device_name, dev.caps().subgroup_size,
                          dev.caps().min_subgroup_size, dev.caps().max_subgroup_size,
                          dev.caps().timestamp_period_ns).c_str());

    gpu::MemoryAllocator alloc;
    if (auto r = alloc.init(dev, o.path); !r) {
        std::puts(std::format("memory path unavailable: {}", r.error().str()).c_str());
        return 1;
    }
    const char* path_name = o.path == MemoryPath::ExternalMemoryHost
                              ? "B external_memory_host" : "A device-local|host-visible";

    // --- checkpoint ----------------------------------------------------------
    auto mf = Manifest::load(store::ShardSet::join(o.model_dir, layout::kManifestFile));
    if (!mf) { std::puts(mf.error().str().c_str()); return 1; }
    store::ShardSet shards;
    if (auto r = shards.open_all(o.model_dir, *mf, true); !r) { std::puts(r.error().str().c_str()); return 1; }

    IoConfig io_cfg;
    auto backend = storage::make_default_backend(io_cfg);
    if (!backend) { std::puts(backend.error().str().c_str()); return 1; }
    storage::IoEngine io;
    if (auto r = io.start(std::move(*backend), io_cfg); !r) { std::puts(r.error().str().c_str()); return 1; }

    const uint32_t needed = o.slots * o.layer_cycle;
    CacheConfig cache;
    cache.slots_per_slab = needed;
    cache.budget_bytes   = uint64_t(needed) * layout::kExpertSlotBytes;

    auto backing = alloc.make_slab_backing();
    if (!backing) { std::puts(backing.error().str().c_str()); return 1; }
    store::ExpertStore estore;
    if (auto r = estore.init(std::move(*backing), cache); !r) {
        std::puts(std::format("expert store: {}", r.error().str()).c_str());
        return 1;
    }
    std::puts(std::format("path {}: {} expert slots, {:.1f} MB of slab",
                          path_name, estore.slot_count(), estore.capacity_bytes() / 1e6).c_str());

    store::Planner planner;
    if (auto r = planner.init(estore, io, *mf, shards, cache, PrefetchConfig{}); !r) {
        std::puts(r.error().str().c_str());
        return 1;
    }

    // This is design §9.6's zero-copy claim under test: FILE_FLAG_NO_BUFFERING
    // reads landing directly in mapped GPU memory.
    const auto t_fill = Clock::now();
    uint64_t filled_bytes = 0;
    for (uint32_t L = 0; L < o.layer_cycle; ++L) {
        for (uint32_t e = 0; e < o.slots; ++e) {
            const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(e)};
            auto f = planner.fetch(key, IoPriority::BlockingMiss, 1, L);
            if (!f) { std::puts(std::format("fetch {}:{}: {}", L, e, f.error().str()).c_str()); return 1; }
            filled_bytes += f->bytes;
        }
    }
    io.drain();
    const double fill_s = std::chrono::duration<double>(Clock::now() - t_fill).count();
    for (uint32_t L = 0; L < o.layer_cycle; ++L)
        for (uint32_t e = 0; e < o.slots; ++e)
            if (!estore.resident(ExpertKey{static_cast<uint16_t>(L), static_cast<uint16_t>(e)})) {
                std::puts(std::format("expert {}:{} did not become resident", L, e).c_str());
                return 1;
            }
    std::puts(std::format("filled {} experts ({:.1f} MB) straight into {} memory in {:.3f} s "
                          "= {:.2f} GB/s NVMe -> GPU",
                          needed, filled_bytes / 1e6, path_name, fill_s,
                          filled_bytes / 1e9 / fill_s).c_str());

    // --- the fp8 shared expert (design §7.9) ---------------------------------
    // One per layer of the cycle, so a cycled iteration reads a different copy
    // exactly as it does for the routed experts.
    std::vector<SharedExpert> shared;
    if (!o.no_fp8) {
        const auto t0 = Clock::now();
        for (uint32_t L = 0; L < o.layer_cycle; ++L) {
            auto sh = load_shared_expert(alloc, *mf, shards, io, L);
            if (!sh) {
                std::puts(std::format("shared expert layer {}: {} -- continuing without the "
                                      "fp8 section", L, sh.error().str()).c_str());
                shared.clear();
                break;
            }
            shared.push_back(*sh);
        }
        if (!shared.empty()) {
            const double s = std::chrono::duration<double>(Clock::now() - t0).count();
            std::puts(std::format("filled {} fp8 shared experts ({:.1f} MB, {} + {} B per "
                                  "matrix) in {:.3f} s = {:.2f} GB/s",
                                  shared.size(),
                                  shared.size() * 3.0 * (shared[0].value_bytes
                                                         + shared[0].scale_bytes) / 1e6,
                                  shared[0].value_bytes, shared[0].scale_bytes, s,
                                  shared.size() * 3.0 * (shared[0].value_bytes
                                                         + shared[0].scale_bytes) / 1e9 / s).c_str());
        }
    }

    // --- variants ------------------------------------------------------------
    std::vector<Variant> variants;
    auto add = [&](gpu::MoeSpec sp, const char* section = "M sweep", uint32_t fp8 = 0) {
        variants.push_back(Variant{sp, fp8, section});
    };
    if (o.quick) {
        add(gpu::MoeSpec{1, 32, 32, 0, 0, 1});
        add(gpu::MoeSpec{6, 16, 32, 0, 0, 2, 1});
    } else if (o.p1) {
        // design §7.1's knobs at the two batch sizes that matter: M=1 is
        // decode, M=6 a full speculative verify batch. RowsPerLane > 1 costs
        // registers, so it is only swept where it can pay: at M=1 the lane has
        // registers to spare, at M=6 it does not.
        for (uint32_t m : {1u, 6u})
            for (uint32_t dec : {0u, 1u, 2u})
                for (uint32_t lanes : {16u, 32u, 64u})
                    for (uint32_t sg : {32u, 64u})
                        add(gpu::MoeSpec{m, lanes, sg, dec, 0, 1}, "P1 knobs");
        for (uint32_t rpl : {2u, 4u})
            for (uint32_t lanes : {16u, 32u, 64u})
                for (uint32_t sg : {32u, 64u})
                    add(gpu::MoeSpec{1, lanes, sg, 0, 0, rpl}, "P1 knobs");
        for (uint32_t lanes : {16u, 32u})
            for (uint32_t sg : {32u, 64u})
                add(gpu::MoeSpec{6, lanes, sg, 0, 0, 2}, "P1 knobs");
        // fp32 h doubles the h traffic; worth measuring only on the variants
        // that are otherwise competitive.
        for (uint32_t m : {1u, 6u})
            for (uint32_t lanes : {16u, 32u})
                for (uint32_t rpl : {1u, 2u})
                    add(gpu::MoeSpec{m, lanes, 32, 0, 1, rpl}, "P1 knobs");
    } else {
        // P2 (docs/kernel_p2_moe.md): the knobs P1 pinned are fixed at their
        // winners -- constant-table FP4 decode, wave32, fp16 h -- and the sweep
        // runs over the two things that decide speculative decoding: the verify
        // batch size M and how the activation reaches the FMA.
        // RowsPerLane is the knob that divides the activation traffic: a
        // workgroup re-reads the whole of x once per row group, so the x bytes
        // that cross L2 are 2304 * slots * M * 10 KiB / RowsPerLane and nothing
        // else in the kernel changes them (docs/kernel_p2_moe.md §3).
        for (uint32_t m = 1; m <= 6; ++m)
            for (uint32_t lanes : {16u, 32u, 64u})
                for (uint32_t rpl : {1u, 2u, 4u, 8u, 16u})
                    add(gpu::MoeSpec{m, lanes, 32, 0, 0, rpl, 0}, "M x RowsPerLane");
        for (uint32_t m = 1; m <= 6; ++m) {
            for (uint32_t x = 0; x <= 5; ++x) {
                // (lanes, rows per lane): 32/1 is the P1 decode winner, 16/2 the
                // P1 M=6 winner; the int8 path stages one (column, block) per
                // thread, so M * lanes must fit a workgroup.
                add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, x});
                add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, x});
            }
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 4, 1});
            // design §7.9.1 open item 3: dispatch A and dispatch B want
            // different specialisations. B's K is 2304, which is 72 blocks and
            // therefore not a multiple of LanesPerRow, so its tiled form pays a
            // ragged last chunk that A (160 blocks) does not.
            for (uint32_t x : {1u, 2u, 3u, 4u, 5u}) {
                gpu::MoeSpec sp{m, 16, 32, 0, 0, 2, x};
                sp.x_mode_b = 0;
                add(sp);
            }
            // The other direction: A plain, B staged or packed. Dispatch B's
            // 2304 K-elements are 72 blocks, so its tiled form has a ragged
            // last chunk that A (160 blocks) does not.
            for (uint32_t xb : {3u, 4u, 5u}) {
                gpu::MoeSpec sp{m, 16, 32, 0, 0, 2, 0};
                sp.x_mode_b = xb;
                add(sp);
            }
            // The pair the two halves each win with on their own.
            for (uint32_t lanes : {16u, 32u})
                for (uint32_t xb : {3u, 4u}) {
                    gpu::MoeSpec sp{m, lanes, 32, 0, 0, 2, 4};
                    sp.x_mode_b = xb;
                    add(sp);
                }
        }
        // The fp8 quantisation of h before w2 (design §7.9 v0.6). L16 R2 owns
        // 32 rows of h per workgroup, which is what writing whole fp8 blocks
        // from dispatch A requires.
        //
        // The unquantised M sweep winners are repeated inside this section as
        // its control. §2's drift is 8% *between sections of one run*, and the
        // thing being measured here is 1-5%, so "what does HQuant cost" can
        // only be read off rows that were measured next to each other.
        // Both shapes at every M, with and without HQuant = 3, so the
        // "what does the fp8 h cost per M" table is a set of adjacent pairs.
        for (uint32_t m = 1; m <= 6; ++m)
            for (uint32_t hq : {0u, 3u}) {
                add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, 0, hq}, "h fp8");
                gpu::MoeSpec sp{m, 16, 32, 0, 0, 2, 4, hq};
                sp.x_mode_b = 3;
                add(sp, "h fp8");
            }
        for (uint32_t m : {1u, 6u})
            for (uint32_t x : {0u, 4u})
                for (uint32_t hq : {1u, 2u})
                    add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, x, hq}, "h fp8");
        // HQuant 1 has no row-count constraint, so it is the only way to get
        // the fp8 h onto the M=1 decode winner (L32 R1, which owns 8 rows per
        // workgroup and cannot write a whole 32-element fp8 block).
        for (uint32_t m : {1u, 6u})
            for (uint32_t x : {0u, 4u})
                add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, x, 1}, "h fp8");
        add(gpu::MoeSpec{1, 32, 32, 0, 0, 4, 0, 2}, "h fp8");   // 32 rows per wg via R=4
        // HQuant 3 moves the same quantisation into its own dispatch, which
        // frees dispatch A from the 32-row constraint: the point of the section
        // is whether L32 R1 + hqP gets back the 27% §5.3 lost. The L16 R2 rows
        // are the control -- 3 and 2 must cost the same there.
        for (uint32_t m : {1u, 6u}) {
            add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, 0, 3}, "h fp8");
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, 0, 3}, "h fp8");
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, 4, 3}, "h fp8");
        }
        // The production pair: the gate/up half packed fp16, the down half
        // int8-through-LDS, with the fp8 h of design §7.9 v0.6 in between.
        for (uint32_t m : {1u, 6u})
            for (uint32_t hq : {2u, 3u}) {
                gpu::MoeSpec sp{m, 16, 32, 0, 0, 2, 4, hq};
                sp.x_mode_b = 3;
                add(sp, "h fp8");
            }
        // docs/kernel_p2_moe.md §3.6 / §8 item 2: x quantised to int8 once per
        // token by a tiny pre-dispatch, so dispatch A's per (column, block,
        // row) cost is 8 dot4add_i8packed and nothing else. The knob that could
        // not pay for itself at XMode 5 is RowsPerLane -- the int8 path holds
        // no per-row float arrays, so the register cliff §3.3 found at R >= 4
        // may not be there. Hence R up to 8.
        for (uint32_t m = 1; m <= 6; ++m) {
            // Same reasoning as the h fp8 section: the fp16 winners are the
            // in-section control, not a row from somewhere else in the run.
            add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, 0}, "int8 x");
            {
                gpu::MoeSpec sp{m, 16, 32, 0, 0, 2, 4};
                sp.x_mode_b = 3;
                add(sp, "int8 x");
            }
            add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, 6}, "int8 x");
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, 6}, "int8 x");
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 4, 6}, "int8 x");
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 8, 6}, "int8 x");
            gpu::MoeSpec sp{m, 16, 32, 0, 0, 4, 6};
            sp.x_mode_b = 3;                      // B on int8 h through LDS
            add(sp, "int8 x");
        }
    }
    // The fp8 shared expert (design §7.9): slot 6 stops being an FP4 stand-in
    // and becomes the real thing, which reads twice the weight bytes. Both the
    // mixed 6+1 list and the shared expert alone are measured, so its own
    // effective bandwidth can be separated from the routed experts'.
    if (!shared.empty() && !o.quick) {
        for (uint32_t m : {1u, 6u}) {
            add(gpu::MoeSpec{m, 32, 32, 0, 0, 1, 0, 0, 1}, "fp8 shared", 1);
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, 4, 0, 1}, "fp8 shared", 1);
            add(gpu::MoeSpec{m, 16, 32, 0, 0, 2, 4, 2, 1}, "fp8 shared", 1);
        }
    }

    // One spare expert index per layer holds the fp8 shared expert, so a slot
    // can point at it through the same pointer table (design §5.3).
    const uint32_t experts_per_layer = layout::kRoutedExperts + 1;
    const uint32_t shared_index      = layout::kRoutedExperts;

    gpu::MoeDims dims_base;
    dims_base.layer             = 0;
    dims_base.slots             = o.slots;
    dims_base.layer_cycle       = o.layer_cycle;
    dims_base.experts_per_layer = experts_per_layer;

    // A whole sweep is ~25 minutes, which is long enough for the part to warm
    // several degrees (section 2's drift). --only re-measures one section on a
    // machine in a known state.
    if (!o.only.empty()) {
        std::vector<Variant> keep;
        for (const Variant& v : variants)
            if (std::string_view(v.section).find(o.only) != std::string_view::npos)
                keep.push_back(v);
        variants.swap(keep);
        std::puts(std::format("--only \"{}\": {} variants", o.only, variants.size()).c_str());
    }

    const std::string shader_dir = gpu::default_shader_dir();
    std::vector<Row> rows(variants.size());

    // Match the MoE working set so the MALL sees the same reuse.
    const uint64_t per_iter = uint64_t(o.slots) *
        (2 * (uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 2 +
              uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 32) +
         (uint64_t(layout::kHiddenSize) * layout::kMoeIntermediate / 2 +
          uint64_t(layout::kHiddenSize) * layout::kMoeIntermediate / 32));

    double ceiling = 0.0, launch_us = 0.0;
    // The raw-read ceiling drifts with the GPU's thermal and power state, so it
    // is measured before and after the sweep and both are reported: the pair
    // brackets how much of a variant-to-variant difference could be drift.
    auto measure_ceiling = [&](const char* when) {
        std::puts(std::format("\n== raw-read ceiling {} the sweep "
                              "(design section 7.1 rule 2) ==", when).c_str());
        double best = 0.0;
        for (uint32_t groups : {160u, 320u, 640u, 1280u}) {
            gpu::RawReadKernel raw;
            if (auto r = raw.create(dev, alloc, shader_dir, groups); !r) {
                std::puts(r.error().str().c_str());
                break;
            }
            const uint64_t want = raw.round_bytes(per_iter * o.layer_cycle);
            auto buf = alloc.allocate_slab(align_up(want, kPageSize));
            if (!buf) { std::puts(buf.error().str().c_str()); break; }
            std::memset(buf->host_ptr, 0x5A, static_cast<size_t>(buf->bytes));
            const uint32_t passes = std::max(1u, o.iters / o.layer_cycle);
            (void)raw.run(*buf, want, 1);
            double g = 0.0;
            bool timed = false;
            for (uint32_t r = 0; r < o.repeats; ++r)
                if (auto res = raw.run(*buf, want, passes); res) {
                    g = std::max(g, res->gbps);
                    timed = res->gpu_timed;
                }
            best = std::max(best, g);
            std::puts(std::format("  {:>5} workgroups  {:.1f} GB/s over {:.1f} MB x {} passes ({})",
                                  groups, g, want / 1e6, passes,
                                  timed ? "gpu timestamps" : "wall clock").c_str());
            if (groups == 320u) {
                (void)raw.run_empty(*buf, 128);
                double us = 1e9;
                for (uint32_t r = 0; r < o.repeats; ++r)
                    if (auto e = raw.run_empty(*buf, 1024); e)
                        us = std::min(us, e->seconds / 1024 * 1e6);
                if (us < 1e8) launch_us = us;
            }
            alloc.free(*buf);
        }
        ceiling = std::max(ceiling, best);
        return best;
    };

    // Bring the GPU to a steady clock before anything is timed: about a second
    // of the raw-read kernel over the same memory.
    {
        gpu::RawReadKernel warm;
        if (warm.create(dev, alloc, shader_dir, 320)) {
            if (auto b = alloc.allocate_slab(256u << 20)) {
                std::memset(b->host_ptr, 0x5A, static_cast<size_t>(b->bytes));
                const auto t0 = Clock::now();
                while (std::chrono::duration<double>(Clock::now() - t0).count() < 1.0)
                    (void)warm.run(*b, warm.round_bytes(b->bytes), 4);
                alloc.free(*b);
            }
        }
    }
    const double ceiling_before = measure_ceiling("before");

    // `sweeps` passes over the whole variant list, keeping each variant's best.
    // One pass alone would systematically favour whichever variants ran while
    // the part was coolest.
    std::puts(std::format("\n== MoE kernels (design section 7.9), {} sweeps x {} measurements "
                          "x {} iterations ==", o.sweeps, o.repeats, o.iters).c_str());
    for (uint32_t sweep = 0; sweep < o.sweeps; ++sweep) {
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            const gpu::MoeSpec& v = variants[vi].spec;
            const uint32_t fp8_slots = variants[vi].fp8_slots;
            gpu::MoeDims dims = dims_base;
            dims.fp8_slot_count = fp8_slots;
            gpu::MoeRunner runner;
            if (auto r = runner.create(dev, alloc, shader_dir, v, dims); !r) {
                if (sweep == 0) std::puts(std::format("{:<34} {}", v.name(), r.error().str()).c_str());
                continue;
            }
            // The store's table is [layers][384][6]; the runner's is one expert
            // wider, so it is copied a layer at a time and the spare index gets
            // the shared expert of that layer.
            uint64_t* table = runner.pointer_table();
            std::memset(table, 0, runner.pointer_table_entries() * sizeof(uint64_t));
            for (uint32_t L = 0; L < layout::kTotalLogicalLayers; ++L) {
                std::memcpy(table + size_t(L) * experts_per_layer * kExpertPartCount,
                            estore.pointer_table()
                                + size_t(L) * layout::kRoutedExperts * kExpertPartCount,
                            size_t(layout::kRoutedExperts) * kExpertPartCount * sizeof(uint64_t));
                if (L < shared.size()) {
                    uint64_t* row = table + (size_t(L) * experts_per_layer + shared_index)
                                            * kExpertPartCount;
                    for (uint32_t i = 0; i < kExpertPartCount; ++i) row[i] = shared[L].addr[i];
                }
            }
            for (uint32_t s = 0; s < o.slots; ++s) {
                const bool is_fp8 = s >= o.slots - fp8_slots;
                runner.ids()[s] = is_fp8 ? (shared_index | gpu::kSlotFp8) : s;
                runner.slot_list()[s] = s;
            }
            runner.set_list_count(o.slots);
            for (uint32_t i = 0; i < v.m * o.slots; ++i) runner.route_weights()[i] = 1.0f / o.slots;
            // A deterministic, well-scaled x: the numbers only have to be
            // realistic in magnitude; correctness lives in tests/test_gpu_moe.cpp.
            uint32_t seed = 12345;
            for (uint32_t i = 0; i < v.m * layout::kHiddenSize; ++i) {
                seed = seed * 1664525u + 1013904223u;
                const float u = float(seed >> 8) / float(1u << 24) - 0.5f;
                runner.x_fp16()[i] = cpu::float_to_fp16(u * 0.1f);
            }

            // Clocks ramp over tens of milliseconds, so a single 50 ms
            // measurement mostly reports where DVFS happened to be. Warm up,
            // then keep the fastest run: it is the one least contaminated by a
            // clock step or by the desktop compositor.
            if (auto r = runner.run(8); !r) {
                if (sweep == 0)
                    std::puts(std::format("{:<24} warmup failed: {}", v.name(), r.error().str()).c_str());
                continue;
            }
            auto best_of = [&](gpu::MoePhase phase) -> Result<gpu::MoeTiming> {
                Result<gpu::MoeTiming> best = fail(Err::Internal, "no run");
                for (uint32_t r = 0; r < o.repeats; ++r) {
                    auto t = runner.run(o.iters, phase);
                    if (!t) return t;
                    if (!best || t->seconds_total < best->seconds_total) best = t;
                }
                return best;
            };
            auto both   = best_of(gpu::MoePhase::Both);
            auto only_a = best_of(gpu::MoePhase::GateUpOnly);
            auto only_b = best_of(gpu::MoePhase::DownOnly);
            if (!both || !only_a || !only_b) {
                if (sweep == 0) std::puts(std::format("{:<24} run failed", v.name()).c_str());
                continue;
            }

            Row row;
            row.variant = v.name();
            row.section = variants[vi].section;
            row.m = v.m; row.lanes = v.lanes_per_row; row.subgroup = v.subgroup_size;
            row.decode = v.decode_mode; row.hprec = v.h_precision; row.rpl = v.rows_per_lane;
            row.xmode = v.x_mode; row.hquant = v.h_quant; row.fp8 = v.fp8_slots;
            row.fp8_slots = fp8_slots; row.list_count = o.slots;
            row.bytes_a = runner.bytes_dispatch_a();
            row.bytes_b = runner.bytes_dispatch_b();
            row.ms_a     = only_a->seconds_total * 1e3;
            row.ms_b     = only_b->seconds_total * 1e3;
            row.ms_total = both->seconds_total * 1e3;
            row.ms_wall  = both->wall_seconds / o.iters * 1e3;
            // CPU-side cost of putting one A+B pair into the command buffer,
            // and whatever the wall clock has left over after the GPU's own
            // time: the two halves of design 3.4's dispatch overhead.
            row.record_us = both->record_seconds / o.iters * 1e6;
            row.submit_us = std::max(0.0, both->wall_seconds - both->seconds_total * o.iters
                                          - both->record_seconds) * 1e6;
            row.gbps_a     = double(runner.bytes_dispatch_a()) / 1e9 / only_a->seconds_total;
            row.gbps_b     = double(runner.bytes_dispatch_b()) / 1e9 / only_b->seconds_total;
            row.gbps_total = double(runner.bytes_per_iteration()) / 1e9 / both->seconds_total;
            // The number speculative decoding is actually bought with: one A+B
            // pair produces M tokens' worth of MoE work (design §10.1).
            row.ms_per_token = row.ms_total / double(v.m);
            if (row.gbps_total > rows[vi].gbps_total) rows[vi] = row;
        }
        std::puts(std::format("  sweep {}/{} done", sweep + 1, o.sweeps).c_str());
    }

    const double ceiling_after = measure_ceiling("after");

    std::puts("\nvariant                              A GB/s   B GB/s  A+B GB/s   %ceil    A ms"
              "    B ms  A+B ms  ms/tok");
    std::puts("---------------------------------------------------------------------------------"
              "----------------------------");
    const char* section = nullptr;
    for (const Row& r : rows) {
        if (r.variant.empty()) continue;
        if (!section || std::strcmp(section, r.section) != 0) {
            section = r.section;
            std::puts(std::format("-- {} --", section).c_str());
        }
        std::puts(std::format("{:<36} {:>7.1f} {:>8.1f} {:>9.1f} {:>7.0f} {:>7.3f} {:>7.3f} "
                              "{:>7.3f} {:>7.3f}",
                              r.variant, r.gbps_a, r.gbps_b, r.gbps_total,
                              ceiling > 0 ? 100.0 * r.gbps_total / ceiling : 0.0,
                              r.ms_a, r.ms_b, r.ms_total, r.ms_per_token).c_str());
    }

    std::puts(std::format("\nraw-read ceiling {:.1f} GB/s before the sweep, {:.1f} GB/s after "
                          "(the spread is thermal drift, not kernel behaviour);\n"
                          "per-dispatch launch + global barrier {:.2f} us "
                          "(design section 3.4 assumed 5-20)",
                          ceiling_before, ceiling_after, launch_us).c_str());
    if (ceiling > 0) {
        // Per M: the best variant and what it costs per token. This is the
        // table design §10.3's T_hot(M) curve is built from.
        std::puts("\nbest variant per verify batch size (fp4 routed experts only):");
        for (uint32_t m = 1; m <= 6; ++m) {
            const Row* best_m = nullptr;
            for (const Row& r : rows)
                if (r.m == m && r.fp8_slots == 0 && r.hquant == 0 && !r.variant.empty() &&
                    (!best_m || r.gbps_total > best_m->gbps_total)) best_m = &r;
            if (!best_m) continue;
            std::puts(std::format("  M={}  {:<32} {:>7.1f} GB/s = {:>3.0f}% of ceiling   "
                                  "{:.3f} ms/pair   {:.4f} ms/token",
                                  m, best_m->variant, best_m->gbps_total,
                                  100.0 * best_m->gbps_total / ceiling,
                                  best_m->ms_total, best_m->ms_per_token).c_str());
        }
        double best = 0; std::string best_name;
        double best6 = 0; std::string best6_name; double ms6 = 0;
        for (const Row& r : rows) {
            if (r.m == 1 && r.fp8_slots == 0 && r.hquant == 0 && r.gbps_total > best) {
                best = r.gbps_total; best_name = r.variant;
            }
            if (r.m == 6 && r.fp8_slots == 0 && r.hquant == 0 && r.gbps_total > best6) {
                best6 = r.gbps_total; best6_name = r.variant; ms6 = r.ms_total;
            }
        }
        std::puts(std::format("best M=1 variant {} at {:.1f} GB/s = {:.0f}% of raw read "
                             "(design section 15 P2 wants >= 80%)",
                             best_name, best, 100.0 * best / ceiling).c_str());
        std::puts(std::format("best M=6 variant {} at {:.1f} GB/s effective "
                             "({:.3f} ms for six tokens = {:.3f} ms/token)",
                             best6_name, best6, ms6, ms6 / 6.0).c_str());
        // design 9.4 needs T_layer for the MoE part to derive the lookahead d.
        for (const Row& r : rows)
            if (r.m == 1 && r.variant == best_name)
                std::puts(std::format("T_layer(MoE, M=1) = {:.3f} ms -> d >= T_io/T_layer "
                                      "= 4.0 / {:.3f} = {:.1f} layers of lookahead",
                                      r.ms_total, r.ms_total, 4.0 / r.ms_total).c_str());
    }

    // --- partial dispatch (design §7.9 "compute the experts that arrived
    // first") -------------------------------------------------------------
    // The same seven experts, once as one A+B pair over a seven-entry slot list
    // and once as two pairs over a 3- and a 4-entry list. The weight bytes are
    // identical; what the split costs is two more dispatches, two more global
    // barriers, and a second pass over x and h.
    {
        std::puts("\n== partial dispatch (design section 7.9) ==");
        for (uint32_t m : {1u, 6u}) {
            // The winner of the M sweep, so the split is priced against the
            // configuration decode would actually run.
            gpu::MoeSpec sp = (m == 1) ? gpu::MoeSpec{1, 32, 32, 0, 0, 1, 0}
                                       : gpu::MoeSpec{6, 16, 32, 0, 0, 2, 4};
            if (m != 1) sp.x_mode_b = 3;
            gpu::MoeDims dims = dims_base;
            gpu::MoeRunner runner;
            if (auto r = runner.create(dev, alloc, shader_dir, sp, dims); !r) {
                std::puts(r.error().str().c_str());
                continue;
            }
            std::memset(runner.pointer_table(), 0,
                        runner.pointer_table_entries() * sizeof(uint64_t));
            for (uint32_t L = 0; L < layout::kTotalLogicalLayers; ++L)
                std::memcpy(runner.pointer_table()
                                + size_t(L) * experts_per_layer * kExpertPartCount,
                            estore.pointer_table()
                                + size_t(L) * layout::kRoutedExperts * kExpertPartCount,
                            size_t(layout::kRoutedExperts) * kExpertPartCount * sizeof(uint64_t));
            for (uint32_t sIdx = 0; sIdx < o.slots; ++sIdx) runner.ids()[sIdx] = sIdx;
            for (uint32_t i = 0; i < m * o.slots; ++i) runner.route_weights()[i] = 1.0f / o.slots;
            uint32_t seed = 12345;
            for (uint32_t i = 0; i < m * layout::kHiddenSize; ++i) {
                seed = seed * 1664525u + 1013904223u;
                runner.x_fp16()[i] =
                    cpu::float_to_fp16((float(seed >> 8) / float(1u << 24) - 0.5f) * 0.1f);
            }
            // Nine measurements that have to be comparable with each other,
            // taken over a couple of seconds during which the part's clocks
            // move (section 2: 7% between runs, 8% between sections of one
            // run). Measuring all the repeats of one configuration before
            // starting the next would put each of them in a different thermal
            // state and the differences here are 1-3%, so the loop is
            // round-robin over the configurations and each keeps its own best.
            struct Cfg { uint32_t lo, hi; bool acc; gpu::MoePhase ph; const char* what; };
            const uint32_t cut = 3;
            const Cfg cfgs[] = {
                {0, o.slots, false, gpu::MoePhase::Both,       "whole pair"},
                {0, o.slots, false, gpu::MoePhase::GateUpOnly, "whole A"},
                {0, o.slots, false, gpu::MoePhase::DownOnly,   "whole B"},
                {0, cut,     false, gpu::MoePhase::GateUpOnly, "A lo"},
                {cut, o.slots, false, gpu::MoePhase::GateUpOnly, "A hi"},
                {0, cut,     false, gpu::MoePhase::DownOnly,   "B lo"},
                {cut, o.slots, true,  gpu::MoePhase::DownOnly, "B hi"},
                {0, cut,     false, gpu::MoePhase::Both,       "pair lo"},
                {cut, o.slots, true,  gpu::MoePhase::Both,     "pair hi"},
            };
            constexpr size_t kCfg = sizeof(cfgs) / sizeof(cfgs[0]);
            double best[kCfg];
            for (size_t i = 0; i < kCfg; ++i) best[i] = 1e9;
            auto run_cfg = [&](const Cfg& c) -> double {
                for (uint32_t i = c.lo; i < c.hi; ++i) runner.slot_list()[i - c.lo] = i;
                runner.set_list_count(c.hi - c.lo);
                runner.set_accumulate(c.acc);
                auto t = runner.run(o.iters, c.ph);
                return t ? t->seconds_total * 1e3 : 1e9;
            };
            for (const Cfg& c : cfgs) (void)run_cfg(c);         // one warm pass
            for (uint32_t r = 0; r < o.repeats + 2; ++r)
                for (size_t i = 0; i < kCfg; ++i)
                    best[i] = std::min(best[i], run_cfg(cfgs[i]));

            const double whole = best[0], whole_a = best[1], whole_b = best[2];
            const double a_lo = best[3], a_hi = best[4];
            const double b_lo = best[5], b_hi = best[6];
            // Schedule 1 (what P2 v0.1 section 6.2 measured): every arrival
            // group is a full A+B pair, the second accumulating into y. Timed
            // as pairs, because the A->B barrier inside a pair is part of what
            // the split costs.
            const double pairs = best[7] + best[8];
            // Schedule 2: split only dispatch A -- whose work is exactly
            // proportional to the slots it is handed -- and run dispatch B once
            // at the end over the whole list. B needs every slot's h anyway, so
            // nothing is lost by waiting, and the result is bit-identical to
            // the one-shot run (tests/test_gpu_moe.cpp).
            const double deferred = a_lo + a_hi + whole_b;
            std::puts(std::format("  M={}  {}  one pair over {} slots {:.3f} ms "
                                  "(A {:.3f} + B {:.3f} = {:.3f})",
                                  m, sp.name(), o.slots, whole, whole_a, whole_b,
                                  whole_a + whole_b).c_str());
            std::puts(std::format("        A {} + {} slots  {:.3f} + {:.3f} = {:.3f} ms "
                                  "({:+.3f} vs A whole)",
                                  cut, o.slots - cut, a_lo, a_hi, a_lo + a_hi,
                                  a_lo + a_hi - whole_a).c_str());
            std::puts(std::format("        B {} + {} slots  {:.3f} + {:.3f} = {:.3f} ms "
                                  "({:+.3f} vs B whole)",
                                  cut, o.slots - cut, b_lo, b_hi, b_lo + b_hi,
                                  b_lo + b_hi - whole_b).c_str());
            std::puts(std::format("        two A+B pairs      {:.3f} + {:.3f} = {:.3f} ms  "
                                  "({:+.3f} ms, {:+.1f}%)",
                                  best[7], best[8], pairs, pairs - whole,
                                  100.0 * (pairs - whole) / whole).c_str());
            std::puts(std::format("        split A, one B     {:.3f} ms  "
                                  "({:+.3f} ms, {:+.1f}%)",
                                  deferred, deferred - whole,
                                  100.0 * (deferred - whole) / whole).c_str());
        }
    }

    if (!o.csv.empty()) {
        std::FILE* f = std::fopen(o.csv.c_str(), "wb");
        if (f) {
            std::fprintf(f, "section,variant,m,lanes_per_row,rows_per_lane,subgroup,decode_mode,"
                            "h_precision,x_mode,h_quant,fp8_path,fp8_slots,slots,layer_cycle,"
                            "iters,path,bytes_a,bytes_b,gbps_a,gbps_b,gbps_total,pct_ceiling,"
                            "ms_a,ms_b,ms_total,ms_per_token,ms_wall,record_us,submit_us,"
                            "raw_read_gbps,launch_us\n");
            for (const Row& r : rows) {
                if (r.variant.empty()) continue;
                std::fprintf(f, "%s,%s,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%s,"
                                "%llu,%llu,%.3f,%.3f,%.3f,%.2f,"
                                "%.4f,%.4f,%.4f,%.5f,%.4f,%.3f,%.3f,%.3f,%.3f\n",
                             r.section, r.variant.c_str(), r.m, r.lanes, r.rpl, r.subgroup,
                             r.decode, r.hprec, r.xmode, r.hquant, r.fp8, r.fp8_slots,
                             r.list_count, o.layer_cycle, o.iters, path_name,
                             static_cast<unsigned long long>(r.bytes_a),
                             static_cast<unsigned long long>(r.bytes_b),
                             r.gbps_a, r.gbps_b, r.gbps_total,
                             ceiling > 0 ? 100.0 * r.gbps_total / ceiling : 0.0,
                             r.ms_a, r.ms_b, r.ms_total, r.ms_per_token, r.ms_wall,
                             r.record_us, r.submit_us, ceiling, launch_us);
            }
            std::fclose(f);
            std::puts(std::format("wrote {}", o.csv).c_str());
        }
    }

    for (SharedExpert& sh : shared) alloc.free(sh.buf);
    io.stop();
    return 0;
}
