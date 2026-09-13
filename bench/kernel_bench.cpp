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
        "  --quick           only the default variant\n");
    return 2;
}

struct Row {
    std::string variant;
    uint32_t    m = 0, lanes = 0, subgroup = 0, decode = 0, hprec = 0, rpl = 0;
    double      gbps_a = 0, gbps_b = 0, gbps_total = 0;
    double      ms_a = 0, ms_b = 0, ms_total = 0, ms_wall = 0;
    double      record_us = 0, submit_us = 0;
};

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

    // --- variants ------------------------------------------------------------
    std::vector<gpu::MoeSpec> variants;
    if (o.quick) {
        variants.push_back(gpu::MoeSpec{1, 32, 32, 0, 0, 1});
        variants.push_back(gpu::MoeSpec{6, 16, 64, 0, 0, 2});
    } else {
        // design §7.1's knobs at the two batch sizes that matter: M=1 is
        // decode, M=6 a full speculative verify batch. RowsPerLane > 1 costs
        // registers, so it is only swept where it can pay: at M=1 the lane has
        // registers to spare, at M=6 it does not.
        for (uint32_t m : {1u, 6u})
            for (uint32_t dec : {0u, 1u, 2u})
                for (uint32_t lanes : {16u, 32u, 64u})
                    for (uint32_t sg : {32u, 64u})
                        variants.push_back(gpu::MoeSpec{m, lanes, sg, dec, 0, 1});
        for (uint32_t rpl : {2u, 4u})
            for (uint32_t lanes : {16u, 32u, 64u})
                for (uint32_t sg : {32u, 64u})
                    variants.push_back(gpu::MoeSpec{1, lanes, sg, 0, 0, rpl});
        for (uint32_t lanes : {16u, 32u})
            for (uint32_t sg : {32u, 64u})
                variants.push_back(gpu::MoeSpec{6, lanes, sg, 0, 0, 2});
        // fp32 h doubles the h traffic; worth measuring only on the variants
        // that are otherwise competitive.
        for (uint32_t m : {1u, 6u})
            for (uint32_t lanes : {16u, 32u})
                for (uint32_t rpl : {1u, 2u})
                    variants.push_back(gpu::MoeSpec{m, lanes, 32, 0, 1, rpl});
    }

    gpu::MoeDims dims;
    dims.layer       = 0;
    dims.slots       = o.slots;
    dims.layer_cycle = o.layer_cycle;

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
            const gpu::MoeSpec& v = variants[vi];
            gpu::MoeRunner runner;
            if (auto r = runner.create(dev, alloc, shader_dir, v, dims); !r) {
                if (sweep == 0) std::puts(std::format("{:<24} {}", v.name(), r.error().str()).c_str());
                continue;
            }
            std::memcpy(runner.pointer_table(), estore.pointer_table(), estore.pointer_table_bytes());
            for (uint32_t s = 0; s < o.slots; ++s) { runner.ids()[s] = s; runner.slot_list()[s] = s; }
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
            row.m = v.m; row.lanes = v.lanes_per_row; row.subgroup = v.subgroup_size;
            row.decode = v.decode_mode; row.hprec = v.h_precision; row.rpl = v.rows_per_lane;
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
            if (row.gbps_total > rows[vi].gbps_total) rows[vi] = row;
        }
        std::puts(std::format("  sweep {}/{} done", sweep + 1, o.sweeps).c_str());
    }

    const double ceiling_after = measure_ceiling("after");

    std::puts("\nvariant                    A GB/s   B GB/s   A+B GB/s   A ms    B ms   A+B ms  rec us");
    std::puts("----------------------------------------------------------------------------------------");
    for (const Row& r : rows) {
        if (r.variant.empty()) continue;
        std::puts(std::format("{:<27} {:>8.1f} {:>8.1f} {:>10.1f} {:>7.3f} {:>7.3f} {:>8.3f} {:>7.2f}",
                              r.variant, r.gbps_a, r.gbps_b, r.gbps_total,
                              r.ms_a, r.ms_b, r.ms_total, r.record_us).c_str());
    }

    std::puts(std::format("\nraw-read ceiling {:.1f} GB/s before the sweep, {:.1f} GB/s after "
                          "(the spread is thermal drift, not kernel behaviour);\n"
                          "per-dispatch launch + global barrier {:.2f} us "
                          "(design section 3.4 assumed 5-20)",
                          ceiling_before, ceiling_after, launch_us).c_str());
    if (ceiling > 0) {
        double best = 0; std::string best_name;
        double best6 = 0; std::string best6_name; double ms6 = 0;
        for (const Row& r : rows) {
            if (r.m == 1 && r.gbps_total > best) { best = r.gbps_total; best_name = r.variant; }
            if (r.m == 6 && r.gbps_total > best6) { best6 = r.gbps_total; best6_name = r.variant; ms6 = r.ms_total; }
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

    if (!o.csv.empty()) {
        std::FILE* f = std::fopen(o.csv.c_str(), "wb");
        if (f) {
            std::fprintf(f, "variant,m,lanes_per_row,rows_per_lane,subgroup,decode_mode,h_precision,"
                            "slots,layer_cycle,iters,path,"
                            "bytes_a,bytes_b,gbps_a,gbps_b,gbps_total,"
                            "ms_a,ms_b,ms_total,ms_wall,raw_read_gbps,launch_us\n");
            const uint64_t ba = uint64_t(o.slots) * 2 *
                (uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 2 +
                 uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 32);
            const uint64_t bb = per_iter - ba;
            for (const Row& r : rows) {
                if (r.variant.empty()) continue;
                std::fprintf(f, "%s,%u,%u,%u,%u,%u,%u,%u,%u,%u,%s,%llu,%llu,"
                                "%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                             r.variant.c_str(), r.m, r.lanes, r.rpl, r.subgroup, r.decode, r.hprec,
                             o.slots, o.layer_cycle, o.iters, path_name,
                             static_cast<unsigned long long>(ba),
                             static_cast<unsigned long long>(bb),
                             r.gbps_a, r.gbps_b, r.gbps_total,
                             r.ms_a, r.ms_b, r.ms_total, r.ms_wall,
                             r.record_us, r.submit_us, ceiling, launch_us);
            }
            std::fclose(f);
            std::puts(std::format("wrote {}", o.csv).c_str());
        }
    }

    io.stop();
    return 0;
}
