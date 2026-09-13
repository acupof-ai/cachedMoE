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
    uint32_t    m = 0, lanes = 0, subgroup = 0, decode = 0, hprec = 0;
    double      gbps_a = 0, gbps_b = 0, gbps_total = 0;
    double      ms_a = 0, ms_b = 0, ms_total = 0, ms_wall = 0;
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
        variants.push_back(gpu::MoeSpec{1, 32, 32, 1, 0});
    } else {
        for (uint32_t lanes : {16u, 32u, 64u})
            for (uint32_t sg : {32u, 64u})
                variants.push_back(gpu::MoeSpec{1, lanes, sg, 1, 0});
        for (uint32_t dec : {0u, 2u})
            variants.push_back(gpu::MoeSpec{1, 32, 32, dec, 0});
        variants.push_back(gpu::MoeSpec{1, 32, 32, 1, 1});      // fp32 h
        for (uint32_t lanes : {16u, 32u, 64u})
            variants.push_back(gpu::MoeSpec{6, lanes, 32, 1, 0});
        variants.push_back(gpu::MoeSpec{6, 32, 64, 1, 0});
    }

    gpu::MoeDims dims;
    dims.layer       = 0;
    dims.slots       = o.slots;
    dims.layer_cycle = o.layer_cycle;

    const std::string shader_dir = gpu::default_shader_dir();
    std::vector<Row> rows;

    std::puts("\n== MoE kernels (design section 7.9) ==");
    std::puts("variant                    A GB/s   B GB/s   A+B GB/s   A ms    B ms   A+B ms");
    std::puts("--------------------------------------------------------------------------------");

    for (const gpu::MoeSpec& v : variants) {
        gpu::MoeRunner runner;
        if (auto r = runner.create(dev, alloc, shader_dir, v, dims); !r) {
            std::puts(std::format("{:<24} {}", v.name(), r.error().str()).c_str());
            continue;
        }
        std::memcpy(runner.pointer_table(), estore.pointer_table(), estore.pointer_table_bytes());
        for (uint32_t s = 0; s < o.slots; ++s) { runner.ids()[s] = s; runner.slot_list()[s] = s; }
        runner.set_list_count(o.slots);
        for (uint32_t i = 0; i < v.m * o.slots; ++i) runner.route_weights()[i] = 1.0f / o.slots;
        // A deterministic, well-scaled x: the numbers only have to be realistic
        // in magnitude, correctness lives in tests/test_gpu_moe.cpp.
        uint32_t seed = 12345;
        for (uint32_t i = 0; i < v.m * layout::kHiddenSize; ++i) {
            seed = seed * 1664525u + 1013904223u;
            const float u = float(seed >> 8) / float(1u << 24) - 0.5f;
            runner.x_fp16()[i] = cpu::float_to_fp16(u * 0.1f);
        }

        if (auto r = runner.run(2); !r) {
            std::puts(std::format("{:<24} warmup failed: {}", v.name(), r.error().str()).c_str());
            continue;
        }
        auto both = runner.run(o.iters);
        auto only_a = runner.run(o.iters, gpu::MoePhase::GateUpOnly);
        auto only_b = runner.run(o.iters, gpu::MoePhase::DownOnly);
        if (!both || !only_a || !only_b) {
            std::puts(std::format("{:<24} run failed", v.name()).c_str());
            continue;
        }

        Row row;
        row.variant = v.name();
        row.m = v.m; row.lanes = v.lanes_per_row; row.subgroup = v.subgroup_size;
        row.decode = v.decode_mode; row.hprec = v.h_precision;
        row.ms_a     = only_a->seconds_total * 1e3;
        row.ms_b     = only_b->seconds_total * 1e3;
        row.ms_total = both->seconds_total * 1e3;
        row.ms_wall  = both->wall_seconds / o.iters * 1e3;
        row.gbps_a     = double(runner.bytes_dispatch_a()) / 1e9 / only_a->seconds_total;
        row.gbps_b     = double(runner.bytes_dispatch_b()) / 1e9 / only_b->seconds_total;
        row.gbps_total = double(runner.bytes_per_iteration()) / 1e9 / both->seconds_total;
        rows.push_back(row);
        std::puts(std::format("{:<24} {:>8.1f} {:>8.1f} {:>10.1f} {:>7.3f} {:>7.3f} {:>8.3f}",
                              row.variant, row.gbps_a, row.gbps_b, row.gbps_total,
                              row.ms_a, row.ms_b, row.ms_total).c_str());
    }

    // --- the ceiling ---------------------------------------------------------
    std::puts("\n== raw-read ceiling over the same memory (design section 7.1 rule 2) ==");
    // Match the MoE working set so the MALL sees the same reuse.
    const uint64_t per_iter = uint64_t(o.slots) *
        (2 * (uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 2 +
              uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 32) +
         (uint64_t(layout::kHiddenSize) * layout::kMoeIntermediate / 2 +
          uint64_t(layout::kHiddenSize) * layout::kMoeIntermediate / 32));
    double ceiling = 0.0;
    double launch_us = 0.0;
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
        if (auto r = raw.run(*buf, want, passes); r) {
            ceiling = std::max(ceiling, r->gbps);
            std::puts(std::format("  {:>5} workgroups  {:.1f} GB/s over {:.1f} MB x {} passes ({})",
                                  groups, r->gbps, want / 1e6, passes,
                                  r->gpu_timed ? "gpu timestamps" : "wall clock").c_str());
        } else {
            std::puts(std::format("  {:>5} workgroups: {}", groups, r.error().str()).c_str());
        }
        if (groups == 320u) {
            if (auto e = raw.run_empty(*buf, 512); e)
                launch_us = e->seconds / 512 * 1e6;
        }
        alloc.free(*buf);
    }
    std::puts(std::format("\nraw-read ceiling {:.1f} GB/s; per-dispatch launch + global barrier "
                          "{:.2f} us (design section 3.4 assumed 5-20)",
                          ceiling, launch_us).c_str());
    if (ceiling > 0) {
        double best = 0; std::string best_name;
        for (const Row& r : rows) if (r.gbps_total > best) { best = r.gbps_total; best_name = r.variant; }
        std::puts(std::format("best MoE variant {} at {:.1f} GB/s = {:.0f}% of raw read "
                             "(design section 15 P2 wants >= 80%)",
                             best_name, best, 100.0 * best / ceiling).c_str());
        // design §9.4 needs T_layer for the MoE part to derive the lookahead d.
        for (const Row& r : rows)
            if (r.m == 1 && r.variant == best_name)
                std::puts(std::format("T_layer(MoE, M=1) = {:.3f} ms -> d >= T_io/T_layer "
                                      "= 4.0 / {:.3f} = {:.1f} layers of lookahead",
                                      r.ms_total, r.ms_total, 4.0 / r.ms_total).c_str());
    }

    if (!o.csv.empty()) {
        std::FILE* f = std::fopen(o.csv.c_str(), "wb");
        if (f) {
            std::fprintf(f, "variant,m,lanes_per_row,subgroup,decode_mode,h_precision,"
                            "slots,layer_cycle,iters,path,"
                            "bytes_a,bytes_b,gbps_a,gbps_b,gbps_total,"
                            "ms_a,ms_b,ms_total,ms_wall,raw_read_gbps,launch_us\n");
            const uint64_t ba = uint64_t(o.slots) * 2 *
                (uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 2 +
                 uint64_t(layout::kMoeIntermediate) * layout::kHiddenSize / 32);
            const uint64_t bb = per_iter - ba;
            for (const Row& r : rows)
                std::fprintf(f, "%s,%u,%u,%u,%u,%u,%u,%u,%u,%s,%llu,%llu,"
                                "%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f\n",
                             r.variant.c_str(), r.m, r.lanes, r.subgroup, r.decode, r.hprec,
                             o.slots, o.layer_cycle, o.iters, path_name,
                             static_cast<unsigned long long>(ba),
                             static_cast<unsigned long long>(bb),
                             r.gbps_a, r.gbps_b, r.gbps_total,
                             r.ms_a, r.ms_b, r.ms_total, r.ms_wall, ceiling, launch_us);
            std::fclose(f);
            std::puts(std::format("wrote {}", o.csv).c_str());
        }
    }

    io.stop();
    return 0;
}
