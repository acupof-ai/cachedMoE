// Shared bring-up for the P5 feasibility probes (Track W, docs/plan_p5.md §5).
//
// A probe is a question that costs minutes and can close a week of kernel
// work. fleet-mi300x's microbench (g) is the model: one probe retired two
// separate optimisations there, each of which would otherwise have been
// discovered expensive. docs/STATUS.md §3 items 10, 11 and 17 are the same
// shape, discovered the expensive way.
//
// These probes are written to be RUN BY THE GPU OWNER on a quiet machine. The
// track that wrote them never started a GPU process (another agent owns the
// device, and a freeze happened when several ran at once), so what is
// guaranteed here is that they compile, that their shaders pass slangc +
// spirv-val, and that every loop in them is bounded. The exact commands and
// the expected shape of the output are in docs/plan_p5.md §5.
//
// Ownership/threading: one instance, one thread, RAII teardown in reverse.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/log.h"
#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/descriptor.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/pipeline.h"

namespace deepmoe::probe {

// Device + allocator + command pool + a two-slot query pool: everything a
// probe needs before it can dispatch anything.
struct Gpu {
    gpu::Device          device;
    gpu::MemoryAllocator alloc;
    gpu::CommandPool     pool;
    gpu::CommandBuffer   cmd{};
    gpu::QueryPool       queries;
    bool                 timed = false;
    std::string          shader_dir;

    Result<void> create(MemoryPath path = MemoryPath::Auto) {
        if (auto r = device.create(); !r) return r;
        if (auto r = alloc.init(device, path); !r) return r;
        if (auto r = pool.create(device); !r) return r;
        auto cb = pool.acquire();
        if (!cb) return std::unexpected(cb.error());
        cmd = *cb;
        timed = queries.create(device, 2).has_value() && device.caps().timestamp_valid_bits > 0;
        shader_dir = gpu::default_shader_dir();
        if (!timed)
            log_warn("probe: no GPU timestamps on this queue; falling back to wall clock");
        return {};
    }

    void destroy() {
        queries.destroy();
        pool.destroy();
        alloc.shutdown();
        device.destroy();
    }

    ~Gpu() { destroy(); }
};

// Runs one already-recorded command buffer and returns the best of `reps`
// GPU-timed spans (wall clock when the queue has no timestamps).
inline Result<double> best_seconds(Gpu& g, gpu::CommandBuffer& cmd, uint32_t reps = 3) {
    double best = 1e30;
    for (uint32_t i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (auto r = gpu::submit_and_wait(g.device, cmd); !r) return std::unexpected(r.error());
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (g.timed)
            if (auto q = g.queries.elapsed_seconds(0, 1); q && *q > 0.0) s = *q;
        if (s < best) best = s;
    }
    return best;
}

// A probe's CSV row. Probes print a table and, with --csv, append the same
// numbers to bench/results/ so a later session can diff them.
struct Csv {
    FILE* f = nullptr;

    void open(const std::string& path, const char* header) {
        if (path.empty()) return;
#if defined(_WIN32)
        (void)fopen_s(&f, path.c_str(), "wb");
#else
        f = std::fopen(path.c_str(), "wb");
#endif
        if (f) std::fprintf(f, "%s\n", header);
    }
    template <class... A>
    void row(const char* fmt, A... a) {
        if (f) { std::fprintf(f, fmt, a...); std::fputc('\n', f); }
    }
    void close(const std::string& path) {
        if (!f) return;
        std::fclose(f);
        f = nullptr;
        std::printf("-> %s\n", path.c_str());
    }
    ~Csv() { if (f) std::fclose(f); }
};

inline std::string arg_after(int argc, char** argv, const char* flag, const char* dflt) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return dflt;
}
inline uint64_t arg_u64(int argc, char** argv, const char* flag, uint64_t dflt) {
    const std::string v = arg_after(argc, argv, flag, "");
    return v.empty() ? dflt : std::strtoull(v.c_str(), nullptr, 10);
}
inline bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

}  // namespace deepmoe::probe
