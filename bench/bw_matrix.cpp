// Bandwidth matrix -- the other half of P-1 (design §9.2, §3.3).
//
// The question is not "how fast is LPDDR5X" but "how much of the 256 GB/s does
// each consumer get, alone and concurrently, on each of the two unified-memory
// paths":
//
//   CPU alone       multi-threaded streaming read
//   GPU alone       a raw-read compute shader over the same buffer
//   CPU + GPU       both at once -- the number design §8's cost model needs
//
// and each of those under path A (DEVICE_LOCAL|HOST_VISIBLE, CPU writes) and
// path B (host memory imported with VK_EXT_external_memory_host). design §3.3
// says to run the whole matrix at VGM = 64 GB and at the BIOS minimum and pick
// the better bandwidth x capacity product.
//
// The CPU half is implemented. The GPU half needs gpu/vulkan/memory.h, which is
// a P2 stub, so it reports Unimplemented rather than guessing.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "core/log.h"
#include "core/profiler.h"
#include "core/status.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"

using namespace deepmoe;

namespace {

struct Options {
    uint64_t buffer_bytes = 4ull << 30;
    uint32_t max_threads  = 0;      // 0 = hardware_concurrency
    uint32_t repeats      = 3;
    bool     run_gpu      = true;
};

// Streaming read: sum 64-byte-strided loads so the compiler cannot elide the
// traffic and the prefetcher sees a clean sequential stream.
uint64_t stream_read(const std::byte* p, size_t bytes) {
    uint64_t acc = 0;
    const auto* q = reinterpret_cast<const uint64_t*>(p);
    const size_t n = bytes / 8;
    for (size_t i = 0; i < n; i += 8) acc += q[i];      // one 64 B line per step
    return acc;
}

struct BwResult {
    uint32_t threads;
    double   gbps;
    double   seconds;
};

BwResult cpu_read_bandwidth(const AlignedBuffer& buf, uint32_t threads, uint32_t repeats) {
    const size_t bytes = buf.size();
    const size_t per   = bytes / threads;
    std::atomic<uint64_t> sink{0};
    double best = 0.0, best_secs = 0.0;

    for (uint32_t r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (uint32_t t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const size_t n = (t == threads - 1) ? bytes - per * t : per;
                sink.fetch_add(stream_read(buf.data() + per * t, n), std::memory_order_relaxed);
            });
        }
        for (auto& th : pool) th.join();
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        const double gbps = bytes / 1e9 / secs;
        if (gbps > best) { best = gbps; best_secs = secs; }
    }
    return BwResult{threads, best, best_secs};
}

int usage() {
    std::puts(
        "bw_matrix -- design section 9.2 / 3.3, the P-1 bandwidth matrix\n"
        "  --size-gb N    buffer size (default 4)\n"
        "  --threads N    maximum CPU thread count to sweep to (default: all cores)\n"
        "  --repeats N    repeats per point, best kept (default 3)\n"
        "  --no-gpu       skip the Vulkan half\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    set_log_level(LogLevel::Warn);
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string_view {
            if (i + 1 >= argc) { std::fputs("missing value\n", stderr); std::exit(2); }
            return argv[++i];
        };
        if (a == "--size-gb")      o.buffer_bytes = uint64_t(std::atoll(std::string(next()).c_str())) << 30;
        else if (a == "--threads") o.max_threads = static_cast<uint32_t>(std::atoi(std::string(next()).c_str()));
        else if (a == "--repeats") o.repeats = static_cast<uint32_t>(std::atoi(std::string(next()).c_str()));
        else if (a == "--no-gpu")  o.run_gpu = false;
        else return usage();
    }
    if (o.max_threads == 0) o.max_threads = std::max(1u, std::thread::hardware_concurrency());

    std::puts(std::format("buffer {:.2f} GB, up to {} threads, {} repeats\n",
                          o.buffer_bytes / 1e9, o.max_threads, o.repeats).c_str());

    AlignedBuffer buf;
    if (!buf.reset(static_cast<size_t>(o.buffer_bytes), kPageSize)) {
        std::fputs("cannot allocate the buffer; lower --size-gb\n", stderr);
        return 1;
    }
    // Touch every page so the measurement is not of the page-fault handler.
    std::memset(buf.data(), 1, buf.size());

    std::puts("== CPU streaming read (design section 3.3 path A/B denominator) ==");
    std::puts("threads      GB/s   seconds");
    std::puts("-----------------------------");
    double peak = 0.0;
    uint32_t peak_threads = 1;
    for (uint32_t t = 1; t <= o.max_threads; t = (t < 4 ? t + 1 : t * 2)) {
        const BwResult r = cpu_read_bandwidth(buf, t, o.repeats);
        if (r.gbps > peak) { peak = r.gbps; peak_threads = t; }
        std::puts(std::format("{:>7} {:>9.2f} {:>9.4f}", r.threads, r.gbps, r.seconds).c_str());
    }
    std::puts(std::format("\npeak {:.2f} GB/s at {} threads "
                          "(design section 1.1 quotes ~256 GB/s theoretical, "
                          "section 3.1 assumes 200 GB/s effective)",
                          peak, peak_threads).c_str());
    // The number design §3.1 actually turns into a TPS bound.
    std::puts(std::format("  -> 8.5 GB of resident weights per token = {:.1f} ms at this rate",
                          8.5 / peak * 1000.0).c_str());

    if (!o.run_gpu) return 0;

    std::puts("\n== GPU read bandwidth ==");
    gpu::Device dev;
    if (auto r = dev.create(); !r) {
        std::puts(std::format("vulkan unavailable: {}", r.error().str()).c_str());
        return 0;
    }
    std::puts(std::format("device: {}", dev.caps().device_name).c_str());

    for (MemoryPath path : {MemoryPath::DeviceLocalHostVisible, MemoryPath::ExternalMemoryHost}) {
        const char* name = path == MemoryPath::DeviceLocalHostVisible
                             ? "A device-local|host-visible" : "B external_memory_host";
        gpu::MemoryAllocator alloc;
        auto init = alloc.init(dev, path);
        if (!init) {
            std::puts(std::format("  path {}: {}", name, init.error().str()).c_str());
            continue;
        }
        if (auto t = alloc.chosen_memory_type(); t)
            std::puts(std::format("  path {}: memory type {} (heap {})", name, t->index, t->heap_index).c_str());
        // TODO(design §9.2): allocate a 2 GiB slab, run the raw-read shader
        // (the upper bound every kernel of design §7.1 is measured against),
        // then repeat with the CPU loop running concurrently. Needs
        // MemoryAllocator::allocate and a dispatch path -- both P2.
        auto b = alloc.allocate(1ull << 30, true, true);
        std::puts(std::format("  path {}: {}", name,
                              b ? "allocated" : b.error().str()).c_str());
    }
    std::puts("\nGPU and concurrent CPU+GPU points need the P2 Vulkan memory and dispatch paths "
              "(design section 7); see the TODOs in gpu/vulkan/memory.cpp.");
    return 0;
}
