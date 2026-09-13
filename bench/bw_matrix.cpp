// Bandwidth matrix -- the other half of P-1 (design §9.2, §3.3).
//
// The question is not "how fast is LPDDR5X" but "how much of the 256 GB/s does
// each consumer get, alone and concurrently, on each of the two unified-memory
// paths":
//
//   CPU alone       multi-threaded streaming read of ordinary host memory
//   GPU alone       gpu/shaders/rawread.slang over path A / path B / plain
//                   DEVICE_LOCAL memory -- the ceiling every kernel of design
//                   §7.1 rule 2 is scored against
//   CPU + GPU       both at once: the number design §8's cost model needs
//   CPU write       memcpy and non-temporal stores into path A's mapped
//                   device memory, which is what the IoEngine's NVMe reads do
//                   (design §9.6 zero-copy)
//
// design §3.3 says to run the whole matrix at VGM = 64 GB and at the BIOS
// minimum and pick the better bandwidth x capacity product. The VGM = 64 GB
// column is what this run produces; the minimum-VGM column needs a reboot.
//
// Output: a table on stdout and bench/results/bw_matrix.csv.
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

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "core/align.h"
#include "core/config.h"
#include "core/log.h"
#include "core/profiler.h"
#include "core/status.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/rawread.h"

using namespace deepmoe;

namespace {

struct Options {
    uint64_t cpu_bytes    = 4ull << 30;
    uint64_t gpu_bytes    = 1ull << 30;
    uint32_t max_threads  = 0;      // 0 = hardware_concurrency
    uint32_t repeats      = 3;
    uint32_t groups       = 320;    // rawread workgroups
    bool     run_gpu      = true;
    std::string csv       = "bench/results/bw_matrix.csv";
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

// Non-temporal store copy: what an NVMe landing buffer wants, since the bytes
// are never read by the CPU again (design §3.3 path A: CPU reads of that
// mapping are uncached and useless).
void copy_nontemporal(std::byte* dst, const std::byte* src, size_t bytes) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 64 <= bytes; i += 64)
        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + i),
                            _mm512_loadu_si512(reinterpret_cast<const void*>(src + i)));
    if (i < bytes) std::memcpy(dst + i, src + i, bytes - i);
    _mm_sfence();
#elif defined(__AVX__)
    size_t i = 0;
    for (; i + 32 <= bytes; i += 32)
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i),
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i)));
    if (i < bytes) std::memcpy(dst + i, src + i, bytes - i);
    _mm_sfence();
#else
    std::memcpy(dst, src, bytes);
#endif
}

struct BwResult {
    uint32_t threads = 0;
    double   gbps    = 0.0;
    double   seconds = 0.0;
};

BwResult cpu_read_bandwidth(const std::byte* data, size_t bytes, uint32_t threads,
                            uint32_t repeats) {
    const size_t per = bytes / threads;
    std::atomic<uint64_t> sink{0};
    BwResult best;
    best.threads = threads;

    for (uint32_t r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (uint32_t t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const size_t n = (t == threads - 1) ? bytes - per * t : per;
                sink.fetch_add(stream_read(data + per * t, n), std::memory_order_relaxed);
            });
        }
        for (auto& th : pool) th.join();
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        const double gbps = double(bytes) / 1e9 / secs;
        if (gbps > best.gbps) { best.gbps = gbps; best.seconds = secs; }
    }
    return best;
}

// A CPU load generator that runs until stopped and reports how many bytes it
// moved, so a concurrent GPU measurement can be attributed to one time window.
class CpuLoad {
public:
    CpuLoad(const std::byte* data, size_t bytes, uint32_t threads)
        : data_(data), bytes_(bytes), threads_(threads) {}

    void start() {
        stop_.store(false, std::memory_order_relaxed);
        const size_t per = bytes_ / threads_;
        for (uint32_t t = 0; t < threads_; ++t) {
            pool_.emplace_back([this, per, t] {
                const size_t n = (t == threads_ - 1) ? bytes_ - per * t : per;
                uint64_t acc = 0;
                while (!stop_.load(std::memory_order_relaxed)) {
                    acc += stream_read(data_ + per * t, n);
                    moved_.fetch_add(n, std::memory_order_relaxed);
                }
                sink_.fetch_add(acc, std::memory_order_relaxed);
            });
        }
    }
    uint64_t moved() const { return moved_.load(std::memory_order_relaxed); }
    void stop() {
        stop_.store(true, std::memory_order_relaxed);
        for (auto& t : pool_) t.join();
        pool_.clear();
    }

private:
    const std::byte* data_;
    size_t           bytes_;
    uint32_t         threads_;
    std::atomic<bool>     stop_{true};
    std::atomic<uint64_t> moved_{0};
    std::atomic<uint64_t> sink_{0};
    std::vector<std::thread> pool_;
};

struct Row {
    std::string consumer;     // cpu_read / gpu_read / cpu_write / ...
    std::string memory;       // host / path_a / path_b / device_local
    std::string concurrency;  // alone / with_gpu / with_cpu
    uint32_t    threads = 0;
    double      gbps = 0.0;
    std::string note;
};

std::vector<Row> rows;

void add(std::string consumer, std::string memory, std::string concurrency,
         uint32_t threads, double gbps, std::string note = {}) {
    rows.push_back(Row{std::move(consumer), std::move(memory), std::move(concurrency),
                       threads, gbps, std::move(note)});
}

int usage() {
    std::puts(
        "bw_matrix -- design section 9.2 / 3.3, the P-1 bandwidth matrix\n"
        "  --cpu-size-gb N  host buffer for the CPU sweep (default 4)\n"
        "  --size-gb N      GPU buffer per memory path (default 1)\n"
        "  --threads N      maximum CPU thread count to sweep to (default: all cores)\n"
        "  --repeats N      repeats per point, best kept (default 3)\n"
        "  --groups N       rawread workgroups (default 320)\n"
        "  --csv PATH       CSV output (default bench/results/bw_matrix.csv)\n"
        "  --no-gpu         skip the Vulkan half\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    set_log_level(LogLevel::Warn);
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fputs("missing value\n", stderr); std::exit(2); }
            return argv[++i];
        };
        if (a == "--cpu-size-gb")  o.cpu_bytes = uint64_t(std::atoll(next().c_str())) << 30;
        else if (a == "--size-gb") o.gpu_bytes = uint64_t(std::atoll(next().c_str())) << 30;
        else if (a == "--threads") o.max_threads = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--repeats") o.repeats = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--groups")  o.groups = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--csv")     o.csv = next();
        else if (a == "--no-gpu")  o.run_gpu = false;
        else return usage();
    }
    if (o.max_threads == 0) o.max_threads = std::max(1u, std::thread::hardware_concurrency());

    std::puts(std::format("host buffer {:.2f} GB, GPU buffer {:.2f} GB, up to {} threads, {} repeats\n",
                          o.cpu_bytes / 1e9, o.gpu_bytes / 1e9, o.max_threads, o.repeats).c_str());

    AlignedBuffer buf;
    if (!buf.reset(static_cast<size_t>(o.cpu_bytes), kPageSize)) {
        std::fputs("cannot allocate the buffer; lower --cpu-size-gb\n", stderr);
        return 1;
    }
    // Touch every page so the measurement is not of the page-fault handler.
    std::memset(buf.data(), 1, buf.size());

    std::puts("== CPU streaming read of ordinary host memory ==");
    std::puts("threads      GB/s   seconds");
    std::puts("-----------------------------");
    double peak = 0.0;
    uint32_t peak_threads = 1;
    for (uint32_t t = 1; t <= o.max_threads; t = (t < 4 ? t + 1 : t * 2)) {
        const BwResult r = cpu_read_bandwidth(buf.data(), buf.size(), t, o.repeats);
        if (r.gbps > peak) { peak = r.gbps; peak_threads = t; }
        std::puts(std::format("{:>7} {:>9.2f} {:>9.4f}", r.threads, r.gbps, r.seconds).c_str());
        add("cpu_read", "host", "alone", t, r.gbps);
    }
    std::puts(std::format("\npeak {:.2f} GB/s at {} threads "
                          "(design section 1.1 quotes ~256 GB/s theoretical, "
                          "section 3.1 assumes 200 GB/s effective)",
                          peak, peak_threads).c_str());
    std::puts(std::format("  -> 8.5 GB of resident weights per token = {:.1f} ms at this rate",
                          8.5 / peak * 1000.0).c_str());

    if (!o.run_gpu) return 0;

    std::puts("\n== GPU read bandwidth (gpu/shaders/rawread.slang) ==");
    gpu::Device dev;
    gpu::DeviceOptions dopts;
    dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
    if (auto r = dev.create(dopts); !r) {
        std::puts(std::format("vulkan unavailable: {}", r.error().str()).c_str());
        return 0;
    }
    std::puts(std::format("device: {}", dev.caps().device_name).c_str());
    for (const gpu::HeapInfo& h : dev.caps().heaps)
        std::puts(std::format("  heap[{}] {:.1f} GiB {}", h.index, h.bytes / 1073741824.0,
                              h.device_local ? "DEVICE_LOCAL" : "host").c_str());

    const std::string shader_dir = gpu::default_shader_dir();

    // The three memories of design §3.3, plus the pure-VRAM ceiling if the
    // driver exposes one.
    struct MemKind {
        const char* key;
        const char* label;
        MemoryPath  path;
        bool        large_pages;
        bool        device_local_only;
    };
    std::vector<MemKind> kinds = {
        {"path_a", "A  DEVICE_LOCAL|HOST_VISIBLE, mapped", MemoryPath::DeviceLocalHostVisible, false, false},
        {"path_b", "B  external_memory_host (VirtualAlloc)", MemoryPath::ExternalMemoryHost, false, false},
        {"path_b_lp", "B  external_memory_host (large pages)", MemoryPath::ExternalMemoryHost, true, false},
    };
    {
        gpu::MemoryAllocator probe;
        if (probe.init(dev, MemoryPath::DeviceLocalHostVisible)) {
            if (!probe.device_local_only_types().empty())
                kinds.push_back({"device_local", "   DEVICE_LOCAL only (no host access)",
                                 MemoryPath::DeviceLocalHostVisible, false, true});
            else
                std::puts("\n  (no DEVICE_LOCAL-only memory type on this device: on a UMA part "
                          "every device-local type is also host-visible, so path A *is* the ceiling)");
        }
    }

    std::puts("\nmemory                                    GB/s   alloc note");
    std::puts("--------------------------------------------------------------------------------");

    struct PathBw { std::string key; double alone = 0, with_cpu = 0, cpu_with_gpu = 0; };
    std::vector<PathBw> path_bw;
    double gpu_ceiling = 0.0;

    for (const MemKind& k : kinds) {
        gpu::MemoryAllocator alloc;
        if (auto r = alloc.init(dev, k.path); !r) {
            std::puts(std::format("{:<40} {}", k.label, r.error().str()).c_str());
            continue;
        }
        gpu::RawReadKernel raw;
        if (auto r = raw.create(dev, alloc, shader_dir, o.groups); !r) {
            std::puts(std::format("{:<40} {}", k.label, r.error().str()).c_str());
            continue;
        }
        const uint64_t want = raw.round_bytes(o.gpu_bytes);

        Result<gpu::GpuBuffer> b = fail(Err::Internal, "unset");
        std::string note;
        if (k.device_local_only) {
            b = alloc.allocate_from_type(want, alloc.device_local_only_types().front(),
                                         /*map=*/false, /*device_address=*/true);
            note = std::format("memory type {}", alloc.device_local_only_types().front());
        } else if (k.path == MemoryPath::ExternalMemoryHost) {
            auto host = gpu::alloc_host_pages(want, k.large_pages);
            if (!host) {
                std::puts(std::format("{:<40} {}", k.label, host.error().str()).c_str());
                continue;
            }
            note = host->note.empty() ? (host->large_pages ? "large pages" : "4 KiB pages")
                                      : host->note;
            if (k.large_pages && !host->large_pages) {
                std::puts(std::format("{:<40} skipped: {}", k.label, note).c_str());
                add("gpu_read", k.key, "alone", 0, 0.0, "large pages unavailable: " + note);
                gpu::free_host_pages(*host);
                continue;
            }
            b = alloc.import_host_memory(host->ptr, host->bytes, true);
            if (b) { b->host_alloc = host->ptr; b->large_pages = host->large_pages; }
            else gpu::free_host_pages(*host);
        } else {
            b = alloc.allocate_slab(want);
            if (auto t = alloc.chosen_memory_type(); t)
                note = std::format("memory type {} (heap {})", t->index, t->heap_index);
        }
        if (!b) {
            std::puts(std::format("{:<40} {}", k.label, b.error().str()).c_str());
            add("gpu_read", k.key, "alone", 0, 0.0, b.error().str());
            continue;
        }
        if (b->host_ptr) std::memset(b->host_ptr, 0x5A, static_cast<size_t>(want));

        (void)raw.run(*b, want, 1);   // warm up
        double best = 0.0;
        bool gpu_timed = false;
        for (uint32_t r = 0; r < o.repeats; ++r)
            if (auto res = raw.run(*b, want, 2); res) {
                best = std::max(best, res->gbps);
                gpu_timed = res->gpu_timed;
            }
        std::puts(std::format("{:<40} {:>7.1f}   {}{}", k.label, best, note,
                              gpu_timed ? "" : " [wall clock]").c_str());
        add("gpu_read", k.key, "alone", 0, best, note);
        gpu_ceiling = std::max(gpu_ceiling, best);

        PathBw pb;
        pb.key = k.key;
        pb.alone = best;

        // --- concurrency: CPU streaming host memory while the GPU streams this
        if (!k.large_pages) {
            CpuLoad load(buf.data(), buf.size(), peak_threads);
            load.start();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let it ramp
            const uint64_t c0 = load.moved();
            const auto t0 = Clock::now();
            auto res = raw.run(*b, want, 4);
            const double window = std::chrono::duration<double>(Clock::now() - t0).count();
            const uint64_t c1 = load.moved();
            load.stop();
            if (res) {
                pb.with_cpu = res->gbps;
                pb.cpu_with_gpu = double(c1 - c0) / 1e9 / window;
                std::puts(std::format("  + concurrent CPU read: GPU {:>6.1f} GB/s, "
                                      "CPU {:>6.1f} GB/s, sum {:.1f} GB/s "
                                      "({:.0f}% of the two alone)",
                                      pb.with_cpu, pb.cpu_with_gpu,
                                      pb.with_cpu + pb.cpu_with_gpu,
                                      100.0 * (pb.with_cpu + pb.cpu_with_gpu) / (pb.alone + peak)).c_str());
                add("gpu_read", k.key, "with_cpu", peak_threads, pb.with_cpu);
                add("cpu_read", "host", std::string("with_gpu_") + k.key, peak_threads, pb.cpu_with_gpu);
            }
        }

        // --- CPU write into this memory: the NVMe landing path of design §9.6
        if (b->host_ptr) {
            AlignedBuffer src;
            const size_t chunk = size_t(256) << 20;
            if (src.reset(chunk, kPageSize)) {
                std::memset(src.data(), 0x33, src.size());
                const size_t reps = static_cast<size_t>(std::min<uint64_t>(want / chunk, 4));
                for (int mode = 0; mode < 2; ++mode) {
                    auto* dst = static_cast<std::byte*>(b->host_ptr);
                    const auto t0 = Clock::now();
                    for (size_t r = 0; r < reps; ++r) {
                        if (mode == 0) std::memcpy(dst + r * chunk, src.data(), chunk);
                        else           copy_nontemporal(dst + r * chunk, src.data(), chunk);
                    }
                    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
                    const double gbps = double(chunk * reps) / 1e9 / secs;
                    const char* how = mode == 0 ? "memcpy" : "non-temporal stores";
                    std::puts(std::format("  + CPU write ({:<19}): {:>6.2f} GB/s "
                                          "({:.1f} ms for one 18.8 MB expert)",
                                          how, gbps, 18.8e6 / (gbps * 1e9) * 1e3).c_str());
                    add("cpu_write", k.key, "alone", 1, gbps, how);
                }
            }
        }

        path_bw.push_back(pb);
        alloc.free(*b);
    }

    // --- launch overhead, design §3.4 ----------------------------------------
    {
        gpu::MemoryAllocator alloc;
        if (alloc.init(dev, MemoryPath::DeviceLocalHostVisible)) {
            gpu::RawReadKernel raw;
            if (raw.create(dev, alloc, shader_dir, 1)) {
                if (auto b = alloc.allocate_slab(1u << 20)) {
                    (void)raw.run_empty(*b, 64);
                    if (auto e = raw.run_empty(*b, 1024)) {
                        const double us = e->seconds / 1024 * 1e6;
                        std::puts(std::format("\nper-dispatch launch + global barrier: {:.2f} us "
                                              "({} -- design section 3.4 assumed 5-20 us)",
                                              us, e->gpu_timed ? "gpu timestamps" : "wall clock").c_str());
                        add("dispatch_overhead", "path_a", "alone", 0, us, "microseconds, not GB/s");
                    }
                    alloc.free(*b);
                }
            }
        }
    }

    std::puts(std::format("\nGPU raw-read ceiling {:.1f} GB/s; CPU read peak {:.2f} GB/s.",
                          gpu_ceiling, peak).c_str());

    std::FILE* f = std::fopen(o.csv.c_str(), "wb");
    if (!f) {
        std::puts(std::format("cannot write {} (does bench/results/ exist?)", o.csv).c_str());
        return 0;
    }
    std::fprintf(f, "consumer,memory,concurrency,threads,gbps,note\n");
    for (const Row& r : rows)
        std::fprintf(f, "%s,%s,%s,%u,%.3f,%s\n", r.consumer.c_str(), r.memory.c_str(),
                     r.concurrency.c_str(), r.threads, r.gbps, r.note.c_str());
    std::fclose(f);
    std::puts(std::format("wrote {}", o.csv).c_str());
    return 0;
}
