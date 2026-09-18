// io_dst_bench -- Track Q2 experiment E1 (docs/p4_p0_queue.md §next).
//
// Q1 measured the engine moving 3.55 GB/s of P0 misses while bench/nvme_bench
// reported 5.07-5.16 GB/s on "the same drive" at any request size. Two things
// differ between those two measurements and Q1 separated neither:
//
//   1. the DESTINATION. nvme_bench reads into AlignedBuffer (ordinary pageable
//      host RAM); the runtime reads into an expert slot, which on path A is
//      DEVICE_LOCAL|HOST_VISIBLE, coherent, UNCACHED Vulkan memory.
//   2. the FILE. nvme_bench's default test file lands in %TEMP% -- on C: --
//      while the runtime reads the safetensors shards on D:.
//
// This bench holds everything else fixed (same IoEngine, same chunking, same
// priority class, same request size) and varies only the destination, on
// whatever file it is pointed at. Point it at a shard on D: and the drive
// confound goes away too.
//
//   io_dst_bench --file PATH [--req-kb N] [--chunk-kb N] [--qd LIST]
//                [--dst LIST] [--reads N] [--csv FILE] [--copy]
//
// --dst is a comma-separated subset of:
//   ram      AlignedBuffer, ordinary pageable host memory (what nvme_bench uses)
//   pinned   VirtualAlloc + VirtualLock, non-pageable host memory
//   patha    a mapped DEVICE_LOCAL|HOST_VISIBLE Vulkan allocation (the runtime)
//   pathb    VirtualAlloc host memory imported with VK_EXT_external_memory_host
//   stage    Track S1: reads land in a path B (imported host) ring and a
//            vkCmdCopyBuffer moves each completed request into a path A slot.
//            This is the "staged fill" of docs/p4_p0_queue.md §16: pay the
//            cheap 70 us submit instead of path A's 704 us, and buy the bytes
//            back with a UMA-speed GPU copy. The reported GB/s is END TO END --
//            the wall clock covers the reads AND the copies -- so it is
//            directly comparable with the `patha` row.
//
// --copy additionally times the three ways a staging fix would move one expert
// (18.8 MB) into path A memory: a CPU memcpy from pinned host memory, and a
// vkCmdCopyBuffer from path B (both as a standalone submit and as recorded
// work, so a copy folded into an existing command buffer can be priced too).
// --gpu-copy is a synonym for adding `stage` to --dst.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <format>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "core/align.h"
#include "core/config.h"
#include "core/log.h"
#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "model/layout.h"
#include "storage/backend.h"
#include "storage/file.h"
#include "storage/io_engine.h"

using namespace deepmoe;
using namespace deepmoe::storage;

namespace {

struct Options {
    std::string file;
    uint32_t req_kb   = 9184;        // one manifest run of an expert (8.97 MiB)
    uint32_t chunk_kb = 4096;        // the runtime's P0 chunk
    std::vector<uint32_t> qd{4, 8, 16};
    std::vector<std::string> dst{"ram", "pinned", "patha", "pathb"};
    uint32_t reads = 64;
    bool     copy  = false;
    std::string csv;
};

std::vector<uint32_t> parse_u32(std::string_view s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = s.find(',', i);
        const auto tok = s.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i);
        if (!tok.empty()) out.push_back(uint32_t(std::atoi(std::string(tok).c_str())));
        if (j == std::string_view::npos) break;
        i = j + 1;
    }
    return out;
}
std::vector<std::string> parse_str(std::string_view s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = s.find(',', i);
        const auto tok = s.substr(i, j == std::string_view::npos ? std::string_view::npos : j - i);
        if (!tok.empty()) out.emplace_back(tok);
        if (j == std::string_view::npos) break;
        i = j + 1;
    }
    return out;
}

// --- destinations ------------------------------------------------------------
// Every kind hands back `count` sector-aligned pointers of `bytes` each and a
// note describing what it actually got.

struct Destination {
    std::vector<void*> slots;
    std::string        note;
    virtual ~Destination() = default;
};

struct RamDestination final : Destination {
    std::vector<AlignedBuffer> bufs;
};

struct PinnedDestination final : Destination {
    void*  base = nullptr;
    size_t bytes = 0;
    ~PinnedDestination() override {
#if defined(_WIN32)
        if (base) { ::VirtualUnlock(base, bytes); ::VirtualFree(base, 0, MEM_RELEASE); }
#endif
    }
};

struct VulkanDestination final : Destination {
    gpu::MemoryAllocator* alloc = nullptr;
    gpu::GpuBuffer        buf{};
    ~VulkanDestination() override { if (alloc && buf.valid()) alloc->free(buf); }
};

Result<std::unique_ptr<Destination>> make_ram(uint32_t count, uint64_t bytes) {
    auto d = std::make_unique<RamDestination>();
    d->bufs.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!d->bufs[i].reset(size_t(bytes), kPageSize))
            return fail(Err::ResourceExhausted, "AlignedBuffer");
        d->slots.push_back(d->bufs[i].data());
    }
    d->note = "pageable host (AlignedBuffer)";
    return std::unique_ptr<Destination>(std::move(d));
}

Result<std::unique_ptr<Destination>> make_pinned(uint32_t count, uint64_t bytes) {
#if defined(_WIN32)
    auto d = std::make_unique<PinnedDestination>();
    d->bytes = size_t(align_up(bytes, kPageSize) * count);
    d->base = ::VirtualAlloc(nullptr, d->bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!d->base) return fail(Err::ResourceExhausted, "VirtualAlloc", ::GetLastError());
    // The default working-set ceiling is far below a QD's worth of 9 MiB
    // buffers, so raise it before locking or VirtualLock fails with 1453.
    SIZE_T lo = 0, hi = 0;
    if (::GetProcessWorkingSetSize(::GetCurrentProcess(), &lo, &hi))
        ::SetProcessWorkingSetSize(::GetCurrentProcess(), lo + d->bytes + (64u << 20),
                                   hi + d->bytes + (64u << 20));
    if (::VirtualLock(d->base, d->bytes)) {
        d->note = "VirtualAlloc + VirtualLock (non-pageable)";
    } else {
        d->note = std::format("VirtualAlloc, VirtualLock FAILED ({}) -- pageable", ::GetLastError());
    }
    // Touch every page so the first read is not also a soft fault.
    std::memset(d->base, 0, d->bytes);
    for (uint32_t i = 0; i < count; ++i)
        d->slots.push_back(static_cast<std::byte*>(d->base) + uint64_t(i) * align_up(bytes, kPageSize));
    return std::unique_ptr<Destination>(std::move(d));
#else
    (void)count; (void)bytes;
    return fail(Err::Unavailable, "pinned is Windows-only here");
#endif
}

Result<std::unique_ptr<VulkanDestination>> make_vulkan_v(gpu::MemoryAllocator& alloc, MemoryPath path,
                                                        uint32_t count, uint64_t bytes) {
    auto d = std::make_unique<VulkanDestination>();
    d->alloc = &alloc;
    const uint64_t stride = align_up(bytes, kPageSize);
    auto b = (path == MemoryPath::DeviceLocalHostVisible)
                 ? alloc.allocate_slab(stride * count)
                 : alloc.allocate_imported(stride * count, /*device_address=*/true,
                                           /*try_large_pages=*/false);
    if (!b) return std::unexpected(b.error());
    d->buf = *b;
    if (!d->buf.host_ptr) return fail(Err::Internal, "allocation is not mapped");
    if (!is_aligned(d->buf.host_ptr, kPageSize))
        return fail(Err::Internal, "allocation is not sector-aligned");
    for (uint32_t i = 0; i < count; ++i)
        d->slots.push_back(static_cast<std::byte*>(d->buf.host_ptr) + uint64_t(i) * stride);
    d->note = std::format("memory type {}{}", d->buf.memory_type,
                          d->buf.imported ? ", imported" : ", mapped");
    return d;
}

Result<std::unique_ptr<Destination>> make_vulkan(gpu::MemoryAllocator& alloc, MemoryPath path,
                                                 uint32_t count, uint64_t bytes) {
    auto d = make_vulkan_v(alloc, path, count, bytes);
    if (!d) return std::unexpected(d.error());
    return std::unique_ptr<Destination>(std::move(*d));
}

// --- Track S1: the GPU half of a staged fill ---------------------------------
// One thread owns the command pool: Vulkan queue submission is not thread-safe
// and the runtime has exactly one GPU submit thread (docs/architecture.md), so
// modelling the copies as a serialised stream is the honest shape. It pulls a
// completed slot off the queue and copies that slot out of the path B ring into
// its path A slot. A slot is not handed back to the reader until its copy has
// retired -- the "evictions of a slot with a pending copy must wait" rule.
#if defined(DEEPMOE_ENABLE_VULKAN)
struct CopyRing {
    gpu::Device*       dev = nullptr;
    gpu::CommandPool   pool;
    gpu::CommandBuffer cb;
    VkBuffer src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    uint64_t stride = 0, bytes = 0;

    std::mutex              m;
    std::condition_variable cv;
    std::deque<uint32_t>    q;
    std::vector<uint8_t>    busy;
    bool                    quit = false;
    std::thread             th;

    uint64_t copies = 0;
    double   copy_ms_sum = 0, copy_ms_max = 0;
    bool     failed = false;
    Status   err{Err::Ok};

    Result<void> start(gpu::Device& d, VkBuffer s, VkBuffer t, uint64_t stride_, uint64_t bytes_,
                       uint32_t slots) {
        dev = &d; src = s; dst = t; stride = stride_; bytes = bytes_;
        busy.assign(slots, 0);
        if (auto r = pool.create(d); !r) return r;
        auto c = pool.acquire();
        if (!c) return std::unexpected(c.error());
        cb = *c;
        th = std::thread([this] { run(); });
        return {};
    }

    void run() {
        for (;;) {
            uint32_t slot = 0;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [this] { return quit || !q.empty(); });
                if (q.empty()) return;
                slot = q.front();
                q.pop_front();
            }
            const auto t0 = Clock::now();
            VkBufferCopy region{};
            region.srcOffset = uint64_t(slot) * stride;
            region.dstOffset = uint64_t(slot) * stride;
            region.size      = bytes;
            Status st{Err::Ok};
            if (auto r = cb.begin(); !r) st = r.error();
            if (st.code == Err::Ok) {
                vkCmdCopyBuffer(cb.handle(), src, dst, 1, &region);
                if (auto r = cb.end(); !r) st = r.error();
            }
            if (st.code == Err::Ok) { if (auto r = gpu::submit_and_wait(*dev, cb); !r) st = r.error(); }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            {
                std::lock_guard<std::mutex> lk(m);
                ++copies;
                copy_ms_sum += ms;
                copy_ms_max = std::max(copy_ms_max, ms);
                busy[slot] = 0;
                if (st.code != Err::Ok && !failed) { failed = true; err = st; }
            }
            cv.notify_all();
        }
    }

    // Reader side: block until this slot's previous copy has retired.
    void take(uint32_t slot) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return busy[slot] == 0 || failed; });
        busy[slot] = 1;
    }
    void completed(uint32_t slot) {
        { std::lock_guard<std::mutex> lk(m); q.push_back(slot); }
        cv.notify_all();
    }
    void drain() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return failed || (q.empty() && std::all_of(busy.begin(), busy.end(),
                                                                    [](uint8_t b) { return b == 0; })); });
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(m); quit = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
        pool.destroy();
    }
    ~CopyRing() { stop(); }
};
#endif

// --- one measurement point ---------------------------------------------------

struct Point {
    std::string dst;
    uint32_t req_kb = 0, chunk_kb = 0, qd = 0;
    uint64_t bytes = 0;
    double   seconds = 0, gbps = 0, mean_ms = 0, max_ms = 0;
    double   qd_at_issue = 0;
    double   submit_us = 0;      // Backend::submit(), i.e. ReadFile, per chunk
    // Track S1: the staged-fill columns. `copy_ms` is wall time per
    // vkCmdCopyBuffer including its own submit and wait; zero for every other
    // destination.
    double   copy_ms = 0, copy_ms_max = 0;
    uint64_t copies = 0;
    std::string note;
};

Result<Point> measure(const std::string& path, const std::string& kind,
                      gpu::MemoryAllocator* alloc_a, gpu::MemoryAllocator* alloc_b,
                      gpu::Device* dev,
                      uint32_t req_kb, uint32_t chunk_kb, uint32_t qd, uint32_t reads) {
    auto opened = File::open_read(path, true);
    if (!opened) return std::unexpected(opened.error());
    const File& f = *opened;

    const uint64_t req_bytes = align_up(uint64_t(req_kb) << 10, kPageSize);
    if (req_bytes > f.size()) return fail(Err::InvalidArgument, "request larger than the file");

    // Exactly the runtime's shape: a P0 request of `req_bytes` split into
    // `chunk_kb` chunks, at most `qd` of them in flight.
    IoConfig cfg;
    cfg.chunk_bytes        = uint32_t(align_up(uint64_t(chunk_kb) << 10, kPageSize));
    cfg.max_inflight_ops   = qd;
    cfg.max_inflight_bytes = std::min<uint64_t>(uint64_t(qd) * cfg.chunk_bytes, 512ull << 20);
    cfg.completion_threads = 2;
    cfg.unbuffered         = true;

    const uint32_t bufs   = std::max(qd, 1u);
    const uint64_t stride = align_up(req_bytes, kPageSize);
    std::unique_ptr<Destination> dst;
    bool staged = false;
#if defined(DEEPMOE_ENABLE_VULKAN)
    std::unique_ptr<VulkanDestination> stage_sink;
    CopyRing ring;
#endif
    if (kind == "ram")         { auto d = make_ram(bufs, req_bytes);    if (!d) return std::unexpected(d.error()); dst = std::move(*d); }
    else if (kind == "pinned") { auto d = make_pinned(bufs, req_bytes); if (!d) return std::unexpected(d.error()); dst = std::move(*d); }
    else if (kind == "patha") {
        if (!alloc_a) return fail(Err::Unavailable, "path A allocator is not available");
        auto d = make_vulkan(*alloc_a, MemoryPath::DeviceLocalHostVisible, bufs, req_bytes);
        if (!d) return std::unexpected(d.error());
        dst = std::move(*d);
    } else if (kind == "pathb") {
        if (!alloc_b) return fail(Err::Unavailable, "path B allocator is not available");
        auto d = make_vulkan(*alloc_b, MemoryPath::ExternalMemoryHost, bufs, req_bytes);
        if (!d) return std::unexpected(d.error());
        dst = std::move(*d);
    } else if (kind == "stage") {
#if defined(DEEPMOE_ENABLE_VULKAN)
        if (!alloc_a || !alloc_b || !dev)
            return fail(Err::Unavailable, "stage needs both allocators and a device");
        auto sv = make_vulkan_v(*alloc_b, MemoryPath::ExternalMemoryHost, bufs, req_bytes);
        if (!sv) return std::unexpected(sv.error());
        auto tv = make_vulkan_v(*alloc_a, MemoryPath::DeviceLocalHostVisible, bufs, req_bytes);
        if (!tv) return std::unexpected(tv.error());
        VkBuffer src_buf = (*sv)->buf.buffer;
        stage_sink = std::move(*tv);
        (*sv)->note = std::format("ring {} x {:.2f} MiB path B -> path A (GPU copy)",
                                  bufs, double(req_bytes) / 1048576.0);
        dst = std::unique_ptr<Destination>(std::move(*sv));
        if (auto r = ring.start(*dev, src_buf, stage_sink->buf.buffer, stride, req_bytes, bufs); !r)
            return std::unexpected(r.error());
        staged = true;
#else
        return fail(Err::Unavailable, "stage needs a Vulkan build");
#endif
    } else {
        return fail(Err::InvalidArgument, std::format("unknown destination '{}'", kind));
    }

    auto backend = make_default_backend(cfg);
    if (!backend) return std::unexpected(backend.error());
    IoEngine engine;
    if (auto r = engine.start(std::move(*backend), cfg); !r) return std::unexpected(r.error());

    const uint64_t span = f.size() - req_bytes;
    std::mt19937_64 rng(0xC0FFEEull ^ (uint64_t(req_kb) << 32) ^ qd);
    std::uniform_int_distribution<uint64_t> pick(0, span / kPageSize);

    engine.reset_stats();
    std::atomic<uint32_t> failures{0};
    Status first_error{Err::Ok};

    // One request per buffer in flight, throttled to `qd` requests the way
    // nvme_bench does, so the independent variable really is the destination.
    const uint32_t req_depth = std::max(1u, qd * cfg.chunk_bytes / uint32_t(req_bytes) + 1u);
    const auto t0 = Clock::now();
    for (uint32_t i = 0; i < reads; ++i) {
        const uint32_t slot = i % bufs;
        while (engine.queued_requests() >= req_depth && failures.load() == 0)
            std::this_thread::yield();
#if defined(DEEPMOE_ENABLE_VULKAN)
        // A slot whose copy has not retired is not reusable: the read would
        // overwrite bytes the GPU is still moving.
        if (staged) ring.take(slot);
#endif
        IoRequest r;
        r.priority = IoPriority::BlockingMiss;
        r.file     = &f;
        r.file_off = align_down(pick(rng) * kPageSize);
        r.bytes    = req_bytes;
        r.dst      = dst->slots[slot];
        auto id = engine.submit(r, [&, slot](const IoResult& res) {
            if (!res.ok() && failures.fetch_add(1, std::memory_order_relaxed) == 0)
                first_error = res.status;
#if defined(DEEPMOE_ENABLE_VULKAN)
            if (staged) ring.completed(slot);
#else
            (void)slot;
#endif
        });
        if (!id) {
#if defined(DEEPMOE_ENABLE_VULKAN)
            if (staged) ring.stop();
#endif
            engine.stop();
            return std::unexpected(id.error());
        }
    }
    engine.drain();
#if defined(DEEPMOE_ENABLE_VULKAN)
    if (staged) ring.drain();
#endif
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    const IoStats st = engine.stats();
    engine.stop();
    if (failures.load())
        return fail(Err::Io, std::format("{} request(s) failed: {}", failures.load(), first_error.str()));

    Point p;
    p.dst = kind; p.req_kb = req_kb; p.chunk_kb = chunk_kb; p.qd = qd;
    p.bytes = st.bytes_completed;
    p.seconds = secs;
    p.gbps = secs > 0 ? p.bytes / 1e9 / secs : 0.0;
    p.mean_ms = st.mean_latency_ms();
    p.max_ms  = st.latency_ns_max / 1e6;
    p.qd_at_issue = st.p0_chunks_issued ? double(st.p0_qd_at_issue_sum) / double(st.p0_chunks_issued) : 0.0;
    p.submit_us = st.disp_submit_mean_us();
#if defined(DEEPMOE_ENABLE_VULKAN)
    if (staged) {
        if (ring.failed) { ring.stop(); return std::unexpected(ring.err); }
        p.copies     = ring.copies;
        p.copy_ms    = ring.copies ? ring.copy_ms_sum / double(ring.copies) : 0.0;
        p.copy_ms_max = ring.copy_ms_max;
        ring.stop();
    }
#endif
    p.note = dst->note;
    return p;
}

// --- the staging copies ------------------------------------------------------

void measure_copies(gpu::MemoryAllocator* alloc_a, gpu::MemoryAllocator* alloc_b, gpu::Device* dev) {
    const uint64_t n = layout::kExpertBytes;
    auto src = make_pinned(1, n);
    if (!src) { std::puts(std::format("  staging source: {}", src.error().str()).c_str()); return; }
    if (!alloc_a) { std::puts("  path A allocator unavailable"); return; }
    auto dstv = make_vulkan_v(*alloc_a, MemoryPath::DeviceLocalHostVisible, 1, n);
    if (!dstv) { std::puts(std::format("  path A slab: {}", dstv.error().str()).c_str()); return; }

    void* s = (*src)->slots[0];
    void* d = (*dstv)->slots[0];
    double best = 1e9;
    for (int i = 0; i < 5; ++i) {
        const auto t = Clock::now();
        std::memcpy(d, s, size_t(n));
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
        best = std::min(best, ms);
    }
    std::puts(std::format("  CPU memcpy pinned -> path A: {:.2f} ms for {:.1f} MB ({:.1f} GB/s)",
                          best, n / 1e6, n / 1e9 / (best / 1e3)).c_str());

#if defined(DEEPMOE_ENABLE_VULKAN)
    // Track S1: the other half of the staging price. `submit` is a standalone
    // vkCmdCopyBuffer + vkQueueSubmit2 + vkQueueWaitIdle, which is what the
    // bench's `stage` rows pay; `record` is the same copy with the submit
    // amortised over `kBatch` regions, which is what folding the copy into the
    // layer's existing MoE command buffer would pay.
    if (!alloc_b || !dev) { std::puts("  path B allocator / device unavailable"); return; }
    constexpr uint32_t kBatch = 8;
    auto srcv = make_vulkan_v(*alloc_b, MemoryPath::ExternalMemoryHost, kBatch, n);
    if (!srcv) { std::puts(std::format("  path B ring: {}", srcv.error().str()).c_str()); return; }
    auto sinkv = make_vulkan_v(*alloc_a, MemoryPath::DeviceLocalHostVisible, kBatch, n);
    if (!sinkv) { std::puts(std::format("  path A ring: {}", sinkv.error().str()).c_str()); return; }
    const uint64_t stride = align_up(n, kPageSize);

    gpu::CommandPool pool;
    if (auto r = pool.create(*dev); !r) { std::puts(std::format("  pool: {}", r.error().str()).c_str()); return; }
    auto cbr = pool.acquire();
    if (!cbr) { std::puts(std::format("  cmdbuf: {}", cbr.error().str()).c_str()); return; }
    gpu::CommandBuffer cb = *cbr;

    auto run = [&](uint32_t regions) -> double {
        std::vector<VkBufferCopy> rs(regions);
        for (uint32_t i = 0; i < regions; ++i) {
            rs[i].srcOffset = uint64_t(i) * stride;
            rs[i].dstOffset = uint64_t(i) * stride;
            rs[i].size      = n;
        }
        double b = 1e9;
        for (int it = 0; it < 5; ++it) {
            const auto t = Clock::now();
            if (auto r = cb.begin(); !r) return -1.0;
            vkCmdCopyBuffer(cb.handle(), (*srcv)->buf.buffer, (*sinkv)->buf.buffer, regions, rs.data());
            if (auto r = cb.end(); !r) return -1.0;
            if (auto r = gpu::submit_and_wait(*dev, cb); !r) return -1.0;
            b = std::min(b, std::chrono::duration<double, std::milli>(Clock::now() - t).count());
        }
        return b;
    };
    const double one   = run(1);
    const double batch = run(kBatch);
    if (one < 0 || batch < 0) { std::puts("  vkCmdCopyBuffer failed"); return; }
    std::puts(std::format("  vkCmdCopyBuffer path B -> path A, own submit: {:.3f} ms for {:.1f} MB ({:.1f} GB/s)",
                          one, n / 1e6, n / 1e9 / (one / 1e3)).c_str());
    std::puts(std::format("  vkCmdCopyBuffer path B -> path A, {} per submit: {:.3f} ms each ({:.1f} GB/s)"
                          " -- submit overhead {:.3f} ms",
                          kBatch, batch / kBatch, n / 1e9 / (batch / kBatch / 1e3),
                          one - batch / kBatch).c_str());
#else
    (void)alloc_b; (void)dev;
#endif
}

int usage() {
    std::puts(
        "io_dst_bench -- Track Q2 E1: does the destination memory cost the drive its speed?\n"
        "  --file PATH     file to read (default: a shard under DEEPMOE_MODEL_DIR)\n"
        "  --req-kb N      request size in KiB (default 9184 = one expert run)\n"
        "  --chunk-kb N    chunk size in KiB (default 4096, the runtime's P0 chunk)\n"
        "  --qd LIST       chunk queue depths (default 4,8,16)\n"
        "  --dst LIST      ram,pinned,patha,pathb,stage (default the first four)\n"
        "  --gpu-copy      shorthand for adding `stage` to --dst\n"
        "  --reads N       requests per point (default 64)\n"
        "  --copy          also time the staging copies of one expert\n"
        "  --csv FILE      write the table as CSV\n");
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
        if (a == "--file")           o.file = next();
        else if (a == "--req-kb")    o.req_kb = uint32_t(std::atoi(std::string(next()).c_str()));
        else if (a == "--chunk-kb")  o.chunk_kb = uint32_t(std::atoi(std::string(next()).c_str()));
        else if (a == "--qd")        o.qd = parse_u32(next());
        else if (a == "--dst")       o.dst = parse_str(next());
        else if (a == "--reads")     o.reads = uint32_t(std::atoi(std::string(next()).c_str()));
        else if (a == "--copy")      o.copy = true;
        else if (a == "--gpu-copy") {
            if (std::find(o.dst.begin(), o.dst.end(), "stage") == o.dst.end()) o.dst.emplace_back("stage");
        }
        else if (a == "--csv")       o.csv = next();
        else if (a == "-h" || a == "--help") return usage();
        else { std::fprintf(stderr, "unknown option %.*s\n", int(a.size()), a.data()); return usage(); }
    }
    if (o.file.empty()) {
        const char* md = std::getenv("DEEPMOE_MODEL_DIR");
        if (!md) { std::fputs("no --file and no DEEPMOE_MODEL_DIR\n", stderr); return 2; }
        o.file = std::string(md) + "/model-00020-of-00048.safetensors";
    }
    {
        auto probe = File::open_read(o.file, true);
        if (!probe) { std::fprintf(stderr, "cannot open %s: %s\n", o.file.c_str(), probe.error().str().c_str()); return 1; }
        std::puts(std::format("file       {} ({:.2f} GB, unbuffered {}, sector {} B)",
                              probe->path(), probe->size() / 1e9,
                              probe->unbuffered() ? "yes" : "no", probe->sector_size()).c_str());
    }

    // The Vulkan destinations need a device; the host ones do not, so a machine
    // without Vulkan still gets the ram/pinned columns.
    gpu::Device dev;
    gpu::MemoryAllocator alloc_a, alloc_b;
    gpu::MemoryAllocator* pa = nullptr;
    gpu::MemoryAllocator* pb = nullptr;
    gpu::Device* pdev = nullptr;
    const bool wants_gpu = std::any_of(o.dst.begin(), o.dst.end(),
                                       [](const std::string& s) {
                                           return s == "patha" || s == "pathb" || s == "stage";
                                       });
    if (wants_gpu || o.copy) {
        gpu::DeviceOptions dopts;
        if (auto r = dev.create(dopts); !r) {
            std::puts(std::format("vulkan unavailable: {}", r.error().str()).c_str());
        } else {
            std::puts(std::format("device     {}", dev.caps().device_name).c_str());
            pdev = &dev;
            if (alloc_a.init(dev, MemoryPath::DeviceLocalHostVisible)) pa = &alloc_a;
            if (alloc_b.init(dev, MemoryPath::ExternalMemoryHost))     pb = &alloc_b;
        }
    }

    std::puts(std::format("\nrequest {} KiB, chunk {} KiB, {} reads per point\n",
                          o.req_kb, o.chunk_kb, o.reads).c_str());
    std::puts("dst        qd      GB/s   mean_ms    max_ms   QD@issue  submit_us   copy_ms  note");
    std::puts("--------------------------------------------------------------------------------------------------");

    std::vector<Point> results;
    for (const std::string& k : o.dst) {
        for (uint32_t qd : o.qd) {
            auto p = measure(o.file, k, pa, pb, pdev, o.req_kb, o.chunk_kb, qd, o.reads);
            if (!p) {
                std::puts(std::format("{:<10} {:>3}   {}", k, qd, p.error().str()).c_str());
                continue;
            }
            results.push_back(*p);
            std::puts(std::format("{:<10} {:>3} {:>9.3f} {:>9.3f} {:>9.3f} {:>10.2f} {:>10.1f} {:>9.3f}  {}",
                                  p->dst, p->qd, p->gbps, p->mean_ms, p->max_ms,
                                  p->qd_at_issue, p->submit_us, p->copy_ms, p->note).c_str());
        }
    }

    if (o.copy) {
        std::puts("\nstaging copy of one expert (18.8 MB):");
        measure_copies(pa, pb, pdev);
    }

    if (!o.csv.empty()) {
        if (std::FILE* c = std::fopen(o.csv.c_str(), "wb")) {
            std::fputs("dst,req_kb,chunk_kb,qd,bytes,seconds,gbps,mean_ms,max_ms,qd_at_issue,"
                       "submit_us,copies,copy_ms,copy_ms_max,note\n", c);
            for (const Point& p : results)
                std::fputs(std::format("{},{},{},{},{},{:.6f},{:.4f},{:.4f},{:.4f},{:.2f},"
                                       "{:.1f},{},{:.4f},{:.4f},{}\n",
                                       p.dst, p.req_kb, p.chunk_kb, p.qd, p.bytes, p.seconds,
                                       p.gbps, p.mean_ms, p.max_ms, p.qd_at_issue,
                                       p.submit_us, p.copies, p.copy_ms, p.copy_ms_max, p.note).c_str(), c);
            std::fclose(c);
            std::puts(std::format("\nwrote {}", o.csv).c_str());
        }
    }
    return 0;
}
