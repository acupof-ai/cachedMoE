// How much expert cache does the GPU actually get, at the BIOS setting this
// machine is booted with right now?
//
// docs/kernel_p1.md §4.1 sized the expert cache from *reported* numbers: the
// 74.4 GiB device-local heap, the 63.6 GB Windows sees, minus guesses for the
// pin set and staging. Reported heap sizes on a UMA part are not a budget --
// heap 1 and heap 0 are the same LPDDR5X, the driver over-reports both, and
// WDDM decides what a commitment really costs only when the pages are touched.
//
// The user has ruled the BIOS VGM knob irrelevant: kernel_p1 §2.2/§2.3 measured
// path A and path B reading at the same ~216 GB/s, so the slab pool will span
// both heaps as-is and the only open question is the real allocatable maximum.
// This bench answers it by allocating until failure instead of asking:
//
//   Path A   DEVICE_LOCAL|HOST_VISIBLE slabs of --slab-gib (2 GiB = the
//            maxMemoryAllocationSize of design §1.1), mapped, with a device
//            address, every 4 KiB page written from the CPU so WDDM has to
//            commit it. After every slab: GlobalMemoryStatusEx. That delta is
//            the real question -- does a VGM carve-out commitment come for free
//            from the CPU's point of view, or does every path A byte also cost
//            a byte of CPU-visible RAM?
//   Path B   VirtualAlloc + VK_EXT_external_memory_host import, same drill,
//            plus a probe of imports larger than maxMemoryAllocationSize
//            (the limit is specified for allocations; whether the driver
//            enforces it on an *import* is an empirical question).
//   Mixed    path A to its ceiling, then path B on top of it: the grand total
//            the GPU can address simultaneously, with a raw-read pass over the
//            last slab of each kind to prove the bytes are real and still at
//            full bandwidth.
//
// Safety: this bench exists to push the machine to its allocation ceiling, so
// it must never push it into swap. Every slab is gated on GlobalMemoryStatusEx
// leaving at least --min-free-gib of available physical memory, and SIGINT sets
// a flag that both the allocation loop and the page-touch loop poll, after
// which everything is freed on the way out.
//
// Output: a table per phase on stdout plus bench/results/heap_capacity.csv.
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "core/log.h"
#include "core/profiler.h"
#include "core/status.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/rawread.h"

// docs/architecture.md §1.2 confines platform headers to storage/windows and
// storage/linux, with gpu/vulkan/memory.cpp as the documented exception. This
// bench is a second one, for the same kind of reason: "how much memory does
// Windows still think it has" is the measurement, and GlobalMemoryStatusEx is
// the only thing that answers it. Everything under the guard degrades to
// "unknown" elsewhere, and the bench still runs.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace deepmoe;

namespace {

constexpr uint64_t kGiB = 1ull << 30;

double gib(uint64_t bytes) { return double(bytes) / double(kGiB); }

// --- interruption ------------------------------------------------------------
// Ctrl+C must not leave 60 GiB committed and the machine swapping. The handler
// only sets a flag; the allocation loop and the page-touch loop poll it and
// unwind through the normal free path.
volatile std::sig_atomic_t g_interrupt = 0;
extern "C" void on_interrupt(int) { g_interrupt = 1; }
bool interrupted() { return g_interrupt != 0; }

// --- what the OS still thinks it has -----------------------------------------

struct MemStatus {
    bool     valid        = false;
    uint64_t total_phys   = 0;
    uint64_t avail_phys   = 0;
    uint64_t total_commit = 0;   // ullTotalPageFile: RAM + page file
    uint64_t avail_commit = 0;
};

MemStatus mem_status() {
    MemStatus m;
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) {
        m.valid        = true;
        m.total_phys   = s.ullTotalPhys;
        m.avail_phys   = s.ullAvailPhys;
        m.total_commit = s.ullTotalPageFile;
        m.avail_commit = s.ullAvailPageFile;
    }
#endif
    return m;
}

// The floor check. `need` is what the next slab would cost in the worst case
// (every byte of it landing in physical memory). Without a working
// GlobalMemoryStatusEx we cannot police the floor, so we say so once and let
// the allocation failure be the only stop condition.
bool floor_would_break(uint64_t need, uint64_t min_free, std::string& why) {
    const MemStatus m = mem_status();
    if (!m.valid) return false;
    if (m.avail_phys < need + min_free) {
        why = std::format("available physical {:.2f} GiB < slab {:.2f} + floor {:.2f} GiB",
                          gib(m.avail_phys), gib(need), gib(min_free));
        return true;
    }
    // The commit charge is the other way to wedge a Windows box: running the
    // page file dry stalls every process on the machine, not just this one.
    if (m.avail_commit && m.avail_commit < need + min_free) {
        why = std::format("available commit {:.2f} GiB < slab {:.2f} + floor {:.2f} GiB",
                          gib(m.avail_commit), gib(need), gib(min_free));
        return true;
    }
    return false;
}

// --- page touching -----------------------------------------------------------

// One store per 4 KiB page, which is what forces WDDM/the VMM to back the
// range: a mapped VkDeviceMemory or a MEM_COMMIT reservation costs nothing real
// until something writes to it, and an untouched sweep would "prove" a capacity
// that evaporates on first use. The store is volatile so it survives -O2, and
// it is a *write* because path A's mapping is write-combining -- CPU reads of
// it are uncached and pointlessly slow (design §3.3).
//
// Returns false if a Ctrl+C arrived partway through.
bool touch_pages(void* p, uint64_t bytes) {
    auto* base = static_cast<unsigned char*>(p);
    constexpr uint64_t kPollEvery = 64ull << 20;
    for (uint64_t off = 0; off < bytes; off += kPageSize) {
        *reinterpret_cast<volatile uint32_t*>(base + off) = static_cast<uint32_t>(off >> 12) + 1u;
        if ((off & (kPollEvery - 1)) == 0 && interrupted()) return false;
    }
    return true;
}

// --- CSV ---------------------------------------------------------------------

struct Row {
    std::string phase;         // path_a / path_b / oversize_import / mixed_a / mixed_b / summary
    uint32_t    index = 0;     // slab number within the phase, 0 for non-slab rows
    uint64_t    slab_bytes = 0;
    uint64_t    cum_bytes  = 0;   // cumulative within the phase
    double      alloc_ms = 0.0;
    double      touch_ms = 0.0;
    double      commit_gibps = 0.0;   // slab_bytes / touch_ms
    uint64_t    avail_phys = 0;
    uint64_t    avail_commit = 0;
    double      gpu_read_gbps = 0.0;  // only on the raw-read sanity rows
    std::string status;
};

std::vector<Row> g_rows;

void write_csv(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::puts(std::format("cannot write {} (does bench/results/ exist?)", path).c_str());
        return;
    }
    std::fprintf(f, "phase,index,slab_bytes,cum_bytes,alloc_ms,touch_ms,commit_gibps,"
                    "avail_phys_bytes,avail_commit_bytes,gpu_read_gbps,status\n");
    for (const Row& r : g_rows)
        std::fprintf(f, "%s,%u,%llu,%llu,%.3f,%.3f,%.3f,%llu,%llu,%.3f,%s\n",
                     r.phase.c_str(), r.index,
                     static_cast<unsigned long long>(r.slab_bytes),
                     static_cast<unsigned long long>(r.cum_bytes),
                     r.alloc_ms, r.touch_ms, r.commit_gibps,
                     static_cast<unsigned long long>(r.avail_phys),
                     static_cast<unsigned long long>(r.avail_commit),
                     r.gpu_read_gbps, r.status.c_str());
    std::fclose(f);
    std::puts(std::format("\nwrote {}", path).c_str());
}

// --- options -----------------------------------------------------------------

struct Options {
    uint64_t    slab_bytes   = 2 * kGiB;   // the maxMemoryAllocationSize of design §1.1
    uint64_t    max_bytes    = 0;          // 0 = no cap: stop only on failure or the floor
    uint64_t    min_free     = 6 * kGiB;
    uint32_t    groups       = 320;        // rawread workgroups, same as bw_matrix
    bool        run_a        = true;
    bool        run_b        = true;
    bool        run_mixed    = true;
    bool        run_oversize = true;
    std::string csv          = "bench/results/heap_capacity.csv";
};

int usage() {
    std::puts(
        "heap_capacity -- the real allocatable ceiling on both heaps, at the current BIOS setting\n"
        "  --slab-gib F      bytes per allocation (default 2, the maxMemoryAllocationSize)\n"
        "  --max-gib F       stop a sweep after this much (default 0 = only failure stops it)\n"
        "  --min-free-gib F  never let GlobalMemoryStatusEx availPhys drop below this (default 6)\n"
        "  --groups N        rawread workgroups for the sanity pass (default 320)\n"
        "  --skip-a          skip the path A sweep\n"
        "  --skip-b          skip the path B sweep\n"
        "  --skip-mixed      skip the A-then-B-on-top phase\n"
        "  --skip-oversize   skip the >maxMemoryAllocationSize import probe\n"
        "  --csv PATH        CSV output (default bench/results/heap_capacity.csv)\n"
        "\n"
        "Ctrl+C is safe: the sweep stops at the next slab boundary and frees everything.\n");
    return 2;
}

// --- one sweep ---------------------------------------------------------------

enum class Kind { PathA, PathB };

const char* kind_name(Kind k) { return k == Kind::PathA ? "A" : "B"; }

struct SweepResult {
    uint32_t    slabs = 0;
    uint64_t    bytes = 0;
    std::string stop_reason = "not run";
    MemStatus   before{};
    MemStatus   after{};
};

// Allocates `kind` slabs of o.slab_bytes into `live` until something says stop.
// `live` is the caller's -- the mixed phase keeps path A alive while path B
// runs on top of it, and the caller frees everything either way.
SweepResult run_sweep(const char* phase, Kind kind, gpu::MemoryAllocator& alloc,
                      std::vector<gpu::GpuBuffer>& live, const Options& o) {
    SweepResult res;
    res.before = mem_status();
    const MemStatus base = res.before;

    std::puts(std::format("\n== path {} sweep: {:.2f} GiB per allocation, floor {:.2f} GiB ==",
                          kind_name(kind), gib(o.slab_bytes), gib(o.min_free)).c_str());
    std::puts("  #   slab GiB    cum GiB   alloc ms   touch ms   commit GiB/s   availPhys GiB   "
              "dAvailPhys GiB   ratio   device address");
    std::puts("---------------------------------------------------------------------------------"
              "--------------------------------------");

    for (uint32_t i = 0;; ++i) {
        if (interrupted()) { res.stop_reason = "interrupted (Ctrl+C)"; break; }
        if (o.max_bytes && res.bytes + o.slab_bytes > o.max_bytes) {
            res.stop_reason = std::format("--max-gib {:.2f} reached", gib(o.max_bytes));
            break;
        }
        std::string why;
        if (floor_would_break(o.slab_bytes, o.min_free, why)) {
            res.stop_reason = "floor: " + why;
            break;
        }

        const auto t0 = Clock::now();
        Result<gpu::GpuBuffer> buf = fail(Err::Internal, "unset");
        if (kind == Kind::PathA)
            buf = alloc.allocate_slab(o.slab_bytes);          // map + device address
        else
            buf = alloc.allocate_imported(o.slab_bytes, /*device_address=*/true,
                                          /*try_large_pages=*/false);
        const double alloc_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        if (!buf) {
            // ResourceExhausted is what memory.cpp turns VK_ERROR_OUT_OF_*_MEMORY
            // and a failed VirtualAlloc into; the VkResult code is in the message.
            res.stop_reason = std::format("allocation {} failed: {}", i, buf.error().str());
            const MemStatus m = mem_status();
            g_rows.push_back(Row{phase, i, o.slab_bytes, res.bytes, alloc_ms, 0.0, 0.0,
                                 m.avail_phys, m.avail_commit, 0.0, buf.error().str()});
            break;
        }
        if (!buf->host_ptr) {
            alloc.free(*buf);
            res.stop_reason = "allocation succeeded but is not host-mapped; cannot commit it";
            break;
        }

        const auto t1 = Clock::now();
        const bool whole = touch_pages(buf->host_ptr, buf->bytes);
        const double touch_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

        live.push_back(*buf);
        res.bytes += buf->bytes;
        res.slabs += 1;

        const MemStatus m = mem_status();
        const double commit_gibps = touch_ms > 0 ? gib(buf->bytes) / (touch_ms / 1000.0) : 0.0;
        const int64_t d_avail = base.valid && m.valid
                                    ? static_cast<int64_t>(base.avail_phys) - static_cast<int64_t>(m.avail_phys)
                                    : 0;
        // The whole point of the phase: how much CPU-visible RAM did this
        // cumulative commitment actually cost? 0% means the carve-out is free
        // from Windows' view, 100% means every GPU byte is a lost CPU byte.
        const double ratio = res.bytes ? 100.0 * double(d_avail) / double(res.bytes) : 0.0;

        std::puts(std::format("{:>4} {:>10.2f} {:>10.2f} {:>10.1f} {:>10.1f} {:>14.2f} {:>15.2f} "
                              "{:>16.2f} {:>6.0f}%   {:#018x}",
                              i, gib(buf->bytes), gib(res.bytes), alloc_ms, touch_ms, commit_gibps,
                              m.valid ? gib(m.avail_phys) : 0.0,
                              double(d_avail) / double(kGiB), ratio,
                              static_cast<uint64_t>(buf->dev_addr))
                      .c_str());

        g_rows.push_back(Row{phase, i, buf->bytes, res.bytes, alloc_ms, touch_ms, commit_gibps,
                             m.avail_phys, m.avail_commit, 0.0,
                             whole ? "ok" : "touch interrupted"});

        if (!whole) { res.stop_reason = "interrupted (Ctrl+C) while committing pages"; break; }
    }

    res.after = mem_status();
    std::puts(std::format("path {}: {} slabs, {:.2f} GiB allocated and committed; stopped because {}",
                          kind_name(kind), res.slabs, gib(res.bytes), res.stop_reason).c_str());
    if (res.before.valid && res.after.valid && res.bytes) {
        const int64_t d = static_cast<int64_t>(res.before.avail_phys) -
                          static_cast<int64_t>(res.after.avail_phys);
        const double pct = 100.0 * double(d) / double(res.bytes);
        std::puts(std::format("  available physical memory fell {:.2f} GiB for {:.2f} GiB allocated "
                              "= {:.0f}% -- {}",
                              double(d) / double(kGiB), gib(res.bytes), pct,
                              pct < 25.0    ? "the commitment is (mostly) invisible to Windows: "
                                              "this capacity is free from the CPU's view"
                              : pct > 75.0  ? "every allocated byte costs a CPU-visible byte: "
                                              "this capacity competes with host RAM"
                                            : "partially carved out; see the per-slab ratio column")
                      .c_str());
    }
    g_rows.push_back(Row{std::string(phase) + "_total", res.slabs, o.slab_bytes, res.bytes, 0.0, 0.0,
                         0.0, res.after.avail_phys, res.after.avail_commit, 0.0, res.stop_reason});
    return res;
}

void free_all(gpu::MemoryAllocator& alloc, std::vector<gpu::GpuBuffer>& live) {
    for (gpu::GpuBuffer& b : live) alloc.free(b);
    live.clear();
}

// --- the >2 GiB import probe -------------------------------------------------
//
// maxMemoryAllocationSize is specified as a limit on vkAllocateMemory, and
// gpu/vulkan/memory.cpp enforces it on imports too (conservatively). Whether
// the *driver* enforces it on VK_EXT_external_memory_host is a different
// question, and a positive answer would cut the slab count -- and with it the
// descriptor and pointer-table bookkeeping of design §5.3 -- by whatever factor
// the real limit allows. So this probe goes around MemoryAllocator and talks to
// Vulkan directly. It deliberately does NOT touch the pages: the point is where
// the import stops being accepted, not another capacity sweep, and committing
// 16 GiB to find that out is exactly the kind of thing the floor forbids.

#if defined(DEEPMOE_ENABLE_VULKAN)

const char* vk_result_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS:                         return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY:        return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:   return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_TOO_MANY_OBJECTS:          return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_INITIALIZATION_FAILED:     return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_MEMORY_MAP_FAILED:         return "VK_ERROR_MEMORY_MAP_FAILED";
        default: break;
    }
    return "VkResult (see vulkan_core.h)";
}

struct ImportProbe {
    bool        memory_ok = false;
    bool        buffer_ok = false;
    bool        address_ok = false;
    std::string note;
};

ImportProbe probe_oversize_import(gpu::Device& dev, uint64_t bytes) {
    ImportProbe p;
    auto host = gpu::alloc_host_pages(bytes, /*try_large_pages=*/false);
    if (!host) { p.note = host.error().str(); return p; }

    VkDevice d = dev.handle();
    auto props_fn = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetDeviceProcAddr(d, "vkGetMemoryHostPointerPropertiesEXT"));
    if (!props_fn) {
        p.note = "vkGetMemoryHostPointerPropertiesEXT is missing";
        gpu::free_host_pages(*host);
        return p;
    }
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (VkResult r = props_fn(d, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                              host->ptr, &hp);
        r != VK_SUCCESS || hp.memoryTypeBits == 0) {
        p.note = std::format("vkGetMemoryHostPointerPropertiesEXT: {}", vk_result_name(r));
        gpu::free_host_pages(*host);
        return p;
    }

    // Try a buffer of the full size too. It may fail on maxBufferSize long
    // before the allocation does; that is a separate limit and worth knowing,
    // so the allocation is still attempted when the buffer is refused.
    VkExternalMemoryBufferCreateInfo emb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    emb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &emb};
    bci.size  = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    uint32_t type_bits = hp.memoryTypeBits;
    if (VkResult r = vkCreateBuffer(d, &bci, nullptr, &buffer); r == VK_SUCCESS) {
        p.buffer_ok = true;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(d, buffer, &req);
        type_bits &= req.memoryTypeBits;
    } else {
        buffer = VK_NULL_HANDLE;
        p.note = std::format("vkCreateBuffer({} B): {}; ", bytes, vk_result_name(r));
    }
    if (type_bits == 0) {
        p.note += "no memory type accepts both the host pointer and the buffer";
        if (buffer) vkDestroyBuffer(d, buffer, nullptr);
        gpu::free_host_pages(*host);
        return p;
    }
    uint32_t type_index = 0;
    while (!(type_bits & (1u << type_index))) ++type_index;

    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = host->ptr;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, &imp};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                             p.buffer_ok ? static_cast<void*>(&flags) : static_cast<void*>(&imp)};
    mai.allocationSize  = host->bytes;
    mai.memoryTypeIndex = type_index;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    const VkResult ar = vkAllocateMemory(d, &mai, nullptr, &memory);
    if (ar == VK_SUCCESS) {
        p.memory_ok = true;
        if (p.buffer_ok && vkBindBufferMemory(d, buffer, memory, 0) == VK_SUCCESS) {
            VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            ai.buffer = buffer;
            p.address_ok = vkGetBufferDeviceAddress(d, &ai) != 0;
        } else if (p.buffer_ok) {
            p.note += "vkBindBufferMemory failed";
        }
    } else {
        p.note += std::format("vkAllocateMemory(import {} B, type {}): {}", mai.allocationSize,
                              type_index, vk_result_name(ar));
    }

    if (memory) vkFreeMemory(d, memory, nullptr);
    if (buffer) vkDestroyBuffer(d, buffer, nullptr);
    gpu::free_host_pages(*host);
    return p;
}

void run_oversize_probe(gpu::Device& dev, const Options& o) {
    const uint64_t limit = dev.caps().max_memory_allocation_size;
    std::puts(std::format("\n== import sizes above maxMemoryAllocationSize ({:.2f} GiB) ==",
                          gib(limit)).c_str());
    std::puts("  Pages are NOT touched here: the question is where the driver refuses the import,\n"
              "  not how much can be committed. A success only proves the handle is accepted.");
    std::puts("   size GiB   memory   buffer   device addr   note");
    std::puts("------------------------------------------------------------------------------");

    for (uint64_t bytes = limit + kGiB; ; bytes *= 2) {
        if (interrupted()) break;
        if (o.max_bytes && bytes > o.max_bytes) break;
        std::string why;
        if (floor_would_break(bytes, o.min_free, why)) {
            std::puts(std::format("{:>11.2f}   stopped by the floor: {}", gib(bytes), why).c_str());
            g_rows.push_back(Row{"oversize_import", 0, bytes, 0, 0.0, 0.0, 0.0, 0, 0, 0.0,
                                 "floor: " + why});
            break;
        }
        const ImportProbe p = probe_oversize_import(dev, bytes);
        std::puts(std::format("{:>11.2f} {:>8} {:>8} {:>13}   {}", gib(bytes),
                              p.memory_ok ? "ok" : "FAIL", p.buffer_ok ? "ok" : "FAIL",
                              p.address_ok ? "ok" : "-", p.note)
                      .c_str());
        g_rows.push_back(Row{"oversize_import", 0, bytes, 0, 0.0, 0.0, 0.0, 0, 0, 0.0,
                             std::format("memory={} buffer={} addr={} {}",
                                         p.memory_ok ? "ok" : "fail", p.buffer_ok ? "ok" : "fail",
                                         p.address_ok ? "ok" : "no", p.note)});
        if (!p.memory_ok) {
            std::puts(std::format("  -> the import limit is between {:.2f} and {:.2f} GiB; "
                                  "maxMemoryAllocationSize applies to imports as well",
                                  gib(bytes / 2), gib(bytes)).c_str());
            break;
        }
        if (bytes >= 64 * kGiB) break;   // enough: nothing in design §5.3 wants a bigger slab
    }
}

#endif  // DEEPMOE_ENABLE_VULKAN

// A raw-read pass over one slab, to prove the last committed bytes are real and
// still arrive at the ~216 GB/s of kernel_p1 §2.2 rather than through some
// fallback path the driver silently picked when it ran out of the fast heap.
void sanity_read(gpu::RawReadKernel& raw, const gpu::GpuBuffer& buf, const char* label,
                 const char* phase) {
    if (!buf.valid()) return;
    const uint64_t want = raw.round_bytes(std::min<uint64_t>(buf.bytes, 1 * kGiB));
    if (want == 0) {
        std::puts(std::format("  {}: slab too small for {} workgroups", label, raw.groups()).c_str());
        return;
    }
    (void)raw.run(buf, want, 1);   // warm up
    auto r = raw.run(buf, want, 2);
    if (!r) {
        std::puts(std::format("  {}: raw read failed: {}", label, r.error().str()).c_str());
        g_rows.push_back(Row{phase, 0, buf.bytes, 0, 0.0, 0.0, 0.0, 0, 0, 0.0,
                             "rawread failed: " + r.error().str()});
        return;
    }
    std::puts(std::format("  {}: {:.1f} GB/s over {:.2f} GiB{}", label, r->gbps, gib(want),
                          r->gpu_timed ? "" : " [wall clock]").c_str());
    g_rows.push_back(Row{phase, 0, buf.bytes, 0, 0.0, 0.0, 0.0, 0, 0, r->gbps, "rawread"});
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
        auto as_bytes = [](const std::string& s) {
            return align_down(static_cast<uint64_t>(std::atof(s.c_str()) * double(kGiB)), kPageSize);
        };
        if (a == "--slab-gib")           o.slab_bytes = as_bytes(next());
        else if (a == "--max-gib")       o.max_bytes  = as_bytes(next());
        else if (a == "--min-free-gib")  o.min_free   = as_bytes(next());
        else if (a == "--groups")        o.groups     = static_cast<uint32_t>(std::atoi(next().c_str()));
        else if (a == "--skip-a")        o.run_a = false;
        else if (a == "--skip-b")        o.run_b = false;
        else if (a == "--skip-mixed")    o.run_mixed = false;
        else if (a == "--skip-oversize") o.run_oversize = false;
        else if (a == "--csv")           o.csv = next();
        else return usage();
    }
    if (o.slab_bytes == 0) { std::fputs("--slab-gib must be positive\n", stderr); return 2; }
    if (o.groups == 0) o.groups = 320;

    std::signal(SIGINT, on_interrupt);
    std::signal(SIGTERM, on_interrupt);
#if defined(SIGBREAK)
    std::signal(SIGBREAK, on_interrupt);
#endif

    // --- 1. what the driver *says* -------------------------------------------
    gpu::Device dev;
    gpu::DeviceOptions dopts;
    dopts.enable_validation = std::getenv("VK_INSTANCE_LAYERS") != nullptr;
    if (auto r = dev.create(dopts); !r) {
        std::puts(std::format("vulkan unavailable: {}", r.error().str()).c_str());
        return 1;
    }
    const gpu::DeviceCaps& caps = dev.caps();
    std::fputs(caps.to_string().c_str(), stdout);

    std::puts("\nreported heaps and memory types (this is what the bench is here to distrust):");
    std::puts("  heap  reported GiB   flags");
    for (const gpu::HeapInfo& h : caps.heaps)
        std::puts(std::format("  {:>4}  {:>12.2f}   {}", h.index, gib(h.bytes),
                              h.device_local ? "DEVICE_LOCAL" : "host").c_str());
    std::puts("  type  heap   flags");
    for (const gpu::MemoryTypeInfo& t : caps.memory_types) {
        std::string f;
        if (t.device_local)  f += "DEVICE_LOCAL|";
        if (t.host_visible)  f += "HOST_VISIBLE|";
        if (t.host_coherent) f += "HOST_COHERENT|";
        if (t.host_cached)   f += "HOST_CACHED|";
        if (!f.empty()) f.pop_back(); else f = "(none)";
        std::puts(std::format("  {:>4}  {:>4}   {}", t.index, t.heap_index, f).c_str());
    }

    const MemStatus start = mem_status();
    if (start.valid) {
        std::puts(std::format("\nGlobalMemoryStatusEx at start: {:.2f} GiB physical total, "
                              "{:.2f} GiB available; commit {:.2f} / {:.2f} GiB",
                              gib(start.total_phys), gib(start.avail_phys),
                              gib(start.total_commit - start.avail_commit), gib(start.total_commit))
                      .c_str());
        const uint64_t heap_sum = [&] {
            uint64_t s = 0;
            for (const gpu::HeapInfo& h : caps.heaps) s += h.bytes;
            return s;
        }();
        std::puts(std::format("  reported heaps sum to {:.2f} GiB against {:.2f} GiB of physical RAM "
                              "-- on this UMA part they overlap, which is exactly why capacity has "
                              "to be measured.",
                              gib(heap_sum), gib(start.total_phys)).c_str());
    } else {
        std::puts("\nGlobalMemoryStatusEx is unavailable on this platform: the --min-free-gib floor "
                  "cannot be enforced and only allocation failure will stop a sweep.");
    }

    if (o.slab_bytes > caps.max_memory_allocation_size && caps.max_memory_allocation_size) {
        std::puts(std::format("\n--slab-gib {:.2f} exceeds maxMemoryAllocationSize {:.2f}; clamping "
                              "(design §5.3: that is what the slab pool is for)",
                              gib(o.slab_bytes), gib(caps.max_memory_allocation_size)).c_str());
        o.slab_bytes = align_down(caps.max_memory_allocation_size, kPageSize);
    }

    // --- the raw-read kernel, built up front ---------------------------------
    // Its sink buffer and descriptor pool must exist *before* the machine is at
    // its allocation ceiling, or the sanity pass in step 4 fails for the one
    // reason it must not: running out of memory.
    gpu::MemoryAllocator util;
    gpu::RawReadKernel   raw;
    bool raw_ready = false;
    if (auto r = util.init(dev, MemoryPath::DeviceLocalHostVisible); r) {
        if (auto c = raw.create(dev, util, gpu::default_shader_dir(), o.groups); c) {
            raw_ready = true;
        } else {
            std::puts(std::format("\nraw-read sanity pass unavailable: {}", c.error().str()).c_str());
        }
    }

    gpu::MemoryAllocator alloc_a, alloc_b;
    const bool a_ok = alloc_a.init(dev, MemoryPath::DeviceLocalHostVisible).has_value();
    const bool b_ok = alloc_b.init(dev, MemoryPath::ExternalMemoryHost).has_value();
    if (!a_ok) std::puts("\npath A unavailable on this device (no DEVICE_LOCAL|HOST_VISIBLE type)");
    if (!b_ok) std::puts("\npath B unavailable on this device (no VK_EXT_external_memory_host)");

    std::vector<gpu::GpuBuffer> live_a, live_b;
    SweepResult a_alone, b_alone;

    // --- 2. path A alone ------------------------------------------------------
    if (o.run_a && a_ok && !interrupted()) {
        a_alone = run_sweep("path_a", Kind::PathA, alloc_a, live_a, o);
        if (raw_ready && !live_a.empty())
            sanity_read(raw, live_a.back(), "last path A slab", "path_a_rawread");
        free_all(alloc_a, live_a);
        std::puts(std::format("  freed; available physical is now {:.2f} GiB",
                              gib(mem_status().avail_phys)).c_str());
    }

    // --- 3. path B alone ------------------------------------------------------
    if (o.run_b && b_ok && !interrupted()) {
        b_alone = run_sweep("path_b", Kind::PathB, alloc_b, live_b, o);
        if (raw_ready && !live_b.empty())
            sanity_read(raw, live_b.back(), "last path B slab", "path_b_rawread");
        free_all(alloc_b, live_b);
        std::puts(std::format("  freed; available physical is now {:.2f} GiB",
                              gib(mem_status().avail_phys)).c_str());
    }
#if defined(DEEPMOE_ENABLE_VULKAN)
    if (o.run_oversize && b_ok && !interrupted()) run_oversize_probe(dev, o);
#endif

    // --- 4. mixed: A to its ceiling, then B on top ----------------------------
    SweepResult mixed_a, mixed_b;
    if (o.run_mixed && !interrupted() && (a_ok || b_ok)) {
        std::puts("\n================ mixed: path A to its ceiling, then path B on top "
                  "================");
        if (a_ok && o.run_a) mixed_a = run_sweep("mixed_a", Kind::PathA, alloc_a, live_a, o);
        if (b_ok && o.run_b && !interrupted())
            mixed_b = run_sweep("mixed_b", Kind::PathB, alloc_b, live_b, o);

        const uint64_t total = mixed_a.bytes + mixed_b.bytes;
        std::puts(std::format("\nGRAND TOTAL simultaneously allocated, committed and "
                              "device-addressable: {:.2f} GiB  ({:.2f} on path A + {:.2f} on path B, "
                              "{} + {} slabs)",
                              gib(total), gib(mixed_a.bytes), gib(mixed_b.bytes),
                              mixed_a.slabs, mixed_b.slabs).c_str());
        g_rows.push_back(Row{"mixed_total", mixed_a.slabs + mixed_b.slabs, o.slab_bytes, total,
                             0.0, 0.0, 0.0, mem_status().avail_phys, mem_status().avail_commit, 0.0,
                             "grand total"});

        if (raw_ready) {
            std::puts("\nraw-read sanity pass at full occupancy (kernel_p1 §2.2 measured "
                      "~216 GB/s on an idle machine):");
            if (!live_a.empty()) sanity_read(raw, live_a.back(), "last path A slab", "mixed_a_rawread");
            if (!live_b.empty()) sanity_read(raw, live_b.back(), "last path B slab", "mixed_b_rawread");
        }

        // design §5.3 arithmetic: an expert is 18.8 MB, and the checkpoint has
        // 15,360 of them. Turning GiB into the only number that matters.
        constexpr double kExpertMB = 18.8;
        constexpr double kTotalExperts = 15360.0;
        const double experts = double(total) / (kExpertMB * 1e6);
        std::puts(std::format("  = {:.0f} resident experts of 18.8 MB = {:.1f}% of the checkpoint's "
                              "{:.0f} (kernel_p1 §4.1 projected 2,980 on path A at VGM 64 GB)",
                              experts, 100.0 * experts / kTotalExperts, kTotalExperts).c_str());
    }

    // --- 5. free everything, always ------------------------------------------
    free_all(alloc_a, live_a);
    free_all(alloc_b, live_b);
    raw.destroy();
    util.shutdown();
    alloc_a.shutdown();
    alloc_b.shutdown();

    const MemStatus end = mem_status();
    std::puts("\n================================ summary ================================");
    std::puts(std::format("path A alone : {:>3} slabs  {:>7.2f} GiB   ({})", a_alone.slabs,
                          gib(a_alone.bytes), a_alone.stop_reason).c_str());
    std::puts(std::format("path B alone : {:>3} slabs  {:>7.2f} GiB   ({})", b_alone.slabs,
                          gib(b_alone.bytes), b_alone.stop_reason).c_str());
    std::puts(std::format("mixed A + B  : {:>3} slabs  {:>7.2f} GiB   (A {:.2f} + B {:.2f})",
                          mixed_a.slabs + mixed_b.slabs, gib(mixed_a.bytes + mixed_b.bytes),
                          gib(mixed_a.bytes), gib(mixed_b.bytes)).c_str());
    if (start.valid && end.valid)
        std::puts(std::format("available physical: {:.2f} GiB at start, {:.2f} GiB at exit "
                              "(everything freed)",
                              gib(start.avail_phys), gib(end.avail_phys)).c_str());
    if (interrupted())
        std::puts("NOTE: interrupted before the sweeps finished; the numbers above are lower "
                  "bounds, not ceilings.");

    g_rows.push_back(Row{"summary", 0, o.slab_bytes, mixed_a.bytes + mixed_b.bytes, 0.0, 0.0, 0.0,
                         end.avail_phys, end.avail_commit, 0.0,
                         interrupted() ? "interrupted" : "complete"});
    write_csv(o.csv);
    return 0;
}
