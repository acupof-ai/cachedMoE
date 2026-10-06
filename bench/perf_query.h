// VK_KHR_performance_query around one recorded workload: the hardware counters
// behind docs/STATUS.md's "the GEMM sits at ~1/3 of the mma peak, and we do not
// know which third is missing".
//
// The extension is env-gated in gpu/vulkan/device.cpp (CACHEDMOE_PERF_COUNTERS),
// so nothing here runs unless a bench asks for it. Three RADV facts shape the
// API:
//
//   * counters are only answered on the general (graphics) queue family --
//     radv_perfcounter.c returns 0 counters for anything else -- so
//     CACHEDMOE_PERF_COUNTERS also makes Device pick that family instead of the
//     async compute engine. A counter run's *times* are therefore not
//     comparable with the platform's numbers; its counts and ratios are.
//   * a set of counters may need several passes, and the driver's multi-pass
//     protocol is easy to get subtly wrong (asking for 9 counters in 2 passes
//     returned the first pass's counters and zeros for the rest). So this class
//     never uses more than one pass: it splits the selection into groups that
//     each fit in a single pass and runs the workload once a group. The caller's
//     workload must be idempotent, which it already had to be.
//   * the instruction-count counters UNDER-report, and only ever under-report:
//     the same 576-wave dispatch read VALU Instructions as 2,484,288 /
//     2,691,312 / 2,898,336 on three identical runs and 3,312,384 on a fourth,
//     and LDS Instructions as 998,400 / 1,152,000 / 1,152,000 (STATUS §7 0bq).
//     The largest read is the honest one -- it is the only one divisible into a
//     whole number per wave. So every group runs CACHEDMOE_PERF_REPEAT times
//     (default 3) and `Value::value` is the maximum, with `lo` kept so the
//     caller can see how far it moved. Waves is exact, GPU active cycles move
//     +-2% and VRAM read size +-10%, both of which are real machine noise
//     rather than lost counts.
//   * results are only defined while the profiling lock is held, and the lock
//     must be held for as long as a command buffer holding the query is
//     recording, executable or pending -- so `run` resets the command buffer
//     before releasing it.
//
// RADV exposes instruction counts by class (VALU/SALU/VMEM/SMEM/LDS), VALU and
// SALU busy, VRAM read/write size and L0/L1/L2 hit ratios. It exposes no stall
// or wait counters, so "where did the cycles go" is answered by subtraction,
// not read off directly.
#pragma once

#include "core/namespace.h"

#include "core/env.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/status.h"
#include "gpu/vulkan/cmdbuf.h"
#include "gpu/vulkan/device.h"

namespace cachedmoe::bench {

#if defined(CACHEDMOE_ENABLE_VULKAN)

class PerfCounters {
public:
    struct Value {
        std::string name;
        std::string unit;        // "generic", "percent", "bytes", "cycles", ...
        double      value = 0;   // the largest of `repeats` reads (see below)
        double      lo = 0;      // the smallest, so the caller can see the spread
        uint32_t    reads = 1;
    };

    ~PerfCounters() { destroy(); }
    PerfCounters() = default;
    PerfCounters(const PerfCounters&) = delete;
    PerfCounters& operator=(const PerfCounters&) = delete;

    bool enabled() const { return !groups_.empty(); }
    size_t selected() const { return sel_.size(); }
    // How many times run() replays the workload: one per single-pass group.
    size_t replays() const { return groups_.size(); }
    uint32_t repeats() const { return repeats_; }
    // Every counter this driver has, in enumeration order: what a bench's
    // --perf-counters can name.
    const std::vector<std::string>& available() const { return names_; }

    // `want` selects counters by case-insensitive substring; empty selects none.
    // Not an error when the extension is off -- enabled() then stays false and
    // run() returns an empty vector, so a bench can call it unconditionally.
    Result<void> create(gpu::Device& device, const std::vector<std::string>& want) {
        dev_ = &device;
        auto gp = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR>(
            vkGetInstanceProcAddr(device.instance(),
                                  "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR"));
        passes_fn_ = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR>(
            vkGetInstanceProcAddr(device.instance(),
                                  "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR"));
        if (!device.caps().perf_counters || !gp || !passes_fn_) return {};

        uint32_t n = 0;
        gp(device.physical(), device.compute_queue_family(), &n, nullptr, nullptr);
        if (n == 0) return {};
        std::vector<VkPerformanceCounterKHR> cs(n, {VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR});
        std::vector<VkPerformanceCounterDescriptionKHR> ds(
            n, {VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR});
        if (gp(device.physical(), device.compute_queue_family(), &n, cs.data(), ds.data()) != VK_SUCCESS)
            return fail(Err::Internal, "enumerate performance counters failed");
        for (uint32_t i = 0; i < n; ++i) {
            names_.emplace_back(ds[i].name);
            unit_.push_back(unit_of(cs[i].unit));
            storage_.push_back(cs[i].storage);
        }

        for (uint32_t i = 0; i < n; ++i)
            for (const std::string& w : want)
                if (icontains(names_[i], w)) { sel_.push_back(i); break; }
        if (sel_.empty()) return {};

        acquire_ = reinterpret_cast<PFN_vkAcquireProfilingLockKHR>(
            vkGetDeviceProcAddr(device.handle(), "vkAcquireProfilingLockKHR"));
        release_ = reinterpret_cast<PFN_vkReleaseProfilingLockKHR>(
            vkGetDeviceProcAddr(device.handle(), "vkReleaseProfilingLockKHR"));
        if (!acquire_ || !release_) return fail(Err::Internal, "no vkAcquireProfilingLockKHR");

        // Greedily pack the selection into groups that each need exactly one
        // pass. A counter that needs more than one pass on its own cannot be
        // measured this way; none of RADV's do, but say so rather than lie.
        // RADV's pass count is computed per hardware block, but the SQ block's
        // counter slots are shared in ways the count does not capture: 13
        // counters it called one pass returned zeros for five of them, four at
        // a time still lost VALU Instructions and VALU Busy, and one at a time
        // read fine. So cap the group size too: CACHEDMOE_PERF_GROUP, default 1,
        // is the only size this machine was measured to be honest at
        // (docs/STATUS.md). A replay of a sub-ms kernel is free.
        size_t cap = 1;
        if (const char* g = ::cachedmoe::environment::get("CACHEDMOE_PERF_GROUP"); g && *g)
            cap = std::max(1, std::atoi(g));
        std::vector<uint32_t> cur;
        for (uint32_t c : sel_) {
            cur.push_back(c);
            if (cur.size() <= cap && pass_count(cur) <= 1) continue;
            if (cur.size() == 1)
                return fail(Err::Unimplemented,
                            std::string("counter needs several passes: ") + names_[c]);
            cur.pop_back();
            if (auto r = add_group(cur); !r) return r;
            cur.assign(1, c);
        }
        if (!cur.empty())
            if (auto r = add_group(cur); !r) return r;
        if (const char* r = ::cachedmoe::environment::get("CACHEDMOE_PERF_REPEAT"); r && *r)
            repeats_ = uint32_t(std::max(1, std::atoi(r)));
        return {};
    }

    void destroy() {
        if (dev_)
            for (Group& g : groups_)
                if (g.pool != VK_NULL_HANDLE) vkDestroyQueryPool(dev_->handle(), g.pool, nullptr);
        groups_.clear();
    }

    // Records `fn` once a group per repeat and returns every selected counter,
    // in the order create() selected them. `fn` runs replays() * repeats()
    // times, so it must leave behind no state that changes the next run's work
    // (a GEMM into a scratch buffer is fine).
    template <class F>
    Result<std::vector<Value>> run(gpu::CommandBuffer& cmd, F&& fn) {
        std::vector<Value> out;
        if (!enabled()) return out;

        VkAcquireProfilingLockInfoKHR li{VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR};
        li.timeout = UINT64_MAX;
        if (acquire_(dev_->handle(), &li) != VK_SUCCESS)
            return fail(Err::Unavailable, "vkAcquireProfilingLockKHR failed (another profiler?)");
        // The lock has to outlive every command buffer state that holds the
        // query, so the reset comes before the release, not after.
        struct Unlock {
            PFN_vkReleaseProfilingLockKHR f; VkDevice d; gpu::CommandBuffer* c;
            ~Unlock() { if (c->handle()) vkResetCommandBuffer(c->handle(), 0); f(d); }
        } unlock{release_, dev_->handle(), &cmd};

        for (const Group& g : groups_) {
          std::vector<double> hi(g.idx.size(), -1e300), lo(g.idx.size(), 1e300);
          for (uint32_t rep = 0; rep < repeats_; ++rep) {
            if (auto r = cmd.begin(); !r) return std::unexpected(r.error());
            vkCmdResetQueryPool(cmd.handle(), g.pool, 0, 1);
            vkCmdBeginQuery(cmd.handle(), g.pool, 0, 0);
            if (auto r = fn(cmd); !r) return std::unexpected(r.error());
            vkCmdEndQuery(cmd.handle(), g.pool, 0);
            if (auto r = cmd.end(); !r) return std::unexpected(r.error());

            VkPerformanceQuerySubmitInfoKHR psi{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
            psi.counterPassIndex = 0;
            VkCommandBuffer cb = cmd.handle();
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &psi};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            if (vkQueueSubmit(dev_->compute_queue(), 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
                return fail(Err::Internal, "vkQueueSubmit (performance pass) failed");
            if (vkQueueWaitIdle(dev_->compute_queue()) != VK_SUCCESS)
                return fail(Err::Internal, "vkQueueWaitIdle (performance pass) failed");

            std::vector<VkPerformanceCounterResultKHR> res(g.idx.size());
            if (vkGetQueryPoolResults(dev_->handle(), g.pool, 0, 1,
                                      res.size() * sizeof(res[0]), res.data(),
                                      res.size() * sizeof(res[0]), VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
                return fail(Err::Internal, "vkGetQueryPoolResults (performance) failed");
            for (size_t i = 0; i < g.idx.size(); ++i) {
                const double v = decode(g.storage[i], res[i]);
                hi[i] = std::max(hi[i], v);
                lo[i] = std::min(lo[i], v);
            }
          }
          for (size_t i = 0; i < g.idx.size(); ++i)
              out.push_back(Value{names_[g.idx[i]], g.unit[i], hi[i], lo[i], repeats_});
        }
        return out;
    }

private:
    struct Group {
        VkQueryPool pool = VK_NULL_HANDLE;
        std::vector<uint32_t> idx;
        std::vector<const char*> unit;
        std::vector<VkPerformanceCounterStorageKHR> storage;
    };

    uint32_t pass_count(const std::vector<uint32_t>& idx) const {
        VkQueryPoolPerformanceCreateInfoKHR pci{VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR};
        pci.queueFamilyIndex = dev_->compute_queue_family();
        pci.counterIndexCount = static_cast<uint32_t>(idx.size());
        pci.pCounterIndices = idx.data();
        uint32_t p = 1;
        passes_fn_(dev_->physical(), &pci, &p);
        return p;
    }

    Result<void> add_group(const std::vector<uint32_t>& idx) {
        Group g;
        g.idx = idx;
        // Units and storage come from the enumeration cached in create().
        for (uint32_t c : idx) { g.unit.push_back(unit_[c]); g.storage.push_back(storage_[c]); }
        VkQueryPoolPerformanceCreateInfoKHR pci{VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR};
        pci.queueFamilyIndex = dev_->compute_queue_family();
        pci.counterIndexCount = static_cast<uint32_t>(g.idx.size());
        pci.pCounterIndices = g.idx.data();
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, &pci};
        qci.queryType = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
        qci.queryCount = 1;
        if (vkCreateQueryPool(dev_->handle(), &qci, nullptr, &g.pool) != VK_SUCCESS)
            return fail(Err::Internal, "create performance query pool failed");
        groups_.push_back(std::move(g));
        return {};
    }
    static bool icontains(const std::string& hay, const std::string& needle) {
        if (needle.empty()) return false;
        auto low = [](char c) { return char(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c); };
        std::string h, n;
        for (char c : hay) h += low(c);
        for (char c : needle) n += low(c);
        return h.find(n) != std::string::npos;
    }
    static const char* unit_of(VkPerformanceCounterUnitKHR u) {
        switch (u) {
            case VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR: return "generic";
            case VK_PERFORMANCE_COUNTER_UNIT_PERCENTAGE_KHR: return "percent";
            case VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR: return "ns";
            case VK_PERFORMANCE_COUNTER_UNIT_BYTES_KHR: return "bytes";
            case VK_PERFORMANCE_COUNTER_UNIT_BYTES_PER_SECOND_KHR: return "B/s";
            case VK_PERFORMANCE_COUNTER_UNIT_KELVIN_KHR: return "K";
            case VK_PERFORMANCE_COUNTER_UNIT_WATTS_KHR: return "W";
            case VK_PERFORMANCE_COUNTER_UNIT_VOLTS_KHR: return "V";
            case VK_PERFORMANCE_COUNTER_UNIT_AMPS_KHR: return "A";
            case VK_PERFORMANCE_COUNTER_UNIT_HERTZ_KHR: return "Hz";
            case VK_PERFORMANCE_COUNTER_UNIT_CYCLES_KHR: return "cycles";
            default: return "?";
        }
    }
    static double decode(VkPerformanceCounterStorageKHR s, const VkPerformanceCounterResultKHR& r) {
        switch (s) {
            case VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR: return double(r.int32);
            case VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR: return double(r.int64);
            case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR: return double(r.uint32);
            case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR: return double(r.uint64);
            case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR: return double(r.float32);
            case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR: return r.float64;
            default: return 0.0;
        }
    }

    gpu::Device* dev_ = nullptr;
    std::vector<std::string> names_;
    std::vector<const char*> unit_;
    std::vector<VkPerformanceCounterStorageKHR> storage_;
    std::vector<uint32_t> sel_;
    std::vector<Group>    groups_;
    uint32_t              repeats_ = 3;
    PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR passes_fn_ = nullptr;
    PFN_vkAcquireProfilingLockKHR acquire_ = nullptr;
    PFN_vkReleaseProfilingLockKHR release_ = nullptr;
};

#endif  // CACHEDMOE_ENABLE_VULKAN

}  // namespace cachedmoe::bench
