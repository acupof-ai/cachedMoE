// Per-token timeline and online counters. This is the measurement record the
// whole project is judged by (design §13.1):
//
//   T_token = T_hot_gemv + T_expert_hit + T_attn_misc
//           + T_dispatch_overhead + T_nvme_stall + T_cpu_sync
//
// plus the NVMe acceptance metrics of §9.8 (hit rate, miss bytes/token,
// prefetch precision/recall, stall_ms, nvme_util, effective NVMe GB/s).
// One JSONL record per token is appended to the sink so a run can be replayed
// offline by tools/bench_report.py.
//
// Ownership/threading: a Profiler is owned by the Engine and shared by
// reference. Phase accumulators and counters are relaxed atomics because the
// IOCP completion threads and the planner thread report into the same token
// record as the submit thread. `token_begin`/`token_end` are called only from
// the submit thread; a token boundary is the synchronisation point, so no lock
// is taken on the hot path.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>

#include "core/status.h"
#include "core/types.h"

namespace deepmoe {

using Clock    = std::chrono::steady_clock;
using Nanos    = std::chrono::nanoseconds;
using TimePoint = Clock::time_point;

// Timeline buckets. The names are the JSONL field names, so they are also the
// column names in every report; keep them stable.
enum class Phase : uint8_t {
    HotGemv = 0,  // resident weights: attention, shared expert, router, mHC, head
    ExpertHit,    // routed experts that were already resident
    AttnMisc,     // sparse attention, compressor, indexer, engram gating
    Dispatch,     // Vulkan submit + barrier overhead attributed per token
    NvmeStall,    // GPU blocked on a timeline wait for an expert (design §9.8)
    CpuSync,      // router readback spin, planner handoff, host signal
    Draft,        // DSpark draft cycle (design §7.12)
    Sample,       // logits -> token
    Count_
};
inline constexpr size_t kPhaseCount = static_cast<size_t>(Phase::Count_);

constexpr std::string_view phase_name(Phase p) noexcept {
    switch (p) {
        case Phase::HotGemv:   return "hot_gemv";
        case Phase::ExpertHit: return "expert_hit";
        case Phase::AttnMisc:  return "attn_misc";
        case Phase::Dispatch:  return "dispatch";
        case Phase::NvmeStall: return "nvme_stall";
        case Phase::CpuSync:   return "cpu_sync";
        case Phase::Draft:     return "draft";
        case Phase::Sample:    return "sample";
        case Phase::Count_:    return "?";
    }
    return "?";
}

// Snapshot of one token, emitted as a single JSONL line.
struct TokenRecord {
    TokenIndex token       = 0;
    uint64_t   wall_ns     = 0;
    std::array<uint64_t, kPhaseCount> phase_ns{};

    // residency (design §9.8)
    uint32_t expert_requests = 0;   // routed expert lookups this token (layers x top-6)
    uint32_t expert_hits     = 0;
    uint32_t expert_misses   = 0;
    uint64_t hot_bytes       = 0;   // resident weight bytes read by the GPU
    uint64_t miss_bytes      = 0;   // bytes pulled from NVMe for this token
    uint64_t nvme_busy_ns    = 0;   // union of time with >=1 I/O in flight

    // prefetch quality (design §9.4)
    uint32_t prefetch_issued = 0;
    uint32_t prefetch_used   = 0;

    // speculative decoding (design §10)
    uint32_t draft_len    = 0;
    uint32_t accepted_len = 0;

    double hit_rate() const {
        return expert_requests ? static_cast<double>(expert_hits) / expert_requests : 0.0;
    }
    double prefetch_precision() const {
        return prefetch_issued ? static_cast<double>(prefetch_used) / prefetch_issued : 0.0;
    }
    // Effective NVMe GB/s over the busy window, the §9.8 acceptance number.
    double nvme_gbps() const {
        return nvme_busy_ns ? static_cast<double>(miss_bytes) / static_cast<double>(nvme_busy_ns) : 0.0;
    }
    double nvme_util() const {
        return wall_ns ? static_cast<double>(nvme_busy_ns) / static_cast<double>(wall_ns) : 0.0;
    }

    std::string to_jsonl() const;
};

// Running totals over a whole run, printed at the end of a benchmark.
struct RunSummary {
    uint64_t tokens = 0;
    uint64_t wall_ns = 0;
    std::array<uint64_t, kPhaseCount> phase_ns{};
    uint64_t expert_requests = 0, expert_hits = 0;
    uint64_t hot_bytes = 0, miss_bytes = 0, nvme_busy_ns = 0;
    uint64_t prefetch_issued = 0, prefetch_used = 0;
    uint64_t accepted_len = 0, draft_len = 0;

    double tokens_per_second() const {
        return wall_ns ? static_cast<double>(tokens) * 1e9 / static_cast<double>(wall_ns) : 0.0;
    }
    double hit_rate() const {
        return expert_requests ? static_cast<double>(expert_hits) / expert_requests : 0.0;
    }
    std::string to_string() const;
};

class Profiler {
public:
    Profiler() = default;
    ~Profiler();

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Opens (appends to) a JSONL sink. Without a sink the profiler still keeps
    // the in-memory summary, which is what unit tests use.
    Result<void> open_jsonl(const std::string& path);
    void         close();
    bool         enabled() const { return enabled_; }
    void         set_enabled(bool on) { enabled_ = on; }

    // --- token lifecycle (submit thread only) ---
    void token_begin(TokenIndex token);
    TokenRecord token_end();          // flushes one JSONL line, folds into the summary

    // --- reporting (any thread) ---
    void add_phase(Phase p, Nanos d) {
        phase_ns_[static_cast<size_t>(p)].fetch_add(static_cast<uint64_t>(d.count()),
                                                    std::memory_order_relaxed);
    }
    void note_expert_lookup(bool hit) {
        requests_.fetch_add(1, std::memory_order_relaxed);
        if (hit) hits_.fetch_add(1, std::memory_order_relaxed);
    }
    void note_hot_bytes(uint64_t b)  { hot_bytes_.fetch_add(b, std::memory_order_relaxed); }
    void note_miss_bytes(uint64_t b) { miss_bytes_.fetch_add(b, std::memory_order_relaxed); }
    void note_nvme_busy(Nanos d)     { nvme_busy_ns_.fetch_add(static_cast<uint64_t>(d.count()), std::memory_order_relaxed); }
    void note_prefetch_issued(uint32_t n = 1) { pf_issued_.fetch_add(n, std::memory_order_relaxed); }
    void note_prefetch_used(uint32_t n = 1)   { pf_used_.fetch_add(n, std::memory_order_relaxed); }
    void note_speculation(uint32_t draft_len, uint32_t accepted) {
        draft_len_.store(draft_len, std::memory_order_relaxed);
        accepted_.store(accepted, std::memory_order_relaxed);
    }

    const RunSummary& summary() const { return summary_; }

private:
    bool        enabled_ = true;
    std::FILE*  sink_ = nullptr;
    std::mutex  sink_mutex_;           // only taken once per token, at the boundary

    TokenIndex  token_ = 0;
    TimePoint   token_start_{};

    std::array<std::atomic<uint64_t>, kPhaseCount> phase_ns_{};
    std::atomic<uint32_t> requests_{0}, hits_{0};
    std::atomic<uint64_t> hot_bytes_{0}, miss_bytes_{0}, nvme_busy_ns_{0};
    std::atomic<uint32_t> pf_issued_{0}, pf_used_{0};
    std::atomic<uint32_t> draft_len_{0}, accepted_{0};

    RunSummary summary_{};

    void reset_token_counters();
};

// RAII phase timer. Cheap enough for per-dispatch use (two steady_clock reads).
class ScopedPhase {
public:
    ScopedPhase(Profiler& p, Phase ph) : p_(&p), ph_(ph), t0_(Clock::now()) {}
    ~ScopedPhase() { if (p_) p_->add_phase(ph_, Clock::now() - t0_); }
    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;
    void dismiss() { p_ = nullptr; }
private:
    Profiler* p_;
    Phase     ph_;
    TimePoint t0_;
};

// The same timer for a profiler that may not exist. Most of the runtime is
// usable without one (tests, the CPU oracle), and `ScopedPhase(*p, ...)` on a
// null pointer is undefined behaviour even when the reference is never read.
class ScopedPhaseIf {
public:
    ScopedPhaseIf(Profiler* p, Phase ph) : p_(p), ph_(ph), t0_(Clock::now()) {}
    ~ScopedPhaseIf() { if (p_) p_->add_phase(ph_, Clock::now() - t0_); }
    ScopedPhaseIf(const ScopedPhaseIf&) = delete;
    ScopedPhaseIf& operator=(const ScopedPhaseIf&) = delete;
    // Charges what has elapsed so far and stops the timer, so a scope can be
    // split into two buckets without an inner block.
    void close() {
        if (p_) p_->add_phase(ph_, Clock::now() - t0_);
        p_ = nullptr;
    }
private:
    Profiler* p_;
    Phase     ph_;
    TimePoint t0_;
};

}  // namespace deepmoe
