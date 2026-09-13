#include "core/profiler.h"

#include <format>

namespace deepmoe {

std::string TokenRecord::to_jsonl() const {
    std::string s = std::format(
        R"({{"token":{},"wall_ms":{:.4f})", token, wall_ns / 1e6);
    for (size_t i = 0; i < kPhaseCount; ++i)
        s += std::format(R"(,"{}_ms":{:.4f})", phase_name(static_cast<Phase>(i)), phase_ns[i] / 1e6);
    s += std::format(
        R"(,"expert_requests":{},"expert_hits":{},"expert_misses":{},"hit_rate":{:.4f})"
        R"(,"hot_bytes":{},"miss_bytes":{},"nvme_busy_ms":{:.4f},"nvme_util":{:.4f},"nvme_gbps":{:.3f})"
        R"(,"prefetch_issued":{},"prefetch_used":{},"prefetch_precision":{:.4f})"
        R"(,"draft_len":{},"accepted_len":{}}})",
        expert_requests, expert_hits, expert_misses, hit_rate(),
        hot_bytes, miss_bytes, nvme_busy_ns / 1e6, nvme_util(), nvme_gbps(),
        prefetch_issued, prefetch_used, prefetch_precision(),
        draft_len, accepted_len);
    return s;
}

std::string RunSummary::to_string() const {
    std::string s = std::format(
        "tokens {}  wall {:.3f} s  {:.3f} tok/s  hit_rate {:.3f}\n",
        tokens, wall_ns / 1e9, tokens_per_second(), hit_rate());
    for (size_t i = 0; i < kPhaseCount; ++i) {
        double ms = phase_ns[i] / 1e6;
        double pct = wall_ns ? 100.0 * static_cast<double>(phase_ns[i]) / static_cast<double>(wall_ns) : 0.0;
        s += std::format("  {:<11} {:9.2f} ms  ({:5.1f}%)\n", phase_name(static_cast<Phase>(i)), ms, pct);
    }
    s += std::format("  hot_bytes  {:.2f} GB   miss_bytes {:.2f} GB   nvme_busy {:.2f} s",
                     hot_bytes / 1e9, miss_bytes / 1e9, nvme_busy_ns / 1e9);
    if (nvme_busy_ns)
        s += std::format("   eff {:.2f} GB/s", static_cast<double>(miss_bytes) / static_cast<double>(nvme_busy_ns));
    s += "\n";
    if (draft_len)
        s += std::format("  draft {} accepted {}  mean accept {:.2f}\n",
                         draft_len, accepted_len,
                         tokens ? static_cast<double>(accepted_len) / static_cast<double>(tokens) : 0.0);
    return s;
}

Profiler::~Profiler() { close(); }

Result<void> Profiler::open_jsonl(const std::string& path) {
    close();
    std::FILE* f = std::fopen(path.c_str(), "ab");
    if (!f) return fail(Err::Io, std::format("cannot open profiler sink '{}'", path));
    std::lock_guard lk(sink_mutex_);
    sink_ = f;
    return {};
}

void Profiler::close() {
    std::lock_guard lk(sink_mutex_);
    if (sink_) { std::fflush(sink_); std::fclose(sink_); sink_ = nullptr; }
}

void Profiler::reset_token_counters() {
    for (auto& a : phase_ns_) a.store(0, std::memory_order_relaxed);
    requests_.store(0, std::memory_order_relaxed);
    hits_.store(0, std::memory_order_relaxed);
    hot_bytes_.store(0, std::memory_order_relaxed);
    miss_bytes_.store(0, std::memory_order_relaxed);
    nvme_busy_ns_.store(0, std::memory_order_relaxed);
    pf_issued_.store(0, std::memory_order_relaxed);
    pf_used_.store(0, std::memory_order_relaxed);
    draft_len_.store(0, std::memory_order_relaxed);
    accepted_.store(0, std::memory_order_relaxed);
}

void Profiler::token_begin(TokenIndex token) {
    token_ = token;
    token_start_ = Clock::now();
    reset_token_counters();
}

TokenRecord Profiler::token_end() {
    TokenRecord r;
    r.token   = token_;
    r.wall_ns = static_cast<uint64_t>((Clock::now() - token_start_).count());
    for (size_t i = 0; i < kPhaseCount; ++i)
        r.phase_ns[i] = phase_ns_[i].load(std::memory_order_relaxed);
    r.expert_requests = requests_.load(std::memory_order_relaxed);
    r.expert_hits     = hits_.load(std::memory_order_relaxed);
    r.expert_misses   = r.expert_requests - r.expert_hits;
    r.hot_bytes       = hot_bytes_.load(std::memory_order_relaxed);
    r.miss_bytes      = miss_bytes_.load(std::memory_order_relaxed);
    r.nvme_busy_ns    = nvme_busy_ns_.load(std::memory_order_relaxed);
    r.prefetch_issued = pf_issued_.load(std::memory_order_relaxed);
    r.prefetch_used   = pf_used_.load(std::memory_order_relaxed);
    r.draft_len       = draft_len_.load(std::memory_order_relaxed);
    r.accepted_len    = accepted_.load(std::memory_order_relaxed);

    summary_.tokens          += 1;
    summary_.wall_ns         += r.wall_ns;
    for (size_t i = 0; i < kPhaseCount; ++i) summary_.phase_ns[i] += r.phase_ns[i];
    summary_.expert_requests += r.expert_requests;
    summary_.expert_hits     += r.expert_hits;
    summary_.hot_bytes       += r.hot_bytes;
    summary_.miss_bytes      += r.miss_bytes;
    summary_.nvme_busy_ns    += r.nvme_busy_ns;
    summary_.prefetch_issued += r.prefetch_issued;
    summary_.prefetch_used   += r.prefetch_used;
    summary_.draft_len       += r.draft_len;
    summary_.accepted_len    += r.accepted_len;

    if (enabled_) {
        std::lock_guard lk(sink_mutex_);
        if (sink_) {
            std::string line = r.to_jsonl();
            line.push_back('\n');
            std::fwrite(line.data(), 1, line.size(), sink_);
        }
    }
    return r;
}

}  // namespace deepmoe
