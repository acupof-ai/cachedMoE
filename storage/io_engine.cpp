#include "storage/io_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <format>
#include <vector>

#include "core/align.h"
#include "core/log.h"

namespace deepmoe::storage {

std::string IoStats::to_string() const {
    std::string s = std::format(
        "io: {} req ({} done, {} failed, {} cancelled), {} chunks, {} / {} bytes, "
        "busy {:.3f} s, eff {:.2f} GB/s, mean lat {:.3f} ms, peak QD {} / {:.1f} MiB\n",
        requests_submitted, requests_completed, requests_failed, requests_cancelled,
        chunks_completed, bytes_completed, bytes_requested,
        busy_ns / 1e9, effective_gbps(), mean_latency_ms(),
        peak_inflight_ops, peak_inflight_bytes / 1048576.0);
    for (uint8_t p = 0; p < kIoPriorityCount; ++p) {
        if (!per_priority_requests[p]) continue;
        s += std::format("  {:<13} {:6} req  {:.1f} MiB\n",
                         io_priority_name(static_cast<IoPriority>(p)),
                         per_priority_requests[p], per_priority_bytes[p] / 1048576.0);
    }
    if (p0_requests) {
        const double n = double(p0_requests);
        s += std::format(
            "  P0: {} req, {:.1f} MiB, lat mean {:.2f} / p50 {:.2f} / p95 {:.2f} / max {:.2f} ms"
            "  (queue wait {:.2f} + service {:.2f})\n",
            p0_requests, p0_bytes / 1048576.0,
            p0_lat_ns_sum / 1e6 / n, p0_lat_p50_ns / 1e6, p0_lat_p95_ns / 1e6,
            p0_lat_ns_max / 1e6,
            p0_queue_wait_ns_sum / 1e6 / n, p0_service_ns_sum / 1e6 / n);
        s += std::format(
            "  P0: first-of-burst {} at {:.2f} ms, behind {} at {:.2f} ms; "
            "{} ({:.1f}%) issued with a non-P0 chunk in flight, mean {:.2f}; "
            "mean QD at issue {:.2f}\n",
            p0_first_n, p0_first_n ? p0_first_lat_ns_sum / 1e6 / double(p0_first_n) : 0.0,
            p0_behind_n, p0_behind_n ? p0_behind_lat_ns_sum / 1e6 / double(p0_behind_n) : 0.0,
            p0_with_bg_n, 100.0 * double(p0_with_bg_n) / n,
            p0_with_bg_n ? double(p0_bg_inflight_sum) / double(p0_with_bg_n) : 0.0,
            p0_chunks_issued ? double(p0_qd_at_issue_sum) / double(p0_chunks_issued) : 0.0);
    }
    if (sources.size() > 1) {
        // Track D2: which drive served what. `inflight` is a live read, so a
        // status.json taken mid-decode shows both queues rather than a total.
        for (size_t i = 0; i < sources.size(); ++i) {
            const SourceStats& e = sources[i];
            s += std::format(
                "  src[{}] {}  w {:.2f} GB/s  {} req  {:.1f} GiB ({:.1f}%)  "
                "mean lat {:.2f} ms  inflight {} req / {:.1f} MiB\n",
                i, e.root.empty() ? std::string("(primary)") : e.root, e.weight,
                e.requests, e.bytes / 1073741824.0,
                bytes_completed ? 100.0 * double(e.bytes) / double(bytes_completed) : 0.0,
                e.mean_latency_ms(), e.inflight_requests,
                e.outstanding_bytes / 1048576.0);
        }
    }
    if (disp_iters) {
        const double tot = double(disp_issue_ns + disp_poll_ns + disp_handle_ns);
        s += std::format(
            "  dispatcher: {} iters, issue {:.2f} s / poll {:.2f} s / handle {:.2f} s"
            " (callbacks {:.2f} s, {} at {:.1f} us mean)\n",
            disp_iters, disp_issue_ns / 1e9, disp_poll_ns / 1e9, disp_handle_ns / 1e9,
            disp_cb_ns / 1e9, disp_cbs, disp_cb_mean_us());
        s += std::format(
            "  dispatcher: Backend::submit {:.2f} s over {} chunks, mean {:.1f} us,"
            " max {:.2f} ms  <- synchronous, on the only issuing thread\n",
            disp_submit_ns / 1e9, disp_submits, disp_submit_mean_us(),
            disp_submit_ns_max / 1e6);
        s += std::format(
            "  dispatcher: handle is {:.1f}% of the non-waiting loop;"
            " refill gap {:.1f} us mean over {} reaps\n",
            tot > 0 ? 100.0 * double(disp_handle_ns) / tot : 0.0,
            disp_refill_mean_us(), disp_refills);
    }
    return s;
}

std::vector<IoEngine::Chunk> IoEngine::plan_chunks(uint64_t off, uint64_t bytes,
                                                   uint32_t max_chunk, uint32_t alignment) {
    std::vector<Chunk> out;
    if (bytes == 0 || max_chunk == 0) return out;
    // Round the step down to the alignment so every chunk but the last starts
    // and ends on a sector boundary. The manifest's runs (design §5.1) make
    // `bytes` itself a 4 KiB multiple, so the last chunk is aligned too.
    uint32_t step = alignment ? static_cast<uint32_t>(align_down(max_chunk, alignment)) : max_chunk;
    if (step == 0) step = alignment ? alignment : max_chunk;
    out.reserve(static_cast<size_t>((bytes + step - 1) / step));
    uint64_t cur = off, rem = bytes;
    while (rem) {
        const uint32_t n = static_cast<uint32_t>(std::min<uint64_t>(step, rem));
        out.push_back(Chunk{cur, n});
        cur += n;
        rem -= n;
    }
    return out;
}

std::string IoEngine::Tuning::to_string() const {
    return std::format("bg_cap_busy {} throttle_engram {} p0_qd {} p0_inflight {} MiB "
                       "p0_chunk {} MiB bg_qd {} bg_inflight {} MiB",
                       bg_cap_busy, throttle_engram ? 1 : 0, p0_qd,
                       p0_inflight_bytes >> 20, p0_chunk_bytes >> 20,
                       bg_qd, bg_inflight_bytes >> 20) +
           std::format(" submit_threads {}", submit_threads);
}

void IoEngine::widen_for_env(IoConfig& cfg) {
    const Tuning t = tuning_from_env(cfg);
    if (t.p0_qd > cfg.max_inflight_ops)                 cfg.max_inflight_ops = t.p0_qd;
    if (t.p0_inflight_bytes > cfg.max_inflight_bytes)   cfg.max_inflight_bytes = t.p0_inflight_bytes;
    if (t.p0_chunk_bytes > cfg.chunk_bytes)             cfg.chunk_bytes = t.p0_chunk_bytes;
}

IoEngine::Tuning IoEngine::tuning_from_env(const IoConfig& cfg) {
    Tuning t;
    t.p0_qd             = cfg.max_inflight_ops;
    t.p0_inflight_bytes = cfg.max_inflight_bytes;
    t.p0_chunk_bytes    = cfg.chunk_bytes;
    auto u32 = [](const char* name, uint32_t& dst) {
        if (const char* e = std::getenv(name); e && *e) {
            char* end = nullptr;
            const unsigned long v = std::strtoul(e, &end, 10);
            if (end != e) dst = static_cast<uint32_t>(v);
        }
    };
    t.bg_qd             = cfg.max_inflight_ops;
    t.bg_inflight_bytes = cfg.max_inflight_bytes;
    u32("DEEPMOE_IO_BG_CAP_BUSY", t.bg_cap_busy);
    if (const char* e = std::getenv("DEEPMOE_IO_BG_THROTTLE_P2"); e && *e)
        t.throttle_engram = (*e != '0');
    const uint32_t qd_before  = t.p0_qd;
    const uint64_t byt_before = t.p0_inflight_bytes;
    u32("DEEPMOE_IO_P0_QD", t.p0_qd);
    uint32_t mb = 0;
    u32("DEEPMOE_IO_P0_INFLIGHT_MB", mb);
    if (mb) t.p0_inflight_bytes = uint64_t(mb) << 20;
    mb = 0;
    u32("DEEPMOE_IO_P0_CHUNK_MB", mb);
    if (mb) t.p0_chunk_bytes = uint64_t(mb) << 20;
    // widen_for_env raises the IoConfig -- and with it the backend's queue
    // depth -- to whatever P0 asked for. Only THEN do the background classes
    // have to be held to the ceiling they shipped with. An untouched runtime,
    // and bench/nvme_bench (which builds its own IoConfig per point), keep
    // exactly the config they were given.
    //
    // Track Q2 made the runtime raise the same ceilings in its own IoConfig
    // (runtime/engine.cpp), not only through the environment, so the clamp is
    // now unconditional: whatever P0's depth ends up being, P1-P3 keep the
    // depth the IoConfig shipped with. Every caller that deliberately runs a
    // shallower or deeper queue -- bench/nvme_bench, bench/io_dst_bench -- uses
    // P0 only, so this does not touch a measurement.
    const IoConfig d{};
    (void)qd_before; (void)byt_before;
    if (t.bg_qd > d.max_inflight_ops)               t.bg_qd = d.max_inflight_ops;
    if (t.bg_inflight_bytes > d.max_inflight_bytes) t.bg_inflight_bytes = d.max_inflight_bytes;
    u32("DEEPMOE_IO_BG_QD", t.bg_qd);
    u32("DEEPMOE_IO_SUBMIT_THREADS", t.submit_threads);
    if (t.submit_threads == 0) t.submit_threads = 1;
    if (t.submit_threads > 16) t.submit_threads = 16;
    return t;
}


// --- Track D2: the second read source ---------------------------------------

void IoEngine::set_sources(const std::vector<std::string>& roots,
                           const std::vector<double>& weights) {
    src_roots_.assign(roots.begin(),
                      roots.begin() + std::min<size_t>(roots.size(), kMaxIoSources));
    src_weights_.assign(src_roots_.size(), 1.0);
    for (size_t i = 0; i < src_weights_.size() && i < weights.size(); ++i)
        if (weights[i] > 0.0) src_weights_[i] = weights[i];
    std::lock_guard lk(src_mutex_);
    for (uint32_t i = 0; i < kMaxIoSources; ++i) {
        src_outstanding_[i] = 0;
        src_inflight_[i]    = 0;
        src_stats_[i]       = SourceStats{};
        if (i < src_roots_.size()) {
            src_stats_[i].root   = src_roots_[i];
            src_stats_[i].weight = src_weights_[i];
        }
    }
    // One root is the ordinary run. The router only switches on when there is
    // something to choose between, so "no mirror" costs a bool test per submit
    // and nothing else.
    mirrors_on_ = src_roots_.size() > 1;
    // Which priority classes may be routed. P0 is the whole point; P3 (the
    // idle backfill) goes too so that a quiet period fills slots from both
    // drives. P1/P2 stay on the primary: P1 is small and speculative, and P2's
    // 264 B engram rows are latency-bound, where the slower drive is a loss.
    if (const char* e = std::getenv("DEEPMOE_MIRROR_CLASSES"); e && *e) {
        uint32_t m = 0;
        for (const char* c = e; *c; ++c)
            if (*c >= '0' && *c <= '3') m |= 1u << uint32_t(*c - '0');
        if (m) route_classes_ = m;
    }
    if (mirrors_on_) {
        std::string w;
        for (size_t i = 0; i < src_roots_.size(); ++i)
            w += std::format("{}{} @ {:.2f} GB/s", i ? ", " : "", src_roots_[i], src_weights_[i]);
        log_info("IoEngine: {} read sources ({}), routing classes 0x{:x}",
                 src_roots_.size(), w, route_classes_);
    }
}

Result<void> IoEngine::add_mirror(const File* primary, uint32_t src, const File* alt) {
    if (!primary || !alt) return fail(Err::InvalidArgument, "add_mirror needs two open files");
    if (src == 0 || src >= kMaxIoSources)
        return fail(Err::OutOfRange, std::format("mirror source {} out of range", src));
    if (!alt->is_open()) return fail(Err::InvalidArgument, "mirror file is not open");
    // Byte-identical or nothing: a mirror whose size differs is a different
    // checkpoint, and serving half an expert from it would be silent garbage.
    if (alt->size() != primary->size())
        return fail(Err::Corrupt,
                    std::format("mirror '{}' is {} B, the primary '{}' is {} B",
                                alt->path(), alt->size(), primary->path(), primary->size()));
    if (alt->unbuffered() != primary->unbuffered())
        return fail(Err::FailedPrecondition, "mirror and primary disagree on unbuffered");
    auto& row = alts_[primary];
    row[0] = primary;
    row[src] = alt;
    return {};
}

Result<double> IoEngine::probe_source_gbps(const std::string& sample_path,
                                           uint32_t ms, uint32_t qd) {
    // Deliberately NOT the runtime's handle: a Win32 handle belongs to one
    // completion port for life (storage/backend.h), and this one is closed
    // before the engine ever sees the file.
    auto f = File::open(sample_path,
                        FileFlags::Unbuffered | FileFlags::Overlapped | FileFlags::Random);
    if (!f) return std::unexpected(f.error());
    constexpr uint64_t kBlock = 4ull << 20;
    if (f->size() < kBlock * 8) return fail(Err::OutOfRange, "probe file is too small");
    if (qd == 0) qd = 1;
    if (qd > 32) qd = 32;
    const uint64_t span = (f->size() - kBlock) & ~(kPageSize - 1ull);
    std::atomic<uint64_t> moved{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> ts;
    ts.reserve(qd);
    for (uint32_t t = 0; t < qd; ++t) {
        ts.emplace_back([&, t] {
            AlignedBuffer buf(kBlock, kPageSize);
            // A cheap deterministic stride per thread: no RNG, no shared state,
            // and it still lands all over the file rather than in one extent.
            uint64_t x = 0x9E3779B97F4A7C15ull * (t + 1);
            while (!stop.load(std::memory_order_relaxed)) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                const uint64_t off = align_down(x % (span ? span : 1), kPageSize);
                auto r = f->read_at(off, MutBytes(buf.data(), kBlock));
                if (!r) break;
                moved.fetch_add(*r, std::memory_order_relaxed);
            }
        });
    }
    const TimePoint t0 = Clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : ts) t.join();
    const double sec = double((Clock::now() - t0).count()) / 1e9;
    if (sec <= 0.0) return fail(Err::Internal, "probe measured no time");
    return double(moved.load()) / sec / 1e9;
}

IoEngine::~IoEngine() { stop(); }

Result<void> IoEngine::start(std::unique_ptr<Backend> backend, const IoConfig& cfg,
                             Profiler* profiler) {
    if (running_.load(std::memory_order_acquire))
        return fail(Err::FailedPrecondition, "IoEngine is already running");
    if (!backend) return fail(Err::InvalidArgument, "IoEngine needs a backend");
    backend_  = std::move(backend);
    cfg_      = cfg;
    profiler_ = profiler;
    if (cfg_.chunk_bytes > backend_->caps().max_chunk_bytes)
        cfg_.chunk_bytes = backend_->caps().max_chunk_bytes;
    if (cfg_.max_inflight_ops > backend_->caps().max_queue_depth)
        cfg_.max_inflight_ops = backend_->caps().max_queue_depth;
    tune_ = tuning_from_env(cfg_);
    // P1-P3 keep the chunk size the config shipped with even when P0 widened it.
    bg_chunk_bytes_ = cfg_.chunk_bytes;
    if (tune_.p0_chunk_bytes > IoConfig{}.chunk_bytes && bg_chunk_bytes_ > IoConfig{}.chunk_bytes)
        bg_chunk_bytes_ = IoConfig{}.chunk_bytes;
    if (tune_.p0_qd > backend_->caps().max_queue_depth)
        tune_.p0_qd = backend_->caps().max_queue_depth;
    if (tune_.p0_chunk_bytes > backend_->caps().max_chunk_bytes)
        tune_.p0_chunk_bytes = backend_->caps().max_chunk_bytes;
    p0_lat_us_.reserve(1 << 16);
    stopping_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] { dispatcher(); });
    if (tune_.submit_threads > 1) {
        submit_workers_.reserve(tune_.submit_threads);
        for (uint32_t i = 0; i < tune_.submit_threads; ++i)
            submit_workers_.emplace_back([this] { submit_worker(); });
    }
    log_info("IoEngine tuning: {}", tune_.to_string());
    log_debug("IoEngine started on '{}' (chunk {} B, QD {}, {} MiB in flight)",
              backend_->caps().name, cfg_.chunk_bytes, cfg_.max_inflight_ops,
              cfg_.max_inflight_bytes >> 20);
    return {};
}

void IoEngine::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    stopping_.store(true, std::memory_order_release);
    cv_.notify_all();
    sq_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    sq_cv_.notify_all();
    for (auto& t : submit_workers_) if (t.joinable()) t.join();
    submit_workers_.clear();
    if (backend_) backend_->cancel_all();
    backend_.reset();
}

Result<IoRequestId> IoEngine::submit(const IoRequest& in_req, IoCallback cb) {
    if (!running_.load(std::memory_order_acquire))
        return fail(Err::FailedPrecondition, "IoEngine is not running");
    if (!in_req.file || !in_req.file->is_open()) return fail(Err::InvalidArgument, "IoRequest has no open file");

    // Track D2: pick the drive BEFORE anything else looks at req.file, so the
    // alignment and EOF checks below are made against the handle that will
    // actually be read. A request stays on one source for all of its chunks --
    // splitting a 9 MiB run across two drives would make its latency the max of
    // the two rather than either one's.
    IoRequest req = in_req;
    uint32_t source = 0;
    bool routed = false;
    if (mirrors_on_ && (route_classes_ & (1u << static_cast<uint8_t>(req.priority)))) {
        if (auto it = alts_.find(req.file); it != alts_.end()) {
            uint32_t mask = 0;
            for (uint32_t i = 0; i < kMaxIoSources; ++i)
                if (it->second[i] && it->second[i]->is_open()) mask |= 1u << i;
            std::lock_guard lk(src_mutex_);
            const uint32_t pick = pick_source(src_weights_,
                                              std::span<const uint64_t>(src_outstanding_,
                                                                        src_weights_.size()),
                                              mask, req.bytes);
            if (pick < kMaxIoSources && it->second[pick]) {
                source = pick;
                routed = true;
                req.file = it->second[pick];
                src_outstanding_[source] += req.bytes;
                ++src_inflight_[source];
            }
        }
    }
    if (!req.dst)   return fail(Err::InvalidArgument, "IoRequest has no destination");
    if (req.bytes == 0) return fail(Err::InvalidArgument, "IoRequest has zero length");

    // Every failure path below has to give the routed bytes back, or a source
    // that rejected one malformed request would look busy forever.
    struct RouteGuard {
        IoEngine* e; uint32_t src; uint64_t bytes; bool armed;
        ~RouteGuard() {
            if (!armed) return;
            std::lock_guard lk(e->src_mutex_);
            if (e->src_outstanding_[src] >= bytes) e->src_outstanding_[src] -= bytes;
            else e->src_outstanding_[src] = 0;
            if (e->src_inflight_[src]) --e->src_inflight_[src];
        }
    } guard{this, source, req.bytes, routed};

    const uint32_t a = backend_->caps().alignment;
    if (req.file->unbuffered()) {
        if (!is_aligned(req.file_off, a))
            return fail(Err::InvalidArgument,
                        std::format("unbuffered read offset {} is not {}-aligned", req.file_off, a));
        if (!is_aligned(req.dst, a))
            return fail(Err::InvalidArgument, "unbuffered read destination is not sector-aligned");
        if (!is_aligned(req.bytes, a))
            return fail(Err::InvalidArgument,
                        std::format("unbuffered read length {} is not {}-aligned", req.bytes, a));
    }

    // design §5.1 (v0.5): the last tensor of a shard ends at the file's byte
    // length, so the sector-aligned read that covers it asks for up to one
    // sector past EOF. Only the in-file part has to arrive.
    const uint64_t size = req.file->size();
    const uint64_t in_file = (size > req.file_off) ? (size - req.file_off) : 0;

    auto p = std::make_shared<Pending>();
    p->req       = req;
    p->source    = source;
    p->routed    = routed;
    p->cb        = std::move(cb);
    const uint32_t chunk = (req.priority == IoPriority::BlockingMiss)
                               ? tune_.p0_chunk_bytes : bg_chunk_bytes_;
    p->chunks    = plan_chunks(req.file_off, req.bytes, chunk, a);
    p->required_bytes = std::min<uint64_t>(req.bytes, in_file);
    p->queued_at = Clock::now();
    if (p->required_bytes == 0)
        return fail(Err::OutOfRange,
                    std::format("read at {} is entirely past the end of '{}' ({} B)",
                                req.file_off, req.file->path(), size));

    IoRequestId id;
    if (req.priority == IoPriority::BlockingMiss)
        last_p0_ns_.store(std::chrono::duration_cast<Nanos>(Clock::now().time_since_epoch()).count(),
                          std::memory_order_relaxed);
    {
        std::lock_guard lk(mutex_);
        id = next_request_id_++;
        p->id = id;
        if (req.priority == IoPriority::BlockingMiss) {
            p->p0_ahead = p0_outstanding_;
            ++p0_outstanding_;
        }
        queues_[static_cast<uint8_t>(req.priority)].push_back(p);
        ++outstanding_requests_;
    }
    {
        std::lock_guard lk(stats_mutex_);
        ++stats_.requests_submitted;
        stats_.bytes_requested += req.bytes;
        stats_.per_priority_requests[static_cast<uint8_t>(req.priority)] += 1;
        stats_.per_priority_bytes[static_cast<uint8_t>(req.priority)] += req.bytes;
    }
    guard.armed = false;   // finish() owns the accounting from here
    cv_.notify_one();
    return id;
}

Result<std::future<IoResult>> IoEngine::submit_future(const IoRequest& req) {
    auto promise = std::make_shared<std::promise<IoResult>>();
    auto fut = promise->get_future();
    auto r = submit(req, [promise](const IoResult& res) { promise->set_value(res); });
    if (!r) return std::unexpected(r.error());
    return fut;
}

Result<void> IoEngine::cancel(IoRequestId id) {
    std::shared_ptr<Pending> found;
    {
        std::lock_guard lk(mutex_);
        for (auto& q : queues_) {
            auto it = std::find_if(q.begin(), q.end(),
                                   [&](const std::shared_ptr<Pending>& p) { return p->id == id; });
            if (it == q.end()) continue;
            if ((*it)->next_chunk != 0) return fail(Err::FailedPrecondition, "request is already in flight");
            found = *it;
            q.erase(it);
            --outstanding_requests_;
            break;
        }
    }
    if (!found) return fail(Err::NotFound, std::format("no queued request {}", id));
    {
        std::lock_guard lk(stats_mutex_);
        ++stats_.requests_cancelled;
    }
    if (found->cb) {
        IoResult r;
        r.id = found->id; r.key = found->req.key;
        r.status = Status{Err::Cancelled, "cancelled before issue"};
        found->cb(r);
    }
    idle_cv_.notify_all();
    return {};
}

void IoEngine::drain() {
    std::unique_lock lk(mutex_);
    idle_cv_.wait(lk, [this] { return outstanding_requests_ == 0 || !running_.load(std::memory_order_acquire); });
}

uint32_t IoEngine::queued_requests() const {
    std::lock_guard lk(mutex_);
    return outstanding_requests_;
}

IoStats IoEngine::stats() const {
    std::lock_guard lk(stats_mutex_);
    IoStats s = stats_;
    s.disp_submit_threads = tune_.submit_threads;
    if (!src_roots_.empty()) {
        std::lock_guard sl(src_mutex_);
        s.sources.reserve(src_roots_.size());
        for (size_t i = 0; i < src_roots_.size(); ++i) {
            SourceStats e = src_stats_[i];
            e.outstanding_bytes = src_outstanding_[i];
            e.inflight_requests = src_inflight_[i];
            s.sources.push_back(std::move(e));
        }
    }
    // Fold in the currently open busy window so a mid-run reader sees a live
    // utilisation figure rather than a stale one.
    if (busy_) s.busy_ns += static_cast<uint64_t>((Clock::now() - busy_since_).count());
    if (!p0_lat_us_.empty()) {
        std::vector<uint32_t> v = p0_lat_us_;
        std::sort(v.begin(), v.end());
        const size_t i50 = v.size() / 2;
        const size_t i95 = std::min(v.size() - 1, size_t(double(v.size()) * 0.95));
        s.p0_lat_p50_ns = uint64_t(v[i50]) * 1000;
        s.p0_lat_p95_ns = uint64_t(v[i95]) * 1000;
    }
    return s;
}

void IoEngine::reset_stats() {
    std::lock_guard lk(stats_mutex_);
    stats_ = IoStats{};
    p0_lat_us_.clear();
    // The per-source COUNTERS reset; the roots, weights and anything still in
    // flight do not -- resetting outstanding_bytes mid-run would make the
    // router think an empty queue was waiting on it.
    std::lock_guard sl(src_mutex_);
    for (size_t i = 0; i < src_roots_.size(); ++i) {
        src_stats_[i].requests = 0;
        src_stats_[i].bytes = 0;
        src_stats_[i].lat_ns_sum = 0;
    }
}

size_t IoEngine::issue_ready_chunks() {
    size_t issued = 0;
    for (;;) {
        std::shared_ptr<Pending> p;
        Chunk ch{};
        uint64_t cid = 0;
        {
            std::lock_guard lk(mutex_);
            // Strict priority: a queued P0 keeps every lower class from issuing
            // another chunk (design §9.6 preemption).
            for (uint8_t pr = 0; pr < kIoPriorityCount && !p; ++pr) {
                auto& q = queues_[pr];
                while (!q.empty() && q.front()->next_chunk >= q.front()->chunks.size())
                    q.pop_front();            // fully issued; chunk_owner_ keeps it alive
                if (!q.empty()) p = q.front();
            }
            if (!p) break;
            const IoPriority cls = p->req.priority;
            // Track Q1 (b): the queue-depth and in-flight-byte ceilings are a
            // property of the class being issued, so a P0 burst can run the
            // drive deeper than the background classes are allowed to.
            const bool is_p0 = (cls == IoPriority::BlockingMiss);
            const uint32_t qd_cap = is_p0 ? tune_.p0_qd : tune_.bg_qd;
            const uint64_t byte_cap = is_p0 ? tune_.p0_inflight_bytes : tune_.bg_inflight_bytes;
            if (inflight_ops_.load(std::memory_order_relaxed) >= qd_cap) break;
            if (inflight_bytes_.load(std::memory_order_relaxed) >= byte_cap) break;
            if (cls == IoPriority::Lookahead || cls == IoPriority::Backfill ||
                (tune_.throttle_engram && cls == IoPriority::Engram)) {
                const int64_t since = std::chrono::duration_cast<Nanos>(Clock::now().time_since_epoch()).count() -
                                      last_p0_ns_.load(std::memory_order_relaxed);
                uint32_t bg =
                    inflight_class_[uint8_t(IoPriority::Lookahead)].load(std::memory_order_relaxed) +
                    inflight_class_[uint8_t(IoPriority::Backfill)].load(std::memory_order_relaxed);
                if (tune_.throttle_engram)
                    bg += inflight_class_[uint8_t(IoPriority::Engram)].load(std::memory_order_relaxed);
                // Track Q1 (a): while P0 work is recent the background classes
                // hold at most `bg_cap_busy` chunks; 0 means they stop dead.
                if (since < std::chrono::duration_cast<Nanos>(kBackgroundQuiet).count() &&
                    bg >= tune_.bg_cap_busy)
                    break;
            }
            ch  = p->chunks[p->next_chunk++];
            cid = next_chunk_id_++;
            chunk_owner_.emplace(cid, InflightChunk{p, ch.bytes});
            if (is_p0 && !p->issued_once) {
                p->issued_once   = true;
                p->first_issue_at = Clock::now();
                p->bg_at_issue =
                    inflight_class_[uint8_t(IoPriority::Lookahead)].load(std::memory_order_relaxed) +
                    inflight_class_[uint8_t(IoPriority::Backfill)].load(std::memory_order_relaxed) +
                    inflight_class_[uint8_t(IoPriority::Engram)].load(std::memory_order_relaxed);
            }
            if (is_p0) {
                std::lock_guard sk(stats_mutex_);
                ++stats_.p0_chunks_issued;
                stats_.p0_qd_at_issue_sum += inflight_ops_.load(std::memory_order_relaxed);
            }
        }

        ChunkRequest cr;
        cr.chunk_id = cid;
        cr.file     = p->req.file;
        cr.file_off = ch.off;
        cr.bytes    = ch.bytes;
        // Only the part of this chunk that lies inside the file has to arrive.
        const uint64_t fsize = p->req.file->size();
        cr.min_bytes = static_cast<uint32_t>(
            std::min<uint64_t>(ch.bytes, fsize > ch.off ? fsize - ch.off : 0));
        cr.dst      = static_cast<std::byte*>(p->req.dst) + (ch.off - p->req.file_off);

        // The queue slot, the bytes and the class are charged HERE, before the
        // backend call, so the depth caps above hold whether the submit happens
        // on this thread or on a submitter. chunk_owner_ was already populated
        // under the lock, so a completion that lands mid-submit still resolves.
        const TimePoint handed_at = Clock::now();
        ++p->issued_chunks;
        ++issued;
        inflight_class_[static_cast<uint8_t>(p->req.priority)].fetch_add(1, std::memory_order_relaxed);
        const uint32_t ops = inflight_ops_.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t byt = inflight_bytes_.fetch_add(ch.bytes, std::memory_order_relaxed) + ch.bytes;
        {
            std::lock_guard lk(stats_mutex_);
            ++stats_.chunks_submitted;
            // Track Q2 E2: how long the freed queue slot stayed empty. Only the
            // first chunk after a reap closes the gap; the rest of the same
            // issue burst are limited by the backend, not by the refill.
            if (reap_pending_) {
                stats_.disp_refill_ns += uint64_t((handed_at - reaped_at_).count());
                ++stats_.disp_refills;
                reap_pending_ = false;
            }
            if (ops > stats_.peak_inflight_ops)   stats_.peak_inflight_ops = ops;
            if (byt > stats_.peak_inflight_bytes) stats_.peak_inflight_bytes = byt;
            if (!busy_) { busy_ = true; busy_since_ = Clock::now(); }
        }

        if (submit_workers_.empty()) {
            do_submit(PickedChunk{p, cr});          // exactly the old behaviour
        } else {
            {
                std::lock_guard lk(sq_mutex_);
                submit_q_.push_back(PickedChunk{p, cr});
            }
            sq_cv_.notify_one();
        }
    }
    return issued;
}

void IoEngine::do_submit(PickedChunk pc) {
    const TimePoint sub0 = Clock::now();
    auto r = backend_->submit(pc.req);
    const uint64_t sub_ns = uint64_t((Clock::now() - sub0).count());
    {
        std::lock_guard sk(stats_mutex_);
        stats_.disp_submit_ns += sub_ns;
        ++stats_.disp_submits;
        if (sub_ns > stats_.disp_submit_ns_max) stats_.disp_submit_ns_max = sub_ns;
    }
    if (r) return;
    // The IOCP backend reports a failed ReadFile through the completion path
    // and only refuses synchronously when the chunk is unusable or its own
    // queue is full -- and the queue cannot be full, because the engine charges
    // its own slot before calling and its cap is never above the backend's. So
    // this is the unusable case: turn it into a failed completion and let the
    // dispatcher unwind it, which keeps every mutation of request state on one
    // thread (storage/backend.h).
    log_warn("io: backend submit failed: {}", r.error().str());
    ChunkCompletion c;
    c.chunk_id = pc.req.chunk_id;
    c.status   = r.error();
    {
        std::lock_guard lk(fq_mutex_);
        failed_q_.push_back(c);
    }
    cv_.notify_one();
}

void IoEngine::submit_worker() {
    for (;;) {
        PickedChunk pc;
        {
            std::unique_lock lk(sq_mutex_);
            sq_cv_.wait(lk, [this] {
                return !submit_q_.empty() || stopping_.load(std::memory_order_acquire);
            });
            if (submit_q_.empty()) {
                if (stopping_.load(std::memory_order_acquire)) return;
                continue;
            }
            pc = std::move(submit_q_.front());
            submit_q_.pop_front();
        }
        do_submit(std::move(pc));
    }
}

void IoEngine::handle_completion(const ChunkCompletion& c) {
    std::shared_ptr<Pending> p;
    uint32_t charged = 0;
    // Whether the freed queue slot has something waiting for it. Only then is
    // the wall time until the next submit a REFILL gap; otherwise it is just
    // the engine being idle between layers.
    bool ready_to_issue = false;
    {
        std::lock_guard lk(mutex_);
        auto it = chunk_owner_.find(c.chunk_id);
        if (it == chunk_owner_.end()) {
            log_warn("io: completion for unknown chunk {}", c.chunk_id);
            return;
        }
        p = it->second.owner;
        charged = it->second.bytes;
        chunk_owner_.erase(it);
        ++p->done_chunks;
        p->bytes_moved += c.bytes_moved;
        if (!c.ok() && !p->failed) {
            p->failed = true;
            p->status = c.status;
            // Abandon the chunks that were never issued; the destination buffer
            // is now undefined and the owner will drop the slot.
            p->next_chunk = p->chunks.size();
        }
        for (auto& q : queues_) {
            for (const auto& w : q)
                if (w->next_chunk < w->chunks.size()) { ready_to_issue = true; break; }
            if (ready_to_issue) break;
        }
    }

    inflight_ops_.fetch_sub(1, std::memory_order_relaxed);
    inflight_bytes_.fetch_sub(charged, std::memory_order_relaxed);
    inflight_class_[static_cast<uint8_t>(p->req.priority)].fetch_sub(1, std::memory_order_relaxed);
    uint64_t closed_window_ns = 0;
    {
        std::lock_guard lk(stats_mutex_);
        ++stats_.chunks_completed;
        stats_.bytes_completed += c.bytes_moved;
        reaped_at_ = Clock::now();
        reap_pending_ = ready_to_issue;
        if (busy_ && inflight_ops_.load(std::memory_order_relaxed) == 0) {
            closed_window_ns = static_cast<uint64_t>((Clock::now() - busy_since_).count());
            stats_.busy_ns += closed_window_ns;
            busy_ = false;
        }
    }
    // design 13.1's `nvme_util` is the fraction of the token during which the
    // drive had at least one chunk in flight -- the UNION of the in-flight
    // intervals, which is exactly the window just closed. It used to be fed
    // each request's own latency from `finish`, which counts every overlapping
    // interval once per request: at a queue depth of 8 that is up to 8x the
    // union, which is how nvme_util read 6.2 and the profiler's eff GB/s read
    // a sixth of the IoEngine's own (docs/p2_decode.md §5.2).
    if (closed_window_ns && profiler_) profiler_->note_nvme_busy(Nanos(int64_t(closed_window_ns)));

    bool complete;
    {
        std::lock_guard lk(mutex_);
        complete = (p->done_chunks == p->issued_chunks) && (p->next_chunk == p->chunks.size());
    }
    if (complete) finish(std::move(p));
}

void IoEngine::finish(std::shared_ptr<Pending> p) {
    IoResult r;
    r.id          = p->id;
    r.key         = p->req.key;
    r.bytes_moved = p->bytes_moved;
    r.latency     = Clock::now() - p->queued_at;
    r.status      = p->status;
    if (!p->failed && p->bytes_moved < p->required_bytes)
        r.status = Status{Err::Io, std::format("short read: {} of {} bytes (needed {})",
                                               p->bytes_moved, p->req.bytes, p->required_bytes)};

    const bool is_p0 = (p->req.priority == IoPriority::BlockingMiss);
    if (p->routed) {
        // Track D2: the source stops carrying these bytes the moment the last
        // chunk lands, which is also when its latency sample is complete.
        std::lock_guard lk(src_mutex_);
        const uint32_t s = p->source;
        const uint64_t want = p->req.bytes;
        if (src_outstanding_[s] >= want) src_outstanding_[s] -= want; else src_outstanding_[s] = 0;
        if (src_inflight_[s]) --src_inflight_[s];
        ++src_stats_[s].requests;
        src_stats_[s].bytes      += r.bytes_moved;
        src_stats_[s].lat_ns_sum += static_cast<uint64_t>(r.latency.count());
    }
    {
        std::lock_guard lk(stats_mutex_);
        if (r.ok()) ++stats_.requests_completed; else ++stats_.requests_failed;
        const uint64_t lat = static_cast<uint64_t>(r.latency.count());
        stats_.latency_ns_sum += lat;
        if (lat > stats_.latency_ns_max) stats_.latency_ns_max = lat;
        if (is_p0) {
            ++stats_.p0_requests;
            stats_.p0_bytes += r.bytes_moved;
            stats_.p0_lat_ns_sum += lat;
            if (lat > stats_.p0_lat_ns_max) stats_.p0_lat_ns_max = lat;
            const uint64_t wait = p->issued_once
                ? uint64_t((p->first_issue_at - p->queued_at).count()) : lat;
            stats_.p0_queue_wait_ns_sum += wait;
            stats_.p0_service_ns_sum    += (lat > wait) ? (lat - wait) : 0;
            if (p->p0_ahead == 0) { ++stats_.p0_first_n;  stats_.p0_first_lat_ns_sum  += lat; }
            else                  { ++stats_.p0_behind_n; stats_.p0_behind_lat_ns_sum += lat; }
            if (p->bg_at_issue) {
                ++stats_.p0_with_bg_n;
                stats_.p0_bg_inflight_sum += p->bg_at_issue;
            }
            p0_lat_us_.push_back(static_cast<uint32_t>(std::min<uint64_t>(lat / 1000, 0xFFFFFFFFull)));
        }
    }
    if (is_p0 && profiler_) profiler_->note_p0(static_cast<uint64_t>(r.latency.count()));
    // Busy time is reported per in-flight window in handle_completion, not
    // per request here; see the note there.
    if (profiler_) profiler_->note_miss_bytes(r.bytes_moved);
    // The callback BEFORE the request stops counting as outstanding. `drain()`
    // promises that everything issued has finished, and for an expert fill
    // "finished" means the callback has settled the slot: with the notify
    // first, drain could return while the last run's callback was still on
    // its way to ExpertStore::finish_run, and the MoE dispatch found a slot
    // still Filling ("layer 30 expert 212 is not resident after the gate: it
    // was a miss this layer, its slot is filling", docs/p2_decode.md §11.4).
    if (p->cb) {
        const TimePoint cb0 = Clock::now();
        p->cb(r);                 // runs on the dispatcher thread, must be short
        const uint64_t cb_ns = uint64_t((Clock::now() - cb0).count());
        std::lock_guard lk(stats_mutex_);
        stats_.disp_cb_ns += cb_ns;
        ++stats_.disp_cbs;
    }
    {
        std::lock_guard lk(mutex_);
        --outstanding_requests_;
        if (is_p0 && p0_outstanding_) --p0_outstanding_;
    }
    idle_cv_.notify_all();
}

void IoEngine::dispatcher() {
    std::vector<ChunkCompletion> comps(std::max<uint32_t>(cfg_.max_inflight_ops, 8));
    for (;;) {
        const TimePoint t_issue0 = Clock::now();
        const size_t issued = issue_ready_chunks();
        const TimePoint t_issue1 = Clock::now();

        // Synchronous submit refusals, unwound here so that every mutation of
        // request state stays on this thread.
        for (;;) {
            ChunkCompletion fc;
            {
                std::lock_guard lk(fq_mutex_);
                if (failed_q_.empty()) break;
                fc = failed_q_.front();
                failed_q_.pop_front();
            }
            handle_completion(fc);
        }

        size_t got = 0;
        TimePoint t_poll1 = t_issue1, t_handle1 = t_issue1;
        const bool had_inflight = inflight_ops_.load(std::memory_order_relaxed) > 0;
        if (had_inflight) {
            auto n = backend_->poll(comps, std::chrono::milliseconds(1));
            t_poll1 = Clock::now();
            if (!n) {
                log_error("io: backend poll failed: {}", n.error().str());
            } else {
                got = *n;
                for (size_t i = 0; i < got; ++i) handle_completion(comps[i]);
            }
            t_handle1 = Clock::now();
        }
        // Track Q2 E2. Only iterations that actually had work in flight are
        // counted, so an idle engine does not dilute the shares.
        if (had_inflight) {
            std::lock_guard lk(stats_mutex_);
            ++stats_.disp_iters;
            stats_.disp_issue_ns  += uint64_t((t_issue1 - t_issue0).count());
            stats_.disp_poll_ns   += uint64_t((t_poll1 - t_issue1).count());
            stats_.disp_handle_ns += uint64_t((t_handle1 - t_poll1).count());
        }

        if (issued == 0 && got == 0) {
            std::unique_lock lk(mutex_);
            const bool has_work = outstanding_requests_ > 0;
            if (!has_work && stopping_.load(std::memory_order_acquire)) break;
            if (!has_work)
                cv_.wait_for(lk, std::chrono::milliseconds(2));
            else
                cv_.wait_for(lk, std::chrono::microseconds(50));
            if (stopping_.load(std::memory_order_acquire) && outstanding_requests_ == 0) break;
        }
    }
    idle_cv_.notify_all();
}

}  // namespace deepmoe::storage
