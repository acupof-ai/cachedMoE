#include "storage/io_engine.h"

#include <algorithm>
#include <format>

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
    stopping_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] { dispatcher(); });
    log_debug("IoEngine started on '{}' (chunk {} B, QD {}, {} MiB in flight)",
              backend_->caps().name, cfg_.chunk_bytes, cfg_.max_inflight_ops,
              cfg_.max_inflight_bytes >> 20);
    return {};
}

void IoEngine::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    stopping_.store(true, std::memory_order_release);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (backend_) backend_->cancel_all();
    backend_.reset();
}

Result<IoRequestId> IoEngine::submit(const IoRequest& req, IoCallback cb) {
    if (!running_.load(std::memory_order_acquire))
        return fail(Err::FailedPrecondition, "IoEngine is not running");
    if (!req.file || !req.file->is_open()) return fail(Err::InvalidArgument, "IoRequest has no open file");
    if (!req.dst)   return fail(Err::InvalidArgument, "IoRequest has no destination");
    if (req.bytes == 0) return fail(Err::InvalidArgument, "IoRequest has zero length");

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
    p->cb        = std::move(cb);
    p->chunks    = plan_chunks(req.file_off, req.bytes, cfg_.chunk_bytes, a);
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
    // Fold in the currently open busy window so a mid-run reader sees a live
    // utilisation figure rather than a stale one.
    if (busy_) s.busy_ns += static_cast<uint64_t>((Clock::now() - busy_since_).count());
    return s;
}

void IoEngine::reset_stats() {
    std::lock_guard lk(stats_mutex_);
    stats_ = IoStats{};
}

size_t IoEngine::issue_ready_chunks() {
    size_t issued = 0;
    for (;;) {
        std::shared_ptr<Pending> p;
        Chunk ch{};
        uint64_t cid = 0;
        {
            std::lock_guard lk(mutex_);
            if (inflight_ops_.load(std::memory_order_relaxed) >= cfg_.max_inflight_ops) break;
            if (inflight_bytes_.load(std::memory_order_relaxed) >= cfg_.max_inflight_bytes) break;
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
            if (cls == IoPriority::Lookahead || cls == IoPriority::Backfill) {
                const int64_t since = std::chrono::duration_cast<Nanos>(Clock::now().time_since_epoch()).count() -
                                      last_p0_ns_.load(std::memory_order_relaxed);
                const uint32_t bg =
                    inflight_class_[uint8_t(IoPriority::Lookahead)].load(std::memory_order_relaxed) +
                    inflight_class_[uint8_t(IoPriority::Backfill)].load(std::memory_order_relaxed);
                if (since < std::chrono::duration_cast<Nanos>(kBackgroundQuiet).count() &&
                    bg >= kBackgroundOpsWhileBusy)
                    break;
            }
            ch  = p->chunks[p->next_chunk++];
            cid = next_chunk_id_++;
            chunk_owner_.emplace(cid, InflightChunk{p, ch.bytes});
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

        auto r = backend_->submit(cr);
        if (!r) {
            bool give_up = false;
            {
                std::lock_guard lk(mutex_);
                chunk_owner_.erase(cid);
                --p->next_chunk;              // still at the head of its queue
                if (r.error().code == Err::ResourceExhausted) {
                    // The backend's own queue is full: retry after the next poll.
                } else {
                    // A real failure (a bad handle, a closed file). Retrying
                    // would spin forever, so fail the request and let the owner
                    // drop the slot.
                    log_warn("io: backend submit failed: {}", r.error().str());
                    p->failed = true;
                    p->status = r.error();
                    p->next_chunk = p->chunks.size();   // abandon the rest
                    give_up = (p->done_chunks == p->issued_chunks);
                }
            }
            if (give_up) finish(p);           // nothing of this request is in flight
            break;
        }
        ++p->issued_chunks;
        ++issued;
        inflight_class_[static_cast<uint8_t>(p->req.priority)].fetch_add(1, std::memory_order_relaxed);

        const uint32_t ops = inflight_ops_.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t byt = inflight_bytes_.fetch_add(ch.bytes, std::memory_order_relaxed) + ch.bytes;
        {
            std::lock_guard lk(stats_mutex_);
            ++stats_.chunks_submitted;
            if (ops > stats_.peak_inflight_ops)   stats_.peak_inflight_ops = ops;
            if (byt > stats_.peak_inflight_bytes) stats_.peak_inflight_bytes = byt;
            if (!busy_) { busy_ = true; busy_since_ = Clock::now(); }
        }
    }
    return issued;
}

void IoEngine::handle_completion(const ChunkCompletion& c) {
    std::shared_ptr<Pending> p;
    uint32_t charged = 0;
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
    }

    inflight_ops_.fetch_sub(1, std::memory_order_relaxed);
    inflight_bytes_.fetch_sub(charged, std::memory_order_relaxed);
    inflight_class_[static_cast<uint8_t>(p->req.priority)].fetch_sub(1, std::memory_order_relaxed);
    uint64_t closed_window_ns = 0;
    {
        std::lock_guard lk(stats_mutex_);
        ++stats_.chunks_completed;
        stats_.bytes_completed += c.bytes_moved;
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

    {
        std::lock_guard lk(stats_mutex_);
        if (r.ok()) ++stats_.requests_completed; else ++stats_.requests_failed;
        const uint64_t lat = static_cast<uint64_t>(r.latency.count());
        stats_.latency_ns_sum += lat;
        if (lat > stats_.latency_ns_max) stats_.latency_ns_max = lat;
    }
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
    if (p->cb) p->cb(r);          // runs on the dispatcher thread, must be short
    {
        std::lock_guard lk(mutex_);
        --outstanding_requests_;
    }
    idle_cv_.notify_all();
}

void IoEngine::dispatcher() {
    std::vector<ChunkCompletion> comps(std::max<uint32_t>(cfg_.max_inflight_ops, 8));
    for (;;) {
        const size_t issued = issue_ready_chunks();

        size_t got = 0;
        if (inflight_ops_.load(std::memory_order_relaxed) > 0) {
            auto n = backend_->poll(comps, std::chrono::milliseconds(1));
            if (!n) {
                log_error("io: backend poll failed: {}", n.error().str());
            } else {
                got = *n;
                for (size_t i = 0; i < got; ++i) handle_completion(comps[i]);
            }
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
