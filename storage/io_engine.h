// Priority-queued, chunking async I/O front end (design §9.6).
//
// An expert is 18,800,640 B. Reading it as one operation leaves the drive
// idle between completions, so the engine splits every request into chunks
// (2-4 MiB, Q6 decides) and keeps `max_inflight_bytes` in flight across all
// chunks. Requests are drained strictly by priority class:
//
//   P0 BlockingMiss  the GPU is waiting on a timeline value for this expert
//   P1 Lookahead     predicted for layer L+d
//   P2 Engram        264 B rows for a token that is already decided
//   P3 Backfill      idle-time refill of free slots
//
// A P0 arriving mid-stream preempts: no further chunk of a lower class is
// issued until the P0 queue is empty. Chunks already in flight are not
// cancelled -- at 4 MiB they retire in about a millisecond.
//
// Idle-time classes (P1, P3) are also THROTTLED while P0 work is recent: within
// `kBackgroundQuiet` of the last P0 submit they may hold at most
// `kBackgroundOpsWhileBusy` chunks in flight, so a decode that issues a P0 burst
// every layer finds the queue nearly empty instead of eight 4 MiB background
// chunks ahead of it (docs/p4_hitrate.md §5). Once decode goes quiet they get
// the whole queue depth. P2 (the engram's 4 KiB rows) is not throttled.
//
// Ownership/threading:
//   - IoEngine owns the Backend and one dispatcher thread.
//   - submit()/cancel() are safe from any thread (planner, engine, prefetcher).
//   - Completion callbacks run ON THE DISPATCHER THREAD. They must be short:
//     flip a slot to Resident, signal a timeline, wake a condition variable.
//     Doing real work there stalls every other outstanding request.
//   - The destination buffer must stay alive and untouched until the callback
//     fires; the ExpertStore guarantees this by holding the slot in Filling.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/profiler.h"
#include "core/status.h"
#include "core/types.h"
#include "storage/backend.h"
#include "storage/file.h"

namespace deepmoe::storage {

using IoRequestId = uint64_t;

// design §5.4
struct IoRequest {
    ExpertKey  key{};                 // what this is for; {0,0} for non-expert reads
    IoPriority priority = IoPriority::Backfill;
    const File* file    = nullptr;    // borrowed, must outlive the request
    uint64_t   file_off = 0;          // 4 KiB aligned
    uint64_t   bytes    = 0;          // 4 KiB multiple
    void*      dst      = nullptr;    // 4 KiB aligned, GPU-visible on the hot path
    TokenIndex issue_token   = 0;     // for the profiler's per-token attribution
    uint32_t   deadline_layer = 0;    // the layer that will block on this
};

struct IoResult {
    IoRequestId id = 0;
    ExpertKey   key{};
    uint64_t    bytes_moved = 0;
    Nanos       latency{0};
    Status      status{Err::Ok};
    bool ok() const { return status.code == Err::Ok; }
};

using IoCallback = std::function<void(const IoResult&)>;

struct IoStats {
    uint64_t requests_submitted = 0, requests_completed = 0, requests_failed = 0, requests_cancelled = 0;
    uint64_t chunks_submitted = 0, chunks_completed = 0;
    uint64_t bytes_requested = 0, bytes_completed = 0;
    uint64_t per_priority_requests[kIoPriorityCount] = {};
    uint64_t per_priority_bytes[kIoPriorityCount]    = {};
    uint64_t busy_ns = 0;              // wall time with at least one chunk in flight
    uint64_t latency_ns_sum = 0, latency_ns_max = 0;
    uint32_t peak_inflight_ops = 0;
    uint64_t peak_inflight_bytes = 0;

    double mean_latency_ms() const {
        return requests_completed ? latency_ns_sum / 1e6 / static_cast<double>(requests_completed) : 0.0;
    }
    // The §9.8 "effective NVMe GB/s" figure: bytes over the busy window only.
    double effective_gbps() const {
        return busy_ns ? static_cast<double>(bytes_completed) / static_cast<double>(busy_ns) : 0.0;
    }
    std::string to_string() const;
};

class IoEngine {
public:
    IoEngine() = default;
    ~IoEngine();

    IoEngine(const IoEngine&) = delete;
    IoEngine& operator=(const IoEngine&) = delete;

    // Takes ownership of `backend` and starts the dispatcher thread.
    Result<void> start(std::unique_ptr<Backend> backend, const IoConfig& cfg,
                       Profiler* profiler = nullptr);
    // Drains outstanding work, then joins. Safe to call twice.
    void stop();
    bool running() const { return running_.load(std::memory_order_acquire); }

    // Queues a request. `cb` runs on the dispatcher thread when every chunk of
    // the request has completed (or the first one failed).
    Result<IoRequestId> submit(const IoRequest& req, IoCallback cb);

    // Same, but hands back a future. Convenient for benchmarks and tests; the
    // runtime uses the callback form so nothing blocks the submit thread.
    Result<std::future<IoResult>> submit_future(const IoRequest& req);

    // Best-effort: drops the request if no chunk has been issued yet.
    // Returns NotFound when it is already in flight or finished.
    Result<void> cancel(IoRequestId id);

    // Blocks until every queued request has completed.
    void drain();

    uint32_t queued_requests() const;
    uint32_t inflight_chunks() const { return inflight_ops_.load(std::memory_order_relaxed); }
    uint64_t inflight_bytes()  const { return inflight_bytes_.load(std::memory_order_relaxed); }

    IoStats  stats() const;
    void     reset_stats();

    const BackendCaps& backend_caps() const { return backend_->caps(); }
    const IoConfig&    config() const { return cfg_; }

    // Splits [off, off+bytes) into at most `max_chunk` sized, alignment-
    // respecting pieces. Exposed for the unit test of design §9.6's chunking.
    struct Chunk { uint64_t off; uint32_t bytes; };
    static std::vector<Chunk> plan_chunks(uint64_t off, uint64_t bytes,
                                          uint32_t max_chunk, uint32_t alignment);

private:
    struct Pending {
        IoRequestId id = 0;
        IoRequest   req{};
        IoCallback  cb;
        std::vector<Chunk> chunks;
        // How many bytes must actually arrive. Less than req.bytes only when the
        // aligned read runs past EOF (storage/backend.h ChunkRequest::min_bytes).
        uint64_t required_bytes = 0;
        size_t   next_chunk    = 0;  // index of the first not-yet-issued chunk
        size_t   issued_chunks = 0;  // handed to the backend (<= next_chunk after a rollback)
        size_t   done_chunks   = 0;
        uint64_t bytes_moved   = 0;
        TimePoint queued_at{};
        Status    status{Err::Ok};
        bool      failed = false;
    };

    // What an in-flight chunk is charged against, so the accounting unwinds
    // correctly even when a chunk fails with zero bytes moved.
    struct InflightChunk {
        std::shared_ptr<Pending> owner;
        uint32_t bytes = 0;
    };

    void dispatcher();                  // the single I/O policy thread
    size_t issue_ready_chunks();        // returns how many chunks were handed to the backend
    void   handle_completion(const ChunkCompletion& c);
    void   finish(std::shared_ptr<Pending> p);
    Pending* top_of_queue();            // highest-priority non-empty queue head

public:
    static constexpr uint32_t kBackgroundOpsWhileBusy = 1;
    static constexpr std::chrono::milliseconds kBackgroundQuiet{100};
    // Chunks of each class in flight right now (tests, the bench).
    uint32_t inflight_chunks(IoPriority p) const {
        return inflight_class_[static_cast<uint8_t>(p)].load(std::memory_order_relaxed);
    }
private:
    std::atomic<uint32_t> inflight_class_[kIoPriorityCount] = {};
    std::atomic<int64_t>  last_p0_ns_{INT64_MIN / 2};

    std::unique_ptr<Backend> backend_;
    IoConfig   cfg_{};
    Profiler*  profiler_ = nullptr;

    std::thread            thread_;
    std::atomic<bool>      running_{false};
    std::atomic<bool>      stopping_{false};

    mutable std::mutex     mutex_;
    std::condition_variable cv_;          // dispatcher wakes on new work
    std::condition_variable idle_cv_;     // drain() waits on this
    std::deque<std::shared_ptr<Pending>> queues_[kIoPriorityCount];
    std::unordered_map<uint64_t, InflightChunk> chunk_owner_;  // chunk_id -> owner + charged bytes
    uint64_t next_request_id_ = 1;
    uint64_t next_chunk_id_   = 1;
    uint32_t outstanding_requests_ = 0;

    std::atomic<uint32_t> inflight_ops_{0};
    std::atomic<uint64_t> inflight_bytes_{0};
    TimePoint busy_since_{};
    bool      busy_ = false;

    mutable std::mutex stats_mutex_;
    IoStats            stats_{};
};

}  // namespace deepmoe::storage
