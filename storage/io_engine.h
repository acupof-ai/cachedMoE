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
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/config.h"
#include "core/profiler.h"
#include "core/status.h"
#include "core/types.h"
#include "storage/backend.h"
#include "storage/file.h"
#include "storage/source_router.h"

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

    // --- P0 (BlockingMiss) queue accounting, docs/p4_p0_queue.md ---------
    // The decode's stall is the sum of these latencies, so they are the only
    // ones with a per-request breakdown. `queue_wait` is submit -> first chunk
    // handed to the backend; `service` is first chunk -> last chunk done.
    uint64_t p0_requests = 0, p0_bytes = 0;
    uint64_t p0_lat_ns_sum = 0, p0_lat_ns_max = 0;
    uint64_t p0_queue_wait_ns_sum = 0;
    uint64_t p0_service_ns_sum = 0;
    // Split by whether another P0 was already outstanding when this one was
    // queued: "first" is the head of a layer's burst (the QD ramp), "behind"
    // is every later miss of the same layer.
    uint64_t p0_first_n = 0, p0_first_lat_ns_sum = 0;
    uint64_t p0_behind_n = 0, p0_behind_lat_ns_sum = 0;
    // Non-P0 chunks the drive was already carrying when this P0's first chunk
    // went out -- the contention the hypothesis is about.
    uint64_t p0_with_bg_n = 0, p0_bg_inflight_sum = 0;
    // Filled in by IoEngine::stats() from the retained per-request samples.
    uint64_t p0_lat_p50_ns = 0, p0_lat_p95_ns = 0;
    // Mean chunk queue depth seen at issue, over P0 chunks only.
    uint64_t p0_chunks_issued = 0, p0_qd_at_issue_sum = 0;

    // --- dispatcher refill accounting, Track Q2 E2 --------------------------
    // The dispatcher is one thread running issue -> poll -> handle -> issue.
    // Anything it spends in `handle` (which is where completion CALLBACKS run,
    // and an expert fill's callback settles the slot) is time in which no new
    // chunk can go to the drive, however deep the queue nominally is. These
    // four counters split one loop iteration so that "the queue never fills"
    // can be attributed to a phase instead of guessed at.
    uint64_t disp_iters     = 0;   // loop iterations with work in flight
    uint64_t disp_issue_ns  = 0;   // inside issue_ready_chunks()
    uint64_t disp_poll_ns   = 0;   // inside backend_->poll()
    uint64_t disp_handle_ns = 0;   // inside handle_completion(), callbacks included
    uint64_t disp_cb_ns     = 0;   // the callbacks alone, a subset of handle
    uint64_t disp_cbs       = 0;   // how many callbacks that was
    // Backend::submit() alone, a subset of issue: on Windows that is a
    // synchronous ReadFile, which has to probe-and-lock the destination pages
    // before it can return ERROR_IO_PENDING.
    uint64_t disp_submit_ns = 0, disp_submits = 0, disp_submit_ns_max = 0;
    uint32_t disp_submit_threads = 1;   // how many threads shared that time
    // Wall time from a chunk completion being reaped to the next chunk reaching
    // the backend, counted only when the engine had a chunk ready to issue --
    // the refill gap the drive sees.
    uint64_t disp_refill_ns = 0, disp_refills = 0;

    double disp_cb_mean_us() const {
        return disp_cbs ? disp_cb_ns / 1e3 / double(disp_cbs) : 0.0;
    }
    double disp_submit_mean_us() const {
        return disp_submits ? disp_submit_ns / 1e3 / double(disp_submits) : 0.0;
    }
    double disp_refill_mean_us() const {
        return disp_refills ? disp_refill_ns / 1e3 / double(disp_refills) : 0.0;
    }

    // --- per-source accounting, Track D2 (docs/p4_dual_source.md) ----------
    // Empty unless a mirror was configured. Index 0 is always the primary
    // model directory, so a one-entry vector and an empty one mean the same
    // thing and the no-mirror run prints nothing extra.
    std::vector<SourceStats> sources;

    double p0_mean_ms() const {
        return p0_requests ? p0_lat_ns_sum / 1e6 / double(p0_requests) : 0.0;
    }

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

    // --- second read source (Track D2, docs/p4_dual_source.md) --------------
    // Declares the read sources. `roots[0]` is the primary model directory and
    // must always be present; every further entry is a mirror holding
    // byte-identical copies of some or all of the same shards. `weights` are
    // GB/s (measured by probe_source_gbps or DEEPMOE_MIRROR_WEIGHTS) and only
    // their ratio matters. With fewer than two roots the router stays off and
    // submit() is byte-for-byte the function it was before.
    //
    // Call after start() and before any submit(): the mirror table is written
    // once and read lock-free on the hot path.
    void set_sources(const std::vector<std::string>& roots, const std::vector<double>& weights);
    // Registers `alt` as source `src`'s handle for the shard whose primary
    // handle is `primary`. Both must be open and the same size.
    Result<void> add_mirror(const File* primary, uint32_t src, const File* alt);
    bool     mirrors_enabled() const { return mirrors_on_; }
    uint32_t source_count() const { return static_cast<uint32_t>(src_roots_.size()); }

    // The startup probe: 4 MiB random reads at queue depth `qd` against
    // `sample_path` for `ms` milliseconds, returning GB/s. Opens and closes its
    // own handle, so it must not be pointed at a File already handed to a
    // backend (storage/backend.h: one IOCP port per handle, for life).
    static Result<double> probe_source_gbps(const std::string& sample_path,
                                            uint32_t ms = 1000, uint32_t qd = 8);

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
        TimePoint first_issue_at{};      // when the backend took chunk 0
        bool      issued_once = false;
        uint32_t  bg_at_issue = 0;       // non-P0 chunks in flight at that moment
        uint32_t  p0_ahead    = 0;       // other P0 requests outstanding at submit
        uint32_t  source      = 0;       // which read source served it (Track D2)
        bool      routed      = false;   // charged against src_outstanding_[source]
        Status    status{Err::Ok};
        bool      failed = false;
    };

    // What an in-flight chunk is charged against, so the accounting unwinds
    // correctly even when a chunk fails with zero bytes moved.
    struct InflightChunk {
        std::shared_ptr<Pending> owner;
        uint32_t bytes = 0;
    };

    // One chunk that policy has already committed to: its queue slot, its
    // bytes and its chunk id are all accounted for, and the only thing left is
    // the call into the backend. Track Q2 E2 made that call the bottleneck --
    // ReadFile into path A memory costs ~700 us -- so it can be handed to a
    // submitter thread instead of running on the dispatcher.
    struct PickedChunk {
        std::shared_ptr<Pending> owner;
        ChunkRequest req{};
    };

    void dispatcher();                  // the single I/O policy thread
    void submit_worker();               // optional: drains submit_q_
    void do_submit(PickedChunk pc);     // the backend call plus its failure path
    size_t issue_ready_chunks();        // returns how many chunks were handed to the backend
    void   handle_completion(const ChunkCompletion& c);
    void   finish(std::shared_ptr<Pending> p);
    Pending* top_of_queue();            // highest-priority non-empty queue head

public:
    static constexpr uint32_t kBackgroundOpsWhileBusy = 1;
    // Track Q2: Backend::submit() is a synchronous ReadFile, and into path A
    // (DEVICE_LOCAL|HOST_VISIBLE) memory it costs ~700 us per 4 MiB chunk
    // against ~70 us into ordinary host pages -- the kernel has to probe and
    // lock the destination pages before the transfer can even start. On one
    // thread that caps issue at ~1.4 chunks/ms, so a layer's burst of ~10
    // chunks takes ~7 ms just to reach the drive and the queue never fills
    // (mean depth 3.80 of 8). Submitting from several threads decouples the
    // issue rate from that cost. Measured on the 4-turn chat: 1 -> 8 threads
    // with the deeper P0 queue below is +4.4% tok/s. docs/p4_p0_queue.md §9.
    static constexpr uint32_t kDefaultSubmitThreads = 8;
    static constexpr std::chrono::milliseconds kBackgroundQuiet{100};

    // Track Q1 knobs (docs/p4_p0_queue.md). All default to today's behaviour,
    // so an A/B is an environment variable and not a rebuild.
    //   DEEPMOE_IO_BG_CAP_BUSY   non-P0 chunks allowed in flight while P0 work
    //                            is recent (default kBackgroundOpsWhileBusy = 1)
    //   DEEPMOE_IO_BG_THROTTLE_P2  1 = the engram class yields too (default 0)
    //   DEEPMOE_IO_P0_QD         chunk queue depth used while the head of the
    //                            queue is a P0 (default = cfg.max_inflight_ops)
    //   DEEPMOE_IO_P0_INFLIGHT_MB in-flight byte ceiling for the same case
    //   DEEPMOE_IO_P0_CHUNK_MB   chunk size for P0 requests (default = cfg)
    //   DEEPMOE_IO_SUBMIT_THREADS  how many threads call Backend::submit.
    //                            1 (the default before Track Q2) means the
    //                            dispatcher does it inline; N > 1 starts N
    //                            submitter threads and the dispatcher only
    //                            decides which chunk goes next.
    struct Tuning {
        uint32_t bg_cap_busy      = kBackgroundOpsWhileBusy;
        bool     throttle_engram  = false;
        uint32_t p0_qd            = 0;          // 0 = use cfg_.max_inflight_ops
        uint64_t p0_inflight_bytes = 0;         // 0 = use cfg_.max_inflight_bytes
        uint32_t p0_chunk_bytes   = 0;          // 0 = use cfg_.chunk_bytes
        // The ceilings the background classes keep when P0's are raised, so
        // "deeper queue for P0" does not silently become "deeper queue for the
        // backfill as well".
        uint32_t bg_qd            = 0;
        uint64_t bg_inflight_bytes = 0;
        uint32_t submit_threads   = kDefaultSubmitThreads;
        std::string to_string() const;
    };
    const Tuning& tuning() const { return tune_; }

    // The backend's queue depth is fixed at construction from the IoConfig, so
    // a raised P0 depth has to be reflected there before the backend is made.
    // Call this on the IoConfig once, before make_default_backend().
    static void widen_for_env(IoConfig& cfg);
    // Chunks of each class in flight right now (tests, the bench).
    uint32_t inflight_chunks(IoPriority p) const {
        return inflight_class_[static_cast<uint8_t>(p)].load(std::memory_order_relaxed);
    }
private:
    static Tuning tuning_from_env(const IoConfig& cfg);
    Tuning tune_{};
    uint32_t bg_chunk_bytes_ = 0;         // chunk size for P1-P3
    std::vector<uint32_t> p0_lat_us_;     // one sample per completed P0, for p50/p95
    uint32_t p0_outstanding_ = 0;         // P0 requests submitted but not finished
    std::atomic<uint32_t> inflight_class_[kIoPriorityCount] = {};
    std::atomic<int64_t>  last_p0_ns_{INT64_MIN / 2};

    std::unique_ptr<Backend> backend_;
    IoConfig   cfg_{};
    Profiler*  profiler_ = nullptr;

    std::thread            thread_;
    std::vector<std::thread> submit_workers_;
    mutable std::mutex       sq_mutex_;
    std::condition_variable  sq_cv_;
    std::deque<PickedChunk>  submit_q_;
    // Submit failures the backend reports synchronously. They are drained and
    // turned into completions by the dispatcher, so request state stays
    // single-threaded (storage/backend.h).
    std::mutex                  fq_mutex_;
    std::deque<ChunkCompletion> failed_q_;
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
    // Set when handle_completion() frees a queue slot, cleared by the next
    // chunk that reaches the backend: the two ends of the refill gap.
    TimePoint reaped_at_{};
    bool      reap_pending_ = false;

    // --- Track D2: the mirror table and the router's live state ------------
    // `alts_[primary][s]` is source s's handle for that shard, null when that
    // source does not hold it. Built at startup, never mutated afterwards, so
    // the lookup on the submit path needs no lock; only the byte counters do.
    std::vector<std::string> src_roots_;
    std::vector<double>      src_weights_;
    std::unordered_map<const File*, std::array<const File*, kMaxIoSources>> alts_;
    bool     mirrors_on_    = false;
    uint32_t route_classes_ = (1u << static_cast<uint8_t>(IoPriority::BlockingMiss)) |
                              (1u << static_cast<uint8_t>(IoPriority::Backfill));
    mutable std::mutex src_mutex_;
    uint64_t src_outstanding_[kMaxIoSources] = {};
    uint32_t src_inflight_[kMaxIoSources]    = {};
    SourceStats src_stats_[kMaxIoSources]{};

    mutable std::mutex stats_mutex_;
    IoStats            stats_{};
};

}  // namespace deepmoe::storage
