// IOCP backend: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED reads completing
// on a small pool of dedicated threads (design §9.6).
//
// Ownership/threading:
//   - `submit` runs on the IoEngine dispatcher thread only.
//   - N completion threads (IoConfig::completion_threads, 2 by default) sit in
//     GetQueuedCompletionStatus and push finished chunks onto `done_`.
//   - `poll` drains `done_` from the dispatcher thread.
//   The OVERLAPPED blocks are pooled: an in-flight one is owned by the kernel
//   until its completion packet is dequeued, so they are never freed early.
#if defined(_WIN32)

#include <atomic>
#include <format>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_set>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "core/align.h"
#include "core/log.h"
#include "storage/backend.h"

namespace deepmoe::storage {
namespace {

struct IocpOp {
    OVERLAPPED ov{};          // must be first: the completion packet points at it
    uint64_t   chunk_id = 0;
    uint32_t   bytes    = 0;
};

class IocpBackend final : public Backend {
public:
    explicit IocpBackend(const IoConfig& cfg) {
        caps_.name             = "iocp";
        caps_.alignment        = static_cast<uint32_t>(kPageSize);
        caps_.max_queue_depth  = cfg.max_inflight_ops ? cfg.max_inflight_ops : 64;
        caps_.max_chunk_bytes  = 64u << 20;
        caps_.unbuffered       = cfg.unbuffered;
        caps_.direct_to_device_memory = true;   // the dst is whatever the caller mapped
        threads_ = cfg.completion_threads ? cfg.completion_threads : 2;
    }

    ~IocpBackend() override { shutdown(); }

    Result<void> init() {
        port_ = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, threads_);
        if (!port_)
            return fail(Err::Io, "CreateIoCompletionPort", static_cast<uint32_t>(::GetLastError()));
        workers_.reserve(threads_);
        for (uint32_t i = 0; i < threads_; ++i) workers_.emplace_back([this] { completion_loop(); });
        return {};
    }

    const BackendCaps& caps() const override { return caps_; }

    Result<void> submit(const ChunkRequest& req) override {
        if (!req.file || !req.file->is_open()) return fail(Err::InvalidArgument, "chunk has no open file");
        if (inflight_.load(std::memory_order_relaxed) >= caps_.max_queue_depth)
            return fail(Err::ResourceExhausted, "IOCP queue is full");
        if (auto r = associate(static_cast<HANDLE>(req.file->native())); !r) return r;

        IocpOp* op = acquire_op();
        op->chunk_id = req.chunk_id;
        op->bytes    = req.bytes;
        op->ov = OVERLAPPED{};
        op->ov.Offset     = static_cast<DWORD>(req.file_off & 0xFFFFFFFFull);
        op->ov.OffsetHigh = static_cast<DWORD>(req.file_off >> 32);

        inflight_.fetch_add(1, std::memory_order_relaxed);
        DWORD moved = 0;
        const BOOL ok = ::ReadFile(static_cast<HANDLE>(req.file->native()), req.dst,
                                   static_cast<DWORD>(req.bytes), &moved, &op->ov);
        if (!ok) {
            const DWORD e = ::GetLastError();
            if (e != ERROR_IO_PENDING) {
                inflight_.fetch_sub(1, std::memory_order_relaxed);
                ChunkCompletion c;
                c.chunk_id = req.chunk_id;
                c.status = (e == ERROR_HANDLE_EOF)
                             ? Status{Err::OutOfRange, "read past end of file", e}
                             : Status{Err::Io, std::format("ReadFile off {} len {}", req.file_off, req.bytes), e};
                release_op(op);
                push(c);
                return {};      // reported through the completion path, not as a submit error
            }
        }
        // On success the packet is still queued to the port (we do not set
        // FILE_SKIP_COMPLETION_PORT_ON_SUCCESS), so there is exactly one
        // completion per submit either way.
        return {};
    }

    Result<size_t> poll(std::span<ChunkCompletion> out, std::chrono::milliseconds timeout) override {
        if (out.empty()) return size_t{0};
        std::unique_lock lk(done_mutex_);
        if (done_.empty() && timeout.count() > 0)
            done_cv_.wait_for(lk, timeout, [this] { return !done_.empty(); });
        size_t n = 0;
        while (n < out.size() && !done_.empty()) {
            out[n++] = done_.front();
            done_.pop_front();
        }
        return n;
    }

    uint32_t inflight() const override { return inflight_.load(std::memory_order_relaxed); }

    void cancel_all() override {
        std::lock_guard lk(assoc_mutex_);
        for (HANDLE h : associated_) ::CancelIoEx(h, nullptr);
    }

    void shutdown() {
        if (!port_) return;
        stopping_.store(true, std::memory_order_release);
        cancel_all();
        for (uint32_t i = 0; i < threads_; ++i)
            ::PostQueuedCompletionStatus(port_, 0, 0, nullptr);   // null OVERLAPPED = quit
        for (auto& t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        ::CloseHandle(port_);
        port_ = nullptr;
        for (IocpOp* op : pool_) delete op;
        pool_.clear();
    }

private:
    Result<void> associate(HANDLE h) {
        std::lock_guard lk(assoc_mutex_);
        if (associated_.count(h)) return {};
        if (!::CreateIoCompletionPort(h, port_, reinterpret_cast<ULONG_PTR>(h), 0))
            return fail(Err::Io, "CreateIoCompletionPort(file)", static_cast<uint32_t>(::GetLastError()));
        associated_.insert(h);
        return {};
    }

    IocpOp* acquire_op() {
        std::lock_guard lk(pool_mutex_);
        if (pool_.empty()) return new IocpOp();
        IocpOp* op = pool_.back();
        pool_.pop_back();
        return op;
    }
    void release_op(IocpOp* op) {
        std::lock_guard lk(pool_mutex_);
        if (pool_.size() < 512) pool_.push_back(op); else delete op;
    }

    void push(const ChunkCompletion& c) {
        {
            std::lock_guard lk(done_mutex_);
            done_.push_back(c);
        }
        done_cv_.notify_one();
    }

    void completion_loop() {
        for (;;) {
            DWORD moved = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* ov = nullptr;
            const BOOL ok = ::GetQueuedCompletionStatus(port_, &moved, &key, &ov, INFINITE);
            if (!ov) break;                       // the shutdown packet
            IocpOp* op = reinterpret_cast<IocpOp*>(ov);

            ChunkCompletion c;
            c.chunk_id    = op->chunk_id;
            c.bytes_moved = ok ? static_cast<uint32_t>(moved) : 0;
            if (!ok) {
                const DWORD e = ::GetLastError();
                c.status = (e == ERROR_OPERATION_ABORTED)
                             ? Status{Err::Cancelled, "io cancelled", e}
                             : Status{Err::Io, "overlapped read failed", e};
            } else if (moved != op->bytes) {
                c.status = Status{Err::Io, std::format("short read: {} of {}", moved, op->bytes)};
            }
            const uint32_t expected = op->bytes;
            (void)expected;
            release_op(op);
            inflight_.fetch_sub(1, std::memory_order_relaxed);
            push(c);
        }
    }

    BackendCaps caps_;
    uint32_t    threads_ = 2;
    HANDLE      port_    = nullptr;
    std::atomic<bool>     stopping_{false};
    std::atomic<uint32_t> inflight_{0};

    std::vector<std::thread> workers_;

    std::mutex                 assoc_mutex_;
    std::unordered_set<HANDLE> associated_;

    std::mutex             pool_mutex_;
    std::vector<IocpOp*>   pool_;

    std::mutex                  done_mutex_;
    std::condition_variable     done_cv_;
    std::deque<ChunkCompletion> done_;
};

}  // namespace

Result<std::unique_ptr<Backend>> make_iocp_backend(const IoConfig& cfg) {
    auto b = std::make_unique<IocpBackend>(cfg);
    if (auto r = b->init(); !r) return std::unexpected(r.error());
    return std::unique_ptr<Backend>(std::move(b));
}

Result<std::unique_ptr<Backend>> make_io_uring_backend(const IoConfig&) {
    return fail(Err::Unavailable, "io_uring is a Linux backend");
}

Result<std::unique_ptr<Backend>> make_default_backend(const IoConfig& cfg) {
    if (auto ds = make_directstorage_backend(cfg)) return ds;
    return make_iocp_backend(cfg);
}

}  // namespace deepmoe::storage

#endif  // _WIN32
