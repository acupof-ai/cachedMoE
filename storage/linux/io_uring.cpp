// io_uring backend, written against the raw kernel interface (no liburing --
// design's "no third-party deps" rule). Linux is not a target platform for
// deepMoE; this exists so the portable half of the runtime can be built and
// unit-tested in CI with a real async backend behind it (design §14).
//
// Ownership/threading: one ring, driven exclusively by the IoEngine dispatcher
// thread. The kernel owns the SQ/CQ shared mappings; this class owns the ring
// fd and the three mmaps and unmaps them in the destructor.
#if defined(__linux__)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>
#include <memory>
#include <thread>

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "core/align.h"
#include "storage/backend.h"

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif

namespace deepmoe::storage {
namespace {

int sys_io_uring_setup(unsigned entries, struct io_uring_params* p) {
    return static_cast<int>(::syscall(__NR_io_uring_setup, entries, p));
}
int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return static_cast<int>(::syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags,
                                      nullptr, 0));
}

// The ring head/tail words are shared with the kernel; the acquire/release
// pairing below is the one documented in the io_uring manpages.
inline unsigned load_acquire(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline void     store_release(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

class IoUringBackend final : public Backend {
public:
    explicit IoUringBackend(const IoConfig& cfg) {
        caps_.name            = "io_uring";
        caps_.alignment       = static_cast<uint32_t>(kPageSize);
        caps_.max_queue_depth = cfg.max_inflight_ops ? cfg.max_inflight_ops : 64;
        caps_.max_chunk_bytes = 64u << 20;
        caps_.unbuffered      = cfg.unbuffered;
        caps_.direct_to_device_memory = true;
        entries_ = caps_.max_queue_depth;
    }

    ~IoUringBackend() override { teardown(); }

    Result<void> init() {
        struct io_uring_params p {};
        std::memset(&p, 0, sizeof p);
        // Round the depth up to a power of two, which io_uring_setup requires.
        unsigned e = 1;
        while (e < entries_) e <<= 1;
        ring_fd_ = sys_io_uring_setup(e, &p);
        if (ring_fd_ < 0)
            return fail(Err::Unavailable, "io_uring_setup", static_cast<uint32_t>(errno));
        params_ = p;

        size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        size_t cq_sz = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);
        const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) { sq_sz = cq_sz = std::max(sq_sz, cq_sz); }

        sq_map_ = ::mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                         ring_fd_, IORING_OFF_SQ_RING);
        if (sq_map_ == MAP_FAILED) { teardown(); return fail(Err::Io, "mmap(SQ ring)", static_cast<uint32_t>(errno)); }
        sq_map_size_ = sq_sz;

        if (single) {
            cq_map_ = sq_map_;
            cq_map_size_ = 0;                       // not separately unmapped
        } else {
            cq_map_ = ::mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                             ring_fd_, IORING_OFF_CQ_RING);
            if (cq_map_ == MAP_FAILED) { teardown(); return fail(Err::Io, "mmap(CQ ring)", static_cast<uint32_t>(errno)); }
            cq_map_size_ = cq_sz;
        }

        sqe_map_size_ = p.sq_entries * sizeof(struct io_uring_sqe);
        sqe_map_ = ::mmap(nullptr, sqe_map_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                          ring_fd_, IORING_OFF_SQES);
        if (sqe_map_ == MAP_FAILED) { teardown(); return fail(Err::Io, "mmap(SQEs)", static_cast<uint32_t>(errno)); }

        auto* sq = static_cast<unsigned char*>(sq_map_);
        auto* cq = static_cast<unsigned char*>(cq_map_);
        sq_head_ = reinterpret_cast<unsigned*>(sq + p.sq_off.head);
        sq_tail_ = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
        sq_mask_ = reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
        sq_array_ = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
        cq_head_ = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
        cq_tail_ = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
        cq_mask_ = reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
        cqes_    = reinterpret_cast<struct io_uring_cqe*>(cq + p.cq_off.cqes);
        sqes_    = static_cast<struct io_uring_sqe*>(sqe_map_);
        caps_.max_queue_depth = p.sq_entries;
        return {};
    }

    const BackendCaps& caps() const override { return caps_; }

    Result<void> submit(const ChunkRequest& req) override {
        if (!req.file || !req.file->is_open()) return fail(Err::InvalidArgument, "chunk has no open file");
        if (inflight_ >= caps_.max_queue_depth) return fail(Err::ResourceExhausted, "SQ is full");

        const unsigned tail = *sq_tail_;
        const unsigned head = load_acquire(sq_head_);
        if (tail - head >= params_.sq_entries) return fail(Err::ResourceExhausted, "SQ is full");

        const unsigned index = tail & *sq_mask_;
        struct io_uring_sqe* sqe = &sqes_[index];
        std::memset(sqe, 0, sizeof *sqe);
        sqe->opcode    = IORING_OP_READ;
        sqe->fd        = req.file->native();
        sqe->off       = req.file_off;
        sqe->addr      = reinterpret_cast<uint64_t>(req.dst);
        sqe->len       = req.bytes;
        sqe->user_data = req.chunk_id;
        sq_array_[index] = index;
        store_release(sq_tail_, tail + 1);

        const int r = sys_io_uring_enter(ring_fd_, 1, 0, 0);
        if (r < 0) return fail(Err::Io, "io_uring_enter(submit)", static_cast<uint32_t>(errno));
        ++inflight_;
        // The in-file part of the chunk is what must arrive; a read that
        // straddles EOF is legally short (storage/backend.h).
        pending_bytes_[req.chunk_id & (kPendingMask)] =
            req.min_bytes ? req.min_bytes : req.bytes;
        return {};
    }

    Result<size_t> poll(std::span<ChunkCompletion> out, std::chrono::milliseconds timeout) override {
        if (out.empty()) return size_t{0};
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        size_t n = 0;
        for (;;) {
            n += drain(out.subspan(n));
            if (n > 0 || timeout.count() == 0) break;
            if (std::chrono::steady_clock::now() >= deadline) break;
            // The kernel has no "wait with timeout" on enter(); an IORING_OP_TIMEOUT
            // SQE would be the production answer. CI only needs correctness.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        return n;
    }

    uint32_t inflight() const override { return inflight_; }

    void cancel_all() override { /* the ring is torn down at shutdown; nothing partial survives */ }

private:
    static constexpr uint64_t kPendingMask = 1023;

    size_t drain(std::span<ChunkCompletion> out) {
        size_t n = 0;
        unsigned head = *cq_head_;
        const unsigned tail = load_acquire(cq_tail_);
        while (head != tail && n < out.size()) {
            const struct io_uring_cqe& cqe = cqes_[head & *cq_mask_];
            ChunkCompletion c;
            c.chunk_id = cqe.user_data;
            if (cqe.res < 0) {
                c.status = Status{Err::Io, "io_uring read failed", static_cast<uint32_t>(-cqe.res)};
            } else {
                c.bytes_moved = static_cast<uint32_t>(cqe.res);
                const uint32_t want = pending_bytes_[c.chunk_id & kPendingMask];
                if (want && c.bytes_moved < want)
                    c.status = Status{Err::Io, std::format("short read: {} of {}", c.bytes_moved, want)};
            }
            out[n++] = c;
            ++head;
            --inflight_;
        }
        store_release(cq_head_, head);
        return n;
    }

    void teardown() {
        if (sqe_map_ && sqe_map_ != MAP_FAILED) ::munmap(sqe_map_, sqe_map_size_);
        if (cq_map_size_ && cq_map_ && cq_map_ != MAP_FAILED) ::munmap(cq_map_, cq_map_size_);
        if (sq_map_ && sq_map_ != MAP_FAILED) ::munmap(sq_map_, sq_map_size_);
        if (ring_fd_ >= 0) ::close(ring_fd_);
        sqe_map_ = cq_map_ = sq_map_ = nullptr;
        ring_fd_ = -1;
    }

    BackendCaps caps_;
    unsigned    entries_ = 64;
    int         ring_fd_ = -1;
    struct io_uring_params params_ {};

    void*  sq_map_  = nullptr; size_t sq_map_size_  = 0;
    void*  cq_map_  = nullptr; size_t cq_map_size_  = 0;
    void*  sqe_map_ = nullptr; size_t sqe_map_size_ = 0;

    unsigned* sq_head_ = nullptr; unsigned* sq_tail_ = nullptr;
    unsigned* sq_mask_ = nullptr; unsigned* sq_array_ = nullptr;
    unsigned* cq_head_ = nullptr; unsigned* cq_tail_ = nullptr;
    unsigned* cq_mask_ = nullptr;
    struct io_uring_cqe* cqes_ = nullptr;
    struct io_uring_sqe* sqes_ = nullptr;

    uint32_t inflight_ = 0;
    uint32_t pending_bytes_[kPendingMask + 1] = {};
};

}  // namespace

Result<std::unique_ptr<Backend>> make_io_uring_backend(const IoConfig& cfg) {
    auto b = std::make_unique<IoUringBackend>(cfg);
    if (auto r = b->init(); !r) return std::unexpected(r.error());
    return std::unique_ptr<Backend>(std::move(b));
}

Result<std::unique_ptr<Backend>> make_iocp_backend(const IoConfig&) {
    return fail(Err::Unavailable, "IOCP is a Windows backend");
}

Result<std::unique_ptr<Backend>> make_directstorage_backend(const IoConfig&) {
    return fail(Err::Unavailable, "DirectStorage is a Windows backend");
}

Result<std::unique_ptr<Backend>> make_default_backend(const IoConfig& cfg) {
    return make_io_uring_backend(cfg);
}

}  // namespace deepmoe::storage

#endif  // __linux__
