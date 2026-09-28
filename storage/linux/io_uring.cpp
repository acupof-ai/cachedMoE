// io_uring backend, written against the raw kernel interface (no liburing --
// design's "no third-party deps" rule). It began as the CI backend (design §14);
// since 2026-09-28 the dev machine itself runs Linux, so this is the production
// read path there.
//
// Ownership/threading: one ring. `submit` may be called from several threads at
// once -- IoEngine's submit pool (Track Q2, kDefaultSubmitThreads = 8) does
// exactly that -- so the SQ producer side is serialised by `sq_mutex_`; the SQ
// ring is single-producer and two unserialised writers overwrite each other's
// SQE, which loses a chunk and leaves the dispatcher waiting for it forever.
// `poll` runs only on the dispatcher thread. The kernel owns the SQ/CQ shared
// mappings; this class owns the ring fd and the three mmaps.
//
// Device mappings: O_DIRECT pins the destination pages with get_user_pages,
// which a DRM buffer mapping (VM_PFNMAP -- what vkMapMemory of a DEVICE_LOCAL
// |HOST_VISIBLE type returns on RADV, i.e. path A) refuses: the read completes
// with -EFAULT. Windows allows it (that is Track Q2's 704 us probe). So a chunk
// whose destination lies in a /dev/dri mapping is read into a host bounce
// buffer and copied on completion. The check is made at submit time from a
// cached copy of /proc/self/maps; an -EFAULT that still gets through marks the
// mapping and resubmits the chunk through a bounce buffer.
//
// The copies out of the bounce buffers run on the polling thread; their total
// is logged at teardown. A pool of helper threads for them was tried and
// measured nothing (docs/STATUS.md 0j): once the slabs are pre-faulted a copy
// is ~22 GB/s and overlaps the reads still in flight.
#if defined(__linux__)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "core/align.h"
#include "core/log.h"
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
int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags,
                       const void* arg = nullptr, size_t argsz = 0) {
    return static_cast<int>(::syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags,
                                      arg, argsz));
}

// Which address ranges are DRM buffer mappings (see the header comment). The
// cache is refreshed from /proc/self/maps only when an address falls outside
// every range it knows, which after start-up is essentially never.
class MappingClassifier {
public:
    bool is_device(const void* p) {
        const auto a = reinterpret_cast<uintptr_t>(p);
        std::lock_guard lk(mu_);
        if (const Range* r = find(a)) return r->device;
        reload();
        if (const Range* r = find(a)) return r->device;
        return false;
    }
    void mark_device(const void* p) {
        const auto a = reinterpret_cast<uintptr_t>(p);
        std::lock_guard lk(mu_);
        if (Range* r = find(a)) { r->device = true; return; }
        ranges_.push_back({a & ~uintptr_t(4095), (a & ~uintptr_t(4095)) + 4096, true});
    }

private:
    struct Range { uintptr_t lo, hi; bool device; };
    Range* find(uintptr_t a) {
        for (Range& r : ranges_) if (a >= r.lo && a < r.hi) return &r;
        return nullptr;
    }
    void reload() {
        std::vector<Range> forced;
        for (const Range& r : ranges_) if (r.device) forced.push_back(r);
        ranges_.clear();
        std::FILE* f = std::fopen("/proc/self/maps", "r");
        if (f) {
            char line[1024];
            while (std::fgets(line, sizeof line, f)) {
                unsigned long lo = 0, hi = 0;
                int path_at = 0;
                if (std::sscanf(line, "%lx-%lx %*s %*s %*s %*s %n", &lo, &hi, &path_at) < 2) continue;
                const bool dev = path_at > 0 && std::strncmp(line + path_at, "/dev/dri/", 9) == 0;
                ranges_.push_back({lo, hi, dev});
            }
            std::fclose(f);
        }
        for (const Range& r : forced) {
            bool covered = false;
            for (Range& x : ranges_)
                if (r.lo >= x.lo && r.lo < x.hi) { x.device = true; covered = true; }
            if (!covered) ranges_.push_back(r);
        }
    }
    std::mutex mu_;
    std::vector<Range> ranges_;
};

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
        ext_arg_ = (p.features & IORING_FEAT_EXT_ARG) != 0;

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
        Slot slot;
        slot.fd    = req.file->native();
        slot.off   = req.file_off;
        slot.dst   = req.dst;
        slot.bytes = req.bytes;
        slot.want  = req.min_bytes ? req.min_bytes : req.bytes;
        if (maps_.is_device(req.dst)) {
            slot.bounce = take_bounce(req.bytes);
            if (!slot.bounce) return fail(Err::ResourceExhausted, "io_uring bounce buffer");
        }
        std::lock_guard lk(sq_mutex_);
        if (inflight_.load(std::memory_order_relaxed) >= caps_.max_queue_depth) {
            give_bounce(slot.bounce);
            return fail(Err::ResourceExhausted, "SQ is full");
        }
        auto r = push_locked(req.chunk_id, slot);
        if (!r) { give_bounce(slot.bounce); return r; }
        inflight_.fetch_add(1, std::memory_order_relaxed);
        return {};
    }

    Result<size_t> poll(std::span<ChunkCompletion> out, std::chrono::milliseconds timeout) override {
        if (out.empty()) return size_t{0};
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        size_t n = 0;
        for (;;) {
            n += drain(out.subspan(n));
            if (n > 0 || timeout.count() == 0) break;
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            if (ext_arg_) {
                // Sleep in the kernel until a CQE lands or the deadline passes,
                // instead of the old 200 us sleep loop (it added up to 200 us to
                // every miss the decode waited on).
                const auto left = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
                struct __kernel_timespec ts {};
                ts.tv_sec  = left.count() / 1'000'000'000;
                ts.tv_nsec = left.count() % 1'000'000'000;
                struct io_uring_getevents_arg arg {};
                arg.ts = reinterpret_cast<uint64_t>(&ts);
                (void)sys_io_uring_enter(ring_fd_, 0, 1, IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG,
                                         &arg, sizeof arg);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
        return n;
    }

    uint32_t inflight() const override { return inflight_.load(std::memory_order_relaxed); }

    void cancel_all() override { /* the ring is torn down at shutdown; nothing partial survives */ }

private:
    static constexpr uint64_t kPendingMask = 1023;

    struct Slot {
        int      fd     = -1;
        uint64_t off    = 0;
        void*    dst    = nullptr;
        uint32_t bytes  = 0;
        uint32_t want   = 0;       // the in-file part that must arrive (backend.h)
        void*    bounce = nullptr; // host buffer the kernel reads into, or null
    };

    // Caller holds sq_mutex_. The slot is recorded before the tail is
    // published, so the completion can never be seen before its record.
    Result<void> push_locked(uint64_t chunk_id, const Slot& slot) {
        const unsigned tail = *sq_tail_;
        const unsigned head = load_acquire(sq_head_);
        if (tail - head >= params_.sq_entries) return fail(Err::ResourceExhausted, "SQ is full");

        slots_[chunk_id & kPendingMask] = slot;
        const unsigned index = tail & *sq_mask_;
        struct io_uring_sqe* sqe = &sqes_[index];
        std::memset(sqe, 0, sizeof *sqe);
        sqe->opcode    = IORING_OP_READ;
        sqe->fd        = slot.fd;
        sqe->off       = slot.off;
        sqe->addr      = reinterpret_cast<uint64_t>(slot.bounce ? slot.bounce : slot.dst);
        sqe->len       = slot.bytes;
        sqe->user_data = chunk_id;
        sq_array_[index] = index;
        store_release(sq_tail_, tail + 1);

        const int r = sys_io_uring_enter(ring_fd_, 1, 0, 0);
        if (r < 0) return fail(Err::Io, "io_uring_enter(submit)", static_cast<uint32_t>(errno));
        return {};
    }

    void* take_bounce(uint32_t bytes) {
        const size_t want = (static_cast<size_t>(bytes) + (1u << 20) - 1) & ~size_t((1u << 20) - 1);
        {
            std::lock_guard lk(bounce_mutex_);
            for (size_t i = 0; i < bounce_free_.size(); ++i)
                if (bounce_free_[i].bytes >= want) {
                    void* p = bounce_free_[i].ptr;
                    bounce_size_.push_back({p, bounce_free_[i].bytes});
                    bounce_free_.erase(bounce_free_.begin() + static_cast<long>(i));
                    return p;
                }
        }
        void* p = std::aligned_alloc(kPageSize, want);
        if (!p) return nullptr;
        std::lock_guard lk(bounce_mutex_);
        bounce_size_.push_back({p, want});
        return p;
    }
    void give_bounce(void* p) {
        if (!p) return;
        std::lock_guard lk(bounce_mutex_);
        for (size_t i = 0; i < bounce_size_.size(); ++i)
            if (bounce_size_[i].ptr == p) {
                bounce_free_.push_back(bounce_size_[i]);
                bounce_size_.erase(bounce_size_.begin() + static_cast<long>(i));
                return;
            }
    }

    size_t drain(std::span<ChunkCompletion> out) {
        size_t n = 0;
        unsigned head = *cq_head_;
        const unsigned tail = load_acquire(cq_tail_);
        while (head != tail && n < out.size()) {
            const struct io_uring_cqe cqe = cqes_[head & *cq_mask_];
            ++head;
            Slot& slot = slots_[cqe.user_data & kPendingMask];

            if (cqe.res == -EFAULT && !slot.bounce) {
                // A device mapping the classifier did not know about: remember
                // it and read this chunk again through a bounce buffer.
                maps_.mark_device(slot.dst);
                Slot again = slot;
                again.bounce = take_bounce(again.bytes);
                if (again.bounce) {
                    std::lock_guard lk(sq_mutex_);
                    if (push_locked(cqe.user_data, again)) continue;   // still in flight
                }
                give_bounce(again.bounce);
            }

            ChunkCompletion c;
            c.chunk_id = cqe.user_data;
            if (cqe.res < 0) {
                c.status = Status{Err::Io, "io_uring read failed", static_cast<uint32_t>(-cqe.res)};
            } else {
                c.bytes_moved = static_cast<uint32_t>(cqe.res);
                if (slot.bounce) {
                    const auto t0 = std::chrono::steady_clock::now();
                    std::memcpy(slot.dst, slot.bounce, c.bytes_moved);
                    copy_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                          std::chrono::steady_clock::now() - t0).count());
                    copy_bytes_ += c.bytes_moved;
                    ++copies_;
                }
                if (slot.want && c.bytes_moved < slot.want)
                    c.status = Status{Err::Io, std::format("short read: {} of {}", c.bytes_moved, slot.want)};
            }
            give_bounce(slot.bounce);
            slot.bounce = nullptr;
            out[n++] = c;
            inflight_.fetch_sub(1, std::memory_order_relaxed);
        }
        store_release(cq_head_, head);
        return n;
    }

    void teardown() {
        if (copies_)
            log_info("io_uring: bounce copies {:.1f} MiB in {:.3f} s on the polling thread "
                     "({:.1f} GB/s, {} chunks)", copy_bytes_ / 1048576.0, copy_ns_ / 1e9,
                     copy_ns_ ? copy_bytes_ / double(copy_ns_) : 0.0, copies_);
        if (sqe_map_ && sqe_map_ != MAP_FAILED) ::munmap(sqe_map_, sqe_map_size_);
        if (cq_map_size_ && cq_map_ && cq_map_ != MAP_FAILED) ::munmap(cq_map_, cq_map_size_);
        if (sq_map_ && sq_map_ != MAP_FAILED) ::munmap(sq_map_, sq_map_size_);
        if (ring_fd_ >= 0) ::close(ring_fd_);
        sqe_map_ = cq_map_ = sq_map_ = nullptr;
        ring_fd_ = -1;
        for (auto& b : bounce_free_) std::free(b.ptr);
        for (auto& b : bounce_size_) std::free(b.ptr);
        bounce_free_.clear();
        bounce_size_.clear();
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

    bool ext_arg_ = false;                  // IORING_FEAT_EXT_ARG: poll can wait in the kernel
    std::atomic<uint32_t> inflight_{0};
    std::mutex sq_mutex_;                   // serialises the SQ producer side
    Slot slots_[kPendingMask + 1] = {};

    struct Bounce { void* ptr; size_t bytes; };
    std::mutex bounce_mutex_;
    std::vector<Bounce> bounce_free_;       // idle bounce buffers
    std::vector<Bounce> bounce_size_;       // bounce buffers in flight
    MappingClassifier maps_;

    uint64_t copy_ns_ = 0, copy_bytes_ = 0, copies_ = 0;   // poll thread only
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
