// 4 KiB alignment helpers. Everything that touches NVMe is sector-aligned by
// construction: FILE_FLAG_NO_BUFFERING and O_DIRECT both require the file
// offset, the byte count and the destination pointer to be sector multiples
// (design §5.1, §9.6).
//
// Ownership/threading: pure functions, no state.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace deepmoe {

inline constexpr uint64_t kPageSize   = 4096;  // NVMe/DirectIO granularity used everywhere
inline constexpr uint64_t kSlabAlign  = 4096;

constexpr bool is_pow2(uint64_t v) noexcept { return v && (v & (v - 1)) == 0; }

constexpr uint64_t align_up(uint64_t v, uint64_t a = kPageSize) noexcept {
    return (v + a - 1) & ~(a - 1);
}
constexpr uint64_t align_down(uint64_t v, uint64_t a = kPageSize) noexcept {
    return v & ~(a - 1);
}
constexpr bool is_aligned(uint64_t v, uint64_t a = kPageSize) noexcept {
    return (v & (a - 1)) == 0;
}
inline bool is_aligned(const void* p, uint64_t a = kPageSize) noexcept {
    return is_aligned(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)), a);
}

// Number of `a`-sized pages spanned by [off, off+bytes).
constexpr uint64_t page_span(uint64_t off, uint64_t bytes, uint64_t a = kPageSize) noexcept {
    if (bytes == 0) return 0;
    return (align_up(off + bytes, a) - align_down(off, a)) / a;
}

// Aligned heap buffer, used for I/O staging and for the host-memory ExpertStore
// backend. Freed with the matching sized/aligned delete.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(size_t bytes, size_t alignment = kPageSize) { reset(bytes, alignment); }
    ~AlignedBuffer() { release(); }

    AlignedBuffer(AlignedBuffer&& o) noexcept
        : p_(o.p_), n_(o.n_), a_(o.a_) { o.p_ = nullptr; o.n_ = 0; }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) { release(); p_ = o.p_; n_ = o.n_; a_ = o.a_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // Returns false on allocation failure (no exception escapes).
    bool reset(size_t bytes, size_t alignment = kPageSize) {
        release();
        if (bytes == 0) return true;
        size_t n = static_cast<size_t>(align_up(bytes, alignment));
        void* p = ::operator new(n, std::align_val_t{alignment}, std::nothrow);
        if (!p) return false;
        p_ = static_cast<std::byte*>(p); n_ = n; a_ = alignment;
        return true;
    }
    void release() noexcept {
        // Plain aligned delete: libc++ on the mingw target does not declare the
        // sized+aligned overload.
        if (p_) ::operator delete(p_, std::align_val_t{a_});
        p_ = nullptr; n_ = 0;
    }

    std::byte*       data()       noexcept { return p_; }
    const std::byte* data() const noexcept { return p_; }
    size_t           size() const noexcept { return n_; }
    explicit operator bool() const noexcept { return p_ != nullptr; }

private:
    std::byte* p_ = nullptr;
    size_t     n_ = 0;
    size_t     a_ = kPageSize;
};

}  // namespace deepmoe
