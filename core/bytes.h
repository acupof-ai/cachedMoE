// Byte-span aliases and little-endian scalar readers used by the manifest
// reader, the safetensors shard readers and the dequant reference paths.
//
// Ownership/threading: non-owning views. A ByteSpan never outlives the buffer
// it points into; the callers are single-threaded parsers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#include "core/status.h"

namespace deepmoe {

using ByteSpan  = std::span<const std::byte>;
using MutBytes  = std::span<std::byte>;

inline ByteSpan as_bytes(const void* p, size_t n) {
    return ByteSpan(static_cast<const std::byte*>(p), n);
}
inline MutBytes as_mut_bytes(void* p, size_t n) {
    return MutBytes(static_cast<std::byte*>(p), n);
}

// x86-64 and every target this project builds for is little-endian; these
// readers exist to make unaligned access well-defined, not to swap bytes.
template <class T>
Result<T> read_le(ByteSpan s, size_t off) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (off + sizeof(T) > s.size()) return fail(Err::OutOfRange, "read past end of buffer");
    T v{};
    std::memcpy(&v, s.data() + off, sizeof(T));
    return v;
}

template <class T>
T read_le_unchecked(const void* p) {
    static_assert(std::is_trivially_copyable_v<T>);
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return v;
}

inline Result<ByteSpan> subspan(ByteSpan s, size_t off, size_t n) {
    if (off > s.size() || n > s.size() - off) return fail(Err::OutOfRange, "subspan out of range");
    return s.subspan(off, n);
}

// Formats a byte count the way every deepMoE report does.
inline std::string human_bytes(uint64_t b) {
    static const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(b);
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[64];
    std::snprintf(buf, sizeof buf, u == 0 ? "%.0f %s" : "%.2f %s", v, unit[u]);
    return buf;
}

}  // namespace deepmoe
