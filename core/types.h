// Core value types shared by every module. Mirrors design §5.4.
//
// Ownership/threading: all types here are plain values with no ownership.
// ExpertKey/SlotState/Tier are read from the planner thread, the IOCP
// completion threads and the GPU submit thread, so they are kept trivially
// copyable and small enough to sit in an atomic where needed.
#pragma once

#include <compare>
#include <cstdint>
#include <string_view>

namespace deepmoe {

// --- weight element formats (design §6) -------------------------------------
enum class QuantType : uint8_t {
    Fp4E2M1,   // routed/mtp experts; packed 2-per-byte, UE8M0 scale per 32 along K
    Fp8E4M3,   // attention / shared expert / engram / main_proj; UE8M0 per 32x32 block
    E8M0,      // the scale exponent itself: value = 2^(e-127)
    Bf16,      // embed / head / router W
    Fp32,      // mHC, gate bias, sinks, scale vectors
    Fp16,      // activations in flight
    Unknown,
};

constexpr std::string_view quant_name(QuantType q) noexcept {
    switch (q) {
        case QuantType::Fp4E2M1: return "fp4_e2m1";
        case QuantType::Fp8E4M3: return "fp8_e4m3";
        case QuantType::E8M0:    return "e8m0";
        case QuantType::Bf16:    return "bf16";
        case QuantType::Fp32:    return "f32";
        case QuantType::Fp16:    return "f16";
        case QuantType::Unknown: return "unknown";
    }
    return "unknown";
}

// Bits per element, numerator over 8 for sub-byte types.
constexpr uint32_t quant_bits(QuantType q) noexcept {
    switch (q) {
        case QuantType::Fp4E2M1: return 4;
        case QuantType::Fp8E4M3:
        case QuantType::E8M0:    return 8;
        case QuantType::Bf16:
        case QuantType::Fp16:    return 16;
        case QuantType::Fp32:    return 32;
        case QuantType::Unknown: return 0;
    }
    return 0;
}

// --- residency (design §9.3) ------------------------------------------------
enum class Tier : uint8_t {
    Pinned,  // attention, shared experts, mtp, embed and head (~17.7 GB), never evicted
    Cached,  // routed experts living in the slab pool, evicted by policy
    Cold,    // only on NVMe
};

// Slab slot lifecycle (design §5.3).
//   Free -> Filling (I/O in flight) -> Resident -> (Evictable) -> Free
enum class SlotState : uint8_t { Free = 0, Filling = 1, Resident = 2 };

constexpr std::string_view slot_state_name(SlotState s) noexcept {
    switch (s) {
        case SlotState::Free:     return "free";
        case SlotState::Filling:  return "filling";
        case SlotState::Resident: return "resident";
    }
    return "?";
}

// --- identity ---------------------------------------------------------------
// layer < 40 = main model; layer 40..42 = the three DSpark (mtp) blocks.
struct ExpertKey {
    uint16_t layer  = 0;
    uint16_t expert = 0;

    friend constexpr auto operator<=>(const ExpertKey&, const ExpertKey&) = default;

    constexpr uint32_t packed() const noexcept {
        return (static_cast<uint32_t>(layer) << 16) | expert;
    }
    static constexpr ExpertKey unpack(uint32_t v) noexcept {
        return ExpertKey{static_cast<uint16_t>(v >> 16), static_cast<uint16_t>(v & 0xFFFF)};
    }
};

struct ExpertKeyHash {
    size_t operator()(ExpertKey k) const noexcept {
        // splitmix-ish finaliser: the packed key is dense, buckets must not alias by layer.
        uint64_t x = k.packed() * 0x9E3779B97F4A7C15ull;
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        return static_cast<size_t>(x);
    }
};

// GPU virtual address from VK_KHR_buffer_device_address; 0 means "not resident"
// in the expert pointer table (design §5.3).
using DeviceAddress = uint64_t;
inline constexpr DeviceAddress kNoDeviceAddress = 0;

// Monotonic counters. TokenIndex counts accepted tokens; TimelineValue is the
// Vulkan timeline semaphore counter used as the eviction guard.
using TokenIndex    = uint64_t;
using TimelineValue = uint64_t;

// --- I/O (design §5.4, §9.6) ------------------------------------------------
// Priority is a small ordinal, 0 is most urgent. The IoEngine keeps one FIFO
// per class and drains them in order, preempting lower classes when a P0
// arrives.
enum class IoPriority : uint8_t {
    BlockingMiss = 0,  // current layer miss, the GPU is stalled on it
    Lookahead    = 1,  // predicted expert for layer L+d
    Engram       = 2,  // 264 B engram rows for a known token
    Backfill     = 3,  // idle-time refill of free slots by static heat
};
inline constexpr uint8_t kIoPriorityCount = 4;

constexpr std::string_view io_priority_name(IoPriority p) noexcept {
    switch (p) {
        case IoPriority::BlockingMiss: return "p0_blocking";
        case IoPriority::Lookahead:    return "p1_lookahead";
        case IoPriority::Engram:       return "p2_engram";
        case IoPriority::Backfill:     return "p3_backfill";
    }
    return "?";
}

}  // namespace deepmoe
