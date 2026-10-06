// An explicit exact-routing request may follow an asynchronous mask request.
// Its completed GPU fence retires guards, but does not retire disk fills.
#pragma once

#include <algorithm>
#include <chrono>
#include <format>
#include <optional>

#include "store/expert_store.h"

namespace deepmoe::runtime {

struct DecodeBoundaryCapacity {
    uint32_t free = 0, filling = 0, pinned = 0, evictable = 0, guarded = 0;
    TimelineValue completed = 0, max_guard = 0;

    uint32_t available() const { return free + evictable; }
    std::string describe() const {
        return std::format("free {} / evictable {} / filling {} / pinned {} / guarded {} / "
                           "completed {} / max guard {}",
                           free, evictable, filling, pinned, guarded, completed, max_guard);
    }
};

inline DecodeBoundaryCapacity decode_boundary_capacity(const store::ExpertStore& store) {
    DecodeBoundaryCapacity capacity;
    capacity.completed = store.completed_timeline();
    for (uint32_t index = 0; index < store.slot_count(); ++index) {
        auto slot = store.slot_info(index);
        if (!slot) continue;
        if (slot->state == SlotState::Free) {
            ++capacity.free;
        } else if (slot->state == SlotState::Filling) {
            ++capacity.filling;
        } else if (slot->tier == Tier::Pinned) {
            ++capacity.pinned;
        } else {
            capacity.max_guard = std::max(capacity.max_guard, slot->guard_timeline);
            if (slot->guard_timeline <= capacity.completed) ++capacity.evictable;
            else ++capacity.guarded;
        }
    }
    return capacity;
}

struct DecodeBoundaryWait {
    DecodeBoundaryCapacity before, after;
    uint32_t waits = 0;
    double elapsed_ms = 0;
};

inline Result<DecodeBoundaryWait> settle_decode_boundary(
    store::ExpertStore& store, uint32_t required, std::chrono::milliseconds timeout) {
    if (timeout.count() < 0)
        return fail(Err::InvalidArgument, "negative decode boundary timeout");
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + timeout;
    DecodeBoundaryWait result;
    result.before = result.after = decode_boundary_capacity(store);
    while (result.after.available() < required) {
        // A pinned fill cannot yield an evictable slot. Never clear guards or
        // wait for unrelated pinned resources to disguise insufficient capacity.
        std::optional<ExpertKey> filling;
        for (uint32_t index = 0; index < store.slot_count(); ++index) {
            auto slot = store.slot_info(index);
            if (slot && slot->state == SlotState::Filling && slot->tier != Tier::Pinned) {
                filling = slot->key;
                break;
            }
        }
        if (!filling)
            return fail(Err::ResourceExhausted,
                        std::format("off-plain requires {} admission slots; before [{}], now [{}]; "
                                    "no mutable fill can settle", required,
                                    result.before.describe(), result.after.describe()));
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0)
            return fail(Err::ResourceExhausted,
                        std::format("off-plain boundary timed out after {} ms; required {}; "
                                    "before [{}], now [{}]", timeout.count(), required,
                                    result.before.describe(), result.after.describe()));
        // A failed fill yields a Free slot too. The return value reports
        // residency, so remeasure capacity instead of treating false as failure.
        store.wait_settled(*filling, remaining);
        ++result.waits;
        result.after = decode_boundary_capacity(store);
    }
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

} // namespace deepmoe::runtime
