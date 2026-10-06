#include "core/env.h"
#include "store/slab.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <thread>
#include <vector>

#include "core/log.h"

namespace deepmoe::store {

Result<SlabMemory> HostSlabBacking::allocate(uint64_t bytes) {
    auto buf = std::make_unique<AlignedBuffer>();
    if (!buf->reset(static_cast<size_t>(bytes), kPageSize))
        return fail(Err::ResourceExhausted, std::format("host slab of {} B", bytes));
    SlabMemory m;
    m.host_ptr = buf->data();
    m.bytes    = buf->size();
    // No device address: a host backing is invisible to Vulkan until path B
    // imports it (gpu/vulkan/memory.h).
    owned_.push_back(std::move(buf));
    return m;
}

void HostSlabBacking::release(const SlabMemory& mem) {
    for (auto it = owned_.begin(); it != owned_.end(); ++it) {
        if ((*it)->data() == mem.host_ptr) { owned_.erase(it); return; }
    }
}

Result<void> SlabPool::init(std::unique_ptr<SlabBacking> backing, const SlabConfig& cfg) {
    if (!backing) return fail(Err::InvalidArgument, "SlabPool needs a backing");
    if (cfg.slot_bytes == 0 || cfg.slots_per_slab == 0)
        return fail(Err::InvalidArgument, "slot_bytes and slots_per_slab must be non-zero");
    if (!is_aligned(cfg.slot_bytes))
        return fail(Err::InvalidArgument,
                    std::format("slot_bytes {} is not 4 KiB aligned", cfg.slot_bytes));

    reset();
    backing_    = std::move(backing);
    cfg_        = cfg;
    slab_bytes_ = cfg.slot_bytes * cfg.slots_per_slab;

    // The 2 GiB Vulkan allocation cap of design §1.1 is the reason slabs exist;
    // refuse a configuration that would violate it rather than fail at runtime.
    if (slab_bytes_ > (2ull << 30))
        return fail(Err::InvalidArgument,
                    std::format("slab of {} B exceeds the 2 GiB allocation limit", slab_bytes_));

    const uint32_t want = static_cast<uint32_t>(cfg.budget_bytes / slab_bytes_);
    if (want == 0)
        return fail(Err::ResourceExhausted,
                    std::format("budget {} B is smaller than one {} B slab", cfg.budget_bytes, slab_bytes_));

    slabs_.reserve(want);
    for (uint32_t i = 0; i < want; ++i) {
        auto m = backing_->allocate(slab_bytes_);
        if (!m) {
            if (slabs_.empty()) { reset(); return std::unexpected(m.error()); }
            log_warn("slab pool stopped at {} slabs: {}", slabs_.size(), m.error().str());
            break;
        }
        slabs_.push_back(*m);
    }
    slot_count_ = static_cast<uint32_t>(slabs_.size()) * cfg.slots_per_slab;
    prefault();
    log_debug("slab pool: {} x {} slots on '{}' ({:.2f} GiB)",
              slabs_.size(), cfg.slots_per_slab, backing_->name(), bytes() / 1073741824.0);
    return {};
}

// Linux (RADV): a DEVICE_LOCAL|HOST_VISIBLE allocation is populated lazily --
// TTM allocates and clears the pages on the first CPU touch. The runtime's
// first touch of a slot is the io_uring bounce copy of its first fill, so the
// first ~5,400 fills of every process paid it on the decode's critical path:
// 10.5 ms per miss against 5.4 ms once every slot had been filled once (the
// 8-turn chat, ~27 s per run; STATUS §7 0h). Touch one byte per page here, from
// several threads, so the kernel does that work at startup and in parallel.
// CACHEDMOE_PREFAULT=0 turns it off.
void SlabPool::prefault() {
#if defined(__linux__)
    if (const char* e = ::deepmoe::environment::get("CACHEDMOE_PREFAULT"); e && *e == '0') return;
    std::vector<std::pair<std::byte*, uint64_t>> spans;
    for (const SlabMemory& m : slabs_)
        if (m.host_ptr) spans.emplace_back(static_cast<std::byte*>(m.host_ptr), m.bytes);
    if (spans.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned n  = std::min<unsigned>(std::min(hw, 16u), static_cast<unsigned>(spans.size()));
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < n; ++t)
        pool.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1)) < spans.size();) {
                volatile std::byte* p = spans[i].first;
                for (uint64_t off = 0; off < spans[i].second; off += kPageSize) p[off] = std::byte{0};
            }
        });
    for (auto& th : pool) th.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log_info("slab pool: prefaulted {} slabs ({:.1f} GiB) in {:.1f} s on {} threads",
             spans.size(), bytes() / 1073741824.0, s, n);
#endif
}

void SlabPool::reset() {
    if (backing_) for (const SlabMemory& m : slabs_) backing_->release(m);
    slabs_.clear();
    backing_.reset();
    slot_count_ = 0;
    slab_bytes_ = 0;
}

Result<SlotAddress> SlabPool::address(uint32_t slot) const {
    if (slot >= slot_count_)
        return fail(Err::OutOfRange, std::format("slot {} >= {}", slot, slot_count_));
    const auto [slab, idx] = decompose(slot);
    const SlabMemory& m = slabs_[slab];
    SlotAddress a;
    const uint64_t off = uint64_t(idx) * cfg_.slot_bytes;
    a.host_ptr = m.host_ptr ? static_cast<std::byte*>(m.host_ptr) + off : nullptr;
    a.dev_addr = m.device_address ? m.device_address + off : kNoDeviceAddress;
    return a;
}

}  // namespace deepmoe::store
