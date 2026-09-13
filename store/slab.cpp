#include "store/slab.h"

#include <format>

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
    log_debug("slab pool: {} x {} slots on '{}' ({:.2f} GiB)",
              slabs_.size(), cfg.slots_per_slab, backing_->name(), bytes() / 1073741824.0);
    return {};
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
