#include "store/expert_store.h"

#include <algorithm>
#include <format>

#include "core/align.h"
#include "core/log.h"

namespace deepmoe::store {

std::string ExpertStoreStats::to_string() const {
    return std::format(
        "expert store: {} slots ({} resident, {} filling, {} free, {} pinned)  "
        "lookups {} hits {} ({:.3f})  fills {}/{} ok  runs {}/{}  "
        "evictions {} (+{} guard-blocked)",
        resident + filling + free, resident, filling, free, pinned,
        lookups, hits, hit_rate(), fills_ok, fills_started, runs_done, runs_started,
        evictions, eviction_blocked_by_guard);
}

Result<void> ExpertStore::init(std::unique_ptr<SlabBacking> backing,
                               const CacheConfig& cache,
                               uint32_t layers, uint32_t experts_per_layer) {
    if (layers == 0 || experts_per_layer == 0)
        return fail(Err::InvalidArgument, "ExpertStore needs a non-empty (layer, expert) space");

    SlabConfig sc;
    sc.slots_per_slab = cache.slots_per_slab;
    sc.slot_bytes     = layout::kExpertSlotBytes;
    sc.budget_bytes   = cache.budget_bytes;
    if (auto r = pool_.init(std::move(backing), sc); !r) return r;

    std::lock_guard lk(mutex_);
    cache_             = cache;
    layers_            = layers;
    experts_per_layer_ = experts_per_layer;

    const uint32_t n = pool_.slot_count();
    slots_.assign(n, ExpertSlot{});
    free_list_.clear();
    free_list_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto a = pool_.address(i);
        if (!a) return std::unexpected(a.error());
        const auto [slab, idx] = pool_.decompose(i);
        slots_[i].slot     = i;
        slots_[i].slab     = slab;
        slots_[i].host_ptr = a->host_ptr;
        slots_[i].dev_addr = a->dev_addr;
        slots_[i].state    = SlotState::Free;
        // Freshest slot first, so a cold start fills in slab order and the
        // backfill stream of design §9.6 stays sequential on disk.
        free_list_.push_back(n - 1 - i);
    }
    index_.clear();
    index_.reserve(n * 2);
    table_.assign(static_cast<size_t>(layers_) * experts_per_layer_ * kExpertPartCount,
                  kNoDeviceAddress);
    stats_ = ExpertStoreStats{};
    stats_.free = n;

    log_info("expert store: {} slots x {} ({:.2f} GiB) on '{}', table {} entries ({} per expert)",
             n, pool_.slot_bytes(), pool_.bytes() / 1073741824.0, pool_.backing_name(),
             table_.size(), kExpertPartCount);
    return {};
}

void ExpertStore::publish_locked(uint32_t slot) {
    const ExpertSlot& s = slots_[slot];
    // A host-only backing has no device address; publish the host pointer so
    // the table is exercised identically by tests and the CPU oracle.
    const uint64_t base = s.dev_addr ? s.dev_addr
                                     : reinterpret_cast<uint64_t>(s.host_ptr);
    for (uint8_t p = 0; p < kExpertPartCount; ++p) {
        table_[table_index(s.key, static_cast<ExpertPart>(p))] =
            base ? base + s.part_offset[p] : kNoDeviceAddress;
    }
}

void ExpertStore::unpublish_locked(uint32_t slot) {
    const ExpertSlot& s = slots_[slot];
    for (uint8_t p = 0; p < kExpertPartCount; ++p)
        table_[table_index(s.key, static_cast<ExpertPart>(p))] = kNoDeviceAddress;
}

std::optional<SlotAddress> ExpertStore::lookup(ExpertKey key, TokenIndex token) {
    std::lock_guard lk(mutex_);
    ++stats_.lookups;
    auto it = index_.find(key);
    if (it == index_.end() || slots_[it->second].state != SlotState::Resident) {
        ++stats_.misses;
        return std::nullopt;
    }
    ExpertSlot& s = slots_[it->second];
    s.last_use_token = token;
    ++stats_.hits;
    return SlotAddress{s.host_ptr, s.dev_addr};
}

bool ExpertStore::resident(ExpertKey key) const {
    std::lock_guard lk(mutex_);
    auto it = index_.find(key);
    return it != index_.end() && slots_[it->second].state == SlotState::Resident;
}

Result<ExpertStore::Reservation> ExpertStore::reserve_locked(ExpertKey key, Tier tier) {
    if (!key_in_range(key))
        return fail(Err::OutOfRange, std::format("expert ({}, {}) is outside the table",
                                                 key.layer, key.expert));
    if (auto it = index_.find(key); it != index_.end()) {
        const SlotState st = slots_[it->second].state;
        return fail(Err::AlreadyExists,
                    std::format("expert ({}, {}) is already {}", key.layer, key.expert,
                                slot_state_name(st)));
    }
    if (free_list_.empty())
        return fail(Err::ResourceExhausted, "no free slot; evict before filling");

    const uint32_t slot = free_list_.back();
    free_list_.pop_back();
    ExpertSlot& s = slots_[slot];
    s.key   = key;
    s.state = SlotState::Filling;
    s.tier  = tier;
    s.guard_timeline = 0;
    s.runs_done  = 0;
    s.run_failed = false;
    index_.emplace(key, slot);

    --stats_.free;
    ++stats_.filling;
    ++stats_.fills_started;

    Reservation r;
    r.slot = slot;
    r.addr = SlotAddress{s.host_ptr, s.dev_addr};
    return r;
}

Result<ExpertStore::Reservation> ExpertStore::begin_fill(ExpertKey key, const ExpertEntry& entry,
                                                         Tier tier) {
    if (entry.runs.empty())
        return fail(Err::InvalidArgument,
                    std::format("expert ({}, {}) has no runs in the manifest",
                                key.layer, key.expert));
    if (entry.runs.size() > layout::kMaxExpertRuns)
        return fail(Err::InvalidArgument,
                    std::format("expert ({}, {}) has {} runs, more than kMaxExpertRuns {}",
                                key.layer, key.expert, entry.runs.size(), layout::kMaxExpertRuns));
    if (entry.slot_bytes > pool_.slot_bytes())
        return fail(Err::InvalidArgument,
                    std::format("expert ({}, {}) needs {} B but a slot is {} B",
                                key.layer, key.expert, entry.slot_bytes, pool_.slot_bytes()));

    std::lock_guard lk(mutex_);
    auto res = reserve_locked(key, tier);
    if (!res) return res;

    ExpertSlot& s = slots_[res->slot];
    s.runs_total = static_cast<uint32_t>(entry.runs.size());
    for (uint8_t p = 0; p < kExpertPartCount; ++p) {
        s.part_offset[p] = entry.part_offset[p];
        s.part_bytes[p]  = entry.part_bytes[p];
    }

    res->run_count = s.runs_total;
    for (size_t i = 0; i < entry.runs.size(); ++i) {
        const Run& run = entry.runs[i];
        FillRun& f = res->runs[i];
        f.file     = run.file;
        f.file_off = run.aligned_off;
        f.bytes    = run.aligned_bytes;
        f.dst      = s.host_ptr ? static_cast<std::byte*>(s.host_ptr) + run.slot_offset : nullptr;
        f.dev_dst  = s.dev_addr ? s.dev_addr + run.slot_offset : kNoDeviceAddress;
    }
    stats_.runs_started += s.runs_total;
    return res;
}

Result<ExpertStore::Reservation> ExpertStore::begin_fill(ExpertKey key, Tier tier) {
    std::lock_guard lk(mutex_);
    auto res = reserve_locked(key, tier);
    if (!res) return res;

    ExpertSlot& s = slots_[res->slot];
    s.runs_total = 1;
    // Planar synthetic layout: w1 | w1.scale | w2 | w2.scale | w3 | w3.scale.
    uint64_t off = 0;
    for (uint8_t p = 0; p < kExpertPartCount; ++p) {
        const bool is_scale = (p & 1) != 0;
        s.part_offset[p] = off;
        s.part_bytes[p]  = is_scale ? layout::kExpertScaleBytesPerMat
                                    : layout::kExpertWeightBytesPerMat;
        off += s.part_bytes[p];
    }
    res->run_count   = 1;
    res->runs[0].file     = 0;
    res->runs[0].file_off = 0;
    res->runs[0].bytes    = pool_.slot_bytes();
    res->runs[0].dst      = s.host_ptr;
    res->runs[0].dev_dst  = s.dev_addr;
    ++stats_.runs_started;
    return res;
}

void ExpertStore::release_locked(uint32_t slot) {
    ExpertSlot& s = slots_[slot];
    index_.erase(s.key);
    s.key   = ExpertKey{};
    s.state = SlotState::Free;
    s.heat  = 0.0f;
    s.guard_timeline = 0;
    s.runs_total = s.runs_done = 0;
    s.run_failed = false;
    for (uint8_t p = 0; p < kExpertPartCount; ++p) { s.part_offset[p] = 0; s.part_bytes[p] = 0; }
    free_list_.push_back(slot);
    ++stats_.free;
}

void ExpertStore::settle_locked(uint32_t slot, bool ok, TokenIndex token) {
    ExpertSlot& s = slots_[slot];
    --stats_.filling;
    if (ok) {
        s.state = SlotState::Resident;
        s.last_use_token = token;
        publish_locked(slot);
        ++stats_.resident;
        ++stats_.fills_ok;
        if (s.tier == Tier::Pinned) ++stats_.pinned;
    } else {
        release_locked(slot);
        ++stats_.fills_failed;
    }
}

Result<bool> ExpertStore::finish_run(uint32_t slot, bool ok, TokenIndex token) {
    std::lock_guard lk(mutex_);
    if (slot >= slots_.size()) return fail(Err::OutOfRange, std::format("slot {} out of range", slot));
    ExpertSlot& s = slots_[slot];
    if (s.state != SlotState::Filling)
        return fail(Err::FailedPrecondition,
                    std::format("slot {} is {}, not filling", slot, slot_state_name(s.state)));
    if (s.runs_done >= s.runs_total)
        return fail(Err::FailedPrecondition,
                    std::format("slot {} already has all {} runs reported", slot, s.runs_total));
    ++s.runs_done;
    ++stats_.runs_done;
    if (!ok) s.run_failed = true;
    if (s.runs_done < s.runs_total) return false;
    settle_locked(slot, !s.run_failed, token);
    return true;
}

Result<void> ExpertStore::finish_fill(uint32_t slot, bool ok, TokenIndex token) {
    std::lock_guard lk(mutex_);
    if (slot >= slots_.size()) return fail(Err::OutOfRange, std::format("slot {} out of range", slot));
    ExpertSlot& s = slots_[slot];
    if (s.state != SlotState::Filling)
        return fail(Err::FailedPrecondition,
                    std::format("slot {} is {}, not filling", slot, slot_state_name(s.state)));
    stats_.runs_done += s.runs_total - s.runs_done;
    s.runs_done = s.runs_total;
    settle_locked(slot, ok && !s.run_failed, token);
    return {};
}

Result<void> ExpertStore::evict(uint32_t slot) {
    std::lock_guard lk(mutex_);
    if (slot >= slots_.size()) return fail(Err::OutOfRange, std::format("slot {} out of range", slot));
    ExpertSlot& s = slots_[slot];
    if (s.state != SlotState::Resident)
        return fail(Err::FailedPrecondition,
                    std::format("slot {} is {}, not resident", slot, slot_state_name(s.state)));
    if (s.tier == Tier::Pinned)
        return fail(Err::FailedPrecondition, std::format("slot {} is pinned", slot));
    // design §5.3: a resident slot may not be recycled while a submitted
    // command buffer can still read it.
    if (s.guard_timeline > completed_timeline_) {
        ++stats_.eviction_blocked_by_guard;
        return fail(Err::FailedPrecondition,
                    std::format("slot {} is guarded by timeline {} (completed {})",
                                slot, s.guard_timeline, completed_timeline_));
    }
    unpublish_locked(slot);
    release_locked(slot);
    --stats_.resident;
    ++stats_.evictions;
    return {};
}

Result<void> ExpertStore::evict_key(ExpertKey key) {
    uint32_t slot;
    {
        std::lock_guard lk(mutex_);
        auto it = index_.find(key);
        if (it == index_.end())
            return fail(Err::NotFound, std::format("expert ({}, {}) is not held", key.layer, key.expert));
        slot = it->second;
    }
    return evict(slot);
}

void ExpertStore::set_completed_timeline(TimelineValue v) {
    std::lock_guard lk(mutex_);
    if (v > completed_timeline_) completed_timeline_ = v;
}

TimelineValue ExpertStore::completed_timeline() const {
    std::lock_guard lk(mutex_);
    return completed_timeline_;
}

Result<void> ExpertStore::set_guard(uint32_t slot, TimelineValue v) {
    std::lock_guard lk(mutex_);
    if (slot >= slots_.size()) return fail(Err::OutOfRange, std::format("slot {} out of range", slot));
    ExpertSlot& s = slots_[slot];
    if (v > s.guard_timeline) s.guard_timeline = v;
    return {};
}

void ExpertStore::note_heat(ExpertKey key, float score, float alpha) {
    std::lock_guard lk(mutex_);
    auto it = index_.find(key);
    if (it == index_.end()) return;
    float& h = slots_[it->second].heat;
    h = (1.0f - alpha) * h + alpha * score;
}

std::optional<ExpertSlot> ExpertStore::slot_info(uint32_t slot) const {
    std::lock_guard lk(mutex_);
    if (slot >= slots_.size()) return std::nullopt;
    return slots_[slot];
}

std::optional<ExpertSlot> ExpertStore::slot_for(ExpertKey key) const {
    std::lock_guard lk(mutex_);
    auto it = index_.find(key);
    if (it == index_.end()) return std::nullopt;
    return slots_[it->second];
}

std::vector<ExpertSlot> ExpertStore::evictable() const {
    std::lock_guard lk(mutex_);
    std::vector<ExpertSlot> out;
    out.reserve(slots_.size());
    for (const ExpertSlot& s : slots_)
        if (s.state == SlotState::Resident && s.tier != Tier::Pinned &&
            s.guard_timeline <= completed_timeline_)
            out.push_back(s);
    return out;
}

uint32_t ExpertStore::free_slots() const {
    std::lock_guard lk(mutex_);
    return static_cast<uint32_t>(free_list_.size());
}

Result<uint64_t> ExpertStore::table_entry(ExpertKey key, ExpertPart part) const {
    std::lock_guard lk(mutex_);
    if (!key_in_range(key))
        return fail(Err::OutOfRange, std::format("expert ({}, {}) is outside the table",
                                                 key.layer, key.expert));
    return table_[table_index(key, part)];
}

ExpertStoreStats ExpertStore::stats() const {
    std::lock_guard lk(mutex_);
    ExpertStoreStats s = stats_;
    s.free = static_cast<uint32_t>(free_list_.size());
    return s;
}

void ExpertStore::reset_stats() {
    std::lock_guard lk(mutex_);
    const uint32_t res = stats_.resident, fil = stats_.filling, pin = stats_.pinned;
    stats_ = ExpertStoreStats{};
    stats_.resident = res;
    stats_.filling  = fil;
    stats_.pinned   = pin;
    stats_.free     = static_cast<uint32_t>(free_list_.size());
}

}  // namespace deepmoe::store
