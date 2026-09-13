// store/: the slab pool geometry, the ExpertStore slot state machine
// (design §5.3, §5.4) and the LRU eviction baseline (design §9.3).
#include <algorithm>
#include <memory>
#include <vector>

#include "core/config.h"
#include "store/expert_store.h"
#include "store/planner.h"
#include "store/predictor.h"
#include "store/slab.h"
#include "tests/test_framework.h"

using namespace deepmoe;
using namespace deepmoe::store;

namespace {

// A small pool: 4 slots per slab, 2 slabs = 8 slots, so eviction is reachable
// in a test without allocating 1.88 GB.
CacheConfig small_cache(uint32_t slots_per_slab = 4, uint32_t slabs = 2) {
    CacheConfig c;
    c.slots_per_slab = slots_per_slab;
    c.budget_bytes   = uint64_t(slots_per_slab) * layout::kExpertBytes * slabs;
    c.policy         = CachePolicy::Lru;
    return c;
}

Result<void> fill(ExpertStore& s, ExpertKey k, TokenIndex token, Tier tier = Tier::Cached) {
    auto r = s.begin_fill(k, tier);
    if (!r) return std::unexpected(r.error());
    return s.finish_fill(r->slot, true, token);
}

}  // namespace

DEEPMOE_TEST(slab, pool_geometry_and_addressing) {
    SlabConfig cfg;
    cfg.slots_per_slab = 4;
    cfg.slot_bytes     = layout::kExpertBytes;
    cfg.budget_bytes   = 4ull * layout::kExpertBytes * 3;   // room for 3 slabs
    SlabPool pool;
    REQUIRE_OK(pool.init(std::make_unique<HostSlabBacking>(), cfg));

    CHECK_EQ(pool.slab_count(), 3u);
    CHECK_EQ(pool.slot_count(), 12u);
    CHECK_EQ(pool.slot_bytes(), layout::kExpertBytes);

    auto a0 = pool.address(0);
    REQUIRE_OK(a0);
    auto a1 = pool.address(1);
    REQUIRE_OK(a1);
    // Slots inside one slab are contiguous at the expert stride.
    CHECK_EQ(static_cast<std::byte*>(a1->host_ptr) - static_cast<std::byte*>(a0->host_ptr),
             static_cast<ptrdiff_t>(layout::kExpertBytes));
    CHECK(is_aligned(a0->host_ptr));
    CHECK(is_aligned(a1->host_ptr));

    const auto [slab, idx] = pool.decompose(5);
    CHECK_EQ(slab, 1u);
    CHECK_EQ(idx, 1u);
    CHECK_ERR(pool.address(12), Err::OutOfRange);
}

DEEPMOE_TEST(slab, rejects_a_slab_over_the_two_gib_vulkan_limit) {
    // design §1.1: maxMemoryAllocationSize is 2 GiB on gfx1151, which is the
    // entire reason slabs exist.
    SlabConfig cfg;
    cfg.slots_per_slab = 200;                       // 200 x 18.8 MB = 3.76 GB
    cfg.slot_bytes     = layout::kExpertBytes;
    cfg.budget_bytes   = 8ull << 30;
    SlabPool pool;
    CHECK_ERR(pool.init(std::make_unique<HostSlabBacking>(), cfg), Err::InvalidArgument);

    // A budget smaller than one slab is also an error, not a silent zero pool.
    SlabConfig tiny = cfg;
    tiny.slots_per_slab = 4;
    tiny.budget_bytes   = layout::kExpertBytes;     // less than 4 slots
    SlabPool p2;
    CHECK_ERR(p2.init(std::make_unique<HostSlabBacking>(), tiny), Err::ResourceExhausted);

    // A non-4 KiB slot size would break the unbuffered-I/O contract.
    SlabConfig bad = cfg;
    bad.slots_per_slab = 1;
    bad.slot_bytes     = 1000;
    SlabPool p3;
    CHECK_ERR(p3.init(std::make_unique<HostSlabBacking>(), bad), Err::InvalidArgument);
}

DEEPMOE_TEST(expert_store, slot_state_machine) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    CHECK_EQ(s.slot_count(), 8u);
    CHECK_EQ(s.free_slots(), 8u);

    const ExpertKey k{2, 5};
    CHECK(!s.resident(k));
    CHECK(!s.lookup(k, 1).has_value());

    // Free -> Filling
    auto res = s.begin_fill(k);
    REQUIRE_OK(res);
    CHECK_EQ(s.free_slots(), 7u);
    CHECK(res->addr.host_ptr != nullptr);
    auto info = s.slot_info(res->slot);
    REQUIRE(info.has_value());
    CHECK_EQ(info->state, SlotState::Filling);
    // A Filling expert is not resident, and must not be reserved twice.
    CHECK(!s.resident(k));
    CHECK_ERR(s.begin_fill(k), Err::AlreadyExists);
    // Nor is it visible in the pointer table yet.
    CHECK_EQ(s.table_entry(k).value_or(1), kNoDeviceAddress);

    // Filling -> Resident
    REQUIRE_OK(s.finish_fill(res->slot, true, 10));
    CHECK(s.resident(k));
    auto addr = s.lookup(k, 11);
    REQUIRE(addr.has_value());
    CHECK_EQ(addr->host_ptr, res->addr.host_ptr);
    // Published: with a host backing the table carries the host pointer.
    CHECK_EQ(s.table_entry(k).value_or(0), reinterpret_cast<uint64_t>(res->addr.host_ptr));
    CHECK_ERR(s.finish_fill(res->slot, true, 12), Err::FailedPrecondition);

    // Resident -> Free
    REQUIRE_OK(s.evict(res->slot));
    CHECK(!s.resident(k));
    CHECK_EQ(s.free_slots(), 8u);
    CHECK_EQ(s.table_entry(k).value_or(1), kNoDeviceAddress);
    CHECK_ERR(s.evict(res->slot), Err::FailedPrecondition);
    CHECK_ERR(s.evict_key(k), Err::NotFound);
}

DEEPMOE_TEST(expert_store, failed_fill_releases_the_slot) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    auto res = s.begin_fill(ExpertKey{1, 1});
    REQUIRE_OK(res);
    CHECK_EQ(s.free_slots(), 7u);
    REQUIRE_OK(s.finish_fill(res->slot, false, 0));
    CHECK_EQ(s.free_slots(), 8u);
    CHECK(!s.resident(ExpertKey{1, 1}));
    CHECK_EQ(s.stats().fills_failed, 1u);
    // The key is free again, so a retry works.
    CHECK_OK(s.begin_fill(ExpertKey{1, 1}));
}

DEEPMOE_TEST(expert_store, exhaustion_and_range_checks) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(2, 1), 4, 8));
    CHECK_EQ(s.slot_count(), 2u);
    REQUIRE_OK(fill(s, ExpertKey{0, 0}, 1));
    REQUIRE_OK(fill(s, ExpertKey{0, 1}, 2));
    CHECK_ERR(s.begin_fill(ExpertKey{0, 2}), Err::ResourceExhausted);
    // Out of the (layer, expert) space entirely.
    CHECK_ERR(s.begin_fill(ExpertKey{99, 0}), Err::OutOfRange);
    CHECK_ERR(s.table_entry(ExpertKey{0, 99}), Err::OutOfRange);
}

DEEPMOE_TEST(expert_store, timeline_guards_eviction) {
    // design §5.3: a resident slot may not be recycled while a submitted
    // command buffer can still read it.
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    auto res = s.begin_fill(ExpertKey{3, 3});
    REQUIRE_OK(res);
    REQUIRE_OK(s.finish_fill(res->slot, true, 1));
    REQUIRE_OK(s.set_guard(res->slot, 100));

    CHECK_ERR(s.evict(res->slot), Err::FailedPrecondition);
    CHECK_EQ(s.stats().eviction_blocked_by_guard, 1u);
    // A guarded slot must not be offered to the policy at all.
    CHECK(s.evictable().empty());

    s.set_completed_timeline(99);
    CHECK_ERR(s.evict(res->slot), Err::FailedPrecondition);
    s.set_completed_timeline(100);
    CHECK_EQ(s.evictable().size(), 1u);
    REQUIRE_OK(s.evict(res->slot));
    // The guard only ever moves forward.
    s.set_completed_timeline(50);
    CHECK_EQ(s.completed_timeline(), 100u);
}

DEEPMOE_TEST(expert_store, pinned_slots_never_evict) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    auto res = s.begin_fill(ExpertKey{0, 0}, Tier::Pinned);
    REQUIRE_OK(res);
    REQUIRE_OK(s.finish_fill(res->slot, true, 1));
    CHECK_ERR(s.evict(res->slot), Err::FailedPrecondition);
    CHECK(s.evictable().empty());
    CHECK_EQ(s.stats().pinned, 1u);
}

DEEPMOE_TEST(expert_store, lookup_refreshes_lru_and_counts_stats) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    REQUIRE_OK(fill(s, ExpertKey{0, 0}, 5));
    auto before = s.slot_for(ExpertKey{0, 0});
    REQUIRE(before.has_value());
    CHECK_EQ(before->last_use_token, 5u);

    CHECK(s.lookup(ExpertKey{0, 0}, 42).has_value());
    auto after = s.slot_for(ExpertKey{0, 0});
    REQUIRE(after.has_value());
    CHECK_EQ(after->last_use_token, 42u);

    CHECK(!s.lookup(ExpertKey{0, 1}, 42).has_value());
    const auto st = s.stats();
    CHECK_EQ(st.lookups, 2u);
    CHECK_EQ(st.hits, 1u);
    CHECK_EQ(st.misses, 1u);
    CHECK_CLOSE(st.hit_rate(), 0.5, 1e-9);

    // `resident()` is the silent query and must not move the counters.
    CHECK(s.resident(ExpertKey{0, 0}));
    CHECK_EQ(s.stats().lookups, 2u);
}

DEEPMOE_TEST(expert_store, heat_ewma_tracks_near_misses) {
    // design §9.3: a high-scoring expert that was not selected still warms up.
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    const ExpertKey k{0, 0};
    REQUIRE_OK(fill(s, k, 1));
    CHECK_CLOSE(s.slot_for(k)->heat, 0.0, 1e-9);
    s.note_heat(k, 1.0f, 0.5f);
    CHECK_CLOSE(s.slot_for(k)->heat, 0.5, 1e-6);
    s.note_heat(k, 1.0f, 0.5f);
    CHECK_CLOSE(s.slot_for(k)->heat, 0.75, 1e-6);
    // Heat on an absent key is a no-op, not a crash.
    s.note_heat(ExpertKey{0, 7}, 1.0f);
    // Eviction clears it, so a recycled slot starts cold.
    REQUIRE_OK(s.evict_key(k));
    REQUIRE_OK(fill(s, k, 2));
    CHECK_CLOSE(s.slot_for(k)->heat, 0.0, 1e-9);
}

DEEPMOE_TEST(planner, lru_policy_picks_the_oldest) {
    auto policy = make_lru_policy();
    std::vector<ExpertSlot> c(4);
    for (uint32_t i = 0; i < 4; ++i) {
        c[i].slot = i;
        c[i].state = SlotState::Resident;
    }
    c[0].last_use_token = 30;
    c[1].last_use_token = 10;      // oldest
    c[2].last_use_token = 40;
    c[3].last_use_token = 20;

    auto v = policy->choose_victims(c, 2);
    REQUIRE_EQ(v.size(), 2u);
    CHECK_EQ(v[0], 1u);
    CHECK_EQ(v[1], 3u);

    // Asking for more than exists returns everything, worst first.
    auto all = policy->choose_victims(c, 99);
    REQUIRE_EQ(all.size(), 4u);
    CHECK_EQ(all[0], 1u);
    CHECK_EQ(all[3], 2u);
    // An empty candidate set is legal.
    CHECK(policy->choose_victims({}, 3).empty());
}

DEEPMOE_TEST(planner, lru_ties_break_deterministically) {
    auto policy = make_lru_policy();
    std::vector<ExpertSlot> c(3);
    for (uint32_t i = 0; i < 3; ++i) { c[i].slot = 2 - i; c[i].last_use_token = 7; }
    auto a = policy->choose_victims(c, 2);
    auto b = policy->choose_victims(c, 2);
    REQUIRE_EQ(a.size(), 2u);
    CHECK_EQ(a[0], 0u);            // lowest slot index first
    CHECK_EQ(a[1], 1u);
    CHECK(a == b);
}

DEEPMOE_TEST(planner, policy_factory_falls_back_loudly) {
    CHECK(std::string(make_policy(CachePolicy::Lru)->name()) == "lru");
    // design §16: the unimplemented policies must not silently claim to be
    // themselves. Their names say what actually ran.
    const std::string sa = make_policy(CachePolicy::ScoreAwareLru)->name();
    CHECK(sa.find("fallback:lru") != std::string::npos);
    CHECK(std::string(make_policy(CachePolicy::Lfu)->name()) == "lru");
}

DEEPMOE_TEST(planner, unimplemented_paths_report_themselves) {
    // design §16 forbids writing these before the P1 measurements; the point of
    // the test is that they fail loudly rather than pretending to work.
    Planner p;
    CHECK_ERR(p.prefetch_lookahead(0, 0), Err::Unimplemented);
    CHECK_ERR(p.backfill(0), Err::Unimplemented);
    auto pred = make_null_predictor();
    auto empty = pred->predict({}, 0, 4, 12);
    REQUIRE_OK(empty);
    CHECK(empty->empty());
}
