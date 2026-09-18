// store/: the slab pool geometry, the ExpertStore slot state machine
// (design §5.3, §5.4) and the LRU eviction baseline (design §9.3).
#include <algorithm>
#include <memory>
#include <vector>

#include "core/align.h"
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
// in a test without allocating 1.88 GB. A slot is kExpertSlotBytes -- the two
// sector-aligned runs of an expert, not the 18,800,640 B payload (design §5.1).
CacheConfig small_cache(uint32_t slots_per_slab = 4, uint32_t slabs = 2) {
    CacheConfig c;
    c.slots_per_slab = slots_per_slab;
    c.budget_bytes   = uint64_t(slots_per_slab) * layout::kExpertSlotBytes * slabs;
    c.policy         = CachePolicy::Lru;
    return c;
}

// The manifest entry of a two-run expert, shaped exactly like the real ones:
// a 1,110,016 B scales run followed by a 17,698,816 B weights run, each with
// three skewed parts inside it.
ExpertEntry two_run_entry(uint32_t file = 0, uint64_t scale_off = 8'269'824,
                          uint64_t weight_off = 594'984'960) {
    ExpertEntry e;
    Run scales;
    scales.file = file;
    scales.aligned_off = scale_off;
    scales.aligned_bytes = 1'110'016;
    scales.slot_offset = 0;
    const uint32_t scale_skew = 3896;
    for (uint8_t i = 0; i < 3; ++i) {
        RunPart p;
        p.part = static_cast<ExpertPart>(2 * i + 1);          // w1/w2/w3 .scale
        p.skew = scale_skew + i * static_cast<uint32_t>(layout::kExpertScaleBytesPerMat);
        p.bytes = layout::kExpertScaleBytesPerMat;
        p.slot_offset = scales.slot_offset + p.skew;
        scales.parts.push_back(p);
    }
    Run weights;
    weights.file = file;
    weights.aligned_off = weight_off;
    weights.aligned_bytes = 17'698'816;
    weights.slot_offset = scales.aligned_bytes;
    const uint32_t weight_skew = 1592;
    for (uint8_t i = 0; i < 3; ++i) {
        RunPart p;
        p.part = static_cast<ExpertPart>(2 * i);              // w1/w2/w3 .weight
        p.skew = weight_skew + i * static_cast<uint32_t>(layout::kExpertWeightBytesPerMat);
        p.bytes = layout::kExpertWeightBytesPerMat;
        p.slot_offset = weights.slot_offset + p.skew;
        weights.parts.push_back(p);
    }
    e.runs = {scales, weights};
    e.slot_bytes = scales.aligned_bytes + weights.aligned_bytes;
    for (const Run& r : e.runs)
        for (const RunPart& p : r.parts) {
            e.part_offset[static_cast<uint8_t>(p.part)] = p.slot_offset;
            e.part_bytes[static_cast<uint8_t>(p.part)]  = p.bytes;
        }
    return e;
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
    cfg.slot_bytes     = layout::kExpertSlotBytes;
    cfg.budget_bytes   = 4ull * layout::kExpertSlotBytes * 3;   // room for 3 slabs
    SlabPool pool;
    REQUIRE_OK(pool.init(std::make_unique<HostSlabBacking>(), cfg));

    CHECK_EQ(pool.slab_count(), 3u);
    CHECK_EQ(pool.slot_count(), 12u);
    CHECK_EQ(pool.slot_bytes(), layout::kExpertSlotBytes);

    auto a0 = pool.address(0);
    REQUIRE_OK(a0);
    auto a1 = pool.address(1);
    REQUIRE_OK(a1);
    // Slots inside one slab are contiguous at the expert stride.
    CHECK_EQ(static_cast<std::byte*>(a1->host_ptr) - static_cast<std::byte*>(a0->host_ptr),
             static_cast<ptrdiff_t>(layout::kExpertSlotBytes));
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
    cfg.slot_bytes     = layout::kExpertSlotBytes;
    cfg.budget_bytes   = 8ull << 30;
    SlabPool pool;
    CHECK_ERR(pool.init(std::make_unique<HostSlabBacking>(), cfg), Err::InvalidArgument);

    // A budget smaller than one slab is also an error, not a silent zero pool.
    SlabConfig tiny = cfg;
    tiny.slots_per_slab = 4;
    tiny.budget_bytes   = layout::kExpertSlotBytes;  // less than 4 slots
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

DEEPMOE_TEST(expert_store, two_run_fill_publishes_every_part) {
    // design §5.1 (v0.5): a slot is filled by one IoRequest per run, and the
    // pointer table hands out slot_base + run.slot_offset + part.skew per part.
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    const ExpertEntry entry = two_run_entry(3);
    const ExpertKey k{1, 2};

    auto res = s.begin_fill(k, entry);
    REQUIRE_OK(res);
    REQUIRE_EQ(res->run_count, 2u);
    CHECK_EQ(s.free_slots(), 7u);

    // Every run is a legal unbuffered read into a sector-aligned destination.
    auto* base = static_cast<std::byte*>(res->addr.host_ptr);
    uint64_t covered = 0;
    for (uint32_t i = 0; i < res->run_count; ++i) {
        const FillRun& r = res->runs[i];
        CHECK_EQ(r.file, 3u);
        CHECK(is_aligned(r.file_off));
        CHECK(is_aligned(r.bytes));
        CHECK(is_aligned(r.dst));
        CHECK_EQ(r.dst, base + entry.runs[i].slot_offset);
        covered += r.bytes;
    }
    CHECK_EQ(covered, layout::kExpertSlotBytes);
    CHECK(covered <= s.slot_bytes());

    // Filling ends only when BOTH runs have reported, and the table stays empty
    // until then -- a half-filled slot must never be addressable.
    auto first = s.finish_run(res->slot, true, 7);
    REQUIRE_OK(first);
    CHECK(!*first);
    CHECK(!s.resident(k));
    CHECK_EQ(s.table_entry(k, ExpertPart::W1Weight).value_or(1), kNoDeviceAddress);

    auto second = s.finish_run(res->slot, true, 7);
    REQUIRE_OK(second);
    CHECK(*second);
    CHECK(s.resident(k));
    CHECK_ERR(s.finish_run(res->slot, true, 7), Err::FailedPrecondition);

    // Six entries per expert, each at its own skew inside the slot.
    const uint64_t b = reinterpret_cast<uint64_t>(base);
    for (uint8_t i = 0; i < kExpertPartCount; ++i) {
        const auto part = static_cast<ExpertPart>(i);
        auto a = s.table_entry(k, part);
        REQUIRE_OK(a);
        CHECK_EQ(*a, b + entry.offset_of(part));
        // Nothing may point outside its own slot.
        CHECK(entry.offset_of(part) + entry.bytes_of(part) <= s.slot_bytes());
    }
    CHECK_EQ(s.pointer_table_stride(), kExpertPartCount);
    CHECK_EQ(s.pointer_table_entries(), size_t{4} * 8 * kExpertPartCount);
    // The default accessor is part 0, which is NOT the slot base here: the
    // weights run sits behind the scales run.
    CHECK_EQ(s.table_entry(k).value_or(0), b + entry.offset_of(ExpertPart::W1Weight));
    CHECK(entry.offset_of(ExpertPart::W1Weight) != 0u);

    // Eviction clears all six.
    REQUIRE_OK(s.evict_key(k));
    for (uint8_t i = 0; i < kExpertPartCount; ++i)
        CHECK_EQ(s.table_entry(k, static_cast<ExpertPart>(i)).value_or(1), kNoDeviceAddress);
}

DEEPMOE_TEST(expert_store, one_bad_run_fails_the_whole_fill) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    const ExpertEntry entry = two_run_entry();
    const ExpertKey k{0, 4};

    auto res = s.begin_fill(k, entry);
    REQUIRE_OK(res);
    // The failure arrives first; the slot still waits for the other run rather
    // than being recycled under an in-flight read.
    auto a = s.finish_run(res->slot, false, 1);
    REQUIRE_OK(a);
    CHECK(!*a);
    CHECK_EQ(s.free_slots(), 7u);
    auto b = s.finish_run(res->slot, true, 1);
    REQUIRE_OK(b);
    CHECK(*b);
    CHECK(!s.resident(k));
    CHECK_EQ(s.free_slots(), 8u);
    CHECK_EQ(s.stats().fills_failed, 1u);
    CHECK_EQ(s.stats().runs_started, 2u);
    CHECK_EQ(s.stats().runs_done, 2u);
    CHECK_EQ(s.table_entry(k).value_or(1), kNoDeviceAddress);
}

DEEPMOE_TEST(expert_store, rejects_an_entry_that_does_not_fit_a_slot) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(), 4, 8));
    ExpertEntry big = two_run_entry();
    big.slot_bytes += 4096;
    CHECK_ERR(s.begin_fill(ExpertKey{0, 0}, big), Err::InvalidArgument);

    ExpertEntry empty;
    CHECK_ERR(s.begin_fill(ExpertKey{0, 0}, empty), Err::InvalidArgument);

    ExpertEntry many = two_run_entry();
    many.runs.resize(layout::kMaxExpertRuns + 1, many.runs[0]);
    CHECK_ERR(s.begin_fill(ExpertKey{0, 0}, many), Err::InvalidArgument);
    // None of the rejections may consume a slot.
    CHECK_EQ(s.free_slots(), 8u);
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

// --- Track R1 (docs/p4_hitrate.md) --------------------------------------------

namespace {

Result<void> fill_stamped(ExpertStore& s, ExpertKey k, TokenIndex stamp) {
    auto r = s.begin_fill(k);
    if (!r) return std::unexpected(r.error());
    return s.finish_fill(r->slot, true, stamp);
}

}  // namespace

DEEPMOE_TEST(expert_store, touch_oldest_stamp_and_the_guard_in_evict_lru) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(4, 1)));
    for (uint16_t e = 0; e < 4; ++e) REQUIRE_OK(fill_stamped(s, ExpertKey{1, e}, 10 + e));
    CHECK_EQ(*s.oldest_evictable_stamp(), 10u);
    // touch only ever makes a key newer, and does not count a lookup.
    CHECK(s.touch(ExpertKey{1, 0}, 50));
    CHECK(s.touch(ExpertKey{1, 0}, 20));
    CHECK_EQ(s.slot_for(ExpertKey{1, 0})->last_use_token, 50u);
    CHECK(!s.touch(ExpertKey{1, 9}, 60));
    CHECK_EQ(s.stats().lookups, 0u);
    CHECK_EQ(*s.oldest_evictable_stamp(), 11u);
    // The eviction guard: a slot a submitted buffer still reads is skipped, and
    // the next-oldest goes instead; once the timeline passes, it is fair game.
    const uint32_t slot1 = *s.slot_of(ExpertKey{1, 1});
    REQUIRE_OK(s.set_guard(slot1, 7));
    s.set_completed_timeline(6);
    CHECK_EQ(*s.oldest_evictable_stamp(), 12u);
    REQUIRE_OK(s.evict_lru());
    CHECK(!s.resident(ExpertKey{1, 2}));
    CHECK(s.resident(ExpertKey{1, 1}));
    s.set_completed_timeline(7);
    CHECK_EQ(*s.oldest_evictable_stamp(), 11u);
    REQUIRE_OK(s.evict_lru());
    CHECK(!s.resident(ExpertKey{1, 1}));
    // wait_settled on a key nobody is filling returns at once.
    CHECK(s.wait_settled(ExpertKey{1, 3}, std::chrono::milliseconds(1)));
    CHECK(!s.wait_settled(ExpertKey{1, 2}, std::chrono::milliseconds(1)));
}

// docs/p4_hitrate.md §7: per-turn reheat. What the backfill pass is handed is
// the resident experts ordered by their decayed heat, so the test is that
// decay + order really do put the turn that just ran on top, and that the order
// is empty-safe and deterministic.
DEEPMOE_TEST(expert_store, heat_decay_and_order_drive_the_reheat_pass) {
    ExpertStore s;
    REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(4, 2), 2, 4));
    for (uint16_t e = 0; e < 4; ++e) REQUIRE_OK(fill(s, ExpertKey{0, e}, 1 + e));
    for (uint16_t e = 0; e < 4; ++e) REQUIRE_OK(fill(s, ExpertKey{1, e}, 1 + e));
    // Turn A routed (0,0) and (0,1) eight times; then a turn boundary halves
    // everything, and turn B routes (1,2) and (1,3) eight times. Turn A's
    // experts still carry the larger heat (they were routed before the halving,
    // so their value saturated and was then only halved), but the point of the
    // exercise is the ORDER and the SCALE, not which of the two wins:
    //   * exactly the four routed experts are above zero, the other four at 0;
    //   * the order is a pure function of heat (layer, then id, break ties);
    //   * a further boundary rescales the hot end to 1.0 and preserves ratios.
    for (int i = 0; i < 8; ++i) {
        s.note_heat(ExpertKey{0, 0}, 1.0f);
        s.note_heat(ExpertKey{0, 1}, 1.0f);
    }
    s.decay_heat(0.5f);
    for (int i = 0; i < 8; ++i) {
        s.note_heat(ExpertKey{1, 2}, 1.0f);
        s.note_heat(ExpertKey{1, 3}, 1.0f);
    }
    // The heat the two turns leave behind, from the EWMA definition plus the
    // renormalisation: turn A's experts saturated and the boundary rescaled the
    // hot end straight back to 1.0, and turn B's eight notes reach 1 - 0.875^8.
    const float sat = 1.0f - std::pow(0.875f, 8);     // 8 notes at alpha 0.125
    auto want = [&](uint16_t layer, uint16_t expert) -> float {
        if (layer == 0 && (expert == 0 || expert == 1)) return 1.0f;
        if (layer == 1 && (expert == 2 || expert == 3)) return sat;
        return 0.0f;
    };
    auto ord = s.heat_order();
    REQUIRE_EQ(ord.size(), 8u);
    for (const ExpertKey& k : ord) CHECK_CLOSE(s.slot_for(k)->heat, want(k.layer, k.expert), 1e-4);
    // The warm four come first, in heat order; the four never-routed ones last.
    uint32_t warm_n = 0;
    for (uint32_t i = 0; i < ord.size(); ++i)
        if (want(ord[i].layer, ord[i].expert) > 0.0f) ++warm_n;
    CHECK_EQ(warm_n, 4u);
    for (uint32_t i = 1; i < ord.size(); ++i)
        CHECK(s.slot_for(ord[i - 1])->heat >= s.slot_for(ord[i])->heat);
    CHECK_EQ(s.slot_for(ord[7])->heat, 0.0f);

    // A turn boundary rescales the hot end to 1.0 -- the EWMA's own ceiling --
    // so that a floor like "5% of the hottest expert" means the same thing at
    // every turn. The RATIOS are what the ranking is, and they survive.
    const float head = s.slot_for(ord[0])->heat;
    const float warm = s.slot_for(ord[2])->heat;
    CHECK(warm > 0.0f);
    CHECK(warm < head);
    s.decay_heat(0.1f);
    auto ord_b = s.heat_order();
    REQUIRE_EQ(ord_b.size(), 8u);
    CHECK_CLOSE(s.slot_for(ord_b[0])->heat, 1.0f, 1e-5);
    CHECK(ord_b[0] == ord[0]);            // rescaling does not reorder anything
    CHECK(ord_b[2] == ord[2]);
    CHECK_CLOSE(s.slot_for(ord_b[2])->heat, warm / head, 1e-4);
    CHECK_EQ(s.slot_for(ord_b[7])->heat, 0.0f);
    s.decay_heat(1.0f);                       // the identity: no rescale, no ageing
    CHECK_CLOSE(s.slot_for(ord_b[0])->heat, 1.0f, 1e-5);
    CHECK_CLOSE(s.slot_for(ord_b[2])->heat, warm / head, 1e-4);

    // A second boundary with no routing since leaves the ranking alone, and the
    // hot end is back at 1.0 -- the scale is the same at every turn.
    s.decay_heat(0.5f);
    auto ord2 = s.heat_order();
    REQUIRE_EQ(ord2.size(), 8u);
    CHECK(ord2[0] == ord[0]);
    CHECK_CLOSE(s.slot_for(ord2[0])->heat, 1.0f, 1e-5);
    CHECK_CLOSE(s.slot_for(ord2[2])->heat, warm / head, 1e-4);
    s.decay_heat(1.0f);
    CHECK_CLOSE(s.slot_for(ord2[0])->heat, 1.0f, 1e-5);
    auto ord3 = s.heat_order();
    CHECK(ord3[0] == ord[0]);

    // Every key in the order is resident -- that is what makes it safe to hand
    // to a backfill that never evicts -- and a pinned expert is not in it, since
    // a reheat pass may not move the pinned set either.
    for (const ExpertKey& k : ord) CHECK(s.resident(k));
    REQUIRE_OK(s.evict_key(ExpertKey{1, 0}));
    REQUIRE_OK(fill(s, ExpertKey{1, 0}, 1, Tier::Pinned));
    auto ord4 = s.heat_order();
    REQUIRE_EQ(ord4.size(), 7u);
    for (const ExpertKey& k : ord4) CHECK(!(k.layer == 1 && k.expert == 0));
    CHECK_EQ(s.heat_slots(), 7u);
}

// design 9.7.3 / docs/p3_prefill.md 3.4: after a prefill streams its experts
// layer by layer, in batches whose reads are issued before the previous batch
// computes, the cache holds exactly what a global LRU over the prompt's routing
// table -- walked token by token, layer by layer, in gate order -- would hold,
// including what was resident before.
DEEPMOE_TEST(planner, streamed_admission_is_the_lru_of_the_routing_table) {
    constexpr uint32_t kLayers = 4, kTopk = 3, kExperts = 12;
    for (uint32_t trial = 0; trial < 6; ++trial) {
        const uint32_t cap = 3 + trial;              // 3..8 slots
        const uint32_t T   = 6 + 3 * trial;          // prompt rows
        ExpertStore s;
        REQUIRE_OK(s.init(std::make_unique<HostSlabBacking>(), small_cache(1, cap)));
        uint64_t seed = 0x9E3779B97F4A7C15ull * (trial + 1);
        auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };

        // What was resident before: stamps below every prompt stamp.
        std::vector<uint32_t> ref;   // global keys, oldest first
        const uint32_t pre = std::min<uint32_t>(cap, 2 + trial);
        for (uint32_t i = 0; i < pre; ++i) {
            const uint32_t k = static_cast<uint32_t>(rnd() % (kLayers * kExperts));
            if (std::find(ref.begin(), ref.end(), k) != ref.end()) continue;
            REQUIRE_OK(fill_stamped(s, ExpertKey{static_cast<uint16_t>(k / kExperts),
                                                 static_cast<uint16_t>(k % kExperts)}, 100 + i));
            ref.push_back(k);
        }
        // The routing table [T][layers][topk], distinct ids per row.
        const uint32_t pool = (trial % 2) ? 5 : kExperts;
        std::vector<uint32_t> ids(size_t(T) * kLayers * kTopk);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t L = 0; L < kLayers; ++L)
                for (uint32_t j = 0; j < kTopk; ++j) {
                    uint32_t e = 0;
                    bool dup = true;
                    while (dup) {
                        e = static_cast<uint32_t>(rnd() % pool);
                        dup = false;
                        for (uint32_t q = 0; q < j; ++q)
                            dup |= ids[(size_t(t) * kLayers + L) * kTopk + q] == e;
                    }
                    ids[(size_t(t) * kLayers + L) * kTopk + j] = e;
                }
        // Reference: the simulator's LRU, token-major.
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t L = 0; L < kLayers; ++L)
                for (uint32_t j = 0; j < kTopk; ++j) {
                    const uint32_t k = L * kExperts + ids[(size_t(t) * kLayers + L) * kTopk + j];
                    auto it = std::find(ref.begin(), ref.end(), k);
                    if (it != ref.end()) ref.erase(it);
                    ref.push_back(k);
                    if (ref.size() > cap) ref.erase(ref.begin());
                }
        // Streamed: layer-major, experts in id order ("shard order"), batches of
        // two whose admissions happen before the previous batch has settled.
        const TokenIndex base = 1000;
        TimelineValue guard_clock = 0;
        uint32_t dropped = 0;
        struct Held { StreamAdmit a; TokenIndex stamp; TimelineValue guard; };
        for (uint32_t L = 0; L < kLayers; ++L) {
            std::vector<int64_t> last(kExperts, -1);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t j = 0; j < kTopk; ++j)
                    last[ids[(size_t(t) * kLayers + L) * kTopk + j]] =
                        int64_t((uint64_t(t) * kLayers + L) * kTopk + j);
            std::vector<uint32_t> used;
            for (uint32_t e = 0; e < kExperts; ++e) if (last[e] >= 0) used.push_back(e);
            std::vector<Held> prev, cur;
            for (size_t i = 0; i < used.size(); i += 2) {
                cur.clear();
                for (size_t q = i; q < std::min(used.size(), i + 2); ++q) {
                    const uint32_t e = used[q];
                    const TokenIndex stamp = base + TokenIndex(last[e]);
                    const TimelineValue g = ++guard_clock;
                    auto a = admit_streamed(s, ExpertKey{static_cast<uint16_t>(L), static_cast<uint16_t>(e)},
                                            stamp, nullptr, g);
                    REQUIRE_OK(a);
                    if (a->kind == StreamKind::Drop) ++dropped;
                    cur.push_back({*a, stamp, g});
                }
                // The previous batch "computes" only now, then settles.
                for (const Held& h : prev) REQUIRE_OK(finish_streamed(s, h.a, true, h.stamp));
                s.set_completed_timeline(cur.front().guard - 1);
                prev = cur;
            }
            for (const Held& h : prev) REQUIRE_OK(finish_streamed(s, h.a, true, h.stamp));
            s.set_completed_timeline(guard_clock);
        }
        std::vector<uint32_t> got;
        for (uint32_t k = 0; k < kLayers * kExperts; ++k)
            if (s.resident(ExpertKey{static_cast<uint16_t>(k / kExperts), static_cast<uint16_t>(k % kExperts)}))
                got.push_back(k);
        std::vector<uint32_t> want = ref;
        std::sort(want.begin(), want.end());
        CHECK(got == want);
        CHECK_EQ(s.stats().filling, 0u);
        if (trial == 0) CHECK(dropped > 0);
    }
}
