// storage/: chunk planning, priority ordering and preemption against a fake
// backend (design §9.6), plus one real unbuffered round trip through the
// platform backend (IOCP on Windows, io_uring on Linux).
#include "tests/env_guard.h"
#include "core/env.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "model/layout.h"
#include "runtime/engine.h"
#include "storage/backend.h"
#include "storage/file.h"
#include "storage/io_engine.h"
#include "tests/fake_backend.h"
#include "tests/test_framework.h"

using namespace cachedmoe;
using namespace cachedmoe::storage;

CACHEDMOE_TEST(io, gpu_options_resolve_once_and_preserve_overrides) {
    test::ScopedEnvironment route("CACHEDMOE_BATCH_GPU_ROUTE"), early("CACHEDMOE_BATCH_ENGRAM_EARLY"),
      onecb("CACHEDMOE_DSPARK_ONECB"), readout("CACHEDMOE_SPEC_GPU_READOUT");
    route.set("1");
    early.set("0");
    onecb.set("1");
    readout.set("0");
    RuntimeConfig cfg;
    cfg.gpu.apply_environment();
    route.set("0");
    early.set("1");
    onecb.set("0");
    readout.set("1");
    CHECK(cfg.gpu.batch_gpu_route && cfg.gpu.draft_onecb);
    CHECK(!cfg.gpu.batch_engram_early && !cfg.gpu.spec_gpu_readout);
    GpuExecutionConfig next;
    next.apply_environment();
    CHECK(!next.batch_gpu_route && !next.draft_onecb);
    CHECK(next.batch_engram_early && next.spec_gpu_readout);
    // With no environment override, an explicit API configuration survives.
    route.set(nullptr);
    next.batch_gpu_route = true;
    next.apply_environment();
    CHECK(next.batch_gpu_route);
}

CACHEDMOE_TEST(io, weighted_mask_rejects_invalid_limits_and_speculation) {
    test::ScopedEnvironment tau("CACHEDMOE_MASK_WAIT_TAU"), budget("CACHEDMOE_MASK_WAIT_BUDGET");
    budget.set(nullptr);
    RuntimeConfig cfg;
    runtime::Engine engine;
    for (const char* bad : {"", "nan", "-0.1", "1.1", ".2junk"}) {
        tau.set(bad); CHECK_ERR(engine.init(cfg), Err::InvalidArgument);
    }
    tau.set(".1");
    for (const char* bad : {"8", "-1,20", "8,-1", "8,nan", "8,", "8,20junk"}) {
        budget.set(bad); CHECK_ERR(engine.init(cfg), Err::InvalidArgument);
    }
    budget.set("0,0"); cfg.speculation.enabled = true;
    CHECK_ERR(engine.init(cfg), Err::FailedPrecondition);
}

CACHEDMOE_TEST(io, mask_cache_policy_default_and_explicit_override) {
    test::ScopedEnvironment environment("CACHEDMOE_MASK_DYNAMIC_LRU");
    auto set = [&](const char* value) { environment.set(value); };
    runtime::Engine normal;
    normal.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(!normal.store().fixed_cache());
    set("0");
    runtime::Engine frozen;
    frozen.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(frozen.store().fixed_cache());
    frozen.set_mask_cache_fixed(false); // explicit CLI beats legacy env
    frozen.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(!frozen.store().fixed_cache());
    set("1");
    normal.set_mask_cache_fixed(true);
    normal.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(normal.store().fixed_cache());
    normal.set_resident_only(runtime::Engine::ResidentOnly::Off);
    CHECK(!normal.store().fixed_cache());
    normal.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    CHECK(normal.store().fixed_cache());
}

namespace {

std::vector<std::byte> pattern_bytes(size_t n) {
    std::vector<std::byte> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<std::byte>((i * 31 + (i >> 8)) & 0xFF);
    return v;
}

// A File the fake backend never actually reads from -- IoEngine only needs it
// open for validation. A temp file of the right size is the simplest way to
// get one without special-casing the engine.
struct ScratchFile {
    std::string path;
    File        file;

    ScratchFile() = default;
    // A user-declared destructor suppresses the implicit move operations, and
    // File is move-only, so they have to be brought back explicitly.
    ScratchFile(ScratchFile&&) = default;
    ScratchFile& operator=(ScratchFile&&) = default;
    ~ScratchFile() { file.close(); if (!path.empty()) (void)remove_file(path); }
};

Result<ScratchFile> make_scratch(const char* name, uint64_t bytes, bool unbuffered) {
    ScratchFile s;
    s.path = temp_dir() + "/deepmoe_test_" + name + ".bin";
    (void)remove_file(s.path);
    FileFlags flags = FileFlags::Create | FileFlags::Write | FileFlags::Overlapped;
    if (unbuffered) flags = flags | FileFlags::Unbuffered;
    auto f = File::open(s.path, flags);
    if (!f) return std::unexpected(f.error());
    s.file = *std::move(f);
    if (auto r = s.file.set_size(bytes); !r) return std::unexpected(r.error());
    return s;
}

}  // namespace

CACHEDMOE_TEST(io, chunk_planning) {
    // design §9.6: an 18.8 MB expert is split into 4 MiB chunks.
    auto c = IoEngine::plan_chunks(0, layout::kExpertBytes, 4u << 20, 4096);
    REQUIRE_EQ(c.size(), 5u);
    uint64_t total = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        CHECK(is_aligned(c[i].off));
        if (i) CHECK_EQ(c[i].off, c[i - 1].off + c[i - 1].bytes);
        total += c[i].bytes;
    }
    CHECK_EQ(total, layout::kExpertBytes);
    CHECK_EQ(c[0].off, 0u);
    CHECK_EQ(c[0].bytes, 4u << 20);
    // The tail is the remainder, and it is still 4 KiB aligned because the
    // expert size is (design §2.3).
    CHECK_EQ(c[4].bytes, static_cast<uint32_t>(layout::kExpertBytes - 4ull * (4u << 20)));
    CHECK_EQ(c[4].bytes % 4096, 0u);
}

CACHEDMOE_TEST(io, engram_wait_bypasses_async_p0_backlog) {
    const size_t bytes = 1u << 20;
    auto content = pattern_bytes(bytes);
    auto scratch = make_scratch("engramdeadline", bytes, false);
    REQUIRE_OK(scratch);
    IoConfig cfg;
    cfg.chunk_bytes = 4096;
    cfg.max_inflight_ops = 2;
    cfg.max_inflight_bytes = bytes;
    cfg.engram_qd = 12;
    auto backend = std::make_unique<test::FakeBackend>(content, 16);
    auto* fake = backend.get();
    fake->hold_completions(true);
    runtime::Engine owner;
    auto& engine = owner.io();
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    AlignedBuffer expert(32 * 4096), rows(20 * 4096);
    IoRequest p0;
    p0.file = &scratch->file;
    p0.priority = IoPriority::BlockingMiss;
    p0.bytes = expert.size(); p0.dst = expert.data();
    auto p0_done = engine.submit_future(p0);
    REQUIRE_OK(p0_done);
    auto wait_count = [&](size_t n) {
        const auto until = Clock::now() + std::chrono::seconds(1);
        while (fake->submit_count() < n && Clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    wait_count(2);
    std::vector<std::future<IoResult>> pending;
    for (uint32_t i = 0; i < 20; ++i) {
        IoRequest r;
        r.priority = IoPriority::Engram; r.file = &scratch->file;
        r.file_off = (64 + i) * 4096; r.bytes = 4096;
        r.dst = rows.data() + i * 4096;
        auto f = engine.submit_future(r);
        REQUIRE_OK(f);
        pending.push_back(std::move(*f));
    }
    {
        // Off mode's scope is inert: no P2 can pass a held P0 queue.
        auto ordinary = engine.engram_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK_EQ(fake->submit_count(), 2u);
    }
    // The serve CLI sets this after init. Exercise that same setter rather
    // than manually enabling the scheduler, so policy wiring is covered.
    owner.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    {
        // Dynamic mask preserves P0 catch-up by default. No P2 bypass.
        auto ordinary = engine.engram_wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK_EQ(fake->submit_count(), 2u);
        CHECK_EQ(engine.stats().p2_deadline_chunks, 0u);
    }
    owner.set_mask_cache_fixed(true);
    owner.set_resident_only(runtime::Engine::ResidentOnly::Mask);
    {
        auto urgent = engine.engram_wait();
        wait_count(12);
        CHECK_EQ(fake->submit_count(), 12u); // backend/Engram QD still enforced
        CHECK_EQ(engine.inflight_chunks(IoPriority::BlockingMiss), 2u);
        CHECK_EQ(engine.inflight_chunks(IoPriority::Engram), 10u);
        CHECK_EQ(engine.stats().inflight_by_priority[0], 2u);
        CHECK_EQ(engine.stats().inflight_by_priority[2], 10u);
        CHECK(engine.stats().peak_by_priority[2] >= 10u);
        CHECK_EQ(engine.stats().p2_deadline_chunks, 10u);
    }
    fake->release_all();
    engine.drain();
    CHECK(p0_done->get().ok());
    for (auto& f : pending) CHECK(f.get().ok());
    const auto log = fake->log();
    // Leaving the scope restores strict P0 ordering for the unissued rows.
    CHECK_EQ(log[12].file_off, 2u * 4096);
    CHECK_EQ(std::memcmp(expert.data(), content.data(), expert.size()), 0);
    CHECK_EQ(std::memcmp(rows.data(), content.data() + 64 * 4096, rows.size()), 0);
    CHECK_EQ(engine.stats().p2_requests, 20u);
    CHECK_EQ(engine.stats().requests_cancelled, 0u);
    engine.stop();
}

CACHEDMOE_TEST(io, engram_wait_has_a_bounded_issue_window) {
    const size_t bytes = 2u << 20;
    auto content = pattern_bytes(bytes);
    auto scratch = make_scratch("engramdeadlinecap", bytes, false);
    REQUIRE_OK(scratch);
    IoConfig cfg;
    cfg.chunk_bytes = 4096; cfg.max_inflight_ops = 2;
    cfg.max_inflight_bytes = bytes; cfg.engram_qd = 256;
    auto backend = std::make_unique<test::FakeBackend>(content, 256);
    auto* fake = backend.get(); fake->hold_completions(true);
    IoEngine engine;
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    engine.set_engram_wait_priority(true);
    AlignedBuffer expert(32 * 4096), rows(128 * 4096);
    IoRequest r;
    r.priority = IoPriority::BlockingMiss; r.file = &scratch->file;
    r.bytes = expert.size(); r.dst = expert.data();
    REQUIRE_OK(engine.submit(r, [](const IoResult&) {}));
    auto await_count = [&](size_t n) {
        const auto until = Clock::now() + std::chrono::seconds(1);
        while (fake->submit_count() < n && Clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    await_count(2);
    for (uint32_t i = 0; i < 128; ++i) {
        r.priority = IoPriority::Engram;
        r.file_off = (64 + i) * 4096; r.bytes = 4096;
        r.dst = rows.data() + i * 4096;
        REQUIRE_OK(engine.submit(r, [](const IoResult&) {}));
    }
    {
        auto waiting = engine.engram_wait();
        await_count(2 + IoEngine::kEngramWaitChunks);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK_EQ(engine.inflight_chunks(IoPriority::Engram), IoEngine::kEngramWaitChunks);
        CHECK_EQ(fake->submit_count(), 2u + IoEngine::kEngramWaitChunks);
    }
    fake->release_all(); engine.drain();
    CHECK_EQ(engine.stats().requests_completed, 129u);
    CHECK_EQ(engine.stats().requests_failed, 0u);
    CHECK_EQ(std::memcmp(rows.data(), content.data() + 64 * 4096, rows.size()), 0);
    engine.stop();
}

CACHEDMOE_TEST(io, chunk_planning_edge_cases) {
    CHECK(IoEngine::plan_chunks(0, 0, 4096, 4096).empty());
    CHECK(IoEngine::plan_chunks(0, 4096, 0, 4096).empty());

    // A chunk size below the alignment is clamped up to it, never to zero.
    auto c = IoEngine::plan_chunks(0, 8192, 100, 4096);
    REQUIRE_EQ(c.size(), 2u);
    CHECK_EQ(c[0].bytes, 4096u);

    // A non-multiple chunk size is rounded down so every chunk but the last
    // starts on a sector boundary.
    auto d = IoEngine::plan_chunks(4096, 3 * 4096, 5000, 4096);
    REQUIRE_EQ(d.size(), 3u);
    CHECK_EQ(d[0].off, 4096u);
    CHECK_EQ(d[0].bytes, 4096u);
    CHECK_EQ(d[2].off, 3u * 4096u);

    // One chunk when it all fits.
    auto e = IoEngine::plan_chunks(8192, 4096, 1u << 20, 4096);
    REQUIRE_EQ(e.size(), 1u);
    CHECK_EQ(e[0].off, 8192u);
    CHECK_EQ(e[0].bytes, 4096u);
}

CACHEDMOE_TEST(io, fake_backend_round_trip) {
    const size_t kFileBytes = 1u << 20;
    auto content = pattern_bytes(kFileBytes);
    auto scratch = make_scratch("fake_rt", kFileBytes, false);
    REQUIRE_OK(scratch);

    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    cfg.max_inflight_ops = 8;
    cfg.max_inflight_bytes = 1u << 20;

    IoEngine engine;
    auto backend = std::make_unique<test::FakeBackend>(content, 8);
    test::FakeBackend* fake = backend.get();
    fake->set_copy_ns(100);
    REQUIRE_OK(engine.start(std::move(backend), cfg));

    AlignedBuffer dst(256 * 1024);
    REQUIRE(static_cast<bool>(dst));
    IoRequest req;
    req.key      = ExpertKey{7, 11};
    req.priority = IoPriority::BlockingMiss;
    req.file     = &scratch->file;
    req.file_off = 4096;
    req.bytes    = 256 * 1024;
    req.dst      = dst.data();

    auto fut = engine.submit_future(req);
    REQUIRE_OK(fut);
    const IoResult r = fut->get();
    CHECK(r.ok());
    CHECK_EQ(r.bytes_moved, req.bytes);
    CHECK_EQ(r.key, req.key);
    CHECK_EQ(std::memcmp(dst.data(), content.data() + 4096, req.bytes), 0);
    // 256 KiB at 64 KiB chunks.
    CHECK_EQ(fake->submit_count(), 4u);

    engine.drain();
    const IoStats st = engine.stats();
    CHECK_EQ(st.p0_copy_ns_sum, st.p0_chunks_issued * 100);
    CHECK_EQ(st.p0_queue_wait_ns_sum + st.p0_service_ns_sum, st.p0_lat_ns_sum);
    CHECK_EQ(st.requests_submitted, 1u);
    CHECK_EQ(st.requests_completed, 1u);
    CHECK_EQ(st.bytes_completed, req.bytes);
    CHECK_EQ(st.per_priority_requests[0], 1u);
    CHECK(st.busy_ns > 0);
    engine.stop();
}

CACHEDMOE_TEST(io, priority_ordering_and_preemption) {
    // design §9.6: P0 preempts. With completions held, the engine fills its
    // queue from the lowest-numbered non-empty class only, so a P0 submitted
    // after a P3 is still issued first.
    const size_t kFileBytes = 4u << 20;
    auto content = pattern_bytes(kFileBytes);
    auto scratch = make_scratch("prio", kFileBytes, false);
    REQUIRE_OK(scratch);

    IoConfig cfg;
    cfg.chunk_bytes        = 64 * 1024;
    cfg.max_inflight_ops   = 2;          // tiny window so ordering is observable
    cfg.max_inflight_bytes = 128 * 1024;

    IoEngine engine;
    auto backend = std::make_unique<test::FakeBackend>(content, 64);
    test::FakeBackend* fake = backend.get();
    fake->hold_completions(true);
    REQUIRE_OK(engine.start(std::move(backend), cfg));

    AlignedBuffer b0(256 * 1024), b1(256 * 1024), b3(256 * 1024);
    auto mk = [&](IoPriority p, uint64_t off, AlignedBuffer& buf) {
        IoRequest q;
        q.priority = p;
        q.file     = &scratch->file;
        q.file_off = off;
        q.bytes    = 256 * 1024;
        q.dst      = buf.data();
        return q;
    };

    // Submit worst-priority first, then better ones, while nothing completes.
    std::atomic<int> done{0};
    REQUIRE_OK(engine.submit(mk(IoPriority::Backfill,     2u << 20, b3), [&](const IoResult&) { ++done; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE_OK(engine.submit(mk(IoPriority::Lookahead,    1u << 20, b1), [&](const IoResult&) { ++done; }));
    REQUIRE_OK(engine.submit(mk(IoPriority::BlockingMiss, 0,        b0), [&](const IoResult&) { ++done; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    auto log = fake->log();
    REQUIRE(log.size() >= 2);
    // The in-flight window is 2 chunks and they were both taken by the P3 that
    // arrived first; nothing else could be issued until something retires.
    CHECK(log[0].file_off >= (2u << 20));
    CHECK_EQ(fake->submit_count(), 2u);

    fake->release_all();
    engine.drain();
    CHECK_EQ(done.load(), 3);

    // Once the window opened, the P0 and P1 had to be issued before the rest of
    // the P3: find the first chunk of each request in the submit log.
    log = fake->log();
    size_t first_p0 = SIZE_MAX, first_p1 = SIZE_MAX, third_p3 = SIZE_MAX;
    int p3_seen = 0;
    for (size_t i = 0; i < log.size(); ++i) {
        const uint64_t off = log[i].file_off;
        if (off < (1u << 20)) { if (first_p0 == SIZE_MAX) first_p0 = i; }
        else if (off < (2u << 20)) { if (first_p1 == SIZE_MAX) first_p1 = i; }
        else { if (++p3_seen == 3) third_p3 = i; }
    }
    REQUIRE(first_p0 != SIZE_MAX);
    REQUIRE(first_p1 != SIZE_MAX);
    REQUIRE(third_p3 != SIZE_MAX);
    CHECK(first_p0 < first_p1);        // P0 before P1
    CHECK(first_p0 < third_p3);        // both before the P3 tail
    CHECK(first_p1 < third_p3);

    // Every byte still landed where it belonged.
    CHECK_EQ(std::memcmp(b0.data(), content.data(), 256 * 1024), 0);
    CHECK_EQ(std::memcmp(b1.data(), content.data() + (1u << 20), 256 * 1024), 0);
    CHECK_EQ(std::memcmp(b3.data(), content.data() + (2u << 20), 256 * 1024), 0);

    const IoStats st = engine.stats();
    CHECK_EQ(st.requests_completed, 3u);
    CHECK_EQ(st.per_priority_requests[0], 1u);
    CHECK_EQ(st.per_priority_requests[1], 1u);
    CHECK_EQ(st.per_priority_requests[3], 1u);
    CHECK(st.peak_inflight_ops <= 2);
    engine.stop();
}

CACHEDMOE_TEST(io, submit_validation_and_cancel) {
    const size_t kFileBytes = 1u << 20;
    auto scratch = make_scratch("valid", kFileBytes, false);
    REQUIRE_OK(scratch);

    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    IoEngine engine;
    auto backend = std::make_unique<test::FakeBackend>(pattern_bytes(kFileBytes), 64);
    test::FakeBackend* fake = backend.get();
    fake->hold_completions(true);
    REQUIRE_OK(engine.start(std::move(backend), cfg));

    AlignedBuffer dst(64 * 1024);
    IoRequest good;
    good.file = &scratch->file;
    good.bytes = 64 * 1024;
    good.dst = dst.data();
    good.priority = IoPriority::Backfill;

    IoRequest no_file = good; no_file.file = nullptr;
    CHECK_ERR(engine.submit(no_file, {}), Err::InvalidArgument);
    IoRequest no_dst = good; no_dst.dst = nullptr;
    CHECK_ERR(engine.submit(no_dst, {}), Err::InvalidArgument);
    IoRequest no_bytes = good; no_bytes.bytes = 0;
    CHECK_ERR(engine.submit(no_bytes, {}), Err::InvalidArgument);

    // Cancelling a request that has not been issued yet succeeds; an unknown id
    // is NotFound.
    fake->hold_completions(true);
    auto id = engine.submit(good, {});
    REQUIRE_OK(id);
    const auto cancelled = engine.cancel(*id);
    CHECK(cancelled || cancelled.error().code == Err::FailedPrecondition);
    CHECK_ERR(engine.cancel(999999), Err::NotFound);

    fake->release_all();
    engine.drain();
    engine.stop();
    // Submitting to a stopped engine must fail rather than hang.
    CHECK_ERR(engine.submit(good, {}), Err::FailedPrecondition);
}

CACHEDMOE_TEST(io, platform_backend_reads_a_real_unbuffered_file) {
    // The real thing: FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED + IOCP on
    // Windows (design §9.6), O_DIRECT + io_uring on Linux.
    const uint64_t kBytes = 8u << 20;
    const std::string path = temp_dir() + "/deepmoe_test_real.bin";
    (void)remove_file(path);

    // Write the content buffered, then reopen unbuffered for the read.
    auto content = pattern_bytes(kBytes);
    {
        auto w = File::open(path, FileFlags::Create | FileFlags::Write | FileFlags::Overlapped);
        REQUIRE_OK(w);
        REQUIRE_OK(w->set_size(kBytes));
        auto n = w->write_at(0, ByteSpan(content.data(), content.size()));
        REQUIRE_OK(n);
        CHECK_EQ(*n, kBytes);
    }

    auto f = File::open_read(path, true);
    REQUIRE_OK(f);
    CHECK(f->unbuffered());
    CHECK(f->sector_size() <= 4096);
    CHECK_EQ(f->size(), kBytes);

    // A misaligned unbuffered read must be refused by us, not by the OS.
    AlignedBuffer small(8192);
    CHECK_ERR(f->read_at(1, MutBytes(small.data(), 4096)), Err::InvalidArgument);
    CHECK_ERR(f->read_at(0, MutBytes(small.data(), 100)), Err::InvalidArgument);

    // Synchronous aligned read.
    auto got = f->read_at(4096, MutBytes(small.data(), 8192));
    REQUIRE_OK(got);
    CHECK_EQ(*got, 8192u);
    CHECK_EQ(std::memcmp(small.data(), content.data() + 4096, 8192), 0);

    // Asynchronous read through the real backend.
    IoConfig cfg;
    cfg.chunk_bytes        = 1u << 20;
    cfg.max_inflight_ops   = 8;
    cfg.max_inflight_bytes = 8u << 20;
    cfg.completion_threads = 2;

    auto backend = make_default_backend(cfg);
    REQUIRE_OK(backend);
    CHECK(!(*backend)->caps().name.empty());

    IoEngine engine;
    REQUIRE_OK(engine.start(std::move(*backend), cfg));

    AlignedBuffer dst(static_cast<size_t>(kBytes));
    REQUIRE(static_cast<bool>(dst));
    IoRequest req;
    req.key      = ExpertKey{1, 2};
    req.priority = IoPriority::BlockingMiss;
    req.file     = &*f;
    req.file_off = 0;
    req.bytes    = kBytes;
    req.dst      = dst.data();

    auto fut = engine.submit_future(req);
    REQUIRE_OK(fut);
    const IoResult r = fut->get();
    CHECK(r.ok());
    CHECK_EQ(r.bytes_moved, kBytes);
    CHECK_EQ(std::memcmp(dst.data(), content.data(), static_cast<size_t>(kBytes)), 0);
    CHECK(r.latency.count() > 0);

    engine.drain();
    const IoStats st = engine.stats();
    CHECK_EQ(st.requests_completed, 1u);
    CHECK_EQ(st.chunks_completed, 8u);       // 8 MiB at 1 MiB chunks
    CHECK_EQ(st.bytes_completed, kBytes);
    CHECK(st.effective_gbps() > 0.0);
    engine.stop();

    f->close();
    CHECK_OK(remove_file(path));
}

CACHEDMOE_TEST(io, opening_a_missing_file_fails_cleanly) {
    auto f = File::open_read(temp_dir() + "/deepmoe_does_not_exist_9f3a.bin", true);
    CHECK(!f);
    CHECK_EQ(f.error().code, Err::Io);
    CHECK(f.error().os_code != 0);
    File closed;
    CHECK(!closed.is_open());
    AlignedBuffer b(4096);
    CHECK_ERR(closed.read_at(0, MutBytes(b.data(), 4096)), Err::FailedPrecondition);
}

// Track R1 (docs/p4_hitrate.md 5): P3 is not only behind a queued P0, it is
// throttled to IoEngine::kBackgroundOpsWhileBusy chunks while P0 work is recent,
// and gets the whole queue depth back once the drive has been quiet for
// kBackgroundQuiet.
CACHEDMOE_TEST(io, background_is_throttled_while_p0_is_recent) {
    const size_t kFileBytes = 4u << 20;
    auto content = pattern_bytes(kFileBytes);
    auto scratch = make_scratch("bgthrottle", kFileBytes, false);
    REQUIRE_OK(scratch);
    IoConfig cfg;
    cfg.chunk_bytes        = 64 * 1024;
    cfg.max_inflight_ops   = 4;
    cfg.max_inflight_bytes = 1u << 20;
    IoEngine engine;
    auto backend = std::make_unique<test::FakeBackend>(content, 64);
    test::FakeBackend* fake = backend.get();
    fake->hold_completions(true);
    REQUIRE_OK(engine.start(std::move(backend), cfg));

    AlignedBuffer b0(64 * 1024), b3(256 * 1024);
    IoRequest p0;
    p0.priority = IoPriority::BlockingMiss;
    p0.file     = &scratch->file;
    p0.bytes    = 64 * 1024;
    p0.dst      = b0.data();
    IoRequest p3 = p0;
    p3.priority = IoPriority::Backfill;
    p3.file_off = 1u << 20;
    p3.bytes    = 256 * 1024;       // four chunks
    p3.dst      = b3.data();
    REQUIRE_OK(engine.submit(p0, [](const IoResult&) {}));
    REQUIRE_OK(engine.submit(p3, [](const IoResult&) {}));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK_EQ(engine.inflight_chunks(IoPriority::BlockingMiss), 1u);
    CHECK_EQ(engine.inflight_chunks(IoPriority::Backfill), IoEngine::kBackgroundOpsWhileBusy);
    std::this_thread::sleep_for(IoEngine::kBackgroundQuiet + std::chrono::milliseconds(60));
    CHECK_EQ(engine.inflight_chunks(IoPriority::Backfill), 3u);   // the queue depth less the P0
    fake->release_all();
    engine.drain();
    CHECK_EQ(std::memcmp(b3.data(), content.data() + (1u << 20), 256 * 1024), 0);
    engine.stop();
}

// STATUS §7 0v: the engram class runs at its own depth (IoConfig::engram_qd),
// not the background classes' 8, and the backfill keeps 8.
CACHEDMOE_TEST(io, engram_rows_run_at_their_own_depth) {
    const size_t kFileBytes = 1u << 20;
    auto content = pattern_bytes(kFileBytes);
    auto scratch = make_scratch("engramqd", kFileBytes, false);
    REQUIRE_OK(scratch);
    IoConfig cfg;
    cfg.chunk_bytes        = 4096;
    cfg.max_inflight_ops   = 16;
    cfg.max_inflight_bytes = 1u << 20;
    cfg.engram_qd          = 12;
    IoEngine engine;
    auto backend = std::make_unique<test::FakeBackend>(content, 64);
    test::FakeBackend* fake = backend.get();
    fake->hold_completions(true);
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    CHECK_EQ(engine.tuning().engram_qd, 12u);
    CHECK_EQ(engine.tuning().bg_qd, 8u);

    AlignedBuffer dst(32 * 4096);
    auto burst = [&](IoPriority cls, uint64_t first) {
        for (uint32_t i = 0; i < 16; ++i) {
            IoRequest r;
            r.priority = cls;
            r.file     = &scratch->file;
            r.file_off = (first + i) * 4096;
            r.bytes    = 4096;
            r.dst      = dst.data() + (first + i) * 4096;
            REQUIRE_OK(engine.submit(r, [](const IoResult&) {}));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    };
    burst(IoPriority::Engram, 0);
    CHECK_EQ(engine.inflight_chunks(IoPriority::Engram), 12u);
    fake->release_all();   // also stops holding
    engine.drain();
    fake->hold_completions(true);
    burst(IoPriority::Backfill, 16);
    CHECK_EQ(engine.inflight_chunks(IoPriority::Backfill), 8u);
    fake->release_all();
    engine.drain();
    CHECK_EQ(std::memcmp(dst.data(), content.data(), 32 * 4096), 0);
    engine.stop();

#if defined(__linux__)
    // serve's shape: the ring grows to the engram depth, P0 keeps its 8
    IoConfig rs;
    IoEngine::runtime_shape(rs);
    CHECK_EQ(rs.engram_qd, 512u);
    CHECK(rs.max_inflight_ops >= 512u);
    CHECK_EQ(rs.p0_qd, 8u);
#endif
}

// --- Track D2: the second read source (docs/p4_dual_source.md) --------------

// The chooser on its own, with no engine and no drives.
//
// The thing to test is NOT the steady-state byte split: plain
// least-outstanding-bytes is self-balancing, because a slow source drains
// slowly and so accumulates queue until it stops being chosen, and it too ends
// up rate-proportional in the limit. What the weights buy is the PER-REQUEST
// choice, and that is where the two rules differ outright: with D: at 4.6 GB/s
// and E: at 1.0, a 9 MiB run should still go to D: while D: is carrying up to
// 3.6 runs, because 4.6 of it drains in the time E: would take to do one.
// Dropping the divide moves that crossover from 3.6 queued runs to 0 -- every
// request that finds D: non-empty goes to the USB drive, which is the mutation
// this case is written against.
CACHEDMOE_TEST(io, source_router_respects_weights) {
    const double w[2] = {4.6, 1.0};
    constexpr uint64_t kRun = 9u << 20;      // one expert run

    // Crossover: pick 0 while (o0 + B)/4.6 <= B, i.e. o0 <= 3.6 B.
    struct Case { double queued_runs; uint32_t want; };
    const Case cases[] = {
        {0.0, 0},   // both idle -> the fast drive, not an alternation
        {1.0, 0},   // one run queued on D: -- LOB would already flip here
        {3.0, 0},   // still faster to wait behind three runs on D:
        {4.5, 1},   // now E: really is the sooner finish
        {8.0, 1},
    };
    for (const Case& c : cases) {
        const uint64_t outstanding[2] = {
            static_cast<uint64_t>(c.queued_runs * double(kRun)), 0};
        CHECK_EQ(pick_source(std::span<const double>(w, 2),
                             std::span<const uint64_t>(outstanding, 2), 0b11, kRun),
                 c.want);
    }

    // And the symmetric direction: a queue on the SLOW source is worth much
    // less than the same queue on the fast one, so one run on E: is already
    // enough to send the next back to D:.
    const uint64_t slow_busy[2] = {0, kRun};
    CHECK_EQ(pick_source(std::span<const double>(w, 2),
                         std::span<const uint64_t>(slow_busy, 2), 0b11, kRun), 0u);

    // Over a run of requests against drives that drain at their own rates, the
    // weighted rule leaves the fast drive with the large majority of them.
    // (Both rules converge to a similar byte split; this is a sanity check on
    // the loop, not the discriminator above.)
    uint64_t outstanding[2] = {0, 0};
    uint32_t picks[2] = {0, 0};
    constexpr double kDt = 1e-3;
    for (int step = 0; step < 2000; ++step) {
        for (uint32_t s = 0; s < 2; ++s) {
            const uint64_t drained = static_cast<uint64_t>(w[s] * 1e9 * kDt);
            outstanding[s] = outstanding[s] > drained ? outstanding[s] - drained : 0;
        }
        const uint32_t s = pick_source(std::span<const double>(w, 2),
                                       std::span<const uint64_t>(outstanding, 2), 0b11, kRun);
        REQUIRE(s < 2);
        ++picks[s];
        outstanding[s] += kRun;
    }
    CHECK(picks[0] > picks[1] * 2);
}

// Degenerate inputs the runtime actually produces: one source, or a shard that
// only the primary holds. Both have to come back as source 0, because that is
// what makes "no mirror configured" byte-identical to the old behaviour.
CACHEDMOE_TEST(io, source_router_defaults_to_the_primary) {
    const double w[2] = {4.6, 1.0};
    const uint64_t zero[2] = {0, 0};
    CHECK_EQ(pick_source(std::span<const double>(w, 1), std::span<const uint64_t>(zero, 1),
                         0b1, 1 << 20), 0u);
    // Only the primary holds this shard.
    CHECK_EQ(pick_source(std::span<const double>(w, 2), std::span<const uint64_t>(zero, 2),
                         0b1, 1 << 20), 0u);
    // Both idle: the faster source wins, it does not alternate.
    CHECK_EQ(pick_source(std::span<const double>(w, 2), std::span<const uint64_t>(zero, 2),
                         0b11, 1 << 20), 0u);
    // Nothing holds it: the caller has to fall back.
    CHECK_EQ(pick_source(std::span<const double>(w, 2), std::span<const uint64_t>(zero, 2),
                         0u, 1 << 20), kMaxIoSources);
    // A primary carrying a full queue hands the next one to the slow drive.
    const uint64_t busy[2] = {96u << 20, 0};
    CHECK_EQ(pick_source(std::span<const double>(w, 2), std::span<const uint64_t>(busy, 2),
                         0b11, 9u << 20), 1u);
}

// Track D4 (docs/p4_e_drive_diag.md §5.2): a mirror that stops answering is
// dropped, and "stops answering" means CONSECUTIVE failures.
//
// The discriminator is the success in the middle. A drive that has fallen off
// the bus fails every read after the first -- Track DX watched E: go from 0.85
// GB/s to zero completions inside one second and never come back -- so three in
// a row is a dead source. A drive that returns one error in a million and
// serves everything else is a working drive, and a cumulative counter would
// eventually drop it on a long enough run for no reason. Deleting the reset in
// note_success() (or counting `errors` instead of `consecutive`) is the
// mutation this case is written against.
CACHEDMOE_TEST(io, source_health_drops_a_mirror_that_keeps_failing) {
    SourceHealth h(3);
    CHECK_EQ(h.live_mask(0b11), 0b11u);

    // Under budget: still a candidate.
    CHECK(!h.note_error(1));
    CHECK(!h.note_error(1));
    CHECK_EQ(h.consecutive_errors(1), 2u);
    CHECK(!h.dropped(1));
    CHECK_EQ(h.live_mask(0b11), 0b11u);

    // A success in between puts the whole budget back -- this is the line the
    // mutation deletes.
    h.note_success(1);
    CHECK_EQ(h.consecutive_errors(1), 0u);
    CHECK(!h.note_error(1));
    CHECK(!h.note_error(1));
    CHECK(!h.dropped(1));
    CHECK_EQ(h.live_mask(0b11), 0b11u);

    // Three in a row, and only the third one reports the drop, so the caller
    // logs once rather than once per failed read.
    h.note_success(1);
    CHECK(!h.note_error(1));
    CHECK(!h.note_error(1));
    CHECK(h.note_error(1));
    CHECK(h.dropped(1));
    CHECK(!h.note_error(1));           // already out: no second announcement
    CHECK_EQ(h.live_mask(0b11), 0b01u);

    // And the router never picks it again, even while it is the idle one and
    // the primary is carrying a full queue -- the case that would otherwise
    // send the next 9 MiB run straight back to the dead drive.
    const double w[2] = {4.6, 1.0};
    const uint64_t busy[2] = {96u << 20, 0};
    CHECK_EQ(pick_source(std::span<const double>(w, 2), std::span<const uint64_t>(busy, 2),
                         h.live_mask(0b11), 9u << 20), 0u);

    // The primary is never dropped: it is the only copy the run is guaranteed
    // to have, and hiding its failures here would turn a hard I/O error into a
    // silently wrong read.
    SourceHealth p(1);
    for (int i = 0; i < 10; ++i) CHECK(!p.note_error(0));
    CHECK(!p.dropped(0));
    CHECK_EQ(p.live_mask(0b11), 0b11u);

    // The startup gate uses the same switch, with no errors involved.
    SourceHealth g(3);
    g.drop(1);
    CHECK(g.dropped(1));
    CHECK_EQ(g.live_mask(0b11), 0b01u);
}

// A hot mirror rests at hot_c and comes back only at cool_c: in between it
// stays whatever it was, so a drive sitting at the threshold does not flap in
// and out of the router every second. The primary never rests.
CACHEDMOE_TEST(io, thermal_gate_rests_a_hot_mirror_with_hysteresis) {
    ThermalGate t{80, 72, 0};
    CHECK(!t.update(1, 74));
    CHECK_EQ(t.live_mask(0b11), 0b11u);
    CHECK(t.update(1, 80));                 // reaches hot_c: rests, reported once
    CHECK(!t.update(1, 85));
    CHECK_EQ(t.live_mask(0b11), 0b01u);
    CHECK(!t.update(1, 76));                // cooler, not yet cool: still resting
    CHECK_EQ(t.live_mask(0b11), 0b01u);
    CHECK(t.update(1, 72));                 // back: reported once
    CHECK_EQ(t.live_mask(0b11), 0b11u);
    CHECK(!t.update(1, 79));                // under hot_c again: stays in
    CHECK_EQ(t.live_mask(0b11), 0b11u);
    CHECK(!t.update(0, 99));                // the primary is never rested
    CHECK_EQ(t.live_mask(0b11), 0b11u);
}

// The engine end: with no sources declared, nothing about a request changes and
// IoStats carries no per-source rows -- the "off by default is off" check.
// Track D6: the keep-alive scheduler (storage/source_router.h).
//
// The D5 hypothesis was that the mirror's high mean latency is a POWER state --
// few bytes, so it idles, so every burst pays a wake-up, so it is offered even
// fewer bytes. The keep-alive is the proposed way out: one 4 KiB read whenever
// the source has been quiet for `idle_ms`.
//
// The dangerous half of that idea is the half this test pins. A poke issued
// while a real request is in flight buys NOTHING -- the drive is already awake,
// by definition -- and costs a queue slot in the backend queue that the P0
// burst is competing for. So "never while a real request is in flight" is not
// an optimisation, it is the difference between a keep-alive and a regression.
//
// The mutation (tests/mutate.py, `keepalive_never_races_a_real_request`):
//
//     if (s.real_inflight != 0) return false;   // correct
//     if (false)               return false;    // mutant: pokes into real work
//
// Everything else in this test passes under that mutant; only the third block
// notices, which is the point of writing it as its own block.
CACHEDMOE_TEST(io, keepalive_never_races_a_real_request) {
    const int64_t ms = 1000000;
    const int64_t window = 15 * ms;

    KeepAliveState s;
    s.has_file = true;
    s.last_activity_ns = 0;

    // --- the ordinary case: quiet for long enough, so poke ------------------
    CHECK(!keepalive_due(s, 14 * ms, window));   // not yet
    CHECK(keepalive_due(s, 15 * ms, window));    // exactly the window
    CHECK(keepalive_due(s, 900 * ms, window));   // long past it

    // --- off is off ---------------------------------------------------------
    // `CACHEDMOE_MIRROR_KEEPALIVE_MS=0` has to be byte-for-byte the old engine,
    // so the disabled path answers false however idle the source is.
    CHECK(!keepalive_due(s, 900 * ms, 0));
    CHECK(!keepalive_due(s, 900 * ms, -1));

    // --- THE MUTATION TARGET: a real request in flight forbids the poke -----
    // One outstanding read is enough. The drive is awake; there is nothing to
    // keep alive and a slot to lose.
    {
        KeepAliveState busy = s;
        busy.real_inflight = 1;
        CHECK(!keepalive_due(busy, 900 * ms, window));
        busy.real_inflight = 24;
        CHECK(!keepalive_due(busy, 3600 * ms, window));
        // ... and the moment the last one retires it is allowed again.
        busy.real_inflight = 0;
        CHECK(keepalive_due(busy, 900 * ms, window));
    }

    // --- at most one of ours outstanding per source -------------------------
    // A poke against a drive that has stopped answering must not turn into a
    // queue of pokes: without this, an enclosure that hangs would collect one
    // 4 KiB read every 15 ms for the rest of the run.
    {
        KeepAliveState mine = s;
        mine.probe_inflight = true;
        CHECK(!keepalive_due(mine, 3600 * ms, window));
    }

    // --- a source with nothing to read, and a dropped one, are never poked --
    {
        KeepAliveState nofile = s;
        nofile.has_file = false;
        CHECK(!keepalive_due(nofile, 900 * ms, window));

        KeepAliveState gone = s;
        gone.dropped = true;      // SourceHealth took it out of the router
        CHECK(!keepalive_due(gone, 900 * ms, window));
    }
}

// Track D6: the keep-alive's counters are its own, and the bytes it moves are
// not the run's bytes. A poke that landed in `bytes_completed` would inflate
// `eff GB/s` and, worse, show up in the per-source byte split that every A/B
// table in docs/p4_dual_source.md is read from.
CACHEDMOE_TEST(io, keepalive_is_off_without_a_second_source) {
    const std::vector<std::byte> content = pattern_bytes(1u << 20);
    IoEngine engine;
    IoConfig cfg;
    REQUIRE_OK(engine.start(std::make_unique<test::FakeBackend>(content, 8), cfg));

    // One root is the ordinary single-drive run: there is no mirror to keep
    // awake, and the primary -- the drive the decode is already hammering -- is
    // never poked. `CACHEDMOE_MIRROR_KEEPALIVE_MS` cannot change that, which is
    // what makes the no-mirror path still byte-for-byte what it was.
    engine.set_sources({"/models"}, {4.8});
    CHECK(!engine.mirrors_enabled());
    CHECK_EQ(engine.keepalive_ms(), 0u);
    CHECK_EQ(engine.keepalive_reads(0), 0u);
    CHECK_EQ(engine.keepalive_reads(1), 0u);

    const IoStats st = engine.stats();
    CHECK_EQ(st.bytes_completed, uint64_t{0});
    CHECK_EQ(st.chunks_submitted, uint64_t{0});
    engine.stop();
}

CACHEDMOE_TEST(io, no_mirror_leaves_the_stats_untouched) {
    auto scratch = make_scratch("d2_nomirror", 4u << 20, false);
    REQUIRE(scratch.has_value());
    const std::vector<std::byte> content = pattern_bytes(4u << 20);
    IoEngine engine;
    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    auto backend = std::make_unique<test::FakeBackend>(content, 8);
    test::FakeBackend* fake = backend.get();
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    CHECK(!engine.mirrors_enabled());
    CHECK_EQ(engine.source_count(), 0u);
    AlignedBuffer b(64 * 1024);
    IoRequest r;
    r.priority = IoPriority::BlockingMiss;
    r.file     = &scratch->file;
    r.bytes    = 64 * 1024;
    r.dst      = b.data();
    REQUIRE_OK(engine.submit(r, [](const IoResult&) {}));
    fake->release_all();
    engine.drain();
    CHECK(engine.stats().sources.empty());
    engine.stop();
}

// Linux port: the external USB4 drive drops its link about once an hour, and
// the reads in flight at that moment used to reach the caller as failures --
// an expert fill that failed, a decode that died -- before SourceHealth had
// counted enough errors to take the drive out of the router. A read the mirror
// cannot serve is now re-read from the primary: the caller sees the right
// bytes, the mirror's error and failover counters say what happened, and a
// failure on the PRIMARY is still reported (one retry, never a loop).
CACHEDMOE_TEST(io, mirror_error_is_reread_from_the_primary) {
    auto prim = make_scratch("lx_failover_p", 4u << 20, false);
    auto mirr = make_scratch("lx_failover_m", 4u << 20, false);
    REQUIRE(prim.has_value());
    REQUIRE(mirr.has_value());
    const std::vector<std::byte> content = pattern_bytes(4u << 20);
    IoEngine engine;
    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    auto backend = std::make_unique<test::FakeBackend>(content, 8);
    test::FakeBackend* fake = backend.get();
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    // The mirror weighted so heavily that an idle router always picks it.
    engine.set_sources({"primary", "mirror"}, {0.001, 1000.0});
    REQUIRE_OK(engine.add_mirror(&prim->file, 1, &mirr->file));
    REQUIRE(engine.mirrors_enabled());
    engine.set_stripe(false);   // the whole-request path; striped is below
    fake->fail_file(&mirr->file);

    constexpr uint32_t kBytes = 256 * 1024;   // four chunks: a multi-chunk run
    AlignedBuffer b(kBytes);
    std::memset(b.data(), 0, kBytes);
    IoRequest r;
    r.priority = IoPriority::BlockingMiss;
    r.file     = &prim->file;
    r.file_off = 64 * 1024;
    r.bytes    = kBytes;
    r.dst      = b.data();
    auto fut = engine.submit_future(r);
    REQUIRE(fut.has_value());
    const IoResult res = fut->get();
    CHECK(res.ok());
    CHECK_EQ(res.bytes_moved, uint64_t(kBytes));
    CHECK(std::memcmp(b.data(), content.data() + r.file_off, kBytes) == 0);
    engine.drain();
    {
        const IoStats st = engine.stats();
        REQUIRE_EQ(st.sources.size(), size_t(2));
        CHECK_EQ(st.sources[1].errors, uint64_t(1));
        CHECK_EQ(st.sources[1].failovers, uint64_t(1));
        CHECK_EQ(st.requests_failed, uint64_t(0));
        CHECK_EQ(st.requests_completed, uint64_t(1));
    }

    // Both drives failing is still an error the caller sees -- one retry on
    // the primary, not a loop between the two.
    fake->fail_file(&prim->file);
    auto fut2 = engine.submit_future(r);
    REQUIRE(fut2.has_value());
    CHECK(!fut2->get().ok());
    engine.drain();
    engine.stop();
}

// File::reopen keeps the handle's value and points it at whatever the path
// names now -- a new inode, the way a remounted drive's file is one -- and
// refuses, leaving the handle as it was, when that is not the same length.
CACHEDMOE_TEST(io, file_reopen_keeps_the_handle_and_reads_the_new_file) {
#if !defined(_WIN32)
    constexpr uint64_t kBytes = 64 * 1024;
    auto a = make_scratch("reopen", kBytes, false);
    REQUIRE(a.has_value());
    AlignedBuffer page(kPageSize);
    auto fill = [&](File& f, uint8_t v) {
        std::memset(page.data(), v, kPageSize);
        return f.write_at(0, ByteSpan(page.data(), kPageSize));
    };
    auto replace = [&](uint64_t bytes, uint8_t v) -> bool {
        if (!remove_file(a->path)) return false;
        auto f = File::open(a->path, FileFlags::Create | FileFlags::Write);
        return f && f->set_size(bytes) && fill(*f, v);
    };
    auto first_byte = [&]() -> int {
        auto n = a->file.read_at(0, MutBytes(page.data(), kPageSize));
        return n ? std::to_integer<int>(page.data()[0]) : -1;
    };
    REQUIRE_OK(fill(a->file, 0x11));
    const auto handle = a->file.native();

    REQUIRE(replace(kBytes, 0x22));
    CHECK_EQ(first_byte(), 0x11);                 // still the old, unlinked file
    REQUIRE_OK(a->file.reopen());
    CHECK_EQ(a->file.native(), handle);
    CHECK_EQ(first_byte(), 0x22);

    REQUIRE(replace(2 * kBytes, 0x33));
    CHECK_ERR(a->file.reopen(), Err::Corrupt);
    CHECK_EQ(first_byte(), 0x22);
#endif
}

// The USB4 box drops off the bus about once an hour and systemd remounts it
// seconds later, but the handles opened before the drop read EIO forever. A
// dropped mirror comes back through readmit_source: each of its handles is
// re-pointed at a fresh open of its path -- the same handle value, so the
// mirror table needs no change -- and it gets its whole error budget back. A
// mirror whose file is gone stays out. (File::reopen is POSIX-only.)
CACHEDMOE_TEST(io, dropped_mirror_is_readmitted_when_it_answers_again) {
#if !defined(_WIN32)
    auto prim = make_scratch("readmit_p", 2u << 20, false);
    auto mirr = make_scratch("readmit_m", 2u << 20, false);
    REQUIRE(prim.has_value());
    REQUIRE(mirr.has_value());
    const std::vector<std::byte> content = pattern_bytes(2u << 20);
    IoEngine engine;
    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    auto backend = std::make_unique<test::FakeBackend>(content, 8);
    test::FakeBackend* fake = backend.get();
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    engine.set_sources({"primary", "mirror"}, {0.001, 1000.0});   // idle, it picks the mirror
    REQUIRE_OK(engine.add_mirror(&prim->file, 1, &mirr->file));
    engine.set_stripe(false);

    AlignedBuffer b(64 * 1024);
    IoRequest r;
    r.priority = IoPriority::BlockingMiss;
    r.file     = &prim->file;
    r.bytes    = 64 * 1024;
    r.dst      = b.data();
    auto read_once = [&] {
        auto fut = engine.submit_future(r);
        const bool ok = fut && fut->get().ok();
        engine.drain();
        return ok;
    };

    CHECK_ERR(engine.readmit_source(1), Err::FailedPrecondition);   // not dropped
    fake->fail_file(&mirr->file);
    for (int i = 0; i < 3; ++i) CHECK(read_once());   // each one re-read from the primary
    REQUIRE(engine.source_dropped(1));

    // A mirror whose probe page is not the primary's stays out.
    AlignedBuffer page(kPageSize);
    std::memset(page.data(), 0x5a, kPageSize);
    REQUIRE_OK(mirr->file.write_at(1u << 20, ByteSpan(page.data(), kPageSize)));
    CHECK_ERR(engine.readmit_source(1), Err::Corrupt);
    std::memset(page.data(), 0, kPageSize);
    REQUIRE_OK(mirr->file.write_at(1u << 20, ByteSpan(page.data(), kPageSize)));

    // The file answers again (readmit_source reads it directly, not through
    // the fake): same handle value, back in the router, and with its whole
    // budget -- the first error after it is a failover, not a drop.
    const auto handle = mirr->file.native();
    REQUIRE_OK(engine.readmit_source(1));
    CHECK_EQ(mirr->file.native(), handle);
    CHECK(!engine.source_dropped(1));
    CHECK(!engine.stats().sources[1].dropped);
    CHECK_EQ(engine.stats().sources[1].readmits, 1u);
    CHECK(read_once());
    CHECK(!engine.source_dropped(1));
    fake->clear_failures();
    CHECK(read_once());
    CHECK(fake->log().back().file == &mirr->file);

    // A read from before a drop can still come back failed later; while one
    // is out the mirror stays dropped, so it cannot spend the new budget.
    fake->hold_completions(true);
    auto held = engine.submit_future(r);
    REQUIRE(held.has_value());
    engine.drop_source(1);
    CHECK_ERR(engine.readmit_source(1), Err::FailedPrecondition);
    fake->release_all();
    CHECK(held->get().ok());
    engine.drain();
    REQUIRE_OK(engine.readmit_source(1));

    // A mirror whose file is gone stays out.
    fake->fail_file(&mirr->file);
    for (int i = 0; i < 3; ++i) CHECK(read_once());
    REQUIRE(engine.source_dropped(1));
    REQUIRE_OK(remove_file(mirr->path));
    CHECK_ERR(engine.readmit_source(1), Err::Io);
    CHECK(engine.source_dropped(1));
    engine.stop();
#endif
}

// Track ST: with striping on, ONE P0 request's chunks are routed one by one,
// so a single expert run is read from both drives in proportion to their rates
// instead of from whichever drive won the whole request. The weights are the
// Linux box's probe (4.87 : 3.69 GB/s); 16 equal chunks on idle drives split
// greedily by `(outstanding + chunk) / rate`, which lands 9 : 7.
//
// Mutation this pins: routing the request whole (the pre-ST behaviour) puts
// all 16 chunks on one handle, and `per_file[1] == 0`.
CACHEDMOE_TEST(io, stripe_splits_one_p0_across_both_sources_by_weight) {
    auto prim = make_scratch("st_split_p", 2u << 20, false);
    auto mirr = make_scratch("st_split_m", 2u << 20, false);
    REQUIRE(prim.has_value());
    REQUIRE(mirr.has_value());
    const std::vector<std::byte> content = pattern_bytes(2u << 20);
    IoEngine engine;
    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    auto backend = std::make_unique<test::FakeBackend>(content, 8);
    test::FakeBackend* fake = backend.get();
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    engine.set_sources({"primary", "mirror"}, {4.87, 3.69});
    REQUIRE_OK(engine.add_mirror(&prim->file, 1, &mirr->file));
    REQUIRE(engine.mirrors_enabled());
    engine.set_stripe(true);
    REQUIRE(engine.stripe());

    constexpr uint32_t kBytes = 1u << 20;     // sixteen 64 KiB chunks
    AlignedBuffer b(kBytes);
    std::memset(b.data(), 0, kBytes);
    IoRequest r;
    r.priority = IoPriority::BlockingMiss;
    r.file     = &prim->file;
    r.file_off = 128 * 1024;
    r.bytes    = kBytes;
    r.dst      = b.data();
    auto fut = engine.submit_future(r);
    REQUIRE(fut.has_value());
    const IoResult res = fut->get();
    CHECK(res.ok());
    CHECK_EQ(res.bytes_moved, uint64_t(kBytes));
    CHECK(std::memcmp(b.data(), content.data() + r.file_off, kBytes) == 0);
    engine.drain();

    uint32_t per_file[2] = {0, 0};
    for (const auto& rec : fake->log()) {
        if (rec.file == &prim->file) ++per_file[0];
        else if (rec.file == &mirr->file) ++per_file[1];
    }
    CHECK_EQ(per_file[0] + per_file[1], 16u);
    CHECK_EQ(per_file[0], 9u);
    CHECK_EQ(per_file[1], 7u);

    const IoStats st = engine.stats();
    REQUIRE_EQ(st.sources.size(), size_t(2));
    CHECK_EQ(st.sources[0].stripe_chunks, uint64_t(9));
    CHECK_EQ(st.sources[1].stripe_chunks, uint64_t(7));
    CHECK_EQ(st.sources[0].bytes + st.sources[1].bytes, uint64_t(kBytes));
    CHECK_EQ(st.sources[1].bytes, uint64_t(7 * 64 * 1024));
    // Both shares count as one request on each drive, and every charge -- per
    // chunk on the way in, per chunk on the way out -- has been given back.
    CHECK_EQ(st.sources[0].p0_requests, uint64_t(1));
    CHECK_EQ(st.sources[1].p0_requests, uint64_t(1));
    CHECK_EQ(st.sources[0].outstanding_bytes, uint64_t(0));
    CHECK_EQ(st.sources[1].outstanding_bytes, uint64_t(0));
    CHECK_EQ(st.sources[0].inflight_requests, 0u);
    CHECK_EQ(st.sources[1].inflight_requests, 0u);
    CHECK_EQ(st.requests_completed, uint64_t(1));
    engine.stop();
}

// Track ST: striping is the default with a mirror and never without one, and
// it is a P0 policy -- the backfill keeps whole-request routing, and with
// striping off (CACHEDMOE_MIRROR_STRIPE=0) a P0 does too.
CACHEDMOE_TEST(io, stripe_is_the_default_and_leaves_backfill_whole) {
    auto prim = make_scratch("st_whole_p", 2u << 20, false);
    auto mirr = make_scratch("st_whole_m", 2u << 20, false);
    REQUIRE(prim.has_value());
    REQUIRE(mirr.has_value());
    const std::vector<std::byte> content = pattern_bytes(2u << 20);
    constexpr uint32_t kBytes = 1u << 20;
    AlignedBuffer b(kBytes);
    const char* env = ::cachedmoe::environment::get("CACHEDMOE_MIRROR_STRIPE");
    const bool stripe_off_env = env && *env == '0';
    {
        IoEngine engine;
        REQUIRE_OK(engine.start(std::make_unique<test::FakeBackend>(content, 8), IoConfig{}));
        engine.set_sources({"primary"}, {4.87});
        CHECK(!engine.stripe());
        engine.set_stripe(true);
        CHECK(!engine.stripe());
        engine.stop();
    }

    // Which drives one 1 MiB request (sixteen chunks) read from, and how many
    // of its chunks were routed one by one. `stripe` < 0 leaves the default.
    struct Use { uint32_t files = 99; uint64_t stripe_chunks = 0; };
    auto use = [&](int stripe, IoPriority pr) -> Use {
        IoEngine engine;
        IoConfig cfg;
        cfg.chunk_bytes = 64 * 1024;
        auto backend = std::make_unique<test::FakeBackend>(content, 8);
        test::FakeBackend* fake = backend.get();
        if (!engine.start(std::move(backend), cfg)) return {};
        engine.set_sources({"primary", "mirror"}, {4.87, 3.69});
        if (!engine.add_mirror(&prim->file, 1, &mirr->file)) return {};
        if (stripe >= 0) engine.set_stripe(stripe != 0);
        IoRequest r;
        r.priority = pr;
        r.file     = &prim->file;
        r.bytes    = kBytes;
        r.dst      = b.data();
        auto fut = engine.submit_future(r);
        if (!fut || !fut->get().ok()) return {};
        engine.drain();
        bool seen[2] = {false, false};
        for (const auto& rec : fake->log()) {
            if (rec.file == &prim->file) seen[0] = true;
            if (rec.file == &mirr->file) seen[1] = true;
        }
        const IoStats st = engine.stats();
        engine.stop();
        return {uint32_t(seen[0]) + uint32_t(seen[1]),
                st.sources[0].stripe_chunks + st.sources[1].stripe_chunks};
    };
    if (!stripe_off_env) {
        const Use d = use(-1, IoPriority::BlockingMiss);
        CHECK_EQ(d.files, 2u);
        CHECK_EQ(d.stripe_chunks, uint64_t(16));
    }
    const Use off = use(0, IoPriority::BlockingMiss);
    CHECK_EQ(off.files, 1u);
    CHECK_EQ(off.stripe_chunks, uint64_t(0));
    const Use backfill = use(1, IoPriority::Backfill);
    CHECK_EQ(backfill.files, 1u);
    CHECK_EQ(backfill.stripe_chunks, uint64_t(0));
}

// Track ST: the USB4 drive's link drops about once an hour. A striped request
// whose MIRROR share fails is re-read whole from the primary -- the caller gets
// the right bytes, one failover is counted against the mirror, and no charge is
// left behind (a leaked charge would make the router shun a drive forever).
// A failure on the PRIMARY's own share is still an error the caller sees.
CACHEDMOE_TEST(io, striped_mirror_error_is_reread_from_the_primary) {
    auto prim = make_scratch("st_fail_p", 2u << 20, false);
    auto mirr = make_scratch("st_fail_m", 2u << 20, false);
    REQUIRE(prim.has_value());
    REQUIRE(mirr.has_value());
    const std::vector<std::byte> content = pattern_bytes(2u << 20);
    IoEngine engine;
    IoConfig cfg;
    cfg.chunk_bytes = 64 * 1024;
    auto backend = std::make_unique<test::FakeBackend>(content, 8);
    test::FakeBackend* fake = backend.get();
    REQUIRE_OK(engine.start(std::move(backend), cfg));
    engine.set_sources({"primary", "mirror"}, {4.87, 3.69});
    REQUIRE_OK(engine.add_mirror(&prim->file, 1, &mirr->file));
    engine.set_stripe(true);
    fake->fail_file(&mirr->file);

    constexpr uint32_t kBytes = 1u << 20;
    AlignedBuffer b(kBytes);
    std::memset(b.data(), 0, kBytes);
    IoRequest r;
    r.priority = IoPriority::BlockingMiss;
    r.file     = &prim->file;
    r.file_off = 64 * 1024;
    r.bytes    = kBytes;
    r.dst      = b.data();
    auto fut = engine.submit_future(r);
    REQUIRE(fut.has_value());
    const IoResult res = fut->get();
    CHECK(res.ok());
    CHECK_EQ(res.bytes_moved, uint64_t(kBytes));
    CHECK(std::memcmp(b.data(), content.data() + r.file_off, kBytes) == 0);
    engine.drain();
    {
        const IoStats st = engine.stats();
        REQUIRE_EQ(st.sources.size(), size_t(2));
        CHECK_EQ(st.sources[1].errors, uint64_t(1));
        CHECK_EQ(st.sources[1].failovers, uint64_t(1));
        CHECK_EQ(st.sources[0].errors, uint64_t(0));
        CHECK_EQ(st.sources[0].outstanding_bytes, uint64_t(0));
        CHECK_EQ(st.sources[1].outstanding_bytes, uint64_t(0));
        CHECK_EQ(st.sources[0].inflight_requests, 0u);
        CHECK_EQ(st.sources[1].inflight_requests, 0u);
        CHECK_EQ(st.requests_failed, uint64_t(0));
        CHECK_EQ(st.requests_completed, uint64_t(1));
    }

    // Now the primary is the one that is gone and the mirror is healthy: the
    // primary's share fails, nothing is retried, and the caller is told.
    engine.set_sources({"primary", "mirror"}, {4.87, 3.69});
    REQUIRE_OK(engine.add_mirror(&prim->file, 1, &mirr->file));
    engine.set_stripe(true);
    fake->clear_failures();
    fake->fail_file(&prim->file);
    auto fut2 = engine.submit_future(r);
    REQUIRE(fut2.has_value());
    CHECK(!fut2->get().ok());
    engine.drain();
    {
        const IoStats st = engine.stats();
        CHECK_EQ(st.sources[0].errors, uint64_t(1));
        CHECK_EQ(st.sources[1].failovers, uint64_t(0));
        CHECK_EQ(st.sources[0].outstanding_bytes, uint64_t(0));
        CHECK_EQ(st.sources[1].outstanding_bytes, uint64_t(0));
    }
    engine.stop();
}

// Track D5: the startup probe must measure the drive, not its wake-up.
//
// A USB4 NVMe enclosure that has been idle answers its first read in about a
// second. The probe's window was 1 s, so the whole window was that one read:
// E: reported 0.03 GB/s against a real 3.77, the weighted router handed it
// 0.0% of the bytes, and the second read source bought nothing at all
// (docs/p4_dual_source.md §9.2). The fix is a warmup whose bytes are NOT
// counted -- the baseline is taken after it, not before.
//
// The mutation this pins is `base` going away:
//
//     const uint64_t base = moved.load();   // correct
//     const uint64_t base = 0;              // mutant: warmup bytes counted
//
// With a warmup three times the measurement window, the mutant reports roughly
// four times the rate, because it divides ~4 windows of bytes by 1 window of
// time. A local scratch file has no wake-up, so the two calls below measure the
// same steady rate and must agree; only the mutant makes them diverge.
CACHEDMOE_TEST(io, probe_does_not_count_its_warmup) {
    // Eight 4 MiB blocks is probe_source_gbps's own minimum.
    const uint64_t kBytes = 64u << 20;
    auto scratch = make_scratch("d5_probe", kBytes, false);
    REQUIRE(scratch.has_value());
    {
        const std::vector<std::byte> content = pattern_bytes(1u << 20);
        for (uint64_t off = 0; off < kBytes; off += content.size())
            REQUIRE_OK(scratch->file.write_at(off, ByteSpan(content.data(), content.size())));
    }
    scratch->file = File{};  // close before the probe opens its own handle

    // Same file, same steady rate; only the warmup differs.
    auto cold = IoEngine::probe_source_gbps(scratch->path, 200, 4, 0);
    REQUIRE_OK(cold);
    auto warm = IoEngine::probe_source_gbps(scratch->path, 200, 4, 600);
    REQUIRE_OK(warm);
    CHECK(*cold > 0.0);
    CHECK(*warm > 0.0);
    // Generous: this is a wall-clock measurement on a shared machine. The
    // mutant lands at ~4x, which is nowhere near inside this band.
    CHECK(*warm < *cold * 2.5);
    CHECK(*cold < *warm * 2.5);

    (void)remove_file(scratch->path);
}
