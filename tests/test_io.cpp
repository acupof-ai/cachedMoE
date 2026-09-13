// storage/: chunk planning, priority ordering and preemption against a fake
// backend (design §9.6), plus one real unbuffered round trip through the
// platform backend (IOCP on Windows, io_uring on Linux).
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "model/layout.h"
#include "storage/backend.h"
#include "storage/file.h"
#include "storage/io_engine.h"
#include "tests/fake_backend.h"
#include "tests/test_framework.h"

using namespace deepmoe;
using namespace deepmoe::storage;

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

DEEPMOE_TEST(io, chunk_planning) {
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

DEEPMOE_TEST(io, chunk_planning_edge_cases) {
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

DEEPMOE_TEST(io, fake_backend_round_trip) {
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
    CHECK_EQ(st.requests_submitted, 1u);
    CHECK_EQ(st.requests_completed, 1u);
    CHECK_EQ(st.bytes_completed, req.bytes);
    CHECK_EQ(st.per_priority_requests[0], 1u);
    CHECK(st.busy_ns > 0);
    engine.stop();
}

DEEPMOE_TEST(io, priority_ordering_and_preemption) {
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

DEEPMOE_TEST(io, submit_validation_and_cancel) {
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

DEEPMOE_TEST(io, platform_backend_reads_a_real_unbuffered_file) {
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

DEEPMOE_TEST(io, opening_a_missing_file_fails_cleanly) {
    auto f = File::open_read(temp_dir() + "/deepmoe_does_not_exist_9f3a.bin", true);
    CHECK(!f);
    CHECK_EQ(f.error().code, Err::Io);
    CHECK(f.error().os_code != 0);
    File closed;
    CHECK(!closed.is_open());
    AlignedBuffer b(4096);
    CHECK_ERR(closed.read_at(0, MutBytes(b.data(), 4096)), Err::FailedPrecondition);
}
