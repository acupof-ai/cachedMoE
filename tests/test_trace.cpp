// Track W: the per-dispatch trace format and the resolver, on the CPU.
//
// Nothing here needs a GPU. The tracer's only GPU coupling is the stamp
// callback, so these tests drive it with a fake one that hands out slots from
// a synthetic tick array -- which is also how a regression in the wrap or the
// missing-slot handling would show up on the real machine.
//
// What is pinned:
//   1. the 32-byte record layout, field by field, at the byte offsets
//      tools/trace_timeline.py unpacks;
//   2. a header/record/name round trip through a real file;
//   3. a synthetic token of 3 layers x 4 dispatches: the resolver's ns
//      arithmetic, the submit index, the dispatch order, and the busy/gap
//      decomposition the timeline tool prints;
//   4. a query pool that runs out mid-token (flags, not silence);
//   5. a tick counter that wraps inside its valid bits.

#include <cstdio>
#include <string>
#include <vector>

#include "runtime/trace.h"
#include "tests/test_framework.h"

namespace {

using namespace deepmoe;
using namespace deepmoe::trace;

std::string temp_path(const char* stem) {
#if defined(_WIN32)
    const char* base = std::getenv("TEMP");
    if (!base || !*base) base = ".";
#else
    const char* base = "/tmp";
#endif
    return std::string(base) + "/deepmoe_" + stem + ".dmtrace";
}

// A stamp callback over a fixed number of slots; slot i is `ticks[i]`.
struct FakeGpu {
    uint32_t              next  = 0;
    uint32_t              slots = 0;
    std::vector<uint64_t> ticks;

    static uint32_t stamp(void* ctx) {
        FakeGpu* g = static_cast<FakeGpu*>(ctx);
        if (g->next >= g->slots) return ~0u;
        return g->next++;
    }
};

}  // namespace

DEEPMOE_TEST(trace, record_layout_is_32_bytes_at_fixed_offsets) {
    REQUIRE_EQ(sizeof(Record), size_t(32));

    Record r;
    r.token    = 0x11223344u;
    r.seq      = 0x55667788u;
    r.layer    = 0x99aau;
    r.stage    = 0xbbccu;
    r.cls      = uint8_t(Cls::Ced);
    r.flags    = kFlagNoEnd | kFlagWrapped;
    r.submit   = 0xddeeu;
    r.begin_ns = 0x0102030405060708ull;
    r.end_ns   = 0x1112131415161718ull;

    uint8_t buf[32] = {};
    encode_record(r, buf);

    // Little-endian, at the offsets the Python reader's struct format uses.
    CHECK_EQ(uint32_t(buf[0]), 0x44u);
    CHECK_EQ(uint32_t(buf[3]), 0x11u);
    CHECK_EQ(uint32_t(buf[4]), 0x88u);
    CHECK_EQ(uint32_t(buf[8]), 0xaau);
    CHECK_EQ(uint32_t(buf[9]), 0x99u);
    CHECK_EQ(uint32_t(buf[10]), 0xccu);
    CHECK_EQ(uint32_t(buf[12]), uint32_t(Cls::Ced));
    CHECK_EQ(uint32_t(buf[13]), uint32_t(kFlagNoEnd | kFlagWrapped));
    CHECK_EQ(uint32_t(buf[14]), 0xeeu);
    CHECK_EQ(uint32_t(buf[16]), 0x08u);
    CHECK_EQ(uint32_t(buf[23]), 0x01u);
    CHECK_EQ(uint32_t(buf[24]), 0x18u);
    CHECK_EQ(uint32_t(buf[31]), 0x11u);

    const Record back = decode_record(buf);
    CHECK_EQ(back.token, r.token);
    CHECK_EQ(back.seq, r.seq);
    CHECK_EQ(back.layer, r.layer);
    CHECK_EQ(back.stage, r.stage);
    CHECK_EQ(back.cls, r.cls);
    CHECK_EQ(back.flags, r.flags);
    CHECK_EQ(back.submit, r.submit);
    CHECK_EQ(back.begin_ns, r.begin_ns);
    CHECK_EQ(back.end_ns, r.end_ns);
}

DEEPMOE_TEST(trace, header_round_trips_and_rejects_a_foreign_file) {
    Header h;
    h.timestamp_period_ns = 10.0;
    h.record_count        = 1234;
    h.name_count          = 7;
    uint8_t buf[32] = {};
    encode_header(h, buf);

    auto back = decode_header(buf, sizeof buf);
    REQUIRE(back.has_value());
    CHECK_EQ(back->version, 1u);
    CHECK_EQ(back->record_bytes, 32u);
    CHECK(deepmoe::test::close(back->timestamp_period_ns, 10.0));
    CHECK_EQ(back->record_count, 1234u);
    CHECK_EQ(back->name_count, 7u);

    uint8_t bad[32] = {};
    std::memcpy(bad, "NOTATRCE", 8);
    CHECK(!decode_header(bad, sizeof bad).has_value());
    CHECK(!decode_header(buf, 8).has_value());  // truncated

    // A version this build does not read must be refused rather than misparsed.
    uint8_t v2[32];
    std::memcpy(v2, buf, 32);
    v2[8] = 2;
    CHECK(!decode_header(v2, sizeof v2).has_value());
}

// One synthetic token: 3 layers, 4 dispatches each, one submit per layer.
// Slot i gets tick `100 * i`, and the period is 10 ns, so dispatch d of the
// token occupies [2000*d, 2000*d + 1000] ns and the gap to the previous one is
// 1000 ns. Those are the three numbers tools/trace_timeline.py prints.
DEEPMOE_TEST(trace, synthetic_token_resolves_to_busy_and_gap) {
    const std::string path = temp_path("synth");
    FakeGpu gpu;
    gpu.slots = 64;
    for (uint32_t i = 0; i < gpu.slots; ++i) gpu.ticks.push_back(100ull * i);

    Tracer t;
    REQUIRE(t.open(path).has_value());
    t.bind_stamp(&FakeGpu::stamp, &gpu);
    t.set_period_ns(10.0);
    t.set_valid_bits(64);

    t.token_begin(7);
    for (uint16_t L = 0; L < 3; ++L) {
        for (uint16_t d = 0; d < 4; ++d) {
            const uint32_t h = t.open_dispatch(L, uint8_t(Cls::Attention), d, "stage");
            t.close_dispatch(h);
        }
        t.note_submit();
    }
    t.token_end(gpu.ticks.data(), gpu.next, 0);
    CHECK_EQ(t.written(), 12u);
    CHECK_EQ(t.dropped_stamps(), 0u);
    t.close();

    auto f = read_file(path);
    REQUIRE(f.has_value());
    REQUIRE_EQ(f->records.size(), size_t(12));
    CHECK_EQ(f->header.record_count, 12u);
    // One name for the one (cls, stage) pair per stage: 4 distinct stages.
    CHECK_EQ(f->names.size(), size_t(4));

    for (uint32_t i = 0; i < 12; ++i) {
        const Record& r = f->records[i];
        CHECK_EQ(r.token, 7u);
        CHECK_EQ(r.seq, i);
        CHECK_EQ(r.layer, uint16_t(i / 4));
        CHECK_EQ(r.stage, uint16_t(i % 4));
        CHECK_EQ(r.cls, uint8_t(Cls::Attention));
        CHECK_EQ(r.flags, uint8_t(kFlagNone));
        // The submit counter advances at the end of a layer, so layer L's
        // dispatches all carry L.
        CHECK_EQ(r.submit, uint16_t(i / 4));
        CHECK_EQ(r.begin_ns, uint64_t(2000 * i));
        CHECK_EQ(r.end_ns, uint64_t(2000 * i + 1000));
        CHECK_EQ(r.busy_ns(), uint64_t(1000));
        if (i) CHECK_EQ(r.begin_ns - f->records[i - 1].end_ns, uint64_t(1000));
    }
    std::remove(path.c_str());
}

DEEPMOE_TEST(trace, a_pool_that_runs_out_is_flagged_not_silent) {
    const std::string path = temp_path("short");
    FakeGpu gpu;
    gpu.slots = 3;  // enough for one pair and one lone begin
    for (uint32_t i = 0; i < gpu.slots; ++i) gpu.ticks.push_back(100ull * i);

    Tracer t;
    REQUIRE(t.open(path).has_value());
    t.bind_stamp(&FakeGpu::stamp, &gpu);
    t.set_period_ns(1.0);
    t.token_begin(0);
    for (uint16_t d = 0; d < 3; ++d) t.close_dispatch(t.open_dispatch(0, 0, d, "s"));
    t.token_end(gpu.ticks.data(), gpu.next, 0);
    t.close();

    auto f = read_file(path);
    REQUIRE(f.has_value());
    REQUIRE_EQ(f->records.size(), size_t(3));
    CHECK_EQ(f->records[0].flags, uint8_t(kFlagNone));
    CHECK_EQ(f->records[1].flags, uint8_t(kFlagNoEnd));   // begin got slot 2, end did not
    CHECK_EQ(f->records[2].flags, uint8_t(kFlagNoBegin | kFlagNoEnd));
    // A record with no stamps must not claim time.
    CHECK_EQ(f->records[2].begin_ns, uint64_t(0));
    CHECK_EQ(f->records[2].end_ns, uint64_t(0));
    std::remove(path.c_str());
}

DEEPMOE_TEST(trace, a_wrapped_tick_counter_does_not_produce_a_negative_span) {
    const std::string path = temp_path("wrap");
    // 32 valid bits: begin near the top, end just past the wrap.
    FakeGpu gpu;
    gpu.slots = 2;
    gpu.ticks = {0xffffffffull - 9ull, 90ull};

    Tracer t;
    REQUIRE(t.open(path).has_value());
    t.bind_stamp(&FakeGpu::stamp, &gpu);
    t.set_period_ns(1.0);
    t.set_valid_bits(32);
    t.token_begin(0);
    t.close_dispatch(t.open_dispatch(0, 0, 0, "s"));
    t.token_end(gpu.ticks.data(), gpu.next, 0);
    t.close();

    auto f = read_file(path);
    REQUIRE(f.has_value());
    REQUIRE_EQ(f->records.size(), size_t(1));
    CHECK_EQ(f->records[0].flags, uint8_t(kFlagWrapped));
    CHECK_EQ(f->records[0].begin_ns, uint64_t(0));
    // 10 ticks to the wrap plus 90 after it.
    CHECK_EQ(f->records[0].busy_ns(), uint64_t(100));
    std::remove(path.c_str());
}

DEEPMOE_TEST(trace, a_tracer_with_no_sink_is_a_null_test) {
    Tracer t;
    CHECK(!t.enabled());
    CHECK_EQ(open_dispatch(&t, 0, Cls::Attention, 0, "s"), kNoDispatch);
    close_dispatch(&t, kNoDispatch);
    close_dispatch(nullptr, 0);
    CHECK_EQ(open_dispatch(nullptr, 0, Cls::Attention, 0, "s"), kNoDispatch);
    CHECK_EQ(t.written(), 0u);
    // The pool suggestion has to cover a 40-layer token's worst case: 40
    // layers x (16 attention + 13 ced + 3 MoE + 2 engram) x 2 stamps.
    CHECK(Tracer::suggested_pool(40) >= 2 * 40 * 34);
}
