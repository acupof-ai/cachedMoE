// Per-dispatch GPU trace (Track W).
//
// The per-token profiler (core/profiler.h) answers "where did this token's
// milliseconds go" at seven buckets. It cannot answer "which of the 16-29
// dispatches of layer 20 was slow, and how much of the layer is barrier tails
// rather than kernels". That second question is what every optimisation in
// docs/STATUS.md §2 was attributed by, and today it is answered by five
// hand-placed timestamp pairs (`Engine::ts_attn_` and friends) that bracket
// whole phases.
//
// This file makes the brackets per dispatch. A `Tracer` is a pure host-side
// bookkeeper: it holds no Vulkan handle. The GPU stamp is a callback the owner
// binds (`Engine::cmd_stamp`, which writes into `Engine::tsq_` and returns the
// query slot, or ~0u when the pool is full or no buffer is open). After the
// token's fence the owner hands the raw ticks back and the tracer resolves
// every pending record into nanoseconds and appends it to the sink.
//
// That split is deliberate: everything in this file compiles and is tested
// without a GPU (tests/test_trace.cpp round-trips the format and resolves a
// synthetic token), and the only thing GPU validation adds is whether the
// query slots carry sensible ticks.
//
// Opt in with `deepmoe run --trace FILE` / `deepmoe serve --trace FILE`. Off,
// `Tracer::enabled()` is false and every hook is a null test.
//
// FILE FORMAT (little-endian, version 1) -- also documented in
// tools/trace_timeline.py, which is the reader:
//
//   header, 32 bytes
//     0   char[8]  "DMTRACE1"
//     8   u32      version          = 1
//     12  u32      record_bytes     = 32
//     16  f64      timestamp_period_ns   (VkPhysicalDeviceLimits::timestampPeriod)
//     24  u32      record_count     (patched on close)
//     28  u32      name_count       (patched on close)
//
//   records, record_count x 32 bytes, in the order the dispatches were
//   RECORDED into command buffers (which is program order, not completion
//   order):
//     0   u32      token        the decode position this dispatch belongs to
//     4   u32      seq          index of this dispatch within the token
//     8   u16      layer        0..num_layers-1, or kNoLayer for the tail
//     10  u16      stage        the stage enumerator within `cls`
//     12  u8       cls          Cls: 0 attention, 1 ced, 2 moe, 3 engram,
//                               4 tail, 5 other
//     13  u8       flags        Flags below
//     14  u16      submit       which submit of the token carried it
//     16  u64      begin_ns     0 when Flags::NoBegin
//     24  u64      end_ns       0 when Flags::NoEnd
//
//   names, name_count entries, immediately after the records:
//     u8 cls, u16 stage, u8 len, char[len]      (no terminator)
//
//   Both stamps are bottom-of-pipe. Dispatches are barrier-separated, so a
//   record's `begin` is "everything before this dispatch has completed" and
//   `end` is "this dispatch has completed": `end - begin` is the dispatch's
//   busy time and `begin - previous end` is the barrier/gap in front of it.
//   A stamp pair whose two ticks are equal is reported as 0 busy rather than
//   dropped -- an unwritten slot is flagged instead (Flags::NoBegin/NoEnd).

#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "core/status.h"

namespace deepmoe::trace {

inline constexpr uint32_t kNoDispatch = ~0u;
inline constexpr uint16_t kNoLayer    = 0xffffu;
inline constexpr uint32_t kMagicWords = 2;  // "DMTRACE1" is two u32s

// The phase a dispatch belongs to. Kept deliberately coarse: the point of the
// class is to let tools/trace_timeline.py roll a token up by phase without
// knowing any stage enumerator, while `stage` + the name table carry the
// detail.
enum class Cls : uint8_t {
    Attention = 0,  // dispatches 1-9 of design §7.14
    Ced       = 1,  // design §7.4's compressor and indexer, source layers only
    Moe       = 2,  // gate/up, h-quant, down
    Engram    = 3,  // design §7.10
    Tail      = 4,  // collapse, head, argmax / sample_topk
    Other     = 5,
    Count_,
};

const char* cls_name(Cls c);

enum Flags : uint8_t {
    kFlagNone    = 0,
    kFlagNoBegin = 1u << 0,  // the begin stamp did not get a query slot
    kFlagNoEnd   = 1u << 1,  // the end stamp did not get a query slot
    kFlagWrapped = 1u << 2,  // the tick counter wrapped between the two stamps
};

// One resolved dispatch. 32 bytes, and `static_assert`ed to stay that way:
// the reader indexes the file by `record_bytes` from the header, so a change
// here that is not mirrored in tools/trace_timeline.py would silently
// misparse. tests/test_trace.cpp pins the field offsets.
struct Record {
    uint32_t token    = 0;
    uint32_t seq      = 0;
    uint16_t layer    = kNoLayer;
    uint16_t stage    = 0;
    uint8_t  cls      = static_cast<uint8_t>(Cls::Other);
    uint8_t  flags    = kFlagNone;
    uint16_t submit   = 0;
    uint64_t begin_ns = 0;
    uint64_t end_ns   = 0;

    uint64_t busy_ns() const { return end_ns >= begin_ns ? end_ns - begin_ns : 0; }
};
static_assert(sizeof(Record) == 32, "the on-disk record is 32 bytes");

// Serialisation, exposed so the round-trip test can use exactly what the
// writer uses. `out` must have room for 32 bytes.
void encode_record(const Record& r, uint8_t* out);
Record decode_record(const uint8_t* in);

struct Header {
    uint32_t version              = 1;
    uint32_t record_bytes         = static_cast<uint32_t>(sizeof(Record));
    double   timestamp_period_ns  = 1.0;
    uint32_t record_count         = 0;
    uint32_t name_count           = 0;
};

void   encode_header(const Header& h, uint8_t* out);  // 32 bytes
Result<Header> decode_header(const uint8_t* in, size_t n);

class Tracer {
public:
    // Returns a GPU query slot, or ~0u when none is available. The Engine
    // binds `Engine::cmd_stamp`.
    using StampFn = uint32_t (*)(void* ctx);

    Tracer() = default;
    ~Tracer() { close(); }

    Tracer(const Tracer&)            = delete;
    Tracer& operator=(const Tracer&) = delete;

    Result<void> open(const std::string& path);
    void         close();
    bool         enabled() const { return sink_ != nullptr; }

    void bind_stamp(StampFn fn, void* ctx) { stamp_ = fn; stamp_ctx_ = ctx; }
    void set_period_ns(double ns) { header_.timestamp_period_ns = ns; }
    void set_valid_bits(uint32_t bits) { valid_bits_ = bits; }

    // A token's worth of dispatches. `token_begin` drops anything left
    // pending from a token whose fence never resolved.
    void token_begin(uint32_t token);

    // Stamps the begin of a dispatch and returns a handle for `close_dispatch`,
    // or kNoDispatch when tracing is off. `name` must outlive the tracer (a
    // string literal or the `*_stage_name` tables); it is recorded once per
    // (cls, stage) pair.
    uint32_t open_dispatch(uint16_t layer, uint8_t cls, uint16_t stage, const char* name);
    void     close_dispatch(uint32_t handle);

    // Call where the owner submits; the submit index is stamped onto every
    // record opened afterwards.
    void note_submit() { ++submit_; }

    // After the token's fence: `ticks` is the raw query pool prefix the owner
    // read, `first` the slot `ticks[0]` corresponds to. Resolves every pending
    // record and appends it to the sink.
    void token_end(const uint64_t* ticks, uint32_t n, uint32_t first = 0);

    uint32_t written() const { return header_.record_count; }
    // How many dispatches of the current token lost a stamp because the query
    // pool ran out. Non-zero means the pool is too small for the trace.
    uint32_t dropped_stamps() const { return dropped_; }

    // How many query slots a token of `layers` layers needs, worst case:
    // every layer's 16 attention dispatches plus 13 ced ones plus 3 MoE and
    // 2 engram, plus a 6-dispatch tail, two stamps each.
    static uint32_t suggested_pool(uint32_t layers) { return 2 * (layers * 34 + 8) + 64; }

private:
    struct Pending {
        Record   rec;
        uint32_t q_begin = ~0u;
        uint32_t q_end   = ~0u;
    };

    Result<void> flush_header();

    FILE*                 sink_ = nullptr;
    std::string           path_;
    Header                header_{};
    StampFn               stamp_     = nullptr;
    void*                 stamp_ctx_ = nullptr;
    uint32_t              valid_bits_ = 64;
    uint32_t              token_      = 0;
    uint32_t              seq_        = 0;
    uint16_t              submit_     = 0;
    uint32_t              dropped_    = 0;
    std::vector<Pending>  pending_;
    // (cls << 16 | stage) -> name, written as the file's name table on close.
    std::map<uint32_t, std::string> names_;
};

// The two hooks callers use, so a call site is one line and null-safe.
inline uint32_t open_dispatch(Tracer* t, uint16_t layer, Cls c, uint16_t stage,
                              const char* name) {
    return t ? t->open_dispatch(layer, static_cast<uint8_t>(c), stage, name) : kNoDispatch;
}
inline void close_dispatch(Tracer* t, uint32_t handle) {
    if (t && handle != kNoDispatch) t->close_dispatch(handle);
}

// Reader, used by tests and by anything that wants the file in C++.
struct File {
    Header              header;
    std::vector<Record> records;
    std::map<uint32_t, std::string> names;
};
Result<File> read_file(const std::string& path);

}  // namespace deepmoe::trace
