#include "runtime/trace.h"

#include <cstring>
#include <format>

namespace deepmoe::trace {
namespace {

constexpr char kMagic[8] = {'D', 'M', 'T', 'R', 'A', 'C', 'E', '1'};

void put_u16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void put_u32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
void put_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i));
}
uint16_t get_u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8); }
uint32_t get_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i);
    return v;
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
    return v;
}

}  // namespace

const char* cls_name(Cls c) {
    switch (c) {
        case Cls::Attention: return "attention";
        case Cls::Ced:       return "ced";
        case Cls::Moe:       return "moe";
        case Cls::Engram:    return "engram";
        case Cls::Tail:      return "tail";
        case Cls::Other:     return "other";
        case Cls::Count_:    break;
    }
    return "?";
}

void encode_record(const Record& r, uint8_t* out) {
    put_u32(out + 0, r.token);
    put_u32(out + 4, r.seq);
    put_u16(out + 8, r.layer);
    put_u16(out + 10, r.stage);
    out[12] = r.cls;
    out[13] = r.flags;
    put_u16(out + 14, r.submit);
    put_u64(out + 16, r.begin_ns);
    put_u64(out + 24, r.end_ns);
}

Record decode_record(const uint8_t* in) {
    Record r;
    r.token    = get_u32(in + 0);
    r.seq      = get_u32(in + 4);
    r.layer    = get_u16(in + 8);
    r.stage    = get_u16(in + 10);
    r.cls      = in[12];
    r.flags    = in[13];
    r.submit   = get_u16(in + 14);
    r.begin_ns = get_u64(in + 16);
    r.end_ns   = get_u64(in + 24);
    return r;
}

void encode_header(const Header& h, uint8_t* out) {
    std::memcpy(out, kMagic, 8);
    put_u32(out + 8, h.version);
    put_u32(out + 12, h.record_bytes);
    uint64_t bits = 0;
    std::memcpy(&bits, &h.timestamp_period_ns, 8);
    put_u64(out + 16, bits);
    put_u32(out + 24, h.record_count);
    put_u32(out + 28, h.name_count);
}

Result<Header> decode_header(const uint8_t* in, size_t n) {
    if (n < 32) return fail(Err::Corrupt, "trace: file shorter than its 32-byte header");
    if (std::memcmp(in, kMagic, 8) != 0)
        return fail(Err::Corrupt, "trace: bad magic, this is not a DMTRACE1 file");
    Header h;
    h.version      = get_u32(in + 8);
    h.record_bytes = get_u32(in + 12);
    const uint64_t bits = get_u64(in + 16);
    std::memcpy(&h.timestamp_period_ns, &bits, 8);
    h.record_count = get_u32(in + 24);
    h.name_count   = get_u32(in + 28);
    if (h.version != 1)
        return fail(Err::Corrupt, std::format("trace: version {}, this build reads 1", h.version));
    if (h.record_bytes != uint32_t(sizeof(Record)))
        return fail(Err::Corrupt,
                    std::format("trace: {}-byte records, this build reads {}", h.record_bytes,
                                sizeof(Record)));
    return h;
}

Result<void> Tracer::open(const std::string& path) {
    close();
#if defined(_WIN32)
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0 || !f)
#else
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
#endif
        return fail(Err::Io, std::format("trace: cannot open {} for writing", path));
    sink_  = f;
    path_  = path;
    header_.record_count = 0;
    header_.name_count   = 0;
    seq_    = 0;
    submit_ = 0;
    dropped_ = 0;
    pending_.clear();
    names_.clear();
    uint8_t hdr[32];
    encode_header(header_, hdr);
    if (std::fwrite(hdr, 1, sizeof hdr, sink_) != sizeof hdr) {
        close();
        return fail(Err::Io, std::format("trace: cannot write {}'s header", path));
    }
    return {};
}

Result<void> Tracer::flush_header() {
    if (!sink_) return {};
    uint8_t hdr[32];
    header_.name_count = static_cast<uint32_t>(names_.size());
    encode_header(header_, hdr);
    if (std::fseek(sink_, 0, SEEK_SET) != 0) return fail(Err::Io, "trace: seek to header failed");
    if (std::fwrite(hdr, 1, sizeof hdr, sink_) != sizeof hdr)
        return fail(Err::Io, "trace: rewriting the header failed");
    return {};
}

void Tracer::close() {
    if (!sink_) return;
    // Names go after the records, so the header's counts are the only thing
    // that has to be patched.
    (void)std::fseek(sink_, 0, SEEK_END);
    for (const auto& [key, name] : names_) {
        const uint8_t  cls   = uint8_t(key >> 16);
        const uint16_t stage = uint16_t(key & 0xffffu);
        const uint8_t  len   = uint8_t(name.size() > 255 ? 255 : name.size());
        uint8_t hdr[4] = {cls, 0, 0, len};
        put_u16(hdr + 1, stage);
        (void)std::fwrite(hdr, 1, 4, sink_);
        (void)std::fwrite(name.data(), 1, len, sink_);
    }
    (void)flush_header();
    (void)std::fclose(sink_);
    sink_ = nullptr;
    pending_.clear();
}

void Tracer::token_begin(uint32_t token) {
    token_  = token;
    seq_    = 0;
    submit_ = 0;
    pending_.clear();
}

uint32_t Tracer::open_dispatch(uint16_t layer, uint8_t cls, uint16_t stage, const char* name) {
    if (!sink_) return kNoDispatch;
    Pending p;
    p.rec.token  = token_;
    p.rec.seq    = seq_++;
    p.rec.layer  = layer;
    p.rec.stage  = stage;
    p.rec.cls    = cls;
    p.rec.submit = submit_;
    p.q_begin    = stamp_ ? stamp_(stamp_ctx_) : ~0u;
    if (p.q_begin == ~0u) {
        p.rec.flags |= kFlagNoBegin;
        ++dropped_;
    }
    if (name && *name) {
        const uint32_t key = (uint32_t(cls) << 16) | stage;
        if (!names_.count(key)) names_.emplace(key, name);
    }
    pending_.push_back(p);
    return static_cast<uint32_t>(pending_.size() - 1);
}

void Tracer::close_dispatch(uint32_t handle) {
    if (!sink_ || handle >= pending_.size()) return;
    Pending& p = pending_[handle];
    p.q_end = stamp_ ? stamp_(stamp_ctx_) : ~0u;
    if (p.q_end == ~0u) {
        p.rec.flags |= kFlagNoEnd;
        ++dropped_;
    }
}

void Tracer::token_end(const uint64_t* ticks, uint32_t n, uint32_t first) {
    if (!sink_) return;
    const uint32_t bits = valid_bits_;
    const uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const double   ns   = header_.timestamp_period_ns;

    auto tick_at = [&](uint32_t slot, uint64_t& out) -> bool {
        if (!ticks || slot == ~0u || slot < first || slot - first >= n) return false;
        out = ticks[slot - first] & mask;
        return true;
    };

    uint8_t buf[32];
    for (Pending& p : pending_) {
        uint64_t a = 0, b = 0;
        const bool ha = tick_at(p.q_begin, a);
        const bool hb = tick_at(p.q_end, b);
        if (!ha) p.rec.flags |= kFlagNoBegin;
        if (!hb) p.rec.flags |= kFlagNoEnd;
        if (ha && hb) {
            // Both stamps are bottom-of-pipe on the same queue, so `b < a`
            // means the counter wrapped inside its valid bits.
            if (b < a) {
                p.rec.flags |= kFlagWrapped;
                b = b + (mask - a) + 1;
                a = 0;
            }
            p.rec.begin_ns = uint64_t(double(a) * ns);
            p.rec.end_ns   = uint64_t(double(b) * ns);
        } else if (ha) {
            p.rec.begin_ns = uint64_t(double(a) * ns);
        } else if (hb) {
            p.rec.end_ns = uint64_t(double(b) * ns);
        }
        encode_record(p.rec, buf);
        if (std::fwrite(buf, 1, sizeof buf, sink_) != sizeof buf) break;
        ++header_.record_count;
    }
    pending_.clear();
}

Result<File> read_file(const std::string& path) {
#if defined(_WIN32)
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || !f)
#else
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
#endif
        return fail(Err::NotFound, std::format("trace: cannot open {}", path));

    std::vector<uint8_t> all;
    uint8_t chunk[65536];
    for (;;) {
        const size_t got = std::fread(chunk, 1, sizeof chunk, f);
        if (got == 0) break;
        all.insert(all.end(), chunk, chunk + got);
    }
    (void)std::fclose(f);

    auto h = decode_header(all.data(), all.size());
    if (!h) return std::unexpected(h.error());

    File out;
    out.header = *h;
    const size_t need = 32 + size_t(h->record_count) * h->record_bytes;
    if (all.size() < need)
        return fail(Err::Corrupt,
                    std::format("trace: header claims {} records ({} bytes) but the file is {}",
                                h->record_count, need, all.size()));
    out.records.reserve(h->record_count);
    for (uint32_t i = 0; i < h->record_count; ++i)
        out.records.push_back(decode_record(all.data() + 32 + size_t(i) * h->record_bytes));

    size_t off = need;
    for (uint32_t i = 0; i < h->name_count; ++i) {
        if (off + 4 > all.size())
            return fail(Err::Corrupt, "trace: the name table is truncated");
        const uint8_t  cls   = all[off];
        const uint16_t stage = get_u16(all.data() + off + 1);
        const uint8_t  len   = all[off + 3];
        off += 4;
        if (off + len > all.size())
            return fail(Err::Corrupt, "trace: a name runs past the end of the file");
        out.names.emplace((uint32_t(cls) << 16) | stage,
                          std::string(reinterpret_cast<const char*>(all.data() + off), len));
        off += len;
    }
    return out;
}

}  // namespace deepmoe::trace
