// tests/data/l1_layer<L>_expert<E>.bin -- the oracle L1 golden vectors of
// design §12, written by `tools/oracle.py --level l1`.
//
// Shared by tests/test_integration.cpp (CPU path: IoEngine -> ExpertStore ->
// cpu/gemv_fp4_ref) and tests/test_gpu_moe.cpp (GPU path: the same slot read by
// the design §7.9 kernels). Both compare against the same torch fp32 answer, so
// the two paths are compared to each other by transitivity.
//
// Ownership/threading: plain value types and pure functions.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/status.h"
#include "model/manifest.h"

namespace deepmoe::testing {

// The same 64-bit block checksum tools/oracle.py computes: SplitMix64 over
// index-offset 64-bit words, XOR-accumulated, with the byte length folded into
// the seed. Not cryptographic -- the oracle records a SHA-256 per part as well.
// This exists so a test can say "these are the oracle's bytes" without linking
// a crypto library into deepmoe_tests.
inline uint64_t block_hash64(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0xCBF29CE484222325ull ^ static_cast<uint64_t>(bytes);
    const size_t whole = bytes / 8;
    uint64_t index = 0;
    auto mix = [](uint64_t x) {
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
        x ^= x >> 27; x *= 0x94D049BB133111EBull;
        x ^= x >> 31;
        return x;
    };
    for (size_t i = 0; i < whole; ++i) {
        uint64_t w;
        std::memcpy(&w, p + i * 8, 8);
        h ^= mix(w + 0x9E3779B97F4A7C15ull * (++index));
    }
    if (const size_t tail = bytes % 8; tail) {
        uint64_t w = 0;
        std::memcpy(&w, p + whole * 8, tail);   // zero-padded, as the oracle does
        h ^= mix(w + 0x9E3779B97F4A7C15ull * (++index));
    }
    return h;
}

struct Golden {
    uint32_t layer = 0, expert = 0, dim = 0, inter = 0;
    uint64_t seed = 0;
    float    swiglu_limit = 0.0f;
    uint32_t slot_bytes = 0;
    uint64_t part_offset[kExpertPartCount]{};
    uint64_t part_bytes[kExpertPartCount]{};
    uint64_t part_hash[kExpertPartCount]{};
    std::vector<float> x, y;
};

inline Result<Golden> load_golden(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::Io, "cannot open " + path);
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};

    char magic[4] = {};
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "DML1", 4) != 0)
        return fail(Err::Corrupt, path + ": not a DML1 file");
    uint32_t head[5] = {};
    uint64_t seed = 0;
    if (std::fread(head, sizeof(uint32_t), 5, f) != 5 ||
        std::fread(&seed, sizeof(uint64_t), 1, f) != 1)
        return fail(Err::Corrupt, path + ": truncated header");
    if (head[0] != 1) return fail(Err::Corrupt, path + ": unsupported version");

    Golden g;
    g.layer = head[1]; g.expert = head[2]; g.dim = head[3]; g.inter = head[4];
    g.seed = seed;
    if (std::fread(&g.swiglu_limit, sizeof(float), 1, f) != 1 ||
        std::fread(&g.slot_bytes, sizeof(uint32_t), 1, f) != 1)
        return fail(Err::Corrupt, path + ": truncated header");
    for (uint8_t i = 0; i < kExpertPartCount; ++i) {
        uint64_t triple[3] = {};
        if (std::fread(triple, sizeof(uint64_t), 3, f) != 3)
            return fail(Err::Corrupt, path + ": truncated part table");
        g.part_offset[i] = triple[0];
        g.part_bytes[i]  = triple[1];
        g.part_hash[i]   = triple[2];
    }
    g.x.resize(g.dim);
    g.y.resize(g.dim);
    if (std::fread(g.x.data(), sizeof(float), g.dim, f) != g.dim ||
        std::fread(g.y.data(), sizeof(float), g.dim, f) != g.dim)
        return fail(Err::Corrupt, path + ": truncated vectors");
    return g;
}

// How two fp32 vectors differ, in the three ways that mean different things.
struct Compare {
    double cosine = 0.0;
    double max_abs = 0.0;        // max |got - want|
    double rel_to_scale = 0.0;   // max_abs / max |want|
    double worst_elem_rel = 0.0; // max |got-want| / |want| over significant elements
    double norm_got = 0.0, norm_want = 0.0;
};

inline Compare compare(const std::vector<float>& got, const std::vector<float>& want) {
    Compare c;
    double dot = 0.0, ymax = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        c.norm_got  += double(got[i]) * got[i];
        c.norm_want += double(want[i]) * want[i];
        dot         += double(got[i]) * want[i];
        ymax = std::fmax(ymax, std::fabs(double(want[i])));
        c.max_abs = std::fmax(c.max_abs, std::fabs(double(got[i]) - want[i]));
    }
    // Only elements within four decades of the vector's scale carry a
    // meaningful elementwise ratio; below that the denominator is noise.
    const double floor = ymax * 1e-4;
    for (size_t i = 0; i < want.size(); ++i)
        if (std::fabs(double(want[i])) >= floor)
            c.worst_elem_rel = std::fmax(c.worst_elem_rel,
                                         std::fabs(double(got[i]) - want[i]) / std::fabs(double(want[i])));
    c.norm_got  = std::sqrt(c.norm_got);
    c.norm_want = std::sqrt(c.norm_want);
    c.cosine = (c.norm_got > 0 && c.norm_want > 0) ? dot / (c.norm_got * c.norm_want) : 0.0;
    c.rel_to_scale = ymax > 0 ? c.max_abs / ymax : c.max_abs;
    return c;
}

// DEEPMOE_MODEL_DIR, or null when the checkpoint is not on this machine.
inline const char* model_dir() {
#if defined(_MSC_VER)
    static std::string v;
    char* buf = nullptr; size_t n = 0;
    if (_dupenv_s(&buf, &n, "DEEPMOE_MODEL_DIR") == 0 && buf) { v = buf; free(buf); return v.c_str(); }
    return nullptr;
#else
    return std::getenv("DEEPMOE_MODEL_DIR");
#endif
}

// A skip is a pass, but it must be loud: ctest keys its SKIP_REGULAR_EXPRESSION
// off the phrase "set DEEPMOE_MODEL_DIR".
inline bool skip_without_model(const char* what) {
    if (model_dir()) return false;
    std::printf("       SKIP %s: set DEEPMOE_MODEL_DIR to the checkpoint "
                "(e.g. D:\\models\\DeepSeek-V4.1-Flash) to run it\n", what);
    return true;
}

}  // namespace deepmoe::testing
