// Track R2's KV store (runtime/kvstore.h, docs/p4_kv_ux.md): the byte
// accounting, the packed forms, and the store itself -- planes on the kv
// sources only, growth, the window floor, pack / unpack and the row backup a
// replay restores from.
//
// `kvstore.*` needs nothing; `gpu.kvstore_*` needs a Vulkan device, no weights.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cpu/dequant.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "model/v41_config.h"
#include "runtime/kvcache.h"
#include "runtime/kvstore.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using runtime::KvStoreConfig;

namespace {

std::string cfg_path() { return std::string(DEEPMOE_TEST_DATA_DIR) + "/v41_config.json"; }

const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_round(float v) {   // nearest E2M1, ties to the even code
    const float a = std::fabs(std::min(6.0f, std::max(-6.0f, v)));
    int best = 0;
    for (int i = 1; i < 8; ++i) {
        const float d = std::fabs(kMag[i] - a), db = std::fabs(kMag[best] - a);
        if (d < db || (d == db && (i % 2) == 0)) best = i;
    }
    return v < 0 ? -kMag[best] : kMag[best];
}

// fp4_act_quant(x, block, scale_dtype=E4M3 | E8M0), the reference's arithmetic
// on the host: one block's scale, the rounded values, bf16 out.
void quantise_row(const float* x, uint32_t n, uint32_t block, bool e8m0, uint16_t* out) {
    for (uint32_t b0 = 0; b0 < n; b0 += block) {
        float amax = 0;
        for (uint32_t i = 0; i < block; ++i) amax = std::max(amax, std::fabs(x[b0 + i]));
        amax = std::max(amax, 6.0f * 1.5625e-2f * 0.125f);   // a floor, as the kernels have
        float sc;
        if (e8m0) {
            sc = std::pow(2.0f, std::ceil(std::log2(amax / 6.0f)));
        } else {
            sc = cpu::fp8_e4m3_to_float(cpu::fp8_encode_rn(std::min(amax / 6.0f, 448.0f)));
        }
        for (uint32_t i = 0; i < block; ++i)
            out[b0 + i] = cpu::float_to_bf16(fp4_round(x[b0 + i] / sc) * sc);
    }
}

bool skip_without_gpu(gpu::Device& dev) {
    if (auto r = dev.create(); !r) {
        std::printf("       SKIP gpu: kvstore needs a Vulkan device (%s)\n", r.error().str().c_str());
        return true;
    }
    return false;
}

}  // namespace

DEEPMOE_TEST(kvstore, accounting_per_source_planes) {
    auto c = V41Config::load(cfg_path());
    REQUIRE_OK(c);
    const TextConfig& t = c->text;
    const KvStoreConfig k = KvStoreConfig::for_model(t, 65536);
    // model.py: compressor, compressed cache and index keys on kv_source_layer_ids
    // only; every other compressed layer reads the last source at or above it.
    CHECK_EQ(k.owners().size(), size_t(4));
    CHECK_EQ(k.owner(0), KvStoreConfig::kNoPlane);
    CHECK_EQ(k.owner(1), KvStoreConfig::kNoPlane);
    CHECK_EQ(k.owner(2), 2u);
    CHECK_EQ(k.owner(7), 2u);
    CHECK_EQ(k.owner(13), 8u);
    CHECK_EQ(k.owner(19), 14u);
    CHECK_EQ(k.owner(21), 20u);
    CHECK_EQ(k.owner(39), 20u);
    CHECK_EQ(k.plane_ratio(8), 2u);
    CHECK_EQ(k.plane_ratio(20), 1u);

    // docs/p4_kv_ux.md §1: bf16 planes 2.5 rows a token x (1,024 + 256) B.
    for (uint32_t ctx : {4096u, 17010u, 65536u}) {
        const uint64_t rows = 3ull * ((ctx + 1) / 2) + ctx;
        CHECK_EQ(k.compressed_bytes(ctx), rows * 1024);
        CHECK_EQ(k.index_key_bytes(ctx), rows * 256);
        // the model's packed formats: 288 + 68 B a row, plus three 2x512 fp32 carries x2
        CHECK_EQ(k.packed_bytes(ctx), (3ull * (ctx / 2) + ctx) * (288 + 68) + 3ull * 2 * 512 * 8);
        std::printf("    %6u positions: allocated %.1f MB (window %.2f, compressed %.2f, keys %.2f, "
                    "state %.2f, top-k %.2f), packed non-SWA %.2f MB\n", ctx,
                    k.total_bytes(ctx) / 1e6, k.window_bytes() / 1e6, k.compressed_bytes(ctx) / 1e6,
                    k.index_key_bytes(ctx) / 1e6, k.cmp_state_bytes() / 1e6, k.topk_bytes() / 1e6,
                    k.packed_bytes(ctx) / 1e6);
    }
    CHECK(k.total_bytes(65536) < 220'000'000ull);
    CHECK(k.total_bytes(17010) < 60'000'000ull);
    // Track Q's layout for comparison: every layer a plane of every row.
    KvStoreConfig old;
    old.layers = 40;
    old.max_context = 17010 + 64;
    CHECK(old.total_bytes(17074) > 870'000'000ull);
}

DEEPMOE_TEST(kvstore, fp4_packing_is_bit_exact) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    uint32_t rows_ok = 0, rows = 0;
    for (bool e8m0 : {false, true}) {
        const uint32_t n = e8m0 ? 128 : 512, block = e8m0 ? 32 : 16;
        std::vector<float> x(n);
        std::vector<uint16_t> q(n), back(n);
        std::vector<uint8_t> nib(n / 2), sc(n / block);
        for (int trial = 0; trial < 400; ++trial) {
            const float mag = std::pow(10.0f, float(trial % 9) - 4.0f);
            for (uint32_t i = 0; i < n; ++i) x[i] = nd(rng) * mag;
            if (trial % 7 == 0) for (uint32_t i = 0; i < block; ++i) x[i] = 0.0f;      // zero block
            if (trial % 11 == 0) x[3] = -0.0f;
            if (trial % 13 == 0) for (uint32_t i = 0; i < block; ++i) x[block + i] = 1e-7f * nd(rng);
            quantise_row(x.data(), n, block, e8m0, q.data());
            ++rows;
            if (!runtime::pack_fp4_row(q.data(), n, block, e8m0, nib.data(), sc.data())) continue;
            runtime::unpack_fp4_row(nib.data(), sc.data(), n, block, e8m0, back.data());
            if (std::memcmp(q.data(), back.data(), n * 2) == 0) ++rows_ok;
        }
    }
    std::printf("    %u of %u quantised rows packed and came back bit-identical\n", rows_ok, rows);
    CHECK_EQ(rows_ok, rows);

    // Off the grid: refused, not rounded.
    std::vector<uint16_t> off(512);
    for (uint32_t i = 0; i < 512; ++i) off[i] = cpu::float_to_bf16(0.123f * float(i + 1));
    std::vector<uint8_t> nib(256), sc(32);
    CHECK(!runtime::pack_fp4_row(off.data(), 512, 16, false, nib.data(), sc.data()));
}

DEEPMOE_TEST(kvcache, geometry_counts_index_keys_on_kv_sources) {
    auto c = V41Config::load(cfg_path());
    REQUIRE_OK(c);
    runtime::KvGeometry g;
    g.max_context = 65536;
    g.compressed_fp4 = true;
    // model.py Indexer.owns_k: four key caches, 2.5 rows a token x 68 B.
    CHECK_EQ(g.indexer_bytes(c->text), (3ull * 32768 + 65536) * 68);
}

DEEPMOE_TEST(gpu, kvstore_planes_growth_floor_and_packing) {
    gpu::Device dev;
    if (skip_without_gpu(dev)) return;
    gpu::MemoryAllocator alloc;
    REQUIRE_OK(alloc.init(dev, MemoryPath::DeviceLocalHostVisible));
    auto c = V41Config::load(cfg_path());
    REQUIRE_OK(c);
    const TextConfig& t = c->text;

    runtime::KvStore kv;
    REQUIRE_OK(kv.create(alloc, KvStoreConfig::for_model(t, 16384, 256)));
    CHECK_EQ(kv.capacity(), 256u);
    const uint64_t small = kv.bytes();

    // Reuse layers see their source's addresses.
    auto v2 = kv.layer(2), v5 = kv.layer(5), v20 = kv.layer(20), v33 = kv.layer(33), v0 = kv.layer(0);
    REQUIRE_OK(v2); REQUIRE_OK(v5); REQUIRE_OK(v20); REQUIRE_OK(v33); REQUIRE_OK(v0);
    CHECK_EQ(v2->cmp_kv, v5->cmp_kv);
    CHECK_EQ(v2->idx_key, v5->idx_key);
    CHECK_EQ(v2->cmp_state_kv, v5->cmp_state_kv);
    CHECK_EQ(v20->cmp_kv, v33->cmp_kv);
    CHECK(v2->cmp_kv != v20->cmp_kv);
    CHECK(v2->win_val != v5->win_val);
    CHECK(v2->top_idx != v5->top_idx);
    CHECK_EQ(v0->plane_owner, KvStoreConfig::kNoPlane);
    CHECK_EQ(v33->plane_owner, 20u);

    // Seed quantised rows on every source, as a context of 300 positions.
    const uint32_t P = 300;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    for (uint32_t L : {2u, 8u, 14u, 20u}) {
        const uint32_t r = t.compress_ratio(L), rows = P / r;
        std::vector<float> x(size_t(rows) * 512), k(size_t(rows) * 128);
        std::vector<uint16_t> q(512), qk(128);
        std::vector<float> xf(512), kf(128);
        for (uint32_t row = 0; row < rows; ++row) {
            for (auto& e : xf) e = nd(rng);
            for (auto& e : kf) e = nd(rng);
            quantise_row(xf.data(), 512, 16, false, q.data());
            quantise_row(kf.data(), 128, 32, true, qk.data());
            for (uint32_t i = 0; i < 512; ++i) x[size_t(row) * 512 + i] = cpu::bf16_to_float(q[i]);
            for (uint32_t i = 0; i < 128; ++i) k[size_t(row) * 128 + i] = cpu::bf16_to_float(qk[i]);
        }
        REQUIRE_OK(kv.seed_compressed(L, x.data(), rows));
        REQUIRE_OK(kv.seed_index_k(L, k.data(), rows));
        if (r > 1) {
            std::vector<float> skv(2 * 512), ssc(2 * 512, -std::numeric_limits<float>::infinity());
            for (auto& e : skv) e = nd(rng);
            for (uint32_t i = 0; i < 512; ++i) ssc[i] = nd(rng);
            REQUIRE_OK(kv.seed_cmp_state(L, skv.data(), ssc.data(), 2));
        }
    }
    CHECK(kv.capacity() >= P);
    CHECK(kv.bytes() > small);
    std::printf("    grew 256 -> %u positions: %.2f -> %.2f MB\n", kv.capacity(), small / 1e6, kv.bytes() / 1e6);

    // Grow again and check the rows survived the copy.
    auto before = kv.pack(P);
    REQUIRE_OK(before);
    CHECK_EQ(before->raw_rows(), 0u);
    REQUIRE_OK(kv.reserve(5000));
    CHECK(kv.capacity() >= 5000u);
    auto after = kv.pack(P);
    REQUIRE_OK(after);
    for (size_t i = 0; i < before->planes.size(); ++i) {
        CHECK(before->planes[i].cmp_fp4 == after->planes[i].cmp_fp4);
        CHECK(before->planes[i].key_scale == after->planes[i].key_scale);
        CHECK(before->planes[i].carry_score == after->planes[i].carry_score);
    }
    std::printf("    packed %u positions: %.3f MB (%u raw rows); bf16 live %.3f MB\n", P,
                before->bytes() / 1e6, before->raw_rows(),
                (kv.config().compressed_bytes(P) + kv.config().index_key_bytes(P)) / 1e6);

    // Unpack into a cleared store: the planes come back bit-identical.
    std::vector<uint16_t> ref20(size_t(P) * 512);
    std::memcpy(ref20.data(), kv.layer(20)->cmp_kv_host, ref20.size() * 2);
    std::vector<uint16_t> ref8k(size_t(P / 2) * 128);
    std::memcpy(ref8k.data(), kv.layer(8)->idx_key_host, ref8k.size() * 2);
    std::vector<float> carry14(1024);
    std::memcpy(carry14.data(), kv.layer(14)->cmp_state_kv_host, 1024 * 4);
    kv.clear();
    CHECK_EQ(kv.layer(20)->cmp_kv_host[5], uint16_t(0));
    REQUIRE_OK(kv.unpack(*after));
    CHECK(std::memcmp(ref20.data(), kv.layer(33)->cmp_kv_host, ref20.size() * 2) == 0);
    CHECK(std::memcmp(ref8k.data(), kv.layer(12)->idx_key_host, ref8k.size() * 2) == 0);
    CHECK(std::memcmp(carry14.data(), kv.layer(14)->cmp_state_kv_host, 1024 * 4) == 0);
    CHECK_EQ(kv.layer(25)->n_cmp, P);
    CHECK_EQ(kv.layer(9)->n_cmp, P / 2);

    // Row backup: overwrite what positions [200, 300) complete, restore step by step.
    auto bk = kv.backup_rows(200, 300);
    REQUIRE_OK(bk);
    for (uint32_t L : {2u, 20u}) {
        auto v = kv.layer(L);
        const uint32_t r = t.compress_ratio(L);
        std::memset(v->cmp_kv_host + size_t(200 / r) * 512, 0xAB, size_t(100 / r) * 512 * 2);
    }
    for (uint32_t p = 200; p < 300; ++p) REQUIRE_OK(kv.restore_rows(*bk, p));
    CHECK(std::memcmp(ref20.data(), kv.layer(20)->cmp_kv_host, ref20.size() * 2) == 0);
    auto repacked = kv.pack(P);
    REQUIRE_OK(repacked);
    CHECK(repacked->planes[0].cmp_fp4 == after->planes[0].cmp_fp4);

    // The window floor: at position 300 with the floor at 250, entries for
    // positions 173..249 are -1 and 250..300 name their ring slots.
    kv.set_window_floor(250);
    REQUIRE_OK(kv.set_decode_topk(0, 300, 0, 0));
    auto v0b = kv.layer(0);
    const auto* top = reinterpret_cast<const int32_t*>(v0b->top_idx_host);
    uint32_t live = 0;
    bool right = true;
    for (uint32_t i = 0; i < 128; ++i) {
        const int64_t pos = 300 - int64_t(127 - i);
        if (pos < 250) right = right && top[i] == -1;
        else { right = right && top[i] == int32_t(pos % 128); ++live; }
    }
    CHECK(right);
    CHECK_EQ(live, 51u);
    kv.set_window_floor(0);
    REQUIRE_OK(kv.set_decode_topk(0, 300, 0, 0));
    CHECK(reinterpret_cast<const int32_t*>(kv.layer(0)->top_idx_host)[0] == int32_t(173 % 128));

    // Past the limit: refused.
    CHECK(!kv.reserve(16385).has_value());
    kv.destroy();
    alloc.shutdown();
}
