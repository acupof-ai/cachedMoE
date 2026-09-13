// End-to-end check against the real 510 GB checkpoint (design §5.1 v0.5, §12).
//
// Everything else in tests/ runs on fixtures. This file is the one place where
// the whole read path is exercised on the bytes that actually matter:
//
//   deepmoe_manifest.json -> ShardSet -> IoEngine (IOCP, FILE_FLAG_NO_BUFFERING)
//     -> ExpertStore slot -> pointer table -> cpu/dequant + cpu/gemv
//
// and the answer is compared against tools/oracle.py, which read the same
// tensors through the `safetensors` library and computed the FFN in torch fp32.
// If the manifest's run/skew arithmetic is wrong by one byte, this is the test
// that says so.
//
// It needs the checkpoint, so it is gated on DEEPMOE_MODEL_DIR:
//
//   ctest --test-dir build                       # skips, and says why
//   DEEPMOE_MODEL_DIR=D:\models\DeepSeek-V4.1-Flash ctest --test-dir build
//
// A skip is a pass. A machine that has the weights must not be able to leave
// this test silently unrun, so the skip prints a line naming the variable.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "cpu/dequant.h"
#include "cpu/gemv_avx512.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/planner.h"
#include "store/shard_set.h"
#include "tests/l1_golden.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::store;
using namespace deepmoe::testing;

namespace {

std::string join(const std::string& dir, const char* name) {
    return ShardSet::join(dir, name);
}

std::string data_path(const std::string& name) {
    return std::string(DEEPMOE_TEST_DATA_DIR) + "/" + name;
}

// One routed expert's FFN, in fp32, straight out of an ExpertStore slot.
// y = w2( silu(clamp(w1 x, max=L)) * clamp(w3 x, -L, L) ), design §2.4.
Result<std::vector<float>> expert_ffn_from_slot(const std::byte* base,
                                                const ExpertEntry& e,
                                                const std::vector<float>& x,
                                                uint32_t dim, uint32_t inter,
                                                float limit) {
    auto span_of = [&](ExpertPart p) {
        return std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(base + e.offset_of(p)),
            static_cast<size_t>(e.bytes_of(p)));
    };

    std::vector<float> gate(inter), up(inter), h(inter), y(dim);
    cpu::GemvShape wide{inter, dim, 1};
    if (auto r = cpu::gemv_fp4_ref(span_of(ExpertPart::W1Weight),
                                   span_of(ExpertPart::W1Scale),
                                   x, wide, gate); !r) return std::unexpected(r.error());
    if (auto r = cpu::gemv_fp4_ref(span_of(ExpertPart::W3Weight),
                                   span_of(ExpertPart::W3Scale),
                                   x, wide, up); !r) return std::unexpected(r.error());
    for (uint32_t i = 0; i < inter; ++i) {
        const float g = std::fmin(gate[i], limit);
        const float u = std::fmin(std::fmax(up[i], -limit), limit);
        h[i] = (g / (1.0f + std::exp(-g))) * u;          // silu(g) * u
    }
    cpu::GemvShape narrow{dim, inter, 1};
    if (auto r = cpu::gemv_fp4_ref(span_of(ExpertPart::W2Weight),
                                   span_of(ExpertPart::W2Scale),
                                   h, narrow, y); !r) return std::unexpected(r.error());
    return y;
}

}  // namespace

// Fills (0, 0) through the real IoEngine and checks the slot byte for byte
// against the oracle, then runs the FFN out of the slot's pointers.
DEEPMOE_TEST(integration, fills_an_expert_slot_from_the_real_checkpoint) {
    if (skip_without_model("integration.fills_an_expert_slot_from_the_real_checkpoint")) return;
    const std::string dir = model_dir();

    auto golden = load_golden(data_path("l1_layer0_expert0.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;
    CHECK_EQ(g.slot_bytes, static_cast<uint32_t>(layout::kExpertSlotBytes));

    auto mf = Manifest::load(join(dir, layout::kManifestFile));
    REQUIRE_OK(mf);
    REQUIRE_OK(mf->validate());
    CHECK_EQ(mf->version(), 2u);
    CHECK_EQ(mf->files().size(), 48u);
    CHECK_EQ(mf->expert_slot_bytes(), layout::kExpertSlotBytes);

    ShardSet shards;
    REQUIRE_OK(shards.open_all(dir, *mf, /*unbuffered=*/true));

    auto entry = mf->require_expert(ExpertKey{static_cast<uint16_t>(g.layer),
                                              static_cast<uint16_t>(g.expert)});
    REQUIRE_OK(entry);
    const ExpertEntry& e = **entry;
    // The oracle laid the same slot out in Python; the two must agree on where
    // every part ends up, or the C++ pointer table is addressing the wrong bytes.
    for (uint8_t i = 0; i < kExpertPartCount; ++i) {
        CHECK_EQ(e.part_offset[i], g.part_offset[i]);
        CHECK_EQ(e.part_bytes[i], g.part_bytes[i]);
    }

    IoConfig io_cfg;
    io_cfg.unbuffered = true;
    auto backend = storage::make_default_backend(io_cfg);
    REQUIRE_OK(backend);
    storage::IoEngine io;
    REQUIRE_OK(io.start(std::move(*backend), io_cfg));
    CHECK_EQ(io.backend_caps().alignment, 4096u);

    CacheConfig cache;
    cache.slots_per_slab = 2;
    cache.budget_bytes   = 2 * layout::kExpertSlotBytes;
    ExpertStore store;
    REQUIRE_OK(store.init(std::make_unique<HostSlabBacking>(), cache,
                          layout::kTotalLogicalLayers, layout::kRoutedExperts));
    REQUIRE_EQ(store.slot_count(), 2u);

    Planner planner;
    REQUIRE_OK(planner.init(store, io, *mf, shards, cache, PrefetchConfig{}));

    const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    auto fetch = planner.fetch(key, IoPriority::BlockingMiss, /*token=*/1, /*deadline=*/0);
    REQUIRE_OK(fetch);
    // One IoRequest per run: two, for every expert in this checkpoint.
    CHECK_EQ(fetch->ids.size(), e.runs.size());
    CHECK_EQ(fetch->bytes, e.slot_bytes);
    io.drain();

    REQUIRE(store.resident(key));
    const auto st = store.stats();
    CHECK_EQ(st.fills_ok, 1u);
    CHECK_EQ(st.runs_started, static_cast<uint64_t>(e.runs.size()));
    CHECK_EQ(st.runs_done, static_cast<uint64_t>(e.runs.size()));

    auto addr = store.lookup(key, 1);
    REQUIRE(addr.has_value());
    const auto* base = static_cast<const std::byte*>(addr->host_ptr);
    REQUIRE(base != nullptr);

    // (b) the six parts, byte for byte, from where the pointer table says.
    for (uint8_t i = 0; i < kExpertPartCount; ++i) {
        const auto part = static_cast<ExpertPart>(i);
        auto entry_addr = store.table_entry(key, part);
        REQUIRE_OK(entry_addr);
        CHECK_EQ(*entry_addr, reinterpret_cast<uint64_t>(base) + e.offset_of(part));
        const uint64_t got = block_hash64(reinterpret_cast<const void*>(*entry_addr),
                                          static_cast<size_t>(e.bytes_of(part)));
        if (got != g.part_hash[i])
            _ctx.fail(__FILE__, __LINE__,
                      std::format("part '{}' hash {:#018x} != oracle {:#018x} "
                                  "(slot offset {}, {} B)",
                                  expert_part_name(part), got, g.part_hash[i],
                                  e.offset_of(part), e.bytes_of(part)));
    }

    // (c) dequant + FFN out of the slot, against the oracle's torch fp32 result.
    auto y = expert_ffn_from_slot(base, e, g.x, g.dim, g.inter, g.swiglu_limit);
    REQUIRE_OK(y);
    REQUIRE_EQ(y->size(), g.y.size());
    const Compare c = compare(*y, g.y);
    std::printf("       (%u, %u): ||y|| %.6f vs oracle %.6f, cos %.12f, "
                "max|dy| %.3e (%.3e of |y|max), worst elementwise rel %.3e\n",
                g.layer, g.expert, c.norm_got, c.norm_want, c.cosine,
                c.max_abs, c.rel_to_scale, c.worst_elem_rel);
    // design §12 L1: relative error <= 1e-3. Both sides are fp32 and differ only
    // in the order of a 2304- and a 5120-term sum, so 1e-4 is the bar here --
    // measured against the magnitude of the output, not element by element: at
    // an output element that is six orders of magnitude below the vector's
    // scale, an elementwise ratio measures fp32 cancellation, not correctness.
    CHECK(c.rel_to_scale <= 1e-4);
    CHECK(c.cosine >= 1.0 - 1e-9);

    io.stop();
}

// The same expert store, driven twice, at the other end of the model: the last
// routed expert of the last layer, whose weights run is the last thing in its
// shard and therefore the read that runs past EOF.
DEEPMOE_TEST(integration, fills_the_last_expert_of_the_last_layer) {
    if (skip_without_model("integration.fills_the_last_expert_of_the_last_layer")) return;
    const std::string dir = model_dir();

    auto golden = load_golden(data_path("l1_layer39_expert383.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;

    auto mf = Manifest::load(join(dir, layout::kManifestFile));
    REQUIRE_OK(mf);
    ShardSet shards;
    REQUIRE_OK(shards.open_all(dir, *mf, true));

    const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    auto entry = mf->require_expert(key);
    REQUIRE_OK(entry);
    const ExpertEntry& e = **entry;

    IoConfig io_cfg;
    auto backend = storage::make_default_backend(io_cfg);
    REQUIRE_OK(backend);
    storage::IoEngine io;
    REQUIRE_OK(io.start(std::move(*backend), io_cfg));

    CacheConfig cache;
    cache.slots_per_slab = 1;
    cache.budget_bytes   = layout::kExpertSlotBytes;
    ExpertStore store;
    REQUIRE_OK(store.init(std::make_unique<HostSlabBacking>(), cache,
                          layout::kTotalLogicalLayers, layout::kRoutedExperts));
    Planner planner;
    REQUIRE_OK(planner.init(store, io, *mf, shards, cache, PrefetchConfig{}));

    auto fetch = planner.fetch(key, IoPriority::BlockingMiss, 1, 39);
    REQUIRE_OK(fetch);
    io.drain();
    REQUIRE(store.resident(key));

    auto addr = store.lookup(key, 1);
    REQUIRE(addr.has_value());
    const auto* base = static_cast<const std::byte*>(addr->host_ptr);
    for (uint8_t i = 0; i < kExpertPartCount; ++i)
        CHECK_EQ(block_hash64(base + e.offset_of(static_cast<ExpertPart>(i)),
                              static_cast<size_t>(e.bytes_of(static_cast<ExpertPart>(i)))),
                 g.part_hash[i]);

    auto y = expert_ffn_from_slot(base, e, g.x, g.dim, g.inter, g.swiglu_limit);
    REQUIRE_OK(y);
    const Compare c = compare(*y, g.y);
    std::printf("       (%u, %u): cos %.12f, max|dy| %.3e (%.3e of |y|max), "
                "worst elementwise rel %.3e\n",
                g.layer, g.expert, c.cosine, c.max_abs, c.rel_to_scale, c.worst_elem_rel);
    CHECK(c.rel_to_scale <= 1e-4);
    CHECK(c.cosine >= 1.0 - 1e-9);

    io.stop();
}

// The reads the manifest asks for must be legal unbuffered reads of files that
// really exist -- including the 43 experts whose weights run is the last thing
// in its shard and therefore ends a few hundred bytes past EOF once widened to
// a sector boundary (storage/backend.h ChunkRequest::min_bytes).
DEEPMOE_TEST(integration, every_run_is_a_legal_unbuffered_read) {
    if (skip_without_model("integration.every_run_is_a_legal_unbuffered_read")) return;
    const std::string dir = model_dir();

    auto mf = Manifest::load(join(dir, layout::kManifestFile));
    REQUIRE_OK(mf);
    REQUIRE_OK(mf->validate());

    uint64_t runs = 0, experts = 0, past_eof = 0, payload = 0;
    uint64_t widest_slot = 0;
    uint32_t max_runs = 0;
    for (uint32_t layer = 0; layer < mf->expert_layers(); ++layer) {
        for (uint32_t id = 0; id < mf->experts_in_layer(layer); ++id) {
            const ExpertEntry* e = mf->expert(ExpertKey{static_cast<uint16_t>(layer),
                                                        static_cast<uint16_t>(id)});
            REQUIRE(e != nullptr);
            ++experts;
            widest_slot = std::max(widest_slot, e->slot_bytes);
            max_runs = std::max(max_runs, static_cast<uint32_t>(e->runs.size()));
            for (const Run& r : e->runs) {
                ++runs;
                CHECK(is_aligned(r.aligned_off));
                CHECK(is_aligned(r.aligned_bytes));
                CHECK(is_aligned(r.slot_offset));
                const FileEntry* f = mf->file(r.file);
                REQUIRE(f != nullptr);
                // The payload always ends inside the file; only the sector
                // padding may spill over, and by less than one sector.
                uint64_t last = 0;
                for (const RunPart& p : r.parts) {
                    last = std::max(last, r.aligned_off + p.skew + p.bytes);
                    payload += p.bytes;
                }
                CHECK(last <= f->bytes);
                if (r.aligned_off + r.aligned_bytes > f->bytes) {
                    ++past_eof;
                    CHECK(r.aligned_off + r.aligned_bytes - f->bytes < kPageSize);
                }
            }
        }
    }
    std::printf("       %llu experts, %llu runs (max %u per expert), "
                "%llu widened past EOF, widest slot %llu B\n",
                static_cast<unsigned long long>(experts),
                static_cast<unsigned long long>(runs), max_runs,
                static_cast<unsigned long long>(past_eof),
                static_cast<unsigned long long>(widest_slot));
    // 40 x 384 routed + 3 x 128 DSpark.
    CHECK_EQ(experts, 15744ull);
    CHECK_EQ(runs, 2ull * experts);
    CHECK_EQ(max_runs, 2u);
    CHECK_EQ(widest_slot, layout::kExpertSlotBytes);
    CHECK_EQ(payload, experts * layout::kExpertBytes);
    // One per shard that ends on an expert; the I/O layer has to tolerate these.
    CHECK(past_eof > 0);
    CHECK(past_eof <= mf->files().size());
}
