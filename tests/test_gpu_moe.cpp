// The design §7.9 MoE kernels against the oracle, on the real checkpoint.
//
// This is the GPU half of what tests/test_integration.cpp does on the CPU, and
// it closes the loop design §12 L1 asks for: the same expert bytes, read off
// NVMe by the same IoEngine, land this time in *GPU-visible* slab memory
// (design §3.3 path A), are addressed through the buffer_device_address pointer
// table of design §5.3, and are consumed by gpu/shaders/moe_gateup.slang and
// moe_down.slang. The answer is compared with tools/oracle.py's torch fp32 y.
//
// What each case is really testing:
//   * that FILE_FLAG_NO_BUFFERING reads can land directly in mapped device
//     memory (the zero-copy claim of design §9.6);
//   * that the pointer table's 8-byte-aligned part addresses are legal loads on
//     gfx1151 (the cost of the no-repack decision of design §5.1 v0.5);
//   * that the FP4 nibble order, the E8M0 ldexp and the §2.4 clamps in the
//     shader agree with cpu/dequant.cpp;
//   * that every specialisation variant of design §7.1 computes the same thing;
//   * that the expert indirection list of design §7.9 selects the right expert.
//
// Gated on DEEPMOE_MODEL_DIR and on a working Vulkan device; a skip is a pass.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <future>
#include <span>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include "core/align.h"
#include "core/config.h"
#include "cpu/gemv_avx512.h"
#include "cpu/dequant.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/moe_bridge.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/pinned.h"
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

std::string data_path(const std::string& name) {
    return std::string(DEEPMOE_TEST_DATA_DIR) + "/" + name;
}

// Everything a GPU MoE case needs, brought up once and torn down in order.
// Declaration order matters: the ExpertStore's slabs come from the allocator,
// so the store must die first.
struct Rig {
    gpu::Device          device;
    gpu::MemoryAllocator alloc;
    Manifest             manifest;
    ShardSet             shards;
    storage::IoEngine    io;
    ExpertStore          store;
    Planner              planner;
    bool                 ready = false;
    std::string          why;

    ~Rig() { if (ready) io.stop(); }

    bool bring_up(uint32_t slots) {
        const std::string dir = model_dir();
        gpu::DeviceOptions opts;
        if (auto r = device.create(opts); !r) { why = r.error().str(); return false; }
        if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
            why = r.error().str(); return false;
        }
        auto mf = Manifest::load(ShardSet::join(dir, layout::kManifestFile));
        if (!mf) { why = mf.error().str(); return false; }
        manifest = std::move(*mf);
        if (auto r = shards.open_all(dir, manifest, /*unbuffered=*/true); !r) {
            why = r.error().str(); return false;
        }
        IoConfig io_cfg;
        auto backend = storage::make_default_backend(io_cfg);
        if (!backend) { why = backend.error().str(); return false; }
        if (auto r = io.start(std::move(*backend), io_cfg); !r) { why = r.error().str(); return false; }
        ready = true;

        CacheConfig cache;
        cache.slots_per_slab = slots;
        cache.budget_bytes   = uint64_t(slots) * layout::kExpertSlotBytes;
        auto backing = alloc.make_slab_backing();
        if (!backing) { why = backing.error().str(); return false; }
        if (auto r = store.init(std::move(*backing), cache); !r) { why = r.error().str(); return false; }
        if (auto r = planner.init(store, io, manifest, shards, cache, PrefetchConfig{}); !r) {
            why = r.error().str(); return false;
        }
        return true;
    }

    bool fill(ExpertKey key) {
        auto f = planner.fetch(key, IoPriority::BlockingMiss, 1, key.layer);
        if (!f) { why = f.error().str(); return false; }
        io.drain();
        if (!store.resident(key)) { why = "expert did not become resident"; return false; }
        return true;
    }
};

// Runs one variant over one resident expert and returns y.
Result<std::vector<float>> run_variant(Rig& rig, const gpu::MoeSpec& spec,
                                       ExpertKey key, const std::vector<float>& x,
                                       uint32_t slots, uint32_t slot_of_interest) {
    gpu::MoeDims dims;
    dims.layer = key.layer;
    dims.slots = slots;
    gpu::MoeRunner runner;
    if (auto r = runner.create(rig.device, rig.alloc, gpu::default_shader_dir(), spec, dims); !r)
        return std::unexpected(r.error());

    std::memcpy(runner.pointer_table(), rig.store.pointer_table(), rig.store.pointer_table_bytes());
    for (uint32_t s = 0; s < slots; ++s) runner.ids()[s] = 0;
    runner.ids()[slot_of_interest] = key.expert;
    runner.slot_list()[0] = slot_of_interest;
    runner.set_list_count(1);
    for (uint32_t i = 0; i < spec.m * slots; ++i) runner.route_weights()[i] = 1.0f;
    for (uint32_t m = 0; m < spec.m; ++m)
        for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
            runner.x_fp16()[m * layout::kHiddenSize + i] = cpu::float_to_fp16(x[i]);
    // These cases assert that every column agrees, so every column must be
    // computed: a dispatch's live column count defaults to 1, the decode shape
    // (MoeRunner::set_live_columns).
    runner.set_live_columns(spec.m);

    if (auto r = runner.run(1); !r) return std::unexpected(r.error());
    std::vector<float> y(size_t(spec.m) * layout::kHiddenSize);
    std::memcpy(y.data(), runner.y(), y.size() * sizeof(float));
    return y;
}

}  // namespace

// The headline case: expert (0, 0), every specialisation variant of design
// §7.1, all against the one torch fp32 answer.
DEEPMOE_TEST(gpu_moe, matches_the_oracle_across_every_variant) {
    if (skip_without_model("gpu_moe.matches_the_oracle_across_every_variant")) return;

    auto golden = load_golden(data_path("l1_layer0_expert0.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;

    Rig rig;
    if (!rig.bring_up(/*slots=*/2)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    REQUIRE(rig.fill(key));

    // The fp32 reference x rounded to fp16 is the activation precision design
    // §6 specifies, so the floor on the error is x's own rounding, not the
    // kernel's. Report it once so the two contributions can be told apart.
    double x_rel = 0.0, xn = 0.0;
    for (float v : g.x) {
        const double d = double(cpu::fp16_to_float(cpu::float_to_fp16(v))) - v;
        x_rel += d * d;
        xn += double(v) * v;
    }
    std::printf("       x fp16 round-trip: relative L2 error %.3e\n", std::sqrt(x_rel / xn));

    struct Case { gpu::MoeSpec spec; const char* what; double tol = 1e-3; };
    const Case cases[] = {
        {{1, 32, 32, 1, 0}, "baseline: wave32, arithmetic decode, fp16 h"},
        {{1, 16, 32, 1, 0}, "16 lanes per row"},
        {{1, 64, 32, 1, 0}, "64 lanes per row over 32-wide subgroups"},
        {{1, 64, 64, 1, 0}, "wave64"},
        {{1, 32, 64, 1, 0}, "wave64, 32 lanes per row"},
        {{1, 32, 32, 0, 0}, "constant-table FP4 decode"},
        {{1, 32, 32, 2, 0}, "select-tree FP4 decode"},
        {{1, 32, 32, 1, 1}, "fp32 h"},
        {{1, 32, 32, 0, 0, 2}, "2 weight rows per lane"},
        {{1, 16, 32, 0, 0, 4}, "4 weight rows per lane, 16 lanes"},
        {{1, 64, 64, 0, 0, 2}, "2 rows per lane, wave64, 64 lanes"},
        {{6, 32, 32, 1, 0}, "M=6, fp16 h"},
        {{6, 32, 32, 1, 1}, "M=6, fp32 h"},
        {{6, 16, 32, 0, 0, 2}, "M=6, 2 rows per lane"},
        // P2: the three x-staging designs of docs/kernel_p2_moe.md.
        {{1, 32, 32, 0, 0, 1, 1}, "x tiled through LDS"},
        {{1, 32, 32, 0, 0, 1, 2}, "x tiled through LDS, packed fp16 FMA", 2e-3},
        {{1, 32, 32, 0, 0, 1, 3}, "x quantised to int8, dot4", 6e-3},
        {{1, 16, 32, 0, 0, 2, 1}, "LDS tile, 16 lanes, 2 rows per lane"},
        {{6, 16, 32, 0, 0, 2, 1}, "M=6, LDS tile"},
        {{6, 16, 32, 0, 0, 2, 2}, "M=6, LDS tile, packed fp16 FMA", 2e-3},
        {{6, 16, 32, 0, 0, 2, 3}, "M=6, LDS tile, int8 dot4", 6e-3},
        {{6, 32, 32, 0, 0, 1, 1}, "M=6, LDS tile, 32 lanes"},
        {{6, 32, 32, 0, 0, 1, 3}, "M=6, int8 dot4, 32 lanes", 6e-3},
        {{1, 32, 32, 0, 0, 1, 4}, "packed fp16 from global, no LDS", 2e-3},
        {{6, 16, 32, 0, 0, 2, 4}, "M=6, packed fp16 from global", 2e-3},
        {{1, 32, 32, 0, 0, 1, 5}, "int8 dot4 from global, no LDS", 6e-3},
        {{6, 16, 32, 0, 0, 2, 5}, "M=6, int8 dot4 from global", 6e-3},
        // XMode 6: x pre-quantised to int8 by gpu/shaders/moe_xquant.slang and
        // consumed by dispatch A only (dispatch B's activation is h, so it
        // falls back to the fp32 path). This is the §3.6 design whose accuracy
        // decides whether it can ever be a default -- the printed
        // rel_to_scale is the number docs/kernel_p2_moe.md §8 item 2 quotes.
        {{1, 32, 32, 0, 0, 1, 6}, "int8 x from the pre-pass, A only", 6e-3},
        {{1, 16, 32, 0, 0, 2, 6}, "int8 x pre-pass, 16 lanes, 2 rows", 6e-3},
        {{6, 16, 32, 0, 0, 2, 6}, "M=6, int8 x pre-pass", 6e-3},
    };

    for (const Case& c : cases) {
        auto y = run_variant(rig, c.spec, key, g.x, /*slots=*/1, /*slot_of_interest=*/0);
        if (!y) {
            _ctx.fail(__FILE__, __LINE__,
                      std::format("{}: {}", c.spec.name(), y.error().str()));
            continue;
        }
        // With M > 1 every column got the same x, so every column must agree.
        // Not bit for bit: the M accumulators are independent chains and the
        // shader compiler is free to contract them into FMAs differently, which
        // costs 1-2 ULP. Anything larger would mean the columns are reading
        // different weights.
        double ymax = 0.0, col_spread = 0.0;
        for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
            ymax = std::fmax(ymax, std::fabs(double((*y)[i])));
        for (uint32_t m = 1; m < c.spec.m; ++m)
            for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
                col_spread = std::fmax(col_spread,
                                       std::fabs(double((*y)[m * layout::kHiddenSize + i]) - (*y)[i]));
        if (c.spec.m > 1) {
            const double rel = ymax > 0 ? col_spread / ymax : col_spread;
            if (rel > 1e-6)
                _ctx.fail(__FILE__, __LINE__,
                          std::format("{}: columns disagree by {:.3e} of |y|max", c.spec.name(), rel));
        }

        std::vector<float> col0((*y).begin(), (*y).begin() + layout::kHiddenSize);
        const Compare cmp = compare(col0, g.y);
        std::printf("       %-26s %-44s cos %.9f  max|dy| %.3e (%.2e of |y|max)\n",
                    c.spec.name().c_str(), c.what, cmp.cosine, cmp.max_abs, cmp.rel_to_scale);
        // design §12 L1: <= 1e-3 relative with fp16 activations. The GPU path
        // rounds x to fp16 where the CPU oracle kept fp32, so the bar here is
        // the L1 bar, not the 1e-4 the all-fp32 CPU path is held to.
        // design §12 L1: 1e-3 relative for the fp16 path, 5e-3 for int8.
        // The int8 x path lands at 5.4e-3 / cos 1-6.8e-5 on this expert, which
        // is over the §12 bar; docs/kernel_p2_moe.md §4 has the arithmetic and
        // the recommendation.
        CHECK(cmp.rel_to_scale <= c.tol);
        CHECK(cmp.cosine >= (c.tol > 1e-3 ? 1.0 - 1e-4 : 1.0 - 1e-5));
    }
}

// The expert indirection list of design §7.9: two slots, only the second one is
// in the compute list, so the answer must be the second slot's expert and the
// first slot's expert must not contribute.
DEEPMOE_TEST(gpu_moe, the_indirection_list_picks_the_expert) {
    if (skip_without_model("gpu_moe.the_indirection_list_picks_the_expert")) return;

    auto golden = load_golden(data_path("l1_layer39_expert383.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;

    Rig rig;
    if (!rig.bring_up(/*slots=*/2)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const ExpertKey want{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    const ExpertKey decoy{static_cast<uint16_t>(g.layer), 0};
    REQUIRE(rig.fill(want));
    REQUIRE(rig.fill(decoy));

    gpu::MoeDims dims;
    dims.layer = want.layer;
    dims.slots = 2;
    gpu::MoeRunner runner;
    REQUIRE_OK(runner.create(rig.device, rig.alloc, gpu::default_shader_dir(),
                             gpu::MoeSpec{1, 32, 32, 1, 0}, dims));
    std::memcpy(runner.pointer_table(), rig.store.pointer_table(), rig.store.pointer_table_bytes());
    runner.ids()[0] = decoy.expert;
    runner.ids()[1] = want.expert;
    runner.slot_list()[0] = 1;          // compute slot 1 only
    runner.set_list_count(1);
    runner.route_weights()[0] = 1.0f;
    runner.route_weights()[1] = 1.0f;
    for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
        runner.x_fp16()[i] = cpu::float_to_fp16(g.x[i]);
    REQUIRE_OK(runner.run(1));

    std::vector<float> y(layout::kHiddenSize);
    std::memcpy(y.data(), runner.y(), y.size() * sizeof(float));
    const Compare cmp = compare(y, g.y);
    std::printf("       (%u, %u) selected out of two slots: cos %.9f, max|dy| %.3e (%.2e)\n",
                g.layer, g.expert, cmp.cosine, cmp.max_abs, cmp.rel_to_scale);
    CHECK(cmp.rel_to_scale <= 1e-3);
    CHECK(cmp.cosine >= 1.0 - 1e-6);
}

// docs/kernel_p2_moe.md §8 item 2: XMode 6 replaces x with an int8 + block-32
// approximation, so its error is a property of *x*, not of the kernel -- and it
// therefore varies from expert to expert. Expert (0, 0) lands at 2.9e-3, inside
// design §12's 5e-3 int8 bar; this is the other golden, which does not. The
// same two numbers come out of tools/oracle_shared.py's x_quant_study on the
// CPU, which is what makes the per-row / residual comparison there trustworthy.
DEEPMOE_TEST(gpu_moe, the_int8_x_pre_pass_is_expert_dependent) {
    if (skip_without_model("gpu_moe.the_int8_x_pre_pass_is_expert_dependent")) return;

    Rig rig;
    if (!rig.bring_up(/*slots=*/2)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const char* files[2] = {"l1_layer0_expert0.bin", "l1_layer39_expert383.bin"};
    double worst = 0.0;
    for (const char* file : files) {
        auto golden = load_golden(data_path(file));
        REQUIRE_OK(golden);
        const Golden& g = *golden;
        const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
        REQUIRE(rig.fill(key));
        for (uint32_t x_mode : {0u, 6u}) {
            gpu::MoeSpec spec{1, 32, 32, 0, 0, 1, x_mode};
            auto y = run_variant(rig, spec, key, g.x, /*slots=*/1, /*slot_of_interest=*/0);
            REQUIRE_OK(y);
            std::vector<float> col0(y->begin(), y->begin() + layout::kHiddenSize);
            const Compare cmp = compare(col0, g.y);
            std::printf("       (%2u,%3u) %-6s cos %.9f  %.3e of |y|max\n", g.layer, g.expert,
                        x_mode == 6 ? "int8 x" : "fp16 x", cmp.cosine, cmp.rel_to_scale);
            if (x_mode == 6) worst = std::fmax(worst, cmp.rel_to_scale);
            else CHECK(cmp.rel_to_scale <= 1e-3);
        }
    }
    // Not an assertion that int8 x is good enough -- it is the opposite. The
    // bound is here so the number cannot silently get worse, and so that
    // anything claiming int8 x is within §12's 5e-3 has to explain this test.
    std::printf("       worst int8-x error over the two goldens: %.3e "
                "(design §12's int8 bar is 5e-3)\n", worst);
    CHECK(worst > 5e-3);
    CHECK(worst <= 1.5e-2);
}

// The route weight of design §7.9 is applied inside dispatch A, and dispatch B
// sums the slots with no atomics. Running the same expert in two slots at half
// weight each must reproduce the single-slot answer.
DEEPMOE_TEST(gpu_moe, route_weights_and_the_slot_reduction) {
    if (skip_without_model("gpu_moe.route_weights_and_the_slot_reduction")) return;

    auto golden = load_golden(data_path("l1_layer0_expert0.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;

    Rig rig;
    if (!rig.bring_up(/*slots=*/1)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    REQUIRE(rig.fill(key));

    gpu::MoeDims dims;
    dims.layer = key.layer;
    dims.slots = 2;
    gpu::MoeRunner runner;
    REQUIRE_OK(runner.create(rig.device, rig.alloc, gpu::default_shader_dir(),
                             gpu::MoeSpec{1, 32, 32, 1, 0}, dims));
    std::memcpy(runner.pointer_table(), rig.store.pointer_table(), rig.store.pointer_table_bytes());
    runner.ids()[0] = key.expert;
    runner.ids()[1] = key.expert;       // the same expert in both slots
    runner.slot_list()[0] = 0;
    runner.slot_list()[1] = 1;
    runner.set_list_count(2);
    runner.route_weights()[0] = 0.25f;
    runner.route_weights()[1] = 0.75f;  // 0.25 + 0.75 == 1
    for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
        runner.x_fp16()[i] = cpu::float_to_fp16(g.x[i]);
    REQUIRE_OK(runner.run(1));

    std::vector<float> y(layout::kHiddenSize);
    std::memcpy(y.data(), runner.y(), y.size() * sizeof(float));
    const Compare cmp = compare(y, g.y);
    std::printf("       0.25 + 0.75 of the same expert: cos %.9f, max|dy| %.3e (%.2e)\n",
                cmp.cosine, cmp.max_abs, cmp.rel_to_scale);
    CHECK(cmp.rel_to_scale <= 1e-3);
    CHECK(cmp.cosine >= 1.0 - 1e-6);
}

// ---------------------------------------------------------------------------
// P2 (docs/kernel_p2_moe.md): the speculative-decode batch, the fp8 shared
// expert, the fp8 quantisation of h, and the partial dispatch of design §7.9.
// ---------------------------------------------------------------------------

namespace {

// tests/data/l1_shared_layer<L>.bin and l1q_layer<L>_expert<E>.bin, both written
// by tools/oracle_shared.py: the same x with three answers.
//   y_ref   fp32 everywhere -- what tools/oracle.py's l1_*.bin holds
//   y_hq    `silu(gate)*up` fp8-quantised before w2 (design §7.9 v0.6)
//   y_full  x fp8-quantised too, i.e. the reference implementation
//   y_hq16  x and h in fp16 (design §6) and then h quantised -- the arithmetic
//           the kernel actually performs, and therefore the vector it is held
//           to. y_hq16 and y_hq are 2.8e-3 apart on (0, 0) even though their
//           inputs differ by 2e-4: `fast_round_scale` is a step function of the
//           block amax, so a block sitting near a power of two lands on a grid
//           twice as coarse in one of the two. That is the quantiser's
//           sensitivity, not the kernel's error.
struct TriGolden {
    uint32_t layer = 0, expert = 0, dim = 0, inter = 0;
    uint64_t seed = 0;
    float    swiglu_limit = 0.0f;
    std::vector<float> x, y_ref, y_hq, y_full, y_hq16;
};

Result<TriGolden> load_tri(const std::string& path, const char* magic, bool has_expert,
                           uint32_t extra_u64 = 0) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::Io, "cannot open " + path);
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    char got[4] = {};
    uint32_t version = 0;
    if (std::fread(got, 1, 4, f) != 4 || std::memcmp(got, magic, 4) != 0 ||
        std::fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1)
        return fail(Err::Corrupt, path + ": not a version-1 " + magic + " file");
    TriGolden g;
    uint32_t head[4] = {};
    const uint32_t nhead = has_expert ? 4u : 3u;
    if (std::fread(head, sizeof(uint32_t), nhead, f) != nhead ||
        std::fread(&g.seed, sizeof(uint64_t), 1, f) != 1 ||
        std::fread(&g.swiglu_limit, sizeof(float), 1, f) != 1)
        return fail(Err::Corrupt, path + ": truncated header");
    uint32_t i = 0;
    g.layer = head[i++];
    if (has_expert) g.expert = head[i++];
    g.dim   = head[i++];
    g.inter = head[i++];
    for (uint32_t k = 0; k < extra_u64; ++k) {
        uint64_t skip = 0;
        if (std::fread(&skip, sizeof(uint64_t), 1, f) != 1)
            return fail(Err::Corrupt, path + ": truncated hash table");
    }
    for (std::vector<float>* v : {&g.x, &g.y_ref, &g.y_hq, &g.y_full, &g.y_hq16}) {
        v->resize(g.dim);
        if (std::fread(v->data(), sizeof(float), g.dim, f) != g.dim)
            return fail(Err::Corrupt, path + ": truncated vectors");
    }
    return g;
}

// design §7.9's expert FFN on the CPU for M columns at once, straight out of an
// ExpertStore slot. This is the M > 1 reference: the GPU kernel is handed the
// *same* fp16-rounded activations, so anything beyond fp32 summation order is a
// kernel bug rather than an activation-precision effect. cpu/gemv_fp4_ref
// already carries the M dimension (cpu/gemv_avx512.h GemvShape::M), so nothing
// had to be added there.
Result<std::vector<float>> ffn_ref_fp4_m(const std::byte* base, const ExpertEntry& e,
                                         const std::vector<float>& x,   // [M][dim]
                                         uint32_t m, uint32_t dim, uint32_t inter,
                                         float limit) {
    auto span_of = [&](ExpertPart p) {
        return std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(base + e.offset_of(p)),
            static_cast<size_t>(e.bytes_of(p)));
    };
    std::vector<float> gate(size_t(m) * inter), up(size_t(m) * inter);
    std::vector<float> h(size_t(m) * inter), y(size_t(m) * dim);
    const cpu::GemvShape wide{inter, dim, m};
    if (auto r = cpu::gemv_fp4_ref(span_of(ExpertPart::W1Weight), span_of(ExpertPart::W1Scale),
                                   x, wide, gate); !r) return std::unexpected(r.error());
    if (auto r = cpu::gemv_fp4_ref(span_of(ExpertPart::W3Weight), span_of(ExpertPart::W3Scale),
                                   x, wide, up); !r) return std::unexpected(r.error());
    for (size_t i = 0; i < h.size(); ++i) {
        const float gv = std::fmin(gate[i], limit);
        const float uv = std::fmin(std::fmax(up[i], -limit), limit);
        h[i] = (gv / (1.0f + std::exp(-gv))) * uv;
    }
    const cpu::GemvShape narrow{dim, inter, m};
    if (auto r = cpu::gemv_fp4_ref(span_of(ExpertPart::W2Weight), span_of(ExpertPart::W2Scale),
                                   h, narrow, y); !r) return std::unexpected(r.error());
    return y;
}

// The fp8 shared expert's six tensors, read off NVMe into one GPU-visible
// buffer. Unlike a routed expert it is not an ExpertStore slot -- it is
// resident for the life of the process (design §9.3) -- so the six device
// addresses are built here and dropped into the runner's pointer table at a
// reserved expert index.
struct SharedExpert {
    gpu::MemoryAllocator* alloc = nullptr;
    gpu::GpuBuffer        buf{};
    uint64_t              addr[kExpertPartCount] = {};
    uint64_t              bytes[kExpertPartCount] = {};
    std::string           why;

    ~SharedExpert() { if (alloc && buf.valid()) alloc->free(buf); }

    bool load(Rig& rig, uint32_t layer) {
        // The pointer-table part order is w1.weight, w1.scale, w2.weight,
        // w2.scale, w3.weight, w3.scale -- the same six slots the FP4 path uses.
        const char* mats[3] = {"w1", "w2", "w3"};
        AlignedRead reads[kExpertPartCount];
        for (uint32_t i = 0; i < 3; ++i) {
            const std::string name =
                std::format("layers.{}.ffn.shared_experts.{}.weight", layer, mats[i]);
            auto v = rig.manifest.tensor_read(name);
            if (!v) { why = v.error().str(); return false; }
            auto s = rig.manifest.tensor_scale_read(name);
            if (!s) { why = s.error().str(); return false; }
            reads[2 * i]     = *v;
            reads[2 * i + 1] = *s;
        }
        uint64_t total = 0;
        uint64_t off[kExpertPartCount] = {};
        for (uint32_t i = 0; i < kExpertPartCount; ++i) {
            off[i] = total;
            total += align_up(reads[i].aligned_bytes, kPageSize);
        }
        auto b = rig.alloc.allocate(total, /*host_visible=*/true, /*device_address=*/true);
        if (!b) { why = b.error().str(); return false; }
        alloc = &rig.alloc;
        buf = *b;
        if (reinterpret_cast<uintptr_t>(buf.host_ptr) % kPageSize) {
            why = "the shared-expert buffer is not page aligned";
            return false;
        }
        std::vector<std::future<storage::IoResult>> futures;
        for (uint32_t i = 0; i < kExpertPartCount; ++i) {
            auto file = rig.shards.require(reads[i].file);
            if (!file) { why = file.error().str(); return false; }
            storage::IoRequest req;
            req.priority = IoPriority::BlockingMiss;
            req.file     = *file;
            req.file_off = reads[i].aligned_off;
            req.bytes    = reads[i].aligned_bytes;
            req.dst      = static_cast<std::byte*>(buf.host_ptr) + off[i];
            auto fut = rig.io.submit_future(req);
            if (!fut) { why = fut.error().str(); return false; }
            futures.push_back(std::move(*fut));
            addr[i]  = buf.dev_addr + off[i] + reads[i].skew;
            bytes[i] = reads[i].bytes;
        }
        for (auto& f : futures)
            if (!f.get().ok()) { why = "shared expert read failed"; return false; }
        // design §5.1: safetensors offsets are 8-byte multiples, so the part
        // addresses the kernel sees here are 8-aligned exactly as a routed
        // expert's are. Hold the kernels to it.
        for (uint64_t a : addr)
            if (a % 8) { why = "a shared-expert part address is not 8-byte aligned"; return false; }
        return true;
    }
};

// Copies the ExpertStore's [layers][384][6] table into a runner whose expert
// dimension is one entry wider, and puts the shared expert in the spare slot.
void fill_table_with_shared(gpu::MoeRunner& runner, const ExpertStore& store,
                            uint32_t layer, uint32_t experts_per_layer,
                            const SharedExpert* shared) {
    uint64_t* dst = runner.pointer_table();
    std::memset(dst, 0, runner.pointer_table_entries() * sizeof(uint64_t));
    const uint64_t* src = store.pointer_table();
    const uint32_t src_stride = layout::kRoutedExperts * kExpertPartCount;
    const uint32_t dst_stride = experts_per_layer * kExpertPartCount;
    for (uint32_t L = 0; L < layout::kTotalLogicalLayers; ++L)
        std::memcpy(dst + size_t(L) * dst_stride, src + size_t(L) * src_stride,
                    src_stride * sizeof(uint64_t));
    if (shared) {
        uint64_t* row = dst + (size_t(layer) * experts_per_layer + experts_per_layer - 1)
                              * kExpertPartCount;
        for (uint32_t i = 0; i < kExpertPartCount; ++i) row[i] = shared->addr[i];
    }
}

}  // namespace

// The speculative verify batch with *different* activations per column. The
// M = 6 case in the variant sweep above replicates one x, which cannot catch a
// column-indexing bug in the LDS x tile: every column would read the right
// numbers by accident. Here column m is a different linear map of x, so any
// confusion between columns shows up immediately.
DEEPMOE_TEST(gpu_moe, the_verify_batch_computes_one_answer_per_column) {
    if (skip_without_model("gpu_moe.the_verify_batch_computes_one_answer_per_column")) return;

    auto golden = load_golden(data_path("l1_layer0_expert0.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;

    Rig rig;
    if (!rig.bring_up(/*slots=*/1)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    REQUIRE(rig.fill(key));
    auto slot = rig.store.lookup(key, 1);
    REQUIRE(slot.has_value());
    auto entry = rig.manifest.require_expert(key);
    REQUIRE_OK(entry);

    constexpr uint32_t kM = 6;
    const uint32_t dim = layout::kHiddenSize;
    std::vector<uint16_t> x16(size_t(kM) * dim);
    std::vector<float>    x32(size_t(kM) * dim);
    for (uint32_t m = 0; m < kM; ++m)
        for (uint32_t i = 0; i < dim; ++i) {
            const float v = g.x[i] * (1.0f + 0.25f * float(m))
                          + 0.05f * float(m) * g.x[(i + 37 * (m + 1)) % dim];
            const uint16_t hh = cpu::float_to_fp16(v);
            x16[size_t(m) * dim + i] = hh;
            x32[size_t(m) * dim + i] = cpu::fp16_to_float(hh);
        }

    // The slab is path-A write-combined memory: CPU reads of it are uncached
    // and the scalar reference touches every weight byte twice, so copy the
    // slot out in one streaming pass first (kernel_p1.md §2.4).
    std::vector<std::byte> host_slot(static_cast<size_t>(layout::kExpertSlotBytes));
    std::memcpy(host_slot.data(), slot->host_ptr, host_slot.size());
    auto want = ffn_ref_fp4_m(host_slot.data(), **entry,
                              x32, kM, dim, layout::kMoeIntermediate, g.swiglu_limit);
    REQUIRE_OK(want);
    // The six columns must not be near-copies of each other, or the test proves
    // nothing. Report the spread the reference itself has.
    double col_spread = 0.0, ymax = 0.0;
    for (uint32_t i = 0; i < dim; ++i) ymax = std::fmax(ymax, std::fabs((*want)[i]));
    for (uint32_t m = 1; m < kM; ++m)
        for (uint32_t i = 0; i < dim; ++i)
            col_spread = std::fmax(col_spread,
                                   std::fabs(double((*want)[size_t(m) * dim + i]) - (*want)[i]));
    std::printf("       reference columns differ by %.3f of |y|max\n", col_spread / ymax);
    CHECK(col_spread / ymax > 0.1);

    struct Case { gpu::MoeSpec spec; const char* what; double tol; };
    const Case cases[] = {
        {{kM, 32, 32, 0, 0, 1, 0}, "global x", 2e-4},
        {{kM, 16, 32, 0, 0, 2, 0}, "global x, 16 lanes, 2 rows", 2e-4},
        {{kM, 16, 32, 0, 0, 2, 1}, "LDS x tile", 2e-4},
        {{kM, 32, 32, 0, 0, 1, 1}, "LDS x tile, 32 lanes", 2e-4},
        {{kM, 16, 32, 0, 0, 2, 2}, "LDS x tile, packed fp16", 3e-3},
        {{kM, 16, 32, 0, 0, 2, 3}, "LDS x tile, int8 dot4", 1.5e-2},
        {{kM, 16, 32, 0, 0, 2, 4}, "packed fp16 from global", 3e-3},
        {{kM, 16, 32, 0, 0, 2, 5}, "int8 dot4 from global", 1.5e-2},
        {{kM, 16, 32, 0, 0, 2, 6}, "int8 x from the pre-pass", 1.5e-2},
        {{kM, 32, 32, 0, 0, 1, 6}, "int8 x pre-pass, 32 lanes", 1.5e-2},
    };
    for (const Case& c : cases) {
        gpu::MoeDims dims;
        dims.layer = key.layer;
        dims.slots = 1;
        gpu::MoeRunner runner;
        auto ok = runner.create(rig.device, rig.alloc, gpu::default_shader_dir(), c.spec, dims);
        if (!ok) {
            _ctx.fail(__FILE__, __LINE__, std::format("{}: {}", c.spec.name(), ok.error().str()));
            continue;
        }
        std::memcpy(runner.pointer_table(), rig.store.pointer_table(),
                    rig.store.pointer_table_bytes());
        runner.ids()[0] = key.expert;
        runner.slot_list()[0] = 0;
        runner.set_list_count(1);
        for (uint32_t i = 0; i < kM; ++i) runner.route_weights()[i] = 1.0f;
        std::memcpy(runner.x_fp16(), x16.data(), x16.size() * sizeof(uint16_t));
        runner.set_live_columns(kM);      // this case checks all M columns
        REQUIRE_OK(runner.run(1));
        std::vector<float> y(size_t(kM) * dim);
        std::memcpy(y.data(), runner.y(), y.size() * sizeof(float));

        double worst = 0.0;
        uint32_t worst_m = 0;
        for (uint32_t m = 0; m < kM; ++m) {
            std::vector<float> got(y.begin() + size_t(m) * dim, y.begin() + size_t(m + 1) * dim);
            std::vector<float> ref(want->begin() + size_t(m) * dim,
                                   want->begin() + size_t(m + 1) * dim);
            const Compare cmp = compare(got, ref);
            if (cmp.rel_to_scale > worst) { worst = cmp.rel_to_scale; worst_m = m; }
        }
        std::printf("       %-32s %-28s worst column %u at %.3e of |y|max\n",
                    c.spec.name().c_str(), c.what, worst_m, worst);
        CHECK(worst <= c.tol);
    }
}

// MoE(M): what one layer's MoE costs when the batch's columns route to
// DIFFERENT experts -- the half of the verify batch that C(M) does not cover
// (docs/p4_mgt1.md section 4). The experts are made resident first, so this is
// the kernel's cost with the I/O out of the way; the union of six columns'
// top-6 is ~26 experts a layer (docs/p3_dspark.md section 12.6), so the real
// loop's NVMe side grows with M instead of amortising.
//
// Registered under its own suite name so a correctness run does not pay for it.
// The bring-up a batch MoE test needs: device, manifest, shards, io, the pinned
// set, the expert cache and the planner. Shared by the M-curve benchmarks and by
// the union-batch cases below.
struct FullRig {
    gpu::Device         device;
    gpu::MemoryAllocator alloc;
    Manifest            manifest;
    V41Config           config;
    ShardSet            shards;
    storage::IoEngine   io;
    store::PinnedStore  pinned;
    ExpertStore         store;
    Planner             planner;
    bool io_started = false;
    std::string why;

    ~FullRig() { if (io_started) io.stop(); }

    bool bring_up(uint32_t slots) {
        const std::string dir = model_dir() ? model_dir() : "";
        if (auto r = device.create({}); !r) { why = r.error().str(); return false; }
        if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
            why = r.error().str(); return false;
        }
        auto cfg = V41Config::load(ShardSet::join(dir, "config.json"));
        if (!cfg) { why = cfg.error().str(); return false; }
        config = std::move(*cfg);
        auto mf = Manifest::load(ShardSet::join(dir, layout::kManifestFile));
        if (!mf) { why = mf.error().str(); return false; }
        manifest = std::move(*mf);
        if (auto r = shards.open_all(dir, manifest, /*unbuffered=*/true); !r) {
            why = r.error().str(); return false;
        }
        IoConfig iocfg;
        auto backend = storage::make_default_backend(iocfg);
        if (!backend) { why = backend.error().str(); return false; }
        if (auto r = io.start(std::move(*backend), iocfg); !r) {
            why = r.error().str(); return false;
        }
        io_started = true;
        auto pb = alloc.make_slab_backing();
        if (!pb) { why = pb.error().str(); return false; }
        store::PinnedConfig pc;
        pc.region_bytes = 1ull << 30;
        if (auto r = pinned.init(std::move(*pb), pc); !r) { why = r.error().str(); return false; }
        CacheConfig cache;
        cache.slots_per_slab = slots;
        cache.budget_bytes   = uint64_t(slots) * layout::kExpertSlotBytes;
        auto eb = alloc.make_slab_backing();
        if (!eb) { why = eb.error().str(); return false; }
        if (auto r = store.init(std::move(*eb), cache); !r) { why = r.error().str(); return false; }
        if (auto r = planner.init(store, io, manifest, shards, cache, PrefetchConfig{}); !r) {
            why = r.error().str(); return false;
        }
        return true;
    }
};

DEEPMOE_TEST(mgt1, moe_m_curve) {
    if (skip_without_model("mgt1.moe_m_curve")) return;


    const uint32_t iters = [] {
        if (const char* e = std::getenv("DEEPMOE_MOE_M_ITERS"); e && *e)
            return uint32_t(std::strtoul(e, nullptr, 10));
        return 10u;
    }();
    constexpr uint32_t kTopk = 6;
    constexpr uint32_t kMaxM = 6;
    constexpr uint32_t kUnion = kTopk;      // per column: its own six experts
    FullRig rig;
    if (!rig.bring_up(/*slots=*/kMaxM * kUnion + 4)) {
        std::printf("       SKIP mgt1.moe_m_curve: %s\n", rig.why.c_str());
        return;
    }
    const uint32_t layer = 0;
    const uint32_t dim = rig.config.text.hidden_size;
    {
        std::vector<std::string> names = store::pinned_global_tensors(rig.manifest);
        auto per = store::pinned_layer_tensors(rig.manifest, layer);
        names.insert(names.end(), per.begin(), per.end());
        auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io, names);
        if (!r) { std::printf("       SKIP mgt1.moe_m_curve: pinned: %s\n", r.error().str().c_str()); return; }
    }

    runtime::GpuMoeBridge bridge;
    REQUIRE_OK(bridge.create(rig.device, rig.alloc, gpu::default_shader_dir(), rig.store,
                             rig.planner, rig.pinned, rig.config.text));

    std::vector<uint32_t> ids(kMaxM * kTopk);
    std::vector<float>    w(kMaxM * kTopk);
    std::vector<float>    x(size_t(kMaxM) * dim);
    std::vector<float>    y(size_t(kMaxM) * dim);
    for (uint32_t m = 0; m < kMaxM; ++m) {
        for (uint32_t s = 0; s < kTopk; ++s) {
            ids[m * kTopk + s] = 10 + m * kTopk + s;      // disjoint sets, m = 6 -> 40..45
            w[m * kTopk + s]   = 0.05f + 0.02f * float(s);
        }
        for (uint32_t i = 0; i < dim; ++i)
            x[size_t(m) * dim + i] = std::sin(0.01f * float((i + 37 * (m + 1)) % 977));
    }
    for (uint32_t i = 0; i < kMaxM * kTopk; ++i) {
        const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(ids[i])};
        auto f = rig.planner.fetch(key, IoPriority::BlockingMiss, 1, layer);
        if (!f) { std::printf("       SKIP mgt1.moe_m_curve: fetch: %s\n", f.error().str().c_str()); return; }
    }
    rig.io.drain();

    std::printf("    MoE(M), layer %u, disjoint expert sets a column, %u iterations a case\n",
                layer, iters);
    double base = 0.0;
    for (uint32_t M : {1u, 2u, 4u, 6u}) {
        runtime::GpuMoeBridge::BatchCall call;
        call.layer = layer; call.m = M; call.ids = ids.data(); call.weights = w.data();
        call.topk = kTopk; call.x = x.data(); call.y = y.data(); call.hidden = dim;
        for (uint32_t i = 0; i < 3; ++i) REQUIRE_OK(bridge.run_batch(call));
        const auto t0 = Clock::now();
        for (uint32_t i = 0; i < iters; ++i) REQUIRE_OK(bridge.run_batch(call));
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count()
                          / double(iters);
        if (M == 1) base = ms;
        std::printf("      M=%u  %8.3f ms/layer  %8.3f ms/token  C(M)/C(1) %.2fx\n",
                    M, ms, ms / double(M), base > 0 ? ms / base : 0.0);
    }
    std::printf("      (the six columns' expert union is %u of the %u per layer; the real\n"
                "       loop's residency gate reads that union from NVMe, which this does not)\n",
                kMaxM * kUnion, rig.config.text.n_routed_experts);
}


// ============================================================================
// Track F1: the verify batch as ONE dispatch over the columns' expert UNION.
//
// docs/p4_dspark_runtime.md §2.2's plan A -- one dispatch a column -- is correct
// and 8.34x the M = 1 MoE at M = 6 (docs/p4_mgt1.md §4), which makes a verify
// batch about 5x more expensive than running the same six tokens one at a time.
// Plan C without a new kernel: widen the slot axis to the union of the columns'
// top-6, fill `route_weights[m][slot]` with zero wherever token m does not route
// to that slot, and every expert's weights are read ONCE for the whole batch.
//
// Two things are checked and one is measured:
//   * per column, the union answer against the same column of the per-column
//     path, which the case below has already pinned to the M = 1 answer bit for
//     bit. Not bit-exact: dispatch B sums the slots in slot order and the union
//     re-associates that sum, so the criterion is 1e-6 of |y|max;
//   * the zero-weight slots really contribute nothing;
//   * ms/layer against `run_batch` at the same M, for disjoint expert sets (the
//     worst case for the union: |union| = 6M) and overlapping ones (the real
//     shape, docs/p3_dspark.md §12.6's ~26 of 36).
// ============================================================================
DEEPMOE_TEST(gpu_moe, the_verify_batch_runs_its_expert_union_once) {
    if (skip_without_model("gpu_moe.the_verify_batch_runs_its_expert_union_once")) return;

    constexpr uint32_t kTopk = 6;
    constexpr uint32_t kM    = 6;
    FullRig rig;
    if (!rig.bring_up(/*slots=*/kM * kTopk + 4)) {
        std::printf("       SKIP gpu_moe.the_verify_batch_runs_its_expert_union_once: %s\n",
                    rig.why.c_str());
        return;
    }
    const uint32_t layer = 0;
    const uint32_t dim = rig.config.text.hidden_size;
    {
        std::vector<std::string> names = store::pinned_global_tensors(rig.manifest);
        auto per = store::pinned_layer_tensors(rig.manifest, layer);
        names.insert(names.end(), per.begin(), per.end());
        auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io, names);
        if (!r) { std::printf("       SKIP: pinned: %s\n", r.error().str().c_str()); return; }
    }
    runtime::GpuMoeBridge bridge;
    REQUIRE_OK(bridge.create(rig.device, rig.alloc, gpu::default_shader_dir(), rig.store,
                             rig.planner, rig.pinned, rig.config.text));

    std::vector<float> x(size_t(kM) * dim);
    for (uint32_t m = 0; m < kM; ++m)
        for (uint32_t i = 0; i < dim; ++i)
            x[size_t(m) * dim + i] = std::sin(0.01f * float((i + 37 * (m + 1)) % 977));

    std::vector<uint32_t> ids_disjoint(size_t(kM) * kTopk), ids_overlap(size_t(kM) * kTopk),
                          ids_reordered(size_t(kM) * kTopk);
    std::vector<float>    w(size_t(kM) * kTopk);
    for (uint32_t m = 0; m < kM; ++m)
        for (uint32_t s = 0; s < kTopk; ++s) {
            ids_disjoint[m * kTopk + s] = 10 + m * kTopk + s;
            // Three experts every column shares, three of its own.
            ids_overlap[m * kTopk + s] = (s < 3) ? (10 + s) : (10 + kTopk + m * 3 + (s - 3));
            // The adversarial case for the union: every column routes to the
            // SAME six experts, odd columns in the opposite order. The union
            // fixes one slot order for all of them, so an odd column's seven
            // contributions are summed in a different order than the M = 1 path
            // sums them -- which is the only way the two can differ at all.
            ids_reordered[m * kTopk + s] = (m % 2) ? (10 + kTopk - 1 - s) : (10 + s);
            w[m * kTopk + s] = 0.05f + 0.02f * float(s) + 0.01f * float(m);
        }
    // Every distinct expert of both routings, fetched once: a second fetch of a
    // resident expert is an `already-exists` refusal, not a hit.
    {
        std::vector<uint32_t> all;
        for (const auto* v : {&ids_disjoint, &ids_overlap, &ids_reordered})
            for (uint32_t e : *v)
                if (std::find(all.begin(), all.end(), e) == all.end()) all.push_back(e);
        for (uint32_t e : all) {
            const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(e)};
            auto f = rig.planner.fetch(key, IoPriority::BlockingMiss, 1, layer);
            if (!f) {
                std::printf("       SKIP: expert %u would not fetch: %s\n", e,
                            f.error().str().c_str());
                return;
            }
        }
        rig.io.drain();
    }

    struct Routing { const char* what; const std::vector<uint32_t>* ids; };
    const Routing routings[] = {{"disjoint sets", &ids_disjoint},
                                {"overlapping sets", &ids_overlap},
                                {"reordered sets", &ids_reordered}};

    for (const Routing& r : routings) {
        runtime::GpuMoeBridge::BatchCall call;
        call.layer = layer; call.m = kM; call.ids = r.ids->data(); call.weights = w.data();
        call.topk = kTopk; call.x = x.data(); call.hidden = dim;

        const auto u = bridge.union_experts(call);
        std::vector<float> y_union(size_t(kM) * dim), y_cols(size_t(kM) * dim);
        call.y = y_union.data();
        REQUIRE_OK(bridge.run_batch_union(call));
        const auto info = bridge.union_info();
        CHECK(info.routed == u.size());
        call.y = y_cols.data();
        REQUIRE_OK(bridge.run_batch(call));

        double worst = 0.0, ymax = 0.0;
        uint32_t worst_m = 0;
        for (float v : y_cols) ymax = std::fmax(ymax, std::fabs(double(v)));
        for (uint32_t m = 0; m < kM; ++m) {
            double d = 0.0;
            for (uint32_t i = 0; i < dim; ++i)
                d = std::fmax(d, std::fabs(double(y_union[size_t(m) * dim + i])
                                           - double(y_cols[size_t(m) * dim + i])));
            if (d > worst) { worst = d; worst_m = m; }
        }
        std::printf("       %-18s union %2u of %2u experts | worst column %u at %.3e of |y|max\n",
                    r.what, info.routed, kM * kTopk, worst_m, ymax > 0 ? worst / ymax : 0.0);
        CHECK(ymax > 0.0);
        CHECK(worst / ymax <= 1e-6);

        // Columns must not be near-copies, or the comparison above proves little.
        double spread = 0.0;
        for (uint32_t m = 1; m < kM; ++m)
            for (uint32_t i = 0; i < dim; ++i)
                spread = std::fmax(spread, std::fabs(double(y_cols[size_t(m) * dim + i])
                                                     - double(y_cols[i])));
        CHECK(spread / ymax > 0.05);
    }

    // The zero-weight slots contribute exactly nothing: give column 0 a weight
    // row of zeros while every other column keeps its routing, and column 0 must
    // come out the shared expert's answer alone -- which is what a column whose
    // slot is not in its own top-6 has to look like for the union to be sound.
    {
        std::vector<float> w0 = w;
        for (uint32_t s = 0; s < kTopk; ++s) w0[s] = 0.0f;
        runtime::GpuMoeBridge::BatchCall call;
        call.layer = layer; call.m = kM; call.ids = ids_disjoint.data(); call.weights = w0.data();
        call.topk = kTopk; call.x = x.data(); call.hidden = dim;
        std::vector<float> y_zero(size_t(kM) * dim);
        call.y = y_zero.data();
        REQUIRE_OK(bridge.run_batch_union(call));
        runtime::GpuMoeBridge::BatchCall one = call;
        one.m = 1;
        std::vector<float> y_one(dim);
        one.y = y_one.data();
        REQUIRE_OK(bridge.run_batch(one));
        double d = 0.0, ymax = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            ymax = std::fmax(ymax, std::fabs(double(y_one[i])));
            d = std::fmax(d, std::fabs(double(y_zero[i]) - double(y_one[i])));
        }
        std::printf("       zero-weight column: union vs the shared expert alone %.3e of |y|max "
                    "(|y|max %.3f)\n", ymax > 0 ? d / ymax : 0.0, ymax);
        CHECK(ymax > 0.0);
        CHECK(d / ymax <= 1e-6);
    }
}

// MoE(M) again, but for the union path: what the SAME verify batch costs as one
// dispatch over its expert union instead of M dispatches over M expert sets.
// The number that matters for docs/p4_dspark_runtime.md's arithmetic is
// ms/token: at M = 1 the union path is the M = 1 path, and every column after
// that should cost the incremental experts, not a whole token.
DEEPMOE_TEST(mgt1, moe_union_m_curve) {
    if (skip_without_model("mgt1.moe_union_m_curve")) return;

    const uint32_t iters = [] {
        if (const char* e = std::getenv("DEEPMOE_MOE_M_ITERS"); e && *e)
            return uint32_t(std::strtoul(e, nullptr, 10));
        return 10u;
    }();
    constexpr uint32_t kTopk = 6;
    constexpr uint32_t kMaxM = 6;
    FullRig rig;
    if (!rig.bring_up(/*slots=*/kMaxM * kTopk + 4)) {
        std::printf("       SKIP mgt1.moe_union_m_curve: %s\n", rig.why.c_str());
        return;
    }
    const uint32_t layer = 0;
    const uint32_t dim = rig.config.text.hidden_size;
    {
        std::vector<std::string> names = store::pinned_global_tensors(rig.manifest);
        auto per = store::pinned_layer_tensors(rig.manifest, layer);
        names.insert(names.end(), per.begin(), per.end());
        auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io, names);
        if (!r) { std::printf("       SKIP: pinned: %s\n", r.error().str().c_str()); return; }
    }
    runtime::GpuMoeBridge bridge;
    REQUIRE_OK(bridge.create(rig.device, rig.alloc, gpu::default_shader_dir(), rig.store,
                             rig.planner, rig.pinned, rig.config.text));

    std::vector<float> x(size_t(kMaxM) * dim), y(size_t(kMaxM) * dim);
    std::vector<float> w(size_t(kMaxM) * kTopk);
    std::vector<uint32_t> ids_d(size_t(kMaxM) * kTopk), ids_o(size_t(kMaxM) * kTopk);
    for (uint32_t m = 0; m < kMaxM; ++m) {
        for (uint32_t s = 0; s < kTopk; ++s) {
            ids_d[m * kTopk + s] = 10 + m * kTopk + s;
            ids_o[m * kTopk + s] = (s < 3) ? (10 + s) : (10 + kTopk + m * 3 + (s - 3));
            w[m * kTopk + s] = 0.05f + 0.02f * float(s);
        }
        for (uint32_t i = 0; i < dim; ++i)
            x[size_t(m) * dim + i] = std::sin(0.01f * float((i + 37 * (m + 1)) % 977));
    }
    {
        std::vector<uint32_t> all;
        for (const auto* v : {&ids_d, &ids_o})
            for (uint32_t e : *v)
                if (std::find(all.begin(), all.end(), e) == all.end()) all.push_back(e);
        for (uint32_t e : all) {
            const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(e)};
            auto f = rig.planner.fetch(key, IoPriority::BlockingMiss, 1, layer);
            if (!f) { std::printf("       SKIP: fetch: %s\n", f.error().str().c_str()); return; }
        }
        rig.io.drain();
    }

    std::printf("    MoE(M) union vs per column, layer %u, %u iterations a case\n", layer, iters);
    struct Set { const char* what; const std::vector<uint32_t>* ids; };
    const Set sets[] = {{"disjoint", &ids_d}, {"overlapping", &ids_o}};
    for (const Set& st : sets) {
        double base = 0.0;
        for (uint32_t M : {1u, 2u, 4u, 6u}) {
            runtime::GpuMoeBridge::BatchCall call;
            call.layer = layer; call.m = M; call.ids = st.ids->data(); call.weights = w.data();
            call.topk = kTopk; call.x = x.data(); call.y = y.data(); call.hidden = dim;
            bool ran = true;
            auto time_it = [&](bool uni) -> double {
                for (uint32_t i = 0; i < 3 && ran; ++i)
                    ran = bool(uni ? bridge.run_batch_union(call) : bridge.run_batch(call));
                const auto t0 = Clock::now();
                for (uint32_t i = 0; i < iters && ran; ++i)
                    ran = bool(uni ? bridge.run_batch_union(call) : bridge.run_batch(call));
                return std::chrono::duration<double, std::milli>(Clock::now() - t0).count()
                       / double(iters);
            };
            const double per_col = time_it(false);
            const double uni = time_it(true);
            CHECK(ran);
            if (!ran) return;
            const uint32_t n_union = bridge.union_info().routed;
            if (M == 1) base = uni;
            std::printf("      %-12s M=%u  union %2u experts  per-column %7.3f ms  "
                        "union %7.3f ms (%6.3f ms/token, %.2fx M=1)  speedup %.2fx\n",
                        st.what, M, n_union, per_col, uni, uni / double(M),
                        base > 0 ? uni / base : 0.0, uni > 0 ? per_col / uni : 0.0);
        }
    }
}

// ============================================================================
// The DSpark verify batch: M tokens with DIFFERENT expert sets in one call
// (docs/p4_dspark_runtime.md §2.2). MoeRunner can express one expert set per
// dispatch -- `ids()` is `[slots]`, only `route_weights` has an `[m]` axis -- so
// GpuMoeBridge::run_batch runs one column a dispatch over the same activations
// buffer, staging x/act_quant once for all columns.
//
// What this checks is the equivalence the spec-decode loop depends on: column m
// of the batch is bit-for-bit the M = 1 result for token m with the same
// routing. Anything less and accepting a prefix of the batch would mean
// accepting tokens the M = 1 path would not have produced, which is exactly the
// design §10.2 invariant (temperature 0: speculation must not change the output).
//
// The kernel has no live-count column mask -- M is a specialisation constant and
// both shaders loop over all M columns -- so one dispatch recomputes every
// column of h and of y, and only the column whose x and routing-weight row the
// host fed is meaningful: dispatch m has to be paired with x[m],
// `route_weights()[m]` and y[m], all three. The `column pairing` block below
// pins that (docs/p4_dspark_runtime.md: the appendix added 2026-09-17).
// ============================================================================
DEEPMOE_TEST(gpu_moe, the_verify_batch_takes_one_expert_set_per_column) {
    if (skip_without_model("gpu_moe.the_verify_batch_takes_one_expert_set_per_column")) return;

    struct BatchRig {
        gpu::Device        device;
        gpu::MemoryAllocator alloc;
        Manifest           manifest;
        V41Config          config;
        ShardSet           shards;
        storage::IoEngine  io;
        store::PinnedStore pinned;
        ExpertStore        store;
        Planner            planner;
        bool io_started = false;
        std::string why;

        ~BatchRig() {
            if (io_started) io.stop();
        }

        bool bring_up(uint32_t slots) {
            const std::string dir = model_dir() ? model_dir() : "";
            if (auto r = device.create({}); !r) { why = r.error().str(); return false; }
            if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
            if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
                why = r.error().str(); return false;
            }
            auto cfg = V41Config::load(ShardSet::join(dir, "config.json"));
            if (!cfg) { why = cfg.error().str(); return false; }
            config = std::move(*cfg);
            auto mf = Manifest::load(ShardSet::join(dir, layout::kManifestFile));
            if (!mf) { why = mf.error().str(); return false; }
            manifest = std::move(*mf);
            if (auto r = shards.open_all(dir, manifest, /*unbuffered=*/true); !r) {
                why = r.error().str(); return false;
            }
            IoConfig iocfg;
            auto backend = storage::make_default_backend(iocfg);
            if (!backend) { why = backend.error().str(); return false; }
            if (auto r = io.start(std::move(*backend), iocfg); !r) {
                why = r.error().str(); return false;
            }
            io_started = true;
            auto pb = alloc.make_slab_backing();
            if (!pb) { why = pb.error().str(); return false; }
            store::PinnedConfig pc;
            pc.region_bytes = 1ull << 30;
            if (auto r = pinned.init(std::move(*pb), pc); !r) { why = r.error().str(); return false; }
            CacheConfig cache;
            cache.slots_per_slab = slots;
            cache.budget_bytes   = uint64_t(slots) * layout::kExpertSlotBytes;
            auto eb = alloc.make_slab_backing();
            if (!eb) { why = eb.error().str(); return false; }
            if (auto r = store.init(std::move(*eb), cache); !r) { why = r.error().str(); return false; }
            if (auto r = planner.init(store, io, manifest, shards, cache, PrefetchConfig{}); !r) {
                why = r.error().str(); return false;
            }
            return true;
        }
    };

    constexpr uint32_t kM = 3;                    // three tokens, three expert sets
    constexpr uint32_t kTopk = 6;
    BatchRig rig;
    if (!rig.bring_up(/*slots=*/kM * kTopk + 4)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const uint32_t layer = 0;
    const uint32_t dim = rig.config.text.hidden_size;

    // The shared expert and every other pinned tensor of this layer: the bridge
    // resolves `layers.<L>.ffn.shared_experts.*` through the pinned store.
    {
        std::vector<std::string> names = store::pinned_global_tensors(rig.manifest);
        auto per = store::pinned_layer_tensors(rig.manifest, layer);
        names.insert(names.end(), per.begin(), per.end());
        auto r = rig.pinned.load(rig.manifest, rig.shards, rig.io, names);
        if (!r) { std::printf("       SKIP gpu_moe: pinned: %s\n", r.error().str().c_str()); return; }
    }

    // Three different expert sets, and three different activations, both
    // deterministic. Disjoint ids per column so a column that read the wrong
    // routing could not accidentally agree.
    uint32_t ids[kM * kTopk];
    std::vector<float> x(size_t(kM) * dim);
    uint32_t seed = 0x12345678u;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return seed; };
    for (uint32_t m = 0; m < kM; ++m)
        for (uint32_t s = 0; s < kTopk; ++s) ids[m * kTopk + s] = 10 + m * kTopk + s;
    for (uint32_t m = 0; m < kM; ++m) {
        const uint32_t n = 37 * (m + 1);          // a different pattern per column
        for (uint32_t i = 0; i < dim; ++i) {
            const float u = float(rnd() >> 8) / float(1u << 24);
            x[size_t(m) * dim + i] = std::sin(0.01f * float((i + n) % 977)) * (0.5f + u);
        }
    }
    // Weights that are not all equal, so the reduction is exercised per column.
    std::vector<float> w(kM * kTopk);
    for (uint32_t m = 0; m < kM; ++m)
        for (uint32_t s = 0; s < kTopk; ++s) w[m * kTopk + s] = 0.05f + 0.02f * float(s) + 0.1f * float(m);

    // Every routed expert resident, the way the residency gate would have left
    // them before the batch's MoE dispatch.
    for (uint32_t i = 0; i < kM * kTopk; ++i) {
        const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(ids[i])};
        auto f = rig.planner.fetch(key, IoPriority::BlockingMiss, 1, layer);
        if (!f) { std::printf("       SKIP gpu_moe: fetch: %s\n", f.error().str().c_str()); return; }
    }
    rig.io.drain();
    for (uint32_t i = 0; i < kM * kTopk; ++i) {
        const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(ids[i])};
        if (!rig.store.resident(key)) {
            _ctx.fail(__FILE__, __LINE__,
                      std::format("expert ({}, {}) did not become resident", key.layer, key.expert));
            return;
        }
    }

    runtime::GpuMoeBridge bridge;
    REQUIRE_OK(bridge.create(rig.device, rig.alloc, gpu::default_shader_dir(), rig.store,
                             rig.planner, rig.pinned, rig.config.text));

    // --- (1) each token on its own, through the M = 1 interface --------------
    std::vector<float> one(size_t(kM) * dim);
    for (uint32_t m = 0; m < kM; ++m) {
        runtime::MoeCall call;
        call.layer   = layer;
        call.ids     = ids + m * kTopk;
        call.weights = w.data() + m * kTopk;
        call.topk    = kTopk;
        call.x       = x.data() + size_t(m) * dim;
        call.y       = one.data() + size_t(m) * dim;
        call.hidden  = dim;
        REQUIRE_OK(bridge.run(call));
    }

    // --- (2) the batch -------------------------------------------------------
    std::vector<float> batch(size_t(kM) * dim);
    runtime::GpuMoeBridge::BatchCall call;
    call.layer   = layer;
    call.m       = kM;
    call.ids     = ids;
    call.weights = w.data();
    call.topk    = kTopk;
    call.x       = x.data();
    call.y       = batch.data();
    call.hidden  = dim;
    REQUIRE_OK(bridge.run_batch(call));
    std::printf("       batch: %s\n", bridge.batch_timing().to_string().c_str());
    CHECK_EQ(bridge.batch_timing().columns, kM);

    // Snapshot the batch's answer AND the x it ran on, before any further call
    // restages the runner's x columns (the isolation runs below stage their own).
    const std::vector<float> batch_snapshot = batch;
    std::vector<uint16_t> x_after_batch(size_t(kM) * dim);
    std::memcpy(x_after_batch.data(), bridge.debug_x(), x_after_batch.size() * sizeof(uint16_t));

    // --- the column pairing: x[m], route_weights()[m] and y[m] --------------
    // Every column carries the SAME expert set, so the number of dispatches
    // cannot change anything, and only x and the routing weights vary by column.
    // The runner's y buffer is read COLUMN BY COLUMN here, which is what caught
    // the bug this test is about: `run_batch` used to copy `runner_.y()`
    // (column 0) into every column, and its x staging clobbered all six columns
    // with the current column's activation before each dispatch, so column m was
    // evaluated as (x[m], route_weights()[0]) while the M = 1 reference is
    // (x[m], route_weights()[m]). Reading y[2] after that gave the M = 1 answer
    // bit for bit while what run_batch handed back did not
    // (docs/p4_dspark_runtime.md: the appendix added 2026-09-17).
    {
        uint32_t eid[kM * kTopk];
        std::vector<float> ew(kM * kTopk);
        for (uint32_t m = 0; m < kM; ++m) {
            std::memcpy(eid + m * kTopk, ids, kTopk * sizeof(uint32_t));   // token 0's experts
            for (uint32_t s = 0; s < kTopk; ++s) ew[m * kTopk + s] = w[m * kTopk + s];
        }
        // The M = 1 reference for column m is the M = 1 run of that column's own
        // x and its own routing weights against the same experts.
        std::vector<float> ref(size_t(kM) * dim);
        for (uint32_t m = 0; m < kM; ++m) {
            runtime::MoeCall c;
            c.layer = layer; c.ids = eid; c.weights = ew.data() + m * kTopk;
            c.topk = kTopk; c.x = x.data() + size_t(m) * dim;
            c.y = ref.data() + size_t(m) * dim; c.hidden = dim;
            REQUIRE_OK(bridge.run(c));
        }
        runtime::GpuMoeBridge::BatchCall e;
        e.layer = layer; e.m = kM; e.ids = eid; e.weights = ew.data();
        e.topk = kTopk; e.x = x.data(); e.hidden = dim;
        std::vector<float> ey(size_t(kM) * dim);
        e.y = ey.data();
        REQUIRE_OK(bridge.run_batch(e));
        const float* ycol = bridge.y_host();
        for (uint32_t m = 0; m < kM; ++m) {
            double d_y = 0.0, d_out = 0.0;
            for (uint32_t i = 0; i < dim; ++i) {
                d_y   = std::fmax(d_y,   std::fabs(double(ycol[size_t(m) * dim + i])
                                                    - double(ref[size_t(m) * dim + i])));
                d_out = std::fmax(d_out, std::fabs(double(ey[size_t(m) * dim + i])
                                                    - double(ref[size_t(m) * dim + i])));
            }
            std::printf("       column pairing %u (same experts, per-column x+w): "
                        "runner y[%u] vs M=1 %.3e | run_batch's column %.3e\n",
                        m, m, d_y, d_out);
            // Both must be exact: y[m] IS the M = 1 result for this column's x
            // and routing weights, and run_batch has to hand that column back.
            CHECK(d_y   <= 1e-7);
            CHECK(d_out <= 1e-7);
        }
    }

    // --- (2d) the isolation runs ---------------------------------------------
    // (a) all columns identical   -> the batch collapses to the M = 1 answer;
    // (b) same x, different weights -> the per-column weights really are read
    //     per column (zero spread here is the bug, not a pass);
    // (c) same weights, different x -> the per-column activations really are.
    {
        struct Iso { const char* what; bool per_x; bool per_w; };
        const Iso isos[] = {{"identical columns", false, false},
                            {"per-column weights only", false, true},
                            {"per-column x only", true, false}};
        for (const Iso& iso : isos) {
            std::vector<float> ix(size_t(kM) * dim);
            std::vector<float> iw(kM * kTopk);
            uint32_t iid[kM * kTopk];
            for (uint32_t m = 0; m < kM; ++m) {
                std::memcpy(ix.data() + size_t(m) * dim,
                            iso.per_x ? x.data() + size_t(m) * dim : x.data(),
                            size_t(dim) * sizeof(float));
                std::memcpy(iid + m * kTopk, ids, kTopk * sizeof(uint32_t));
                for (uint32_t s = 0; s < kTopk; ++s)
                    iw[m * kTopk + s] = iso.per_w ? w[m * kTopk + s] : w[s];
            }
            runtime::GpuMoeBridge::BatchCall ci;
            ci.layer = layer; ci.m = kM; ci.ids = iid; ci.weights = iw.data();
            ci.topk = kTopk; ci.x = ix.data(); ci.hidden = dim;
            std::vector<float> yi(size_t(kM) * dim);
            ci.y = yi.data();
            REQUIRE_OK(bridge.run_batch(ci));
            double worst = 0.0;
            for (uint32_t m = 1; m < kM; ++m)
                for (uint32_t i = 0; i < dim; ++i)
                    worst = std::fmax(worst, std::fabs(double(yi[i]) - double(yi[size_t(m) * dim + i])));
            double vs1 = 0.0;
            for (uint32_t i = 0; i < dim; ++i)
                vs1 = std::fmax(vs1, std::fabs(double(one[i]) - double(yi[i])));
            std::printf("       %-26s columns spread %.3e, col 0 vs the M=1 run %.3e\n",
                        iso.what, worst, vs1);
            // Column 0 is token 0's M = 1 result in all three configurations, and
            // a per-column axis only counts if it actually moves the answer.
            CHECK(vs1 <= 1e-7);
            if (iso.per_x || iso.per_w) CHECK(worst > 0.0);
        }
    }

    for (uint32_t m = 0; m < kM; ++m) {
        std::vector<float> onec(dim);
        runtime::GpuMoeBridge::BatchCall c1;
        c1.layer = layer; c1.m = 1;
        c1.ids = ids + m * kTopk; c1.weights = w.data() + m * kTopk;
        c1.topk = kTopk; c1.x = x.data() + size_t(m) * dim;
        c1.y = onec.data(); c1.hidden = dim;
        REQUIRE_OK(bridge.run_batch(c1));
        double d_m1 = 0.0, d_b = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            d_m1 = std::fmax(d_m1, std::fabs(one[size_t(m) * dim + i] - onec[i]));
            d_b  = std::fmax(d_b,  std::fabs(batch_snapshot[size_t(m) * dim + i] - onec[i]));
        }
        std::printf("       column %u: run_batch(M=1) vs M=1 %.3e | batch(M=%u) vs run_batch(M=1) %.3e\n",
                    m, d_m1, kM, d_b);
    }
    // The x each column of the batch ran on, against the host quantisation of
    // what `stage_batch` staged for it: column m must hold token m's activation,
    // or the column-pairing check above is not testing what it says.
    for (uint32_t m = 0; m < kM; ++m) {
        std::vector<uint16_t> q(dim);
        std::vector<float> scratch(dim);
        runtime::debug_act_quant_to_fp16(x.data() + size_t(m) * dim, q.data(),
                                         scratch.data(), dim);
        uint32_t bad = 0;
        for (uint32_t i = 0; i < dim; ++i)
            if (x_after_batch[size_t(m) * dim + i] != q[i]) ++bad;
        std::printf("       x column %u ran on: %u/%u words differ from the staged x\n",
                    m, bad, dim);
    }


    // --- the equivalence -----------------------------------------------------
    double worst = 0.0;
    for (uint32_t m = 0; m < kM; ++m) {
        double d = 0.0, ymax = 0.0;
        for (uint32_t i = 0; i < dim; ++i) {
            const double a = one[size_t(m) * dim + i], b = batch[size_t(m) * dim + i];
            d = std::fmax(d, std::fabs(a - b));
            ymax = std::fmax(ymax, std::fabs(a));
        }
        const double rel = ymax > 0 ? d / ymax : d;
        std::printf("       column %u: max |batch - one| %.3e (%.2e of |y|max)\n", m, d, rel);
        worst = std::fmax(worst, rel);
    }
    CHECK(worst <= 1e-7);      // the same code over the same buffers: bit-identical

    // And the columns really are different, or the check above proves nothing.
    double spread = 0.0, ymax = 0.0;
    for (uint32_t i = 0; i < dim; ++i) ymax = std::fmax(ymax, std::fabs(double(batch[i])));
    for (uint32_t m = 1; m < kM; ++m)
        for (uint32_t i = 0; i < dim; ++i)
            spread = std::fmax(spread, std::fabs(double(batch[size_t(m) * dim + i]) - batch[i]));
    std::printf("       columns differ by %.3f of |y|max\n", spread / std::max(ymax, 1e-30));
    CHECK(spread / std::max(ymax, 1e-30) > 0.1);
}

// design §7.9's fp8 shared expert, through the same two dispatches as the FP4
// routed ones: one slot of the list carries kSlotFp8 and the kernel switches
// weight format, scale layout and byte stride on it.
DEEPMOE_TEST(gpu_moe, the_fp8_shared_expert_runs_in_the_same_dispatches) {
    if (skip_without_model("gpu_moe.the_fp8_shared_expert_runs_in_the_same_dispatches")) return;

    auto golden = load_tri(data_path("l1_shared_layer0.bin"), "DMS1", /*has_expert=*/false,
                           /*extra_u64=*/6);
    if (!golden) {
        std::printf("       SKIP gpu_moe: %s (run tools/oracle_shared.py --shared 0)\n",
                    golden.error().str().c_str());
        return;
    }
    const TriGolden& g = *golden;

    Rig rig;
    if (!rig.bring_up(/*slots=*/1)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    SharedExpert shared;
    if (!shared.load(rig, g.layer)) {
        _ctx.fail(__FILE__, __LINE__, "shared expert: " + shared.why);
        return;
    }
    std::printf("       shared expert layer %u: %llu B of values + %llu B of scales per matrix, "
                "part address mod 16 = %llu\n", g.layer,
                static_cast<unsigned long long>(shared.bytes[0]),
                static_cast<unsigned long long>(shared.bytes[1]),
                static_cast<unsigned long long>(shared.addr[0] % 16));

    const uint32_t experts_per_layer = layout::kRoutedExperts + 1;
    struct Case { gpu::MoeSpec spec; const char* what; double tol; };
    const Case cases[] = {
        {{1, 32, 32, 0, 0, 1, 0, 0, 1}, "M=1, global x", 1e-3},
        {{1, 16, 32, 0, 0, 2, 0, 0, 1}, "M=1, 16 lanes, 2 rows", 1e-3},
        {{6, 16, 32, 0, 0, 2, 1, 0, 1}, "M=6, LDS tile (fp8 keeps global x)", 1e-3},
        {{1, 32, 32, 0, 0, 1, 0, 1, 1}, "M=1, h fp8-quantised in dispatch B", 1e-3},
    };
    for (const Case& c : cases) {
        gpu::MoeDims dims;
        dims.layer             = g.layer;
        dims.slots             = 1;
        dims.experts_per_layer = experts_per_layer;
        dims.fp8_slot_count    = 1;
        gpu::MoeRunner runner;
        auto ok = runner.create(rig.device, rig.alloc, gpu::default_shader_dir(), c.spec, dims);
        if (!ok) {
            _ctx.fail(__FILE__, __LINE__, std::format("{}: {}", c.spec.name(), ok.error().str()));
            continue;
        }
        fill_table_with_shared(runner, rig.store, g.layer, experts_per_layer, &shared);
        runner.ids()[0] = (experts_per_layer - 1) | gpu::kSlotFp8;
        runner.slot_list()[0] = 0;
        runner.set_list_count(1);
        for (uint32_t i = 0; i < c.spec.m; ++i) runner.route_weights()[i] = 1.0f;
        for (uint32_t m = 0; m < c.spec.m; ++m)
            for (uint32_t i = 0; i < g.dim; ++i)
                runner.x_fp16()[m * g.dim + i] = cpu::float_to_fp16(g.x[i]);
        REQUIRE_OK(runner.run(1));
        std::vector<float> y(g.dim);
        std::memcpy(y.data(), runner.y(), y.size() * sizeof(float));

        const std::vector<float>& want = c.spec.h_quant ? g.y_hq16 : g.y_ref;
        const Compare cmp = compare(y, want);
        std::printf("       %-36s %-40s cos %.9f  %.3e of |y|max\n",
                    c.spec.name().c_str(), c.what, cmp.cosine, cmp.rel_to_scale);
        CHECK(cmp.rel_to_scale <= c.tol);
        CHECK(cmp.cosine >= 1.0 - 1e-6);
    }

    // The bytes really are 2x a routed expert's: the effective-GB/s denominator
    // in bench/kernel_bench depends on this.
    CHECK_EQ(shared.bytes[0], 11796480ull);
    CHECK_EQ(shared.bytes[1], 11520ull);
}

// design §7.9 v0.6: `silu(gate)*up` makes an fp8 E4M3 round trip with a UE8M0
// block-32 scale before w2. Two implementations must agree with each other and
// with tools/oracle_shared.py's y_hq, and both must move AWAY from the
// unquantised l1_*.bin golden -- that vector is the v0.5 answer.
DEEPMOE_TEST(gpu_moe, the_fp8_h_quantisation_matches_the_reference) {
    if (skip_without_model("gpu_moe.the_fp8_h_quantisation_matches_the_reference")) return;

    auto tri = load_tri(data_path("l1q_layer0_expert0.bin"), "DMQ1", /*has_expert=*/true);
    if (!tri) {
        std::printf("       SKIP gpu_moe: %s (run tools/oracle_shared.py --expert 0:0)\n",
                    tri.error().str().c_str());
        return;
    }
    const TriGolden& g = *tri;

    Rig rig;
    if (!rig.bring_up(/*slots=*/1)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    const ExpertKey key{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(g.expert)};
    REQUIRE(rig.fill(key));

    // L16 R2 owns 32 rows of h per workgroup, which is what HQuant = 2 needs to
    // own a whole fp8 block; L32 R1 owns 8 and is rejected at create() time.
    // The tolerance against y_hq16 is 1e-3 for the fp32-accumulating variants
    // and looser for the packed-fp16 one, for a reason worth writing down: the
    // quantiser amplifies. xgf16's h differs from the fp32 path's by 4.5e-4,
    // and `fast_round_scale` turns that into 3.2e-3 on y whenever a block's
    // amax crosses a power of two. It is the same 7x amplification y_hq16 was
    // introduced to account for, one level down.
    struct Case { gpu::MoeSpec spec; const char* what; double tol = 1e-3; };
    const Case cases[] = {
        {{1, 32, 32, 0, 0, 1, 0, 0}, "no quantisation (design v0.5)"},
        {{1, 32, 32, 0, 0, 1, 0, 1}, "fp8 round trip inside dispatch B"},
        {{1, 16, 32, 0, 0, 2, 0, 1}, "fp8 round trip inside dispatch B, 32-row groups"},
        {{1, 16, 32, 0, 0, 2, 0, 2}, "fp8 h written by dispatch A"},
        {{1, 16, 32, 0, 0, 2, 1, 2}, "fp8 h written by dispatch A, LDS x tile"},
        {{6, 16, 32, 0, 0, 2, 1, 2}, "M=6, fp8 h written by dispatch A"},
        {{6, 16, 32, 0, 0, 2, 4, 2}, "M=6, fp8 h, packed fp16 gate/up", 5e-3},
        // HQuant 3 is the same arithmetic as HQuant 2 moved into its own
        // dispatch, so it must land on the same answer -- and, unlike 2, it
        // must do so on the L32 R1 shape that owns only 8 rows of h per
        // workgroup (docs/kernel_p2_moe.md §8 item 1).
        {{1, 32, 32, 0, 0, 1, 0, 3}, "fp8 h by the third dispatch, L32 R1"},
        {{1, 16, 32, 0, 0, 2, 0, 3}, "fp8 h by the third dispatch, L16 R2"},
        {{1, 64, 32, 0, 0, 1, 0, 3}, "fp8 h by the third dispatch, 64 lanes"},
        {{6, 32, 32, 0, 0, 1, 0, 3}, "M=6, fp8 h by the third dispatch"},
        {{6, 16, 32, 0, 0, 2, 4, 3}, "M=6, fp8 h by the third dispatch, packed "
                                     "fp16 gate/up", 5e-3},
    };
    std::vector<float> first_quantised;
    for (const Case& c : cases) {
        auto y = run_variant(rig, c.spec, key, g.x, /*slots=*/1, /*slot_of_interest=*/0);
        if (!y) {
            _ctx.fail(__FILE__, __LINE__, std::format("{}: {}", c.spec.name(), y.error().str()));
            continue;
        }
        std::vector<float> col0(y->begin(), y->begin() + g.dim);
        const Compare vs_ref  = compare(col0, g.y_ref);
        const Compare vs_hq   = compare(col0, g.y_hq);
        const Compare vs_hq16 = compare(col0, g.y_hq16);
        std::printf("       %-36s %-44s vs y_ref %.3e  vs y_hq %.3e  vs y_hq16 %.3e\n",
                    c.spec.name().c_str(), c.what, vs_ref.rel_to_scale,
                    vs_hq.rel_to_scale, vs_hq16.rel_to_scale);
        if (c.spec.h_quant == 0) {
            // Without the quantisation the kernel matches the *unquantised*
            // answer and is far from the reference's.
            CHECK(vs_ref.rel_to_scale <= 1e-3);
            CHECK(vs_hq.rel_to_scale > 5e-3);
        } else {
            // With it, the two swap: this is the "moves towards the reference"
            // claim, as a bound rather than a statement. The comparand is
            // y_hq16 -- the reference's quantiser fed the fp16 numbers design
            // §6 says the kernel carries -- because `fast_round_scale` is a
            // step function and y_hq's own inputs are fp32.
            CHECK(vs_hq16.rel_to_scale <= c.tol);
            CHECK(vs_ref.rel_to_scale > 5e-3);
            CHECK(vs_hq.rel_to_scale < vs_ref.rel_to_scale);
            // HQuant 1 and 2 quantise the same fp16 number with the same rule,
            // so they must agree far more tightly than either agrees with the
            // oracle.
            if (c.tol <= 1e-3) {
                if (first_quantised.empty()) first_quantised = col0;
                else CHECK(compare(col0, first_quantised).rel_to_scale <= 1e-4);
            }
            // Every column of a verify batch gets the same treatment.
            for (uint32_t m = 1; m < c.spec.m; ++m) {
                std::vector<float> colm(y->begin() + size_t(m) * g.dim,
                                        y->begin() + size_t(m + 1) * g.dim);
                CHECK(compare(colm, col0).rel_to_scale <= 1e-6);
            }
        }
    }
}

// design §7.9 "compute the experts that arrived first": running the slot list
// in two pieces, the second with accumulate set, must reproduce the one-shot
// answer. fp32 addition is not associative and the per-lane partials are summed
// in a different order, so the claim is a bound, not bit-equality -- the test
// measures which it is.
DEEPMOE_TEST(gpu_moe, a_partial_dispatch_reduces_to_the_same_y) {
    if (skip_without_model("gpu_moe.a_partial_dispatch_reduces_to_the_same_y")) return;

    auto golden = load_golden(data_path("l1_layer0_expert0.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;

    constexpr uint32_t kSlots = 7;   // 6 routed + the shared-expert stand-in
    Rig rig;
    if (!rig.bring_up(kSlots)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    for (uint32_t e = 0; e < kSlots; ++e)
        REQUIRE(rig.fill(ExpertKey{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(e)}));

    struct Case { gpu::MoeSpec spec; uint32_t split; };
    const Case cases[] = {
        {{1, 32, 32, 0, 0, 1, 0}, 3},
        {{6, 16, 32, 0, 0, 2, 4}, 3},
        {{6, 16, 32, 0, 0, 2, 4, 2}, 4},   // with fp8 h, which dispatch A writes
        {{1, 32, 32, 0, 0, 1, 0, 3}, 3},   // with fp8 h from the third dispatch
        {{1, 32, 32, 0, 0, 1, 0}, 1},
        {{1, 32, 32, 0, 0, 1, 0}, 6},
    };
    for (const Case& c : cases) {
        gpu::MoeDims dims;
        dims.layer = g.layer;
        dims.slots = kSlots;
        gpu::MoeRunner runner;
        auto ok = runner.create(rig.device, rig.alloc, gpu::default_shader_dir(), c.spec, dims);
        if (!ok) {
            _ctx.fail(__FILE__, __LINE__, std::format("{}: {}", c.spec.name(), ok.error().str()));
            continue;
        }
        auto setup = [&]() {
            std::memcpy(runner.pointer_table(), rig.store.pointer_table(),
                        rig.store.pointer_table_bytes());
            for (uint32_t s = 0; s < kSlots; ++s) runner.ids()[s] = s;
            for (uint32_t i = 0; i < c.spec.m * kSlots; ++i)
                runner.route_weights()[i] = 0.5f + 0.1f * float(i % kSlots);
            for (uint32_t m = 0; m < c.spec.m; ++m)
                for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
                    runner.x_fp16()[m * layout::kHiddenSize + i] =
                        cpu::float_to_fp16(g.x[i] * (1.0f + 0.1f * float(m)));
        };

        setup();
        for (uint32_t s = 0; s < kSlots; ++s) runner.slot_list()[s] = s;
        runner.set_list_count(kSlots);
        runner.set_accumulate(false);
        REQUIRE_OK(runner.run(1));
        std::vector<float> once(size_t(c.spec.m) * layout::kHiddenSize);
        std::memcpy(once.data(), runner.y(), once.size() * sizeof(float));

        // Now the same seven slots as two dispatch pairs. Dispatch A is run for
        // each subset too, so this is exactly the "the first three arrived,
        // start on them" schedule of design §7.9.
        setup();
        std::memset(runner.y(), 0, once.size() * sizeof(float));
        for (uint32_t phase = 0; phase < 2; ++phase) {
            const uint32_t lo = phase ? c.split : 0;
            const uint32_t hi = phase ? kSlots : c.split;
            for (uint32_t i = lo; i < hi; ++i) runner.slot_list()[i - lo] = i;
            runner.set_list_count(hi - lo);
            runner.set_accumulate(phase != 0);
            REQUIRE_OK(runner.run(1));
        }
        std::vector<float> split(once.size());
        std::memcpy(split.data(), runner.y(), split.size() * sizeof(float));

        size_t exact = 0;
        for (size_t i = 0; i < once.size(); ++i) exact += (once[i] == split[i]) ? 1 : 0;
        const Compare cmp = compare(split, once);
        std::printf("       %-32s %u + %u slots: %zu/%zu words bit-identical, "
                    "max|dy| %.3e of |y|max\n", c.spec.name().c_str(), c.split,
                    kSlots - c.split, exact, once.size(), cmp.rel_to_scale);
        // The reduction is re-associated across the split, so a handful of ULPs
        // is expected; anything above that would mean a slot was dropped or
        // counted twice.
        CHECK(cmp.rel_to_scale <= 1e-6);
        CHECK(cmp.cosine >= 1.0 - 1e-12);

        // The schedule docs/kernel_p2_moe.md §6.3 recommends instead: split
        // only dispatch A -- whose work is exactly proportional to the slots it
        // is given -- and run dispatch B once, at the end, over the whole list.
        // Dispatch B then does the *same* reduction in the *same* order as the
        // one-shot run, so this is bit-identical rather than 1-2 ULP away, and
        // it is what makes "compute the experts that arrived first" nearly
        // free (§6.2's +0.193 ms is almost all the second dispatch B).
        setup();
        std::memset(runner.y(), 0, once.size() * sizeof(float));
        for (uint32_t phase = 0; phase < 2; ++phase) {
            const uint32_t lo = phase ? c.split : 0;
            const uint32_t hi = phase ? kSlots : c.split;
            for (uint32_t i = lo; i < hi; ++i) runner.slot_list()[i - lo] = i;
            runner.set_list_count(hi - lo);
            runner.set_accumulate(false);
            REQUIRE_OK(runner.run(1, gpu::MoePhase::GateUpOnly));
        }
        for (uint32_t i = 0; i < kSlots; ++i) runner.slot_list()[i] = i;
        runner.set_list_count(kSlots);
        runner.set_accumulate(false);
        REQUIRE_OK(runner.run(1, gpu::MoePhase::DownOnly));
        std::vector<float> deferred(once.size());
        std::memcpy(deferred.data(), runner.y(), deferred.size() * sizeof(float));
        size_t same = 0;
        for (size_t i = 0; i < once.size(); ++i) same += (once[i] == deferred[i]) ? 1 : 0;
        std::printf("       %-32s split A + one B: %zu/%zu words bit-identical\n",
                    c.spec.name().c_str(), same, once.size());
        CHECK_EQ(same, once.size());
    }
}

// Track R1 (docs/p4_hitrate.md 4): the decode loop's "compute the experts that
// arrived first". Dispatch A over the resident slots goes out in ONE submit
// (through the alternate slot list), the late slots' A and the one dispatch B in
// the NEXT -- with the main list rewritten in between, as the engine does. y must
// be bit-identical to the one-shot run, for every early/late partition shape.
DEEPMOE_TEST(gpu_moe, gateup_split_across_submits_is_bit_identical) {
    if (skip_without_model("gpu_moe.gateup_split_across_submits_is_bit_identical")) return;
    auto golden = load_golden(data_path("l1_layer0_expert0.bin"));
    REQUIRE_OK(golden);
    const Golden& g = *golden;
    constexpr uint32_t kSlots = 7;
    Rig rig;
    if (!rig.bring_up(kSlots)) {
        std::printf("       SKIP gpu_moe: %s\n", rig.why.c_str());
        return;
    }
    for (uint32_t e = 0; e < kSlots; ++e)
        REQUIRE(rig.fill(ExpertKey{static_cast<uint16_t>(g.layer), static_cast<uint16_t>(e)}));

    const gpu::MoeSpec spec{1, 32, 32, 0, 0, 1, 0, 3};   // the decode specialisation
    gpu::MoeDims dims;
    dims.layer = g.layer;
    dims.slots = kSlots;
    gpu::MoeRunner runner;
    REQUIRE_OK(runner.create(rig.device, rig.alloc, gpu::default_shader_dir(), spec, dims));
    auto setup = [&]() {
        std::memcpy(runner.pointer_table(), rig.store.pointer_table(), rig.store.pointer_table_bytes());
        for (uint32_t s = 0; s < kSlots; ++s) runner.ids()[s] = s;
        for (uint32_t i = 0; i < kSlots; ++i) runner.route_weights()[i] = 0.5f + 0.1f * float(i);
        for (uint32_t i = 0; i < layout::kHiddenSize; ++i)
            runner.x_fp16()[i] = cpu::float_to_fp16(g.x[i]);
        for (uint32_t s = 0; s < kSlots; ++s) runner.slot_list()[s] = s;
        runner.set_list_count(kSlots);
        runner.set_accumulate(false);
    };
    setup();
    REQUIRE_OK(runner.run(1));
    std::vector<float> once(layout::kHiddenSize);
    std::memcpy(once.data(), runner.y(), once.size() * sizeof(float));

    gpu::CommandPool pool;
    REQUIRE_OK(pool.create(rig.device));
    auto cb = pool.acquire();
    REQUIRE_OK(cb);
    gpu::CommandBuffer cmd = *cb;
    // early / late partitions: late experts anywhere, the shared slot (6) early.
    const std::vector<std::vector<uint32_t>> lates = {{0}, {5}, {1, 3}, {0, 1, 2, 3, 4, 5}, {2, 4, 5}};
    for (const auto& late : lates) {
        std::vector<uint32_t> early;
        for (uint32_t s = 0; s < kSlots; ++s)
            if (std::find(late.begin(), late.end(), s) == late.end()) early.push_back(s);
        setup();
        std::memset(runner.y(), 0, once.size() * sizeof(float));
        std::memcpy(runner.slot_list_alt(), early.data(), early.size() * sizeof(uint32_t));
        REQUIRE_OK(cmd.begin());
        REQUIRE_OK(runner.record_gateup_alt(cmd, static_cast<uint32_t>(early.size())));
        REQUIRE_OK(cmd.end());
        REQUIRE_OK(gpu::submit_and_wait(rig.device, cmd));
        std::memcpy(runner.slot_list_alt(), late.data(), late.size() * sizeof(uint32_t));
        REQUIRE_OK(cmd.begin());
        REQUIRE_OK(runner.record_gateup_alt(cmd, static_cast<uint32_t>(late.size())));
        REQUIRE_OK(runner.record_into(cmd, gpu::MoePhase::DownOnly));
        REQUIRE_OK(cmd.end());
        REQUIRE_OK(gpu::submit_and_wait(rig.device, cmd));
        size_t same = 0;
        for (size_t i = 0; i < once.size(); ++i) same += (once[i] == runner.y()[i]) ? 1 : 0;
        std::printf("       %zu early + %zu late across two submits: %zu/%zu words bit-identical\n",
                    early.size(), late.size(), same, once.size());
        CHECK_EQ(same, once.size());
    }
    cmd = gpu::CommandBuffer{};
    pool.destroy();
}
