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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "cpu/dequant.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
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

    struct Case { gpu::MoeSpec spec; const char* what; };
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
        CHECK(cmp.rel_to_scale <= 1e-3);
        CHECK(cmp.cosine >= 1.0 - 1e-6);
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
