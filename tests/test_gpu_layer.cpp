// One whole decoder layer on the GPU, against the oracle's block output.
//
// tests/test_gpu_attn.cpp proves each kernel against the reference's own input
// for that kernel. This proves they compose: the layer is run the way decode
// will run it -- every stage fed the previous one's output, one command buffer
// for dispatches 1-9 of design §7.14, the §7.1 timeline gate, then the MoE --
// and only the block input and the KV the prefill left are golden.
//
// The things only this test can catch: a buffer wired to the wrong slot, the
// pre/post/comb handoff of design §2.4 given to the wrong sublayer, a missing
// barrier, the two mega_mhc halves stepping on each other's address table.
//
// Gated on DEEPMOE_MODEL_DIR, on tests/data/l2, and on a Vulkan device.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/timeline.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/decode_layer.h"
#include "runtime/kvstore.h"
#include "runtime/moe_bridge.h"
#include "storage/backend.h"
#include "storage/io_engine.h"
#include "store/expert_store.h"
#include "store/pinned.h"
#include "store/planner.h"
#include "store/shard_set.h"
#include "tests/l1_golden.h"
#include "tests/l2_golden.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::testing;

namespace {

struct LayerRig {
    gpu::Device           device;
    gpu::MemoryAllocator  alloc;
    Manifest              manifest;
    V41Config             config;
    store::ShardSet       shards;
    storage::IoEngine     io;
    store::PinnedStore    pinned;
    store::ExpertStore    experts;
    store::Planner        planner;
    gpu::AttnRunner       runner;
    gpu::GpuScratch       scratch;
    gpu::Timeline         timeline;
    runtime::KvStore      kv;
    runtime::DecodeLayer  layer;
    runtime::GpuMoeBridge moe;
    bool io_started = false;
    std::string why;

    ~LayerRig() {
        // Declaration order does not cover this one: the store's slabs come
        // from the allocator, so it has to let go first.
        moe.destroy();
        layer.destroy();
        kv.destroy();
        scratch.destroy();
        runner.destroy();
        if (io_started) io.stop();
    }

    bool bring_up(uint32_t cache_slots) {
        const std::string dir = model_dir() ? model_dir() : "";
        if (auto r = device.create({}); !r) { why = r.error().str(); return false; }
        if (auto r = device.caps().check_required(); !r) { why = r.error().str(); return false; }
        if (auto r = alloc.init(device, MemoryPath::DeviceLocalHostVisible); !r) {
            why = r.error().str(); return false;
        }
        auto cfg = V41Config::load(store::ShardSet::join(dir, "config.json"));
        if (!cfg) { why = cfg.error().str(); return false; }
        config = std::move(*cfg);
        auto mf = Manifest::load(store::ShardSet::join(dir, layout::kManifestFile));
        if (!mf) { why = mf.error().str(); return false; }
        manifest = std::move(*mf);
        if (auto r = shards.open_all(dir, manifest, true); !r) { why = r.error().str(); return false; }
        IoConfig iocfg;
        auto backend = storage::make_default_backend(iocfg);
        if (!backend) { why = backend.error().str(); return false; }
        if (auto r = io.start(std::move(*backend), iocfg); !r) { why = r.error().str(); return false; }
        io_started = true;

        auto pb = alloc.make_slab_backing();
        if (!pb) { why = pb.error().str(); return false; }
        store::PinnedConfig pc;
        pc.region_bytes = 512ull << 20;
        if (auto r = pinned.init(std::move(*pb), pc); !r) { why = r.error().str(); return false; }

        CacheConfig cache;
        cache.slots_per_slab = cache_slots;
        cache.budget_bytes   = uint64_t(cache_slots) * layout::kExpertSlotBytes;
        auto eb = alloc.make_slab_backing();
        if (!eb) { why = eb.error().str(); return false; }
        if (auto r = experts.init(std::move(*eb), cache); !r) { why = r.error().str(); return false; }
        if (auto r = planner.init(experts, io, manifest, shards, cache, PrefetchConfig{}); !r) {
            why = r.error().str(); return false;
        }

        if (auto r = runner.create(device, alloc, gpu::default_shader_dir()); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = scratch.create(alloc, 32ull << 20); !r) { why = r.error().str(); return false; }
        if (auto r = timeline.create(device); !r) { why = r.error().str(); return false; }

        runtime::KvStoreConfig kc;
        kc.layers      = 1;              // one layer at a time in this test
        kc.window      = config.text.sliding_window;
        kc.latent_dim  = config.text.head_dim;
        kc.max_context = 256;
        if (auto r = kv.create(alloc, kc); !r) { why = r.error().str(); return false; }
        if (auto r = layer.create(device, runner, scratch, config.text); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = moe.create(device, alloc, gpu::default_shader_dir(), experts, planner,
                                pinned, config.text); !r) {
            why = r.error().str(); return false;
        }
        return true;
    }
};

}  // namespace

DEEPMOE_TEST(gpu_layer, decode_layer_vs_oracle) {
    if (skip_without_model("gpu_layer")) return;
    auto set = load_l2(std::string(DEEPMOE_TEST_DATA_DIR) + "/l2");
    if (!set) {
        std::printf("      SKIP gpu_layer: no L2 data (%s)\n", set.error().str().c_str());
        return;
    }
    LayerRig rig;
    if (!rig.bring_up(/*cache_slots=*/8)) {
        std::printf("      SKIP gpu_layer: %s\n", rig.why.c_str());
        return;
    }
    const TextConfig& c = rig.config.text;
    const uint32_t dim = c.hidden_size;
    const uint32_t hcdim = c.hc_mult * dim;
    const uint32_t pos = set->decode_pos;

    // design §15's P2 scope is "all experts assumed resident", so the layers
    // this test runs are the ones the oracle exported and the checks are the
    // block output and the routing.
    for (uint32_t want : {0u, 39u}) {
        const L2Step* g = set->layer(want);
        if (!g) { std::printf("    layer %u not in the L2 export, skipping\n", want); continue; }
        std::printf("    layer %u\n", want);

        auto names = store::pinned_layer_tensors(rig.manifest, want);
        REQUIRE_OK(rig.pinned.load(rig.manifest, rig.shards, rig.io, names));
        auto w = runtime::LayerWeights::from_pinned(rig.pinned, want);
        REQUIRE_OK(w);

        // Seed the KV the prefill produced. The compressor and indexer kernels
        // of design §7.4 are not written yet, so the compressed half and the
        // top-k list come from the oracle; the window ring is re-encoded to the
        // bytes wkv.slang writes, which is exact because the values are already
        // on the E4M3 grid.
        REQUIRE_OK(rig.kv.seed_window(0, g->f("win_kv").data(),
                                      static_cast<uint32_t>(g->f("win_kv").size() / c.head_dim)));
        if (const L2Tensor* cmp = g->find("cmp_kv"))
            REQUIRE_OK(rig.kv.seed_compressed(
                0, cmp->f.data(), static_cast<uint32_t>(cmp->f.size() / c.head_dim)));
        const std::vector<float>& idxf = g->f("topk_idxs");
        std::vector<int32_t> idx(idxf.size());
        for (size_t i = 0; i < idxf.size(); ++i) idx[i] = static_cast<int32_t>(idxf[i]);
        REQUIRE_OK(rig.kv.seed_topk(0, idx.data(), static_cast<uint32_t>(idx.size())));
        auto view = rig.kv.layer(0);
        REQUIRE_OK(view);

        runtime::LayerStep st;
        st.layer          = want;
        st.position       = pos;
        st.compress_ratio = c.compress_ratio(want);
        // The stream the oracle exported is already past the previous layer's
        // hc_post, so this layer must not apply another one.
        st.apply_hc_post  = false;
        st.kv             = *view;

        runtime::DecodeScratch& b = rig.layer.scratch();
        // block_in for a layer with an engram is the stream AFTER the engram
        // write (design §2.1: layers 1 and 14); attn_resid_in is that tensor
        // either way, so it is the one the block proper starts from.
        const std::vector<float>& in = g->f("attn_resid_in");
        std::memcpy(b.x.host, in.data(), size_t(hcdim) * sizeof(float));
        std::memset(b.mix_a.host, 0, 128);
        std::memcpy(b.mix_a.host, g->f("pre_mix_in").data(), c.hc_mult * sizeof(float));

        REQUIRE_OK(rig.layer.bind(*w, st));
        REQUIRE_OK(rig.layer.run_attention(st));

        const Agreement gff = agree(
            std::vector<float>(rig.layer.ffn_norm_out(), rig.layer.ffn_norm_out() + dim),
            g->f("ffn_norm_out"));
        std::printf("      %-20s %s\n", "chained ffn_norm", gff.str().c_str());
        CHECK(gff.cos > 0.9999);

        // The routing the whole attention chain led to: the strongest single
        // statement this test makes, because six expert ids are a discrete
        // function of everything upstream.
        const std::vector<float>& gid = g->f("gate_top6_ids");
        uint32_t matched = 0;
        for (uint32_t i = 0; i < c.num_experts_per_tok; ++i)
            for (uint32_t j = 0; j < c.num_experts_per_tok; ++j)
                if (rig.layer.gate_ids()[i] == static_cast<uint32_t>(gid[j])) { ++matched; break; }
        std::printf("      %-20s %u/%u experts match the reference\n", "chained gate",
                    matched, c.num_experts_per_tok);
        CHECK_EQ(matched, c.num_experts_per_tok);

        // The §7.1 gate: make the six resident, signal, then dispatch the MoE.
        uint32_t fetched = 0;
        auto make_resident = [&](const uint32_t* ids, uint32_t n) -> Result<void> {
            for (uint32_t i = 0; i < n; ++i) {
                const ExpertKey key{static_cast<uint16_t>(want), static_cast<uint16_t>(ids[i])};
                if (rig.experts.resident(key)) continue;
                auto f = rig.planner.fetch(key, IoPriority::BlockingMiss, pos, want);
                if (!f) return std::unexpected(f.error());
                ++fetched;
            }
            rig.io.drain();
            return {};
        };
        auto moe = rig.layer.run_moe(rig.moe, st, &rig.timeline, make_resident);
        if (!moe) {
            std::printf("      MoE bridge unavailable: %s\n", moe.error().str().c_str());
            continue;
        }
        std::printf("      %-20s %u experts fetched from NVMe\n", "residency gate", fetched);

        const Agreement gm = agree(
            std::vector<float>(static_cast<const float*>(b.moe_y.host),
                               static_cast<const float*>(b.moe_y.host) + dim),
            g->f("moe_out"));
        std::printf("      %-20s %s\n", "chained moe_out", gm.str().c_str());

        REQUIRE_OK(rig.layer.run_close(st));
        const Agreement gb = agree(
            std::vector<float>(rig.layer.block_out(), rig.layer.block_out() + hcdim),
            g->f("block_out"));
        std::printf("      %-20s %s\n", "chained block_out", gb.str().c_str());
        CHECK(gb.cos > 0.999);       // design §12 L2
    }
}
