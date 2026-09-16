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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <cstdio>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "cpu/dequant.h"
#include "gpu/vulkan/attn_kernels.h"
#include "gpu/vulkan/decode_kernels.h"
#include "gpu/vulkan/device.h"
#include "gpu/vulkan/memory.h"
#include "gpu/vulkan/moe_kernels.h"
#include "gpu/vulkan/timeline.h"
#include "model/layout.h"
#include "model/manifest.h"
#include "model/v41_config.h"
#include "runtime/decode_layer.h"
#include "runtime/decode_state.h"
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

// ============================================================================
// Track T: the verify batch, one layer at a time (docs/p4_mgt1.md).
//
// For each probe layer, each M in {2, 4, 6} and each context (the L3 prompt's
// 64 tokens; 4K with DEEPMOE_LONGCTX_DIR), from the export's prefill state and
// with the reference batch's own block input at that layer:
//
//   (1) M one-token steps through the M = 1 DecodeLayer, one position after
//       the other, each writing the ring, the compressor state, the compressed
//       row, the index key and the top-k list exactly as the Engine does;
//   (2) the same M tokens as ONE batch through the M > 1 kernels;
//
// and compares (1) with (2) stage by stage and state by state -- the
// equivalence the batch has to have -- and (2) with the reference's generalised
// batch (tools/oracle_dspark.py --mgt1), per stage.
//
// Data: DEEPMOE_MGT1_DIR (default <repo>/traces/mgt1), written by
// `tools/oracle_dspark.py --mgt1`; the 4K state from DEEPMOE_LONGCTX_DIR.
// ============================================================================

namespace {

std::string mgt1_root() {
    if (const char* e = std::getenv("DEEPMOE_MGT1_DIR")) return e;
    return std::string(DEEPMOE_TEST_DATA_DIR) + "/../../traces/mgt1";
}
std::string mgt1_state_dir(const std::string& ctx) {
    if (ctx == "l3") return std::string(DEEPMOE_TEST_DATA_DIR) + "/l3";
    const char* e = std::getenv("DEEPMOE_LONGCTX_DIR");
    return (e ? std::string(e) : std::string(DEEPMOE_TEST_DATA_DIR) + "/../../traces/longctx") +
           "/" + ctx;
}
bool mgt1_exists(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}
bool mgt1_ctx_wanted(const std::string& ctx) {
    const char* e = std::getenv("DEEPMOE_MGT1_CTX");
    const std::string list = e ? e : "l3,ctx4k";
    return list.find(ctx) != std::string::npos;
}
const L2Step* find_step(const L2Set& s, uint32_t layer, const std::string& step) {
    for (const L2Step& x : s.steps)
        if (x.layer == layer && x.step == step) return &x;
    return nullptr;
}
std::vector<float> bf16_plane(const uint16_t* p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = bf16_to_f32(p[i]);
    return v;
}

struct Mgt1Rig {
    gpu::Device           device;
    gpu::MemoryAllocator  alloc;
    Manifest              manifest;
    V41Config             config;
    store::ShardSet       shards;
    storage::IoEngine     io;
    store::PinnedStore    pinned;
    gpu::AttnRunner       runner;
    gpu::MgtRunner        mgt;
    gpu::GpuScratch       scratch, bscratch;
    runtime::KvStore      kv;
    runtime::DecodeLayer  layer;
    bool io_started = false;
    std::string why;

    ~Mgt1Rig() {
        layer.destroy();
        kv.destroy();
        bscratch.destroy();
        scratch.destroy();
        mgt.destroy();
        runner.destroy();
        if (io_started) io.stop();
    }

    bool bring_up() {
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
        pc.region_bytes = 1ull << 30;
        if (auto r = pinned.init(std::move(*pb), pc); !r) { why = r.error().str(); return false; }
        if (auto r = runner.create(device, alloc, gpu::default_shader_dir()); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = mgt.create(device, alloc, gpu::default_shader_dir()); !r) {
            why = r.error().str(); return false;
        }
        if (auto r = scratch.create(alloc, 32ull << 20); !r) { why = r.error().str(); return false; }
        if (auto r = bscratch.create(alloc, 64ull << 20); !r) { why = r.error().str(); return false; }
        return true;
    }

    // A KV store sized the way Engine::load_decode_state sizes it, and the
    // decode layer with a batch scratch for that many compressed positions.
    bool make_state(const runtime::DecodeState& st) {
        kv.destroy();
        layer.destroy();
        scratch.rewind();
        bscratch.rewind();
        const TextConfig& c = config.text;
        runtime::KvStoreConfig kc;
        kc.layers = c.num_hidden_layers;
        kc.window = c.sliding_window;
        kc.latent_dim = c.head_dim;
        kc.index_dim = c.index_head_dim;
        kc.max_context = std::max<uint32_t>(
            256, std::max(st.max_compressed(), st.max_prefill_rows()) + 64);
        if (auto r = kv.create(alloc, kc); !r) { why = r.error().str(); return false; }
        if (auto r = layer.create(device, runner, scratch, c); !r) { why = r.error().str(); return false; }
        if (auto r = layer.create_batch(mgt, bscratch, gpu::kMgtMaxM, kc.max_context); !r) {
            why = r.error().str(); return false;
        }
        return true;
    }
};

// model.py's shared_attn for one layer: which layer's compressed plane and
// which top-k list it reads (Engine::build_ced_plan), and that list's index in
// the batch scratch (0 for a window-only layer, 1 + rank of the index source).
struct CedSrc { uint32_t cmp = 0, idx = 0, list = 0; };
CedSrc ced_src(const TextConfig& c, uint32_t L) {
    CedSrc s;
    uint32_t rank = 0;
    for (uint32_t l = 0; l <= L; ++l) {
        if (c.is_kv_source(l)) s.cmp = l;
        if (c.is_index_source(l)) { s.idx = l; ++rank; s.list = rank; }
    }
    if (c.compress_ratio(L) == 0) s.list = 0;
    return s;
}

uint32_t last_kv_source(const TextConfig& c) {
    uint32_t last = 0;
    for (uint32_t l = 0; l < c.num_hidden_layers; ++l)
        if (c.is_kv_source(l)) last = l;
    return last;
}

}  // namespace

DEEPMOE_TEST(gpu_layer, mgt1_layer_batch_vs_steps) {
    if (skip_without_model("gpu_layer")) return;
    const std::string root = mgt1_root();
    Mgt1Rig rig;
    bool up = false;
    for (const std::string ctx : {std::string("l3"), std::string("ctx4k")}) {
        if (!mgt1_ctx_wanted(ctx)) continue;
        const std::string sdir = mgt1_state_dir(ctx);
        if (!mgt1_exists(root + "/" + ctx + "/l2/index.json") || !mgt1_exists(sdir + "/index.json")) {
            std::printf("      SKIP gpu_layer mgt1 %s: no reference at %s or state at %s "
                        "(DEEPMOE_MGT1_DIR, DEEPMOE_LONGCTX_DIR)\n", ctx.c_str(),
                        (root + "/" + ctx).c_str(), sdir.c_str());
            continue;
        }
        if (!up) {
            if (!rig.bring_up()) {
                std::printf("      SKIP gpu_layer mgt1: %s\n", rig.why.c_str());
                return;
            }
            up = true;
        }
        auto ref = load_l2(root + "/" + ctx + "/l2");
        REQUIRE_OK(ref);
        auto st = runtime::DecodeState::load(sdir);
        REQUIRE_OK(st);
        REQUIRE(rig.make_state(*st));
        const TextConfig& c = rig.config.text;
        const uint32_t dim = c.hidden_size, hcdim = c.hc_mult * dim;
        const uint32_t p0 = st->prefill_len();
        const uint32_t W = c.sliding_window;
        const uint32_t qrows = c.num_attention_heads * c.head_dim;
        const uint32_t last_src = last_kv_source(c);
        std::printf("    context %s: prefill %u tokens, batch at %u\n", ctx.c_str(), p0, p0);

        const char* lenv = std::getenv("DEEPMOE_MGT1_LAYERS");
        std::vector<uint32_t> layers = {0u, 2u, 14u, 20u, 24u};
        if (lenv) {
            layers.clear();
            for (const char* p = lenv; *p;) {
                layers.push_back(uint32_t(std::strtoul(p, nullptr, 10)));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
        for (uint32_t L : layers) {
            auto names = store::pinned_layer_tensors(rig.manifest, L);
            REQUIRE_OK(rig.pinned.load(rig.manifest, rig.shards, rig.io, names));
            auto w = runtime::LayerWeights::from_pinned(rig.pinned, L);
            REQUIRE_OK(w);
            const uint32_t ratio = c.compress_ratio(L);
            const CedSrc src = ced_src(c, L);

            for (uint32_t M : {2u, 4u, 6u}) {
                const L2Step* g = find_step(*ref, L, std::format("m{}", M));
                if (!g) { std::printf("      L%02u M=%u: not exported\n", L, M); continue; }
                const std::vector<float>& xin = g->f("attn_resid_in");
                const std::vector<float>& pmix = g->f("pre_mix_in");

                // ---- (1) M one-token steps ------------------------------------
                rig.kv.clear();
                REQUIRE_OK(st->seed_prefill(rig.kv));
                struct Tok {
                    std::vector<float> ffn, qr, kv, o, woa, wob, gw;
                    std::vector<uint32_t> ids;
                    std::vector<uint16_t> q;
                    std::vector<int32_t> list;
                };
                std::vector<Tok> seq(M);
                runtime::DecodeScratch& sb = rig.layer.scratch();
                for (uint32_t m = 0; m < M; ++m) {
                    const uint32_t pos = p0 + m;
                    for (uint32_t l : {L, src.cmp, src.idx}) {
                        const uint32_t r = c.compress_ratio(l);
                        const uint32_t n_cmp = r ? (pos + 1) / r : 0u;
                        const uint32_t n_sel = std::min<uint32_t>(n_cmp, c.index_topk);
                        if (r == 0 || c.is_index_source(l))
                            REQUIRE_OK(rig.kv.set_decode_topk(l, pos, n_cmp, n_sel));
                        else
                            REQUIRE_OK(rig.kv.set_counts(l, n_cmp, W + n_sel));
                    }
                    auto view = rig.kv.layer(L);
                    REQUIRE_OK(view);
                    runtime::LayerStep ls;
                    ls.layer = L;
                    ls.position = pos;
                    ls.compress_ratio = ratio;
                    ls.apply_hc_post = false;
                    ls.kv = *view;
                    if (ratio) {
                        ls.n_cmp = view->n_cmp;
                        ls.run_compressor = c.is_kv_source(L);
                        ls.run_indexer = c.is_index_source(L);
                        ls.cmp_complete = ((pos + 1) % ratio) == 0;
                        ls.idx_key_write = view->idx_key;
                        const uint32_t pub = (ls.run_compressor && ls.cmp_complete) ? L : last_src;
                        ls.kv.cmp_kv = rig.kv.layer(src.cmp)->cmp_kv;
                        ls.kv.top_idx = rig.kv.layer(src.idx)->top_idx;
                        ls.kv.top_idx_host = rig.kv.layer(src.idx)->top_idx_host;
                        ls.kv.idx_key = rig.kv.layer(pub)->idx_key;
                    }
                    std::memcpy(sb.x.host, xin.data() + uint64_t(m) * hcdim, uint64_t(hcdim) * 4);
                    std::memset(sb.mix_a.host, 0, 128);
                    std::memcpy(sb.mix_a.host, pmix.data() + uint64_t(m) * c.hc_mult, c.hc_mult * 4);
                    REQUIRE_OK(rig.layer.bind(*w, ls));
                    REQUIRE_OK(rig.layer.run_attention(ls));
                    Tok& t = seq[m];
                    auto grab = [](const gpu::GpuScratch::View& v, size_t n) {
                        const float* p = static_cast<const float*>(v.host);
                        return std::vector<float>(p, p + n);
                    };
                    t.ffn = std::vector<float>(rig.layer.ffn_norm_out(), rig.layer.ffn_norm_out() + dim);
                    t.qr  = grab(sb.qr, c.q_lora_rank);
                    t.kv  = grab(sb.kv, c.head_dim);
                    t.o   = grab(sb.o, qrows);
                    t.woa = grab(sb.woa, c.o_groups * c.o_lora_rank);
                    t.wob = grab(sb.wob, dim);
                    t.ids.assign(rig.layer.gate_ids(), rig.layer.gate_ids() + c.num_experts_per_tok);
                    t.gw.assign(rig.layer.gate_weights(), rig.layer.gate_weights() + c.num_experts_per_tok);
                    const auto* q16 = static_cast<const uint16_t*>(sb.q.host);
                    t.q.assign(q16, q16 + qrows);
                    if (ratio && c.is_index_source(L)) {
                        const uint32_t n_sel = std::min<uint32_t>(ls.n_cmp, c.index_topk);
                        const auto* li = reinterpret_cast<const int32_t*>(view->top_idx_host);
                        t.list.assign(li + W, li + W + n_sel);
                    }
                }
                auto sv = rig.kv.layer(L);
                REQUIRE_OK(sv);
                std::vector<uint8_t> ring_seq(sv->win_val_host, sv->win_val_host + uint64_t(W) * c.head_dim);
                std::vector<uint8_t> rsc_seq(sv->win_scale_host,
                                             sv->win_scale_host + uint64_t(W) * (c.head_dim / 32));
                const uint32_t cmp_rows_hi = ratio ? (p0 + M) / ratio + 1 : 0u;
                std::vector<uint16_t> cmp_seq, key_seq;
                std::vector<float> state_seq;
                if (c.is_kv_source(L)) {
                    cmp_seq.assign(sv->cmp_kv_host, sv->cmp_kv_host + uint64_t(cmp_rows_hi) * c.head_dim);
                    key_seq.assign(sv->idx_key_host,
                                   sv->idx_key_host + uint64_t(cmp_rows_hi) * c.index_head_dim);
                    state_seq.assign(sv->cmp_state_kv_host,
                                     sv->cmp_state_kv_host + uint64_t(ratio) * c.head_dim);
                    state_seq.insert(state_seq.end(), sv->cmp_state_score_host,
                                     sv->cmp_state_score_host + uint64_t(ratio) * c.head_dim);
                }

                // ---- (2) the batch ---------------------------------------------
                rig.kv.clear();
                REQUIRE_OK(st->seed_prefill(rig.kv));
                auto bv = rig.kv.layer(L);
                REQUIRE_OK(bv);
                std::vector<uint8_t> ring_before(bv->win_val_host, bv->win_val_host + uint64_t(W) * c.head_dim);
                runtime::BatchScratch& bb = rig.layer.batch();
                runtime::write_batch_window_lists(bb, W, p0, M);
                runtime::BatchStep bs;
                bs.layer = L;
                bs.p0 = p0;
                bs.m = M;
                bs.compress_ratio = ratio;
                bs.apply_hc_post = false;
                bs.kv = *bv;
                bs.list = src.list;
                if (ratio) {
                    bs.run_compressor = c.is_kv_source(L);
                    bs.run_indexer = c.is_index_source(L);
                    bs.kv.cmp_kv = rig.kv.layer(src.cmp)->cmp_kv;
                    bs.idx_key_own = bv->idx_key;
                    bs.idx_key_pub = rig.kv.layer(last_src)->idx_key;
                    for (uint32_t m = 0; m < M; ++m)
                        if (bs.run_compressor && ((p0 + m + 1) % ratio) == 0) bs.key_sel |= 1u << m;
                }
                std::memcpy(bb.x.host, xin.data(), uint64_t(M) * hcdim * 4);
                std::memset(bb.mix_a.host, 0, uint64_t(M) * 128);
                for (uint32_t m = 0; m < M; ++m)
                    std::memcpy(static_cast<std::byte*>(bb.mix_a.host) + uint64_t(m) * 128,
                                pmix.data() + uint64_t(m) * c.hc_mult, c.hc_mult * 4);
                REQUIRE_OK(rig.layer.bind_batch(*w, bs));
                REQUIRE_OK(rig.layer.run_attention_batch(bs));

                // ---- (1) vs (2) -------------------------------------------------
                double worst_ffn = 1.0, worst_wob = 1.0, worst_o = 1.0, worst_kv = 1.0;
                double worst_woa = 1.0, worst_qr = 1.0;
                uint32_t ids_equal = 0;
                double worst_q = 1.0;
                for (uint32_t m = 0; m < M; ++m) {
                    const Tok& t = seq[m];
                    auto bat = [&](const gpu::GpuScratch::View& v, uint64_t stride) {
                        return static_cast<const float*>(v.host) + uint64_t(m) * stride;
                    };
                    worst_ffn = std::min(worst_ffn, agree(bat(bb.u, dim), t.ffn.data(), dim).cos);
                    worst_wob = std::min(worst_wob, agree(bat(bb.wob, dim), t.wob.data(), dim).cos);
                    worst_woa = std::min(worst_woa, agree(bat(bb.woa, c.o_groups * c.o_lora_rank),
                                                          t.woa.data(), c.o_groups * c.o_lora_rank).cos);
                    worst_o   = std::min(worst_o, agree(bat(bb.o, qrows), t.o.data(), qrows).cos);
                    worst_kv  = std::min(worst_kv, agree(bat(bb.kv_out, c.head_dim), t.kv.data(),
                                                         c.head_dim).cos);
                    worst_qr  = std::min(worst_qr, agree(bat(bb.qr_raw, c.q_lora_rank), t.qr.data(),
                                                         c.q_lora_rank).cos);
                    const auto* ids = static_cast<const uint32_t*>(bb.gate_ids.host) + uint64_t(m) * 16;
                    if (std::equal(t.ids.begin(), t.ids.end(), ids)) ++ids_equal;
                    const auto* q16 = static_cast<const uint16_t*>(bb.q.host) + uint64_t(m) * qrows;
                    const std::vector<float> a = bf16_plane(q16, qrows), b2 = bf16_plane(t.q.data(), qrows);
                    worst_q = std::min(worst_q, agree(a.data(), b2.data(), qrows).cos);
                }
                auto bv2 = rig.kv.layer(L);
                const bool ring_eq = std::equal(ring_seq.begin(), ring_seq.end(), bv2->win_val_host) &&
                                     std::equal(rsc_seq.begin(), rsc_seq.end(), bv2->win_scale_host);
                uint32_t ovf_ok = 0;
                for (uint32_t j = 1; j < M; ++j) {
                    const uint32_t slot = (p0 + j) % W;
                    if (std::memcmp(static_cast<const uint8_t*>(bb.ovf_val.host) + uint64_t(j - 1) * c.head_dim,
                                    ring_before.data() + uint64_t(slot) * c.head_dim, c.head_dim) == 0)
                        ++ovf_ok;
                }
                std::printf("      L%02u M=%u batch vs steps: wq_a %.9f q %.9f kv %.9f attn %.9f "
                            "wo_a %.9f wo_b %.9f ffn_norm %.9f | gate ids %u/%u ring %s overflow %u/%u\n",
                            L, M, worst_qr, worst_q, worst_kv, worst_o, worst_woa, worst_wob, worst_ffn,
                            ids_equal, M, ring_eq ? "bit-equal" : "DIFF", ovf_ok, M - 1);
                CHECK(worst_ffn > 0.99999);
                CHECK(worst_wob > 0.99999);
                CHECK_EQ(ids_equal, M);
                CHECK(ring_eq);
                CHECK_EQ(ovf_ok, M - 1);
                if (c.is_kv_source(L)) {
                    const bool cmp_eq = std::equal(cmp_seq.begin(), cmp_seq.end(), bv2->cmp_kv_host);
                    const bool key_eq = std::equal(key_seq.begin(), key_seq.end(), bv2->idx_key_host);
                    std::vector<float> state_b(bv2->cmp_state_kv_host,
                                               bv2->cmp_state_kv_host + uint64_t(ratio) * c.head_dim);
                    state_b.insert(state_b.end(), bv2->cmp_state_score_host,
                                   bv2->cmp_state_score_host + uint64_t(ratio) * c.head_dim);
                    const Agreement sa = agree(state_b.data(), state_seq.data(), state_b.size());
                    std::printf("      L%02u M=%u   compressed rows %s, index keys %s, carried state "
                                "max|d| %.3e\n", L, M, cmp_eq ? "bit-equal" : "DIFF",
                                key_eq ? "bit-equal" : "DIFF", sa.max_abs);
                    CHECK(cmp_eq);
                    CHECK(key_eq);
                }
                if (ratio && c.is_index_source(L)) {
                    uint32_t lists_eq = 0;
                    const uint32_t off = W + M - 1;
                    for (uint32_t m = 0; m < M; ++m) {
                        const int32_t* row = bb.list_host(src.list) + uint64_t(m) * bb.list_stride;
                        bool eq = true;
                        for (size_t i = 0; i < seq[m].list.size(); ++i)
                            if (row[off + i] - int32_t(off) != seq[m].list[i] - int32_t(W)) { eq = false; break; }
                        if (eq) ++lists_eq;
                    }
                    std::printf("      L%02u M=%u   compressed top-k lists identical %u/%u\n", L, M,
                                lists_eq, M);
                    CHECK_EQ(lists_eq, M);
                }

                // ---- (2) vs the reference batch, per stage ----------------------
                auto cmpf = [&](const char* name, const float* ours, const char* key) {
                    const L2Tensor* t = g->find(key);
                    if (!t) return;
                    const Agreement a = agree(ours, t->f.data(), t->f.size());
                    std::printf("        ref %-18s %s\n", name, a.str().c_str());
                };
                cmpf("wq_a", static_cast<const float*>(bb.qr_raw.host), "wq_a_out");
                cmpf("q_norm", static_cast<const float*>(bb.qr.host), "qr");
                {
                    const std::vector<float> qf = bf16_plane(static_cast<const uint16_t*>(bb.q.host),
                                                             uint64_t(M) * qrows);
                    cmpf("wq_b+rope", qf.data(), "q");
                }
                cmpf("kv (ring value)", static_cast<const float*>(bb.kv_out.host), "kv");
                cmpf("attn+inv rope", static_cast<const float*>(bb.o.host), "attn_out_irope");
                cmpf("wo_a", static_cast<const float*>(bb.woa.host), "wo_a_out");
                cmpf("wo_b", static_cast<const float*>(bb.wob.host), "wo_b_out");
                cmpf("ffn_norm", static_cast<const float*>(bb.u.host), "ffn_norm_out");
                if (const L2Tensor* gi = g->find("gate_ids")) {
                    uint32_t match = 0;
                    for (uint32_t m = 0; m < M; ++m)
                        for (uint32_t i = 0; i < c.num_experts_per_tok; ++i)
                            for (uint32_t j = 0; j < c.num_experts_per_tok; ++j)
                                if (static_cast<const uint32_t*>(bb.gate_ids.host)[m * 16 + i] ==
                                    uint32_t(gi->f[m * c.num_experts_per_tok + j])) { ++match; break; }
                    std::printf("        ref %-18s %u/%u experts\n", "gate ids", match,
                                M * c.num_experts_per_tok);
                }
                if (c.is_kv_source(L) && g->find("latent_pre_rope"))
                    cmpf("compressor latent", static_cast<const float*>(bb.latent.host), "latent_pre_rope");
                if (c.is_index_source(L)) {
                    const uint32_t irows = c.index_n_heads * c.index_head_dim;
                    const std::vector<float> iq = bf16_plane(static_cast<const uint16_t*>(bb.idx_q.host),
                                                             uint64_t(M) * irows);
                    cmpf("index q", iq.data(), "index_q");
                    // The export is weights_proj's output; the kernel folds in
                    // softmax_scale * n_heads^-0.5 as the M = 1 stage does.
                    if (const L2Tensor* tw = g->find("index_weights")) {
                        std::vector<float> scaled(tw->f);
                        const float ws = 1.0f / std::sqrt(float(c.index_head_dim)) /
                                         std::sqrt(float(c.index_n_heads));
                        for (float& v : scaled) v = cpu::bf16_to_float(cpu::float_to_bf16(v * ws));
                        const Agreement a = agree(static_cast<const float*>(bb.idx_w.host),
                                                  scaled.data(), scaled.size());
                        std::printf("        ref %-18s %s\n", "index weights", a.str().c_str());
                    }
                    if (const L2Tensor* ti = g->find("topk_idxs")) {
                        const uint32_t n = uint32_t(ti->shape.back());
                        const uint32_t off = W + M - 1;
                        uint64_t common = 0, total = 0;
                        for (uint32_t m = 0; m < M; ++m) {
                            std::vector<int32_t> a, b2;
                            const int32_t* row = bb.list_host(src.list) + uint64_t(m) * bb.list_stride;
                            for (uint32_t i = off; i < std::min(n, bb.list_stride); ++i)
                                if (row[i] >= 0) a.push_back(row[i]);
                            for (uint32_t i = off; i < n; ++i)
                                if (ti->f[uint64_t(m) * n + i] >= 0)
                                    b2.push_back(int32_t(ti->f[uint64_t(m) * n + i]));
                            std::sort(a.begin(), a.end());
                            std::sort(b2.begin(), b2.end());
                            std::vector<int32_t> in;
                            std::set_intersection(a.begin(), a.end(), b2.begin(), b2.end(),
                                                  std::back_inserter(in));
                            common += in.size();
                            total += b2.size();
                        }
                        std::printf("        ref %-18s %llu/%llu compressed picks in common\n",
                                    "top-k", (unsigned long long)common, (unsigned long long)total);
                    }
                }
            }
        }
    }
}
