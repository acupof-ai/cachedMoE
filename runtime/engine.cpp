#include "runtime/engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>

#include "core/align.h"
#include "core/log.h"
#include "gpu/vulkan/moe_kernels.h"   // gpu::default_shader_dir
#include "model/layout.h"
#include "storage/backend.h"

namespace deepmoe::runtime {
namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    const char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
#if defined(_WIN32)
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

double ms_since(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

float bf16_to_f32(uint16_t h) {
    const uint32_t b = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

// design §3.3: fill path A, then path B.
//
// Path A is the DEVICE_LOCAL|HOST_VISIBLE heap -- 74 GiB on this machine, and
// what `bench/results/heap_capacity_idle.csv` measured as the larger of the
// two. When it runs out the pool keeps going on path B (imported host memory,
// ~26 GiB) rather than stopping, which is the difference between a 100 GiB
// cache and a 74 GiB one. The ExpertStore never learns which it got: both
// hand back a (host_ptr, device_address) pair, which is the whole point of
// store::SlabBacking.
class DualPathBacking final : public store::SlabBacking {
public:
    DualPathBacking(std::unique_ptr<store::SlabBacking> a,
                    std::unique_ptr<store::SlabBacking> b)
        : a_(std::move(a)), b_(std::move(b)) {}

    Result<store::SlabMemory> allocate(uint64_t bytes) override {
        if (a_ && !a_done_) {
            auto m = a_->allocate(bytes);
            if (m) { owner_.push_back({m->host_ptr, false}); ++a_slabs_; return m; }
            log_info("slab pool: path A full after {} slabs ({}), continuing on path B",
                     a_slabs_, m.error().str());
            a_done_ = true;
        }
        if (!b_) return fail(Err::ResourceExhausted, "path A is full and there is no path B");
        auto m = b_->allocate(bytes);
        if (!m) return m;
        owner_.push_back({m->host_ptr, true});
        ++b_slabs_;
        return m;
    }

    void release(const store::SlabMemory& mem) override {
        for (auto it = owner_.begin(); it != owner_.end(); ++it) {
            if (it->first != mem.host_ptr) continue;
            (it->second ? b_ : a_)->release(mem);
            owner_.erase(it);
            return;
        }
    }
    const char* name() const override { return "path A+B"; }

    uint32_t a_slabs() const { return a_slabs_; }
    uint32_t b_slabs() const { return b_slabs_; }

private:
    std::unique_ptr<store::SlabBacking> a_, b_;
    std::vector<std::pair<void*, bool>> owner_;
    uint32_t a_slabs_ = 0, b_slabs_ = 0;
    bool     a_done_ = false;
};

}  // namespace

Engine::~Engine() { shutdown(); }

Result<void> Engine::open_model_files() {
    // design §5.1 (v0.5): there are no repacked blobs. Every file the runtime
    // reads is an original safetensors shard named by the manifest.
    if (auto r = shards_.open_all(cfg_.model_dir, manifest_, cfg_.io.unbuffered); !r) return r;
    log_info("engine: {} shards open, {}", shards_.size(), human_bytes(shards_.total_bytes()));
    return {};
}

Result<void> Engine::init(const RuntimeConfig& cfg) {
    shutdown();
    cfg_ = cfg;

    if (!cfg_.profile_jsonl.empty()) {
        if (auto r = profiler_.open_jsonl(cfg_.profile_jsonl); !r) return r;
        log_info("engine: profiler -> {}", cfg_.profile_jsonl);
    }

    if (cfg_.model_dir.empty())
        return fail(Err::InvalidArgument, "RuntimeConfig::model_dir is empty");

    // config.json ships with the checkpoint, so a run is self-describing.
    auto mc = V41Config::load(join_path(cfg_.model_dir, "config.json"));
    if (!mc) return std::unexpected(mc.error());
    model_cfg_ = *std::move(mc);
    if (auto r = model_cfg_.validate_against_layout(); !r) return r;
    log_info("engine: {}", model_cfg_.summary());

    auto mf = Manifest::load(join_path(cfg_.model_dir, layout::kManifestFile));
    if (!mf) return std::unexpected(mf.error());
    manifest_ = *std::move(mf);
    if (auto r = manifest_.validate(); !r) return r;
    log_info("engine: manifest v{} '{}', {} tensors, {}", manifest_.version(), manifest_.model(),
             manifest_.tensors().size(), human_bytes(manifest_.total_bytes()));

    if (auto r = open_model_files(); !r) return r;

    auto backend = storage::make_default_backend(cfg_.io);
    if (!backend) return std::unexpected(backend.error());
    if (auto r = io_.start(std::move(*backend), cfg_.io, &profiler_); !r) return r;

    // Without a GPU the host backing is the honest choice and the only one the
    // storage tests need (design §3.3). init_gpu() re-backs the store.
    //
    // `budget_bytes == 0` means "size it from the machine", which only
    // init_gpu() can do -- and it would be absurd to commit gigabytes of host
    // memory here only to hand them straight back. A single slot is enough to
    // keep the store, the planner and the pointer table well-formed for a
    // caller that never brings up a GPU.
    CacheConfig boot = cfg_.cache;
    if (boot.budget_bytes == 0) {
        boot.slots_per_slab = 1;
        boot.budget_bytes   = layout::kExpertSlotBytes;
    }
    std::unique_ptr<store::SlabBacking> backing = std::make_unique<store::HostSlabBacking>();
    if (auto r = store_.init(std::move(backing), boot,
                             layout::kTotalLogicalLayers, layout::kRoutedExperts); !r)
        return r;

    if (auto r = planner_.init(store_, io_, manifest_, shards_, boot, cfg_.prefetch,
                               &profiler_); !r)
        return r;

    token_ = 0;
    ready_ = true;
    return {};
}

// --- GPU bring-up -----------------------------------------------------------

Result<void> Engine::load_pinned() {
    // design §9.3's pinned set: attention, shared experts, router, mHC, norms,
    // engram wkv, embed and head, read once and never evicted. §9.3's 17.7 GB
    // also counts the three DSpark (mtp) blocks; decode does not touch them, so
    // what is loaded here is the forty backbone layers plus the three globals
    // and it measures 9.17 GiB (docs/p2_decode.md §2.1).
    std::vector<std::string> names = store::pinned_global_tensors(manifest_);
    for (uint32_t L = 0; L < model_cfg_.text.num_hidden_layers; ++L) {
        auto per = store::pinned_layer_tensors(manifest_, L);
        names.insert(names.end(), per.begin(), per.end());
    }
    const uint64_t want = store::pinned_bytes(manifest_, names);

    // Both halves at once, before anything is allocated: design §5.2 / §9.2.2
    // measured the Windows commit limit as the binding constraint, and a
    // failure 14 GB into a 17.7 GB load is a worse diagnostic than a sentence.
    if (auto r = store::check_commit_available(want + cache_budget_,
                                               "pinned weights + expert cache"); !r)
        return r;

    store::PinnedConfig pc;
    pc.region_bytes = 1ull << 30;   // under the 2 GiB maxMemoryAllocationSize of §1.1
    auto pb = alloc_a_.make_slab_backing();
    if (!pb) return std::unexpected(pb.error());
    if (auto r = pinned_.init(std::move(*pb), pc); !r) return r;

    const TimePoint t0 = Clock::now();
    if (auto r = pinned_.load(manifest_, shards_, io_, names); !r) return r;
    const double s = ms_since(t0) / 1000.0;
    log_info("engine: pinned {} tensors, {} in {:.1f}s ({:.2f} GB/s), {} regions",
             pinned_.tensor_count(), human_bytes(pinned_.bytes_loaded()), s,
             s > 0 ? pinned_.bytes_loaded() / s / 1e9 : 0.0, pinned_.region_count());
    return {};
}

Result<void> Engine::build_expert_cache() {
    CacheConfig cache = cfg_.cache;
    cache.budget_bytes = cache_budget_;
    auto a = alloc_a_.make_slab_backing();
    if (!a) return std::unexpected(a.error());
    std::unique_ptr<store::SlabBacking> b;
    if (auto rb = alloc_b_.make_slab_backing(); rb) b = std::move(*rb);
    auto dual = std::make_unique<DualPathBacking>(std::move(*a), std::move(b));
    DualPathBacking* raw = dual.get();

    if (auto r = store_.init(std::move(dual), cache, layout::kTotalLogicalLayers,
                             layout::kRoutedExperts); !r)
        return r;
    if (auto r = planner_.init(store_, io_, manifest_, shards_, cache, cfg_.prefetch,
                               &profiler_); !r)
        return r;
    log_info("engine: expert cache {} slots, {} ({} slabs on path A, {} on path B)",
             store_.slot_count(), human_bytes(store_.capacity_bytes()),
             raw->a_slabs(), raw->b_slabs());
    return {};
}

Result<void> Engine::resolve_weights() {
    // Resolved once per layer, not per token: the whole point of the pinned set
    // is that these addresses never move.
    weights_.clear();
    weights_.reserve(model_cfg_.text.num_hidden_layers);
    layer_hot_bytes_.assign(model_cfg_.text.num_hidden_layers, 0);
    for (uint32_t L = 0; L < model_cfg_.text.num_hidden_layers; ++L) {
        auto w = LayerWeights::from_pinned(pinned_, L);
        if (!w) return std::unexpected(w.error());
        weights_.push_back(*w);
        // Everything this layer pins, which is everything a token reads from
        // it -- minus the compressor and indexer tensors on a source layer,
        // which are pinned but not dispatched until design 7.4's kernels land.
        layer_hot_bytes_[L] =
            store::pinned_bytes(manifest_, store::pinned_layer_tensors(manifest_, L));
    }
    auto n = pinned_.require("norm.weight");
    if (!n) return std::unexpected(n.error());
    norm_w_ = (*n)->data;
    auto h = pinned_.require("head.weight");
    if (!h) return std::unexpected(h.error());
    head_w_ = (*h)->data;
    auto e = pinned_.require("embed.weight");
    if (!e) return std::unexpected(e.error());
    embed_ = *e;
    return {};
}

Result<void> Engine::init_gpu() {
    if (!ready_) return fail(Err::FailedPrecondition, "call init() first");
    if (gpu_ready_) return {};

    if (auto r = device_.create(); !r) return r;
    if (auto r = device_.caps().check_required(); !r) return r;
    if (auto r = timeline_.create(device_, 0); !r) return r;
    if (auto r = alloc_a_.init(device_, MemoryPath::DeviceLocalHostVisible); !r) return r;
    // Path B is optional: it only widens the expert cache. A machine that
    // cannot import host memory still runs, with a path-A-sized cache.
    if (auto r = alloc_b_.init(device_, MemoryPath::ExternalMemoryHost); !r)
        log_warn("engine: path B unavailable ({}); the expert cache is path A only",
                 r.error().str());

    cache_budget_ = cfg_.cache.budget_bytes;
    if (cache_budget_ == 0) {
        // "As much as the machine will give". On Windows that is the commit
        // charge (design §5.2 / §9.2.2), minus the pinned set and a margin for
        // everything else the process does. The pool itself stops early and
        // says so if an allocation refuses before the budget is spent, so an
        // over-estimate costs a log line rather than a failure; the ceiling is
        // there so a machine with a very large pagefile does not spend a minute
        // allocating memory a decode step will never touch.
        //
        // Both paths, each by what bounds IT (docs/p2_decode.md §10):
        //   path A  the DEVICE_LOCAL|HOST_VISIBLE heap (74 GiB here), which the
        //           pinned set, the KV store and the activations share. It
        //           charges commit but not physical memory.
        //   path B  imported host pages, which charge BOTH. Bounded by what is
        //           physically free now, less a floor for the OS and whatever
        //           else the machine is running.
        // and the sum by the commit that is actually available. P2 step 2
        // capped the whole thing at path A's heap size, which with the pinned
        // set also living there left ~64 GiB of experts and never touched B.
        constexpr uint64_t kCommitMargin = 8ull << 30;
        constexpr uint64_t kPhysFloor    = 12ull << 30;
        constexpr uint64_t kPathAOther   = 1ull << 30;   // KV, scratch, logits, runners
        const uint64_t avail_commit = store::available_commit_bytes();
        const uint64_t avail_phys   = store::available_physical_bytes();
        std::vector<std::string> pnames = store::pinned_global_tensors(manifest_);
        for (uint32_t L = 0; L < model_cfg_.text.num_hidden_layers; ++L) {
            auto per = store::pinned_layer_tensors(manifest_, L);
            pnames.insert(pnames.end(), per.begin(), per.end());
        }
        const uint64_t pinned = store::pinned_bytes(manifest_, pnames);
        uint64_t heap_a = 74ull << 30;
        if (auto t = alloc_a_.chosen_memory_type(); t)
            for (const gpu::HeapInfo& h : device_.caps().heaps)
                if (h.index == t->heap_index) heap_a = h.bytes;
        const uint64_t a_cache = heap_a > pinned + kPathAOther ? heap_a - pinned - kPathAOther : 0;
        const bool     have_b  = alloc_b_.path() == MemoryPath::ExternalMemoryHost;
        const uint64_t b_cache = (have_b && avail_phys > kPhysFloor) ? avail_phys - kPhysFloor : 0;
        uint64_t want = a_cache + b_cache;
        if (avail_commit) {
            const uint64_t commit_cap =
                avail_commit > pinned + kCommitMargin ? avail_commit - pinned - kCommitMargin : 0;
            want = std::min(want, commit_cap);
        }
        cache_budget_ = std::max<uint64_t>(want, 8ull << 30);
        const uint64_t avail = avail_commit;
        log_info("engine: cache budget auto -> {} (path A {} after {} pinned, path B {} of "
                 "{} physical free; {} of commit available)",
                 human_bytes(cache_budget_), human_bytes(a_cache), human_bytes(pinned),
                 human_bytes(b_cache), human_bytes(avail_phys), human_bytes(avail));
    }

    if (auto r = load_pinned(); !r) return r;
    if (auto r = build_expert_cache(); !r) return r;
    if (auto r = resolve_weights(); !r) return r;

    const std::string dir = gpu::default_shader_dir();
    if (auto r = attn_.create(device_, alloc_a_, dir); !r) return r;
    if (auto r = dec_.create(device_, alloc_a_, dir); !r) return r;
    if (auto r = scratch_.create(alloc_a_, 32ull << 20); !r) return r;

    const TextConfig& c = model_cfg_.text;
    auto lg = alloc_a_.allocate(uint64_t(c.vocab_size) * sizeof(float), true, true);
    if (!lg) return std::unexpected(lg.error());
    logits_ = *lg;
    auto sm = alloc_a_.allocate_host_coherent(sizeof(gpu::SampleOut));
    if (!sm) return std::unexpected(sm.error());
    sample_ = *sm;
    std::memset(sample_.host_ptr, 0, sizeof(gpu::SampleOut));

    if (auto r = layer_.create(device_, attn_, scratch_, c); !r) return r;
    if (auto r = moe_.create(device_, alloc_a_, dir, store_, planner_, pinned_, c); !r) return r;
    // The MoE output stays on the GPU: the next layer's hc_post reads the
    // bridge's `y` by address instead of the host copying it into scratch.
    layer_.set_moe_output(moe_.y_address(), moe_.y_host());

    // The token loop's buffer, its completion fence and its timestamps.
    if (auto r = fence_.create(device_, 0); !r) return r;
    fence_value_ = 0;
    if (auto r = tok_pool_.create(device_); !r) return r;
    {
        auto cb = tok_pool_.acquire();
        if (!cb) return std::unexpected(cb.error());
        tok_cmd_ = *cb;
    }
    // 40 layers x (attention + MoE) x 2 stamps, two engram layers, the tail.
    if (auto r = tsq_.create(device_, 256); !r)
        log_warn("engine: no GPU timestamps ({}); the breakdown will be host-only",
                 r.error().str());

    build_ced_plan();
    timings_.assign(c.num_hidden_layers, LayerTiming{});
    gpu_ready_ = true;
    log_info("engine: gpu ready on {}", device_.caps().device_name);
    return {};
}

// model.py's `shared_attn`, written down. A source publishes its cache and
// every layer under it until the next source reads that same buffer; the
// reader's own `compress_ratio` still decides how much of it it may see.
void Engine::build_ced_plan() {
    const TextConfig& c = model_cfg_.text;
    ced_.assign(c.num_hidden_layers, CedPlan{});
    uint32_t cmp_src = 0, idx_src = 0;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        CedPlan& p = ced_[L];
        p.ratio           = c.compress_ratio(L);
        p.is_kv_source    = c.is_kv_source(L);
        p.is_index_source = c.is_index_source(L);
        if (p.is_kv_source) cmp_src = L;
        if (p.is_index_source) idx_src = L;
        p.cmp_src = cmp_src;
        p.idx_src = idx_src;
    }
    // A full prefill pass ends with the last kv_source layer having published
    // -- layer 20 at ratio 1, which completes at every position -- so that is
    // what the first decode step's ratio-2 indexers score against.
    pub_index_k_ = cmp_src;
}

// How many compressed positions each layer may read this step, and the window
// half of its top-k list. Both are pure functions of the position and the
// layer's ratio; the compressed half is the indexer's to write.
Result<void> Engine::prepare_ced(uint32_t position) {
    const TextConfig& c = model_cfg_.text;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        const CedPlan& p = ced_[L];
        const uint32_t n_cmp = p.ratio ? (position + 1) / p.ratio : 0u;
        if (n_cmp > kMaxIndexPositions)
            return fail(Err::ResourceExhausted,
                        std::format("layer {} wants {} compressed positions; the "
                                    "indexer's score plane holds {}", L, n_cmp,
                                    kMaxIndexPositions));
        // design §2.1's two-level top-k: layer 20 picks candidate_topk_blocks
        // blocks of candidate_block_size and layers 24..36 score only inside
        // them. Below that product every block is a candidate and the mask is
        // the identity, which is the only case this runtime implements -- so
        // it refuses rather than silently dropping the second level.
        if (p.is_index_source && L > c.candidate_source_layer_id &&
            n_cmp > c.candidate_topk_blocks * c.candidate_block_size)
            return fail(Err::Unimplemented,
                        std::format("{} compressed positions at layer {} needs design "
                                    "2.1's candidate-block mask, which is not written",
                                    n_cmp, L));
        // A layer whose plane is read by others -- an index source, or a
        // window-only layer that has no source -- owns its top-k list; the rest
        // are pointed at their source's and only need the counts.
        if (p.ratio == 0 || p.is_index_source) {
            if (auto r = kvs_.set_decode_topk(L, position, n_cmp); !r) return r;
        } else if (auto r = kvs_.set_counts(L, n_cmp, c.sliding_window + n_cmp); !r) {
            return r;
        }
    }
    return {};
}

Result<KvLayerView> Engine::effective_kv(uint32_t l) const {
    auto v = kvs_.layer(l);
    if (!v) return v;
    if (!produce_ced_ || l >= ced_.size() || ced_[l].ratio == 0) return v;
    auto cmp = kvs_.layer(ced_[l].cmp_src);
    if (!cmp) return cmp;
    auto idx = kvs_.layer(ced_[l].idx_src);
    if (!idx) return idx;
    v->cmp_kv       = cmp->cmp_kv;
    v->cmp_kv_host  = cmp->cmp_kv_host;
    v->top_idx      = idx->top_idx;
    v->top_idx_host = idx->top_idx_host;
    return v;
}

Result<void> Engine::load_decode_state(const std::string& dir) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    auto st = DecodeState::load(dir);
    if (!st) return std::unexpected(st.error());
    state_ = std::make_unique<DecodeState>(std::move(*st));

    const TextConfig& c = model_cfg_.text;
    KvStoreConfig kc;
    kc.layers      = c.num_hidden_layers;
    kc.window      = c.sliding_window;
    kc.latent_dim  = c.head_dim;
    // Sized from what the export actually holds, plus the room the remaining
    // steps will need; the compressed half grows by one row a step at ratio 1.
    kc.index_dim   = c.index_head_dim;
    kc.max_context = std::max<uint32_t>(256, state_->max_compressed() + 64);
    if (kc.max_context > kMaxIndexPositions)
        return fail(Err::ResourceExhausted,
                    std::format("the export needs {} compressed positions and the "
                                "indexer's score plane holds {}", kc.max_context,
                                kMaxIndexPositions));
    if (auto r = kvs_.create(alloc_a_, kc); !r) return r;
    if (auto r = state_->seed_prefill(kvs_); !r) return r;
    // Producing the compressed KV and the top-k list is the default, and the
    // only thing that can stop it is an export that predates the prefill-state
    // record -- which cannot supply the index-key cache the indexer scores
    // against, and from which it cannot be recovered (§7.4's key comes off the
    // PRE-RoPE latent).
    produce_ced_    = state_->has_prefill_ced();
    prefill_loaded_ = true;
    pub_index_k_    = ced_.empty() ? 0 : ced_.back().cmp_src;

    auto tables = EngramTables::load(dir);
    if (!tables) return std::unexpected(tables.error());
    if (auto r = engram_.create(device_, alloc_a_, dec_, manifest_, shards_, io_, pinned_, c,
                                *std::move(tables)); !r)
        return r;

    history_ = state_->prompt_ids();
    token_   = state_->prefill_len();
    log_info("engine: decode state loaded -- {} prompt tokens, {} reference steps, "
             "KV {} ({})",
             history_.size(), state_->steps(), human_bytes(kvs_.bytes()),
             produce_ced_ ? "prefill state seeded, every per-step tensor ours"
                          : "window REAL, compressed+topk LOADED per step");
    return {};
}

void Engine::shutdown() {
    // Everything that holds memory from an allocator has to let go before the
    // allocator does, and the allocators before the device.
    state_.reset();
    engram_.destroy();
    moe_.destroy();
    layer_.destroy();
    kvs_.destroy();
    tsq_.destroy();
    tok_pool_.destroy();
    tok_cmd_ = gpu::CommandBuffer{};
    tok_open_ = false;
    fence_.destroy();
    if (logits_.valid()) alloc_a_.free(logits_);
    if (sample_.valid()) alloc_a_.free(sample_);
    scratch_.destroy();
    dec_.destroy();
    attn_.destroy();
    store_.reset();      // the slabs came from alloc_a_/alloc_b_; give them back first
    pinned_.reset();
    io_.stop();
    shards_.close();
    alloc_a_.shutdown();
    alloc_b_.shutdown();
    timeline_.destroy();
    device_.destroy();
    kv_.reset();
    weights_.clear();
    timings_.clear();
    history_.clear();
    profiler_.close();
    ready_ = false;
    gpu_ready_ = false;
}

// TODO(design §11, §9.7): encoder over the full prompt, decoder bounded replay
// over the last 128 tokens, expert-major streaming above the length threshold.
Result<void> Engine::prefill(std::span<const uint32_t>) {
    return unimplemented("runtime::Engine::prefill (design §11, P5)");
}

Result<DecodeStepResult> Engine::slow_prefill(std::span<const uint32_t> prompt) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (prompt.empty()) return fail(Err::InvalidArgument, "the prompt is empty");
    if (!produce_ced_)
        return fail(Err::FailedPrecondition,
                    "slow_prefill produces the compressed KV and the top-k list, so "
                    "design 7.4's kernels have to be the ones running; "
                    "set_produce_ced(true)");
    const TextConfig& c = model_cfg_.text;
    if (prompt.size() > c.sliding_window)
        return fail(Err::Unimplemented,
                    std::format("{} prompt tokens against a {}-slot window: past the "
                                "window the ring wraps and a decode-shaped query no "
                                "longer sees the causal prefix, which is what design "
                                "11's chunked prefill is for",
                                prompt.size(), c.sliding_window));

    // Position 0 starts from nothing: an empty ring, an empty compressed plane,
    // an empty key cache and a compressor state of -inf.
    kvs_.clear();
    history_.assign(prompt.begin(), prompt.end());
    pub_index_k_ = ced_.empty() ? 0 : ced_.back().cmp_src;
    // `token_` is deliberately NOT reset. It is the cache's LRU clock, not a
    // position: winding it back makes everything already resident look newer
    // than what the next layer fetches, and the policy then evicts the slot it
    // just filled -- which surfaces as "expert (4, 1) is not resident at the
    // MoE dispatch" one dispatch later.

    const TimePoint t0 = Clock::now();
    DecodeStepResult last{};
    for (uint32_t p = 0; p < prompt.size(); ++p) {
        auto r = decode_step(prompt[p], p, -1);
        if (!r) return r;
        last = *r;
    }
    prefill_loaded_ = false;
    log_info("engine: slow prefill of {} tokens in {:.2f} s ({:.1f} ms/token), "
             "next token {}", prompt.size(), ms_since(t0) / 1000.0,
             ms_since(t0) / double(prompt.size()), last.token);
    return last;
}

// --- one decode step --------------------------------------------------------

Result<void> Engine::embed_token(uint32_t token) {
    const TextConfig& c = model_cfg_.text;
    if (token >= c.vocab_size)
        return fail(Err::OutOfRange, std::format("token {} >= vocab {}", token, c.vocab_size));
    // `h = embed(ids).unsqueeze(2).repeat(1, 1, hc_mult, 1)`: one row, four
    // identical copies. The row is bf16 in a path-A mapping, so it is memcpy'd
    // out in one go (a write-combining READ, design §3.3 -- 10 KiB of it, about
    // 15 us) and widened on the host rather than read element by element.
    std::vector<uint16_t> row(c.hidden_size);
    std::memcpy(row.data(),
                static_cast<const std::byte*>(embed_->data_host) +
                    uint64_t(token) * c.hidden_size * 2,
                size_t(c.hidden_size) * 2);
    // Widen once into host memory, then memcpy each copy. Writing the four hc
    // copies interleaved (`x[j * dim + d]` with j innermost) would touch four
    // addresses 20 KiB apart per element, which flushes a write-combining
    // buffer per store instead of filling it -- the write-side twin of the
    // read problem runtime/moe_bridge.cpp documents.
    std::vector<float> wide(c.hidden_size);
    for (uint32_t d = 0; d < c.hidden_size; ++d) wide[d] = bf16_to_f32(row[d]);
    auto* x = static_cast<float*>(layer_.scratch().x.host);
    for (uint32_t j = 0; j < c.hc_mult; ++j)
        std::memcpy(x + size_t(j) * c.hidden_size, wide.data(),
                    size_t(c.hidden_size) * sizeof(float));
    // `make_identity_pre_mix`: the first sublayer collapses the four copies
    // onto copy 0. post/comb stay zero -- the first sublayer applies no hc_post.
    auto* mix = static_cast<float*>(layer_.scratch().mix_a.host);
    std::memset(mix, 0, 128);
    mix[0] = 1.0f;
    return {};
}

// --- the token loop's command buffer ----------------------------------------
//
// design §7.1 wants one pre-recorded buffer per token. What this implements is
// the closest thing the gate allows without the pointer table moving onto the
// GPU: the host must read a layer's ids between its gate and its MoE, so the
// buffer is CUT THERE and nowhere else. Each submit carries
//
//     [ MoE of layer L-1 ] [ attention of layer L, dispatches 1-9 + §7.4 ]
//
// with the MoE gated on the residency timeline (a semaphore wait inside the
// submit, design §7.1, not a host wait and a resubmit), and completion signals
// a second timeline the host fences on. A token is 40 of those plus one for the
// tail, plus one extra on each engram layer (below) -- 43 against the 128 of
// P2 step 2, whose shape was one submit per dispatch group.

Result<void> Engine::cmd_open() {
    if (tok_open_) return {};
    if (auto r = tok_cmd_.begin(); !r) return r;
    tok_open_ = true;
    if (tok_first_) {
        if (tsq_.count()) (void)tok_cmd_.reset_queries(tsq_, 0, tsq_.count());
        tsq_used_  = 0;
        tok_first_ = false;
    }
    return {};
}

uint32_t Engine::cmd_stamp() {
    if (!tok_open_ || tsq_.count() == 0 || tsq_used_ >= tsq_.count()) return ~0u;
    const uint32_t i = tsq_used_++;
    (void)tok_cmd_.write_timestamp(tsq_, i, /*bottom=*/true);
    return i;
}

Result<void> Engine::cmd_submit(TimelineValue wait_value) {
    if (!tok_open_) return {};
    if (auto r = tok_cmd_.end(); !r) return r;
    tok_open_ = false;
    gpu::Submission s;
    s.cmd = &tok_cmd_;
    TimelineValue w[1] = {wait_value};
    if (wait_value) {
        s.timeline    = &timeline_;
        s.wait_values = std::span<const TimelineValue>(w, 1);
    }
    s.signal_timeline    = &fence_;
    s.signal_value       = ++fence_value_;
    s.signal_on_complete = true;
    const TimePoint t0 = Clock::now();
    if (auto r = gpu::submit(device_, s); !r) return r;
    sub_ms_ += ms_since(t0);
    ++submits_;
    return {};
}

Result<void> Engine::cmd_wait() {
    const TimePoint t0 = Clock::now();
    auto r = fence_.wait(fence_value_, std::chrono::seconds(120));
    wait_ms_ += ms_since(t0);
    return r;
}

void Engine::read_timestamps(DecodeStepResult& res) {
    res.breakdown.gpu_timed = false;
    if (tsq_.count() == 0 || tsq_used_ == 0) return;
    auto raw = tsq_.read_range(0, tsq_used_);
    if (!raw) return;
    const uint32_t bits = device_.caps().timestamp_valid_bits;
    const uint64_t mask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    const double   ns   = device_.caps().timestamp_period_ns;
    auto span_ms = [&](const Stamp& s) {
        if (s.begin == ~0u || s.end == ~0u) return 0.0;
        const uint64_t a = (*raw)[s.begin] & mask, b = (*raw)[s.end] & mask;
        const uint64_t t = b >= a ? b - a : (mask - a) + b + 1;
        return double(t) * ns * 1e-6;
    };
    for (uint32_t L = 0; L < timings_.size(); ++L) {
        timings_[L].attn_ms    = span_ms(ts_attn_[L]);
        timings_[L].moe_gpu_ms = span_ms(ts_moe_[L]);
        timings_[L].moe_ms     = timings_[L].moe_gpu_ms + timings_[L].moe_host_ms;
        timings_[L].engram_ms  = engram_host_ms_[L] + span_ms(ts_engram_[L]);
    }
    res.breakdown.tail_ms   = span_ms(ts_tail_);
    res.breakdown.gpu_timed = true;
}

Result<void> Engine::run_layer(uint32_t L, uint32_t position, bool& apply_post,
                               LayerTiming& t) {
    const TextConfig& c = model_cfg_.text;
    DecodeScratch& b = layer_.scratch();

    auto view = kvs_.layer(L);
    if (!view) return std::unexpected(view.error());

    LayerStep st;
    st.layer          = L;
    st.position       = position;
    st.compress_ratio = c.compress_ratio(L);
    st.apply_hc_post  = apply_post;
    st.kv             = *view;

    if (produce_ced_ && !ced_.empty() && st.compress_ratio) {
        const CedPlan& p = ced_[L];
        st.n_cmp          = view->n_cmp;
        st.run_compressor = p.is_kv_source;
        st.run_indexer    = p.is_index_source;
        st.cmp_complete   = ((position + 1) % st.compress_ratio) == 0;
        st.idx_key_write  = view->idx_key;
        // Publish before reading, exactly as `Indexer.forward` does: a source
        // that completes here is what this layer's own scoring will use.
        if (st.run_compressor && st.cmp_complete) pub_index_k_ = L;
        // The three caches this layer READS may each belong to a different
        // layer: the compressed KV to the last kv_source at or above it, the
        // top-k list to the last index_source, and the index keys to whichever
        // source published last -- which is not the same thing.
        auto cmp = kvs_.layer(p.cmp_src);
        if (!cmp) return std::unexpected(cmp.error());
        auto idx = kvs_.layer(p.idx_src);
        if (!idx) return std::unexpected(idx.error());
        auto key = kvs_.layer(pub_index_k_);
        if (!key) return std::unexpected(key.error());
        st.kv.cmp_kv  = cmp->cmp_kv;
        st.kv.top_idx = idx->top_idx;
        st.kv.idx_key = key->idx_key;
    }

    // The residency gate the open buffer's first dispatch -- the previous
    // layer's MoE -- is waiting on. Zero on layer 0, whose buffer has none.
    const TimelineValue prev_gate = L ? gpu::timeline_value(token_, L - 1) : 0;

    // The engram writes into the residual stream BEFORE the block (design
    // §2.1). design §7.7 normally defers the previous sublayer's hc_post into
    // this layer's first mega_mhc, so on an engram layer that hc_post has to be
    // materialised first -- otherwise the engram would be added to a stream
    // that is one sublayer behind.
    //
    // All of it joins the open buffer: the previous layer's MoE, its closing
    // hc_post, the engram's two dispatches and this layer's attention. The
    // rows were fetched at the start of the token (design §9.5), while layer
    // 0 ran; only a caller that skipped that pays for the fetch here.
    const bool engram = engram_.has_layer(L);
    if (engram) {
        if (!engram_.fetched(L, position)) {
            const TimePoint f0 = Clock::now();
            if (auto r = engram_.fetch(L, history_, position, &profiler_); !r) return r;
            engram_host_ms_[L] += ms_since(f0);
        }
        if (auto r = cmd_open(); !r) return r;
        if (apply_post) {
            // `bind` for L-1 left MhcClose pointing at L-1's hc_ffn weights,
            // which is what closing L-1's block needs -- and `bind_close`
            // below stops this layer's `bind` from overwriting it before the
            // buffer runs.
            LayerStep prev = st;
            prev.layer = L - 1;
            if (auto r = layer_.record_close(tok_cmd_, prev); !r) return r;
        }
        ts_engram_[L].begin = cmd_stamp();
        const DeviceAddress in = apply_post ? b.xout.addr : b.x.addr;
        if (auto r = engram_.record(tok_cmd_, L, in, b.x.addr); !r) return r;
        ts_engram_[L].end = cmd_stamp();
        apply_post = false;
        st.apply_hc_post = false;
        st.bind_close = false;
    }

    {
        const TimePoint b0 = Clock::now();
        if (auto r = layer_.bind(weights_[L], st); !r) return r;
        bind_ms_ += ms_since(b0);
    }
    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        ts_attn_[L].begin = cmd_stamp();
        if (auto r = layer_.record_attention(tok_cmd_, st); !r) return r;
        ts_attn_[L].end = cmd_stamp();
        rec_ms_ += ms_since(r0);
    }
    if (auto r = cmd_submit(prev_gate); !r) return r;
    // design §9.5: the engram's 96 row reads depend only on the token ids, so
    // they go out the moment the first buffer of the token is on the GPU and
    // overlap it, instead of blocking the engram layers when they are reached.
    if (L == 0) {
        for (uint32_t E = 1; E < c.num_hidden_layers; ++E) {
            if (!engram_.has_layer(E)) continue;
            const TimePoint f0 = Clock::now();
            if (auto r = engram_.fetch(E, history_, position, &profiler_); !r) return r;
            engram_host_ms_[E] += ms_since(f0);
        }
    }
    if (auto r = cmd_wait(); !r) return r;

    // design §7.1 / §7.8: the gate's ids are already in host-coherent memory.
    // Classify them, fetch the misses at P0, wait, host-signal the timeline the
    // MoE dispatch is gated on, then stage and record it.
    const uint32_t topk = c.num_experts_per_tok;
    const auto* ids = static_cast<const uint32_t*>(b.gate_ids.host);
    const auto* wts = static_cast<const float*>(b.gate_weights.host);
    {
        const TimePoint g0 = Clock::now();
        uint16_t chosen[16];
        for (uint32_t i = 0; i < topk; ++i) chosen[i] = static_cast<uint16_t>(ids[i]);
        store::RouteDecision route;
        route.layer   = L;
        route.chosen  = std::span<const uint16_t>(chosen, topk);
        route.weights = std::span<const float>(wts, topk);
        auto plan = planner_.plan_layer(route, token_);
        if (!plan) return std::unexpected(plan.error());
        t.hits       = static_cast<uint32_t>(plan->hits.size());
        t.misses     = static_cast<uint32_t>(plan->misses.size());
        t.miss_bytes = plan->miss_bytes;
        if (!plan->issued.empty()) io_.drain();
        t.gate_ms = ms_since(g0);
    }
    {
        const TimelineValue v = gpu::timeline_value(token_, L);
        auto cur = timeline_.value();
        if (cur && *cur < v)
            if (auto r = timeline_.signal(v); !r) return r;
    }
    profiler_.add_phase(Phase::NvmeStall, Nanos(int64_t(t.gate_ms * 1e6)));

    MoeCall call;
    call.layer   = L;
    call.ids     = ids;
    call.weights = wts;
    call.topk    = topk;
    call.x       = static_cast<const float*>(b.u.host);   // ffn_norm output
    call.y       = nullptr;                                // stays on the GPU
    call.hidden  = c.hidden_size;
    if (auto r = moe_.stage(call); !r) return r;
    t.moe_host_ms = moe_.timing().host_ms;
    mx_ms_ += moe_.timing().x_read_ms;
    mq_ms_ += moe_.timing().quant_ms;
    mt_ms_ += moe_.timing().table_ms;

    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        ts_moe_[L].begin = cmd_stamp();
        if (auto r = moe_.record(tok_cmd_); !r) return r;
        ts_moe_[L].end = cmd_stamp();
        rec_ms_ += ms_since(r0);
    }
    profiler_.note_hot_bytes(layer_hot_bytes_[L]);

    // A probe reads this layer's MoE output on the host, so it cannot wait
    // for the next layer's submit to carry it. It costs a submit a layer and
    // is never on in a real run.
    if (layer_probe) {
        if (auto r = cmd_flush(gpu::timeline_value(token_, L)); !r) return r;
        layer_probe(L, layer_);
    }
    apply_post = true;
    return {};
}

Result<DecodeStepResult> Engine::collapse_and_sample(uint32_t position) {
    const TextConfig& c = model_cfg_.text;
    DecodeScratch& b = layer_.scratch();
    const uint32_t n_wg0 = (c.hidden_size + 255) / 256;

    // `h = layer.hc_pre(h, pre_mix); logits = head(norm(h))`. mega_mhc stage 0
    // with the post bit does the last layer's hc_post AND the collapse, stage 2
    // does the RMSNorm; Sinkhorn is skipped because stage 1 is not dispatched
    // and there is no next sublayer to hand mixes to (gpu/shaders/head.slang).
    // The norm weight is the model's own `norm.weight`, not the layer's, so
    // MhcClose's slice is repointed and copied into MhcFinal's -- one slice per
    // stage is why the copy exists at all.
    uint64_t* close = attn_.slots(gpu::AttnStage::MhcClose);
    close[gpu::slot::kNormW] = norm_w_;
    std::memcpy(attn_.slots(gpu::AttnStage::MhcFinal), close, gpu::kAttnStageStride);

    uint64_t* hd = attn_.slots(gpu::AttnStage::Head);
    hd[gpu::slot::kHeadW]      = head_w_;
    hd[gpu::slot::kHeadX]      = b.u.addr;
    hd[gpu::slot::kHeadLogits] = logits_.dev_addr;

    uint64_t* sm = dec_.slots(gpu::DecodeStage::Argmax);
    sm[gpu::dslot::kHeadLogits] = logits_.dev_addr;
    sm[gpu::dslot::kHeadSample] = sample_.dev_addr;

    gpu::MhcPush mp{c.hidden_size, c.hc_mult, (2 + c.hc_mult) * c.hc_mult, n_wg0,
                    c.hc_sinkhorn_iters,
                    gpu::kMhcFlagPost | gpu::kMhcFlagSkipSinkhorn,
                    static_cast<float>(c.rms_norm_eps), static_cast<float>(c.hc_eps)};
    gpu::HeadPush hp{c.vocab_size, c.hidden_size, 0};

    // The last layer's MoE is already in the open buffer; the collapse, the
    // head and the greedy argmax (design §7.11: four words come back, never the
    // 129,280-wide logit vector) go in behind it.
    const TimePoint t0 = Clock::now();
    if (auto r = cmd_open(); !r) return std::unexpected(r.error());
    ts_tail_.begin = cmd_stamp();
    auto rec = [&](gpu::AttnStage s, const void* push, uint32_t bytes,
                   uint32_t groups) -> Result<void> {
        if (auto r = attn_.record(tok_cmd_, s, push, bytes, groups); !r) return r;
        return tok_cmd_.barrier();
    };
    if (auto r = rec(gpu::AttnStage::MhcClose, &mp, sizeof mp, n_wg0); !r)
        return std::unexpected(r.error());
    if (auto r = rec(gpu::AttnStage::MhcFinal, &mp, sizeof mp, n_wg0); !r)
        return std::unexpected(r.error());
    if (auto r = rec(gpu::AttnStage::Head, &hp, sizeof hp,
                     attn_.gemv_groups(gpu::AttnStage::Head, c.vocab_size)); !r)
        return std::unexpected(r.error());
    if (auto r = dec_.record(tok_cmd_, gpu::DecodeStage::Argmax, &hp, sizeof hp, 1); !r)
        return std::unexpected(r.error());
    ts_tail_.end = cmd_stamp();
    if (auto r = cmd_flush(gpu::timeline_value(token_, c.num_hidden_layers - 1)); !r)
        return std::unexpected(r.error());
    (void)t0;
    profiler_.note_hot_bytes(uint64_t(c.vocab_size) * c.hidden_size * 2);

    gpu::SampleOut out{};
    std::memcpy(&out, sample_.host_ptr, sizeof out);
    if (out.rows != c.vocab_size)
        return fail(Err::Internal,
                    std::format("the sampler scanned {} rows, not the {}-wide vocabulary",
                                out.rows, c.vocab_size));

    DecodeStepResult res;
    res.token    = out.token;
    res.position = position;
    res.top1     = out.top1;
    res.top2     = out.top2;
    return res;
}

Result<DecodeStepResult> Engine::decode_step(uint32_t in_token, uint32_t position,
                                             int32_t state_step) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (weights_.empty()) return fail(Err::FailedPrecondition, "no layer weights resolved");
    const TextConfig& c = model_cfg_.text;

    profiler_.token_begin(position);
    const TimePoint t_start = Clock::now();

    if (history_.size() <= position) history_.resize(position + 1, 0);
    history_[position] = in_token;

    const TimePoint t_prep = Clock::now();
    {
        ScopedPhaseIf p(&profiler_, Phase::CpuSync);
        if (produce_ced_) {
            if (auto r = prepare_ced(position); !r) return std::unexpected(r.error());
        } else if (state_step >= 0) {
            if (!state_) return fail(Err::FailedPrecondition, "no decode state is loaded");
            if (auto r = state_->seed_step(kvs_, static_cast<uint32_t>(state_step)); !r)
                return std::unexpected(r.error());
        }
    }

    if (auto r = embed_token(in_token); !r) return std::unexpected(r.error());
    const double prep_ms = ms_since(t_prep);

    // The token's first buffer resets the timestamp pool; everything after
    // appends to it and it is read once, after the last fence.
    timings_.assign(c.num_hidden_layers, LayerTiming{});
    ts_attn_.assign(c.num_hidden_layers, Stamp{});
    ts_moe_.assign(c.num_hidden_layers, Stamp{});
    ts_engram_.assign(c.num_hidden_layers, Stamp{});
    engram_host_ms_.assign(c.num_hidden_layers, 0.0);
    ts_tail_   = Stamp{};
    submits_   = 0;
    rec_ms_ = sub_ms_ = wait_ms_ = bind_ms_ = 0.0;
    mx_ms_ = mq_ms_ = mt_ms_ = 0.0;
    tok_first_ = true;
    if (tok_open_) {                      // a previous step failed mid-buffer
        (void)tok_cmd_.end();
        tok_open_ = false;
    }
    // The residency timeline is per token and monotone; a token that starts
    // below where the last one left it would have every wait satisfied before
    // its experts were resident.
    {
        auto cur = timeline_.value();
        const TimelineValue base = gpu::timeline_value(token_, 0) - 1;
        if (cur && *cur > base)
            return fail(Err::Internal,
                        std::format("the residency timeline is at {} but token {} starts "
                                    "at {}", *cur, token_, base));
    }

    bool apply_post = false;   // the very first sublayer has nothing to fold in
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L)
        if (auto r = run_layer(L, position, apply_post, timings_[L]); !r) {
            if (tok_open_) { (void)tok_cmd_.end(); tok_open_ = false; }
            return std::unexpected(r.error());
        }

    auto res = collapse_and_sample(position);
    if (!res) return res;
    res->wall_ms = ms_since(t_start);
    read_timestamps(*res);
    StepBreakdown& bd = res->breakdown;
    for (const LayerTiming& t : timings_) {
        bd.attn_ms     += t.attn_ms;
        bd.moe_gpu_ms  += t.moe_gpu_ms;
        bd.moe_host_ms += t.moe_host_ms;
        bd.gate_ms     += t.gate_ms;
        bd.engram_ms   += t.engram_ms;
    }
    bd.submits   = submits_;
    bd.record_ms = rec_ms_;
    bd.submit_ms = sub_ms_;
    bd.wait_ms   = wait_ms_;
    bd.bind_ms   = bind_ms_ + prep_ms;
    bd.moe_x_ms     = mx_ms_;
    bd.moe_quant_ms = mq_ms_;
    bd.moe_table_ms = mt_ms_;
    bd.other_ms = res->wall_ms - bd.attn_ms - bd.moe_gpu_ms - bd.moe_host_ms - bd.gate_ms -
                  bd.engram_ms - bd.tail_ms;
    // design 13.1's phases, from the same numbers: the GPU halves are only
    // known once the token's timestamps are read, which is now.
    profiler_.add_phase(Phase::HotGemv, Nanos(int64_t((bd.attn_ms + bd.tail_ms) * 1e6)));
    profiler_.add_phase(Phase::ExpertHit,
                        Nanos(int64_t((bd.moe_gpu_ms + bd.moe_host_ms) * 1e6)));
    res->record  = profiler_.token_end();
    ++token_;
    return res;
}

Result<SampleResult> Engine::decode_step() {
    if (!state_) return fail(Err::FailedPrecondition, "no decode state is loaded");
    if (history_.empty()) return fail(Err::FailedPrecondition, "nothing to decode from");
    // A step's `position` is the position of its INPUT token, which is the last
    // one in the history; the token it produces is appended, so repeated calls
    // walk forward. `state_step` is -1 because there is no way to know which of
    // the export's per-step records this position corresponds to -- a caller
    // that needs the LOADED compressed KV uses the three-argument form.
    const uint32_t position = static_cast<uint32_t>(history_.size()) - 1;
    auto r = decode_step(history_.back(), position, -1);
    if (!r) return std::unexpected(r.error());
    history_.push_back(r->token);
    SampleResult s;
    s.token   = r->token;
    s.logprob = r->top1;
    s.margin  = r->margin();
    return s;
}

Result<GenerateResult> Engine::generate(std::span<const uint32_t> prompt,
                                        const GenerateOptions& opts) {
    if (opts.speculative) return unimplemented("runtime::Engine::generate speculative (design §10, P4)");
    if (!state_)
        return fail(Err::FailedPrecondition,
                    "generate needs the prefill state; call load_decode_state() "
                    "(prefill itself is design 11 / P5)");
    if (!prompt.empty() && prompt.size() != state_->prompt_ids().size())
        return fail(Err::InvalidArgument,
                    std::format("prompt has {} tokens but the loaded state was built "
                                "from {}", prompt.size(), state_->prompt_ids().size()));

    GenerateResult out;
    const uint32_t base = state_->decode_pos();
    // greedy_tokens[0] is the argmax of the PREFILL logits, i.e. the first
    // token and the input to step 0. Prefill is not ours yet, so it is taken
    // from the export.
    uint32_t next = state_->greedy_tokens().front();
    const uint32_t n = std::min<uint32_t>(opts.max_tokens, state_->steps());
    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t in = (s < opts.forced.size()) ? opts.forced[s] : next;
        auto r = decode_step(in, base + s, static_cast<int32_t>(s));
        if (!r) return std::unexpected(r.error());
        out.tokens.push_back(r->token);
        next = r->token;
    }
    out.summary = profiler_.summary();
    return out;
}

Result<double> Engine::measure_submit_overhead(uint32_t iterations) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (iterations == 0) return fail(Err::InvalidArgument, "iterations must be >= 1");
    uint64_t* sm = dec_.slots(gpu::DecodeStage::Argmax);
    sm[gpu::dslot::kHeadLogits] = logits_.dev_addr;
    sm[gpu::dslot::kHeadSample] = sample_.dev_addr;
    gpu::HeadPush hp{model_cfg_.text.vocab_size, model_cfg_.text.hidden_size, 0};
    // One warm-up, so the first submit's pipeline bind does not land in the mean.
    if (auto r = dec_.dispatch_now(gpu::DecodeStage::Argmax, &hp, sizeof hp, 1); !r)
        return std::unexpected(r.error());
    const TimePoint t0 = Clock::now();
    for (uint32_t i = 0; i < iterations; ++i)
        if (auto r = dec_.dispatch_now(gpu::DecodeStage::Argmax, &hp, sizeof hp, 1); !r)
            return std::unexpected(r.error());
    return ms_since(t0) / iterations;
}

std::span<const float> Engine::last_logits() const {
    if (!logits_.valid()) return {};
    return {static_cast<const float*>(logits_.host_ptr), model_cfg_.text.vocab_size};
}

std::string Engine::status() const {
    std::string s;
    s += std::format("model     {}\n", ready_ ? model_cfg_.summary() : std::string("(not loaded)"));
    s += std::format("weights   {} shards, {} (direct read, no repack)\n",
                     shards_.size(), human_bytes(shards_.total_bytes()));
    s += std::format("io        {}\n", io_.running() ? io_.backend_caps().name : "stopped");
    s += "          " + io_.stats().to_string();
    s += std::format("pinned    {} tensors, {} in {} regions\n", pinned_.tensor_count(),
                     human_bytes(pinned_.bytes_loaded()), pinned_.region_count());
    s += std::format("store     {}\n", store_.stats().to_string());
    s += std::format("planner   {}\n", planner_.stats().to_string());
    s += std::format("gpu       {}\n", device_.valid() ? device_.caps().device_name
                                                       : std::string("(not created)"));
    // What a transcript needs to be able to say for itself: which of the
    // inputs to a decode step this run computed and which it was handed.
    if (!gpu_ready_) {
        s += "LOADED    (no GPU: nothing has run)\n";
    } else if (!produce_ced_) {
        s += std::format("LOADED    the window KV after {} prompt tokens, and the "
                         "compressed KV + indexer top-k of every step, from {} "
                         "(design 7.4's kernels exist; this run is not using them)\n",
                         state_ ? state_->prefill_len() : 0,
                         state_ ? state_->dir() : std::string("(nowhere)"));
    } else if (prefill_loaded_) {
        s += std::format("LOADED    the state the prompt left behind, from {}: the "
                         "window KV, the compressed KV cache, the indexer key cache "
                         "and the compressor's carried group, after {} prompt tokens. "
                         "Every per-step tensor is ours (design 7.4)\n",
                         state_ ? state_->dir() : std::string("(nowhere)"),
                         state_ ? state_->prefill_len() : 0);
    } else {
        s += "LOADED    none -- the prompt state came from slow_prefill and every "
             "decode-step tensor from design 7.4's kernels\n";
    }
    return s;
}

}  // namespace deepmoe::runtime
