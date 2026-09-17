#include "runtime/engine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <format>
#include <set>

#include "core/align.h"
#include "core/log.h"
#include "gpu/vulkan/moe_kernels.h"   // gpu::default_shader_dir
#include "gpu/vulkan/prefill_kernels.h"
#include "runtime/engram_tables.h"
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

// VK_EXT_memory_budget's view of the HOST heap: what the OS says this process
// may still put there, given everything every other process already has. Zero
// when the extension is not there. Path B's imports are charged to this heap,
// and on this driver an import past the budget does not fail cleanly -- it
// returns VK_ERROR_INVALID_EXTERNAL_HANDLE and the device is lost on the next
// submit -- so a cache that sizes itself has to ask first.
uint64_t host_heap_headroom(const gpu::Device& d, uint64_t* budget_out, uint64_t* usage_out) {
#if defined(DEEPMOE_ENABLE_VULKAN)
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(d.physical(), nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    vkEnumerateDeviceExtensionProperties(d.physical(), nullptr, &n, ext.data());
    bool have = false;
    for (const VkExtensionProperties& e : ext)
        have |= std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    if (!have) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    props.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(d.physical(), &props);
    for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
        if (props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) continue;
        if (budget_out) *budget_out = budget.heapBudget[i];
        if (usage_out) *usage_out = budget.heapUsage[i];
        return budget.heapBudget[i] > budget.heapUsage[i]
                   ? budget.heapBudget[i] - budget.heapUsage[i] : 0;
    }
#else
    (void)d; (void)budget_out; (void)usage_out;
#endif
    return 0;
}

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

    // init() backed the store with HOST memory sized by `--cache-gb` -- 100 GiB
    // of commit for `--cache-gb 100` -- and build_expert_cache re-backs it on
    // the GPU paths anyway. Let it go first, or the commit check below counts
    // the cache twice and refuses (docs/p4_hitrate.md §2).
    planner_.stop_backfill();
    store_.reset();
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
        uint64_t b_cache = (have_b && avail_phys > kPhysFloor) ? avail_phys - kPhysFloor : 0;
        // Path B is also bounded by the HOST heap the driver reports (37.2 GiB
        // here), which is what imported pages are charged to. Physical memory
        // alone is not the limit: with 50 GB free, an import past ~33 GiB
        // failed with VK_ERROR_INVALID_EXTERNAL_HANDLE and the device was lost
        // on the next submit (docs/p2_decode.md §12.2). bench/heap_capacity's
        // mixed run stopped path B at 26 GiB; 10 GiB of the heap is left for
        // everything else host-heap-backed.
        constexpr uint64_t kHostHeapMargin = 10ull << 30;
        for (const gpu::HeapInfo& h : device_.caps().heaps)
            if (!h.device_local && h.bytes > kHostHeapMargin)
                b_cache = std::min(b_cache, h.bytes - kHostHeapMargin);
        // And by what the OS will actually grant now -- the heap is shared with
        // every other Vulkan process on the machine -- less a 4 GiB margin. On
        // this driver the budget is 35.4 GiB and the usage reads 0 whatever is
        // running, so it is logged and applied but is not what keeps an import
        // from failing.
        uint64_t hb = 0, hu = 0;
        const uint64_t headroom = host_heap_headroom(device_, &hb, &hu);
        if (hb) {
            constexpr uint64_t kBudgetMargin = 4ull << 30;
            b_cache = std::min(b_cache, headroom > kBudgetMargin ? headroom - kBudgetMargin : 0);
            log_info("engine: host heap budget {}, {} reported in use", human_bytes(hb),
                     human_bytes(hu));
        }
        // What does: a fixed ceiling on path B. Imports failed -- and took the
        // device with them -- at 33 GiB on an idle machine with 50 GB free, and
        // at 15.8 GiB while another track's GPU test held host-heap memory.
        // 16 GiB is under both, and puts the auto-sized cache at ~80 GiB.
        // `--cache-gb` asks for more explicitly, and owns the risk.
        constexpr uint64_t kPathBAutoCeiling = 16ull << 30;
        b_cache = std::min(b_cache, kPathBAutoCeiling);
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
    {
        const uint64_t out_bytes =
            uint64_t(kTopKHeaderWords + kTopKThreads + 2ull * kTopKThreads * kTopKCapPerThread) * 4;
        auto to = alloc_a_.allocate_host_coherent(align_up(out_bytes, 4096));
        if (!to) return std::unexpected(to.error());
        topk_out_ = *to;
        std::memset(topk_out_.host_ptr, 0, static_cast<size_t>(topk_out_.bytes));
        auto th = alloc_a_.allocate_host_coherent(uint64_t(kTopKThreads) * kTopKBins * 4);
        if (!th) return std::unexpected(th.error());
        topk_hist_ = *th;
    }

    if (auto r = layer_.create(device_, attn_, scratch_, c); !r) return r;
    if (auto r = moe_.create(device_, alloc_a_, dir, store_, planner_, pinned_, c); !r) return r;
    // The MoE output stays on the GPU: the next layer's hc_post reads the
    // bridge's `y` by address instead of the host copying it into scratch.
    layer_.set_moe_output(moe_.y_address(), moe_.y_host());
    // The FFN input the host reads every layer, in cached host pages rather than
    // path A's write-combining mapping (DecodeLayer::set_ffn_input). Path A if
    // there is no path B: slower, still correct.
    {
        const uint64_t bytes = align_up(uint64_t(c.hidden_size) * sizeof(float), 1ull << 16);
        auto fb = alloc_b_.path() == MemoryPath::ExternalMemoryHost
                      ? alloc_b_.allocate_imported(bytes, /*device_address=*/true)
                      : alloc_a_.allocate(bytes, true, true);
        if (!fb) fb = alloc_a_.allocate(bytes, true, true);
        if (!fb) return std::unexpected(fb.error());
        ffn_in_buf_ = *fb;
        ffn_in_alloc_ = (fb->host_alloc != nullptr) ? &alloc_b_ : &alloc_a_;
        layer_.set_ffn_input(ffn_in_buf_.dev_addr, static_cast<float*>(ffn_in_buf_.host_ptr));
    }

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
    if (const char* e = std::getenv("DEEPMOE_ROUTE_DUMP"); e && *e && !route_dump_) {
        route_dump_ = std::fopen(e, "ab");
        if (route_dump_) log_info("engine: routing dump -> {}", e);
        else log_warn("engine: cannot open the routing dump '{}'", e);
    }
    route_ids_.assign(size_t(c.num_hidden_layers) * c.num_experts_per_tok, 0);
    // docs/p4_hitrate.md §4: on unless DEEPMOE_MOE_OVERLAP=0 (the A/B switch).
    if (const char* e = std::getenv("DEEPMOE_MOE_OVERLAP"); e && *e == '0') overlap_ = false;
    if (const char* e = std::getenv("DEEPMOE_PREFILL_HANDOFF"); e && *e == '0') handoff_ = false;
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
        // design §2.1's two-level top-k (layer 20 keeps candidate_topk_blocks
        // blocks, layers 24..36 score inside them) is DecodeLayer's: it
        // dispatches indexer.slang stages 6-8 once n_cmp passes 16,384.
        // A layer whose plane is read by others -- an index source, or a
        // window-only layer that has no source -- owns its top-k list; the rest
        // are pointed at their source's and only need the counts.
        //
        // The list is the window plus `min(index_topk, n_cmp)` compressed picks
        // (model.py Indexer.forward: `topk = min(self.index_topk, end_pos //
        // ratio)`), NOT the window plus every compressed position. Up to 512
        // compressed positions the two are the same number, which is why this
        // was invisible below ~1K tokens of context: past it, sparse_attn walked
        // n_cmp - 512 entries the indexer never wrote (zeros: window slot 0,
        // attended hundreds of times) and, past n_kv = 1024, overran its
        // per-head score stride -- the collapse and the NaN logits
        // docs/p3_prefill.md 8.3 item 2 reports (docs/p3_chat.md 5).
        const uint32_t n_sel = std::min<uint32_t>(n_cmp, c.index_topk);
        if (p.ratio == 0 || p.is_index_source) {
            if (auto r = kvs_.set_decode_topk(L, position, n_cmp, n_sel); !r) return r;
        } else if (auto r = kvs_.set_counts(L, n_cmp, c.sliding_window + n_sel); !r) {
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
    // Sized from what the export actually holds, plus the room the remaining
    // steps will need; the compressed half grows by one row a step at ratio 1.
    // The PREFILL record's buffers are max_seq_len // ratio rows (2,074 / 4,149
    // at 4K, 8,513 / 17,026 at 17K) -- more than the widest per-step run
    // `max_compressed()` sees -- and seed_compressed is handed all of them.
    // Planes on the kv sources only (Track R2, docs/p4_kv_ux.md).
    KvStoreConfig kc = KvStoreConfig::for_model(
        c, std::max<uint32_t>(256, std::max(state_->max_compressed(), state_->max_prefill_rows()) + 64));
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

Result<void> Engine::reseed_decode_state() {
    if (!state_) return fail(Err::FailedPrecondition, "no decode state is loaded");
    kvs_.clear();
    if (auto r = state_->seed_prefill(kvs_); !r) return r;
    prefill_loaded_ = true;
    history_     = state_->prompt_ids();
    pub_index_k_ = ced_.empty() ? 0 : ced_.back().cmp_src;
    return {};
}

void Engine::shutdown() {
    // Everything that holds memory from an allocator has to let go before the
    // allocator does, and the allocators before the device.
    if (route_dump_) { std::fclose(route_dump_); route_dump_ = nullptr; }
    // A P3 backfill writes into slab memory from the I/O threads: stop issuing
    // and let what is in flight land before the slabs go back.
    planner_.stop_backfill();
    if (io_.running()) io_.drain();
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
    if (ffn_in_buf_.valid() && ffn_in_alloc_) ffn_in_alloc_->free(ffn_in_buf_);
    ffn_in_alloc_ = nullptr;
    if (logits_.valid()) alloc_a_.free(logits_);
    if (sample_.valid()) alloc_a_.free(sample_);
    if (topk_out_.valid()) alloc_a_.free(topk_out_);
    if (topk_hist_.valid()) alloc_a_.free(topk_hist_);
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

// --- R1 round 2: per-turn reheat (docs/p4_hitrate.md §7) ---------------------

std::string Engine::HeatOrder::to_string() const {
    return std::format("reheat turn {}: decay {:.3f}, {} resident experts ranked in {:.1f} ms "
                       "({} above the floor), {} coldest freed ({} were free), {} keys to the backfill",
                       turn, decay, slots, ms, warm, evicted, free_slots, passed);
}

Result<Engine::HeatOrder> Engine::reheat(float decay) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (decay < 0.0f || decay > 1.0f)
        return fail(Err::InvalidArgument, std::format("reheat decay {} outside [0, 1]", decay));

    const TimePoint t0 = Clock::now();
    HeatOrder out;
    out.decay = decay;
    out.turn = ++reheat_turn_;
    out.free_slots = store_.free_slots();
    // The heat the store kept is the same EWMA the planner has been updating for
    // the chosen experts and the near misses, so it does not need a second
    // accumulator. Decaying it makes the last turn's routing the newest
    // information in the order.
    store_.decay_heat(decay);
    const std::vector<ExpertKey> heat = store_.heat_order();
    out.slots = static_cast<uint32_t>(heat.size());
    // How far down the order is still "this conversation". Everything below is
    // an expert the router has not wanted for several turns, and it is what the
    // pass is allowed to reclaim -- but only that: at 5,500 slots a full resident
    // set is ~1 GB of reads a slot, and a pass that evicted an eighth of it every
    // turn would spend 12 GB of NVMe on a topic that did not change. `decay_heat`
    // renormalises, so `head` is ~1.0 and these thresholds mean the same thing
    // from one turn to the next: 0.05 is a single weak note_heat, 10% of the
    // hottest expert in the cache.
    const float head = out.slots ? store_.slot_for(heat.front())->heat : 0.0f;
    const float floor_heat = std::max(0.05f, 0.1f * head);
    uint32_t warm = 0;                       // the prefix of the order at/above the floor
    while (warm < out.slots && store_.slot_for(heat[warm])->heat >= floor_heat) ++warm;
    // Fill the free slots first; when there are none, reclaim the tail. The tail
    // is where the previous topic lives, and evicting it is what makes room --
    // with no decay and no new routing this is exactly what the LRU would have
    // picked anyway, which is why it is a move of the reheat and not a second
    // eviction policy.
    uint32_t want = out.free_slots ? out.free_slots : std::max<uint32_t>(4, out.slots / 32);
    want = std::min(want, out.slots - warm);

    // What the backfill can actually do something with: keys that are NOT in the
    // cache. `Planner::backfill_pump` skips every resident key it is handed
    // (`if (store_->slot_of(key)) continue;`), so an order built resident-first
    // and then truncated to the budget -- which is what the first version did --
    // is a list of keys the pump walks straight past. That is why the 2,200-slot
    // A/B in docs/p4_hitrate.md §7 measured turn 3 decode hit 0.7210 against
    // 0.7210: on a saturated cache the pass evicted `slots/32` experts and then
    // fetched nothing, so it was a pure cost (the +32 MB/token of the §8 sweep).
    //
    // `StaticHeat` is the startup order (store/static_heat.inc, or a heat file
    // from the last run); its head is the best available estimate of what a
    // never-seen-token expert would score, and it is the only signal this
    // process has about an expert that is not resident -- `heat` lives on the
    // slot, so an evicted expert's heat is gone with it.
    std::vector<ExpertKey> order;
    order.reserve(want + out.free_slots);
    const std::vector<ExpertKey>& heat_table =
        heat_order_.empty() ? (heat_order_ = store::static_heat_order()) : heat_order_;
    for (const ExpertKey& k : heat_table) {
        if (order.size() >= size_t(want) + out.free_slots) break;
        if (!store_.resident(k)) order.push_back(k);
    }
    // Never evict without a candidate to put in the hole. With no non-resident
    // candidate left the pass is a no-op rather than a round of eviction the
    // next turn's misses have to pay back.
    const uint32_t have = static_cast<uint32_t>(order.size());
    want = out.free_slots >= have ? 0 : std::min(want, have - out.free_slots);
    for (uint32_t i = 0; i < want; ++i) {
        const ExpertKey& k = heat[out.slots - 1 - i];
        if (store_.evict_key(k)) ++out.evicted;
    }
    const uint32_t budget = out.free_slots + out.evicted;
    if (order.size() > budget) order.resize(budget);
    out.passed = static_cast<uint32_t>(order.size());
    out.warm = warm;
    out.ms = ms_since(t0);
    if (!order.empty())
        if (auto r = planner_.start_backfill(std::move(order), /*inflight=*/2, /*keep=*/true); !r)
            return std::unexpected(r.error());
    log_info("engine: {}", out.to_string());
    return out;
}

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
    if (open_guard_) { inflight_guard_ = open_guard_; open_guard_ = 0; }
    sub_ms_ += ms_since(t0);
    ++submits_;
    return {};
}

Result<void> Engine::cmd_wait() {
    const TimePoint t0 = Clock::now();
    auto r = fence_.wait(fence_value_, std::chrono::seconds(120));
    wait_ms_ += ms_since(t0);
    if (r && inflight_guard_) {
        store_.set_completed_timeline(inflight_guard_);
        inflight_guard_ = 0;
    }
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
        timings_[L].moe_gpu_ms = span_ms(ts_moe_[L]) + span_ms(ts_moe_early_[L]);
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
    // The indexer wrote every compressed entry of the list sparse_attn just read.
    if (auto r = layer_.verify_after_attention(st); !r) return r;

    // design §7.1 / §7.8: the gate's ids are already in host-coherent memory.
    // Classify them, fetch the misses at P0, wait, host-signal the timeline the
    // MoE dispatch is gated on, then stage and record it.
    const uint32_t topk = c.num_experts_per_tok;
    const auto* ids = static_cast<const uint32_t*>(b.gate_ids.host);
    const auto* wts = static_cast<const float*>(b.gate_weights.host);
    MoeCall call;
    call.layer   = L;
    call.ids     = ids;
    call.weights = wts;
    call.topk    = topk;
    call.x       = layer_.ffn_norm_out();                  // ffn_norm output
    call.y       = nullptr;                                // stays on the GPU
    call.hidden  = c.hidden_size;
    // Track R1 (docs/p4_hitrate.md §4): when this layer waits on NVMe, dispatch
    // A over what is already resident -- the hits and the shared expert -- is
    // submitted before the wait instead of after it. `late` is what it did not
    // cover.
    layer_guard_pending_ = false;
    bool split = false;
    uint32_t late[16];
    uint32_t n_late = 0;
    bool staged = false;   // stage_input has run for this layer
    {
        const TimePoint g0 = Clock::now();
        uint16_t chosen[16];
        for (uint32_t i = 0; i < topk; ++i) chosen[i] = static_cast<uint16_t>(ids[i]);
        if (route_dump_)
            std::memcpy(route_ids_.data() + size_t(L) * topk, chosen, topk * sizeof(uint16_t));
        // design §9.3: the gate kernel already wrote the top-16 ids AND their raw
        // scores into the same two host-coherent buffers (gpu/shaders/gate.slang
        // stage 1, GatePush::record = 16), and the Planner's heat EWMA is defined
        // over them -- "so a bursty expert survives one bad round". They were
        // never passed, so every slot's heat stayed at exactly 0 in the decode
        // path: the reheat pass (docs/p4_hitrate.md §7) had nothing to rank, and
        // any future score-aware policy would have been ranking zeros. Entries
        // [topk, 16) carry the RAW score (only the first six are normalised by
        // route_scale, gate.slang), which is what the EWMA wants.
        uint16_t near_ids[16];
        float    near_scores[16];
        for (uint32_t i = 0; i < 16; ++i) {
            near_ids[i]    = static_cast<uint16_t>(ids[i]);
            near_scores[i] = wts[i];
        }
        store::RouteDecision route;
        route.layer       = L;
        route.chosen      = std::span<const uint16_t>(chosen, topk);
        route.weights     = std::span<const float>(wts, topk);
        route.near_ids    = std::span<const uint16_t>(near_ids, 16);
        route.near_scores = std::span<const float>(near_scores, 16);
        auto plan = planner_.plan_layer(route, token_);
        if (!plan) return std::unexpected(plan.error());
        t.hits       = static_cast<uint32_t>(plan->hits.size());
        t.misses     = static_cast<uint32_t>(plan->misses.size());
        t.miss_bytes = plan->miss_bytes;
        t.gate_ms    = ms_since(g0);
        if (overlap_ && (!plan->issued.empty() || !plan->joined.empty())) {
            if (auto r = moe_.stage_input(call); !r) return r;
            staged = true;
            uint32_t early[16];
            uint32_t n_early = 0;
            for (uint32_t s = 0; s < topk; ++s) {
                const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(ids[s])};
                if (store_.resident(key)) early[n_early++] = s;
                else late[n_late++] = s;
            }
            if (n_late) {
                // The eviction guard (design §5.3, docs/p2_decode.md §13): the
                // early slots are read by a buffer that is in flight while the
                // host waits, so nothing may recycle them until the buffer that
                // carries dispatch B -- the last reader -- has completed.
                guard_layer(L, std::span<const uint32_t>(early, n_early), ids);
                if (auto r = moe_.stage_rows(call, std::span<const uint32_t>(early, n_early)); !r)
                    return r;
                early[n_early++] = topk;                        // the shared expert
                if (auto r = cmd_open(); !r) return r;
                const TimePoint r0 = Clock::now();
                ts_moe_early_[L].begin = cmd_stamp();
                if (auto r = moe_.record_gateup(tok_cmd_, std::span<const uint32_t>(early, n_early)); !r)
                    return r;
                ts_moe_early_[L].end = cmd_stamp();
                rec_ms_ += ms_since(r0);
                if (auto r = cmd_submit(0); !r) return r;
                split = true;
            }
        }
        {
            const TimePoint w0 = Clock::now();
            if (auto r = planner_.wait_layer(*plan); !r) return std::unexpected(r.error());
            t.gate_ms += ms_since(w0);
        }
        // Every routed expert must be resident now. When one is not, say how
        // it got that way -- a hit that a later miss in the same layer evicted
        // and a fill whose read failed look identical at the MoE dispatch.
        for (uint32_t i = 0; i < topk; ++i) {
            const ExpertKey key{static_cast<uint16_t>(L), static_cast<uint16_t>(ids[i])};
            if (store_.resident(key)) continue;
            const bool was_hit = std::find(plan->hits.begin(), plan->hits.end(), key) !=
                                 plan->hits.end();
            auto slot = store_.slot_for(key);
            return fail(Err::Internal,
                        std::format("layer {} expert {} is not resident after the gate: it was "
                                    "a {} this layer, its slot is {}; store: {}", L, ids[i],
                                    was_hit ? "HIT" : "miss",
                                    slot ? std::string(slot_state_name(slot->state))
                                         : std::string("gone"),
                                    store_.stats().to_string()));
        }
    }
    {
        const TimelineValue v = gpu::timeline_value(token_, L);
        auto cur = timeline_.value();
        if (cur && *cur < v)
            if (auto r = timeline_.signal(v); !r) return r;
    }
    profiler_.add_phase(Phase::NvmeStall, Nanos(int64_t(t.gate_ms * 1e6)));

    if (split) {
        // The early dispatch reads the alternate slot list while it runs; it is
        // done by now in all but a pathological case, and this wait proves it
        // before the list is rewritten for the late slots.
        if (auto r = cmd_wait(); !r) return r;
        guard_layer(L, std::span<const uint32_t>(late, n_late), ids);
        if (auto r = moe_.stage_rows(call, std::span<const uint32_t>(late, n_late)); !r) return r;
    } else {
        uint32_t all[16];
        for (uint32_t s = 0; s < topk; ++s) all[s] = s;
        guard_layer(L, std::span<const uint32_t>(all, topk), ids);
        if (staged) {
            if (auto r = moe_.stage_rows(call, std::span<const uint32_t>(all, topk)); !r) return r;
        } else if (auto r = moe_.stage(call); !r) {
            return r;
        }
    }
    // stage_input resets the bridge's timing and stage_rows adds to it, so this
    // is the whole layer's host half whichever way it ran.
    t.moe_host_ms = moe_.timing().host_ms;
    mx_ms_ += moe_.timing().x_read_ms;
    mq_ms_ += moe_.timing().quant_ms;
    mt_ms_ += moe_.timing().table_ms;

    if (auto r = cmd_open(); !r) return r;
    {
        const TimePoint r0 = Clock::now();
        ts_moe_[L].begin = cmd_stamp();
        if (split) {
            if (auto r = moe_.record_gateup(tok_cmd_, std::span<const uint32_t>(late, n_late)); !r)
                return r;
            if (auto r = moe_.record_down(tok_cmd_); !r) return r;
        } else if (auto r = moe_.record(tok_cmd_); !r) {
            return r;
        }
        ts_moe_[L].end = cmd_stamp();
        rec_ms_ += ms_since(r0);
    }
    // This buffer is the last reader of the layer's slots.
    open_guard_ = layer_guard_;
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
    // Track P: a sampled step also reduces the logits to their top set and the
    // tail mass (gpu/shaders/sample_topk.slang), in the same buffer.
    const bool sample = sample_step_ && !sampling_.greedy();
    if (sample) {
        uint64_t* tk = dec_.slots(gpu::DecodeStage::SampleTopK);
        tk[gpu::dslot::kTopKLogits] = logits_.dev_addr;
        tk[gpu::dslot::kTopKOut]    = topk_out_.dev_addr;
        tk[gpu::dslot::kTopKHist]   = topk_hist_.dev_addr;
        gpu::TopKPush kp{c.vocab_size, kTopKDefaultK, 1.0f / sampling_.temperature,
                         kTopKBinsPerLogit};
        if (auto r = tok_cmd_.barrier(); !r) return std::unexpected(r.error());
        if (auto r = dec_.record(tok_cmd_, gpu::DecodeStage::SampleTopK, &kp, sizeof kp, 1); !r)
            return std::unexpected(r.error());
    }
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
    res.greedy_token = out.token;
    res.position = position;
    res.top1     = out.top1;
    res.top2     = out.top2;
    if (sample)
        if (auto r = sample_into(res, position); !r) return std::unexpected(r.error());
    return res;
}

Result<void> Engine::sample_into(DecodeStepResult& res, uint32_t position) {
    const TimePoint t0 = Clock::now();
    const TextConfig& c = model_cfg_.text;
    const float T = sampling_.temperature, P = sampling_.top_p;
    const auto* w = static_cast<const uint32_t*>(topk_out_.host_ptr);
    if (w[0] != c.vocab_size || w[4] != kTopKCapPerThread || w[6] != kTopKBins)
        return fail(Err::Internal,
                    std::format("sample_topk header: rows {} cap {} bins {}", w[0], w[4], w[6]));
    TopKLogits tk;
    tk.rows      = w[0];
    tk.max_logit = std::bit_cast<float>(w[1]);
    tk.tail      = double(std::bit_cast<float>(w[2]));
    tk.bin       = w[3];
    tk.overflow  = w[5] != 0;
    if (!tk.overflow) {
        for (uint32_t t = 0; t < kTopKThreads; ++t) {
            const uint32_t n = std::min(w[kTopKHeaderWords + t], kTopKCapPerThread);
            const uint32_t* seg = w + kTopKHeaderWords + kTopKThreads + 2 * t * kTopKCapPerThread;
            for (uint32_t j = 0; j < n; ++j)
                tk.cand.push_back({seg[2 * j], std::bit_cast<float>(seg[2 * j + 1])});
        }
    }
    res.candidates = static_cast<uint32_t>(tk.cand.size());
    Nucleus nuc = nucleus_from_topk(tk, T, P);
    std::vector<float> host;   // only when the logits have to cross
    auto copy_logits = [&] {
        if (host.empty()) {
            host.resize(c.vocab_size);
            std::memcpy(host.data(), logits_.host_ptr, host.size() * sizeof(float));
        }
    };
    if (check_topk_) {
        copy_logits();
        const TopKLogits emu = emulate_topk(host, kTopKDefaultK, T);
        auto key = [](const TopKLogits& k) {
            std::vector<uint64_t> v;
            for (const auto& cd : k.cand) v.push_back((uint64_t(cd.id) << 32) | std::bit_cast<uint32_t>(cd.logit));
            std::sort(v.begin(), v.end());
            return v;
        };
        res.topk_checked  = true;
        res.topk_mismatch = emu.max_logit != tk.max_logit || emu.bin != tk.bin ||
                            emu.overflow != tk.overflow ||
                            (!tk.overflow && key(emu) != key(tk)) ||
                            std::fabs(emu.tail - tk.tail) > 1e-4 * std::max(1.0, emu.tail);
        if (res.topk_mismatch) {
            ++topk_mismatches_;
            log_warn("sample_topk: kernel and emulation differ at position {}: M {} vs {}, bin {} vs "
                     "{}, {} vs {} candidates, tail {} vs {}", position, tk.max_logit, emu.max_logit,
                     tk.bin, emu.bin, tk.cand.size(), emu.cand.size(), tk.tail, emu.tail);
        }
    }
    if (!nuc.exact) {
        copy_logits();
        nuc = nucleus_from_full(host, T, P);
        res.topk_fallback = true;
    }
    if (nuc.ids.empty()) return fail(Err::Internal, "the nucleus is empty");
    const uint32_t tok = sample_nucleus(nuc, uniform01(sampling_.seed, uint64_t(position) + 1));
    res.token        = tok;
    res.sampled      = true;
    res.nucleus_size = static_cast<uint32_t>(nuc.ids.size());
    res.retained     = nuc.retained;
    res.kept         = nuc.kept;
    for (size_t i = 0; i < nuc.ids.size(); ++i)
        if (nuc.ids[i] == tok) { res.p_token = nuc.p[i]; break; }
    res.sample_ms = ms_since(t0);
    return {};
}

// --- Track P: conversations ---------------------------------------------------

uint32_t Engine::max_context() const {
    return std::min<uint32_t>(kvs_.config().max_context, kMaxIndexPositions);
}

Result<void> Engine::begin_session(const SessionConfig& sc) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    const TextConfig& c = model_cfg_.text;
    // Track R2: planes on the kv sources only, allocated for 4,096 positions
    // and grown by the store as the context does.
    const uint32_t limit = std::min<uint32_t>(std::max<uint32_t>(sc.max_context, 256), kMaxIndexPositions);
    KvStoreConfig kc = KvStoreConfig::for_model(c, limit, std::min<uint32_t>(limit, 4096));
    if (auto r = kvs_.create(alloc_a_, kc); !r) return r;
    auto tables = sc.engram_tables_dir.empty() ? derive_engram_tables(cfg_.model_dir, c)
                                               : EngramTables::load(sc.engram_tables_dir);
    if (!tables)
        return fail(tables.error().code,
                    std::format("engram tables from '{}': {}",
                                sc.engram_tables_dir.empty() ? cfg_.model_dir + "/tokenizer.json"
                                                             : sc.engram_tables_dir,
                                tables.error().message));
    if (auto r = engram_.create(device_, alloc_a_, dec_, manifest_, shards_, io_, pinned_, c,
                                *std::move(tables)); !r)
        return r;
    state_.reset();
    produce_ced_ = true;
    reset_context();
    bool backfill = sc.backfill;
    if (const char* e = std::getenv("DEEPMOE_BACKFILL"); e && *e) backfill = *e != '0';
    // One heat order for the whole process: the startup P3 backfill and every
    // later reheat pass rank non-resident experts by the same table, so
    // `DEEPMOE_HEAT_FILE` (tools/hitrate_bench.py --write-heat / --heat-recent)
    // steers both instead of only the first fill.
    if (heat_order_.empty()) {
        if (const char* hf = std::getenv("DEEPMOE_HEAT_FILE"); hf && *hf)
            heat_order_ = store::static_heat_order(hf);
        if (heat_order_.empty()) heat_order_ = store::static_heat_order();
    }
    if (backfill && store_.free_slots() > 0) {
        std::vector<ExpertKey> order = heat_order_;
        if (auto r = planner_.start_backfill(std::move(order)); !r)
            log_warn("engine: backfill: {}", r.error().str());
        else
            log_info("engine: P3 backfill started into {} free slots", store_.free_slots());
    }
    log_info("engine: session -- KV store {} for {} positions, engram tables from {}",
             human_bytes(kvs_.bytes()), max_context(),
             sc.engram_tables_dir.empty() ? std::string("derived from tokenizer.json") : sc.engram_tables_dir);
    return {};
}

Result<void> Engine::set_context_tokens(std::span<const uint32_t> tokens) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (tokens.size() > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("{} tokens against a {}-position context", tokens.size(), max_context()));
    history_.assign(tokens.begin(), tokens.end());
    pub_index_k_ = ced_.empty() ? 0 : ced_.back().cmp_src;
    return {};
}

void Engine::reset_context() {
    kvs_.clear();
    history_.clear();
    prefill_loaded_ = false;
    pub_index_k_ = ced_.empty() ? 0 : ced_.back().cmp_src;
    // `token_` stays: it is the expert cache's LRU clock (see slow_prefill).
}

Result<DecodeStepResult> Engine::feed(
    std::span<const uint32_t> tokens,
    const std::function<void(uint32_t, const DecodeStepResult&)>& on_step) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (!produce_ced_)
        return fail(Err::FailedPrecondition, "feed needs design 7.4's kernels on (begin_session)");
    if (tokens.empty()) return fail(Err::InvalidArgument, "nothing to feed");
    const uint32_t base = context_length();
    if (uint64_t(base) + tokens.size() > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("{} + {} tokens exceed the session's {}-position context", base,
                                tokens.size(), max_context()));
    DecodeStepResult last{};
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        sample_step_ = (i + 1 == tokens.size());
        auto r = decode_step(tokens[i], base + i, -1);
        sample_step_ = false;
        if (!r) return r;
        if (on_step) on_step(i, *r);
        last = *r;
    }
    return last;
}

Result<void> Engine::seed_from_prefill(const gpu::PrefillHandoff& h) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    const TextConfig& c = model_cfg_.text;
    if (h.layers.size() != c.num_hidden_layers)
        return fail(Err::InvalidArgument,
                    std::format("handoff has {} layers, the model {}", h.layers.size(),
                                c.num_hidden_layers));
    if (h.prompt.size() > max_context())
        return fail(Err::ResourceExhausted,
                    std::format("a {}-token prompt against a {}-position session", h.prompt.size(),
                                max_context()));
    kvs_.clear();
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        const gpu::PrefillHandoff::Layer& l = h.layers[L];
        const uint32_t rows = static_cast<uint32_t>(l.win_kv.size() / c.head_dim);
        if (rows)
            if (auto r = kvs_.seed_window(L, l.win_kv.data(), rows); !r) return r;
        if (!l.cmp_cache.empty() && l.n_cmp) {
            if (auto r = kvs_.seed_compressed(L, l.cmp_cache.data(), l.n_cmp); !r) return r;
            if (auto r = kvs_.seed_index_k(L, l.index_k.data(), l.n_cmp); !r) return r;
        }
        if (!l.cmp_state_kv.empty() && l.ratio)
            if (auto r = kvs_.seed_cmp_state(L, l.cmp_state_kv.data(), l.cmp_state_score.data(),
                                             l.ratio); !r)
                return r;
    }
    history_.assign(h.prompt.begin(), h.prompt.end());
    // At start_pos == 0 every source publishes, so the last one did.
    pub_index_k_ = ced_.empty() ? 0 : ced_.back().cmp_src;
    produce_ced_ = true;
    prefill_loaded_ = false;
    return {};
}

Result<DecodeStepResult> Engine::gpu_prefill(std::span<const uint32_t> prompt, uint32_t replay) {
    if (!gpu_ready_) return fail(Err::FailedPrecondition, "call init_gpu() first");
    if (prompt.empty()) return fail(Err::InvalidArgument, "the prompt is empty");
    if (!engram_.tables().valid())
        return fail(Err::FailedPrecondition, "gpu_prefill needs the engram tables (begin_session)");
    const TextConfig& c = model_cfg_.text;
    const TimePoint t0 = Clock::now();
    gpu::PrefillRunner runner;
    if (auto r = runner.create(device_, alloc_a_, gpu::default_shader_dir()); !r) return std::unexpected(r.error());
    gpu::PrefillConfig pc;
    pc.max_tokens = static_cast<uint32_t>(prompt.size());
    pc.replay     = replay;
    gpu::Prefill pf;
    if (auto r = pf.create(device_, alloc_a_, runner, manifest_, shards_, io_, pinned_, c,
                           &engram_.tables(), pc); !r)
        return std::unexpected(r.error());

    // Track R1 (docs/p4_hitrate.md §3): the experts the prefill streams land in
    // the decode cache under design §9.7.3's rule -- what a global LRU over the
    // prompt's routing table in token order would keep -- and a cached expert
    // is computed from where it is instead of being read again. Stamps are the
    // token-major access order `base + (pos * layers + layer) * topk + rank`.
    const uint64_t n_layers = c.num_hidden_layers, k6 = c.num_experts_per_tok;
    const TokenIndex stamp_base = planner_.reserve_stamps(uint64_t(prompt.size()) * n_layers * k6);
    struct Held { store::StreamAdmit a; TokenIndex stamp = 0; TimelineValue guard = 0; bool open = false; };
    std::vector<Held> held;
    std::set<TimelineValue> open_guards;   // of reservations not yet computed
    gpu::PfExpertSink sink;
    sink.reserve = [&](uint32_t layer, uint32_t expert, uint32_t pos, uint32_t rank) {
        gpu::PfExpertSink::Dest d;
        if (!handoff_) return d;   // DEEPMOE_PREFILL_HANDOFF=0: the transit, as before
        const TokenIndex stamp = stamp_base + (uint64_t(pos) * n_layers + layer) * k6 + rank;
        const TimelineValue guard = ++guard_clock_;
        auto a = planner_.admit_streamed({static_cast<uint16_t>(layer), static_cast<uint16_t>(expert)},
                                         stamp, guard);
        if (!a) {
            log_warn("engine: prefill handoff of ({}, {}): {}", layer, expert, a.error().str());
            return d;
        }
        if (a->kind == store::StreamKind::Drop) return d;
        d.kind = a->kind == store::StreamKind::Fill ? gpu::PfExpertSink::Kind::Fill
                                                    : gpu::PfExpertSink::Kind::Resident;
        d.host = a->addr.host_ptr;
        d.dev  = a->addr.dev_addr;
        d.cookie = held.size();
        held.push_back({*a, stamp, guard, true});
        open_guards.insert(guard);
        return d;
    };
    sink.release = [&](uint32_t, uint32_t, const gpu::PfExpertSink::Dest& d, bool ok) {
        Held& hd = held[d.cookie];
        if (!hd.open) return;
        hd.open = false;
        open_guards.erase(hd.guard);
        if (auto r = planner_.finish_streamed(hd.a, ok, hd.stamp); !r)
            log_warn("engine: prefill handoff settle: {}", r.error().str());
        // Batch i+1's reservations are made before batch i computes, so only
        // the guards below the oldest reservation still waiting may retire.
        store_.set_completed_timeline(open_guards.empty() ? guard_clock_ : *open_guards.begin() - 1);
    };
    pf.expert_sink = &sink;
    const store::PlannerStats ps0 = planner_.stats();
    auto h = pf.run(prompt);
    pf.expert_sink = nullptr;
    // A failed prefill leaves reservations open: release them (a Fill whose
    // bytes may be partial is dropped) so no slot stays Filling.
    for (Held& hd : held)
        if (hd.open) {
            (void)planner_.finish_streamed(hd.a, false, hd.stamp);
            hd.open = false;
        }
    store_.set_completed_timeline(guard_clock_);
    if (!h) return std::unexpected(h.error());
    const gpu::PrefillTimes tm = pf.times();
    pf.destroy();
    runner.destroy();
    {
        const store::PlannerStats ps1 = planner_.stats();
        log_info("engine: prefill handoff -- {} experts already cached, {} kept, {} dropped; "
                 "cache {} resident of {}", ps1.streamed_resident - ps0.streamed_resident,
                 ps1.streamed_filled - ps0.streamed_filled,
                 ps1.streamed_dropped - ps0.streamed_dropped, store_.stats().resident,
                 store_.slot_count());
    }
    if (auto r = seed_from_prefill(*h); !r) return std::unexpected(r.error());

    DecodeStepResult res;
    res.position     = static_cast<uint32_t>(prompt.size()) - 1;
    res.token        = h->first_token;
    res.greedy_token = h->first_token;
    res.top1 = h->top1;
    res.top2 = h->top2;
    res.wall_ms = ms_since(t0);
    if (!sampling_.greedy() && h->logits.size() == c.vocab_size) {
        const TimePoint s0 = Clock::now();
        const Nucleus nuc = nucleus_from_full(h->logits, sampling_.temperature, sampling_.top_p);
        if (!nuc.ids.empty()) {
            res.token = sample_nucleus(nuc, uniform01(sampling_.seed, uint64_t(res.position) + 1));
            res.sampled = true;
            res.topk_fallback = true;
            res.nucleus_size = static_cast<uint32_t>(nuc.ids.size());
            res.retained = 1.0;
            res.kept = nuc.kept;
            for (size_t i = 0; i < nuc.ids.size(); ++i)
                if (nuc.ids[i] == res.token) { res.p_token = nuc.p[i]; break; }
        }
        res.sample_ms = ms_since(s0);
    }
    log_info("engine: GPU prefill of {} tokens in {:.1f} s (expert io {:.1f} s, {} experts / {}), "
             "first token {}", prompt.size(), res.wall_ms / 1e3, tm.expert_io / 1e3, tm.experts_read,
             human_bytes(tm.expert_bytes), res.token);
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
    ts_moe_early_.assign(c.num_hidden_layers, Stamp{});
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
        if (cur && *cur == ~0ull)
            return fail(Err::Internal,
                        "the residency timeline reads UINT64_MAX, which is what a LOST "
                        "device reports -- most likely the expert cache's last path-B "
                        "import exhausted the host heap (see the slab pool's warning)");
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
    if (route_dump_) write_route_record(position);
    ++token_;
    return res;
}

// One record: u32 step (the LRU clock `token_`), u32 position, u16[layers x
// topk] gate ids in gate order, u8[layers] hits. 528 B at 40 x 6.
void Engine::guard_layer(uint32_t layer, std::span<const uint32_t> slots, const uint32_t* ids) {
    if (!layer_guard_pending_) {
        layer_guard_ = ++guard_clock_;
        layer_guard_pending_ = true;
    }
    for (uint32_t s : slots) {
        const ExpertKey key{static_cast<uint16_t>(layer), static_cast<uint16_t>(ids[s])};
        if (auto slot = store_.slot_of(key)) (void)store_.set_guard(*slot, layer_guard_);
    }
}

void Engine::write_route_record(uint32_t position) {
    const uint32_t step = static_cast<uint32_t>(token_);
    std::vector<uint8_t> hits(timings_.size());
    for (size_t L = 0; L < timings_.size(); ++L) hits[L] = static_cast<uint8_t>(timings_[L].hits);
    std::fwrite(&step, sizeof step, 1, route_dump_);
    std::fwrite(&position, sizeof position, 1, route_dump_);
    std::fwrite(route_ids_.data(), sizeof(uint16_t), route_ids_.size(), route_dump_);
    std::fwrite(hits.data(), 1, hits.size(), route_dump_);
    std::fflush(route_dump_);
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

