#include "store/pinned.h"

#include <format>

#include "core/align.h"
#include "core/log.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace deepmoe::store {

uint64_t available_commit_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return ms.ullAvailPageFile;
#else
    return 0;
#endif
}

Result<void> check_commit_available(uint64_t bytes, const char* what) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms))
        return {};   // cannot tell; let the allocation itself decide
    // ullAvailPageFile is the remaining commit charge -- the quantity design
    // §5.2 measured as the real ceiling, and the one a bigger pagefile raises.
    const uint64_t avail = ms.ullAvailPageFile;
    const uint64_t need  = bytes + (256ull << 20);   // headroom for everything else
    if (avail >= need) {
        log_info("{}: {:.2f} GB wanted, {:.2f} GB of commit available",
                 what, bytes / 1e9, avail / 1e9);
        return {};
    }
    return fail(Err::ResourceExhausted,
                std::format("{} needs {:.2f} GB but only {:.2f} GB of Windows commit is "
                            "available (total limit {:.2f} GB). This is the constraint "
                            "design 5.2 measured: raise the pagefile, see the pagefile "
                            "section of docs/build.md. Nothing was allocated.",
                            what, bytes / 1e9, avail / 1e9, ms.ullTotalPageFile / 1e9));
#else
    (void)bytes; (void)what;
    return {};
#endif
}

Result<void> PinnedStore::init(std::unique_ptr<SlabBacking> backing, const PinnedConfig& cfg) {
    reset();
    if (!backing) return fail(Err::InvalidArgument, "pinned store needs a slab backing");
    if (cfg.region_bytes < kPageSize || !is_aligned(cfg.region_bytes))
        return fail(Err::InvalidArgument, "region_bytes must be a 4 KiB multiple");
    backing_ = std::move(backing);
    cfg_ = cfg;
    return {};
}

void PinnedStore::reset() {
    if (backing_)
        for (const SlabMemory& m : regions_) backing_->release(m);
    regions_.clear();
    tensors_.clear();
    backing_.reset();
    used_in_region_ = 0;
    cur_region_bytes_ = 0;
    bytes_loaded_ = 0;
}

Result<SlotAddress> PinnedStore::place(uint64_t bytes) {
    const uint64_t need = align_up(bytes);
    // The head is 1.32 GB on its own (design §2.2), which is bigger than any
    // sane default region and still under the 2 GiB maxMemoryAllocationSize of
    // §1.1, so an oversized tensor gets a region of its own rather than an
    // error. It is placed first in that region and the bump pointer is left at
    // the end, so nothing else lands behind it.
    const uint64_t want = need > cfg_.region_bytes ? need : cfg_.region_bytes;
    if (regions_.empty() || used_in_region_ + need > cur_region_bytes_) {
        auto m = backing_->allocate(want);
        if (!m) return std::unexpected(m.error());
        if (!m->host_ptr) {
            backing_->release(*m);
            // design §7.9.1 constraint 2, restated for the pinned set: the
            // IoEngine writes through the host pointer, so a device-only
            // allocation is unusable however fast the GPU reads it.
            return fail(Err::Internal, "pinned region has no host pointer to load into");
        }
        regions_.push_back(*m);
        cur_region_bytes_ = want;
        used_in_region_ = 0;
    }
    const SlabMemory& r = regions_.back();
    SlotAddress a;
    a.host_ptr = static_cast<std::byte*>(r.host_ptr) + used_in_region_;
    a.dev_addr = r.device_address == kNoDeviceAddress ? kNoDeviceAddress
                                                      : r.device_address + used_in_region_;
    used_in_region_ += need;
    return a;
}

namespace {

// One tensor plane on its way in: where it goes and what to read.
struct Plan {
    std::string name;
    bool        is_scale = false;
    AlignedRead read{};
    SlotAddress dst{};
};

}  // namespace

Result<void> PinnedStore::load(const Manifest& manifest, const ShardSet& shards,
                               storage::IoEngine& io, const std::vector<std::string>& names) {
    if (!backing_) return fail(Err::FailedPrecondition, "pinned store is not initialised");

    // Plan first, allocate second, submit third: a name that is not in the
    // manifest must fail before any memory is committed.
    std::vector<Plan> plans;
    uint64_t want = 0;
    for (const std::string& n : names) {
        if (tensors_.contains(n)) continue;
        auto t = manifest.require_tensor(n);
        if (!t) return std::unexpected(t.error());
        auto rd = manifest.tensor_read(n);
        if (!rd) return std::unexpected(rd.error());
        plans.push_back(Plan{n, false, *rd, {}});
        want += align_up(rd->aligned_bytes);
        if ((*t)->scale.present()) {
            auto sr = manifest.tensor_scale_read(n);
            if (!sr) return std::unexpected(sr.error());
            plans.push_back(Plan{n, true, *sr, {}});
            want += align_up(sr->aligned_bytes);
        }
    }
    if (plans.empty()) return {};
    if (auto r = check_commit_available(want, "pinned weights"); !r) return r;

    for (Plan& p : plans) {
        auto a = place(p.read.aligned_bytes);
        if (!a) return std::unexpected(a.error());
        p.dst = *a;
    }

    // One request per plane. The manifest widened each read to sector
    // boundaries already, so the I/O layer sees nothing unusual (design §5.1.2).
    std::vector<std::future<storage::IoResult>> pending;
    pending.reserve(plans.size());
    for (const Plan& p : plans) {
        auto file = shards.require(p.read.file);
        if (!file) return std::unexpected(file.error());
        storage::IoRequest req;
        req.priority = cfg_.priority;
        req.file     = *file;
        req.file_off = p.read.aligned_off;
        req.bytes    = p.read.aligned_bytes;
        req.dst      = p.dst.host_ptr;
        auto f = io.submit_future(req);
        if (!f) return std::unexpected(f.error());
        pending.push_back(std::move(*f));
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        const storage::IoResult r = pending[i].get();
        if (!r.ok())
            return fail(r.status.code,
                        std::format("pinned load of '{}'{}: {}", plans[i].name,
                                    plans[i].is_scale ? " (scale)" : "", r.status.message));
    }

    for (const Plan& p : plans) {
        const TensorEntry* t = manifest.tensor(p.name);
        PinnedTensor& pt = tensors_[p.name];
        pt.dtype = t->dtype;
        pt.shape = t->shape;
        const uint64_t off = p.read.skew;
        if (p.is_scale) {
            pt.scale = p.dst.dev_addr == kNoDeviceAddress ? kNoDeviceAddress
                                                          : p.dst.dev_addr + off;
            pt.scale_host = static_cast<const std::byte*>(p.dst.host_ptr) + off;
            pt.scale_bytes = p.read.bytes;
        } else {
            pt.data = p.dst.dev_addr == kNoDeviceAddress ? kNoDeviceAddress
                                                         : p.dst.dev_addr + off;
            pt.data_host = static_cast<const std::byte*>(p.dst.host_ptr) + off;
            pt.data_bytes = p.read.bytes;
        }
        bytes_loaded_ += p.read.bytes;
    }
    return {};
}

const PinnedTensor* PinnedStore::find(std::string_view name) const {
    auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : &it->second;
}

Result<const PinnedTensor*> PinnedStore::require(std::string_view name) const {
    if (const PinnedTensor* t = find(name)) return t;
    return fail(Err::NotFound, std::format("'{}' is not in the pinned set", name));
}

// --- name lists -------------------------------------------------------------

namespace {
void add_if_present(const Manifest& m, std::vector<std::string>& out, std::string n) {
    if (m.tensor(n)) out.push_back(std::move(n));
}
}  // namespace

std::vector<std::string> pinned_layer_tensors(const Manifest& manifest, uint32_t layer) {
    const std::string p = std::format("layers.{}", layer);
    std::vector<std::string> out;
    // attention (design §2.3: 126.6 MB a layer, every byte read every token)
    for (const char* s : {"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
                          "attn.wo_a.weight", "attn.wo_b.weight", "attn.attn_sink",
                          "attn.q_norm.weight", "attn.kv_norm.weight",
                          "attn_norm.weight", "ffn_norm.weight"})
        add_if_present(manifest, out, std::format("{}.{}", p, s));
    // mHC and the router
    for (const char* s : {"hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                          "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
                          "ffn.gate.weight", "ffn.gate.bias"})
        add_if_present(manifest, out, std::format("{}.{}", p, s));
    // shared expert
    for (const char* s : {"ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
                          "ffn.shared_experts.w3.weight"})
        add_if_present(manifest, out, std::format("{}.{}", p, s));
    // compressor / indexer, present only on a source layer
    for (const char* s : {"attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
                          "attn.compressor.norm.weight", "attn.indexer.wq_b.weight",
                          "attn.indexer.wk.weight", "attn.indexer.k_norm.weight",
                          "attn.indexer.weights_proj.weight"})
        add_if_present(manifest, out, std::format("{}.{}", p, s));
    // engram, present only on layers 1 and 14. The 98 GB row table is NOT
    // pinned -- design §2.2 keeps it on NVMe and fetches 24 rows a token.
    for (const char* s : {"engram.wkv.weight", "engram.q_weight", "engram.k_weight"})
        add_if_present(manifest, out, std::format("{}.{}", p, s));
    return out;
}

std::vector<std::string> pinned_global_tensors(const Manifest& manifest) {
    std::vector<std::string> out;
    for (const char* s : {"embed.weight", "norm.weight", "head.weight"})
        add_if_present(manifest, out, s);
    return out;
}

uint64_t pinned_bytes(const Manifest& manifest, const std::vector<std::string>& names) {
    uint64_t n = 0;
    for (const std::string& s : names)
        if (const TensorEntry* t = manifest.tensor(s)) n += t->bytes + t->scale.bytes;
    return n;
}

}  // namespace deepmoe::store
