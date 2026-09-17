#include "gpu/vulkan/prefill_kernels.h"

#include <cstring>
#include <format>
#include <future>

#include "core/align.h"
#include "cpu/dequant.h"

namespace deepmoe::gpu {

void pf_act_quant_host(const float* x, uint32_t n, uint32_t k, uint16_t* q16, float* scales) {
    const uint32_t blocks = k / 32;
    uint8_t bytes[32];
    for (uint32_t m = 0; m < n; ++m) {
        for (uint32_t b = 0; b < blocks; ++b) {
            const float* v = x + size_t(m) * k + b * 32;
            const float s = cpu::act_quant_block(v, 32, bytes, nullptr);
            scales[size_t(m) * blocks + b] = s;
            for (uint32_t i = 0; i < 32; ++i)
                q16[size_t(m) * k + b * 32 + i] =
                    cpu::float_to_fp16(cpu::fp8_e4m3_to_float(bytes[i]));
        }
    }
}

Result<PfExpert> pf_load_expert(MemoryAllocator& alloc, const Manifest& manifest,
                                const store::ShardSet& shards, storage::IoEngine& io,
                                ExpertKey key) {
    auto e = manifest.require_expert(key);
    if (!e) return std::unexpected(e.error());
    const ExpertEntry& ent = **e;
    auto b = alloc.allocate(align_up(ent.slot_bytes, kPageSize), true, true);
    if (!b) return std::unexpected(b.error());
    PfExpert out;
    out.buf = *b;
    if (reinterpret_cast<uintptr_t>(out.buf.host_ptr) % kPageSize) {
        out.release(alloc);
        return fail(Err::Internal, "expert buffer is not page aligned");
    }
    std::vector<std::future<storage::IoResult>> futs;
    for (const Run& r : ent.runs) {
        auto f = shards.require(r.file);
        if (!f) { out.release(alloc); return std::unexpected(f.error()); }
        storage::IoRequest req;
        req.key      = key;
        req.priority = IoPriority::BlockingMiss;
        req.file     = *f;
        req.file_off = r.aligned_off;
        req.bytes    = r.aligned_bytes;
        req.dst      = static_cast<std::byte*>(out.buf.host_ptr) + r.slot_offset;
        auto fut = io.submit_future(req);
        if (!fut) { out.release(alloc); return std::unexpected(fut.error()); }
        futs.push_back(std::move(*fut));
    }
    for (auto& f : futs)
        if (!f.get().ok()) { out.release(alloc); return fail(Err::Io, "expert read failed"); }
    for (uint32_t p = 0; p < 6; ++p) {
        out.part_off[p] = ent.offset_of(static_cast<ExpertPart>(p));
        out.addr[p] = out.buf.dev_addr + out.part_off[p];
    }
    return out;
}

#if !defined(DEEPMOE_ENABLE_VULKAN)

Result<void> PrefillRunner::create(Device&, MemoryAllocator&, const std::string&, uint32_t) {
    return fail(Err::Unavailable, "built without DEEPMOE_ENABLE_VULKAN");
}
void PrefillRunner::destroy() {}
Result<uint32_t> PrefillRunner::kernel(const PfKernel&) { return fail(Err::Unavailable, "no vulkan"); }
uint64_t* PrefillRunner::slots(uint32_t) { return nullptr; }
const PfKernel* PrefillRunner::spec(uint32_t) const { return nullptr; }
Result<void> PrefillRunner::record(CommandBuffer&, uint32_t, const void*, uint32_t, uint32_t,
                                   uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}
Result<void> PrefillRunner::dispatch_now(uint32_t, const void*, uint32_t, uint32_t, uint32_t) {
    return fail(Err::Unavailable, "no vulkan");
}

#else

static_assert(sizeof(PfGemmPush) <= kPfPushBytes);
static_assert(sizeof(PfJob) == 64);

Result<void> PrefillRunner::create(Device& device, MemoryAllocator& alloc,
                                   const std::string& shader_dir, uint32_t max_kernels) {
    destroy();
    if (!device.valid()) return fail(Err::FailedPrecondition, "device is not created");
    // Every prefill GEMM reduces a row across the 32 lanes of one Wave32
    // (prefill_gemm.slang), and the cooperative matrices are subgroup-scoped.
    if (device.caps().subgroup_size_control == false)
        return fail(Err::FailedPrecondition,
                    "the prefill kernels need VK_EXT_subgroup_size_control for Wave32");
    device_ = &device;
    alloc_  = &alloc;
    shader_dir_ = shader_dir;
    max_kernels_ = max_kernels;

    auto t = alloc.allocate(uint64_t(max_kernels) * kPfKernelStride, true, false);
    if (!t) { destroy(); return std::unexpected(t.error()); }
    table_ = *t;
    std::memset(table_.host_ptr, 0, static_cast<size_t>(table_.bytes));
    if (auto r = descriptors_.create(device, max_kernels, max_kernels); !r) { destroy(); return r; }
    if (auto r = pool_.create(device); !r) { destroy(); return r; }
    return {};
}

void PrefillRunner::destroy() {
    own_cmd_valid_ = false;
    own_cmd_ = CommandBuffer{};
    pool_.destroy();
    descriptors_.destroy();
    for (auto& p : pipes_) if (p) p->destroy();
    pipes_.clear();
    sets_.clear();
    index_.clear();
    if (alloc_ && table_.valid()) alloc_->free(table_);
    table_ = GpuBuffer{};
    device_ = nullptr;
    alloc_  = nullptr;
}

const PfKernel* PrefillRunner::spec(uint32_t handle) const {
    for (const auto& [k, h] : index_)
        if (h == handle) return &k;
    return nullptr;
}

Result<uint32_t> PrefillRunner::kernel(const PfKernel& k) {
    if (auto it = index_.find(k); it != index_.end()) return it->second;
    if (!device_) return fail(Err::FailedPrecondition, "runner is not created");
    if (pipes_.size() >= max_kernels_)
        return fail(Err::ResourceExhausted,
                    std::format("prefill runner: more than {} distinct kernels", max_kernels_));
    PipelineSpec ps;
    ps.m             = 1;
    ps.lanes_per_row = 32;
    ps.rows_per_wg   = 8;
    ps.subgroup_size = 32;
    ps.extra = {k.stage, k.wfmt, k.xfmt, k.tile, k.extra0, k.extra1, k.extra2, k.extra3};
    PipelineLayoutSpec la;
    la.storage_buffers    = 1;
    la.push_constant_size = kPfPushBytes;
    auto p = std::make_unique<Pipeline>();
    if (auto r = p->create(*device_, shader_dir_ + "/" + k.spv + ".spv", la, ps); !r)
        return fail(r.error().code, std::format("{} stage {} w{} x{} t{}: {}", k.spv, k.stage,
                                                k.wfmt, k.xfmt, k.tile, r.error().message));
    const uint32_t i = static_cast<uint32_t>(pipes_.size());
    std::vector<BufferBinding> b(1);
    b[0] = {0, uint64_t(i) * kPfKernelStride, kPfKernelStride, table_.buffer};
    auto set = descriptors_.allocate(*p, b);
    if (!set) return std::unexpected(set.error());
    pipes_.push_back(std::move(p));
    sets_.push_back(*set);
    index_.emplace(k, i);
    return i;
}

uint64_t* PrefillRunner::slots(uint32_t h) {
    if (!table_.host_ptr || h >= max_kernels_) return nullptr;
    return reinterpret_cast<uint64_t*>(static_cast<std::byte*>(table_.host_ptr) +
                                       uint64_t(h) * kPfKernelStride);
}

Result<void> PrefillRunner::record(CommandBuffer& cmd, uint32_t h, const void* push,
                                   uint32_t bytes, uint32_t gx, uint32_t gy) {
    if (h >= pipes_.size()) return fail(Err::InvalidArgument, "unknown prefill kernel");
    if (bytes > kPfPushBytes) return fail(Err::InvalidArgument, "push constants exceed 64 B");
    if (gx == 0 || gy == 0) return fail(Err::InvalidArgument, "zero workgroups");
    // A 1-D kernel past the per-axis limit (a 17K prompt's hc_post is 85,050
    // workgroups) goes out as rows of kWgRowX; its shader indexes through
    // gid_linear (prefill_common.slang) and discards the overshoot.
    if (gy == 1 && gx > 65535) {
        gy = (gx + kPfWgRowX - 1) / kPfWgRowX;
        gx = kPfWgRowX;
    }
    if (gx > 65535 || gy > 65535)
        return fail(Err::InvalidArgument,
                    std::format("{} x {} workgroups is past the 65535 dispatch limit", gx, gy));
    if (auto r = cmd.bind(*pipes_[h], sets_[h]); !r) return r;
    if (bytes) {
        if (auto r = cmd.push(*pipes_[h], push, bytes); !r) return r;
    }
    return cmd.dispatch(gx, gy);
}

Result<void> PrefillRunner::dispatch_now(uint32_t h, const void* push, uint32_t bytes,
                                         uint32_t gx, uint32_t gy) {
    // One buffer, re-begun: the pool is RESET_COMMAND_BUFFER, and acquiring a
    // fresh one per call (DecodeRunner::dispatch_now) leaks one per dispatch.
    if (!own_cmd_valid_) {
        auto cb = pool_.acquire();
        if (!cb) return std::unexpected(cb.error());
        own_cmd_ = *cb;
        own_cmd_valid_ = true;
    }
    CommandBuffer& cmd = own_cmd_;
    if (auto r = cmd.begin(); !r) return r;
    if (auto r = record(cmd, h, push, bytes, gx, gy); !r) return r;
    if (auto r = cmd.end(); !r) return r;
    return submit_and_wait(*device_, cmd);
}

#endif  // DEEPMOE_ENABLE_VULKAN

}  // namespace deepmoe::gpu

// =============================================================================
// Prefill -- docs/p3_prefill.md §2
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/json.h"
#include "cpu/gate.h"
#include "model/layout.h"
#include "model/v41_config.h"
#include "runtime/engram.h"
#include "runtime/rope.h"
#include "store/pinned.h"

namespace deepmoe::gpu {

#if defined(DEEPMOE_ENABLE_VULKAN)

namespace {

using Clk = std::chrono::steady_clock;
double ms_since(Clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count();
}

constexpr uint32_t kHc = 4;
constexpr uint32_t kMix = 24;
constexpr uint32_t kSharedJob = 0;     // job 0 is the shared expert; routed experts start at 1

uint32_t groups_for(uint64_t threads) { return static_cast<uint32_t>((threads + 255) / 256); }
// prefill_coopmat.slang runs 32 threads a workgroup (one wave).
uint32_t groups_for32(uint64_t threads) { return static_cast<uint32_t>((threads + 31) / 32); }

// prefill_coopmat stage 0's CmTokTiles, clamped to what the shader unrolls,
// and the workgroup count for `n` tokens at that geometry. kPfRowSlack rows of
// every staged plane cover the padding the last workgroup reads and writes.
uint32_t pf_tok_tiles(uint32_t want) { return want < 1 ? 1u : want > 8 ? 8u : want; }
uint32_t pf_tok_groups(uint32_t n, uint32_t tt) { return (n + 16 * tt - 1) / (16 * tt); }

// A device address `rows` rows into a buffer of `stride`-byte rows.
uint64_t at(const GpuBuffer& b, uint64_t rows, uint64_t stride) { return b.dev_addr + rows * stride; }
float* fptr(const GpuBuffer& b, uint64_t elems = 0) { return static_cast<float*>(b.host_ptr) + elems; }

}  // namespace

Result<GpuBuffer> Prefill::scratch(uint64_t bytes) {
    auto b = alloc_->allocate(align_up(std::max<uint64_t>(bytes, kPageSize), kPageSize), true, true);
    if (!b) return std::unexpected(b.error());
    std::memset(b->host_ptr, 0, static_cast<size_t>(b->bytes));
    owned_.push_back(*b);
    return *b;
}

void Prefill::destroy() {
    if (alloc_)
        for (GpuBuffer& b : owned_) if (b.valid()) alloc_->free(b);
    owned_.clear();
    sources_.clear();
    b_ = Bufs{};
    qp_.destroy();
    qp_ok_ = false;
    marks_.clear();
    cmd_valid_ = false;
    cmd_ = CommandBuffer{};
    device_ = nullptr;
    alloc_ = nullptr;
}

Result<void> Prefill::create(Device& device, MemoryAllocator& alloc, PrefillRunner& runner,
                             const Manifest& manifest, const store::ShardSet& shards,
                             storage::IoEngine& io, const store::PinnedStore& pinned,
                             const TextConfig& cfg, const runtime::EngramTables* engram,
                             const PrefillConfig& pcfg) {
    destroy();
    device_ = &device; alloc_ = &alloc; runner_ = &runner; manifest_ = &manifest;
    shards_ = &shards; io_ = &io; pinned_ = &pinned; cfg_ = &cfg; engram_ = engram;
    pcfg_ = pcfg;
    w16_src_ = 0;
    if (pcfg_.tile == 0 || pcfg_.tile > 32) return fail(Err::InvalidArgument, "tile must be 1..32");
    const uint64_t N = pcfg_.max_tokens, B = std::min(pcfg_.query_block, pcfg_.max_tokens);
    const uint64_t dim = cfg.hidden_size, inter = cfg.moe_intermediate_size;
    const uint64_t hd = cfg.head_dim, win = cfg.sliding_window;
    const uint64_t n_idx = win + cfg.index_topk;
    const uint64_t k6 = cfg.num_experts_per_tok;
    struct Want { GpuBuffer* b; uint64_t bytes; };
    const Want wants[] = {
        {&b_.h_a, N * kHc * dim * 4}, {&b_.h_b, N * kHc * dim * 4},
        {&b_.h_in_copy, pcfg_.probe_layers ? N * kHc * dim * 4 : 4096},
        {&b_.x, N * dim * 4}, {&b_.rs, N * 4}, {&b_.mix_raw, N * kMix * 4},
        {&b_.mix_a, N * kMix * 4}, {&b_.mix_f, N * kMix * 4}, {&b_.mix_prev, N * kMix * 4},
        {&b_.xq, N * dim * 2}, {&b_.xs, N * (dim / 32) * 4},
        {&b_.kv_raw, N * hd * 4}, {&b_.kv_norm, N * hd * 4}, {&b_.kv, N * hd * 4},
        {&b_.ckv, N * hd * 4}, {&b_.cscore, N * hd * 4}, {&b_.latent_pre, N * hd * 4},
        {&b_.latent, N * hd * 4}, {&b_.key_raw, N * cfg.index_head_dim * 4},
        {&b_.key_norm, N * cfg.index_head_dim * 4},
        {&b_.qr_raw, N * cfg.q_lora_rank * 4}, {&b_.qr, N * cfg.q_lora_rank * 4},
        {&b_.qrq, N * cfg.q_lora_rank * 2}, {&b_.qrs, N * (cfg.q_lora_rank / 32) * 4},
        {&b_.q, B * cfg.num_attention_heads * hd * 4},
        {&b_.iq, B * cfg.index_n_heads * cfg.index_head_dim * 4},
        {&b_.iw, B * cfg.index_n_heads * 4}, {&b_.iscore, B * N * 4},
        {&b_.idx, B * n_idx * 4},
        {&b_.score, B * cfg.num_attention_heads * ((n_idx + 15) / 16 * 16) * 4},
        {&b_.o, B * cfg.num_attention_heads * hd * 4},
        {&b_.woa, B * cfg.o_groups * cfg.o_lora_rank * 4},
        {&b_.woaq, B * cfg.o_groups * cfg.o_lora_rank * 2},
        {&b_.woas, B * (cfg.o_groups * cfg.o_lora_rank / 32) * 4},
        {&b_.attn, N * dim * 4},
        {&b_.fx, N * dim * 4}, {&b_.fxq, N * dim * 2}, {&b_.fxs, N * (dim / 32) * 4},
        {&b_.gate, N * cfg.n_routed_experts * 4},
        {&b_.gids, N * k6 * 4}, {&b_.gwts, N * k6 * 4},
        {&b_.y, N * dim * 4},
        {&b_.hplane, (N * (k6 + 1) + kPfRowSlack) * inter * 4},
        {&b_.hq, (N * (k6 + 1) + kPfRowSlack) * inter * 2},
        {&b_.hs, (N * (k6 + 1) + kPfRowSlack) * (inter / 32) * 4},
        {&b_.csr_idx, (N * (k6 + 1) + 64) * 4}, {&b_.csr_rw, (N * (k6 + 1) + 64) * 4},
        // kPfRowSlack, not 32: a cooperative-matrix workgroup covers
        // 16 * CmTokTiles tokens, so the last one of a dispatch reads and
        // writes up to that many rows past n (their products only reach rows
        // nobody copies, but the memory has to exist).
        {&b_.x16, std::max((N + kPfRowSlack) * dim,
                           (B + kPfRowSlack) * cfg.o_groups * cfg.o_lora_rank) * 2},
        {&b_.h16, (N + kPfRowSlack) * inter * 2},
        {&b_.gu, 2 * (N + kPfRowSlack) * inter * 4},
        {&b_.dout, std::max({(N + kPfRowSlack) * dim,
                             (B + kPfRowSlack) * cfg.num_attention_heads * hd,
                             (N + kPfRowSlack) * cfg.q_lora_rank}) * 4},
        {&b_.w16, std::max({dim * inter, uint64_t(cfg.num_attention_heads) * hd * cfg.q_lora_rank,
                            dim * cfg.o_groups * cfg.o_lora_rank}) * 2},
        {&b_.jobs, (uint64_t(cfg.n_routed_experts) + 2) * 64},
        {&b_.eng_x, N * 6144 * 4}, {&b_.eng_xq, N * 6144 * 2}, {&b_.eng_xs, N * 192 * 4},
        {&b_.eng_kv, N * (kHc + 1) * dim * 4},
        {&b_.transit, uint64_t(2) * pcfg_.transit_slots * layout::kExpertSlotBytes},
        {&b_.rope_win, (N + 8) * cfg.qk_rope_head_dim * 4},
        {&b_.rope_cmp, (N + 8) * cfg.qk_rope_head_dim * 4},
        {&b_.logits, uint64_t(cfg.vocab_size) * 4}, {&b_.nrm, dim * 4},
        {&b_.q16, B * cfg.num_attention_heads * hd * 2},
        {&b_.g16, B * ((n_idx + 15) / 16 * 16) * hd * 2},
        {&b_.p16, B * cfg.num_attention_heads * ((n_idx + 15) / 16 * 16) * 2},
        {&b_.inv, B * cfg.num_attention_heads * 4},
    };
    for (const Want& w : wants) {
        auto b = scratch(w.bytes);
        if (!b) return fail(b.error().code, std::format("prefill buffers ({} B): {}", w.bytes,
                                                        b.error().message));
        *w.b = *b;
    }
    sources_.assign(cfg.num_hidden_layers, SourceState{});
    for (uint32_t L = 0; L < cfg.num_hidden_layers; ++L) {
        if (!cfg.is_kv_source(L)) continue;
        const uint64_t g = N / std::max<uint32_t>(1, cfg.compress_ratio(L));
        auto c = scratch(g * hd * 4);
        auto k = scratch(g * cfg.index_head_dim * 4);
        if (!c || !k) return fail(Err::ResourceExhausted, "compressed-KV buffers");
        sources_[L].cache = *c;
        sources_[L].keys = *k;
    }
    qp_ok_ = static_cast<bool>(qp_.create(device, 64));
    return build_rope(static_cast<uint32_t>(N + 8));
}

Result<void> Prefill::build_rope(uint32_t positions) {
    const uint32_t rd = cfg_->qk_rope_head_dim;
    const runtime::RopeConfig w = runtime::rope_for_layer(0, rd, cfg_->rope_theta,
                                                          cfg_->compress_rope_theta,
                                                          cfg_->rope_scaling.original_max_position,
                                                          cfg_->rope_scaling.factor);
    const runtime::RopeConfig c = runtime::rope_for_layer(1, rd, cfg_->rope_theta,
                                                          cfg_->compress_rope_theta,
                                                          cfg_->rope_scaling.original_max_position,
                                                          cfg_->rope_scaling.factor);
    std::vector<float> tw(size_t(positions) * rd), tc(size_t(positions) * rd);
    for (uint32_t p = 0; p < positions; ++p) {
        const std::vector<float> a = runtime::rope_table(w, p), b = runtime::rope_table(c, p);
        std::memcpy(tw.data() + size_t(p) * rd, a.data(), rd * sizeof(float));
        std::memcpy(tc.data() + size_t(p) * rd, b.data(), rd * sizeof(float));
    }
    std::memcpy(b_.rope_win.host_ptr, tw.data(), tw.size() * sizeof(float));
    std::memcpy(b_.rope_cmp.host_ptr, tc.data(), tc.size() * sizeof(float));
    rope_positions_ = positions;
    return {};
}

Result<PfWeight> Prefill::weight(const std::string& name, uint32_t fmt) const {
    auto t = pinned_->require(name);
    if (!t) return std::unexpected(t.error());
    PfWeight w;
    w.data = (*t)->data;
    w.scale = (*t)->scale;
    w.fmt = fmt;
    w.rows = (*t)->shape.size() > 0 ? static_cast<uint32_t>((*t)->shape[0]) : 0;
    w.k = (*t)->shape.size() > 1 ? static_cast<uint32_t>((*t)->shape[1]) : 1;
    w.bytes = (*t)->data_bytes + (*t)->scale_bytes;
    return w;
}

Result<void> Prefill::flush_one(uint32_t kernel, const void* push, uint32_t bytes, uint32_t gx,
                                uint32_t gy) {
    if (!cmd_valid_) {
        auto cb = runner_->pool().acquire();
        if (!cb) return std::unexpected(cb.error());
        cmd_ = *cb;
        cmd_valid_ = true;
    }
    const auto t0 = Clk::now();
    if (auto r = cmd_.begin(); !r) return r;
    if (auto r = runner_->record(cmd_, kernel, push, bytes, gx, gy); !r) return r;
    if (auto r = cmd_.end(); !r) return r;
    ++times_.dispatches;
    ++times_.submits;
    auto r = submit_and_wait(*device_, cmd_);
    std::string key = std::move(tag_);
    tag_.clear();
    if (key.empty()) {
        const PfKernel* sp = runner_->spec(kernel);
        key = sp ? std::format("{} s{}", sp->spv, sp->stage) : "?";
    }
    auto& slot = times_.per_op[key];
    slot.first += ms_since(t0);
    ++slot.second;
    return r;
}

Result<void> Prefill::cmd_open() {
    if (!cmd_valid_) {
        auto cb = runner_->pool().acquire();
        if (!cb) return std::unexpected(cb.error());
        cmd_ = *cb;
        cmd_valid_ = true;
    }
    if (auto r = cmd_.begin(); !r) return r;
    marks_.clear();
    if (qp_ok_) {
        if (auto r = cmd_.reset_queries(qp_, 0, qp_.count()); !r) return r;
        if (auto r = cmd_.write_timestamp(qp_, 0, false); !r) return r;
    }
    return {};
}

Result<void> Prefill::rec(uint32_t kernel, const void* push, uint32_t bytes, uint32_t gx, uint32_t gy) {
    if (auto r = runner_->record(cmd_, kernel, push, bytes, gx, gy); !r) return r;
    ++times_.dispatches;
    return cmd_.barrier();
}

void Prefill::mark(const char* name) {
    if (!qp_ok_ || marks_.size() + 2 > qp_.count()) return;
    marks_.push_back(name);
    (void)cmd_.write_timestamp(qp_, static_cast<uint32_t>(marks_.size()), true);
}

Result<void> Prefill::cmd_close() {
    if (auto r = cmd_.end(); !r) return r;
    ++times_.submits;
    if (auto r = submit_and_wait(*device_, cmd_); !r) return r;
    if (qp_ok_ && !marks_.empty()) {
        auto v = qp_.read_range(0, static_cast<uint32_t>(marks_.size()) + 1);
        if (v) {
            const double ns = device_->caps().timestamp_period_ns;
            for (size_t i = 0; i < marks_.size(); ++i) {
                auto& slot = times_.per_op[std::string("gpu: ") + marks_[i]];
                slot.first += double((*v)[i + 1] - (*v)[i]) * ns / 1e6;
                ++slot.second;
            }
        }
    }
    marks_.clear();
    return {};
}

Result<void> Prefill::op_act_quant(uint64_t x, uint32_t n, uint32_t d, uint64_t q16, uint64_t sc) {
    auto k = runner_->kernel({"prefill_elem", 0});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = x; s[1] = q16; s[2] = sc;
    PfElemPush p; p.n = n; p.d = d;
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(n) * (d / 32)));
}

Result<void> Prefill::op_gemm(const PfWeight& w, uint32_t xfmt, uint64_t x, uint64_t xs,
                              uint32_t n, uint32_t x_stride, uint64_t y, uint32_t flags,
                              float out_scale, uint32_t rows_per_group, uint64_t row_scale) {
    if (n >= pcfg_.coopmat_dense_min_rows && row_scale == 0 &&
        out_scale == 1.0f && (flags & ~kPfFlagRound) == 0) {
        auto used = op_gemm_coop(w, xfmt, x, xs, n, x_stride, y, flags, rows_per_group);
        if (!used) return std::unexpected(used.error());
        if (*used) return {};
    }
    auto k = runner_->kernel({"prefill_gemm", 0, w.fmt, xfmt, pcfg_.tile});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[kPgW] = w.data; s[kPgS] = w.scale; s[kPgX] = x; s[kPgXS] = xs; s[kPgY] = y;
    s[kPgRowScale] = row_scale;
    PfGemmPush p;
    p.rows = w.rows; p.k = w.k; p.scale_cols = (w.k + 31) / 32; p.n = n; p.x_stride = x_stride;
    p.y_stride = w.rows; p.rows_per_group = rows_per_group; p.flags = flags;
    p.out_scale = out_scale;
    static const char* const kFmt[] = {"fp8", "bf16", "fp32", "fp4", "q"};
    tag_ = std::format("gemm {}x{} w{} x{}", w.rows, w.k, kFmt[std::min<uint32_t>(w.fmt, 3)],
                       xfmt ? "q" : "f32");
    return flush_one(*k, &p, sizeof(p), PrefillRunner::gemm_gx(w.rows),
                     PrefillRunner::gemm_gy(n, pcfg_.tile));
}

Result<bool> Prefill::op_gemm_coop(const PfWeight& w, uint32_t xfmt, uint64_t x, uint64_t xs,
                                   uint32_t n, uint32_t x_stride, uint64_t y, uint32_t flags,
                                   uint32_t rows_per_group) {
    const uint32_t R = w.rows, K = w.k, n32 = (n + 31) / 32 * 32;
    if (R == 0 || R % 16 || K % 32 || w.fmt == kPfFp32) return false;
    const uint32_t rpg = rows_per_group ? rows_per_group : R;
    const uint32_t groups = R / rpg;
    if (rpg % 16 || groups * rpg != R) return false;
    // ungrouped: x is the [n][K] plane itself. grouped: x is [n][groups * K]
    // and group g multiplies its own K columns (prefill_gemm.slang line 160).
    if (x_stride != uint64_t(groups) * K) return false;
    if (groups > 1 && xfmt != kPfActF32) return false;
    if (uint64_t(R) * K * 2 > b_.w16.bytes ||
        (uint64_t(n32) + kPfRowSlack) * K * 2 > b_.x16.bytes ||
        (uint64_t(n32) + kPfRowSlack) * R * 4 > b_.dout.bytes)
        return false;
    const auto t0 = Clk::now();
    auto kd = runner_->kernel({"prefill_gemm", 3, w.fmt, 0, 8});
    auto kx = runner_->kernel({"prefill_coopmat", xfmt ? 1u : 2u, 0, 0, 8, 3, 0});
    const uint32_t tt = pf_tok_tiles(pcfg_.coop_tok_tiles);
    auto km = runner_->kernel({"prefill_coopmat", 0, 0, 0, 8, R, K, tt});
    auto kc = runner_->kernel({"prefill_elem", 10});
    for (auto* k : {&kd, &kx, &km, &kc})
        if (!*k) return std::unexpected(k->error());
    uint64_t* s = runner_->slots(*kd);
    s[kPgW] = w.data; s[kPgS] = w.scale; s[kPgY] = b_.w16.dev_addr;
    s = runner_->slots(*kx);
    s[kPcQ] = x; s[kPcQS] = xs; s[kPcX] = b_.x16.dev_addr;
    s = runner_->slots(*km);
    s[kPcW] = b_.w16.dev_addr; s[kPcX] = b_.x16.dev_addr; s[kPcY] = b_.dout.dev_addr;
    s = runner_->slots(*kc);
    s[0] = b_.dout.dev_addr; s[1] = y;
    if (!cmd_valid_) {
        auto cb = runner_->pool().acquire();
        if (!cb) return std::unexpected(cb.error());
        cmd_ = *cb;
        cmd_valid_ = true;
    }
    auto rec = [&](uint32_t k, const void* p, uint32_t bytes, uint32_t gx, uint32_t gy = 1) -> Result<void> {
        if (auto r = runner_->record(cmd_, k, p, bytes, gx, gy); !r) return r;
        ++times_.dispatches;
        return cmd_.barrier();
    };
    if (auto r = cmd_.begin(); !r) return std::unexpected(r.error());
    const bool decode = w16_src_ != w.data;
    if (decode) {
        PfGemmPush pd;
        pd.rows = R; pd.k = K; pd.scale_cols = (K + 31) / 32;
        if (auto r = rec(*kd, &pd, sizeof(pd), PrefillRunner::per_block_groups(R, K)); !r)
            return std::unexpected(r.error());
    }
    // pc.n = n real rows: the padded columns of x16 hold stale values whose
    // products only reach the padded rows of dout, which nobody copies
    for (uint32_t g = 0; g < groups; ++g) {
        PfCoopPush px; px.n = n; px.k = K; px.idx_off = 0; px.flags = 0;
        px.x_stride = uint32_t(uint64_t(groups) * K); px.x_col0 = g * K;
        if (auto r = rec(*kx, &px, sizeof(px), groups_for32(uint64_t(n) * (K / 32))); !r)
            return std::unexpected(r.error());
        PfCoopPush pg; pg.n = n32; pg.idx_off = 0; pg.flags = 64; pg.row0 = g * rpg;
        if (auto r = rec(*km, &pg, sizeof(pg), pf_tok_groups(n32, tt), rpg / 16); !r)
            return std::unexpected(r.error());
    }
    PfElemPush pc; pc.n = n; pc.d = R; pc.flags = flags & kPfFlagRound;
    if (auto r = rec(*kc, &pc, sizeof(pc), groups_for(uint64_t(n) * (R / 16))); !r)
        return std::unexpected(r.error());
    if (auto r = cmd_.end(); !r) return std::unexpected(r.error());
    ++times_.submits;
    if (auto r = submit_and_wait(*device_, cmd_); !r) return std::unexpected(r.error());
    w16_src_ = w.data;
    static const char* const kFmt[] = {"fp8", "bf16", "fp32", "fp4"};
    auto& slot = times_.per_op[std::format("coop {}x{} w{} x{}", R, K, kFmt[std::min<uint32_t>(w.fmt, 3)],
                                           xfmt ? "q" : "f32")];
    slot.first += ms_since(t0);
    ++slot.second;
    return true;
}

Result<void> Prefill::op_rmsnorm(uint64_t x, uint64_t y, uint32_t n, uint32_t d, uint64_t w) {
    auto k = runner_->kernel({"prefill_elem", 1});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = x; s[1] = y; s[2] = w;
    PfElemPush p; p.n = n; p.d = d; p.eps = static_cast<float>(cfg_->rms_norm_eps);
    return flush_one(*k, &p, sizeof(p), groups_for(n));
}

Result<void> Prefill::op_mhc_pre_norm(uint64_t h, uint32_t n, uint64_t coeff,
                                      uint32_t coeff_stride, uint64_t norm_w, uint64_t out,
                                      uint64_t rs) {
    auto k = runner_->kernel({"prefill_elem", 2});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = h; s[1] = out; s[2] = norm_w; s[3] = coeff; s[4] = rs;
    PfElemPush p; p.n = n; p.d = cfg_->hidden_size; p.a0 = coeff_stride; p.a1 = kHc;
    p.eps = static_cast<float>(cfg_->rms_norm_eps);
    // one wave (32 lanes) per row, not one thread: prefill_elem.slang stage 2
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(n) * 32));
}

Result<void> Prefill::op_gate_topk(uint64_t scores, uint64_t bias, uint32_t n, uint64_t ids,
                                   uint64_t wts) {
    auto k = runner_->kernel({"prefill_elem", 11});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = scores; s[1] = ids; s[2] = bias; s[3] = wts;
    PfElemPush p;
    p.n = n; p.d = cfg_->n_routed_experts; p.a0 = cfg_->num_experts_per_tok;
    p.f1 = static_cast<float>(cfg_->routed_scaling_factor);
    tag_ = "gate top-6 (gpu)";
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(n) * 32));
}

Result<void> Prefill::op_sinkhorn(uint64_t raw, uint64_t out, uint32_t n, uint64_t base,
                                  uint64_t scale) {
    auto k = runner_->kernel({"prefill_elem", 3});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = raw; s[1] = out; s[2] = base; s[3] = scale;
    PfElemPush p; p.n = n; p.a0 = cfg_->hc_sinkhorn_iters; p.f1 = static_cast<float>(cfg_->hc_eps);
    return flush_one(*k, &p, sizeof(p), groups_for(n));
}

Result<void> Prefill::op_mhc_post(uint64_t h, uint64_t a, uint64_t coeff, uint64_t out, uint32_t n) {
    auto k = runner_->kernel({"prefill_elem", 4});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = h; s[1] = out; s[2] = a; s[3] = coeff;
    PfElemPush p; p.n = n; p.d = cfg_->hidden_size; p.a1 = kHc;
    p.flags = pcfg_.round ? kPfFlagRound : 0u;
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(n) * (cfg_->hidden_size / 16)));
}

Result<void> Prefill::op_rope(uint64_t x, uint64_t y, uint32_t n, uint32_t d, uint32_t head_dim,
                              uint32_t mode, uint32_t block, bool compressed_theta, uint32_t pos0,
                              uint32_t pos_step, bool inverse) {
    auto k = runner_->kernel({"prefill_elem", 5});
    if (!k) return std::unexpected(k.error());
    if (uint64_t(pos0) + uint64_t(n ? n - 1 : 0) * pos_step >= rope_positions_)
        return fail(Err::OutOfRange, "rope position past the table");
    uint64_t* s = runner_->slots(*k);
    s[0] = x; s[1] = y; s[2] = compressed_theta ? b_.rope_cmp.dev_addr : b_.rope_win.dev_addr;
    s[3] = 0; s[4] = 0;
    PfElemPush p; p.n = n; p.d = d; p.a0 = head_dim; p.a1 = mode; p.a2 = cfg_->qk_rope_head_dim;
    p.a3 = block; p.pos0 = pos0; p.pos_step = pos_step;
    p.flags = (pcfg_.round ? kPfFlagRound : 0u) | (inverse ? kPfFlagInverse : 0u);
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(n) * (d / block)));
}

Result<void> Prefill::op_cmp_pool(uint64_t kv, uint64_t score, uint64_t out, uint32_t groups,
                                  uint32_t ratio) {
    auto k = runner_->kernel({"prefill_elem", 6});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = kv; s[1] = out; s[2] = score;
    PfElemPush p; p.n = groups; p.d = cfg_->head_dim; p.a0 = ratio;
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(groups) * cfg_->head_dim));
}

Result<void> Prefill::op_engram_gate(uint64_t h, uint64_t kv, uint64_t qw, uint64_t kw,
                                     uint64_t out, uint32_t n) {
    auto k = runner_->kernel({"prefill_elem", 7});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = h; s[1] = out; s[2] = kv; s[3] = qw; s[4] = kw;
    PfElemPush p; p.n = n; p.d = cfg_->hidden_size; p.a1 = kHc;
    p.eps = static_cast<float>(cfg_->rms_norm_eps); p.f1 = 1e-6f;
    return flush_one(*k, &p, sizeof(p), groups_for(uint64_t(n) * kHc));
}

// docs/p4_prefill_speed.md §3: the band attention as two cooperative-matrix
// tile multiplies around a gather and a softmax, one submit.
Result<void> Prefill::op_attention(uint64_t q, uint64_t kv, uint32_t n_win, uint64_t cmp,
                                   uint64_t idx, uint32_t n_idx, uint64_t sink, uint64_t o,
                                   uint32_t b) {
    if (!pcfg_.attn_coop) return op_attention_legacy(q, kv, n_win, cmp, idx, n_idx, sink, o, b);
    const uint32_t H = cfg_->num_attention_heads, D = cfg_->head_dim;
    const uint32_t E = (n_idx + 15) / 16 * 16;
    const uint32_t ht = std::max<uint32_t>(1, std::min<uint32_t>(4, pcfg_.attn_head_tiles));
    if (H % (16 * ht) || D % 16 || uint64_t(b) * E * D * 2 > b_.g16.bytes ||
        uint64_t(b) * H * E * 4 > b_.score.bytes || uint64_t(b) * H * D * 2 > b_.q16.bytes)
        return op_attention_legacy(q, kv, n_win, cmp, idx, n_idx, sink, o, b);
    auto kq = runner_->kernel({"prefill_coopmat", 2, 0, 0, 8, 3, 0});
    auto kg = runner_->kernel({"prefill_attn", 3});
    auto ks = runner_->kernel({"prefill_coopmat", 3, 0, 0, 8, E, D, ht});
    auto km = runner_->kernel({"prefill_attn", 4});
    // P.V holds `dv` 16-dim output tiles instead of `ht` head tiles: the
    // contraction is over the entries, so dim tiles are what make the gathered
    // KV reads contiguous (prefill_coopmat.slang stage 4).
    uint32_t dv = pcfg_.attn_pv_dim_tiles;
    dv = dv <= 1 ? 1u : dv >= 8 ? 8u : dv >= 4 ? 4u : 2u;
    while (dv > 1 && D % (16 * dv)) dv >>= 1;
    const uint32_t pv_ht = dv > 1 ? 1u : ht;
    auto kp = runner_->kernel({"prefill_coopmat", 4, 0, 0, 8, E, D, pv_ht, dv});
    auto kf = runner_->kernel({"prefill_attn", 5});
    for (auto* k : {&kq, &kg, &ks, &km, &kp, &kf})
        if (!*k) return std::unexpected(k->error());
    uint64_t* s = runner_->slots(*kq);
    s[kPcQ] = q; s[kPcX] = b_.q16.dev_addr;
    for (uint32_t kh : {*kg, *km, *kf}) {
        s = runner_->slots(kh);
        s[0] = q; s[1] = kv; s[2] = cmp ? cmp : kv; s[3] = idx; s[4] = sink;
        s[5] = b_.score.dev_addr; s[6] = b_.g16.dev_addr; s[7] = b_.inv.dev_addr;
    }
    runner_->slots(*km)[6] = b_.p16.dev_addr;
    runner_->slots(*kf)[6] = o;
    s = runner_->slots(*ks);
    s[kPcW] = b_.q16.dev_addr; s[kPcX] = b_.g16.dev_addr; s[kPcY] = b_.score.dev_addr;
    s = runner_->slots(*kp);
    s[kPcW] = b_.p16.dev_addr; s[kPcX] = b_.g16.dev_addr; s[kPcY] = o;

    PfAttnPush p;
    p.b = b; p.n_idx = E; p.g = n_idx; p.n_heads = H; p.head_dim = D; p.n_win = n_win;
    p.scale = 1.0f / std::sqrt(float(D));
    p.flags = pcfg_.round ? kPfFlagRound : 0u;
    PfCoopPush pq; pq.n = b; pq.k = H * D;
    PfCoopPush pc; pc.n = b; pc.k = H;
    const uint32_t gy = H / (16 * ht);
    const auto t0 = Clk::now();
    if (auto r = cmd_open(); !r) return r;
    if (auto r = rec(*kq, &pq, sizeof(pq), groups_for32(uint64_t(b) * (H * D / 32))); !r) return r;
    mark("attn q16");
    if (auto r = rec(*kg, &p, sizeof(p), groups_for(uint64_t(b) * E)); !r) return r;
    mark("attn gather");
    if (auto r = rec(*ks, &pc, sizeof(pc), b, gy); !r) return r;
    mark("attn score tiles");
    if (auto r = rec(*km, &p, sizeof(p), groups_for(uint64_t(b) * H)); !r) return r;
    mark("attn softmax");
    if (auto r = rec(*kp, &pc, sizeof(pc), b, H / (16 * pv_ht)); !r) return r;
    mark("attn P.V tiles");
    if (auto r = rec(*kf, &p, sizeof(p), groups_for(uint64_t(b) * H * (D / 16))); !r) return r;
    mark("attn finish");
    if (auto r = cmd_close(); !r) return r;
    auto& slot = times_.per_op["attn (coop, submit+wait)"];
    slot.first += ms_since(t0);
    ++slot.second;
    return {};
}

Result<void> Prefill::op_attention_legacy(uint64_t q, uint64_t kv, uint32_t n_win, uint64_t cmp,
                                          uint64_t idx, uint32_t n_idx, uint64_t sink, uint64_t o,
                                          uint32_t b) {
    auto ks = runner_->kernel({"prefill_attn", 0});
    auto kc = runner_->kernel({"prefill_attn", 1});
    if (!ks) return std::unexpected(ks.error());
    if (!kc) return std::unexpected(kc.error());
    for (uint32_t kh : {*ks, *kc}) {
        uint64_t* s = runner_->slots(kh);
        s[0] = q; s[1] = kv; s[2] = cmp ? cmp : kv; s[3] = idx; s[4] = sink;
        s[5] = b_.score.dev_addr; s[6] = o;
    }
    PfAttnPush p;
    p.b = b; p.n_idx = n_idx; p.n_heads = cfg_->num_attention_heads; p.head_dim = cfg_->head_dim;
    p.n_win = n_win; p.scale = 1.0f / std::sqrt(float(cfg_->head_dim));
    p.flags = pcfg_.round ? kPfFlagRound : 0u;
    if (auto r = flush_one(*ks, &p, sizeof(p), p.n_heads, b); !r) return r;
    return flush_one(*kc, &p, sizeof(p), p.n_heads, b);
}

Result<void> Prefill::op_index_score(uint64_t q, uint64_t keys, uint32_t g, uint64_t w,
                                     uint64_t score, uint32_t b, uint32_t ratio, uint32_t pos0) {
    auto k = runner_->kernel({"prefill_attn", 2});
    if (!k) return std::unexpected(k.error());
    uint64_t* s = runner_->slots(*k);
    s[0] = q; s[1] = keys; s[5] = score; s[7] = w;
    PfAttnPush p;
    p.b = b; p.n_heads = cfg_->index_n_heads; p.head_dim = cfg_->index_head_dim; p.g = g;
    p.ratio = ratio; p.pos0 = pos0;
    return flush_one(*k, &p, sizeof(p), (g + 7) / 8, b);
}

#endif  // DEEPMOE_ENABLE_VULKAN

std::vector<std::vector<uint8_t>> Prefill::candidate_blocks(uint32_t b, uint32_t pos0, uint32_t ratio,
                                                            uint32_t g, const float* scores,
                                                            uint32_t topk_blocks, uint32_t block) {
    std::vector<std::vector<uint8_t>> keep(b);
    if (block == 0 || topk_blocks == 0) return keep;
    const uint32_t nblocks = (g + block - 1) / block;
    std::vector<float> bs;
    std::vector<uint32_t> order;
    for (uint32_t j = 0; j < b; ++j) {
        const uint32_t vis = std::min(g, (pos0 + j + 1) / std::max<uint32_t>(ratio, 1));
        const uint32_t reach = (vis + block - 1) / block;
        if (reach <= topk_blocks) continue;   // every reachable block kept: the identity
        // a block's score is its best visible position; the newest block is pinned in
        const float* sr = scores + size_t(j) * g;
        bs.assign(nblocks, -std::numeric_limits<float>::infinity());
        for (uint32_t i = 0; i < vis; ++i) bs[i / block] = std::max(bs[i / block], sr[i]);
        bs[(vis - 1) / block] = std::numeric_limits<float>::infinity();
        order.resize(nblocks);
        std::iota(order.begin(), order.end(), 0u);
        std::nth_element(order.begin(), order.begin() + (topk_blocks - 1), order.end(),
                         [&](uint32_t a, uint32_t c) { return bs[a] != bs[c] ? bs[a] > bs[c] : a < c; });
        std::vector<uint8_t>& row = keep[j];
        row.assign(nblocks, 0);
        for (uint32_t i = 0; i < topk_blocks; ++i)
            if (bs[order[i]] > -std::numeric_limits<float>::infinity()) row[order[i]] = 1;
    }
    return keep;
}

void Prefill::topk_rows(uint32_t b, uint32_t pos0, uint32_t kv_pos0, uint32_t n_kv_rows,
                        uint32_t window, uint32_t ratio, uint32_t g, uint32_t index_topk,
                        const float* scores, int32_t* out, uint32_t n_idx) {
    (void)index_topk;
    const uint32_t nw = std::min(n_kv_rows, window);
    const uint32_t k = n_idx - nw;
    std::vector<uint32_t> order;
    for (uint32_t j = 0; j < b; ++j) {
        const uint32_t p = pos0 + j;
        int32_t* row = out + size_t(j) * n_idx;
        // get_window_topk_idxs at start_pos 0, in this chunk's row space
        const int64_t first = std::max<int64_t>(kv_pos0, int64_t(p) - int64_t(window) + 1);
        for (uint32_t i = 0; i < nw; ++i) {
            const int64_t pos = first + i;
            row[i] = pos > int64_t(p) ? -1 : int32_t(pos - kv_pos0);
        }
        if (k == 0) continue;
        const uint32_t vis = std::min(g, (p + 1) / std::max<uint32_t>(ratio, 1));
        const uint32_t take = std::min(k, vis);
        order.resize(vis);
        std::iota(order.begin(), order.end(), 0u);
        if (scores && vis > k) {
            const float* sr = scores + size_t(j) * g;
            std::partial_sort(order.begin(), order.begin() + take, order.end(),
                              [&](uint32_t a, uint32_t c) {
                                  return sr[a] != sr[c] ? sr[a] > sr[c] : a < c;
                              });
            std::sort(order.begin(), order.begin() + take);
        }
        for (uint32_t i = 0; i < k; ++i)
            row[nw + i] = i < take ? int32_t(order[i] + n_kv_rows) : -1;
    }
}

}  // namespace deepmoe::gpu

namespace deepmoe::gpu {

#if defined(DEEPMOE_ENABLE_VULKAN)

Result<void> Prefill::engram_rows(uint32_t L, std::span<const uint32_t> prompt, uint64_t out) {
    (void)out;
    const uint32_t n = static_cast<uint32_t>(prompt.size());
    const uint32_t cols = layout::kEngramRowsPerToken;
    std::vector<uint64_t> rows(size_t(n) * cols);
    for (uint32_t p = 0; p < n; ++p)
        if (auto r = engram_->hash_rows(L, prompt, p,
                                        std::span<uint64_t>(rows.data() + size_t(p) * cols, cols));
            !r)
            return r;
    std::vector<uint64_t> uniq(rows);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const uint64_t per = 2 * kPageSize;
    auto stage = alloc_host_pages(uniq.size() * 2 * per, false);
    if (!stage) return std::unexpected(stage.error());
    struct Guard { HostAllocInfo i; ~Guard() { free_host_pages(i); } } guard{*stage};
    auto* base = static_cast<std::byte*>(stage->ptr);
    std::vector<std::future<storage::IoResult>> futs;
    std::vector<uint32_t> skew(uniq.size() * 2);
    futs.reserve(uniq.size() * 2);
    const auto t0 = Clk::now();
    for (size_t u = 0; u < uniq.size(); ++u) {
        auto plan = manifest_->engram_row(L, uniq[u]);
        if (!plan) return std::unexpected(plan.error());
        const AlignedRead rd[2] = {plan->value, plan->scale};
        for (uint32_t j = 0; j < 2; ++j) {
            if (rd[j].aligned_bytes > per) return fail(Err::Internal, "engram read too large");
            auto f = shards_->require(rd[j].file);
            if (!f) return std::unexpected(f.error());
            storage::IoRequest req;
            req.priority = IoPriority::Engram;
            req.file = *f;
            req.file_off = rd[j].aligned_off;
            req.bytes = rd[j].aligned_bytes;
            req.dst = base + (u * 2 + j) * per;
            auto fut = io_->submit_future(req);
            if (!fut) return std::unexpected(fut.error());
            futs.push_back(std::move(*fut));
            skew[u * 2 + j] = rd[j].skew;
        }
    }
    for (auto& f : futs)
        if (!f.get().ok()) return fail(Err::Io, "engram row read failed");
    times_.engram_io += ms_since(t0);
    times_.engram_reads += futs.size();
    // ParallelEngramEmbedding: value.float() * scale per 32, then .to(bf16)
    std::vector<float> x(size_t(n) * cols * 256);
    for (uint32_t p = 0; p < n; ++p)
        for (uint32_t c = 0; c < cols; ++c) {
            const size_t u = static_cast<size_t>(
                std::lower_bound(uniq.begin(), uniq.end(), rows[size_t(p) * cols + c]) - uniq.begin());
            const auto* v = reinterpret_cast<const uint8_t*>(base + u * 2 * per + skew[u * 2]);
            const auto* s = reinterpret_cast<const uint8_t*>(base + (u * 2 + 1) * per + skew[u * 2 + 1]);
            float* dst = x.data() + (size_t(p) * cols + c) * 256;
            for (uint32_t i = 0; i < 256; ++i)
                dst[i] = cpu::bf16_to_float(cpu::float_to_bf16(
                    cpu::fp8_e4m3_to_float(v[i]) * cpu::e8m0_to_float(s[i / 32])));
        }
    std::memcpy(b_.eng_x.host_ptr, x.data(), x.size() * sizeof(float));
    return {};
}

Result<void> Prefill::op_engram_rows(uint32_t L, std::span<const uint32_t> prompt) {
    if (!engram_) return fail(Err::FailedPrecondition, "no EngramTables");
    return engram_rows(L, prompt, b_.eng_x.dev_addr);
}

Result<void> Prefill::op_moe(uint32_t L, uint32_t rows, uint64_t x, std::vector<uint32_t>& ids,
                             std::vector<float>& wts, uint64_t y, bool shared, bool routed) {
    if (auto r = op_act_quant(x, rows, cfg_->hidden_size, b_.fxq.dev_addr, b_.fxs.dev_addr); !r)
        return r;
    return run_moe(L, rows, x, b_.fxq.dev_addr, b_.fxs.dev_addr, y, ids, wts, shared, routed);
}

Result<void> Prefill::run_moe(uint32_t L, uint32_t rows, uint64_t x, uint64_t xq, uint64_t xs,
                              uint64_t y, std::vector<uint32_t>& ids, std::vector<float>& wts,
                              bool shared, bool routed) {
    (void)x;
    const TextConfig& c = *cfg_;
    const uint32_t k6 = c.num_experts_per_tok, dim = c.hidden_size, inter = c.moe_intermediate_size;
    const uint32_t E = c.n_routed_experts;
    const auto t_host = Clk::now();
    // --- CSR by expert: token ids ascending within an expert, as torch.where --
    std::vector<uint32_t> count(E, 0);
    for (uint32_t t = 0; t < rows; ++t)
        for (uint32_t s = 0; s < k6; ++s) ++count[ids[size_t(t) * k6 + s]];
    std::vector<uint32_t> start(E + 1, 0);
    for (uint32_t e = 0; e < E; ++e) start[e + 1] = start[e] + count[e];
    std::vector<uint32_t> csr(size_t(rows) * (k6 + 1));
    std::vector<float> rw(size_t(rows) * (k6 + 1));
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for (uint32_t t = 0; t < rows; ++t)
        for (uint32_t s = 0; s < k6; ++s) {
            const uint32_t e = ids[size_t(t) * k6 + s];
            csr[fill[e]] = t;
            rw[fill[e]] = wts[size_t(t) * k6 + s];
            ++fill[e];
        }
    host_op("host: MoE CSR", t_host);
    const uint32_t shared_off = rows * k6;
    for (uint32_t t = 0; t < rows; ++t) { csr[shared_off + t] = t; rw[shared_off + t] = 1.0f; }
    std::memcpy(b_.csr_idx.host_ptr, csr.data(), csr.size() * sizeof(uint32_t));
    std::memcpy(b_.csr_rw.host_ptr, rw.data(), rw.size() * sizeof(float));
    times_.gate += ms_since(t_host);

    auto kg = runner_->kernel({"prefill_gemm", 1, 0, kPfActQ, pcfg_.tile});
    auto kd = runner_->kernel({"prefill_gemm", 2, 0, kPfActQ, pcfg_.tile});
    auto kq = runner_->kernel({"prefill_elem", 0});
    if (!kg) return std::unexpected(kg.error());
    if (!kd) return std::unexpected(kd.error());
    if (!kq) return std::unexpected(kq.error());
    uint64_t* sg = runner_->slots(*kg);
    sg[kPgX] = xq; sg[kPgXS] = xs; sg[kPgY] = b_.hplane.dev_addr; sg[kPgIdx] = b_.csr_idx.dev_addr;
    sg[kPgRW] = b_.csr_rw.dev_addr; sg[kPgJob] = b_.jobs.dev_addr;
    uint64_t* sd = runner_->slots(*kd);
    sd[kPgX] = b_.hq.dev_addr; sd[kPgXS] = b_.hs.dev_addr; sd[kPgY] = y;
    sd[kPgIdx] = b_.csr_idx.dev_addr; sd[kPgJob] = b_.jobs.dev_addr;
    uint64_t* sq = runner_->slots(*kq);
    sq[0] = b_.hplane.dev_addr; sq[1] = b_.hq.dev_addr; sq[2] = b_.hs.dev_addr;

    const uint32_t round = pcfg_.round ? kPfFlagRound : 0u;
    auto* jobs = static_cast<PfJob*>(b_.jobs.host_ptr);
    // Dispatch A for jobs [j0, j1), the h quantisation over `hrows`, dispatch B
    // for the same jobs: one command buffer, one submit.
    auto compute = [&](uint32_t j0, uint32_t j1, uint32_t hrows) -> Result<void> {
        if (!cmd_valid_) {
            auto cb = runner_->pool().acquire();
            if (!cb) return std::unexpected(cb.error());
            cmd_ = *cb;
            cmd_valid_ = true;
        }
        if (auto r = cmd_.begin(); !r) return r;
        for (uint32_t j = j0; j < j1; ++j) {
            PfGemmPush p;
            p.rows = inter; p.k = dim; p.scale_cols = dim / 32; p.n = jobs[j].n; p.x_stride = dim;
            p.y_stride = inter; p.job = j; p.flags = round;
            p.swiglu_limit = static_cast<float>(c.swiglu_limit);
            if (auto r = runner_->record(cmd_, *kg, &p, sizeof(p), PrefillRunner::gemm_gx(inter),
                                         PrefillRunner::gemm_gy(jobs[j].n, pcfg_.tile)); !r)
                return r;
            ++times_.dispatches;
        }
        if (auto r = cmd_.barrier(); !r) return r;
        PfElemPush pq; pq.n = hrows; pq.d = inter;
        if (auto r = runner_->record(cmd_, *kq, &pq, sizeof(pq),
                                     groups_for(uint64_t(hrows) * (inter / 32))); !r)
            return r;
        if (auto r = cmd_.barrier(); !r) return r;
        for (uint32_t j = j0; j < j1; ++j) {
            PfGemmPush p;
            p.rows = dim; p.k = inter; p.scale_cols = inter / 32; p.n = jobs[j].n;
            p.y_stride = dim; p.job = j; p.flags = round;
            if (auto r = runner_->record(cmd_, *kd, &p, sizeof(p), PrefillRunner::gemm_gx(dim),
                                         PrefillRunner::gemm_gy(jobs[j].n, pcfg_.tile)); !r)
                return r;
            // y rows are shared between experts: each B must see the last's adds
            if (auto r = cmd_.barrier(); !r) return r;
            ++times_.dispatches;
        }
        times_.dispatches += 1;
        if (auto r = cmd_.end(); !r) return r;
        ++times_.submits;
        return submit_and_wait(*device_, cmd_);
    };

    // docs/p3_prefill.md §5 option (a), the measured winner: every matrix of
    // every expert decoded into one fp16 transit buffer and multiplied as
    // cooperative-matrix tiles, 32 rows a wave. Per expert: decode w1, gather x,
    // GEMM -> gate; decode w3, GEMM -> up; SwiGLU x route weight; act_quant(h);
    // stage h; decode w2, GEMM -> down; scatter-add into y. One submit a batch.
    auto kdec = runner_->kernel({"prefill_gemm", 3, 0, 0, 8});
    auto kx1  = runner_->kernel({"prefill_coopmat", 1, 0, 0, 8, 1, 0});
    auto kx2  = runner_->kernel({"prefill_coopmat", 1, 0, 0, 8, 2, 0});
    const uint32_t tt = pf_tok_tiles(pcfg_.coop_tok_tiles);
    auto kgu  = runner_->kernel({"prefill_coopmat", 0, 0, 0, 8, inter, dim, tt});
    auto kdn  = runner_->kernel({"prefill_coopmat", 0, 0, 0, 8, dim, inter, tt});
    auto ksw  = runner_->kernel({"prefill_elem", 8});
    auto ksc  = runner_->kernel({"prefill_elem", 9});
    const uint32_t coop_min = pcfg_.coopmat_min_rows;
    w16_src_ = 0;   // the expert decodes below overwrite the dense transit
    {
        for (auto* k : {&kdec, &kx1, &kx2, &kgu, &kdn, &ksw, &ksc})
            if (!*k) return std::unexpected(k->error());
        uint64_t* s = runner_->slots(*kdec);
        s[kPgJob] = b_.jobs.dev_addr; s[kPgY] = b_.w16.dev_addr;
        s = runner_->slots(*kx1);
        s[kPcQ] = xq; s[kPcQS] = xs; s[kPcX] = b_.x16.dev_addr; s[kPcIdx] = b_.csr_idx.dev_addr;
        s = runner_->slots(*kx2);
        s[kPcQ] = b_.hq.dev_addr; s[kPcQS] = b_.hs.dev_addr; s[kPcX] = b_.h16.dev_addr;
        s = runner_->slots(*kgu);
        s[kPcW] = b_.w16.dev_addr; s[kPcX] = b_.x16.dev_addr; s[kPcY] = b_.gu.dev_addr;
        s = runner_->slots(*kdn);
        s[kPcW] = b_.w16.dev_addr; s[kPcX] = b_.h16.dev_addr; s[kPcY] = b_.dout.dev_addr;
        s = runner_->slots(*ksw);
        s[0] = b_.gu.dev_addr; s[1] = b_.hplane.dev_addr; s[2] = b_.csr_rw.dev_addr;
        s = runner_->slots(*ksc);
        s[0] = b_.dout.dev_addr; s[1] = y; s[2] = b_.csr_idx.dev_addr;
    }
    auto compute_coop = [&](uint32_t j0, uint32_t j1) -> Result<void> {
        if (!cmd_valid_) {
            auto cb = runner_->pool().acquire();
            if (!cb) return std::unexpected(cb.error());
            cmd_ = *cb;
            cmd_valid_ = true;
        }
        if (auto r = cmd_.begin(); !r) return r;
        auto rec = [&](uint32_t k, const void* p, uint32_t bytes, uint32_t gx, uint32_t gy = 1) -> Result<void> {
            if (auto r = runner_->record(cmd_, k, p, bytes, gx, gy); !r) return r;
            ++times_.dispatches;
            return cmd_.barrier();
        };
        for (uint32_t j = j0; j < j1; ++j) {
            const PfJob& jb = jobs[j];
            const uint32_t n = jb.n, n32 = (jb.n + 31) / 32 * 32;
            PfGemmPush pd;
            pd.job = j; pd.flags = kPfFlagFromJob;
            pd.rows = inter; pd.k = dim; pd.scale_cols = dim / 32; pd.idx_off = 0;
            if (auto r = rec(*kdec, &pd, sizeof(pd), PrefillRunner::per_block_groups(inter, dim)); !r) return r;
            PfCoopPush px; px.n = n32; px.k = dim; px.idx_off = jb.rows_off; px.flags = kPfFlagGather;
            if (auto r = rec(*kx1, &px, sizeof(px), groups_for32(uint64_t(n32) * (dim / 32))); !r) return r;
            PfCoopPush pg; pg.n = n32; pg.idx_off = 0; pg.flags = 64;
            if (auto r = rec(*kgu, &pg, sizeof(pg), pf_tok_groups(n32, tt), inter / 16); !r) return r;
            pd.idx_off = 1;
            if (auto r = rec(*kdec, &pd, sizeof(pd), PrefillRunner::per_block_groups(inter, dim)); !r) return r;
            pg.idx_off = n32;
            if (auto r = rec(*kgu, &pg, sizeof(pg), pf_tok_groups(n32, tt), inter / 16); !r) return r;
            PfElemPush ps; ps.n = n; ps.d = inter; ps.a0 = n32; ps.a1 = jb.h_off; ps.a2 = jb.rows_off;
            ps.f1 = static_cast<float>(c.swiglu_limit); ps.flags = round;
            if (auto r = rec(*ksw, &ps, sizeof(ps), groups_for(uint64_t(n) * (inter / 16))); !r) return r;
            PfElemPush pq; pq.n = n; pq.d = inter; pq.row_off = jb.h_off;
            if (auto r = rec(*kq, &pq, sizeof(pq), groups_for(uint64_t(n) * (inter / 32))); !r) return r;
            PfCoopPush ph; ph.n = n32; ph.k = inter; ph.idx_off = jb.h_off; ph.flags = 0;
            if (auto r = rec(*kx2, &ph, sizeof(ph), groups_for32(uint64_t(n32) * (inter / 32))); !r) return r;
            pd.idx_off = 2; pd.rows = dim; pd.k = inter; pd.scale_cols = inter / 32;
            if (auto r = rec(*kdec, &pd, sizeof(pd), PrefillRunner::per_block_groups(dim, inter)); !r) return r;
            PfCoopPush pn; pn.n = n32; pn.idx_off = 0; pn.flags = 64;
            if (auto r = rec(*kdn, &pn, sizeof(pn), pf_tok_groups(n32, tt), dim / 16); !r) return r;
            PfElemPush pc2; pc2.n = n; pc2.d = dim; pc2.a2 = jb.rows_off; pc2.flags = round;
            if (auto r = rec(*ksc, &pc2, sizeof(pc2), groups_for(uint64_t(n) * (dim / 16))); !r) return r;
        }
        if (auto r = cmd_.end(); !r) return r;
        ++times_.submits;
        return submit_and_wait(*device_, cmd_);
    };

    // --- the shared expert (job 0): fp8, every row ------------------------------
    if (shared) {
        const auto t0 = Clk::now();
        const std::string pfx = std::format("layers.{}.ffn.shared_experts.", L);
        auto w1 = weight(pfx + "w1.weight", kPfFp8);
        auto w2 = weight(pfx + "w2.weight", kPfFp8);
        auto w3 = weight(pfx + "w3.weight", kPfFp8);
        if (!w1 || !w2 || !w3) return fail(Err::NotFound, "shared expert weights");
        PfJob j;
        j.w1 = w1->data; j.s1 = w1->scale; j.w2 = w2->data; j.s2 = w2->scale;
        j.w3 = w3->data; j.s3 = w3->scale; j.rows_off = shared_off; j.n = rows; j.h_off = 0;
        j.fmt = kPfFp8;
        jobs[kSharedJob] = j;
        if (auto r = rows >= coop_min ? compute_coop(kSharedJob, kSharedJob + 1)
                                      : compute(kSharedJob, kSharedJob + 1, rows); !r)
            return r;
        times_.shared_expert += ms_since(t0);
    }

    // --- routed experts, expert-major, in shard order ---------------------------
    if (!routed) return {};
    std::vector<uint32_t> used;
    for (uint32_t e = 0; e < E; ++e) if (count[e]) used.push_back(e);
    std::vector<const ExpertEntry*> ents(E, nullptr);
    for (uint32_t e : used) {
        auto ent = manifest_->require_expert({static_cast<uint16_t>(L), static_cast<uint16_t>(e)});
        if (!ent) return std::unexpected(ent.error());
        ents[e] = *ent;
    }
    std::sort(used.begin(), used.end(), [&](uint32_t a, uint32_t b) {
        const Run& ra = ents[a]->runs.front();
        const Run& rb = ents[b]->runs.front();
        return ra.file != rb.file ? ra.file < rb.file : ra.aligned_off < rb.aligned_off;
    });
    const uint32_t K = pcfg_.transit_slots;
    // Track R1: the last row (and its top-6 rank) routed to each expert -- its
    // LRU age in the decode cache it may be handed to.
    std::vector<uint32_t> last_t(E, 0), last_s(E, 0);
    std::vector<PfExpertSink::Dest> dest(expert_sink ? E : 0);
    if (expert_sink)
        for (uint32_t t = 0; t < rows; ++t)
            for (uint32_t s = 0; s < k6; ++s) {
                last_t[ids[size_t(t) * k6 + s]] = t;
                last_s[ids[size_t(t) * k6 + s]] = s;
            }
    struct Batch {
        uint32_t first = 0, count = 0, half = 0;
        std::vector<std::future<storage::IoResult>> futs;
    };
    auto issue = [&](Batch& bt) -> Result<void> {
        for (uint32_t i = 0; i < bt.count; ++i) {
            const uint32_t e = used[bt.first + i];
            const uint64_t slot = uint64_t(bt.half) * K + i;
            if (expert_sink && expert_sink->reserve) {
                dest[e] = expert_sink->reserve(L, e, moe_pos0_ + last_t[e], last_s[e]);
                if (dest[e].kind == PfExpertSink::Kind::Resident) continue;   // no read
            }
            const bool into_cache = expert_sink && dest[e].kind == PfExpertSink::Kind::Fill;
            for (const Run& r : ents[e]->runs) {
                auto f = shards_->require(r.file);
                if (!f) return std::unexpected(f.error());
                storage::IoRequest req;
                req.key = {static_cast<uint16_t>(L), static_cast<uint16_t>(e)};
                req.priority = IoPriority::BlockingMiss;
                req.file = *f;
                req.file_off = r.aligned_off;
                req.bytes = r.aligned_bytes;
                req.dst = into_cache
                    ? static_cast<std::byte*>(dest[e].host) + r.slot_offset
                    : static_cast<std::byte*>(b_.transit.host_ptr) +
                          slot * layout::kExpertSlotBytes + r.slot_offset;
                auto fut = io_->submit_future(req);
                if (!fut) return std::unexpected(fut.error());
                bt.futs.push_back(std::move(*fut));
                times_.expert_bytes += r.aligned_bytes;
            }
        }
        times_.experts_read += bt.count;
        if (expert_sink)
            for (uint32_t i = 0; i < bt.count; ++i)
                if (dest[used[bt.first + i]].kind == PfExpertSink::Kind::Resident) --times_.experts_read;
        return {};
    };
    auto run_batch = [&](Batch& bt) -> Result<void> {
        const auto tw = Clk::now();
        bool read_ok = true;
        for (auto& f : bt.futs) read_ok &= f.get().ok();
        if (!read_ok) {
            if (expert_sink && expert_sink->release)
                for (uint32_t i = 0; i < bt.count; ++i) {
                    const uint32_t e = used[bt.first + i];
                    if (dest[e].kind != PfExpertSink::Kind::Drop) expert_sink->release(L, e, dest[e], false);
                }
            return fail(Err::Io, "expert read failed");
        }
        times_.expert_io += ms_since(tw);
        const auto tg = Clk::now();
        // Small experts first (tiled job table, one quantisation over their h
        // rows), then the large ones (cooperative matrix, one expert at a time).
        std::vector<uint32_t> order(bt.count);
        for (uint32_t i = 0; i < bt.count; ++i) order[i] = i;
        std::stable_partition(order.begin(), order.end(),
                              [&](uint32_t i) { return count[used[bt.first + i]] < coop_min; });
        uint32_t h_off = 0, n_small = 0, h_small = 0;
        for (uint32_t o = 0; o < bt.count; ++o) {
            const uint32_t i = order[o];
            const uint32_t e = used[bt.first + i];
            const uint64_t base = (expert_sink && dest[e].kind != PfExpertSink::Kind::Drop)
                ? dest[e].dev
                : b_.transit.dev_addr + (uint64_t(bt.half) * K + i) * layout::kExpertSlotBytes;
            PfJob j;
            j.w1 = base + ents[e]->offset_of(ExpertPart::W1Weight);
            j.s1 = base + ents[e]->offset_of(ExpertPart::W1Scale);
            j.w2 = base + ents[e]->offset_of(ExpertPart::W2Weight);
            j.s2 = base + ents[e]->offset_of(ExpertPart::W2Scale);
            j.w3 = base + ents[e]->offset_of(ExpertPart::W3Weight);
            j.s3 = base + ents[e]->offset_of(ExpertPart::W3Scale);
            j.rows_off = start[e]; j.n = count[e]; j.h_off = h_off; j.fmt = kPfFp4;
            jobs[1 + o] = j;
            h_off += count[e];
            if (count[e] < coop_min) { ++n_small; h_small = h_off; }
        }
        if (n_small > 0)
            if (auto r = compute(1, 1 + n_small, h_small); !r) return r;
        if (n_small < bt.count)
            if (auto r = compute_coop(1 + n_small, 1 + bt.count); !r) return r;
        times_.expert_gpu += ms_since(tg);
        if (expert_sink && expert_sink->release)
            for (uint32_t i = 0; i < bt.count; ++i) {
                const uint32_t e = used[bt.first + i];
                if (dest[e].kind != PfExpertSink::Kind::Drop) expert_sink->release(L, e, dest[e], true);
            }
        return {};
    };
    std::vector<Batch> batches;
    for (uint32_t i = 0; i < used.size(); i += K) {
        Batch bt;
        bt.first = i;
        bt.count = std::min<uint32_t>(K, static_cast<uint32_t>(used.size()) - i);
        bt.half = static_cast<uint32_t>(batches.size() % 2);
        batches.push_back(std::move(bt));
    }
    // Batch i+1's reads go out before batch i computes: the drive fills one half
    // of the transit while the GPU works on the other (docs/p3_prefill.md §3.3).
    for (size_t i = 0; i < batches.size(); ++i) {
        if (auto r = issue(batches[i]); !r) return r;
        if (i > 0)
            if (auto r = run_batch(batches[i - 1]); !r) return r;
    }
    if (!batches.empty())
        if (auto r = run_batch(batches.back()); !r) return r;
    return {};
}

Result<PrefillHandoff> Prefill::run(std::span<const uint32_t> prompt) {
    const TextConfig& c = *cfg_;
    const uint32_t N = static_cast<uint32_t>(prompt.size());
    if (N == 0 || N > pcfg_.max_tokens)
        return fail(Err::InvalidArgument,
                    std::format("{} prompt tokens; the buffers hold 1..{}", N, pcfg_.max_tokens));
    times_ = PrefillTimes{};
    const auto t_all = Clk::now();
    PrefillHandoff out;
    out.prompt.assign(prompt.begin(), prompt.end());
    out.layers.assign(c.num_hidden_layers, {});

    // --- embedding: one row per token, widened once, four hc copies -----------
    {
        const auto t0 = Clk::now();
        auto emb = pinned_->require("embed.weight");
        if (!emb) return std::unexpected(emb.error());
        const uint32_t dim = c.hidden_size;
        std::vector<float> h(size_t(N) * kHc * dim);
        std::vector<uint16_t> row(dim);
        for (uint32_t p = 0; p < N; ++p) {
            if (prompt[p] >= c.vocab_size) return fail(Err::OutOfRange, "token id past the vocabulary");
            std::memcpy(row.data(), static_cast<const std::byte*>((*emb)->data_host) +
                                        uint64_t(prompt[p]) * dim * 2, size_t(dim) * 2);
            float* dst = h.data() + size_t(p) * kHc * dim;
            for (uint32_t e = 0; e < dim; ++e) dst[e] = cpu::bf16_to_float(row[e]);
            for (uint32_t j = 1; j < kHc; ++j) std::memcpy(dst + j * dim, dst, dim * sizeof(float));
        }
        std::memcpy(b_.h_a.host_ptr, h.data(), h.size() * sizeof(float));
        std::vector<float> mix(size_t(N) * kMix, 0.0f);
        for (uint32_t p = 0; p < N; ++p) mix[size_t(p) * kMix] = 1.0f;   // make_identity_pre_mix
        std::memcpy(b_.mix_prev.host_ptr, mix.data(), mix.size() * sizeof(float));
        times_.embed = ms_since(t0);
    }
    for (SourceState& s : sources_) { s.n = 0; s.valid = false; }
    topk_shared_.clear();
    cand_.clear();
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L)
        if (auto r = run_layer(L, prompt, out); !r)
            return fail(r.error().code, std::format("layer {}: {}", L, r.error().message));

    // --- collapse + norm + head on the last position ---------------------------
    {
        const auto t0 = Clk::now();
        const uint32_t R = std::min(N, pcfg_.replay);
        const uint32_t rows = (R < N) ? R : N;
        const uint64_t hstride = uint64_t(kHc) * c.hidden_size * 4;
        auto nw = pinned_->require("norm.weight");
        if (!nw) return std::unexpected(nw.error());
        if (auto r = op_mhc_pre_norm(at(b_.h_a, rows - 1, hstride), 1,
                                     at(b_.mix_prev, rows - 1, kMix * 4), kMix, (*nw)->data,
                                     b_.nrm.dev_addr, b_.rs.dev_addr); !r)
            return std::unexpected(r.error());
        auto head = weight("head.weight", kPfBf16);
        if (!head) return std::unexpected(head.error());
        if (auto r = op_gemm(*head, kPfActF32, b_.nrm.dev_addr, 0, 1, c.hidden_size,
                             b_.logits.dev_addr, 0); !r)
            return std::unexpected(r.error());
        out.logits.resize(c.vocab_size);
        std::memcpy(out.logits.data(), b_.logits.host_ptr, out.logits.size() * sizeof(float));
        uint32_t best = 0;
        float t1 = -3e38f, t2 = -3e38f;
        for (uint32_t v = 0; v < c.vocab_size; ++v) {
            const float f = out.logits[v];
            if (f > t1) { t2 = t1; t1 = f; best = v; } else if (f > t2) { t2 = f; }
        }
        out.first_token = best; out.top1 = t1; out.top2 = t2;
        times_.head = ms_since(t0);
    }
    times_.total = ms_since(t_all);
    times_.host = times_.total - (times_.embed + times_.engram_io + times_.engram + times_.mhc +
                                  times_.attention + times_.gate + times_.shared_expert +
                                  times_.expert_io + times_.expert_gpu + times_.head);
    return out;
}

#define PF_TRY(expr) do { if (auto _r = (expr); !_r) return std::unexpected(_r.error()); } while (0)

Result<void> Prefill::run_layer(uint32_t L, std::span<const uint32_t> prompt, PrefillHandoff& out) {
    const TextConfig& c = *cfg_;
    const uint32_t N = static_cast<uint32_t>(prompt.size());
    const uint32_t R = std::min(N, pcfg_.replay);
    const bool replay = R < N;
    const uint32_t dim = c.hidden_size, hd = c.head_dim, win = c.sliding_window;
    const uint64_t hstride = uint64_t(kHc) * dim * 4;
    // rows entering the layer, and rows its queries / FFN run over (§1.1)
    const uint32_t front = (replay && L > 20) ? R : N;
    const uint32_t A = (replay && L >= 20) ? R : front;
    const uint32_t aoff = front - A;
    const uint32_t fpos0 = (replay && L > 20) ? N - R : 0;
    const uint32_t apos0 = fpos0 + aoff;
    const uint32_t ratio = c.compress_ratio(L);
    const bool cmp_theta = ratio > 0;
    const uint32_t round = pcfg_.round ? kPfFlagRound : 0u;
    const std::string P = std::format("layers.{}.", L);
    auto pw = [&](const std::string& n) -> uint64_t {
        const store::PinnedTensor* t = pinned_->find(P + n);
        return t ? t->data : 0;
    };
    auto W = [&](const std::string& n, uint32_t fmt) { return weight(P + n, fmt); };

    PrefillProbe pv;
    if (pcfg_.probe_layers)
        std::memcpy(b_.h_in_copy.host_ptr, b_.h_a.host_ptr, size_t(front) * hstride);

    // --- engram -------------------------------------------------------------------
    if (c.is_engram_layer(L)) {
        if (!engram_) return fail(Err::FailedPrecondition, "an engram layer needs EngramTables");
        PF_TRY(engram_rows(L, prompt, b_.eng_x.dev_addr));
        const auto t0 = Clk::now();
        PF_TRY(op_act_quant(b_.eng_x.dev_addr, front, 6144, b_.eng_xq.dev_addr, b_.eng_xs.dev_addr));
        auto wkv = W("engram.wkv.weight", kPfFp8);
        if (!wkv) return std::unexpected(wkv.error());
        PF_TRY(op_gemm(*wkv, kPfActQ, b_.eng_xq.dev_addr, b_.eng_xs.dev_addr, front, 6144,
                       b_.eng_kv.dev_addr, round));
        PF_TRY(op_engram_gate(b_.h_a.dev_addr, b_.eng_kv.dev_addr, pw("engram.q_weight"),
                              pw("engram.k_weight"), b_.h_b.dev_addr, front));
        std::swap(b_.h_a, b_.h_b);
        times_.engram += ms_since(t0);
    }
    if (pcfg_.probe_layers) {
        pv.block_in = fptr(b_.h_in_copy);
        pv.engram_out = c.is_engram_layer(L) ? fptr(b_.h_a) : nullptr;
    }

    // --- mHC, attention half ----------------------------------------------------------
    auto t0 = Clk::now();
    PF_TRY(op_mhc_pre_norm(b_.h_a.dev_addr, front, b_.mix_prev.dev_addr, kMix,
                           pw("attn_norm.weight"), b_.x.dev_addr, b_.rs.dev_addr));
    {
        auto hcf = W("hc_attn_fn", kPfFp32);
        if (!hcf) return std::unexpected(hcf.error());
        PF_TRY(op_gemm(*hcf, kPfActF32, b_.h_a.dev_addr, 0, front, kHc * dim, b_.mix_raw.dev_addr,
                       kPfFlagRowScale, 1.0f, 0, b_.rs.dev_addr));
    }
    PF_TRY(op_sinkhorn(b_.mix_raw.dev_addr, b_.mix_a.dev_addr, front, pw("hc_attn_base"),
                       pw("hc_attn_scale")));
    times_.mhc += ms_since(t0);

    // --- the window KV ------------------------------------------------------------------
    t0 = Clk::now();
    PF_TRY(op_act_quant(b_.x.dev_addr, front, dim, b_.xq.dev_addr, b_.xs.dev_addr));
    {
        auto wkv = W("attn.wkv.weight", kPfFp8);
        if (!wkv) return std::unexpected(wkv.error());
        PF_TRY(op_gemm(*wkv, kPfActQ, at(b_.xq, aoff, uint64_t(dim) * 2),
                       at(b_.xs, aoff, uint64_t(dim / 32) * 4), A, dim, b_.kv_raw.dev_addr, round));
    }
    PF_TRY(op_rmsnorm(b_.kv_raw.dev_addr, b_.kv_norm.dev_addr, A, hd, pw("attn.kv_norm.weight")));
    PF_TRY(op_rope(b_.kv_norm.dev_addr, b_.kv.dev_addr, A, hd, hd, 1, 32, cmp_theta, apos0, 1, false));

    // --- the compressor and the index keys, on a kv source ------------------------------
    if (c.is_kv_source(L)) {
        const uint32_t G = front / ratio;
        SourceState& src = sources_[L];
        PrefillHandoff::Layer& hl = out.layers[L];
        hl.ratio = ratio;
        auto cw = W("attn.compressor.wkv.weight", kPfBf16);
        if (!cw) return std::unexpected(cw.error());
        if (ratio > 1) {
            auto cg = W("attn.compressor.wgate.weight", kPfBf16);
            if (!cg) return std::unexpected(cg.error());
            // x.float(): both projections and the pooling in fp32, no rounding
            PF_TRY(op_gemm(*cw, kPfActF32, b_.x.dev_addr, 0, front, dim, b_.ckv.dev_addr, 0));
            PF_TRY(op_gemm(*cg, kPfActF32, b_.x.dev_addr, 0, front, dim, b_.cscore.dev_addr, 0));
            PF_TRY(op_cmp_pool(b_.ckv.dev_addr, b_.cscore.dev_addr, b_.latent_pre.dev_addr, G, ratio));
            // the trailing partial group waits in the state (Compressor.forward)
            hl.cmp_state_kv.assign(size_t(ratio) * hd, 0.0f);
            hl.cmp_state_score.assign(size_t(ratio) * hd, -std::numeric_limits<float>::infinity());
            const uint32_t rem = front % ratio;
            for (uint32_t j = 0; j < rem; ++j) {
                std::memcpy(hl.cmp_state_kv.data() + size_t(j) * hd,
                            fptr(b_.ckv, size_t(G * ratio + j) * hd), hd * sizeof(float));
                std::memcpy(hl.cmp_state_score.data() + size_t(j) * hd,
                            fptr(b_.cscore, size_t(G * ratio + j) * hd), hd * sizeof(float));
            }
        } else {
            PF_TRY(op_gemm(*cw, kPfActF32, b_.x.dev_addr, 0, front, dim, b_.latent_pre.dev_addr, round));
        }
        PF_TRY(op_rmsnorm(b_.latent_pre.dev_addr, b_.latent.dev_addr, G, hd,
                          pw("attn.compressor.norm.weight")));
        auto wk = W("attn.indexer.wk.weight", kPfBf16);
        if (!wk) return std::unexpected(wk.error());
        PF_TRY(op_gemm(*wk, kPfActF32, b_.latent.dev_addr, 0, G, hd, b_.key_raw.dev_addr, round));
        PF_TRY(op_rmsnorm(b_.key_raw.dev_addr, b_.key_norm.dev_addr, G, c.index_head_dim,
                          pw("attn.indexer.k_norm.weight")));
        PF_TRY(op_rope(b_.key_norm.dev_addr, src.keys.dev_addr, G, c.index_head_dim,
                       c.index_head_dim, 2, 32, true, fpos0, ratio, false));
        PF_TRY(op_rope(b_.latent.dev_addr, src.cache.dev_addr, G, hd, hd, 3, 16, true, fpos0,
                       ratio, false));
        src.n = G;
        src.valid = true;
        cmp_src_ = L;
        key_src_ = L;
        hl.n_cmp = G;
        hl.cmp_cache.assign(fptr(src.cache), fptr(src.cache) + size_t(G) * hd);
        hl.index_k.assign(fptr(src.keys), fptr(src.keys) + size_t(G) * c.index_head_dim);
        pv.cmp_latent = fptr(b_.latent);
        pv.cmp_cache = fptr(src.cache);
        pv.index_k = fptr(src.keys);
        pv.n_cmp = G;
    }

    // --- queries, in blocks ----------------------------------------------------------------
    const uint32_t G = ratio ? sources_[cmp_src_].n : 0;
    const uint32_t nw = std::min(A, win);
    const uint32_t k = ratio ? std::min<uint32_t>(c.index_topk, G) : 0;
    const uint32_t n_idx = nw + k;
    auto wqa = W("attn.wq_a.weight", kPfFp8);
    auto wqb = W("attn.wq_b.weight", kPfFp8);
    auto woa = W("attn.wo_a.weight", kPfFp8);
    auto wob = W("attn.wo_b.weight", kPfFp8);
    if (!wqa || !wqb || !woa || !wob) return fail(Err::NotFound, "attention weights");
    PF_TRY(op_gemm(*wqa, kPfActQ, at(b_.xq, aoff, uint64_t(dim) * 2),
                   at(b_.xs, aoff, uint64_t(dim / 32) * 4), A, dim, b_.qr_raw.dev_addr, round));
    PF_TRY(op_rmsnorm(b_.qr_raw.dev_addr, b_.qr.dev_addr, A, c.q_lora_rank, pw("attn.q_norm.weight")));
    PF_TRY(op_act_quant(b_.qr.dev_addr, A, c.q_lora_rank, b_.qrq.dev_addr, b_.qrs.dev_addr));
    const uint32_t qdim = c.num_attention_heads * hd;
    const uint32_t B = std::min(pcfg_.query_block, pcfg_.max_tokens);
    const uint32_t ql = c.q_lora_rank;
    if (c.is_index_source(L)) topk_shared_.assign(size_t(A) * k, -1);
    std::vector<int32_t> idx_host;
    std::vector<float> scores_host;
    for (uint32_t b0 = 0; b0 < A; b0 += B) {
        const uint32_t nb = std::min(B, A - b0);
        const uint32_t qpos = apos0 + b0;
        PF_TRY(op_gemm(*wqb, kPfActQ, at(b_.qrq, b0, uint64_t(ql) * 2),
                       at(b_.qrs, b0, uint64_t(ql / 32) * 4), nb, ql, b_.q.dev_addr, round));
        PF_TRY(op_rope(b_.q.dev_addr, b_.q.dev_addr, nb, qdim, hd, 0, 32, cmp_theta, qpos, 1, false));
        idx_host.assign(size_t(nb) * n_idx, -1);
        if (c.is_index_source(L) && ratio) {
            auto iqb = W("attn.indexer.wq_b.weight", kPfFp8);
            auto iwp = W("attn.indexer.weights_proj.weight", kPfBf16);
            if (!iqb || !iwp) return fail(Err::NotFound, "indexer weights");
            const uint32_t ih = c.index_n_heads, ihd = c.index_head_dim;
            PF_TRY(op_gemm(*iqb, kPfActQ, at(b_.qrq, b0, uint64_t(ql) * 2),
                           at(b_.qrs, b0, uint64_t(ql / 32) * 4), nb, ql, b_.iq.dev_addr, round));
            PF_TRY(op_rope(b_.iq.dev_addr, b_.iq.dev_addr, nb, ih * ihd, ihd, 2, 32, true, qpos, 1,
                           false));
            PF_TRY(op_gemm(*iwp, kPfActF32, at(b_.x, aoff + b0, uint64_t(dim) * 4), 0, nb, dim,
                           b_.iw.dev_addr, round | kPfFlagRoundPre,
                           static_cast<float>(1.0 / std::sqrt(double(ihd)) / std::sqrt(double(ih)))));
            PF_TRY(op_index_score(b_.iq.dev_addr, sources_[key_src_].keys.dev_addr, G,
                                  b_.iw.dev_addr, b_.iscore.dev_addr, nb, ratio, qpos));
            auto th = Clk::now();
            scores_host.resize(size_t(nb) * G);
            std::memcpy(scores_host.data(), b_.iscore.host_ptr, scores_host.size() * sizeof(float));
            host_op("host: index score readback", th);
            // design §2.1's two-level top-k: layer 20 picks candidate blocks,
            // the later index layers score only inside them. The identity for
            // every query that can reach <= topk_blocks blocks (§1).
            if (c.candidate_block_size && c.candidate_topk_blocks) {
                th = Clk::now();
                if (L == c.candidate_source_layer_id) {
                    if (b0 == 0) cand_.assign(A, {});
                    auto kb = candidate_blocks(nb, qpos, ratio, G, scores_host.data(),
                                               c.candidate_topk_blocks, c.candidate_block_size);
                    for (uint32_t j = 0; j < nb; ++j) cand_[b0 + j] = std::move(kb[j]);
                } else if (L > c.candidate_source_layer_id && cand_.size() == A) {
                    const uint32_t bsz = c.candidate_block_size;
                    for (uint32_t j = 0; j < nb; ++j) {
                        const std::vector<uint8_t>& keep = cand_[b0 + j];
                        if (keep.empty()) continue;
                        float* sr = scores_host.data() + size_t(j) * G;
                        for (uint32_t i = 0; i < G; ++i)
                            if (i / bsz >= keep.size() || !keep[i / bsz])
                                sr[i] = -std::numeric_limits<float>::infinity();
                    }
                }
                host_op("host: candidate blocks", th);
            }
            th = Clk::now();
            topk_rows(nb, qpos, apos0, A, win, ratio, G, c.index_topk, scores_host.data(),
                      idx_host.data(), n_idx);
            host_op("host: index top-k", th);
            for (uint32_t j = 0; j < nb; ++j)
                std::memcpy(topk_shared_.data() + size_t(b0 + j) * k,
                            idx_host.data() + size_t(j) * n_idx + nw, size_t(k) * sizeof(int32_t));
        } else {
            std::vector<int32_t> band(size_t(nb) * nw);
            topk_rows(nb, qpos, apos0, A, win, 0, 0, 0, nullptr, band.data(), nw);
            for (uint32_t j = 0; j < nb; ++j) {
                std::memcpy(idx_host.data() + size_t(j) * n_idx, band.data() + size_t(j) * nw,
                            size_t(nw) * sizeof(int32_t));
                // a reuse layer reads its index source's picks (`shared_attn.topk_idxs`)
                if (k && topk_shared_.size() >= size_t(b0 + j + 1) * k)
                    std::memcpy(idx_host.data() + size_t(j) * n_idx + nw,
                                topk_shared_.data() + size_t(b0 + j) * k, size_t(k) * sizeof(int32_t));
            }
        }
        std::memcpy(b_.idx.host_ptr, idx_host.data(), idx_host.size() * sizeof(int32_t));
        if (b0 + nb == A) {
            // in the reference's numbering: window = absolute positions,
            // compressed = row + N (identical to ours in oracle mode)
            std::vector<int32_t>& tl = out.layers[L].topk_last;
            tl.assign(idx_host.end() - n_idx, idx_host.end());
            for (uint32_t i = 0; i < n_idx; ++i)
                if (tl[i] >= 0) tl[i] = i < nw ? tl[i] + int32_t(apos0) : tl[i] - int32_t(A) + int32_t(N);
        }
        if (b0 == 0) {
            probe_idx_ = idx_host;
            pv.topk_first = probe_idx_.data();
            pv.n_idx = n_idx;
        }
        PF_TRY(op_attention(b_.q.dev_addr, b_.kv.dev_addr, A,
                            ratio ? sources_[cmp_src_].cache.dev_addr : 0, b_.idx.dev_addr, n_idx,
                            pw("attn.attn_sink"), b_.o.dev_addr, nb));
        PF_TRY(op_rope(b_.o.dev_addr, b_.o.dev_addr, nb, qdim, hd, 0, 32, cmp_theta, qpos, 1, true));
        PF_TRY(op_gemm(*woa, kPfActF32, b_.o.dev_addr, 0, nb, qdim, b_.woa.dev_addr, round, 1.0f,
                       c.o_lora_rank));
        const uint32_t orows = c.o_groups * c.o_lora_rank;
        PF_TRY(op_act_quant(b_.woa.dev_addr, nb, orows, b_.woaq.dev_addr, b_.woas.dev_addr));
        PF_TRY(op_gemm(*wob, kPfActQ, b_.woaq.dev_addr, b_.woas.dev_addr, nb, orows,
                       at(b_.attn, b0, uint64_t(dim) * 4), round));
    }
    times_.attention += ms_since(t0);

    // the decode ring: the last min(A, 128) rows, slot p % 128 holds position p
    {
        PrefillHandoff::Layer& hl = out.layers[L];
        hl.win_kv.assign(size_t(win) * hd, 0.0f);
        const uint32_t take = std::min(A, win);
        for (uint32_t i = A - take; i < A; ++i)
            std::memcpy(hl.win_kv.data() + size_t((apos0 + i) % win) * hd,
                        fptr(b_.kv, size_t(i) * hd), hd * sizeof(float));
    }

    // --- hc_post, then the FFN half -------------------------------------------------------
    t0 = Clk::now();
    PF_TRY(op_mhc_post(at(b_.h_a, aoff, hstride), b_.attn.dev_addr, at(b_.mix_a, aoff, kMix * 4),
                       b_.h_b.dev_addr, A));
    PF_TRY(op_mhc_pre_norm(b_.h_b.dev_addr, A, at(b_.mix_a, aoff, kMix * 4), kMix,
                           pw("ffn_norm.weight"), b_.fx.dev_addr, b_.rs.dev_addr));
    {
        auto hcf = W("hc_ffn_fn", kPfFp32);
        if (!hcf) return std::unexpected(hcf.error());
        PF_TRY(op_gemm(*hcf, kPfActF32, b_.h_b.dev_addr, 0, A, kHc * dim, b_.mix_raw.dev_addr,
                       kPfFlagRowScale, 1.0f, 0, b_.rs.dev_addr));
    }
    PF_TRY(op_sinkhorn(b_.mix_raw.dev_addr, b_.mix_f.dev_addr, A, pw("hc_ffn_base"), pw("hc_ffn_scale")));
    times_.mhc += ms_since(t0);

    t0 = Clk::now();
    PF_TRY(op_act_quant(b_.fx.dev_addr, A, dim, b_.fxq.dev_addr, b_.fxs.dev_addr));
    {
        auto gw = W("ffn.gate.weight", kPfBf16);
        if (!gw) return std::unexpected(gw.error());
        // linear(x.float(), weight.float()): no activation round trip, fp32 out
        PF_TRY(op_gemm(*gw, kPfActF32, b_.fx.dev_addr, 0, A, dim, b_.gate.dev_addr, 0));
    }
    const uint32_t E = c.n_routed_experts, k6 = c.num_experts_per_tok;
    const uint64_t bias_dev = pw("ffn.gate.bias");
    probe_ids_.assign(size_t(A) * k6, 0);
    probe_wts_.assign(size_t(A) * k6, 0.0f);
    if (pcfg_.gate_topk_gpu && bias_dev) {
        // sqrtsoftplus + noaux_tc top-6 on the GPU: only A x 6 ids and weights
        // cross the device-mapped bus, not A x 384 scores.
        PF_TRY(op_gate_topk(b_.gate.dev_addr, bias_dev, A, b_.gids.dev_addr, b_.gwts.dev_addr));
        auto tg = Clk::now();
        std::memcpy(probe_ids_.data(), b_.gids.host_ptr, probe_ids_.size() * sizeof(uint32_t));
        std::memcpy(probe_wts_.data(), b_.gwts.host_ptr, probe_wts_.size() * sizeof(float));
        host_op("host: gate top-6 readback", tg);
    } else {
    auto th = Clk::now();
    std::vector<float> sc(size_t(A) * E);
    std::memcpy(sc.data(), b_.gate.host_ptr, sc.size() * sizeof(float));
    host_op("host: gate readback", th);
    th = Clk::now();
    const store::PinnedTensor* bias_t = pinned_->find(P + "ffn.gate.bias");
    // copied once: the pinned host view is device-mapped memory, and the
    // top-6 compares read the bias ~3,000 times a token (58 s of a 4K prefill)
    std::vector<float> bias_host;
    if (bias_t)
        bias_host.assign(static_cast<const float*>(bias_t->data_host),
                         static_cast<const float*>(bias_t->data_host) + E);
    std::span<const float> bias(bias_host);
    for (uint32_t t = 0; t < A; ++t) {
        float* row = sc.data() + size_t(t) * E;
        for (uint32_t e = 0; e < E; ++e) row[e] = cpu::sqrt_softplus(row[e]);
        auto g = cpu::gate_topk(std::span<const float>(row, E), bias, k6, 0,
                                static_cast<float>(c.routed_scaling_factor));
        if (!g) return std::unexpected(g.error());
        for (uint32_t s = 0; s < k6; ++s) {
            probe_ids_[size_t(t) * k6 + s] = g->ids[s];
            probe_wts_[size_t(t) * k6 + s] = g->weights[s];
        }
    }
    host_op("host: gate top-6", th);
    }
    out.layers[L].gate_ids_last.assign(probe_ids_.end() - k6, probe_ids_.end());
    times_.gate += ms_since(t0);
    std::memset(b_.y.host_ptr, 0, size_t(A) * dim * sizeof(float));
    moe_pos0_ = apos0;
    PF_TRY(run_moe(L, A, b_.fx.dev_addr, b_.fxq.dev_addr, b_.fxs.dev_addr, b_.y.dev_addr,
                   probe_ids_, probe_wts_));

    t0 = Clk::now();
    PF_TRY(op_mhc_post(b_.h_b.dev_addr, b_.y.dev_addr, b_.mix_f.dev_addr, b_.h_a.dev_addr, A));
    times_.mhc += ms_since(t0);

    if (pcfg_.probe_layers && probe) {
        pv.layer = L; pv.rows = front; pv.row0 = fpos0; pv.front_rows = front;
        pv.attn_rows = A; pv.attn_row0 = apos0;
        pv.attn_norm_out = fptr(b_.x);
        pv.mix_attn = fptr(b_.mix_a);
        pv.kv = fptr(b_.kv);
        pv.attn_out = fptr(b_.attn);
        pv.attn_block_out = fptr(b_.h_b);
        pv.mix_ffn = fptr(b_.mix_f);
        pv.ffn_norm_out = fptr(b_.fx);
        pv.moe_out = fptr(b_.y);
        pv.block_out = fptr(b_.h_a);
        pv.gate_ids = probe_ids_.data();
        pv.gate_weights = probe_wts_.data();
        probe(pv);
    }
    std::swap(b_.mix_prev, b_.mix_f);
    return {};
}

#undef PF_TRY

#endif  // DEEPMOE_ENABLE_VULKAN

// --- the L3-format handoff directory ------------------------------------------------

namespace {

Result<std::vector<uint8_t>> slurp(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(Err::NotFound, std::format("cannot open '{}'", path));
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(static_cast<size_t>(n < 0 ? 0 : n));
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    if (got != v.size()) return fail(Err::Io, std::format("short read of '{}'", path));
    return v;
}

Result<void> spill(const std::string& path, const uint8_t* p, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return fail(Err::Io, std::format("cannot create '{}'", path));
    const size_t put = n ? std::fwrite(p, 1, n, f) : 0;
    std::fclose(f);
    if (put != n) return fail(Err::Io, std::format("short write of '{}'", path));
    return {};
}

// [begin, end) of the JSON value that starts at `pos` (an object or array),
// skipping string contents.
size_t json_value_end(const std::string& s, size_t pos) {
    int depth = 0;
    bool in_str = false;
    for (size_t i = pos; i < s.size(); ++i) {
        const char ch = s[i];
        if (in_str) {
            if (ch == '\\') { ++i; continue; }
            if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') in_str = true;
        else if (ch == '{' || ch == '[') ++depth;
        else if (ch == '}' || ch == ']') { if (--depth == 0) return i + 1; }
    }
    return std::string::npos;
}

}  // namespace

Result<void> Prefill::write_l3_dir(const PrefillHandoff& h, const TextConfig& cfg,
                                   const std::string& ref, const std::string& dir) {
    auto idx_bytes = slurp(ref + "/index.json");
    if (!idx_bytes) return std::unexpected(idx_bytes.error());
    std::string text(idx_bytes->begin(), idx_bytes->end());
    auto doc = json_parse_file(ref + "/index.json");
    if (!doc) return std::unexpected(doc.error());
    auto steps = doc->at("steps");
    if (!steps) return std::unexpected(steps.error());
    auto arr = (*steps)->as_array();
    if (!arr) return std::unexpected(arr.error());
    if ((*arr)->empty()) return fail(Err::Corrupt, "reference L3 export has no records");
    const JsonValue& pre = (**arr)[0];
    if (pre.string_or("step", "") != "prefill")
        return fail(Err::Corrupt, "reference L3 export does not start with its prefill record");
    const std::string pre_file = pre.string_or("file", "l3_prefill.bin");
    auto pre_blob = slurp(ref + "/" + pre_file);
    if (!pre_blob) return std::unexpected(pre_blob.error());
    const uint64_t base = static_cast<uint64_t>(pre.int_or("data_offset", 12));

    // Our state under the export's names; everything else in the record (the
    // reference's prefill logits, gate biases, collapse inputs) is carried over
    // byte for byte so the loader's reference half is unchanged.
    std::vector<uint8_t> blob;
    std::string tensors;
    auto add = [&](const std::string& name, const std::string& dtype,
                   const std::vector<uint64_t>& shape, const uint8_t* p, size_t n) {
        if (!tensors.empty()) tensors += ",\n";
        std::string sh;
        for (size_t i = 0; i < shape.size(); ++i) sh += (i ? ", " : "") + std::to_string(shape[i]);
        tensors += std::format("   {{\"name\": \"{}\", \"dtype\": \"{}\", \"shape\": [{}], "
                               "\"offset\": {}, \"bytes\": {}}}", name, dtype, sh, blob.size(), n);
        blob.insert(blob.end(), p, p + n);
    };
    const uint32_t hd = cfg.head_dim, ihd = cfg.index_head_dim;
    std::vector<std::string> ours;
    for (uint32_t L = 0; L < h.layers.size(); ++L) {
        const PrefillHandoff::Layer& l = h.layers[L];
        const std::string P = std::format("L{:02d}.", L);
        auto f32 = [&](const std::string& n, const std::vector<float>& v, uint64_t cols) {
            if (v.empty()) return;
            add(P + n, "f32", {v.size() / cols, cols}, reinterpret_cast<const uint8_t*>(v.data()),
                v.size() * sizeof(float));
            ours.push_back(P + n);
        };
        f32("win_kv", l.win_kv, hd);
        f32("cmp_cache", l.cmp_cache, hd);
        f32("index_k", l.index_k, ihd);
        f32("cmp_state_kv", l.cmp_state_kv, hd);
        f32("cmp_state_score", l.cmp_state_score, hd);
    }
    auto tens = pre.at("tensors");
    if (!tens) return std::unexpected(tens.error());
    auto ta = (*tens)->as_array();
    if (!ta) return std::unexpected(ta.error());
    for (const JsonValue& e : **ta) {
        const std::string name = e.string_or("name", "");
        const bool state = name.size() > 4 && name[0] == 'L' &&
                           (name.ends_with(".win_kv") || name.ends_with(".cmp_cache") ||
                            name.ends_with(".index_k") || name.ends_with(".cmp_state_kv") ||
                            name.ends_with(".cmp_state_score"));
        if (state) continue;
        std::vector<uint64_t> shape;
        if (const JsonValue* s = e.find("shape"))
            if (auto a = s->as_array())
                for (const JsonValue& d : **a) shape.push_back(d.as_uint().value_or(0));
        const uint64_t off = base + static_cast<uint64_t>(e.int_or("offset", 0));
        const uint64_t n = static_cast<uint64_t>(e.int_or("bytes", 0));
        if (off + n > pre_blob->size()) return fail(Err::Corrupt, "reference prefill record overruns");
        add(name, e.string_or("dtype", "f32"), shape, pre_blob->data() + off, n);
    }
    std::vector<uint8_t> file = {'D', 'M', 'L', '3', 1, 0, 0, 0, 0, 0, 0, 0};
    file.insert(file.end(), blob.begin(), blob.end());
    if (auto r = spill(dir + "/l3_prefill.bin", file.data(), file.size()); !r) return r;
    const std::string record = std::format(
        "{{\n  \"step\": \"prefill\",\n  \"file\": \"l3_prefill.bin\",\n  \"data_offset\": 12,\n"
        "  \"bytes\": {},\n  \"tensors\": [\n{}\n  ]\n }}", blob.size(), tensors);

    // splice our record over the reference's first `steps` entry
    const size_t sp = text.rfind("\"steps\"");
    if (sp == std::string::npos) return fail(Err::Corrupt, "no steps array");
    const size_t lb = text.find('[', sp);
    const size_t ob = text.find('{', lb);
    const size_t oe = json_value_end(text, ob);
    if (lb == std::string::npos || ob == std::string::npos || oe == std::string::npos)
        return fail(Err::Corrupt, "cannot locate the prefill record");
    text = text.substr(0, ob) + record + text.substr(oe);
    if (auto r = spill(dir + "/index.json", reinterpret_cast<const uint8_t*>(text.data()), text.size()); !r)
        return r;
    // the step records and the engram token map, verbatim
    for (size_t i = 1; i < (*arr)->size(); ++i) {
        const std::string f = (**arr)[i].string_or("file", "");
        auto b = slurp(ref + "/" + f);
        if (!b) return std::unexpected(b.error());
        if (auto r = spill(dir + "/" + f, b->data(), b->size()); !r) return r;
    }
    if (const JsonValue* eg = doc->find("engram")) {
        const std::string mapfile = eg->string_or("token_map_file", "engram_token_map.bin");
        auto b = slurp(ref + "/" + mapfile);
        if (!b) return std::unexpected(b.error());
        if (auto r = spill(dir + "/" + mapfile, b->data(), b->size()); !r) return r;
    }
    (void)ours;
    return {};
}

}  // namespace deepmoe::gpu
