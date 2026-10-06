#include "runtime/gpu_route_state.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <set>

#include "core/wc_read.h"
#include "model/layout.h"
#include "runtime/engine.h"

namespace cachedmoe::runtime {
namespace {
double ms_since(TimePoint start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace

Result<void> GpuRouteBuffers::initialize(gpu::MemoryAllocator& allocator, const TextConfig& c) {
    if (scratch.capacity()) return {};
    if (auto r = scratch.create(allocator, kRouteScratchBytes); !r) return r;
    auto take = [&](gpu::GpuScratch::View& v, uint64_t n) -> Result<void> {
        auto r = scratch.alloc(n);
        if (!r) return std::unexpected(r.error());
        v = *r;
        return {};
    };
    for (auto* views : {&routes, &rope, &rope_lat, &carry_k, &carry_g})
        views->resize(c.num_hidden_layers);
    hidden.resize(c.dspark_target_layer_ids.size());
    for (uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        if (auto r = take(routes[l], layout::kSavedRouteWords * sizeof(uint32_t)); !r) return r;
        if (auto r = take(rope[l], layout::kMoeBatchColumns * c.qk_rope_head_dim * sizeof(float));
            !r)
            return r;
        if (auto r =
                take(rope_lat[l], layout::kMoeBatchColumns * c.qk_rope_head_dim * sizeof(float));
            !r)
            return r;
        if (auto r = take(carry_k[l], layout::kMoeBatchColumns * c.head_dim * sizeof(float)); !r)
            return r;
        if (auto r = take(carry_g[l], layout::kMoeBatchColumns * c.head_dim * sizeof(float)); !r)
            return r;
    }
    for (auto& v : hidden)
        if (auto r = take(v, layout::kMoeBatchColumns * c.hidden_size * sizeof(float)); !r)
            return r;
    return {};
}

Result<void> GpuRouteState::finish(Engine& engine, uint32_t p0, uint32_t M) {
    const auto& c = engine.model_cfg_.text;
    const auto topk = c.num_experts_per_tok;
    constexpr auto records = layout::kGateRecordCount;
    constexpr auto weight_offset = layout::kMoeBatchColumns * records;
    engine.batch_route_host_ms_.fill(0);
    // The batch fence has completed. Read back its immutable routing snapshot
    // before planning asynchronous demand fills for the following cycle.
    for (const auto& st : steps) {
        const uint32_t L = st.layer;
        auto stamp = Clock::now();
        if (auto r = engine.cur_->layer_.verify_after_attention_batch(st); !r) return r;
        if (engine.spec_diagnostics_) engine.batch_route_host_ms_[0] += ms_since(stamp);
        stamp = Clock::now();
        if (const auto capture = engine.draft_layer_slot(L); engine.dspark_ && capture) {
            if (engine.cur_->draft_hidden_.empty())
                engine.cur_->draft_hidden_.resize(layout::kSlidingWindow *
                                                  layout::kDsparkCaptureWidth);
            for (uint32_t m = 0; m < M; ++m) {
                const uint32_t pos = p0 + m, slot = pos % layout::kSlidingWindow;
                if (engine.cur_->draft_position_[slot] != pos) {
                    engine.cur_->draft_position_[slot] = pos;
                    engine.cur_->draft_mask_[slot] = 0;
                }
                wc_readback(engine.cur_->draft_hidden_.data() +
                                size_t(slot) * layout::kDsparkCaptureWidth +
                                *capture * c.hidden_size,
                            static_cast<const float*>(
                                engine.cur_->gpu_route_buffers_.hidden[*capture].host) +
                                size_t(m) * c.hidden_size,
                            c.hidden_size * sizeof(float));
                engine.cur_->draft_mask_[slot] |= uint8_t(1u << *capture);
            }
        }
        if (engine.spec_inflight_ && st.run_compressor && st.compress_ratio > 1) {
            Engine::BatchCarry carry;
            carry.layer = L;
            carry.kv.resize(size_t(M) * c.head_dim);
            carry.score.resize(carry.kv.size());
            wc_readback(carry.kv.data(), engine.cur_->gpu_route_buffers_.carry_k[L].host,
                        carry.kv.size() * sizeof(float));
            wc_readback(carry.score.data(), engine.cur_->gpu_route_buffers_.carry_g[L].host,
                        carry.score.size() * sizeof(float));
            engine.batch_carry_.push_back(std::move(carry));
        }
        if (engine.spec_diagnostics_) engine.batch_route_host_ms_[1] += ms_since(stamp);
        stamp = Clock::now();
        std::array<uint32_t, layout::kSavedRouteWords> saved;
        wc_readback(saved.data(), engine.cur_->gpu_route_buffers_.routes[L].host, sizeof saved);
        if (engine.spec_diagnostics_) engine.batch_route_host_ms_[2] += ms_since(stamp);
        stamp = Clock::now();
        std::vector<uint16_t> chosen, near_ids;
        std::vector<float> chosen_w, near_scores;
        std::set<uint32_t> kept_union;
        uint32_t snapshot_hits = 0;
        for (uint32_t m = 0; m < M; ++m) {
            uint32_t served = 0;
            double total = 0, kept = 0;
            for (uint32_t i = 0; i < records; ++i) {
                const auto e = saved[m * records + i];
                const float w = std::bit_cast<float>(saved[weight_offset + m * records + i]);
                if (e >= c.n_routed_experts)
                    return fail(Err::Internal, "GPU route emitted invalid expert");
                near_ids.push_back(uint16_t(e));
                near_scores.push_back(w);
                if (i >= topk) continue;
                if (engine.spec_diagnostics_ && engine.spec_inflight_)
                    engine.batch_route_requests_[(size_t(L) * M + m) * topk + i] = uint16_t(e);
                total += w;
                // Residency must use the snapshot consumed by the GPU, not
                // the live store where a fill might already have landed.
                const bool hit =
                    snapshot[(size_t(L) * c.n_routed_experts + e) * layout::kExpertAddressWords] !=
                    0;
                if (hit) {
                    ++served;
                    kept += w;
                    if (w != 0) kept_union.insert(e);
                }
                const auto at = std::find(chosen.begin(), chosen.end(), uint16_t(e));
                if (at == chosen.end()) {
                    chosen.push_back(uint16_t(e));
                    chosen_w.push_back(w);
                    if (hit) ++snapshot_hits;
                } else {
                    chosen_w[size_t(at - chosen.begin())] =
                        std::max(chosen_w[size_t(at - chosen.begin())], w);
                }
            }
            ++engine.rr_.layers;
            engine.rr_.requested += topk;
            engine.rr_.served += served;
            engine.rr_.skipped += topk - served;
            engine.rr_.shared_only += served == 0;
            if (total > 0) engine.rr_.mass_lost_sum += 1 - kept / total;
        }
        store::RouteDecision route;
        route.layer = L;
        route.chosen = chosen;
        route.weights = chosen_w;
        route.near_ids = near_ids;
        route.near_scores = near_scores;
        if (engine.spec_diagnostics_) engine.batch_route_host_ms_[3] += ms_since(stamp);
        stamp = Clock::now();
        auto plan = engine.planner_.plan_layer(route, engine.clock_);
        if (!plan) return std::unexpected(plan.error());
        for (const auto& [key, use_stamp] : plan->joined)
            (void)engine.store_.touch(key, use_stamp, true);
        if (engine.spec_diagnostics_) engine.batch_route_host_ms_[4] += ms_since(stamp);
        engine.cur_->timings_[L].hits = snapshot_hits;
        engine.cur_->timings_[L].misses = uint32_t(chosen.size()) - snapshot_hits;
        engine.cur_->timings_[L].miss_bytes = plan->miss_bytes;
        engine.batch_miss_bytes_ += plan->miss_bytes;
        engine.batch_union_ += kept_union.size();
        engine.profiler_.note_hot_bytes(engine.layer_hot_bytes_[L]);
    }
    return {};
}

} // namespace cachedmoe::runtime
