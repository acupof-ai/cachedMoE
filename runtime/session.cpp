#include "runtime/session.h"

#include <algorithm>
#include <array>
#include <span>
#include <chrono>
#include <format>

#include "core/json_write.h"
#include "core/log.h"

namespace deepmoe::runtime {
namespace {

double ms_since(TimePoint t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct CacheCount { uint64_t req = 0, hit = 0, bytes = 0; };

CacheCount count_step(const Engine& e) {
    CacheCount c;
    for (const LayerTiming& t : e.layer_timings()) {
        c.req += t.hits + t.misses;
        c.hit += t.hits;
        c.bytes += t.miss_bytes;
    }
    return c;
}

void add(StepBreakdown& s, const StepBreakdown& b) {
    s.attn_ms += b.attn_ms;       s.moe_gpu_ms += b.moe_gpu_ms; s.moe_host_ms += b.moe_host_ms;
    s.gate_ms += b.gate_ms;       s.engram_ms += b.engram_ms;   s.tail_ms += b.tail_ms;
    s.other_ms += b.other_ms;     s.record_ms += b.record_ms;   s.submit_ms += b.submit_ms;
    s.wait_ms += b.wait_ms;       s.bind_ms += b.bind_ms;       s.submits += b.submits;
}

}  // namespace

std::string GenerateStats::json_fields() const {
    const double n = decode_steps ? double(decode_steps) : 1.0;
    std::string s;
    s += std::format("\"prompt_tokens\":{},\"reused_tokens\":{},\"prefill_tokens\":{},\"generated\":{},",
                     prompt_tokens, reused_tokens, prefilled_tokens, generated);
    s += std::format("\"prefill_mode\":{},\"finish\":{},", json_quote(prefill_mode), json_quote(finish));
    s += std::format("\"prefill_ms\":{},\"prefill_tok_s\":{},\"ttft_ms\":{},\"decode_ms\":{},"
                     "\"decode_steps\":{},\"tok_s\":{},\"total_ms\":{},",
                     json_number(prefill_ms), json_number(prefill_tok_s()), json_number(ttft_ms),
                     json_number(decode_ms), decode_steps, json_number(decode_tok_s()),
                     json_number(total_ms));
    s += std::format("\"prefill_hit_rate\":{},\"decode_hit_rate\":{},\"prefill_nvme_mb\":{},"
                     "\"decode_nvme_mb\":{},",
                     json_number(prefill_hit_rate()), json_number(decode_hit_rate()),
                     json_number(prefill_nvme_bytes / 1e6), json_number(decode_nvme_bytes / 1e6));
    const StepBreakdown& b = decode_sum;
    s += std::format("\"per_token_ms\":{{\"attn\":{},\"moe_gpu\":{},\"moe_host\":{},\"nvme_stall\":{},"
                     "\"engram\":{},\"tail\":{},\"other\":{},\"record\":{},\"submit\":{},\"bind\":{},"
                     "\"fence_wait\":{},\"submits\":{},\"sample_host\":{}}},",
                     json_number(b.attn_ms / n), json_number(b.moe_gpu_ms / n),
                     json_number(b.moe_host_ms / n), json_number(b.gate_ms / n),
                     json_number(b.engram_ms / n), json_number(b.tail_ms / n),
                     json_number(b.other_ms / n), json_number(b.record_ms / n),
                     json_number(b.submit_ms / n), json_number(b.bind_ms / n),
                     json_number(b.wait_ms / n), json_number(b.submits / n),
                     json_number(sampled_steps ? sample_ms_sum / sampled_steps : 0.0));
    s += std::format("\"sampled_steps\":{},\"topk_fallbacks\":{},\"topk_checked\":{},"
                     "\"topk_mismatches\":{},\"mean_nucleus\":{},\"context\":{}",
                     sampled_steps, topk_fallbacks, topk_checked, topk_mismatches,
                     json_number(sampled_steps ? double(nucleus_sum) / sampled_steps : 0.0),
                     context_after);
    return s;
}

Result<GenerateStats> Session::generate(
    const GenerateRequest& req, const std::function<void(const TokenEvent&)>& on_token,
    const std::function<void(uint32_t, uint32_t)>& on_prefill) {
    const TimePoint t0 = Clock::now();
    Engine& e = *engine_;
    GenerateStats st;
    const std::vector<uint32_t>& prompt = req.prompt_ids;
    st.prompt_tokens = static_cast<uint32_t>(prompt.size());
    if (prompt.empty()) return fail(Err::InvalidArgument, "empty prompt");
    for (uint32_t id : prompt)
        if (id >= tok_->vocab_size())
            return fail(Err::InvalidArgument, std::format("token id {} is outside the vocabulary", id));
    if (prompt.size() >= e.max_context())
        return fail(Err::ResourceExhausted,
                    std::format("a {}-token prompt does not fit the {}-position context",
                                prompt.size(), e.max_context()));

    // KV continuation: reuse the history if the prompt strictly extends it.
    const std::vector<uint32_t>& hist = e.history();
    uint32_t reuse = 0;
    if (req.reuse && !hist.empty() && hist.size() < prompt.size() &&
        std::equal(hist.begin(), hist.end(), prompt.begin()))
        reuse = static_cast<uint32_t>(hist.size());
    if (reuse == 0 && !hist.empty()) {
        size_t common = 0;
        while (common < hist.size() && common < prompt.size() && hist[common] == prompt[common]) ++common;
        log_info("session: prompt does not extend the {} tokens in the KV store (common prefix {}); "
                 "resetting", hist.size(), common);
        e.reset_context();
    }
    st.reused_tokens = reuse;
    e.set_sampling(req.sampling);

    const std::span<const uint32_t> suffix(prompt.data() + reuse, prompt.size() - reuse);
    st.prefilled_tokens = static_cast<uint32_t>(suffix.size());
    DecodeStepResult first{};
    const TimePoint tp = Clock::now();
    bool done_prefill = false;
    if (reuse == 0 && opt_.gpu_prefill_min && suffix.size() >= opt_.gpu_prefill_min) {
        auto r = e.gpu_prefill(suffix, opt_.replay);
        if (r) {
            first = *r;
            st.prefill_mode = "gpu";
            done_prefill = true;
        } else {
            log_warn("session: GPU prefill failed ({}); falling back to the decode path",
                     r.error().str());
            e.reset_context();
        }
    }
    if (!done_prefill) {
        st.prefill_mode = "decode";
        const uint32_t total = static_cast<uint32_t>(suffix.size());
        auto r = e.feed(suffix, [&](uint32_t i, const DecodeStepResult&) {
            const CacheCount c = count_step(e);
            st.prefill_requests += c.req;
            st.prefill_hits += c.hit;
            st.prefill_nvme_bytes += c.bytes;
            if (on_prefill && ((i + 1) % std::max<uint32_t>(1, opt_.progress_every) == 0 || i + 1 == total))
                on_prefill(i + 1, total);
        });
        if (!r) return std::unexpected(r.error());
        first = *r;
    }
    st.prefill_ms = ms_since(tp);

    text::StreamDecoder sd(*tok_, /*skip_special=*/true);
    auto account_sample = [&](const DecodeStepResult& r) {
        if (!r.sampled) return;
        ++st.sampled_steps;
        st.topk_fallbacks += r.topk_fallback;
        st.topk_checked += r.topk_checked;
        st.topk_mismatches += r.topk_mismatch;
        st.nucleus_sum += r.nucleus_size;
        st.sample_ms_sum += r.sample_ms;
    };
    account_sample(first);

    DecodeStepResult cur = first;
    double step_ms = first.wall_ms;
    double step_hit = 0.0;
    const TimePoint td = Clock::now();
    for (;;) {
        const uint32_t tok = cur.token;
        ++st.generated;
        TokenEvent ev;
        ev.id = tok;
        ev.text = sd.push(tok);
        ev.t_ms = ms_since(t0);
        ev.step_ms = step_ms;
        ev.p = cur.p_token;
        ev.margin = cur.margin();
        ev.nucleus = cur.nucleus_size;
        ev.fallback = cur.topk_fallback;
        ev.hit_rate = step_hit;
        if (st.generated == 1) st.ttft_ms = ev.t_ms;
        const bool stop = std::find(req.stop_ids.begin(), req.stop_ids.end(), tok) != req.stop_ids.end();
        if (stop) ev.text += sd.flush();
        if (on_token) on_token(ev);
        if (stop) { st.finish = "stop"; break; }
        if (st.generated >= req.max_tokens) { st.finish = "length"; break; }
        if (e.context_length() + 1 >= e.max_context()) { st.finish = "context"; break; }
        const std::array<uint32_t, 1> one{tok};
        auto r = e.feed(one);
        if (!r) return std::unexpected(r.error());
        cur = *r;
        step_ms = cur.wall_ms;
        const CacheCount c = count_step(e);
        st.decode_requests += c.req;
        st.decode_hits += c.hit;
        st.decode_nvme_bytes += c.bytes;
        step_hit = c.req ? double(c.hit) / double(c.req) : 0.0;
        add(st.decode_sum, cur.breakdown);
        ++st.decode_steps;
        account_sample(cur);
    }
    st.decode_ms = st.decode_steps ? ms_since(td) : 0.0;
    st.total_ms = ms_since(t0);
    st.context_after = e.context_length();
    return st;
}

}  // namespace deepmoe::runtime
