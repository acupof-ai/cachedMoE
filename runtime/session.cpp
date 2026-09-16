#include "runtime/session.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <span>

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

// The ring positions a step at `k` needs -- [k - (window - 1), k) -- that the
// store does not hold, as one range starting on an even position.
ReplayPlan plan_ring(const KvStore& kv, uint32_t k, uint32_t replay_max) {
    ReplayPlan p;
    p.keep = k;
    if (k == 0) return p;
    const uint32_t w = kv.config().window;
    const uint32_t need_lo = k > w - 1 ? k - (w - 1) : 0;
    int64_t lo = -1, hi = -1;
    for (uint32_t x = need_lo; x < k; ++x) {
        if (kv.ring_holds(x)) continue;
        if (lo < 0) lo = x;
        hi = x;
    }
    if (lo < 0) return p;
    uint32_t first = static_cast<uint32_t>(lo);
    const uint32_t end = static_cast<uint32_t>(hi) + 1;
    const uint32_t r = std::clamp<uint32_t>(replay_max, 1, w);
    if (end - first > r) first = end - r;
    // A ratio-2 group must not complete on the replay's first step: its carry
    // (slot 0, the position before) is not something the replay wrote.
    if (first & 1u) first -= 1;
    p.first = first;
    p.end = end;
    return p;
}

// Positions [first, end) of the engine's history through the decode path, the
// non-SWA state kept exactly as it was (see the header).
Result<ReplayStats> run_replay(Engine& e, const ReplayPlan& plan) {
    ReplayStats st;
    st.plan = plan;
    KvStore& kv = e.kv_store();
    if (plan.steps() == 0) return st;
    const TimePoint tb = Clock::now();
    auto backup = kv.backup_rows(plan.first, plan.end);
    if (!backup) return std::unexpected(backup.error());
    st.backup_bytes = backup->bytes();
    st.backup_ms = ms_since(tb);
    kv.set_window_floor(plan.first);
    const std::vector<uint32_t> hist = e.history();
    const TimePoint t0 = Clock::now();
    for (uint32_t q = plan.first; q < plan.end; ++q) {
        auto r = e.decode_step(hist[q], q, -1);
        if (!r) return std::unexpected(r.error());
        const TimePoint tr = Clock::now();
        if (auto rr = kv.restore_rows(*backup, q); !rr) return std::unexpected(rr.error());
        st.backup_ms += ms_since(tr);
    }
    if (auto rr = kv.restore_carry(*backup); !rr) return std::unexpected(rr.error());
    st.ms = ms_since(t0);
    return st;
}

}  // namespace

ReplayPlan plan_rollback(Engine& e, uint32_t keep, uint32_t replay_max) {
    KvStore& kv = e.kv_store();
    kv.resolve_ring(e.context_length());
    return plan_ring(kv, keep, replay_max);
}

Result<ReplayStats> rollback_context(Engine& e, uint32_t keep, uint32_t replay_max) {
    const uint32_t n = e.context_length();
    if (keep > n)
        return fail(Err::InvalidArgument, std::format("rollback to {} tokens of a {}-token context", keep, n));
    if (keep & 1u)
        return fail(Err::InvalidArgument,
                    std::format("rollback to {} tokens: the point must be even, so no ratio-2 "
                                "group is left open (see runtime/session.h)", keep));
    if (keep == n) {
        ReplayStats st;
        st.plan.keep = keep;
        return st;
    }
    const ReplayPlan plan = plan_rollback(e, keep, replay_max);
    const std::vector<uint32_t> kept(e.history().begin(), e.history().begin() + keep);
    if (auto r = e.set_context_tokens(kept); !r) return std::unexpected(r.error());
    auto st = run_replay(e, plan);
    if (!st) return st;
    st->dropped = n - keep;
    log_info("session: rolled back {} -> {} tokens, replayed positions [{}, {}) in {:.0f} ms",
             n, keep, plan.first, plan.end, st->ms);
    return st;
}

Result<ParkedContext> park_context(Engine& e) {
    ParkedContext p;
    p.tokens = e.history();
    auto kv = e.kv_store().pack(e.context_length());
    if (!kv) return std::unexpected(kv.error());
    p.kv = std::move(*kv);
    return p;
}

Result<ReplayStats> restore_context(Engine& e, const ParkedContext& p, uint32_t replay_max) {
    if (p.kv.positions != p.tokens.size())
        return fail(Err::InvalidArgument,
                    std::format("parked state for {} positions with {} token ids", p.kv.positions,
                                p.tokens.size()));
    e.reset_context();
    if (p.tokens.empty()) return ReplayStats{};
    if (auto r = e.kv_store().unpack(p.kv); !r) return std::unexpected(r.error());
    if (auto r = e.set_context_tokens(p.tokens); !r) return std::unexpected(r.error());
    const ReplayPlan plan = plan_ring(e.kv_store(), static_cast<uint32_t>(p.tokens.size()), replay_max);
    return run_replay(e, plan);
}

std::string GenerateStats::json_fields() const {
    const double n = decode_steps ? double(decode_steps) : 1.0;
    std::string s;
    s += std::format("\"prompt_tokens\":{},\"reused_tokens\":{},\"prefill_tokens\":{},\"generated\":{},",
                     prompt_tokens, reused_tokens, prefilled_tokens, generated);
    s += std::format("\"prefill_mode\":{},\"finish\":{},", json_quote(prefill_mode), json_quote(finish));
    s += std::format("\"rollback_dropped\":{},\"replay_steps\":{},\"replay_ms\":{},",
                     rollback_dropped, replay_steps, json_number(replay_ms));
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
    const auto cancelled = [&] { return req.cancel && req.cancel(); };

    // KV continuation: extend the history, roll back to the common prefix, or
    // start over -- whichever feeds fewer tokens.
    uint32_t reuse = 0;
    {
        const std::vector<uint32_t>& hist = e.history();
        size_t common = 0;
        while (common < hist.size() && common < prompt.size() && hist[common] == prompt[common]) ++common;
        if (req.reuse && !hist.empty() && common == hist.size() && hist.size() < prompt.size()) {
            reuse = static_cast<uint32_t>(hist.size());
        } else if (!hist.empty()) {
            // Keep at least one prompt token to feed, and stop on a group boundary.
            uint32_t keep = static_cast<uint32_t>(std::min(common, prompt.size() - 1));
            keep &= ~1u;
            ReplayPlan plan;
            const bool try_rb = req.reuse && opt_.rollback && keep > 0;
            if (try_rb) plan = plan_rollback(e, keep, opt_.replay);
            if (try_rb && keep > plan.steps()) {
                auto rb = rollback_context(e, keep, opt_.replay);
                if (!rb) return std::unexpected(rb.error());
                reuse = keep;
                st.rollback_dropped = rb->dropped;
                st.replay_steps = rb->plan.steps();
                st.replay_ms = rb->ms;
            } else {
                log_info("session: prompt does not extend the {} tokens in the KV store (common "
                         "prefix {}); resetting", hist.size(), common);
                e.reset_context();
            }
        }
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
        const uint32_t chunk = std::max<uint32_t>(1, opt_.progress_every);
        for (uint32_t at = 0; at < total;) {
            if (cancelled()) {
                st.prefilled_tokens = at;
                st.finish = "cancel";
                st.prefill_ms = ms_since(tp);
                st.total_ms = ms_since(t0);
                st.context_after = e.context_length();
                return st;
            }
            const uint32_t n = std::min(chunk, total - at);
            auto r = e.feed(suffix.subspan(at, n), [&](uint32_t, const DecodeStepResult&) {
                const CacheCount c = count_step(e);
                st.prefill_requests += c.req;
                st.prefill_hits += c.hit;
                st.prefill_nvme_bytes += c.bytes;
            });
            if (!r) return std::unexpected(r.error());
            at += n;
            if (on_prefill) on_prefill(at, total);
            first = *r;
        }
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
        // Between tokens: the emitted token is not fed, so the store holds
        // exactly history() and the next request continues or rolls back as usual.
        if (cancelled()) { st.finish = "cancel"; break; }
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

// --- named sessions -------------------------------------------------------------------

SessionPool::SessionPool(Engine& engine, const text::Tokenizer& tok, SessionOptions opt,
                         SessionPoolOptions pool)
    : engine_(&engine), session_(engine, tok, opt), pool_(pool) {}

Result<ReplayStats> SessionPool::activate(const std::string& name) {
    if (name == active_) return ReplayStats{};
    Engine& e = *engine_;
    // Park the live context (an empty one is simply forgotten).
    if (e.context_length() > 0) {
        auto p = park_context(e);
        if (!p) return std::unexpected(p.error());
        log_info("session pool: parked '{}' ({} tokens, {:.2f} MB packed)", active_,
                 p->tokens.size(), p->bytes() / 1e6);
        parked_[active_] = std::move(*p);
        lru_.remove(active_);
        lru_.push_front(active_);
        if (!pool_.disk.dir.empty()) {
            if (auto sr = save_parked_context(parked_[active_], pool_.disk, active_); !sr)
                log_info("session pool: disk save '{}' failed: {}", active_, sr.error().str());
        }
    }
    ReplayStats st;
    auto it = parked_.find(name);
    if (it == parked_.end() && !pool_.disk.dir.empty()) {
        auto loaded = load_parked_context(pool_.disk, name);
        if (loaded) {
            log_info("session pool: loaded '{}' from disk ({} tokens, {:.2f} MB packed)",
                     name, loaded->tokens.size(), loaded->bytes() / 1e6);
            parked_[name] = std::move(*loaded);
            lru_.remove(name);
            lru_.push_front(name);
            it = parked_.find(name);
        } else if (loaded.error().code != Err::NotFound) {
            log_info("session pool: disk load '{}' failed: {}", name, loaded.error().str());
        }
    }
    if (it == parked_.end()) {
        e.reset_context();
    } else {
        auto r = restore_context(e, it->second, session_.options().replay);
        if (!r) {
            e.reset_context();
            return std::unexpected(r.error());
        }
        st = *r;
        log_info("session pool: restored '{}' ({} tokens), replayed {} positions in {:.0f} ms", name,
                 it->second.tokens.size(), st.plan.steps(), st.ms);
        parked_.erase(it);
        lru_.remove(name);
    }
    active_ = name;
    enforce_budget();
    return st;
}

bool SessionPool::drop(const std::string& name) {
    if (name == active_) {
        engine_->reset_context();
        drop_parked_context(pool_.disk, name);
        return true;
    }
    lru_.remove(name);
    const bool erased = parked_.erase(name) > 0;
    if (erased) drop_parked_context(pool_.disk, name);
    return erased;
}

std::vector<SessionPool::Info> SessionPool::list() const {
    std::vector<Info> out;
    out.push_back(Info{active_, engine_->context_length(), 0, true});
    for (const std::string& n : lru_) {
        const auto it = parked_.find(n);
        if (it != parked_.end())
            out.push_back(Info{n, static_cast<uint32_t>(it->second.tokens.size()), it->second.bytes(), false});
    }
    return out;
}

void SessionPool::enforce_budget() {
    auto total = [&] {
        uint64_t b = 0;
        for (const auto& [n, p] : parked_) b += p.bytes();
        return b;
    };
    while (!lru_.empty() && (parked_.size() > pool_.max_parked || total() > pool_.max_parked_bytes)) {
        const std::string victim = lru_.back();
        if (!pool_.disk.dir.empty()) {
            if (auto sr = save_parked_context(parked_[victim], pool_.disk, victim); !sr)
                log_info("session pool: disk save '{}' failed: {}", victim, sr.error().str());
        }
        lru_.pop_back();
        parked_.erase(victim);
        ++evicted_;
        log_info("session pool: dropped least recently used session '{}'", victim);
    }
}

}  // namespace deepmoe::runtime

// --- disk session/prefix cache (design §11.4, Track R2) ----------------------
// Appended as a second block in the same namespace so the existing translation
// unit stays untouched. See session.h for the contract.
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <utility>

namespace deepmoe::runtime {
namespace {

uint64_t fnv1a_tag(std::string_view s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

std::string session_file_name(const std::string& name) {
    std::string out;
    out.reserve(name.size() + 4);
    for (unsigned char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        out.push_back(ok ? static_cast<char>(c) : '_');
    }
    if (out.empty()) out = "default";
    if (out.size() > 96) out.resize(96);
    return out + ".pkv";
}

void wr_u32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); }
void wr_u64(std::ofstream& f, uint64_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); }
bool rd_u32(std::ifstream& f, uint32_t& v) { return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof v)); }
bool rd_u64(std::ifstream& f, uint64_t& v) { return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof v)); }

template <class T>
void wr_vec(std::ofstream& f, const std::vector<T>& v) {
    wr_u64(f, static_cast<uint64_t>(v.size()) * sizeof(T));
    if (!v.empty())
        f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template <class T>
bool rd_vec(std::ifstream& f, std::vector<T>& v, uint64_t cap) {
    uint64_t bytes = 0;
    if (!rd_u64(f, bytes) || bytes > cap || bytes % sizeof(T) != 0) return false;
    v.resize(static_cast<size_t>(bytes / sizeof(T)));
    return v.empty() || static_cast<bool>(f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes)));
}

constexpr char     kKvMagic[8]   = {'D','M','O','E','K','V','0','1'};
constexpr char     kKvEnd[8]     = {'K','V','E','N','D','0','0','1'};
constexpr uint32_t kKvVersion    = 1;
constexpr uint64_t kKvVectorCap  = 1ull << 34;

void enforce_disk_budget(const KvDiskOptions& opt) {
    namespace fs = std::filesystem;
    if (opt.max_bytes == 0) return;
    std::error_code ec;
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    uint64_t total = 0;
    for (fs::directory_iterator it(opt.dir, ec), end; !ec && it != end; it.increment(ec)) {
        const fs::path& p = it->path();
        if (p.extension() != ".pkv") continue;
        std::error_code e2;
        const auto sz = fs::file_size(p, e2);
        if (e2) continue;
        total += sz;
        files.emplace_back(fs::last_write_time(p, e2), p);
    }
    if (total <= opt.max_bytes) return;
    std::sort(files.begin(), files.end());
    for (const auto& [t, p] : files) {
        if (total <= opt.max_bytes) break;
        std::error_code e2;
        const auto sz = fs::file_size(p, e2);
        if (fs::remove(p, e2)) total -= (e2 ? 0 : sz);
    }
}

}  // namespace
Result<void> save_parked_context(const ParkedContext& p, const KvDiskOptions& opt,
                                 const std::string& name) {
    namespace fs = std::filesystem;
    if (opt.dir.empty()) return fail(Err::InvalidArgument, "kv disk dir is empty");
    std::error_code ec;
    fs::create_directories(opt.dir, ec);
    if (ec) return fail(Err::Io, "kvdisk mkdir " + opt.dir, static_cast<uint32_t>(ec.value()));
    const fs::path final_path = fs::path(opt.dir) / session_file_name(name);
    const fs::path tmp_path   = final_path.string() + ".tmp";
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) return fail(Err::Io, "kvdisk open " + tmp_path.string());
    f.write(kKvMagic, sizeof kKvMagic);
    wr_u32(f, kKvVersion);
    wr_u64(f, fnv1a_tag(opt.model_tag));
    wr_u32(f, static_cast<uint32_t>(p.tokens.size()));
    if (!p.tokens.empty())
        f.write(reinterpret_cast<const char*>(p.tokens.data()),
                static_cast<std::streamsize>(p.tokens.size() * sizeof(uint32_t)));
    wr_u32(f, p.kv.positions);
    wr_u32(f, static_cast<uint32_t>(p.kv.planes.size()));
    for (const KvPackedPlane& pl : p.kv.planes) {
        wr_u32(f, pl.layer);
        wr_u32(f, pl.ratio);
        wr_u32(f, pl.rows);
        wr_vec(f, pl.cmp_fp4);
        wr_vec(f, pl.cmp_scale);
        wr_vec(f, pl.key_fp4);
        wr_vec(f, pl.key_scale);
        wr_vec(f, pl.raw_rows);
        wr_vec(f, pl.raw_cmp);
        wr_vec(f, pl.raw_key);
        wr_vec(f, pl.carry_kv);
        wr_vec(f, pl.carry_score);
    }
    f.write(kKvEnd, sizeof kKvEnd);
    f.flush();
    if (!f.good()) {
        f.close();
        std::error_code rm;
        fs::remove(tmp_path, rm);
        return fail(Err::Io, "kvdisk write " + tmp_path.string());
    }
    f.close();
    fs::remove(final_path, ec);   // Windows rename does not replace an existing file
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        std::error_code rm;
        fs::remove(tmp_path, rm);
        return fail(Err::Io, "kvdisk rename " + final_path.string(),
                    static_cast<uint32_t>(ec.value()));
    }
    enforce_disk_budget(opt);
    return {};
}
Result<ParkedContext> load_parked_context(const KvDiskOptions& opt, const std::string& name) {
    namespace fs = std::filesystem;
    if (opt.dir.empty()) return fail(Err::InvalidArgument, "kv disk dir is empty");
    const fs::path path = fs::path(opt.dir) / session_file_name(name);
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(Err::NotFound, "kvdisk miss " + path.string());
    char magic[8];
    if (!f.read(magic, sizeof magic) || std::memcmp(magic, kKvMagic, sizeof magic) != 0)
        return fail(Err::Corrupt, "kvdisk bad magic " + path.string());
    uint32_t ver = 0;
    if (!rd_u32(f, ver) || ver != kKvVersion)
        return fail(Err::Corrupt, "kvdisk bad version " + path.string());
    uint64_t fp = 0;
    if (!rd_u64(f, fp)) return fail(Err::Corrupt, "kvdisk bad tag " + path.string());
    if (!opt.model_tag.empty() && fp != fnv1a_tag(opt.model_tag))
        return fail(Err::NotFound, "kvdisk model mismatch " + path.string());
    uint32_t nt = 0;
    if (!rd_u32(f, nt) || nt > (1u << 24))
        return fail(Err::Corrupt, "kvdisk bad token count " + path.string());
    ParkedContext p;
    p.tokens.resize(nt);
    if (nt && !f.read(reinterpret_cast<char*>(p.tokens.data()),
                      static_cast<std::streamsize>(nt * sizeof(uint32_t))))
        return fail(Err::Corrupt, "kvdisk short tokens " + path.string());
    uint32_t planes = 0;
    if (!rd_u32(f, p.kv.positions) || !rd_u32(f, planes) || planes > 64)
        return fail(Err::Corrupt, "kvdisk bad plane count " + path.string());
    p.kv.planes.resize(planes);
    for (KvPackedPlane& pl : p.kv.planes) {
        if (!rd_u32(f, pl.layer) || !rd_u32(f, pl.ratio) || !rd_u32(f, pl.rows))
            return fail(Err::Corrupt, "kvdisk short plane header " + path.string());
        if (!rd_vec(f, pl.cmp_fp4, kKvVectorCap) || !rd_vec(f, pl.cmp_scale, kKvVectorCap) ||
            !rd_vec(f, pl.key_fp4, kKvVectorCap) || !rd_vec(f, pl.key_scale, kKvVectorCap) ||
            !rd_vec(f, pl.raw_rows, kKvVectorCap) || !rd_vec(f, pl.raw_cmp, kKvVectorCap) ||
            !rd_vec(f, pl.raw_key, kKvVectorCap) || !rd_vec(f, pl.carry_kv, kKvVectorCap) ||
            !rd_vec(f, pl.carry_score, kKvVectorCap))
            return fail(Err::Corrupt, "kvdisk short plane " + path.string());
    }
    char end[8];
    if (!f.read(end, sizeof end) || std::memcmp(end, kKvEnd, sizeof end) != 0)
        return fail(Err::Corrupt, "kvdisk short trailer " + path.string());
    return p;
}

bool drop_parked_context(const KvDiskOptions& opt, const std::string& name) {
    if (opt.dir.empty()) return false;
    std::error_code ec;
    return std::filesystem::remove(std::filesystem::path(opt.dir) / session_file_name(name), ec);
}

}  // namespace deepmoe::runtime
