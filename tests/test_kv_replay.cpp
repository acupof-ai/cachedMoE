// Track R2 (docs/p4_kv_ux.md): the window ring is never stored. A context comes
// back -- after a rollback, or from a parked session -- from its non-SWA state
// (compressed rows, index keys, the compressor carry, the token ids) plus a
// replay of the last <= 128 positions through the decode path. These cases
// check that, on the real checkpoint:
//
//   kv_replay.l3_64      the L3 export (64 tokens): rollback inside the window is
//                        bit-exact against the continuous run; park + restore
//                        from our own slow prefill rebuilds the ring bit-exactly;
//                        park + restore from the reference's state, 8 + 8 tokens;
//                        the Session's rollback, cancel and named sessions.
//   kv_replay.longctx    Track M's 4K export: the ring rebuilt by a 127-token
//                        replay against the reference's ring, 8 teacher-forced
//                        and 8 free-running tokens; a 7-token rollback; a
//                        133-token rollback (full 128-token replay) followed by
//                        re-feeding the dropped prompt and decoding.
//
// Needs DEEPMOE_MODEL_DIR, tests/data/l3, and DEEPMOE_LONGCTX_DIR (default
// <repo>/traces/longctx) for the second case.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include "core/config.h"
#include "cpu/dequant.h"
#include "runtime/engine.h"
#include "runtime/session.h"
#include "tests/l1_golden.h"
#include "tests/test_framework.h"
#include "text/tokenizer.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::testing;

namespace {

bool exists(const std::string& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}

std::string longctx_root() {
    if (const char* e = std::getenv("DEEPMOE_LONGCTX_DIR")) return e;
    return std::string(DEEPMOE_TEST_DATA_DIR) + "/../../traces/longctx";
}

bool bring_up(runtime::Engine& e, const std::string& state_dir, std::string& why) {
    RuntimeConfig rc;
    rc.model_dir = model_dir();
    rc.cache.budget_bytes = 8ull << 30;
    rc.cache.slots_per_slab = 100;
    if (auto r = e.init(rc); !r) { why = r.error().str(); return false; }
    if (auto r = e.init_gpu(); !r) { why = r.error().str(); return false; }
    if (auto r = e.load_decode_state(state_dir); !r) { why = r.error().str(); return false; }
    if (!e.produce_ced()) { why = "export has no prefill CED state"; return false; }
    return true;
}

// Slot `s` of layer `L`'s ring, dequantised.
std::vector<float> ring_slot(const runtime::Engine& e, uint32_t L, uint32_t s) {
    auto v = e.kv().layer(L);
    const uint32_t d = e.model().text.head_dim;
    std::vector<float> out(d);
    for (uint32_t i = 0; i < d; ++i)
        out[i] = cpu::fp8_e4m3_to_float(v->win_val_host[size_t(s) * d + i]) *
                 cpu::e8m0_to_float(v->win_scale_host[size_t(s) * (d / 32) + i / 32]);
    return out;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
    return (aa == 0 || bb == 0) ? (aa == bb ? 1.0 : 0.0) : ab / std::sqrt(aa * bb);
}

// The raw ring bytes of every layer for the slots holding positions [lo, hi).
std::vector<uint8_t> ring_bytes(const runtime::Engine& e, uint32_t lo, uint32_t hi) {
    const TextConfig& c = e.model().text;
    std::vector<uint8_t> out;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        auto v = e.kv().layer(L);
        for (uint32_t p = lo; p < hi; ++p) {
            const uint32_t s = p % c.sliding_window;
            out.insert(out.end(), v->win_val_host + size_t(s) * c.head_dim,
                       v->win_val_host + size_t(s + 1) * c.head_dim);
            out.insert(out.end(), v->win_scale_host + size_t(s) * (c.head_dim / 32),
                       v->win_scale_host + size_t(s + 1) * (c.head_dim / 32));
        }
    }
    return out;
}

bool same_packed(const runtime::KvPacked& a, const runtime::KvPacked& b, bool carry) {
    if (a.positions != b.positions || a.planes.size() != b.planes.size()) return false;
    for (size_t i = 0; i < a.planes.size(); ++i) {
        const auto& x = a.planes[i];
        const auto& y = b.planes[i];
        if (x.cmp_fp4 != y.cmp_fp4 || x.cmp_scale != y.cmp_scale || x.key_fp4 != y.key_fp4 ||
            x.key_scale != y.key_scale || x.raw_rows != y.raw_rows || x.raw_cmp != y.raw_cmp ||
            x.raw_key != y.raw_key)
            return false;
        if (carry && (x.carry_kv != y.carry_kv || x.carry_score != y.carry_score)) return false;
    }
    return true;
}

struct Step { uint32_t token; float top1, top2; };

// Ring agreement against the export's prefill ring, over the slots holding
// positions [lo, hi): worst per-layer mean cosine and worst single slot.
struct RingCmp { double worst_layer = 1.0, worst_slot = 1.0, mean = 0.0; uint32_t worst_l = 0; };
RingCmp ring_vs_export(const runtime::Engine& e, const runtime::DecodeState& st, uint32_t lo, uint32_t hi) {
    const TextConfig& c = e.model().text;
    RingCmp r;
    double sum = 0;
    for (uint32_t L = 0; L < c.num_hidden_layers; ++L) {
        const runtime::StateTensor* w = st.tensor(0, std::format("L{:02d}.win_kv", L));
        if (!w) continue;
        double lsum = 0;
        for (uint32_t p = lo; p < hi; ++p) {
            const uint32_t s = p % c.sliding_window;
            std::vector<float> ref(w->f.begin() + size_t(s) * c.head_dim, w->f.begin() + size_t(s + 1) * c.head_dim);
            const double cs = cosine(ring_slot(e, L, s), ref);
            lsum += cs;
            r.worst_slot = std::min(r.worst_slot, cs);
        }
        const double lm = lsum / double(hi - lo);
        sum += lm;
        if (lm < r.worst_layer) { r.worst_layer = lm; r.worst_l = L; }
    }
    r.mean = sum / c.num_hidden_layers;
    return r;
}

}  // namespace

DEEPMOE_TEST(kv_replay, l3_64) {
    if (skip_without_model("kv_replay")) return;
    const std::string l3 = std::string(DEEPMOE_TEST_DATA_DIR) + "/l3";
    if (!exists(l3 + "/index.json")) { std::printf("      SKIP kv_replay: no L3 export\n"); return; }
    runtime::Engine e;
    std::string why;
    if (!bring_up(e, l3, why)) { std::printf("      SKIP kv_replay: %s\n", why.c_str()); return; }
    const runtime::DecodeState* st = e.decode_state();
    const uint32_t base = st->decode_pos();
    const std::vector<uint32_t>& ref = st->greedy_tokens();
    const uint32_t S = st->steps();
    std::printf("    KV store %.2f MB for %u positions (capacity %u)\n", e.kv().bytes() / 1e6,
                e.max_context(), e.kv().capacity());

    // --- (1) continuous, teacher-forced from the export state ---------------
    std::vector<Step> cont;
    std::vector<uint8_t> ring68;
    runtime::KvPacked rows68;
    for (uint32_t s = 0; s < S; ++s) {
        auto r = e.decode_step(ref[s], base + s, -1);
        REQUIRE_OK(r);
        cont.push_back({r->token, r->top1, r->top2});
        if (base + s + 1 == base + 4) {
            ring68 = ring_bytes(e, 0, base + 4);
            auto p = e.kv_store().pack(base + 4);
            REQUIRE_OK(p);
            rows68 = std::move(*p);
        }
    }
    uint32_t cont_match = 0;
    for (uint32_t s = 0; s < S; ++s) cont_match += cont[s].token == ref[s + 1];
    std::printf("    (1) continuous from the export: %u/%u teacher-forced\n", cont_match, S);

    // --- (2) rollback inside the window: bit-exact ---------------------------
    for (uint32_t keep : {base + 4, base + 2}) {
        auto plan = runtime::plan_rollback(e, keep);
        auto rb = runtime::rollback_context(e, keep);
        REQUIRE_OK(rb);
        std::printf("    (2) rollback %u -> %u: %u replayed positions\n", base + S, keep, plan.steps());
        CHECK_EQ(plan.steps(), 0u);
        CHECK_EQ(e.context_length(), keep);
        if (keep == base + 4) {
            CHECK(ring_bytes(e, 0, keep) == ring68);
            auto p = e.kv_store().pack(keep);
            REQUIRE_OK(p);
            CHECK(same_packed(*p, rows68, /*carry=*/false));
        }
        uint32_t exact = 0;
        for (uint32_t pos = keep; pos < base + S; ++pos) {
            const uint32_t s = pos - base;
            auto r = e.decode_step(ref[s], pos, -1);
            REQUIRE_OK(r);
            exact += (r->token == cont[s].token && r->top1 == cont[s].top1 && r->top2 == cont[s].top2);
        }
        std::printf("        re-decoded %u steps: %u bit-identical logits (top1/top2) to the continuous run\n",
                    base + S - keep, exact);
        CHECK_EQ(exact, base + S - keep);
    }
    // An odd rollback point is refused; the Session rounds down.
    CHECK(!runtime::rollback_context(e, base + 3).has_value());

    // --- (3) park + restore on the reference's state -------------------------
    {
        REQUIRE_OK(e.reseed_decode_state());
        e.kv_store().resolve_ring(base);
        auto parked = runtime::park_context(e);
        REQUIRE_OK(parked);
        auto rs = runtime::restore_context(e, *parked);
        REQUIRE_OK(rs);
        const RingCmp rc = ring_vs_export(e, *st, rs->plan.first, rs->plan.end);
        std::printf("    (3) restore of the export state: replayed [%u, %u) in %.0f ms; ring vs the "
                    "reference's: mean cos %.5f, worst layer %.5f (L%u), worst slot %.5f\n",
                    rs->plan.first, rs->plan.end, rs->ms, rc.mean, rc.worst_layer, rc.worst_l, rc.worst_slot);
        auto after = e.kv_store().pack(base);
        REQUIRE_OK(after);
        CHECK(same_packed(*after, parked->kv, true));
        uint32_t tf = 0;
        for (uint32_t s = 0; s < S; ++s) {
            auto r = e.decode_step(ref[s], base + s, -1);
            REQUIRE_OK(r);
            tf += r->token == ref[s + 1];
        }
        REQUIRE_OK(runtime::restore_context(e, *parked));
        uint32_t fr = 0, tok = ref[0];
        bool div = false;
        std::string margins;
        for (uint32_t s = 0; s < S; ++s) {
            auto r = e.decode_step(tok, base + s, -1);
            REQUIRE_OK(r);
            if (!div && r->token == ref[s + 1]) ++fr; else div = true;
            margins += std::format(" {:.2f}", r->margin());
            tok = r->token;
        }
        std::printf("        rebuilt state: %u/%u teacher-forced, %u/%u free-running (margins%s)\n", tf, S, fr, S,
                    margins.c_str());
        CHECK(tf * 8 >= S * 7);
        CHECK(fr >= 1);
    }

    // --- (4) park + restore on our own state: the replay is bit-exact --------
    {
        auto pre = e.slow_prefill(st->prompt_ids());
        REQUIRE_OK(pre);
        const std::vector<uint8_t> ring0 = ring_bytes(e, 0, base);
        auto rows0 = e.kv_store().pack(base);
        REQUIRE_OK(rows0);
        auto parked = runtime::park_context(e);
        REQUIRE_OK(parked);
        std::vector<Step> own;
        for (uint32_t s = 0; s < S; ++s) {
            auto r = e.decode_step(ref[s], base + s, -1);
            REQUIRE_OK(r);
            own.push_back({r->token, r->top1, r->top2});
        }
        auto rs = runtime::restore_context(e, *parked);
        REQUIRE_OK(rs);
        std::printf("    (4) park (%.3f MB packed, %u raw rows) + restore of our slow prefill: replayed "
                    "[%u, %u) in %.0f ms (%.1f ms/token)\n", parked->bytes() / 1e6, parked->kv.raw_rows(),
                    rs->plan.first, rs->plan.end, rs->ms, rs->ms / std::max(1u, rs->plan.steps()));
        CHECK_EQ(rs->plan.first, 0u);
        CHECK_EQ(rs->plan.end, base);
        CHECK_EQ(parked->kv.raw_rows(), 0u);
        for (const auto& pl : parked->kv.planes)
            for (size_t i = 0; i < pl.raw_rows.size() && i < 3; ++i) {
                const uint32_t cd = 512, kd = 128;
                std::vector<uint8_t> nb(256), sb(32);
                const bool okc = runtime::pack_fp4_row(pl.raw_cmp.data() + i * cd, cd, 16, false, nb.data(), sb.data());
                const bool okk = runtime::pack_fp4_row(pl.raw_key.data() + i * kd, kd, 32, true, nb.data(), sb.data());
                std::printf("        raw row: layer %u row %u, compressed %s, key %s\n", pl.layer, pl.raw_rows[i],
                            okc ? "packs" : "does NOT pack", okk ? "packs" : "does NOT pack");
                for (uint32_t b0 = 0; b0 < (okc ? kd : cd); b0 += (okc ? 32 : 16)) {
                    std::vector<uint8_t> n2(16), s2(2);
                    const bool ok = okc ? runtime::pack_fp4_row(pl.raw_key.data() + i * kd + b0, 32, 32, true, n2.data(), s2.data())
                                        : runtime::pack_fp4_row(pl.raw_cmp.data() + i * cd + b0, 16, 16, false, n2.data(), s2.data());
                    if (ok) continue;
                    std::string vals;
                    const uint16_t* src = okc ? pl.raw_key.data() + i * kd + b0 : pl.raw_cmp.data() + i * cd + b0;
                    for (uint32_t j = 0; j < (okc ? 32u : 16u); ++j) vals += std::format(" {:04x}={:g}", src[j], cpu::bf16_to_float(src[j]));
                    std::printf("          block at %u:%s\n", b0, vals.c_str());
                    break;
                }
            }
        const bool ring_same = ring_bytes(e, 0, base) == ring0;
        auto rows1 = e.kv_store().pack(base);
        REQUIRE_OK(rows1);
        const bool rows_same = same_packed(*rows1, *rows0, /*carry=*/true);
        std::printf("        rebuilt ring %s, compressed rows + keys + carry %s\n",
                    ring_same ? "bit-identical" : "DIFFERS", rows_same ? "bit-identical" : "DIFFER");
        CHECK(ring_same);
        CHECK(rows_same);
        uint32_t exact = 0;
        for (uint32_t s = 0; s < S; ++s) {
            auto r = e.decode_step(ref[s], base + s, -1);
            REQUIRE_OK(r);
            exact += (r->token == own[s].token && r->top1 == own[s].top1 && r->top2 == own[s].top2);
        }
        std::printf("        next %u steps: %u bit-identical to the continuous run\n", S, exact);
        CHECK_EQ(exact, S);
    }

    // --- (5) the Session: rollback, cancel, named sessions --------------------
    auto tok = text::Tokenizer::load(std::string(model_dir()) + "/tokenizer.json");
    REQUIRE_OK(tok);
    e.reset_context();
    runtime::SessionOptions so;
    so.progress_every = 8;
    runtime::SessionPool pool(e, *tok, so);
    runtime::GenerateRequest req;
    req.prompt_ids = st->prompt_ids();
    req.max_tokens = 6;
    req.sampling.temperature = 0.0f;
    auto g1 = pool.live().generate(req, {});
    REQUIRE_OK(g1);
    std::printf("    (5) turn 1: %u prompt, %u generated, context %u\n", g1->prompt_tokens, g1->generated,
                g1->context_after);
    // A prompt that diverges at 61: rolls back to 60, no replay (nothing wrapped).
    runtime::GenerateRequest req2 = req;
    req2.prompt_ids.resize(61);
    req2.prompt_ids.push_back(1337);
    req2.prompt_ids.push_back(42);
    auto g2 = pool.live().generate(req2, {});
    REQUIRE_OK(g2);
    std::printf("        turn 2 (diverges at 61): reused %u, dropped %u, replay %u steps, prefilled %u, "
                "generated %u\n", g2->reused_tokens, g2->rollback_dropped, g2->replay_steps,
                g2->prefilled_tokens, g2->generated);
    CHECK_EQ(g2->reused_tokens, 60u);
    CHECK(g2->rollback_dropped > 0);
    CHECK_EQ(g2->prefilled_tokens, 3u);
    // Cancel after two tokens: the store holds prompt + first token.
    runtime::GenerateRequest req3 = req2;
    req3.max_tokens = 50;
    uint32_t seen = 0;
    req3.cancel = [&] { return seen >= 2; };
    auto g3 = pool.live().generate(req3, [&](const runtime::TokenEvent&) { ++seen; });
    REQUIRE_OK(g3);
    std::printf("        turn 3 cancelled: finish %s, generated %u, context %u\n", g3->finish.c_str(),
                g3->generated, g3->context_after);
    CHECK_EQ(g3->finish, std::string("cancel"));
    CHECK_EQ(g3->generated, 2u);
    CHECK_EQ(g3->context_after, uint32_t(req3.prompt_ids.size() + 1));
    const std::vector<uint32_t> def_hist = e.history();
    // A second named session, then back.
    auto a1 = pool.activate("other");
    REQUIRE_OK(a1);
    CHECK_EQ(e.context_length(), 0u);
    runtime::GenerateRequest ro = req;
    ro.prompt_ids.resize(20);
    auto go = pool.live().generate(ro, {});
    REQUIRE_OK(go);
    auto a2 = pool.activate("default");
    REQUIRE_OK(a2);
    std::printf("        switched to 'other' (%u tokens) and back: 'default' restored with %u tokens, "
                "replayed %u positions in %.0f ms\n", go->context_after, e.context_length(),
                a2->plan.steps(), a2->ms);
    CHECK(e.history() == def_hist);
    CHECK_EQ(a2->plan.steps(), uint32_t(def_hist.size()));
    runtime::GenerateRequest rc = req3;
    rc.cancel = {};
    rc.max_tokens = 2;
    rc.prompt_ids = def_hist;
    rc.prompt_ids.push_back(7);
    auto gc = pool.live().generate(rc, {});
    REQUIRE_OK(gc);
    CHECK_EQ(gc->reused_tokens, uint32_t(def_hist.size()));
    CHECK_EQ(gc->prefilled_tokens, 1u);
    const auto infos = pool.list();
    CHECK_EQ(infos.size(), size_t(2));

    // --- (6) thinking mode, drop_thinking=True: the rollback chat.py takes on
    //         EVERY turn -------------------------------------------------------
    //
    // The official thinking-mode template drops an earlier turn's reasoning from
    // the prompt (tools/chat.py), so turn N+1's prompt is NOT an extension of
    // what the KV store holds: it agrees up to where the reply began and then
    // differs. That is a prefix match plus a truncation, and when the divergence
    // point is odd it lands inside a ratio-2 group whose carry the store does not
    // keep per group -- which is exactly what the even floor of runtime/session.h
    // is for. The check is end to end: the turn taken by rollback has to produce
    // what the same prompt produces from an empty store, token for token, and
    // leave the same compressed rows, index keys and carry behind.
    {
        pool.drop("other");
        e.reset_context();
        runtime::GenerateRequest turn1 = req;
        turn1.max_tokens = 6;
        auto s1 = pool.live().generate(turn1, {});
        REQUIRE_OK(s1);

        runtime::GenerateRequest turn2;
        turn2.max_tokens = 6;
        turn2.sampling.temperature = 0.0f;
        const uint32_t cut = 45;              // odd: a half-filled ratio-2 group
        turn2.prompt_ids.assign(req.prompt_ids.begin(), req.prompt_ids.begin() + cut);
        turn2.prompt_ids.push_back(1337);
        turn2.prompt_ids.push_back(42);
        std::vector<uint32_t> by_rollback;
        auto s2 = pool.live().generate(
            turn2, [&](const runtime::TokenEvent& ev) { by_rollback.push_back(ev.id); });
        REQUIRE_OK(s2);
        std::printf("    (6) drop_thinking turn: history %u, prompt diverges at %u -> reused %u, "
                    "dropped %u, replayed %u in %.0f ms, prefilled %u\n",
                    s1->context_after, cut, s2->reused_tokens, s2->rollback_dropped,
                    s2->replay_steps, s2->replay_ms, s2->prefilled_tokens);
        CHECK_EQ(s2->reused_tokens, cut - 1);          // 45 rounded down to even
        CHECK(s2->rollback_dropped > 0);
        CHECK_EQ(s2->prefilled_tokens, uint32_t(turn2.prompt_ids.size()) - (cut - 1));
        auto packed_rb = e.kv_store().pack(e.context_length());
        REQUIRE_OK(packed_rb);
        const uint32_t ctx_rb = e.context_length();

        runtime::GenerateRequest fresh = turn2;
        fresh.reuse = false;                           // the same turn from nothing
        std::vector<uint32_t> by_prefill;
        auto s3 = pool.live().generate(
            fresh, [&](const runtime::TokenEvent& ev) { by_prefill.push_back(ev.id); });
        REQUIRE_OK(s3);
        CHECK_EQ(s3->reused_tokens, 0u);
        CHECK_EQ(s3->context_after, ctx_rb);
        auto packed_fresh = e.kv_store().pack(e.context_length());
        REQUIRE_OK(packed_fresh);
        const bool same_tokens = by_rollback == by_prefill;
        const bool same_rows = same_packed(*packed_rb, *packed_fresh, /*carry=*/true);
        std::printf("        vs the same turn from an empty store: tokens %s, compressed rows + "
                    "keys + carry %s\n", same_tokens ? "identical" : "DIFFER",
                    same_rows ? "bit-identical" : "DIFFER");
        CHECK(same_tokens);
        CHECK(same_rows);
    }
}

DEEPMOE_TEST(kv_replay, longctx) {
    if (skip_without_model("kv_replay")) return;
    const std::string dir = longctx_root() + "/ctx4k";
    if (!exists(dir + "/index.json")) {
        std::printf("      SKIP kv_replay: no 4K export at %s (DEEPMOE_LONGCTX_DIR)\n", dir.c_str());
        return;
    }
    runtime::Engine e;
    std::string why;
    if (!bring_up(e, dir, why)) { std::printf("      SKIP kv_replay: %s\n", why.c_str()); return; }
    const runtime::DecodeState* st = e.decode_state();
    const uint32_t N = st->decode_pos();
    const std::vector<uint32_t>& ref = st->greedy_tokens();
    const uint32_t S = st->steps();
    std::printf("    4K export: N = %u, KV store %.1f MB (capacity %u positions)\n", N, e.kv().bytes() / 1e6,
                e.kv().capacity());

    // --- (1) continuous from the reference's state ---------------------------
    std::vector<Step> cont;
    {
        uint32_t tok = ref[0], fr = 0;
        bool div = false;
        std::string m;
        for (uint32_t s = 0; s < S; ++s) {
            auto r = e.decode_step(tok, N + s, -1);
            REQUIRE_OK(r);
            cont.push_back({r->token, r->top1, r->top2});
            if (!div && r->token == ref[s + 1]) ++fr; else div = true;
            m += std::format(" {:.2f}", r->margin());
            tok = r->token;
        }
        std::printf("    (1) continuous: %u/%u free-running (margins%s)\n", fr, S, m.c_str());
    }

    // --- (2) park + restore: the ring rebuilt by a <= 128-token replay --------
    REQUIRE_OK(e.reseed_decode_state());
    e.kv_store().resolve_ring(N);
    auto parked = runtime::park_context(e);
    REQUIRE_OK(parked);
    std::printf("    (2) parked %u tokens: %.2f MB (%u raw rows; the live bf16 planes are %.2f MB)\n", N,
                parked->bytes() / 1e6, parked->kv.raw_rows(),
                (e.kv().config().compressed_bytes(N) + e.kv().config().index_key_bytes(N)) / 1e6);
    CHECK_EQ(parked->kv.raw_rows(), 0u);
    auto rs = runtime::restore_context(e, *parked);
    REQUIRE_OK(rs);
    CHECK(rs->plan.steps() <= 128u);
    CHECK_EQ(rs->plan.end, N);
    const RingCmp rc = ring_vs_export(e, *st, rs->plan.first, rs->plan.end);
    std::printf("        replayed [%u, %u): %u steps in %.1f s (%.1f ms/token; row backup %.1f KB, %.1f ms); "
                "ring vs the reference's: mean cos %.5f, worst layer %.5f (L%u), worst slot %.5f\n",
                rs->plan.first, rs->plan.end, rs->plan.steps(), rs->ms / 1e3,
                rs->ms / std::max(1u, rs->plan.steps()), rs->backup_bytes / 1e3, rs->backup_ms,
                rc.mean, rc.worst_layer, rc.worst_l, rc.worst_slot);
    {
        auto again = e.kv_store().pack(N);
        REQUIRE_OK(again);
        CHECK(same_packed(*again, parked->kv, true));
    }
    uint32_t tf = 0;
    std::string tfm;
    for (uint32_t s = 0; s < S; ++s) {
        auto r = e.decode_step(ref[s], N + s, -1);
        REQUIRE_OK(r);
        tf += r->token == ref[s + 1];
        tfm += std::format(" {:.2f}", r->margin());
    }
    std::printf("        rebuilt: %u/%u teacher-forced (margins%s)\n", tf, S, tfm.c_str());
    REQUIRE_OK(runtime::restore_context(e, *parked));
    uint32_t fr = 0, tok = ref[0];
    bool div = false;
    std::string frm;
    std::vector<uint32_t> gen;
    for (uint32_t s = 0; s < S; ++s) {
        auto r = e.decode_step(tok, N + s, -1);
        REQUIRE_OK(r);
        if (!div && r->token == ref[s + 1]) ++fr; else div = true;
        frm += std::format(" {:.2f}", r->margin());
        tok = r->token;
        gen.push_back(r->token);
    }
    std::printf("        rebuilt: %u/%u free-running (margins%s)\n", fr, S, frm.c_str());
    CHECK_EQ(tf, S);
    CHECK_EQ(fr, S);

    // --- (3) a 7-token rollback: only the overwritten slots are replayed ------
    {
        std::vector<uint32_t> hist = e.history();   // N prompt + S - 1 generated... as fed
        const uint32_t n = e.context_length();
        const uint32_t keep = (N + 1) & ~1u;
        auto plan = runtime::plan_rollback(e, keep);
        auto rb = runtime::rollback_context(e, keep);
        REQUIRE_OK(rb);
        std::printf("    (3) rollback %u -> %u: replayed [%u, %u) (%u steps) in %.0f ms\n", n, keep,
                    rb->plan.first, rb->plan.end, rb->plan.steps(), rb->ms);
        CHECK_EQ(plan.steps(), rb->plan.steps());
        CHECK(rb->plan.steps() <= n - keep + 1);
        uint32_t same = 0;
        for (uint32_t p = keep; p < n; ++p) {
            auto r = e.decode_step(hist[p], p, -1);
            REQUIRE_OK(r);
            same += r->token == gen[p - N];
        }
        std::printf("        re-decoded %u positions: %u/%u tokens as before the rollback\n", n - keep, same,
                    n - keep);
        CHECK_EQ(same, n - keep);
    }

    // --- (4) a 133-token rollback: full replay, re-feed, decode --------------
    {
        REQUIRE_OK(e.reseed_decode_state());
        const uint32_t keep = 4000;
        auto rb = runtime::rollback_context(e, keep);
        REQUIRE_OK(rb);
        std::printf("    (4) rollback %u -> %u: replayed [%u, %u) (%u steps) in %.1f s\n", N, keep,
                    rb->plan.first, rb->plan.end, rb->plan.steps(), rb->ms / 1e3);
        CHECK_EQ(rb->plan.steps(), 128u);
        const std::vector<uint32_t> tail(st->prompt_ids().begin() + keep, st->prompt_ids().end());
        const TimePoint t0 = Clock::now();
        auto fed = e.feed(tail);
        REQUIRE_OK(fed);
        const double feed_s = std::chrono::duration<double>(Clock::now() - t0).count();
        const RingCmp rc2 = ring_vs_export(e, *st, N - 128, N);
        std::printf("        re-fed %zu prompt tokens in %.1f s; ring at %u vs the reference's: mean cos %.5f, "
                    "worst layer %.5f (L%u); next token %u (reference %u)\n", tail.size(), feed_s, N,
                    rc2.mean, rc2.worst_layer, rc2.worst_l, fed->greedy_token, ref[0]);
        CHECK_EQ(fed->greedy_token, ref[0]);
        uint32_t fr2 = 0, t2 = ref[0];
        bool div2 = false;
        for (uint32_t s = 0; s < S; ++s) {
            auto r = e.decode_step(t2, N + s, -1);
            REQUIRE_OK(r);
            if (!div2 && r->token == ref[s + 1]) ++fr2; else div2 = true;
            t2 = r->token;
        }
        std::printf("        then %u/%u free-running\n", fr2, S);
        CHECK(fr2 >= S - 1);
    }
}


// Pure CPU: the SSD parked-context format round-trips, rejects a model-tag
// mismatch, and drop removes the file. No checkpoint, no GPU. (suite.kvdisk)
#include <filesystem>
DEEPMOE_TEST(kvdisk, roundtrip) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "deepmoe_kvdisk_roundtrip";
    std::error_code ec;
    fs::remove_all(dir, ec);
    runtime::KvDiskOptions opt;
    opt.dir = dir.string();
    opt.model_tag = "unit-test-model";
    runtime::ParkedContext p;
    p.tokens = {1, 2, 3, 4, 5};
    p.kv.positions = 4;
    runtime::KvPackedPlane pl;
    pl.layer = 2; pl.ratio = 2; pl.rows = 2;
    pl.cmp_fp4 = {1, 2, 3, 4};
    pl.cmp_scale = {7, 8};
    pl.key_fp4 = {9, 10};
    pl.key_scale = {11, 12};
    pl.raw_rows = {0, 1};
    pl.raw_cmp = {100, 200};
    pl.raw_key = {300, 400};
    pl.carry_kv = {1.5f, 2.5f};
    pl.carry_score = {-1.0f, 0.25f};
    p.kv.planes.push_back(pl);
    REQUIRE_OK(runtime::save_parked_context(p, opt, "unit/session one"));

    // The on-disk format, byte for byte (docs/p4_kv_ux.md 6). Asserting the
    // exact size is how "the window ring is NEVER written" is checked: there is
    // no room in the file for anything but what is counted here, and a window
    // would add 40 x 128 x 528 B = 2.70 MB whatever the context.
    //   magic "DMOEKV01"        8
    //   version u32             4
    //   FNV-1a of model_tag     8
    //   n_tokens u32 + ids      4 + 4 n
    //   positions u32           4
    //   n_planes u32            4
    //   per plane: layer, ratio, rows (3 x u32) then nine
    //     (u64 byte count + payload) vectors: cmp_fp4, cmp_scale, key_fp4,
    //     key_scale, raw_rows, raw_cmp, raw_key, carry_kv, carry_score
    //   trailer "KVEND001"      8
    uint64_t want = 8 + 4 + 8 + 4 + p.tokens.size() * 4 + 4 + 4 + 8;
    want += 3 * 4 + 9 * 8;
    want += pl.cmp_fp4.size() + pl.cmp_scale.size() + pl.key_fp4.size() + pl.key_scale.size();
    want += pl.raw_rows.size() * 4 + (pl.raw_cmp.size() + pl.raw_key.size()) * 2;
    want += (pl.carry_kv.size() + pl.carry_score.size()) * 4;
    const auto on_disk = fs::file_size(fs::path(opt.dir) / "unit_session_one.pkv", ec);
    CHECK(!ec);
    CHECK_EQ(static_cast<uint64_t>(on_disk), want);

    auto loaded = runtime::load_parked_context(opt, "unit/session one");
    REQUIRE_OK(loaded);
    REQUIRE_EQ(loaded->tokens.size(), p.tokens.size());
    for (size_t i = 0; i < p.tokens.size(); ++i) CHECK_EQ(loaded->tokens[i], p.tokens[i]);
    CHECK_EQ(loaded->kv.positions, p.kv.positions);
    REQUIRE_EQ(loaded->kv.planes.size(), p.kv.planes.size());
    const runtime::KvPackedPlane& q = loaded->kv.planes[0];
    CHECK_EQ(q.layer, pl.layer);
    CHECK_EQ(q.ratio, pl.ratio);
    CHECK_EQ(q.rows, pl.rows);
    CHECK(q.cmp_fp4 == pl.cmp_fp4 && q.key_fp4 == pl.key_fp4 && q.raw_rows == pl.raw_rows);
    CHECK(q.raw_cmp == pl.raw_cmp && q.raw_key == pl.raw_key);
    CHECK(q.carry_kv == pl.carry_kv && q.carry_score == pl.carry_score);
    runtime::KvDiskOptions other = opt;
    other.model_tag = "another-model";
    auto miss = runtime::load_parked_context(other, "unit/session one");
    REQUIRE(!miss);
    CHECK_EQ(static_cast<int>(miss.error().code), static_cast<int>(Err::NotFound));
    CHECK(runtime::drop_parked_context(opt, "unit/session one"));
    CHECK(!runtime::load_parked_context(opt, "unit/session one"));
    fs::remove_all(dir, ec);
}
