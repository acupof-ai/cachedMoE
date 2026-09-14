// cpu/dspark_tree against tools/dspark_tree.py (docs/p3_dspark.md §3, Track K2).
//
// tests/data/dspark/tree_golden.bin holds a few real draft cycles from
// tools/oracle_dspark.py --tree: the anchored top-32 candidates of the five draft
// positions, the gathered Markov rows, the confidence head's hidden state, the
// verify matrix's top-32 rows and the uniforms, plus everything the Python
// reference computed from them for K in {4, 8, 16, 32} x {tail, no tail}. The
// contract is bit-exactness, so every float compare here is ==, and the logq /
// bias arrays are compared through an FNV-1a hash of their bytes.
//
// Needs no model and no GPU: registered as a unit suite.
#include "cpu/dspark_tree.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "cpu/dequant.h"
#include "tests/test_framework.h"

#ifndef DEEPMOE_TEST_DATA_DIR
#define DEEPMOE_TEST_DATA_DIR "tests/data"
#endif

using namespace deepmoe;
using namespace deepmoe::cpu;
using namespace deepmoe::cpu::dspark;

namespace {

constexpr uint32_t kP = kPositions;
constexpr uint32_t kKs[4] = {4, 8, 16, 32};

struct Reader {
    const std::vector<char>& b;
    size_t at = 0;
    bool ok = true;
    void raw(void* dst, size_t n) {
        if (at + n > b.size()) { ok = false; std::memset(dst, 0, n); return; }
        std::memcpy(dst, b.data() + at, n);
        at += n;
    }
    template <class T> T get() { T v{}; raw(&v, sizeof v); return v; }
    template <class T> std::vector<T> vec(size_t n) { std::vector<T> v(n); raw(v.data(), n * sizeof(T)); return v; }
    std::vector<float> bf16(size_t n) {
        std::vector<uint16_t> h = vec<uint16_t>(n);
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i) f[i] = bf16_to_float(h[i]);
        return f;
    }
};

struct Expect {
    uint64_t hash_logq = 0, hash_bias = 0;
    uint32_t paths[3][kP]{};
    double   eal = 0.0;
    float    conf[3][kP]{};
    uint32_t kconf[3]{};
    uint32_t sample[kP]{};
    uint32_t s_a = 0, s_n = 0;
    int32_t  s_tok[6]{};
    uint32_t g_a = 0, g_n = 0;
    int32_t  g_tok[6]{};
};

struct Case {
    uint32_t input_token = 0;
    std::vector<int32_t> ids;          // [5*32]
    std::vector<float> cl, lse;        // [5*32], [5]
    std::vector<float> e_in, e_cand, h_cand, x;
    std::vector<int32_t> vids, argmax; // [6*32], [6]
    std::vector<float> vlog;
    double u_path[kP]{}, u_acc[kP]{}, u_res[kP + 1]{};
    Expect exp[4][2];
    std::vector<double> diag;          // K = 16, tail: logq [5*16*16]
};

uint64_t fnv1a(const void* p, size_t n) {
    const auto* s = static_cast<const unsigned char*>(p);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= s[i]; h *= 0x100000001b3ull; }
    return h;
}

struct Golden {
    uint32_t Kf = 0, Kv = 0;
    std::vector<float> w;
    std::vector<Case> cases;
};

bool load_golden(Golden& g) {
    const std::string path = std::string(DEEPMOE_TEST_DATA_DIR) + "/dspark/tree_golden.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Reader r{buf};
    char magic[4];
    r.raw(magic, 4);
    if (std::memcmp(magic, "DMTR", 4) != 0) return false;
    const uint32_t version = r.get<uint32_t>(), n = r.get<uint32_t>();
    g.Kf = r.get<uint32_t>();
    g.Kv = r.get<uint32_t>();
    if (version != 1 || g.Kf != 32 || g.Kv != 32) return false;
    g.w = r.bf16(kHidden + kRank);
    for (uint32_t ci = 0; ci < n; ++ci) {
        Case c;
        const uint32_t K = g.Kf;
        c.input_token = r.get<uint32_t>();
        c.ids = r.vec<int32_t>(kP * K);
        c.cl = r.vec<float>(kP * K);
        c.lse = r.vec<float>(kP);
        c.e_in = r.bf16(kRank);
        c.e_cand = r.bf16((kP - 1) * K * kRank);
        c.h_cand = r.bf16(kP * K * kRank);
        c.x = r.bf16(kP * kHidden);
        c.vids = r.vec<int32_t>((kP + 1) * g.Kv);
        c.vlog = r.vec<float>((kP + 1) * g.Kv);
        c.argmax = r.vec<int32_t>(kP + 1);
        r.raw(c.u_path, sizeof c.u_path);
        r.raw(c.u_acc, sizeof c.u_acc);
        r.raw(c.u_res, sizeof c.u_res);
        for (auto& perk : c.exp)
            for (auto& e : perk) {
                e.hash_logq = r.get<uint64_t>();
                e.hash_bias = r.get<uint64_t>();
                for (auto& p : e.paths) r.raw(p, sizeof p);
                e.eal = r.get<double>();
                for (int o = 0; o < 3; ++o) { r.raw(e.conf[o], sizeof e.conf[o]); e.kconf[o] = r.get<uint32_t>(); }
                r.raw(e.sample, sizeof e.sample);
                e.s_a = r.get<uint32_t>(); e.s_n = r.get<uint32_t>(); r.raw(e.s_tok, sizeof e.s_tok);
                e.g_a = r.get<uint32_t>(); e.g_n = r.get<uint32_t>(); r.raw(e.g_tok, sizeof e.g_tok);
            }
        c.diag = r.vec<double>(kP * 16 * 16);
        g.cases.push_back(std::move(c));
    }
    return r.ok && r.at == buf.size();
}

// The top-K prefix of the stored top-32 layout, as contiguous spans.
struct Sub {
    std::vector<int32_t> ids;
    std::vector<float> cl, e_cand, h_cand;
    LatticeInput in(const Case& c, uint32_t K, bool tail) const {
        LatticeInput li;
        li.K = K;
        li.cand_ids = ids;
        li.cand_logit = cl;
        li.e_in = c.e_in;
        li.e_cand = e_cand;
        li.h_cand = h_cand;
        if (tail) li.lse_anchor = c.lse;
        return li;
    }
};

Sub make_sub(const Case& c, uint32_t Kf, uint32_t K) {
    Sub s;
    for (uint32_t i = 0; i < kP; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            s.ids.push_back(c.ids[i * Kf + k]);
            s.cl.push_back(c.cl[i * Kf + k]);
            const float* h = c.h_cand.data() + (i * Kf + k) * kRank;
            s.h_cand.insert(s.h_cand.end(), h, h + kRank);
            if (i + 1 < kP) {
                const float* e = c.e_cand.data() + (i * Kf + k) * kRank;
                s.e_cand.insert(s.e_cand.end(), e, e + kRank);
            }
        }
    return s;
}

}  // namespace

DEEPMOE_TEST(dspark_tree, exp_log_series_are_accurate) {
    for (double x : {-700.0, -40.0, -3.3, -0.7, 0.0, 1e-9, 0.5, 3.0, 40.0}) {
        CHECK(test::close(dm_exp(x), std::exp(x), 1e-13, 0.0));
    }
    CHECK_EQ(dm_exp(-800.0), 0.0);
    for (double y : {1e-300, 1e-20, 0.3, 0.70710678, 1.0, 1.4, 2.0, 1e5, 1e300}) {
        CHECK(test::close(dm_log(y), std::log(y), 1e-13, 1e-15));
    }
    CHECK(std::isinf(dm_log(0.0)) && dm_log(0.0) < 0);
}

DEEPMOE_TEST(dspark_tree, dot16_is_lane_ordered) {
    // A pattern whose value depends on the summation order in float32.
    std::vector<float> a(256), b(256);
    for (int i = 0; i < 256; ++i) {
        a[i] = (i % 3 == 0 ? 1e8f : 1.0f) * (i % 2 ? -1.0f : 1.0f);
        b[i] = 1.0f + static_cast<float>(i) * 1e-3f;
    }
    float acc[16] = {};
    for (int r = 0; r < 16; ++r)
        for (int l = 0; l < 16; ++l) { const float p = a[r * 16 + l] * b[r * 16 + l]; acc[l] = acc[l] + p; }
    float want = 0.0f;
    for (float v : acc) want = want + v;
    CHECK_EQ(dot16(a.data(), b.data(), 256), want);
}

DEEPMOE_TEST(dspark_tree, matches_python_bit_exactly) {
    Golden g;
    REQUIRE(load_golden(g));
    REQUIRE(!g.cases.empty());
    const Objective objs[3] = {Objective::Viterbi, Objective::Eal, Objective::Chain};
    for (size_t ci = 0; ci < g.cases.size(); ++ci) {
        const Case& c = g.cases[ci];
        for (int ki = 0; ki < 4; ++ki) {
            const uint32_t K = kKs[ki];
            const Sub sub = make_sub(c, g.Kf, K);
            for (int ti = 0; ti < 2; ++ti) {
                const bool tail = ti == 0;
                const Expect& e = c.exp[ki][ti];
                Lattice lat;
                REQUIRE(lat.build(sub.in(c, K, tail)).has_value());
                const auto lq = lat.logq_all();
                const auto bs = lat.bias_all();
                CHECK_EQ(fnv1a(lq.data(), lq.size_bytes()), e.hash_logq);
                CHECK_EQ(fnv1a(bs.data(), bs.size_bytes()), e.hash_bias);
                if (K == 16 && tail)
                    for (size_t i = 0; i < c.diag.size(); ++i)
                        if (lq[i] != c.diag[i]) {
                            CHECK_EQ(lq[i], c.diag[i]);
                            break;
                        }
                Path paths[3];
                for (int o = 0; o < 3; ++o) {
                    paths[o] = lat.path(objs[o]);
                    for (uint32_t i = 0; i < kP; ++i) CHECK_EQ(paths[o][i], e.paths[o][i]);
                    std::array<float, kP> conf{};
                    for (uint32_t i = 0; i < kP; ++i) {
                        conf[i] = confidence(c.x.data() + i * kHidden, lat.prev_embed(paths[o], i),
                                             g.w.data());
                        CHECK_EQ(conf[i], e.conf[o][i]);
                    }
                    CHECK_EQ(k_from_confidence(conf, 0.5), e.kconf[o]);
                }
                CHECK_EQ(lat.eal_value(paths[1]), e.eal);

                const Path sp = lat.sample(std::span<const double, kP>(c.u_path, kP));
                for (uint32_t i = 0; i < kP; ++i) CHECK_EQ(sp[i], e.sample[i]);
                const Accept as = accept_sampling(lat, sp, 5, c.vids, c.vlog, g.Kv,
                                                  std::span<const double, kP>(c.u_acc, kP),
                                                  std::span<const double, kP + 1>(c.u_res, kP + 1));
                CHECK_EQ(as.accepted, e.s_a);
                CHECK_EQ(as.n_emitted, e.s_n);
                for (uint32_t i = 0; i < as.n_emitted && i < 6; ++i) CHECK_EQ(as.tokens[i], e.s_tok[i]);

                const auto toks = lat.tokens(paths[1]);
                const Accept ag = accept_greedy(toks, c.argmax, 5);
                CHECK_EQ(ag.accepted, e.g_a);
                CHECK_EQ(ag.n_emitted, e.g_n);
                for (uint32_t i = 0; i < ag.n_emitted && i < 6; ++i) CHECK_EQ(ag.tokens[i], e.g_tok[i]);
            }
        }
    }
}

// Not an assertion: the CPU cost of steps 2 and 4 (docs/p3_dspark.md §5), printed.
DEEPMOE_TEST(dspark_tree, cpu_cost_microseconds) {
    Golden g;
    REQUIRE(load_golden(g));
    const Case& c = g.cases[0];
    for (uint32_t K : kKs) {
        const Sub sub = make_sub(c, g.Kf, K);
        const int iters = K >= 32 ? 400 : 2000;
        double t_build = 0, t_path = 0, t_conf = 0, t_greedy = 0, t_build_nt = 0, t_samp = 0, t_acc = 0;
        using clk = std::chrono::steady_clock;
        auto us = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::micro>(b - a).count();
        };
        for (int it = 0; it < iters; ++it) {
            Lattice lat, lat2;
            const auto t0 = clk::now();
            (void)lat.build(sub.in(c, K, true));
            const auto t1 = clk::now();
            const Path p = lat.path(Objective::Eal);
            const auto t2 = clk::now();
            float cs = 0.0f;
            for (uint32_t i = 0; i < kP; ++i)
                cs += confidence(c.x.data() + i * kHidden, lat.prev_embed(p, i), g.w.data());
            const auto t3 = clk::now();
            const auto toks = lat.tokens(p);
            (void)accept_greedy(toks, c.argmax, 5);
            const auto t4 = clk::now();
            (void)lat2.build(sub.in(c, K, false));
            const auto t5 = clk::now();
            const Path sp = lat2.sample(std::span<const double, kP>(c.u_path, kP));
            const auto t6 = clk::now();
            (void)accept_sampling(lat2, sp, 5, c.vids, c.vlog, g.Kv,
                                  std::span<const double, kP>(c.u_acc, kP),
                                  std::span<const double, kP + 1>(c.u_res, kP + 1));
            const auto t7 = clk::now();
            (void)cs;
            t_build += us(t0, t1); t_path += us(t1, t2); t_conf += us(t2, t3); t_greedy += us(t3, t4);
            t_build_nt += us(t4, t5); t_samp += us(t5, t6); t_acc += us(t6, t7);
        }
        std::printf("  K=%2u  lattice %.1f us  eal path %.2f us  confidence x5 %.2f us  accept_greedy %.3f us"
                    " | lattice(no tail) %.1f us  sample %.2f us  accept_sampling(Kv=%u) %.2f us\n",
                    K, t_build / iters, t_path / iters, t_conf / iters, t_greedy / iters,
                    t_build_nt / iters, t_samp / iters, g.Kv, t_acc / iters);
    }
    DM_UNUSED_CTX();
}
