// prefill_elem -- everything in a prefill layer that is not a GEMM and not the
// attention: the per-row arithmetic of `inference/model.py`'s prefill branch,
// batched over N rows (docs/p3_prefill.md §2).
//
// Stages (`Stage`):
//   0  act_quant           x [n][d] -> E4M3 value as fp16 [n][d] + scale [n][d/32]
//                          (`linear()`'s activation round trip, hoisted: once per
//                          layer for the FFN input instead of once per expert)
//   1  rmsnorm             [n][d] -> [n][d], bf16 out (`RMSNorm.forward`)
//   2  mhc_pre_norm        stream [n][hc][dim] -> sublayer input [n][dim], plus
//                          rsqrt(mean(stream^2)) [n] for hc_mixes
//                          (`hc_mixes`' statistic, `hc_pre`, `attn_norm`)
//   3  sinkhorn            raw mixes [n][24] -> pre / post / comb [n][24]
//                          (`hc_split_sinkhorn`, ending on a COLUMN normalise)
//   4  mhc_post            stream' = post (x) a + comb^T stream  (`hc_post`)
//   5  rope (+quant)       RoPE over the last rope_dim of every head at a
//                          per-row position, then one of: nothing; fp8 E4M3 /
//                          UE8M0-32 (the window KV); FP4 / UE8M0-32 (the index
//                          q and k); FP4 / E4M3-16 (the compressed KV)
//   6  cmp_pool            ratio-way softmax pooling per element (`Compressor`)
//   7  engram_gate         the engram's normalised-dot gate and residual write
//   8  swiglu              the cooperative-matrix MoE's SwiGLU x route weight
//   9  scatter_add         y[idx] += expert rows
//  10  copy_round          [n][d] -> [n][d], bf16-rounded under kFlagRound (a
//                          cooperative-matrix GEMM's padded output into y)
//
// Every stage is one thread per row or per (row, block): no stage here needs a
// reduction across rows, so there are no barriers and no LDS beyond registers.
// Row-major loops over 5120 or 20480 elements are serial inside a thread; they
// are the lightest arithmetic in the layer and the bench says whether that
// ever matters (docs/p3_prefill.md §10).

#include "prefill_common.slang"

static const uint kIn = 0, kOut = 1, kA2 = 2, kA3 = 3, kA4 = 4;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct ElemPush {
    uint  n;          // rows
    uint  d;          // row width
    uint  a0;         // stage-specific (see each stage)
    uint  a1;
    uint  a2;
    uint  a3;
    uint  flags;      // kFlagRound, kFlagInverse
    float eps;
    float f1;
    uint  pos0;       // rope: position of row 0
    uint  pos_step;   // rope: position increment per row
    uint  row_off;    // rope: row index offset into the position sequence
};
[[vk::push_constant]] ElemPush pc;

uint thread_index(uint3 gid, uint tid) { return gid_linear(gid) * kWg + tid; }

void load_row32(uint64_t base, out float v[32]) {
    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint4 a = load16(base + (uint64_t)i * 16);
        [unroll] for (uint j = 0; j < 4; ++j) v[i * 4 + j] = asfloat(a[j]);
    }
}

// --- 0: act_quant -------------------------------------------------------------
// kIn = x f32, kOut = fp16 values, kA2 = f32 scales. flags & kFlagRound rounds
// the input onto bf16 first (`x.to(dtype)` before `linear()`).
void s_act_quant(uint3 gid, uint tid) {
    const uint blocks = pc.d / kBlockK;
    const uint idx = thread_index(gid, tid);
    const uint rr = idx / blocks, blk = idx % blocks;
    if (rr >= pc.n) return;
    const uint row = rr + pc.row_off;      // rows [row_off, row_off + n)
    float v[32];
    load_row32(Ptr[kIn] + ((uint64_t)row * pc.d + (uint64_t)blk * 32) * 4, v);
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 32; ++i) {
        v[i] = maybe_round(v[i], pc.flags);
        amax = max(amax, abs(v[i]));
    }
    const float s = act_scale(amax);
    const float inv = 1.0f / s;
    const uint64_t q = Ptr[kOut] + ((uint64_t)row * pc.d + (uint64_t)blk * 32) * 2;
    [unroll] for (uint i = 0; i < 32; ++i)
        *(uint16_t*)(q + (uint64_t)i * 2) =
            (uint16_t)f32tof16(fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF)));
    store_f32(Ptr[kA2] + ((uint64_t)row * blocks + blk) * 4, s);
}

// --- 1: rmsnorm ---------------------------------------------------------------
// kIn [n][d] f32, kOut [n][d], kA2 = bf16 weight [d]. `(w * (x * rsqrt(var +
// eps))).to(bf16)`; the input is taken as given (already bf16 where the
// reference's is).
void s_rmsnorm(uint3 gid, uint tid) {
    const uint row = thread_index(gid, tid);
    if (row >= pc.n) return;
    const uint64_t x = Ptr[kIn] + (uint64_t)row * pc.d * 4;
    float sq = 0.0f;
    for (uint e = 0; e < pc.d; ++e) { const float v = load_f32(x + (uint64_t)e * 4); sq += v * v; }
    const float rs = rsqrt(sq / float(pc.d) + pc.eps);
    const uint64_t y = Ptr[kOut] + (uint64_t)row * pc.d * 4;
    for (uint e = 0; e < pc.d; ++e) {
        const float w = bf16_decode(load_u16(Ptr[kA2] + (uint64_t)e * 2));
        store_f32(y + (uint64_t)e * 4, bf16_round((load_f32(x + (uint64_t)e * 4) * rs) * w));
    }
}

// --- 2: mhc_pre_norm ------------------------------------------------------------
// kIn = stream [n][hc][d], kOut = [n][d], kA2 = bf16 norm weight [d],
// kA3 = coefficients [n][a0] with `pre` at offset 0, kA4 = rsqrt(mean(stream^2)
// + eps) [n] for the hc_fn GEMM's row scale. a1 = hc.
static const uint kMaxHc = 4;
void s_mhc_pre_norm(uint3 gid, uint tid) {
    const uint row = thread_index(gid, tid);
    if (row >= pc.n) return;
    const uint hc = pc.a1, d = pc.d;
    const uint64_t h = Ptr[kIn] + (uint64_t)row * hc * d * 4;
    float pre[kMaxHc];
    [unroll] for (uint j = 0; j < kMaxHc; ++j)
        pre[j] = (j < hc) ? load_f32(Ptr[kA3] + ((uint64_t)row * pc.a0 + j) * 4) : 0.0f;
    float sq_all = 0.0f, sq_u = 0.0f;
    const uint64_t y = Ptr[kOut] + (uint64_t)row * d * 4;
    for (uint e = 0; e < d; ++e) {
        float u = 0.0f;
        [unroll] for (uint j = 0; j < kMaxHc; ++j) {
            if (j >= hc) continue;
            const float v = load_f32(h + ((uint64_t)j * d + e) * 4);
            sq_all += v * v;
            u += pre[j] * v;
        }
        u = bf16_round(u);           // hc_pre ends in `.to(x.dtype)`
        sq_u += u * u;
        store_f32(y + (uint64_t)e * 4, u);
    }
    store_f32(Ptr[kA4] + (uint64_t)row * 4, rsqrt(sq_all / float(hc * d) + pc.eps));
    const float rs = rsqrt(sq_u / float(d) + pc.eps);
    for (uint e = 0; e < d; ++e) {
        const float w = bf16_decode(load_u16(Ptr[kA2] + (uint64_t)e * 2));
        store_f32(y + (uint64_t)e * 4, bf16_round((load_f32(y + (uint64_t)e * 4) * rs) * w));
    }
}

// --- 3: sinkhorn ----------------------------------------------------------------
// kIn = raw mixes [n][24] (hc_fn @ x times the row rsqrt), kOut = [n][24],
// kA2 = base [24], kA3 = scale [3]. a0 = iterations, f1 = hc_eps.
void s_sinkhorn(uint3 gid, uint tid) {
    const uint row = thread_index(gid, tid);
    if (row >= pc.n) return;
    const uint64_t mr = Ptr[kIn] + (uint64_t)row * 24 * 4;
    const uint64_t out = Ptr[kOut] + (uint64_t)row * 24 * 4;
    const uint64_t bs = Ptr[kA2], sc = Ptr[kA3];
    const float s0 = load_f32(sc), s1 = load_f32(sc + 4), s2 = load_f32(sc + 8);
    const float eps = pc.f1;
    [unroll] for (uint j = 0; j < kMaxHc; ++j) {
        const float m = load_f32(mr + (uint64_t)j * 4) * s0 + load_f32(bs + (uint64_t)j * 4);
        store_f32(out + (uint64_t)j * 4, 1.0f / (1.0f + exp(-m)) + eps);
        const uint k = kMaxHc + j;
        const float p = load_f32(mr + (uint64_t)k * 4) * s1 + load_f32(bs + (uint64_t)k * 4);
        store_f32(out + (uint64_t)k * 4, 2.0f / (1.0f + exp(-p)));
    }
    float c[kMaxHc][kMaxHc];
    [unroll] for (uint j = 0; j < kMaxHc; ++j) {
        float mx = -3.0e38f;
        [unroll] for (uint k = 0; k < kMaxHc; ++k) {
            const uint i = 2 * kMaxHc + j * kMaxHc + k;
            c[j][k] = load_f32(mr + (uint64_t)i * 4) * s2 + load_f32(bs + (uint64_t)i * 4);
            mx = max(mx, c[j][k]);
        }
        float sum = 0.0f;
        [unroll] for (uint k = 0; k < kMaxHc; ++k) { c[j][k] = exp(c[j][k] - mx); sum += c[j][k]; }
        [unroll] for (uint k = 0; k < kMaxHc; ++k) c[j][k] = c[j][k] / sum + eps;
    }
    [unroll] for (uint k = 0; k < kMaxHc; ++k) {
        float s = eps;
        [unroll] for (uint j = 0; j < kMaxHc; ++j) s += c[j][k];
        [unroll] for (uint j = 0; j < kMaxHc; ++j) c[j][k] /= s;
    }
    // Bounded by the specialisation-free constant 19 when a0 == 20: a loop
    // bounded by a push constant around `c` is the scratch-memory trap of
    // p2_attention.md §4.4, so the iteration count is fixed at compile time
    // and a0 only gates it.
    [unroll] for (uint it = 1; it < 20; ++it) {
        if (it >= pc.a0) continue;
        [unroll] for (uint j = 0; j < kMaxHc; ++j) {
            float s = eps;
            [unroll] for (uint k = 0; k < kMaxHc; ++k) s += c[j][k];
            [unroll] for (uint k = 0; k < kMaxHc; ++k) c[j][k] /= s;
        }
        [unroll] for (uint k = 0; k < kMaxHc; ++k) {
            float s = eps;
            [unroll] for (uint j = 0; j < kMaxHc; ++j) s += c[j][k];
            [unroll] for (uint j = 0; j < kMaxHc; ++j) c[j][k] /= s;
        }
    }
    [unroll] for (uint j = 0; j < kMaxHc; ++j)
        [unroll] for (uint k = 0; k < kMaxHc; ++k)
            store_f32(out + (uint64_t)(2 * kMaxHc + j * kMaxHc + k) * 4, c[j][k]);
}

// --- 4: mhc_post ----------------------------------------------------------------
// kIn = stream [n][hc][d], kOut = stream' [n][hc][d] (a different buffer),
// kA2 = sublayer output a [n][d], kA3 = coefficients [n][24] (post at 4, comb
// at 8). One thread per (row, 16 elements). a1 = hc. comb[i][j] contracts
// against residual[i]: comb's FIRST index (design §7.7).
void s_mhc_post(uint3 gid, uint tid) {
    const uint chunks = pc.d / 16;
    const uint idx = thread_index(gid, tid);
    const uint row = idx / chunks, c0 = (idx % chunks) * 16;
    if (row >= pc.n) return;
    const uint hc = pc.a1, d = pc.d;
    const uint64_t cf = Ptr[kA3] + (uint64_t)row * 24 * 4;
    float post[kMaxHc];
    float comb[kMaxHc][kMaxHc];
    [unroll] for (uint j = 0; j < kMaxHc; ++j) {
        post[j] = load_f32(cf + (uint64_t)(kMaxHc + j) * 4);
        [unroll] for (uint i = 0; i < kMaxHc; ++i)
            comb[i][j] = load_f32(cf + (uint64_t)(2 * kMaxHc + i * kMaxHc + j) * 4);
    }
    const uint64_t h = Ptr[kIn] + (uint64_t)row * hc * d * 4;
    const uint64_t o = Ptr[kOut] + (uint64_t)row * hc * d * 4;
    [unroll] for (uint q = 0; q < 16; ++q) {
        const uint e = c0 + q;
        // The sublayer output enters as the bf16 tensor the reference hands
        // hc_post (`ffn_out.to(residual.dtype)`); a no-op for wo_b's rounded one.
        const float a = maybe_round(load_f32(Ptr[kA2] + ((uint64_t)row * d + e) * 4), pc.flags);
        float r[kMaxHc];
        [unroll] for (uint i = 0; i < kMaxHc; ++i) r[i] = load_f32(h + ((uint64_t)i * d + e) * 4);
        [unroll] for (uint j = 0; j < kMaxHc; ++j) {
            float s = 0.0f;
            [unroll] for (uint i = 0; i < kMaxHc; ++i) s += comb[i][j] * r[i];
            store_f32(o + ((uint64_t)j * d + e) * 4, maybe_round(post[j] * a + s, pc.flags));
        }
    }
}

// --- 5: rope (+quant) ------------------------------------------------------------
// kIn [n][d], kOut [n][d] (may alias kIn), kA2 = (cos, sin) table
// [positions][rope_dim/2][2], kA3 = packed bytes (fp8 [n][d] or fp4 [n][d/2]),
// kA4 = scale bytes [n][d/block]; either may be 0.
// a0 = head_dim, a1 = mode (0 none, 1 fp8/UE8M0-32, 2 fp4/UE8M0-32,
// 3 fp4/E4M3-16), a2 = rope_dim, a3 = block (32 or 16).
float idx_fp4_scale(float amax) {
    const float v = max(amax, 6.0f * 5.877472e-39f) * (1.0f / 6.0f);
    const uint  b = asuint(v);
    const int   e = int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}
void s_rope(uint3 gid, uint tid) {
    const uint blk_sz = pc.a3;
    const uint bpr = pc.d / blk_sz;
    const uint idx = thread_index(gid, tid);
    const uint row = idx / bpr, blk = idx % bpr;
    if (row >= pc.n) return;
    const uint e0 = blk * blk_sz;
    const uint hd = pc.a0;
    const uint lo = hd - pc.a2;
    const uint pos = pc.pos0 + (pc.row_off + row) * pc.pos_step;
    const bool inverse = (pc.flags & kFlagInverse) != 0u;

    float v[32];
    [unroll] for (uint i = 0; i < 32; ++i)
        v[i] = (i < blk_sz) ? load_f32(Ptr[kIn] + ((uint64_t)row * pc.d + e0 + i) * 4) : 0.0f;
    const uint64_t tab = Ptr[kA2] + (uint64_t)pos * pc.a2 * 4;   // rope_dim/2 pairs x 2 floats
    [unroll] for (uint i = 0; i < 32; i += 2) {
        if (i >= blk_sz) continue;
        const uint rel = (e0 + i) % hd;
        if (rel < lo) continue;
        const uint j = (rel - lo) >> 1;
        const float c = load_f32(tab + (uint64_t)j * 8);
        const float s = load_f32(tab + (uint64_t)j * 8 + 4);
        rope_pair(c, s, v[i], v[i + 1], inverse);
        v[i] = maybe_round(v[i], pc.flags);
        v[i + 1] = maybe_round(v[i + 1], pc.flags);
    }
    const uint64_t out = Ptr[kOut] + ((uint64_t)row * pc.d + e0) * 4;
    if (pc.a1 == 0) {
        [unroll] for (uint i = 0; i < 32; ++i) if (i < blk_sz) store_f32(out + (uint64_t)i * 4, v[i]);
        return;
    }
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 32; ++i) if (i < blk_sz) amax = max(amax, abs(v[i]));
    if (pc.a1 == 1) {
        // act_quant(kv, 32, "ue8m0", inplace=True): the window KV.
        const float sc = act_scale(amax);
        const float inv = 1.0f / sc;
        [unroll] for (uint i = 0; i < 32; ++i) {
            const float q = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
            store_f32(out + (uint64_t)i * 4, q * sc);
            if (Ptr[kA3] != 0)
                *(uint8_t*)(Ptr[kA3] + (uint64_t)row * pc.d + e0 + i) = (uint8_t)fp8_pack(q);
        }
        if (Ptr[kA4] != 0)
            *(uint8_t*)(Ptr[kA4] + (uint64_t)row * bpr + blk) = (uint8_t)((asuint(sc) >> 23) & 0xFFu);
        return;
    }
    // fp4_act_quant: the indexer's (block 32, power-of-two scale, floor
    // 6 * 2^-126) or the compressed KV's (block 16, E4M3 scale, floor 6 * 2^-9).
    float sc;
    if (pc.a1 == 2) sc = idx_fp4_scale(amax);
    else            sc = fp8_round(clamp(max(amax, 6.0f * 0.001953125f) / 6.0f, 0.0f, kFp8MaxF));
    const float inv = 1.0f / sc;
    uint packed[16];
    [unroll] for (uint z = 0; z < 16; ++z) packed[z] = 0u;
    [unroll] for (uint i = 0; i < 32; ++i) {
        if (i >= blk_sz) continue;
        const float q = fp4_round(clamp(v[i] * inv, -6.0f, 6.0f));
        store_f32(out + (uint64_t)i * 4, q * sc);
        const uint code = fp4_code(q);
        if ((i & 1u) == 0u) packed[i >> 1] = code; else packed[i >> 1] |= code << 4;
    }
    if (Ptr[kA3] != 0)
        [unroll] for (uint i = 0; i < 16; ++i)
            if (i < blk_sz / 2)
                *(uint8_t*)(Ptr[kA3] + ((uint64_t)row * pc.d + e0) / 2 + i) = (uint8_t)packed[i];
    if (Ptr[kA4] != 0) {
        const uint code = (pc.a1 == 2) ? ((asuint(sc) >> 23) & 0xFFu) : fp8_pack(sc);
        *(uint8_t*)(Ptr[kA4] + (uint64_t)row * bpr + blk) = (uint8_t)code;
    }
}

// --- 6: cmp_pool ----------------------------------------------------------------
// kIn = wkv [T][d] f32, kA2 = wgate [T][d] f32, kOut = pooled [G][d], bf16 out
// (`kv.to(dtype)` before the norm). a0 = ratio. One thread per (group, element).
static const uint kMaxRatio = 8;
void s_cmp_pool(uint3 gid, uint tid) {
    const uint idx = thread_index(gid, tid);
    const uint g = idx / pc.d, e = idx % pc.d;
    if (g >= pc.n) return;
    const uint r = pc.a0;
    float s[kMaxRatio];
    float mx = -3.0e38f;
    [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
        s[j] = (j < r) ? load_f32(Ptr[kA2] + ((uint64_t)(g * r + j) * pc.d + e) * 4) : -3.0e38f;
        mx = max(mx, s[j]);
    }
    float den = 0.0f;
    [unroll] for (uint j = 0; j < kMaxRatio; ++j) { s[j] = (j < r) ? exp(s[j] - mx) : 0.0f; den += s[j]; }
    float acc = 0.0f;
    [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
        if (j >= r) continue;
        acc += load_f32(Ptr[kIn] + ((uint64_t)(g * r + j) * pc.d + e) * 4) * (s[j] / den);
    }
    store_f32(Ptr[kOut] + ((uint64_t)g * pc.d + e) * 4, bf16_round(acc));
}

// --- 7: engram_gate -------------------------------------------------------------
// kIn = stream [n][hc][d], kOut = stream' [n][hc][d], kA2 = wkv out [n][(hc+1) d]
// (hc keys then the value, bf16-valued), kA3 = q_weight [hc][d] bf16,
// kA4 = k_weight [hc][d] bf16. a1 = hc, eps = norm_eps, f1 = 1e-6 (the sqrt clamp).
void s_engram_gate(uint3 gid, uint tid) {
    const uint hc = pc.a1, d = pc.d;
    const uint idx = thread_index(gid, tid);
    const uint row = idx / hc, c = idx % hc;
    if (row >= pc.n) return;
    const uint64_t h = Ptr[kIn] + ((uint64_t)row * hc + c) * d * 4;
    const uint64_t key = Ptr[kA2] + ((uint64_t)row * (hc + 1) * d + (uint64_t)c * d) * 4;
    const uint64_t val = Ptr[kA2] + ((uint64_t)row * (hc + 1) * d + (uint64_t)hc * d) * 4;
    float sh = 0.0f, sk = 0.0f, dot = 0.0f;
    for (uint e = 0; e < d; ++e) {
        const float hv = load_f32(h + (uint64_t)e * 4);
        const float kv = load_f32(key + (uint64_t)e * 4);
        const float w = bf16_decode(load_u16(Ptr[kA3] + ((uint64_t)c * d + e) * 2)) *
                        bf16_decode(load_u16(Ptr[kA4] + ((uint64_t)c * d + e) * 2));
        sh += hv * hv; sk += kv * kv;
        dot += (hv * w) * kv;
    }
    const float rstd = rsqrt(sh / float(d) + pc.eps) * rsqrt(sk / float(d) + pc.eps);
    const float dt = dot * rstd * (1.0f / sqrt(float(d)));
    const float g = 1.0f / (1.0f + exp(-(sqrt(max(abs(dt), pc.f1)) * sign(dt))));
    const uint64_t o = Ptr[kOut] + ((uint64_t)row * hc + c) * d * 4;
    for (uint e = 0; e < d; ++e)
        store_f32(o + (uint64_t)e * 4,
                  bf16_round(load_f32(h + (uint64_t)e * 4) + g * load_f32(val + (uint64_t)e * 4)));
}

// --- 8: swiglu --------------------------------------------------------------------
// The cooperative-matrix MoE's elementwise middle (prefill_gemm.slang stage 1
// does it inside the GEMV). kIn = gate rows [0, n) and up rows [a0, a0 + n)
// of one plane, kOut = h rows [a1, a1 + n), kA2 = route weights at [a2, a2 + n).
// f1 = swiglu_limit. `Expert.forward`: bf16 gate and up widened, clamp, SiLU,
// times the route weight, `.to(dtype)`.
void s_swiglu(uint3 gid, uint tid) {
    const uint chunks = pc.d / 16;
    const uint idx = thread_index(gid, tid);
    const uint row = idx / chunks, c0 = (idx % chunks) * 16;
    if (row >= pc.n) return;
    const float rw = load_f32(Ptr[kA2] + (uint64_t)(pc.a2 + row) * 4);
    const uint64_t gi = Ptr[kIn] + ((uint64_t)row * pc.d + c0) * 4;
    const uint64_t ui = Ptr[kIn] + ((uint64_t)(pc.a0 + row) * pc.d + c0) * 4;
    const uint64_t ho = Ptr[kOut] + ((uint64_t)(pc.a1 + row) * pc.d + c0) * 4;
    [unroll] for (uint q = 0; q < 16; ++q) {
        const float g = maybe_round(load_f32(gi + (uint64_t)q * 4), pc.flags);
        const float u = maybe_round(load_f32(ui + (uint64_t)q * 4), pc.flags);
        store_f32(ho + (uint64_t)q * 4, maybe_round(swiglu(g, u, pc.f1) * rw, pc.flags));
    }
}

// --- 9: scatter_add ---------------------------------------------------------------
// `y[idx] += expert(x[idx])`: kIn = the expert's output rows [n][d], kOut = y,
// kA2 = the token list, rows at [a2, a2 + n). Tokens are distinct within one
// expert, so no two threads of a dispatch write the same element.
void s_scatter_add(uint3 gid, uint tid) {
    const uint chunks = pc.d / 16;
    const uint idx = thread_index(gid, tid);
    const uint row = idx / chunks, c0 = (idx % chunks) * 16;
    if (row >= pc.n) return;
    const uint t = load_u32(Ptr[kA2] + (uint64_t)(pc.a2 + row) * 4);
    const uint64_t src = Ptr[kIn] + ((uint64_t)row * pc.d + c0) * 4;
    const uint64_t dst = Ptr[kOut] + ((uint64_t)t * pc.d + c0) * 4;
    [unroll] for (uint q = 0; q < 16; ++q)
        store_f32(dst + (uint64_t)q * 4,
                  load_f32(dst + (uint64_t)q * 4) + maybe_round(load_f32(src + (uint64_t)q * 4), pc.flags));
}

// --- 10: copy_round -----------------------------------------------------------------
// kIn rows [0, n) -> kOut rows [0, n), 16 elements a thread, rounded onto bf16
// under kFlagRound. d is a multiple of 16.
void s_copy_round(uint3 gid, uint tid) {
    const uint chunks = pc.d / 16;
    const uint idx = thread_index(gid, tid);
    const uint row = idx / chunks, c0 = (idx % chunks) * 16;
    if (row >= pc.n) return;
    const uint64_t src = Ptr[kIn] + ((uint64_t)row * pc.d + c0) * 4;
    const uint64_t dst = Ptr[kOut] + ((uint64_t)row * pc.d + c0) * 4;
    [unroll] for (uint q = 0; q < 16; ++q)
        store_f32(dst + (uint64_t)q * 4, maybe_round(load_f32(src + (uint64_t)q * 4), pc.flags));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 10)     s_copy_round(gid, tid);
    else if (Stage == 8) s_swiglu(gid, tid);
    else if (Stage == 9) s_scatter_add(gid, tid);
    else if (Stage == 0) s_act_quant(gid, tid);
    else if (Stage == 1) s_rmsnorm(gid, tid);
    else if (Stage == 2) s_mhc_pre_norm(gid, tid);
    else if (Stage == 3) s_sinkhorn(gid, tid);
    else if (Stage == 4) s_mhc_post(gid, tid);
    else if (Stage == 5) s_rope(gid, tid);
    else if (Stage == 6) s_cmp_pool(gid, tid);
    else                 s_engram_gate(gid, tid);
}
