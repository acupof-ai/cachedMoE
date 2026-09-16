// mgt1_idx -- indexer.slang for a verify batch of M queries (Track T,
// docs/p4_mgt1.md). The indexer's wq_b itself is mgt1_gemv.slang stages 0-1 on
// the already-normalised qr; this file is everything after it.
//
// Per query, not per batch
// ------------------------
// Query m at position P = p0 + m sees n_pos(m) = (P + 1) / ratio compressed
// positions, keeps min(index_topk, n_pos(m)) of them, and scores them against
// the key cache it would have seen had the batch been decoded one token at a
// time (docs/p3_dspark.md §4.8, the reference's `_dm_emulate_decode`): its own
// source layer's cache when its position completes a group there, otherwise
// whatever was published before -- the host passes both addresses and one bit
// per query (`key_sel`, 1 = own). Rows of the own cache written by LATER
// queries of the batch are beyond n_pos(m), so they are never scored. That is
// the whole causal mask of the compressed half; the window half is the host's.
//
// Stages (`Stage`; the numbering continues indexer.slang's where it can):
//   1  RoPE + FP4 block-32/UE8M0 on q, gid = (0, m)
//   2  the index key of completion c: k_norm(wk(latent_c)), rotated at the
//      latent's position, FP4, into cache row (p0 + m_c) / ratio; gid = (0, c)
//   3  weights_proj, bf16 [32 x 5120] against token m's input; gid = (0, m)
//   4  scores, eight positions a workgroup; gid = (tile, m)
//   5  the top-k into query m's list at `offset`; gid = (0, m)
//   6  design §2.1 block keys; gid = (chunk, m)
//   7  design §2.1 block selection; gid = (0, m)
//   8  design §2.1 mask on a consumer's scores; gid = (0, m)

#include "attn_common.slang"

static const uint kQRaw = 0, kQ = 1, kQFp4 = 2, kQScale = 3, kRope = 4;
static const uint kWk = 5, kKNormW = 6, kLatent = 7, kKRaw = 8, kKCache = 9;
static const uint kKFp4 = 10, kKScale = 11, kWProjW = 12, kX = 13, kWeights = 14;
static const uint kScore = 15, kOut = 16, kKCachePub = 17, kBlkKey = 18, kCand = 19;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct Mgt1IdxPush {
    uint  n_heads;        // 32
    uint  head_dim;       // 128
    uint  rope_dim;       // 64
    uint  k;              // 3: dim 5120; 2: latent 512; 6-8: candidate_block_size
    uint  p0;
    uint  ratio;
    uint  topk;           // 5: index_topk; 7: candidate_topk_blocks
    uint  offset;         // 5: where the compressed half starts (window + M - 1)
    uint  score_stride;   // floats between query rows of kScore
    uint  list_stride;    // int32 entries between query rows of kOut
    uint  key_sel;        // bit m: query m scores its own source's cache
    uint  blk_stride;     // 6-8: entries between query rows of kBlkKey / kCand
    float norm_eps;
    float wscale;
};
[[vk::push_constant]] Mgt1IdxPush pc;

static const uint kPool     = 5120;
static const uint kKTile    = 8;
static const uint kIdxBlock = 32;
static const float kFp4Max  = 6.0f;
groupshared float gPool[kPool];
groupshared uint  gHist[256];
groupshared uint  gScan[256];
groupshared uint  gSel[4];

uint n_pos(uint m) { return (pc.p0 + m + 1u) / pc.ratio; }

float idx_fp4_scale(float amax) {
    const float v = max(amax, kFp4Max * 5.877472e-39f) * (1.0f / kFp4Max);
    const uint  b = asuint(v);
    const int   e = int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}

void rope_quant_block(uint64_t src, uint64_t dstVal, uint64_t dstFp4, uint64_t dstScale,
                      uint64_t rope, uint blk_in_row, uint d0) {
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    float v[kIdxBlock];
    [unroll] for (uint i = 0; i < kIdxBlock; ++i) v[i] = load_f32(src + (uint64_t)i * 4);
    [unroll] for (uint i = 0; i < kIdxBlock; i += 2) {
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        rope_pair(load_f32(rope + (uint64_t)j * 8), load_f32(rope + (uint64_t)j * 8 + 4),
                  v[i], v[i + 1], false);
        v[i] = bf16_round(v[i]);
        v[i + 1] = bf16_round(v[i + 1]);
    }
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < kIdxBlock; ++i) amax = max(amax, abs(v[i]));
    const float sc = idx_fp4_scale(amax);
    const float inv = 1.0f / sc;
    uint packed[kIdxBlock / 2];
    [unroll] for (uint z = 0; z < kIdxBlock / 2; ++z) packed[z] = 0u;
    [unroll] for (uint i = 0; i < kIdxBlock; ++i) {
        const float q = fp4_round(clamp(v[i] * inv, -kFp4Max, kFp4Max));
        const uint  code = fp4_code(q);
        if ((i & 1) == 0) packed[i >> 1] = code;
        else              packed[i >> 1] |= code << 4;
        *(uint16_t*)(dstVal + (uint64_t)i * 2) = (uint16_t)bf16_encode(q * sc);
    }
    [unroll] for (uint i = 0; i < kIdxBlock / 2; ++i)
        *(uint8_t*)(dstFp4 + (uint64_t)i) = (uint8_t)packed[i];
    *(uint8_t*)(dstScale + (uint64_t)blk_in_row) = (uint8_t)((asuint(sc) >> 23) & 0xFFu);
}

// --- stage 1 --------------------------------------------------------------------
void stage_q_finish(uint m, uint tid) {
    const uint rows = pc.n_heads * pc.head_dim;
    const uint blocks = rows / kIdxBlock;
    if (tid >= blocks) return;
    const uint per_row = pc.head_dim / kIdxBlock;
    const uint d0 = (tid % per_row) * kIdxBlock;
    rope_quant_block(Ptr[kQRaw] + ((uint64_t)m * rows + (uint64_t)tid * kIdxBlock) * 4,
                     Ptr[kQ] + ((uint64_t)m * rows + (uint64_t)tid * kIdxBlock) * 2,
                     Ptr[kQFp4] + ((uint64_t)m * (rows / 2) + (uint64_t)tid * (kIdxBlock / 2)),
                     Ptr[kQScale] + (uint64_t)m * blocks,
                     Ptr[kRope] + (uint64_t)m * (pc.rope_dim / 2) * 8, tid, d0);
}

// --- stage 2 --------------------------------------------------------------------
void stage_key(uint c, uint tid) {
    const uint64_t lat = Ptr[kLatent] + (uint64_t)c * pc.k * 4;
    for (uint i = tid; i < pc.k; i += kWg) gPool[i] = load_f32(lat + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();
    float dot = 0.0f;
    if (tid < pc.head_dim) {
        const uint64_t row = Ptr[kWk] + (uint64_t)tid * pc.k * 2;
        for (uint base = 0; base < pc.k; base += 8) {
            const uint4 w = load16(row + (uint64_t)base * 2);
            [unroll] for (uint j = 0; j < 4; ++j) {
                dot += bf16_decode(w[j] & 0xFFFFu) * gPool[base + j * 2];
                dot += bf16_decode(w[j] >> 16)     * gPool[base + j * 2 + 1];
            }
        }
        dot = bf16_round(dot);
    }
    const float rs = rsqrt(wg_reduce(dot * dot, tid) / float(pc.head_dim) + pc.norm_eps);
    const uint64_t kraw = Ptr[kKRaw] + (uint64_t)c * pc.head_dim * 4;
    if (tid < pc.head_dim)
        *(float*)(kraw + (uint64_t)tid * 4) =
            bf16_round(dot * rs * bf16_decode(load_u16(Ptr[kKNormW] + (uint64_t)tid * 2)));
    GroupMemoryBarrierWithGroupSync();

    const uint blocks = pc.head_dim / kIdxBlock;
    if (tid >= blocks) return;
    const uint first = (pc.ratio - 1u - pc.p0 % pc.ratio) % pc.ratio;
    const uint k_row = (pc.p0 + first + c * pc.ratio) / pc.ratio;
    const uint64_t base = Ptr[kKCache] + (uint64_t)k_row * pc.head_dim * 2;
    rope_quant_block(kraw + (uint64_t)tid * kIdxBlock * 4,
                     base + (uint64_t)tid * kIdxBlock * 2,
                     Ptr[kKFp4] + ((uint64_t)c * (pc.head_dim / 2) + (uint64_t)tid * (kIdxBlock / 2)),
                     Ptr[kKScale] + (uint64_t)c * blocks,
                     Ptr[kRope] + (uint64_t)c * (pc.rope_dim / 2) * 8, tid, tid * kIdxBlock);
}

// --- stage 3 --------------------------------------------------------------------
void stage_weights(uint m, uint tid) {
    const uint64_t X = Ptr[kX] + (uint64_t)m * pc.k * 4;
    for (uint i = tid; i < pc.k; i += kWg) gPool[i] = load_f32(X + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();
    const uint sublane = tid % LanesPerRow;
    const uint rowsAtOnce = kWg / LanesPerRow;
    for (uint rb = 0; rb < pc.n_heads; rb += rowsAtOnce) {
        const uint row = rb + tid / LanesPerRow;
        float acc = 0.0f;
        if (row < pc.n_heads) {
            const uint64_t w = Ptr[kWProjW] + (uint64_t)row * pc.k * 2;
            for (uint base = sublane * 8; base < pc.k; base += LanesPerRow * 8) {
                const uint4 v = load16(w + (uint64_t)base * 2);
                [unroll] for (uint j = 0; j < 4; ++j) {
                    acc += bf16_decode(v[j] & 0xFFFFu) * gPool[base + j * 2];
                    acc += bf16_decode(v[j] >> 16)     * gPool[base + j * 2 + 1];
                }
            }
        }
        const float dot = row_reduce(acc, tid, sublane);
        if (sublane == 0 && row < pc.n_heads)
            *(float*)(Ptr[kWeights] + ((uint64_t)m * pc.n_heads + row) * 4) =
                bf16_round(bf16_round(dot) * pc.wscale);
    }
}

// --- stage 4 --------------------------------------------------------------------
void stage_score(uint gid, uint m, uint tid) {
    const uint hd = pc.head_dim, nh = pc.n_heads;
    const uint np = n_pos(m);
    const uint64_t q = Ptr[kQ] + (uint64_t)m * nh * hd * 2;
    const uint64_t kc = (((pc.key_sel >> m) & 1u) != 0u) ? Ptr[kKCache] : Ptr[kKCachePub];
    for (uint i = tid; i < nh * hd; i += kWg)
        gPool[i] = bf16_decode(load_u16(q + (uint64_t)i * 2));
    const uint t0 = gid * kKTile;
    for (uint i = tid; i < kKTile * hd; i += kWg) {
        const uint t = t0 + i / hd;
        gPool[nh * hd + i] = (t < np)
            ? bf16_decode(load_u16(kc + ((uint64_t)t * hd + i % hd) * 2)) : 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint p = tid / nh, h = tid % nh;
    float dot = 0.0f;
    if (p < kKTile) {
        for (uint d = 0; d < hd; ++d)
            dot += gPool[h * hd + d] * gPool[nh * hd + p * hd + d];
    }
    const float contrib = bf16_round(max(bf16_round(dot), 0.0f) *
                                     load_f32(Ptr[kWeights] + ((uint64_t)m * nh + h) * 4));
    const float s = bf16_round(row_reduce(contrib, tid, h));
    if (h == 0 && p < kKTile && t0 + p < np)
        *(float*)(Ptr[kScore] + ((uint64_t)m * pc.score_stride + t0 + p) * 4) = s;
}

// --- stage 5 --------------------------------------------------------------------
uint score_key(float f) {
    const uint u = asuint(f);
    return ((u & 0x80000000u) != 0u) ? ~u : (u | 0x80000000u);
}
uint sel_key_at(uint64_t base, bool blocks, uint t) {
    return blocks ? load_u32(base + (uint64_t)t * 4) : score_key(load_f32(base + (uint64_t)t * 4));
}

uint wg_exscan(uint v, uint tid) {
    gScan[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 1; s < kWg; s <<= 1) {
        const uint a = (tid >= s) ? gScan[tid - s] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gScan[tid] += a;
        GroupMemoryBarrierWithGroupSync();
    }
    const uint inclusive = gScan[tid];
    if (tid == kWg - 1) gSel[3] = inclusive;
    GroupMemoryBarrierWithGroupSync();
    return inclusive - v;
}

void radix_threshold(uint64_t base, bool blocks, uint n, uint kk, uint tid,
                     out uint prefix_out, out uint need_out) {
    uint prefix = 0u, mask = 0u, need = kk;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24u - 8u * pass;
        gHist[tid] = 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint t = tid; t < n; t += kWg) {
            const uint key = sel_key_at(base, blocks, t);
            if ((key & mask) == prefix) InterlockedAdd(gHist[(key >> shift) & 0xFFu], 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0) {
            uint cum = 0u, bin = 0u;
            for (uint i = 0; i < 256u; ++i) {
                const uint b = 255u - i;
                if (cum + gHist[b] >= need) { bin = b; break; }
                cum += gHist[b];
            }
            gSel[0] = bin;
            gSel[1] = need - cum;
        }
        GroupMemoryBarrierWithGroupSync();
        prefix |= gSel[0] << shift;
        mask   |= 0xFFu << shift;
        need    = gSel[1];
        GroupMemoryBarrierWithGroupSync();
    }
    prefix_out = prefix;
    need_out   = need;
}

void stage_topk(uint m, uint tid) {
    const uint n = n_pos(m);
    const uint kk = min(pc.topk, n);
    const uint64_t sc = Ptr[kScore] + (uint64_t)m * pc.score_stride * 4;
    const uint64_t out = Ptr[kOut] + ((uint64_t)m * pc.list_stride + pc.offset) * 4;
    if (kk >= n) {
        for (uint t = tid; t < n; t += kWg)
            *(int*)(out + (uint64_t)t * 4) = int(t + pc.offset);
        return;
    }
    uint prefix, need;
    radix_threshold(sc, false, n, kk, tid, prefix, need);
    const uint chunk = (n + kWg - 1) / kWg;
    const uint lo = min(tid * chunk, n), hi = min(lo + chunk, n);
    uint ties = 0u;
    for (uint t = lo; t < hi; ++t) if (sel_key_at(sc, false, t) == prefix) ++ties;
    const uint tieBase = wg_exscan(ties, tid);
    uint taken = 0u, tie = tieBase;
    for (uint t = lo; t < hi; ++t) {
        const uint key = sel_key_at(sc, false, t);
        if (key > prefix) ++taken;
        else if (key == prefix) { if (tie < need) ++taken; ++tie; }
    }
    const uint outBase = wg_exscan(taken, tid);
    uint o = outBase;
    tie = tieBase;
    for (uint t = lo; t < hi; ++t) {
        const uint key = sel_key_at(sc, false, t);
        bool keep = key > prefix;
        if (key == prefix) { keep = tie < need; ++tie; }
        if (keep) {
            *(int*)(out + (uint64_t)o * 4) = int(t + pc.offset);
            ++o;
        }
    }
}

// --- stages 6-8: design §2.1, per query ------------------------------------------
void stage_block_keys(uint gid, uint m, uint tid) {
    const uint n = n_pos(m), bs = pc.k;
    const uint nb = (n + bs - 1) / bs;
    const uint b = gid * kWg + tid;
    if (b >= nb) return;
    const uint64_t sc = Ptr[kScore] + (uint64_t)m * pc.score_stride * 4;
    uint key = 0xFFFFFFFFu;
    if (b != nb - 1) {
        key = 0u;
        const uint t0 = b * bs;
        const uint t1 = min(t0 + bs, n);
        for (uint t = t0; t < t1; ++t) key = max(key, score_key(load_f32(sc + (uint64_t)t * 4)));
    }
    *(uint*)(Ptr[kBlkKey] + ((uint64_t)m * pc.blk_stride + b) * 4) = key;
}

void stage_block_select(uint m, uint tid) {
    const uint n = n_pos(m), bs = pc.k;
    const uint nb = (n + bs - 1) / bs;
    const uint kk = min(pc.topk, nb);
    const uint chunk = (nb + kWg - 1) / kWg;
    const uint lo = min(tid * chunk, nb), hi = min(lo + chunk, nb);
    const uint64_t keys = Ptr[kBlkKey] + (uint64_t)m * pc.blk_stride * 4;
    const uint64_t cand = Ptr[kCand] + (uint64_t)m * pc.blk_stride * 4;
    if (kk >= nb) {
        for (uint b = lo; b < hi; ++b) *(uint*)(cand + (uint64_t)b * 4) = 1u;
        return;
    }
    uint prefix, need;
    radix_threshold(keys, true, nb, kk, tid, prefix, need);
    uint ties = 0u;
    for (uint b = lo; b < hi; ++b) if (sel_key_at(keys, true, b) == prefix) ++ties;
    const uint tieBase = wg_exscan(ties, tid);
    uint tie = tieBase;
    for (uint b = lo; b < hi; ++b) {
        const uint key = sel_key_at(keys, true, b);
        bool keep = key > prefix;
        if (key == prefix) { keep = tie < need; ++tie; }
        *(uint*)(cand + (uint64_t)b * 4) = keep ? 1u : 0u;
    }
}

void stage_apply_candidates(uint m, uint tid) {
    const uint n = n_pos(m), bs = pc.k;
    const uint chunk = (n + kWg - 1) / kWg;
    const uint lo = min(tid * chunk, n), hi = min(lo + chunk, n);
    const uint64_t sc = Ptr[kScore] + (uint64_t)m * pc.score_stride * 4;
    const uint64_t cand = Ptr[kCand] + (uint64_t)m * pc.blk_stride * 4;
    for (uint t = lo; t < hi; ++t)
        if (load_u32(cand + (uint64_t)(t / bs) * 4) == 0u)
            *(uint*)(sc + (uint64_t)t * 4) = 0xFF800000u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 1)      stage_q_finish(gid.y, tid);
    else if (Stage == 2) stage_key(gid.y, tid);
    else if (Stage == 3) stage_weights(gid.y, tid);
    else if (Stage == 4) stage_score(gid.x, gid.y, tid);
    else if (Stage == 5) stage_topk(gid.y, tid);
    else if (Stage == 6) stage_block_keys(gid.x, gid.y, tid);
    else if (Stage == 7) stage_block_select(gid.y, tid);
    else                 stage_apply_candidates(gid.y, tid);
}
