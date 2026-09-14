// indexer -- the compressed-position selector of design §7.4, the second of the
// two things docs/p2_attention.md §6 listed as "LOADED" rather than produced.
//
// A small side attention: 32 fp4 query heads of 128 dims against ONE shared
// fp4 key per compressed position, the per-head scores rectified and combined
// by `weights_proj`, then the best `index_topk` positions kept. It is the only
// decode kernel whose work grows with the context: at 64K and ratio 2 there are
// 32K positions, 32 x 128 each, which is 134 MFLOP and 2 MB of keys.
//
// Six dispatches (`Stage`):
//
//   0  `indexer.wq_b`: fp8 [4096 x 1280] with `q_norm` and the `act_quant`
//      round trip fused, exactly like the main wq_b. Output is pre-RoPE fp32.
//   1  RoPE over the last `rope_dim` of each 128-dim head, then FP4 block-32
//      with a UE8M0 scale -- the indexer's quantisation, NOT the compressed
//      KV's (block 16, E4M3). One 32-element block a thread.
//   2  the index KEY, on a layer that owns one: `k_norm(wk(latent))` off the
//      compressor's PRE-RoPE latent, rotated at the latent's own position and
//      FP4-quantised the same way, into the key cache.
//   3  `weights_proj`: bf16 [32 x 5120] against the sublayer input, scaled by
//      `softmax_scale * n_heads ** -0.5`.
//   4  the scores: `sum_h relu(q[h] . k[t]) * w[h]`, one position a wave.
//   5  the top-k, written into the compressed half of `topk_idxs`.
//
// and, past design §2.1's `candidate_topk_blocks * candidate_block_size` =
// 16,384 compressed positions, the two-level top-k (`select_candidate_blocks`):
//
//   6  the candidate SOURCE (layer 20): one key per block of
//      `candidate_block_size` positions, the block's best score, with the block
//      holding the query's newest position pinned above everything.
//   7  the candidate source: the `candidate_topk_blocks` best blocks, as a
//      keep flag per block.
//   8  a candidate CONSUMER (index layers 24..36): every score outside a kept
//      block becomes -inf before its own stage 5, which is model.py's
//      `index_score.masked_fill(~shared_attn.candidates, -inf)`.
//
// Below the threshold every block is kept and 6-8 are the identity, so the
// host does not dispatch them.
//
// What the reference decides, and this reproduces
// ----------------------------------------------
//
// * **The key owner is the layer that compresses, not the layer that scores.**
//   Stage 2 runs only on a `kv_source_layer` whose group just completed; every
//   other index source reads the cache that layer published. A ratio-2 source
//   at a position where its group does NOT complete publishes nothing and
//   scores against whatever key cache was last published -- which is layer
//   20's, not its own. `model.py` calls that deliberate ("every source writes
//   before its consumers read") and the oracle reproduces it, so the host, not
//   this kernel, decides which cache address goes in the slot.
//
// * **Two FP4 formats in one layer.** The indexer is block 32 with a
//   power-of-two scale; the compressed KV is block 16 with an E4M3 scale. They
//   differ in the amax floor as well (6*2^-126 against 6*2^-9).
//
// * **`weights_proj` sees the sublayer input `x`, not `qr`.** Only the query
//   projection is low-rank.
//
// * **Ties in the top-k.** `index_score.topk(k, sorted=False).indices.sort()`
//   leaves torch to break exact ties however its kernel happens to; stages 5
//   and 7 take the lowest position (block) indices among equal scores. Past
//   1,024 tokens of context bf16 ties at the 512th score are the NORMAL case
//   (docs/p3_longctx.md §4.3), so any comparison against the reference has to
//   be tie-aware (tests/test_decode_longctx.cpp).

// The indexer's own wq_b is [4096 x 1280]: q_lora_rank, the same K as the main
// wq_b, so the same small LDS budget (fp8_gemv.slang).
#define DEEPMOE_GEMV_MAX_K 1280

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kQr = 2, kQNormW = 3, kRope = 4;
static const uint kQRaw = 5, kQ = 6, kWk = 7, kKNormW = 8, kLatent = 9;
static const uint kKRaw = 10, kKCache = 11, kKFp4 = 12, kKScale = 13;
static const uint kWProjW = 14, kX = 15, kWeights = 16, kScore = 17, kOut = 18;
static const uint kQFp4 = 19, kQScale = 20;
// §2.1's candidate blocks: [n_blocks] uint32 block keys (stage 6 -> 7) and
// [n_blocks] uint32 keep flags (stage 7 -> 8, read by every consumer layer).
static const uint kBlkKey = 21, kCand = 22;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct IdxPush {
    uint  rows;        // stage 0: n_heads * index_head_dim = 4096; stage 3: 32
    uint  k;           // stage 0: 1280; stage 2: head_dim 512; stage 3: dim 5120;
                       // stages 6-8: candidate_block_size
    uint  scale_cols;  // stage 0: k / 32 = 40
    uint  n_heads;     // index_n_heads = 32
    uint  head_dim;    // index_head_dim = 128
    uint  rope_dim;    // rope_head_dim = 64
    uint  n_pos;       // compressed positions visible to this query
    uint  topk;        // min(index_topk, n_pos); stage 7: candidate_topk_blocks
    uint  offset;      // window_size: where the compressed half starts, in both
                       // the topk_idxs array and the values written into it
    uint  k_row;       // start_pos / ratio: the key-cache row stage 2 writes
    float norm_eps;    // 1e-20
    float wscale;      // softmax_scale * n_heads ** -0.5
};
[[vk::push_constant]] IdxPush pc;

// One pool, carved differently per stage, because LDS is allocated per pipeline
// and this .spv carries six of them: stage 3 stages a 5120-wide activation,
// stage 4 wants the 4096 query floats plus a tile of eight 128-wide keys, and
// 4096 + 1024 is the same 5120. Sizing them separately would cost 36 KiB of LDS
// in every stage including the only one that reads real weight bandwidth.
static const uint kPool     = 5120;
static const uint kMaxHeads = 32;         // index_n_heads
static const uint kKTile    = 8;          // positions a workgroup scores at once
static const uint kIdxBlock = 32;         // the indexer's FP4 block
static const float kFp4Max  = 6.0f;
groupshared float gPool[kPool];
groupshared uint  gHist[256];
groupshared uint  gScan[256];
groupshared uint  gSel[4];

// --- stage 0: indexer.wq_b, with q_norm and act_quant fused -----------------
void stage_q_gemv(uint gid, uint tid) {
    fp8_table_fill(tid);
    stage_x_fp8_rmsnorm(Ptr[kQr], Ptr[kQNormW], pc.k, tid, pc.norm_eps,
                        Ptr[kS] + (uint64_t)((gid * gemv_rows_per_wg()) >> 5) * pc.scale_cols);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = gemv_row_base(gid, tid);
    float acc[kMaxRowsPerLane];
    gemv8_rows(Ptr[kW], Ptr[kS], rowBase, pc.k, pc.scale_cols, sublane, acc);
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float y = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kQRaw] + (uint64_t)(rowBase + r) * 4) = y;
    }
}

// FP4 with a UE8M0 scale, one `kIdxBlock`-element block, in place.
// `fp4_quant_kernel`'s non-E4M3 branch: floor the amax at 6 * 2^-126 and round
// the scale UP to a power of two, which `act_scale`'s sibling does off the
// exponent field.
float idx_fp4_scale(float amax) {
    const float v = max(amax, kFp4Max * 5.877472e-39f) * (1.0f / kFp4Max);
    const uint  b = asuint(v);
    const int   e = int((b >> 23) & 0xFFu) - 127 + (((b & 0x7FFFFFu) != 0u) ? 1 : 0);
    return ldexp(1.0f, e);
}

// Rotate and FP4-quantise one 32-element block of a `head_dim`-wide row, then
// write the dequantised value as bf16 and the packed nibbles and scale byte
// beside it. `d0` is the block's offset inside the row.
void rope_quant_block(uint64_t src, uint64_t dstVal, uint64_t dstFp4,
                      uint64_t dstScale, uint blk_in_row, uint d0) {
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    float v[kIdxBlock];
    [unroll] for (uint i = 0; i < kIdxBlock; ++i)
        v[i] = load_f32(src + (uint64_t)i * 4);
    [unroll] for (uint i = 0; i < kIdxBlock; i += 2) {
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
        const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
        rope_pair(c, s, v[i], v[i + 1], false);
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

// --- stage 1: RoPE + FP4 on q ----------------------------------------------
void stage_q_finish(uint tid) {
    const uint blocks = pc.rows / kIdxBlock;          // 4096 / 32 = 128
    if (tid >= blocks) return;
    const uint per_row = pc.head_dim / kIdxBlock;     // 4
    const uint d0 = (tid % per_row) * kIdxBlock;
    // q's packed plane and its scales go to their own buffers: the score stage
    // reads the dequantised bf16, so nothing downstream needs them, but sharing
    // the key's would mean the two stages' write order decided what the test
    // sees.
    rope_quant_block(Ptr[kQRaw] + (uint64_t)tid * kIdxBlock * 4,
                     Ptr[kQ] + (uint64_t)tid * kIdxBlock * 2,
                     Ptr[kQFp4] + (uint64_t)tid * (kIdxBlock / 2),
                     Ptr[kQScale], tid, d0);
}

// --- stage 2: the index key -------------------------------------------------
// One workgroup: the latent into LDS, one output row a thread, then the RMS
// statistic over all `head_dim` of them, then four FP4 blocks.
void stage_key(uint tid) {
    for (uint i = tid; i < pc.k; i += kWg) gPool[i] = load_f32(Ptr[kLatent] + (uint64_t)i * 4);
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
        // wk is a bf16 Linear, so k_norm sees a bf16 value.
        dot = bf16_round(dot);
    }
    const float rs = rsqrt(wg_reduce(dot * dot, tid) / float(pc.head_dim) + pc.norm_eps);
    if (tid < pc.head_dim)
        *(float*)(Ptr[kKRaw] + (uint64_t)tid * 4) =
            bf16_round(dot * rs * bf16_decode(load_u16(Ptr[kKNormW] + (uint64_t)tid * 2)));
    GroupMemoryBarrierWithGroupSync();

    const uint blocks = pc.head_dim / kIdxBlock;      // 4
    if (tid >= blocks) return;
    const uint64_t base = Ptr[kKCache] + (uint64_t)pc.k_row * pc.head_dim * 2;
    rope_quant_block(Ptr[kKRaw] + (uint64_t)tid * kIdxBlock * 4,
                     base + (uint64_t)tid * kIdxBlock * 2,
                     Ptr[kKFp4] + (uint64_t)tid * (kIdxBlock / 2),
                     Ptr[kKScale], tid, tid * kIdxBlock);
}

// --- stage 3: weights_proj --------------------------------------------------
void stage_weights(uint tid) {
    for (uint i = tid; i < pc.k; i += kWg) gPool[i] = load_f32(Ptr[kX] + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowsAtOnce = kWg / LanesPerRow;
    for (uint rb = 0; rb < pc.rows; rb += rowsAtOnce) {
        const uint row = rb + tid / LanesPerRow;
        float acc = 0.0f;
        if (row < pc.rows) {
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
        // `weights_proj(x) * (softmax_scale * n_heads ** -0.5)`, both halves in
        // bf16 because the module and the tensor it multiplies are.
        if (sublane == 0 && row < pc.rows)
            *(float*)(Ptr[kWeights] + (uint64_t)row * 4) = bf16_round(bf16_round(dot) * pc.wscale);
    }
}

// --- stage 4: the scores ----------------------------------------------------
// `index_score[t] = sum_h relu(q[h] . k[t]) * w[h]`, which is
// `torch.einsum("bshd,btd->bsht", q, index_k)` rectified, weighted and summed
// over the head axis.
//
// One workgroup covers kKTile positions. Thread `tid` is (position tid /
// n_heads, head tid % n_heads), so the 32 heads of one position are the 32
// lanes of one Wave32 and the head sum is a single `WaveActiveSum` --
// `row_reduce` with LanesPerRow == n_heads == the subgroup size. The keys go
// through LDS because all 32 heads of a position read the same 128 values.
void stage_score(uint gid, uint tid) {
    const uint hd = pc.head_dim, nh = pc.n_heads;
    for (uint i = tid; i < nh * hd; i += kWg)
        gPool[i] = bf16_decode(load_u16(Ptr[kQ] + (uint64_t)i * 2));
    const uint t0 = gid * kKTile;
    for (uint i = tid; i < kKTile * hd; i += kWg) {
        const uint t = t0 + i / hd;
        gPool[nh * hd + i] = (t < pc.n_pos)
            ? bf16_decode(load_u16(Ptr[kKCache] + ((uint64_t)t * hd + i % hd) * 2))
            : 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint p = tid / nh, h = tid % nh;
    float dot = 0.0f;
    if (p < kKTile) {
        for (uint d = 0; d < hd; ++d)
            dot += gPool[h * hd + d] * gPool[nh * hd + p * hd + d];
    }
    // Three bf16 roundings, because all three tensors the reference computes
    // here are bf16: `einsum` on bf16 inputs accumulates in fp32 and casts the
    // result, `* weights` is bf16 x bf16, and `.sum(dim=2)` over the head axis
    // casts back again. Keeping fp32 through them cost a factor of five on the
    // agreement (cosine 0.99999 against 0.9999998) for no precision anyone
    // downstream can use -- the same argument attn_common.slang's `bf16_round`
    // makes in front of a quantiser, one step removed.
    const float contrib = bf16_round(max(bf16_round(dot), 0.0f) *
                                     load_f32(Ptr[kWeights] + (uint64_t)h * 4));
    const float s = bf16_round(row_reduce(contrib, tid, h));
    if (h == 0 && p < kKTile && t0 + p < pc.n_pos)
        *(float*)(Ptr[kScore] + (uint64_t)(t0 + p) * 4) = s;
}

// --- stage 5: the top-k -----------------------------------------------------
// A float's bit pattern ordered as an unsigned key: flip every bit of a
// negative, set the sign bit of a positive. Monotone, so a radix select on the
// key is a select on the float.
uint score_key(float f) {
    const uint u = asuint(f);
    return ((u & 0x80000000u) != 0u) ? ~u : (u | 0x80000000u);
}
uint score_key_at(uint t) { return score_key(load_f32(Ptr[kScore] + (uint64_t)t * 4)); }
// Stage 5 selects over the score plane; stage 7 over stage 6's block keys,
// which are already keys (the pinned block is 0xFFFFFFFF, above +inf's key).
uint sel_key_at(bool blocks, uint t) {
    return blocks ? load_u32(Ptr[kBlkKey] + (uint64_t)t * 4) : score_key_at(t);
}

// Exclusive prefix sum of `v` over the 256 threads; returns this thread's
// exclusive sum and leaves the total in gSel[3].
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

// Four 8-bit radix passes narrow the key of the kk-th largest of `n` keys.
// Leaves that key in `prefix_out` and how many of the positions holding it
// are wanted in `need_out`. Every thread runs the same barriers, so any
// single-workgroup stage can call it.
void radix_threshold(bool blocks, uint n, uint kk, uint tid, out uint prefix_out,
                     out uint need_out) {
    uint prefix = 0u, mask = 0u, need = kk;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24u - 8u * pass;
        gHist[tid] = 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint t = tid; t < n; t += kWg) {
            const uint key = sel_key_at(blocks, t);
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
            gSel[1] = need - cum;      // still wanted from inside `bin`
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

void stage_topk(uint tid) {
    const uint n = pc.n_pos;
    const uint kk = min(pc.topk, n);

    // Every position selected -- every context short enough that
    // n <= index_topk.
    if (kk >= n) {
        for (uint t = tid; t < n; t += kWg)
            *(int*)(Ptr[kOut] + (uint64_t)(pc.offset + t) * 4) = int(t + pc.offset);
        return;
    }

    uint prefix, need;
    radix_threshold(false, n, kk, tid, prefix, need);
    // `prefix` is now the exact key of the kk-th largest score and `need` is how
    // many of the positions holding it are wanted.

    // Ascending index order, in three chunked passes: the tie rank of each
    // equal-keyed position, then the output rank of each selected one, then the
    // write. A single thread walking 32K positions would be 65 us.
    const uint chunk = (n + kWg - 1) / kWg;
    const uint lo = min(tid * chunk, n), hi = min(lo + chunk, n);

    uint ties = 0u;
    for (uint t = lo; t < hi; ++t) if (score_key_at(t) == prefix) ++ties;
    const uint tieBase = wg_exscan(ties, tid);

    uint taken = 0u, tie = tieBase;
    for (uint t = lo; t < hi; ++t) {
        const uint key = score_key_at(t);
        if (key > prefix) ++taken;
        else if (key == prefix) { if (tie < need) ++taken; ++tie; }
    }
    const uint outBase = wg_exscan(taken, tid);

    uint o = outBase;
    tie = tieBase;
    for (uint t = lo; t < hi; ++t) {
        const uint key = score_key_at(t);
        bool keep = key > prefix;
        if (key == prefix) { keep = tie < need; ++tie; }
        if (keep) {
            *(int*)(Ptr[kOut] + (uint64_t)(pc.offset + o) * 4) = int(t + pc.offset);
            ++o;
        }
    }
}

// --- stage 6: candidate block keys (design §2.1, level one) -----------------
// `scores.unflatten(-1, (-1, block_size)).amax(dim=-1)` -- the last block is
// padded with -inf, i.e. simply shorter here -- then the block holding the
// query's newest position, `(n_pos - 1) // block_size`, which at decode is the
// LAST block, pinned to +inf so an older full block can never outscore it.
// Stored as radix keys: the max of the keys is the key of the max, and the
// pinned block gets 0xFFFFFFFF, above the key of any float including +inf.
// One thread a block, `ceil(n_blocks / 256)` workgroups, no barriers.
void stage_block_keys(uint gid, uint tid) {
    const uint n = pc.n_pos, bs = pc.k;
    const uint nb = (n + bs - 1) / bs;
    const uint b = gid * kWg + tid;
    if (b >= nb) return;
    uint key = 0xFFFFFFFFu;
    if (b != nb - 1) {
        key = 0u;
        const uint t0 = b * bs;
        const uint t1 = min(t0 + bs, n);
        for (uint t = t0; t < t1; ++t) key = max(key, score_key_at(t));
    }
    *(uint*)(Ptr[kBlkKey] + (uint64_t)b * 4) = key;
}

// --- stage 7: the candidate_topk_blocks best blocks, as keep flags ----------
// `scores.topk(min(topk_blocks, num_blocks))` scattered into a bool mask. The
// `top.values > -inf` clause cannot drop anything at decode: all n_pos
// positions are reachable, so every block key is a finite score's.
void stage_block_select(uint tid) {
    const uint n = pc.n_pos, bs = pc.k;
    const uint nb = (n + bs - 1) / bs;
    const uint kk = min(pc.topk, nb);
    const uint chunk = (nb + kWg - 1) / kWg;
    const uint lo = min(tid * chunk, nb), hi = min(lo + chunk, nb);
    if (kk >= nb) {
        for (uint b = lo; b < hi; ++b) *(uint*)(Ptr[kCand] + (uint64_t)b * 4) = 1u;
        return;
    }
    uint prefix, need;
    radix_threshold(true, nb, kk, tid, prefix, need);

    uint ties = 0u;
    for (uint b = lo; b < hi; ++b) if (sel_key_at(true, b) == prefix) ++ties;
    const uint tieBase = wg_exscan(ties, tid);
    uint tie = tieBase;
    for (uint b = lo; b < hi; ++b) {
        const uint key = sel_key_at(true, b);
        bool keep = key > prefix;
        if (key == prefix) { keep = tie < need; ++tie; }
        *(uint*)(Ptr[kCand] + (uint64_t)b * 4) = keep ? 1u : 0u;
    }
}

// --- stage 8: level two, mask a consumer's scores ---------------------------
// Chunked over 256 threads like stage 5. -inf's key is below every finite
// score's, and at least `(topk_blocks - 1) * block_size + 1` >= index_topk
// positions stay unmasked, so stage 5 never selects a masked one.
void stage_apply_candidates(uint tid) {
    const uint n = pc.n_pos, bs = pc.k;
    const uint chunk = (n + kWg - 1) / kWg;
    const uint lo = min(tid * chunk, n), hi = min(lo + chunk, n);
    for (uint t = lo; t < hi; ++t)
        if (load_u32(Ptr[kCand] + (uint64_t)(t / bs) * 4) == 0u)
            *(uint*)(Ptr[kScore] + (uint64_t)t * 4) = 0xFF800000u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_q_gemv(gid.x, tid);
    else if (Stage == 1) stage_q_finish(tid);
    else if (Stage == 2) stage_key(tid);
    else if (Stage == 3) stage_weights(tid);
    else if (Stage == 4) stage_score(gid.x, tid);
    else if (Stage == 5) stage_topk(tid);
    else if (Stage == 6) stage_block_keys(gid.x, tid);
    else if (Stage == 7) stage_block_select(tid);
    else                 stage_apply_candidates(tid);
}
