// compressor -- the KV compressor of design §7.4, the first of the two things
// docs/p2_attention.md §6 listed as "LOADED" rather than produced.
//
// `Compressor.forward` pools `compress_ratio` consecutive tokens into one KV
// latent with a learned softmax gate, and returns None while a group is still
// filling up. Three dispatches, in the order `Attention._compress_kv` imposes:
//
//   Stage 0  `wkv` (and, at ratio > 1, `wgate`): bf16 [512 x 5120] GEMV. Two
//            dispatches of this stage with different slots, the way mega_mhc
//            runs three times.
//   Stage 1  the state write and the pooling: slot `start_pos % ratio` of
//            `kv_state` / `score_state` takes this token, and when the group
//            just completed the two states collapse into one latent through a
//            softmax over the ratio axis. Then `.to(bf16)` and the RMSNorm.
//            Output: the latent BEFORE RoPE, which is what the indexer needs.
//   Stage 2  RoPE over the last `rope_dim` dims, the FP4 quantisation, and the
//            write into the compressed-KV cache.
//
// Three things the reference decides that are not obvious
// ------------------------------------------------------
//
// 1. **The softmax is over the ratio axis, per element.** `kv_state` and
//    `score_state` are both [ratio, head_dim] and the pooling is
//    `(kv_state * score_state.softmax(dim=1)).sum(dim=1)`, so each of the 512
//    dims gets its OWN two-way (or ratio-way) softmax. It is not one weight
//    per token.
//
// 2. **The dtype changes twice.** At ratio > 1 the reference runs `x.float()`
//    and holds both projections and the pooling in fp32 -- the checkpoint's
//    bf16 weights are promoted at load -- then rounds to bf16 exactly once,
//    where `kv.to(dtype)` feeds the norm. At ratio 1 there is no gate and no
//    fp32: `self.norm(self.wkv(x))` is a plain bf16 Linear. So the bf16
//    rounding sits in a different place in the two paths, and this kernel
//    rounds where the reference does (see `bf16_round` in attn_common.slang
//    for why being more precise in front of a quantiser costs agreement).
//
// 3. **The latent stands for the FIRST token of its group**, so it is rotated
//    at position `start_pos + 1 - ratio`, not at `start_pos`. The host builds
//    the table; this kernel only reads (cos, sin) pairs.
//
// Storage. The compressed half is kept as BF16 values -- the numbers the FP4
// quantiser produced -- because that is what sparse_attn.slang reads and
// changing it is a cross-track interface change. The packed E2M1 nibbles and
// their E4M3 scale bytes of design §11.3 are written to two SEPARATE buffers
// alongside, which costs 288 B a latent and is what `tests/test_gpu_attn.cpp`
// compares against the oracle's `latent_fp4` / `latent_scale_e4m3`. When the
// packed form becomes the cache format, the bf16 plane is the one that goes.

#include "attn_common.slang"

static const uint kW = 0, kX = 1, kY = 2, kG = 3;
static const uint kKvState = 4, kScoreState = 5, kNormW = 6, kLatent = 7;
static const uint kRope = 8, kVal = 9, kFp4 = 10, kScaleB = 11, kLatentQ = 12;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct CmpPush {
    uint  rows;        // head_dim = 512
    uint  k;           // dim = 5120
    uint  ratio;       // compress_ratio: 1 (layer 20+) or 2 (layers 2..19)
    uint  slot;        // start_pos % ratio
    uint  complete;    // 1 when (start_pos + 1) % ratio == 0
    uint  rope_dim;    // 64
    uint  cmp_row;     // start_pos / ratio: the row of the compressed cache
    float norm_eps;    // 1e-20
};
[[vk::push_constant]] CmpPush pc;

static const uint kMaxDim = 5120;
groupshared float gXf[kMaxDim];

// --- stage 0: the bf16 projection ------------------------------------------
// Same shape as head.slang: bf16 weights against an fp32 activation, staged
// once per workgroup. `wgate` is the identical call with different slots.
void stage_gemv(uint gid, uint tid) {
    for (uint i = tid; i < pc.k; i += kWg) gXf[i] = load_f32(Ptr[kX] + (uint64_t)i * 4);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = gid * ((kWg / LanesPerRow) * RowsPerLane) +
                         (tid / LanesPerRow) * RowsPerLane;
    float acc[kMaxRowsPerLane];
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) acc[r] = 0.0f;
    for (uint base = sublane * 8; base < pc.k; base += LanesPerRow * 8) {
        float x[8];
        [unroll] for (uint j = 0; j < 8; ++j) x[j] = gXf[base + j];
        [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
            if (r >= RowsPerLane || rowBase + r >= pc.rows) continue;
            const uint4 v = load16(Ptr[kW] + ((uint64_t)(rowBase + r) * pc.k + base) * 2);
            [unroll] for (uint j = 0; j < 4; ++j) {
                acc[r] += bf16_decode(v[j] & 0xFFFFu) * x[j * 2];
                acc[r] += bf16_decode(v[j] >> 16)     * x[j * 2 + 1];
            }
        }
    }
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float dot = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kY] + (uint64_t)(rowBase + r) * 4) = dot;
    }
}

// --- stage 1: state, pooling, norm ------------------------------------------
// One workgroup. `rows` is 512, so two dims a thread at 256 threads; the RMS
// statistic is a whole-vector reduction, which is why this cannot ride along
// with the GEMV.
static const uint kMaxRatio = 8;
static const uint kMaxPer   = 4;   // head_dim <= 1024 at 256 threads

void stage_norm(uint tid) {
    const uint per = (pc.rows + kWg - 1) / kWg;      // 2 at head_dim 512
    float pre[kMaxPer];
    float sq = 0.0f;
    // Unrolled against the compile-time bound so `pre` stays in registers:
    // a loop bounded by a push constant puts it in scratch (p2_attention.md
    // §4.4), which is a 50x mistake, not a 5% one.
    [unroll] for (uint i = 0; i < kMaxPer; ++i) {
        const uint d = tid * per + i;
        pre[i] = 0.0f;
        if (i >= per || d >= pc.rows) continue;
        const float kv = load_f32(Ptr[kY] + (uint64_t)d * 4);
        float v;
        if (pc.ratio <= 1) {
            // A plain bf16 Linear: its result IS bf16 before the norm sees it.
            v = bf16_round(kv);
        } else {
            const float sc = load_f32(Ptr[kG] + (uint64_t)d * 4);
            // Publish this token into the carried group, then pool if it just
            // completed. Both writes happen every step, completion or not.
            *(float*)(Ptr[kKvState] + ((uint64_t)pc.slot * pc.rows + d) * 4) = kv;
            *(float*)(Ptr[kScoreState] + ((uint64_t)pc.slot * pc.rows + d) * 4) = sc;
            if (pc.complete == 0) continue;
            // softmax over the ratio axis, this dim only.
            float mx = -3.0e38f;
            float s[kMaxRatio];
            [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
                if (j >= pc.ratio) { s[j] = 0.0f; continue; }
                s[j] = load_f32(Ptr[kScoreState] + ((uint64_t)j * pc.rows + d) * 4);
                mx = max(mx, s[j]);
            }
            float den = 0.0f, num = 0.0f;
            [unroll] for (uint j = 0; j < kMaxRatio; ++j) {
                if (j >= pc.ratio) continue;
                const float e = exp(s[j] - mx);
                den += e;
                num += e * load_f32(Ptr[kKvState] + ((uint64_t)j * pc.rows + d) * 4);
            }
            // `kv.to(dtype)`: the one bf16 rounding on the ratio > 1 path.
            v = bf16_round(num / den);
        }
        pre[i] = v;
        sq += v * v;
    }
    // Every thread must reach the reduction, completion or not.
    const float rs = rsqrt(wg_reduce(sq, tid) / float(pc.rows) + pc.norm_eps);
    if (pc.ratio > 1 && pc.complete == 0) return;
    [unroll] for (uint i = 0; i < kMaxPer; ++i) {
        const uint d = tid * per + i;
        if (i >= per || d >= pc.rows) continue;
        *(float*)(Ptr[kLatent] + (uint64_t)d * 4) =
            bf16_round(pre[i] * rs * bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)d * 2)));
    }
}

// --- stage 2: RoPE, FP4, the cache write ------------------------------------
// One workgroup, one 16-element FP4 block a thread (32 of them at head_dim
// 512). A RoPE pair never straddles a block because `rope_lo` and the block
// size are both even.
static const uint kCmpBlock = 16;          // design §11.3 / model.py's `16`
static const float kFp4Max  = 6.0f;
static const float kFp4Floor = 6.0f * 0.001953125f;    // 6 * 2^-9

void stage_store(uint tid) {
    const uint blocks = pc.rows / kCmpBlock;
    if (tid >= blocks) return;
    const uint d0 = tid * kCmpBlock;
    const uint rope_lo = pc.rows - pc.rope_dim;

    float v[kCmpBlock];
    [unroll] for (uint i = 0; i < kCmpBlock; ++i)
        v[i] = load_f32(Ptr[kLatent] + (uint64_t)(d0 + i) * 4);
    [unroll] for (uint i = 0; i < kCmpBlock; i += 2) {
        const uint d = d0 + i;
        if (d < rope_lo) continue;
        const uint j = (d - rope_lo) >> 1;
        const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
        const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
        rope_pair(c, s, v[i], v[i + 1], false);
        // apply_rotary_emb writes back into the bf16 tensor it was handed.
        v[i] = bf16_round(v[i]);
        v[i + 1] = bf16_round(v[i + 1]);
    }

    // fp4_act_quant(latent, 16, inplace=True, scale_dtype=float8_e4m3fn):
    // the amax floor keeps an all-zero group's scale nonzero, and the scale
    // itself is an E4M3 number, not a power of two.
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < kCmpBlock; ++i) amax = max(amax, abs(v[i]));
    const float sc  = fp8_decode(fp8_encode_rn(clamp(max(amax, kFp4Floor) / kFp4Max, 0.0f, kFp8MaxF)));
    const float inv = 1.0f / sc;

    uint packed[kCmpBlock / 2];
    [unroll] for (uint z = 0; z < kCmpBlock / 2; ++z) packed[z] = 0u;
    [unroll] for (uint i = 0; i < kCmpBlock; ++i) {
        const float q = fp4_round(clamp(v[i] * inv, -kFp4Max, kFp4Max));
        const uint  code = fp4_code(q);
        if ((i & 1) == 0) packed[i >> 1] = code;
        else              packed[i >> 1] |= code << 4;
        const float out = q * sc;
        *(uint16_t*)(Ptr[kVal] + ((uint64_t)pc.cmp_row * pc.rows + d0 + i) * 2) =
            (uint16_t)bf16_encode(out);
        *(float*)(Ptr[kLatentQ] + (uint64_t)(d0 + i) * 4) = out;
    }
    [unroll] for (uint i = 0; i < kCmpBlock / 2; ++i)
        *(uint8_t*)(Ptr[kFp4] + (uint64_t)(d0 / 2 + i)) = (uint8_t)packed[i];
    *(uint8_t*)(Ptr[kScaleB] + (uint64_t)tid) = (uint8_t)fp8_encode_rn(sc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_gemv(gid.x, tid);
    else if (Stage == 1) stage_norm(tid);
    else                 stage_store(tid);
}
