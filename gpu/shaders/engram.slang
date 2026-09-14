// engram -- the n-gram lookup layers 1 and 14 write into the residual stream
// (design §2.4, §7.10; the two dispatches of §7.14 that only two layers run).
//
// What the reference does
// -----------------------
// `Engram.forward` in inference/model.py, in full:
//
//     kv          = wkv(embed(hash_ids).flatten(-2))      [25600]
//     key, value  = kv.split([hc*dim, dim])               [4][5120], [5120]
//     weight      = q_weight * k_weight                   [4][5120], fp32
//     rstd[j]     = rsqrt(mean(h[j]^2) + eps) * rsqrt(mean(key[j]^2) + eps)
//     dot[j]      = sum_d h[j][d] * weight[j][d] * key[j][d] * rstd[j] * dim^-0.5
//     gate[j]     = sigmoid(copysign(sqrt(clamp_min(|dot[j]|, 1e-6)), dot[j]))
//     out[j][d]   = h[j][d] + gate[j] * value[d]
//
// `embed` is 24 rows of a 384M-row FP8 table that lives on NVMe and is fetched
// per token (design §9.6 P2). The host hands this kernel the raw 24 x 256 E4M3
// bytes and their 24 x 8 UE8M0 scales; the decode is fused into the activation
// staging, which is also where `act_quant` happens -- `wkv` is an fp8 `Linear`,
// so its activation makes the round trip every other fp8 Linear makes
// (attn_common.slang, docs/p2_attention.md §2.1).
//
// Three details worth stating
// ---------------------------
// 1. **The row's own UE8M0 block and `act_quant`'s block are both 32 wide and
//    they line up**: a row is 256 values = 8 scale blocks, and 6144 = 24 x 256,
//    so block `b` of the flattened input is (row b/8, sub-block b%8). The
//    dequantised value is therefore re-quantised against an amax drawn from
//    exactly the same 32 numbers -- which is what the reference does too, since
//    it materialises the bf16 row and then calls `linear`.
// 2. **The two RMS statistics are per (token, hc copy)**, over `dim` and NOT
//    jointly over the copies. One workgroup per copy is the whole of stage 1.
// 3. **`clamp_min` is on the magnitude, and the sign is restored afterwards**,
//    so a dot of exactly 0 comes out as sigmoid(+1e-3), not sigmoid(0).
//
// Stage 0  rows/(WG/LanesPerRow) workgroups: the 6144 -> 25600 fp8 GEMV.
// Stage 1  hc workgroups: the gate and the residual update.
//
// The staging and the GEMV inner loop are written out here rather than taken
// from fp8_gemv.slang because this kernel's activation does not arrive as
// fp32 -- it arrives as the E4M3 bytes of 24 hash rows, and the decode belongs
// inside the staging pass that is already holding them in registers.

#include "attn_common.slang"

static const uint kRowVal = 0;   // [24][256] E4M3, the fetched hash rows
static const uint kRowSc  = 1;   // [24][8]   UE8M0, one per 32 values
static const uint kW      = 2;   // wkv [25600][6144] E4M3
static const uint kS      = 3;   // wkv scales [800][192] UE8M0 (32x32 tiles)
static const uint kKv     = 4;   // [25600] fp32: key[4][5120] then value[5120]
static const uint kX      = 5;   // [hc][dim] fp32, the residual stream in
static const uint kQW     = 6;   // q_weight [hc][dim] bf16
static const uint kKW     = 7;   // k_weight [hc][dim] bf16
static const uint kXout   = 8;   // [hc][dim] fp32, the residual stream out

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct EngramPush {
    uint  rows;        // 25600 = dim * (hc + 1)
    uint  k;           // 6144 = 24 rows x 256
    uint  scale_cols;  // k / 32 = 192
    uint  dim;         // 5120
    uint  hc;          // 4
    float norm_eps;
};
[[vk::push_constant]] EngramPush pc;

static const uint kEngramK      = 6144;
static const uint kEngramBlocks = kEngramK / kBlockK;   // 192
static const float kGateFloor   = 1e-6f;                // Engram.clamp_value

groupshared uint  gXQ[kEngramK / 2];     // the staged activation, two bf16 a word
groupshared float gXS[kEngramBlocks];    // its per-32 power-of-two scale

// One 32-element block: 32 E4M3 bytes of one hash row, dequantised with that
// row's UE8M0 scale, rounded to bf16 the way `ParallelEngramEmbedding.forward`
// returns it, then `act_quant`ed for the fp8 GEMM. One thread a block; 192 of
// the 256 threads are live, which is the shape of the data and not a choice.
void stage_rows(uint tid) {
    if (tid >= kEngramBlocks) return;
    const uint row = tid >> 3, sub = tid & 7u;       // 8 blocks of 32 per 256-wide row
    const uint64_t vaddr = Ptr[kRowVal] + (uint64_t)row * 256 + (uint64_t)sub * 32;
    const int e = (int)load_u8(Ptr[kRowSc] + (uint64_t)row * 8 + sub) - 127;

    float v[32];
    float amax = 0.0f;
    [unroll] for (uint i = 0; i < 2; ++i) {
        const uint4 w = load16(vaddr + (uint64_t)i * 16);
        [unroll] for (uint j = 0; j < 4; ++j) {
            const uint b = w[j];
            [unroll] for (uint n = 0; n < 4; ++n) {
                const float d = bf16_round(ldexp(gTbl[(b >> (8 * n)) & 0xFFu], e));
                v[i * 16 + j * 4 + n] = d;
                amax = max(amax, abs(d));
            }
        }
    }
    float s = 1.0f;
    if (ActQuant != 0) {
        s = act_scale(amax);
        const float inv = 1.0f / s;
        [unroll] for (uint i = 0; i < 32; ++i)
            v[i] = fp8_round(clamp(v[i] * inv, -kFp8MaxF, kFp8MaxF));
    }
    [unroll] for (uint i = 0; i < 16; ++i)
        gXQ[tid * 16 + i] = bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
    gXS[tid] = s;
}

void stage_gemv(uint gid, uint tid) {
    // Two barriers, not one. `stage_rows` DECODES through gTbl, so the table
    // has to be complete before it runs -- thread t fills gTbl[t] and then
    // reads whatever byte its row happens to hold, which is another thread's
    // entry. Within a single Wave32 the two are in lockstep and entries 0..31
    // are safe, which is exactly why skipping this barrier does not fail: it
    // produces slightly wrong engram values, non-deterministically, depending
    // on how the eight waves interleave. Symptom, when it was missing: the
    // same eight-token decode gave different tokens run to run, and
    // occasionally NaN logits.
    fp8_table_fill(tid);
    GroupMemoryBarrierWithGroupSync();
    stage_rows(tid);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = gid * (kWg / LanesPerRow) + tid / LanesPerRow;
    const uint blocks = pc.k / kBlockK;
    const uint64_t wrow = Ptr[kW] + (uint64_t)row * (uint64_t)pc.k;
    const uint64_t srow = Ptr[kS] + (uint64_t)(row >> 5) * pc.scale_cols;

    float acc = 0.0f;
    if (row < pc.rows) {
        for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
            // The staged block into registers: sixteen LDS words rather than
            // thirty-two indexed reads (fp8_gemv.slang makes the same trade).
            float x[32];
            [unroll] for (uint i = 0; i < 16; ++i) {
                const uint w = gXQ[blk * 16 + i];
                x[i * 2]     = bf16_decode(w & 0xFFFFu);
                x[i * 2 + 1] = bf16_decode(w >> 16);
            }
            const uint4 w0 = load16(wrow + (uint64_t)blk * 32);
            const uint4 w1 = load16(wrow + (uint64_t)blk * 32 + 16);
            const int   e  = (int)load_u8(srow + blk) - 127;
            float p = 0.0f;
            [unroll] for (uint i = 0; i < 4; ++i) {
                const uint a = w0[i], b = w1[i];
                [unroll] for (uint j = 0; j < 4; ++j) {
                    p += gTbl[(a >> (8 * j)) & 0xFFu] * x[i * 4 + j];
                    p += gTbl[(b >> (8 * j)) & 0xFFu] * x[16 + i * 4 + j];
                }
            }
            acc += ldexp(p, e) * gXS[blk];
        }
    }
    const float dot = row_reduce(acc, tid, sublane);
    if (sublane == 0 && row < pc.rows)
        *(float*)(Ptr[kKv] + (uint64_t)row * 4) = dot;
}

// One workgroup per hc copy. The two RMS statistics and the weighted dot are
// three reductions over `dim`; the update that follows needs only this copy's
// gate, so nothing crosses workgroups.
void stage_gate(uint j, uint tid) {
    const uint64_t X   = Ptr[kX]  + (uint64_t)j * pc.dim * 4;
    const uint64_t Key = Ptr[kKv] + (uint64_t)j * pc.dim * 4;
    const uint64_t Val = Ptr[kKv] + (uint64_t)pc.hc * pc.dim * 4;
    const uint64_t QW  = Ptr[kQW] + (uint64_t)j * pc.dim * 2;
    const uint64_t KW  = Ptr[kKW] + (uint64_t)j * pc.dim * 2;

    float hsq = 0.0f, ksq = 0.0f, num = 0.0f;
    for (uint d = tid; d < pc.dim; d += kWg) {
        const float h = load_f32(X + (uint64_t)d * 4);
        const float k = load_f32(Key + (uint64_t)d * 4);
        const float w = bf16_decode(load_u16(QW + (uint64_t)d * 2)) *
                        bf16_decode(load_u16(KW + (uint64_t)d * 2));
        hsq += h * h;
        ksq += k * k;
        num += h * w * k;
    }
    const float sh = wg_reduce(hsq, tid);
    const float sk = wg_reduce(ksq, tid);
    const float sn = wg_reduce(num, tid);

    const float rstd = rsqrt(sh / float(pc.dim) + pc.norm_eps) *
                       rsqrt(sk / float(pc.dim) + pc.norm_eps);
    const float dot  = sn * rstd * rsqrt(float(pc.dim));
    // `copysign(|dot|.clamp_min(1e-6).sqrt(), dot)`: the floor is on the
    // magnitude and the sign goes back on afterwards.
    const float mag  = sqrt(max(abs(dot), kGateFloor));
    const float sdot = (dot < 0.0f) ? -mag : mag;
    const float gate = 1.0f / (1.0f + exp(-sdot));

    const uint64_t Out = Ptr[kXout] + (uint64_t)j * pc.dim * 4;
    for (uint d = tid; d < pc.dim; d += kWg) {
        const float h = load_f32(X + (uint64_t)d * 4);
        const float v = load_f32(Val + (uint64_t)d * 4);
        *(float*)(Out + (uint64_t)d * 4) = h + gate * v;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0) stage_gemv(gid.x, tid);
    else            stage_gate(gid.x, tid);
}
