// mgt1_engram -- engram.slang for a verify batch of M tokens (Track T,
// docs/p4_mgt1.md). Layers 1 and 14 only.
//
// Each token has its own 24 hashed rows ([M][24][256] E4M3 + [M][24][8] UE8M0,
// fetched by the host), so the activation is M columns of 6144 while the
// 157 MB wkv [25600 x 6144] is read once. engram.slang's arithmetic, per token:
// the rows dequantised with their own scale and rounded to bf16, `act_quant`ed
// for the fp8 GEMM, the GEMM, then the gate and the residual update per hc copy.
//
// Stage 0 is cut along the contraction axis into `pc.slices` slices, each its
// own dispatch over the same row groups, run one after another: slice 0 writes
// kv[m][row], later slices add (the same scheme as mgt1_head.slang, and for the
// same reason -- M columns of 6144 would be 78 KiB of LDS). A slice is at most
// 768 elements = 24 blocks, so M * 24 <= 144 threads stage one block each.
//
// Stage 1: the gate, gid = (hc copy j, token m).

#include "attn_common.slang"

static const uint kMaxM = 6;
static const uint kRowVal = 0, kRowSc = 1, kW = 2, kS = 3, kKv = 4;
static const uint kX = 5, kQW = 6, kKW = 7, kXout = 8;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct Mgt1EngramPush {
    uint  rows;        // 25600
    uint  k;           // 6144
    uint  scale_cols;  // 192
    uint  dim;         // 5120
    uint  hc;          // 4
    float norm_eps;
    uint  slice;
    uint  slices;
};
[[vk::push_constant]] Mgt1EngramPush pc;

static const uint kSliceBlk  = 32;                 // LDS column stride, in blocks
static const uint kXStride   = kBlockK / 2 + 1;    // 17
static const float kGateFloor = 1e-6f;
groupshared uint  gXQ[M * kSliceBlk * kXStride];
groupshared float gXS[M * kSliceBlk];

void stage_gemv(uint gid, uint tid) {
    fp8_table_fill(tid);
    GroupMemoryBarrierWithGroupSync();

    const uint kc = pc.k / pc.slices;
    const uint blocks = kc / kBlockK;
    const uint blk0 = (pc.slice * kc) / kBlockK;
    {
        const uint m = tid / kSliceBlk, b = tid % kSliceBlk;
        if (m < M && b < blocks) {
            const uint fb = blk0 + b;                 // block of the flattened 6144
            const uint row = fb >> 3, sub = fb & 7u;
            const uint64_t vaddr = Ptr[kRowVal] + ((uint64_t)m * 24 + row) * 256 + (uint64_t)sub * 32;
            const int e = (int)load_u8(Ptr[kRowSc] + ((uint64_t)m * 24 + row) * 8 + sub) - 127;
            float v[32];
            float amax = 0.0f;
            [unroll] for (uint i = 0; i < 2; ++i) {
                const uint4 w = load16(vaddr + (uint64_t)i * 16);
                [unroll] for (uint j = 0; j < 4; ++j) {
                    const uint bb = w[j];
                    [unroll] for (uint n = 0; n < 4; ++n) {
                        const float d = bf16_round(ldexp(gTbl[(bb >> (8 * n)) & 0xFFu], e));
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
            const uint o = (m * kSliceBlk + b) * kXStride;
            [unroll] for (uint i = 0; i < 16; ++i)
                gXQ[o + i] = bf16_encode(v[i * 2]) | (bf16_encode(v[i * 2 + 1]) << 16);
            gXS[m * kSliceBlk + b] = s;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = gid * (kWg / LanesPerRow) + tid / LanesPerRow;
    float acc[M];
    [unroll] for (uint m = 0; m < kMaxM; ++m) if (m < M) acc[m] = 0.0f;
    if (row < pc.rows) {
        const uint64_t wrow = Ptr[kW] + (uint64_t)row * (uint64_t)pc.k;
        const uint64_t srow = Ptr[kS] + (uint64_t)(row >> 5) * pc.scale_cols;
        for (uint b = sublane; b < blocks; b += LanesPerRow) {
            const uint fb = blk0 + b;
            const uint4 w0 = load16(wrow + (uint64_t)fb * 32);
            const uint4 w1 = load16(wrow + (uint64_t)fb * 32 + 16);
            const float we = ldexp(1.0f, (int)load_u8(srow + fb) - 127);
            float wv[32];
            [unroll] for (uint i = 0; i < 4; ++i) {
                const uint a = w0[i], c = w1[i];
                [unroll] for (uint j = 0; j < 4; ++j) {
                    wv[i * 4 + j]      = gTbl[(a >> (8 * j)) & 0xFFu];
                    wv[16 + i * 4 + j] = gTbl[(c >> (8 * j)) & 0xFFu];
                }
            }
            [unroll] for (uint m = 0; m < kMaxM; ++m) {
                if (m >= M) continue;
                const uint o = (m * kSliceBlk + b) * kXStride;
                float p = 0.0f;
                [unroll] for (uint i = 0; i < 16; ++i) {
                    const uint xw = gXQ[o + i];
                    p += wv[i * 2]     * bf16_decode(xw & 0xFFFFu);
                    p += wv[i * 2 + 1] * bf16_decode(xw >> 16);
                }
                acc[m] += p * (we * gXS[m * kSliceBlk + b]);
            }
        }
    }
    [unroll] for (uint m = 0; m < kMaxM; ++m) {
        if (m >= M) continue;
        const float dot = row_reduce(acc[m], tid, sublane);
        if (sublane == 0 && row < pc.rows) {
            const uint64_t a = Ptr[kKv] + ((uint64_t)m * pc.rows + row) * 4;
            *(float*)a = (pc.slice == 0u) ? dot : (load_f32(a) + dot);
        }
    }
}

void stage_gate(uint j, uint m, uint tid) {
    const uint64_t X   = Ptr[kX]  + ((uint64_t)m * pc.hc + j) * pc.dim * 4;
    const uint64_t Key = Ptr[kKv] + ((uint64_t)m * pc.rows + (uint64_t)j * pc.dim) * 4;
    const uint64_t Val = Ptr[kKv] + ((uint64_t)m * pc.rows + (uint64_t)pc.hc * pc.dim) * 4;
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
    const float mag  = sqrt(max(abs(dot), kGateFloor));
    const float sdot = (dot < 0.0f) ? -mag : mag;
    const float gate = 1.0f / (1.0f + exp(-sdot));
    const uint64_t Out = Ptr[kXout] + ((uint64_t)m * pc.hc + j) * pc.dim * 4;
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
    else            stage_gate(gid.x, gid.y, tid);
}
