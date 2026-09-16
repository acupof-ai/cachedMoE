// mgt1_head -- the LM head and the per-row readouts for M rows (Track T,
// docs/p4_mgt1.md): a verify batch's M = k + 1 rows, or a draft cycle's five.
//
// bf16 [129280 x 5120] against M fp32 activations, promoted to fp32
// (`ParallelHead`), no activation quantisation -- head.slang's arithmetic. The
// 1.32 GB is read once for all M rows.
//
// Geometry
// --------
// head.slang stages the 5120-wide activation as fp32 in LDS (20 KiB). M columns
// of it would be 120 KiB, so the contraction axis is cut into `pc.slices`
// slices and each slice is its OWN dispatch over the same row groups: slice 0
// writes the logits, every later slice adds to them. The dispatches run one
// after another (a barrier between them), so the read-modify-write needs no
// partial plane, and a dispatch is still thousands of workgroups on forty CUs.
//
// The activation is staged as bf16 in 17-word blocks of 32 (fp8_gemv.slang's
// layout: a lane owns one LDS bank). That is exact: the head's input is an
// RMSNorm output and RMSNorm.forward ends in `.to(bf16)`.
//
// Stages:
//   0  one slice of the GEMV, gid.x = row group; `pc.slice` which, `pc.slices`
//      of how many
//   1  argmax + runner-up of row m (head.slang stage 1), gid.y = m
//   2  top set + tail mass of row m (sample_topk.slang's arithmetic, one
//      workgroup a row), gid.y = m

#include "attn_common.slang"

static const uint kMaxM = 6;
static const uint kW = 0, kX = 1, kLogits = 2, kSample = 3, kTopOut = 4, kHist = 5;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct Mgt1HeadPush {
    uint  rows;             // vocab 129280
    uint  k;                // 5120
    uint  slice;            // 0: this dispatch's slice
    uint  slices;           // 0: how many
    uint  x_stride;         // elements between rows of X
    uint  topk_k;           // 2: candidates to reach
    float inv_t;            // 2: 1 / temperature
    float bins_per_logit;   // 2
};
[[vk::push_constant]] Mgt1HeadPush pc;

static const uint kMaxSliceK  = 1024;
static const uint kSliceBlk   = kMaxSliceK / kBlockK;     // 32
static const uint kXStride    = kBlockK / 2 + 1;          // 17
groupshared uint gXQ[M * kSliceBlk * kXStride];

void stage_gemv(uint gid, uint tid) {
    const uint kc = pc.k / pc.slices;
    const uint k0 = pc.slice * kc;
    const uint blocks = kc / kBlockK;
    // Staging: M * blocks threads, one block each.
    {
        const uint m = tid / kSliceBlk, blk = tid % kSliceBlk;
        if (m < M && blk < blocks) {
            const uint64_t base = Ptr[kX] + ((uint64_t)m * pc.x_stride + k0 + (uint64_t)blk * 32) * 4;
            [unroll] for (uint i = 0; i < 8; ++i) {
                const uint4 w = load16(base + (uint64_t)i * 16);
                const uint o = (m * kSliceBlk + blk) * kXStride + i * 2;
                gXQ[o]     = bf16_encode(asfloat(w[0])) | (bf16_encode(asfloat(w[1])) << 16);
                gXQ[o + 1] = bf16_encode(asfloat(w[2])) | (bf16_encode(asfloat(w[3])) << 16);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint row = gid * (kWg / LanesPerRow) + tid / LanesPerRow;
    float acc[M];
    [unroll] for (uint m = 0; m < kMaxM; ++m) if (m < M) acc[m] = 0.0f;
    if (row < pc.rows) {
        const uint64_t wrow = Ptr[kW] + ((uint64_t)row * pc.k + k0) * 2;
        for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
            float wv[32];
            [unroll] for (uint i = 0; i < 4; ++i) {
                const uint4 v = load16(wrow + (uint64_t)blk * 64 + (uint64_t)i * 16);
                [unroll] for (uint j = 0; j < 4; ++j) {
                    wv[i * 8 + j * 2]     = bf16_decode(v[j] & 0xFFFFu);
                    wv[i * 8 + j * 2 + 1] = bf16_decode(v[j] >> 16);
                }
            }
            [unroll] for (uint m = 0; m < kMaxM; ++m) {
                if (m >= M) continue;
                float p = 0.0f;
                [unroll] for (uint i = 0; i < 16; ++i) {
                    const uint xw = gXQ[(m * kSliceBlk + blk) * kXStride + i];
                    p += wv[i * 2]     * bf16_decode(xw & 0xFFFFu);
                    p += wv[i * 2 + 1] * bf16_decode(xw >> 16);
                }
                acc[m] += p;
            }
        }
    }
    [unroll] for (uint m = 0; m < kMaxM; ++m) {
        if (m >= M) continue;
        const float dot = row_reduce(acc[m], tid, sublane);
        if (sublane == 0 && row < pc.rows) {
            const uint64_t a = Ptr[kLogits] + ((uint64_t)m * pc.rows + row) * 4;
            *(float*)a = (pc.slice == 0u) ? dot : (load_f32(a) + dot);
        }
    }
}

// --- stage 1: argmax -----------------------------------------------------------
groupshared float gBest[kWg];
groupshared uint  gBestId[kWg];
groupshared float gSecond[kWg];

void top2_merge(inout float ab, inout uint ai, inout float a2, float bb, uint bi, float b2) {
    if (ab >= bb) { a2 = max(a2, bb); }
    else          { a2 = max(ab, b2); ab = bb; ai = bi; }
}

void stage_argmax(uint m, uint tid) {
    float best = -3.0e38f, second = -3.0e38f;
    uint  bid = 0;
    const uint64_t L = Ptr[kLogits] + (uint64_t)m * pc.rows * 4;
    for (uint i = tid; i < pc.rows; i += kWg) {
        const float v = load_f32(L + (uint64_t)i * 4);
        if (v > best)        { second = best; best = v; bid = i; }
        else if (v > second) { second = v; }
    }
    gBest[tid] = best; gBestId[tid] = bid; gSecond[tid] = second;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float ab = gBest[tid], a2 = gSecond[tid];
            uint  ai = gBestId[tid];
            top2_merge(ab, ai, a2, gBest[tid + s], gBestId[tid + s], gSecond[tid + s]);
            gBest[tid] = ab; gBestId[tid] = ai; gSecond[tid] = a2;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0) {
        uint* out = (uint*)(Ptr[kSample] + (uint64_t)m * 16);
        out[0] = gBestId[0];
        out[1] = asuint(gBest[0]);
        out[2] = asuint(gSecond[0]);
        out[3] = pc.rows;
    }
}

// --- stage 2: the top set and the tail, per row ---------------------------------
static const uint kBins = 256;
static const uint kCap  = 32;
static const uint kHdr  = 8;
groupshared float gMax[kWg];
groupshared float gTail[kWg];
groupshared uint  gCnt[kWg];
groupshared uint  gJ[1];

// Words of one row's output record; the host lays the M records end to end.
uint topk_record_words() { return kHdr + kWg + kWg * kCap * 2; }

void stage_topk(uint m, uint tid) {
    const uint64_t L = Ptr[kLogits] + (uint64_t)m * pc.rows * 4;
    const uint64_t O = Ptr[kTopOut] + (uint64_t)m * topk_record_words() * 4;
    const uint64_t H = Ptr[kHist] + (uint64_t)m * kWg * kBins * 4;

    float mx = -3.0e38f;
    for (uint i = tid; i < pc.rows; i += kWg) mx = max(mx, load_f32(L + (uint64_t)i * 4));
    gMax[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) gMax[tid] = max(gMax[tid], gMax[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float Mx = gMax[0];
    {
        uint* hist = (uint*)(H + (uint64_t)tid * kBins * 4);
        for (uint b = 0; b < kBins; ++b) hist[b] = 0;
        for (uint i = tid; i < pc.rows; i += kWg) {
            const float fb = (Mx - load_f32(L + (uint64_t)i * 4)) * pc.bins_per_logit;
            if (fb < (float)kBins) hist[(uint)fb] += 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        uint cum = 0;
        uint J = kBins - 1;
        for (uint b = 0; b < kBins; ++b) {
            for (uint t = 0; t < kWg; ++t) cum += load_u32(H + (uint64_t)(t * kBins + b) * 4);
            if (cum >= pc.topk_k) { J = b; break; }
        }
        gJ[0] = J;
    }
    GroupMemoryBarrierWithGroupSync();
    {
        const float lim = (float)(gJ[0] + 1);
        uint* seg = (uint*)(O + (uint64_t)(kHdr + kWg + tid * kCap * 2) * 4);
        uint n = 0;
        float acc = 0.0f;
        for (uint i = tid; i < pc.rows; i += kWg) {
            const float v = load_f32(L + (uint64_t)i * 4);
            const float d = Mx - v;
            if (d * pc.bins_per_logit < lim) {
                if (n < kCap) { seg[n * 2] = i; seg[n * 2 + 1] = asuint(v); }
                n += 1;
            } else {
                acc += exp(-d * pc.inv_t);
            }
        }
        gCnt[tid] = n;
        gTail[tid] = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        uint* out = (uint*)O;
        float tail = 0.0f;
        uint ovf = 0;
        for (uint t = 0; t < kWg; ++t) {
            tail += gTail[t];
            out[kHdr + t] = gCnt[t];
            if (gCnt[t] > kCap) ovf = 1;
        }
        out[0] = pc.rows;
        out[1] = asuint(Mx);
        out[2] = asuint(tail);
        out[3] = gJ[0];
        out[4] = kCap;
        out[5] = ovf;
        out[6] = kBins;
        out[7] = m;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_gemv(gid.x, tid);
    else if (Stage == 1) stage_argmax(gid.y, tid);
    else                 stage_topk(gid.y, tid);
}
