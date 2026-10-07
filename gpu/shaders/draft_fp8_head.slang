// Draft-only row-scaled E4M3FN head. The shared target head remains BF16.
// Preserve native BF16 activation staging and reduction geometry; dequantize
// weights to float32 before multiplication, matching the offline oracle.
#include "attn_common.slang"


static const uint kMaxM = 6;
static const uint kW = 0, kX = 1, kLogits = 2, kSample = 3, kTopOut = 4, kHist = 5;
static const uint kDraft = 6, kRank = 7, kRowScale = 8;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct DraftFp8HeadPush {
    uint  rows;             // vocab 129280
    uint  k;                // 0: 5120; 3: number of draft tokens
    uint  slice;            // 0: this dispatch's slice; 2: high bit selects spec row
    uint  slices;           // 0: how many
    uint  x_stride;         // 0: X row stride; 2 spec mode: acceptance top-K
    uint  topk_k;           // 2: candidates to reach
    float inv_t;            // 2: 1 / temperature
    float bins_per_logit;   // 2
};
[[vk::push_constant]] DraftFp8HeadPush pc;

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
        const uint64_t wrow = Ptr[kW] + (uint64_t)row * pc.k + k0;
        const float scale = load_f32(Ptr[kRowScale] + (uint64_t)row * 4);
        for (uint blk = sublane; blk < blocks; blk += LanesPerRow) {
            float wv[32];
            [unroll] for (uint i = 0; i < 2; ++i) {
                const uint4 v = load16(wrow + (uint64_t)blk * 32 + (uint64_t)i * 16);
                [unroll] for (uint j = 0; j < 4; ++j) {
                    [unroll] for (uint byte = 0; byte < 4; ++byte)
                        wv[i * 16 + j * 4 + byte] =
                            fp8_decode((v[j] >> (byte * 8)) & 255u) * scale;
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

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    stage_gemv(gid.x, tid);
}
