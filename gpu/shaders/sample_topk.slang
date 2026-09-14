// sample_topk -- the sampled half of design §7.11's sampler (Track P,
// docs/p3_chat.md §2).
//
// The greedy stage in head.slang reduces the 129,280 logits to four words. A
// temperature / top-p draw needs more than the argmax, but not the whole
// vector: it needs every token that can be in the nucleus and the softmax
// normaliser. This kernel returns exactly that, in one workgroup of 256
// threads, each thread owning the stride of ids i = tid (mod 256):
//
//   1. each thread's maximum, then a reduction tree to the global maximum M;
//   2. each thread histograms d = M - l into kBins bins of 1/BinsPerLogit
//      logit, in its own slice of the histogram buffer;
//   3. thread 0 sums the histograms in bin order and picks the first bin J
//      whose cumulative count reaches `k` (the last bin if none does);
//   4. each thread writes every (id, logit) with d * BinsPerLogit < J + 1 --
//      i.e. bins 0..J, a true top set -- into its own kCap-pair segment, and
//      sums exp(-d / T) over the tokens it did NOT keep (the tail);
//   5. thread 0 writes the header and the per-thread counts.
//
// The host (runtime/sampling.cpp) adds the candidates' weights in double and
// the tail to get the full-vocabulary normaliser, so its nucleus is the
// full-vocab nucleus whenever the candidates hold top_p of the mass. A thread
// that finds more than kCap candidates sets the overflow word and the host
// copies the logit vector instead. The arithmetic is mirrored float for float
// by `runtime::emulate_topk`, which the tests check it against.
//
// Output buffer, uint32 words:
//   [0] rows scanned   [1] M (float bits)   [2] tail (float bits)   [3] J
//   [4] kCap           [5] overflow         [6] kBins               [7] 0
//   [8 .. 8+256)       candidates found per thread (may exceed kCap)
//   [264 + 2*kCap*t ..] thread t's (id, logit bits) pairs

#include "attn_common.slang"

static const uint kLogits = 0, kOut = 1, kHist = 2;
static const uint kBins = 256;
static const uint kCap  = 32;
static const uint kHdr  = 8;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct TopKPush {
    uint  rows;             // vocab_size = 129280
    uint  k;                // the candidate count to reach (1024)
    float inv_t;            // 1 / temperature
    float bins_per_logit;   // 16
};
[[vk::push_constant]] TopKPush pc;

groupshared float gMax[kWg];
groupshared float gTail[kWg];
groupshared uint  gCnt[kWg];
groupshared uint  gJ[1];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint64_t L = Ptr[kLogits];

    // 1. the maximum
    float m = -3.0e38f;
    for (uint i = tid; i < pc.rows; i += kWg) m = max(m, load_f32(L + (uint64_t)i * 4));
    gMax[tid] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) gMax[tid] = max(gMax[tid], gMax[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float M = gMax[0];

    // 2. this thread's histogram
    {
        uint* hist = (uint*)(Ptr[kHist] + (uint64_t)tid * kBins * 4);
        for (uint b = 0; b < kBins; ++b) hist[b] = 0;
        for (uint i = tid; i < pc.rows; i += kWg) {
            const float fb = (M - load_f32(L + (uint64_t)i * 4)) * pc.bins_per_logit;
            if (fb < (float)kBins) hist[(uint)fb] += 1;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. the bin
    if (tid == 0) {
        uint cum = 0;
        uint J = kBins - 1;
        for (uint b = 0; b < kBins; ++b) {
            for (uint t = 0; t < kWg; ++t)
                cum += load_u32(Ptr[kHist] + (uint64_t)(t * kBins + b) * 4);
            if (cum >= pc.k) { J = b; break; }
        }
        gJ[0] = J;
    }
    GroupMemoryBarrierWithGroupSync();

    // 4. collect, and the tail
    {
        const float lim = (float)(gJ[0] + 1);
        uint* seg = (uint*)(Ptr[kOut] + (uint64_t)(kHdr + kWg + tid * kCap * 2) * 4);
        uint n = 0;
        float acc = 0.0f;
        for (uint i = tid; i < pc.rows; i += kWg) {
            const float v = load_f32(L + (uint64_t)i * 4);
            const float d = M - v;
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

    // 5. the header
    if (tid == 0) {
        uint* out = (uint*)Ptr[kOut];
        float tail = 0.0f;
        uint ovf = 0;
        for (uint t = 0; t < kWg; ++t) {
            tail += gTail[t];
            out[kHdr + t] = gCnt[t];
            if (gCnt[t] > kCap) ovf = 1;
        }
        out[0] = pc.rows;
        out[1] = asuint(M);
        out[2] = asuint(tail);
        out[3] = gJ[0];
        out[4] = kCap;
        out[5] = ovf;
        out[6] = kBins;
        out[7] = 0;
    }
}
