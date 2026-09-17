// dspark_verify -- the four numbers per verify row that exact, lossless
// speculative acceptance needs (docs/p4_dspark_runtime.md §2.4,
// docs/p3_dspark.md §3.3 rule 2).
//
// `cpu::dspark::accept_sampling_exact` implements the standard speculative
// sampling rule against the TRUE temperature-1 distribution rather than a
// top-Kv truncation of it, and to do that it needs, per verify row j:
//
//   p(c) = exp(l_c - lse)  for c in C_j   ->  the row's logits AT the draft
//                                             candidates, and the row's
//                                             full-vocabulary logsumexp;
//   on rejection, a draw from p restricted to the COMPLEMENT of C_j;
//   on full acceptance, a draw from the whole last row (the bonus).
//
// None of that is a new kernel family: the head has already produced the row,
// and all four are one pass over it. This is that pass -- one workgroup per row,
// M workgroups, one mapped readback for the whole verify batch.
//
// Why the samples are drawn HERE and not on the host
// --------------------------------------------------
// The alternative is to copy 129,280 fp32 per row back and sample on the CPU:
// 3.1 MB a row, 18.6 MB a verify batch, against the 68 words this writes. The
// row is already in device memory and the draw is a reduction over it, which is
// what a GPU is.
//
// The draw is Gumbel-max, which is exact
// --------------------------------------
// argmax_i (l_i / T + g_i) with g_i = -log(-log(u_i)) and u_i iid uniform is
// distributed exactly as softmax(l / T); restricting the argmax to a subset is
// distributed exactly as softmax restricted to that subset, which is precisely
// the "p outside C_j" the residual rule wants. So the masked draw is the same
// scan as the full one with the candidates skipped, and nothing has to be
// renormalised -- the renormalisation is what Gumbel-max does for free.
//
// The uniforms are counter-based, not stateful: u(seed, row, i). Two dispatches
// with the same seed produce the same two samples, which is what makes
// `tests/test_gpu_dspark.cpp`'s host emulation able to compare word for word,
// and what lets a replay of a speculative run reproduce it.
//
// Determinism
// -----------
// The two argmaxes break ties by the LOWEST id (`>` against the running best,
// scanned id-ascending within a thread and then by thread index), the same rule
// head.slang's argmax uses. Exact ties in a Gumbel key are measure-zero in
// exact arithmetic but not in fp32, and a sampler that resolves them by
// whichever thread got there first is not reproducible.
//
// NaN: `v > best` is false for NaN, so an all-NaN row reports token 0, as
// design §7.11 records for the greedy sampler. `rows` in the header is there so
// a caller can tell the dispatch saw the whole vocabulary.

#include "dspark_common.slang"

// Slot indices; mirrored by `gpu::dvslot` in gpu/vulkan/dspark_kernels.h.
static const uint kVLogits = 0;   // [M][vocab] fp32, the head's output
static const uint kVCand   = 1;   // [M][cand_stride] int32 candidate ids, -1 = unused
static const uint kVOut    = 2;   // [M][kVerifyRecordWords] uint32

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct DsparkVerifyPush {
    uint  m;             // live verify rows; the dispatch covers min(M, m)
    uint  rows;          // vocab_size = 129280
    uint  n_cand;        // candidates per row, <= kMaxCand
    uint  cand_stride;   // int32s between candidate rows
    uint  logit_stride;  // floats between logit rows = 129280
    uint  seed;          // the cycle's counter-based RNG seed
    uint  flags;         // reserved; 0
    float inv_t;         // 1 / temperature; 1.0 for the lossless rule
};
[[vk::push_constant]] DsparkVerifyPush pc;

// The record this writes per row. Mirrored by `gpu::DsparkVerifyRow`.
//   [0] rows scanned
//   [1] row max logit          (float bits)
//   [2] full-vocabulary lse    (float bits)
//   [3] the masked draw's token id       (candidates excluded)
//   [4] the full row's draw token id     (the bonus)
//   [5] n_cand actually read
//   [6] the masked draw's Gumbel key     (float bits, diagnostic)
//   [7] the full draw's Gumbel key       (float bits, diagnostic)
//   [8 .. 8 + kMaxCand)  this row's logit at candidate c, in candidate order
static const uint kVHdr    = 8;
static const uint kMaxCand = 64;
static const uint kRecordWords = kVHdr + kMaxCand;

groupshared float gVRed[kWg];
groupshared float gKey[kWg];
groupshared uint  gId[kWg];
groupshared int   gCand[kMaxCand];     // sorted ascending, for the mask test
groupshared uint  gNCand[1];

// --- the uniforms -----------------------------------------------------------
// A counter-based hash of (seed, row, id): three rounds of the 32-bit mix
// Stafford variant 13 of splitmix32 uses. It has to be a hash and not a stream,
// because every thread evaluates its own subset of ids in its own order and the
// draw must not depend on which thread got which id.
uint mix32(uint z) {
    z ^= z >> 16; z *= 0x7FEB352Du;
    z ^= z >> 15; z *= 0x846CA68Bu;
    z ^= z >> 16;
    return z;
}
uint hash3(uint a, uint b, uint c) {
    return mix32(mix32(a ^ 0x9E3779B9u) ^ mix32(b + 0x85EBCA6Bu) ^ (c * 0xC2B2AE35u));
}
// (0, 1): the top 24 bits, offset by half a step so neither 0 nor 1 is attainable
// -- log(-log(u)) is only finite on the open interval.
float u01(uint h) {
    return ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}
// -log(-log(u)); u in (0, 1) so log(u) < 0 and -log(u) > 0.
float gumbel(uint seed, uint row, uint id) {
    return -log(-log(u01(hash3(seed, row, id))));
}

// Is `id` one of this row's candidates? `gCand` is ascending, n <= 64, so this
// is six steps rather than a linear scan of the row's candidate list per
// vocabulary entry.
bool is_candidate(int id, uint n) {
    uint lo = 0, hi = n;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (gCand[mid] < id) lo = mid + 1;
        else                 hi = mid;
    }
    return lo < n && gCand[lo] == id;
}

// max over the workgroup, into gVRed[0].
float wg_reduce_max(float v, uint tid) {
    gVRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) gVRed[tid] = max(gVRed[tid], gVRed[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    return gVRed[0];
}

// sum over the workgroup, into gVRed[0]. The tree is fixed-shape, so the sum is
// reproducible run to run on the same geometry.
float wg_reduce_sum(float v, uint tid) {
    gVRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) gVRed[tid] = gVRed[tid] + gVRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    return gVRed[0];
}

// argmax over the workgroup of (gKey, gId), ties to the lowest id.
void wg_reduce_argmax(uint tid) {
    for (uint s = kWg >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const bool take = (gKey[tid + s] > gKey[tid]) ||
                              (gKey[tid + s] == gKey[tid] && gId[tid + s] < gId[tid]);
            if (take) { gKey[tid] = gKey[tid + s]; gId[tid] = gId[tid + s]; }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint row = gid.x;
    if (row >= mlive(pc.m)) return;

    const uint64_t L = Ptr[kVLogits] + (uint64_t)row * pc.logit_stride * 4;
    const uint64_t C = Ptr[kVCand]   + (uint64_t)row * pc.cand_stride * 4;
    const uint64_t O = Ptr[kVOut]    + (uint64_t)row * kRecordWords * 4;

    // 0. the candidates, sorted ascending in LDS. A row may carry fewer than
    //    n_cand: an entry < 0 ends the list, which is how a shorter lattice row
    //    reaches the kernel without a second push constant.
    if (tid == 0) {
        uint n = 0;
        for (uint i = 0; i < pc.n_cand && i < kMaxCand; ++i) {
            const int v = (int)load_u32(C + (uint64_t)i * 4);
            if (v < 0) break;
            // insertion sort; n <= 64 and this runs once a row
            uint j = n;
            while (j > 0 && gCand[j - 1] > v) { gCand[j] = gCand[j - 1]; --j; }
            gCand[j] = v;
            ++n;
        }
        gNCand[0] = n;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint n_cand = gNCand[0];

    // 1. the row maximum.
    float mx = -3.0e38f;
    for (uint i = tid; i < pc.rows; i += kWg) mx = max(mx, load_f32(L + (uint64_t)i * 4));
    const float M_ = wg_reduce_max(mx, tid);
    GroupMemoryBarrierWithGroupSync();

    // 2. the exact full-vocabulary logsumexp. Shifted by the maximum, so the
    //    sum is in [1, rows] and no term overflows.
    float acc = 0.0f;
    for (uint i = tid; i < pc.rows; i += kWg) acc += exp(load_f32(L + (uint64_t)i * 4) - M_);
    const float S = wg_reduce_sum(acc, tid);
    GroupMemoryBarrierWithGroupSync();
    const float lse = M_ + log(S);

    // 3. the two draws, in ONE pass: the same Gumbel key feeds both argmaxes and
    //    the masked one simply skips the candidates.
    float best_all = -3.0e38f, best_msk = -3.0e38f;
    uint  id_all = 0, id_msk = 0;
    for (uint i = tid; i < pc.rows; i += kWg) {
        const float key = load_f32(L + (uint64_t)i * 4) * pc.inv_t + gumbel(pc.seed, row, i);
        if (key > best_all) { best_all = key; id_all = i; }
        if (!is_candidate((int)i, n_cand) && key > best_msk) { best_msk = key; id_msk = i; }
    }
    gKey[tid] = best_all; gId[tid] = id_all;
    GroupMemoryBarrierWithGroupSync();
    wg_reduce_argmax(tid);
    const float key_all = gKey[0];
    const uint  tok_all = gId[0];
    GroupMemoryBarrierWithGroupSync();
    gKey[tid] = best_msk; gId[tid] = id_msk;
    GroupMemoryBarrierWithGroupSync();
    wg_reduce_argmax(tid);
    const float key_msk = gKey[0];
    const uint  tok_msk = gId[0];
    GroupMemoryBarrierWithGroupSync();

    // 4. the candidates' logits, in CANDIDATE order (not the sorted order the
    //    mask test uses): the host pairs them with the lattice's candidate
    //    index, which is what `accept_sampling_exact` indexes by.
    for (uint i = tid; i < pc.n_cand && i < kMaxCand; i += kWg) {
        const int v = (int)load_u32(C + (uint64_t)i * 4);
        const float l = (v >= 0 && (uint)v < pc.rows) ? load_f32(L + (uint64_t)v * 4)
                                                      : -3.0e38f;
        *(uint*)(O + (uint64_t)(kVHdr + i) * 4) = asuint(l);
    }

    if (tid == 0) {
        uint* out = (uint*)O;
        out[0] = pc.rows;
        out[1] = asuint(M_);
        out[2] = asuint(lse);
        out[3] = tok_msk;
        out[4] = tok_all;
        out[5] = n_cand;
        out[6] = asuint(key_msk);
        out[7] = asuint(key_all);
    }
}
