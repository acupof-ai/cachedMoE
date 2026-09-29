// prefill_topk -- the indexer's top-k rows on the GPU: Prefill::topk_rows
// (gpu/vulkan/prefill_kernels.cpp) for one query per workgroup, the same
// answer bit for bit (docs/STATUS.md §7 0ae).
//
// For query row j at prompt position p = pos0 + j the output row [n_idx] is
// the window part -- the last `window` positions, -1 past p -- followed by
// the k = n_idx - nw compressed positions with the best indexer scores among
// the `vis` the query can see, ordered by (score descending, position
// ascending) and written in position order (+ n_kv_rows), -1 for the rest.
// A candidate-block mask (flags & 1, [b][nblocks] bytes) turns a block's
// scores into -inf first, as the host does.
//
// The k-th best key is found by a 4-pass radix select over the scores'
// order-preserving bit patterns (256-bin histogram in LDS, one bin a thread
// for the suffix count), then every key above it and the first r keys equal
// to it, in position order, are compacted into the row by a workgroup scan.
// Keys equal in value are ordered by position, as the host's comparator is;
// -0 is +0 (the host compares floats).

#include "prefill_common.slang"

static const uint kS = 0, kOut = 1, kMask = 2;
static const uint kT = 256;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct TopkPush {
    uint b, g, n_idx, window, pos0, kv_pos0, n_kv_rows, ratio, block, nblocks, flags, pad;
};
[[vk::push_constant]] TopkPush pc;

groupshared uint hist[256];
groupshared uint wsum[8];
groupshared uint found[2];

// The score of compressed position i of query `row` as an order-preserving key.
uint key_of(uint i, uint row) {
    float f = load_f32(Ptr[kS] + ((uint64_t)row * pc.g + i) * 4);
    if ((pc.flags & 1u) != 0u) {
        const uint blk = i / pc.block;
        const bool keep = blk < pc.nblocks && *(uint8_t*)(Ptr[kMask] + (uint64_t)row * pc.nblocks + blk) != 0;
        if (!keep) f = -1.0f / 0.0f;
    }
    uint x = asuint(f);
    if (x == 0x80000000u) x = 0u;
    return (x & 0x80000000u) != 0u ? ~x : (x | 0x80000000u);
}

// Exclusive prefix of v over the workgroup's 256 threads, and the total.
uint scan_wg(uint v, uint tid, out uint total) {
    const uint w = tid >> 5;
    const uint pre = WavePrefixSum(v), tot = WaveActiveSum(v);
    if ((tid & 31u) == 0u) wsum[w] = tot;
    GroupMemoryBarrierWithGroupSync();
    uint off = 0;
    total = 0;
    for (uint i = 0; i < 8; ++i) { if (i < w) off += wsum[i]; total += wsum[i]; }
    GroupMemoryBarrierWithGroupSync();
    return off + pre;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint row = gid.x, p = pc.pos0 + row;
    const uint nw = min(pc.n_kv_rows, pc.window), k = pc.n_idx - nw;
    int* out = (int*)(Ptr[kOut] + (uint64_t)row * pc.n_idx * 4);
    const int first = max(int(pc.kv_pos0), int(p) - int(pc.window) + 1);
    for (uint i = tid; i < nw; i += kT) {
        const int pos = first + int(i);
        out[i] = pos > int(p) ? -1 : pos - int(pc.kv_pos0);
    }
    if (k == 0u) return;
    const uint vis = min(pc.g, (p + 1u) / max(pc.ratio, 1u)), take = min(k, vis);
    uint base = 0;
    if (vis <= k) {
        for (uint i = tid; i < vis; i += kT) out[nw + i] = int(i + pc.n_kv_rows);
        base = vis;
    } else {
        uint prefix = 0, r = take;
        for (int pass = 3; pass >= 0; --pass) {
            const uint shift = 8u * uint(pass), hi = shift + 8u;
            hist[tid] = 0;
            GroupMemoryBarrierWithGroupSync();
            for (uint i = tid; i < vis; i += kT) {
                const uint u = key_of(i, row);
                if (hi >= 32u || (u >> hi) == (prefix >> hi)) InterlockedAdd(hist[(u >> shift) & 255u], 1u);
            }
            GroupMemoryBarrierWithGroupSync();
            // bin = 255 - tid: `above` = keys in higher bins, this bin holds the r-th
            const uint bin = 255u - tid, h = hist[bin];
            uint tot;
            const uint above = scan_wg(h, tid, tot);
            if (above < r && above + h >= r) { found[0] = bin; found[1] = above; }
            GroupMemoryBarrierWithGroupSync();
            prefix |= found[0] << shift;
            r -= found[1];
            GroupMemoryBarrierWithGroupSync();
        }
        // every key above the threshold, and the first r equal to it, in position order
        uint eq_base = 0;
        for (uint c0 = 0; c0 < vis; c0 += kT) {
            const uint i = c0 + tid;
            const bool valid = i < vis;
            const uint u = valid ? key_of(i, row) : 0u;
            const uint eq = (valid && u == prefix) ? 1u : 0u;
            uint eq_tot, sel_tot;
            const uint eq_rank = eq_base + scan_wg(eq, tid, eq_tot);
            const uint sel = (valid && (u > prefix || (eq != 0u && eq_rank < r))) ? 1u : 0u;
            const uint pos = base + scan_wg(sel, tid, sel_tot);
            if (sel != 0u) out[nw + pos] = int(i + pc.n_kv_rows);
            eq_base += eq_tot;
            base += sel_tot;
        }
    }
    for (uint i = base + tid; i < k; i += kT) out[nw + i] = -1;
}
