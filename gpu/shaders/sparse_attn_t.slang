// sparse_attn_t -- the design §7.5 attention on a head-group x KV-tile grid,
// with a partial-softmax combine (docs/p2_attention.md §11 item 2, §13).
//
// What this is instead of
// -----------------------
// `sparse_attn.slang` gives a workgroup one head (or `HeadsPerWg` of them) and
// the WHOLE KV list. At one head a workgroup that is 64 workgroups each pulling
// 320 KiB through L2 twice -- 41 MB of traffic for 320 KiB of unique data --
// and §10.2 measured that raising `HeadsPerWg` to take the traffic back costs
// more than it saves, because 64 heads at 8 a workgroup is eight workgroups on
// forty CUs.
//
// The grid here is two-dimensional, flattened into `gid`:
//
//     gid = head_group * n_tiles + tile
//
// so 8 heads a workgroup and 8 tiles is 64 workgroups -- the occupancy §10.2
// says is worth more than the traffic -- while the KV is read once per HEAD
// GROUP rather than once per head, i.e. 8 times instead of 64. Both knobs are
// free: `n_tiles` is a push constant, `HeadsPerWg` the existing specialisation
// constant, and neither changes the arithmetic.
//
// Why it is three dispatches and not two
// --------------------------------------
// `sparse_attn_kernel` rounds p to bf16 AFTER subtracting the final row max,
// and that max is over the KV scores only -- the per-head `attn_sink` enters
// the denominator as `exp(sink - mx)` but never competes for the max. That is
// what rules out an online softmax (§4.2), and tiling does not change it: a
// tile cannot know the row max until every tile has scored. So
//
//   stage 0  scores its (head group, tile) into the score plane, and writes
//            one tile maximum per head;
//   stage 1  reads the `n_tiles` tile maxima of its head -- which is the whole
//            reduction, `n_tiles` scalars, no dispatch of its own -- takes the
//            final max from them, rounds `p = bf16(exp(s - mx))` against it
//            exactly where `acc_s_cast` does, and accumulates P.V over its own
//            tile into an fp32 partial;
//   stage 2  adds the `pv_tiles` partials, adds `exp(sink - mx)` to the
//            denominator once, divides, and rotates the output back.
//
// Stage 1 has its OWN grid: `pv_tiles` tiles and `HeadsPerWg` of its own
// pipeline, flattened the same way. The two are separate because they want
// different things. Stage 0 reads a KV row once and dots it against every head
// in the workgroup, so eight heads a workgroup really does divide its traffic
// by eight. Stage 1 cannot do that without 8 * 16 accumulators a lane -- a
// lane owns sixteen OUTPUT dims of ONE head -- so at eight heads a workgroup
// each of its eight waves re-reads the whole tile for its own head, and the
// grouping buys nothing but fewer workgroups. Measured in §13.
//
// The bf16 rounding of p therefore sees the same number the reference's does,
// which an online softmax could not manage. What is re-associated is the
// denominator and P.V, both of which are plain fp32 sums cut into n_tiles
// pieces.
//
// A slot whose index is -1 holds nothing; it scores -3e38, never wins a tile
// max, and contributes p = 0. The final max is clamped at a finite -1e30 for
// the same reason the reference clamps it: a query whose whole index row is -1
// must come out zero, not NaN.
//
// Long context
// ------------
// Nothing here stages the index list in LDS, which is what lets the same .spv
// run at `n_kv` = 193 (the L3 decode geometry) and at 32768: an index is four
// bytes read once per position per workgroup against the 512 bytes of KV that
// position costs, and the 32 lanes of a wave read the same address. `n_tiles`
// then scales the workgroup count with the context instead of the tile length.

#include "attn_common.slang"

static const uint kQ = 0, kWinVal = 1, kWinScale = 2, kCmpKv = 3, kTopIdx = 4;
static const uint kSink = 5, kRope = 6, kScore = 7, kO = 8;
// The three planes the tiling adds: one max per (head, tile), one fp32 output
// partial per (tile, head, dim) and one denominator partial per (tile, head).
static const uint kTileMax = 9, kPartO = 10, kPartD = 11;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct AttnTPush {
    uint  n_kv;            // entries in topk_idxs
    uint  n_win;           // window_size, and the offset of the compressed half
    uint  head_dim;        // 512
    uint  rope_dim;        // 64
    uint  score_stride;    // row stride of the score plane, >= n_kv
    float softmax_scale;   // head_dim ** -0.5
    uint  n_heads;         // 64
    uint  n_tiles;         // KV tiles of stage 0; also kTileMax's stride
    uint  tile_len;        // ceil(n_kv / n_tiles)
    uint  part_stride;     // floats between tiles in kPartO = n_heads * head_dim
    uint  pv_tiles;        // KV tiles of stage 1 and the partial planes; >= 1
    uint  pv_tile_len;     // ceil(n_kv / pv_tiles)
};
[[vk::push_constant]] AttnTPush pc;

static const uint kMaxSpan = 16;          // head_dim / 32
static const uint kMaxHeadsPerWg = 8;

// Stage 0 stages q for its head group (HeadsPerWg * head_dim floats); stage 1
// uses the same pool for the P.V partials of its threads when a head is split
// across more than one wave. At head_dim 512 and HeadsPerWg 8 both are 4096.
// **The staged q is transposed, and the P.V partials are padded**, both for
// the same reason fp8_gemv.slang pads gXQ: a lane owns `span` = 16 CONSECUTIVE
// dims, so the natural layout puts lane l word i at `l * 16 + i`, whose bank is
// `(16 l + i) mod 32` -- two banks across 32 lanes, a 16-way conflict on every
// read. Storing q as [dim-within-span][lane] instead makes lane l read index
// `i * 32 + l`, one bank each; and giving each thread's P.V partial a stride of
// span + 1 = 17 does the same by being odd. Measured in
// docs/p2_attention.md §13.
static const uint kPvStride = kMaxSpan + 1;            // 17, odd on purpose
// max(HeadsPerWg * head_dim, kWg * kPvStride)
groupshared float gPool[kWg * (kMaxSpan + 1)];

// Where element `d` of head `hh` lives, given `span` dims to a lane.
uint q_slot(uint hh, uint head_dim, uint d, uint span) {
    return hh * head_dim + (d % span) * 32 + (d / span);
}

// `span` consecutive KV elements starting at `d0`, dequantised -- the window
// half fp8 E4M3 with a UE8M0 scale per 32 dims exactly as wkv.slang wrote it,
// the compressed half bf16. Identical to sparse_attn.slang's; the two files are
// deliberately not sharing it, because this one is allowed to change shape
// without touching the kernel Track I is already calling.
void kv_span(uint idx, uint d0, uint span, out float v[kMaxSpan]) {
    [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
    if (idx < pc.n_win) {
        const uint64_t base = Ptr[kWinVal] + (uint64_t)idx * pc.head_dim + d0;
        const uint e = load_u8(Ptr[kWinScale] + (uint64_t)idx * (pc.head_dim / kBlockK)
                               + (d0 / kBlockK));
        const float sc = ldexp(1.0f, int(e) - 127);
        [unroll] for (uint w = 0; w < kMaxSpan / 4; ++w) {
            if (w * 4 >= span) continue;
            const uint b = load_u32(base + (uint64_t)w * 4);
            [unroll] for (uint j = 0; j < 4; ++j)
                v[w * 4 + j] = fp8_tbl((b >> (8 * j)) & 0xFFu) * sc;
        }
        return;
    }
    const uint64_t base = Ptr[kCmpKv] + ((uint64_t)(idx - pc.n_win) * pc.head_dim + d0) * 2;
    [unroll] for (uint w = 0; w < kMaxSpan / 2; ++w) {
        if (w * 2 >= span) continue;
        const uint b = load_u32(base + (uint64_t)w * 4);
        v[w * 2]     = bf16_decode(b & 0xFFFFu);
        v[w * 2 + 1] = bf16_decode(b >> 16);
    }
}

float grp_sum(float v, uint tid, uint n) {
    if (n == SubgroupSize) return WaveActiveSum(v);
    gRed[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = n >> 1; s > 0; s >>= 1) {
        if ((tid % n) < s) gRed[tid] = gRed[tid] + gRed[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float r = gRed[tid - (tid % n)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// The final row max of `head`: the largest of its n_tiles tile maxima, clamped
// the way the reference clamps its running max. Cheap enough that stages 1 and
// 2 each just do it rather than pay a dispatch to share it.
float row_max(uint head) {
    float m = -3.0e38f;
    for (uint z = 0; z < pc.n_tiles; ++z)
        m = max(m, load_f32(Ptr[kTileMax] + ((uint64_t)head * pc.n_tiles + z) * 4));
    return max(m, -1.0e30f);
}

// --- stage 0: scores, and one max per (head, tile) --------------------------
// 256 threads = eight waves; wave w walks the tile's positions t0 + w, t0 + w +
// 8, ... and its 32 lanes split the 512-dim row sixteen dims apiece, so a KV row
// is one 16-byte load and one scale byte per lane and the per-head dot product
// is a `WaveActiveSum`. The row is read once for all HeadsPerWg heads.
void stage_score(uint gid, uint tid) {
    fp8_table_fill(tid);
    const uint hg = gid / pc.n_tiles, tl = gid % pc.n_tiles;
    const uint h0 = hg * HeadsPerWg;
    const uint hn = min(h0 + HeadsPerWg, pc.n_heads);
    const uint g  = hn - h0;
    const uint span = pc.head_dim / 32;
    // Thread `tid` fills LDS slots [16 tid, 16 tid + 16) of the transposed
    // layout, so the head, the dim-within-span and the lane of a slot are
    // shifts and masks rather than the four integer divisions by push
    // constants the direct form costs. head_dim is 512 (AttnRunner checks).
    [unroll] for (uint w = 0; w < 16; ++w) {
        const uint k  = tid * 16 + w;
        const uint hh = k >> 9;
        if (hh >= g) continue;
        const uint d  = ((k & 31u) << 4) | ((k >> 5) & 15u);    // lane * 16 + i
        gPool[k] = bf16_decode(load_u16(Ptr[kQ] + ((uint64_t)(h0 + hh) * 512 + d) * 2));
    }
    GroupMemoryBarrierWithGroupSync();

    const uint t0 = tl * pc.tile_len;
    const uint t1 = min(t0 + pc.tile_len, pc.n_kv);
    const uint wave = tid / 32, lane = tid % 32, waves = kWg / 32;
    const uint d0 = lane * span;

    // Bounded by the compile-time kMaxHeadsPerWg, never by `g`: a local array
    // indexed by a loop the compiler cannot unroll goes to scratch memory, and
    // docs/p2_attention.md §4.4 is the record of what that costs.
    float hmax[kMaxHeadsPerWg];
    [unroll] for (uint z = 0; z < kMaxHeadsPerWg; ++z) hmax[z] = -3.0e38f;

    for (uint t = t0 + wave; t < t1; t += waves) {
        const int idx = (int)load_u32(Ptr[kTopIdx] + (uint64_t)t * 4);
        float v[kMaxSpan];
        [unroll] for (uint z = 0; z < kMaxSpan; ++z) v[z] = 0.0f;
        if (idx >= 0) kv_span((uint)idx, d0, span, v);
        [unroll] for (uint hh = 0; hh < kMaxHeadsPerWg; ++hh) {
            if (hh >= g) continue;
            float p = 0.0f;
            if (idx >= 0) {
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
                    if (i >= span) continue;
                    p += gPool[hh * pc.head_dim + i * 32 + lane] * v[i];
                }
            }
            // A KV row is exactly one wave's lanes, so this is one
            // WaveActiveSum -- and it must be, because the t loop is not
            // uniform across the workgroup and an LDS tree would deadlock.
            const float s0 = WaveActiveSum(p);
            const float s  = (idx >= 0) ? (s0 * pc.softmax_scale) : -3.0e38f;
            hmax[hh] = max(hmax[hh], s);
            if (lane == 0)
                *(float*)(Ptr[kScore] + ((uint64_t)(h0 + hh) * pc.score_stride + t) * 4) = s;
        }
    }

    // Control flow is uniform again: every wave has left the t loop.
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        [unroll] for (uint hh = 0; hh < kMaxHeadsPerWg; ++hh)
            gRed[wave * kMaxHeadsPerWg + hh] = hmax[hh];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < g) {
        float m = -3.0e38f;
        for (uint w = 0; w < waves; ++w) m = max(m, gRed[w * kMaxHeadsPerWg + tid]);
        *(float*)(Ptr[kTileMax] + ((uint64_t)(h0 + tid) * pc.n_tiles + tl) * 4) = m;
    }
}

// --- stage 1: p against the FINAL max, and P.V over this tile ---------------
// ONE pass and two barriers. A wave owns a disjoint subset of the tile's
// positions -- `t = t0 + grp, t0 + grp + ngrp, ...` -- and all 32 of its lanes
// visit exactly those positions, sixteen output dims apiece. So `p` for a
// position is the same number in every lane of the wave: it is computed right
// there, in wave form, rounded to bf16 against the final row max exactly where
// `acc_s_cast` rounds it, added to the denominator partial and multiplied into
// the KV span in the same iteration.
//
// sparse_attn.slang's combine does this in three steps instead -- p over 256
// threads written back to the score plane, an LDS tree over those 256 threads
// for the denominator, a device-memory barrier, then P.V reading p back -- which
// is thirteen workgroup barriers and a global write and read per position. The
// numbers are the same; the denominator's association is not (§13 prints the
// difference).
//
// At `pv_tiles == 1` there is nothing left to combine, so this stage divides,
// rotates and writes the final output itself and AttnFinishT is not dispatched
// (AttnRunner::attn_tiled_finish_needed).
void stage_pv(uint gid, uint tid) {
    fp8_table_fill(tid);
    const uint hg = gid / pc.pv_tiles, tl = gid % pc.pv_tiles;
    const uint h0 = hg * HeadsPerWg;
    const uint hn = min(h0 + HeadsPerWg, pc.n_heads);
    const uint g    = hn - h0;
    const uint tph  = kWg / g;                    // threads on one head, >= 32
    const uint sub  = tid % tph;
    const uint ngrp = tph / 32;                   // position subsets = waves per head
    const uint grp  = sub / 32;
    const uint dl   = sub % 32;
    const uint span = pc.head_dim / 32;           // 16 dims a lane, always
    const uint d0   = dl * span;
    const uint head = h0 + tid / tph;
    const uint t0   = tl * pc.pv_tile_len;
    const uint t1   = min(t0 + pc.pv_tile_len, pc.n_kv);
    GroupMemoryBarrierWithGroupSync();            // gTbl

    const float mx = row_max(head);
    const uint64_t sc = Ptr[kScore] + (uint64_t)head * pc.score_stride * 4;

    float acc[kMaxSpan];
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) acc[i] = 0.0f;
    float dsum = 0.0f;
    for (uint t = t0 + grp; t < t1; t += ngrp) {
        const int idx = (int)load_u32(Ptr[kTopIdx] + (uint64_t)t * 4);
        if (idx < 0) continue;
        const float p = bf16_decode(bf16_encode(exp(load_f32(sc + (uint64_t)t * 4) - mx)));
        dsum += p;
        if (p == 0.0f) continue;
        float v[kMaxSpan];
        kv_span((uint)idx, d0, span, v);
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
            if (i >= span) continue;
            acc[i] += p * v[i];
        }
    }

    // The `ngrp` waves of one head meet in LDS: sixteen P.V partials and the
    // denominator partial per thread, at the odd 17-float stride that keeps
    // the 32 lanes in 32 banks.
    if (ngrp > 1) {
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) gPool[tid * kPvStride + i] = acc[i];
        gPool[tid * kPvStride + kMaxSpan] = dsum;
        GroupMemoryBarrierWithGroupSync();
        if (grp == 0) {
            for (uint gg = 1; gg < ngrp; ++gg) {
                const uint b = (tid + gg * 32) * kPvStride;
                [unroll] for (uint i = 0; i < kMaxSpan; ++i) acc[i] += gPool[b + i];
                dsum += gPool[b + kMaxSpan];
            }
        }
    }
    if (grp != 0) return;

    if (pc.pv_tiles == 1u) {
        // The whole row is this tile: finish here. The sink enters the
        // denominator once and never competed for the max.
        const float inv = 1.0f / (dsum + exp(load_f32(Ptr[kSink] + (uint64_t)head * 4) - mx));
        float o[kMaxSpan];
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) o[i] = acc[i] * inv;
        const uint rope_lo = pc.head_dim - pc.rope_dim;
        // `span` and `d0` are both even, so a rotation pair is owned by one lane.
        [unroll] for (uint i = 0; i + 1 < kMaxSpan; i += 2) {
            if (i + 1 >= span) continue;
            const uint d = d0 + i;
            if (d < rope_lo) continue;
            const uint j = (d - rope_lo) >> 1;
            const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
            const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
            rope_pair(c, s, o[i], o[i + 1], true);
        }
        [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
            if (i >= span) continue;
            *(float*)(Ptr[kO] + ((uint64_t)head * pc.head_dim + d0 + i) * 4) = o[i];
        }
        return;
    }

    const uint64_t po = Ptr[kPartO] + ((uint64_t)tl * pc.part_stride
                                       + (uint64_t)head * pc.head_dim + d0) * 4;
    [unroll] for (uint i = 0; i < kMaxSpan; ++i) {
        if (i >= span) continue;
        *(float*)(po + (uint64_t)i * 4) = acc[i];
    }
    if (dl == 0)
        *(float*)(Ptr[kPartD] + ((uint64_t)tl * pc.n_heads + head) * 4) = dsum;
}

// --- stage 2: combine the tiles, divide, inverse RoPE -----------------------
// A thread owns one adjacent PAIR of output dims, which is what the inverse
// rotation wants; `ceil(n_heads * head_dim / 512)` workgroups.
void stage_finish(uint gid, uint tid) {
    const uint e = (gid * kWg + tid) * 2;
    if (e >= pc.n_heads * pc.head_dim) return;
    const uint head = e / pc.head_dim;
    const uint d    = e % pc.head_dim;

    float den = 0.0f, a0 = 0.0f, a1 = 0.0f;
    for (uint z = 0; z < pc.pv_tiles; ++z) {
        den += load_f32(Ptr[kPartD] + ((uint64_t)z * pc.n_heads + head) * 4);
        const uint64_t po = Ptr[kPartO] + ((uint64_t)z * pc.part_stride + e) * 4;
        a0 += load_f32(po);
        a1 += load_f32(po + 4);
    }
    // The sink enters the denominator once, and never competed for the max.
    den += exp(load_f32(Ptr[kSink] + (uint64_t)head * 4) - row_max(head));

    const float inv = 1.0f / den;
    a0 *= inv;
    a1 *= inv;
    const uint rope_lo = pc.head_dim - pc.rope_dim;
    if (d >= rope_lo) {
        const uint j = (d - rope_lo) >> 1;
        const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
        const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
        rope_pair(c, s, a0, a1, true);            // inverse: conjugate
    }
    *(float*)(Ptr[kO] + (uint64_t)e * 4) = a0;
    *(float*)(Ptr[kO] + (uint64_t)(e + 1) * 4) = a1;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_score(gid.x, tid);
    else if (Stage == 1) stage_pv(gid.x, tid);
    else                 stage_finish(gid.x, tid);
}
