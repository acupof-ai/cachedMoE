// wkv -- latent KV projection, kv_norm, RoPE and the fp8 window-cache write
// (design §7.4, dispatch #4 of the §7.14 list). fp8 [512 x 5120], 2.6 MB.
//
// Tiny but on the critical path: sparse attention cannot start until this
// token's KV row is in the ring.
//
// Two stages of one pipeline, because kv_norm is an RMS statistic over all 512
// outputs and 2.6 MB cannot be read by the single workgroup that would need to
// own them all:
//   Stage 0  rows / (256 / LanesPerRow) workgroups: the GEMV, plus a partial
//            sum of squares per workgroup.
//   Stage 1  one workgroup, one adjacent pair per thread: finish the norm,
//            rotate the last `rope_dim` dims, then `act_quant` the WHOLE
//            512-wide vector -- RoPE tail included, which is what
//            `Attention._window_kv` does -- and store the E4M3 bytes and the
//            UE8M0 scale byte into the ring slot.
//
// The dequantised value is written out as well. That is not redundancy: the
// attention reads this position's KV back out of the ring like any other, so
// the fp32 copy exists only for the oracle, which needs the post-quantisation
// value to compare against.

// The LDS budget for the staged activation: wkv's K is the model dim = 5120, and a
// workgroup never stages more (fp8_gemv.slang).
#define DEEPMOE_GEMV_MAX_K 5120
// 512 KV dims / 32 = 16 fp8 blocks; finish_body's per-block amax lives in LDS.
#define DEEPMOE_WKV_MAX_BLOCKS 16

#include "fp8_gemv.slang"

static const uint kW = 0, kS = 1, kX = 2, kNormW = 3, kRope = 4;
static const uint kKvRaw = 5, kScratch = 6, kKvVal = 7, kKvScale = 8, kKv = 9;
// The fp32 partial plane gemv_ksplit.slang's stage 0 writes, read by stage 2.
static const uint kPart = 10;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct WkvPush {
    uint  rows;         // head_dim = 512
    uint  k;            // dim = 5120
    uint  scale_cols;   // k / 32 = 160
    uint  rope_dim;     // 64
    uint  slot;         // ring slot = start_pos % window_size
    uint  n_wg0;        // workgroups stage 0 used
    float norm_eps;     // 1e-20
    uint  part_stride;  // stage 2 only: floats between K-slices in kPart
};
[[vk::push_constant]] WkvPush pc;

void stage_gemv(uint gid, uint tid) {
    fp8_table_fill(tid);
    stage_x_fp8(Ptr[kX], pc.k, tid,
                Ptr[kS] + (uint64_t)((gid * gemv_rows_per_wg()) >> 5) * pc.scale_cols);
    GroupMemoryBarrierWithGroupSync();

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = gemv_row_base(gid, tid);
    float acc[kMaxRowsPerLane];
    gemv8_rows(Ptr[kW], Ptr[kS], rowBase, pc.k, pc.scale_cols, sublane, acc);
    float sq = 0.0f;
    // Unrolled against the compile-time bound, predicated on the
    // specialisation constant: see gemv8_rows in fp8_gemv.slang.
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float y = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows) {
            *(float*)(Ptr[kKvRaw] + (uint64_t)(rowBase + r) * 4) = y;
            sq += y * y;
        }
    }
    const float tot = wg_reduce(sq, tid);
    if (tid == 0) *(float*)(Ptr[kScratch] + (uint64_t)gid * 4) = tot;
}

// Everything after the RMS statistic: the norm, the RoPE tail, the fp8 block
// quantisation and the ring write. Split out of `stage_finish` so the K-split
// combine (stage 2) can reach it with an `rs` it computed itself instead of one
// stage 0 left in scratch.
//
// Every thread takes one adjacent PAIR of the 512 dims -- a RoPE pair is never
// split -- and a 32-wide block's amax is an LDS atomic max over its sixteen
// threads (non-negative fp32 bits order like the floats). The first version
// gave each of 16 threads a whole block in series and measured 13 us a layer;
// the arithmetic per element is unchanged, and max does not care about order,
// so the bytes are the same. Every thread of the workgroup must call this: it
// has a barrier.
groupshared uint gBlockAmax[DEEPMOE_WKV_MAX_BLOCKS];

void finish_body(uint tid, float rs) {
    const uint blocks = pc.rows / kBlockK;          // 16
    const uint d = tid * 2;
    const bool live = d < pc.rows;
    const uint rope_lo = pc.rows - pc.rope_dim;
    if (tid < blocks) gBlockAmax[tid] = 0u;

    float v0 = 0.0f, v1 = 0.0f;
    if (live) {
        // kv_norm is an RMSNorm, so its result is bf16 before RoPE touches it.
        v0 = bf16_round(load_f32(Ptr[kKvRaw] + (uint64_t)d * 4) * rs
                        * bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)d * 2)));
        v1 = bf16_round(load_f32(Ptr[kKvRaw] + (uint64_t)(d + 1) * 4) * rs
                        * bf16_decode(load_u16(Ptr[kNormW] + (uint64_t)(d + 1) * 2)));
        if (d >= rope_lo) {
            const uint j = (d - rope_lo) >> 1;
            const float c = load_f32(Ptr[kRope] + (uint64_t)j * 8);
            const float s = load_f32(Ptr[kRope] + (uint64_t)j * 8 + 4);
            rope_pair(c, s, v0, v1, false);
            // apply_rotary_emb writes back into the bf16 tensor it was handed.
            v0 = bf16_round(v0);
            v1 = bf16_round(v1);
        }
    }
    GroupMemoryBarrierWithGroupSync();            // gBlockAmax cleared
    const uint blk = d / kBlockK;
    if (live) InterlockedMax(gBlockAmax[blk], asuint(max(abs(v0), abs(v1))));
    GroupMemoryBarrierWithGroupSync();
    if (!live) return;

    const float sc = act_scale(asfloat(gBlockAmax[blk]));
    const float inv = 1.0f / sc;
    const uint64_t val = Ptr[kKvVal] + (uint64_t)pc.slot * pc.rows + d;
    const uint b0 = fp8_encode_rn(clamp(v0 * inv, -kFp8MaxF, kFp8MaxF));
    const uint b1 = fp8_encode_rn(clamp(v1 * inv, -kFp8MaxF, kFp8MaxF));
    *(uint16_t*)val = (uint16_t)(b0 | (b1 << 8));
    *(float*)(Ptr[kKv] + (uint64_t)d * 4) = fp8_decode(b0) * sc;
    *(float*)(Ptr[kKv] + (uint64_t)(d + 1) * 4) = fp8_decode(b1) * sc;
    // UE8M0 byte: a power-of-two fp32's exponent field is already the code.
    if ((d % kBlockK) == 0)
        *(uint8_t*)(Ptr[kKvScale] + (uint64_t)pc.slot * blocks + blk) =
            (uint8_t)((asuint(sc) >> 23) & 0xFFu);
}

void stage_finish(uint tid) {
    float sum = 0.0f;
    for (uint i = 0; i < pc.n_wg0; ++i) sum += load_f32(Ptr[kScratch] + (uint64_t)i * 4);
    finish_body(tid, rsqrt(sum / float(pc.rows) + pc.norm_eps));
}

// --- stage 2: the K-split combine, with the whole finish folded in ----------
// gemv_ksplit.slang's stage 0 leaves `KSplit` fp32 partials per row; this adds
// them, takes the RMS statistic over the result in the same dispatch, and then
// runs the norm/RoPE/quantise/ring-write tail. So the K-split costs wkv no
// extra dispatch at all: stages 0+1 (gemv + finish) become split + this one.
//
// One workgroup. 512 rows over 256 threads is two rows each, and the sum of
// squares is the workgroup reduction `stage_finish` used to read out of the
// scratch buffer stage 0's workgroups wrote.
void stage_kcombine(uint tid) {
    float sq = 0.0f;
    for (uint r = tid; r < pc.rows; r += kWg) {
        float y = 0.0f;
        for (uint s = 0; s < KSplit; ++s)
            y += load_f32(Ptr[kPart] + ((uint64_t)s * pc.part_stride + r) * 4);
        *(float*)(Ptr[kKvRaw] + (uint64_t)r * 4) = y;
        sq += y * y;
    }
    const float tot = wg_reduce(sq, tid);
    // kv_raw went to DEVICE memory and finish_body reads rows this thread did
    // not write, so the barrier has to cover the buffer and not only LDS.
    AllMemoryBarrierWithGroupSync();
    finish_body(tid, rsqrt(tot / float(pc.rows) + pc.norm_eps));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_gemv(gid.x, tid);
    else if (Stage == 1) stage_finish(tid);
    else                 stage_kcombine(tid);
}
