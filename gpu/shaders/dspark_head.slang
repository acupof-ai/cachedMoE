// The DSpark Markov head and confidence head (design §7.12's third bullet).
//
// What a draft cycle does after the 1.32 GB main head read
// -------------------------------------------------------
//     x      = self.hc_pre(x, pre_mix)                  # [1, 5, 5120]
//     logits = self.head(self.norm(x), full_logits=True)   # [5, 129280] fp32
//     output_ids[0] = input_ids                            # the last accepted token
//     for i in range(5):
//         logits_bias, markov_embed = self.markov_head(output_ids[i])
//         logits[i] += logits_bias
//         output_ids[i + 1] = argmax(logits[i])            # temperature == 0
//     confidence = proj(cat([x, stack(markov_embeds)], -1).float())
//
// `markov_head` is two bf16 `[129280 x 256]` tables: `embed` is an EMBEDDING,
// one row gathered by the token id, and `head` is a GEMV against that row
// producing a 129,280-wide fp32 bias. 66.2 MB of weight read per position, 331 MB
// per cycle -- small beside the head's 1.32 GB, but it happens five times and,
// critically, **sequentially**: position i's bias depends on the token sampled
// at position i - 1. design §7.12 calls this out as five small dispatches, and
// five small dispatches is what this is.
//
// The main head GEMM itself is NOT here. `head.slang` already computes
// `[129280 x 5120]` bf16 against an fp32 activation and stage 1 of this file
// consumes its `[M][129280]` output; duplicating it at M = 5 would be a second
// copy of the single largest read in the model.
//
// Stages
// ------
//   Stage 0  MarkovBias    bias[v] = sum_{d<256} head_w[v][d] * embed[tok][d]
//   Stage 1  AddBiasArgmax logits[pos][v] += bias[v]; argmax over 129280
//   Stage 2  Confidence    c[m] = dot(cat(x[m], markov_embed[m]), proj[5376])
//
// Closing the sequential loop without a host round trip
// -----------------------------------------------------
// Stage 1 writes the sampled id both into the four-word `SampleOut` block and
// into `output_ids[pos + 1]`, and stage 0 will read its token id from
// `output_ids[pos]` instead of the push constant when `kFlagTokenFromBuf` is
// set. So the five (MarkovBias, AddBiasArgmax) pairs can be recorded into one
// command buffer with plain barriers between them, and the host never stalls to
// read a token back. Without the flag, `pc.token` is used and the caller is
// responsible for the readback -- which is what a test driving one stage at a
// time against the oracle wants.
//
// Numerics
// --------
// `ParallelHead.forward` is `F.linear(x.float(), weight)` with the weight
// promoted to fp32, so there is no activation quantisation anywhere in this
// file: bf16 weights decoded to fp32 against an fp32 activation, fp32
// accumulation, fp32 output. The same is true of `DSparkConfidenceHead.forward`,
// which is `self.proj(hidden.float()).squeeze(-1)` -- a BARE projection, **no
// activation function**. It is not a probability and must not be squashed here;
// tools/oracle_dspark.py's `conf_to_prob` is where the logistic assumption
// lives, flagged as an assumption, with `stats.confidence_vs_accept` as the
// evidence for or against it.
//
// One silent failure mode, recorded in design §7.11: `v > best` is false for
// NaN, so a step whose logits are all NaN is reported as token 0 with margin
// 0.0000 rather than as an error. That is not a bug in the sampler, but it is
// how §7.16.4's second bug showed itself, and the `rows` field of the sample
// block is there so a caller can at least tell that the dispatch saw the whole
// vocabulary.

#include "dspark_common.slang"

// Stage 0: MarkovBias.
static const uint kMkW = 0, kMkEmbed = 1, kMkTokenIn = 2, kMkBias = 3, kMkEmbedOut = 4;
// Stage 1: AddBiasArgmax.
static const uint kAbLogits = 0, kAbBias = 1, kAbSample = 2, kAbOutIds = 3;
// Stage 2: Confidence.
static const uint kCfX = 0, kCfEmbed = 1, kCfProj = 2, kCfOut = 3;

[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> Ptr;

struct DsparkHeadPush {
    uint m;             // 2: live draft positions; the loop runs min(M, m)
    uint rows;          // 0, 1: vocab_size = 129280
    uint k;             // 0: dspark_markov_rank = 256.  2: dim = 5120
    uint rank;          // 2: dspark_markov_rank = 256
    uint row_base;      // 0: first vocabulary row of the dispatch
    uint pos;           // 0, 1: which of the five draft positions (0..4)
    uint token;         // 0: the previous token id, unless kFlagTokenFromBuf
    uint x_stride;      // 2: elements between draft columns of x = 5120
    uint logit_stride;  // 1: floats between logit rows = 129280
    uint flags;         // kFlagInBf16 | kFlagTokenFromBuf
};
[[vk::push_constant]] DsparkHeadPush pc;

// The gathered embedding row. `dspark_markov_rank` is 256; 512 leaves room and
// costs 2 KiB of LDS in a file whose other stages use 3.
static const uint kMaxRank = 512;
groupshared float gE[kMaxRank];

groupshared float gBest[kWg];
groupshared uint  gBestId[kWg];
groupshared float gSecond[kWg];

// --- stage 0: the Markov bias ------------------------------------------------
// `markov_head.embed` is indexed IN THE SHADER rather than gathered on the host:
// the token id is one push word (or one buffer word, see kFlagTokenFromBuf), the
// table base is a slot, and the row address is `embed + tok * rank * 2`. Doing
// it host-side would mean either a copy or a descriptor rewrite between the five
// sequential steps, and the whole point of the flag is that there is neither.
//
// Workgroup 0 also republishes the row as fp32 at `kMkEmbedOut[pos]`, which is
// what stage 2 concatenates onto `x`. bf16 -> fp32 is exact, so this is the same
// number the reference stacks, not a re-derivation.
void stage_markov(uint gid, uint tid) {
    const uint tok = ((pc.flags & kFlagTokenFromBuf) != 0u)
                   ? load_u32(Ptr[kMkTokenIn] + (uint64_t)pc.pos * 4)
                   : pc.token;
    const uint64_t erow = Ptr[kMkEmbed] + (uint64_t)tok * (uint64_t)pc.k * 2;
    for (uint i = tid; i < pc.k; i += kWg)
        gE[i] = bf16_decode(load_u16(erow + (uint64_t)i * 2));
    GroupMemoryBarrierWithGroupSync();

    if (gid == 0) {
        for (uint i = tid; i < pc.k; i += kWg)
            *(float*)(Ptr[kMkEmbedOut] + ((uint64_t)pc.pos * pc.k + i) * 4) = gE[i];
    }

    const uint sublane = tid % LanesPerRow;
    const uint rowBase = pc.row_base + gid * ((kWg / LanesPerRow) * RowsPerLane)
                       + (tid / LanesPerRow) * RowsPerLane;

    // RowsPerLane rows walked together: the embedding row is read from LDS once
    // for all of them and their weight loads are independent streams.
    float acc[kMaxRowsPerLane];
    [unroll] for (uint r0 = 0; r0 < kMaxRowsPerLane; ++r0) acc[r0] = 0.0f;
    for (uint base = sublane * 8; base < pc.k; base += LanesPerRow * 8) {
        float x[8];
        [unroll] for (uint j = 0; j < 8; ++j) x[j] = gE[base + j];
        [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
            if (r >= RowsPerLane || rowBase + r >= pc.rows) continue;
            const uint4 v = load16(Ptr[kMkW] + ((uint64_t)(rowBase + r) * pc.k + base) * 2);
            [unroll] for (uint j = 0; j < 4; ++j) {
                acc[r] += bf16_decode(v[j] & 0xFFFFu) * x[j * 2];
                acc[r] += bf16_decode(v[j] >> 16)     * x[j * 2 + 1];
            }
        }
    }
    // Unrolled against the compile-time bound, predicated on the specialisation
    // constant: see gemv8_rows in fp8_gemv.slang.
    [unroll] for (uint r = 0; r < kMaxRowsPerLane; ++r) {
        if (r >= RowsPerLane) continue;
        const float dot = row_reduce(acc[r], tid, sublane);
        if (sublane == 0 && rowBase + r < pc.rows)
            *(float*)(Ptr[kMkBias] + (uint64_t)(rowBase + r) * 4) = dot;
    }
}

// --- stage 1: the bias add and the greedy sample -----------------------------
// One workgroup reduces 129,280 logits to (id, top1, top2), so the only thing
// that ever crosses PCIe is four words. The runner-up is there because design
// §12 L3 asks for the top1-top2 margin at any disagreement with the oracle: a
// divergence at a margin of 1e-6 is a different finding from one at 0.5.
//
// The bias add is fused into the same strided pass rather than run as its own
// dispatch: a thread reads exactly the elements it will later scan, so there is
// no cross-thread visibility to arrange and no barrier between them.
//
// **Ties go to the lower vocabulary id**, which is `torch.argmax`'s rule and
// therefore what the oracle's tokens assume. head.slang reaches that by
// preferring the lower THREAD, which is only the same thing when a thread's ids
// are contiguous -- they are strided here, as they are there -- so this compares
// ids explicitly. The cost is one integer compare per merge step.
void top2_merge(inout float ab, inout uint ai, inout float a2,
                float bb, uint bi, float b2) {
    const bool akeep = (ab > bb) || (ab == bb && ai <= bi);
    if (akeep) { a2 = max(a2, bb); }
    else       { a2 = max(ab, b2); ab = bb; ai = bi; }
}

void stage_argmax(uint tid) {
    const uint64_t L = Ptr[kAbLogits] + (uint64_t)pc.pos * (uint64_t)pc.logit_stride * 4;
    const uint64_t B = Ptr[kAbBias];

    float best = -3.0e38f, second = -3.0e38f;
    uint  bid = 0;
    // Strided so consecutive lanes read consecutive words.
    for (uint i = tid; i < pc.rows; i += kWg) {
        const float v = load_f32(L + (uint64_t)i * 4) + load_f32(B + (uint64_t)i * 4);
        *(float*)(L + (uint64_t)i * 4) = v;
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
        uint* out = (uint*)Ptr[kAbSample];
        out[0] = gBestId[0];
        out[1] = asuint(gBest[0]);
        out[2] = asuint(gSecond[0]);
        out[3] = pc.rows;
        // output_ids[pos + 1], so the next MarkovBias can read it with
        // kFlagTokenFromBuf and the five steps chain on the GPU.
        *(uint*)(Ptr[kAbOutIds] + (uint64_t)(pc.pos + 1) * 4) = gBestId[0];
    }
}

// --- stage 2: the confidence head -------------------------------------------
// `c[m] = dot(cat(x[m] (5120), markov_embed[m] (256)), proj[5376])` in fp32,
// `proj` bf16 in the checkpoint and promoted here. One workgroup, one draft
// position at a time; `ml` is a push constant so `wg_reduce`'s barriers stay in
// uniform control flow.
//
// `x` is `hc_pre(x, pre_mix)` -- the PRE-norm hidden state, the same tensor the
// head's RMSNorm consumes, not its output.
void stage_conf(uint tid) {
    const uint ml = mlive(pc.m);
    const bool xb = in_bf16(pc.flags);
    for (uint m = 0; m < ml; ++m) {
        float s = 0.0f;
        for (uint i = tid; i < pc.k; i += kWg)
            s += bf16_decode(load_u16(Ptr[kCfProj] + (uint64_t)i * 2))
               * act_load(Ptr[kCfX], act_index(m, pc.x_stride, i), xb);
        for (uint i = tid; i < pc.rank; i += kWg)
            s += bf16_decode(load_u16(Ptr[kCfProj] + (uint64_t)(pc.k + i) * 2))
               * load_f32(Ptr[kCfEmbed] + ((uint64_t)m * pc.rank + i) * 4);
        const float c = wg_reduce(s, tid);
        if (tid == 0) *(float*)(Ptr[kCfOut] + (uint64_t)m * 4) = c;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (Stage == 0)      stage_markov(gid.x, tid);
    else if (Stage == 1) stage_argmax(tid);
    else                 stage_conf(tid);
}
