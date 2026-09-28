// probe_model -- the constants of the per-dispatch GPU model (tools/gpu_model.py,
// bench/probes/model_probe.cpp): what a dispatch costs before it does
// anything, how fast W workgroups can stream B bytes, what one barrier / one
// wave reduction / one dependent load costs, and the fp32 FMA rate.
//
// Every mode ends in a store the host never reads but the compiler cannot
// prove dead, so nothing is eliminated.
//
//   0 empty    nothing (the dispatch + barrier overhead)
//   1 stream   probe_stream's disjoint-slice loop: workgroup g reads
//              [g * per_group, (g+1) * per_group) uint4s, 4 loads deep
//   2 barrier  one workgroup, `n` rounds of LDS write + barrier + read + barrier
//   3 chase    lane 0 of workgroup 0 follows `n` dependent loads through Chase
//   4 fma      every thread, `n` rounds of 8 independent fp32 FMA chains
//   5 wave     `n` dependent WaveActiveSum reductions in every wave

[[vk::binding(0, 0)]] StructuredBuffer<uint4>   Src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> Sink;    // one uint4 per invocation
[[vk::binding(2, 0)]] StructuredBuffer<uint>    Chase;

struct ModelPush {
    uint mode;
    uint n;
    uint per_group;    // mode 1: uint4 elements per workgroup, a multiple of 1024
    uint base;         // mode 1: first uint4 element (the cache flush reads elsewhere)
};
[[vk::push_constant]] ModelPush pc;

groupshared float gL[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint out = gid.x * 256 + lane;
    if (pc.mode == 0u) {
        if (pc.n == 0xFFFFFFFFu) Sink[out] = uint4(lane, 0, 0, 0);
    } else if (pc.mode == 1u) {
        const uint base = pc.base + gid.x * pc.per_group;
        uint4 acc = uint4(0, 0, 0, 0);
        for (uint o = 0; o + 1024 <= pc.per_group; o += 1024) {
            uint4 a = Src[base + o +   0 + lane];
            uint4 b = Src[base + o + 256 + lane];
            uint4 c = Src[base + o + 512 + lane];
            uint4 d = Src[base + o + 768 + lane];
            acc += (a ^ b) + (c ^ d);
        }
        Sink[out] = acc;
    } else if (pc.mode == 2u) {
        float v = float(lane);
        for (uint i = 0; i < pc.n; ++i) {
            gL[lane] = v;
            GroupMemoryBarrierWithGroupSync();
            v = gL[(lane + 1u) & 255u] + 1.0f;
            GroupMemoryBarrierWithGroupSync();
        }
        Sink[out] = uint4(asuint(v), 0, 0, 0);
    } else if (pc.mode == 3u) {
        if (gid.x == 0u && lane == 0u) {
            uint idx = 0;
            for (uint i = 0; i < pc.n; ++i) idx = Chase[idx];
            Sink[0] = uint4(idx, 0, 0, 0);
        }
    } else if (pc.mode == 4u) {
        float a = float(lane), b = a + 1.0f, c = a + 2.0f, d = a + 3.0f;
        float e = a + 4.0f, f = a + 5.0f, g = a + 6.0f, h = a + 7.0f;
        for (uint i = 0; i < pc.n; ++i) {
            a = a * 0.9999f + 0.5f; b = b * 0.9999f + 0.5f;
            c = c * 0.9999f + 0.5f; d = d * 0.9999f + 0.5f;
            e = e * 0.9999f + 0.5f; f = f * 0.9999f + 0.5f;
            g = g * 0.9999f + 0.5f; h = h * 0.9999f + 0.5f;
        }
        Sink[out] = uint4(asuint(a + b + c + d), asuint(e + f + g + h), 0, 0);
    } else {
        float v = float(lane);
        for (uint i = 0; i < pc.n; ++i) v = WaveActiveSum(v) * 0.001f + float(lane);
        Sink[out] = uint4(asuint(v), 0, 0, 0);
    }
}
