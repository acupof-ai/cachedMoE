// probe_stream -- rawread with one extra knob: whether the workgroups read
// DISJOINT slices or the SAME bytes (Track W, docs/plan_p5.md §5).
//
// rawread.slang gives workgroup g the slice [g * per_group, (g+1) * per_group),
// which measures aggregate bandwidth over a working set that grows with the
// workgroup count. Two questions it cannot answer:
//
//   (g) Is sharing free? If N workgroups reading the SAME bytes each get the
//       same per-workgroup rate as one workgroup does, then merging two
//       consumers of one weight stream into one workgroup buys nothing --
//       which retires a whole class of "read the KV once" ideas before anyone
//       writes the kernel (docs/STATUS.md §3 items 10 and 11 are exactly that
//       class, measured the expensive way).
//
//   (h) Where is the capacity cliff? Sweeping the working set at a FIXED
//       reader count separates a cache-capacity effect (the rate falls once
//       and then stays flat) from a paging/TLB effect (it keeps falling).
//       Strix Halo has a 32 MB MALL between the CUs and LPDDR5X and the answer
//       decides whether a tile size is worth tuning at all.
//
// `share = 1` points every workgroup at offset 0, so `per_group` is the whole
// working set and every workgroup walks all of it. `share = 0` is rawread's
// behaviour.
//
// The loop body is rawread's, unchanged, so the two shaders' numbers are
// comparable: 16 B per lane per load, four loads deep, one uint4 store per
// invocation at the end so nothing is eliminated.

[[vk::binding(0, 0)]] StructuredBuffer<uint4>   Src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> Sink;   // one uint4 per invocation

struct ProbeStreamPush {
    uint per_group;    // uint4 elements one workgroup walks; a multiple of 1024
    uint share;        // 0 = disjoint slices (rawread), 1 = every group at 0
    uint repeats;      // walk the slice this many times inside one dispatch
    uint reserved;
};
[[vk::push_constant]] ProbeStreamPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint base = (pc.share != 0u) ? 0u : gid.x * pc.per_group;
    const uint reps = max(pc.repeats, 1u);
    uint4 acc = uint4(0, 0, 0, 0);
    for (uint r = 0; r < reps; ++r) {
        for (uint o = 0; o + 1024 <= pc.per_group; o += 1024) {
            uint4 a = Src[base + o +   0 + lane];
            uint4 b = Src[base + o + 256 + lane];
            uint4 c = Src[base + o + 512 + lane];
            uint4 d = Src[base + o + 768 + lane];
            acc += (a ^ b) + (c ^ d);
        }
    }
    Sink[gid.x * 256 + lane] = acc;
}
