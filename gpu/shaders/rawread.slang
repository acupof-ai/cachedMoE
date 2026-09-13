// rawread -- the upper bound every GEMV kernel of design §7.1 rule 2 is scored
// against: a compute shader that does nothing but stream a buffer through the
// memory system.
//
// Each lane loads 16 B per iteration (`uint4`), four loads deep so the wave has
// enough outstanding requests to saturate, and XOR/adds them into a register so
// nothing is dead-code eliminated. Consecutive lanes read consecutive 16 B, so
// one wave32 iteration is a 512 B burst and one workgroup iteration is 4 KiB.
//
// Each workgroup owns one contiguous slice of the buffer (`per_group` uint4s),
// which is the access pattern the MoE kernels have: many workgroups, each
// walking its own expert rows forward.
//
// bench/bw_matrix runs this over path A memory (DEVICE_LOCAL|HOST_VISIBLE),
// path B memory (VK_EXT_external_memory_host import) and, where the driver
// exposes one, a DEVICE_LOCAL-only type, which answers design §3.3.

[[vk::binding(0, 0)]] StructuredBuffer<uint4>   Src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> Sink;   // one uint4 per invocation

struct RawReadPush {
    uint per_group;    // uint4 elements per workgroup; a multiple of 1024
    uint reserved0;
    uint reserved1;
    uint reserved2;
};
[[vk::push_constant]] RawReadPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint base = gid.x * pc.per_group;
    uint4 acc = uint4(0, 0, 0, 0);
    for (uint o = 0; o + 1024 <= pc.per_group; o += 1024) {
        uint4 a = Src[base + o +   0 + lane];
        uint4 b = Src[base + o + 256 + lane];
        uint4 c = Src[base + o + 512 + lane];
        uint4 d = Src[base + o + 768 + lane];
        acc += (a ^ b) + (c ^ d);
    }
    // One store per invocation (16 B, ~0.1% of the read traffic) so the loads
    // cannot be dead-code eliminated and no two lanes race on the same word.
    Sink[gid.x * 256 + lane] = acc;
}
