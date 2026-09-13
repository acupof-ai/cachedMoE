// moe_down -- FP4 w2 projection with the 7-way reduction folded in
// (design §7.9, dispatch B).
//
// Dispatch #11 of design §7.14. One workgroup owns a row block and loops over
// all seven experts internally, so the sum happens in registers: no atomics and
// no third kernel. rows_per_wg = 8 gives 640 workgroups for 40 CUs.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.9): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> ExpertAddr;
[[vk::binding(1, 0)]] StructuredBuffer<uint>     Ids;
[[vk::binding(2, 0)]] StructuredBuffer<half>     H;   // h[7][2304]
[[vk::binding(3, 0)]] RWStructuredBuffer<float>  Y;   // y[5120] fp32

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
