// wo_a -- grouped output down-projection, fp8 block diagonal 8 x [1024 x 4096],
// 33.6 MB (design §7.6).
//
// Dispatch #6 of design §7.14. The group index comes from the high bits of the
// workgroup id, so all eight blocks are one dispatch; the input o is sliced
// 4096 dims per group.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.6): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint8_t> W;   // fp8 [8192, 4096]
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t> S;   // e8m0 [256, 128]
[[vk::binding(2, 0)]] StructuredBuffer<half>    O;   // o[8][4096]
[[vk::binding(3, 0)]] RWStructuredBuffer<half>  Oa;  // [8][1024]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
