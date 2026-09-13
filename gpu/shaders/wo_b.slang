// wo_b -- output up-projection, fp8 [5120 x 8192], 41.9 MB (design §7.6).
//
// Dispatch #7 of design §7.14, the other big attention read. Its output a[5120]
// is the hc_post input of the next mega_mhc (design §7.7).
//
// Placeholder: bindings and entry point only.
// TODO(design §7.6): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint8_t> W;   // fp8 [5120, 8192]
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t> S;   // e8m0 [160, 256]
[[vk::binding(2, 0)]] StructuredBuffer<half>    Oa;  // [8192]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> A;   // a[5120] fp32

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
