// wq_b -- Q up-projection + q_norm + RoPE, fp8 [32768 x 1280], 41.9 MB
// (design §7.3).
//
// Dispatch #3 of design §7.14, one of the two biggest attention reads.
// q_norm (RMS over 1280) is computed in LDS before the weight loop. RoPE
// touches only the last 64 of each 512-wide head, so rows with d >= 448 are
// rotated in pairs (d, d+32) on write-out and the same lane owns both rows.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.3): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint8_t> W;       // fp8 [32768, 1280]
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t> S;       // e8m0 [1024, 40]
[[vk::binding(2, 0)]] StructuredBuffer<half>    Qr;      // qr[1280]
[[vk::binding(3, 0)]] StructuredBuffer<float>   RopeTab; // cos/sin, YaRN scaled
[[vk::binding(4, 0)]] RWStructuredBuffer<half>  Q;       // q[64][512]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
