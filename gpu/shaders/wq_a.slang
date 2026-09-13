// wq_a -- Q down-projection, fp8 E4M3 [1280 x 5120], 6.6 MB (design §7.3).
//
// Dispatch #2 of design §7.14. Scales are a [40, 160] E8M0 tile plane:
// scale[row/32][k/32], one ldexp per 32 K elements. The output qr[1280] feeds
// q_norm, which is folded into wq_b input loading rather than being its own
// dispatch.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.3): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint8_t>  W;   // fp8 e4m3 [1280, 5120]
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t>  S;   // e8m0 [40, 160]
[[vk::binding(2, 0)]] StructuredBuffer<half>     U;   // u[5120]
[[vk::binding(3, 0)]] RWStructuredBuffer<half>   Qr;  // qr[1280]

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
