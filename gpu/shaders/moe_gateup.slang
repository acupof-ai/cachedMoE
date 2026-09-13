// moe_gateup -- fused FP4 gate/up + SwiGLU for 7 experts (design §7.9, dispatch A).
//
// Dispatch #10 of design §7.14: 6 routed experts at 12.5 MB each plus the fp8
// shared expert at 23.6 MB. Expert base addresses come from the pointer table
// of design §5.3 through buffer_device_address, so there is no descriptor per
// expert.
//
// Per lane: 32 K elements = 16 B of packed FP4 plus one E8M0 scale; decode,
// FMA against x in LDS, ldexp the block, accumulate in fp32. Tail:
// gate = min(gate, 10), up = clamp(up, -10, 10), h = silu(gate) * up * w_route.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.9): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint64_t> ExpertAddr; // pointer table addr[layer][expert]
[[vk::binding(1, 0)]] StructuredBuffer<uint>     Ids;        // top-6 expert ids
[[vk::binding(2, 0)]] StructuredBuffer<float>    Weights;    // routing weights
[[vk::binding(3, 0)]] StructuredBuffer<half>     X;          // x[5120] fp16
[[vk::binding(4, 0)]] RWStructuredBuffer<half>   H;          // h[7][2304] fp16

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
