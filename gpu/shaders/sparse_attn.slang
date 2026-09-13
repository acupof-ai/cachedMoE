// sparse_attn -- MQA over window 128 + compressed top-512, with inverse RoPE
// fused into the write-out (design §7.5).
//
// Dispatch #5 of design §7.14. 640 KV entries x 512 B = 320 KiB and ~42 MFLOP:
// a pure latency kernel, target under 30 us. One workgroup handles all 64 heads
// because MQA gives them a single shared KV; KV arrives in 16 KiB tiles of 32
// entries, double buffered inside the 32 KiB LDS budget of design §1.1. The
// per-head attn_sink logit seeds the online-softmax max and sum. Output is
// written in the 8 x 4096 grouping that wo_a wants.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.5): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<half>    Q;      // q[64][512] fp16, RoPE applied
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t> WinKv;  // fp8 window ring
[[vk::binding(2, 0)]] StructuredBuffer<uint8_t> CompKv; // fp8 or fp4 compressed KV
[[vk::binding(3, 0)]] StructuredBuffer<uint>    TopIdx; // indexer top-512 positions
[[vk::binding(4, 0)]] StructuredBuffer<float>   Sink;   // attn_sink[64]
[[vk::binding(5, 0)]] StructuredBuffer<float>   RopeTab;
[[vk::binding(6, 0)]] RWStructuredBuffer<half>  O;      // o[8][4096]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
