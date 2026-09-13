// wkv -- latent KV projection + kv_norm + RoPE + fp8 window-cache write,
// fp8 [512 x 5120], 2.6 MB (design §7.4).
//
// Dispatch #4 of design §7.14. Tiny but latency critical: it gates the sparse
// attention. On a kv_source layer (2, 8, 14, 20) the compressor and indexer
// kernels chain off this output. The window cache is a 128-slot ring of
// 512 B + scale, block quantised to fp8 exactly as model.py does it, so the
// oracle can compare layer by layer.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.4): body, plus the compressor and indexer variants. P2.
[[vk::binding(0, 0)]] StructuredBuffer<uint8_t>   W;       // fp8 [512, 5120]
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t>   S;       // e8m0 [16, 160]
[[vk::binding(2, 0)]] StructuredBuffer<half>      U;       // u[5120]
[[vk::binding(3, 0)]] StructuredBuffer<float>     RopeTab;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint8_t> KvCache; // ring: 128 x (512 B + scale)
[[vk::binding(5, 0)]] RWStructuredBuffer<half>    Kv;      // kv[512] for this position

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
