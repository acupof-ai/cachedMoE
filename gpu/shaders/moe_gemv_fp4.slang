// FP4 (E2M1) dequant GEMV -- the toolchain smoke test that proves
// slangc -> SPIR-V with int8 storage and subgroup ops, and the simplest
// readable statement of what design §7.9's kernels do.
//
// The real MoE kernels are moe_gateup.slang and moe_down.slang: they read the
// weights through the buffer_device_address pointer table instead of a bound
// descriptor (design §5.3), carry the specialisation constants of §7.1, and are
// what bench/kernel_bench measures. This file stays because `deepmoe info` and
// docs/build.md use it to check the shader toolchain without a checkpoint, and
// because it is 20 lines rather than 150.
//
// One wave cooperates on one output row (design §7.1 rule 3). A row of K=5120
// FP4 elements is 2560 B; each lane owns whole 32-element blocks so the E8M0
// scale applies to a register-resident partial sum and is folded in with ldexp,
// exactly as cpu/dequant.cpp does it (the L0/L1 oracle of design §12).
//
// Specialisation constants match gpu/vulkan/pipeline.h PipelineSpec.
[[vk::binding(0, 0)]] StructuredBuffer<uint>    W;      // packed fp4, 8 per uint
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t> S;      // E8M0 scale per 32-block
[[vk::binding(2, 0)]] StructuredBuffer<half>    X;      // activation
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y;      // fp32 accumulation (design §6)

// Low nibble = even element (PyTorch float4_e2m1fn_x2), design appendix A.
static const float kE2M1[16] = { 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6 };

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint row = gid.x;                      // one wave per output row
    const uint K = 5120;
    float acc = 0;
    // each lane owns 32-element blocks strided by wave width
    for (uint blk = lane; blk < K / 32; blk += 64) {
        uint base = (row * K + blk * 32) / 8;
        float part = 0;
        [unroll] for (uint w = 0; w < 4; ++w) {
            uint packed = W[base + w];
            [unroll] for (uint j = 0; j < 8; ++j) {
                uint nib = (packed >> (4 * j)) & 0xF;
                part += kE2M1[nib] * (float)X[blk * 32 + w * 8 + j];
            }
        }
        int e = (int)S[row * (K / 32) + blk] - 127;
        acc += ldexp(part, e);
    }
    acc = WaveActiveSum(acc);
    if (lane == 0) Y[row] = acc;
}
