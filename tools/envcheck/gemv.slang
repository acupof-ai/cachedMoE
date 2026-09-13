// Slang smoke test: a trivial FP4(E2M1)-dequant GEMV skeleton to prove the toolchain path
// slangc -> SPIR-V with int8 dot product + subgroup ops available.
[[vk::binding(0, 0)]] StructuredBuffer<uint>   W;      // packed fp4, 8 per uint
[[vk::binding(1, 0)]] StructuredBuffer<uint8_t> S;     // E8M0 scales per 32-block
[[vk::binding(2, 0)]] StructuredBuffer<half>   X;      // activation
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y;

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
