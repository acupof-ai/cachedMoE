// probe_mma -- the fp16 cooperative-matrix rate (bench/probes/model_probe.cpp):
// every wave runs `n` rounds of 8 independent 16x16x16 fp16 x fp16 -> fp32
// multiply-adds on tiles held in registers, so nothing but the matrix unit is
// timed. The ceiling the prefill model (tools/prefill_model.py) scores every
// GEMM and attention tile multiply against. The tile values come from a push
// constant the compiler cannot see, and the accumulators end in a store.
//
// Wave32 is required (the pipeline sets it): 256 threads are 8 waves.

[[vk::binding(0, 0)]] StructuredBuffer<uint4>   Src;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> Sink;    // 8 tiles a wave, every workgroup the same 64 KB
[[vk::binding(2, 0)]] StructuredBuffer<uint>    Chase;

typealias TileA = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>;
typealias TileB = linalg.CoopMat<half,  MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>;
typealias TileC = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>;

struct ModelPush {
    uint mode, n, per_group, base;
};
[[vk::push_constant]] ModelPush pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    const TileA a = TileA(half(float(pc.per_group) * 1e-3f));
    const TileB b = TileB(half(float(pc.base) * 1e-3f));
    TileC c[8];
    [ForceUnroll] for (uint t = 0; t < 8; ++t) c[t] = TileC(float(t));
    for (uint i = 0; i < pc.n; ++i)
        [ForceUnroll] for (uint t = 0; t < 8; ++t)
            c[t] = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a, b, c[t]);
    [ForceUnroll] for (uint t = 0; t < 8; ++t)
        c[t].Store<linalg.CoopMatMatrixLayout.RowMajor>(Sink, ((lane / 32) * 8 + t) * 256, 16);
}
