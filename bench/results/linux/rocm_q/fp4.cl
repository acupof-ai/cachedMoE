#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// The FP4 (E2M1) weight-block decode of prefill_gemm.slang's fmt==3 path,
// written for LLVM's AMDGPU backend so its codegen can be counted against
// ACO's for the same arithmetic: 32 nibbles out of 16 bytes, each a lookup in
// a 16-entry fp32 constant table, then 32 FMAs against fp16 activations.
__constant float kE2M1[16] = { 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

__kernel void fp4_block(__global const uint4* W, __global const half* X,
                        __global float* Out, uint blocks) {
    const uint gid = get_global_id(0);
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        const uint4 a = W[gid * blocks + b];
        float w[32];
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 8; ++j)
                w[i * 8 + j] = kE2M1[(((const uint*)&a)[i] >> (4 * j)) & 0xFu];
        for (uint i = 0; i < 32; ++i)
            acc += w[i] * (float)X[(gid * blocks + b) * 32 + i];
    }
    Out[gid] = acc;
}
