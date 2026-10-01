#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Variant B: the table as a function-local const array, which lets LLVM choose
// selects or bit arithmetic instead of a memory lookup.
__kernel void fp4_block(__global const uint4* W, __global const half2* X,
                        __global float* Out, uint blocks) {
    const float t[16] = { 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
    const uint gid = get_global_id(0);
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        const uint4 a = W[gid * blocks + b];
        for (uint i = 0; i < 4; ++i) {
            const uint ai = ((const uint*)&a)[i];
            for (uint j = 0; j < 8; ++j) {
                const half2 x = X[((gid * blocks + b) * 32 + i * 8 + j) / 2];
                acc += t[(ai >> (4 * j)) & 0xFu] * (float)(((j & 1) ? x.y : x.x));
            }
        }
    }
    Out[gid] = acc;
}
