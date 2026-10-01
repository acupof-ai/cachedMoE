#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Variant C: the decode written as bit arithmetic by hand, which is the floor
// both compilers are measured against. E2M1 nibble s|ee|m -> fp32:
//   ee != 0 : (1 + m/2) * 2^(ee-1)  = sign | (126+ee)<<23 | m<<22
//   ee == 0 : m * 0.5               (the one case the splice cannot express)
__kernel void fp4_block(__global const uint4* W, __global const half2* X,
                        __global float* Out, uint blocks) {
    const uint gid = get_global_id(0);
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        const uint4 a = W[gid * blocks + b];
        for (uint i = 0; i < 4; ++i) {
            const uint ai = ((const uint*)&a)[i];
            for (uint j = 0; j < 8; ++j) {
                const uint n  = (ai >> (4 * j)) & 0xFu;
                const uint ee = (n >> 1) & 3u;
                const uint m  = n & 1u;
                const uint bits = (n << 28 & 0x80000000u) |
                                  ((ee != 0u) ? (((126u + ee) << 23) | (m << 22))
                                              : (m ? 0x3F000000u : 0u));
                const half2 x = X[((gid * blocks + b) * 32 + i * 8 + j) / 2];
                acc += as_float(bits) * (float)(((j & 1) ? x.y : x.x));
            }
        }
    }
    Out[gid] = acc;
}
