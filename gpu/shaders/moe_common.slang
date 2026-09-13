// Shared pieces of the two FP4 MoE dispatches of design §7.9.
//
// Weight addressing (design §5.1 v0.5, §5.3): an expert's six tensors live in
// the slab slot exactly as they lie in the safetensors shard, and the GPU-side
// pointer table holds one device address per part:
//
//     table[(layer * experts_per_layer + expert) * 6 + part]
//     part order = w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale
//
// Those addresses are `slot_base + run.slot_offset + part.skew`. skew is a
// multiple of 8 (safetensors offsets are 8-byte multiples, sector-aligned run
// starts are 4096-byte multiples), so every load below declares alignment 8 and
// never 16. This is the single hardest constraint the no-repack decision of
// design §5.1 puts on the kernels.
//
// FP4 E2M1: two elements per byte, LOW nibble = the EVEN element along K
// (design appendix A, pinned by cpu/dequant.cpp and the L0 oracle). One UE8M0
// exponent per 32 elements along K; the block sum is scaled with ldexp, which
// is exact, so the kernel and cpu/gemv_fp4_ref differ only in summation order.

// Specialisation constants; ids match gpu/vulkan/pipeline.h PipelineSpec.
[vk::constant_id(0)] const uint M            = 1;    // accumulators per lane (design §7.1 rule 7)
[vk::constant_id(1)] const uint LanesPerRow  = 32;   // A/B over {16, 32, 64} (rule 3)
[vk::constant_id(2)] const uint RowsPerWgUnused = 8; // reserved: rows_per_wg is WG/LanesPerRow here
[vk::constant_id(3)] const uint SubgroupSize = 32;   // applied via VK_EXT_subgroup_size_control
[vk::constant_id(4)] const uint DecodeMode   = 1;    // 0 = const table, 1 = arithmetic, 2 = select tree
[vk::constant_id(5)] const uint HPrecision   = 0;    // 0 = fp16 h, 1 = fp32 h
// Weight rows per lane: the dual of M. M reads the weights once and uses them
// for M activation columns; RowsPerLane reads the activation once and uses it
// for RowsPerLane weight rows. Both are pure register-blocking knobs.
[vk::constant_id(6)] const uint RowsPerLane  = 1;

static const uint kWgSize    = 256;
static const uint kMaxM      = 6;      // design §1.2: verify batch <= 6
static const uint kMaxRows   = 4;      // upper bound on RowsPerLane
static const uint kBlockK    = 32;     // elements per UE8M0 scale
static const uint kChunkBlks = 64;     // blocks staged per outer iteration

// design appendix A: {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit.
static const float kE2M1[16] = { 0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

// The three decode variants design §7.1 rule 5 asks to A/B.
float fp4_decode(uint nib) {
    if (DecodeMode == 0) {
        return kE2M1[nib & 15];
    } else if (DecodeMode == 1) {
        // Build the fp32 bit pattern directly: e2m1 exponent 0 is the
        // subnormal pair {0, 0.5}, otherwise (1 + m/2) * 2^(e-1).
        uint e    = (nib >> 1) & 3;
        uint mbit = nib & 1;
        uint bits = (e == 0) ? (mbit != 0 ? 0x3F000000u : 0u)
                             : (((e + 126u) << 23) | (mbit << 22));
        return asfloat(bits | ((nib & 8) << 28));
    } else {
        float mag = (nib & 4) != 0
            ? ((nib & 2) != 0 ? ((nib & 1) != 0 ? 6.0f : 4.0f) : ((nib & 1) != 0 ? 3.0f : 2.0f))
            : ((nib & 2) != 0 ? ((nib & 1) != 0 ? 1.5f : 1.0f) : ((nib & 1) != 0 ? 0.5f : 0.0f));
        return (nib & 8) != 0 ? -mag : mag;
    }
}

// 16 B of packed FP4 (32 elements) from a device address that is only
// guaranteed 8-byte aligned. Slang emits `OpLoad ... Aligned 4` for a pointer
// dereference, which is exactly the promise the skew arithmetic can keep and
// is enough for RDNA's global_load_dwordx4.
uint4 load_fp4_block(uint64_t addr) {
    return *(uint4*)addr;
}

// One UE8M0 exponent byte.
uint load_scale_byte(uint64_t addr) {
    return (uint)(*(uint8_t*)addr);
}

// Half `j` of a packed pair buffer entry.
float unpack_half(uint4 v, uint j) {
    uint w = v[j >> 1];
    return f16tof32((j & 1) != 0 ? (w >> 16) : (w & 0xFFFFu));
}

// design §2.4: gate is clamped above only, up on both sides, then SwiGLU.
float swiglu(float gate, float up, float limit) {
    float g = min(gate, limit);
    float u = clamp(up, -limit, limit);
    return (g / (1.0f + exp(-g))) * u;
}
