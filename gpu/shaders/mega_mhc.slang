// mega_mhc -- fused hc_post + hc_mixes + Sinkhorn + RMSNorm (design §7.2, §7.7).
//
// Dispatch #1 and #8 of design §7.14, ~2 MB per invocation, dominated by
// hc_fn [24, 20480] fp32 (1.97 MB). 24 waves each take one 20480-wide dot
// product; the 20-iteration Sinkhorn that turns the 24 outputs into
// pre[4] / post[4] / comb[4][4] is doubly stochastic and stays in registers.
// The residual stream is fp32 throughout (design §2.4).
//
// Placeholder: bindings and entry point only, so the dispatch list of design
// §7.14 exists end to end (CMake runs slangc and spirv-val on it) before the
// body is written.
// TODO(design §7.2, §7.7): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<float>   X;        // residual [hc=4][5120] fp32
[[vk::binding(1, 0)]] StructuredBuffer<float>   A;        // sublayer output a[5120]
[[vk::binding(2, 0)]] StructuredBuffer<float>   HcFn;     // [24, 20480] fp32
[[vk::binding(3, 0)]] StructuredBuffer<float>   HcConst;  // hc_base[24], hc_scale[3], norm weights
[[vk::binding(4, 0)]] RWStructuredBuffer<float> Xout;     // updated residual
[[vk::binding(5, 0)]] RWStructuredBuffer<half>  U;        // this sublayer input u[5120] fp16
[[vk::binding(6, 0)]] RWStructuredBuffer<float> Mix;      // pre[4], post[4], comb[4][4]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
