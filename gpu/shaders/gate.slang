// gate -- router: sqrt(softplus(x . W)) then noaux_tc top-6 (design §7.8).
//
// Dispatch #9 of design §7.14. W is bf16 [384 x 5120], 3.9 MB. One workgroup:
// 6 waves x 64 rows = 384 dot products, then six rounds of WaveActiveMax with
// masking for the top-6 of (score + bias). Selection uses the biased score but
// the weights use the unbiased one, normalised and multiplied by 1.5
// (design §2.4); cpu/gate.cpp is the oracle for this.
//
// Outputs go to host-coherent memory so the CPU planner can read ids[6] without
// a fence, plus top16 for the heat EWMA of design §9.3, plus an atomic
// layer_done counter the CPU spins on.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.8): body. P2 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint>    W;         // bf16 [384, 5120], 2 per uint
[[vk::binding(1, 0)]] StructuredBuffer<float>   Bias;      // [384] fp32 (bias_vl for image tokens)
[[vk::binding(2, 0)]] StructuredBuffer<float>   Xf;        // ffn_norm input, fp32
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>  Ids;       // top-6 then top-16 ids
[[vk::binding(4, 0)]] RWStructuredBuffer<float> Weights;   // top-6 weights, top-16 scores
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>  LayerDone; // atomic counter, host coherent

[shader("compute")]
[numthreads(384, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
