// head -- LM head, bf16 [129280 x 5120], 1.32 GB (design §7.11).
//
// The single largest resident read of a decode token, about 10% of the
// per-token budget of design §2.3, and therefore where speculative decoding
// pays best: at M=5 the same 1.32 GB yields logits for five positions.
//
// Placeholder: bindings and entry point only.
// TODO(design §7.11): body, plus GPU-side Gumbel-max / argmax sampling driven
// by a Philox counter so a run is reproducible. P2 and P4 per design §15.
[[vk::binding(0, 0)]] StructuredBuffer<uint>    W;      // bf16 [129280, 5120], 2 per uint
[[vk::binding(1, 0)]] StructuredBuffer<half>    X;      // normed hidden, M positions
[[vk::binding(2, 0)]] RWStructuredBuffer<float> Logits; // [M][129280] fp32

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {}
