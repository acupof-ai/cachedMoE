// Batch reuse of the decode cooperative-matrix arithmetic. Each query has
// its own causal list, overflow KV, RoPE and scratch plane.
#define DEEPMOE_CM_BATCH 1
#include "decode_attn_cm.slang"
