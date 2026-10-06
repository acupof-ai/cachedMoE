# Per-turn decode modes

The web Decode selector sets the policy for each submitted turn. The selection
is captured when the request enters the queue; changing the selector later
does not change an already queued request.

| Request `decode_mode` | Routing for new decode/feed steps | Decode |
|---|---|---|
| `mask-spec` | Compute resident experts; mask misses while dynamic LRU loads them asynchronously. | DSpark speculation on one main path. |
| `mask-plain` | The same masked routing and asynchronous dynamic LRU. | Ordinary decoding, zero speculative cycles. |
| `off-plain` | Exact expert routing, waiting for required misses. | Ordinary decoding, zero speculative cycles. |

Explicit selection requires a GPU engine with one dynamic-cache stream, no
fixed cache and no weighted mask waits. `mask-spec` also requires DSpark enabled
at startup. Unsupported selections are rejected before generation. The
engine advertises `ready.decode_modes.available` and its startup `default`;
the HTTP configuration exposes those capabilities. Callers that omit
`decode_mode` inherit their existing startup behavior unchanged.

With DSpark enabled at startup, its resources remain loaded in every mode.
Its **384 pinned expert slots are included in the 5500 total**, leaving 5116
target-cache slots. Plain decoding does not unload the draft model. Draft length
and GPU-routing defaults remain pending the formal D decision.

The existing GPU prefill path remains exact; decode-style prompt feeding and
replay use the selected routing policy. Selection affects only newly computed
prompt or output tokens. Existing KV
from masked computation is reused, including after a named-session restore;
selecting `off-plain` does not recompute that history. Use **New chat** when the
whole conversation must have an exact history.

The actual completion reports `done.decode_mode` and
`done.speculation_enabled`. Saved message history preserves those engine fields
under `stat`; it does not substitute the requested mode for a missing result.
The temporary policy is restored on normal completion, error and cancellation.
Speculation validates only the main path, with one target forward per cycle.

Validation passed **7 CPU cases and 1 GPU fixture**: exact feed tokens, margins
and KV window bytes matched bit for bit; 129 plain steps followed by speculation,
named-session restoration, zero plain cycles, one target per speculative cycle,
and error/cancel restoration passed. In the 512-slot pressure fixture, the exact
boundary waited for admission capacity: evictable slots **3 → 6**, Filling
**125 → 122**, pinned **384**, guarded **0**, completed timeline **6360** unchanged.
Remaining fills continued asynchronously. This fixture establishes functional
behavior; production throughput and zero reserve failures require separate gates.

Local evidence lives in the main checkout's ignored result directory:
[validation](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/final_web_policy/recovered_boundary/validation_receipt.json),
[binary and shaders](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/final_web_policy/recovered_boundary/binary_receipt.json),
[GPU result](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/final_web_policy/recovered_boundary/check_results.json),
[CPU results](/home/chenkailun/projects/cachedMoE/bench/results/mask_quality/final_web_policy/cpu_boundary_fix.log).
These receipts cover implementation validation; web service deployment is tracked separately.
