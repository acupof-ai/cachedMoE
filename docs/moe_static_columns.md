# Constant live-column MoE probe — 2026-10-07

Neither candidate improves GPU time. Production keeps the original kernels,
cache policy and k2/top4 speculation. No new environment switch ships.

| Configuration | GPU ms/layer | Change from this baseline |
|---|---:|---:|
| Six-column storage, dynamic three-column loop | 1.986774897 | reference |
| Six-column storage, constant three-column loop | 2.029320735 | +2.14% |
| Three-column storage, constant three-column loop | 1.987997219 | +0.06% |

These small differences do not establish a regression beyond the jitter band;
they establish no measured saving. Both fail the halved 2 ms/cycle entry gate,
so no full-cycle speed run or new MMLU score is claimed. The earlier array-width
probe is a separate measurement and is not pooled into this baseline.

The constant loop uses existing shader specialization `StaticM=3`. The first
candidate keeps six-column buffer strides, FP8 activation plane offsets and
reduction geometry. The second combines three-column storage with the constant
loop. M1 retains its existing twin; other live counts fall back. Timing uses
12 resident routed FP4 experts per layer, a three-layer stream, one warmup and
one eight-iteration measurement per configuration. The compact candidate uses
the existing baseline without timing that configuration again. Balanced/AC,
original temperature thresholds, zero thermal pauses; the two-drive mirror
passes 48/48 size and header checks. This micro times primary-resident expert
computation, not expert IO or shared FP8 work.

Initial outputs match bit for bit for all supported live widths. A separate
untimed final check varies each column's activation and routing weights and
checks each of the three fixture layers. The retained `MoeSpec::static_m3`
option is used by explicit GPU probes only, defaults false, has no runtime
configuration binding and rejects GPU routing. Operational configuration stays
unchanged. Raw timed logs survive; the original timed binaries were replaced
by later builds, so their binary hashes are unavailable. Final validation
artifacts are retained separately; they are not relabelled as timed binaries.

The next open substantial change is bounding prefill activation workspaces.
It must preserve absolute KV positions, CED/compressor carry and Engram history;
feeding independent chunks would not meet that requirement. This track does
not implement or validate it.

[Machine receipt](moe_static_columns_receipt.json).

Actual main delivery passes CPU31/31, tools57/57 and MoE14/14 with no skips.
The untimed varying-column check passes in both storage widths across all three
fixture layers. Off NLL stays .622784; both GPU validation jobs have zero thermal
pauses, and 53 shaders are unchanged. No new short-decode or MMLU result is
claimed. The implementation is pushed with its remote SHA verified. Balanced
dynamic-mask web is live with k2/top4, native BF16, 5500 slots, two sources and
the original thermal policy. Six user files retain SHA/size/mtime; the browser
was not refreshed. Final binaries/shaders are archived, and the owned worktree
and branch are removed. [Delivery receipt](moe_static_columns_delivery_receipt.json).
