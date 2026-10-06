# Power-profile comparison

The owner authorized a fresh comparison after the previous power-saver and
performance numbers used different workloads. No new speed conclusion exists.
The first arm's cold-start gate was blocked after 900.289 seconds: the external
NVMe's minimum was 60.85°C, above the required 60°C. No measured arms or generated
tokens ran. The temporary engine shut down normally and drained KV; the user
web resumed its original policy without a browser refresh or transcript edits.

The committed driver is `bench/power_profile_compare.py`. It uses the web
launch policy and isolated port/state, preserving dynamic 5,500 slots, dual RO
sources, k2/top4, ONECB, CPU routing, 1M capacity and 4GB disk KV. One engine and
one named session run power-saver, balanced, then performance, once per arm.
KV resets per arm; expert cache carries in that fixed order. All eight prompts
request exactly 512 outputs; an early stop makes the arm incomplete. Before
each arm every GPU/NVMe composite sensor must be <=60°C. During work the existing
80/72°C per-sensor latched guard pauses only the bound engine child.

The reducer reports engine decode wall ms per timed decode step, including
thermal pauses, plus request wall and total-engine cost per output. First two
and last two turns are reported separately. Active time subtracts saved CPU
suspension overlaps only and never selects a default. Telemetry includes peaks,
pause entries, cooling and wall-time-weighted reported GPU clock. Acceptance,
hit rate, stage costs and all four repetition metrics accompany each arm.
Within 3% of the raw fastest, lower-power policy wins; different short/sustained
raw winners require an owner decision, with no automatic length-based switch.
Four CPU report gates cover ties, invalid/incomplete costs, thermal intersections
and the owner-decision boundary.

Local raw roots: `bench/results/mask_quality/power_profiles_prepared/` and
`power_profiles_v1/`. Relevant receipts are `launch_receipt.json`,
`source_freeze_receipt.json`, `cold_start_blocked_receipt.json`,
`idle_status_after_startup.json`, `idle_power_readonly_observations.json` and
`web_restore_original_policy_acceptance.json`. The launched scripts are frozen
from `d73bb7f`; `5387be7` corrects report-only accounting. Raw data will be
reprocessed without another GPU run if needed, retaining original reports.

The cold-start samples record APST disabled and the external drive active.
This might affect idle heat, but no causal test or storage power-policy change
was made. The requested physical cooling remains necessary to meet the start
condition. The power default and draft head stay unchanged. The ordered
FP8/vocabulary-subset stage has not begun.
