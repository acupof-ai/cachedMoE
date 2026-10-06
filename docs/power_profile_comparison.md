# Power-profile comparison

The owner authorized a fresh comparison after the previous power-saver and
performance numbers used different workloads. No new speed conclusion exists.
The **v1** first arm's cold-start gate was blocked after 900.289 seconds: the external
NVMe's minimum was 60.85°C, above the required 60°C. No measured arms or generated
tokens ran. The temporary engine shut down normally and drained KV; the user
web resumed its original policy without a browser refresh or transcript edits.

The committed driver is `bench/power_profile_compare.py`. It uses the web
launch policy and isolated port/state, preserving dynamic 5,500 slots, dual RO
sources, k2/top4, ONECB, CPU routing, 1M capacity and 4GB disk KV. One engine and
one named session run power-saver, balanced, then performance, once per arm.
KV resets per arm; expert cache carries in that fixed order. All eight prompts
request exactly 512 outputs; an early stop makes the arm incomplete. Before
each arm the current owner-approved policy requires GPU <=60°C and NVMe
Composite <=65°C. During work GPU pauses/resumes at 85/77°C and NVMe at
80/72°C. Only sensors that triggered hold the verified engine child paused.
Six parameters share `runtime_defaults.ThermalPolicy`; command, manifest,
telemetry, result and web API record the actual selected values. The v1 receipt
retains its original uniform 80/72°C policy and <=60°C cold gate.

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
was made. The owner subsequently replaced the NVMe cold gate with <=65°C in TODO §0.7.
The new `power_profiles_v2/` comparison is running from `bfb7fa0` after a normal
web shutdown and disk-KV drain. Its 48 primary/mirror shard headers and sizes
match, the mirror is RO, and all six original web files have unchanged hashes
and mtimes. Related CPU gates pass 73/73; the main tools gate passes 52/52.
No power winner has been selected. The ordered FP8/vocabulary-subset stage
has not begun; the web will resume with the new per-device guard.
