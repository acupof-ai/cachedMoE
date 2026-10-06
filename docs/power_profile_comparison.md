# Power-profile comparison

**Select balanced for future web/benchmark defaults.** The complete v4 run has
the same raw winner for all eight turns, the first two and the last two. It
shuts down normally with disk-KV drain and no source drop or IO/load failures.
All 4,096 output IDs match across the three modes; no turn loops. Acceptance
is 85.8902% and decode hit rate 91.4210% in every arm.

| Mode | All raw ms/token | First two | Last two | Decode pauses | Decode cooling, s | Mean decode GPU MHz | GPU peak, C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| power-saver | 107.830 | 88.891 | 118.063 | 0 | 0 | 945 | 72 |
| balanced | **80.944** | **73.688** | **81.556** | 0 | 0 | 1,599 | 80 |
| performance | 113.801 | 105.762 | 116.768 | 1,567 | 176.603 | 2,516 | 89 |

Balanced reduces raw decode cost by 28.9% against performance (1.406x throughput)
and 24.9% against power-saver (1.332x). Performance's active estimate is 70.601
ms/token, but cooling consumes 38.0% of decode wall time. The 85 C trigger is a
pause threshold, not a temperature cap; GPU work in flight can overshoot it.
The complete arm, including prefill/gaps, has 1,586 pauses / 179.175 s cooling.
External NVMe peaks at 74.85 C in all arms; internal peaks are
57.85 / 59.85 / 57.85 C. All cold starts meet GPU 60 / NVMe 65 C.

Repetition is identical across modes: maximum same-token run 1, no short-period
loop, per-turn repeated-4gram fraction .01375–.18271 and distinct-2 .62818–.89237.
The primary denominator excludes each turn's initial prefill output: 4,088 timed
steps for 4,096 outputs. Request/total-engine timing, individual repetition metrics,
per-cycle draft/verify/commit/CPU costs and telemetry are in
[the complete result](power_profile_result.json), with full raw data under
`bench/results/mask_quality/power_profiles_v4/` and its prepared receipt directory.
Future automatic power defaults have one authority,
`runtime_defaults.DEFAULT_POWER_PROFILE`; explicit profiles and historical results
keep their meaning. Default-policy CPU gates include a balanced-policy mock.
Web/API acceptance will be recorded after the ordered draft-head GPU work.

The owner authorized a fresh comparison after the previous power-saver and
performance numbers used different workloads. The earlier attempts follow.
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
The new `power_profiles_v2/` comparison ran from `bfb7fa0` after a normal
web shutdown and disk-KV drain. Its 48 primary/mirror shard headers and sizes
match, the mirror is RO, and all six original web files have unchanged hashes
and mtimes. Related CPU gates pass 73/73; the main tools gate passes 52/52.
No power winner has been selected. The ordered FP8/vocabulary-subset stage
has not begun.

## v2 interrupted by USB4 link loss

At 2026-10-06 19:41:01 +08:00 the kernel reports PCIe `Link Down`, followed by
the Ugreen USB4 device/retimer disconnect. The engine drops mirror source 1
after three IO errors. The fourth turn had only begun prefill. Three turns
completed 512 outputs each (raw decode 84.202/91.648/105.677 ms per timed token),
but no complete arm exists. These partial values cannot select a power default.
The last sensor sample is GPU 67°C, internal NVMe 49.85°C, external NVMe 74.85°C;
no thermal pause triggered. The observed link loss is proved; whether enclosure
heat, the cable or another transport fault caused it remains unproven.

The original v2 harness stops the removed-sensor engine, terminates the HTTP
reader, then forces cleanup of the remaining owned child. Its shutdown receipt
is explicitly **not graceful**, so private benchmark KV drain is unconfirmed.
The user web had already drained normally before this run; its six files
remain unchanged. The sensor error could be hidden behind an incomplete-output
error and then a cleanup error. The corrected harness keeps primary/controller/
cleanup errors separately, restores the previous power profile even on failure,
and tests owned emergency cleanup using a CPU-only parent/child. Related CPU
checks pass 77/77 plus the additional owned-process case (five failure cases).

The previously authorized PCIe-port rescan restores NVMe enumeration and the
system's existing read-only mount. APST and other storage-stability settings
stay unchanged. Physical enclosure cooling/USB4 connection checking is requested
before a new complete comparison. Future measured attempts use a fresh directory;
the v1 cold-start block and v2 invalid hardware-interrupted samples are retained.
The web is restored and API/state/log agree on the new per-device thresholds,
performance, dynamic 5,500 slots, dual 48/48 sources, k2/top4/ONECB/CPU routing,
disk KV and 1M capacity. All six original files retain SHA256/mtime, without a
browser refresh. Standard main CPU gates now pass 54/54, including the nine
report/failure cases. [Web acceptance summary](power_profile_web_receipt.json).
No speed default or draft-head change is selected; both owner tasks remain open.

## v3 resumed on the other USB4 port

The box initially cycled between USB4 and USB/UAS on domain1, then NVMe
reappeared at PCIe 63:00.0 after a port rescan. The mirror is RO and all 48
headers/sizes match. Six user files still retain hashes/mtimes; the previous
web guard has stopped after losing its sensor, with drain unconfirmed.
Fresh v3 measurement starts from `13c0d98`, with the same original workload
and per-device thresholds, after verifying idle GPU ownership. Both NVMe
Composite sensors are now mandatory at dual-drive startup. A source-dropped
engine log also invalidates an arm independently of temperature availability.
Related CPU cases pass 80 and the standard main gate passes 54/54.
The 61 frozen inputs and actual HTTP configuration agree.

At 20:51:50 the domain1 device disconnects and at 20:51:58 it appears on
domain0. The owner confirms a manual port change during this measurement.
Power-saver and balanced each finish eight turns; performance saves four
complete turns before the interrupted fifth. The same-engine comparison is
incomplete and selects no default. Power-saver / balanced raw decode costs
are 112.998 / 80.612 ms per timed token; performance's four-turn 111.164 is
partial and cannot be compared as an eight-turn result. Its partial cooling
is 85.255 seconds, with 69.454 ms/token active estimate. Raw samples remain
unchanged; `power_profiles_v3/partial_receipt.json` labels the scope.

The owner authorizes a fresh v4 comparison on the now stable original port.
Fresh mirror headers/sizes match 48/48; all six user files remain unchanged.
The v4 run freezes 61 inputs from `334eef9`; actual API configuration agrees.
No benchmark runs alongside the user web. V4 subsequently completes; the
decision and measurements are at the top of this report.
