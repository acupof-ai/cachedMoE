Track D5 -- second read source on a USB4 enclosure, 2026-09-20.
See docs/p4_dual_source.md section 9.

abab/           FIRST ABAB, ABANDONED after 2 cells. The default 1 s weight probe
                read E: at 0.03 GB/s (it is 3.77), the router gave the mirror 0.0%
                of the bytes, and `on` was +1.4% -- noise. Kept because the
                status.json of y_turns_on_0 is the evidence: 775 req / 0.1 GiB,
                mean lat 1315 ms.
probe_w/        One cell with DEEPMOE_MIRROR_WEIGHTS=4.72;3.66 forced, which is
                what proved the drive itself was fine: 5.1046 tok/s, stall 97.1,
                split 71.4 : 28.5, zero errors.
abab2/          THE ABAB THE VERDICT USES. 12 cells, probe fixed, NO weight
                override. y_turns +12.50% nominal / +9.40% against the two
                uncontaminated off cells; long_turns +4.63%. Zero E: errors.
                abab2_harness.txt is the harness stdout.
ms/             Two-stream A/B (bench/d5_ms_mirror_abab.py): mirror off 5.4507 ->
                on 6.1867 tok/s = +13.50%, 2 pairs, sched pipeline.
gates/          suite.decode + suite.integration (ctest_gates.txt), l3_ppl with
                the mirror ON (l3/off.txt shows `2 read sources`), NLL 0.630051,
                and the full CPU ctest (ctest_cpu.txt, 46/46).

CAVEAT recorded in the doc: y_turns_off_1 (4.3911 vs 4.8027 / 4.7703) was
polluted by a cmake build I ran on the same machine during that cell. The
verdict uses the clean cells.

NOT kept: route.bin and profile.jsonl (the raw routing dump and the per-step
design 13.1 record) -- 16 of the 21 MB, and nothing in section 9 is derived from
them. events.jsonl (per-token step_ms + the `done` events bench/d2_abab.py's
cell_stats reads) and status.json (the io section with the src[] rows) are what
the table in section 9.4 is computed from, and both are here.

NOTE: *.log is .gitignore'd repo-wide, so the per-cell serve.log files are not
here. serve_log_sources.txt carries the lines that matter out of every one of
them -- the probe values, the `2 read sources` line, the health probe, and
(nowhere, which is the point) any win32 / dropped-source line.
