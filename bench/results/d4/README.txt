Track D4 -- the third attempt at an end-to-end mirror A/B (docs/p4_dual_source.md §8).

abab_harness.log      bench/d2_abab.py stdout: the one off cell that ran, then the
                      on cell's failure. 12 cells planned, 2 run.
on_cell_serve.log     the on cell's engine stderr. Read it top to bottom: both
                      startup probes measured E: alive (1.03 GB/s), the NEW mirror
                      health gate passed ("8 pinned-sized reads, all served"), and
                      then the drive left the bus mid-pinned-load -- the runtime
                      source-drop fired with win32 433 = ERROR_NO_SUCH_DEVICE, and
                      PinnedStore::load still returned the first failed request as
                      a fatal error (win32 1117 on 'norm.weight').
winevent_d4.txt       System log, 40 minutes around the failure: 39 x disk 153
                      (retry) + 7 x UASPStor 129 (bus reset), all starting 19:51:30,
                      the second the on cell's engine came up.
abab/                 the per-cell output directories. y_turns_off_0 is complete
                      (4.5607 tok/s, nvme_stall 113.0 ms, hit 0.8957, sources empty);
                      y_turns_on_0 holds only the failed launch.

The DX §5.1 acceptance gate that passed BEFORE all this (dx_probe, 48 handles x
QD 24, 2 x 300 s at 1.04 GB/s, zero errors, handles in 158 ms) is recorded in
docs/p4_e_drive_diag.md §5.1.1. Passing it turned out not to be sufficient.
